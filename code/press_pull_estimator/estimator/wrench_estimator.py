"""Press-and-pull estimation of mass, CoM height, and support friction.

Standalone port of the estimator used for the ISRR 2026 paper
(irb120_control/estimation/{estimate_params,com_estimation}.py), with the ROS,
plotting, and ablation code removed. Conventions follow Modern Robotics: wrenches
are [tau, f] (moment first), twists are [omega, v], and Ad_T is the 6x6 adjoint.

Frames:  {B} robot base / table,  {S} F/T sensor,  {O} object frame at the pivot edge.

Pipeline for one press-and-pull trial:
    1. The fingertip presses down (SQUASH), holds (LULL), then sweeps a circular arc
       about the pivot edge (ARC) and back (UNARC). The normal force pins the pivot.
    2. The object tilt comes from proprioception alone: under no-slip, the ball
       centre is rigidly carried by the object, so its rotation about the pivot IS
       the object rotation (rotvec_between).
    3. The measured sensor wrench is mapped to the applied wrench on the object:
           {}^O w_app = -Ad_{T_SO}^T {}^S w_meas
    4. Pivot torque balance tau_app(theta) + tau_grav(theta; m, z_c) = 0 is solved
       for (m, z_c) by non-linear least squares, separately for ARC and UNARC.
    5. ARC and UNARC estimates are averaged. Fingertip friction and controller lag
       bias the two sweeps in opposite directions, so the average cancels them
       (push/retract hysteresis cancellation).
    6. Friction: Mode 1 (a low sliding push) gives the Coulomb product
       mu_t * m = f_slip / g. It is cached, and once Mode 2 returns m it resolves
       mu_t = f_slip / (m g).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.spatial.transform import Rotation

G = 9.81

# Controller state ids written by the robot's arc controller.
STATE_SQUASH, STATE_LULL, STATE_ARC, STATE_UNARC, STATE_RETRACT = 1, 2, 3, 4, 5

# Fingertip ball centre -> distal sensor face (where NetFT reports moments), along the
# tool's local +x. The URDF chain is tool0 -> root_finger (0.08225) -> ball (0.0866512),
# and the sensor moments are about root_finger.
BALL_TO_WRENCH_ORIGIN_X = 0.0866512

# Default pivot edge in {B}. It must match the controller's ARC_CENTER.
PIVOT_DEFAULT = np.array([0.61, 0.0, 0.0])

# Peak-to-peak variation of |p_ball - p_pivot| above which the rigid no-slip model (and
# so trajectory-based pivot recovery) is considered invalid.
NOSLIP_TOL_MM = 4.0


# ------------------------------------------------------------------------------------
#  SE(3) helpers (batched)
# ------------------------------------------------------------------------------------
def skew(v: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,3,3) skew-symmetric matrices."""
    v = np.atleast_2d(v)
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S


def make_T(p: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Batched homogeneous transforms from (N,3) positions and (N,3,3) rotations."""
    T = np.zeros((len(p), 4, 4))
    T[:, :3, :3], T[:, :3, 3], T[:, 3, 3] = R, p, 1.0
    return T


def trans_inv(T: np.ndarray) -> np.ndarray:
    R, p = T[:, :3, :3], T[:, :3, 3]
    Rt = R.transpose(0, 2, 1)
    return make_T(-np.einsum("nij,nj->ni", Rt, p), Rt)


def adjoint(T: np.ndarray) -> np.ndarray:
    """(N,4,4) -> (N,6,6) adjoint  [[R, 0], [p^ R, R]]  for [omega, v] twists."""
    R, p = T[:, :3, :3], T[:, :3, 3]
    Ad = np.zeros((len(T), 6, 6))
    Ad[:, :3, :3] = R
    Ad[:, 3:, 3:] = R
    Ad[:, 3:, :3] = skew(p) @ R
    return Ad


def rotvec_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation vectors taking unit(a) (3,) onto each unit(b) row (N,3)."""
    a = np.atleast_2d(a) / np.linalg.norm(a)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis, axis=-1)
    c = np.einsum("ni,ni->n", np.broadcast_to(a, b.shape), b)
    axis = axis / np.where(s[:, None] > 1e-9, s[:, None], 1.0)
    return axis * np.arctan2(s, c)[:, None]


# ------------------------------------------------------------------------------------
#  Wrench models
# ------------------------------------------------------------------------------------
def applied_wrench_in_object(w_meas_S: np.ndarray, T_B_S: np.ndarray, T_B_O: np.ndarray) -> np.ndarray:
    """{}^O w_app = -Ad_{T_SO}^T {}^S w_meas.

    The minus sign is Newton's third law: the sensor reads the reaction to what the
    finger applies to the object. All inputs are batched over N samples; wrenches
    are [tau, f].
    """
    T_S_O = trans_inv(T_B_S) @ T_B_O
    AdT = adjoint(T_S_O).transpose(0, 2, 1)
    return -np.einsum("nij,nj->ni", AdT, w_meas_S)


def gravity_wrench_in_object(rot_vecs_B: np.ndarray, p_com_O: np.ndarray, mass: float) -> np.ndarray:
    """Gravity wrench [tau, f] in {O} for an object tilted by rot_vecs_B about the pivot."""
    R_T = Rotation.from_rotvec(rot_vecs_B).as_matrix().transpose(0, 2, 1)
    f_O = R_T @ np.array([0.0, 0.0, -mass * G])
    tau_O = -np.cross(p_com_O, f_O)
    return np.hstack((tau_O, f_O))


# ------------------------------------------------------------------------------------
#  Filtering
# ------------------------------------------------------------------------------------
_BUTTER = butter(4, 6, fs=500, btype="low")


def lowpass(x: np.ndarray, method: str = "butter", window: int = 51, order: int = 3) -> np.ndarray:
    """Zero-phase smoothing along axis 0.

    "butter" is the 4th-order 6 Hz Butterworth (filtfilt) used in the paper.
    "savgol" is a Savitzky-Golay filter, which keeps peak shape better. It is useful
    when smoothing the arc wrench, which the paper leaves unfiltered.
    """
    x = np.asarray(x, float)
    if x.shape[0] <= max(20, window):
        return x
    if method == "savgol":
        return savgol_filter(x, window, order, axis=0)
    return filtfilt(*_BUTTER, x, axis=0)


# ------------------------------------------------------------------------------------
#  Data container
# ------------------------------------------------------------------------------------
@dataclass
class Trial:
    """One press-and-pull log on the pose time grid.

    t:        (N,)   s
    p_ball:   (N,3)  fingertip ball centre in {B}, m
    q_ball:   (N,4)  fingertip orientation in {B}, [x, y, z, w]
    wrench_S: (N,6)  measured wrench in {S}, [tau, f] (F/T interpolated onto t)
    state:    (N,)   controller state id
    """
    t: np.ndarray
    p_ball: np.ndarray
    q_ball: np.ndarray
    wrench_S: np.ndarray
    state: np.ndarray
    name: str = ""

    @classmethod
    def from_streams(cls, t_ft, ft, t_pose, pose, state, name=""):
        """Build from raw streams: ft (M,6) [fx fy fz tx ty tz], pose (N,7) [x y z qx qy qz qw]."""
        ft_on_pose = np.column_stack([np.interp(t_pose, t_ft, ft[:, i]) for i in range(6)])
        wrench = np.hstack((ft_on_pose[:, 3:], ft_on_pose[:, :3]))
        return cls(t_pose, pose[:, :3], pose[:, 3:], wrench, state.astype(int), name)

    def sensor_pose(self) -> np.ndarray:
        """Pose of the wrench origin in {B}. It has the ball's orientation and sits BALL_TO_WRENCH_ORIGIN_X behind it."""
        R = Rotation.from_quat(self.q_ball).as_matrix()
        p = self.p_ball - R @ np.array([BALL_TO_WRENCH_ORIGIN_X, 0.0, 0.0])
        return make_T(p, R)


# ------------------------------------------------------------------------------------
#  Pivot recovery from proprioception
# ------------------------------------------------------------------------------------
def noslip_deviation_mm(trial: Trial, pivot: np.ndarray = PIVOT_DEFAULT) -> float:
    """Peak-to-peak |p_ball - pivot| over the arc. Near-constant means the finger did not slip."""
    arc = np.isin(trial.state, [STATE_ARC, STATE_UNARC])
    if not arc.any():
        return np.nan
    return float(np.ptp(np.linalg.norm((trial.p_ball - pivot)[arc], axis=1)) * 1000)


def pivot_from_trajectory(trial: Trial, z_fixed: float = 0.0):
    """Fit the circle swept by the ball centre during ARC. Returns (pivot, radius, rms_resid).

    Under no-slip the ball centre is carried rigidly by the object, so it moves on a
    circle centred on the pivot. The swept arc is short (~15-20 deg), so z is pinned
    to the table plane and only x is fitted.
    """
    arc = trial.state == STATE_ARC
    if arc.sum() < 50:
        return None, np.nan, np.nan
    x, z = trial.p_ball[arc, 0], trial.p_ball[arc, 2]

    def resid(p):
        d = np.hypot(x - p[0], z - z_fixed)
        return d - d.mean()

    sol = least_squares(resid, x0=[float(np.median(x))])
    d = np.hypot(x - sol.x[0], z - z_fixed)
    return np.array([sol.x[0], 0.0, z_fixed]), float(d.mean()), float(np.std(d))


def relative_pivots(trials: list[Trial], default: np.ndarray = PIVOT_DEFAULT) -> list[np.ndarray]:
    """Per-trial pivots that remove object creep between back-to-back trials.

    Only each trial's deviation from the object's mean recovered pivot is applied, so
    the mean stays at `default`. This is a relative correction, not a fit to ground
    truth. If the object fails the no-slip check, the arc is not a circle and every
    trial gets `default`.
    """
    fits = [pivot_from_trajectory(tr)[0] for tr in trials]
    devs = [noslip_deviation_mm(tr, default) for tr in trials]
    xs = np.array([np.nan if f is None else f[0] for f in fits])
    if not np.isfinite(xs).any() or np.nanmean(devs) > NOSLIP_TOL_MM:
        return [default.copy() for _ in trials]
    mu = np.nanmean(xs)
    return [default + np.array([x - mu, 0, 0]) if np.isfinite(x) else default.copy() for x in xs]


# ------------------------------------------------------------------------------------
#  Mode 2: press-and-pull wrench balance
# ------------------------------------------------------------------------------------
@dataclass
class PhaseFit:
    mass: float
    com_z: float
    theta_star_deg: float
    rms_resid_nm: float


@dataclass
class PressPullResult:
    mass: float
    com_z: float
    theta_star_deg: float
    arc: PhaseFit
    unarc: PhaseFit
    pivot: np.ndarray
    noslip_dev_mm: float
    # Per-sample signals for plotting / inspection (contact window only).
    signals: dict = field(default_factory=dict, repr=False)


def object_tilt(trial: Trial, pivot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Object rotation vector in {B} per sample, from the ball's rotation about the pivot.

    Returns (rot_vec (N,3), in_contact (N,)). Samples outside contact are zero.
    """
    in_contact = np.isin(trial.state, [STATE_LULL, STATE_ARC, STATE_UNARC, STATE_RETRACT])
    r = trial.p_ball - pivot
    rv = rotvec_between(r[np.argmax(in_contact)], r)
    rv[~in_contact] = 0.0
    return rv, in_contact


def estimate_press_pull(trial: Trial, com_x: float, pivot: np.ndarray = PIVOT_DEFAULT,
                        smooth: str | None = None, x0: tuple[float, float] | None = None) -> PressPullResult:
    """Recover (m, z_c) from one press-and-pull trial.

    com_x is the horizontal CoM offset from the pivot edge. It is taken as known,
    which is the paper's assumption. tau(theta) only identifies the amplitude
    m*|r_com| and the phase theta*, and fixing com_x separates m from z_c.

    smooth: None (paper), "butter", or "savgol". Filters the measured wrench before
    the transform.
    """
    pivot = np.asarray(pivot, float)
    rv, in_contact = object_tilt(trial, pivot)
    # Tipping about +y makes rv_y negative. Skip the first degree (contact settling).
    mask = np.isin(trial.state, [STATE_ARC, STATE_UNARC]) & (rv[:, 1] < -np.deg2rad(1.0))

    w_S = lowpass(trial.wrench_S, smooth) if smooth else trial.wrench_S
    T_B_S = trial.sensor_pose()
    T_B_O = make_T(np.tile(pivot, (len(trial.t), 1)), Rotation.from_rotvec(rv).as_matrix())
    w_app = applied_wrench_in_object(w_S[mask], T_B_S[mask], T_B_O[mask])

    rv_c, st_c = rv[mask], trial.state[mask]
    tip_axis = rv_c.mean(0) / np.linalg.norm(rv_c.mean(0))
    tau_meas_all = w_app[:, :3] @ tip_axis

    # Data-driven starting point: max |tau| ~ m g |r| and z_c ~ 1.5 com_x. It only
    # needs to be in the right basin, since the problem has two parameters.
    if x0 is None:
        z0 = max(1.5 * abs(com_x), 0.05)
        x0 = (max(np.abs(tau_meas_all).max() / (G * np.hypot(com_x, z0)), 1e-3), z0)

    def fit(sel) -> PhaseFit:
        rv_ph, tau = rv_c[sel], tau_meas_all[sel]

        def resid(p):
            model = gravity_wrench_in_object(rv_ph, np.array([com_x, 0.0, p[1]]), p[0])[:, :3] @ tip_axis
            return model - tau

        sol = least_squares(resid, x0=x0, bounds=([1e-6, 1e-3], [np.inf, np.inf]), method="trf")
        m, zc = sol.x
        return PhaseFit(m, zc, float(np.degrees(np.arctan2(com_x, zc))),
                        float(np.sqrt(np.mean(sol.fun ** 2))))

    arc, unarc = fit(st_c == STATE_ARC), fit(st_c == STATE_UNARC)

    theta_deg = -np.rad2deg(rv_c[:, 1])
    signals = {
        "t": trial.t[mask] - trial.t[mask][0],
        "theta_deg": theta_deg,
        "phase": np.where(st_c == STATE_ARC, "arc", "unarc"),
        "tau_meas": tau_meas_all,
        "f_app_O": w_app[:, 3:],
        "tau_fit_arc": gravity_wrench_in_object(rv_c, np.array([com_x, 0, arc.com_z]), arc.mass)[:, :3] @ tip_axis,
        "tau_fit_unarc": gravity_wrench_in_object(rv_c, np.array([com_x, 0, unarc.com_z]), unarc.mass)[:, :3] @ tip_axis,
    }
    return PressPullResult(
        mass=0.5 * (arc.mass + unarc.mass),
        com_z=0.5 * (arc.com_z + unarc.com_z),
        theta_star_deg=0.5 * (arc.theta_star_deg + unarc.theta_star_deg),
        arc=arc, unarc=unarc, pivot=pivot,
        noslip_dev_mm=noslip_deviation_mm(trial, pivot),
        signals=signals,
    )


# ------------------------------------------------------------------------------------
#  Mode 1: sliding push -> Coulomb product, resolved once m is known
# ------------------------------------------------------------------------------------
def coulomb_product_from_push(fx: np.ndarray, fy: np.ndarray,
                              pre_frac: float = 0.10, slide_window=(0.50, 0.85)) -> float:
    """mu_t * m (kg) from a low sliding push.

    The first `pre_frac` of the log (before contact) gives the per-axis F/T zero bias,
    which is subtracted before taking the tangential magnitude. The steady-sliding
    force is the median over `slide_window` (fractions of the log).
    """
    fx, fy = lowpass(fx), lowpass(fy)
    n = len(fx)
    pre = slice(0, int(pre_frac * n))
    f_tan = np.hypot(fx - fx[pre].mean(), fy - fy[pre].mean())
    f_slip = float(np.median(f_tan[int(slide_window[0] * n):int(slide_window[1] * n)]))
    return f_slip / G


def resolve_friction(mu_m: float, mass: float) -> float:
    """mu_t = f_t / (m g) = (mu_t m) / m."""
    return mu_m / mass
