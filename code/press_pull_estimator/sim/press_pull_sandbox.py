#!/usr/bin/env python3
"""Minimal MuJoCo press-and-pull sandbox. It needs only `pip install mujoco`.

A floating spherical fingertip (standing in for the robot arm) interacts
with an object whose mass and CoM the estimator is never told:

    Mode 1  PUSH    Slide the object with a low horizontal push. This caches mu_t * m.
    Mode 2  SQUASH  Press down on the top face until the normal force reaches a target.
            LULL    Hold briefly.
            ARC     Sweep the ball centre on a circle about the pivot edge (the near
                    bottom edge), with a radial force loop keeping the press on. This
                    tips the object while the normal force pins the pivot.
            UNARC   Sweep back.
            RETRACT Lift off.

A virtual F/T sensor sums the contact wrench the object exerts on the fingertip and
reports it at a sensor origin behind the ball, like the real Net F/T. Those logs go
through the same estimator used on the robot data.

    python -m press_pull_estimator.sim.press_pull_sandbox
    python -m press_pull_estimator.sim.press_pull_sandbox --object slab --mass 0.8 --com-z 0.09
    python -m press_pull_estimator.sim.press_pull_sandbox --viewer           # watch it live
    python -m press_pull_estimator.sim.press_pull_sandbox --record sim.mp4   # offscreen render (needs imageio)
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

from press_pull_estimator.estimator.wrench_estimator import (
    BALL_TO_WRENCH_ORIGIN_X,
    STATE_ARC, STATE_LULL, STATE_RETRACT, STATE_SQUASH, STATE_UNARC,
    Trial,
    coulomb_product_from_push,
    estimate_press_pull,
    resolve_friction,
)

BALL_R = 0.012
PIVOT_X = 0.61          # near bottom edge of the object, in world x (matches the robot setup)


def build_xml(obj: str, mass: float, com_x: float, com_z: float, mu_table: float) -> tuple[str, float, float]:
    """The object sits with its near bottom edge on x = PIVOT_X. Its CoM is placed by
    `inertial` and is independent of the geometry, so the estimator cannot read it off the mesh."""
    # (half-extents, rgba). Both are prisms, so the pivot is a full edge. MuJoCo cylinders
    # pivot on a single rim point and tend to roll sideways, which a real press resists.
    hx, hy, hz = {"box": (0.05, 0.06, 0.15), "slab": (0.075, 0.04, 0.10)}[obj]
    rgba = {"box": ".55 .75 .95 .55", "slab": ".95 .7 .3 .6"}[obj]
    geom = (f'<geom name="obj" type="box" size="{hx} {hy} {hz}" pos="{hx} 0 {hz}" '
            f'friction="{mu_table} .005 .0001" rgba="{rgba}"/>')
    top_z, top_x = 2 * hz, 2 * hx
    # Inertia of a comparable solid box. Only the mass and CoM matter quasi-statically.
    i = mass * 0.01
    return f"""
<mujoco model="press_pull_sandbox">
  <!-- noslip iterations suppress the slow tangential creep of soft contacts, which would
       otherwise let the ball drift across the top face and corrupt the tilt estimate. -->
  <option timestep="0.001" cone="elliptic" impratio="10" noslip_iterations="10" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".86 .89 .93" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".92 .92 .92" rgb2=".82 .82 .82" width="512" height="512"/>
    <material name="table" texture="grid" texrepeat="8 8" reflectance=".05"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.6 1.2" dir="0.2 0.5 -1" diffuse=".9 .9 .9"/>
    <geom name="table" type="plane" size="1.2 1.2 .01" material="table" friction="{mu_table} .005 .0001"/>
    <body name="object" pos="{PIVOT_X} 0 0">
      <freejoint/>
      <inertial pos="{com_x} 0 {com_z}" mass="{mass}" diaginertia="{i} {i} {i}"/>
      {geom}
      <site name="com" pos="{com_x} 0 {com_z}" size=".007" rgba=".9 .15 .15 1"/>
    </body>
    <!-- Fingertip on three stiff position-controlled slide joints (a Cartesian impedance
         stand-in for the arm). It has to be a dynamic body, not a mocap body: MuJoCo friction
         acts on contact velocity, and a teleported mocap body has none. -->
    <body name="finger" gravcomp="1">
      <joint name="fx" type="slide" axis="1 0 0" damping="5"/>
      <joint name="fy" type="slide" axis="0 1 0" damping="5"/>
      <joint name="fz" type="slide" axis="0 0 1" damping="5"/>
      <joint name="wrist" type="hinge" axis="0 1 0" damping="0.5"/>
      <geom name="ball" type="sphere" size="{BALL_R}" mass="0.05" rgba=".2 .2 .25 1" friction="2.0 .01 .001" priority="1" solref=".004 1"/>
      <geom type="cylinder" size=".006 .06" pos="-.06 0 .02" euler="0 -70 0" rgba=".35 .35 .4 1" mass="0" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="fx" kp="4000" kv="60"/>
    <position joint="fy" kp="4000" kv="60"/>
    <position joint="fz" kp="4000" kv="60"/>
    <position joint="wrist" kp="300" kv="5"/>
  </actuator>
</mujoco>""", top_z, top_x


class Sandbox:
    def __init__(self, obj="box", mass=0.676, com_x=0.05, com_z=0.15, mu_table=0.25, render=None):
        xml, self.top_z, self.top_x = build_xml(obj, mass, com_x, com_z, mu_table)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.ball = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        self.obj_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "obj")
        self.render = render            # callable(model, data) invoked at ~30 Hz, or None
        self.log = {"t": [], "p": [], "ft": [], "state": []}
        self.set_target(np.array([PIVOT_X - 0.1, 0.0, self.top_z + 0.05]))
        self.d.qpos[-4:-1] = self.target
        mujoco.mj_forward(self.m, self.d)

    # The finger's joints are the last four qpos entries: x, y, z slides (its world position),
    # then the wrist pitch about y.
    def set_target(self, p, pitch=None):
        self.target = np.asarray(p, float).copy()
        self.d.ctrl[:3] = self.target
        if pitch is not None:
            self.d.ctrl[3] = pitch

    def ball_pos(self) -> np.ndarray:
        """Measured (not commanded) ball centre: the robot's proprioception."""
        return self.d.qpos[-4:-1].copy()

    def ball_quat(self) -> np.ndarray:
        """Measured finger orientation [x, y, z, w] (pitch about y)."""
        a = self.d.qpos[-1]
        return np.array([0.0, np.sin(a / 2), 0.0, np.cos(a / 2)])

    # -- virtual F/T ------------------------------------------------------------------
    def ball_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        """Force and moment the object exerts on the fingertip, in world axes, about the
        sensor origin (BALL_TO_WRENCH_ORIGIN_X behind the ball along the finger's x axis)."""
        f_tot, tau_tot = np.zeros(3), np.zeros(3)
        p_sensor = self.ball_pos() - self.R() @ np.array([BALL_TO_WRENCH_ORIGIN_X, 0, 0])
        c6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if self.ball not in (c.geom1, c.geom2):
                continue
            mujoco.mj_contactForce(self.m, self.d, i, c6)
            # c6[:3] is in the contact frame and is the force geom1 exerts on geom2.
            f = c.frame.reshape(3, 3).T @ c6[:3]
            if c.geom1 == self.ball:
                f = -f
            f_tot += f
            tau_tot += np.cross(c.pos - p_sensor, f)
        return f_tot, tau_tot

    def R(self) -> np.ndarray:
        a = self.d.qpos[-1]
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

    # -- stepping -----------------------------------------------------------------------
    def step(self, state: int, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            f, tau = self.ball_wrench()
            R = self.R()
            self.log["t"].append(self.d.time)
            self.log["p"].append(np.r_[self.ball_pos(), self.ball_quat()])
            self.log["ft"].append(np.r_[R.T @ f, R.T @ tau])     # expressed in the sensor frame
            self.log["state"].append(state)
            if self.render and len(self.log["t"]) % 33 == 0:
                self.render(self.m, self.d)

    def move_to(self, target, speed, state):
        p0 = self.target.copy()
        steps = max(1, int(np.linalg.norm(target - p0) / speed / self.m.opt.timestep))
        for k in range(1, steps + 1):
            self.set_target(p0 + (target - p0) * k / steps)
            self.step(state)

    def reset_log(self):
        self.log = {k: [] for k in self.log}

    def trial(self, name="sim") -> Trial:
        t = np.array(self.log["t"])
        ft = np.array(self.log["ft"])
        pose = np.array(self.log["p"])
        return Trial.from_streams(t, ft, t, pose, np.array(self.log["state"]), name)

    # -- Mode 1 -------------------------------------------------------------------------
    def push(self, height=0.02, dist=0.08, speed=0.02) -> tuple[np.ndarray, np.ndarray]:
        """Low horizontal push on the near face, toward +x. Returns (fx, fy) at the sensor.

        Only the forward stroke is logged. It starts 2 cm short of the face, so the first
        ~10% of the log is contact-free and gives the zero bias, as on the robot.
        The fingertip contact is made nearly frictionless for this stroke. Otherwise the
        grippy ball drags the face downward and loads the table contact beyond m g.
        """
        mu_ball = self.m.geom_friction[self.ball, 0]
        self.m.geom_friction[self.ball, 0] = 1e-3
        self.move_to(np.array([PIVOT_X - BALL_R - 0.02, 0, height]), 0.1, 0)
        self.step(0, 300)
        self.reset_log()
        self.move_to(self.target + [0.02 + dist, 0, 0], speed, 0)
        ft = np.array(self.log["ft"])
        self.move_to(self.target - [0.03, 0, 0], speed, 0)
        self.m.geom_friction[self.ball, 0] = mu_ball
        return ft[:, 0], ft[:, 1]

    # -- Mode 2 -------------------------------------------------------------------------
    def press_and_pull(self, f_press=8.0, max_arc_deg=30.0, arc_speed=0.005, ramp_s=2.0, stop_frac=0.1,
                       kp=4e-5, rotate_finger=True):
        body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "object")
        # Near bottom edge = the object's body origin, read from its current pose (Mode 1 moved it).
        pivot = np.array([self.d.xpos[body][0], 0.0, 0.0])
        base0 = self.d.xpos[body][:2].copy()
        self.base_slip_mm = 0.0
        press_x = pivot[0] + BALL_R + 0.008   # just inboard of the near top edge, nearly above the pivot (as on the robot)

        # Travel over the object, then SQUASH: descend until the normal force reaches f_press.
        clear = self.top_z + BALL_R + 0.03
        self.move_to(np.r_[self.target[:2], clear], 0.05, 0)
        self.move_to(np.array([press_x, 0, clear]), 0.05, 0)
        self.reset_log()
        self.move_to(np.array([press_x, 0, self.top_z + BALL_R + 0.005]), 0.02, STATE_SQUASH)
        while self.ball_wrench()[0][2] < f_press and self.d.time < 60:
            self.set_target(self.target - [0, 0, 0.004 * self.m.opt.timestep])
            self.step(STATE_SQUASH)
        self.step(STATE_LULL, 1000)

        # ARC / UNARC: tangential sweep about the pivot, with the radial press held by a P-loop.
        r_vec = self.target - pivot
        radius, phi0 = np.linalg.norm(r_vec[[0, 2]]), np.arctan2(r_vec[2], r_vec[0])
        dphi_max = arc_speed / radius * self.m.opt.timestep
        phi, flip_count, pull_peak = phi0, 0, 0.0
        for state, direction in ((STATE_ARC, +1), (STATE_UNARC, -1)):
            n = 0
            while True:
                # Ramp the tangential speed up (ARC_TANGENTIAL_RAMP_SEC on the robot). A step in
                # speed spikes the tangential load and makes the fingertip skid.
                n += 1
                dphi = dphi_max * min(1.0, n * self.m.opt.timestep / ramp_s)
                f, _ = self.ball_wrench()
                u = np.array([np.cos(phi), 0, np.sin(phi)])
                f_n = f @ u                            # the object pushes the ball outward along u
                # Radial force loop, rate-limited to 5 mm/s like the robot's MAX_NORMAL_SPEED.
                radius -= np.clip(kp * (f_press - f_n), -5e-6, 5e-6)
                phi += direction * dphi
                # By default the wrist pitches with the arc, so the ball turns with the object and the
                # contact can stay stuck. The paper keeps the finger world-fixed (Sec. 2.3) and relies
                # on the compliant fingertip's rolling contact patch. A rigid simulated sphere can't
                # model that, so a world-fixed finger here has to skid about R*theta (~4 mm), and the
                # tilt estimate comes out a few percent low. Use --world-fixed-finger to see the effect.
                pitch = -(phi - phi0) if rotate_finger else 0.0
                self.set_target(pivot + radius * np.array([np.cos(phi), 0, np.sin(phi)]), pitch=pitch)
                self.step(state)
                self.base_slip_mm = max(self.base_slip_mm, np.linalg.norm(self.d.xpos[body][:2] - base0) * 1000)
                swept = np.degrees(phi - phi0)
                if state == STATE_ARC:
                    # Before theta* the object drags the ball back along -t. The robot sweeps
                    # until that tangential (torque-carrying) force flips sign. Here we stop a
                    # little earlier, once it has decayed to stop_frac of its peak. Past theta*
                    # a light object leans on the finger, and the tilted press can then shove
                    # its base along the table. The fit recovers theta* from the curve shape.
                    pull = -f @ np.array([-np.sin(phi), 0, np.cos(phi)])
                    pull_peak = max(pull_peak, pull)
                    flip_count = flip_count + 1 if (swept > 3 and pull < stop_frac * pull_peak) else 0
                    if flip_count > 10 or swept > max_arc_deg:
                        break
                elif swept <= 0.5:
                    break
        self.set_target(self.target, pitch=0.0)
        self.move_to(self.target + [0, 0, 0.04], 0.01, STATE_RETRACT)
        return pivot


def make_viewer():
    """Render callback that opens the interactive viewer and paces the sim to real time."""
    import time

    from mujoco import viewer as mj_viewer
    state = {}

    def render(m, d):
        if "v" not in state:
            state["v"], state["t0"] = mj_viewer.launch_passive(m, d), time.time() - d.time
        state["v"].sync()
        time.sleep(max(0.0, d.time - (time.time() - state["t0"])))
    return render


def make_recorder(frames: list, width=1280, height=720):
    """Render callback that appends offscreen side-view frames to `frames`."""
    state = {}

    def render(m, d):
        if "r" not in state:
            state["r"] = mujoco.Renderer(m, height, width)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [PIVOT_X + 0.02, 0, 0.16]
            cam.distance, cam.azimuth, cam.elevation = 0.85, 90, -10
            state["cam"] = cam
        state["r"].update_scene(d, state["cam"])
        frames.append(state["r"].render())
    return render


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", choices=["box", "slab"], default="box")
    ap.add_argument("--mass", type=float, default=None, help="true mass, hidden from the estimator (kg)")
    ap.add_argument("--com-z", type=float, default=None, help="true CoM height, hidden from the estimator (m)")
    ap.add_argument("--mu-table", type=float, default=None, help="true support friction (default 0.25)")
    ap.add_argument("--press", type=float, default=8.0, help="Mode 2 normal press force (N)")
    ap.add_argument("--world-fixed-finger", action="store_true",
                    help="keep the finger orientation world-fixed during the arc (the paper's strategy)")
    ap.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    ap.add_argument("--record", default=None, help="write an MP4 of the run (requires imageio[ffmpeg])")
    args = ap.parse_args()

    # (mass, com_x, com_z, mu_table). The slab's CoM sits off-centre and high, as in a loaded
    # container, so theta* differs from what its geometry would suggest.
    defaults = {"box": (0.676, 0.05, 0.15, 0.25), "slab": (1.2, 0.06, 0.13, 0.25)}
    mass, com_x, com_z, mu_table = defaults[args.object]
    mass, com_z = args.mass or mass, args.com_z or com_z
    args.mu_table = args.mu_table or mu_table

    frames = []
    render = make_viewer() if args.viewer else make_recorder(frames) if args.record else None
    sb = Sandbox(args.object, mass, com_x, com_z, args.mu_table, render)
    sb.step(0, 500)                                  # settle

    print(f"[sim] {args.object}: true m={mass:.3f} kg, z_c={com_z * 100:.2f} cm, mu_t={args.mu_table:.3f}")
    fx, fy = sb.push()
    mu_m = coulomb_product_from_push(fx, fy)
    print(f"[Mode 1] Coulomb product  mu_t*m = {mu_m:.4f} kg  (cached)")

    sb.step(0, 1000)
    pivot = sb.press_and_pull(f_press=args.press, rotate_finger=not args.world_fixed_finger)
    if sb.base_slip_mm > 2.0:
        print(f"[warn] the object's base moved {sb.base_slip_mm:.1f} mm during the press-and-pull, so the pivot "
              f"was not pinned and the estimate is unreliable. The press has a horizontal component that grows "
              f"with tilt, Lower --press or raise --mu-table.")
    trial = sb.trial()
    r = estimate_press_pull(trial, com_x=com_x, pivot=pivot)
    mu = resolve_friction(mu_m, r.mass)
    err = lambda e, g: abs(e - g) / g * 100  # noqa: E731
    print(f"[Mode 2] arc m={r.arc.mass:.3f} z_c={r.arc.com_z * 100:.2f} cm | "
          f"unarc m={r.unarc.mass:.3f} z_c={r.unarc.com_z * 100:.2f} cm")
    print(f"[result] m   = {r.mass:.4f} kg   (true {mass:.4f}, err {err(r.mass, mass):.1f}%)")
    print(f"         z_c = {r.com_z * 100:.2f} cm   (true {com_z * 100:.2f}, err {err(r.com_z, com_z):.1f}%)")
    print(f"         mu_t= {mu:.4f}      (true {args.mu_table:.4f}, err {err(mu, args.mu_table):.1f}%)")
    print(f"         theta* = {r.theta_star_deg:.2f} deg, no-slip dev {r.noslip_dev_mm:.2f} mm")

    if args.record and frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.record, frames, fps=30, quality=8, macro_block_size=8)
        print(f"[sim] wrote {args.record} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
