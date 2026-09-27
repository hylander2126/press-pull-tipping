"""The released estimator + data must reproduce the paper's Table 2."""
import numpy as np
import pytest

from press_pull_estimator.estimator.wrench_estimator import (
    coulomb_product_from_push, estimate_press_pull, relative_pivots, resolve_friction,
)
from press_pull_estimator.io import load_push, load_trial, trial_paths
from press_pull_estimator.objects import OBJECTS

# Table 2: (m mean, m sd, z_c mean [cm], z_c sd [cm], mu mean)
TABLE_2 = {
    "box":        (0.707, 0.012, 15.10, 0.28, 0.149),
    "heart":      (0.242, 0.006, 11.51, 0.27, 0.253),
    "flashlight": (0.396, 0.004,  9.38, 0.12, 0.234),
    "monitor":    (5.274, 0.008, 24.42, 0.05, 0.614),
}


@pytest.mark.parametrize("obj", list(TABLE_2))
def test_table_2(obj):
    gt = OBJECTS[obj]
    trials = [load_trial(p) for p in trial_paths(obj)]
    assert len(trials) == 10
    res = [estimate_press_pull(t, gt["com_x"], p) for t, p in zip(trials, relative_pivots(trials))]
    m = np.array([r.mass for r in res])
    zc = np.array([r.com_z for r in res]) * 100
    mu = np.array([resolve_friction(coulomb_product_from_push(*load_push(obj)), r.mass) for r in res])
    m_mean, m_sd, zc_mean, zc_sd, mu_mean = TABLE_2[obj]
    assert m.mean() == pytest.approx(m_mean, abs=6e-4)
    assert m.std(ddof=1) == pytest.approx(m_sd, abs=6e-4)
    assert zc.mean() == pytest.approx(zc_mean, abs=6e-3)
    assert zc.std(ddof=1) == pytest.approx(zc_sd, abs=6e-3)
    assert mu.mean() == pytest.approx(mu_mean, abs=6e-4)


def test_adjoint_round_trip():
    """A pure force at the object origin, seen from an offset sensor, maps back unchanged."""
    from scipy.spatial.transform import Rotation
    from press_pull_estimator.estimator.wrench_estimator import applied_wrench_in_object, make_T
    R = Rotation.from_euler("y", 20, degrees=True).as_matrix()[None]
    T_B_O = make_T(np.zeros((1, 3)), np.eye(3)[None])
    T_B_S = make_T(np.array([[0.1, 0.0, 0.3]]), R)
    f_B = np.array([1.0, 0.0, -2.0])                   # force the object exerts on the sensor
    # Sensor reading: that force, expressed in {S}, plus its moment about the sensor origin.
    f_S = R[0].T @ f_B
    tau_S = R[0].T @ np.cross(-T_B_S[0, :3, 3], f_B)
    w = applied_wrench_in_object(np.r_[tau_S, f_S][None], T_B_S, T_B_O)[0]
    np.testing.assert_allclose(w[3:], -f_B, atol=1e-12)   # Newton's third law
    np.testing.assert_allclose(w[:3], 0.0, atol=1e-12)    # the line of action passes through {O}
