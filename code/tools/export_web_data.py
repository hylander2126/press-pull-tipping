#!/usr/bin/env python3
"""Generate the JSON the project page reads (../src/data/). Run it from code/.

    python tools/export_web_data.py

It writes results.json, with per-object ground truth, the estimates (mean +- SD over
trials), and the ARC-only / UNARC-only / averaged errors behind the hysteresis callout.
It also writes wrench_<object>.json, the wrench time series and torque-vs-tilt data for
one representative trial. Every number on the page comes from here, so rerunning it
after changing the estimator keeps the page consistent.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from press_pull_estimator.estimator.wrench_estimator import (  # noqa: E402
    STATE_ARC, STATE_LULL, STATE_RETRACT, STATE_SQUASH, STATE_UNARC,
    applied_wrench_in_object, coulomb_product_from_push, estimate_press_pull, gravity_wrench_in_object,
    make_T, object_tilt, relative_pivots, resolve_friction,
)
from press_pull_estimator.io import load_push, load_trial, trial_paths  # noqa: E402
from press_pull_estimator.objects import OBJECTS  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src", "data")
SHOWCASE_TRIAL = 2          # the trial whose video is on the page
PHASE_NAMES = {STATE_SQUASH: "press", STATE_LULL: "hold", STATE_ARC: "pull (ARC)",
               STATE_UNARC: "return (UNARC)", STATE_RETRACT: "retract"}


def r(x, n=4):
    return np.round(np.asarray(x, float), n).tolist()


def binned(x, y, n):
    """Average y into n equal-count bins (keeps the shape, drops sensor noise)."""
    idx = np.array_split(np.arange(len(x)), min(n, len(x)))
    return [x[i].mean() for i in idx], [y[i].mean() for i in idx]


def wrench_timeseries(trial, pivot):
    """Applied wrench in {O} over the whole interaction (press, hold, pull, return, retract)."""
    rv, _ = object_tilt(trial, pivot)
    sel = np.isin(trial.state, [STATE_SQUASH, STATE_LULL, STATE_ARC, STATE_UNARC, STATE_RETRACT])
    T_B_O = make_T(np.tile(pivot, (len(trial.t), 1)), Rotation.from_rotvec(rv).as_matrix())
    w = applied_wrench_in_object(trial.wrench_S[sel], trial.sensor_pose()[sel], T_B_O[sel])
    t = trial.t[sel] - trial.t[sel][0]
    st = trial.state[sel]
    # ~25 Hz bins within each contiguous phase run, so phase boundaries stay sharp.
    # (The hold state occurs twice: before the pull and at the reversal.)
    out = {"t": [], "Fx": [], "Fz": [], "tau": [], "phases": []}
    cuts = np.flatnonzero(np.diff(st)) + 1
    for seg in np.split(np.arange(len(st)), cuts):
        n = max(2, int((t[seg][-1] - t[seg][0]) * 25))
        # tau is the tipping torque about the pivot edge (-y), positive while the finger tips the object.
        for key, col, sign in (("Fx", 3, 1.0), ("Fz", 5, 1.0), ("tau", 1, -1.0)):
            tt, yy = binned(t[seg], sign * w[seg, col], n)
            out[key] += yy
        out["t"] += tt
        out["phases"].append({"name": PHASE_NAMES[int(st[seg][0])],
                              "t0": round(float(t[seg][0]), 3), "t1": round(float(t[seg][-1]), 3)})
    for k in ("t", "Fx", "Fz", "tau"):
        out[k] = r(out[k])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for obj, gt in OBJECTS.items():
        trials = [load_trial(p) for p in trial_paths(obj)]
        pivots = relative_pivots(trials)
        mu_m = coulomb_product_from_push(*load_push(obj))
        res = [estimate_press_pull(tr, gt["com_x"], pv) for tr, pv in zip(trials, pivots)]

        def stats(vals, truth):
            v = np.asarray(vals)
            return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "err_pct": float(abs(v.mean() - truth) / truth * 100)}

        mu_gt = resolve_friction(mu_m, gt["mass"])
        results[obj] = {
            "label": gt["label"], "n_trials": len(res),
            "gt": {"m": gt["mass"], "zc": gt["com_z"], "mu": mu_gt},
            "est": {"m": stats([x.mass for x in res], gt["mass"]),
                    "zc": stats([x.com_z for x in res], gt["com_z"]),
                    "mu": stats([resolve_friction(mu_m, x.mass) for x in res], mu_gt)},
            # Hysteresis: each sweep alone vs. the push/retract average.
            "hysteresis": {
                "m": {"arc": stats([x.arc.mass for x in res], gt["mass"]),
                      "unarc": stats([x.unarc.mass for x in res], gt["mass"]),
                      "avg": stats([x.mass for x in res], gt["mass"])},
                "zc": {"arc": stats([x.arc.com_z for x in res], gt["com_z"]),
                       "unarc": stats([x.unarc.com_z for x in res], gt["com_z"]),
                       "avg": stats([x.com_z for x in res], gt["com_z"])},
            },
            "noslip_dev_mm": float(np.mean([x.noslip_dev_mm for x in res])),
        }

        k = SHOWCASE_TRIAL - 1
        tr, pv, rs = trials[k], pivots[k], res[k]
        s = rs.signals
        arc = s["phase"] == "arc"
        th_grid = np.linspace(0, max(s["theta_deg"].max(), rs.theta_star_deg) + 2, 120)
        rv_grid = np.column_stack([np.zeros_like(th_grid), -np.deg2rad(th_grid), np.zeros_like(th_grid)])
        tip = np.array([0, -1.0, 0])      # tipping axis; tau is reported as the torque that tips the object

        def fit_curve(pf):
            return r(gravity_wrench_in_object(rv_grid, np.array([gt["com_x"], 0, pf.com_z]), pf.mass)[:, :3] @ tip)

        # Flip the sign if needed so the finger's tipping torque plots as positive before theta*.
        sgn = -1.0 if np.median(s["tau_meas"]) < 0 else 1.0
        a_th, a_tau = binned(s["theta_deg"][arc], sgn * s["tau_meas"][arc], 90)
        u_th, u_tau = binned(s["theta_deg"][~arc], sgn * s["tau_meas"][~arc], 90)
        plot = {
            "object": obj, "label": gt["label"], "trial": SHOWCASE_TRIAL,
            "timeseries": wrench_timeseries(tr, pv),
            "torque": {
                "arc": {"theta": r(a_th, 3), "tau": r(a_tau)},
                "unarc": {"theta": r(u_th, 3), "tau": r(u_tau)},
                "fit_theta": r(th_grid, 3),
                "fit_arc": r(sgn * np.array(fit_curve(rs.arc))),
                "fit_unarc": r(sgn * np.array(fit_curve(rs.unarc))),
                "theta_star": {"arc": rs.arc.theta_star_deg, "unarc": rs.unarc.theta_star_deg,
                               "avg": rs.theta_star_deg,
                               "gt": float(np.degrees(np.arctan2(gt["com_x"], gt["com_z"])))},
            },
            "estimate": {"m": rs.mass, "zc": rs.com_z, "m_arc": rs.arc.mass, "m_unarc": rs.unarc.mass,
                         "mu": resolve_friction(mu_m, rs.mass)},
        }
        with open(os.path.join(OUT, f"wrench_{obj}.json"), "w") as fh:
            json.dump(plot, fh, separators=(",", ":"))
        print(f"{obj}: m={results[obj]['est']['m']['mean']:.3f}  wrote wrench_{obj}.json")

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {OUT}/results.json")


if __name__ == "__main__":
    main()
