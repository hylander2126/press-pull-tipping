#!/usr/bin/env python3
"""Offline press-and-pull evaluation on the recorded benchmark trials.

    python run_offline_eval.py --object flashlight
    python run_offline_eval.py --object all
    python run_offline_eval.py --object heart --trial 3 --plot

For each trial it recovers (m, z_c) from the Mode 2 press-and-pull wrench balance,
then resolves mu_t from the Mode 1 sliding push. It reports mean +- SD over trials
and the relative error against ground truth.
"""
import argparse

import numpy as np

from press_pull_estimator.estimator.wrench_estimator import (
    PIVOT_DEFAULT,
    coulomb_product_from_push,
    estimate_press_pull,
    relative_pivots,
    resolve_friction,
)
from press_pull_estimator.io import load_push, load_trial, trial_paths
from press_pull_estimator.objects import OBJECTS


def run_object(obj: str, trial: int | None, pivot_mode: str, smooth: str | None, verbose: bool):
    gt = OBJECTS[obj]
    trials = [load_trial(p) for p in trial_paths(obj)]
    if not trials:
        raise SystemExit(f"No trials found for '{obj}' in data/{obj}/")
    # Relative pivot correction needs every trial of the object, so compute it before selecting one.
    pivots = relative_pivots(trials) if pivot_mode == "relative" else [PIVOT_DEFAULT] * len(trials)
    idx = [trial - 1] if trial else range(len(trials))

    mu_m = coulomb_product_from_push(*load_push(obj))   # Mode 1, cached
    mu_gt = resolve_friction(mu_m, gt["mass"])

    rows, results = [], []
    for i in idx:
        r = estimate_press_pull(trials[i], gt["com_x"], pivots[i], smooth=smooth)
        mu = resolve_friction(mu_m, r.mass)
        rows.append((r.mass, r.com_z, mu))
        results.append(r)
        if verbose:
            print(f"  {trials[i].name}:  m={r.mass:.4f} kg  z_c={r.com_z * 100:6.2f} cm  mu_t={mu:.4f}   "
                  f"[arc m={r.arc.mass:.3f}, unarc m={r.unarc.mass:.3f}; no-slip dev {r.noslip_dev_mm:.1f} mm]")
    est = np.array(rows)
    return est, (gt["mass"], gt["com_z"], mu_gt), results


def fmt_row(label, est, truth):
    mean = est.mean(0)
    sd = est.std(0, ddof=1) if len(est) > 1 else np.zeros(3)
    err = np.abs(mean - truth) / np.abs(truth) * 100
    return (f"{label:<20s} {truth[0]:7.3f} {mean[0]:7.3f} ± {sd[0]:.3f} ({err[0]:4.1f}%)   "
            f"{truth[1] * 100:6.2f} {mean[1] * 100:6.2f} ± {sd[1] * 100:.2f} ({err[1]:4.1f}%)   "
            f"{truth[2]:.3f} {mean[2]:.3f} ± {sd[2]:.3f} ({err[2]:4.1f}%)")


def plot(obj, result):
    import matplotlib.pyplot as plt
    s = result.signals
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.5))
    a0.plot(s["t"], s["f_app_O"][:, 0], label="$F_x$")
    a0.plot(s["t"], s["f_app_O"][:, 2], label="$F_z$")
    a0.plot(s["t"], s["tau_meas"], label=r"$\tau$ (tip axis)")
    a0.set(xlabel="time in contact (s)", ylabel="N  /  N·m", title=f"{obj}: applied wrench in {{O}}")
    a0.legend()
    arc = s["phase"] == "arc"
    a1.plot(s["theta_deg"][arc], s["tau_meas"][arc], ".", ms=2, label="measured, pull (ARC)")
    a1.plot(s["theta_deg"][~arc], s["tau_meas"][~arc], ".", ms=2, label="measured, retract (UNARC)")
    a1.plot(s["theta_deg"], s["tau_fit_arc"], "C0-", lw=1, label="fit ARC")
    a1.plot(s["theta_deg"], s["tau_fit_unarc"], "C1-", lw=1, label="fit UNARC")
    a1.axvline(result.theta_star_deg, color="k", ls=":", label=rf"$\theta^*$ = {result.theta_star_deg:.1f}°")
    a1.set(xlabel="tilt θ (deg)", ylabel="pivot torque (N·m)", title="torque balance fit")
    a1.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", default="all", choices=["all", *OBJECTS])
    ap.add_argument("--trial", type=int, default=None, help="1-based trial index (default: all trials)")
    ap.add_argument("--pivot", choices=["relative", "fixed"], default="relative",
                    help="'relative' (paper default) removes per-trial object creep, recovered from the "
                         "fingertip arc; 'fixed' uses the nominal pivot for every trial")
    ap.add_argument("--smooth", choices=["butter", "savgol"], default=None,
                    help="optionally smooth the arc wrench before fitting (the paper uses none)")
    ap.add_argument("--plot", action="store_true", help="plot wrench and torque fit for the last trial run")
    ap.add_argument("-q", "--quiet", action="store_true", help="suppress per-trial lines")
    args = ap.parse_args()

    objs = list(OBJECTS) if args.object == "all" else [args.object]
    summary = []
    for obj in objs:
        print(f"\n[{OBJECTS[obj]['label']}]")
        est, truth, results = run_object(obj, args.trial, args.pivot, args.smooth, not args.quiet)
        summary.append((OBJECTS[obj]["label"], est, np.array(truth)))
        if args.plot:
            plot(obj, results[-1])

    n = len(summary[0][1])
    print(f"\n{'':<20s} {'m (kg)':^32s}   {'z_c (cm)':^31s}   {'mu_t':^30s}")
    print(f"{'object':<20s} {'GT':>7s} {f'est (N={n})':^24s}   {'GT':>6s} {'est':^24s}   {'GT':>5s} {'est':^24s}")
    for label, est, truth in summary:
        print(fmt_row(label, est, truth))
    print("\nest = mean ± SD over trials (relative error of the mean). mu_t GT is the same Mode 1 push "
          "resolved with the scale mass.")


if __name__ == "__main__":
    main()
