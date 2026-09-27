#!/usr/bin/env python3
"""Convert the lab's raw .npz trial logs into the gzipped CSVs shipped in data/.

Maintainer tool only. It is how data/ was produced from the irb120_ws runtime_logs,
and it is kept so the conversion can be audited. Users do not need to run it.

    python tools/export_npz_to_csv.py --src ~/irb120_ws/runtime_logs --dst data

Per object it writes:
    trial_XX_ft.csv.gz    t_s, fx, fy, fz, tx, ty, tz        (Net F/T, sensor frame, ~470 Hz)
    trial_XX_pose.csv.gz  t_s, x, y, z, qx, qy, qz, qw, state (fingertip ball pose in base frame, ~80 Hz)
    push_ft.csv.gz        t_s, fx, fy, fz, tx, ty, tz        (Mode 1 sliding push)

Aborted runs (empty streams) are skipped. The arc logs never recorded a valid sensor
TF, so the estimator always rebuilds the sensor pose from the fingertip pose and no
sensor-pose columns are exported.
"""
import argparse
import glob
import os

import numpy as np

OBJECTS = ("box", "heart", "flashlight", "monitor")
FT_COLS = ("fx", "fy", "fz", "tx", "ty", "tz")
POSE_COLS = ("x", "y", "z", "qx", "qy", "qz", "qw")


def _write(path, header, cols, fmts):
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(header),
               comments="", fmt=fmts)


def _export_ft(d, path):
    t = d["ft_time_s"]
    _write(path, ("t_s",) + FT_COLS, [t - t[0]] + [d[k] for k in FT_COLS],
           ["%.5f"] + ["%.5f"] * 6)
    return t[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="runtime_logs directory")
    ap.add_argument("--dst", required=True, help="output data directory")
    args = ap.parse_args()

    for obj in OBJECTS:
        out = os.path.join(args.dst, obj)
        os.makedirs(out, exist_ok=True)
        logs = sorted(f for f in glob.glob(os.path.join(args.src, obj, "arc_squash", "arc_static*.npz"))
                      if os.path.basename(f) != "most_recent.npz")
        k = 0
        for f in logs:
            d = np.load(f)
            if len(d["pose_time_s"]) == 0:
                print(f"skip (aborted): {f}")
                continue
            k += 1
            t0 = _export_ft(d, os.path.join(out, f"trial_{k:02d}_ft.csv.gz"))
            # Both streams share one time origin (the first F/T sample) so they stay aligned.
            _write(os.path.join(out, f"trial_{k:02d}_pose.csv.gz"),
                   ("t_s",) + POSE_COLS + ("state",),
                   [d["pose_time_s"] - t0] + [d[c] for c in POSE_COLS] + [d["controller_state_id"]],
                   ["%.5f"] + ["%.7f"] * 7 + ["%d"])
            print(f"{obj} trial {k:02d} <- {os.path.basename(f)}")
        _export_ft(np.load(os.path.join(args.src, obj, "push", "most_recent.npz")),
                   os.path.join(out, "push_ft.csv.gz"))


if __name__ == "__main__":
    main()
