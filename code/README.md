# press-pull-estimator

A standalone, hardware-free implementation of the estimator from
**Multi-Modal Non-Prehensile Estimation of Physical Parameters via Press-and-Pull Tipping** (ISRR 2026).
It recovers an object's **mass m**, **CoM height z_c**, and **support friction μ_t** from wrist
force/torque data and robot proprioception.

It includes the recorded data for all 40 press-and-pull trials (4 objects × 10) and the four
Mode 1 pushes, and it reproduces Table 2 of the paper exactly. You don't need an ABB robot, ROS, or a camera.

```text
code/
├── run_offline_eval.py                    # CLI: run the estimator on the recorded trials
├── press_pull_estimator/
│   ├── estimator/wrench_estimator.py      # adjoint transform, torque-balance NLS, friction, filters
│   ├── sim/press_pull_sandbox.py          # minimal MuJoCo press-and-pull sandbox
│   ├── io.py                              # CSV loaders
│   └── objects.py                         # ground truth for the 4 benchmark objects
├── data/<object>/
│   ├── trial_XX_ft.csv.gz                 # t_s, fx, fy, fz, tx, ty, tz   (Net F/T, sensor frame, 500 Hz nominal)
│   ├── trial_XX_pose.csv.gz               # t_s, x, y, z, qx, qy, qz, qw, state (fingertip ball pose, base frame)
│   └── push_ft.csv.gz                     # Mode 1 sliding push, same columns as *_ft
├── tests/test_reproduce_paper.py          # asserts Table 2 is reproduced
└── tools/                                 # maintainer scripts (raw .npz → CSV, page JSON export)
```

## Install

```bash
cd code
pip install -e .            # numpy, scipy, matplotlib, mujoco. Python >= 3.10.
# pip install -e ".[record]"  # adds imageio for --record in the sandbox
```

## Offline evaluation on the robot data

```bash
python run_offline_eval.py --object flashlight          # one object, all 10 trials
python run_offline_eval.py --object all -q              # the full Table 2
python run_offline_eval.py --object heart --trial 3 --plot   # wrench and torque-fit plots for one trial
```

```text
                                  m (kg)                           z_c (cm)                            mu_t
object                    GT        est (N=10)              GT           est                 GT           est
Hollow Acrylic Box     0.676   0.707 ± 0.012 ( 4.6%)    15.00  15.10 ± 0.28 ( 0.7%)   0.156 0.149 ± 0.003 ( 4.4%)
Heart Prism            0.239   0.242 ± 0.006 ( 1.1%)    10.00  11.51 ± 0.27 (15.1%)   0.256 0.253 ± 0.006 ( 1.1%)
Curved Flashlight      0.387   0.396 ± 0.004 ( 2.4%)     9.38   9.38 ± 0.12 ( 0.0%)   0.240 0.234 ± 0.002 ( 2.3%)
Monitor                5.040   5.274 ± 0.008 ( 4.6%)    23.20  24.42 ± 0.05 ( 5.3%)   0.643 0.614 ± 0.001 ( 4.4%)
```

Options:

- `--pivot relative|fixed`: `relative` (the default, as in the paper) removes object creep between
  back-to-back trials. It fits a circle to each trial's fingertip arc and applies only that trial's
  deviation from the object's mean. It is skipped for objects that fail the no-slip check.
- `--smooth butter|savgol`: optionally smooth the arc wrench before fitting. The paper's numbers
  are produced **without** this, which is the default.

## How the estimator works

For each press-and-pull trial, `estimate_press_pull()` does the following:

1. **Tilt from proprioception.** Under no-slip the fingertip is carried rigidly by the object, so
   its rotation about the pivot edge is the object's tilt θ (`object_tilt`).
2. **Wrench to the pivot.** ${}^O w_{app} = -\mathrm{Ad}_{T_{SO}}^\top\, {}^S w_{meas}$
   (`applied_wrench_in_object`). This uses only the sensor pose, with no contact-point localization.
3. **Torque-balance NLS.** The table reaction passes through the pivot, so the moment balance
   involves only gravity: `least_squares` fits (m, z_c) to τ(θ). The horizontal CoM offset x_c is an
   input, as in the paper (Sec. 3.3), supplied by prior planar pushing.
4. **Hysteresis cancellation.** The pull (ARC) and return (UNARC) sweeps are fit separately and
   averaged. Fingertip friction reverses between them, so bias with opposite signs cancels.
5. **Friction.** `coulomb_product_from_push()` caches μ_t·m = f_t/g from the Mode 1 push (bias
   removed per axis using the pre-contact window, 6 Hz zero-phase Butterworth, median of the
   steady-sliding window). `resolve_friction()` then divides by the recovered m.

The public functions take plain numpy arrays, so you can feed them your own robot's data. Build a
`Trial` with `Trial.from_streams(t_ft, ft, t_pose, pose, state)`. `state` uses the controller ids
1 = press, 2 = hold, 3 = pull (ARC), 4 = return (UNARC), 5 = retract.

## MuJoCo sandbox

```bash
python -m press_pull_estimator.sim.press_pull_sandbox                    # box: m=0.676 kg, z_c=15 cm
python -m press_pull_estimator.sim.press_pull_sandbox --mass 1.1 --com-z 0.11
python -m press_pull_estimator.sim.press_pull_sandbox --object slab
python -m press_pull_estimator.sim.press_pull_sandbox --object heart --mu-table 0.256 --press 3   # also flashlight, monitor
python -m press_pull_estimator.sim.press_pull_sandbox --object box --mu-table 0.156 --forward-push  # no press: slides
python -m press_pull_estimator.sim.press_pull_sandbox --viewer            # interactive viewer, real time
python -m press_pull_estimator.sim.press_pull_sandbox --record run.mp4    # offscreen (MUJOCO_GL=egl on headless machines)
```

A floating spherical fingertip on stiff position-controlled joints stands in for the arm. It runs a
Mode 1 push, then press → hold → pull along an arc about the pivot → return. A virtual F/T sensor sums
the fingertip contact wrench and reports it in a sensor frame behind the ball, and the result goes
through the same `estimate_press_pull()` used on the real data. The true mass and CoM are set
through `<inertial>` independently of the geometry, and the estimator only gets x_c, as in the paper.

Notes and differences from the physical setup:

- **Finger orientation.** The robot keeps the finger world-fixed (paper Sec. 2.3) and relies on the
  compliant fingertip's rolling contact patch. A rigid simulated sphere can't roll that way, so by
  default the sandbox pitches the wrist with the arc (the alternative the paper mentions).
  `--world-fixed-finger` reproduces the paper's strategy: the ball then skids about R·θ, and z_c
  comes out a few percent high.
- **Arc termination.** As in the paper, the pull stops when the pull force drops to 10% of its peak
  (η_safety = 0.1), just short of θ*. θ* is recovered from the shape of the torque curve.
- **Pivot pinning.** The press's horizontal component grows with tilt. On a low-friction table
  (for example `--object slab --mu-table 0.15`) the base slides out. The sandbox detects this and
  prints a warning instead of reporting a meaningless estimate. On the robot, the adaptive press
  (Sec. 2.3) handles this case.
- **Benchmark stand-ins.** `heart`, `flashlight`, and `monitor` are boxes of roughly each object's
  size with its true mass and CoM. They don't model the heart's outline, the flashlight's curved
  base, or the monitor's flex. The light heart needs `--press 3`; at 8 N its base slides.
- **Forward push.** `--forward-push` skips estimation and pushes the near face at 3/4 height with no
  press, like conventional forward tipping. On the box's measured μ_t = 0.156 it slides instead of tipping.
- Mode 1 uses a nearly frictionless fingertip contact so that the finger does not drag the object
  into the table.

## Reproducing and regenerating

```bash
pytest -q tests                  # Table 2 regression + adjoint sanity check
python tools/export_web_data.py  # regenerate ../src/data/*.json for the project page
```

`tools/export_npz_to_csv.py` shows exactly how `data/` was produced from the lab's raw `.npz` logs.
