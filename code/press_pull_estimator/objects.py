"""Benchmark objects: ground truth and the known horizontal CoM offset.

Box, heart, and flashlight values come from CAD and a scale. The monitor's are
approximate (CoM to about +-1 cm). com_x is the horizontal CoM offset from the
pivot edge, which the estimator takes as known.
"""
OBJECTS = {
    "box":        {"label": "Hollow Acrylic Box", "mass": 0.676, "com_x": 0.05,   "com_z": 0.15},
    "heart":      {"label": "Heart Prism",        "mass": 0.239, "com_x": 0.0458, "com_z": 0.10},
    "flashlight": {"label": "Curved Flashlight",  "mass": 0.387, "com_x": 0.028,  "com_z": 0.0938},
    "monitor":    {"label": "Monitor",            "mass": 5.04,  "com_x": 0.06,   "com_z": 0.232},
}
