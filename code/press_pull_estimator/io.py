"""Load the gzipped CSV trial logs shipped in data/."""
from __future__ import annotations

import glob
import os

import numpy as np

from .estimator.wrench_estimator import Trial

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _read(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def load_trial(ft_path: str) -> Trial:
    """Load trial_XX_ft.csv.gz together with its sibling trial_XX_pose.csv.gz."""
    ft = _read(ft_path)
    pose = _read(ft_path.replace("_ft.csv", "_pose.csv"))
    return Trial.from_streams(ft[:, 0], ft[:, 1:7], pose[:, 0], pose[:, 1:8], pose[:, 8],
                              name=os.path.basename(ft_path).replace("_ft.csv.gz", ""))


def trial_paths(obj: str, data_dir: str = DATA_DIR) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, obj, "trial_*_ft.csv.gz")))


def load_push(obj: str, data_dir: str = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """(fx, fy) of the Mode 1 sliding push."""
    ft = _read(os.path.join(data_dir, obj, "push_ft.csv.gz"))
    return ft[:, 1], ft[:, 2]
