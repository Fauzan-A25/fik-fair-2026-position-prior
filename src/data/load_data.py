"""Load data FIK FAIR 2026 (regresi Anonymous Competition Score, metrik RMSE)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ID, TARGET = "record_id", "target"


def find_data_dir(local: str | Path = "../data/raw") -> Path:
    """Folder yang berisi train.csv kompetisi ini (di /kaggle/input, ikuti symlink)."""
    for root, _, files in os.walk("/kaggle/input", followlinks=True):
        if {"train.csv", "test.csv"} <= set(files):
            if TARGET in pd.read_csv(Path(root) / "train.csv", nrows=1).columns:
                return Path(root)
    return Path(local)


def load(data_dir: str | Path):
    d = Path(data_dir)
    train, test = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    features = [c for c in train.columns if c.startswith("feature_")]
    return train, test, features


def positions(n: int) -> np.ndarray:
    """Posisi baris dalam file, dinormalisasi ke (0, 1): (i + 0,5) / n."""
    return (np.arange(n) + 0.5) / n
