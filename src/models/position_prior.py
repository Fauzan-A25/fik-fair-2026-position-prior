"""Prior posisi: regresi isotonik target terhadap urutan baris, dihitung out-of-fold agar tidak bocor.

Metode ini dari notebook final tim (Lomba Terakhir Bersama Abil, FIK FAIR 2026); fungsinya dipindahkan ke modul dengan
argumen eksplisit, tanpa mengubah perhitungannya.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, RepeatedKFold


def prior_predict(pos_tr, y_tr, pos_q, increasing: bool) -> np.ndarray:
    iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    iso.fit(pos_tr, y_tr)
    return iso.predict(pos_q)


def inner_oof_prior(idx, pos, y, seed, increasing, n_splits=5) -> np.ndarray:
    """Prior OOF untuk baris `idx`, memakai hanya baris lain di `idx` (label residual tanpa bocor)."""
    out = np.zeros(len(idx))
    for a, b in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(idx):
        out[b] = prior_predict(pos[idx[a]], y[idx[a]], pos[idx[b]], increasing)
    return out


def full_oof_prior(pos, y, increasing, n_splits=10, n_repeats=5, seed=2026 + 911) -> np.ndarray:
    s, c = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(pos):
        s[va] += prior_predict(pos[tr], y[tr], pos[va], increasing)
        c[va] += 1
    return s / c


def block_splits(pos, n_blocks=10):
    """Blocked CV: setiap fold menahan satu blok posisi yang utuh."""
    g = np.floor(np.clip(pos, 0, 0.999999) * n_blocks).astype(int)
    return [(np.where(g != b)[0], np.where(g == b)[0]) for b in range(n_blocks) if np.any(g == b)]
