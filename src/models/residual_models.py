"""Model zoo yang mempelajari residual (target − prior posisi) dari 24 fitur, plus CV dan blending NNLS.

Konfigurasi model sama dengan notebook final tim. TabPFN (opsional di notebook asli) tidak dipakai karena butuh
internet untuk mengunduh bobot; run final tim juga berjalan tanpa TabPFN.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .position_prior import inner_oof_prior, prior_predict

MODEL_NAMES = ["ridge", "svr_pca12", "krr_raw", "gp_ard"]


def rmse(a, b) -> float:
    return float(mean_squared_error(a, b) ** 0.5)


def build_models(n_feat: int, seed: int = 2026) -> dict:
    gp_kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                 * Matern(length_scale=np.full(n_feat, 3.0), length_scale_bounds=(0.3, 100.0), nu=2.5)
                 + WhiteKernel(0.5, (1e-3, 5.0)))
    return {
        "ridge": Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=100.0))]),
        "svr_pca12": Pipeline([("s", StandardScaler()), ("p", PCA(n_components=min(12, n_feat), whiten=True, svd_solver="full")),
                               ("m", SVR(kernel="rbf", C=15, gamma=0.08, epsilon=0.10))]),
        "krr_raw": Pipeline([("s", StandardScaler()), ("m", KernelRidge(kernel="rbf", alpha=0.5, gamma=0.03))]),
        "gp_ard": Pipeline([("s", StandardScaler()), ("m", GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True,
                                                                                 n_restarts_optimizer=0, random_state=seed))]),
    }


def fit_predict_models(Xtr, ytr, Xq_list, seed=2026) -> dict:
    out = {}
    for name, mdl in build_models(Xtr.shape[1], seed).items():
        mdl.fit(Xtr, ytr)
        out[name] = [mdl.predict(Xq) for Xq in Xq_list]
    return out


def run_residual_cv(splits, X, y, pos, increasing, seed=2026):
    """Untuk setiap fold: label residual dari prior OOF di dalam fold latih, prior validasi dari seluruh fold latih."""
    n = len(y)
    prior_s, cnt, res_s = np.zeros(n), np.zeros(n), {}
    for k, (tr, va) in enumerate(splits):
        res_tr = y[tr] - inner_oof_prior(tr, pos, y, seed=seed + k, increasing=increasing)
        prior_s[va] += prior_predict(pos[tr], y[tr], pos[va], increasing)
        cnt[va] += 1
        for name, (p,) in fit_predict_models(X[tr], res_tr, [X[va]], seed).items():
            res_s.setdefault(name, np.zeros(n))[va] += p
    return prior_s / cnt, {k: v / cnt for k, v in res_s.items()}


def nnls_blend(res_list: list[dict], targets: list[np.ndarray], names: list[str]) -> np.ndarray:
    """Bobot non-negatif yang meminimalkan galat residual pada gabungan beberapa skema OOF."""
    R = np.vstack([np.column_stack([r[n] for n in names]) for r in res_list])
    w, _ = nnls(R, np.concatenate(targets))
    return w
