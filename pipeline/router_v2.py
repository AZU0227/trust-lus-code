#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRUST-LUS V4.2 — Router for the OOF mechanism pilot (soft-regret target + Gate), v2.

Frozen references (every number is quoted; nothing here is invented)
---------------------------------------------------------------------
Soft target — ``_router_plan.txt`` §4.2 L111-L115:
    ℓ_e = (p_e − y)²
    q_4 = exp(−ℓ_4/τ) / [exp(−ℓ_3/τ) + exp(−ℓ_4/τ)]
    L_route = BCE(α, q_4)

τ / C grid and the selection rule — ``Lock_R1_v2_draft_v1.1.md`` §7 B-01 (L475):
    τ ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
    C ∈ {0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0}
    "C 与 τ 由训练域患者分组交叉拟合 OOF 联合网格选出，主准则 = 平均 OOF L_route 最小
     （L_route = BCE(α, q_4)）；并列取较小 τ、再较小 C"
    "必须同时报告 τ 全网格上的 ΔAUC 与 ΔBrier"
    "代码中的 oof_brier 准则已废止（仅作 P2-B 历史记录）"

Gate structure — §7 B-05 (L479):
    main   = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000) on the
             5 standardized signals, output α = sigmoid(w·z + b)
    sensitivity = GBDT(max_depth=2, n_estimators=30), NOT tuned
    "✗ 不得声称实现了 V1.0 的双项损失 L = L_route + λ L_pred"

Direction isolation — ``Protocol_v2_v1_1_FINAL.md`` §5.6 L-43(a)-(d):
    DIR-A gate may reference OpenPOCUS source OOF only; DIR-B COVID-BLUeS only;
    the two directions never share supervision, normalizer, τ or C.

Table contract — ``_router_plan.txt`` L462 (V1.0) + L135-L157 (Step B):
``router_oof_table.csv`` has exactly 14 fields:
    dataset, patient_id, video_id, y, p3, p4, conf3, conf4,
    temp3, temp4, disagreement, loss3, loss4, q4

Implementation note on the τ criterion (a documented conflict, not a choice made
here)
-----------------------------------------------------------------------------
The task brief for this delivery asked for "τ* by minimum source-OOF Brier,
ties → larger τ".  Lock §7 B-01 explicitly **deprecates** the ``oof_brier``
criterion ("代码中的 oof_brier 准则已废止").  This module therefore defaults to
``--tau-criterion l_route`` (the lock) and keeps ``brier`` available **only** as a
deprecated, explicitly-acknowledged path (``--i-know-oof-brier-is-deprecated``)
so the conflict is visible instead of silently resolved.  The τ grid, the C grid,
the tie-break for the lock path (smaller τ, then smaller C) and the reporting
obligation (ΔAUC + ΔBrier over the whole τ grid) are all unchanged.

Target encoding — implementation fix 2026-10-24 (no criterion changed)
---------------------------------------------------------------------
The Gate of B-05 is a *classifier*, so the array handed to
``LogisticRegression`` must contain hard classes, never the continuous soft
target.  The main path is therefore:

    y_hard = I(ℓ_4 < ℓ_3)          (same z_i as the frozen Expert-choice
                                     definition, V1.0 §7.1 / L-11.4; ties -> 0)
    w      = q_4 / mean(q_4)       (soft regret as a mean-1 *sample weight*,
                                     so C keeps its meaning under every τ)

The selection criterion is untouched: it is still the mean OOF
``L_route = BCE(α, q_4)`` on the soft target, and ΔAUC/ΔBrier are still
reported over the whole τ grid.  ``--soft-target ignore`` keeps the hard label
but uses uniform weights (sensitivity only).  Passing a continuous target to
``fit_l2_logistic`` is refused explicitly (with a pointer to the sample-weight
path); no linear-regression surrogate is exposed.

CLI
---
    python src/router_v2.py --self-test
    python src/router_v2.py --fit-input <router_oof_input.csv> --direction DIR-A \
        --out-dir outputs/T1_router_oof_v1/router [--soft-target weights|ignore]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features_v2 import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    STATUS_MISSING_SIGNAL,
    STATUS_OK,
)
from src.norm_v2 import (  # noqa: E402
    DIRECTIONS,
    DirectionIsolationError,
    RobustNormalizer,
    canonical_sha,
    fit as fit_normalizer,
    unit_set_sha256,
)

# --------------------------------------------------------------------- grids
# Lock §7 B-01 (L475), verbatim, ascending.
TAU_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
TAU_CRITERION_LOCK = "l_route"      # B-01 main criterion (default)
TAU_CRITERION_DEPRECATED = "brier"  # B-01: "代码中的 oof_brier 准则已废止"
GBDT_PARAMS = {"max_depth": 2, "n_estimators": 30}   # B-05, not tuned
L2_MAX_ITER = 5000                                    # B-05 (sklearn max_iter)
AUC_NONE = float("nan")
PROB_EPS = 1e-12          # log/BCE guard only; not a threshold
PILOT_SEED = 13           # seed_reference.yaml formal_seeds[0]; s=1 pilot

# Target encoding of the gate (implementation switch, not a scientific choice).
# Lock B-05 says the main gate is a classifier, so the labels are always hard
# y=I(l4<l3); q4 only enters as a mean-1 sample weight in the default mode.
SOFT_TARGET_WEIGHTS = "weights"   # default: hard y + sample_weight=q4/mean(q4)
SOFT_TARGET_IGNORE = "ignore"     # sensitivity: hard y + uniform weights
SOFT_TARGET_MODES = (SOFT_TARGET_WEIGHTS, SOFT_TARGET_IGNORE)
DEFAULT_SOFT_TARGET = SOFT_TARGET_WEIGHTS

# 14-field contract (_router_plan.txt L462 / Step B table). Order is frozen.
TABLE_FIELDS = (
    "dataset",
    "patient_id",
    "video_id",
    "y",
    "p3",
    "p4",
    "conf3",
    "conf4",
    "temp3",
    "temp4",
    "disagreement",
    "loss3",
    "loss4",
    "q4",
)
TABLE_N_FIELDS = len(TABLE_FIELDS)

# feature -> table column mapping (frozen; the 5 signals are the same 5 numbers)
FEATURE_TO_TABLE = {
    "C_3": "conf3",
    "C_4": "conf4",
    "T_3": "temp3",
    "T_4": "temp4",
    "D_34": "disagreement",
}

_HAS_SKLEARN = False
try:  # server has scikit-learn 1.7.2 (environment_reference.yaml key_versions)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    _HAS_SKLEARN = True
except Exception:  # CPU-only prep machines may lack it; numpy fallback is used
    GradientBoostingClassifier = None
    LogisticRegression = None


class CrossDirectionLeakage(RuntimeError):
    """Raised when rows/tables of the two directions are mixed."""


class SingleClassFoldError(RuntimeError):
    """Raised when a fit set has only one hard-label class.

    Such a fold is **not** a failure of the run: the caller records ``n/a`` and
    leaves the fold's test rows unscored (no coefficients may be fabricated).
    """


# ------------------------------------------------------------------ soft target
def soft_target(l3, l4, tau: float) -> np.ndarray:
    """q_4 = softmax over (−ℓ_3/τ, −ℓ_4/τ) → P(M4 is the better expert).

    Implemented in a max-shifted form so that τ = 0.005 with losses up to ~1e4
    cannot overflow (the naive exp form returns nan/inf there).
    """
    tau = float(tau)
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    l3 = np.asarray(l3, dtype=float)
    l4 = np.asarray(l4, dtype=float)
    a = -l3 / tau
    b = -l4 / tau
    m = np.maximum(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    return eb / (ea + eb)


def hard_target(l3, l4) -> np.ndarray:
    """Hard Expert-choice label ``y = I(ℓ_4 < ℓ_3)`` in ``{0, 1}``.

    This is the frozen ``z_i`` of V1.0 §7.1 / Protocol L-11.4 (and the same
    comparison used by :func:`expert_choice_auroc`).  Exact ties give 0 ("M4 is
    strictly better" indicator), so the label stays binary and deterministic.
    """
    a = np.asarray(l3, dtype=float).ravel()
    b = np.asarray(l4, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"l3/l4 length mismatch: {a.size} vs {b.size}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("l3/l4 contain non-finite values; cannot build hard labels")
    return (b < a).astype(np.int64)


def soft_sample_weights(q4, mode: str = DEFAULT_SOFT_TARGET) -> np.ndarray:
    """Sample weights for the gate, normalized to mean **1**.

    ``weights`` (default): ``w = q4 / mean(q4)`` — the soft regret enters as a
    weight, never as a label.  Mean-1 normalization is deliberate: scaling all
    weights by a constant is equivalent to rescaling ``C``, so normalizing keeps
    ``C``'s meaning identical across the τ grid.
    ``ignore`` (sensitivity): uniform weights (hard label only).
    """
    if mode not in SOFT_TARGET_MODES:
        raise ValueError(f"unknown soft-target mode {mode!r}; expected one of {SOFT_TARGET_MODES}")
    q = np.asarray(q4, dtype=float).ravel()
    if q.size == 0:
        raise ValueError("empty q4; cannot build sample weights")
    if not np.all(np.isfinite(q)):
        raise ValueError("q4 contains non-finite values")
    if q.min() < 0.0 or q.max() > 1.0:
        raise ValueError(f"q4 must lie in [0,1], got [{q.min()},{q.max()}]")
    if mode == SOFT_TARGET_IGNORE:
        w = np.ones_like(q)
    else:
        mean = float(q.mean())
        if not mean > 0.0:
            raise ValueError(
                "q4 has zero mean (every row says M3 is better / single-class hard "
                "labels); cannot form mean-1 sample weights"
            )
        w = q / mean
    if not np.all(np.isfinite(w)):
        raise ValueError("sample weights are non-finite after normalization")
    if not np.isclose(float(w.mean()), 1.0, rtol=1e-12, atol=1e-12):
        raise AssertionError(f"sample weights must have mean 1, got {w.mean()}")
    return w


def assert_binary_labels(y) -> np.ndarray:
    """Assert that ``y`` is a subset of ``{0, 1}``; return int64 labels.

    (a) ``LogisticRegression`` only accepts discrete classes.  A continuous
    target such as q4 must be refused here with an actionable message instead
    of leaking into ``clf.fit`` as ``Unknown label type: continuous``.
    """
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("empty label vector; cannot fit the gate")
    if not np.all(np.isfinite(arr)):
        raise ValueError("labels contain NaN/inf; hard labels must be finite")
    uniq = np.unique(arr)
    if not np.all(np.isin(uniq, (0.0, 1.0))):
        shown = [float(v) for v in uniq[:5]]
        raise ValueError(
            "hard binary labels {0,1} required; got "
            f"{uniq.size} distinct value(s) {shown}. Continuous/soft targets such as "
            "q4 are not classification labels — the soft target must go through "
            "sample_weight: y_hard = I(l4 < l3), sample_weight = q4 / mean(q4) "
            "(see --soft-target weights); 软目标需走 sample_weight（q4 只作权重）."
        )
    return arr.astype(np.int64)


def _normalize_sample_weight(sample_weight, n: int):
    """Validate weights, rescale them to mean 1 and return ``(w, info)``."""
    if sample_weight is None:
        return None, {"kind": "none", "mean_before": None, "sum_after": float(n)}
    w = np.asarray(sample_weight, dtype=float).ravel()
    if w.size != n:
        raise ValueError(f"sample_weight length {w.size} != n rows {n}")
    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight contains non-finite values")
    if (w < 0.0).any():
        raise ValueError("sample_weight must be >= 0")
    total = float(w.sum())
    if not total > 0.0:
        raise ValueError("sample_weight sums to 0; cannot fit a weighted gate")
    mean_before = float(w.mean())
    w = w / mean_before
    if not np.isclose(float(w.mean()), 1.0, rtol=1e-12, atol=1e-12):
        raise AssertionError("internal: normalized sample weights must have mean 1")
    return w, {
        "kind": "provided_mean1",
        "mean_before": mean_before,
        "sum_after": float(w.sum()),
    }


def brier(y, p) -> float:
    """Mean Brier score (used only for reporting, per B-01)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def bce(target, prob) -> float:
    """Mean binary cross-entropy L_route = BCE(α, q_4) (B-01)."""
    t = np.asarray(target, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    return float(-np.mean(t * np.log(p) + (1.0 - t) * np.log(1.0 - p)))


def mixture(alpha, p3, p4) -> np.ndarray:
    """p_R = α·p_4 + (1−α)·p_3 (V1.0 §4.4 Router output semantics)."""
    a = np.asarray(alpha, dtype=float)
    return a * np.asarray(p4, dtype=float) + (1.0 - a) * np.asarray(p3, dtype=float)


# ------------------------------------------------------------------------ AUC
def auroc(y, score):
    """Rank (Mann-Whitney) AUROC with average ranks for ties; None if single class."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    if y.size == 0:
        return None
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ordered = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def expert_choice_auroc(alpha, l3, l4, y=None):
    """AUROC(α → z), z = I(ℓ_4 < ℓ_3)  (_router_plan.txt L217-L220, E-1.10)."""
    l3 = np.asarray(l3, dtype=float)
    l4 = np.asarray(l4, dtype=float)
    z = (l4 < l3).astype(int)
    if z.min() == z.max():
        return None
    return auroc(z, alpha)


# ------------------------------------------------------------ logistic (Gate)
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def fit_l2_logistic_numpy(X, y, C: float, max_iter: int = 100, tol: float = 1e-10,
                          sample_weight=None):
    """L2 logistic regression by Newton/IRLS — CPU fallback when sklearn is absent.

    Objective (same family as sklearn's lbfgs penalty):
        min_{w,b} 0.5·‖w‖² + C · Σ_i sample_weight_i · logloss(y_i, σ(w·x_i + b))
    Weights are rescaled to mean 1 first (so C keeps its scale); the intercept is
    unpenalized, matching sklearn's default.  Fully deterministic.
    """
    X = np.asarray(X, dtype=float)
    y = assert_binary_labels(y).astype(float)
    n, d = X.shape
    if n == 0:
        raise ValueError("cannot fit a gate on 0 rows")
    if y.size != n:
        raise ValueError(f"label length {y.size} != n rows {n}")
    sw, _ = _normalize_sample_weight(sample_weight, n)
    if sw is None:
        sw = np.ones(n, dtype=float)
    Z = np.hstack([X, np.ones((n, 1))])
    pen = np.zeros(d + 1)
    pen[:d] = 1.0
    theta = np.zeros(d + 1)
    p0 = float(np.clip((sw * y).sum() / sw.sum(), 1e-6, 1 - 1e-6))
    theta[-1] = math.log(p0 / (1.0 - p0))
    for _ in range(int(max_iter)):
        eta = Z @ theta
        p = _sigmoid(eta)
        v = np.maximum(sw * p * (1.0 - p), 1e-12)
        g = pen * theta + C * (Z.T @ (sw * (p - y)))
        H = np.diag(pen) + C * ((Z * v[:, None]).T @ Z)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        theta -= step
        if float(np.max(np.abs(step))) < tol:
            break
    return theta[:d], float(theta[-1])


def fit_l2_logistic(X, y, C: float, sample_weight=None) -> dict:
    """Fit the main Gate. sklearn path is authoritative (B-05); numpy is fallback.

    ``y`` **must** be hard labels in ``{0,1}`` (usually ``I(ℓ_4 < ℓ_3)``);
    :func:`assert_binary_labels` refuses anything else.  The soft target is
    passed as ``sample_weight`` (mean-1), never as the label vector.
    """
    X = np.asarray(X, dtype=float)
    y = assert_binary_labels(y)
    if X.shape[0] != y.size:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.size}")
    classes = np.unique(y)
    if classes.size < 2:
        raise SingleClassFoldError(
            f"single-class hard labels (all {int(classes[0])}); no gate may be "
            "fitted — record this fold as n/a instead of writing coefficients"
        )
    w, winfo = _normalize_sample_weight(sample_weight, y.size)
    meta = {
        "labels": "hard I(l4 < l3) in {0,1}",
        "label_pos": int(y.sum()),
        "label_neg": int(y.size - y.sum()),
        "sample_weight": winfo["kind"],
        "weight_mean_before": winfo["mean_before"],
        "weight_mean": 1.0 if w is not None else None,
        "weight_sum": (float(w.sum()) if w is not None else None),
        "C_effective": float(C) * (1.0 if w is None else float(w.mean())),
    }
    if _HAS_SKLEARN:
        clf = LogisticRegression(
            penalty="l2",
            C=float(C),
            solver="lbfgs",
            max_iter=L2_MAX_ITER,
            fit_intercept=True,
        )
        if w is None:
            clf.fit(X, y)
        else:
            clf.fit(X, y, sample_weight=w)
        coef = np.asarray(clf.coef_[0], dtype=float)
        intercept = float(np.asarray(clf.intercept_).ravel()[0])
        return {
            "backend": "sklearn.LogisticRegression(lbfgs,l2)",
            "coef": [float(v) for v in coef],
            "intercept": intercept,
            "C": float(C),
            **meta,
        }
    coef, intercept = fit_l2_logistic_numpy(X, y, float(C), sample_weight=w)
    return {
        "backend": "numpy.newton_irls(l2) [sklearn absent; CPU fallback]",
        "coef": [float(v) for v in coef],
        "intercept": intercept,
        "C": float(C),
        **meta,
    }


def predict_alpha(X, params) -> np.ndarray:
    coef = np.asarray(params["coef"], dtype=float)
    return _sigmoid(np.asarray(X, dtype=float) @ coef + float(params["intercept"]))


def fit_gbdt_sensitivity(X, y, sample_weight=None):
    """Sensitivity Gate: GBDT(max_depth=2, n_estimators=30), never tuned (B-05)."""
    y = assert_binary_labels(y)
    if np.unique(y).size < 2:
        return {
            "backend": "sklearn.GradientBoostingClassifier",
            "available": False,
            "status": "n/a",
            "reason": "single-class hard labels; GBDT not fitted (n/a)",
        }
    if not _HAS_SKLEARN:
        return {"backend": "unavailable: scikit-learn not installed", "available": False}
    w, _ = _normalize_sample_weight(sample_weight, y.size)
    clf = GradientBoostingClassifier(random_state=PILOT_SEED, **GBDT_PARAMS)
    if w is None:
        clf.fit(np.asarray(X, dtype=float), y)
    else:
        clf.fit(np.asarray(X, dtype=float), y, sample_weight=w)
    return {
        "backend": "sklearn.GradientBoostingClassifier",
        "available": True,
        "params": dict(GBDT_PARAMS),
        "random_state": PILOT_SEED,
        "labels": "hard I(l4 < l3) in {0,1}",
        "sample_weight": "mean1" if sample_weight is not None else "none",
        "feature_importances": [float(v) for v in clf.feature_importances_],
        "_model": clf,
    }


# --------------------------------------------------------- cross-fitting / grid
def _feature_matrix(rows: pd.DataFrame) -> np.ndarray:
    return rows[list(FEATURE_NAMES)].to_numpy(dtype=float)


def patient_grouped_folds(rows: pd.DataFrame) -> np.ndarray:
    """Frozen fold assignment, with a patient/cluster leakage assertion.

    The fold column is the frozen Stage-B assignment (B-08, 5 folds).  It must
    already keep a patient/cluster inside one fold; this function verifies that
    instead of re-deriving folds.
    """
    if "fold" not in rows.columns:
        raise ValueError("rows lack the frozen 'fold' column")
    folds = rows["fold"].astype(int).to_numpy()
    for key in ("patient_id", "cluster_id"):
        if key in rows.columns:
            bad = rows.groupby(key)["fold"].nunique()
            leaks = bad[bad > 1]
            if len(leaks):
                raise CrossDirectionLeakage(
                    f"{key} appears in more than one fold: {leaks.head(5).to_dict()}"
                )
    return folds


def cross_fit_alpha(rows: pd.DataFrame, tau: float, C: float, direction: str,
                    soft_target_mode: str = DEFAULT_SOFT_TARGET) -> dict:
    """Patient-grouped OOF α for one (τ, C) cell.

    For each frozen fold k: fit the Median/IQR normalizer + the L2 gate on the
    other folds' rows, then predict fold k.  The normalizer is therefore never
    fitted on the rows it scores (no unsupervised leakage either).

    The gate is trained on the **hard** label ``I(ℓ_4 < ℓ_3)``; the soft target
    q4 enters as a mean-1 sample weight when ``soft_target_mode='weights'``.
    A fold whose training labels are single-class is recorded in ``na_folds``
    and its test rows stay ``NaN`` — it is *not* an exception and never gets
    fabricated coefficients.

    Returns a dict (``alpha``, ``na_folds``, ``n_na_rows``, ...).
    """
    if soft_target_mode not in SOFT_TARGET_MODES:
        raise ValueError(
            f"unknown soft-target mode {soft_target_mode!r}; expected {SOFT_TARGET_MODES}"
        )
    folds = patient_grouped_folds(rows)
    alpha = np.full(len(rows), np.nan)
    na_folds = []
    fitted_folds = []
    for fold in sorted(set(int(f) for f in folds)):
        train = rows[folds != fold]
        test = rows[folds == fold]
        if len(train) == 0 or len(test) == 0:
            raise ValueError(f"fold {fold} is empty in one side; cannot cross-fit")
        y_train = hard_target(train["l3"], train["l4"])
        if np.unique(y_train).size < 2:
            na_folds.append(int(fold))
            continue
        w_train = soft_sample_weights(
            soft_target(train["l3"], train["l4"], tau), soft_target_mode
        )
        norm = fit_normalizer(train, direction)
        params = fit_l2_logistic(norm.transform(train), y_train, C, sample_weight=w_train)
        alpha[folds == fold] = predict_alpha(norm.transform(test), params)
        fitted_folds.append(int(fold))
    return {
        "alpha": alpha,
        "na_folds": na_folds,
        "fitted_folds": fitted_folds,
        "n_folds": int(len(set(int(f) for f in folds))),
        "n_na_folds": len(na_folds),
        "n_na_rows": int((~np.isfinite(alpha)).sum()),
        "n_scored_rows": int(np.isfinite(alpha).sum()),
        "soft_target": soft_target_mode,
    }


def evaluate_cell(rows: pd.DataFrame, tau: float, C: float, direction: str,
                  soft_target_mode: str = DEFAULT_SOFT_TARGET) -> dict:
    """One (τ, C) cell: OOF L_route, Brier, Expert-choice AUROC and ΔAUC/ΔBrier.

    B-01 requires reporting ΔAUC and ΔBrier over the whole τ grid.  They are
    computed on the *source* OOF only (no target-domain quantity is touched) and
    are reported, never used as the selection criterion.

    Rows belonging to an ``n/a`` fold are excluded from every metric (they have
    no OOF α by construction); ``status`` is ``ok`` / ``partial`` / ``n/a``.
    """
    fit = cross_fit_alpha(rows, tau, C, direction, soft_target_mode=soft_target_mode)
    alpha = fit["alpha"]
    q4 = soft_target(rows["l3"], rows["l4"], tau)
    scored = np.isfinite(alpha)
    n_scored = int(scored.sum())
    base = {
        "tau": float(tau),
        "C": float(C),
        "n_rows": int(len(rows)),
        "n_scored_rows": n_scored,
        "n_na_rows": int(fit["n_na_rows"]),
        "n_na_folds": int(fit["n_na_folds"]),
        "na_folds": list(fit["na_folds"]),
        "soft_target": soft_target_mode,
    }
    if n_scored == 0:
        base.update({
            "status": "n/a",
            "mean_l_route": None,
            "mean_brier_alpha_vs_y": None,
            "mean_brier_router_vs_y": None,
            "expert_choice_auroc": None,
            "auc_router": None,
            "auc_equal_average": None,
            "delta_auc_router_minus_ea": None,
            "delta_brier_ea_minus_router": None,
            "mean_alpha": None,
        })
        return base
    y = rows["y"].to_numpy(dtype=float)[scored]
    p3 = rows["p3"].to_numpy(dtype=float)[scored]
    p4 = rows["p4"].to_numpy(dtype=float)[scored]
    l3 = rows["l3"].to_numpy(dtype=float)[scored]
    l4 = rows["l4"].to_numpy(dtype=float)[scored]
    a = alpha[scored]
    p_r = mixture(a, p3, p4)
    p_ea = 0.5 * (p3 + p4)
    auc_r = auroc(y, p_r)
    auc_ea = auroc(y, p_ea)
    base.update({
        "status": "ok" if fit["n_na_rows"] == 0 else "partial",
        "mean_l_route": bce(q4[scored], a),
        "mean_brier_alpha_vs_y": brier(y, a),
        "mean_brier_router_vs_y": brier(y, p_r),
        "expert_choice_auroc": expert_choice_auroc(a, l3, l4),
        "auc_router": auc_r,
        "auc_equal_average": auc_ea,
        "delta_auc_router_minus_ea": (auc_r - auc_ea) if (auc_r is not None and auc_ea is not None) else None,
        "delta_brier_ea_minus_router": brier(y, p_ea) - brier(y, p_r),
        "mean_alpha": float(np.mean(a)),
    })
    return base


def choose_from_grid(cells, criterion: str = TAU_CRITERION_LOCK) -> dict:
    """Deterministic tie-broken selection on an already-computed grid.

    * ``l_route`` (Lock §7 B-01): min mean_l_route, then **smaller τ**, then smaller C.
    * ``brier``  (DEPRECATED by B-01; kept only for an explicitly-acknowledged
      audit run): min mean_brier_alpha_vs_y, then **larger τ**, then smaller C.

    Cells whose every fold is ``n/a`` (single-class hard labels) carry no
    L_route and are excluded from *selection* only; this does not alter the
    criterion or its tie-break for fittable cells.
    """
    if not cells:
        raise ValueError("empty grid")

    def usable(c):
        return (
            c.get("status", "ok") != "n/a"
            and c.get("mean_l_route") is not None
            and np.isfinite(float(c["mean_l_route"]))
        )

    pool = [c for c in cells if usable(c)]
    if not pool:
        raise ValueError(
            "no fittable grid cell: every fold has single-class hard labels "
            "I(l4 < l3); the run cannot select (τ, C) — record n/a, do not "
            "fabricate coefficients"
        )
    if criterion == TAU_CRITERION_LOCK:
        def key(c):
            return (float(c["mean_l_route"]), float(c["tau"]), float(c["C"]))
        chosen = min(pool, key=key)
        rule = "min mean_l_route; tie -> smaller tau; then smaller C"
    elif criterion == TAU_CRITERION_DEPRECATED:
        def key(c):
            return (float(c["mean_brier_alpha_vs_y"]), -float(c["tau"]), float(c["C"]))
        chosen = min(pool, key=key)
        rule = "min mean_brier_alpha_vs_y; tie -> larger tau; then smaller C [DEPRECATED per B-01]"
    else:
        raise ValueError(f"unknown tau criterion {criterion!r}")
    return {
        "criterion": criterion,
        "rule": rule,
        "chosen": chosen,
        "n_usable_cells": len(pool),
        "n_na_cells": len(cells) - len(pool),
    }


def select_tau_c(tau_grid=TAU_GRID, c_grid=C_GRID, criterion=TAU_CRITERION_LOCK) -> list:
    """Full τ × C grid, in ascending (τ, C) order (reported for B-01)."""
    if criterion not in (TAU_CRITERION_LOCK, TAU_CRITERION_DEPRECATED):
        raise ValueError(f"unknown tau criterion {criterion!r}")
    return [{"tau": float(t), "C": float(c)} for t in tau_grid for c in c_grid]


def run_grid(rows: pd.DataFrame, direction: str, tau_grid=TAU_GRID, c_grid=C_GRID,
             criterion: str = TAU_CRITERION_LOCK,
             soft_target_mode: str = DEFAULT_SOFT_TARGET) -> dict:
    """Evaluate the full grid and select (τ*, C*) by the requested criterion."""
    if soft_target_mode not in SOFT_TARGET_MODES:
        raise ValueError(
            f"unknown soft-target mode {soft_target_mode!r}; expected {SOFT_TARGET_MODES}"
        )
    if len(rows) < 10:
        raise ValueError(f"too few rows for a {len(tau_grid)}x{len(c_grid)} grid: {len(rows)}")
    cells = [
        evaluate_cell(rows, t, c, direction, soft_target_mode=soft_target_mode)
        for t in tau_grid for c in c_grid
    ]
    pick = choose_from_grid(cells, criterion)
    return {
        "direction": direction,
        "criterion": criterion,
        "soft_target": soft_target_mode,
        "grid": cells,
        "chosen": pick["chosen"],
        "rule": pick["rule"],
        "n_cells": len(cells),
        "n_na_cells": int(pick["n_na_cells"]),
        "tau_grid": [float(t) for t in tau_grid],
        "c_grid": [float(c) for c in c_grid],
    }


# ------------------------------------------------------------- table contract
def build_table(rows: pd.DataFrame, tau: float, dataset: str) -> pd.DataFrame:
    """Build the 14-field router_oof_table for one direction."""
    missing = [c for c in ("p3", "p4", "l3", "l4", "y") + FEATURE_NAMES if c not in rows.columns]
    if missing:
        raise ValueError(f"rows lack columns for the 14-field table: {missing}")
    table = pd.DataFrame(
        {
            "dataset": dataset,
            "patient_id": rows["patient_id"].astype(str).to_numpy(),
            "video_id": rows["unit_id"].astype(str).to_numpy(),
            "y": rows["y"].astype(int).to_numpy(),
            "p3": rows["p3"].astype(float).to_numpy(),
            "p4": rows["p4"].astype(float).to_numpy(),
            "conf3": rows["C_3"].astype(float).to_numpy(),
            "conf4": rows["C_4"].astype(float).to_numpy(),
            "temp3": rows["T_3"].astype(float).to_numpy(),
            "temp4": rows["T_4"].astype(float).to_numpy(),
            "disagreement": rows["D_34"].astype(float).to_numpy(),
            "loss3": rows["l3"].astype(float).to_numpy(),
            "loss4": rows["l4"].astype(float).to_numpy(),
        }
    )
    table["q4"] = soft_target(rows["l3"], rows["l4"], tau)
    return assert_table_contract(table)


def assert_table_contract(table: pd.DataFrame) -> pd.DataFrame:
    """The 14 fields, in the frozen order, and nothing else (no silent extras)."""
    cols = list(table.columns)
    if cols != list(TABLE_FIELDS):
        raise ValueError(
            f"router_oof_table must have exactly the {TABLE_N_FIELDS} frozen fields "
            f"{list(TABLE_FIELDS)}; got {len(cols)}: {cols}"
        )
    if len(table) == 0:
        raise ValueError("router_oof_table is empty")
    for col in ("y", "p3", "p4", "conf3", "conf4", "temp3", "temp4",
                "disagreement", "loss3", "loss4", "q4"):
        vals = table[col].to_numpy(dtype=float)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"router_oof_table column {col} has non-finite values")
    q4 = table["q4"].to_numpy(dtype=float)
    # q4 is a float sigmoid-of-softmax; at small τ it legitimately saturates to
    # exactly 0.0 / 1.0, so the contract is the closed interval (the saturated
    # row count is reported in the sidecar meta, never imputed).
    if q4.min() < 0.0 or q4.max() > 1.0:
        raise ValueError(f"q4 must lie inside [0,1], got [{q4.min()},{q4.max()}]")
    return table


def table_sha256(table: pd.DataFrame) -> str:
    """Canonical hash of a table's *content* (sorted rows, fixed field order)."""
    ordered = table.sort_values(["video_id"]).reset_index(drop=True)
    payload = ordered[list(TABLE_FIELDS)].astype(str).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_table(path) -> pd.DataFrame:
    """Read a router_oof_table.csv so that floats round-trip bit-for-bit.

    pandas' default CSV float parser can differ from the written double by 1 ULP,
    which would silently break ``table_sha256``; ``float_precision='round_trip'``
    is therefore mandatory for every consumer of this file.
    """
    path = Path(path)
    table = pd.read_csv(path, float_precision="round_trip")
    return assert_table_contract(table)


def write_table(table: pd.DataFrame, path: Path, meta: dict) -> dict:
    """Write router_oof_table.csv (LF newlines) + a sidecar meta JSON."""
    table = assert_table_contract(table)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # %.17g is the shortest float format that survives a CSV round-trip bit-for-bit,
    # so table_sha256(content) is stable across write -> read -> hash.
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")
    doc = {
        "direction": meta.get("direction"),
        "dataset": meta.get("dataset"),
        "tau": meta.get("tau"),
        "C": meta.get("C"),
        "tau_criterion": meta.get("tau_criterion"),
        "tau_criterion_is_deprecated": bool(meta.get("tau_criterion") == TAU_CRITERION_DEPRECATED),
        "soft_target": meta.get("soft_target"),
        "n_rows": int(len(table)),
        "fields": list(TABLE_FIELDS),
        "field_count": TABLE_N_FIELDS,
        "table_sha256": table_sha256(table),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "unit_set_sha256": unit_set_sha256(table["video_id"]),
        "q4_saturated_rows": int(
            ((table["q4"].to_numpy(dtype=float) <= 0.0) | (table["q4"].to_numpy(dtype=float) >= 1.0)).sum()
        ),
        "gate": meta.get("gate"),
        "sensitivity_gbdt": meta.get("sensitivity_gbdt"),
        "normalizer": meta.get("normalizer"),
        "grid_summary": meta.get("grid_summary"),
        "references": {
            "table_contract": "_router_plan.txt L462 / L135-L157 (Step B, 14 fields)",
            "tau_C_grid": "Lock_R1_v2_draft_v1.1.md §7 B-01 (L475)",
            "gate": "Lock_R1_v2_draft_v1.1.md §7 B-05 (L479)",
            "isolation": "Protocol_v2_v1_1_FINAL.md §5.6 L-43",
        },
    }
    (path.parent / (path.stem + ".meta.json")).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return doc


# ---------------------------------------------------- direction-leakage guards
def assert_no_shared_labels(table_a: pd.DataFrame, table_b: pd.DataFrame) -> None:
    """Machine-checkable "two directions never share labels" assertion."""
    for name, t in (("A", table_a), ("B", table_b)):
        assert_table_contract(t)
    ds_a = set(table_a["dataset"].astype(str))
    ds_b = set(table_b["dataset"].astype(str))
    if ds_a & ds_b:
        raise CrossDirectionLeakage(f"directions share a dataset: {sorted(ds_a & ds_b)}")
    vid = set(table_a["video_id"].astype(str)) & set(table_b["video_id"].astype(str))
    if vid:
        raise CrossDirectionLeakage(f"directions share {len(vid)} units, e.g. {sorted(vid)[:5]}")
    pat = set(table_a["patient_id"].astype(str)) & set(table_b["patient_id"].astype(str))
    if pat:
        raise CrossDirectionLeakage(f"directions share {len(pat)} patients, e.g. {sorted(pat)[:5]}")
    if unit_set_sha256(table_a["video_id"]) == unit_set_sha256(table_b["video_id"]):
        raise CrossDirectionLeakage("directions have identical unit sets")


def assert_direction_rows(rows: pd.DataFrame, direction: str) -> pd.DataFrame:
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}")
    if "direction" not in rows.columns:
        raise CrossDirectionLeakage("rows lack the 'direction' column")
    seen = sorted(set(rows["direction"].astype(str)))
    if seen != [direction]:
        raise CrossDirectionLeakage(f"gate({direction}) received direction(s) {seen}")
    if "library" in rows.columns:
        lib = DIRECTIONS[direction]["source"]
        libs = sorted(set(rows["library"].astype(str)))
        if libs != [lib]:
            raise CrossDirectionLeakage(
                f"gate({direction}) is bound to {lib!r} supervision only (L-43a/b), got {libs}"
            )
    if "role" in rows.columns:
        roles = sorted(set(rows["role"].astype(str)))
        if roles != ["source_oof"]:
            raise CrossDirectionLeakage(f"gate({direction}) received role(s) {roles}")
    return rows


# ------------------------------------------------------------- full direction
def fit_direction(
    rows: pd.DataFrame,
    direction: str,
    criterion: str = TAU_CRITERION_LOCK,
    tau_grid=TAU_GRID,
    c_grid=C_GRID,
    fit_sensitivity: bool = True,
    soft_target_mode: str = DEFAULT_SOFT_TARGET,
) -> dict:
    """End-to-end Router fit for one direction (table + τ/C + gate + sensitivity)."""
    if soft_target_mode not in SOFT_TARGET_MODES:
        raise ValueError(
            f"unknown soft-target mode {soft_target_mode!r}; expected {SOFT_TARGET_MODES}"
        )
    rows = assert_direction_rows(rows, direction)
    rows = rows.reset_index(drop=True)
    if (rows["status"].astype(str) != STATUS_OK).any():
        raise ValueError(
            "rows contain non-OK status; missing-signal units must be dropped upstream (B-12)"
        )
    dataset = DIRECTIONS[direction]["source"]

    grid = run_grid(
        rows, direction, tau_grid=tau_grid, c_grid=c_grid, criterion=criterion,
        soft_target_mode=soft_target_mode,
    )
    tau = float(grid["chosen"]["tau"])
    C = float(grid["chosen"]["C"])

    table = build_table(rows, tau, dataset)

    # final gate: normalizer + L2 logistic on all rows of this direction.
    # Labels are hard I(l4<l3); q4 is a mean-1 sample weight (default mode).
    norm = fit_normalizer(rows, direction)
    X = norm.transform(rows)
    q4 = soft_target(rows["l3"], rows["l4"], tau)
    y_hard = hard_target(rows["l3"], rows["l4"])
    if np.unique(y_hard).size < 2:
        gate = {
            "status": "n/a",
            "reason": "single-class hard labels I(l4<l3); gate not fitted (n/a)",
            "coef": None,
            "intercept": None,
            "C": float(C),
            "tau": float(tau),
            "tau_criterion": criterion,
            "soft_target": soft_target_mode,
            "n_rows": int(len(rows)),
            "alpha_mean": None,
        }
        w_gate = None
    else:
        w_gate = soft_sample_weights(q4, soft_target_mode)
        gate = fit_l2_logistic(X, y_hard, C, sample_weight=w_gate)
        gate["status"] = "ok"
        gate["tau"] = tau
        gate["tau_criterion"] = criterion
        gate["soft_target"] = soft_target_mode
        gate["target"] = (
            "hard y=I(l4<l3); sample_weight=q4/mean(q4) (B-05 classifier; "
            "soft regret is a weight, NOT a label)"
            if soft_target_mode == SOFT_TARGET_WEIGHTS
            else "hard y=I(l4<l3); uniform weights (sensitivity: soft target ignored)"
        )
        gate["n_rows"] = int(len(rows))
        gate["alpha_mean"] = float(np.mean(predict_alpha(X, gate)))

    sens = None
    if fit_sensitivity:
        sens = fit_gbdt_sensitivity(
            X, y_hard, sample_weight=(w_gate if w_gate is not None else None)
        )
        sens.pop("_model", None)

    return {
        "direction": direction,
        "dataset": dataset,
        "tau": tau,
        "C": C,
        "criterion": criterion,
        "soft_target": soft_target_mode,
        "grid": grid,
        "table": table,
        "normalizer": norm,
        "gate": gate,
        "sensitivity_gbdt": sens,
    }


# ---------------------------------------------------------------- self-test
def _synth_rows(direction="DIR-A", n=240, seed=3, library="openpocus"):
    rng = np.random.default_rng(seed)
    lib = library if direction == "DIR-A" else "covid_blues"
    rows = []
    for i in range(n):
        y = int(i % 2)
        patient = i // 8            # 8 units per patient -> 30 patients at n=240
        fold = (patient % 5) + 1    # patient-grouped: one patient lives in one fold
        # Alternate which expert is the better one every 2 units, so the hard
        # label I(l4<l3) is balanced (a continuous q4 cannot expose a single
        # class, but the hard label can) and the C_3/C_4 contrast is learnable.
        # The better expert is close to y (loss 0.04); the other is near chance
        # on the wrong side (loss 0.3025).
        m4_better = ((i // 2) % 2) == 1
        p3_close = 0.2 + 0.6 * y
        p4_close = 0.2 + 0.6 * y
        p3_near = 0.55 - 0.10 * y
        p4_near = 0.55 - 0.10 * y
        if m4_better:
            p3 = float(np.clip(p3_near + rng.normal(0, 0.03), 0.02, 0.98))
            p4 = float(np.clip(p4_close + rng.normal(0, 0.03), 0.02, 0.98))
        else:
            p3 = float(np.clip(p3_close + rng.normal(0, 0.03), 0.02, 0.98))
            p4 = float(np.clip(p4_near + rng.normal(0, 0.03), 0.02, 0.98))
        r = {
            "unit_id": f"{direction}_u{i:04d}",
            "cluster_id": f"{direction}_c{i // 4:03d}",
            "patient_id": f"{direction}-p{patient:03d}",
            "direction": direction,
            "role": "source_oof",
            "library": lib,
            "dataset": lib,
            "fold": fold,
            "y": y,
            "status": STATUS_OK,
            "n_frames": 40,
            "p3": p3,
            "p4": p4,
            "l3": (p3 - y) ** 2,
            "l4": (p4 - y) ** 2,
            "C_3": max(p3, 1 - p3),
            "C_4": max(p4, 1 - p4),
            "T_3": float(abs(rng.normal(0.05, 0.02))),
            "T_4": float(abs(rng.normal(0.05, 0.02))),
            "D_34": abs(p3 - p4),
        }
        rows.append(r)
    return pd.DataFrame(rows)


def _selftest() -> int:
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        if not cond:
            print(f"[FAIL] {name} {detail}")

    # 1. soft target: limits, symmetry, stability at the smallest tau
    l3 = np.array([1.0, 0.0, 0.25, 0.25])
    l4 = np.array([0.0, 1.0, 0.25, 0.25])
    q = soft_target(l3, l4, 0.05)
    check("q4 -> 1 when M4 is much better (l4 << l3)", abs(q[0] - 1.0) < 1e-6, q[0])
    check("q4 -> 0 when M4 is much worse (l4 >> l3)", abs(q[1] - 0.0) < 1e-6, q[1])
    check("q4 == 0.5 on ties", np.allclose(q[2:], 0.5))
    check("q4(l3,l4)+q4(l4,l3) == 1", np.allclose(q, 1.0 - soft_target(l4, l3, 0.05)))
    big = soft_target(np.array([1e4, 0.0]), np.array([0.0, 1e4]), 0.005)
    check("no overflow at tau=0.005, loss=1e4", np.all(np.isfinite(big)) and big[0] > 0.999999)
    check("tau must be > 0", _raises(lambda: soft_target([0.1], [0.2], 0.0), ValueError))
    check("tau grid is ascending and frozen", list(TAU_GRID) == sorted(TAU_GRID) and len(TAU_GRID) == 7)
    check("C grid is ascending and frozen", list(C_GRID) == sorted(C_GRID) and len(C_GRID) == 7)

    # 2. losses / metrics
    check("brier([0,1],[0,1]) == 0", brier([0, 1], [0.0, 1.0]) == 0.0)
    check("bce(0.5,0.5) == ln2", abs(bce([0.5], [0.5]) - math.log(2.0)) < 1e-12)
    check("auroc perfect == 1", auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0)
    check("auroc reversed == 0", auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0)
    check("auroc all-tied == 0.5", auroc([0, 1], [0.5, 0.5]) == 0.5)
    check("auroc single class -> None", auroc([1, 1], [0.2, 0.3]) is None)
    check(
        "expert_choice = I(l4<l3)",
        expert_choice_auroc(np.array([0.9, 0.1]), np.array([0.2, 0.2]), np.array([0.1, 0.9])) == 1.0,
    )

    # 3. gate fit sanity + determinism (hard label + mean-1 q4 weight)
    rows = _synth_rows("DIR-A", n=240)
    X = rows[list(FEATURE_NAMES)].to_numpy(dtype=float)
    q4 = soft_target(rows["l3"], rows["l4"], 0.05)
    y_hard = hard_target(rows["l3"], rows["l4"])
    w = soft_sample_weights(q4)
    g1 = fit_l2_logistic(X, y_hard, 1.0, sample_weight=w)
    g2 = fit_l2_logistic(X, y_hard, 1.0, sample_weight=w)
    check("gate coefs deterministic", g1["coef"] == g2["coef"] and g1["intercept"] == g2["intercept"])
    check("gate has 5 coefficients", len(g1["coef"]) == N_FEATURES)
    check(
        "gate labels are hard {0,1}",
        g1["label_pos"] + g1["label_neg"] == len(rows) and g1["label_pos"] > 0 and g1["label_neg"] > 0,
        (g1["label_pos"], g1["label_neg"]),
    )
    a = predict_alpha(X, g1)
    check("alpha in (0,1)", float(a.min()) > 0.0 and float(a.max()) < 1.0)
    check(
        "gate beats the weighted constant on the hard label",
        bce(y_hard, a) < bce(y_hard, np.full_like(a, float(np.average(y_hard, weights=w)))),
    )
    if _HAS_SKLEARN:
        sk = fit_l2_logistic(X, y_hard, 1.0, sample_weight=w)
        check("sklearn backend active", "sklearn" in sk["backend"], sk["backend"])
        ncoef, nint = fit_l2_logistic_numpy(X, y_hard, 1.0, sample_weight=w)
        rel = np.max(np.abs(ncoef - np.asarray(sk["coef"]))) / max(
            1e-9, np.max(np.abs(np.asarray(sk["coef"])))
        )
        alpha_gap = float(np.max(np.abs(predict_alpha(X, sk) - _sigmoid(X @ ncoef + nint))))
        # lbfgs stops at its default tol=1e-4, so coefficients can differ by a
        # few percent on near-zero directions while predictions agree tightly.
        check("numpy weighted fallback ~= sklearn (coef <=5%, alpha gap <=1e-3)",
              rel <= 0.05 and alpha_gap <= 1e-3, (rel, alpha_gap))
    else:
        check("sklearn absent -> documented numpy fallback", "numpy" in g1["backend"], g1["backend"])

    # 3a. assertion (a): a continuous target never reaches LogisticRegression
    check("continuous q4 rejected as a label", _raises(lambda: fit_l2_logistic(X, q4, 1.0), ValueError))
    err = None
    try:
        fit_l2_logistic(X, q4, 1.0)
    except ValueError as exc:  # the actionable message is part of the contract
        err = str(exc)
    check("rejection names the sample_weight path", err is not None and "sample_weight" in err, err)
    check("multi-valued labels rejected", _raises(lambda: fit_l2_logistic(X, np.arange(len(X)) % 3, 1.0), ValueError))
    check("NaN labels rejected", _raises(lambda: fit_l2_logistic(X, np.r_[np.nan, y_hard[1:]], 1.0), ValueError))
    check(
        "hard label == I(l4<l3)",
        np.array_equal(y_hard, (rows["l4"].to_numpy() < rows["l3"].to_numpy()).astype(int)),
    )
    check("hard label subset {0,1}", set(np.unique(y_hard)) <= {0, 1})
    check("tie l3==l4 -> 0", hard_target([0.2], [0.2])[0] == 0)
    check("single-class fit refused", _raises(lambda: fit_l2_logistic(X, np.zeros(len(X), dtype=int), 1.0), SingleClassFoldError))

    # 3b. assertion (b): q4 weights are mean-1 and do not drift C semantics
    check("weights mean == 1", np.isclose(float(w.mean()), 1.0))
    check("weights sum == n", np.isclose(float(w.sum()), float(len(w))))
    check("weights are exactly q4 / mean(q4)", np.allclose(w, q4 / q4.mean()))
    check("ignore mode -> uniform weights", np.allclose(soft_sample_weights(q4, SOFT_TARGET_IGNORE), 1.0))
    all_tau_mean1 = True
    for t in TAU_GRID:
        ww = soft_sample_weights(soft_target(rows["l3"], rows["l4"], t))
        all_tau_mean1 = all_tau_mean1 and np.isclose(float(ww.mean()), 1.0)
    check("mean-1 weights across all 7 tau values", all_tau_mean1)
    g3 = fit_l2_logistic(X, y_hard, 1.0, sample_weight=3.0 * w)
    check(
        "C semantics invariant to weight scale (mean-1 normalization)",
        np.allclose(np.asarray(g3["coef"]), np.asarray(g1["coef"]), rtol=1e-8, atol=1e-10)
        and np.isclose(g3["intercept"], g1["intercept"], rtol=1e-8, atol=1e-10),
    )
    g_unit = fit_l2_logistic(X, y_hard, 1.0)
    g_ones = fit_l2_logistic(X, y_hard, 1.0, sample_weight=np.ones(len(X)))
    check(
        "ignore path equals explicit uniform weights",
        np.allclose(np.asarray(g_unit["coef"]), np.asarray(g_ones["coef"]), rtol=1e-8, atol=1e-10),
    )
    check(
        "uniform weights differ from q4 weights (soft target has an effect)",
        not np.allclose(np.asarray(g_unit["coef"]), np.asarray(g1["coef"]), rtol=1e-6, atol=1e-8),
    )
    check("unknown soft-target mode refused", _raises(lambda: soft_sample_weights(q4, "linear"), ValueError))

    # 3c. assertion (c): a single-class fold is n/a, never a fabricated model
    partial = rows.copy()
    other = (partial["fold"].astype(int) != 3).to_numpy()
    partial.loc[other, "l4"] = partial.loc[other, "l3"] + 0.5  # every training set of fold 3 -> all y=0
    fit_partial = cross_fit_alpha(partial, 0.05, 1.0, "DIR-A")
    check("single-class fold recorded as n/a", fit_partial["na_folds"] == [3], fit_partial["na_folds"])
    na_mask = ~np.isfinite(fit_partial["alpha"])
    check(
        "n/a rows are exactly the skipped fold's test rows",
        np.array_equal(na_mask, (partial["fold"].astype(int) == 3).to_numpy()),
    )
    check("other folds still scored", bool(np.isfinite(fit_partial["alpha"][~na_mask]).all()))
    cell_partial = evaluate_cell(partial, 0.05, 1.0, "DIR-A")
    check(
        "partial cell flagged",
        cell_partial["status"] == "partial" and cell_partial["n_na_folds"] == 1,
        (cell_partial["status"], cell_partial["n_na_folds"]),
    )
    all_na = rows.copy()
    all_na["l4"] = all_na["l3"] + 0.5
    cell_na = evaluate_cell(all_na, 0.05, 1.0, "DIR-A")
    check(
        "all-n/a cell has no L_route",
        cell_na["status"] == "n/a" and cell_na["mean_l_route"] is None,
        cell_na["status"],
    )
    check("all-n/a grid refuses selection", _raises(lambda: choose_from_grid([cell_na]), ValueError))

    # 3d. assertion (d): minimal synthetic reproduction of the dry-run path
    tiny = _synth_rows("DIR-A", n=40, seed=11)
    tiny_X = tiny[list(FEATURE_NAMES)].to_numpy(dtype=float)
    tiny_q4 = soft_target(tiny["l3"], tiny["l4"], 0.1)
    check(
        "original dry-run call (q4 as y) now fails loudly, not inside sklearn",
        _raises(lambda: fit_l2_logistic(tiny_X, tiny_q4, 1.0), ValueError),
    )
    grid_tiny = run_grid(tiny, "DIR-A")
    check("minimal synthetic run_grid(49 cells) completes", grid_tiny["n_cells"] == 49 and grid_tiny["chosen"] is not None)
    check("every minimal-grid cell is fittable", all(c["status"] == "ok" for c in grid_tiny["grid"]))
    check("minimal grid chosen L_route is finite", np.isfinite(float(grid_tiny["chosen"]["mean_l_route"])))
    res_ign = fit_direction(rows, "DIR-A", soft_target_mode=SOFT_TARGET_IGNORE, fit_sensitivity=True)
    check(
        "soft-target ignore path runs with uniform weights",
        res_ign["gate"]["status"] == "ok"
        and res_ign["gate"]["sample_weight"] == "provided_mean1"
        and np.isclose(res_ign["gate"]["weight_mean_before"], 1.0),
        res_ign["gate"].get("sample_weight"),
    )

    # 4. tie-break rules are deterministic and match the lock
    base = {"n_rows": 100, "mean_brier_alpha_vs_y": 0.25, "expert_choice_auroc": 0.7}
    cells = [
        dict(base, tau=0.05, C=1.0, mean_l_route=0.50),
        dict(base, tau=0.02, C=3.0, mean_l_route=0.50),   # same L_route, smaller tau wins
        dict(base, tau=0.02, C=1.0, mean_l_route=0.50),   # same tau, smaller C wins
        dict(base, tau=0.50, C=0.01, mean_l_route=0.60),
    ]
    pick = choose_from_grid(cells, TAU_CRITERION_LOCK)
    check("l_route criterion picks min", pick["chosen"]["mean_l_route"] == 0.50)
    check("l_route tie -> smaller tau", pick["chosen"]["tau"] == 0.02, pick["chosen"])
    check("l_route tie -> then smaller C", pick["chosen"]["C"] == 1.0, pick["chosen"])
    b0 = dict(base, tau=0.05, C=1.0, mean_l_route=0.50, mean_brier_alpha_vs_y=0.20)
    b1 = dict(base, tau=0.20, C=1.0, mean_l_route=0.55, mean_brier_alpha_vs_y=0.20)
    pick_b = choose_from_grid([b0, b1], TAU_CRITERION_DEPRECATED)
    check("deprecated brier tie -> larger tau", pick_b["chosen"]["tau"] == 0.20, pick_b["chosen"])
    check("deprecated path is labelled", "DEPRECATED" in pick_b["rule"], pick_b["rule"])
    check("unknown criterion refused", _raises(lambda: choose_from_grid(cells, "nope"), ValueError))
    check(
        "grid is 7x7 = 49 cells in ascending order",
        len(select_tau_c()) == 49 and select_tau_c()[0] == {"tau": 0.005, "C": 0.01},
    )

    # 5. full direction fit: grid -> tau/C -> table -> gate
    res = fit_direction(rows, "DIR-A")
    check("fit_direction returns 49 cells", res["grid"]["n_cells"] == 49)
    check("chosen tau in grid", res["tau"] in TAU_GRID, res["tau"])
    check("chosen C in grid", res["C"] in C_GRID, res["C"])
    check("default criterion is the lock one", res["criterion"] == TAU_CRITERION_LOCK)
    check("expert-choice AUROC > 0.5 on the synthetic signal", (res["grid"]["chosen"]["expert_choice_auroc"] or 0) > 0.5)
    check("normalizer bound to DIR-A", res["normalizer"].direction == "DIR-A")
    check("default soft-target mode is weights", res["soft_target"] == SOFT_TARGET_WEIGHTS)
    check(
        "main gate is hard-label q4-weighted",
        res["gate"]["soft_target"] == SOFT_TARGET_WEIGHTS
        and "q4/mean(q4)" in res["gate"]["target"]
        and res["gate"]["labels"] == "hard I(l4 < l3) in {0,1}",
        res["gate"].get("target"),
    )
    check("selected cell keeps the soft L_route criterion", res["grid"]["chosen"]["mean_l_route"] is not None)

    # 6. the 14-field contract
    t = res["table"]
    check("table has exactly 14 fields", len(t.columns) == TABLE_N_FIELDS == 14, len(t.columns))
    check("table field order frozen", tuple(t.columns) == TABLE_FIELDS, list(t.columns))
    check("table row count == units", len(t) == len(rows))
    check("table dataset == source library", set(t["dataset"]) == {"openpocus"})
    check("q4 within [0,1]", t["q4"].min() >= 0 and t["q4"].max() <= 1)
    check(
        "table conf3 == feature C_3",
        np.allclose(t["conf3"], rows["C_3"]),
    )
    check("table temp4 == feature T_4", np.allclose(t["temp4"], rows["T_4"]))
    check("table disagreement == |p3-p4|", np.allclose(t["disagreement"], (rows["p3"] - rows["p4"]).abs()))
    check("table loss3 == (p3-y)^2", np.allclose(t["loss3"], (rows["p3"] - rows["y"]) ** 2))
    extra = t.copy()
    extra["leak"] = 1
    check("extra column rejected", _raises(lambda: assert_table_contract(extra), ValueError))
    dropped = t.drop(columns=["q4"])
    check("missing field rejected", _raises(lambda: assert_table_contract(dropped), ValueError))

    # 7. table write/read round-trip + hashes
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "router_oof_table.csv"
        doc = write_table(t, p, {"direction": "DIR-A", "dataset": "openpocus",
                                 "tau": res["tau"], "C": res["C"],
                                 "tau_criterion": TAU_CRITERION_LOCK,
                                 "soft_target": res["soft_target"],
                                 "gate": res["gate"], "normalizer": res["normalizer"].params()})
        back = read_table(p)
        check("round-trip preserves 14 fields", tuple(back.columns) == TABLE_FIELDS)
        check("round-trip preserves hash", doc["table_sha256"] == table_sha256(back))
        check(
            "default float parsing would NOT round-trip (helper is required)",
            doc["table_sha256"] != table_sha256(pd.read_csv(p)),
        )
        check("meta sidecar written", (Path(td) / "router_oof_table.meta.json").exists())
        check("meta records field_count", doc["field_count"] == 14)
        check("meta records unit set hash", len(doc["unit_set_sha256"]) == 64)
        check("meta records soft_target mode", doc["soft_target"] == SOFT_TARGET_WEIGHTS)

    # 8. no cross-direction sharing
    rows_b = _synth_rows("DIR-B", n=240, seed=5)
    res_b = fit_direction(rows_b, "DIR-B", fit_sensitivity=True)
    assert_no_shared_labels(res["table"], res_b["table"])
    check("A/B tables are disjoint and accepted", True)
    check("A/B tau/C chosen independently", (res["tau"], res["C"]) != (None, None))
    check(
        "shared unit set rejected",
        _raises(lambda: assert_no_shared_labels(t, t.copy()), CrossDirectionLeakage),
    )
    mixed = pd.concat([rows, rows_b], ignore_index=True)
    check("DIR-A gate refuses DIR-B rows", _raises(lambda: assert_direction_rows(mixed, "DIR-A"), CrossDirectionLeakage))
    check("DIR-B gate refuses DIR-A rows", _raises(lambda: assert_direction_rows(mixed, "DIR-B"), CrossDirectionLeakage))
    bad_lib = rows.copy()
    bad_lib["library"] = "covid_blues"
    check("source-library binding enforced", _raises(lambda: assert_direction_rows(bad_lib, "DIR-A"), CrossDirectionLeakage))
    bad_role = rows.copy()
    bad_role["role"] = "target"
    check("target rows refused", _raises(lambda: assert_direction_rows(bad_role, "DIR-A"), CrossDirectionLeakage))

    # 9. patient-grouped folds: leakage is refused, not silently averaged
    leaky = rows.copy()
    leaky.loc[leaky.index[0], "patient_id"] = leaky.loc[leaky.index[1], "patient_id"]
    leaky.loc[leaky.index[1], "fold"] = (leaky.loc[leaky.index[1], "fold"] % 5) + 1
    check("patient spanning two folds refused", _raises(lambda: patient_grouped_folds(leaky), CrossDirectionLeakage))

    # 10. missing-signal rows are refused, never imputed (B-12)
    broken = rows.copy()
    broken.loc[broken.index[0], "status"] = STATUS_MISSING_SIGNAL
    check("non-OK status refused", _raises(lambda: fit_direction(broken, "DIR-A"), ValueError))

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print("=" * 78)
    print(f"router_v2 self-test: {n_pass}/{len(checks)} PASS")
    print(f"tau_grid={TAU_GRID}")
    print(f"C_grid={C_GRID}")
    print(f"backend={'sklearn' if _HAS_SKLEARN else 'numpy-fallback'}; fields={TABLE_N_FIELDS}")
    print("=" * 78)
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print(json.dumps({"status": "FAIL", "failed": failed}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", "n_checks": len(checks)}, ensure_ascii=False))
    return 0


def _raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description="TRUST-LUS Router OOF fitting v2")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--fit-input", type=Path, help="per-unit OOF predictions csv (one direction)")
    ap.add_argument("--direction", choices=sorted(DIRECTIONS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--tau-criterion", default=TAU_CRITERION_LOCK,
                    choices=[TAU_CRITERION_LOCK, TAU_CRITERION_DEPRECATED])
    ap.add_argument("--i-know-oof-brier-is-deprecated", action="store_true",
                    help="required acknowledgement for --tau-criterion brier (Lock §7 B-01)")
    ap.add_argument("--soft-target", default=DEFAULT_SOFT_TARGET, choices=list(SOFT_TARGET_MODES),
                    help="gate target encoding: 'weights' (default; hard y=I(l4<l3) + "
                         "sample_weight=q4/mean(q4)) or 'ignore' (hard y + uniform weights, "
                         "sensitivity only). A continuous target is never fitted by a classifier.")
    args = ap.parse_args(argv)

    if args.self_test:
        return _selftest()

    if not (args.fit_input and args.direction and args.out_dir):
        ap.print_help()
        return 0
    if args.tau_criterion == TAU_CRITERION_DEPRECATED and not args.i_know_oof_brier_is_deprecated:
        raise SystemExit(
            "REFUSED: Lock_R1_v2_draft_v1.1.md §7 B-01 已废止 oof_brier 准则；"
            "若确需审计复算，请显式加 --i-know-oof-brier-is-deprecated"
        )
    rows = pd.read_csv(args.fit_input)
    res = fit_direction(rows, args.direction, criterion=args.tau_criterion,
                        soft_target_mode=args.soft_target)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    doc = write_table(
        res["table"],
        args.out_dir / f"router_oof_table_{args.direction}.csv",
        {
            "direction": args.direction,
            "dataset": res["dataset"],
            "tau": res["tau"],
            "C": res["C"],
            "tau_criterion": res["criterion"],
            "soft_target": res["soft_target"],
            "gate": res["gate"],
            "sensitivity_gbdt": res["sensitivity_gbdt"],
            "normalizer": res["normalizer"].params(),
            "grid_summary": {
                "n_cells": res["grid"]["n_cells"],
                "n_na_cells": res["grid"]["n_na_cells"],
                "rule": res["grid"]["rule"],
                "cells": res["grid"]["grid"],
            },
        },
    )
    (args.out_dir / f"gate_{args.direction}.json").write_text(
        json.dumps(res["gate"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({k: doc[k] for k in ("direction", "tau", "C", "n_rows", "soft_target", "table_sha256")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
