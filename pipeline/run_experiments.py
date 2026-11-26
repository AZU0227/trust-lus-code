# -*- coding: utf-8 -*-
"""
TRUST-LUS next-four-experiments runner.

CPU-only post-processing. No model training, no GPU calls, no target-label tuning.
Input: frozen wide CSV with one analysis unit per row. Source rows carry fold=1..5;
target rows carry the five-fold mean probability and an empty fold (not OOF).

Advisor rulings implemented in v2.0.0 (line-level map in
stage3/report/next4_run_experiments_patch.md):
  R1  The frozen wide-table SHA256 is checked at startup; a mismatch on a real run
      exits with code 3. The input is opened read-only, is never a write target, and
      its fingerprint is re-checked after the run.
  R2  P1 runs on DIR-A and DIR-B. P2/P3/P4 run on DIR-B only. Any DIR-A target row is
      a hard refusal (exit 6); this module never generates a DIR-A target result.
  R3  Strict stage order p1 -> p2 -> p3 -> p4. --stage {p1,p2,p3,p4,all}. A later
      stage requires PASS markers of every earlier stage under <outdir>/_audit/.
  R4  Candidate-pool selection and router fitting receive split=source frames only.
      Target rows are used for final evaluation only. --assert-split-isolation runs a
      target-row permutation and target-label-poisoning invariance self-check.
  R5  M5 keeps AUROC/AUPRC/ranking/Brier and complementarity. Its tau=0.5 degenerate
      cells (DIR-A f2, DIR-A f5, DIR-B f2) are labelled DEGENERATE in a dedicated
      column. Pool strategies use continuous quantities (Brier, regret, rank
      disagreement), never the tau=0.5 error vector as the primary basis. Primary
      threshold/probability-cost routers use M1-M4 only. M5 may enter only the K=5
      static pool and rank-type supplementary results.
  R6  P4 cross-fitting: for each source fold k the router is fitted on the other four
      folds and evaluated on the held-out fold k. Features, rules, hyperparameters,
      thresholds and metrics are frozen constants (no search). After lock-in the router
      is refitted on all source OOF rows and applied to the DIR-B target. Fitting never
      reads target labels. "Fit on all OOF and evaluate on the same OOF" is not an
      official result (DIAGNOSTIC_NOT_OFFICIAL only, disabled by default).
  R7  Fixed evaluation: AUROC, AUPRC, Brier, EC-AUROC, deltaR and cluster-level 95% CI;
      deltaR is the primary decision metric.
  R8  E1-E8, the original deltaR definition and GO/NO-GO thresholds are untouched.
      This module never contained E1-E8 or GO/NO-GO logic, and deltaR remains
      AUROC(candidate) - AUROC(fixed reference) exactly as in v1.0.0.
  R9  --dry-run prints the SHA result, row/split/direction counts, the rows and columns
      each of P1-P4 will read, the output filenames and the assertion checklist.
      It writes nothing.
  2.0.1  Adds --legacy-sidecar (advisor ruling #6). The 14-column wide table carries no
      p_legacy_router, so P4's legacy M3/M4 replay used to be SKIPPED. When
      --legacy-sidecar <path> points at the sidecar written by
      scripts/112_legacy_router_sidecar.py (columns unit_id, cluster_id, y, p_route,
      source_row_sha256; one row per DIR-B target unit), P4 emits one extra
      legacy_M3M4 row with evaluation_mode=legacy_replay_sidecar computed from
      p_route against the fixed expert M4. That row is never pooled with the new
      router's deltaR, and without the flag the SKIPPED row is unchanged.

Exit codes:
  0 ok
  2 usage / configuration error
  3 frozen wide-table SHA256 mismatch
  4 stage prerequisite missing or stale
  5 data validation error
  6 DIR-A target row refused
  7 split-isolation self-check failed
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy.stats import spearmanr, rankdata
    import sklearn
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    DEPS_OK = True
    DEPS_ERROR = ""
except ImportError as exc:  # dry-run must work without scipy/sklearn
    scipy = None
    sklearn = None
    spearmanr = None
    rankdata = None
    roc_auc_score = None
    average_precision_score = None
    brier_score_loss = None
    LogisticRegression = None
    StandardScaler = None
    make_pipeline = None
    DEPS_OK = False
    DEPS_ERROR = str(exc)

VERSION = "2.0.1"
MODELS = ["M1", "M2", "M3", "M4", "M5"]
PROB_COLS = [f"p_{m}" for m in MODELS]
BASE_KEYS = ["direction", "split", "unit_id", "cluster_id", "y"]

# ----------------------------------------------------------------------------
# Frozen constants (R1, R2, R5, R6). Do not turn these into searchable knobs.
# ----------------------------------------------------------------------------
FROZEN_INPUT_SHA256 = "EF0508481B875A56DC9818C24209D0BF5E5535B2B01128CC791D1A1C61CEE864"
FROZEN_INPUT_BYTES = 1718015
FROZEN_INPUT_ROWS = 2119
FROZEN_TARGET_ROWS = 374
FROZEN_SOURCE_ROWS = 1745

SOURCE_SPLIT = "source"
TARGET_SPLIT = "target"
SPLIT_ALIASES = {"source": SOURCE_SPLIT, "source_oof": SOURCE_SPLIT,
                 "oof": SOURCE_SPLIT, "target": TARGET_SPLIT}

P1_DIRECTIONS = ("DIR-A", "DIR-B")
P234_DIRECTIONS = ("DIR-B",)
TARGET_DIRECTION = "DIR-B"
FORBIDDEN_TARGET_DIRECTION = "DIR-A"

PRIMARY_MODELS = ("M1", "M2", "M3", "M4")
M5_MODEL = "M5"
ALL_MODELS = ("M1", "M2", "M3", "M4", "M5")
STRATEGIES = ("A_TopK", "B_Performance_Diversity",
              "C_Performance_Diversity_Complementarity")

FOLD_VALUES = (1, 2, 3, 4, 5)
TARGET_FOLD_SENTINEL = -1
EXPECTED_SOURCE_FOLD_ROWS = {
    ("DIR-A", 1): 293, ("DIR-A", 2): 254, ("DIR-A", 3): 269,
    ("DIR-A", 4): 284, ("DIR-A", 5): 285,
    ("DIR-B", 1): 74, ("DIR-B", 2): 71, ("DIR-B", 3): 74,
    ("DIR-B", 4): 72, ("DIR-B", 5): 69,
}
M5_TAU05_DEGENERATE_CELLS = frozenset({("DIR-A", 2), ("DIR-A", 5), ("DIR-B", 2)})

STAGE_ORDER = ("p1", "p2", "p3", "p4")
AUDIT_SUBDIR = "_audit"
PRIMARY_DECISION_METRIC = "deltaR_vs_fixed"

# R6: frozen router specification. No grid search, no target-label tuning.
ROUTER_FEATURE_SET = ("p_pool", "binary_entropy")
ROUTER_C = 1.0
ROUTER_MAX_ITER = 1000
ROUTER_SOLVER = "lbfgs"
ROUTER_PENALTY = "l2"
ROUTER_CLASS_WEIGHT = None
ROUTER_RANDOM_STATE = 20261101
ROUTER_REFIT_ON_ALL_SOURCE = True

OFFICIAL_SOURCE_EVAL_MODES = ("cross_fitted_oof",)
OFFICIAL_TARGET_EVAL_MODES = ("target_final_evaluation", "target_full_source_fit",
                              "target_reference")
OFFICIAL_EVAL_MODES = OFFICIAL_SOURCE_EVAL_MODES + OFFICIAL_TARGET_EVAL_MODES
IN_SAMPLE_EVAL_MODE = "in_sample_oof"

SPLIT_ISOLATION_SEED = 20261109

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_SHA = 3
EXIT_STAGE = 4
EXIT_DATA = 5
EXIT_DIRA_TARGET = 6
EXIT_ISOLATION = 7

EXPECTED_COLUMNS = [
    "direction", "split", "unit_id", "cluster_id", "y",
    "p_M1", "p_M2", "p_M3", "p_M4", "p_M5",
    "source_path", "seed", "fold", "input_sha256",
]
ANALYSIS_COLUMNS = [
    "direction", "split", "unit_id", "cluster_id", "y",
    "p_M1", "p_M2", "p_M3", "p_M4", "p_M5", "fold",
]

STAGE_FILES = {
    "p1": ["P1_model_metrics.csv", "P1_model_metrics_by_fold.csv",
           "P1_pairwise_complementarity.csv", "P1_oracle_diagnostics.csv",
           "P1_M5_tau05_degeneracy.csv"],
    "p2": ["P2_pool_strategy_K2.csv"],
    "p3": ["P3_pool_size_comparison.csv"],
    "p4": ["P4_router_comparison.csv", "P4_router_weights.csv",
           "P4_crossfit_folds.csv"],
}
COMMON_FILES = ["RUN_METADATA.json"] + [
    "%s_AUDIT_PASS.json" % s.upper() for s in STAGE_ORDER
]

FROZEN_CRITERIA_NOTE = (
    "E1-E8, GO/NO-GO thresholds and the original deltaR definition are not "
    "implemented in this module and were not modified. deltaR is exactly "
    "AUROC(candidate) - AUROC(fixed reference), as in v1.0.0."
)


class DataError(Exception):
    """Data shape / content validation failure (exit 5)."""


class DirATargetError(Exception):
    """DIR-A target rows are prohibited (exit 6)."""


class IsolationError(Exception):
    """A fitting/selection function received non-source rows (exit 7)."""


class RouterFitError(Exception):
    """Router cannot be fitted for this pool/fold."""


def require_deps():
    if not DEPS_OK:
        print("ERROR: scipy and scikit-learn are required for a real run.")
        print("       missing dependency: %s" % DEPS_ERROR)
        raise SystemExit(EXIT_USAGE)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_fingerprint(path):
    st = Path(path).stat()
    return {"sha256": sha256_file(path), "bytes": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns)}


def verify_frozen_input(input_path, fp, dry_run):
    """R1: startup SHA256 gate. A real run exits 3 on any mismatch."""
    sha_match = fp["sha256"].lower() == FROZEN_INPUT_SHA256.lower()
    bytes_match = fp["bytes"] == FROZEN_INPUT_BYTES
    verdict = {
        "expected_sha256": FROZEN_INPUT_SHA256,
        "actual_sha256": fp["sha256"],
        "expected_bytes": FROZEN_INPUT_BYTES,
        "actual_bytes": fp["bytes"],
        "sha_match": bool(sha_match),
        "bytes_match": bool(bytes_match),
        "dry_run": bool(dry_run),
    }
    print("FROZEN INPUT CHECK (R1)")
    print("  path            : %s" % input_path)
    print("  expected SHA256 : %s" % FROZEN_INPUT_SHA256)
    print("  actual   SHA256 : %s" % fp["sha256"])
    print("  expected bytes  : %d" % FROZEN_INPUT_BYTES)
    print("  actual   bytes  : %d" % fp["bytes"])
    if sha_match and bytes_match:
        print("  verdict         : PASS")
        return verdict
    print("  verdict         : FAIL")
    if not dry_run:
        print("  action          : refusing to continue (exit %d)" % EXIT_SHA)
        raise SystemExit(EXIT_SHA)
    print("  dry-run         : mismatch acknowledged; structure-only checks continue.")
    print("                    A real run on this file would exit %d." % EXIT_SHA)
    print("                    The frozen table must be restored before P1-P4.")
    return verdict


def assert_input_unchanged(before, after):
    if (before["sha256"] != after["sha256"]
            or before["bytes"] != after["bytes"]
            or before["mtime_ns"] != after["mtime_ns"]):
        raise DataError(
            "input file changed while it was being read; it must be read-only. "
            "before=%s after=%s" % (before["sha256"], after["sha256"]))


def auc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def ap(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def metric_pack(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {"n": int(len(y)), "AUROC": auc(y, p), "AUPRC": ap(y, p),
            "Brier": brier(y, p), "Accuracy_at_0.5": float(np.mean((p >= .5) == y))}


def cluster_bootstrap_delta(y, clusters, p_new, p_ref, reps=2000, seed=20261101):
    """FROZEN (R8): deltaR = AUROC(candidate) - AUROC(reference), cluster bootstrap CI."""
    y = np.asarray(y, dtype=int)
    clusters = np.asarray(clusters)
    p_new = np.asarray(p_new, dtype=float)
    p_ref = np.asarray(p_ref, dtype=float)
    uniq = pd.unique(clusters)
    by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(reps)):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_cluster[c] for c in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        a, b = auc(y[idx], p_new[idx]), auc(y[idx], p_ref[idx])
        if np.isfinite(a) and np.isfinite(b):
            vals.append(a - b)
    if not vals:
        return {"delta": auc(y, p_new) - auc(y, p_ref), "ci_low": np.nan,
                "ci_high": np.nan, "valid_reps": 0}
    return {"delta": float(auc(y, p_new) - auc(y, p_ref)),
            "ci_low": float(np.quantile(vals, .025)),
            "ci_high": float(np.quantile(vals, .975)),
            "valid_reps": int(len(vals))}


def cluster_bootstrap_auc(y, clusters, p, reps=2000, seed=20261101):
    y = np.asarray(y, dtype=int)
    clusters = np.asarray(clusters)
    p = np.asarray(p, dtype=float)
    uniq = pd.unique(clusters)
    by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(reps)):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_cluster[c] for c in chosen])
        v = auc(y[idx], p[idx])
        if np.isfinite(v):
            vals.append(v)
    return (float(np.quantile(vals, .025)), float(np.quantile(vals, .975)),
            len(vals)) if vals else (np.nan, np.nan, 0)


def validate_and_aggregate(raw, cfg):
    """Validate the frozen wide table and normalise split/fold semantics."""
    required = BASE_KEYS + PROB_COLS
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise DataError("Input CSV missing required columns: " + ", ".join(missing))
    if "fold" not in raw.columns:
        raise DataError("Input CSV missing required column: fold (needed for P4 cross-fitting).")
    d = raw.copy()
    d["direction"] = d["direction"].astype(str).str.upper()
    raw_split = d["split"].astype(str).str.lower()
    d["split"] = raw_split.map(SPLIT_ALIASES)
    if d["split"].isna().any():
        bad = sorted(set(raw_split[d["split"].isna()].tolist()))
        raise DataError("split values must be source/source_oof or target; got: "
                        + ", ".join(bad))
    d["y"] = pd.to_numeric(d["y"], errors="raise").astype(int)
    if not set(d["y"].unique()).issubset({0, 1}):
        raise DataError("y must contain only binary labels 0/1.")
    if not set(d["direction"].unique()).issubset({"DIR-A", "DIR-B"}):
        raise DataError("direction values must be DIR-A and/or DIR-B.")
    if not set(d["split"].unique()).issubset({SOURCE_SPLIT, TARGET_SPLIT}):
        raise DataError("split values must be source and/or target.")
    for c in PROB_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        bad = d[c].dropna().between(0, 1)
        if len(bad) != d[c].notna().sum():
            raise DataError(f"{c} contains values outside [0,1].")

    # R2: any DIR-A target row is a hard refusal, before any other work.
    bad_dir_a_target = d[(d["split"] == TARGET_SPLIT)
                         & (d["direction"] == FORBIDDEN_TARGET_DIRECTION)]
    if len(bad_dir_a_target):
        raise DirATargetError(
            "DIR-A target rows are prohibited (R2): %d row(s) found. "
            "The target domain exists for DIR-B only." % len(bad_dir_a_target))

    # fold semantics: source rows must carry 1..5; target rows must be blank.
    fold_num = pd.to_numeric(d["fold"], errors="coerce")
    src_mask = d["split"] == SOURCE_SPLIT
    tgt_mask = d["split"] == TARGET_SPLIT
    if fold_num[src_mask].isna().any():
        raise DataError("source rows must have a fold value in 1..5.")
    if fold_num[tgt_mask].notna().any():
        raise DataError("target rows must have an empty fold (5-fold mean, not OOF).")
    src_folds = set(fold_num[src_mask].astype(int).unique())
    if not src_folds.issubset(set(FOLD_VALUES)):
        raise DataError("source fold values must be a subset of 1..5.")
    d["fold"] = TARGET_FOLD_SENTINEL
    d.loc[src_mask, "fold"] = fold_num[src_mask].astype(int)
    d["fold"] = d["fold"].astype(int)

    legacy_col = cfg.get("legacy_router_column", "p_legacy_router")
    if legacy_col and legacy_col in d.columns:
        d[legacy_col] = pd.to_numeric(d[legacy_col], errors="coerce")
        if not d[legacy_col].dropna().between(0, 1).all():
            raise DataError(f"{legacy_col} contains values outside [0,1].")
    else:
        legacy_col = None
    if d[BASE_KEYS].isna().any().any():
        raise DataError("direction/split/unit_id/cluster_id/y cannot be missing.")
    if "seed" in d.columns:
        # A seed row must not be partially missing and then silently averaged away.
        for c in PROB_COLS:
            miss_by_unit = d.groupby(["direction", "split", "unit_id"])[c].apply(
                lambda s: s.isna().all() or s.notna().all())
            if not bool(miss_by_unit.all()):
                raise DataError(f"{c}: inconsistent missingness across seeds for the same unit.")
        dup = d.duplicated(["direction", "split", "unit_id", "seed"], keep=False)
        if dup.any():
            raise DataError("Duplicate rows for the same direction/split/unit_id/seed.")
        group = ["direction", "split", "unit_id", "cluster_id", "y", "fold"]
        agg = {c: "mean" for c in PROB_COLS}
        if legacy_col:
            agg[legacy_col] = "mean"
        d = d.groupby(group, as_index=False).agg(agg)
    else:
        dup = d.duplicated(["direction", "split", "unit_id"], keep=False)
        if dup.any():
            raise DataError("Duplicate unit rows without a seed column. Aggregate views first.")
    if d.duplicated(["direction", "split", "unit_id"], keep=False).any():
        raise DataError("A unit has inconsistent cluster/label/fold metadata across seeds.")
    # One row per analysis unit. All five source OOF probabilities are mandatory.
    for direction in d["direction"].unique():
        s = d[(d.direction == direction) & (d.split == SOURCE_SPLIT)]
        if s.empty:
            raise DataError(f"{direction}: no source rows.")
        if s[PROB_COLS].isna().any().any():
            bad = [c for c in PROB_COLS if s[c].isna().any()]
            raise DataError(f"{direction}: source probabilities missing for {bad}.")
        folds = sorted(int(x) for x in s["fold"].unique())
        if folds != list(FOLD_VALUES):
            raise DataError(f"{direction}: expected source folds 1..5, got {folds}.")
    return d, legacy_col


def require_source_df(df, where):
    """R4: selection and fitting may only ever see split=source rows."""
    if "split" not in df.columns:
        raise IsolationError("%s: frame has no split column" % where)
    vals = set(df["split"].astype(str).unique())
    if vals != {SOURCE_SPLIT}:
        raise IsolationError("%s: expected source-only rows, got %s"
                             % (where, sorted(vals)))


def split_frames(d, direction):
    """Return (source, target) frames for one direction; DIR-A target is refused."""
    src = d[(d["direction"] == direction) & (d["split"] == SOURCE_SPLIT)].copy().reset_index(drop=True)
    tgt = d[(d["direction"] == direction) & (d["split"] == TARGET_SPLIT)].copy().reset_index(drop=True)
    if direction == FORBIDDEN_TARGET_DIRECTION and len(tgt):
        raise DirATargetError("DIR-A target rows are prohibited (R2).")
    if src.empty:
        raise DataError("%s: no source rows." % direction)
    return src, tgt


def model_auc_map(src):
    return {m: auc(src.y, src[f"p_{m}"]) for m in MODELS}


def error_vec(y, p):
    return ((np.asarray(p) >= .5) != np.asarray(y, dtype=int)).astype(int)


def diversity(src, a, b):
    """AUXILIARY ONLY (R5): tau=0.5 error disagreement. Never used for pool selection."""
    ea, eb = error_vec(src.y, src[f"p_{a}"]), error_vec(src.y, src[f"p_{b}"])
    return float(np.mean(ea != eb))


def pool_probs(df, pool):
    cols = [f"p_{m}" for m in pool]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError("missing probability columns: " + ", ".join(missing))
    return df[cols].to_numpy(float)


def brier_rank_disagreement(df, a, b):
    """Continuous complementarity: mean absolute rank distance of per-unit Brier losses."""
    y = df.y.to_numpy(float)
    la = (df[f"p_{a}"].to_numpy(float) - y) ** 2
    lb = (df[f"p_{b}"].to_numpy(float) - y) ** 2
    n = len(la)
    if n < 2:
        return 0.0
    ra = rankdata(la)
    rb = rankdata(lb)
    return float(np.mean(np.abs(ra - rb)) / (n - 1))


def brier_regret(df, model):
    """Continuous quantity: Brier(model) - min Brier over the frozen primary models."""
    y = df.y.to_numpy(float)
    b = float(np.mean((df[f"p_{model}"].to_numpy(float) - y) ** 2))
    best = min(float(np.mean((df[f"p_{m}"].to_numpy(float) - y) ** 2))
               for m in PRIMARY_MODELS)
    return b - best


def oracle_for_pool(df, pool):
    """Label-aware post-hoc diagnostic only; never used for selection or fitting."""
    ps = df[[f"p_{m}" for m in pool]].to_numpy(float)
    yy = df.y.to_numpy(float)
    losses = (ps - yy[:, None]) ** 2
    choice = np.argmin(losses, axis=1)
    chosen_p = ps[np.arange(len(df)), choice]
    return metric_pack(df.y, chosen_p), {
        m: float(np.mean(np.array(pool)[choice] == m)) for m in pool}


def oracle_choice(df, pool):
    ps = pool_probs(df, pool)
    yy = df.y.to_numpy(float)
    losses = (ps - yy[:, None]) ** 2
    choice = np.argmin(losses, axis=1)
    return ps[np.arange(len(ps)), choice], choice


def choose_fixed_expert(df, pool):
    """Identity of the fixed expert; callers pass a source frame (R4)."""
    aucs = {m: auc(df.y, df[f"p_{m}"]) for m in pool}
    return max(pool, key=lambda m: (aucs[m] if np.isfinite(aucs[m]) else -1,
                                    -MODELS.index(m)))


def onehot_weights(n, idx, k):
    w = np.zeros((n, k), dtype=float)
    w[:, idx] = 1.0
    return w


def uniform_weights(n, k):
    return np.full((n, k), 1.0 / float(k), dtype=float)


def ec_auroc(y, ps, weights, ref_idx):
    """EC-AUROC = AUROC(z, alpha) (frozen direction, generalised to a pool).

    z_i = I(loss(reference) < min loss over the other pool members); alpha_i is the
    router weight on the reference expert. For K=2 this is exactly the frozen
    definition z_i = I(l_ref < l_other), alpha = weight on the reference expert.
    Constant weights (fixed expert / equal average) carry no choice information and
    return 0.5 by construction.
    """
    y = np.asarray(y, dtype=float)
    ps = np.asarray(ps, dtype=float)
    if ps.ndim != 2 or ps.shape[1] < 2:
        return float("nan"), "K<2: EC-AUROC undefined"
    if ref_idx is None or ref_idx < 0 or ref_idx >= ps.shape[1]:
        return float("nan"), "no reference expert"
    if weights is None:
        weights = uniform_weights(ps.shape[0], ps.shape[1])
    weights = np.asarray(weights, dtype=float)
    if weights.shape != ps.shape:
        return float("nan"), "weights/pool shape mismatch"
    losses = (ps - y[:, None]) ** 2
    others = [j for j in range(ps.shape[1]) if j != ref_idx]
    l_ref = losses[:, ref_idx]
    l_alt = losses[:, others].min(axis=1)
    z = (l_ref < l_alt).astype(int)
    if len(np.unique(z)) < 2:
        return float("nan"), "z is single-class"
    s = weights[:, ref_idx]
    if not np.all(np.isfinite(s)):
        return float("nan"), "non-finite router weights"
    if np.allclose(s, s[0]):
        return 0.5, "constant weights: no expert-choice ranking information"
    return float(roc_auc_score(z, s)), "OK"


def select_pool(src, strategy, k, max_auc_gap):
    """Source-only candidate-pool selection (R4, R5).

    K<=4 draws from M1-M4 only. K=5 is the frozen static all-five pool. Strategies B
    and C score candidates with continuous Brier, regret and rank disagreement, never
    with the tau=0.5 error vector.
    """
    require_source_df(src, "select_pool")
    k = int(k)
    aucs = model_auc_map(src)
    if k == 5:
        return {"pool": list(ALL_MODELS), "fallback": 0,
                "pool_type": "STATIC_ALL5", "contains_m5": True,
                "strategy_effective": False,
                "selection_rule": "K=5 static pool (all five); strategy not applied"}
    if k not in (1, 2, 3, 4):
        raise DataError("pool size K=%d is not allowed; allowed K are 1,2,3,5." % k)
    candidates = list(PRIMARY_MODELS)
    ranked = sorted(candidates,
                    key=lambda m: (-(aucs[m] if np.isfinite(aucs[m]) else -1),
                                   MODELS.index(m)))
    if strategy == "A_TopK":
        return {"pool": ranked[:k], "fallback": 0,
                "pool_type": "PRIMARY_M1_M4", "contains_m5": False,
                "strategy_effective": True,
                "selection_rule": "Top-K by source OOF AUROC among M1-M4"}
    best = aucs[ranked[0]]
    eligible = [m for m in ranked if np.isfinite(aucs[m]) and aucs[m] >= best - max_auc_gap]
    fallback = 0
    if len(eligible) < k:
        for m in ranked:
            if m not in eligible:
                eligible.append(m)
                fallback += 1
                if len(eligible) >= k:
                    break
    chosen = [ranked[0]]
    while len(chosen) < k:
        cand = [m for m in eligible if m not in chosen]
        if not cand:
            cand = [m for m in ranked if m not in chosen]
        if strategy == "B_Performance_Diversity":
            def score(m):
                rdis = float(np.mean([brier_rank_disagreement(src, m, s) for s in chosen]))
                return (rdis, -brier_regret(src, m),
                        aucs[m] if np.isfinite(aucs[m]) else -1, -MODELS.index(m))
            nxt = max(cand, key=score)
        elif strategy == "C_Performance_Diversity_Complementarity":
            current_p = src[[f"p_{m}" for m in chosen]].mean(axis=1).to_numpy()
            current_b = brier(src.y, current_p)

            def score(m):
                new_p = src[[f"p_{x}" for x in chosen + [m]]].mean(axis=1).to_numpy()
                gain = current_b - brier(src.y, new_p)
                rdis = float(np.mean([brier_rank_disagreement(src, m, s) for s in chosen]))
                return (gain, rdis, -brier_regret(src, m),
                        aucs[m] if np.isfinite(aucs[m]) else -1, -MODELS.index(m))
            nxt = max(cand, key=score)
        else:
            raise DataError("unknown strategy: %s" % strategy)
        chosen.append(nxt)
    return {"pool": chosen, "fallback": int(fallback),
            "pool_type": "PRIMARY_M1_M4", "contains_m5": False,
            "strategy_effective": True,
            "selection_rule": "max_auc_gap=%.4f on source OOF AUROC; continuous Brier/regret/rank disagreement"
                              % max_auc_gap}


def eval_one(rows, base, df, pool, method, p, weights, ref_name, ref_idx,
             reps, seed, evaluation_mode, evaluation_split, tier, official,
             fixed_reference_basis, notes=""):
    # R6: official source evaluation must be cross-fitted; official target
    # evaluation may be a final target evaluation, but never an in-sample OOF row.
    if official:
        if evaluation_mode == IN_SAMPLE_EVAL_MODE:
            raise DataError("in-sample OOF evaluation cannot be official (R6).")
        if str(evaluation_split).startswith("source") \
                and evaluation_mode not in OFFICIAL_SOURCE_EVAL_MODES:
            raise DataError("official source evaluation must be cross-fitted, got '%s'"
                            % evaluation_mode)
    ps = pool_probs(df, pool)
    p_ref = ps[:, ref_idx]
    met = metric_pack(df.y, p)
    ci = cluster_bootstrap_delta(df.y, df.cluster_id, p, p_ref, reps, seed)
    ec, ec_status = ec_auroc(df.y, ps, weights, ref_idx)
    row = dict(base)
    row.update({
        "method": method,
        "evaluation_mode": evaluation_mode,
        "evaluation_split": evaluation_split,
        "tier": tier,
        "official": bool(official),
        "fixed_reference": ref_name,
        "fixed_reference_basis": fixed_reference_basis,
        "n": met["n"],
        "AUROC": met["AUROC"],
        "AUPRC": met["AUPRC"],
        "Brier": met["Brier"],
        "EC_AUROC": ec,
        "EC_AUROC_status": ec_status,
        "deltaR_vs_fixed": ci["delta"],
        "deltaR_ci_low": ci["ci_low"],
        "deltaR_ci_high": ci["ci_high"],
        "valid_bootstrap": ci["valid_reps"],
        "notes": notes,
    })
    rows.append(row)


def add_pool_baselines(rows, base, df, pool, fixed, ref_idx, reps, seed,
                       evaluation_mode, evaluation_split, tier, official,
                       fixed_reference_basis, oracle_note):
    ps = pool_probs(df, pool)
    n = len(df)
    p_fix = ps[:, ref_idx]
    eval_one(rows, base, df, pool, "fixed_expert", p_fix,
             onehot_weights(n, ref_idx, len(pool)), fixed, ref_idx, reps, seed,
             evaluation_mode, evaluation_split, tier, official,
             fixed_reference_basis,
             "fixed reference expert; deltaR is 0 by construction")
    eval_one(rows, base, df, pool, "equal_average", ps.mean(axis=1),
             uniform_weights(n, len(pool)), fixed, ref_idx, reps, seed,
             evaluation_mode, evaluation_split, tier, official,
             fixed_reference_basis,
             "equal-weight average of the pool")
    p_or, choice = oracle_choice(df, pool)
    w_or = np.zeros((n, len(pool)), dtype=float)
    w_or[np.arange(n), choice] = 1.0
    eval_one(rows, base, df, pool, "oracle_label_aware_diagnostic", p_or, w_or,
             fixed, ref_idx, reps, seed, evaluation_mode, evaluation_split,
             "%s_posthoc_oracle" % tier, False, fixed_reference_basis, oracle_note)


def make_features(df, pool):
    """FROZEN feature set (R6): pool probabilities plus binary entropy. No target data."""
    ps = df[[f"p_{m}" for m in pool]].to_numpy(float)
    eps = 1e-7
    ent = -(np.clip(ps, eps, 1 - eps) * np.log(np.clip(ps, eps, 1 - eps)) +
            (1 - np.clip(ps, eps, 1 - eps)) * np.log(1 - np.clip(ps, eps, 1 - eps)))
    return np.concatenate([ps, ent], axis=1)


def assert_router_frozen(cfg):
    c = float(cfg.get("router_C", ROUTER_C))
    mi = int(cfg.get("router_max_iter", ROUTER_MAX_ITER))
    if c != ROUTER_C or mi != ROUTER_MAX_ITER:
        print("ERROR: router hyperparameters are frozen at C=%.1f, max_iter=%d (R6)."
              % (ROUTER_C, ROUTER_MAX_ITER))
        raise SystemExit(EXIT_USAGE)


def fit_router_model(src_fit, pool, cfg):
    """R6: fit only on source rows; the expert-choice label is derived from source y."""
    require_source_df(src_fit, "fit_router_model")
    assert_router_frozen(cfg)
    ps = pool_probs(src_fit, pool)
    y = src_fit.y.to_numpy(float)
    choice = np.argmin((ps - y[:, None]) ** 2, axis=1)
    classes = np.unique(choice)
    if len(classes) < 2:
        raise RouterFitError("source OOF expert-choice labels contain fewer than 2 classes")
    X = make_features(src_fit, pool)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=ROUTER_C, max_iter=ROUTER_MAX_ITER, solver=ROUTER_SOLVER,
        penalty=ROUTER_PENALTY, class_weight=ROUTER_CLASS_WEIGHT,
        random_state=ROUTER_RANDOM_STATE))
    model.fit(X, choice)
    return model


def router_weights(model, df, pool):
    """Apply a fitted router to features only (safe for target rows: no labels read)."""
    if len(df) == 0:
        return np.zeros((0, len(pool)), dtype=float)
    qraw = model.predict_proba(make_features(df, pool))
    cls = model[-1].classes_
    q = np.zeros((len(df), len(pool)), dtype=float)
    for j, c in enumerate(cls):
        q[:, int(c)] = qraw[:, j]
    return q


def load_legacy_sidecar(path, tgt):
    """Rulings #6: load the legacy M3/M4 replay sidecar (unit_id, p_route).

    The sidecar is the only channel through which the legacy router enters this
    module. Only ``p_route`` is read as the replay signal; ``p3``/``p4`` never are
    (they are reference columns in the sidecar digest, not inputs here). The join
    must cover every DIR-B target row exactly once, must not disagree on ``y``, and
    must produce probabilities in [0, 1].
    """
    path = Path(path).resolve()
    if not path.exists():
        raise DataError("--legacy-sidecar not found: %s" % path)
    sd = pd.read_csv(path, encoding="utf-8-sig")
    need = {"unit_id", "p_route"}
    if not need.issubset(set(sd.columns)):
        raise DataError("--legacy-sidecar is missing columns: %s"
                        % sorted(need - set(sd.columns)))
    if sd["unit_id"].duplicated().any():
        raise DataError("--legacy-sidecar has duplicate unit_id rows")
    sd = sd.set_index("unit_id")
    units = tgt["unit_id"].tolist()
    missing = [u for u in units if u not in sd.index]
    if missing:
        raise DataError("--legacy-sidecar is missing %d target units (e.g. %s)"
                        % (len(missing), missing[:5]))
    sub = sd.reindex(units)
    p = sub["p_route"].to_numpy(dtype=float)
    if not np.isfinite(p).all():
        raise DataError("--legacy-sidecar p_route has non-finite values")
    if not ((p >= 0.0) & (p <= 1.0)).all():
        raise DataError("--legacy-sidecar p_route has values outside [0,1]")
    if "y" in sub.columns:
        y_side = sub["y"].to_numpy(dtype=float)
        if not np.array_equal(y_side, tgt["y"].to_numpy(dtype=float)):
            raise DataError("--legacy-sidecar y disagrees with the wide table")
    if "cluster_id" in sub.columns:
        n_diff = int((sub["cluster_id"].astype(str).to_numpy()
                      != tgt["cluster_id"].astype(str).to_numpy()).sum())
        if n_diff:
            print("WARN: --legacy-sidecar cluster_id differs on %d target rows" % n_diff)
    if len(sd) != len(units):
        print("WARN: --legacy-sidecar has %d rows, target block has %d; replay uses "
              "the joined rows only" % (len(sd), len(units)))
    return {"p": p, "path": str(path), "sha256": sha256_file(path),
            "rows": int(len(sd)),
            "source": "legacy M3/M4 router sidecar (p_route only)"}


def cross_fit_router(base, src, pool, cfg, reps, seed, weight_rows):
    """R6: leave-one-fold-out router. Fit on four folds, evaluate the held-out fold.

    Returns pooled out-of-fold soft/hard scores, the per-row weight matrix and the
    per-fold diagnostic rows. The in-sample path is never produced here.
    """
    require_source_df(src, "cross_fit_router")
    assert_router_frozen(cfg)
    n = len(src)
    k = len(pool)
    W = np.full((n, k), np.nan, dtype=float)
    soft = np.full(n, np.nan, dtype=float)
    hard = np.full(n, np.nan, dtype=float)
    fold_rows = []
    src_fold = src["fold"].to_numpy()
    for fold in FOLD_VALUES:
        fit_mask = src_fold != fold
        ev_mask = ~fit_mask
        fit_df = src.loc[fit_mask].reset_index(drop=True)
        ev_df = src.loc[ev_mask].reset_index(drop=True)
        if ev_df.empty:
            continue
        try:
            model = fit_router_model(fit_df, pool, cfg)
        except RouterFitError as exc:
            fold_rows.append({"fold": fold, "status": "SKIPPED: %s" % exc,
                              "n": int(len(ev_df))})
            continue
        q = router_weights(model, ev_df, pool)
        ps = pool_probs(ev_df, pool)
        soft[ev_mask] = (q * ps).sum(axis=1)
        hard[ev_mask] = ps[np.arange(len(ps)), np.argmax(q, axis=1)]
        W[ev_mask] = q
        fixed_fold = choose_fixed_expert(fit_df, pool)
        ridx = pool.index(fixed_fold)
        p_fix = ps[:, ridx]
        for method, p in (("router_soft_crossfit", soft[ev_mask]),
                          ("router_hard_crossfit", hard[ev_mask])):
            met = metric_pack(ev_df.y, p)
            ci = cluster_bootstrap_delta(ev_df.y, ev_df.cluster_id, p, p_fix,
                                         reps, seed + fold)
            ec, ec_status = ec_auroc(ev_df.y, ps, q, ridx)
            row = dict(base)
            row.update({
                "fold": fold,
                "method": method,
                "evaluation_mode": "cross_fitted_oof",
                "tier": "crossfitted_fold_diagnostic",
                "official": False,
                "fixed_reference": fixed_fold,
                "fixed_reference_basis": "source_train_4folds_auroc",
                "n": met["n"], "AUROC": met["AUROC"], "AUPRC": met["AUPRC"],
                "Brier": met["Brier"], "EC_AUROC": ec,
                "EC_AUROC_status": ec_status,
                "deltaR_vs_fold_fixed": ci["delta"],
                "deltaR_ci_low": ci["ci_low"], "deltaR_ci_high": ci["ci_high"],
                "valid_bootstrap": ci["valid_reps"],
                "notes": "held-out fold %d; router fitted on the other four folds" % fold,
            })
            fold_rows.append(row)
        for j, m in enumerate(pool):
            weight_rows.append({
                "scope": "crossfit_oof", "fold": fold,
                "direction": base["direction"], "strategy": base["strategy"],
                "K": base["K"], "pool": base["pool"], "expert": m,
                "mean_weight": float(np.mean(q[:, j])),
                "sd_weight": float(np.std(q[:, j])),
                "n": int(len(ev_df)),
            })
    mask = np.isfinite(soft) & np.isfinite(hard) & np.isfinite(W).all(axis=1)
    if mask.sum() == n:
        status = "OK"
    elif mask.any():
        status = "PARTIAL_CROSSFIT"
    else:
        status = "SKIPPED"
    return {"status": status, "mask": mask, "weights": W, "soft": soft,
            "hard": hard, "folds": fold_rows}


def m5_tau05_cell_status(direction, fold):
    return "DEGENERATE" if (direction, int(fold)) in M5_TAU05_DEGENERATE_CELLS else "OK"


def m5_pooled_status(direction):
    folds = sorted(f for (dd, f) in M5_TAU05_DEGENERATE_CELLS if dd == direction)
    if not folds:
        return "OK"
    return "DEGENERATE_IN_FOLD_" + "_".join(str(f) for f in folds)


def observed_one_sided(series):
    p = np.asarray(series, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return True
    pred = p >= 0.5
    return bool(np.unique(p).size <= 1 or pred.all() or (~pred).all())


def run_p1(d, cfg, ctx):
    rows, fold_rows, pair_rows, oracle_rows, m5_rows = [], [], [], [], []
    reps, seed = ctx["reps"], ctx["seed"]
    for direction in P1_DIRECTIONS:
        src, tgt = split_frames(d, direction)
        for m in MODELS:
            p = src[f"p_{m}"].to_numpy()
            met = metric_pack(src.y, p)
            lo, hi, nv = cluster_bootstrap_auc(src.y, src.cluster_id, p,
                                               reps, seed + MODELS.index(m))
            rows.append({
                "direction": direction, "split": SOURCE_SPLIT, "model": m, **met,
                "AUROC_ci_low": lo, "AUROC_ci_high": hi, "valid_bootstrap": nv,
                "tau05_status": (m5_pooled_status(direction) if m == M5_MODEL else "N/A"),
                "threshold_metrics_valid": not (m == M5_MODEL),
                "official": False, "tier": "source_descriptive",
                "notes": ("M5 tau=0.5 is degenerate in some folds; AUROC/AUPRC/Brier "
                          "remain valid, Accuracy_at_0.5 is not interpretable"
                          if m == M5_MODEL else ""),
            })
            for fold in FOLD_VALUES:
                f = src[src["fold"] == fold]
                if len(f) == 0:
                    continue
                metf = metric_pack(f.y, f[f"p_{m}"].to_numpy())
                fold_rows.append({
                    "direction": direction, "split": SOURCE_SPLIT, "fold": fold,
                    "model": m, **metf,
                    "tau05_status": (m5_tau05_cell_status(direction, fold)
                                     if m == M5_MODEL else "N/A"),
                    "observed_one_sided": (observed_one_sided(f[f"p_{m}"])
                                           if m == M5_MODEL else None),
                    "official": False, "tier": "source_descriptive_by_fold",
                })
        for i, a in enumerate(MODELS):
            for b in MODELS[i + 1:]:
                y = src.y.to_numpy(float)
                la = (src[f"p_{a}"].to_numpy(float) - y) ** 2
                lb = (src[f"p_{b}"].to_numpy(float) - y) ** 2
                ea = error_vec(src.y, src[f"p_{a}"])
                eb = error_vec(src.y, src[f"p_{b}"])
                avgp = src[[f"p_{a}", f"p_{b}"]].mean(axis=1).to_numpy()
                best_auc = max(auc(src.y, src[f"p_{a}"]), auc(src.y, src[f"p_{b}"]))
                rho = (float(spearmanr(la, lb).statistic)
                       if np.std(la) > 0 and np.std(lb) > 0 else np.nan)
                contains_m5 = (a == M5_MODEL or b == M5_MODEL)
                pair_rows.append({
                    "direction": direction, "model_a": a, "model_b": b, "n": len(src),
                    "loss_spearman": rho,
                    "brier_rank_disagreement": brier_rank_disagreement(src, a, b),
                    "error_disagreement_aux": float(np.mean(ea != eb)),
                    "a_better_loss_rate": float(np.mean(la < lb)),
                    "b_better_loss_rate": float(np.mean(lb < la)),
                    "equal_average_AUROC": auc(src.y, avgp),
                    "equal_average_AUROC_gain_vs_best": auc(src.y, avgp) - best_auc,
                    "equal_average_Brier": brier(src.y, avgp),
                    "equal_average_Brier_gain_vs_best":
                        min(brier(src.y, src[f"p_{a}"]), brier(src.y, src[f"p_{b}"]))
                        - brier(src.y, avgp),
                    "contains_m5": contains_m5,
                    "hard_threshold_status": ("AUXILIARY_ONLY_M5_TAU05_DEGENERATE"
                                              if contains_m5 else "AUXILIARY"),
                    "primary_complementarity_measure": "loss_spearman+brier_rank_disagreement",
                    "official": False, "tier": "source_descriptive",
                })
        om, win = oracle_for_pool(src, MODELS)
        oracle_rows.append({
            "direction": direction, "split": SOURCE_SPLIT, "pool": "+".join(MODELS),
            **{f"oracle_{k}": v for k, v in om.items()},
            "label_aware_choice_share": json.dumps(win, ensure_ascii=False),
            "warning": "Label-aware post-hoc diagnostic only; not deployable and not used for selection.",
            "official": False, "tier": "posthoc_diagnostic",
        })
        for fold in FOLD_VALUES:
            f = src[src["fold"] == fold]
            if len(f) == 0:
                continue
            p5 = f["p_M5"].to_numpy(float)
            pred = p5 >= 0.5
            status = m5_tau05_cell_status(direction, fold)
            observed = observed_one_sided(p5)
            if status == "DEGENERATE" and not observed:
                note = "expected degenerate cell but not observed one-sided; ruling kept"
            elif status == "OK" and observed:
                note = "unexpected one-sided M5 cell; flagged, ruling unchanged"
            else:
                note = ""
            m5_rows.append({
                "direction": direction, "fold": fold, "model": M5_MODEL,
                "n": int(len(f)),
                "positive_rate_at_0.5": float(np.mean(pred)),
                "unique_prob_values": int(np.unique(p5).size),
                "expected_degenerate": status == "DEGENERATE",
                "observed_one_sided": bool(observed),
                "tau05_status": status,
                "note": note,
            })
        # Target rows are evaluation-only and exist for DIR-B only (R2).
        if len(tgt):
            if direction != TARGET_DIRECTION:
                raise DirATargetError("DIR-A target rows are prohibited (R2).")
            complete = tgt[PROB_COLS].notna().all().all()
            if complete:
                for m in MODELS:
                    met = metric_pack(tgt.y, tgt[f"p_{m}"].to_numpy())
                    rows.append({
                        "direction": direction, "split": TARGET_SPLIT, "model": m,
                        **met, "AUROC_ci_low": np.nan, "AUROC_ci_high": np.nan,
                        "valid_bootstrap": 0,
                        "tau05_status": ("TARGET_NO_FOLD_CONTRACT"
                                         if m == M5_MODEL else "N/A"),
                        "threshold_metrics_valid": not (m == M5_MODEL),
                        "official": False, "tier": "target_descriptive_posthoc",
                        "notes": ("target is a 5-fold mean (no fold id); source M5 "
                                  "degeneracy cannot be localised here"
                                  if m == M5_MODEL
                                  else "target descriptive only; no CI; not a deployable claim"),
                    })
                om, win = oracle_for_pool(tgt, MODELS)
                oracle_rows.append({
                    "direction": direction, "split": "target_posthoc_diagnostic_only",
                    "pool": "+".join(MODELS),
                    **{f"oracle_{k}": v for k, v in om.items()},
                    "label_aware_choice_share": json.dumps(win, ensure_ascii=False),
                    "warning": "Uses target labels post hoc for diagnosis only; never used for selection or tuning.",
                    "official": False, "tier": "posthoc_diagnostic",
                })
    save = ctx["save"]
    save(rows, "P1_model_metrics.csv")
    save(fold_rows, "P1_model_metrics_by_fold.csv")
    save(pair_rows, "P1_pairwise_complementarity.csv")
    save(oracle_rows, "P1_oracle_diagnostics.csv")
    save(m5_rows, "P1_M5_tau05_degeneracy.csv")
    return list(STAGE_FILES["p1"])


def run_p2(d, cfg, ctx):
    rows = []
    src, tgt = split_frames(d, TARGET_DIRECTION)
    reps, seed = ctx["reps"], ctx["seed"]
    gap, primary_k = ctx["gap"], ctx["primary_k"]
    target_complete = len(tgt) > 0 and tgt[PROB_COLS].notna().all().all()
    for strategy in STRATEGIES:
        sel = select_pool(src, strategy, primary_k, gap)
        pool = sel["pool"]
        fixed = choose_fixed_expert(src, pool)
        ridx = pool.index(fixed)
        base = {"direction": TARGET_DIRECTION, "strategy": strategy, "K": primary_k,
                "pool": "+".join(pool), "pool_type": sel["pool_type"],
                "contains_m5": sel["contains_m5"],
                "selection_rule": sel["selection_rule"],
                "performance_filter_fallback_added": sel["fallback"]}
        add_pool_baselines(rows, base, src, pool, fixed, ridx, reps, seed,
                           "source_pool_selection", "source",
                           "source_selection_contaminated", False,
                           "source_full_auroc",
                           "label-aware oracle diagnostic on source rows")
        if target_complete:
            add_pool_baselines(rows, base, tgt, pool, fixed, ridx, reps, seed,
                               "target_final_evaluation", "target",
                               "target_primary", True, "source_full_auroc",
                               "label-aware oracle diagnostic on target rows")
        else:
            row = dict(base)
            row.update({"method": "ALL", "evaluation_mode": "target_final_evaluation",
                        "evaluation_split": "target", "tier": "target_primary",
                        "official": False,
                        "status": "SKIPPED: complete target M1-M5 probabilities are required",
                        "notes": ""})
            rows.append(row)
    ctx["save"](rows, "P2_pool_strategy_K2.csv")
    return list(STAGE_FILES["p2"])


def run_p3(d, cfg, ctx):
    rows = []
    src, tgt = split_frames(d, TARGET_DIRECTION)
    reps, seed = ctx["reps"], ctx["seed"]
    gap, primary_k = ctx["gap"], ctx["primary_k"]
    target_complete = len(tgt) > 0 and tgt[PROB_COLS].notna().all().all()
    for strategy in STRATEGIES:
        for k in ctx["pool_sizes"]:
            sel = select_pool(src, strategy, k, gap)
            pool = sel["pool"]
            fixed = choose_fixed_expert(src, pool)
            ridx = pool.index(fixed)
            base = {"direction": TARGET_DIRECTION, "strategy": strategy, "K": k,
                    "pool": "+".join(pool), "pool_type": sel["pool_type"],
                    "contains_m5": sel["contains_m5"],
                    "selection_rule": sel["selection_rule"],
                    "performance_filter_fallback_added": sel["fallback"]}
            add_pool_baselines(rows, base, src, pool, fixed, ridx, reps, seed,
                               "source_pool_selection", "source",
                               "source_selection_contaminated", False,
                               "source_full_auroc",
                               "label-aware oracle diagnostic on source rows")
            if target_complete:
                official = not sel["contains_m5"]
                tier = ("target_primary" if official
                        else "supplementary_static_all5")
                add_pool_baselines(rows, base, tgt, pool, fixed, ridx, reps, seed,
                                   "target_final_evaluation", "target", tier,
                                   official, "source_full_auroc",
                                   "label-aware oracle diagnostic on target rows")
            else:
                row = dict(base)
                row.update({"method": "ALL", "official": False,
                            "evaluation_mode": "target_final_evaluation",
                            "evaluation_split": "target", "tier": "target_primary",
                            "status": "SKIPPED: complete target M1-M5 probabilities are required"})
                rows.append(row)
    ctx["save"](rows, "P3_pool_size_comparison.csv")
    return list(STAGE_FILES["p3"])


def run_p4(d, cfg, ctx):
    rows, weight_rows, fold_rows = [], [], []
    src, tgt = split_frames(d, TARGET_DIRECTION)
    reps, seed = ctx["reps"], ctx["seed"]
    gap, primary_k = ctx["gap"], ctx["primary_k"]
    target_complete = len(tgt) > 0 and tgt[PROB_COLS].notna().all().all()
    for strategy in STRATEGIES:
        sel = select_pool(src, strategy, primary_k, gap)
        pool = sel["pool"]
        fixed = choose_fixed_expert(src, pool)
        ridx = pool.index(fixed)
        base = {"direction": TARGET_DIRECTION, "strategy": strategy, "K": primary_k,
                "pool": "+".join(pool), "pool_type": sel["pool_type"],
                "contains_m5": sel["contains_m5"],
                "selection_rule": sel["selection_rule"],
                "performance_filter_fallback_added": sel["fallback"]}
        # Source-side baselines (selection-contaminated) and the official cross-fitted router.
        add_pool_baselines(rows, base, src, pool, fixed, ridx, reps, seed,
                           "source_pool_selection", "source",
                           "source_selection_contaminated", False,
                           "source_full_auroc",
                           "label-aware oracle diagnostic on source rows")
        cf = cross_fit_router(base, src, pool, cfg, reps, seed, weight_rows)
        fold_rows.extend(cf["folds"])
        if cf["status"] in ("OK", "PARTIAL_CROSSFIT"):
            mask = cf["mask"]
            sub = src.loc[mask].reset_index(drop=True)
            W = cf["weights"][mask]
            for method, key in (("router_soft_crossfit", "soft"),
                                ("router_hard_crossfit", "hard")):
                eval_one(rows, base, sub, pool, method, cf[key][mask], W,
                         fixed, ridx, reps, seed, "cross_fitted_oof",
                         "source_oof_crossfit", "crossfitted_pooled", True,
                         "source_full_auroc",
                         "official cross-fitted source result; neither the router nor "
                         "the evaluation row was in its own training folds")
        else:
            row = dict(base)
            row.update({"method": "router_crossfit", "official": False,
                        "evaluation_mode": "cross_fitted_oof",
                        "evaluation_split": "source_oof_crossfit",
                        "tier": "crossfitted_pooled",
                        "status": "SKIPPED: cross-fitting produced no held-out predictions"})
            rows.append(row)
        # Target-side evaluation: router refit on ALL source OOF rows, then applied.
        if target_complete:
            add_pool_baselines(rows, base, tgt, pool, fixed, ridx, reps, seed,
                               "target_final_evaluation", "target",
                               "target_reference", True, "source_full_auroc",
                               "label-aware oracle diagnostic on target rows")
            try:
                model = fit_router_model(src, pool, cfg)
                q = router_weights(model, tgt, pool)
                ps = pool_probs(tgt, pool)
                p_soft = (q * ps).sum(axis=1)
                p_hard = ps[np.arange(len(ps)), np.argmax(q, axis=1)]
                for method, p in (("router_soft_target", p_soft),
                                  ("router_hard_target", p_hard)):
                    eval_one(rows, base, tgt, pool, method, p, q, fixed, ridx,
                             reps, seed, "target_full_source_fit", "target",
                             "target_primary_router", True, "source_full_auroc",
                             "final router fitted on all source OOF rows; target labels "
                             "used only in this evaluation")
                for j, m in enumerate(pool):
                    weight_rows.append({
                        "scope": "target_final_full_source_fit", "fold": -1,
                        "direction": TARGET_DIRECTION, "strategy": strategy,
                        "K": primary_k, "pool": "+".join(pool), "expert": m,
                        "mean_weight": float(np.mean(q[:, j])),
                        "sd_weight": float(np.std(q[:, j])),
                        "n": int(len(tgt)),
                    })
            except RouterFitError as exc:
                row = dict(base)
                row.update({"method": "router_target", "official": False,
                            "evaluation_mode": "target_full_source_fit",
                            "evaluation_split": "target",
                            "tier": "target_primary_router",
                            "status": "SKIPPED: %s" % exc})
                rows.append(row)
        else:
            row = dict(base)
            row.update({"method": "router_target", "official": False,
                        "evaluation_mode": "target_full_source_fit",
                        "evaluation_split": "target",
                        "tier": "target_primary_router",
                        "status": "SKIPPED: complete target M1-M5 probabilities are required"})
            rows.append(row)
        # R6: the in-sample path is never official; disabled by default.
        if ctx["emit_in_sample"]:
            try:
                model = fit_router_model(src, pool, cfg)
                q = router_weights(model, src, pool)
                ps = pool_probs(src, pool)
                p_soft = (q * ps).sum(axis=1)
                p_hard = ps[np.arange(len(ps)), np.argmax(q, axis=1)]
                diag = []
                for method, p in (("router_soft_in_sample", p_soft),
                                  ("router_hard_in_sample", p_hard)):
                    eval_one(diag, base, src, pool, method, p, q, fixed, ridx,
                             reps, seed, IN_SAMPLE_EVAL_MODE, "source_oof",
                             "DIAGNOSTIC_NOT_OFFICIAL", False, "source_full_auroc",
                             "fit on all source OOF and evaluated on the same rows; "
                             "not an official result")
                ctx["save"](diag, "P4_in_sample_diagnostic_NOT_OFFICIAL.csv")
            except RouterFitError as exc:
                print("in-sample diagnostic skipped: %s" % exc)
    # Supplementary: K=5 static pool, no learned router (R5).
    pool = list(ALL_MODELS)
    fixed = choose_fixed_expert(src, pool)
    ridx = pool.index(fixed)
    base = {"direction": TARGET_DIRECTION, "strategy": "STATIC", "K": 5,
            "pool": "+".join(pool), "pool_type": "STATIC_ALL5",
            "contains_m5": True,
            "selection_rule": "K=5 static pool; M5 supplementary only"}
    add_pool_baselines(rows, base, src, pool, fixed, ridx, reps, seed,
                       "source_pool_selection", "source",
                       "supplementary_static_all5", False, "source_full_auroc",
                       "label-aware oracle diagnostic on source rows")
    if target_complete:
        add_pool_baselines(rows, base, tgt, pool, fixed, ridx, reps, seed,
                           "target_final_evaluation", "target",
                           "supplementary_static_all5", False, "source_full_auroc",
                           "label-aware oracle diagnostic on target rows")
    # Legacy M3/M4 router replay. Primary channel: the --legacy-sidecar file built by
    # scripts/112_legacy_router_sidecar.py (ruling #6). Fallback: a p_legacy_router
    # column, if a future wide table version carries one. Either way the replay is its
    # own row with its own evaluation_mode and is never pooled with the new router.
    legacy_col = ctx["legacy_col"]
    sidecar_path = getattr(ctx["args"], "legacy_sidecar", None)
    if sidecar_path:
        if not target_complete or tgt["p_M4"].isna().any():
            raise DataError("--legacy-sidecar requires complete DIR-B target rows "
                            "with p_M4")
        sidecar = load_legacy_sidecar(sidecar_path, tgt)
        ctx["legacy_sidecar_meta"] = {k: sidecar[k]
                                      for k in ("path", "sha256", "rows", "source")}
        p_old = sidecar["p"]
        p4 = tgt["p_M4"].to_numpy(dtype=float)
        met = metric_pack(tgt.y, p_old)
        ci = cluster_bootstrap_delta(tgt.y, tgt.cluster_id, p_old, p4, reps, seed)
        rows.append({
            "direction": TARGET_DIRECTION, "strategy": "legacy_M3M4", "K": 2,
            "pool": "M3+M4", "pool_type": "LEGACY", "contains_m5": False,
            "selection_rule": "frozen legacy M3/M4 router replay from sidecar",
            "method": "legacy_M3M4_router",
            "evaluation_mode": "legacy_replay_sidecar",
            "evaluation_split": "target", "tier": "target_legacy_replay",
            "official": False, "fixed_reference": "M4",
            "fixed_reference_basis": "legacy_contract",
            "n": met["n"], "AUROC": met["AUROC"], "AUPRC": met["AUPRC"],
            "Brier": met["Brier"], "EC_AUROC": np.nan,
            "EC_AUROC_status": "legacy router: EC-AUROC not defined in this module",
            "deltaR_vs_fixed": ci["delta"], "deltaR_ci_low": ci["ci_low"],
            "deltaR_ci_high": ci["ci_high"], "valid_bootstrap": ci["valid_reps"],
            "legacy_sidecar_sha256": sidecar["sha256"], "status": "OK",
            "notes": ("same-unit replay on the DIR-B target rows; separate "
                      "evaluation_mode, never mixed with the new-pool deltaR"),
        })
        print("P4 legacy replay: OK (sidecar %s, sha256 %s)"
              % (sidecar["path"], sidecar["sha256"].upper()[:16]))
    elif legacy_col and target_complete and tgt[legacy_col].notna().all() and tgt["p_M4"].notna().all():
        p_old = tgt[legacy_col].to_numpy()
        p4 = tgt["p_M4"].to_numpy()
        met = metric_pack(tgt.y, p_old)
        ci = cluster_bootstrap_delta(tgt.y, tgt.cluster_id, p_old, p4, reps, seed)
        rows.append({
            "direction": TARGET_DIRECTION, "strategy": "legacy_M3M4", "K": 2,
            "pool": "M3+M4", "pool_type": "LEGACY", "contains_m5": False,
            "selection_rule": "frozen legacy M3/M4 router replay",
            "method": "legacy_M3M4_router",
            "evaluation_mode": "target_final_evaluation",
            "evaluation_split": "target", "tier": "target_legacy_replay",
            "official": False, "fixed_reference": "M4",
            "fixed_reference_basis": "legacy_contract",
            "n": met["n"], "AUROC": met["AUROC"], "AUPRC": met["AUPRC"],
            "Brier": met["Brier"], "EC_AUROC": np.nan,
            "EC_AUROC_status": "legacy router: EC-AUROC not defined in this module",
            "deltaR_vs_fixed": ci["delta"], "deltaR_ci_low": ci["ci_low"],
            "deltaR_ci_high": ci["ci_high"], "valid_bootstrap": ci["valid_reps"],
            "notes": "kept separate from new-pool deltaR; not a P4 primary result",
        })
    else:
        rows.append({
            "direction": TARGET_DIRECTION, "strategy": "legacy_M3M4", "K": 2,
            "pool": "M3+M4", "pool_type": "LEGACY", "contains_m5": False,
            "method": "legacy_M3M4_router",
            "evaluation_mode": "target_final_evaluation",
            "evaluation_split": "target", "tier": "target_legacy_replay",
            "official": False,
            "status": ("SKIPPED: p_legacy_router not present in the 14-column wide "
                       "table and no --legacy-sidecar was supplied"),
        })
    ctx["save"](rows, "P4_router_comparison.csv")
    ctx["save"](weight_rows, "P4_router_weights.csv")
    ctx["save"](fold_rows, "P4_crossfit_folds.csv")
    files = list(STAGE_FILES["p4"])
    if ctx["emit_in_sample"]:
        files.append("P4_in_sample_diagnostic_NOT_OFFICIAL.csv")
    return files


def selection_fingerprint(d, cfg):
    """Everything P2-P4 selection/fitting derives from source rows only (R4)."""
    src, _tgt = split_frames(d, TARGET_DIRECTION)
    out = {"source_auc": {m: auc(src.y, src[f"p_{m}"]) for m in MODELS}}
    for strategy in STRATEGIES:
        for k in (1, 2, 3, 5):
            sel = select_pool(src, strategy, k, cfg.get("max_auc_gap", 0.02))
            out["pool_%s_K%d" % (strategy, k)] = list(sel["pool"])
    for strategy in STRATEGIES:
        sel = select_pool(src, strategy, 2, cfg.get("max_auc_gap", 0.02))
        try:
            model = fit_router_model(src, sel["pool"], cfg)
            out["router_%s" % strategy] = {
                "pool": list(sel["pool"]),
                "coef": np.asarray(model[-1].coef_, dtype=float).round(12).tolist(),
                "intercept": np.asarray(model[-1].intercept_, dtype=float).round(12).tolist(),
                "classes": np.asarray(model[-1].classes_, dtype=int).tolist(),
            }
        except RouterFitError as exc:
            out["router_%s" % strategy] = {"status": str(exc)}
    return out


def _fingerprint_json(fp):
    return json.dumps(fp, sort_keys=True, ensure_ascii=True)


def assert_split_isolation(d, cfg):
    """R4 self-check: permuting target rows or poisoning target labels must not move
    any pool selection or router coefficient."""
    require_deps()
    base = selection_fingerprint(d, cfg)
    tmask = d["split"] == TARGET_SPLIT
    target_block = d.loc[tmask]
    shuffled = target_block.sample(frac=1.0, random_state=SPLIT_ISOLATION_SEED)
    shuffled = shuffled.set_axis(target_block.index)
    d_perm = pd.concat([d.loc[~tmask], shuffled]).sort_index()
    perm_fp = selection_fingerprint(d_perm, cfg)
    d_poison = d.copy()
    d_poison.loc[tmask, "y"] = 1 - d_poison.loc[tmask, "y"]
    poison_fp = selection_fingerprint(d_poison, cfg)
    result = {
        "target_rows": int(tmask.sum()),
        "permutation_invariant": _fingerprint_json(base) == _fingerprint_json(perm_fp),
        "label_poisoning_invariant": _fingerprint_json(base) == _fingerprint_json(poison_fp),
        "baseline": base,
        "permuted": perm_fp,
        "poisoned": poison_fp,
    }
    result["status"] = ("PASS" if result["permutation_invariant"]
                        and result["label_poisoning_invariant"] else "FAIL")
    return result


def validate_config_frozen(cfg):
    """Reject configuration that would undo the frozen design (R2, R3, R5, R6)."""
    if int(cfg.get("primary_k", 2)) != 2:
        print("ERROR: primary_k is frozen at 2 (P2 main comparison).")
        raise SystemExit(EXIT_USAGE)
    sizes = sorted(set(int(x) for x in cfg.get("pool_sizes", [1, 2, 3, 5])))
    if sizes != [1, 2, 3, 5]:
        print("ERROR: pool_sizes is frozen at [1, 2, 3, 5]; got %s" % sizes)
        raise SystemExit(EXIT_USAGE)
    if int(cfg.get("bootstrap_reps", 2000)) < 1:
        print("ERROR: bootstrap_reps must be >= 1.")
        raise SystemExit(EXIT_USAGE)
    assert_router_frozen(cfg)
    for key in ("p1_directions", "p234_directions"):
        if key in cfg:
            expected = list(P1_DIRECTIONS if key == "p1_directions" else P234_DIRECTIONS)
            if list(cfg[key]) != expected:
                print("ERROR: %s is frozen at %s (R2)." % (key, expected))
                raise SystemExit(EXIT_USAGE)
    if "directions" in cfg:
        print("CONFIG NOTE: legacy key 'directions' is ignored; P1=%s, P2/P3/P4=%s (R2)."
              % (list(P1_DIRECTIONS), list(P234_DIRECTIONS)))


def stage_scope_counts(d):
    counts = {}
    for stage in STAGE_ORDER:
        dirs = P1_DIRECTIONS if stage == "p1" else P234_DIRECTIONS
        sub = d[d["direction"].isin(dirs)]
        if stage == "p1":
            sub = sub[(sub["split"] == SOURCE_SPLIT)
                      | ((sub["split"] == TARGET_SPLIT)
                         & (sub["direction"] == TARGET_DIRECTION))]
        detail = []
        for split in (SOURCE_SPLIT, TARGET_SPLIT):
            for direction in dirs:
                n = int(((sub["split"] == split) & (sub["direction"] == direction)).sum())
                if n:
                    detail.append("%s/%s=%d" % (split, direction, n))
        counts[stage] = {"rows": int(len(sub)), "detail": detail}
    return counts


def marker_path(outdir, stage):
    return Path(outdir) / AUDIT_SUBDIR / ("%s_AUDIT_PASS.json" % stage.upper())


def write_stage_marker(ctx, stage, files, rows_used):
    path = marker_path(ctx["outdir"], stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage.upper(),
        "status": "PASS",
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(ctx["input_path"]),
        "input_sha256": ctx["input_fp"]["sha256"],
        "input_bytes": ctx["input_fp"]["bytes"],
        "input_sha_frozen_match": bool(ctx["sha_verdict"]["sha_match"]),
        "config_sha256": ctx["config_sha256"],
        "code_sha256": ctx["code_sha256"],
        "rows_used": int(rows_used),
        "outputs": list(files),
        "primary_decision_metric": PRIMARY_DECISION_METRIC,
        "official_eval_modes": list(OFFICIAL_EVAL_MODES),
        "gpu_used": False,
        "model_retraining": False,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    ctx["written"].append(str(path))


def require_stage_prereqs(stage, ctx):
    """R3: p2 needs p1 PASS; p3 needs p1+p2; p4 needs p1+p2+p3."""
    idx = STAGE_ORDER.index(stage)
    problems = []
    for prev in STAGE_ORDER[:idx]:
        p = marker_path(ctx["outdir"], prev)
        if not p.exists():
            problems.append("%s marker missing (%s)" % (prev.upper(), p))
            continue
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append("%s marker unreadable: %s" % (prev.upper(), exc))
            continue
        if m.get("status") != "PASS":
            problems.append("%s marker status=%s" % (prev.upper(), m.get("status")))
        elif m.get("input_sha256") != ctx["input_fp"]["sha256"]:
            problems.append("%s marker was written for a different input SHA" % prev.upper())
        elif m.get("config_sha256") != ctx["config_sha256"]:
            problems.append("%s marker was written for a different config" % prev.upper())
        elif m.get("code_sha256") != ctx["code_sha256"]:
            problems.append("%s marker was written by different code" % prev.upper())
    if problems:
        print("STAGE PREREQUISITE CHECK (R3): FAIL")
        for p in problems:
            print("  - %s" % p)
        print("  Refusing to run --stage %s (exit %d). Run the earlier stages first."
              % (stage, EXIT_STAGE))
        raise SystemExit(EXIT_STAGE)
    print("STAGE PREREQUISITE CHECK (R3): PASS for --stage %s" % stage)


def run_stage(stage, d, cfg, ctx):
    if stage == "p1":
        files = run_p1(d, cfg, ctx)
    elif stage == "p2":
        files = run_p2(d, cfg, ctx)
    elif stage == "p3":
        files = run_p3(d, cfg, ctx)
    elif stage == "p4":
        files = run_p4(d, cfg, ctx)
    else:
        raise SystemExit(EXIT_USAGE)
    write_stage_marker(ctx, stage, files, ctx["stage_counts"][stage]["rows"])
    print("STAGE %s: PASS (%d output file(s))" % (stage.upper(), len(files)))
    return files


def build_assertions(d, ctx, args, iso):
    a = []
    sv = ctx["sha_verdict"]
    raw = ctx["raw_columns"]
    a.append({"id": "A01", "status": "PASS" if sv["sha_match"] else "FAIL",
              "detail": "frozen wide-table SHA256 matches EF05...864" if sv["sha_match"]
              else "SHA256 mismatch (%s); a real run exits 3" % sv["actual_sha256"][:16]})
    a.append({"id": "A02", "status": "PASS" if sv["bytes_match"] else "FAIL",
              "detail": "size bytes == %d" % FROZEN_INPUT_BYTES})
    n_rows = int(len(d))
    a.append({"id": "A03", "status": "PASS" if n_rows == FROZEN_INPUT_ROWS else "FAIL",
              "detail": "rows after aggregation == %d (got %d)" % (FROZEN_INPUT_ROWS, n_rows)})
    n_source = int((d["split"] == SOURCE_SPLIT).sum())
    a.append({"id": "A04", "status": "PASS" if n_source == FROZEN_SOURCE_ROWS else "FAIL",
              "detail": "source rows == %d (got %d)" % (FROZEN_SOURCE_ROWS, n_source)})
    n_target = int((d["split"] == TARGET_SPLIT).sum())
    dir_a_target = int(((d["split"] == TARGET_SPLIT)
                        & (d["direction"] == FORBIDDEN_TARGET_DIRECTION)).sum())
    ok_a05 = (n_target == FROZEN_TARGET_ROWS and dir_a_target == 0
              and set(d.loc[d["split"] == TARGET_SPLIT, "direction"].unique()) == {TARGET_DIRECTION})
    a.append({"id": "A05", "status": "PASS" if ok_a05 else "FAIL",
              "detail": "target rows == %d, all DIR-B, DIR-A target == 0 (got %d / %d)"
                        % (FROZEN_TARGET_ROWS, n_target, dir_a_target)})
    a.append({"id": "A06", "status": "PASS" if set(EXPECTED_COLUMNS).issubset(set(raw)) else "FAIL",
              "detail": "14-column schema present (%d columns read)" % len(raw)})
    a.append({"id": "A07", "status": "PASS" if not d["y"].isna().any()
              and set(d["y"].unique()).issubset({0, 1}) else "FAIL",
              "detail": "y is binary and complete"})
    src_probs_ok = not d.loc[d["split"] == SOURCE_SPLIT, PROB_COLS].isna().any().any()
    tgt_probs_ok = not d.loc[d["split"] == TARGET_SPLIT, PROB_COLS].isna().any().any()
    a.append({"id": "A08", "status": "PASS" if src_probs_ok else "FAIL",
              "detail": "source p_M1..p_M5 complete; target complete=%s" % tgt_probs_ok})
    a.append({"id": "A09", "status": "PASS"
              if not d.duplicated(["direction", "split", "unit_id"]).any() else "FAIL",
              "detail": "one row per (direction, split, unit_id)"})
    fold_ok = (set(d.loc[d["split"] == SOURCE_SPLIT, "fold"].unique()).issubset(set(FOLD_VALUES))
               and bool((d.loc[d["split"] == TARGET_SPLIT, "fold"] == TARGET_FOLD_SENTINEL).all()))
    a.append({"id": "A10", "status": "PASS" if fold_ok else "FAIL",
              "detail": "source fold in 1..5; target fold blank (sentinel %d)"
                        % TARGET_FOLD_SENTINEL})
    fold_problems = []
    for (direction, fold), expected in EXPECTED_SOURCE_FOLD_ROWS.items():
        got = int(((d["direction"] == direction) & (d["split"] == SOURCE_SPLIT)
                   & (d["fold"] == fold)).sum())
        if got != expected:
            fold_problems.append("%s f%d %d!=%d" % (direction, fold, got, expected))
    a.append({"id": "A11", "status": "PASS" if not fold_problems else "FAIL",
              "detail": "source fold counts match 293/254/269/284/285 and 74/71/74/72/69"
                        + ("" if not fold_problems else " problems=%s" % fold_problems)})
    m5_cell_problems = []
    for direction, fold in sorted(M5_TAU05_DEGENERATE_CELLS):
        f = d[(d["direction"] == direction) & (d["split"] == SOURCE_SPLIT)
              & (d["fold"] == fold)]
        if not observed_one_sided(f["p_M5"]):
            m5_cell_problems.append("%s f%d" % (direction, fold))
    a.append({"id": "A12", "status": "PASS" if not m5_cell_problems else "WARN",
              "detail": "M5 tau=0.5 degenerate cells DIR-A f2/f5, DIR-B f2 observed"
                        + ("" if not m5_cell_problems
                           else "; not one-sided: %s (ruling kept)" % m5_cell_problems)})
    a.append({"id": "A13", "status": "PASS" if ctx["fingerprint_unchanged"] else "FAIL",
              "detail": "input fingerprint unchanged after read (read-only)"})
    input_resolved = str(Path(ctx["input_path"]).resolve()).lower()
    out_collisions = [f for f in sum(STAGE_FILES.values(), []) + COMMON_FILES
                      if str((Path(ctx["outdir"]) / f).resolve()).lower() == input_resolved]
    a.append({"id": "A14", "status": "PASS" if not out_collisions else "FAIL",
              "detail": "no output path equals the input path; input never opened for write"})
    a.append({"id": "A15", "status": "PASS" if P234_DIRECTIONS == ("DIR-B",) else "FAIL",
              "detail": "P2/P3/P4 directions == ('DIR-B',)"})
    both = set(d.loc[d["split"] == SOURCE_SPLIT, "direction"].unique()) == set(P1_DIRECTIONS)
    a.append({"id": "A16", "status": "PASS" if both else "FAIL",
              "detail": "P1 covers DIR-A and DIR-B source rows"})
    a.append({"id": "A17", "status": "ENFORCED",
              "detail": "select_pool/fit_router_model/cross_fit_router assert source-only frames"})
    if iso is None:
        a.append({"id": "A18", "status": "PLANNED",
                  "detail": "target-row permutation leaves selection/fitting unchanged "
                            "(run with --assert-split-isolation)"})
        a.append({"id": "A19", "status": "PLANNED",
                  "detail": "target-label poisoning leaves selection/fitting unchanged "
                            "(run with --assert-split-isolation)"})
    else:
        a.append({"id": "A18", "status": "PASS" if iso["permutation_invariant"] else "FAIL",
                  "detail": "target-row permutation invariant (%d target rows)" % iso["target_rows"]})
        a.append({"id": "A19", "status": "PASS" if iso["label_poisoning_invariant"] else "FAIL",
                  "detail": "target-label poisoning invariant"})
    a.append({"id": "A20", "status": "ENFORCED",
              "detail": "official evaluation modes limited to %s" % (list(OFFICIAL_EVAL_MODES),)})
    a.append({"id": "A21", "status": "ENFORCED",
              "detail": "in-sample OOF results are DIAGNOSTIC_NOT_OFFICIAL and disabled by default"})
    a.append({"id": "A22", "status": "ENFORCED",
              "detail": "primary pools K<=4 draw from M1-M4 only; M5 only in K=5 static pool"})
    a.append({"id": "A23", "status": "ENFORCED",
              "detail": "K=5 static pool = M1..M5, supplementary tier"})
    a.append({"id": "A24", "status": "PASS",
              "detail": "router constants frozen (C=%.1f, max_iter=%d, %s)"
                        % (ROUTER_C, ROUTER_MAX_ITER, "+".join(ROUTER_FEATURE_SET))})
    stage = args.stage
    if stage == "all":
        a.append({"id": "A25", "status": "PLANNED",
                  "detail": "--stage all runs P1->P2->P3->P4 in order and writes markers"})
    else:
        idx = STAGE_ORDER.index(stage)
        missing = [prev.upper() for prev in STAGE_ORDER[:idx]
                   if not marker_path(ctx["outdir"], prev).exists()]
        a.append({"id": "A25", "status": "PASS" if not missing else "BLOCKED",
                  "detail": "prerequisites for --stage %s: %s"
                            % (stage, "all PASS markers present" if not missing
                               else "missing %s (exit 4)" % missing)})
    gpu_mods = [m for m in ("torch", "tensorflow", "jax", "cupy") if m in sys.modules]
    a.append({"id": "A26", "status": "PASS" if not gpu_mods else "FAIL",
              "detail": "no GPU/training modules imported (%s)" % (gpu_mods or "none")})
    a.append({"id": "A27", "status": "PASS",
              "detail": "deltaR definition unchanged: AUROC(candidate)-AUROC(fixed reference)"})
    a.append({"id": "A28", "status": "N/A", "detail": FROZEN_CRITERIA_NOTE})
    return a


def print_dry_run(d, ctx, args, iso):
    print("")
    print("=" * 78)
    print("DRY RUN (R9) - no output file is written")
    print("=" * 78)
    print("script version   : %s" % VERSION)
    print("code SHA256      : %s" % ctx["code_sha256"])
    print("config           : %s" % ctx["config_path"])
    print("config SHA256    : %s" % ctx["config_sha256"])
    print("input            : %s" % ctx["input_path"])
    print("input read-only  : YES (never opened for write; fingerprint re-checked)")
    print("utc              : %s" % datetime.now(timezone.utc).isoformat())
    print("")
    print("[1] FROZEN INPUT (R1)")
    sv = ctx["sha_verdict"]
    print("  expected SHA256 : %s" % sv["expected_sha256"])
    print("  actual   SHA256 : %s" % sv["actual_sha256"])
    print("  expected bytes  : %d" % sv["expected_bytes"])
    print("  actual   bytes  : %d" % sv["actual_bytes"])
    print("  SHA verdict     : %s" % ("PASS" if sv["sha_match"] else "FAIL"))
    print("  bytes verdict   : %s" % ("PASS" if sv["bytes_match"] else "FAIL"))
    if not sv["sha_match"]:
        print("  input trust     : UNFROZEN / SYNTHETIC (dry-run only; real run exits 3)")
        print("  REAL_RUN_READY  : NO")
    else:
        print("  REAL_RUN_READY  : YES")
    print("")
    print("[2] SHAPE")
    print("  rows (aggregated) : %d" % len(d))
    print("  rows (raw)        : %d" % ctx["raw_rows"])
    print("  source / target   : %d / %d"
          % (int((d["split"] == SOURCE_SPLIT).sum()), int((d["split"] == TARGET_SPLIT).sum())))
    print("  DIR-A / DIR-B     : %d / %d"
          % (int((d["direction"] == "DIR-A").sum()), int((d["direction"] == "DIR-B").sum())))
    for split in (SOURCE_SPLIT, TARGET_SPLIT):
        for direction in ("DIR-A", "DIR-B"):
            n = int(((d["split"] == split) & (d["direction"] == direction)).sum())
            if n:
                print("    %-6s %-5s : %d" % (split, direction, n))
    print("  columns (%d)      : %s" % (len(ctx["raw_columns"]), ", ".join(ctx["raw_columns"])))
    print("  source folds      : DIR-A 293/254/269/284/285; DIR-B 74/71/74/72/69")
    print("  target fold       : blank by contract (5-fold mean; not OOF)")
    print("  DIR-A target rows : 0 (PROHIBITED, hard refusal exit 6)")
    print("")
    print("[3] STAGE SCOPE (rows x columns actually read)")
    for stage in STAGE_ORDER:
        info = ctx["stage_counts"][stage]
        print("  %s : %d rows  [%s]" % (stage.upper(), info["rows"], ", ".join(info["detail"])))
        print("       columns: %s" % ", ".join(ANALYSIS_COLUMNS))
    print("")
    print("[4] OUTPUT FILES (would be written by a real run)")
    for stage in STAGE_ORDER:
        print("  %s : %s" % (stage.upper(), ", ".join(STAGE_FILES[stage])))
    print("  COMMON : RUN_METADATA.json, %s"
          % ", ".join("%s_AUDIT_PASS.json" % s.upper() for s in STAGE_ORDER))
    print("  marker dir: %s" % str(Path(ctx["outdir"]) / AUDIT_SUBDIR))
    print("  in-sample diagnostic: %s"
          % ("ENABLED -> P4_in_sample_diagnostic_NOT_OFFICIAL.csv"
             if ctx["emit_in_sample"] else "DISABLED (DIAGNOSTIC_NOT_OFFICIAL only)"))
    print("")
    print("[5] ASSERTIONS (R1-R8)")
    for item in build_assertions(d, ctx, args, iso):
        print("  %-4s %-9s %s" % (item["id"], item["status"], item["detail"]))
    print("")
    if iso is not None:
        print("[6] SPLIT-ISOLATION SELF-CHECK (R4)")
        print("  status                : %s" % iso["status"])
        print("  target rows permuted  : %d" % iso["target_rows"])
        print("  permutation invariant : %s" % iso["permutation_invariant"])
        print("  label poisoning       : %s" % iso["label_poisoning_invariant"])
        print("")
    print("[7] RESULT")
    print("  dry_run         : YES")
    print("  outputs written : 0")
    print("  stage plan      : %s"
          % ("P1 -> P2 -> P3 -> P4" if args.stage == "all" else args.stage.upper()))
    print("  exit code       : 0")
    print("=" * 78)


def write_metadata(ctx, stages_executed):
    metadata = {
        "package_version": VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__ if scipy is not None else None,
        "sklearn": sklearn.__version__ if sklearn is not None else None,
        "input_csv": str(ctx["input_path"]),
        "input_sha256": ctx["input_fp"]["sha256"],
        "input_bytes": ctx["input_fp"]["bytes"],
        "frozen_input_sha256": FROZEN_INPUT_SHA256,
        "input_sha_frozen_match": bool(ctx["sha_verdict"]["sha_match"]),
        "input_read_only": True,
        "input_rows_raw": int(ctx["raw_rows"]),
        "input_rows_after_seed_aggregation": int(len(ctx["data"])),
        "stages_executed": list(stages_executed),
        "legacy_router_column": ctx["legacy_col"],
        "bootstrap_reps": ctx["reps"],
        "bootstrap_seed": ctx["seed"],
        "max_auc_gap_source_only": ctx["gap"],
        "primary_k": ctx["primary_k"],
        "pool_sizes": list(ctx["pool_sizes"]),
        "p1_directions": list(P1_DIRECTIONS),
        "p234_directions": list(P234_DIRECTIONS),
        "target_direction": TARGET_DIRECTION,
        "dir_a_target_allowed": False,
        "primary_models": list(PRIMARY_MODELS),
        "m5_primary_routing": False,
        "m5_allowed_in": "K=5 static pool and rank-type supplementary results only",
        "m5_tau05_degenerate_cells": ["%s:f%d" % (d, f)
                                      for (d, f) in sorted(M5_TAU05_DEGENERATE_CELLS)],
        "primary_decision_metric": PRIMARY_DECISION_METRIC,
        "p4_evaluation": "cross_fitted_oof + target_after_full_source_fit",
        "p4_in_sample_official": False,
        "p4_in_sample_diagnostic_emitted": bool(ctx["emit_in_sample"]),
        "router_constants": {
            "features": list(ROUTER_FEATURE_SET),
            "C": ROUTER_C,
            "max_iter": ROUTER_MAX_ITER,
            "solver": ROUTER_SOLVER,
            "penalty": ROUTER_PENALTY,
            "class_weight": "none",
            "random_state": ROUTER_RANDOM_STATE,
            "refit_on_all_source": ROUTER_REFIT_ON_ALL_SOURCE,
        },
        "pool_selection_uses_target_labels": False,
        "router_training_uses_target_labels": False,
        "target_labels_used_only_for_final_evaluation_and_posthoc_diagnostics": True,
        "e1_e8_modified": False,
        "go_no_go_thresholds_modified": False,
        "deltaR_definition": "AUROC(candidate) - AUROC(fixed reference)",
        "legacy_sidecar": ctx.get("legacy_sidecar_meta"),
        "legacy_sidecar_evaluation_mode": "legacy_replay_sidecar",
        "gpu_used": False,
        "model_retraining": False,
        "split_isolation": ctx["isolation_status"],
        "assertions": {item["id"]: item["status"]
                       for item in build_assertions(ctx["data"], ctx, ctx["args"],
                                                    ctx["isolation_result"])},
    }
    path = Path(ctx["outdir"]) / "RUN_METADATA.json"
    path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    ctx["written"].append(str(path))


def main():
    apx = argparse.ArgumentParser(
        description="TRUST-LUS next-four-experiments runner (CPU post-processing only).")
    apx.add_argument("--config", required=True)
    apx.add_argument("--stage", default="all", choices=["p1", "p2", "p3", "p4", "all"])
    apx.add_argument("--dry-run", action="store_true",
                     help="print the full plan and assertions; write nothing")
    apx.add_argument("--assert-split-isolation", action="store_true",
                     help="verify that target rows cannot influence selection or fitting")
    apx.add_argument("--emit-in-sample-diagnostic", action="store_true",
                     help="write the NOT_OFFICIAL in-sample diagnostic (default off)")
    apx.add_argument("--legacy-sidecar", default=None, metavar="PATH",
                     help=("optional legacy M3/M4 router sidecar (unit_id, p_route, "
                           "cluster_id, y) built by scripts/112_legacy_router_sidecar.py; "
                           "when supplied, the P4 legacy_M3M4 row is replayed as "
                           "evaluation_mode=legacy_replay_sidecar, otherwise it stays "
                           "SKIPPED exactly as before"))
    args = apx.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print("ERROR: config not found: %s" % cfg_path)
        return EXIT_USAGE
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("ERROR: config is not valid JSON: %s" % exc)
        return EXIT_USAGE
    validate_config_frozen(cfg)

    input_path = (cfg_path.parent / cfg["input_csv"]).resolve() \
        if not Path(cfg["input_csv"]).is_absolute() else Path(cfg["input_csv"])
    if not input_path.exists():
        print("ERROR: input CSV not found: %s" % input_path)
        return EXIT_USAGE
    outdir = (cfg_path.parent / cfg.get("output_dir", "outputs")).resolve()

    fp_before = input_fingerprint(input_path)
    sha_verdict = verify_frozen_input(input_path, fp_before, args.dry_run)
    try:
        raw = pd.read_csv(input_path, encoding="utf-8-sig")
    except Exception as exc:
        print("ERROR: cannot read input CSV: %s" % exc)
        return EXIT_DATA
    fp_after = input_fingerprint(input_path)
    try:
        assert_input_unchanged(fp_before, fp_after)
        fingerprint_unchanged = True
    except DataError as exc:
        print("ERROR: %s" % exc)
        return EXIT_DATA
    try:
        d, legacy_col = validate_and_aggregate(raw, cfg)
    except DirATargetError as exc:
        print("REFUSED (R2): %s" % exc)
        return EXIT_DIRA_TARGET
    except DataError as exc:
        print("DATA ERROR: %s" % exc)
        return EXIT_DATA

    cfg_digest = sha256_text(json.dumps(cfg, sort_keys=True, ensure_ascii=True,
                                        default=str))
    ctx = {
        "input_path": input_path,
        "input_fp": fp_before,
        "sha_verdict": sha_verdict,
        "fingerprint_unchanged": fingerprint_unchanged,
        "config_path": str(cfg_path),
        "config_sha256": cfg_digest,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "outdir": outdir,
        "reps": int(cfg.get("bootstrap_reps", 2000)),
        "seed": int(cfg.get("bootstrap_seed", 20261101)),
        "gap": float(cfg.get("max_auc_gap", 0.02)),
        "primary_k": int(cfg.get("primary_k", 2)),
        "pool_sizes": [int(x) for x in cfg.get("pool_sizes", [1, 2, 3, 5])],
        "legacy_col": legacy_col,
        "raw_columns": list(raw.columns),
        "raw_rows": int(len(raw)),
        "data": d,
        "dry_run": bool(args.dry_run),
        "emit_in_sample": bool(args.emit_in_sample_diagnostic),
        "written": [],
        "stage_counts": stage_scope_counts(d),
        "args": args,
        "isolation_result": None,
        "isolation_status": "NOT_RUN",
        "save": None,
    }

    iso = None
    if args.assert_split_isolation:
        iso = assert_split_isolation(d, cfg)
        ctx["isolation_result"] = iso
        ctx["isolation_status"] = iso["status"]
        if iso["status"] != "PASS":
            print("SPLIT ISOLATION (R4): FAIL")
            print("  target rows can influence selection or fitting; refusing (exit %d)."
                  % EXIT_ISOLATION)
            return EXIT_ISOLATION
        if not args.dry_run:
            outdir.mkdir(parents=True, exist_ok=True)
            path = Path(outdir) / AUDIT_SUBDIR / "SPLIT_ISOLATION.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(iso, ensure_ascii=True, indent=2), encoding="utf-8")
            print("SPLIT ISOLATION (R4): PASS; audit written to %s" % path)

    if args.dry_run:
        print_dry_run(d, ctx, args, iso)
        return EXIT_OK

    require_deps()
    stages = list(STAGE_ORDER) if args.stage == "all" else [args.stage]
    if args.stage != "all":
        require_stage_prereqs(args.stage, ctx)
    outdir.mkdir(parents=True, exist_ok=True)

    def save_rows(rows, name):
        path = (Path(outdir) / name).resolve()
        if path == Path(input_path).resolve():
            raise SystemExit(EXIT_USAGE)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        ctx["written"].append(str(path))
        return name
    ctx["save"] = save_rows

    executed = []
    for stage in stages:
        run_stage(stage, d, cfg, ctx)
        executed.append(stage)
    write_metadata(ctx, executed)
    print("Completed TRUST-LUS next-four experiment post-processing (v%s)." % VERSION)
    print("Output directory: %s" % outdir)
    print("P1: model metrics, by-fold metrics, pairwise complementarity, oracle, M5 degeneracy")
    print("P2: K=2 candidate-pool strategy comparison (DIR-B)")
    print("P3: K=1/2/3/5 pool-size comparison (DIR-B)")
    print("P4: cross-fitted router + target router + K=5 supplementary + legacy replay")
    print("No model was retrained; no GPU was used; input was read-only.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
