# -*- coding: utf-8 -*-
"""Shared-cluster sensitivity: exclude Vave clips whose cluster_id_v3 also occurs
in the Butterfly development cohort, then compare M4 / equal / soft / hard.

Post hoc. Frozen expert probabilities are reused unchanged; only the evaluation
subset changes. Paired cluster bootstrap per cohort (seed 20261101, 2,000 valid).
"""
import csv, io, os, json, hashlib, itertools
import numpy as np

W = r"C:\Users\99243\Documents\deepseek-harness\default-workspace\stage3"
OUT = os.path.join(W, r"report\v010_2026-11-26")
TAB = os.path.join(OUT, "tables"); VER = os.path.join(OUT, "verify")
os.makedirs(TAB, exist_ok=True); os.makedirs(VER, exist_ok=True)

UNITS = os.path.join(W, r"stageB_data_v3p1\units_v3p1.csv")
FOLD = os.path.join(W, r"stageB_data_v3p1\fold_assignment_v3p1.csv")
MANIF = r"E:\TRUST_LUS\data\manifests\openpocus.csv"
ROUTE = os.path.join(W, r"report\p4_source360_2026-11-26\tables\Vave374_soft_hard_rebuilt.csv")
VAVE = os.path.join(W, r"report\oof_vave_package_2026-11-26\tables\Vave_target_pM1_pM5_reconstructed.csv")

SEED = 20261101
NREP = 2000
PCTL = (2.5, 97.5)
FROZEN = {"M4": 0.9629390495867769, "soft": 0.9681043388429752,
          "hard": 0.9668130165289257, "equal_delta": -0.004390}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def load(p):
    with io.open(p, encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))

def wr(p, cols, rows):
    with io.open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

def auroc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    o = np.argsort(s, kind="mergesort"); ss = s[o]; yy = y[o]
    n = len(s); r = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    npos = int((yy == 1).sum()); nneg = n - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return (r[yy == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

def average_precision(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]; s = s[order]
    npos = int((y == 1).sum())
    if npos == 0:
        return np.nan
    tp = fp = 0; ap = 0.0; prev = 0.0; i = 0; n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        for k in range(i, j + 1):
            if y[k] == 1: tp += 1
            else: fp += 1
        rec = tp / npos; prec = tp / (tp + fp) if (tp + fp) else 0.0
        ap += (rec - prev) * prec; prev = rec
        i = j + 1
    return ap

def brier(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    return float(np.mean((s - y) ** 2))

# ---------- 1. device × cluster inventory ----------
units = load(UNITS)
folds = {r["unit_id"]: r for r in load(FOLD)}
manif = load(MANIF)
dev_by_clip = {r["clip_id"]: r.get("device_code", "") for r in manif}

op = [r for r in units if r["library"] == "openpocus"]
print("openpocus units:", len(op))
dev = {}
for r in op:
    u = r["unit_id"]
    fg = folds.get(u, {}).get("device_group", "")
    dc = dev_by_clip.get(u, "")
    dev[u] = (fg, dc)
from collections import Counter
print("device_group counts:", Counter(v[0] for v in dev.values()))
print("device_code counts:", Counter(v[1] for v in dev.values()))

clu_by_dev = {"butterfly": {}, "vave": {}}
for r in op:
    u = r["unit_id"]; fg = dev[u][0]
    c = r["cluster_id"]
    key = "butterfly" if "butter" in fg.lower() else ("vave" if "vave" in fg.lower() else "other")
    if key == "other":
        continue
    clu_by_dev[key].setdefault(c, []).append(u)

for k in clu_by_dev:
    print("%s: units=%d clusters=%d" % (k, sum(len(v) for v in clu_by_dev[k].values()), len(clu_by_dev[k])))
shared = sorted(set(clu_by_dev["butterfly"]) & set(clu_by_dev["vave"]))
print("shared clusters:", len(shared), shared)

# ---------- 2. Vave table + shared-cluster exclusion ----------
vave_rows = {r["unit_id"]: r for r in load(VAVE)}
route_rows = {r["unit_id"]: r for r in load(ROUTE)}
assert set(vave_rows) == set(route_rows)
clu_of = {r["unit_id"]: r["cluster_id"] for r in load(ROUTE)}
y_of = {r["unit_id"]: int(float(r["y"])) for r in load(ROUTE)}

vave_units = sorted(route_rows)
print("vave units:", len(vave_units), "clusters:", len(set(clu_of[u] for u in vave_units)))

shared_units = [u for u in vave_units if clu_of[u] in shared]
kept_units = [u for u in vave_units if clu_of[u] not in shared]
def stats(us):
    pos = sum(y_of[u] for u in us)
    return len(us), len(set(clu_of[u] for u in us)), pos, len(us) - pos, \
        len(set(clu_of[u] for u in us if y_of[u] == 1))
print("shared-cluster Vave units: n=%d clusters=%d pos=%d neg=%d posclusters=%d" % ((len(shared_units),) + stats(shared_units)[1:]))
print("kept (350)                : n=%d clusters=%d pos=%d neg=%d posclusters=%d" % stats(kept_units))
print("full (374)                : n=%d clusters=%d pos=%d neg=%d posclusters=%d" % stats(vave_units))
# Butterfly positive clusters
bfr_clusters = clu_by_dev["butterfly"]
bfr_y = {u: int(float(vave_rows.get(u, {}).get("y", 0))) for u in []}  # placeholder
# Butterfly labels from units_v3p1.label_bline
ub = {r["unit_id"]: r for r in op}
bfr_units = [u for c in bfr_clusters for u in bfr_clusters[c]]
bfr_pos_clusters = len(set(ub[u]["cluster_id"] for u in bfr_units if str(ub[u]["label_bline"]).strip() == "1"))
print("butterfly positive clusters: %d of %d" % (bfr_pos_clusters, len(bfr_clusters)))

# ---------- 3. four arms ----------
def arms(us):
    y = np.array([y_of[u] for u in us], float)
    p4 = np.array([float(route_rows[u]["p_M4"]) for u in us])
    p1 = np.array([float(route_rows[u]["p_M1"]) for u in us])
    eq = (p4 + p1) / 2.0
    soft = np.array([float(route_rows[u]["p_soft"]) for u in us])
    hard = np.array([float(route_rows[u]["p_hard"]) for u in us])
    return y, {"M4": p4, "equal": eq, "soft": soft, "hard": hard}, p4, p1

def summarise(us, tag):
    y, a, p4, p1 = arms(us)
    r = {}
    for k, v in a.items():
        r[k] = {"auroc": auroc(y, v), "auprc": average_precision(y, v), "brier": brier(y, v)}
    print("\n[%s] n=%d pos=%d clusters=%d" % (tag, len(us), int(y.sum()), len(set(clu_of[u] for u in us))))
    for k in ("M4", "equal", "soft", "hard"):
        print("   %-6s AUROC=%.10f AUPRC=%.6f Brier=%.6f" % (k, r[k]["auroc"], r[k]["auprc"], r[k]["brier"]))
    for k in ("equal", "soft", "hard"):
        print("   Δ(%s − M4) = %+.10f" % (k, r[k]["auroc"] - r["M4"]["auroc"]))
    return r, a, y

r374, a374, y374 = summarise(vave_units, "FULL 374")
r350, a350, y350 = summarise(kept_units, "EXCL-SHARED 350")

print("\nfrozen check: M4 %.10f (d=%.1e) soft %.10f (d=%.1e) hard %.10f (d=%.1e)" % (
    r374["M4"]["auroc"], abs(r374["M4"]["auroc"] - FROZEN["M4"]),
    r374["soft"]["auroc"], abs(r374["soft"]["auroc"] - FROZEN["soft"]),
    r374["hard"]["auroc"], abs(r374["hard"]["auroc"] - FROZEN["hard"])))

# ---------- 4. paired cluster bootstrap ----------
def boot(us, tag):
    y, a, _, _ = arms(us)
    cids = np.array([clu_of[u] for u in us])
    keys = np.array(sorted(set(cids)))
    kidx = {k: i for i, k in enumerate(keys)}
    ci = np.array([kidx[c] for c in cids])
    members = [[] for _ in keys]
    for i, k in enumerate(ci):
        members[k].append(i)
    rng = np.random.default_rng(SEED)
    reps = {k: [] for k in ("equal", "soft", "hard")}
    n_att = n_inv = 0
    while len(reps["soft"]) < NREP and n_att < 200000:
        n_att += 1
        draw = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([members[d] for d in draw])
        yb = y[idx]
        if yb.min() == yb.max():
            n_inv += 1
            continue
        base = auroc(yb, a["M4"][idx])
        for k in reps:
            reps[k].append(auroc(yb, a[k][idx]) - base)
    out = {}
    for k, v in reps.items():
        out[k] = {"point": float(r374[k]["auroc"] - r374["M4"]["auroc"]) if tag == "374" else float(r350[k]["auroc"] - r350["M4"]["auroc"]),
                  "mean": float(np.mean(v)),
                  "ci": [float(np.percentile(v, PCTL[0])), float(np.percentile(v, PCTL[1]))],
                  "n_valid": len(v)}
    print("\n[%s] bootstrap attempted=%d valid=%d invalid=%d" % (tag, n_att, len(reps["soft"]), n_inv))
    for k in ("equal", "soft", "hard"):
        print("   %-6s Δ=%+.10f  CI [%+.6f, %+.6f]" % (k, out[k]["point"], out[k]["ci"][0], out[k]["ci"][1]))
    return out, n_att, len(reps["soft"]), n_inv, len(keys)

b374, att374, val374, inv374, cl374 = boot(vave_units, "374")
b350, att350, val350, inv350, cl350 = boot(kept_units, "350")

# ---------- 5. write Tables ----------
def rows_for(us, tag, r, b):
    out = []
    for k in ("M4", "equal", "soft", "hard"):
        d = "reference" if k == "M4" else "%+.10f" % b[k]["point"]
        ci = "—" if k == "M4" else "%+.10f to %+.10f" % tuple(b[k]["ci"])
        out.append({"cohort": tag, "method": k, "n_clips": len(us),
                    "n_clusters": len(set(clu_of[u] for u in us)),
                    "n_pos": sum(y_of[u] for u in us), "n_neg": len(us) - sum(y_of[u] for u in us),
                    "auroc": "%.10f" % r[k]["auroc"], "auprc": "%.6f" % r[k]["auprc"],
                    "brier": "%.6f" % r[k]["brier"], "delta_vs_M4": d, "delta_ci95": ci})
    return out

s6 = rows_for(vave_units, "full_374", r374, b374) + rows_for(kept_units, "excl_shared_350", r350, b350)
wr(os.path.join(TAB, "Table_S6_shared_cluster_sensitivity.csv"), list(s6[0].keys()), s6)

# exclusion map (internal only)
ex = []
for c in shared:
    vus = clu_by_dev["vave"].get(c, [])
    bus = clu_by_dev["butterfly"].get(c, [])
    for u in vus:
        ex.append({"cluster_id_v3": c, "unit_id": u, "device": "vave",
                   "y": y_of[u], "butterfly_units_in_same_cluster": len(bus),
                   "butterfly_units": ";".join(bus)})
wr(os.path.join(TAB, "exclusion_map_shared_clusters_INTERNAL.csv"),
   list(ex[0].keys()), ex)

meta = {
    "analysis": "post hoc sensitivity — exclude Vave clips in clusters shared with Butterfly",
    "seed": SEED, "reps_target": NREP, "pctl": list(PCTL),
    "bootstrap": {"unit": "cluster_id", "per_cohort": True,
                  "shared_draws_across_arms": True,
                  "rng": "np.random.default_rng(20261101)"},
    "cohorts": {
        "full_374": {"n": 374, "clusters": cl374, "attempted": att374, "valid": val374, "invalid": inv374},
        "excl_350": {"n": 350, "clusters": cl350, "attempted": att350, "valid": val350, "invalid": inv350},
    },
    "inventory": {"butterfly_units": sum(len(v) for v in clu_by_dev["butterfly"].values()),
                  "butterfly_clusters": len(clu_by_dev["butterfly"]),
                  "butterfly_positive_clusters": bfr_pos_clusters,
                  "vave_units": len(vave_units), "vave_clusters": len(set(clu_of.values())),
                  "shared_clusters": shared,
                  "shared_vave_units": len(shared_units),
                  "shared_vave_pos": sum(y_of[u] for u in shared_units)},
    "frozen_check": {"M4": r374["M4"]["auroc"], "soft": r374["soft"]["auroc"],
                     "hard": r374["hard"]["auroc"],
                     "abs_diff": [abs(r374["M4"]["auroc"] - FROZEN["M4"]),
                                  abs(r374["soft"]["auroc"] - FROZEN["soft"]),
                                  abs(r374["hard"]["auroc"] - FROZEN["hard"])]},
    "results": {"full_374": r374, "excl_350": r350},
    "boot": {"full_374": b374, "excl_350": b350},
    "inputs": {os.path.basename(p): {"bytes": os.path.getsize(p), "sha256": sha(p)}
               for p in (UNITS, FOLD, ROUTE, VAVE)},
}
with io.open(os.path.join(VER, "shared_cluster_sensitivity.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("\nwritten:", TAB)
