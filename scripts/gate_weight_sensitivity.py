# -*- coding: utf-8 -*-
"""TRUST_LUS gate weight sensitivity (S5). Pure numpy, exact Newton optimum.

Schemes on source-domain training rows only:
  A  w_i = q4_i / mean_train(q4)          (original)
  B  w_i = 1
  C  w_i = |loss3-loss4| / mean_train(|loss3-loss4|)

Model: z = 1(loss4<loss3) ; features [C3,C4,T3,T4,disagreement] ;
       x' = (x - median)/ (IQR + 1e-8) computed on the training rows ;
       L2 logistic: min 0.5*||w||^2 + C * sum_i sw_i * logloss, C=0.01
       (same minimiser as sklearn's C*sum + 0.5||w||^2)
       Newton with backtracking -> exact optimum, solver-independent.

Target: refit on all source rows, apply to C2 target features.
Bootstrap: cluster-level, 2000 valid, SeedSequence(20260927, spawn_key=(1,idx)).
"""
import csv, io, os, sys, json, math, hashlib, platform
import numpy as np

W = r"C:\Users\99243\Documents\deepseek-harness\default-workspace\stage3"
OUT = os.path.join(W, r"report\gate_weight_sensitivity_2026-11-26")
TAB = os.path.join(OUT, "tables"); VER = os.path.join(OUT, "verify")
os.makedirs(TAB, exist_ok=True); os.makedirs(VER, exist_ok=True)

SRC_A = os.path.join(W, r"report\oof_vave_package_2026-11-26\tables\OOF_source_OpenPOCUS_to_COVIDBLUES.csv")
SRC_B = os.path.join(W, r"report\oof_vave_package_2026-11-26\tables\OOF_source_COVIDBLUES_to_OpenPOCUS.csv")
TGT_A = os.path.join(W, r"report\supplementary_v04_2026-11-23\tables\C2_OpenPOCUS_to_COVIDBLUES.csv")
TGT_B = os.path.join(W, r"report\supplementary_v04_2026-11-23\tables\C2_COVIDBLUES_to_OpenPOCUS.csv")
GATE_JSON = os.path.join(W, r"report\server_artifacts\router\pilot_gate_report.json")

FEATS = ["C3", "C4", "T3", "T4", "disagreement"]
C_REG = 0.01
FOLDS = ["1", "2", "3", "4", "5"]
MASTER_SEED = 20260927
N_VALID = 2000
MAX_ATTEMPTS = 200000
PCTL = (2.5, 97.5)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def load(p):
    with io.open(p, encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))

def wr(path, cols, rows):
    with io.open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, "") for c in cols})

def fnum(s):
    return float(str(s).strip())

# ---------- features / scaling ----------
def med_iqr(X):
    med = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0); q3 = np.percentile(X, 75, axis=0)
    return med, (q3 - q1)

def transform(X, med, iqr):
    return (X - med) / (iqr + 1e-8)

# ---------- exact L2 logistic (Newton) ----------
def fit_gate(X, z, sw, C=C_REG, max_iter=200, tol=1e-12):
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])          # intercept as last column
    w = np.zeros(d + 1)
    def obj(w):
        s = A @ w
        # stable logloss
        ll = np.logaddexp(0.0, s) - z * s
        return 0.5 * np.dot(w[:d], w[:d]) + C * np.sum(sw * ll)
    f = obj(w)
    for it in range(max_iter):
        s = A @ w
        p = 1.0 / (1.0 + np.exp(-s))
        g = np.empty(d + 1)
        r = sw * (p - z)
        g[:d] = w[:d] + C * (X.T @ r)
        g[d] = C * np.sum(r)
        wq = sw * p * (1 - p)
        H = np.zeros((d + 1, d + 1))
        H[:d, :d] = np.eye(d) + C * (X.T * wq) @ X
        H[:d, d] = C * (X.T @ wq)
        H[d, :d] = H[:d, d]
        H[d, d] = C * np.sum(wq)
        step = np.linalg.solve(H, -g)
        t = 1.0
        while t > 1e-10:
            wn = w + t * step
            fn = obj(wn)
            if fn <= f + 1e-4 * t * np.dot(g, step):
                break
            t *= 0.5
        w = wn
        if abs(f - fn) < tol and np.max(np.abs(t * step)) < 1e-10:
            f = fn; break
        f = fn
    return w[:d], w[d], it + 1

def predict(X, w, b):
    s = X @ w + b
    return 1.0 / (1.0 + np.exp(-s))

# ---------- AUROC ----------
def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]; yy = y[order]
    n = len(s)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        ranks[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    npos = int((yy == 1).sum()); nneg = n - npos
    if npos == 0 or nneg == 0:
        return np.nan
    s1 = ranks[yy == 1].sum()
    return (s1 - npos * (npos + 1) / 2.0) / (npos * nneg)

log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

# ================= data =================
def prep_source(path, tag):
    rows = load(path)
    n = len(rows)
    X = np.array([[fnum(r[f]) for f in FEATS] for r in rows], dtype=float)
    z = np.array([1 if fnum(r["loss3"] if False else r["loss4"]) < fnum(r["loss3"]) else 0 for r in rows], dtype=float)
    # explicit (avoid confusion)
    z = np.array([1 if fnum(r["loss4"]) < fnum(r["loss3"]) else 0 for r in rows], dtype=float)
    l3 = np.array([fnum(r["loss3"]) for r in rows]); l4 = np.array([fnum(r["loss4"]) for r in rows])
    q4 = np.array([fnum(r["q4"]) for r in rows])
    g = np.abs(l3 - l4)
    fold = np.array([str(r["fold"]).strip() for r in rows])
    y = np.array([int(fnum(r["y"])) for r in rows])
    cid = np.array([str(r["cluster_id_v3"]).strip() for r in rows])
    uid = np.array([str(r["unit_id"]).strip() for r in rows])
    P("[%s] rows=%d  z=1:%d  ties=%d  q4 mean=%.6f  |l3-l4| mean=%.6f  folds=%s" % (
        tag, n, int(z.sum()), int((l4 == l3).sum()), q4.mean(), g.mean(),
        {f: int((fold == f).sum()) for f in FOLDS}))
    return dict(X=X, z=z, q4=q4, g=g, fold=fold, y=y, cid=cid, uid=uid, rows=rows)

def prep_target(path, tag):
    rows = load(path)
    X = np.array([[fnum(r[f]) for f in FEATS] for r in rows], dtype=float)
    y = np.array([int(fnum(r["y"])) for r in rows])
    cid = np.array([str(r["cluster_id"]).strip() for r in rows])
    uid = np.array([str(r["unit_id"]).strip() for r in rows])
    p3 = np.array([fnum(r["p_M3"]) for r in rows]); p4 = np.array([fnum(r["p_M4"]) for r in rows])
    alpha_orig = np.array([fnum(r["alpha"]) for r in rows]) if rows[0].get("alpha", "") != "" else None
    pr_orig = np.array([fnum(r["p_router"]) for r in rows]) if rows[0].get("p_router", "") != "" else None
    P("[%s] rows=%d  clusters=%d  y=1:%d" % (tag, len(rows), len(set(cid)), int(y.sum())))
    return dict(X=X, y=y, cid=cid, uid=uid, p3=p3, p4=p4, alpha_orig=alpha_orig,
                p_router_orig=pr_orig, rows=rows)

def scheme_weights(name, d):
    if name == "A":
        w = d["q4"] / d["q4"].mean()
    elif name == "B":
        w = np.ones_like(d["q4"])
    elif name == "C":
        w = d["g"] / d["g"].mean()
    else:
        raise ValueError(name)
    return w

# ---- 5-fold OOF EC-AUROC ----
def oof_ec_auroc(src, scheme):
    alpha = np.full(len(src["z"]), np.nan)
    for f in FOLDS:
        tr = src["fold"] != f
        te = src["fold"] == f
        med, iqr = med_iqr(src["X"][tr])
        Xtr = transform(src["X"][tr], med, iqr)
        Xte = transform(src["X"][te], med, iqr)
        w = scheme_weights(scheme, src)[tr]
        w = w / w.mean()          # mean 1 on TRAINING rows only
        coef, b, _ = fit_gate(Xtr, src["z"][tr], w)
        alpha[te] = predict(Xte, coef, b)
    return alpha, auroc(src["z"], alpha)

# ---- full-source fit + target inference ----
def fit_full_and_predict(src, tgt, scheme):
    med, iqr = med_iqr(src["X"])
    Xtr = transform(src["X"], med, iqr)
    w = scheme_weights(scheme, src)
    w = w / w.mean()
    coef, b, n_it = fit_gate(Xtr, src["z"], w)
    a_t = predict(transform(tgt["X"], med, iqr), coef, b)
    return coef, b, a_t, n_it

# ================= main =================
DIRS = [("OpenPOCUS_to_COVIDBLUES", "A", SRC_A, TGT_A, (1, 1),
         (0.6246396755, -0.0099036766), (-0.02194, 0.00223)),
        ("COVIDBLUES_to_OpenPOCUS", "B", SRC_B, TGT_B, (1, 2),
         (0.6374193548, -0.0231197512), (-0.04368, -0.00278))]

gate_ref = json.load(io.open(GATE_JSON, encoding="utf-8"))
REF_COEF = {}
for k in ("DIR-A", "DIR-B"):
    g = gate_ref["directions"][k].get("gate") or {}
    REF_COEF[k] = (g.get("coef"), g.get("intercept"), g.get("n_rows"))

results = {}
for name, idx, sp, tp, spawn, ref_vals, ref_ci in DIRS:
    P("")
    P("=" * 78)
    P("DIRECTION %s  (spawn_key=%s)" % (name, str(spawn)))
    P("=" * 78)
    src = prep_source(sp, name + " SRC")
    tgt = prep_target(tp, name + " TGT")

    # --- baseline A: OOF EC-AUROC ---
    a_oof, ec = oof_ec_auroc(src, "A")
    d_ec = abs(ec - ref_vals[0])
    P("A  EC-AUROC(5-fold OOF) = %.10f   ref %.10f   |diff| = %.3e" % (ec, ref_vals[0], d_ec))

    coefA, bA, aA_t, itA = fit_full_and_predict(src, tgt, "A")
    P("A  full-source coef = %s" % np.array2string(coefA, precision=8))
    P("A  intercept = %.8f   (newton iters=%d)" % (bA, itA))
    rk = "DIR-A" if idx == "A" else "DIR-B"
    rc, rb, rn = REF_COEF[rk]
    if rc:
        P("A  archived coef = %s" % np.array2string(np.array(rc), precision=8))
        P("A  archived intercept = %.8f  n_rows=%s" % (rb, rn))
        P("A  max|coef - archived| = %.3e   |intercept diff| = %.3e" % (
            np.max(np.abs(coefA - np.array(rc))), abs(bA - rb)))

    # target alpha / p_router vs archived columns
    if tgt["alpha_orig"] is not None:
        P("A  target alpha  max|diff| vs C2 column = %.3e" % np.max(np.abs(aA_t - tgt["alpha_orig"])))
    prA = aA_t * tgt["p4"] + (1 - aA_t) * tgt["p3"]
    if tgt["p_router_orig"] is not None:
        P("A  target p_router max|diff| vs C2 column = %.3e" % np.max(np.abs(prA - tgt["p_router_orig"])))

    auc_M4 = auroc(tgt["y"], tgt["p4"])
    auc_A = auroc(tgt["y"], prA)
    dA = auc_A - auc_M4
    P("target AUROC: M4=%.10f  A=%.10f  dA(A-M4)=%.10f  ref=%.10f  |diff|=%.3e" % (
        auc_M4, auc_A, dA, ref_vals[1], abs(dA - ref_vals[1])))

    # --- B and C ---
    store = {"A": (aA_t, prA, coefA, bA)}
    for sch in ("B", "C"):
        a_oof_s, ec_s = oof_ec_auroc(src, sch)
        coef_s, b_s, a_s, it_s = fit_full_and_predict(src, tgt, sch)
        pr_s = a_s * tgt["p4"] + (1 - a_s) * tgt["p3"]
        store[sch] = (a_s, pr_s, coef_s, b_s)
        P("%s  EC-AUROC(OOF)=%.10f  coef=%s  b=%.6f" % (
            sch, ec_s, np.array2string(coef_s, precision=6), b_s))

    # --- bootstrap, cluster-level, paired across arms ---
    keys = np.array(sorted(set(tgt["cid"])))
    kidx = {k: i for i, k in enumerate(keys)}
    cidx = np.array([kidx[c] for c in tgt["cid"]])
    members = [[] for _ in keys]
    for i, ci in enumerate(cidx):
        members[ci].append(i)
    ss = np.random.SeedSequence(MASTER_SEED, spawn_key=spawn)
    rng = np.random.default_rng(ss)
    arms = {"M4": tgt["p4"], "A": prA, "B": store["B"][1], "C": store["C"][1]}
    reps = {k: [] for k in ("A-M4", "B-M4", "C-M4", "B-A", "C-A")}
    n_att = 0; n_inv = 0
    while len(reps["A-M4"]) < N_VALID and n_att < MAX_ATTEMPTS:
        n_att += 1
        draw = rng.integers(0, len(keys), size=len(keys))
        idx_list = []
        for d in draw:
            idx_list.extend(members[d])
        idx = np.array(idx_list, dtype=int)
        yb = tgt["y"][idx]
        if yb.min() == yb.max():
            n_inv += 1
            continue
        a = {k: auroc(yb, v[idx]) for k, v in arms.items()}
        reps["A-M4"].append(a["A"] - a["M4"])
        reps["B-M4"].append(a["B"] - a["M4"])
        reps["C-M4"].append(a["C"] - a["M4"])
        reps["B-A"].append(a["B"] - a["A"])
        reps["C-A"].append(a["C"] - a["A"])
    n_val = len(reps["A-M4"])
    P("bootstrap: attempted=%d valid=%d invalid=%d" % (n_att, n_val, n_inv))
    ci = {k: (float(np.percentile(v, PCTL[0])), float(np.percentile(v, PCTL[1]))) for k, v in reps.items()}
    P("  A-M4 CI = [%.6f, %.6f]   ref [%.5f, %.5f]" % (ci["A-M4"][0], ci["A-M4"][1], ref_ci[0], ref_ci[1]))
    for k in ("B-M4", "C-M4", "B-A", "C-A"):
        P("  %-5s mean=%+.6f  CI=[%+.6f, %+.6f]" % (k, float(np.mean(reps[k])), ci[k][0], ci[k][1]))

    # --- alpha summary + target predictions ---
    rows_t = []
    for i in range(len(tgt["y"])):
        rows_t.append({
            "unit_id": tgt["uid"][i], "cluster_id": tgt["cid"][i], "y": int(tgt["y"][i]),
            "p_M3": repr(float(tgt["p3"][i])), "p_M4": repr(float(tgt["p4"][i])),
            "alpha_A": repr(float(store["A"][0][i])), "p_router_A": repr(float(store["A"][1][i])),
            "alpha_B": repr(float(store["B"][0][i])), "p_router_B": repr(float(store["B"][1][i])),
            "alpha_C": repr(float(store["C"][0][i])), "p_router_C": repr(float(store["C"][1][i])),
        })
    wr(os.path.join(TAB, "target_predictions_%s.csv" % name), list(rows_t[0].keys()), rows_t)

    results[name] = dict(
        ec_auroc=ec, ec_ref=ref_vals[0], ec_diff=d_ec,
        auc_M4=auc_M4, auc_A=auc_A, dA=dA, dA_ref=ref_vals[1],
        arms={s: auroc(tgt["y"], store[s][1]) for s in ("A", "B", "C")},
        ci=ci, reps_mean={k: float(np.mean(v)) for k, v in reps.items()},
        n_att=n_att, n_val=n_val, n_inv=n_inv,
        alpha={s: dict(mean=float(np.mean(store[s][0])), median=float(np.median(store[s][0])),
                       n_lt_05=int((store[s][0] < 0.5).sum())) for s in ("A", "B", "C")},
        coef={s: np.asarray(store[s][2]).tolist() for s in ("A", "B", "C")},
        intercept={s: float(store[s][3]) for s in ("A", "B", "C")},
    )

# ================= S5 =================
s5 = []
for name, idx, sp, tp, spawn, ref_vals, ref_ci in DIRS:
    r = results[name]
    for sch in ("A", "B", "C"):
        if sch == "A":
            d_m4 = r["dA"]; ci_m4 = r["ci"]["A-M4"]
            d_a = 0.0; ci_a = (0.0, 0.0); note = "reference (same predictions)"
        else:
            d_m4 = r["arms"][sch] - r["auc_M4"]; ci_m4 = r["ci"][sch + "-M4"]
            d_a = r["arms"][sch] - r["arms"]["A"]; ci_a = r["ci"][sch + "-A"]
            note = ""
        s5.append({
            "direction": name, "scheme": sch,
            "target_auroc": "%.10f" % r["arms"][sch],
            "delta_vs_M4": "%.10f" % d_m4,
            "delta_vs_M4_ci_low": "%.10f" % ci_m4[0], "delta_vs_M4_ci_high": "%.10f" % ci_m4[1],
            "delta_vs_A": "%.10f" % d_a,
            "delta_vs_A_ci_low": "%.10f" % ci_a[0], "delta_vs_A_ci_high": "%.10f" % ci_a[1],
            "note": note,
        })
wr(os.path.join(TAB, "Table_S5_weight_sensitivity.csv"), list(s5[0].keys()), s5)

al = []
for name, idx, sp, tp, spawn, ref_vals, ref_ci in DIRS:
    for sch in ("A", "B", "C"):
        a = results[name]["alpha"][sch]
        al.append({"direction": name, "scheme": sch,
                   "alpha_mean": "%.6f" % a["mean"], "alpha_median": "%.6f" % a["median"],
                   "n_alpha_lt_0.5": a["n_lt_05"],
                   "n_units": results[name]["n_val"] and len(results[name]["coef"][sch]) and ""})
wr(os.path.join(TAB, "target_gate_weight_summary.csv"),
   ["direction", "scheme", "alpha_mean", "alpha_median", "n_alpha_lt_0.5"], al)

meta = {
    "generated_utc": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat(),
    "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
    "note_solver": "exact Newton on 0.5*||w||^2 + C*sum(sw*logloss); C=0.01; equivalent optimum to sklearn lbfgs+StandardScaler-free median/IQR scaling",
    "sklearn_available": False, "scipy_available": False,
    "C": C_REG, "max_iter": 5000, "features": FEATS,
    "bootstrap": {"unit": "cluster_id", "n_valid_target": N_VALID, "master_seed": MASTER_SEED,
                  "spawn_key": {"OpenPOCUS_to_COVIDBLUES": [1, 1], "COVIDBLUES_to_OpenPOCUS": [1, 2]},
                  "pctl": list(PCTL), "forbidden": ["video/unit-level resampling", "normal approx", "BCa", "one-sided"]},
    "inputs": {os.path.basename(p): {"path": p, "bytes": os.path.getsize(p), "sha256": sha256(p)}
               for p in (SRC_A, SRC_B, TGT_A, TGT_B, GATE_JSON)},
    "outputs": {os.path.basename(p): {"path": p, "bytes": os.path.getsize(p), "sha256": sha256(p)}
                for p in [os.path.join(TAB, "Table_S5_weight_sensitivity.csv"),
                          os.path.join(TAB, "target_gate_weight_summary.csv")] +
                        [os.path.join(TAB, "target_predictions_%s.csv" % d[0]) for d in DIRS]},
    "results": results,
}
def conv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))
with io.open(os.path.join(VER, "sensitivity_results.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, default=conv)
with io.open(os.path.join(VER, "run_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
P("")
P("done -> %s" % OUT)
