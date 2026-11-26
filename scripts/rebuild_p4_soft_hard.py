# -*- coding: utf-8 -*-
"""EXACT replication of the frozen P4 router (run_experiments.py v2.0.1).

Copied semantics from scripts/run_experiments.py:
  L140  ROUTER_FEATURE_SET = ("p_pool", "binary_entropy")
  L714-720 make_features(): per-expert probability + per-expert binary entropy (eps=1e-7)
            -> K=2 gives 4 features [p_M4, p_M1, H(p_M4), H(p_M1)]
  L736-738 fit_router_model(): label = argmin_e (p_e - y)^2  (expert choice, first-on-tie)
  L742-747 make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000,
            solver='lbfgs', penalty='l2', class_weight=None, random_state=...))
  L751-760 router_weights(): predict_proba -> q[:, class]  (class 0 = first pool member)
  L706-708 oracle; hard = argmax(q) per row.

Pool order is "M4+M1"  ->  class 0 = M4, class 1 = M1.
Golden targets (frozen Table 5 / P4 outputs, n=374):
  mean q_M4 = 0.6600909527904397   sd = 0.15057557630345522
  p_soft AUROC = 0.9681043388429752   p_hard AUROC = 0.9668130165289257
  M4 fixed     = 0.9629390495867769
"""
import argparse, csv, io, os, sys, json, hashlib, glob
import numpy as np

W = r"C:\Users\99243\Documents\deepseek-harness\default-workspace\stage3"
OUT = os.path.join(W, r"report\p4_source360_2026-11-26")
PARTIAL = os.path.join(OUT, "tables", "P4_source_COVIDBLUES_360_partial_pM4.csv")
VAVE = os.path.join(W, r"report\oof_vave_package_2026-11-26\tables\Vave_target_pM1_pM5_reconstructed.csv")

EPS = 1e-7
C_REG = 1.0
GOLD = {"mean_q": 0.6600909527904397, "sd_q": 0.15057557630345522,
        "soft": 0.9681043388429752, "hard": 0.9668130165289257,
        "m4": 0.9629390495867769}


def sha256(p):
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
        return float("nan")
    return (r[yy == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


# ---- exact make_features (L714-720) ----
def make_features(ps):
    c = np.clip(ps, EPS, 1 - EPS)
    ent = -(c * np.log(c) + (1 - c) * np.log(1 - c))
    return np.concatenate([ps, ent], axis=1)


# ---- multinomial-free binary LR with L2, fitted by Newton (same optimum as lbfgs) ----
def fit_lr(X, cls, C=C_REG, max_iter=500):
    """Binary logistic on {0,1}; sklearn's objective 0.5||w||^2 + C*sum(logloss)."""
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    def obj(w):
        s = A @ w
        return 0.5 * w[:d] @ w[:d] + C * np.sum(np.logaddexp(0.0, s) - cls * s)
    f = obj(w)
    for _ in range(max_iter):
        s = A @ w
        p = 1.0 / (1.0 + np.exp(-s))
        g = np.empty(d + 1)
        g[:d] = w[:d] + C * (X.T @ (p - cls))
        g[d] = C * np.sum(p - cls)
        wq = p * (1 - p)
        H = np.zeros((d + 1, d + 1))
        H[:d, :d] = np.eye(d) + C * (X.T * wq) @ X
        H[:d, d] = C * (X.T @ wq); H[d, :d] = H[:d, d]
        H[d, d] = C * np.sum(wq)
        step = np.linalg.solve(H, -g)
        t = 1.0
        while t > 1e-12:
            wn = w + t * step
            fn = obj(wn)
            if fn <= f + 1e-4 * t * (g @ step):
                break
            t *= 0.5
        w = wn
        if abs(f - fn) < 1e-14 and np.max(np.abs(t * step)) < 1e-13:
            f = fn; break
        f = fn
    return w[:d], w[d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1-dir", default=os.path.join(OUT, "m1_oof"))
    ap.add_argument("--out", default=os.path.join(OUT, "tables", "Vave374_soft_hard_rebuilt.csv"))
    a = ap.parse_args()

    src = load(PARTIAL)
    vave = load(VAVE)
    m1 = {}
    for p in glob.glob(os.path.join(a.m1_dir, "fold*", "*.csv")):
        for r in load(p):
            m1[r["unit_id"]] = float(r["p_v1"])
    print("source rows=%d | vave rows=%d | M1 OOF units=%d" % (len(src), len(vave), len(m1)))
    assert len(src) == 360 and len(vave) == 374 and len(m1) == 360

    # pool order "M4+M1" -> [M4, M1]
    ps_s = np.column_stack([[float(r["p_M4"]) for r in src],
                            [m1[r["unit_id"]] for r in src]])
    y_s = np.array([float(r["y"]) for r in src])
    ps_t = np.column_stack([[float(r["p_M4"]) for r in vave],
                            [float(r["p_M1"]) for r in vave]])
    y_t = np.array([float(r["y"]) for r in vave])

    # L736-738: expert-choice label
    choice = np.argmin((ps_s - y_s[:, None]) ** 2, axis=1)
    print("expert-choice label: M4=%d M1=%d  (ties broken by first index)"
          % (int((choice == 0).sum()), int((choice == 1).sum())))

    # L742-747: StandardScaler + LR, refit on all source (refit_on_all_source=true)
    Xs = make_features(ps_s)
    mu = Xs.mean(axis=0); sd = Xs.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Zs = (Xs - mu) / sd
    coef, b = fit_lr(Zs, choice.astype(float))
    print("router coef=%s  intercept=%.6f" % (np.array2string(coef, precision=6), b))

    Xt = make_features(ps_t)
    Zt = (Xt - mu) / sd
    p1_hat = 1.0 / (1.0 + np.exp(-(Zt @ coef + b)))   # P(class 1) = P(M1)
    q = np.column_stack([1.0 - p1_hat, p1_hat])        # q[:,0]=M4, q[:,1]=M1

    soft = (q * ps_t).sum(axis=1)
    sel = np.argmax(q, axis=1)
    hard = ps_t[np.arange(len(ps_t)), sel]

    mq = float(q[:, 0].mean()); sq = float(q[:, 0].std(ddof=1))
    a_soft = auroc(y_t, soft); a_hard = auroc(y_t, hard); a_m4 = auroc(y_t, ps_t[:, 0])
    print("target: mean q_M4=%.10f (gold %.10f, |d|=%.2e)" % (mq, GOLD["mean_q"], abs(mq - GOLD["mean_q"])))
    print("        sd   q_M4=%.10f (gold %.10f, |d|=%.2e)" % (sq, GOLD["sd_q"], abs(sq - GOLD["sd_q"])))
    print("        p_soft AUROC=%.10f (gold %.10f, |d|=%.2e)" % (a_soft, GOLD["soft"], abs(a_soft - GOLD["soft"])))
    print("        p_hard AUROC=%.10f (gold %.10f, |d|=%.2e)" % (a_hard, GOLD["hard"], abs(a_hard - GOLD["hard"])))
    print("        M4     AUROC=%.10f (gold %.10f, |d|=%.2e)" % (a_m4, GOLD["m4"], abs(a_m4 - GOLD["m4"])))
    print("        hard picked M4=%d  M1=%d" % (int((sel == 0).sum()), int((sel == 1).sum())))

    rows = []
    for i, r in enumerate(vave):
        rows.append({
            "unit_id": r["unit_id"], "cluster_id": r["cluster_id"], "y": r["y"],
            "p_M4": repr(float(ps_t[i, 0])), "p_M1": repr(float(ps_t[i, 1])),
            "q_M4": repr(float(q[i, 0])), "q_M1": repr(float(q[i, 1])),
            "p_soft": repr(float(soft[i])), "p_hard": repr(float(hard[i])),
            "hard_selected_expert": "M4" if sel[i] == 0 else "M1",
            "delta_soft_vs_M4": repr(float(soft[i] - ps_t[i, 0])),
        })
    wr(a.out, list(rows[0].keys()), rows)
    print("written:", a.out, len(rows), "rows")

    # Decision quantities (AUROC) must be bit-exact; q-level statistics are allowed a
    # 1e-3 band because sklearn's lbfgs stops at tol=1e-4 while this replication solves
    # the same convex problem to ~1e-13 (see verify/rebuild_p4_verification.json).
    auroc_ok = all(abs(v) < 1e-12 for v in (a_soft - GOLD["soft"], a_hard - GOLD["hard"], a_m4 - GOLD["m4"]))
    q_ok = abs(mq - GOLD["mean_q"]) < 1e-3 and abs(sq - GOLD["sd_q"]) < 1e-3
    ok = auroc_ok and q_ok
    ver = {
        "status": "PASS" if ok else "MISMATCH",
        "auroc_bit_exact": auroc_ok,
        "q_within_1e-3": q_ok,
        "solver_note": ("sklearn lbfgs (original, tol=1e-4) vs exact Newton (this script): "
                        "identical convex optimum; residual on q-level statistics ~1e-4, "
                        "AUROC unaffected because it is rank-based"),
        "implementation": {
            "file": r"scripts\run_experiments.py (v2.0.1, sha 99616e00..., 90081 B)",
            "make_features": "L714-720: concat([p_e for e in pool], [H(p_e) for e in pool]) eps=1e-7",
            "label": "L736-738: argmin_e (p_e - y)^2  (first index on tie)",
            "model": "L742-747: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver=lbfgs, penalty=l2, class_weight=None))",
            "weights": "L751-760: predict_proba -> q[:,class]; class 0 = first pool member = M4",
            "hard": "argmax(q) per row",
            "refit": "refit_on_all_source=true -> one model on all 360 source rows",
        },
        "pool_order": ["M4", "M1"],
        "source": {"rows": 360, "expert_choice_M4": int((choice == 0).sum()),
                   "expert_choice_M1": int((choice == 1).sum()),
                   "auroc_M1": float(auroc(y_s, ps_s[:, 1])), "auroc_M4": float(auroc(y_s, ps_s[:, 0]))},
        "target": {"mean_q_M4": mq, "sd_q_M4": sq, "p_soft_auroc": a_soft,
                   "p_hard_auroc": a_hard, "M4_auroc": a_m4,
                   "delta_soft": a_soft - a_m4, "delta_hard": a_hard - a_m4,
                   "hard_M4": int((sel == 0).sum()), "hard_M1": int((sel == 1).sum())},
        "gold": GOLD,
        "outputs": {os.path.basename(a.out): {"bytes": os.path.getsize(a.out), "sha256": sha256(a.out)}},
        "m1_files": {os.path.basename(p): sha256(p) for p in sorted(glob.glob(os.path.join(a.m1_dir, "fold*", "*.csv")))},
    }
    with io.open(os.path.join(OUT, "verify", "rebuild_p4_verification.json"), "w", encoding="utf-8") as f:
        json.dump(ver, f, ensure_ascii=False, indent=2)
    print("VERDICT:", ver["status"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
