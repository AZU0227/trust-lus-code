# -*- coding: utf-8 -*-
"""Build two source-domain OOF tables (joined with v3p1 folds) and the Vave 5-model table.
Read-only on frozen inputs. Outputs to tables/ and prints verification to verify/.
"""
import csv, io, os, sys, math, hashlib, json
from collections import Counter, defaultdict

W = r"C:\Users\99243\Documents\deepseek-harness\default-workspace\stage3"
OUT = os.path.join(W, r"report\oof_vave_package_2026-11-26")
TAB = os.path.join(OUT, "tables")
VER = os.path.join(OUT, "verify")
os.makedirs(TAB, exist_ok=True); os.makedirs(VER, exist_ok=True)

OOF_A = os.path.join(W, r"report\server_artifacts\router\router_oof_table_DIR-A.csv")
OOF_B = os.path.join(W, r"report\server_artifacts\router\router_oof_table_DIR-B.csv")
FOLD  = os.path.join(W, r"stageB_data_v3p1\fold_assignment_v3p1.csv")
UNITS = os.path.join(W, r"stageB_data_v3p1\units_v3p1.csv")
X1DIR = r"E:\TRUST_LUS\outputs\external\x1\predictions"
C2_1385 = os.path.join(W, r"report\supplementary_v04_2026-11-23\tables\C2_COVIDBLUES_to_OpenPOCUS.csv")

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
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

log = []
def P(s=""):
    print(s); log.append(s)

P("=" * 78)
P("PART 1 — source-domain OOF tables joined with v3p1 folds")
P("=" * 78)

folds = load(FOLD)
P("fold_assignment_v3p1.csv: %d rows" % len(folds))
# 1:1 index by unit_id
fold_by_unit = {}
dup_fold = []
for r in folds:
    u = r["unit_id"]
    if u in fold_by_unit:
        dup_fold.append(u)
    fold_by_unit[u] = r
P("  unit_id duplicates in fold table: %d" % len(dup_fold))

units = load(UNITS)
unit_by_id = {r["unit_id"]: r for r in units}
P("units_v3p1.csv: %d rows, library dist=%s" % (len(units), dict(Counter(r["library"] for r in units))))

SPEC = [("p3", "p_M3"), ("p4", "p_M4"), ("conf3", "C3"), ("conf4", "C4"),
        ("temp3", "T3"), ("temp4", "T4")]
EXTRA = ["disagreement", "loss3", "loss4", "q4"]

def build(src, lib, tag, outname):
    P("")
    P("-" * 78)
    P("[%s] source=%s  filter library=%s" % (tag, os.path.basename(src), lib))
    rows = load(src)
    P("  rows before lib filter: %d" % len(rows))
    rows = [r for r in rows if r["dataset"].strip() == lib]
    P("  rows after lib filter : %d" % len(rows))
    # dup key check
    keys = [r["video_id"] for r in rows]
    dups = [k for k, v in Counter(keys).items() if v > 1]
    P("  duplicate video_id    : %d" % len(dups))
    out = []
    unmatched = []
    null_fold = null_cluster = null_patient = 0
    label_conflict = []
    for i, r in enumerate(rows, start=1):
        u = r["video_id"]
        fr = fold_by_unit.get(u)
        un = unit_by_id.get(u)
        if fr is None:
            unmatched.append(u)
            rec = {"fold": "", "original_patient_id": "", "cluster_id_v3": ""}
            null_fold += 1
        else:
            rec = {"fold": fr.get("fold", ""),
                   "original_patient_id": fr.get("original_patient_id", ""),
                   "cluster_id_v3": fr.get("cluster_id_v3", "")}
            if not rec["fold"]: null_fold += 1
            if not rec["original_patient_id"]: null_patient += 1
            if not rec["cluster_id_v3"]: null_cluster += 1
        if un is not None and str(un.get("label_bline", "")).strip() not in ("", r["y"].strip()):
            label_conflict.append((u, r["y"], un.get("label_bline")))
        row = {"src_row": i, "unit_id": u, "patient_id": r["patient_id"],
               "y": r["y"], "library": r["dataset"]}
        for a, b in SPEC:
            row[b] = r[a]
        row["disagreement"] = r["disagreement"]
        row["loss3"] = r["loss3"]; row["loss4"] = r["loss4"]; row["q4"] = r["q4"]
        row["fold"] = rec["fold"]
        row["original_patient_id"] = rec["original_patient_id"]
        row["cluster_id_v3"] = rec["cluster_id_v3"]
        out.append(row)
    P("  joined rows           : %d" % len(out))
    P("  unmatched video_id    : %d %s" % (len(unmatched), unmatched[:5]))
    P("  null fold/patient/cluster: %d / %d / %d" % (null_fold, null_patient, null_cluster))
    P("  label conflicts (y vs units_v3p1.label_bline): %d %s" % (len(label_conflict), label_conflict[:3]))
    pos = sum(1 for r in out if r["y"].strip() == "1")
    neg = sum(1 for r in out if r["y"].strip() == "0")
    P("  y: pos=%d neg=%d  (rows=%d)" % (pos, neg, len(out)))
    P("  患者数(original_patient_id): %d   簇数(cluster_id_v3): %d" % (
        len(set(r["original_patient_id"] for r in out if r["original_patient_id"])),
        len(set(r["cluster_id_v3"] for r in out if r["cluster_id_v3"]))))
    # fold crossing
    pat2f = defaultdict(set); clu2f = defaultdict(set)
    for r in out:
        if r["fold"]:
            pat2f[r["original_patient_id"]].add(r["fold"])
            clu2f[r["cluster_id_v3"]].add(r["fold"])
    bad_pat = {k: sorted(v) for k, v in pat2f.items() if len(v) > 1}
    bad_clu = {k: sorted(v) for k, v in clu2f.items() if len(v) > 1}
    P("  跨折患者: %d %s" % (len(bad_pat), list(bad_pat.items())[:3]))
    P("  跨折簇  : %d %s" % (len(bad_clu), list(bad_clu.items())[:3]))
    P("  逐折 可评估行/阳性:")
    for f in ("1", "2", "3", "4", "5"):
        sub = [r for r in out if r["fold"] == f]
        P("    fold %s: %4d 行, 阳性 %3d" % (f, len(sub), sum(1 for r in sub if r["y"].strip() == "1")))
    # z = I(loss4 < loss3)
    ties = 0; zpos = 0; nonnum = 0
    for r in out:
        try:
            l3 = float(r["loss3"]); l4 = float(r["loss4"])
        except Exception:
            nonnum += 1; continue
        if l4 < l3: zpos += 1
        elif l4 == l3: ties += 1
    P("  z=I(loss4<loss3): z=1 共 %d, 平局 %d, 非数值 %d" % (zpos, ties, nonnum))
    cols = ["src_row", "unit_id", "patient_id", "original_patient_id", "cluster_id_v3",
            "fold", "y", "p_M3", "p_M4", "C3", "C4", "T3", "T4", "disagreement",
            "loss3", "loss4", "q4"]
    p = os.path.join(TAB, outname)
    wr(p, cols, out)
    P("  written: %s (%d rows, %d B)" % (outname, len(out), os.path.getsize(p)))
    return out, {"rows": len(out), "pos": pos, "neg": neg, "unmatched": len(unmatched),
                 "dup": len(dups), "bad_pat": len(bad_pat), "bad_clu": len(bad_clu),
                 "ties": ties, "zpos": zpos, "path": p}

resA, stA = build(OOF_A, "openpocus", "OpenPOCUS source -> target COVID-BLUeS",
                  "OOF_source_OpenPOCUS_to_COVIDBLUES.csv")
resB, stB = build(OOF_B, "covid_blues", "COVID-BLUeS source -> target OpenPOCUS",
                  "OOF_source_COVIDBLUES_to_OpenPOCUS.csv")

# 两条短视频的折号
P("")
P("-" * 78)
P("[short videos in COVID fold table]")
for v in ("patient_3_R2", "patient_5_R3"):
    fr = fold_by_unit.get(v)
    P("  %-14s fold=%s  (in OOF table: %s)" % (
        v, fr.get("fold") if fr else "NOT_IN_FOLD_TABLE",
        any(r["unit_id"] == v for r in resB)))

P("")
P("=" * 78)
P("PART 2 — Vave 374 five-model probability table")
P("=" * 78)

FAMS = ["M1", "M2", "M3", "M4", "M5"]
FOLDS = ["1", "2", "3", "4", "5"]
per = {}
p2log = []
for fam in FAMS:
    ids_ref = None
    for f in FOLDS:
        p = os.path.join(X1DIR, "%s_seed13_fold%s.csv" % (fam, f))
        if not os.path.exists(p):
            raise SystemExit("MISSING " + p)
        rows = load(p)
        d = {}
        bad = 0
        for r in rows:
            u = r["unit_id"]
            v = float(r["prob"])
            if not math.isfinite(v) or v < 0 or v > 1:
                bad += 1
            d[u] = v
        p2log.append("  %s fold%s: rows=%d unique=%d bad_prob=%d" % (fam, f, len(rows), len(d), bad))
        if ids_ref is None:
            ids_ref = set(d)
        elif set(d) != ids_ref:
            p2log.append("    !! ID set differs from fold1: +%d -%d" % (
                len(set(d) - ids_ref), len(ids_ref - set(d))))
        per.setdefault(fam, {})[f] = d
for s in p2log: P(s)

ids = set(per["M1"]["1"])
P("  common unit_id across 5 folds: %d" % len(ids))
for fam in FAMS:
    for f in FOLDS:
        assert set(per[fam][f]) == ids, (fam, f)

# 冻结身份表
ident = {}
for r in units:
    if r["library"] == "openpocus":
        ident[r["unit_id"]] = r
c2 = {r["unit_id"]: r for r in load(C2_1385)}
P("  units_v3p1 openpocus: %d ; C2_1385 keys: %d" % (len(ident), len(c2)))

rows2 = []
miss_ident = []
for u in sorted(ids):
    if u not in ident:
        miss_ident.append(u); continue
    r = ident[u]
    rec = {"unit_id": u, "patient_id": r["original_patient_id"], "cluster_id": r["cluster_id"],
           "y": r["label_bline"]}
    for fam in FAMS:
        vals = [per[fam][f][u] for f in FOLDS]
        rec["p_" + fam] = repr(sum(vals) / 5.0)
    rec["folds_contributing"] = "5 (M%s seed13 fold1-5)" % ""
    rec["n_folds"] = 5
    rows2.append(rec)
P("  rows built: %d ; missing identity: %d %s" % (len(rows2), len(miss_ident), miss_ident[:3]))

pos2 = sum(1 for r in rows2 if str(r["y"]).strip() == "1")
neg2 = sum(1 for r in rows2 if str(r["y"]).strip() == "0")
P("  y: pos=%d neg=%d" % (pos2, neg2))
P("  patients=%d clusters=%d" % (len(set(r["patient_id"] for r in rows2)),
                                 len(set(r["cluster_id"] for r in rows2))))
# 与 C2 身份一致性
conf = 0
for r in rows2:
    c = c2.get(r["unit_id"])
    if c is None: continue
    if str(c["y"]).strip() != str(r["y"]).strip(): conf += 1
    if str(c["cluster_id"]).strip() != str(r["cluster_id"]).strip(): conf += 1
P("  vs C2_1385 identity conflicts (y or cluster): %d" % conf)

cols2 = ["unit_id", "patient_id", "cluster_id", "y", "p_M1", "p_M2", "p_M3", "p_M4", "p_M5", "n_folds"]
p2 = os.path.join(TAB, "Vave_target_pM1_pM5_reconstructed.csv")
wr(p2, cols2, rows2)
P("  written: Vave_target_pM1_pM5_reconstructed.csv (%d rows, %d B)" % (len(rows2), os.path.getsize(p2)))

# ---- metrics ----
def auroc(y, s):
    pairs = sorted(zip(s, y))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    npos = sum(1 for _, yy in pairs if yy == 1)
    nneg = n - npos
    if npos == 0 or nneg == 0: return float("nan")
    s1 = sum(r for r, (_, yy) in zip(ranks, pairs) if yy == 1)
    return (s1 - npos * (npos + 1) / 2.0) / (npos * nneg)

def average_precision(y, s):
    order = sorted(range(len(s)), key=lambda i: -s[i])
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    npos = sum(1 for v in y if v == 1)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            if y[order[k]] == 1: tp += 1
            else: fp += 1
        recall = tp / npos
        prec = tp / (tp + fp)
        ap += (recall - prev_recall) * prec
        prev_recall = recall
        i = j + 1
    return ap

def brier(y, s):
    return sum((si - yi) ** 2 for si, yi in zip(s, y)) / len(y)

yv = [1 if str(r["y"]).strip() == "1" else 0 for r in rows2]
FROZEN = {  # frozen P1 target values
    "M1": (0.9075413223140496, 0.5937322010614062, 0.0454727495122793),
    "M2": (0.9484762396694214, 0.6596669467435441, 0.0370949217375052),
    "M3": (0.9817923553719008, 0.7725015007165782, 0.03770546387127821),
    "M4": (0.9629390495867769, 0.8031320558061372, 0.025747249627160494),
    "M5": (0.9461518595041323, 0.7700604231769619, 0.02766903539072392),
}
P("")
P("-" * 78)
P("[P1 metric recomputation on reconstructed Vave slice vs frozen P1]")
P("  %-3s %-12s %-12s %-11s  %-9s %-9s %-9s" % ("fam", "AUROC_this", "AUROC_frozen", "dAUROC", "dAUPRC", "dBrier", ""))
mx = {"auroc": 0.0, "auprc": 0.0, "brier": 0.0}
for fam in FAMS:
    s = [float(r["p_" + fam]) for r in rows2]
    a, ap, br = auroc(yv, s), average_precision(yv, s), brier(yv, s)
    fa, fp_, fb = FROZEN[fam]
    da, dp, db = abs(a - fa), abs(ap - fp_), abs(br - fb)
    mx["auroc"] = max(mx["auroc"], da); mx["auprc"] = max(mx["auprc"], dp); mx["brier"] = max(mx["brier"], db)
    P("  %-3s %-12.9f %-12.9f %-11.2e  %-9.2e %-9.2e" % (fam, a, fa, da, dp, db))
P("  max|diff|: AUROC=%.3e  AUPRC=%.3e  Brier=%.3e" % (mx["auroc"], mx["auprc"], mx["brier"]))

# ---- write verification json ----
ver = {
    "generated_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "inputs": {os.path.basename(p): {"path": p, "bytes": os.path.getsize(p), "sha256": sha256(p)}
               for p in (OOF_A, OOF_B, FOLD, UNITS, C2_1385)},
    "outputs": {os.path.basename(p): {"path": p, "bytes": os.path.getsize(p), "sha256": sha256(p)}
                for p in (stA["path"], stB["path"], p2)},
    "part1": {"openpocus": {k: v for k, v in stA.items() if k != "path"},
              "covid": {k: v for k, v in stB.items() if k != "path"}},
    "part2": {"rows": len(rows2), "pos": pos2, "neg": neg2,
              "patients": len(set(r["patient_id"] for r in rows2)),
              "clusters": len(set(r["cluster_id"] for r in rows2)),
              "identity_conflicts_vs_C2": conf,
              "metric_max_abs_diff": mx},
    "x1_files": sorted(os.listdir(X1DIR))[:0] or "see verify/x1_file_sha256.csv",
}
with io.open(os.path.join(VER, "verification.json"), "w", encoding="utf-8") as f:
    json.dump(ver, f, ensure_ascii=False, indent=2)

with io.open(os.path.join(VER, "x1_file_sha256.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "bytes", "sha256"])
    for fam in FAMS:
        for fo in FOLDS:
            p = os.path.join(X1DIR, "%s_seed13_fold%s.csv" % (fam, fo))
            w.writerow([os.path.basename(p), os.path.getsize(p), sha256(p)])

with io.open(os.path.join(VER, "run_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))

P("")
P("verification written to %s" % VER)
