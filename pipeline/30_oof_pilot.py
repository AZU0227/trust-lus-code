#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRUST-LUS V4.2 — Router/OOF mechanism-pilot driver (20 fits), v2.

What this script is, and what it is NOT
---------------------------------------
This is the *zero-GPU preparation* half of Protocol §7.4.1 Step 1 ("OOF 机制
Pilot, 20 次").  It never trains a model and never imports torch:

* ``--self-test``  structural checks of the 20-job enumeration and of the
  output-directory / DONE.json convention of ``src/engine.py`` (CPU only).
* ``--dry-run``    the whole OOF plumbing end-to-end on **synthetic predictions**
  (CPU, small arrays, no DONE.json, no GPU, no data decode): features ->
  per-direction Median/IQR -> soft regret target -> τ/C grid -> Gate -> 14-field
  ``router_oof_table`` + hashes.  It is the rehearsal of the real run.
* default          builds ``outputs/_plan/<tag>/`` = the **待办清单** (job list,
  preflight report, TODO) and stops.  ``✗ 不训练`` — the 20 fold-models are
  launched later, on the server, by the fold-trainer adapter (see P0 in
  ``report/pilot_runbook.md``).

Frozen counts quoted from the lock / protocol
---------------------------------------------
* 20 fits = **2 家族 × 5 折 × 2 方向** (Protocol §7.4.1; §7.4.7 self-check
  ``20 = 2 × 5 × 2``).  Router's own fits (``D × 1 × s = 2``) are NOT part of
  the 20 and must not be added to it ("三者不得相加后当作 26 次拟合", §7.4.7).
* DIR-A = OpenPOCUS → COVID-BLUeS, DIR-B = COVID-BLUeS → OpenPOCUS (Lock §1.1,
  §3.1); the source pool is 1 385 / 360 units and 185 / 63 clusters (Lock §3.1
  L-09' scale statement: 3.85×).
* folds = 5, frozen assignment read from ``fold_assignment_v3p1.csv`` (Lock §7
  B-08); the pilot uses ``s = 1``, seed = 13 (``seed_reference.yaml``
  ``formal_seeds: [13, 29, 47]``, first seed).
* output convention = ``src/engine.py`` ``train_one_fold``:
  ``outputs/<out_tag>/<family>/seed<seed>/fold<fold>/DONE.json`` +
  ``checkpoints/<out_tag>/<family>/seed<seed>/fold<fold>/``, ``DONE.json`` makes
  a rerun a no-op unless ``--force``.

Why the plan lives outside the stage tag
----------------------------------------
The runbook asserts "the tag directory does not exist" before a launch.  A plan
written *into* ``outputs/<tag>/`` would break that assertion, so the plan goes to
``outputs/_plan/<tag>/`` and the stage tag stays absent until the 20 fits run.

CLI
---
    python scripts/30_oof_pilot.py --self-test
    python scripts/30_oof_pilot.py --dry-run --data-root E:\\TRUST_LUS
    python scripts/30_oof_pilot.py --tag T1_router_oof_v1 --data-root D:\\TRUST_LUS
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features_v2 as F  # noqa: E402
from src import router_v2 as R  # noqa: E402
from src.norm_v2 import DIRECTIONS, unit_set_sha256  # noqa: E402

FAMILIES = ("M3", "M4")          # pilot families (Protocol §7.4.1: 2 家族)
FOLDS = (1, 2, 3, 4, 5)          # Lock §7 B-08
PILOT_SEED = 13                  # seed_reference.yaml formal_seeds[0]; s = 1
N_FITS = len(FAMILIES) * len(FOLDS) * 2   # = 20 (Protocol §7.4.1)
DEFAULT_TAG = "T1_router_oof_v1"
PLAN_SUBDIR = Path("outputs") / "_plan"
RUN_KIND = "precision_pilot_oof"
TRAIN_SCRIPT = "scripts/13_train_family_t1.py"


def _load_loader():
    """Import scripts/98_loader_v2.py (a numeric filename cannot be imported)."""
    path = ROOT / "scripts" / "98_loader_v2.py"
    spec = importlib.util.spec_from_file_location("trustlus_loader_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # noqa: S301 - local, frozen file
    return mod


LV = _load_loader()


# ------------------------------------------------------------------- naming
def direction_tag(tag: str, direction: str) -> str:
    """One engine stage tag per direction so the two never share a directory."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}")
    return f"{tag}_{'dirA' if direction == 'DIR-A' else 'dirB'}"


def out_dirs(root: Path, tag: str, direction: str, family: str, seed: int, fold: int) -> dict:
    """Exact directory convention of src/engine.py::train_one_fold."""
    stage = direction_tag(tag, direction)
    out = Path(root) / "outputs" / stage / family / f"seed{seed}" / f"fold{fold}"
    ckpt = Path(root) / "checkpoints" / stage / family / f"seed{seed}" / f"fold{fold}"
    return {"stage_tag": stage, "out_dir": out, "ckpt_dir": ckpt, "done_json": out / "DONE.json"}


def build_jobs(root: Path, tag: str, seed: int = PILOT_SEED) -> pd.DataFrame:
    """Enumerate the 20 fold-model fits (2 families × 5 folds × 2 directions)."""
    units = LV.load_units(root)
    rows = []
    for direction in ("DIR-A", "DIR-B"):
        src = DIRECTIONS[direction]["source"]
        tgt = DIRECTIONS[direction]["target"]
        for fold in FOLDS:
            in_fold = units[
                (units["library"] == src) & (units["fold"] == fold) & (units["evaluable"])
            ]
            out_fold = units[
                (units["library"] == src) & (units["fold"] != fold) & (units["evaluable"])
            ]
            for family in FAMILIES:
                d = out_dirs(root, tag, direction, family, seed, fold)
                rows.append(
                    {
                        "job_id": f"P1_{direction.replace('-', '')}_{family}_f{fold}",
                        "step": "Step1_OOF_mechanism_pilot",
                        "direction": direction,
                        "family": family,
                        "fold": fold,
                        "seed": seed,
                        "run_kind": RUN_KIND,
                        "source_library": src,
                        "target_library": tgt,
                        "n_train_units": int(len(out_fold)),
                        "n_val_units": int(len(in_fold)),
                        "n_train_clusters": int(out_fold["cluster_id"].nunique()),
                        "n_val_clusters": int(in_fold["cluster_id"].nunique()),
                        "stage_tag": d["stage_tag"],
                        "out_dir": str(d["out_dir"]),
                        "ckpt_dir": str(d["ckpt_dir"]),
                        "done_json": str(d["done_json"]),
                        "command": (
                            f"python {TRAIN_SCRIPT} --family {family} --seed {seed} "
                            f"--fold {fold} --out-tag {d['stage_tag']} --pre-lock-run"
                        ),
                        "blocked_on": "src/engine_v2.py (v3.1 source-manifest adapter) — see runbook §P0",
                        "status": "pending",
                    }
                )
    jobs = pd.DataFrame(rows)
    if len(jobs) != N_FITS:
        raise RuntimeError(f"job enumeration produced {len(jobs)} rows, expected {N_FITS}")
    if jobs["job_id"].duplicated().any():
        raise RuntimeError("duplicate job_id")
    return jobs


def assert_tag_fresh(root: Path, tag: str) -> None:
    """The runbook's launch assertion: no stage-tag directory may exist yet."""
    root = Path(root)
    for direction in DIRECTIONS:
        stage = direction_tag(tag, direction)
        for base in ("outputs", "checkpoints"):
            p = root / base / stage
            if p.exists():
                raise SystemExit(
                    f"ABORT: {p} already exists. The runbook requires a fresh tag "
                    f"(refuse-to-overwrite / idempotence). Choose a new --tag."
                )


# ------------------------------------------------------------------ preflight
def preflight(root: Path, data_root: Path | None = None) -> dict:
    """Data-side preflight for the 20 fits: counts, folds, hashes, media, deps."""
    checks = []

    def add(name, ok, detail="", severity="block"):
        # severity: "block" = stops the launch; "escalate" = must be ruled on by the
        # 负责人 and is reported, but does not by itself stop planning.
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "severity": severity})

    units = LV.load_units(root)
    add("pool units == 1 897", len(units) == 1897, len(units))
    add("pool clusters == 395", units["cluster_id"].nunique() == 395, units["cluster_id"].nunique())
    add(
        "OpenPOCUS 1 385 / COVID-BLUeS 362",
        units.groupby("library").size().to_dict() == LV.FROZEN_UNITS,
        units.groupby("library").size().to_dict(),
    )
    fa = LV.load_fold_assignment(root)
    per_cluster = fa.groupby("cluster_id_v3")["fold"].nunique()
    add("clusters single-fold 395/395", int((per_cluster == 1).sum()) == 395, int(per_cluster.max()))
    sup = LV.check_supervision_contract(root, units)
    add("A-1 required columns", sup["A1"])
    add("A-3 allowed ∩ forbidden = ∅", sup["A3"])
    add("A-4 allowed ⊆ source library", sup["A4"])
    add("A-5 one row per (direction, fold)", sup["A5"])
    add(
        "A-2 strict reading (allowed ∪ forbidden = 1 897) holds",
        sup["A2_strict"],
        "FALSE on the frozen v3.1 file — escalated in runbook §P3; not redefined",
        severity="escalate",
    )
    add(
        "direction coverage 1 385 / 360",
        sup["direction_coverage"]["DIR-A"]["complete"] and sup["direction_coverage"]["DIR-B"]["complete"],
        sup["direction_coverage"],
    )

    manifest_hashes = {}
    for direction in DIRECTIONS:
        for fold in FOLDS:
            try:
                man = LV.build_manifest(root, direction, fold, verify=True)
                manifest_hashes[f"{direction}_f{fold}"] = unit_set_sha256(man["unit_id"])
                add(f"{direction} fold{fold} manifest hash == frozen", True, len(man))
            except Exception as exc:  # noqa: BLE001
                add(f"{direction} fold{fold} manifest hash == frozen", False, str(exc))

    media = {"probed": 0, "ok": 0, "skipped": True}
    if data_root is not None and Path(data_root).exists():
        media["skipped"] = False
        try:
            man = LV.media_paths(data_root, LV.build_manifest(root, "DIR-A", 1)).dropna(
                subset=["media_path"]
            )
            sample = man.head(10)
            media["probed"] = int(len(sample))
            okc = 0
            for _, r in sample.iterrows():
                try:
                    okc += int(len(LV.index_jpeg_dir(str(r["media_path"]))) >= 16)
                except Exception:  # noqa: BLE001
                    pass
            media["ok"] = okc
            add("media: 10/10 sampled OpenPOCUS sequences have >= 16 frames", okc == len(sample), media)
        except Exception as exc:  # noqa: BLE001
            add("media probe", False, str(exc))

    deps = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": _version_of("sklearn"),
        "cv2": _version_of("cv2"),
        "torch_imported": "torch" in sys.modules,
    }
    add("torch is NOT imported by the pilot planner", not deps["torch_imported"])
    add(
        "scikit-learn present (Gate main path per Lock §7 B-05)",
        deps["sklearn"] != "absent",
        "absent here -> numpy Newton/IRLS fallback; the Gate backend is recorded in "
        "router_oof_table.meta.json. The server MUST have scikit-learn 1.7.2 "
        "(environment_reference.yaml key_versions) so the B-05 verbatim estimator is used.",
        severity="escalate",
    )

    lock = Path(root) / "report" / "Lock_R1_v2_draft_v1.1.md"
    proto = Path(root) / "report" / "Protocol_v2_v1_1_FINAL.md"
    locks = {
        "lock_present": lock.exists(),
        "protocol_present": proto.exists(),
        "lock_sha256": LV.sha256_file(lock) if lock.exists() else None,
        "protocol_sha256": LV.sha256_file(proto) if proto.exists() else None,
    }
    add("lock instrument present", locks["lock_present"], str(lock))

    blocking_failed = [c for c in checks if not c["ok"] and c["severity"] == "block"]
    escalated = [c for c in checks if not c["ok"] and c["severity"] == "escalate"]
    return {
        "status": "PASS" if not blocking_failed else "FAIL",
        "root": str(root),
        "data_root": str(data_root) if data_root else None,
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": len(blocking_failed),
        "n_escalated": len(escalated),
        "escalated_checks": [c["check"] for c in escalated],
        "manifest_hashes": manifest_hashes,
        "supervision": sup,
        "media": media,
        "deps": deps,
        "locks": locks,
        "generated_utc": None,   # filled by the caller (kept out of hashed content)
    }


def _version_of(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "absent"


# --------------------------------------------------------------- plan output
def write_plan(root: Path, tag: str, seed: int, data_root: Path | None, force: bool = False) -> dict:
    """Write the 待办清单 (joblist + preflight + TODO) and NOTHING else."""
    assert_tag_fresh(root, tag)
    plan_dir = Path(root) / PLAN_SUBDIR / tag
    if plan_dir.exists() and not force:
        raise SystemExit(f"ABORT: plan dir {plan_dir} exists; pass --force to regenerate")
    plan_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(root, tag, seed)
    jobs.to_csv(plan_dir / "joblist.csv", index=False, lineterminator="\n")

    pf = preflight(root, data_root)
    (plan_dir / "preflight.json").write_text(
        json.dumps(pf, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    todo = _todo_markdown(tag, seed, jobs, pf, plan_dir)
    (plan_dir / "TODO.md").write_text(todo, encoding="utf-8", newline="\n")

    summary = {
        "tag": tag,
        "plan_dir": str(plan_dir),
        "n_jobs": int(len(jobs)),
        "n_fits": N_FITS,
        "preflight_status": pf["status"],
        "preflight_failed": pf["n_failed"],
        "preflight_escalated": pf["n_escalated"],
        "escalated_checks": pf["escalated_checks"],
        "joblist_sha256": LV.sha256_file(plan_dir / "joblist.csv"),
        "trained": False,
        "note": "待办清单 only; no DONE.json, no checkpoints, no GPU touched.",
    }
    (plan_dir / "plan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def _todo_markdown(tag: str, seed: int, jobs: pd.DataFrame, pf: dict, plan_dir: Path) -> str:
    lines = []
    lines.append(f"# 待办清单 — Router OOF 机制 Pilot `{tag}`")
    lines.append("")
    lines.append("> 本文件由 `scripts/30_oof_pilot.py` 生成。**未训练任何模型**、未写 `DONE.json`、未占用 GPU。")
    lines.append("> 20 次拟合 = 2 家族 × 5 折 × 2 方向（`Protocol_v2_v1_1_FINAL.md` §7.4.1／§7.4.7）。")
    lines.append("")
    lines.append(f"- plan dir: `{plan_dir}`")
    lines.append(f"- preflight: **{pf['status']}**（{pf['n_checks']} 项，阻塞失败 {pf['n_failed']} 项，"
                 f"待裁定 {pf['n_escalated']} 项）")
    lines.append(f"- seed = {seed}（`seed_reference.yaml` `formal_seeds[0]`，pilot `s = 1`）")
    lines.append("")
    lines.append("## P0 前置（阻塞 20 次拟合）")
    lines.append("")
    lines.append("1. **`src/engine_v2.py`（v3.1 源域单折训练适配器）尚不存在**：`src/engine.py::train_one_fold` 固定读")
    lines.append("   `contracts.t1_family` 的 1 011/134 剪贴表（`src/data.py` 硬断言），不能在 1 385/360 池上复用；")
    lines.append("   本交付**不改**引擎（只新增），故 20 次拟合的入口是 P0 的一个新文件。")
    lines.append("2. **依赖清单哈希**：`environment_reference.yaml` 的 `authoritative_package_list.sha256` 仍为 `PENDING_3D`，")
    lines.append("   G-17 需要实测值（服务器 `pip freeze` 的 SHA-256）。")
    lines.append("3. **A-2 严格读法不成立**（见 preflight `supervision.rows`）：需负责人裁定 A-2 的口径；")
    lines.append("   该条是 §7.4.1 机制门槛的合取项之一，未裁定前门槛无法闭合。")
    lines.append("")
    lines.append("## S1 — 20 次折模型拟合（服务器，GPU）")
    lines.append("")
    lines.append("```bash")
    lines.append(f"# 环境（只碰物理 GPU 1；GPU 0 属 BMC2_GPU，Lock/engine 有隔离守卫）")
    lines.append(f"export CUDA_VISIBLE_DEVICES=1")
    lines.append(f"export PYTHONUNBUFFERED=1")
    lines.append(f"cd D:/TRUST_LUS")
    lines.append("")
    lines.append(f"# 逐条执行 joblist.csv 的 command 列（共 {len(jobs)} 条，DONE.json 幂等，可断点续跑）")
    lines.append("```")
    lines.append("")
    lines.append("| job_id | direction | family | fold | out_dir |")
    lines.append("|---|---|---|---|---|")
    for _, r in jobs.iterrows():
        lines.append(
            f"| `{r['job_id']}` | {r['direction']} | {r['family']} | {r['fold']} | `{r['out_dir']}` |"
        )
    lines.append("")
    lines.append("## S2 — 导出 per-unit OOF 预测（20 次拟合之后）")
    lines.append("")
    lines.append("每个 `out_dir` 需要一份含 `unit_id, patient_id, y, p3 或 p4,` 及 4 个时间视图概率的表：")
    lines.append("`oof_pred_<family>_<direction>_f<fold>.csv`。K=4 视图概率（B-02 偏移表）由 `src/features_v2.py` 定义。")
    lines.append("")
    lines.append("## S3 — 方向内 Router 拟合（CPU）")
    lines.append("")
    lines.append("```bash")
    lines.append("python src/router_v2.py --fit-input <merged_oof_DIRA.csv> --direction DIR-A --out-dir outputs/T1_router_oof_v1/router")
    lines.append("python src/router_v2.py --fit-input <merged_oof_DIRB.csv> --direction DIR-B --out-dir outputs/T1_router_oof_v1/router")
    lines.append("```")
    lines.append("")
    lines.append("产出 `router_oof_table_<DIR>.csv`（14 字段）+ `.meta.json`（τ、C、gate 系数、表哈希）。")
    lines.append("")
    lines.append("## S4 — 机制门槛记录（`Protocol §7.4.4`，每步必备字段）")
    lines.append("")
    lines.append("`step, direction, n_fits, gate_status, expert_choice_auroc, delta_r_ci_low, delta_auc_ci_low,`")
    lines.append("`capture_ratio, f_family_hat, r_violations, approved_by, timestamp_utc, artifact_sha256`")
    lines.append("")
    lines.append("门槛（合取）：`Expert-choice AUROC > 0.5`（两方向各自）∧ A-1…A-5 全过 ∧ 无 R-7 违例。")
    lines.append("")
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------- dry-run
def _synthetic_predictions(unit_id: str, y: int, n: int = 4):
    """Deterministic fake 4-view probabilities for one unit (no data, no model)."""
    seed = int(hashlib.sha256(unit_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    base3 = 0.5 + (0.08 if y == 0 else -0.08)
    base4 = 0.5 + (0.08 if y == 1 else -0.08)
    p3 = np.clip(base3 + rng.normal(0, 0.10, n) + rng.normal(0, 0.02, n), 0.01, 0.99)
    p4 = np.clip(base4 + rng.normal(0, 0.10, n) + rng.normal(0, 0.02, n), 0.01, 0.99)
    return p3.tolist(), p4.tolist()


def _stratified_subsample(man: pd.DataFrame, max_units: int) -> pd.DataFrame:
    """Label-stratified deterministic subsample (keeps both classes present)."""
    if len(man) <= max_units:
        return man.reset_index(drop=True)
    parts = []
    for y in (0, 1):
        g = man[man["label"] == y]
        if g.empty:
            continue
        k = max(1, int(round(max_units * len(g) / len(man))))
        if len(g) > k:
            idx = np.linspace(0, len(g) - 1, k).round().astype(int)
            g = g.iloc[idx]
        parts.append(g)
    out = pd.concat(parts).sort_values("unit_id").head(max_units)
    return out.reset_index(drop=True)


def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def dry_run(root: Path, data_root: Path | None, out_dir: Path | None, max_units: int = 150) -> dict:
    """Rehearse the whole OOF -> table pipeline on synthetic predictions (CPU)."""
    if "torch" in sys.modules:
        raise SystemExit("ABORT: dry-run must run without torch")
    plan_dir = Path(out_dir) if out_dir else Path(root) / PLAN_SUBDIR / "dryrun"
    plan_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for direction in ("DIR-A", "DIR-B"):
        frames = []
        for fold in FOLDS:
            man = LV.build_manifest(root, direction, fold, verify=True)
            frames.append(man)
        man = pd.concat(frames, ignore_index=True).sort_values("unit_id").reset_index(drop=True)
        man = _stratified_subsample(man, max_units)
        if man["label"].nunique() < 2:
            raise RuntimeError(f"dry-run: {direction} subsample lost a class")

        rows = []
        for i, r in man.iterrows():
            uid = str(r["unit_id"])
            y = int(r["label"])
            p3, p4 = _synthetic_predictions(uid, y)
            if i % 50 == 7:
                p3[2] = float("nan")   # exercise the B-12 missing-signal path
            rec = F.build_feature_row(
                uid, y, p3, p4, int(r["n_frames"]),
                {"patient_id": str(r["patient_id"]), "cluster_id": str(r["cluster_id"]),
                 "library": str(r["library"]), "direction": direction, "role": "source_oof",
                 "fold": int(r["fold"])},
            )
            rows.append(rec)
        kept, dropped, counts = F.partition_missing(rows)
        if not kept:
            raise RuntimeError(f"dry-run: no usable rows for {direction}")

        res = R.fit_direction(pd.DataFrame(kept), direction)
        table = res["table"]
        doc = R.write_table(
            table,
            plan_dir / f"router_oof_table_{direction}.csv",
            {
                "direction": direction,
                "dataset": res["dataset"],
                "tau": res["tau"],
                "C": res["C"],
                "tau_criterion": res["criterion"],
                "gate": res["gate"],
                "sensitivity_gbdt": res["sensitivity_gbdt"],
                "normalizer": res["normalizer"].params(),
                "grid_summary": {"n_cells": res["grid"]["n_cells"], "rule": res["grid"]["rule"]},
            },
        )
        # determinism: rebuild from the same kept rows and compare the content hash
        res2 = R.fit_direction(pd.DataFrame(kept), direction)
        stable = R.table_sha256(res2["table"]) == doc["table_sha256"]
        results[direction] = {
            "n_units_used": len(kept),
            "n_dropped": counts["n_dropped"],
            "n_crop_clamped": counts["n_crop_clamped"],
            "tau": res["tau"],
            "C": res["C"],
            "n_cells": res["grid"]["n_cells"],
            "expert_choice_auroc": res["grid"]["chosen"]["expert_choice_auroc"],
            "table_sha256": doc["table_sha256"],
            "file_sha256": doc["file_sha256"],
            "fields": doc["field_count"],
            "deterministic": bool(stable),
            "gate_backend": res["gate"]["backend"],
            "sensitivity": (res["sensitivity_gbdt"] or {}).get("backend"),
            "normalizer_sha256": res["normalizer"].params_sha256(),
        }

    # cross-direction assertion on the two produced tables
    ta = R.read_table(plan_dir / "router_oof_table_DIR-A.csv")
    tb = R.read_table(plan_dir / "router_oof_table_DIR-B.csv")
    R.assert_no_shared_labels(ta, tb)

    jobs = build_jobs(root, DEFAULT_TAG, PILOT_SEED)
    checks = [
        {"check": "dry-run: 20 jobs enumerated", "ok": len(jobs) == 20, "detail": len(jobs)},
        {"check": "dry-run: 14 fields in both tables", "ok": all(v["fields"] == 14 for v in results.values())},
        {"check": "dry-run: both tables deterministic", "ok": all(v["deterministic"] for v in results.values())},
        {"check": "dry-run: no cross-direction sharing", "ok": True},
        {"check": "dry-run: no DONE.json written", "ok": not any(plan_dir.rglob("DONE.json"))},
        {"check": "dry-run: no checkpoints written", "ok": not (Path(root) / "checkpoints").exists()
         or not any((Path(root) / "checkpoints").rglob("*.pt"))},
        {"check": "dry-run: torch never imported", "ok": "torch" not in sys.modules},
        {"check": "dry-run: tau in frozen grid", "ok": all(v["tau"] in R.TAU_GRID for v in results.values())},
        {"check": "dry-run: C in frozen grid", "ok": all(v["C"] in R.C_GRID for v in results.values())},
        {"check": "dry-run: expert-choice AUROC > 0.5 (synthetic signal)",
         "ok": all((v["expert_choice_auroc"] or 0) > 0.5 for v in results.values()),
         "detail": "synthetic data, not a scientific result"},
    ]
    summary = {
        "status": "PASS" if all(c["ok"] for c in checks) else "FAIL",
        "mode": "dry_run_synthetic",
        "plan_dir": str(plan_dir),
        "n_checks": len(checks),
        "n_failed": sum(0 if c["ok"] else 1 for c in checks),
        "checks": checks,
        "directions": results,
        "note": "CPU, synthetic probabilities, no data decode, no training, no DONE.json.",
    }
    (plan_dir / "dryrun_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("=" * 78)
    print(f"30_oof_pilot --dry-run: {summary['status']} ({summary['n_checks']} checks)")
    for d, v in results.items():
        print(
            f"  {d}: n={v['n_units_used']} dropped={v['n_dropped']} tau={v['tau']} C={v['C']} "
            f"cells={v['n_cells']} EC-AUROC={_fmt(v['expert_choice_auroc'])} "
            f"table={v['table_sha256'][:12]}"
        )
    print(f"  outputs: {plan_dir}")
    print("=" * 78)
    return summary


# ------------------------------------------------------------------ self-test
def _selftest(root: Path) -> int:
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        if not cond:
            print(f"[FAIL] {name} {detail}")

    jobs = build_jobs(root, DEFAULT_TAG, PILOT_SEED)
    check("20 jobs (2 families x 5 folds x 2 directions)", len(jobs) == 20, len(jobs))
    check("N_FITS == 20", N_FITS == 20)
    check(
        "job grid covers every (direction, family, fold)",
        set(zip(jobs["direction"], jobs["family"], jobs["fold"]))
        == {(d, f, k) for d in ("DIR-A", "DIR-B") for f in FAMILIES for k in FOLDS},
    )
    check("seed is the first formal seed (13)", set(jobs["seed"]) == {PILOT_SEED} == {13})
    check(
        "each direction has its own stage tag (no shared dir)",
        set(jobs[jobs.direction == "DIR-A"]["stage_tag"]) == {f"{DEFAULT_TAG}_dirA"}
        and set(jobs[jobs.direction == "DIR-B"]["stage_tag"]) == {f"{DEFAULT_TAG}_dirB"},
    )
    check(
        "DONE.json path matches src/engine.py convention",
        all(
            r["done_json"].replace("\\", "/").endswith(
                f"outputs/{r['stage_tag']}/{r['family']}/seed{r['seed']}/fold{r['fold']}/DONE.json"
            )
            for _, r in jobs.iterrows()
        ),
    )
    check(
        "train+val units partition the direction's source pool",
        all(
            int(r["n_train_units"]) + int(r["n_val_units"])
            == (1385 if r["direction"] == "DIR-A" else 360)
            for _, r in jobs.iterrows()
        ),
        jobs[["direction", "fold", "n_train_units", "n_val_units"]].head(2).to_dict("records"),
    )
    check(
        "per-fold val sizes are the frozen counts",
        list(jobs[(jobs.direction == "DIR-A") & (jobs.family == "M3")].sort_values("fold")["n_val_units"])
        == [293, 254, 269, 284, 285]
        and list(jobs[(jobs.direction == "DIR-B") & (jobs.family == "M3")].sort_values("fold")["n_val_units"])
        == [74, 71, 74, 72, 69],
    )
    check("no duplicate job ids", not jobs["job_id"].duplicated().any())
    check(
        "every job is explicitly blocked on the adapter (no silent 'runnable')",
        jobs["blocked_on"].str.contains("engine_v2").all(),
    )
    check(
        "commands use the frozen engine entry + per-direction out-tag",
        all(TRAIN_SCRIPT in c and "--out-tag" in c and "--pre-lock-run" in c for c in jobs["command"]),
    )

    # tag freshness logic
    probe = root / "outputs" / f"{DEFAULT_TAG}_dirA"
    if probe.exists():
        print(f"  [warn] {probe} already exists; freshness check exercised in the negative only")
        check("assert_tag_fresh accepts nothing that exists", True)
    else:
        assert_tag_fresh(root, DEFAULT_TAG)
        check("assert_tag_fresh passes when the tags are absent", True)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "outputs" / f"{DEFAULT_TAG}_dirB").mkdir(parents=True)
        try:
            assert_tag_fresh(tmp, DEFAULT_TAG)
            check("assert_tag_fresh refuses an existing dir", False)
        except SystemExit:
            check("assert_tag_fresh refuses an existing dir", True)

    # no-training guarantee, checked on the AST (textual greps would match the checks)
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported = set()
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.add(node.module.split(".")[0])
            imported |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name:
                called.add(name)
    check("driver imports no torch", not any(m == "torch" or m.startswith("torch.") for m in imported),
          sorted(m for m in imported if "torch" in m))
    check("driver imports no training engine", "src.engine" not in imported and "engine" not in called,
          sorted(imported))
    check("driver never calls train_one_fold", "train_one_fold" not in called)
    check("driver imports the frozen feature/router modules", {"src.features_v2", "src.router_v2"} <= imported)
    check("torch is not in sys.modules", "torch" not in sys.modules)

    # synthetic prediction helper is deterministic and finite
    a1, b1 = _synthetic_predictions("Pt09.19_z01", 1)
    a2, b2 = _synthetic_predictions("Pt09.19_z01", 1)
    check("synthetic predictions deterministic", a1 == a2 and b1 == b2)
    check("synthetic predictions are 4 views in [0,1]",
          len(a1) == 4 and all(0.0 <= v <= 1.0 for v in a1 + b1))

    # features -> table smoke path without any data decode
    rec = F.build_feature_row("u", 1, a1, b1, 40, {"direction": "DIR-A", "role": "source_oof",
                                                   "library": "openpocus", "patient_id": "p",
                                                   "cluster_id": "c", "fold": 1})
    check("smoke row status OK", rec["status"] == F.STATUS_OK)
    check("smoke row has 5 finite features", all(np.isfinite(rec[n]) for n in F.FEATURE_NAMES))

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print("=" * 78)
    print(f"30_oof_pilot self-test: {n_pass}/{len(checks)} PASS")
    print(f"fits={N_FITS} = {len(FAMILIES)} families x {len(FOLDS)} folds x 2 directions")
    print("=" * 78)
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print(json.dumps({"status": "FAIL", "failed": failed}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", "n_checks": len(checks)}, ensure_ascii=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="TRUST-LUS Router OOF mechanism pilot (20 fits)")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--data-root", type=Path, default=None, help="project root holding data/ (media probe)")
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--seed", type=int, default=PILOT_SEED)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None, help="dry-run output directory")
    ap.add_argument("--max-units", type=int, default=150, help="dry-run unit cap per direction")
    ap.add_argument("--force", action="store_true", help="overwrite an existing plan dir")
    args = ap.parse_args(argv)
    root = args.root.resolve()

    if args.self_test:
        return _selftest(root)
    if args.dry_run:
        s = dry_run(root, args.data_root, args.out_dir, max_units=args.max_units)
        return 0 if s["status"] == "PASS" else 1

    summary = write_plan(root, args.tag, args.seed, args.data_root, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n*** NOT TRAINING: 20 fits are enumerated in joblist.csv, not executed. ***")
    print("*** No DONE.json, no checkpoints, no GPU was touched.                    ***")
    return 0 if summary["preflight_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
