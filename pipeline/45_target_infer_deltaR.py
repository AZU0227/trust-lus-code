#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRUST-LUS V4.2 — 目标域推理 + ΔR_D + 簇级配对 bootstrap（step-2 准备件）, v1.

本文件是 `Protocol_v2_v1_1_FINAL.md` §7.4 第 2 步的**推理与统计**部分：
在**不训练任何模型**的前提下，把已拟合好的 5 折源域专家模型套用到**完全留出的
目标域**，重建源域 Router（τ、C 固定），输出逐单元 α，并给出逐方向
`ΔR_D = AUC_D(Router) − AUC_D(m_fixed)` 及其患者/簇级配对 bootstrap 95% CI。

已拍板的两个前提（见 `implementation_decisions`，写入报告与 JSON）
------------------------------------------------------------------
1. **不做全量重训**：源模型 = 同方向 5 个 `best.pt` 在目标域上逐视图预测取均值
   （= 主聚合口径 A：per-seed 5 折集成；`A_review.md` S2-4）。与 OOF 表口径一致，
   `p_expert = p_v1`（Lock §7 B-02）。
2. 目标域推理 GPU 已获批准（≈0.3–0.6 GPU·h，仅物理 GPU 1；负责人 2026-10-26）。

冻结依据（逐条引用，不在本脚本内新定义阈值）
------------------------------------------
* ΔR / m_fixed：`Protocol_v2_v1_1_FINAL.md` §5.1 L-09；`Lock_R1_v2_draft_v1.1.md`
  §3.1 L-09'（L193-L199）。
* bootstrap：Protocol §5.5 L-12（簇级配对重采样）、L-13（2000 有效重复、单类
  invalid 不计数、百分位 95% CI、持久化计数）、L-14（禁视频级 / best-seed /
  正态近似 / BCa / 单侧）、L-15（master seed 20260927、新增分析 `spawn_key=(1,idx)`）、
  L-16（门控双区间；本件只做**固定门控**主口径，见“未做项”）。
* 随机源索引：Protocol 附录 A.3 R-06.2；Lock v1.1 L-15' 的 9 个互异索引中，
  本件用 `router_pilot_dirA=(1,1)` / `router_pilot_dirB=(1,2)`。
* Gate：Lock §7 B-05（5 维 L2 逻辑回归，α=sigmoid(w·z+b)）；B-01（τ/C 网格；
  本件按已冻结的 gate 报告固定 τ=0.5、C=0.01，不重选）。
* 归一化：`src/norm_v2.py`（Median/IQR 只由源域 OOF 估计；目标域只允许套用，
  禁止用目标域统计重新标定）；Protocol §5.6 L-43(a)-(d)。
* 特征：`src/features_v2.py`（C_3,C_4,T_3,T_4,D_34；K=4 偏移表；T=Std ddof=1）。
* 缺信号：Lock §7 B-12（排除并计数，✗ 不填补）。
* R-7：Protocol §7.2 R-7 / §5.6 L-43（目标域标签不得进入 Gate 拟合、选择、
  阈值、归一化或校准）；本脚本在结构上（`alpha_for_target` 无 y 参数）与运行时
  （自检中打乱目标域 y 后 α 逐位不变）双重验证。

边界
----
* **✗ 不训练**：不建 optimizer、不 backward、不写 checkpoint / DONE.json。
* **✗ 不改锁**：只读 `src/*`、`scripts/98_loader_v2.py`、lock/config、OOF 表与
  gate 报告；只写 `outputs/T1_router_oof_v1/target/`。
* **零 GPU 本机**：`--self-test` 只用 numpy/pandas + 合成数据；本机不跑推理。
* 只做**机械判定**：两方向 ΔR 的 95% CI 下界同时 > 0 → GO，否则 NO-GO；
  ✗ 不写科学结论（§5.4 GO-A 另需 ΔR_macro 区间，本件未算，见 §8）。

CLI
---
    python scripts/45_target_infer_deltaR.py --self-test          # CPU, synthetic
    python scripts/45_target_infer_deltaR.py --root D:\\TRUST_LUS --preflight
    python scripts/45_target_infer_deltaR.py --root D:\\TRUST_LUS --direction ALL
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import engine_v2 as E  # noqa: E402
from src import features_v2 as F  # noqa: E402
from src import norm_v2 as N  # noqa: E402
from src import router_v2 as R  # noqa: E402

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_NAME = "scripts/45_target_infer_deltaR.py"
VERSION = "v1"
KIND = "target_inference_plus_deltaR"

TAG = "T1_router_oof_v1"
PILOT_SEED = 13
FAMILIES = ("M3", "M4")
FOLDS = (1, 2, 3, 4, 5)

# 已冻结 / 已拍板的常数（✗ 不暴露成 CLI，防止被顺手改掉）
TAU_FIXED = 0.5
C_FIXED = 0.01
MASTER_SEED = 20260927
BOOTSTRAP_VALID = 2000
MAX_BOOTSTRAP_ATTEMPTS = 200000
CI_PERCENTILES = (2.5, 97.5)
BATCH_SIZE = 2
AMP = True

# Lock v1.1 L-15' / R-06.2：router_pilot_dirA = (1,1)，router_pilot_dirB = (1,2)
SUBSTREAM_KEY = {"DIR-A": (1, 1), "DIR-B": (1, 2)}
SUBSTREAM_INDEX = {"DIR-A": 1, "DIR-B": 2}

SOURCE_UNITS = {"DIR-A": 1385, "DIR-B": 360}
SOURCE_CLUSTERS = {"DIR-A": 185, "DIR-B": 63}
TARGET_UNITS = {"DIR-A": 360, "DIR-B": 1385}
TARGET_CLUSTERS = {"DIR-A": 63, "DIR-B": 185}

TARGET_CSV_FIELDS = (
    "unit_id",
    "patient_id",
    "cluster_id",
    "y",
    "p3",
    "p4",
    "C3",
    "C4",
    "T3",
    "T4",
    "disagreement",
    "alpha",
    "p_route",
    "fold_ensemble_n",
)

FEATURE_TO_FRAME = {
    "C_3": "C3",
    "C_4": "C4",
    "T_3": "T3",
    "T_4": "T4",
    "D_34": "disagreement",
}

# 数值容差（实现层，不是科学阈值）
NORM_PARAM_TOL = 1e-9        # median / IQR：同一批 OOF 值应逐位一致
GATE_COEF_TOL = 1e-4         # gate 系数：容忍 sklearn/numpy 后端收敛差；数据漂移会大得多
TABLE_SELFCHECK_TOL = 1e-12  # 14 字段表的内部恒等式

PRODUCT_NAMES = (
    "target_predictions_DIR-A.csv",
    "target_predictions_DIR-B.csv",
    "target_raw_views_DIR-A.npz",
    "target_raw_views_DIR-B.npz",
    "deltaR_report.md",
    "deltaR_report.json",
    "SHA256SUMS_target.txt",
)


class PreflightError(RuntimeError):
    """输入不满足契约（缺文件、计数漂移、Gate 报告不一致等）。"""


class CheckpointHashError(PreflightError):
    """best.pt 与 run_metadata.json 记录的哈希不符（立即失败，✗ 不得重算凑上）。"""


class GateReportError(PreflightError):
    """gate 报告缺失 / τ、C 不符 / 无法重建。"""


# ------------------------------------------------------------------ small utils
def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path) -> str:
    return E.sha256_file(Path(path))


def _jf(value, nd: int = 12):
    """JSON 友好的 float（None/NaN/inf -> None，其余 round）。"""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, nd)


def _force_utf8_stdio() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _jsonable(obj):
    """递归把 numpy / Path 转成 JSON 原生类型（避免 default=str 把数字变字符串）。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _as_int(value, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"{what} 不是整数: {value!r}") from exc


def _as_float(value, what: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"{what} 不是数值: {value!r}") from exc
    if not np.isfinite(v):
        raise PreflightError(f"{what} 不是有限值: {value!r}")
    return v


# ------------------------------------------------------------ static discipline
FORBIDDEN_CALL_ATTRS = {"backward", "train"}
FORBIDDEN_CALL_NAMES = {"train_one_fold", "save_checkpoint", "build_optimizer"}


def ast_no_training(path: Path = SCRIPT_PATH) -> list:
    """机械复核：本脚本不训练、不在模块层 import torch/sklearn/yaml。

    返回问题列表（空 = 通过）。允许函数体内 `import torch`（推理），但模块顶层
    不允许；`model.eval()` 不是训练；`torch.save` / `torch.optim` 一律禁止。
    """
    problems = []
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        for name in names:
            if name in {"torch", "sklearn", "yaml"}:
                problems.append(f"module-level import of {name!r}")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALL_NAMES:
            problems.append(f"call to {func.id}()")
        if isinstance(func, ast.Attribute):
            if func.attr in FORBIDDEN_CALL_ATTRS:
                problems.append(f"call to .{func.attr}()")
            if func.attr == "save" and isinstance(func.value, ast.Name) and func.value.id == "torch":
                problems.append("call to torch.save()")
    return problems


# ---------------------------------------------------------------- layout / inputs
def target_manifest(layout, direction: str) -> tuple:
    """目标域单元表：library == DIRECTIONS[direction]['target'] 的全部可评估且带标签单元。

    目标域 = 另一个库的**全部单元**（Protocol §1.2 L-05：单源迁移，另一个域完全留出）。
    `evaluable == True` 且 `label_bline` 非空（B-12：缺帧/缺标签只排除、不填补）。
    """
    if direction not in N.DIRECTIONS:
        raise PreflightError(f"unknown direction {direction!r}")
    target_lib = N.DIRECTIONS[direction]["target"]
    try:
        units = E.LV().load_units(layout.table_root)
    except E.ContractError as exc:
        raise PreflightError(f"{direction}: 单元表契约失败: {exc}") from exc
    sel = units[
        (units["library"] == target_lib) & units["evaluable"] & units["label"].notna()
    ].copy()
    n_units = int(len(sel))
    n_clusters = int(sel["cluster_id"].nunique())
    if n_units != TARGET_UNITS[direction]:
        raise PreflightError(
            f"{direction} 目标域单元数 {n_units} != 冻结值 {TARGET_UNITS[direction]} "
            f"(library={target_lib})"
        )
    if n_clusters != TARGET_CLUSTERS[direction]:
        raise PreflightError(
            f"{direction} 目标域簇数 {n_clusters} != 冻结值 {TARGET_CLUSTERS[direction]}"
        )
    man = pd.DataFrame(
        {
            "unit_id": sel["unit_id"].astype(str).to_numpy(),
            "patient_id": sel["patient_id"].astype(str).to_numpy(),
            "cluster_id": sel["cluster_id"].astype(str).to_numpy(),
            "library": str(target_lib),
            "label": sel["label"].astype(int).to_numpy(),
            "n_frames": sel["n_frames"].astype(int).to_numpy(),
            "fold": sel["fold"].astype(int).to_numpy(),
        }
    ).sort_values("unit_id").reset_index(drop=True)
    if man["unit_id"].duplicated().any():
        raise PreflightError(f"{direction} 目标域 unit_id 重复")
    if man["n_frames"].min() < F.MIN_FRAMES:
        bad = man[man["n_frames"] < F.MIN_FRAMES][["unit_id", "n_frames"]].head(5)
        raise PreflightError(f"{direction} 目标域存在 <16 帧单元（B-02 排除）:\n{bad}")
    try:
        man = E.resolve_media(layout.media_root, man, strict=True)
    except E.ContractError as exc:
        raise PreflightError(f"{direction}: 目标域媒体解析失败: {exc}") from exc
    return man, {
        "library": target_lib,
        "n_units": n_units,
        "n_clusters": n_clusters,
        "n_positive": int((man["label"] == 1).sum()),
        "n_negative": int((man["label"] == 0).sum()),
        "unit_set_sha256": N.unit_set_sha256(man["unit_id"]),
        "cluster_set_sha256": N.unit_set_sha256(man["cluster_id"]),
        "n_crop_clamped": int((man["n_frames"] < F.CROP_CLAMPED_MAX_FRAMES).sum()),
        "media_kind": sorted(set(man["media_kind"].astype(str))),
    }


def verify_fold_artifact(
    ckpt_path: Path,
    meta_path: Path,
    done_path: Path,
    direction: str,
    family: str,
    fold: int,
    seed: int = PILOT_SEED,
) -> dict:
    """核对 1 个折模型：best.pt 哈希必须等于 run_metadata.json 里的记录。

    任一不符 → `CheckpointHashError`（立即失败；✗ 不重算哈希凑上、✗ 不改 metadata）。
    """
    for label, p in (("best.pt", ckpt_path), ("run_metadata.json", meta_path)):
        if not Path(p).exists():
            raise PreflightError(f"{family} {direction} f{fold}: 缺 {label}（{p}）")
    meta = _read_json(Path(meta_path))
    done = _read_json(Path(done_path)) if Path(done_path).exists() else None

    for key, expected in (
        ("family", family),
        ("direction", direction),
        ("fold", int(fold)),
        ("seed", int(seed)),
        ("status", "COMPLETE"),
    ):
        got = meta.get(key)
        if str(got) != str(expected):
            raise CheckpointHashError(
                f"{family} {direction} f{fold}: run_metadata {key}={got!r} != {expected!r}"
            )
    if done is not None:
        for key, expected in (
            ("family", family),
            ("direction", direction),
            ("fold", int(fold)),
            ("seed", int(seed)),
            ("status", "COMPLETE"),
        ):
            got = done.get(key)
            if str(got) != str(expected):
                raise CheckpointHashError(
                    f"{family} {direction} f{fold}: DONE.json {key}={got!r} != {expected!r}"
                )

    expected_sha = str(meta.get("best_checkpoint_sha256", "")).strip()
    if len(expected_sha) != 64:
        raise CheckpointHashError(
            f"{family} {direction} f{fold}: run_metadata 缺 best_checkpoint_sha256"
        )
    actual_sha = _sha256(ckpt_path)
    if actual_sha != expected_sha:
        raise CheckpointHashError(
            f"{family} {direction} f{fold}: best.pt 哈希不符 "
            f"(actual={actual_sha} expected={expected_sha}) — 立即失败，✗ 不得重算凑上"
        )
    oof_sha = str(meta.get("oof_pred_sha256", "")).strip()
    if done is not None and oof_sha and str(done.get("oof_pred_sha256", "")).strip() != oof_sha:
        raise CheckpointHashError(
            f"{family} {direction} f{fold}: DONE.json 与 run_metadata 的 oof_pred_sha256 不一致"
        )
    return {
        "family": family,
        "direction": direction,
        "fold": int(fold),
        "seed": int(seed),
        "best_pt": str(ckpt_path),
        "best_pt_sha256": actual_sha,
        "run_metadata": str(meta_path),
        "run_metadata_sha256": _sha256(meta_path),
        "done_json": (str(done_path) if done is not None else None),
        "best_epoch": meta.get("best_epoch"),
        "config_sha256": meta.get("config_sha256"),
        "data_version_sha256": meta.get("data_version_sha256"),
        "stage_tag": meta.get("stage_tag"),
        "oof_pred_sha256": oof_sha,
    }


def verify_all_folds(layout, direction: str) -> list:
    """20 个折模型（本方向 10 个）逐条核对；同家族 5 折的 config/data 版本必须一致。"""
    stage = E.stage_tag(TAG, direction)
    records = []
    for family in FAMILIES:
        for fold in FOLDS:
            base_out = layout.outputs_dir / stage / family / f"seed{PILOT_SEED}" / f"fold{fold}"
            base_ck = layout.checkpoints_dir / stage / family / f"seed{PILOT_SEED}" / f"fold{fold}"
            records.append(
                verify_fold_artifact(
                    base_ck / "best.pt",
                    base_out / "run_metadata.json",
                    base_out / "DONE.json",
                    direction,
                    family,
                    fold,
                )
            )
    for family in FAMILIES:
        sub = [r for r in records if r["family"] == family]
        for key in ("config_sha256", "data_version_sha256"):
            vals = sorted(set(str(r[key]) for r in sub))
            if len(vals) != 1:
                raise CheckpointHashError(
                    f"{family} {direction}: 5 折的 {key} 不一致 {vals}（混合了不同 run）"
                )
    return records


def load_source_table(layout, direction: str) -> tuple:
    """读取 14 字段源域 OOF 表 + sidecar，并核对文件哈希与 τ/C。"""
    router_dir = layout.outputs_dir / TAG / "router"
    path = router_dir / f"router_oof_table_{direction}.csv"
    meta_path = router_dir / f"router_oof_table_{direction}.meta.json"
    if not path.exists():
        raise PreflightError(f"缺源 OOF 表：{path}（先跑 scripts/40_merge_oof_and_gates.py）")
    if not meta_path.exists():
        raise PreflightError(f"缺源 OOF 表 sidecar：{meta_path}")
    try:
        table = R.read_table(path)
    except Exception as exc:
        raise PreflightError(f"{direction}: 源 OOF 表不是 14 字段契约: {exc}") from exc
    if len(table) != SOURCE_UNITS[direction]:
        raise PreflightError(
            f"{direction} 源 OOF 表 {len(table)} 行 != 冻结 {SOURCE_UNITS[direction]}；"
            "✗ 可能误用了合并表或错方向"
        )
    ds = sorted(set(table["dataset"].astype(str)))
    src_lib = N.DIRECTIONS[direction]["source"]
    if ds != [src_lib]:
        raise PreflightError(f"{direction} 源表 dataset={ds} != {src_lib!r}")
    if table["video_id"].duplicated().any():
        raise PreflightError(f"{direction} 源表 video_id 重复")
    if int(table["y"].nunique()) < 2:
        raise PreflightError(f"{direction} 源表单类标签，无法选 m_fixed")

    meta = _read_json(meta_path)
    actual_file = _sha256(path)
    if str(meta.get("file_sha256", "")).strip() != actual_file:
        raise PreflightError(
            f"{direction} 源表文件哈希与 sidecar 不符 "
            f"({actual_file} != {meta.get('file_sha256')})"
        )
    if str(meta.get("table_sha256", "")).strip() != R.table_sha256(table):
        raise PreflightError(f"{direction} 源表内容哈希与 sidecar 不符")
    if str(meta.get("direction")) != direction:
        raise PreflightError(f"{direction} sidecar.direction={meta.get('direction')!r}")
    tau = _as_float(meta.get("tau"), f"{direction} sidecar.tau")
    cval = _as_float(meta.get("C"), f"{direction} sidecar.C")
    if abs(tau - TAU_FIXED) > 1e-12 or abs(cval - C_FIXED) > 1e-12:
        raise PreflightError(
            f"{direction} sidecar τ/C=({tau},{cval}) != 已拍板 ({TAU_FIXED},{C_FIXED})；"
            "✗ 不得在本步改 τ/C"
        )
    return table, {
        "path": str(path),
        "file_sha256": actual_file,
        "table_sha256": str(meta.get("table_sha256")),
        "n_rows": int(len(table)),
        "tau": tau,
        "C": cval,
        "tau_criterion": meta.get("tau_criterion"),
        "sidecar_sha256": _sha256(meta_path),
    }


def load_gate_report(layout, direction: str) -> tuple:
    """读取 pilot_gate_report.json 中本方向的 gate / normalizer，并核对 τ=0.5、C=0.01。"""
    path = layout.outputs_dir / TAG / "router" / "pilot_gate_report.json"
    if not path.exists():
        raise GateReportError(f"缺 gate 报告：{path}（第 1 步产物；本步不重选 τ/C）")
    doc = _read_json(path)
    if "directions" not in doc or direction not in doc["directions"]:
        raise GateReportError(f"gate 报告里没有 directions.{direction}")
    dr = doc["directions"][direction]
    info = assert_gate_report(dr, direction)
    info.update(
        {
            "path": str(path),
            "sha256": _sha256(path),
            "report_verdict": (doc.get("verdict") or {}).get("label"),
        }
    )
    return doc, info


def assert_gate_report(dr: dict, direction: str) -> dict:
    """gate 报告单方向块的结构 + τ/C 断言（纯函数，供 self-test 复用）。"""
    if not isinstance(dr, dict):
        raise GateReportError(f"{direction}: gate 报告块不是 dict")
    for key in ("tau", "C", "gate", "normalizer"):
        if key not in dr:
            raise GateReportError(f"{direction}: gate 报告块缺 {key!r}")
    tau = _as_float(dr["tau"], f"{direction}.tau")
    cval = _as_float(dr["C"], f"{direction}.C")
    if abs(tau - TAU_FIXED) > 1e-12:
        raise GateReportError(
            f"{direction}: gate 报告 τ={tau} != 已拍板 {TAU_FIXED}（✗ 不重选 τ）"
        )
    if abs(cval - C_FIXED) > 1e-12:
        raise GateReportError(
            f"{direction}: gate 报告 C={cval} != 已拍板 {C_FIXED}（✗ 不重选 C）"
        )
    gate = dr["gate"]
    if not isinstance(gate, dict) or str(gate.get("status")) != "ok":
        raise GateReportError(
            f"{direction}: gate.status={gate.get('status') if isinstance(gate, dict) else gate!r}"
            "（n/a 或缺失时无法输出 α）"
        )
    coef = np.asarray(gate.get("coef") or [], dtype=float)
    if coef.shape != (F.N_FEATURES,):
        raise GateReportError(
            f"{direction}: gate.coef 长度 {coef.size} != {F.N_FEATURES}"
        )
    if not np.all(np.isfinite(coef)):
        raise GateReportError(f"{direction}: gate.coef 含非有限值")
    intercept = _as_float(gate.get("intercept"), f"{direction}.gate.intercept")
    if "C" in gate and abs(_as_float(gate.get("C"), f"{direction}.gate.C") - cval) > 1e-12:
        raise GateReportError(f"{direction}: gate.C 与 direction.C 不一致")
    crit = dr.get("criteria") or {}
    if "soft_target" in crit and str(crit["soft_target"]) != R.DEFAULT_SOFT_TARGET:
        raise GateReportError(
            f"{direction}: soft_target={crit['soft_target']!r} != {R.DEFAULT_SOFT_TARGET!r}"
            "（主口径要求 q4 作 mean-1 权重）"
        )
    norm = dr["normalizer"]
    if not isinstance(norm, dict):
        raise GateReportError(f"{direction}: normalizer 不是 dict")
    if str(norm.get("direction")) != direction:
        raise GateReportError(f"{direction}: normalizer.direction={norm.get('direction')!r}")
    if "median" not in norm or "iqr" not in norm:
        raise GateReportError(f"{direction}: normalizer 缺 median/iqr")
    median = np.asarray(norm["median"], dtype=float)
    iqr = np.asarray(norm["iqr"], dtype=float)
    if median.shape != (F.N_FEATURES,) or iqr.shape != (F.N_FEATURES,):
        raise GateReportError(f"{direction}: normalizer median/iqr 维度不是 5")
    if not np.all(np.isfinite(median)) or not np.all(np.isfinite(iqr)):
        raise GateReportError(f"{direction}: normalizer 含非有限值")
    if np.any(iqr < 0):
        raise GateReportError(f"{direction}: normalizer IQR 为负")
    return {
        "direction": direction,
        "tau": tau,
        "C": cval,
        "gate_backend": gate.get("backend"),
        "gate_status": "ok",
        "coef": [float(v) for v in coef],
        "intercept": intercept,
        "alpha_mean_source": _jf(gate.get("alpha_mean")),
        "gate_n_rows": gate.get("n_rows"),
        "normalizer": {
            "direction": direction,
            "source_library": norm.get("source_library"),
            "median": [float(v) for v in median],
            "iqr": [float(v) for v in iqr],
            "eps": _as_float(norm.get("eps", N.EPS), f"{direction}.normalizer.eps"),
            "n_units": norm.get("n_units"),
            "n_clusters": norm.get("n_clusters"),
            "unit_set_sha256": norm.get("unit_set_sha256"),
            "cluster_set_sha256": norm.get("cluster_set_sha256"),
            "degenerate_features": list(norm.get("degenerate_features") or []),
        },
    }


def source_rows_from_table(table: pd.DataFrame, direction: str) -> pd.DataFrame:
    """把 14 字段表还原成 Gate 拟合所需的行（含 direction/role/library），并自检恒等式。"""
    src_lib = N.DIRECTIONS[direction]["source"]
    y = table["y"].to_numpy(dtype=int)
    p3 = table["p3"].to_numpy(dtype=float)
    p4 = table["p4"].to_numpy(dtype=float)
    conf3 = table["conf3"].to_numpy(dtype=float)
    conf4 = table["conf4"].to_numpy(dtype=float)
    temp3 = table["temp3"].to_numpy(dtype=float)
    temp4 = table["temp4"].to_numpy(dtype=float)
    dis = table["disagreement"].to_numpy(dtype=float)
    l3 = table["loss3"].to_numpy(dtype=float)
    l4 = table["loss4"].to_numpy(dtype=float)
    if not np.allclose(conf3, np.maximum(p3, 1.0 - p3), atol=TABLE_SELFCHECK_TOL, rtol=0):
        raise PreflightError(f"{direction}: conf3 != max(p3,1-p3)（表内口径不一致）")
    if not np.allclose(conf4, np.maximum(p4, 1.0 - p4), atol=TABLE_SELFCHECK_TOL, rtol=0):
        raise PreflightError(f"{direction}: conf4 != max(p4,1-p4)（表内口径不一致）")
    if not np.allclose(dis, np.abs(p3 - p4), atol=TABLE_SELFCHECK_TOL, rtol=0):
        raise PreflightError(f"{direction}: disagreement != |p3-p4|")
    if not np.allclose(l3, (p3 - y) ** 2, atol=TABLE_SELFCHECK_TOL, rtol=0):
        raise PreflightError(f"{direction}: loss3 != (p3-y)^2")
    if not np.allclose(l4, (p4 - y) ** 2, atol=TABLE_SELFCHECK_TOL, rtol=0):
        raise PreflightError(f"{direction}: loss4 != (p4-y)^2")
    rows = pd.DataFrame(
        {
            "unit_id": table["video_id"].astype(str).to_numpy(),
            "patient_id": table["patient_id"].astype(str).to_numpy(),
            "direction": direction,
            "role": "source_oof",
            "library": src_lib,
            "y": y,
            "p3": p3,
            "p4": p4,
            "C_3": conf3,
            "C_4": conf4,
            "T_3": temp3,
            "T_4": temp4,
            "D_34": dis,
            "l3": l3,
            "l4": l4,
        }
    )
    return rows


def fit_router_from_source(table: pd.DataFrame, direction: str) -> dict:
    """用源 OOF 表重建 Router（normalizer + L2 logistic Gate），τ/C 固定。

    只看到源域行：`source_rows_from_table` 生成 role=source_oof、direction=本方向的行；
    `norm_v2.fit` 与 `router_v2.fit_l2_logistic` 都会拒绝其它 direction / role。
    """
    rows = source_rows_from_table(table, direction)
    norm = N.fit(rows, direction)
    X = norm.transform(rows)
    q4 = R.soft_target(rows["l3"], rows["l4"], TAU_FIXED)
    w_gate = R.soft_sample_weights(q4, R.DEFAULT_SOFT_TARGET)
    y_hard = R.hard_target(rows["l3"], rows["l4"])
    if np.unique(y_hard).size < 2:
        raise GateReportError(f"{direction}: 源域硬标签单类，Gate 无法拟合")
    gate = R.fit_l2_logistic(X, y_hard, C_FIXED, sample_weight=w_gate)
    gate["status"] = "ok"
    gate["tau"] = TAU_FIXED
    gate["C"] = C_FIXED
    gate["n_rows"] = int(len(rows))
    return {
        "norm": norm,
        "gate": gate,
        "rows": rows,
        "q4": q4,
        "y_hard": y_hard,
        "alpha_mean": float(np.mean(R.predict_alpha(X, gate))),
    }


def compare_gate_rebuild(rebuild: dict, report: dict) -> dict:
    """核对“源表重建的 Router”与 gate 报告是否一致；超容差即失败。"""
    norm = rebuild["norm"]
    gate = rebuild["gate"]
    rnorm = report["normalizer"]
    med = np.asarray(rnorm["median"], dtype=float)
    iqr = np.asarray(rnorm["iqr"], dtype=float)
    coef = np.asarray(report["coef"], dtype=float)
    d_med = float(np.max(np.abs(np.asarray(norm.median, dtype=float) - med)))
    d_iqr = float(np.max(np.abs(np.asarray(norm.iqr, dtype=float) - iqr)))
    d_coef = float(np.max(np.abs(np.asarray(gate["coef"], dtype=float) - coef)))
    d_b = float(abs(float(gate["intercept"]) - float(report["intercept"])))
    if rnorm.get("unit_set_sha256") and str(rnorm["unit_set_sha256"]) != str(norm.unit_set_sha256):
        raise GateReportError(
            f"{report['direction']}: normalizer 的源域单元集哈希与 gate 报告不符"
        )
    if rnorm.get("n_units") is not None and int(rnorm["n_units"]) != int(norm.n_units):
        raise GateReportError(
            f"{report['direction']}: normalizer.n_units={rnorm['n_units']} != 重建 {norm.n_units}"
        )
    if report.get("gate_n_rows") is not None and int(report["gate_n_rows"]) != int(gate["n_rows"]):
        raise GateReportError(
            f"{report['direction']}: gate.n_rows={report['gate_n_rows']} != 重建 {gate['n_rows']}"
        )
    if d_med > NORM_PARAM_TOL or d_iqr > NORM_PARAM_TOL:
        raise GateReportError(
            f"{report['direction']}: 重建 normalizer 与 gate 报告不符 "
            f"(max|Δmedian|={d_med:.3e}, max|Δiqr|={d_iqr:.3e}) — 源表或报告版本漂移"
        )
    if d_coef > GATE_COEF_TOL or d_b > GATE_COEF_TOL:
        raise GateReportError(
            f"{report['direction']}: 重建 gate 系数与 gate 报告不符 "
            f"(max|Δcoef|={d_coef:.3e}, |Δb|={d_b:.3e}) — ✗ 不得静默改用另一套系数"
        )
    return {
        "normalizer_max_abs_diff": {"median": _jf(d_med, 15), "iqr": _jf(d_iqr, 15)},
        "gate_max_abs_diff": {"coef": _jf(d_coef, 15), "intercept": _jf(d_b, 15)},
        "tolerance": {"normalizer": NORM_PARAM_TOL, "gate": GATE_COEF_TOL},
        "rebuild_backend": gate.get("backend"),
        "report_backend": report.get("gate_backend"),
        "consistent": True,
    }


# ------------------------------------------------------------- target transform
def transform_target_features(norm, target_df: pd.DataFrame, target_library: str) -> np.ndarray:
    """把**已冻结的源域 normalizer** 套用到目标域特征（✗ 不重估、✗ 不用目标域统计）。

    与 `RobustNormalizer.transform` 同一仿射形式；但后者带 `role == source_oof`
    隔离断言（目标域行不能走那条路径），故这里显式重写同一公式并加目标域断言。
    自检中与 `norm.transform(源域行)` 逐位比对，防止两处公式漂移。
    """
    if str(target_library) == str(norm.source_library):
        raise PreflightError("目标域库与 normalizer 的源域库相同：方向/库绑定错误")
    x = target_df[list(FEATURE_TO_FRAME.values())].to_numpy(dtype=float)
    if not np.all(np.isfinite(x)):
        raise PreflightError("目标域特征含 NaN/inf（B-12：✗ 不填补）")
    med = np.asarray(norm.median, dtype=float)
    iqr = np.asarray(norm.iqr, dtype=float)
    z = (x - med) / (iqr + float(norm.eps))
    for idx, name in enumerate(F.FEATURE_NAMES):
        if name in norm.degenerate_features:
            z[:, idx] = 0.0
    return z


def alpha_for_target(norm, target_df: pd.DataFrame, target_library: str) -> np.ndarray:
    """目标域 α；**签名里没有 y**（R-7 的结构性保证）。"""
    gate = target_df.attrs.get("_gate")
    if gate is None:
        raise PreflightError("alpha_for_target 需要 target_df.attrs['_gate']（冻结的 gate 参数）")
    z = transform_target_features(norm, target_df, target_library)
    alpha = R.predict_alpha(z, gate)
    if not np.all(np.isfinite(alpha)) or alpha.min() < 0.0 or alpha.max() > 1.0:
        raise PreflightError(f"α 越界: [{alpha.min()}, {alpha.max()}]")
    return alpha


def build_target_frame(
    man: pd.DataFrame,
    p3_views: np.ndarray,
    p4_views: np.ndarray,
    norm,
    gate: dict,
    target_library: str,
) -> pd.DataFrame:
    """组装逐单元目标域表（14 列，顺序冻结）；p_expert = p_v1（B-02）。"""
    if p3_views.shape != (len(man), F.VIEWS_PER_UNIT) or p4_views.shape != (len(man), F.VIEWS_PER_UNIT):
        raise PreflightError(
            f"视图概率形状 {p3_views.shape}/{p4_views.shape} 与 {len(man)} 单元 × K=4 不符"
        )
    if not np.all(np.isfinite(p3_views)) or not np.all(np.isfinite(p4_views)):
        raise PreflightError("视图概率含 NaN/inf（B-12：✗ 不填补）")
    if p3_views.min() < 0 or p3_views.max() > 1 or p4_views.min() < 0 or p4_views.max() > 1:
        raise PreflightError("视图概率越界 [0,1]")
    p3 = p3_views[:, 0]
    p4 = p4_views[:, 0]
    df = pd.DataFrame(
        {
            "unit_id": man["unit_id"].astype(str).to_numpy(),
            "patient_id": man["patient_id"].astype(str).to_numpy(),
            "cluster_id": man["cluster_id"].astype(str).to_numpy(),
            "y": man["label"].astype(int).to_numpy(),
            "p3": p3,
            "p4": p4,
            "C3": [F.max_class_prob(p) for p in p3],
            "C4": [F.max_class_prob(p) for p in p4],
            "T3": [F.temporal_instability(p3_views[i]) for i in range(len(man))],
            "T4": [F.temporal_instability(p4_views[i]) for i in range(len(man))],
            "disagreement": [F.expert_disagreement(a, b) for a, b in zip(p3, p4)],
        }
    )
    df.attrs["_gate"] = gate
    df["alpha"] = alpha_for_target(norm, df, target_library)
    df["p_route"] = R.mixture(df["alpha"].to_numpy(dtype=float), p3, p4)
    df["fold_ensemble_n"] = len(FOLDS)
    df = df[list(TARGET_CSV_FIELDS)].copy()
    if not np.all(np.isfinite(df[["p3", "p4", "C3", "C4", "T3", "T4", "disagreement", "alpha", "p_route"]].to_numpy(float))):
        raise PreflightError("目标域表含非有限值")
    return df


# --------------------------------------------------------------- statistics
def auroc_fast(y, score):
    """Mann-Whitney AUROC（平均秩并列），与 `router_v2.auroc` 数值等价但向量化。

    单类返回 None（调用方决定 invalid / n/a）。bootstrap 里每 replicate 都要算 6 个臂，
    用向量化版本避免 Python 级并列扫描。
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    n = y.size
    n1 = int((y == 1).sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    uniq, inv, counts = np.unique(ss, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - (counts - 1) / 2.0
    ranks = np.empty(n, dtype=float)
    ranks[order] = avg[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _cluster_members(cluster_ids) -> tuple:
    keys = []
    members = {}
    for i, key in enumerate(np.asarray(cluster_ids).astype(str)):
        if key not in members:
            members[key] = []
            keys.append(key)
        members[key].append(i)
    return keys, [np.asarray(members[k], dtype=int) for k in keys]


def resample_indices(members, draw) -> np.ndarray:
    """把簇级 draw 展开成单元索引：同一簇被抽 k 次 → 其全部单元进入 k 次。"""
    return np.concatenate([members[int(d)] for d in np.asarray(draw, dtype=int)])


def bootstrap_paired(
    y,
    arms: dict,
    cluster_ids,
    direction: str,
    n_valid: int = BOOTSTRAP_VALID,
    master_seed: int = MASTER_SEED,
    max_attempts: int = MAX_BOOTSTRAP_ATTEMPTS,
) -> dict:
    """簇级配对 bootstrap（Protocol §5.5 L-12/L-13/L-14）。

    * 重采样单位 = `cluster_id`；同一簇被抽 k 次 → 其全部单元进入 k 次。
    * 同一 replicate 内同时计算全部臂（配对）。
    * 单类 replicate 记 invalid、不计数，继续抽到 n_valid 个有效重复。
    * 区间 = 有效重复的百分位 95% CI。
    * 随机源 = `SeedSequence(20260927, spawn_key=(1, idx))`，DIR-A idx=1 / DIR-B idx=2。
    """
    y = np.asarray(y, dtype=int)
    if y.size == 0 or y.min() == y.max():
        raise PreflightError(f"{direction}: 目标域单类标签，AUROC 无定义")
    for name, arr in arms.items():
        if np.asarray(arr).shape != y.shape:
            raise PreflightError(f"{direction}: 臂 {name} 长度与 y 不符")
    keys, members = _cluster_members(cluster_ids)
    if not keys:
        raise PreflightError(f"{direction}: 没有 cluster")
    ss = np.random.SeedSequence(master_seed, spawn_key=SUBSTREAM_KEY[direction])
    rng = np.random.default_rng(ss)
    arm_names = list(arms)
    reps = {name: [] for name in arm_names}
    n_attempted = 0
    n_invalid = 0
    sizes = []
    while len(reps[arm_names[0]]) < n_valid and n_attempted < max_attempts:
        n_attempted += 1
        draw = rng.integers(0, len(keys), size=len(keys))
        idx = resample_indices(members, draw)
        yb = y[idx]
        if yb.min() == yb.max():
            n_invalid += 1
            continue
        ok = True
        vals = {}
        for name in arm_names:
            v = auroc_fast(yb, np.asarray(arms[name], dtype=float)[idx])
            if v is None:
                ok = False
                break
            vals[name] = v
        if not ok:
            n_invalid += 1
            continue
        for name in arm_names:
            reps[name].append(vals[name])
        sizes.append(len(idx))
    n_got = len(reps[arm_names[0]])
    if n_got < n_valid:
        raise PreflightError(
            f"{direction}: 只有 {n_got} 个有效 bootstrap 重复（attempted={n_attempted}, "
            f"invalid={n_invalid}）；冻结要求 {n_valid} 个有效重复"
        )
    reps = {name: np.asarray(v, dtype=float) for name, v in reps.items()}
    deltas = {
        "delta_r": reps["router"] - reps["m_fixed"],
        "delta_auc_router_minus_ea": reps["router"] - reps["equal_average"],
        "delta_auc_router_minus_m3": reps["router"] - reps["m3"],
        "delta_auc_router_minus_m4": reps["router"] - reps["m4"],
    }
    arm_stats = {}
    for name, arr in reps.items():
        arm_stats[name] = {
            "mean": _jf(np.mean(arr)),
            "sd_ddof1": _jf(np.std(arr, ddof=1)),
            "ci95": [_jf(np.percentile(arr, CI_PERCENTILES[0])), _jf(np.percentile(arr, CI_PERCENTILES[1]))],
        }
    delta_stats = {}
    for name, arr in deltas.items():
        delta_stats[name] = {
            "mean": _jf(np.mean(arr)),
            "sd_ddof1": _jf(np.std(arr, ddof=1)),
            "ci95": [_jf(np.percentile(arr, CI_PERCENTILES[0])), _jf(np.percentile(arr, CI_PERCENTILES[1]))],
        }
    return {
        "meta": {
            "master_seed": int(master_seed),
            "substream_index": int(SUBSTREAM_INDEX[direction]),
            "spawn_key": list(SUBSTREAM_KEY[direction]),
            "n_attempted": int(n_attempted),
            "n_valid": int(n_got),
            "n_invalid": int(n_invalid),
            "n_clusters": int(len(keys)),
            "n_units": int(y.size),
            "n_units_per_replicate_mean": _jf(np.mean(sizes)) if sizes else None,
            "n_units_per_replicate_min": int(min(sizes)) if sizes else None,
            "n_units_per_replicate_max": int(max(sizes)) if sizes else None,
            "ci_method": "np.percentile(replicates, [2.5, 97.5]) — 冻结百分位法",
            "unit": "cluster_id（同一簇被抽 k 次 → 全部单元进入 k 次）",
            "single_class_rule": "invalid，不计数，继续抽样至 n_valid",
            "forbidden": ["video/frame-level bootstrap", "best-seed", "normal approx", "BCa", "one-sided"],
        },
        "arms": arm_stats,
        "deltas": delta_stats,
        "replicates": reps,
        "delta_replicates": deltas,
    }


def select_m_fixed(source_auc_m3, source_auc_m4) -> str:
    """m_fixed = 源域 OOF AUROC 更高的家族；**平局取 M3**（已拍板）。"""
    if source_auc_m3 is None or source_auc_m4 is None:
        raise PreflightError("源域单类，无法选 m_fixed")
    return "M3" if float(source_auc_m3) >= float(source_auc_m4) else "M4"


def p_two_sided(deltas) -> float:
    """Protocol §6.1 L-21 的双侧百分位 p 值（截断于 1）。"""
    d = np.asarray(deltas, dtype=float)
    b = d.size
    if b == 0:
        return float("nan")
    lo = float(np.sum(d <= 0.0)) / b
    hi = float(np.sum(d >= 0.0)) / b
    return float(min(1.0, 2.0 * min(lo, hi)))


def compute_metrics(
    target_df: pd.DataFrame,
    source_table: pd.DataFrame,
    direction: str,
    n_valid: int = BOOTSTRAP_VALID,
) -> dict:
    """逐方向端点：AUC(M3/M4/m_fixed/EA/Router/Oracle) + ΔR + 簇级配对 bootstrap。"""
    y = target_df["y"].to_numpy(dtype=int)
    if int(np.unique(y).size) < 2:
        raise PreflightError(f"{direction}: 目标域单类标签，AUROC 无定义（F-F）")
    p3 = target_df["p3"].to_numpy(dtype=float)
    p4 = target_df["p4"].to_numpy(dtype=float)
    alpha = target_df["alpha"].to_numpy(dtype=float)
    p_route = target_df["p_route"].to_numpy(dtype=float)
    p_ea = 0.5 * (p3 + p4)

    # m_fixed：源域 OOF AUROC 更高的家族；平局取 M3（已拍板）
    src_y = source_table["y"].to_numpy(dtype=int)
    src_p3 = source_table["p3"].to_numpy(dtype=float)
    src_p4 = source_table["p4"].to_numpy(dtype=float)
    src_auc_m3 = R.auroc(src_y, src_p3)
    src_auc_m4 = R.auroc(src_y, src_p4)
    if src_auc_m3 is None or src_auc_m4 is None:
        raise PreflightError(f"{direction}: 源域单类，无法选 m_fixed")
    fixed_family = select_m_fixed(src_auc_m3, src_auc_m4)
    p_fixed = p3 if fixed_family == "M3" else p4

    # Oracle（仅诊断上界；✗ 不进 GO）：逐单元选 Brier 更小的专家（并列取 M3）
    l3 = (p3 - y) ** 2
    l4 = (p4 - y) ** 2
    alpha_oracle = (l4 < l3).astype(float)
    p_oracle = alpha_oracle * p4 + (1.0 - alpha_oracle) * p3

    auc_m3 = auroc_fast(y, p3)
    auc_m4 = auroc_fast(y, p4)
    auc_fixed = auroc_fast(y, p_fixed)
    auc_ea = auroc_fast(y, p_ea)
    auc_router = auroc_fast(y, p_route)
    auc_oracle = auroc_fast(y, p_oracle)
    aucs = {
        "m3": auc_m3,
        "m4": auc_m4,
        "m_fixed": auc_fixed,
        "equal_average": auc_ea,
        "router": auc_router,
        "oracle": auc_oracle,
    }
    if any(v is None for v in aucs.values()):
        raise PreflightError(f"{direction}: 有端点 AUC 无定义")
    m_star = "M3" if auc_m3 >= auc_m4 else "M4"
    auc_m_star = max(auc_m3, auc_m4)

    arms = {
        "m3": p3,
        "m4": p4,
        "m_fixed": p_fixed,
        "equal_average": p_ea,
        "router": p_route,
        "oracle": p_oracle,
    }
    boot = bootstrap_paired(y, arms, target_df["cluster_id"].to_numpy(), direction, n_valid=n_valid)
    boot["meta"]["n_patients"] = int(target_df["patient_id"].nunique())
    boot["meta"]["n_cases"] = int(target_df["cluster_id"].nunique())
    d_ci = [float(v) for v in np.percentile(boot["delta_replicates"]["delta_r"], CI_PERCENTILES)]
    ea_ci = [float(v) for v in np.percentile(boot["delta_replicates"]["delta_auc_router_minus_ea"], CI_PERCENTILES)]
    delta_r_point = float(auc_router - auc_fixed)
    delta_ea_point = float(auc_router - auc_ea)
    point_nonpositive = bool(delta_r_point <= 0.0)
    ci_low_positive = bool(d_ci[0] > 0.0)

    return {
        "direction": direction,
        "source_library": N.DIRECTIONS[direction]["source"],
        "target_library": N.DIRECTIONS[direction]["target"],
        "n_units": int(len(target_df)),
        "n_clusters": int(target_df["cluster_id"].nunique()),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "fold_ensemble_n": int(len(FOLDS)),
        "m_fixed": {
            "family": fixed_family,
            "basis": "源域 OOF AUROC 更高的家族；平局取 M3",
            "source_auc_m3": _jf(src_auc_m3),
            "source_auc_m4": _jf(src_auc_m4),
            "source_margin_m3_minus_m4": _jf(float(src_auc_m3 - src_auc_m4)),
            "tie_rule": "M3",
            "source_table_rows": int(len(source_table)),
        },
        "auc": {
            "m3": _jf(auc_m3),
            "m4": _jf(auc_m4),
            "m_fixed": _jf(auc_fixed),
            "equal_average": _jf(auc_ea),
            "router": _jf(auc_router),
            "m_star": _jf(auc_m_star),
            "oracle": _jf(auc_oracle),
            "m_star_family": m_star,
            "oracle_kind": "per-unit oracle mixture（逐单元取 Brier 更小的专家；仅诊断上界，✗ 不进 GO）",
            "oracle_never_deployable": True,
        },
        "regret": {
            "m_star_family": m_star,
            "R_fixed": _jf(float(auc_m_star - auc_fixed)),
            "R_router": _jf(float(auc_m_star - auc_router)),
            "delta_r_check": _jf(float((auc_m_star - auc_fixed) - (auc_m_star - auc_router))),
        },
        "delta_r": {
            "point": delta_r_point,
            "ci95": d_ci,
            "ci_low_positive": ci_low_positive,
            "point_nonpositive": point_nonpositive,
            "p_two_sided": _jf(p_two_sided(boot["delta_replicates"]["delta_r"])),
            "ci_method": boot["meta"]["ci_method"],
            "n_valid": boot["meta"]["n_valid"],
            "n_invalid": boot["meta"]["n_invalid"],
            "n_attempted": boot["meta"]["n_attempted"],
            "L11_5_annotation": bool(point_nonpositive and ci_low_positive),
        },
        "delta_auc_router_minus_ea": {
            "point": delta_ea_point,
            "ci95": ea_ci,
            "ci_low_positive": bool(ea_ci[0] > 0.0),
            "p_two_sided": _jf(p_two_sided(boot["delta_replicates"]["delta_auc_router_minus_ea"])),
            "note": "Key secondary（Protocol §5.4 L-11.2）；本件只作并列报告，不进入 step-level GO",
        },
        "alpha": {
            "min": _jf(np.min(alpha)),
            "p05": _jf(np.percentile(alpha, 5)),
            "median": _jf(np.median(alpha)),
            "mean": _jf(np.mean(alpha)),
            "p95": _jf(np.percentile(alpha, 95)),
            "max": _jf(np.max(alpha)),
            "frac_zero": _jf(float(np.mean(alpha <= 0.0))),
            "frac_one": _jf(float(np.mean(alpha >= 1.0))),
            "frac_nonfinite": _jf(float(np.mean(~np.isfinite(alpha)))),
        },
        "features": {
            "C3_mean": _jf(float(target_df["C3"].mean())),
            "C4_mean": _jf(float(target_df["C4"].mean())),
            "T3_mean": _jf(float(target_df["T3"].mean())),
            "T4_mean": _jf(float(target_df["T4"].mean())),
            "disagreement_mean": _jf(float(target_df["disagreement"].mean())),
        },
        "bootstrap": {
            "meta": boot["meta"],
            "arms": boot["arms"],
            "deltas": boot["deltas"],
        },
        "direction_pass": ci_low_positive,
        "_boot": boot,
    }


def mechanical_verdict(results: dict) -> dict:
    """机械 GO/NO-GO（本步判据）：两方向 ΔR 的 95% CI 下界同时 > 0。"""
    dirs = sorted(results)
    if len(dirs) != 2:
        return {
            "label": "PARTIAL",
            "criterion": "两方向 ΔR 的 95% CI 下界同时 > 0",
            "directions_run": dirs,
            "direction_pass": {d: bool(results[d]["direction_pass"]) for d in dirs},
            "note": "只跑了一个方向：整体不得判 GO（Protocol §7.4.1 两方向各自）",
            "scientific_conclusion": "none",
        }
    passes = {d: bool(results[d]["direction_pass"]) for d in dirs}
    ci_lows = {d: results[d]["delta_r"]["ci95"][0] for d in dirs}
    points = {d: results[d]["delta_r"]["point"] for d in dirs}
    annotations = {
        d: bool(points[d] is not None and points[d] <= 0.0 and ci_lows[d] is not None and ci_lows[d] > 0.0)
        for d in dirs
    }
    delta_macro = 0.5 * (float(points["DIR-A"]) + float(points["DIR-B"]))
    return {
        "label": "GO" if all(passes.values()) else "NO-GO",
        "criterion": "DIR-A 与 DIR-B 的 ΔR 95% CI 下界同时 > 0（本步机械判据）",
        "direction_pass": passes,
        "ci95_low": ci_lows,
        "point": points,
        "delta_r_macro_point": _jf(delta_macro),
        "delta_r_macro_ci95": None,
        "delta_r_macro_note": (
            "未计算：宏平均区间需要跨方向联合 replicate 与独立随机子流索引，"
            "R-06.2/L-15' 的 9 个索引未为本件分配宏平均子流（G-09）；"
            "✗ 不用两方向 CI 拼凑宏平均区间"
        ),
        "lock_L11_1_note": (
            "Protocol §5.4 GO-A 还要求 ΔR_macro 的 95% CI 下界 > 0；本件未产出该区间，"
            "故此处 label 仅为 step-level 机械判读，✗ 不得引用为 §5.4 GO-A"
        ),
        "L11_5_annotations": annotations,
        "scientific_conclusion": "none",
    }


# ------------------------------------------------------------------ inference
def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - server has torch
        raise PreflightError(
            "目标域推理需要 PyTorch（environment_reference.yaml: torch 2.9.1+cu128）"
        ) from exc
    return torch


def assert_gpu1(torch) -> dict:
    """`CUDA_VISIBLE_DEVICES=1` 的硬断言 + 引擎侧 GPU 隔离守卫。"""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is None or str(env).strip() != "1":
        raise PreflightError(
            f"CUDA_VISIBLE_DEVICES 必须恰为 '1'（仅 GPU1），实际 {env!r}；"
            "物理 GPU0 属 BMC2_GPU，✗ 不得占用"
        )
    if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise PreflightError(
            "缺 CUBLAS_WORKSPACE_CONFIG（确定性 cuBLAS 需要；设为 ':4096:8'，见 runbook §2）"
        )
    if not torch.cuda.is_available():
        raise PreflightError("torch.cuda.is_available() = False")
    guard = E.check_gpu_isolation(torch.cuda.device_count(), env)
    facts = E.assert_gpu_isolation(torch)
    if int(facts.get("physical_gpu_index") or -1) != 1:
        raise PreflightError(
            f"解析出的物理 GPU={facts.get('physical_gpu_index')} != 1；拒绝启动"
        )
    return {"isolation_guard": guard, "device": facts}


def _decode_unit_views(row, image_size: int) -> tuple:
    """解码 1 个单元、K=4 视图各 16 帧（每单元只解码/预处理一次，4 视图共享）。

    数值等价于 `engine_v2.FoldDataset(view=1..4)`：同一 `F.view_positions`、
    同一 `engine_v2._prepare_image`、无增强（train=False, flip=False）。
    """
    src = None
    try:
        try:
            src = E.open_frame_source(str(row["media_kind"]), str(row["media_path"]), int(row["n_frames"]))
            n = int(src.n_frames())
            if n < F.MIN_FRAMES:
                raise PreflightError(
                    f"{row['unit_id']}: 实际可解码 {n} 帧 < {F.MIN_FRAMES}（B-02 排除）"
                )
            positions = {v: F.view_positions(n, v) for v in range(1, F.VIEWS_PER_UNIT + 1)}
            union = sorted({p for ps in positions.values() for p in ps})
            images = src.images(union)
        except (E.FrameSourceError, E.ContractError) as exc:
            raise PreflightError(f"{row['unit_id']}: 媒体解码失败: {exc}") from exc
        by_pos = dict(zip(union, images))
        frames = {
            v: np.stack([E._prepare_image(by_pos[p], int(image_size), False) for p in positions[v]], axis=0)
            for v in range(1, F.VIEWS_PER_UNIT + 1)
        }
        return frames, n
    finally:
        if src is not None:
            src.close()


def infer_fold_ensemble(layout, direction: str, man: pd.DataFrame, artifacts: list) -> dict:
    """5 折 × 4 视图 × 1 次目标域推理（AMP、batch=2、无增强）。

    为了不把同一个单元解码 20 遍（20 模型 × 4 视图），这里按**单元对**循环：
    每个单元对只解码一次（4 视图共享），再对 10 个折模型逐视图前向。
    前向的批大小仍为 2、采样/预处理/权重全部不变 —— 只省 CPU 解码，不改数值。
    """
    torch = _import_torch()
    from src.models import build_model
    from src import repro

    t0 = time.time()
    device_facts = assert_gpu1(torch)
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    repro.set_seed(PILOT_SEED, True)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    # 配置：从冻结 lock/config 读；两家族的 frames/image_size 必须一致
    configs = {}
    for family in FAMILIES:
        try:
            configs[family] = E.load_config(layout, family)
        except Exception as exc:
            raise PreflightError(f"{family}: 读取冻结训练配置失败: {exc}") from exc
    sizes = {(cfg["frames_per_clip"], cfg["image_size"]) for cfg in configs.values()}
    if len(sizes) != 1:
        raise PreflightError(f"家族间 (frames_per_clip,image_size) 不一致: {sizes}")
    frames_per_clip, image_size = sizes.pop()
    if int(frames_per_clip) != 16 or int(image_size) != 224:
        raise PreflightError(f"采样契约漂移: frames={frames_per_clip}, image_size={image_size}")

    models = []
    for family in FAMILIES:
        cfg = configs[family]
        for fold in FOLDS:
            rec = next(r for r in artifacts if r["family"] == family and r["fold"] == fold)
            ckpt_path = Path(rec["best_pt"])
            if _sha256(ckpt_path) != rec["best_pt_sha256"]:
                raise CheckpointHashError(
                    f"{family} {direction} f{fold}: 推理前 best.pt 哈希发生变化"
                )
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise PreflightError(f"{family} {direction} f{fold}: best.pt 读取失败: {exc}") from exc
            best_epoch = ckpt.get("epoch")
            if str(best_epoch) != str(rec["best_epoch"]):
                raise CheckpointHashError(
                    f"{family} {direction} f{fold}: best.pt epoch={best_epoch} != run_metadata "
                    f"best_epoch={rec['best_epoch']}"
                )
            meta = ckpt.get("meta") or {}
            for key, expected in (("family", family), ("direction", direction), ("fold", fold)):
                if key in meta and str(meta[key]) != str(expected):
                    raise CheckpointHashError(
                        f"{family} {direction} f{fold}: checkpoint meta.{key}={meta[key]!r}"
                    )
            try:
                model = build_model(family, cfg)
                model.load_state_dict(ckpt["model_state"], strict=True)
            except Exception as exc:
                raise PreflightError(
                    f"{family} {direction} f{fold}: 模型构建/权重加载失败: {exc}"
                ) from exc
            model.to(dev).eval()
            models.append({"family": family, "fold": fold, "model": model})
            del ckpt
    if len(models) != len(FAMILIES) * len(FOLDS):
        raise PreflightError(f"模型数 {len(models)} != {len(FAMILIES) * len(FOLDS)}")

    n = len(man)
    fold_prob = {family: np.full((n, len(FOLDS), F.VIEWS_PER_UNIT), np.nan) for family in FAMILIES}
    n_frames_seen = np.full(n, -1, dtype=int)
    fold_index = {f: i for i, f in enumerate(FOLDS)}

    print(
        f"[infer] {direction}: {n} 单元 × {len(models)} 模型 × K={F.VIEWS_PER_UNIT} 视图 | "
        f"batch={BATCH_SIZE} | AMP={AMP} | device=GPU1"
    )
    t_start = time.time()
    done_units = 0
    for start in range(0, n, BATCH_SIZE):
        idxs = list(range(start, min(start + BATCH_SIZE, n)))
        rows = [man.iloc[i] for i in idxs]
        decoded = [_decode_unit_views(r, int(image_size)) for r in rows]
        for k, (frames4, nf) in enumerate(decoded):
            j = idxs[k]
            if n_frames_seen[j] == -1:
                n_frames_seen[j] = int(nf)
            elif n_frames_seen[j] != int(nf):
                raise PreflightError(
                    f"{man.iloc[j]['unit_id']}: 同一单元在不同视图解出不同帧数 "
                    f"({n_frames_seen[j]} vs {nf})"
                )
        for view in range(1, F.VIEWS_PER_UNIT + 1):
            batch = np.stack([frames4[view] for frames4, _ in decoded], axis=0)
            x = torch.from_numpy(batch).to(dev, non_blocking=True)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                    for entry in models:
                        logits = entry["model"](x)
                        probs = torch.sigmoid(logits).float().cpu().numpy().reshape(-1)
                        fold_prob[entry["family"]][idxs, fold_index[entry["fold"]], view - 1] = probs
        done_units += len(idxs)
        if done_units % 40 < BATCH_SIZE or done_units == n:
            elapsed = time.time() - t_start
            rate = done_units / max(elapsed, 1e-9)
            eta = (n - done_units) / max(rate, 1e-9)
            print(
                f"[infer] {direction}: {done_units}/{n} units | {rate:.1f} units/s | "
                f"ETA {eta/60:.1f} min | elapsed {elapsed/60:.1f} min"
            )

    for family in FAMILIES:
        if not np.all(np.isfinite(fold_prob[family])):
            raise PreflightError(f"{family}: 目标域推理结果含 NaN")
        if fold_prob[family].min() < 0 or fold_prob[family].max() > 1:
            raise PreflightError(f"{family}: 概率越界")
    if (n_frames_seen < F.MIN_FRAMES).any():
        bad = man.loc[n_frames_seen < F.MIN_FRAMES, "unit_id"].head(5).tolist()
        raise PreflightError(f"存在 <16 帧单元: {bad}")

    peak = 0.0
    if torch.cuda.is_available():
        peak = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
    elapsed = time.time() - t0
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "fold_prob": fold_prob,
        "n_frames": n_frames_seen,
        "elapsed_sec": elapsed,
        "peak_vram_gb": peak,
        "device": device_facts,
        "configs": {
            family: {
                "config_source": configs[family].get("config_source"),
                "config_sha256_recorded": next(
                    r["config_sha256"] for r in artifacts if r["family"] == family
                ),
                "frames_per_clip": int(configs[family]["frames_per_clip"]),
                "image_size": int(configs[family]["image_size"]),
                "amp": bool(AMP),
                "batch_size": int(BATCH_SIZE),
            }
            for family in FAMILIES
        },
    }


# --------------------------------------------------------------- IO products
def write_predictions_csv(df: pd.DataFrame, path: Path) -> str:
    if list(df.columns) != list(TARGET_CSV_FIELDS):
        raise PreflightError(f"{path.name}: 列顺序/字段不符 {list(df.columns)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")
    return _sha256(path)


def write_raw_npz(payload: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return _sha256(path)


def write_sha256sums(out_dir: Path, names) -> Path:
    path = out_dir / "SHA256SUMS_target.txt"
    lines = []
    for name in sorted(names):
        p = out_dir / name
        if not p.exists():
            raise PreflightError(f"SHA256SUMS: 缺产物 {p}")
        lines.append(f"{_sha256(p)}  {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def check_outputs_absent(out_dir: Path, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [n for n in PRODUCT_NAMES if (out_dir / n).exists()]
    if existing:
        raise PreflightError(
            f"{out_dir} 已存在产物 {existing}；为避免覆盖历史判定，默认拒绝重写（加 --overwrite）"
        )


# --------------------------------------------------------------- md renderer
def _fmt(value, nd: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(doc: dict) -> str:
    L = []
    A = L.append
    A(f"# 目标域推理 + ΔR + 簇级配对 bootstrap（`{TAG}`）")
    A("")
    A(f"- 生成时间（UTC）：`{doc['created_utc']}`")
    A(f"- 工程根：`{doc['root']}`；输出目录：`{doc['out_dir']}`")
    A(f"- 脚本：`{SCRIPT_NAME}`（{VERSION}）；seed = {PILOT_SEED}（pilot s = 1）")
    A(f"- 配置源：`{doc['lock_dir']}`（`{doc['lock_source']}`；只读）")
    A(f"- 机械判定：**{doc['verdict']['label']}**（判据：{doc['verdict']['criterion']}；"
      f"{'已满足' if doc['verdict']['label'] == 'GO' else '未满足或不可判'}）")
    A("")
    A("> **性质**：本报告是 step-2 的**推理 + 统计准备件**，只做机械判读。")
    A("> ✗ 不写科学结论、✗ 不写「Router 有效/无效」、✗ 不改任何锁件或既有脚本。")
    A("> **机械 GO ≠ 科学 GO**：Protocol §5.4 GO-A 还要求 `ΔR_macro` 的 95% CI 下界 > 0；")
    A("> 本件未产出该区间（见 §8），故此处 `GO` 不得引用为 §5.4 GO-A。")
    A("")
    A("## 0　一句话")
    A("")
    v = doc["verdict"]
    if v["label"] == "GO":
        A("两方向 `ΔR_D` 的 95% CI 下界**同时 > 0** → 本步机械判读 **GO**（不是科学结论）。")
    elif v["label"] == "NO-GO":
        A("至少一个方向的 `ΔR_D` 95% CI 下界 **≤ 0** → 本步机械判读 **NO-GO**（不是科学结论）。")
    else:
        A("只跑了一个方向 → **PARTIAL**，整体不得判 GO（Protocol §7.4.1 两方向各自）。")
    A("")
    A("## 1　实现层决定（已拍板，✗ 不得与本报告分开引用）")
    A("")
    A("| # | 决定 | 依据 / 说明 |")
    A("|---|---|---|")
    for item in doc["implementation_decisions"]:
        A(f"| {item['id']} | {item['decision']} | {item['basis']} |")
    A("")
    A("## 2　输入与哈希链")
    A("")
    A("| 输入 | 路径 | SHA256 |")
    A("|---|---|---|")
    inp = doc["inputs"]
    A(f"| 单元表 | `{inp['units']['path']}` | `{inp['units']['sha256']}` |")
    A(f"| 折分配表 | `{inp['fold_assignment']['path']}` | `{inp['fold_assignment']['sha256']}` |")
    for name, rec in inp["media_manifests"].items():
        A(f"| 媒体清单 {name} | `{rec['path']}` | `{rec['sha256']}` |")
    for d, rec in inp["source_tables"].items():
        A(f"| 源 OOF 表 {d} | `{rec['path']}` | `{rec['file_sha256']}` |")
    for d, rec in inp["gate_reports"].items():
        A(f"| Gate 报告 {d} | `{rec['path']}` | `{rec['sha256']}` |")
    A("")
    A("20 个折模型（2 方向 × 2 家族 × 5 折）的 `best.pt` 哈希已逐条与各自 "
      "`run_metadata.json` 的 `best_checkpoint_sha256` 核对；逐条清单见 "
      "`deltaR_report.json → inputs.folds`。")
    A("")
    A("## 3　目标域推理（5 折集成，✗ 未训练）")
    A("")
    A("| 方向 | 目标域库 | 单元 / 簇 | 阳性 / 阴性 | crop_clamped | fold_ensemble_n |")
    A("|---|---|---|---|---|---|")
    for d in sorted(doc["directions"]):
        r = doc["directions"][d]
        t = inp["targets"][d]
        A(f"| {d} | `{r['target_library']}` | {r['n_units']} / {r['n_clusters']} | "
          f"{r['n_positive']} / {r['n_negative']} | {t['n_crop_clamped']} | {r['fold_ensemble_n']} |")
    A("")
    A("口径：每个折模型对目标域**全部单元**做 K=4 视图推理（v1..v4，AMP，batch=2，无增强），")
    A("逐视图取 5 折均值 → `p_v1..p_v4`；`p_expert = p_v1`（Lock §7 B-02），")
    A("`C_e = max(p_e,1−p_e)`、`T_e = Std(p^(1..4), ddof=1)`、`D_34 = |p_3−p_4|`。")
    A("目标域单元 = 该库**全部可评估且带标签**单元；不可评估（<16 帧）/缺标签按 B-12 排除并计数，"
      "✗ 不插帧、✗ 不填补（COVID-BLUeS：362 总数 → 360 可评估）。")
    A("")
    A("**规模悬殊声明（Protocol §5.1 强制）**：`OpenPOCUS`（1 385 单元 / 185 簇）vs "
      "`COVID-BLUeS`（360 单元 / 63 簇）相差 **3.85 倍**；两方向等权（含 `ΔR_macro` 点估计）"
      "必然使小域获得远超其样本量的权重。✗ 不以按样本量加权替代等权、✗ 不看结果后换口径。")
    A("")
    drift = {d: (doc.get("runtime", {}).get("per_direction", {}).get(d, {}) or {}).get("n_frames_drift_units") for d in sorted(doc["directions"])}
    A("实际解码帧数与 `units_v3p1` 冻结值不一致的单元数：" +
      "；".join(f"{d} = {drift[d]}" for d in drift) + "（仅记录；✗ 不改写 units、✗ 不插帧）。")
    A("")
    A("## 4　Router / α（源域拟合，目标域只套用）")
    A("")
    A("| 方向 | 源域 OOF 行 | τ | C | Gate backend | 重建 vs 报告 max|Δcoef| | α mean [min,max] |")
    A("|---|---|---|---|---|---|---|")
    for d in sorted(doc["directions"]):
        r = doc["directions"][d]
        i = inp["gate_rebuild"][d]
        a = r["alpha"]
        A(f"| {d} | {inp['source_tables'][d]['n_rows']} | {TAU_FIXED} | {C_FIXED} | "
          f"`{i['rebuild_backend']}` | {i['gate_max_abs_diff']['coef']:.3e} | "
          f"{_fmt(a['mean'])} [{_fmt(a['min'])},{_fmt(a['max'])}] |")
    A("")
    A("Median/IQR normalizer **只用源域 OOF** 估计并冻结；目标域行只做同一仿射变换 "
      "`z=(x−median)/(IQR+ε)`，✗ 不用目标域统计重新标定、✗ 不用目标域标签（Protocol §5.6 L-43(d)；R-7）。")
    A("")
    A("| 方向 | α min | p05 | median | mean | p95 | max | frac(α=0) | frac(α=1) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for d in sorted(doc["directions"]):
        a = doc["directions"][d]["alpha"]
        A(f"| {d} | {_fmt(a['min'])} | {_fmt(a['p05'])} | {_fmt(a['median'])} | {_fmt(a['mean'])} | "
          f"{_fmt(a['p95'])} | {_fmt(a['max'])} | {_fmt(a['frac_zero'])} | {_fmt(a['frac_one'])} |")
    A("")
    A("目标域标签 `y` 只进入 AUC / Oracle / Bootstrap 等**评估**量；Gate 拟合与 α 计算不读 y"
      "（`alpha_for_target` 签名无 y；自检以打乱 y 后 α 逐位不变作运行时验证）。")
    A("")
    A("## 5　端点（逐方向）")
    A("")
    A("| 方向 | AUC(M3) | AUC(M4) | m_fixed | AUC(m_fixed) | AUC(EA) | AUC(Router) | AUC(Oracle)† |")
    A("|---|---|---|---|---|---|---|---|")
    for d in sorted(doc["directions"]):
        r = doc["directions"][d]
        a = r["auc"]
        A(f"| {d} | {_fmt(a['m3'])} | {_fmt(a['m4'])} | {r['m_fixed']['family']} | "
          f"{_fmt(a['m_fixed'])} | {_fmt(a['equal_average'])} | {_fmt(a['router'])} | {_fmt(a['oracle'])} |")
    A("")
    A("† `Oracle` = 逐单元取 Brier 更小专家的混合（诊断上界）；**✗ 不进 GO**、✗ 不作为可部署规则"
      "（Protocol §6.1 L-22）。目标域 best single expert 的 `AUC(m*) = max(AUC(M3),AUC(M4))` "
      "另见 JSON `auc.m_star`（同样仅诊断）。")
    A("")
    A("| 方向 | 源域 AUC(M3) | 源域 AUC(M4) | 裕度 M3−M4 | m_fixed（目标域用） |")
    A("|---|---|---|---|---|")
    for d in sorted(doc["directions"]):
        mf = doc["directions"][d]["m_fixed"]
        A(f"| {d} | {_fmt(mf['source_auc_m3'])} | {_fmt(mf['source_auc_m4'])} | "
          f"{_fmt(mf['source_margin_m3_minus_m4'])} | **{mf['family']}**（平局取 M3） |")
    A("")
    A("| 方向 | ΔR_D | 95% CI | p（双侧百分位） | n_valid / n_invalid / n_attempted | 方向通过 |")
    A("|---|---|---|---|---|---|")
    for d in sorted(doc["directions"]):
        r = doc["directions"][d]
        dr = r["delta_r"]
        ci = dr["ci95"]
        A(f"| {d} | {_fmt(dr['point'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(dr['p_two_sided'], 4)} | "
          f"{dr['n_valid']} / {dr['n_invalid']} / {dr['n_attempted']} | "
          f"{'PASS' if dr['ci_low_positive'] else '**FAIL**'} |")
    A("")
    A(f"`m_fixed` 的选择依据：源域 OOF AUROC 更高的家族，平局取 M3。逐方向源域 AUROC 与裕度见 JSON。")
    A("")
    A("## 6　簇级配对 bootstrap（2000 有效重复）")
    A("")
    A("| 方向 | 簇 / 单元 | master seed | 子流 spawn_key | n_attempted | n_valid | n_invalid |")
    A("|---|---|---|---|---|---|---|")
    for d in sorted(doc["directions"]):
        m = doc["directions"][d]["bootstrap"]["meta"]
        A(f"| {d} | {m['n_clusters']} / {m['n_units']} | {m['master_seed']} | "
          f"`{tuple(m['spawn_key'])}` | {m['n_attempted']} | {m['n_valid']} | {m['n_invalid']} |")
    A("")
    A("同一 replicate 内**同时**重采样并计算 M3 / M4 / m_fixed / Equal Average / Router / Oracle；")
    A("重采样单位 = `cluster_id`，同一簇被抽 k 次则其全部单元进入 k 次；单类 replicate 记 `invalid`、")
    A("不计数、按预注册 RNG 顺序继续抽样至 2000 有效；区间 = 百分位 95% CI。")
    A("主口径 gate 系数在每个 replicate 内**固定**（不重估）。")
    A("")
    A("## 7　机械 GO / NO-GO")
    A("")
    A(f"**{v['label']}** — 判据「{v['criterion']}」"
      f"{'已满足' if v['label'] == 'GO' else '未满足'}。")
    A("")
    if v.get("ci95_low"):
        A("| 方向 | ΔR 点估计 | 95% CI 下界 | 方向通过 |")
        A("|---|---|---|---|")
        for d in sorted(v["ci95_low"]):
            A(f"| {d} | {_fmt(v['point'][d])} | {_fmt(v['ci95_low'][d])} | "
              f"{'PASS' if v['direction_pass'][d] else '**FAIL**'} |")
        A("")
        A(f"`ΔR_macro`（点估计，仅并列）：{_fmt(v['delta_r_macro_point'])}；区间：**未计算**（见 §8）。")
        A("")
        ann = {d: True for d, ok in (v.get("L11_5_annotations") or {}).items() if ok}
        if ann:
            A(f"**L-11.5 标注**：{sorted(ann)} 的点估计 ≤ 0 但 95% CI 下界 > 0；"
              "按 Protocol §5.4 L-11.5 判为成立，上面已同时给出点估计与区间。")
            A("")
    A("## 8　口径分歧与未做项（✗ 不静默省略）")
    A("")
    A("1. **ΔR_macro 的 95% CI 未计算**：宏平均需要跨方向联合 replicate 与独立随机子流索引；")
    A("   R-06.2 / L-15' 的 9 个索引未为本件分配宏平均子流（G-09），故只给点估计。")
    A("   **因此本报告的 `GO` 不是 Protocol §5.4 GO-A**（后者要求宏平均区间）。")
    A("2. **门控重估敏感性区间未做**（Protocol §5.5 L-16(b)）：本件只实现主口径（gate 系数")
    A("   在 replicate 内固定）。**主口径区间不含门控重估不确定性，真实不确定性更大。**")
    A("3. **`m_fixed` 选择裕度的 95% CI 未计算**：Protocol §6.2 要求报告 `Score(m_fixed) − Score(m_other)`")
    A("   及其区间；本件只给源域点估计裕度，区间留待下一步（不占用未分配的随机子流）。")
    A("4. **全量源域重训被替换为 5 折集成**（实现层决定 ID-1）：与 `lock_r1_freeze_proposal.md`")
    A("   Step 2 的「4 次全量源域拟合」字面不同；依据主聚合口径 A，经负责人拍板。")
    A("5. **推理解码优化**（实现层决定 ID-5）：每单元只解码一次、4 视图共享，再对 10 个模型逐视图")
    A("   前向；采样、预处理、batch=2、AMP 全部不变，只是省 CPU 解码，不改数值。")
    A("")
    A("## 9　产物与 SHA256")
    A("")
    A("| 产物 | SHA256 |")
    A("|---|---|")
    for name in sorted(doc.get("products", {})):
        A(f"| `{name}` | `{doc['products'][name]}` |")
    A("| `deltaR_report.md` | 见 `SHA256SUMS_target.txt`（报告无法内嵌自身哈希） |")
    A("| `deltaR_report.json` / `SHA256SUMS_target.txt` | 见 `SHA256SUMS_target.txt` |")
    A("")
    A("## 10　复现")
    A("")
    A("```powershell")
    A("$env:CUDA_VISIBLE_DEVICES = \"1\"")
    A("& $py scripts\\45_target_infer_deltaR.py --root D:\\TRUST_LUS --direction ALL")
    A("```")
    A("")
    A("> 自检（CPU、合成数据）：`& $py scripts\\45_target_infer_deltaR.py --self-test`。")
    A("> 零 GPU 预检：`& $py scripts\\45_target_infer_deltaR.py --root D:\\TRUST_LUS --preflight`。")
    A("")
    return "\n".join(L)


# ------------------------------------------------------------- orchestration
def _code_hashes() -> dict:
    files = {
        "script": SCRIPT_PATH,
        "features_v2": Path(F.__file__),
        "norm_v2": Path(N.__file__),
        "router_v2": Path(R.__file__),
        "engine_v2": Path(E.__file__),
        "loader_v2": ROOT / "scripts" / "98_loader_v2.py",
    }
    return {k: ({"path": str(v), "sha256": _sha256(v)} if v.exists() else None) for k, v in files.items()}


def _implementation_decisions() -> list:
    return [
        {
            "id": "ID-1",
            "decision": "不做全量重训：源模型 = 同方向 5 个 best.pt 在目标域上逐视图概率均值",
            "basis": "主聚合口径 A（A_review.md S2-4：A = 每个种子内 5 折集成，主口径）；与 OOF 表口径一致，p_expert=p_v1（Lock §7 B-02）。"
                     "偏离 lock_r1_freeze_proposal.md Step 2 的『4 次全量源域拟合』字面 —— 经负责人拍板，记为实现层决定。",
        },
        {
            "id": "ID-2",
            "decision": "目标域推理 GPU 已获批准：≈0.3–0.6 GPU·h，仅物理 GPU 1（CUDA_VISIBLE_DEVICES=1）",
            "basis": "负责人批准（2026-10-26）；见 report/target_infer_deltaR_runbook.md §GPU 批准记录。",
        },
        {
            "id": "ID-3",
            "decision": "目标域只套用源域拟合的 Median/IQR normalizer；✗ 不重估、✗ 不用目标域统计",
            "basis": "src/norm_v2.py（_router_plan.txt L117-L119 逐字）；Protocol §5.6 L-43(d)；R-7。",
        },
        {
            "id": "ID-4",
            "decision": "τ=0.5、C=0.01 固定；用源 OOF 表重建 Router，并与 pilot_gate_report.json 逐项核对（超容差即失败）",
            "basis": "Lock §7 B-01/B-05；已冻结 gate 报告；本件不重选 τ/C。",
        },
        {
            "id": "ID-5",
            "decision": "推理时每单元只解码一次、4 视图共享，再对 10 个折模型逐视图前向（batch=2、AMP、无增强）",
            "basis": "与 engine_v2.FoldDataset(view=1..4) 同一 F.view_positions / _prepare_image；仅省 CPU 解码，数值不变。",
        },
        {
            "id": "ID-6",
            "decision": "bootstrap 只实现主口径（gate 系数在 replicate 内固定）",
            "basis": "Protocol §5.5 L-16：主口径固定；门控重估敏感性未做，报告 §8 显式声明。",
        },
    ]


def _environment_facts(torch=None) -> dict:
    def _ver(name):
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "present")
        except Exception:
            return "absent"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": _ver("sklearn"),
        "torch": (getattr(torch, "__version__", None) if torch is not None else _ver("torch")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "hf_home": os.environ.get("HF_HOME"),
        "torch_home": os.environ.get("TORCH_HOME"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
    }


def run_preflight(layout, directions) -> dict:
    """零 GPU 预检：输入完整性 + 哈希链 + gate 一致性（不 import torch）。"""
    for name in ("training_contract.yaml", "model_configs/m3.yaml", "model_configs/m4.yaml"):
        if not (Path(layout.lock_dir) / name).exists():
            raise PreflightError(
                f"缺冻结配置 {Path(layout.lock_dir) / name}（推理需要 frames/image_size/模型结构）"
            )
    required_mods = {"yaml", "torch", "transformers", "huggingface_hub", "PIL"}
    if "DIR-A" in directions:      # DIR-A 的目标域 COVID-BLUeS 是 MP4，需要 cv2
        required_mods.add("cv2")
    missing = [m for m in sorted(required_mods) if importlib.util.find_spec(m) is None]
    if missing:
        raise PreflightError(f"推理环境缺依赖 {missing}（服务器 .venv 应齐备）")
    targets, manifests, folds, tables, gate_infos, rebuilds = {}, {}, {}, {}, {}, {}
    for direction in directions:
        man, tinfo = target_manifest(layout, direction)
        manifests[direction] = man
        targets[direction] = tinfo
        folds[direction] = verify_all_folds(layout, direction)
        table, tmeta = load_source_table(layout, direction)
        tables[direction] = (table, tmeta)
        _gate_doc, ginfo = load_gate_report(layout, direction)
        gate_infos[direction] = ginfo
        try:
            rebuild = fit_router_from_source(table, direction)
        except PreflightError:
            raise
        except Exception as exc:
            raise PreflightError(f"{direction}: 用源 OOF 表重建 Router 失败: {exc}") from exc
        rebuilds[direction] = {
            "rebuild": rebuild,
            "compare": compare_gate_rebuild(rebuild, ginfo),
        }
        # 方向隔离 / 标签隔离：源表单元与目标域单元不得相交
        overlap = set(table["video_id"].astype(str)) & set(man["unit_id"].astype(str))
        if overlap:
            raise PreflightError(
                f"{direction}: 源表与目标域共享 {len(overlap)} 个单元（方向隔离失败）"
            )
    units_path = layout.table_root / "stageB_data_v3p1" / "units_v3p1.csv"
    fold_path = layout.table_root / "stageB_data_v3p1" / "fold_assignment_v3p1.csv"
    try:
        fold_tbl = E.LV().load_fold_assignment(layout.table_root)
    except E.ContractError as exc:
        raise PreflightError(f"折分配表契约失败: {exc}") from exc
    fold_ids = set(fold_tbl["unit_id"].astype(str))
    for direction in directions:
        missing_fold = set(manifests[direction]["unit_id"].astype(str)) - fold_ids
        if missing_fold:
            raise PreflightError(
                f"{direction}: {len(missing_fold)} 个目标域单元不在冻结折分配表中"
            )
    media_manifests = {}
    for name in ("openpocus_frames.csv", "covid_blues.csv"):
        p = layout.media_root / "data" / "manifests" / name
        media_manifests[name] = {"path": str(p), "sha256": (_sha256(p) if p.exists() else None)}
    inputs = {
        "units": {"path": str(units_path), "sha256": _sha256(units_path)},
        "fold_assignment": {"path": str(fold_path), "sha256": _sha256(fold_path), "n_rows": int(len(fold_tbl))},
        "media_manifests": media_manifests,
        "code": _code_hashes(),
        "targets": targets,
        "folds": {d: folds[d] for d in directions},
        "source_tables": {d: tables[d][1] for d in directions},
        "gate_reports": {d: gate_infos[d] for d in directions},
        "gate_rebuild": {d: rebuilds[d]["compare"] for d in directions},
    }
    return {
        "root": str(layout.project_root),
        "table_root": str(layout.table_root),
        "media_root": str(layout.media_root),
        "lock_dir": str(layout.lock_dir),
        "lock_source": layout.lock_source,
        "out_dir": str(layout.outputs_dir / TAG / "target"),
        "directions": list(directions),
        "inputs": inputs,
        "manifests": manifests,
        "tables": {d: tables[d][0] for d in directions},
        "gate": {d: gate_infos[d] for d in directions},
        "router": {d: rebuilds[d]["rebuild"] for d in directions},
    }


def _public_inputs(pre) -> dict:
    inputs = pre["inputs"]
    out = dict(inputs)
    out["folds"] = {
        d: [
            {
                "family": r["family"],
                "fold": r["fold"],
                "best_pt_sha256": r["best_pt_sha256"],
                "run_metadata_sha256": r["run_metadata_sha256"],
                "best_epoch": r["best_epoch"],
                "config_sha256": r["config_sha256"],
                "data_version_sha256": r["data_version_sha256"],
                "oof_pred_sha256": r["oof_pred_sha256"],
            }
            for r in inputs["folds"][d]
        ]
        for d in inputs["folds"]
    }
    return out


def execute(pre, overwrite: bool = False) -> dict:
    layout = pre["_layout"]
    out_dir = Path(pre["out_dir"])
    check_outputs_absent(out_dir, overwrite)
    directions = list(pre["directions"])
    results, frames, raws = {}, {}, {}
    torch = None
    t0 = time.time()
    for direction in directions:
        man = pre["manifests"][direction]
        table = pre["tables"][direction]
        rebuild = pre["router"][direction]
        print("=" * 78)
        print(f"[{direction}] target={N.DIRECTIONS[direction]['target']} | n={len(man)} | "
              f"models=10 | frames=16 | image=224")
        infer = infer_fold_ensemble(layout=layout, direction=direction, man=man,
                                    artifacts=pre["inputs"]["folds"][direction])
        torch = _import_torch()
        p3_views = infer["fold_prob"]["M3"].mean(axis=1)
        p4_views = infer["fold_prob"]["M4"].mean(axis=1)
        if not (np.all(np.isfinite(p3_views)) and np.all(np.isfinite(p4_views))):
            raise PreflightError(f"{direction}: 5 折均值含 NaN")
        frame = build_target_frame(
            man, p3_views, p4_views, rebuild["norm"], rebuild["gate"],
            N.DIRECTIONS[direction]["target"],
        )
        metrics = compute_metrics(frame, table, direction)
        metrics["_infer_meta"] = {
            "elapsed_sec": infer["elapsed_sec"],
            "peak_vram_gb": infer["peak_vram_gb"],
            "device": infer["device"],
            "configs": infer["configs"],
            "n_frames_decoded_sha256": N.unit_set_sha256(
                [f"{u}:{n}" for u, n in zip(man["unit_id"], infer["n_frames"])]
            ),
            "n_frames_vs_frozen": int((infer["n_frames"] != man["n_frames"].to_numpy()).sum()),
            "n_frames_min": int(infer["n_frames"].min()),
            "n_frames_max": int(infer["n_frames"].max()),
        }
        metrics["n_frames_drift_units"] = int(
            (infer["n_frames"] != man["n_frames"].to_numpy()).sum()
        )
        frames[direction] = frame
        results[direction] = metrics
        raws[direction] = {
            "unit_id": man["unit_id"].astype(str).to_numpy(),
            "cluster_id": man["cluster_id"].astype(str).to_numpy(),
            "y": frame["y"].to_numpy(dtype=int),
            "p3_views": p3_views,
            "p4_views": p4_views,
            "p3_fold": infer["fold_prob"]["M3"][:, :, 0],
            "p4_fold": infer["fold_prob"]["M4"][:, :, 0],
            "alpha": frame["alpha"].to_numpy(dtype=float),
            "p_route": frame["p_route"].to_numpy(dtype=float),
            "n_frames": infer["n_frames"].astype(int),
            "delta_r_reps": metrics["_boot"]["delta_replicates"]["delta_r"],
            "delta_auc_ea_reps": metrics["_boot"]["delta_replicates"]["delta_auc_router_minus_ea"],
            "auc_router_reps": metrics["_boot"]["replicates"]["router"],
            "auc_fixed_reps": metrics["_boot"]["replicates"]["m_fixed"],
            "spawn_key": np.asarray(SUBSTREAM_KEY[direction], dtype=int),
        }
        print(f"[{direction}] ΔR={metrics['delta_r']['point']:+.4f} "
              f"CI95=[{metrics['delta_r']['ci95'][0]:+.4f}, {metrics['delta_r']['ci95'][1]:+.4f}] "
              f"| n_valid={metrics['delta_r']['n_valid']} invalid={metrics['delta_r']['n_invalid']}")

    doc = assemble_document(pre, results, torch, t0)
    return write_all_products(doc, Path(pre["out_dir"]), frames, raws, directions)


def assemble_document(pre, results, torch, t0) -> dict:
    """把逐方向结果组装成机读报告（不含产品哈希；哈希由 write_all_products 补）。"""
    directions = sorted(results)
    verdict = mechanical_verdict(results)
    return {
        "script": SCRIPT_NAME,
        "version": VERSION,
        "kind": KIND,
        "created_utc": _utc_now(),
        "gpu_used": True,
        "trained": False,
        "checkpoints_written": False,
        "target_labels_used_for_gate": False,
        "root": pre["root"],
        "table_root": pre["table_root"],
        "media_root": pre["media_root"],
        "lock_dir": pre["lock_dir"],
        "lock_source": pre["lock_source"],
        "out_dir": pre["out_dir"],
        "tag": TAG,
        "seed": PILOT_SEED,
        "implementation_decisions": _implementation_decisions(),
        "frozen_anchors": {
            "tau": TAU_FIXED,
            "C": C_FIXED,
            "master_seed": MASTER_SEED,
            "bootstrap_valid": BOOTSTRAP_VALID,
            "substream": {d: {"spawn_key": list(SUBSTREAM_KEY[d]), "index": SUBSTREAM_INDEX[d]} for d in directions},
            "delta_r": "AUC(Router) − AUC(m_fixed)  (Protocol §5.1 L-09 / Lock §3.1 L-09')",
            "m_fixed": "源域 OOF AUROC 更高的家族；平局取 M3",
            "oracle": "逐单元 Brier 最小专家混合；诊断上界，✗ 不进 GO",
            "ci": "np.percentile(replicates,[2.5,97.5])",
            "p": "Protocol §6.1 L-21 双侧百分位",
        },
        "environment": _environment_facts(torch),
        "inputs": _public_inputs(pre),
        "directions": {d: {k: v for k, v in results[d].items() if not k.startswith("_")} for d in directions},
        "verdict": verdict,
        "runtime": {
            "elapsed_sec": round(time.time() - t0, 1),
            "gpu_hours": _jf((time.time() - t0) / 3600.0),
            "per_direction": {
                d: {
                    "elapsed_sec": _jf((results[d].get("_infer_meta") or {}).get("elapsed_sec"), 1),
                    "peak_vram_gb": _jf((results[d].get("_infer_meta") or {}).get("peak_vram_gb"), 3),
                    "n_frames_drift_units": results[d].get("n_frames_drift_units"),
                }
                for d in directions
            },
        },
        "products": {},
        "products_note": "deltaR_report.json 与 SHA256SUMS_target.txt 的哈希见 SHA256SUMS_target.txt",
        "scientific_conclusion": "none",
    }


def write_all_products(doc, out_dir: Path, frames: dict, raws: dict, directions, verbose: bool = True) -> dict:
    """写 CSV / NPZ → MD → JSON → SHA256SUMS；返回带产品哈希的 doc。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    products = {}
    for d in directions:
        name = f"target_predictions_{d}.csv"
        products[name] = write_predictions_csv(frames[d], out_dir / name)
    for d in directions:
        name = f"target_raw_views_{d}.npz"
        products[name] = write_raw_npz(raws[d], out_dir / name)
    doc["products"] = dict(products)
    md = render_markdown(doc)
    md_path = out_dir / "deltaR_report.md"
    md_path.write_text(md, encoding="utf-8", newline="\n")
    products["deltaR_report.md"] = _sha256(md_path)
    doc["products"] = dict(products)
    js_path = out_dir / "deltaR_report.json"
    _write_json(js_path, doc)
    products["deltaR_report.json"] = _sha256(js_path)
    sums = write_sha256sums(
        out_dir, [n for n in PRODUCT_NAMES if (out_dir / n).exists() and n != "SHA256SUMS_target.txt"]
    )
    if verbose:
        print("=" * 78)
        print(f"[products] {out_dir}")
        for name in sorted(products):
            print(f"  {products[name]}  {name}")
        print(f"  SHA256SUMS_target.txt -> {sums}")
    return doc


# ------------------------------------------------------------------ self-test
def _raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def _synth_rows(direction="DIR-A", n=60, seed=5, shift=0.0):
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(n)])
    p3 = np.clip(0.15 + 0.65 * y + shift + rng.normal(0, 0.08, n), 0.01, 0.99)
    p4 = np.clip(0.20 + 0.55 * y + shift + rng.normal(0, 0.08, n), 0.01, 0.99)
    out = []
    for i in range(n):
        p3v = np.clip(p3[i] + rng.normal(0, 0.02, 4), 0.01, 0.99)
        p4v = np.clip(p4[i] + rng.normal(0, 0.02, 4), 0.01, 0.99)
        out.append(
            {
                "unit_id": f"{direction}_u{i:03d}",
                "patient_id": f"{direction}_p{i//3:02d}",
                "cluster_id": f"{direction}_c{i//5:02d}",
                "direction": direction,
                "role": "source_oof",
                "library": N.DIRECTIONS[direction]["source"],
                "y": int(y[i]),
                "p3": float(p3[i]),
                "p4": float(p4[i]),
                "C_3": float(max(p3[i], 1 - p3[i])),
                "C_4": float(max(p4[i], 1 - p4[i])),
                "T_3": float(np.std(p3v, ddof=1)),
                "T_4": float(np.std(p4v, ddof=1)),
                "D_34": float(abs(p3[i] - p4[i])),
                "l3": float((p3[i] - y[i]) ** 2),
                "l4": float((p4[i] - y[i]) ** 2),
            }
        )
    return out


def _synth_table(direction="DIR-A", n=80, seed=7) -> pd.DataFrame:
    rows = _synth_rows(direction, n=n, seed=seed)
    df = pd.DataFrame(rows)
    df["video_id"] = df["unit_id"]
    df["dataset"] = N.DIRECTIONS[direction]["source"]
    df["conf3"] = df["C_3"]
    df["conf4"] = df["C_4"]
    df["temp3"] = df["T_3"]
    df["temp4"] = df["T_4"]
    df["disagreement"] = df["D_34"]
    df["loss3"] = df["l3"]
    df["loss4"] = df["l4"]
    df["q4"] = R.soft_target(df["l3"], df["l4"], TAU_FIXED)
    return df[list(R.TABLE_FIELDS)]


def _synth_target_frame(direction="DIR-A", n=40, seed=11) -> pd.DataFrame:
    rows = _synth_rows(direction, n=n, seed=seed, shift=-0.05)
    df = pd.DataFrame(rows).rename(
        columns={"C_3": "C3", "C_4": "C4", "T_3": "T3", "T_4": "T4", "D_34": "disagreement"}
    )
    df["alpha"] = np.clip(0.5 + 0.2 * (df["p3"] - df["p4"]) * 10, 0.0, 1.0)
    df["p_route"] = df["alpha"] * df["p4"] + (1 - df["alpha"]) * df["p3"]
    df["fold_ensemble_n"] = 5
    df = df[list(TARGET_CSV_FIELDS)].copy()
    return df


def _selftest() -> int:
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        if not cond:
            print(f"[FAIL] {name} {detail}")

    # ---------------- A. 常量 / 静态纪律 ----------------
    check("TAU_FIXED == 0.5", TAU_FIXED == 0.5, TAU_FIXED)
    check("C_FIXED == 0.01", C_FIXED == 0.01, C_FIXED)
    check("FOLDS == 1..5", tuple(FOLDS) == (1, 2, 3, 4, 5))
    check("FAMILIES == (M3,M4)", tuple(FAMILIES) == ("M3", "M4"))
    check("target CSV 14 字段（冻结顺序）", len(TARGET_CSV_FIELDS) == 14, TARGET_CSV_FIELDS)
    check("两类子流互异", SUBSTREAM_KEY["DIR-A"] != SUBSTREAM_KEY["DIR-B"])
    check("目标域计数 360/63, 1385/185",
          TARGET_UNITS == {"DIR-A": 360, "DIR-B": 1385} and TARGET_CLUSTERS == {"DIR-A": 63, "DIR-B": 185})
    check("MASTER_SEED == 20260927", MASTER_SEED == 20260927)
    problems = ast_no_training()
    check("AST: 无训练调用 / 无顶层 torch", not problems, problems)

    # ---------------- B. 5 折集成 + 特征 ----------------
    man = pd.DataFrame(
        {
            "unit_id": [f"u{i}" for i in range(6)],
            "patient_id": [f"p{i//2}" for i in range(6)],
            "cluster_id": [f"c{i//2}" for i in range(6)],
            "label": [0, 1, 0, 1, 0, 1],
            "n_frames": [30] * 6,
        }
    )
    rng = np.random.default_rng(3)
    fold_p3 = rng.uniform(0.1, 0.9, size=(6, 5, 4))
    fold_p4 = rng.uniform(0.1, 0.9, size=(6, 5, 4))
    p3v = fold_p3.mean(axis=1)
    p4v = fold_p4.mean(axis=1)
    check("fold 均值 = 手工 np.mean(axis=folds)", np.allclose(p3v, fold_p3.mean(1)))
    check("p_expert == v_1", np.allclose(p3v[:, 0], fold_p3.mean(1)[:, 0]))
    check("5 折集成 n=5", fold_p3.shape[1] == 5)
    row = F.build_feature_row("u0", 1, p3v[0], p4v[0], 30)
    check("C3 == max(p3,1-p3)", abs(row["C_3"] - max(p3v[0, 0], 1 - p3v[0, 0])) < 1e-12)
    check("T3 == Std(ddof=1)", abs(row["T_3"] - np.std(p3v[0], ddof=1)) < 1e-12)
    check("D_34 == |p3-p4|", abs(row["D_34"] - abs(p3v[0, 0] - p4v[0, 0])) < 1e-12)
    check("l3 == (p3-y)^2", abs(row["l3"] - (p3v[0, 0] - 1) ** 2) < 1e-12)
    check("p3v/p4v 落在 [0,1]", p3v.min() >= 0 and p3v.max() <= 1 and p4v.min() >= 0 and p4v.max() <= 1)
    check("特征名冻结", F.FEATURE_NAMES == ("C_3", "C_4", "T_3", "T_4", "D_34"))

    # ---------------- C. normalizer / gate / α（R-7） ----------------
    table = _synth_table("DIR-A", n=80)
    router = fit_router_from_source(table, "DIR-A")
    norm, gate = router["norm"], router["gate"]
    check("normalizer 方向绑定", norm.direction == "DIR-A")
    check("normalizer 只用源域行（unit 集一致）",
          norm.unit_set_sha256 == N.unit_set_sha256(router["rows"]["unit_id"]))
    med_manual = np.percentile(np.column_stack([table["conf3"], table["conf4"], table["temp3"],
                                                table["temp4"], table["disagreement"]]), 50, axis=0)
    check("median == 手工百分位", np.allclose(norm.median, med_manual, atol=1e-12))
    check("gate 系数长度 5", len(gate["coef"]) == 5)
    check("gate 两次拟合确定", np.allclose(gate["coef"], fit_router_from_source(table, "DIR-A")["gate"]["coef"]))
    target_frame = _synth_target_frame("DIR-A", n=40)
    target_frame.attrs["_gate"] = gate
    z = transform_target_features(norm, target_frame, "covid_blues")
    x = target_frame[list(FEATURE_TO_FRAME.values())].to_numpy(float)
    z_manual = (x - np.asarray(norm.median)) / (np.asarray(norm.iqr) + norm.eps)
    check("目标域 z 公式一致", np.allclose(z, z_manual, atol=1e-12))
    norm_before = (np.asarray(norm.median).copy(), np.asarray(norm.iqr).copy())
    df2 = target_frame.copy()
    df2[["C3", "C4", "T3", "T4", "disagreement"]] += 0.3
    df2.attrs["_gate"] = gate
    alpha2 = alpha_for_target(norm, df2, "covid_blues")
    check("目标域统计不参与 normalizer（参数不变）",
          np.array_equal(np.asarray(norm.median), norm_before[0]) and np.array_equal(np.asarray(norm.iqr), norm_before[1]))
    check("α 随目标域特征变化（确实被使用）", not np.allclose(alpha2, alpha_for_target(norm, target_frame, "covid_blues")))
    a = alpha_for_target(norm, target_frame, "covid_blues")
    check("α ∈ [0,1]", a.min() >= 0 and a.max() <= 1)
    extreme = target_frame.copy()
    extreme[["C3", "C4", "T3", "T4", "disagreement"]] = 1e6
    extreme.attrs["_gate"] = gate
    a_ext = alpha_for_target(norm, extreme, "covid_blues")
    check("极端 z 下 α 仍在 [0,1]", a_ext.min() >= 0 and a_ext.max() <= 1)
    p_route = R.mixture(a, target_frame["p3"].to_numpy(float), target_frame["p4"].to_numpy(float))
    check("p_route == αp4+(1−α)p3", np.allclose(p_route, a * target_frame["p4"] + (1 - a) * target_frame["p3"]))
    check("p_route ∈ [0,1]", p_route.min() >= 0 and p_route.max() <= 1)
    import inspect
    sig = inspect.signature(alpha_for_target)
    check("R-7: alpha_for_target 签名无 y", "y" not in sig.parameters, list(sig.parameters))
    df_perm = target_frame.copy()
    df_perm["y"] = np.random.default_rng(99).permutation(df_perm["y"].to_numpy())
    df_perm.attrs["_gate"] = gate
    check("R-7: 打乱目标域 y 后 α 逐位不变",
          np.array_equal(alpha_for_target(norm, df_perm, "covid_blues"), a))
    bad_rows = router["rows"].copy()
    bad_rows["role"] = "target"
    check("R-7: normalizer 拒绝 target 行", _raises(lambda: N.fit(bad_rows, "DIR-A"), N.DirectionIsolationError))
    check("R-7: gate 拟合输入必须是源域行",
          all(str(r) == "source_oof" for r in set(router["rows"]["role"])))
    deg_rows = router["rows"].copy()
    deg_rows["T_3"] = 0.5
    norm_deg = N.fit(deg_rows, "DIR-A")
    check("退化特征被记录且 z 置 0", "T_3" in norm_deg.degenerate_features)

    # ---------------- D. AUC / oracle / ΔR ----------------
    y = np.array([0, 0, 1, 1])
    check("AUROC 完美 = 1", abs(auroc_fast(y, [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12)
    check("AUROC 反向 = 0", abs(auroc_fast(y, [0.9, 0.8, 0.2, 0.1]) - 0.0) < 1e-12)
    check("AUROC 全并列 = 0.5", abs(auroc_fast(y, [0.5] * 4) - 0.5) < 1e-12)
    check("AUROC 单类 = None", auroc_fast([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]) is None)
    yb = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    sb = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.7, 0.3, 0.5, 0.5, 0.5, 0.5][:8])
    check("auroc_fast == router_v2.auroc（含并列）",
          abs(auroc_fast(yb, np.round(sb, 1)) - R.auroc(yb, np.round(sb, 1))) < 1e-12)
    y_o = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p3_o = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.6, 0.4])
    p4_o = 1.0 - p3_o
    l3 = (p3_o - y_o) ** 2
    l4 = (p4_o - y_o) ** 2
    ao = (l4 < l3).astype(float)
    p_or = ao * p4_o + (1 - ao) * p3_o
    auc_or = auroc_fast(y_o, p_or)
    auc_m3 = auroc_fast(y_o, p3_o)
    auc_m4 = auroc_fast(y_o, p4_o)
    check("oracle 混合 = 1（构造数据）", abs(auc_or - 1.0) < 1e-12, auc_or)
    check("oracle ≥ 单专家", auc_or >= max(auc_m3, auc_m4))
    frame = _synth_target_frame("DIR-A", n=60)
    src = _synth_table("DIR-A", n=80, seed=9)
    met = compute_metrics(frame, src, "DIR-A", n_valid=30)
    check("m_fixed ∈ {M3,M4}", met["m_fixed"]["family"] in ("M3", "M4"))
    check("ΔR = AUC(Router)−AUC(m_fixed)",
          abs(met["delta_r"]["point"] - (met["auc"]["router"] - met["auc"]["m_fixed"])) < 1e-12)
    check("ΔR 与 R_fixed−R_router 一致",
          abs(met["regret"]["delta_r_check"] - met["delta_r"]["point"]) < 1e-12)
    check("Oracle 仅诊断（不进 GO）", met["auc"]["oracle_kind"].startswith("per-unit oracle"))
    check("m_fixed 平局取 M3", select_m_fixed(0.81, 0.81) == "M3")
    check("m_fixed 按源域 AUROC 选高者", select_m_fixed(0.70, 0.75) == "M4")

    # ---------------- E. bootstrap ----------------
    y_bs = np.array([0, 1] * 30)
    clusters = np.array([f"c{i//3}" for i in range(60)])
    rng_arm = np.random.default_rng(21)
    arms = {
        "m3": np.clip(0.35 + 0.30 * y_bs + rng_arm.normal(0, 0.30, 60), 0.01, 0.99),
        "m4": np.clip(0.45 + 0.10 * y_bs + rng_arm.normal(0, 0.30, 60), 0.01, 0.99),
        "m_fixed": np.clip(0.35 + 0.30 * y_bs + rng_arm.normal(0, 0.30, 60), 0.01, 0.99),
        "equal_average": np.full(60, 0.5),
        "router": np.clip(0.30 + 0.40 * y_bs + rng_arm.normal(0, 0.30, 60), 0.01, 0.99),
        "oracle": np.clip(0.10 + 0.80 * y_bs + rng_arm.normal(0, 0.10, 60), 0.01, 0.99),
    }
    b1 = bootstrap_paired(y_bs, arms, clusters, "DIR-A", n_valid=50)
    b2 = bootstrap_paired(y_bs, arms, clusters, "DIR-A", n_valid=50)
    check("bootstrap 同种子可复现（CI）", b1["deltas"]["delta_r"]["ci95"] == b2["deltas"]["delta_r"]["ci95"])
    check("bootstrap 同种子可复现（重复）",
          np.array_equal(b1["delta_replicates"]["delta_r"], b2["delta_replicates"]["delta_r"]))
    check("bootstrap 计数 attempted == valid+invalid",
          b1["meta"]["n_attempted"] == b1["meta"]["n_valid"] + b1["meta"]["n_invalid"])
    check("bootstrap n_valid == 50", b1["meta"]["n_valid"] == 50)
    bB = bootstrap_paired(y_bs, arms, clusters, "DIR-B", n_valid=50)
    check("DIR-A/DIR-B 子流不同", b1["delta_replicates"]["delta_r"][0] != bB["delta_replicates"]["delta_r"][0])
    check("CI == 百分位（手工）",
          abs(b1["deltas"]["delta_r"]["ci95"][0] - np.percentile(b1["delta_replicates"]["delta_r"], 2.5)) < 1e-12)
    check("CI 下界 ≤ 上界", b1["deltas"]["delta_r"]["ci95"][0] <= b1["deltas"]["delta_r"]["ci95"][1])
    check("replicate ΔR == AUC(Router)−AUC(fixed)",
          np.allclose(b1["delta_replicates"]["delta_r"],
                      b1["replicates"]["router"] - b1["replicates"]["m_fixed"]))
    check("p 值 ∈ [0,1]", 0.0 <= p_two_sided(b1["delta_replicates"]["delta_r"]) <= 1.0)
    # 强制单类 invalid：2 个簇，一簇全阳一簇全阴
    y2 = np.array([1, 1, 1, 0, 0, 0])
    c2 = np.array(["a", "a", "a", "b", "b", "b"])
    arms2 = {k: np.linspace(0.1, 0.9, 6) for k in arms}
    b2c = bootstrap_paired(y2, arms2, c2, "DIR-A", n_valid=20)
    check("单类 replicate 记 invalid 且继续抽样", b2c["meta"]["n_invalid"] > 0 and b2c["meta"]["n_attempted"] > 20,
          b2c["meta"])
    keys2, members2 = _cluster_members(np.array(["a", "a", "b", "b", "b"]))
    idx_dup = resample_indices(members2, np.array([0, 0, 1]))
    check("同一簇抽 k 次 → 单元进入 k 次", len(idx_dup) == 2 + 2 + 3, len(idx_dup))
    check("重采样索引都在范围内", idx_dup.max() < 5 and idx_dup.min() >= 0)

    # ---------------- F. 机械判定 ----------------
    def fake_dir(ci_low, point):
        return {"direction_pass": ci_low > 0, "delta_r": {"ci95": [ci_low, ci_low + 0.02], "point": point}}

    v_go = mechanical_verdict({"DIR-A": fake_dir(0.01, 0.02), "DIR-B": fake_dir(0.005, 0.01)})
    v_no = mechanical_verdict({"DIR-A": fake_dir(0.01, 0.02), "DIR-B": fake_dir(-0.001, 0.01)})
    v_part = mechanical_verdict({"DIR-A": fake_dir(0.01, 0.02)})
    v_l115 = mechanical_verdict({"DIR-A": fake_dir(0.005, -0.001), "DIR-B": fake_dir(0.005, 0.01)})
    check("GO: 两方向 CI 下界 >0", v_go["label"] == "GO")
    check("NO-GO: 任一方向 CI 下界 ≤0", v_no["label"] == "NO-GO")
    check("PARTIAL: 单方向不判 GO", v_part["label"] == "PARTIAL")
    check("L-11.5: 点估计≤0 但 CI 下界>0 → 通过并标注",
          v_l115["label"] == "GO" and v_l115["L11_5_annotations"]["DIR-A"] is True)
    check("宏平均区间未计算（不伪造）", v_go["delta_r_macro_ci95"] is None)

    # ---------------- G. checkpoint 哈希核对 ----------------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ck = td / "best.pt"
        ck.write_bytes(b"fake-weights")
        good = {
            "family": "M3", "direction": "DIR-A", "fold": 1, "seed": 13, "status": "COMPLETE",
            "best_checkpoint_sha256": _sha256(ck), "best_epoch": 7,
            "oof_pred_sha256": "a" * 64,
        }
        (td / "run_metadata.json").write_text(json.dumps(good), encoding="utf-8")
        (td / "DONE.json").write_text(json.dumps({**good, "ok": True}), encoding="utf-8")
        rec = verify_fold_artifact(ck, td / "run_metadata.json", td / "DONE.json", "DIR-A", "M3", 1)
        check("checkpoint 哈希匹配 → 通过", rec["best_pt_sha256"] == _sha256(ck))
        bad = dict(good)
        bad["best_checkpoint_sha256"] = "0" * 64
        (td / "run_metadata.json").write_text(json.dumps(bad), encoding="utf-8")
        check("checkpoint 哈希不符 → 立即失败",
              _raises(lambda: verify_fold_artifact(ck, td / "run_metadata.json", td / "DONE.json", "DIR-A", "M3", 1),
                      CheckpointHashError))
        (td / "run_metadata.json").unlink()
        check("缺 run_metadata → 失败",
              _raises(lambda: verify_fold_artifact(ck, td / "run_metadata.json", td / "DONE.json", "DIR-A", "M3", 1),
                      PreflightError))
        (td / "run_metadata.json").write_text(json.dumps({**good, "fold": 2}), encoding="utf-8")
        check("metadata fold 不符 → 失败",
              _raises(lambda: verify_fold_artifact(ck, td / "run_metadata.json", td / "DONE.json", "DIR-A", "M3", 1),
                      CheckpointHashError))

    # ---------------- H. gate 报告断言 ----------------
    good_gate = assert_gate_report(
        {
            "tau": 0.5, "C": 0.01,
            "gate": {"status": "ok", "coef": [0.1] * 5, "intercept": 0.2, "alpha_mean": 0.5},
            "normalizer": {"direction": "DIR-A", "median": [0.5] * 5, "iqr": [0.1] * 5,
                           "eps": 1e-8, "source_library": "openpocus"},
        },
        "DIR-A",
    )
    check("gate 报告解析（τ/C/coef）", good_gate["tau"] == 0.5 and len(good_gate["coef"]) == 5)
    for bad_tau, bad_c in ((0.2, 0.01), (0.5, 1.0)):
        check(f"gate 报告 τ/C 不符 → 失败 ({bad_tau},{bad_c})",
              _raises(lambda bt=bad_tau, bc=bad_c: assert_gate_report(
                  {"tau": bt, "C": bc, "gate": {"status": "ok", "coef": [0.1] * 5, "intercept": 0.0},
                   "normalizer": {"direction": "DIR-A", "median": [0.5] * 5, "iqr": [0.1] * 5}},
                  "DIR-A"), GateReportError))
    check("gate.status != ok → 失败",
          _raises(lambda: assert_gate_report(
              {"tau": 0.5, "C": 0.01, "gate": {"status": "n/a"},
               "normalizer": {"direction": "DIR-A", "median": [0.5] * 5, "iqr": [0.1] * 5}},
              "DIR-A"), GateReportError))
    check("normalizer 方向不符 → 失败",
          _raises(lambda: assert_gate_report(
              {"tau": 0.5, "C": 0.01, "gate": {"status": "ok", "coef": [0.1] * 5, "intercept": 0.0},
               "normalizer": {"direction": "DIR-B", "median": [0.5] * 5, "iqr": [0.1] * 5}},
              "DIR-A"), GateReportError))

    # ---------------- I. 产物 ----------------
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "target_predictions_DIR-A.csv"
        h = write_predictions_csv(_synth_target_frame("DIR-A", n=12), p)
        back = pd.read_csv(p)
        check("CSV 14 字段且顺序冻结", list(back.columns) == list(TARGET_CSV_FIELDS))
        check("CSV 行数一致", len(back) == 12)
        check("CSV SHA256 可复算", h == _sha256(p))
        names = ["target_predictions_DIR-A.csv"]
        sums = write_sha256sums(td, names)
        check("SHA256SUMS 1 行且哈希一致", sums.read_text(encoding="utf-8").strip() == f"{h}  target_predictions_DIR-A.csv")
        npz = td / "target_raw_views_DIR-A.npz"
        write_raw_npz({"alpha": np.asarray([0.1, 0.2]), "spawn_key": np.asarray([1, 1])}, npz)
        check("NPZ 可读且数组在", np.allclose(np.load(npz)["alpha"], [0.1, 0.2]))
        check("overwrite 保护：已存在产物时拒绝", _raises(lambda: check_outputs_absent(td, False), PreflightError))
        check("overwrite=True 时放行", check_outputs_absent(td, True) is None)

    # ---------------- J. 报告组装 + 产物闭环（合成上下文） ----------------
    frames_j, results_j = {}, {}
    inputs_j = {
        "units": {"path": "units_v3p1.csv", "sha256": "0" * 64},
        "fold_assignment": {"path": "fold_assignment_v3p1.csv", "sha256": "9" * 64, "n_rows": 1897},
        "media_manifests": {},
        "code": {},
        "targets": {},
        "folds": {},
        "source_tables": {},
        "gate_reports": {},
        "gate_rebuild": {},
    }
    for d in ("DIR-A", "DIR-B"):
        frame_d = _synth_target_frame(d, n=30, seed=13 if d == "DIR-A" else 17)
        table_d = _synth_table(d, n=40, seed=19 if d == "DIR-A" else 23)
        met_d = compute_metrics(frame_d, table_d, d, n_valid=20)
        met_d["_infer_meta"] = {"elapsed_sec": 1.0, "peak_vram_gb": 0.0}
        met_d["n_frames_drift_units"] = 0
        frames_j[d] = frame_d
        results_j[d] = met_d
        inputs_j["targets"][d] = {
            "library": N.DIRECTIONS[d]["target"],
            "n_units": len(frame_d),
            "n_clusters": int(frame_d["cluster_id"].nunique()),
            "n_crop_clamped": 0,
        }
        inputs_j["folds"][d] = [
            {
                "family": "M3", "fold": 1, "best_pt_sha256": "1" * 64,
                "run_metadata_sha256": "2" * 64, "best_epoch": 7,
                "config_sha256": "3" * 64, "data_version_sha256": "4" * 64,
                "oof_pred_sha256": "5" * 64,
            }
        ]
        inputs_j["source_tables"][d] = {
            "path": f"router_oof_table_{d}.csv", "file_sha256": "6" * 64,
            "n_rows": len(table_d), "tau": TAU_FIXED, "C": C_FIXED,
        }
        inputs_j["gate_reports"][d] = {
            "path": "pilot_gate_report.json", "sha256": "7" * 64,
            "tau": TAU_FIXED, "C": C_FIXED, "coef": [0.1] * 5, "intercept": 0.0,
            "gate_backend": "synthetic", "normalizer": {},
        }
        inputs_j["gate_rebuild"][d] = {
            "normalizer_max_abs_diff": {"median": 0.0, "iqr": 0.0},
            "gate_max_abs_diff": {"coef": 0.0, "intercept": 0.0},
            "tolerance": {"normalizer": NORM_PARAM_TOL, "gate": GATE_COEF_TOL},
            "rebuild_backend": "synthetic", "report_backend": "synthetic", "consistent": True,
        }
    pre_j = {
        "root": "<synthetic>", "table_root": "<synthetic>", "media_root": "<synthetic>",
        "lock_dir": "<synthetic>", "lock_source": "synthetic", "out_dir": "<tmp>",
        "directions": ["DIR-A", "DIR-B"], "inputs": inputs_j,
    }
    doc_j = assemble_document(pre_j, results_j, None, time.time())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        doc_j["out_dir"] = str(td)
        doc_j = write_all_products(doc_j, td, frames_j, {"DIR-A": {"x": np.asarray([1])},
                                                         "DIR-B": {"x": np.asarray([2])}},
                                   ["DIR-A", "DIR-B"], verbose=False)
        present = sorted(p.name for p in td.iterdir())
        check("产物 7 件齐全", len(present) == 7, present)
        js = json.loads((td / "deltaR_report.json").read_text(encoding="utf-8"))
        check("JSON 两方向齐备", set(js["directions"]) == {"DIR-A", "DIR-B"})
        check("JSON verdict ∈ {GO,NO-GO}", js["verdict"]["label"] in ("GO", "NO-GO"))
        check("JSON 保留产品哈希（csv/npz/md 共 5 项）", len(js["products"]) == 5, list(js["products"]))
        check("JSON 内 md 哈希与实际一致",
              js["products"]["deltaR_report.md"] == _sha256(td / "deltaR_report.md"))
        md = (td / "deltaR_report.md").read_text(encoding="utf-8")
        check("MD 含两方向端点表", "DIR-A" in md and "DIR-B" in md)
        check("MD 含 Oracle 诊断声明", "Oracle" in md and "✗ 不进 GO" in md)
        check("MD 含未做项（ΔR_macro 区间）", "ΔR_macro" in md and "未计算" in md)
        check("MD 含机械判定", ("**GO**" in md) or ("**NO-GO**" in md))
        sums_text = (td / "SHA256SUMS_target.txt").read_text(encoding="utf-8").strip().splitlines()
        check("SHA256SUMS 6 行（不含自身）", len(sums_text) == 6, len(sums_text))
        ok_sums = all(_sha256(td / line.split("  ", 1)[1]) == line.split("  ", 1)[0] for line in sums_text)
        check("SHA256SUMS 每条哈希可复算", ok_sums)

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print("=" * 78)
    print(f"45_target_infer_deltaR self-test: {n_pass}/{len(checks)} PASS")
    print(f"tau={TAU_FIXED} C={C_FIXED} seed={MASTER_SEED} bootstrap={BOOTSTRAP_VALID} | "
          f"substreams={SUBSTREAM_KEY}")
    print("=" * 78)
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print(json.dumps({"status": "FAIL", "failed": failed}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", "n_checks": len(checks)}, ensure_ascii=False))
    return 0


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    _force_utf8_stdio()
    ap = argparse.ArgumentParser(description="TRUST-LUS target inference + ΔR + cluster bootstrap")
    ap.add_argument("--root", type=Path, default=ROOT, help="同时含 outputs/ 与 stageB_data_v3p1/ 的那一层")
    ap.add_argument("--data-root", type=Path, default=None, help="可选：媒体根（默认自动解析）")
    ap.add_argument("--direction", choices=("DIR-A", "DIR-B", "ALL"), default="ALL")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _selftest()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"[FAIL] --root 不存在: {root}")
        return 2
    layout = E.resolve_layout(root, data_root=args.data_root)
    directions = ["DIR-A", "DIR-B"] if args.direction == "ALL" else [args.direction]
    out_dir = Path(args.out_dir) if args.out_dir else layout.outputs_dir / TAG / "target"

    try:
        print("=" * 78)
        print(f"TRUST-LUS {KIND} {VERSION}")
        print(f"root={layout.project_root} | table_root={layout.table_root} | media={layout.media_root}")
        print(f"lock={layout.lock_dir} ({layout.lock_source}) | directions={directions}")
        print(f"out_dir={out_dir}")
        pre = run_preflight(layout, directions)
        pre["out_dir"] = str(out_dir)
        pre["_layout"] = layout
        print("[preflight] PASS：输入哈希链、20 个折模型哈希、gate 重建一致性、方向隔离 全部通过")
        if args.preflight:
            print(json.dumps({
                "status": "PASS",
                "root": str(root),
                "out_dir": str(out_dir),
                "directions": directions,
                "targets": pre["inputs"]["targets"],
                "source_tables": {d: {k: v for k, v in pre["inputs"]["source_tables"][d].items()
                                      if k in ("n_rows", "file_sha256", "tau", "C")} for d in directions},
                "gate_rebuild": pre["inputs"]["gate_rebuild"],
                "folds": {d: len(pre["inputs"]["folds"][d]) for d in directions},
            }, ensure_ascii=False, indent=2, default=str))
            return 0
        check_outputs_absent(Path(out_dir), args.overwrite)
        doc = execute(pre, overwrite=args.overwrite)
        bits = []
        for d in directions:
            r = doc["directions"][d]["delta_r"]
            bits.append(f"ΔR_{d}={r['point']:+.4f} CI=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]")
        print(f"[VERDICT] {doc['verdict']['label']} | " + " | ".join(bits))
        return 0 if doc["verdict"]["label"] == "GO" else 1
    except PreflightError as exc:
        print(f"[FAIL] {exc}")
        return 2
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
