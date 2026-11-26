#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRUST-LUS V4.2 — OOF 合并 + 第 1 步「机制门槛」判定（零 GPU / 纯 CPU），v1.

本脚本做什么／不做什么
----------------------
* 做：把**该方向** 2 家族 × 5 折的逐折 OOF 预测（20 个文件里的 10 个）合并成方向内单元级
  行 → 用 ``src/features_v2.py::build_feature_row`` 构造 5 维可靠性特征 →
  ``partition_missing`` 按 B-12 排除缺信号单元（✗ 不填补）→ 用
  ``src/router_v2.py::fit_direction`` 以 **L_route 准则**在 τ×C 网格上选 τ*/C →
  产出 **14 字段** ``router_oof_table_<DIR>.csv`` → 实算机制门槛 →
  写 ``pilot_gate_report.md`` / ``pilot_gate_report.json``。
* 不做：✗ 不训练、✗ 不碰 GPU、✗ 不改任何既有脚本/锁件、✗ 不重算折、✗ 不换 τ/C 网格、
  ✗ 不插值/不伪造缺失列、✗ 不下科学结论。

机制门槛（``Protocol_v2_v1_1_FINAL.md`` §7.4.1，合取，全过才「第 1 步通过」）
--------------------------------------------------------------------------
> ① ``Expert-choice AUROC > 0.5``（两方向**各自**；§6.2 首行 / §5.4 L-11.4）
> ② §5.6 A-1…A-5 **全过**（Gate 监督契约无越界）
> ③ 无 ``R-7`` 违例（§7.2 R-7，对象已由「留出域标签」扩展为「**任何越界监督源**」）
> 另加一项定价前提（**非**科学门槛，单列）：``f_family`` 已复核且未触发 R-2；
> 若 L-25b 未获批准则记 ``not_applicable``，**不得据此拦截第 1 步**。

A-2 采用 **v1.2 修正后措辞**（``Lock_R1_v2_draft_v1.2_amendment.md`` A-1 / `C-104a`）：
> 「**方向内覆盖**：每个 gate-eligible 行 ``allowed ∪ forbidden`` = 该方向源域单元集在
> 本折的切片 ∪ 其补集；**方向级**（5 折取并）= 该方向的源域单元集 =
> **DIR-A 1 385 / DIR-B 360**（= D-1 的逐方向计数，合计 1 745）。」
旧读法「= 全部 1 897」**已被撤回**（1 897 是全库口径，只适用于 ``MERGED_ALL``
存储／描述性行），脚本**照旧打印**该旧读法的实测值，但**不**把它计入门槛。

口径红线
--------
* **门 ≠ GO**（§7.4 引言）：本报告只做 §7.4.1 第 1 步的**机械**放行判定，
  ✗ 不是 §5.4 L-11 的 GO 判定、✗ 不是「可科学 GO」、✗ 不是「机制已证实」（§7.4.6）。
* **``Expert-choice AUROC`` 是门槛判定，不是比较**（§6.1 L-17），不做多重校正。
* **两方向不得共享**监督集／normalizer／τ·C（§5.6 L-43(d)）：每次调用都只有单一方向的行。
* **合并 OOF 表仅存储／描述性**（L-43(c)）：本脚本的合并只发生在**单方向内部**
  （2 家族 × 5 折），✗ 从不构造跨方向合并表，✗ 不用它拟合任何 Gate。

``p_expert`` 的两种口径（脚本行为，不是科学选择）
------------------------------------------------
``--expert-prob v1``（**默认**，锁件口径）：``p_e = p_v1``。依据
``src/features_v2.py`` L188-L190：「The *primary* expert probability used for the losses /
q_4 target is v_1, because B-02 fixes v_1 = the frozen single-view inference」；且引擎
``src/engine_v2.py`` L1686-L1691 导出的同行 ``C_expert = max(p_v1, 1-p_v1)``、
``loss_expert = (p_v1-y)^2`` 本身即 **p_v1 口径**。
``--expert-prob mean4``（敏感性）：``p_e = mean(p_v1..p_v4)``，其余 4 维信号不变；
此时 ``C_e``/``ℓ_e`` 由均值重算，**与同文件 ``C_expert``/``loss_expert`` 列不再逐行相等**
（报告里逐条记录，✗ 不静默混用）。``--expert-prob both`` 同时产出两套表与两个 EC-AUROC。

CLI
---
    python scripts/40_merge_oof_and_gates.py --root D:\\TRUST_LUS --probe
    python scripts/40_merge_oof_and_gates.py --root D:\\TRUST_LUS --direction DIR-A
    python scripts/40_merge_oof_and_gates.py --root D:\\TRUST_LUS --direction ALL
    python scripts/40_merge_oof_and_gates.py --self-test          # 合成数据，✗ 不碰真实产物

引用的冻结锚点
--------------
* 14 字段表 + 字段顺序：``_router_plan.txt`` L462 / L135-L157（Step B）→ ``router_v2.TABLE_FIELDS``
* τ/C 网格与并列规则：``Lock_R1_v2_draft_v1.1.md`` §7 B-01（L475）→ ``router_v2.TAU_GRID``/``C_GRID``/
  ``choose_from_grid``（min mean L_route；并列取较小 τ，再较小 C）
* Gate 结构：``Lock §7 B-05``（L479）→ ``router_v2.fit_l2_logistic``
* 缺信号排除：``Lock §7 B-12``（L486）→ ``features_v2.partition_missing``
* 视图偏移／T_e：``Lock §7 B-02``（L476）→ ``features_v2.view_positions``/``temporal_instability``
* 方向隔离：``Protocol §5.6`` L-43(a)-(d)
* 五条断言与哈希规范：``Protocol §5.6`` L-44(a)-(e)；哈希规范 = 升序 + "\\n" 连接 + 尾随
  "\\n" + sha256（``norm_v2.unit_set_sha256``）
* 第 1 步门槛与记录义务：``Protocol §7.4.1`` / ``§7.4.4``（13 字段，缺一即 ``not_run``）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features_v2 as F  # noqa: E402
from src import router_v2 as R  # noqa: E402
from src.norm_v2 import DIRECTIONS, unit_set_sha256  # noqa: E402

# --------------------------------------------------------------------- 常量
FAMILIES = ("M3", "M4")            # 2 家族（Protocol §7.4.1）
FOLDS = (1, 2, 3, 4, 5)            # Lock §7 B-08
PILOT_SEED = 13                    # seed_reference.yaml formal_seeds[0]；s = 1
DEFAULT_TAG = "T1_router_oof_v1"   # 30_oof_pilot.py DEFAULT_TAG
DEFAULT_OUT_SUBDIR = Path("outputs") / DEFAULT_TAG / "router"
TABLE_PREFIX = "router_oof_table"
REPORT_STEM = "pilot_gate_report"
STEP1 = "step1_oof_mechanism_pilot"
N_FITS_PER_DIRECTION = len(FAMILIES) * len(FOLDS)   # 10
N_FITS_TOTAL = N_FITS_PER_DIRECTION * 2             # 20 = 2 × 5 × 2（§7.4.7）

# OOF 导出的冻结列序（src/engine_v2.py L1693-L1697，逐字；✗ 不得自创字段）
OOF_REQUIRED_COLUMNS = (
    "unit_id", "patient_id", "cluster_id", "library", "direction", "role", "family",
    "seed", "fold", "status", "y", "n_frames", "crop_clamped",
    "p_v1", "p_v2", "p_v3", "p_v4", "C_expert", "T_expert", "loss_expert", "best_epoch",
)
VIEW_COLS = ("p_v1", "p_v2", "p_v3", "p_v4")       # K = 4（B-02）
N_VIEWS = len(VIEW_COLS)
ROLE_SOURCE_OOF = "source_oof"

EXPERT_PROB_V1 = "v1"
EXPERT_PROB_MEAN4 = "mean4"
EXPERT_PROB_BOTH = "both"
EXPERT_PROB_MODES = (EXPERT_PROB_V1, EXPERT_PROB_MEAN4, EXPERT_PROB_BOTH)
DEFAULT_EXPERT_PROB = EXPERT_PROB_V1

# 机制门槛阈值：§6.2 首行「Expert-choice AUROC > 0.5」（严格大于）
EC_AUROC_THRESHOLD = 0.5
# D-1 / C-104a：方向级源域可评估单元集（= 5 折 allowed 取并）
FROZEN_DIR_COVERAGE = {"DIR-A": 1385, "DIR-B": 360}
FROZEN_D1_TOTAL = 1745
# 合并行数的量级护栏（输入完整性，✗ 不是科学门槛）
SCALE_TOLERANCE = 0.10

# §7.4.4 十三字段（顺序冻结；缺任一字段即该步判 not_run）
GATE_RECORD_FIELDS = (
    "step", "direction", "n_fits", "gate_status", "expert_choice_auroc",
    "delta_r_ci_low", "delta_auc_ci_low", "capture_ratio", "f_family_hat",
    "r_violations", "approved_by", "timestamp_utc", "artifact_sha256",
)
# 第 1 步不产出、也 ✗ 不得伪造的第 2 步量：显式标注，而不是留空或填 0
NOT_APPLICABLE_STEP1 = "not_applicable_step1"

STAGEB_DIR = "stageB_data_v3p1"
UNITS_CSV = "units_v3p1.csv"
OOF_MANIFEST_CSV = "oof_supervision_manifest_v3p1.csv"


class OofContractError(RuntimeError):
    """OOF 预测文件与本脚本/引擎契约不符（✗ 不自作修补）。"""


class GateInputError(RuntimeError):
    """输入完整性失败（哈希不符、单元集合不符、量级异常等）。"""


# ------------------------------------------------------------------ 小工具
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(obj):
    """递归转成 JSON 可序列化的内建类型（numpy 标量 -> Python）。"""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


# ------------------------------------------------------------- 路径约定
def stage_tag(tag: str, direction: str) -> str:
    """每个方向一个引擎 stage tag（与 engine_v2.py L290 完全一致）。"""
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}; expected {sorted(DIRECTIONS)}")
    return f"{tag}_{'dirA' if direction == 'DIR-A' else 'dirB'}"


def fold_dir(root, tag: str, direction: str, family: str, seed: int, fold: int) -> Path:
    """outputs/<tag>_dirX/<family>/seed<seed>/fold<fold>/（engine_v2::train_one_fold）。"""
    return Path(root) / "outputs" / stage_tag(tag, direction) / family / f"seed{seed}" / f"fold{fold}"


def oof_path(root, tag: str, direction: str, family: str, seed: int, fold: int) -> Path:
    return fold_dir(root, tag, direction, family, seed, fold) / f"oof_pred_{family}_{direction}_f{fold}.csv"


def run_metadata_path(root, tag: str, direction: str, family: str, seed: int, fold: int) -> Path:
    return fold_dir(root, tag, direction, family, seed, fold) / "run_metadata.json"


def table_path(out_dir, direction: str, suffix: str = "") -> Path:
    return Path(out_dir) / f"{TABLE_PREFIX}_{direction}{suffix}.csv"


# ------------------------------------------------------------- 探查 / schema
def probe(root, tag: str = DEFAULT_TAG, seed: int = PILOT_SEED,
          directions=("DIR-A", "DIR-B")) -> dict:
    """实际读取 ``oof_pred_*.csv``：列名、前 2 行、是否含 4 视图。

    这是任务的**探查**入口：先看真实文件的列，再决定能不能构建特征。
    ✗ 不猜测、✗ 不用契约文档替代实测。
    """
    entries = []
    for direction in directions:
        for family in FAMILIES:
            for fold in FOLDS:
                p = oof_path(root, tag, direction, family, seed, fold)
                e = {
                    "direction": direction, "family": family, "fold": fold,
                    "path": str(p), "exists": p.exists(),
                }
                if p.exists():
                    df = pd.read_csv(p)          # 探查用：不施加 schema 断言
                    e.update({
                        "size_bytes": int(p.stat().st_size),
                        "n_rows": int(len(df)),
                        "columns": [str(c) for c in df.columns],
                        "n_columns": int(df.shape[1]),
                        "has_four_views": bool(all(c in df.columns for c in VIEW_COLS)),
                        "missing_contract_columns": [
                            c for c in OOF_REQUIRED_COLUMNS if c not in df.columns
                        ],
                        "extra_columns": [
                            str(c) for c in df.columns if c not in OOF_REQUIRED_COLUMNS
                        ],
                        "first_rows": jsonable(df.head(2).to_dict("records")),
                        "head2_text": df.head(2).to_string(),
                        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
                    })
                entries.append(e)

    present = [e for e in entries if e["exists"]]
    four_view = [e for e in present if e.get("has_four_views")]
    verdict = {
        "n_expected_files": len(entries),
        "n_present": len(present),
        "n_with_four_views": len(four_view),
        "can_build_features": bool(present) and len(four_view) == len(present),
        "missing_files": [e["path"] for e in entries if not e["exists"]],
        "missing_columns_union": sorted({c for e in present for c in e.get("missing_contract_columns", [])}),
        "engine_side_contract": {
            "writer": "src/engine_v2.py L1693-L1700",
            "columns_in_order": list(OOF_REQUIRED_COLUMNS),
            "note": "列由引擎导出块冻结；缺列只能由引擎侧补，脚本 ✗ 不自造、✗ 不插值。",
        },
    }
    return {"entries": entries, "verdict": verdict}


def read_oof_table(path) -> pd.DataFrame:
    """读一个 OOF 文件（float 按 bit 往返，避免 1 ULP 漂移）。"""
    path = Path(path)
    if not path.exists():
        raise OofContractError(
            f"缺少 OOF 预测文件：{path}\n"
            "  期望位置 = outputs/<tag>_dirX/<family>/seed<seed>/fold<k>/oof_pred_<family>_<DIR>_f<k>.csv\n"
            "  （src/engine_v2.py L1699 的命名约定）。请先完成第 1 步的 20 次折模型拟合。"
        )
    return pd.read_csv(path, float_precision="round_trip")


def verify_oof_schema(df: pd.DataFrame, path, direction: str, family: str, fold: int) -> dict:
    """逐条核对 OOF 文件与冻结契约；任一不符即抛错（✗ 不自作修补）。"""
    path = Path(path)
    missing = [c for c in OOF_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise OofContractError(
            f"{path.name}: 缺少契约列 {missing}\n"
            f"  → 4 视图 {list(VIEW_COLS)} 必须由引擎侧（src/engine_v2.py L1693-L1700 的导出块）提供。\n"
            "  → 本脚本 ✗ 不伪造列、✗ 不插值、✗ 不重命名、✗ 不用 p_v1 冒充 4 视图。"
        )
    if len(df) == 0:
        raise OofContractError(f"{path.name}: 0 行，OOF 预测为空")

    bad = {}
    if not (df["direction"].astype(str) == direction).all():
        bad["direction"] = sorted(set(df["direction"].astype(str)))[:5]
    if not (df["family"].astype(str) == family).all():
        bad["family"] = sorted(set(df["family"].astype(str)))[:5]
    if not (df["fold"].astype(int) == int(fold)).all():
        bad["fold"] = sorted({int(v) for v in df["fold"]})[:5]
    if not (df["role"].astype(str) == ROLE_SOURCE_OOF).all():
        bad["role"] = sorted(set(df["role"].astype(str)))[:5]
    src = DIRECTIONS[direction]["source"]
    if not (df["library"].astype(str) == src).all():
        bad["library"] = sorted(set(df["library"].astype(str)))[:5]
    if bad:
        raise OofContractError(
            f"{path.name}: 列内容与 direction={direction}/family={family}/fold={fold} 不符：{bad}。"
            "这是 R-7 风险（越界监督源）→ 直接判失败，✗ 不自动纠正。"
        )

    dup = df["unit_id"].astype(str).duplicated()
    if dup.any():
        raise OofContractError(
            f"{path.name}: unit_id 重复 {int(dup.sum())} 行（A-5「一分析一索引」）"
        )
    y = pd.to_numeric(df["y"], errors="coerce")
    if y.isna().any() or not set(np.unique(y.astype(int))) <= {0, 1}:
        raise OofContractError(f"{path.name}: y 必须是 {{0,1}} 且非空")
    for c in VIEW_COLS:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(v)):
            raise OofContractError(f"{path.name}: 视图列 {c} 含 NaN/inf（B-12：✗ 不得填补）")
        if v.min() < 0.0 or v.max() > 1.0:
            raise OofContractError(f"{path.name}: 视图列 {c} 越界 [{v.min()},{v.max()}] ⊄ [0,1]")
    nf = pd.to_numeric(df["n_frames"], errors="coerce")
    if nf.isna().any():
        raise OofContractError(f"{path.name}: n_frames 含缺失（B-02 需 n_frames 才能判定 n<16）")

    return {
        "n_rows": int(len(df)),
        "n_units": int(df["unit_id"].astype(str).nunique()),
        "n_views": N_VIEWS,
        "columns": [str(c) for c in df.columns],
        "n_columns": int(df.shape[1]),
        "status_counts": {str(k): int(v) for k, v in df["status"].astype(str).value_counts().items()},
        "y_counts": {str(int(k)): int(v) for k, v in y.astype(int).value_counts().items()},
        "unit_set_sha256": unit_set_sha256(df["unit_id"].astype(str)),
    }


def verify_run_metadata(root, tag: str, direction: str, family: str,
                        seed: int, fold: int, oof_file) -> dict:
    """核对 run_metadata.json 的 ``oof_pred_sha256``（✗ 不跳过、✗ 不忽略不符）。"""
    p = run_metadata_path(root, tag, direction, family, seed, fold)
    actual = sha256_file(oof_file)
    if not p.exists():
        return {
            "present": False, "path": str(p),
            "oof_pred_sha256_actual": actual, "sha256_match": None,
            "warning": "run_metadata.json 缺失：无法核对 oof_pred_sha256（如实记录，不补造）",
        }
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise GateInputError(f"{p}: 无法解析 run_metadata.json（{exc}）") from exc
    declared = meta.get("oof_pred_sha256")
    if declared is None:
        raise GateInputError(f"{p}: run_metadata.json 缺 oof_pred_sha256 字段")
    if str(declared) != actual:
        raise GateInputError(
            f"{p.name}（{direction}/{family}/f{fold}）: oof_pred_sha256 不符\n"
            f"  声明 = {declared}\n  实测 = {actual}\n"
            "  → 文件被改动或复制错位；✗ 不得继续（✗ 不重算哈希来「凑上」）。"
        )
    lock_hashes = meta.get("lock_hashes") or {}
    return {
        "present": True, "path": str(p),
        "run_metadata_sha256": sha256_file(p),
        "oof_pred_sha256_declared": str(declared),
        "oof_pred_sha256_actual": actual,
        "sha256_match": True,
        "config_sha256": meta.get("config_sha256"),
        "data_version_sha256": meta.get("data_version_sha256"),
        "training_contract_sha256": meta.get("training_contract_sha256"),
        "environment_reference_sha256": meta.get("environment_reference_sha256"),
        "lock_r1_v2_draft_v1_1_sha256": lock_hashes.get("lock_r1_v2_draft_v1_1_sha256"),
        "protocol_v2_v1_1_final_sha256": lock_hashes.get("protocol_v2_v1_1_final_sha256"),
        "source_manifest_unit_set_sha256": meta.get("source_manifest_unit_set_sha256"),
        "val_units": meta.get("val_units"),
        "run_kind": meta.get("run_kind"),
        "formal_result": meta.get("formal_result"),
        "git_commit": meta.get("git_commit"),
    }


# ------------------------------------------------------- 特征行（单元级）
def _finite(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def build_rows_for_fold(m3: pd.DataFrame, m4: pd.DataFrame, direction: str, family_pair,
                        fold: int, expert_prob: str) -> list:
    """把同一折的 M3/M4 两份 OOF 拼成**单元级**特征行。

    ``p3_views`` = M3 的 ``p_v1..p_v4``；``p4_views`` = M4 的 ``p_v1..p_v4``。
    特征由 ``features_v2.build_feature_row`` 构造（5 维：C_3,C_4,T_3,T_4,D_34）。

    ``expert_prob``:
      * ``v1``    — 锁件口径：p_e = p_v1（build_feature_row 的默认语义，B-02）；
      * ``mean4`` — 敏感性：p_e = mean(p_v1..p_v4)，随之重算 ℓ_e / C_e / D_34。
    """
    left = m3.set_index(m3["unit_id"].astype(str))
    right = m4.set_index(m4["unit_id"].astype(str))
    set_a, set_b = set(left.index), set(right.index)
    if set_a != set_b:
        only_a = sorted(set_a - set_b)[:5]
        only_b = sorted(set_b - set_a)[:5]
        raise GateInputError(
            f"fold {fold}: M3 与 M4 的单元集不一致（M3-独有 {len(set_a - set_b)} 个，"
            f"M4-独有 {len(set_b - set_a)} 个）。示例 M3-only={only_a} M4-only={only_b}。"
            "两家族必须来自同一 val_man（engine_v2.py L1649）；✗ 不自动 inner-join。"
        )

    rows, mismatch_y = [], []
    for uid in sorted(set_a):
        a = left.loc[uid]
        b = right.loc[uid]
        ya, yb = int(a["y"]), int(b["y"])
        if ya != yb:
            mismatch_y.append((uid, ya, yb))
            continue
        p3_views = [float(a[c]) for c in VIEW_COLS]
        p4_views = [float(b[c]) for c in VIEW_COLS]
        n_frames = int(a["n_frames"])
        meta = {
            "patient_id": str(a["patient_id"]),
            "cluster_id": str(a["cluster_id"]),
            "library": str(a["library"]),
            "direction": direction,
            "role": ROLE_SOURCE_OOF,
            "fold": int(fold),
            "n_views": N_VIEWS,
            "expert_prob": expert_prob,
        }
        if n_frames < F.MIN_FRAMES:
            # B-02：n < 16 → INSUFFICIENT_FRAMES，排除（✗ 不填补）。build_feature_row
            # 在 n<16 时会在采样层抛错，故在此显式记一条被排除的行。
            rec = {name: float("nan") for name in F.FEATURE_NAMES}
            rec.update({
                "unit_id": str(uid), "y": ya, "n_frames": n_frames,
                "status": F.STATUS_INSUFFICIENT_FRAMES, "p3": float("nan"), "p4": float("nan"),
                "l3": float("nan"), "l4": float("nan"),
                "crop_clamped": bool(F.crop_clamped(n_frames)),
                **meta,
            })
            rows.append(rec)
            continue
        rec = F.build_feature_row(str(uid), ya, p3_views, p4_views, n_frames, meta)
        if expert_prob == EXPERT_PROB_MEAN4 and rec["status"] == F.STATUS_OK:
            p3m = float(np.mean(p3_views))
            p4m = float(np.mean(p4_views))
            rec["p3"] = p3m
            rec["p4"] = p4m
            rec["l3"] = F.brier_loss(p3m, ya)
            rec["l4"] = F.brier_loss(p4m, ya)
            rec["C_3"] = F.max_class_prob(p3m)
            rec["C_4"] = F.max_class_prob(p4m)
            rec["D_34"] = F.expert_disagreement(p3m, p4m)
            if not all(_finite(rec[n]) for n in F.FEATURE_NAMES):
                rec["status"] = F.STATUS_MISSING_SIGNAL
        rows.append(rec)

    if mismatch_y:
        raise GateInputError(
            f"fold {fold}: {len(mismatch_y)} 个单元的 y 在两家族间不一致，示例 {mismatch_y[:5]}。"
            "→ 直接报错；✗ 不得静默取其一。"
        )
    return rows


def classify_dropped(dropped: list) -> dict:
    """被排除行的原因分类（B-12 只允许「排除」，不允许「填补」）。"""
    classes = {}
    for r in dropped:
        status = str(r.get("status", "UNKNOWN"))
        if status == F.STATUS_OK:
            status = f"{F.STATUS_MISSING_SIGNAL}(non-finite feature)"
        classes[status] = classes.get(status, 0) + 1
    return {"by_status": classes, "n_dropped": len(dropped)}


def merge_direction(root, tag: str, direction: str, seed: int, expert_prob: str,
                    allow_scale_mismatch: bool = False, expected_units=None) -> dict:
    """合并一个方向的 2 家族 × 5 折 OOF → 单元级特征行（B-12 分区）。"""
    src = DIRECTIONS[direction]["source"]
    tgt = DIRECTIONS[direction]["target"]
    rows, per_fold, problems = [], {}, []

    for fold in FOLDS:
        p3f = oof_path(root, tag, direction, FAMILIES[0], seed, fold)
        p4f = oof_path(root, tag, direction, FAMILIES[1], seed, fold)
        m3 = read_oof_table(p3f)
        m4 = read_oof_table(p4f)
        s3 = verify_oof_schema(m3, p3f, direction, FAMILIES[0], fold)
        s4 = verify_oof_schema(m4, p4f, direction, FAMILIES[1], fold)
        md3 = verify_run_metadata(root, tag, direction, FAMILIES[0], seed, fold, p3f)
        md4 = verify_run_metadata(root, tag, direction, FAMILIES[1], seed, fold, p4f)
        for md, fam in ((md3, FAMILIES[0]), (md4, FAMILIES[1])):
            if md.get("warning"):
                problems.append(f"{direction}/f{fold}/{fam}: {md['warning']}")
            declared_units_sha = md.get("source_manifest_unit_set_sha256") if md["present"] else None
            if declared_units_sha and declared_units_sha != s3["unit_set_sha256"]:
                raise GateInputError(
                    f"{direction}/f{fold}/{fam}: OOF 单元集哈希 "
                    f"{s3['unit_set_sha256']} != run_metadata.source_manifest_unit_set_sha256 "
                    f"{declared_units_sha}（折清单漂移）"
                )
        rows.extend(build_rows_for_fold(m3, m4, direction, FAMILIES, fold, expert_prob))
        per_fold[int(fold)] = {
            "n_rows_M3": s3["n_rows"], "n_rows_M4": s4["n_rows"],
            "n_units_merged": s3["n_units"],
            "unit_set_sha256": s3["unit_set_sha256"],
            "status_counts_M3": s3["status_counts"], "status_counts_M4": s4["status_counts"],
            "oof_pred_sha256_M3": md3["oof_pred_sha256_actual"],
            "oof_pred_sha256_M4": md4["oof_pred_sha256_actual"],
            "run_metadata_M3": md3, "run_metadata_M4": md4,
            "files": {"M3": str(p3f), "M4": str(p4f)},
        }

    # 一单元只能出现在一折（B-08 患者/簇级单折）
    fold_of, dup_units = {}, []
    for r in rows:
        uid = str(r["unit_id"])
        if uid in fold_of and fold_of[uid] != int(r["fold"]):
            dup_units.append(uid)
        fold_of[uid] = int(r["fold"])
    if dup_units:
        raise GateInputError(f"{direction}: {len(dup_units)} 个单元跨折出现，示例 {dup_units[:5]}")

    kept, dropped, counts = F.partition_missing(rows)
    n_merged = len(rows)

    # 输入完整性：量级护栏（✗ 科学门槛）。期望值 = A-2 方向级源域单元集（1385 / 360）。
    expected = int((expected_units or FROZEN_DIR_COVERAGE)[direction])
    dev = abs(n_merged - expected) / float(expected)
    scale = {
        "expected_source_units": int(expected),
        "n_merged": int(n_merged),
        "relative_deviation": float(dev),
        "status": "exact" if n_merged == expected else ("warn" if dev <= SCALE_TOLERANCE else "fail"),
        "note": "量级护栏（输入完整性），✗ 不是科学门槛；超过 10% 直接判失败。",
    }
    if scale["status"] == "fail" and not allow_scale_mismatch:
        raise GateInputError(
            f"{direction}: 合并后 {n_merged} 行，与方向级源域单元集 {expected} 相差 "
            f"{dev:.1%}（> {SCALE_TOLERANCE:.0%}）→ 输入不可信。"
            "确认 OOF 目录/tag 是否正确；确需放行请显式加 --allow-scale-mismatch。"
        )

    return {
        "direction": direction,
        "source_library": src,
        "target_library": tgt,
        "expert_prob": expert_prob,
        "rows": rows,
        "kept": kept,
        "dropped": dropped,
        "counts": counts,
        "n_merged": int(n_merged),
        "per_fold": per_fold,
        "scale": scale,
        "problems": problems,
        "dropped_classes": classify_dropped(dropped),
    }


# --------------------------------------------------- 监督契约 A-1…A-5（实算）
def normalise_units(df: pd.DataFrame) -> pd.DataFrame:
    """把 units 表规整成 {unit_id, library, cluster_id, patient_id, fold, evaluable, label}。"""
    ren = {}
    if "n_frames" not in df.columns and "n_frames_decodable" in df.columns:
        ren["n_frames_decodable"] = "n_frames"
    if "patient_id" not in df.columns and "original_patient_id" in df.columns:
        ren["original_patient_id"] = "patient_id"
    if "label" not in df.columns and "label_bline" in df.columns:
        ren["label_bline"] = "label"
    u = df.rename(columns=ren)
    need = ["unit_id", "library", "cluster_id", "fold"]
    missing = [c for c in need if c not in u.columns]
    if missing:
        raise GateInputError(f"units 表缺列 {missing}")
    out = u[need].copy()
    out["unit_id"] = out["unit_id"].astype(str)
    out["library"] = out["library"].astype(str)
    out["cluster_id"] = out["cluster_id"].astype(str)
    out["fold"] = out["fold"].astype(int)
    out["n_frames"] = (
        pd.to_numeric(u["n_frames"], errors="coerce") if "n_frames" in u.columns else np.nan
    )
    out["patient_id"] = u["patient_id"].astype(str) if "patient_id" in u.columns else "NA"
    out["evaluable"] = (
        u["evaluable"].astype(str).str.upper() == "TRUE" if "evaluable" in u.columns else True
    )
    out["label"] = pd.to_numeric(u["label"], errors="coerce") if "label" in u.columns else np.nan
    return out.reset_index(drop=True)


def normalise_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """规整监督 manifest（接受 allowed_units_sha256 两种列名）。"""
    m = df.rename(columns={
        "allowed_units_sha256": "allowed_sha256",
        "forbidden_units_sha256": "forbidden_sha256",
    })
    for col in ("direction", "fold", "allowed_sha256", "forbidden_sha256"):
        if col not in m.columns:
            raise GateInputError(f"监督 manifest 缺列 {col!r}")
    if "gate_eligible" not in m.columns:
        m["gate_eligible"] = "TRUE"
    if "gate_supervision_source" not in m.columns:
        m["gate_supervision_source"] = ""
    if "allowed_forbidden_intersection_n" not in m.columns:
        m["allowed_forbidden_intersection_n"] = 0
    m = m.copy()
    m["direction"] = m["direction"].astype(str).str.strip()
    m["fold"] = m["fold"].astype(str).str.strip()
    m["gate_eligible"] = m["gate_eligible"].astype(str).str.upper() == "TRUE"
    return m


def evaluate_supervision(man: pd.DataFrame, units: pd.DataFrame,
                         expected_coverage=None) -> dict:
    """实算 §5.6 A-1…A-5（A-2 用 v1.2 `C-104a` 方向内覆盖口径）。

    设计要点（✗ 不照抄结论，全部现算）：
      * ``allowed`` / ``forbidden`` 由 units 表**重算**，再与 manifest 声明的哈希逐行比对；
      * A-4 对照的是该行**声明**的 ``gate_supervision_source``（锁件原文「该行声明的」）；
      * A-2 的**方向级**判据 = 5 折 ``allowed`` 取并 == 该方向源域可评估单元集；
      * 旧读法（∪ = 全部 1 897）单独实测并标注**已撤回**，✗ 不计入门槛。
    """
    mu = units
    mm = man
    all_units = set(mu["unit_id"])
    by_lib = {lib: set(g["unit_id"]) for lib, g in mu.groupby("library")}
    eu = mu[mu["evaluable"]]

    rows = []
    a1 = a2 = a3 = a4 = a5 = True
    for _, r in mm.iterrows():
        direction = str(r["direction"])
        fold = str(r["fold"])
        eligible = bool(r["gate_eligible"])
        declared_src = str(r["gate_supervision_source"]).strip()
        allowed_sha = str(r["allowed_sha256"]).strip()
        forbidden_sha = str(r["forbidden_sha256"]).strip()

        ok_a1 = all(
            str(r.get(c, "")).strip() not in ("", "nan", "None")
            for c in ("direction", "gate_supervision_source", "allowed_sha256", "forbidden_sha256")
        )
        # A-1…A-5 的合取只累计在 **gate-eligible 行**（10 行）；
        # ``MERGED_ALL`` 行是 L-43(c) 的存储／描述性台账，它自己的不变式是
        # ``gate_eligible = FALSE``，不能拿「单库 source」去要求 A-4。
        if eligible:
            a1 &= ok_a1

        if direction in DIRECTIONS:
            src = DIRECTIONS[direction]["source"]
            allowed = set(eu[(eu["library"] == src) & (eu["fold"] == int(fold))]["unit_id"])
            forbidden = set(mu[mu["library"] != src]["unit_id"])
            # A-2 方向内覆盖：本折源域切片 ∪ 其补集
            union = allowed | forbidden
            expected_union = set(
                eu[(eu["library"] == src) & (eu["fold"] == int(fold))]["unit_id"]
            ) | (all_units - by_lib.get(src, set()))
            ok_a2_row = (union == expected_union)
        elif direction == "MERGED_ALL":
            src = "ALL"
            allowed = set(all_units)
            forbidden = set()
            union = set(all_units)
            ok_a2_row = len(union) == len(all_units)
        else:
            raise GateInputError(f"监督 manifest 出现未知 direction：{direction!r}")

        inter = allowed & forbidden
        declared_inter = int(float(r.get("allowed_forbidden_intersection_n", 0) or 0))
        ok_a3 = (len(inter) == 0) and (declared_inter == 0)
        if declared_src == "ALL":
            # 全库台账行：allowed 必须等于全库单元集（这是它的可满足读法）
            ok_a4 = bool(allowed == all_units)
        elif declared_src:
            ok_a4 = bool(allowed <= by_lib.get(declared_src, set()))
        else:
            ok_a4 = False
        if eligible:
            a3 &= ok_a3
            a4 &= ok_a4
            a2 &= bool(ok_a2_row)

        rows.append({
            "direction": direction, "fold": fold, "gate_eligible": eligible,
            "gate_supervision_source": declared_src,
            "allowed_count": len(allowed), "forbidden_count": len(forbidden),
            "union_count": len(union),
            "n_intersection": len(inter),
            "allowed_sha256_match": unit_set_sha256(allowed) == allowed_sha,
            "forbidden_sha256_match": unit_set_sha256(forbidden) == forbidden_sha,
            "A1_columns_present": bool(ok_a1),
            "A2_direction_internal_coverage": bool(ok_a2_row),
            "A2_old_reading_union_1897": bool(len(union) == len(all_units)),
            "A3_disjoint": bool(ok_a3),
            "A4_allowed_subset_of_declared_source": bool(ok_a4),
        })

    gate_rows = [r for r in rows if r["gate_eligible"]]
    per_dir = pd.Series([r["direction"] for r in gate_rows]).value_counts().to_dict() if gate_rows else {}
    a5 = (
        per_dir.get("DIR-A") == 5 and per_dir.get("DIR-B") == 5 and len(gate_rows) == 10
        and all(r["allowed_sha256_match"] and r["forbidden_sha256_match"] for r in gate_rows)
    )

    coverage = {}
    for direction in DIRECTIONS:
        src = DIRECTIONS[direction]["source"]
        ids = set()
        for f in FOLDS:
            ids |= set(eu[(eu["library"] == src) & (eu["fold"] == f)]["unit_id"])
        source_evaluable = int((eu["library"] == src).sum())
        exp = None if expected_coverage is None else int(expected_coverage[direction])
        coverage[direction] = {
            "union_allowed": len(ids),
            "source_evaluable": source_evaluable,
            "complete": len(ids) == source_evaluable,
            "expected_frozen": exp,
            "matches_frozen_d1": (None if exp is None else len(ids) == exp),
        }
        if not coverage[direction]["complete"]:
            a2 = False
        if exp is not None and len(ids) != exp:
            a2 = False

    merged_row = [r for r in rows if r["direction"] == "MERGED_ALL"]
    return {
        "rows": rows,
        "A1": bool(a1), "A2": bool(a2), "A3": bool(a3), "A4": bool(a4), "A5": bool(a5),
        "direction_coverage": coverage,
        "d1_total_measured": sum(v["union_allowed"] for v in coverage.values()),
        "d1_total_frozen": FROZEN_D1_TOTAL,
        "n_gate_eligible_rows": len(gate_rows),
        "n_rows": len(rows),
        "n_merged_all_rows": len(merged_row),
        "merged_all_gate_eligible": [bool(r["gate_eligible"]) for r in merged_row],
        "A2_wording": "C-104a 方向内覆盖（DIR-A 1385 / DIR-B 360）；旧读法 ∪=1 897 已撤回",
        "old_reading_holds": [
            r["A2_old_reading_union_1897"] for r in rows if r["direction"] != "MERGED_ALL"
        ],
        "references": {
            "A-1…A-5": "Protocol_v2_v1_1_FINAL.md §5.6 L-44(b)",
            "A-2 新措辞": "Lock_R1_v2_draft_v1.2_amendment.md §A-1（C-104a）",
            "哈希规范": "Protocol §5.6 L-44(c) → norm_v2.unit_set_sha256",
        },
    }


def check_r7(kept_rows: list, direction: str, units: pd.DataFrame, man: pd.DataFrame,
             table: pd.DataFrame, dir_entries: dict) -> dict:
    """R-7「无越界监督源」的机器可判检查（§7.2 R-7 + §5.6 L-44(d) ②）。

    逐条检查（全部可机器判定）：
      1. 监督行里**没有**目标域库的单元；
      2. 监督行 ∩ 本方向 ``forbidden`` = ∅（含每折 ``⊆ allowed``）；
      3. 14 字段表的 ``dataset`` == 源域库（✗ 不是目标域）；
      4. 表中 ``patient_id`` 不出现在目标域患者集里；
      5. 两方向不共享单元/患者/数据集（仅当两方向都跑过）。
    """
    tgt = DIRECTIONS[direction]["target"]
    src = DIRECTIONS[direction]["source"]
    violations = []
    evidence = {}

    tgt_units = set(units[units["library"] == tgt]["unit_id"])
    row_units = {str(r["unit_id"]) for r in kept_rows}
    hit = sorted(row_units & tgt_units)
    evidence["target_units_in_supervision"] = len(hit)
    if hit:
        violations.append({
            "check": "target-domain units in gate supervision",
            "n": len(hit), "example": hit[:5],
            "reference": "Protocol §5.6 L-43(a)(b) / §7.2 R-7",
        })

    mm = man[(man["direction"] == direction) & man["gate_eligible"]]
    per_fold_ok = True
    for fold in FOLDS:
        rf = {str(r["unit_id"]) for r in kept_rows if int(r["fold"]) == int(fold)}
        if not rf:
            continue
        src_fold = set(units[(units["library"] == src) & (units["fold"] == int(fold))]["unit_id"])
        forbidden = set(units[units["library"] != src]["unit_id"])
        if not rf <= src_fold:
            per_fold_ok = False
            violations.append({
                "check": f"fold {fold}: gate rows ⊄ allowed",
                "n": len(rf - src_fold), "example": sorted(rf - src_fold)[:5],
                "reference": "Protocol §5.6 L-44(d)②",
            })
        if rf & forbidden:
            per_fold_ok = False
            violations.append({
                "check": f"fold {fold}: gate rows ∩ forbidden ≠ ∅",
                "n": len(rf & forbidden), "example": sorted(rf & forbidden)[:5],
                "reference": "Protocol §5.6 L-44(d)②",
            })
    evidence["per_fold_subset_of_allowed"] = bool(per_fold_ok)
    evidence["n_manifest_rows_checked"] = int(len(mm))

    # L-43(c)：合并 OOF 表只许存储／描述性 → MERGED_ALL 行必须 gate_eligible = FALSE
    merged_all = man[man["direction"] == "MERGED_ALL"]
    ledger_bad = [bool(v) for v in merged_all["gate_eligible"]] if len(merged_all) else []
    evidence["merged_all_gate_eligible"] = ledger_bad
    if any(ledger_bad):
        violations.append({
            "check": "MERGED_ALL ledger row is gate_eligible (must be FALSE)",
            "n": int(sum(1 for v in ledger_bad if v)), "example": "L-43(c)",
            "reference": "Protocol §5.6 L-43(c) / §7.2 R-7",
        })

    evidence["table_dataset"] = sorted(set(table["dataset"].astype(str)))
    if evidence["table_dataset"] != [src]:
        violations.append({
            "check": "14-field table dataset != source library",
            "n": 1, "example": evidence["table_dataset"],
            "reference": "Protocol §5.6 L-43(a)(b)",
        })

    if "patient_id" in units.columns:
        tgt_patients = set(units[units["library"] == tgt]["patient_id"].astype(str))
        shared_pat = set(table["patient_id"].astype(str)) & tgt_patients
        evidence["target_patients_in_table"] = len(shared_pat)
        if shared_pat:
            violations.append({
                "check": "target-domain patients in table",
                "n": len(shared_pat), "example": sorted(shared_pat)[:5],
                "reference": "Protocol §5.6 L-43 / §7.2 R-7",
            })

    cross = None
    other = "DIR-B" if direction == "DIR-A" else "DIR-A"
    if dir_entries and dir_entries.get(other) is not None:
        try:
            R.assert_no_shared_labels(dir_entries[direction], dir_entries[other])
            cross = {"ok": True, "detail": f"{direction} 与 {other} 不共享 dataset/unit/patient"}
        except Exception as exc:  # noqa: BLE001
            cross = {"ok": False, "detail": str(exc)}
            violations.append({
                "check": "cross-direction sharing",
                "n": 1, "example": str(exc),
                "reference": "Protocol §5.6 L-43(d) / router_v2.assert_no_shared_labels",
            })

    return {
        "direction": direction,
        "clean": not violations,
        "violations": violations,
        "evidence": evidence,
        "cross_direction": cross,
        "target_domain_used_for_supervision": bool(hit),
        "references": {
            "R-7": "Protocol §7.2 R-7（对象已扩展为「任何越界监督源」）",
            "每折 ⊆ allowed": "Protocol §5.6 L-44(d)②",
        },
    }


# --------------------------------------------------------------- 机械判定
def mechanical_verdict(dir_results: dict, supervision: dict, r7: dict) -> dict:
    """§7.4.1 第 1 步机制门槛的**机械**合取判定（只判门槛，✗ 科学结论）。"""
    items = []
    for d in ("DIR-A", "DIR-B"):
        res = dir_results.get(d)
        if res is None:
            items.append({
                "item": f"EC-AUROC({d}) > {EC_AUROC_THRESHOLD}",
                "ok": False, "detail": "方向未运行",
                "reference": "Protocol §6.2 / §7.4.1 ①",
            })
            continue
        ec = res.get("expert_choice_auroc")
        ok = ec is not None and float(ec) > EC_AUROC_THRESHOLD
        items.append({
            "item": f"EC-AUROC({d}) > {EC_AUROC_THRESHOLD}",
            "ok": bool(ok), "detail": None if ec is None else float(ec),
            "reference": "Protocol §6.2 首行 / §7.4.1 ①",
        })
    for key in ("A1", "A2", "A3", "A4", "A5"):
        items.append({
            "item": key,
            "ok": bool(supervision.get(key)),
            "detail": None,
            "reference": "Protocol §5.6 L-44(b) / §7.4.1 ②",
        })
    items.append({
        "item": "无 R-7 越界",
        "ok": all((r7.get(d) or {}).get("clean", False) for d in r7) if r7 else False,
        "detail": {d: len((r7.get(d) or {}).get("violations", [])) for d in r7},
        "reference": "Protocol §7.2 R-7 / §7.4.1 ③",
    })
    failed = [i for i in items if not i["ok"]]
    go = not failed
    complete = all(d in dir_results for d in ("DIR-A", "DIR-B"))
    label = ("GO" if go else "NO-GO") if complete else "PARTIAL"
    return {
        "go": bool(go) and complete,
        "complete": bool(complete),
        "label": label,
        "scope": "§7.4.1 第 1 步机制门槛的机械合取（门 ≠ GO，✗ 非科学结论）",
        "items": items,
        "n_items": len(items),
        "n_failed": len(failed),
        "failed_items": [i["item"] for i in failed],
        "partial_note": (
            None if complete else
            "只跑了一个方向：§7.4.1 要求两方向**各自**过 EC-AUROC，故整体不得判 GO；"
            "本方向结论见 gate_records，另一方向须补跑后重出报告。"
        ),
        "pricing_premise": {
            "item": "f_family 已复核且未触发 R-2",
            "status": NOT_APPLICABLE_STEP1,
            "note": "§7.4.1：L-25b 未获批准时记 not_applicable，且**不得据此拦截第 1 步**。",
        },
    }


def _sup_all(supervision: dict) -> bool:
    return all(bool(supervision.get(k)) for k in ("A1", "A2", "A3", "A4", "A5"))


def gate_record(direction: str, res: dict, supervision: dict, r7: dict,
                verdict: dict, timestamp: str) -> dict:
    """§7.4.4 十三字段记录（缺任一字段即 not_run；第 2 步量显式标 not_applicable）。"""
    ec_ok = res.get("ec_auroc_pass")
    rec = {
        "step": STEP1,
        "direction": direction,
        "n_fits": N_FITS_PER_DIRECTION,
        "gate_status": "pass" if (ec_ok and _sup_all(supervision) and r7["clean"]) else "fail",
        "expert_choice_auroc": res.get("expert_choice_auroc"),
        "delta_r_ci_low": NOT_APPLICABLE_STEP1,
        "delta_auc_ci_low": NOT_APPLICABLE_STEP1,
        "capture_ratio": NOT_APPLICABLE_STEP1,
        "f_family_hat": NOT_APPLICABLE_STEP1,
        "r_violations": r7["violations"],
        "approved_by": None,
        "timestamp_utc": timestamp,
        "artifact_sha256": res.get("table_file_sha256"),
        "n_fits_scope": f"{len(FAMILIES)} 家族 × {len(FOLDS)} 折 = {N_FITS_PER_DIRECTION}（本方向）",
        "note": (
            "13 字段齐备；delta_r_ci_low / delta_auc_ci_low / capture_ratio / f_family_hat 属"
            "§7.4.2 第 2 步，第 1 步不产出（✗ 不伪造 0）→ 显式 not_applicable_step1。"
            "approved_by 待负责人签署；缺签名不改变机械判定。"
        ),
        "references": {"record": "Protocol §7.4.4（13 字段）", "gate": "Protocol §7.4.1"},
    }
    missing = [k for k in GATE_RECORD_FIELDS if k not in rec]
    if missing:
        rec["gate_status"] = "not_run"
        rec["missing_fields"] = missing
    return rec


# ------------------------------------------------------------------ 主流程
def run_direction(root, tag: str, direction: str, seed: int, out_dir, expert_prob: str,
                  units: pd.DataFrame, man: pd.DataFrame,
                  allow_scale_mismatch: bool = False, expected_units=None) -> dict:
    """单方向：合并 → 特征 → τ*/C → 14 字段表 → 门槛素材（不写报告）。"""
    merged = merge_direction(root, tag, direction, seed, expert_prob,
                             allow_scale_mismatch=allow_scale_mismatch,
                             expected_units=expected_units)
    if not merged["kept"]:
        raise GateInputError(f"{direction}: 合并后没有可用单元（全部被 B-12 排除），无法拟合")
    rows_df = pd.DataFrame(merged["kept"]).reset_index(drop=True)
    for col in F.FEATURE_NAMES:
        rows_df[col] = rows_df[col].astype(float)

    fit = R.fit_direction(rows_df, direction)
    chosen = fit["grid"]["chosen"]
    table = fit["table"]
    suffix = "" if expert_prob == EXPERT_PROB_V1 else f"_{expert_prob}"
    out_path = table_path(out_dir, direction, suffix)
    doc = R.write_table(
        table, out_path,
        {
            "direction": direction,
            "dataset": fit["dataset"],
            "tau": fit["tau"],
            "C": fit["C"],
            "tau_criterion": fit["criterion"],
            "soft_target": fit["soft_target"],
            "gate": fit["gate"],
            "sensitivity_gbdt": fit["sensitivity_gbdt"],
            "normalizer": fit["normalizer"].params(),
            "grid_summary": {
                "n_cells": fit["grid"]["n_cells"],
                "rule": fit["grid"]["rule"],
                "criterion": fit["grid"]["criterion"],
            },
        },
    )

    # 表的有限性/非空率（供报告；✗ 不填补）
    fin = {
        c: float(np.isfinite(pd.to_numeric(table[c], errors="coerce").to_numpy(dtype=float)).mean())
        for c in ("p3", "p4", "conf3", "conf4", "temp3", "temp4", "disagreement", "q4")
    }

    # τ 全网格的 ΔAUC / ΔBrier（B-01 报告义务）
    grid_report = [
        {
            "tau": float(c["tau"]), "C": float(c["C"]), "status": c.get("status"),
            "mean_l_route": c.get("mean_l_route"),
            "delta_auc_router_minus_ea": c.get("delta_auc_router_minus_ea"),
            "delta_brier_ea_minus_router": c.get("delta_brier_ea_minus_router"),
            "expert_choice_auroc": c.get("expert_choice_auroc"),
        }
        for c in fit["grid"]["grid"]
    ]

    ec = chosen.get("expert_choice_auroc")
    ec_pass = ec is not None and float(ec) > EC_AUROC_THRESHOLD
    return {
        "direction": direction,
        "expert_prob": expert_prob,
        "source_library": DIRECTIONS[direction]["source"],
        "target_library": DIRECTIONS[direction]["target"],
        "n_merged": merged["n_merged"],
        "n_units": int(len(table)),
        "n_units_kept": int(len(merged["kept"])),
        "n_dropped": int(merged["counts"]["n_dropped"]),
        "n_dropped_missing_signal": int(merged["counts"]["n_dropped_missing_signal"]),
        "n_crop_clamped": int(merged["counts"]["n_crop_clamped"]),
        "dropped_classes": merged["dropped_classes"],
        "counts": merged["counts"],
        "scale": merged["scale"],
        "per_fold": merged["per_fold"],
        "problems": merged["problems"],
        "tau": float(fit["tau"]),
        "C": float(fit["C"]),
        "criteria": {
            "tau_criterion": fit["criterion"],
            "rule": fit["grid"]["rule"],
            "soft_target": fit["soft_target"],
            "n_cells": fit["grid"]["n_cells"],
            "n_na_cells": fit["grid"]["n_na_cells"],
            "tau_grid": [float(t) for t in fit["grid"]["tau_grid"]],
            "c_grid": [float(c) for c in fit["grid"]["c_grid"]],
        },
        "chosen_cell": {
            "tau": float(chosen["tau"]), "C": float(chosen["C"]),
            "mean_l_route": chosen.get("mean_l_route"),
            "mean_brier_alpha_vs_y": chosen.get("mean_brier_alpha_vs_y"),
            "mean_brier_router_vs_y": chosen.get("mean_brier_router_vs_y"),
            "expert_choice_auroc": ec,
            "auc_router": chosen.get("auc_router"),
            "auc_equal_average": chosen.get("auc_equal_average"),
            "delta_auc_router_minus_ea": chosen.get("delta_auc_router_minus_ea"),
            "delta_brier_ea_minus_router": chosen.get("delta_brier_ea_minus_router"),
            "status": chosen.get("status"),
            "n_scored_rows": chosen.get("n_scored_rows"),
            "n_na_rows": chosen.get("n_na_rows"),
            "n_na_folds": chosen.get("n_na_folds"),
        },
        "expert_choice_auroc": ec,
        "ec_auroc_pass": bool(ec_pass),
        "ec_auroc_threshold": EC_AUROC_THRESHOLD,
        "table_fields": [str(c) for c in table.columns],
        "n_table_fields": int(len(table.columns)),
        "table_sha256": doc["table_sha256"],
        "table_file_sha256": doc["file_sha256"],
        "table_path": str(out_path),
        "table_unit_set_sha256": doc["unit_set_sha256"],
        "q4_saturated_rows": doc["q4_saturated_rows"],
        "finite_fraction": fin,
        "gate": {k: fit["gate"].get(k) for k in
                 ("backend", "status", "coef", "intercept", "C", "n_rows", "alpha_mean",
                  "sample_weight", "label_pos", "label_neg")},
        "sensitivity_gbdt": fit["sensitivity_gbdt"],
        "normalizer_sha256": fit["normalizer"].params_sha256(),
        "normalizer": fit["normalizer"].params(),
        "grid_report": grid_report,
        "table_object": table,
        "_rows_df": rows_df,
    }


# --------------------------------------------------------------- 报告渲染
def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _okmark(flag) -> str:
    return "PASS" if flag else "**FAIL**"


def render_markdown(doc: dict) -> str:
    L = []
    A = L.append
    A(f"# OOF 合并 + 第 1 步机制门槛报告（`{doc['tag']}`）")
    A("")
    A(f"- 生成时间（UTC）：`{doc['timestamp_utc']}`")
    A(f"- 工程根：`{doc['root']}`；输出目录：`{doc['out_dir']}`")
    A(f"- seed = {doc['seed']}（`seed_reference.yaml` `formal_seeds[0]`；pilot s = 1）")
    A(f"- `p_expert` 口径：`{doc['expert_prob']}`"
      + ("（锁件口径 B-02：p_e = p_v1）" if doc["expert_prob"] == EXPERT_PROB_V1
         else "（敏感性：p_e = mean(p_v1..p_v4)）"))
    A(f"- 环境：numpy {doc['env']['numpy']} / pandas {doc['env']['pandas']} / "
      f"scikit-learn {doc['env']['sklearn']} / torch {doc['env']['torch']}")
    A("")
    A("> **门 ≠ GO**（`Protocol §7.4` 引言）：本报告只做 §7.4.1 第 1 步的**机械**放行判定。")
    A("> ✗ 不是 §5.4 L-11 的 GO 判定，✗ 不是「可科学 GO」，✗ 不是「机制已证实」（§7.4.6）。")
    A("> 零 GPU：本脚本不训练、不导入 torch、不写 `DONE.json`、不写 checkpoint。")
    A("")

    # ---- 0 口径
    A("## 0　口径与冻结锚点")
    A("")
    A("| 项 | 口径 | 依据 |")
    A("|---|---|---|")
    A("| 拟合次数 | 第 1 步 = 2 家族 × 5 折 × 2 方向 = **20**；本报告每方向 10 | `Protocol §7.4.1`／§7.4.7 |")
    A("| 5 维输入 | `C_3, C_4, T_3, T_4, D_34`（顺序冻结） | `_router_plan.txt` L97 → `features_v2.FEATURE_NAMES` |")
    A("| K=4 视图 | `p_v1..p_v4`，偏移 (0,+2,-2,+4)，`T_e = Std(ddof=1)` | `Lock §7 B-02`（L476） |")
    A("| 缺信号 | 任一维缺失 → **排除并计数**，✗ 不填补 | `Lock §7 B-12`（L486） |")
    A("| τ / C 网格 | τ ∈ {0.005,0.01,0.02,0.05,0.1,0.2,0.5}；C ∈ {0.01,0.03,0.1,0.3,1,3,10} | `Lock §7 B-01`（L475） |")
    A("| 选择准则 | min 平均 OOF `L_route = BCE(α, q_4)`；并列取较小 τ，再较小 C | `Lock §7 B-01`（L475） |")
    A("| 14 字段表 | `dataset,patient_id,video_id,y,p3,p4,conf3,conf4,temp3,temp4,disagreement,loss3,loss4,q4` | `_router_plan.txt` L462 |")
    A("| 方向隔离 | DIR-A 只用 OpenPOCUS 源域 OOF；DIR-B 只用 COVID-BLUeS | `Protocol §5.6` L-43(a)-(d) |")
    A("| A-1…A-5 | 五条断言 + 哈希规范（升序 + `\\n` + 尾随 `\\n` + sha256） | `Protocol §5.6` L-44(a)-(c) |")
    A("| A-2 措辞 | **方向内覆盖**：DIR-A 1 385 / DIR-B 360（旧读法 ∪=1 897 已撤回） | `Lock v1.2 amendment §A-1`（`C-104a`） |")
    A("| 机制门槛 | `EC-AUROC > 0.5`（两向各自）∧ A-1…A-5 ∧ 无 R-7 | `Protocol §7.4.1` / §6.2 / §7.2 R-7 |")
    A("| 记录义务 | 13 字段，缺一即 `not_run` | `Protocol §7.4.4` |")
    A("")

    # ---- 1 探查
    A("## 1　OOF 探查（实测列名与前 2 行）")
    A("")
    pv = doc["probe"]["verdict"]
    A(f"- 期望文件 **{pv['n_expected_files']}** 个（2 方向 × 2 家族 × 5 折）；实测存在 "
      f"**{pv['n_present']}** 个；含 `p_v1..p_v4` 的 **{pv['n_with_four_views']}** 个。")
    A(f"- 结论：**{'特征可构建' if pv['can_build_features'] else '特征无法构建 ✗'}**")
    if pv["missing_files"]:
        A("")
        A("<details><summary>缺失文件清单（" + str(len(pv["missing_files"])) + "）</summary>")
        A("")
        for p in pv["missing_files"][:40]:
            A(f"- `{p}`")
        A("")
        A("</details>")
    first = next((e for e in doc["probe"]["entries"] if e.get("exists")), None)
    if first is not None:
        A("")
        A(f"### 1.1　样例文件 `{Path(first['path']).name}`"
          f"（{first['direction']}/{first['family']}/f{first['fold']}）")
        A("")
        A(f"- 行数 **{first['n_rows']}**；列数 **{first['n_columns']}**；"
          f"大小 {first['size_bytes']} B；`p_v1..p_v4` = "
          f"**{'YES' if first['has_four_views'] else 'NO'}**；"
          f"缺契约列 {first['missing_contract_columns']}")
        A("")
        A("列名（按文件顺序）：")
        A("")
        A("```")
        A(", ".join(first["columns"]))
        A("```")
        A("")
        A("前 2 行：")
        A("")
        A("```")
        A(first["head2_text"])
        A("```")
        A("")
        A("> 该列序与 `src/engine_v2.py` L1693-L1700 的导出块逐字一致（21 列）。")
        A("> 若 `p_v1..p_v4` 缺失或行数异常 → 交付即判「特征无法构建」，需引擎侧补齐，"
          "本脚本 ✗ 不自造字段。")

    # ---- 2 合并与特征
    A("")
    A("## 2　合并与特征（每方向）")
    A("")
    A("| 方向 | 合并行数 | 期望源域单元 | 偏差 | 进表单元 | 被排除 | 缺信号 | crop_clamped | 表 14 字段 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for d, r in doc["directions"].items():
        A(f"| {d} | {r['n_merged']} | {r['scale']['expected_source_units']} | "
          f"{r['scale']['relative_deviation']:.2%} | {r['n_units']} | {r['n_dropped']} | "
          f"{r['n_dropped_missing_signal']} | {r['n_crop_clamped']} | {r['n_table_fields']} |")
    A("")
    for d, r in doc["directions"].items():
        A(f"**{d} 排除分类**（B-12：只排除、✗ 不填补）："
          f"`{json.dumps(r['dropped_classes']['by_status'], ensure_ascii=False)}`；"
          f"量级护栏 = `{r['scale']['status']}`。")
        A("")
        A(f"**{d} 字段非空率**（有限值占比）：" + "；".join(
            f"`{k}`={v:.4f}" for k, v in r["finite_fraction"].items()))
        A("")
        if r["problems"]:
            A(f"**{d} 输入告警**：")
            for p in r["problems"]:
                A(f"- {p}")
            A("")

    # ---- 3 τ*/C
    A("## 3　Router 选参（τ*/C，`L_route` 准则）")
    A("")
    A("| 方向 | τ* | C* | 平均 OOF `L_route` | `EC-AUROC` | n_cells | n/a cells | gate backend |")
    A("|---|---|---|---|---|---|---|---|")
    for d, r in doc["directions"].items():
        A(f"| {d} | **{r['tau']}** | **{r['C']}** | {_fmt(r['chosen_cell']['mean_l_route'], 6)} | "
          f"{_fmt(r['expert_choice_auroc'])} | {r['criteria']['n_cells']} | "
          f"{r['criteria']['n_na_cells']} | `{r['gate']['backend']}` |")
    A("")
    first_dir = list(doc["directions"])[0]
    A(f"并列规则：`{doc['directions'][first_dir]['criteria']['rule']}`"
      "（`Lock §7 B-01`；`oof_brier` 准则已废止，本脚本不提供该路径）。")
    A("")
    A("τ 全网格 ΔAUC / ΔBrier（B-01 报告义务；49 格明细见 JSON `grid_report`）：")
    A("")
    A("| 方向 | 最优单元格 | ΔAUC(Router−EA) | ΔBrier(EA−Router) | AUC(Router) | AUC(EA) |")
    A("|---|---|---|---|---|---|")
    for d, r in doc["directions"].items():
        c = r["chosen_cell"]
        A(f"| {d} | τ={c['tau']}, C={c['C']} | {_fmt(c['delta_auc_router_minus_ea'])} | "
          f"{_fmt(c['delta_brier_ea_minus_router'])} | {_fmt(c['auc_router'])} | "
          f"{_fmt(c['auc_equal_average'])} |")
    A("")

    # ---- 4 表哈希
    A("## 4　14 字段表与哈希")
    A("")
    A("| 方向 | 文件 | 行数 | 字段数 | `table_sha256` | `file_sha256` | `unit_set_sha256` |")
    A("|---|---|---|---|---|---|---|")
    for d, r in doc["directions"].items():
        A(f"| {d} | `{Path(r['table_path']).name}` | {r['n_units']} | {r['n_table_fields']} | "
          f"`{r['table_sha256']}` | `{r['table_file_sha256']}` | `{r['table_unit_set_sha256']}` |")
    A("")
    A("`q4` 饱和行数（=0 或 =1；τ 小属正常，✗ 不算缺陷）：" + "；".join(
        f"{d}={r['q4_saturated_rows']}" for d, r in doc["directions"].items()))
    A("")

    # ---- 5 监督契约
    A("## 5　A-1…A-5 监督契约（实算，`Protocol §5.6` L-44）")
    A("")
    sup = doc["supervision"]
    cov = sup["direction_coverage"]
    A("| 断言 | 判据（冻结措辞） | 结果 | 实测 |")
    A("|---|---|---|---|")
    A("| **A-1** | 必需列齐备（direction/gate_supervision_source/"
      "allowed_sha256/forbidden_sha256 非空） | "
      f"{_okmark(sup['A1'])} | {sup['n_rows']} 行 manifest |")
    A(f"| **A-2** | **方向内覆盖**：5 折 `allowed` 取并 = 源域单元集"
      f"（DIR-A 1 385 / DIR-B 360，`C-104a`） | {_okmark(sup['A2'])} | "
      f"DIR-A {cov['DIR-A']['union_allowed']}/{cov['DIR-A']['source_evaluable']}"
      f"（冻结 {cov['DIR-A']['expected_frozen']}）、"
      f"DIR-B {cov['DIR-B']['union_allowed']}/{cov['DIR-B']['source_evaluable']}"
      f"（冻结 {cov['DIR-B']['expected_frozen']}）；合计 "
      f"{sup['d1_total_measured']}（D-1 = {sup['d1_total_frozen']}） |")
    A(f"| **A-3** | `allowed ∩ forbidden = ∅`（含声明列 `intersection_n == 0`） | "
      f"{_okmark(sup['A3'])} | 逐行交集详见 JSON `supervision.rows` |")
    A(f"| **A-4** | `allowed ⊆ 该行声明的 gate_supervision_source` 库 | {_okmark(sup['A4'])} | 逐行核 |")
    A(f"| **A-5** | `(direction, fold)` 恰好一行，两向各 5 行；且两个声明哈希逐行命中 | "
      f"{_okmark(sup['A5'])} | gate-eligible {sup['n_gate_eligible_rows']} 行 / "
      f"共 {sup['n_rows']} 行 |")
    A("")
    A(f"> **A-2 旧读法**（`allowed ∪ forbidden` = 全部 1 897）实测逐行 "
      f"`{sup['old_reading_holds']}` → **已按 `C-104a` 撤回**，✗ 不计入门槛；"
      f"`MERGED_ALL` 存储行的 `gate_eligible = {sup['merged_all_gate_eligible']}`"
      "（`L-43(c)`：合并表仅存储／描述性）。")
    A("")
    A("逐行明细：")
    A("")
    A("| direction | fold | allowed | forbidden | ∪ | ∩ | allowed sha | forbidden sha | A-2 行内 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for row in sup["rows"]:
        A(f"| {row['direction']} | {row['fold']} | {row['allowed_count']} | "
          f"{row['forbidden_count']} | {row['union_count']} | {row['n_intersection']} | "
          f"{'v' if row['allowed_sha256_match'] else 'x'} | "
          f"{'v' if row['forbidden_sha256_match'] else 'x'} | "
          f"{'v' if row['A2_direction_internal_coverage'] else 'x'} |")
    A("")

    # ---- 6 R-7
    A("## 6　R-7「无越界监督源」")
    A("")
    A("| 方向 | 目标域单元进入监督 | 每折 ⊆ allowed | 与另一方向共享 | 结论 |")
    A("|---|---|---|---|---|")
    for d, r in doc["r7"].items():
        ev = r["evidence"]
        cx = r["cross_direction"]
        A(f"| {d} | {ev['target_units_in_supervision']} | "
          f"{'YES' if ev['per_fold_subset_of_allowed'] else 'NO'} | "
          f"{'n/a' if cx is None else ('clean' if cx['ok'] else 'SHARED')} | "
          f"**{'PASS' if r['clean'] else 'FAIL'}** |")
    A("")
    for d, r in doc["r7"].items():
        if r["violations"]:
            A(f"**{d} 违例明细**：")
            for v in r["violations"]:
                A(f"- `{v['check']}` n={v['n']} 例={v['example']}（{v['reference']}）")
            A("")
        else:
            A(f"- {d}：无违例。目标域 `{DIRECTIONS[d]['target']}` 的单元/标签未以任何方式进入"
              "监督、门控、阈值或归一化（`§7.2 R-7`、`§5.6 L-43`）。")
    A("")

    # ---- 7 机械判定
    A("## 7　机械判定（GO / NO-GO，只按门槛）")
    A("")
    v = doc["verdict"]
    A(f"### 结果：**{v['label']}**")
    A("")
    A(f"判定范围：{v['scope']}。失败项 {v['n_failed']} / {v['n_items']}。")
    if v.get("partial_note"):
        A("")
        A(f"> `{v['label']}`：{v['partial_note']}")
    A("")
    A("| # | 门槛项 | 结果 | 实测 | 依据 |")
    A("|---|---|---|---|---|")
    for i, it in enumerate(v["items"], 1):
        det = it["detail"]
        if det is None or isinstance(det, bool):
            det_s = "n/a"
        elif isinstance(det, (int, float)):
            det_s = _fmt(det)
        else:
            det_s = json.dumps(det, ensure_ascii=False)
        A(f"| {i} | {it['item']} | {_okmark(it['ok'])} | {det_s} | {it['reference']} |")
    A("")
    if v["failed_items"]:
        A(f"**未过项**：{', '.join('`' + x + '`' for x in v['failed_items'])}。")
        A("")
    A(f"定价前提（**非**科学门槛，单列）：`{v['pricing_premise']['item']}` → "
      f"`{v['pricing_premise']['status']}`。{v['pricing_premise']['note']}")
    A("")
    A("### §7.4.4 十三字段门槛记录")
    A("")
    A("| direction | gate_status | n_fits | EC-AUROC | r_violations | artifact_sha256 | approved_by |")
    A("|---|---|---|---|---|---|---|")
    for d, rec in doc["gate_records"].items():
        A(f"| {d} | `{rec['gate_status']}` | {rec['n_fits']} | "
          f"{_fmt(rec['expert_choice_auroc'])} | {len(rec['r_violations'])} | "
          f"`{rec['artifact_sha256']}` | {rec['approved_by'] or '未签'} |")
    A("")
    A(f"第 1 步不产出、因而显式标注 `{NOT_APPLICABLE_STEP1}` 的字段："
      "`delta_r_ci_low`、`delta_auc_ci_low`、`capture_ratio`、`f_family_hat`"
      "（属 §7.4.2 第 2 步；✗ 不伪造 0，✗ 不据此类字段判 `not_run`）。")
    A("")

    # ---- 8 判读与处置
    A("## 8　判读规则与处置")
    A("")
    A("| 观察 | 判读 | 处置 |")
    A("|---|---|---|")
    A("| `EC-AUROC <= 0.5`（任一方向） | 机制门槛 ① 未过 | 不得启动第 2 步（`R-5` 语义）；"
      "Router 转声明式次要／负结果（`§7.4.1 未过处置`、`§8.3 L-27`）；"
      "✗ 不得追加特征/换模型/重跑救 |")
    A("| A-1…A-5 任一 FAIL | 机制门槛 ② 未过 | 修 manifest/数据版本后**重新生成**该方向产物；"
      "✗ 不得在本地改断言口径 |")
    A("| 存在 R-7 违例 | 机制门槛 ③ 未过 | 该轮**作废**，不得进端点表/`ΔR_macro`/稿件；"
      "✗ 不得重跑救（`§7.2 R-7`） |")
    A("| 三项全过 | 第 1 步**机械放行** | 只放行第 2 步；✗ 不是 GO、"
      "✗ 不是「机制已证实」（`§7.4` 引言、`§7.4.6`） |")
    A("")

    # ---- 9 完整性
    A("## 9　输入完整性（逐文件哈希）")
    A("")
    A("| 方向 | fold | 家族 | `oof_pred_sha256` | 与 run_metadata 一致 |")
    A("|---|---|---|---|---|")
    for d, r in doc["directions"].items():
        for fold, pf in r["per_fold"].items():
            for fam in FAMILIES:
                md = pf[f"run_metadata_{fam}"]
                sha = md.get("oof_pred_sha256_actual")
                if md.get("sha256_match") is True:
                    mark = "yes"
                elif not md.get("present"):
                    mark = "no run_metadata"
                else:
                    mark = "NO"
                A(f"| {d} | {fold} | {fam} | `{sha}` | {mark} |")
    A("")
    A("运行环境与免责：")
    A("")
    A("- 本脚本**零 GPU**：不导入 torch、不写 `DONE.json`/checkpoint、不训练任何模型。")
    A("- 本脚本**只新增文件**：未修改 `src/*`、`scripts/30_oof_pilot.py`、"
      "`scripts/98_loader_v2.py` 或任何锁件。")
    A("- 本报告**只**承载机械判读；科学结论、等效性主张与 GO 判定均不在本文件范围内。")
    A("")
    return "\n".join(L)


# ------------------------------------------------------------------ 主控
def collect_environment() -> dict:
    def _ver(name):
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            return "absent"

    return {
        "python": sys.version.split()[0],
        "numpy": _ver("numpy"),
        "pandas": _ver("pandas"),
        "sklearn": _ver("sklearn"),
        "torch": _ver("torch"),
        "torch_imported": "torch" in sys.modules,
    }


def execute(root, tag: str, directions, seed: int, out_dir, expert_prob: str,
            units: pd.DataFrame, man: pd.DataFrame, probe_doc: dict,
            expected_coverage=None, allow_scale_mismatch: bool = False,
            overwrite: bool = False) -> dict:
    """跑全流程（每方向 → 门槛 → 报告）。返回报告 dict，并写盘。"""
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{REPORT_STEM}.md"
    js_path = out_dir / f"{REPORT_STEM}.json"
    if not overwrite:
        for p in (md_path, js_path):
            if p.exists():
                raise GateInputError(
                    f"{p} 已存在。为避免覆盖历史判定，默认拒绝重写；确需重跑请加 --overwrite。"
                )

    timestamp = utc_now()
    scale_expectation = expected_coverage or {
        d: int(((units["library"] == DIRECTIONS[d]["source"]) & units["evaluable"]).sum())
        for d in DIRECTIONS
    }
    dir_results, r7_by_dir, tables = {}, {}, {}
    for direction in directions:
        res = run_direction(root, tag, direction, seed, out_dir, expert_prob,
                            units, man, allow_scale_mismatch=allow_scale_mismatch,
                            expected_units=scale_expectation)
        tables[direction] = res.pop("table_object")
        dir_results[direction] = res
    # R-7 的两方向交叉断言要求两张表都在手，故放在第二遍（✗ 不能边算边判）
    for direction in directions:
        r7_by_dir[direction] = check_r7(
            dir_results[direction]["_rows_df"].to_dict("records"), direction, units, man,
            tables[direction], tables,
        )

    supervision = evaluate_supervision(man, units, expected_coverage=expected_coverage)
    verdict = mechanical_verdict(
        {d: {"expert_choice_auroc": r["expert_choice_auroc"]} for d, r in dir_results.items()},
        supervision, r7_by_dir,
    )
    records = {
        d: gate_record(d, r, supervision, r7_by_dir[d], verdict, timestamp)
        for d, r in dir_results.items()
    }

    doc = {
        "script": "scripts/40_merge_oof_and_gates.py",
        "version": "v1",
        "kind": "oof_merge_plus_step1_mechanism_gate",
        "gpu": False, "trained": False,
        "timestamp_utc": timestamp,
        "root": str(root),
        "tag": tag,
        "seed": int(seed),
        "out_dir": str(out_dir),
        "directions_run": list(directions),
        "expert_prob": expert_prob,
        "expert_prob_note": (
            "v1 = 锁件口径 p_e = p_v1（features_v2.build_feature_row 的默认语义 / B-02；"
            "engine_v2 同行 C_expert/loss_expert 亦为 p_v1 口径）。"
            "mean4 = 敏感性 p_e = mean(p_v1..p_v4)，C_e/ℓ_e 由均值重算，"
            "与同文件 C_expert/loss_expert 列不再逐行相等。"
        ),
        "env": collect_environment(),
        "probe": probe_doc,
        "supervision": supervision,
        "r7": r7_by_dir,
        "verdict": verdict,
        "gate_records": records,
        "gate_record_fields": list(GATE_RECORD_FIELDS),
        "targets": {
            "step1_thresholds": {
                "expert_choice_auroc_gt": EC_AUROC_THRESHOLD,
                "A_items": ["A-1", "A-2", "A-3", "A-4", "A-5"],
                "A2_wording": "C-104a 方向内覆盖（DIR-A 1385 / DIR-B 360）",
                "r7": "无越界监督源",
            },
            "n_fits_total": N_FITS_TOTAL,
            "n_fits_per_direction": N_FITS_PER_DIRECTION,
            "frozen_dir_coverage": FROZEN_DIR_COVERAGE,
            "references": {
                "gate": "Protocol_v2_v1_1_FINAL.md §7.4.1",
                "record": "Protocol_v2_v1_1_FINAL.md §7.4.4",
                "ec_auroc": "Protocol §6.2 首行 / §5.4 L-11.4",
                "a_items": "Protocol §5.6 L-44(b)",
                "a2": "Lock_R1_v2_draft_v1.2_amendment.md §A-1 (C-104a)",
                "r7": "Protocol §7.2 R-7",
                "tau_C": "Lock_R1_v2_draft_v1.1.md §7 B-01 (L475)",
                "gate_struct": "Lock_R1_v2_draft_v1.1.md §7 B-05 (L479)",
                "missing": "Lock_R1_v2_draft_v1.1.md §7 B-12 (L486)",
                "views": "Lock_R1_v2_draft_v1.1.md §7 B-02 (L476)",
                "table": "_router_plan.txt L462 / L135-L157",
            },
        },
        "directions": {},
    }
    for d, r in dir_results.items():
        r2 = dict(r)
        r2.pop("_rows_df", None)
        doc["directions"][d] = r2

    doc = jsonable(doc)
    md_path.write_text(render_markdown(doc), encoding="utf-8", newline="\n")
    js_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    doc["_written"] = {"markdown": str(md_path), "json": str(js_path)}
    return doc


# ------------------------------------------------------------- 合成数据自检
def _synth_preds(uid: str, y: int, signal: bool = True):
    """确定性的合成 4 视图概率（无数据、无模型）。"""
    seed = int(hashlib.sha256(uid.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    idx = int(uid.split("u")[-1])
    m4_better = ((idx // 2) % 2) == 1
    good = 0.2 + 0.6 * y
    near = 0.55 - 0.10 * y
    if not signal:
        # 平坦：两个专家都不带信息 → 硬标签仍均衡，但 5 维信号无判别力
        good = near = 0.5
        m4_better = (idx % 2) == 1
    if m4_better:
        p3 = np.clip(near + rng.normal(0, 0.02, N_VIEWS), 0.02, 0.98)
        p4 = np.clip(good + rng.normal(0, 0.02, N_VIEWS), 0.02, 0.98)
    else:
        p3 = np.clip(good + rng.normal(0, 0.02, N_VIEWS), 0.02, 0.98)
        p4 = np.clip(near + rng.normal(0, 0.02, N_VIEWS), 0.02, 0.98)
    return p3.tolist(), p4.tolist()


def synth_project(base: Path, n_patients: int = 12, per_patient: int = 5,
                  signal: bool = True, seed: int = 5) -> dict:
    """在 base 下造一份**形状与真实工程一致**的合成工程（✗ 不碰真实产物）。"""
    base = Path(base)
    sb = base / STAGEB_DIR
    sb.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    unit_rows, fold_of, patient_of, cluster_of = [], {}, {}, {}
    for direction in ("DIR-A", "DIR-B"):
        lib = DIRECTIONS[direction]["source"]
        for p in range(n_patients):
            fold = (p % len(FOLDS)) + 1
            for j in range(per_patient):
                uid = f"{lib}_u{p * per_patient + j:04d}"
                y = (p * per_patient + j) % 2
                fold_of[uid] = fold
                patient_of[uid] = f"{lib}-p{p:03d}"
                cluster_of[uid] = f"{lib}-c{p:03d}"
                unit_rows.append({
                    "unit_id": uid, "library": lib,
                    "original_patient_id": f"{lib}-p{p:03d}",
                    "cluster_id": f"{lib}-c{p:03d}",
                    "n_frames_decodable": 40 + int(rng.integers(0, 12)),
                    "evaluable": "TRUE", "label_bline": int(y), "fold": int(fold),
                })
    units = pd.DataFrame(unit_rows)
    units.to_csv(sb / UNITS_CSV, index=False, lineterminator="\n")

    # 折清单（真实工程里由 98_loader_v2 读；本脚本不依赖，但保持形状一致）
    fa = units.rename(columns={"cluster_id": "cluster_id_v3"}).copy()
    fa[["unit_id", "cluster_id_v3", "fold", "library", "original_patient_id"]].to_csv(
        sb / "fold_assignment_v3p1.csv", index=False, lineterminator="\n"
    )

    u = normalise_units(units)
    uby = u.set_index("unit_id")
    rows = []
    for direction in ("DIR-A", "DIR-B"):
        src = DIRECTIONS[direction]["source"]
        eu = u[(u["library"] == src) & u["evaluable"]]
        for fold in FOLDS:
            allowed = set(eu[eu["fold"] == fold]["unit_id"])
            forbidden = set(u[u["library"] != src]["unit_id"])
            rows.append({
                "direction": direction, "gate_supervision_source": src,
                "target_domain": DIRECTIONS[direction]["target"],
                "fold": fold, "oof_role": "source_domain_out_of_fold",
                "gate_eligible": "TRUE", "allowed_scope": "source_domain_evaluable_oof",
                "allowed_sha256": unit_set_sha256(allowed), "allowed_count": len(allowed),
                "forbidden_sha256": unit_set_sha256(forbidden), "forbidden_count": len(forbidden),
                "allowed_forbidden_intersection_n": 0,
            })
    rows.append({
        "direction": "MERGED_ALL", "gate_supervision_source": "ALL", "target_domain": "",
        "fold": "ALL", "oof_role": "storage_descriptive_only", "gate_eligible": "FALSE",
        "allowed_scope": "all_units", "allowed_sha256": unit_set_sha256(set(u["unit_id"])),
        "allowed_count": len(u), "forbidden_sha256": unit_set_sha256(set()),
        "forbidden_count": 0, "allowed_forbidden_intersection_n": 0,
    })
    pd.DataFrame(rows).to_csv(sb / OOF_MANIFEST_CSV, index=False, lineterminator="\n")

    # 20 个 OOF 预测文件 + run_metadata.json（与 engine_v2 同一列序 / 同一目录约定）
    tag = DEFAULT_TAG
    lib_of = {k: str(v) for k, v in zip(uby.index, uby["library"])}
    y_of = {k: int(v) for k, v in zip(uby.index, uby["label"])}
    nf_of = {k: int(v) for k, v in zip(uby.index, uby["n_frames"])}
    for direction in ("DIR-A", "DIR-B"):
        src = DIRECTIONS[direction]["source"]
        for fold in FOLDS:
            uids = sorted([x for x, f in fold_of.items() if f == fold and lib_of[x] == src])
            for family in FAMILIES:
                preds = {}
                for uid in uids:
                    p3, p4 = _synth_preds(uid, y_of[uid], signal=signal)
                    preds[uid] = (p3 if family == "M3" else p4)
                recs = []
                for uid in uids:
                    pv = preds[uid]
                    y = y_of[uid]
                    recs.append({
                        "unit_id": uid,
                        "patient_id": patient_of[uid],
                        "cluster_id": cluster_of[uid],
                        "library": src,
                        "direction": direction,
                        "role": ROLE_SOURCE_OOF,
                        "family": family,
                        "seed": PILOT_SEED,
                        "fold": int(fold),
                        "status": "OK",
                        "y": y,
                        "n_frames": nf_of[uid],
                        "crop_clamped": bool(F.crop_clamped(nf_of[uid])),
                        "p_v1": pv[0], "p_v2": pv[1], "p_v3": pv[2], "p_v4": pv[3],
                        "C_expert": F.max_class_prob(pv[0]),
                        "T_expert": F.temporal_instability(pv),
                        "loss_expert": F.brier_loss(pv[0], y),
                        "best_epoch": 5,
                    })
                d = fold_dir(base, tag, direction, family, PILOT_SEED, fold)
                d.mkdir(parents=True, exist_ok=True)
                f = d / f"oof_pred_{family}_{direction}_f{fold}.csv"
                pd.DataFrame(recs, columns=list(OOF_REQUIRED_COLUMNS)).to_csv(
                    f, index=False, lineterminator="\n", float_format="%.17g"
                )
                meta = {
                    "project": "TRUST-LUS V4.2 (SYNTHETIC)", "family": family,
                    "direction": direction, "seed": PILOT_SEED, "fold": int(fold),
                    "config_sha256": "c" * 64, "data_version_sha256": "d" * 64,
                    "training_contract_sha256": "t" * 64,
                    "environment_reference_sha256": "e" * 64,
                    "lock_hashes": {"lock_r1_v2_draft_v1_1_sha256": "a" * 64,
                                    "protocol_v2_v1_1_final_sha256": "b" * 64},
                    "source_manifest_unit_set_sha256": unit_set_sha256(uids),
                    "val_units": len(uids), "run_kind": "precision_pilot_oof",
                    "formal_result": False, "git_commit": "SYNTHETIC",
                    "oof_pred_file": f.name, "oof_pred_sha256": sha256_file(f),
                }
                (d / "run_metadata.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n",
                )

    man = normalise_manifest(pd.read_csv(sb / OOF_MANIFEST_CSV, dtype=str, low_memory=False))
    return {
        "root": base, "units": u, "man": man,
        "expected_coverage": {
            d: int(((u["library"] == DIRECTIONS[d]["source"]) & u["evaluable"]).sum())
            for d in DIRECTIONS
        },
    }


def _selftest() -> int:
    """合成数据全流程自检（✗ 不碰真实产物；全部在临时目录）。"""
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        if not cond:
            print(f"[FAIL] {name} {detail}")

    def raises(fn, exc=Exception):
        try:
            fn()
        except exc:
            return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def try_err(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            return str(e)
        return ""

    check("torch not imported by the script", "torch" not in sys.modules)
    check("script is zero-GPU (AST: no torch/engine/train calls)",
          _ast_allows_no_training(Path(__file__)))

    with tempfile.TemporaryDirectory() as td:
        proj = synth_project(Path(td) / "proj", n_patients=12, per_patient=5, signal=True)
        root = proj["root"]
        exp = proj["expected_coverage"]

        # ---------- 1 探查
        pr = probe(root)
        v = pr["verdict"]
        check("probe: 20 OOF files present", v["n_present"] == 20, v["n_present"])
        check("probe: all have p_v1..p_v4", v["n_with_four_views"] == 20, v["n_with_four_views"])
        check("probe: features buildable", v["can_build_features"] is True)
        e0 = next(e for e in pr["entries"] if e.get("exists"))
        check("probe: columns == frozen OOF contract",
              tuple(e0["columns"]) == OOF_REQUIRED_COLUMNS, e0["columns"])
        check("probe: prints 2 example rows", len(e0["first_rows"]) == 2)
        check("probe: n_views == 4", e0.get("has_four_views") is True)

        # ---------- 2 schema 断言（缺列如实报错，✗ 不修补）
        f1 = oof_path(root, DEFAULT_TAG, "DIR-A", "M3", PILOT_SEED, 1)
        df = pd.read_csv(f1)
        check("schema: full contract accepted",
              verify_oof_schema(df, f1, "DIR-A", "M3", 1)["n_views"] == 4)
        broken = df.drop(columns=["p_v3", "p_v4"])
        err = try_err(lambda: verify_oof_schema(broken, f1, "DIR-A", "M3", 1))
        check("schema: missing p_v3/p_v4 raises OofContractError", "p_v3" in err, err[:80])
        check("schema: error names the engine-side contract",
              "engine_v2" in err or "引擎侧" in err)
        bad_dir = df.copy()
        bad_dir["direction"] = "DIR-B"
        check("schema: wrong direction column refused",
              raises(lambda: verify_oof_schema(bad_dir, f1, "DIR-A", "M3", 1), OofContractError))
        bad_lib = df.copy()
        bad_lib["library"] = "covid_blues"
        check("schema: wrong library refused (R-7)",
              raises(lambda: verify_oof_schema(bad_lib, f1, "DIR-A", "M3", 1), OofContractError))
        bad_nan = df.copy()
        bad_nan.loc[0, "p_v2"] = np.nan
        check("schema: NaN view refused (B-12 no imputation)",
              raises(lambda: verify_oof_schema(bad_nan, f1, "DIR-A", "M3", 1), OofContractError))

        # ---------- 3 run_metadata 哈希核对
        md = verify_run_metadata(root, DEFAULT_TAG, "DIR-A", "M3", PILOT_SEED, 1, f1)
        check("run_metadata: sha256 match recorded", md["present"] and md["sha256_match"] is True)
        check("run_metadata: lock/protocol hashes surfaced",
              len(str(md["lock_r1_v2_draft_v1_1_sha256"])) == 64
              and len(str(md["protocol_v2_v1_1_final_sha256"])) == 64)
        tampered = Path(td) / "tampered.csv"
        tampered.write_text(f1.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        check("run_metadata: tampered file refused",
              raises(lambda: verify_run_metadata(root, DEFAULT_TAG, "DIR-A", "M3",
                                                 PILOT_SEED, 1, tampered), GateInputError))

        # ---------- 4 合并 / 单元级特征
        mg = merge_direction(root, DEFAULT_TAG, "DIR-A", PILOT_SEED, EXPERT_PROB_V1,
                             expected_units=exp)
        n_expected = exp["DIR-A"]
        check("merge: row count == fold partition", mg["n_merged"] == n_expected,
              (mg["n_merged"], n_expected))
        check("merge: scale guard exact", mg["scale"]["status"] == "exact", mg["scale"])
        check("merge: all folds present", sorted(mg["per_fold"]) == list(FOLDS))
        check("merge: features come from build_feature_row (5 finite names)",
              all(_finite(mg["kept"][0][n]) for n in F.FEATURE_NAMES), F.FEATURE_NAMES)
        check("merge: p3 carries the expert_prob tag (v1 semantics, B-02)",
              mg["kept"][0]["expert_prob"] == EXPERT_PROB_V1)
        m4 = merge_direction(root, DEFAULT_TAG, "DIR-A", PILOT_SEED, EXPERT_PROB_MEAN4,
                             expected_units=exp)
        k0, k0m = mg["kept"][0], m4["kept"][0]
        check("mean4: expert_prob recorded per row", k0m["expert_prob"] == EXPERT_PROB_MEAN4)
        check("mean4: C_3 recomputed as max(p3,1-p3)",
              abs(k0m["C_3"] - max(k0m["p3"], 1 - k0m["p3"])) < 1e-12)
        check("mean4: p3 != v1 p3 only when the 4 views differ",
              isinstance(k0["p3"], float) and isinstance(k0m["p3"], float))

        # y 不一致必须报错
        proj2 = synth_project(Path(td) / "proj_y", n_patients=5, per_patient=4)
        f_m4 = oof_path(proj2["root"], DEFAULT_TAG, "DIR-A", "M4", PILOT_SEED, 1)
        d4 = pd.read_csv(f_m4)
        d4.loc[0, "y"] = 1 - int(d4.loc[0, "y"])
        d4.to_csv(f_m4, index=False, lineterminator="\n", float_format="%.17g")
        check("merge: y mismatch across families raises",
              raises(lambda: merge_direction(proj2["root"], DEFAULT_TAG, "DIR-A",
                                             PILOT_SEED, EXPERT_PROB_V1), GateInputError))

        # 单元集不一致必须报错
        proj3 = synth_project(Path(td) / "proj_set", n_patients=5, per_patient=4)
        f_m3 = oof_path(proj3["root"], DEFAULT_TAG, "DIR-A", "M3", PILOT_SEED, 1)
        d3 = pd.read_csv(f_m3).iloc[:-1]
        d3.to_csv(f_m3, index=False, lineterminator="\n", float_format="%.17g")
        check("merge: unit-set mismatch between families raises",
              raises(lambda: merge_direction(proj3["root"], DEFAULT_TAG, "DIR-A",
                                             PILOT_SEED, EXPERT_PROB_V1), GateInputError))

        # ---------- 5 监督契约 A-1…A-5（实算 + 逐条变异）
        man, units = proj["man"], proj["units"]
        sup = evaluate_supervision(man, units, expected_coverage=exp)
        check("A-1 PASS on pristine manifest", sup["A1"] is True)
        check("A-2 PASS on pristine manifest (direction-level coverage)", sup["A2"] is True)
        check("A-3 PASS on pristine manifest", sup["A3"] is True)
        check("A-4 PASS on pristine manifest", sup["A4"] is True)
        check("A-5 PASS on pristine manifest", sup["A5"] is True)
        check("A-2: direction coverage equals expected per direction",
              all(sup["direction_coverage"][d]["union_allowed"] == exp[d] for d in DIRECTIONS),
              sup["direction_coverage"])
        check("A-2: old 1 897 reading is False and reported",
              set(sup["old_reading_holds"]) == {False}, sup["old_reading_holds"])
        check("A-2: MERGED_ALL row is storage-only (gate_eligible False)",
              sup["merged_all_gate_eligible"] == [False])
        check("A-5: 11 rows == 5 + 5 + 1", sup["n_rows"] == 11, sup["n_rows"])
        check("A-5: 10 gate-eligible rows", sup["n_gate_eligible_rows"] == 10)

        m = man.copy()
        m.loc[0, "allowed_sha256"] = ""
        check("A-1 FAIL when a required column is empty",
              evaluate_supervision(m, units, exp)["A1"] is False)
        m = man.copy()
        m.loc[0, "allowed_forbidden_intersection_n"] = "1"
        check("A-3 FAIL when declared intersection != 0",
              evaluate_supervision(m, units, exp)["A3"] is False)
        m = man.copy()
        m.loc[0, "gate_supervision_source"] = DIRECTIONS["DIR-A"]["target"]
        check("A-4 FAIL when declared source is the wrong library",
              evaluate_supervision(m, units, exp)["A4"] is False)
        m = man.copy()
        m.loc[0, "allowed_sha256"] = "0" * 64
        check("A-5 FAIL on declared-hash mismatch",
              evaluate_supervision(m, units, exp)["A5"] is False)
        units_x = units.copy()
        extra = units_x[units_x["library"] == DIRECTIONS["DIR-A"]["source"]].head(3).copy()
        extra["unit_id"] = [u + "_x" for u in extra["unit_id"]]
        units_x = pd.concat([units_x, extra], ignore_index=True)
        check("A-2 FAIL when direction-level coverage drifts",
              evaluate_supervision(man, units_x, exp)["A2"] is False)
        check("A-5 FAIL when a fold row is missing",
              evaluate_supervision(
                  man.drop(index=man.index[(man["direction"] == "DIR-A") & (man["fold"] == "5")]),
                  units, exp)["A5"] is False)

        # ---------- 6 端到端（正信号 → GO）
        out1 = Path(td) / "out1"
        doc = execute(root, DEFAULT_TAG, ("DIR-A", "DIR-B"), PILOT_SEED, out1,
                      EXPERT_PROB_V1, units, man, pr, expected_coverage=exp)
        check("e2e: both tables written",
              all(table_path(out1, d).exists() for d in DIRECTIONS))
        check("e2e: 14 fields per table",
              all(len(doc["directions"][d]["table_fields"]) == 14 for d in DIRECTIONS),
              {d: doc["directions"][d]["n_table_fields"] for d in DIRECTIONS})
        check("e2e: field order is the frozen 14",
              all(tuple(doc["directions"][d]["table_fields"]) == R.TABLE_FIELDS
                  for d in DIRECTIONS))
        check("e2e: tables read back through read_table",
              all(len(R.read_table(table_path(out1, d))) > 0 for d in DIRECTIONS))
        check("e2e: tau in frozen grid",
              all(doc["directions"][d]["tau"] in R.TAU_GRID for d in DIRECTIONS))
        check("e2e: C in frozen grid",
              all(doc["directions"][d]["C"] in R.C_GRID for d in DIRECTIONS))
        check("e2e: EC-AUROC > 0.5 on the synthetic signal (both directions)",
              all(doc["directions"][d]["ec_auroc_pass"] for d in DIRECTIONS),
              {d: doc["directions"][d]["expert_choice_auroc"] for d in DIRECTIONS})
        check("e2e: authoritative verdict is GO on a clean synthetic run",
              doc["verdict"]["label"] == "GO", doc["verdict"]["failed_items"])
        check("e2e: all three gate families pass",
              doc["supervision"]["A1"] and doc["supervision"]["A2"]
              and doc["supervision"]["A3"] and doc["supervision"]["A4"]
              and doc["supervision"]["A5"]
              and all(doc["r7"][d]["clean"] for d in DIRECTIONS))
        check("e2e: 13-field record complete for both directions",
              all(all(k in doc["gate_records"][d] for k in GATE_RECORD_FIELDS)
                  for d in DIRECTIONS))
        check("e2e: step-2 quantities marked not_applicable_step1",
              all(doc["gate_records"][d]["f_family_hat"] == NOT_APPLICABLE_STEP1
                  and doc["gate_records"][d]["capture_ratio"] == NOT_APPLICABLE_STEP1
                  for d in DIRECTIONS))
        check("e2e: reports written",
              (out1 / f"{REPORT_STEM}.md").exists()
              and (out1 / f"{REPORT_STEM}.json").exists())
        check("e2e: JSON parses and is self-consistent",
              json.loads((out1 / f"{REPORT_STEM}.json").read_text(encoding="utf-8")
                         )["verdict"]["label"] == "GO")
        check("e2e: markdown states gate != GO",
              "门 ≠ GO" in (out1 / f"{REPORT_STEM}.md").read_text(encoding="utf-8"))
        check("e2e: table hashes recorded (64 hex)",
              all(len(doc["directions"][d]["table_sha256"]) == 64
                  and len(doc["directions"][d]["table_file_sha256"]) == 64
                  for d in DIRECTIONS))
        out1b = Path(td) / "out1b"
        doc_b = execute(root, DEFAULT_TAG, ("DIR-A",), PILOT_SEED, out1b,
                        EXPERT_PROB_V1, units, man, pr, expected_coverage=exp)
        check("e2e: deterministic table_sha256 across runs",
              doc_b["directions"]["DIR-A"]["table_sha256"]
              == doc["directions"]["DIR-A"]["table_sha256"])
        check("e2e: no DONE.json written", not list(Path(td).rglob("DONE.json")))
        check("e2e: single-direction run is labelled PARTIAL, never GO",
              json.loads((out1b / f"{REPORT_STEM}.json").read_text(encoding="utf-8")
                         )["verdict"]["label"] == "PARTIAL")
        check("e2e: no checkpoint files written",
              not [p for p in Path(td).rglob("*") if p.suffix in (".pt", ".pth")])
        check("e2e: torch still not imported", "torch" not in sys.modules)
        check("e2e: refuses to overwrite an existing report",
              raises(lambda: execute(root, DEFAULT_TAG, ("DIR-A",), PILOT_SEED, out1b,
                                     EXPERT_PROB_V1, units, man, pr, expected_coverage=exp),
                     GateInputError))
        check("e2e: overwrite works when asked",
              execute(root, DEFAULT_TAG, ("DIR-A",), PILOT_SEED, out1b, EXPERT_PROB_V1,
                      units, man, pr, expected_coverage=exp,
                      overwrite=True)["verdict"]["label"] in ("GO", "NO-GO", "PARTIAL"))

        # ---------- 7 mean4 敏感性路径
        out2 = Path(td) / "out2"
        doc2 = execute(root, DEFAULT_TAG, ("DIR-A",), PILOT_SEED, out2, EXPERT_PROB_MEAN4,
                       units, man, pr, expected_coverage=exp)
        check("mean4: separate table file written", table_path(out2, "DIR-A", "_mean4").exists())
        check("mean4: expert_prob recorded in the doc", doc2["expert_prob"] == EXPERT_PROB_MEAN4)
        check("mean4: 14 fields still hold", doc2["directions"]["DIR-A"]["n_table_fields"] == 14)

        # ---------- 8 R-7 逐条触发
        kept = merge_direction(root, DEFAULT_TAG, "DIR-A", PILOT_SEED, EXPERT_PROB_V1,
                               expected_units=exp)["kept"]
        injected = [dict(r) for r in kept]
        tgt_units = units[units["library"] == DIRECTIONS["DIR-A"]["target"]]["unit_id"].tolist()
        injected[0] = dict(injected[0], unit_id=tgt_units[0])
        r7 = check_r7(injected, "DIR-A", units, man, R.read_table(table_path(out1, "DIR-A")), {})
        check("R-7: target-domain unit in supervision detected", r7["clean"] is False)
        check("R-7: violation cites Protocol §7.2 R-7",
              any("R-7" in v["reference"] for v in r7["violations"]))
        r7_ok = check_r7(kept, "DIR-A", units, man,
                         R.read_table(table_path(out1, "DIR-A")),
                         {"DIR-A": R.read_table(table_path(out1, "DIR-A")),
                          "DIR-B": R.read_table(table_path(out1, "DIR-B"))})
        check("R-7: clean rows produce no violation", r7_ok["clean"] is True, r7_ok["violations"])
        check("R-7: cross-direction assertion ran", r7_ok["cross_direction"]["ok"] is True)
        tb = R.read_table(table_path(out1, "DIR-B"))
        r7_bad = check_r7(kept, "DIR-A", units, man, tb, {})
        check("R-7: table from the wrong dataset fails the dataset check",
              r7_bad["clean"] is False)

        # ---------- 9 机械判定逻辑（边界 + 逐项否决）
        sup_ok = {"A1": True, "A2": True, "A3": True, "A4": True, "A5": True}
        r7_clean = {"DIR-A": {"clean": True}, "DIR-B": {"clean": True}}
        ec_hi = {"DIR-A": {"expert_choice_auroc": 0.62}, "DIR-B": {"expert_choice_auroc": 0.58}}
        check("verdict: GO only when everything passes",
              mechanical_verdict(ec_hi, sup_ok, r7_clean)["label"] == "GO")
        check("verdict: EC-AUROC == 0.5 exactly is NOT a pass (strict >)",
              mechanical_verdict({"DIR-A": {"expert_choice_auroc": 0.5},
                                  "DIR-B": {"expert_choice_auroc": 0.58}},
                                 sup_ok, r7_clean)["label"] == "NO-GO")
        check("verdict: None EC-AUROC is a fail",
              mechanical_verdict({"DIR-A": {"expert_choice_auroc": None},
                                  "DIR-B": {"expert_choice_auroc": 0.58}},
                                 sup_ok, r7_clean)["label"] == "NO-GO")
        for key in ("A1", "A2", "A3", "A4", "A5"):
            s = dict(sup_ok)
            s[key] = False
            check(f"verdict: {key} FAIL forces NO-GO",
                  mechanical_verdict(ec_hi, s, r7_clean)["label"] == "NO-GO")
        check("verdict: any R-7 violation forces NO-GO",
              mechanical_verdict(ec_hi, sup_ok,
                                 {"DIR-A": {"clean": False}, "DIR-B": {"clean": True}}
                                 )["label"] == "NO-GO")
        check("verdict: a missing direction can never yield GO",
              mechanical_verdict({"DIR-A": {"expert_choice_auroc": 0.7}}, sup_ok, r7_clean
                                 )["label"] == "PARTIAL"
              and mechanical_verdict({"DIR-A": {"expert_choice_auroc": 0.7}}, sup_ok,
                                     r7_clean)["go"] is False)
        check("verdict: pricing premise never blocks step 1",
              mechanical_verdict(ec_hi, sup_ok, r7_clean)["pricing_premise"]["status"]
              == NOT_APPLICABLE_STEP1)

        # ---------- 10 量级护栏
        proj_small = synth_project(Path(td) / "proj_small", n_patients=5, per_patient=2)
        check("scale guard: obvious mismatch raises",
              raises(lambda: merge_direction(proj_small["root"], DEFAULT_TAG, "DIR-A",
                                             PILOT_SEED, EXPERT_PROB_V1), GateInputError))
        check("scale guard: --allow-scale-mismatch downgrades to warn",
              merge_direction(proj_small["root"], DEFAULT_TAG, "DIR-A", PILOT_SEED,
                              EXPERT_PROB_V1, allow_scale_mismatch=True)["scale"]["status"]
              == "fail")

        # ---------- 11 探查：缺文件如实报告
        empty = Path(td) / "empty"
        empty.mkdir()
        pr_empty = probe(empty)
        check("probe: absent files reported, not invented",
              pr_empty["verdict"]["n_present"] == 0
              and pr_empty["verdict"]["can_build_features"] is False)

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print("=" * 78)
    print(f"40_merge_oof_and_gates self-test: {n_pass}/{len(checks)} PASS")
    print(f"fits={N_FITS_TOTAL} = {len(FAMILIES)} families x {len(FOLDS)} folds x 2 directions")
    print(f"gate = EC-AUROC > {EC_AUROC_THRESHOLD} (per direction) AND A-1..A-5 AND no R-7")
    print("=" * 78)
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print(json.dumps({"status": "FAIL", "failed": failed}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", "n_checks": len(checks)}, ensure_ascii=False))
    return 0


def _ast_allows_no_training(path: Path) -> bool:
    """AST 级核对：不导入 torch、不导入训练引擎、不调用 train_one_fold。"""
    import ast

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    imported, called = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name:
                called.add(name)
    ok = not any(m == "torch" or m.startswith("torch.") for m in imported)
    ok = ok and not any("engine" in m for m in imported if m)
    ok = ok and "train_one_fold" not in called
    return ok


# ------------------------------------------------------------------ CLI
def load_supervision(root) -> tuple:
    """读 units + 监督 manifest（真实模式走 stageB_data_v3p1 的冻结件）。"""
    root = Path(root)
    units_p = root / STAGEB_DIR / UNITS_CSV
    man_p = root / STAGEB_DIR / OOF_MANIFEST_CSV
    if not units_p.exists():
        raise GateInputError(f"缺少 {units_p}（A-1…A-5 无法实算）")
    if not man_p.exists():
        raise GateInputError(f"缺少 {man_p}（A-1…A-5 无法实算）")
    units = normalise_units(pd.read_csv(units_p, dtype=str, low_memory=False))
    man = normalise_manifest(pd.read_csv(man_p, dtype=str, low_memory=False))
    return units, man


def _force_utf8_stdio() -> None:
    """把 stdout/stderr 切到 UTF-8（Windows GBK 控制台否则会因中文/✗ 崩掉）。

    只是防崩溃（errors="replace"），✗ 不改变任何判定；报告文件始终以 UTF-8 落盘。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 老解释器/被重定向时无 reconfigure
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="TRUST-LUS OOF 合并 + 第 1 步机制门槛（零 GPU / 纯 CPU）"
    )
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="工程根（含 outputs/ 与 stageB_data_v3p1/）")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="引擎 stage tag 基名（dirA/dirB 后缀自动加）")
    ap.add_argument("--seed", type=int, default=PILOT_SEED)
    ap.add_argument("--direction", choices=("DIR-A", "DIR-B", "ALL"), default="ALL")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"产物目录（默认 <root>/{DEFAULT_OUT_SUBDIR}）")
    ap.add_argument("--expert-prob", choices=EXPERT_PROB_MODES, default=DEFAULT_EXPERT_PROB,
                    help="v1=锁件口径 p_e=p_v1（默认）；mean4=p_e=mean(p_v1..p_v4)（敏感性）；both=两套都跑")
    ap.add_argument("--expected-coverage", choices=("frozen", "measured"), default="frozen",
                    help="A-2 方向级期望值：frozen=1 385/360（锁件，默认）；measured=按本机 units 现算")
    ap.add_argument("--allow-scale-mismatch", action="store_true",
                    help="合并行数与源域单元集偏差 >10% 时仍继续（✗ 不推荐；会记 warn）")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的报告/表")
    ap.add_argument("--probe", action="store_true", help="只探查 oof_pred_*.csv 的列名与前 2 行")
    ap.add_argument("--self-test", action="store_true", help="合成数据全流程自检（✗ 不碰真实产物）")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    _force_utf8_stdio()

    if args.self_test:
        return _selftest()

    directions = ("DIR-A", "DIR-B") if args.direction == "ALL" else (args.direction,)
    probe_doc = probe(root, args.tag, args.seed, directions)

    if args.probe:
        v = probe_doc["verdict"]
        for e in probe_doc["entries"]:
            if not e["exists"]:
                print(f"[MISSING] {e['path']}")
                continue
            print("=" * 78)
            print(f"[{e['direction']}/{e['family']}/f{e['fold']}] {e['path']}")
            print(f"  rows={e['n_rows']}  cols={e['n_columns']}  bytes={e['size_bytes']}  "
                  f"p_v1..p_v4={'YES' if e['has_four_views'] else 'NO'}")
            print(f"  columns: {', '.join(e['columns'])}")
            if e["missing_contract_columns"]:
                print(f"  MISSING CONTRACT COLUMNS: {e['missing_contract_columns']}")
            print("  --- first 2 rows ---")
            print(e["head2_text"])
        print("=" * 78)
        print(json.dumps(v, ensure_ascii=True, indent=2))
        return 0 if v["can_build_features"] else 1

    if not probe_doc["verdict"]["can_build_features"]:
        print(json.dumps(probe_doc["verdict"], ensure_ascii=True, indent=2))
        print("\n*** 特征无法构建 ✗：先解决上面的缺失文件/列，再跑门槛。 ***")
        return 1

    units, man = load_supervision(root)
    expected = FROZEN_DIR_COVERAGE if args.expected_coverage == "frozen" else None
    out_dir = Path(args.out_dir) if args.out_dir else root / DEFAULT_OUT_SUBDIR

    modes = ((EXPERT_PROB_V1, EXPERT_PROB_MEAN4) if args.expert_prob == EXPERT_PROB_BOTH
             else (args.expert_prob,))

    all_docs = {}
    for mode in modes:
        doc = execute(root, args.tag, directions, args.seed, out_dir, mode, units, man,
                      probe_doc, expected_coverage=expected,
                      allow_scale_mismatch=args.allow_scale_mismatch,
                      overwrite=args.overwrite)
        all_docs[mode] = doc
        print("=" * 78)
        print(f"expert_prob={mode}: {doc['verdict']['label']} "
              f"({doc['verdict']['n_failed']}/{doc['verdict']['n_items']} 项未过)")
        for d, r in doc["directions"].items():
            print(f"  {d}: n={r['n_units']} dropped={r['n_dropped']} tau={r['tau']} "
                  f"C={r['C']} EC-AUROC={_fmt(r['expert_choice_auroc'])} "
                  f"table={r['table_sha256'][:12]}")
        print(f"  A-1..A-5 = {[doc['supervision'][k] for k in ('A1', 'A2', 'A3', 'A4', 'A5')]}"
              f"  R-7 clean = { {d: doc['r7'][d]['clean'] for d in doc['r7']} }")
        print(f"  reports: {doc['_written']['markdown']} | {doc['_written']['json']}")
        print("=" * 78)

    return 0 if all(d["verdict"]["label"] == "GO" for d in all_docs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
