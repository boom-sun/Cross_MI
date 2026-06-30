# -*- coding: utf-8 -*-
"""
Step14_PerSubject_Intensity_Discriminability_Accuracy_FIXED.py

修正版 Step14
============

用途
----
从 Step1 保存的 TOPO/TF 特征文件和分类结果 .mat 文件中生成结构清楚的长表：

1. per_subject_features_all.csv
   每行 = subject × stim × feature_scene × feature_type × roi × band/window

2. accuracy_long.csv
   每行 = subject × algorithm × protocol × accuracy_scene × feature_scene × train_stim × test_stim

3. linked_data_target_feature.csv
   每行 accuracy 合并 test_stim 对应的 target feature
   这是与你原 Step14 最接近的输出，但列名更清楚。

4. linked_data_pair_aware.csv
   在 target feature 基础上额外合并 train_stim 对应的 source feature，
   并生成 target-source 差异，专门用于 cross-paradigm 解释。

为什么修正
----------
原 Step14 的合并方向基本正确：accuracy 的 test_stim 应该匹配 test_stim 的特征。
但原输出容易造成两类误解：

1. scene 字段混用：
   分类结果中的 S1/S2/cross_task1/cross_task2 与特征来源 s1/s2 应该分开。
   本脚本统一使用：
       accuracy_scene = S1/S2/cross_task1/cross_task2
       feature_scene  = s1/s2

2. cross-paradigm 只合并 target feature：
   这适合解释 target 是否容易被分类，但不足以解释 source→target 迁移。
   本脚本额外生成 source feature 与 target-source distance。

依赖
----
numpy, pandas, scipy, hdf5storage, matplotlib

运行示例
--------
python Step14_PerSubject_Intensity_Discriminability_Accuracy_FIXED.py ^
  --topo-root "E:/.../脑地形图数据" ^
  --tf-root "E:/.../时频图数据" ^
  --result-root "E:/Code/Cross/Cross_MI" ^
  --save-dir "E:/.../Step14_FIXED" ^
  --dataset ssmvep_hybrid ^
  --dataclass 1

只生成 TOPO 主分析表：
python Step14_PerSubject_Intensity_Discriminability_Accuracy_FIXED.py ^
  --topo-root "E:/.../脑地形图数据" ^
  --result-root "E:/Code/Cross/Cross_MI" ^
  --save-dir "E:/.../Step14_FIXED" ^
  --skip-tf
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

try:
    import hdf5storage
except Exception:
    hdf5storage = None


# ============================================================
# default experiment config
# ============================================================

STIM_ID_MAP = {1: "ssvideo", 2: "video", 3: "ssmvep", 4: "cue"}
STIM_NAMES = ["ssvideo", "video", "ssmvep", "cue"]
FEATURE_SCENES = ["s1", "s2"]

WITHIN_SCENES = ["S1", "S2"]

CROSS_CONFIGS = [
    (1, 3, "cross_task1"),  # ssvideo -> ssmvep
    (3, 1, "cross_task1"),  # ssmvep -> ssvideo
    (1, 4, "cross_task1"),  # ssvideo -> cue
    (4, 1, "cross_task1"),  # cue -> ssvideo
    (2, 3, "cross_task1"),  # video -> ssmvep
    (3, 2, "cross_task1"),  # ssmvep -> video
    (1, 3, "cross_task2"),
    (3, 1, "cross_task2"),
    (1, 4, "cross_task2"),
    (4, 1, "cross_task2"),
    (2, 3, "cross_task2"),
    (3, 2, "cross_task2"),
]

CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4",
    "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6",
    "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1",
    "OZ", "O2",
]

ROI_CHANNELS = {
    "whole": CHANNELS,
    "motor_C3CzC4": ["C3", "CZ", "C4"],
    "motor_extended": [
        "FC3", "FC1", "FCZ", "FC2", "FC4",
        "C3", "C1", "CZ", "C2", "C4",
        "CP3", "CP1", "CPZ", "CP2", "CP4",
    ],
    "occipital": ["PO3", "POZ", "PO4", "O1", "OZ", "O2"],
}

DEFAULT_ALGORITHMS = ["CSPSVM", "FBCSP", "FGMDRM", "EEGNET"]

# 输出目录
SAVE_DIR = r'E:\Datasets\4_跨场景因素研究v2\画图结果\PerSubject_IntensityDisc_Accuracy'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'figures'), exist_ok=True)

# ============================================================
# small utils
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_subjects(s: str) -> List[int]:
    """
    Accept:
      "1-37"
      "1,2,3"
      "1-10,15,20-22"
    """
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def ch_indices(roi_name: str) -> List[int]:
    upper = [c.upper() for c in CHANNELS]
    return [upper.index(c.upper()) for c in ROI_CHANNELS[roi_name] if c.upper() in upper]


def safe_mean(x) -> float:
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def l2_normalized(vec) -> float:
    arr = np.asarray(vec, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.linalg.norm(arr) / np.sqrt(arr.size))


def cohen_dz(x1, x2) -> float:
    d = np.asarray(x1, dtype=float).ravel() - np.asarray(x2, dtype=float).ravel()
    d = d[np.isfinite(d)]
    if d.size < 2:
        return np.nan
    sd = np.std(d, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(np.mean(d) / sd)


def fisher_score(x1, x2) -> float:
    a = np.asarray(x1, dtype=float).ravel()
    b = np.asarray(x2, dtype=float).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    return float((np.mean(a) - np.mean(b)) ** 2 / (np.var(a, ddof=1) + np.var(b, ddof=1) + 1e-12))


def time_to_sec(times) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    if np.nanmax(np.abs(t)) > 20:
        return t / 1000.0
    return t


def scene_to_feature_scene(scene: str) -> str:
    s = str(scene).strip().lower()
    if s in {"s1", "cross_task1"}:
        return "s1"
    if s in {"s2", "cross_task2"}:
        return "s2"
    warnings.warn(f"Unknown scene={scene}; use lower-case value as feature_scene.")
    return s


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - x.mean()) / sd


# ============================================================
# feature loading
# ============================================================

def load_topo(topo_root: Path, stim: str, class_id: int, trial_clean_suffix: str):
    path = topo_root / f"TOPO_{stim}_class{class_id}{trial_clean_suffix}.mat"
    if not path.exists():
        warnings.warn(f"[TOPO] missing: {path}")
        return None
    return sio.loadmat(path)["topo"][0, 0]


def load_tf(tf_root: Path, stim: str, class_id: int, trial_clean_suffix: str):
    if hdf5storage is None:
        warnings.warn("hdf5storage is not installed; TF features are skipped.")
        return None, None, None

    p_tf = tf_root / f"TF_{stim}_class{class_id}{trial_clean_suffix}.mat"
    p_ax = tf_root / f"times+freqs_{stim}_class{class_id}.mat"

    if not p_tf.exists() or not p_ax.exists():
        warnings.warn(f"[TF] missing: {p_tf} or {p_ax}")
        return None, None, None

    tf_struct = sio.loadmat(p_tf)["tf"][0, 0]
    ax = hdf5storage.loadmat(str(p_ax))
    times = np.squeeze(ax["times"]).astype(float)
    freqs = np.squeeze(ax["freqs"]).astype(float)
    return tf_struct, times, freqs


def compute_topo_metrics(
    topo_root: Path,
    stim: str,
    feature_scene: str,
    band_key: str,
    roi_name: str,
    trial_clean_suffix: str,
    c1: int = 1,
    c2: int = 2,
) -> Optional[pd.DataFrame]:

    topo1 = load_topo(topo_root, stim, c1, trial_clean_suffix)
    topo2 = load_topo(topo_root, stim, c2, trial_clean_suffix)
    if topo1 is None or topo2 is None:
        return None

    try:
        x1_all = np.asarray(topo1[feature_scene][0, 0][band_key], dtype=float)
        x2_all = np.asarray(topo2[feature_scene][0, 0][band_key], dtype=float)
    except Exception as e:
        warnings.warn(f"[TOPO] cannot extract stim={stim}, scene={feature_scene}, key={band_key}: {e}")
        return None

    idx = ch_indices(roi_name)
    if not idx:
        return None

    x1 = x1_all[idx, :]
    x2 = x2_all[idx, :]
    n_subjects = x1.shape[1]

    rows = []
    for sub_idx in range(n_subjects):
        v1 = x1[:, sub_idx]
        v2 = x2[:, sub_idx]
        rows.append({
            "subject": sub_idx + 1,
            "stim": stim,
            "feature_scene": feature_scene,
            "feature_type": "topo",
            "roi": roi_name,
            "band_key": band_key,
            "n_features": len(v1),
            "intensity": safe_mean((np.abs(v1) + np.abs(v2)) / 2.0),
            "discriminability": l2_normalized(v1 - v2),
            "cohens_dz": cohen_dz(v1, v2),
            "fisher_score": fisher_score(v1, v2),
        })

    return pd.DataFrame(rows)


def compute_tf_metrics(
    tf_root: Path,
    stim: str,
    feature_scene: str,
    freq_band: Tuple[float, float],
    time_win: Tuple[float, float],
    roi_name: str,
    trial_clean_suffix: str,
    c1: int = 1,
    c2: int = 2,
) -> Optional[pd.DataFrame]:

    tf1, times, freqs = load_tf(tf_root, stim, c1, trial_clean_suffix)
    tf2, _, _ = load_tf(tf_root, stim, c2, trial_clean_suffix)
    if tf1 is None or tf2 is None:
        return None

    try:
        a1 = np.asarray(tf1[feature_scene], dtype=float)
        a2 = np.asarray(tf2[feature_scene], dtype=float)
    except Exception as e:
        warnings.warn(f"[TF] cannot extract stim={stim}, scene={feature_scene}: {e}")
        return None

    times_sec = time_to_sec(times)
    f_idx = np.where((freqs >= freq_band[0]) & (freqs <= freq_band[1]))[0]
    t_idx = np.where((times_sec >= time_win[0]) & (times_sec <= time_win[1]))[0]
    c_idx = ch_indices(roi_name)

    if len(f_idx) == 0 or len(t_idx) == 0 or len(c_idx) == 0:
        warnings.warn(f"[TF] empty crop stim={stim}, scene={feature_scene}, roi={roi_name}")
        return None

    a1c = a1[:, f_idx][:, :, t_idx][:, :, :, c_idx]
    a2c = a2[:, f_idx][:, :, t_idx][:, :, :, c_idx]

    rows = []
    for sub_idx in range(a1c.shape[0]):
        v1 = a1c[sub_idx].ravel()
        v2 = a2c[sub_idx].ravel()
        rows.append({
            "subject": sub_idx + 1,
            "stim": stim,
            "feature_scene": feature_scene,
            "feature_type": "tf",
            "roi": roi_name,
            "band_key": f"freq{freq_band}_time{time_win}",
            "n_features": len(v1),
            "intensity": safe_mean((np.abs(v1) + np.abs(v2)) / 2.0),
            "discriminability": l2_normalized(v1 - v2),
            "cohens_dz": cohen_dz(v1, v2),
            "fisher_score": fisher_score(v1, v2),
        })

    return pd.DataFrame(rows)


def build_feature_table(args) -> pd.DataFrame:
    parts = []
    topo_root = Path(args.topo_root) if args.topo_root else None
    tf_root = Path(args.tf_root) if args.tf_root else None
    suffix = "_trialClean" if args.trial_clean else ""

    rois = parse_csv_list(args.rois)

    for stim in STIM_NAMES:
        for feature_scene in FEATURE_SCENES:
            if topo_root is not None and not args.skip_topo:
                for roi in rois:
                    if roi not in ROI_CHANNELS:
                        warnings.warn(f"Unknown ROI={roi}; skipped.")
                        continue
                    df = compute_topo_metrics(
                        topo_root=topo_root,
                        stim=stim,
                        feature_scene=feature_scene,
                        band_key=args.topo_band_key,
                        roi_name=roi,
                        trial_clean_suffix=suffix,
                    )
                    if df is not None:
                        parts.append(df)

            if tf_root is not None and not args.skip_tf:
                # TF can be large. By default only compute primary ROI unless --tf-all-rois is used.
                tf_rois = rois if args.tf_all_rois else [args.primary_roi]
                for roi in tf_rois:
                    if roi not in ROI_CHANNELS:
                        continue
                    df = compute_tf_metrics(
                        tf_root=tf_root,
                        stim=stim,
                        feature_scene=feature_scene,
                        freq_band=(args.tf_freq_low, args.tf_freq_high),
                        time_win=(args.tf_time_start, args.tf_time_end),
                        roi_name=roi,
                        trial_clean_suffix=suffix,
                    )
                    if df is not None:
                        parts.append(df)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["subject"] = out["subject"].astype(int)
    return out


# ============================================================
# accuracy loading
# ============================================================

def mat_to_namespace(path: Path):
    d = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)

    def convert(obj):
        # scipy private type warning is harmless here; keep compatibility with MATLAB structs.
        if hasattr(obj, "_fieldnames"):
            ns = SimpleNamespace()
            for fn in obj._fieldnames:
                setattr(ns, fn, convert(getattr(obj, fn)))
            return ns
        return obj

    return {k: convert(v) for k, v in d.items() if not k.startswith("__")}


def extract_scalar_acc(acc_struct, algorithm: str) -> float:
    alg = str(algorithm).upper()
    if acc_struct is None or not hasattr(acc_struct, alg):
        return np.nan
    arr = np.asarray(getattr(acc_struct, alg), dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def result_path_within(result_root: Path, dataset: str, dataclass: int, subject: int, scene: str, stim: str) -> Path:
    return result_root / f"result_{dataset}" / f"{dataclass}result_sub{subject}_0_{scene}_{stim}.mat"


def result_path_cross(
    result_root: Path,
    dataset: str,
    dataclass: int,
    subject: int,
    scene: str,
    src_stim: str,
    tgt_stim: str,
) -> Path:
    return result_root / f"result_{dataset}" / f"{dataclass}result_sub{subject}_1_{scene}_{src_stim}_{tgt_stim}.mat"


def load_accuracy_from_path(path: Path, algorithms: List[str]) -> Dict[str, float]:
    if not path.exists():
        return {alg: np.nan for alg in algorithms}

    try:
        d = mat_to_namespace(path)
        acc_struct = d.get("acc", None)
        return {alg: extract_scalar_acc(acc_struct, alg) for alg in algorithms}
    except Exception as e:
        warnings.warn(f"[ACC] cannot load {path}: {e}")
        return {alg: np.nan for alg in algorithms}


def build_accuracy_table(args) -> pd.DataFrame:
    result_root = Path(args.result_root)
    algorithms = parse_csv_list(args.algorithms)
    subjects = parse_subjects(args.subjects)

    rows = []

    for accuracy_scene in WITHIN_SCENES:
        feature_scene = scene_to_feature_scene(accuracy_scene)
        for stim in STIM_NAMES:
            for subject in subjects:
                path = result_path_within(result_root, args.dataset, args.dataclass, subject, accuracy_scene, stim)
                accs = load_accuracy_from_path(path, algorithms)
                for alg, acc in accs.items():
                    rows.append({
                        "subject": subject,
                        "algorithm": alg.upper(),
                        "protocol": "within_paradigm",
                        "accuracy_scene": accuracy_scene,
                        "feature_scene": feature_scene,
                        "train_stim": stim,
                        "test_stim": stim,
                        "task_pair": f"{stim}→{stim}",
                        "accuracy": acc,
                        "result_file": str(path),
                    })

    for src_id, tgt_id, accuracy_scene in CROSS_CONFIGS:
        src_stim = STIM_ID_MAP[src_id]
        tgt_stim = STIM_ID_MAP[tgt_id]
        feature_scene = scene_to_feature_scene(accuracy_scene)
        for subject in subjects:
            path = result_path_cross(
                result_root, args.dataset, args.dataclass,
                subject, accuracy_scene, src_stim, tgt_stim
            )
            accs = load_accuracy_from_path(path, algorithms)
            for alg, acc in accs.items():
                rows.append({
                    "subject": subject,
                    "algorithm": alg.upper(),
                    "protocol": "cross_paradigm",
                    "accuracy_scene": accuracy_scene,
                    "feature_scene": feature_scene,
                    "train_stim": src_stim,
                    "test_stim": tgt_stim,
                    "task_pair": f"{src_stim}→{tgt_stim}",
                    "accuracy": acc,
                    "result_file": str(path),
                })

    out = pd.DataFrame(rows)
    out["accuracy"] = pd.to_numeric(out["accuracy"], errors="coerce")
    return out


# ============================================================
# linking
# ============================================================

def select_primary_features(feat: pd.DataFrame, primary_feature: str, primary_roi: str) -> pd.DataFrame:
    if feat.empty:
        return pd.DataFrame()

    sel = feat[
        (feat["feature_type"].astype(str).str.lower() == primary_feature.lower())
        & (feat["roi"].astype(str) == primary_roi)
    ].copy()

    if sel.empty:
        warnings.warn(f"No primary feature rows for feature_type={primary_feature}, roi={primary_roi}")
        return sel

    keep = [
        "subject", "stim", "feature_scene", "feature_type", "roi", "band_key", "n_features",
        "intensity", "discriminability", "cohens_dz", "fisher_score",
    ]
    return sel[keep].drop_duplicates()


def build_linked_tables(feat: pd.DataFrame, acc: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    primary = select_primary_features(feat, args.primary_feature, args.primary_roi)

    if primary.empty or acc.empty:
        return pd.DataFrame(), pd.DataFrame()

    target = primary.rename(columns={
        "stim": "test_stim",
        "intensity": "target_intensity",
        "discriminability": "target_discriminability",
        "cohens_dz": "target_cohens_dz",
        "fisher_score": "target_fisher_score",
        "feature_type": "target_feature_type",
        "roi": "target_roi",
        "band_key": "target_band_key",
        "n_features": "target_n_features",
    })

    linked_target = acc.merge(
        target,
        on=["subject", "feature_scene", "test_stim"],
        how="left",
        validate="many_to_one",
    )

    source = primary.rename(columns={
        "stim": "train_stim",
        "intensity": "source_intensity",
        "discriminability": "source_discriminability",
        "cohens_dz": "source_cohens_dz",
        "fisher_score": "source_fisher_score",
        "feature_type": "source_feature_type",
        "roi": "source_roi",
        "band_key": "source_band_key",
        "n_features": "source_n_features",
    })

    linked_pair = linked_target.merge(
        source,
        on=["subject", "feature_scene", "train_stim"],
        how="left",
        validate="many_to_one",
    )

    # keep old names for backward compatibility
    linked_pair["intensity"] = linked_pair["target_intensity"]
    linked_pair["discriminability"] = linked_pair["target_discriminability"]
    if "target_cohens_dz" in linked_pair:
        linked_pair["cohens_dz"] = linked_pair["target_cohens_dz"]
    if "target_fisher_score" in linked_pair:
        linked_pair["fisher_score"] = linked_pair["target_fisher_score"]

    linked_pair["delta_intensity_target_minus_source"] = (
        linked_pair["target_intensity"] - linked_pair["source_intensity"]
    )
    linked_pair["delta_discriminability_target_minus_source"] = (
        linked_pair["target_discriminability"] - linked_pair["source_discriminability"]
    )
    linked_pair["abs_delta_intensity"] = linked_pair["delta_intensity_target_minus_source"].abs()
    linked_pair["abs_delta_discriminability"] = linked_pair["delta_discriminability_target_minus_source"].abs()

    for c in [
        "accuracy", "target_intensity", "target_discriminability",
        "source_intensity", "source_discriminability",
        "abs_delta_intensity", "abs_delta_discriminability",
    ]:
        if c in linked_pair.columns:
            linked_pair[c + "_z"] = zscore(linked_pair[c])

    return linked_target, linked_pair


def audit_outputs(feat: pd.DataFrame, acc: pd.DataFrame, linked_pair: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(item, value):
        rows.append({"item": item, "value": value})

    add("n_feature_rows_all", len(feat))
    add("n_accuracy_rows", len(acc))
    add("n_linked_rows_pair_aware", len(linked_pair))

    if not feat.empty:
        add("feature_subjects", feat["subject"].nunique())
        add("feature_stims", ", ".join(sorted(feat["stim"].astype(str).unique())))
        add("feature_scenes", ", ".join(sorted(feat["feature_scene"].astype(str).unique())))
        add("feature_types", ", ".join(sorted(feat["feature_type"].astype(str).unique())))
        add("rois", ", ".join(sorted(feat["roi"].astype(str).unique())))

    if not acc.empty:
        add("accuracy_subjects", acc["subject"].nunique())
        add("algorithms", ", ".join(sorted(acc["algorithm"].astype(str).unique())))
        add("protocols", ", ".join(sorted(acc["protocol"].astype(str).unique())))
        add("task_pairs", ", ".join(sorted(acc["task_pair"].astype(str).unique())))
        add("n_missing_accuracy", int(acc["accuracy"].isna().sum()))

    if not linked_pair.empty:
        add("n_missing_target_intensity", int(linked_pair["target_intensity"].isna().sum()))
        add("n_missing_target_discriminability", int(linked_pair["target_discriminability"].isna().sum()))
        add("n_missing_source_intensity", int(linked_pair["source_intensity"].isna().sum()))
        add("n_missing_source_discriminability", int(linked_pair["source_discriminability"].isna().sum()))

        feature_cell_dup = linked_pair.groupby(["subject", "feature_scene", "test_stim"]).size()
        add("mean_rows_per_target_feature_cell", float(feature_cell_dup.mean()))
        add("max_rows_per_target_feature_cell", int(feature_cell_dup.max()))

    return pd.DataFrame(rows)


# ============================================================
# main
# ============================================================

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--result-root",default=r"E:\Code\Cross\Cross_MI")
    p.add_argument("--save-dir", default=r"E:\Datasets\4_跨场景因素研究v2\画图结果\PerSubject_IntensityDisc_Accuracy")
    p.add_argument("--topo-root", default=r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\脑地形图数据')
    p.add_argument("--tf-root", default=r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\时频图数据')

    p.add_argument("--dataset", default="ssmvep_hybrid")
    p.add_argument("--dataclass", type=int, default=1)
    p.add_argument("--subjects", default="1-37")
    p.add_argument("--algorithms", default="CSPSVM,FBCSP,FGMDRM,EEGNET")

    p.add_argument("--trial-clean", action="store_true", default=True)
    p.add_argument("--no-trial-clean", dest="trial_clean", action="store_false")

    p.add_argument("--skip-topo", action="store_true")
    p.add_argument("--skip-tf", action="store_true")

    p.add_argument("--topo-band-key", default="freq(8, 13)time(0, 4)")
    p.add_argument("--tf-freq-low", type=float, default=8)
    p.add_argument("--tf-freq-high", type=float, default=30)
    p.add_argument("--tf-time-start", type=float, default=0)
    p.add_argument("--tf-time-end", type=float, default=4)

    p.add_argument("--rois", default="whole,motor_C3CzC4,motor_extended,occipital")
    p.add_argument("--primary-roi", default="motor_extended")
    p.add_argument("--primary-feature", default="topo", choices=["topo", "tf"])
    p.add_argument("--tf-all-rois", action="store_true")

    return p


def main():
    args = make_parser().parse_args()
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    if args.skip_topo and args.skip_tf:
        raise ValueError("Both --skip-topo and --skip-tf are set; no features will be computed.")

    print("[Step14 FIXED] Building per-subject feature table...")
    feat = build_feature_table(args)
    feat.to_csv(save_dir / "per_subject_features_all.csv", index=False, encoding="utf-8-sig")

    print("[Step14 FIXED] Building accuracy table...")
    acc = build_accuracy_table(args)
    acc.to_csv(save_dir / "accuracy_long.csv", index=False, encoding="utf-8-sig")

    print("[Step14 FIXED] Linking target and pair-aware features...")
    linked_target, linked_pair = build_linked_tables(feat, acc, args)
    linked_target.to_csv(save_dir / "linked_data_target_feature.csv", index=False, encoding="utf-8-sig")
    linked_pair.to_csv(save_dir / "linked_data_pair_aware.csv", index=False, encoding="utf-8-sig")

    audit = audit_outputs(feat, acc, linked_pair)
    audit.to_csv(save_dir / "00_step14_fixed_audit.csv", index=False, encoding="utf-8-sig")

    # Also save a backward-compatible filename for later scripts.
    linked_pair.to_csv(save_dir / "linked_data.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Step14 FIXED finished.")
    print(f"Feature rows: {len(feat)}")
    print(f"Accuracy rows: {len(acc)}")
    print(f"Linked pair-aware rows: {len(linked_pair)}")
    print(f"Output dir: {save_dir.resolve()}")
    print("Main output for Step15: linked_data_pair_aware.csv or linked_data.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
