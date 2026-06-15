"""
Step14_PerSubject_Intensity_Discriminability_Accuracy.py
=========================================================
目的
----
在 Step12 group-level 分析的基础上，下沉到 **被试级别**（per-subject），
对每个 (被试 × 范式/stim × 场景) 三元组计算：

  Intensity        = mean((|class1_feature| + |class2_feature|) / 2)
  Discriminability = ||mean_class1_feature - mean_class2_feature||₂ / √n_features
                     （也输出 Cohen's dz 和 Fisher score 供参考）

再把两个特征量与分类准确率合并，画散点：
  X = Intensity  ,  Y = Discriminability
  X = Intensity  ,  Y = Accuracy  (within-paradigm  &  cross-paradigm 各一列)
  X = Discriminability  ,  Y = Accuracy

输入
----
1) Step1_TOPO_TF_save.py 保存的 TOPO_*.mat / TF_*.mat
   TOPO 结构: mat['topo'][0,0][scene][0,0]['freq(l, h)time(t0, t1)']  → [n_ch, n_sub]
   TF   结构: mat['tf'][0,0][scene]                                    → [n_sub, n_freq, n_time, n_ch]

2) 分类流水线（main.py）保存的 result .mat 文件
   a) Within-paradigm  (sence='S1'/'S2')：
      {RESULT_ROOT}/result_{dataset}/{dataclass}result_sub{subject}_{sence}_{stim}.mat
   b) Cross-paradigm   (sence='cross_task1'/'cross_task2')：
      {RESULT_ROOT}/result_{dataset}/{dataclass}result_sub{subject}_{sence}_{src_stim}_{tgt_stim}.mat

输出
----
  per_subject_features.csv          — 每行 = 1 subject × stim × scene × band × roi
  accuracy_long.csv                 — 每行 = 1 subject × algorithm × protocol × stim
  linked_data.csv                   — 两表合并
  figures/*.png                     — 散点图

准确率获取说明
--------------
  • Within-paradigm: 先用 feature_ssmvep_hybrid.yaml (sence=S1/S2, n_class=[0, stim_id])
    运行 main.py 生成各 stim 的 S1/S2 准确率文件，再由本脚本读取。
  • Cross-paradigm : 用 ssmvep_hybrid.yaml (sence=cross_task1/2, n_class=[1, src, tgt])
    运行 main.py 生成跨范式准确率文件，再由本脚本读取。
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import hdf5storage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ======================================================================
# CONFIG — 按实际路径和实验设置修改
# ======================================================================

# 数据集
DATASET        = 'ssmvep_hybrid'
DATACLASS      = 1                             # 与 main.py 保持一致
N_SUBJECTS     = list(range(1, 37 + 1))        # 1-37

# 范式（stim）映射
STIM_ID_MAP    = {1: 'ssvideo', 2: 'video', 3: 'ssmvep', 4: 'cue'}
STIM_NAMES     = ['ssvideo', 'video', 'ssmvep', 'cue']   # ssmvep_hybrid 四个范式
SCENES         = ['s1', 's2']                             # Step1 保存时用小写

# 特征文件路径（Step1 输出）
TOPO_ROOT = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\脑地形图数据'
TF_ROOT   = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\时频图数据'
IS_TRIAL_CLEAN = True

# 分类结果根目录（与 sun_data_saver.py 中的 save_path 一致）
RESULT_ROOT = r'E:\Code\Cross\Cross_MI'

# 算法列表（与 main.py config 保持一致）
# ALGORITHMS 用于读取所有结果；FOCUS_ALGORITHMS 用于重点统计和绘图。
ALGORITHMS = ['CSPSVM', 'FBCSP', 'FGMDRM', 'EEGNET']
FOCUS_ALGORITHMS = ['FBCSP', 'FGMDRM', 'EEGNET']

# Within-paradigm：同时读取 S1 和 S2，后续按场景匹配特征。
# n_class=[0, stim_id] 时文件名含单个 stim
WITHIN_SENCES = ['S1', 'S2']

# Cross-paradigm 组合：[(source_stim_id, target_stim_id, sence_str), ...]
# 与运行 main.py 时的 n_class=[1, src, tgt] + sence 对应
CROSS_CONFIGS = [
    (1, 3, 'cross_task1'),   # ssvideo → ssmvep,  S1 内跨范式
    (3, 1, 'cross_task1'),   # ssmvep  → ssvideo
    (1, 4, 'cross_task1'),   # ssvideo → cue
    (4, 1, 'cross_task1'),   # cue     → ssvideo
    (2, 3, 'cross_task1'),   # video   → ssmvep
    (3, 2, 'cross_task1'),   # ssmvep  → video
    (1, 3, 'cross_task2'),   # ssvideo → ssmvep,  S2 内跨范式
    (3, 1, 'cross_task2'),   # ssmvep  → ssvideo
    (1, 4, 'cross_task2'),   # ssvideo → cue
    (4, 1, 'cross_task2'),   # cue     → ssvideo
    (2, 3, 'cross_task2'),   # video   → ssmvep
    (3, 2, 'cross_task2'),   # ssmvep  → video
]

# 特征计算参数
TOPO_BAND_KEY     = 'freq(8, 13)time(0, 4)'   # 与 Step1 保存时使用的 key 保持一致
TF_FREQ_BAND      = (8, 30)
TF_TIME_WIN       = (0, 4)    # seconds

CHANNELS = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ',
    'PO4','PO6','PO8','O1','OZ','O2'
]
ROI_CHANNELS = {
    'whole':           CHANNELS,
    'motor_C3CzC4':    ['C3', 'CZ', 'C4'],
    'motor_extended':  ['FC3','FC1','FCZ','FC2','FC4','C3','C1','CZ','C2','C4',
                        'CP3','CP1','CPZ','CP2','CP4'],
    'occipital':       ['PO3','POZ','PO4','O1','OZ','O2'],
}
PRIMARY_ROI = 'motor_extended'    # 主散点图使用的 ROI
PRIMARY_FEATURE = 'topo'          # 'topo' 或 'tf'

# 输出目录
SAVE_DIR = r'E:\Datasets\4_跨场景因素研究v2\画图结果\PerSubject_IntensityDisc_Accuracy'

# ======================================================================

_trialClean = '_trialClean' if IS_TRIAL_CLEAN else ''
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'figures'), exist_ok=True)


# ======================================================================
# 工具函数
# ======================================================================

def _ch_indices(roi_name: str) -> List[int]:
    ch_upper = [c.upper() for c in CHANNELS]
    return [ch_upper.index(c.upper()) for c in ROI_CHANNELS[roi_name]
            if c.upper() in ch_upper]


def _safe(x):
    """确保返回 float，NaN-safe。"""
    v = float(np.nanmean(np.asarray(x, dtype=float).ravel()))
    return v if np.isfinite(v) else np.nan


def _l2_normalized(vec):
    vec = np.asarray(vec, dtype=float).ravel()
    n = vec.size
    return float(np.linalg.norm(vec) / np.sqrt(n + 1e-12))


def _cohen_dz(x1: np.ndarray, x2: np.ndarray) -> float:
    diff = x1.ravel() - x2.ravel()
    sd   = np.nanstd(diff, ddof=1)
    return float(np.nanmean(diff) / (sd + 1e-12))


def _fisher_score(x1: np.ndarray, x2: np.ndarray) -> float:
    m1, m2 = np.nanmean(x1), np.nanmean(x2)
    v1, v2 = np.nanvar(x1, ddof=1), np.nanvar(x2, ddof=1)
    return float((m1 - m2) ** 2 / (v1 + v2 + 1e-12))


def _time_to_sec(times):
    t = np.asarray(times, dtype=float)
    return t / 1000.0 if np.nanmax(np.abs(t)) > 20 else t


def _sence_to_feature_scene(sence: str) -> str:
    """
    将分类结果中的 sence 字段映射到 Step1 特征文件中的 scene。
    S1 / cross_task1 使用 s1 特征；S2 / cross_task2 使用 s2 特征。
    """
    s = str(sence).lower()
    if s in {'s1', 'cross_task1'}:
        return 's1'
    if s in {'s2', 'cross_task2'}:
        return 's2'
    warnings.warn(f'[SCENE_MAP] 未知 sence={sence}，将按小写值作为 feature_scene')
    return s


def _selected_algorithms(df: pd.DataFrame, algorithms: Optional[List[str]] = None) -> List[str]:
    """返回当前数据中实际存在、且需要重点展示的算法。"""
    algorithms = algorithms or FOCUS_ALGORITHMS
    if 'algorithm' not in df.columns:
        return []
    present = set(df['algorithm'].dropna().astype(str).unique())
    return [alg for alg in algorithms if alg in present]


# ======================================================================
# 1. 特征文件加载
# ======================================================================

def load_topo(stim: str, class_id: int):
    """返回 sio.loadmat(...)['topo'][0,0]，即 MATLAB struct。"""
    p = os.path.join(TOPO_ROOT, f'TOPO_{stim}_class{class_id}{_trialClean}.mat')
    if not os.path.exists(p):
        warnings.warn(f'[TOPO] 文件不存在: {p}')
        return None
    return sio.loadmat(p)['topo'][0, 0]


def load_tf(stim: str, class_id: int):
    """返回 (tf_struct, times, freqs)。"""
    p_tf = os.path.join(TF_ROOT, f'TF_{stim}_class{class_id}{_trialClean}.mat')
    p_ax = os.path.join(TF_ROOT, f'times+freqs_{stim}_class{class_id}.mat')
    for p in (p_tf, p_ax):
        if not os.path.exists(p):
            warnings.warn(f'[TF] 文件不存在: {p}')
            return None, None, None
    tf_struct = sio.loadmat(p_tf)['tf'][0, 0]
    ax = hdf5storage.loadmat(p_ax)
    times = np.squeeze(ax['times']).astype(float)
    freqs = np.squeeze(ax['freqs']).astype(float)
    return tf_struct, times, freqs


# ======================================================================
# 2. Per-subject 特征计算
# ======================================================================

def per_subject_topo_metrics(
        stim: str, scene: str, band_key: str, roi_name: str,
        c1: int = 1, c2: int = 2) -> Optional[pd.DataFrame]:
    """
    从 TOPO 文件提取每个被试的 Intensity 和 Discriminability。
    返回列: subject, stim, scene, roi, band_key,
            intensity, discriminability, cohens_dz, fisher_score
    """
    topo1 = load_topo(stim, c1)
    topo2 = load_topo(stim, c2)
    if topo1 is None or topo2 is None:
        return None

    scene_key = scene.lower()
    try:
        # [n_channels, n_subjects]
        x1_all = np.asarray(topo1[scene_key][0, 0][band_key], dtype=float)
        x2_all = np.asarray(topo2[scene_key][0, 0][band_key], dtype=float)
    except (KeyError, ValueError) as e:
        warnings.warn(f'[TOPO] 无法提取 scene={scene_key}, key={band_key}: {e}')
        return None

    ch_idx = _ch_indices(roi_name)
    if not ch_idx:
        warnings.warn(f'[TOPO] ROI {roi_name} 没有有效通道索引')
        return None

    x1_roi = x1_all[ch_idx, :]   # [n_roi_ch, n_subjects]
    x2_roi = x2_all[ch_idx, :]

    n_sub = x1_roi.shape[1]
    rows = []
    for s_idx in range(n_sub):
        v1 = x1_roi[:, s_idx]   # [n_roi_ch]
        v2 = x2_roi[:, s_idx]
        intensity        = _safe((np.abs(v1) + np.abs(v2)) / 2.0)
        discriminability = _l2_normalized(v1 - v2)
        cohens_dz        = _cohen_dz(v1, v2)
        fisher_sc        = _fisher_score(v1, v2)
        rows.append({
            'subject':         s_idx + 1,
            'stim':            stim,
            'scene':           scene_key,
            'feature_type':    'topo',
            'roi':             roi_name,
            'band_key':        band_key,
            'intensity':       intensity,
            'discriminability': discriminability,
            'cohens_dz':       cohens_dz,
            'fisher_score':    fisher_sc,
        })
    return pd.DataFrame(rows)


def per_subject_tf_metrics(
        stim: str, scene: str,
        freq_band: Tuple[float, float], time_win: Tuple[float, float],
        roi_name: str, c1: int = 1, c2: int = 2) -> Optional[pd.DataFrame]:
    """
    从 TF 文件提取每个被试的 Intensity 和 Discriminability。
    x_sub shape: [n_freqs_crop, n_times_crop, n_ch_roi] → flattened vector
    """
    tf1, times, freqs = load_tf(stim, c1)
    tf2, _, _         = load_tf(stim, c2)
    if tf1 is None or tf2 is None:
        return None

    scene_key = scene.lower()
    try:
        a1 = np.asarray(tf1[scene_key], dtype=float)   # [n_sub, n_freq, n_time, n_ch]
        a2 = np.asarray(tf2[scene_key], dtype=float)
    except (KeyError, ValueError) as e:
        warnings.warn(f'[TF] 无法提取 scene={scene_key}: {e}')
        return None

    times_sec = _time_to_sec(times)
    f_idx = np.where((freqs >= freq_band[0]) & (freqs <= freq_band[1]))[0]
    t_idx = np.where((times_sec >= time_win[0]) & (times_sec <= time_win[1]))[0]
    ch_idx = _ch_indices(roi_name)
    if not all([len(f_idx), len(t_idx), len(ch_idx)]):
        warnings.warn(f'[TF] 空裁剪: band={freq_band}, time={time_win}, roi={roi_name}')
        return None

    # 裁剪 → [n_sub, n_freq_crop, n_time_crop, n_ch_roi]
    a1c = a1[:, f_idx][:, :, t_idx][:, :, :, ch_idx]
    a2c = a2[:, f_idx][:, :, t_idx][:, :, :, ch_idx]

    n_sub = a1c.shape[0]
    rows = []
    for s_idx in range(n_sub):
        v1 = a1c[s_idx].ravel()
        v2 = a2c[s_idx].ravel()
        intensity        = _safe((np.abs(v1) + np.abs(v2)) / 2.0)
        discriminability = _l2_normalized(v1 - v2)
        cohens_dz        = _cohen_dz(v1, v2)
        fisher_sc        = _fisher_score(v1, v2)
        rows.append({
            'subject':         s_idx + 1,
            'stim':            stim,
            'scene':           scene_key,
            'feature_type':    'tf',
            'roi':             roi_name,
            'band_key':        f'{freq_band}_{time_win}',
            'intensity':       intensity,
            'discriminability': discriminability,
            'cohens_dz':       cohens_dz,
            'fisher_score':    fisher_sc,
        })
    return pd.DataFrame(rows)


# ======================================================================
# 3. 准确率读取
# ======================================================================

def _mat_namespace(path: str):
    """使用与 sun_data_saver.py 相同的方式加载 mat 文件。"""
    d = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    def _to_ns(obj):
        if isinstance(obj, sio.matlab.mio5_params.mat_struct):
            ns = SimpleNamespace()
            for fn in obj._fieldnames:
                setattr(ns, fn, _to_ns(getattr(obj, fn)))
            ns._fieldnames = obj._fieldnames
            return ns
        return obj
    return {k: _to_ns(v) for k, v in d.items() if not k.startswith('__')}


def _extract_scalar_acc(acc_struct, algorithm: str) -> float:
    """
    从加载的 acc struct 中提取算法的标量准确率（取所有值的平均）。
    兼容 within-paradigm [n_train, n_fold] 矩阵和 cross-paradigm 标量/列表。
    """
    alg_up = algorithm.upper()
    if not hasattr(acc_struct, alg_up):
        return np.nan
    val = getattr(acc_struct, alg_up)
    arr = np.asarray(val, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else np.nan


def _result_path_within(dataset: str, dataclass: int, subject: int,
                         sence: str, stim: str) -> str:
    stim_id = STIM_NAMES.index(stim) + 1   # 1-indexed
    return os.path.join(
        RESULT_ROOT,
        f'result_{dataset}',
        f'{dataclass}result_sub{subject}_0_{sence}_{stim}.mat'
    )


def _result_path_cross(dataset: str, dataclass: int, subject: int,
                        sence: str, src_stim: str, tgt_stim: str) -> str:
    return os.path.join(
        RESULT_ROOT,
        f'result_{dataset}',
        f'{dataclass}result_sub{subject}_1_{sence}_{src_stim}_{tgt_stim}.mat'
    )


def load_within_accuracy(subject: int, sence: str, stim: str,
                          algorithms: List[str]) -> Dict[str, float]:
    path = _result_path_within(DATASET, DATACLASS, subject, sence, stim)
    if not os.path.exists(path):
        return {alg: np.nan for alg in algorithms}
    try:
        d = _mat_namespace(path)
        acc_struct = d.get('acc', None)
        if acc_struct is None:
            return {alg: np.nan for alg in algorithms}
        return {alg: _extract_scalar_acc(acc_struct, alg) for alg in algorithms}
    except Exception as e:
        warnings.warn(f'[ACC_WITHIN] subject={subject}, stim={stim}: {e}')
        return {alg: np.nan for alg in algorithms}


def load_cross_accuracy(subject: int, sence: str, src_stim: str, tgt_stim: str,
                         algorithms: List[str]) -> Dict[str, float]:
    path = _result_path_cross(DATASET, DATACLASS, subject, sence, src_stim, tgt_stim)
    if not os.path.exists(path):
        return {alg: np.nan for alg in algorithms}
    try:
        d = _mat_namespace(path)
        acc_struct = d.get('acc', None)
        if acc_struct is None:
            return {alg: np.nan for alg in algorithms}
        return {alg: _extract_scalar_acc(acc_struct, alg) for alg in algorithms}
    except Exception as e:
        warnings.warn(f'[ACC_CROSS] subject={subject}, {src_stim}→{tgt_stim}: {e}')
        return {alg: np.nan for alg in algorithms}


# ======================================================================
# 4. 构建完整数据表
# ======================================================================

def build_feature_table() -> pd.DataFrame:
    """计算所有 (stim × scene × roi) 的 per-subject 特征指标。"""
    parts = []
    for stim in STIM_NAMES:
        for scene in SCENES:
            # ---- TOPO ----
            for roi in ROI_CHANNELS.keys():
                df = per_subject_topo_metrics(
                    stim, scene, TOPO_BAND_KEY, roi)
                if df is not None:
                    parts.append(df)
            # ---- TF (仅 PRIMARY_ROI 以控制计算量) ----
            df = per_subject_tf_metrics(
                stim, scene, TF_FREQ_BAND, TF_TIME_WIN, PRIMARY_ROI)
            if df is not None:
                parts.append(df)

    if not parts:
        print('[WARN] 没有找到任何特征文件，特征表为空。')
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_accuracy_table() -> pd.DataFrame:
    """为每个被试读取 within-paradigm 和 cross-paradigm 准确率。"""
    rows = []

    # Within-paradigm: 同时读取 S1 和 S2
    for sence in WITHIN_SENCES:
        feature_scene = _sence_to_feature_scene(sence)
        for stim in STIM_NAMES:
            for subject in N_SUBJECTS:
                accs = load_within_accuracy(subject, sence, stim, ALGORITHMS)
                for alg, acc in accs.items():
                    rows.append({
                        'subject':       subject,
                        'algorithm':     alg,
                        'protocol':      'within_paradigm',
                        'sence':         sence,
                        'feature_scene': feature_scene,
                        'test_stim':     stim,
                        'train_stim':    stim,
                        'accuracy':      acc,
                    })

    # Cross-paradigm: cross_task1 匹配 s1，cross_task2 匹配 s2
    for (src_id, tgt_id, sence) in CROSS_CONFIGS:
        src_stim = STIM_ID_MAP[src_id]
        tgt_stim = STIM_ID_MAP[tgt_id]
        feature_scene = _sence_to_feature_scene(sence)
        for subject in N_SUBJECTS:
            accs = load_cross_accuracy(
                subject, sence, src_stim, tgt_stim, ALGORITHMS)
            for alg, acc in accs.items():
                rows.append({
                    'subject':       subject,
                    'algorithm':     alg,
                    'protocol':      'cross_paradigm',
                    'sence':         sence,
                    'feature_scene': feature_scene,
                    'test_stim':     tgt_stim,
                    'train_stim':    src_stim,
                    'accuracy':      acc,
                })

    return pd.DataFrame(rows)


def build_linked_table(feat_df: pd.DataFrame,
                        acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并特征表和准确率表：
    - 每行准确率先根据 sence 映射到 feature_scene；
    - S1 / cross_task1 匹配 s1 特征；S2 / cross_task2 匹配 s2 特征；
    - 合并键为 (subject, test_stim, feature_scene)；
    - 当前主分析仍默认使用 PRIMARY_ROI + PRIMARY_FEATURE。
    """
    if feat_df.empty or acc_df.empty:
        print('[WARN] 特征表或准确率表为空，无法合并。')
        return pd.DataFrame()

    feat_sel = feat_df[
        (feat_df['roi'] == PRIMARY_ROI) &
        (feat_df['feature_type'] == PRIMARY_FEATURE)
    ][['subject', 'stim', 'scene', 'intensity', 'discriminability',
       'cohens_dz', 'fisher_score']].copy()

    feat_sel = feat_sel.rename(columns={
        'stim': 'test_stim',
        'scene': 'feature_scene'
    })

    # 确保 acc_df 中存在 feature_scene。旧 CSV 或旧代码读取时也能兼容。
    acc2 = acc_df.copy()
    if 'feature_scene' not in acc2.columns:
        acc2['feature_scene'] = acc2['sence'].apply(_sence_to_feature_scene)

    linked = acc2.merge(
        feat_sel,
        on=['subject', 'test_stim', 'feature_scene'],
        how='left'
    )

    return linked


# ======================================================================
# 5. 绘图
# ======================================================================

STIM_COLORS = {
    'ssvideo': '#4C72B0',
    'video':   '#DD8452',
    'ssmvep':  '#55A868',
    'cue':     '#C44E52',
}
STIM_MARKERS = {
    'ssvideo': 'o',
    'video':   's',
    'ssmvep':  '^',
    'cue':     'D',
}


def _add_regression(ax, x, y, color='gray', alpha=0.7):
    """在 ax 上添加回归线和 Spearman ρ 注释。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    if xv.size < 5 or np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        ax.text(0.04, 0.96, 'n<5 or constant',
                transform=ax.transAxes, va='top', fontsize=8, color=color)
        return
    rho, p = stats.spearmanr(xv, yv)
    z = np.polyfit(xv, yv, 1)
    xline = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(xline, np.polyval(z, xline), '--', color=color, alpha=alpha, lw=1.5)
    stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    ax.text(0.04, 0.96, f'ρ={rho:.2f} {stars}\nn={xv.size}',
            transform=ax.transAxes, va='top', fontsize=8, color=color)


def _scatter_by_stim(ax, df: pd.DataFrame, x_col: str, y_col: str,
                     stim_col: str = 'test_stim', label: bool = True) -> None:
    """按 stim 上色/形状绘制散点。"""
    for stim in STIM_NAMES:
        g = df[df[stim_col] == stim]
        if g.empty:
            continue
        ax.scatter(
            g[x_col], g[y_col],
            label=stim if label else None,
            color=STIM_COLORS.get(stim, 'gray'),
            marker=STIM_MARKERS.get(stim, 'o'),
            s=35, alpha=0.75, linewidths=0.3, edgecolors='k'
        )


def plot_intensity_vs_discriminability(feat_df: pd.DataFrame,
                                        save_path: str) -> None:
    """
    散点：X = Intensity，Y = Discriminability。
    每个点 = 1 subject；颜色/形状 = stim；分面 = scene。
    """
    df = feat_df[
        (feat_df['roi'] == PRIMARY_ROI) &
        (feat_df['feature_type'] == PRIMARY_FEATURE)
    ].dropna(subset=['intensity', 'discriminability'])

    if df.empty:
        print('[SKIP] Intensity vs Discriminability: 数据为空')
        return

    scenes = [sc for sc in SCENES if sc in df['scene'].unique()]
    fig, axes = plt.subplots(1, len(scenes), figsize=(5 * len(scenes), 4.5),
                              constrained_layout=True, squeeze=False)
    for ax, sc in zip(axes[0], scenes):
        sub = df[df['scene'] == sc]
        for stim in STIM_NAMES:
            g = sub[sub['stim'] == stim]
            if g.empty:
                continue
            ax.scatter(g['intensity'], g['discriminability'],
                       label=stim, color=STIM_COLORS[stim],
                       marker=STIM_MARKERS[stim], s=35, alpha=0.75, linewidths=0.3,
                       edgecolors='k')
        _add_regression(ax, sub['intensity'].values, sub['discriminability'].values)
        ax.set_xlabel('Intensity  mean((|L|+|R|)/2)', fontsize=9)
        ax.set_ylabel('Discriminability  ‖L–R‖₂/√n', fontsize=9)
        ax.set_title(f'Scene: {sc.upper()}', fontsize=10)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f'Intensity vs Discriminability  [{PRIMARY_ROI}, {PRIMARY_FEATURE}]',
                 fontsize=11, fontweight='bold')
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_feature_vs_accuracy_by_protocol_scene(linked: pd.DataFrame,
                                                protocol: str,
                                                feature_scene: str,
                                                save_path: str,
                                                algorithms: Optional[List[str]] = None) -> None:
    """
    每个 protocol + scene 单独一张图。
    行：Intensity / Discriminability；列：FBCSP / FGMDRM / EEGNET。
    """
    sub = linked[
        (linked['protocol'] == protocol) &
        (linked['feature_scene'] == feature_scene) &
        linked['accuracy'].notna() &
        linked['intensity'].notna()
    ].copy()
    if sub.empty:
        print(f'[SKIP] {protocol}, scene={feature_scene}: accuracy 或 feature 数据为空')
        return

    algs = _selected_algorithms(sub, algorithms)
    if not algs:
        print(f'[SKIP] {protocol}, scene={feature_scene}: 没有可绘制的重点算法')
        return

    fig, axes = plt.subplots(2, len(algs), figsize=(5 * len(algs), 8),
                              constrained_layout=True, squeeze=False)

    def _panel(ax, data, x_col, xlabel, alg):
        g = data[data['algorithm'] == alg].dropna(subset=[x_col, 'accuracy'])
        _scatter_by_stim(ax, g, x_col, 'accuracy', stim_col='test_stim')
        _add_regression(ax, g[x_col].values, g['accuracy'].values)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Accuracy', fontsize=9)
        ax.set_title(f'{alg} | {protocol} | {feature_scene.upper()}', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, ncol=2)

    for ci, alg in enumerate(algs):
        _panel(axes[0][ci], sub, 'intensity',
               'Intensity  mean((|L|+|R|)/2)', alg)
        _panel(axes[1][ci], sub, 'discriminability',
               'Discriminability  ‖L–R‖₂/√n', alg)

    fig.suptitle(
        f'Feature Quality vs Accuracy — {protocol}, scene={feature_scene.upper()}\n'
        f'ROI={PRIMARY_ROI} | Feature={PRIMARY_FEATURE}',
        fontsize=11, fontweight='bold'
    )
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_feature_vs_accuracy_scene_matrix(linked: pd.DataFrame,
                                           protocol: str,
                                           x_col: str,
                                           save_path: str,
                                           algorithms: Optional[List[str]] = None) -> None:
    """
    场景 × 算法矩阵图：
    行 = s1/s2；列 = FBCSP/FGMDRM/EEGNET；X = intensity 或 discriminability；Y = accuracy。
    """
    sub = linked[
        (linked['protocol'] == protocol) &
        linked['accuracy'].notna() &
        linked[x_col].notna()
    ].copy()
    if sub.empty:
        print(f'[SKIP] {protocol}, x={x_col}: 数据为空')
        return

    algs = _selected_algorithms(sub, algorithms)
    scenes = [sc for sc in SCENES if sc in sub['feature_scene'].unique()]
    if not algs or not scenes:
        print(f'[SKIP] {protocol}, x={x_col}: 没有可绘制算法或场景')
        return

    fig, axes = plt.subplots(len(scenes), len(algs),
                              figsize=(5 * len(algs), 4.2 * len(scenes)),
                              constrained_layout=True, squeeze=False)
    for ri, sc in enumerate(scenes):
        for ci, alg in enumerate(algs):
            ax = axes[ri][ci]
            g = sub[(sub['feature_scene'] == sc) & (sub['algorithm'] == alg)]
            _scatter_by_stim(ax, g, x_col, 'accuracy', stim_col='test_stim')
            _add_regression(ax, g[x_col].values, g['accuracy'].values)
            xlabel = 'Intensity' if x_col == 'intensity' else 'Discriminability'
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel('Accuracy', fontsize=9)
            ax.set_title(f'{sc.upper()} | {alg}', fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        f'{x_col.capitalize()} vs Accuracy — {protocol}\n'
        f'ROI={PRIMARY_ROI} | Feature={PRIMARY_FEATURE}',
        fontsize=11, fontweight='bold'
    )
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_within_vs_cross_accuracy_by_scene(linked: pd.DataFrame,
                                            save_path: str,
                                            algorithms: Optional[List[str]] = None) -> None:
    """
    同一 subject/test_stim/algorithm/scene 的 within accuracy vs cross accuracy。
    行 = s1/s2；列 = FBCSP/FGMDRM/EEGNET。
    注意：cross 可能有多个 train_stim，因此同一个 within 点会对应多个 cross 点。
    """
    within = linked[linked['protocol'] == 'within_paradigm'][
        ['subject', 'test_stim', 'algorithm', 'feature_scene', 'accuracy']
    ].rename(columns={'accuracy': 'acc_within'})
    cross = linked[linked['protocol'] == 'cross_paradigm'][
        ['subject', 'test_stim', 'train_stim', 'algorithm', 'feature_scene', 'accuracy']
    ].rename(columns={'accuracy': 'acc_cross'})

    merged = within.merge(
        cross,
        on=['subject', 'test_stim', 'algorithm', 'feature_scene'],
        how='inner'
    ).dropna(subset=['acc_within', 'acc_cross'])

    if merged.empty:
        print('[SKIP] within vs cross scatter: 数据为空')
        return

    algs = _selected_algorithms(merged, algorithms)
    scenes = [sc for sc in SCENES if sc in merged['feature_scene'].unique()]
    if not algs or not scenes:
        print('[SKIP] within vs cross scatter: 没有可绘制算法或场景')
        return

    fig, axes = plt.subplots(len(scenes), len(algs),
                              figsize=(5 * len(algs), 4.3 * len(scenes)),
                              constrained_layout=True, squeeze=False)
    for ri, sc in enumerate(scenes):
        for ci, alg in enumerate(algs):
            ax = axes[ri][ci]
            g = merged[(merged['feature_scene'] == sc) & (merged['algorithm'] == alg)]
            _scatter_by_stim(ax, g, 'acc_within', 'acc_cross', stim_col='test_stim')
            _add_regression(ax, g['acc_within'].values, g['acc_cross'].values)
            ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.4)
            ax.set_xlabel('Within-paradigm accuracy', fontsize=9)
            ax.set_ylabel('Cross-paradigm accuracy', fontsize=9)
            ax.set_title(f'{sc.upper()} | {alg}', fontsize=9)
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle('Within-paradigm vs Cross-paradigm Accuracy by Scene',
                 fontsize=11, fontweight='bold')
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_overview_by_algorithm_scene(linked: pd.DataFrame,
                                      feat_df: pd.DataFrame,
                                      algorithm: str,
                                      feature_scene: str,
                                      save_path: str) -> None:
    """
    每个算法 × 每个场景一张 5-panel 总览图：
    1) Intensity vs Discriminability
    2) Intensity vs Within-ACC
    3) Intensity vs Cross-ACC
    4) Discriminability vs Within-ACC
    5) Discriminability vs Cross-ACC
    """
    feat_primary = feat_df[
        (feat_df['roi'] == PRIMARY_ROI) &
        (feat_df['feature_type'] == PRIMARY_FEATURE) &
        (feat_df['scene'] == feature_scene)
    ].dropna(subset=['intensity', 'discriminability'])

    within = linked[
        (linked['protocol'] == 'within_paradigm') &
        (linked['algorithm'] == algorithm) &
        (linked['feature_scene'] == feature_scene)
    ].dropna(subset=['accuracy'])

    cross = linked[
        (linked['protocol'] == 'cross_paradigm') &
        (linked['algorithm'] == algorithm) &
        (linked['feature_scene'] == feature_scene)
    ].dropna(subset=['accuracy'])

    if feat_primary.empty and within.empty and cross.empty:
        print(f'[SKIP] Overview: {algorithm}, scene={feature_scene} 数据为空')
        return

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = gridspec.GridSpec(3, 2, figure=fig)
    axes = [
        fig.add_subplot(gs[0, :]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    # 1) Intensity vs Discriminability
    ax = axes[0]
    for stim in STIM_NAMES:
        g = feat_primary[feat_primary['stim'] == stim]
        if g.empty:
            continue
        ax.scatter(g['intensity'], g['discriminability'],
                   label=stim, color=STIM_COLORS[stim], marker=STIM_MARKERS[stim],
                   s=40, alpha=0.75, linewidths=0.3, edgecolors='k')
    _add_regression(ax, feat_primary['intensity'].values, feat_primary['discriminability'].values)
    ax.set_xlabel('Intensity', fontsize=9)
    ax.set_ylabel('Discriminability', fontsize=9)
    ax.set_title(f'Feature structure | {feature_scene.upper()}', fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc='upper right')

    def _panel(ax, df, x_col, title):
        df2 = df.dropna(subset=[x_col, 'accuracy'])
        _scatter_by_stim(ax, df2, x_col, 'accuracy', stim_col='test_stim')
        _add_regression(ax, df2[x_col].values, df2['accuracy'].values)
        ax.set_xlabel('Intensity' if x_col == 'intensity' else 'Discriminability', fontsize=8)
        ax.set_ylabel('Accuracy', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, ncol=2)

    _panel(axes[1], within, 'intensity',
           f'Intensity → Within-ACC | {algorithm}')
    _panel(axes[2], cross, 'intensity',
           f'Intensity → Cross-ACC | {algorithm}')
    _panel(axes[3], within, 'discriminability',
           f'Discriminability → Within-ACC | {algorithm}')
    _panel(axes[4], cross, 'discriminability',
           f'Discriminability → Cross-ACC | {algorithm}')

    fig.suptitle(
        f'Feature Quality vs Classification Accuracy — {algorithm}, scene={feature_scene.upper()}\n'
        f'ROI={PRIMARY_ROI} | Feature={PRIMARY_FEATURE}',
        fontsize=12, fontweight='bold'
    )
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


# ======================================================================
# 6. 相关性统计表
# ======================================================================

def compute_correlation_table(linked: pd.DataFrame,
                              algorithms: Optional[List[str]] = None) -> pd.DataFrame:
    """
    按 protocol × feature_scene × algorithm 分层计算相关性。
    这样不会把 s1 和 s2 混在一起，也不会用 s1 特征解释 s2 准确率。
    """
    rows = []
    if linked.empty:
        return pd.DataFrame()

    algs = algorithms or FOCUS_ALGORITHMS
    for protocol in ['within_paradigm', 'cross_paradigm']:
        for feature_scene in SCENES:
            sub = linked[
                (linked['protocol'] == protocol) &
                (linked['feature_scene'] == feature_scene)
            ]
            if sub.empty:
                continue
            for alg in algs:
                g = sub[sub['algorithm'] == alg].dropna(
                    subset=['intensity', 'discriminability', 'accuracy'])
                if len(g) < 5:
                    continue
                for x_col, x_label in [('intensity', 'Intensity'),
                                       ('discriminability', 'Discriminability')]:
                    if g[x_col].nunique(dropna=True) < 2 or g['accuracy'].nunique(dropna=True) < 2:
                        rho, p = np.nan, np.nan
                    else:
                        rho, p = stats.spearmanr(g[x_col], g['accuracy'])
                    rows.append({
                        'protocol':      protocol,
                        'feature_scene': feature_scene,
                        'algorithm':     alg,
                        'x_var':         x_label,
                        'n':             len(g),
                        'spearman_rho':  round(float(rho), 4) if np.isfinite(rho) else np.nan,
                        'p_value':       round(float(p), 6) if np.isfinite(p) else np.nan,
                        'sig': '***' if np.isfinite(p) and p < 0.001 else
                               ('**' if np.isfinite(p) and p < 0.01 else
                                ('*' if np.isfinite(p) and p < 0.05 else 'ns')),
                    })
    return pd.DataFrame(rows)


# ======================================================================
# 7. 主流程
# ======================================================================

def main():
    print('=' * 70)
    print('Step14: Per-subject Intensity / Discriminability / Accuracy')
    print('Scene-matched version: S1/cross_task1→s1, S2/cross_task2→s2')
    print('=' * 70)

    # ---- 特征表 ----
    print('\n[1/4] 计算 per-subject 特征指标 ...')
    feat_df = build_feature_table()
    if not feat_df.empty:
        p = os.path.join(SAVE_DIR, 'per_subject_features.csv')
        feat_df.to_csv(p, index=False, encoding='utf-8-sig')
        print(f'  特征行数: {len(feat_df):,}')
        print(f'  [SAVED] {p}')
    else:
        print('  [WARN] 特征表为空（TOPO/TF 文件不存在？）')

    # ---- 准确率表 ----
    print('\n[2/4] 读取 per-subject 准确率 ...')
    acc_df = build_accuracy_table()
    non_nan = acc_df['accuracy'].notna().sum() if 'accuracy' in acc_df.columns else 0
    print(f'  准确率行数: {len(acc_df):,}，有效值: {non_nan}')
    if non_nan == 0:
        print('  [WARN] 准确率全为 NaN — 请先运行 main.py 生成结果文件，')
        print('         再执行本脚本。已生成模板 CSV 供参考。')
    else:
        miss = (acc_df[acc_df['accuracy'].isna()]
                .groupby(['protocol', 'sence', 'train_stim', 'test_stim'])
                .size()
                .reset_index(name='missing_rows'))
        if not miss.empty:
            print('\n  缺失准确率分布：')
            print(miss.to_string(index=False))
    p = os.path.join(SAVE_DIR, 'accuracy_long.csv')
    acc_df.to_csv(p, index=False, encoding='utf-8-sig')
    print(f'  [SAVED] {p}')

    # ---- 合并 ----
    print('\n[3/4] 按场景合并特征与准确率 ...')
    linked = build_linked_table(feat_df, acc_df)
    if not linked.empty:
        p = os.path.join(SAVE_DIR, 'linked_data.csv')
        linked.to_csv(p, index=False, encoding='utf-8-sig')
        print(f'  合并行数: {len(linked):,}')
        print(f'  [SAVED] {p}')

        merge_miss = linked[['intensity', 'discriminability']].isna().any(axis=1).sum()
        print(f'  合并后缺失特征行数: {merge_miss}')

        corr = compute_correlation_table(linked, FOCUS_ALGORITHMS)
        if not corr.empty:
            p = os.path.join(SAVE_DIR, 'correlation_table.csv')
            corr.to_csv(p, index=False, encoding='utf-8-sig')
            print(f'\n  相关性摘要（重点算法：{", ".join(FOCUS_ALGORITHMS)}）:')
            print(corr.to_string(index=False))
            print(f'\n  [SAVED] {p}')

    # ---- 绘图 ----
    print('\n[4/4] 绘图 ...')
    figdir = os.path.join(SAVE_DIR, 'figures')
    os.makedirs(figdir, exist_ok=True)
    overview_dir = os.path.join(figdir, 'overview_by_algorithm_scene')
    os.makedirs(overview_dir, exist_ok=True)

    if not feat_df.empty:
        plot_intensity_vs_discriminability(
            feat_df,
            os.path.join(figdir, 'Fig1_Intensity_vs_Discriminability_byScene.png'))

    if not linked.empty:
        # 1) 每个 protocol × scene 一张 2×3 图，重点算法分列
        for protocol in ['within_paradigm', 'cross_paradigm']:
            for sc in SCENES:
                plot_feature_vs_accuracy_by_protocol_scene(
                    linked, protocol, sc,
                    os.path.join(figdir, f'Fig2_Feature_vs_Accuracy_{protocol}_{sc}.png'),
                    FOCUS_ALGORITHMS)

        # 2) 场景 × 算法矩阵图，分别看 Intensity 和 Discriminability
        for protocol in ['within_paradigm', 'cross_paradigm']:
            plot_feature_vs_accuracy_scene_matrix(
                linked, protocol, 'intensity',
                os.path.join(figdir, f'Fig3_Intensity_vs_Accuracy_matrix_{protocol}.png'),
                FOCUS_ALGORITHMS)
            plot_feature_vs_accuracy_scene_matrix(
                linked, protocol, 'discriminability',
                os.path.join(figdir, f'Fig4_Discriminability_vs_Accuracy_matrix_{protocol}.png'),
                FOCUS_ALGORITHMS)

        # 3) within vs cross，按场景和算法拆开
        plot_within_vs_cross_accuracy_by_scene(
            linked,
            os.path.join(figdir, 'Fig5_Within_vs_Cross_Accuracy_bySceneAlgorithm.png'),
            FOCUS_ALGORITHMS)

        # 4) 每个算法 × 每个场景一张总览图
        for alg in FOCUS_ALGORITHMS:
            for sc in SCENES:
                plot_overview_by_algorithm_scene(
                    linked, feat_df, alg, sc,
                    os.path.join(overview_dir, f'Overview_{alg}_{sc}.png'))

    print('\nDone. 输出目录:', SAVE_DIR)


if __name__ == '__main__':
    main()