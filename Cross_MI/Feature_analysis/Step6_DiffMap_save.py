# ===== Step6_DiffMap_save.py =====
# 目的：从“特征强度图”转向“类别差异图谱”。
# 输入：Step1_TOPO_TF_save.py 已保存的 TOPO_*.mat 和 TF_*.mat
# 输出：每个范式/场景下 class1-class2 的差异脑地形图、差异时频图、效应量、Fisher score、FDR p值。

import os
from pathlib import Path
import numpy as np
import hdf5storage
import scipy.io as sio
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ------------------- 基本参数（按需修改） -------------------
sence = 'ssmvep_hybrid'  # 'ssmvep_hybrid' | 'ssmvep' | 'graz' | 'kjz'
class_pair = (1, 2)      # 默认分析 Left - Right
significance = 0.05

# 与原 Feature_analysis 保持一致
topo_band = [(8, 13), (13, 30), (13, 20), (20, 30)]
topo_timewin = [(0, 4), (-2, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2),
                (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4)]
is_trialClean=True
if is_trialClean == True:
    _trialClean = '_trialClean'
else:
    _trialClean = None
CHANNELS = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ',
    'PO4','PO6','PO8','O1','OZ','O2'
]

# ------------------- 路径配置（按你的机器路径修改） -------------------
if sence == 'ssmvep_hybrid':
    stim_name = ('ssvideo', 'video', 'ssmvep', 'cue')
    save_root_TOPO = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\脑地形图数据'
    save_root_TF = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\时频图数据'
    save_root_DIFF = r'E:\Datasets\4_跨场景因素研究v2\画图结果\差异图'

elif sence == 'ssmvep':
    stim_name = ('default',)
    save_root_TOPO = r'E:\Datasets\2_MI_SSMVEP范式\画图数据\脑地形图数据'
    save_root_TF = r'E:\Datasets\2_MI_SSMVEP范式\画图数据\时频图数据'
    save_root_DIFF = r'E:\Datasets\2_MI_SSMVEP范式\画图数据\差异图谱数据'

elif sence == 'graz':
    stim_name = ('default',)
    save_root_TOPO = r'E:\Datasets\1_Graz范式\Graz画图数据\脑地形图数据'
    save_root_TF = r'E:\Datasets\1_Graz范式\Graz画图数据\时频图数据'
    save_root_DIFF = r'E:\Datasets\1_Graz范式\Graz画图数据\差异图谱数据'

elif sence == 'kjz':
    stim_name = ('default',)
    save_root_TOPO = r'E:\Datasets\6_天基测试数据\脑地形图数据\熊猫采集数据'
    save_root_TF = r'E:\Datasets\6_天基测试数据\时频图数据\熊猫采集数据'
    save_root_DIFF = r'E:\Datasets\6_天基测试数据\差异图谱数据\熊猫采集数据'

Path(save_root_DIFF).mkdir(parents=True, exist_ok=True)


def _load_topo(class_choose, stim='default'):
    if sence == 'ssmvep_hybrid':
        p_topo = os.path.join(save_root_TOPO, f'TOPO_{stim}_class{class_choose}{_trialClean}.mat')
    else:
        p_topo = os.path.join(save_root_TOPO, f'TOPO_class{class_choose}{_trialClean}.mat')
    return sio.loadmat(p_topo)['topo'][0,0]


def _load_tf(class_choose, stim='default'):
    if sence == 'ssmvep_hybrid':
        p_tf = os.path.join(save_root_TF, f'TF_{stim}_class{class_choose}{_trialClean}.mat')
        p_timefreq = os.path.join(save_root_TF, f'times+freqs_{stim}_class{class_choose}.mat')
    else:
        p_tf = os.path.join(save_root_TF, f'TF_class{class_choose}{_trialClean}.mat')
        p_timefreq = os.path.join(save_root_TF, f'times+freqs_class{class_choose}.mat')
    tf = sio.loadmat(p_tf)['tf'][0,0]
    tf_axis = hdf5storage.loadmat(p_timefreq)
    return tf, np.squeeze(tf_axis['times']), np.squeeze(tf_axis['freqs'])


def paired_effect_size(x1, x2, axis=0, eps=1e-8):
    """配对 Cohen's dz: mean(diff) / std(diff)."""
    diff = x1 - x2
    return np.mean(diff, axis=axis) / (np.std(diff, axis=axis, ddof=1) + eps)


def fisher_score(x1, x2, axis=0, eps=1e-8):
    """两类均值差异相对类内方差，强调分类可分性。"""
    return (np.mean(x1, axis=axis) - np.mean(x2, axis=axis)) ** 2 / (
        np.var(x1, axis=axis, ddof=1) + np.var(x2, axis=axis, ddof=1) + eps)


def fdr_pvals(p, alpha=0.05):
    shape = p.shape
    _, p_corr, _, _ = multipletests(p.reshape(-1), alpha=alpha, method='fdr_bh')
    p_corr = p_corr.reshape(shape)
    sig_mask = p_corr <= alpha
    return p_corr, sig_mask


for stim in stim_name:
    c1, c2 = class_pair

    # ------------------- 1) 类别差异脑地形图 -------------------
    topo_c1 = _load_topo(c1, stim)
    topo_c2 = _load_topo(c2, stim)

    topo_diff = {'s1': {}, 's2': {}}
    topo_stats = {'s1': {}, 's2': {}}

    for sence_id in ['s1', 's2']:
        for band in topo_band:
            for time in topo_timewin:
                key = 'freq' + str(band) + 'time' + str(time)
                x1 = topo_c1[sence_id][0,0][key]  # [channel, subject]
                x2 = topo_c2[sence_id][0,0][key]
                diff = x1 - x2  # 之前已经做了对数处理

                t_res = stats.ttest_rel(x1, x2, axis=1, nan_policy='omit')
                p_corr, sig_mask = fdr_pvals(t_res.pvalue, alpha=significance)

                topo_diff[sence_id][key] = diff
                topo_stats[sence_id][key] = {
                    'mean_diff': np.mean(diff, axis=1),
                    'abs_strength': np.mean(np.abs(diff), axis=1),
                    'cohens_dz': paired_effect_size(x1, x2, axis=1),  # 配对 Cohen's dz: mean(diff) / std(diff).
                    'fisher_score': fisher_score(x1, x2, axis=1),    # 两类均值差异相对类内方差，强调分类可分性
                    't_value': t_res.statistic,
                    'p_value': t_res.pvalue,
                    'p_fdr': p_corr,
                    'sig_fdr': sig_mask.astype(np.int8)
                }

    # ------------------- 2) 类别差异时频图 -------------------
    tf_c1, times, freqs = _load_tf(c1, stim)
    tf_c2, _, _ = _load_tf(c2, stim)

    tf_diff = {'s1': None, 's2': None}
    tf_stats = {'s1': {}, 's2': {}}

    for sence_id in ['s1', 's2']:
        x1 = tf_c1[sence_id]  # [subject, freq, time, channel]
        x2 = tf_c2[sence_id]
        diff = x1 - x2
        t_res = stats.ttest_rel(x1, x2, axis=0, nan_policy='omit')
        p_corr, sig_mask = fdr_pvals(t_res.pvalue, alpha=significance)

        tf_diff[sence_id] = diff
        tf_stats[sence_id] = {
            'mean_diff': np.mean(diff, axis=0),
            'abs_strength': np.mean(np.abs(diff), axis=0),
            'cohens_dz': paired_effect_size(x1, x2, axis=0),
            'fisher_score': fisher_score(x1, x2, axis=0),
            't_value': t_res.statistic,
            'p_value': t_res.pvalue,
            'p_fdr': p_corr,
            'sig_fdr': sig_mask.astype(np.int8)
        }

    save_name = f'DiffMap_{stim}_class{c1}_minus_class{c2}{_trialClean}.mat' if sence == 'ssmvep_hybrid' else \
        f'DiffMap_class{c1}_minus_class{c2}{_trialClean}.mat'
    p_save = os.path.join(save_root_DIFF, save_name)
    sio.savemat(p_save, {
        'topo_diff': topo_diff,
        'topo_stats': topo_stats,
        'tf_diff': tf_diff,
        'tf_stats': tf_stats,
        'times': times,
        'freqs': freqs,
        'channels': np.array(CHANNELS, dtype=object),
        'class_pair': np.array(class_pair),
        'topo_band': np.array(topo_band, dtype=object),
        'topo_timewin': np.array(topo_timewin, dtype=object)
    })
    print(f'Saved: {p_save}')
