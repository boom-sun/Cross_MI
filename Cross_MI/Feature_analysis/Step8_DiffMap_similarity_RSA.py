# ===== Step8_DiffMap_similarity_RSA.py =====
# 目的：比较不同范式之间“类别差异结构”的相似性。
# 重点：不是比较原始强度，而是比较 class1-class2 后的差异向量。
# 输出：Pearson/Spearman/Cosine 相似性矩阵、CKA相似性矩阵，以及“强度相似性 vs 差异结构相似性”的表。

import os
from pathlib import Path
import numpy as np
import hdf5storage
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- 基本参数 -------------------
class_pair = (1, 2)
sence_id = 's1'  # 's1' or 's2'
analysis_level = 'topo'  # 'tf' or 'topo'

topo_band = (8, 13)
topo_time = (0, 4)
tf_band = (8, 30)
tf_timewin = (0, 4)
channels_choose = ['C3', 'C4', 'CZ']  # 关注运动区；也可加入 'OZ' 检查视觉贡献

CHANNELS = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ',
    'PO4','PO6','PO8','O1','OZ','O2'
]

# 每个范式对应 Step6 输出目录。按你的实际路径修改。
PARADIGM_PATH = {
    'ssvideo': r'E:\Datasets\4_跨场景因素研究v2\画图结果\差异图\DiffMap_ssvideo_class1_minus_class2.mat',
    'video': r'E:\Datasets\4_跨场景因素研究v2\画图结果\差异图\DiffMap_video_class1_minus_class2.mat',
    'ssmvep': r'E:\Datasets\4_跨场景因素研究v2\画图结果\差异图\DiffMap_ssmvep_class1_minus_class2.mat',
    'cue': r'E:\Datasets\4_跨场景因素研究v2\画图结果\差异图\DiffMap_cue_class1_minus_class2.mat',
}

save_dir = r'E:\Datasets\4_跨场景因素研究v2\画图结果\差异图'
Path(save_dir).mkdir(parents=True, exist_ok=True)


def flatten_topo_diff(data):
    key = 'freq' + str(topo_band) + 'time' + str(topo_time)
    v = np.squeeze(data['topo_stats'][sence_id][key]['mean_diff'])
    return v.reshape(1, -1)


def flatten_tf_diff(data):
    times = np.squeeze(data['times'])
    freqs = np.squeeze(data['freqs'])
    times_idx = np.where((times >= tf_timewin[0] * 1000) & (times <= tf_timewin[1] * 1000))[0]
    freqs_idx = np.where((freqs >= tf_band[0]) & (freqs <= tf_band[1]))[0]
    ch_idx = [CHANNELS.index(ch) for ch in channels_choose]

    x = data['tf_stats'][sence_id]['mean_diff']  # [freq, time, channel]
    x = np.take(x, indices=freqs_idx, axis=0)
    x = np.take(x, indices=times_idx, axis=1)
    x = np.take(x, indices=ch_idx, axis=2)
    return x.reshape(1, -1)


def linear_cka(x, y, eps=1e-12):
    """Linear CKA for two row-wise observations matrices.
    若这里只有一个平均差异向量，则等价于归一化相似度；如要 trial/subject 层 CKA，可改为输入 subject x features。
    """
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(x.T @ y, ord='fro') ** 2
    norm_x = np.linalg.norm(x.T @ x, ord='fro')
    norm_y = np.linalg.norm(y.T @ y, ord='fro')
    return hsic / (norm_x * norm_y + eps)


def safe_pearson(x, y):
    return stats.pearsonr(x.ravel(), y.ravel())[0]


def safe_spearman(x, y):
    return stats.spearmanr(x.ravel(), y.ravel()).correlation


names = list(PARADIGM_PATH.keys())
vectors = {}
strength = {}

for name, path in PARADIGM_PATH.items():
    data = hdf5storage.loadmat(path)
    if analysis_level == 'topo':
        vec = flatten_topo_diff(data)
    else:
        vec = flatten_tf_diff(data)
    vectors[name] = vec
    strength[name] = float(np.mean(np.abs(vec)))

pearson_mat = np.zeros((len(names), len(names)))
spearman_mat = np.zeros_like(pearson_mat)
cosine_mat = np.zeros_like(pearson_mat)
cka_mat = np.zeros_like(pearson_mat)

rows = []
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        xi = vectors[ni]
        xj = vectors[nj]
        pearson_mat[i, j] = safe_pearson(xi, xj)
        spearman_mat[i, j] = safe_spearman(xi, xj)
        cosine_mat[i, j] = cosine_similarity(xi, xj)[0, 0]
        cka_mat[i, j] = linear_cka(xi, xj)
        rows.append({
            'paradigm_i': ni,
            'paradigm_j': nj,
            'pearson_diff_similarity': pearson_mat[i, j],
            'spearman_diff_similarity': spearman_mat[i, j],
            'cosine_diff_similarity': cosine_mat[i, j],
            'cka_diff_similarity': cka_mat[i, j],
            'strength_i': strength[ni],
            'strength_j': strength[nj],
            'strength_abs_gap': abs(strength[ni] - strength[nj])
        })

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(save_dir, f'DiffMap_similarity_{analysis_level}_{sence_id}.csv'), index=False, encoding='utf-8-sig')

for metric_name, mat in [('Pearson', pearson_mat), ('Spearman', spearman_mat), ('Cosine', cosine_mat), ('CKA', cka_mat)]:
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(mat, vmin=-1 if metric_name != 'CKA' else 0, vmax=1, cmap='RdBu_r' if metric_name != 'CKA' else 'viridis')
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right')
    ax.set_yticklabels(names)
    ax.set_title(f'{metric_name} similarity of class-difference maps')
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(save_dir, f'{metric_name}_DiffMap_similarity_{analysis_level}_{sence_id}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

print(summary)
print(f'Saved similarity results to: {save_dir}')
