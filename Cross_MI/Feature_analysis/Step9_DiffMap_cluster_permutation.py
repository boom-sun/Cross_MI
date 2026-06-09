# ===== Step9_DiffMap_cluster_permutation.py =====
# 目的：用 cluster-based permutation test 检验“类别差异时频图”中的连续显著区域。
# 注意：该脚本建议在单个通道、单个范式、单个场景 s1/s2 下使用，避免一次性计算过重。

import os
from pathlib import Path
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test

# ------------------- 基本参数 -------------------
sence_id = 's1'
stim = 'cue'
class_pair = (1, 2)
ch_name = 'C3'
tf_band = (5, 40)
tf_timewin = (-2, 4)
n_permutations = 1000
threshold = None  # None 表示 MNE 自动设置阈值；也可设 t 阈值
alpha_cluster = 0.05

CHANNELS = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ',
    'PO4','PO6','PO8','O1','OZ','O2'
]

save_root_DIFF = r'E:\Datasets\4_跨场景因素研究v2\画图结果\差异图'
save_dir = r'E:\Datasets\4_跨场景因素研究v2\画图结果\差异图谱聚类检验结果'
Path(save_dir).mkdir(parents=True, exist_ok=True)

c1, c2 = class_pair
p_diff = os.path.join(save_root_DIFF, f'DiffMap_{stim}_class{c1}_minus_class{c2}.mat')
data = hdf5storage.loadmat(p_diff)

tf_diff = data['tf_diff'][sence_id]  # [subject, freq, time, channel]
times = np.squeeze(data['times'])
freqs = np.squeeze(data['freqs'])

ch_idx = CHANNELS.index(ch_name)
times_idx = np.where((times >= tf_timewin[0] * 1000) & (times <= tf_timewin[1] * 1000))[0]
freqs_idx = np.where((freqs >= tf_band[0]) & (freqs <= tf_band[1]))[0]

x = np.take(tf_diff, indices=freqs_idx, axis=1)
x = np.take(x, indices=times_idx, axis=2)
x = x[:, :, :, ch_idx]  # [subject, freq, time]

# 1-sample cluster permutation: 检验 class1-class2 是否显著偏离0
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    x,
    n_permutations=n_permutations,
    threshold=threshold,
    tail=0,
    out_type='mask',
    seed=42,
    n_jobs=1
)

sig_mask = np.zeros(T_obs.shape, dtype=bool)
for cl, p in zip(clusters, cluster_p_values):
    if p <= alpha_cluster:
        sig_mask |= cl

x_times = times[times_idx] / 1000.0
y_freqs = freqs[freqs_idx]
mean_diff = np.mean(x, axis=0)

fig, axes = plt.subplots(2, 1, figsize=(4.2, 6.5), constrained_layout=True)
im1 = axes[0].pcolormesh(x_times, y_freqs, mean_diff, shading='auto', cmap='RdBu_r')
axes[0].axvline(0.0, color='k', ls='--', lw=1)
axes[0].set_title(f'Mean class-difference TF: {stim}-{sence_id}-{ch_name}')
axes[0].set_ylabel('Hz')
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04).set_label('Power diff')

im2 = axes[1].pcolormesh(x_times, y_freqs, sig_mask.astype(float), shading='auto', cmap='Reds', vmin=0, vmax=1)
axes[1].axvline(0.0, color='k', ls='--', lw=1)
axes[1].set_title(f'Significant clusters, p <= {alpha_cluster}')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Hz')
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04).set_label('cluster mask')

p_save = os.path.join(save_dir, f'ClusterPerm_DiffTF_{stim}_{sence_id}_class{c1}-{c2}_{ch_name}.png')
fig.savefig(p_save, dpi=300, bbox_inches='tight')
plt.close(fig)

hdf5storage.savemat(os.path.join(save_dir, f'ClusterPerm_DiffTF_{stim}_{sence_id}_class{c1}-{c2}_{ch_name}.mat'), {
    'T_obs': T_obs,
    'cluster_p_values': cluster_p_values,
    'sig_mask': sig_mask.astype(np.int8),
    'times': x_times,
    'freqs': y_freqs
})

print(f'Saved cluster permutation result: {p_save}')
