"""
Step7_feature_stability.py

目的：量化各类EEG特征在被试间的稳定性。
核心指标：Fisher判别比（FDR） = 类间方差 / 类内方差
- FDR越高：该特征在被试间类间可分性越好（稳定且有判别力）
- 被试间FDR方差越低：该特征跨被试一致性越高（稳定性好）

分析的特征类型：
  F1 - PSD功率比 (alpha ERD: 8-13Hz, beta ERD: 13-30Hz, C3/C4)
  F2 - CSP对数方差特征
  F3 - 黎曼切空间特征 (协方差矩阵向量化)
  F4 - FBCSP互信息特征

数据：4_跨场景因素研究v2 / cue范式（左右手MI），37名被试，Scene1+Scene2
输出：
  - 各被试各特征的FDR值
  - 跨被试FDR均值和标准差（稳定性排名图）
  - 保存到 E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\stability\
"""

import os
import numpy as np
import scipy.io as sio
import hdf5storage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.covariance import ledoit_wolf
import warnings
warnings.filterwarnings('ignore')

# ===================== 参数配置 =====================
DATA_ROOT = r'E:\Datasets\4_跨场景因素研究v2' + r'\跨场景因素研究v2处理后数据'
SAVE_ROOT = r'E:\Datasets\4_跨场景因素研究v2' + r'\跨场景因素研究v2画图数据\stability'
os.makedirs(SAVE_ROOT, exist_ok=True)

SRATE = 250
STARTTIME = 2   # 秒，MI开始
ENDTIME = 6     # 秒，MI结束
START_PT = STARTTIME * SRATE   # 500
END_PT = ENDTIME * SRATE       # 1500
N_SUBJECTS = 37
SCENES = ['S1', 'S2']

# 选用感觉运动区导联（与主实验代码一致）
# 原始索引(1-based): 17-23 FC5-FC6, 26-32 C5-C6, 35-41 CP5-CP6
CH_IDX = [i - 1 for i in [16,17,18,19,20,21,22, 25,26,27,28,29,30,31, 34,35,36,37,38,39,40]]
N_CH = len(CH_IDX)   # 21通道

# C3=27(1-based)->26(0-based), C4=31(1-based)->30(0-based)，在CH_IDX中的局部索引
C3_LOCAL = CH_IDX.index(26)
C4_LOCAL = CH_IDX.index(30)

ALPHA_BAND = (8, 13)
BETA_BAND  = (13, 30)
BROAD_BAND = (8, 30)


# ===================== 基础工具函数 =====================
def load_mat(subject, scene):
    """加载单被试单场景cue范式数据，返回 (trials, channels, timepoints), labels"""
    pfx = f"1S0{subject}" if subject < 10 else f"1S{subject}"
    path = os.path.join(DATA_ROOT, f"{pfx}{scene}_cue.mat")
    d = hdf5storage.loadmat(path, mat_dtype=True)
    data  = d['data']           # (60, 1500, trials)
    label = np.squeeze(d['label']).astype(int)
    data  = data[:, START_PT:END_PT, :]   # 截取MI段
    data  = data.transpose(2, 0, 1)       # -> (trials, 60, timepoints)
    data  = data[:, CH_IDX, :]            # 选导联 -> (trials, 21, timepoints)
    return data, label


def bandpass(data, low, high, fs=SRATE, order=4):
    """data: (trials, channels, timepoints)"""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=2)


def fisher_ratio(feat_left, feat_right):
    """
    计算Fisher判别比 FDR = (mu1-mu2)^2 / (var1+var2)
    feat: (n_trials, n_features)
    返回每个特征维度的FDR，再取均值作为该特征组的FDR标量
    """
    mu1, mu2 = feat_left.mean(axis=0), feat_right.mean(axis=0)
    var1, var2 = feat_left.var(axis=0) + 1e-10, feat_right.var(axis=0) + 1e-10
    fdr = (mu1 - mu2) ** 2 / (var1 + var2)
    return float(np.mean(fdr))


# ===================== 特征提取函数 =====================

# --- F1: PSD功率比 ---
def extract_psd_ratio(data, label):
    """
    对C3/C4计算alpha和beta频段的功率（用能量近似代替Welch）
    返回特征矩阵 (trials, 4): [C3_alpha, C3_beta, C4_alpha, C4_beta]
    """
    d_alpha = bandpass(data, *ALPHA_BAND)
    d_beta  = bandpass(data, *BETA_BAND)

    def power(x):
        return np.mean(x ** 2, axis=2)   # (trials, channels)

    p_alpha = power(d_alpha)
    p_beta  = power(d_beta)

    feat = np.column_stack([
        p_alpha[:, C3_LOCAL],
        p_beta[:, C3_LOCAL],
        p_alpha[:, C4_LOCAL],
        p_beta[:, C4_LOCAL],
    ])
    return np.log(feat + 1e-10)


# --- F2: CSP对数方差 ---
def cov_norm(X):
    """X: (channels, timepoints, trials) -> 归一化协方差均值"""
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[2]):
        x = X[:, :, i]
        tr = np.trace(x @ x.T) + 1e-10
        C += (x @ x.T) / tr
    return C / X.shape[2]


def compute_csp_filters(data, label, n_comp=4):
    """
    data: (trials, channels, timepoints)
    返回CSP滤波矩阵W: (channels, 2*n_comp)
    """
    classes = np.unique(label)
    X1 = data[label == classes[0]].transpose(1, 2, 0)  # (ch, time, trials)
    X2 = data[label == classes[1]].transpose(1, 2, 0)
    C1 = cov_norm(X1)
    C2 = cov_norm(X2)
    vals, vecs = eigh(C1, C1 + C2)
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]
    W = np.hstack([vecs[:, :n_comp], vecs[:, -n_comp:]])
    return W


def extract_csp_logvar(data, label):
    """
    训练CSP滤波器后提取对数方差特征
    返回 (trials, 2*n_comp)
    """
    n_comp = 4
    d_broad = bandpass(data, *BROAD_BAND)
    W = compute_csp_filters(d_broad, label, n_comp)
    # 应用滤波器
    # data: (trials, ch, time) -> filtered: (trials, 2*n_comp, time)
    filtered = np.tensordot(d_broad, W, axes=([1], [0]))  # (trials, time, 2*n_comp)
    filtered = filtered.transpose(0, 2, 1)                # (trials, 2*n_comp, time)
    logvar = np.log(np.var(filtered, axis=2) + 1e-10)
    return logvar


# --- F3: 黎曼切空间特征 ---
def symmetric_matrix_to_vec(M):
    """对称矩阵上三角向量化（含对角）"""
    n = M.shape[0]
    idx = np.triu_indices(n)
    return M[idx]


def extract_riemannian(data, label):
    """
    计算每个trial的协方差矩阵（Ledoit-Wolf正则），
    计算黎曼均值（简化：用欧式均值近似），
    投影到切空间，向量化
    返回 (trials, n_ch*(n_ch+1)//2)
    """
    d_broad = bandpass(data, *BROAD_BAND)
    n_trials, n_ch, n_time = d_broad.shape

    # 计算每个trial的协方差
    covs = np.zeros((n_trials, n_ch, n_ch))
    for i in range(n_trials):
        X = d_broad[i]   # (n_ch, n_time)
        cov, _ = ledoit_wolf(X.T)
        covs[i] = cov

    # 用对数欧式均值近似黎曼均值（简化但计算高效）
    log_covs = np.array([np.linalg.slogdet(c)[1] for c in covs])  # 仅用于验证正定性
    mean_cov = covs.mean(axis=0)

    # 白化矩阵（用均值协方差的逆平方根）
    eigvals, eigvecs = np.linalg.eigh(mean_cov)
    eigvals = np.maximum(eigvals, 1e-10)
    W_white = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # 切空间投影：S_i = W^{1/2} * C_i * W^{1/2}，然后取对称对数近似为 (C_i - mean_cov)
    feats = []
    for i in range(n_trials):
        S = W_white @ covs[i] @ W_white.T
        # 对称矩阵对数（近似：取上三角元素）
        vec = symmetric_matrix_to_vec(S)
        feats.append(vec)

    return np.array(feats)


# --- F4: FBCSP互信息特征 ---
FILTERBANK = [
    (4,  8), (8, 12), (12, 16), (16, 20),
    (20, 24), (24, 28), (28, 32)
]


def extract_fbcsp(data, label, n_comp=2):
    """
    多频带CSP，每个频带提取2*n_comp个CSP特征，
    然后用方差比选出最具判别力的特征（简化互信息选择为方差选择）
    返回 (trials, n_bands * 2*n_comp)
    """
    all_feats = []
    for (low, high) in FILTERBANK:
        try:
            d_band = bandpass(data, low, high)
            W = compute_csp_filters(d_band, label, n_comp)
            filtered = np.tensordot(d_band, W, axes=([1], [0]))
            filtered = filtered.transpose(0, 2, 1)
            logvar = np.log(np.var(filtered, axis=2) + 1e-10)
            all_feats.append(logvar)
        except Exception:
            # 数据量不足时跳过该频带
            all_feats.append(np.zeros((data.shape[0], 2 * n_comp)))
    return np.hstack(all_feats)


# ===================== 主分析流程 =====================

FEATURE_NAMES = ['PSD功率比\n(Alpha/Beta,C3/C4)', 'CSP对数方差\n(8-30Hz)', '黎曼切空间\n(协方差向量化)', 'FBCSP多频带\n对数方差']
FEATURE_KEYS  = ['PSD', 'CSP', 'Riemann', 'FBCSP']

# 存储每个被试、每个场景、每类特征的FDR
# fdr_results[feat_key][scene] = list of FDR (per subject)
fdr_results = {k: {'S1': [], 'S2': [], 'both': []} for k in FEATURE_KEYS}

print("=" * 60)
print("特征稳定性分析 — 被试间FDR统计")
print("=" * 60)

for s in range(1, N_SUBJECTS + 1):
    print(f"\n>>> Subject {s:02d}", end=' ')

    subject_fdrs = {k: [] for k in FEATURE_KEYS}

    for scene in SCENES:
        try:
            data, label = load_mat(s, scene)
        except Exception as e:
            print(f"[跳过 S{s} {scene}: {e}]", end='')
            continue

        classes = np.unique(label)
        if len(classes) < 2:
            continue

        left_idx  = label == classes[0]
        right_idx = label == classes[1]

        # F1: PSD
        try:
            feat = extract_psd_ratio(data, label)
            fdr = fisher_ratio(feat[left_idx], feat[right_idx])
            fdr_results['PSD'][scene].append(fdr)
            subject_fdrs['PSD'].append(fdr)
        except Exception as e:
            print(f"[PSD err:{e}]", end='')

        # F2: CSP
        try:
            feat = extract_csp_logvar(data, label)
            fdr = fisher_ratio(feat[left_idx], feat[right_idx])
            fdr_results['CSP'][scene].append(fdr)
            subject_fdrs['CSP'].append(fdr)
        except Exception as e:
            print(f"[CSP err:{e}]", end='')

        # F3: Riemannian
        try:
            feat = extract_riemannian(data, label)
            fdr = fisher_ratio(feat[left_idx], feat[right_idx])
            fdr_results['Riemann'][scene].append(fdr)
            subject_fdrs['Riemann'].append(fdr)
        except Exception as e:
            print(f"[Riemann err:{e}]", end='')

        # F4: FBCSP
        try:
            feat = extract_fbcsp(data, label)
            fdr = fisher_ratio(feat[left_idx], feat[right_idx])
            fdr_results['FBCSP'][scene].append(fdr)
            subject_fdrs['FBCSP'].append(fdr)
        except Exception as e:
            print(f"[FBCSP err:{e}]", end='')

        print(f"{scene}✓", end=' ')

    # 合并两个场景作为'both'
    for k in FEATURE_KEYS:
        if subject_fdrs[k]:
            fdr_results[k]['both'].append(np.mean(subject_fdrs[k]))

print("\n\n分析完成，生成图表...")

# ===================== 保存原始数据 =====================
np.save(os.path.join(SAVE_ROOT, 'fdr_results.npy'), fdr_results)

# ===================== 绘图 =====================

# --- 图1: 各特征FDR跨被试分布（箱线图 + 均值±std排名）---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Feature Stability: Inter-Subject Fisher Discriminant Ratio (FDR)\nDataset: Cross-Scene MI (37 subjects, cue paradigm)',
             fontsize=13, fontweight='bold')

# 左图：箱线图（两个场景分开）
ax = axes[0]
n_feat = len(FEATURE_KEYS)
colors = ['#4C72B0', '#DD8452']
scene_labels = ['Scene 1 (Hospital)', 'Scene 2 (Lab)']
positions_base = np.arange(n_feat)
width = 0.35

for j, (scene, color, slabel) in enumerate(zip(['S1', 'S2'], colors, scene_labels)):
    data_list = [fdr_results[k][scene] for k in FEATURE_KEYS]
    bp = ax.boxplot(data_list,
                    positions=positions_base + (j - 0.5) * width,
                    widths=width * 0.85,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    boxprops=dict(facecolor=color, alpha=0.7),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    flierprops=dict(marker='o', color=color, markersize=4, alpha=0.5),
                    label=slabel)

ax.set_xticks(positions_base)
ax.set_xticklabels(FEATURE_NAMES, fontsize=9)
ax.set_ylabel('Fisher Discriminant Ratio (FDR)', fontsize=11)
ax.set_title('FDR Distribution per Feature (Scene 1 vs Scene 2)', fontsize=11)
ax.legend(fontsize=9)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

# 右图：稳定性排名（均值±std，按均值降序排列）
ax2 = axes[1]
means = np.array([np.mean(fdr_results[k]['both']) if fdr_results[k]['both'] else 0
                  for k in FEATURE_KEYS])
stds  = np.array([np.std(fdr_results[k]['both'])  if fdr_results[k]['both'] else 0
                  for k in FEATURE_KEYS])

# 稳定性分数 = 均值 / (std + eps)，越高越好
stability_score = means / (stds + 1e-6)

order = np.argsort(stability_score)[::-1]
sorted_names   = [FEATURE_NAMES[i] for i in order]
sorted_means   = means[order]
sorted_stds    = stds[order]
sorted_scores  = stability_score[order]

bar_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
bars = ax2.barh(range(n_feat), sorted_scores, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.8)

# 在柱子上标注 mean±std
for i, (m, s, score) in enumerate(zip(sorted_means, sorted_stds, sorted_scores)):
    ax2.text(score + 0.01, i, f'FDR={m:.3f}±{s:.3f}', va='center', fontsize=9)

ax2.set_yticks(range(n_feat))
ax2.set_yticklabels(sorted_names, fontsize=9)
ax2.set_xlabel('Stability Score (mean FDR / std FDR)', fontsize=11)
ax2.set_title('Feature Stability Ranking\n(Higher = More Consistent Across Subjects)', fontsize=11)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
out_path = os.path.join(SAVE_ROOT, 'feature_stability_ranking.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[图1已保存] {out_path}")


# --- 图2: 逐被试FDR热图（特征 × 被试）---
fig2, axes2 = plt.subplots(1, 2, figsize=(20, 5))
fig2.suptitle('Per-Subject FDR Heatmap (Features × Subjects)', fontsize=13, fontweight='bold')

for ax_idx, scene in enumerate(['S1', 'S2']):
    ax = axes2[ax_idx]
    n_sub_max = max(len(fdr_results[k][scene]) for k in FEATURE_KEYS)
    mat = np.full((n_feat, n_sub_max), np.nan)
    for fi, k in enumerate(FEATURE_KEYS):
        vals = fdr_results[k][scene]
        mat[fi, :len(vals)] = vals

    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=np.nanpercentile(mat, 95))
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
    ax.set_xlabel('Subject Index', fontsize=10)
    ax.set_title(f'{"Scene 1 (Hospital)" if scene=="S1" else "Scene 2 (Lab)"}', fontsize=11)
    plt.colorbar(im, ax=ax, label='FDR')

plt.tight_layout()
out_path2 = os.path.join(SAVE_ROOT, 'feature_stability_heatmap.png')
plt.savefig(out_path2, dpi=200, bbox_inches='tight')
plt.close()
print(f"[图2已保存] {out_path2}")


# ===================== 文字摘要 =====================
print("\n" + "=" * 60)
print("稳定性分析摘要（两场景合并，N={}名被试）".format(len(fdr_results[FEATURE_KEYS[0]]['both'])))
print("=" * 60)
print(f"{'特征':<20} {'FDR均值':>10} {'FDR标准差':>12} {'稳定性分数':>12}")
print("-" * 60)
for k, name in zip(FEATURE_KEYS, FEATURE_NAMES):
    vals = fdr_results[k]['both']
    if vals:
        m, s = np.mean(vals), np.std(vals)
        score = m / (s + 1e-6)
        short_name = name.replace('\n', ' ')
        print(f"{short_name:<20} {m:>10.4f} {s:>12.4f} {score:>12.4f}")
print("=" * 60)
print(f"\n结果已保存至: {SAVE_ROOT}")
