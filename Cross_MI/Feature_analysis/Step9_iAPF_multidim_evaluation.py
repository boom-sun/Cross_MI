"""
Step9_iAPF_multidim_evaluation.py

目的：
  1. 估计每名被试的个体化Alpha峰值频率（iAPF）
  2. 对比固定频带 vs iAPF自适应频带特征，以及Step8中其他有潜力的特征
  3. 用三维评价框架综合评估每种特征的"稳定可用性"：
       - Dim1 CSCDC  : 跨被试类方向一致性（稳定性）
       - Dim2 FDR    : 被试内Fisher判别比（判别力）
       - Dim3 SceneR : 同被试跨场景delta相关性（可靠性）
  4. 输出雷达图 + 综合分数排名图 + iAPF分布图

数据：4_跨场景因素研究v2 / cue范式，37名被试
"""

import os
import numpy as np
import hdf5storage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.covariance import ledoit_wolf
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ===================== 参数 =====================
DATA_ROOT = r'E:\Datasets\4_跨场景因素研究v2' + r'\跨场景因素研究v2处理后数据'
SAVE_ROOT = r'E:\Datasets\4_跨场景因素研究v2' + r'\跨场景因素研究v2画图数据\stability'
os.makedirs(SAVE_ROOT, exist_ok=True)

SRATE    = 250
START_PT = 2 * SRATE
END_PT   = 6 * SRATE
N_SUB    = 37
CH_IDX   = [i-1 for i in [16,17,18,19,20,21,22, 25,26,27,28,29,30,31, 34,35,36,37,38,39,40]]
C3_L     = CH_IDX.index(26)
C4_L     = CH_IDX.index(30)

# Laplacian邻居（局部索引）
NEIGH_C3 = [CH_IDX.index(i) for i in [17,27,24,35] if i in CH_IDX]
NEIGH_C4 = [CH_IDX.index(i) for i in [19,29,28,37] if i in CH_IDX]

# ===================== 数据加载 =====================
def load_scene(subject, scene):
    pfx  = f"1S0{subject}" if subject < 10 else f"1S{subject}"
    path = os.path.join(DATA_ROOT, f"{pfx}{scene}_cue.mat")
    d    = hdf5storage.loadmat(path, mat_dtype=True)
    data = d['data'][:, START_PT:END_PT, :].transpose(2,0,1)
    data = data[:, CH_IDX, :]
    label = np.squeeze(d['label']).astype(int)
    return data, label

# ===================== 基础工具 =====================
def bandpass(data, low, high, order=4):
    nyq = SRATE / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=2)

def band_power(data, low, high):
    return np.mean(bandpass(data, low, high)**2, axis=2)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

def normalize_feat(feat):
    mu  = feat.mean(axis=0)
    std = feat.std(axis=0) + 1e-10
    return (feat - mu) / std

def get_delta(feat, label):
    classes = np.unique(label)
    f = normalize_feat(feat)
    return f[label==classes[0]].mean(0) - f[label==classes[1]].mean(0)

def fisher_ratio(feat, label):
    classes = np.unique(label)
    f = normalize_feat(feat)
    c0, c1 = f[label==classes[0]], f[label==classes[1]]
    mu0, mu1 = c0.mean(0), c1.mean(0)
    var0 = c0.var(0) + 1e-10
    var1 = c1.var(0) + 1e-10
    return float(np.mean((mu0-mu1)**2 / (var0+var1)))

# ===================== iAPF 估计 =====================
def estimate_iapf(data):
    """
    输入 data: (trials, channels, time)
    在C3/C4上用Welch PSD估计alpha峰值频率（搜索范围7-14Hz）
    返回 iAPF（Hz）
    """
    x = data[:, [C3_L, C4_L], :].reshape(-1, data.shape[2])
    freqs, psd = welch(x, fs=SRATE, nperseg=256, axis=1)
    psd_mean = psd.mean(axis=0)
    alpha_mask = (freqs >= 7) & (freqs <= 14)
    peak_freq = freqs[alpha_mask][np.argmax(psd_mean[alpha_mask])]
    return float(peak_freq)

# ===================== 特征提取函数 =====================

def feat_fixed_alpha(data, label, lo=8, hi=13):
    p = band_power(data, lo, hi)
    return np.log(p[:, [C3_L, C4_L]] + 1e-10)

def feat_fixed_mu(data, label, lo=8, hi=12):
    p = band_power(data, lo, hi)
    return np.log(p[:, [C3_L, C4_L]] + 1e-10)

def feat_fixed_beta(data, label, lo=13, hi=30):
    p = band_power(data, lo, hi)
    return np.log(p[:, [C3_L, C4_L]] + 1e-10)

def feat_iapf_alpha(data, label, iapf, half_bw=2.0):
    lo, hi = max(iapf - half_bw, 4.0), min(iapf + half_bw, 20.0)
    p = band_power(data, lo, hi)
    return np.log(p[:, [C3_L, C4_L]] + 1e-10)

def feat_iapf_beta(data, label, iapf, offset=4.0, bw=8.0):
    """Beta定义为iAPF+offset ~ iAPF+offset+bw（beta与alpha位置相关）"""
    lo, hi = iapf + offset, iapf + offset + bw
    hi = min(hi, 40.0)
    p = band_power(data, lo, hi)
    return np.log(p[:, [C3_L, C4_L]] + 1e-10)

def feat_fixed_laplacian(data, label, lo=8, hi=30):
    d = bandpass(data, lo, hi)
    c3 = d[:, C3_L, :] - d[:, NEIGH_C3, :].mean(axis=1)
    c4 = d[:, C4_L, :] - d[:, NEIGH_C4, :].mean(axis=1)
    return np.log(np.column_stack([
        np.mean(c3**2, axis=1),
        np.mean(c4**2, axis=1)
    ]) + 1e-10)

def feat_iapf_laplacian(data, label, iapf, half_bw=2.0):
    lo, hi = max(iapf - half_bw, 4.0), min(iapf + half_bw, 20.0)
    d = bandpass(data, lo, hi)
    c3 = d[:, C3_L, :] - d[:, NEIGH_C3, :].mean(axis=1)
    c4 = d[:, C4_L, :] - d[:, NEIGH_C4, :].mean(axis=1)
    return np.log(np.column_stack([
        np.mean(c3**2, axis=1),
        np.mean(c4**2, axis=1)
    ]) + 1e-10)

def feat_cov_diag(data, label, lo=8, hi=30):
    d = bandpass(data, lo, hi)
    feats = [np.log(np.var(d[i], axis=1) + 1e-10) for i in range(d.shape[0])]
    return np.array(feats)

def feat_stft_power_fn(data, label):
    from scipy.signal import stft as _stft
    feats = []
    for ch_l in [C3_L, C4_L]:
        x = data[:, ch_l, :]
        ap, bp = [], []
        for trial in range(x.shape[0]):
            f, _, Z = _stft(x[trial], fs=SRATE, nperseg=64, noverlap=32)
            psd = np.abs(Z)**2
            ap.append(np.log(psd[(f>=8)&(f<=13)].mean() + 1e-10))
            bp.append(np.log(psd[(f>=13)&(f<=30)].mean() + 1e-10))
        feats += [ap, bp]
    return np.column_stack(feats)

# ===================== 特征配置表 =====================
# 每条记录: (显示名, 组名, 提取函数工厂)
# 工厂接收 (data, label, iapf) -> feat矩阵

def make_fixed(fn, **kwargs):
    return lambda data, label, iapf: fn(data, label, **kwargs)

FEAT_CFG = [
    # ---- 固定频带基线 ----
    ('Alpha Fixed\n(8-13Hz)',       'Fixed Spectral',
     make_fixed(feat_fixed_alpha)),
    ('Mu Fixed\n(8-12Hz)',          'Fixed Spectral',
     make_fixed(feat_fixed_mu)),
    ('Beta Fixed\n(13-30Hz)',       'Fixed Spectral',
     make_fixed(feat_fixed_beta)),
    ('Laplacian Fixed\n(8-30Hz)',   'Fixed Spatial',
     make_fixed(feat_fixed_laplacian)),
    # ---- iAPF自适应 ----
    ('Alpha iAPF\n(iAPF±2Hz)',      'iAPF Adaptive',
     lambda d,l,f: feat_iapf_alpha(d,l,f, half_bw=2.0)),
    ('Alpha iAPF\n(iAPF±1.5Hz)',    'iAPF Adaptive',
     lambda d,l,f: feat_iapf_alpha(d,l,f, half_bw=1.5)),
    ('Beta iAPF\n(iAPF+4~+12Hz)',   'iAPF Adaptive',
     lambda d,l,f: feat_iapf_beta(d,l,f)),
    ('Laplacian iAPF\n(iAPF±2Hz)',  'iAPF Adaptive',
     lambda d,l,f: feat_iapf_laplacian(d,l,f, half_bw=2.0)),
    # ---- 参照特征（Step8表现较好）----
    ('Cov Diag Log\n(8-30Hz)',      'Reference',
     make_fixed(feat_cov_diag)),
    ('STFT Power\n(Alpha+Beta)',    'Reference',
     make_fixed(feat_stft_power_fn)),
]

CAT_COLORS = {
    'Fixed Spectral': '#4C72B0',
    'Fixed Spatial':  '#DD8452',
    'iAPF Adaptive':  '#2ca02c',
    'Reference':      '#9467bd',
}

# ===================== 主分析 =====================
print("=" * 65)
print("Step9: iAPF自适应特征 vs 固定频带 — 三维评价")
print("=" * 65)

n_feat = len(FEAT_CFG)

# 存储结构
all_deltas_s1  = [[] for _ in range(n_feat)]   # dim1/dim3用
all_deltas_s2  = [[] for _ in range(n_feat)]
all_fdr        = [[] for _ in range(n_feat)]   # dim2
iapf_list      = []   # 记录每人iAPF
valid_subs     = []

for s in range(1, N_SUB+1):
    try:
        d1, l1 = load_scene(s, 'S1')
        d2, l2 = load_scene(s, 'S2')
    except Exception as e:
        print(f"  Sub{s:02d}: skip ({e})")
        continue
    if len(np.unique(l1)) < 2 or len(np.unique(l2)) < 2:
        continue

    # 用两场景合并数据估计iAPF（更稳定）
    data_all  = np.concatenate([d1, d2], axis=0)
    label_all = np.concatenate([l1, l2], axis=0)
    iapf = estimate_iapf(data_all)
    iapf_list.append(iapf)
    valid_subs.append(s)

    print(f"  Sub{s:02d} iAPF={iapf:.1f}Hz  ", end='', flush=True)

    for fi, (name, cat, fn) in enumerate(FEAT_CFG):
        try:
            # --- Dim2: FDR (两场景合并) ---
            feat_all = fn(data_all, label_all, iapf)
            fdr = fisher_ratio(feat_all, label_all)
            all_fdr[fi].append(fdr)

            # --- Dim1/Dim3: delta向量（分场景）---
            f1 = fn(d1, l1, iapf)
            f2 = fn(d2, l2, iapf)
            delta1 = get_delta(f1, l1)
            delta2 = get_delta(f2, l2)
            all_deltas_s1[fi].append(delta1)
            all_deltas_s2[fi].append(delta2)
            print('.', end='', flush=True)
        except Exception as e:
            all_fdr[fi].append(np.nan)
            all_deltas_s1[fi].append(None)
            all_deltas_s2[fi].append(None)
            print('x', end='', flush=True)
    print()

N = len(valid_subs)
print(f"\n有效被试: {N}名，iAPF范围: {min(iapf_list):.1f}~{max(iapf_list):.1f}Hz")

# ===================== 计算三个维度 =====================
scores = []
for fi in range(n_feat):
    name, cat, _ = FEAT_CFG[fi]

    # Dim1: CSCDC（用S1+S2合并的delta）
    deltas_both = []
    for i in range(N):
        d1i = all_deltas_s1[fi][i]
        d2i = all_deltas_s2[fi][i]
        if d1i is not None and d2i is not None:
            deltas_both.append((d1i + d2i) / 2.0)
        elif d1i is not None:
            deltas_both.append(d1i)
        elif d2i is not None:
            deltas_both.append(d2i)

    cscdc_vals = []
    for i, j in combinations(range(len(deltas_both)), 2):
        s = cosine_sim(deltas_both[i], deltas_both[j])
        if not np.isnan(s):
            cscdc_vals.append(s)
    dim1 = float(np.mean(cscdc_vals)) if cscdc_vals else np.nan

    # Dim2: 平均FDR
    fdrs = [v for v in all_fdr[fi] if not np.isnan(v)]
    dim2 = float(np.mean(fdrs)) if fdrs else np.nan

    # Dim3: 跨场景delta相关性（同被试S1 vs S2）
    scene_sims = []
    for i in range(N):
        d1i = all_deltas_s1[fi][i]
        d2i = all_deltas_s2[fi][i]
        if d1i is not None and d2i is not None:
            s = cosine_sim(d1i, d2i)
            if not np.isnan(s):
                scene_sims.append(s)
    dim3 = float(np.mean(scene_sims)) if scene_sims else np.nan

    scores.append({
        'name': name, 'cat': cat,
        'dim1': dim1, 'dim2': dim2, 'dim3': dim3,
    })

# 打印结果
print(f"\n{'特征':<28} {'CSCDC':>8} {'FDR':>8} {'SceneR':>8}")
print("-" * 55)
for r in scores:
    n = r['name'].replace('\n',' ')
    print(f"  {n:<26} {r['dim1']:>8.3f} {r['dim2']:>8.3f} {r['dim3']:>8.3f}")

# ===================== 归一化 + 综合分数 =====================
for key in ['dim1','dim2','dim3']:
    vals = np.array([r[key] for r in scores])
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    rng = vmax - vmin + 1e-10
    for r in scores:
        r[f'{key}_norm'] = (r[key] - vmin) / rng

for r in scores:
    r['composite'] = (r['dim1_norm'] + r['dim2_norm'] + r['dim3_norm']) / 3.0

scores.sort(key=lambda x: x['composite'], reverse=True)
np.save(os.path.join(SAVE_ROOT, 'step9_scores.npy'), scores)

# ===================== 绘图 =====================

# --- 图1: 综合分数排名（含三维分解堆叠条形图）---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Multi-Dimensional Feature Evaluation: Stability vs Usability\n'
             f'N={N} subjects | Dim1=CSCDC(cross-subj consistency) | '
             'Dim2=FDR(discriminability) | Dim3=SceneR(reliability)',
             fontsize=11, fontweight='bold')

# 左图：综合分数横条
ax = axes[0]
names      = [r['name'].replace('\n',' ') for r in scores]
composites = [r['composite'] for r in scores]
bar_colors = [CAT_COLORS[r['cat']] for r in scores]
ax.barh(range(len(scores)), composites, color=bar_colors, alpha=0.85,
        edgecolor='white', linewidth=0.7)
for i, (r, c) in enumerate(zip(scores, composites)):
    ax.text(c + 0.005, i,
            f"D1={r['dim1']:.3f}  D2={r['dim2']:.3f}  D3={r['dim3']:.3f}",
            va='center', fontsize=8)
ax.set_yticks(range(len(scores)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Composite Score (normalized mean of 3 dims)', fontsize=10)
ax.set_title('Overall Ranking', fontsize=11)
ax.grid(axis='x', alpha=0.3)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=c, label=k, alpha=0.85) for k,c in CAT_COLORS.items()]
ax.legend(handles=legend_handles, fontsize=8, loc='lower right', title='Category')

# 右图：三维雷达图（显示Top4 + 两个基线）
ax2 = axes[1]
# 选取展示：Top2 iAPF + Alpha Fixed + Laplacian Fixed + Cov Diag
display_names = []
display_list  = []
# 先加排名前4
for r in scores[:4]:
    display_list.append(r)
    display_names.append(r['name'].replace('\n',' '))
# 再加固定Alpha基线（若不在前4）
for r in scores:
    if 'Alpha Fixed' in r['name'] and r not in display_list:
        display_list.append(r)
        display_names.append(r['name'].replace('\n',' '))
        break

angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
angles += angles[:1]
labels_radar = ['CSCDC\n(Cross-Subj\nConsistency)',
                'FDR\n(Discriminability)',
                'SceneR\n(Reliability)']

ax2.remove()
ax2 = fig.add_subplot(1, 2, 2, polar=True)
ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels_radar, fontsize=9)
ax2.set_ylim(0, 1)
ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
ax2.set_yticklabels(['0.25','0.50','0.75','1.00'], fontsize=7)
ax2.set_title('Radar: Top Features vs Baseline\n(normalized scores)', fontsize=10, pad=15)

radar_colors = ['#2ca02c','#d62728','#ff7f0e','#9467bd','#4C72B0']
for idx, (r, rcolor) in enumerate(zip(display_list, radar_colors)):
    vals = [r['dim1_norm'], r['dim2_norm'], r['dim3_norm']]
    vals += vals[:1]
    ax2.plot(angles, vals, 'o-', linewidth=2, color=rcolor,
             label=r['name'].replace('\n',' '), alpha=0.85)
    ax2.fill(angles, vals, alpha=0.08, color=rcolor)

ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)

plt.tight_layout()
out1 = os.path.join(SAVE_ROOT, 'step9_multidim_ranking.png')
plt.savefig(out1, dpi=200, bbox_inches='tight')
plt.close()
print(f"\n[图1已保存] {out1}")

# --- 图2: iAPF分布图 ---
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle(f'Individual Alpha Peak Frequency (iAPF) Distribution\n'
              f'N={N} subjects, estimated from C3+C4 across both scenes',
              fontsize=11, fontweight='bold')

ax = axes2[0]
ax.hist(iapf_list, bins=14, color='#2ca02c', edgecolor='white', alpha=0.85)
ax.axvline(np.mean(iapf_list), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean={np.mean(iapf_list):.1f}Hz')
ax.axvspan(8, 13, alpha=0.12, color='blue', label='Fixed band (8-13Hz)')
ax.set_xlabel('iAPF (Hz)', fontsize=10)
ax.set_ylabel('Subject Count', fontsize=10)
ax.set_title('iAPF Histogram', fontsize=10)
ax.legend(fontsize=9)

ax2b = axes2[1]
sorted_iapf = sorted(enumerate(iapf_list), key=lambda x: x[1])
sidx, svals = zip(*sorted_iapf)
ax2b.bar(range(N), svals, color='#2ca02c', alpha=0.8, edgecolor='white')
ax2b.axhline(8,  color='blue',  linestyle='--', linewidth=1, label='Fixed band lower (8Hz)')
ax2b.axhline(13, color='navy',  linestyle='--', linewidth=1, label='Fixed band upper (13Hz)')
ax2b.set_xlabel('Subject (sorted by iAPF)', fontsize=10)
ax2b.set_ylabel('iAPF (Hz)', fontsize=10)
ax2b.set_title('Per-Subject iAPF (sorted)', fontsize=10)
ax2b.legend(fontsize=9)
ax2b.set_ylim(5, 16)

plt.tight_layout()
out2 = os.path.join(SAVE_ROOT, 'step9_iapf_distribution.png')
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f"[图2已保存] {out2}")

# --- 图3: Fixed vs iAPF 三维对比散点图 ---
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle('Fixed Band vs iAPF Adaptive: Dimension-by-Dimension Comparison',
              fontsize=11, fontweight='bold')

pairs = [
    ('Alpha Fixed\n(8-13Hz)', 'Alpha iAPF\n(iAPF±2Hz)'),
    ('Mu Fixed\n(8-12Hz)',    'Alpha iAPF\n(iAPF±1.5Hz)'),
    ('Laplacian Fixed\n(8-30Hz)', 'Laplacian iAPF\n(iAPF±2Hz)'),
]
dim_labels = ['CSCDC (Cross-Subj Consistency)', 'FDR (Discriminability)', 'SceneR (Reliability)']

score_dict = {r['name']: r for r in scores}

for ai, (dim_key, dim_label) in enumerate(zip(['dim1','dim2','dim3'], dim_labels)):
    ax = axes3[ai]
    for pair_color, (fixed_name, iapf_name) in zip(['#4C72B0','#DD8452','#55A868'], pairs):
        rf = score_dict.get(fixed_name)
        ri = score_dict.get(iapf_name)
        if rf and ri:
            fv = rf[dim_key]
            iv = ri[dim_key]
            ax.annotate('', xy=(iv, 0.5+pairs.index((fixed_name,iapf_name))*0.15),
                        xytext=(fv, 0.5+pairs.index((fixed_name,iapf_name))*0.15),
                        arrowprops=dict(arrowstyle='->', color=pair_color, lw=2))
            ax.scatter([fv, iv],
                       [0.5+pairs.index((fixed_name,iapf_name))*0.15]*2,
                       c=[pair_color], s=80, zorder=5)
            ax.text(fv - 0.002, 0.5+pairs.index((fixed_name,iapf_name))*0.15 + 0.02,
                    fixed_name.split('\n')[0], fontsize=7, color=pair_color)
            ax.text(iv + 0.002, 0.5+pairs.index((fixed_name,iapf_name))*0.15 + 0.02,
                    iapf_name.split('\n')[0]+' iAPF', fontsize=7, color=pair_color)
    ax.set_xlabel(dim_label, fontsize=9)
    ax.set_yticks([])
    ax.set_title(f'Dim: {dim_label.split("(")[0].strip()}', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(-0.05, max(
        max(score_dict[n][dim_key] for n,_ in pairs if n in score_dict),
        max(score_dict[n][dim_key] for _,n in pairs if n in score_dict)
    ) * 1.25)

plt.tight_layout()
out3 = os.path.join(SAVE_ROOT, 'step9_fixed_vs_iapf_comparison.png')
plt.savefig(out3, dpi=200, bbox_inches='tight')
plt.close()
print(f"[图3已保存] {out3}")

# ===================== 文字摘要 =====================
print("\n" + "=" * 65)
print("综合分数排名（CSCDC + FDR + SceneR 归一化均值）")
print("=" * 65)
print(f"{'排名':<4} {'特征':<30} {'类别':<18} {'CSCDC':>7} {'FDR':>7} {'SceneR':>8} {'综合':>7}")
print("-" * 65)
for rank, r in enumerate(scores, 1):
    n = r['name'].replace('\n',' ')
    print(f"{rank:<4} {n:<30} {r['cat']:<18} "
          f"{r['dim1']:>7.3f} {r['dim2']:>7.3f} {r['dim3']:>8.3f} {r['composite']:>7.3f}")
print("=" * 65)
