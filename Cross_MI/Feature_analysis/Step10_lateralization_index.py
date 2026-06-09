"""
Step10_lateralization_index.py

目的：
  评估侧化指数（Lateralization Index, LI）的跨被试通用性，
  与Step9所有特征在相同三维框架下进行对比。

LI的神经生理学动机：
  - MI的核心现象是对侧ERD（左手MI→右脑C4抑制，右手MI→左脑C3抑制）
  - LI = (P_C3 - P_C4) / (P_C3 + P_C4) 自归一化，消除被试间绝对幅度差异
  - LI_Lap：加Laplacian空间滤波增强空间特异性

评估的LI变体（以频带为维度构成向量，支持CSCDC计算）：
  LI_Alpha     : Alpha频段 (8-13Hz)，原始C3/C4
  LI_Mu        : Mu频段 (8-12Hz)，原始C3/C4
  LI_Beta      : Beta频段 (13-30Hz)，原始C3/C4
  LI_AlphaBeta : Alpha+Beta多频段LI向量 (7维)
  LI_Lap_Alpha : Laplacian + Alpha
  LI_Lap_Mu    : Laplacian + Mu
  LI_Lap_Beta  : Laplacian + Beta
  LI_Lap_Multi : Laplacian + 多频段LI向量 (7维，核心候选)

三维评价指标（与Step9完全一致）：
  Dim1 CSCDC  : 跨被试类方向一致性
  Dim2 FDR    : 被试内Fisher判别比
  Dim3 SceneR : 同被试跨场景delta相关性
"""

import os
import numpy as np
import hdf5storage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
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
CH_IDX   = [i-1 for i in [16,17,18,19,20,21,22,
                            25,26,27,28,29,30,31,
                            34,35,36,37,38,39,40]]
C3_L = CH_IDX.index(26)
C4_L = CH_IDX.index(30)
NEIGH_C3 = [CH_IDX.index(i) for i in [17,27,24,35] if i in CH_IDX]
NEIGH_C4 = [CH_IDX.index(i) for i in [19,29,28,37] if i in CH_IDX]

# 多频段滤波器组（用于LI向量）
FBANK = [(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]

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

def laplacian(data, low, high):
    d = bandpass(data, low, high)
    c3 = d[:, C3_L, :] - d[:, NEIGH_C3, :].mean(axis=1)
    c4 = d[:, C4_L, :] - d[:, NEIGH_C4, :].mean(axis=1)
    p_c3 = np.mean(c3**2, axis=1)
    p_c4 = np.mean(c4**2, axis=1)
    return p_c3, p_c4

def li_ratio(p_c3, p_c4):
    """LI = (C3 - C4) / (C3 + C4)，逐trial计算"""
    return (p_c3 - p_c4) / (p_c3 + p_c4 + 1e-10)

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
    var = c0.var(0) + c1.var(0) + 1e-10
    return float(np.mean((mu0-mu1)**2 / var))

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

# ===================== LI 特征提取 =====================

def feat_li_alpha(data, label):
    p3 = band_power(data, 8, 13)[:, C3_L]
    p4 = band_power(data, 8, 13)[:, C4_L]
    return li_ratio(p3, p4).reshape(-1, 1)

def feat_li_mu(data, label):
    p3 = band_power(data, 8, 12)[:, C3_L]
    p4 = band_power(data, 8, 12)[:, C4_L]
    return li_ratio(p3, p4).reshape(-1, 1)

def feat_li_beta(data, label):
    p3 = band_power(data, 13, 30)[:, C3_L]
    p4 = band_power(data, 13, 30)[:, C4_L]
    return li_ratio(p3, p4).reshape(-1, 1)

def feat_li_multibands(data, label):
    """7维LI向量：每个频段一个LI值"""
    lis = []
    for (lo, hi) in FBANK:
        p3 = band_power(data, lo, hi)[:, C3_L]
        p4 = band_power(data, lo, hi)[:, C4_L]
        lis.append(li_ratio(p3, p4))
    return np.column_stack(lis)   # (trials, 7)

def feat_li_lap_alpha(data, label):
    p3, p4 = laplacian(data, 8, 13)
    return li_ratio(p3, p4).reshape(-1, 1)

def feat_li_lap_mu(data, label):
    p3, p4 = laplacian(data, 8, 12)
    return li_ratio(p3, p4).reshape(-1, 1)

def feat_li_lap_beta(data, label):
    p3, p4 = laplacian(data, 13, 30)
    return li_ratio(p3, p4).reshape(-1, 1)

def feat_li_lap_multibands(data, label):
    """Laplacian + 7维LI向量（核心候选特征）"""
    lis = []
    for (lo, hi) in FBANK:
        p3, p4 = laplacian(data, lo, hi)
        lis.append(li_ratio(p3, p4))
    return np.column_stack(lis)   # (trials, 7)

# ===================== Step9参照特征（用于对比）=====================
def feat_alpha_fixed(data, label):
    p = band_power(data, 8, 13)
    return np.log(p[:, [C3_L, C4_L]] + 1e-10)

def feat_mu_fixed(data, label):
    p = band_power(data, 8, 12)
    return np.log(p[:, [C3_L, C4_L]] + 1e-10)

def feat_laplacian_fixed(data, label):
    p3, p4 = laplacian(data, 8, 30)
    return np.log(np.column_stack([p3, p4]) + 1e-10)

def feat_cov_diag(data, label):
    from sklearn.covariance import ledoit_wolf
    d = bandpass(data, 8, 30)
    return np.array([np.log(np.var(d[i], axis=1) + 1e-10) for i in range(d.shape[0])])

# ===================== 特征配置 =====================
CAT_COLORS = {
    'LI (Raw C3/C4)':      '#2ca02c',
    'LI (Laplacian)':      '#d62728',
    'Reference (Step9)':   '#7f7f7f',
}

FEAT_CFG = [
    # --- LI变体（原始C3/C4）---
    ('LI Alpha\n(8-13Hz)',        'LI (Raw C3/C4)',    feat_li_alpha),
    ('LI Mu\n(8-12Hz)',           'LI (Raw C3/C4)',    feat_li_mu),
    ('LI Beta\n(13-30Hz)',        'LI (Raw C3/C4)',    feat_li_beta),
    ('LI Multi-band\n(7-dim)',    'LI (Raw C3/C4)',    feat_li_multibands),
    # --- LI变体（Laplacian）---
    ('LI Lap Alpha\n(8-13Hz)',    'LI (Laplacian)',    feat_li_lap_alpha),
    ('LI Lap Mu\n(8-12Hz)',       'LI (Laplacian)',    feat_li_lap_mu),
    ('LI Lap Beta\n(13-30Hz)',    'LI (Laplacian)',    feat_li_lap_beta),
    ('LI Lap Multi\n(7-dim)',     'LI (Laplacian)',    feat_li_lap_multibands),
    # --- Step9参照特征 ---
    ('Alpha Fixed\n(8-13Hz)',     'Reference (Step9)', feat_alpha_fixed),
    ('Mu Fixed\n(8-12Hz)',        'Reference (Step9)', feat_mu_fixed),
    ('Laplacian Fixed\n(8-30Hz)', 'Reference (Step9)', feat_laplacian_fixed),
    ('Cov Diag Log\n(8-30Hz)',    'Reference (Step9)', feat_cov_diag),
]

# ===================== 主分析 =====================
print("=" * 65)
print("Step10: 侧化指数（LI）三维评价 vs Step9参照特征")
print("=" * 65)

n_feat = len(FEAT_CFG)
all_deltas_s1 = [[] for _ in range(n_feat)]
all_deltas_s2 = [[] for _ in range(n_feat)]
all_fdr       = [[] for _ in range(n_feat)]
valid_subs    = []

for s in range(1, N_SUB+1):
    try:
        d1, l1 = load_scene(s, 'S1')
        d2, l2 = load_scene(s, 'S2')
    except Exception as e:
        print(f"  Sub{s:02d}: skip ({e})")
        continue
    if len(np.unique(l1)) < 2 or len(np.unique(l2)) < 2:
        continue

    data_all  = np.concatenate([d1, d2], axis=0)
    label_all = np.concatenate([l1, l2], axis=0)
    valid_subs.append(s)

    print(f"  Sub{s:02d}: ", end='', flush=True)

    for fi, (name, cat, fn) in enumerate(FEAT_CFG):
        try:
            feat_all = fn(data_all, label_all)
            all_fdr[fi].append(fisher_ratio(feat_all, label_all))

            f1 = fn(d1, l1)
            f2 = fn(d2, l2)
            all_deltas_s1[fi].append(get_delta(f1, l1))
            all_deltas_s2[fi].append(get_delta(f2, l2))
            print('.', end='', flush=True)
        except Exception as e:
            all_fdr[fi].append(np.nan)
            all_deltas_s1[fi].append(None)
            all_deltas_s2[fi].append(None)
            print('x', end='', flush=True)
    print()

N = len(valid_subs)
print(f"\n有效被试: {N}名")

# ===================== 三维指标计算 =====================
scores = []
for fi, (name, cat, _) in enumerate(FEAT_CFG):

    # Dim1: CSCDC
    deltas = []
    for i in range(N):
        d1i, d2i = all_deltas_s1[fi][i], all_deltas_s2[fi][i]
        if d1i is not None and d2i is not None:
            deltas.append((d1i + d2i) / 2.0)
        elif d1i is not None:
            deltas.append(d1i)
        elif d2i is not None:
            deltas.append(d2i)

    cscdc_vals = [cosine_sim(deltas[i], deltas[j])
                  for i,j in combinations(range(len(deltas)), 2)]
    cscdc_vals = [v for v in cscdc_vals if not np.isnan(v)]
    dim1 = float(np.mean(cscdc_vals)) if cscdc_vals else np.nan

    # Dim2: FDR
    fdrs = [v for v in all_fdr[fi] if not np.isnan(v)]
    dim2 = float(np.mean(fdrs)) if fdrs else np.nan

    # Dim3: SceneR
    scene_sims = []
    for i in range(N):
        d1i, d2i = all_deltas_s1[fi][i], all_deltas_s2[fi][i]
        if d1i is not None and d2i is not None:
            v = cosine_sim(d1i, d2i)
            if not np.isnan(v):
                scene_sims.append(v)
    dim3 = float(np.mean(scene_sims)) if scene_sims else np.nan

    scores.append({'name': name, 'cat': cat,
                   'dim1': dim1, 'dim2': dim2, 'dim3': dim3})

# 归一化 + 综合分数
for key in ['dim1','dim2','dim3']:
    vals = np.array([r[key] for r in scores])
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    rng = vmax - vmin + 1e-10
    for r in scores:
        r[f'{key}_norm'] = (r[key] - vmin) / rng

for r in scores:
    r['composite'] = (r['dim1_norm'] + r['dim2_norm'] + r['dim3_norm']) / 3.0

scores_sorted = sorted(scores, key=lambda x: x['composite'], reverse=True)
np.save(os.path.join(SAVE_ROOT, 'step10_scores.npy'), scores_sorted)

# 打印
print(f"\n{'排名':<4} {'特征':<28} {'类别':<22} {'CSCDC':>7} {'FDR':>7} {'SceneR':>8} {'综合':>7}")
print("-" * 80)
for rank, r in enumerate(scores_sorted, 1):
    n = r['name'].replace('\n',' ')
    mark = ' <-- LI' if 'LI' in r['cat'] else ''
    print(f"{rank:<4} {n:<28} {r['cat']:<22} "
          f"{r['dim1']:>7.3f} {r['dim2']:>7.3f} {r['dim3']:>8.3f} "
          f"{r['composite']:>7.3f}{mark}")

# ===================== 绘图 =====================

# --- 图1: 综合排名 + 三维原始值条形图 ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Lateralization Index vs Reference Features: Multi-Dimensional Evaluation\n'
             f'N={N} subjects | CSCDC=cross-subj consistency | '
             'FDR=discriminability | SceneR=session reliability',
             fontsize=11, fontweight='bold')

# 左图：综合分数
ax = axes[0]
names      = [r['name'].replace('\n',' ') for r in scores_sorted]
composites = [r['composite'] for r in scores_sorted]
bar_colors = [CAT_COLORS[r['cat']] for r in scores_sorted]
bars = ax.barh(range(len(scores_sorted)), composites,
               color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.7)
for i, r in enumerate(scores_sorted):
    ax.text(r['composite'] + 0.005, i,
            f"D1={r['dim1']:.3f} D2={r['dim2']:.3f} D3={r['dim3']:.3f}",
            va='center', fontsize=8)
ax.set_yticks(range(len(scores_sorted)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Composite Score', fontsize=10)
ax.set_title('Overall Ranking', fontsize=11)
ax.grid(axis='x', alpha=0.3)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=c, label=k, alpha=0.85) for k,c in CAT_COLORS.items()]
ax.legend(handles=legend_handles, fontsize=9, loc='lower right')

# 右图：雷达图（Top LI候选 vs Best Reference）
ax2 = axes[1]
# 选择：LI最优 + LI_Lap最优 + Reference最优2个
li_best = next((r for r in scores_sorted if 'LI (Raw' in r['cat']), None)
li_lap_best = next((r for r in scores_sorted if 'LI (Lap' in r['cat']), None)
ref_list = [r for r in scores_sorted if 'Reference' in r['cat']][:2]
display_list = [r for r in [li_best, li_lap_best] + ref_list if r is not None]

angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
angles += angles[:1]
radar_labels = ['CSCDC\n(Cross-Subj\nConsistency)',
                'FDR\n(Discriminability)',
                'SceneR\n(Reliability)']
radar_colors = ['#d62728', '#2ca02c', '#7f7f7f', '#aec7e8']

ax2.remove()
ax2 = fig.add_subplot(1, 2, 2, polar=True)
ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(radar_labels, fontsize=9)
ax2.set_ylim(0, 1)
ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
ax2.set_yticklabels(['0.25','0.50','0.75','1.00'], fontsize=7)
ax2.set_title('Radar: Best LI vs Best Reference\n(normalized scores)', fontsize=10, pad=15)

for r, rcolor in zip(display_list, radar_colors):
    vals = [r['dim1_norm'], r['dim2_norm'], r['dim3_norm']] + [r['dim1_norm']]
    ax2.plot(angles, vals, 'o-', linewidth=2, color=rcolor,
             label=r['name'].replace('\n',' '), alpha=0.9)
    ax2.fill(angles, vals, alpha=0.08, color=rcolor)
ax2.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=8)

plt.tight_layout()
out1 = os.path.join(SAVE_ROOT, 'step10_li_ranking.png')
plt.savefig(out1, dpi=200, bbox_inches='tight')
plt.close()
print(f"\n[图1已保存] {out1}")

# --- 图2: LI三维对比折线图（LI变体对比，突出Lap和多频段的贡献）---
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('LI Variants: Effect of Laplacian Filtering and Frequency Band\n'
              'Raw vs Laplacian | Single-band vs Multi-band',
              fontsize=11, fontweight='bold')

dim_labels = ['CSCDC (Cross-Subj Consistency)',
              'FDR (Discriminability)',
              'SceneR (Session Reliability)']
dim_keys   = ['dim1', 'dim2', 'dim3']

# 按频带分组对比
band_groups = [
    ('Alpha (8-13Hz)', 'LI Alpha\n(8-13Hz)',     'LI Lap Alpha\n(8-13Hz)'),
    ('Mu (8-12Hz)',    'LI Mu\n(8-12Hz)',         'LI Lap Mu\n(8-12Hz)'),
    ('Beta (13-30Hz)', 'LI Beta\n(13-30Hz)',      'LI Lap Beta\n(13-30Hz)'),
    ('Multi-band',     'LI Multi-band\n(7-dim)',  'LI Lap Multi\n(7-dim)'),
]
score_dict = {r['name']: r for r in scores}
x = np.arange(len(band_groups))
bw = 0.35

for ai, (dk, dl) in enumerate(zip(dim_keys, dim_labels)):
    ax = axes2[ai]
    raw_vals = []
    lap_vals = []
    for (_, raw_name, lap_name) in band_groups:
        rv = score_dict.get(raw_name, {}).get(dk, np.nan)
        lv = score_dict.get(lap_name, {}).get(dk, np.nan)
        raw_vals.append(rv)
        lap_vals.append(lv)

    ax.bar(x - bw/2, raw_vals, width=bw, label='LI (Raw C3/C4)',
           color='#2ca02c', alpha=0.75, edgecolor='white')
    ax.bar(x + bw/2, lap_vals, width=bw, label='LI (Laplacian)',
           color='#d62728', alpha=0.75, edgecolor='white')

    # 加参照线：Alpha Fixed和Laplacian Fixed
    for ref_name, ref_color, ref_ls in [
        ('Alpha Fixed\n(8-13Hz)', '#4C72B0', '--'),
        ('Laplacian Fixed\n(8-30Hz)', '#DD8452', ':')
    ]:
        rv = score_dict.get(ref_name, {}).get(dk, None)
        if rv is not None:
            short = ref_name.split('\n')[0]
            ax.axhline(rv, color=ref_color, linestyle=ref_ls,
                       linewidth=1.5, label=f'Ref: {short}')

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in band_groups], fontsize=8)
    ax.set_ylabel(dl.split('(')[0], fontsize=9)
    ax.set_title(dl, fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out2 = os.path.join(SAVE_ROOT, 'step10_li_variants_comparison.png')
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f"[图2已保存] {out2}")

# --- 图3: LI_Lap_Multi逐被试delta一致性热图 ---
fi_best = next(i for i,(n,_,_) in enumerate(FEAT_CFG) if 'Lap Multi' in n)
deltas_best = []
for i in range(N):
    d1i = all_deltas_s1[fi_best][i]
    d2i = all_deltas_s2[fi_best][i]
    if d1i is not None and d2i is not None:
        deltas_best.append((d1i + d2i) / 2.0)
    elif d1i is not None:
        deltas_best.append(d1i)
    else:
        deltas_best.append(None)

deltas_valid = [d for d in deltas_best if d is not None]
n_v = len(deltas_valid)
mat = np.full((n_v, n_v), np.nan)
for i in range(n_v):
    for j in range(n_v):
        mat[i,j] = cosine_sim(deltas_valid[i], deltas_valid[j])

fig3, ax3 = plt.subplots(figsize=(8, 7))
im = ax3.imshow(mat, vmin=-1, vmax=1, cmap='RdYlGn', aspect='auto')
plt.colorbar(im, ax=ax3, label='Cosine Similarity of Delta Vectors')
ax3.set_title('LI Lap Multi-band: Per-Subject Delta Similarity\n'
              '(Green = consistent class-separation direction across subject pairs)',
              fontsize=10)
ax3.set_xlabel('Subject Index', fontsize=10)
ax3.set_ylabel('Subject Index', fontsize=10)
plt.tight_layout()
out3 = os.path.join(SAVE_ROOT, 'step10_li_lap_multi_heatmap.png')
plt.savefig(out3, dpi=200, bbox_inches='tight')
plt.close()
print(f"[图3已保存] {out3}")
