"""
Step8_universal_feature_analysis.py

目的：系统评估主流MI EEG特征的跨被试通用性。

核心指标：跨被试类方向一致性（Cross-Subject Class Direction Consistency, CSCDC）
    - 对每个被试计算差向量 delta_i = centroid_left_i - centroid_right_i
    - 计算所有被试对之间 delta 向量的余弦相似度
    - 均值越高 → 类分离方向跨被试越一致 → 特征越通用
    - 该指标与分类器无关，纯粹衡量特征空间结构的跨被试一致性

涵盖7类17种特征：
  [频谱]     F01 Alpha功率(C3/C4), F02 Beta功率(C3/C4), F03 Mu功率(C3/C4), F04 Alpha/Beta比值
  [空间滤波] F05 CSP对数方差, F06 FBCSP多频带对数方差, F07 Laplacian滤波功率
  [黎曼]     F08 协方差切空间向量, F09 协方差对角log
  [时频]     F10 小波能量(近似db4), F11 STFT频段功率
  [连接性]   F12 PLV相位锁定值(C3-C4), F13 C3-C4相干性
  [非线性]   F14 样本熵, F15 Hjorth参数
  [统计时域] F16 AR模型系数, F17 高阶统计矩

数据：4_跨场景因素研究v2 / cue范式，37名被试，Scene1+Scene2合并分析
输出：跨被试通用性排名图 + 逐被试热图
"""

import os
import numpy as np
import hdf5storage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, stft, coherence
from scipy.linalg import eigh, toeplitz
from sklearn.covariance import ledoit_wolf
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ===================== 参数 =====================
DATA_ROOT = r'E:\Datasets\4_跨场景因素研究v2' + r'\跨场景因素研究v2处理后数据'
SAVE_ROOT = r'E:\Datasets\4_跨场景因素研究v2' + r'\跨场景因素研究v2画图数据\stability'
os.makedirs(SAVE_ROOT, exist_ok=True)

SRATE      = 250
START_PT   = 2 * SRATE   # 500
END_PT     = 6 * SRATE   # 1500
N_SUBJECTS = 37

# 感觉运动区21通道（0-based）
CH_IDX  = [i-1 for i in [16,17,18,19,20,21,22, 25,26,27,28,29,30,31, 34,35,36,37,38,39,40]]
N_CH    = len(CH_IDX)   # 21

# C3/C4在CH_IDX中的局部索引
C3_L = CH_IDX.index(26)   # C3
C4_L = CH_IDX.index(30)   # C4

# ===================== 数据加载 =====================
def load_subject(subject, scene):
    pfx  = f"1S0{subject}" if subject < 10 else f"1S{subject}"
    path = os.path.join(DATA_ROOT, f"{pfx}{scene}_cue.mat")
    d    = hdf5storage.loadmat(path, mat_dtype=True)
    data = d['data'][:, START_PT:END_PT, :].transpose(2, 0, 1)  # (trials,60,time)
    data = data[:, CH_IDX, :]                                    # (trials,21,time)
    label = np.squeeze(d['label']).astype(int)
    return data, label

# ===================== 基础工具 =====================
def bandpass(data, low, high, fs=SRATE, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=2)

def band_power(data, low, high):
    """(trials,ch,time) -> (trials,ch) 频带功率"""
    d = bandpass(data, low, high)
    return np.mean(d**2, axis=2)

def cov_norm(X):
    """X:(ch,time,trials) -> 归一化协方差"""
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[2]):
        x = X[:, :, i]
        tr = np.trace(x @ x.T) + 1e-10
        C += (x @ x.T) / tr
    return C / X.shape[2]

def compute_csp(data, label, n=4):
    classes = np.unique(label)
    X1 = data[label==classes[0]].transpose(1,2,0)
    X2 = data[label==classes[1]].transpose(1,2,0)
    C1, C2 = cov_norm(X1), cov_norm(X2)
    vals, vecs = eigh(C1, C1+C2)
    vecs = vecs[:, np.argsort(vals)[::-1]]
    return np.hstack([vecs[:, :n], vecs[:, -n:]])

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

# ===================== 特征提取函数 =====================

# --- F01-F03: 频段功率 ---
def feat_alpha_power(data, label):
    p = band_power(data, 8, 13)
    return np.log(p[:, [C3_L, C4_L]] + 1e-10)

def feat_beta_power(data, label):
    p = band_power(data, 13, 30)
    return np.log(p[:, [C3_L, C4_L]] + 1e-10)

def feat_mu_power(data, label):
    p = band_power(data, 8, 12)
    return np.log(p[:, [C3_L, C4_L]] + 1e-10)

# --- F04: Alpha/Beta 比值 ---
def feat_alpha_beta_ratio(data, label):
    pa = band_power(data, 8, 13)[:, [C3_L, C4_L]]
    pb = band_power(data, 13, 30)[:, [C3_L, C4_L]]
    return pa / (pb + 1e-10)

# --- F05: CSP 对数方差 ---
def feat_csp(data, label):
    d = bandpass(data, 8, 30)
    W = compute_csp(d, label, n=4)
    filt = np.einsum('tc,nct->nt', W, d)   # (trials, 2n, time) wrong shape fix:
    # d:(trials,ch,time), W:(ch,2n) -> filt:(trials,2n,time)
    filt = np.tensordot(d, W, axes=([1],[0])).transpose(0,2,1)
    return np.log(np.var(filt, axis=2) + 1e-10)

# --- F06: FBCSP ---
FBANK = [(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]
def feat_fbcsp(data, label, n=2):
    feats = []
    for (lo, hi) in FBANK:
        try:
            d = bandpass(data, lo, hi)
            W = compute_csp(d, label, n=n)
            filt = np.tensordot(d, W, axes=([1],[0])).transpose(0,2,1)
            feats.append(np.log(np.var(filt, axis=2) + 1e-10))
        except Exception:
            feats.append(np.zeros((data.shape[0], 2*n)))
    return np.hstack(feats)

# --- F07: Laplacian 滤波功率 ---
def feat_laplacian(data, label):
    # 简化Laplacian：C3 - 周边均值（FC3,CP3,C1,C5），C4类似
    # 在21通道局部索引中近似：C3=6(FC5侧), 用C3减去相邻通道均值
    # 精确索引：CH_IDX对应原始通道(0-based): FC5-FC6=15-21, C5-C6=24-30, CP5-CP6=33-39
    # C3=26,C4=30; 相邻: FC3~17,CP3~35,C1~27,C5~24 -> 局部索引
    neighbors_c3 = [CH_IDX.index(i) for i in [17,27,24,35] if i in CH_IDX]
    neighbors_c4 = [CH_IDX.index(i) for i in [19,29,28,37] if i in CH_IDX]
    d_broad = bandpass(data, 8, 30)
    c3_lap = d_broad[:, C3_L, :] - d_broad[:, neighbors_c3, :].mean(axis=1)
    c4_lap = d_broad[:, C4_L, :] - d_broad[:, neighbors_c4, :].mean(axis=1)
    p_c3 = np.log(np.mean(c3_lap**2, axis=1, keepdims=True) + 1e-10)
    p_c4 = np.log(np.mean(c4_lap**2, axis=1, keepdims=True) + 1e-10)
    return np.hstack([p_c3, p_c4])

# --- F08: 黎曼切空间 ---
def feat_riemannian_tangent(data, label):
    d = bandpass(data, 8, 30)
    n_trials, n_ch, _ = d.shape
    covs = np.array([ledoit_wolf(d[i].T)[0] for i in range(n_trials)])
    mean_cov = covs.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(mean_cov)
    eigvals = np.maximum(eigvals, 1e-10)
    W = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
    idx = np.triu_indices(n_ch)
    feats = []
    for i in range(n_trials):
        S = W @ covs[i] @ W.T
        feats.append(S[idx])
    return np.array(feats)

# --- F09: 协方差对角 log ---
def feat_cov_diag(data, label):
    d = bandpass(data, 8, 30)
    n_trials = d.shape[0]
    feats = []
    for i in range(n_trials):
        cov, _ = ledoit_wolf(d[i].T)
        feats.append(np.log(np.diag(cov) + 1e-10))
    return np.array(feats)

# --- F10: 小波能量（用滤波器组近似db4分解）---
def feat_wavelet_energy(data, label):
    # db4近似：用一组窄带滤波器近似小波子带能量
    # 子带中心频率近似对应: 1-4Hz, 4-8Hz, 8-16Hz, 16-32Hz (尺度1-4)
    bands_wavelet = [(1,4),(4,8),(8,16),(16,32)]
    feats = []
    for (lo, hi) in bands_wavelet:
        p = band_power(data, lo, hi)[:, [C3_L, C4_L]]
        feats.append(np.log(p + 1e-10))
    return np.hstack(feats)   # (trials, 8)

# --- F11: STFT 频段功率 ---
def feat_stft_power(data, label):
    # 对C3/C4做STFT，提取alpha和beta频段的时均功率
    feats = []
    for ch_l in [C3_L, C4_L]:
        x = data[:, ch_l, :]   # (trials, time)
        alpha_pow = []
        beta_pow  = []
        for trial in range(x.shape[0]):
            f, _, Zxx = stft(x[trial], fs=SRATE, nperseg=64, noverlap=32)
            psd = np.abs(Zxx)**2
            alpha_idx = (f >= 8) & (f <= 13)
            beta_idx  = (f >= 13) & (f <= 30)
            alpha_pow.append(np.log(psd[alpha_idx].mean() + 1e-10))
            beta_pow.append(np.log(psd[beta_idx].mean()  + 1e-10))
        feats.append(np.array(alpha_pow))
        feats.append(np.array(beta_pow))
    return np.column_stack(feats)   # (trials, 4)

# --- F12: PLV 相位锁定值 C3-C4 ---
def feat_plv(data, label):
    feats = []
    for (lo, hi) in [(8,13),(13,30)]:
        d = bandpass(data, lo, hi)
        c3 = d[:, C3_L, :]
        c4 = d[:, C4_L, :]
        # 希尔伯特变换求瞬时相位
        from scipy.signal import hilbert
        phase_c3 = np.angle(hilbert(c3, axis=1))
        phase_c4 = np.angle(hilbert(c4, axis=1))
        plv = np.abs(np.mean(np.exp(1j * (phase_c3 - phase_c4)), axis=1))
        feats.append(plv)
    return np.column_stack(feats)   # (trials, 2)

# --- F13: C3-C4 相干性 ---
def feat_coherence(data, label):
    feats = []
    for trial in range(data.shape[0]):
        c3 = data[trial, C3_L, :]
        c4 = data[trial, C4_L, :]
        f, Cxy = coherence(c3, c4, fs=SRATE, nperseg=128)
        alpha_coh = Cxy[(f>=8)&(f<=13)].mean()
        beta_coh  = Cxy[(f>=13)&(f<=30)].mean()
        feats.append([alpha_coh, beta_coh])
    return np.array(feats)   # (trials, 2)

# --- F14: 样本熵 (手动实现，避免依赖外部包) ---
def sample_entropy(x, m=2, r_factor=0.2):
    """单通道时间序列的样本熵"""
    N = len(x)
    r = r_factor * np.std(x)
    def phi(m_):
        count = 0
        templates = np.array([x[i:i+m_] for i in range(N-m_)])
        for i in range(len(templates)):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dists <= r) - 1  # 减去自身
        return count / (N - m_) / (N - m_ - 1 + 1e-10)
    p1, p2 = phi(m), phi(m+1)
    if p1 < 1e-10:
        return 0.0
    return -np.log(p2 / p1 + 1e-10)

def feat_sample_entropy(data, label):
    # 只对C3/C4计算，降采样加速
    feats = []
    for trial in range(data.shape[0]):
        row = []
        for ch_l in [C3_L, C4_L]:
            x = data[trial, ch_l, ::4]   # 降采样到62.5Hz
            row.append(sample_entropy(x[:100]))   # 只用100点保证速度
        feats.append(row)
    return np.array(feats)   # (trials, 2)

# --- F15: Hjorth 参数 ---
def feat_hjorth(data, label):
    """Activity, Mobility, Complexity for C3/C4"""
    feats = []
    for trial in range(data.shape[0]):
        row = []
        for ch_l in [C3_L, C4_L]:
            x  = data[trial, ch_l, :]
            dx = np.diff(x)
            ddx = np.diff(dx)
            activity   = np.var(x)
            mobility   = np.sqrt(np.var(dx) / (np.var(x) + 1e-10))
            complexity = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-10)) / (mobility + 1e-10)
            row.extend([np.log(activity+1e-10), mobility, complexity])
        feats.append(row)
    return np.array(feats)   # (trials, 6)

# --- F16: AR 模型系数 (Burg方法近似，用Yule-Walker代替) ---
def yule_walker_ar(x, order=8):
    """Yule-Walker AR系数估计"""
    r = np.correlate(x, x, mode='full')
    r = r[len(r)//2:]
    R = toeplitz(r[:order])
    b = r[1:order+1]
    try:
        a = np.linalg.solve(R, b)
    except np.linalg.LinAlgError:
        a = np.zeros(order)
    return a

def feat_ar(data, label, order=8):
    # 对C3/C4的宽带信号提取AR系数
    d = bandpass(data, 4, 40)
    feats = []
    for trial in range(d.shape[0]):
        row = []
        for ch_l in [C3_L, C4_L]:
            x = d[trial, ch_l, :]
            row.extend(yule_walker_ar(x, order))
        feats.append(row)
    return np.array(feats)   # (trials, 2*order)

# --- F17: 高阶统计矩 ---
def feat_moments(data, label):
    """方差、偏度、峰度 for C3/C4 in alpha/beta band"""
    from scipy.stats import skew, kurtosis
    feats = []
    for trial in range(data.shape[0]):
        row = []
        for band in [(8,13),(13,30)]:
            d_band = bandpass(data[trial:trial+1], *band)[0]
            for ch_l in [C3_L, C4_L]:
                x = d_band[ch_l]
                row.extend([np.var(x), skew(x), kurtosis(x)])
        feats.append(row)
    return np.array(feats)   # (trials, 12)

# ===================== 特征注册表 =====================
FEATURES = [
    ('F01_Alpha_Power',     'Spectral',        feat_alpha_power),
    ('F02_Beta_Power',      'Spectral',        feat_beta_power),
    ('F03_Mu_Power',        'Spectral',        feat_mu_power),
    ('F04_Alpha_Beta_Ratio','Spectral',        feat_alpha_beta_ratio),
    ('F05_CSP_LogVar',      'Spatial',         feat_csp),
    ('F06_FBCSP_LogVar',    'Spatial',         feat_fbcsp),
    ('F07_Laplacian_Power', 'Spatial',         feat_laplacian),
    ('F08_Riemann_Tangent', 'Riemannian',      feat_riemannian_tangent),
    ('F09_Cov_Diag_Log',    'Riemannian',      feat_cov_diag),
    ('F10_Wavelet_Energy',  'TimeFreq',        feat_wavelet_energy),
    ('F11_STFT_Power',      'TimeFreq',        feat_stft_power),
    ('F12_PLV_C3C4',        'Connectivity',    feat_plv),
    ('F13_Coherence_C3C4',  'Connectivity',    feat_coherence),
    ('F14_SampleEntropy',   'Nonlinear',       feat_sample_entropy),
    ('F15_Hjorth',          'Nonlinear',       feat_hjorth),
    ('F16_AR_Coeff',        'Statistical',     feat_ar),
    ('F17_Moments',         'Statistical',     feat_moments),
]

CATEGORY_COLORS = {
    'Spectral':     '#4C72B0',
    'Spatial':      '#DD8452',
    'Riemannian':   '#55A868',
    'TimeFreq':     '#C44E52',
    'Connectivity': '#8172B2',
    'Nonlinear':    '#937860',
    'Statistical':  '#DA8BC3',
}

# ===================== CSCDC 计算 =====================
def compute_cscdc(delta_list):
    """
    delta_list: list of delta向量（每个被试一个）
    返回所有被试对的余弦相似度均值（跨被试类方向一致性）
    """
    sims = []
    for i, j in combinations(range(len(delta_list)), 2):
        s = cosine_sim(delta_list[i], delta_list[j])
        if not np.isnan(s):
            sims.append(s)
    return float(np.mean(sims)) if sims else np.nan


# ===================== 主流程 =====================
print("=" * 65)
print("跨被试通用特征分析 (CSCDC指标)")
print("=" * 65)

# 收集所有被试的特征delta向量
# all_deltas[feat_name] = list of delta vectors (per subject)
all_deltas = {name: [] for name, _, _ in FEATURES}
valid_subjects = []

for s in range(1, N_SUBJECTS + 1):
    sub_data_list  = []
    sub_label_list = []
    for scene in ['S1', 'S2']:
        try:
            d, lbl = load_subject(s, scene)
            sub_data_list.append(d)
            sub_label_list.append(lbl)
        except Exception as e:
            pass

    if not sub_data_list:
        print(f"Subject {s:02d}: 数据缺失，跳过")
        continue

    # 合并两场景
    data_all  = np.concatenate(sub_data_list,  axis=0)
    label_all = np.concatenate(sub_label_list, axis=0)
    classes   = np.unique(label_all)
    if len(classes) < 2:
        continue

    valid_subjects.append(s)
    print(f"Subject {s:02d} (n={len(label_all)}): ", end='', flush=True)

    for feat_name, category, feat_fn in FEATURES:
        try:
            feat = feat_fn(data_all, label_all)
            # 标准化（消除被试间尺度差异，保留方向结构）
            mu  = feat.mean(axis=0)
            std = feat.std(axis=0) + 1e-10
            feat_norm = (feat - mu) / std
            c0 = feat_norm[label_all == classes[0]].mean(axis=0)
            c1 = feat_norm[label_all == classes[1]].mean(axis=0)
            delta = c0 - c1
            all_deltas[feat_name].append(delta)
            print('.', end='', flush=True)
        except Exception as e:
            all_deltas[feat_name].append(None)
            print('x', end='', flush=True)

    print()

print(f"\n有效被试数: {len(valid_subjects)}")

# ===================== 计算 CSCDC =====================
print("\n计算跨被试类方向一致性...")
results = []
for feat_name, category, _ in FEATURES:
    deltas = [d for d in all_deltas[feat_name] if d is not None]
    if len(deltas) < 5:
        cscdc = np.nan
    else:
        cscdc = compute_cscdc(deltas)
    results.append({
        'name':     feat_name,
        'category': category,
        'cscdc':    cscdc,
        'n_valid':  len(deltas),
    })
    print(f"  {feat_name:<30s} CSCDC={cscdc:.4f}  (n={len(deltas)})")

# 保存结果
np.save(os.path.join(SAVE_ROOT, 'cscdc_results.npy'), results)

# ===================== 绘图 =====================
valid_results = [r for r in results if not np.isnan(r['cscdc'])]
valid_results.sort(key=lambda x: x['cscdc'], reverse=True)

feat_names  = [r['name'].replace('_',' ') for r in valid_results]
cscdc_vals  = [r['cscdc'] for r in valid_results]
bar_colors  = [CATEGORY_COLORS[r['category']] for r in valid_results]
categories  = [r['category'] for r in valid_results]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(valid_results)), cscdc_vals,
               color=bar_colors, edgecolor='white', linewidth=0.8, alpha=0.88)

# 数值标注
for i, (val, r) in enumerate(zip(cscdc_vals, valid_results)):
    ax.text(val + 0.003, i, f'{val:.3f}', va='center', fontsize=9)

ax.set_yticks(range(len(valid_results)))
ax.set_yticklabels(feat_names, fontsize=9)
ax.set_xlabel('Cross-Subject Class Direction Consistency (CSCDC)', fontsize=11)
ax.set_title('Universal Feature Analysis: Cross-Subject Class Direction Consistency\n'
             f'(N={len(valid_subjects)} subjects, cue paradigm, Scene1+Scene2 combined)\n'
             'Higher = class separation direction more consistent across subjects',
             fontsize=11, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

# 图例（类别颜色）
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=k, alpha=0.85)
                   for k, c in CATEGORY_COLORS.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
          title='Feature Category', title_fontsize=9)

plt.tight_layout()
out1 = os.path.join(SAVE_ROOT, 'universal_feature_ranking.png')
plt.savefig(out1, dpi=200, bbox_inches='tight')
plt.close()
print(f"\n[图1已保存] {out1}")

# --- 图2: 逐被试 delta 向量相似度热图（Top5特征）---
top5 = [r['name'] for r in valid_results[:5]]

fig2, axes2 = plt.subplots(1, 5, figsize=(20, 6))
fig2.suptitle('Per-Subject Pairwise Delta Cosine Similarity (Top-5 Features)',
              fontsize=12, fontweight='bold')

for ax_i, feat_name in enumerate(top5):
    deltas = [d for d in all_deltas[feat_name] if d is not None]
    n = len(deltas)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            mat[i, j] = cosine_sim(deltas[i], deltas[j])

    ax = axes2[ax_i]
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdYlGn', aspect='auto')
    ax.set_title(feat_name.replace('_', '\n'), fontsize=8)
    ax.set_xlabel('Subject', fontsize=8)
    if ax_i == 0:
        ax.set_ylabel('Subject', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
out2 = os.path.join(SAVE_ROOT, 'universal_feature_similarity_heatmap.png')
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f"[图2已保存] {out2}")

# ===================== 文字摘要 =====================
print("\n" + "=" * 65)
print(f"CSCDC排名（N={len(valid_subjects)}名被试）")
print("=" * 65)
print(f"{'排名':<4} {'特征':<32} {'类别':<14} {'CSCDC':>8}")
print("-" * 65)
for rank, r in enumerate(valid_results, 1):
    print(f"{rank:<4} {r['name']:<32} {r['category']:<14} {r['cscdc']:>8.4f}")
print("=" * 65)
