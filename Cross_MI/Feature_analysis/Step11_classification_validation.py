"""
Step11_classification_validation.py

目的：验证LI Alpha特征在跨被试分类场景下的实际分类性能。

对比方法：
  M1: LI_Alpha_threshold  -- LI(8-13Hz) + 阈值分类（群体均值为阈值，零个体校准）
  M2: LI_Alpha_LDA        -- LI(8-13Hz) 1D特征 + LDA，LOSO跨被试
  M3: LI_Mu_threshold     -- LI(8-12Hz) + 阈值分类（零个体校准）
  M4: Alpha_LDA           -- log功率[C3,C4] + LDA，LOSO跨被试
  M5: CSP_LDA_within      -- CSP+LDA，被试内10折交叉验证（校准上界）
  M6: CSP_LDA_cross       -- 公共空间CSP+LDA，LOSO跨被试

评估场景：
  S1: 标准LOSO（合并两场景训练，测试同被试两场景）
  S2: Cross-Double（训练Scene1所有被试，测试Scene2其余被试）

输出：
  - 逐被试准确率箱线图
  - 方法间统计比较（paired t-test）
  - 综合结论表格
"""

import os, sys
import numpy as np
import hdf5storage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel, wilcoxon
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

# ===================== 数据加载 =====================
def load_scene(subject, scene):
    pfx  = f"1S0{subject}" if subject < 10 else f"1S{subject}"
    path = os.path.join(DATA_ROOT, f"{pfx}{scene}_cue.mat")
    d    = hdf5storage.loadmat(path, mat_dtype=True)
    data = d['data'][:, START_PT:END_PT, :].transpose(2,0,1)
    data = data[:, CH_IDX, :]
    label = np.squeeze(d['label']).astype(int)
    return data, label

def load_subject_both(s):
    """加载单被试两场景合并数据"""
    parts_d, parts_l = [], []
    for sc in ['S1','S2']:
        try:
            d, l = load_scene(s, sc)
            parts_d.append(d); parts_l.append(l)
        except: pass
    if not parts_d: return None, None
    return np.concatenate(parts_d,0), np.concatenate(parts_l,0)

# ===================== 基础工具 =====================
def bandpass(data, low, high, order=4):
    nyq = SRATE/2.0
    b,a = butter(order,[low/nyq,high/nyq],btype='band')
    return filtfilt(b,a,data,axis=2)

def band_power(data, low, high):
    return np.mean(bandpass(data,low,high)**2, axis=2)

def li_feature(data, low=8, high=13):
    p3 = band_power(data, low, high)[:, C3_L]
    p4 = band_power(data, low, high)[:, C4_L]
    return (p3 - p4) / (p3 + p4 + 1e-10)   # (trials,)

def alpha_logpower(data):
    p = band_power(data, 8, 13)
    return np.log(p[:, [C3_L, C4_L]] + 1e-10)   # (trials, 2)

def cov_norm(X):
    C = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[2]):
        x = X[:,:,i]; tr = np.trace(x@x.T)+1e-10
        C += (x@x.T)/tr
    return C/X.shape[2]

def compute_csp(data, label, n=4):
    classes = np.unique(label)
    X1 = data[label==classes[0]].transpose(1,2,0)
    X2 = data[label==classes[1]].transpose(1,2,0)
    C1,C2 = cov_norm(X1), cov_norm(X2)
    vals,vecs = eigh(C1,C1+C2)
    vecs = vecs[:,np.argsort(vals)[::-1]]
    return np.hstack([vecs[:,:n],vecs[:,-n:]])

def csp_features(data, W):
    filt = np.tensordot(data, W, axes=([1],[0])).transpose(0,2,1)
    return np.log(np.var(filt,axis=2)+1e-10)

def accuracy(pred, true):
    return np.mean(pred == true)

# ===================== 预加载所有被试数据 =====================
print("预加载数据...")
all_data, all_label = [], []
valid_subs = []
for s in range(1, N_SUB+1):
    d, l = load_subject_both(s)
    if d is None or len(np.unique(l)) < 2:
        all_data.append(None); all_label.append(None)
    else:
        all_data.append(d); all_label.append(l)
        valid_subs.append(s)

print(f"有效被试: {len(valid_subs)}名")

# 预计算每个被试的特征（节省重复计算）
print("预计算特征...")
feat_li_alpha  = []   # LI 8-13Hz
feat_li_mu     = []   # LI 8-12Hz
feat_alpha_log = []   # log power [C3,C4] 8-13Hz
feat_csp_data  = []   # 原始宽带数据（用于CSP）

for i, s in enumerate(range(1, N_SUB+1)):
    if all_data[i] is None:
        feat_li_alpha.append(None)
        feat_li_mu.append(None)
        feat_alpha_log.append(None)
        feat_csp_data.append(None)
    else:
        d_broad = bandpass(all_data[i], 8, 30)
        feat_li_alpha.append(li_feature(all_data[i], 8, 13))
        feat_li_mu.append(li_feature(all_data[i], 8, 12))
        feat_alpha_log.append(alpha_logpower(all_data[i]))
        feat_csp_data.append(d_broad)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{N_SUB}")

# ===================== LOSO 评估框架 =====================
def loso_evaluate():
    """
    对每个有效被试作为测试集，其余所有被试作为训练集
    返回各方法的逐被试准确率列表
    """
    results = {
        'M1_LI_Alpha_thresh': [],
        'M2_LI_Alpha_LDA':    [],
        'M3_LI_Mu_thresh':    [],
        'M4_Alpha_LDA':       [],
        'M5_CSP_within':      [],
        'M6_CSP_cross':       [],
    }

    valid_idx = [i for i in range(N_SUB) if all_data[i] is not None]

    for ti, test_i in enumerate(valid_idx):
        train_idx = [i for i in valid_idx if i != test_i]

        d_test  = all_data[test_i]
        l_test  = all_label[test_i]
        classes = np.unique(l_test)

        # ---- 准备训练特征 ----
        li_a_train  = np.concatenate([feat_li_alpha[i] for i in train_idx])
        li_m_train  = np.concatenate([feat_li_mu[i]    for i in train_idx])
        alg_train   = np.concatenate([feat_alpha_log[i] for i in train_idx])
        l_train_all = np.concatenate([all_label[i]      for i in train_idx])

        # ---- M1: LI Alpha threshold ----
        li_a_test = feat_li_alpha[test_i]
        m1_c0 = np.mean(li_a_train[l_train_all == classes[0]])
        m1_c1 = np.mean(li_a_train[l_train_all == classes[1]])
        threshold = (m1_c0 + m1_c1) / 2.0
        # 判断哪个类对应LI较大
        if m1_c0 > m1_c1:
            pred_m1 = np.where(li_a_test > threshold, classes[0], classes[1])
        else:
            pred_m1 = np.where(li_a_test > threshold, classes[1], classes[0])
        results['M1_LI_Alpha_thresh'].append(accuracy(pred_m1, l_test))

        # ---- M2: LI Alpha LDA ----
        lda2 = LDA()
        lda2.fit(li_a_train.reshape(-1,1), l_train_all)
        pred_m2 = lda2.predict(li_a_test.reshape(-1,1))
        results['M2_LI_Alpha_LDA'].append(accuracy(pred_m2, l_test))

        # ---- M3: LI Mu threshold ----
        li_m_test = feat_li_mu[test_i]
        m3_c0 = np.mean(li_m_train[l_train_all == classes[0]])
        m3_c1 = np.mean(li_m_train[l_train_all == classes[1]])
        thr3 = (m3_c0 + m3_c1) / 2.0
        if m3_c0 > m3_c1:
            pred_m3 = np.where(li_m_test > thr3, classes[0], classes[1])
        else:
            pred_m3 = np.where(li_m_test > thr3, classes[1], classes[0])
        results['M3_LI_Mu_thresh'].append(accuracy(pred_m3, l_test))

        # ---- M4: Alpha log power LDA ----
        alg_test = feat_alpha_log[test_i]
        lda4 = LDA()
        lda4.fit(alg_train, l_train_all)
        pred_m4 = lda4.predict(alg_test)
        results['M4_Alpha_LDA'].append(accuracy(pred_m4, l_test))

        # ---- M5: CSP + LDA within-subject (10-fold) ----
        d_broad_test = feat_csp_data[test_i]
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_accs = []
        for tr_idx, te_idx in skf.split(d_broad_test, l_test):
            try:
                W = compute_csp(d_broad_test[tr_idx], l_test[tr_idx], n=4)
                f_tr = csp_features(d_broad_test[tr_idx], W)
                f_te = csp_features(d_broad_test[te_idx], W)
                clf = LDA()
                clf.fit(f_tr, l_test[tr_idx])
                fold_accs.append(accuracy(clf.predict(f_te), l_test[te_idx]))
            except:
                fold_accs.append(0.5)
        results['M5_CSP_within'].append(np.mean(fold_accs))

        # ---- M6: CSP cross-subject (公共子空间) ----
        # 在训练被试数据上计算CSP，投影到该公共空间后训练LDA
        try:
            d_csp_train = np.concatenate([feat_csp_data[i] for i in train_idx])
            W_cross = compute_csp(d_csp_train, l_train_all, n=4)
            f_tr6 = csp_features(d_csp_train, W_cross)
            f_te6 = csp_features(d_broad_test, W_cross)
            lda6 = LDA()
            lda6.fit(f_tr6, l_train_all)
            results['M6_CSP_cross'].append(accuracy(lda6.predict(f_te6), l_test))
        except:
            results['M6_CSP_cross'].append(0.5)

        sys.stdout.write(f"\r  LOSO {ti+1}/{len(valid_idx)} (Sub{test_i+1:02d})")
        sys.stdout.flush()

    print("\n LOSO完成")
    return results

# ===================== Cross-Double 评估 =====================
def cross_double_evaluate():
    """
    训练集: Scene1全部被试
    测试集: Scene2全部被试（同一批人，但作为独立测试集）
    """
    print("\n加载Scene1/Scene2数据...")
    data_s1, label_s1 = [], []
    data_s2, label_s2 = [], []
    cd_valid = []

    for s in range(1, N_SUB+1):
        try:
            d1, l1 = load_scene(s, 'S1')
            d2, l2 = load_scene(s, 'S2')
            if len(np.unique(l1)) >= 2 and len(np.unique(l2)) >= 2:
                data_s1.append(d1); label_s1.append(l1)
                data_s2.append(d2); label_s2.append(l2)
                cd_valid.append(s)
        except: pass

    print(f"  Cross-Double有效被试: {len(cd_valid)}名")

    # 合并Scene1作为训练
    D_train = np.concatenate(data_s1, axis=0)
    L_train = np.concatenate(label_s1)
    D_test  = np.concatenate(data_s2, axis=0)
    L_test  = np.concatenate(label_s2)

    classes = np.unique(L_train)

    cd_results = {}

    # M1: LI threshold
    li_tr = li_feature(D_train, 8, 13)
    li_te = li_feature(D_test, 8, 13)
    m1c0  = np.mean(li_tr[L_train==classes[0]])
    m1c1  = np.mean(li_tr[L_train==classes[1]])
    thr1  = (m1c0+m1c1)/2.0
    pred  = np.where(li_te>thr1, classes[0], classes[1]) if m1c0>m1c1 else \
            np.where(li_te>thr1, classes[1], classes[0])
    cd_results['M1_LI_Alpha_thresh'] = accuracy(pred, L_test)

    # M2: LI LDA
    li_tr2, li_te2 = li_tr.reshape(-1,1), li_te.reshape(-1,1)
    lda2 = LDA(); lda2.fit(li_tr2, L_train)
    cd_results['M2_LI_Alpha_LDA'] = accuracy(lda2.predict(li_te2), L_test)

    # M3: LI Mu threshold
    li_mu_tr = li_feature(D_train, 8, 12)
    li_mu_te = li_feature(D_test, 8, 12)
    m3c0 = np.mean(li_mu_tr[L_train==classes[0]])
    m3c1 = np.mean(li_mu_tr[L_train==classes[1]])
    thr3 = (m3c0+m3c1)/2.0
    pred3= np.where(li_mu_te>thr3,classes[0],classes[1]) if m3c0>m3c1 else \
           np.where(li_mu_te>thr3,classes[1],classes[0])
    cd_results['M3_LI_Mu_thresh'] = accuracy(pred3, L_test)

    # M4: Alpha LDA
    alg_tr = alpha_logpower(D_train)
    alg_te = alpha_logpower(D_test)
    lda4 = LDA(); lda4.fit(alg_tr, L_train)
    cd_results['M4_Alpha_LDA'] = accuracy(lda4.predict(alg_te), L_test)

    # M6: CSP cross
    d_tr_b = bandpass(D_train,8,30); d_te_b = bandpass(D_test,8,30)
    try:
        W6 = compute_csp(d_tr_b, L_train, n=4)
        f_tr6 = csp_features(d_tr_b,W6); f_te6 = csp_features(d_te_b,W6)
        lda6 = LDA(); lda6.fit(f_tr6,L_train)
        cd_results['M6_CSP_cross'] = accuracy(lda6.predict(f_te6),L_test)
    except:
        cd_results['M6_CSP_cross'] = 0.5

    return cd_results

# ===================== 运行 =====================
print("\n" + "="*60)
print("场景1: LOSO 跨被试评估")
print("="*60)
loso_res = loso_evaluate()

print("\n" + "="*60)
print("场景2: Cross-Double 评估")
print("="*60)
cd_res = cross_double_evaluate()

# 保存原始结果
np.save(os.path.join(SAVE_ROOT,'step11_loso.npy'), loso_res)
np.save(os.path.join(SAVE_ROOT,'step11_cd.npy'),   cd_res)

# ===================== 统计分析 =====================
METHOD_LABELS = {
    'M1_LI_Alpha_thresh': 'LI-Alpha\nThreshold\n(zero-calib)',
    'M2_LI_Alpha_LDA':    'LI-Alpha\nLDA\n(LOSO)',
    'M3_LI_Mu_thresh':    'LI-Mu\nThreshold\n(zero-calib)',
    'M4_Alpha_LDA':       'Alpha-Power\nLDA\n(LOSO)',
    'M5_CSP_within':      'CSP+LDA\nWithin-Sub\n(calibrated)',
    'M6_CSP_cross':       'CSP+LDA\nCross-Sub\n(LOSO)',
}
METHOD_ORDER = list(METHOD_LABELS.keys())
COLORS = ['#d62728','#ff7f0e','#e377c2','#7f7f7f','#1f77b4','#aec7e8']

print("\n" + "="*60)
print("LOSO准确率摘要")
print("="*60)
print(f"{'方法':<28} {'均值':>7} {'中位数':>7} {'std':>7} {'最小':>7} {'最大':>7}")
print("-"*60)
for m in METHOD_ORDER:
    vals = np.array(loso_res[m])
    print(f"  {METHOD_LABELS[m].replace(chr(10),' '):<28} "
          f"{vals.mean():>7.3f} {np.median(vals):>7.3f} "
          f"{vals.std():>7.3f} {vals.min():>7.3f} {vals.max():>7.3f}")

# paired t-test：LI_Alpha_thresh vs 其他
ref = np.array(loso_res['M1_LI_Alpha_thresh'])
print("\npaired t-test vs M1_LI_Alpha_thresh:")
for m in METHOD_ORDER[1:]:
    comp = np.array(loso_res[m])
    t, p = ttest_rel(ref, comp)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  vs {m:<28}: t={t:+.3f}, p={p:.4f} {sig}")

print(f"\nCross-Double准确率:")
for m, v in cd_res.items():
    print(f"  {METHOD_LABELS[m].replace(chr(10),' '):<35}: {v:.3f}")

# ===================== 绘图 =====================

# --- 图1: LOSO箱线图 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Classification Performance: LI Alpha vs Baselines\n'
             f'N={len(valid_subs)} subjects, cue paradigm (left/right hand MI)',
             fontsize=12, fontweight='bold')

ax = axes[0]
data_plot = [loso_res[m] for m in METHOD_ORDER]
bp = ax.boxplot(data_plot, patch_artist=True, notch=False,
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=4, alpha=0.5))
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color); patch.set_alpha(0.75)

ax.axhline(0.5, color='black', linestyle='--', linewidth=1, label='Chance (50%)')
ax.set_xticks(range(1, len(METHOD_ORDER)+1))
ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8)
ax.set_ylabel('Classification Accuracy', fontsize=11)
ax.set_title('LOSO Cross-Subject Accuracy', fontsize=11)
ax.set_ylim(0.3, 1.0)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

# 显著性标注（M1 vs M5, M1 vs M6）
def significance_bar(ax, x1, x2, y, p):
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    ax.plot([x1,x1,x2,x2],[y,y+0.01,y+0.01,y],lw=1.2,c='black')
    ax.text((x1+x2)/2, y+0.012, sig, ha='center', va='bottom', fontsize=9)

m1v = np.array(loso_res['M1_LI_Alpha_thresh'])
for xi, mk in [(5,'M5_CSP_within'),(6,'M6_CSP_cross')]:
    _, p = ttest_rel(m1v, np.array(loso_res[mk]))
    significance_bar(ax, 1, xi, 0.92 + (xi-5)*0.03, p)

# --- 图2: Cross-Double对比 + 均值对比 ---
ax2 = axes[1]
cd_methods = [m for m in METHOD_ORDER if m in cd_res and m != 'M5_CSP_within']
cd_vals    = [cd_res[m] for m in cd_methods]
loso_means = [np.mean(loso_res[m]) for m in cd_methods]
bar_colors2= [COLORS[METHOD_ORDER.index(m)] for m in cd_methods]

x = np.arange(len(cd_methods))
bw = 0.35
bars_loso = ax2.bar(x - bw/2, loso_means, bw,
                    color=bar_colors2, alpha=0.6, edgecolor='black',
                    linewidth=0.8, label='LOSO mean')
bars_cd   = ax2.bar(x + bw/2, cd_vals,    bw,
                    color=bar_colors2, alpha=0.95, edgecolor='black',
                    linewidth=0.8, label='Cross-Double', hatch='//')

# LOSO±std误差棒
for xi, m in enumerate(cd_methods):
    std = np.std(loso_res[m])
    ax2.errorbar(xi - bw/2, np.mean(loso_res[m]), yerr=std,
                 fmt='none', color='black', capsize=4, linewidth=1.5)

ax2.axhline(0.5, color='black', linestyle='--', linewidth=1)
ax2.set_xticks(x)
ax2.set_xticklabels([METHOD_LABELS[m] for m in cd_methods], fontsize=7.5)
ax2.set_ylabel('Classification Accuracy', fontsize=11)
ax2.set_title('LOSO mean±std vs Cross-Double\n(bar=mean, error=std, //=cross-double)',
              fontsize=10)
ax2.set_ylim(0.3, 1.0)
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)

# 在柱子上标数值
for xi, (lm, cdv) in enumerate(zip(loso_means, cd_vals)):
    ax2.text(xi-bw/2, lm+0.01, f'{lm:.2f}', ha='center', va='bottom', fontsize=7.5)
    ax2.text(xi+bw/2, cdv+0.01, f'{cdv:.2f}', ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
out1 = os.path.join(SAVE_ROOT,'step11_classification.png')
plt.savefig(out1, dpi=200, bbox_inches='tight')
plt.close()
print(f"\n[图1已保存] {out1}")

# --- 图2: 逐被试准确率散点图（M1 vs M5）---
fig2, ax3 = plt.subplots(figsize=(8, 8))
m1_accs = np.array(loso_res['M1_LI_Alpha_thresh'])
m5_accs = np.array(loso_res['M5_CSP_within'])
m2_accs = np.array(loso_res['M2_LI_Alpha_LDA'])

ax3.scatter(m5_accs, m1_accs, c='#d62728', alpha=0.75, s=60,
            label=f'LI-Alpha Threshold (mean={m1_accs.mean():.3f})', zorder=3)
ax3.scatter(m5_accs, m2_accs, c='#ff7f0e', alpha=0.75, s=60, marker='^',
            label=f'LI-Alpha LDA (mean={m2_accs.mean():.3f})', zorder=3)
ax3.plot([0.3,1.0],[0.3,1.0], 'k--', linewidth=1, label='y=x (equal performance)')
ax3.axhline(0.5, color='gray', linestyle=':', linewidth=1)
ax3.axvline(0.5, color='gray', linestyle=':', linewidth=1)
ax3.set_xlabel('CSP+LDA Within-Subject (calibrated baseline)', fontsize=11)
ax3.set_ylabel('LI-Alpha Methods (zero/minimal calibration)', fontsize=11)
ax3.set_title('Per-Subject: LI-Alpha vs CSP+LDA Within-Subject\n'
              'Points above diagonal = LI outperforms calibrated CSP',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_xlim(0.3,1.0); ax3.set_ylim(0.3,1.0)
ax3.grid(alpha=0.3)

# 标注哪些被试LI超过了CSP
n_better = np.sum(m1_accs > m5_accs)
ax3.text(0.35, 0.95, f'LI-Thresh > CSP: {n_better}/{len(m1_accs)} subjects',
         transform=ax3.transAxes, fontsize=10, color='#d62728',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
out2 = os.path.join(SAVE_ROOT,'step11_persubject_scatter.png')
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f"[图2已保存] {out2}")

print("\n" + "="*60)
print("完成！")
print("="*60)
