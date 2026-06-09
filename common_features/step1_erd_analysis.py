"""
Step 1: 跨被试/跨会话 ERD/ERS 共性分析

目标: 找到在所有被试和会话中一致出现的运动想象频谱-空间激活模式。

方法:
  1. 对每个被试/会话提取 mu(8-12Hz) 和 beta(13-30Hz) 频段功率
  2. 计算基线期(0-1s)和想象期(2-5s)的 ERD(%) = (power_mi - power_base) / power_base * 100
  3. 跨被试统计检验 (paired t-test + FDR校正) 找到一致激活通道
  4. 可视化: 每通道ERD均值热图 + 统计显著图

输出:
  - results/step1_erd_results.npz  (各被试ERD矩阵)
  - results/step1_erd_heatmap.png  (跨被试平均ERD热图)
  - results/step1_consistent_channels.txt  (一致激活通道列表)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import signal, stats
from scipy.stats import ttest_1samp
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_all_subjects, FS, MOTOR_CH_NAMES

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# 时间段设置 (单位: 样本点)
# 试验长度 6秒 × 250Hz = 1500点
# 基线: 0-1s (0-250点)
# 想象期: 2-5s (500-1250点) — Graz范式提示后2s开始
BASE_START, BASE_END = 0, int(1.0 * FS)       # 基线段
MI_START,   MI_END   = int(2.0 * FS), int(5.0 * FS)  # 想象段


def bandpower(data, fs, fmin, fmax, method='welch'):
    """
    计算频段功率
    data: (channels, timepoints)
    返回: (channels,) 每通道频段平均功率
    """
    nperseg = min(data.shape[-1], 4 * fs)
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=-1)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    return np.mean(psd[:, freq_mask], axis=-1)


def compute_erd(data, label, fs=FS):
    """
    计算每个类别的 ERD
    data: (trials, channels, timepoints)
    返回: dict {class_id: erd_array (channels,)}
    """
    n_ch = data.shape[1]
    erd_dict = {}
    for cls in np.unique(label):
        X_cls = data[label == cls]  # (n_trials, ch, time)
        base = X_cls[:, :, BASE_START:BASE_END]   # 基线段
        mi   = X_cls[:, :, MI_START:MI_END]        # 想象段

        # 各频段功率 (trials, channels)
        power_base_mu   = np.array([bandpower(t, fs, 8, 12)  for t in base])
        power_mi_mu     = np.array([bandpower(t, fs, 8, 12)  for t in mi])
        power_base_beta = np.array([bandpower(t, fs, 13, 30) for t in base])
        power_mi_beta   = np.array([bandpower(t, fs, 13, 30) for t in mi])

        # ERD(%) = (mi - base) / base * 100, 均值
        erd_mu   = np.mean((power_mi_mu   - power_base_mu)   / (power_base_mu   + 1e-10) * 100, axis=0)
        erd_beta = np.mean((power_mi_beta - power_base_beta) / (power_base_beta + 1e-10) * 100, axis=0)
        erd_dict[int(cls)] = {'mu': erd_mu, 'beta': erd_beta}
    return erd_dict


def run_erd_analysis(subjects=None):
    """主分析流程"""
    print('=== Step 1: ERD/ERS 跨被试共性分析 ===')
    dataset = load_all_subjects(subjects=subjects, sessions=(1, 2), classes=(1, 2))

    n_ch = 60
    all_erd_mu   = []   # 每个 (sub, sess) 的 ERD mu (2类之差)
    all_erd_beta = []   # 每个 (sub, sess) 的 ERD beta
    keys = []

    for (sub, sess), (data, label) in dataset.items():
        print(f'  Processing S{sub:02d} Session{sess} ...')
        erd = compute_erd(data, label)

        # 对比度: cls0-cls1 (左-右 ERD差)
        if 0 in erd and 1 in erd:
            diff_mu   = erd[0]['mu']   - erd[1]['mu']
            diff_beta = erd[0]['beta'] - erd[1]['beta']
            all_erd_mu.append(diff_mu)
            all_erd_beta.append(diff_beta)
            keys.append((sub, sess))

    all_erd_mu   = np.array(all_erd_mu)    # (N_domains, n_ch)
    all_erd_beta = np.array(all_erd_beta)  # (N_domains, n_ch)

    # 跨域统计检验: H0: ERD差 = 0 (无类别效应)
    t_mu,   p_mu   = ttest_1samp(all_erd_mu,   0, axis=0)
    t_beta, p_beta = ttest_1samp(all_erd_beta, 0, axis=0)

    # FDR 校正
    from statsmodels.stats.multitest import fdrcorrection
    sig_mu,   p_mu_fdr   = fdrcorrection(p_mu)
    sig_beta, p_beta_fdr = fdrcorrection(p_beta)

    mean_erd_mu   = np.mean(all_erd_mu,   axis=0)
    mean_erd_beta = np.mean(all_erd_beta, axis=0)

    # 保存数值结果
    np.savez(os.path.join(RESULT_DIR, 'step1_erd_results.npz'),
             all_erd_mu=all_erd_mu, all_erd_beta=all_erd_beta,
             mean_erd_mu=mean_erd_mu, mean_erd_beta=mean_erd_beta,
             t_mu=t_mu, p_mu=p_mu, sig_mu=sig_mu,
             t_beta=t_beta, p_beta=p_beta, sig_beta=sig_beta,
             keys=np.array(keys))
    print(f'  数值结果已保存至 results/step1_erd_results.npz')

    # 记录一致激活通道
    consistent_mu   = [MOTOR_CH_NAMES[i] for i in range(n_ch) if sig_mu[i]]
    consistent_beta = [MOTOR_CH_NAMES[i] for i in range(n_ch) if sig_beta[i]]
    with open(os.path.join(RESULT_DIR, 'step1_consistent_channels.txt'), 'w', encoding='utf-8') as f:
        f.write('=== Mu频段(8-12Hz) 跨域一致激活通道 ===\n')
        f.write('\n'.join(f'  CH{i:02d} {MOTOR_CH_NAMES[i]}: t={t_mu[i]:.3f}, p_fdr={p_mu_fdr[i]:.4f}'
                          for i in range(n_ch) if sig_mu[i]) + '\n')
        f.write('\n=== Beta频段(13-30Hz) 跨域一致激活通道 ===\n')
        f.write('\n'.join(f'  CH{i:02d} {MOTOR_CH_NAMES[i]}: t={t_beta[i]:.3f}, p_fdr={p_beta_fdr[i]:.4f}'
                          for i in range(n_ch) if sig_beta[i]) + '\n')
    print(f'  Mu频段一致激活通道: {len(consistent_mu)}个')
    print(f'  Beta频段一致激活通道: {len(consistent_beta)}个')

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('跨被试/会话 ERD/ERS 共性分析\n(左手-右手 ERD对比度)', fontsize=14)

    ch_idx = np.arange(n_ch)

    # 1. Mu频段 ERD 跨域分布 (箱线图)
    ax = axes[0, 0]
    sorted_idx = np.argsort(mean_erd_mu)[::-1]
    top20 = sorted_idx[:20]
    bp = ax.boxplot([all_erd_mu[:, i] for i in top20],
                    labels=[f'CH{i}' for i in top20], patch_artist=True)
    for patch, idx in zip(bp['boxes'], top20):
        patch.set_facecolor('lightcoral' if sig_mu[idx] else 'lightgray')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Mu频段(8-12Hz) ERD - Top20通道')
    ax.set_ylabel('ERD差值(%)')
    ax.tick_params(axis='x', rotation=60)

    # 2. Beta频段 ERD 跨域分布
    ax = axes[0, 1]
    sorted_idx_b = np.argsort(mean_erd_beta)[::-1]
    top20_b = sorted_idx_b[:20]
    bp2 = ax.boxplot([all_erd_beta[:, i] for i in top20_b],
                     labels=[f'CH{i}' for i in top20_b], patch_artist=True)
    for patch, idx in zip(bp2['boxes'], top20_b):
        patch.set_facecolor('lightblue' if sig_beta[idx] else 'lightgray')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Beta频段(13-30Hz) ERD - Top20通道')
    ax.set_ylabel('ERD差值(%)')
    ax.tick_params(axis='x', rotation=60)

    # 3. Mu频段 t统计量 (全通道)
    ax = axes[1, 0]
    colors_mu = ['red' if s else 'steelblue' for s in sig_mu]
    ax.bar(ch_idx, t_mu, color=colors_mu, alpha=0.7)
    ax.set_title('Mu频段 t统计量 (红=FDR显著, p<0.05)')
    ax.set_xlabel('通道编号')
    ax.set_ylabel('t值')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    # 4. Beta频段 t统计量 (全通道)
    ax = axes[1, 1]
    colors_beta = ['red' if s else 'steelblue' for s in sig_beta]
    ax.bar(ch_idx, t_beta, color=colors_beta, alpha=0.7)
    ax.set_title('Beta频段 t统计量 (红=FDR显著, p<0.05)')
    ax.set_xlabel('通道编号')
    ax.set_ylabel('t值')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, 'step1_erd_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  图像已保存至 {save_path}')
    plt.close()

    return {
        'consistent_mu_idx':   [i for i in range(n_ch) if sig_mu[i]],
        'consistent_beta_idx': [i for i in range(n_ch) if sig_beta[i]],
        'mean_erd_mu':   mean_erd_mu,
        'mean_erd_beta': mean_erd_beta,
    }


if __name__ == '__main__':
    result = run_erd_analysis()
    print('\n=== 完成 ===')
    print('Mu显著通道索引:', result['consistent_mu_idx'])
    print('Beta显著通道索引:', result['consistent_beta_idx'])
