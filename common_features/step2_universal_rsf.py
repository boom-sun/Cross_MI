"""
Step 2: 通用 Riemannian 空间滤波器 (Universal RSF)

目标: 学习一组跨被试/跨会话通用的空间滤波器, 使滤波后的信号在所有域中
      都能最大化区分两类运动想象。

方法:
  1. 对所有被试/会话计算类条件协方差矩阵的 Riemannian 均值 (类-1和类-2各一个)
  2. 将多个域的类条件均值再次做 Riemannian 均值聚合 → 全局类均值 C1_global, C2_global
  3. 用现有 RSF 算法求解最优空间滤波器 W_universal: 最大化 C1_global 和 C2_global 的区分度
  4. 分析 W_universal 的空间拓扑 (投影到原始通道空间)
  5. 可视化滤波器权重热图

输出:
  - results/step2_universal_rsf.npz  (W_universal, C1_global, C2_global)
  - results/step2_filter_weights.png (滤波器权重可视化)
  - results/step2_domain_cov_analysis.png (协方差矩阵域间差异分析)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_all_subjects, FS

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULT_DIR, exist_ok=True)


def bandpass_filter(data, fs, fmin=8, fmax=30, order=4):
    """
    带通滤波
    data: (trials, channels, timepoints)
    """
    b, a = butter(order, [fmin / (fs/2), fmax / (fs/2)], btype='band')
    return filtfilt(b, a, data, axis=-1)


def lwf_cov(X):
    """
    Ledoit-Wolf 正则化协方差矩阵估计
    X: (channels, timepoints)
    """
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf()
    lw.fit(X.T)
    return lw.covariance_


def compute_class_riemannian_mean(data, label, cls):
    """
    计算单个域某类别协方差矩阵的 Riemannian 均值
    data: (trials, channels, timepoints)
    """
    from pyriemann.utils.covariance import covariances
    from pyriemann.utils.mean import mean_covariance

    X_cls = data[label == cls]
    covs = covariances(X_cls, estimator='lwf')
    return mean_covariance(covs, metric='riemann')


def riemann_mean_of_means(list_of_covs):
    """对多个协方差矩阵求 Riemannian 均值"""
    from pyriemann.utils.mean import mean_covariance
    covs = np.array(list_of_covs)
    return mean_covariance(covs, metric='riemann')


def run_universal_rsf(subjects=None, n_filters=8):
    """主流程"""
    print('=== Step 2: 通用 Riemannian 空间滤波器 ===')
    dataset = load_all_subjects(subjects=subjects, sessions=(1, 2), classes=(1, 2))

    # MI时间段: 2-5s (500-1250点)
    mi_start = int(2.0 * FS)
    mi_end   = int(5.0 * FS)

    class0_means = []  # 每个域的类0 Riemannian均值
    class1_means = []  # 每个域的类1 Riemannian均值
    domain_keys  = []

    for (sub, sess), (data, label) in dataset.items():
        print(f'  S{sub:02d} Session{sess}: 计算类条件Riemannian均值...')
        # 截取想象段 + 带通滤波
        X = data[:, :, mi_start:mi_end]
        X = bandpass_filter(X, FS, 8, 30)

        try:
            m0 = compute_class_riemannian_mean(X, label, 0)
            m1 = compute_class_riemannian_mean(X, label, 1)
            class0_means.append(m0)
            class1_means.append(m1)
            domain_keys.append((sub, sess))
        except Exception as e:
            print(f'    跳过: {e}')

    print(f'\n  共 {len(domain_keys)} 个域参与全局均值计算')

    # 计算全局类均值 (Riemannian mean of means)
    C0_global = riemann_mean_of_means(class0_means)
    C1_global = riemann_mean_of_means(class1_means)
    print('  全局类均值计算完成')

    # 用 RSF 算法求通用空间滤波器
    sys.path.insert(0, r'E:\Code\Cross\Cross_MI\auxiliary')
    from rsf import optimizeRiemann
    W_universal, _ = optimizeRiemann(C0_global, C1_global, N=n_filters, maxiter=3000)
    print(f'  通用空间滤波器 W_universal shape: {W_universal.shape}')

    # 分析滤波器的 activation pattern (A = C @ W @ inv(W.T @ C @ W))
    C_mean = riemann_mean_of_means([C0_global, C1_global])
    WtCW = W_universal.T @ C_mean @ W_universal
    A = C_mean @ W_universal @ np.linalg.inv(WtCW)  # activation pattern

    # 域间协方差距离分析
    from pyriemann.utils.distance import distance
    n_domains = len(class0_means)
    dist_matrix_cls0 = np.zeros((n_domains, n_domains))
    dist_matrix_cls1 = np.zeros((n_domains, n_domains))
    for i in range(n_domains):
        for j in range(n_domains):
            dist_matrix_cls0[i,j] = distance(class0_means[i], class0_means[j], metric='riemann')
            dist_matrix_cls1[i,j] = distance(class1_means[i], class1_means[j], metric='riemann')

    # 保存结果
    np.savez(os.path.join(RESULT_DIR, 'step2_universal_rsf.npz'),
             W_universal=W_universal,
             activation_pattern=A,
             C0_global=C0_global,
             C1_global=C1_global,
             class0_means=np.array(class0_means),
             class1_means=np.array(class1_means),
             dist_matrix_cls0=dist_matrix_cls0,
             dist_matrix_cls1=dist_matrix_cls1,
             domain_keys=np.array(domain_keys))
    print('  结果已保存至 results/step2_universal_rsf.npz')

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('通用Riemannian空间滤波器 (Universal RSF)', fontsize=14)

    # 1. 类0和类1全局协方差矩阵
    ax = axes[0, 0]
    im = ax.imshow(C0_global, cmap='RdBu_r', aspect='auto')
    ax.set_title('全局类0协方差矩阵 (左手MI)')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(C1_global, cmap='RdBu_r', aspect='auto')
    ax.set_title('全局类1协方差矩阵 (右手MI)')
    plt.colorbar(im, ax=ax)

    # 2. 两类差异
    ax = axes[0, 2]
    diff = C0_global - C1_global
    vmax = np.abs(diff).max()
    im = ax.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_title('类间协方差差异 (C0 - C1)')
    plt.colorbar(im, ax=ax)

    # 3. 通用空间滤波器权重 (前8个)
    ax = axes[1, 0]
    im = ax.imshow(W_universal.T, cmap='RdBu_r', aspect='auto')
    ax.set_title(f'通用空间滤波器权重 W ({n_filters}×60)')
    ax.set_xlabel('通道编号')
    ax.set_ylabel('滤波器编号')
    plt.colorbar(im, ax=ax)

    # 4. Activation pattern
    ax = axes[1, 1]
    im = ax.imshow(A.T, cmap='RdBu_r', aspect='auto')
    ax.set_title('Activation Pattern (可解释性)')
    ax.set_xlabel('通道编号')
    ax.set_ylabel('模式编号')
    plt.colorbar(im, ax=ax)

    # 5. 域间Riemannian距离热图 (类0)
    ax = axes[1, 2]
    im = ax.imshow(dist_matrix_cls0, cmap='viridis', aspect='auto')
    labels_tick = [f'S{k[0]}S{k[1]}' for k in domain_keys]
    n = len(labels_tick)
    if n <= 20:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels_tick, rotation=45, fontsize=6)
        ax.set_yticklabels(labels_tick, fontsize=6)
    ax.set_title('域间Riemannian距离 (类0协方差)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, 'step2_filter_weights.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  图像已保存至 {save_path}')
    plt.close()

    # 打印主要结论
    avg_within_dist0 = (dist_matrix_cls0.sum() - np.trace(dist_matrix_cls0)) / (n_domains*(n_domains-1))
    avg_within_dist1 = (dist_matrix_cls1.sum() - np.trace(dist_matrix_cls1)) / (n_domains*(n_domains-1))
    from pyriemann.utils.distance import distance as rd
    between_dist = rd(C0_global, C1_global, metric='riemann')
    print(f'\n  域内类0平均Riemannian距离: {avg_within_dist0:.4f}')
    print(f'  域内类1平均Riemannian距离: {avg_within_dist1:.4f}')
    print(f'  全局两类间Riemannian距离:  {between_dist:.4f}')
    print(f'  (类间/类内越大, 共性特征越强)')


if __name__ == '__main__':
    run_universal_rsf(n_filters=8)
    print('\n=== 完成 ===')
