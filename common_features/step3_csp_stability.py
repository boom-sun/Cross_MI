"""
Step 3: CSP 空间滤波器跨域稳定性分析

目标: 量化 CSP 空间滤波器在不同被试/会话之间的相似性,
      找到高度稳定(共性)的滤波器方向。

方法:
  1. 对每个域独立训练 CSP (n=4对滤波器, 共8个)
  2. 计算所有域 CSP 滤波器的两两余弦相似度
  3. 对滤波器矩阵做 PCA/主成分聚合 → 提取跨域共性方向
  4. 计算共性滤波器在每个域上的区分度 (Fisher's ratio)
  5. 对比个性(per-domain) vs. 共性(universal) 滤波器性能

输出:
  - results/step3_csp_stability.npz   (所有域CSP滤波器 + 稳定性指标)
  - results/step3_stability_analysis.png (可视化)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from pyriemann.spatialfilters import CSP
from pyriemann.utils.covariance import covariances
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_all_subjects, FS

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULT_DIR, exist_ok=True)


def bandpass_filter(data, fs, fmin=8, fmax=30, order=4):
    b, a = butter(order, [fmin/(fs/2), fmax/(fs/2)], btype='band')
    return filtfilt(b, a, data, axis=-1)


def train_csp(data, label, n_filters=4):
    """
    训练 Riemannian CSP (RCSP)
    返回: W (n_ch, 2*n_filters) 滤波器矩阵
    """
    csp = CSP(nfilter=n_filters, metric='riemann')
    covs = covariances(data, estimator='lwf')
    csp.fit(covs, label)
    return csp.filters_.T  # (n_ch, 2*n_filters)


def cosine_similarity_matrix(W1, W2):
    """
    计算两个滤波器组的最大匹配余弦相似度 (匈牙利匹配)
    W1, W2: (n_ch, n_filters)
    返回: 平均最大相似度 (0~1)
    """
    from scipy.optimize import linear_sum_assignment
    n1 = W1.shape[1]
    n2 = W2.shape[1]
    # 归一化
    W1n = W1 / (np.linalg.norm(W1, axis=0, keepdims=True) + 1e-10)
    W2n = W2 / (np.linalg.norm(W2, axis=0, keepdims=True) + 1e-10)
    # 相似度矩阵 (考虑符号翻转)
    sim = np.abs(W1n.T @ W2n)  # (n1, n2)
    row_ind, col_ind = linear_sum_assignment(-sim)
    return sim[row_ind, col_ind].mean()


def fisher_ratio(data, label, W):
    """
    计算滤波器 W 投影后特征的 Fisher ratio (类间方差/类内方差)
    data: (trials, ch, time)
    W: (ch, n_filters)
    """
    # 滤波+对数方差特征
    X_filt = np.einsum('ij,kjl->kil', W.T, data)  # (trials, n_filt, time)
    log_var = np.log(np.var(X_filt, axis=-1) + 1e-10)  # (trials, n_filt)

    classes = np.unique(label)
    mu_all = np.mean(log_var, axis=0)
    sw = sum(np.cov(log_var[label == c].T) for c in classes) / len(classes)
    sb = sum(len(label[label == c]) * np.outer(np.mean(log_var[label == c], axis=0) - mu_all,
                                                np.mean(log_var[label == c], axis=0) - mu_all)
             for c in classes) / len(label)
    # Scalar Fisher ratio: trace(sb) / (trace(sw) + 1e-10)
    return np.trace(sb) / (np.trace(sw) + 1e-10)


def run_csp_stability(subjects=None):
    """主流程"""
    print('=== Step 3: CSP 跨域稳定性分析 ===')
    dataset = load_all_subjects(subjects=subjects, sessions=(1, 2), classes=(1, 2))

    mi_start = int(2.0 * FS)
    mi_end   = int(5.0 * FS)
    N_FILTERS = 4

    all_W = []      # 每个域的 CSP 滤波器 (n_ch, 2*N_FILTERS)
    domain_keys = []
    all_fisher  = []

    for (sub, sess), (data, label) in dataset.items():
        print(f'  S{sub:02d} Session{sess}: 训练CSP...')
        X = data[:, :, mi_start:mi_end]
        X = bandpass_filter(X, FS, 8, 30)
        try:
            W = train_csp(X, label, N_FILTERS)
            fr = fisher_ratio(X, label, W)
            all_W.append(W)
            all_fisher.append(fr)
            domain_keys.append((sub, sess))
        except Exception as e:
            print(f'    跳过: {e}')

    n_domains = len(all_W)
    print(f'  共 {n_domains} 个域')

    # 计算两两余弦相似度矩阵
    sim_matrix = np.zeros((n_domains, n_domains))
    for i in range(n_domains):
        for j in range(n_domains):
            sim_matrix[i, j] = cosine_similarity_matrix(all_W[i], all_W[j])

    # 每个滤波器组的平均跨域相似度 (该域与其他所有域)
    stability_scores = np.array([
        (sim_matrix[i].sum() - sim_matrix[i, i]) / (n_domains - 1)
        for i in range(n_domains)
    ])

    # PCA 提取共性滤波器方向
    # 将所有域的 CSP 滤波器列向量堆叠
    all_filters_flat = np.hstack([W for W in all_W])  # (n_ch, n_domains * 2*N_FILTERS)
    pca = PCA(n_components=min(16, all_filters_flat.shape[1]))
    pca.fit(all_filters_flat.T)
    W_common = pca.components_.T  # (n_ch, n_components) — 共性方向
    explained_var = pca.explained_variance_ratio_

    print(f'  前8个PCA成分解释方差: {explained_var[:8].sum()*100:.1f}%')

    # 计算共性滤波器在每个域的Fisher ratio
    common_fisher = []
    for (sub, sess), (data, label) in dataset.items():
        X = data[:, :, mi_start:mi_end]
        X = bandpass_filter(X, FS, 8, 30)
        try:
            fr = fisher_ratio(X, label, W_common[:, :2*N_FILTERS])
            common_fisher.append(fr)
        except:
            pass

    print(f'  个性CSP Fisher ratio 均值: {np.mean(all_fisher):.4f} ± {np.std(all_fisher):.4f}')
    print(f'  共性CSP Fisher ratio 均值: {np.mean(common_fisher):.4f} ± {np.std(common_fisher):.4f}')

    # 保存结果
    np.savez(os.path.join(RESULT_DIR, 'step3_csp_stability.npz'),
             all_W=np.array(all_W),
             W_common=W_common,
             sim_matrix=sim_matrix,
             stability_scores=stability_scores,
             all_fisher=np.array(all_fisher),
             common_fisher=np.array(common_fisher),
             explained_var=explained_var,
             domain_keys=np.array(domain_keys))
    print('  结果已保存至 results/step3_csp_stability.npz')

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('CSP 空间滤波器跨域稳定性分析', fontsize=14)

    # 1. 域间余弦相似度热图
    ax = axes[0, 0]
    im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    labels_tick = [f'S{k[0]}S{k[1]}' for k in domain_keys]
    n = len(labels_tick)
    if n <= 20:
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels_tick, rotation=45, fontsize=6)
        ax.set_yticklabels(labels_tick, fontsize=6)
    ax.set_title(f'CSP滤波器余弦相似度矩阵\n(均值={sim_matrix[sim_matrix<1].mean():.3f})')
    plt.colorbar(im, ax=ax)

    # 2. 域稳定性分数
    ax = axes[0, 1]
    sorted_idx = np.argsort(stability_scores)[::-1]
    colors = plt.cm.RdYlGn(stability_scores[sorted_idx])
    ax.bar(range(n_domains), stability_scores[sorted_idx], color=colors)
    ax.set_xticks(range(n_domains))
    ax.set_xticklabels([labels_tick[i] for i in sorted_idx], rotation=45, fontsize=6)
    ax.set_title('各域CSP稳定性分数 (跨域平均余弦相似度)')
    ax.set_ylabel('稳定性分数')

    # 3. PCA解释方差
    ax = axes[0, 2]
    cumvar = np.cumsum(explained_var)
    ax.bar(range(1, len(explained_var)+1), explained_var*100, alpha=0.7, label='个体方差')
    ax.plot(range(1, len(explained_var)+1), cumvar*100, 'r-o', label='累积方差')
    ax.axhline(80, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('PCA提取共性方向 (解释方差)')
    ax.set_xlabel('主成分编号')
    ax.set_ylabel('解释方差(%)')
    ax.legend(fontsize=8)

    # 4. 共性CSP滤波器权重 (前8个)
    ax = axes[1, 0]
    im = ax.imshow(W_common[:, :8].T, cmap='RdBu_r', aspect='auto')
    ax.set_title('共性CSP滤波器 W_common (前8个)')
    ax.set_xlabel('通道编号')
    ax.set_ylabel('滤波器编号')
    plt.colorbar(im, ax=ax)

    # 5. 个性 vs. 共性 Fisher ratio 对比
    ax = axes[1, 1]
    ax.scatter(range(len(all_fisher)), sorted(all_fisher), label='个性CSP', alpha=0.7, color='steelblue')
    ax.scatter(range(len(common_fisher)), sorted(common_fisher), label='共性CSP', alpha=0.7, color='tomato')
    ax.set_title('个性 vs. 共性 CSP Fisher Ratio')
    ax.set_xlabel('域排序')
    ax.set_ylabel('Fisher Ratio')
    ax.legend()

    # 6. Fisher ratio 对比箱线图
    ax = axes[1, 2]
    bp = ax.boxplot([all_fisher, common_fisher], labels=['个性CSP\n(per-domain)', '共性CSP\n(universal)'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('tomato')
    ax.set_title('Fisher Ratio 分布对比')
    ax.set_ylabel('Fisher Ratio')
    # 统计显著性
    from scipy.stats import wilcoxon
    if len(all_fisher) == len(common_fisher):
        _, pval = wilcoxon(all_fisher, common_fisher)
        ax.text(0.5, 0.95, f'Wilcoxon p={pval:.4f}', transform=ax.transAxes,
                ha='center', va='top', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, 'step3_stability_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  图像已保存至 {save_path}')
    plt.close()


if __name__ == '__main__':
    run_csp_stability()
    print('\n=== 完成 ===')
