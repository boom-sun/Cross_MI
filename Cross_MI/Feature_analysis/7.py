import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, wilcoxon, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =================================
# 路径参数
# =================================
stats_root = r'E:\Datasets\4_跨场景因素研究v2\画图数据\画图结果\CSP_TSNE_STATS'
save_root = r'E:\Datasets\4_跨场景因素研究v2\画图数据\画图结果\CSP_GROUP_ANALYSIS'

Path(save_root).mkdir(parents=True, exist_ok=True)


# =================================
# 名称映射
# =================================
stim_order = ['ssvideo', 'video', 'ssmvep', 'cue']
scene_order = ['S1', 'S2']

stim_title = {
    'ssvideo': 'SSVideo',
    'video': 'Video',
    'ssmvep': 'SSMVEP',
    'cue': 'Graz'
}


# =================================
# 读取并合并所有 *_csp_stats.csv
# =================================
def load_all_stats(stats_root):
    files = sorted([
        f for f in os.listdir(stats_root)
        if f.endswith('_csp_stats.csv')
    ])

    all_df = []
    for f in files:
        path = os.path.join(stats_root, f)
        df = pd.read_csv(path)
        all_df.append(df)

    if len(all_df) == 0:
        raise ValueError('No *_csp_stats.csv files found.')

    df_all = pd.concat(all_df, ignore_index=True)

    df_all['stim'] = pd.Categorical(df_all['stim'], categories=stim_order, ordered=True)
    df_all['scene'] = pd.Categorical(df_all['scene'], categories=scene_order, ordered=True)

    return df_all


# =================================
# 工具：构造固定顺序向量
# 保证每个 subject 在每个 scene 下只有 4 个 stim 点
# =================================
def get_subject_scene_vector(df_all, subject, scene, metric):
    d = df_all[(df_all['subject'] == subject) & (df_all['scene'] == scene)]
    d = d.set_index('stim')
    vals = []
    for stim in stim_order:
        if stim in d.index:
            vals.append(d.loc[stim, metric])
        else:
            vals.append(np.nan)
    return np.array(vals, dtype=float)


# =================================
# 1) 每个被试的 stim-profile 折线图
# 修正版：严格只有 4 列
# =================================
def plot_subject_profile(df_all, metric='lda_acc'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    x = np.arange(len(stim_order))
    x_labels = [stim_title[s] for s in stim_order]

    for ax, scene in zip(axes, scene_order):
        for subj in sorted(df_all['subject'].unique()):
            y = get_subject_scene_vector(df_all, subj, scene, metric)

            if np.all(np.isnan(y)):
                continue

            ax.plot(
                x, y,
                marker='o',
                alpha=0.45,
                linewidth=1.1
            )

        # 群体均值
        mean_y = []
        for stim in stim_order:
            vals = df_all[
                (df_all['scene'] == scene) &
                (df_all['stim'] == stim)
            ][metric].values
            mean_y.append(np.nanmean(vals))

        ax.plot(
            x, mean_y,
            marker='o',
            linewidth=3,
            color='black',
            label='Mean'
        )

        ax.set_title(f'{scene}')
        ax.set_xlabel('Stim')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(metric)
    axes[1].legend()
    fig.suptitle(f'Subject profile plot ({metric})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f'subject_profile_{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =================================
# 2) 画 S2-S1 箱线图
# =================================
def compute_scene_diff(df_all, metric='lda_acc'):
    rows = []

    for subj in sorted(df_all['subject'].unique()):
        for stim in stim_order:
            d1 = df_all[
                (df_all['subject'] == subj) &
                (df_all['scene'] == 'S1') &
                (df_all['stim'] == stim)
            ]
            d2 = df_all[
                (df_all['subject'] == subj) &
                (df_all['scene'] == 'S2') &
                (df_all['stim'] == stim)
            ]

            if len(d1) == 0 or len(d2) == 0:
                continue

            diff = d2.iloc[0][metric] - d1.iloc[0][metric]
            rows.append({
                'subject': subj,
                'stim': stim,
                'diff': diff
            })

    diff_df = pd.DataFrame(rows)
    diff_df['stim'] = pd.Categorical(diff_df['stim'], categories=stim_order, ordered=True)
    return diff_df


def plot_scene_diff_boxplot(diff_df, metric='lda_acc'):
    data = [diff_df[diff_df['stim'] == s]['diff'].values for s in stim_order]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=[stim_title[s] for s in stim_order], showfliers=True)
    plt.axhline(0, linestyle='--', color='gray', linewidth=1.2)
    plt.ylabel(f'S2 - S1 ({metric})')
    plt.title(f'Scene difference boxplot ({metric})')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f'scene_diff_boxplot_{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =================================
# 3) Fisher vs LDA Acc 相关图
# =================================
def plot_fisher_acc_corr(df_all):
    x = df_all['fisher'].values
    y = df_all['lda_acc'].values

    rho, p = spearmanr(x, y)

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=36, alpha=0.75)
    plt.xlabel('Fisher')
    plt.ylabel('LDA Acc')
    plt.title(f'Fisher vs LDA Acc\nSpearman r={rho:.3f}, p={p:.3e}')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, 'fisher_vs_lda_acc.png'), dpi=300, bbox_inches='tight')
    plt.close()

    corr_df = pd.DataFrame([{
        'spearman_rho': rho,
        'p_value': p
    }])
    corr_df.to_csv(
        os.path.join(save_root, 'fisher_vs_lda_acc_corr.csv'),
        index=False,
        encoding='utf-8-sig'
    )


# =================================
# 4) 群体均值柱状图
# =================================
def plot_group_mean_bar(df_all, metric='lda_acc'):
    mean_df = (
        df_all.groupby(['scene', 'stim'], observed=False)[metric]
        .mean()
        .reset_index()
    )

    s1 = mean_df[mean_df['scene'] == 'S1'].set_index('stim').reindex(stim_order)[metric]
    s2 = mean_df[mean_df['scene'] == 'S2'].set_index('stim').reindex(stim_order)[metric]

    x = np.arange(len(stim_order))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, s1.values, width=width, label='S1')
    plt.bar(x + width / 2, s2.values, width=width, label='S2')

    plt.xticks(x, [stim_title[s] for s in stim_order], rotation=15)
    plt.ylabel(metric)
    plt.title(f'Group mean bar plot ({metric})')
    plt.legend()
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f'group_mean_bar_{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =================================
# 5) Friedman test：比较不同 stim
# 在每个 scene 下分别做
# 输入必须是重复测量：每个被试对 4 个 stim 都有值
# =================================
def run_friedman_test(df_all, metric='lda_acc'):
    rows = []

    for scene in scene_order:
        subject_vectors = []

        valid_subjects = []
        for subj in sorted(df_all['subject'].unique()):
            y = get_subject_scene_vector(df_all, subj, scene, metric)
            if np.any(np.isnan(y)):
                continue
            subject_vectors.append(y)
            valid_subjects.append(subj)

        subject_vectors = np.array(subject_vectors)

        if len(subject_vectors) < 2:
            rows.append({
                'scene': scene,
                'metric': metric,
                'n_subjects': len(subject_vectors),
                'friedman_stat': np.nan,
                'p_value': np.nan
            })
            continue

        stat, p = friedmanchisquare(
            subject_vectors[:, 0],
            subject_vectors[:, 1],
            subject_vectors[:, 2],
            subject_vectors[:, 3]
        )

        rows.append({
            'scene': scene,
            'metric': metric,
            'n_subjects': len(subject_vectors),
            'friedman_stat': stat,
            'p_value': p
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(
        os.path.join(save_root, f'friedman_test_{metric}.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    return out_df


# =================================
# 6) Wilcoxon signed-rank：比较 S1 vs S2
# 对每个 stim 分别做
# =================================
def run_wilcoxon_scene_test(df_all, metric='lda_acc'):
    rows = []

    for stim in stim_order:
        s1_vals = []
        s2_vals = []

        for subj in sorted(df_all['subject'].unique()):
            d1 = df_all[
                (df_all['subject'] == subj) &
                (df_all['scene'] == 'S1') &
                (df_all['stim'] == stim)
            ]
            d2 = df_all[
                (df_all['subject'] == subj) &
                (df_all['scene'] == 'S2') &
                (df_all['stim'] == stim)
            ]

            if len(d1) == 0 or len(d2) == 0:
                continue

            s1_vals.append(d1.iloc[0][metric])
            s2_vals.append(d2.iloc[0][metric])

        s1_vals = np.array(s1_vals, dtype=float)
        s2_vals = np.array(s2_vals, dtype=float)

        if len(s1_vals) < 2:
            rows.append({
                'stim': stim,
                'metric': metric,
                'n_subjects': len(s1_vals),
                'wilcoxon_stat': np.nan,
                'p_value': np.nan,
                'median_S1': np.nan,
                'median_S2': np.nan,
                'median_diff_S2_minus_S1': np.nan
            })
            continue

        stat, p = wilcoxon(s1_vals, s2_vals)

        rows.append({
            'stim': stim,
            'metric': metric,
            'n_subjects': len(s1_vals),
            'wilcoxon_stat': stat,
            'p_value': p,
            'median_S1': np.median(s1_vals),
            'median_S2': np.median(s2_vals),
            'median_diff_S2_minus_S1': np.median(s2_vals - s1_vals)
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(
        os.path.join(save_root, f'wilcoxon_scene_test_{metric}.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    return out_df


# =================================
# 7) 被试聚类
# 目标：
# - SSVideo-sensitive
# - Scene-sensitive
# - Stable
#
# 先构造特征：
# A. ssvideo_advantage: SSVideo 相对其他 stim 的平均优势
# B. scene_sensitivity: |S2-S1| 的平均绝对变化
# C. instability: 所有条件下 metric 的标准差
#
# 然后 KMeans=3 聚类，再按簇中心规则命名
# =================================
def build_subject_cluster_features(df_all, metric='lda_acc'):
    rows = []

    for subj in sorted(df_all['subject'].unique()):
        dsub = df_all[df_all['subject'] == subj].copy()

        # 8 个条件向量
        values = []
        for scene in scene_order:
            for stim in stim_order:
                d = dsub[(dsub['scene'] == scene) & (dsub['stim'] == stim)]
                if len(d) == 0:
                    values.append(np.nan)
                else:
                    values.append(d.iloc[0][metric])
        values = np.array(values, dtype=float)

        if np.any(np.isnan(values)):
            continue

        # A. SSVideo 相对其他 stim 的优势
        ssv_vals = dsub[dsub['stim'] == 'ssvideo'][metric].values
        other_vals = dsub[dsub['stim'] != 'ssvideo'][metric].values
        ssvideo_advantage = np.mean(ssv_vals) - np.mean(other_vals)

        # B. 场景敏感性：各 stim 下 |S2-S1| 的均值
        diffs = []
        for stim in stim_order:
            d1 = dsub[(dsub['scene'] == 'S1') & (dsub['stim'] == stim)][metric].values
            d2 = dsub[(dsub['scene'] == 'S2') & (dsub['stim'] == stim)][metric].values
            if len(d1) == 0 or len(d2) == 0:
                continue
            diffs.append(abs(d2[0] - d1[0]))
        scene_sensitivity = np.mean(diffs)

        # C. 整体波动性
        instability = np.std(values)

        # D. 整体均值
        overall_mean = np.mean(values)

        rows.append({
            'subject': subj,
            'ssvideo_advantage': ssvideo_advantage,
            'scene_sensitivity': scene_sensitivity,
            'instability': instability,
            'overall_mean': overall_mean
        })

    feat_df = pd.DataFrame(rows)
    return feat_df


def assign_cluster_names(cluster_centers_df):
    # 规则命名：
    # scene_sensitivity 最大 -> Scene-sensitive
    # ssvideo_advantage 最大 -> SSVideo-sensitive
    # instability 最小 -> Stable
    #
    # 若有冲突，按顺序分配，剩下的默认 Stable

    name_map = {}

    remaining = list(cluster_centers_df.index)

    idx_scene = cluster_centers_df['scene_sensitivity'].idxmax()
    name_map[idx_scene] = 'Scene-sensitive'
    if idx_scene in remaining:
        remaining.remove(idx_scene)

    temp = cluster_centers_df.drop(index=[idx_scene], errors='ignore')
    if len(temp) > 0:
        idx_ssv = temp['ssvideo_advantage'].idxmax()
        name_map[idx_ssv] = 'SSVideo-sensitive'
        if idx_ssv in remaining:
            remaining.remove(idx_ssv)

    for idx in remaining:
        name_map[idx] = 'Stable'

    return name_map


def run_subject_clustering(df_all, metric='lda_acc'):
    feat_df = build_subject_cluster_features(df_all, metric=metric)

    X = feat_df[['ssvideo_advantage', 'scene_sensitivity', 'instability', 'overall_mean']].values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    cluster_id = kmeans.fit_predict(Xz)

    feat_df['cluster_id'] = cluster_id

    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=['ssvideo_advantage', 'scene_sensitivity', 'instability', 'overall_mean']
    )
    centers['cluster_id'] = centers.index

    name_map = assign_cluster_names(centers.set_index('cluster_id'))
    feat_df['cluster_name'] = feat_df['cluster_id'].map(name_map)
    centers['cluster_name'] = centers['cluster_id'].map(name_map)

    feat_df.to_csv(
        os.path.join(save_root, f'subject_clusters_{metric}.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    centers.to_csv(
        os.path.join(save_root, f'cluster_centers_{metric}.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    return feat_df, centers


def plot_subject_clusters(feat_df, metric='lda_acc'):
    color_map = {
        'SSVideo-sensitive': '#F58518',
        'Scene-sensitive': '#54A24B',
        'Stable': '#4C78A8'
    }

    plt.figure(figsize=(7, 6))

    for cname in ['SSVideo-sensitive', 'Scene-sensitive', 'Stable']:
        d = feat_df[feat_df['cluster_name'] == cname]
        if len(d) == 0:
            continue

        plt.scatter(
            d['ssvideo_advantage'],
            d['scene_sensitivity'],
            s=70,
            alpha=0.85,
            color=color_map[cname],
            label=cname
        )

        for _, row in d.iterrows():
            plt.text(
                row['ssvideo_advantage'],
                row['scene_sensitivity'],
                f" {row['subject']}",
                fontsize=8
            )

    plt.xlabel('SSVideo advantage')
    plt.ylabel('Scene sensitivity')
    plt.title(f'Subject clustering ({metric})')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f'subject_clustering_{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =================================
# 主流程
# =================================
if __name__ == '__main__':
    df_all = load_all_stats(stats_root)

    # 保存总表
    all_csv_path = os.path.join(save_root, 'all_subjects_csp_stats.csv')
    df_all.to_csv(all_csv_path, index=False, encoding='utf-8-sig')
    print(f'saved -> {all_csv_path}')

    # 1. profile plot（修正版）
    plot_subject_profile(df_all, metric='lda_acc')
    plot_subject_profile(df_all, metric='fisher')

    # 2. scene diff 箱线图
    diff_lda = compute_scene_diff(df_all, metric='lda_acc')
    diff_fisher = compute_scene_diff(df_all, metric='fisher')

    diff_lda.to_csv(os.path.join(save_root, 'scene_diff_lda_acc.csv'), index=False, encoding='utf-8-sig')
    diff_fisher.to_csv(os.path.join(save_root, 'scene_diff_fisher.csv'), index=False, encoding='utf-8-sig')

    plot_scene_diff_boxplot(diff_lda, metric='lda_acc')
    plot_scene_diff_boxplot(diff_fisher, metric='fisher')

    # 3. Fisher vs Acc 相关
    plot_fisher_acc_corr(df_all)

    # 4. group mean bar
    plot_group_mean_bar(df_all, metric='lda_acc')
    plot_group_mean_bar(df_all, metric='fisher')

    # 5. Friedman test
    friedman_lda = run_friedman_test(df_all, metric='lda_acc')
    friedman_fisher = run_friedman_test(df_all, metric='fisher')
    print('\nFriedman test (lda_acc):')
    print(friedman_lda)
    print('\nFriedman test (fisher):')
    print(friedman_fisher)

    # 6. Wilcoxon S1 vs S2
    wilcoxon_lda = run_wilcoxon_scene_test(df_all, metric='lda_acc')
    wilcoxon_fisher = run_wilcoxon_scene_test(df_all, metric='fisher')
    print('\nWilcoxon scene test (lda_acc):')
    print(wilcoxon_lda)
    print('\nWilcoxon scene test (fisher):')
    print(wilcoxon_fisher)

    # 7. 被试聚类
    feat_df_lda, centers_lda = run_subject_clustering(df_all, metric='lda_acc')
    feat_df_fisher, centers_fisher = run_subject_clustering(df_all, metric='fisher')

    plot_subject_clusters(feat_df_lda, metric='lda_acc')
    plot_subject_clusters(feat_df_fisher, metric='fisher')

    print('\nSubject clustering (lda_acc):')
    print(feat_df_lda[['subject', 'cluster_name', 'ssvideo_advantage', 'scene_sensitivity', 'instability', 'overall_mean']])

    print('\nSubject clustering (fisher):')
    print(feat_df_fisher[['subject', 'cluster_name', 'ssvideo_advantage', 'scene_sensitivity', 'instability', 'overall_mean']])

    print('\ndone')