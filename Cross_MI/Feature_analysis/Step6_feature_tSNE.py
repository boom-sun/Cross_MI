import os
from pathlib import Path

import numpy as np
import pandas as pd
import hdf5storage
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from eeg_filter import ERPs_Filter


# =================================
# 参数
# =================================
datatype = 1
freqwindow = [8, 30]
TaskDuration = 4
RestDuration = 2

data_4_filepath = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2处理后数据'
save_root = r'E:\Datasets\4_跨场景因素研究v2\画图数据\画图结果\CSP_TSNE_STATS'

stim_name = ('ssvideo', 'video', 'ssmvep', 'cue')
stim_show_name = ('SSVideo', 'Video', 'SSMVEP', 'Graz')
stim_title = {
    'ssvideo': 'SSVideo',
    'video': 'Video',
    'ssmvep': 'SSMVEP',
    'cue': 'Graz'
}
stim_short = {
    'ssvideo': 'SSV',
    'video': 'VID',
    'ssmvep': 'MVEP',
    'cue': 'GRAZ'
}

subjectchoose = list(range(1, 38))

n_csp_pairs = 3
tsne_perplexity = 20
tsne_random_state = 42

# 可视化参数
point_size = 12
point_alpha = 0.5
center_size = 50
line_width = 1


# =================================
# 基础函数
# =================================
def pfx(s):
    return f"{datatype}S0{s}" if s <= 9 else f"{datatype}S{s}"


def load_block(path_mat):
    M = hdf5storage.loadmat(path_mat)
    data = M['data']
    label = np.array(M['label']).ravel()
    fs = float(np.squeeze(M['fs']))
    return data, label, fs


def crop_task_window(data, fs, rest_dur=2, task_dur=4):
    t0 = int(rest_dur * fs)
    t1 = int((rest_dur + task_dur) * fs)
    return data[:, t0:t1, :]


def avg_norm_cov(X):
    # X: (n_trials, n_channels, n_times)
    C = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[0]):
        Xi = X[i]
        Ci = Xi @ Xi.T
        C += Ci / np.trace(Ci)
    return C / X.shape[0]


def compute_csp_filters(X_left, X_right, n_pairs=3):
    C1 = avg_norm_cov(X_left)
    C2 = avg_norm_cov(X_right)

    eigvals, eigvecs = eigh(C1, C1 + C2)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    W = np.hstack([eigvecs[:, :n_pairs], eigvecs[:, -n_pairs:]])
    return W


def extract_csp_features(X, W):
    # X: (n_trials, n_channels, n_times)
    feat = []
    for i in range(X.shape[0]):
        Z = W.T @ X[i]
        var = np.var(Z, axis=1)
        var = var / np.sum(var)
        feat.append(np.log(var))
    return np.asarray(feat)


def run_tsne(feat):
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        init='pca',
        learning_rate='auto',
        random_state=tsne_random_state
    )
    emb = tsne.fit_transform(feat)
    return emb


def get_center(X):
    return X.mean(axis=0)


def mean_sq_radius(X, mu):
    return np.mean(np.sum((X - mu) ** 2, axis=1))


# =================================
# 统计量：全部在 CSP 原始特征空间算
# =================================
def compute_stats(feat, label):
    X_left = feat[label == 1]
    X_right = feat[label == 2]

    mu_left = get_center(X_left)
    mu_right = get_center(X_right)

    cent_dist = np.linalg.norm(mu_left - mu_right)

    within_left = mean_sq_radius(X_left, mu_left)
    within_right = mean_sq_radius(X_right, mu_right)
    within = 0.5 * (within_left + within_right)

    fisher = (cent_dist ** 2) / (within + 1e-12)

    sil = np.nan
    if len(np.unique(label)) == 2 and len(label) >= 4:
        sil = silhouette_score(feat, label)

    lda_acc = np.nan
    if min(len(X_left), len(X_right)) >= 2:
        n_splits = min(5, len(X_left), len(X_right))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            clf = LinearDiscriminantAnalysis()
            scores = cross_val_score(clf, feat, label, cv=cv, scoring='accuracy')
            lda_acc = scores.mean()

    return {
        'cent_dist': cent_dist,
        'within': within,
        'fisher': fisher,
        'silhouette': sil,
        'lda_acc': lda_acc
    }


# =================================
# 绘图：上面是 t-SNE，下面是统计面板
# 子图 = 场景 S1 / S2
# 颜色 = 类别 Left / Right
# marker = stim
# =================================
def plot_subject_scene_split_with_stats(
    all_feat, all_label, all_scene, all_stim, subj_id, save_path, stats_df
):
    emb = run_tsne(all_feat)

    color_map = {
        1: 'tab:blue',      # Left
        2: 'tab:orange'     # Right
    }

    marker_map = {
        'ssvideo': 'o',
        'video': 's',
        'ssmvep': '^',
        'cue': 'D'
    }

    scene_list = ['S1', 'S2']

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3.3, 1.8],
        hspace=0.22,
        wspace=0.16
    )

    scatter_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    stat_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    for ax_scatter, ax_stat, sen in zip(scatter_axes, stat_axes, scene_list):
        idx_scene = (all_scene == sen)

        # ---------- 上部散点 ----------
        center_dict = {}

        for cls in [1, 2]:
            for stim in stim_name:
                idx = idx_scene & (all_label == cls) & (all_stim == stim)
                if np.sum(idx) == 0:
                    continue

                cls_name = 'Left' if cls == 1 else 'Right'

                # 原始 trial 点：淡化
                ax_scatter.scatter(
                    emb[idx, 0],
                    emb[idx, 1],
                    s=point_size,
                    alpha=point_alpha,
                    c=color_map[cls],
                    marker=marker_map[stim],
                    label=f'{cls_name}-{stim_short[stim]}'
                )

                # 中心点：放大
                center = emb[idx].mean(axis=0)
                center_dict[(cls, stim)] = center

                ax_scatter.scatter(
                    center[0],
                    center[1],
                    s=center_size,
                    c=color_map[cls],
                    marker=marker_map[stim],
                    edgecolors='k',
                    linewidths=1.3,
                    zorder=10
                )

                cls_short = 'L' if cls == 1 else 'R'
                ax_scatter.text(
                    center[0],
                    center[1],
                    f' {cls_short}-{stim_short[stim]}',
                    fontsize=9,
                    color='k',
                    zorder=11
                )

        # 同一个 stim 的 Left / Right 中心连线
        for stim in stim_name:
            if (1, stim) in center_dict and (2, stim) in center_dict:
                c1 = center_dict[(1, stim)]
                c2 = center_dict[(2, stim)]
                ax_scatter.plot(
                    [c1[0], c2[0]],
                    [c1[1], c2[1]],
                    linestyle='--',
                    linewidth=line_width,
                    color='gray',
                    alpha=0.9,
                    zorder=8
                )

        ax_scatter.set_title(f'{subj_id} | {sen}', fontsize=13)
        ax_scatter.set_xlabel('t-SNE 1')
        ax_scatter.set_ylabel('t-SNE 2')

        handles, labels = ax_scatter.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        ax_scatter.legend(uniq.values(), uniq.keys(), fontsize=8, loc='best')

        # ---------- 下部统计面板 ----------
        df_scene = stats_df[stats_df['scene'] == sen].copy()
        df_scene['stim_order'] = df_scene['stim'].map({k: i for i, k in enumerate(stim_name)})
        df_scene = df_scene.sort_values('stim_order').reset_index(drop=True)

        y = np.arange(len(df_scene))
        fisher_vals = df_scene['fisher'].values

        ax_stat.barh(y, fisher_vals, alpha=0.75)
        ax_stat.set_yticks(y)
        ax_stat.set_yticklabels([stim_title[s] for s in df_scene['stim']])
        ax_stat.invert_yaxis()
        ax_stat.set_xlabel('Fisher ratio')
        ax_stat.set_title(f'{sen} | Statistics in CSP feature space', fontsize=12)

        # 每一行写详细统计量
        x_max = np.nanmax(fisher_vals) if len(fisher_vals) > 0 else 1.0
        if not np.isfinite(x_max) or x_max <= 0:
            x_max = 1.0
        ax_stat.set_xlim(0, x_max * 1.55)

        for i, row in df_scene.iterrows():
            text = (
                f"Dist={row['cent_dist']:.3f}   "
                f"Within={row['within']:.3f}   "
                f"Sil={row['silhouette']:.3f}   "
                f"Acc={row['lda_acc']:.3f}"
            )
            ax_stat.text(
                x_max * 0.03,  # row['fisher'] + x_max * 0.03
                i,
                text,
                va='center',
                fontsize=6
            )

    fig.suptitle(
        f'{subj_id} | CSP+t-SNE visualization + statistics',
        fontsize=15
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =================================
# 主流程
# =================================
Path(save_root).mkdir(parents=True, exist_ok=True)

for s in subjectchoose:
    subj_id = pfx(s)
    print(f'===== {subj_id} =====')

    subj_feat_list = []
    subj_label_list = []
    subj_scene_list = []
    subj_stim_list = []

    # 每个 scene × stim 的统计量单独保存
    stat_rows = []

    for sen in ['S1', 'S2']:
        for stim in stim_name:
            path_mat = os.path.join(data_4_filepath, f'{subj_id}{sen}_{stim}.mat')
            if not os.path.exists(path_mat):
                continue

            data, label, fs = load_block(path_mat)

            data = ERPs_Filter(
                data,
                freqs=freqwindow,
                fs=fs,
                filterflag='filtfilt'
            )

            data = crop_task_window(
                data,
                fs=fs,
                rest_dur=RestDuration,
                task_dur=TaskDuration
            )

            # (channels, times, trials) -> (trials, channels, times)
            X = np.transpose(data, (2, 0, 1))

            X_left = X[label == 1]
            X_right = X[label == 2]

            if len(X_left) == 0 or len(X_right) == 0:
                continue

            W = compute_csp_filters(X_left, X_right, n_pairs=n_csp_pairs)
            feat = extract_csp_features(X, W)

            # 收集用于总 t-SNE 图
            subj_feat_list.append(feat)
            subj_label_list.append(label)
            subj_scene_list.append(np.array([sen] * len(label)))
            subj_stim_list.append(np.array([stim] * len(label)))

            # 原始 CSP 特征空间统计量
            stats = compute_stats(feat, label)
            stat_rows.append({
                'subject': subj_id,
                'scene': sen,
                'stim': stim,
                **stats
            })

    if len(subj_feat_list) == 0:
        continue

    all_feat = np.vstack(subj_feat_list)
    all_label = np.concatenate(subj_label_list)
    all_scene = np.concatenate(subj_scene_list)
    all_stim = np.concatenate(subj_stim_list)

    stats_df = pd.DataFrame(stat_rows)

    save_dir = os.path.join(save_root, f'sub{s}')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1) 保存统计表
    csv_path = os.path.join(save_root, f'sub{s}_csp_stats.csv')
    stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 2) 保存综合图
    fig_path = os.path.join(save_root, f'sub{s}_scene_split_csp_tsne_stats.png')
    plot_subject_scene_split_with_stats(
        all_feat=all_feat,
        all_label=all_label,
        all_scene=all_scene,
        all_stim=all_stim,
        subj_id=subj_id,
        save_path=fig_path,
        stats_df=stats_df
    )

    print(f'saved -> {csv_path}')
    print(f'saved -> {fig_path}')

print('done')