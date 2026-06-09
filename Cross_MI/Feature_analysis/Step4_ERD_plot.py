import os
import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path
from itertools import combinations
from matplotlib.patches import Patch

# ======================= 1. 参数与设置 (保持不变) =======================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
CLASS_CHOOSE_LIST = [1, 2]
stim_name = ('cue', 'ssmvep', 'ssvideo', 'video')
stim_name_new = ('Graz', 'SSMVEP', 'SSVideo', 'Video')
Class_name = {1: 'Left', 2: 'Right'}
save_dir = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\ERD结果'  # 使用新的文件夹以避免覆盖
save_root_TF = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\时频图数据'
Path(save_dir).mkdir(parents=True, exist_ok=True)

sen_name_map = {'s1': 'Scene 1', 's2': 'Scene 2'}
significance = 0.05
ch_names = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2'
]
channels_to_plot = ['C3', 'C4', 'OZ']
CH_ind = [ch_names.index(ch) for ch in channels_to_plot]
bands = [('θ(4–8)', 4, 8), ('α(8–13)', 8, 13), ('low-β(13–20)', 13, 20), ('high-β(20–30)', 20, 30)]
band_names = [b[0] for b in bands]
stim_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


# ======================= 2. 显著性标注辅助函数 (已优化) =======================
def add_stat_annotation_downward(ax, x1, x2, y_min, p_val):
    if p_val >= significance: return y_min
    p_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'

    ### 微调点 1: 缩小标注线之间的垂直间距 ###
    # line_height 决定了标注线的“厚度”和它与下一条线的距离
    line_height = 0.03  # 从 0.05 减小到 0.03
    y_start = y_min - line_height  # 从柱子底部留出一点空隙

    ax.plot([x1, x1, x2, x2], [y_start, y_start - line_height, y_start - line_height, y_start], lw=1.5, c='black')
    ax.text((x1 + x2) / 2, y_start - line_height, p_text, ha='center', va='top', color='black', fontsize=12)

    return y_start - line_height  # 返回新的最低点


# ======================= 3. 最终微调的核心绘图函数 =======================
def plot_channel_clusters(ch_name, data_for_plot, title_prefix, ylabel, save_filename):
    # 画布尺寸保持不变，通过调整间距来使图像更紧凑
    fig, ax = plt.subplots(figsize=(24, 6))

    # --- 数据准备 ---
    bar_positions, bar_heights, bar_errors = [], [], []
    all_p_values, stat_test_info = [], []

    ### 微调点 2: 缩小X轴上柱子和簇之间的间距 ###
    bar_width, stim_gap, cluster_gap = 0.8, 0.1, 1.2  # stim_gap从0.2->0.1, cluster_gap从2.0->1.2
    current_x = 0

    # 循环准备绘图数据 (逻辑不变)
    for scene in ['s1', 's2']:
        for band_name in band_names:
            stim_data_in_cluster = [data_for_plot[scene][band_name].get(stim, np.array([])) for stim in stim_name]
            for (i, d1), (j, d2) in combinations(enumerate(stim_data_in_cluster), 2):
                if len(d1) > 1 and len(d2) > 1:
                    p = ttest_rel(d1, d2, nan_policy='omit').pvalue
                    all_p_values.append(p)
                    stat_test_info.append({'x1_idx': len(bar_positions) + i, 'x2_idx': len(bar_positions) + j,
                                           'p_idx': len(all_p_values) - 1})
            for stim_idx, data in enumerate(stim_data_in_cluster):
                bar_positions.append(current_x)
                bar_heights.append(np.nanmean(data))
                bar_errors.append(np.nanstd(data, ddof=1) / np.sqrt(len(data)) if len(data) > 0 else 0)
                current_x += bar_width + stim_gap
            current_x += cluster_gap

    for band_idx, band_name in enumerate(band_names):
        for stim_idx, stim in enumerate(stim_name):
            d1 = data_for_plot['s1'][band_name].get(stim, np.array([]))
            d2 = data_for_plot['s2'][band_name].get(stim, np.array([]))
            if len(d1) > 1 and len(d2) > 1:
                p = ttest_rel(d1, d2, nan_policy='omit').pvalue
                all_p_values.append(p)
                bar1_idx = (band_idx * 4) + stim_idx
                bar2_idx = (4 * 4) + (band_idx * 4) + stim_idx
                stat_test_info.append({'x1_idx': bar1_idx, 'x2_idx': bar2_idx, 'p_idx': len(all_p_values) - 1})

    pvals_corrected = fdrcorrection(all_p_values, alpha=significance)[1] if all_p_values else []

    # --- 绘图与美化 ---
    ax.bar(bar_positions, bar_heights, width=bar_width, color=[stim_colors[i % 4] for i in range(len(bar_positions))],
           edgecolor='black')
    ax.errorbar(bar_positions, bar_heights, yerr=[np.array(bar_errors), np.zeros_like(bar_errors)],
                fmt='none', ecolor='black', capsize=5, elinewidth=1.5)

    # 绘制显著性标注 (逻辑不变, 但会因函数内line_height变小而更紧凑)
    stat_test_info.sort(key=lambda item: abs(bar_positions[item['x1_idx']] - bar_positions[item['x2_idx']]))
    annotation_heights = {}
    for test in stat_test_info:
        p = pvals_corrected[test['p_idx']]
        if p < significance:
            x1_pos, x2_pos = bar_positions[test['x1_idx']], bar_positions[test['x2_idx']]
            indices = range(min(test['x1_idx'], test['x2_idx']), max(test['x1_idx'], test['x2_idx']) + 1)
            base_y = min([bar_heights[i] - bar_errors[i] for i in indices])
            y_level = min([annotation_heights.get(i, base_y) for i in indices])
            new_y_min = add_stat_annotation_downward(ax, x1_pos, x2_pos, y_level, p)
            for i in indices:
                annotation_heights[i] = new_y_min

    # --- 布局与美化 ---
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_title(f'{title_prefix} - Channel: {ch_name}', fontsize=20, pad=20)
    ax.set_ylabel(ylabel, fontsize=18)

    ### 微调点 3: 统一并固定Y轴范围 ###
    ax.set_ylim(-1.6, 0)

    # X轴主标签
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([stim_name_new[i % 4] for i in range(len(bar_positions))], rotation=45, ha='right')
    ax.tick_params(axis='x', which='major', length=0)

    # 健壮的二级标签和图例定位 (逻辑不变)
    transform = ax.get_xaxis_transform()
    cluster_centers = [np.mean(bar_positions[i:i + 4]) for i in range(0, len(bar_positions), 4)]
    for i, pos in enumerate(cluster_centers):
        ax.text(pos, -0.18, band_names[i % 4], ha='center', va='top', fontsize=18, weight='bold', transform=transform)
    ax.text(np.mean(cluster_centers[:4]), -0.25, sen_name_map['s1'], ha='center', va='top', fontsize=18, weight='bold',
            transform=transform)
    ax.text(np.mean(cluster_centers[4:]), -0.25, sen_name_map['s2'], ha='center', va='top', fontsize=18, weight='bold',
            transform=transform)

    legend_elements = [Patch(facecolor=c, label=n) for c, n in zip(stim_colors, stim_name_new)]
    ax.legend(handles=legend_elements, loc='lower right',fontsize=18)

    # 最终布局方案 (逻辑不变)
    plt.savefig(os.path.join(save_dir, save_filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] {save_filename}")


# ======================= 4. 主执行流程 (保持不变) =======================
if __name__ == "__main__":
    # --- 步骤1: 加载您的数据 (与您的代码一致) ---
    results_by_class = {1: {}, 2: {}}
    try:
        for class_choose in CLASS_CHOOSE_LIST:
            for stim in stim_name:
                tf_path = os.path.join(save_root_TF, f"times+freqs_{stim}_class{class_choose}.mat")
                mat_content = hdf5storage.loadmat(tf_path)
                Times, Freqs = mat_content['times'].flatten(), mat_content['freqs'].flatten()
                for sen in ['s1', 's2']:
                    p_tf = os.path.join(save_root_TF, f"TF_{stim}_class{class_choose}.mat")
                    tf_data = hdf5storage.loadmat(p_tf)['tf']
                    if sen in tf_data:
                        power_task = tf_data[sen]
                        erds_full_freq = np.mean(power_task, axis=2)
                        erds_selected_ch = erds_full_freq[:, :, CH_ind]
                        band_erds = np.array([np.mean(erds_selected_ch[:, (Freqs >= fmin) & (Freqs < fmax), :], axis=1)
                                              for _, fmin, fmax in bands]).transpose(1, 2, 0)
                        results_by_class[class_choose][(sen, stim)] = band_erds
    except Exception as e:
        print(f"数据加载失败: {e}. 将使用随机数据生成示例图片...")
        for class_choose in CLASS_CHOOSE_LIST:
            for stim in stim_name:
                for sen in ['s1', 's2']:
                    random_data = (np.random.rand(10, len(channels_to_plot), len(bands)) * 0.8) - 0.9
                    results_by_class[class_choose][(sen, stim)] = random_data

    # --- 步骤2: 重新组织数据以适应新的绘图函数 (保持不变) ---
    final_data = {}
    for cid in Class_name:
        final_data[cid] = {ch: {'s1': {b: {} for b in band_names}, 's2': {b: {} for b in band_names}} for ch in
                           channels_to_plot}
    for cid, data_dict in results_by_class.items():
        for (sen, stim), raw_data in data_dict.items():
            for ch_idx, ch_name in enumerate(channels_to_plot):
                for band_idx, band_name in enumerate(band_names):
                    final_data[cid][ch_name][sen][band_name][stim] = raw_data[:, ch_idx, band_idx]

    # --- 步骤3 & 4: 绘图流程 (保持不变) ---
    for cid, name in Class_name.items():
        for ch in channels_to_plot:
            plot_channel_clusters(ch, final_data[cid][ch], f'ERSP ({name})', 'ERSP',
                                  f'ERSP_Clustered_{name}_{ch}.png')

    diff_data = {ch: {'s1': {b: {} for b in band_names}, 's2': {b: {} for b in band_names}} for ch in channels_to_plot}
    for ch in channels_to_plot:
        for sen in ['s1', 's2']:
            for band in band_names:
                for stim in stim_name:
                    d1 = final_data[1][ch][sen][band].get(stim)
                    d2 = final_data[2][ch][sen][band].get(stim)
                    if d1 is not None and d2 is not None and len(d1) > 0 and len(d2) > 0:
                        n = min(len(d1), len(d2))
                        diff_data[ch][sen][band][stim] = d2[:n] - d1[:n]
    for ch in channels_to_plot:
        plot_channel_clusters(ch, diff_data[ch], 'ERSP Difference (Right - Left)', 'Power Difference (dB)',
                              f'ERSP_Diff_Clustered_{ch}.png')

