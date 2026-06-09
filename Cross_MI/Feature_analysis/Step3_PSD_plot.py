import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import hdf5storage
import os

from eeg_filter import ERPs_Filter
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.lines import Line2D

# =========================
# 参数
# =========================
datatype = 1
srate = 250
freqwindow = [1, 40]
frames = 1500
TaskDuration = 4
RestDuration = 2
tlimits = [-1000 * RestDuration, TaskDuration * 1000 - 1]  # ms
sence = 'ssmvep_hybrid'
feature = 'pearson'
fre_band = [1, 40]
significance = 0.05

ch_names = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2'
]

LOCS_FILE = 'channel_location_60_neuroscan.locs'
montage = mne.channels.read_custom_montage(LOCS_FILE)
info = mne.create_info(ch_names, sfreq=250, ch_types='eeg')
info.set_montage(montage, on_missing='warn')

channels_to_plot = ['C3', 'C4']
bands = [
    ('θ(4–8)', 4, 8),
    ('α(8–13)', 8, 13),
    ('β(13–30)', 13, 30),
]

band_colors = [
    '#fde0dd',  # θ
    '#e0f3db',  # α
    '#deebf7'   # β
]
band_alpha = 0.5

PSD_ylim = [-4, 4]
PSD_diff_ylim = [-1, 1.5]
PSD_topo_clar = [-4, 4]

RAW_STIM_ORDER = ('ssvideo', 'video', 'ssmvep', 'cue')
RAW_TO_DISPLAY = {
    'ssvideo': 'SSVideo',
    'video': 'Video',
    'ssmvep': 'SSMVEP',
    'cue': 'Graz'
}
BIG_STIM_ORDER = ['Graz', 'SSMVEP', 'SSVideo', 'Video']


# =========================
# 工具函数
# =========================
def pfx(s):
    return f"{datatype}S0{s}" if s <= 9 else f"{datatype}S{s}"


def load_block(p):
    M = hdf5storage.loadmat(p)
    return M['data'], M['label'].ravel(), float(np.squeeze(M['fs']))


def shade_bands(ax, bands_to_draw, colors=None, alpha=None, z=0.05):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#dddddd'] * len(bands_to_draw))
    if alpha is None:
        alpha = band_alpha
    for (_, fmin, fmax), col in zip(bands_to_draw, colors):
        ax.axvspan(fmin, fmax, facecolor=col, edgecolor='none', alpha=alpha, zorder=z)


def make_scientific_colors(n):
    palette = [
        '#3A6EA5',  # 蓝
        '#C44E52',  # 红
        '#85B760',  # 绿
        '#FF9F4A',  # 橙
    ]
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]


def build_sig_cmap(alpha_sig):
    boundaries = [0.0, alpha_sig, 1.0]
    reds = plt.get_cmap('Reds_r')(np.linspace(0.25, 0.95, 1))
    colors = list(reds) + [(1, 1, 1, 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N, extend='neither')
    ticks = [1.0, alpha_sig, 0.0]
    return cmap, norm, ticks


def add_box_label(ax, text, xy=(0.5, 1.02), fontsize=12, weight='bold', ha='center', va='bottom'):
    ax.text(
        xy[0], xy[1], text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=fontsize, fontweight=weight,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none')
    )


def validate_topo_shape(psd_s1, psd_s2, info_obj):
    n_ch = len(info_obj['ch_names'])
    if psd_s1.shape[1] != n_ch or psd_s2.shape[1] != n_ch:
        raise ValueError(
            f"Topomap数据通道数与info不一致：info={n_ch}, "
            f"psd_s1={psd_s1.shape[1]}, psd_s2={psd_s2.shape[1]}"
        )


# =========================
# PSD计算
# =========================
def compute_subject_psd(X, label, ch_idx, sfreq, fmin, fmax):
    # 仅 C3/C4 给曲线图
    baseline_data_left = X[label == 1, :, :(RestDuration * sfreq)]
    task_data_left = X[label == 1, :, (RestDuration * sfreq):]
    baseline_data_right = X[label == 2, :, :(RestDuration * sfreq)]
    task_data_right = X[label == 2, :, (RestDuration * sfreq):]

    psd_baseline_left, freqs = mne.time_frequency.psd_array_welch(
        baseline_data_left, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=256, verbose=False
    )
    psd_task_left, _ = mne.time_frequency.psd_array_welch(
        task_data_left, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=256, verbose=False
    )
    psd_baseline_right, _ = mne.time_frequency.psd_array_welch(
        baseline_data_right, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=256, verbose=False
    )
    psd_task_right, _ = mne.time_frequency.psd_array_welch(
        task_data_right, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=256, verbose=False
    )

    psd_baseline_left = 10 * np.log10(psd_baseline_left)
    psd_task_left = 10 * np.log10(psd_task_left)
    psd_baseline_right = 10 * np.log10(psd_baseline_right)
    psd_task_right = 10 * np.log10(psd_task_right)

    corrected_psd_left_all = np.mean(psd_task_left, axis=0) - np.mean(psd_baseline_left, axis=0)
    corrected_psd_right_all = np.mean(psd_task_right, axis=0) - np.mean(psd_baseline_right, axis=0)
    corrected_psd_left = corrected_psd_left_all[ch_idx,:]
    corrected_psd_right = corrected_psd_right_all[ch_idx,:]

    return corrected_psd_left, corrected_psd_right, corrected_psd_left_all, corrected_psd_right_all, freqs


def collect_all_psd(data_4_filepath, subjectchoose):
    ch_idx = [ch_names.index(ch) for ch in channels_to_plot]

    psd_results = {}
    topo_results = {}
    freqs_ref = None

    for raw_stim in RAW_STIM_ORDER:
        stim = RAW_TO_DISPLAY[raw_stim]
        psd_results[stim] = {'S1': {}, 'S2': {}}
        topo_results[stim] = {'S1': {}, 'S2': {}}

        for sen in ['S1', 'S2']:
            PSD_left, PSD_right = [], []
            PSD_left_all, PSD_right_all = [], []

            for s in subjectchoose:
                mat_path = os.path.join(data_4_filepath, f"{pfx(s)}{sen}_{raw_stim}.mat")
                data, label, fs = load_block(mat_path)
                Data = ERPs_Filter(data, freqs=freqwindow, fs=fs, filterflag='filtfilt')
                X = np.transpose(Data, (2, 0, 1))  # (trials, chan, time)

                cur_left, cur_right, cur_left_all, cur_right_all, freqs = compute_subject_psd(
                    X=X,
                    label=label,
                    ch_idx=ch_idx,
                    sfreq=srate,
                    fmin=fre_band[0],
                    fmax=fre_band[1]
                )

                freqs_ref = freqs
                PSD_left.append(cur_left)
                PSD_right.append(cur_right)
                PSD_left_all.append(cur_left_all)
                PSD_right_all.append(cur_right_all)

            psd_results[stim][sen]['L'] = np.array(PSD_left)      # (n_sub, 2, n_freqs)
            psd_results[stim][sen]['R'] = np.array(PSD_right)     # (n_sub, 2, n_freqs)
            topo_results[stim][sen]['L'] = np.array(PSD_left_all)   # (n_sub, 60, n_freqs)
            topo_results[stim][sen]['R'] = np.array(PSD_right_all)  # (n_sub, 60, n_freqs)

    return psd_results, topo_results, freqs_ref


# =========================
# 小图：PSD
# =========================
def _draw_psd_panel(ax, data_left, data_right, freqs, kind, channels_to_plot_local, ylim_main, ylim_diff, show_legend=True):
    colors = make_scientific_colors(len(channels_to_plot_local))
    cmap_diff = plt.get_cmap('Greens')
    handles = []
    labels = []

    for i, ch in enumerate(channels_to_plot_local):
        mean_left = np.mean(data_left[:, i, :], axis=0)
        mean_right = np.mean(data_right[:, i, :], axis=0)
        std_left = np.std(data_left[:, i, :], axis=0)
        std_right = np.std(data_right[:, i, :], axis=0)

        if kind == 'Left':
            color = colors[i]
            ax.plot(freqs, mean_left, color=color, linewidth=1.4)
            ax.fill_between(freqs, mean_left - std_left, mean_left + std_left, alpha=0.18, color=color)
            handles.append(Line2D([0], [0], color=color, lw=2))
            labels.append(ch)

        elif kind == 'Right':
            color = colors[i]
            ax.plot(freqs, mean_right, color=color, linewidth=1.4)
            ax.fill_between(freqs, mean_right - std_right, mean_right + std_right, alpha=0.18, color=color)
            handles.append(Line2D([0], [0], color=color, lw=2))
            labels.append(ch)

        else:
            color = cmap_diff(0.55 + 0.25 * i / max(1, len(channels_to_plot_local) - 1))
            diff = mean_left - mean_right
            ax.plot(freqs, diff, color=color, linewidth=1.6)
            handles.append(Line2D([0], [0], color=color, lw=2))
            labels.append(ch)

    shade_bands(ax, bands, colors=band_colors, alpha=band_alpha)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlim([fre_band[0], fre_band[1]])
    ax.set_xlabel('Frequency (Hz)')

    if kind in ['Left', 'Right']:
        ax.set_ylabel('PSD (dB)')
        ax.set_ylim(ylim_main)
    else:
        ax.set_ylabel('Power Difference (dB)')
        ax.set_ylim(ylim_diff)

    if show_legend:
        ax.legend(handles, labels, loc='upper right', frameon=True, fontsize=9)

    return handles, labels


def save_small_psd_figures(psd_results, freqs, save_dir):
    for stim in BIG_STIM_ORDER:
        for sen in ['S1', 'S2']:
            data_left = psd_results[stim][sen]['L']
            data_right = psd_results[stim][sen]['R']

            for kind in ['Left', 'Right', 'Diff']:
                fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.2))
                fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.90)

                _draw_psd_panel(
                    ax=ax,
                    data_left=data_left,
                    data_right=data_right,
                    freqs=freqs,
                    kind=kind,
                    channels_to_plot_local=channels_to_plot,
                    ylim_main=PSD_ylim,
                    ylim_diff=PSD_diff_ylim,
                    show_legend=True
                )

                ax.set_title(f'{sen} - {stim} - {kind}', fontsize=12)
                out_path = os.path.join(save_dir, f"PSD_{stim}_{sen}_{kind}.png")
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close(fig)


# =========================
# 大图：PSD（3张）
# =========================
def save_big_psd_figures(psd_results, freqs, save_dir):
    for kind in ['Left', 'Right', 'Diff']:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), sharex=True)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.86, wspace=0.18, hspace=0.22)

        legend_handles, legend_labels = None, None

        for r, sen in enumerate(['S1', 'S2']):
            for c, stim in enumerate(BIG_STIM_ORDER):
                ax = axes[r, c]
                data_left = psd_results[stim][sen]['L']
                data_right = psd_results[stim][sen]['R']

                handles, labels = _draw_psd_panel(
                    ax=ax,
                    data_left=data_left,
                    data_right=data_right,
                    freqs=freqs,
                    kind=kind,
                    channels_to_plot_local=channels_to_plot,
                    ylim_main=PSD_ylim,
                    ylim_diff=PSD_diff_ylim,
                    show_legend=False
                )

                if legend_handles is None:
                    legend_handles, legend_labels = handles, labels

                if r == 0:
                    add_box_label(ax, stim, xy=(0.5, 1.05), fontsize=13)

                if c == 0:
                    ax.text(
                        -0.28, 0.5, sen,
                        transform=ax.transAxes,
                        ha='center', va='center',
                        rotation=90,
                        fontsize=14,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='none')
                    )

        title_map = {
            'Left': 'PSD Big Figure - Left Hand',
            'Right': 'PSD Big Figure - Right Hand',
            'Diff': 'PSD Big Figure - Left minus Right'
        }
        fig.suptitle(title_map[kind], fontsize=16, fontweight='bold', y=0.97)

        fig.legend(
            legend_handles, legend_labels,
            loc='upper center',
            ncol=len(legend_labels),
            bbox_to_anchor=(0.5, 0.915),
            frameon=True,
            fontsize=11
        )

        out_path = os.path.join(save_dir, f"PSD_BIG_{kind}.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


# =========================
# topo统计
# =========================
def compute_topo_stats(psd_s1, psd_s2, freqs):
    validate_topo_shape(psd_s1, psd_s2, info)

    stats_by_band = {}
    for band_name, fmin, fmax in bands:
        freq_mask = (freqs >= fmin) & (freqs <= fmax)

        band_power_s1 = np.mean(psd_s1[:, :, freq_mask], axis=2)
        band_power_s2 = np.mean(psd_s2[:, :, freq_mask], axis=2)

        mean_power_s1 = np.mean(band_power_s1, axis=0)
        mean_power_s2 = np.mean(band_power_s2, axis=0)
        power_diff = mean_power_s1 - mean_power_s2

        p_topo = stats.ttest_rel(band_power_s1, band_power_s2, axis=0).pvalue
        _, pvals_corrected, _, _ = multipletests(p_topo, alpha=significance, method='fdr_bh')
        p_mask = np.where(pvals_corrected > significance, 1.0, pvals_corrected)

        stats_by_band[band_name] = {
            'S1': mean_power_s1,
            'S2': mean_power_s2,
            'Diff': power_diff,
            'P-values': p_mask
        }
    return stats_by_band


# =========================
# 小图：topo
# =========================
def save_small_topo_figures(topo_results, freqs, save_dir):
    sig_cmap, sig_norm, sig_ticks = build_sig_cmap(significance)

    for stim in BIG_STIM_ORDER:
        for side in ['L', 'R']:
            psd_s1 = topo_results[stim]['S1'][side]
            psd_s2 = topo_results[stim]['S2'][side]
            stats_by_band = compute_topo_stats(psd_s1, psd_s2, freqs)

            for band_name in [b[0] for b in bands]:
                for row_name in ['S1', 'S2', 'Diff', 'P-values']:
                    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0))
                    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.88)

                    if row_name == 'P-values':
                        im, _ = mne.viz.plot_topomap(
                            stats_by_band[band_name][row_name], info,
                            axes=ax, show=False, cmap=sig_cmap,
                            vlim=[0, 1], contours=0
                        )
                        cbar = fig.colorbar(
                            ScalarMappable(cmap=sig_cmap, norm=sig_norm),
                            ax=ax, fraction=0.046, pad=0.04, ticks=sig_ticks
                        )
                        cbar.set_label('p value')
                    else:
                        im, _ = mne.viz.plot_topomap(
                            stats_by_band[band_name][row_name], info,
                            axes=ax, show=False, cmap='jet',
                            vlim=PSD_topo_clar, contours=6
                        )
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('PSD Power (dB)' if row_name != 'Diff' else 'Power Difference (dB)')

                    ax.set_title(f'{row_name} - {stim}-{side} - {band_name}', fontsize=11)
                    out_path = os.path.join(save_dir, f"Topo_Stats_{row_name}-{stim}-{side}-{band_name}.png")
                    fig.savefig(out_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)


# =========================
# 大图：topo（每个刺激一张）
# =========================
def save_big_topo_figures(topo_results, freqs, save_dir):
    sig_cmap, sig_norm, sig_ticks = build_sig_cmap(significance)
    power_norm = Normalize(vmin=PSD_topo_clar[0], vmax=PSD_topo_clar[1])

    for stim in BIG_STIM_ORDER:
        fig = plt.figure(figsize=(15.5, 10.0))
        gs = fig.add_gridspec(
            nrows=4, ncols=6,
            left=0.07, right=0.88, bottom=0.08, top=0.88,
            wspace=0.05, hspace=0.18
        )

        axes = np.empty((4, 6), dtype=object)
        for r in range(4):
            for c in range(6):
                axes[r, c] = fig.add_subplot(gs[r, c])

        row_order = ['S1', 'S2', 'Diff', 'P-values']
        band_names = [b[0] for b in bands]

        # 左半边 L，右半边 R
        for side_idx, side in enumerate(['L', 'R']):
            psd_s1 = topo_results[stim]['S1'][side]
            psd_s2 = topo_results[stim]['S2'][side]
            stats_by_band = compute_topo_stats(psd_s1, psd_s2, freqs)

            col_offset = 0 if side == 'L' else 3

            for r, row_name in enumerate(row_order):
                for c, band_name in enumerate(band_names):
                    ax = axes[r, c + col_offset]

                    if row_name == 'P-values':
                        mne.viz.plot_topomap(
                            stats_by_band[band_name][row_name], info,
                            axes=ax, show=False, cmap=sig_cmap,
                            vlim=[0, 1], contours=0
                        )
                    else:
                        mne.viz.plot_topomap(
                            stats_by_band[band_name][row_name], info,
                            axes=ax, show=False, cmap='jet',
                            vlim=PSD_topo_clar, contours=6
                        )

                    if r == 0:
                        add_box_label(ax, band_name, xy=(0.5, 1.06), fontsize=12)

                    if c == 0:
                        ax.text(
                            -0.25, 0.5, row_name,
                            transform=ax.transAxes,
                            ha='center', va='center',
                            rotation=90,
                            fontsize=13, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='none')
                        )

        # L / R 顶部总标签
        fig.text(
            0.275, 0.915, f'{stim} - L',
            ha='center', va='center', fontsize=15, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='none')
        )
        fig.text(
            0.665, 0.915, f'{stim} - R',
            ha='center', va='center', fontsize=15, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='none')
        )

        # 主标题
        fig.suptitle(f'Topomap Big Figure - {stim}', fontsize=17, fontweight='bold', y=0.965)

        # 共享colorbar：功率
        cax_power = fig.add_axes([0.905, 0.30, 0.02, 0.45])
        sm_power = ScalarMappable(norm=power_norm, cmap='jet')
        sm_power.set_array([])
        cb1 = fig.colorbar(sm_power, cax=cax_power)
        cb1.set_label('PSD / Diff (dB)', fontsize=11)

        # 共享colorbar：p值
        cax_p = fig.add_axes([0.905, 0.10, 0.02, 0.12])
        sm_p = ScalarMappable(norm=sig_norm, cmap=sig_cmap)
        sm_p.set_array([])
        cb2 = fig.colorbar(sm_p, cax=cax_p, ticks=sig_ticks)
        cb2.set_label('p value', fontsize=10)

        out_path = os.path.join(save_dir, f"TOPO_BIG_{stim}.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


# =========================
# 主程序
# =========================
if __name__ == '__main__':
    if sence == 'ssmvep_hybrid':
        data_4_filepath = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2处理后数据'
        save_dir = r'E:\Datasets\4_跨场景因素研究v2\画图数据\画图结果\PSD结果'
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        subjectchoose = list(range(1, 37 + 1))

        print('Step 1/4: collecting PSD results...')
        psd_results, topo_results, freqs = collect_all_psd(
            data_4_filepath=data_4_filepath,
            subjectchoose=subjectchoose
        )

        print('Step 2/4: saving small PSD figures...')
        save_small_psd_figures(psd_results, freqs, save_dir)

        print('Step 3/4: saving big PSD figures...')
        save_big_psd_figures(psd_results, freqs, save_dir)

        print('Step 4/4: saving topo figures (small + big)...')
        save_small_topo_figures(topo_results, freqs, save_dir)
        save_big_topo_figures(topo_results, freqs, save_dir)

        print('Done.')