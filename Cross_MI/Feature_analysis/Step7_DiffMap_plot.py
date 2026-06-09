# ===== Step7_DiffMap_plot.py =====
# 目的：画 class1-class2 的差异图谱，而不是单类强度图。
# 输出：差异脑地形图、效应量脑地形图、显著性脑地形图、差异时频图、Fisher时频图。
import os
from pathlib import Path
import numpy as np
import hdf5storage
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
import mne
from Functions import TrialOutlierParams, apply_trial_outlier_removal

# ------------------- 基本参数 -------------------
sence = 'ssmvep_hybrid'
class_pair = (1, 2)
tf_ch = ['C3', 'C4', 'CZ', 'OZ']
tf_band = (5, 40)
tf_timewin = (-2, 4)
topo_band = [(8, 13), (13, 30), (13, 20), (20, 30)]
topo_timewin = [(0, 4)] #[(0, 4), (-2, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2),(2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4)]
significance = 0.05
diff_clim = (-1.5, 1.5)
effect_clim = (-1.0, 1.0)
fisher_clim = (0.0, 1.0)

CHANNELS = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ',
    'PO4','PO6','PO8','O1','OZ','O2'
]
LOCS_FILE = 'channel_location_60_neuroscan.locs'
montage = mne.channels.read_custom_montage(LOCS_FILE)
info = mne.create_info(CHANNELS, sfreq=250, ch_types='eeg')
info.set_montage(montage, on_missing='warn')
is_trialClean=True
if is_trialClean == True:
    _trialClean = '_trialClean'
else:
    _trialClean = None
# ------------------- 路径配置 -------------------
if sence == 'ssmvep_hybrid':
    stim_name = ('ssvideo', 'video', 'ssmvep', 'cue')
    save_root_DIFF = r'E:\Datasets\4_跨场景因素研究v2\画图结果\差异图'
    save_dir = r'E:\Datasets\4_跨场景因素研究v2\画图结果\差异图'
else:
    stim_name = ('default',)
    save_root_DIFF = r'E:\Datasets\1_Graz范式\Graz画图数据\差异图谱数据'
    save_dir = r'E:\Datasets\1_Graz范式\Graz画图数据\差异图谱结果'

save_dir_topo = os.path.join(save_dir, '脑地形差异图')
save_dir_tf = os.path.join(save_dir, '时频差异图')
Path(save_dir_topo).mkdir(parents=True, exist_ok=True)
Path(save_dir_tf).mkdir(parents=True, exist_ok=True)

def build_sig_cmap(significance=0.05):
    boundaries = [0.0, significance, 1.0]
    reds = plt.get_cmap('Reds_r')(np.linspace(0.25, 0.95, 1))
    colors = list(reds) + [(1, 1, 1, 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N, extend='neither')
    ticks = [1.0, significance, 0.0]
    return cmap, norm, ticks


def build_sig_cbar(fig, ax_for_cbar):
    sig_cmap, sig_norm, sig_ticks = build_sig_cmap(significance=significance)
    sm = ScalarMappable(cmap=sig_cmap, norm=sig_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_for_cbar, fraction=0.046, pad=0.04, ticks=sig_ticks)
    cbar.set_label('p value')
    return cbar


for stim in stim_name:
    c1, c2 = class_pair
    p_diff = os.path.join(save_root_DIFF, f'DiffMap_{stim}_class{c1}_minus_class{c2}{_trialClean}.mat') \
        if sence == 'ssmvep_hybrid' else os.path.join(save_root_DIFF, f'DiffMap_class{c1}_minus_class{c2}{_trialClean}.mat')
    data = sio.loadmat(p_diff)
    topo_stats = data['topo_stats'][0,0]
    tf_stats = data['tf_stats'][0,0]
    times = np.squeeze(data['times'])
    freqs = np.squeeze(data['freqs'])

    # ------------------- 脑地形差异图 -------------------
    for sence_id in ['s1', 's2']:
        for band in topo_band:
            for time in topo_timewin:
                key = 'freq' + str(band) + 'time' + str(time)
                mean_diff = np.squeeze(topo_stats[sence_id][0,0][key][0,0]['mean_diff'])
                dz = np.squeeze(topo_stats[sence_id][0,0][key][0,0]['cohens_dz'])
                fisher = np.squeeze(topo_stats[sence_id][0,0][key][0,0]['fisher_score'])
                p_fdr = np.squeeze(topo_stats[sence_id][0,0][key][0,0]['p_fdr'])
                sig_mask = np.squeeze(topo_stats[sence_id][0,0][key][0,0]['sig_fdr']).astype(bool)
                p_mask = np.where(sig_mask, p_fdr, 1.0)

                fig, axes = plt.subplots(4, 1, figsize=(3.6, 11.2), constrained_layout=True)
                im1, _ = mne.viz.plot_topomap(mean_diff, info, axes=axes[0], cmap='RdBu_r', vlim=diff_clim, contours=6, show=False)
                axes[0].set_title(f'{stim}-{sence_id}: C{c1}-C{c2} Diff {band}Hz {time}s')
                cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
                cb1.set_label('Power difference (dB)')

                im2, _ = mne.viz.plot_topomap(dz, info, axes=axes[1], cmap='RdBu_r', vlim=effect_clim, contours=6, show=False)
                axes[1].set_title("Effect size: Cohen's dz")
                cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
                cb2.set_label("Cohen's dz")

                im3, _ = mne.viz.plot_topomap(fisher, info, axes=axes[2], cmap='viridis', vlim=fisher_clim, contours=6, show=False)
                axes[2].set_title('Fisher score')
                cb3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
                cb3.set_label("Fisher score")

                sig_cmap, sig_norm, sig_ticks = build_sig_cmap(significance=significance)
                im4, _ = mne.viz.plot_topomap(p_mask, info, axes=axes[3], cmap=sig_cmap, vlim=[0, 1], contours=0, show=False)
                axes[3].set_title('FDR corrected p-map')
                build_sig_cbar(fig, axes[3])

                p_save = os.path.join(save_dir_topo, f'DiffTOPO_{stim}_{sence_id}_class{c1}-{c2}_'
                                                     f'Freq{band[0]}-{band[1]}Hz_Time{time[0]}-{time[1]}s{_trialClean}.png')
                fig.savefig(p_save, dpi=300, bbox_inches='tight')
                plt.close(fig)

    # ------------------- 时频差异图 -------------------
    times_mask = (times >= tf_timewin[0] * 1000) & (times <= tf_timewin[1] * 1000)
    freqs_mask = (freqs >= tf_band[0]) & (freqs <= tf_band[1])
    times_idx = np.where(times_mask)[0]
    freqs_idx = np.where(freqs_mask)[0]
    x_times = times[times_idx] / 1000.0
    y_freqs = freqs[freqs_idx]

    for sence_id in ['s1', 's2']:
        mean_diff = tf_stats[sence_id][0,0]['mean_diff']
        dz = tf_stats[sence_id][0,0]['cohens_dz']
        fisher = tf_stats[sence_id][0,0]['fisher_score']
        p_fdr = tf_stats[sence_id][0,0]['p_fdr']
        sig = tf_stats[sence_id][0,0]['sig_fdr'].astype(bool)

        for ch in tf_ch:
            ch_index = CHANNELS.index(ch)
            diff_plot = np.take(mean_diff, indices=freqs_idx, axis=0)
            diff_plot = np.take(diff_plot, indices=times_idx, axis=1)[:, :, ch_index]
            dz_plot = np.take(dz, indices=freqs_idx, axis=0)
            dz_plot = np.take(dz_plot, indices=times_idx, axis=1)[:, :, ch_index]
            fisher_plot = np.take(fisher, indices=freqs_idx, axis=0)
            fisher_plot = np.take(fisher_plot, indices=times_idx, axis=1)[:, :, ch_index]
            p_plot = np.take(p_fdr, indices=freqs_idx, axis=0)
            p_plot = np.take(p_plot, indices=times_idx, axis=1)[:, :, ch_index]
            sig_plot = np.take(sig, indices=freqs_idx, axis=0)
            sig_plot = np.take(sig_plot, indices=times_idx, axis=1)[:, :, ch_index]
            p_mask = np.where(sig_plot, p_plot, 1.0)

            fig, axes = plt.subplots(4, 1, figsize=(4.2, 10), constrained_layout=True)
            im1 = axes[0].pcolormesh(x_times, y_freqs, diff_plot, shading='auto', cmap='RdBu_r', vmin=diff_clim[0], vmax=diff_clim[1])
            axes[0].axvline(0.0, color='k', ls='--', lw=1)
            axes[0].set_title(f'{stim}-{sence_id} Channel-{ch}: C{c1}-C{c2}')
            axes[0].set_ylabel('Hz')
            fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04).set_label('Power diff (dB)')

            im2 = axes[1].pcolormesh(x_times, y_freqs, dz_plot, shading='auto', cmap='RdBu_r', vmin=effect_clim[0], vmax=effect_clim[1])
            axes[1].axvline(0.0, color='k', ls='--', lw=1)
            axes[1].set_title("Effect size: Cohen's dz")
            axes[1].set_ylabel('Hz')
            fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04).set_label("Cohen's dz")

            im3 = axes[2].pcolormesh(x_times, y_freqs, fisher_plot, shading='auto', cmap='viridis', vmin=fisher_clim[0], vmax=fisher_clim[1])
            axes[2].axvline(0.0, color='k', ls='--', lw=1)
            axes[2].set_title('Fisher score')
            axes[2].set_ylabel('Hz')
            fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04).set_label('Fisher score')

            sig_cmap, sig_norm, sig_ticks = build_sig_cmap(significance=significance)
            im4 = axes[3].pcolormesh(x_times, y_freqs, p_mask, shading='auto', cmap=sig_cmap, vmin=0.0, vmax=1.0)
            axes[3].axvline(0.0, color='k', ls='--', lw=1)
            axes[3].set_title('FDR corrected p-map')
            axes[3].set_xlabel('Time (s)')
            axes[3].set_ylabel('Hz')
            build_sig_cbar(fig, axes[3])

            p_save = os.path.join(save_dir_tf, f'DiffTF_{stim}_{sence_id}_class{c1}-{c2}_Channel-{ch}{_trialClean}.png')
            fig.savefig(p_save, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f'Finished plotting: {stim}')
