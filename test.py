# ===== plot_per_subject_by_class_joblib.py =====
import os
import hdf5storage
import gc
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multitest import multipletests
import mne
import h5py
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats
from pathlib import Path


def build_sig_cmap(significance):
    boundaries = [0.0, significance, 1.0]
    reds = plt.get_cmap('Reds_r')(np.linspace(0.25, 0.95, 1))  # 红色渐变
    colors = list(reds) + [(1, 1, 1, 1)]  # 添加白色 (1.0 对应白色)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N, extend='neither')
    ticks = [1.0,significance, 0.0]
    return cmap, norm, ticks
def build_sig_cbar(fig, ax_for_cbar):
    sm = ScalarMappable(cmap=sig_cmap, norm=sig_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_for_cbar, fraction=0.046, pad=0.04, ticks=sig_ticks)
    cbar.set_label('p value')
    return cbar

# ------------------- 基本参数（按需修改） -------------------
save_dir_tf = r'.\时频图结果'
save_dir_topo = r'.\脑地形图结果'
Path(save_dir_tf).mkdir(parents=True, exist_ok=True)
Path(save_dir_topo).mkdir(parents=True, exist_ok=True)
topo_band = [(8,13),(13, 30),(13,20),(20,30)]
tf_band = (5,40)
topo_timewin = [(0, 4)]  # [(0, 4),(-2,0),(0,0.5),(0.5,1),(1,1.5),(1.5,2),(2,2.5),(2.5,3),(3,3.5),(3.5,4)]
tf_timewin = (-2, 4)
tf_ch = ['C3','C4','OZ']
significance = 0.05
topo_clim = (-3.0, 3.0)
tf_clim = (-4.0, 4.0)
CLASS_CHOOSE_LIST = [1, 2]
Class_name = ('Left', 'Right')
save_root_ERSP = r'.\ERSP特征'
save_root_TOPO = r'.\脑地形图特征'
save_root_TF = r'.\时频图特征'

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

# ------------------- 脑地形图 -------------------
for class_choose in CLASS_CHOOSE_LIST:
    class_label = Class_name[class_choose - 1]
    p_topo = os.path.join(save_root_TOPO, f"TOPO_class{class_choose}.mat")
    topo_data = hdf5storage.loadmat(p_topo)['topo']
    for band in topo_band:  # 建立空数组
        for time in topo_timewin:
            topo1 = topo_data['s1']['freq' + str(band) + 'time' + str(time)]
            topo2 = topo_data['s2']['freq' + str(band) + 'time' + str(time)]
            fig, axes = plt.subplots(3, 1, figsize=(3.4 * 1, 8), constrained_layout=True)
            im1, _ = mne.viz.plot_topomap(np.mean(topo1,1), info, axes=axes[0], cmap='jet',
                                         vlim=topo_clim, contours=6, show=False)
            axes[0].set_title(f'S1-Freq{str(band[0])}-{str(band[1])}Hz Time{str(time[0])}-{str(time[1])}s')
            cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            cb1.set_label('PSD (dB)')
            im2, _ = mne.viz.plot_topomap(np.mean(topo2,1), info, axes=axes[1], cmap='jet',
                                         vlim=topo_clim, contours=6, show=False)
            axes[1].set_title(f'S2-Freq{str(band[0])}-{str(band[1])}Hz Time{str(time[0])}-{str(time[1])}s')
            cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            cb2.set_label('PSD (dB)')
            p_topo = stats.ttest_ind(topo1,topo2, axis=1).pvalue
            rejected, pvals_corrected, _,_= multipletests(p_topo, alpha=significance, method='fdr_bh')
            p_mask = np.where(pvals_corrected>significance, 1.0, p_topo)  # 显著显示原始p值，不显著显示白色
            sig_cmap, sig_norm, sig_ticks = build_sig_cmap(significance=significance)
            im3, _ = mne.viz.plot_topomap(p_mask, info, axes=axes[2], cmap=sig_cmap,
                                         vlim=[0,1], show=False, contours=0)
            axes[2].set_title(f'p(FDR)-Freq{str(band[0])}-{str(band[1])}Hz Time{str(time[0])}-{str(time[1])}s')
            sm = ScalarMappable(cmap=sig_cmap, norm=sig_norm)
            sm.set_array([])
            cb3= fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
            cb3.set_label('p-values')
            p_save = os.path.join(save_dir_topo, f"TOPO_class{class_choose}_Freq{str(band[0])}-{str(band[1])}"
                                            f"Hz Time{str(time[0])}-{str(time[1])}.png")
            fig.savefig(p_save, dpi=300, bbox_inches='tight')
            plt.close(fig)

# ------------------- 时频图 -------------------
for class_choose in CLASS_CHOOSE_LIST:
    class_label = Class_name[class_choose - 1]
    tf_path = os.path.join(save_root_ERSP, f"times+freqs_class{class_choose}.mat")
    Times, Freqs = hdf5storage.loadmat(tf_path)['times'], hdf5storage.loadmat(tf_path)['freqs']
    times = Times[(Times>=tf_timewin[0]*1000).__and__(Times<=tf_timewin[1]*1000)]
    freqs = Freqs[(Freqs>=tf_band[0]).__and__(Freqs<=tf_band[1])]
    times_mask = (Times >= tf_timewin[0] * 1000) & (Times <= tf_timewin[1] * 1000)
    freqs_mask = (Freqs >= tf_band[0]) & (Freqs <= tf_band[1])
    freqs_idx = np.where(freqs_mask)[0]  # 一维整数数组，形状 (35,)
    times_idx = np.where(times_mask)[0]  # 一维整数数组，形状 (选择的点数,)
    p_tf = os.path.join(save_root_TF, f"TF_class{class_choose}.mat")
    tf_data = hdf5storage.loadmat(p_tf)['tf']
    for tf_id in tf_ch:
        ch_index_tf = CHANNELS.index(tf_id)
        temp = np.take(tf_data['s1'], indices=freqs_idx, axis=1)
        temp = np.take(temp, indices=[ch_index_tf], axis=3)
        temp = np.take(temp, indices=times_idx, axis=2)
        tf_data1 = np.squeeze(temp, axis=3)
        temp = np.take(tf_data['s2'], indices=freqs_idx, axis=1)
        temp = np.take(temp, indices=[ch_index_tf], axis=3)
        temp = np.take(temp, indices=times_idx, axis=2)
        tf_data2 = np.squeeze(temp, axis=3)
        fig, axes = plt.subplots(3, 1, figsize=(3.8, 8), constrained_layout=True)
        im1=axes[0].pcolormesh(times/1000.0, freqs, np.mean(tf_data1,0), shading='auto', cmap='jet',
                                    vmin=tf_clim[0], vmax=tf_clim[1])
        axes[0].axvline(0.0, color='k', ls='--', lw=1)
        axes[0].set_title(f'S1-Channel-{tf_id}')
        axes[0].set_ylabel('Hz')
        axes[0].set_yticks([5, 10, 20, 30, 40])
        cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cb1.set_label('PSD (dB)')
        im2 = axes[1].pcolormesh(times/1000.0, freqs, np.mean(tf_data2,0), shading='auto', cmap='jet',
                                    vmin=tf_clim[0], vmax=tf_clim[1])
        axes[1].axvline(0.0, color='k', ls='--', lw=1)
        axes[1].set_title(f'S2-Channel-{tf_id}')
        axes[1].set_ylabel('Hz')
        axes[1].set_yticks([5, 10, 20, 30, 40])
        cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cb2.set_label('PSD (dB)')
        p_tf = stats.ttest_ind(tf_data1, tf_data2, axis=0).pvalue
        p_tf_flat = p_tf.flatten()
        rejected, pvals_corrected, _, _ = multipletests(p_tf_flat,alpha=significance,method='fdr_bh')
        rejected_reshaped = rejected.reshape(p_tf.shape)
        pvals_corrected_reshaped = pvals_corrected.reshape(p_tf.shape)
        p_mask = np.where(pvals_corrected_reshaped > significance, 1.0, p_tf)  # 显著显示原始p值，不显著显示白色
        sig_cmap, sig_norm, sig_ticks = build_sig_cmap(significance=significance)
        im3=axes[2].pcolormesh(times/1000.0, freqs, p_mask, shading='auto',
                                cmap=sig_cmap, vmin=0.0, vmax=1.0)
        axes[2].axvline(0.0, color='k', ls='--', lw=1)
        axes[2].set_title(f'p(FDR)-Channel-{tf_id}')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Hz')
        axes[2].set_yticks([5, 10, 20, 30, 40])
        build_sig_cbar(fig, axes[2])
        fig.patch.set_facecolor('white')
        p_save = os.path.join(save_dir_tf, f"TF_class{class_choose}_Channel-{tf_id}.png")
        fig.savefig(p_save, dpi=300, bbox_inches='tight')
        plt.close(fig)