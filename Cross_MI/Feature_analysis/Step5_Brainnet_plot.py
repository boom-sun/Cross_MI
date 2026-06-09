import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs
import matplotlib.pyplot as plt
from pathlib import Path
import hdf5storage
import os
from eeg_filter import ERPs_Filter
import mne_connectivity

# ===== 参数 =====
datatype = 1
srate = 250
freqwindow = [1, 40]
frames = 1500
TaskDuration = 4
RestDuration = 2
tlimits = [-1000*RestDuration, TaskDuration*1000 - 1]  # ms
sence = 'ssmvep_hybrid'
feature='pearson'   # 'plv'
fre_band = [8,13]
ch_names = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ',
    'PO4','PO6','PO8','O1','OZ','O2'
]
ch_names_32 =['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1',
        'FC2', 'FC6', 'T7', 'C3','CZ', 'C4', 'CP5', 'CP1', 'CPZ', 'CP2',
        'CP4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P6', 'P8', 'POZ', 'O1',
        'OZ', 'O2']
CH_names=ch_names_32
CH_ind=[ch_names.index(CH_names[i]) for i in range(len(CH_names))]
LOCS_FILE = 'channel_location_60_neuroscan.locs'
montage = mne.channels.read_custom_montage(LOCS_FILE)
info = mne.create_info(CH_names, sfreq=250, ch_types='eeg')
info.set_montage(montage, on_missing='warn')
n_channels = len(CH_names)
connection_threshold = 0.95

def pfx(s):
    return f"{datatype}S0{s}" if s <= 9 else f"{datatype}S{s}"

def load_block(p):
    M = hdf5storage.loadmat(p)
    return M['data'], M['label'].ravel(), float(np.squeeze(M['fs']))

def brainnet_net(data, fre_win, sence, c, method='plv'):
    n_trials, n_channels, n_samples = data.shape
    if method == 'pearson':
        con_matrix_list = []
        for i in range(n_trials):
            trial_data = data[i, :, :]
            filtered_data = mne.filter.filter_data(trial_data, srate, l_freq=fre_band[0], h_freq=fre_band[1],
                                                   verbose=False)
            corr_matrix = np.corrcoef(filtered_data)
            con_matrix_list.append(np.abs(corr_matrix)) # 对于可视化，通常取绝对值，因为负相关也是一种强连接
        avg_con_matrix = np.mean(con_matrix_list, axis=0)

    elif method in ['plv', 'wpli', 'coh', 'imaginary_coh', 'granger']:
        # PLV, wPLI, Coherence 等都属于频谱连接性
        # mne_connectivity 可以一次性高效地处理所有 trial
        # MNE-Connectivity 接受的 method 参数名与我们定义的一致
        con = spectral_connectivity_epochs(
            data, method=method, mode='multitaper', sfreq=srate,
            fmin=fre_band[0], fmax=fre_band[1], faverage=True, verbose=False,
            tmin=0, tmax=4.0, n_jobs=1
        )
        avg_con_matrix = con.get_data('dense')[:, :, 0]
    else:
        raise ValueError(f"未知的方法: '{method}'. 请从 'pearson', 'plv', 'wpli', 'coh' 等中选择。")

    np.fill_diagonal(avg_con_matrix, 0)     # 将对角线（自身连接）设置为0

    # ---  可视化为热图矩阵 ---
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(avg_con_matrix, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_channels))
    ax.set_yticks(np.arange(n_channels))
    ax.set_xticklabels(CH_names, rotation=90)
    ax.set_yticklabels(CH_names)
    ax.set_title(f'FC Matrix (PLV, Sence{sence[-1]}, {c},'
                 f' {str(fre_win[0])}-{str(fre_win[1])} Hz)', fontsize=14, weight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Phase Locking Value (PLV)', fontsize=12)
    plt.tight_layout()
    p_save = os.path.join(save_dir_brainnet, f"Brain_FCMatrix_{c}_sence{sence}.png")
    fig.savefig(p_save, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # --- 可视化为脑网络图 ---
    left_color = 'lightcoral'
    right_color = 'skyblue'
    center_color = 'lightgray'

    node_colors = []
    for name in CH_names:
        if name.endswith('z'):
            node_colors.append(center_color)
        elif name[-1].isdigit() and int(name[-1]) % 2 != 0:
            node_colors.append(left_color)
        elif name[-1].isdigit() and int(name[-1]) % 2 == 0:
            node_colors.append(right_color)
        else:
            node_colors.append('white')  # 备用颜色

    fig_circle = plt.figure(figsize=(8, 8), facecolor='w')

    ax_circle = fig_circle.add_axes([0, -0.2, 1.3, 1.3], projection='polar')
    ax_circle.set_axis_off()
    mne_connectivity.viz.plot_connectivity_circle(
        avg_con_matrix,
        CH_names,
        ax=ax_circle,
        node_colors=node_colors,
        facecolor='w',
        textcolor='k',  # 这个 textcolor 可能是内部设置，我们后面会覆盖
        vmin=connection_threshold,
        colormap='viridis',
        colorbar_pos=(-0.1, 0.5),
        show=False,
        title=f'Connectivity Matrix (PLV, Sence{sence[-1]}, {c}, {fre_win[0]}-{fre_win[1]} Hz)'
    )
    ax_circle.title.set_fontsize(25)  # 设置一个比较大的值，例如 20
    channel_name_set = set(CH_names)
    for text_obj in ax_circle.texts:
        if text_obj.get_text() in channel_name_set:
            text_obj.set_fontsize(20)  # 设置一个你想要的大小，例如 14
    cbar_ax = fig_circle.axes[1]
    cbar_ax.tick_params(labelsize=20)  # 设置你想要的大小，例如 12
    pos = cbar_ax.get_position()
    current_left = pos.x0
    current_bottom = pos.y0
    current_width = pos.width
    current_height = pos.height
    new_width = current_width * 5   # 例如，让它变粗到原来的 2.5 倍
    new_height = current_height * 1.2 # 例如，让它变长一点点
    cbar_ax.set_position([current_left, current_bottom, new_width, new_height])
    p1_save = os.path.join(save_dir_brainnet, f"BrainNet_{c}_{sence}.png")
    fig_circle.savefig(p1_save, dpi=300, bbox_inches='tight')
    plt.close(fig_circle)

if sence == 'graz':
    data_4_filepath = r'E:\Datasets\1_Graz范式\处理后数据'
    save_dir_brainnet = r'E:\Datasets\1_Graz范式\Graz画图数据\画图结果\脑网络结果'
    Path(save_dir_brainnet).mkdir(parents=True, exist_ok=True)
    Class_name = ('Left', 'Right', 'Feet', 'Rest')
    subjectchoose = list(range(1, 14 + 1))
    for sen in ['S1', 'S2']:
        Data_all = []
        Label_all = []
        for s in [sc for sc in subjectchoose]:
            paths = {'1': os.path.join(data_4_filepath, f"{pfx(s)}{sen}.mat")}
            data, label, fs = load_block(paths['1'])
            Data = ERPs_Filter(data, freqs=freqwindow, fs=fs, filterflag='filtfilt')
            X = np.transpose(Data, (2, 0, 1))  # (trials,chan, time)
            Data_all.append(X)
            Label_all.append(label)
        X=np.concatenate(Data_all, axis=0)
        y=np.concatenate(Label_all, axis=0)
        for class_choose in [1, 2, 3, 4]:
            X_temp = X[y == class_choose]
            X1 = mne.EpochsArray(X_temp, info)
            X1.set_montage(montage)
            brainnet_net(X1, fre_band,sen, Class_name[class_choose-1], method=feature)
if sence == 'ssmvep':
    data_4_filepath = r'E:\Datasets\2_MI_SSMVEP范式\MI_SSMVEP处理后数据'
    save_dir_brainnet = r'E:\Datasets\2_MI_SSMVEP范式\画图数据\画图结果\脑网络结果'
    Path(save_dir_brainnet).mkdir(parents=True, exist_ok=True)
    Class_name =  ('Left_MI', 'Right_MI', 'Left_AO', 'Right_AO')
    subjectchoose = list(range(1, 22 + 1))
    for sen in ['S1', 'S2']:
        Data_all = []
        Label_all = []
        for s in [sc for sc in subjectchoose]:
            paths = {'1': os.path.join(data_4_filepath, f"{pfx(s)}{sen}.mat")}
            data, label, fs = load_block(paths['1'])
            Data = ERPs_Filter(data, freqs=freqwindow, fs=fs, filterflag='filtfilt')
            X = np.transpose(Data, (2, 0, 1))  # (trials,chan, time)
            Data_all.append(X)
            Label_all.append(label)
        X=np.concatenate(Data_all, axis=0)
        y=np.concatenate(Label_all, axis=0)
        for class_choose in [1, 2, 3, 4]:
            X_temp = X[y == class_choose]
            X1 = mne.EpochsArray(X_temp, info)
            X1.set_montage(montage)
            brainnet_net(X1, fre_band,sen, Class_name[class_choose-1], method=feature)

if sence == 'ssmvep_hybrid':
    data_4_filepath = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2处理后数据'
    save_dir_brainnet = r'E:\Datasets\4_跨场景因素研究v2\画图数据\画图结果\脑网络结果'
    Path(save_dir_brainnet).mkdir(parents=True, exist_ok=True)
    stim_name = ('ssvideo', 'video', 'ssmvep', 'cue')
    stim_new_name = ('SSVideo', 'Video', 'SSMVEP', 'Graz')
    Class_name = ('Left', 'Right')
    subjectchoose = list(range(1, 37 + 1))
    for sen in ['S1', 'S2']:
        for i, stim in enumerate(stim_name):
            Data_all = []
            Label_all = []
            for s in [sc for sc in subjectchoose]:
                paths = {'1': os.path.join(data_4_filepath, f"{pfx(s)}{sen}_{stim}.mat")}
                data, label, fs = load_block(paths['1'])
                Data = ERPs_Filter(data, freqs=freqwindow, fs=fs, filterflag='filtfilt')
                X = np.transpose(Data, (2, 0, 1))  # (trials,chan, time)
                Data_all.append(X)
                Label_all.append(label)
            X=np.concatenate(Data_all, axis=0)
            y=np.concatenate(Label_all, axis=0)
            for class_choose in [1, 2]:
                X_temp = X[y == class_choose]
                X1 = mne.EpochsArray(X_temp, info)
                X1.set_montage(montage)
                new_name=f'{stim_new_name[i]}, {Class_name[class_choose-1]}'
                brainnet_net(X1, fre_band,sen, new_name, method=feature)

