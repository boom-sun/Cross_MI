import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.linalg import eigh
from eeg_filter import ERPs_Filter
from mne import channels
from mne.viz import plot_topomap

# ========== 基本参数，与 Step1 一致 ==========
srate = 250
freqwindow = [1, 40]
frames = 1500
TaskDuration = 4
RestDuration = 2
tlimits = [-1000*RestDuration, TaskDuration*1000 - 1]

data_root = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2处理后数据'
save_root = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据_CSP'
os.makedirs(save_root, exist_ok=True)

stim_name = ('ssvideo','video','ssmvep','cue')

def pfx(s):
    return f"1S0{s}" if s <= 9 else f"1S{s}"

def load_block(p):
    M = sio.loadmat(p)
    data = M['data']
    label = np.array(M['label']).ravel()
    fs = float(np.squeeze(M['fs']))
    return data, label, fs

# ========== CSP ==========
def compute_CSP(X1, X2, n_components=2):
    """ 输入：X1/X2: shape = (chan, time, trials)
        输出：W_csp: (chan, n_components*2)
    """
    # 协方差矩阵
    def covnorm(X):
        C = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[2]):
            x = X[:, :, i]
            C += np.dot(x, x.T) / np.trace(np.dot(x, x.T))
        return C / X.shape[2]

    C1 = covnorm(X1)
    C2 = covnorm(X2)

    # 广义特征分解
    vals, vecs = eigh(C1, C1 + C2)

    # 排序（大→小）
    ind = np.argsort(vals)[::-1]
    vecs = vecs[:, ind]

    # 取前后各 n_components
    W = np.hstack([vecs[:, :n_components], vecs[:, -n_components:]])
    return W

# ========== Topomap 画图 ==========
def plot_topomap_for_CSP(W, ch_names, outpath):
    """ W: (n_channels, n_components*2) """
    # 使用 mne 的 plot_topomap 进行 Topomap 可视化
    info = channels.create_info(ch_names, srate, ch_type='eeg')
    fig, axes = plt.subplots(1, W.shape[1], figsize=(12, 4))
    for i, ax in enumerate(axes):
        plot_topomap(W[:, i], info, axes=ax, show=False)
        ax.set_title(f"CSP{i+1}")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ========== 主流程：计算 CSP、绘制 Topomap 和跨被试平均 ==========
CSP_avg = None  # 初始化 CSP 平均值

for s in range(1, 38):
    print(f"=== Subject {s} ===")

    for stim in stim_name:
        print(f"    -> {stim}")

        path1 = os.path.join(data_root, f"{pfx(s)}S1_{stim}.mat")
        path2 = os.path.join(data_root, f"{pfx(s)}S2_{stim}.mat")
        if not os.path.exists(path1):
            continue

        data1, label1, fs = load_block(path1)
        data2, label2, _ = load_block(path2)

        # 预处理（与 ERD 完全一致）
        X1 = ERPs_Filter(data1, freqs=freqwindow, fs=fs, filterflag='filtfilt')
        X2 = ERPs_Filter(data2, freqs=freqwindow, fs=fs, filterflag='filtfilt')

        # 转为 CSP 格式 (chan, time, trials)
        X1 = np.transpose(X1, (0,1,2))
        X2 = np.transpose(X2, (0,1,2))

        # 左右试次
        L1 = X1[:,:,label1==1]
        R1 = X1[:,:,label1==2]

        if L1.shape[2] < 5 or R1.shape[2] < 5:
            continue

        # ======= 计算 CSP =======
        W = compute_CSP(L1, R1, n_components=2)

        # 更新 CSP 平均值
        if CSP_avg is None:
            CSP_avg = W
        else:
            CSP_avg += W

        # ======== 保存 CSP Topomap ========
        out_folder = os.path.join(save_root, f"sub{s}", stim)
        os.makedirs(out_folder, exist_ok=True)
        plot_topomap_for_CSP(W, ch_names=["Cz", "Fp1", "Fp2", "F3", "F4"],  # 举例使用电极名
                             outpath=os.path.join(out_folder, f"CSP_topomap.jpg"))

# ========== 跨被试平均 CSP ==========
CSP_avg /= 37  # 跨 37 个被试取平均
avg_out_folder = os.path.join(save_root, "average_CSP")
os.makedirs(avg_out_folder, exist_ok=True)
plot_topomap_for_CSP(CSP_avg, ch_names=["Cz", "Fp1", "Fp2", "F3", "F4"],
                     outpath=os.path.join(avg_out_folder, "average_CSP_topomap.jpg"))

print("CSP Topomap 可视化与跨被试平均完成。")
