# eeg_filter.py
import numpy as np
from scipy.signal import butter, filtfilt, lfilter, detrend

def ERPs_Filter(data, freqs, channel=None, timewindow=None, fs=250, filterorder=5, filterflag='filter'):
    """
    data: (channels, points, samples)
    freqs: [low, high] 或 {'custom': (b,a)} 但此处按 [low, high] 实现
    返回: 同维度数组
    """
    X = np.array(data, dtype=float, copy=True)

    # 选通道
    if channel is not None and len(channel) > 0:
        if max(channel) > X.shape[0]:
            print("警告: 所选导联超出范围，已取消筛选。")
        else:
            X = X[np.array(channel)-1, :, :]  # MATLAB 通常 1-based

    # 时间窗
    if timewindow is not None and len(timewindow) > 0:
        t0, t1 = (timewindow[0], timewindow[-1]) if len(timewindow) > 1 else (0, timewindow[0])
        t1 = min(t1, X.shape[1]/fs)
        s0 = int(round(t0*fs))
        s1 = int(round(t1*fs))
        X = X[:, s0:s1, :]

    # 滤波器
    low, high = freqs
    Wn = [2*low/fs, 2*high/fs]
    b, a = butter(filterorder, Wn, btype='bandpass')

    Y = np.zeros_like(X)
    for s in range(X.shape[2]):
        data1 = detrend(X[:, :, s].T, axis=0)  # → (time, chan)
        if filterflag == 'filtfilt':
            data2 = filtfilt(b, a, data1, axis=0)
        else:
            data2 = lfilter(b, a, data1, axis=0)
        Y[:, :, s] = data2.T
    return Y