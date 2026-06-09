# eeg_stft.py
from numpy.fft import fft
import numpy as np

def tftb_window(L, win_name='hamming'):
    """
    生成各种窗函数

    参数:
        L: 窗长度
        win_name: 窗类型，支持:
            'hamming', 'hanning', 'hann', 'rect', 'rectangular',
            'bartlett', 'triang', 'blackman', 'gauss', 'gaussian'

    返回:
        w: 窗函数向量，长度为L
    """
    L = int(L)
    if L <= 0:
        raise ValueError("Window length must be positive")

    # 确保窗长度为奇数（如原代码要求）
    if L % 2 == 0:
        L = L + 1

    win_name = win_name.lower()

    if win_name in ['rect', 'rectangular']:
        w = np.ones(L)

    elif win_name in ['hamming', 'hamming']:
        # Hamming窗
        n = np.arange(L)
        w = 0.54 - 0.46 * np.cos(2 * np.pi * n / (L - 1))

    elif win_name in ['hanning', 'hann']:
        # Hanning窗（Hann窗）
        n = np.arange(L)
        w = 0.5 * (1 - np.cos(2 * np.pi * n / (L - 1)))

    elif win_name == 'bartlett':
        # Bartlett窗（三角窗）
        n = np.arange(L)
        w = 1 - 2 * np.abs(n - (L - 1) / 2) / (L - 1)

    elif win_name == 'triang':
        # Triangular窗（Bartlett的变体）
        n = np.arange(L)
        w = 1 - 2 * np.abs(n - (L - 1) / 2) / L

    elif win_name == 'blackman':
        # Blackman窗
        n = np.arange(L)
        w = (0.42 - 0.5 * np.cos(2 * np.pi * n / (L - 1)) +
             0.08 * np.cos(4 * np.pi * n / (L - 1)))

    elif win_name in ['gauss', 'gaussian']:
        # 高斯窗
        n = np.arange(L)
        alpha = 2.5  # 默认参数，与MATLAB的gausswin函数类似
        w = np.exp(-0.5 * (alpha * (n - (L - 1) / 2) / ((L - 1) / 2)) ** 2)

    elif win_name == 'flattop':
        # 平顶窗
        n = np.arange(L)
        a0 = 0.21557895
        a1 = 0.41663158
        a2 = 0.277263158
        a3 = 0.083578947
        a4 = 0.006947368
        w = (a0 - a1 * np.cos(2 * np.pi * n / (L - 1)) +
             a2 * np.cos(4 * np.pi * n / (L - 1)) -
             a3 * np.cos(6 * np.pi * n / (L - 1)) +
             a4 * np.cos(8 * np.pi * n / (L - 1)))

    else:
        raise ValueError(f"Unknown window type: {win_name}. "
                         f"Supported types: hamming, hanning, rect, bartlett, "
                         f"triang, blackman, gauss, flattop")

    # 确保窗函数是实数，并归一化为单位能量
    w = w.astype(np.float64)
    w = w / np.linalg.norm(w)

    return w


def _ensure_odd(L):
    return int(L + 1 - (L % 2 == 0))

def tfrstft(x, N=None, h=None):
    """
    等价 MATLAB: [tfr,t,f] = tfrstft(x, t, N, h)
    这里简化为：t 取全长 1:len(x)，trace=0，h 为给定窗（需奇长且单位能量），N 为 FFT 长度。
    返回:
      tfr: (N, T) 复数，频轴 0..N-1（后续只用正频）
      f_norm: (N,) 归一化频率 [-0.5,0.5) 的重排，保持与原函数一致的定义
    """
    x = np.asarray(x).reshape(-1)
    T = x.shape[0]
    if N is None:
        N = T

    if h is None:
        # 原函数默认 hamming(N/4) 且奇长
        hlen = _ensure_odd(np.floor(N/4.0))
        h = tftb_window(int(hlen), 'hamming')

    h = np.asarray(h).reshape(-1, 1)  # (Lh,1)
    if h.shape[0] % 2 == 0:
        raise ValueError("Window length must be odd")
    h = h / np.linalg.norm(h)  # 单位能量
    Lh = (h.shape[0]-1)//2

    # 构造 (N,T) 矩阵
    tfr = np.zeros((N, T), dtype=complex)
    halfN = int(np.floor(N/2.0))

    for ti in range(T):
        # tau 范围
        tau_min = -min(halfN-1, Lh, ti)
        tau_max =  min(halfN-1, Lh, T-1-ti)
        tau = np.arange(tau_min, tau_max+1)
        idx = ( (N + tau) % N )  # 0..N-1
        # 填充
        tfr[idx, ti] = x[ti+tau] * np.conjugate(h[Lh+tau, 0])

    tfr = fft(tfr, axis=0)

    # 频率向量（归一化频率）
    if N % 2 == 0:
        f = np.concatenate([np.arange(0, N//2), np.arange(-N//2, 0)]) / N
    else:
        f = np.concatenate([np.arange(0, (N-1)//2 + 1), np.arange(-(N-1)//2, 0)]) / N

    return tfr, f

def m_tfr(data, fs, N=256, win_name='gauss', win_len=287,
          out_freq_bins=53, out_time_len=1500):
    """
    输入 data 维度可为:
      - (time, channels, trials)  或
      - (channels, time, trials)
    输出:
      ttfr: (out_freq_bins, out_time_len, trials, channels)
      t:    (out_time_len,) 采样点索引 (1..out_time_len)，与 MATLAB 对齐
      f:    (out_freq_bins,) 频率(Hz)，基于 fs 与 N 的分辨率 (fs/N)
    """
    # 归一成 (time, channels, trials)
    if data.shape[0] in (1500, 1800, 2000):  # 常见时间长度
        time_major = True
    else:
        time_major = False

    if time_major:
        T, C, Tr = data.shape
        X = data
    else:
        C, T, Tr = data.shape
        X = np.transpose(data, (1, 0, 2))  # → (time, channels, trials)

    # 窗
    h = tftb_window(win_len, win_name)

    # 逐通道、逐 trial 计算 STFT
    ttfr = np.zeros((out_freq_bins, out_time_len, Tr, C), dtype=complex)
    for ci in range(C):
        for tj in range(Tr):
            tfr, f_norm = tfrstft(X[:, ci, tj], N=N, h=h)
            # 只取正频（0..N/2），与你 MATLAB 的 tfr(1:53,:) 一致
            # 同时裁剪时间 1:out_time_len
            tfr_pos = tfr[:out_freq_bins, :out_time_len]
            ttfr[:, :, tj, ci] = tfr_pos

    # 频率刻度（Hz）
    df = fs / float(N)
    f_hz = np.arange(out_freq_bins) * df
    t_idx = np.arange(1, out_time_len+1)  # 1..1500

    return ttfr, t_idx, f_hz
