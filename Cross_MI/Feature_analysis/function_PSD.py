
import os
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import scipy.signal as sps
import csv



# =========================
# 字体（避免中文方框）
# =========================
def set_cjk_font():
    candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Sarasa Gothic SC']
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
set_cjk_font()


def pfx(s, datatype): return f"{datatype}S0{s}" if s <= 9 else f"{datatype}S{s}"

def load_block(p):
    M = hdf5storage.loadmat(p)
    data = M['data']                    # (n_chan, n_time, n_trials)
    label = M['label'].ravel()          # (n_trials,)
    fs = float(np.squeeze(M['fs']))
    return data, label, fs, M


class QCParams:
    def __init__(self,
                 rms_z_thr=3.5,
                 flatline_std_floor_ratio=0.05,
                 line_peak_db_thr=8.0,
                 hf_ratio_thr=2.5,
                 min_good_trials=10,
                 min_good_channels=20,
                 car_before_qc=True,
                 detrend_before_qc='linear',
                 welch_win_sec=2.0,
                 welch_noverlap=0.5,
                 welch_window='hann'):
        self.rms_z_thr = rms_z_thr
        self.flatline_std_floor_ratio = flatline_std_floor_ratio
        self.line_peak_db_thr = line_peak_db_thr
        self.hf_ratio_thr = hf_ratio_thr
        self.min_good_trials = min_good_trials
        self.min_good_channels = min_good_channels
        self.car_before_qc = car_before_qc
        self.detrend_before_qc = detrend_before_qc
        self.welch_win_sec = welch_win_sec
        self.welch_noverlap = welch_noverlap
        self.welch_window = welch_window

def _qc_car(X):  # (chan,time)
    return X - X.mean(axis=0, keepdims=True)

def _qc_detrend(X, mode):
    if mode == 'linear':
        return sps.detrend(X, axis=-1, type='linear')
    elif mode == 'constant':
        return X - X.mean(axis=-1, keepdims=True)
    return X

def _robust_z(x):
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med)) + 1e-12
    return (x - med) / mad

def _auto_line_freq(fs, f, P):
    def band_snr(center):
        m  = (f >= center-1) & (f <= center+1)
        nb = (f >= center-3) & (f <= center+3) & (~m)
        num = np.nanmean(P[..., m], axis=-1)
        den = np.nanmean(P[..., nb], axis=-1) + 1e-15
        return 10*np.log10(num/den)
    s50 = band_snr(50) if fs/2 >= 55 else -np.inf
    s60 = band_snr(60) if fs/2 >= 65 else -np.inf
    # 用中位 SNR 更稳
    s50m = np.nanmedian(s50) if np.all(np.isfinite(s50)) else -np.inf
    s60m = np.nanmedian(s60) if np.all(np.isfinite(s60)) else -np.inf
    return 60 if s60m > s50m else 50

def qc_detect_trials_channels(data_chn_time_trl, fs, params: QCParams):
    """
    输入: data( n_chan, n_time, n_trials )
    输出:
      keep_trials(bool[N]), keep_chans(bool[C]),
      qc_info(dict) —— 包含各类指标（用于 CSV 导出）
    """
    C, T, N = data_chn_time_trl.shape
    X = data_chn_time_trl.copy()

    # 只用于 QC 计算的预处理
    for tr in range(N):
        seg = X[:, :, tr]
        if params.car_before_qc:        seg = _qc_car(seg)
        if params.detrend_before_qc:    seg = _qc_detrend(seg, params.detrend_before_qc)
        X[:, :, tr] = seg

    # —— Trial 级：RMS / Var 异常 —— #
    trial_rms = np.sqrt(np.nanmean(X**2, axis=(0,1)))
    trial_var = np.nanvar(X, axis=(0,1))
    z_rms = _robust_z(trial_rms)
    z_var = _robust_z(trial_var)
    bad_trial_idx = np.where((np.abs(z_rms) > params.rms_z_thr) | (np.abs(z_var) > params.rms_z_thr))[0]

    # —— Channel 级：平线 + 频域指标 —— #
    # 平线：通道在所有 trial 上的时域 std 中位数过小
    chan_std_med = np.median(np.std(X, axis=1, ddof=1), axis=1)
    floor = np.median(chan_std_med) * params.flatline_std_floor_ratio
    flat_ch = np.where(chan_std_med < floor)[0]

    # 频域 Welch
    nper = int(round(fs * params.welch_win_sec))
    nper = max(16, 1 << int(np.floor(np.log2(max(16, nper)))))
    nper = min(nper, T)
    nover = int(min(nper-1, nper * params.welch_noverlap))
    f, P = sps.welch(np.transpose(X, (0,2,1)), fs=fs, window=params.welch_window,
                     nperseg=nper, noverlap=nover, detrend=False,
                     scaling='density', axis=-1, return_onesided=True)  # -> (C, N, F)
    Pm = np.nanmedian(P, axis=1)  # (C, F)

    # 工频
    line_freq = _auto_line_freq(fs, f, Pm)
    m_line = (f >= line_freq-1) & (f <= line_freq+1)
    m_nb   = (f >= line_freq-3) & (f <= line_freq+3) & (~m_line)
    line_snr_db = 10*np.log10(np.nanmean(Pm[:, m_line], axis=-1) / (np.nanmean(Pm[:, m_nb], axis=-1)+1e-15))
    line_ch = np.where(line_snr_db > params.line_peak_db_thr)[0]

    # 肌电：高频/α 比
    m_hf = (f >= 30) & (f <= min(40, fs/2-1))
    m_al = (f >= 8)  & (f <= 13)
    hf_ratio = (np.nanmean(Pm[:, m_hf], axis=-1) + 1e-15) / (np.nanmean(Pm[:, m_al], axis=-1) + 1e-15)
    emg_ch = np.where(hf_ratio > params.hf_ratio_thr)[0]

    bad_ch_idx = np.unique(np.concatenate([flat_ch, line_ch, emg_ch])).astype(int)

    keep_trials = np.ones(N, dtype=bool); keep_trials[bad_trial_idx] = False
    keep_chans  = np.ones(C, dtype=bool); keep_chans[bad_ch_idx]     = False

    qc_info = dict(
        n_chan_raw=C, n_trials_raw=N,
        bad_trial_idx=bad_trial_idx.tolist(),
        bad_ch_idx=bad_ch_idx.tolist(),
        flat_ch=flat_ch.tolist(),
        line_ch=line_ch.tolist(),
        emg_ch=emg_ch.tolist(),
        line_freq_used=int(line_freq),
        line_snr_db=line_snr_db.tolist(),   # len=C
        hf_ratio=hf_ratio.tolist(),         # len=C
        trial_rms_z=z_rms.tolist(),         # len=N
        trial_var_z=z_var.tolist(),         # len=N
        n_good_trials=int(keep_trials.sum()),
        n_good_channels=int(keep_chans.sum()),
        flag_drop_file=(keep_trials.sum() < params.min_good_trials or keep_chans.sum() < params.min_good_channels)
    )
    return keep_trials, keep_chans, qc_info

def _join_ints(lst):   # 辅助把列表写进 CSV
    return ';'.join(str(int(x)) for x in lst)

def write_qc_csvs(out_dir, file_logs, chan_logs, trial_logs, ch_names=None):
    os.makedirs(out_dir, exist_ok=True)
    # 文件级汇总
    fpath = os.path.join(out_dir, "QC_files_summary.csv")
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        fields = ["subj","scene","stim","fs",
                  "n_chan_raw","n_trials_raw","n_good_channels","n_good_trials",
                  "flag_drop_file","line_freq_used",
                  "bad_ch_idx","bad_trial_idx","flat_ch","line_ch","emg_ch"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in file_logs:
            writer.writerow({
                "subj": r["subj"], "scene": r["scene"], "stim": r["stim"], "fs": r["fs"],
                "n_chan_raw": r["qc"]["n_chan_raw"], "n_trials_raw": r["qc"]["n_trials_raw"],
                "n_good_channels": r["qc"]["n_good_channels"], "n_good_trials": r["qc"]["n_good_trials"],
                "flag_drop_file": int(r["qc"]["flag_drop_file"]),
                "line_freq_used": r["qc"]["line_freq_used"],
                "bad_ch_idx": _join_ints(r["qc"]["bad_ch_idx"]),
                "bad_trial_idx": _join_ints(r["qc"]["bad_trial_idx"]),
                "flat_ch": _join_ints(r["qc"]["flat_ch"]),
                "line_ch": _join_ints(r["qc"]["line_ch"]),
                "emg_ch": _join_ints(r["qc"]["emg_ch"]),
            })
    print(f"[QC] 已写出 {fpath}")

    # 通道级细节
    cpath = os.path.join(out_dir, "QC_channels_detail.csv")
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        fields = ["subj","scene","stim","ch_idx","ch_name",
                  "line_snr_db","hf_ratio","flag_flat","flag_line","flag_emg"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in chan_logs:
            chname = ch_names[r["ch_idx"]] if (ch_names is not None and 0 <= r["ch_idx"] < len(ch_names)) else f"Ch{r['ch_idx']+1}"
            writer.writerow({
                "subj": r["subj"], "scene": r["scene"], "stim": r["stim"],
                "ch_idx": r["ch_idx"], "ch_name": chname,
                "line_snr_db": f"{r['line_snr_db']:.3f}",
                "hf_ratio": f"{r['hf_ratio']:.3f}",
                "flag_flat": int(r["flag_flat"]), "flag_line": int(r["flag_line"]), "flag_emg": int(r["flag_emg"]),
            })
    print(f"[QC] 已写出 {cpath}")

    # 试次级细节
    tpath = os.path.join(out_dir, "QC_trials_detail.csv")
    with open(tpath, "w", newline="", encoding="utf-8") as f:
        fields = ["subj","scene","stim","trial_idx","rms_z","var_z","flag_bad"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in trial_logs:
            writer.writerow({
                "subj": r["subj"], "scene": r["scene"], "stim": r["stim"],
                "trial_idx": r["trial_idx"],
                "rms_z": f"{r['rms_z']:.3f}", "var_z": f"{r['var_z']:.3f}",
                "flag_bad": int(r["flag_bad"])
            })
    print(f"[QC] 已写出 {tpath}")


