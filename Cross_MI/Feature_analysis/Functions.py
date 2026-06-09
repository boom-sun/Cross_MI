
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import scipy.signal as sps
from scipy.stats import ttest_1samp, wilcoxon
import csv
import hdf5storage
from pathlib import Path
from joblib import Parallel, delayed
import mne
from mne.stats import permutation_cluster_1samp_test
from scipy.io import savemat
# =========================
# 字体设置
# =========================
def set_cjk_font():
    """设置中文字体以避免中文显示为方框"""
    candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Sarasa Gothic SC']
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


# =========================
# 文件操作函数
# =========================
def pfx(s, datatype):
    """生成被试文件名前缀"""
    return f"{datatype}S0{s}" if s <= 9 else f"{datatype}S{s}"


def load_block(p):
    """加载.mat数据块"""
    M = hdf5storage.loadmat(p)
    data = M['data']  # (n_chan, n_time, n_trials)
    label = M['label'].ravel()  # (n_trials,)
    fs = float(np.squeeze(M['fs']))
    return data, label, fs, M


# =========================
# 质量控制相关函数
# =========================
class QCParams:
    """质量控制参数类"""

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


def _qc_car(X):
    """平均参考（CAR）"""
    return X - X.mean(axis=0, keepdims=True)


def _qc_detrend(X, mode):
    """去趋势处理"""
    if mode == 'linear':
        return sps.detrend(X, axis=-1, type='linear')
    elif mode == 'constant':
        return X - X.mean(axis=-1, keepdims=True)
    return X


def _robust_z(x):
    """稳健Z-score计算（基于中位数和MAD）"""
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med)) + 1e-12
    return (x - med) / mad


def _auto_line_freq(fs, f, P):
    """自动检测工频（50Hz或60Hz）"""

    def band_snr(center):
        m = (f >= center - 1) & (f <= center + 1)
        nb = (f >= center - 3) & (f <= center + 3) & (~m)
        num = np.nanmean(P[..., m], axis=-1)
        den = np.nanmean(P[..., nb], axis=-1) + 1e-15
        return 10 * np.log10(num / den)

    s50 = band_snr(50) if fs / 2 >= 55 else -np.inf
    s60 = band_snr(60) if fs / 2 >= 65 else -np.inf

    # 用中位 SNR 更稳
    s50m = np.nanmedian(s50) if np.all(np.isfinite(s50)) else -np.inf
    s60m = np.nanmedian(s60) if np.all(np.isfinite(s60)) else -np.inf
    return 60 if s60m > s50m else 50


def qc_detect_trials_channels(data_chn_time_trl, fs, params: QCParams):
    """
    检测需要排除的试次和通道

    参数:
        data_chn_time_trl: EEG数据 (n_chan, n_time, n_trials)
        fs: 采样率
        params: QC参数对象

    返回:
        keep_trials: 保留的试次掩码
        keep_chans: 保留的通道掩码
        qc_info: 质量控制信息字典
    """
    C, T, N = data_chn_time_trl.shape
    X = data_chn_time_trl.copy()

    # 只用于 QC 计算的预处理
    for tr in range(N):
        seg = X[:, :, tr]
        if params.car_before_qc:
            seg = _qc_car(seg)
        if params.detrend_before_qc:
            seg = _qc_detrend(seg, params.detrend_before_qc)
        X[:, :, tr] = seg

    # Trial 级：RMS / Var 异常
    trial_rms = np.sqrt(np.nanmean(X ** 2, axis=(0, 1)))
    trial_var = np.nanvar(X, axis=(0, 1))
    z_rms = _robust_z(trial_rms)
    z_var = _robust_z(trial_var)
    bad_trial_idx = np.where((np.abs(z_rms) > params.rms_z_thr) | (np.abs(z_var) > params.rms_z_thr))[0]

    # Channel 级：平线 + 频域指标
    # 平线检测
    chan_std_med = np.median(np.std(X, axis=1, ddof=1), axis=1)
    floor = np.median(chan_std_med) * params.flatline_std_floor_ratio
    flat_ch = np.where(chan_std_med < floor)[0]

    # 频域 Welch 分析
    nper = int(round(fs * params.welch_win_sec))
    nper = max(16, 1 << int(np.floor(np.log2(max(16, nper)))))
    nper = min(nper, T)
    nover = int(min(nper - 1, nper * params.welch_noverlap))
    f, P = sps.welch(np.transpose(X, (0, 2, 1)), fs=fs, window=params.welch_window,
                     nperseg=nper, noverlap=nover, detrend=False,
                     scaling='density', axis=-1, return_onesided=True)  # -> (C, N, F)
    Pm = np.nanmedian(P, axis=1)  # (C, F)

    # 工频干扰检测
    line_freq = _auto_line_freq(fs, f, Pm)
    m_line = (f >= line_freq - 1) & (f <= line_freq + 1)
    m_nb = (f >= line_freq - 3) & (f <= line_freq + 3) & (~m_line)
    line_snr_db = 10 * np.log10(np.nanmean(Pm[:, m_line], axis=-1) / (np.nanmean(Pm[:, m_nb], axis=-1) + 1e-15))
    line_ch = np.where(line_snr_db > params.line_peak_db_thr)[0]

    # 肌电干扰检测（高频/α 比）
    m_hf = (f >= 30) & (f <= min(40, fs / 2 - 1))
    m_al = (f >= 8) & (f <= 13)
    hf_ratio = (np.nanmean(Pm[:, m_hf], axis=-1) + 1e-15) / (np.nanmean(Pm[:, m_al], axis=-1) + 1e-15)
    emg_ch = np.where(hf_ratio > params.hf_ratio_thr)[0]

    bad_ch_idx = np.unique(np.concatenate([flat_ch, line_ch, emg_ch])).astype(int)

    keep_trials = np.ones(N, dtype=bool);
    keep_trials[bad_trial_idx] = False
    keep_chans = np.ones(C, dtype=bool);
    keep_chans[bad_ch_idx] = False

    qc_info = dict(
        n_chan_raw=C, n_trials_raw=N,
        bad_trial_idx=bad_trial_idx.tolist(),
        bad_ch_idx=bad_ch_idx.tolist(),
        flat_ch=flat_ch.tolist(),
        line_ch=line_ch.tolist(),
        emg_ch=emg_ch.tolist(),
        line_freq_used=int(line_freq),
        line_snr_db=line_snr_db.tolist(),  # len=C
        hf_ratio=hf_ratio.tolist(),  # len=C
        trial_rms_z=z_rms.tolist(),  # len=N
        trial_var_z=z_var.tolist(),  # len=N
        n_good_trials=int(keep_trials.sum()),
        n_good_channels=int(keep_chans.sum()),
        flag_drop_file=(keep_trials.sum() < params.min_good_trials or keep_chans.sum() < params.min_good_channels)
    )
    return keep_trials, keep_chans, qc_info


def _join_ints(lst):
    """将整数列表转换为分号分隔的字符串"""
    return ';'.join(str(int(x)) for x in lst)


def write_qc_csvs(out_dir, file_logs, chan_logs, trial_logs, ch_names=None):
    """写出质量控制报告到CSV文件"""
    os.makedirs(out_dir, exist_ok=True)

    # 文件级汇总
    fpath = os.path.join(out_dir, "QC_files_summary.csv")
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        fields = ["subj", "scene", "stim", "fs",
                  "n_chan_raw", "n_trials_raw", "n_good_channels", "n_good_trials",
                  "flag_drop_file", "line_freq_used",
                  "bad_ch_idx", "bad_trial_idx", "flat_ch", "line_ch", "emg_ch"]
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
        fields = ["subj", "scene", "stim", "ch_idx", "ch_name",
                  "line_snr_db", "hf_ratio", "flag_flat", "flag_line", "flag_emg"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in chan_logs:
            chname = ch_names[r["ch_idx"]] if (
                        ch_names is not None and 0 <= r["ch_idx"] < len(ch_names)) else f"Ch{r['ch_idx'] + 1}"
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
        fields = ["subj", "scene", "stim", "trial_idx", "rms_z", "var_z", "flag_bad"]
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


# =========================
# 数据预处理函数
# =========================
def _apply_car(X):
    """平均参考（CAR）"""
    return X - X.mean(axis=0, keepdims=True)


def _maybe_detrend(X, mode='linear'):
    """去趋势处理"""
    if mode == 'linear':
        return sps.detrend(X, axis=-1, type='linear')
    elif mode == 'constant':
        return X - X.mean(axis=-1, keepdims=True)
    else:
        return X


# =========================
# PSD计算函数
# =========================
def compute_subject_psd_by_channel_fast(
        data_chn_time_trl, labels, target_cls, fs,
        fmin, fmax,
        welch_win_sec=2.0, welch_noverlap=0.5, welch_window='hann',
        use_car=True, detrend_mode='linear',
        bad_ch_idx=None, car_good_only=True,
        RestDuration=2.0, TaskDuration=4.0, baseline_mode='db'
):
    """
    快速计算单个被试的PSD

    参数:
        data_chn_time_trl: EEG数据 (n_chan, n_time, n_trials)
        labels: 试次标签
        target_cls: 目标类别
        fs: 采样率
        fmin, fmax: 频率范围
        ... 其他PSD计算参数

    返回:
        psd_avg: 平均PSD (n_chan, n_freqs)
        freqs: 频率数组
    """
    idx = np.where(labels == target_cls)[0]
    if idx.size == 0:
        return None, None

    C, n_time, _ = data_chn_time_trl.shape
    bad_mask = np.zeros(C, dtype=bool)
    if bad_ch_idx is not None and len(bad_ch_idx) > 0:
        bad_mask[np.asarray(bad_ch_idx, dtype=int)] = True
    good_mask = ~bad_mask

    # 任务/基线窗口
    t0 = int(round(RestDuration * fs))
    t1 = min(n_time, t0 + int(round(TaskDuration * fs)))
    L = max(0, t1 - t0)
    b0 = max(0, t0 - L);
    b1 = t0

    # 取该类所有 trial → (n_trials, C, T)，float32
    task = data_chn_time_trl[:, t0:t1, :][:, :, idx].transpose(2, 0, 1).astype(np.float32, copy=False)
    base = data_chn_time_trl[:, b0:b1, :][:, :, idx].transpose(2, 0, 1).astype(np.float32, copy=False)

    # CAR：仅用好通道估计参考均值，减到所有通道
    if use_car:
        if car_good_only and np.any(good_mask):
            m_task = task[:, good_mask, :].mean(axis=1, keepdims=True)
            m_base = base[:, good_mask, :].mean(axis=1, keepdims=True)
        else:
            m_task = task.mean(axis=1, keepdims=True)
            m_base = base.mean(axis=1, keepdims=True)
        task -= m_task;
        base -= m_base

    # 去趋势
    if detrend_mode == 'linear':
        task = sps.detrend(task, axis=-1, type='linear')
        base = sps.detrend(base, axis=-1, type='linear')
    elif detrend_mode == 'constant':
        task -= task.mean(axis=-1, keepdims=True)
        base -= base.mean(axis=-1, keepdims=True)

    # Welch 参数（2 的幂更快）
    nper = int(round(fs * welch_win_sec))
    nper = max(16, 1 << int(np.floor(np.log2(max(16, nper)))))
    nper = min(nper, task.shape[-1], base.shape[-1])
    nover = int(min(nper - 1, nper * welch_noverlap))

    # 批量 Welch
    f_task, P_task = sps.welch(task, fs=fs, window=welch_window,
                               nperseg=nper, noverlap=nover,
                               detrend=False, return_onesided=True,
                               scaling='density', axis=-1)
    f_base, P_base = sps.welch(base, fs=fs, window=welch_window,
                               nperseg=nper, noverlap=nover,
                               detrend=False, return_onesided=True,
                               scaling='density', axis=-1)

    # 频率截取
    fmask = (f_task >= fmin) & (f_task <= fmax)
    freqs = f_task[fmask]
    P_task = P_task[..., fmask]
    P_base = P_base[..., fmask]

    eps = 1e-15
    P_task = np.maximum(P_task, eps)
    P_base = np.maximum(P_base, eps)

    # 基线校正
    if baseline_mode == 'db':
        spec = 10.0 * np.log10(P_task / P_base)  # (n_trials, C, F)
    elif baseline_mode == 'pct':
        spec = (P_task - P_base) / P_base
    else:
        spec = P_task

    # trial 平均
    psd_avg = np.nanmean(spec, axis=0)  # (C, F)

    # 坏通道置 NaN（保持通道维度一致）
    if np.any(bad_mask):
        psd_avg[bad_mask, :] = np.nan

    return psd_avg, freqs


def process_one_subject(s, cfg):
    """
    处理单个被试的所有数据

    返回: (psd_results, file_logs, chan_logs, trial_logs)
    """
    psd_results = []
    file_logs, chan_logs, trial_logs = [], [], []
    subj_id = pfx(s, cfg['datatype'])

    qc_params = QCParams(
        rms_z_thr=3.5,
        flatline_std_floor_ratio=0.05,
        line_peak_db_thr=8.0,
        hf_ratio_thr=2.5,
        min_good_trials=10,
        min_good_channels=20,
        car_before_qc=True,
        detrend_before_qc='linear',
        welch_win_sec=cfg['welch_win_sec'],
        welch_noverlap=cfg['welch_noverlap'],
        welch_window=cfg['welch_window']
    )

    for st in cfg['stim_name']:
        path_S1 = os.path.join(cfg['data_4_filepath'], f"{subj_id}S1_{st}.mat")
        path_S2 = os.path.join(cfg['data_4_filepath'], f"{subj_id}S2_{st}.mat")
        if not (os.path.isfile(path_S1) and os.path.isfile(path_S2)):
            miss = []
            if not os.path.isfile(path_S1): miss.append('S1')
            if not os.path.isfile(path_S2): miss.append('S2')
            print(f"[跳过] {subj_id}, stim={st}, 缺失: {','.join(miss)}")
            continue

        try:
            # 注意：这里使用自定义的ERPs_Filter，请确保已安装
            from eeg_filter import ERPs_Filter

            data1, lab1, fs1, _ = load_block(path_S1)
            data2, lab2, fs2, _ = load_block(path_S2)

            data1_f = ERPs_Filter(data1, freqs=cfg['freqwindow'], fs=fs1, filterflag='filtfilt').astype(np.float32,
                                                                                                     copy=False)
            data2_f = ERPs_Filter(data2, freqs=cfg['freqwindow'], fs=fs2, filterflag='filtfilt').astype(np.float32,
                                                                                                     copy=False)

        except Exception as e:
            print(f"[处理失败] {subj_id}, stim={st}: {e}")
            continue

        # S1 / S2 分别做 QC，并记录日志
        for dat, lab, fs, scene_tag in [(data1_f, lab1, fs1, 'S1'), (data2_f, lab2, fs2, 'S2')]:
            keep_trials, keep_ch, qc = qc_detect_trials_channels(dat, fs, qc_params)

            # 文件级日志
            file_logs.append(dict(subj=subj_id, scene=scene_tag, stim=st, fs=fs, qc=qc))

            # 通道级日志
            for ci in range(dat.shape[0]):
                chan_logs.append(dict(
                    subj=subj_id, scene=scene_tag, stim=st, ch_idx=ci,
                    line_snr_db=qc["line_snr_db"][ci], hf_ratio=qc["hf_ratio"][ci],
                    flag_flat=(ci in qc["flat_ch"]),
                    flag_line=(ci in qc["line_ch"]),
                    flag_emg=(ci in qc["emg_ch"])
                ))

            # 试次级日志
            for ti in range(dat.shape[2]):
                trial_logs.append(dict(
                    subj=subj_id, scene=scene_tag, stim=st, trial_idx=ti,
                    rms_z=qc["trial_rms_z"][ti], var_z=qc["trial_var_z"][ti],
                    flag_bad=(ti in qc["bad_trial_idx"])
                ))

            # 应用 QC 掩码
            bad_ch_idx = np.where(~keep_ch)[0]
            dat = dat[:, :, keep_trials]
            lab = lab[keep_trials]

            # PSD 计算
            for csel in (1, 2):
                psd, f = compute_subject_psd_by_channel_fast(
                    dat, lab, target_cls=csel, fs=fs,
                    fmin=cfg['freqwindow'][0], fmax=cfg['freqwindow'][1],
                    welch_win_sec=cfg['welch_win_sec'],
                    welch_noverlap=cfg['welch_noverlap'],
                    welch_window=cfg['welch_window'],
                    use_car=cfg['use_car'],
                    detrend_mode=cfg['welch_detrend'],
                    bad_ch_idx=bad_ch_idx,
                    car_good_only=True,
                    RestDuration=cfg['RestDuration'],
                    TaskDuration=cfg['TaskDuration'],
                    baseline_mode=cfg['baseline_mode']
                )
                if psd is not None:
                    key = (scene_tag, st)
                    psd_results.append((csel, key, psd, f, subj_id))

    return psd_results, file_logs, chan_logs, trial_logs


# =========================
# 数据管理函数
# =========================
def interp_to_ref(psd, freqs, freqs_ref):
    """将PSD插值到参考频率轴"""
    if psd is None: return None
    if np.array_equal(freqs, freqs_ref): return psd
    out = np.empty((psd.shape[0], len(freqs_ref)), dtype=float)
    for ch in range(psd.shape[0]):
        out[ch] = np.interp(freqs_ref, freqs, psd[ch])
    return out


def initialize_data_containers(scenes, stim_name):
    """初始化数据容器"""
    group_raw = {1: {}, 2: {}}
    freqs_map = {1: {}, 2: {}}
    group_subj = {1: {}, 2: {}}
    group_mean = {1: {}, 2: {}}
    group_std = {1: {}, 2: {}}
    group_n = {1: {}, 2: {}}

    for cls in (1, 2):
        for sc in scenes:
            for st in stim_name:
                key = (sc, st)
                group_raw[cls][key] = []
                freqs_map[cls][key] = None
                group_subj[cls][key] = []
                group_mean[cls][key] = None
                group_std[cls][key] = None
                group_n[cls][key] = 0

    return group_raw, freqs_map, group_subj, group_mean, group_std, group_n


def aggregate_results(par_outputs):
    """汇总并行处理结果"""
    all_psd_results = []
    all_file_logs, all_chan_logs, all_trial_logs = [], [], []

    for psd_results, file_logs, chan_logs, trial_logs in par_outputs:
        all_psd_results.extend(psd_results)
        all_file_logs.extend(file_logs)
        all_chan_logs.extend(chan_logs)
        all_trial_logs.extend(trial_logs)

    return all_psd_results, all_file_logs, all_chan_logs, all_trial_logs


def organize_data_to_containers(all_psd_results, group_raw, freqs_map, group_subj, scenes, stim_name):
    """将PSD结果组织到数据容器中"""
    for (csel, key, psd, f, subj_id) in all_psd_results:
        if freqs_map[csel][key] is None:
            freqs_map[csel][key] = f
        group_raw[csel][key].append(interp_to_ref(psd, f, freqs_map[csel][key]))
        group_subj[csel][key].append(subj_id)


def compute_group_statistics(group_raw, group_mean, group_std, group_n, scenes, stim_name):
    """计算组水平统计量"""
    for cls in (1, 2):
        for key in group_raw[cls]:
            curves = group_raw[cls][key]  # list of (n_chan, n_freqs)，内部可能含 NaN
            if len(curves) == 0:
                group_mean[cls][key] = None
                group_std[cls][key] = None
                group_n[cls][key] = 0
            else:
                try:
                    stack = np.stack(curves, axis=0)  # (n_subj, n_chan, n_freqs)
                except Exception as e:
                    shapes = [getattr(a, 'shape', None) for a in curves]
                    raise RuntimeError(f"堆叠失败：存在形状不一致 {shapes}") from e
                group_mean[cls][key] = np.nanmean(stack, axis=0)
                group_std[cls][key] = np.nanstd(stack, axis=0, ddof=1)
                # 有效被试数（至少某个频点非 NaN）
                valid = ~np.all(np.isnan(stack), axis=(1, 2))
                group_n[cls][key] = int(valid.sum())


# =========================
# 统计函数
# =========================
def by_fdr(pvals, q=0.05):
    """Benjamini-Yekutieli FDR校正"""
    p = np.asarray(pvals).ravel()
    ok = np.isfinite(p)
    if not np.any(ok): return np.zeros_like(p, dtype=bool).reshape(pvals.shape)
    p_ok = p[ok]
    m = p_ok.size
    order = np.argsort(p_ok)
    pv = p_ok[order]
    c_m = np.sum(1.0 / np.arange(1, m + 1))  # BY 修正项
    thresh = q * (np.arange(1, m + 1) / (m * c_m))
    is_sig_sorted = pv <= thresh
    sig_ok = np.zeros_like(p_ok, dtype=bool)
    if np.any(is_sig_sorted):
        k = np.max(np.where(is_sig_sorted)[0])
        cutoff = thresh[k]
        sig_ok = (p_ok <= cutoff)
    sig = np.zeros_like(p, dtype=bool);
    sig[np.where(ok)[0]] = sig_ok
    return sig.reshape(pvals.shape)


def cohens_d(x, axis=0):
    """计算Cohen's d效应量"""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1) + 1e-12
    return mu / sd


def contiguous_regions(mask):
    """找到连续True区域的起始和结束索引"""
    mask = np.asarray(mask, dtype=bool)
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]: starts = [0] + starts
    if mask[-1]: ends = ends + [len(mask)]
    return list(zip(starts, ends))


# =========================
# 工具函数
# =========================
def normalize_chan_select(chan_select, n_chan):
    """标准化通道选择"""
    if chan_select is None: return list(range(n_chan))
    arr = np.asarray(chan_select)
    if arr.ndim == 0: arr = arr[None]
    arr = arr.ravel()
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("chan_select 必须是数值索引")
    if not np.all(np.isfinite(arr)):
        raise ValueError("chan_select 含 NaN/Inf")
    if np.any(np.abs(arr - np.round(arr)) > 1e-6):
        raise ValueError(f"需为整数索引：{arr}")
    arr = np.round(arr).astype(int)
    bad = arr[(arr < 0) | (arr > n_chan)]
    if bad.size > 0:
        print(f"[提示] 有通道索引超界被移除：{bad.tolist()}（总通道={n_chan}）")
    arr = np.unique(arr[(arr >= 0) & (arr <= n_chan)])
    return arr.tolist() if arr.size else list(range(n_chan))


def finalize_legend_labels(ch_idx, legend_labels=None):
    """生成最终图例标签"""

    def default_name(c):
        return f"Ch{c + 1}"

    if legend_labels is None:
        names = [default_name(c) for c in ch_idx]
    elif isinstance(legend_labels, dict):
        names = [str(legend_labels.get(c, default_name(c))) for c in ch_idx]
    else:
        arr = list(legend_labels);
        names = []
        for i, c in enumerate(ch_idx):
            names.append(str(arr[i]) if i < len(arr) else default_name(c))
    # 唯一化
    seen = {};
    out = []
    for nm, c in zip(names, ch_idx):
        if nm in seen:
            seen[nm] += 1;
            out.append(f"{nm} (ch {c + 1})")
        else:
            seen[nm] = 1;
            out.append(nm)
    return out


def make_channel_colors(n):
    """生成通道颜色"""
    base = list(plt.get_cmap('tab20').colors)
    return base[:n] if n <= len(base) else [plt.get_cmap('hsv')(i / n) for i in range(n)]

# =========================
# Trial-level outlier removal
# =========================
class TrialOutlierParams:
    def __init__(
            self,
            method='robust_z',
            robust_z_thr=3.5,
            z_thr=3.0,
            iqr_k=1.5,
            pct_low=1,
            pct_high=99,
            min_keep=5,
            verbose=True):
        self.method = method
        self.robust_z_thr = robust_z_thr
        self.z_thr = z_thr
        self.iqr_k = iqr_k
        self.pct_low = pct_low
        self.pct_high = pct_high
        self.min_keep = min_keep
        self.verbose = verbose


def detect_trial_outliers(x, params: TrialOutlierParams, trial_axis=-2):
    """
    对单个被试、单个类别、单个场景的数据做 trial-level 离群值检测

    推荐输入:
        ERSP shape = [freq, time, channel, trial]

    返回:
        keep_trials: bool mask
        info: 删除统计信息
    """
    x = np.asarray(x, dtype=float)

    # 把 trial 维移动到第 0 维，方便统一处理
    x_trial_first = np.moveaxis(x, trial_axis, 0)   # [trial, ...]

    # 每个 trial 压缩成一个分数
    # 分数越大，说明该 trial 整体能量越异常
    score = np.nanmean(
        np.abs(x_trial_first),
        axis=tuple(range(1, x_trial_first.ndim))
    )

    n_raw = x_trial_first.shape[0]

    if params.method == 'robust_z':
        med = np.nanmedian(score)
        mad = 1.4826 * np.nanmedian(np.abs(score - med)) + 1e-12
        metric = (score - med) / mad
        keep_trials = np.abs(metric) <= params.robust_z_thr

    elif params.method == 'zscore':
        mu = np.nanmean(score)
        sd = np.nanstd(score, ddof=1) + 1e-12
        metric = (score - mu) / sd
        keep_trials = np.abs(metric) <= params.z_thr

    elif params.method == 'iqr':
        q1 = np.nanpercentile(score, 25)
        q3 = np.nanpercentile(score, 75)
        iqr = q3 - q1
        low = q1 - params.iqr_k * iqr
        high = q3 + params.iqr_k * iqr
        metric = score
        keep_trials = (score >= low) & (score <= high)

    elif params.method == 'percentile':
        low = np.nanpercentile(score, params.pct_low)
        high = np.nanpercentile(score, params.pct_high)
        metric = score
        keep_trials = (score >= low) & (score <= high)

    else:
        raise ValueError(f'Unknown outlier method: {params.method}')

    n_keep = int(keep_trials.sum())
    n_removed = int((~keep_trials).sum())

    # 防止某个类别 trial 太少时被删光
    if n_keep < params.min_keep:
        if params.verbose:
            print(f'[TrialOutlier] 保留试次数过少: {n_keep} < {params.min_keep}，取消删除')
        keep_trials = np.ones(n_raw, dtype=bool)
        n_keep = int(keep_trials.sum())
        n_removed = 0

    info = {
        'method': params.method,
        'n_total': int(n_raw),
        'n_keep': int(n_keep),
        'n_removed': int(n_removed),
        'remove_ratio': float(n_removed / n_raw),
        'bad_trials_idx': np.where(~keep_trials)[0].tolist(),

        # 分数信息，方便后期检查
        'score_mean': float(np.nanmean(score)),
        'score_std': float(np.nanstd(score)),
        'score_median': float(np.nanmedian(score)),
        'score_min': float(np.nanmin(score)),
        'score_max': float(np.nanmax(score))
    }

    return keep_trials, info


def apply_trial_outlier_removal(
        x,
        params: TrialOutlierParams,
        trial_axis=-1,
        subject=None,
        sence_name=None,
        class_name=None,
        stim_name=None):
    """
    应用 trial-level 离群值删除，并把统计信息写入 info

    参数:
        subject: 被试编号
        sence_name: sence1 / sence2
        class_name: 类别编号
        stim_name: ssvideo / video / ssmvep / cue，可选
    """
    keep_trials, info = detect_trial_outliers(
        x,
        params=params,
        trial_axis=trial_axis
    )

    x_trial_first = np.moveaxis(x, trial_axis, 0)
    x_clean = x_trial_first[keep_trials]
    x_clean = np.moveaxis(x_clean, 0, trial_axis)

    # 把身份信息也放进 info
    info['subject'] = subject
    info['sence'] = sence_name
    info['class'] = class_name
    info['stim'] = stim_name

    if params.verbose:
        label = f"sub{subject}_{sence_name}"
        if stim_name is not None:
            label += f"_{stim_name}"
        label += f"_class{class_name}"

        print(
            f"[TrialOutlier] {label}: "
            f"remove {info['n_removed']}/{info['n_total']} trials "
            f"({info['remove_ratio'] * 100:.2f}%)"
        )

    return x_clean, keep_trials, info
# =========================
# 画图前离群值去除函数
# =========================
class OutlierParams:
    """画图阶段离群值检测参数类"""
    def __init__(
            self,
            method='robust_z',
            z_thr=3.0,
            robust_z_thr=3.5,
            iqr_k=1.5,
            pct_low=1,
            pct_high=99,
            axis=0,
            min_keep=5,
            verbose=True):
        self.method = method
        self.z_thr = z_thr
        self.robust_z_thr = robust_z_thr
        self.iqr_k = iqr_k
        self.pct_low = pct_low
        self.pct_high = pct_high
        self.axis = axis
        self.min_keep = min_keep
        self.verbose = verbose


def _plot_robust_z(x, axis=0):
    """稳健Z-score：基于中位数和MAD"""
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(x - med), axis=axis, keepdims=True) + 1e-12
    return (x - med) / mad


def _get_subject_score(data, reduce_axis=None):
    """
    将每个被试压缩成一个离群值评分

    参数:
        data: np.ndarray，默认第0维为被试维
              例如:
              topo: (n_subj, n_chan)
              tf:   (n_subj, n_freq, n_time)
              psd:  (n_subj, n_chan, n_freq)
        reduce_axis: 被试维之外需要平均的维度

    返回:
        score: (n_subj,)
    """
    data = np.asarray(data, dtype=float)

    if reduce_axis is None:
        reduce_axis = tuple(range(1, data.ndim))

    score = np.nanmean(np.abs(data), axis=reduce_axis)
    return score


def detect_plot_outliers(data, params: OutlierParams, reduce_axis=None):
    """
    检测画图阶段的离群被试

    返回:
        keep_subj: 保留被试mask，shape=(n_subj,)
        outlier_info: 离群值信息
    """
    data = np.asarray(data, dtype=float)
    score = _get_subject_score(data, reduce_axis=reduce_axis)

    if params.method == 'zscore':
        mu = np.nanmean(score)
        sd = np.nanstd(score, ddof=1) + 1e-12
        z = (score - mu) / sd
        keep_subj = np.abs(z) <= params.z_thr
        metric = z

    elif params.method == 'robust_z':
        z = _plot_robust_z(score, axis=0)
        keep_subj = np.abs(z) <= params.robust_z_thr
        metric = z

    elif params.method == 'iqr':
        q1 = np.nanpercentile(score, 25)
        q3 = np.nanpercentile(score, 75)
        iqr = q3 - q1
        low = q1 - params.iqr_k * iqr
        high = q3 + params.iqr_k * iqr
        keep_subj = (score >= low) & (score <= high)
        metric = score

    elif params.method == 'percentile':
        low = np.nanpercentile(score, params.pct_low)
        high = np.nanpercentile(score, params.pct_high)
        keep_subj = (score >= low) & (score <= high)
        metric = score

    else:
        raise ValueError(f"未知离群值检测方法: {params.method}")

    # 防止删太多
    if keep_subj.sum() < params.min_keep:
        if params.verbose:
            print(f"[Outlier] 保留被试数过少: {keep_subj.sum()} < {params.min_keep}，取消离群值去除")
        keep_subj = np.ones(data.shape[0], dtype=bool)

    outlier_info = dict(
        method=params.method,
        n_raw=int(data.shape[0]),
        n_keep=int(keep_subj.sum()),
        n_remove=int((~keep_subj).sum()),
        bad_subj_idx=np.where(~keep_subj)[0].tolist(),
        score=score.tolist(),
        metric=np.asarray(metric).ravel().tolist()
    )

    if params.verbose:
        print(f"[Outlier] method={params.method}, remove={outlier_info['bad_subj_idx']}")

    return keep_subj, outlier_info


def apply_plot_outlier_removal(data, params: OutlierParams, reduce_axis=None):
    """
    对画图数据应用离群值去除

    返回:
        data_clean: 去除离群被试后的数据
        keep_subj: 保留被试mask
        outlier_info: 离群值信息
    """
    keep_subj, outlier_info = detect_plot_outliers(
        data,
        params=params,
        reduce_axis=reduce_axis
    )
    data_clean = np.asarray(data)[keep_subj]
    return data_clean, keep_subj, outlier_info

def make_matlab_cell(x):
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = x
    return cell


def clean_trial_info(info):
    out = {}
    for k, v in info.items():
        if k == 'bad_trials_idx':
            arr = np.asarray(v).ravel()
            cell = np.empty((1, arr.size), dtype=object)
            for i, val in enumerate(arr):
                cell[0, i] = val
            out[k] = cell
        elif isinstance(v, np.generic):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = np.asarray(v)
        else:
            out[k] = v
    return out
def savemat_my(path, ersp_name, ersp_data, info):
    savemat(
        path,
        {
            ersp_name: make_matlab_cell(ersp_data.astype(np.float32)),
            'trial_outlier_info': clean_trial_info(info)
        },
        do_compression=False,
        long_field_names=True
    )