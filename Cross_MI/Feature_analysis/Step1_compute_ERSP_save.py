# Step1_compute_ERSP_save.py
import os
import numpy as np
from pathlib import Path
import hdf5storage
from eeg_filter import ERPs_Filter
from eeg_stft import m_tfr
from scipy.signal import butter, cheby1, filtfilt, resample
from Functions import TrialOutlierParams, apply_trial_outlier_removal, savemat_my

# ===== 参数 =====
datatype = 1
srate = 250
freqwindow = [1, 40]
frames = 1500
TaskDuration = 4
RestDuration = 2
tlimits = [-1000*RestDuration, TaskDuration*1000 - 1]  # ms
sence = 'ssmvep_hybrid'
# ------------------- Trial-level 离群值参数 -------------------
use_trial_outlier = True
trial_clean_suffix = '_trialClean'
trial_params = TrialOutlierParams(
    method='robust_z',
    robust_z_thr=3.5,
    min_keep=5,
    verbose=True
)

def pfx(s):
    return f"{datatype}S0{s}" if s <= 9 else f"{datatype}S{s}"

def load_block(p):
    M = hdf5storage.loadmat(p)
    return M['data'], M['label'].ravel(), float(np.squeeze(M['fs']))




if sence == 'graz':
    data_4_filepath = r'E:\Datasets\1_Graz范式\处理后数据'
    save_root_ERSP = r'E:\Datasets\1_Graz范式\Graz画图数据\ERSP数据'
    Path(save_root_ERSP).mkdir(parents=True, exist_ok=True)
    subjectchoose = list(range(1, 14 + 1))
    for s in [sc for sc in subjectchoose]:
        paths = {'1': os.path.join(data_4_filepath, f"{pfx(s)}S1.mat"),
                 '2': os.path.join(data_4_filepath, f"{pfx(s)}S2.mat"), }
        data_1, label_1, fs = load_block(paths['1'])
        data_2, label_2, _ = load_block(paths['2'])

        Data1 = ERPs_Filter(data_1, freqs=freqwindow, fs=fs, filterflag='filtfilt')
        Data2 = ERPs_Filter(data_2, freqs=freqwindow, fs=fs, filterflag='filtfilt')

        X1 = np.transpose(Data1, (1, 0, 2))  # (time, chan, trials)
        X2 = np.transpose(Data2, (1, 0, 2))
        for class_choose in [1, 2, 3, 4]:
            X1_temp = X1[:, :, label_1 == class_choose]
            X2_temp = X2[:, :, label_2 == class_choose]
            TFR1, t_idx, f_hz = m_tfr(X1_temp, fs=fs, N=256, win_name='gauss', win_len=251,
                                      out_freq_bins=81, out_time_len=frames)
            TFR2, _, _ = m_tfr(X2_temp, fs=fs, N=256, win_name='gauss', win_len=251,
                               out_freq_bins=81, out_time_len=frames)

            power1 = 10 * np.log10(np.maximum(np.abs(TFR1) ** 2, np.finfo(float).eps))  # 功率(dB)，并做基线（-2000~0ms）扣除
            power2 = 10 * np.log10(np.maximum(np.abs(TFR2) ** 2, np.finfo(float).eps))

            times = (np.arange(frames) / fs * 1000.0 + tlimits[
                0])  # 长度 1500，[-2000 .. ~3996] 构造 times(ms) 与 freqs(Hz)，对齐 tlimits
            freqs = f_hz  # 0..约 51.9 Hz（53 个点）

            # 基线期掩码（-2000..0ms）
            bmask = (times >= -1000 * RestDuration) & (times <= 0)
            base1 = power1[:, bmask, :, :].mean(axis=1, keepdims=True)
            base2 = power2[:, bmask, :, :].mean(axis=1, keepdims=True)
            ERSP_1 = power1 - base1
            ERSP_2 = power2 - base2
            ERSP_1 = ERSP_1.astype(np.float32)
            ERSP_2 = ERSP_2.astype(np.float32)

            hdf5storage.savemat(os.path.join(save_root_ERSP, f"sub{s}_sence1_class{class_choose}.mat"),
                                {'ERSP_1': [ERSP_1]})
            hdf5storage.savemat(os.path.join(save_root_ERSP, f"sub{s}_sence2_class{class_choose}.mat"),
                                {'ERSP_2': [ERSP_2]})
            hdf5storage.savemat(os.path.join(save_root_ERSP, f"times+freqs_class{class_choose}.mat"),
                                {'times': times, 'freqs': freqs})
            # ------------------- Trial-level 清洗版 ERSP 另存 -------------------
            if use_trial_outlier:
                ERSP_1_clean, keep_trials_1, info_1 = apply_trial_outlier_removal(
                    ERSP_1,
                    params=trial_params,
                    trial_axis=3,
                    subject=s,
                    sence_name='sence1',
                    class_name=class_choose
                )

                ERSP_2_clean, keep_trials_2, info_2 = apply_trial_outlier_removal(
                    ERSP_2,
                    params=trial_params,
                    trial_axis=3,
                    subject=s,
                    sence_name='sence2',
                    class_name=class_choose
                )

                ERSP_1_clean = ERSP_1_clean.astype(np.float32)
                ERSP_2_clean = ERSP_2_clean.astype(np.float32)

                hdf5storage.savemat(
                    os.path.join(save_root_ERSP, f"sub{s}_sence1_class{class_choose}{trial_clean_suffix}.mat"),
                    {
                        'ERSP_1': [ERSP_1_clean],
                        'trial_outlier_info': info_1
                    }
                )

                hdf5storage.savemat(
                    os.path.join(save_root_ERSP, f"sub{s}_sence2_class{class_choose}{trial_clean_suffix}.mat"),
                    {
                        'ERSP_2': [ERSP_2_clean],
                        'trial_outlier_info': info_2
                    }
                )
        print('被试:', s, "ERSP 计算并保存完成。")

if sence == 'ssmvep':
    data_4_filepath = r'E:\Datasets\2_MI_SSMVEP范式\MI_SSMVEP处理后数据'
    save_root_ERSP = r'E:\Datasets\2_MI_SSMVEP范式\画图数据\ERSP数据'
    Path(save_root_ERSP).mkdir(parents=True, exist_ok=True)
    subjectchoose = list(range(1, 22 + 1))
    for s in [sc for sc in subjectchoose]:
        paths = {'1': os.path.join(data_4_filepath, f"{pfx(s)}S1.mat"),
                 '2': os.path.join(data_4_filepath, f"{pfx(s)}S2.mat"), }
        data_1, label_1, fs = load_block(paths['1'])
        data_2, label_2,_= load_block(paths['2'])

        Data1 = ERPs_Filter(data_1, freqs=freqwindow, fs=fs, filterflag='filtfilt')
        Data2 = ERPs_Filter(data_2, freqs=freqwindow, fs=fs, filterflag='filtfilt')

        X1 = np.transpose(Data1, (1, 0, 2))  # (time, chan, trials)
        X2 = np.transpose(Data2, (1, 0, 2))
        for class_choose in [1, 2, 3, 4]:
            X1_temp = X1[:, :, label_1 == class_choose]
            X2_temp = X2[:, :, label_2 == class_choose]
            TFR1, t_idx, f_hz = m_tfr(X1_temp, fs=fs, N=256, win_name='gauss', win_len=251,
                                      out_freq_bins=81, out_time_len=frames)
            TFR2, _, _ = m_tfr(X2_temp, fs=fs, N=256, win_name='gauss', win_len=251,
                               out_freq_bins=81, out_time_len=frames)

            power1 = 10 * np.log10(np.maximum(np.abs(TFR1) ** 2, np.finfo(float).eps))  # 功率(dB)，并做基线（-2000~0ms）扣除
            power2 = 10 * np.log10(np.maximum(np.abs(TFR2) ** 2, np.finfo(float).eps))

            times = (np.arange(frames) / fs * 1000.0 + tlimits[
                0])  # 长度 1500，[-2000 .. ~3996] 构造 times(ms) 与 freqs(Hz)，对齐 tlimits
            freqs = f_hz  # 0..约 51.9 Hz（53 个点）

            # 基线期掩码（-2000..0ms）
            bmask = (times >= -1000 * RestDuration) & (times <= 0)
            base1 = power1[:, bmask, :, :].mean(axis=1, keepdims=True)
            base2 = power2[:, bmask, :, :].mean(axis=1, keepdims=True)
            ERSP_1 = power1 - base1
            ERSP_2 = power2 - base2
            ERSP_1 = ERSP_1.astype(np.float32)
            ERSP_2 = ERSP_2.astype(np.float32)

            hdf5storage.savemat(os.path.join(save_root_ERSP, f"sub{s}_sence1_class{class_choose}.mat"),
                                {'ERSP_1': [ERSP_1]})
            hdf5storage.savemat(os.path.join(save_root_ERSP, f"sub{s}_sence2_class{class_choose}.mat"),
                                {'ERSP_2': [ERSP_2]})
            hdf5storage.savemat(os.path.join(save_root_ERSP, f"times+freqs_class{class_choose}.mat"),
                                {'times': times, 'freqs': freqs})
            # ------------------- Trial-level 清洗版 ERSP 另存 -------------------
            if use_trial_outlier:
                ERSP_1_clean, keep_trials_1, info_1 = apply_trial_outlier_removal(
                    ERSP_1,
                    params=trial_params,
                    trial_axis=3,
                    subject=s,
                    sence_name='sence1',
                    class_name=class_choose
                )

                ERSP_2_clean, keep_trials_2, info_2 = apply_trial_outlier_removal(
                    ERSP_2,
                    params=trial_params,
                    trial_axis=3,
                    subject=s,
                    sence_name='sence2',
                    class_name=class_choose
                )

                ERSP_1_clean = ERSP_1_clean.astype(np.float32)
                ERSP_2_clean = ERSP_2_clean.astype(np.float32)

                hdf5storage.savemat(
                    os.path.join(save_root_ERSP, f"sub{s}_sence1_class{class_choose}{trial_clean_suffix}.mat"),
                    {
                        'ERSP_1': [ERSP_1_clean],
                        'trial_outlier_info': info_1
                    }
                )

                hdf5storage.savemat(
                    os.path.join(save_root_ERSP, f"sub{s}_sence2_class{class_choose}{trial_clean_suffix}.mat"),
                    {
                        'ERSP_2': [ERSP_2_clean],
                        'trial_outlier_info': info_2
                    }
                )
        print('被试:', s, "ERSP 计算并保存完成。")


if sence == 'ssmvep_hybrid':
    data_4_filepath = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2处理后数据'
    save_root_ERSP = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\ERSP数据'
    Path(save_root_ERSP).mkdir(parents=True, exist_ok=True)
    stim_name = ('ssvideo', 'video', 'ssmvep', 'cue')
    subjectchoose = list(range(1, 37 + 1))
    for s in [sc for sc in subjectchoose]:
        for stim in stim_name:
            paths = {'1': os.path.join(data_4_filepath, f"{pfx(s)}S1_{stim}.mat"),
                    '2': os.path.join(data_4_filepath, f"{pfx(s)}S2_{stim}.mat"),}
            data_1, label_1, fs = load_block(paths['1'])
            data_2, label_2, _  = load_block(paths['2'])

            Data1 = ERPs_Filter(data_1, freqs=freqwindow, fs=fs, filterflag='filtfilt')
            Data2 = ERPs_Filter(data_2, freqs=freqwindow, fs=fs, filterflag='filtfilt')

            X1 = np.transpose(Data1, (1, 0, 2))  # (time, chan, trials)
            X2 = np.transpose(Data2, (1, 0, 2))
            for class_choose in [1, 2]:
                X1_temp = X1[:, :, label_1==class_choose]
                X2_temp = X2[:, :, label_2 == class_choose]
                TFR1, t_idx, f_hz = m_tfr(X1_temp, fs=fs, N=256, win_name='gauss', win_len=251,
                                          out_freq_bins=81, out_time_len=frames)
                TFR2, _, _        = m_tfr(X2_temp, fs=fs, N=256, win_name='gauss', win_len=251,
                                          out_freq_bins=81, out_time_len=frames)

                power1 = 10*np.log10(np.maximum(np.abs(TFR1)**2, np.finfo(float).eps))# 功率(dB)，并做基线（-2000~0ms）扣除
                power2 = 10*np.log10(np.maximum(np.abs(TFR2)**2, np.finfo(float).eps))
                times = (np.arange(frames)/fs*1000.0 + tlimits[0])  # 长度 1500，[-2000 .. ~3996] 构造 times(ms) 与 freqs(Hz)，对齐 tlimits
                freqs = f_hz  # 0..约 51.9 Hz（53 个点）
                # 基线期掩码（-2000..0ms）
                bmask = (times >= -1000*RestDuration) & (times <= 0)
                base1 = power1[:, bmask, :, :].mean(axis=1, keepdims=True)
                base2 = power2[:, bmask, :, :].mean(axis=1, keepdims=True)
                ERSP_1 = power1 - base1
                ERSP_2 = power2 - base2
                ERSP_1 = ERSP_1.astype(np.float32)
                ERSP_2 = ERSP_2.astype(np.float32)

                # hdf5storage.savemat(os.path.join(save_root_ERSP, f"sub{s}_sence1_{stim}_class{class_choose}.mat"),
                #             {'ERSP_1': [ERSP_1]})
                # hdf5storage.savemat(os.path.join(save_root_ERSP, f"sub{s}_sence2_{stim}_class{class_choose}.mat"),
                #             {'ERSP_2': [ERSP_2]})
                hdf5storage.savemat(os.path.join(save_root_ERSP, f"times+freqs_{stim}_class{class_choose}.mat"),
                            {'times': times, 'freqs': freqs})
                # ------------------- Trial-level 清洗版 ERSP 另存 -------------------
                if use_trial_outlier:
                    ERSP_1_clean, keep_trials_1, info_1 = apply_trial_outlier_removal(
                        ERSP_1,
                        params=trial_params,
                        trial_axis=2,
                        subject=s,
                        sence_name='sence1',
                        class_name=class_choose,
                        stim_name=stim)
                    ERSP_2_clean, keep_trials_2, info_2 = apply_trial_outlier_removal(
                        ERSP_2,
                        params=trial_params,
                        trial_axis=2,
                        subject=s,
                        sence_name='sence2',
                        class_name=class_choose,
                        stim_name=stim)
                    ERSP_1_clean = np.mean(ERSP_1_clean, axis=2).astype(np.float32) # 试次维度平均
                    ERSP_2_clean = np.mean(ERSP_2_clean, axis=2).astype(np.float32)  # 试次维度平均

                    savemat_my(
                        os.path.join(save_root_ERSP,
                                     f"sub{s}_sence1_{stim}_class{class_choose}{trial_clean_suffix}.mat"),
                        'ERSP_1', ERSP_1_clean, info_1)
                    savemat_my(
                        os.path.join(save_root_ERSP,
                                     f"sub{s}_sence2_{stim}_class{class_choose}{trial_clean_suffix}.mat"),
                        'ERSP_2', ERSP_2_clean, info_2)
        print('被试:', s, "ERSP 计算并保存完成。")

if sence == 'kjz':
    data_4_filepath = r'E:\Datasets\6_天基测试数据\熊猫采集数据\20250920-21乘组-前测数据'
    save_root_ERSP = r'E:\Datasets\6_天基测试数据\ERSP数据\熊猫采集数据\20250920-21乘组-前测数据'
    Path(save_root_ERSP).mkdir(parents=True, exist_ok=True)
    # subject_file = {1:f"SZ21-1\start_test_30_offline_20250826_091328.mat",
    #                 2:f"SZ21-2\start_test_30_offline_20250826_092817.mat",
    #                 3:f"SZ21-3\start_test_30_offline_20250826_113611.mat",}
    subject_file = {1:f"SZ21-1\start_test_30_offline_20250920_143059.mat",
                    2:f"SZ21-2\start_test_30_offline_20250920_143153.mat",
                    3:f"SZ21-3\start_test_30_offline_20250920_150346.mat",}
    for s in [1,2,3]:
        paths = os.path.join(data_4_filepath, subject_file[s])
        data = hdf5storage.loadmat(paths)['data']
        label = data[:,-2]
        data1 = data[:,:32]
        Data = []
        Label = []
        for i, l in enumerate(label):
            if l==1 or l==2 or l==3:
                Data.append(data1[i-2000:i+4000, :])
                Label.append(l)
        Data = np.array(Data)
        Label = np.array(Label)
        n_points = int(Data.shape[1] * srate / 1000)
        Data1 = np.transpose(resample(Data, n_points, axis=1), (2, 1, 0))  # (channels, points, samples)
        Data2 = ERPs_Filter(Data1, freqs=freqwindow, fs=srate, filterflag='filtfilt')

        X = np.transpose(Data2, (1, 0, 2))  # (time, chan, trials)
        for class_choose in [1, 2, 3]:
            X_temp = X[:, :, Label == class_choose]
            TFR1, t_idx, f_hz = m_tfr(X_temp, fs=srate, N=256, win_name='gauss', win_len=251,
                                      out_freq_bins=81, out_time_len=frames)

            power1 = 10 * np.log10(np.maximum(np.abs(TFR1) ** 2, np.finfo(float).eps))  # 功率(dB)，并做基线（-2000~0ms）扣除
            times = (np.arange(frames) / srate * 1000.0 + tlimits[0])  # 长度 1500，[-2000 .. ~3996] 构造 times(ms) 与 freqs(Hz)，对齐 tlimits
            freqs = f_hz  # 0..约 51.9 Hz（53 个点）
            bmask = (times >= -1000 * RestDuration) & (times <= 0)
            base1 = power1[:, bmask, :, :].mean(axis=1, keepdims=True)
            ERSP_1 = power1 - base1
            ERSP_1 = ERSP_1.astype(np.float32)
            hdf5storage.savemat(os.path.join(save_root_ERSP, f"sub{s}_class{class_choose}.mat"),
                                {'ERSP': [ERSP_1]})
            hdf5storage.savemat(os.path.join(save_root_ERSP, f"times+freqs_class{class_choose}.mat"),
                                {'times': times, 'freqs': freqs})
            # ------------------- Trial-level 清洗版 ERSP 另存 -------------------
            if use_trial_outlier:
                ERSP_1_clean, keep_trials_1, info_1 = apply_trial_outlier_removal(
                    ERSP_1,
                    params=trial_params,
                    trial_axis=3,
                    subject=s,
                    sence_name='kjz',
                    class_name=class_choose
                )

                ERSP_1_clean = ERSP_1_clean.astype(np.float32)

                hdf5storage.savemat(
                    os.path.join(save_root_ERSP, f"sub{s}_class{class_choose}{trial_clean_suffix}.mat"),
                    {
                        'ERSP': [ERSP_1_clean],
                        'trial_outlier_info': info_1
                    }
                )
        print('被试:', s, "ERSP 计算并保存完成。")
