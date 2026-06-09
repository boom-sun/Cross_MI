import os
import hdf5storage
import gc
import numpy as np
import scipy.io as sio
import mne
from pathlib import Path
from Functions import TrialOutlierParams, apply_trial_outlier_removal, savemat_my
# ------------------- 基本参数 -------------------
sence = 'ssmvep_hybrid'

topo_band = [(8,13),(13, 30),(13,20),(20,30)]
topo_timewin = [(0, 4),(-2,0),(0,0.5),(0.5,1),(1,1.5),(1.5,2),(2,2.5),(2.5,3),(3,3.5),(3.5,4)]
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
# ------------------- Trial-level 离群值参数 -------------------
use_trial_outlier = True
trial_clean_suffix = '_trialClean'

trial_params = TrialOutlierParams(
    method='robust_z',
    robust_z_thr=3.5,
    min_keep=5,
    verbose=True
)
# ------------------- 数据载入 -------------------
if sence == 'ssmvep_hybrid':
    save_dir = r'E:\\Datasets\\4_跨场景因素研究v2\\跨场景因素研究v2画图数据\\ERSP数据'
    save_root_TOPO = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\脑地形图数据'
    Path(save_root_TOPO).mkdir(parents=True, exist_ok=True)
    save_root_TF = r'E:\Datasets\4_跨场景因素研究v2\跨场景因素研究v2画图数据\时频图数据'
    Path(save_root_TF).mkdir(parents=True, exist_ok=True)
    subjects = list(range(1, 37 + 1))
    CLASS_CHOOSE_LIST = [1, 2]
    stim_name = ('ssvideo', 'video', 'ssmvep', 'cue')
    Class_name = ('Left', 'Right')
    for class_choose in CLASS_CHOOSE_LIST:
        for stim in stim_name:
            topo, TOPO = {'s1': {}, 's2': {}},{'s1': {}, 's2': {}}
            tf,TF = {'s1': [], 's2': []},{'s1': None, 's2': None}
            tf_path = os.path.join(save_dir, f"times+freqs_{stim}_class{class_choose}.mat")
            times, freqs = hdf5storage.loadmat(tf_path)['times'], hdf5storage.loadmat(tf_path)['freqs']
            Flen, Tlen = len(times), len(freqs)
            for band in topo_band:  # 建立空数组
                for time in topo_timewin:
                    topo['s1']['freq' + str(band) + 'time' + str(time)] = np.empty((len(CHANNELS),0))
                    topo['s2']['freq' + str(band) + 'time' + str(time)]=topo['s1']['freq' + str(band) + 'time' + str(time)]

            for s in subjects:
                p1 = os.path.join(save_dir, f"sub{s}_sence1_{stim}_class{class_choose}_trialClean.mat")
                p2 = os.path.join(save_dir, f"sub{s}_sence2_{stim}_class{class_choose}_trialClean.mat")
                ersp_s1 = hdf5storage.loadmat(p1)['ERSP_1'][0][0]
                ersp_s2 = hdf5storage.loadmat(p2)['ERSP_2'][0][0]
                for band in topo_band:
                    for time in topo_timewin:
                        topo_temp= ersp_s1[(freqs>=band[0]).__and__(freqs<band[1]),:, :]
                        topo_temp1 = topo_temp[:,(times>=time[0]*1000).__and__(times<time[1]*1000),:]
                        topo_mean = np.expand_dims(
                            np.mean(np.mean(topo_temp1, axis=0),axis=0), axis=1)  # 频率、时间维度求平均
                        topo['s1']['freq' + str(band) + 'time' + str(time)] = np.concatenate(
                            (topo['s1']['freq' + str(band) + 'time' + str(time)], topo_mean),axis=1)
                        topo_temp= ersp_s2[(freqs>=band[0]).__and__(freqs<band[1]),:,  :]
                        topo_temp1 = topo_temp[:,(times>=time[0]*1000).__and__(times<time[1]*1000),:]
                        topo_mean = np.expand_dims(
                            np.mean(np.mean(topo_temp1, axis=0),axis=0), axis=1)  # 频率、时间维度求平均
                        topo['s2']['freq' + str(band) + 'time' + str(time)] = np.concatenate(
                            (topo['s2']['freq' + str(band) + 'time' + str(time)], topo_mean),axis=1)
                tf['s1'].append(ersp_s1)  # 时频图不需要再滤波和截取，保留原来的时间频率,所有导联保留
                tf['s2'].append(ersp_s2)
            TF['s1'] = np.array(tf['s1'])
            TF['s2'] = np.array(tf['s2'])
            p = os.path.join(save_root_TF, f"TF_{stim}_class{class_choose}_trialClean.mat")   # [subjects,freq, time, channel]
            sio.savemat(p, {'tf': TF})
            # hdf5storage.savemat(p, {'tf': TF})
            p = os.path.join(save_root_TOPO, f"TOPO_{stim}_class{class_choose}_trialClean.mat")  # [channel, subjects]
            sio.savemat(p, {'topo': topo})
            # hdf5storage.savemat(p, {'topo': topo})
if sence == 'graz':
    save_dir = r'E:\Datasets\1_Graz范式\Graz画图数据\ERSP数据'
    save_root_TOPO = r'E:\Datasets\1_Graz范式\Graz画图数据\脑地形图数据'
    Path(save_root_TOPO).mkdir(parents=True, exist_ok=True)
    save_root_TF = r'E:\Datasets\1_Graz范式\Graz画图数据\时频图数据'
    Path(save_root_TF).mkdir(parents=True, exist_ok=True)
    subjects = list(range(1, 14 + 1))
    CLASS_CHOOSE_LIST = [1, 2, 3, 4]
    Class_name = ('Left', 'Right', 'Feet', 'Rest')
    for class_choose in CLASS_CHOOSE_LIST:
        topo, TOPO = {'s1': {}, 's2': {}},{'s1': {}, 's2': {}}
        tf,TF = {'s1': [], 's2': []},{'s1': None, 's2': None}
        tf_path = os.path.join(save_dir, f"times+freqs_class{class_choose}.mat")
        times, freqs = hdf5storage.loadmat(tf_path)['times'], hdf5storage.loadmat(tf_path)['freqs']
        Flen, Tlen = len(times), len(freqs)
        for band in topo_band:  # 建立空数组
            for time in topo_timewin:
                topo['s1']['freq' + str(band) + 'time' + str(time)] = np.empty((len(CHANNELS),0))
                topo['s2']['freq' + str(band) + 'time' + str(time)]=topo['s1']['freq' + str(band) + 'time' + str(time)]
        for s in subjects:
            p1 = os.path.join(save_dir, f"sub{s}_sence1_class{class_choose}.mat")
            p2 = os.path.join(save_dir, f"sub{s}_sence2_class{class_choose}.mat")
            ersp_s1 = hdf5storage.loadmat(p1)['ERSP_1'][0]
            ersp_s2 = hdf5storage.loadmat(p2)['ERSP_2'][0]
            for band in topo_band:
                for time in topo_timewin:
                    topo_temp= ersp_s1[(freqs>=band[0]).__and__(freqs<band[1]),:, :, :]
                    topo_temp1 = topo_temp[:,(times>=time[0]*1000).__and__(times<time[1]*1000),:,:]
                    topo_mean = np.expand_dims(
                        np.mean(np.mean(np.mean(topo_temp1, axis=0),axis=0),axis=0), axis=1)  # 频率、时间、试次维度求平均
                    topo['s1']['freq' + str(band) + 'time' + str(time)] = np.concatenate(
                        (topo['s1']['freq' + str(band) + 'time' + str(time)], topo_mean),axis=1)
                    topo_temp= ersp_s2[(freqs>=band[0]).__and__(freqs<band[1]),:, :, :]
                    topo_temp1 = topo_temp[:,(times>=time[0]*1000).__and__(times<time[1]*1000),:,:]
                    topo_mean = np.expand_dims(
                        np.mean(np.mean(np.mean(topo_temp1, axis=0),axis=0),axis=0), axis=1)  # 频率、时间、试次维度求平均
                    topo['s2']['freq' + str(band) + 'time' + str(time)] = np.concatenate(
                        (topo['s2']['freq' + str(band) + 'time' + str(time)], topo_mean),axis=1)
            tf['s1'].append(np.mean(ersp_s1,axis=2))  # 时频图不需要再滤波和截取，保留原来的时间频率,所有导联保留，试次维度平均
            tf['s2'].append(np.mean(ersp_s2,axis=2))
        TF['s1'] = np.array(tf['s1'])
        TF['s2'] = np.array(tf['s2'])
        p = os.path.join(save_root_TF, f"TF_class{class_choose}.mat")   # [subjects,freq, time, channel]
        hdf5storage.savemat(p, {'tf': TF})
        p = os.path.join(save_root_TOPO, f"TOPO_class{class_choose}.mat")  # [channel, subjects]
        hdf5storage.savemat(p, {'topo': topo})
        print(f'TOPO_class{class_choose}.mat')

if sence == 'ssmvep':
    save_dir = r'E:\Datasets\2_MI_SSMVEP范式\画图数据\ERSP数据'
    save_root_TOPO = r'E:\Datasets\2_MI_SSMVEP范式\画图数据\脑地形图数据'
    Path(save_root_TOPO).mkdir(parents=True, exist_ok=True)
    save_root_TF = r'E:\Datasets\2_MI_SSMVEP范式\画图数据\时频图数据'
    Path(save_root_TF).mkdir(parents=True, exist_ok=True)
    subjects = list(range(1, 22 + 1))
    CLASS_CHOOSE_LIST = [1, 2, 3, 4]
    Class_name = ('Left_MI', 'Right_MI', 'Left_AO', 'Right_AO')
    for class_choose in CLASS_CHOOSE_LIST:
        topo, TOPO = {'s1': {}, 's2': {}},{'s1': {}, 's2': {}}
        tf,TF = {'s1': [], 's2': []},{'s1': None, 's2': None}
        tf_path = os.path.join(save_dir, f"times+freqs_class{class_choose}.mat")
        times, freqs = hdf5storage.loadmat(tf_path)['times'], hdf5storage.loadmat(tf_path)['freqs']
        Flen, Tlen = len(times), len(freqs)
        for band in topo_band:  # 建立空数组
            for time in topo_timewin:
                topo['s1']['freq' + str(band) + 'time' + str(time)] = np.empty((len(CHANNELS),0))
                topo['s2']['freq' + str(band) + 'time' + str(time)]=topo['s1']['freq' + str(band) + 'time' + str(time)]
        for s in subjects:
            p1 = os.path.join(save_dir, f"sub{s}_sence1_class{class_choose}.mat")
            p2 = os.path.join(save_dir, f"sub{s}_sence2_class{class_choose}.mat")
            ersp_s1 = hdf5storage.loadmat(p1)['ERSP_1'][0]
            ersp_s2 = hdf5storage.loadmat(p2)['ERSP_2'][0]
            for band in topo_band:
                for time in topo_timewin:
                    topo_temp= ersp_s1[(freqs>=band[0]).__and__(freqs<band[1]),:, :, :]
                    topo_temp1 = topo_temp[:,(times>=time[0]*1000).__and__(times<time[1]*1000),:,:]
                    topo_mean = np.expand_dims(
                        np.mean(np.mean(np.mean(topo_temp1, axis=0),axis=0),axis=0), axis=1)  # 频率、时间、试次维度求平均
                    topo['s1']['freq' + str(band) + 'time' + str(time)] = np.concatenate(
                        (topo['s1']['freq' + str(band) + 'time' + str(time)], topo_mean),axis=1)
                    topo_temp= ersp_s2[(freqs>=band[0]).__and__(freqs<band[1]),:, :, :]
                    topo_temp1 = topo_temp[:,(times>=time[0]*1000).__and__(times<time[1]*1000),:,:]
                    topo_mean = np.expand_dims(
                        np.mean(np.mean(np.mean(topo_temp1, axis=0),axis=0),axis=0), axis=1)  # 频率、时间、试次维度求平均
                    topo['s2']['freq' + str(band) + 'time' + str(time)] = np.concatenate(
                        (topo['s2']['freq' + str(band) + 'time' + str(time)], topo_mean),axis=1)
            tf['s1'].append(np.mean(ersp_s1,axis=2))  # 时频图不需要再滤波和截取，保留原来的时间频率,所有导联保留，试次维度平均
            tf['s2'].append(np.mean(ersp_s2,axis=2))
        TF['s1'] = np.array(tf['s1'])
        TF['s2'] = np.array(tf['s2'])
        p = os.path.join(save_root_TF, f"TF_class{class_choose}.mat")   # [subjects,freq, time, channel]
        hdf5storage.savemat(p, {'tf': TF})
        p = os.path.join(save_root_TOPO, f"TOPO_class{class_choose}.mat")  # [channel, subjects]
        hdf5storage.savemat(p, {'topo': topo})
        print(f'TOPO_class{class_choose}.mat')

if sence == 'kjz':
    save_root_ERSP = r'E:\Datasets\6_天基测试数据\ERSP数据\熊猫采集数据\20250920-21乘组-前测数据'
    save_root_TOPO = r'E:\Datasets\6_天基测试数据\脑地形图数据\熊猫采集数据\20250920-21乘组-前测数据'
    Path(save_root_TOPO).mkdir(parents=True, exist_ok=True)
    save_root_TF = r'E:\Datasets\6_天基测试数据\时频图数据\熊猫采集数据\20250920-21乘组-前测数据'
    Path(save_root_TF).mkdir(parents=True, exist_ok=True)
    subjects = list(range(1, 3 + 1))
    CLASS_CHOOSE_LIST = [1, 2, 3]
    Class_name = ('Left', 'Right', 'Rest')
    for class_choose in CLASS_CHOOSE_LIST:
        tf_path = os.path.join(save_root_ERSP, f"times+freqs_class{class_choose}.mat")
        times, freqs = hdf5storage.loadmat(tf_path)['times'], hdf5storage.loadmat(tf_path)['freqs']
        Flen, Tlen = len(times), len(freqs)
        for s in subjects:
            topo={}
            p = os.path.join(save_root_ERSP, f"sub{s}_class{class_choose}.mat")
            ersp_s = hdf5storage.loadmat(p)['ERSP'][0]
            for band in topo_band:
                for time in topo_timewin:
                    topo_temp= ersp_s[(freqs>=band[0]).__and__(freqs<band[1]),:, :, :]
                    topo_temp1 = topo_temp[:,(times>=time[0]*1000).__and__(times<time[1]*1000),:,:]
                    topo['freq' + str(band) + 'time' + str(time)] = np.mean(np.mean(np.mean(topo_temp1, axis=0),axis=0),axis=0)  # 频率、时间、试次维度求平均
            p1 = os.path.join(save_root_TOPO, f"TOPO_sub{s}_class{class_choose}.mat")  # [channel, subjects]
            hdf5storage.savemat(p1, {'topo': topo})
            tf=np.mean(ersp_s,axis=2) # 时频图不需要再滤波和截取，保留原来的时间频率,所有导联保留，试次维度平均
            p2 = os.path.join(save_root_TF, f"TF_sub{s}_class{class_choose}.mat")   # [subjects,freq, time, channel]
            hdf5storage.savemat(p2, {'tf': tf})
        print(f'TOPO_class{class_choose}.mat')
