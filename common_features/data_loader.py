"""
公用数据加载模块 - 支持Graz范式数据集
数据格式: HDF5 v7.3 mat文件
data: (trials, timepoints, channels)
label: (trials, 1)  标签值 1/2/3/4 (左/右/双脚/舌)
fs: 250 Hz, 6秒试验, 60通道
"""

import h5py
import numpy as np
import os

# 数据根目录 (Windows路径)
DATA_ROOT = r'E:\Datasets\1_Graz范式\处理后数据'
N_SUBJECTS = 14
N_SESSIONS = 2
FS = 250

# Graz 60通道布局 (Neuroscan系统, 基于标准10-20扩展)
# 运动皮层关键通道索引 (0-based), C3/C4及周围
MOTOR_CH_NAMES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
    'CP5','CP3','CP1','CPz','CP2','CP4','CP6',
    'P7','P3','Pz','P4','P8',
    'PO7','PO3','POz','PO4','PO8',
    'O1','Oz','O2',
    'FT9','FT10','TP9','TP10',
    'F1','F2','FC1b','FC2b',
    'C1b','C2b','CP1b','CP2b',
    'P1','P2','PO1','PO2',
    'AF3','AF4',
]
# 若不足60个, 用占位符
while len(MOTOR_CH_NAMES) < 60:
    MOTOR_CH_NAMES.append(f'CH{len(MOTOR_CH_NAMES)+1}')
MOTOR_CH_NAMES = MOTOR_CH_NAMES[:60]

# 运动皮层通道 (C3/Cz/C4区域) 的索引
MOTOR_CORE_IDX = [15, 16, 17, 18, 19, 20, 21,   # C5-C4
                  7,  8,  9, 10, 11, 12, 13,    # FC5-FC6
                  23, 24, 25, 26, 27, 28, 29]   # CP5-CP6


def load_subject_session(subject, session, data_root=DATA_ROOT):
    """
    加载单个被试单个会话数据
    返回: data (trials, channels, timepoints), labels (trials,)
    """
    fname = os.path.join(data_root, f'1S{subject:02d}S{session}.mat')
    with h5py.File(fname, 'r') as f:
        data = np.array(f['data'])    # (trials, timepoints, channels)
        label = np.array(f['label']).flatten().astype(int)  # (trials,)
    # 转置为 (trials, channels, timepoints)
    data = data.transpose(0, 2, 1)
    return data, label


def load_all_subjects(subjects=None, sessions=(1, 2), data_root=DATA_ROOT,
                      classes=(1, 2)):
    """
    加载所有指定被试和会话的数据
    返回: dict {(sub, sess): (data, label)}
    classes: 只保留指定类别 (默认左手=1, 右手=2)
    """
    if subjects is None:
        subjects = list(range(1, N_SUBJECTS + 1))
    dataset = {}
    for sub in subjects:
        for sess in sessions:
            try:
                data, label = load_subject_session(sub, sess, data_root)
                if classes is not None:
                    mask = np.isin(label, classes)
                    data, label = data[mask], label[mask]
                    # 重映射标签为 0/1
                    label_new = np.zeros_like(label)
                    for i, c in enumerate(classes):
                        label_new[label == c] = i
                    label = label_new
                dataset[(sub, sess)] = (data, label)
            except Exception as e:
                print(f'  跳过 S{sub:02d}S{sess}: {e}')
    return dataset


if __name__ == '__main__':
    d = load_subject_session(1, 1)
    print('data shape:', d[0].shape)
    print('label:', np.unique(d[1], return_counts=True))
    print('fs:', FS)
