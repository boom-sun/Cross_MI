from pathlib import Path
import os

import mne
import numpy as np
from scipy.io import loadmat
import hdf5storage

DATA_PATH = "C:\\Users\\Administrator\\PycharmProjects\\python310\\MrLiuTuo\\MNE_weibo_2014"  # 数据存储路径

def load_custom_data_per_subject(subject_id: int, preprocessing_dict: dict, train_rate:float):
    """
    参数:
        subject_id (int): 被试编号，例如 1, 2 等。
        preprocessing_dict (dict): 预处理参数，包括采样率、滤波范围等。

    返回:
        trials_dict (dict): 包含训练和测试试次的字典。
        labels_dict (dict): 包含训练和测试标签的字典。
    """ 
    file_path = os.path.join(DATA_PATH, f"subject_{subject_id}.mat")

    mat = hdf5storage.loadmat(file_path)
    data = mat["data"]  # 数据格式: 导联 × 采样点 × 试次
    labels = mat["label"].flatten()  # 标签格式: (试次数,)
    sfreq=200
    data=data[:60, int(sfreq*3):int(sfreq*7),np.logical_or(labels==4,labels==7)]
    labels=labels[np.logical_or(labels==4,labels==7)]
    for i in range(len(labels)):
        if labels[i]==4:
            labels[i]=0
        if labels[i]==7:
            labels[i]=1

    # 创建 MNE 的 Info 对象
    ch_names = [f"EEG{i+1}" for i in range(data.shape[0])]  # 假设导联名称为 EEG1, EEG2, ...
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    # 将数据转换为 MNE 的 RawArray 格式
    raw_data = np.concatenate([data[:, :, i] for i in range(data.shape[2])], axis=1)  # 将所有试次拼接
    raw = mne.io.RawArray(raw_data, info)

    # 数据预处理
    raw.resample(preprocessing_dict["sfreq"])
    raw.filter(l_freq=preprocessing_dict["low_cut"], h_freq=preprocessing_dict["high_cut"])

    # 选择特定通道（如果需要）
    channel_selection = preprocessing_dict.get("channel_selection", False)
    if channel_selection:
        channels = preprocessing_dict["pick_ch"] 
        raw.pick_channels(channels)

    # 提取试次数据
    start = int(preprocessing_dict["sfreq"] * preprocessing_dict["start"])
    stop = int(preprocessing_dict["sfreq"] * preprocessing_dict["stop"])
    trial_length = int(preprocessing_dict["sfreq"] * 4) - start + stop 
    trials = np.zeros((data.shape[2], raw._data.shape[0], trial_length))  # 试次 × 导联 × 采样点

    for i in range(data.shape[2]):
        trials[i] = raw._data[:, i * trial_length:(i + 1) * trial_length]

    indices = np.random.permutation(len(labels))  # 随机打乱所有索引
    shuffled_trials = np.array([trials[i] for i in indices])
    shuffled_labels = np.array([labels[i] for i in indices])

    # 划分训练集和测试集
    test_idx = int(len(shuffled_labels) * train_rate)  # 80%训练集，20%测试集
    trials_dict = {
        "train": shuffled_trials[:test_idx],
        "test": shuffled_trials[test_idx:]
    }
    labels_dict = {
        "train": shuffled_labels[:test_idx],
        "test": shuffled_labels[test_idx:]
    }

    return trials_dict, labels_dict


def load_Yi_data(subject_ids=["01"], prepr_dict=None, train_rate=0.8):
    """
    加载多个被试的数据。

    参数:
        subject_ids (list): 被试编号列表，例如 ["01", "02"]。
        prepr_dict (dict): 预处理参数。

    返回:
        dict: 包含所有被试的数据和标签。
    """
    lookup_dict = {
        "1": "1","2": "2","3": "3","4": "4","5": "5",
        "6": "6","7": "7","8": "8","9": "9","10": "10",
    }
    data, labels = {}, {}
    for subject_id in subject_ids:
        data[str(subject_id)], labels[str(subject_id)] = load_custom_data_per_subject(
            subject_id, prepr_dict, train_rate,
        )
    return {"data": data, "labels": labels}

