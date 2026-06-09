from pathlib import Path
import os

import mne
import numpy as np
from scipy.io import loadmat
import hdf5storage
import scipy.io as sio

DATA_PATH = "C:\\Users\\Administrator\\PycharmProjects\\python310\\MrLiuTuo\\MNE_D2"  # 数据存储路径

def load_custom_data_per_subject(subject_id: int, preprocessing_dict: dict, train_rate:float):
    """
    参数:
        subject_id (int): 被试编号，例如 1, 2 等。
        preprocessing_dict (dict): 预处理参数，包括采样率、滤波范围等。

    返回:
        trials_dict (dict): 包含训练和测试试次的字典。
        labels_dict (dict): 包含训练和测试标签的字典。
    """
    Data_train=[]
    Labels_train=[]
    Data_test=[]
    Labels_test=[]
    for block in [2]:
        file_path1 = os.path.join(DATA_PATH, f"D2S{subject_id}B{block}.mat")
        raw_y_index = np.where(sio.loadmat(file_path1)['data'][-1, :] > 0)  # loadmat()函数加载.mat数据文件
        labels = sio.loadmat(file_path1)['data'][-1, :][raw_y_index] - 1
        data = np.array(np.split(sio.loadmat(file_path1)['data'][0:-1, :], len(labels), axis=1))
        i_begin = 0
        i_end = 20
        ii_begin = 20
        ii_end = 50
        y_train0 = np.where(labels[i_begin:i_end] == 0)[0]
        y_train1 = np.where(labels[i_begin:i_end] == 1)[0]
        y_test0 = np.where(labels[ii_begin:ii_end] == 0)[0]
        y_test1 = np.where(labels[ii_begin:ii_end] == 1)[0]
        index_train = np.concatenate((y_train0[0:10], y_train1[0:10]))
        index_test = np.concatenate((y_test0[0:15], y_test1[0:15]))
        Data_train.append(data[i_begin:i_end][index_train])
        Labels_train.append(labels[i_begin:i_end][index_train])
        Data_test.append(data[ii_begin:ii_end][index_test])
        Labels_test.append(labels[ii_begin:ii_end][index_test])
    data_train=np.concatenate(Data_train, axis=0)
    data_test=np.concatenate(Data_test, axis=0)
    labels_train=np.concatenate(Labels_train, axis=0)
    labels_test=np.concatenate(Labels_test, axis=0)

    # 划分训练集和测试集
    trials_dict = {
        "train": data_train,
        "test": data_test
    }
    labels_dict = {
        "train": labels_train,
        "test": labels_test
    }

    return trials_dict, labels_dict


def load_Liu_data(subject_ids=["01"], prepr_dict=None, train_rate=0.8):
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

