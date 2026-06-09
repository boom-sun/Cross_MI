# -*- coding: utf-8 -*-
# License: MIT License
"""
MI Feedback on NeuroScan and FES.

"""
import time
import numpy as np

import mne
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import NeuroScan, Marker
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.utils.model_selection \
    import EnhancedLeaveOneGroupOut
from metabci.brainda.algorithms.decomposition.csp import FBCSP
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from scipy import signal
import threading
from metabci.brainflow.ElectroStimulator import ElectroStimulator


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


class MaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        X = X.reshape((-1, X.shape[-1]))
        y = np.argmax(X, axis=-1)
        return y


def read_data(run_files, chs, interval, labels):
    Xs, ys = [], []
    for run_file in run_files:
        raw = read_raw_cnt(run_file, preload=True, verbose=False)
        raw = upper_ch_names(raw)
        raw.filter(6, 30, l_trans_bandwidth=2, h_trans_bandwidth=5,
                   phase='zero-double')
        events = mne.events_from_annotations(
            raw, event_id=lambda x: int(x), verbose=False)[0]
        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        epochs = mne.Epochs(raw, events,
                            event_id=labels,
                            tmin=interval[0],
                            tmax=interval[1],
                            baseline=None,
                            picks=ch_picks,
                            verbose=False)

        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]
            Xs.append(X)
            ys.append(np.ones((len(X))) * label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)

    return Xs, ys, ch_picks


def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2 * freq0 / srate
    wn2 = 2 * freq1 / srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new


def detect_ssmvep(data, srate=1000, stim_freq=2.4, threshold=0.15):
    """
    检测SSMVEP是否存在，即被试是否在注视屏幕

    参数:
    - data: 脑电数据
    - srate: 采样率
    - stim_freq: 刺激频率(5帧旋转，假设每秒12帧，则旋转频率为2.4Hz)
    - threshold: 检测阈值

    返回:
    - attention: 布尔值，True表示检测到SSMVEP，被试正在注视屏幕
    - max_snr: 最大信噪比值，用于诊断
    """
    # 标准化数据
    data = data - np.mean(data, axis=-1, keepdims=True)
    data = data / np.std(data, axis=-1, keepdims=True)

    # 应用带通滤波器，中心频率为刺激频率
    bandwidth = 0.5  # Hz
    low_freq = stim_freq - bandwidth / 2
    high_freq = stim_freq + bandwidth / 2
    filtered_data = bandpass(data, low_freq, high_freq, srate)

    # 计算功率
    power = np.mean(filtered_data ** 2, axis=-1)

    # 计算SNR - 信号与周围频带的比值
    surrounding_low = bandpass(data, low_freq - 1, low_freq - 0.5, srate)
    surrounding_high = bandpass(data, high_freq + 0.5, high_freq + 1, srate)
    surrounding_power = np.mean((surrounding_low ** 2 + surrounding_high ** 2) / 2, axis=-1)

    snr = power / (surrounding_power + 1e-10)  # 避免除零

    # 使用最大SNR作为决策指标
    max_snr = np.max(snr)
    attention = max_snr > threshold

    return attention, max_snr


# 训练模型
def train_model(X, y, srate=1000):
    y = np.reshape(y, (-1))
    # 降采样
    X = resample(X, up=256, down=srate)
    # 滤波
    # X = bandpass(X, 6, 30, 256)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # brainda.algorithms.decomposition.csp.MultiCSP
    wp = [(4, 8), (8, 12), (12, 30)]
    ws = [(2, 10), (6, 14), (10, 32)]
    filterbank = generate_filterbank(wp, ws, srate=256, order=4, rp=0.5)
    # model = make_pipeline(
    #     MultiCSP(n_components = 2),
    #     LinearDiscriminantAnalysis())
    model = make_pipeline(*[
        FBCSP(n_components=5,
              n_mutualinfo_components=4,
              filterbank=filterbank),
        SVC()
    ])
    # fit()训练模型
    model = model.fit(X, y)

    return model


# 预测标签


def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # 降采样
    X = resample(X, up=256, down=srate)
    # 滤波
    X = bandpass(X, 8, 30, 256)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # predict()预测标签
    p_labels = model.predict(X)
    return p_labels


# 计算离线正确率


def offline_validation(X, y, srate=1000):
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        model = train_model(X_train, y_train, srate=srate)
        p_labels = model_predict(X_test, srate=srate, model=model)
        kfold_accs.append(np.mean(p_labels == y_test))

    return np.mean(kfold_accs)


class FeedbackWorker(ProcessWorker):
    def __init__(self,
                 run_files,
                 pick_chs,
                 stim_interval,
                 stim_labels,
                 srate,
                 lsl_source_id,
                 timeout,
                 worker_name,
                 ssmvep_freq=2.4,  # 新增参数: SSMVEP刺激频率
                 attention_threshold=0.15):  # 新增参数: 注意力检测阈值
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        self.ssmvep_freq = ssmvep_freq    # 新增参数
        self.attention_threshold = attention_threshold  # 新增参数
        super().__init__(timeout=timeout, name=worker_name)
        self.stimulator = None  # 电刺激器
        self.stim_lock = None  # 线程锁

    def pre(self):
        X, y, ch_ind = read_data(run_files=self.run_files,
                                 chs=self.pick_chs,
                                 interval=self.stim_interval,
                                 labels=self.stim_labels)
        print("Loding data successfully")
        acc = offline_validation(X, y, srate=self.srate)  # 计算离线准确率
        print("Current Model accuracy:", acc)
        self.estimator = train_model(X, y, srate=self.srate)
        self.stimulator = ElectroStimulator('COM3')
        self.stim_lock = threading.Lock()  # 在子进程中初始化锁
        print("电刺激器初始化成功")
        self.ch_ind = ch_ind

        # 修改: 增加频道数量为2，一个表示注意力状态，一个表示MI分类结果
        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=2,  # 修改: 从1变成2
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)
        print('Waiting connection...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('Connected')

    def _stimulate(self, channels, params_list, duration=4):
        """电刺激线程函数"""
        with self.stim_lock:
            try:
                # 清除所有已选通道
                for ch in list(self.stimulator._selected_channels):
                    self.stimulator.disable_channel(ch)

                # 设置多个通道参数
                for channel, params in zip(channels, params_list):
                    self.stimulator.select_channel(channel)
                    self.stimulator.set_channel_parameters(channel, params)
                self.stimulator.lock_parameters()
                self.stimulator.run_stimulation(duration)

            except Exception as e:
                print(f"电刺激控制出错: {e}")

    def consume(self, data):
        # 电刺激参数配置
        params_ch1 = {
            ElectroStimulator._Param.current_positive: 2,
            ElectroStimulator._Param.current_negative: 2,
            ElectroStimulator._Param.pulse_positive: 250,
            ElectroStimulator._Param.pulse_negative: 250,
            ElectroStimulator._Param.frequency: 50,
            ElectroStimulator._Param.rise_time: 500,
            ElectroStimulator._Param.stable_time: 3000,
            ElectroStimulator._Param.descent_time: 500
        }
        params_ch2 = {
            ElectroStimulator._Param.current_positive: 2,
            ElectroStimulator._Param.current_negative: 2,
            ElectroStimulator._Param.pulse_positive: 250,
            ElectroStimulator._Param.pulse_negative: 250,
            ElectroStimulator._Param.frequency: 50,
            ElectroStimulator._Param.rise_time: 500,
            ElectroStimulator._Param.stable_time: 3000,
            ElectroStimulator._Param.descent_time: 500
        }
        data = np.array(data, dtype=np.float64).T
        data_all_channels = data.copy()  # 保存所有通道数据用于SSMVEP检测
        data = data[self.ch_ind]  # 只选MI相关通道进行MI分类

        # 修改: 增加SSMVEP检测
        # 第一步：SSMVEP检测，判断被试是否注视屏幕
        attention, attention_snr = detect_ssmvep(
            data_all_channels,
            srate=self.srate,
            stim_freq=self.ssmvep_freq,
            threshold=self.attention_threshold
        )

        # 输出注意力状态
        print(f"注意力状态: {'关注' if attention else '未关注'}, SNR: {attention_snr:.3f}")

        # 修改: 注意力检测逻辑
        # 只有检测到注意力才进行MI分类，否则直接返回
        if attention:
            # 进行MI分类
            p_labels = model_predict(data, srate=self.srate, model=self.estimator)
            p_labels = int(p_labels)
            p_labels = p_labels + 1

            # 根据分类结果激活电刺激
            if p_labels == 1:
                print("激活通道1")
                stim_thread = threading.Thread(
                    target=self._stimulate,
                    args=([1], [params_ch1]))
                stim_thread.start()
            elif p_labels == 2:
                print("激活通道2")
                stim_thread = threading.Thread(
                    target=self._stimulate,
                    args=([2], [params_ch2]))
                stim_thread.start()
            else:
                p_labels = 0  # 无效分类

            # 发送分类结果 [注意力状态(1), MI分类结果]
            output = [1, p_labels]
        else:
            # 未检测到注意力
            output = [0, 0]  # 第一个0表示无注意力，第二个0表示无MI分类

        # 推送结果到LSL数据流
        print(f"输出结果: {output}")
        if self.outlet.have_consumers():
            self.outlet.push_sample(output)

    def post(self):
        # 关闭电刺激器连接
        if self.stimulator:
            self.stimulator.close()


if __name__ == '__main__':
    # 放大器的采样率
    srate = 1000
    # 截取数据的时间段，考虑进视觉刺激延迟140ms
    stim_interval = [0, 4]
    # 事件标签
    stim_labels = list(range(1, 3))
    cnts = 1  # .cnt数目
    # 数据路径
    filepath = "D:\\hyx_data"
    runs = list(range(1, cnts + 1))
    run_files = ['{:s}\\{:d}.cnt'.format(
        filepath, run) for run in runs]  # 具体数据路径
    pick_chs = ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
                'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'P5',
                'P3', 'P1', 'PZ', 'P2', 'P4', 'P6']

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(run_files=run_files,
                            pick_chs=pick_chs,
                            stim_interval=stim_interval,
                            stim_labels=stim_labels,
                            srate=srate,
                            lsl_source_id=lsl_source_id,
                            timeout=5e-2,
                            worker_name=feedback_worker_name,
                            ssmvep_freq=2.4,  # 新增: SSMVEP频率参数
                            attention_threshold=0.15)  # 新增: 注意力阈值参数

    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels)  # 打标签全为1
    # worker.pre()

    ns = NeuroScan(
        device_address=('169.254.80.232', 4000),
        srate=srate,
        num_chans=64)  # NeuroScan parameter

    # 与ns建立tcp连接
    ns.connect_tcp()
    # ns开始采集波形数据
    ns.start_acq()

    # register worker来实现在线处理
    ns.register_worker(feedback_worker_name, worker, marker)
    # 开启在线处理进程
    ns.up_worker(feedback_worker_name)
    # 等待 0.5s
    time.sleep(0.5)

    # ns开始截取数据线程，并把数据传递数据给处理进程
    ns.start_trans()

    # 任意键关闭处理进程
    input('press any key to close\n')
    # 关闭处理进程
    ns.down_worker('feedback_worker')
    # 等待 1s
    time.sleep(1)

    # ns停止在线截取线程
    ns.stop_trans()
    # ns停止采集波形数据
    ns.stop_acq()
    ns.close_connection()  # 与ns断开连接
    ns.clear()
    print('bye')
