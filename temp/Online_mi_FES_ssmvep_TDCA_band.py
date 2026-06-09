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
from metabci.brainda.algorithms.decomposition.tdca import TDCA,FBTDCA
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

def detect_ssmvep_snr(data, srate=1000, stim_freq=12, threshold=0.15):
    """
    使用传统SNR方法检测SSMVEP是否存在，作为备用方法

    参数:
    - data: 脑电数据
    - srate: 采样率
    - stim_freq: 刺激频率(5帧旋转，假设每秒60帧，则旋转频率为12Hz)
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

def detect_ssmvep_tdca(data, srate=1000, stim_freq=12, threshold=0.15):
    """
    使用TDCA方法检测SSMVEP是否存在，即被试是否在注视屏幕

    参数:
    - data: 脑电数据
    - srate: 采样率
    - stim_freq: 刺激频率(5帧旋转，屏幕刷新率为60，则旋转频率为12Hz)
    - threshold: 检测阈值

    返回:
    - attention: 布尔值，True表示检测到SSMVEP，被试正在注视屏幕
    - max_snr: 最大信噪比值，用于诊断
    """
    try:
        # 标准化数据
        data = data - np.mean(data, axis=-1, keepdims=True)
        data = data / np.std(data, axis=-1, keepdims=True)

        # 设置用于SSMVEP检测的频带
        wp = [(stim_freq - 1, stim_freq + 1)]  # 中心频带围绕刺激频率
        ws = [(stim_freq - 1.5, stim_freq + 1.5)]  # 过渡带稍宽一些
        filterbank = generate_filterbank(wp, ws, srate=srate, order=4, rp=0.5)

        # 构造参考信号 (模拟SSMVEP)
        t = np.arange(0, data.shape[-1]) / srate
        ref_signals = []

        # 构建基本正弦波参考信号及其谐波分量
        n_harmonics = 2  # 使用基频和一个谐波
        for h in range(1, n_harmonics + 1):
            ref_signals.append(np.sin(2 * np.pi * h * stim_freq * t))
            ref_signals.append(np.cos(2 * np.pi * h * stim_freq * t))

        # 对参考信号进行堆叠，生成完整参考矩阵
        Yf = np.array(ref_signals)

        # 准备TDCA所需的数据格式
        # TDCA期望形状为 [n_trials, n_channels, n_samples]
        X = data.reshape(1, data.shape[0], data.shape[1])

        # 构造虚拟标签 (对检测来说不重要，但TDCA API需要)
        y = np.array([0])  # 0表示"关注"类

        # 创建并配置FBTDCA模型
        model = FBTDCA(
            filterbank=filterbank,
            padding_len=5,  # 时域相关信息的补偿长度
            n_components=2,  # 提取的组件数量
            filterweights=np.array([1.0])  # 单个频带权重
        )

        # 创建待比较的参考模板
        # 使用相同数据但偏移一些，作为"不关注"的参考
        X_ref_attending = X.copy()

        # 对当前信号和参考模板进行TDCA变换
        features = model.fit(X=X, y=y, Yf=Yf).transform(X)

        # TDCA输出的特征通常是与每个类别相关性的向量
        # 使用最大相关性作为检测指标
        confidence = np.max(features)
        attention = confidence > threshold

        return attention, confidence

    except Exception as e:
        print(f"TDCA分析SSMVEP出错: {e}")
        # 如果TDCA分析失败，回退到传统方法
        return detect_ssmvep_snr(data, srate, stim_freq, threshold)





# 单频带CSP训练模型
def train_model_csp(X, y, srate=1000):
    """
    使用单频带CSP方法训练运动想象分类模型

    参数:
    - X: 训练数据，形状为 [n_trials, n_channels, n_samples]
    - y: 标签，形状为 [n_trials]
    - srate: 采样率

    返回:
    - model: 训练好的模型（管道包含CSP和SVC）
    """
    y = np.reshape(y, (-1))

    # 降采样
    X = resample(X, up=256, down=srate)

    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    # 创建滤波器组
    wp = [(4, 8), (8, 12), (12, 30)]  # 常用频带：θ, α, β
    ws = [(2, 10), (6, 14), (10, 32)]
    filterbank = generate_filterbank(wp, ws, srate=256, order=4, rp=0.5)

    # 创建FBCSP+SVC管道
    model = make_pipeline(*[
        FBCSP(n_components=5,
              n_mutualinfo_components=4,
              filterbank=filterbank),
        SVC(kernel='linear', C=1, decision_function_shape='ovr')
    ])

    # 训练模型
    model = model.fit(X, y)

    return model


# 单频带CSP预测
def model_predict_csp(X, srate=1000, model=None):
    """
    使用单频带CSP模型进行运动想象分类预测

    参数:
    - X: 测试数据，形状为 [n_trials, n_channels, n_samples]
    - srate: 采样率
    - model: 训练好的模型（管道包含CSP和SVC）

    返回:
    - p_labels: 预测标签
    """
    if model is None:
        raise ValueError("Model cannot be None.")

    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))

    # 降采样
    X = resample(X, up=256, down=srate)

    # 滤波 (可选，因为FBCSP内部会处理)
    # X = bandpass(X, 8, 30, 256)

    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    # 使用模型预测
    p_labels = model.predict(X)

    return p_labels


# 单频带CSP交叉验证
def offline_validation_csp(X, y, srate=1000):
    """
    使用单频带CSP进行离线交叉验证评估

    参数:
    - X: 数据，形状为 [n_trials, n_channels, n_samples]
    - y: 标签，形状为 [n_trials]
    - srate: 采样率

    返回:
    - mean_acc: 平均准确率
    """
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        # 训练模型
        model = train_model_csp(X_train, y_train, srate=srate)

        # 预测
        p_labels = model_predict_csp(X_test, srate=srate, model=model)

        # 计算准确率
        fold_acc = np.mean(p_labels == y_test)
        kfold_accs.append(fold_acc)

    mean_acc = np.mean(kfold_accs)
    return mean_acc


# 训练多频带CSP模型用于运动想象分类
def train_model_multi_band_csp(X, y, srate=1000):
    """
    使用多频带CSP方法训练运动想象分类模型

    参数:
    - X: 训练数据，形状为 [n_trials, n_channels, n_samples]
    - y: 标签，形状为 [n_trials]
    - srate: 采样率

    返回:
    - model_list: 每个频带对应的CSP+SVM模型列表
    """
    y = np.reshape(y, (-1))

    # 降采样
    X = resample(X, up=256, down=srate)

    # 零均值单位方差归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    # 定义多个频带窗口
    freq_windows = []
    min_freq = 8
    min_freq_max = 14
    max_freq = 30
    min_step = 2
    max_step = 10

    # 生成频带列表 (类似TffModel中的方法)
    for low in range(min_freq, min_freq_max + 1):
        for high in range(low + min_step, min(max_freq + 1, low + max_step)):
            freq_windows.append([low, high])

    print(f"总共评估{len(freq_windows)}个频带")

    # 存储每个频带的索引及准确率
    acc_freq_list = []

    # 由于计算量较大，可选择使用多进程加速
    # 这里使用简单的单进程实现
    for idx, freq_window in enumerate(freq_windows):
        # 使用交叉验证评估当前频带的性能
        m_X = X.copy()
        # 带通滤波到当前频带
        m_X = bandpass(m_X, freq_window[0], freq_window[1], 256)

        # 使用简单的5折交叉验证
        from sklearn.model_selection import KFold
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        acc = 0

        for train_idx, test_idx in kf.split(m_X):
            X_train, y_train = np.copy(m_X[train_idx]), np.copy(y[train_idx])
            X_test, y_test = np.copy(m_X[test_idx]), np.copy(y[test_idx])

            # CSP特征提取
            csp = mne.decoding.CSP(n_components=4, reg=None, log=None, norm_trace=False)
            csp.fit(X_train, y_train)

            # 转换数据
            X_train_csp = csp.transform(X_train)
            X_test_csp = csp.transform(X_test)

            # SVM分类
            svc = SVC(kernel='linear', C=1, decision_function_shape='ovr')
            svc.fit(X_train_csp, y_train)
            y_pred = svc.predict(X_test_csp)

            # 计算准确率
            acc += np.mean(y_pred == y_test)

        # 计算平均准确率
        acc_mean = acc / k
        result = {'index': idx, 'freq': freq_window, 'acc': acc_mean}
        acc_freq_list.append(result)
        print(f"频带 {freq_window} 准确率: {acc_mean:.4f}")

    # 按准确率排序
    acc_freq_list.sort(key=lambda x: x['acc'], reverse=True)

    # 选择前5个最佳频带
    best_bands = 5
    best_freq_list = [acc_freq_list[i]['freq'] for i in range(min(best_bands, len(acc_freq_list)))]
    print(f"选择的{best_bands}个最佳频带: {best_freq_list}")

    # 为每个选定的频带创建模型
    model_list = []
    for freq_window in best_freq_list:
        # 过滤数据到当前频带
        m_X = X.copy()
        m_X = bandpass(m_X, freq_window[0], freq_window[1], 256)

        # CSP特征提取
        csp = mne.decoding.CSP(n_components=4, reg=None, log=None, norm_trace=False)
        csp.fit(m_X, y)

        # 转换数据
        m_X_csp = csp.transform(m_X)

        # SVM分类
        svc = SVC(kernel='linear', C=1, decision_function_shape='ovr')
        svc.fit(m_X_csp, y)

        # 保存模型信息
        model_info = {'csp': csp, 'svc': svc, 'freq': freq_window}
        model_list.append(model_info)

    return model_list


# 使用多频带模型进行预测
def model_predict_multi_band_csp(X, srate=1000, model_list=None):
    """
    使用多频带CSP模型进行运动想象分类预测

    参数:
    - X: 测试数据，形状为 [n_trials, n_channels, n_samples]
    - srate: 采样率
    - model_list: 模型列表，每个元素包含一个频带的CSP和SVM模型

    返回:
    - pred_labels: 预测标签
    """
    if model_list is None or len(model_list) == 0:
        raise ValueError("模型列表为空")

    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))

    # 降采样
    X = resample(X, up=256, down=srate)

    # 零均值单位方差归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    # 存储每个频带模型的预测结果
    all_preds = []
    all_probs = []

    # 对每个频带模型进行预测
    for model_info in model_list:
        csp = model_info['csp']
        svc = model_info['svc']
        freq = model_info['freq']

        # 过滤数据到当前频带
        m_X = X.copy()
        m_X = bandpass(m_X, freq[0], freq[1], 256)

        # CSP特征提取
        m_X_csp = csp.transform(m_X)

        # SVM预测
        if hasattr(svc, 'decision_function'):
            # 获取决策函数值（可用于计算置信度）
            probs = svc.decision_function(m_X_csp)
            # 如果是二分类问题，确保形状正确
            if len(probs.shape) == 1:
                probs = np.column_stack([-probs, probs])
        else:
            # 如果没有decision_function，使用predict_proba
            probs = svc.predict_proba(m_X_csp)

        preds = svc.predict(m_X_csp)

        all_preds.append(preds)
        all_probs.append(probs)

    # 融合多个频带的预测结果
    # 1. 使用投票法
    all_preds = np.array(all_preds)
    if all_preds.shape[1] == 1:  # 只有一个样本
        # 使用众数作为最终预测
        from scipy import stats
        pred_labels = stats.mode(all_preds.flatten())[0]
    else:
        # 按列统计众数
        from scipy import stats
        pred_labels = np.array([stats.mode(all_preds[:, i])[0][0] for i in range(all_preds.shape[1])])

    return pred_labels


# 计算离线正确率 - 使用多频带CSP
def offline_validation_multi_band_csp(X, y, srate=1000):
    """
    使用多频带CSP进行离线交叉验证评估

    参数:
    - X: 数据，形状为 [n_trials, n_channels, n_samples]
    - y: 标签，形状为 [n_trials]
    - srate: 采样率

    返回:
    - mean_acc: 平均准确率
    - best_model_list: 最佳模型列表
    """
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    best_model_list = None

    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        # 使用多频带CSP训练模型
        model_list = train_model_multi_band_csp(X_train, y_train, srate=srate)

        # 使用多频带CSP预测
        p_labels = model_predict_multi_band_csp(X_test, srate=srate, model_list=model_list)

        # 计算准确率
        fold_acc = np.mean(p_labels == y_test)
        kfold_accs.append(fold_acc)

        # 保存最后一折的模型作为最终结果
        best_model_list = model_list

    mean_acc = np.mean(kfold_accs)
    return mean_acc, best_model_list


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
                 use_tdca_for_ssmvep=True,  # 是否使用TDCA算法检测SSMVEP
                 use_multi_band_csp=True,  # 是否使用多频带CSP处理运动想象
                 ssmvep_freq=2.4,  # SSMVEP刺激频率
                 attention_threshold=0.15):  # 注意力检测阈值
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        self.ssmvep_freq = ssmvep_freq
        self.attention_threshold = attention_threshold
        self.use_tdca_for_ssmvep = use_tdca_for_ssmvep  # TDCA只用于SSMVEP
        self.use_multi_band_csp = use_multi_band_csp  # 使用多频带CSP处理MI
        super().__init__(timeout=timeout, name=worker_name)
        self.stimulator = None  # 电刺激器
        self.stim_lock = None  # 线程锁

    def pre(self):
        X, y, ch_ind = read_data(run_files=self.run_files,
                                 chs=self.pick_chs,
                                 interval=self.stim_interval,
                                 labels=self.stim_labels)
        print("Loading data successfully")

        # 使用不同的模型训练方法和输出不同的提示
        if self.use_multi_band_csp:
            print("使用多频带CSP进行运动想象分类...")
            acc, model_list = offline_validation_multi_band_csp(X, y, srate=self.srate)
            print(f"当前多频带CSP模型准确率: {acc:.4f}")
            self.estimator = train_model_multi_band_csp(X, y, srate=self.srate)
        else:
            print("使用单频带CSP进行运动想象分类...")
            acc = offline_validation_csp(X, y, srate=self.srate)
            print(f"当前单频带CSP模型准确率: {acc:.4f}")
            self.estimator = train_model_csp(X, y, srate=self.srate)

        self.stimulator = ElectroStimulator('COM3')
        self.stim_lock = threading.Lock()  # 在子进程中初始化锁
        print("电刺激器初始化成功")
        self.ch_ind = ch_ind

        # 增加频道数量为2，一个表示注意力状态，一个表示MI分类结果
        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=2,
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

        # SSMVEP检测，判断被试是否注视屏幕
        # 使用TDCA方法或传统SNR方法检测SSMVEP
        if self.use_tdca_for_ssmvep:
            attention, attention_value = detect_ssmvep_tdca(
                data_all_channels,
                srate=self.srate,
                stim_freq=self.ssmvep_freq,
                threshold=self.attention_threshold
            )
            metric_name = "置信度"
        else:
            attention, attention_value = detect_ssmvep_snr(
                data_all_channels,
                srate=self.srate,
                stim_freq=self.ssmvep_freq,
                threshold=self.attention_threshold
            )
            metric_name = "SNR"

        # 输出注意力状态
        print(f"注意力状态: {'关注' if attention else '未关注'}, {metric_name}: {attention_value:.3f}")

        # 注意力检测逻辑
        # 只有检测到注意力才进行MI分类，否则直接返回
        if attention:
            # 根据配置选择使用多频带CSP或单频带CSP进行运动想象分类
            if self.use_multi_band_csp:
                p_labels = model_predict_multi_band_csp(data, srate=self.srate, model_list=self.estimator)
            else:
                p_labels = model_predict_csp(data, srate=self.srate, model=self.estimator)

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

    # 创建worker，使用两种优化方式：
    # 1. TDCA算法检测SSMVEP
    # 2. 多频带CSP处理运动想象
    worker = FeedbackWorker(run_files=run_files,
                            pick_chs=pick_chs,
                            stim_interval=stim_interval,
                            stim_labels=stim_labels,
                            srate=srate,
                            lsl_source_id=lsl_source_id,
                            timeout=5e-2,
                            worker_name=feedback_worker_name,
                            use_tdca_for_ssmvep=True,  # 使用TDCA检测SSMVEP
                            use_multi_band_csp=True,  # 使用多频带CSP处理运动想象
                            ssmvep_freq=2.4,
                            attention_threshold=0.15)

    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels)  # 打标签全为1
    # worker.pre()

    ns = NeuroScan(
        device_address=('169.254.80.232', 4000),
        srate=srate,
        num_chans=64)  # NeuroScan parameter

    print("=========================================")
    print("MI反馈系统初始化完成")
    print("使用TDCA算法检测SSMVEP（注意力状态）")
    print("使用多频带CSP算法处理运动想象（左右手）")
    print("=========================================")

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
    input('处理运行中，按任意键关闭...\n')
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
    print('系统已关闭，再见！')

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