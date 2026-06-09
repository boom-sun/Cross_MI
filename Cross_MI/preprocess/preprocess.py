import numpy as np
from scipy.signal import butter, cheby1, filtfilt, resample
from sklearn.base import BaseEstimator, TransformerMixin


# 带通滤波器类
class BandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, fs, lowcut, highcut, order=5, filter_type='butter'):
        """
        带通滤波器类

        参数:
        lowcut (float): 带通滤波器的低频截止频率。
        highcut (float): 带通滤波器的高频截止频率。
        fs (float): 采样频率。
        order (int): 滤波器的阶数。
        filter_type (str): 滤波器类型，'butter' 或 'cheby1'。
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.filter_type = filter_type

    def __repr__(self):
        return f"BandpassFilter(lowcut={self.lowcut}, highcut={self.highcut}, " \
               f"fs={self.fs}, order={self.order}, filter_type={self.filter_type})"

    def _get_filter_coeff(self):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        if self.filter_type == 'butter':
            self.b, self.a = butter(self.order, [low, high], btype='band')
        elif self.filter_type == 'cheby1':
            self.b, self.a = cheby1(self.order, 0.5, [low, high], btype='band')
        else:
            raise ValueError("filter_type must be 'butter' or 'cheby1'")


    def _bandpass_filter(self, data):

        return filtfilt(self.b, self.a, data, axis=-1)

    def fit(self):
        self._get_filter_coeff()
        return self

    def transform(self, X, y=None):
        """
        输入：
        X (array-like): 输入信号。
        shape=(n_trials, n_channels, n_samples) or (n_channels, n_samples)

        输出：
        array-like: 带通滤波后的信号。
        """
        return self._bandpass_filter(X)

class Preprocess():
    def __init__(self, config):
        self.fre_win = config['fre_win']
        self.srate = config['srate']

    def _make_filter(self):
        bandpassfilter = BandpassFilter(fs=self.srate, lowcut=self.fre_win[0], highcut=self.fre_win[1])
        bandpassfilter.fit()
        return bandpassfilter

    def preprocess(self, X_train, X_test):
        bandpassfilter = self._make_filter()
        X_train1 = bandpassfilter.transform(X_train)
        X_test1 = bandpassfilter.transform(X_test)
        return X_train1, X_test1
    def transform(self, X):
        bandpassfilter = self._make_filter()
        return bandpassfilter.transform(X)

