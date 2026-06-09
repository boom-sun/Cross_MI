import numpy as np
from Cross_MI.auxiliary.functions import EA, RA, CORAL2, RPA
from pyriemann.utils.covariance import covariances
class Alignment():
    '''
    RPA得到的是协方差矩阵
    '''
    def __init__(self, srate, Alignment):
            """Init."""
            self.srate = np.squeeze(srate)
            self.Alignment = Alignment

    # 新增：标准 fit_transform / transform
    def fit_transform(self, X_train, y_train, clabel_train=[], clabel_test=[]):
        # 兼容旧配置中写成 'RSF' 的误用；以及常见 'EA' 标识
        if (isinstance(self.Alignment, str) and (('EA' in self.Alignment) or ('RSF' in self.Alignment))) or \
           (isinstance(self.Alignment, (list, tuple)) and any([('EA' in str(a)) or ('RSF' in str(a)) for a in self.Alignment])):
            self.Alignment = 'EA'
            return EA(X_train)
        else:
            self.align_method = 'NONE'
            return X_train

    def transform(self, X_test):
        if self.align_method == 'EA':
            return EA(X_test)
        return X_test

    # 兼容旧接口
    def alignment(self, X_train, X_test, y_train, clabel_train=[], clabel_test=[]):
        X1 = self.fit_transform(X_train, y_train, clabel_train, clabel_test)
        X2 = self.transform(X_test)
        return X1, X2
