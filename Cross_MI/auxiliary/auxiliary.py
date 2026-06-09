import numpy as np
from .rsf import RSF
from pyriemann.utils.covariance import covariances
class Auxiliary():
    def __init__(self, srate, Auxiliary):
            """Init."""
            self.srate = np.squeeze(srate)
            self.Auxiliary = Auxiliary
            self.aux_model = None  # 新增：缓存辅助模型（如 RSF）

    # 新增：标准 fit / transform / fit_transform
    def fit(self, X_train, y_train, clabel_train=[]):
        if 'RSF' in self.Auxiliary:
            self.aux_model = 'RSF'
            self.aux_model = RSF(dim=8, method='default')
            X_train_out = self.aux_model.fit_transform(X_train, y_train)
        else:
            self.aux_model = None  # 恒等
            X_train_out = X_train
        return X_train_out

    def transform(self, X):
        if self.aux_model is not None:
            return self.aux_model.transform(X)
        return X

    def fit_transform(self, X_train, y_train, clabel_train=[]):
        X_train_out = self.fit(X_train, y_train, clabel_train)
        return X_train_out

    # 兼容旧接口（不改原调用处也能工作）
    def auxiliary(self, X_train, X_test, y_train, clabel_train=[], clabel_test=[]):
        X1 = self.fit_transform(X_train, y_train, clabel_train)
        X2 = self.transform(X_test)
        return X1, X2
