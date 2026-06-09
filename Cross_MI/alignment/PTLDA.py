from pyriemann.estimation import Shrinkage
import numpy as np
from pyriemann.tangentspace import TangentSpace, mean_covariance
from scipy.linalg import sqrtm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
class PTLDA():
    def __init__(self):
        pass
    def fit(self,X,y,c_label):
        '''
        :param X: 输入协方差矩阵
        :param y:
        :param clabel: 被试标签
        :return:
        '''
        self.X_train = X
        self.c_label_train = c_label
        self.y_train =y
    def predict(self, X,c_label):
        X_all = np.concatenate((self.X_train, X), axis=0)
        c_label_all = np.concatenate((self.c_label_train, c_label), axis=0)
        shrinkage = Shrinkage(shrinkage=0)
        X_s = shrinkage.transform(X_all)
        c_type = np.unique(c_label_all)
        P = np.zeros((len(c_type), X_s.shape[1], X_s.shape[2]))  # 保存各个子集（域）的黎曼均值
        for c in c_type:
            P[int(c), :, :] = mean_covariance(X_s[c_label == c])
        P_all = mean_covariance(P)  # 所有子集黎曼均值的黎曼均值
        E = np.zeros((len(c_type), X_s.shape[1], X_s.shape[2]))
        PT_all = np.zeros(X_s.shape)  # 保存各个子集（域）的并行传输
        for c in c_type:
            # v1, Q1 = np.linalg.eig(P_all @ np.linalg.inv(P[int(c), :, :]))
            # V1 = np.diag(v1 ** (0.5))
            # E[int(c), :, :] = Q1 * V1 * np.linalg.inv(Q1)
            E[int(c), :, :] = sqrtm(P_all @ np.linalg.inv(P[int(c), :, :]))
        for i in range(X_s.shape[0]):
            c = c_label[i]
            PT_all[i, :, :] = np.dot(np.dot(E[int(c)], X_s[i, :, :]), E[int(c)].T)  # 计算并行传输
        ts = TangentSpace()
        ts_PT_all = ts.transform(PT_all)
        ts_PT_train = ts_PT_all[:self.Xtrain.shape[0], :]
        ts_PT_test = ts_PT_all[(self.Xtrain.shape[0]):, :]
        lda = LDA(solver='eigen', shrinkage='auto')
        lda.fit(ts_PT_train, self.y_train)
        res = lda.predict(ts_PT_test)
        return res
