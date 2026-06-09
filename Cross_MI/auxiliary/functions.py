from functools import reduce
import numpy as np
from pyriemann.tangentspace import TangentSpace, mean_covariance
from pyriemann.utils.covariance import covariances
from sklearn.covariance import LedoitWolf
import autograd.numpy as anp
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.base import invsqrtm
import math
import sys
from pymanopt import manifolds, Problem
from pymanopt_master.src.pymanopt_new import optimizers
import pymanopt
import os
from pyriemann.estimation import Shrinkage
from sklearn.model_selection import KFold

# 求Large-Dimensional 协方差矩阵
def LDcov(data):
    data = data.copy()
    data = data.transpose(0, 2, 1)
    if data.ndim == 2:
        LDcov = LedoitWolf().fit(data)
        cov = LDcov.covariance_
    if data.ndim == 3:
        cov = np.zeros((data.shape[0], data.shape[2], data.shape[2]))
        for i in range(len(data)):
            LDcov = LedoitWolf().fit(data[i, :, :])
            cov[i] = LDcov.covariance_
    return cov

#欧式空间对齐
def EA(X):
    #计算R矩阵
    R=np.zeros((X.shape[1], X.shape[1]))
#     for i in range(X.shape[0]):
#         R=R+np.dot(X[i,:,:],X[i,:,:].T)
    R=reduce(lambda R,y:R+np.dot(y,y.T),X,R)
    R=R/X.shape[0]
    for i in range(X.shape[0]):
        X[i,:,:]=np.dot(fuduiban(R),X[i,:,:])
    return X
#计算矩阵的负二分之一
def fuduiban(R):
    v, Q = np.linalg.eig(R)
    ss = np.diag(v ** (-0.5))
    #若出现异常值 就e补充0
    ss[np.isnan(ss)] = 0
    re = np.dot(Q, np.dot(ss, np.linalg.inv(Q)))
    #取实数部分
    return np.real(re)

def CORAL3(Xtrain, Xtest):
    Xtrain1 = []
    Xtest1 = []
    for i in range(Xtrain.shape[0]):
        cov_train = covariances(Xtrain[i,:,:].T)+np.eye(Xtrain.shape[2])
        cov_test = covariances(Xtest[i, :, :].T) + np.eye(Xtest.shape[2])
        v1, Q1 = np.linalg.eig(cov_train)
        V1 = np.diag(v1 ** (-0.5))
        T1 = Q1 * V1 * np.linalg.inv(Q1)
        v2, Q2 = np.linalg.eig(cov_test)
        V2 = np.diag(v2 ** (-0.5))
        T2 = Q2 * V2 * np.linalg.inv(Q2)
        # T1 = fuduiban(cov_train)
        # T2 = fuduiban(cov_test)
        A_coral = np.dot(T1, T2)
        Sim_coral = np.dot(np.dot(Xtrain[i,:,:], A_coral), Xtest[i,:,:].T)
        Sim_Trn = np.dot(np.dot(Xtrain[i,:,:], A_coral), Xtrain[i,:,:].T)
        Xtrain1.append(Sim_Trn)
        Xtest1.append(Sim_coral)
    return Xtrain1, Xtest1

# 这个是计算较快的
def CORAL2(Xtrain, Xtest):
    Xtrain1 = []
    Xtest1 = []
    for i in range(Xtrain.shape[0]):
        cov_train = covariances(Xtrain[i,:,:])+np.eye(Xtrain.shape[1])
        cov_test = covariances(Xtest[i, :, :]) + np.eye(Xtest.shape[1])
        v1, Q1 = np.linalg.eig(cov_train)
        V1 = np.diag(v1 ** (-0.5))
        T1 = Q1 * V1 * np.linalg.inv(Q1)
        v2, Q2 = np.linalg.eig(cov_test)
        V2 = np.diag(v2 ** (-0.5))
        T2 = Q2 * V2 * np.linalg.inv(Q2)
        # T1 = fuduiban(cov_train)
        # T2 = fuduiban(cov_test)
        A_coral = np.dot(T1, T2)
        Sim_coral = np.dot(np.dot(Xtrain[i,:,:].T, A_coral), Xtest[i,:,:])
        Sim_Trn = np.dot(np.dot(Xtrain[i,:,:].T, A_coral), Xtrain[i,:,:])
        Xtrain1.append(Sim_Trn)
        Xtest1.append(Sim_coral)
    Xtrain1 = np.array(Xtrain1)
    Xtest1 = np.array(Xtest1)
    # Xtrain2 = LDcov(Xtrain1)
    # Xtest2 = LDcov(Xtest1)
    return Xtrain1, Xtest1

def RA(Xtrain, Xtest):
    covdata = covariances(Xtrain)
    P = mean_covariance(covdata)
    v1, Q1 = np.linalg.eig(P)
    V1 = np.diag(v1 ** (-0.5))
    P1 = Q1 * V1 * np.linalg.inv(Q1)
    Xtrain1 = []
    Xtest1 = []
    for s in range(Xtrain.shape[0]):
        Xtrain1.append(np.dot(P1, Xtrain[s, : , :]))
    for s in range(Xtest.shape[0]):
        Xtest1.append(np.dot(P1, Xtest[s, : , :]))
    Xtrain1 = np.array(Xtrain1)
    Xtest1 = np.array(Xtest1)
    return Xtrain1, Xtest1

def RPA(Xtrain, Xtest, ytrain, train_clabel, test_clabel, metric="euclid"):
    data_s =covariances(Xtrain[train_clabel != test_clabel[0], :, :])
    data_tl = covariances(Xtrain[train_clabel == test_clabel[0], :, :])
    data_tu = covariances(Xtest)
    label_s = ytrain[train_clabel != test_clabel[0]]
    label_tl = ytrain[train_clabel == test_clabel[0]]
    M_s = mean_covariance(data_s)
    M_tl = mean_covariance(data_tl)
    data_rct_s = np.array([invsqrtm(M_s) @ data_s[i, :, :] @ invsqrtm(M_s) for i in range(data_s.shape[0])])
    data_rct_tl = np.array([invsqrtm(M_tl) @ data_tl[i, :, :] @ invsqrtm(M_tl) for i in range(data_tl.shape[0])])
    data_rct_tu = np.array([invsqrtm(M_tl) @ data_tu[i, :, :] @ invsqrtm(M_tl) for i in range(data_tu.shape[0])])
    d_s = sum([distance_riemann(M_s, data_s[i, :, :]) for i in range(data_s.shape[0])])
    d_tl = sum([distance_riemann(M_tl, data_tl[i, :, :]) for i in range(data_tl.shape[0])])
    s = math.sqrt(d_s / d_tl)
    data_str_tl = np.array(
        [np.linalg.matrix_power(data_rct_tl[i, :, :], int(s)) for i in range(data_rct_tl.shape[0])])
    data_str_tu = np.array(
        [np.linalg.matrix_power(data_rct_tu[i, :, :], int(s)) for i in range(data_rct_tu.shape[0])])
    G_k_s = []
    G_k_tl = []
    for i in set(label_tl):
        M_k_s = mean_covariance(data_s[label_s == i, :, :])
        M_k_tl = mean_covariance(data_tl[label_tl == i, :, :])
        G_k_s.append(invsqrtm(M_s) @ M_k_s @ invsqrtm(M_s))
        G_k_tl.append(invsqrtm(M_tl) @ M_k_tl @ invsqrtm(M_tl))
    w_k = [0.5, 0.5]
    manifold = manifolds.stiefel.Stiefel(G_k_tl[0].shape[0], G_k_tl[0].shape[1])
    if metric == "euclid":
        @pymanopt.function.autograd(manifold)
        def cost(point):
            return sum([anp.linalg.norm(G_k_s[i] - anp.dot(point, anp.dot(G_k_tl[i], point.T))) ** 2 for i in
                                 range(len(G_k_tl))])
    if metric == "riemann":
        @pymanopt.function.autograd(manifold)
        def cost(point):
            # point = np.array(point, dtype=float)
            return sum([w_k[i] * distance_riemann(G_k_tl[i], point @ G_k_s[i] @ point.T) for i in
                                 range(len(G_k_tl))])
            # return sum([w_k[i] * 4 * logm(G_k_tl[i] @ point @ G_k_s[i] @ point.T) for i in
            #                      range(len(G_k_tl))])
    sys.stdout = open(os.devnull, 'w')  # 关闭print
    problem = Problem(manifold, cost)
    # solver = SteepestDescent(mingradnorm=1e-3)
    # U = solver.solve(problem)
    optimizer = optimizers.SteepestDescent()
    result = optimizer.run(problem)
    U = result.point
    sys.stdout = sys.__stdout__  # 打开print
    data_rot_tl = np.array(
        [U.T @ data_str_tl[i, :, :] @ U for i in range(data_str_tl.shape[0])])
    data_rot_tu = np.array(
        [U.T @ data_str_tu[i, :, :] @ U for i in range(data_str_tu.shape[0])])
    data_train = np.concatenate((data_rct_s, np.array(data_rot_tl)), axis=0)
    label_train = np.concatenate((label_s, label_tl), axis=0)
    data_test = np.array(data_rot_tu)
    shrinkage = Shrinkage(shrinkage=0.01)
    data_train_s = shrinkage.transform(data_train)
    data_test_s = shrinkage.transform(data_test)
    return data_train_s, label_train, data_test_s


from sklearn.model_selection import StratifiedKFold
import numpy as np


def Split_Sets_random_state(total_fold, random_state, data, labels=None):
    """
    增强版函数：如果有labels就用分层抽样，没有就用普通抽样
    """
    train_index = []
    test_index = []
    # 如果没有传入labels，使用普通KFold
    if labels is None:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=total_fold, shuffle=True, random_state=random_state)
        splitter = kf.split(data)
    else:
        # 检查是否有足够的类别
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            print("警告：数据中只有一个类别，使用普通KFold")
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=total_fold, shuffle=True, random_state=random_state)
            splitter = kf.split(data)
        else:
            # 使用分层抽样
            skf = StratifiedKFold(n_splits=total_fold, shuffle=True, random_state=random_state)
            splitter = skf.split(data, labels)

    for train_i, test_i in splitter:
        train_index.append(train_i)
        test_index.append(test_i)

    return train_index, test_index