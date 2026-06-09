import numpy as np
import MrLiuTuo.MatrixHacker.matrixhacker.algorithms.manifold.riemann as RGC
from MrLiuTuo.MatrixHacker.matrixhacker.algorithms.utils.filtering import OnlineBlockFilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import scipy
import scipy.io as sio
from MetaBCI.metabci.brainda.algorithms.decomposition.csp import CSP
from MetaBCI.metabci.brainda.algorithms.decomposition.csp import FBCSP
from sklearn.metrics import accuracy_score
from MetaBCI.metabci.brainda.algorithms.decomposition.base import generate_filterbank
from MrLiuTuo.MatrixHacker.matrixhacker.algorithms.deeplearning.EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
import h5py
import random

s_filename = ['杨涵', '姬智敏', '姜传力', '辛海玲', '杨明明','孙新维','姜传力_5_30','潘林聪']
sencename = ['sys', 'xe']
subjectname = ['yh', 'jzm', 'jcl', 'xhl', 'ymm', 'sxw', 'jcl_5_30', 'plc']
mat_filpath = 'E:\跨场景实验\预实验数据\分段后数据\\'

# 参数设置
cross = 0   # 0为非跨场景，1为跨场景
choose = 0  # 1为九折交叉，0为依次增加选择训练数据，
choose_subject = range(5)    # 选择被试
# choose_subject = [0]      # 选择被试
choose_sence = [1]        # 选择场景
num_block = 6             # block数
n_trial = 12            # 试次数
n_channel = 64          # 导联数
n_sample = 4000         # 任务态采样点数
rest_sample = 2000      # 数据内静息时间
srate = 1000            # 采样率



wp = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30), (5, 30)]
ws = [(2, 10), (6, 14), (10, 18), (14, 22), (18, 26), (22, 30)]
filterbank = generate_filterbank(wp, ws, srate=250)

def cov_matrix(data):
    data=data.copy()
    if data.ndim==2:
        for i in range(len(data)):
            data_i_mean=np.mean(data[i])
            data[i]=data[i]-data_i_mean
        cov=(1/(data.shape[1]-1))*np.dot(data,data.T)
    if data.ndim==3:
        cov=np.zeros((data.shape[0],data.shape[1],data.shape[1]))
        for i in range(len(data)):
            for j in range(len(data[0])):
                data_ij_mean=np.mean(data[i][j])
                data[i][j]=data[i][j]-data_ij_mean
            cov[i]=(1/(data.shape[2]-1))*np.dot(data[i],data[i].T)
    return cov

Online_filter = OnlineBlockFilter(srate=srate, filters=[[4, 30]])
Online_filter.fit()


if choose == 0:
    if cross == 0:
        all_sub_acc = []

        # 选择被试
        for n_sname in choose_subject:
            # 创建空列表，以便储存每个算法的准确率
            csp_single_sub_acc = []
            fbcsp_single_sub_acc = []
            eegnet_single_sub_acc = []
            mdm_single_sub_acc = []
            tslda_single_sub_acc = []
            # 创建空列表，以便储存单个被试的准确率
            single_sub_acc = []
            # 选择场景
            for n_sence in choose_sence:
                # 选择block数
                for n_train in range(1, num_block):
                    Xtrain = np.empty((n_channel, n_sample, 0), int)
                    Xtest = np.empty((n_channel, n_sample, 0), int)
                    y = np.empty((0), int)
                    y_test = np.empty((0), int)
                    # y_single = np.concatenate((np.ones(12), np.ones(12) * 2, np.ones(12) * 3, np.ones(12) * 4), axis=0)
                    y_single = np.concatenate((np.zeros(n_trial), np.ones(n_trial)), axis=0)
                    for n_block in range(1, num_block+1):
                        filepath = [mat_filpath, str(s_filename[n_sname]), '\\', str(sencename[n_sence]), '_', str(subjectname[n_sname]), '_', str(n_block), '_fenduan.mat']  # mat文件路径
                        filepath = ''.join(filepath)
                        data = scipy.io.loadmat(filepath)
                        data1 = data['data1'][:, rest_sample:(rest_sample+n_sample), :]
                        data2 = data['data2'][:, rest_sample:(rest_sample+n_sample), :]
                        if n_block in range(1, n_train+1):
                            Xtrain = np.concatenate((Xtrain, data1, data2), axis=2)
                            y = np.hstack((y, y_single))
                        else:
                            Xtest = np.concatenate((Xtest, data1, data2), axis=2)
                            y_test = np.hstack((y_test, y_single))

                    Xtrain = Xtrain.transpose(2, 0, 1)
                    Xtest = Xtest.transpose(2, 0, 1)
                    f_X_train = np.squeeze(Online_filter.transform(Xtrain))
                    f_X_test = np.squeeze(Online_filter.transform(Xtest))
                    X_mat_train = cov_matrix(f_X_train)   # 求协方差矩阵
                    X_mat_test = cov_matrix(f_X_test)  # 求协方差矩阵

                    # # csp+svm
                    # csp = CSP()
                    # csp.fit(Xtrain, y)
                    # csp_x_train = csp.transform(Xtrain)
                    # cspsvm = svm.SVC()
                    # cspsvm.fit(csp_x_train, y)
                    # cspsvm_res = []
                    # csp_X_test = csp.transform(Xtest)
                    # cspsvm_res = cspsvm.predict(csp_X_test)
                    # csp_single_sub_acc.append(accuracy_score(y_test, cspsvm_res))
                    #
                    # # fbcsp+svm
                    # fbcsp = FBCSP(n_components=8, n_mutualinfo_components=20, filterbank=filterbank)
                    # fbcsp.fit(Xtrain, y)
                    # fbcsp_x_train = fbcsp.transform(Xtrain)
                    # fbcspsvm = svm.SVC()
                    # fbcspsvm.fit(fbcsp_x_train, y)
                    # fbcspsvm_res = []
                    # fbcsp_X_test = fbcsp.transform(Xtest)
                    # fbcspsvm_res = fbcspsvm.predict(fbcsp_X_test)
                    # fbcsp_single_sub_acc.append(accuracy_score(y_test, fbcspsvm_res))

                    # eegnet
                    eegnet = EEGNet(Chans=64, Samples=4000, kernLength=64,
                                    F1=8, D=2, F2=16)
                    eegnet.compile(loss='binary_crossentropy', optimizer='adam')
                    eegnet.fit(Xtrain, np_utils.to_categorical(y), batch_size=None,
                               epochs=200)
                    eegnet_res = eegnet.predict(Xtest).argmax(axis=-1)
                    eegnet_single_sub_acc.append(accuracy_score(y_test, eegnet_res))

                    # mdm
                    # mdm = RGC.MDM()
                    # mdm.fit(X_mat_train, y)
                    # mdm_res = mdm.predict(X_mat_test)
                    # mdm_single_sub_acc.append(accuracy_score(y_test, mdm_res))
                    #
                    # # tslda
                    # tslda = RGC.TSclassifier(clf=LinearDiscriminantAnalysis())
                    # tslda.fit(X_mat_train, y)
                    # tslda_res = tslda.predict(X_mat_test)
                    # tslda_single_sub_acc.append(accuracy_score(y_test, tslda_res))


                # single_sub_acc.append(csp_single_sub_acc)
                # single_sub_acc.append(fbcsp_single_sub_acc)
                single_sub_acc.append(eegnet_single_sub_acc)
                # single_sub_acc.append(mdm_single_sub_acc)
                # single_sub_acc.append(tslda_single_sub_acc)
                print('一个人结束')

            all_sub_acc.append(single_sub_acc)


        scipy.io.savemat('acc_all_no_cross_xe_eegnet_1.mat', {'acc': all_sub_acc})
        # scipy.io.savemat('acc_all_no_cross_jcl.mat', {str(subjectname[0]): all_sub_acc[0], str(subjectname[1]): all_sub_acc[1]
        #                                  , str(subjectname[2]): all_sub_acc[2], str(subjectname[3]): all_sub_acc[3]
        #                                  , str(subjectname[4]): all_sub_acc[4]
        #                                  })
        scipy.io.savemat('acc_all_no_cross_xe_eegnet.mat', {str(subjectname[0]): all_sub_acc[0]
                                        })


    if cross == 1:
        all_sub_acc = []
        for n_sname in choose_subject:
            single_sub_acc = []
            csp_single_sub_acc = []
            fbcsp_single_sub_acc = []
            eegnet_single_sub_acc = []
            mdm_single_sub_acc = []
            tslda_single_sub_acc = []

            for n_train in range(num_block):
                Xtrain = np.empty((n_channel, n_sample, 0), int)
                y = np.empty((0), int)
                y_single = np.concatenate((np.zeros(n_trial), np.ones(n_trial)), axis=0)
                for n_block in range(1, num_block + 1):
                    filepath = [mat_filpath, str(s_filename[n_sname]), '\\', 'sys', '_', str(subjectname[n_sname]), '_',
                                str(n_block), '_fenduan.mat']  # mat文件路径
                    filepath = ''.join(filepath)
                    data = scipy.io.loadmat(filepath)
                    data1 = data['data1'][:, rest_sample:(rest_sample + n_sample), :]
                    data2 = data['data2'][:, rest_sample:(rest_sample + n_sample), :]
                    Xtrain = np.concatenate((Xtrain, data1, data2), axis=2)
                    y = np.hstack((y, y_single))

                Xtest = np.empty((n_channel, n_sample, 0), int)
                y_test = np.empty((0), int)

                for n_block in range(1, num_block+1):
                    filepath = [mat_filpath, str(s_filename[n_sname]), '\\', 'xe', '_', str(subjectname[n_sname]), '_', str(n_block), '_fenduan.mat']  # mat文件路径
                    filepath = ''.join(filepath)
                    data = scipy.io.loadmat(filepath)
                    data1 = data['data1'][:, rest_sample:(rest_sample+n_sample), :]
                    data2 = data['data2'][:, rest_sample:(rest_sample+n_sample), :]
                    if n_block in range(n_train+1):
                        Xtrain = np.concatenate((Xtrain, data1, data2), axis=2)
                        y = np.hstack((y, y_single))
                    else:
                        Xtest = np.concatenate((Xtest, data1, data2), axis=2)
                        y_test = np.hstack((y_test, y_single))


                Xtrain = Xtrain.transpose(2, 0, 1)
                Xtest = Xtest.transpose(2, 0, 1)
                f_X_train = np.squeeze(Online_filter.transform(Xtrain))
                f_X_test = np.squeeze(Online_filter.transform(Xtest))
                X_mat_train = cov_matrix(f_X_train)   # 求协方差矩阵
                X_mat_test = cov_matrix(f_X_test)  # 求协方差矩阵

                # csp+svm
                csp = CSP()
                csp.fit(Xtrain, y)
                csp_x_train = csp.transform(Xtrain)
                cspsvm = svm.SVC()
                cspsvm.fit(csp_x_train, y)
                cspsvm_res = []
                csp_X_test = csp.transform(Xtest)
                cspsvm_res = cspsvm.predict(csp_X_test)
                csp_single_sub_acc.append(accuracy_score(y_test, cspsvm_res))

                # fbcsp+svm
                fbcsp = FBCSP(n_components=8, n_mutualinfo_components=20, filterbank=filterbank)
                fbcsp.fit(Xtrain, y)
                fbcsp_x_train = fbcsp.transform(Xtrain)
                fbcspsvm = svm.SVC()
                fbcspsvm.fit(fbcsp_x_train, y)
                fbcspsvm_res = []
                fbcsp_X_test = fbcsp.transform(Xtest)
                fbcspsvm_res = fbcspsvm.predict(fbcsp_X_test)
                fbcsp_single_sub_acc.append(accuracy_score(y_test, fbcspsvm_res))

                # eegnet
                eegnet = EEGNet(Chans=64, Samples=4000, kernLength=64,
                                F1=8, D=2, F2=16)
                eegnet.compile(loss='binary_crossentropy', optimizer='adam')
                eegnet.fit(Xtrain, np_utils.to_categorical(y), batch_size=None,
                           epochs=200)
                eegnet_res = eegnet.predict(Xtest).argmax(axis=-1)
                eegnet_single_sub_acc.append(accuracy_score(y_test, eegnet_res))
                #
                # # mdm
                # mdm = RGC.MDM()
                # mdm.fit(X_mat_train, y)
                # mdm_res = mdm.predict(X_mat_test)
                # mdm_single_sub_acc.append(accuracy_score(y_test, mdm_res))
                #
                # # tslda
                # tslda = RGC.TSclassifier(clf=LinearDiscriminantAnalysis())
                # tslda.fit(X_mat_train, y)
                # tslda_res = tslda.predict(X_mat_test)
                # tslda_single_sub_acc.append(accuracy_score(y_test, tslda_res))


            single_sub_acc.append(csp_single_sub_acc)
            single_sub_acc.append(fbcsp_single_sub_acc)
            single_sub_acc.append(eegnet_single_sub_acc)
            # single_sub_acc.append(mdm_single_sub_acc)
            # single_sub_acc.append(tslda_single_sub_acc)
            print('一个人结束')

        scipy.io.savemat('acc_all_cross_1.mat', {'acc': all_sub_acc})
        scipy.io.savemat('acc_all_cross.mat', {str(subjectname[0]): all_sub_acc[0], str(subjectname[1]): all_sub_acc[1]
                                         , str(subjectname[2]): all_sub_acc[2], str(subjectname[3]): all_sub_acc[3]
                                         , str(subjectname[4]): all_sub_acc[4]
                                         })


if choose == 1:
    if cross == 0:
        all_sub_acc = []

        for n_sname in choose_subject:
            csp_single_sub_acc = []
            fbcsp_single_sub_acc = []
            eegnet_single_sub_acc = []
            mdm_single_sub_acc = []
            tslda_single_sub_acc = []
            single_sub_acc = []
            for n_sence in choose_sence:
                Xall = np.empty((n_channel, n_sample, 0), int)
                yall = np.empty((0), int)
                y_single = np.concatenate((np.zeros(n_trial), np.ones(n_trial)), axis=0)
                for n_block in range(1, num_block + 1):
                    filepath = [mat_filpath, str(s_filename[n_sname]), '\\', str(sencename[n_sence]), '_',
                                str(subjectname[n_sname]), '_', str(n_block), '_fenduan.mat']  # mat文件路径
                    filepath = ''.join(filepath)
                    data = scipy.io.loadmat(filepath)
                    data1 = data['data1'][:, rest_sample:(rest_sample + n_sample), :]
                    data2 = data['data2'][:, rest_sample:(rest_sample + n_sample), :]
                    Xall = np.concatenate((Xall, data1, data2), axis=2)
                    yall = np.hstack((yall, y_single))

                Xall = Xall.transpose(2, 0, 1)
                index = list(range(len(Xall)))
                random.shuffle(index)
                Xall = Xall[index, :, :]
                yall = yall[index]
                nall = Xall.shape[0]

                for n_train in range(9):
                    index_test = index[int(n_train*nall/9):int((n_train+1)*nall/9)]
                    index_train = index[:]
                    index_train[int(n_train*nall/9):int((n_train+1)*nall/9)] = []
                    raw_Xtrain = Xall[index_train, :, :]
                    raw_Xtest = Xall[index_test, :, :]
                    y = yall[index_train]
                    y_test = yall[index_test]

                    Xtrain = np.squeeze(Online_filter.transform(raw_Xtrain))
                    Xtest = np.squeeze(Online_filter.transform(raw_Xtest))
                    X_mat_train = cov_matrix(Xtrain)  # 求协方差矩阵
                    X_mat_test = cov_matrix(Xtest)  # 求协方差矩阵

                    # csp+svm
                    csp = CSP()
                    model = csp.fit(Xtrain, y)
                    csp_x_train = csp.transform(Xtrain)
                    cspsvm = svm.SVC()
                    cspsvm.fit(csp_x_train, y)
                    cspsvm_res = []
                    csp_X_test = csp.transform(Xtest)
                    cspsvm_res = cspsvm.predict(csp_X_test)
                    csp_single_sub_acc.append(accuracy_score(y_test, cspsvm_res))

                    # fbcsp+svm
                    fbcsp = FBCSP(n_components=8, n_mutualinfo_components=20, filterbank=filterbank)
                    fbcsp.fit(Xtrain, y)
                    fbcsp_x_train = fbcsp.transform(Xtrain)
                    fbcspsvm = svm.SVC()
                    fbcspsvm.fit(fbcsp_x_train, y)
                    fbcspsvm_res = []
                    fbcsp_X_test = fbcsp.transform(Xtest)
                    fbcspsvm_res = fbcspsvm.predict(fbcsp_X_test)
                    fbcsp_single_sub_acc.append(accuracy_score(y_test, fbcspsvm_res))

                    # eegnet
                    eegnet = EEGNet(Chans=64, Samples=4000, kernLength=64,
                                    F1=8, D=2, F2=16)
                    eegnet.compile(loss='binary_crossentropy', optimizer='adam')
                    eegnet.fit(Xtrain, np_utils.to_categorical(y), batch_size=None,
                               epochs=200)
                    eegnet_res = eegnet.predict(Xtest).argmax(axis=-1)
                    eegnet_single_sub_acc.append(accuracy_score(y_test, eegnet_res))

                    # # mdm
                    # mdm = RGC.MDM()
                    # mdm.fit(X_mat_train, y)
                    # mdm_res = mdm.predict(X_mat_test)
                    # mdm_single_sub_acc.append(accuracy_score(y_test, mdm_res))
                    #
                    # # tslda
                    # tslda = RGC.TSclassifier(clf=LinearDiscriminantAnalysis())
                    # tslda.fit(X_mat_train, y)
                    # tslda_res = tslda.predict(X_mat_test)
                    # tslda_single_sub_acc.append(accuracy_score(y_test, tslda_res))

                single_sub_acc.append(csp_single_sub_acc)
                single_sub_acc.append(fbcsp_single_sub_acc)
                single_sub_acc.append(eegnet_single_sub_acc)
                # single_sub_acc.append(mdm_single_sub_acc)
                # single_sub_acc.append(tslda_single_sub_acc)
                print('一个人结束')

            all_sub_acc.append(single_sub_acc)

        scipy.io.savemat('acc_all_no_cross_xe_10_3_1.mat', {'acc': all_sub_acc})
        scipy.io.savemat('acc_all_no_cross_xe_10_3.mat', {str(subjectname[0]): all_sub_acc[0], str(subjectname[1]): all_sub_acc[1]
                                         , str(subjectname[2]): all_sub_acc[2], str(subjectname[3]): all_sub_acc[3]
                                         , str(subjectname[4]): all_sub_acc[4]
                                         })
        # scipy.io.savemat('acc_all_no_cross_xe_eegnet.mat', {str(subjectname[0]): all_sub_acc[0]
        #                                                     })

    if cross == 1:
        all_sub_acc = []
        for n_sname in choose_subject:
            single_sub_acc = []
            csp_single_sub_acc = []
            fbcsp_single_sub_acc = []
            eegnet_single_sub_acc = []
            mdm_single_sub_acc = []
            tslda_single_sub_acc = []

            Xtrain = np.empty((n_channel, n_sample, 0), int)
            y = np.empty((0), int)
            Xtest = np.empty((n_channel, n_sample, 0), int)
            y_test = np.empty((0), int)
            y_single = np.concatenate((np.zeros(n_trial), np.ones(n_trial)), axis=0)

            for n_block in range(1, num_block + 1):
                filepath = [mat_filpath, str(s_filename[n_sname]), '\\', 'sys', '_', str(subjectname[n_sname]), '_',
                                str(n_block), '_fenduan.mat']  # mat文件路径
                filepath = ''.join(filepath)
                data = scipy.io.loadmat(filepath)
                data1 = data['data1'][:, rest_sample:(rest_sample + n_sample), :]
                data2 = data['data2'][:, rest_sample:(rest_sample + n_sample), :]
                Xtrain = np.concatenate((Xtrain, data1, data2), axis=2)
                y = np.hstack((y, y_single))
            Xtrain = Xtrain.transpose(2, 0, 1)

            for n_block in range(1, num_block + 1):
                filepath = [mat_filpath, str(s_filename[n_sname]), '\\', 'xe', '_', str(subjectname[n_sname]), '_',
                                str(n_block), '_fenduan.mat']  # mat文件路径
                filepath = ''.join(filepath)
                data = scipy.io.loadmat(filepath)
                data1 = data['data1'][:, rest_sample:(rest_sample + n_sample), :]
                data2 = data['data2'][:, rest_sample:(rest_sample + n_sample), :]
                Xtest = np.concatenate((Xtest, data1, data2), axis=2)
                y_test = np.hstack((y_test, y_single))
            Xtest = Xtest.transpose(2, 0, 1)

            f_X_train = np.squeeze(Online_filter.transform(Xtrain))
            f_X_test = np.squeeze(Online_filter.transform(Xtest))
            X_mat_train = cov_matrix(f_X_train)  # 求协方差矩阵
            X_mat_test = cov_matrix(f_X_test)  # 求协方差矩阵

            # csp+svm
            csp = CSP()
            csp.fit(Xtrain, y)
            csp_x_train = csp.transform(Xtrain)
            cspsvm = svm.SVC()
            cspsvm.fit(csp_x_train, y)
            cspsvm_res = []
            csp_X_test = csp.transform(Xtest)
            cspsvm_res = cspsvm.predict(csp_X_test)
            csp_single_sub_acc.append(accuracy_score(y_test, cspsvm_res))

            # fbcsp+svm
            fbcsp = FBCSP(n_components=8, n_mutualinfo_components=20, filterbank=filterbank)
            fbcsp.fit(Xtrain, y)
            fbcsp_x_train = fbcsp.transform(Xtrain)
            fbcspsvm = svm.SVC()
            fbcspsvm.fit(fbcsp_x_train, y)
            fbcspsvm_res = []
            fbcsp_X_test = fbcsp.transform(Xtest)
            fbcspsvm_res = fbcspsvm.predict(fbcsp_X_test)
            fbcsp_single_sub_acc.append(accuracy_score(y_test, fbcspsvm_res))

            # eegnet
            eegnet = EEGNet(Chans=64, Samples=4000, kernLength=64,
                            F1=8, D=2, F2=16)
            eegnet.compile(loss='binary_crossentropy', optimizer='adam')
            eegnet.fit(Xtrain, np_utils.to_categorical(y), batch_size=None,
                       epochs=200)
            eegnet_res = eegnet.predict(Xtest).argmax(axis=-1)
            eegnet_single_sub_acc.append(accuracy_score(y_test, eegnet_res))
            #
            # # mdm
            # mdm = RGC.MDM()
            # mdm.fit(X_mat_train, y)
            # mdm_res = mdm.predict(X_mat_test)
            # mdm_single_sub_acc.append(accuracy_score(y_test, mdm_res))
            #
            # # tslda
            # tslda = RGC.TSclassifier(clf=LinearDiscriminantAnalysis())
            # tslda.fit(X_mat_train, y)
            # tslda_res = tslda.predict(X_mat_test)
            # tslda_single_sub_acc.append(accuracy_score(y_test, tslda_res))

            single_sub_acc.append(csp_single_sub_acc)
            single_sub_acc.append(fbcsp_single_sub_acc)
            single_sub_acc.append(eegnet_single_sub_acc)
            # single_sub_acc.append(mdm_single_sub_acc)
            # single_sub_acc.append(tslda_single_sub_acc)
            print('一个人结束')

            all_sub_acc.append(single_sub_acc)

        scipy.io.savemat('acc_all_cross_1.mat', {'acc': all_sub_acc})
        scipy.io.savemat('acc_all_cross.mat', {str(subjectname[0]): all_sub_acc[0], str(subjectname[1]): all_sub_acc[1]
            , str(subjectname[2]): all_sub_acc[2], str(subjectname[3]): all_sub_acc[3]
            , str(subjectname[4]): all_sub_acc[4]
                                               })