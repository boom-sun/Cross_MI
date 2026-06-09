import numpy as np
from MetaBCI.metabci.brainda.algorithms.decomposition.csp import CSP
from MetaBCI.metabci.brainda.algorithms.decomposition.csp import FBCSP
import scipy
from sklearn import svm
from sklearn.metrics import accuracy_score
from MetaBCI.metabci.brainda.algorithms.decomposition.base import generate_filterbank
import h5py


cross = 1   # 1为非跨场景，2为跨场景
choose = 0  # 1为五折交叉，0为依次增加选择训练数据，
acc_cspsvm = np.empty((5, 5), int)
acc_fbcspsvm = np.empty((5, 5), int)
acc_single = np.array([])
fbcsp_acc_single = np.empty((5), int)
s_filename = ['杨涵', '姬智敏', '姜传力', '辛海玲', '杨明明','孙新维','姜传力_5_30','潘林聪']
sencename = ['sys', 'xe']
subjectname = ['yh', 'jzm', 'jcl', 'xhl', 'ymm','sxw', 'jcl','plc']
wp = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30), (5, 30)]
ws = [(2, 10), (6, 14), (10, 18), (14, 22), (18, 26), (22, 30)]
filterbank = generate_filterbank(wp, ws, srate=250)

if choose == 0:
    for n_sname in [7]:
        for n_train in range(1, 6):
            Xtrain = np.empty((64, 4000, 0), int)
            Xtest = np.empty((64, 4000, 0), int)
            y = np.empty((0), int)
            y_test = np.empty((0), int)
            # y_single = np.concatenate((np.ones(12), np.ones(12) * 2, np.ones(12) * 3, np.ones(12) * 4), axis=0)
            y_single = np.concatenate((np.ones(15), np.ones(15) * 2), axis=0)
            for n_block in range(1, 7):
                filepath = ['E:\跨场景实验\预实验数据\分段后数据\\', str(s_filename[n_sname]), '\\', 'sys', '_', str(subjectname[n_sname]), '_', str(n_block), '_fenduan.mat']  # mat文件路径
                filepath = ''.join(filepath)
                data = scipy.io.loadmat(filepath)
                data1 = data['data1'][:, 2000:6000, :]
                data2 = data['data2'][:, 2000:6000, :]
                if n_block in range(1, n_train+1):
                    Xtrain = np.concatenate((Xtrain, data1, data2), axis=2)
                    y = np.hstack((y, y_single))
                else:
                    Xtest = np.concatenate((Xtest, data1, data2), axis=2)
                    y_test = np.hstack((y_test, y_single))
                # data3 = data['data3'][:, 2000:6000, :]
                # data4 = data['data4'][:, 2000:6000, :]
                # if n_block == 1:
                #     Xtrain = np.concatenate((Xtrain, data1, data2, data3, data4), axis=2)
                #     y = np.hstack((y, y_single))
                # else:
                #     Xtest = np.concatenate((Xtest, data1, data2, data3, data4), axis=2)
            Xtrain = Xtrain.transpose(2, 0, 1)
            Xtest = Xtest.transpose(2, 0, 1)

            # IndXtrain=np.size(Xtrain, 2)/5
            # np.random.randint



            # # csp+svm
            # csp = CSP()
            # csp.fit(Xtrain, y)
            # csp_x_train = csp.transform(Xtrain)
            # cspsvm = svm.SVC()
            # cspsvm.fit(csp_x_train, y)
            # cspsvm_res = []
            # csp_X_test = csp.transform(Xtest)
            # cspsvm_res = cspsvm.predict(csp_X_test)
            # acc_single.append(accuracy_score(y_test, cspsvm_res))

            # fbcsp+svm
            fbcsp = FBCSP(n_components=8, n_mutualinfo_components=20, filterbank=filterbank)
            fbcsp.fit(Xtrain, y)
            fbcsp_x_train = fbcsp.transform(Xtrain)
            fbcspsvm = svm.SVC()
            fbcspsvm.fit(fbcsp_x_train, y)
            fbcspsvm_res = []
            fbcsp_X_test = fbcsp.transform(Xtest)
            fbcspsvm_res = fbcspsvm.predict(fbcsp_X_test)

            # np.append(fbcsp_acc_single, values=accuracy_score(y_test, fbcspsvm_res))
            # fbcsp_acc_single[n_train-1] = acc
            print('测试集准确率：', accuracy_score(y_test, fbcspsvm_res))
        print('一个人结束')
        # fbcsp_acc_single = np.array(fbcsp_acc_single)
        # # acc_cspsvm[int(n_sname), :] = acc_single
        # acc_fbcspsvm[n_sname, :] = fbcsp_acc_single
        # fbcsp_acc_single = np.empty((5), int)
        # print(acc_fbcspsvm)
    # scipy.io.savemat('acc_fbcspsvm.mat', {str(subjectname[1]):acc_cspsvm[1], str(subjectname[2]):acc_cspsvm[2], str(subjectname[3]):acc_cspsvm[3], str(subjectname[4]):acc_cspsvm[4], str(subjectname[5]):acc_cspsvm[5]})
    # scipy.io.savemat('acc_fbcspsvm.mat', fbcsp_acc_single)

if choose == 1:
    for n_sname in [5]:
            Xall = np.empty((64, 4000, 0), int)
            yall = np.empty((0), int)
            # y_single = np.concatenate((np.ones(12), np.ones(12) * 2, np.ones(12) * 3, np.ones(12) * 4), axis=0)
            y_single = np.concatenate((np.ones(12), np.ones(12) * 2), axis=0)
            for n_block in range(2, 11, 2):
                filepath = ['E:\跨场景实验\预实验数据\分段后数据\\', str(s_filename[n_sname]), '\\', 'sys', '_', str(subjectname[n_sname]), '_', str(n_block), '_fenduan.mat']  # mat文件路径
                filepath = ''.join(filepath)
                data = scipy.io.loadmat(filepath)
                data1 = data['data1'][:, 2000:6000, :]
                data2 = data['data2'][:, 2000:6000, :]
                Xall = np.concatenate((Xall, data1, data2), axis=2)
                yall = np.hstack((yall, y_single))

            Xall = Xall.transpose(2, 0, 1)

            IndX = np.size(Xall, 0)
            Ind = np.random.permutation(IndX)
            Xall = Xall[Ind, :, :]
            yall = yall[Ind]
            for n_train in range(1, 6):
                Ind_start = int((IndX/5+1)*(n_train-1))
                Ind_end = int((IndX/5+1)*n_train)
                # Ind_temp = np.arange(Ind_start, Ind_end)
                Xtest = Xall[Ind_start:Ind_end, :, :]
                y_test = yall[Ind_start:Ind_end]
                Ind_test = [int(i) for i in list(range(0, Ind_start))+list(range(Ind_end, IndX))]
                Xtrain = Xall[Ind_test, :, :]
                y = yall[Ind_test]


                # # csp+svm
                # csp = CSP()
                # csp.fit(Xtrain, y)
                # csp_x_train = csp.transform(Xtrain)
                # cspsvm = svm.SVC()
                # cspsvm.fit(csp_x_train, y)
                # cspsvm_res = []
                # csp_X_test = csp.transform(Xtest)
                # cspsvm_res = cspsvm.predict(csp_X_test)
                # acc_single.append(accuracy_score(y_test, cspsvm_res))

                # fbcsp+svm
                fbcsp = FBCSP(n_components=8, n_mutualinfo_components=20, filterbank=filterbank)
                fbcsp.fit(Xtrain, y)
                fbcsp_x_train = fbcsp.transform(Xtrain)
                fbcspsvm = svm.SVC()
                fbcspsvm.fit(fbcsp_x_train, y)
                fbcspsvm_res = []
                fbcsp_X_test = fbcsp.transform(Xtest)
                fbcspsvm_res = fbcspsvm.predict(fbcsp_X_test)

                # np.append(fbcsp_acc_single, values=accuracy_score(y_test, fbcspsvm_res))
                # fbcsp_acc_single[n_train-1] = acc
                print('测试集准确率：', accuracy_score(y_test, fbcspsvm_res))

        # fbcsp_acc_single = np.array(fbcsp_acc_single)
        # # acc_cspsvm[int(n_sname), :] = acc_single
        # acc_fbcspsvm[n_sname, :] = fbcsp_acc_single
        # fbcsp_acc_single = np.empty((5), int)
        # print(acc_fbcspsvm)
    # scipy.io.savemat('acc_fbcspsvm.mat', {str(subjectname[1]):acc_cspsvm[1], str(subjectname[2]):acc_cspsvm[2], str(subjectname[3]):acc_cspsvm[3], str(subjectname[4]):acc_cspsvm[4], str(subjectname[5]):acc_cspsvm[5]})
    # scipy.io.savemat('acc_fbcspsvm.mat', fbcsp_acc_single)


if choose == 2:
    filepath = ['E:\跨场景实验\RIGEL-main_new\dataset1\\S08D1.mat']  # mat文件路径
    filepath = ''.join(filepath)
    data = scipy.io.loadmat(filepath)
    # data = h5py.File(filepath, 'r')
    # data = scipy.io.loadmat(filepath)
    Xall = data['data'][:, 500:1500, :]
    yall = np.array(data['label'])
    yall = yall.T


    IndX = np.size(Xall, 2)
    for n_train in range(1, 6):
                Ind_start = int((IndX/5+1)*(n_train-1))
                Ind_end = int((IndX/5+1)*n_train)
                # Ind_temp = np.arange(Ind_start, Ind_end)
                Xtest = Xall[Ind_start:Ind_end, :, :]
                y_test = yall[Ind_start:Ind_end]
                Ind_test = [int(i) for i in list(range(0, Ind_start))+list(range(Ind_end, IndX))]
                Xtrain = Xall[Ind_test, :, :]
                y = yall[Ind_test]


                # # csp+svm
                # csp = CSP()
                # csp.fit(Xtrain, y)
                # csp_x_train = csp.transform(Xtrain)
                # cspsvm = svm.SVC()
                # cspsvm.fit(csp_x_train, y)
                # cspsvm_res = []
                # csp_X_test = csp.transform(Xtest)
                # cspsvm_res = cspsvm.predict(csp_X_test)
                # acc_single.append(accuracy_score(y_test, cspsvm_res))

                # fbcsp+svm
                fbcsp = FBCSP(n_components=8, n_mutualinfo_components=20, filterbank=filterbank)
                fbcsp.fit(Xtrain, y)
                fbcsp_x_train = fbcsp.transform(Xtrain)
                fbcspsvm = svm.SVC()
                fbcspsvm.fit(fbcsp_x_train, y)
                fbcspsvm_res = []
                fbcsp_X_test = fbcsp.transform(Xtest)
                fbcspsvm_res = fbcspsvm.predict(fbcsp_X_test)

                # np.append(fbcsp_acc_single, values=accuracy_score(y_test, fbcspsvm_res))
                # fbcsp_acc_single[n_train-1] = acc
                print('测试集准确率：', accuracy_score(y_test, fbcspsvm_res))

        # fbcsp_acc_single = np.array(fbcsp_acc_single)
        # # acc_cspsvm[int(n_sname), :] = acc_single
        # acc_fbcspsvm[n_sname, :] = fbcsp_acc_single
        # fbcsp_acc_single = np.empty((5), int)
        # print(acc_fbcspsvm)
    # scipy.io.savemat('acc_fbcspsvm.mat', {str(subjectname[1]):acc_cspsvm[1], str(subjectname[2]):acc_cspsvm[2], str(subjectname[3]):acc_cspsvm[3], str(subjectname[4]):acc_cspsvm[4], str(subjectname[5]):acc_cspsvm[5]})
    # scipy.io.savemat('acc_fbcspsvm.mat', fbcsp_acc_single)