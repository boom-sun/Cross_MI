from Cross_MI.auxiliary.sun_data_loader import Dataloader
from Cross_MI.auxiliary.sun_data_saver import Saver
import os
from pathlib import Path
import yaml
from argparse import ArgumentParser
from Cross_MI.auxiliary.basemodel import Basemodel
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from MetaBCI.brainda.algorithms.decomposition.csp import csp_feature
from MetaBCI.metabci.brainda.algorithms.pyriemann.utils import mean_covariance
from Cross_MI.preprocess.preprocess import BandpassFilter
from sklearn.pipeline import make_pipeline
from MetaBCI.metabci.brainda.algorithms.decomposition.csp import CSP
from pyriemann.utils.covariance import covariances
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "Cross_MI\\configs")
DEFAULT_CONFIG = "ssmvep.yaml"
Subject = list(range(1, 22+1))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    with open(os.path.join(CONFIG_DIR, args.config), 'rb') as f:
        config = yaml.safe_load(f)

    loader = Dataloader(config)
    model = Basemodel(config)
    saver = Saver(config, Subject)
    lambda_diff = 1
    Acc_all = []
    for subject in Subject:
        data, label, sub_label = loader.loader_data(subject)
        bandpassfilter = BandpassFilter(fs=config['srate'], lowcut=config['fre_win'][0], highcut=config['fre_win'][1])
        bandpassfilter.fit()
        X=bandpassfilter.transform(data)
        y=label
        # X=X[:,14:40,:]
        Acc=[]
        for fold in range(10):
            traindata, testdata, trainlabel, testlabel = (
                train_test_split(X, y, test_size=0.1, stratify=y, random_state=fold))

            ## baseline——58.4%  运动区63.36%
            C_train = covariances(traindata[(trainlabel==0)|(trainlabel==1)], estimator='lwf')
            label_train = trainlabel[(trainlabel==0)|(trainlabel==1)]
            model_csp = CSP(n_components=4)
            model_csp.fit(C_train, label_train)
            csp_train_feature = model_csp.transform(C_train)
            model_svm = SVC()
            model_svm.fit(csp_train_feature, label_train)

            X_test = testdata[(testlabel == 0) | (testlabel == 1)]
            C_test = covariances(X_test, estimator='lwf')
            label_test = testlabel[(testlabel == 0) | (testlabel == 1)]
            csp_test_feature = model_csp.transform(C_test)
            y_pred = model_svm.predict(csp_test_feature)
            acc = accuracy_score(label_test, y_pred)
            Acc.append(acc)

            # ## v1——52.36%  空域去平均
            # C_train = covariances(traindata, estimator='lwf')
            # C_train_l = C_train[trainlabel==0]
            # C_train_r = C_train[trainlabel == 1]
            # C_mean_l = mean_covariance(C_train[trainlabel==2], metric='euclid')
            # C_mean_r = mean_covariance(C_train[trainlabel == 3], metric='euclid')
            # C_diff_l = C_train_l- lambda_diff*np.repeat(C_mean_l[np.newaxis, :, :],
            #                                            repeats=len(C_train[trainlabel==2]), axis=0)
            # C_diff_r = C_train_r - lambda_diff*np.repeat(C_mean_r[np.newaxis, :, :],
            #                                            repeats=len(C_train[trainlabel==3]), axis=0)
            # C_diff_train = np.concatenate((C_diff_l, C_diff_r),axis=0)
            # label_train = np.concatenate((np.ones(np.shape(trainlabel[trainlabel==0])),
            #                               np.ones(np.shape(trainlabel[trainlabel==1]))*2))
            # model_csp = CSP(n_components=4)
            # model_csp.fit(C_diff_train, label_train)
            # csp_train_feature = model_csp.transform(C_diff_train)
            # model_svm = SVC()
            # model_svm.fit(csp_train_feature, label_train)
            #
            # C_test = covariances(testdata[(testlabel==0)|(testlabel==1)],estimator='lwf')
            # C_diff_test = C_test- np.repeat(C_mean_l[np.newaxis, :, :],
            #                                            repeats=len(C_test), axis=0)
            # label_test = testlabel[(testlabel==0)|(testlabel==1)]+1
            # csp_test_feature = model_csp.transform(C_diff_test)
            # y_pred = model_svm.predict(csp_test_feature)
            # acc = accuracy_score(label_test, y_pred)
            # Acc.append(acc)

            # ## v2——59.68%  运动区60.86%
            # mean_l = np.mean(traindata[trainlabel == 2], axis=0)
            # mean_r = np.mean(traindata[trainlabel == 3], axis=0)
            # X_train_l = traindata[trainlabel == 0] - np.repeat(mean_l[np.newaxis, :, :],
            #                                                    repeats=len(traindata[trainlabel == 2]), axis=0)
            # X_train_r = traindata[trainlabel == 1] - np.repeat(mean_l[np.newaxis, :, :],
            #                                                    repeats=len(traindata[trainlabel == 3]), axis=0)
            # X_train = np.concatenate((X_train_l, X_train_r), axis=0)
            # C_train = covariances(X_train, estimator='lwf')
            # label_train = np.concatenate((np.ones(np.shape(trainlabel[trainlabel == 0])),
            #                               np.ones(np.shape(trainlabel[trainlabel == 1])) * 2))
            # model_csp = CSP(n_components=4)
            # model_csp.fit(C_train, label_train)
            # csp_train_feature = model_csp.transform(C_train)
            # model_svm = SVC()
            # model_svm.fit(csp_train_feature, label_train)
            #
            # X_test = testdata[(testlabel == 0) | (testlabel == 1)] - np.repeat(mean_l[np.newaxis, :, :],
            #                                                                    repeats=len(testdata[(testlabel == 0) | (
            #                                                                                testlabel == 1)]), axis=0)
            # C_test = covariances(X_test, estimator='lwf')
            # label_test = testlabel[(testlabel == 0) | (testlabel == 1)] + 1
            # csp_test_feature = model_csp.transform(C_test)
            # y_pred = model_svm.predict(csp_test_feature)
            # acc = accuracy_score(label_test, y_pred)
            # Acc.append(acc)

            ## v3——58.4%  运动区60.6%
            mean_l = np.mean(np.mean(traindata[trainlabel == 2, 40:60,:], axis=0),axis=0)
            mean_r = np.mean(np.mean(traindata[trainlabel == 3, 40:60,:], axis=0),axis=0)
            X_train_l = traindata[trainlabel == 0]-np.tile(mean_l,
                                    (len(traindata[trainlabel == 2]), np.shape(traindata)[1],1))
            X_train_r = traindata[trainlabel == 1] - np.tile(mean_r,
                                    (len(traindata[trainlabel == 2]), np.shape(traindata)[1],1))
            X_train = np.concatenate((X_train_l, X_train_r), axis=0)[:,14:40,:]
            C_train = covariances(X_train, estimator='lwf')
            label_train = np.concatenate((np.ones(np.shape(trainlabel[trainlabel == 0])),
                                          np.ones(np.shape(trainlabel[trainlabel == 1])) * 2))
            model_csp = CSP(n_components=4)
            model_csp.fit(C_train, label_train)
            csp_train_feature = model_csp.transform(C_train)
            model_svm = SVC()
            model_svm.fit(csp_train_feature, label_train)

            X_test = (testdata[(testlabel == 0) | (testlabel == 1)]- np.tile(mean_l,
                    (len(testdata[(testlabel == 0) | (testlabel == 1)]), np.shape(testdata)[1],1)))[:,14:40,:]
            C_test = covariances(X_test, estimator='lwf')
            label_test = testlabel[(testlabel == 0) | (testlabel == 1)] + 1
            csp_test_feature = model_csp.transform(C_test)
            y_pred = model_svm.predict(csp_test_feature)
            acc = accuracy_score(label_test, y_pred)
            Acc.append(acc)

        print('sub',subject,' acc: ', np.mean(Acc))
        Acc_all.append(np.mean(Acc))
    print(np.mean(Acc_all))





