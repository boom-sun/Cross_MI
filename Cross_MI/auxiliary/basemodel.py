import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from Cross_MI.auxiliary.classifiers import Classier
from Cross_MI.auxiliary.functions import Split_Sets_random_state
from Cross_MI.alignment.alignment import Alignment
from Cross_MI.auxiliary.auxiliary import Auxiliary
from Cross_MI.auxiliary.model_evaluations import evaluate_model, METRIC_KEYS
from Cross_MI.preprocess.preprocess import Preprocess
from Cross_MI.auxiliary.sun_data_saver import ModelIO
import torch
import random

seed = 123
random.seed(seed)
np.random.seed(seed)


def _empty_metric_dict(algorithms):
    return {alg: {k: [] for k in METRIC_KEYS} for alg in algorithms}


class Basemodel:
    def __init__(self, config):
        self.startpoint = config['starttime'] * config['srate']
        self.endpoint   = config['endtime']   * config['srate']
        self.srate      = config['srate']
        self.dataset    = config['dataset_name']
        self.sence      = config['sence']
        self.dataclass  = config['dataclass']
        self.class_class = config['class_class']
        self.n_class    = config['n_class']
        self.Subject    = config['Subject']
        self.ALignment  = config['ALignment']
        self.Algrithm   = config['Algrithm']
        self.Auxiliary  = config['Auxiliary']
        self.n_train    = config['n_train']
        self.fold       = config['fold']
        self.train_ratio  = config['train_ratio']
        self.test_ratio   = config['test_ratio']
        self.config     = config

        self._standardize_configuration()
        self.model_io = ModelIO(config)

    def _standardize_configuration(self):
        for i in range(len(self.Algrithm)):
            self.Algrithm[i] = self.Algrithm[i].upper()

    # ------------------------------------------------------------------
    # Core classification step – returns (y_pred, y_score)
    # ------------------------------------------------------------------
    def _classification(self, X_train, y_train, X_test, trainclabel, testclabel, algrithm):
        model = Classier(srate=self.srate, Algrithm=algrithm)
        model.fit(X_train, y_train, trainclabel)
        y_pred  = model.predict(X_test)
        y_score = model.predict_proba(X_test)   # None if not available
        return y_pred, y_score

    def Model_Framework(self, algrithm, traindata, trainlabel, testdata, trainclabel, testclabel):
        X_train, X_test   = self._apply_pre(traindata, testdata)
        X_train1, X_test1 = self._apply_auxiliary(X_train, X_test, trainlabel, trainclabel, testclabel)
        X_train2, X_test2 = self._apply_alignment(X_train1, X_test1, trainlabel, trainclabel, testclabel)
        y_pred, y_score   = self._classification(X_train2, trainlabel, X_test2, trainclabel, testclabel, algrithm)
        return y_pred, y_score

    def _apply_pre(self, traindata, testdata):
        model = Preprocess(self.config)
        return model.preprocess(traindata, testdata)

    def _apply_alignment(self, X_train, X_test, y_train, clabel_train=[], clabel_test=[]):
        model = Alignment(srate=self.srate, Alignment=self.ALignment)
        return model.alignment(X_train, X_test, y_train, clabel_train, clabel_test)

    def _apply_auxiliary(self, X_train, X_test, y_train, clabel_train=[], clabel_test=[]):
        model = Auxiliary(srate=self.srate, Auxiliary=self.Auxiliary)
        return model.auxiliary(X_train, X_test, y_train, clabel_train, clabel_test)

    # ------------------------------------------------------------------
    # Single-session: k-fold CV
    # ------------------------------------------------------------------
    def onesence_Class(self, subject, X, y, clabel):
        self.alltrain_acc     = {key: [] for key in self.Algrithm}
        self.alltrain_res     = {key: [] for key in self.Algrithm}
        self.alltrain_metrics = _empty_metric_dict(self.Algrithm)

        for train in range(self.n_train):
            index = np.arange(len(y))
            rng = np.random.default_rng(seed + train)
            rng.shuffle(index)
            X_sh = X[index]
            y_sh = y[index]

            train_index, test_index = Split_Sets_random_state(
                total_fold=self.fold,
                random_state=train,
                data=X_sh,
                labels=y_sh,
            )
            allfold_acc     = {key: [] for key in self.Algrithm}
            allfold_res     = {key: [] for key in self.Algrithm}
            allfold_metrics = _empty_metric_dict(self.Algrithm)

            for algrithm in self.Algrithm:
                for fold in range(self.fold):
                    traindata  = X_sh[train_index[fold]]
                    testdata   = X_sh[test_index[fold]]
                    trainlabel = y_sh[train_index[fold]]
                    testlabel  = y_sh[test_index[fold]]
                    train_clabel = np.array(clabel)[index][train_index[fold]]
                    test_clabel  = np.array(clabel)[index][test_index[fold]]

                    y_pred, y_score = self.Model_Framework(
                        algrithm, traindata, trainlabel, testdata,
                        train_clabel, test_clabel,
                    )
                    m = evaluate_model(testlabel, y_pred, y_score)
                    allfold_acc[algrithm].append(m['acc'])
                    allfold_res[algrithm].append(y_pred)
                    for k in METRIC_KEYS:
                        allfold_metrics[algrithm][k].append(m[k])

                print('准确率为：', np.mean(allfold_acc[algrithm]) * 100, '%')

            for key in self.Algrithm:
                self.alltrain_acc[key].append(allfold_acc[key])
                self.alltrain_res[key].append(allfold_res[key])
                for k in METRIC_KEYS:
                    self.alltrain_metrics[key][k].append(allfold_metrics[key][k])

        return self.alltrain_acc, self.alltrain_res, self.alltrain_metrics

    # ------------------------------------------------------------------
    # Cross-scene / cross-subject (zero-shot)
    # ------------------------------------------------------------------
    def cross_Class1(self, subject, X, y, clabel):
        self.alltrain_acc     = {key: [] for key in self.Algrithm}
        self.alltrain_res     = {key: [] for key in self.Algrithm}
        self.alltrain_metrics = _empty_metric_dict(self.Algrithm)

        data1, data2   = X[0], X[1]
        label1, label2 = y[0], y[1]
        c_label1 = np.array(clabel[0])
        c_label2 = np.array(clabel[1])

        rng = np.random.default_rng(seed)
        idx1 = np.arange(len(label1)); rng.shuffle(idx1)
        idx2 = np.arange(len(label2)); rng.shuffle(idx2)
        traindata,  trainlabel,  train_clabel = data1[idx1], label1[idx1], c_label1[idx1]
        testdata,   testlabel,   test_clabel  = data2[idx2], label2[idx2], c_label2[idx2]

        for algrithm in self.Algrithm:
            y_pred, y_score = self.Model_Framework(
                algrithm, traindata, trainlabel, testdata, train_clabel, test_clabel,
            )
            m = evaluate_model(testlabel, y_pred, y_score)
            self.alltrain_acc[algrithm] = [m['acc']]
            self.alltrain_res[algrithm] = [y_pred]
            for k in METRIC_KEYS:
                self.alltrain_metrics[algrithm][k] = [m[k]]

        print('模型计算成功')
        print('准确率计算成功，为：', m['acc'] * 100, '%')
        return self.alltrain_acc, self.alltrain_res, self.alltrain_metrics

    # ------------------------------------------------------------------
    # Global cross-double: train-on-first-subject, predict on rest
    # ------------------------------------------------------------------
    def _train_and_save_global(self, algrithm, traindata, trainlabel, testdata, trainclabel, testclabel):
        pre = Preprocess(self.config)
        Xtr_pre = pre.transform(traindata)
        Xte_pre = pre.transform(testdata)

        aux = Auxiliary(srate=self.srate, Auxiliary=self.Auxiliary)
        Xtr_aux = aux.fit_transform(Xtr_pre, trainlabel, trainclabel)
        Xte_aux = aux.transform(Xte_pre)

        aln = Alignment(srate=self.srate, Alignment=self.ALignment)
        Xtr_aln = aln.fit_transform(Xtr_aux, trainlabel, trainclabel, testclabel)
        Xte_aln = aln.transform(Xte_aux)

        clf = Classier(srate=self.srate, Algrithm=algrithm)
        clf.fit(Xtr_aln, trainlabel, trainclabel)

        meta = dict(
            srate=self.srate,
            fre_win=self.config.get('fre_win'),
            Auxiliary=self.Auxiliary,
            ALignment=self.ALignment,
            Algrithm=algrithm,
            n_channels=int(Xtr_aln.shape[1]),
            n_samples=int(Xtr_aln.shape[2]),
            n_classes=int(len(np.unique(trainlabel))),
        )

        pipeline = dict(
            meta=meta,
            aux_model=getattr(aux, 'aux_model', None),
            align_method=getattr(aln, 'align_method', None),
        )

        if algrithm.upper() == 'EEGNET':
            weights_path = self.model_io.weights_path(algrithm)
            try:
                clf.model.save_params(f_params=weights_path)
            except Exception:
                try:
                    torch.save(clf.model.module_.state_dict(), weights_path)
                except Exception as e2:
                    raise RuntimeError("EEGNET 权重保存失败，无法序列化该模型。") from e2
            pipeline['serialization'] = 'EEGNET_SKORCH'
            pipeline['weights_path'] = weights_path
        else:
            if hasattr(clf.model, 'state_dict'):
                model_state_dict_path = self.model_io.model_state_dict_path(algrithm)
                torch.save(clf.model.state_dict(), model_state_dict_path)
                pipeline['serialization'] = algrithm
                pipeline['model_state_dict_path'] = model_state_dict_path
            else:
                from joblib import dump
                clf_path = self.model_io.model_path(algrithm)
                dump(clf, clf_path)
                pipeline['serialization'] = 'joblib'
                pipeline['clf_path'] = clf_path

        self.model_io.save(algrithm, pipeline)

        y_pred  = clf.predict(Xte_aln)
        y_score = clf.predict_proba(Xte_aln)
        return y_pred, y_score

    def _predict_with_saved_global(self, algrithm, testdata, testclabel):
        if not self.model_io.exists(algrithm):
            raise RuntimeError(
                f"[{algrithm}] 的已训练流水线未找到，请先在首被试上完成训练保存。"
            )

        pipe = self.model_io.load(algrithm)

        pre = Preprocess(self.config)
        Xte_pre = pre.transform(testdata)

        aux = Auxiliary(srate=self.srate, Auxiliary=self.Auxiliary)
        aux.aux_model = pipe.get('aux_model', None)
        Xte_aux = aux.transform(Xte_pre)

        aln = Alignment(srate=self.srate, Alignment=self.ALignment)
        aln.align_method = pipe.get('align_method', None)
        Xte_aln = aln.transform(Xte_aux)

        if 'clf' in pipe and pipe['clf'] is not None:
            clf = pipe['clf']
            return clf.predict(Xte_aln), clf.predict_proba(Xte_aln)

        meta = pipe['meta']
        if meta['Algrithm'].upper() == 'EEGNET':
            from MetaBCI.metabci.brainda.algorithms.deep_learning.eegnet import EEGNet
            net = EEGNet(
                n_channels=meta['n_channels'],
                n_samples=meta['n_samples'],
                n_classes=meta['n_classes'],
            )
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            net.set_params(device=device)
            net.initialize()
            try:
                net.load_params(f_params=pipe['weights_path'])
            except Exception as e:
                raise RuntimeError(
                    "EEGNET 权重加载失败，请检查 weights.pt 是否存在且与当前代码版本匹配。"
                ) from e
            clf = Classier(srate=self.srate, Algrithm='EEGNET')
            clf.model = net
            return clf.predict(Xte_aln), clf.predict_proba(Xte_aln)

        raise RuntimeError(f"未找到可用的分类器或权重（算法={meta['Algrithm']}）。")

    def global_cross_Class(self, subject, X, y, clabel):
        self.alltrain_acc     = {key: [] for key in self.Algrithm}
        self.alltrain_res     = {key: [] for key in self.Algrithm}
        self.alltrain_metrics = _empty_metric_dict(self.Algrithm)

        is_first = (subject == self.config['Subject_choose'][0])

        if is_first:
            data1, data2   = X[0], X[1]
            label1, label2 = y[0], y[1]
            c_label1 = np.array(clabel[0])
            c_label2 = np.array(clabel[1])

            rng = np.random.default_rng(seed)
            idx1 = np.arange(len(label1)); rng.shuffle(idx1)
            idx2 = np.arange(len(label2)); rng.shuffle(idx2)
            traindata,  trainlabel,  train_clabel = data1[idx1], label1[idx1], c_label1[idx1]
            testdata,   testlabel,   test_clabel  = data2[idx2], label2[idx2], c_label2[idx2]

            for alg in self.Algrithm:
                y_pred, y_score = self._train_and_save_global(
                    alg, traindata, trainlabel, testdata, train_clabel, test_clabel,
                )
                m = evaluate_model(testlabel, y_pred, y_score)
                self.alltrain_acc[alg] = [m['acc']]
                self.alltrain_res[alg] = [y_pred]
                for k in METRIC_KEYS:
                    self.alltrain_metrics[alg][k] = [m[k]]
        else:
            data2  = X[0]
            label2 = y[0]
            c_label2 = np.array(clabel[0])

            rng = np.random.default_rng(seed)
            idx2 = np.arange(len(label2)); rng.shuffle(idx2)
            testdata, testlabel, test_clabel = data2[idx2], label2[idx2], c_label2[idx2]

            for alg in self.Algrithm:
                y_pred, y_score = self._predict_with_saved_global(alg, testdata, test_clabel)
                m = evaluate_model(testlabel, y_pred, y_score)
                self.alltrain_acc[alg] = [m['acc']]
                self.alltrain_res[alg] = [y_pred]
                for k in METRIC_KEYS:
                    self.alltrain_metrics[alg][k] = [m[k]]

        print("Sub_%2d 's calculation is complete!" % (subject))
        return self.alltrain_acc, self.alltrain_res, self.alltrain_metrics

    # ------------------------------------------------------------------
    # Cross with supervised target-domain samples
    # ------------------------------------------------------------------
    def cross_Class2(self, subject, X, y, clabel):
        data1, data2   = X[0], X[1]
        label1, label2 = y[0], y[1]
        c_label1 = np.array(clabel[0])
        c_label2 = np.array(clabel[1])

        for train_ratio in self.train_ratio1:
            allfold_acc     = {key: [] for key in self.Algrithm}
            allfold_res     = {key: [] for key in self.Algrithm}
            allfold_metrics = _empty_metric_dict(self.Algrithm)

            sss = StratifiedShuffleSplit(
                n_splits=self.n_train,
                test_size=self.test_ratio,
                train_size=train_ratio,
                random_state=seed,
            )
            for algrithm in self.Algrithm:
                for train_index, test_index in sss.split(data2, label2):
                    traindata1, testdata   = data2[train_index], data2[test_index]
                    trainlabel1, testlabel = label2[train_index], label2[test_index]
                    train_clabel1, test_clabel = c_label2[train_index], c_label2[test_index]
                    traindata  = np.concatenate((data1,  traindata1),  axis=0)
                    trainlabel = np.concatenate((label1, trainlabel1), axis=0)
                    train_clabel = np.concatenate((c_label1, train_clabel1), axis=0)

                    y_pred, y_score = self.Model_Framework(
                        algrithm, traindata, trainlabel, testdata,
                        train_clabel, test_clabel,
                    )
                    m = evaluate_model(testlabel, y_pred, y_score)
                    allfold_acc[algrithm].append(m['acc'])
                    allfold_res[algrithm].append(y_pred)
                    for k in METRIC_KEYS:
                        allfold_metrics[algrithm][k].append(m[k])

            print("Sub_%2d's train_ratio_%1f calculation is complete!" % (subject, train_ratio))
            for key in self.Algrithm:
                self.alltrain_acc[key].append(allfold_acc[key])
                self.alltrain_res[key].append(allfold_res[key])
                for k in METRIC_KEYS:
                    self.alltrain_metrics[key][k].append(allfold_metrics[key][k])

        return self.alltrain_acc, self.alltrain_res, self.alltrain_metrics

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def classier(self, subject, X, y, clabel):
        if self.sence in ['S1', 'S2']:
            self.onesence_Class(subject, X, y, clabel)

        if self.sence in [
            'cross1', 'cross2', 'cross_task1', 'cross_task2',
            'cross_subject1', 'cross_subject2',
            'cross_double1', 'cross_double2',
            'S1_online', 'S2_online',
        ]:
            if self.train_ratio[0] == 0:
                self.cross_Class1(subject, X, y, clabel)
                self.train_ratio1 = self.train_ratio[1:]
            else:
                self.alltrain_acc     = {key: [] for key in self.Algrithm}
                self.alltrain_res     = {key: [] for key in self.Algrithm}
                self.alltrain_metrics = _empty_metric_dict(self.Algrithm)
                self.train_ratio1 = self.train_ratio

            # Fix: was `> 1`; should be `>= 1` so a single remaining ratio is processed
            if len(self.train_ratio1) >= 1:
                self.cross_Class2(subject, X, y, clabel)

        if self.sence in ['global_cross_double1', 'global_cross_double2']:
            self.global_cross_Class(subject, X, y, clabel)

        return self.alltrain_acc, self.alltrain_res, self.alltrain_metrics
