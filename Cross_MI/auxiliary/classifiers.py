import numpy as np
from MetaBCI.metabci.brainda.algorithms.deep_learning.eegnet import EEGNet, EEGNetManager
from skorch import NeuralNetClassifier
from MetaBCI.metabci.brainda.algorithms.deep_learning.base import NeuralNetClassifierNoLog
from MetaBCI.metabci.brainda.algorithms.deep_learning.shallownet import ShallowNet
from sklearn.pipeline import make_pipeline
from MetaBCI.metabci.brainda.algorithms.pyriemann.spatialfilters import CSP
from MetaBCI.metabci.brainda.algorithms.decomposition import FBCSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from Cross_MI.alignment.PTLDA import PTLDA
from pyriemann.classification import MDM, FgMDM
from pyriemann.classification import TSClassifier
from pyriemann.utils.covariance import covariances
from MetaBCI.metabci.brainda.algorithms.decomposition.base import generate_filterbank
from MetaBCI.metabci.brainda.algorithms.decomposition import DCPM
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from Cross_MI.models.atcnet import ATCNet_
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from braindecode.models.atcnet import ATCNet
from braindecode.models.eegconformer import EEGConformer
from braindecode.models.eeginception_mi import EEGInceptionMI
from Cross_MI.channel_utils import pick_mi_21
from Cross_MI.models.eeg_gnn_21 import EEGGCN21

import random
seed = 123
random.seed(seed)

import contextlib
import io

@contextlib.contextmanager
def suppress_stdout():
    with io.StringIO() as fake_stdout:
        with contextlib.redirect_stdout(fake_stdout):
            yield

class Classier():
    def __init__(self, srate, Algrithm):
            """Init."""
            self.srate = srate
            self.Algrithm = Algrithm
    def fit(self, X, y, clabel=[]):
        '''
        :param X: 训练集数据
        :param y: 训练集标签
        :return:
        '''
        X_cov = covariances(X,estimator='lwf')
        n_classes = len(np.unique(y))
        if 'CSPSVM' == self.Algrithm:
            self.model = make_pipeline(CSP(nfilter=8), SVC(probability=True))
            self.model.fit(X_cov, y)
        elif 'FBCSP' == self.Algrithm:
            wp = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (4, 12), (12, 20), (20, 28)]
            ws = [(2, 10), (6, 14), (10, 18), (14, 22), (18, 26), (22, 30), (26, 34), (2, 14), (10, 22), (18, 30)]
            filterbank = generate_filterbank(wp, ws, srate=self.srate, order=4, rp=0.5)
            self.model = make_pipeline(*[
                FBCSP(n_components=8, n_mutualinfo_components=10, filterbank=filterbank),
                SVC(probability=True)])
            self.model.fit(X, y)
        elif 'FBCSPLDA' == self.Algrithm:
            wp = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (4, 12), (12, 20), (20, 28)]
            ws = [(2, 10), (6, 14), (10, 18), (14, 22), (18, 26), (22, 30), (26, 34), (2, 14), (10, 22), (18, 30)]
            filterbank = generate_filterbank(wp, ws, srate=self.srate, order=4, rp=0.5)
            self.model = make_pipeline(*[
                FBCSP(n_components=8, n_mutualinfo_components=10, filterbank=filterbank),
                LDA(solver='eigen', shrinkage='auto')])
            self.model.fit(X, y)
        elif 'CSPLDA' == self.Algrithm:
            self.model = make_pipeline(CSP(nfilter=8), LDA(solver='eigen', shrinkage='auto'))
            self.model.fit(X_cov, y)
        elif 'TSLDA' == self.Algrithm:
            self.model = TSClassifier()
            self.model.fit(X_cov,y)
        elif 'PTLDA' == self.Algrithm:
            self.model=PTLDA()
            self.model.fit(X,y,clabel)
        elif 'DCPM' == self.Algrithm:
            self.model = DCPM()
            self.model.fit(X_cov, y)
        elif 'MDRM' == self.Algrithm:
            self.model = MDM()
            self.model.fit(X_cov, y)
        elif 'FGMDRM' == self.Algrithm:
            self.model = FgMDM()
            self.model.fit(X_cov, y)
        elif 'EEGNET' == self.Algrithm:
            n_classes = len(np.unique(y))
            X_, y_ = map(torch.tensor, (X.copy(), y.copy()))
            del X, y
            with suppress_stdout():
                self.model = EEGNet(
                    n_channels=X_.shape[1],
                    n_samples=X_.shape[2],
                    n_classes=n_classes
                )
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model.set_params(device=device)
                self.model.fit(X_, y_.long())
            del X_, y_
        elif 'SHALLOWNET' == self.Algrithm:
            n_classes = len(np.unique(y))
            X_, y_ = map(torch.tensor, (X.copy(), y.copy()))
            del X, y
            self.model = ShallowNet(n_channels=X_.shape[1], n_samples=X_.shape[2],
                                    n_classes=n_classes)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.set_params(device=device)
            self.model.fit(X_, y_.long())
            del X_, y_
        # elif 'ATCNET' == self.Algrithm:
        #     num_samples, num_channels, num_timepoints = X.shape
        #     num_classes = len(np.unique(y))  # 计算类别数量（假设标签从 0 开始连续）
        #     y_onehot = to_categorical(y, num_classes)
        #     X = X.reshape((num_samples, 1, num_channels, num_timepoints))
        #     train_conf = {'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': 0.001,
        #                   'LearnCurves': True, 'n_train': 10, 'model': 'ATCNet'}
        #     self.model = ATCNet_(
        #         # Dataset parameters
        #         n_classes=num_classes,
        #         in_chans=num_channels,
        #         in_samples=num_timepoints,
        #         # Sliding window (SW) parameter
        #         n_windows=5,
        #         # Attention (AT) block parameter
        #         attention='mha', # Options: None, 'mha','mhla', 'cbam', 'se'
        #         # Convolutional (CV) block parameters
        #         eegn_F1 = 16,
        #         eegn_D = 2,
        #         eegn_kernelSize = 64,
        #         eegn_poolSize = 7,
        #         eegn_dropout = 0.3,
        #         # Temporal convolutional (TC) block parameters
        #         tcn_depth = 2,
        #         tcn_kernelSize = 4,
        #         tcn_filters = 32,
        #         tcn_dropout = 0.3,
        #         tcn_activation='elu'
        #     )
        #     self.model.compile(loss=categorical_crossentropy,
        #                        optimizer=Adam(learning_rate=train_conf.get('lr')),
        #                        metrics=['accuracy'])
        #     self.model.fit(X, y_onehot, epochs=train_conf.get('epochs'),
        #                    batch_size=train_conf.get('batch_size'),
        #                    validation_split=0.2,  # 从训练集中拆分出 20% 用于验证
        #                    verbose=0)  # 设为 1 以显示训练进度条
        elif 'ATCNET' == self.Algrithm:
            X_ = X.astype(np.float32)
            self._le = LabelEncoder()
            y_enc = self._le.fit_transform(y).astype(np.int64)
            del X, y
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            net = NeuralNetClassifier(
                module=ATCNet,
                module__n_chans=X_.shape[1],
                module__n_outputs=n_classes,
                module__sfreq=self.srate,
                module__input_window_seconds=(X_.shape[2] / self.srate),
                max_epochs=200,
                lr=1e-3,
                batch_size=128,
                iterator_train__shuffle=True,
                optimizer=torch.optim.Adam,
                criterion=nn.CrossEntropyLoss,
                callbacks=[EarlyStopping(patience=50)],
                device=device
            )
            net.fit(X_, y_enc)
            self.model = net
            del X_, y_enc
        elif 'EEGCONFORMER' == self.Algrithm:
            X_ = X.astype(np.float32)
            self._le = LabelEncoder()
            y_enc = self._le.fit_transform(y).astype(np.int64)
            del X, y
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            net = NeuralNetClassifier(
                module=EEGConformer,
                module__n_chans=X_.shape[1],
                module__n_outputs=n_classes,
                module__sfreq=self.srate,
                module__input_window_seconds=(X_.shape[2] / self.srate),
                max_epochs=200,
                lr=1e-3,
                batch_size=128,
                iterator_train__shuffle=True,
                optimizer=torch.optim.Adam,
                criterion=nn.CrossEntropyLoss,
                callbacks=[EarlyStopping(patience=50)],
                device=device
            )
            net.fit(X_, y_enc)
            self.model = net
            del X_, y_enc

        elif 'SHALLOWFBCSP' == self.Algrithm:
            X_ = X.astype(np.float32)
            self._le = LabelEncoder()
            y_enc = self._le.fit_transform(y).astype(np.int64)
            del X, y
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            net = NeuralNetClassifier(
                module=ShallowFBCSPNet,
                module__n_chans=X_.shape[1],
                module__n_outputs=n_classes,
                module__sfreq=self.srate,
                module__input_window_seconds=(X_.shape[2] / self.srate),
                module__final_conv_length='auto',
                max_epochs=200,
                lr=1e-2,
                batch_size=128,
                iterator_train__shuffle=True,
                optimizer=torch.optim.Adam,
                criterion=nn.CrossEntropyLoss,
                callbacks=[EarlyStopping(patience=50)],
                device=device
            )
            net.fit(X_, y_enc)
            self.model = net
            del X_, y_enc
        elif 'EEGINCEPTIONMI' == self.Algrithm:
            X_ = X.astype(np.float32)
            self._le = LabelEncoder()
            y_enc = self._le.fit_transform(y).astype(np.int64)
            del X, y
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            net = NeuralNetClassifier(
                module=EEGInceptionMI,
                module__n_chans=X_.shape[1],
                module__n_outputs=n_classes,
                module__sfreq=self.srate,
                module__input_window_seconds=(X_.shape[2] / self.srate),
                max_epochs=200,
                lr=1e-2,
                batch_size=128,
                iterator_train__shuffle=True,
                optimizer=torch.optim.Adam,
                criterion=nn.CrossEntropyLoss,
                callbacks=[EarlyStopping(patience=50)],
                device=device
            )
            net.fit(X_, y_enc)
            self.model = net
            del X_, y_enc
        if self.Algrithm == 'EEGGCN_21':
            loc_path = "channel_location_60_neuroscan.locs"
            chan_idx21, coords21, labels21 = pick_mi_21(loc_path)

            X_sel = X[:, chan_idx21, :]        # [N, 21, T]
            n_time = X.shape[2]
            n_classes = int(len(np.unique(y)))

            self.model = EEGGCN21(
                n_time=n_time,
                n_classes=n_classes,
                adj_type="hybrid",      # 你可以试 'struct' 或 'hybrid'
                k_struct=4,
                k_func=4,
                alpha=0.5,
                gnn_type="gcn",        # 先用 gcn，如果想再加强就改 'gat'
                gnn_hidden=64,
                gnn_layers=2,
                temporal_D=32,
                lr=1e-3,
                weight_decay=1e-4,
                epochs=100,
                batch_size=64,
                laplacian_lambda=1e-4, # 建议先 1e-4 或 0，看表现再调
                supcon_lambda=0.05,
                center_lambda=0.01,
            )
            self._mi21_idx = chan_idx21
            self.model.fit(X_sel, y, coords21,
                           clabel=np.array(clabel) if len(clabel) else None)
            return

    def predict(self,X):
        X_cov=covariances(X,estimator='lwf')
        if ('CSPSVM' == self.Algrithm) or ('CSPLDA'== self.Algrithm) or ('TSLDA' == self.Algrithm) or \
                ('DCPM' == self.Algrithm) or ('MDRM' == self.Algrithm) or ('FGMDRM' == self.Algrithm):
            y_pred = self.model.predict(X_cov)
        elif ('FBCSP' == self.Algrithm) or ('PTLDA'== self.Algrithm) or ('EEGNET' == self.Algrithm) \
                or ('SHALLOWNET' in self.Algrithm) or ('FBCSPLDA' == self.Algrithm)  :
            X_ =torch.tensor(X.copy())
            y_pred = self.model.predict(X_)
        elif ('SHALLOWFBCSP' == self.Algrithm) or ('ATCNET' == self.Algrithm) or ('EEGCONFORMER' == self.Algrithm)\
                or ('EEGINCEPTIONMI' == self.Algrithm):
            X_ = X.astype(np.float32)
            y_pred = self.model.predict(X_)
            if self._le is not None:
                return self._le.inverse_transform(y_pred)
            else:
                return y_pred
        if self.Algrithm == 'EEGGCN_21':
            X_sel = X[:, self._mi21_idx, :]
            return self.model.predict(X_sel)
        return y_pred

    def predict_proba(self, X):
        """Return class probability estimates; None when the model does not support it."""
        X_cov = covariances(X, estimator='lwf')
        try:
            if self.Algrithm in ('CSPSVM', 'CSPLDA', 'TSLDA', 'MDRM', 'FGMDRM', 'DCPM'):
                return self.model.predict_proba(X_cov)
            elif self.Algrithm in ('FBCSP','FBCSPLDA'):
                return self.model.predict_proba(X)
            elif self.Algrithm == 'PTLDA':
                X_ = torch.tensor(X.copy(), dtype=torch.float32)
                return self.model.predict_proba(X_)
            elif self.Algrithm in ('EEGNET', 'SHALLOWNET'):
                X_ = torch.tensor(X.copy(), dtype=torch.float32)
                return self.model.predict_proba(X_)
            elif self.Algrithm in ('ATCNET', 'EEGCONFORMER', 'SHALLOWFBCSP', 'EEGINCEPTIONMI'):
                X_ = X.astype(np.float32)
                return self.model.predict_proba(X_)
            elif self.Algrithm == 'EEGGCN_21':
                X_sel = X[:, self._mi21_idx, :]
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X_sel)
        except Exception:
            pass
        return None
