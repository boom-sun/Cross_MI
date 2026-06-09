from Cross_MI.auxiliary.sun_data_loader import Dataloader
from Cross_MI.auxiliary.sun_data_saver import Saver
import os
from pathlib import Path
import yaml
from argparse import ArgumentParser
from Cross_MI.auxiliary.basemodel import Basemodel

CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "Cross_MI\\configs")
DEFAULT_CONFIG = "ssmvep_hybrid.yaml"
Subject = list(range(2, 2+1))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    with open(os.path.join(CONFIG_DIR, args.config), 'rb') as f:
        config = yaml.safe_load(f)

    loader = Dataloader(config)
    model = Basemodel(config)
    saver = Saver(config, Subject)
    for subject in Subject:
        data, label, sub_label = loader.loader_data(subject)

        from mne.filter import resample
        import numpy as np
        from MetaBCI.metabci.brainda.algorithms.decomposition.base import generate_filterbank
        from MetaBCI.metabci.brainda.algorithms.decomposition.csp import FBCSP
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        from Cross_MI.auxiliary.classifiers import Classier
        import dill as pickle

        # 降采样
        X = resample(data[0], up=250, down=1000)
        X = X - np.mean(X, axis=-1, keepdims=True)
        X = X / np.std(X, axis=(-1, -2), keepdims=True)
        model = Classier(srate=250, Algrithm='EEGNET')
        model.fit(X, label[0])
        # import pickle
        with open('ssvideo_ATCNet.pkl', 'wb') as f:
            pickle.dump(model.model, f)



        # brainda.algorithms.decomposition.csp.MultiCSP
        wp = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (4, 12), (12, 20), (20, 28)]
        ws = [(2, 10), (6, 14), (10, 18), (14, 22), (18, 26), (22, 30), (26, 34), (2, 14), (10, 22), (18, 30)]
        filterbank = generate_filterbank(wp, ws, srate=250, order=4, rp=0.5)
        # model = make_pipeline(
        #     MultiCSP(n_components = 2),
        #     LinearDiscriminantAnalysis())
        model = make_pipeline(*[
            FBCSP(n_components=5,
                  n_mutualinfo_components=4,
                  filterbank=filterbank),
            SVC()
        ])
        # fit()训练模型
        model = model.fit(X, label[0])
        import pickle
        with open('ssarrow_FBCSP.pkl', 'wb') as f:
            pickle.dump(model, f)


        # acc, res, auc = model.classier(subject, data, label, sub_label)
        # saver.saver(subject, acc, res, auc)

