from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from .base import BaseDataModule
from utils.load_sun import load_sun_data

class SUN_SSMVEP(BaseDataModule):
    all_subject_ids = list(range(1, 23))
    class_names = ["hand(L)", "hand(R)","rest(L)","rest(R)"]

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_sun_data([self.subject_id], self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        X = self.dataset["data"][str(self.subject_id)]["train"]
        y = self.dataset["labels"][str(self.subject_id)]["train"]
        X_test = self.dataset["data"][str(self.subject_id)]["test"]
        y_test = self.dataset["labels"][str(self.subject_id)]["test"]

        # scale data
        if self.preprocessing_dict["z_scale"]:
            for ch_idx in range(X.shape[1]):
                sc = StandardScaler()
                X[:, ch_idx, :] = sc.fit_transform(X[:, ch_idx, :])
                X_test[:, ch_idx, :] = sc.transform(X_test[:, ch_idx, :])

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
