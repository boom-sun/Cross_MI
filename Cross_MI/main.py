from Cross_MI.auxiliary.sun_data_loader import Dataloader
from Cross_MI.auxiliary.sun_data_saver import Saver
import os
from pathlib import Path
import yaml
from Cross_MI.auxiliary.basemodel import Basemodel

CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "Cross_MI\\configs")
DEFAULT_CONFIG = "online_hybrid.yaml"
Subject = list(range(1,11+1))

if __name__ == "__main__":
    with open(os.path.join(CONFIG_DIR, DEFAULT_CONFIG), 'rb') as f:
        config = yaml.safe_load(f)
    config["Subject_choose"] = Subject

    loader = Dataloader(config)
    model = Basemodel(config)
    saver = Saver(config, Subject)
    for subject in Subject:
        data, label, sub_label = loader.loader_data(subject)
        acc, res, auc = model.classier(subject, data, label, sub_label)
        saver.saver(subject, acc, res, auc)
        del data, label, sub_label

