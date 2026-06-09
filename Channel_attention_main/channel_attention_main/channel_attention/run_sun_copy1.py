from argparse import ArgumentParser
import os
from pathlib import Path
import numpy as np
import yaml
import sys
import scipy.io as io
# sys.path.append('channel_attention_main')
from train_test_sun import train_and_test

CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "configs")
DEFAULT_CONFIG = "Yi_basenet.yaml"


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    # load config
    with open(os.path.join(CONFIG_DIR, args.config)) as f:
        config = yaml.safe_load(f)
    Acc = []
    for train_rate in np.array(range(10, 151, 10))/160:
        acc = train_and_test(config, train_rate)
        Acc.append(acc)
        io.savemat('result_basenet_{}_4.mat'.format(train_rate), {'acc': acc})
    io.savemat('result_basenet_4.mat', {'acc': Acc})
        