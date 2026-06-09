import numpy as np
import scipy.io as sio
from skorch import NeuralNetClassifier
from types import SimpleNamespace
from joblib import dump, load  # 新增
import joblib
import os
import random
seed = 123
random.seed(seed)

def load_mat_as_namespace(file_path):
    """
    加载MAT文件并将其内容转换为命名空间对象，
    允许通过字段名称进行点操作符访问。
    """
    def dict_to_namespace(d):
        """
        将字典转换为命名空间对象，递归处理嵌套字典。
        """
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = dict_to_namespace(value)
            elif isinstance(value, list):
                d[key] = [dict_to_namespace(item) if isinstance(item, dict) else item for item in value]
        return SimpleNamespace(**d)

    # 加载MAT文件
    data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)

    # 清理MAT文件中的元信息字段
    cleaned_data = {key: value for key, value in data.items() if not key.startswith('__')}

    # 将字典转换为命名空间对象
    return dict_to_namespace(cleaned_data)
def mat_struct_to_dict(mat_obj):
    """
    将 mat_struct 对象递归地转换为字典。
    """
    if isinstance(mat_obj, SimpleNamespace):
        return {key: mat_struct_to_dict(value) for key, value in mat_obj.__dict__.items()}
    elif isinstance(mat_obj, list):
        return [mat_struct_to_dict(item) for item in mat_obj]
    else:
        return mat_obj

def cross_value_mean(data):
    import numpy
    data_mean = []
    if type(data) is numpy.ndarray:
        return np.array(data)
    elif type(data) is float:
        return np.array(data)
    else:
        data_mean.append(data[0])
        if len(data) > 1:
            for i in range(1, len(data)):
                data_mean.append(np.mean(data[i], axis=0))
        return np.array(data_mean,dtype='float')



stim_name = ('ssvideo','video','ssmvep','cue')
stim_name_online = ('ssvideo', 'video', 'ssmveparrow', 'arrow')
save_path='E:\\Code\\Cross\\Cross_MI\\'
class Saver():
    def __init__(self, config, subject_all):
        self.sence=config['sence']
        self.Subject = config['Subject']
        self.Algrithm = config['Algrithm']
        self.n_class = config['n_class']
        self.dataset_name = config['dataset_name']
        self.dataclass = config['dataclass']
        self.class_class = config['class_class']
        self.subject_all = subject_all
        for i in range(len(self.Algrithm)):
            self.Algrithm[i] = self.Algrithm[i].upper()

    def saver(self, subject, acc, res, auc):
        if self.dataset_name in ['online_hybrid']:
            stim_name= ('ssvideo', 'video', 'ssmveparrow', 'arrow')
        else:
            stim_name=('ssvideo','video','ssmvep','cue')
        if self.class_class == []:
            if len(self.n_class) < 3:
                savename = save_path+'\\result_' + self.dataset_name + \
                               '\\{}result_sub{}_{}'.format(self.dataclass, subject, self.n_class[0]) + \
                               '_' + self.sence + '_' + stim_name[self.n_class[1]-1] + '.mat'
            else:
                savename = save_path+'\\result_' + self.dataset_name + \
                               '\\{}result_sub{}_{}'.format(self.dataclass, subject, self.n_class[0]) + \
                               '_' + self.sence + '_' + stim_name[self.n_class[1]-1] + '_' + stim_name[
                                   self.n_class[2]-1] + '.mat'
        else:
            if len(self.class_class) == 2:
                savename = save_path+'\\result_' + self.dataset_name + \
                               '\\{}result_sub{}'.format(self.dataclass,subject) + \
                               '_' + self.sence + '_class{}_{}'.format(self.class_class[0],
                                                                       self.class_class[1]) + '.mat'
            else:
                savename = save_path+'\\result_' + self.dataset_name + \
                               '\\{}result_sub{}'.format(self.dataclass,subject) + \
                               '_' + self.sence + '_class4.mat'
        sio.savemat(savename, {'acc': acc, 'res': res, 'auc': auc})
        if subject == self.subject_all[-1]:
            self.saver_all()

    def saver_all(self):
        if self.dataset_name in ['online_hybrid']:
            stim_name=stim_name_online
        else:
            stim_name = ('ssvideo', 'video', 'ssmvep', 'cue')
        for subject in self.subject_all:
            if self.class_class == []:
                if len(self.n_class) < 3:
                    savename = save_path + '\\result_' +self.dataset_name + \
                               '\\{}result_sub{}_{}'.format(self.dataclass, subject, self.n_class[0]) + \
                               '_' + self.sence + '_' + stim_name[self.n_class[1]-1] + '.mat'
                else:
                    savename = save_path+ '\\result_' +self.dataset_name + \
                               '\\{}result_sub{}_{}'.format(self.dataclass, subject, self.n_class[0]) + \
                               '_' + self.sence + '_' + stim_name[self.n_class[1]-1] + '_' + stim_name[
                                   self.n_class[2]-1] + '.mat'
            else:
                if len(self.class_class) == 2:
                    savename = save_path+'\\result_' +self.dataset_name + \
                               '\\{}result_sub{}'.format(self.dataclass, subject) + \
                               '_' + self.sence + '_class{}_{}'.format(self.class_class[0],
                                                                       self.class_class[1]) + '.mat'
                else:
                    savename = save_path+ '\\result_' +self.dataset_name + \
                               '\\{}result_sub{}'.format(self.dataclass, subject) + \
                               '_' + self.sence + '_class4.mat'
            data_struct = load_mat_as_namespace(savename)
            acc = data_struct.acc
            res = data_struct.res
            auc = data_struct.auc
            if subject == self.subject_all[0]:
                self.allsub_acc = {key: [] for key in self.Algrithm}
                self.allsub_res = {key: [] for key in self.Algrithm}
                self.allsub_auc = {key: [] for key in self.Algrithm}
            if self.sence in ['S1', 'S2']:
                for (key) in acc._fieldnames:  # 准确率：train*fold，平均后为train次训练fold折后平均
                    self.allsub_acc[format(key)].append(np.mean([np.mean(acc.__dict__[key], axis=0)], axis=0))
                for (key) in res._fieldnames:
                    self.allsub_res[format(key)].append(res.__dict__[key])
                for (key) in auc._fieldnames:  # auc：train*fold，平均后为train次训练fold折后平均
                    self.allsub_auc[format(key)].append(np.mean([np.mean(auc.__dict__[key], axis=0)], axis=0))
            if self.sence in ['cross1', 'cross2', 'cross_subject_S1','cross_task1', 'cross_task2',
                              'cross_subject_S2', 'cross_double1', 'cross_double2','S1_online', 'S2_online',
                              'global_cross_double1', 'global_cross_double2']:
                for (key) in acc._fieldnames:
                    self.allsub_acc[format(key)].append(cross_value_mean(acc.__dict__[key]))
                for (key) in res._fieldnames:
                    self.allsub_res[format(key)].append((res.__dict__[key]))
                for (key) in auc._fieldnames:
                    self.allsub_auc[format(key)].append(cross_value_mean(auc.__dict__[key]))
        for (key) in acc._fieldnames:
            print('所有被试平均准确率为：', '%.2f' % (np.mean(self.allsub_acc[format(key)])*100), '%')

        if self.class_class == []:
            if len(self.n_class) < 3:
                savename_all = save_path+'\\result_'+self.dataset_name + \
                '\\{}result_{}'.format(self.dataclass, self.n_class[0]) + \
                '_' + self.sence + '_' + stim_name[self.n_class[1]-1]+'.mat'
            else:
                savename_all = save_path+'\\result_'+self.dataset_name + \
                '\\{}result_{}'.format(self.dataclass, self.n_class[0]) + \
                '_' + self.sence + '_' + stim_name[self.n_class[1]-1]+'_'+stim_name[self.n_class[2]-1]+'.mat'
        else:
            if len(self.class_class) == 2:
                savename_all = save_path+'\\result_'+self.dataset_name + \
                '\\{}result'.format(self.dataclass) + \
                '_' + self.sence + '_class{}_{}'.format(self.class_class[0],self.class_class[1]) +'.mat'
            else:
                savename_all = save_path+'\\result_'+self.dataset_name + \
                '\\{}result'.format(self.dataclass) + \
                '_' + self.sence + '_class4.mat'
        sio.savemat(savename_all, {'acc': self.allsub_acc})

class ModelIO:
    """
    统一管理模型流水线的保存与加载。
    保存内容：aux_model（如 RSF）、对齐状态、已训练分类器（Classier 实例）等。
    """
    def __init__(self, config):
        self.config = config
        self.dataset = config['dataset_name']
        self.sence = config['sence']
        self.dataclass = config['dataclass']
        self.n_class = config['n_class']
        self.root = os.path.join(save_path, f"result_{self.dataset}", f"models_{self.dataset}")
        os.makedirs(self.root, exist_ok=True)

    def _subdir(self):
        # 用 n_class 构成可区分跨任务/非跨任务的子目录
        if isinstance(self.n_class, (list, tuple)) and len(self.n_class) >= 2:
            tag = "_".join([str(x) for x in self.n_class])
        else:
            tag = str(self.n_class)
        return os.path.join(self.root, f"{self.dataclass}_{self.sence}_{tag}")

    def _model_path(self, algrithm):
        d = os.path.join(self._subdir(), algrithm.upper())
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "pipeline.joblib")

    def exists(self, algrithm):
        return os.path.exists(self._model_path(algrithm))

    def save(self, algrithm, pipeline_dict):
        dump(pipeline_dict, self._model_path(algrithm))

    def load(self, algrithm):
        return load(self._model_path(algrithm))
    # 追加到 ModelIO 类里
    def weights_path(self, algrithm):
        d = os.path.join(self._subdir(), algrithm.upper())
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "weights.pt")


if __name__ == "__main__":
    import os
    from pathlib import Path
    import yaml
    from argparse import ArgumentParser

    CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "configs")
    DEFAULT_CONFIG = "online_hybrid.yaml"
    Subject = [1,2,5,7,8,10,11]
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    with open(os.path.join(CONFIG_DIR, args.config), 'rb') as f:
        config = yaml.safe_load(f)
    model = Saver(config, Subject)
    model.saver_all()
