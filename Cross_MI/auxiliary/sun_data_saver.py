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

    def saver(self, subject, acc, res, metrics):
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
        # Build per-metric dicts from the nested metrics structure for MATLAB compatibility
        auc     = {alg: metrics[alg]['auc']          for alg in metrics}
        bacc    = {alg: metrics[alg]['balanced_acc']  for alg in metrics}
        f1      = {alg: metrics[alg]['macro_f1']      for alg in metrics}
        kappa   = {alg: metrics[alg]['kappa']         for alg in metrics}
        mcc     = {alg: metrics[alg]['mcc']           for alg in metrics}
        sens    = {alg: metrics[alg]['sensitivity']   for alg in metrics}
        spec    = {alg: metrics[alg]['specificity']   for alg in metrics}
        conf    = {alg: metrics[alg]['conf_matrix']   for alg in metrics}
        sio.savemat(savename, {
            'acc': acc, 'res': res,
            'auc': auc, 'bacc': bacc, 'f1': f1,
            'kappa': kappa, 'mcc': mcc,
            'sens': sens, 'spec': spec, 'conf': conf,
        })
        if subject == self.subject_all[-1]:
            self.saver_all()

    # ------------------------------------------------------------------ helpers
    def _subject_savename(self, subject, stim_name):
        if self.class_class == []:
            if len(self.n_class) < 3:
                return (save_path + '\\result_' + self.dataset_name +
                        '\\{}result_sub{}_{}'.format(self.dataclass, subject, self.n_class[0]) +
                        '_' + self.sence + '_' + stim_name[self.n_class[1]-1] + '.mat')
            else:
                return (save_path + '\\result_' + self.dataset_name +
                        '\\{}result_sub{}_{}'.format(self.dataclass, subject, self.n_class[0]) +
                        '_' + self.sence + '_' + stim_name[self.n_class[1]-1] +
                        '_' + stim_name[self.n_class[2]-1] + '.mat')
        else:
            if len(self.class_class) == 2:
                return (save_path + '\\result_' + self.dataset_name +
                        '\\{}result_sub{}'.format(self.dataclass, subject) +
                        '_' + self.sence +
                        '_class{}_{}'.format(self.class_class[0], self.class_class[1]) + '.mat')
            else:
                return (save_path + '\\result_' + self.dataset_name +
                        '\\{}result_sub{}'.format(self.dataclass, subject) +
                        '_' + self.sence + '_class4.mat')

    def _all_savename(self, stim_name):
        if self.class_class == []:
            if len(self.n_class) < 3:
                return (save_path + '\\result_' + self.dataset_name +
                        '\\{}result_{}'.format(self.dataclass, self.n_class[0]) +
                        '_' + self.sence + '_' + stim_name[self.n_class[1]-1] + '.mat')
            else:
                return (save_path + '\\result_' + self.dataset_name +
                        '\\{}result_{}'.format(self.dataclass, self.n_class[0]) +
                        '_' + self.sence + '_' + stim_name[self.n_class[1]-1] +
                        '_' + stim_name[self.n_class[2]-1] + '.mat')
        else:
            if len(self.class_class) == 2:
                return (save_path + '\\result_' + self.dataset_name +
                        '\\{}result'.format(self.dataclass) +
                        '_' + self.sence +
                        '_class{}_{}'.format(self.class_class[0], self.class_class[1]) + '.mat')
            else:
                return (save_path + '\\result_' + self.dataset_name +
                        '\\{}result'.format(self.dataclass) +
                        '_' + self.sence + '_class4.mat')

    # Scalar metrics that are aggregated the same way as accuracy
    _SCALAR_FIELDS = ('acc', 'auc', 'bacc', 'f1', 'kappa', 'mcc', 'sens', 'spec')

    def saver_all(self):
        stim_name = stim_name_online if self.dataset_name in ['online_hybrid'] \
                    else ('ssvideo', 'video', 'ssmvep', 'cue')

        # Initialise per-subject aggregation containers
        allsub = {m: {key: [] for key in self.Algrithm} for m in self._SCALAR_FIELDS}
        allsub['res']  = {key: [] for key in self.Algrithm}
        allsub['conf'] = {key: [] for key in self.Algrithm}

        is_cv = self.sence in ['S1', 'S2']

        for subject in self.subject_all:
            savename    = self._subject_savename(subject, stim_name)
            data_struct = load_mat_as_namespace(savename)

            # ---- scalar metrics ----
            for field_name in self._SCALAR_FIELDS:
                field = getattr(data_struct, field_name, None)
                if field is None:
                    continue
                for key in field._fieldnames:
                    val = field.__dict__[key]
                    if is_cv:
                        allsub[field_name][format(key)].append(
                            np.mean([np.mean(val, axis=0)], axis=0))
                    else:
                        allsub[field_name][format(key)].append(cross_value_mean(val))

            # ---- predictions ----
            res_field = getattr(data_struct, 'res', None)
            if res_field is not None:
                for key in res_field._fieldnames:
                    allsub['res'][format(key)].append(res_field.__dict__[key])

            # ---- confusion matrices (store raw; sum in aggregate) ----
            conf_field = getattr(data_struct, 'conf', None)
            if conf_field is not None:
                for key in conf_field._fieldnames:
                    allsub['conf'][format(key)].append(conf_field.__dict__[key])

        # Print accuracy summary
        for key in self.Algrithm:
            vals = allsub['acc'].get(format(key), [])
            if vals:
                print('所有被试平均准确率为：',
                      '%.2f' % (np.mean(vals) * 100), '%',
                      '  [算法: %s]' % key)

        # ---- save aggregate ----
        savename_all = self._all_savename(stim_name)
        save_dict = {m: allsub[m] for m in self._SCALAR_FIELDS}
        save_dict['conf'] = allsub['conf']
        sio.savemat(savename_all, save_dict)

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
