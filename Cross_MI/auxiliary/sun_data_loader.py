import hdf5storage
import numpy as np
import random
seed = 123
random.seed(seed)
np.random.seed(seed)
filepath_graz = 'E:\\Datasets\\1_Graz范式\\处理后数据\\'
filepath_ssmvep = 'E:\\Datasets\\2_MI_SSMVEP范式\\MI_SSMVEP处理后数据\\'
filepath_ssmvep_hybrid = 'E:\\Datasets\\4_跨场景因素研究v2\\跨场景因素研究v2处理后数据\\'
filepath_online_hybrid = 'E:\\Datasets\\5_跨场景因素在线\\跨场景在线处理后数据\\'

stim_name_online = ('ssvideo', 'video', 'ssmveparrow', 'arrow')
def find_element(matrix, target):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == target:
                return (i, j)
    return None

class Dataloader:
    def __init__(self, config):
        self.startpoint = config['starttime']*config['srate']
        self.endpoint = config['endtime']*config['srate']
        self.srate = config['srate']
        self.dataset = config['dataset_name']
        self.sence = config['sence']
        self.dataclass = config['dataclass']
        self.class_class = config['class_class']
        self.n_class = config['n_class']
        self.Subject = config['Subject']
        self.ch_choose = list(np.array(config['ch_choose'])-1)
        self.config = config
    def loader_data(self, subject):
        if self.dataset in ['graz']:
            data, label, sub_label = self.graz_paradigm(sence=self.sence, subject=subject, dataclass=self.dataclass,
                                    class_class=self.class_class, Subject=self.Subject, ch_choose=self.ch_choose)
        if self.dataset in ['ssmvep']:
            data, label, sub_label = self.ssmvep_paradigm(sence=self.sence, subject=subject, dataclass=self.dataclass,
                                    class_class=self.class_class, Subject=self.Subject, ch_choose=self.ch_choose)
        if self.dataset in ['ssmvep_hybrid']:
            data, label, sub_label = self.ssmvep_hybrid_paradigm(sence=self.sence, subject=subject, dataclass=self.dataclass,
                                    n_class=self.n_class, Subject=self.Subject, ch_choose=self.ch_choose)
        if self.dataset in ['online_hybrid']:
            data, label, sub_label = self.online_hybrid_paradigm(sence=self.sence, subject=subject, dataclass=self.dataclass,
                                    n_class=self.n_class, Subject=self.Subject, ch_choose=self.ch_choose)
        if subject == self.config['Subject_choose'][0]:
            print('当前场景为：',self.sence)

        return data, label, sub_label

    def loader_single(self, sence, subject, dataclass, n_class, class_class, ch_choose):
        stim_name = ('ssvideo','video','ssmvep','cue')
        if self.dataset in ['online_hybrid']:
            stim_name_online = ('ssvideo', 'video', 'ssmveparrow', 'arrow')
            if sence in ['S1']:
                if subject < 10:
                    filepath1 = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject,
                                                                     stim_name_online[n_class[1]-1])+'.mat'
                elif subject >= 10:
                    filepath1 = self.filepath + '{}S{}S1_{}'.format(dataclass, subject,
                                                                    stim_name_online[n_class[1]-1])+'.mat'
            if sence in ['S2']:
                if subject < 10:
                    filepath1 = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject,
                                                                     stim_name_online[n_class[1]-1])+'.mat'
                elif subject >= 10:
                    filepath1 = self.filepath + '{}S{}S2_{}'.format(dataclass, subject,
                                                                    stim_name_online[n_class[1]-1])+'.mat'

        else:
            if n_class == []:
                if sence in ['S1']:
                    if subject < 10:
                        filepath1 = self.filepath+'{}S0{}S1.mat'.format(dataclass, subject)
                    elif subject >= 10:
                        filepath1 = self.filepath+'{}S{}S1.mat'.format(dataclass, subject)
                elif sence in ['S2']:
                    if subject < 10:
                        filepath1 = self.filepath+'{}S0{}S2.mat'.format(dataclass, subject)
                    elif subject >= 10:
                        filepath1 = self.filepath+'{}S{}S2.mat'.format(dataclass, subject)
            elif n_class[0] == 0:
                if sence in ['S1']:
                    if subject < 10:
                        filepath1 = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    elif subject >= 10:
                        filepath1 = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                if sence in ['S2']:
                    if subject < 10:
                        filepath1 = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    elif subject >= 10:
                        filepath1 = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
        data1 = hdf5storage.loadmat(filepath1, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
        label1 = hdf5storage.loadmat(filepath1, mat_dtype=True)['label']
        # state = hdf5storage.loadmat(filepath1, mat_dtype=True)['fs']
        data1 = data1.transpose(2, 0, 1)
        label1 = np.squeeze(label1)
        if class_class == []:
            class_class=np.unique(label1)
        data = np.concatenate(([data1[label1 == class_class[i]] for i in range(len(class_class))]), axis=0)
        label = np.concatenate(([[i]*len(label1[label1 == class_class[i]]) for i in range(len(class_class))]), axis=0)# 让标签从0开始排
        index = np.arange(len(label))
        np.random.shuffle(index)
        data = data[index, :, :]
        label = label[index]
        if ch_choose==[]:
            pass
        else:
            data = data[:, ch_choose, :]
        sub_label = [0]*data.shape[0]  # 代表属于哪个被试的数据
        return data, label, sub_label

    def loader_cross_sence(self, sence, subject, dataclass, n_class, class_class,ch_choose):
        stim_name = ('ssvideo','video','ssmvep','cue')
        if n_class == []:
            if sence in ['cross1']:
                if subject < 10:
                    source_filepath = self.filepath + '{}S0{}S1.mat'.format(dataclass, subject)
                    target_filepath = self.filepath + '{}S0{}S2.mat'.format(dataclass, subject)
                elif subject >= 10:
                    source_filepath = self.filepath + '{}S{}S1.mat'.format(dataclass, subject)
                    target_filepath = self.filepath + '{}S{}S2.mat'.format(dataclass, subject)
            if sence in ['cross2']:
                if subject < 10:
                    source_filepath = self.filepath + '{}S0{}S2.mat'.format(dataclass, subject)
                    target_filepath = self.filepath + '{}S0{}S1.mat'.format(dataclass, subject)
                elif subject >= 10:
                    source_filepath = self.filepath + '{}S{}S2.mat'.format(dataclass, subject)
                    target_filepath = self.filepath + '{}S{}S1.mat'.format(dataclass, subject)
        elif n_class[0]==0:
            if sence in ['cross1']:
                if subject < 10:
                    source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                elif subject >= 10:
                    source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
            if sence in ['cross2']:
                if subject < 10:
                    source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                elif subject >= 10:
                    source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
        elif n_class[0] == 1:
            if sence in ['cross1']:
                if subject < 10:
                    source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
                elif subject >= 10:
                    source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
            if sence in ['cross2']:
                if subject < 10:
                    source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
                elif subject >= 10:
                    source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
        if self.dataset in ['online_hybrid']:
            if n_class[0]==0:
                if sence in ['cross1']:
                    if subject < 10:
                        source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                        target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'_online.mat'
                    elif subject >= 10:
                        source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                        target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'_online.mat'
                if sence in ['cross2']:
                    if subject < 10:
                        source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                        target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'_online.mat'
                    elif subject >= 10:
                        source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                        target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'_online.mat'
            elif n_class[0] == 1:
                if sence in ['cross1']:
                    if subject < 10:
                        source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                        target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[2]-1])+'_online.mat'
                    elif subject >= 10:
                        source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                        target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[2]-1])+'_online.mat'
                if sence in ['cross2']:
                    if subject < 10:
                        source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                        target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[2]-1])+'_online.mat'
                    elif subject >= 10:
                        source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                        target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[2]-1])+'_online.mat'

        source_data = hdf5storage.loadmat(source_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
        source_label = hdf5storage.loadmat(source_filepath, mat_dtype=True)['label']
        target_data = hdf5storage.loadmat(target_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
        target_label = hdf5storage.loadmat(target_filepath, mat_dtype=True)['label']
        # state = hdf5storage.loadmat(filepath2, mat_dtype=True)['fs']
        source_data = source_data.transpose(2, 0, 1)
        source_label = np.squeeze(source_label)
        target_data = target_data.transpose(2, 0, 1)
        target_label = np.squeeze(target_label)
        if class_class==[]:
            class_class = np.unique(source_label)
        Source_data=np.concatenate(([source_data[source_label==class_class[i]] for i in range(len(class_class))]),axis=0)
        Source_label=np.concatenate(([[i]*len(source_label[source_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
        Target_data=np.concatenate(([target_data[target_label==class_class[i]] for i in range(len(class_class))]),axis=0)
        Target_label=np.concatenate(([[i]*len(target_label[target_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
        index = np.arange(len(Source_label))
        np.random.shuffle(index)
        Source_data = Source_data[index, :, :]
        Source_label=Source_label[index]
        index = np.arange(len(Target_label))
        np.random.shuffle(index)
        Target_data=Target_data[index,:,:]
        Target_label=Target_label[index]
        data,label,sub_label=[],[],[]
        if ch_choose==[]:
            pass
        else:
            Source_data = Source_data[:, ch_choose, :]
            Target_data = Target_data[:, ch_choose, :]
        data.append(Source_data)
        data.append(Target_data)
        label.append(Source_label)
        label.append(Target_label)
        sub_label.append([0] * Source_data.shape[0])
        sub_label.append([1] * Target_data.shape[0])
        return data, label, sub_label

    def loader_cross_task(self, sence, subject, dataclass, n_class, class_class,ch_choose):
        stim_name = ('ssvideo','video','ssmvep','cue')
        if sence in ['cross_task1']:
            if subject < 10:
                source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
            elif subject >= 10:
                source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
        if sence in ['cross_task2']:
            if subject < 10:
                source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
            elif subject >= 10:
                source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
        if self.dataset in ['online_hybrid']:
            if sence in ['cross1']:
                if subject < 10:
                    source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[2]-1])+'_online.mat'
                elif subject >= 10:
                    source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[2]-1])+'_online.mat'
            if sence in ['cross2']:
                if subject < 10:
                    source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[2]-1])+'_online.mat'
                elif subject >= 10:
                    source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                    target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[2]-1])+'_online.mat'

        source_data = hdf5storage.loadmat(source_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
        source_label = hdf5storage.loadmat(source_filepath, mat_dtype=True)['label']
        target_data = hdf5storage.loadmat(target_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
        target_label = hdf5storage.loadmat(target_filepath, mat_dtype=True)['label']
        source_data = source_data.transpose(2, 0, 1)
        source_label = np.squeeze(source_label)
        target_data = target_data.transpose(2, 0, 1)
        target_label = np.squeeze(target_label)
        if class_class==[]:
            class_class = np.unique(source_label)
        Source_data=np.concatenate(([source_data[source_label==class_class[i]] for i in range(len(class_class))]),axis=0)
        Source_label=np.concatenate(([[i]*len(source_label[source_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
        Target_data=np.concatenate(([target_data[target_label==class_class[i]] for i in range(len(class_class))]),axis=0)
        Target_label=np.concatenate(([[i]*len(target_label[target_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
        index = np.arange(len(Source_label))
        np.random.shuffle(index)
        Source_data = Source_data[index, :, :]
        Source_label=Source_label[index]
        index = np.arange(len(Target_label))
        np.random.shuffle(index)
        Target_data=Target_data[index,:,:]
        Target_label=Target_label[index]
        data,label,sub_label=[],[],[]
        if ch_choose==[]:
            pass
        else:
            Source_data = Source_data[:, ch_choose, :]
            Target_data = Target_data[:, ch_choose, :]
        data.append(Source_data)
        data.append(Target_data)
        label.append(Source_label)
        label.append(Target_label)
        sub_label.append([0] * Source_data.shape[0])
        sub_label.append([1] * Target_data.shape[0])
        return data, label, sub_label

    def loader_single_online(self, sence, subject, dataclass, n_class, class_class, ch_choose):
        if sence in ['S1_online']:
            if subject < 10:
                source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'_online.mat'
            elif subject >= 10:
                source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'_online.mat'
        if sence in ['S2_online']:
            if subject < 10:
                source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'_online.mat'
            elif subject >= 10:
                source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'.mat'
                target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name_online[n_class[1]-1])+'_online.mat'

        source_data = hdf5storage.loadmat(source_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
        source_label = hdf5storage.loadmat(source_filepath, mat_dtype=True)['label']
        target_data = hdf5storage.loadmat(target_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
        target_label = hdf5storage.loadmat(target_filepath, mat_dtype=True)['label']
        # state = hdf5storage.loadmat(filepath2, mat_dtype=True)['fs']
        source_data = source_data.transpose(2, 0, 1)
        source_label = np.squeeze(source_label)
        target_data = target_data.transpose(2, 0, 1)
        target_label = np.squeeze(target_label)
        if class_class==[]:
            class_class = np.unique(source_label)
        Source_data=np.concatenate(([source_data[source_label==class_class[i]] for i in range(len(class_class))]),axis=0)
        Source_label=np.concatenate(([[i]*len(source_label[source_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
        Target_data=np.concatenate(([target_data[target_label==class_class[i]] for i in range(len(class_class))]),axis=0)
        Target_label=np.concatenate(([[i]*len(target_label[target_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
        index = np.arange(len(Source_label))
        np.random.shuffle(index)
        Source_data = Source_data[index, :, :]
        Source_label=Source_label[index]
        index = np.arange(len(Target_label))
        np.random.shuffle(index)
        Target_data=Target_data[index,:,:]
        Target_label=Target_label[index]
        data,label,sub_label=[],[],[]
        if ch_choose==[]:
            pass
        else:
            Source_data = Source_data[:, ch_choose, :]
            Target_data = Target_data[:, ch_choose, :]
        data.append(Source_data)
        data.append(Target_data)
        label.append(Source_label)
        label.append(Target_label)
        sub_label.append([0] * Source_data.shape[0])
        sub_label.append([1] * Target_data.shape[0])
        return data, label, sub_label
    def loader_cross_subject(self, sence, subject, dataclass, n_class, class_class, Subject, ch_choose):
        stim_name = ('ssvideo','video','ssmvep','cue')
        SUB_subject = np.array_split(Subject[0], Subject[1])  # 将所有被试分成几组
        if subject in Subject:
            sub_position = find_element(SUB_subject, subject)
            sub_subject = SUB_subject[sub_position[0]]
            source_subject = sub_subject.tolist()
            source_subject.remove(subject)
        else:
            source_subject=Subject
        if n_class==[]:
            if sence in ['cross_subject1']:
                if subject < 10:
                    target_filepath = self.filepath + '{}S0{}S1.mat'.format(dataclass, subject)
                elif subject >= 10:
                    target_filepath = self.filepath + '{}S{}S1.mat'.format(dataclass, subject)
            if sence in ['cross_subject2']:
                if subject < 10:
                    target_filepath = self.filepath + '{}S0{}S2.mat'.format(dataclass, subject)
                elif subject >= 10:
                    target_filepath = self.filepath + '{}S{}S2.mat'.format(dataclass, subject)
        elif n_class[0]==0:
            if sence in ['cross_subject1']:
                if subject < 10:
                    target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                elif subject >= 10:
                    target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
            if sence in ['cross_subject2']:
                if subject < 10:
                    target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                elif subject >= 10:
                    target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
        elif n_class[0]==1:
            if sence in ['cross_subject1']:
                if subject < 10:
                    target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
                elif subject >= 10:
                    target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
            if sence in ['cross_subject2']:
                if subject < 10:
                    target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
                elif subject >= 10:
                    target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'

        target_data = hdf5storage.loadmat(target_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
        target_label = hdf5storage.loadmat(target_filepath, mat_dtype=True)['label']
        target_data = target_data.transpose(2, 0, 1)
        target_label = np.squeeze(target_label)
        if class_class==[]:
            class_class=np.unique(target_label)
        Target_data=np.concatenate(([target_data[target_label==class_class[i]] for i in range(len(class_class))]),axis=0)
        Target_label=np.concatenate(([[i]*len(target_label[target_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
        index = np.arange(len(Target_label))
        np.random.shuffle(index)
        Target_data=Target_data[index,:,:]
        Target_label=Target_label[index]
        Target_sub_label=[subject] * Target_data.shape[0]

        Source_Data, Source_Label, Source_sub_label=[],[],[]
        for sub_sub in source_subject:
            if n_class==[]:
                if sence in ['cross_subject1']:
                    if sub_sub < 10:
                        source_filepath = self.filepath + '{}S0{}S1.mat'.format(dataclass, sub_sub)
                    elif sub_sub >= 10:
                        source_filepath = self.filepath + '{}S{}S1.mat'.format(dataclass, sub_sub)
                if sence in ['cross_subject2']:
                    if sub_sub < 10:
                        source_filepath = self.filepath + '{}S0{}S2.mat'.format(dataclass, sub_sub)
                    elif sub_sub >= 10:
                        source_filepath = self.filepath + '{}S{}S2.mat'.format(dataclass, sub_sub)
            elif n_class[0]==0:
                if sence in ['cross_subject1']:
                    if sub_sub < 10:
                        source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass,sub_sub, stim_name[n_class[1]-1])+'.mat'
                    elif sub_sub >= 10:
                        source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                if sence in ['cross_subject2']:
                    if sub_sub < 10:
                        source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                    elif sub_sub >= 10:
                        source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
            elif n_class[0]==1:
                if sence in ['cross_subject1']:
                    if sub_sub < 10:
                        source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                    elif sub_sub >= 10:
                        source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                if sence in ['cross_subject2']:
                    if sub_sub < 10:
                        source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                    elif sub_sub >= 10:
                        source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'

            source_data = hdf5storage.loadmat(source_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
            source_label = hdf5storage.loadmat(source_filepath, mat_dtype=True)['label']
            source_data = source_data.transpose(2, 0, 1)
            source_label = np.squeeze(source_label)
            Source_data=np.concatenate(([source_data[source_label==class_class[i]] for i in range(len(class_class))]),axis=0)
            Source_label=np.concatenate(([[i]*len(source_label[source_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
            index = np.arange(len(Source_label))
            np.random.shuffle(index)
            Source_data = Source_data[index, :, :]
            Source_label=Source_label[index]
            Source_Data.append(Source_data)
            Source_Label.append(Source_label)
            Source_sub_label.append([sub_sub] * Source_data.shape[0])
        Source_Data=np.concatenate((Source_Data),axis=0)
        Source_Label=np.concatenate((Source_Label),axis=0)
        Source_sub_label=np.concatenate((Source_sub_label),axis=0)

        data,label,sub_label=[],[],[]
        if ch_choose==[]:
            pass
        else:
            Source_Data = Source_Data[:, ch_choose, :]
            Target_data = Target_data[:, ch_choose, :]
        data.append(Source_Data)
        data.append(Target_data)
        label.append(Source_Label)
        label.append(Target_label)
        sub_label.append(Source_sub_label)
        sub_label.append(Target_sub_label)
        return data, label, sub_label

    def loader_cross_double(self, sence, subject, dataclass, n_class, class_class,Subject, ch_choose):
        stim_name = ('ssvideo','video','ssmvep','cue')
        SUB_subject = np.array_split(Subject[0], Subject[1])  # 将所有被试分成几组
        if subject in Subject[0]:
            sub_position = find_element(SUB_subject, subject)
            sub_subject = SUB_subject[sub_position[0]]
            source_subject = sub_subject.tolist()
            source_subject.remove(subject)
        else:
            source_subject=Subject[0]
        if self.dataset in ['online_hybrid']:
            if n_class[0] == 0:
                if sence in ['cross_double1']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject,
                                                                               stim_name_online[n_class[1] - 1]) + '_online.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject,
                                                                              stim_name_online[n_class[1] - 1]) + '_online.mat'
                if sence in ['cross_double2']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject,
                                                                               stim_name_online[n_class[1] - 1]) + '_online.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject,
                                                                              stim_name_online[n_class[1] - 1]) + '_online.mat'
            elif n_class[0] == 1:
                if sence in ['cross_double1']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject,
                                                                               stim_name_online[n_class[2] - 1]) + '_online.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject,
                                                                              stim_name_online[n_class[2] - 1]) + '_online.mat'
                if sence in ['cross_double2']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject,
                                                                               stim_name_online[n_class[2] - 1]) + '_online.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject,
                                                                              stim_name_online[n_class[2] - 1]) + '_online.mat'
        else:
            if n_class==[]:
                if sence in ['cross_double1']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S2.mat'.format(dataclass, subject)
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S2.mat'.format(dataclass, subject)
                if sence in ['cross_double2']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S1.mat'.format(dataclass, subject)
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S1.mat'.format(dataclass, subject)
            elif n_class[0]==0:
                if sence in ['cross_double1']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                if sence in ['cross_double2']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name[n_class[1]-1])+'.mat'
            elif n_class[0]==1:
                if sence in ['cross_double1']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
                if sence in ['cross_double2']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject, stim_name[n_class[2]-1])+'.mat'

        target_data = hdf5storage.loadmat(target_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
        target_label = hdf5storage.loadmat(target_filepath, mat_dtype=True)['label']
        target_data = target_data.transpose(2, 0, 1)
        target_label = np.squeeze(target_label)
        if class_class==[]:
            class_class=np.unique(target_label)
        Target_data=np.concatenate(([target_data[target_label==class_class[i]] for i in range(len(class_class))]),axis=0)
        Target_label=np.concatenate(([[i]*len(target_label[target_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
        index = np.arange(len(Target_label))
        np.random.shuffle(index)
        Target_data=Target_data[index,:,:]
        Target_label=Target_label[index]
        Target_sub_label=[subject] * Target_data.shape[0]

        Source_Data, Source_Label, Source_sub_label=[],[],[]
        for sub_sub in source_subject:
            if self.dataset in ['online_hybrid']:
                if n_class[0] == 0:
                    if sence in ['cross_double1']:
                        if sub_sub < 10:
                            source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, sub_sub,
                                                                                   stim_name_online[n_class[1] - 1]) + '.mat'
                        elif sub_sub >= 10:
                            source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, sub_sub,
                                                                                  stim_name_online[n_class[1] - 1]) + '.mat'
                    if sence in ['cross_double2']:
                        if sub_sub < 10:
                            source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, sub_sub,
                                                                                   stim_name_online[n_class[1] - 1]) + '.mat'
                        elif sub_sub >= 10:
                            source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, sub_sub,
                                                                                  stim_name_online[n_class[1] - 1]) + '.mat'
                elif n_class[0] == 1:
                    if sence in ['cross_double1']:
                        if sub_sub < 10:
                            source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, sub_sub,
                                                                                   stim_name_online[n_class[1] - 1]) + '.mat'
                        elif sub_sub >= 10:
                            source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, sub_sub,
                                                                                  stim_name_online[n_class[1] - 1]) + '.mat'
                    if sence in ['cross_double2']:
                        if sub_sub < 10:
                            source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, sub_sub,
                                                                                   stim_name_online[n_class[1] - 1]) + '.mat'
                        elif sub_sub >= 10:
                            source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, sub_sub,
                                                                                  stim_name_online[n_class[1] - 1]) + '.mat'
            else:
                if n_class==[]:
                    if sence in ['cross_double1']:
                        if sub_sub < 10:
                            source_filepath = self.filepath + '{}S0{}S1.mat'.format(dataclass, sub_sub)
                        elif sub_sub >= 10:
                            source_filepath = self.filepath + '{}S{}S1.mat'.format(dataclass, sub_sub)
                    if sence in ['cross_double2']:
                        if sub_sub < 10:
                            source_filepath = self.filepath + '{}S0{}S2.mat'.format(dataclass, sub_sub)
                        elif sub_sub >= 10:
                            source_filepath = self.filepath + '{}S{}S2.mat'.format(dataclass, sub_sub)
                elif n_class[0]==0:
                    if sence in ['cross_double1']:
                        if sub_sub < 10:
                            source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass,sub_sub, stim_name[n_class[1]-1])+'.mat'
                        elif sub_sub >= 10:
                            source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                    if sence in ['cross_double2']:
                        if sub_sub < 10:
                            source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                        elif sub_sub >= 10:
                            source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                elif n_class[0]==1:
                    if sence in ['cross_double1']:
                        if sub_sub < 10:
                            source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                        elif sub_sub >= 10:
                            source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                    if sence in ['cross_double2']:
                        if sub_sub < 10:
                            source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                        elif sub_sub >= 10:
                            source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'

            source_data = hdf5storage.loadmat(source_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
            source_label = hdf5storage.loadmat(source_filepath, mat_dtype=True)['label']
            source_data = source_data.transpose(2, 0, 1)
            source_label = np.squeeze(source_label)
            Source_data=np.concatenate(([source_data[source_label==class_class[i]] for i in range(len(class_class))]),axis=0)
            Source_label=np.concatenate(([[i]*len(source_label[source_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
            index = np.arange(len(Source_label))
            np.random.shuffle(index)
            Source_data = Source_data[index, :, :]
            Source_label=Source_label[index]
            Source_Data.append(Source_data)
            Source_Label.append(Source_label)
            Source_sub_label.append([sub_sub] * Source_data.shape[0])
        Source_Data=np.concatenate((Source_Data),axis=0)
        Source_Label=np.concatenate((Source_Label),axis=0)
        Source_sub_label=np.concatenate((Source_sub_label),axis=0)

        data,label,sub_label=[],[],[]
        if ch_choose==[]:
            pass
        else:
            Source_Data = Source_Data[:, ch_choose, :]
            Target_data = Target_data[:, ch_choose, :]
        data.append(Source_Data)
        data.append(Target_data)
        label.append(Source_Label)
        label.append(Target_label)
        sub_label.append(Source_sub_label)
        sub_label.append(Target_sub_label)
        return data, label, sub_label

    def loader_global_cross_double(self, sence, subject, dataclass, n_class, class_class,Subject, ch_choose):
        stim_name = ('ssvideo','video','ssmvep','cue')
        if self.dataset in ['online_hybrid']:
            if n_class[0] == 0:
                if sence in ['global_cross_double1']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject,
                                                                               stim_name_online[
                                                                                   n_class[1] - 1]) + '_online.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject,
                                                                              stim_name_online[
                                                                                  n_class[1] - 1]) + '_online.mat'
                if sence in ['global_cross_double2']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject,
                                                                               stim_name_online[
                                                                                   n_class[1] - 1]) + '_online.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject,
                                                                              stim_name_online[
                                                                                  n_class[1] - 1]) + '_online.mat'
            elif n_class[0] == 1:
                if sence in ['global_cross_double1']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject,
                                                                               stim_name_online[
                                                                                   n_class[2] - 1]) + '_online.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject,
                                                                              stim_name_online[
                                                                                  n_class[2] - 1]) + '_online.mat'
                if sence in ['global_cross_double2']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject,
                                                                               stim_name_online[
                                                                                   n_class[2] - 1]) + '_online.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject,
                                                                              stim_name_online[
                                                                                  n_class[2] - 1]) + '_online.mat'
        else:
            if n_class == []:
                if sence in ['global_cross_double1']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S2.mat'.format(dataclass, subject)
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S2.mat'.format(dataclass, subject)
                if sence in ['global_cross_double2']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S1.mat'.format(dataclass, subject)
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S1.mat'.format(dataclass, subject)
            elif n_class[0] == 0:
                if sence in ['global_cross_double1']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject,
                                                                               stim_name[n_class[1] - 1]) + '.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject,
                                                                              stim_name[n_class[1] - 1]) + '.mat'
                if sence in ['global_cross_double2']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject,
                                                                               stim_name[n_class[1] - 1]) + '.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject,
                                                                              stim_name[n_class[1] - 1]) + '.mat'
            elif n_class[0] == 1:
                if sence in ['global_cross_double1']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, subject,
                                                                               stim_name[n_class[2] - 1]) + '.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, subject,
                                                                              stim_name[n_class[2] - 1]) + '.mat'
                if sence in ['global_cross_double2']:
                    if subject < 10:
                        target_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, subject,
                                                                               stim_name[n_class[2] - 1]) + '.mat'
                    elif subject >= 10:
                        target_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, subject,
                                                                              stim_name[n_class[2] - 1]) + '.mat'

        target_data = hdf5storage.loadmat(target_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
        target_label = hdf5storage.loadmat(target_filepath, mat_dtype=True)['label']
        target_data = target_data.transpose(2, 0, 1)
        target_label = np.squeeze(target_label)
        if class_class==[]:
            class_class=np.unique(target_label)
        Target_data=np.concatenate(([target_data[target_label==class_class[i]] for i in range(len(class_class))]),axis=0)
        Target_label=np.concatenate(([[i]*len(target_label[target_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
        index = np.arange(len(Target_label))
        np.random.shuffle(index)
        Target_data=Target_data[index,:,:]
        Target_label=Target_label[index]
        Target_sub_label=[subject] * Target_data.shape[0]

        source_subject = Subject[0]
        if subject == self.config['Subject_choose'][0]:
            Source_Data, Source_Label, Source_sub_label=[],[],[]
            for sub_sub in source_subject:
                if self.dataset in ['online_hybrid']:
                    if n_class[0] == 0:
                        if sence in ['global_cross_double1']:
                            if sub_sub < 10:
                                source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, sub_sub,
                                                                                       stim_name_online[n_class[1] - 1]) + '.mat'
                            elif sub_sub >= 10:
                                source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, sub_sub,
                                                                                      stim_name_online[n_class[1] - 1]) + '.mat'
                        if sence in ['global_cross_double2']:
                            if sub_sub < 10:
                                source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, sub_sub,
                                                                                       stim_name_online[n_class[1] - 1]) + '.mat'
                            elif sub_sub >= 10:
                                source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, sub_sub,
                                                                                      stim_name_online[n_class[1] - 1]) + '.mat'
                    elif n_class[0] == 1:
                        if sence in ['global_cross_double1']:
                            if sub_sub < 10:
                                source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, sub_sub,
                                                                                       stim_name_online[n_class[1] - 1]) + '.mat'
                            elif sub_sub >= 10:
                                source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, sub_sub,
                                                                                      stim_name_online[n_class[1] - 1]) + '.mat'
                        if sence in ['global_cross_double2']:
                            if sub_sub < 10:
                                source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, sub_sub,
                                                                                       stim_name_online[n_class[1] - 1]) + '.mat'
                            elif sub_sub >= 10:
                                source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, sub_sub,
                                                                                      stim_name_online[n_class[1] - 1]) + '.mat'
                else:
                    if n_class==[]:
                        if sence in ['global_cross_double1']:
                            if sub_sub < 10:
                                source_filepath = self.filepath + '{}S0{}S1.mat'.format(dataclass, sub_sub)
                            elif sub_sub >= 10:
                                source_filepath = self.filepath + '{}S{}S1.mat'.format(dataclass, sub_sub)
                        if sence in ['global_cross_double2']:
                            if sub_sub < 10:
                                source_filepath = self.filepath + '{}S0{}S2.mat'.format(dataclass, sub_sub)
                            elif sub_sub >= 10:
                                source_filepath = self.filepath + '{}S{}S2.mat'.format(dataclass, sub_sub)
                    elif n_class[0]==0:
                        if sence in ['global_cross_double1']:
                            if sub_sub < 10:
                                source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass,sub_sub, stim_name[n_class[1]-1])+'.mat'
                            elif sub_sub >= 10:
                                source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                        if sence in ['global_cross_double2']:
                            if sub_sub < 10:
                                source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                            elif sub_sub >= 10:
                                source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                    elif n_class[0]==1:
                        if sence in ['global_cross_double1']:
                            if sub_sub < 10:
                                source_filepath = self.filepath + '{}S0{}S1_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                            elif sub_sub >= 10:
                                source_filepath = self.filepath + '{}S{}S1_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                        if sence in ['global_cross_double2']:
                            if sub_sub < 10:
                                source_filepath = self.filepath + '{}S0{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'
                            elif sub_sub >= 10:
                                source_filepath = self.filepath + '{}S{}S2_{}'.format(dataclass, sub_sub, stim_name[n_class[1]-1])+'.mat'

                source_data = hdf5storage.loadmat(source_filepath, mat_dtype=True)['data'][:, self.startpoint:self.endpoint, :]
                source_label = hdf5storage.loadmat(source_filepath, mat_dtype=True)['label']
                source_data = source_data.transpose(2, 0, 1)
                source_label = np.squeeze(source_label)
                Source_data=np.concatenate(([source_data[source_label==class_class[i]] for i in range(len(class_class))]),axis=0)
                Source_label=np.concatenate(([[i]*len(source_label[source_label==class_class[i]]) for i in range(len(class_class))]),axis=0)# 让标签从0开始排
                index = np.arange(len(Source_label))
                np.random.shuffle(index)
                Source_data = Source_data[index, :, :]
                Source_label=Source_label[index]
                Source_Data.append(Source_data)
                Source_Label.append(Source_label)
                Source_sub_label.append([sub_sub] * Source_data.shape[0])
            Source_Data=np.concatenate((Source_Data),axis=0)
            Source_Label=np.concatenate((Source_Label),axis=0)
            Source_sub_label=np.concatenate((Source_sub_label),axis=0)
            data,label,sub_label=[],[],[]
            if ch_choose==[]:
                pass
            else:
                Source_Data = Source_Data[:, ch_choose, :]
                Target_data = Target_data[:, ch_choose, :]
            data.append(Source_Data)
            data.append(Target_data)
            label.append(Source_Label)
            label.append(Target_label)
            sub_label.append(Source_sub_label)
            sub_label.append(Target_sub_label)
        else:
            data, label, sub_label = [], [], []
            if ch_choose == []:
                pass
            else:
                Target_data = Target_data[:, ch_choose, :]
            data.append(Target_data)
            label.append(Target_label)
            sub_label.append(Target_sub_label)
        return data, label, sub_label

    def ssmvep_hybrid_paradigm(self, subject, dataclass, sence, n_class=[], class_class=[], Subject=[], ch_choose=[]):
        '''
        subject:选择被试编号
        n_class:np数组,内包含[跨任务标识(0/1),跨任务类别[源,目]或[非跨任务域]] 
                1-ssvideo,2-video,3-ssmvep,4-cue
        dataclass:选择使用的数据类型（预处理不同，以后可以加入进来）
        sence:单场景非跨分类——"S1","S2"
              跨场景非跨人分类——"cross1","cross2"
              跨人非跨场景分类——"cross_subject1","cross_subject2"
              跨人跨场景分类——"cross_double1"
        class_class:np数组,[分类类别]
                1-left,2-right
        '''
        self.filepath = filepath_ssmvep_hybrid
        if sence in ['S1', 'S2']:
            data, label, sub_label=self.loader_single(subject=subject, dataclass=dataclass, sence=sence, n_class=n_class, class_class=[], ch_choose=ch_choose)
        if sence in ['cross1','cross2']:
            data,label,sub_label=self.loader_cross_sence(subject=subject, dataclass=dataclass, sence=sence, n_class=n_class,class_class=[], ch_choose=ch_choose)
        if sence in ['cross_task1', 'cross_task2']:
            data, label, sub_label = self.loader_cross_task(subject=subject, dataclass=dataclass, sence=sence,
                                                             n_class=n_class, class_class=[], ch_choose=ch_choose)
        if sence in ['cross_subject1', 'cross_subject2']:
            data,label,sub_label=self.loader_cross_subject(subject=subject, dataclass=dataclass, sence=sence, n_class=n_class,class_class=[],Subject=Subject, ch_choose=ch_choose)
        if sence in ['cross_double1', 'cross_double2']:
            data,label,sub_label=self.loader_cross_double(subject=subject, dataclass=dataclass, sence=sence, n_class=n_class,class_class=[],Subject=Subject, ch_choose=ch_choose)
        if sence in ['global_cross_double1', 'global_cross_double2']:
            data, label, sub_label = self.loader_global_cross_double(subject=subject, dataclass=dataclass, sence=sence,
                                                              n_class=n_class, class_class=[], Subject=Subject,
                                                              ch_choose=ch_choose)

        return data, label, sub_label

    def ssmvep_paradigm(self, subject,dataclass, sence, n_class=[], class_class=[],Subject=[],ch_choose=[]):
        '''
        subject:选择被试编号
        dataclass:选择使用的数据类型（预处理不同，以后可以加入进来）
        sence:单场景非跨分类——"S1","S2"
              跨场景非跨人分类——"cross1","cross2"
              跨人非跨场景分类——"cross_subject1","cross_subject2"
              跨人跨场景分类——"cross_double1"
        class_class:np数组,[分类类别]
                1-left,2-right,3-rest
        '''
        self.filepath = filepath_ssmvep
        if sence in ['S1', 'S2']:
            data,label,sub_label=self.loader_single(subject=subject,dataclass=dataclass, sence=sence, n_class=[],
                                                    class_class=class_class, ch_choose=ch_choose)
        if sence in ['cross1','cross2']:
            data,label,sub_label=self.loader_cross_sence(subject=subject,dataclass=dataclass, sence=sence, n_class=[],
                                                         class_class=class_class, ch_choose=ch_choose)
        if sence in ['cross_subject1','cross_subject2']:
            data,label,sub_label=self.loader_cross_subject(subject=subject,dataclass=dataclass, sence=sence, n_class=[],
                                                           class_class=class_class, Subject=Subject, ch_choose=ch_choose)
        if sence in ['cross_double1','cross_double2']:
            data,label,sub_label=self.loader_cross_double(subject=subject,dataclass=dataclass, sence=sence, n_class=[],
                                                          class_class=class_class, Subject=Subject, ch_choose=ch_choose)
        return data, label, sub_label

    def graz_paradigm(self, subject,dataclass, sence, n_class=[], class_class=[], Subject=[], ch_choose=[]):
        '''
        subject:选择被试编号
        dataclass:选择使用的数据类型（预处理不同，以后可以加入进来）
        sence:单场景非跨分类——"S1","S2"
              跨场景非跨人分类——"cross1","cross2"
              跨人非跨场景分类——"cross_subject1","cross_subject2"
              跨人跨场景分类——"cross_double1"
        class_class:np数组,[分类类别]
                1-left,2-right,3-rest
        '''
        self.filepath = filepath_graz
        if sence in ['S1', 'S2']:
            data,label,sub_label=self.loader_single(subject=subject,dataclass=dataclass, sence=sence, n_class=[],
                                                    class_class=class_class, ch_choose=ch_choose)
        if sence in ['cross1','cross2']:
            data,label,sub_label=self.loader_cross_sence(subject=subject,dataclass=dataclass, sence=sence, n_class=[],
                                                         class_class=class_class, ch_choose=ch_choose)
        if sence in ['cross_subject1','cross_subject2']:
            data,label,sub_label=self.loader_cross_subject(subject=subject,dataclass=dataclass, sence=sence, n_class=[],
                                                           class_class=class_class, Subject=Subject, ch_choose=ch_choose)
        if sence in ['cross_double1','cross_double2']:
            data,label,sub_label=self.loader_cross_double(subject=subject,dataclass=dataclass, sence=sence, n_class=[],
                                                          class_class=class_class, Subject=Subject, ch_choose=ch_choose)
        return data, label, sub_label


    def online_hybrid_paradigm(self, subject, dataclass, sence, n_class=[], class_class=[], Subject=[], ch_choose=[]):
        '''
        subject:选择被试编号
        n_class:np数组,内包含[跨任务标识(0/1),跨任务类别[源,目]或[非跨任务域]]
                1-ssvideo,2-video,3-ssmvep,4-cue
        dataclass:选择使用的数据类型（预处理不同，以后可以加入进来）
        sence:单场景非跨分类——"S1","S2",计算离线准确率
              跨场景非跨人分类——"cross1","cross2"，计算在线准确率
              跨人非跨场景分类——"cross_subject1","cross_subject2"，计算在线准确率
              跨人跨场景分类——"cross_double1"，计算在线准确率
        class_class:np数组,[分类类别]
                1-left,2-right
        '''
        self.filepath = filepath_online_hybrid
        if sence in ['S1', 'S2']:
            data,label,sub_label=self.loader_single(subject=subject, dataclass=dataclass, sence=sence,
                                                    n_class=n_class, class_class=[], ch_choose=ch_choose)
        if sence in ['cross1','cross2']:
            data,label,sub_label=self.loader_cross_sence(subject=subject, dataclass=dataclass, sence=sence,
                                                         n_class=n_class,class_class=[], ch_choose=ch_choose)
        if sence in ['cross_task1','cross_task2']:
            data,label,sub_label=self.loader_cross_task(subject=subject, dataclass=dataclass, sence=sence,
                                                         n_class=n_class,class_class=[], ch_choose=ch_choose)
        if sence in ['cross_subject1', 'cross_subject2']:
            data,label,sub_label=self.loader_cross_subject(subject=subject, dataclass=dataclass, sence=sence, n_class=n_class,class_class=[],Subject=Subject)
        if sence in ['cross_double1', 'cross_double2']:
            data,label,sub_label=self.loader_cross_double(subject=subject, dataclass=dataclass, sence=sence,
                                                          n_class=n_class,class_class=[],Subject=Subject, ch_choose=ch_choose)
        if sence in ['S1_online', 'S2_online']:
            data, label, sub_label = self.loader_single_online(subject=subject, dataclass=dataclass, sence=sence,
                                                             n_class=n_class, class_class=[], ch_choose=ch_choose)
        return data, label, sub_label

import os
from pathlib import Path
import yaml
from argparse import ArgumentParser
CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "configs")
DEFAULT_CONFIG = "ssmvep_hybrid.yaml"
Subject = list(range(1, 37+1))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    with open(os.path.join(CONFIG_DIR, args.config), 'rb') as f:
        config = yaml.safe_load(f)
    loader = Dataloader(config)
    for subject in Subject:
        data, label, sub_label = loader.loader_data(subject)
    # data, label, sub_label = loader.ssmvep_paradigm(sence='S1', subject=1, dataclass=1, class_class=[1,2], Subject=[list(range(1,20)),2])
    # data, label, sub_label = loader.ssmvep_hybrid_paradigm(sence='cross_task', subject=1, dataclass=1, n_class=[0,1,2], Subject=[list(range(1,20)),2])