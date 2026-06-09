# --config=./train.yaml

import os
import time
import random
import argparse
from datetime import datetime
import torch.nn.functional as F

import yaml
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
import higher
import copy

import utils as ut
from model import get_model
from data_generator import task_generator


#  ------------------------------------------------------------------------------------------

#
# def main(config, model, iteration, device, dataset, mini_task_size, ti, Train=False):
#     # Create tasks
#     srcdat, srclbl,src_slbl, zrdat, zrlbl, zr_slbl = (dataset[0], dataset[1], dataset[2],
#                                                       dataset[3], dataset[4], dataset[5])
#
#     if Train:
#         tasks_data, tasks_labels = task_generator(config, srcdat, srclbl, src_slbl, iteration, mini_task_size, ti, Subject)
#     else:
#         tasks_data, tasks_labels = list([list([srcdat, zrdat])]), list([list([srclbl, zrlbl])])
#
#     # To control update parameter
#     pytorch_model = model.module
#     pytorch_model.to(device)
#     pytorch_model.train()
#     body_params = []
#     head_params = []
#     for name, param in pytorch_model.named_parameters():
#         if 'classifier' in name:
#             head_params.append(param)
#         elif config['backbone'] not in name:  # 确保 backbone 也不在 body 中
#             body_params.append(param)
#     # outer optimizer
#     meta_optimizer = torch.optim.Adam([{'params': body_params, 'lr': config['train']['meta_lr']},
#                                        {'params': head_params, 'lr': config['train']['meta_lr']
#                                        if iteration != 0 and (iteration + 1) % config['train'][
#                                            'freeze_epoch'] == 0 else 0}])
#
#     inner_optimizer = torch.optim.Adam([{'params': body_params, 'lr': config['train']['task_lr']},
#                                         {'params': head_params, 'lr': config['train']['task_lr']
#                                         if iteration != 0 and (iteration + 1) % config['train'][
#                                             'freeze_epoch'] == 0 else 0}])
#
#     meta_optimizer.zero_grad()
#     inner_optimizer.zero_grad()
#
#     total_loss = torch.tensor(0., device=device)
#     accuracy = torch.tensor(0., device=device)
#
#     for task_idx, (task_data, task_label) in enumerate(zip(tasks_data, tasks_labels)):
#         with higher.innerloop_ctx(pytorch_model.float(), inner_optimizer, copy_initial_weights=False) as (fnet, diffopt):
#             outer_loss = torch.tensor(0., device=device)
#             spt_data, qry_data = task_data[0].float(), task_data[1].float()
#             spt_label, qry_label = task_label[0].long(), task_label[1].long()
#
#             src_tensor = TensorDataset(spt_data, spt_label)
#             src_loader = DataLoader(src_tensor,
#                                     batch_size=config['train']['inner_batch_size'] if Train else config['train'][
#                                         'test_batch_size'], shuffle=True, drop_last=False)
#
#             if Train:
#                 for batch_idx, (inputs, labels) in enumerate(src_loader):
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     spt_logits = fnet(inputs)
#                     spt_loss = F.cross_entropy(spt_logits, labels)
#                     diffopt.step(spt_loss)
#
#                 inputs, labels = qry_data.to(device), qry_label.to(device)
#                 query_logit = fnet(inputs)
#                 task_loss = F.cross_entropy(query_logit, labels)
#
#                 outer_loss = outer_loss + task_loss# 直接累积到 outer_loss，保持计算图
#                 with torch.no_grad():
#                     accuracy += ut.get_accuracy(query_logit, labels)# 只在计算准确率时使用 no_grad
#                 total_loss = total_loss + outer_loss # 累积到 total_loss，保持计算图
#
#                 if task_idx != 0 and (task_idx+1) % config['train']['meta_batch_size'] == 0 or (task_idx + 1) == len(
#                         tasks_data):
#                     total_loss.div_(config['train']['meta_batch_size'])
#                     total_loss.backward()
#                     meta_optimizer.step()
#                     # total_loss = torch.tensor(0.).to(device)
#                     total_loss = 0.0
#
#             else:
#                 for batch_idx, (inputs, labels) in enumerate(src_loader):
#                     inner_optimizer.zero_grad()
#                     inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
#                     logit = pytorch_model(inputs)
#                     loss = criterion(logit, labels)
#                     loss.backward()
#                     inner_optimizer.step()
#
#                 inputs, labels = qry_data.to(device), qry_label.to(device)
#                 query_logit = pytorch_model(inputs)
#                 outer_loss += F.cross_entropy(query_logit, labels)
#
#                 with torch.no_grad():
#                     accuracy += ut.get_accuracy(query_logit, labels)
#                     total_loss += outer_loss
#
#     if Train:
#         accuracy.div_(task_idx + 1)
#         total_loss.div_(task_idx + 1)
#
#     return total_loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()

def main(config, model, iteration, device, dataset, mini_task_size, ti, Train=False):
    """
    config       : 训练配置（包含 train 字段）
    model        : 可能是 DataParallel 包过的模型
    iteration    : 当前 meta-iteration 计数（从 0 开始）
    device       : cuda / cpu
    dataset      : [srcdat, srclbl, src_slbl, zrdat, zrlbl, zr_slbl]
    mini_task_size : task_generator 里支持集切分粒度（你现在一般用 2880，不切块）
    ti           : 当前外层 subject id
    Train        : True = meta-train, False = meta-eval（验证）
    """

    # -----------------------------
    # 1. 解包数据
    # -----------------------------
    srcdat, srclbl, src_slbl, zrdat, zrlbl, zr_slbl = (
        dataset[0], dataset[1], dataset[2],
        dataset[3], dataset[4], dataset[5]
    )

    if Train:
        # 使用元任务生成器（可内部做“6 个 query subject”采样）
        tasks_data, tasks_labels = task_generator(
            config, srcdat, srclbl, src_slbl,
            iteration, mini_task_size, ti, Subject
        )
    else:
        # 验证阶段：用原始代码的逻辑，支持 = 源数据，查询 = 目标数据
        tasks_data  = [ [srcdat, zrdat] ]
        tasks_labels = [ [srclbl, zrlbl] ]

    # -----------------------------
    # 2. 把 DataParallel 拿掉 & 模型放到设备
    # -----------------------------
    pytorch_model = model.module                 # skorch 的内部 torch 模型
    pytorch_model = pytorch_model.float()        # ✅ 先统一成 float32
    pytorch_model = pytorch_model.to(device)     # ✅ 再搬到 GPU / CPU

    if Train:
        pytorch_model.train()
    else:
        pytorch_model.train()  # 验证阶段我们会在 copy_model 上操作，这里保持一致就行

    # -----------------------------
    # 3. 拆 body / head 参数（给 meta_optimizer 用）
    # -----------------------------
    body_params = []
    head_params = []
    for name, param in pytorch_model.named_parameters():
        if 'classifier' in name:
            head_params.append(param)
        else:
            body_params.append(param)

    # -----------------------------
    # 4. 定义优化器
    #    - inner_optimizer: 给 higher 用（只需要 lr 等超参数）
    #    - meta_optimizer : outer-loop 更新真正的 pytorch_model
    # -----------------------------
    if Train:
        # inner-loop 用 SGD 更稳定，也更符合原始 MAML 写法
        inner_optimizer = torch.optim.SGD(
            pytorch_model.parameters(),
            lr=config['train']['task_lr']
        )

        # 是否冻结 head，可根据 iteration 和 freeze_epoch 控制
        if iteration != 0 and (iteration + 1) % config['train']['freeze_epoch'] == 0:
            head_lr = config['train']['meta_lr']
        else:
            head_lr = config['train']['meta_lr']

        meta_optimizer = torch.optim.Adam(
            [
                {'params': body_params, 'lr': config['train']['meta_lr']},
                {'params': head_params,  'lr': head_lr},
            ],
            weight_decay=config['train'].get('w_decay', 0.0)
        )
        meta_optimizer.zero_grad()

    # 内外都会用的记录量
    total_loss = 0.0
    total_acc = 0.0

    # -----------------------------
    # 5. Train（使用 higher.MAML）
    # -----------------------------
    if Train:
        meta_batch_size = config['train']['meta_batch_size']
        accum_loss = None   # 用来在 meta-batch 内累积几次 task_loss
        accum_count = 0

        for task_idx, (task_data, task_label) in enumerate(zip(tasks_data, tasks_labels)):
            spt_data, qry_data = task_data[0].float(), task_data[1].float()
            spt_label, qry_label = task_label[0].long(), task_label[1].long()

            # support loader（inner-loop 迭代用）
            src_tensor = TensorDataset(spt_data, spt_label)
            src_loader = DataLoader(
                src_tensor,
                batch_size=config['train']['inner_batch_size'],
                shuffle=True,
                drop_last=False
            )

            # 每个 task 都创建一个 inner-loop 上下文
            with higher.innerloop_ctx(
                pytorch_model,
                inner_optimizer,
                copy_initial_weights=False
            ) as (fnet, diffopt):

                # ------- inner-loop：在 support 上做快速适应 -------
                for inputs, labels in src_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits_spt = fnet(inputs)
                    loss_spt = F.cross_entropy(logits_spt, labels)
                    diffopt.step(loss_spt)

                # ------- outer-loop：在 query 上评估该 task -------
                qry_inputs, qry_labels = qry_data.to(device), qry_label.to(device)
                qry_logits = fnet(qry_inputs)
                task_loss = F.cross_entropy(qry_logits, qry_labels)

            # 统计用的 total_loss / total_acc（不参与梯度）
            total_loss += task_loss.detach().item()
            with torch.no_grad():
                total_acc += ut.get_accuracy(qry_logits, qry_labels).item()

            # meta-batch 内累积 loss（这里保持计算图）
            if accum_loss is None:
                accum_loss = task_loss
            else:
                accum_loss = accum_loss + task_loss
            accum_count += 1

            # 当累积到 meta_batch_size 或到了最后一个 task，就做一次 meta 更新
            if (accum_count == meta_batch_size) or ((task_idx + 1) == len(tasks_data)):
                meta_optimizer.zero_grad()
                meta_loss = accum_loss / accum_count
                meta_loss.backward()
                meta_optimizer.step()

                accum_loss = None
                accum_count = 0

        num_tasks = len(tasks_data)
        avg_loss = total_loss / num_tasks
        avg_acc  = total_acc / num_tasks

        return avg_loss, avg_acc

    # -----------------------------
    # 6. Eval / Validation（不用 higher，不算 meta-grad）
    # -----------------------------
    else:
        # 这里一般在外层先 deep copy 了 model，传进来的是 copy_model
        # 所以你可以在这个 copy 上自由做 inner 训练，不会影响原模型
        criterion = nn.CrossEntropyLoss()
        for task_idx, (task_data, task_label) in enumerate(zip(tasks_data, tasks_labels)):
            spt_data, qry_data = task_data[0].float(), task_data[1].float()
            spt_label, qry_label = task_label[0].long(), task_label[1].long()

            src_tensor = TensorDataset(spt_data, spt_label)
            src_loader = DataLoader(
                src_tensor,
                batch_size=config['train']['test_batch_size'],
                shuffle=True,
                drop_last=False
            )

            # 在 copy_model 上做普通的 finetune（不需要 higher，也不需要 meta-grad）
            inner_optimizer = torch.optim.SGD(
                pytorch_model.parameters(),
                lr=config['train']['task_lr']
            )

            # ------- 在 support 上做若干步微调 -------
            for inputs, labels in src_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                inner_optimizer.zero_grad()
                logits = pytorch_model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                inner_optimizer.step()

            # ------- 在 query 上评估 -------
            inputs_q = qry_data.to(device)
            labels_q = qry_label.to(device)
            with torch.no_grad():
                qry_logits = pytorch_model(inputs_q)
                outer_loss = F.cross_entropy(qry_logits, labels_q)
                total_loss += outer_loss.item()
                total_acc  += ut.get_accuracy(qry_logits, labels_q).item()

        num_tasks = len(tasks_data)
        avg_loss = total_loss / num_tasks
        avg_acc  = total_acc / num_tasks

        return avg_loss, avg_acc

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

from Cross_MI.auxiliary.sun_data_loader import Dataloader
from Cross_MI.auxiliary.sun_data_saver import Saver
import os
from pathlib import Path
import yaml
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder
import numpy as np

CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "Cross_MI\\configs")
DEFAULT_CONFIG = "ssmvep_hybrid.yaml"
Subject = list(range(1, 37+1))
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    with open(os.path.join(CONFIG_DIR, args.config), 'rb') as f:
        config_1 = yaml.safe_load(f)
    loader = Dataloader(config_1)

    now = datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument('--date', default=now.strftime('%Y-%m-%d'), help="Please do not enter any value.")
    parser.add_argument('--time', default=now.strftime('%H:%M:%S'), help="Please do not enter any value.")
    parser.add_argument('--config', default='train.yaml', help='configuration file')

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r', encoding='UTF8'), Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_path = str(os.getcwd())

    # ✅ 新增：控制多久验证一次（比如每 5 轮做一次验证）
    val_interval = config['train'].get('val_interval', 5)  # 可以在 train.yaml 里加一个 val_interval: 5

    for ti in Subject:  # 还是对每个被试单独训练一个模型
        default_path = config['save_path']

        if config['backbone'] == 'EEGNet':
            save_path = f'{current_path}/{default_path}/EEGNet/'
        elif config['backbone'] == 'DeepConvNet':
            save_path = f'{current_path}/{default_path}/DeepConvNet/'

        save_result_path = '{}/subject0{}/accuracy/'.format(save_path, str(ti))
        save_trainmodel_path = '{}/subject0{}/trained_model/'.format(save_path, str(ti))

        print("Save Path =", save_result_path)
        ut.create_dir(save_result_path)
        ut.create_dir(save_trainmodel_path)

        criterion = nn.CrossEntropyLoss()

        # 加载当前被试的数据
        data, label, sub_label = loader.loader_data(ti)
        _le = LabelEncoder()
        srcdat_, srclbl_, srcslbl_ = (
            torch.tensor(data[0]),
            torch.tensor(_le.fit_transform(label[0].astype(np.int64))),
            sub_label[0]
        )
        zrdat_, zrlbl_, zrslbl_ = (
            torch.tensor(data[1]),
            torch.tensor(_le.fit_transform(label[1].astype(np.int64))),
            sub_label[1]
        )

        model = get_model(args, device, config, ti, srcdat_, srclbl_, current_path)
        all_dataset = [srcdat_, srclbl_, srcslbl_, zrdat_, zrlbl_, zrslbl_]

        train_iter = []
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        for meta_iteration in range(config['train']['batch_iter']):
            iter_time = time.time()
            print('Meta iteration =', meta_iteration + 1)
            if meta_iteration != 0 and (meta_iteration + 1) % config['train']['freeze_epoch'] == 0:
                print("Update head parameters")

            mini_task_size = config['train']['mini_task_size']

            # ✅ 1. 只做一次 meta-train（最耗时的部分）
            meta_train_loss, meta_train_acc = main(
                config, model, meta_iteration, device,
                all_dataset, mini_task_size, ti,
                Train=True
            )

            train_iter.append(meta_iteration + 1)
            train_loss.append(meta_train_loss)
            train_acc.append(meta_train_acc)

            # ✅ 2. 不是每一轮都验证，而是每 val_interval 轮验证一次
            if (meta_iteration + 1) % val_interval == 0 or (meta_iteration + 1) == config['train']['batch_iter']:
                print("Running validation ...")
                # 用当前 model 做一个拷贝，防止在 val 中意外改到参数
                copy_model = copy.deepcopy(model)
                meta_val_loss, meta_val_acc = main(
                    config, copy_model, meta_iteration, device,
                    all_dataset, mini_task_size, ti,
                    Train=False
                )

                val_loss.append(meta_val_loss)
                val_acc.append(meta_val_acc)

                # ✅ 只在做验证的轮次导出 val 结果
                ut.dataexport(args, config, ti, train_iter, train_loss, train_acc,
                              save_result_path, mode='train')
                ut.dataexport(args, config, ti, train_iter, val_loss, val_acc,
                              save_result_path, mode='val')

                print("Train Iter Loss = {:.4f}".format(meta_train_loss))
                print("Train Iter Acc = {:.4f}".format(meta_train_acc))
                print("Validation Iter Loss = {:.4f}".format(meta_val_loss))
                print("Validation Iter Acc = {:.4f}".format(meta_val_acc))
            else:
                # ✅ 只在训练轮次更新 train 曲线，先不导出 val
                ut.dataexport(args, config, ti, train_iter, train_loss, train_acc,
                              save_result_path, mode='train')

                print("Train Iter Loss = {:.4f}".format(meta_train_loss))
                print("Train Iter Acc = {:.4f}".format(meta_train_acc))

            m, s = divmod(time.time() - iter_time, 60)
            h, m = divmod(m, 60)
            print("Iteration total_time = {} mins {:.6} secs".format(m, s))
            print("=" * 30)

        # 训练结束，保存模型
        filename = os.path.join(
            '{}/epoch{}_subject0{}_model_state_dict.pt'.format(
                save_trainmodel_path, str(meta_iteration + 1), str(ti)
            )
        )
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)
