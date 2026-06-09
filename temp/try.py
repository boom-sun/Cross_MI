# ===== 放在文件最顶部、任何 import torch 之前 =====
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # 让设备序号与 nvidia-smi 对齐（可选）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"         # 只暴露第 0 张卡（可选）

from Cross_MI.auxiliary.sun_data_loader import Dataloader
from Cross_MI.auxiliary.sun_data_saver import Saver
from pathlib import Path
import yaml
from argparse import ArgumentParser
from Cross_MI.auxiliary.basemodel import Basemodel
import numpy as np
import torch
from braindecode.models import ATCNet
from braindecode.classifier import EEGClassifier
from braindecode.util import set_random_seeds
import torch.optim as optim
from torch import nn
from sklearn.model_selection import train_test_split
from skorch.callbacks import Callback

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)

CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "Cross_MI", "configs")
DEFAULT_CONFIG = "ssmvep_hybrid.yaml"
Subject = list(range(1, 37 + 1))


def zscore_per_channel(train_x, x):
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = train_x.std(axis=(0, 2), keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (x - mean) / std


# ---- 批次级断言与显存打印 ----
class AssertBatchOnCuda(Callback):
    def on_train_batch_begin(self, net, batch, **kwargs):
        X, y = batch
        def first_tensor(obj):
            if isinstance(obj, torch.Tensor):
                return obj
            if isinstance(obj, (list, tuple)):
                for it in obj:
                    t = first_tensor(it)
                    if t is not None:
                        return t
            if isinstance(obj, dict):
                for it in obj.values():
                    t = first_tensor(it)
                    if t is not None:
                        return t
            return None
        t = first_tensor(X)
        if t is not None and torch.cuda.is_available():
            assert t.is_cuda, f"Batch tensor 在 {t.device}，期望在 cuda"
    def on_epoch_end(self, net, **kwargs):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**2)
            resv  = torch.cuda.memory_reserved() / (1024**2)
            print(f"[GPU mem] allocated={alloc:.1f}MB reserved={resv:.1f}MB")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    with open(os.path.join(CONFIG_DIR, args.config), "rb") as f:
        config = yaml.safe_load(f)

    loader = Dataloader(config)
    _ = Basemodel(config)
    saver = Saver(config, Subject)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"🧠 Using device: cuda:0 -> {torch.cuda.get_device_name(0)}")
        print(f"CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
    else:
        print("🧠 Using device: cpu")

    for subject in Subject:
        print(f"\n🚀 Training subject {subject}...")

        data, label, sub_label = loader.loader_data(subject)
        X_train, y_train = data[0], label[0]
        X_test, y_test = data[1], label[1]

        # 类型统一 + 连续内存，避免额外复制
        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        X_test  = np.ascontiguousarray(X_test,  dtype=np.float32)
        y_train = np.ascontiguousarray(np.array(y_train).astype("int64"))
        y_test  = np.ascontiguousarray(np.array(y_test).astype("int64"))

        # 标准化
        X_train = zscore_per_channel(X_train, X_train)
        X_test = zscore_per_channel(X_train, X_test)

        # 划分验证集（你已手动切分，后面会把 skorch 的 train_split 关掉）
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # 模型初始化
        n_channels = X_train.shape[1]
        input_window_seconds = X_train.shape[2] / config["srate"]
        n_classes = int(np.unique(y_train).size)

        model = ATCNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            input_window_seconds=input_window_seconds,
            sfreq=config["srate"],
        )

        # —— 关键：让 Skorch 负责把模型和 batch 都搬到 device；并让 DataLoader 不当瓶颈 ——
        num_workers = min(8, max(1, (os.cpu_count() or 2) - 1))
        clf = EEGClassifier(
            module=model,
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            optimizer__lr=1e-3,
            batch_size=512 if device.type == "cuda" else 64,  # 适当加大 batch，提升利用率
            max_epochs=200,
            device=device,             # 明确指定到 cuda:0
            train_split=None,          # 你已手动切出验证集，避免重复验证
            iterator_train__pin_memory=(device.type == "cuda"),
            iterator_valid__pin_memory=(device.type == "cuda"),
            iterator_train__num_workers=num_workers,
            iterator_valid__num_workers=num_workers,
            iterator_train__prefetch_factor=2,
            iterator_valid__prefetch_factor=2,
            iterator_train__persistent_workers=True,
            iterator_valid__persistent_workers=True,
            callbacks=[AssertBatchOnCuda()],
        )

        # （可选）在 PyTorch 2 上进一步加速
        # if torch.__version__.startswith("2"):
        #     clf.set_params(module=torch.compile(model))  # 需要 PyTorch 2.x

        # 初始化（fit 会自动 initialize，这里显式无妨）
        clf.initialize()
        print("模型参数所在设备：", next(clf.module_.parameters()).device)

        # 手动测试 GPU 是否可用（确保能在 nvidia-smi 看到波动）
        if device.type == "cuda":
            x = torch.randn(2048, 2048, device=device)
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            print("GPU 测试矩阵运算 OK，结果均值：", y.mean().item())

        # 开始训练
        clf.fit(X_train, y_train)

        # 评估
        val_acc = clf.score(X_val, y_val)
        test_acc = clf.score(X_test, y_test)
        print(f"✅ Validation accuracy (subject {subject}): {val_acc * 100:.2f}%")
        print(f"🎯 Test accuracy (subject {subject}): {test_acc * 100:.2f}%")

        print("模型参数最终设备：", next(clf.module_.parameters()).device)

        # （可选）释放一下，便于多被试循环看显存
        if device.type == "cuda":
            torch.cuda.empty_cache()

