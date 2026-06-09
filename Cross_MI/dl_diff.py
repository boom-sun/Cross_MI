import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from Cross_MI.auxiliary.sun_data_loader import Dataloader
from Cross_MI.auxiliary.sun_data_saver import Saver
import os
from pathlib import Path
import yaml
from argparse import ArgumentParser
from Cross_MI.auxiliary.basemodel import Basemodel
import numpy as np
from sklearn.model_selection import train_test_split
from Cross_MI.preprocess.preprocess import BandpassFilter
from torch.utils.data import TensorDataset
import torch

CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "Cross_MI\\configs")
DEFAULT_CONFIG = "ssmvep.yaml"
Subject = list(range(1, 22+1))

# 自定义数据集类
class EEGDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        """
        初始化EEG数据集
        Args:
            data_path: 数据路径
            mode: 'train', 'val' 或 'test'
        """
        # 这里假设数据已预处理并保存为numpy数组
        # 实际使用时需要根据您的数据格式进行调整
        if mode == 'train':
            self.data = np.load(f'{data_path}/train_data.npy')  # 形状: (样本数, 通道数, 时间点)
            self.labels = np.load(f'{data_path}/train_labels.npy')
        elif mode == 'val':
            self.data = np.load(f'{data_path}/val_data.npy')
            self.labels = np.load(f'{data_path}/val_labels.npy')
        else:
            self.data = np.load(f'{data_path}/test_data.npy')
            self.labels = np.load(f'{data_path}/test_labels.npy')

        # 转换为PyTorch张量
        self.data = torch.FloatTensor(self.data)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x形状: (批次大小, 通道数, 时间点)
        b, c, t = x.size()

        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x).view(b, c))

        # 合并两个分支
        out = avg_out + max_out
        # 应用sigmoid激活函数获取注意力权重
        scale = torch.sigmoid(out).unsqueeze(2)

        # 应用注意力权重
        return x * scale


# 时间注意力模块
class TemporalAttention(nn.Module):
    def __init__(self, in_timesteps, reduction_ratio=16):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_timesteps, in_timesteps // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_timesteps // reduction_ratio, in_timesteps),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x形状: (批次大小, 通道数, 时间点)
        b, c, t = x.size()

        # 计算每个时间点的均值
        avg_pool = torch.mean(x, dim=1)  # 形状: (批次大小, 时间点)
        # 计算每个时间点的最大值
        max_pool, _ = torch.max(x, dim=1)  # 形状: (批次大小, 时间点)

        # 平均池化分支
        avg_out = self.fc(avg_pool)
        # 最大池化分支
        max_out = self.fc(max_pool)

        # 合并两个分支
        out = avg_out + max_out
        # 应用sigmoid激活函数获取注意力权重
        scale = torch.sigmoid(out).unsqueeze(1)

        # 应用注意力权重
        return x * scale


# 主干网络 - 时空注意力EEGNet
class STAttentionEEGNet(nn.Module):
    def __init__(self, n_channels, n_timesteps, n_classes):
        super(STAttentionEEGNet, self).__init__()

        # 第一卷积层：空间滤波
        self.conv1 = nn.Conv2d(1, 16, (1, n_channels), padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)

        # 深度可分离卷积：时间卷积
        self.depthwise_conv = nn.Conv2d(16, 32, (n_timesteps, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.5)

        # 通道注意力模块
        self.channel_attention = ChannelAttention(32)

        # 分离卷积
        self.separable_conv = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.5)

        # 时间注意力模块
        self.temporal_attention = TemporalAttention(32)  # 需要根据实际尺寸调整

        # 分类层
        self.flatten = nn.Flatten()
        # 计算全连接层的输入尺寸
        self.fc_input_size = self._get_fc_input_size(n_channels, n_timesteps)
        self.classifier = nn.Linear(self.fc_input_size, n_classes)

    def _get_fc_input_size(self, n_channels, n_timesteps):
        # 创建一个虚拟输入以计算全连接层的输入尺寸
        x = torch.randn(1, 1, n_timesteps, n_channels)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.channel_attention(x)
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = self.temporal_attention(x)
        return x.numel() // x.shape[0]  # 除以批次大小

    def forward(self, x):
        # 输入x形状: (批次大小, 通道数, 时间点)
        # 添加一个维度以适应卷积层输入要求
        x = x.unsqueeze(1)  # 形状: (批次大小, 1, 时间点, 通道数)

        # 第一卷积层：空间滤波
        x = self.conv1(x)  # 形状: (批次大小, 16, 时间点, 1)
        x = self.batchnorm1(x)

        # 深度可分离卷积：时间卷积
        x = self.depthwise_conv(x)  # 形状: (批次大小, 32, 1, 1)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        # 调整形状以适配注意力模块
        x = x.squeeze(2).squeeze(2)  # 形状: (批次大小, 32)
        # 但我们需要 (批次大小, 通道数, 时间点) 的形状
        # 这里需要根据网络的实际输出调整

        # 由于网络结构可能需要调整，这里简化处理
        # 实际应用中可能需要调整网络结构以确保维度匹配

        # 通道注意力
        x = self.channel_attention(x)

        # 分离卷积
        # 这里需要重新调整x的形状以适应后续层
        # 具体实现可能需要根据实际数据维度进行调整

        # 时间注意力
        x = self.temporal_attention(x)

        # 分类
        x = self.flatten(x)
        x = self.classifier(x)

        return x


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')

    return train_losses, val_losses, val_accuracies


# 可视化注意力权重
def visualize_attention(model, dataloader, device, n_examples=5):
    model.eval()
    attention_maps = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= n_examples:
                break

            inputs = inputs.to(device)

            # 前向传播并获取中间层输出（需要修改模型以返回注意力权重）
            # 这里需要根据实际模型结构调整
            # 假设我们有一个方法可以返回注意力权重
            # outputs, channel_att, temporal_att = model.get_attention(inputs)

            # 存储注意力权重用于可视化
            # attention_maps.append((channel_att.cpu(), temporal_att.cpu()))
            pass

    # 绘制注意力权重
    # 这里需要根据实际注意力权重格式进行可视化
    print("注意：需要实现get_attention方法以返回注意力权重")


# 主函数
def main(X_train,X_val,X_test,y_train,y_val,y_test):
    # 设置超参数
    n_channels = 26  # 根据您的EEG通道数调整
    n_timesteps = 1000  # 根据您的采样率和试验时长调整
    n_classes = 2  # 左手 vs 右手
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 将 NumPy 数组转换为 PyTorch 张量
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = STAttentionEEGNet(n_channels, n_timesteps, n_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # 可视化注意力权重
    visualize_attention(model, test_loader, device)

    # 在测试集上评估模型
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f'测试集准确率: {test_accuracy:.2f}%')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    with open(os.path.join(CONFIG_DIR, args.config), 'rb') as f:
        config = yaml.safe_load(f)

    loader = Dataloader(config)
    lambda_diff = 1
    Acc_all = []
    for subject in Subject:
        data, label, sub_label = loader.loader_data(subject)
        bandpassfilter = BandpassFilter(fs=config['srate'], lowcut=config['fre_win'][0], highcut=config['fre_win'][1])
        bandpassfilter.fit()
        X=bandpassfilter.transform(data)
        y=label
        X=X[:,14:40,:]
        Acc=[]
        for fold in range(10):
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42  # 设置随机种子保证结果可重现
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=0.2,
                random_state=42  # 同样设置随机种子
            )
            main(X_train, X_val, X_test, y_train, y_val, y_test)
