import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_

# ---------------- CNN 特征提取器 ----------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=21, num_features=256, p_drop=0.3):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=25, stride=2, padding=12)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=25, stride=2, padding=12)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=25, stride=2, padding=12)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, num_features)
        self.drop  = nn.Dropout(p_drop)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 21, T]
        x = self.drop(F.relu(self.bn1(self.conv1(x))))  # [B, 32, ·]
        x = self.drop(F.relu(self.bn2(self.conv2(x))))  # [B, 64, ·]
        x = self.drop(F.relu(self.bn3(self.conv3(x))))  # [B, 128, ·]
        x = self.pool(x)                                # [B, 128, 1]
        x = torch.flatten(x, 1)                         # [B, 128]
        x = F.relu(self.fc(x))                          # [B, num_features]
        return x

# ---------------- 特征维注意力（稳定版） ----------------
class SelfAttention(nn.Module):
    def __init__(self, input_dim: int, p_drop=0.1):
        super(SelfAttention, self).__init__()
        d = input_dim
        self.to_score = nn.Linear(d, d, bias=False)
        self.to_value = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(p_drop)

        nn.init.xavier_normal_(self.to_score.weight)
        nn.init.xavier_normal_(self.to_value.weight)

    def forward(self, x):
        # x: [B, D]
        scores = self.to_score(x) / (x.size(-1) ** 0.5)  # [B, D]
        attn = F.softmax(scores, dim=-1)                 # [B, D]
        values = self.to_value(x)                        # [B, D]
        out = self.drop(attn * values)                   # [B, D]
        return out

# ---------------- 主网络 ----------------
class MultiModalModel(nn.Module):
    def __init__(self, input_dim=256, num_classes=2, p_drop=0.5):
        super(MultiModalModel, self).__init__()
        self.cnn = CNNFeatureExtractor(in_channels=21, num_features=input_dim, p_drop=0.3)
        self.combined_dim = input_dim * 2  # 拼接两个模态后的维度
        self.attention = SelfAttention(self.combined_dim, p_drop=0.1)
        self.lstm = nn.LSTM(self.combined_dim, 128, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(128, num_classes),
        )

        # 线性层初始化
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1, x2):
        # x1, x2: [B, 21, T]
        f1 = self.cnn(x1)                          # [B, input_dim]
        f2 = self.cnn(x2)                          # [B, input_dim]
        combined = torch.cat([f1, f2], dim=-1)     # [B, 2*input_dim]
        attended = self.attention(combined)        # [B, 2*input_dim]
        lstm_out, (h_n, c_n) = self.lstm(attended.unsqueeze(1))  # [B, 1, 128]
        feat = h_n[-1]                             # [B, 128]
        logits = self.head(feat)                   # [B, num_classes]
        return logits

# ---------------- 训练/评估/预测 ----------------
class EEGModel:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

        # 低版本 PyTorch 可能不支持 label_smoothing
        try:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        except TypeError:
            self.criterion = nn.CrossEntropyLoss()

        # 低版本可能没有 AdamW
        try:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        except AttributeError:
            self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-4)

        # 一些旧版不支持 verbose，去掉即可
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
            # 可选: cooldown=1, min_lr=1e-6
        )
    def fit(self, train_loader, val_loader, epochs=50, early_stopping_patience=10, max_grad_norm=1.0):
        best_val = float('inf')
        patience = 0
        prev_lr = self.optimizer.param_groups[0]['lr']  # 跟踪学习率

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for data_batch, label_batch in train_loader:
                data_batch, label_batch = data_batch.to(self.device), label_batch.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(data_batch, data_batch)
                loss = self.criterion(logits, label_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                running_loss += loss.item()
                _, pred = torch.max(logits, 1)
                correct += (pred == label_batch).sum().item()
                total += label_batch.size(0)

            train_loss = running_loss / max(1, len(train_loader))
            train_acc = 100.0 * correct / max(1, total)

            # 验证 + 调度器
            val_loss, val_acc = self.evaluate(val_loader)
            self.scheduler.step(val_loss)

            curr_lr = self.optimizer.param_groups[0]['lr']
            if curr_lr < prev_lr:
                print(f"[Scheduler] LR reduced: {prev_lr:.2e} -> {curr_lr:.2e}")
            prev_lr = curr_lr

            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | "
                  f"LR {curr_lr:.2e}")

            # Early Stopping
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}. Best val loss: {best_val:.4f}")
                    self.model.load_state_dict(best_state)
                    break

    def evaluate(self, val_loader):
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data_batch, label_batch in val_loader:
                data_batch, label_batch = data_batch.to(self.device), label_batch.to(self.device)
                logits = self.model(data_batch, data_batch)
                loss = self.criterion(logits, label_batch)
                val_loss += loss.item()
                _, pred = torch.max(logits, 1)
                correct += (pred == label_batch).sum().item()
                total += label_batch.size(0)
        val_loss = val_loss / max(1, len(val_loader))
        val_acc = 100.0 * correct / max(1, total)
        return val_loss, val_acc

    def predict(self, test_loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data_batch, label_batch in test_loader:
                data_batch, label_batch = data_batch.to(self.device), label_batch.to(self.device)
                logits = self.model(data_batch, data_batch)
                _, pred = torch.max(logits, 1)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(label_batch.cpu().numpy())
        all_preds = np.concatenate(all_preds) if all_preds else np.array([])
        all_labels = np.concatenate(all_labels) if all_labels else np.array([])
        if all_preds.size > 0:
            acc = (all_preds == all_labels).mean() * 100
            print(f"Test Accuracy: {acc:.2f}%")
        else:
            print("Test set is empty.")
        return all_preds, all_labels

# ---------------- 自定义数据集 ----------------
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # [N, 21, T]
        self.labels = labels  # [N]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
