# Cross_MI/models/eeg_gnn_21.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_layers import GraphConv, GATv2
from .graph_utils import build_adjacency_21


# -------------------------
# 1. 时域编码 + 通道注意力
# -------------------------
class DepthwiseTemporal(nn.Module):
    """
    通道内时域表征：Depthwise Conv1d + 全局池化
    输入:  [B, C, T] -> 输出: [B, C, D]
    """
    def __init__(self, C, D=32, k=64, drop=0.1):
        super().__init__()
        self.dw = nn.Conv1d(
            in_channels=C,
            out_channels=C * D,
            kernel_size=k,
            groups=C,
            padding=k // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(C * D)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.C, self.D = C, D

    def forward(self, x):
        # x: [B, C, T]
        h = self.dw(x)                      # [B, C*D, T]
        h = self.bn(h)
        h = self.act(h)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)  # [B, C*D]
        h = self.drop(h)
        return h.view(x.size(0), self.C, self.D)


class ChannelSE(nn.Module):
    """
    简单的通道注意力（Squeeze-Excitation）:
    对每个通道生成一个权重，强调信息更丰富的通道。
    """
    def __init__(self, C, r=4):
        super().__init__()
        self.fc1 = nn.Linear(C, C // r, bias=True)
        self.fc2 = nn.Linear(C // r, C, bias=True)

    def forward(self, x):
        # x: [B, C, F]
        w = x.mean(dim=2)          # [B, C]
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))  # [B, C]
        return x * w.unsqueeze(-1)      # [B, C, F]


# -------------------------
# 2. 图 GNN 主干
# -------------------------
class EEGGraphBackbone21(nn.Module):
    """
    Temporal(通道内) -> SE 通道注意力 -> GNN * L -> 池化 -> 分类
    支持 gnn_type = 'gcn' 或 'gat'
    """
    def __init__(self, C=21, T=1000, n_classes=2,
                 D=32, gnn_hidden=64, gnn_layers=2,
                 gnn_type="gcn", drop=0.1, laplacian_lambda=0.0):
        super().__init__()
        self.gnn_type = gnn_type
        self.laplacian_lambda = laplacian_lambda

        self.temporal = DepthwiseTemporal(C, D=D, k=64, drop=drop)
        self.se = ChannelSE(C)

        self.proj = nn.Linear(D, gnn_hidden)

        self.layers = nn.ModuleList()
        for _ in range(gnn_layers):
            if gnn_type == "gat":
                self.layers.append(GATv2(gnn_hidden, gnn_hidden, dropout=drop))
            else:
                self.layers.append(GraphConv(gnn_hidden, gnn_hidden))

        self.norm = nn.LayerNorm(gnn_hidden)
        self.head = nn.Sequential(
            nn.Linear(gnn_hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, A_hat, A_bin=None):
        """
        x: [B, 21, T]
        A_hat: [21, 21]  对称规范化邻接 (GCN 用)
        A_bin: [21, 21]  0/1 邻接 (GAT 用，可为空)
        """
        H = self.temporal(x)             # [B, 21, D]
        H = self.se(H)                   # [B, 21, D]
        H = self.proj(H)                 # [B, 21, F]

        for layer in self.layers:
            if self.gnn_type == "gat":
                H = layer(H, A_bin)
            else:
                H = layer(H, A_hat)

        H = self.norm(H)                 # [B, 21, F]

        H_mean = H.mean(dim=1)           # [B, F]
        H_max = H.max(dim=1).values      # [B, F]
        Z = torch.cat([H_mean, H_max], dim=-1)  # [B, 2F]

        logits = self.head(Z)            # [B, n_classes]
        return logits, H, Z

    def laplacian_loss(self, H, A_hat):
        """
        空间平滑正则：鼓励相邻通道特征相似
        H: [B, 21, F]
        """
        I = torch.eye(A_hat.size(0), device=A_hat.device, dtype=A_hat.dtype)
        L = I - A_hat
        LH = torch.matmul(L, H)   # [B, 21, F]
        return (LH ** 2).mean()


# -------------------------
# 3. SupCon + Center 损失
# -------------------------
def supervised_contrastive_loss(z, y, temperature=0.1):
    """
    简化版 Supervised Contrastive Loss:
    z: [B, D] 已经经过投影/归一化的特征
    y: [B] 标签
    """
    device = z.device
    B = z.size(0)
    if B <= 1:
        return z.new_tensor(0.0)

    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.T) / temperature  # [B, B]

    y = y.view(-1, 1)
    mask = torch.eq(y, y.T).float().to(device)  # [B, B]

    logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask  # [B, B]
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

    mask_sum = mask.sum(dim=1)  # [B]
    loss = -(mask * log_prob).sum(dim=1) / (mask_sum + 1e-6)

    valid = mask_sum > 0
    if valid.sum() == 0:
        return z.new_tensor(0.0)
    return loss[valid].mean()


def batch_center_loss(z, y):
    """
    类中心损失（batch 版本）：
    对当前 batch，每一类求一个中心，让同类样本靠近中心。
    z: [B, D]
    y: [B]
    """
    if z.size(0) <= 1:
        return z.new_tensor(0.0)

    loss_terms = []
    for cls in torch.unique(y):
        idx = (y == cls)
        if idx.sum() <= 1:
            continue
        z_c = z[idx]                      # [n_c, D]
        center = z_c.mean(dim=0, keepdim=True)  # [1, D]
        loss_terms.append(((z_c - center) ** 2).mean())

    if not loss_terms:
        return z.new_tensor(0.0)
    return torch.stack(loss_terms).mean()


# -------------------------
# 4. 封装成 EEGGCN21：提供 fit / predict
# -------------------------
class EEGGCN21:
    """
    21 通道 GNN 分类器（增强版，GPU 友好）：
      - 时域 depthwise conv + 通道注意力 + GCN/GAT
      - 基于训练集的通道标准化（跨被试一致性）
      - 监督对比损失 + 类中心损失（提升类间可分性）
      - 可选 Laplacian 正则
    """
    def __init__(self,
                 n_time,
                 n_classes,
                 adj_type="hybrid",         # 'struct' / 'func' / 'hybrid'
                 k_struct=4, k_func=4, alpha=0.5,
                 gnn_type="gcn",           # 'gcn' 或 'gat'
                 gnn_hidden=64, gnn_layers=2,
                 temporal_D=32,
                 lr=1e-3, weight_decay=1e-4,
                 epochs=40, batch_size=64,
                 laplacian_lambda=0.0,
                 supcon_lambda=0.05,
                 center_lambda=0.01):
        self.C = 21
        self.T = n_time
        self.K = n_classes

        self.adj_cfg = dict(
            adj_type=adj_type,
            k_struct=k_struct,
            k_func=k_func,
            alpha=alpha,
        )
        self.lr = lr
        self.wd = weight_decay
        self.epochs = epochs
        self.bs = batch_size
        self.laplacian_lambda = laplacian_lambda
        self.supcon_lambda = supcon_lambda
        self.center_lambda = center_lambda

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EEGGraphBackbone21(
            C=self.C, T=self.T, n_classes=self.K,
            D=temporal_D, gnn_hidden=gnn_hidden,
            gnn_layers=gnn_layers, gnn_type=gnn_type,
            drop=0.1, laplacian_lambda=laplacian_lambda,
        ).to(self.device)

        self.A_hat = None    # [21, 21] float32
        self.A_bin = None    # [21, 21] float32 (GAT 用)

        # 训练集通道标准化参数
        self.mean_ = None    # [21]
        self.std_ = None     # [21]

    # ---------- 通道标准化 ----------
    def _compute_norm(self, X_cpu):
        """
        X_cpu: torch.Tensor [N, 21, T] 在 CPU 上
        """
        X_np = X_cpu.numpy()
        mean = X_np.mean(axis=(0, 2))        # [21]
        std = X_np.std(axis=(0, 2)) + 1e-6   # [21]
        self.mean_ = torch.from_numpy(mean).float().to(self.device)
        self.std_ = torch.from_numpy(std).float().to(self.device)

    def _apply_norm(self, X):
        """
        X: [N, 21, T] 任意 device
        """
        if self.mean_ is None or self.std_ is None:
            return X
        mean = self.mean_.view(1, self.C, 1).to(X.device)
        std = self.std_.view(1, self.C, 1).to(X.device)
        return (X - mean) / std

    # ---------- 构图 ----------
    def _build_adj(self, X_train_cpu, coords21):
        """
        X_train_cpu: torch.Tensor [N, 21, T] 在 CPU 上（已标准化）
        """
        X_np = X_train_cpu.numpy()
        A_hat, A_bin = build_adjacency_21(X_np, coords21, **self.adj_cfg)
        self.A_hat = torch.tensor(A_hat, device=self.device, dtype=torch.float32)
        self.A_bin = torch.tensor(A_bin, device=self.device, dtype=torch.float32)

    # ---------- batch 迭代 ----------
    @staticmethod
    def _iter_batches(X, y, bs):
        N = len(y)
        idx = torch.randperm(N)
        for i in range(0, N, bs):
            j = idx[i:i + bs]
            yield X[j], y[j]

    # ---------- 训练 ----------
    def fit(self, X, y, coords21, clabel=None):
        """
        X: [N, 21, T]
        y: [N]，值为 0..K-1
        coords21: [21, 2]
        clabel: 被试/域标签 [N]（当前未用）
        """
        # 1) 先在 CPU 上创建，再做标准化 & 构图，然后整体搬到 GPU
        X_cpu = torch.tensor(X, dtype=torch.float32)  # [N, 21, T] CPU
        y = torch.tensor(y, dtype=torch.long).to(self.device)

        # 训练集均值/方差（CPU → device 缓存）
        self._compute_norm(X_cpu)

        # 标准化后的 X（在 CPU）
        X_cpu = self._apply_norm(X_cpu)

        # 基于标准化后的训练数据构图（用 CPU 数据 -> numpy）
        self._build_adj(X_cpu, coords21)

        # 再把标准化后的 X 搬到 GPU
        X_gpu = X_cpu.to(self.device)

        # 简单划分内部 train/val（注意：这只是监控，不等价于 LOSO）
        N = len(y)
        perm = torch.randperm(N)
        n_val = max(1, int(0.1 * N))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        Xtr, ytr = X_gpu[tr_idx], y[tr_idx]
        Xva, yva = X_gpu[val_idx], y[val_idx]

        opt = torch.optim.AdamW(self.model.parameters(),
                                lr=self.lr, weight_decay=self.wd)

        best = {"acc": 0.0, "state": None}

        for ep in range(self.epochs):
            self.model.train()
            losses = []
            tr_correct, tr_total = 0, 0

            for xb, yb in self._iter_batches(Xtr, ytr, self.bs):
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                logits, H, Z = self.model(xb, self.A_hat, self.A_bin)

                ce_loss = F.cross_entropy(logits, yb)
                supcon_loss = supervised_contrastive_loss(Z, yb)
                center_loss = batch_center_loss(Z, yb)

                if self.laplacian_lambda > 0:
                    lap_loss = self.model.laplacian_loss(H, self.A_hat)
                else:
                    lap_loss = torch.tensor(0.0, device=self.device)

                loss = (
                    ce_loss
                    + self.supcon_lambda * supcon_loss
                    + self.center_lambda * center_loss
                    + self.laplacian_lambda * lap_loss
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(float(loss))
                with torch.no_grad():
                    pred = logits.argmax(1)
                    tr_correct += (pred == yb).sum().item()
                    tr_total += len(yb)

            # 验证（仍然在 GPU 上跑）
            self.model.eval()
            with torch.no_grad():
                logits_va, _, _ = self.model(Xva, self.A_hat, self.A_bin)
                pred_va = logits_va.argmax(1)
                acc_va = (pred_va == yva).float().mean().item()

            tr_acc = tr_correct / max(1, tr_total)
            print(
                f"[EEGGCN_21] Epoch {ep+1}/{self.epochs} "
                f"loss={np.mean(losses):.4f} "
                f"train_acc={tr_acc:.3f} val_acc={acc_va:.3f}"
            )

            if acc_va > best["acc"]:
                best["acc"] = acc_va
                best["state"] = {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                }

        if best["state"] is not None:
            self.model.load_state_dict(best["state"])

    # ---------- 预测 ----------
    def predict(self, X):
        """
        X: [N, 21, T]
        使用训练集归一化 + 固定图，直接预测
        """
        self.model.eval()
        X_cpu = torch.tensor(X, dtype=torch.float32)  # CPU
        X_cpu = self._apply_norm(X_cpu)               # 用训练集 mean/std 标准化
        X_gpu = X_cpu.to(self.device)

        with torch.no_grad():
            logits, _, _ = self.model(X_gpu, self.A_hat, self.A_bin)
            pred = logits.argmax(1).cpu().numpy()

        unique, counts = np.unique(pred, return_counts=True)
        print("[EEGGCN_21] 预测类别分布:", dict(zip(unique.tolist(), counts.tolist())))
        return pred
