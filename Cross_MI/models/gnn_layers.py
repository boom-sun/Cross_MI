# Cross_MI/models/gnn_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    """ 简单的 GCN 层 """
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, H, A_hat):
        HW = self.lin(H)              # [B, C, F]
        H2 = torch.matmul(A_hat, HW)  # 邻居聚合
        B, C, Fout = H2.shape
        out = self.bn(H2.view(B * C, Fout)).view(B, C, Fout)
        return F.gelu(out)

class GATv2(nn.Module):
    """ 图注意力层（GATv2） """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(out_dim * 2, 1))
        nn.init.xavier_uniform_(self.a)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, H, A_bin):
        Q = self.lin(H)  # [B, C, F]
        B, C, Fd = Q.shape
        Q_i = Q.unsqueeze(2).expand(B, C, C, Fd)
        Q_j = Q.unsqueeze(1).expand(B, C, C, Fd)
        e_ij = torch.matmul(torch.cat([Q_i, Q_j], dim=-1), self.a).squeeze(-1)  # [B, C, C]
        mask = (A_bin > 0).unsqueeze(0).expand(B, -1, -1)
        e_ij = e_ij.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e_ij, dim=-1)
        alpha = self.dropout(alpha)
        H2 = torch.matmul(alpha, Q)  # [B, C, F]
        out = self.bn(H2.reshape(B * C, Fd)).reshape(B, C, Fd)
        return F.gelu(out)
