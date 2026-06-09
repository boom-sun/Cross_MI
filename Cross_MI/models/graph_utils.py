# Cross_MI/models/graph_utils.py
import numpy as np

def _knn_from_coords(coords: np.ndarray, k: int = 4) -> np.ndarray:
    """
    coords: [C, d] (d=2/3). 返回 0/1 邻接（无向、自环后续加）
    k: 每个点连接的最近邻个数（21 通道用 3~5 比较合适，这里默认 4）
    """
    C = coords.shape[0]
    A = np.zeros((C, C), dtype=np.float32)
    for i in range(C):
        d = np.linalg.norm(coords - coords[i:i + 1, :], axis=1)
        idx = np.argsort(d)[1 : k + 1]  # 排除自身
        A[i, idx] = 1.0
        A[idx, i] = 1.0
    return A

def _functional_from_data(X: np.ndarray, topk: int = 4) -> np.ndarray:
    """
    X: [N, C, T]  所有训练 trial（已经只保留 21 通道）
    计算通道间皮尔逊相关，取每个通道绝对相关 topk 作为边。
    """
    N, C, T = X.shape
    x = X.transpose(1, 0, 2).reshape(C, N * T)  # [C, N*T]
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
    corr = np.dot(x, x.T) / x.shape[1]
    np.fill_diagonal(corr, 0.0)
    A = np.zeros_like(corr, dtype=np.float32)
    for i in range(C):
        idx = np.argsort(-np.abs(corr[i]))[:topk]
        A[i, idx] = 1.0
        A[idx, i] = 1.0
    return A

def normalize_adj(A: np.ndarray, self_loop: bool = True) -> np.ndarray:
    """
    对称规范化 A_hat = D^{-1/2} (A + I) D^{-1/2}
    """
    if self_loop:
        A = A.copy()
        np.fill_diagonal(A, 1.0)
    d = A.sum(axis=1) + 1e-6
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat.astype(np.float32)

def build_adjacency_21(X_train: np.ndarray,
                       coords21: np.ndarray,
                       adj_type: str = "hybrid",
                       k_struct: int = 4,
                       k_func: int = 4,
                       alpha: float = 0.5):
    """
    专为 21 通道构图。
    X_train: [N, 21, T]
    coords21: [21, 2]
    adj_type: 'struct' | 'func' | 'hybrid'
    返回:
        A_hat: [21, 21]  对称规范化邻接
        A_bin: [21, 21]  0/1 邻接（二值，GAT 掩码）
    """
    if adj_type == "struct":
        A_s = _knn_from_coords(coords21, k=k_struct)
        A = A_s
    elif adj_type == "func":
        A_f = _functional_from_data(X_train, topk=k_func)
        A = A_f
    else:
        A_s = _knn_from_coords(coords21, k=k_struct)
        A_f = _functional_from_data(X_train, topk=k_func)
        As = A_s / (A_s.sum() + 1e-6)
        Af = A_f / (A_f.sum() + 1e-6)
        A = alpha * As + (1 - alpha) * Af
        # 简单二值化：保留较大的边
        if np.any(A > 0):
            thr = np.percentile(A[A > 0], 50)
            A = (A >= thr).astype(np.float32)
        else:
            A = np.zeros_like(A, dtype=np.float32)
    A_hat = normalize_adj(A, self_loop=True)
    A_bin = (A_hat > 0).astype(np.float32)
    return A_hat, A_bin
