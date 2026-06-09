import os
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Literal

# ========== 配置区 ==========
@dataclass
class Band:
    name: str
    fmin: float
    fmax: float

DEFAULT_BANDS = [
    Band("mu", 8, 13),
    Band("beta", 13, 30),
]

# ========== 工具函数：信号预处理（带通 + Hilbert） ==========
def bandpass(data: np.ndarray, sfreq: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    """
    二阶节段IIR带通滤波（filtfilt零相位），data: (..., n_times)
    """
    nyq = 0.5 * sfreq
    low = fmin / nyq
    high = fmax / nyq
    b, a = sps.butter(order, [low, high], btype='band')
    return sps.filtfilt(b, a, data, axis=-1)

def analytic_signal(data: np.ndarray) -> np.ndarray:
    """
    Hilbert 变换得到解析信号：返回 complex array，便于提取相位和瞬时振幅
    data: (..., n_times)
    """
    return sps.hilbert(data, axis=-1)

# ========== 功能连接：PLV / PLI / wPLI ==========
def compute_plv(phases: np.ndarray) -> np.ndarray:
    """
    PLV: |E[exp(iΔϕ)]|，在时间或试次维度上平均。
    phases: (n_epochs, n_channels, n_times) 的相位（弧度）
    返回: (n_channels, n_channels) 矩阵
    """
    n_epochs, n_ch, n_times = phases.shape
    plv = np.zeros((n_ch, n_ch))
    # 合并时间和试次统计，提高稳定性
    # 也可改为先每 trial 求，再平均
    ph = phases.reshape(n_epochs, n_ch, n_times)
    for i in range(n_ch):
        for j in range(i, n_ch):
            dphi = np.exp(1j * (ph[:, i, :] - ph[:, j, :]))  # (n_epochs, n_times)
            val = np.abs(dphi.mean(axis=(0, 1)))
            plv[i, j] = plv[j, i] = val
    return plv

def compute_wpli(analytic: np.ndarray) -> np.ndarray:
    """
    wPLI: |E[Im(x1 * conj(x2))]| / E[|Im(x1 * conj(x2))|]
    analytic: (n_epochs, n_channels, n_times) 解析信号
    返回: (n_channels, n_channels)
    """
    n_epochs, n_ch, n_times = analytic.shape
    wpli = np.zeros((n_ch, n_ch))
    # 将 epochs 和 time 作为样本汇总
    Z = analytic.reshape(n_epochs, n_ch, n_times)
    for i in range(n_ch):
        xi = Z[:, i, :].reshape(-1)  # (n_epochs*n_times,)
        for j in range(i+1, n_ch):
            xj = Z[:, j, :].reshape(-1)
            im_cross = np.imag(xi * np.conj(xj))  # 各样本的虚部
            num = np.abs(np.mean(im_cross))
            den = np.mean(np.abs(im_cross)) + 1e-12
            val = num / den
            wpli[i, j] = wpli[j, i] = val
    np.fill_diagonal(wpli, 0.0)
    return wpli

def compute_pli(phases: np.ndarray) -> np.ndarray:
    """
    PLI: |E[sign(sin(Δϕ))]|，对零相位差不敏感，降低体积传导影响
    phases: (n_epochs, n_channels, n_times)
    返回: (n_channels, n_channels)
    """
    n_epochs, n_ch, n_times = phases.shape
    pli = np.zeros((n_ch, n_ch))
    ph = phases.reshape(n_epochs, n_ch, n_times)
    for i in range(n_ch):
        for j in range(i+1, n_ch):
            dphi = ph[:, i, :] - ph[:, j, :]
            s = np.sign(np.sin(dphi))
            val = np.abs(np.mean(s))
            pli[i, j] = pli[j, i] = val
    np.fill_diagonal(pli, 0.0)
    return pli

# ========== 有效连接：格兰杰因果（pairwise） ==========
from statsmodels.tsa.stattools import grangercausalitytests

def _gc_pair(x: np.ndarray, y: np.ndarray, maxlag: int = 10) -> float:
    """
    计算 y→x 的格兰杰因果F统计的最大显著性（或取1-p）作为强度指标。
    x, y: 一维时间序列（已拼接epochs）长度 T
    返回：强度分数（0~1，越大表示因果越强；这里用 1 - p_min）
    """
    # 构造二维矩阵：列顺序为 [x, y]，测试 y causes x
    data = np.vstack([x, y]).T
    # statsmodels 的 grangercausalitytests 对每个lag给出检验与p值
    res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag+1)]
    p_min = np.min(pvals)
    score = 1.0 - float(p_min)
    return score

def compute_gc_matrix(X: np.ndarray, maxlag: int = 10) -> np.ndarray:
    """
    对多通道信号计算 pairwise Granger 因果矩阵（y→x），
    X: (n_channels, n_times_total)
    返回: (n_channels, n_channels) 矩阵，元素[i,j]表示 j→i 的强度
    """
    n_ch, _ = X.shape
    G = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            G[i, j] = _gc_pair(X[i], X[j], maxlag=maxlag)  # j causes i
    return G

# ========== 可视化 ==========
def plot_connectivity_matrix(mat: np.ndarray, ch_names: List[str], title: str, cmap: str = "viridis", vmin: float = 0.0, vmax: float = 1.0, save_path: Optional[str] = None):
    plt.figure(figsize=(6,5))
    im = plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(ch_names)), ch_names, rotation=90)
    plt.yticks(range(len(ch_names)), ch_names)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_graph(mat: np.ndarray, ch_names: List[str], title: str, threshold: float = 0.3, directed: bool = False, save_path: Optional[str] = None):
    """
    将连接矩阵绘制为网络图；threshold为边保留阈值。
    无向：功能连接；有向：格兰杰因果。
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for i, ni in enumerate(ch_names):
        G.add_node(ni)

    n_ch = len(ch_names)
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            w = mat[i, j]
            if w >= threshold:
                if directed:
                    G.add_edge(ch_names[j], ch_names[i], weight=w)  # j -> i
                else:
                    if j > i:
                        G.add_edge(ch_names[i], ch_names[j], weight=w)

    pos = nx.spring_layout(G, seed=42)  # 没有电极位置信息时使用力导向布局
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    plt.figure(figsize=(6,5))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightgrey", edgecolors="k")
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, width=[2*w for w in weights], arrows=directed, arrowstyle='-|>' if directed else None)
    plt.title(f"{title}  (thr={threshold})")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ========== 主流程：按频段计算功能连接；按条件/范式聚合 ==========
def connectivity_pipeline(
    X: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    y: Optional[np.ndarray] = None,
    bands: List[Band] = DEFAULT_BANDS,
    conn_metrics: List[Literal["wPLI","PLI","PLV"]] = ["wPLI","PLI","PLV"],
    out_dir: str = "./conn_outputs",
    title_prefix: str = "SubjectA_Scenario1_Paradigm1",
    gc_maxlag: int = 10,
    do_gc: bool = True
):
    """
    X: (n_trials, n_channels, n_times)  已滤波(1-40Hz)
    y: (n_trials,) 左右手标签，可为 {0,1} 或 {'L','R'}；若为空则整体统计
    bands: 频段列表
    conn_metrics: 选择计算的功能连接指标
    do_gc: 是否计算格兰杰因果（有效连接）
    """
    os.makedirs(out_dir, exist_ok=True)
    n_trials, n_ch, n_times = X.shape

    # 统一标签到 {0,1}，0=左，1=右（如果有的话）
    if y is not None:
        if y.dtype.kind in {'U','S','O'}:
            y_bin = np.array([1 if str(v).lower().startswith(('r','right')) else 0 for v in y])
        else:
            # 假设已有0/1
            y_bin = y.astype(int)
        conds = {"Left": (y_bin==0), "Right": (y_bin==1)}
    else:
        conds = {"All": np.ones(n_trials, dtype=bool)}

    # 遍历条件与频段
    for cond_name, mask in conds.items():
        Xc = X[mask]  # (n_sel_trials, n_ch, n_times)
        if Xc.size == 0:
            continue

        for band in bands:
            # 1) 频段滤波 + Hilbert 相位
            Xf = bandpass(Xc, sfreq, band.fmin, band.fmax, order=4)         # (tr, ch, t)
            Z  = analytic_signal(Xf)                                        # (tr, ch, t) complex
            phases = np.angle(Z)                                            # (tr, ch, t)

            # 2) 功能连接
            if "PLV" in conn_metrics:
                plv = compute_plv(phases)
                plot_connectivity_matrix(plv, ch_names, f"{title_prefix} | {cond_name} | {band.name} | PLV",
                                         save_path=os.path.join(out_dir, f"{title_prefix}_{cond_name}_{band.name}_PLV_mat.png"))
                plot_graph(plv, ch_names, f"{title_prefix} | {cond_name} | {band.name} | PLV",
                           threshold=np.quantile(plv, 0.85), directed=False,
                           save_path=os.path.join(out_dir, f"{title_prefix}_{cond_name}_{band.name}_PLV_net.png"))

            if "PLI" in conn_metrics:
                pli = compute_pli(phases)
                plot_connectivity_matrix(pli, ch_names, f"{title_prefix} | {cond_name} | {band.name} | PLI",
                                         save_path=os.path.join(out_dir, f"{title_prefix}_{cond_name}_{band.name}_PLI_mat.png"))
                plot_graph(pli, ch_names, f"{title_prefix} | {cond_name} | {band.name} | PLI",
                           threshold=np.quantile(pli, 0.85), directed=False,
                           save_path=os.path.join(out_dir, f"{title_prefix}_{cond_name}_{band.name}_PLI_net.png"))

            if "wPLI" in conn_metrics:
                wpli = compute_wpli(Z)
                plot_connectivity_matrix(wpli, ch_names, f"{title_prefix} | {cond_name} | {band.name} | wPLI",
                                         save_path=os.path.join(out_dir, f"{title_prefix}_{cond_name}_{band.name}_wPLI_mat.png"))
                plot_graph(wpli, ch_names, f"{title_prefix} | {cond_name} | {band.name} | wPLI",
                           threshold=np.quantile(wpli, 0.85), directed=False,
                           save_path=os.path.join(out_dir, f"{title_prefix}_{cond_name}_{band.name}_wPLI_net.png"))

        # 3) 有效连接（格兰杰）：默认对全带（已1–40Hz）信号做，必要时可切换到特定频段
        if do_gc:
            # 将 trial 级时间连接起来，形成一个长序列以满足GC需求
            # 也可以对每个trial单独估计再平均（样本不足时建议合并）
            Xc_long = Xc.transpose(1,0,2).reshape(n_ch, -1)  # (ch, trials*times)
            # 预处理：去均值 & 轻微去趋势
            Xc_long = sps.detrend(Xc_long - Xc_long.mean(axis=1, keepdims=True), axis=1, type='linear')
            G = compute_gc_matrix(Xc_long, maxlag=gc_maxlag)
            # 可视化（有向）
            plot_connectivity_matrix(G, ch_names, f"{title_prefix} | {cond_name} | Granger (1-p) strength", cmap="magma", vmin=0, vmax=1,
                                     save_path=os.path.join(out_dir, f"{title_prefix}_{cond_name}_GC_mat.png"))
            thr = np.quantile(G[G>0], 0.90) if np.any(G>0) else 0.0
            plot_graph(G, ch_names, f"{title_prefix} | {cond_name} | Granger (1-p) strength", threshold=thr, directed=True,
                       save_path=os.path.join(out_dir, f"{title_prefix}_{cond_name}_GC_net.png"))

    # 4) 如果有左右两类，提供“差异网络（右-左）”示例（以wPLI为例，β频段）
    if y is not None:
        band = [b for b in DEFAULT_BANDS if b.name == "beta"][0]
        mats = {}
        for cond_name, mask in conds.items():
            Xc = X[mask]
            if Xc.size == 0:
                continue
            Xf = bandpass(Xc, sfreq, band.fmin, band.fmax, order=4)
            Z  = analytic_signal(Xf)
            mats[cond_name] = compute_wpli(Z)
        if "Left" in mats and "Right" in mats:
            diff = mats["Right"] - mats["Left"]
            # 归一以便可视化（可选）
            dmax = np.max(np.abs(diff)) + 1e-12
            diff_norm = diff / dmax
            plot_connectivity_matrix(diff_norm, ch_names, f"{title_prefix} | beta | wPLI (Right - Left) normalized", cmap="bwr", vmin=-1, vmax=1,
                                     save_path=os.path.join(out_dir, f"{title_prefix}_beta_wPLI_RightMinusLeft.png"))
            # 网络图：仅展示正差异（右>左）
            pos_diff = np.clip(diff, 0, None)
            thr = np.quantile(pos_diff[pos_diff>0], 0.9) if np.any(pos_diff>0) else 0.0
            plot_graph(pos_diff, ch_names, f"{title_prefix} | beta | wPLI edges stronger in Right", threshold=thr, directed=False,
                       save_path=os.path.join(out_dir, f"{title_prefix}_beta_wPLI_RightStronger_net.png"))

# ========== 示例入口 ==========
if __name__ == "__main__":
    # ======== 示例数据（请替换为你的真实数据） ========
    np.random.seed(0)
    n_trials, n_channels, n_times = 60, 16, 750   # 60个trial，16导，3秒@250Hz
    sfreq = 250.0
    X = np.random.randn(n_trials, n_channels, n_times) * 1e-6  # 微伏级
    # 构造左右手标签（各30个）
    y = np.array([0]*30 + [1]*30)
    ch_names = [f"Ch{c+1}" for c in range(n_channels)]

    # 跑一个范式的例子（真实使用时：被试/场景/范式循环中调用）
    connectivity_pipeline(
        X=X,
        sfreq=sfreq,
        ch_names=ch_names,
        y=y,
        bands=DEFAULT_BANDS,
        conn_metrics=["wPLI", "PLI", "PLV"],
        out_dir="./conn_outputs_demo",
        title_prefix="Sub01_SceneA_StableFreqVideo",  # 你可以改为对应的“被试_场景_范式”
        gc_maxlag=8,
        do_gc=True
    )
