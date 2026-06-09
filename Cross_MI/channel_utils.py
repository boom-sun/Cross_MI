# Cross_MI/channel_utils.py
import numpy as np

# 你想用的 21 个导联（MI 区域）
MI_21_CHANNELS = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5",  "C3",  "C1",  "CZ",  "C2",  "C4",  "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
]

# 注意：有些 .locs 文件里是小写 / 无 Z，大写统一处理
def _norm_name(name: str) -> str:
    return name.strip().upper()

def load_channel_locs(loc_path: str):
    """
    从 Neuroscan .locs 文件读取所有通道坐标和名字
    返回:
        labels: list[str] 长度 C
        coords: np.ndarray, shape [C, 2] (X, Y)
    """
    labels = []
    coords = []
    with open(loc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            # 一般格式: idx  X  Y  Name
            try:
                x = float(parts[1])
                y = float(parts[2])
                name = parts[3]
            except ValueError:
                continue
            labels.append(_norm_name(name))
            coords.append([x, y])
    return labels, np.asarray(coords, dtype=np.float32)

def pick_mi_21(loc_path: str):
    """
    返回:
        idx21 : list[int]  21 个通道在原 60 导联中的索引
        coords21 : np.ndarray, [21, 2]
        labels21 : list[str]
    """
    labels, coords = load_channel_locs(loc_path)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    idx21 = []
    coords21 = []
    labels21 = []
    for name in MI_21_CHANNELS:
        key = _norm_name(name)
        if key not in label_to_idx:
            raise ValueError(f"通道 {key} 在 {loc_path} 中找不到，请检查命名是否一致")
        i = label_to_idx[key]
        idx21.append(i)
        coords21.append(coords[i])
        labels21.append(key)
    coords21 = np.asarray(coords21, dtype=np.float32)
    return idx21, coords21, labels21
