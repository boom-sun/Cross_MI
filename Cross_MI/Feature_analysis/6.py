# -*- coding: utf-8 -*-
"""
Step6_stability_screen_hybrid_v2.py

浣滅敤锛�
1. 鐩存帴璇诲彇 Step1_compute_ERSP_save.py 鐢熸垚鐨� Hybrid ERSP .mat 鏂囦欢
2. 鐢熸垚璁烘枃3鏈€闇€瑕佺殑 4 寮犱富琛細
   - stable_feature_subject_scene_stim.csv
   - scene_stability_subject_stim.csv
   - paradigm_stability_subject_scene.csv
   - feature_ranking_stability_discriminability.csv

鏀瑰姩鐐癸細
- 涓嶅啀寮哄埗鍛戒护琛屽弬鏁�
- 鏀寔椤堕儴 CONFIG 鐩存帴鏀硅矾寰�
- 濡傛灉浣犱粛鎯冲懡浠よ浼犲弬锛屼篃鏀寔 --ersp_dir / --out_dir / --subjects
"""

from __future__ import annotations
import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import hdf5storage
    HAS_HDF5STORAGE = True
except Exception:
    HAS_HDF5STORAGE = False

from scipy.io import loadmat

# =========================
# CONFIG锛氱洿鎺ユ敼杩欓噷灏辫兘璺�
# =========================
CONFIG = {
    # 浣犵殑 Hybrid ERSP 杈撳嚭鐩綍锛圫tep1_compute_ERSP_save.py 鐨� save_root_ERSP锛�
    "ERSP_DIR": r'E:\\Datasets\\4_跨场景因素研究v2\\跨场景因素研究v2画图数据\\ERSP数据',
    # 杈撳嚭鐩綍
    "OUT_DIR": r'E:\\Datasets\\4_跨场景因素研究v2\\跨场景因素研究v2画图数据\\ERSP分析',
    # 琚瘯鑼冨洿锛屾敮鎸� "1-37" 鎴� "1,2,3"
    "SUBJECTS": "1-37",
    # 榛樿 4 涓� paradigm
    "STIMS": ["ssvideo", "video", "ssmvep", "cue"],
    # 閫氶亾鍚嶇О锛屽拰浣犱粨搴� Step1_TOPO_TF_save.py 淇濇寔涓€鑷�
    "CHANNELS": [
        'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
        'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
        'CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
        'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ',
        'PO4','PO6','PO8','O1','OZ','O2'
    ],
    # 棰戝甫锛圚z锛�
    "BANDS": {
        "mu": (8, 13),
        "beta": (13, 30),
        "low_beta": (13, 20),
        "high_beta": (20, 30),
    },
    # 鏃堕棿绐楋紙绉掞級
    "WINDOWS": {
        "full_task": (0.0, 4.0),
        "early": (0.5, 1.5),
        "mid": (1.5, 2.5),
        "late": (2.5, 4.0),
    },
}

# 宸﹀彸 ROI锛氬彲浠ユ寜浣犲悗闈㈤渶瑕佸啀寰皟
LEFT_ROI = ["FC3", "C5", "C3", "C1", "CP3", "CP1"]
RIGHT_ROI = ["FC4", "C6", "C4", "C2", "CP4", "CP2"]


def parse_subjects(spec: str) -> List[int]:
    spec = str(spec).strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid ERSP stability screening")
    parser.add_argument("--ersp_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--subjects", type=str, default=None)
    return parser.parse_args()


def get_cfg() -> Dict:
    args = parse_args()
    cfg = dict(CONFIG)
    if args.ersp_dir:
        cfg["ERSP_DIR"] = args.ersp_dir
    if args.out_dir:
        cfg["OUT_DIR"] = args.out_dir
    if args.subjects:
        cfg["SUBJECTS"] = args.subjects
    return cfg


def smart_loadmat(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if HAS_HDF5STORAGE:
        try:
            return hdf5storage.loadmat(str(path))
        except Exception:
            pass
    return loadmat(str(path))


def extract_array(mat_dict: Dict, preferred_key: str | None = None) -> np.ndarray:
    if preferred_key and preferred_key in mat_dict:
        arr = np.array(mat_dict[preferred_key])
    else:
        candidates = []
        for k, v in mat_dict.items():
            if k.startswith("__"):
                continue
            arr = np.array(v)
            if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                candidates.append((k, arr))
        if not candidates:
            raise ValueError("mat 鏂囦欢涓病鏈夋壘鍒版暟鍊兼暟缁�")
        candidates.sort(key=lambda kv: kv[1].size, reverse=True)
        arr = candidates[0][1]

    arr = np.array(arr)
    if arr.dtype == object and arr.size == 1:
        arr = np.array(arr.ravel()[0])
    if arr.ndim >= 1 and arr.shape[0] == 1 and arr.dtype != object:
        arr = np.squeeze(arr, axis=0)
    return np.array(arr)


def load_ersp(ersp_dir: Path, subject: int, scene: int, stim: str, class_id: int) -> np.ndarray:
    path = ersp_dir / f"sub{subject}_sence{scene}_{stim}_class{class_id}.mat"
    mat = smart_loadmat(path)
    key = "ERSP_1" if scene == 1 else "ERSP_2"
    arr = extract_array(mat, preferred_key=key)

    # 鐩爣褰㈢姸搴斾负 [freq, time, channel, trial]
    arr = np.array(arr)
    if arr.ndim != 4:
        raise ValueError(f"{path.name} 鐨� ERSP 缁村害涓嶆槸4锛岃€屾槸 {arr.shape}")
    return arr.astype(np.float32)


def load_times_freqs(ersp_dir: Path, stim: str, class_id: int) -> Tuple[np.ndarray, np.ndarray]:
    path = ersp_dir / f"times+freqs_{stim}_class{class_id}.mat"
    mat = smart_loadmat(path)
    times = np.squeeze(np.array(mat["times"])).astype(float)
    freqs = np.squeeze(np.array(mat["freqs"])).astype(float)
    # 浠撳簱閲� times 鏄� ms
    if np.nanmax(np.abs(times)) > 100:
        times = times / 1000.0
    return times, freqs


def channel_index_map(channels: List[str]) -> Dict[str, int]:
    return {ch.upper(): i for i, ch in enumerate(channels)}


def roi_mean(vec: np.ndarray, idx_map: Dict[str, int], names: List[str]) -> float:
    idx = [idx_map[n.upper()] for n in names if n.upper() in idx_map]
    if not idx:
        return np.nan
    return float(np.nanmean(vec[idx]))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if np.allclose(np.std(a), 0) or np.allclose(np.std(b), 0):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def normalized_similarity(x: float, y: float, eps: float = 1e-8) -> float:
    if np.isnan(x) or np.isnan(y):
        return np.nan
    return float(1.0 - abs(x - y) / (abs(x) + abs(y) + eps))


def corr_to_unit(x: float) -> float:
    if np.isnan(x):
        return np.nan
    return float((x + 1.0) / 2.0)


def band_window_average(
    ersp: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    band: Tuple[float, float],
    window: Tuple[float, float],
) -> np.ndarray:
    # 杈撳叆 ersp: [freq, time, channel, trial]
    fmask = (freqs >= band[0]) & (freqs <= band[1])
    tmask = (times >= window[0]) & (times <= window[1])
    if not np.any(fmask):
        raise ValueError(f"band {band} 鍦� freqs 涓病鏈夐噰鏍风偣")
    if not np.any(tmask):
        raise ValueError(f"window {window} 鍦� times 涓病鏈夐噰鏍风偣")
    # 骞冲潎鍒� [channel, trial]
    x = ersp[np.ix_(fmask, tmask, np.arange(ersp.shape[2]), np.arange(ersp.shape[3]))]
    x = np.nanmean(x, axis=(0, 1))
    return x  # [channel, trial]


def build_feature_rows(
    ersp_dir: Path,
    subjects: List[int],
    stims: List[str],
    channels: List[str],
    bands: Dict[str, Tuple[float, float]],
    windows: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    idx_map = channel_index_map(channels)
    rows = []

    c3_idx = idx_map["C3"]
    c4_idx = idx_map["C4"]

    for stim in stims:
        times, freqs = load_times_freqs(ersp_dir, stim, class_id=1)

        for s in subjects:
            for scene in [1, 2]:
                left = load_ersp(ersp_dir, s, scene, stim, 1)
                right = load_ersp(ersp_dir, s, scene, stim, 2)

                for band_name, band in bands.items():
                    for win_name, win in windows.items():
                        left_ct = band_window_average(left, freqs, times, band, win)   # [ch, trial]
                        right_ct = band_window_average(right, freqs, times, band, win) # [ch, trial]

                        left_map = np.nanmean(left_ct, axis=1)
                        right_map = np.nanmean(right_ct, axis=1)
                        contrast_map = left_map - right_map

                        left_left_roi = roi_mean(left_map, idx_map, LEFT_ROI)
                        left_right_roi = roi_mean(left_map, idx_map, RIGHT_ROI)
                        right_left_roi = roi_mean(right_map, idx_map, LEFT_ROI)
                        right_right_roi = roi_mean(right_map, idx_map, RIGHT_ROI)

                        # 宸﹀彸浠诲姟鍦ㄥ弻渚ц繍鍔ㄥ尯鐨勪晶鍖栧樊寮�
                        li_left = left_left_roi - left_right_roi
                        li_right = right_left_roi - right_right_roi
                        disc_li = abs(li_left - li_right)

                        c3_left = float(left_map[c3_idx])
                        c4_left = float(left_map[c4_idx])
                        c3_right = float(right_map[c3_idx])
                        c4_right = float(right_map[c4_idx])

                        disc_c3c4 = abs((c3_left - c4_left) - (c3_right - c4_right))
                        motor_indices = [idx_map[x.upper()] for x in LEFT_ROI + RIGHT_ROI if x.upper() in idx_map]
                        motor_contrast_strength = float(np.nanmean(np.abs(contrast_map[motor_indices])))
                        whole_contrast_strength = float(np.nanmean(np.abs(contrast_map)))

                        rows.append({
                            "subject": s,
                            "scene": f"S{scene}",
                            "stim": stim,
                            "band": band_name,
                            "window": win_name,
                            "li_left": li_left,
                            "li_right": li_right,
                            "disc_li": disc_li,
                            "c3_left": c3_left,
                            "c4_left": c4_left,
                            "c3_right": c3_right,
                            "c4_right": c4_right,
                            "disc_c3c4": disc_c3c4,
                            "motor_contrast_strength": motor_contrast_strength,
                            "whole_contrast_strength": whole_contrast_strength,
                            "contrast_map": json_dumps_array(contrast_map),
                            "left_map": json_dumps_array(left_map),
                            "right_map": json_dumps_array(right_map),
                        })
    return pd.DataFrame(rows)


def json_dumps_array(arr: np.ndarray) -> str:
    # 淇濆瓨鎴愬瓧绗︿覆锛宑sv 鍙洖璇�
    return "[" + ",".join(f"{float(x):.8f}" for x in np.asarray(arr).ravel()) + "]"


def json_loads_array(s: str) -> np.ndarray:
    s = s.strip()
    if not s.startswith("["):
        raise ValueError("涓嶆槸鏁扮粍瀛楃涓�")
    body = s[1:-1].strip()
    if not body:
        return np.array([], dtype=float)
    return np.array([float(x) for x in body.split(",")], dtype=float)


def compute_scene_stability(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["subject", "stim", "band", "window"]

    for key, g in feature_df.groupby(group_cols):
        if set(g["scene"]) != {"S1", "S2"}:
            continue
        g1 = g[g["scene"] == "S1"].iloc[0]
        g2 = g[g["scene"] == "S2"].iloc[0]

        m1 = json_loads_array(g1["contrast_map"])
        m2 = json_loads_array(g2["contrast_map"])

        pattern_corr = safe_corr(m1, m2)
        pattern_sim = corr_to_unit(pattern_corr)

        disc_li_stab = normalized_similarity(float(g1["disc_li"]), float(g2["disc_li"]))
        disc_c3c4_stab = normalized_similarity(float(g1["disc_c3c4"]), float(g2["disc_c3c4"]))
        strength_stab = normalized_similarity(
            float(g1["motor_contrast_strength"]),
            float(g2["motor_contrast_strength"])
        )

        values = [x for x in [pattern_sim, disc_li_stab, disc_c3c4_stab, strength_stab] if not np.isnan(x)]
        si = float(np.mean(values)) if values else np.nan

        rows.append({
            "subject": key[0],
            "stim": key[1],
            "band": key[2],
            "window": key[3],
            "scene_pattern_corr": pattern_corr,
            "scene_pattern_similarity": pattern_sim,
            "scene_disc_li_stability": disc_li_stab,
            "scene_disc_c3c4_stability": disc_c3c4_stab,
            "scene_strength_stability": strength_stab,
            "scene_stability_index": si,
            "mean_disc_li": float(np.mean([g1["disc_li"], g2["disc_li"]])),
            "mean_disc_c3c4": float(np.mean([g1["disc_c3c4"], g2["disc_c3c4"]])),
            "mean_motor_contrast_strength": float(np.mean([g1["motor_contrast_strength"], g2["motor_contrast_strength"]])),
        })

    return pd.DataFrame(rows)


def compute_paradigm_stability(feature_df: pd.DataFrame, stims: List[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["subject", "scene", "band", "window"]

    for key, g in feature_df.groupby(group_cols):
        present = sorted(g["stim"].unique().tolist())
        if len(present) < 2:
            continue

        pair_pattern = []
        pair_disc = []
        pair_strength = []

        stim_to_row = {row["stim"]: row for _, row in g.iterrows()}

        for a, b in itertools.combinations(present, 2):
            ga = stim_to_row[a]
            gb = stim_to_row[b]

            ma = json_loads_array(ga["contrast_map"])
            mb = json_loads_array(gb["contrast_map"])

            pcorr = safe_corr(ma, mb)
            pair_pattern.append(corr_to_unit(pcorr))
            pair_disc.append(normalized_similarity(float(ga["disc_li"]), float(gb["disc_li"])))
            pair_strength.append(
                normalized_similarity(float(ga["motor_contrast_strength"]), float(gb["motor_contrast_strength"]))
            )

        vals = [np.nanmean(pair_pattern), np.nanmean(pair_disc), np.nanmean(pair_strength)]
        vals = [x for x in vals if not np.isnan(x)]
        pi = float(np.mean(vals)) if vals else np.nan

        rows.append({
            "subject": key[0],
            "scene": key[1],
            "band": key[2],
            "window": key[3],
            "paradigm_pattern_similarity": float(np.nanmean(pair_pattern)),
            "paradigm_disc_stability": float(np.nanmean(pair_disc)),
            "paradigm_strength_stability": float(np.nanmean(pair_strength)),
            "paradigm_stability_index": pi,
        })

    return pd.DataFrame(rows)


def build_ranking(scene_stab_df: pd.DataFrame, paradigm_stab_df: pd.DataFrame) -> pd.DataFrame:
    # 浠� band 脳 window 涓哄崟浣嶏紝寰楀埌璁烘枃3鏈€鍏抽敭鐨� ranking
    s = scene_stab_df.groupby(["band", "window"]).agg(
        scene_stability_index=("scene_stability_index", "mean"),
        scene_pattern_similarity=("scene_pattern_similarity", "mean"),
        mean_disc_li=("mean_disc_li", "mean"),
        mean_disc_c3c4=("mean_disc_c3c4", "mean"),
        mean_motor_contrast_strength=("mean_motor_contrast_strength", "mean"),
    ).reset_index()

    p = paradigm_stab_df.groupby(["band", "window"]).agg(
        paradigm_stability_index=("paradigm_stability_index", "mean"),
    ).reset_index()

    out = s.merge(p, on=["band", "window"], how="left")
    out["stability_combined"] = out[["scene_stability_index", "paradigm_stability_index"]].mean(axis=1)

    # 鍒ゅ埆鎬э細杩欓噷鍏堢敤 disc_li 鍜� disc_c3c4 鐨勫潎鍊�
    out["discriminability_score"] = out[["mean_disc_li", "mean_disc_c3c4"]].mean(axis=1)

    # 鍥涜薄闄愭爣璁帮細鍚庨潰鐢荤ǔ瀹氭€р€斿垽鍒€т簩缁村浘鏃剁洿鎺ョ敤
    stab_med = out["stability_combined"].median()
    disc_med = out["discriminability_score"].median()

    def quad(r):
        hs = r["stability_combined"] >= stab_med
        hd = r["discriminability_score"] >= disc_med
        if hs and hd:
            return "high_stability_high_disc"
        if hs and not hd:
            return "high_stability_low_disc"
        if (not hs) and hd:
            return "low_stability_high_disc"
        return "low_stability_low_disc"

    out["quadrant"] = out.apply(quad, axis=1)
    out = out.sort_values(["stability_combined", "discriminability_score"], ascending=False).reset_index(drop=True)
    return out


def build_transfer_bridge(scene_stab_df: pd.DataFrame) -> pd.DataFrame:
    # 杩欐槸缁欏悗闈㈡帴 cross-scene decoding 鐢ㄧ殑妗ヨ〃
    out = scene_stab_df.groupby(["stim", "band", "window"]).agg(
        scene_stability_index=("scene_stability_index", "mean"),
        scene_pattern_similarity=("scene_pattern_similarity", "mean"),
        mean_disc_li=("mean_disc_li", "mean"),
        mean_disc_c3c4=("mean_disc_c3c4", "mean"),
        mean_motor_contrast_strength=("mean_motor_contrast_strength", "mean"),
    ).reset_index()
    out = out.sort_values("scene_stability_index", ascending=False).reset_index(drop=True)
    return out


def main():
    cfg = get_cfg()
    ersp_dir = Path(cfg["ERSP_DIR"])
    out_dir = Path(cfg["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = parse_subjects(cfg["SUBJECTS"])
    stims = list(cfg["STIMS"])
    channels = list(cfg["CHANNELS"])
    bands = dict(cfg["BANDS"])
    windows = dict(cfg["WINDOWS"])

    print(f"[INFO] ERSP_DIR = {ersp_dir}")
    print(f"[INFO] OUT_DIR  = {out_dir}")
    print(f"[INFO] SUBJECTS = {subjects[:3]} ... {subjects[-3:] if len(subjects) > 3 else subjects}")

    feature_df = build_feature_rows(
        ersp_dir=ersp_dir,
        subjects=subjects,
        stims=stims,
        channels=channels,
        bands=bands,
        windows=windows,
    )
    feature_csv = out_dir / "stable_feature_subject_scene_stim.csv"
    feature_df.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 鍐欏嚭 {feature_csv}")

    scene_stab_df = compute_scene_stability(feature_df)
    scene_csv = out_dir / "scene_stability_subject_stim.csv"
    scene_stab_df.to_csv(scene_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 鍐欏嚭 {scene_csv}")

    paradigm_stab_df = compute_paradigm_stability(feature_df, stims=stims)
    paradigm_csv = out_dir / "paradigm_stability_subject_scene.csv"
    paradigm_stab_df.to_csv(paradigm_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 鍐欏嚭 {paradigm_csv}")

    ranking_df = build_ranking(scene_stab_df, paradigm_stab_df)
    ranking_csv = out_dir / "feature_ranking_stability_discriminability.csv"
    ranking_df.to_csv(ranking_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 鍐欏嚭 {ranking_csv}")

    bridge_df = build_transfer_bridge(scene_stab_df)
    bridge_csv = out_dir / "transfer_prediction_bridge_table.csv"
    bridge_df.to_csv(bridge_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 鍐欏嚭 {bridge_csv}")

    print("\n[DONE] 绋冲畾鎬т富琛ㄥ凡鐢熸垚銆�")
    print("涓嬩竴姝ュ缓璁細")
    print("1) 鍏堢湅 feature_ranking_stability_discriminability.csv")
    print("2) 閫夊嚭 high_stability_high_disc 鐨� band/window")
    print("3) 鍐嶆妸杩欎簺绐楀彛閫佸叆 cross-scene 瑙ｇ爜楠岃瘉")


if __name__ == "__main__":
    main()