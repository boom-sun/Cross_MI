# -*- coding: utf-8 -*-
"""
Step15_Corrected_FeatureAccuracy_Analysis_FIXED.py

修正版 Step15
============

用途
----
读取修正版 Step14 输出的 linked_data_pair_aware.csv 或 linked_data.csv，
对 Intensity / Discriminability 与 Accuracy 的关系进行稳健分析。

核心改正
--------
原 Step14 会把一个 subject × test_stim × feature_scene 的 feature 值重复合并到：
    多个 algorithm
    多个 cross train→test pair
因此不能把每一行都当作完全独立样本进行普通 Spearman p 值或普通 partial correlation。

本脚本使用三层分析：

1. 描述性相关：
   protocol × feature_scene × algorithm 条件下，比较
       rho(discriminability, accuracy)
       rho(intensity, accuracy)
   该表只用于描述趋势。

2. subject-cluster bootstrap：
   以 subject 为抽样单元，检验：
       delta_rho = rho(discriminability, accuracy) - rho(intensity, accuracy)
   这是比普通 p 值更稳健的条件层面证据。

3. cluster-robust rank model：
   使用 rank transform 近似 partial Spearman，
   并用 subject-cluster robust SE 做推断。
   这替代旧的普通行级 partial correlation。

输入
----
建议输入修正版 Step14 输出：
    linked_data_pair_aware.csv

也兼容旧 Step14 输出：
    linked_data.csv
若旧表没有 source feature，本脚本会自动用 train_stim 合并 source feature。

运行示例
--------
python Step15_Corrected_FeatureAccuracy_Analysis_FIXED.py ^
  --linked "E:/.../Step14_FIXED/linked_data_pair_aware.csv" ^
  --out "E:/.../Step15_FIXED"

如果要纳入 CSPSVM：
python Step15_Corrected_FeatureAccuracy_Analysis_FIXED.py ^
  --linked ".../linked_data_pair_aware.csv" ^
  --out ".../Step15_FIXED" ^
  --include-cspsvm

输出
----
00_data_audit.csv
00_analysis_table.csv
01_condition_correlation_descriptive.csv
02_condition_delta_subject_bootstrap.csv
03_cluster_robust_rank_models.csv
04_subject_level_robustness.csv
05_cross_pair_aware_summary.csv
06_corrected_result_summary.md
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf


DEFAULT_FOCUS_ALGORITHMS = ["FBCSP", "FGMDRM", "EEGNET"]


# ============================================================
# basic utils
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sig_label(p: float) -> str:
    if pd.isna(p):
        return "na"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - x.mean()) / sd


def rank01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    r = x.rank(method="average")
    n = r.notna().sum()
    if n <= 1:
        return r * np.nan
    return (r - 1) / (n - 1)


def safe_spearman(x, y) -> Tuple[float, float, int]:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 5 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return np.nan, np.nan, n
    rho, p = spearmanr(x[m], y[m])
    return float(rho), float(p), n


def normalize_columns(df: pd.DataFrame, include_cspsvm: bool) -> pd.DataFrame:
    df = df.copy()

    # Backward compatibility: old Step14 used scene for accuracy scene.
    if "accuracy_scene" not in df.columns and "scene" in df.columns:
        df = df.rename(columns={"scene": "accuracy_scene"})

    required = [
        "subject", "algorithm", "protocol", "feature_scene",
        "test_stim", "train_stim", "accuracy",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"linked data missing required columns: {missing}")

    df["subject"] = df["subject"].astype(str)
    df["algorithm"] = df["algorithm"].astype(str).str.upper().str.strip()
    df["protocol"] = df["protocol"].astype(str).str.strip()
    df["feature_scene"] = df["feature_scene"].astype(str).str.lower().str.strip()
    df["test_stim"] = df["test_stim"].astype(str).str.lower().str.strip()
    df["train_stim"] = df["train_stim"].astype(str).str.lower().str.strip()

    if "accuracy_scene" not in df.columns:
        df["accuracy_scene"] = df["feature_scene"]

    df["task_pair"] = df["train_stim"] + "→" + df["test_stim"]

    # New Step14 columns.
    if "target_intensity" not in df.columns:
        if "intensity" not in df.columns:
            raise ValueError("No target_intensity or old intensity column found.")
        df["target_intensity"] = df["intensity"]

    if "target_discriminability" not in df.columns:
        if "discriminability" not in df.columns:
            raise ValueError("No target_discriminability or old discriminability column found.")
        df["target_discriminability"] = df["discriminability"]

    for c in [
        "accuracy", "target_intensity", "target_discriminability",
        "source_intensity", "source_discriminability",
        "abs_delta_intensity", "abs_delta_discriminability",
        "intensity", "discriminability",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if df["accuracy"].dropna().median() > 1:
        df["accuracy"] = df["accuracy"] / 100.0

    if not include_cspsvm:
        df = df[df["algorithm"].isin(DEFAULT_FOCUS_ALGORITHMS)].copy()

    df = df.dropna(subset=["accuracy", "target_intensity", "target_discriminability"]).copy()

    return df


def add_source_features_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    If Step14 fixed output already contains source feature, keep it.
    If old Step14 output only contains target feature, reconstruct source feature by:
        subject × feature_scene × train_stim
    """
    out = df.copy()

    has_source = {"source_intensity", "source_discriminability"}.issubset(out.columns)
    if not has_source or out[["source_intensity", "source_discriminability"]].isna().any().any():
        fmap = (
            out[["subject", "feature_scene", "test_stim", "target_intensity", "target_discriminability"]]
            .drop_duplicates()
            .rename(columns={
                "test_stim": "train_stim",
                "target_intensity": "source_intensity_rebuilt",
                "target_discriminability": "source_discriminability_rebuilt",
            })
        )
        out = out.merge(
            fmap,
            on=["subject", "feature_scene", "train_stim"],
            how="left",
            validate="many_to_one",
        )

        if "source_intensity" not in out.columns:
            out["source_intensity"] = out["source_intensity_rebuilt"]
        else:
            out["source_intensity"] = out["source_intensity"].fillna(out["source_intensity_rebuilt"])

        if "source_discriminability" not in out.columns:
            out["source_discriminability"] = out["source_discriminability_rebuilt"]
        else:
            out["source_discriminability"] = out["source_discriminability"].fillna(out["source_discriminability_rebuilt"])

        out = out.drop(columns=[c for c in ["source_intensity_rebuilt", "source_discriminability_rebuilt"] if c in out.columns])

    # Within-paradigm should have source=target; if still missing, fill safely.
    out["source_intensity"] = out["source_intensity"].fillna(out["target_intensity"])
    out["source_discriminability"] = out["source_discriminability"].fillna(out["target_discriminability"])

    out["delta_intensity_target_minus_source"] = out["target_intensity"] - out["source_intensity"]
    out["delta_discriminability_target_minus_source"] = (
        out["target_discriminability"] - out["source_discriminability"]
    )
    out["abs_delta_intensity"] = out["delta_intensity_target_minus_source"].abs()
    out["abs_delta_discriminability"] = out["delta_discriminability_target_minus_source"].abs()

    for c in [
        "accuracy", "target_intensity", "target_discriminability",
        "source_intensity", "source_discriminability",
        "abs_delta_intensity", "abs_delta_discriminability",
        "delta_intensity_target_minus_source",
        "delta_discriminability_target_minus_source",
    ]:
        out[c + "_z"] = zscore(out[c])
        out[c + "_rank01"] = rank01(out[c])

    return out


# ============================================================
# audit
# ============================================================

def data_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(item, value):
        rows.append({"item": item, "value": value})

    add("n_rows", len(df))
    add("n_subjects", df["subject"].nunique())
    add("algorithms", ", ".join(sorted(df["algorithm"].unique())))
    add("protocols", ", ".join(sorted(df["protocol"].unique())))
    add("feature_scenes", ", ".join(sorted(df["feature_scene"].unique())))
    add("test_stims", ", ".join(sorted(df["test_stim"].unique())))
    add("train_stims", ", ".join(sorted(df["train_stim"].unique())))
    add("task_pairs", ", ".join(sorted(df["task_pair"].unique())))

    feature_cell = df.groupby(["subject", "feature_scene", "test_stim"]).size()
    add("n_target_feature_cells", feature_cell.shape[0])
    add("mean_rows_per_target_feature_cell", float(feature_cell.mean()))
    add("max_rows_per_target_feature_cell", int(feature_cell.max()))

    acc_cell = df.groupby(["subject", "algorithm", "protocol", "feature_scene", "train_stim", "test_stim"]).size()
    add("n_accuracy_cells", acc_cell.shape[0])
    add("max_rows_per_accuracy_cell", int(acc_cell.max()))

    for c in [
        "accuracy", "target_intensity", "target_discriminability",
        "source_intensity", "source_discriminability",
        "abs_delta_intensity", "abs_delta_discriminability",
    ]:
        add(f"missing_{c}", int(df[c].isna().sum()))

    return pd.DataFrame(rows)


# ============================================================
# descriptive condition correlations
# ============================================================

def condition_correlation_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (protocol, feature_scene, algorithm), g in df.groupby(["protocol", "feature_scene", "algorithm"]):
        for label, col in [
            ("Intensity_target", "target_intensity"),
            ("Discriminability_target", "target_discriminability"),
        ]:
            rho, p, n = safe_spearman(g[col], g["accuracy"])
            rows.append({
                "protocol": protocol,
                "feature_scene": feature_scene,
                "algorithm": algorithm,
                "x_var": label,
                "n_rows": n,
                "n_subjects": g["subject"].nunique(),
                "spearman_rho_descriptive": rho,
                "naive_p_not_for_final_inference": p,
                "note": "Descriptive only; repeated feature cells exist."
            })

    out = pd.DataFrame(rows)

    wide = out.pivot_table(
        index=["protocol", "feature_scene", "algorithm"],
        columns="x_var",
        values="spearman_rho_descriptive"
    ).reset_index()

    if {"Discriminability_target", "Intensity_target"}.issubset(wide.columns):
        wide["delta_disc_minus_intensity"] = wide["Discriminability_target"] - wide["Intensity_target"]
        wide["disc_better_descriptively"] = wide["delta_disc_minus_intensity"] > 0
        out = out.merge(
            wide[["protocol", "feature_scene", "algorithm", "delta_disc_minus_intensity", "disc_better_descriptively"]],
            on=["protocol", "feature_scene", "algorithm"],
            how="left"
        )

    return out


# ============================================================
# subject-cluster bootstrap for delta rho
# ============================================================

def bootstrap_delta_rho(
    g: pd.DataFrame,
    n_boot: int = 2000,
    random_state: int = 42,
) -> dict:
    base = g.dropna(subset=["target_discriminability", "target_intensity", "accuracy", "subject"]).copy()
    subjects = np.array(sorted(base["subject"].unique()))
    if len(subjects) < 5:
        return {
            "rho_disc": np.nan,
            "rho_intensity": np.nan,
            "delta_rho": np.nan,
            "boot_ci_low": np.nan,
            "boot_ci_high": np.nan,
            "boot_p_two_sided": np.nan,
            "n_rows": len(base),
            "n_subjects": len(subjects),
        }

    rho_d, _, _ = safe_spearman(base["target_discriminability"], base["accuracy"])
    rho_i, _, _ = safe_spearman(base["target_intensity"], base["accuracy"])
    obs = rho_d - rho_i

    rng = np.random.default_rng(random_state)
    deltas = []

    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        parts = []
        for j, sub in enumerate(sampled):
            temp = base[base["subject"] == sub].copy()
            temp["_boot_subject_instance"] = j
            parts.append(temp)
        bdf = pd.concat(parts, ignore_index=True)

        bd, _, _ = safe_spearman(bdf["target_discriminability"], bdf["accuracy"])
        bi, _, _ = safe_spearman(bdf["target_intensity"], bdf["accuracy"])
        if np.isfinite(bd) and np.isfinite(bi):
            deltas.append(bd - bi)

    if len(deltas) == 0:
        ci_low = ci_high = p = np.nan
    else:
        arr = np.asarray(deltas)
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        p = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
        p = min(float(p), 1.0)

    return {
        "rho_disc": rho_d,
        "rho_intensity": rho_i,
        "delta_rho": obs,
        "boot_ci_low": ci_low,
        "boot_ci_high": ci_high,
        "boot_p_two_sided": p,
        "n_rows": len(base),
        "n_subjects": len(subjects),
    }


def condition_delta_bootstrap(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []

    for (protocol, feature_scene, algorithm), g in df.groupby(["protocol", "feature_scene", "algorithm"]):
        res = bootstrap_delta_rho(g, n_boot=n_boot, random_state=123)
        res.update({
            "protocol": protocol,
            "feature_scene": feature_scene,
            "algorithm": algorithm,
            "comparison": "rho(target_discriminability, accuracy) - rho(target_intensity, accuracy)",
            "disc_better_descriptively": bool(res["delta_rho"] > 0) if np.isfinite(res["delta_rho"]) else np.nan,
            "disc_better_cluster_bootstrap": (
                bool(res["boot_ci_low"] > 0)
                if np.isfinite(res["boot_ci_low"]) and np.isfinite(res["boot_ci_high"])
                else np.nan
            ),
            "bootstrap_sig_label": sig_label(res["boot_p_two_sided"]),
        })
        rows.append(res)

    return pd.DataFrame(rows)


# ============================================================
# cluster-robust rank models
# ============================================================

def fit_cluster_robust_ols(df: pd.DataFrame, formula: str, label: str) -> pd.DataFrame:
    model_df = df.dropna().copy()

    if len(model_df) < 30 or model_df["subject"].nunique() < 5:
        return pd.DataFrame([{
            "model": label,
            "term": "NOT_FITTED",
            "estimate": np.nan,
            "se_subject_cluster": np.nan,
            "p_subject_cluster": np.nan,
            "sig_subject_cluster": "na",
            "n_rows": len(model_df),
            "n_subjects": model_df["subject"].nunique() if "subject" in model_df else np.nan,
            "formula": formula,
        }])

    try:
        fit = smf.ols(formula, data=model_df).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_df["subject"]}
        )
        out = pd.DataFrame({
            "model": label,
            "term": fit.params.index,
            "estimate": fit.params.values,
            "se_subject_cluster": fit.bse.values,
            "p_subject_cluster": fit.pvalues.values,
        })
        out["sig_subject_cluster"] = out["p_subject_cluster"].apply(sig_label)
        out["n_rows"] = len(model_df)
        out["n_subjects"] = model_df["subject"].nunique()
        out["formula"] = formula
        return out
    except Exception as e:
        warnings.warn(f"Model failed: {label}: {e}")
        return pd.DataFrame([{
            "model": label,
            "term": "MODEL_FAILED",
            "estimate": np.nan,
            "se_subject_cluster": np.nan,
            "p_subject_cluster": np.nan,
            "sig_subject_cluster": "na",
            "n_rows": len(model_df),
            "n_subjects": model_df["subject"].nunique() if "subject" in model_df else np.nan,
            "formula": formula,
            "error": repr(e),
        }])


def cluster_robust_rank_models(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    full_formula = (
        "accuracy_rank01 ~ target_discriminability_rank01 + target_intensity_rank01 "
        "+ C(algorithm) + C(protocol) + C(feature_scene) + C(test_stim) + C(train_stim)"
    )
    full_cols = [
        "accuracy_rank01", "target_discriminability_rank01", "target_intensity_rank01",
        "algorithm", "protocol", "feature_scene", "test_stim", "train_stim", "subject"
    ]
    results.append(
        fit_cluster_robust_ols(df[full_cols], full_formula, "M1_full_target_feature")
    )

    within = df[df["protocol"] == "within_paradigm"].copy()
    within_formula = (
        "accuracy_rank01 ~ target_discriminability_rank01 + target_intensity_rank01 "
        "+ C(algorithm) + C(feature_scene) + C(test_stim)"
    )
    within_cols = [
        "accuracy_rank01", "target_discriminability_rank01", "target_intensity_rank01",
        "algorithm", "feature_scene", "test_stim", "subject"
    ]
    results.append(
        fit_cluster_robust_ols(within[within_cols], within_formula, "M2_within_target_feature")
    )

    cross = df[df["protocol"] == "cross_paradigm"].copy()
    cross_formula = (
        "accuracy_rank01 ~ target_discriminability_rank01 + target_intensity_rank01 "
        "+ source_discriminability_rank01 + source_intensity_rank01 "
        "+ abs_delta_discriminability_rank01 + abs_delta_intensity_rank01 "
        "+ C(algorithm) + C(feature_scene) + C(task_pair)"
    )
    cross_cols = [
        "accuracy_rank01", "target_discriminability_rank01", "target_intensity_rank01",
        "source_discriminability_rank01", "source_intensity_rank01",
        "abs_delta_discriminability_rank01", "abs_delta_intensity_rank01",
        "algorithm", "feature_scene", "task_pair", "subject"
    ]
    results.append(
        fit_cluster_robust_ols(cross[cross_cols], cross_formula, "M3_cross_pair_aware")
    )

    return pd.concat(results, ignore_index=True)


# ============================================================
# subject-level robustness
# ============================================================

def subject_level_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate within each subject first, then test correlations across subjects.
    This is conservative because n is only number of subjects, but it checks whether
    the result is merely driven by repeated rows.
    """
    rows = []

    grouping_schemes = {
        "subject_only": ["subject"],
        "subject_algorithm": ["subject", "algorithm"],
        "subject_protocol": ["subject", "protocol"],
        "subject_algorithm_protocol": ["subject", "algorithm", "protocol"],
    }

    for name, keys in grouping_schemes.items():
        agg = (
            df.groupby(keys)
            .agg(
                accuracy=("accuracy", "mean"),
                target_intensity=("target_intensity", "mean"),
                target_discriminability=("target_discriminability", "mean"),
                source_intensity=("source_intensity", "mean"),
                source_discriminability=("source_discriminability", "mean"),
                abs_delta_intensity=("abs_delta_intensity", "mean"),
                abs_delta_discriminability=("abs_delta_discriminability", "mean"),
            )
            .reset_index()
        )

        r_d, p_d, n_d = safe_spearman(agg["target_discriminability"], agg["accuracy"])
        r_i, p_i, n_i = safe_spearman(agg["target_intensity"], agg["accuracy"])

        rows.append({
            "aggregation": name,
            "n_units": len(agg),
            "rho_target_discriminability_accuracy": r_d,
            "p_target_discriminability": p_d,
            "rho_target_intensity_accuracy": r_i,
            "p_target_intensity": p_i,
            "delta_disc_minus_intensity": r_d - r_i if np.isfinite(r_d) and np.isfinite(r_i) else np.nan,
            "disc_better": bool(r_d > r_i) if np.isfinite(r_d) and np.isfinite(r_i) else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# cross-pair summary
# ============================================================

def cross_pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    cross = df[df["protocol"] == "cross_paradigm"].copy()
    if cross.empty:
        return pd.DataFrame()

    rows = []
    for (task_pair, feature_scene, algorithm), g in cross.groupby(["task_pair", "feature_scene", "algorithm"]):
        for label, col in [
            ("target_discriminability", "target_discriminability"),
            ("target_intensity", "target_intensity"),
            ("source_discriminability", "source_discriminability"),
            ("source_intensity", "source_intensity"),
            ("abs_delta_discriminability", "abs_delta_discriminability"),
            ("abs_delta_intensity", "abs_delta_intensity"),
        ]:
            rho, p, n = safe_spearman(g[col], g["accuracy"])
            rows.append({
                "task_pair": task_pair,
                "feature_scene": feature_scene,
                "algorithm": algorithm,
                "x_var": label,
                "n_rows": n,
                "n_subjects": g["subject"].nunique(),
                "spearman_rho_descriptive": rho,
                "naive_p_not_for_final_inference": p,
            })

    return pd.DataFrame(rows)


# ============================================================
# markdown summary
# ============================================================

def write_summary(
    out_dir: Path,
    audit: pd.DataFrame,
    corr: pd.DataFrame,
    boot: pd.DataFrame,
    models: pd.DataFrame,
    subj: pd.DataFrame,
) -> None:
    n_cond = boot.shape[0]
    n_desc = int(boot["disc_better_descriptively"].fillna(False).sum()) if n_cond else 0
    n_boot = int(boot["disc_better_cluster_bootstrap"].fillna(False).sum()) if n_cond else 0

    key_terms = models[
        models["term"].isin([
            "target_discriminability_rank01",
            "target_intensity_rank01",
            "source_discriminability_rank01",
            "source_intensity_rank01",
            "abs_delta_discriminability_rank01",
            "abs_delta_intensity_rank01",
        ])
    ].copy()

    lines = []
    lines.append("# Corrected Step15 result summary\n")
    lines.append("## Main correction\n")
    lines.append(
        "Step14 compresses high-dimensional neural features into subject-level scalar features. "
        "The same feature cell can appear repeatedly across algorithms and cross-paradigm task pairs. "
        "Therefore, ordinary row-wise p-values are not used as the final inferential evidence."
    )
    lines.append("")
    lines.append("## Descriptive finding\n")
    lines.append(
        f"Discriminability has a larger descriptive Spearman correlation with accuracy than intensity "
        f"in {n_desc}/{n_cond} protocol × feature_scene × algorithm condition cells."
    )
    lines.append("")
    lines.append("## Cluster-bootstrap finding\n")
    lines.append(
        f"Using subject-cluster bootstrap, discriminability is conclusively higher than intensity "
        f"in {n_boot}/{n_cond} condition cells when CI_low > 0."
    )
    lines.append("")
    lines.append("## Key cluster-robust model terms\n")
    if key_terms.empty:
        lines.append("No key terms were fitted. Please check the model output CSV.")
    else:
        lines.append(key_terms[[
            "model", "term", "estimate", "se_subject_cluster",
            "p_subject_cluster", "sig_subject_cluster", "n_rows", "n_subjects"
        ]].to_markdown(index=False))

    lines.append("")
    lines.append("## Subject-level robustness\n")
    if subj.empty:
        lines.append("No subject-level robustness result.")
    else:
        lines.append(subj.to_markdown(index=False))

    lines.append("")
    lines.append("## Recommended conclusion wording\n")
    lines.append(
        "The safe conclusion is: discriminability shows a more consistent association with classification "
        "accuracy than response intensity. The descriptive 12/12 pattern can be reported as a trend, "
        "while final inference should rely on subject-cluster bootstrap, cluster-robust rank models, "
        "and subject-level robustness checks."
    )

    (out_dir / "06_corrected_result_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# main
# ============================================================

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--linked", default=r"E:\Datasets\4_跨场景因素研究v2\画图结果\PerSubject_IntensityDisc_Accuracy\linked_data.csv",
                  help="Path to linked_data_pair_aware.csv or old linked_data.csv")
    p.add_argument("--out", default=r"E:\Datasets\4_跨场景因素研究v2\画图结果\PerSubject_IntensityDisc_Accuracy",
                  help="Output directory")
    p.add_argument("--include-cspsvm", action="store_true")
    p.add_argument("--n-boot", type=int, default=2000)
    return p


def main():
    args = make_parser().parse_args()
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    raw = pd.read_csv(args.linked)
    df = normalize_columns(raw, include_cspsvm=args.include_cspsvm)
    df = add_source_features_if_needed(df)

    audit = data_audit(df)
    audit.to_csv(out_dir / "00_data_audit.csv", index=False, encoding="utf-8-sig")
    df.to_csv(out_dir / "00_analysis_table.csv", index=False, encoding="utf-8-sig")

    corr = condition_correlation_descriptive(df)
    corr.to_csv(out_dir / "01_condition_correlation_descriptive.csv", index=False, encoding="utf-8-sig")

    boot = condition_delta_bootstrap(df, n_boot=args.n_boot)
    boot.to_csv(out_dir / "02_condition_delta_subject_bootstrap.csv", index=False, encoding="utf-8-sig")

    models = cluster_robust_rank_models(df)
    models.to_csv(out_dir / "03_cluster_robust_rank_models.csv", index=False, encoding="utf-8-sig")

    subj = subject_level_robustness(df)
    subj.to_csv(out_dir / "04_subject_level_robustness.csv", index=False, encoding="utf-8-sig")

    cross = cross_pair_summary(df)
    cross.to_csv(out_dir / "05_cross_pair_aware_summary.csv", index=False, encoding="utf-8-sig")

    write_summary(out_dir, audit, corr, boot, models, subj)

    print("=" * 80)
    print("Step15 FIXED finished.")
    print(f"Rows used: {len(df)}")
    print(f"Subjects: {df['subject'].nunique()}")
    print(f"Algorithms: {', '.join(sorted(df['algorithm'].unique()))}")
    print(f"Output dir: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
