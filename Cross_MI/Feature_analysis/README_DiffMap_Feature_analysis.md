# 差异图谱分析脚本说明

这组脚本用于把分析重点从“单类特征强度”转向“类别差异图谱”，适配原仓库 `Feature_analysis` 的分步脚本风格。

## 文件

1. `Step6_DiffMap_save.py`
   - 输入：已有 `TOPO_*.mat` 和 `TF_*.mat`
   - 输出：`DiffMap_*.mat`
   - 特征：mean_diff、abs_strength、Cohen's dz、Fisher score、t值、p值、FDR显著性

2. `Step7_DiffMap_plot.py`
   - 输入：`DiffMap_*.mat`
   - 输出：差异脑地形图、差异时频图、效应量图、Fisher图、FDR显著性图

3. `Step8_DiffMap_similarity_RSA.py`
   - 输入：多个范式的 `DiffMap_*.mat`
   - 输出：跨范式差异图谱相似性矩阵和 CSV
   - 方法：Pearson、Spearman、Cosine、linear CKA

4. `Step9_DiffMap_cluster_permutation.py`
   - 输入：单个范式的 `DiffMap_*.mat`
   - 输出：时频差异图的 cluster-based permutation 显著区域

## 推荐运行顺序

```bash
python Step6_DiffMap_save.py
python Step7_DiffMap_plot.py
python Step8_DiffMap_similarity_RSA.py
python Step9_DiffMap_cluster_permutation.py
```

## 重要解释

- `mean_diff = class1 - class2`：类别差异方向。
- `abs_strength = mean(abs(class1 - class2))`：不关心方向的差异强度。
- `Cohen's dz`：被试内配对效应量，更适合跨范式比较。
- `Fisher score`：分类可分性指标，比单纯功率强度更贴近分类。
- `RSA/CKA similarity`：检验不同范式是否保留相似的类别差异结构。
- `cluster permutation`：比逐点 FDR 更适合时频图连续显著区域检验。
