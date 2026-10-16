# -*- coding: utf-8 -*-
"""
19_v16_feature_importance.py  V1.6:最终模型的特征重要性(问答弹药 + 可解释性留痕)

方法: 置换重要性(permutation importance), 24C_2A 留一折的最终模型, 在测试集上评估
产出: results/v16_importance.csv, figures/v16_importance.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, "..", "figures"); os.makedirs(FIG, exist_ok=True)

BASE_FEATS = ["cycle", "ambient_temp", "Re", "Rct",
              "dis_duration_s", "t_drop_36_30_s", "dis_temp_mean", "dis_current_mean",
              "ic_pos_max", "ic_height_max", "ic_area", "ic_mean", "ic_std",
              "cc_capacity_ah", "cc_duration_s", "cv_duration_s", "charge_temp_mean"]
LAG_FEATS = ["cap_lag1", "cap_lag3_mean", "cap_roll_slope"]
FEATURES = BASE_FEATS + LAG_FEATS
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=10, l2_regularization=1.0, random_state=42)

CN = {"cycle": "循环数", "ambient_temp": "环境温度", "Re": "欧姆内阻 Re", "Rct": "电荷转移阻抗 Rct",
      "dis_duration_s": "放电时长", "t_drop_36_30_s": "3.6→3.0V 等压降时间", "dis_temp_mean": "放电均温",
      "dis_current_mean": "放电均流", "ic_pos_max": "IC 峰位置", "ic_height_max": "IC 峰高度",
      "ic_area": "IC 曲线下面积", "ic_mean": "IC 均值", "ic_std": "IC 波动",
      "cc_capacity_ah": "CC 段充入容量", "cc_duration_s": "CC 充电时长", "cv_duration_s": "CV 充电时长",
      "charge_temp_mean": "充电均温", "cap_lag1": "上一循环容量", "cap_lag3_mean": "近 3 循环容量均值",
      "cap_roll_slope": "滚动衰减斜率"}


def add_lag(df):
    df = df.sort_values(["battery", "cycle"]).copy()
    g = df.groupby("battery")["capacity_ah"]
    df["cap_lag1"] = g.shift(1)
    df["cap_lag3_mean"] = g.shift(1).rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df["cap_roll_slope"] = (g.shift(1).rolling(5, min_periods=3).mean().reset_index(0, drop=True)
                            - g.shift(1).rolling(20, min_periods=5).mean().reset_index(0, drop=True))
    for c in LAG_FEATS:
        df[c] = df.groupby("battery")[c].transform(lambda s: s.bfill())
    return df


def main():
    df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "feature_table.csv"))
    df = df[df.group != "4C_mixed"].reset_index(drop=True)
    df = add_lag(df)
    tr = df[df.group != "24C_2A"]; te = df[df.group == "24C_2A"]

    variants = {"full": FEATURES, "no_lag": BASE_FEATS}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=False)
    for ax, (tag, feats) in zip(axes, variants.items()):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **PARAMS).fit(tr[feats], tr.capacity_ah)
        imp = permutation_importance(m, te[feats], te.capacity_ah, n_repeats=8,
                                     random_state=42, scoring="neg_mean_absolute_error")
        s = pd.Series(-imp.importances_mean, index=[CN[f] for f in feats]).sort_values()
        s.to_csv(os.path.join(RES, f"v16_importance_{tag}.csv"), encoding="utf-8-sig")
        colors = ["#d62728" if ("循环容量" in k or "斜率" in k or "上一循环" in k)
                  else "#1f77b4" if ("IC" in k or "内阻" in k or "阻抗" in k) else "#7f7f7f" for k in s.index]
        ax.barh(s.index, s.values, color=colors, alpha=0.85)
        title = "有滞后历史(全特征)" if tag == "full" else "无历史(仅曲线/物理特征)"
        ax.set_title(f"{title}\n红=滞后历史 蓝=曲线物理特征 灰=工况统计", fontsize=10.5)
        ax.set_xlabel("置换重要性(MAE 增量, Ah)")
        ax.grid(alpha=0.3, axis="x")
        print(f"[{tag}] top5:\n{s.sort_values(ascending=False).head(5).round(4).to_string()}\n")
    fig.suptitle("V1.6 特征重要性:容量惯性 vs 曲线物理特征(24 ℃/2A 留一折)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v16_importance.png"), dpi=160)
    print("图已保存: figures/v16_importance.png")


if __name__ == "__main__":
    main()
