# -*- coding: utf-8 -*-
"""
11_roadshow_fig.py  路演核心四联图(1 页 PPT 的主视觉)

四格叙事:
  (a) 同工况:预测精准(留一电芯,校准区间)
  (b) 跨工况:从崩溃到救赎(4 ℃ MAE 阶梯 0.299→0.176→0.088)
  (c) 安全机制:OOD 检测器自动标记未见过工况(标记组=失败组)
  (d) 闭环收益:自适应限流总吞吐 +11.2%
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")

FEATURES = ["cycle", "ambient_temp", "Re", "Rct",
            "dis_duration_s", "t_drop_36_30_s", "dis_temp_mean", "dis_current_mean",
            "ic_pos_max", "ic_height_max", "ic_area", "ic_mean", "ic_std",
            "cc_capacity_ah", "cc_duration_s", "cv_duration_s", "charge_temp_mean"]
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=10, l2_regularization=1.0, random_state=42)

df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "feature_table.csv"))
df = df[df.group != "4C_mixed"].reset_index(drop=True)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ---- (a) 同工况精度: B0005 留一电芯 + 校准区间 ----
ax = axes[0, 0]
tr = df[(df.group == "24C_2A") & (df.battery != "B0005")]
te = df[df.battery == "B0005"]
models = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **PARAMS).fit(tr[FEATURES], tr.capacity_ah)
          for q in (0.05, 0.5, 0.9)}
delta = 0.171  # V0.0.2 保形校准量(训练集 LOBO 残差 90% 分位)
p50 = models[0.5].predict(te[FEATURES])
p05 = models[0.05].predict(te[FEATURES]) - delta
p90 = models[0.9].predict(te[FEATURES]) + delta
ax.plot(te.cycle, te.capacity_ah, "k-", lw=2, label="实测容量")
ax.plot(te.cycle, p50, color="#d62728", lw=1.5, label="AI 预测(中位数)")
ax.fill_between(te.cycle, p05, p90, color="#d62728", alpha=0.18, label="90% 校准区间")
ax.set_title("(a) 同工况:电池「体检」已可实用\nMAE≈0.02 Ah(≈1%),区间覆盖率校准后 95%", fontsize=11)
ax.set_xlabel("循环数"); ax.set_ylabel("容量 (Ah)")
ax.legend(fontsize=8.5, loc="lower left"); ax.grid(alpha=0.3)

# ---- (b) 4 ℃ 误差阶梯 ----
ax = axes[0, 1]
steps = ["无适配\n(V0.0.2)", "首循环归一化\n(V0.0.6 K=1)", "+前 10 循环校正\n(V0.0.6 K=10)"]
vals = [0.299, 0.176, 0.088]
colors = ["#7f7f7f", "#1f77b4", "#d62728"]
bars = ax.bar(steps, vals, color=colors, alpha=0.85, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("4 ℃ 工况预测 MAE (Ah)")
ax.set_title("(b) 跨工况:从「自信地错」到可运维\n前 10 循环冷启动,误差降低 71%", fontsize=11)
ax.set_ylim(0, 0.34); ax.grid(alpha=0.3, axis="y")

# ---- (c) OOD 检测器 ----
ax = axes[1, 0]
ood = pd.read_csv(os.path.join(RES, "v07_ood.csv"))
labels = {"24C_2A": "24℃/2A", "24C_4A": "24℃/4A", "43C_4A": "43℃/4A", "4C_2A": "4℃/2A"}
xs = np.arange(len(ood))
dists = ood.test_dist_mean.values
threshs = ood.threshold.values
flagged = ood.flagged_ratio.values > 0.5
ax.bar(xs, dists, 0.5, color=["#d62728" if f else "#2ca02c" for f in flagged], alpha=0.85)
ax.plot(xs, threshs, "k--", lw=1.5, marker="o", ms=4, label="拒识阈值")
for x, d, f in zip(xs, dists, flagged):
    ax.text(x, d + 1.5, "拒识" if f else "放行", ha="center", fontsize=9,
            color="#d62728" if f else "#2ca02c", fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([labels[g] for g in ood.test_group], fontsize=9)
ax.set_ylabel("工况指纹距离")
ax.set_title("(c) 安全机制:分布外检测器自动拒识\n被标记的组 = 预测失败的组,一一对应", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

# ---- (d) 闭环收益 ----
ax = axes[1, 1]
vals = [182.7, 203.2]
bars = ax.bar(["固定 2A 策略", "自适应限流\n(预测区间下限驱动)"], vals,
              color=["#1f77b4", "#d62728"], alpha=0.85, width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f} Ah", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("寿命期总吞吐电量 (Ah)")
ax.set_ylim(0, 235)
ax.set_title("(d) 闭环收益:预测反哺策略\n寿命期总吞吐 +11.2%(概念验证仿真)", fontsize=11)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("让钠电储能自己会「体检」:AI 预测寿命,提前躲开安全风险\nNASA PCoE 19 只电芯 · 4 种工况 · 五轮迭代验证(风象 08 天秤座)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(FIG, "roadshow_core.png")
fig.savefig(out, dpi=170)
print("saved:", out)
