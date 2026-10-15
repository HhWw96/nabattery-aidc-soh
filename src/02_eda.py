# -*- coding: utf-8 -*-
"""
02_eda.py  EDA 第一版:四工况组容量衰减曲线(帖子 1 / PPT 素材)
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

cap = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "capacity_clean.csv"))
cap = cap[cap.group != "4C_mixed"]   # 变倍率组不进主图,仅作备选分析

GROUP_META = {
    "24C_2A": ("24 ℃ / 2A 常温基准(训练组)", "#1f77b4"),
    "24C_4A": ("24 ℃ / 4A 常温高倍率", "#2ca02c"),
    "43C_4A": ("43 ℃ / 4A 高温", "#d62728"),
    "4C_2A":  ("4 ℃ / 2A 低温", "#9467bd"),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for g, (label, color) in GROUP_META.items():
    sub = cap[cap.group == g]
    for bid, s in sub.groupby("battery"):
        axes[0].plot(s.cycle, s.capacity_ah, color=color, alpha=0.75, lw=1.2)
        axes[1].plot(s.cycle, s.soh_rated, color=color, alpha=0.75, lw=1.2)
    # 组均值线
    mean_c = sub.groupby("cycle")["capacity_ah"].mean()
    mean_s = sub.groupby("cycle")["soh_rated"].mean()
    axes[0].plot(mean_c.index, mean_c.values, color=color, lw=2.6, label=label)
    axes[1].plot(mean_s.index, mean_s.values, color=color, lw=2.6, label=label)

axes[0].axhline(1.6, color="gray", ls="--", lw=1)
axes[0].text(2, 1.62, "80% 额定容量 (1.6 Ah)", fontsize=9, color="gray")
axes[0].set_xlabel("放电循环数")
axes[0].set_ylabel("放电容量 (Ah)")
axes[0].set_title("容量衰减曲线:四种工况")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].axhline(0.8, color="gray", ls="--", lw=1)
axes[1].text(2, 0.805, "SOH = 80%(寿命终点常用阈值)", fontsize=9, color="gray")
axes[1].set_xlabel("放电循环数")
axes[1].set_ylabel("SOH(容量 / 2Ah 额定)")
axes[1].set_title("SOH 轨迹:低温与高倍率衰减形态差异显著")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

fig.suptitle("NASA PCoE 18650 老化数据 · 16 只电芯 · 4 种温度×倍率工况(已剔除异常段)", fontsize=12)
fig.tight_layout()
out = os.path.join(FIG, "eda_capacity_fade.png")
fig.savefig(out, dpi=160)
print("saved:", out)

# 关键统计:各组达到 SOH 80% 的循环数(按组均值曲线)
print("\n各组均值曲线到达 SOH 80% 的循环数:")
for g in GROUP_META:
    s = cap[cap.group == g].groupby("cycle")["soh_rated"].mean()
    below = s[s <= 0.8]
    print(f"  {g}: {'第 ' + str(below.index[0]) + ' 循环' if len(below) else '实验内未跌破'}")
