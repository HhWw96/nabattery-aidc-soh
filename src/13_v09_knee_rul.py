# -*- coding: utf-8 -*-
"""
13_v09_knee_rul.py  V0.0.9:拐点感知退化模型 —— 物理模型 vs ML 的 RUL 对决

文献经典的 RUL 路线: 对容量衰减轨迹拟合参数化退化模型, 外推寿命终点(EOL)。
  双指数模型(饱和+加速两项): capacity(t) = a·exp(b·t) + c·exp(d·t)
  拐点分段模型: capacity(t) = p0 + p1·t + p2·max(0, t - t_knee)

实验: 只用「前半生」(到达 EOL 前的 50%/70% 循环)拟合 → 外推 EOL 循环 → 与真值对比
对比: V0.0.7 的 ML RUL 头(逐快照估计) vs 参数化模型(轨迹外推)
产出: results/v09_knee_rul.csv, figures/v09_knee_rul.png
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, "..", "figures"); os.makedirs(FIG, exist_ok=True)


def double_exp(t, a, b, c, d):
    return a * np.exp(b * t) + c * np.exp(d * t)


def knee_model(t, p0, p1, p2, t_knee):
    return p0 + p1 * t + p2 * np.maximum(0, t - t_knee)


def fit_and_extrapolate(t, y, cap0):
    """双指数 + 拐点模型拟合, 外推至 y = 0.8*cap0 的循环数"""
    eol_true = 0.8 * cap0
    out = {}
    # 双指数
    try:
        p, _ = curve_fit(double_exp, t, y, p0=[y[0], -1e-3, 0.1, -2e-2],
                         bounds=([0, -0.5, 0, -0.5], [3, 0, 2, 0]), maxfev=20000)
        t_grid = np.arange(1, 600)
        yg = double_exp(t_grid, *p)
        below = t_grid[yg < eol_true]
        out["double_exp"] = int(below[0]) if len(below) else np.nan
        out["_de_curve"] = (t_grid, yg)
    except Exception:
        out["double_exp"] = np.nan
    # 拐点模型
    try:
        p, _ = curve_fit(knee_model, t, y, p0=[y[0], -1e-3, -5e-3, t[len(t) // 2]],
                         bounds=([0, -0.1, -0.1, 1], [3, 0, 0, 600]), maxfev=20000)
        t_grid = np.arange(1, 600)
        yg = knee_model(t_grid, *p)
        below = t_grid[yg < eol_true]
        out["knee"] = int(below[0]) if len(below) else np.nan
        out["_knee_curve"] = (t_grid, yg)
        out["_knee_point"] = p[3]
    except Exception:
        out["knee"] = np.nan
    return out


def main():
    cap = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "capacity_clean.csv"))
    cap = cap[cap.group != "4C_mixed"]

    rows, case = [], None
    for bid, s in cap.groupby("battery"):
        s = s.sort_values("cycle")
        cap0 = s[s.cycle <= 3].capacity_ah.mean()
        eol_true = s[s.capacity_ah < 0.8 * cap0].cycle.min()
        if np.isnan(eol_true):
            continue                       # 实验内未达 EOL, 无法验证
        for frac in (0.5, 0.7):
            cut = int(eol_true * frac)
            part = s[s.cycle <= cut]
            if len(part) < 15:
                continue
            r = fit_and_extrapolate(part.cycle.values, part.capacity_ah.values, cap0)
            row = dict(battery=bid, group=s.group.iloc[0], frac=frac, eol_true=int(eol_true),
                       eol_double_exp=r.get("double_exp"), eol_knee=r.get("knee"))
            row["err_knee"] = None if pd.isna(row["eol_knee"]) else row["eol_knee"] - eol_true
            row["err_de"] = None if pd.isna(row["eol_double_exp"]) else row["eol_double_exp"] - eol_true
            rows.append(row)
            if bid == "B0005" and frac == 0.5:
                case = (s, part, r, cap0, eol_true)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "v09_knee_rul.csv"), index=False, encoding="utf-8-sig")

    print("EOL 外推误差(预测-真值, 循环):")
    for frac in (0.5, 0.7):
        sub = res[res.frac == frac]
        print(f"  用前 {frac:.0%} 寿命: 双指数 MAE={sub.err_de.abs().mean():.0f} 循环 "
              f"(成功率 {sub.err_de.notna().mean():.0%}) | "
              f"拐点模型 MAE={sub.err_knee.abs().mean():.0f} 循环 "
              f"(成功率 {sub.err_knee.notna().mean():.0%}) | n={len(sub)}")

    # ---- 图: B0005 案例 + 误差对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    s, part, r, cap0, eol_true = case
    axes[0].plot(s.cycle, s.capacity_ah, "k-", lw=1.8, label="实测(全生命周期)")
    axes[0].plot(part.cycle, part.capacity_ah, "o", ms=3, color="#1f77b4", label="前半生(可见数据)")
    if "_de_curve" in r:
        tg, yg = r["_de_curve"]
        axes[0].plot(tg, yg, "--", color="#d62728", lw=1.6, label=f"双指数外推(EOL≈{r['double_exp']})")
    if "_knee_curve" in r:
        tg, yg = r["_knee_curve"]
        axes[0].plot(tg, yg, ":", color="#2ca02c", lw=2, label=f"拐点模型外推(EOL≈{r['knee']})")
    axes[0].axhline(0.8 * cap0, color="gray", ls="--", lw=1)
    axes[0].axvline(eol_true, color="k", ls="-.", lw=1)
    axes[0].text(eol_true + 3, 1.05, f"真实 EOL={eol_true}", fontsize=9)
    axes[0].set_xlim(0, 260); axes[0].set_ylim(0.9, 2.0)
    axes[0].set_xlabel("循环数"); axes[0].set_ylabel("容量 (Ah)")
    axes[0].set_title(f"案例 B0005:只用前半生拟合,外推寿命终点")
    axes[0].legend(fontsize=8.5); axes[0].grid(alpha=0.3)

    sub = res[res.frac == 0.5]
    e1 = sub.err_de.abs().mean(); e2 = sub.err_knee.abs().mean()
    bars = axes[1].bar(["双指数模型", "拐点分段模型", "ML 快照估计\n(V0.0.7, 供对照)"],
                       [e1, e2, 49.2], color=["#d62728", "#2ca02c", "#7f7f7f"], alpha=0.85, width=0.5)
    for b, v in zip(bars, [e1, e2, 49.2]):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f} 循环", ha="center", fontsize=11)
    axes[1].set_ylabel("RUL/EOL 预测误差 MAE(循环)")
    axes[1].set_title("参数化轨迹外推 vs ML 快照估计(越小越好)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("V0.0.9:RUL 的物理-ML 对决(用前 50% 寿命外推 EOL)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v09_knee_rul.png"), dpi=160)
    print("图已保存: figures/v09_knee_rul.png")


if __name__ == "__main__":
    main()
