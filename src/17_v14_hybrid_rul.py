# -*- coding: utf-8 -*-
"""
17_v14_hybrid_rul.py  V1.4:RUL 混合模型实证 —— 「轨迹给趋势 + ML 给修正」不只停在口头

V0.0.9 的结论: 拐点模型长程趋势稳但早期外推差(前50%寿命 MAE 82 循环);
V0.0.4 的结论: GBR-window 短程外推准(同工况 MAE 0.012 Ah)。
混合策略: 拐点模型外推整条轨迹 → GBR-window 学习「真实轨迹 - 拐点轨迹」的残差 → 合成校正轨迹 → 读 EOL

协议: 用前 50% 寿命数据; n=7 只到达 EOL 的电芯
产出: results/v14_hybrid_rul.csv, figures/v14_hybrid_rul.png
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, "..", "figures"); os.makedirs(FIG, exist_ok=True)
W = 10


def knee_model(t, p0, p1, p2, t_knee):
    return p0 + p1 * t + p2 * np.maximum(0, t - t_knee)


def fit_knee(t, y):
    p, _ = curve_fit(knee_model, t, y, p0=[y[0], -1e-3, -5e-3, t[len(t) // 2]],
                     bounds=([0, -0.1, -0.1, 1], [3, 0, 0, 600]), maxfev=20000)
    return p


def window_xy(cap_series, target_residual=None):
    """滑窗: 近 W 循环容量 → 下一循环(容量或残差)"""
    X, y = [], []
    for i in range(W, len(cap_series)):
        X.append(cap_series[i - W:i])
        y.append(target_residual[i] if target_residual is not None else cap_series[i])
    return np.array(X), np.array(y)


def main():
    cap = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "capacity_clean.csv"))
    cap = cap[cap.group != "4C_mixed"]

    rows, case = [], None
    for bid, s in cap.groupby("battery"):
        # 清洗造成循环号空洞 → 重排到连续网格并线性插值
        s = s.sort_values("cycle")
        full_grid = np.arange(1, int(s.cycle.max()) + 1)
        cap_interp = np.interp(full_grid, s.cycle.values, s.capacity_ah.values)
        s = pd.DataFrame({"cycle": full_grid, "capacity_ah": cap_interp, "group": s.group.iloc[0]})
        s = s.reset_index(drop=True)
        cap0 = s[s.cycle <= 3].capacity_ah.mean()
        eol_true = s[s.capacity_ah < 0.8 * cap0].cycle.min()
        if np.isnan(eol_true) or eol_true < 40:
            continue
        cut = int(eol_true * 0.5)
        part = s[s.cycle <= cut]
        t_all = s.cycle.values.astype(float)
        y_all = s.capacity_ah.values

        # 1) 拐点模型: 前半生拟合 → 全轨迹外推
        p = fit_knee(part.cycle.values.astype(float), part.capacity_ah.values)
        traj_knee = knee_model(t_all, *p)
        below = t_all[traj_knee < 0.8 * cap0]
        eol_knee = float(below[0]) if len(below) else np.nan

        # 2) 混合: 前半生残差训 GBR-window, 递归外推后半生轨迹
        resid_train = part.capacity_ah.values - knee_model(part.cycle.values.astype(float), *p)
        Xr, yr = window_xy(part.capacity_ah.values, resid_train)
        if len(Xr) < 20:
            continue
        gbr = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                            max_leaf_nodes=15, random_state=42).fit(Xr, yr)
        hist_win = list(part.capacity_ah.values[-W:])
        traj_hyb = list(part.capacity_ah.values)
        for cyc in range(cut + 1, int(t_all[-1]) + 1):
            base = knee_model(np.array([cyc]), *p)[0]
            resid_hat = gbr.predict(np.array([hist_win[-W:]]))[0]
            nxt = base + resid_hat
            traj_hyb.append(nxt); hist_win.append(nxt)
        traj_hyb = np.array(traj_hyb)
        below_h = t_all[traj_hyb < 0.8 * cap0]
        eol_hyb = float(below_h[0]) if len(below_h) else np.nan

        rows.append(dict(battery=bid, group=s.group.iloc[0], eol_true=int(eol_true),
                         eol_knee=eol_knee, eol_hybrid=eol_hyb,
                         err_knee=None if np.isnan(eol_knee) else eol_knee - eol_true,
                         err_hybrid=None if np.isnan(eol_hyb) else eol_hyb - eol_true))
        print(f"{bid}: 真EOL={int(eol_true)} 拐点={eol_knee:.0f} 混合={eol_hyb:.0f}")
        if bid == "B0005":
            case = (t_all, y_all, traj_knee, traj_hyb, cut, eol_true, cap0)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "v14_hybrid_rul.csv"), index=False, encoding="utf-8-sig")
    print(f"\nEOL 外推 MAE(前 50% 寿命): 拐点 {res.err_knee.abs().mean():.0f} 循环 → "
          f"混合 {res.err_hybrid.abs().mean():.0f} 循环 (n={len(res)})")

    # ---- 图: B0005 轨迹对比 + 误差对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    t_all, y_all, traj_knee, traj_hyb, cut, eol_true, cap0 = case
    axes[0].plot(t_all, y_all, "k-", lw=1.8, label="实测")
    axes[0].plot(t_all, traj_knee, "--", color="#2ca02c", lw=1.6, label="拐点模型外推")
    axes[0].plot(t_all, traj_hyb, "-.", color="#d62728", lw=1.6, label="混合(拐点+ML 残差)")
    axes[0].axvline(cut, color="gray", ls=":", lw=1.2)
    axes[0].text(cut + 3, 1.9, "← 仅用左侧数据", fontsize=9, color="gray")
    axes[0].axhline(0.8 * cap0, color="gray", ls="--", lw=1)
    axes[0].axvline(eol_true, color="k", ls="-.", lw=1)
    axes[0].set_xlim(0, 180); axes[0].set_ylim(1.0, 2.0)
    axes[0].set_xlabel("循环数"); axes[0].set_ylabel("容量 (Ah)")
    axes[0].set_title(f"案例 B0005(真实 EOL={int(eol_true)})")
    axes[0].legend(fontsize=8.5); axes[0].grid(alpha=0.3)

    e1, e2 = res.err_knee.abs().mean(), res.err_hybrid.abs().mean()
    bars = axes[1].bar(["拐点模型", "混合模型"], [e1, e2], color=["#2ca02c", "#d62728"], alpha=0.85, width=0.45)
    for b, v in zip(bars, [e1, e2]):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f} 循环", ha="center", fontsize=11)
    axes[1].set_ylabel("EOL 外推误差 MAE(循环)")
    axes[1].set_title("前 50% 寿命外推 EOL:趋势+残差分工的收益")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("V1.4 RUL 混合模型:拐点模型给长程趋势,梯度提升修正残差", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v14_hybrid_rul.png"), dpi=160)
    print("图已保存: figures/v14_hybrid_rul.png")


if __name__ == "__main__":
    main()
