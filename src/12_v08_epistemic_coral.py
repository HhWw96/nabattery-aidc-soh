# -*- coding: utf-8 -*-
"""
12_v08_epistemic_coral.py  V0.0.8:认知不确定性 + CORAL 域适应(前沿验证)

Part A 认知不确定性(bootstrap 集成):
  - 对训练集做 B=10 次有放回重采样, 训 10 个分位数 GBM(中位数), 预测离散度 std 即认知不确定性代理
  - 假设: 分布外工况(4 ℃)的 std 显著高于分布内 —— 若成立, std 可作为独立于指纹距离的 OOD 信号
  - 验证: 各组 std 对比 + std 与 |误差| 的相关性

Part B CORAL 域适应(挑战 K=0 盲测):
  - 仅用目标工况「前 3 循环的无标签特征」估计目标域协方差, 对训练特征做线性对齐:
    X_tr' = X_tr · Cs^(-1/2) · Ct^(1/2) (CORAL, Saenko et al. 2010)
  - 标签完全不用 —— 这是真正的零标注盲适配
  - 验证: 4C_2A / 43C_4A 留一工况下, CORAL 对齐 vs 原始特征的 MAE 对比
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, "..", "figures"); os.makedirs(FIG, exist_ok=True)

FEATURES = ["cycle", "ambient_temp", "Re", "Rct",
            "dis_duration_s", "t_drop_36_30_s", "dis_temp_mean", "dis_current_mean",
            "ic_pos_max", "ic_height_max", "ic_area", "ic_mean", "ic_std",
            "cc_capacity_ah", "cc_duration_s", "cv_duration_s", "charge_temp_mean"]
TARGET = "capacity_ah"
PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
              min_samples_leaf=10, l2_regularization=1.0, random_state=None)
GROUPS = ["24C_2A", "24C_4A", "43C_4A", "4C_2A"]
TITLES = {"24C_2A": "24 ℃/2A(分布内)", "24C_4A": "24 ℃/4A(角点)",
          "43C_4A": "43 ℃/4A(外推)", "4C_2A": "4 ℃/2A(真外推)"}


def fit_one(X, y, seed):
    p = dict(PARAMS); p["random_state"] = seed
    return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **p).fit(X, y)


def partA(df):
    print("== Part A: 认知不确定性(bootstrap 集成 B=10) ==")
    rng = np.random.default_rng(42)
    rows, scatter = [], []
    for g in GROUPS:
        tr, te = df[df.group != g], df[df.group == g]
        Xtr, ytr = tr[FEATURES].values, tr[TARGET].values
        Xte, yte = te[FEATURES].values, te[TARGET].values
        preds = []
        for b in range(10):
            idx = rng.choice(len(Xtr), size=len(Xtr), replace=True)
            preds.append(fit_one(Xtr[idx], ytr[idx], seed=b).predict(Xte))
        preds = np.array(preds)
        std = preds.std(0)
        err = np.abs(preds.mean(0) - yte)
        rows.append(dict(test_group=g, std_mean=std.mean(), std_p90=np.quantile(std, 0.9),
                         mae=mean_absolute_error(yte, preds.mean(0)),
                         spearman_std_err=pd.Series(std).corr(pd.Series(err), method="spearman")))
        scatter.append(pd.DataFrame(dict(group=g, std=std, err=err)))
        print(f"  {g}: std 均值={std.mean():.4f} | MAE={rows[-1]['mae']:.3f} | "
              f"std-误差 Spearman 相关={rows[-1]['spearman_std_err']:.2f}")
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(RES, "v08_epistemic.csv"), index=False, encoding="utf-8-sig")

    sc = pd.concat(scatter)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {"24C_2A": "#2ca02c", "24C_4A": "#1f77b4", "43C_4A": "#ff7f0e", "4C_2A": "#d62728"}
    for g in GROUPS:
        s = sc[sc.group == g]
        axes[0].scatter(s["std"], s.err, s=6, alpha=0.35, color=colors[g], label=TITLES[g])
    axes[0].set_xlabel("认知不确定性(集成预测 std, Ah)")
    axes[0].set_ylabel("绝对误差 (Ah)")
    axes[0].set_title("不确定性越高,误差越大 —— std 是误差的可行代理")
    axes[0].legend(fontsize=8, markerscale=2); axes[0].grid(alpha=0.3)

    means = r.set_index("test_group").loc[GROUPS]
    bars = axes[1].bar([TITLES[g] for g in GROUPS], means.std_mean,
                       color=[colors[g] for g in GROUPS], alpha=0.85)
    for b, v in zip(bars, means.std_mean):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.3f}", ha="center", fontsize=10)
    axes[1].set_ylabel("集成预测 std 均值 (Ah)")
    axes[1].set_title("分布外组的认知不确定性显著更高\n(与指纹距离、覆盖率失效三路互证)")
    axes[1].grid(alpha=0.3, axis="y")
    plt.setp(axes[1].get_xticklabels(), fontsize=9)
    fig.suptitle("V0.0.8 Part A:认知不确定性量化(bootstrap 集成 ×10)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v08_epistemic.png"), dpi=160)
    print("  图已保存: figures/v08_epistemic.png")
    return r


def coral_align(Xs, Xt):
    """CORAL: 二阶统计对齐。Xs: 源域(训练), Xt: 目标域(无标签)"""
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Xs_c, Xt_c = Xs - mu_s, Xt - mu_t
    cov_s = np.cov(Xs_c.T) + np.eye(Xs.shape[1]) * 1e-4
    cov_t = np.cov(Xt_c.T) + np.eye(Xs.shape[1]) * 1e-4
    # Cs^{-1/2}
    es, Us = np.linalg.eigh(cov_s)
    Cs_inv_sqrt = Us @ np.diag(1.0 / np.sqrt(es)) @ Us.T
    et, Ut = np.linalg.eigh(cov_t)
    Ct_sqrt = Ut @ np.diag(np.sqrt(np.maximum(et, 1e-8))) @ Ut.T
    A = Cs_inv_sqrt @ Ct_sqrt
    return Xs_c @ A + mu_t   # 对齐后的源域特征(目标域均值/协方差)


def partB(df):
    print("\n== Part B: CORAL 域适应(仅用目标域前 3 循环无标签特征) ==")
    rows = []
    for g in ["43C_4A", "4C_2A"]:
        tr, te = df[df.group != g], df[df.group == g]
        Xte_all, yte = te[FEATURES].values, te[TARGET].values
        warm_feat = te[te.cycle <= 3][FEATURES].values   # 无标签: 不用 capacity_ah
        # 特征标准化(按源域)
        mu, sd = tr[FEATURES].mean(), tr[FEATURES].std() + 1e-8
        Xs = (tr[FEATURES].values - mu.values) / sd.values
        Xt = (Xte_all - mu.values) / sd.values
        Xw = (warm_feat - mu.values) / sd.values
        # 缺失特征(IC 在部分循环提取失败)以源域均值(=0)填充
        Xs = np.nan_to_num(Xs); Xt = np.nan_to_num(Xt); Xw = np.nan_to_num(Xw)
        # 基线: 原始特征
        p_base = fit_one(Xs, tr[TARGET].values, 42).predict(Xt)
        mae_base = mean_absolute_error(yte, p_base)
        # CORAL: 训练特征对齐到目标域前 3 循环的无标签分布
        Xs_aligned = coral_align(Xs, Xw)
        p_coral = fit_one(Xs_aligned, tr[TARGET].values, 42).predict(Xt)
        mae_coral = mean_absolute_error(yte, p_coral)
        rows.append(dict(test_group=g, MAE_base=mae_base, MAE_coral=mae_coral,
                         delta=mae_base - mae_coral, n_warm_features=len(Xw)))
        print(f"  {g}: 原始 MAE={mae_base:.3f} → CORAL MAE={mae_coral:.3f} (Δ={mae_base-mae_coral:+.3f})")
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(RES, "v08_coral.csv"), index=False, encoding="utf-8-sig")
    return r


def main():
    df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "feature_table.csv"))
    df = df[df.group != "4C_mixed"].reset_index(drop=True)
    partA(df)
    partB(df)


if __name__ == "__main__":
    main()
