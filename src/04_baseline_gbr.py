# -*- coding: utf-8 -*-
"""
04_baseline_gbr.py  基线模型 v1:分位数梯度提升预测放电容量

模型: sklearn HistGradientBoostingRegressor(loss='quantile'), q ∈ {0.05, 0.5, 0.9}
  - 原生支持 NaN 特征(低温组 IC 缺失无需插补)
  - 一次训练同时得到点预测 + 90% 预测区间

验证:
  A) 同工况留一电芯(LOBO): 24C_2A 组 4 只电芯轮流做测试
  B) 跨工况泛化: 训练=24C_2A 全部 → 测试=24C_4A / 43C_4A / 4C_2A

输出:
  results/v1_metrics.csv
  figures/v1_lobo_pred.png, figures/v1_cross_condition.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

FEATURES = ["cycle", "ambient_temp", "Re", "Rct",
            "dis_duration_s", "t_drop_36_30_s", "dis_temp_mean", "dis_current_mean",
            "ic_pos_max", "ic_height_max", "ic_area", "ic_mean", "ic_std",
            "cc_capacity_ah", "cc_duration_s", "cv_duration_s", "charge_temp_mean"]
TARGET = "capacity_ah"
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=10, l2_regularization=1.0, random_state=42)


def fit_quantile_models(X, y):
    return {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **PARAMS).fit(X, y)
            for q in (0.05, 0.5, 0.9)}


def evaluate(y_true, y_pred, y_lo, y_hi):
    picp = np.mean((y_true >= y_lo) & (y_true <= y_hi))
    return dict(MAE=mean_absolute_error(y_true, y_pred),
                RMSE=root_mean_squared_error(y_true, y_pred),
                R2=r2_score(y_true, y_pred),
                PICP90=picp,
                width90=np.mean(y_hi - y_lo),
                n=len(y_true))


def main():
    df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "feature_table.csv"))
    df = df[df.group != "4C_mixed"].reset_index(drop=True)

    metrics = []

    # ---- A) 同工况留一电芯 ----
    train_group = df[df.group == "24C_2A"]
    lobo_preds = {}
    for bid in sorted(train_group.battery.unique()):
        tr = train_group[train_group.battery != bid]
        te = train_group[train_group.battery == bid]
        models = fit_quantile_models(tr[FEATURES], tr[TARGET])
        p50 = models[0.5].predict(te[FEATURES])
        p05 = models[0.05].predict(te[FEATURES])
        p90 = models[0.9].predict(te[FEATURES])
        m = evaluate(te[TARGET].values, p50, p05, p90)
        m.update(split="LOBO_24C_2A", test=bid)
        metrics.append(m)
        lobo_preds[bid] = (te.cycle.values, te[TARGET].values, p50, p05, p90)
        print(f"LOBO {bid}: MAE={m['MAE']:.4f} Ah, R²={m['R2']:.3f}, PICP90={m['PICP90']:.0%}")

    # ---- B) 跨工况泛化 ----
    Xtr, ytr = train_group[FEATURES], train_group[TARGET]
    cross_preds = {}
    for g in ["24C_4A", "43C_4A", "4C_2A"]:
        te = df[df.group == g]
        models = fit_quantile_models(Xtr, ytr)
        p50 = models[0.5].predict(te[FEATURES])
        p05 = models[0.05].predict(te[FEATURES])
        p90 = models[0.9].predict(te[FEATURES])
        m = evaluate(te[TARGET].values, p50, p05, p90)
        m.update(split="cross_condition", test=g)
        metrics.append(m)
        cross_preds[g] = (te, p50, p05, p90)
        print(f"跨工况 {g}: MAE={m['MAE']:.4f} Ah, R²={m['R2']:.3f}, PICP90={m['PICP90']:.0%}")

    pd.DataFrame(metrics).to_csv(os.path.join(RES, "v1_metrics.csv"),
                                 index=False, encoding="utf-8-sig")

    # ---- 图 1: LOBO 逐电芯预测 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (bid, (cyc, y, p50, p05, p90)) in zip(axes.ravel(), lobo_preds.items()):
        ax.plot(cyc, y, "k-", lw=1.6, label="实测容量")
        ax.plot(cyc, p50, color="#d62728", lw=1.4, label="预测(中位数)")
        ax.fill_between(cyc, p05, p90, color="#d62728", alpha=0.2, label="90% 区间")
        ax.set_title(f"{bid}(留一电芯外推)", fontsize=11)
        ax.set_xlabel("循环数"); ax.set_ylabel("容量 (Ah)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("基线 v1:同工况留一电芯验证 — 分位数梯度提升(仅用前序工况内电芯训练)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v1_lobo_pred.png"), dpi=160)

    # ---- 图 2: 跨工况泛化 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = {"24C_4A": "24 ℃/4A(倍率泛化)", "43C_4A": "43 ℃/4A(温度+倍率)", "4C_2A": "4 ℃/2A(温度泛化)"}
    for ax, (g, (te, p50, p05, p90)) in zip(axes, cross_preds.items()):
        te = te.copy(); te["p50"], te["p05"], te["p90"] = p50, p05, p90
        for bid, s in te.groupby("battery"):
            ax.plot(s.cycle, s.capacity_ah, color="gray", alpha=0.45, lw=0.9)
        gm = te.groupby("cycle").agg(actual=("capacity_ah", "mean"), p50=("p50", "mean"),
                                     p05=("p05", "mean"), p90=("p90", "mean"))
        ax.plot(gm.index, gm.actual, "k-", lw=2, label="实测(组均值)")
        ax.plot(gm.index, gm.p50, color="#d62728", lw=1.8, label="预测(组均值)")
        ax.fill_between(gm.index, gm.p05, gm.p90, color="#d62728", alpha=0.2, label="90% 区间")
        m = [x for x in metrics if x.get("test") == g][0]
        ax.set_title(f"训练=24 ℃/2A → 测试 {titles[g]}\nMAE={m['MAE']:.3f} Ah, R²={m['R2']:.2f}, PICP90={m['PICP90']:.0%}", fontsize=10)
        ax.set_xlabel("循环数"); ax.set_ylabel("容量 (Ah)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("基线 v1:跨工况泛化 — 训练集只含 24 ℃/2A,模型未见过的工况表现如何?", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v1_cross_condition.png"), dpi=160)
    print("\n图已保存: figures/v1_lobo_pred.png, figures/v1_cross_condition.png")

    # ---- 特征重要性(排列重要性太贵,用同工况全量模型的分裂增益近似:此处用单电芯相关) ----
    corr = train_group[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
    print("\n与容量的 Pearson 相关(24C_2A 组,前 8):")
    print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
