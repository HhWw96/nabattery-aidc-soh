# -*- coding: utf-8 -*-
"""
20_v17_eval_protocol.py  V1.7:评估协议对比 —— 量化「随机划分」的精度虚高

动机: 电池退化是时序数据。随机 70/30 划分让测试循环与训练循环交错,
      模型见过「未来」的衰减形态 → 精度系统性虚高(数据泄漏)。
实验: 同一特征集、同一模型(GBR), 三种协议:
  A) 随机 70/30(常见但错误)
  B) 时间外推(前 70% 循环训练 → 后 30% 测试)
  C) 留一电芯(整只电芯外推, 我们的协议)
产出: results/v17_protocol.csv, figures/v17_protocol.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

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
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=10, l2_regularization=1.0, random_state=42)


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


def fit_pred(Xtr, ytr, Xte):
    return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **PARAMS).fit(Xtr, ytr).predict(Xte)


def main():
    df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "feature_table.csv"))
    df = df[df.group != "4C_mixed"].reset_index(drop=True)
    df = add_lag(df)
    rng = np.random.default_rng(42)

    rows = []
    # A) 随机 70/30(带滞后特征 = 泄漏最严重形态; 不带滞后 = 仍泄漏工况分布)
    for use_lag, tag in [(True, "含滞后特征"), (False, "不含滞后")]:
        feats = BASE_FEATS + (LAG_FEATS if use_lag else [])
        idx = rng.permutation(len(df))
        ntr = int(len(df) * 0.7)
        tr_i, te_i = idx[:ntr], idx[ntr:]
        p = fit_pred(df.iloc[tr_i][feats], df.iloc[tr_i].capacity_ah, df.iloc[te_i][feats])
        rows.append(dict(protocol=f"A 随机70/30({tag})", MAE=mean_absolute_error(df.iloc[te_i].capacity_ah, p),
                         R2=r2_score(df.iloc[te_i].capacity_ah, p)))
        print(f"A 随机70/30({tag}): MAE={rows[-1]['MAE']:.4f}, R²={rows[-1]['R2']:.4f}")

    # B) 时间外推: 逐电芯前 70% 训练, 后 30% 测试
    feats = BASE_FEATS + LAG_FEATS
    tr = df[df.cycle <= df.groupby("battery").cycle.transform("max") * 0.7]
    te = df[df.cycle > df.groupby("battery").cycle.transform("max") * 0.7]
    p = fit_pred(tr[feats], tr.capacity_ah, te[feats])
    rows.append(dict(protocol="B 时间外推(前70%→后30%)", MAE=mean_absolute_error(te.capacity_ah, p),
                     R2=r2_score(te.capacity_ah, p)))
    print(f"B 时间外推: MAE={rows[-1]['MAE']:.4f}, R²={rows[-1]['R2']:.4f}")

    # C) 留一电芯(我们的主协议, 24C_2A 组内示范)
    sub = df[df.group == "24C_2A"]
    maes = []
    for bid in sub.battery.unique():
        tr_b, te_b = sub[sub.battery != bid], sub[sub.battery == bid]
        p = fit_pred(tr_b[feats], tr_b.capacity_ah, te_b[feats])
        maes.append(mean_absolute_error(te_b.capacity_ah, p))
    rows.append(dict(protocol="C 留一电芯(24C_2A)", MAE=np.mean(maes), R2=np.nan))
    print(f"C 留一电芯: MAE={np.mean(maes):.4f}(逐芯 {[round(m,3) for m in maes]})")

    # D) 对手协议复刻: 逐电芯 Ridge + 简单统计特征, 随机 70/30 vs 时间 70/30
    from sklearn.linear_model import Ridge
    SIMPLE = ["cycle", "dis_duration_s", "dis_temp_mean", "dis_current_mean",
              "ambient_temp", "t_drop_36_30_s"]
    for split, tag in [("random", "随机"), ("temporal", "时间")]:
        maes_d, r2_d = [], []
        for bid, s in df.groupby("battery"):
            s = s.sort_values("cycle")
            X = s[SIMPLE].fillna(s[SIMPLE].mean()).values
            y = s.capacity_ah.values
            if split == "random":
                idx = rng.permutation(len(s)); ntr = int(len(s) * 0.7)
                tri, tei = idx[:ntr], idx[ntr:]
            else:
                ntr = int(len(s) * 0.7)
                tri, tei = np.arange(ntr), np.arange(ntr, len(s))
            r = Ridge(alpha=1.0).fit(X[tri], y[tri])
            p = r.predict(X[tei])
            maes_d.append(mean_absolute_error(y[tei], p)); r2_d.append(r2_score(y[tei], p))
        rows.append(dict(protocol=f"D 逐芯Ridge-{tag}70/30", MAE=np.mean(maes_d), R2=np.mean(r2_d)))
        print(f"D 逐芯 Ridge {tag}: MAE={np.mean(maes_d):.4f}, R²={np.mean(r2_d):.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "v17_protocol.csv"), index=False, encoding="utf-8-sig")

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(11, 4.6))
    labels = [r["protocol"] for r in rows]
    vals = [r["MAE"] for r in rows]
    colors = ["#F05548", "#F5A623", "#1f77b4", "#22D3A5", "#8A84C7", "#C9A869"][:len(rows)]
    bars = ax.bar(range(len(rows)), vals, color=colors, alpha=0.88, width=0.55)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([l.replace("(", "\n(") for l in labels], fontsize=9)
    ax.set_ylabel("容量预测 MAE (Ah)")
    ax.set_title("V1.7 评估协议对比:同一数据,切法决定分数——随机划分 < 时间外推 < 整芯外推 < 留一工况\n"
                 "(D 组为对手协议复刻:逐芯 Ridge + 简单统计特征)", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v17_protocol.png"), dpi=160)
    print("\n图已保存: figures/v17_protocol.png")


if __name__ == "__main__":
    main()
