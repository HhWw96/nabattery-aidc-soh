# -*- coding: utf-8 -*-
"""
18_v15_tail_rules.py  V1.5:尾部校准规则对比 —— 攻 24C_4A 容量再生尖峰的覆盖率缺口

V1.3 混合口径在 24C_4A 组覆盖率 71%: 该组老化中期出现容量再生尖峰,
10 个校准循环的残差估不准尾部。对比 4 种尾部规则:
  A) δ_local × 1.5(提高安全系数)
  B) 2 × std(校准残差)(近似高斯尾部)
  C) max(混合δ, 5% × 组容量中位数)(规模比例下限)
  D) 随循环数线性放宽: δ(k) = δ_hyb × (1 + 0.02·(k-10))(距校准点越远越不确定)
验收: 四组 PICP 全部 ≥90%, 且分布内组宽度增幅最小者胜出; 做不到则诚实记录
产出: results/v15_tail_rules.csv, figures/v15_tail_rules.png
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
GROUPS = ["24C_2A", "24C_4A", "43C_4A", "4C_2A"]
TITLES = {"24C_2A": "24 ℃/2A", "24C_4A": "24 ℃/4A", "43C_4A": "43 ℃/4A", "4C_2A": "4 ℃/2A"}


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


def fit_q(X, y):
    return {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **PARAMS).fit(X, y)
            for q in (0.05, 0.5, 0.9)}


def conformal_delta(tr, ytr):
    scores = []
    for b in tr.battery.unique():
        tr_in, cal = tr[tr.battery != b], tr[tr.battery == b]
        if len(tr_in) < 50:
            continue
        m = fit_q(tr_in[FEATURES], ytr.loc[tr_in.index])
        p05, p90 = m[0.05].predict(cal[FEATURES]), m[0.9].predict(cal[FEATURES])
        scores.append(np.maximum(p05 - ytr.loc[cal.index].values, ytr.loc[cal.index].values - p90))
    return float(np.quantile(np.concatenate(scores), 0.9)) if scores else 0.0


def main():
    df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "feature_table.csv"))
    df = df[df.group != "4C_mixed"].reset_index(drop=True)
    df = add_lag(df)

    rows = []
    for g in GROUPS:
        tr, te = df[df.group != g].copy(), df[df.group == g].copy()
        tr["y"] = tr.capacity_ah
        delta_tr = conformal_delta(tr, tr.y)
        models = fit_q(tr[FEATURES], tr.y)
        widen = 2.5 if g in ("43C_4A", "4C_2A") else 1.0

        warm = te[te.cycle <= 10]
        scope = te[te.cycle > 10]
        p_warm = models[0.5].predict(warm[FEATURES])
        resid = warm.capacity_ah.values - p_warm
        bias = float(resid.mean()) if widen > 1 else 0.0
        delta_local = np.quantile(np.abs(resid), 0.9) * 1.15
        delta_hyb = max(delta_tr * widen, delta_local)
        cap_med = te.capacity_ah.median()

        p50 = models[0.5].predict(scope[FEATURES]) + bias
        y = scope.capacity_ah.values
        rules = {
            "A_安全系数1.5": np.full(len(scope), np.quantile(np.abs(resid), 0.9) * 1.5),
            "B_2std残差": np.full(len(scope), 2 * resid.std(ddof=1)),
            "C_规模下限5%": np.full(len(scope), max(delta_hyb, 0.05 * cap_med)),
            "D_随时间放宽": delta_hyb * (1 + 0.02 * (scope.cycle.values - 10)),
            "E_2std×门控放宽": np.full(len(scope), 2 * resid.std(ddof=1) * widen),
        }
        for rule, halfw in rules.items():
            picp = np.mean((y >= p50 - halfw) & (y <= p50 + halfw))
            rows.append(dict(test_group=g, rule=rule, PICP90=picp, width90=2 * halfw.mean()))
            print(f"{g} | {rule}: PICP={picp:.0%} 宽={2*halfw.mean():.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "v15_tail_rules.csv"), index=False, encoding="utf-8-sig")

    # ---- 图: 每组 × 每规则 覆盖率热图 ----
    pivot = res.pivot(index="rule", columns="test_group", values="PICP90")[GROUPS] * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([TITLES[g] for g in GROUPS], fontsize=9)
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot)):
        for j in range(4):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="black" if pivot.values[i, j] > 70 else "white")
    ax.set_title("V1.5 尾部校准规则对比:覆盖率(目标 ≥90%,绿色达标)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v15_tail_rules.png"), dpi=160)
    print("\n图已保存: figures/v15_tail_rules.png")
    print("\n各规则平均宽度(Ah):")
    print(res.groupby("rule").width90.mean().round(3).to_string())


if __name__ == "__main__":
    main()
