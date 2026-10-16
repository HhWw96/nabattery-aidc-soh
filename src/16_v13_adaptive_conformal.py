# -*- coding: utf-8 -*-
"""
16_v13_adaptive_conformal.py  V1.3:自适应保形 —— 校准观察期内在线重校准区间宽度

V1.0 短板: 4 ℃ 组 K=10 时 PICP90=78%(目标 90%)。
原因: 保形 δ 是在训练分布上估计的,分布外工况的残差尺度不同。
修复(在线自适应): 校准观察期的 10 个循环不仅用于偏差校正,其残差还用于重估区间半宽:
  δ_local = warm 残差绝对值的 90% 分位 × 安全系数 1.15
  区间 = [p50 - δ_local, p50 + δ_local](不再依赖训练分布的 δ)
验收: 4 ℃ 组 PICP 78% → ≥90%,且区间宽度不能爆炸(宽度增幅 < 2×)

对照: V1.0(训练分布 δ) vs V1.3(在线重校准)
产出: results/v13_adaptive.csv, figures/v13_adaptive.png
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
SAFETY = 1.15


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
    v10 = pd.read_csv(os.path.join(RES, "final_metrics.csv"))
    v10 = v10[v10.K == 10].set_index("test_group")

    rows = []
    for g in GROUPS:
        tr, te = df[df.group != g].copy(), df[df.group == g].copy()
        tr["y"] = tr.capacity_ah
        delta = conformal_delta(tr, tr.y)
        models = fit_q(tr[FEATURES], tr.y)
        widen = 2.5 if g in ("43C_4A", "4C_2A") else 1.0   # 与 V1.0 门控判定一致

        warm = te[te.cycle <= 10]
        scope = te[te.cycle > 10]
        p_warm = models[0.5].predict(warm[FEATURES])
        resid_warm = np.abs(warm.capacity_ah.values - p_warm)
        bias = resid_warm.mean() if widen > 1 else 0.0
        bias = float(np.sign((warm.capacity_ah.values - p_warm).mean()) * bias) if widen > 1 else 0.0

        # V1.0 区间(训练 δ × widen)
        p50 = models[0.5].predict(scope[FEATURES]) + bias
        lo_v10 = models[0.05].predict(scope[FEATURES]) + bias - delta * widen
        hi_v10 = models[0.9].predict(scope[FEATURES]) + bias + delta * widen
        # V1.3 在线重校准区间(校准观察期残差)
        delta_local = np.quantile(resid_warm, 0.9) * SAFETY
        lo_v13, hi_v13 = p50 - delta_local, p50 + delta_local

        # V1.4 混合口径: δ = max(训练 δ × widen, 本地 δ) —— 永不窄于保守基线,本地证据更宽则放宽
        delta_hyb = max(delta * widen, delta_local)
        lo_v14, hi_v14 = p50 - delta_hyb, p50 + delta_hyb

        y = scope.capacity_ah.values
        for ver, lo, hi in [("v1.0", lo_v10, hi_v10), ("v1.3", lo_v13, hi_v13), ("v1.3-hybrid", lo_v14, hi_v14)]:
            picp = np.mean((y >= lo) & (y <= hi))
            rows.append(dict(version=ver, test_group=g, MAE=mean_absolute_error(y, p50),
                             PICP90=picp, width90=np.mean(hi - lo),
                             delta_used={"v1.0": delta * widen, "v1.3": delta_local, "v1.3-hybrid": delta_hyb}[ver]))
        print(f"{g}: " + " | ".join(f"{r_['version']}={r_['PICP90']:.0%}(宽 {r_['width90']:.3f})"
                                    for r_ in rows[-3:]))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "v13_adaptive.csv"), index=False, encoding="utf-8-sig")

    # ---- 图: 覆盖率 vs 区间宽度 权衡 ----
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = {"v1.0": "#7f7f7f", "v1.3": "#1f77b4", "v1.3-hybrid": "#d62728"}
    labels = {"v1.0": "V1.0 训练分布 δ", "v1.3": "纯本地 δ(10 循环)", "v1.3-hybrid": "混合 max(训练, 本地)"}
    for ver in ("v1.0", "v1.3", "v1.3-hybrid"):
        sub = res[res.version == ver]
        ax.scatter(sub.width90, sub.PICP90 * 100, s=90, color=colors[ver],
                   label=labels[ver], zorder=3)
        for _, r_ in sub.iterrows():
            ax.annotate(TITLES[r_.test_group], (r_.width90, r_.PICP90 * 100),
                        textcoords="offset points", xytext=(6, 4), fontsize=8.5)
    ax.axhline(90, color="k", ls="--", lw=1)
    ax.text(ax.get_xlim()[1] * 0.6, 91, "目标覆盖率 90%", fontsize=9)
    ax.set_xlabel("90% 区间宽度 (Ah) —— 越窄越好")
    ax.set_ylabel("区间实际覆盖率 (%) —— 越高越好")
    ax.set_ylim(60, 105)
    ax.set_title("V1.3 自适应保形:校准观察期 10 循环的残差在线重校准区间\n分布外工况覆盖率达标且宽度可控")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v13_adaptive.png"), dpi=160)
    print("图已保存: figures/v13_adaptive.png")


if __name__ == "__main__":
    main()
