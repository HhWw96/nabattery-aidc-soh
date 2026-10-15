# -*- coding: utf-8 -*-
"""
05_v02_loco_conformal.py  V0.0.2:多工况留一工况(LOCO)验证 + 保形校准

针对 V0.0.1 的两个硬伤:
  1. 单工况训练无法跨工况 → 改为多工况混合训练,留一工况做测试(4 折)
     叙事: 24C_4A 是「角点补全」(4A 见过@43℃,24℃ 见过@2A);
           4C_2A 是「真外推」(训练集温度仅 24/43℃)——预期仍难,引出 V0.0.3
  2. 分位数区间覆盖率不足(32-57%) → split-conformal:
     训练集内做留一电芯(LOBO)得到样本外残差,求单边一致性得分
     s = max(p05 - y, y - p90) 的 90% 分位数 δ,测试区间扩为 [p05-δ, p90+δ]

输出:
  results/v02_metrics.csv (+ 与 v0.0.1 对比列)
  figures/v02_loco.png  4 折测试组预测图(含校准后区间)
  figures/v02_coverage.png  校准前后 PICP 对比
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
RES = os.path.join(HERE, "..", "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, "..", "figures"); os.makedirs(FIG, exist_ok=True)

FEATURES = ["cycle", "ambient_temp", "Re", "Rct",
            "dis_duration_s", "t_drop_36_30_s", "dis_temp_mean", "dis_current_mean",
            "ic_pos_max", "ic_height_max", "ic_area", "ic_mean", "ic_std",
            "cc_capacity_ah", "cc_duration_s", "cv_duration_s", "charge_temp_mean"]
TARGET = "capacity_ah"
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=10, l2_regularization=1.0, random_state=42)
GROUPS = ["24C_2A", "24C_4A", "43C_4A", "4C_2A"]
TITLES = {"24C_2A": "24 ℃/2A(常温基准)", "24C_4A": "24 ℃/4A(角点补全)",
          "43C_4A": "43 ℃/4A(高温)", "4C_2A": "4 ℃/2A(真外推:训练无低温)"}


def fit_q(X, y):
    return {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **PARAMS).fit(X, y)
            for q in (0.05, 0.5, 0.9)}


def conformal_delta(tr):
    """训练集内 LOBO 样本外残差 → 保形扩展量 δ(90% 目标覆盖)"""
    scores = []
    bats = sorted(tr.battery.unique())
    if len(bats) < 2:
        return 0.0
    for b in bats:
        tr_in = tr[tr.battery != b]; cal = tr[tr.battery == b]
        if len(tr_in) < 50:
            continue
        m = fit_q(tr_in[FEATURES], tr_in[TARGET])
        p05, p90 = m[0.05].predict(cal[FEATURES]), m[0.9].predict(cal[FEATURES])
        s = np.maximum(p05 - cal[TARGET].values, cal[TARGET].values - p90)
        scores.append(s)
    s = np.concatenate(scores)
    return float(np.quantile(s, 0.9))


def evaluate(y, p50, p05, p90):
    return dict(MAE=mean_absolute_error(y, p50),
                RMSE=root_mean_squared_error(y, p50),
                R2=r2_score(y, p50),
                PICP90=np.mean((y >= p05) & (y <= p90)),
                width90=np.mean(p90 - p05), n=len(y))


def main():
    df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "feature_table.csv"))
    df = df[df.group != "4C_mixed"].reset_index(drop=True)

    metrics, preds = [], {}
    for g in GROUPS:
        tr, te = df[df.group != g], df[df.group == g]
        delta = conformal_delta(tr)
        m = fit_q(tr[FEATURES], tr[TARGET])
        p50 = m[0.5].predict(te[FEATURES])
        p05 = m[0.05].predict(te[FEATURES]) - delta
        p90 = m[0.9].predict(te[FEATURES]) + delta
        raw = evaluate(te[TARGET].values, p50,
                       m[0.05].predict(te[FEATURES]), m[0.9].predict(te[FEATURES]))
        cal = evaluate(te[TARGET].values, p50, p05, p90)
        metrics.append(dict(version="v0.0.2", test_group=g, delta=delta,
                            **{f"raw_{k}": v for k, v in raw.items()},
                            **{f"cal_{k}": v for k, v in cal.items()}))
        preds[g] = (te, p50, p05, p90)
        print(f"LOCO 测试 {g}: MAE={cal['MAE']:.4f} Ah, R²={cal['R2']:.3f}, "
              f"PICP90 校准前 {raw['PICP90']:.0%} → 校准后 {cal['PICP90']:.0%} (δ={delta:.3f})")

    pd.DataFrame(metrics).to_csv(os.path.join(RES, "v02_metrics.csv"),
                                 index=False, encoding="utf-8-sig")

    # ---- 图 1: 4 折 LOCO ----
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    for ax, (g, (te, p50, p05, p90)) in zip(axes, preds.items()):
        te = te.copy(); te["p50"], te["p05"], te["p90"] = p50, p05, p90
        for bid, s in te.groupby("battery"):
            ax.plot(s.cycle, s.capacity_ah, color="gray", alpha=0.45, lw=0.9)
        gm = te.groupby("cycle").agg(actual=("capacity_ah", "mean"), p50=("p50", "mean"),
                                     p05=("p05", "mean"), p90=("p90", "mean"))
        ax.plot(gm.index, gm.actual, "k-", lw=2, label="实测(组均值)")
        ax.plot(gm.index, gm.p50, color="#d62728", lw=1.8, label="预测(组均值)")
        ax.fill_between(gm.index, gm.p05, gm.p90, color="#d62728", alpha=0.2, label="90% 区间(已校准)")
        mm = [x for x in metrics if x["test_group"] == g][0]
        ax.set_title(f"测试 {TITLES[g]}\nMAE={mm['cal_MAE']:.3f} Ah, R²={mm['cal_R2']:.2f}, PICP={mm['cal_PICP90']:.0%}", fontsize=9.5)
        ax.set_xlabel("循环数"); ax.set_ylabel("容量 (Ah)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("V0.0.2:多工况留一工况验证(训练=其余 3 个工况组,保形校准区间)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v02_loco.png"), dpi=160)

    # ---- 图 2: 校准前后覆盖率对比 ----
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(metrics))
    raw_p = [m["raw_PICP90"] for m in metrics]
    cal_p = [m["cal_PICP90"] for m in metrics]
    ax.bar(x - 0.18, raw_p, 0.36, label="校准前", color="#bdbdbd")
    ax.bar(x + 0.18, cal_p, 0.36, label="保形校准后", color="#d62728")
    ax.axhline(0.9, color="k", ls="--", lw=1, label="目标 90%")
    ax.set_xticks(x); ax.set_xticklabels([TITLES[m["test_group"]] for m in metrics], fontsize=9)
    ax.set_ylabel("90% 区间实际覆盖率(PICP)")
    ax.set_title("V0.0.2:保形校准把「名义 90% 区间」变成「真 90% 区间」")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v02_coverage.png"), dpi=160)
    print("\n图已保存: figures/v02_loco.png, figures/v02_coverage.png")


if __name__ == "__main__":
    main()
