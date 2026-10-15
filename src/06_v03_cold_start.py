# -*- coding: utf-8 -*-
"""
06_v03_cold_start.py  V0.0.3:冷启动适配 —— 新工况只给前 K 个循环,误差降多少?

场景: AIDC 现场接入一个新工况(如低温仓库)的电池,模型没见过这个工况。
       两种适配策略对比:
         A) 偏差校正: 用前 K 循环的实测-预测残差均值做平移校正(零重训,可在线)
         B) 混合重训: 把前 K 循环并入训练集重训(离线批量)
输出:
  results/v03_cold_start.csv
  figures/v03_cold_start.png   MAE vs K 曲线(两种策略 × 4 个测试工况)
  figures/v03_case_4C.png     4C_2A 组 K=10 的预测对比案例图
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

FEATURES = ["cycle", "ambient_temp", "Re", "Rct",
            "dis_duration_s", "t_drop_36_30_s", "dis_temp_mean", "dis_current_mean",
            "ic_pos_max", "ic_height_max", "ic_area", "ic_mean", "ic_std",
            "cc_capacity_ah", "cc_duration_s", "cv_duration_s", "charge_temp_mean"]
TARGET = "capacity_ah"
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=10, l2_regularization=1.0, random_state=42)
GROUPS = ["24C_2A", "24C_4A", "43C_4A", "4C_2A"]
TITLES = {"24C_2A": "24 ℃/2A", "24C_4A": "24 ℃/4A(角点)",
          "43C_4A": "43 ℃/4A", "4C_2A": "4 ℃/2A(真外推)"}
KS = [0, 5, 10, 20, 30]


def fit_median(X, y):
    return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **PARAMS).fit(X, y)


def main():
    df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "feature_table.csv"))
    df = df[df.group != "4C_mixed"].reset_index(drop=True)

    rows, case_store = [], None
    for g in GROUPS:
        te_all, tr_base = df[df.group == g], df[df.group != g]
        model0 = fit_median(tr_base[FEATURES], tr_base[TARGET])
        for K in KS:
            warm = te_all[te_all.cycle <= K]      # 前 K 循环(适配数据)
            test = te_all[te_all.cycle > K]       # 剩余循环(真测试)
            if len(test) < 10:
                continue
            # 策略 A: 偏差校正(不重训)
            if K > 0:
                bias = (warm[TARGET] - model0.predict(warm[FEATURES])).mean()
            else:
                bias = 0.0
            pA = model0.predict(test[FEATURES]) + bias
            # 策略 B: 混合重训
            tr_mix = pd.concat([tr_base, warm]) if K > 0 else tr_base
            pB = fit_median(tr_mix[FEATURES], tr_mix[TARGET]).predict(test[FEATURES])
            for tag, p in [("bias_correct", pA), ("retrain", pB)]:
                rows.append(dict(version="v0.0.3", test_group=g, K=K, strategy=tag,
                                 MAE=mean_absolute_error(test[TARGET], p),
                                 R2=r2_score(test[TARGET], p), n=len(test)))
            if g == "4C_2A" and K == 10:
                case_store = (test.copy(), pA, pB)
        r = [x for x in rows if x["test_group"] == g]
        print(f"{g}: K=0 MAE={r[0]['MAE']:.3f} → K=10 "
              f"偏差校正 {[x for x in r if x['K']==10 and x['strategy']=='bias_correct'][0]['MAE']:.3f} / "
              f"重训 {[x for x in r if x['K']==10 and x['strategy']=='retrain'][0]['MAE']:.3f} Ah")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "v03_cold_start.csv"), index=False, encoding="utf-8-sig")

    # ---- 图 1: MAE vs K ----
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), sharey=False)
    colors = {"bias_correct": "#1f77b4", "retrain": "#d62728"}
    labels = {"bias_correct": "偏差校正(在线,零重训)", "retrain": "混合重训(离线)"}
    for ax, g in zip(axes, GROUPS):
        sub = res[res.test_group == g]
        for st in ["bias_correct", "retrain"]:
            s = sub[sub.strategy == st]
            ax.plot(s.K, s.MAE, "o-", color=colors[st], label=labels[st], lw=1.8)
        ax.set_title(f"测试工况 {TITLES[g]}", fontsize=10)
        ax.set_xlabel("冷启动可用循环数 K"); ax.set_ylabel("MAE (Ah)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("V0.0.3 冷启动适配:新工况只需前几个循环的实测数据,预测误差大幅下降", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v03_cold_start.png"), dpi=160)

    # ---- 图 2: 4C_2A 案例(K=10) ----
    if case_store:
        test, pA, pB = case_store
        fig, ax = plt.subplots(figsize=(8, 4.5))
        test = test.copy(); test["pA"], test["pB"] = pA, pB
        for bid, s in test.groupby("battery"):
            ax.plot(s.cycle, s.capacity_ah, color="gray", alpha=0.4, lw=0.9)
        gm = test.groupby("cycle").agg(actual=("capacity_ah", "mean"),
                                       pA=("pA", "mean"), pB=("pB", "mean"))
        ax.plot(gm.index, gm.actual, "k-", lw=2.2, label="实测")
        ax.plot(gm.index, gm.pA, color="#1f77b4", lw=1.6, label="偏差校正(K=10)")
        ax.plot(gm.index, gm.pB, color="#d62728", lw=1.6, ls="--", label="混合重训(K=10)")
        ax.axvline(10, color="green", ls=":", lw=1.5)
        ax.text(11, ax.get_ylim()[1]*0.95, "← 前 10 循环用于适配", color="green", fontsize=9)
        mae0 = res[(res.test_group=="4C_2A")&(res.K==0)&(res.strategy=="retrain")].MAE.iloc[0]
        maeA = res[(res.test_group=="4C_2A")&(res.K==10)&(res.strategy=="bias_correct")].MAE.iloc[0]
        ax.set_title(f"案例:4 ℃/2A 低温工况 · 无适配 MAE={mae0:.3f} Ah → K=10 偏差校正 MAE={maeA:.3f} Ah")
        ax.set_xlabel("循环数"); ax.set_ylabel("容量 (Ah)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "v03_case_4C.png"), dpi=160)
    print("\n图已保存: figures/v03_cold_start.png, figures/v03_case_4C.png")


if __name__ == "__main__":
    main()
