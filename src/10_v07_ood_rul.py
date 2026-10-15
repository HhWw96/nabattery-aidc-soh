# -*- coding: utf-8 -*-
"""
10_v07_ood_rul.py  V0.0.7:分布外检测(拒绝预测机制) + RUL 剩余寿命预测头

Part 1 OOD 检测器(安全机制的落地):
  - 每只电芯用前 3 循环的 4 个温度敏感特征均值做指纹:
    [ambient_temp, dis_temp_mean, cc_capacity_ah, ic_height_max]
  - 测试电芯指纹到训练集电芯指纹分布的马氏距离 > 阈值(训练集内 LOBO 最大距离) → 标记 OOD
  - 验收标准: 4 ℃ 电芯应全部被标记; 分布内电芯应全部通过
  - 通过/被标记两组的预测质量对比 → 「拒识」是可信预警的最后防线

Part 2 RUL 预测头:
  - 目标: 距 SOH<80%(以本电芯首 3 循环均值为基准)的剩余循环数
  - 分位数梯度提升输出 RUL 中位数与区间, 留一工况评估
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
FINGERPRINT = ["ambient_temp", "dis_temp_mean", "cc_capacity_ah", "ic_height_max"]
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=10, l2_regularization=1.0, random_state=42)
GROUPS = ["24C_2A", "24C_4A", "43C_4A", "4C_2A"]
TITLES = {"24C_2A": "24 ℃/2A", "24C_4A": "24 ℃/4A", "43C_4A": "43 ℃/4A", "4C_2A": "4 ℃/2A"}


def fingerprints(df):
    """每只电芯前 3 循环的指纹向量"""
    fp = df[df.cycle <= 3].groupby(["battery", "group"])[FINGERPRINT].mean()
    fp["cc_capacity_ah"] = fp["cc_capacity_ah"].fillna(df.cc_capacity_ah.mean())
    fp["ic_height_max"] = fp["ic_height_max"].fillna(df.ic_height_max.mean())
    return fp


def knn_detector(train_fp: pd.DataFrame, margin=1.5):
    """kNN(k=1) 距离检测器: z-score 标准化后, 到最近训练电芯指纹的欧氏距离。
       阈值 = 训练电芯留一最近邻距离的 max × margin(小样本下比马氏距离稳健)"""
    X = train_fp[FINGERPRINT].values
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xn = (X - mu) / sd
    # 训练集留一最近邻距离
    loo = []
    for i in range(len(Xn)):
        d = np.sqrt(((Xn[np.arange(len(Xn)) != i] - Xn[i]) ** 2).sum(1))
        loo.append(d.min())
    thresh = max(loo) * margin
    def dist(v):
        vn = (v - mu) / sd
        return np.sqrt(((Xn[None, :, :] - vn[:, None, :]) ** 2).sum(2)).min(1)
    return dist, thresh


def part1(df):
    print("== Part 1: OOD 检测器 ==")
    rows = []
    for g in GROUPS:
        tr, te = df[df.group != g], df[df.group == g]
        fp_tr, fp_te = fingerprints(tr), fingerprints(te)
        dist, thresh = knn_detector(fp_tr)
        d_te = dist(fp_te[FINGERPRINT].values)
        flagged = d_te > thresh
        # 同组内自检: 训练集指纹距离分布
        d_tr_self = dist(fp_tr[FINGERPRINT].values)
        rows.append(dict(test_group=g, threshold=round(thresh, 2),
                         test_dist_mean=round(d_te.mean(), 2),
                         train_dist_mean=round(d_tr_self.mean(), 2),
                         flagged_ratio=flagged.mean(),
                         n_flagged=int(flagged.sum()), n_test=len(d_te)))
        print(f"  测试 {g}: 距离 {d_te.mean():.1f} vs 阈值 {thresh:.1f} "
              f"(训练组均值 {d_tr_self.mean():.1f}) → 标记 {flagged.sum()}/{len(d_te)} 只电芯")
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(RES, "v07_ood.csv"), index=False, encoding="utf-8-sig")
    return r


def part2(df):
    print("\n== Part 2: RUL 预测头 ==")
    # EOL: 本电芯首 3 循环容量均值 × 0.8
    first3 = df[df.cycle <= 3].groupby("battery").capacity_ah.mean().rename("cap0")
    df = df.merge(first3, on="battery")
    df["eol"] = df.cap0 * 0.8
    eol_cycle = (df[df.capacity_ah < df.eol].groupby("battery").cycle.min())
    max_cycle = df.groupby("battery").cycle.max()
    def rul_of(bid, cyc):
        end = eol_cycle.get(bid, np.nan)
        if np.isnan(end):
            end = max_cycle[bid] + 30        # 实验内未达 EOL → 右删失, 用 max+30 近似并标记
        return max(end - cyc, 0)
    df["rul"] = [rul_of(b, c) for b, c in zip(df.battery, df.cycle)]
    df["rul"] = df.rul.clip(upper=250)

    rows, case = [], None
    for g in GROUPS:
        tr, te = df[df.group != g], df[df.group == g]
        models = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **PARAMS).fit(tr[FEATURES], tr.rul)
                  for q in (0.05, 0.5, 0.9)}
        p50 = models[0.5].predict(te[FEATURES])
        p05, p90 = models[0.05].predict(te[FEATURES]), models[0.9].predict(te[FEATURES])
        picp = np.mean((te.rul >= p05) & (te.rul <= p90))
        rows.append(dict(test_group=g, MAE=mean_absolute_error(te.rul, p50), PICP90=picp, n=len(te)))
        print(f"  测试 {g}: RUL MAE={rows[-1]['MAE']:.1f} 循环, PICP90={picp:.0%}")
        if g == "24C_2A":
            case = (te, p50, p05, p90)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "v07_rul.csv"), index=False, encoding="utf-8-sig")

    # RUL 案例图: B0005 逐循环 RUL 预测
    te, p50, p05, p90 = case
    s = te[te.battery == "B0005"].sort_values("cycle")
    idx = s.index
    fig, ax = plt.subplots(figsize=(9, 4.6))
    pos = [list(te.index).index(i) for i in idx]
    ax.plot(s.cycle, s.rul, "k-", lw=2, label="真实 RUL")
    ax.plot(s.cycle, p50[pos], color="#d62728", lw=1.6, label="预测 RUL(中位数)")
    ax.fill_between(s.cycle, p05[pos], p90[pos], color="#d62728", alpha=0.2, label="90% 区间")
    ax.set_xlabel("当前循环数"); ax.set_ylabel("剩余寿命(循环)")
    ax.set_title("V0.0.7 RUL 预测案例:B0005(留一工况训练,仅前 3 循环可见的电芯级 EOL 基准)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v07_rul_case.png"), dpi=160)
    print("  图已保存: figures/v07_rul_case.png")
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "feature_table.csv"))
    df = df[df.group != "4C_mixed"].reset_index(drop=True)
    ood = part1(df)
    rul = part2(df)
    return ood, rul


if __name__ == "__main__":
    main()
