# -*- coding: utf-8 -*-
"""
09_v06_physics_temp.py  V0.0.6:物理温度校正 —— Arrhenius 因子 + 归一化目标

思路: 放电可用容量对温度的依赖是物理性的(低温下倍率性能下降),不是老化。
       把它从「老化信号」中剥离:
         c(T) = T 温度下首循环容量 / 24 ℃ 首循环容量(同倍率,组均值)
         训练目标改为 soh_eq = capacity / c(T) / 初始容量(温度归一化 SOH)
       测试时: 预测 soh_eq → 乘回 c(T_test) × 测试电芯首循环容量
       c(T_test) 的三种获得方式构成「盲测 → 冷启动」阶梯:
         K=0(盲测): 由训练组温度点 Arrhenius 拟合 ln c ~ 1/T 外推
         K=1:       用测试电芯首循环实测容量(运维上接入即知)
         K=10:      叠加前 10 循环偏差校正(V0.0.3 协议)
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
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=10, l2_regularization=1.0, random_state=42)
RATED = 2.0
GROUPS = ["24C_2A", "24C_4A", "43C_4A", "4C_2A"]
TITLES = {"24C_2A": "24 ℃/2A", "24C_4A": "24 ℃/4A", "43C_4A": "43 ℃/4A", "4C_2A": "4 ℃/2A"}
GROUP_TEMP = {"24C_2A": 24.0, "24C_4A": 24.0, "43C_4A": 43.0, "4C_2A": 4.0}


def fit_median(X, y):
    return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **PARAMS).fit(X, y)


def group_c(df):
    """各组温度因子 c(T): 组内首 3 循环容量均值 / 24C_2A 组同口径均值"""
    base = df[df.group == "24C_2A"]
    c0 = base[base.cycle <= 3].capacity_ah.mean()
    out = {}
    for g in GROUPS:
        s = df[df.group == g]
        out[g] = s[s.cycle <= 3].capacity_ah.mean() / c0
    return out


def arrhenius_extrap(c_train: dict, t_target_C: float) -> float:
    """由训练温度点拟合 ln c = A - B/T(K), 外推 t_target(℃)"""
    Ts = np.array([GROUP_TEMP[g] + 273.15 for g in c_train])
    lnc = np.log(np.array([c_train[g] for g in c_train]))
    A = np.vstack([np.ones_like(Ts), -1.0 / Ts]).T
    coef, *_ = np.linalg.lstsq(A, lnc, rcond=None)
    return float(np.exp(coef[0] + coef[1] * (-1.0 / (t_target_C + 273.15))))


def main():
    df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "feature_table.csv"))
    df = df[df.group != "4C_mixed"].reset_index(drop=True)
    c_all = group_c(df)
    print("各组温度因子 c(T):", {k: round(v, 3) for k, v in c_all.items()})

    rows, case = [], {}
    for g in GROUPS:
        tr, te = df[df.group != g].copy(), df[df.group == g].copy()
        T_te = GROUP_TEMP[g]
        # --- 三种 c(T_test) 估计 ---
        c_train = {k: v for k, v in c_all.items() if k != g}
        # K=0: Arrhenius 外推(仅当测试温度在训练温度范围外才有意义; 同温度直接取 1 或最近组)
        c_k0 = arrhenius_extrap(c_train, T_te)
        # K=1: 测试电芯首循环实测(逐电芯; 首循环被清洗剔除的电芯回退到组均值)
        first1 = te[te.cycle <= 1].groupby("battery").capacity_ah.mean()
        first1_map = te.battery.map(first1)
        first1_map = first1_map.fillna(te[te.cycle <= 3].capacity_ah.mean())

        # --- 训练: 目标为温度归一化 SOH ---
        tr["y"] = tr.capacity_ah / tr.group.map(c_all) / RATED
        model = fit_median(tr[FEATURES], tr.y)
        soh_hat = model.predict(te[FEATURES])

        for tag, cap_hat in [
            ("K0_Arrhenius", soh_hat * RATED * c_k0),
            ("K1_first_cycle", soh_hat * first1_map.values),
        ]:
            rows.append(dict(version="v0.0.6", test_group=g, mode=tag,
                             c_est=round(c_k0, 3), c_true=round(c_all[g], 3),
                             MAE=mean_absolute_error(te.capacity_ah, cap_hat),
                             R2=r2_score(te.capacity_ah, cap_hat)))
        # K=10: K1 基础上叠加偏差校正
        warm = te[te.cycle <= 10]
        warm_pred = fit_median(tr[FEATURES], tr.y).predict(warm[FEATURES]) * warm.battery.map(first1).fillna(warm.capacity_ah.mean()).values
        bias = (warm.capacity_ah - warm_pred).mean()
        test = te[te.cycle > 10]
        pred = fit_median(tr[FEATURES], tr.y).predict(test[FEATURES]) * test.battery.map(first1).fillna(test.capacity_ah.mean()).values + bias
        rows.append(dict(version="v0.0.6", test_group=g, mode="K10_bias",
                         c_est=np.nan, c_true=round(c_all[g], 3),
                         MAE=mean_absolute_error(test.capacity_ah, pred),
                         R2=r2_score(test.capacity_ah, pred)))
        case[g] = (te, soh_hat * RATED * c_k0, soh_hat * first1_map.values)
        print(f"{g}: c_true={c_all[g]:.3f} c_Arrhenius={c_k0:.3f} | "
              f"K0 MAE={rows[-3]['MAE']:.3f} K1 MAE={rows[-2]['MAE']:.3f} K10 MAE={rows[-1]['MAE']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "v06_physics_temp.csv"), index=False, encoding="utf-8-sig")

    # ---- 图: 4 工况 × 3 种模式 MAE 对比 ----
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(GROUPS)); w = 0.26
    modes = [("K0_Arrhenius", "K=0 盲测(Arrhenius 外推)", "#7f7f7f"),
             ("K1_first_cycle", "K=1(首循环归一化)", "#1f77b4"),
             ("K10_bias", "K=10(叠加偏差校正)", "#d62728")]
    for i, (m, label, color) in enumerate(modes):
        vals = [res[(res.test_group == g) & (res["mode"] == m)].MAE.iloc[0] for g in GROUPS]
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=color, alpha=0.85)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    # 参考线: V0.0.3 同协议结果(原始容量目标)
    ax.axhline(0.299, color="gray", ls=":", lw=1)
    ax.text(2.6, 0.305, "V0.0.2 无适配 4 ℃ MAE=0.299", fontsize=8, color="gray")
    ax.set_xticks(x); ax.set_xticklabels([TITLES[g] for g in GROUPS])
    ax.set_ylabel("MAE (Ah)")
    ax.set_title("V0.0.6 物理温度校正:温度归一化目标 + 三档冷启动阶梯的跨工况误差")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v06_physics_temp.png"), dpi=160)
    print("图已保存: figures/v06_physics_temp.png")


if __name__ == "__main__":
    main()
