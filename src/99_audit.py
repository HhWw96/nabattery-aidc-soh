# -*- coding: utf-8 -*-
"""
99_audit.py  冻结前审计:帖子/PPT/讲稿中引用的每一个数字 vs results/*.csv 与数据文件

每条断言: PASS / FAIL(实际值) —— 任何 FAIL 必须在发布物中修正
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")

results = []

def check(name, actual, expected, tol=0.015, kind="approx"):
    if kind == "approx":
        ok = abs(actual - expected) <= tol * max(abs(expected), 1e-9) if expected != 0 else abs(actual) <= tol
    elif kind == "range":
        lo, hi = expected
        ok = lo - tol <= actual <= hi + tol
    elif kind == "eq":
        ok = actual == expected
    results.append((name, actual, expected, ok))
    print(f"{'PASS' if ok else 'FAIL'} | {name}: 实际 {actual} vs 引用 {expected}")


# ---------- 数据层 ----------
clean = pd.read_csv(os.path.join(ROOT, "data", "processed", "capacity_clean.csv"))
raw = pd.read_csv(os.path.join(ROOT, "data", "processed", "capacity_table.csv"))
main = clean[clean.group != "4C_mixed"]

check("清洗后放电记录数(帖子1:1949)", len(clean), 1949, kind="eq")
check("清洗剔除条数(帖子1:204)", len(raw) - len(clean), 204, kind="eq")
check("电芯总数(帖子1:19)", clean.battery.nunique(), 19, kind="eq")

f24 = main[(main.group == "24C_2A") & (main.cycle <= 3)].capacity_ah.mean()
f4 = main[(main.group == "4C_2A") & (main.cycle <= 3)].capacity_ah.mean()
check("24 ℃/2A 首循环容量(帖子1:1.90)", round(f24, 2), 1.90, kind="eq")
check("4 ℃/2A 首循环容量(帖子1:1.21)", round(f4, 2), 1.21, kind="eq")
check("低温容量损失(帖子1/PPT:36%)", round((1 - f4 / f24) * 100), 36, kind="eq")

# ---------- V0.0.1 ----------
v1 = pd.read_csv(os.path.join(RES, "v1_metrics.csv"))
lobo = v1[v1.split == "LOBO_24C_2A"]
check("V0.0.1 同工况 MAE 下限(0.020)", lobo.MAE.min(), 0.020, tol=0.005)
check("V0.0.1 同工况 MAE 上限(0.045)", lobo.MAE.max(), 0.045, tol=0.005)
check("V0.0.1 同工况 R² 下限(0.945)", lobo.R2.min(), 0.945, tol=0.005)
check("V0.0.1 4 ℃ MAE(0.489)", v1[v1.test == "4C_2A"].MAE.iloc[0], 0.489, tol=0.005)
check("V0.0.1 4 ℃ PICP=0%", v1[v1.test == "4C_2A"].PICP90.iloc[0], 0.0, kind="eq")

# ---------- V0.0.2 ----------
v02 = pd.read_csv(os.path.join(RES, "v02_metrics.csv")).set_index("test_group")
check("V0.0.2 24C_2A 校准前 PICP(25%)", v02.loc["24C_2A", "raw_PICP90"], 0.25, tol=0.03)
check("V0.0.2 24C_2A 校准后 PICP(95%)", v02.loc["24C_2A", "cal_PICP90"], 0.95, tol=0.03)
check("V0.0.2 4 ℃ 校准后 PICP(8%)", v02.loc["4C_2A", "cal_PICP90"], 0.08, tol=0.03)
check("V0.0.2 24C_4A MAE(0.086)", v02.loc["24C_4A", "cal_MAE"], 0.086, tol=0.005)
check("V0.0.2 4 ℃ MAE(0.299)", v02.loc["4C_2A", "cal_MAE"], 0.299, tol=0.005)

# ---------- V0.0.3 ----------
v03 = pd.read_csv(os.path.join(RES, "v03_cold_start.csv"))
def v03get(g, K, s):
    return v03[(v03.test_group == g) & (v03.K == K) & (v03.strategy == s)].MAE.iloc[0]
check("V0.0.3 4 ℃ K=10 偏差校正(0.131)", v03get("4C_2A", 10, "bias_correct"), 0.131, tol=0.005)
check("V0.0.3 43 ℃ K=10 偏差校正(0.044)", v03get("43C_4A", 10, "bias_correct"), 0.044, tol=0.005)
check("V0.0.3 24C_2A K=10 偏差校正变差(0.214)", v03get("24C_2A", 10, "bias_correct"), 0.214, tol=0.005)

# ---------- V0.0.4 ----------
v04 = pd.read_csv(os.path.join(RES, "v04_seq_metrics.csv"))
same = v04[v04.scenario.str.startswith("同工况")]
check("V0.0.4 同工况 GBR-window(0.012)", same[same.model == "GBR-window"].MAE.iloc[0], 0.012, tol=0.003)
rate = v04[v04.scenario.str.startswith("倍率")]
check("V0.0.4 倍率泛化 GBR-window(0.028)", rate[rate.model == "GBR-window"].MAE.iloc[0], 0.028, tol=0.003)
temp = v04[v04.scenario.str.startswith("温度")]
check("V0.0.4 温度泛化全部 R²<0", bool((temp.R2 < 0).all()), True, kind="eq")

# ---------- V0.0.6 ----------
v06 = pd.read_csv(os.path.join(RES, "v06_physics_temp.csv"))
v06_4c = v06[v06.test_group == "4C_2A"].set_index("mode")
check("V0.0.6 4 ℃ K=1(0.176)", v06_4c.loc["K1_first_cycle", "MAE"], 0.176, tol=0.005)
check("V0.0.6 4 ℃ K=10(0.088)", v06_4c.loc["K10_bias", "MAE"], 0.088, tol=0.005)
check("PPT -71% 降幅(0.299→0.088)", round((1 - 0.088 / 0.299) * 100), 71, kind="eq")

# ---------- V0.0.7 ----------
v07 = pd.read_csv(os.path.join(RES, "v07_ood.csv")).set_index("test_group")
check("V0.0.7 放行 24C_2A(0/4)", v07.loc["24C_2A", "n_flagged"], 0, kind="eq")
check("V0.0.7 放行 24C_4A(0/3)", v07.loc["24C_4A", "n_flagged"], 0, kind="eq")
check("V0.0.7 拒识 43C_4A(4/4)", v07.loc["43C_4A", "n_flagged"], 4, kind="eq")
check("V0.0.7 拒识 4C_2A(4/4)", v07.loc["4C_2A", "n_flagged"], 4, kind="eq")

# ---------- V0.0.8 ----------
v08 = pd.read_csv(os.path.join(RES, "v08_epistemic.csv")).set_index("test_group")
check("V0.0.8 4 ℃ 集成 std(0.031)", v08.loc["4C_2A", "std_mean"], 0.031, tol=0.003)
check("V0.0.8 4 ℃ std-误差相关(0.00)", v08.loc["4C_2A", "spearman_std_err"], 0.00, tol=0.05)
coral = pd.read_csv(os.path.join(RES, "v08_coral.csv")).set_index("test_group")
check("V0.0.8 CORAL 43 ℃ 变差(0.104→0.255)", coral.loc["43C_4A", "MAE_coral"], 0.255, tol=0.005)

# ---------- V0.0.9 ----------
v09 = pd.read_csv(os.path.join(RES, "v09_knee_rul.csv"))
f50 = v09[v09.frac == 0.5]; f70 = v09[v09.frac == 0.7]
check("V0.0.9 前50% 双指数 MAE(71)", f50.err_de.abs().mean(), 71, tol=0.1)
check("V0.0.9 前50% 拐点 MAE(82)", f50.err_knee.abs().mean(), 82, tol=0.1)
check("V0.0.9 前70% 拐点 MAE(18)", f70.err_knee.abs().mean(), 18, tol=0.1)

# ---------- V1.0 定稿 ----------
v10 = pd.read_csv(os.path.join(RES, "final_metrics.csv"))
v10m = v10.set_index(["test_group", "K"])
FINAL = {("24C_2A", 0): 0.049, ("24C_2A", 10): 0.042,
         ("24C_4A", 0): 0.051, ("24C_4A", 10): 0.051,
         ("43C_4A", 0): 0.100, ("43C_4A", 10): 0.012,
         ("4C_2A", 0): 0.267, ("4C_2A", 10): 0.125}
for (g, K), exp in FINAL.items():
    check(f"V1.0 {g} K={K} MAE({exp})", v10m.loc[(g, K), "MAE"], exp, tol=0.003)
FINAL_P = {("24C_2A", 10): 0.97, ("24C_4A", 10): 0.95, ("43C_4A", 10): 1.00, ("4C_2A", 10): 0.78}
for (g, K), exp in FINAL_P.items():
    check(f"V1.0 {g} K={K} PICP({exp:.0%})", v10m.loc[(g, K), "PICP90"], exp, tol=0.03)

print("\n" + "=" * 50)
fails = [r for r in results if not r[3]]
print(f"审计完成: {len(results) - len(fails)}/{len(results)} 项通过")
if fails:
    print("FAIL 项:")
    for name, actual, expected, _ in fails:
        print(f"  - {name}: 实际 {actual} vs 引用 {expected}")
