# -*- coding: utf-8 -*-
"""
01_download_parse.py
解析 NASA PCoE 锂电池老化 .mat 数据 → 逐循环汇总表

数据来源: github.com/devhaaana/nasa-battery-aging-dataset (NASA PCoE 官方数据镜像)
原始引用: B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository.

输出:
  data/processed/cycle_summary.csv   所有循环段(含 charge/discharge/impedance)
  data/processed/capacity_table.csv  放电循环容量表(含 SOH)
"""
import os
import numpy as np
import pandas as pd
import scipy.io

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nasa")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT, exist_ok=True)

# 工况分组(依据各组官方 README)
GROUPS = {
    "B0005": "24C_2A", "B0006": "24C_2A", "B0007": "24C_2A", "B0018": "24C_2A",
    "B0033": "24C_4A", "B0034": "24C_4A", "B0036": "24C_4A",   # 常温高倍率(B0036 实为 2A)
    "B0029": "43C_4A", "B0030": "43C_4A", "B0031": "43C_4A", "B0032": "43C_4A",
    "B0053": "4C_2A", "B0054": "4C_2A", "B0055": "4C_2A", "B0056": "4C_2A",  # 低温,倍率与训练组一致
    "B0041": "4C_mixed", "B0042": "4C_mixed", "B0043": "4C_mixed", "B0044": "4C_mixed",  # 寿命内变倍率,备选
}
NOMINAL_CAPACITY_AH = 2.0


def scalar(x):
    """从 mat 结构字段安全取标量"""
    try:
        return float(np.asarray(x).ravel()[0])
    except Exception:
        return np.nan


def parse_battery(bid: str) -> pd.DataFrame:
    path = os.path.join(RAW, f"{bid}.mat")
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    cycles = np.atleast_1d(mat[bid].cycle)

    rows = []
    seq = {"charge": 0, "discharge": 0, "impedance": 0}
    for c in cycles:
        ctype = str(c.type)
        seq[ctype] = seq.get(ctype, 0) + 1
        row = {
            "battery": bid,
            "group": GROUPS[bid],
            "type": ctype,
            "seq_idx": seq[ctype],          # 该类型内的序号(放电序号即常用 cycle 号)
            "ambient_temp": scalar(c.ambient_temperature),
        }
        d = c.data
        if ctype == "discharge":
            row["capacity_ah"] = scalar(getattr(d, "Capacity", np.nan))
            row["temp_mean"] = scalar(np.nanmean(np.atleast_1d(d.Temperature_measured)))
            row["current_mean"] = scalar(np.nanmean(np.abs(np.atleast_1d(d.Current_measured))))
            row["voltage_min"] = scalar(np.nanmin(np.atleast_1d(d.Voltage_measured)))
            row["duration_s"] = scalar(np.nanmax(np.atleast_1d(d.Time)))
        elif ctype == "charge":
            row["temp_mean"] = scalar(np.nanmean(np.atleast_1d(d.Temperature_measured)))
            row["duration_s"] = scalar(np.nanmax(np.atleast_1d(d.Time)))
            row["n_samples"] = int(np.atleast_1d(d.Time).size)   # 供后续 IC/DV 特征判断是否可用
        elif ctype == "impedance":
            row["Re"] = scalar(getattr(d, "Re", np.nan))
            row["Rct"] = scalar(getattr(d, "Rct", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    frames = []
    for bid in GROUPS:
        df = parse_battery(bid)
        frames.append(df)
        n_dis = (df.type == "discharge").sum()
        print(f"{bid} [{GROUPS[bid]}]: {len(df)} 段记录, 放电 {n_dis} 次")
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(os.path.join(OUT, "cycle_summary.csv"), index=False, encoding="utf-8-sig")

    cap = full[full.type == "discharge"].copy()
    cap = cap.rename(columns={"seq_idx": "cycle"})
    cap["soh_rated"] = cap["capacity_ah"] / NOMINAL_CAPACITY_AH
    # 按电池首循环容量归一的 SOH(文献常用口径之一)
    cap["soh_initial"] = cap.groupby("battery")["capacity_ah"].transform(lambda s: s / s.iloc[0])
    cap.to_csv(os.path.join(OUT, "capacity_table.csv"), index=False, encoding="utf-8-sig")

    # ---- 清洗:剔除异常放电段(中断测试/伪记录) ----
    # 规则:容量在 [0.5, 2.3] Ah 之外 → 丢弃;
    #       与同电芯前后 5 循环滚动中位数偏差 > 0.35 Ah → 丢弃(孤立尖峰/深谷)
    clean = cap[(cap.capacity_ah >= 0.5) & (cap.capacity_ah <= 2.3)].copy()
    def _flag_outlier(s):
        med = s.rolling(5, center=True, min_periods=1).median()
        return (s - med).abs() > 0.35
    outlier = clean.groupby("battery")["capacity_ah"].transform(_flag_outlier)
    dropped = clean[outlier]
    clean = clean[~outlier]
    clean.to_csv(os.path.join(OUT, "capacity_clean.csv"), index=False, encoding="utf-8-sig")
    print(f"\ncapacity_table.csv: {len(cap)} 行 | 清洗后 capacity_clean.csv: {len(clean)} 行 (丢弃 {len(cap)-len(clean)} 行)")
    if len(dropped):
        print("丢弃明细:", dropped.groupby("battery").size().to_dict())
    print(clean.groupby("group")["cycle"].agg(["count", "max"]))


if __name__ == "__main__":
    main()
