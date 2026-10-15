# -*- coding: utf-8 -*-
"""
03_features.py  逐循环特征工程

数据来源: data/raw/nasa/*.mat (NASA PCoE)
输出:     data/processed/feature_table.csv

每个放电循环一行,特征来自:
  - 同循环充电段: CC/CV 时长、IC(dQ/dV)曲线峰位/峰高/面积、充电温升
  - 放电段本身: 时长、3.6V→3.0V 等压降时间、温度/电流统计
  - 最近一次阻抗测量: Re(欧姆内阻)、Rct(电荷转移阻抗)
目标变量: capacity_ah (放电容量)
"""
import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import savgol_filter, find_peaks

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nasa")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

GROUPS = {
    "B0005": "24C_2A", "B0006": "24C_2A", "B0007": "24C_2A", "B0018": "24C_2A",
    "B0033": "24C_4A", "B0034": "24C_4A", "B0036": "24C_4A",
    "B0029": "43C_4A", "B0030": "43C_4A", "B0031": "43C_4A", "B0032": "43C_4A",
    "B0053": "4C_2A", "B0054": "4C_2A", "B0055": "4C_2A", "B0056": "4C_2A",
    "B0041": "4C_mixed", "B0042": "4C_mixed", "B0043": "4C_mixed", "B0044": "4C_mixed",
}

V_GRID = np.arange(3.45, 4.195, 0.005)   # IC 分析电压窗口(CC 段)


def arr(x):
    return np.atleast_1d(np.asarray(x, dtype=float).ravel())


def ic_features(V, I, T):
    """充电 CC 段 IC(dQ/dV)曲线特征(稳健统计口径,不依赖 find_peaks)"""
    out = dict(ic_pos_max=np.nan, ic_height_max=np.nan, ic_area=np.nan,
               ic_mean=np.nan, ic_std=np.nan, cc_capacity_ah=np.nan,
               cc_duration_s=np.nan, cv_duration_s=np.nan, charge_temp_mean=np.nan)
    if len(V) < 50:
        return out
    out["charge_temp_mean"] = float(np.nanmean(T))
    cc = I > 1.4                      # 恒流段:电流 ≈1.5A
    if cc.sum() < 30:
        return out
    out["cc_duration_s"] = float(T[cc][-1] - T[cc][0])
    idx_cv = np.where(~cc)[0]
    idx_cv = idx_cv[idx_cv > np.where(cc)[0][-1]]
    if len(idx_cv):
        out["cv_duration_s"] = float(T[idx_cv[-1]] - T[idx_cv[0]])
    Vc, Tc, Ic = V[cc], T[cc], I[cc]
    order = np.argsort(Tc)
    Vc, Tc, Ic = Vc[order], Tc[order], Ic[order]
    dT = np.diff(Tc)
    valid = dT > 0
    q = np.concatenate([[0], np.cumsum(Ic[1:][valid] * dT[valid])]) / 3600.0  # Ah
    Vc = Vc[1:][valid]
    if len(Vc) < 30 or Vc.max() < 4.1:
        return out
    Vu, idx_u = np.unique(Vc, return_index=True)
    Qu = q[idx_u]
    v_lo, v_hi = Vu.min() + 0.01, 4.175
    if v_hi - v_lo < 0.25:            # CC 电压跨度过小,曲线不可用
        return out
    grid = np.arange(v_lo, v_hi, 0.005)
    if len(grid) < 30:
        return out
    Qg = np.interp(grid, Vu, Qu)
    win = min(31, len(grid) // 2 * 2 - 1)
    if win < 7:
        return out
    Qs = savgol_filter(Qg, win, 3)
    dQdV = np.gradient(Qs, grid)
    out["cc_capacity_ah"] = float(Qg[-1] - Qg[0])     # CC 段充入容量(经典健康因子)
    out["ic_area"] = float(np.trapezoid(dQdV, grid))
    out["ic_mean"] = float(dQdV.mean())
    out["ic_std"] = float(dQdV.std())
    imax = int(np.argmax(dQdV))
    out["ic_pos_max"] = float(grid[imax])
    out["ic_height_max"] = float(dQdV[imax])
    return out


def discharge_features(V, I, T):
    out = dict(dis_duration_s=np.nan, t_drop_36_30_s=np.nan,
               dis_temp_mean=np.nan, dis_current_mean=np.nan)
    if len(V) < 10:
        return out
    out["dis_duration_s"] = float(T.max())
    out["dis_temp_mean"] = float(np.nanmean(T))
    out["dis_current_mean"] = float(np.nanmean(np.abs(I)))
    mask = (V <= 3.6) & (V >= 3.0)
    if mask.sum() >= 2:
        out["t_drop_36_30_s"] = float(T[mask][-1] - T[mask][0])
    return out


def parse_battery(bid):
    mat = scipy.io.loadmat(os.path.join(RAW, f"{bid}.mat"),
                           struct_as_record=False, squeeze_me=True)
    cycles = np.atleast_1d(mat[bid].cycle)

    rows, last_charge, last_imp = [], None, (np.nan, np.nan)
    dis_n = 0
    for c in cycles:
        ctype = str(c.type)
        d = c.data
        if ctype == "charge":
            last_charge = ic_features(arr(d.Voltage_measured),
                                      arr(d.Current_measured), arr(d.Time))
        elif ctype == "impedance":
            last_imp = (float(getattr(d, "Re", np.nan)), float(getattr(d, "Rct", np.nan)))
        elif ctype == "discharge":
            cap = float(np.atleast_1d(getattr(d, "Capacity", np.nan)).ravel()[0])
            dis_n += 1
            row = {"battery": bid, "group": GROUPS[bid], "cycle": dis_n,
                   "capacity_ah": cap,
                   "ambient_temp": float(np.atleast_1d(c.ambient_temperature).ravel()[0]),
                   "Re": last_imp[0], "Rct": last_imp[1]}
            row.update(discharge_features(arr(d.Voltage_measured),
                                          arr(d.Current_measured), arr(d.Time)))
            if last_charge:
                row.update(last_charge)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    frames = []
    for bid in GROUPS:
        df = parse_battery(bid)
        frames.append(df)
        ic_ok = df.ic_height_max.notna().mean()
        print(f"{bid} [{GROUPS[bid]}]: {len(df)} 循环 | IC 特征成功率 {ic_ok:.0%}")
    feat = pd.concat(frames, ignore_index=True)

    # 与清洗规则对齐:剔除异常放电段
    feat = feat[(feat.capacity_ah >= 0.5) & (feat.capacity_ah <= 2.3)]
    def _outlier(s):
        med = s.rolling(5, center=True, min_periods=1).median()
        return (s - med).abs() > 0.35
    feat = feat[~feat.groupby("battery")["capacity_ah"].transform(_outlier)]
    feat["soh_rated"] = feat.capacity_ah / 2.0
    feat.to_csv(os.path.join(OUT, "feature_table.csv"), index=False, encoding="utf-8-sig")
    print(f"\nfeature_table.csv: {len(feat)} 行 × {feat.shape[1]} 列")
    print("IC 特征总体成功率: {:.0%}".format(feat.ic_height_max.notna().mean()))
    print("阻抗特征覆盖率: Re {:.0%} / Rct {:.0%}".format(feat.Re.notna().mean(), feat.Rct.notna().mean()))


if __name__ == "__main__":
    main()
