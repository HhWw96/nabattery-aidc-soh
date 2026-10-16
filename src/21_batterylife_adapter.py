# -*- coding: utf-8 -*-
"""
21_batterylife_adapter.py  BatteryLife NA-ion 数据适配器(预写,待数据到手即跑)

状态声明: ⚠️ 未经实测 —— 数据获取受阻记录:
  1. HuggingFace 镜像(hf-mirror): 源仓库 Battery-Life/BatteryLife_Processed 为受限仓库, 需申请权限
  2. Zenodo 整包: NA-ion.zip 289 MB, 实测限速 ~1-15 KB/s, 冻结窗口内不可行
  3. Zenodo zip 分段手术: 已解析中央目录(65 个条目, 每芯 ~4.5-5 MB deflate),
     可定点下载单芯(HTTP Range), 但并行下载被限速到 ~1 KB/s/连接 → 放弃, 保留 fetch 方案注释
  4. 已向社区内已下载该数据的伙伴发出协同邀请(见帖子 04 评论区)

数据格式(BatteryLife processed, 遵循 BatteryML 标准):
  每个 pkl 为一只电芯: dict, 含 cycle_data(list[dict]), 每循环字段含
  current_in_A / voltage_in_V / charge_capacity_in_Ah / discharge_capacity_in_Ah / time_in_s 等;
  Life labels/NA-ion_labels.json 给出各电芯寿命标签
已知规格(据社区帖): 18650 钠电, 1.0 Ah, 2.0-4.0 V, 2.5C 充放, 25 ℃, 64 只, 循环数 91-251

适配输出: 与 NASA 管线相同的逐循环表 schema (battery/cycle/capacity_ah/...),
          之后 src/03 → src/18 全流程可直接复跑
"""
import os
import pickle
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "batterylife_naion")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def pkl_to_cycle_table(path: str) -> pd.DataFrame:
    """BatteryLife pkl → 逐循环容量表(对齐 NASA 管线 schema)"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    bid = os.path.basename(path).replace(".pkl", "")
    cycles = d["cycle_data"] if isinstance(d, dict) and "cycle_data" in d else d
    rows = []
    for i, c in enumerate(cycles, 1):
        # BatteryML 标准字段; 放电容量优先取 discharge_capacity_in_Ah
        cap = c.get("discharge_capacity_in_Ah") or c.get("discharge_capacity")
        if hasattr(cap, "__len__"):
            cap = float(np.max(cap))      # 序列则取末值
        rows.append(dict(battery=bid, group="NA_ion_25C_2.5C", cycle=i,
                         capacity_ah=float(cap), ambient_temp=25.0))
    return pd.DataFrame(rows)


def planned_validation(na_table: pd.DataFrame, nasa_feature_table: pd.DataFrame):
    """计划中的验证协议(数据到手后执行):
    1. EDA: 钠电衰减曲线 vs 锂电同倍率组(膝点位置、再生现象对比)
    2. 零校准迁移: NASA 24 ℃ 组训练的 V1.0 模型直接预测钠电 → 量化跨化学体系偏移
    3. 冷启动校准: 前 10 循环偏差校正 → 验证「锂电预训练+钠电校准」假设
    4. 区间审计: V1.5 协议(2σ×门控)在钠电上的覆盖率
    """
    raise NotImplementedError("等待钠电数据到手")


if __name__ == "__main__":
    import glob
    pkls = glob.glob(os.path.join(RAW, "*.pkl"))
    if not pkls:
        print("未发现钠电 pkl 数据。下载受阻记录见本文件 docstring。")
        print("zenodo 分段手术目录: data/raw/batterylife_naion/zip_catalog.json(65 条目已解析)")
    else:
        tabs = [pkl_to_cycle_table(p) for p in pkls]
        out = pd.concat(tabs, ignore_index=True)
        out.to_csv(os.path.join(OUT, "naion_capacity.csv"), index=False, encoding="utf-8-sig")
        print(f"已适配 {len(pkls)} 只钠电芯 → naion_capacity.csv ({len(out)} 行)")
