# -*- coding: utf-8 -*-
"""
15_v12_aidc_pulse.py  V1.2:AIDC 脉冲工况闭环仿真(升级 V0.0.5 的玩具模型)

升级点:负载从「恒定 2A」换成 AIDC 真实画像 —— 基线负载 + GPU 任务脉冲:
  - 基线:0.5C 持续负载(推理/空闲)
  - 脉冲:每循环内含 8% 时间的 2C 脉冲(GPU 训练任务突发)
  - 老化规则(假设参数,已在文中声明):
      每循环老化 = k · (基线吞吐 · 1.0 + 脉冲吞吐 · 1.5)   ← 脉冲老化惩罚 1.5×
      k 由 NASA 24C_2A 组实测衰减率标定(0.0036 Ah/循环 @ 1.9 Ah/循环)
  - 自适应策略:预测 SOH 的 90% 区间下限 < 0.85 → 脉冲限幅到 1C
      (脉冲惩罚降为 1.1×,但被限幅的脉冲能量只计 50% 服务质量)
  - 评估三指标:寿命终点循环数 / 寿命期总吞吐 / 脉冲服务质量(全额脉冲能量占比)

诚实声明:脉冲惩罚系数为假设(文献中高倍率加速老化),非实测。
产出: results/v12_aidc_pulse.csv, figures/v12_aidc_pulse.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, "..", "figures"); os.makedirs(FIG, exist_ok=True)

CAP0 = 1.9
EOL = 0.8
THRESH = 0.85
K_FADE = 0.0036 / 1.9        # 每 Ah「等效吞吐」的老化(Ah),由 24C_2A 标定
BASE_RATE_AH = 1.9 * 0.5     # 每循环基线吞吐(0.5C × 1h 等效)
PULSE_AH = 1.9 * 2.0 * 0.08  # 每循环脉冲吞吐(2C × 8% 时间)
PULSE_PENALTY = 1.5          # 脉冲老化惩罚(假设)
PULSE_PENALTY_CAPPED = 1.1   # 限幅 1C 后的惩罚(假设)
PULSE_SERVICE_CAPPED = 0.5   # 限幅后脉冲能量计 50% 服务质量
PRED_SIGMA = 0.02 / 1.9      # 预测噪声(V0.0.4 同工况水平,SOH 尺度)


def simulate(adaptive, seed, horizon=400):
    rng = np.random.default_rng(seed)
    cap = CAP0
    hist = []
    for cyc in range(1, horizon + 1):
        soh = cap / CAP0
        pred = soh + rng.normal(0, PRED_SIGMA)
        lower = pred - 1.64 * PRED_SIGMA
        capped = adaptive and (lower < THRESH)
        penalty = PULSE_PENALTY_CAPPED if capped else PULSE_PENALTY
        aging = K_FADE * (BASE_RATE_AH + PULSE_AH * penalty)
        cap -= aging
        throughput = BASE_RATE_AH + PULSE_AH * (PULSE_SERVICE_CAPPED if capped else 1.0)
        hist.append(dict(cycle=cyc, soh=cap / CAP0, throughput=throughput, capped=capped))
        if cap / CAP0 < EOL:
            break
    return pd.DataFrame(hist)


def main():
    rows = []
    for seed in range(20):
        for adaptive in (False, True):
            h = simulate(adaptive, seed)
            rows.append(dict(strategy="自适应限幅" if adaptive else "固定全幅脉冲",
                             seed=seed, cycles=len(h), total_ah=h.throughput.sum(),
                             capped_ratio=h.capped.mean()))
    r = pd.DataFrame(rows)
    summ = r.groupby("strategy").agg(cycles=("cycles", "mean"), total_ah=("total_ah", "mean"),
                                     capped=("capped_ratio", "mean")).round(2)
    print(summ)
    gain_thpt = (summ.loc["自适应限幅", "total_ah"] / summ.loc["固定全幅脉冲", "total_ah"] - 1) * 100
    gain_life = (summ.loc["自适应限幅", "cycles"] / summ.loc["固定全幅脉冲", "cycles"] - 1) * 100
    print(f"寿命 +{gain_life:.1f}%, 吞吐 +{gain_thpt:.1f}%")
    r.to_csv(os.path.join(RES, "v12_aidc_pulse.csv"), index=False, encoding="utf-8-sig")

    # ---- 图: 负载示意 + 轨迹 + 双指标 ----
    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.4, 1])
    # (a) 负载画像
    ax0 = fig.add_subplot(gs[0])
    t = np.linspace(0, 3600, 500)
    load = np.full_like(t, 0.5)
    rng = np.random.default_rng(3)
    for start in rng.choice(np.arange(0, 3400, 100), 6, replace=False):
        load[(t >= start) & (t < start + 180)] = 2.0
    ax0.fill_between(t / 60, 0, load, color="#22D3A5", alpha=0.7)
    ax0.set_xlabel("时间 (分钟)"); ax0.set_ylabel("负载率 (C)")
    ax0.set_title("AIDC 负载画像(示意)\n基线 0.5C + GPU 脉冲 2C", fontsize=10)
    ax0.set_ylim(0, 2.4); ax0.grid(alpha=0.3)
    # (b) SOH 轨迹
    ax1 = fig.add_subplot(gs[1])
    h0, h1 = simulate(False, 7), simulate(True, 7)
    ax1.plot(h0.cycle, h0.soh, color="#1f77b4", lw=2, label="固定全幅脉冲")
    ax1.plot(h1.cycle, h1.soh, color="#d62728", lw=2, label="自适应限幅(区间下限驱动)")
    ax1.fill_between(h1.cycle[h1.capped], 0.76, 1.0, color="#d62728", alpha=0.08, label="限幅区间")
    ax1.axhline(EOL, color="gray", ls="--", lw=1); ax1.axhline(THRESH, color="green", ls=":", lw=1)
    ax1.set_xlabel("循环数"); ax1.set_ylabel("SOH"); ax1.set_ylim(0.76, 1.0)
    ax1.set_title(f"脉冲工况下的 SOH 轨迹:寿命 +{gain_life:.0f}%", fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    # (c) 双指标
    ax2 = fig.add_subplot(gs[2])
    x = np.arange(2)
    thpt = [summ.loc["固定全幅脉冲", "total_ah"], summ.loc["自适应限幅", "total_ah"]]
    bars = ax2.bar(["固定", "自适应"], thpt, color=["#1f77b4", "#d62728"], alpha=0.85, width=0.45)
    for b, v in zip(bars, thpt):
        ax2.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f} Ah", ha="center", fontsize=10)
    ax2.set_ylabel("寿命期总吞吐 (Ah)")
    ax2.set_title(f"吞吐 +{gain_thpt:.1f}%(脉冲服务质量代价已计入)", fontsize=10)
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("V1.2 AIDC 脉冲工况闭环仿真(概念验证,脉冲惩罚系数为假设值)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v12_aidc_pulse.png"), dpi=160)
    print("图已保存: figures/v12_aidc_pulse.png")


if __name__ == "__main__":
    main()
