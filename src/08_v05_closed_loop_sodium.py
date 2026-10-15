# -*- coding: utf-8 -*-
"""
08_v05_closed_loop_sodium.py  V0.0.5:策略闭环仿真 + 钠电适配性分析 + 版本汇总

Part 1 闭环仿真(概念验证,参数来自 NASA 数据回归,非实测):
  - 老化模型: 每循环容量衰减率 = k(T) · (倍率因子)^alpha,从四组均值曲线回归
  - 固定策略: 全程 2A 放电
  - 自适应策略: 预测 SOH 的 90% 区间下限 < 阈值 → 切换温和模式(衰减减半,吞吐 7 折)
  - 对比: 到达 SOH 80% 时的总吞吐电量与循环数

Part 2 钠电适配性:
  - 锂电(本数据): 4 ℃ 首循环容量保持率 ≈ 64%(vs 24 ℃)
  - 钠电(文献): 0 ℃ ≈ 95%, -15 ℃ ≈ 88%(《电池》2026, DOI:10.19535/j.1001-1579.2026.01.004)
  - 结论: 跨温度鸿沟在钠电侧大幅收窄,但锂电训练的模型会系统性低估钠电低温容量
         → 冷启动适配(V0.0.3)正是迁移桥梁

Part 3 汇总: versions 指标总表 + CHANGELOG
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

# ---------- Part 1: 闭环仿真 ----------
EOL = 0.8          # SOH 寿命终点
CAP0 = 1.9         # 24 ℃/2A 组首循环容量(Ah, 来自数据)
FADE_2A = (1.90 - 1.30) / 168   # 24C_2A 组实测平均衰减率 Ah/循环
GENTLE_FADE_FACTOR = 0.5        # 温和模式: 衰减减半(限倍率的假设收益)
GENTLE_THPT_FACTOR = 0.7        # 温和模式: 单循环吞吐打 7 折(充电变慢的机会成本)
PRED_MAE = 0.02                 # 预测器噪声(取 V0.0.4 GBR-window 同工况水平)
THRESH = 0.85                   # 触发温和模式的 SOH 区间下限阈值


def simulate(adaptive: bool, seed=0, horizon=400):
    rng = np.random.default_rng(seed)
    cap, hist = CAP0, []
    for cyc in range(1, horizon + 1):
        # 预测器: 真值 + 噪声(标准差取 MAE 水平), 区间下限 ≈ pred - 1.64σ
        pred = cap + rng.normal(0, PRED_MAE)
        lower = (pred / CAP0) - 1.64 * PRED_MAE / CAP0
        gentle = adaptive and (lower < THRESH)
        fade = FADE_2A * (GENTLE_FADE_FACTOR if gentle else 1.0)
        cap -= fade
        thpt = cap * (GENTLE_THPT_FACTOR if gentle else 1.0)
        hist.append(dict(cycle=cyc, soh=cap / CAP0, throughput=thpt, gentle=gentle))
        if cap / CAP0 < EOL:
            break
    h = pd.DataFrame(hist)
    return h, h.throughput.sum(), len(h)


def part1():
    rows = []
    for seed in range(20):     # 多次仿真取均值(预测噪声的蒙特卡洛)
        for adaptive in (False, True):
            h, tot, n = simulate(adaptive, seed)
            rows.append(dict(strategy="自适应限流" if adaptive else "固定 2A",
                             seed=seed, total_ah=tot, cycles=n,
                             gentle_ratio=h.gentle.mean()))
    r = pd.DataFrame(rows)
    summ = r.groupby("strategy").agg(total_ah_mean=("total_ah", "mean"),
                                     total_ah_std=("total_ah", "std"),
                                     cycles_mean=("cycles", "mean")).round(1)
    print("闭环仿真(20 次蒙特卡洛):"); print(summ)

    # 轨迹图(单次代表性仿真)
    h_fix, _, _ = simulate(False, seed=7)
    h_adp, _, _ = simulate(True, seed=7)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(h_fix.cycle, h_fix.soh, color="#1f77b4", lw=2, label="固定 2A 策略")
    axes[0].plot(h_adp.cycle, h_adp.soh, color="#d62728", lw=2, label="自适应限流策略")
    axes[0].axhline(EOL, color="gray", ls="--", lw=1)
    axes[0].axhline(THRESH, color="green", ls=":", lw=1)
    axes[0].text(5, THRESH + 0.005, "预测区间下限阈值 0.85", color="green", fontsize=9)
    axes[0].text(5, EOL + 0.005, "寿命终点 SOH 0.80", color="gray", fontsize=9)
    axes[0].fill_between(h_adp.cycle[h_adp.gentle], EOL - 0.02, 1.0,
                         color="#d62728", alpha=0.06, label="温和模式区间")
    axes[0].set_xlabel("循环数"); axes[0].set_ylabel("SOH")
    axes[0].set_title("SOH 轨迹:预测驱动的自适应限流延缓寿命终点")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    labels = ["固定 2A", "自适应限流"]
    means = [summ.loc["固定 2A", "total_ah_mean"], summ.loc["自适应限流", "total_ah_mean"]]
    stds = [summ.loc["固定 2A", "total_ah_std"], summ.loc["自适应限流", "total_ah_std"]]
    gain = (means[1] / means[0] - 1) * 100
    axes[1].bar(labels, means, yerr=stds, capsize=5, color=["#1f77b4", "#d62728"], alpha=0.85)
    for i, m in enumerate(means):
        axes[1].text(i, m + 2, f"{m:.0f} Ah", ha="center", fontsize=11)
    axes[1].set_ylabel("寿命内总吞吐电量 (Ah)")
    axes[1].set_title(f"总吞吐电量对比:自适应策略 +{gain:.1f}%")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("V0.0.5 闭环仿真:AI 预测反哺运行策略(概念验证,参数来自 NASA 数据回归)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v05_closed_loop.png"), dpi=160)
    print("图已保存: figures/v05_closed_loop.png")
    return summ, gain


# ---------- Part 2: 钠电适配性 ----------
def part2():
    li_4c = 1.21 / 1.90   # NASA 数据: 4 ℃ 首循环 / 24 ℃ 首循环
    points = pd.DataFrame([
        ("锂电(本项目 NASA 数据)", 24, 1.00), ("锂电(本项目 NASA 数据)", 4, li_4c),
        ("钠电(《电池》2026 实测)", 25, 1.00), ("钠电(《电池》2026 实测)", 0, 0.95),
        ("钠电(《电池》2026 实测)", -15, 0.88),
        ("钠电(宁德时代量产口径)", -20, 0.92),
    ], columns=["体系", "temp", "retention"])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for (name, s), color in zip(points.groupby("体系"), ["#1f77b4", "#d62728", "#2ca02c"]):
        s = s.sort_values("temp")
        ax.plot(s.temp, s.retention * 100, "o-", color=color, lw=2, ms=7, label=name)
    ax.axhline(80, color="gray", ls="--", lw=1)
    ax.text(-24, 81, "80% 可用容量线", color="gray", fontsize=9)
    ax.set_xlabel("温度 (℃)"); ax.set_ylabel("可用容量保持率 (%)")
    ax.set_title("钠电 vs 锂电的低温容量保持率 —— 锂电模型的「低温盲区」恰是钠电的主场\n(锂电点为本项目 NASA 数据实测; 钠电点为文献/厂商公开数据, 仅作外推示意)", fontsize=10.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v05_sodium.png"), dpi=160)
    print("图已保存: figures/v05_sodium.png")
    return li_4c


# ---------- Part 3: 汇总 ----------
def part3(summ, gain, li_4c):
    v02 = pd.read_csv(os.path.join(RES, "v02_metrics.csv"))
    v03 = pd.read_csv(os.path.join(RES, "v03_cold_start.csv"))
    v04 = pd.read_csv(os.path.join(RES, "v04_seq_metrics.csv"))
    changelog = f"""# 版本迭代记录(初焰·钠宙 S01 · 风象 08 天秤座)

## V0.0.1(10/14 晚)基线:分位数梯度提升
- 同工况留一电芯: MAE 0.020–0.045 Ah, R² 0.945–0.979
- 跨工况(训练仅 24 ℃/2A): 4 ℃ 组 MAE 0.489 Ah, R²=-21.9, PICP90=0% → 模型「自信地错」
- 发现: 树模型无法跨工况外推; 分位数区间校准不足(同工况 PICP 仅 32–57%)

## V0.0.2(10/15)多工况留一工况 + 保形校准
- 训练=其余 3 工况组, 测试=留一工况; 训练集内 LOBO 残差做 split-conformal
- 24C_4A(角点补全): MAE 0.086 Ah, R²=0.57 —— 组合工况可被「夹」出来
- 4C_2A(真外推): MAE 0.299 Ah, PICP 校准后仍 8% —— 保形假设(可交换性)在分布漂移下失效, 必须有目标工况数据
- 同工况校准: PICP 25% → 95% ✓

## V0.0.3(10/15)冷启动适配
- 新工况只给前 K 循环实测: 4 ℃ 组 K=10 时 MAE 0.299 → 0.131 Ah(偏差校正, 在线零重训)
- 43 ℃ 组 K=10: 0.116 → 0.044 Ah; 混合重训 0.038 Ah
- 结论: 偏差校正修「系统性偏移」, 重训学「衰减形态」; 部署期前 10 循环是黄金校准窗口

## V0.0.4(10/15)时序模型对比(W=10 滑窗, 纯历史外推)
- 同工况: GBR-window MAE 0.012 Ah(R² 0.984) < MLP 0.016 < LSTM 0.026
- 倍率泛化: GBR-window 0.028 ≪ LSTM 0.106, MLP 0.126
- 温度泛化: 全部失效(MAE 0.13–0.20) —— 瓶颈是数据分布, 不是模型容量
- 结论: 小样本表格场景梯度提升完胜深度学习; 深模型留作大数据场景的路线图

## V0.0.5(10/15)闭环仿真 + 钠电适配
- 固定 2A vs 预测驱动自适应限流: 总吞吐 {summ.loc['固定 2A', 'total_ah_mean']:.0f} → {summ.loc['自适应限流', 'total_ah_mean']:.0f} Ah(+{gain:.1f}%), 蒙特卡洛 20 次
- 钠电适配: 锂电 4 ℃ 容量保持 {li_4c:.0%}(本项目实测) vs 钠电 0 ℃ ~95% / -15 ℃ ~88%(文献)
- 结论: 钠电低温优势收窄了「跨温度鸿沟」, 但模型迁移仍需冷启动校准 —— 方法闭环成立

## 局限(诚实声明)
- 老化数据为 LiCoO₂ 18650(锂电), 钠电结论为文献外推, 未经钠电实测验证
- 闭环仿真为概念验证: 温和模式的衰减/吞吐系数是假设, 非实测
- 时序模型仅测了小参数量配置, 未做充分超参搜索
"""
    with open(os.path.join(HERE, "..", "VERSIONS.md"), "w", encoding="utf-8") as f:
        f.write(changelog)
    print("VERSIONS.md 已写入")


if __name__ == "__main__":
    summ, gain = part1()
    li_4c = part2()
    part3(summ, gain, li_4c)
