# -*- coding: utf-8 -*-
"""
07_v04_seq_models.py  V0.0.4:时序模型对比 —— 纯历史序列外推(真·预测)

与 V0.0.1/0.0.2 的区别(诚实的任务定义):
  V0.0.1/0.0.2 用「同循环」的充电/阻抗特征回归容量 → 诊断式估计(nowcast)
  V0.0.4 只用「过去 10 个循环」的历史序列预测下一循环容量 → 外推式预测(forecast)
  为公平对比, GBR 也改用同样的滑窗历史输入(GBR-window)

模型: GBR-window / MLP-window / LSTM(torch, hidden=32)
场景: ① 同工况(24C_2A 内分电芯) ② 倍率泛化(测 24C_4A) ③ 温度泛化(测 4C_2A)
输出: results/v04_seq_metrics.csv, figures/v04_seq_compare.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(42); np.random.seed(42)

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, "..", "figures"); os.makedirs(FIG, exist_ok=True)

SEQ_FEATS = ["capacity_ah", "Re", "Rct", "dis_temp_mean", "cc_capacity_ah"]
W = 10                      # 滑窗长度
TARGET = "capacity_ah"


def build_windows(df):
    """按电芯构造滑窗样本: X=(W,F), y=下一循环容量"""
    X, y, meta = [], [], []
    for bid, s in df.groupby("battery"):
        s = s.sort_values("cycle").reset_index(drop=True)
        F = s[SEQ_FEATS].copy()
        for c in SEQ_FEATS:                    # 缺失值:电芯内前向填充,再用组均值兜底
            F[c] = F[c].ffill().bfill()
        F = F.fillna(df[SEQ_FEATS].mean())
        cap = F.values
        for i in range(W, len(s)):
            X.append(cap[i - W:i])
            y.append(s.loc[i, TARGET])
            meta.append((bid, s.loc[i, "cycle"], s.loc[i, "group"]))
    return np.array(X), np.array(y), pd.DataFrame(meta, columns=["battery", "cycle", "group"])


class LSTMReg(nn.Module):
    def __init__(self, n_feat, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


def train_lstm(Xtr, ytr, epochs=150, lr=1e-3):
    model = LSTMReg(Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    best, patience = None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward(); opt.step()
        if best is None or loss.item() < best - 1e-6:
            best, patience = loss.item(), 0
        else:
            patience += 1
            if patience >= 25:
                break
    model.eval()
    return model


def run_scenario(name, df, test_mask):
    X, y, meta = build_windows(df)
    te = test_mask(meta)
    tr = ~te
    Xtr, ytr, Xte, yte = X[tr.values], y[tr.values], X[te.values], y[te.values]

    # 标准化(按训练集)
    mu = Xtr.reshape(-1, Xtr.shape[2]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-8
    Xtr_n = (Xtr - mu) / sd; Xte_n = (Xte - mu) / sd
    ymu, ysd = ytr.mean(), ytr.std() + 1e-8

    out = {}
    # GBR-window
    gbr = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                        max_leaf_nodes=31, random_state=42)
    gbr.fit(Xtr_n.reshape(len(Xtr_n), -1), ytr)
    out["GBR-window"] = gbr.predict(Xte_n.reshape(len(Xte_n), -1))
    # MLP-window
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True,
                       random_state=42)
    mlp.fit(Xtr_n.reshape(len(Xtr_n), -1), (ytr - ymu) / ysd)
    out["MLP-window"] = mlp.predict(Xte_n.reshape(len(Xte_n), -1)) * ysd + ymu
    # LSTM
    lstm = train_lstm(Xtr_n, (ytr - ymu) / ysd)
    with torch.no_grad():
        out["LSTM"] = lstm(torch.tensor(Xte_n, dtype=torch.float32)).numpy() * ysd + ymu

    rows = []
    for mname, p in out.items():
        rows.append(dict(version="v0.0.4", scenario=name, model=mname,
                         MAE=mean_absolute_error(yte, p), R2=r2_score(yte, p), n=len(yte)))
        print(f"  {name} | {mname}: MAE={rows[-1]['MAE']:.4f} Ah, R²={rows[-1]['R2']:.3f}")
    return rows, meta[te].reset_index(drop=True), yte, out


def main():
    df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "feature_table.csv"))
    df = df[df.group != "4C_mixed"].reset_index(drop=True)

    scenarios = [
        ("同工况(测 B0007/B0018)", df, lambda m: m.battery.isin(["B0007", "B0018"])),
        ("倍率泛化(测 24C_4A)", df, lambda m: m.group == "24C_4A"),
        ("温度泛化(测 4C_2A)", df, lambda m: m.group == "4C_2A"),
    ]
    all_rows, plots = [], []
    for name, d, mask in scenarios:
        rows, meta_te, yte, out = run_scenario(name, d, mask)
        all_rows += rows
        plots.append((name, meta_te, yte, out))

    pd.DataFrame(all_rows).to_csv(os.path.join(RES, "v04_seq_metrics.csv"),
                                  index=False, encoding="utf-8-sig")

    # ---- 图:3 场景 × 3 模型 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    colors = {"GBR-window": "#1f77b4", "MLP-window": "#2ca02c", "LSTM": "#d62728"}
    for ax, (name, meta_te, yte, out) in zip(axes, plots):
        order = np.argsort(meta_te.cycle.values)
        cyc = meta_te.cycle.values[order]
        # 组均值曲线(避免多电芯重叠混乱)
        df_p = pd.DataFrame({"cycle": cyc, "y": yte[order]})
        for mname, p in out.items():
            df_p[mname] = p[order]
        gm = df_p.groupby("cycle").mean()
        ax.plot(gm.index, gm.y, "k-", lw=2.2, label="实测(均值)")
        for mname in out:
            ax.plot(gm.index, gm[mname], color=colors[mname], lw=1.4,
                    label=f"{mname} (MAE={mean_absolute_error(yte, out[mname]):.3f})")
        ax.set_title(name, fontsize=10.5)
        ax.set_xlabel("循环数"); ax.set_ylabel("容量 (Ah)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("V0.0.4:纯历史序列外推(W=10)三模型对比 — 时序模型 vs 梯度提升", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v04_seq_compare.png"), dpi=160)
    print("\n图已保存: figures/v04_seq_compare.png")


if __name__ == "__main__":
    main()
