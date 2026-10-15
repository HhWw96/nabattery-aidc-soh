# 钠电 × AIDC:AI 驱动的电池全生命周期运维(初焰·钠宙 S01 · 风象 08 天秤座)

> 用 AI 研究电池的状态、寿命与风险:SOH 预测、跨工况泛化、不确定性量化、冷启动适配与策略闭环。
> 1836 开源社区留痕帖:01 立项 / 02 中期 / 03 深入 / 04 收束(见 `posts/` 目录草稿)

## 核心结论(TL;DR)

1. **同工况 SOH 预测已可用**:分位数梯度提升,留一电芯 MAE 0.012–0.045 Ah,R²>0.94
2. **跨温度外推会「自信地错」**:4 ℃ 工况 MAE 0.489 Ah,90% 置信区间覆盖率 0%——保形校准也救不了分布漂移
3. **「前 10 循环」冷启动协议**:新工况接入后仅前 10 个循环的实测数据做在线偏差校正,4 ℃ 误差 0.299→0.131 Ah,零重训
4. **预测驱动自适应限流**:闭环仿真中寿命期总吞吐 +11.2%(概念验证,参数为假设)
5. **钠电适配**:锂电 4 ℃ 容量保持 64% vs 钠电 0 ℃ ~95%(文献)——低温鸿沟收窄,但锂电预训练模型需冷启动校准迁移

## 目录结构

```
src/01_download_parse.py   下载说明 + .mat 解析 → 逐循环表(数据获取见下)
src/02_eda.py              EDA:四工况容量衰减曲线
src/03_features.py         特征工程:IC(dQ/dV)稳健统计 + EIS 内阻 + 曲线统计
src/04_baseline_gbr.py     V0.0.1 基线:分位数梯度提升 + 留一电芯/跨工况验证
src/05_v02_loco_conformal.py  V0.0.2 留一工况 + split-conformal 保形校准
src/06_v03_cold_start.py   V0.0.3 冷启动适配(偏差校正 vs 混合重训)
src/07_v04_seq_models.py   V0.0.4 时序模型对比(GBR-window / MLP / LSTM)
src/08_v05_closed_loop_sodium.py  V0.0.5 策略闭环仿真 + 钠电适配分析
data/processed/            解析产物(csv)
results/                   指标表(csv)
figures/                   全部产出图(png)
VERSIONS.md                五轮迭代的完整记录与局限声明
```

## 复现

```bash
pip install -r requirements.txt
# 1. 下载 NASA PCoE 数据集(约 160 MB,19 只电芯 .mat 文件)
#    镜像: https://github.com/devhaaana/nasa-battery-aging-dataset (data/ 目录)
#    放入 data/raw/nasa/ 后:
python src/01_download_parse.py
python src/02_eda.py
python src/03_features.py
python src/04_baseline_gbr.py
python src/05_v02_loco_conformal.py
python src/06_v03_cold_start.py
python src/07_v04_seq_models.py   # 需要 torch(CPU 版即可)
python src/08_v05_closed_loop_sodium.py
```

## 数据来源与引用

- NASA Ames Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset
  - B. Saha and K. Goebel (2007), "Battery Data Set", NASA Ames Prognostics Data Repository
  - 镜像:https://github.com/devhaaana/nasa-battery-aging-dataset
- 钠电低温数据:《钠离子和锂离子电池老化及产热特性对比》,《电池》2026, DOI: 10.19535/j.1001-1579.2026.01.004

## 局限声明

- 老化数据为 LiCoO₂ 18650 锂电;钠电结论为文献外推,未经钠电实测验证
- 闭环仿真为概念验证,策略收益系数是假设值,+11.2% 应理解为量级示意
- 时序模型仅测试小参数量配置,未做充分超参搜索

## License

MIT(代码)。NASA PCoE 数据遵循其原始发布条款。
