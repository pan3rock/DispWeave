# 冻结相关联合提议的独立复验

完成一次事先固定种子的同预算复验，没有追加运行、重新训练或调参。
**原第 0 链偏向 K=2 的现象没有重复，相关联合提议的收益再次出现；
但两次运行都仍有一个深度点未达标，完整后验尚未验收。**
建议结束这一轮通用采样器调参，保留候选和限制，不为了越过单个 R-hat 门槛反复换种子。

[两次实验对照图](../results/rj-pt-correlated-replication/replication-comparison.png) ·
[完整对照统计](../results/rj-pt-correlated-replication/replication-comparison.json) ·
[本次分段和条件分布](../results/rj-pt-correlated-replication/remaining-diagnosis/diagnosis.png) ·
[此前定位](rj-remaining-diagnosis.md)

## 冻结了哪些条件

复验目录 `results/rj-pt-correlated-replication`。
原种子 `20261025`，本次种子 **`20261026`**，在采样前写入 protocol。
对解析后的完整配置进行对比，唯一差异为 `rjmcmc.seed`。

- 复制原实验的冻结可执行文件，二进制 hash 相同；没有重新编译采样器。
- 观测数据、协方差、目标模型、先验和四个困难起点不变。
- 原先按 K 拟合的全部 Gaussian 分量及权重、协方差逐项不变；没有新拟合。
- 仍为四梯×四温度 1/2/4/8，每副本 2500 预热＋25000 正式，共 **440000 次本地提议**。
- 联合提议占全部动作约 10%，交换间隔及局部步幅不变。
- 保留全部正式状态和拒绝重复状态，不选择较好的链、不丢弃早段、不合并为一个“通过”的结果。

训练元数据保留旧训练文件 hash 和成本；其中配置绑定 hash 更新为新种子配置，
另外保留原配置 hash，并明确标注没有重新拟合。两次运行共用冻结 q，属于条件于同一 q 的
不同随机流复验，不是对重新训练提议过程或不同合成数据集的复验。

## 两次结果并列

| 指标 | 原实验 20261025 | 复验 20261026 |
|---|---:|---:|
| K=2 四链占比 | 53.10 / 40.56 / 37.25 / 38.63% | 42.48 / 43.18 / 40.33 / 40.99% |
| K=2 R-hat | 1.01364 | 1.00307 |
| K=2 bulk ESS | 910.4 | 1068.3 |
| 已访问 K=2/3/4 全部诊断通过 | 否 | 是 |
| 深度诊断通过点数 | 30/31 | 30/31 |
| 深度最大 R-hat | 1.01300 | 1.01098 |
| 未通过点深度 | 1.667 km | 1.500 km |
| 深度最低 bulk / tail ESS | 968.5 / 715.8 | 887.7 / 564.6 |
| 1.667 km 的 R-hat | 1.01300 | 1.00425 |
| 联合提议接受 / 尝试 | 8353 / 40088 | 8184 / 40012 |
| 联合提议接受率 | 20.84% | 20.45% |
| 正式期全温度失败 / 评估 | 14244 / 357833 | 13616 / 354992 |
| 正式期正演失败率 | 3.98% | 3.84% |
| numerical_failure | 0 | 0 |
| 总目标评估次数 | 393653 | 391005 |

复验 K=3/4 的 R-hat 分别为 1.00278/1.00129；所有 31 个深度点 bulk/tail ESS 均超过 400。
1.5 km 点的 R-hat=1.0109756 仍大于预定 1.01，不向下取整宣称通过。
四梯全部温度连通，每副本正式期 61–75 次温度往返。
两次的 K=2/3/4 总体占比分别为 42.384/34.897/22.719% 和 41.743/33.227/25.030%；
这些仍是未完全验收的观测占比，不能取代后验概率的可靠性判断。

## 哪些差异重复，哪些没有

原第 0 链约 53% 的 K=2 占比没有重复。本次四链占比集中在 40.3%–43.2%，
不支持“第 0 链起点必然导致 K=2 偏高”的判断。
这也不是证明第一轮异常完全由有限链长造成；每次随机轨迹仍可能访问不同结构。

在此前定位的 1.667 km 处，按相同 K 分组后：

| 指标（km/s） | 原实验 | 复验 |
|---|---:|---:|
| K=2 条件速度均值的四链跨度 | 0.00327 | 0.00300 |
| K=3 条件速度均值的四链跨度 | 0.13679 | 0.06218 |
| K=4 条件速度均值的四链跨度 | 0.04753 | 0.07914 |

K=2 内部相近的现象再次出现；K=3 的差异缩小，但 K=4 没有同步改善。
因此不能说所有条件分布均已一致。条件统计保留重复状态，只用于描述，
没有将按 K 过滤后的不连续时间序列拿去做 R-hat。
本次第 0 链与其他链的该深度均值差为 0.03085 km/s，其中层数占比项约 0.00204、
条件均值项约 0.02881；这是均值恒等式分解，不是收敛证明。

**判断：这轮改进值得保留，暂不继续增加通用提议机制。**
复验没有重现某个固定链必然卡住的模式；当前预算下仍有主界面附近的残余差异。
不再自动加长链、扫描比例或尝试第三个种子。若后续应用要求精确层数概率或完整后验验收，
再单独规定验证预算，并处理 birth/death 效率和未解析正演；不能直接拿本轮当作已完成认证。
K=1 两次均未访问，不能认证概率精确为零；正演失败继续保留为目标支持范围的限制。

## 成本与检查

本次采样及输出约 28.47 秒，不作严格墙钟加速结论。
新目标评估 **391005** 次；两次新实验合计 784658 次，
加共享的旧训练样本生成成本 400546 次，总计 **1185204** 次。
训练成本只计一次，旧训练样本未并入后验。
独立核查另做 **64 个模型的正演**（48 正式＋16 预热末），不混入采样成本。

检查全部通过：

- 配置仅种子变化；冻结二进制、数据、协方差和真值完全相同；protocol 输入 hash 保持不变。
- 本地状态、交换排列、计数、失败记录及全部 K/深度诊断核对；94666 个已接受正式转移的接受比重算。
- 40016 个含初值预热状态核查，9352 个接受、30648 个拒绝更新核对。
- 12 个旧经验分量独立重算，完整 Gaussian 混合 q 比最大差 2.85e−14。
- 64 个模型似然重算最大差小于 7.41e−11；这验证缓存，不证明全局根搜索完整。
- 分段、条件分布及对照脚本运行成功，Python 语法与文件差异检查通过。

本次没有修改 C++ 内核，因此没有将此前 80 项测试表述为本次重新运行。

复现命令会创建新目录；采样脚本只运行一次，后续命令只分析或审计：

```sh
MPLCONFIGDIR=/tmp/dispweave-mpl python scripts/replicate_rj_correlated.py --output results/rj-pt-correlated-replication-new
MPLCONFIGDIR=/tmp/dispweave-mpl python scripts/analyze_rj_remaining.py results/rj-pt-correlated-replication-new --output results/rj-pt-correlated-replication-new/remaining-diagnosis
python scripts/audit_rj_correlated.py results/rj-pt-correlated-replication-new
python scripts/audit_rj_pt_transdim.py results/rj-pt-correlated-replication-new
MPLCONFIGDIR=/tmp/dispweave-mpl python scripts/audit_rj_pt_warmup.py results/rj-pt-correlated-replication-new
MPLCONFIGDIR=/tmp/dispweave-mpl python scripts/compare_rj_replication.py results/rj-pt-correlated results/rj-pt-correlated-replication-new
```
