# NUTS 实现复查与协方差信息保留

本次复查 Stan、NumPyro、BlackJAX 和 PyMC 的官方源码。采用固定的主分支快照，
不把开发分支功能描述成所有已发布版本的默认行为；提交号与文件 SHA-256 见
[源码记录](nuts-source-audit.json)。

## 对照结果

| 实现 | 与本项目有关的做法 | 本次处理 |
|---|---|---|
| [Stan](https://github.com/stan-dev/stan/tree/c3db9b995dae2f2f9e785cc55c6e96ab1127cdab/src/stan/mcmc) | multinomial 树、广义掉头、全轨迹接受统计；分窗口协方差适应和更新后的步长搜索 | 现有树构造通过固定源码对照；Welford、正则项和 dual averaging 公式未发现方向或系数错误 |
| [NumPyro](https://github.com/pyro-ppl/numpyro/blob/1e42fd58e5d6d60f8a767915c524edb787f2e81b/numpyro/infer/hmc.py) | 可分别指定预热/正式树深；启发式步长搜索是独立选项，默认关闭 | 搜索不是必然提速；保留本项目独立开关。其掉头细节与 Stan 不完全相同，不能随意混合替换 |
| [BlackJAX](https://github.com/blackjax-devs/blackjax/blob/2d6a51c3b0f7b4a221ac7ba218be09b97f1df608/blackjax/adaptation/mass_matrix.py) | 可用伪计数保留上一窗口的 inverse mass，即位置协方差信息 | 本次实现 `metric_shrinkage`，重点验证第一窗口的信息丢失 |
| [PyMC](https://github.com/pymc-devs/pymc/blob/d1a761e84174cb5d5427e6f0c52352ffe42aa14e/pymc/step_methods/hmc/nuts.py) | 默认前 200 次调节使用较小的最大树深 8，正式阶段通常为 10 | 适合作为预热成本保护，尚未移植；它本身不能改善质量矩阵估计 |

BlackJAX 的 [迭代 NUTS](https://github.com/blackjax-devs/blackjax/blob/2d6a51c3b0f7b4a221ac7ba218be09b97f1df608/blackjax/mcmc/nuts.py)
也值得参考，但改成迭代控制流不会自动减少昂贵的目标评估次数。
本次不同时更换树构造、积分器、掉头判据或目标接受率，保证对照只考察质量适应。

本项目的步长搜索仍是有 50 次上限的独立实现：Stan 会在确定方向后重新试验初始步长，
本项目直接按方向增减；因此两者不保证 warmup 随机路径一致。
既有 21,000 次逐步对照只覆盖共享密度/积分器下的固定树内核，不是完整 Stan 预热复刻。

## 上一轮变慢的位置

上一轮现代内核＋搜索在第 100 次预热更新质量矩阵后突然变慢：

| 原初值 | 前 100 次评估数 | 更新后已完成转移 | 这些转移的评估数 |
|---|---:|---:|---:|
| chain 0 | 2,454 | 12 | 9,624 |
| chain 1 | 4,968 | 9 | 7,167 |

第一个慢窗口只收集 25 个 45 维状态，中心化矩阵秩实测均为 24。
样本协方差必然秩亏。原来的 `1e-3 I` 正则项令其可逆，但缺少已知的方向相关信息。
这使首次更新可以大幅偏离初始 Gauss–Newton 预条件。
这是与性能突变一致的机制解释，不能据此证明所有后段慢链都有相同原因。

用同一批窗口状态重新计算第一次更新，比较相对于初始协方差的广义特征值条件数：

| 原初值 | 原更新 | 保留 25 个伪样本的更新 |
|---|---:|---:|
| chain 0 | 6,647 | 10.49 |
| chain 1 | 19,580 | 28.93 |

这是更新相对于**初始预条件**的变化，不是未知真实后验的条件数，也不是实测加速倍数。
结果见 `results/nuts-metric-audit/first-update-analysis.json`。
超时运行的非进度文件可能尚未刷完缓冲；初始矩阵取自此前完整运行的对应初值文件，
没有读取超时目录中可能截断的矩阵。

## 实现

令窗口状态数为 n，样本协方差为 S，上一窗口位置协方差为 Cprev，伪计数为 k：

```text
Cnew = n/(n+5+k) * S + k/(n+5+k) * Cprev + 5/(n+5+k) * 1e-3 I
Mnew = inverse(Cnew)
```

这里混合的是**位置协方差 / inverse mass**，不是动量质量矩阵 M。
第一窗口的 Cprev 来自用户提供或 Gauss–Newton 初始 M 的逆；以后使用上一窗口的 Cnew。
对角适应只保留对角元素。窗口、搜索和冻结规则沿用已有实现。

在 `[nuts]` 中可选设置：

```toml
kernel = "multinomial"
find_initial_step_size = true
metric_adaptation = "dense"
metric_shrinkage = 25.0
```

`metric_shrinkage` 默认 0，保持既有随机路径。只允许有限非负值，正值要求开启质量适应。
k=25 在第一个 n=25 的窗口给上一窗口约 45.5% 权重；随后随着窗口增大，权重减小。
本轮只试验这个预先指定的值，没有扫描后择优。
这是对公开数学公式的独立实现，不增加 BlackJAX、JAX 或 Python 运行时依赖。
它只改变 warmup 的预条件估计，数据、先验和目标密度没有变化；正式阶段继续冻结适应。

验证覆盖秩亏窗口、位置/动量协方差约定、三个 seed 的相关正态分布矩、
保留初始矩阵、步长搜索日程、正式阶段冻结、增大采样数的前缀一致性及 CLI 参数检查。

## 限时试验协议

沿用上一轮两个原初值、seed 20261023、300 预热＋100 正式状态、目标接受率 0.99、
start Gauss–Newton、dense 适应、最大深度 10 与每链 120 秒上限，仅增加 `metric_shrinkage=25`。
两条数值进程并行。旧 k=0 对照来自完整保存的上轮记录，不追加重复长跑；
因此时间比较不是同批次配对基准，应优先看评估数和采样质量。
预期新旧前 100 次转移逐字段相同，首次矩阵更新后才开始不同。
不以短测替代完整后验验收，也不把此前第 450–950 次预热的慢链问题标为已解决。

```sh
python scripts/benchmark_nuts_warmup.py --exe build/dispweave \
  --reference results/crust-nuts-final --output results/nuts-metric-pilot-new \
  --modern-only --metric-shrinkage 25
```

## 实测结果与结论

| 原初值 | k=0：120 秒完成预热次数 | k=25：120 秒完成预热次数 | 进度比 |
|---|---:|---:|---:|
| chain 0 | 112 | 266 | 2.38 |
| chain 1 | 109 | 172 | 1.58 |

前 100 次转移逐字段一致，确认初始模型、目标、搜索和随机流在首次质量更新前相同。
按相同转移数比较，chain 0 的前 112 次已完成转移评估数从 12,078 降到 2,692；
chain 1 的前 109 次从 12,135 降到 5,513。首次更新之后两条路径不同，
这些数字不能解释为相同后验位置上的效率或有效样本数比较。

新两链均在 120 秒超时，未完成 300 次预热，没有正式状态，不能计算正式 ESS/秒或宣称
完整后验验收。新 chain 0/1 的预热转移发散分别为 5/2，数值失败为 2/1；
转移数不同，不能据这些绝对次数比较失败率。全部失败调用另保存在原日志中。
因此本次结果是**首次窗口造成的性能问题有所缓解**，并非稳定性和全程效率都已改善。

新增选项保持默认关闭，已有正式后验和本地网页保持原有结果。
后续优先项是针对窗口末端保留的状态与矩阵做局部诊断，再评估预热树深保护或更可靠的
协方差估计；改变目标接受率应另做对照，不与本次改动混合归因。

结果：`results/nuts-metric-pilot/{protocol,summary,comparison,compatibility}.json`。
Release CTest **76/76**，NUTS UBSan **4/4**。k=0 时 slice 和 multinomial 各三条链的
样本、轨迹、矩阵、更新、搜索与调用成本共 **42 个文件**与修改前冻结二进制逐字节一致。

随后完成 [第 150 次更新的固定状态核查](nuts-window150-audit.md)：48 个短转移显示
chain 0 主要是步长重启过冲，chain 1 同时出现局部矩阵估计恶化；
因此后续需要分别验证步长重启策略和矩阵更新可靠性。
