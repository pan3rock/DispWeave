# NUTS 现代内核与限时效率对照

在已验收的地壳后验之后增加可选优化，不改变数据、先验、参数边界或正演。
原来的 slice 内核仍为默认值，旧配置继续使用原来的随机数路径。

## 实现

参考 [Stan NUTS 固定源码](https://github.com/stan-dev/stan/blob/c3db9b995dae2f2f9e785cc55c6e96ab1127cdab/src/stan/mcmc/hmc/nuts/base_nuts.hpp)
增加 `kernel = "multinomial"`：按 Hamiltonian 权重选择候选，子树内用加权抽样，
树外扩展用 biased progressive transition；掉头判断采用累计动量和三组广义检查。
步长适应的接受统计覆盖本次访问的全部叶节点，包括最终未采用的子树。

参考 [Stan 步长初始化](https://github.com/stan-dev/stan/blob/c3db9b995dae2f2f9e785cc55c6e96ab1127cdab/src/stan/mcmc/hmc/base_hmc.hpp)
和 [稠密质量矩阵适应](https://github.com/stan-dev/stan/blob/c3db9b995dae2f2f9e785cc55c6e96ab1127cdab/src/stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp)，
增加 `find_initial_step_size = true`：warmup 开始及每次质量矩阵更新后重新搜索步长，
随后重启 dual averaging。搜索使用约 0.8 的单步接受率阈值，以 2 倍增减步长，
每次刷新动量、最多 50 次探测，不能找到区间则明确失败。这是有上限的独立实现，
不是 Stan 搜索过程的逐随机数复刻。正式采样期间冻结步长和质量矩阵。

在已有 `[nuts]` 配置中添加：

```toml
kernel = "multinomial"
find_initial_step_size = true
```

需要 `warmup > 0` 且 `adapt_step_size = true`；质量矩阵仍由原配置控制。
两项开关独立，可以单独比较候选/掉头算法与步长搜索的效果。

`chain_N_step_size_searches.txt` 记录每次搜索发生的 warmup 位置、调用数和最终状态；
`chain_N_cost.txt` 满足 `target_evaluations = 1 + transition_evaluations + step_search_evaluations`。
初始合法性检查及可选 Gauss–Newton 初始质量矩阵的调用另外列出。
因此不能再仅用 trace 的累计调用数定位 `failed_evaluations`：还需加入初始化和穿插的搜索调用。
这些是目标密度/梯度调用数；单次调用内部的模式支持检查等正演工作不另计为一次采样器调用。

## 验证范围

- 对解析正态、相关正态、香蕉形分布、半正态及圆盘硬边界进行统计检查。
  香蕉形测试两种内核统一使用目标接受率 0.99；0.95 时现代内核出现发散，
  因此没有把所有后验在固定接受率下都无发散当作算法保证。
- 从极小和极大初始步长出发，检查搜索与质量更新日程、全部调用成本、
  正式阶段冻结、相同 seed 重放、增加 draw 数不改变已有前缀、无法搜索时明确失败。
- 直接编译未修改的 Stan `base_nuts.hpp`：7 类目标/步长情形 × 3 个 seed × 1000 次转移，
  对照候选状态、梯度调用数和接受统计。测试适配层共用本项目的密度、
  积分器和质量矩阵；这项证据限于树构造和候选选择，不代表完整 Stan 独立端到端验算。
- 已知物理边界仍单独记为 support rejection；真正的数值失败保持 failure/divergence。
  树深记尝试过的最大深度，与 Stan 的诊断计数约定略有差别。

上游源码、固定哈希及 BSD 许可见 [第三方记录](../third_party/stan-nuts-reference/README.md)。

最终 Release CTest **75/75**，UBSan 下 `nuts_core`、`nuts_multinomial`、
`nuts_stan_reference` **3/3**。与修改前冻结二进制对照，三层案例三条链的
Vs、z、trace、mass、metric_updates 共 15 个文件逐字节一致，确认旧配置的采样路径保持不变。
Stan 逐步对照覆盖 21,000 次转移，包括强相关、硬边界、数值失败及过大步长导致的发散。

## 地壳效率试验

`scripts/benchmark_nuts_modern.py` 在启动前写出协议与输入/可执行文件 SHA-256。
两种内核共用已验收运行的 chain 0 最终质量矩阵及步长，分别从原四条链的末状态开始，
无 warmup，各保留 200 个状态，固定 seed 20261024；两条数值进程并行，单链上限 120 秒。
数据、125 个观测、45 个参数、0.1 km/s 参考先验及物理边界保持不变。
记录全部声明的链；超时或不完整则不汇总成完整比较，不丢弃慢链。

这是从已有后验状态出发的**内核效率短试验**，不能评估原始分散初值的 warmup，
也不能替代此前 4 × 2000 的正式后验验收。ESS/秒使用各链耗时之和，
ESS/调用数包含初始化和预检查。短链 ESS 和时间比值只作为探索性证据。
原网页继续展示已有正式结果。

本次固定协议结果保存在 `results/nuts-modern-kernel-pilot/`；八条链均完成，无发散、
数值失败、物理支持拒绝或树深超限。

| 指标 | slice | multinomial |
|---|---:|---:|
| 四链合计状态数 | 800 | 800 |
| 四链累计耗时（秒） | 242.04 | 248.92 |
| 目标评估次数（含初始化/预检查） | 24,712 | 25,288 |
| 最低 bulk ESS | 374.00 | 405.26 |
| 最低 tail ESS | 359.07 | 398.33 |
| 最低 bulk ESS / 累计秒 | 1.545 | 1.628 |
| 最低 bulk ESS / 千次评估 | 15.135 | 16.026 |
| 最大 R-hat | 1.0222 | 1.0287 |

现代内核的最低 bulk ESS/秒约高 **5.4%**，ESS/调用数约高 **5.9%**；生成相同状态数
反而耗时约多 2.8%。不能把这次短链比值解释成显著或普遍提速。
两者相对已有正式后验的最大均值差分别为 2.23、2.18 个合并 MCSE；
这里复用了旧运行末状态，合并 MCSE 只作粗略一致性核对，不是独立统计检验。
两者最大 R-hat 均超过 1.01，故这批短链不替代正式后验，也不标记新的后验验收通过。
步长搜索的完整地壳 warmup 收益尚未测量；尤其不能宣称此前慢链的问题已解决。

复现命令（输出目录必须不存在）：

```sh
python scripts/benchmark_nuts_modern.py --exe build/dispweave \
  --reference results/crust-nuts-final --output results/nuts-modern-kernel-pilot-new
```

该次比较使用同一冻结二进制运行两种内核，不含同时运行的测试/编译任务。
输入协议和成本详见 `protocol.json`、`summary.json` 及各链 `*-cost.json`。
