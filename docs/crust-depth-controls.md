# 地壳深部参数化对照

2026-10-23。接续 [近邻根修复](clustered-root-repair.md)，比较固定半空间和合并均匀深部。
本项是既有地壳失败的控制实验，不是自适应分层或模型选择。

## 控制条件与接口

使用已准备的 `results/root-repair/final-examples/crustmantle/`：同一份 125 点、0–3 阶观测，
σ=0.02 km/s，同一组四个初值的对应自由分量、同样的上下界、固定 Vp/密度，
隐式导数、精细 Dunkin 求根、最多 200 次 LM 迭代、投影梯度阈值 1e-5。
本对照没有加正则项，以单独考察深部参数变化；这不意味着实际反演应当没有正则化。

| 方案 | 物理层数（含半空间） | 自由 Vs 数 | 半空间 |
|---|---:|---:|---|
| A_free | 44 | 44 | 320 km 起，Vs 自由 |
| B_fixed_halfspace | 44 | 43 | 320 km 起，Vs 固定 4.699 km/s |
| C_merged_tail | 34 | 33 | 100 km 起，Vs 固定 4.699 km/s |

原参考模型 100 km 以下 Vs、Vp、ρ 完全相同，C 合并的是这个均匀段；该参考模型的
125 个正演结果在合并前后最大差小于 1e-12 km/s。反演过程中 C 不再允许该深部自由变化。
固定值来自已提供模型，是本次合成对照条件，不是数据估计结果或通用地壳设置。

新增 `[inversion] fixed_vs_layers = [43]`，层号从 0 开始。
固定值为 `model.vs` 对应项，所有 `starts_vs` 必须一致；重复、越界、全固定或不匹配初值报错。
优化在自由参数子空间中进行，不使用极窄边界模拟固定；输出仍为完整模型及核函数。
投影梯度、步长与触界数只统计自由参数，`metadata.txt` 记录固定层与自由维度。
目前此选项用于 `invert/synthetic`；HMC/NUTS 显式拒绝它，避免静默忽略。

`forward --root-accuracy refined` 暴露已有精细容限，使外部 FD 与 LM 使用相同精度；
默认仍为 standard，候选区间和根枚举算法不变。

## 实际结果

A 组 `runs.txt` 与修复后的既有基线逐字一致。下表均是末次模型，**没有一次通过收敛标准**。

| 方案 | 运行 | 终止原因 | cost | 相速度 RMS km/s | 投影梯度 | h=1e-6 FD |
|---|---:|---|---:|---:|---:|---|
| A | 0 | 停滞 | 307.466817 | 0.044360 | 0.999947 | 未通过 |
| A | 1 | 迭代到限 | 1306.010037 | 0.091425 | 0.970648 | 未通过 |
| A | 2 | 迭代到限 | 40.782406 | 0.016156 | 0.541329 | 未通过，细步长改善 |
| A | 3 | 初值未解析 | — | — | — | 未运行 |
| B | 0 | 停滞 | 325.777879 | 0.045662 | 0.998615 | 未通过 |
| B | 1 | 初值未解析 | — | — | — | 未运行 |
| B | 2 | 迭代到限 | 37.778226 | 0.015549 | 0.985118 | 通过 |
| B | 3 | 停滞 | 40.801202 | 0.016159 | 0.588306 | 未通过 |
| C | 0 | 迭代到限 | 752.610348 | 0.069402 | 0.116486 | 通过 |
| C | 1 | 初值未解析 | — | — | — | 未运行 |
| C | 2 | 迭代到限 | 38.779894 | 0.015754 | 0.680080 | 通过 |
| C | 3 | 迭代到限 | 38.859264 | 0.015770 | 1.000000 | 通过 |

FD 对全部自由 Vs 做 log 扰动 h=1e-5、1e-6，检查所有观测行；容限为
`2e-6 + 1e-3 max(abs(FD), abs(implicit))`。若未通过，再对最差绝对差项检查 h=1e-7、1e-8。
每步完整预测、状态及 FD 数组保存为 NPZ；未解析点不作为通过项剔除。
FD 允许在参数边界外做微小物理扰动，只用于检查物理映射，并非可接受 LM 步。

- A0、B0、B3 的细步长差分仍存在有限相速度跳变，导数随步长缩小约反比增长。
  B0 的最差扰动层为 38，B3 为 39，均不是已固定的半空间：固定末层不足以消除问题。
- A2 的最差项随步长减小逼近隐式导数；不能把所有 FD 超差都归为漏根。
- C 三个可计算终态全部通过局部 FD，其最大归一化误差分别为 0.00699、0.00183、0.00183。
  未发现本轮“归一化步长 <1e-8 且 cost 跳变 >1e-4”标记；A/B 标记数分别为 10/29。
  这不是整条路径或整个模型空间的完备性证明。
- C2/C3 的 RMS 约 0.0158，小于假设 σ，但自由梯度仍大，不能改记为收敛。
  对提供模型的 0–100 km 深度加权 Vs RMS 差仍约 0.800/0.784 km/s；低残差不等于结构恢复。

因此，合并均匀深部改善了本次终态附近的数值行为，但没有单独解决优化收敛和结构约束问题。
没有据此增加迭代预算、放宽终止条件或更改观测模态。

## QEDispInv 的固定层厚与本测试不同

核对本地 QEDispInv `ed1b5dd` 的 `src/main_inversion.cc`、`src/model.cc`、`src/problem.cc`：

- `rand_depth=false` 使用 `model_ref` 的深度，半空间 Vs 仍属于优化向量。
- `add_last_layer=true`：若基阶最大观测波长的一半超过 `zmax` 且不小于末层顶深，
  在该深度追加半空间；新层参考 Vs 为 `1.01*cmax`，边界为 `[cmax,cmax+vs_width]`，
  其中 `cmax=max(最大基阶观测相速度, 原末层参考 Vs)`。不是固定半空间速度，也不是截断深度。
- 当前地壳配置的 `model_ref=mref.txt` 有 45 层，末层始于 208.925 km；基阶半最大波长约
  208.925 km。若改为固定层厚，当前浮点数比较不再追加层，半空间 Vs 参考值为 4.59635，
  `vs_width=1.2` 对应普通边界约 [3.99635,5.19635] km/s。
- 此示例配置 `vs2model=brocher05`，Vp/密度随 Vs 更新。使用 L-BFGS-B，并对相对参考模型
  的速度扰动作相邻差分正则化，包含最后两个参数之间的差分。
- 本次 DispWeave 使用的是 `model_data.txt` 的 44 层/320 km 分层，半空间原边界
  [3.05435,6.34365] km/s，固定 Vp/密度，无正则项，使用有界 LM。
- QEDispInv 这版 `update_fitness` 会跳过 NaN 预测并按有效点数归一化；DispWeave 有任何
  观测未解析则判整次评估失败。因此“运行结束”也不是同一个成功定义。

本轮不是复现 QEDispInv 的整套反演。固定层厚不等于固定半空间 Vs；去掉模型选择准则
也不等于去掉正则化。后续显式正则化与经验物性关系需独立验证，并保留本轮控制结果。

## 保留的正演失败与软件验收

新增 `tests/fixtures/rayleigh/halfspace-grid-model.txt` 与 `benchmarks/halfspace_root_probe.py`。
在 f=0.2727 Hz、末层 Vs 乘 exp(-1e-5) 时仍漏掉独立验证的
`4.250427445383965`、`4.250444859366714 km/s`，返回更高模态。
全局 P/SV 边界矩阵再次验证这两个根，最小/最大奇异值比分别约 3.3e-14、1.6e-13。
该诊断当前明确返回 **2**，没有作为预期失败而伪装成 CTest 通过。

新增固定坐标的解析条件最优解、非零固定方向梯度、非法层号、完整结果维度、
CLI 固定值/初值一致性、控制组输入保持和采样器拒绝未支持选项的检查。
Release CTest **64/64**，30.02 s；UBSan LM/CLI 定向 **7/7**，3.52 s；第三方哈希 **449/449**。
构建/测试期间存在并发任务，本轮不据墙钟时间比较方案效率。

## 复现与输出

```bash
cmake -S . -B build-depth-control -DCMAKE_BUILD_TYPE=Release
cmake --build build-depth-control -j2
ctest --test-dir build-depth-control --output-on-failure
python scripts/benchmark_crust_depth.py \
  --case results/root-repair/final-examples/crustmantle \
  --exe build-depth-control/dispweave --output results/crust-depth-new
MPLCONFIGDIR=/tmp/dw-mpl python scripts/benchmark_crust_depth.py \
  --output results/crust-depth-new --plot-only
python benchmarks/halfspace_root_probe.py \
  --exe build-depth-control/dispweave --output results/crust-root-new
# 上一行当前返回 2；输出完整保留。独立检查漏掉的两个根：
OPENBLAS_NUM_THREADS=1 python benchmarks/rayleigh_boundary_reference.py \
  results/crust-root-new/-1-model.txt --frequency .2727 \
  --bracket 4.25042 4.250435 --bracket 4.250435 4.25045 \
  --output results/crust-root-new/independent.json
```

首次准备外部案例的方法见 [原应用测试](qedisp-examples-validation.md)。脚本拒绝覆盖旧实验目录。
本次结果为 `results/crust-depth-control/`，含摘要、输入、12 次运行、FD、PNG/PDF 图；
已知失败及独立验证在 `results/crust-depth-root-regression/`。
