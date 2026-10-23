# 物理 Vs 参考先验与地壳 NUTS 验证

2026-10-23。延续 [正则化敏感性扫描](crust-regularization-validation.md)，将固定参考模型的
偏差约束显式定义为概率先验，验证固定维度 NUTS 后验与预测。
接口、解析分布、三层示例和软件回归已完成。**45 参数地壳 NUTS 未通过验收**：
两种目标接受率均遇到正式期正演失败，负向扰动初值还出现严重预热受阻。
本项交付先验实现、复现实验和失败诊断，不将地壳条件区间列为已验证科学结果。

## 先验与测度

新增 C++ `bounded_gaussian_prior(mean, sigma, lower, upper, measure)`：各分量独立，
正态分布在包含其均值的有限盒内归一化。支持物理 Vs 或 log Vs 的明确测度。
本次命令行仅新增物理 Vs 的参考先验，中心固定取 `model.vs`。

令 v=Vs、μ=参考 Vs、s=参考标准差，a/b 是生成后的物理速度边界：

```text
Z_i = Φ((b_i - μ_i)/s_i) - Φ((a_i - μ_i)/s_i)
p(v) = Π_i 1[a_i ≤ v_i ≤ b_i] exp(-(v_i-μ_i)²/(2s_i²)) / (s_i sqrt(2π) Z_i)
d log p / dv_i = -(v_i - μ_i) / s_i²
```

归一化因子保留在密度中。均值位于盒内的契约允许用两个非负 erf 项求和计算 Z，
避免相近尾部 CDF 相减。均值不在盒内、非正/非有限标准差、错维、非法边界或无法解析的
归一化质量会显式拒绝；该 API 不声称支持均值位于截断区间外的任意尾部正态。

采样坐标保持 q=log(v)，再通过 logistic 变换映射到无约束 z。

```text
log π_z = log L(v) + log p(v) + Σ q_i + Σ log(dq_i/dz_i)
```

两个 Jacobian 都保留，Vp/ρ 由 Vs 确定，不新增独立概率参数或额外密度 Jacobian。
当所有 s=0.1 km/s 时，物理 Vs 密度的负对数先验与 λ=100 的参考阻尼仅差固定常数。
因此在相同物理模型和支持域下，物理 Vs 的 MAP 对应同一个惩罚目标；采样坐标的众数及
后验均值不要求等于 LM 解。参考标准差与观测 σ=0.02 km/s 是不同量。

## 配置、输出和兼容性

```toml
[inversion]
vs2model = "brocher05"
# 保留 model、速度边界和 starts_vs。

[inversion.regularization]
type = "none"
lambda = 0.0

[nuts] # HMC 的 [hmc] 同样支持
prior = "reference_gaussian_vs"
prior_sigma_vs = [0.1, 0.1, 0.1] # 每层一个值，含半空间，单位 km/s
metric_adaptation = "dense"
warmup = 1000
draws = 2000
max_depth = 10
step_size = 0.03
target_accept = 0.95
adapt_step_size = true
seed = 20261023
```

旧的 `uniform_vs` / `uniform_log_vs` 仍可用；这些先验若提供 `prior_sigma_vs` 会报错。
采样器继续拒绝有效 LM 正则项，不能把同一约束以惩罚和先验重复计入。
所有参考标准差必须显式给出，程序不会根据 LM 的 λ 自动生成先验。

新增 `prior_parameters.txt` 保存逐层中心、标准差、上下界；元数据记录先验测度与归一化约定。
模型、样本、预热标记、失败状态、质量矩阵和预测的既有格式保持不变。

可选 `chain_id_offset` 为非负整数，默认 0。独立进程运行某条链时可使用与串行多链调用
相同的 seed/chain-id 随机流；本地输出仍从 `chain_0` 编号，元数据标明偏移。
HMC/NUTS 工作流测试验证拆分运行的样本、z、轨迹、质量矩阵和适应记录与串行对应链逐字一致。
实验脚本最多同时执行四个数值采样进程，最终按链 ID 汇总；全部开发和验收由单代理完成。

可选 `record_failures=true` 保存 `chain_N_failed_evaluations.txt`，记录失败的目标调用编号、
状态及完整无约束 z，便于重建试探模型。编号从初始化调用 0 开始；它计数实际目标调用，
不是采样步号，也不能无条件与 leapfrog 评估次数等同。默认关闭，不改变随机流。
接口测试分别核对 HMC/NUTS 开启日志前后的完整状态和轨迹一致。
最终实现逐条刷新失败日志，以保留中断前已写出的试探点；实验旧二进制的缓冲日志可能在
进程中止时丢失末尾尚未刷新的记录，具体保留范围在下文单独说明。

## 独立数值与三层示例

- 正态先验用独立 Simpson 积分验证归一化，覆盖不对称区间、均值在边界、极窄区间和宽区间。
  物理 Vs 与 log Vs 测度分别验证，支持域外拒绝及非法输入单独检查。
- 物理先验密度差与 λ=100 参考阻尼独立核对；有限差分检查先验及两级坐标变换梯度。
  六种既有物性关系在偏离参考的位置核对完整后验梯度，包含 Brocher05 的密度链项。
- 一层半空间通过独立 Rayleigh 三次方程得到 c=k·Vs，将正态似然与先验相乘得到解析
  截断正态后验。三个随机种子各运行 600 步预热、12000 步正式 NUTS；均值/方差达到预声明
  误差标准，正式期无发散、无效计算或树深截断。该测试同时检验截断效应和测度变换。
- `examples/nuts/reference_gaussian.toml` 是三层 Brocher05 含噪示例。四条链各预热 1000、
  正式 2000；最大 R-hat=1.00160，最低 bulk ESS≈3305，最低 tail ESS≈3392，正式期无异常。
  这是接口与低维分布示例，不能替代下述地壳验收。

## 地壳实验的固定条件

直接读取上轮 `results/crust-regularization-scan/lambda-100/` 的参考配置、125 条观测和协方差。
45 个 Vs 全自由，界面固定、半空间顶深 208.925 km；Brocher05、原速度交集边界、观测顺序和
σ=0.02 km/s 保持不变。先验中心仍为 mref，所有标准差 0.1 km/s；本项不调整先验强度。

四个初值分别为参考整体 −0.2、+0.2 km/s，以及物理深度上的正弦/余弦 ±0.2 km/s 扰动，
周期 100 km。没有用 LM 终态替换链的初值。层数、界面、物性关系和观测误差均不采样。

初始质量矩阵由参考模型处的白化物理 Vs Jacobian 构造：

```text
M_initial = D [J_vsᵀ C⁻¹ J_vs + diag(1/s_i²)] D,    D = diag(dVs/dz)
```

它是无约束 z 中的动量协方差，只用于预条件化，不是最终后验协方差或新的目标项。
参考 Jacobian 用 refined 求根计算；NUTS/预测沿用已实现的标准精度正演。
生产配置选择 dense 质量矩阵适应，正式期冻结矩阵和步长；没有修改 NUTS 核心算法。

单链短试运行保留于 `results/crust-nuts-pilot/`（300 预热、100 正式）及
`results/crust-nuts-pilot-fixed/`（100 预热、100 正式，固定初始质量矩阵）。两者正式期均无
发散、无效计算或树深截断，但都只有一条短链，不计入后验验收。
适应矩阵组正式期共 1396 次梯度评估，固定矩阵组 3100 次；二者预热预算不同，
这仅支持本次选择适应矩阵，不作为普遍效率结论。
两组预热均有 22 次正演失败，适应组另有一次有效计算的能量发散；所有失败保留，
未将正演未解析解释成模型先验概率为零。

正式设置在启动前写入配置和 provenance：四条链，每链预热 1000、正式 2000，
目标接受率 0.95、最大树深 10、初始步长 0.1、seed=20261023，链 ID=0–3。
保留全部正式状态，包括拒绝后的重复状态，不做稀释。

验收同时要求：

- 全部 45 个 Vs、三个深度区间平均 Vs 及参考二次项的最大 R-hat≤1.01，bulk/tail ESS≥400。
- 正式期无发散、无效计算、树深截断或整链未移动，实际步长和质量矩阵冻结。
- 四条链各固定选第 1000 个零基正式状态，对全自由参数做 h=1e-5、1e-6 的 refined FD。
  标准/精化预测差不超过观测 σ 的 1e-4，二者 Jacobian 用既有 FD 误差尺度比较。
  这项选择和阈值在四条链完成前保存于 `posterior-audit-plan.json`。
- 每链等间隔选 200 个状态做后验预测，完整保留潜在曲线、带观测噪声的复制数据及失败。

## 地壳实际结果

### 首轮 0.95：未通过，原始证据保留

`results/crust-nuts-reference/` 已完成链 1、2、3，各保留 2000 个正式状态。
链 1、3 无正式期异常，链 2 在零基正式步 1108 出现一次正演失败和发散。
长任务执行会话随后以 143 结束，链 0 未落盘；未对缺链组报告完整多链验收。
由于已出现正式期失败，该组不能通过数值门槛。

三条完整链各自的第 1000 个正式状态仍按原计划做导数复核：全 45 参数、两个 FD 步长
均通过，最大缩放误差 0.001743（门槛 1）；标准/refined 相速度差最多为观测 σ 的
2.49e-11，Jacobian 缩放差最多 7.48e-7。结果保存在 `completed-chain-audit/`，
这里只支持这三个局部点，不消除失败试探点，也不补足缺失链。

用相同配置、种子与链 ID 重放链 2，保留 1109 个正式状态，Vs/z/trace 与原运行前缀逐字一致。
失败目标调用编号 31754，对应预热后第 1108 步，完整证据在
`results/crust-nuts-failure-replay/replay-verification.json`。
失败试探模型的半空间 Vs=4.427261789 km/s；观测行 74（零基）、f=0.1612 Hz、mode=2
在标准和 refined 精度下均为 `missing_or_unresolved`，仅前两个模态被解析。

仅为诊断，将该模型半空间 Vs 改为 4.45/4.48/4.50/4.55/4.60 km/s 并同步更新 Brocher05
物性后，同一频率的 mode=2 出现在约 4.439194324 km/s。该速度高于原试探点半空间 Vs，
结果与模态截止有关的解释相符，但不构成原模型根不存在的独立证明。
没有据此抬高采样速度下限、删除观测、改变先验，或把未解析状态改为概率零。

### 补充 0.99：同一目标与随机流

只提高目标接受率至 0.99，以减小试探步长；四个初值、seed、链 ID、先验、协方差、
正式预算和验收阈值均保持原定义。输出在 `results/crust-nuts-conservative/`。
这组首次执行也随会话结束而中断，未保存正式链；恢复时完整保留中断目录并从各链原初值
重跑，没有拼接不同随机流或把未完成轨迹纳入统计。采样可执行文件按 SHA256 固定并保存副本。

| 原链 ID | 初值 | 完成正式状态 | 正式发散/无效 | 正式树深截断 | 正式平均树深 | 冻结步长 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 整体 −0.2 | 0 | 不适用 | 不适用 | 不适用 | 未冻结 |
| 1 | 整体 +0.2 | 2000 | 1 / 1 | 0 | 4.9810 | 0.145736 |
| 2 | 正弦扰动 | 2000 | 3 / 3 | 0 | 4.9885 | 0.145677 |
| 3 | 余弦扰动 | 2000 | 1 / 1 | 0 | 4.9905 | 0.141635 |

链 1 预热有一次 `numerical_failure` 和一次正演失败，链 2 有一次 `numerical_failure`
及一次有效计算的能量发散，链 3 有一次正演失败；
正式期的 5 次失败均重放为观测行 74、0.1612 Hz、mode=2 未解析，标准/refined 结果一致。
对应零基正式步为链 1 的 525、链 2 的 142/175/1314、链 3 的 1731。
试探点半空间 Vs 为 4.4387–4.4496 km/s，完整模型和检查在 `failed-trial-audit/`。

链 0 持续计算约 25.6 分钟，仍未完成预热。只读调试快照核对了实际结果矩阵尺寸及
已完成转移向量：199/1000 步预热、0 个正式状态，111 步含无效计算/发散，77 步树深截断，
累计 124898 次评估，平均树深 9.010。最后一步步长约 0.00861、树深 10、1023 次评估。
在其余三条链已违反正式期零失败门槛后终止该受阻链，返回信号 15；没有修改进程内存、
采样目标或 RNG。`chain-0-progress.json`、`chain-0-warmup-prefix.txt` 保留快照，
它们不是完整正式轨迹，不能当作可恢复检查点。旧二进制只保留已刷新的失败试探日志。
其中 107 个完整的 `forward_failure` 试探点全部完成标准/refined 重放，均包含观测行 74
未解析；一个点还同时缺失另外五行。`chain-0-failed-trial-audit/` 保留全部输入和结果。

所以整组 `acceptance.json` 明确为 false，原四链没有合并成通过结果。
只对已完成的原链 1、2、3 建立 `completed-chain-diagnostics/`，本地编号 0–2，
配置注明 `chain_id_offset=1`，provenance 保留原链映射及缺失初值区域。
该三链子集全部参数与派生量的最大 R-hat=1.00317、最低 bulk ESS≈2593、tail ESS≈3238；
**这些数值不能替代原四链混合验收，也不能排除遗漏区域的后验质量。**

三条完整链的指定中间状态均通过两个步长、45 参数的 FD 检查，最大缩放误差 0.001777；
标准/refined 预测差最多为 σ 的 2.49e-11，Jacobian 缩放差最多 1.05e-6。
每链等间隔 200 个状态共 600 次预测均有效，PPC 超越比例约 0.5133。
各模态后验均值曲线的 RMS 和逐点复制数据 95% 区间包含数如下：

| 模态 | 点数 | RMS km/s | 包含观测数 |
|---:|---:|---:|---:|
| 0 | 43 | 0.010965 | 42 |
| 1 | 31 | 0.023663 | 28 |
| 2 | 31 | 0.020568 | 31 |
| 3 | 20 | 0.022349 | 19 |

先验/样本区间、轨迹、相关矩阵和预测图均输出 PNG/PDF，并标记
`VALIDATION FAILED (3/4 chains): diagnostic distributions only`。
这些图展示已完成链所访问区域中的分布及拟合，不能据此发布地壳后验不确定性结论。

### 下一项的范围

先对保存的失败模型进行独立模态存在性/截止与根枚举检查，区分物理缺模态和数值漏根，
明确对应的似然适用域及失败语义，再重跑相同四个初值的地壳实验。
不能只继续减小步长、换种子或删掉负向初值来获得通过标记，也不能用含噪观测的最大相速度
自动修改半空间先验。LM 自适应分层继续暂缓。

## 软件检查与复现

本轮 Release CTest **71/71**（18.37 s），相关 UBSan **5/5**（12.79 s）；最终日志刷新调整
再通过两项受影响的工作流与诊断回归。日志为 `ctest-final.log`、`ubsan-complete.log`、
`flush-regression.log`。无需新增数值或采样依赖。
Python 诊断回归覆盖正常独立链、停在不同区域的链、预热/正式失败区分、缺链、越界样本和
正式期步长变化；缺失或失败结果不能被静默删除以生成通过标记。

三层示例：

```bash
build/dispweave synthetic --config examples/nuts/reference_gaussian.toml --output results/reference-data-new
build/dispweave nuts --config examples/nuts/reference_gaussian.toml \
  --data results/reference-data-new/observations.txt \
  --covariance results/reference-data-new/covariance.txt --output results/reference-nuts-new
python3 scripts/posterior_diagnostics.py results/reference-nuts-new --output results/reference-nuts-diagnostics-new.json
```

地壳实验（`--case` 须为参考验证脚本准备的 λ=100 案例）：

```bash
MPLCONFIGDIR=/tmp/dw-mpl OPENBLAS_NUM_THREADS=1 python3 scripts/benchmark_crust_nuts.py \
  --case results/crust-regularization-scan/lambda-100 --exe build/dispweave \
  --output results/crust-nuts-new --warmup 1000 --draws 2000 \
  --chains 4 --metric-adaptation dense --workers 4
```

补充设置加 `--target-accept 0.99`。新实验若因会话中断且尚未写入结束记录，
可使用同一二进制与原输出目录恢复：

```bash
MPLCONFIGDIR=/tmp/dw-mpl OPENBLAS_NUM_THREADS=1 python3 scripts/benchmark_crust_nuts.py \
  --exe build/dispweave --output results/crust-nuts-new --resume-sampling --workers 4
```

恢复前校验原输入与二进制哈希、各链配置。已有成功结束记录的链（包括返回 5 的诊断失败链）
原样保留；中断链目录移为 `chain-N-interrupted-K`，以原种子、链 ID、初值重新执行。
这是整链重放，不是中途状态续采；失败链不会因诊断不好而自动重跑。
本次最终人工终止的链已写入 -15 结束记录，因此 `crust-nuts-conservative` 不会自动恢复；
后续修复后应建立新实验目录，保留本轮失败证据。

新实验输出目录必须不存在。`--workers 1` 顺序执行同一组随机流；`--prepare-only` 仅准备，
`--analyze-only --exe ... --output ...` 分析尚未生成报告的完整链输出。
单链只用于试运行；整个后验验收通过才返回 0，统计/数值验收失败返回 4，所有证据保留。
子进程中断或输入/输出不完整时返回非零错误，不进入完整后验验收。

结果包括输入与脚本快照、初始矩阵、逐链配置和输出、合并样本、全部参数/派生量诊断、
后验导数复核、预测、先验/后验结构图、指定诊断轨迹和相关矩阵。
不将条件可信区间解释为层数、界面或物性关系的不确定性，也不将区间收窄解释为深度分辨率。
