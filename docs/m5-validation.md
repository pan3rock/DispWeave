# M5 固定维度 HMC 验收

本项实现自有静态 HMC，并接入固定层顶、固定密度的 Vs 频散后验。
无 NUTS、自动质量矩阵估计、层数变化或层厚反演；这些不计为已完成功能。
生产构建仅使用既有 Eigen、toml11、CLI11、invLM。MCMCLib 仅用于独立采样器对照。

## 数学与接口

`include/dispweave/hmc.hpp` 提供 `LogDensity`、`LogisticBox`、`EuclideanMetric`、
`leapfrog`、`sample_hmc` 和频散目标适配器。目标返回 log density、同坐标梯度和明确状态；
维度错误、NaN、无效根或无效梯度不会被替换为零梯度。
目标必须在一次运行中保持确定性，回调中的程序异常直接传播。

质量矩阵 `M` 明确定义为动量协方差：

- `p ~ N(0,M)`，`K(p)=pᵀM⁻¹p/2`。
- 通过 Cholesky 分解抽取动量和求解速度，不显式求逆。
- leapfrog 按半步动量、整步位置、半步动量更新，位置不裁剪。
- 固定步数 `L`；每次独立刷新动量，以 `min(1,exp(H_old-H_new))` 接受。
- 拒绝时保留原位置作为样本，不删除重复状态，不做 thinning。
- 动量不持久保存，最后的动量翻转不影响位置边缘转移。

对高斯位置协方差 `C`，理想预条件质量是 `C⁻¹`，不是 `C`。
核心支持任意用户提供的 SPD 矩阵；当前不估计或适应质量矩阵。

有界参数首先为 `q=log(Vs)`，再使用
`q=lower+(upper-lower)*sigmoid(z)`。
采样目标是 `log p(q(z)) + log|dq/dz|`。
若先验以物理 Vs 为测度，已有后验接口包含 `sum(q)`，此处再加入 logistic Jacobian；
若先验以 log Vs 为测度，则不加入 `sum(q)`。两种测度不混用。
稳定 softplus 计算 log Jacobian；浮点数把 q 舍入到端点时显式报 numerical failure，
不会把裁剪后的端点当作可逆变换。初值必须严格在界内。

预热使用 dual averaging（gamma=.05、t0=10、kappa=.75、mu=log(10*initial_step)），
最后冻结平滑后的 nominal step。正式样本不参与任何适应。
`step_jitter` 默认 .1，每次从 `[1-j,1+j]` 独立均匀抽取步长乘数，
避免固定周期共振；可显式设为 0。这是与位置无关的可逆转移混合，不是自适应轨迹长度。
`target_accept` 默认 .8，但弯曲后验可能需要更高值及更小步长。

每条链由 `(seed, chain_id)` 的四个 32-bit 字通过 `seed_seq` 初始化私有 `mt19937_64`。
链间不共享随机状态；同一可执行程序、输入和种子可重放；标准库分布实现差异意味着
不承诺跨编译器/标准库逐 bit 一致，也不声称数学上证明不同流绝对独立。

`HMCTransition` 包含预热标记、接受结果、实际步长、接受概率、能量差、发散、
目标状态和梯度评估次数。`|ΔH| > divergence_threshold`（默认 1000）标记能量异常，
无效评估也标记异常并拒绝。能量异常标记本身不改变有限提议的 Metropolis 接受规则。
根/核失败不代表真正的零后验；出现时应调查正演支持域，不能把链当成已认证后验。
CLI 正式样本有异常或零接受时返回 5，并保留结果；返回 0 只代表本轮没有这些异常，
不等价于多链收敛或全局混合证明。初值无效或配置错误返回 1。

算法定义参考 [Stan HMC 说明](https://mc-stan.org/docs/reference-manual/mcmc.html)；
步长适应参考 [Hoffman 与 Gelman, 2014](https://jmlr.org/papers/v15/hoffman14a.html)。
代码为本项目实现，没有复制 Stan/MCMCLib 实现。

## 测试设计和发现

新增测试：

- `hmc_mechanics`：非对角质量下的 leapfrog 可逆性（误差 <2e-13）、固定积分时间下
  步长减半的能量误差比 3.8–4.2；5 万个动量的协方差及质量矩阵方向检查。
- `hmc_transform`：双向变换、有限差分梯度、极端坐标诊断、均匀先验变换后采样矩。
- `hmc_distributions`：标准高斯、相关系数 .995 的高斯、香蕉形非高斯。
  非高斯为 `x ~ N(0,1), y=e+.5*(x²-1), e ~ N(0,1)`，因此均值为 0、
  方差为 `(1,1.5)`、协方差为 0；相关高斯先白化后检查矩，避免弱方向被掩盖。
  种子固定为 17、2026、99173，各 800 预热 + 12000 样本。
- `hmc_failures`：重放、链流区分、梯度失败拒绝、重复状态保留、非法输入、
  预热后步长冻结以及过大步长导致的能量异常。
- `hmc_physical`：均匀半空间，使用独立 Rayleigh 三次方程计算 `c/Vs`；
  固定 Vp/Vs 后物理 Vs 后验为几乎不受边界截断的已知高斯。
  检查完整两级变换的梯度及采样均值/方差，不使用 invLM golden 值作为真值。
- `hmc_workflow`：三链 CLI 重放、协方差保存、样本行数、边界、缺根和输出保护。

解析统计验收阈值在首轮试验前确定：60 个 batch means，均值及二阶矩误差
不超过 `max(.04,6*batch_MCSE)`，MCSE < .10；接受率 >.55，正式采样无发散。
未使用独立同分布样本的标准误，也未在失败后放宽这些阈值。

首轮固定积分长度在高斯上产生共振：接受率 .9765，二阶矩却为 1.393，
MCSE .173，测试正确失败。加入独立步长扰动后解决了这个测试中的共振。
香蕉形目标在接受率目标 .8 和 .95 时仍有正式采样发散，最终设为 .99；
高斯使用 .95。配置变化如实保留在测试代码中，不能由此声称默认 .8 对所有目标可靠。
最终九组解析试验接受率 .93625–.98975，正式采样均无发散，所有矩通过阈值。
半空间标准化样本均值 .02851（MCSE .03030）、二阶矩 .98274（MCSE .02995）。

## 实际运行记录

2026-10-22，本机 Intel Core Ultra 9 285H，GCC 13.3.0，版本 0.7.0：

- Release：35/35 通过，最终完整回归 1.92 s。
- Debug + UBSan：35/35 通过，212.07 s；使用
  `-fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer`。
  之后补充的变换饱和和冻结步长断言另做 UBSan 定向回归，2/2 通过（1.26 s）。
- 同捆依赖校验：449 文件通过，物理源码没有改动。
- 三链频散示例共 3000 预热和 12000 正式样本，单次墙钟 18.31 s、峰值 RSS 6480 KiB，
  包含进程初始化和文件写出；这是本机测量，不是跨平台性能保证。
- `plot_hmc.py` 已生成并目视检查轨迹/边缘分布图；生成结果放在忽略目录 `results/m5/`。

## 可复现的频散示例

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
build/dispweave synthetic --config examples/hmc/synthetic.toml --output results/m5/data
build/dispweave hmc --config examples/hmc/synthetic.toml \
  --data results/m5/data/observations.txt --covariance results/m5/data/covariance.txt \
  --output results/m5/final
python scripts/plot_hmc.py results/m5/final
```

输出目录必须为空或新建。配置共享 LM 的 `[model]`、`[inversion]` 中的边界、初值、
Vp 策略和灵敏度方法；`[hmc]` 定义先验、抽样和质量矩阵。
CLI 当前提供 `uniform_vs` / `uniform_log_vs`，C++ 接口也能使用已有 Gaussian prior。
直接对实际观测使用同一 `hmc --data ... --covariance ...` 命令，不需生成合成数据。

示例为三层、基阶加第一高阶、14 个观测、sigma=.02 km/s、相关系数 .3 的协方差，
噪声种子 2026。固定 Vp/rho、隐式核、物理 Vs 均匀先验，三条不同初值，
每条 1000 预热 + 4000 样本，L=12，M=100I，目标接受率 .95。
结果条件于指定的层数、层顶、Vp/rho 和先验，未包含模型选择不确定性。

| 链 | 正式接受数 / 4000 | 冻结 nominal step | 正式发散 / 无效评估 |
|---|---:|---:|---:|
| 0 | 3797 | .26985684 | 0 / 0 |
| 1 | 3811 | .28125909 | 0 / 0 |
| 2 | 3797 | .28004969 | 0 / 0 |

`chain_*_vs.txt` 和 `chain_*_z.txt` 仅保存正式样本；trace 包括明确标记的预热。
`chains.txt`、`mass.txt`、配置、观测、完整协方差及版本/编译器/种子 metadata 一起保存。
LM 同数据最佳数据代价为 3.37935，仅作为定位参考，不把 LM 模型当作后验均值。

## 独立实现对照

使用本地 MCMCLib 2.1.0 header-only 副本，Apache-2.0，来源于
DispNUTS 仓库 `fbe681ba45bc43ae1b96990a2abed2f8d5610fd3` 下的 `third_party/mcmc`。
这不是声称该副本等于最新 upstream。实际 `hmc.hpp` SHA-256：
`ca2f71cbbd73f2656f9bbaf53ae90e2b8f26523db75a5eb5bca892f60dc3499c`。
完整 header-only 树 83 文件的排序内容摘要为
`ba4c3fd2429fb866a99469ac20f1b5ef9d26fb91944f6c652f59eda5eef62b2d`
（依相对 POSIX 路径排序，每个依次 hash `路径+NUL+文件内容+NUL`）。

对照不使用其内置边界变换，而把同一个完整 z-space log posterior/gradient 传入，
独立比较动量生成、积分、接受和抽样；因此它不能独立证明共享物理核或似然正确。
后者由 M1/M2 和新增解析半空间后验测试承担。
参考库的 `min(.01,log_ratio)` 对均匀随机数接受事件与 `min(0,log_ratio)` 等价；
检查到这一代码细节后没有修改第三方源码。遇到无效梯度时对照程序直接报错。
MCMCLib 无本项目完整异常诊断，不能把它返回成功等同于全局收敛。

```bash
cmake -S . -B build-reference -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DDW_MCMCLIB_INCLUDE=/path/to/mcmc/header_only_version
cmake --build build-reference --target hmc_reference -j2
for seed in 17 2026 99173; do
  build-reference/hmc_reference examples/hmc/synthetic.toml \
    results/m5/data/observations.txt results/m5/data/covariance.txt \
    results/m5/reference_${seed}.txt 0.27 "$seed" 4000
done
python scripts/compare_hmc.py results/m5/final results/m5/reference_*.txt
```

引用实现固定 epsilon=.27、L=12、M=100I，每个种子 1000 预热 + 4000 样本，
无步长适应。接受数分别 3722、3744、3766。编译目标默认关闭，不安装也不打包第三方库。

对照前定义阈值：按链 40 个 batch，比较标准化均值和全部上三角中心二阶矩，
差异不超过 `max(.04,6*combined_MCSE)`，两方合并均值 MCSE 均 <.1 个边缘标准差。
三个均值及六个二阶矩均通过；这是一组具体分布回归，不是任意后验的认证。

| 各层物理 Vs 均值 (km/s) | DispWeave | MCMCLib |
|---|---:|---:|
| 0 | 1.002312 | 1.002488 |
| 1 | 1.970592 | 1.971905 |
| 2 | 3.094741 | 3.082446 |

两方均值 MCSE 分别约 `(0.000089,0.00151,0.0188)`、`(0.000077,0.00151,0.0184)`。
半空间的 MCSE 明显更大，当前固定质量下抽样效率仍有限；高接受率没有消除这一问题。
分位数和轨迹可用于检查，尚未提供可靠 ESS、rank-normalized split R-hat 或多峰探索保证。

## 剩余工作

M6 再加入 NUTS、分窗口质量矩阵适应、完整多链诊断及后验预测/覆盖率实验。
当前不把狭窄可信区间等同于深度分辨能力，也不把多个种子的通过当作全局多峰探索证明。
对截止、分支切换、根拥挤和未认证复杂模型，M1/M2 已记录的限制继续适用。
