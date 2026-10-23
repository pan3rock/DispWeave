# 地壳 NUTS：模态支持域与预热收尾

2026-10-24。继续原有 45 参数、125 个含噪观测的地壳案例。
本轮处理缺模态试探点的分类和负向初值的预热受阻；保持观测、协方差、参考先验、
物性映射、速度边界和四个初值不变。LM 结果保持原样。

## 完整受俘获域的数值检查

`src/mode_support.cpp` 直接组装全局 P/SV 位移和牵引连续条件；有限层使用
`cosh(qξ)`、`sinh(qξ)/q` 正则基底，半空间取衰减 P/S 势。
独立于生产 Dunkin 递推和根搜索。矩阵正定标后使用带主元的带状消元求行列式相位；
原上下带宽均为 5，主元交换后上带宽最多 10。
验证参考 `benchmarks/rayleigh_mode_audit.py::trapped_count` 使用独立的稠密 LAPACK LU。

旧审计的有限相速度窗口不足以定义一般支持域。本轮补齐两个端点：

1. 下端使用弹性能量比较得到的保守下界。令
   `μ₀ = min_layers(min(μ, λ+μ))`、`ρmax = max_layers(ρ)`。
   平面应变能 `λ|εxx+εzz|² + 2μ(|εxx|²+|εzz|²) + μ|γxz|²`
   不小于 `λ=0, μ=μ₀` 均匀半空间的应变能，质量不大于 `ρmax` 对应的质量。
   在相同自由表面变分空间、固定实波数 k 下比较 Rayleigh 商，可得
   `c² ≥ (3−√5) μ₀/ρmax`。其中 `3−√5` 是 λ=0 均匀半空间
   Rayleigh 方程的物理解。计数下端取该速度的 0.95 倍，严格留在下界以下。
   这个推导只用于稳定、无耗散、各向同性固体模型。
2. 令 `t = sqrt(1−(c/Vs_halfspace)²)`，半空间 S 垂向波数直接写成 `k*t`，
   `c=Vs_halfspace*sqrt(1−t²)`。于是半空间截止位于解析的 t=0，计数覆盖
   `0 ≤ t ≤ sqrt(1−(c_lower/Vs_halfspace)²)`，没有人为省略的截止尾段。

复轮廓在 t=0 附近收窄，避免纳入沿虚轴出现的泄漏延拓。上下边高度为
`min(h, 0.25*t_real+1e−10)`，h 分别取 0.02、0.01；实部同时采用均匀与对数网格。
两种轮廓各需相邻两级网格计数一致、相位增量小于 π/4、绕数整数误差小于 1e−8；
默认从 64 段细化到最多 2048 段。半空间 P 分支点及 t=±1 不在轮廓内。
采用解析无零因子 `exp(2kH)` 的相位消除背景旋转，正行/列定标不改变零点或绕数。

两个实端点额外计算全矩阵奇异值，最小/最大比值低于 1e−12 时返回未解决。
开发中的极限控制发现：单独检查消元主元不足以识别截止端点的整体病态性，故新增此保护；
保护前的试跑保留于 `results/crust-nuts-closeout/`，没有用作最终四链验收。
临界半空间 Vs=4.439194324296876 的正式失败模型控制中，临界点返回未解决，
±1e−8 km/s 两侧分别数到 2、3 个模态。该极限控制已加入永久回归。

这仍是有收敛门槛的浮点数值检查，**不是区间算术无根证明**。
计数预算耗尽、轮廓不一致、端点病态、与生产根前缀不一致均保留为未解决。
复杂零点可能使计数偏多，此时不会据此声明所请求的模态不存在。

## 似然与采样语义

新增显式选项 `mode_support="trapped_modes"`，表示每个已标记观测对应的受俘获模态必须存在。
先执行原正演；仅当某查询为 `missing_or_unresolved` 时调用完整域计数。
确认总数等于生产正演已找到的前缀且不大于请求的零基模态编号时，目标为
`outside_support`、`log_likelihood=value=−∞`。一个必需模态不存在即可使该模型的整体似然为零，
没有删掉观测行或只拟合剩余观测。

已有根的正演和梯度路径不变。默认 `strict_forward` 继续将未解决根保留为正演失败。
普通正演数值失败不能凭状态码变为支持域拒绝；故意漏掉均匀半空间已有基阶根的回归仍返回失败。
这个选项没有扩展到联合观测或 RJ 模型。

HMC 对已知支持域外提议正常拒绝；NUTS 停止对应子树、保持其切片候选数为零。
这与原先失败试探的状态选择路径一致，单独区分诊断分类。
`outside_support` 必须同时提供负无穷密度；NaN 等错误值仍为数值失败。
轨迹的无限能量误差保留，但已知支持域拒绝不计作数值发散或无效评估。
`record_failures` 仍保存这类试探点及明确状态，诊断另报 `support_rejections`。

解析半正态×正态和均匀单位圆盘各用三个种子、1000 预热和 16000 正式状态检查一、二阶矩；
保留拒绝后的重复状态，要求实际遇到边界。已有正常、相关、弯曲目标以及真实数值失败回归保留。
NUTS 仍使用 [Hoffman–Gelman 2014 Algorithm 6](https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf)，
未引入反射动力学或改变子树抽样规则。

## 预热几何

原实验把参考模型处的局部 Gauss–Newton 矩阵用于全部四个初值。
新增 `initial_metric="start_gauss_newton"`，在每条链自己的原初值处计算
`M = D (Jvsᵀ C⁻¹ Jvs + diag(1/σprior²)) D`，D=dVs/dz。
这是动量协方差，即位置协方差的逆，不是把两种质量矩阵约定互换。
仅支持显式参考高斯先验；不能同时传入 `mass`。
每链实际初始矩阵保存为 `chain_N_initial_mass.txt`。
启用这种逐链初始化时，兼容输出 `mass.txt` 为单位阵占位，不表示实际使用的逐链矩阵；
读取初始几何应以 `chain_N_initial_mass.txt` 为准。
密集协方差适应窗口、目标接受率 0.99、步长适应和冻结规则均保持原样。

同一负向初值、种子及目标的试跑对照：

| 初始矩阵 | 已完成预热 | 目标评估 | 支持域遭遇 | 树深截断 | 数值无效 |
|---|---:|---:|---:|---:|---:|
| 参考点矩阵，人工停止并保留前缀 | 61 | 18657 | 57 | 15 | 0 |
| 原初值处矩阵，完整预热 | 1000 | 30389 | 1 | 0 | 0 |

后一试跑预热有 1 次有限能量发散，正式 8 步无发散/无效/截断；这 8 步只用于试跑确认，
没有作为后验验收。图与表在 `results/crust-nuts-closeout-warmup/`。
两试跑使用加入端点 SVD 保护前的冻结版本，最终版本另从原四个初值重跑。
只调整最初矩阵已足以消除本例的严重受阻，因此没有再更改适应窗口或重新选择初值。

新增 `record_progress=true` 可在每个 NUTS 转移后刷盘迭代、状态、步长、树深及 z，
包括预热；观察回调不消耗随机数，不影响轨迹，已有确定性回归。

## 复现

```bash
cmake --build build -j 4
ctest --test-dir build --output-on-failure -j 4
OPENBLAS_NUM_THREADS=1 python3 scripts/audit_crust_mode_support.py \
  --cases results/crust-mode-audit-refined/cases.json \
  --exe build/mode_support_audit --output results/mode-support-new
OPENBLAS_NUM_THREADS=1 python3 scripts/benchmark_crust_nuts.py \
  --case results/crust-regularization-scan/lambda-100 --exe build/dispweave \
  --output results/nuts-new --chains 4 --warmup 1000 --draws 2000 \
  --initial-metric start --mode-support trapped_modes --target-accept .99 --workers 4
```

`workers` 是独立数值采样进程；开发、审查和收尾均为单代理。
本轮正式结果目录 `results/crust-nuts-final/`，冻结可执行文件在
`results/crust-nuts-final-tools/`；输入和二进制哈希在运行前保存。
最终验收结果在该目录 `acceptance.json`，不使用旧三链子集替代原四链。

## 最终验收结果

原四个初值全部完成，每链 1000 预热、2000 正式状态，共 8000 个正式状态。
**四链验收通过。** 最大 R-hat=1.00155835，最低 bulk ESS=4352.72、tail ESS=3766.20，
均满足原门槛（R-hat≤1.01、bulk/tail ESS≥400）。正式期数值无效、发散和树深截断均为零。

| 链 | 正式状态 | 支持域拒绝 | 正式平均树深 | 冻结步长 |
|---:|---:|---:|---:|---:|
| 0 | 2000 | 1 | 4.9930 | 0.13496011 |
| 1 | 2000 | 0 | 6.6915 | 0.04157641 |
| 2 | 2000 | 2 | 4.9930 | 0.13287257 |
| 3 | 2000 | 1 | 4.9880 | 0.14363967 |

4 次正式支持域拒绝不是丢弃观测或删除正式状态；所有转移和拒绝后的重复状态均保留。
本轮全部支持域外试探另以 refined 正演和独立稠密完整域计数重放，见 `support-replay/`。
四个预先指定的中间正式状态完成全参数、两个步长的有限差分检查，并比较标准/精细根导数。
后验预测选取 800 个状态，失败 0；总体 PPC 超越比例 0.52625。

Release CTest 73/73 通过；新增端点保护后的 mode_support、NUTS core 的 UBSan 检查 2/2 通过。
118/118 个历史失败查询在最终版本与独立稠密计数间一致，源输入哈希保持不变。
原观测/协方差逐字一致，model、inversion、先验及原采样日程逐项一致；改变的是明确的
模态存在域选项与初始预条件矩阵。试跑到最终版本的已保存轨迹前缀完全重现，校验在
`results/crust-nuts-final-validation/`。

图在 `results/crust-target-final/target-nuts.png/pdf` 和 `results/crust-nuts-final/`，
[本地结果页](../web/results/index.html) 已更新，保留原目标、原数据和 LM 对照。
历史失败、三链诊断、受阻前缀及端点保护前的试跑全部保留，未覆盖旧结果。
本轮确认该参数化与先验条件下的采样及预测诊断通过；不代表所有初值都同样高效，
也不包含界面、层数、先验中心、经验物性或噪声模型的不确定性。
