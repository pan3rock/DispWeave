# 架构与核心接口

本文定义整体接口契约。M3 已实现模型、正演与灵敏度、固定 Vs 参数化、协方差似然和后验目标，
以及有界 LM 和多初值文件工作流；M4a 增加固定参数化的局部谱与岭分辨诊断。
自适应分层和采样尚未实现。
主体使用 C++17，Python 只负责
处理与可视化。优先用普通数据结构和少量可替换接口，不建立通用工厂层。

## 依赖和分发

- CMake：构建、CTest、安装目标及构建信息。
- Eigen：向量、矩阵、Cholesky、QR、SVD；不要求外部 BLAS。
- toml11：模型、运行及算法参数配置；M1 固定为 4.4.0。
- CLI11：命令行与子命令；M1 固定为 2.4.2。
- invLM 数值源码：随仓库保存，不要求用户预先安装 invLM。
- Python：NumPy 与 Matplotlib 为绘图依赖；SciPy 仅用于可选独立验证。

“自己的实现”指主干算法、接口与测试在本仓库可维护，使用者不必安装外部反演/采样框架。
不因此重写已可靠的频散求根，也不重写 Eigen、TOML 或命令行解析。
HMC/NUTS 与 RJMCMC 后续分阶段实现；开源算法用于参考与独立对照。

依赖采用固定版本源码包或受控的随库源码，保留许可证与校验值。
默认配置过程不得自动联网；已准备依赖后应能在断网环境完成构建。
M1 默认且仅支持随库依赖；显式使用系统包的选项留到确有需要时增加，并报告实际版本。将第三方源码更新作为独立变更审查。
M0 不承诺未经测试的平台支持，M1 首先验收 Linux 的干净构建。

## 最小布局

```text
include/dispweave/       公共模型、正演、观测、参数化、目标函数接口
src/                    C++ 实现及 CLI；有足够代码后再按职责分目录
third_party/invLM/      带来源清单的物理代码
third_party/            固定版本基础依赖及许可证
cmake/                  依赖、安装和版本信息
scripts/                Python 数据处理和绘图
examples/               纳入版本管理的输入与运行说明
tests/                  数值、接口与 CLI 验收
benchmarks/             计时、内存与困难模型报告
docs/                   数学定义、开发计划和结果记录
```

M1 不创建空的 HMC、RJMCMC 或联合反演包装层。
核心库不负责文件输出；CLI 组合配置、算法与结果写入。

## 单位和模型

长度与深度 km，速度 km/s，密度 g/cm³，频率 Hz，角频率 rad/s。
所有模态从 0 编号；按同一频率下相速度递增定义序号。
序号约定不构成跨频率物理分支追踪保证；截止和近邻根需独立诊断。

```cpp
struct LayeredModel {
    Eigen::VectorXd depth; // 层顶，depth[0]=0，严格递增
    Eigen::VectorXd vs, vp, rho; // 与 depth 等长；最后一项为半空间
    void validate() const;
};
```

首版只接受全固体、各向同性、弹性介质；所有数值有限，Vs/rho 正值，
`Vp² > (4/3) Vs²` 保证正体积模量。允许低速层和高速夹层，不要求 Vs 单调。
半空间不使用虚假的有限层厚。液层、泄漏波及衰减在首版明确不支持。

## 多阶正演

```cpp
struct DispersionQuery { double frequency_hz; int mode; };
ForwardResult evaluate(const LayeredModel&, const std::vector<DispersionQuery>&,
                       SensitivityMethod = SensitivityMethod::none) const;
```

输入为逐观测点的 `(frequency, mode)`，允许乱序、重复频率、各阶不等采样。
输出与输入逐行对应，绝不缩短数组。内部可按频率共用根枚举，但不得改变行顺序。

`ForwardResult` 包含：

- 相速度向量；无效位置为 NaN。
- 每行 `RootStatus`：`valid`、`missing_or_unresolved`、`numerical_failure`。
- 模态存在性：根已找到，或未知；在没有完备性证据时不报告“已证实不存在”。
- 根数量、搜索异常、备用值来源（若后端暴露）、说明文本。
- 可选 `dc_dvs`、`dc_dvp`、`dc_drho`，行对应观测，列对应层。
- 独立的 `sensitivity_valid`，以及能量残差、重构标记或 root correction。

Phase 1 没有层厚导数；后续灵敏度接口增加显式参数描述，不暗中改变列顺序。
只有正演值有效并且所需梯度有效，梯度型试算才可接受。

后端通过一个 C++ 抽象接口替换；首个实现使用 invLM 的 `search` 和核函数。
默认不启用全局可变缓存。每次评估独立持有数值工作区，为后续并行实验留出线程安全边界。

## 观测及似然

```cpp
struct DispersionObservations {
    std::vector<DispersionQuery> queries;
    Eigen::VectorXd phase_velocity;
    Eigen::MatrixXd covariance;
};
```

输入文本约定 `frequency phase_velocity mode sigma`；每行 sigma 必须显式提供且为正。
可另行提供完整协方差，顺序严格对应文本行；完整协方差取代对角 sigma 模型，
不能同时重复计权。输出保存实际使用的完整协方差。

验证维度、有限性、对称性、正定性。用 Cholesky `C=LLᵀ` 白化：

\[
r=L^{-1}(g(m)-d),\quad J_w=L^{-1}J,
\]
\[
\log p(d\mid m)=-\tfrac12[r^Tr+2\sum_i\log L_{ii}+n\log(2\pi)].
\]

不显式求逆。不能丢掉缺失预测行，也不能按每次成功正演的子集改变目标函数。
求根失败返回数值失败状态；LM 拒绝 trial，初值失败即终止并报告。
将来概率采样器必须区别物理支持外与数值失败，后者需要计数和阈值终止策略，
不能静默转成物理上的零概率。噪声超参数变化时重算分解和归一化项。

后续观测模块共享“预测—误差模型—似然值/可选梯度”契约，独立声明梯度能力。
H/V 不等于 Rayleigh 椭圆率；接收函数还需滤波、时间偏移及震源等参数。
独立观测的 log likelihood 可相加；相关观测必须采用联合协方差。

## 参数化

首版固定层顶，仅反演各层 Vs（含半空间）。优化变量 `q_i=log(Vs_i)`，
设置显式有限下上界；物理模型可自由出现速度逆转。

```cpp
Model decode(const Eigen::VectorXd& q) const;
Eigen::VectorXd encode(const Model&) const;
Eigen::MatrixXd phase_jacobian(const Eigen::VectorXd& q, const ForwardResult&) const;
double log_abs_det_jacobian(const Eigen::VectorXd& q) const;
Eigen::VectorXd log_abs_det_gradient(const Eigen::VectorXd& q) const;
```

`encode` 必须校验模型确实属于该参数化，不能静默丢失层厚、Vp 或密度变化。
首版 Vp 支持固定或固定比值 `Vp=a Vs`，密度固定。
此时 `dc/dq_i = Vs_i (dc/dVs_i + a dc/dVp_i)`；固定 Vp 时第二项为零。
未来经验关系必须提供自身导数，不能只更新模型值却遗漏链式项。
联合反演弱约束参数须显式配置先验，作为后续独立功能。

可变层厚用正厚度增量或其他有序变换防止界面交叉；样条/节点参数化提供
显式投影及链式导数。自适应增删参数产生新的参数化实例，并记录操作历史。
RJMCMC 使用自身可变维度状态，不强行套用固定长度 q。

## Log posterior

```cpp
PosteriorEvaluation log_posterior(const DispersionProblem&, const Eigen::VectorXd& q,
                                 const Prior&, bool gradient=true, double log_scale=0);
```

返回 log likelihood、log prior、变换项、总值、可选梯度及状态，分别保存来源。
先验必须声明密度相对于什么测度。若先验定义在物理参数 θ，且 θ=T(q)：

\[
\log\pi_q(q)=\log p(d\mid T(q))+\log\pi_\theta(T(q))
+\log|\det DT(q)|.
\]

对于仅 Vs 的对数变换，变换项为 `sum(q)`，梯度为全 1。
如果先验本来定义在 q，则不能再次添加此项。
确定性最小二乘不自动加入采样坐标的 Jacobian 项。

有限边界的 log-Vs 可用于 LM；后续 HMC 优先用无约束平滑变换进入有限支持区间，
或经验证的边界处理，不能把 LM 的投影操作复制进 leapfrog。

## 固定层数 LM

保留残差和 Jacobian 的统一评估。M3 在边界宽度归一化的坐标中求解阻尼最小二乘，
用列范数定义阻尼尺度，通过增广 QR 避免直接求逆。依据实际/预测下降比接受试算并更新阻尼。
边界处理必须用实际可行步计算预测下降；非法根或核的试算拒绝。

记录代价、阻尼、增益比、接受状态、根/核失败次数、终止原因和实际评估次数。
小步长不自动等于成功：检查边界投影梯度和是否持续遭遇无效 trial。
支持多个初值的独立运行并报告全部结果，不能只展示最优一次。
首版不默认强平滑或自适应增层；这些改变科学问题，另行验收。
实际接口为 `bounded_lm` / `invert_lm`，数值常量、投影 Cauchy 步、终止定义与局限见
[M3 验收报告](m3-validation.md)。

## 局部信息诊断（M4a）

`local_resolution` 以白化 Jacobian 和明确参数尺度为输入，以完整 SVD 保留欠定零空间。
数值秩单独报告；给定 `alpha>0`，计算 `s²/(s²+alpha²)`、其总和与模型分辨矩阵。
它们对应指定尺度下的岭线性估计器，不是层数选择或后验可信度。
`resolution` CLI 当前只分析固定 Vp/密度/界面的 log-Vs 参考模型，支持完整协方差。
具体测度、文件与解释边界见 [M4a 验收](m4a-validation.md)。

## 主要风险

| 主题 | 数学风险 | 工程与验收要求 |
| --- | --- | --- |
| 隐式导数 | `D_c≈0`、根聚集、截止、尺度/归一化不光滑 | 隐式/能量核/多步长差分分别比较；报告病态状态，不能假定能穿透离散搜根 |
| 模态编号 | 漏根会使后续阶数错位，近根不代表同一分支 | 独立根参考与截止扫描；warm start 必须等价于确定性基线 |
| 自适应分层 | 参数尺度和先验影响奇异值；改善拟合不等于增加信息 | 白化且声明参数度量；独立比较 SVD、信息准则和复杂度代价 |
| HMC/NUTS | 不可微目标、错误梯度、多峰、不充分适应 | 独立分布基准、能量误差、发散、ESS/R-hat、多初值多链；阶段冻结适应 |
| RJMCMC | birth/death 逆映射、选择概率、Jacobian 或层数先验遗漏 | 可解析跨维目标、正反提议比与详细平衡验证；维度驻留和往返诊断 |
| 不确定性解释 | 条件后验误当模型选择后验；窄区间误当高分辨率 | 分开报告固定参数化与跨维不确定性、先验敏感性和分辨证据 |

## 结果与性能

记录模型、原始观测、实际协方差、预测、诊断、配置快照、随机种子、软件/编译器版本
及上游物理版本。文本/TOML 足以服务首版，后续按数据规模评估二进制格式。

性能报告区分首次初始化和重复调用，分别记录普通模型与困难模型的墙钟时间、
调用次数、峰值内存、硬件与构建参数。先建立正确性和性能 baseline，再优化热点。
不得为局部速度收益移除数值诊断或添加不透明的替代根。
