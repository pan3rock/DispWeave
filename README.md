# DispWeave

An extensible framework for adaptive multimode surface-wave inversion and joint seismic inference.

面向一维层状介质，研究多阶 Rayleigh 频散如何约束地下结构，以及数据支持多少模型自由度。
长期目标是物理可靠性、自适应复杂度和可靠的不确定性评估。

## 当前状态

本地结果网页：[反演结果](web/results/index.html)，按算例展示目标、LM/NUTS/RJMCMC结构与数据拟合，注明固定几何和变维反演的区别；固定层扩展结果折叠保留。
[RJMCMC阶段收尾](docs/rj-predictive-closeout.md#当前阶段总览)：停止当前通用调参与界面重排研究，报告主要特征恢复和分层非唯一性；完整定量后验仍未验收。[复现索引](docs/rj-results-index.json) 保存配置、种子、输入hash和命令，[旧诊断页面](web/results/history.html) 保留历史证据。
使用方式见 [网页说明](web/results/README.md)，支持浏览器直接打开，无需联网或发布。
[推荐配置与运行命令](docs/nuts-recommended-use.md) 已固定本算例通过验收的 0.95＋步长搜索设置。
[深度约束分析](docs/crust-depth-constraints.md) 区分单层速度与分段平均值：薄层细节先验依赖较强，部分宽区间平均速度明显收缩。
[先验宽度敏感性](docs/crust-prior-sensitivity.md) 已完成四链验收：放宽至 0.2 后薄层区间明显扩大，19–23 km 层出现下界影响迹象。
[本算例收尾判断](docs/crust-study-closeout.md)：已查明下界来自参考速度窗口；停止追加调参，局部幅度保留边界依赖的限制。
[独立六层低速模型验证](docs/small-lvz-validation.md) 已完成：四链与导数检查通过；LM/NUTS 中心接近，但低速对比仍不确定，不追加调参。
[RJMCMC 与固定界面 NUTS 对照](docs/rj-nuts-comparison.md) 已完成一次有限预算验证：NUTS 通过，RJ 的层数和界面混合未达标，保留警告基线。
[固定 K=2 的自由界面检查](docs/rj-fixed-k2.md) 四链通过；[K=3/4 检查](docs/rj-fixed-higher-k.md) 均失败，优先改进高层数内部结构探索，暂不重写 birth/death。
[开源实现调研](docs/rj-open-source-review.md) 已核查近期 BayesBay/Eryn/openMCMC 与 BayHunter：保留当前正确性基线。[PT 对照已完成](docs/rj-tempering-validation.md)：K=4 最差 R-hat 从 1.706 降至 1.051，成本接近，但仍未达到验收线。
[PT 慢变量分析](docs/rj-pt-bottleneck.md) 已定位第三界面相关结构访问不一致；暂不继续调固定 K=4，[原 K=1–4 PT 对照已完成](docs/rj-pt-transdim.md)：层数混合改善，深度仍未达标，发现第 0 梯最热链隔离及较多正演失败。
[异常热链独立核查](docs/rj-hot-branch-audit.md) 已完成：17 个模型、两档根扫描未发现明确不一致；温度梯在早期预热已隔离，转向检查初始化/预热衔接，仍不解除正演限制。
[预热记录与冷启动对照](docs/rj-pt-cold-start.md) 已完成：500 步 T=1 初始化未消除隔离，K=2 诊断反而变差；保留记录功能，停止预热长度扫描，默认策略不变。
[联合速度提议对照](docs/rj-joint-refresh.md) 已完成：热链隔离未复现，但联合接受率仅 0.157%、失败率升至 8.05%，整体仍未验收。默认关闭，并暂停继续内核调参。
[冻结相关联合提议](docs/rj-correlated-proposal.md) 已完成：联合接受率 20.84%，深度最低 bulk/tail ESS≈968/716，30/31 点通过；K=2 和一个深度点 R-hat 仍超标。计入旧训练样本成本，保留为默认关闭的有效候选。
[剩余差异定位](docs/rj-remaining-diagnosis.md) 已完成：未发现单次超长停留；层数占比和 K=3/4 内部分布都有差异。仅分析旧轨迹，建议冻结配置后独立复验，暂不增加机制。
[冻结配置独立复验](docs/rj-correlated-replication.md) 已完成：K=2/3/4 指示量通过，联合接受率20.45%，深度30/31点通过；停止本轮通用调参，仍保留完整后验未验收结论。
[RJ 后验预测与阶段收尾](docs/rj-predictive-closeout.md)：两次各400个预测模型无失败，均能解释12点合成观测；主界面附近与额外分层仍不确定。停止本轮通用调参，保留完整后验未验收结论。
[三层低速夹层迁移验证](docs/rj-lvz-transfer.md)：使用新预运行训练提议，恢复了低速层特征，但正式深度仅3/31点通过、最大R-hat≈1.050；不能将两层算例表现推广为通用收敛保证。
[低速夹层混合问题定位](docs/rj-lvz-mixing-diagnosis.md)：离线重建44万步、无新增正演；冷链K=4联合接受率仅0.61%，主要损失是越界和拟合变差，优先验证固定K的局部相关更新。
[固定K=4局部相关对照](docs/rj-local-k4.md)：已实现并验证，但尺度0.2提高接受率后反而降低ESS、加剧结构表示差异；不采用为推荐配置，停止本轮尺度调参。
[结构表示与切换路径](docs/rj-structure-paths.md)：跨维链约97%的A/B切换经过K=3；已有预测差异小于噪声，但地下剖面不同。暂不开发重排提议，按分层非唯一性收尾，保留后验未验收限制。
[新增观测辨别力筛查](docs/rj-structure-paths.md#后续数据辨别力筛查扩频与高阶模态)：64个已有模型与目标共65次正演；第一高阶0.6 Hz存在差异线索，但未形成通过预设规则的相邻频段。保留模态未解析和未收敛限制，不启动新链。
[第一高阶窄频段复核](docs/rj-structure-paths.md#第一高阶窄频段复核)：两组选样在0.60–0.80 Hz的5个连续网格点共同通过，45/45代表模型—频率根核查通过；支持下一步有限预算新增数据对照，尚未运行新链，不宣称结构唯一。
[新增两个高阶观测的配对反演](docs/rj-lvz-transfer.md#新增两个高阶观测的配对对照)：顶部/底部样本区间缩至基阶对照的44%/55%，但深度最大R-hat增至1.107、最低ESS下降，完整后验仍未验收；保留两组结果，不自动加长链或改内核。
目标与参考的区别及新增恢复对照见 [合成目标补图记录](docs/crust-target-comparison.md)。

**M8 已完成：频散、Rayleigh 模态椭圆率与 P 接收函数的固定维度联合反演。**
M8a 正演、M8b 接收函数、M8c 组合目标、M8d 端到端实验均已验收。
原九组实验中的标准冲突组曾发生采样混合失败；加入冻结的独立提议后，保持同一后验目标
的补充实验通过多链诊断与独立积分核对。联合 LM 的根精度停滞已修复。
原失败证据、补充结果和适用限制见 [M8d 验收记录](docs/m8d-validation.md)。
此前的办公室打包快照保留于 [M8 交接说明](docs/handoff-m8.md)，不代表最新进度。

已用 QEDispInv 自带合成示例完成一轮应用测试，见 [测试记录与复现](docs/qedisp-examples-validation.md)。
浅层多阶及匹配的 m1 例子通过固定参数化 LM/NUTS 测试；地壳 44 参数反演未通过。
已同步修复 DispWeave/invLM 默认 Dunkin 后端的已定位近邻根漏检，见 [修复验收](docs/clustered-root-repair.md)。
原 44 参数、无正则化的地壳配置未整体收敛，fast_scalar 在该困难模型上仍漏根；不能将局部修复视为全域正演可靠性证明。
已完成 [地壳半空间/深部分层 A/B/C 对照](docs/crust-depth-controls.md)：固定半空间不足以消除跳变，
合并均匀深部后的无正则化对照仍未收敛。后续已接入 [经验物性与显式正则化](docs/material-regularization.md)，
C 组加入参考阻尼后，固定物性/Brocher05 均为 3/4 收敛，剩余初值未解析。
随后使用 QEDispInv 的 `mref.txt`、45 个自由 Vs（含半空间）、交集边界、Brocher05 与
参考阻尼完成 [参考模型地壳验证](docs/crust-reference-validation.md)：**5/5 初值收敛**，
全部终态通过全参数 FD 检查；相速度 RMS≈0.0190 km/s，σ=0.02 为本轮假设。
随后完成 [参考阻尼敏感性扫描](docs/crust-regularization-validation.md)：固定其他条件，
λ=0/10/30/100/300 共 **25/25 收敛**且全部通过两步长 FD。λ=0 的解有 13–15 个参数触界，
组内最大逐层差达 1.2 km/s；非零阻尼各组内解一致，但跨权重浅部结构仍明显变化。
本轮未自动选权重，也未将收敛或低残差解释为结构唯一。
随后已接入 [物理 Vs 参考高斯先验与地壳 NUTS 实验](docs/crust-nuts-validation.md)：
先验归一化、解析分布、三层示例和软件回归通过；**45 参数地壳后验尚未通过验收**。
目标接受率 0.95/0.99 均遇到正式期缺模态，负向初值预热严重受阻；原失败记录和仅供诊断的
先验/样本区间图已保留。随后完成 [独立模态截止核查](docs/crust-mode-domain.md)：
113 条失败记录的 **118/118 个查询**均支持审计域内缺模态，未发现域内漏根；两个代表模型
验证了截止两侧根数由 2 变为 3。该浮点核查不是通用无根证明。
下一项是实现模态存在域与数值失败的分类契约，验证边界拒绝，再处理预热并重验原四链。

已提供 invLM Dunkin 正演、能量核与隐式导数、逐行根/梯度诊断、固定 Vs 参数化、
完整协方差似然和明确测度的后验目标，并已接入自有有界 LM、多初值与失败诊断。
Phase 1 最小可运行版本已完成；新增参考模型处的灵敏度谱、显式参数尺度和岭分辨矩阵。
固定层数 LM 已改用有界阻尼子问题；HMC 提供平滑边界变换、固定质量矩阵、预热步长适应及多链输出。
LM 自适应分层暂缓。
固定层数不等于固定层厚：当前固定层厚，若以后反演层厚，须先验证相应参数化和导数。
已移除候选模型的 AIC/BIC、验证评分及排名实验。M6a 已补齐 ESS/R-hat、相关性和后验预测；
M6b 已增加对角/稠密质量矩阵适应；M6c 已接入 slice NUTS。
M7a/M7b 已完成跨维状态、归一化先验、可逆增删层提议及独立 RJMCMC 核心；
解析跨维验证已通过。M7c 已完成频散 CLI、跨维诊断、后验预测和先验对照；正演支持域仍有明确数值限制。
每个里程碑完成验收后单独 commit，不提前铺开全部算法。

C++17 为主体，Python 用于数据处理和绘图。Eigen 3.4.1、toml11 4.4.0、
CLI11 2.4.2 和所需 invLM 数值源码随仓库提供，构建无需联网或安装反演框架。

## 构建与运行

已验收 Linux、GCC 13.3、CMake 3.28；需要 C++17 编译器与 CMake ≥3.16。
不需要 Python、Ceres、HDF5、BLAS 或包管理器即可编译运行核心。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
build/dispweave forward --config examples/forward/normal.toml --output build/normal.txt
```

配置中的 `queries.frequency_hz` 与 `queries.mode` 按行配对，可不同模态不等采样、乱序、重复。
模型数组为层顶深度 km、Vs/Vp km/s、密度 g/cm³；最后一层是半空间，模态从 0 开始。

输出文本包含版本、模型、每个原始查询的速度、状态和诊断。
`roots_found` 只是本次找到的请求根前缀数量，不是全部模态数或完备性证明。

```bash
build/dispweave forward --config examples/forward/missing.toml --output build/missing.txt
```

这个半空间示例特意请求不存在的高阶根，进程返回 **2**，输出仍保留两行。
无效相速度为 `nan`，状态为 `missing_or_unresolved`；不能当作有效频散参与拟合。
返回码 **0** 表示所有请求均得到有效根；**1** 表示配置、模型或 I/O 错误。
CLI11 参数解析错误使用其自身非零返回码。
输出文件的父目录需存在；已有结果文件会被覆盖，但禁止输出指向输入配置文件。

可选安装到自己的目录：

```bash
cmake --install build --prefix "$PWD/build/install"
build/install/bin/dispweave --version
```

可选源码校验与性能测量：

```bash
python scripts/verify_vendor.py
/usr/bin/time -v build/forward_benchmark normal
/usr/bin/time -v build/forward_benchmark dense_low_velocity
```

## 灵敏度与目标函数

```bash
build/dispweave forward --config examples/forward/normal.toml \
  --sensitivity energy --output build/energy.txt
build/dispweave forward --config examples/forward/normal.toml \
  --sensitivity implicit --output build/implicit.txt
build/objective_example
```

两种灵敏度接口独立保留有效根和梯度诊断。请求梯度但只有梯度失败时返回码为 3。
后验示例通过 C++ API 使用相关误差，输出目标分解及梯度，不执行优化或采样。
见 [M2 接口与验收](docs/m2-validation.md)。

## 固定层数反演

```bash
build/dispweave synthetic --config examples/inversion/synthetic.toml --output results/synthetic
build/dispweave invert --config examples/inversion/synthetic.toml \
  --data results/synthetic/observations.txt --covariance results/synthetic/covariance.txt \
  --output results/replay
python scripts/plot_inversion.py results/synthetic
```

反演输出目录必须新建或为空；保存所有初值、终止原因、正演/梯度异常和试算轨迹。
若没有初值通过收敛标准，返回码 **4**，不能将末次模型当成已收敛解。
`runs.txt` 的 `termination` 区分 `converged`、`stalled`、`failed`；小步停滞不视为成功。
观测列为 `frequency_hz phase_km_s mode sigma_km_s`；可提供完整协方差。
当前固定层顶、反演 Vs，Vp/密度可固定，或声明 `vs2model` 使用 Brocher05、Gardner、
NearSurface 等 invLM 经验关系及完整链式导数。`inversion.regularization` 显式指定类型和 lambda，
支持参考阻尼、扰动平滑、Vs 平滑；默认关闭，数据/正则化代价分别输出。
可运行示例为 `examples/inversion/brocher_regularized.toml`，定义与验收见上述物性/正则化报告。
速度边界推荐用 `[inversion] vs_min / vs_max / vs_width`：每层取全局区间与
`model.vs ± vs_width/2` 的交集，width 是 km/s 的完整宽度。也兼容逐层 `lower_vs / upper_vs`，
两种配置不可混用。LM、HMC/NUTS 和后验预测共用生成规则，输出 `parameter_bounds.txt`。
半空间顶深固定、Vs 默认自由，使用同样边界；不以含噪观测的最大相速度自动抬高速度下限。
配置与独立真值的顶深对照见 [边界与半空间验收](docs/velocity-bounds.md)：

```bash
MPLCONFIGDIR=/tmp/dw-mpl python scripts/check_halfspace_depth.py --demo \
  --exe build/dispweave --output results/halfspace-window-demo
```

`[inversion] fixed_vs_layers = [2]` 可将零基第 2 层的 Vs 固定为 `model.vs[2]`；
对应的每个 `starts_vs` 分量必须匹配。输出保留全部层，梯度终止条件仅针对自由参数。
该选项目前用于 `invert/synthetic`，HMC/NUTS 会显式拒绝，见上述地壳对照报告。

一条命令完成构建、测试、六组基阶/多模态及噪声对照、绘图与观测重放：

```bash
python scripts/run_inversion_example.py --output results/m3
```

绘图需要 NumPy、Matplotlib；核心无需它们。参数、数学定义和实际收敛局限见
[M3 验收报告](docs/m3-validation.md)。反演结果没有后验可信区间，触界不代表数据可靠约束。

固定层数的扩展回归及算法变更见 [M4d 验收](docs/m4d-validation.md)：

```bash
python scripts/benchmark_fixed_lm.py --extended --output results/fixed-lm
```

## 固定层数 HMC

```bash
build/dispweave synthetic --config examples/hmc/synthetic.toml --output results/hmc-data
build/dispweave hmc --config examples/hmc/synthetic.toml \
  --data results/hmc-data/observations.txt --covariance results/hmc-data/covariance.txt \
  --output results/hmc
python scripts/plot_hmc.py results/hmc
```

同一 `hmc` 命令可读取实际观测。先验在配置中明确指定为物理 Vs 或 log Vs 均匀分布；
当前固定层顶，支持固定物性、固定 Vp/Vs 及 `vs2model` 经验关系。质量矩阵是无约束坐标中**动量的协方差**。
LM 正则化不自动转成采样先验；采样配置中须关闭该惩罚并明确指定概率先验。
输出包含每条链样本、预热标记、接受率、能量异常和根/梯度失败；无效初值返回 1，
正式采样出现异常或零接受返回 5。返回 0 不表示已经证明收敛。
后验条件于给定参数化，不包含层数不确定性；ESS/R-hat 与后验预测已在 M6a 接入；质量矩阵适应已在 M6b 接入；NUTS 使用下述独立命令。
数学定义、解析测试、独立 MCMCLib 对照与限制见 [M5 验收](docs/m5-validation.md)。
生产构建不依赖 MCMCLib 或其他采样框架。

`[hmc].metric_adaptation` 可选 `none`（默认）、`diagonal`、`dense`。
适应仅发生在预热窗口，正式采样冻结质量矩阵和 nominal step。
示例见 `examples/hmc-adaptive/synthetic.toml`，算法和效率对照见 [M6b 验收](docs/m6b-validation.md)。

## 固定层数 NUTS

```bash
build/dispweave synthetic --config examples/nuts/synthetic.toml --output results/nuts-data
build/dispweave nuts --config examples/nuts/synthetic.toml \
  --data results/nuts-data/observations.txt --covariance results/nuts-data/covariance.txt \
  --output results/nuts
python scripts/posterior_diagnostics.py results/nuts --output results/nuts-diagnostics.json
build/dispweave predict --run results/nuts --output results/nuts-predictive --draws-per-chain 200
python scripts/plot_predictive.py results/nuts-predictive
```

默认采用 Hoffman–Gelman 2014 的 slice NUTS；可选 Stan 风格现代内核，见
[实现与效率对照](docs/nuts-modernization.md)。在 `[nuts]` 中设置
`kernel = "multinomial"`、`find_initial_step_size = true`，启用加权候选、广义掉头检查和
warmup/质量更新后的步长搜索；旧配置默认不变。与静态 HMC 共享目标密度、边界变换、
步长和分窗口质量适应。配置使用 `[nuts]`，`max_depth` 默认 10、范围 1–20。
NUTS 不接受 `leapfrog_steps` 或 `step_jitter`。默认质量适应仍为 `none`，示例显式选择 `diagonal`。
可选 `metric_shrinkage = 25.0` 在质量适应窗口之间保留协方差信息，默认 0；
见 [Stan / NumPyro / BlackJAX / PyMC 对照与限时试验](docs/nuts-implementation-audit.md)。
短测改善了现代内核的预热进度，尚未完成地壳正式后验验收。
另有默认关闭的 `conservative_warmup = true`（温和步长重启）与
`metric_max_stiffness_ratio = 4.0`（局部刚性检查，需质量适应及物理 Vs 高斯先验），
实现、额外 Jacobian 成本和检查点短测见 [预热实验选项](docs/nuts-warmup-policies.md)。
后续 [原始初值在线检查](docs/nuts-online-warmup-audit.md) 发现全程温和策略拖慢启动，
目前不建议据检查点短测启用为默认策略。
可另设 `conservative_metric_restart = true`，同时保持 `conservative_warmup = false`，
仅对接受质量更新后的重启使用温和策略；见 [独立重启验证](docs/nuts-metric-restart-audit.md)。
后续 [固定矩阵的步长目标对照](docs/nuts-step-size-audit.md) 中，目标接受率 0.90/0.95
比 0.99 显示更高的短链 ESS/成本；尚未验证从原始初值出发的完整流程，不据此切换默认配置。
新种子的 [4×1000 状态确认](docs/nuts-step-size-confirmation.md) 中，现代 0.90 与 slice 0.95
均通过固定矩阵条件诊断；不同量的效率排序不同，仍需验证原始初值的完整适应。
随后 [原始四初值 0.95 限时验证](docs/nuts-original-095.md) 三链完成、一链超时；原慢链改善，
但最后一次矩阵更新后出现新的慢链，尚未通过完整流程验收。
[固定慢链矩阵的步长重启对照](docs/nuts-terminal-step.md) 中，搜索改善限时进度，但正式每状态
仍约 50–59 次评估；三组均未完成全部重复，尚不能比较 pooled ESS/成本。
随后 [更新前后矩阵隔离对照](docs/nuts-metric-pair.md) 八次全部完成：新矩阵每状态更贵，
但最低 bulk ESS/总评估约为旧矩阵的 7 倍，tail 效率基本相当；不能仅据步长小就判定退化。
最新 [原始四初值完整验收](docs/nuts-original-095-search.md) 中，slice 0.95＋合理步长搜索
完成 4×1000 正式状态并通过数值、混合与预测检查；含预热的最低 bulk/tail ESS/评估较原
0.99 相同样本数提高约 2.30/3.79 倍。推荐用于本算例后续运行，库级默认值和原后验保持不变。
输出保留全部正式状态，trace 同时包含预热标记、树深、树深截断、最大绝对能量误差与失败类型。
正式采样有发散、无效计算、树深截断或整链未移动时返回 5，结果仍保存；返回 0 不保证充分混合。
诊断脚本额外汇总树深和梯度评估代价。NUTS 的 `accepted` 是候选替换指示，不是额外 Metropolis 检验。

算法细节、独立分布对照、漏斗/多峰失败案例和覆盖率局限见 [M6c 验收记录](docs/m6c-validation.md)。
当前仅反演固定层顶下的 Vs，后验不包含模型层数选择的不确定性。

HMC/NUTS 可显式设置 `prior = "reference_gaussian_vs"` 与逐层 `prior_sigma_vs`，
采用以 `model.vs` 为中心、在速度边界内归一化的物理 Vs 高斯先验。
LM 正则化须关闭，避免重复计入约束。三层示例为 `examples/nuts/reference_gaussian.toml`；
测度、测试与地壳四链实验见 [参考先验 NUTS 验证](docs/crust-nuts-validation.md)。
可选 `record_failures=true` 保存失败试探点以便重放。
可选 `mode_support="trapped_modes"` 显式要求观测的受俘获模态存在：只有完整受俘获域的
独立计数收敛、与已找到根数一致并确认所需阶次缺失时，才按零似然的支持域拒绝处理；
未解决的正演、病态截止端点和计数预算耗尽仍记为数值失败。默认 `strict_forward` 保留原行为。
支持域拒绝单独统计，不算数值发散；其轨迹能量误差仍保留为无穷大。
参考高斯先验下可选 `initial_metric="start_gauss_newton"`，在各链原初值处设置数据与先验的
局部预条件矩阵，不能同时提供 `mass`。它不改变初值、先验或采样目标。
NUTS 可选 `record_progress=true` 逐步刷盘预热和正式轨迹。
详见 [地壳 NUTS 收尾验证](docs/crust-nuts-closeout.md)。

## 多链诊断与后验预测

```bash
python scripts/posterior_diagnostics.py results/hmc --output results/hmc-diagnostics.json
build/dispweave predict --run results/hmc --output results/predictive --draws-per-chain 200 --seed 2026
python scripts/plot_predictive.py results/predictive
```

诊断分别输出 rank/folded split R-hat、bulk/tail ESS、均值 Monte Carlo 标准误、参数相关性与采样异常。
预测区分无噪声理论曲线和加入完整协方差噪声的模拟观测；预测失败不删行。
未触发诊断阈值不代表全局混合；参数后验仍条件于给定层数与层顶。
定义、参考实现对照与 M5 实际诊断见 [M6a 验收](docs/m6a-validation.md)。

## 局部灵敏度与分辨诊断

```bash
build/dispweave resolution --config examples/resolution/reference.toml --output results/local-resolution
python scripts/run_resolution_example.py --output results/m4a
```

在固定 Vp、密度、界面的参考模型处，输出白化且按声明参数尺度缩放的奇异值、
数值秩、给定岭强度下的有效维度和模型分辨矩阵。支持完整观测协方差。
数值秩不等于可辨识层数，分辨矩阵不等于后验协方差；本项未自动选取层数或正则化。
见 [M4a 数学定义与验收](docs/m4a-validation.md)。

## 已知边界

支持全固体、各向同性、弹性的一维模型；允许低速层和高速夹层。
提供 Rayleigh 相速度及 Vs/Vp/rho 灵敏度，不支持液层、泄漏波或层厚导数。
缺根可能是物理截止，也可能是搜索未解析，不能据此断言模态不存在。
上游没有提供一般 Rayleigh 根全集的完备性证明。

本次修复了低 Vp/Vs 半空间的搜索下界漏根，保留原版与修改版校验值；
普通分层和部分低速层根已通过独立边界匹配验证。
高频、强对比及复杂根分布的全面认证仍未完成，具体覆盖见验收报告。

## 文档

- [M7 设计](docs/m7-design.md)
- [M7a 验收结果](docs/m7a-validation.md)
- [M7b 验收结果](docs/m7b-validation.md)

- [M6c 验收记录](docs/m6c-validation.md)
- [M6b 验收结果](docs/m6b-validation.md)
- [M6a 验收结果](docs/m6a-validation.md)
- [M5 验收结果](docs/m5-validation.md)
- [M4d 验收结果](docs/m4d-validation.md)
- [M4a 验收结果](docs/m4a-validation.md)
- [M3 验收结果](docs/m3-validation.md)
- [M2 验收结果](docs/m2-validation.md)
- [M1 验收结果](docs/m1-validation.md)
- [现有代码审查](docs/code-audit.md)
- [架构和数学接口](docs/architecture.md)
- [里程碑与验收](docs/milestones.md)
- [采样器实现调查](docs/samplers.md)
- [第三方源码及授权](third_party/README.md)

DispWeave 自有代码使用 Apache-2.0；第三方代码保留其各自许可证和来源声明。

## RJMCMC 频散示例

Python 分析需要 Python >=3.11、NumPy；绘图另外需要 Matplotlib。
C++ 采样器无新增外部依赖。初版固定 Vp/Vs 和密度，允许层数、界面及 Vs 变化。

```bash
build/dispweave synthetic --config examples/rjmcmc/synthetic.toml --output results/rj-data
build/dispweave rjmcmc --config examples/rjmcmc/synthetic.toml --data results/rj-data/observations.txt --covariance results/rj-data/covariance.txt --output results/rj-run
python scripts/rj_diagnostics.py results/rj-run --output results/rj-diagnostics.json
build/dispweave rj-predict --run results/rj-run --output results/rj-predictive --draws-per-chain 300 --seed 2026
python scripts/plot_rjmcmc.py results/rj-diagnostics.json --output results/rj-summary.png --truth-config examples/rjmcmc/synthetic.toml
python scripts/plot_predictive.py results/rj-predictive
```

输出目录必须尚不存在。采样返回 5 表示存在正演/数值失败或整链无接受；
样本、失败模型和逐行根诊断仍会保存，分析命令可以单独运行，不能据此宣称后验已获认证。
后验预测返回 5 时同样保留失败行，绘图拒绝静默丢弃这些行。
`rj_starts` 每项对应一条链；`rjmcmc.warmup` 仅丢弃预热样本，不自适应提议尺度。
层数包含半空间，深度支持 `depth_limit` 限制界面先验，而非截断半空间。

详见 [M7c 当前验收记录及数值限制](docs/m7c-validation.md)。

本地 M7 结果页（需先生成验收实验与图表）：

```bash
python scripts/build_rj_demo_page.py
```

直接打开 `web/m7-demo/dist/index.html`。页面内嵌全部图表，不需要服务器或发布。

M7 后续已修复受俘获 Rayleigh 搜索域外采样污染；独立核查与后验重跑见
[搜索域修复验收](docs/forward-domain-validation.md)。一般层状介质的无根性仍未认证。

## Rayleigh 模态椭圆率（M8a）

```bash
build/dispweave ellipticity --config examples/ellipticity/layered.toml --output build/ellipticity.txt
python scripts/plot_ellipticity.py build/ellipticity.txt --output build/ellipticity.png
```

输出地表无量纲 `|ur|/|uz|`，使用精细根收敛；**不是原始背景噪声 H/V**，
不含旋转方向，不提供解析梯度。缺根、特征函数失败或近极点时返回 2，仍保存所有行。
`examples/ellipticity/pole.toml` 提供故意含极点与零点的诊断示例。
详见 [M8a 验收](docs/m8a-validation.md)。

## P 接收函数正演（M8b）

```bash
build/dispweave receiver --config examples/receiver/layered.toml --output results/receiver-layered
python scripts/plot_receiver.py results/receiver-layered --time-max 25 --output results/receiver-layered.png
build/dispweave receiver --config examples/receiver/low_velocity.toml --output results/receiver-low-velocity
python scripts/plot_receiver.py results/receiver-low-velocity --time-max 25 --output results/receiver-low-velocity.png
```

一维弹性 P-SV 完整边界求解，包含转换和多次波；入射 P 在半空间顶面定义，
水平沿传播方向、垂直向上。慢度 s/km，采样间隔/时移为秒，Gaussian 宽度为 rad/s。
`source` 可提供与输出等长的实数时间样点，默认单位脉冲；水位反褶积后按垂直自反褶积峰值归一化。
输出保存完整频谱、波形、处理参数和失败诊断；输出目录必须新建或为空。
返回 0 表示计算有效，2 表示有数值失败且结果行已保留，1 表示输入或 I/O 错误。
需要检查时间窗与采样收敛；默认 1024×0.1 s 仅是示例设置，不保证适合全部实测资料。
当前没有已验证的接收函数模型梯度。算法、独立验证、参考程序问题和限制见
[M8b 验收](docs/m8b-validation.md)。

## 组合观测与后验接口（M8c）

```bash
build/joint_example
ctest --test-dir build -R '^joint_' --output-on-failure
```

C++ 示例同时评估频散、已提取的模态椭圆率和 P 接收函数，使用同一个固定层数地下模型，
各块独立定义完整协方差、固定或待估计的噪声尺度及其先验。联合目标保留所有归一化项，
模型先验只计一次；调用者必须明确声明块间条件独立假设。
状态为 `log(Vs)` 后接各块待估计的 `log(噪声尺度)`；输出逐块贡献与后验分解。
示例含 14 个观测值、3 个 Vs、2 个噪声尺度，源码见 `examples/joint/evaluate.cpp`。

LM 复用原有有界求解器，一次运行中固定噪声尺度；频散保留核导数，其他块可显式使用带步长检查的有限差分。
HMC/NUTS 接口要求全部观测提供已验证梯度，当前椭圆率/接收函数组合会被拒绝。
详见 [M8c 接口与验收](docs/m8c-validation.md)。M8d 已完成联合反演 CLI、
单观测对照、冲突数据实验和逐块后验预测；不包含联合 HMC 或联合跨维采样。

## 联合反演工作流（M8d）

```bash
build/dispweave joint-data --config examples/joint/synthetic.toml --output results/joint-data
build/dispweave joint --config results/joint-data/observations.toml --output results/joint-run
python scripts/joint_results.py results/joint-run --output results/joint-run/diagnostics.json
python scripts/plot_joint.py results/joint-run --output results/joint-plots
```

`joint` 要求每块显式提供观测值，不会因存在合成配置而替换数据；输出目录须新建或为空。
当前采用自有无梯度 Metropolis，仅在预热中适应提议；正式样本保留拒绝后的重复状态。
可选冻结的 Gaussian 混合独立提议帮助跨峰，使用完整 Hastings 密度修正；
`scripts/prepare_joint_proposal.py` 可从单独试运行构造提议，旧样本不并入新后验。
可选 LM 保持噪声尺度固定，单独报告收敛/停滞，后验链仍使用配置中的分散初值。
每块输出潜在信号及带相关噪声的后验预测，支持独立估计噪声尺度。
图表明确标记采样诊断失败；独立积分检查必须同时通过多链诊断，不能仅凭均值落在 MCSE 容限内验收。
分析需要 Python >=3.11、NumPy；绘图另需 Matplotlib。
完整实验、独立积分方法和已知限制见 [M8d 验收记录](docs/m8d-validation.md)。
