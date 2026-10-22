# M8c：组合观测、独立噪声尺度与统一后验

状态：核心接口、数值验证及 C++ 示例完成。M8d 的单观测/联合后验实验、冲突数据、
后验预测与先验敏感性仍待验收，M8 整体尚未完成。

## 接口与职责

| 接口 | 职责 |
|---|---|
| GaussianData / GaussianDataLikelihood | 任意有符号观测值与正定完整协方差；白化、归一化 Gaussian 似然及尺度导数 |
| ObservationBlock | 具名物理量、单位、预测回调、梯度声明、数据协方差与独立噪声策略 |
| dispersion_block | invLM 多模态相速度；保留 energy/implicit 核函数及逐行诊断 |
| ellipticity_block | 明确声明的 Rayleigh 模态椭圆率；当前无已验证模型梯度 |
| receiver_block | M8b 上行 P 接收函数，固定处理网格上的指定样点；当前无已验证模型梯度 |
| JointLikelihood | 给定同一 LayeredModel 和各块尺度，评估数据项，不添加任何先验 |
| JointProblem | 固定 Vs 参数化映射一次，共享同一个地下模型；提供数据目标和 LM 局部 Jacobian |
| joint_log_posterior | 将模型先验、独立尺度先验及坐标变换各计入一次，输出完整分解 |
| joint_density | 为 HMC/NUTS 提供值与可靠梯度；不支持的组合在创建入口时明确拒绝 |

通用 Gaussian 计算从原频散专用类中抽出；旧 GaussianLikelihood、DispersionProblem、
log_posterior 接口保持兼容。旧接口仍把噪声尺度作为外部给定值；新接口才包含噪声尺度先验。
FixedVs 的 `model_jacobian` 统一处理任意观测的物理 Vs/Vp 偏导与参数链式法则；原频散接口也复用它。
没有复制 LM/HMC/RJMCMC 算法，没有新增生产依赖。

JointLikelihood 直接接受模型，给后续其他参数化或独立 RJMCMC 留出数据目标接口；
本项没有声称已完成联合 RJMCMC，也没有改变原 RJ 状态和提议。
频散回调捕获的 Forward 后端必须比观测块及其副本存活更久；其余数据与处理参数由块持有。

## 观测与缺测契约

频散为 km/s，模态从 0 开始，频率 Hz；不同模态、不同块可使用不同查询网格，
顺序由观测行决定。椭圆率与归一化接收函数均无量纲，但具有不同物理含义和独立处理约定。
椭圆率必须显式传入 `rayleigh_modal_ellipticity` 标签，`ambient_hv` 等其他标签被拒绝；
调用者仍须确认数据确为已提取的模态椭圆率，标签不是波场识别算法。

接收函数观测用唯一、可乱序的 sample_indices 指定，时间为 `i*dt-pre_arrival`；
原始完整频谱/波形仍由 M8b 算法计算。水位、源、滤波、时移及归一化定义与 M8b 相同。
只提供实际观测到的频率/样点，缺测应在创建数据块前明确表达；数据中的 NaN 被拒绝。
拟合时不根据某个试算模型是否成功重新选择观测。任何请求预测失败，联合似然整体失败，
保留各块结果与消息，不能通过少拟合几个数据获得更高似然。

协方差单位是对应数据单位的平方；s 是无量纲标准差倍率，不是方差倍率。
每块可使用非对角协方差。组合时必须显式选择 `BlockDependence::conditional_independence`，
即给定模型/尺度后的块间条件独立假设。未声明时拒绝构造；不把跨块相关协方差默认为零。
有跨观测误差相关性时，需要包含相关项的联合物理预测/协方差块，当前三个适配器不自动建立它。

## 数学定义与先验测度

第 b 块基准协方差 `C0_b=L_b L_b^T`，`eta_b=log(s_b)`：

```
r_b = exp(-eta_b) * L_b^-1 * (g_b(m)-y_b)
log L_b = -0.5*r_b^T*r_b - 0.5*log|C0_b| - n_b*eta_b - n_b/2*log(2*pi)
d log L_b / d eta_b = r_b^T*r_b - n_b
log L_joint = sum_b log L_b
```

不能只最小化残差来估计噪声；`n_b*eta_b` 与协方差归一化完整保留。
状态固定为 `x=[q=log(Vs), inferred eta in block declaration order]`。
固定尺度不占状态维度，也没有尺度先验或变换项。

每个块独立选择：

- `NoiseScale::fixed(s)`：固定正值尺度，默认 1。
- `NoiseScale::lognormal(mu,sigma)`：s 的归一化 lognormal 先验，等价于 eta 的 Normal(mu,sigma)。
  结果的物理尺度 log prior 含 `-eta`，尺度变换含 `+eta`；两者抵消后 eta 梯度为 `-(eta-mu)/sigma²`。
- `NoiseScale::exponential(rate)`：s>0 上的归一化指数先验；物理尺度 log prior 为
  `log(rate)-rate*exp(eta)`，尺度变换为 `+eta`，eta 梯度为 `1-rate*exp(eta)`。

模型先验沿用明确的 physical_vs / log_vs 测度。前者加入 `sum(q)`，后者不加模型变换项。
`joint_log_posterior` 输出 log_likelihood、log_model_prior、log_noise_prior、log_transform、总值和可用梯度。
模型先验仅调用一次，不随块数增加；没有把固定参数化后验声称为模型维度不确定性。
正规化的噪声先验不自动证明任意数据/模型的后验适定性，特别是可精确拟合的数据与
接近零尺度仍有有限密度的先验组合；M8d 须检查实际实验的后验行为。

数值失败保留 NaN 目标和显式失败状态，与参数/先验支持域之外的 `-infinity` 区分。
预测器不能把不可用物理结果声明成先验支持域拒绝。超参数溢出、错误梯度、缺行或非有限值均有失败诊断。

## LM 与梯度能力

固定 Vp 时 `J_q=J_Vs*diag(Vs)`；固定 Vp/Vs=ratio 时
`J_q=(J_Vs+ratio*J_Vp)*diag(Vs)`，随后对应各块协方差/尺度白化。
`JointProblem::least_squares(x)` 默认只接受已验证梯度；开启有限差分时：

- 已声明可靠梯度的频散块继续使用核函数；核失败不能静默改用差分。
- 只为尚未声明可靠梯度的块计算差分。默认 q 步长 1e-5，比较 h 与 h/2 的二阶结果。
  内部点用中心差分，接近界限时使用可行方向二阶单边差分；h 最多为支持宽度的 1/4。
- 每个元素要求 `|J_h-J_h2| <= 1e-5 + .005*max(|J_h|,|J_h2|)`。
  失败报告参数、观测行与两级估计；不返回伪零或部分有效的完整 Jacobian。
- 返回较细步长结果和批次评估数。差分一致性是局部检查，不保证全局可微、分支连续或可供 HMC 使用。

LM 的 Jacobian 列仅含 q，噪声尺度在一次 LM 运行中必须保持固定；复用原 bounded_lm 即可。
不能把噪声尺度当成普通最小二乘参数，从而漏掉 log determinant 项。
三观测组合的 `joint_density` 明确拒绝 HMC/NUTS；只有所有块均声明且实际返回可靠梯度时才允许进入。
直接请求不支持的梯度仍保留有效正演与目标值，但返回 gradient_failure 和空梯度。

## 实际验收

测试包括：两块手算非对角协方差与独立尺度；噪声先验/变换/全状态梯度中心差分；
模型先验调用次数；改变单位后的密度变换；换序不变性；固定 Vp 与固定比值链式法则；
单频散块与旧目标/后验一致；三类真实正演的不同采样网格和联合 LM；
边界单边差分、步长不一致、扰动模型失败、已验证核失败、链式法则溢出及缺测/错误状态传播。

三观测例：3 个 Vs 参数、4 个频散值、3 个椭圆率值、7 个 RF 样点。
步长 1e-3 和 1e-4 在 RF 的一个弱导数上未通过检查；1e-5 与 h/2 的最大差为 2.306e-6，
改用 5e-6 后 Jacobian 最大变化为 5.775e-7；1e-6 的两级最大差为 5.181e-8。
没有放宽差分容限。3 参数的内部点一次 Jacobian 需 13 次整组观测评估；这是额外成本的明确基线。
无噪声合成数据，从 Vs 真值的 .97 倍出发，固定尺度 LM 在 5 次求解器目标评估后收敛，
白化最小二乘代价约 1.40e-20。此例验证接口与求解器集成，不是独立的反演分辨率或后验覆盖率结论。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
build/joint_example
```

`examples/joint/evaluate.cpp` 使用确定性观测扰动、完整频散协方差、两种独立噪声先验，
输出三个块的贡献和目标分解。实际得到 log likelihood=47.8226506036，模型 log prior=1.82015894375，
噪声 log prior=-1.22579135264，log transform=1.79175946923，总 log posterior=50.2087776639。
LM Jacobian 为 14×3，HMC_gradient_available=0。示例 RF 使用 1024 样点完整处理网格。

Release 完整 CTest 57/57 通过（7.34 s）；UBSan 联合接口与旧目标定向 10/10 通过（19.79 s）。
补充已验证梯度失败不可静默改用差分、以及链式法则/差分比较溢出保护后，
Release 联合接口定向 6/6 再次通过（.33 s），UBSan 定向 6/6 再次通过（15.67 s）。
