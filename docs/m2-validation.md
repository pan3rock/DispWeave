# M2 灵敏度与目标函数验收

M2 接入 invLM 能量核和隐式导数，建立固定 Vs 参数化、观测协方差、似然和后验目标。
没有加入优化器或采样器。源码版本为 0.2.0，数值后端仍基于 M1 固定的 invLM commit。

## 实现与使用

新增的 `dispersion_kernel.cpp`、`dunkin_derivative.cpp` 及三个配套头文件原样导入，
没有修改上游核函数、容差或求根算法。新增来源记录后，第三方文件校验共 449 个。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
build/dispweave forward --config examples/forward/normal.toml \
  --sensitivity energy --output build/energy.txt
build/dispweave forward --config examples/forward/normal.toml \
  --sensitivity implicit --output build/implicit.txt
build/objective_example
```

不指定 `--sensitivity` 时保持 M1 的仅正演路径和数据列。
指定后追加梯度状态、能量残差、root correction、重构标记、诊断文本，以及
按 `Vs[0..n-1], Vp[0..n-1], rho[0..n-1]` 顺序排列的导数列。
Vs/Vp 导数为相速度相对于 km/s 的导数，密度导数相对于 g/cm³。
能量法不计算 root correction，隐式法不计算能量残差，对应输出为 NaN。

CLI 返回码：0 全部所需结果有效；1 配置/模型/I/O 错误；2 至少一行根不可用；
3 根均有效但至少一行所需灵敏度不可用。参数解析错误保留 CLI11 非零返回码。
请求缺根时行仍保留，梯度为 NaN，不返回伪零梯度。

## C++ 接口

- `Forward::evaluate(model, queries, SensitivityMethod)`：`none`、`energy`、`implicit`。
- `rayleigh_sensitivity(model, f, c, method)`：在已解析相速度上求核，不重写输入相速度。
- `FixedVs`：固定层顶和密度，q=log(Vs)；Vp 固定或 Vp=a Vs。
- `DispersionObservations`：逐行频率/模态、观测相速度、完整协方差。
- `GaussianLikelihood`：Cholesky 白化、归一化似然和整体误差尺度导数。
- `DispersionProblem`：统一数据残差、Jacobian、最小二乘代价和失败状态。
- `log_posterior`：分解返回 log likelihood、log prior、变换项、总值与 q 梯度。

M2 的观测和参数化通过 C++ API 使用；文本观测加载、优化器配置及反演 CLI 留到 M3。
可运行示例在 `examples/objective/evaluate.cpp`，采用三条观测和非对角协方差。
`DispersionProblem` 持有后端引用，后端对象必须比问题对象存活更久。

### 失败与测度约定

`EvaluationStatus` 分为 `valid`、`outside_support`、`forward_failure`、
`gradient_failure`、`numerical_failure`。只有已知参数/先验支持域外返回负无穷；
不可解析的正演或梯度使后验总值为 NaN，并给出状态和说明。
后续采样器必须检查状态，不能把所有失败都作为正常的零概率提议。
配置维度等编程错误抛出 `invalid_argument`，部分底层数值溢出抛出 `runtime_error`；
问题评估对已识别的计算失败提供结构化状态，不承诺捕获任意用户回调异常。

梯度失败不会清除已求出的根或已经可计算的似然值；纯数值评估可独立成功。
能量残差使用上游的 1e-7 阈值；隐式法保留上游的奇异导数与 1e-8 km/s
root correction 检查。`valid` 表示通过这些数值检查，不是对整个邻域可微性的证明。

先验必须声明 `physical_vs` 或 `log_vs`：

- 物理 Vs 测度：加 `sum(q)`，梯度加全 1，并对先验梯度乘 Vs。
- log-Vs 测度：不再添加变换项。
- 最小二乘数据项不自动加入先验或变换项。

提供归一化的独立 Gaussian 和有限区间 uniform 密度函数。
Gaussian 与参数化的硬边界取交集后，目标还可能差一个参数无关的截断归一化常数；
这里返回的是用于固定模型推断的未归一化后验目标，**不能直接用于模型证据比较**。
界面数、先验超参数和支持域变化时必须重新处理这些常数。
均匀先验边界和硬参数边界不是光滑 HMC 坐标；本阶段没有 HMC 适配。

### 协方差与误差尺度

协方差检查有限性、维度、对称性和正定性；只在 1e-12 的相对对称容差内平均两三角。
实际使用的对称矩阵可由 `likelihood().data()` 读取。
`log_scale=s` 定义 `C(s)=exp(2s) C0`，似然保留 `2ns` 归一化项。

\[
\frac{\partial\log L}{\partial s}=r(s)^T r(s)-n.
\]

返回的 `log_scale_gradient` 仅为似然对该尺度的导数，**不包括噪声先验或其变换项**。
尚未实现噪声超参数优化/采样、各模态独立尺度或任意协方差参数的梯度。

## 验证结果

Linux x86_64 / GCC 13.3.0 / Eigen 3.4.1。Release 与启用 UBSan 的 Debug
各 **18/18 CTest 通过**，包含 M1 全部回归。

```bash
cmake -S . -B build-check -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer'
cmake --build build-check -j2
ctest --test-dir build-check --output-on-failure
python scripts/verify_vendor.py
```

新增检查包括：

1. 半空间独立 Rayleigh 三次方程的解析隐式导数，覆盖 Vp/Vs=1.2、1.73、3，
   分别核对能量法和隐式法的 Vs、Vp、rho 导数，绝对容差 1e-8。
2. 普通、低速层和复杂低速层模型的逐层 Vs/Vp/rho 三类导数，多步长中心差分。
3. 固定 Vp 和固定 Vp/Vs 参数化的链式导数；模型编码/解码与边界、结构丢失拒绝。
4. 2×2 非对角协方差的手算似然/白化结果、错误协方差拒绝、噪声尺度值及导数。
5. Gaussian/uniform 先验的归一化、两种测度的后验值及梯度、变换 Jacobian。
6. 支持域外、缺根、错误先验梯度，以及注入核失败时的根/似然保留。
7. CLI 的两种灵敏度输出及独立 C++ 后验示例。

### 多步长差分

步长为被扰动物性值乘相对步长。表中是全部测试行/层/物性的最大绝对误差；
不同物性导数的单位不同，表用于定位数值差异，不作为统一物理分辨指标。
普通与低速层用模态 0、1、2；复杂低速层用 0、5、6，其中后两根接近。

| 模型 | 相对步长 1e-3 | 1e-4 | 1e-5 | 能量核与隐式法最大差 |
| --- | ---: | ---: | ---: | ---: |
| normal | 8.03e-6 | 8.01e-8 | 1.79e-8 | 2.09e-12 |
| low_velocity | 1.62e-2 | 1.66e-4 | 1.79e-6 | 4.35e-14 |
| dense | 6.27e-7 | 2.25e-8 | 2.29e-7 | 5.18e-12 |

验收阈值：能量/隐式绝对差 <2e-5；最细步长下 `abs(FD-J)/(1+abs(J)) <3e-4`。
这是数值误差尺度，不能称为相对误差或条件数。更细差分不一定更准确：
复杂低速层在 1e-5 步长开始受求根精度/舍入误差影响。
两种核法一致也不能替代差分和独立半空间检查。

### 截止与近邻根

测试对普通模型第一高阶的“已找到/未解析”转换点作二分定位，这不是物理截止真值。
在约 0.130710185 Hz，根距离半空间 Vs 约 2.87e-9 km/s，隐式 root correction
约 4.99e-13 km/s。能量与隐式核均通过自身检查，但扰动半空间 Vs 时：

| 绝对步长 km/s | 两侧根均有效 | 差分与隐式导数绝对差 |
| --- | --- | ---: |
| 1e-4 | 否 | 不计算 |
| 1e-6 | 是 | 3.12e-7 |
| 1e-8 | 是 | 3.03e-5 |

稍远的约 0.130723125 Hz，半空间间距约 1.72e-5 km/s，1e-6 步长误差约 1.01e-8。
这些诊断以报告为主，不在跨越缺根的两侧拼接伪导数。
复杂低速层在 100 Hz 的第 5、6 阶约为 0.118897402 和 0.119397482 km/s，
两者均纳入独立逐物性差分，而非只检查根存在。

当前输出包含根修正/能量残差，测试报告附带根间距；尚无对一般层状模型的完整
导数条件数或分支连续性证书。上游只在 `D_c` 非有限/为零等条件下直接拒绝，
不会自动证明所有接近奇异的导数都可靠。

## 可运行后验示例结果

固定三层模型、固定 Vp/密度、三条相关观测和物理 Vs 均匀先验：

```text
log_likelihood=14.0897511678
log_prior_dVs=1.82015894375
log_transform=1.79175946923
log_posterior_dq=17.7016695808
gradient_q=49.8370081924 57.0039065173 16.9972732751
likelihood_log_noise_scale_gradient=-1.5651194227
```

这是一次目标函数评估，不是优化结果或后验样本。

## 灵敏度计算基线

`build/sensitivity_benchmark` 对每个模型的 3 个查询执行一次首调用，再重复 100 次。
时间包含求根和全部物性核；同一进程依次测量各方法，因此首调用不是独立进程冷启动。
同 M1 主机，Release 单线程，本次实测：

| 模型/方法 | 首次 ms | 重复平均 ms |
| --- | ---: | ---: |
| normal / energy | 0.087874 | 0.016484 |
| normal / implicit | 0.025923 | 0.013758 |
| dense / energy | 0.168665 | 0.060907 |
| dense / implicit | 0.047453 | 0.039821 |

两种方法请求完全相同的根，结果均有效。该小型基线不证明全部模型上的速度关系。
当前适配层按请求行建立核工作区，尚未优化重复初始化；M2 优先验证正确性。

## 下一步与边界

M3 将以这里的残差/Jacobian 接入固定层数 LM，补齐合成与实测观测文件入口、
失败试算拒绝、终止诊断、多初值和 Python 绘图。
M2 不反演层厚、Vp 或密度，不包含自适应分层、HMC、RJMCMC 或联合观测。
M1 的根枚举完备性限制仍然存在；导数正确性只在已经隔离的平滑模态分支上成立。
