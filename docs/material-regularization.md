# 经验物性关系与显式 LM 正则化

2026-10-23。按用户要求，复用 invLM 已有实现，补充 Vp/密度策略与常用正则化。
去掉模型选择准则不意味着去掉正则化；前一轮 [A/B/C](crust-depth-controls.md) 为隔离深部参数影响
而没有加正则项，其结果保留。本项独立提交，不改变那个对照的设置和结论。

## 物性关系

`[inversion] vs2model` 明确指定策略，输入 Vs/Vp 为 km/s，密度为 g/cm³：

| 名称 | Vp | 密度 |
|---|---|---|
| `fixvprho`（默认） | 参考模型固定值 | 参考模型固定值 |
| `constant_ratio` | `vp_vs_ratio * Vs` | 各层参考密度固定 |
| `brocher05` | invLM 的 Vs 四次多项式 | invLM 的 Vp 五次多项式 |
| `gardner` | `vp_vs_ratio * Vs` | `gardner_alpha * (1000*Vp)^gardner_beta` |
| `nearsurface` | `vp_vs_ratio * Vs` | invLM 的 `-0.22374079 Vs² + 1.32248261 Vs + 1.54840433` |
| `constant_rho` | `vp_vs_ratio * Vs` | `constant_density`，所有层相同 |

默认经验关系参数：Vp/Vs=1.7321，Gardner α=0.31、β=0.25，constant_rho 密度=2。
旧的 `vp_vs_ratio` 非零且没有 `vs2model` 时仍表示固定各层密度的 constant_ratio；零仍表示固定 Vp/ρ。
Brocher05 与额外指定 Vp/Vs 冲突时拒绝输入。经验关系不需要用户手动同步输入 Vp/ρ：
参考模型和每个试探模型都会重算它们，最终模型文件保存实际 Vp/ρ，元数据记录关系与参数。
constant_ratio 的旧参考 Vp 一致性检查保持不变。

关系应用于包括半空间在内的全部层；固定某层 Vs 时，相应派生物性也固定。
生成模型逐次检查正值、有限值和正体积模量；不截断多项式输出。
这些检查不等于验证经验关系适用于任意岩性、深度或速度范围，派生物性的不确定性尚未独立采样。

`FixedVs` 的 q=log(Vs) 参数化、频散似然、固定维度 HMC/NUTS 及后验预测共用该关系。
完整链式导数为

```text
dc/dq_j = Vs_j * [dc/dVs_j + dc/dVp_j * dVp_j/dVs_j + dc/dρ_j * dρ_j/dVs_j].
```

未遗漏密度项，也不把独立 Vp/ρ 核当成总导数。密度随 Vs 变化而调用者没有提供密度偏导时显式报错。
联合反演 CLI 的密度偏导还未连接，`parameters.vs2model` 会被拒绝；既有固定物性/固定比值联合流程不变。
`forward` 仍直接接收完整物理模型，`vs2model` 属于反演参数化，不能期待它修改单独 forward 输入。
物性确定映射不新增概率自由度；physical-Vs 先验仍只使用 Vs→log(Vs) 的 Jacobian。

## 正则化数学定义

LM 最小化 `F=0.5 ||L⁻¹(c(m)-d)||² + R`，`LLᵀ=观测协方差`。
令 v=Vs、v₀=配置中的参考 Vs、z 为层顶深度：

| type | R |
|---|---|
| `none` | 0，lambda 必须为 0 |
| `reference_damping` | `lambda/2 * Σ(v_j-v₀_j)²` |
| `perturbation_smooth` | `lambda/2 * Σ[(v_{j+1}-v₀_{j+1})-(v_j-v₀_j)]²/(z_{j+1}-z_j)` |
| `vs_smooth` | `lambda/2 * Σ(v_{j+1}-v_j)²/(z_{j+1}-z_j)` |

相邻差分包含最深有限层与半空间。固定参数不进入优化向量，但与自由邻层的正则化联系仍保留。
使用 invLM 原生残差与解析 Vs Jacobian，再乘 diag(Vs) 转为 q Jacobian。
平滑项按**层顶间距**归一；这是 invLM 的定义，与 QEDispInv 的参考界面对比加权/目标归一方式不同。
不能把 QEDispInv 的 lambda 数值原封不动当成相同正则化强度。

lambda 是代价权重，不是残差倍率；平方根进入残差。
参考阻尼的 lambda 单位为 `(km/s)^-2`，平滑项为 `km/(km/s)^2`，使 R 无量纲。
固定 lambda=100 的参考阻尼对应 0.1 km/s 的高斯偏差尺度，但这里作为显式确定性惩罚使用。
权重必须由配置给出；没有自动选权重、L 曲线或层数选择。
默认关闭，既有配置的无正则化运行结果保持不变。

```toml
[inversion]
# 仍需速度边界与 starts_vs；边界支持逐层数组或 vs_min/vs_max/vs_width
vs2model = "brocher05"

[inversion.regularization]
type = "reference_damping"
lambda = 100.0
```

`runs.txt`/原 trace 的 cost 是总目标，新增 `costs.txt` 和 `run_*_cost_trace.txt` 分列数据与正则化代价。
高斯 log likelihood 保持纯数据定义。HMC/NUTS 不会把 LM 惩罚静默当成概率先验：
带有效 LM 正则项的采样配置会报错，需另行明确先验。经验关系本身可用于采样与预测。

## 复用与验证

新增四个随仓库提供的 invLM 文件，来源 `86ad6fa`，与本地 `5fd7f27` 中对应文件完全一致：
`model_builder.hpp/.cpp`、`regularization.hpp/.cpp`。
物性公式与导数直接复用；正则化只移除原生 Config 适配构造器，保留 Settings 接口与数值实现。
不引入 Ceres 或新安装依赖，原始/修改后哈希记录于 provenance.json。

新增验证覆盖：六种物性策略的模型与物性导数、包含密度链的频散与 log posterior 多步长 FD；
不符合弹性条件的外推、缺失密度偏导；非均匀间距下三种惩罚的独立手算代价与 FD；
有固定坐标的解析正则化最优解；零权重精确重放；CLI 生成物性、代价分解与 HMC/预测重放。
HMC 的新增短运行是接口验证，不是多链混合质量验收。

完整 Release CTest **66/66**，39.67 s；第三方哈希 **453/453**。
针对本次改动的 UBSan 检查 **3/3**，6.05 s，覆盖物性/正则化、LM 工作流及短 HMC/预测重放，
日志保存于 `results/crust-material-control/ubsan.log`。扩展 UBSan 批次的 HMC 工作流也通过，
NUTS 工作流因调试构建耗时较长而主动停止，未计为通过；该批次日志单独保留。

完整示例 `examples/inversion/brocher_regularized.toml` 使用 Brocher05、扰动平滑 lambda=10、
固定种子 20261023 的含噪数据：3/3 初值收敛，数据 cost≈3.9029404、正则 cost≈0.00425488，
总 cost≈3.90719528。没有放宽收敛阈值。

## 地壳四组对照

沿用 C 组的 34 层、33 个自由 Vs、100 km 起固定半空间；相同数据、σ=0.02 km/s、
四个对应初值、边界、200 次预算。仅比较物性关系及是否加入 reference_damping，lambda=100。
该权重在查看运行结果前明确指定，未校准。固定物性/无正则组的 `runs.txt` 与 C 组逐字一致。

| 物性 / 正则化 | 收敛数 | 其余结果 | 收敛模型数据 cost / 正则 cost | 相速度 RMS km/s |
|---|---:|---|---|---:|
| 固定 / 无 | 0/4 | 3 次到限，1 次初值未解析 | — | 最低末态约 0.015754 |
| 固定 / 参考阻尼 | 3/4 | 1 次初值未解析 | 46.568596 / 1.901477 | 0.017264 |
| Brocher05 / 无 | 0/4 | 1 次停滞，2 次到限，1 次初值未解析 | — | 最低末态约 0.015685 |
| Brocher05 / 参考阻尼 | 3/4 | 1 次初值未解析 | 49.559888 / 1.256171 | 0.017810 |

六个收敛终态全部通过 h=1e-6 的全自由参数、全观测 FD 检查（Brocher 同时扰动 Vp/ρ）。
每个正则化组的三个有效初值得到一致的目标值。对提供模型的 0–100 km 深度加权 Vs RMS 差
分别约 0.0309、0.0291 km/s；这是加入参考约束后的结果，不能解读成数据独立提供了这些精细结构。
正则化以略高数据残差换取参考约束；它既没有增加数据分辨率，也没有自动确定合理权重。

指定半空间漏根诊断在新程序上仍返回 **2**，保存在 `results/material-root-regression/`。
这里的收敛只针对上述模型、初值与显式惩罚，不能外推为正演全域可靠，也不等于 invLM 原生 Ceres
程序已完成同等应用测试。其原生完整构建仍需 Ceres；本轮没有修改 invLM 主仓库。

## 复现

```bash
cmake -S . -B build-properties -DCMAKE_BUILD_TYPE=Release
cmake --build build-properties -j2
ctest --test-dir build-properties --output-on-failure
build-properties/dispweave synthetic --config examples/inversion/brocher_regularized.toml \
  --output results/brocher-demo-new
MPLCONFIGDIR=/tmp/dw-mpl python scripts/plot_inversion.py results/brocher-demo-new
MPLCONFIGDIR=/tmp/dw-mpl python scripts/benchmark_crust_materials.py \
  --case results/crust-depth-control/C_merged_tail --exe build-properties/dispweave \
  --output results/material-control-new --regularization reference_damping --lambda 100
```

本次输出为 `results/brocher-regularized-demo/`、`results/crust-material-control/`，含全部失败、
完整模型和代价、导数审计、PNG/PDF 对照图。案例源准备方式见 A/B/C 报告。
