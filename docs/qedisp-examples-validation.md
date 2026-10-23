# QEDispInv 示例应用测试

后续：默认 Dunkin 的已定位近邻根漏检已修复，见 [修复与重跑记录](clustered-root-repair.md)。
下文保留修复前结果；修复后地壳反演仍未整体收敛。

2026-10-23，使用本地 `/home/lei/Work/QEDispInv` 中的合成示例。
这是 M8 后的应用验证，不新增算法里程碑，也不是实测数据验收。
**浅层多阶和匹配的 m1 例子通过固定参数化 LM/NUTS 测试；地壳反演未通过，
发现停滞点附近的根枚举不连续。** 本轮提交测试脚本、输入契约测试、图表工具和记录，未修改物理后端。

## 来源与问题定义

QEDispInv HEAD 为 `ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa`。
已有 `bin/forward` 可用，其 SHA-256 为
`4daa58b63fe0428edc170c46f3db9536f110c476307221fe2ccc4ec077db5910`；
不能据此断言这个已有二进制由当前 HEAD 构建。
未修改其源码、数据、构建或原有三个 dirty submodule。
外部输入仅复制到忽略的 `results/`，没有将 QEDispInv 代码或数据加入 Git/编译依赖。

模型列为 `index depth_km rho_g_cm3 Vs_km_s Vp_km_s`，末层为半空间；
观测列为 `frequency_hz phase_km_s mode`，模态从 0 开始。
地壳模型存在重复显示序号，按深度保持物理顺序，不重新排序或合层。
三份数据均未提供误差列。

| 测试名 | 外部目录 / 模型文件 | 层数 | 观测数 / 模态 | 频段 Hz | 假设 σ (km/s) |
|---|---|---:|---|---|---:|
| nearsurface | `demo/syn-nearsurface/model_data.txt` | 5 | 80 / 0–3 | 6–96 | 0.005 |
| crustmantle | `demo/syn-crustmantle/model_data.txt` | 44 | 125 / 0–3 | 0.01–0.4 | 0.02 |
| lvl-l4 | `examples/lvl-l4/model_data.txt` | 4 | 50 / 0 | 0.1–30 | 0.002 |
| lvl-l4-m1 | 同目录 `m1.txt`，同一份 50 点数据 | 5 | 50 / 0 | 0.1–30 | 0.002 |

先核对文件配对：`lvl-l4/data.txt` 与配置指向的四层低速层模型不符，相速度 RMS 差为
`0.0507446 km/s`；同目录五层 `m1.txt` 的 RMS 差仅 `1.02e-7 km/s`。
因此补测数据一致的 m1，同时保留原配对失败记录。
这是数值配对证据，不是已查明的原始数据生成历史；m1 单调增速，不能将其成功称为低速层恢复。

所有测试固定所选模型的界面、Vp、密度，仅反演 Vs；高斯独立误差使用上述显式假设。
边界为 `0.65 Vs_ref` 到 `min(1.35 Vs_ref, 0.98 sqrt(3/4) Vp)`。
四个初值分别为 `0.9 Vs_ref`、`1.1 Vs_ref`、
`Vs_ref*(1+0.08 sin(j+1))`、`Vs_ref*(1-0.08 cos(j+1))`。
这是围绕已有模型的合成恢复实验，不是未知结构的盲测；提供的模型也不能未经核对就视为数据真值。

QEDispInv 原浅层/地壳配置使用随机分层、多初值、模态权重和光滑正则化；地壳还通过
Brocher 关系更新 Vp/密度。其原确定性目标的模式权重并非本测试的高斯误差权重。
因此本轮验证“相同观测上的 DispWeave 固定参数化”，没有复现或比较完整 QEDispInv 反演算法。

## 正演与核函数

四种配对共 **305 次参考查询**，对应 255 个不同的输入数据行；额外 50 次来自 m1 配对。
所有参考查询都返回根及能量/隐式核，逐行保留原频率、模态和顺序。
QEDispInv 可执行程序会省略未求出的根，本脚本要求数量和查询顺序完全匹配才进行整组比较。

| 模型 | 与 QEDispInv 正演的最大差 (km/s) | 与输入观测的 RMS 差 (km/s) | log Vs 导数 FD 最大绝对差，h=1e-4 |
|---|---:|---:|---:|
| nearsurface | 4.980e-8 | 0.00222911 | 6.921e-8 |
| crustmantle | 4.958e-8 | 0.0185671 | 4.593e-8 |
| lvl-l4 | 4.974e-8 | 0.0507446 | 5.338e-8 |
| lvl-l4-m1 | 4.788e-8 | 1.020e-7 | 7.939e-9 |

QEDispInv 输出速度到小数点后 7 位，参考正演比较容限为 `1e-7 km/s`。
两个实现具有共同数值来源，因此此项是交叉回归证据，不能替代独立物理验证。
独立参考的既有验收仍见 [M2](m2-validation.md) 和相关正演测试。

核函数在固定 Vp/密度下用 `Vs_j exp(±h)` 验证，对照 `dc/dlog(Vs_j)=Vs_j dc/dVs_j`。
h 取 `1e-3, 1e-4, 1e-5`，完整记录三个步长，最细步长的检查条件为
`|FD-K| <= 1e-7 + 1e-3 max(|FD|,|K|)`。
四个参考模型均通过；最细步长的最大归一化误差分别为 0.4865、0.4989、0.2483、0.4997。
部分小导数已出现求根精度平台，缩小步长不再持续改善。
能量核与隐式核在参考模型处最大差小于 `1.4e-11`，但二者一致不证明沿模型空间的分支连续。

## 固定参数化 LM 与暴露的问题

隐式灵敏度，最多 200 次迭代，投影梯度阈值 `1e-5`，初始阻尼 `0.001`。
没有删数据、放宽收敛阈值或自动改变层数。

| 模型 | 四次运行结果 | 终态相速度 RMS (km/s) |
|---|---|---:|
| nearsurface | 4/4 收敛 | 0.000902206 |
| crustmantle | 3 次停滞，1 次初值正演失败 | 停滞分别 0.0459172、0.110872、0.0161566 |
| lvl-l4 | 4/4 停滞，均有一个参数触界 | 0.0105457 |
| lvl-l4-m1 | 4/4 收敛 | 6.807e-8 |

浅层 LM 的 Vs 约为 `[0.40346,0.20009,0.29803,0.60182,0.62424] km/s`。
第四层偏离所提供模型的 0.5，不能把低残差解释为结构完全恢复。
源观测在 f=6 Hz、mode=0 为 0.476，而所提供模型预测约 0.494113；该差异保留，没有当作异常值删除。
m1 的恢复与提供模型最大 Vs 差约 `6.02e-6 km/s`。

地壳 run 3 初值有四个未解析查询：mode 2 的 0.1612、0.1692、0.1771 Hz，以及 mode 3 的
0.2488 Hz。报告 `invalid_initial_evaluation`，不将“未解析”断言为物理不存在。
其他三次终态梯度仍很大（约 0.999、0.993、0.573），不是可接受收敛。
run 2 的试探步长降至 `3.29e-12` 时，目标值仍从 40.7867 跳至 87.5926。

对三个停滞模型重新做多步长 FD，全部失败，即使根及梯度状态仍为 `valid`。
最小的可复现线索位于 run 2，f=0.392 Hz，mode 3，扰动零基参数 Vs[15]：

| log Vs[15] 扰动 | DispWeave mode 3 | QEDispInv mode 3 | DispWeave mode 4 | QEDispInv mode 4 |
|---|---:|---:|---:|---:|
| -1e-5 | 3.937654171 | 3.9376542 | 3.937716230 | 3.9377162 |
| 0 | 3.937655406 | 3.9376554 | 3.937716231 | 3.9377162 |
| +1e-5 | **4.170184952** | **3.9376566** | **4.317773053** | **3.9377162** |

FD 导数约 11626.54，隐式导数约 0.123403；与 QEDispInv 相比，当前后端漏过了近邻根，
随后的根获得了较低的模态编号。此证据高度指向根枚举/分支连续性问题。
近邻根的物理身份仍需要独立特征函数或根计数验证；不能仅因 QEDispInv 返回它们就保证正确。
也不能因隐式求导在某个根上正确，就把返回的梯度视为完整数值映射的可靠导数。

原 lvl-l4 配对的四个终态 FD 检查则全部通过。其模型/数据对应、边界和数值停滞与地壳漏根问题应分别处理，
本轮未通过调阈值或更换模型掩盖失败。

## 两个可用例子的 NUTS 与后验预测

采用现有自有 slice NUTS；4 链，每链 1000 预热 + 2000 正式样本，稠密质量矩阵适应，
目标接受率 0.95、最大树深 10、种子 83261。先验在上述物理 Vs 边界内均匀；
使用前述独立初值，不以 LM 最优解复制四条链。地壳和原 lvl-l4 失败配对未用于后验结论。

| 例子 | 最大 R-hat | 最小 bulk ESS | 最小 tail ESS | 正式期发散/无效/触树深 | 墙钟 / 最大 RSS |
|---|---:|---:|---:|---|---|
| nearsurface | 1.00480 | 1254.6 | 1596.5 | 0 / 0 / 0 | 213.08 s / 7316 KiB |
| lvl-l4-m1 | 1.00271 | 2390.3 | 2568.9 | 0 / 0 / 0 | 63.96 s / 7188 KiB |

每个例子抽取 400 个后验样本做预测，全部有效。平方白化残差的后验预测 p 值均为 1；
观测差异均值分别约 7.34、5.10，复现数据约 79.75、50.05。
这表明当前合成数据的残差相对于人为假设的 σ 很小，不能据此声称噪声模型已经校准。
浅层第四层的 95% 区间约 `[0.5060,0.6690]`，半空间约 `[0.6086,0.6459] km/s`，
均未包含所提供模型对应值 0.5、0.65。这里只是一次给定数据/模型的比较，不是覆盖率实验。
后验条件于所选界面、Vp/密度、先验和误差，不含层数不确定性；良好多链诊断也不证明全局混合。

## 复现与本地结果

环境：Linux/WSL2、Intel Core Ultra 9 285H，GCC 13.3 Release，DispWeave 0.9.0，
物理代码对应提交 `52661e2`；Python 3.12.12、NumPy 2.3.5、Matplotlib 3.10.8。
无需新增核心依赖。脚本需要 Python >=3.11 + NumPy，绘图另需 Matplotlib。
外部仓库需自行取得且包含上述输入和可运行的 `bin/forward`。

```bash
mkdir -p results
python scripts/check_qedisp_examples.py --source ../QEDispInv --exe build-m8/dispweave \
  --output results/qedisp-new --stage forward --sigma .005 .02 .002
python scripts/check_qedisp_examples.py --source ../QEDispInv --exe build-m8/dispweave \
  --output results/qedisp-new --stage sensitivity
python scripts/check_qedisp_examples.py --source ../QEDispInv --exe build-m8/dispweave \
  --output results/qedisp-new --stage lm
python scripts/check_qedisp_examples.py --source ../QEDispInv --exe build-m8/dispweave \
  --output results/qedisp-new --stage stalled
```

当前正演与参考灵敏度阶段返回 0，LM 阶段返回 **4**，停滞点检查返回 **2**，属于如实保留的失败结果。
`stalled` 自动保存最差 FD 查询的 ±h/0 模型，并用两个正演程序比较根前缀。
脚本拒绝覆盖已有实验目录/报告。`--case` 仅选择首次准备的案例，后续阶段读取已保存的案例清单。

对 nearsurface 和 lvl-l4-m1 各运行一次以下命令（替换案例目录），随后绘图：

```bash
build-m8/dispweave nuts --config results/qedisp-new/nearsurface/inversion.toml \
  --data results/qedisp-new/nearsurface/observations.txt --output results/qedisp-new/nearsurface/nuts
python scripts/posterior_diagnostics.py results/qedisp-new/nearsurface/nuts \
  --output results/qedisp-new/nearsurface/posterior-diagnostics.json
build-m8/dispweave predict --run results/qedisp-new/nearsurface/nuts \
  --output results/qedisp-new/nearsurface/predictive --draws-per-chain 100
MPLCONFIGDIR=/tmp/dw-mpl python scripts/plot_predictive.py results/qedisp-new/nearsurface/predictive
MPLCONFIGDIR=/tmp/dw-mpl python scripts/plot_qedisp_examples.py results/qedisp-new --output results/qedisp-new-plots
```

当前本地保存位置：

- `results/qedisp-final/`：最终脚本重放的四组输入、来源哈希、参考正演、FD、16 次 LM 及停滞/分支检查。
- `results/qedisp-examples/nearsurface/`、`results/qedisp-matched/lvl-l4-m1/`：长链、诊断及后验预测。
- `results/qedisp-plots/`：四组曲线/全部终态模型/可用后验区间和 `summary.json`；未将停滞模型选为最佳解。
- `results/qedisp-examples/`、`results/qedisp-validation/`：保留首轮测试和追查记录；最终配对解释以本报告为准。

完整 CTest **62/62 通过，14.36 s**；新增契约测试覆盖列/单位映射、低速层与重复显示序号、
误差必须显式给出、非法模态、查询丢失/重排。常规回归通过不代表上述地壳应用案例通过。

下一项优先处理地壳案例的近邻根枚举和诊断：先独立确认根的物理身份，形成最小回归，
再修复 invLM 适配后端并重验 LM 路径与梯度连续性。此问题修复前不扩展该例子的 HMC/RJMCMC 结论。
