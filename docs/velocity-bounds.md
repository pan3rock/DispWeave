# 速度边界与半空间

2026-10-23。为固定层数频散 LM、HMC/NUTS 及后验预测统一生成物理 Vs 边界。
本次保留显式正则化；没有新增模型选择、自动权重或自适应分层。

## 配置与定义

```toml
[inversion]
vs_min = 0.6       # 全局最小 Vs，km/s
vs_max = 3.6       # 全局最大 Vs，km/s
vs_width = 1.2     # 完整窗口宽度，km/s；半宽为 0.6
vs2model = "brocher05"
starts_vs = [[1.2, 1.8, 2.8]]

[inversion.regularization]
type = "reference_damping"
lambda = 10.0
```

对每层（包括半空间），使用 `model.vs[i]` 为参考：

```text
lower[i] = max(vs_min, reference_vs[i] - vs_width/2)
upper[i] = min(vs_max, reference_vs[i] + vs_width/2)
```

这是交集。全局边界可能使窗口不再对称；取并集会放宽约束。
范围围绕参考模型生成，与 `starts_vs` 无关，多初值共享同一组边界。
在物理 Vs 空间定义范围，再转换为内部 log Vs 参数；不是在 log 空间作加减。
正则化 lambda 不控制窗口宽度；窗口也不代替正则化。

三项必须一起提供、有限且满足 `0 < vs_min < vs_max` 和 `vs_width > 0`。
空交集、零宽度或参考模型超出范围时，报告零基层号，不自动扩张窗口或移动参考模型。
初值不被截断：LM 保留 `invalid_initial_bounds` 失败记录，HMC/NUTS 在启动前拒绝；
采样器的 logistic 变换需要严格位于边界内部的初值。

旧的 `lower_vs`/`upper_vs` 逐层数组仍可使用，必须成对提供，不能与上述三项混用。
这两种方式是同一数学支持域的不同配置入口，旧结果可精确重放。
窗口边界当前用于 `synthetic/invert/hmc/nuts/predict`；联合反演 CLI 仍要求逐层数组，
遇到窗口字段显式报错。RJMCMC 的跨维先验和分辨率分析的尺度定义不作改变。

### 物理条件

固定 Vp 时，每层整个 Vs 区间必须满足 `Vs < sqrt(3/4)*Vp`，保持正体积模量；
与之冲突的配置明确拒绝并指出层号，不暗中裁剪用户给出的边界。
经验关系下逐次检查派生 Vp/ρ 的正值、有限值和弹性稳定性；不保证所有经验公式
对任意速度或岩性有效，公式无效的试探模型仍在支持域之外。

HMC/NUTS 的 `uniform_vs` 先验使用生成后的物理上下限与其归一化；
`uniform_log_vs` 使用对应 log 区间。两种先验的测度仍不同，窗口设置不替代先验声明。
LM 的有效正则项仍不能静默进入采样后验，采样需另行声明概率先验。

## 半空间约定与输出

最后一个 `model.depth` 是半空间顶深，以下无限延伸；固定层厚只固定层界面。
半空间 Vs 默认参与反演，使用同样窗口，没有强制单调递增约束。
LM 的参考阻尼包含半空间，差分平滑包含最深有限层与半空间的相邻差分。

`fixed_vs_layers` 中显式包含末层零基索引才固定其 Vs，固定值取 `model.vs`，
对应初值必须一致；该固定坐标接口目前仅用于 LM，HMC/NUTS 继续显式拒绝。
保留固定选项不意味着默认固定深部，也不意味着数据已经约束该固定值。

当前正演只接受 `c_pred < Vs_halfspace` 的束缚模态。这是预测的物理条件，
不是对每个含噪观测值的硬不等式。程序不采用 `max(c_obs)+delta` 的默认速度下限，
也没有在此次修改中增加该可选策略。缺失/未解析模态仍导致整次目标评估失败，不丢弃数据。

每次 LM、采样或预测输出新增 `parameter_bounds.txt`：
`layer top_km reference_vs lower_vs upper_vs fixed halfspace`。
`fixed=1` 时速度固定在 reference_vs，所列区间只是配置范围。
`metadata.txt` 保存边界来源、全局设置、半空间顶深、free/fixed 状态及 observation_floor=none。
原始配置照常保存，旧的模型/轨迹文件格式不变。

## 半空间顶深对照

脚本 `scripts/check_halfspace_depth.py` 在明确给定的几个顶深下重复反演：仅移动末层顶界，
因此最深有限层厚度随之改变；层数、参考 Vs、初值、边界、观测、正则化设置和预算保持相同。
若使用按深度归一的平滑项，几何变化也会改变差分权重，这是其定义带来的变化，
不能声称正则化矩阵完全相同。本次 demo 使用不依赖深度间距的参考阻尼。
脚本不自动选深度，不把半波长或某个拟合阈值声明为可靠探测深度。

```bash
MPLCONFIGDIR=/tmp/dw-mpl python scripts/check_halfspace_depth.py --demo \
  --exe build-properties/dispweave --output results/halfspace-window-new

# 自己的观测；三个深度只是调用形式示例，应按模型与研究目标选择
python scripts/check_halfspace_depth.py --exe build-properties/dispweave \
  --config reference.toml --data observations.txt --covariance covariance.txt \
  --depths 100 150 200 --output results/halfspace-custom
```

每个顶深保留全部初值、失败、完整模型及轨迹。图上优先显示最低代价的收敛运行；
没有收敛运行时显示有限代价的末态并标记，全部失败则无拟合曲线。
仅对两者都收敛的情况计算与第一个顶深方案的共同深度区间 Vs 差异。
某个顶深没有任何收敛运行，脚本仍保存结果并返回 4，不伪装成成功。

### 实际 demo

输出：`results/halfspace-window-control/`，含 PNG/PDF、输入、真值、观测、逐层边界和摘要。
真值 Vs 为 `[1,2,3]`，层顶 `[0,2,4]` km；参考 Vs 为 `[1.2,1.8,2.8]`。
二者都用 Brocher05 派生 Vp/ρ，真值只用于生成观测和画图；反演调用使用单独参考配置。
0.2–1 Hz、0–1 阶，共 14 点；独立高斯噪声 σ=0.005 km/s，NumPy 随机种子 20261023，
实际观测保存以便跨环境重放。参考阻尼 lambda=10，最多 150 次，投影梯度阈值 1e-5。
这些参数是演示设置，没有针对结果自动调优。

| 半空间顶深 km | 收敛初值 | 收敛解数据 cost | 正则化 cost | 相速度 RMS km/s |
|---|---:|---:|---:|---:|
| 4 | 2/3 | 4.596811 | 0.544418 | 0.004052 |
| 6 | 2/3 | 108.493753 | 2.784810 | 0.019684 |
| 8 | 1/3 | 186.908697 | 3.013533 | 0.025837 |

4 km 方案的自由半空间 Vs≈2.96222；6/8 km 方案达到其上界 3.4 km/s。
共同 0–4 km 区间相对首方案的 Vs RMS 差约 0.13250/0.17053 km/s。
这说明本例对底部几何敏感，不能把加深顶界后的结果直接视为更可靠；
触界也不能单独证明需要扩大速度范围。
四个未收敛运行仍记为小步停滞，即使代价值接近同组收敛解也未重新标记。
本次不修改 LM 收敛阈值或求根算法，既有地壳近邻根漏检仍未解决。

## 软件验证

针对性 Release 检查 4/4：手算交集、非法输入、固定 Vp 的物理冲突、
半空间自由/固定、观测相速度高于半空间 Vs 时仍可计算残差、
新旧配置 LM 精确重放、HMC/NUTS 两种先验测度的短链及预测精确重放。
短链用于配置与目标一致性验证，不作为后验混合验收。
完整 Release CTest **67/67**，30.99 s；上述针对性 UBSan **4/4**，7.43 s。
日志保存于 `results/halfspace-window-control/ctest.log`、`ubsan.log`。
