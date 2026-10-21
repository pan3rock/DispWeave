# M3：固定层数 LM 与端到端工作流

M3 完成 Phase 1 的最小可运行版本。主体仍是 C++17；新增 LM 为项目自有实现，
正演和物性核继续复用已验证的 invLM。没有引入额外 C++ 依赖或外部优化框架。
固定层顶、密度，反演各层及半空间 Vs；Vp 固定或按指定 Vp/Vs 随动。
本项不优化界面，不引入自适应分层、HMC 或 RJMCMC。

## 数学与接口

`include/dispweave/lm.hpp` 的 `bounded_lm` 接收残差/Jacobian 回调、初值、有限边界及选项，
`invert_lm` 将已有 `DispersionProblem` 接到同一求解器。测试可以脱离频散验证数学行为。
回调提供 `DataEvaluation`，成功时残差与 Jacobian 必须同样白化、行数不变。
目标为 `F = ||L⁻¹(c(q)−d)||²/2`，`LLᵀ=C`；不添加先验、变换 Jacobian 或平滑项。
完整归一化似然仍由 M2 的目标函数提供。

参数 `q=log(Vs)`；令 `w=upper−lower`，在 `x=(q−lower)/w` 中计算：

- 缩放 Jacobian `Js=J diag(w)`；已在边界且梯度指向边界外的分量冻结。
- 使用列主元 QR 解增广系统 `[Js; sqrt(lambda D)] delta = [-r; 0]`，
  `Djj=max(||Js[:,j]||², 1e-12)`，不形成正规方程的逆。
- 将试算投影回边界，用实际可行步计算线性模型预测下降量。
  若截断耦合步后不下降，使用投影 Cauchy 步，步长为 `1/||Js_full||²_F`。
  这是线性化目标梯度 Lipschitz 常数的上界所定义的下降步，不是数据相关的特例。
- 实际/预测下降比大于 `1e-4` 接受。接受后按
  `lambda *= max(1/3, 1−(2*rho−1)^3)` 更新，阻尼下限 `1e-15`；
  拒绝后依次乘 2、4、8……，下一次接受重置为 2。阻尼超过 `1e30` 返回失败。
- 收敛标准仅为 `||x−project_[0,1](x−Js_fullᵀr)||∞ <= gradient_tolerance`。
  它是所声明参数尺度下的盒约束 KKT 指标，不是模型误差或分辨率。
  微小步、残差降低停滞、根失败、梯度失败不单独作为成功条件。

非下降/不可表示的步、阻尼耗尽、迭代上限返回明确失败。缺模态试算整体拒绝并增加阻尼；
初值缺根直接结束该初值。所有初值独立运行，只在**已收敛**结果中按代价选最佳；
若一个都未收敛，`best_run=-1`、进程返回 4，仍保存诊断。
`LMResult` 保存最终投影梯度、接受/拒绝次数、Cauchy 次数、根/核失败次数与逐步记录。

## CLI 与文件约定

```bash
build/dispweave synthetic --config examples/inversion/synthetic.toml --output results/synthetic
build/dispweave invert --config examples/inversion/synthetic.toml \
  --data results/synthetic/observations.txt --covariance results/synthetic/covariance.txt \
  --output results/replay
python scripts/plot_inversion.py results/synthetic
```

输出目录必须新建或为空，防止旧结果和新结果混在一起。参数解析、数据、I/O 错误非零退出。
`invert` 直接采用观测文件的查询行，不要求配置含 `[queries]` 或 `[synthetic]`。
观测文件每行四列：`frequency_hz phase_km_s mode sigma_km_s`，支持 `#` 注释；
模态是从 0 开始的整数。每行 sigma 必须正值；完整协方差文件若提供则覆盖 sigma² 对角阵，
单位 `(km/s)^2`，行列顺序与观测完全相同，要求对称正定。没有静默跳过坏行。

TOML 的 `[model]` 在 `synthetic` 中是真值，在 `invert` 中只是固定物性的参考模型；
不是隐含先验。`[inversion]` 指定 `lower_vs`、`upper_vs`、二维数组 `starts_vs`，
可选 `sensitivity`（energy/implicit）、`max_iterations`、`gradient_tolerance`、`initial_damping`、
`vp_vs_ratio`。不设置比值时固定 Vp；设置时参考模型的 Vp 必须一致。
参考 Vs 必须在边界内；单个初值越界保存为该初值失败，长度错误是配置错误。

合成噪声为 `L z`，`Cij=sigma² correlation^|i−j|`，相关性按查询的**行序**定义，
不是通用地震误差模型。`mt19937 + normal_distribution`、种子及配置被记录，
但标准库跨平台不保证逐位相同正态序列；保存的观测和协方差是跨平台重放依据。
`add_noise=false` 仍按配置的非零 sigma 加权目标。

结果包含：

- `config.toml`、`observations.txt`、`covariance.txt`：输入快照和最终使用的数据。
- `metadata.txt`：软件/编译器/上游/依赖版本、后端、灵敏度方法、种子、最佳初值与总耗时。
- `runs.txt`：所有初值是否收敛、代价、评估计数、根/核失败、Cauchy 次数、投影梯度、
  最终触界参数数和终止原因。代价是半平方白化残差，不是 reduced chi-square。
- `run_N_model.txt`、`run_N_prediction.txt`、`run_N_trace.txt`、`run_N_diagnostic.txt`：
  初值/最终模型、逐行正演及核诊断、试算接受记录和终态诊断。越界初值没有正演文件。
- Python 另生成 `inversion.png`：速度剖面、各阶数据与预测、按边际 sigma 缩放的残差。
  对相关误差，该绘图残差不等于优化使用的白化残差；图中明确标注，无后验区间。

## 可重复验收

从项目根目录运行一条命令：

```bash
python scripts/run_inversion_example.py --output results/m3
```

需要 Python、NumPy、Matplotlib；本次使用 Python 3.12、NumPy 2.3.5。
脚本依次构建 Release、CTest、生成六组配对实验、绘图、重放相关误差观测。
核心 C++ 构建/运行不需要 Python；CMake 检测到 Python 时自动加入 stdlib-only 工作流测试。
重复运行使用另一个空输出目录，不覆盖历史结果。

Linux/WSL、GCC 13.3、CMake 3.28.3 下，Release **22/22** 测试通过。
Debug + UBSan（`-fsanitize=undefined -fno-sanitize-recover=all`）也为 **22/22**，
总计 3.88 s，未报告未定义行为；Release 测试总计 0.16 s。
第三方源码哈希校验 449 个文件全部通过。
包括原有 18 项、3 项 LM 测试、1 项端到端工作流测试：

- 解析线性最小二乘、盒约束解析最优点、耦合步裁剪后不下降的回归测试；
- Rosenbrock 非线性目标及拒绝路径；
- 越界初值、不可用初值、缺模态试算回调、梯度失败、迭代上限；
- 无噪声/独立噪声/相关噪声、多初值、真实后端缺模态初值、坏数据/协方差、
  目录覆盖保护、保存数据重放后逐位相同的代价/轨迹统计。

缺模态**试算**路径用受控回调测试，真实物理后端另测试缺模态初值；
它们不证明跨截止的全局分支连续性。原有根/导数/截止测试继续独立保留。

## 配对实验结果与解释

真值层顶 `[0,2,4]` km、Vs `[1,2,3]` km/s，Vp 和密度见示例。
基阶为 7 个频点 0.2–1 Hz；多模态追加相同频点的第一高阶，共 14 点。
每组 3 个初值，噪声 sigma=0.005 km/s，相关案例相关系数 0.5，种子 2026。
以下为已收敛的最低代价结果；最大相对误差对三层 Vs 取最大值。

| 数据 | 噪声 | 代价 | 最大 Vs 相对误差 | 收敛初值数 |
|---|---|---:|---:|---:|
| 基阶 | 无 | 4.67e-20 | 8.15e-11 | 3/3 |
| 基阶 | 独立 | 1.43339 | 16.67% | 1/3 |
| 基阶 | 相关 | 1.98217 | 16.67% | 1/3 |
| 基阶+第一高阶 | 无 | 8.41e-19 | 7.00e-11 | 3/3 |
| 基阶+第一高阶 | 独立 | 3.16467 | 5.91% | 2/3 |
| 基阶+第一高阶 | 相关 | 3.39177 | 6.29% | 3/3 |

两组带噪声基阶最优解的半空间 Vs 都触及 3.5 km/s 上界。追加第一高阶后对应值为
3.17716、3.18867 km/s，但这一次噪声实现不能证明普遍的分辨提升。
各组数据量/协方差不同，原始代价不能直接用于跨组模型优劣排名。
无噪声自洽恢复不是独立的物理正确性验证；独立验证由 M1/M2 承担。
六组普通案例的所有初值合计根失败和核失败均为 0；故障注入测试单独验证拒绝行为。

本次六组命令的三初值总耗时（含配置、合成和结果写出，不含绘图）依次为
3.78、39.29、20.13、5.20、8.19、5.73 ms。单独多模态无噪声复测为 5.08 ms，
`/usr/bin/time -v` 测得最大 RSS 5920 KiB；外部计时的百分之一秒分辨率不足以精确计量本例。
以上是本机一次运行记录，不是跨机器性能保证。首次/重复正演及困难模型 baseline 见 M1/M2。
结果保存在 `results/m3/`（已忽略，不提交生成文件），完整参数和每次失败可在各子目录审阅。

## 限制与下一项

- 基阶弱约束深部参数可能触界；解仍条件于固定层界面、固定 Vp/密度策略及边界。
  本项没有参数协方差、后验采样、维度不确定性或深度分辨率输出。
- 有些初值达到几乎相同的代价，却因根求解/浮点精度下的下降量不可靠而耗尽阻尼；
  也有投影 Cauchy 步进展缓慢而达到 150 次上限。本项如实报告失败，未放宽标准伪造成功。
  它是局部优化器，没有全局最优或所有初值收敛的保证。
- 缺根不等于物理模态不存在；未认证的复杂高频/强对比场景仍受 M1/M2 限制。
- 仍缺厚度核与界面移动验收，不能直接进入自适应增层。

下一项 M4 宜先独立验收固定参数化的白化灵敏度谱、参数尺度和深度约束比较，
同时补充弱约束/近截止案例。之后再实现并比较增加/合并参数的准则。
