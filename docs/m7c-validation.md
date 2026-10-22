# M7c：频散集成与跨维结果

M7c 实现与下列验收已完成。数值限制仍须随结果报告；完成里程碑不意味着认证所有先验模型的物理正演。

已接入 `RJDispersionProblem`：每层 Vp=固定比例*Vs，密度固定，末层无限半空间。
复用 invLM 无梯度正演和完整协方差 GaussianLikelihood，显式返回支持外、正演失败或数值失败。
物理映射、固定维度似然一致性、独立均匀半空间 Rayleigh 三次方程、协方差归一化及缺模态测试通过。
Release 与 UBSan 的 `rj_dispersion` 均通过；`rj_workflow` 验证 CLI 重放、可变长度状态、
逐次状态与层数一致性、输入协方差保留、输出保护和无效模型/缺模态输入。

CLI `rjmcmc` 读取 `[rj_prior]`、`[rj_physics]`、`[rjmcmc]` 与 `[[rj_starts]]`。
`chain_*_models.txt` 每行是 `draw K log_likelihood log_prior interfaces[K-1] vs[K]`，
单位 km 和 km/s；保存全部正式样本，包括拒绝后的重复值。
trace 另含预热、动作、提议状态、目标是否实际评估、目标状态、接受与前后层数。
正演/数值失败或整链没有接受时返回 5，全部样本与诊断仍保存。
没有把所有正演失败强行解释为物理模态不存在。

## 首轮频散运行：尚不能据此认证后验

`examples/rjmcmc/synthetic.toml`：真值层顶 (0,1.5)、Vs=(1,2.3)、Vp/Vs=1.8、rho=2.3；
六个基阶观测、sigma=.03、相关系数 .2。K∈[1,4]、等层数权重、界面支持 Z=5、h=.3、
Vs∈[.8,2.7]。三条链从 K=1、2、3 开始，每链预热 10000、正式 50000。

```bash
build/dispweave synthetic --config examples/rjmcmc/synthetic.toml --output results/m7/example-data
build/dispweave rjmcmc --config examples/rjmcmc/synthetic.toml \
  --data results/m7/example-data/observations.txt --covariance results/m7/example-data/covariance.txt \
  --output results/m7/rj
```

已运行版本保存在 `results/m7/first-rj`（输入 `results/m7/data`）。三链分别：

| 链 | 接受数 | 层数变化数 | 正演失败 | 数值失败 |
|---|---:|---:|---:|---:|
| 0 | 12437 | 1966 | 2102 | 0 |
| 1 | 12883 | 1662 | 1637 | 0 |
| 2 | 12905 | 1953 | 1715 | 0 |

总耗时约 5.75 s，返回 5。层数变化不等于跨维充分混合；
这些正演失败尚需进一步调查，不能把当前采样直接宣称为已认证物理后验。
首轮数据生成附带 LM 因严格默认梯度容差停滞；示例已使用已有 HMC 示例相同的 1e-5 容差，
LM 正常收敛，观测文件经逐字节比较完全一致。RJMCMC 不依赖 LM 最优模型初始化。

以下记录首轮发现的问题及随后完成的审查、扩展实验与回归。

## 失败模型审查与首轮跨维诊断

已增加 `chain_*_failed_models.txt` 和 `chain_*_failed_rows.txt`，保存每个失败目标评估的
完整界面/Vs、每个观测行的根状态、找到的根数及相速度。evaluation_index 与 trace 对应，
包括预热；不只保存少量例子。`results/m7/audited-rj` 与原首轮的全部三链模型文件逐字节一致，
记录诊断没有改变 RNG 或目标。CLI 回归已核对全部三链的失败记录与 trace evaluation ID 一致性。

对链 0 前 20 个失败模型，在全部 6 个频率执行独立 P/SV 边界匹配的 5000 段扫描。
模型/工具见 `benchmarks/rj_forward_audit.cpp`；结果保存在 `results/m7/forward-audit-*.txt`。
6 个频率后端无效的数量依次是 5、8、19、20、20、20，独立扫描均未找到有效根，
没有扫描异常；其余有效结果与独立根一致。多个失败模型是高速上层覆盖较低速半空间。
这与受俘获 Rayleigh 模态可能缺失相容，但有限扫描不是无根证明，
也不能外推为所有正演失败都已被认证。本项未修改 invLM 搜根规则或放宽有效根检查。

`scripts/rj_diagnostics.py` 以每个 K 的指示变量计算多链诊断，不把不同层数的层编号
直接拼成参数向量。输出各链层数概率、层数转移矩阵、驻留长度、支持端点往返次数，
及公共深度网格的 Vs 后验/诊断。界面结果分别给出每个深度 bin 内“至少一个界面”的概率
与期望界面数，二者不能混为概率密度。未访问的 K 标记常数链，不能声称其概率严格为零。
读取/分析工具使用 Python >=3.11 的标准库 tomllib 与 NumPy，无新第三方依赖。

```bash
python scripts/rj_diagnostics.py results/m7/first-rj \
  --output results/m7/first-diagnostics.json --grid-points 31
```

首轮 K=1/2/3/4 的合并概率约为 0/.3972/.3705/.2323。
K=2 的逐链概率 .4377/.3269/.4270，指示变量诊断警告；不能只凭维度变化次数宣布混合充分。
K=1 未被正式样本访问，同样明确报告，不将其常数指示变量强行记为有效 ESS。
诊断脚本已通过构造的互不混合链、界面 bin 多界面、精确界面处 Vs 映射、驻留/往返
及 malformed row 的检查。后续结果见下节。

## 扩展采样、独立保留模型检查与后验预测

将同一配置的预热改为 20000、正式样本改为 200000，三个起点和随机种子不变。
配置与结果分别保存在 `results/m7/long.toml`、`results/m7/long-rj`；采样耗时 20.27 s。
三条链正式阶段的正演失败分别为 6949/6904/7660，无数值失败。
K=2/3/4 合并概率为 .41403/.35179/.23418，指示变量 R-hat 为
1.00166/1.00101/1.00057，bulk ESS 为 1341/4182/2049；K=2 的概率 MCSE 为 .01345。
K=1 未访问，不能根据零次出现证明其后验质量为零。整体诊断仍为 warning。

跨维后验预测每链等间隔取 300 个模型，总计 900 个，不按预测成功与否筛选。
全部预测有效；生成重复观测时保留完整协方差。
观测白化残差平方均值 3.4445，重复数据对应均值 6.1196，后验预测尾部比例 .7356。
这是使用同一数据拟合后的模型检查，不是均匀校准 p 值，也不是覆盖率验证。

```bash
python scripts/rj_diagnostics.py results/m7/long-rj --output results/m7/long-diagnostics.json --grid-points 31
build/dispweave rj-predict --run results/m7/long-rj --output results/m7/long-predictive --draws-per-chain 300 --seed 2026
python scripts/plot_rjmcmc.py results/m7/long-diagnostics.json --output results/m7/long-summary.png --truth-config examples/rjmcmc/synthetic.toml
python scripts/plot_predictive.py results/m7/long-predictive --output results/m7/long-predictive.png
```

从这 900 个保存模型中按索引 floor(j*900/20)、j=0..19 选取 20 个模型，
在六个观测频率分别运行独立 P/SV 边界匹配扫描。
120 项均找到独立根，无扫描异常；后端相速度与独立扫描最低根的最大差
5.014e-13 km/s。选取未基于数值结果筛选，输入保存在 `retained-audit-models.txt`，
输出为 `retained-audit-<frequency>.txt`。这支持抽查模型的根正确性，
不证明整个连续先验支持域的根完备性，尤其不能解除先前失败提议的数值限制。

已实际渲染并检查层数/深度 Vs/界面占据概率三联图和频散预测图；图中保留警告，
Vs 区间是逐深度可信区间，不能解释为深度分辨率。已修复单模态预测图标题裁切。
本轮 Release 全套 CTest 46/46 通过（6.70 s）；UBSan 定向 7/7 通过（23.34 s），涵盖 RJ 契约、解析分布、失败路径、物理适配、CLI、预测与分析。

## 层数先验敏感性对照

保持观测、协方差、起点、种子、提议尺度和采样预算不变，仅将
`layer_weights` 改为 `[1,1,.5,.25]`。配置 `results/m7/prior-contrast.toml`，
结果 `results/m7/prior-contrast-rj`，诊断 `prior-contrast-diagnostics.json`。

| 层数 K | 等权先验的后验概率 | 对照先验的后验概率 | 对照 R-hat | 对照 bulk ESS |
|---|---:|---:|---:|---:|
| 1 | 未访问 | 未访问 | 不可估计 | 不可估计 |
| 2 | .41403 | .61499 | 1.00597 | 789 |
| 3 | .35179 | .28081 | 1.00177 | 1883 |
| 4 | .23418 | .10421 | 1.00504 | 1223 |

对照 K=2 概率 MCSE=.01732，逐链概率 .65866/.63688/.54943，
仍有可见的 Monte Carlo 波动，不能把 pooled estimate 当成精确值。
两组 31 个深度网格 Vs 的最大 R-hat 分别为 1.00612/1.00407，均未触发阈值；
这不证明全局混合或没有未发现的后验模态。
对照另外取 900 个模型完成预测，零失败，观测白化残差平方均值 3.19174，
重复数据均值 6.11964，后验预测尾部比例 .76222。
两个实验使用相同预测种子和协方差，重复数据残差平方均值相同是预期行为。

这一对照说明六个基阶观测无法消除层数先验影响；拟合相近不等于结构复杂度相同。
首轮仅认证工具链与所列数值检查，不据此宣称完成多阶信息增益或重复覆盖率研究。

## 最终验收对应关系

| 要求 | 当前证据 |
|---|---|
| 归一化层数/界面/Vs 先验、可逆增删层、Jacobian | M7a 验收；`rj_contracts` |
| 连续更新、边界自转移、解析层数概率、跨维往返 | M7b 验收；`rjmcmc_distribution`、`rjmcmc_failures` |
| 物理适配、完整协方差、不丢缺失模态 | `rj_dispersion`；独立半空间及分层根检查 |
| CLI 可复现状态/失败记录与预测行 | `rj_workflow`；全部三链重放与失败 evaluation ID 对应 |
| 多链层数、驻留、往返、深度 Vs、界面概率 | `rj_diagnostics` 的非混合构造与上述真实长链结果 |
| 后验预测及先验敏感性 | 两组各 900 次有效预测；上表层数先验对照 |
| 可重复示例、可视化与使用说明 | `examples/rjmcmc/synthetic.toml`、README、两种绘图脚本 |
| 本地结果展示 | `scripts/build_rj_demo_page.py`；`web/m7-demo/dist/index.html` |

本地页内嵌四张已渲染检查的 PNG，HTML 结构检查确认页内导航目标和全部 PNG 数据有效，
不依赖网络资源。当前环境无浏览器自动化，因此未声称通过真实浏览器排版验证。
页面生成器随代码提交，生成的 HTML 和实验大文件留在本地。
构建版本更新为 0.9.0；上述长链原始元数据保留实际运行时的 0.8.0，不修改历史结果。

仍未解决：一般层状介质的根完备性、失败提议的全域物理分类、弱支持维度的概率精度，
以及更复杂后验下的效率。当前采样结果条件于后端可解析的模型域；未解析失败若实际有有效根，
拒绝这些状态会改变目标。不得将本示例当作已认证的全域物理后验或实测数据结论。
下一项最有价值的数值工作是对失败模型扩大独立核查和区分物理缺模态与求根失败，
随后再开展多阶/多噪声及重复覆盖率研究；不在 M7 中耦合 HMC 或恢复自适应 LM。

最终 0.9.0 构建后再次运行完整 Release CTest：46/46 通过（7.56 s）。
独立审查工具新增 malformed header 拒绝检查，正常输入输出与此前审查逐字节一致。
UBSan 的可重复配置与运行命令：

```bash
cmake -S . -B build-check -DCMAKE_BUILD_TYPE=Debug \
  '-DCMAKE_CXX_FLAGS=-fsanitize=undefined -fno-sanitize-recover=all' \
  -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=undefined
cmake --build build-check -j2
ctest --test-dir build-check -R '^(rj_|rjmcmc_|predictive)' --output-on-failure
```

实际本轮使用 `/tmp/dispweave-m7-ubsan` 作为等价构建目录，避免影响现有构建。
