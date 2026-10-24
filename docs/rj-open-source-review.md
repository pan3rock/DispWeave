# RJMCMC 开源实现对照与改进优先级

核查日期：2026-10-24。结论是：**当前实现适合作为正确性基线，但高层数的探索机制过于简单，
不足以称为稳健高效的通用实现。** 本次核查没有发现需要推翻先验或 birth/death 接受率的证据。
结合 K=3/4 的失败结果，下一步优先做一次保持原目标不变的 parallel tempering（PT，温度交换）
对照，暂不自行增加多种联合/重排启发式，也不把近期机器学习提议直接作为生产替代。

此次完成的是源码和文献审查，没有安装、执行外部项目，没有改生产内核或启动新采样。
以下“当前版本”指本次能核实的公开版本，不宣称覆盖了所有最新研究。

## 核查到的版本和具体机制

| 项目 | 已核实版本或提交 | 对当前问题最有用的机制 |
|---|---|---|
| BayesBay | changelog v0.4.0，2026-07-13；`fd8af14706c8` | 邻域引导的新生参数、Voronoi 节点移动与关联参数重排、PT；2026 年修复边界重抽偏差 |
| BayHunter | README v2.1；master `af06830c266f`，2020-10-06 | 与频散最接近的应用参考：Voronoi 节点、邻域 Gaussian birth、单参数更新和尺度调整 |
| Eryn | GitHub release v1.2.7，2026-02-04；main `c317bf417059`，2026-10-12 | ensemble moves、PT、可配置的 RJ 生成分布及 multiple-try 机制 |
| openMCMC | GitHub release v1.0.9，2026-08-19；main `8bec80136c87`，2026-10-20 | 显式正反提议密度、可选 prediction-matching birth/death |

BayesBay/BayHunter 的 releases/latest API 未返回正式 release，故使用 changelog/README 标识，
不把网页爬取时间当作发布日。BayHunter 是有应用经验的旧实现，不把它称为近期更新的软件。
四个仓库共检查 19 个固定提交文件，URL、SHA256、版本日期与核查范围保存在
[源码清单](rj-source-audit.json)；本地快照位于 `results/rj-source-review/`。

### BayesBay：最直接的地球物理参考

`Voronoi1D.perturb_value` 对移动后的节点排序，并同步排列关联参数；新生参数可以从旧模型附近
生成，并保留先验/提议修正。其 sampler 层另有 PT，而不要求只靠局部步长解决跨区域探索。
[v0.4.0 Voronoi 源码](https://github.com/fmagrini/bayes-bay/blob/fd8af14706c834b083e4f3ba1db8a4d175de0b69/src/bayesbay/discretization/_voronoi.py)
· [邻域参数初始化](https://github.com/fmagrini/bayes-bay/blob/fd8af14706c834b083e4f3ba1db8a4d175de0b69/src/bayesbay/discretization/_discretization.py)
· [PT 源码](https://github.com/fmagrini/bayes-bay/blob/fd8af14706c834b083e4f3ba1db8a4d175de0b69/src/bayesbay/samplers/_samplers.py)

2026-07-10 的 v0.3.12 修复了位置提议越界后反复重抽的问题；v0.4.0 进一步明确无效提议计为
一次拒绝，并支持正演错误拒绝或抛出。这与我们“越界拒绝、保留自转移”的原则一致，
不是应当为了提高接受率而删掉的开销。
[官方变更记录](https://github.com/fmagrini/bayes-bay/blob/fd8af14706c834b083e4f3ba1db8a4d175de0b69/CHANGELOG.md)

### BayHunter：局部更新本身并不反常

它同样使用单速度和单节点位置的 Gaussian 更新；birth 速度围绕邻近旧节点生成，
移动节点后带着对应速度一起排序。由此不能把“我们用了随机游走”本身当作算法错误，
但两者参数化和结构移动能力不相同。
[实际提议与接受率代码](https://github.com/jenndrei/BayHunter/blob/af06830c266fb9790e4c4c59503c08aed2cf8023/src/SingleChain.py)

### Eryn / openMCMC：组合机制，而非唯一万能提议

Eryn 的 `DistributionGenerateRJ` 允许使用用户给定分布（包括先验）生成新参数；
因此从先验 birth 是合理基线，并非过时错误。它还提供集合和温度机制，值得参考整体组织方式。
[生成式 RJ](https://github.com/lisa-analysis-tools/Eryn/blob/c317bf41705927a2de36dec4123f1a6924b64c90/src/eryn/moves/distgenrj.py)
· [温度交换](https://github.com/lisa-analysis-tools/Eryn/blob/c317bf41705927a2de36dec4123f1a6924b64c90/src/eryn/moves/tempering.py)

openMCMC 可通过匹配预测来生成关联系数，并计算相应正反提议；这是“尽量减少提议造成的数据拟合
突变”的可借鉴思想。但其基函数系数匹配不能直接套进非线性 Rayleigh 频散，且其排列因子的抵消
依赖自身先验。不能把简化接受率直接复制到我们有最小层厚约束的有序界面先验。
[源码](https://github.com/sede-open/openMCMC/blob/8bec80136c874b192d95910db2681d38bdcd17ca/src/openmcmc/sampler/reversible_jump.py)

## 近期研究也查了，但区分论文与可核实实现

- 2026 年的 **t-roo** 建于 Eryn，使用 PT、stretch 和问题相关的跨模型映射；它也先进行模型内探索。
  这支持把模型内混合和跨模型提议分开验证。其引力波专用参数映射不能直接迁移到分层速度模型。
  本次只做论文层审查，代码网页缓存未获取成功，没有声称核查其源码。
  [作者论文](https://arxiv.org/abs/2607.21484)
- **Transport RJ + VI/normalizing flows** 的论文 v2 更新于 2026-02-02，学习模型内/模型间映射。
  初版给出的 `YinPingping111/TRJ_VINFs` 链接本次通过 GitHub API 和网页均返回 404，故不列为已核实
  可运行的成熟候选。较早的 2023 transport RJ 参考代码仓库仍可查看。
  [2026 修订论文](https://arxiv.org/abs/2512.12742) ·
  [初版代码声明](https://arxiv.org/html/2512.12742v1) ·
  [2023 作者仓库](https://github.com/daviesl/trjp)

这里不以论文较新或报告加速为理由引入训练流程。对当前小型模型，训练和额外验证是否值得，
需要独立的总成本证据；本次没有做这种性能比较。

## 对本项目的判断

| 当前做法 | 判断与依据 |
|---|---|
| 按 K 归一化的层数/有序界面/Vs 先验 | 应保留，不能省略随 K 变化的体积与阶乘项 |
| birth/death 的插入/删除映射、Jacobian=1、完整正反密度 | 与本项目状态定义一致；不是 Voronoi 公式的直接拷贝 |
| 固定动作概率、不可用动作保留自转移 | 正确；固定 K 试验约一半步骤为空动作是控制设计，不能当成半数正演成本 |
| 全先验范围生成 birth Vs | 合理但可能低效；可后续研究邻域 birth，不过它不能解决固定 K=3/4 的失败 |
| 单参数、小范围界面与速度随机游走 | 有成熟先例，但现有证据显示对我们的高层数问题不够有效 |
| 只靠增加样本或扫描步长 | 不作为当前优先方向；已经定位到结构探索不足 |
| 正演失败显式拒绝并记录 | 正确保留诊断；与其他软件类似处理并不能认证整个物理支持域 |

以上前三项还由本项目已有解析/可逆性测试支持。本次重新运行 RJ 定向 CTest，**6/6 通过**。
这不是新证明的全域正确性：未解析正演造成的支持域限制仍在。

特别需要避免混淆：BayHunter/BayesBay 移动的是 **Voronoi 节点**，物理界面由相邻节点决定；
我们直接采样 **K−1 个物理界面** 并要求最小间距。照搬“位置加扰动后排序且交换 Vs”会改变
映射及隐含先验，必须重新推导，不能当作无害的排序修复。

## 调整后的下一步：先做一个有预算上限的 PT 对照

这是结合源码机制与本项目证据作出的工程判断，不是开源项目替我们证明了 PT 必然有效。
K=3/4 的失败说明需要更强探索能力，但**尚未证明某个自定义联合提议一定是最佳修复**。
PT 有成熟实现可对照，而且可保留当前直接界面参数化、先验及局部转移核。
[BayesBay 官方 PT 文档](https://bayes-bay.readthedocs.io/en/latest/api/generated/bayesbay.samplers.ParallelTempering.html)

建议下一项仅在固定 K=4 做如下有限试验：

1. 目标为 `p_T(x) ∝ p(x) L(x)^(1/T)`，只加热似然，保留先验和硬支持。冷链 T=1 才用于原后验。
2. 使用 4 个独立温度梯组，各有 T=[1,2,4,8] 的 4 个副本。每副本 2500 预热＋25000 正式，
   共 440000 次局部提议，与原 4×(10000+100000) 的总局部提议数相同。
   温度梯度只是预先声明的初试设计，不称为最优；不扫温度、不自动延长。
3. 每 10 个局部步骤交替尝试相邻温度对交换，使用缓存 log likelihood：
   `log alpha = (1/T_i − 1/T_j) * (log L(x_j) − log L(x_i))`。
   保存温度身份与副本往返记录；不能把同一梯组的热链当独立冷链算 R-hat。
4. 用 4 条独立冷链检查原有界面、Vs 和共同深度统计，并记录交换接受率与副本温度往返。
   成本统计包含所有温度、预热、起点和任何附加正演，既看通过标准，也看 ESS/总目标评估。
   固定总提议数不保证实际正演调用数相同。
5. 接入前验证交换详细平衡、T=1 极限和已知分布；复用缓存时检查状态与似然一致。
   正式阶段不适应温度。热链遇到未解析正演仍拒绝并记录，不借加热掩盖数值失败。

若该有限对照没有改善冷链混合与总成本效率，就记录失败，再考虑针对冗余界面的非局部结构提议。
若有效，再单独验证接回跨维采样；不同时改 birth/death、先验、步长和参数化。
**上述 PT 为下一项设计，尚未实现或运行，不计入当前已完成成果。**

后续实施记录：上述限定预算的 PT 已实现并完成一次实验，见 [PT 验证](rj-tempering-validation.md)。改善明显但尚未通过，不追加调参。
