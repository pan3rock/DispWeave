# M7 后续修复：受俘获 Rayleigh 搜索域

状态：已完成本次可确认问题的修复和对照；未认证一般层状介质的无根性。
这是 M7 实验暴露问题的独立修复，不是 M8，也没有扩展采样器或自适应 LM。

## 确认的问题与修复

保存的失败模型之一：层顶 (0,3.5846558879346531) km，
Vs=(1.8,1.0219876911644241) km/s，Vp/Vs=1.8，rho=2.3。
旧版原生 invLM `search(f,1)` 在 .15/.2/.3/.4/.6/.9 Hz 分别给出约
1.46293/1.51287/1.58399/1.62359/1.65503/1.66232 km/s 的候选，
都超过半空间 Vs。原适配层将它们判为数值失败，没有把这些速度用于拟合。

原因是 `get_samples()` 无条件插入表层 Rayleigh 速度估计和最小 Vs 附近采样点。
当上层速度高于半空间时，这些点可能超出受俘获模态的定义域。
对于本项目弹性半空间的衰减边界条件，S 波垂向衰减要求 c<Vs_halfspace。
域外实数延拓的零点不能作为受俘获 Rayleigh 根；泄漏波不在当前支持范围内。

本次只在 Rayleigh 搜索采样完成后、频散函数评估前移除非有限、非正和
c>=Vs_halfspace 的点。保留现有下界、域内网格、求根器、灵敏度及似然。
Love 路径未改。没有根据残差或抽样效果增加备用相速度，也没有将失败行丢弃。

回归 `trapped_domain` 固定保存上述模型和一个接近截止的三层模型：
其 .15 Hz 基阶约为 1.14343358939869 km/s，半空间 Vs=1.1436030468024461。
独立 P/SV 行列式在 [1.14343,1.14344] 内给出同一根；请求四个模态仍保留该基阶，
其余模态为 missing_or_unresolved。Dunkin 和 fast-scalar 均运行此测试。
回归在旧实现上确实失败（Out-of-domain candidate contaminated search），修复后通过。

## 扩大的独立核查

从 M7 等权长链的三份失败模型文件中，每链取 100 个等间隔样本：
索引 floor(j*N/100)，j=0..99，包括预热失败；不根据核查结果筛选。
提交的 `benchmarks/data/rj_failed_models.txt` 保留全部 300 个模型及原链/评估索引。
对每个模型在六个观测频率用独立 P/SV 边界匹配方法扫描，分别使用 5000 和
10000 段，速度范围 [.5*min(Vs), Vs_halfspace*(1-1e-8)]。
这不是生产求根后端，不能处理所有病态或极厚模型，也不是模态计数证明。

每种扫描密度均得到下表（1800 个模型/频率组合）：

| 后端状态 | 修复前 | 修复后 |
|---|---:|---:|
| valid | 730 | 730 |
| numerical_failure | 70 | 0 |
| missing_or_unresolved | 1000 | 1070 |

两种网格均没有扫描异常；没有出现“后端未解析而独立扫描找到根”的组合。
730 个有效结果与独立扫描最低根的最大差为 5.227e-13 km/s。
因此本次确认的是搜索域/诊断污染，不是已经确认并修好的漏根。
有限扫描可能漏掉紧邻根、偶重根、极端病态根或极靠近半空间边界的根；
网格加密一致也不能将 missing_or_unresolved 升格为物理无根。

可独立复现某个频率：

```bash
cmake --build build -j2 --target rj_forward_audit
build/rj_forward_audit benchmarks/data/rj_failed_models.txt 1.8 2.3 0.15 300 5000
build/rj_forward_audit benchmarks/data/rj_failed_models.txt 1.8 2.3 0.15 300 10000
```

其余频率为 .2/.3/.4/.6/.9。工具新增的最后两个可选参数是模型数与扫描段数，
省略时仍为原来的 20、5000。核查文件位于 `results/forward-repair/baseline-*` 和
`fixed-*`；baseline 使用 HEAD=83f7c47 的原始 dispersion.cpp 单独编译并冻结执行文件。
初步 `before-*` 输出未纳入统计（其间工作区正在构建），避免混用版本。

## RJMCMC 重跑：没有改变这两组后验结果

重跑原等权先验和 [1,1,.5,.25] 层数权重对照。每组三链，每链预热 20000、
保留 200000；起点、种子、步长、数据和协方差与 M7 原实验一致。
新结果在 `results/forward-repair/long-rj`、`prior-contrast-rj`。
六份 `chain_*_models.txt` 与各自原文件逐字节一致，共 120 万个保留模型。
因此两组原有层数概率、界面分布、Vs 后验及其统计诊断保持不变，无需重估。
这也符合该修复只影响此前已被拒绝候选的事实，不能外推到所有模型和观测组合。
两组采样仍返回 5：未解析模型仍显式拒绝，失败数没有被抹去。

```bash
build/dispweave rjmcmc --config results/m7/long.toml \
  --data results/m7/example-data/observations.txt \
  --covariance results/m7/example-data/covariance.txt \
  --output results/forward-repair/long-rj
```

对照使用 `results/m7/prior-contrast.toml` 和不同输出目录。
这些配置均由 `examples/rjmcmc/synthetic.toml` 修改 warmup=20000、draws=200000 得到；
先验对照再改 layer_weights=[1,1,.5,.25]。输出目录需新建。

## 验收与限制

- Release CTest 47/47，通过新增回归及所有原有正演、灵敏度、反演和采样检查。
- UBSan 定向 9/9，覆盖新回归、半空间、golden、独立参考、截止、数值异常、
  物理适配及 RJ CLI；其构建方式与 M7c 验收报告一致。
- invLM 修改说明及本地 dispersion.cpp SHA256 已更新，未改原外部仓库。

当前可可靠识别该类域外候选，但仍没有一般 Rayleigh 模态计数/无根证书。
继续使用 missing_or_unresolved，保留失败模型和根诊断；不得把模型拒绝全部解释为物理 cutoff。
后续若要解除这一限制，需要单独验证具有根计数或稳定谱信息的独立方法，
而非继续增加有限扫描密度或将当前警告改为成功。
