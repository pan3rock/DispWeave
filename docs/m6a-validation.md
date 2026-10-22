# M6a：多链诊断与后验预测

M6a 是 M6 的第一个独立交付。尚不包含质量矩阵适应或 NUTS，M6 整体仍在进行。
数值正演与后验预测使用 C++；样本处理、诊断与绘图使用 Python/NumPy。

## 诊断定义与依赖

`scripts/posterior_diagnostics.py` 接受已有 `chain_*_vs.txt`，要求至少两条等长链、
每链至少 8 个有限正式样本。保留重复状态；不读取预热样本，不把多链串接后计算 ESS。
奇数长度在 split 时舍弃正中间一个样本，各半链保持相同长度。

- 池化 average ranks，按 `(rank-3/8)/(S+1/4)` 转为标准正态分位。
- rank-normalized split R-hat 与折叠偏差 `|x-median(x)|` 的 R-hat 取最大值。
- bulk ESS 使用 rank-normalized 半链；tail ESS 取 5%/95% 分位指示变量的较小 ESS。
- FFT 自协方差、链内/链间方差结合、Geyer initial positive/monotone sequence。
  terminal even lag 的修正按 ArviZ 0.21.0 的约定适配，保留来源与 Apache-2.0 许可说明；
  不依赖 ArviZ 运行，也不把该参考算法宣称为原创。
- 原尺度 mean ESS 用于 `sample_sd/sqrt(ESS_mean)` 的均值 MCSE。
  它是 Monte Carlo 误差，不是后验标准差；无限方差目标的原尺度均值 MCSE 无可靠解释。
- 另输出经验参数相关矩阵和已有 trace 中的正式发散、无效评估、平均接受概率。

[R-hat/ESS 文献](https://arxiv.org/abs/1903.08008)、
[Stan 定义](https://mc-stan.org/docs/reference-manual/analysis.html)、
[ArviZ 0.21.0 参考](https://github.com/arviz-devs/arviz/blob/v0.21.0/arviz/stats/diagnostics.py)。

筛查阈值为 R-hat >1.01 或 bulk/tail ESS <100×链数。`no_flag` 只表示未触发当前阈值，
不是收敛证明。常数链显式标记；全部样本相同的 ESS/R-hat/MCSE 输出 null，不报告虚假的完美收敛。
链间不同、链内不动时 R-hat 可为无穷，JSON 记录 null 并保留 `constant_chain` 状态。
诊断无法发现所有链都困在同一个模态的情形，必须配合分散初值和多峰基准。

## 后验预测

核心接口 `posterior_predictive(problem, q_draws, seed)`：

1. 对每个保留的参数样本计算完整观测顺序上的无噪声相速度 `g(m)`。
2. 从完整观测协方差 `C=LLᵀ` 抽取 `y_rep=g(m)+Lz, z~N(0,I)`。
3. 保存观测与模拟观测的平方白化残差；不把相关噪声替换成独立边缘噪声。

当前固定原观测协方差；没有引入噪声超参数推断或新的观测网格。
根失败或支持外样本保留原行，值为 NaN，记录状态；不通过删除失败样本改变预测分布。
噪声随机流按原行消费，失败不改变后续样本的随机数对应关系。

`predict --run ...` 在每条链按 `floor(j*N/K)` 选取 K 个样本，仅用于限制预测计算量，
不是采样器 thinning。输出 latent、replicated、selected log Vs、原链编号和原样本索引，
连同配置、观测和完整协方差。正式预测存在失败返回 5；绘图脚本拒绝静默过滤失败行。

## 复现

```bash
python scripts/posterior_diagnostics.py results/m5/final --output results/m6/m5-diagnostics.json
build/dispweave predict --run results/m5/final --output results/m6/predictive \
  --draws-per-chain 200 --seed 2026
python scripts/plot_predictive.py results/m6/predictive
```

这些命令的输入是 M5 已保存的完整正式样本，无需重新采样。
诊断 JSON 和预测输出目录均拒绝覆盖既有结果。NumPy 仅用于 Python 处理；
没有 NumPy 的机器仍可构建 C++ 核心，CTest 会跳过对应 Python 诊断测试。

## 独立验证和实际结果

版本 0.7.1。Release 完整回归 37/37（1.87 s），新增预测/诊断 UBSan 定向回归
2/2（1.77 s）。新增 CLI 的正常与失败路径已在 Release 工作流回归中验证。
后续仅版本号和文档调整未改变数值算法。

在 `/tmp/dispweave-validation` 中安装 ArviZ 0.21.0 用于独立比对，未加入生产依赖。
测试参考值保存于 `tests/fixtures/diagnostics_arviz.json`；
`tests/diagnostics_tests.py --reference` 可在有该版本 ArviZ 的环境重新比较。

- 250 组随机四链数据（50 种子 × 5 种长度）R-hat、bulk/tail ESS 与 ArviZ 相对容差 1e-10 内一致。
- 已纳入回归的 8 类案例：独立高斯、正相关 AR、反相关 AR、ties、Cauchy、位置分离、
  尺度不一致、链内常数但链间不同；R-hat、bulk/tail ESS、mean MCSE 对照容差 1e-9。
- 原尺度常数、非有限输入、链数不足和过短输入有独立失败测试。
- C++ 12000 次模拟检查完整协方差、均值、期望 chi-square、随机重放、无噪声预测，
  以及支持外/缺模态保留失败行。
- CLI 回归检查多链抽样索引、输出行数、相同种子重放、拒绝覆盖及缺模态失败结果。

M5 三链、每链 4000 正式样本的诊断：

| 参数 | R-hat | bulk ESS | tail ESS | mean MCSE (km/s) | 状态 |
|---|---:|---:|---:|---:|---|
| Vs 0 | 1.00146 | 6048.3 | 7311.1 | 0.0000853 | no_flag |
| Vs 1 | 1.00230 | 1418.8 | 3787.2 | 0.001746 | no_flag |
| Vs 2 | 1.02464 | 93.8 | 160.3 | 0.02749 | warning |

因此 M5 半空间混合不足的观察得到量化，不能把其零发散和高接受率作为充分混合证据。
上述 ESS/MCSE 的精度也受有限链长限制，应在改进采样后重新评估。

600 个预测样本全部成功。观测 chi-square 平均 9.54393，模拟观测 chi-square 平均
14.01055（14 个观测），`Pr(T_rep >= T_obs)` 的样本估计为 0.77667。
这是用过同一数据拟合后得到的 posterior predictive discrepancy，不是服从均匀分布的
频率学校准 p 值，也不是可信区间覆盖率试验。预测图展示逐点区间，不是同时置信带。
半空间链的诊断警告仍适用于这些预测，拟合良好不能消除参数混合问题。

下一项 M6b 用分窗口质量矩阵适应改善参数尺度差异；M6c 再交付 NUTS 及完整的
高相关、漏斗、多峰、重复合成覆盖率与先验敏感性验收。
