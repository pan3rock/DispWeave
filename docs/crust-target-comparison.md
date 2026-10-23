# 合成数据与目标模型对照补图

> 2026-10-24 后续：[NUTS 收尾与原四链完整验收](crust-nuts-closeout.md)已完成。本文保留当时的实验与失败记录。

2026-10-23。本地结果页之前只展示参考模型、已有反演结果和诊断，遗漏了合成目标模型，
也未单独说明原始数据。本轮补齐目标、参考、观测与恢复结果的对应关系，保持原反演定义。

## 三类输入的区别

QEDispInv `demo/syn-crustmantle/config.toml` 的 `[forward].file_model` 指向 `model_data.txt`，
`[inversion].model_ref` 指向 `mref.txt`。本页明确称前者为**示例提供的合成目标模型**，后者为
**反演参考模型**，不将 mref 标作真值。

| 文件 | 角色 | 内容 |
|---|---|---|
| `model_data.txt` | 提供的合成目标 | 44 层；列为显示序号、深度、密度、Vs、Vp；末层半空间顶深 320 km |
| `mref.txt` | 反演参考中心 | 45 层；界面固定，末层半空间顶深 208.925 km；定义初值、速度窗口及参考先验中心 |
| `data.txt` | 原始合成观测 | 125 行；频率、相速度、零基模态；模态 0–3 分别有 43、31、31、20 点 |

目标表中的重复显示序号不影响物理深度顺序，未重新排序或合层。
模型比较按各自原生层界面绘制，没有按层号强行配对。两模型的深部均匀段也完整保留。
原数据无误差列；σ=0.02 km/s 是既有反演的假设，不是已查明的原始加噪标准差。
本轮未重建原始噪声种子或数据生成过程。

本次只读 QEDispInv HEAD `bc94cd6`，另有未跟踪的参考模型生成脚本，未执行或修改。
源 `data.txt`、`mref.txt` 和 `config.toml` 的哈希逐一匹配已冻结的正则化扫描输入。
目标文件、参考、数据和配置的原样副本及 SHA-256 保存在
`results/crust-target-comparison/provenance.json`。外部输入仍只保存在忽略的结果目录。

## 目标物性与反演参数化

反演使用 45 个自由 Vs，固定参考界面，以 Brocher05 派生 Vp/ρ。
目标模型的原始物性并不完全满足同一关系。以目标 Vs 代入 Brocher05，与文件给定值相比，
最大 Vp 差为 **0.634876 km/s**，最大密度差为 **0.140835 g/cm³**。
这是逐层确定性差异，不是测量噪声或拟合残差。

因此本轮目标预测使用文件给定的全部 Vp、Vs、ρ 和界面；没有先把目标改写为 Brocher05 模型。
参考预测则使用 mref 的 Vs 与界面及 Brocher05 派生物性，与既有反演定义一致。
LM 预测读取保存终态的全部物性。不同界面与物性约束意味着当前参数化不包含完整原目标，
不能仅凭低频散残差宣称目标结构已被完整恢复。

## 新增图及数值核对

| 图 | 展示内容 |
|---|---|
| `target-recovery.png/pdf` | 原目标、参考和已有 λ=100 的 run 0 LM 解；全深度与浅部 Vs 对照 |
| `synthetic-data.png/pdf` | 原始观测、目标原物性预测、参考预测、已有 LM 预测；四模态及标准化残差 |
| `target-nuts.png/pdf` | 原目标与参考、LM、三条已完成链的样本 95% 区间和中位数叠加 |

三组模型仅在原始 125 个频率/模态查询上重新正演，逐行核对数量与顺序，全部有效。
对应相速度 RMS 为：

| 模型 | 与原始观测的 RMS km/s |
|---|---:|
| 提供的目标，保留原物性 | 0.0185671309 |
| 参考模型，Brocher05 | 0.0485886111 |
| 已有 λ=100 LM 解 | 0.0190017709 |

目标 RMS 与最初应用测试记录的 0.0185671 一致；这支持数值配对，不重建原噪声历史。
LM 的新预测与已保存终态预测逐行差异小于 2e−9 km/s。
三条已完成链的 6000 个正式状态（不含预热、包含重复状态）重新计算的分位数
与既有 `diagnostics.json` 一致，容差 1e−12 km/s。
**NUTS 四链仍未通过验收**；新增目标叠加不改变区间仅供诊断的性质，不进行覆盖率验收。

本项没有重新优化或采样，没有根据已知目标修改参考、先验、误差、观测或初值。
数值核心与原结果文件未改变；新增三张图纳入本地网页，图集由 8 张增加为 11 张。

## 复现

```bash
MPLCONFIGDIR=/tmp/dw-mpl OPENBLAS_NUM_THREADS=1 python3 scripts/plot_crust_target.py \
  --source /home/lei/Work/QEDispInv \
  --scan results/crust-regularization-scan \
  --nuts results/crust-nuts-conservative/completed-chain-diagnostics \
  --exe build/dispweave --output results/crust-target-comparison-new
```

输出目录须不存在。失败保留已有输出，不删除或忽略无效数据。
脚本检查冻结输入、已有 LM 正演重放和样本分位数；输出来源哈希、正演输入/输出、摘要及 PNG/PDF。
网页继续仅本地查看，通过 `web/results/index.html` 直接打开。

网页在 Chromium 断网模式通过检查：11 张图及 11 个 PDF 均存在，新增分类、默认目标图、
大图切换与键盘操作正常；目标叠加的 NUTS 图仍显示未通过警示。
375–1920 px 的六种宽度无页面横向溢出，关闭 JavaScript 仍显示目标对照图；
无脚本运行错误、无外部请求。记录与截图位于 `results/results-web-target-validation/`。
本轮没有修改 C++，未重新运行数值核心 CTest。
