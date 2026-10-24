# 推荐配置与本地结果查看

本算例推荐 slice NUTS、目标接受率 0.95、合理步长搜索、标准 dense 适应和逐初值 GN。
[完整四链验收](nuts-original-095-search.md)已通过；这是本算例的推荐配置，不更改库级默认值。

## 现成输入与命令

完整配置与数据已打包在 `results/crust-results-recommended/inputs/`，与已验收运行逐字节一致：

- `nuts.toml`：全部模型、四初值、先验、边界与采样设置。
- `observations.txt`、`covariance.txt`：相同观测与协方差。
- 上一级 `run.sh`、`RUN.txt`：直接调用现有 CLI 的命令。

从项目根目录执行，输出目录须为新的目录：

```sh
sh results/crust-results-recommended/run.sh results/crust-nuts-recommended-new
```

此直接 CLI 命令串行运行四链，耗时不能与四进程验收的约 5 分钟直接相比。
返回 0 仍需计算 R-hat/ESS，不能直接声明验收通过。建议实际复现实验时使用已有完整并行验收入口：

```sh
python scripts/benchmark_nuts_original_starts.py --exe build/dispweave \
  --reference results/crust-nuts-final --output results/crust-nuts-recommended-validation-new \
  --with-step-search --timeout 1800
```

该入口从原冻结输入构建同一推荐配置，四个数值进程运行，完成后执行混合诊断、梯度核查和预测。
不覆盖已验收目录。本轮仅保存命令，没有再次执行采样。

## 查看与更新结果

直接打开 [本地结果页](../web/results/index.html)，主区只有三张图：
模型对照、NUTS 预测与观测、目标/参考/LM 与原始数据。采样轨迹、先验/后验及效率图收在补充区。
页面显示的是本次 4000 个正式状态，原 8000 状态的 0.99 结果仍可从补充区打开。

更新图及配置包时复用已保存样本、预测和旧目标对照缓存，不做正演或采样：

```sh
MPLCONFIGDIR=/tmp/dw-matplotlib python scripts/refresh_crust_results.py \
  --nuts results/nuts-original-095-search --comparison results/crust-target-comparison \
  --output results/crust-results-recommended-new
```

输出目录必须不存在；生成后按需要更新页面相对路径。脚本核查验收状态、输入哈希、
目标/参考/观测配对及后验分位数。配置、外部源数据和生成图保存在忽略的 results 目录，
新克隆需恢复这些冻结数据后再运行刷新脚本。

目标模型保持 44 层原生界面和物性；反演为 45 个固定界面的 Vs，并用 Brocher05 派生物性。
参考模型是先验中心，不是真值。模型区间是逐层条件后验；观察数据约束时应同时查看先验／后验图，
不能把先验造成的窄区间全部解释为数据分辨率。
