# Rayleigh 近邻根搜索修复

2026-10-23，按用户要求同步修复 DispWeave 内置后端和本地 invLM。
这是 [QEDispInv 示例测试](qedisp-examples-validation.md) 暴露问题的修复，
不新增模型选择、自适应分层或采样算法。

## 原因与最小修改

原 `locate_extremum` 只有在中间采样值同时大于或小于两侧时才进行抛物线插值。
这不是插值顶点位于区间内的必要条件；看似单调的三点之间也可能隐藏极值和两个过零点。
按 QEDispInv 的做法，现对每组三个相邻点尝试插值，顶点严格位于区间内才计算频散函数。

保留已有累计补点、零值采样根、数值异常、受俘获波搜索域和 Love 模态计数处理。
没有修改 Dunkin/fast-scalar 的频散方程、核函数、根去重阈值、补点轮数或模型约束。
没有把整份旧版 QEDispInv 搜根代码覆盖进来，也未修改 QEDispInv 仓库。

对于保存的地壳终态，f=0.392 Hz，Vs[15] 乘 exp(1e-5) 后：

| 查询 | 修复前 | 修复后 (km/s) |
|---|---:|---:|
| mode 3 | 4.170184952 | 3.937656640 |
| mode 4 | 4.317773053 | 3.937716232 |

二根相距约 `5.9592e-5 km/s`（0.0596 m/s），附近初始采样间距约 22.8 m/s。
原补点未进入二根之间的负值区，根区间没有生成；提高 TOMS748 精度无法恢复缺失区间。
修复后的候选区间包含两次过零，后续模态不再因这一对漏根而被前移。

## 独立物理核对

新增验证工具 `benchmarks/rayleigh_boundary_reference.py`：显式组装各层 P/SV 波的
位移、应力连续性及地表零应力条件，半空间只保留向下衰减波。
有限层波按端点定标，避免直接使用指数增长的传播矩阵；求归一化全局矩阵的行列式零点。
端点定标引入的已知复相位在符号比较前移除。
它不调用 invLM 或 Dunkin，不作为生产正演后端。

174×174 边界矩阵给出的两个根为 `3.937656639506221`、`3.937716231795848 km/s`，
与修复后结果相差小于 `1e-12 km/s`；根处最小/最大奇异值比约 `2.8e-14`。
另外核对前三个根，及 Vp/Vs=1.2、1.73、3 的均匀半空间独立三次方程解，均一致。
这是指定区间内的根存在性验证，不是整个相速度区间的模态计数证书。

## 回归与精度

两个仓库保存完全一致的独立可运行夹具和 C++ 测试，不依赖外部 QEDispInv 示例：

- 原实现确实失败；修改后默认 Dunkin 通过。
- Vs[15] 的 log 扰动为 -1e-4、-1e-5、0、1e-5、1e-4；均保留近邻根。
- 单模态请求、五模态和七模态请求的前缀一致，不使用替代相速度。
- 对两个近邻分支、五个不同深度参数核对 log-Vs 导数。
- 共用候选搜索中的 Love 计数仍保留间距约 `5.75e-8 km/s` 的两根。

近邻分支的曲率很大：Vs[29]、mode 3 在 h=1e-5 时中心差分误差约 3.25e-5，
减至 h=1e-6 后约 4.64e-7；h=1e-7 开始受默认根精度影响。
回归使用 h=2e-6、1e-6，容限 `2e-6 + 2e-4*abs(kernel)`，
最大实际绝对差约 `1.3791e-6`。步长扫描保存在 `results/root-repair/clustered-fd-sweep.txt`。

新增区间也改变部分普通模型根的最后几位舍入，使原有两个有界 LM 测试在严格梯度阈值前停滞。
DispWeave 固定层数 LM 因此与既有联合 LM 一样使用 `RootAccuracy::refined`：
绝对容限 0，相对容限 8×机器精度，并将 `root_accuracy=refined` 写入结果元数据。
LM 更新公式、接受规则、收敛阈值和正则化均未修改；独立 forward 与采样入口的求根容限不变。

实际验证：

| 检查 | 结果 |
|---|---|
| DispWeave 完整 Release CTest | **63/63**，20.53 s |
| UBSan：近邻根、golden、独立根、排序、搜索域、灵敏度 | **6/6**，11.76 s |
| invLM 原生正演/隐式导数源码独立编译 | 默认 Dunkin 回归通过 |
| 第三方文件哈希 | **449/449** |
| QEDispInv 四种模型配对，305 次参考查询 | 全部有效，交叉正演最大差仍小于 5e-8 km/s |
| 四个参考模型的多步长 FD | 全部通过 |

invLM 本机没有完整依赖环境。尝试配置其原生 CMake 后，利用已有头文件解决 CLI11 查找，
仍因缺少 Ceres 而停止，因此**未声称 invLM 完整 Ceres CLI/原有 Catch2 全套通过**。
实际直接编译了 invLM 自身的 `dispersion.cpp`、`secfunc.cpp`、`secfunc_fast_scalar.cpp`、
`dunkin_derivative.cpp`、`toms748.cpp` 和新回归；无需安装额外软件。

## 性能与应用重跑

同机 GCC 13.3 Release，每种程序五次执行、每次重复 100 次正演，取中位数。
每次正演为 40 个频率/模态查询；不含 Python 或文件 I/O：

| 案例 | 首次 ms，原→新 | 重复均值 ms，原→新 |
|---|---|---|
| normal | 0.1081 → 0.1406 | 0.06521 → 0.08532 |
| dense_low_velocity | 0.2817 → 0.3213 | 0.21487 → 0.27111 |

额外补点有约 26%–31% 的重复计算开销，两个基准的有效/未解析数量保持不变。
本轮选择修复正确性，不为抵消这一开销加入额外筛选启发式。

使用相同初值、边界、数据、σ 和 200 次迭代预算重跑四种配对：

| 案例 | 最终结果 |
|---|---|
| nearsurface | 4/4 收敛 |
| lvl-l4-m1 | 4/4 收敛 |
| 原 lvl-l4 配对 | 1/4 收敛、3 次停滞；cost 仍约 695.075，模型/数据配对限制仍在 |
| crustmantle | 1 次停滞、2 次到迭代上限、1 次初值未解析；0/4 收敛 |

地壳前三次最终 cost 为 307.4668、1306.0100、40.7824，投影梯度仍约 1.000、0.971、0.541。
**指定的近邻根漏检已修复，不代表 44 参数反演已经成功。** 未增加正则项、放宽终止条件，
也未把迭代到限当成收敛。新的终态/路径还需单独检查其他漏根、强非线性和弱约束参数的影响。
本轮不重新认证已有长链后验结果。

`FAST_SCALAR` 虽共享修改后的搜索流程，在此困难模型上仍漏掉两根；
其频散函数定标不同，当前有限轮极值补查仍不足。测试程序的 `--fast-scalar` 入口保留这一失败。
默认 Dunkin 的修复不能外推为全部后端、全部模型均已可靠，也没有新增 Rayleigh 根计数保证。

## 复现

```bash
cmake -S . -B build-root-repair -DCMAKE_BUILD_TYPE=Release
cmake --build build-root-repair -j2
ctest --test-dir build-root-repair --output-on-failure
build-root-repair/rayleigh_clustered_regression
build-root-repair/rayleigh_clustered_regression --dump-fixture > /tmp/clustered-model.txt
OPENBLAS_NUM_THREADS=1 python benchmarks/rayleigh_boundary_reference.py /tmp/clustered-model.txt \
  --frequency .392 --bracket 3.93765 3.93769 --bracket 3.93769 3.93775 \
  --output /tmp/clustered-reference.json
```

参考输出拒绝覆盖旧文件。`--fast-scalar` 当前预期返回非零。
外部示例按原测试脚本运行，换用 `build-root-repair/dispweave` 和新结果目录即可。
最终数据在 `results/root-repair/final-examples/`；修复前与仅改搜索阶段的输出均保留。
`results/root-repair/` 同时包含独立参考、有限差分扫描、性能与各项验收日志。
