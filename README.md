# DispWeave

An extensible framework for adaptive multimode surface-wave inversion and joint seismic inference.

面向一维层状介质，研究多阶 Rayleigh 频散如何约束地下结构，以及数据支持多少模型自由度。
长期目标是物理可靠性、自适应复杂度和可靠的不确定性评估。

## 当前状态

**M4b：四类模型的候选参数化与复杂度准则对照。**

已提供 invLM Dunkin 正演、能量核与隐式导数、逐行根/梯度诊断、固定 Vs 参数化、
完整协方差似然和明确测度的后验目标，并已接入自有有界 LM、多初值与失败诊断。
Phase 1 最小可运行版本已完成；新增参考模型处的灵敏度谱、显式参数尺度和岭分辨矩阵。
已增加 AIC/BIC 与独立验证预测评分，保留未收敛、触界和不完整比较状态。
自适应分层尚未实现，下一项 M4c；采样器尚未实现。
每个里程碑完成验收后单独 commit，不提前铺开全部算法。

C++17 为主体，Python 用于数据处理和绘图。Eigen 3.4.1、toml11 4.4.0、
CLI11 2.4.2 和所需 invLM 数值源码随仓库提供，构建无需联网或安装反演框架。

## 构建与运行

已验收 Linux、GCC 13.3、CMake 3.28；需要 C++17 编译器与 CMake ≥3.16。
不需要 Python、Ceres、HDF5、BLAS 或包管理器即可编译运行核心。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
build/dispweave forward --config examples/forward/normal.toml --output build/normal.txt
```

配置中的 `queries.frequency_hz` 与 `queries.mode` 按行配对，可不同模态不等采样、乱序、重复。
模型数组为层顶深度 km、Vs/Vp km/s、密度 g/cm³；最后一层是半空间，模态从 0 开始。

输出文本包含版本、模型、每个原始查询的速度、状态和诊断。
`roots_found` 只是本次找到的请求根前缀数量，不是全部模态数或完备性证明。

```bash
build/dispweave forward --config examples/forward/missing.toml --output build/missing.txt
```

这个半空间示例特意请求不存在的高阶根，进程返回 **2**，输出仍保留两行。
无效相速度为 `nan`，状态为 `missing_or_unresolved`；不能当作有效频散参与拟合。
返回码 **0** 表示所有请求均得到有效根；**1** 表示配置、模型或 I/O 错误。
CLI11 参数解析错误使用其自身非零返回码。
输出文件的父目录需存在；已有结果文件会被覆盖，但禁止输出指向输入配置文件。

可选安装到自己的目录：

```bash
cmake --install build --prefix "$PWD/build/install"
build/install/bin/dispweave --version
```

可选源码校验与性能测量：

```bash
python scripts/verify_vendor.py
/usr/bin/time -v build/forward_benchmark normal
/usr/bin/time -v build/forward_benchmark dense_low_velocity
```

## 灵敏度与目标函数

```bash
build/dispweave forward --config examples/forward/normal.toml \
  --sensitivity energy --output build/energy.txt
build/dispweave forward --config examples/forward/normal.toml \
  --sensitivity implicit --output build/implicit.txt
build/objective_example
```

两种灵敏度接口独立保留有效根和梯度诊断。请求梯度但只有梯度失败时返回码为 3。
后验示例通过 C++ API 使用相关误差，输出目标分解及梯度，不执行优化或采样。
见 [M2 接口与验收](docs/m2-validation.md)。

## 固定层数反演

```bash
build/dispweave synthetic --config examples/inversion/synthetic.toml --output results/synthetic
build/dispweave invert --config examples/inversion/synthetic.toml \
  --data results/synthetic/observations.txt --covariance results/synthetic/covariance.txt \
  --output results/replay
python scripts/plot_inversion.py results/synthetic
```

反演输出目录必须新建或为空；保存所有初值、终止原因、正演/梯度异常和试算轨迹。
若没有初值通过收敛标准，返回码 **4**，不能将末次模型当成已收敛解。
观测列为 `frequency_hz phase_km_s mode sigma_km_s`；可提供完整协方差。
当前固定层顶与密度，反演 Vs，Vp 固定或按指定 Vp/Vs 随动。

一条命令完成构建、测试、六组基阶/多模态及噪声对照、绘图与观测重放：

```bash
python scripts/run_inversion_example.py --output results/m3
```

绘图需要 NumPy、Matplotlib；核心无需它们。参数、数学定义和实际收敛局限见
[M3 验收报告](docs/m3-validation.md)。反演结果没有后验可信区间，触界不代表数据可靠约束。

## 局部灵敏度与分辨诊断

```bash
build/dispweave resolution --config examples/resolution/reference.toml --output results/local-resolution
python scripts/run_resolution_example.py --output results/m4a
```

在固定 Vp、密度、界面的参考模型处，输出白化且按声明参数尺度缩放的奇异值、
数值秩、给定岭强度下的有效维度和模型分辨矩阵。支持完整观测协方差。
数值秩不等于可辨识层数，分辨矩阵不等于后验协方差；本项未自动选取层数或正则化。
见 [M4a 数学定义与验收](docs/m4a-validation.md)。

## 候选参数化对照

```bash
python scripts/run_selection_study.py --output results/m4b
python scripts/plot_selection_study.py results/m4b
```

四类模型、三组模态、三种频段/覆盖、两种噪声和两个种子，共 432 次候选拟合。
`--smoke` 可运行 6 个拟合的小案例。优化与评分在 C++，Python 负责实验组织和绘图。
有 `[assessment]` 配置时，`invert` 输出 AIC/BIC、局部信息量，并可通过
`--validation-data`（可选 `--validation-covariance`）计算独立验证预测似然。
训练与验证误差块必须独立。未收敛候选不排名，触界等非正规解不参与 AIC/BIC。
各准则仅能在同数据场景内比较；详见 [M4b 验收及局限](docs/m4b-validation.md)。

## 已知边界

支持全固体、各向同性、弹性的一维模型；允许低速层和高速夹层。
提供 Rayleigh 相速度及 Vs/Vp/rho 灵敏度，不支持液层、泄漏波或层厚导数。
缺根可能是物理截止，也可能是搜索未解析，不能据此断言模态不存在。
上游没有提供一般 Rayleigh 根全集的完备性证明。

本次修复了低 Vp/Vs 半空间的搜索下界漏根，保留原版与修改版校验值；
普通分层和部分低速层根已通过独立边界匹配验证。
高频、强对比及复杂根分布的全面认证仍未完成，具体覆盖见验收报告。

## 文档

- [M4b 验收结果](docs/m4b-validation.md)
- [M4a 验收结果](docs/m4a-validation.md)
- [M3 验收结果](docs/m3-validation.md)
- [M2 验收结果](docs/m2-validation.md)
- [M1 验收结果](docs/m1-validation.md)
- [现有代码审查](docs/code-audit.md)
- [架构和数学接口](docs/architecture.md)
- [里程碑与验收](docs/milestones.md)
- [采样器实现调查](docs/samplers.md)
- [第三方源码及授权](third_party/README.md)

DispWeave 自有代码使用 Apache-2.0；第三方代码保留其各自许可证和来源声明。
