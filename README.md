# DispWeave

An extensible framework for adaptive multimode surface-wave inversion and joint seismic inference.

面向一维层状介质，研究多阶 Rayleigh 频散如何约束地下结构，以及数据支持多少模型自由度。
长期目标是物理可靠性、自适应复杂度和可靠的不确定性评估。

## 当前状态

**M1：C++ 构建与多阶 Rayleigh 相速度正演基线。**

已提供模型验证、invLM Dunkin 正演适配、逐行根诊断、命令行与数值测试。
核函数、似然、LM 和采样器尚未接入；下一项是 M2。
每个里程碑完成验收后单独 commit，不提前铺开全部算法。

C++17 为主体，Python 用于后续数据处理和绘图。Eigen 3.4.1、toml11 4.4.0、
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

## 已知边界

支持全固体、各向同性、弹性的一维模型；允许低速层和高速夹层。
只提供 Rayleigh 相速度，不支持液层、泄漏波或梯度接口。
缺根可能是物理截止，也可能是搜索未解析，不能据此断言模态不存在。
上游没有提供一般 Rayleigh 根全集的完备性证明。

本次修复了低 Vp/Vs 半空间的搜索下界漏根，保留原版与修改版校验值；
普通分层和部分低速层根已通过独立边界匹配验证。
高频、强对比及复杂根分布的全面认证仍未完成，具体覆盖见验收报告。

## 文档

- [M1 验收结果](docs/m1-validation.md)
- [现有代码审查](docs/code-audit.md)
- [架构和数学接口](docs/architecture.md)
- [里程碑与验收](docs/milestones.md)
- [采样器实现调查](docs/samplers.md)
- [第三方源码及授权](third_party/README.md)

DispWeave 自有代码使用 Apache-2.0；第三方代码保留其各自许可证和来源声明。
