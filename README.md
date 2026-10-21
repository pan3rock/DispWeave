# DispWeave

An extensible framework for adaptive multimode surface-wave inversion and joint seismic inference.

面向一维层状介质，研究多阶 Rayleigh 频散如何约束地下结构，以及数据支持多少模型自由度。
长期目标是物理可靠性、自适应复杂度和可靠的不确定性评估。

## 当前状态

**M0：仓库审查、架构与验收计划。尚未发布可运行求解器。**

采用 C++17 主体，Python 负责数据处理与可视化；使用 CMake、Eigen、toml11、CLI11。
正演与核函数复用 invLM，并在引入时保存源码、许可证、来源版本及文件校验值。
不使用 disba，不评估 DispersionRootLab。

按独立里程碑开发，每项完成测试与结果记录后单独 commit。当前仅完成设计，
不宣称已经通过频散、梯度、反演或采样的数值验证。

## 设计与开发

- [现有代码审查](docs/code-audit.md)：invLM 实际接口、复用边界与依赖。
- [架构和数学接口](docs/architecture.md)：单位、数据结构、错误语义和后验定义。
- [里程碑与验收](docs/milestones.md)：开发顺序、测试门槛和每次提交范围。
- [采样器实现调查](docs/samplers.md)：HMC/NUTS、RJMCMC 的开源参考与自有实现路线。

下一项为 **M1：C++ 构建基础与 invLM 正演基线**。先证明根的正确性，
再验收核函数，然后接入固定层数 LM。HMC、RJMCMC 和联合反演属于后续独立阶段。
