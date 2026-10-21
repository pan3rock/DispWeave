# 现有代码审查

审查日期：2026-10-21。此项是源码与环境审查，不是数值认证或性能测试。

## 当前仓库

开始审查时，DispWeave 没有 Git 提交，没有模型、正演、反演、测试或构建代码。
本轮开发方向调整后的决定：只提交 M0 文档；未经编译验证的探索草稿不进入版本库。
没有修改 invLM 或其他已有项目。

## invLM 来源

本地可访问路径：`/home/lei/Work/invLM`。
审查时 HEAD：`86ad6fa5aaec3f44c6ddcac3d69b27a2f13c2e88`，工作区未报告修改。
仓库根 LICENSE 为 Apache-2.0，未发现根 NOTICE 文件。
引入时仍需检查选取文件内的来源声明，完整保留许可证及第三方归属；
本次不把仓库顶层许可证视为所有间接依赖的统一授权。

### 已读接口

| 文件或模块 | 实际能力 | DispWeave 处理 |
| --- | --- | --- |
| `include/model.hpp` | `VelocityModel` 的层顶深度、Vs、Vp、rho，含半空间 | 使用明确验证的模型结构，转换为该数值布局 |
| `dispersion.hpp/.cpp` | `Dispersion::search(f, num_mode)` 返回有序根；`search_mode` 支持单阶及 fallback 标记 | 优先按频率批量搜索，恢复观测原始行序；禁止 fallback 进入拟合 |
| `secfunc.hpp/.cpp` | Dunkin 复合矩阵；独立的 FAST_SCALAR 实现入口 | Dunkin 主路线，FAST_SCALAR 作内部数值交叉检查 |
| `dispersion_kernel.hpp/.cpp` | `RayleighKernel::compute(omega,c)`，分离 Vs/Vp/rho 核、群速度、波场、能量残差和重构标记 | 只暴露 Phase 1 必需核及诊断；不将群速度扩展纳入首阶段 |
| `dunkin_derivative.hpp/.cpp` | `dunkin_derivatives(model,omega,c)`，三类物性导数及 root correction | 与能量核及多步长差分分别验证，不等同于自动微分求根 |
| `forward_engine.hpp/.cpp` | 查询数组、OpenMP、warm start、根与核诊断；编译开关选择直接导数 | 参考组合方式；首版不继承可变缓存状态 |
| `inversion_evaluator` / `inversion_solver` | Ceres LM、参数化、正则化及数据评估 | 参考数学和失败处理；首版不引入整个 Ceres/HDF5 依赖树 |

### 必须修正或隔离的语义

1. `search_mode` 在特定缺根条件返回 `Vs_halfspace - ctol`，并标记 fallback。
   这不是已求得的物理模态。适配层优先调用 `search`，并将未找到请求阶数记录为
   `missing_or_unresolved`，不得当作预测值。
2. `RootSearchDiagnostics` 的 `roots_found` 和 `numerical_issue` 不是根全集的完备性证明。
   未找到根不能自动解释为物理截止。
3. `ForwardEngine` 的部分核异常路径会将 `c_pred` 置为 NaN。新结果结构必须保留已求得的根，
   同时把梯度判为不可用；根、导数、截止/分支变化分别测试。
4. 上游能量核包含边界重构与能量残差检查；隐式导数包含奇异性和根修正检查。
   这些检查有价值，但不证明任意复杂层状模型的导数可靠。
5. 原项目的部分无效根策略是大残差配零 Jacobian，且历史路径允许 fallback。
   DispWeave 不照搬这种数据目标：无效 trial 拒绝，初值无效明确失败。
6. 首版关闭 warm start，保证同一模型/查询的输出不依赖前一试算。
   后续若启用，必须验证拒绝步和查询重排不会改变模态编号或后验值。

### 拟引入的最小物理源码

- 源文件：`dispersion.cpp`、`secfunc.cpp`、`secfunc_fast_scalar.cpp`、
  `toms748.cpp`、`dispersion_kernel.cpp`、`dunkin_derivative.cpp`。
- 配套头文件：`dispersion.hpp`、`secfunc.hpp`、`secfunc_fast_scalar.hpp`、
  `toms748.h`、`areclose.hpp`、`dispersion_kernel.hpp`、`dunkin_derivative.hpp`、`model.hpp`。
- 保存上游 LICENSE 和来源清单：commit、相对路径、SHA-256、是否有本地修改。
- 先原样纳入 `third_party/invLM`，项目适配代码单独维护；后续物理修复独立提交并说明依据。

这是源码依赖闭包的候选清单，**尚未完成独立编译验收**。
不需要复制 invLM 的完整应用层、区域成体脚本或全部测试依赖。

### 上游测试证据及边界

已检查 `tests/unit_tests.cpp`、`tests/forward_regression.cpp`、`docs/verification.md` 和
`docs/project_closeout.md`。上游已有普通分层、低速层、接近模态、物性导数差分、
等物性细分不变性等测试；这有助于建立回归集，但本轮没有运行其完整测试套件。

上游 golden roots 可保护迁移一致性，不能单独作为独立物理真值。
DispWeave 还必须加入均匀半空间解析验证、独立分层参考、截止行为和困难模型报告。
上游文档也明确不保证所有临界模态及地层已获认证。

## 工程环境

检测到 CMake、g++、Eigen 头文件，Python 环境中有 NumPy、SciPy、Matplotlib。
本机 vcpkg 安装中可找到 toml11 4.4.0 的 CMake 配置。
没有找到已安装 CLI11；为后续构建下载了官方 CLI11 v2.4.2 压缩包至临时目录，
尚未完成集成编译。压缩包 SHA-256：
`f2d893a65c3b1324c50d4e682c0cdc021dd0477ae2c048544f39eed6654b699a`。

本机绝对路径不能进入正式构建配置；M1 必须提供固定版本、离线可用的源码依赖方案。
Python 不参与生产正演或优化器，不作为 C++ 核心运行依赖。

## 本轮排除

按用户要求暂不评估 DispersionRootLab；也不继续采用 disba。
本轮不导入或实现采样器。`DispNUTS/third_party/mcmc` 中发现的现有 MCMCLib
仅用于[后续算法参考评估](samplers.md)，不是 DispWeave 当前依赖。
