# HMC 与 RJMCMC 开源实现调查

调查日期：2026-10-21。已有可参考的完整开源算法，无需从论文公式孤立地重建全部经验。
用户最终选择是维护项目内实现，以减少使用者安装和编译外部框架的负担。
下列项目因此作为算法参考、可选代码来源和独立验证对象，尚未引入或运行。

## HMC 和 NUTS

| 项目 | 已确认的能力 | 对 DispWeave 的意义 |
| --- | --- | --- |
| Stan | C++ HMC/NUTS，预热适应和成熟服务接口；核心/Math/CmdStan 为 BSD 类许可 | 优先独立对照算法和诊断；整个依赖栈不进入首版 |
| MCMCLib | C++ HMC/NUTS，支持 Eigen，回调提供 log density 与梯度；Apache-2.0 | 较接近目标结构，可审查实现或合规摘取；本地已有副本 |

来源：[Stan 官方源码说明](https://github.com/stan-dev/stan/blob/develop/README.md)、
[Stan 采样配置](https://mc-stan.org/docs/cmdstan-guide/mcmc_config.html)、
[MCMCLib 官方仓库](https://github.com/kthohr/mcmc)。

本地检查位置：`/home/lei/Work/DispNUTS/third_party/mcmc`，已读 README、LICENSE
和 `header_only_version/mcmc/nuts.hpp`。`nuts` 接收初值、目标 log density/gradient 回调、
样本矩阵及用户数据。这里只确认接口，不代表本地版本与网上最新源码一致，
也未验证其质量矩阵适应、发散信息和 ESS/R-hat 是否满足本项目要求。
正式参考或复制前须固定具体版本并检查实现与依赖。

Stan 的通用 C++/自动微分接口不等于可以直接传入任意 `double` 黑盒。
接入 invLM 仍需可微包装或合适的目标适配，并验证手工梯度传递；不能让自动微分
穿透离散根搜索。自有实现同样受这一数学约束。

## RJMCMC

| 项目 | 已确认的能力 | 适用方式与待核查项 |
| --- | --- | --- |
| ANU/iEarth rj-MCMC | 1D/2D 分区跨维推断，允许用户正演，提供 C/Fortran/Python 调用 | C 接口适合 C++ 框架；官方下载需注册，未取得完整源码/许可，暂不定为可分发依赖 |
| GeoscienceAustralia rjmcmcmt | C++ 一维大地电磁跨维反演，使用 ANU RJMCMC 引擎 | 参考地球物理适配方式；应用许可证不能替代引擎许可证核查 |
| BayesBay | Python/Cython 跨维、层级推断，默认 RJMCMC，BSD-2-Clause | 参数化与提议设计参考、可选独立数值对照，不改变 C++ 主体路线 |
| BayHunter | Python 面波/接收函数跨维反演，GPL-3.0 | 地球物理实验参照；直接复用代码需考虑其分发许可要求 |

来源：[ANU/iEarth 官方项目页](https://iearth.edu.au/codes/rj-MCMC/)、
[RJMCMCMT 官方仓库](https://github.com/GeoscienceAustralia/rjmcmcmt)、
[BayesBay 官方仓库](https://github.com/fmagrini/bayes-bay)、
[BayHunter 官方仓库](https://github.com/jenndrei/BayHunter)。

“完整 RJMCMC 实现”不意味着替换 likelihood 就自动得到正确的层状模型后验。
层数、界面排序、最小层厚、半空间处理和参数变换改变先验测度及提议密度。
每个 birth/death 对必须明确正反映射、辅助随机量、选择概率和 Jacobian。
层数上下界处还要考虑提议种类概率的变化。

## 项目内实现的路线

1. 正演、核函数和固定层数 LM 已交付；近期先完善固定层数反演，LM 自适应分层暂缓。
2. 在已验证的统一 log posterior 上先实现固定层数、固定轨迹 HMC，使用解析概率目标验收；不以自适应分层为前置条件。
3. NUTS 单独立项，明确采用哪种候选抽取和终止规则，不能混用不同版本的局部步骤。
4. RJMCMC 独立实现，先用已知层数后验的 toy problem 验证，再进入真实频散。
5. 随机数、变换、先验和诊断可共享；固定维度与跨维状态无需强行统一。
6. 对照成熟实现验证分布、收敛和效率，不能把“自己的实现”当作免除验证的理由。

减少分发依赖会增加本项目的算法维护和测试责任。若后续选择摘取开源源码，
保留许可证、作者归属、固定版本和修改清单；不把复制实现宣称为独立原创。
在其验收完成前，不承诺该实现与成熟框架具有相同稳健性。
