# M8 办公室接续说明

交接日期：2026-10-23。用户要求暂时收尾，目标已设为 **paused**。
本次仅作本地 WIP 保存，不表示 M8d/M8 已完成，没有 push 或发布。
全部九组长链已自然结束，运行会话 3833 最终返回 0；无需停止或重复启动旧进程。

## 代码与数据

- 主分支：main。此前最后完成的里程碑为 M8c，commit `56dfc61`；本次 WIP 在其后。
- M8a 椭圆率、M8b RF、M8c 组合目标已独立提交。M8d 设计见 `docs/m8-design.md`，
  已实现内容与数值记录见 `docs/m8d-validation.md`。
- 生产代码 C++17、CMake、随仓库提供的 Eigen/toml11/CLI11/invLM；处理与绘图用 Python。
- 长链、输入、网格参考、短流程、图表在 `results/m8/joint-validation/`，不进 Git。
- 原有 `scripts/build_demo_page.py` 和 `web/` 保持未跟踪，没有混入本次 WIP；另行打包供迁移。
- `build-m8/` 是最终独立 Release 构建；办公室应重新构建，不复制机器相关的二进制目录。

## 暂停时实际状态

全部实验为 4 链，每链 4000 预热 + 20000 正式样本，每链 200 次后验预测。
九组都有 `diagnostics.json`，预热及正式期数值/正演失败均为 0。

| 案例 | 最大 R-hat | 最小 bulk ESS | 当前验收 |
|---|---:|---:|---|
| dispersion | 1.00085 | 4409 | 已与独立积分比较 |
| ellipticity | 1.00173 | 3876 | 已与独立积分比较 |
| receiver | 1.00086 | 5370 | 已与独立积分比较 |
| joint | 1.00202 | 3115 | 已与独立积分比较 |
| joint-log-prior | 1.00167 | 2455 | 已与独立积分比较 |
| conflict | **1.54046** | **7.23** | **混合不佳；必须检查链和参考后验，不能宣称收敛** |
| conflict-receiver | 1.00174 | 4194 | 尚未做积分比较与图表审阅 |
| conflict-wide-noise | 1.00363 | 1210 | 尚未做积分比较与图表审阅 |
| conflict-fixed-noise | 1.00095 | 3504 | 尚未做积分比较与图表审阅 |

一致与冲突数据的完整 Vs 网格各有 32361 个节点，全部正演有效。
五组已检查实验的 Vs 与噪声尺度均值满足预先声明的积分/MCSE 容限。
完整九组 `grid-comparison.json` **尚未生成**。

已完成验证：最终 Release **60/60**（27.67 s）；UBSan 定向 **11/11**，
详细输出变更后再验联合 CLI/参考积分 **2/2**；449 个依赖文件校验通过。
最终构建重放短流程，19 个采样、目标、LM 和预测文件逐字节一致；安装入口检查通过。

重要限制：联合 LM 在一致数据上以 step_stagnation 停止，投影梯度约 4.09e-4，
没有达到 1e-5。保留诊断，未放宽容限；后验链使用独立初值，不依赖 LM。
一致联合浅层 95% 区间没有覆盖生成真值；已如实写入报告，不更换种子掩盖。
RF 每次重建完整波形是性能热点，**本轮没有修改 RF 算法或加入选点重建优化**。

## 在办公室恢复

交接包中 `DispWeave-m8-wip.bundle` 包含源码和 Git 历史；两个 tar.gz 包包含
联合实验结果与原有本地网页文件。`SHA256SUMS` 用于检查传输完整性。
在一个新的目录恢复，避免覆盖办公室已有工作区：

```bash
sha256sum -c SHA256SUMS
git clone DispWeave-m8-wip.bundle DispWeave
tar -xzf joint-results.tar.gz -C DispWeave
tar -xzf local-web.tar.gz -C DispWeave
cd DispWeave
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

分析要求 Python >=3.11、NumPy；绘图另需 Matplotlib。核心编译无需 Python 或联网。
Bundle clone 的 origin 指向 bundle 本身；不会自动关联或推送 GitHub。

办公室告诉 Codex：**“恢复 M8，先读 docs/handoff-m8.md，保留现有结果，从冲突实验验收继续。”**
新会话需读取此说明；当前会话的目标已暂停，不应自动继续。

## 下一步（尚未执行）

1. 检查 `conflict` 各链位置与轨迹，对照完整网格判断是否困于不同后验峰。
   不丢弃停滞链、不放宽 R-hat 阈值、不自动扩大到 RJ-HMC；如需重跑，使用新目录保留原始证据。
2. 对全部九组运行独立积分比较。脚本会保存报告并在均值对照失败时返回非零；
   对 `conflict` 的失败应分析原因，不能仅把命令执行完视为验收通过。
3. 生成并审阅完整图表，补齐逐块拟合、预测、相关性、噪声/先验敏感性与方法局限。
4. 同步 README、milestones 和 M8d 报告，按实际证据审查 M8a–d 是否全部完成；
   然后再作正式 M8d commit。当前 WIP 不是里程碑完成提交。

```bash
OPENBLAS_NUM_THREADS=1 python scripts/check_joint_grid.py results/m8/joint-validation \
  --consistent-grid results/m8/joint-validation/consistent-grid.txt \
  --conflict-grid results/m8/joint-validation/conflict-grid.txt \
  --output results/m8/joint-validation/grid-comparison.json
python scripts/plot_joint.py results/m8/joint-validation --suite --output results/m8/joint-plots
```

不要直接对这个目录再次执行 `run_joint_demo.py --run-prepared`：已有输出受保护，
它不是自动续跑开关。所有数据、链与参考网格已经保存，可直接分析。
