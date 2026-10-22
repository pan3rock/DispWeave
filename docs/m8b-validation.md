# M8b：P 接收函数正演与处理契约

状态：实现、独立数值验证与示例完成。这里只验收正演；M8c 组合目标和 M8d 联合反演仍待完成。

## 物理接口与计算方法

`receiver_spectrum(model, frequencies, p)` 接受一维各向同性弹性层状模型，
复用 LayeredModel：层顶 km、Vs/Vp km/s、rho g/cm³，末层为半空间。
输入频率 Hz，水平慢度 p 为 s/km，要求 `0 <= p < 1/Vp_halfspace`。
首版仅上行 P 入射，单位位移振幅定义在半空间顶面。x 沿水平传播方向，内部 z 向下，
输出 R 水平、Z 向上，时间约定 `exp(-i*omega*t)`。这些约定决定极性及复相位，不能任意拟合校正。

有限层包含上/下行 P 和 SV，满足自由表面零应力、全部界面位移与应力连续、
半空间辐射条件。求解完整复边界系统，包含所有转换和多次反射；不是只合成 Ps 走时脉冲。
每种波的未知振幅锚定在其入射边界，只出现衰减型倏逝传播因子，避免直接连乘增长型传递矩阵。
系统按行平衡，再用 Eigen FullPivLU；未知数为 `4*(layers-1)+2`。
支持不超过 128 个总层，属于明确资源限制，不是该层数下精度的保证。

没有新增生产依赖。invLM 继续负责频散/椭圆率；入射体波响应采用单独的 C++ 实现，
不修改 invLM 正演或采样器。外部 Fortran、LAPACK、SciPy 只用于生成独立验收参考。
无流体、衰减、各向异性、倾斜界面或 SV 入射支持；没有接收函数解析梯度。

## 波形处理契约

`receiver_function(model, options)` 计算实波形，`N` 为偶数且在 [8,4096]，正频率为
`f_k=k/(N*dt), k=0..N/2`。默认 N=1024、dt=.1 s、p=.06 s/km。
`source` 是 N 个实数时间样点，空数组表示单位脉冲，不能是全零；变换符号为正指数。
先将两分量乘以相同的震源频谱 S，再做：

```
P = |Z*S|²
denominator = max(P, water_level * max(P))
G(omega) = exp(-omega² / (4*gaussian_width²))
H_R = (R*S) * conj(Z*S) / denominator * G
H_Z = P / denominator * G
A = IFFT_convention_minus(H_Z)[0]
RF(t_j) = IFFT_convention_minus(H_R/A)[j*dt-pre_arrival-time_shift]
t_j = j*dt-pre_arrival
```

Gaussian 宽度 a 的单位 rad/s，默认 2.5；水位为最大垂直功率的比例，默认 .01，范围 (0,1]。
归一化 A 是滤波后垂直自反褶积的零时刻值，不是将 R 自身归一化为 1。
源形状可通过水位作用改变波形，公共源的振幅与极性缩放会抵消。
正 time_shift 将波形移向更晚的输出时间；pre_arrival 默认 5 s。
输出波形为周期 DFT 上的实数重建，Nyquist 项取实部；必须采用相同约定处理实测数据。
直接 DFT 为 O(N²)，当前不引入 FFT 库或隐藏衰减，不自动增加窗口或调整滤波。

## 可靠性状态

频率行始终保留原输入顺序（包括零频），每行返回复 R/Z、rcond、相对后向误差与消息。
`valid` 或 `numerical_failure`，失败位移为 NaN。错误输入抛出异常，与数值失败区分。
任一频率失败，则整条接收函数保留 N 行 NaN，禁止删频后悄悄重建“成功”波形。

临界慢度 `|1/v-p| <= 32*epsilon*max(1/v,p)` 显式报告退化基；q² 使用
`(1/v-p)*(1/v+p)`，避免两个接近平方数相减。矩阵不可逆、rcond<1e-12、
相对后向误差>1e-10 或非有限结果均失败，不人为加对角项。
这些是计算保护门限，后向误差小或 rcond 通过不等于真实波形误差已经受严格控制。
近临界与极厚高对比结构需要专门的独立检查，当前不能据此认证所有 128 层模型。

## 独立参考与外部代码审查

检查了 [Telewavesim 源码](https://github.com/paudetseis/Telewavesim) 的 0.2.1 归档。
包含 MIT 声明，Fortran 文件还保留 C. J. Thomson 的原始作者声明；运行库不引入其代码。
验证驱动自由表面装配参考 plane_land，驱动中保留完整 MIT 声明；源文件哈希、归档 SHA256、
测试输入、原始输出和审查修补文件在 `tests/fixtures/receiver/` 中保存。
使用实 omega，明确去掉 plane_land 默认的 `(1+.001i)*omega` 人工衰减。

实际发现两项不能忽略的参考限制：

- 只有半空间时，rmatrix 没有界面递推，输出零透射。半空间由解析解验收，
  外部对照使用明确给定 1 km 同介质覆盖层，保留入射基准深度所产生的相位。
- addit 计算 `reverbi=(I-rds*ru)^-1` 后，四个递推乘积使用了 `reverb` 而非 `reverbi`。
  三层低速夹层即使 f=0 也与底部半空间极限不符。全频段单个实/虚分量最大差为 .486807。
  仅临时修正这四处后，差降到 7.37e-14。原输出、明确标记的修正输出与精确补丁一起保存，
  未将修正后的代码冒称为未经修改的成熟参考；未向上游发消息或修改外部仓库。

实际冻结的验收参考采用另一套状态方程：
`y=(ux,uz,txz/(i*omega),tzz/(i*omega))`，从应力应变与动量平衡直接推导
`dy/dz=i*omega*A*y`，用 SciPy expm 传播物理状态，再施加地表和半空间条件。
有限层不构造波振幅或界面递推，故与生产边界系统具有不同数值路径。
该增长型传播只用于中等光学厚度，不作为生产算法。

验证四组冻结频谱：同介质覆盖层、20 km 层加半空间、三层低速夹层、薄倏逝高速层。
完整复 R 与 Z 的最大差 **1.21e-13**，预定绝对阈值 **2e-11** 未放宽，未拟合比例或相位。
同介质和两层例同时与未经修改的 Telewavesim 一致；三层例同时与明确修正的递推一致。
倏逝例由状态方程验证，不能用 Telewavesim 的实数 q 开方实现认证。

重新生成（路径由运行者提供；正常构建/测试只读取已保存参考，无需这些依赖）：

```bash
python benchmarks/generate_receiver_reference.py \
  --source /path/to/telewavesim-0.2.1 \
  --lapack-library /path/to/libopenblas.so \
  --output /tmp/new-receiver-reference
```

## 验收覆盖与实际误差

- 半空间 p=0/.03/.06/.12，f=0/.01/1/10 Hz，解析 R/Z：
  `2*p*qS/(1/Vs²-2*p²)`；Gaussian RF 解析为该比值乘 `exp(-a²*t²)`。
  比值阈值 1e-12，波形阈值 1e-11；正时移方向单独验证。
- 同介质拆层的完整复位移保持不变；三层模型 f=0 等于底部半空间响应，阈值 1e-12。
- 三条 CLI 波形由冻结独立频谱加 NumPy FFT 重新构造，覆盖不同水位、滤波、正负时移、
  非脉冲源。实际最大差约 1.6e-15，阈值 atol=2e-11、rtol=2e-9。
- 20 km 层的 Ps、PpPs、PsPs/PpSs 理论相对到时为 2.6478、8.52593、11.1737 s，
  波形峰/谷为 2.6、8.5、11.2 s，极性分别正、正、负；到时阈值 .15 s。
- dt 从 .1 减到 .05 s、保持窗口 51.2 s，公共样点最大差 2.17e-19。
  51.2→102.4 s 的窗口变化仍有 1.57e-4 差异，未通过原定 1e-5 窗口要求；
  102.4→204.8 s 在解释区间 [-5,20] s 的差为 **7.31e-8**，通过原阈值。
  因而示例和默认改为 102.4 s，没有放宽门限；另查 204.8→409.6 s，差约 1.15e-14。
  这些结果只针对该模型/处理参数，不代表所有数据自动收敛。
- 临界值及相邻浮点数、失败频率与波形保行、非法输入、SV 请求拒绝、禁止覆盖已有输出。
  另用极端密度作数值故障注入（非地球模型），触发零频 rcond≈3.72e-15；
  其他有效频率及实际震源频谱保持可用，整条 RF 显式失败，未删除单个失败频率。

## 可复现示例与结果

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
build/dispweave receiver --config examples/receiver/layered.toml --output results/m8/receiver-layered
python scripts/plot_receiver.py results/m8/receiver-layered --time-max 25 --output results/m8/receiver-layered-detail.png
build/dispweave receiver --config examples/receiver/low_velocity.toml --output results/m8/receiver-low-velocity
python scripts/plot_receiver.py results/m8/receiver-low-velocity --time-max 25 --output results/m8/receiver-low-velocity-detail.png
```

两例均返回 0，各 1024 个波形样点、513 个有效频率；保存输入 TOML、波形、完整复频谱及版本/处理元数据。
绘图在频率失败处断开并显示标记，不填零；时间缩放仅影响显示，原始波形始终完整保存。

Release 完整 CTest **51/51** 通过（32.31 s，含并发 Debug 构建的机器负载）。
UBSan 接收函数核心/CLI **2/2** 通过（3.33 s）；补充部分频率失败与震源保留检查后，
Release 定向 **2/2**（.30 s）、UBSan 定向 **2/2**（2.81 s）再次通过。
四张全窗/局部示例图均已实际渲染和检查，参考文件与生成脚本的 SHA256 均已核对。

本机 GCC 13.3、Release、1024 样点/513 频率，同一进程重复 100 次；计时包含物理频谱、
水位/滤波与直接 DFT，使用默认脉冲源。没有扣除首调用成本，也没有新增运行时初始化依赖。

| 案例 | 首调用 ms | 重复平均 ms | 进程最大 RSS KiB |
|---|---:|---:|---:|
| 20 km 层 + 半空间 | 3.199 | 3.271 | 4416 |
| 三层低速夹层 | 6.348 | 6.741 | 4520 |
| 薄倏逝高速层 | 3.295 | 3.444 | 4388 |

```bash
/usr/bin/time -v build/receiver_benchmark layered
/usr/bin/time -v build/receiver_benchmark low_velocity
/usr/bin/time -v build/receiver_benchmark evanescent
```

这是当前机器上的 baseline，不是任意层数/采样规模的性能保证。下一项 M8c 复用这些
正演接口，加入独立观测块、协方差和噪声尺度的组合目标；在可靠导数验收前不接入 HMC 梯度。
