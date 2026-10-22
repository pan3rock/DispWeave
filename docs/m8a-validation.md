# M8a：Rayleigh 模态椭圆率正演

状态：实现与下列数值验收已完成；M8b–M8d 尚未完成，M8 整体仍在进行。

## 物理量与接口

`rayleigh_ellipticity(model, queries, method)` 复用 invLM 的模态根和 RayleighKernel，
输出地表 `E=|ur|/|uz|`。无量纲、非负，模态从零开始；不含旋转方向，
不是背景噪声 H/V，也不是多个模态/波型的混合谱比。
该函数的核与根来自同一 invLM 物理模型；根搜索可选 Dunkin 或 fast-scalar，默认 Dunkin。
没有新增运行依赖，没有重写频散求根器，也不声明椭圆率解析梯度。

每行分别保存相速度根状态、找到的根前缀数、根消息、椭圆率状态、能量残差和
特征函数重建标记。所有请求行按原顺序保留，包括重复频率和失败行。
状态为 `valid`、`root_unavailable`、`eigenfunction_failure`、`near_pole`。
根成功不保证特征函数或比值成功；后两类失败仍保留有效相速度。

能量残差采用核已有的 1e-7 门限。另定义归一化不变的极点保护：
`|uz| <= 1e-7 * max(|ur|,|uz|)` 时，输出 near_pole 和 NaN，不截断成巨大有限比值。
这相当于保护比值超过 1e7 的数值区域，不是物理极点位置的证明，也不是误差上界。
零水平位移允许 E=0；全部表面位移为零或非有限时报告特征函数失败。

## 根误差放大与修复

强对比模型：层顶 [0,.1] km，Vs=[.4,2.5]、Vp=[.8,4.5] km/s，rho=[2,2.5]。
基阶垂直位移在约 1.0175858928205 Hz 过零。
初版使用标准频散精度，在 f=1.0174 Hz 得到 E=4021.9808002，独立 P/SV 位移比
为 4021.9731304；相对差 1.907e-6，未达到预定 1e-6 要求，尽管能量残差仅 1.121e-11。
独立重新精化相速度只改变 -4.992e-13 km/s，就消除了主要误差，说明根误差被比值放大。

修复是在已有 invLM/TOMS748 的同一模态括区内使用更细收敛：
`InvLMForward(method, RootAccuracy::refined)` 令 absolute tolerance=0、
relative tolerance=8*double epsilon。原有粗搜索、排序和去重不变。
椭圆率自动使用此精度；普通正演、LM/HMC/RJMCMC 保持原默认 1e-12/1e-14。
TOMS748 收敛错误显式传播，不能把精化失败当作有效根。
修改和来源哈希记录于 `third_party/invLM/PATCHES.md` 与 `provenance.json`。

精化后 f=1.0174、1.0178 Hz 的椭圆率约为 4021.973、3497.84，
相对独立参考误差分别为 4.34e-9、-1.42e-9；预定 1e-6 阈值未放宽。
极点位置仍返回 near_pole，根保持有效。
在 1.8137044387655514 Hz 水平位移过零，得到约 7.22e-15 的有效比值，
不把它当作缺失数据；独立参考、重复查询和无序查询也检查此行为。

## 独立验证覆盖

- 半空间：Vp/Vs=1.2、1.73、3；频率 .01、1、100 Hz。
  使用自由表面 P/SV 解析位移比，误差 <1e-8；不存在的高阶保留为失败行。
- 分层：正常梯度模型的四个频率/模态，以及低速层的 5/10/15 Hz。
  测试参考为独立 P/SV 平面波传播、地表无应力条件的零空间位移比，
  对正常/低速层要求误差 <1e-6*max(1,E)。
- 将相同介质层拆分后，E 变化 <1e-7；根精化保持五个模态的身份与顺序。
- 强对比：极点本身、两侧和水平位移节点，Dunkin/fast-scalar 均检查极点行为。
- CLI：物理标签、行数与顺序、缺根、极点保行，以及禁止输出覆盖输入配置。

独立 P/SV 参考只适用于这些中等光学厚度案例；不据此认证任意极厚/强病态模型。
更精细括区也不解决一般层状介质的根完备性。能量残差不是地表比值误差的严格界，
弱地表振幅、高度聚集模态和极端条件仍需专门的独立验证。

## 使用与实际运行

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
mkdir -p results/m8
build/dispweave ellipticity --config examples/ellipticity/layered.toml --output results/m8/ellipticity-refined.txt
python scripts/plot_ellipticity.py results/m8/ellipticity-refined.txt --output results/m8/ellipticity.png
build/dispweave ellipticity --config examples/ellipticity/pole.toml --output results/m8/pole.txt
python scripts/plot_ellipticity.py results/m8/pole.txt --output results/m8/pole.png
```

第一组示例 10/10 有效；第二组故意含极点，返回 2，全部四行输出保留。
绘图以 NaN 断开无效曲线，在频率轴上标记无效行并报告数量，不静默丢失失败点。
零点保留在线性比值坐标中。输出图片路径必须尚不存在。

完整 Release CTest 49/49 通过（21.09 s）；UBSan 椭圆率核心/CLI 定向 2/2 通过（.23 s）。
之后增加水平位移零点/重复查询检查与根消息输出，按相同定向测试再次验收。
此前所有默认精度正演和采样回归均包含在完整测试中。

补充零点、重复查询与根消息后，Release 定向 2/2（.07 s）、UBSan 定向 2/2（.12 s）
再次通过。两张示例 PNG 已实际渲染并检查，包含显式的极点标记。

普通三层、10 个频率/模态查询的 C++ 基线（GCC 13.3、Release，同一进程重复 1000 次）：
首调用 .167 ms，重复平均 .0629 ms，进程最大 RSS 4444 KiB。这是当前机器的单次基线，
不是跨硬件保证；未测量极端病态模型的性能。可用 `ellipticity_benchmark` 目标复现，
计时明确包含根与特征函数计算，未扣除初始化。
