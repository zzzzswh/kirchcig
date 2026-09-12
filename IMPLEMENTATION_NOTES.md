# kirchcig 实现说明

本文档记录 README 中设计目标到代码的落地方式、性能决策以及验证过程。README 本身是面向用户的接口契约，本实现严格按其 API 落地。

## 1. 项目场景与设计意义

- **场景**：二维叠前 Kirchhoff 偏移，输出保留炮检距（或张开角）轴的共成像点道集（CIG），而不是直接叠加成像。CIG 沿 offset/angle 轴的剩余曲率是偏移速度分析的核心观测量，也是 AVA/AVO 与角度域正则化的输入。
- **意义**：提供**互为精确共轭**的一对线性算子 `forward`（反偏移）/`adjoint`（偏移），使其能直接进入最小二乘偏移（LSQR/CG）与 PyTorch 训练循环（deep-prior LSM、plug-and-play 正则化）。共轭精度到 float 精度，dot-test 是 CI 的核心检验。
- **边界**：不做 SEG-Y I/O、不做算子代数、不做波动方程偏移；2D；暂无反假频滤波（v0.2 计划）。

## 2. 目录结构

```
kirchcig/
  __init__.py        公开 API：migrate, KirchhoffCIG, cuda_available, 走时工具
  _operator.py       KirchhoffCIG 算子类（几何、走时表、分箱、引擎调度、demo、dot_test）
  _kernels.py        CUDA 内核源码（字符串，NVRTC 运行时编译，无需编译器）
  _engine_cuda.py    CUDA 引擎：编译缓存、block 选择、共享内存 opt-in、source 分片、时间分块
  _engine_numpy.py   NumPy 参考引擎（测试 oracle，逐位镜像内核的 float32 运算）
  _traveltime.py     解析走时（常速度）/ eikonal 走时（scikit-fmm）/ 出射角
  torch.py           PyTorch autograd 封装（DLPack 零拷贝，支持二阶导）
tests/               pytest：numpy 引擎、走时、cuda（无 GPU 自动跳过）、torch（无 torch 自动跳过）
examples/            LSQR 最小二乘偏移、deep-prior LSM
benchmarks/bench.py  README 规模问题的 GPU 计时
```

## 3. 算子定义

对每个炮点 s、检波点 r、成像点 (ix, iz)：

```
t = T_s[s, ix, iz] + T_r[r, ix, iz]         # 双程走时
it = floor(t / dt),  w = t/dt - it           # 线性插值索引与权重
h  = bin(s, r) 或 bin(θ_s, θ_r)              # offset 域 / angle 域分箱
adjoint:  cig[h, ix, iz] += (1-w)·d[s,r,it] + w·d[s,r,it+1]
forward:  d[s,r,it] += (1-w)·cig[h,ix,iz];  d[s,r,it+1] += w·cig[h,ix,iz]
```

- **offset 域**：`h = floor(|x_s - x_r| / 2 / (hmax/nh))`，最后一个箱闭区间；超出 `hmax` 的道不参与（`hbin = -1`）。`hmax=None` 时取最大半炮检距，即所有道都参与；`nh=1` 就是叠加成像，没有独立代码路径。
- **angle 域**：出射角 `θ = atan2(∂T/∂x, ∂T/∂z)`（从垂直方向量起），常速度用解析角，否则用走时表二阶差分；半张开角 `γ = |θ_s - θ_r|/2`（差值回绕到 (-π, π]），`hmax` 为最大半张开角（度）。角度与走时打包成 8 字节 `struct {float t; float a;}`，一次 coalesced 读取。
- **精确共轭**：两个内核用**完全相同的 float32 表达式**计算 `t`、`it`、`w`，权重转成累加精度后相乘，因此 forward/adjoint 是同一稀疏矩阵的转置；差别只来自 float64 求和顺序。

## 4. GPU 内核设计（对应 README "Design notes"）

| README 要求 | 实现 |
|---|---|
| 一个线程一个成像点，线程独占整条道集轴，无原子操作 | `kirch_adjoint`：`ip = blockIdx.x*BLOCK + tid`，累加器 `acc[NH][BLOCK]` 在动态共享内存 |
| `[nh][block]` 布局对任意 bin 索引无 bank conflict | 线程 tid 访问的字地址 `h*BLOCK + tid`，bank 与 h 无关 |
| 相邻成像点走时相近，数据读取近似 coalesced | 成像点 z 为内轴，相邻线程读相邻时间样点；走时表 `(n, npts)` 布局保证表读取 coalesced |
| forward 一个 block 一条道，共享内存原子累加，写出一次 | `kirch_forward`：`blockIdx.x = s*nr + r`，`atomicAdd` 到共享内存 `trace[]`，最后一次性写回 |
| 累加 float64，表和数组 float32 | `-DACC=double`（可选 `float`），最终转 float32 输出 |
| 模型布局 `(nh, nx, nz)`，道集轴最外 | adjoint 写回 `out[h*npts + ip]`、forward 读 `model[h*npts + ip]` 均 coalesced |
| `nh`、block、累加类型编译期特化 | `-DNH -DBLOCK -DFBLOCK -DACC -DOUT -DANGLE`，`RawKernel` 进程内缓存 + CuPy 磁盘缓存 |

在 README 之上增加的性能/健壮性措施：

1. **自动 block 选择**：从 {256,128,64,32} 中选最大且使 `NH·BLOCK·sizeof(ACC) ≤ 32 KB` 的值（多个 block 常驻同一 SM）；nh 很大时退到 32 线程并通过 `max_dynamic_shared_size_bytes` opt-in 到设备上限（Volta+ 可到 96–227 KB）。超限时给出明确报错，建议 `acc="float32"` 或减少 bin。
2. **source 分片（adjoint）**：当 `npts/BLOCK` 的 block 数不足以填满 GPU 时，按 `grid.y` 把炮点循环切成若干片，各片写 ACC 精度的部分和，设备上用 float64 归约。内存预算默认 256 MB，可调。
3. **时间分块（forward）**：`nt·sizeof(ACC)` 超过共享内存上限时按 `grid.y` 分窗；线性插值的两个 tap 可能落在不同窗口，每个窗口只加自己拥有的 tap，结果与不分块**逐位相同**。
4. **非有限走时**：NaN/Inf 或超出 10·nt·dt 的走时被推到记录末端之外，永不参与也不会造成 `int` 溢出。
5. **数据流**：numpy 进 numpy 出；cupy 进 cupy 出，无主机往返。torch 侧对 CUDA tensor 用 DLPack 零拷贝，并在非默认 stream 上用 `ExternalStream` 对齐流序。
6. **sm < 60 兼容**：提供 `atomicAdd(double*)` 的 CAS 回退。

## 5. 走时

- 常速度（标量或全同 2D 数组）：解析直射线走时和角度。
- 变速度：`skfmm.travel_time`，零值等值面取以震源为圆心、半径 2 个网格的圆，圆内解析填充（`dist / v_src`），圆外加 `r0 / v_src`。避免点源奇异带来的近场误差，也支持任意非网格点震源位置。常速度下与解析解相对误差最大 ~1%，均值 ~0.2%。
- 自带走时表：`trav=(trav_srcs, trav_recs)`，形状 `(ns, nx, nz)`、`(nr, nx, nz)`，与 README 一致。

## 6. 验证

- `tests/`：dot-test（三种配置）、点散射体聚焦位置、spike 反偏移落在正确走时、`migrate()` 与算子一致、自带走时表、非有限走时、scipy `LinearOperator`、eikonal 对比解析、梯度速度模型下的角度域 dot-test、torch 三项梯度测试；cuda 测试（dot-test、cuda vs numpy 逐点对比、分块/分片精确性、cupy I/O、float32 累加）在无 GPU 时自动跳过。
- **无 GPU 环境下对 CUDA 代码的验证**：开发时把 `_kernels.py` 中的源码原样用 g++ 编译（`std::thread` 模拟 block 内线程、barrier 模拟 `__syncthreads`、CAS 模拟原子操作），并用一个 numpy 支撑的假 `cupy` 模块把 `_engine_cuda.py` 的 launch 配置接到仿真内核上。结果：float64 累加下 offset/angle 域、时间分块、source 分片、>48 KB 共享内存 opt-in 各路径与 numpy 参考引擎**逐位一致**（rel.err = 0），float32 累加误差 ~2e-7。这套仿真脚本不随包发布，但保证了交付的内核逻辑正确；真实 GPU 上还需跑一次 `pytest`（含 cuda 测试）和 `benchmarks/bench.py` 确认 NVRTC 编译与性能。
- README "Verifying an installation" 段落中 `demo_image()` 的语义：用 numpy 参考引擎偏移 `demo_data()` 得到的叠加像，因此该断言在 cuda 引擎下是 **cuda vs oracle** 的一致性检验。

## 7. 性能预期与调参

- 内核为访存受限：每个 (s, r, 成像点) 组合约读 4 B 走时 + ~8 B 数据（或模型），README 规模（ns=100, nr=200, 401×201, nt=1500, nh=32）约 20 GB 有效访存，A100 级别 GPU 上 adjoint/forward 各约 10–20 ms 量级。
- 可调项（`KirchhoffCIG` 构造参数或 `op._eng`）：`acc`（`float64`/`float32`）、`block`、`split`（`"auto"` 或整数）、`op._eng.set_time_chunk()`、`fblock`。
- 反复调用（反演循环）时传入 cupy/CUDA tensor，避免每次主机-设备拷贝。

## 8. 已知限制（与 README 一致）

无反假频滤波；仅 2D；offset 域按绝对半炮检距分箱不区分正负；变速度走时依赖 scikit-fmm 的一阶到达。
