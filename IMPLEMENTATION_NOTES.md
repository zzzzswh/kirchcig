# kirchcig 实现说明

本文档记录 README 中设计目标到代码的落地方式、性能决策以及验证过程。README 本身是面向用户的接口契约，本实现严格按其 API 落地。

## 1. 项目场景与设计意义

- **场景**：二维叠前 Kirchhoff 偏移，输出保留炮检距（或张开角）轴的共成像点道集（CIG），而不是直接叠加成像。CIG 沿 offset/angle 轴的剩余曲率是偏移速度分析的核心观测量，也是 AVA/AVO 与角度域正则化的输入。
- **意义**：提供**互为精确共轭**的一对线性算子 `forward`（反偏移）/`adjoint`（偏移），使其能直接进入最小二乘偏移（LSQR/CG）与 PyTorch 训练循环（deep-prior LSM、plug-and-play 正则化）。共轭精度到 float 精度，dot-test 是 CI 的核心检验。
- **边界**：不做 SEG-Y I/O、不做算子代数、不做波动方程偏移；2D。

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
- **孔径**：`aperture`（锥角半角，度）与 `apt`（横向距离，米）在主机端算成 `(ns, nx, nz)` /
  `(nr, nx, nz)` 布尔掩码（`aperture_mask`），被掩掉的走时表项置为 `t_big = 10·nt·dt`，走的是
  内核里已有的 `it >= nt-1` 跳过路径。不改内核，两个引擎丢的贡献逐位相同，倾角表和角度表都从
  未掩的干净表算（否则掩码边缘会出现巨大的假倾角）。`trav_srcs`/`trav_recs` 属性保持未掩。
  锥角比较写成 `|dx| ≤ (tan(ap) + 1e-9)·dz`，1e-9 的余量是因为 `tan(45°)` 在浮点里是 `1 − 1e-16`，
  不加它正好落在锥面上的点会被排除。加速来自跳过数据读取和累加；表读取和 warp 分歧仍在，
  实际提速小于被丢弃的 pair 比例（README 几何：60° 保留 47%，45° 保留 23%）。
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
| `nh`、block、累加类型编译期特化 | `-DNH -DBLOCK -DFBLOCK -DACC -DOUT -DANGLE -DAA`，`RawKernel` 进程内缓存 + CuPy 磁盘缓存 |
| 反假频（第 8 节） | `-DAA=1`：表元素扩成 `{t,d}` / `{t,a,d,_}`，数据侧读 float64 双重累加 `D`，正演写 float64 部分和 |

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

## 8. 反假频（v0.2，两个引擎都已落地）

算子假频的判据来自 Lumley, Claerbout & Bevc (1994)：沿算子轨迹求和的样点必须满足
`f_max ≤ 1/(2ΔT)`，`ΔT = (dt_k/dρ)·Δρ` 是相邻道之间的算子时差。

- **倾角 dt_k/dρ**：源侧与检波侧导数**之和**（LCB 式 4；Madagascar `sfmig2` 的
  `tx = |x-h|/(v²(t1+dt)) + |x+h|/(v²(t2+dt))`），不是取大者。本实现按 LCB 建议的
  「走时表差分」路线，对 `(n, nx, nz)` 表沿道轴做中心差分得到 `|dT/dx|` [s/m]，
  比时间偏移的双曲近似准确，且自带走时表也能用。要求道轴按测线排序，否则告警。
- **有效道距 Δρ**：两轴间距的均方根（LCB 式 5–7；2D 下 `dx_★` 退化为各自的道距）。
  炮检间距相等时 `Δρ = dx`；只有一个炮点时该轴不参与均值。
- **三角滤波**：`(1/n²)(D[i] − 2D[i−n] + D[i−2n])` 等于半宽 n 的归一化三角滤波，
  `D` 为道的双重累加。3 次读取与 n 无关（LCB 的核心技巧）。半宽
  `n = round((dip_s + dip_r)·aa_factor·Δρ/dt + 1)`，末尾 `+1` 对应 `trimo`/`sfmig2`
  的 `+dt`，`n = 1` 即恒等。
- **`aa_factor`**：等价于 Claerbout `trimo` 与 Madagascar `sfmig2` 的 `antialias`，
  三者默认都是 1.0。2.0 把三角滤波的第一个零点放在假频上（LCB 式 12），是 Claerbout
  偏移时用的值，代价是陡倾处分辨率。
- **归一化**：本实现用精确的 `1/n²`（直流增益为 1），`sfmig2` 用
  `(dt/(dt+tp−tm))²`，后者在 gap=1 时增益是 1/9 而非 1。选前者是为了保住
  「n=1 逐位退化成不滤波」这个可测性质。
- **共轭**：双重累加 `S` 的转置是反向双重累加，正演散射 6 个抽头后施加它，因此
  算子对仍是精确转置，dot-test 在 offset/angle/nh=1 三种配置下均通过。
  （`sfmig2` 用先因果后反因果的 `doubint`，那个算子自共轭，正反演可共用一个例程；
  本实现尚未采用，是后续可以简化的地方。）
- **精度**：float64 的 `D` 做二阶差分存在相消。实测 `n=1` 重建相对误差在白噪声下
  约 4e-9、带限子波下 ~1e-20，且不随 `nt` 增长（测到 4001），远低于 float32 的 1.2e-7。
- **半宽表达式**：`n = int(float64(float32(dip·aaf)) + 1.5)`，再裁到 `[1, aa_max]`。乘法在
  float32 里先舍入，加 1.5 在 float64 里做（精确）再截断。中间那次 float→double 转换是刻意的：
  写成 float32 的 `dip*aaf + 1.5f` 会被 GPU 编译器融合成 FMA，舍入不同就可能落到另一个整数，
  两个引擎的滤波宽度就对不上。numpy 的 `aa_width` 与内核的 `kc_aa_width` 逐位相同。
- **CUDA 实现**（`_engine_cuda.py` / `_kernels.py`）：
  - 倾角表打包进 `tab_t`：offset 域 `{t, d}`（8 B），angle 域 `{t, a, d, _}`（16 B，补一个 pad
    保证对齐），仍是一次合并读取。`aaf` 是 `(ns, nr)` 的设备数组，伴随里对整个 block 一致。
  - 伴随：主机先把数据零填充到 `npad = nt + 2·aa_max + 1`（时间样点 0 在 `pad = aa_max + 1`），
    做两次 float64 `cumsum` 得到 `D`，内核以 `npad` 为行长、`pad` 为偏移读 3 对抽头。
    不开反假频时主机传 `npad = nt`、`pad = 0`，内核代码路径与 0.1 完全相同。
  - 正演：内核把 6 个抽头散射到共享内存道（窗口在 padded 轴上切分，每个抽头单独判界，因此
    时间分块仍然精确），写出 `(ns, nr, npad)` float64；主机反向两次 `cumsum` 后切掉 padding。
  - **正演道累加器在 `AA=1` 下一律 double**，不看 `ACC`：它随后要被积分两次，float32 的舍入噪声
    积分后变成 ~1e-4 的平滑误差（仿真实测 2e-4，改成 double 后 1e-10），而伴随侧不受影响。
    因此 `acc="float32"` 开反假频后 dot-test 误差仍是 ~1.7e-7，与不开时一致。共享内存预算按
    `facc_bytes = 8` 计算。
  - 显存：`D` 与正演输出各需 `ns·nr·npad·8` 字节。README 规模约 250 MB。
  - 仿真验证：伴随与 numpy 引擎逐位一致（差别只在 denormal 量级），正演相对误差 ~6e-9（双重累加
    的求和顺序不同再被反向积分放大），远低于 float32；三种配置 dot-test 通过；时间分块（`tchunk=50`
    远小于 `2·aa_max`）与 source 分片与不分块结果 <1e-6。真实 GPU 上 `cp.cumsum` 是并行 scan，
    与串行求和差在 1e-16·D 量级，对结果的影响同样在 1e-9 以下。
  - **V100 实测**（README 规模，nh=32，float64）：全部 cuda 测试通过（NVRTC 接受 `alignas(16)`
    结构体）；adjoint 98.5 ms（不开 52.9），forward 67.4 ms（不开 25.5），dot-test 2.0e-8。正演的
    2.6 倍比伴随的 1.9 倍重，来源是 6 次共享内存 double 原子加、`(ns,nr,npad)` float64 写出以及
    主机侧两次反向 `cumsum`；如果以后要压这部分，可以考虑在内核里用 block 内 scan 直接做反向双重
    积分，省掉 float64 中间缓冲。
- **验证**：单炮脉冲响应，31 道 @100 m 对 301 道 @10 m 参考。横向粗糙度
  1.358（关闭）→ 0.563（antialias=1.0），参考值 0.554；峰值 0.0645 → 0.0528，参考 0.0568。

**参考**：Lumley, Claerbout & Bevc, *Anti-aliased Kirchhoff 3-D migration*, SEG 1994 /
SEP-80；Claerbout, *Antialiasing with triangles*, SEP-73 / BEI ch. `trimo`；
Gray, *Frequency-selective design of the Kirchhoff migration operator*, Geophys. Prosp. 40, 1992；
Abma, Sun & Bernitsas, *Antialiasing methods in Kirchhoff migration*, Geophysics 64, 1999；
Madagascar `user/yliu/Mmig2.c` (`sfmig2`)。

## 9. 已知限制（与 README 一致）

仅 2D；offset 域按绝对半炮检距分箱不区分正负；变速度走时依赖 scikit-fmm 的一阶到达；无半阶导数滤波和
幅度权（见第 10 节）。

## 10. 路线图：对照 Madagascar 还缺什么

对照 `sfmig2`（user/yliu）、`sfkirmig`（user/llisiw，走时表驱动的叠前深度偏移，与本项目最接近）、
`sfkirmod`、`sfkirchnew`、`sfpreconstkirch`、`sftkirmig`，按重要性排：

1. **半阶导数（rho 滤波，`sf_halfint`）**。Madagascar 每个 2D Kirchhoff 程序都有：`sfmig2` 对像做
   `halfint`，`sfkirmod` 的 Ricker 带 `order=2` 半阶导，`sfkirchnew` 有 `hd` 开关。这是 2D Kirchhoff
   的正确相位/幅度校正，缺了它偏移子波有 45° 相位旋转和低频偏重。实现：对 `(ns, nr, nt)` 数据沿时间
   FFT 滤波 `sqrt(1 − ρ e^{−iω})`（`ρ = 1 − 1/nt`），伴随在内核前施加、正演在内核后施加共轭滤波，
   算子对保持精确转置。开销远小于内核。
2. ~~**孔径控制**~~ 已完成（0.2.0，第 3 节）：`aperture=`（`sfkirmig`）与 `apt=`（`sfmig2`），
   主机端表掩码实现。`sfmig2` 的 `angle=`（倾角孔径，`|x| > tan(angle)·v·t` 跳过）是时间偏移的
   等价物，深度偏移里锥角已经覆盖。边缘余弦 taper 留到幅度权一起做。
3. **幅度权**。`sfkirmod` 用 `obl = 0.5(cos θ_s + cos θ_r)` 与几何扩散，`sfkirchnew` 的 pseudo-unitary
   权 `ps`，`sftkirmig` 的 `amp=`。本项目是纯单位权求和，不是真幅度。实现是每个 (道, 成像点) 乘一个
   两个内核共用的权表达式；出射角表已经有（angle 域），offset 域也可算。
4. **走时表插值**。`sfkirmig` 只在 `ny` 个稀疏地表位置存表，炮检点之间 Hermite 插值。本项目每炮每检
   一张完整表，`(ns+nr)·nx·nz·4` 字节，是上实际数据的真正瓶颈。改动较大。
5. **小项**：数据时间原点 `t0`（`sfkirmig` 的 `tau`）；fold 归一化（`sfmig2` 的 `normalize`，
   `op.adjoint(ones)` 就是照明度，可加 helper）；带符号偏移距。

`sfkirmig` 的反假频宽度用 `max(dip_s·ds, dip_r·dh)`，`sfmig2` 用两侧之和；本项目取后者（第 8 节）。
