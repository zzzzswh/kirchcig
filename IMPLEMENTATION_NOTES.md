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
- **半阶导数**（`halfderiv=True`，`_halfderiv.py`）：`F = H·A`，`Fᵀ = Aᵀ·H*`，`H(ω) = sqrt(1 − ρe^{−iω})`，
  `ρ = 1 − 1/nt`，与 Madagascar `sf_halfint(inv=true)` 相同。数据侧、与引擎无关：伴随在内核前对数据滤波，
  正演在内核后对输出滤波，用引擎所在设备的 `rfft/irfft`（float64），道零填充到 ≥ 2·nt 的 5-smooth 长度，
  避免尾部（~k^{-3/2}）绕回。pad–卷积–截断的转置是 pad–共轭卷积–截断，所以算子对仍是精确转置。
  验证：脉冲响应等于二项式级数 `binom(1/2,k)(−ρ)^k`；`H·H` = 一阶差分；水平反射层反偏移（零炮检距）
  用 Hilbert 瞬时相位测得 −44.5°、谱斜率 ω^−0.63 → 开滤波后 −9.7°、ω^−0.15。残余 −10° 是离散滤波器的
  四分之一样点延迟（相位 `π/4 − ω/4`，后向差分半样点延迟的一半），伴随反向同量，往返无位移。
  **注意实验教训**：用点散射体测相位是错的——所有道的曲线都精确过那个点，求和是相干的，没有驻相，
  单位权 `Aᵀd` 本身就是零相位；45° 只出现在对反射层的横向积分里。另一个教训：深度网格粗于时间采样
  （`2·dz/v > dt`）时正演出来的道是一排插值脉冲，谱分析全是假象，见第 10 节。
- **幅度权**（`weight=`）：每项贡献乘 `float32(w_s·w_r)`，`w` 作为第四个字段打包进 `tab_t`
  （字段顺序固定 `t, a, d, w`，按在用的字段数补到 1/2/4 个 float，所以 16 字节以内装得下全部组合，
  `KC_NFIELDS` 宏 + 主机 `_upload` 同序打包）。可分离乘积的设计是刻意的：不多一次访存，两侧各一张表就能
  表达倾斜、扩散以及用户任意的 `w_s·w_r`。代价是 `sfkirmod` 的算术平均倾斜 `(cos_s+cos_r)/2` 只能
  用几何平均 `sqrt(cos_s·cos_r)` 近似（二阶一致），Bleistein 真幅度权（含 Beylkin 行列式、`∂²τ/∂x²`）
  不可分离、不提供。预设：`obliquity = sqrt(max(cos θ, 0))`，`spreading = 1/sqrt(max(t, dt))`。
  验证：单点模型的加权正演 = 不加权正演 × 每道标量 `w_s(ip)·w_r(ip)`（精确恒等式）；零炮检距单道偏移
  圆上加权/不加权之比 = cos θ；与 aa、halfderiv、aperture 组合 dot-test 通过；仿真下两引擎逐位一致。
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
| 反假频（第 8 节） | `-DAA=1`：数据侧读 float64 双重累加 `D`，正演写 float64 部分和 |
| 表元素打包 | 字段固定顺序 `t, a, d, w, gx, gz`，按在用字段数补到 1/2/4/8 个 float（`KC_WIDTH`），主机 `_upload` 同序；一项贡献每侧一到两次 16 B 对齐读取 |
| 伴随炮点分组（`-DSCH`） | 检波点表 `nr·npts` 元素放不进 L2、每个炮点重读一遍，是伴随最主要的访存流；`SCH` 个炮点的表元素放寄存器，每个检波点元素一组只读一次，表流量 /SCH。尾部用 `kc_min(sc+k, s1-1)` 夹住做有效读取、`hbin` 置 −1 不参与 |

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
- **拉伸项（`aa_stretch`，默认开）**：Madagascar `aastretch` 的思路。成像样点代表一个 `dx×dz` 单元，
  它在道上的时间足迹是 `|∂τ/∂x|·dx + |∂τ/∂z|·dz`（线性函数在矩形上的值域，L1），`τ = T_s + T_r`，
  梯度**先求和再取绝对值**——镜面点处 `∂T_s/∂x` 与 `∂T_r/∂x` 相消，足迹只剩 `2dz/v`；若分别取绝对值
  会在最重要的镜面贡献上过度滤波。与道轴项按均方根合并（两种独立的模糊叠加，二阶矩相加，LCB 式 5 同理），
  `aa_factor` 缩放整体：`n = round(aa_factor·sqrt((dip·Δρ/dt)² + cell²) + 1)`。梯度表 `(n, npts, 2)`
  float32（`table_gradients`，出射角也从它算），内核里是 `float2` 大小的 `grad_t`，每对多读 8 B。
  验证：单炮、dz=5 m、dt=1 ms（每个深度样点隔 5 个时间样点），水平反射层零炮检距道对 dz=1 m 参考的
  相对 L2 误差 0.52 → 0.10，175 Hz 处的谱复制 53.7 → 0.84（参考 0.12）；剩下的 10% 是三角形在 25 Hz
  的固有幅度损失（sinc²）。
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
- **半宽表达式**：float32 里算 `w = aa_factor·sqrtf((dip·aaf)² + cell²)`，每个乘、加都用 `__fmul_rn`/
  `__fadd_rn`（禁止编译器融合成 FMA——融合后舍入不同，落到另一个整数两个引擎就对不上；`sqrtf` 默认
  正确舍入），最后 `n = int(float64(w) + 1.5)` 再裁到 `[1, aa_max]`。numpy 的 `aa_width`/`aa_cell` 按同样
  的运算顺序逐位镜像；g++ 仿真里这两个 intrinsic 就是普通乘加（`-std=c++17` 关掉了 contraction）。
  NVRTC 对这两个 intrinsic 的支持有待真机确认。
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
    结构体和 `__fmul_rn`/`__fadd_rn`）。打包与分组之前，同一 session 内：`aa_stretch=False` adjoint 77.3 /
    forward 67.3 ms，`aa_stretch=True` 128.8 / 72.0 ms（另一天 `aa_stretch=False` 测得 98.5 ms——session
    间差 25%，比较只看同一次测的）。梯度表那 8 B/pair 让伴随慢 1.7 倍，说明伴随受表读取而非算术限制。据此做了
    两件事（第 4 节表）：`gx, gz` 打包进 `tab_t`（offset+aa 正好 16 B，带角度或权重 32 B）；伴随按
    `SCH=4` 个炮点分组复用检波点表元素。仿真下 SCH=1/3/4/8 与 numpy 逐位一致，含不整除 `ns` 的尾部和
    source 分片。
  - **打包+分组之后**（同一 session，全部 `schunk=4`）：基线 adjoint 33.9 / forward 25.6 ms（原 52.9 / 25.5）；
    `aa_stretch=False` 72.7 / 67.3（原 77.3 / 67.3）；`aa_stretch=True` 62.0 / 71.2（原 128.8 / 72.0）；
    angle+aa（8 float / 32 B 元素）91.1 / 113.7。`--schunk 1/4/8` 在 `aa_stretch=True` 上是
    118.1 / 62.0 / 59.5 ms，正演恒为 71.1——收益绝大部分来自分组（1→4 是 1.9 倍），打包本身只占
    128.8→118.1 那一段（跨 session），4→8 只剩 4%，默认留在 4。两个没解释清的点：(a) 打包+分组之后
    `aa_stretch=True` 反而比 `aa_stretch=False` 快（62.0 vs 72.7），而它的元素宽一倍、滤波也更宽，
    说明伴随已经不在表带宽上了，瓶颈换成了什么还不知道；(b) `aa_stretch=False` 没扫 `schunk`，
    它自己从 77.3 降下来多少是未知的。待补：`--aa --no_aa_stretch --schunk 1/8` 定位 (a)(b)，
    以及 `--acc float32`、`--aperture 60`——README 里这两处数字还是分组之前的。正演的
    2.8 倍比伴随的 1.8 倍重，来源是 6 次共享内存 double 原子加、`(ns,nr,npad)` float64 写出以及
    主机侧两次反向 `cumsum`；如果以后要压这部分，可以考虑在内核里用 block 内 scan 直接做反向双重
    积分，省掉 float64 中间缓冲。它现在是 `aa=True` 下更慢的那一侧。
- **验证**：单炮脉冲响应，31 道 @100 m 对 301 道 @10 m 参考。横向粗糙度
  1.358（关闭）→ 0.563（antialias=1.0），参考值 0.554；峰值 0.0645 → 0.0528，参考 0.0568。

**参考**：Lumley, Claerbout & Bevc, *Anti-aliased Kirchhoff 3-D migration*, SEG 1994 /
SEP-80；Claerbout, *Antialiasing with triangles*, SEP-73 / BEI ch. `trimo`；
Gray, *Frequency-selective design of the Kirchhoff migration operator*, Geophys. Prosp. 40, 1992；
Abma, Sun & Bernitsas, *Antialiasing methods in Kirchhoff migration*, Geophysics 64, 1999；
Madagascar `user/yliu/Mmig2.c` (`sfmig2`)。

## 9. 已知限制（与 README 一致）

仅 2D；offset 域按绝对半炮检距分箱不区分正负；变速度走时依赖 scikit-fmm 的一阶到达；幅度权只到可分离的
倾斜/扩散因子，没有真幅度权（见第 10 节）。

## 10. 路线图：对照 Madagascar 还缺什么

对照 `sfmig2`（user/yliu）、`sfkirmig`（user/llisiw，走时表驱动的叠前深度偏移，与本项目最接近）、
`sfkirmod`、`sfkirchnew`、`sfpreconstkirch`、`sftkirmig`，按重要性排：

1. ~~**半阶导数（rho 滤波，`sf_halfint`）**~~ 已完成（0.2.0，第 3 节）：`halfderiv=True`。
2. ~~**孔径控制**~~ 已完成（0.2.0，第 3 节）：`aperture=`（`sfkirmig`）与 `apt=`（`sfmig2`），
   主机端表掩码实现。`sfmig2` 的 `angle=`（倾角孔径，`|x| > tan(angle)·v·t` 跳过）是时间偏移的
   等价物，深度偏移里锥角已经覆盖。边缘余弦 taper 留到幅度权一起做。
3. ~~**幅度权**~~ 可分离部分已完成（0.2.0，第 3 节 `weight=`）。剩下的是不可分离的真幅度权
   （Bleistein/Schleicher，含 `∂²τ/∂x²` 的 Beylkin 行列式），需要每 (道, 成像点) 现场算或另存表，
   以及 `sfkirchnew` 风格的 pseudo-unitary 权 `cos θ/√t`（总走时的函数，也不可分离；近似可用
   `["obliquity","spreading"]`，零炮检距时差一个 `√t` 因子）。
4. **走时表插值**。`sfkirmig` 只在 `ny` 个稀疏地表位置存表，炮检点之间 Hermite 插值。本项目每炮每检
   一张完整表，`(ns+nr)·nx·nz·4` 字节，是上实际数据的真正瓶颈。改动较大。
5. ~~**正演的拉伸抗假频**~~ 已完成（0.2.0，第 8 节 `aa_stretch`）。
6. **小项**：数据时间原点 `t0`（`sfkirmig` 的 `tau`）；fold 归一化（`sfmig2` 的 `normalize`，
   `op.adjoint(ones)` 就是照明度，可加 helper）；带符号偏移距。

`sfkirmig` 的反假频宽度用 `max(dip_s·ds, dip_r·dh)`，`sfmig2` 用两侧之和；本项目取后者（第 8 节）。
