# kirchcig

**GPU 克希霍夫偏移，输出共成像点道集（CIG），并提供精确伴随算子。**

[English](README.md) | 简体中文

把叠前地震数据在 GPU 上偏移成偏移距域或角度域共成像点道集，同时给出配对的反偏移算子，所以这一对算子可以直接用于最小二乘偏移或 PyTorch 训练循环。

![Stacked migrated image and the common-image gather at the scatterer](docs/img/cig.png)

```bash
pip install "kirchcig[cuda12]"     # CUDA 12.x
pip install "kirchcig[cuda11]"     # CUDA 11.x
pip install kirchcig               # 仅 CPU，NumPy 参考引擎
```

不需要编译器。CUDA 核函数由 CuPy 通过 NVRTC 在运行时编译。

## 快速开始

```python
import numpy as np
from kirchcig import migrate

# data  (ns, nr, nt)   叠前道集
# vel   (nx, nz)       平滑后的偏移速度 [m/s]
# srcs  (2, ns)        炮点坐标，两行分别是 (x, z) [m]
# recs  (2, nr)        检波点坐标，两行分别是 (x, z) [m]

cig = migrate(
    data, vel, srcs, recs,
    dt=0.004, dx=10.0, dz=10.0,
    nh=32, hmax=2000.0,        # 32 个半偏移距道集，最大 2000 m
)
# cig -> (32, nx, nz)

image = cig.sum(0)             # (nx, nz) 叠加后的偏移剖面
```

常规用法就这些。叠加剖面不是另一条代码路径,`nh=1` 直接给出。

手头没数据？下面这段开箱即跑：

```python
from kirchcig import KirchhoffCIG
op = KirchhoffCIG.demo()       # 常速度模型 + 单点散射体
assert op.dot_test()
cig = op.adjoint(op.demo_data())
```

## 为什么要道集，而不是直接给一张叠加剖面

CIG 保留了偏移距（或张角）轴，而不是把它叠掉。沿这条轴的剩余曲率是**偏移速度分析**最主要的可观测量：同相轴拉平说明速度正确，上翘或下拉分别说明速度偏低或偏高。叠加会把这个信息彻底抹掉,这正是生产上的克希霍夫偏移必须输出道集的原因。

![用三种速度偏移得到的道集：偏低、正确、偏高](docs/img/vel_analysis.png)

<sub>同一份数据用三种速度偏移。同相轴拉平即速度正确，弯曲程度正是偏移速度分析所测量的量。</sub>

CIG 同时也是 AVO/AVA 分析的天然输入，以及最小二乘偏移中角度相关正则化的基础。

## 算子对

做反演要用算子，而不是一次性函数：

```python
from kirchcig import KirchhoffCIG

op = KirchhoffCIG(
    nx=401, nz=201, dx=10.0, dz=10.0,
    srcs=srcs, recs=recs,
    nt=1500, dt=0.004,
    vel=vel,
    nh=32, hmax=2000.0,
    domain="offset",           # 或 "angle"
    engine="cuda",             # 或 "numpy"
)

cig  = op.adjoint(data)        # (ns, nr, nt) -> (nh, nx, nz)   偏移
data = op.forward(cig)         # (nh, nx, nz) -> (ns, nr, nt)   反偏移

op.dot_test()                  # True
```

`forward` 与 `adjoint` 在累加精度意义下互为精确转置，因此可以直接塞进任何最小二乘求解器：

```python
import scipy.sparse.linalg as spla
cig_lsm = spla.lsqr(op.to_scipy(), data.ravel(), iter_lim=20)[0].reshape(op.shape_model)
```

### 角度域道集

```python
op = KirchhoffCIG(..., domain="angle", nh=30, hmax=60.0)   # 30 个道集，0-60 度
```

`domain="angle"` 时 `hmax` 是最大半张角（度）。道集索引由炮点侧和检波点侧的出射角决定，出射角从走时梯度算出。

### PyTorch 自动微分

```python
import torch
from kirchcig.torch import TorchKirchhoffCIG

top = TorchKirchhoffCIG(op)
cig = top.adjoint(data)        # 对 data 可微
res = top.forward(cig) - data
res.pow(2).sum().backward()
```

张量全程留在 GPU 上，不经过主机内存。由于算子是线性的，`forward` 的反向传播就是 `adjoint`，反之亦然，所以二阶导数也能正常工作。这使得深度先验 LSM 和即插即用正则化变成一行代码：用网络参数化 CIG，剩下的交给 autograd。

### 使用自己的走时表

走时默认来自程函方程求解（`scikit-fmm`）或常速度解析式。如果你有自己的正演器，可以直接传表：

```python
op = KirchhoffCIG(..., trav=(trav_srcs, trav_recs))
# trav_srcs (ns, nx, nz)   炮点到成像点的走时 [s]
# trav_recs (nr, nx, nz)   成像点到检波点的走时 [s]
```

注意维度顺序：炮/检波点轴在最前面。这与某些其他库是转置关系，但正是它保证了伴随核函数的读取是合并访问的。

## 定位

| 你想要 | 用 |
|---|---|
| GPU 上的克希霍夫 CIG，外加做 LSM 的伴随算子 | **kirchcig** |
| SEG-Y 读写 | [segyio](https://github.com/equinor/segyio) |
| 线性算子代数、求解器、正则化 | [PyLops](https://github.com/PyLops/pylops),kirchcig 通过 `to_scipy()` 接入 |
| 波动方程正演与 RTM | [Deepwave](https://github.com/ar4/deepwave) |

## 实现要点

- **伴随算子完全不用原子操作。** 一个线程负责一个成像点并独占整条道集轴，所有写入互不冲突。累加器放在共享内存中，布局为 `[nh][block]`，对任意的每线程道集索引都无 bank 冲突。
- **正演每个道一个 block**，在共享内存里累加完整条道后一次性写出。
- **累加器默认 float64。** 一个偏移样点要累加 10^4 到 10^5 项；用 float32 时点积测试只能过到 1e-3 量级，这个精度足以掩盖真实 bug。
- **模型布局 `(nh, nx, nz)`**，道集轴在最外层，同时保证伴随的回写和正演的模型读取都是合并访问。
- **编译期特化。** `nh`、block 大小、累加器类型通过 `-D` 注入，因此 `nh` 是真正的编译期常量。改动它会触发约一秒的 NVRTC 重编译，之后由 CuPy 缓存。

## 已知限制

- **尚无抗假频滤波。** 克希霍夫算子假频会在大偏移距处表现为陡倾角伪影。计划在 v0.2 加入（三角滤波器组，Lumley-Claerbout）。
- **仅支持 2D。** 瓶颈在走时表，不在核函数。
- **偏移距分道集用的是半偏移距绝对值**，因此不区分正负偏移距。

`numpy` 引擎在任何地方都能跑，包括纯 CPU 机器和笔记本沙箱。它很慢，存在的意义是作为参考实现和测试标准答案，但本文档中的每个例子它都能跑。

## 依赖

Python >= 3.10，`numpy`。可选：与你的 CUDA 版本匹配的 `cupy`（GPU 引擎）、`scikit-fmm`（程函走时）、`torch`（自动微分封装）、`scipy`（`to_scipy()`）。

## 参与贡献

欢迎提 issue 和 PR。`pytest -q` 会在两个引擎下运行点积测试；没有可见 GPU 时 CUDA 测试自动跳过。

## 引用

如果本项目对你的发表工作有帮助，请引用 <TODO: Zenodo DOI>。

## 许可证

MIT

---

<sub>关键词：克希霍夫偏移, 叠前深度偏移, 共成像点道集, CIG, 角道集, 偏移距道集, GPU 地震成像, CUDA, CuPy, 最小二乘偏移, LSM, 精确伴随, 反偏移, 偏移速度分析, AVO, AVA, PyTorch, 地震反演。</sub>
