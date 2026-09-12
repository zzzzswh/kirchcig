# kirchcig

**GPU Kirchhoff migration to common-image gathers (CIGs), with an exact adjoint.**

English | [简体中文](README.zh-CN.md)

Hand-written CUDA kernels, compiled at runtime by NVRTC through CuPy. PyTorch is an optional zero-copy autograd adapter, not part of the compute path.

![Stacked migrated image and the common-image gather at the scatterer](docs/img/cig.png)

## Install

```bash
pip install "kirchcig[cuda12]"     # GPU, CUDA 12.x
pip install "kirchcig[cuda11]"     # GPU, CUDA 11.x
pip install kirchcig               # CPU only, NumPy reference engine
```

No compiler and no nvcc needed; kernels are built at runtime by NVRTC.

| Extra | Pulls in | Gives you |
|---|---|---|
| *(none)* | `numpy` | `engine="numpy"`, the reference engine. Runs anywhere, slow. |
| `cuda12` / `cuda11` | `cupy-cuda12x` / `cupy-cuda11x` | `engine="cuda"`, the GPU engine. Pick the one matching your driver. |
| `eikonal` | `scikit-fmm` | Traveltimes for a non-constant velocity model. |
| `torch` | `torch` | `kirchcig.torch`, the autograd wrapper. |
| `test` | `pytest`, `scipy`, `scikit-fmm` | Test suite, and `op.to_scipy()`. |

Combine them: `pip install "kirchcig[cuda12,eikonal,torch]"`.

## Quick start

```python
import numpy as np
from kirchcig import migrate

# data  (ns, nr, nt)   prestack traces
# vel   (nx, nz)       smooth migration velocity [m/s]
# srcs  (2, ns)        source positions, rows are (x, z) [m]
# recs  (2, nr)        receiver positions, rows are (x, z) [m]

cig = migrate(
    data, vel, srcs, recs,
    dt=0.004, dx=10.0, dz=10.0,
    nh=32, hmax=2000.0,        # 32 half-offset bins out to 2000 m
)
# cig -> (32, nx, nz)

image = cig.sum(0)             # (nx, nz) stacked image; nh=1 gives it directly
```

No data at hand? This runs out of the box:

```python
from kirchcig import KirchhoffCIG
op = KirchhoffCIG.demo()       # constant velocity, one point scatterer
assert op.dot_test()
cig = op.adjoint(op.demo_data())
```

## Usage

### The operator pair

For inversion you want the operator, not the one-shot function:

```python
from kirchcig import KirchhoffCIG

op = KirchhoffCIG(
    nx=401, nz=201, dx=10.0, dz=10.0,
    srcs=srcs, recs=recs,
    nt=1500, dt=0.004,
    vel=vel,
    nh=32, hmax=2000.0,
    domain="offset",           # or "angle"
    engine="cuda",             # or "numpy", "auto"
)

cig  = op.adjoint(data)        # (ns, nr, nt) -> (nh, nx, nz)   migration
data = op.forward(cig)         # (nh, nx, nz) -> (ns, nr, nt)   demigration
op.dot_test()                  # True
```

`forward` and `adjoint` are exact transposes to accumulator precision, so they drop into any least-squares solver:

```python
import scipy.sparse.linalg as spla
cig_lsm = spla.lsqr(op.to_scipy(), data.ravel(), iter_lim=20)[0].reshape(op.shape_model)
```

### Angle-domain gathers

```python
op = KirchhoffCIG(..., domain="angle", nh=30, hmax=60.0)   # 30 bins, 0-60 deg
```

`hmax` is the maximum half opening angle in degrees. Bin indices come from source- and receiver-side emergence angles, computed from the traveltime gradients.

### PyTorch

```python
from kirchcig.torch import TorchKirchhoffCIG

top = TorchKirchhoffCIG(op)
cig = top.adjoint(data)        # differentiable w.r.t. data
res = top.forward(cig) - data
res.pow(2).sum().backward()
```

Tensors stay on the GPU. The operator is linear, so the backward of `forward` is `adjoint` and vice versa; the backward pass is itself recorded, so second-order derivatives work.

### Custom traveltimes

```python
op = KirchhoffCIG(..., trav=(trav_srcs, trav_recs))
# trav_srcs (ns, nx, nz)   source-to-image-point traveltimes [s]
# trav_recs (nr, nx, nz)   image-point-to-receiver traveltimes [s]
```

Otherwise traveltimes come from an eikonal solve (`scikit-fmm`), or analytically for constant velocity. Note the layout: the source/receiver axis comes first, transposed relative to some other libraries, which is what keeps the adjoint reads coalesced.

### Shapes

| | Shape | Notes |
|---|---|---|
| data | `(ns, nr, nt)` | float32 |
| model (CIG) | `(nh, nx, nz)` | gather axis outermost |
| `srcs`, `recs` | `(2, ns)`, `(2, nr)` | rows are `(x, z)` in metres |
| velocity | `(nx, nz)` or scalar | m/s |

## Examples

```bash
python examples/plot_cig.py          # the cover figure
python examples/vel_analysis.py      # the figure below
python examples/lsqr_migration.py    # least-squares migration with SciPy LSQR
python examples/torch_deep_prior.py  # deep-prior LSM
python benchmarks/bench.py           # timings
```

![Gathers migrated with three velocities: too low, correct, too high](docs/img/vel_analysis.png)

<sub>The same data migrated with three velocities. Flat gathers mean the velocity is right; curvature along the offset axis is what migration velocity analysis measures, and stacking destroys it.</sub>

## Performance

Single **Tesla V100-PCIE-32GB** (driver 580.178.04), `nx=401, nz=201, ns=100, nr=200, nt=1500, nh=32`, offset domain. Each operator application evaluates 1.6e9 trace-image-point pairs.

| Accumulator | adjoint (migration) | forward (demigration) | dot-test relative error |
|---|---|---|---|
| `float64` (default) | 52.9 ms — 30.5 G pair-evals/s | 25.5 ms — 63.2 G pair-evals/s | 9.5e-08 |
| `float32` | 33.5 ms — 48.2 G pair-evals/s | 18.8 ms — 85.8 G pair-evals/s | 1.1e-07 |

Building the operator, including traveltime tables and the one-off NVRTC compile, takes about 1.4 s. `forward` is roughly twice as fast as `adjoint`: it accumulates one trace per block in shared memory and writes it out once, while `adjoint` does an irregular gather along the traveltime curves.

float64 accumulation is close to free on Volta and other data-centre cards (1:2 FP64:FP32) and buys bit-identical agreement with the NumPy reference engine. On consumer GeForce parts the ratio is about 1:64, so `acc="float32"` is the sensible default there; it costs roughly 1e-7 of relative accuracy.

Reproduce with `python benchmarks/bench.py`; `--acc float32`, `--nh`, `--domain` and `--engine numpy` are accepted. The operator runs on one device; select it with `cupy.cuda.Device`, or from the PyTorch wrapper by the tensor's device.

## How it works

The kernels live as a CUDA C++ string in `kirchcig/_kernels.py`, compiled by `cupy.RawKernel(..., backend="nvrtc")` on first use and cached by CuPy afterwards. CuPy only allocates memory, compiles and launches; all arithmetic is in the kernels.

- **Why hand-written.** Kirchhoff migration is an irregular gather along traveltime curves, not a matmul or a convolution. No tensor op expresses it without blowing up memory traffic, and writing the kernel directly is what makes the rest of this list possible.
- **No atomics in the adjoint.** One thread per image point owns the whole gather axis, so every write is exclusive. Accumulators sit in shared memory as `[nh][block]`, bank-conflict free for any per-thread bin index. Block size is chosen automatically as the largest of {256, 128, 64, 32} that keeps the accumulators within 32 KB; for large `nh` it drops to 32 threads and opts in to the device limit.
- **Forward: one block per trace**, accumulated in shared memory with cheap shared-memory atomics and written out once.
- **Exact adjointness by construction.** Both kernels compute the sample index and interpolation weights with the *same* float32 expression, so the pair is the transpose of one sparse matrix; only the summation order differs.
- **float64 accumulators by default.** A migrated sample sums 10^4 to 10^5 terms. With float64 accumulation the CUDA engine is bit-identical to the NumPy reference; float32 accumulation costs about 1e-7 relative error for roughly 1.5x speed.
- **Model layout `(nh, nx, nz)`** keeps both the adjoint writeback and the forward model reads coalesced.
- **Compile-time specialisation.** `nh`, block size, accumulator type and the offset/angle switch are `-D` flags, so `nh` is a true compile-time constant. Changing it costs about a second of NVRTC, once.
- **PyTorch does no numerical work.** `kirchcig.torch` exchanges GPU buffers with CuPy through DLPack (nothing leaves the device; non-default streams are honoured) and registers the pair as `autograd.Function`s. Remove torch and the CUDA engine is unaffected.

Large problems are handled by chunking the time axis and splitting the source axis; both are exact, and the test suite checks that a split result equals an unsplit one.

## Limitations

- **No anti-alias filtering yet.** Operator aliasing shows up as steeply dipping artefacts at large offsets. Planned for v0.2 (triangle filter bank, Lumley-Claerbout).
- **2D only.** The traveltime tables are the obstacle, not the kernels.
- **Offset binning uses absolute half-offset**, so positive and negative offsets are not distinguished.

## Related

SEG-Y I/O: [segyio](https://github.com/equinor/segyio). Operator algebra and solvers: [PyLops](https://github.com/PyLops/pylops), which kirchcig plugs into via `to_scipy()`. Wave-equation modelling and RTM: [Deepwave](https://github.com/ar4/deepwave).

## Requirements

Python >= 3.10 and `numpy`. Optional: `cupy` matching your CUDA version (GPU engine), `scikit-fmm` (eikonal traveltimes), `scipy` (`to_scipy()`), `torch` (autograd wrapper only, not used for compute).

Contributions welcome. `pytest -q` runs the dot-product tests under both engines; CUDA tests skip when no GPU is visible.

## Citing

<TODO: Zenodo DOI>

## License

MIT

---

<sub>Keywords: Kirchhoff migration, prestack depth migration, common-image gather, CIG, angle gather, offset gather, GPU seismic imaging, CUDA, CuPy, least-squares migration, LSM, exact adjoint, demigration, migration velocity analysis, MVA, AVO, AVA, PyTorch, seismic inversion.</sub>