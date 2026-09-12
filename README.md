# kirchcig

**GPU Kirchhoff migration to common-image gathers (CIGs), with an exact adjoint.**

English | [简体中文](README.zh-CN.md)

Turns prestack seismic data into offset- or angle-domain common-image gathers on the GPU, and gives you the matched demigration operator, so the pair drops straight into least-squares migration or a PyTorch training loop.

![Stacked migrated image and the common-image gather at the scatterer](docs/img/cig.png)

```bash
pip install "kirchcig[cuda12]"     # CUDA 12.x
pip install "kirchcig[cuda11]"     # CUDA 11.x
pip install kirchcig               # CPU-only NumPy reference engine
```

No compiler needed. CUDA kernels are compiled at runtime by NVRTC through CuPy.

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

image = cig.sum(0)             # (nx, nz) stacked migrated image
```

That is the whole thing for the common case. The stacked image is not a separate code path: `nh=1` gives it directly.

No data at hand? Everything below runs out of the box:

```python
from kirchcig import KirchhoffCIG
op = KirchhoffCIG.demo()       # constant velocity, one point scatterer
assert op.dot_test()
cig = op.adjoint(op.demo_data())
```

## Why gathers instead of a stacked image

A CIG keeps the offset (or opening-angle) axis instead of summing it away. Residual curvature along that axis is the primary observable for migration velocity analysis: flat events mean the velocity is right, upward or downward curvature means it is too low or too high. Stacking destroys that information, which is why production Kirchhoff migration outputs gathers.

![Gathers migrated with three velocities: too low, correct, too high](docs/img/vel_analysis.png)

<sub>The same data migrated with three velocities. Flat gather = correct velocity; the curvature is what migration velocity analysis measures.</sub>

CIGs are also the natural input for AVO/AVA work and for angle-dependent regularisation in least-squares migration.

## The operator pair

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
    engine="cuda",             # or "numpy"
)

cig  = op.adjoint(data)        # (ns, nr, nt) -> (nh, nx, nz)   migration
data = op.forward(cig)         # (nh, nx, nz) -> (ns, nr, nt)   demigration

op.dot_test()                  # True
```

`forward` and `adjoint` are exact transposes of each other to accumulator precision, so they drop into any least-squares solver:

```python
import scipy.sparse.linalg as spla
cig_lsm = spla.lsqr(op.to_scipy(), data.ravel(), iter_lim=20)[0].reshape(op.shape_model)
```

### Angle-domain gathers

```python
op = KirchhoffCIG(..., domain="angle", nh=30, hmax=60.0)   # 30 bins, 0-60 deg
```

In `domain="angle"`, `hmax` is the maximum half opening angle in degrees. The bin index comes from source- and receiver-side emergence angles, computed from the traveltime gradients.

### PyTorch autograd

```python
import torch
from kirchcig.torch import TorchKirchhoffCIG

top = TorchKirchhoffCIG(op)
cig = top.adjoint(data)        # differentiable w.r.t. data
res = top.forward(cig) - data
res.pow(2).sum().backward()
```

Tensors stay on the GPU, no host round-trip. Because the operator is linear, the backward pass of `forward` is `adjoint` and vice versa, so second-order derivatives work too. Deep-prior LSM and plug-and-play regularisation become one-liners: parametrise the CIG with a network and let autograd do the rest.

### Bring your own traveltimes

Traveltimes come from an eikonal solve (`scikit-fmm`) or an analytic expression for constant velocity. If you have your own propagator, pass the tables directly:

```python
op = KirchhoffCIG(..., trav=(trav_srcs, trav_recs))
# trav_srcs (ns, nx, nz)   source-to-image-point traveltimes [s]
# trav_recs (nr, nx, nz)   image-point-to-receiver traveltimes [s]
```

Note the layout: the source/receiver axis comes first. This is transposed relative to some other libraries, and is what keeps the adjoint kernel reads coalesced.

## Where this fits

| You want | Use |
|---|---|
| Kirchhoff CIGs on a GPU, plus the adjoint for LSM | **kirchcig** |
| SEG-Y I/O | [segyio](https://github.com/equinor/segyio) |
| Linear-operator algebra, solvers, regularisation | [PyLops](https://github.com/PyLops/pylops) — kirchcig plugs in via `to_scipy()` |
| Wave-equation modelling and RTM | [Deepwave](https://github.com/ar4/deepwave) |

## Design notes

- **No atomics in the adjoint.** One thread per image point owns the whole gather axis, so every write is exclusive. Accumulators live in shared memory as `[nh][block]`, bank-conflict free for any per-thread bin index.
- **Forward is one block per trace**, accumulating the trace in shared memory and writing it out once.
- **float64 accumulators by default.** A migrated sample sums 10^4 to 10^5 terms; in float32 the dot test passes only to about 1e-3, loose enough to hide real bugs.
- **Model layout `(nh, nx, nz)`**, gather axis outermost, which keeps both the adjoint writeback and the forward model reads coalesced.
- **Compile-time specialisation.** `nh`, block size and accumulator type are `-D` flags, so `nh` is a true compile-time constant. Changing it costs about a second of NVRTC, cached by CuPy afterwards.

## Limitations, honestly

- **No anti-alias filtering yet.** Kirchhoff operator aliasing shows up as steeply dipping artefacts at large offsets. Planned for v0.2 (triangle filter bank, Lumley-Claerbout).
- **2D only.** The traveltime tables are the obstacle, not the kernels.
- **Offset binning uses absolute half-offset**, so positive and negative offsets are not distinguished.

The `numpy` engine runs anywhere, including CPU-only machines and notebook sandboxes. It is slow and exists as the reference implementation and test oracle, but every example here runs under it.

## Requirements

Python >= 3.10, `numpy`. Optional: `cupy` matching your CUDA version (GPU engine), `scikit-fmm` (eikonal traveltimes), `torch` (autograd wrapper), `scipy` (`to_scipy()`).

## Contributing

Issues and PRs welcome. `pytest -q` runs the dot-product tests under both engines; CUDA tests skip automatically when no GPU is visible.

## Citing

If this is useful in published work, please cite <TODO: Zenodo DOI>.

## License

MIT

---

<sub>Keywords: Kirchhoff migration, prestack depth migration, common-image gather, CIG, angle gather, offset gather, GPU seismic imaging, CUDA, CuPy, least-squares migration, LSM, exact adjoint, demigration, migration velocity analysis, MVA, AVO, AVA, PyTorch, seismic inversion.</sub>