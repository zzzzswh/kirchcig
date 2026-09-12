# kirchcig

GPU Kirchhoff migration to common-image gathers, and its exact adjoint.

Turns prestack seismic data into offset- or angle-domain common-image gathers
(CIGs) on the GPU, and provides the matched demigration operator so the pair can
be used inside least-squares migration or a PyTorch training loop.

```bash
pip install kirchcig
```

No compiler needed. Kernels are compiled at runtime by NVRTC via CuPy.

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

That is the whole thing for the common case. The stacked image is not a separate
code path: `nh=1` gives it directly.

## What this is for

A CIG keeps the offset (or opening-angle) axis instead of summing it away. The
residual curvature along that axis is the primary observable for migration
velocity analysis: flat events mean the velocity is right, upward or downward
curvature means it is too low or too high. That information is destroyed by
stacking, which is why production Kirchhoff migration outputs gathers.

CIGs are also the natural input for AVA/AVO work and for angle-dependent
regularisation in least-squares migration.

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

`forward` and `adjoint` are exact adjoints of each other to float precision, so
they drop straight into any least-squares solver:

```python
import scipy.sparse.linalg as spla

A = spla.LinearOperator(
    shape=(op.size_data, op.size_model),
    matvec=lambda x: op.forward(x.reshape(op.shape_model)).ravel(),
    rmatvec=lambda y: op.adjoint(y.reshape(op.shape_data)).ravel(),
    dtype="float32",
)
cig_lsm = spla.lsqr(A, data.ravel(), iter_lim=20)[0].reshape(op.shape_model)
```

## Angle-domain gathers

```python
op = KirchhoffCIG(..., domain="angle", nh=30, hmax=60.0)   # 30 bins, 0-60 deg
```

In `domain="angle"`, `hmax` is the maximum half opening angle in degrees. The
bin index comes from the source- and receiver-side emergence angles, computed
from the traveltime gradients.

## PyTorch

```python
import torch
from kirchcig.torch import TorchKirchhoffCIG

top = TorchKirchhoffCIG(op)

cig = top.adjoint(data)        # differentiable w.r.t. data
res = top.forward(cig) - data  # differentiable w.r.t. cig
res.pow(2).sum().backward()
```

Tensors stay on the GPU; no host round-trip. Because the operator is linear, the
backward pass of `forward` is `adjoint` and vice versa, so no extra derivation
is involved and second-order derivatives work too.

This is what makes deep-prior LSM and plug-and-play regularisation one-liners:
parametrise the CIG with a network, and let autograd do the rest.

## Bring your own traveltimes

By default traveltimes come from an eikonal solve (`scikit-fmm`) or an analytic
expression for constant velocity. If you have your own propagator, pass the
tables directly:

```python
op = KirchhoffCIG(..., trav=(trav_srcs, trav_recs))
# trav_srcs (ns, nx, nz)   source-to-image-point traveltimes [s]
# trav_recs (nr, nx, nz)   image-point-to-receiver traveltimes [s]
```

Note the layout: the source/receiver axis comes first. This is transposed
relative to what some other libraries use, and is what keeps the adjoint kernel
reads coalesced.

## Verifying an installation

```python
from kirchcig import KirchhoffCIG
op = KirchhoffCIG.demo()       # constant velocity, single point scatterer
assert op.dot_test()
cig = op.adjoint(op.demo_data())
assert abs(cig.sum(0) - op.demo_image()).max() < 1e-4
```

## Design notes

**Adjoint.** One CUDA thread per image point. Each thread owns the whole gather
axis for its own image point, so every write is exclusive and the kernel uses no
atomics at all. Accumulators live in shared memory laid out as `[nh][block]`,
which is bank-conflict free for any per-thread bin index. Neighbouring image
points have similar traveltimes, so neighbouring threads read neighbouring time
samples and the scattered data reads coalesce well in practice.

**Forward.** One block per trace. The block accumulates the whole trace in
shared memory using cheap shared-memory atomics and writes it out once.

**Accumulation precision.** A single migrated sample sums 10^4 to 10^5 terms, so
accumulators are float64 by default even though the tables and arrays are
float32. This matters: in float32 the dot-product test passes only to about
1e-3, which is loose enough to hide real bugs.

**Model layout.** The gather axis is outermost, `(nh, nx, nz)`. With one thread
per image point this is what makes both the adjoint writeback and the forward
model reads coalesced.

**Compile-time specialisation.** `nh`, the block size and the accumulator type
are injected as `-D` flags, so `nh` is a true compile-time constant. Changing it
triggers a recompile of about a second, cached by CuPy afterwards.

## Scope

In scope:

- 2D, offset- and angle-domain gathers
- `cuda` engine and a plain `numpy` reference engine
- Exact adjoint, verified by dot-product tests in CI
- PyTorch autograd wrapper

Deliberately out of scope:

- SEG-Y I/O. Use [segyio](https://github.com/equinor/segyio).
- Linear-operator algebra. The operator plugs into
  [PyLops](https://github.com/PyLops/pylops) or SciPy.
- Wave-equation migration. See [Deepwave](https://github.com/ar4/deepwave)
  for modelling and RTM.

Known limitations, honestly:

- **No anti-alias filtering yet.** Kirchhoff operator aliasing shows up as
  steeply dipping artefacts at large offsets. Planned for v0.2 (triangle filter
  bank, Lumley-Claerbout).
- 3D is not implemented. The traveltime tables are the obstacle, not the
  kernels.
- Offset binning uses absolute half-offset, so it does not distinguish positive
  and negative offsets.

## Requirements

- Python >= 3.10
- `numpy`
- `cupy` matching your CUDA version, for the `cuda` engine
- `scikit-fmm`, for eikonal traveltimes
- `torch`, optional, for the autograd wrapper

The `numpy` engine runs anywhere, including CPU-only machines and notebook
sandboxes. It is slow and exists as the reference implementation and test
oracle, but every example in this README runs under it.

## Citing

If this is useful in published work, please cite <TODO: Zenodo DOI / paper>.

## License

MIT
