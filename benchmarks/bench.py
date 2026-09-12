"""Timing of the cuda engine on a README-sized problem.

    python benchmarks/bench.py [--nh 32] [--domain offset|angle] [--acc float64] [--aa] [--aperture 60]

Reports wall time per adjoint / forward and the equivalent trace-image-point
throughput (ns*nr*npts pair evaluations per second).
"""
import argparse
import time

import numpy as np

from kirchcig import KirchhoffCIG, cuda_available

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=401)
p.add_argument("--nz", type=int, default=201)
p.add_argument("--ns", type=int, default=100)
p.add_argument("--nr", type=int, default=200)
p.add_argument("--nt", type=int, default=1500)
p.add_argument("--nh", type=int, default=32)
p.add_argument("--domain", default="offset")
p.add_argument("--acc", default="float64")
p.add_argument("--engine", default="cuda")
p.add_argument("--aa", action="store_true", help="anti-alias filtering")
p.add_argument("--aperture", type=float, default=None, help="migration aperture [deg]")
p.add_argument("--reps", type=int, default=5)
a = p.parse_args()

if a.engine == "cuda" and not cuda_available():
    raise SystemExit("no CUDA device visible; pass --engine numpy for a (slow) CPU run")

srcs = np.stack([np.linspace(0, (a.nx - 1) * 10.0, a.ns), np.zeros(a.ns)])
recs = np.stack([np.linspace(0, (a.nx - 1) * 10.0, a.nr), np.zeros(a.nr)])
hmax = 2000.0 if a.domain == "offset" else 60.0

t0 = time.perf_counter()
op = KirchhoffCIG(nx=a.nx, nz=a.nz, dx=10.0, dz=10.0, srcs=srcs, recs=recs, nt=a.nt, dt=0.004,
                  vel=2000.0, nh=a.nh, hmax=hmax, domain=a.domain, engine=a.engine, acc=a.acc,
                  aa=a.aa, aperture=a.aperture)
print(f"build (tables + compile): {time.perf_counter() - t0:.2f} s   {op._eng!r}")

xp = np
if op.engine == "cuda":
    import cupy as cp
    xp = cp
rng = np.random.default_rng(0)
data = xp.asarray(rng.standard_normal(op.shape_data, dtype=np.float32))
cig = xp.asarray(rng.standard_normal(op.shape_model, dtype=np.float32))


def timeit(fn, x):
    fn(x)                                    # warm-up (first launch compiles / allocates)
    if op.engine == "cuda":
        cp.cuda.Device().synchronize()
    t = time.perf_counter()
    for _ in range(a.reps):
        y = fn(x)
    if op.engine == "cuda":
        cp.cuda.Device().synchronize()
    return (time.perf_counter() - t) / a.reps, y


pairs = op.ns * op.nr * op.nx * op.nz          # evaluated by the kernels, whether or not they contribute
if a.aperture is not None:
    ms, mr = op.aperture_masks()
    kept = (ms.reshape(op.ns, -1).sum(0, dtype=np.float64) * mr.reshape(op.nr, -1).sum(0)).sum()
    print(f"aperture {a.aperture:g} deg keeps {kept / pairs:.1%} of the trace-image-point pairs")
ta, _ = timeit(op.adjoint, data)
tf, _ = timeit(op.forward, cig)
print(f"adjoint : {ta * 1e3:8.1f} ms   {pairs / ta / 1e9:6.1f} G pair-evals/s")
print(f"forward : {tf * 1e3:8.1f} ms   {pairs / tf / 1e9:6.1f} G pair-evals/s")
ok, err = op.dot_test(return_error=True)
print(f"dot test: {ok} (relative error {err:.2e})")
