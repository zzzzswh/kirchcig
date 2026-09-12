"""Operator anti-aliasing on a single-shot impulse response.

A band-limited spike in every trace of one shot migrates to one ellipse per
trace. With the receivers 100 m apart the plain summation leaves the discrete
ellipses visible as steeply dipping, criss-crossing artefacts; with ``aa=True``
the triangle filter widens each contribution where the operator is steep and
the arcs merge into the smooth wavefront that a densely sampled shot gives.

    python examples/antialias.py [--engine auto|cuda|numpy] [--aa_factor 1.0]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kirchcig import KirchhoffCIG
from kirchcig._operator import ricker

p = argparse.ArgumentParser()
p.add_argument("--engine", default="auto")
p.add_argument("--aa_factor", type=float, default=1.0)
a = p.parse_args()

v, nx, nz, dx, dz, dt, nt, f0 = 2000.0, 301, 76, 10.0, 10.0, 0.002, 500, 30.0
srcs = np.stack([[1500.0], [0.0]])


def image(nrec, **extra):
    recs = np.stack([np.linspace(500.0, 2500.0, nrec), np.zeros(nrec)])
    data = np.broadcast_to(ricker(np.arange(nt) * dt - 0.5, f0), (1, nrec, nt))
    op = KirchhoffCIG(nx=nx, nz=nz, dx=dx, dz=dz, srcs=srcs, recs=recs, nt=nt, dt=dt,
                      vel=v, nh=1, engine=a.engine, aa_max=64, **extra)
    return op, np.asarray(op.adjoint(data.astype(np.float32))).sum(0) / nrec


cases = [
    ("31 receivers @ 100 m, no anti-alias", image(31)),
    (f"31 receivers @ 100 m, aa=True", image(31, aa=True, aa_factor=a.aa_factor)),
    ("301 receivers @ 10 m, reference", image(301)),
]

fig, ax = plt.subplots(1, 3, figsize=(16, 3.6), sharey=True)
c = 0.3 * np.abs(cases[2][1][1]).max()          # clip to make the weak alias arcs visible
for axi, (title, (op, img)) in zip(ax, cases):
    axi.imshow(img.T, aspect="auto", cmap="gray", vmin=-c, vmax=c,
               extent=[op.x[0], op.x[-1], op.z[-1], op.z[0]])
    axi.set(title=title, xlabel="x [m]")
ax[0].set_ylabel("z [m]")
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "docs" / "img"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "antialias.png", dpi=100, bbox_inches="tight")
print("wrote", out / "antialias.png")
