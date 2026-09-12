"""Migration velocity analysis in one figure.

The same (correct) data is migrated with three different velocities. Residual
curvature along the offset axis is the observable: flat means the velocity is
right, upward or downward curvature means it is too low or too high.

    python examples/vel_analysis.py [--engine auto|cuda|numpy]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kirchcig import KirchhoffCIG

p = argparse.ArgumentParser()
p.add_argument("--engine", default="auto")
p.add_argument("--nh", type=int, default=16)
p.add_argument("--hmax", type=float, default=500.0)
a = p.parse_args()

V_TRUE = 2000.0
op_true = KirchhoffCIG.demo(engine=a.engine, nh=a.nh, hmax=a.hmax)
data = op_true.demo_data()          # data generated with the true velocity

cases = [(V_TRUE * 0.9, "too low"), (V_TRUE, "correct"), (V_TRUE * 1.1, "too high")]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
for ax, (v, label) in zip(axes, cases):
    op = KirchhoffCIG.demo(engine=a.engine, nh=a.nh, hmax=a.hmax, v=v)
    cig = np.asarray(op.adjoint(data))           # migrate correct data, wrong velocity
    g = cig[:, np.argmin(abs(op.x - 500.0)), :].T
    c = np.abs(g).max()
    ax.imshow(g, aspect="auto", cmap="gray", vmin=-c, vmax=c,
              extent=[0, op.hmax, op.z[-1], op.z[0]])
    ax.set(title=f"v = {v:.0f} m/s  ({label})", xlabel="half-offset [m]")

axes[0].set_ylabel("z [m]")
fig.suptitle("Common-image gathers at x = 500 m, migrated with three velocities",
             fontsize=11, y=1.0)
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "docs" / "img"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "vel_analysis.png", dpi=140, bbox_inches="tight")
print("wrote", out / "vel_analysis.png")
