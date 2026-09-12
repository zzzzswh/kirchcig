"""Stacked migrated image and the common-image gather at the scatterer.

Produces the figure used as the README cover.

    python examples/plot_cig.py [--engine auto|cuda|numpy] [--domain offset|angle]
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
p.add_argument("--domain", default="offset", choices=["offset", "angle"])
p.add_argument("--nh", type=int, default=16)
a = p.parse_args()

hmax = 500.0 if a.domain == "offset" else 60.0
op = KirchhoffCIG.demo(engine=a.engine, nh=a.nh, hmax=hmax, domain=a.domain)
cig = np.asarray(op.adjoint(op.demo_data()))

img = cig.sum(0).T
gather = cig[:, np.argmin(abs(op.x - 500.0)), :].T
xunit = "half-offset [m]" if a.domain == "offset" else "half opening angle [deg]"

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

c = np.abs(img).max()
ax[0].imshow(img, aspect="auto", cmap="gray", vmin=-c, vmax=c,
             extent=[op.x[0], op.x[-1], op.z[-1], op.z[0]])
ax[0].set(title="Stacked migrated image", xlabel="x [m]", ylabel="z [m]")

c = np.abs(gather).max()
ax[1].imshow(gather, aspect="auto", cmap="gray", vmin=-c, vmax=c,
             extent=[0, op.hmax, op.z[-1], op.z[0]])
ax[1].set(title=f"Common-image gather at x = 500 m ({a.domain} domain)",
          xlabel=xunit, ylabel="z [m]")

fig.suptitle(f"kirchcig  |  {op.engine} engine  |  flat gather = correct velocity",
             fontsize=10, y=0.99)
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "docs" / "img"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "cig.png", dpi=140, bbox_inches="tight")
print("wrote", out / "cig.png")
