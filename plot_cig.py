# plot_cig.py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from kirchcig import KirchhoffCIG

op = KirchhoffCIG.demo(engine="auto", nh=16, hmax=500.0)
cig = np.asarray(op.adjoint(op.demo_data()))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
img = cig.sum(0).T
ax[0].imshow(img, aspect="auto", cmap="gray",
             extent=[op.x[0], op.x[-1], op.z[-1], op.z[0]])
ax[0].set(title="stacked image", xlabel="x [m]", ylabel="z [m]")

ix = np.argmin(abs(op.x - 500.0))          # 散射体所在的 x
g = cig[:, ix, :].T
ax[1].imshow(g, aspect="auto", cmap="gray",
             extent=[0, op.hmax, op.z[-1], op.z[0]])
ax[1].set(title=f"CIG at x=500 m ({op.domain})", xlabel="half-offset [m]")
plt.tight_layout(); plt.savefig("cig.png", dpi=130)
print("wrote cig.png")
