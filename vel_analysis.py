# vel_analysis.py —— 速度对了/低了/高了，道集长什么样
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from kirchcig import KirchhoffCIG

op_true = KirchhoffCIG.demo(engine="auto", nh=16, hmax=500.0)
data = op_true.demo_data()                    # 真实速度 2000 m/s 生成的数据

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, v in zip(axes, [1800.0, 2000.0, 2200.0]):
    op = KirchhoffCIG.demo(engine="auto", nh=16, hmax=500.0, v=v)
    cig = np.asarray(op.adjoint(data))        # 用错误速度偏移正确的数据
    ix = np.argmin(abs(op.x - 500.0))
    ax.imshow(cig[:, ix, :].T, aspect="auto", cmap="gray",
              extent=[0, op.hmax, op.z[-1], op.z[0]])
    ax.set(title=f"v = {v:.0f} m/s", xlabel="half-offset [m]")
axes[0].set_ylabel("z [m]")
axes[1].set_title("v = 2000 m/s  (正确)")
plt.tight_layout(); plt.savefig("vel_analysis.png", dpi=130)
print("wrote vel_analysis.png")
