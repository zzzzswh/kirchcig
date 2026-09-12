"""Least-squares migration of the demo data with SciPy's LSQR.

Runs under either engine; on a GPU use engine="cuda" (default "auto").
"""
import numpy as np
import scipy.sparse.linalg as spla

from kirchcig import KirchhoffCIG

op = KirchhoffCIG.demo(engine="auto", nh=8, hmax=500.0)
print(op, "| dot test:", op.dot_test())

data = op.demo_data()
cig_mig = op.adjoint(data)                      # plain migration
A = op.to_scipy()                               # LinearOperator (size_data, size_model)
cig_lsm = spla.lsqr(A, data.ravel(), iter_lim=20, show=False)[0].reshape(op.shape_model)

res_mig = np.linalg.norm(op.forward(cig_mig / np.abs(cig_mig).max()) - data)
res_lsm = np.linalg.norm(op.forward(cig_lsm) - data)
print(f"data residual  migration (scaled): {res_mig:.3e}   LSQR(20): {res_lsm:.3e}")

img = cig_lsm.sum(0)
ix, iz = np.unravel_index(np.argmax(np.abs(img)), img.shape)
print(f"LSM image peaks at x={op.x[ix]:.0f} m, z={op.z[iz]:.0f} m (true scatterer at 500, 300)")
