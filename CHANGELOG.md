# Changelog

## 0.1.0

- Offset- and angle-domain Kirchhoff migration to CIGs with the exact adjoint.
- `cuda` engine (CuPy/NVRTC): atomic-free adjoint, shared-memory forward,
  float64 accumulation, compile-time specialisation on `nh`, automatic block
  selection, source-split for GPU occupancy, exact time windowing for long traces.
- `numpy` reference engine mirroring the kernel arithmetic bit-for-bit.
- Analytic and eikonal (scikit-fmm) traveltimes with near-source correction;
  user-supplied tables.
- PyTorch autograd wrapper with DLPack zero-copy and second-order support.
- `KirchhoffCIG.demo()`, `dot_test()`, `to_scipy()`.
