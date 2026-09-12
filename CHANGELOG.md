# Changelog

## 0.2.0

- Anti-alias filtering, `aa=True`, on both engines: a dip-adaptive triangle
  filter (Lumley, Claerbout and Bevc 1994) applied through the three-tap
  double-integration identity, so the cost per contribution does not depend on
  the filter width. Operator dips come from differencing the traveltime tables,
  so it works with user-supplied tables too. `aa_factor` matches the
  `antialias=` parameter of Claerbout's `trimo` and Madagascar's `sfmig2`;
  `aa_max` caps the half-width. The pair stays an exact transpose.
- CUDA: `AA` compile-time switch; dip tables packed into the traveltime table
  element (`{t, d}` / `{t, a, d, _}`) for one coalesced load; float64 double
  integral prepared and reverse-integrated with CuPy around the kernels; the
  forward's trace accumulator is float64 under anti-aliasing whatever `acc`
  is, so `acc="float32"` costs no more accuracy with the filter than without.
- `KirchhoffCIG.aa_widths()` diagnostic, `examples/antialias.py`,
  `benchmarks/bench.py --aa`.
- Migration aperture: `aperture=` (cone half-angle from the vertical, degrees,
  as `sfkirmig`) and `apt=` (lateral distance, metres, as `sfmig2`). Applied
  as a host-side mask on the traveltime tables (masked entries are pushed past
  the trace end, where the kernels already skip), so both engines drop exactly
  the same contributions, the pair stays an exact transpose, and the skipped
  contributions cost nothing. `aperture_masks()` diagnostic,
  `benchmarks/bench.py --aperture`.
- `KirchhoffCIG.demo()` now forwards unknown keywords to the constructor.
- Fixed the project URL in `pyproject.toml`.

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
