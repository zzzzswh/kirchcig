# Changelog

## Unreleased

- Adjoint memory traffic: the traveltime gradients of `aa_stretch` are packed
  into the table element instead of a separate table (fields `t, a, d, w, gx,
  gz`, padded to 1/2/4/8 floats), and the adjoint kernel processes sources in
  register chunks of `schunk` (default 4), loading each receiver-table element
  once per chunk instead of once per source. `schunk=` constructor knob,
  `bench.py --schunk`.

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
- Anti-aliased stretch, `aa_stretch=True` (default with `aa`): the filter
  width also covers the image cell's time footprint `|dtau/dx| dx + |dtau/dz|
  dz` (Madagascar `aastretch`), RMS-combined with the trace-axis term, so a
  depth grid coarser than the time sampling demigrates into a smooth trace
  instead of a comb. Needs a second, `float2` traveltime-gradient table per
  side (`table_gradients`, shared with the emergence angles). Width arithmetic
  uses `__fmul_rn`/`__fadd_rn` so the two engines stay bit-identical.
- Amplitude weights, `weight=`: per-side tables multiplied in both kernels
  (`w_s * w_r`, packed into the traveltime-table element, no extra memory
  transaction). Presets `"obliquity"` (`sqrt(cos theta)`) and `"spreading"`
  (`1/sqrt(t)`), lists of presets, a callable `f(t, theta, dt)`, or a pair of
  ready tables. `weights` property, `weight_tables()`, `bench.py --weight`.
- Half-derivative (rho) filter, `halfderiv=True`: `H = sqrt(1 - rho e^{-iw})`
  on the traces after `forward`, its exact transpose on the data before
  `adjoint` (Madagascar `sf_halfint`, same default leak `1 - 1/nt`). Float64
  FFTs on the engine's device, traces padded to >= 2 nt. Planar-reflector
  demigration returns a zero-phase wavelet with it (-45 degrees and
  `w^-1/2` without). `kirchcig.HalfDerivative`, `bench.py --halfderiv`.
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
