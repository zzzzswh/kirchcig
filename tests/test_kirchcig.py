import numpy as np
import pytest

from kirchcig import (KirchhoffCIG, analytic_traveltime, cuda_available,
                      eikonal_traveltime, emergence_angles, migrate)

SMALL = dict(nx=41, nz=31, nt=301, ns=5, nr=21)   # keeps the numpy engine fast


def _rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-30)


# --------------------------------------------------------------------- numpy
@pytest.mark.parametrize("kw", [
    dict(nh=1),
    dict(nh=8, hmax=400.0),
    dict(nh=30, hmax=60.0, domain="angle"),
])
def test_dot_test_numpy(kw):
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, **kw)
    ok, err = op.dot_test(return_error=True)
    assert ok, err
    assert err < 1e-6


def test_demo_focuses_on_scatterer():
    op = KirchhoffCIG.demo(engine="numpy")
    cig = op.adjoint(op.demo_data())
    assert cig.shape == op.shape_model
    img = cig.sum(0)
    ix, iz = np.unravel_index(np.argmax(img), img.shape)
    xp, zp = op._demo["scatterer"]
    assert abs(op.x[ix] - xp) <= op.dx and abs(op.z[iz] - zp) <= op.dz
    assert abs(img - op.demo_image()).max() < 1e-4


def test_shapes_sizes_and_axes():
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=8, hmax=400.0)
    assert op.shape_model == (8, 41, 31) and op.shape_data == (5, 21, 301)
    assert op.size_model == 8 * 41 * 31 and op.size_data == 5 * 21 * 301
    assert np.allclose(op.h_axis, (np.arange(8) + 0.5) * 50.0)
    assert op.offset_bins.shape == (5, 21)
    assert (op.offset_bins < 8).all()
    # pairs beyond hmax are dropped, all others binned
    hoff = 0.5 * np.abs(op.srcs[0][:, None] - op.recs[0][None, :])
    assert ((op.offset_bins == -1) == (hoff > 400.0)).all()


def test_default_hmax_uses_every_trace():
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=1, hmax=None)
    assert (op.offset_bins == 0).all()


def test_forward_of_spike_is_a_spike_at_the_traveltime():
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=1)
    xp, zp = 200.0, 150.0
    cig = np.zeros(op.shape_model, np.float32)
    cig[0, int(xp / op.dx), int(zp / op.dz)] = 1.0
    d = op.forward(cig)
    ts = np.hypot(op.srcs[0] - xp, op.srcs[1] - zp) / op._demo["v"]
    tr = np.hypot(op.recs[0] - xp, op.recs[1] - zp) / op._demo["v"]
    t0 = (ts[:, None] + tr[None, :]) / op.dt
    inside = t0 < op.nt - 1                           # traces whose event is recorded
    assert inside.any() and not inside.all()
    it = np.argmax(d, axis=-1)
    assert np.abs(it - t0)[inside].max() <= 1.0
    assert np.allclose(d.sum(-1)[inside], 1.0, atol=1e-5)   # linear interpolation preserves the sum
    assert (d[~inside] == 0).all()                    # events past the trace end contribute nothing


def test_migrate_function_matches_operator():
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=8, hmax=400.0)
    d = op.demo_data()
    vel = np.full((op.nx, op.nz), op._demo["v"])
    cig = migrate(d, vel, op.srcs, op.recs, dt=op.dt, dx=op.dx, dz=op.dz,
                  nh=8, hmax=400.0, engine="numpy")
    assert np.array_equal(cig, op.adjoint(d))
    cig2 = migrate(d, op._demo["v"], op.srcs, op.recs, dt=op.dt, dx=op.dx, dz=op.dz,
                   nh=8, hmax=400.0, engine="numpy", nx=op.nx, nz=op.nz)
    assert np.array_equal(cig2, cig)


def test_user_traveltime_tables():
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=8, hmax=400.0)
    op2 = KirchhoffCIG(nx=op.nx, nz=op.nz, dx=op.dx, dz=op.dz, srcs=op.srcs, recs=op.recs,
                       nt=op.nt, dt=op.dt, trav=(op.trav_srcs, op.trav_recs), nh=8, hmax=400.0,
                       engine="numpy")
    d = op.demo_data()
    assert np.array_equal(op2.adjoint(d), op.adjoint(d))
    with pytest.raises(ValueError):
        KirchhoffCIG(nx=op.nx, nz=op.nz, dx=op.dx, dz=op.dz, srcs=op.srcs, recs=op.recs,
                     nt=op.nt, dt=op.dt, trav=(op.trav_srcs.transpose(1, 2, 0), op.trav_recs),
                     engine="numpy")


def test_non_finite_tables_are_ignored_not_fatal():
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=1)
    ts = op.trav_srcs.copy()
    ts[0, :5, :5] = np.nan
    op2 = KirchhoffCIG(nx=op.nx, nz=op.nz, dx=op.dx, dz=op.dz, srcs=op.srcs, recs=op.recs,
                       nt=op.nt, dt=op.dt, trav=(ts, op.trav_recs), nh=1, engine="numpy")
    cig = op2.adjoint(op.demo_data())
    assert np.isfinite(cig).all()
    assert op2.dot_test()


def test_scipy_linear_operator():
    spla = pytest.importorskip("scipy.sparse.linalg")
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=4, hmax=400.0)
    A = op.to_scipy()
    d = op.demo_data()
    x = spla.lsqr(A, d.ravel(), iter_lim=5)[0].reshape(op.shape_model)
    assert x.shape == op.shape_model and np.isfinite(x).all()


# --------------------------------------------------------------- traveltime
def test_analytic_tables_layout():
    pos = np.array([[0.0, 50.0], [0.0, 0.0]])
    t = analytic_traveltime(pos, 2000.0, 11, 6, 10.0, 10.0)
    assert t.shape == (2, 11, 6) and t.dtype == np.float32
    assert t[1, 5, 0] == 0.0 and np.isclose(t[0, 10, 0], 0.05)


def test_eikonal_matches_analytic_for_constant_velocity():
    pytest.importorskip("skfmm")
    pos = np.array([[300.0, 650.0], [0.0, 0.0]])
    vel = np.full((101, 61), 2000.0)
    te = eikonal_traveltime(pos, vel, 10.0, 10.0)
    ta = analytic_traveltime(pos, 2000.0, 101, 61, 10.0, 10.0)
    mask = ta > 0.05
    rel = np.abs(te - ta)[mask] / ta[mask]
    assert rel.max() < 0.02 and rel.mean() < 0.005


def test_gradient_velocity_operator_dot_test():
    pytest.importorskip("skfmm")
    nx, nz = 61, 41
    vel = np.broadcast_to(1500.0 + 1.0 * np.arange(nz) * 10.0, (nx, nz)).copy()
    srcs = np.stack([np.linspace(100, 500, 4), np.zeros(4)])
    recs = np.stack([np.linspace(0, 600, 25), np.zeros(25)])
    op = KirchhoffCIG(nx=nx, nz=nz, dx=10.0, dz=10.0, srcs=srcs, recs=recs, nt=301, dt=0.002,
                      vel=vel, nh=12, hmax=45.0, domain="angle", engine="numpy")
    assert op.dot_test()
    # emergence angles from the gradient are close to straight rays at shallow depth
    ang = emergence_angles(op.trav_srcs, 10.0, 10.0)
    assert ang.shape == op.trav_srcs.shape
    assert abs(ang[0, int(srcs[0, 0] / 10.0), 20]) < np.radians(3.0)


# ----------------------------------------------------------------- aperture
def test_aperture_mask_geometry():
    from kirchcig import aperture_mask
    x, z = np.arange(11) * 10.0, np.arange(6) * 10.0
    pos = np.array([[50.0, 20.0], [0.0, 10.0]])            # second one buried
    m = aperture_mask(pos, x, z, aperture=45.0)
    # 45 degrees: |dx| <= dz, points at or above the position excluded
    for i in range(2):
        dx = np.abs(x[:, None] - pos[0, i]); dz = z[None, :] - pos[1, i]
        assert np.array_equal(m[i], (dz >= 0) & (dx <= dz))
    assert m[1][:, 0].sum() == 0                           # everything above a buried source is out
    m2 = aperture_mask(pos, x, z, apt=25.0)
    assert np.array_equal(m2[0], (np.abs(x[:, None] - 50.0) <= 25.0) & np.ones((1, 6), bool))
    m3 = aperture_mask(pos, x, z, aperture=45.0, apt=25.0)
    assert np.array_equal(m3, m & m2)
    assert aperture_mask(pos, x, z).all()


def test_aperture_drops_exactly_the_masked_contributions():
    """A spike at one image point demigrates onto every trace whose source and
    receiver both see the point inside their cone, and onto no other."""
    plain = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=1)
    ap = plain._clone(aperture=30.0)
    ix, iz = 20, 20
    spike = np.zeros(plain.shape_model, np.float32); spike[0, ix, iz] = 1.0
    d0, d1 = plain.forward(spike), ap.forward(spike)
    ms, mr = ap.aperture_masks()
    keep = ms[:, ix, iz][:, None] & mr[:, ix, iz][None, :]  # (ns, nr)
    assert keep.any() and not keep.all()
    assert np.array_equal(d1[keep], d0[keep])
    assert (d1[~keep] == 0).all()
    # adjoint: same pairs dropped, so the pair stays an exact transpose
    assert ap.dot_test()
    # tables themselves are left untouched
    assert np.array_equal(ap.trav_srcs, plain.trav_srcs)


def test_aperture_ninety_and_none_are_no_limit():
    plain = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=4, hmax=400.0)
    d = plain.demo_data()
    for kw in (dict(aperture=90.0), dict(aperture=120.0), dict(aperture=None, apt=None)):
        op = plain._clone(**kw)
        assert op.aperture is None
        assert np.array_equal(op.adjoint(d), plain.adjoint(d))
    with pytest.raises(ValueError):
        plain._clone(aperture=0.0)
    with pytest.raises(ValueError):
        plain._clone(apt=-1.0)


def test_apt_lateral_aperture():
    plain = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=1)
    ap = plain._clone(apt=100.0)
    ms, mr = ap.aperture_masks()
    lateral = np.abs(plain.x[None, :, None] - plain.srcs[0][:, None, None]) <= 100.0
    assert np.array_equal(ms, np.broadcast_to(lateral, ms.shape))
    assert ap.dot_test()
    # fewer contributions than without it
    ones = np.ones(plain.shape_data, np.float32)
    assert (ap.adjoint(ones) <= plain.adjoint(ones) + 1e-6).all()
    assert ap.adjoint(ones).sum() < 0.5 * plain.adjoint(ones).sum()


def test_aperture_with_antialias_uses_unmasked_dips():
    """Dips are differenced from the clean tables, so the mask edge must not
    create huge filter widths."""
    op = KirchhoffCIG.demo(engine="numpy", aa=True, aperture=40.0)
    ref = KirchhoffCIG.demo(engine="numpy", aa=True)
    assert op.aa_widths().max() == ref.aa_widths().max()
    assert op.dot_test()


# ------------------------------------------------------------ half-derivative
def test_halfderiv_filter_is_the_half_difference():
    """Impulse response of sqrt(1 - rho z^-1) is the binomial series
    binom(1/2, k) (-rho)^k; applying it twice is the first difference; the
    adjoint is the exact transpose."""
    from kirchcig import HalfDerivative
    nt = 400
    H = HalfDerivative(nt)
    assert H.n >= 2 * nt
    k = np.arange(8)
    coef = np.array([1.0, -0.5, -0.125, -0.0625, -5 / 128, -7 / 256, -21 / 1024, -33 / 2048])
    assert np.abs(H.kernel(8) - coef * H.rho ** k).max() < 1e-5
    rng = np.random.default_rng(0)
    x = rng.standard_normal(nt).astype(np.float32)
    hh = H(H(x))
    ref = x.astype(np.float64); ref[1:] -= H.rho * x[:-1]
    assert _rel(hh, ref) < 1e-4
    y = rng.standard_normal((3, nt)).astype(np.float32)
    lhs = float(np.vdot(H(np.stack([x] * 3)), y)); rhs = float(np.vdot(np.stack([x] * 3), H(y, adj=True)))
    assert abs(lhs - rhs) / abs(lhs) < 1e-6
    # |H| ~ sqrt(omega) in the band, 45-degree phase minus a quarter sample
    # (away from DC, where the leak 1 - rho = 1/nt takes over)
    w = 2 * np.pi * np.arange(20, 80) / H.n
    assert np.allclose(np.abs(H.spec[20:80]), np.sqrt(2 * np.sin(w / 2)), rtol=1e-2)
    assert np.allclose(np.angle(H.spec[20:80]), np.pi / 4 - w / 4, atol=2e-2)


def test_halfderiv_operator_dot_test():
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=4, hmax=400.0, halfderiv=True)
    ok, err = op.dot_test(return_error=True)
    assert ok and err < 1e-6, err
    assert op._clone(aa=True, aperture=50.0).dot_test()


def _wavelet_phase_and_slope(prof, ref, dsamp):
    """Instantaneous phase at the envelope peak (degrees, folded to (-90, 90])
    and the log-log slope of |P|/|R| over 10-50 Hz."""
    hilbert = pytest.importorskip("scipy.signal").hilbert
    a = hilbert(prof)
    ipk = int(np.argmax(np.abs(a)))
    ph = ((np.degrees(np.angle(a[ipk])) + 90) % 180) - 90
    n = 8 * len(prof)
    f = np.fft.rfftfreq(n, d=dsamp)
    P, R = np.abs(np.fft.rfft(prof, n)), np.abs(np.fft.rfft(ref, n))
    band = (f > 10) & (f < 50)
    slope = np.polyfit(np.log(f[band]), np.log(P[band] / R[band]), 1)[0]
    return ph, slope


def test_halfderiv_makes_planar_reflector_modelling_zero_phase():
    """Demigrating a horizontal reflector with a zero-phase wavelet must give
    the same zero-phase wavelet back on the trace. The plain summation leaves
    the stationary-phase factor |w|^-1/2 exp(-i pi/4) of the lateral integral
    behind: -45 degrees and a -1/2 spectral slope. The half-derivative takes
    it out (up to its quarter-sample delay, ~9 degrees at 25 Hz and 4 ms)."""
    from kirchcig._operator import ricker
    v, dx, dz, nx, nz = 2000.0, 10.0, 4.0, 201, 101          # 2 dz / v == dt
    ns, nr, nt, dt, f0, zr = 21, 101, 201, 0.004, 25.0, 200.0
    srcs = np.stack([np.linspace(0, 2000, ns), np.zeros(ns)])
    recs = np.stack([np.linspace(0, 2000, nr), np.zeros(nr)])
    plain = KirchhoffCIG(nx=nx, nz=nz, dx=dx, dz=dz, srcs=srcs, recs=recs, nt=nt, dt=dt,
                         vel=v, nh=1, engine="numpy", aperture=70.0)
    hd = plain._clone(halfderiv=True)
    z, t = np.arange(nz) * dz, np.arange(nt) * dt
    m = np.zeros(plain.shape_model, np.float32)
    m[0] = ricker(2 * (z - zr) / v, f0)[None, :]
    ref = ricker(t - 2 * zr / v, f0)
    ph0, sl0 = _wavelet_phase_and_slope(plain.forward(m)[ns // 2, nr // 2], ref, dt)
    ph1, sl1 = _wavelet_phase_and_slope(hd.forward(m)[ns // 2, nr // 2], ref, dt)
    print(f"plain: {ph0:.1f} deg, w^{sl0:+.2f};  halfderiv: {ph1:.1f} deg, w^{sl1:+.2f}")
    assert abs(ph0 + 45.0) < 10.0 and abs(sl0 + 0.5) < 0.2
    assert abs(ph1) < 15.0 and abs(sl1) < 0.2


# ---------------------------------------------------------------------- cuda
needs_cuda = pytest.mark.skipif(not cuda_available(), reason="needs CuPy and a GPU")


@needs_cuda
@pytest.mark.parametrize("kw", [
    dict(nh=1),
    dict(nh=32, hmax=400.0),
    dict(nh=30, hmax=60.0, domain="angle"),
])
def test_cuda_dot_test(kw):
    op = KirchhoffCIG.demo(engine="cuda", **kw)
    ok, err = op.dot_test(return_error=True)
    assert ok, err


@needs_cuda
@pytest.mark.parametrize("kw", [dict(nh=8, hmax=400.0), dict(nh=30, hmax=60.0, domain="angle")])
def test_cuda_matches_numpy(kw):
    op_c = KirchhoffCIG.demo(engine="cuda", **SMALL, **kw)
    op_n = KirchhoffCIG.demo(engine="numpy", **SMALL, **kw)
    d = op_c.demo_data()
    assert _rel(op_c.adjoint(d), op_n.adjoint(d)) < 1e-5
    x = np.random.default_rng(0).standard_normal(op_c.shape_model, dtype=np.float32)
    assert _rel(op_c.forward(x), op_n.forward(x)) < 1e-5
    assert abs(op_c.adjoint(d).sum(0) - op_c.demo_image()).max() < 1e-4


@needs_cuda
def test_cuda_time_chunking_and_source_split_are_exact():
    op = KirchhoffCIG.demo(engine="cuda", **SMALL, nh=8, hmax=400.0)
    d = op.demo_data()
    x = np.random.default_rng(0).standard_normal(op.shape_model, dtype=np.float32)
    ref_f, ref_a = op.forward(x), op.adjoint(d)
    op._eng.set_time_chunk(64)
    assert _rel(op.forward(x), ref_f) < 1e-6
    op._eng.split = 3
    assert _rel(op.adjoint(d), ref_a) < 1e-6


@needs_cuda
def test_cuda_cupy_in_cupy_out():
    import cupy as cp
    op = KirchhoffCIG.demo(engine="cuda", **SMALL, nh=4, hmax=400.0)
    d = cp.asarray(op.demo_data())
    cig = op.adjoint(d)
    assert isinstance(cig, cp.ndarray) and cig.shape == op.shape_model
    back = op.forward(cig)
    assert isinstance(back, cp.ndarray) and back.shape == op.shape_data


@needs_cuda
def test_cuda_float32_accumulation():
    op = KirchhoffCIG.demo(engine="cuda", **SMALL, nh=8, hmax=400.0, acc="float32")
    assert op.dot_test()          # default tolerance is 1e-3 for float32


@needs_cuda
@pytest.mark.parametrize("kw", [{}, dict(nh=1), dict(domain="angle", nh=12, hmax=50.0)])
def test_cuda_aa_dot_test(kw):
    op = KirchhoffCIG.demo(engine="cuda", aa=True, aa_max=64, **kw)
    assert op.aa_widths().max() > 1, "geometry must actually trigger filtering"
    ok, err = op.dot_test(return_error=True)
    assert ok, err


@needs_cuda
@pytest.mark.parametrize("kw", [{}, dict(domain="angle", nh=12, hmax=50.0)])
def test_cuda_aa_matches_numpy(kw):
    """Same taps, same width expression: the two engines agree to float32
    rounding of the result (float64 cumulative sums are not bit-identical
    between a sequential and a parallel scan, but that is at 1e-16 of D)."""
    op_c = KirchhoffCIG.demo(engine="cuda", aa=True, aa_max=64, **kw)
    op_n = op_c._clone(engine="numpy")
    d = op_c.demo_data()
    assert _rel(op_c.adjoint(d), op_n.adjoint(d)) < 1e-5
    x = np.random.default_rng(0).standard_normal(op_c.shape_model, dtype=np.float32)
    assert _rel(op_c.forward(x), op_n.forward(x)) < 1e-5


@needs_cuda
def test_cuda_aa_time_chunking_and_source_split_are_exact():
    """The six forward taps of one contribution can straddle a window edge and
    the adjoint's float64 D buffer is shared by all splits."""
    op = KirchhoffCIG.demo(engine="cuda", aa=True, aa_max=64)
    d = op.demo_data()
    x = np.random.default_rng(0).standard_normal(op.shape_model, dtype=np.float32)
    ref_f, ref_a = op.forward(x), op.adjoint(d)
    op._eng.set_time_chunk(50)                        # much smaller than 2*aa_max
    assert op._eng.nchunk > 5
    assert _rel(op.forward(x), ref_f) < 1e-6
    op._eng.split = 3
    assert _rel(op.adjoint(d), ref_a) < 1e-6


@needs_cuda
def test_cuda_aa_float32_accumulation():
    """The forward's trace accumulator stays float64 under anti-aliasing (it is
    integrated twice afterwards), so acc='float32' costs no more accuracy with
    the filter than without it."""
    op = KirchhoffCIG.demo(engine="cuda", aa=True, acc="float32")
    ok, err = op.dot_test(return_error=True)
    assert ok and err < 1e-5, err
    x = np.random.default_rng(0).standard_normal(op.shape_model, dtype=np.float32)
    assert _rel(op.forward(x), op._clone(engine="numpy").forward(x)) < 1e-6


@needs_cuda
@pytest.mark.parametrize("kw", [dict(aperture=45.0), dict(apt=300.0, aa=True),
                                dict(aperture=50.0, domain="angle", nh=12, hmax=50.0)])
def test_cuda_aperture_matches_numpy(kw):
    op_c = KirchhoffCIG.demo(engine="cuda", **SMALL, **kw)
    op_n = op_c._clone(engine="numpy")
    d = op_c.demo_data()
    assert _rel(op_c.adjoint(d), op_n.adjoint(d)) < 1e-5
    x = np.random.default_rng(0).standard_normal(op_c.shape_model, dtype=np.float32)
    assert _rel(op_c.forward(x), op_n.forward(x)) < 1e-5
    assert op_c.dot_test()


@needs_cuda
def test_cuda_halfderiv_matches_numpy():
    op_c = KirchhoffCIG.demo(engine="cuda", **SMALL, nh=4, hmax=400.0, halfderiv=True)
    op_n = op_c._clone(engine="numpy")
    d = op_c.demo_data()
    assert _rel(op_c.adjoint(d), op_n.adjoint(d)) < 1e-5
    x = np.random.default_rng(0).standard_normal(op_c.shape_model, dtype=np.float32)
    assert _rel(op_c.forward(x), op_n.forward(x)) < 1e-5
    ok, err = op_c.dot_test(return_error=True)
    assert ok and err < 1e-6, err


@needs_cuda
def test_cuda_aa_output_is_plain_float32_trace():
    op = KirchhoffCIG.demo(engine="cuda", aa=True, **SMALL)
    x = np.random.default_rng(0).standard_normal(op.shape_model, dtype=np.float32)
    d = op.forward(x)
    assert d.shape == op.shape_data and d.dtype == np.float32
    assert np.asarray(d).flags.c_contiguous


# --------------------------------------------------------------------- torch
def _torch():
    return pytest.importorskip("torch")


def test_torch_forward_grad_is_adjoint():
    torch = _torch()
    from kirchcig.torch import TorchKirchhoffCIG
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=4, hmax=400.0)
    top = TorchKirchhoffCIG(op)
    data = torch.from_numpy(op.demo_data())
    cig = torch.zeros(op.shape_model, requires_grad=True)
    res = top.forward(cig) - data
    (0.5 * res.pow(2).sum()).backward()
    expected = -op.adjoint(op.demo_data())
    assert _rel(cig.grad.numpy(), expected) < 1e-5


def test_torch_adjoint_grad_is_forward():
    torch = _torch()
    from kirchcig.torch import TorchKirchhoffCIG
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=4, hmax=400.0)
    top = TorchKirchhoffCIG(op)
    data = torch.from_numpy(op.demo_data()).requires_grad_(True)
    w = torch.randn(op.shape_model)
    (top.adjoint(data) * w).sum().backward()
    assert _rel(data.grad.numpy(), op.forward(w.numpy())) < 1e-5


def test_torch_double_backward():
    torch = _torch()
    from kirchcig.torch import TorchKirchhoffCIG
    op = KirchhoffCIG.demo(engine="numpy", **SMALL, nh=2, hmax=400.0)
    top = TorchKirchhoffCIG(op)
    cig = torch.randn(op.shape_model, requires_grad=True)
    data = torch.from_numpy(op.demo_data())
    loss = 0.5 * (top.forward(cig) - data).pow(2).sum()
    g, = torch.autograd.grad(loss, cig, create_graph=True)
    v = torch.randn(op.shape_model)
    hv, = torch.autograd.grad((g * v).sum(), cig)       # Hessian-vector product A^T A v
    expected = op.adjoint(op.forward(v.numpy()))
    assert _rel(hv.numpy(), expected) < 1e-4


# --------------------------------------------------------------- anti-alias
def test_trace_dips_match_analytic_gradient():
    """|dT/d(trace index)| from the tables vs the constant-velocity gradient."""
    from kirchcig._traveltime import analytic_traveltime, trace_dips
    v, nx, nz, dx, dz, spacing = 2000.0, 41, 21, 25.0, 25.0, 50.0
    recs = np.stack([np.arange(21) * spacing, np.zeros(21)])
    dip = trace_dips(analytic_traveltime(recs, v, nx, nz, dx, dz), recs) * spacing
    x, z = np.arange(nx) * dx, np.arange(nz) * dz
    r = 10
    rad = np.hypot(x[:, None] - recs[0, r], z[None, :] - recs[1, r])
    exact = np.abs((recs[0, r] - x[:, None]) / (v * np.maximum(rad, 1e-9))) * spacing
    m = (rad > 100.0) & (exact > 1e-9)          # exact == 0 right below the receiver
    assert np.median(np.abs(dip[r][m] - exact[m]) / exact[m]) < 0.01
    assert dip.max() <= spacing / v * 1.001          # horizontal emergence is the bound


def test_aa_width_one_is_the_unfiltered_operator():
    """n == 1 is the identity triangle, so the filtered path must reproduce the
    unfiltered one. Not bit for bit: the double integration recovers each
    sample only to float64 cancellation, which is still far below float32."""
    plain = KirchhoffCIG.demo(engine="numpy")
    filt = KirchhoffCIG.demo(engine="numpy", aa=True, aa_factor=1e-9)
    assert filt.aa and filt.aa_widths().max() == 1
    d = plain.demo_data()
    assert _rel(filt.adjoint(d), plain.adjoint(d)) < 1e-6
    m = np.random.default_rng(0).standard_normal(plain.shape_model, dtype=np.float32)
    assert _rel(filt.forward(m), plain.forward(m)) < 1e-6


@pytest.mark.parametrize("kw", [{}, dict(nh=1), dict(domain="angle", nh=12, hmax=50.0)])
def test_aa_dot_test(kw):
    op = KirchhoffCIG.demo(engine="numpy", aa=True, aa_max=64, **kw)
    assert op.aa_widths().max() > 1, "geometry must actually trigger filtering"
    assert op.dot_test()


def test_demo_forwards_unknown_keywords():
    """demo() used to drop keywords it did not recognise, which silently
    disabled aa= in tests."""
    assert KirchhoffCIG.demo(engine="numpy", aa=True).aa
    assert KirchhoffCIG.demo(engine="numpy", acc="float32").acc == "float32"


def test_aa_sums_source_and_receiver_dips():
    """Lumley, Claerbout and Bevc (1994) eq. 4 and sfmig2's tx expression add
    the two sides; taking the larger of them would under-filter."""
    from kirchcig._engine_numpy import aa_width
    n_both = aa_width(np.float32(5e-4), np.float32(5e-4), np.float32(2e4), 999)
    n_one = aa_width(np.float32(5e-4), np.float32(0.0), np.float32(2e4), 999)
    assert int(n_both) - 1 == 2 * (int(n_one) - 1)


def test_aa_suppresses_operator_aliasing():
    """A band-limited spike in every trace of one shot maps to a migration
    ellipse per trace. Coarse trace spacing leaves discrete arcs instead of a
    smooth wavefront; the triangle filter has to bring the lateral roughness
    back down to the densely sampled reference."""
    from kirchcig._operator import ricker
    v, nx, nz, dx, dz, dt, nt, f0 = 2000.0, 201, 61, 10.0, 10.0, 0.002, 500, 30.0
    srcs = np.stack([[1000.0], [0.0]])

    def image(nrec, **extra):
        recs = np.stack([np.linspace(0.0, 2000.0, nrec), np.zeros(nrec)])
        d = np.broadcast_to(ricker(np.arange(nt) * dt - 0.5, f0),
                            (1, nrec, nt)).astype(np.float32)
        op = KirchhoffCIG(nx=nx, nz=nz, dx=dx, dz=dz, srcs=srcs, recs=recs, nt=nt,
                          dt=dt, vel=v, nh=1, engine="numpy", aa_max=64, **extra)
        return np.asarray(op.adjoint(d)).sum(0) / nrec

    def roughness(img):
        m = np.abs(img) > 0.05 * np.abs(img).max()
        d2 = np.zeros_like(img)
        d2[1:-1] = img[2:] - 2 * img[1:-1] + img[:-2]
        return np.sqrt((d2[m] ** 2).sum() / (img[m] ** 2).sum())

    ref = roughness(image(201))                       # 10 m spacing, unaliased
    aliased = roughness(image(21))                    # 100 m spacing
    filtered = roughness(image(21, aa=True))
    print(f"ref {ref:.3f}  aliased {aliased:.3f}  filtered {filtered:.3f}")
    assert aliased > 2.0 * ref
    assert filtered < 1.25 * ref
