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
