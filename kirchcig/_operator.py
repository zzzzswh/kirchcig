"""The ``KirchhoffCIG`` operator and the one-shot ``migrate`` function."""
from __future__ import annotations

import math
import warnings

import numpy as np

from ._engine_numpy import NumpyEngine
from ._halfderiv import HalfDerivative
from ._traveltime import (analytic_angle, emergence_angles, image_axes,
                          table_gradients, trace_dips, trace_spacing,
                          traveltime_tables)

_DOMAINS = ("offset", "angle")
_ENGINES = ("cuda", "numpy")


# --------------------------------------------------------------------- helpers
def _is_cupy(x) -> bool:
    mod = type(x).__module__
    return bool(mod) and mod.split(".")[0] == "cupy"


def _positions(p, name):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"{name} must be a (2, n) array of (x, z) positions")
    if p.shape[0] != 2:
        if p.shape[1] == 2:
            p = p.T
        else:
            raise ValueError(f"{name} must be a (2, n) array of (x, z) positions, got {p.shape}")
    return np.ascontiguousarray(p)


def _check_ordered(p, name):
    """Anti-alias dips are differences between neighbouring table entries, so
    the trace axis has to run along the line. Warn rather than raise: an
    unordered axis still migrates correctly, it just over-filters."""
    x = p[0]
    if x.size > 2 and not (np.all(np.diff(x) >= 0) or np.all(np.diff(x) <= 0)):
        warnings.warn(
            f"{name} are not sorted along x; anti-alias filtering assumes "
            f"neighbouring indices are neighbouring positions and will "
            f"over-filter. Sort {name} (and the matching data axis) first.",
            stacklevel=3)


def _is_table_pair(weight):
    """``(w_srcs, w_recs)`` arrays, as opposed to preset names or a callable."""
    return (isinstance(weight, (tuple, list)) and len(weight) == 2
            and not isinstance(weight[0], str) and not callable(weight[0])
            and np.ndim(weight[0]) == 3)


def _resolve_engine(engine):
    if engine == "auto":
        from ._engine_cuda import cuda_available
        return "cuda" if cuda_available() else "numpy"
    if engine not in _ENGINES:
        raise ValueError(f"engine must be one of {_ENGINES + ('auto',)}, got {engine!r}")
    return engine


def _sanitize_tables(trav, t_big):
    """float32 tables with non-finite / out-of-range entries pushed beyond the
    trace, where the kernels ignore them (never lets ``int(floor(t))``
    overflow)."""
    trav = np.array(trav, dtype=np.float32, copy=True)
    trav[~np.isfinite(trav)] = t_big
    np.clip(trav, 0.0, t_big, out=trav)
    return trav


_WEIGHTS = {
    # per-side factors; the operator multiplies source-side by receiver-side
    "obliquity": lambda t, theta, dt: np.sqrt(np.maximum(np.cos(theta), 0.0)),
    "spreading": lambda t, theta, dt: 1.0 / np.sqrt(np.maximum(t, dt)),
}


def weight_tables(spec, trav, theta, dt):
    """Per-side amplitude weight table for one side, ``(n, nx, nz)`` float32.

    ``spec`` is a preset name, a list of preset names (multiplied), or a
    callable ``f(t, theta, dt) -> array`` of the traveltime table [s] and the
    emergence angle from the vertical [rad]. Presets:

    ``"obliquity"``
        ``sqrt(cos theta)``; the product of the two sides is the geometric mean
        of the source and receiver cosines, the obliquity factor of Kirchhoff
        modelling (``sfkirmod`` uses the arithmetic mean ``(cos_s + cos_r)/2``;
        the two agree to second order in the angle difference).
    ``"spreading"``
        ``1 / sqrt(max(t, dt))``; the 2D Green's function amplitude of one leg
        in a homogeneous medium, ``1 / sqrt(v t)`` up to the constant. Product
        of the sides: the geometrical spreading of the reflection.
    """
    if callable(spec):
        w = spec(trav, theta, dt)
    else:
        names = [spec] if isinstance(spec, str) else list(spec)
        w = np.ones(trav.shape, dtype=np.float64)
        for name in names:
            if name not in _WEIGHTS:
                raise ValueError(f"unknown weight {name!r}; presets are {sorted(_WEIGHTS)}")
            w = w * _WEIGHTS[name](trav, theta, dt)
    w = np.asarray(w, dtype=np.float32)
    if w.shape != trav.shape:
        raise ValueError(f"weight must have the table shape {trav.shape}, got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    return np.ascontiguousarray(w)


def aperture_mask(pos, x, z, aperture=None, apt=None):
    """``(n, nx, nz)`` bool: image points inside the migration aperture of each
    of the ``n`` positions in ``pos`` (rows ``(x, z)``).

    ``aperture`` is the largest angle from the vertical, in degrees, of the
    straight line from the position down to the image point (a cone opening
    downwards, ``sfkirmig``'s ``aperture=``); ``apt`` is the largest lateral
    distance in metres (``sfmig2``'s ``apt=``, in metres rather than traces).
    Either may be ``None``. Points at or above the position are outside any
    cone narrower than 90 degrees.
    """
    pos = np.asarray(pos, dtype=np.float64)
    dxs = x[None, :, None] - pos[0][:, None, None]        # (n, nx, 1)
    dzs = z[None, None, :] - pos[1][:, None, None]        # (n, 1, nz)
    mask = np.ones((pos.shape[1], x.size, z.size), dtype=bool)
    if aperture is not None:
        # tan(45 deg) is 1 - 1e-16 in floating point; the tiny slack keeps
        # points exactly on the cone edge inside, as the definition says.
        tap = math.tan(math.radians(aperture)) + 1e-9
        mask &= (dzs >= 0) & (np.abs(dxs) <= tap * dzs)
    if apt is not None:
        mask &= np.abs(dxs) <= apt
    return mask


def ricker(t, f0):
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


_DEMO = dict(v=2000.0, nx=101, nz=61, dx=10.0, dz=10.0, ns=8, nr=41, nt=401, dt=0.002,
             nh=8, hmax=500.0, domain="offset", scatterer=(500.0, 300.0), f0=25.0)


# -------------------------------------------------------------------- operator
class KirchhoffCIG:
    """Kirchhoff migration to common-image gathers and its exact adjoint.

    ``adjoint`` maps prestack data ``(ns, nr, nt)`` to gathers ``(nh, nx, nz)``
    (migration); ``forward`` maps gathers back to data (demigration). The two
    are exact transposes of each other to accumulator precision.

    Parameters
    ----------
    nx, nz, dx, dz : image grid, ``x = ox + ix*dx``, ``z = oz + iz*dz`` [m]
    srcs, recs : (2, ns), (2, nr) positions, rows are (x, z) [m]
    nt, dt : trace length [samples] and sampling [s]
    vel : float or (nx, nz) array, migration velocity [m/s]. A constant
        velocity uses analytic traveltimes, otherwise an eikonal solve
        (scikit-fmm). Not needed when ``trav`` is given.
    nh, hmax : number of gather bins and the axis extent. Offset domain:
        absolute half-offset in metres (default: the largest half-offset
        present, so every trace is used). Angle domain: maximum half opening
        angle in degrees (default 90).
    domain : "offset" or "angle"
    engine : "cuda", "numpy" or "auto" (cuda when available)
    trav : optional ``(trav_srcs, trav_recs)`` tables of shape ``(ns, nx, nz)``
        and ``(nr, nx, nz)`` [s]
    ox, oz : origin of the image grid [m]
    acc : accumulator precision of the cuda engine, "float64" or "float32"
    block, split : cuda engine tuning knobs (see ``_engine_cuda``)
    eikonal : dict of keyword arguments for the eikonal solver
    aa : anti-alias filtering. Each contribution is read through a triangle
        filter whose half-width follows the local operator dip, which
        suppresses the steeply dipping artefacts the summation would otherwise
        alias in. Both engines support it. The filter works on the float64
        double integral of the traces, so it needs a float64 copy of the data
        (``ns * nr * (nt + 2*aa_max + 1) * 8`` bytes) and reads three taps per
        contribution instead of one; expect the adjoint to be a few times
        slower than without it.
    aa_factor : dimensionless multiplier on the filter width, matching the
        ``antialias`` parameter of Claerbout's ``trimo`` and Madagascar's
        ``sfmig2``, both of which default to 1.0. At 2.0 the triangle's first
        spectral null sits exactly at the alias frequency, which is the
        criterion of Lumley, Claerbout and Bevc (1994) eq. 12 and what
        Claerbout used for migration; it costs resolution on steep dips.
    aa_max : largest triangle half-width in samples (default 32)
    aa_stretch : with ``aa=True`` (default on), also cover the time footprint
        of the image cell, ``|dtau/dx| dx + |dtau/dz| dz``, so that a depth
        grid coarser than the time sampling (``2 dz / v > dt``) demigrates
        into a smooth trace instead of a comb of interpolated spikes. This is
        Madagascar's ``aastretch``; the operator-dip and cell widths combine
        as a root mean square. Reads one extra 8-byte gradient table entry per
        contribution. Off, the filter is the pure trace-axis criterion of
        ``sfmig2``.
    aperture : migration aperture as the largest angle from the vertical, in
        degrees, between an image point and the source *and* the receiver of
        a trace (``sfkirmig``'s ``aperture=``). Contributions outside the cone
        are dropped, which suppresses far-aperture swing noise and skips their
        work. ``None`` (default) or 90 means no limit.
    apt : lateral aperture, the largest ``|x - x_src|`` and ``|x - x_rec|`` in
        metres (``sfmig2``'s ``apt=``, in metres). ``None`` means no limit.
        Both limits may be combined; a contribution needs to pass both.

    weight : amplitude weights, ``None`` (default, unit weights) or a
        per-side factor: every contribution is multiplied by
        ``w_s(src, x, z) * w_r(rec, x, z)``. Give a preset name
        (``"obliquity"``: ``sqrt(cos theta)`` per side, ``"spreading"``:
        ``1 / sqrt(t)`` per side, see :func:`weight_tables`), a list of presets
        to multiply, a callable ``f(t, theta, dt)`` evaluated on each side's
        traveltime and emergence-angle tables, or a pair of ready tables
        ``(w_srcs, w_recs)`` of shapes ``(ns, nx, nz)`` and ``(nr, nx, nz)``.
        Both kernels multiply by the same float32 product, so the pair stays
        an exact transpose: ``forward`` is ``A W``, ``adjoint`` is ``W A^T``.
        These are not the true-amplitude (Bleistein) weights, which are not
        separable; they cover the obliquity and spreading factors that
        Kirchhoff modelling codes such as ``sfkirmod`` apply.
    halfderiv : apply the half-order time derivative that 2D Kirchhoff
        demigration needs (the "rho filter", Madagascar's ``sf_halfint``):
        ``forward`` filters the demigrated traces with
        ``H(w) = sqrt(1 - rho e^{-iw})``, ``adjoint`` filters the data with the
        exact transpose ``H*`` before summing. Without it the migrated wavelet
        is rotated by 45 degrees and weighted towards low frequencies. Cheap
        (one FFT per trace, float64) and independent of the engine.
    halfderiv_rho : leak of the half difference, default ``1 - 1/nt``.

    Both apertures are applied by pushing the masked traveltime-table entries
    beyond the trace end, where the kernels already skip them, so the two
    engines drop exactly the same contributions and the pair stays an exact
    transpose. ``trav_srcs`` / ``trav_recs`` stay unmasked; see
    :meth:`aperture_masks`.
    """

    def __init__(self, nx, nz, dx, dz, srcs, recs, nt, dt, vel=None, nh=1, hmax=None,
                 domain="offset", engine="auto", trav=None, ox=0.0, oz=0.0,
                 acc="float64", block=None, split="auto", eikonal=None,
                 aa=False, aa_factor=1.0, aa_max=32, aa_stretch=True, aperture=None, apt=None,
                 halfderiv=False, halfderiv_rho=None, weight=None, _tables=None):
        self.nx, self.nz = int(nx), int(nz)
        self.dx, self.dz = float(dx), float(dz)
        self.ox, self.oz = float(ox), float(oz)
        self.srcs = _positions(srcs, "srcs")
        self.recs = _positions(recs, "recs")
        self.ns, self.nr = self.srcs.shape[1], self.recs.shape[1]
        self.nt, self.dt = int(nt), float(dt)
        self.nh = int(nh)
        self.domain = domain
        self.acc = acc
        self.aa = bool(aa)
        self.aa_factor = float(aa_factor)
        self.aa_max = int(aa_max)
        self.aa_stretch = bool(aa_stretch) and self.aa
        if self.aa and (self.aa_factor <= 0 or self.aa_max < 1):
            raise ValueError("need aa_factor > 0 and aa_max >= 1")
        if aperture is not None:
            aperture = float(aperture)
            if aperture <= 0:
                raise ValueError("aperture must be in (0, 90] degrees")
            if aperture >= 90.0:
                aperture = None                      # a 90-degree cone is no limit
        if apt is not None and float(apt) <= 0:
            raise ValueError("apt must be positive")
        self.aperture = aperture
        self.apt = None if apt is None else float(apt)
        self.halfderiv = bool(halfderiv)
        self.halfderiv_rho = halfderiv_rho
        self.weight = weight
        self.vel = None if vel is None else (float(vel) if np.ndim(vel) == 0 else np.asarray(vel))
        if self.nh < 1:
            raise ValueError("nh must be >= 1")
        if self.nt < 2 or self.dt <= 0:
            raise ValueError("need nt >= 2 and dt > 0")
        if domain not in _DOMAINS:
            raise ValueError(f"domain must be one of {_DOMAINS}, got {domain!r}")
        if acc not in ("float64", "float32"):
            raise ValueError("acc must be 'float64' or 'float32'")

        shp_s, shp_r = (self.ns, self.nx, self.nz), (self.nr, self.nx, self.nz)
        t_big = 10.0 * self.nt * self.dt
        ang_s = ang_r = None

        # -- traveltime (and angle) tables ------------------------------------
        if _tables is not None:                      # internal: share tables
            trav_s, trav_r = _tables["trav_s"], _tables["trav_r"]
            ang_s, ang_r = _tables.get("ang_s"), _tables.get("ang_r")
            const_vel = None
        elif trav is not None:
            trav_s, trav_r = trav
            trav_s, trav_r = np.asarray(trav_s), np.asarray(trav_r)
            if trav_s.shape != shp_s or trav_r.shape != shp_r:
                raise ValueError(
                    f"trav tables must have shapes {shp_s} and {shp_r} (source/receiver axis "
                    f"first), got {trav_s.shape} and {trav_r.shape}")
            const_vel = None
        else:
            if vel is None:
                raise ValueError("either vel or trav must be given")
            trav_s, trav_r, const_vel = traveltime_tables(
                vel, self.srcs, self.recs, self.nx, self.nz, self.dx, self.dz,
                self.ox, self.oz, **(eikonal or {}))
        self._trav_s = _sanitize_tables(trav_s, t_big)
        self._trav_r = _sanitize_tables(trav_r, t_big)

        need_angles = domain == "angle" or (weight is not None and not _is_table_pair(weight))
        if need_angles and ang_s is None:
            if const_vel is not None:
                ang_s = analytic_angle(self.srcs, self.nx, self.nz, self.dx, self.dz, self.ox, self.oz)
                ang_r = analytic_angle(self.recs, self.nx, self.nz, self.dx, self.dz, self.ox, self.oz)
            else:
                ang_s = emergence_angles(self._trav_s, self.dx, self.dz)
                ang_r = emergence_angles(self._trav_r, self.dx, self.dz)
        self._ang_s = None if ang_s is None else np.ascontiguousarray(ang_s, dtype=np.float32)
        self._ang_r = None if ang_r is None else np.ascontiguousarray(ang_r, dtype=np.float32)

        # -- operator dip tables (anti-alias) -----------------------------------
        dip_s = dip_r = None
        if self.aa:
            if _tables is not None and _tables.get("dip_s") is not None:
                dip_s, dip_r = _tables["dip_s"], _tables["dip_r"]
            else:
                _check_ordered(self.srcs, "srcs")
                _check_ordered(self.recs, "recs")
                dip_s = trace_dips(self._trav_s, self.srcs)
                dip_r = trace_dips(self._trav_r, self.recs)
        self._dip_s = None if dip_s is None else np.ascontiguousarray(dip_s, dtype=np.float32)
        self._dip_r = None if dip_r is None else np.ascontiguousarray(dip_r, dtype=np.float32)
        grd_s = grd_r = None
        if self.aa_stretch:
            if _tables is not None and _tables.get("grd_s") is not None:
                grd_s, grd_r = _tables["grd_s"], _tables["grd_r"]
            else:
                grd_s = table_gradients(self._trav_s, self.dx, self.dz)
                grd_r = table_gradients(self._trav_r, self.dx, self.dz)
        self._grd_s = None if grd_s is None else np.ascontiguousarray(grd_s, dtype=np.float32)
        self._grd_r = None if grd_r is None else np.ascontiguousarray(grd_r, dtype=np.float32)

        # -- amplitude weight tables --------------------------------------------
        w_s = w_r = None
        if weight is not None:
            if _is_table_pair(weight):
                w_s, w_r = (np.asarray(weight[0], dtype=np.float32), np.asarray(weight[1], dtype=np.float32))
                if w_s.shape != shp_s or w_r.shape != shp_r:
                    raise ValueError(f"weight tables must have shapes {shp_s} and {shp_r}, "
                                     f"got {w_s.shape} and {w_r.shape}")
                if not (np.all(np.isfinite(w_s)) and np.all(np.isfinite(w_r))):
                    raise ValueError("weights must be finite")
            else:
                w_s = weight_tables(weight, self._trav_s, self._ang_s, self.dt)
                w_r = weight_tables(weight, self._trav_r, self._ang_r, self.dt)
        self._w_s = None if w_s is None else np.ascontiguousarray(w_s)
        self._w_r = None if w_r is None else np.ascontiguousarray(w_r)

        # -- gather binning -----------------------------------------------------
        if domain == "offset":
            hoff = 0.5 * np.abs(self.srcs[0][:, None] - self.recs[0][None, :])   # (ns, nr)
            if hmax is None:
                hmax = float(hoff.max()) if hoff.max() > 0 else float(self.dx)
            self.hmax = float(hmax)
            if self.hmax <= 0:
                raise ValueError("hmax must be positive")
            dh = self.hmax / self.nh
            hb = np.floor(hoff / dh).astype(np.int32)
            hb[hb >= self.nh] = self.nh - 1          # closed last bin
            hb[hoff > self.hmax] = -1                # beyond the axis: trace unused
            self._hmax_rad = 0.0
            self._ihd = 0.0
        else:
            self.hmax = 90.0 if hmax is None else float(hmax)
            if self.hmax <= 0:
                raise ValueError("hmax must be positive")
            hb = np.zeros((self.ns, self.nr), dtype=np.int32)
            self._hmax_rad = math.radians(self.hmax)
            self._ihd = self.nh / self._hmax_rad
        self._hbin = np.ascontiguousarray(hb)

        # -- aperture: mask by pushing table entries past the trace end ---------
        tabs_t, tabr_t = self._trav_s, self._trav_r
        if self.aperture is not None or self.apt is not None:
            ms, mr = self.aperture_masks()
            tabs_t = np.where(ms, tabs_t, np.float32(t_big))
            tabr_t = np.where(mr, tabr_t, np.float32(t_big))

        # -- engine --------------------------------------------------------------
        self.engine = _resolve_engine(engine)
        npts = self.nx * self.nz
        kw = dict(
            tabs_t=tabs_t.reshape(self.ns, npts),
            tabr_t=tabr_t.reshape(self.nr, npts),
            hbin=self._hbin, nh=self.nh, nt=self.nt, idt=1.0 / self.dt,
            angle=(domain == "angle"),
            tabs_a=None if self._ang_s is None else self._ang_s.reshape(self.ns, npts),
            tabr_a=None if self._ang_r is None else self._ang_r.reshape(self.nr, npts),
            ihd=self._ihd, hmax_rad=self._hmax_rad,
        )
        if self.aa:
            kw.update(
                aa=True,
                tabs_d=self._dip_s.reshape(self.ns, npts),
                tabr_d=self._dip_r.reshape(self.nr, npts),
                aaf=self._aa_scale(), aa_factor=self.aa_factor, aa_max=self.aa_max,
            )
            if self.aa_stretch:
                kw.update(
                    tabs_g=self._grd_s.reshape(self.ns, npts, 2),
                    tabr_g=self._grd_r.reshape(self.nr, npts, 2),
                    dxdt=self.dx / self.dt, dzdt=self.dz / self.dt,
                )
        if self._w_s is not None:
            kw.update(tabs_w=self._w_s.reshape(self.ns, npts),
                      tabr_w=self._w_r.reshape(self.nr, npts))
        if self.engine == "cuda":
            from ._engine_cuda import CudaEngine
            self._eng = CudaEngine(acc=acc, block=block, split=split, **kw)
        else:
            self._eng = NumpyEngine(**kw)

        # -- half-derivative filter (data side, engine independent) -------------
        self._hd = None
        if self.halfderiv:
            if self.engine == "cuda":
                import cupy as xp
            else:
                xp = np
            self._hd = HalfDerivative(self.nt, rho=halfderiv_rho, xp=xp)

    # --------------------------------------------------------------- shapes
    @property
    def shape_model(self):
        return (self.nh, self.nx, self.nz)

    @property
    def shape_data(self):
        return (self.ns, self.nr, self.nt)

    @property
    def size_model(self):
        return self.nh * self.nx * self.nz

    @property
    def size_data(self):
        return self.ns * self.nr * self.nt

    @property
    def x(self):
        return image_axes(self.nx, self.nz, self.dx, self.dz, self.ox, self.oz)[0]

    @property
    def z(self):
        return image_axes(self.nx, self.nz, self.dx, self.dz, self.ox, self.oz)[1]

    @property
    def h_axis(self):
        """Bin centres of the gather axis: metres (offset) or degrees (angle)."""
        dh = self.hmax / self.nh
        return (np.arange(self.nh) + 0.5) * dh

    def _aa_scale(self):
        """(ns, nr) float32 ``drho / dt``: the effective trace spacing of each
        source/receiver pair in samples per unit operator dip.

        ``drho`` is the root-mean-square of the two axis spacings (Lumley,
        Claerbout and Bevc 1994, eq. 5). Equal source and receiver spacings
        give ``drho = dx``; an axis with a single trace does not contribute
        and does not count towards the mean. With a single source *and* a
        single receiver the trace-axis term is zero (only the cell term of
        ``aa_stretch`` remains)."""
        ds = trace_spacing(self.srcs)[:, None]
        dr = trace_spacing(self.recs)[None, :]
        k = (self.ns > 1) + (self.nr > 1)
        if k == 0:
            if not self.aa_stretch:
                raise ValueError("anti-aliasing needs at least two sources or two receivers")
            rho = np.zeros((1, 1))
        else:
            rho = np.sqrt((ds ** 2 + dr ** 2) / k) / self.dt
        return np.ascontiguousarray(np.broadcast_to(rho, (self.ns, self.nr)),
                                    dtype=np.float32)

    def aa_widths(self):
        """(ns, nr, nx, nz) int32 triangle half-widths in samples, or None when
        anti-aliasing is off. Diagnostic only; the engines compute these on the
        fly. Memory: ``ns * nr * nx * nz * 4`` bytes."""
        if not self.aa:
            return None
        from ._engine_numpy import aa_cell, aa_width
        dip = self._dip_s[:, None] + self._dip_r[None, :]
        cell = np.float32(0.0)
        if self.aa_stretch:
            cell = aa_cell(self._grd_s[:, None], self._grd_r[None, :],
                           self.dx / self.dt, self.dz / self.dt)
        return aa_width(dip, self._aa_scale()[:, :, None, None], cell,
                        self.aa_factor, self.aa_max)

    def aperture_masks(self):
        """``(ns, nx, nz)`` and ``(nr, nx, nz)`` bool arrays of the image points
        inside the aperture of every source and receiver (all True without an
        aperture). A trace contributes to an image point when both are True."""
        x, z = image_axes(self.nx, self.nz, self.dx, self.dz, self.ox, self.oz)
        return (aperture_mask(self.srcs, x, z, self.aperture, self.apt),
                aperture_mask(self.recs, x, z, self.aperture, self.apt))

    @property
    def weights(self):
        """``(w_srcs, w_recs)`` per-side amplitude weight tables, or ``None``."""
        return None if self._w_s is None else (self._w_s, self._w_r)

    @property
    def trav_srcs(self):
        return self._trav_s

    @property
    def trav_recs(self):
        return self._trav_r

    @property
    def offset_bins(self):
        """(ns, nr) int32 offset bin of every trace, -1 where unused."""
        return self._hbin

    # ----------------------------------------------------------- conversions
    def _to_engine(self, x, shape, name):
        want_cupy = _is_cupy(x)
        if self.engine == "cuda":
            import cupy as cp
            arr = cp.ascontiguousarray(cp.asarray(x, dtype=cp.float32))
        else:
            arr = x.get() if want_cupy else np.asarray(x)
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.shape != shape:
            if arr.size != int(np.prod(shape)):
                raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
            arr = arr.reshape(shape)
        return arr, want_cupy

    def _from_engine(self, arr, want_cupy):
        if want_cupy:
            if _is_cupy(arr):
                return arr
            import cupy as cp
            return cp.asarray(arr)
        return arr.get() if _is_cupy(arr) else arr

    # ------------------------------------------------------------- operator
    def adjoint(self, data):
        """Migration: data ``(ns, nr, nt)`` -> gathers ``(nh, nx, nz)``.

        NumPy in, NumPy out; CuPy in, CuPy out (no host round-trip)."""
        arr, want_cupy = self._to_engine(data, self.shape_data, "data")
        if self._hd is not None:
            arr = self._hd(arr, adj=True)
        out = self._eng.adjoint(arr).reshape(self.shape_model)
        return self._from_engine(out, want_cupy)

    def forward(self, cig):
        """Demigration: gathers ``(nh, nx, nz)`` -> data ``(ns, nr, nt)``."""
        arr, want_cupy = self._to_engine(cig, self.shape_model, "cig")
        out = self._eng.forward(arr.reshape(self.nh, -1)).reshape(self.shape_data)
        if self._hd is not None:
            out = self._hd(out)
        return self._from_engine(out, want_cupy)

    __call__ = forward

    def dot_test(self, tol=None, seed=0, return_error=False):
        """Check ``<A x, y> == <x, A^T y>`` on random vectors.

        Returns ``True``/``False`` (and the relative error when
        ``return_error``). Default tolerance is 1e-5 for float64 accumulation
        and 1e-3 for float32."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.shape_model, dtype=np.float32)
        y = rng.standard_normal(self.shape_data, dtype=np.float32)
        ax = np.asarray(self.forward(x), dtype=np.float64).ravel()
        aty = np.asarray(self.adjoint(y), dtype=np.float64).ravel()
        lhs = float(np.dot(ax, y.ravel().astype(np.float64)))
        rhs = float(np.dot(x.ravel().astype(np.float64), aty))
        err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(np.float64).tiny)
        if tol is None:
            tol = 1e-5 if self.acc == "float64" else 1e-3
        ok = bool(err < tol)
        return (ok, err) if return_error else ok

    def to_scipy(self):
        """Wrap as a ``scipy.sparse.linalg.LinearOperator`` (float32)."""
        import scipy.sparse.linalg as spla
        return spla.LinearOperator(
            shape=(self.size_data, self.size_model),
            matvec=lambda v: np.asarray(self.forward(np.asarray(v, dtype=np.float32)
                                                     .reshape(self.shape_model))).ravel(),
            rmatvec=lambda v: np.asarray(self.adjoint(np.asarray(v, dtype=np.float32)
                                                      .reshape(self.shape_data))).ravel(),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ demo
    @classmethod
    def demo(cls, engine="auto", **overrides):
        """Small constant-velocity, single-point-scatterer operator.

        Keyword overrides accepted for any entry of the demo geometry
        (``nh``, ``hmax``, ``domain``, ``nt``, ...); anything else is passed
        straight to the constructor (``aa``, ``acc``, ...)."""
        p = dict(_DEMO)
        p.update({k: v for k, v in overrides.items() if k in _DEMO})
        extra = {k: v for k, v in overrides.items() if k not in _DEMO}
        srcs = np.stack([np.linspace(100.0, 900.0, p["ns"]), np.zeros(p["ns"])])
        recs = np.stack([np.linspace(0.0, 1000.0, p["nr"]), np.zeros(p["nr"])])
        op = cls(nx=p["nx"], nz=p["nz"], dx=p["dx"], dz=p["dz"], srcs=srcs, recs=recs,
                 nt=p["nt"], dt=p["dt"], vel=p["v"], nh=p["nh"], hmax=p["hmax"],
                 domain=p["domain"], engine=engine, **extra)
        op._demo = dict(v=p["v"], scatterer=tuple(p["scatterer"]), f0=p["f0"])
        return op

    def demo_data(self):
        """Ricker-wavelet point-scatterer data for the demo geometry, computed
        analytically (independent of the operator)."""
        d = self._require_demo()
        xp, zp = d["scatterer"]
        ts = np.hypot(self.srcs[0] - xp, self.srcs[1] - zp) / d["v"]
        tr = np.hypot(self.recs[0] - xp, self.recs[1] - zp) / d["v"]
        t = np.arange(self.nt) * self.dt
        tau = t[None, None, :] - (ts[:, None] + tr[None, :])[..., None]
        data = ricker(tau, d["f0"]) / (self.ns * self.nr)
        return data.astype(np.float32)

    def demo_image(self):
        """Stacked migration of :meth:`demo_data` by the NumPy reference engine,
        so a cuda-engine result can be checked against the oracle."""
        self._require_demo()
        if getattr(self, "_demo_image_cache", None) is None:
            ref = self if self.engine == "numpy" else self._clone(engine="numpy")
            cig = np.asarray(ref.adjoint(self.demo_data()))
            self._demo_image_cache = cig.sum(0)
        return self._demo_image_cache

    def _require_demo(self):
        d = getattr(self, "_demo", None)
        if d is None:
            raise RuntimeError("demo_data()/demo_image() are only available on KirchhoffCIG.demo()")
        return d

    def _clone(self, engine=None, **overrides):
        """Same geometry and tables, possibly a different engine."""
        kw = dict(nx=self.nx, nz=self.nz, dx=self.dx, dz=self.dz, srcs=self.srcs,
                  recs=self.recs, nt=self.nt, dt=self.dt, vel=self.vel, nh=self.nh,
                  hmax=self.hmax, domain=self.domain, ox=self.ox, oz=self.oz, acc=self.acc,
                  aa=self.aa, aa_factor=self.aa_factor, aa_max=self.aa_max,
                  aa_stretch=self.aa_stretch, aperture=self.aperture, apt=self.apt,
                  halfderiv=self.halfderiv, halfderiv_rho=self.halfderiv_rho,
                  weight=self.weight,
                  engine=self.engine if engine is None else engine,
                  _tables=dict(trav_s=self._trav_s, trav_r=self._trav_r,
                               ang_s=self._ang_s, ang_r=self._ang_r,
                               dip_s=self._dip_s, dip_r=self._dip_r,
                               grd_s=self._grd_s, grd_r=self._grd_r))
        kw.update(overrides)
        op = KirchhoffCIG(**kw)
        if hasattr(self, "_demo"):
            op._demo = dict(self._demo)
        return op

    def __repr__(self):
        return (f"KirchhoffCIG(shape_model={self.shape_model}, shape_data={self.shape_data}, "
                f"domain={self.domain!r}, hmax={self.hmax:g}, aa={self.aa}, "
                f"aperture={self.aperture}, apt={self.apt}, halfderiv={self.halfderiv}, "
                f"weight={'custom' if _is_table_pair(self.weight) else self.weight!r}, "
                f"engine={self.engine!r})")


# --------------------------------------------------------------------- migrate
def migrate(data, vel, srcs, recs, dt, dx, dz, nh=1, hmax=None, domain="offset",
            engine="auto", trav=None, ox=0.0, oz=0.0, aa=False, **kwargs):
    """One-shot Kirchhoff migration of prestack data to CIGs.

    Parameters
    ----------
    data : (ns, nr, nt) prestack traces
    vel : (nx, nz) migration velocity [m/s] (or a scalar together with ``nx``,
        ``nz`` keyword arguments or ``trav`` tables)
    srcs, recs : (2, ns), (2, nr) positions, rows (x, z) [m]
    dt, dx, dz : sampling [s], [m], [m]
    nh, hmax, domain, engine, trav, ox, oz : see :class:`KirchhoffCIG`
    aa : anti-alias filtering (``aa_factor`` and ``aa_max`` pass through)
    aperture, apt : migration aperture in degrees / metres (pass through)
    halfderiv : half-order time derivative, the 2D rho filter (pass through)
    weight : amplitude weights, preset name(s) / callable / tables (pass through)

    Returns
    -------
    (nh, nx, nz) gathers; ``.sum(0)`` is the stacked image. NumPy in, NumPy
    out; CuPy in, CuPy out.
    """
    if data.ndim != 3:
        raise ValueError("data must be (ns, nr, nt)")
    nt = data.shape[-1]
    if trav is not None:
        nx, nz = np.asarray(trav[0]).shape[1:]
    elif np.ndim(vel) == 2:
        nx, nz = np.shape(vel)
    else:
        try:
            nx, nz = kwargs.pop("nx"), kwargs.pop("nz")
        except KeyError as exc:
            raise ValueError("a scalar vel needs nx and nz keyword arguments") from exc
    op = KirchhoffCIG(nx=nx, nz=nz, dx=dx, dz=dz, srcs=srcs, recs=recs, nt=nt, dt=dt,
                      vel=vel, nh=nh, hmax=hmax, domain=domain, engine=engine, trav=trav,
                      ox=ox, oz=oz, aa=aa, **kwargs)
    return op.adjoint(data)
