"""Traveltime tables and emergence angles.

Tables are always laid out ``(n, nx, nz)`` with the source/receiver axis first;
this is what keeps the kernel reads coalesced (see README, "Bring your own
traveltimes").
"""
from __future__ import annotations

import numpy as np


def image_axes(nx, nz, dx, dz, ox=0.0, oz=0.0):
    """Return the ``x`` and ``z`` coordinate vectors of the image grid."""
    x = ox + dx * np.arange(nx, dtype=np.float64)
    z = oz + dz * np.arange(nz, dtype=np.float64)
    return x, z


def analytic_traveltime(pos, vel, nx, nz, dx, dz, ox=0.0, oz=0.0):
    """Straight-ray traveltimes for a constant velocity.

    Parameters
    ----------
    pos : (2, n) array, rows (x, z) [m]
    vel : float, velocity [m/s]

    Returns
    -------
    (n, nx, nz) float32 array of traveltimes [s]
    """
    pos = np.asarray(pos, dtype=np.float64)
    x, z = image_axes(nx, nz, dx, dz, ox, oz)
    out = np.empty((pos.shape[1], nx, nz), dtype=np.float32)
    inv_v = 1.0 / float(vel)
    for i in range(pos.shape[1]):
        out[i] = np.hypot(x[:, None] - pos[0, i], z[None, :] - pos[1, i]) * inv_v
    return out


def analytic_angle(pos, nx, nz, dx, dz, ox=0.0, oz=0.0):
    """Exact straight-ray emergence angle from the vertical, in radians.

    Positive angles lean towards +x. Same layout as :func:`analytic_traveltime`.
    """
    pos = np.asarray(pos, dtype=np.float64)
    x, z = image_axes(nx, nz, dx, dz, ox, oz)
    out = np.empty((pos.shape[1], nx, nz), dtype=np.float32)
    for i in range(pos.shape[1]):
        out[i] = np.arctan2(x[:, None] - pos[0, i], z[None, :] - pos[1, i])
    return out


def eikonal_traveltime(pos, vel, dx, dz, ox=0.0, oz=0.0, order=2, src_radius=None):
    """First-arrival traveltimes with the fast marching method (scikit-fmm).

    The point-source singularity is handled the usual way: the FMM is started
    from a small circle of radius ``src_radius`` (default two cells) around the
    source, and the time inside the circle is filled in analytically with the
    velocity at the source. This also allows arbitrary, off-grid source
    positions.

    Parameters
    ----------
    pos : (2, n) array of (x, z) positions [m]
    vel : (nx, nz) array, velocity [m/s]

    Returns
    -------
    (n, nx, nz) float32 array of traveltimes [s]
    """
    try:
        import skfmm
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "kirchcig needs scikit-fmm for eikonal traveltimes in a variable "
            "velocity model (pip install scikit-fmm), or pass precomputed tables "
            "with trav=(trav_srcs, trav_recs)."
        ) from exc

    vel = np.asarray(vel, dtype=np.float64)
    if vel.ndim != 2:
        raise ValueError(f"vel must be 2D (nx, nz), got shape {vel.shape}")
    if not np.all(np.isfinite(vel)) or vel.min() <= 0:
        raise ValueError("vel must be finite and strictly positive")
    nx, nz = vel.shape
    pos = np.asarray(pos, dtype=np.float64)
    x, z = image_axes(nx, nz, dx, dz, ox, oz)
    X, Z = x[:, None], z[None, :]
    h = float(max(dx, dz))
    r0_default = float(src_radius) if src_radius is not None else 2.0 * h

    out = np.empty((pos.shape[1], nx, nz), dtype=np.float32)
    for i in range(pos.shape[1]):
        px, pz = pos[0, i], pos[1, i]
        dist = np.hypot(X - px, Z - pz)
        # The zero contour must cut through the grid, also for sources that sit
        # just outside it (e.g. towed streamer above a grid starting at z > 0).
        r0 = max(r0_default, float(dist.min()) + h)
        ix = int(np.clip(round((px - ox) / dx), 0, nx - 1))
        iz = int(np.clip(round((pz - oz) / dz), 0, nz - 1))
        v0 = vel[ix, iz]
        phi = dist - r0
        try:
            t = skfmm.travel_time(phi, vel, dx=[dx, dz], order=order)
        except Exception:  # narrow grids reject order=2
            t = skfmm.travel_time(phi, vel, dx=[dx, dz], order=1)
        t = np.ma.filled(t, np.nan) if np.ma.isMaskedArray(t) else np.asarray(t, dtype=np.float64)
        t = t + r0 / v0
        inside = phi < 0
        t[inside] = dist[inside] / v0
        out[i] = t
    return out


def emergence_angles(trav, dx, dz):
    """Emergence angle from the vertical, ``atan2(dT/dx, dT/dz)``, in radians.

    Computed from second-order finite differences of the traveltime tables, so
    it works for any table, including user-provided ones.

    Parameters
    ----------
    trav : (n, nx, nz) traveltimes [s]

    Returns
    -------
    (n, nx, nz) float32 array of angles [rad]
    """
    trav = np.asarray(trav)
    if trav.ndim != 3 or trav.shape[1] < 2 or trav.shape[2] < 2:
        raise ValueError("trav must be (n, nx, nz) with nx, nz >= 2")
    out = np.empty(trav.shape, dtype=np.float32)
    for i in range(trav.shape[0]):
        ti = trav[i].astype(np.float64, copy=False)
        gx = np.gradient(ti, dx, axis=0)
        gz = np.gradient(ti, dz, axis=1)
        out[i] = np.arctan2(gx, gz)
    return out


def trace_spacing(pos):
    """Local trace spacing along the acquisition line [m], from a ``(2, n)``
    array of positions. Central differences inside, one-sided at the ends;
    zero for a single trace."""
    p = np.asarray(pos, dtype=np.float64)
    n = p.shape[1]
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out
    d = np.hypot(np.diff(p[0]), np.diff(p[1]))
    out[1:-1] = 0.5 * (d[:-1] + d[1:])
    out[0], out[-1] = d[0], d[-1]
    return out


def trace_dips(trav, pos):
    """Operator dip ``|dT/dx|`` along the trace axis [s/m].

    This is the derivative that enters the Kirchhoff operator anti-aliasing
    criterion (Lumley, Claerbout and Bevc 1994, eq. 1 and 4): the summation
    along the traveltime curve aliases once the moveout between neighbouring
    traces approaches half a period of the highest frequency present. The paper
    recommends exactly this table-differencing route for depth migration
    driven by traveltime tables, in preference to the hyperbolic time-migration
    approximation.

    The trace axis is assumed to be ordered along the acquisition line;
    :class:`~kirchcig.KirchhoffCIG` checks this and warns otherwise.

    Parameters
    ----------
    trav : (n, nx, nz) traveltimes [s]
    pos : (2, n) trace positions, rows (x, z) [m]

    Returns
    -------
    (n, nx, nz) float32 array [s/m], zero for a single trace
    """
    trav = np.asarray(trav, dtype=np.float32)
    if trav.ndim != 3:
        raise ValueError("trav must be (n, nx, nz)")
    out = np.zeros(trav.shape, dtype=np.float32)
    if trav.shape[0] < 2:
        return out
    p = np.asarray(pos, dtype=np.float64)
    d = np.hypot(np.diff(p[0]), np.diff(p[1]))
    d = np.maximum(d, 1e-9).astype(np.float32)
    out[1:-1] = np.abs(trav[2:] - trav[:-2]) / (d[:-1] + d[1:])[:, None, None]
    out[0] = np.abs(trav[1] - trav[0]) / d[0]
    out[-1] = np.abs(trav[-1] - trav[-2]) / d[-1]
    return out


def traveltime_tables(vel, srcs, recs, nx, nz, dx, dz, ox=0.0, oz=0.0, **eikonal_kwargs):
    """Build the source and receiver tables for a velocity model.

    A scalar velocity, or a 2D model that is constant, uses the analytic
    expression; anything else goes through the eikonal solver.

    Returns
    -------
    trav_srcs : (ns, nx, nz) float32
    trav_recs : (nr, nx, nz) float32
    const_vel : float or None
        The velocity when the analytic path was taken, else ``None``.
    """
    vel = np.asarray(vel, dtype=np.float64)
    if vel.ndim == 0 or vel.size == 1 or np.ptp(vel) == 0.0:
        v = float(vel.reshape(-1)[0])
        if not np.isfinite(v) or v <= 0:
            raise ValueError("velocity must be finite and positive")
        return (
            analytic_traveltime(srcs, v, nx, nz, dx, dz, ox, oz),
            analytic_traveltime(recs, v, nx, nz, dx, dz, ox, oz),
            v,
        )
    if vel.shape != (nx, nz):
        raise ValueError(f"vel has shape {vel.shape}, expected (nx, nz) = {(nx, nz)}")
    return (
        eikonal_traveltime(srcs, vel, dx, dz, ox, oz, **eikonal_kwargs),
        eikonal_traveltime(recs, vel, dx, dz, ox, oz, **eikonal_kwargs),
        None,
    )
