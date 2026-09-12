"""NumPy reference engine.

This is the test oracle for the CUDA engine, so it mirrors the kernel
arithmetic exactly: traveltimes, the interpolation index and weights, the
angle bin and the anti-alias filter width are all computed in float32 with the
same operation order as the kernels, and the sums are accumulated in float64
(``np.bincount``). The two engines therefore agree to float32 rounding of the
final result.

It is vectorised over blocks of receivers per source, which keeps it usable
for tests and small problems on CPU-only machines. It is not fast.

Anti-alias filtering
--------------------
With ``aa=True`` every contribution is read through a normalised triangle
filter whose half-width follows the local operator dip. The triangle is
applied with the double-running-integration identity

    (1/n^2) * (D[i] - 2 D[i-n] + D[i-2n])  =  (T_n * d)[i - n + 1]

where ``D`` is the double cumulative sum of the trace and ``T_n`` is the
triangle of half-width ``n`` (``n = 1`` is the identity, so the unfiltered
case needs no special path). Three taps regardless of ``n``. The transpose of
the double cumulative sum is the reverse double cumulative sum, which is what
``forward`` applies after scattering, so the operator pair stays an exact
transpose.
"""
from __future__ import annotations

import numpy as np

_F32 = np.float32
_PI32 = np.float32(np.pi)
_TWOPI32 = np.float32(2.0 * np.pi)


def angle_bin(ths, thr, ihd, hmax_rad, nh):
    """Angle bin index, mirroring ``kc_angle_bin`` in the CUDA source.

    All inputs are float32 (arrays broadcast against each other). Returns int32
    bins, ``-1`` where the half opening angle exceeds ``hmax_rad``.
    """
    d = np.asarray(ths, dtype=_F32) - np.asarray(thr, dtype=_F32)
    d = np.where(d > _PI32, d - _TWOPI32, d)
    d = np.where(d < -_PI32, d + _TWOPI32, d)
    g = np.abs(d) * _F32(0.5)
    h = (g * _F32(ihd)).astype(np.int32)
    return np.where(h >= nh, np.where(g > _F32(hmax_rad), -1, nh - 1), h).astype(np.int32)


def aa_width(dip, aaf, cell, aa_factor, nmax):
    """Triangle half-width in samples, mirroring ``kc_aa_width`` in the CUDA
    source.

    ``n = clip(round(aa_factor * sqrt((dip * aaf)^2 + cell^2) + 1), 1, nmax)``

    ``dip = dip_s + dip_r`` [s/m] is the operator dip along the trace axis and
    ``aaf = drho / dt`` the effective trace spacing of the (source, receiver)
    pair in samples per s/m, so ``dip * aaf`` is the time shift between
    neighbouring traces in samples (Lumley, Claerbout and Bevc 1994 eq. 4;
    ``sfmig2``'s ``tx``). ``cell`` is the time footprint of the image cell in
    samples, see :func:`aa_cell`; the two are independent smearings of the
    same trace and combine as a root mean square (LCB eq. 5, same reasoning).
    The trailing ``+1`` is Madagascar's ``+dt``: the linear interpolation is
    itself a one-sample triangle. ``n == 1`` is the identity.

    Operation order matters for the bit-for-bit match with the kernel, which
    uses ``__fmul_rn`` / ``__fadd_rn`` so nothing is fused into an FMA:
    every product and sum is rounded to float32 exactly as NumPy does here,
    then 1.5 is added in float64 (exact) and the result truncated.
    """
    a = np.asarray(dip, dtype=_F32) * np.asarray(aaf, dtype=_F32)
    b = np.asarray(cell, dtype=_F32)
    w = _F32(aa_factor) * np.sqrt(a * a + b * b, dtype=_F32)
    n = (w.astype(np.float64) + 1.5).astype(np.int32)
    return np.clip(n, 1, nmax).astype(np.int32)


def aa_cell(gs, gr, dxdt, dzdt):
    """Time footprint of one image cell on the trace, in samples, mirroring
    ``kc_aa_cell`` in the CUDA source: ``|gx_s + gx_r| dx/dt + |gz_s + gz_r|
    dz/dt`` with ``g*`` the traveltime gradients [s/m] of the two tables
    (``(..., 2)`` arrays, last axis ``(d/dx, d/dz)``). The gradients are
    summed *before* the absolute value: at the specular point the source and
    receiver lateral derivatives cancel and the footprint must be small
    there. Zero ``dxdt``/``dzdt`` switch the term off.
    """
    gs = np.asarray(gs, dtype=_F32)
    gr = np.asarray(gr, dtype=_F32)
    bx = np.abs(gs[..., 0] + gr[..., 0]) * _F32(dxdt)
    bz = np.abs(gs[..., 1] + gr[..., 1]) * _F32(dzdt)
    return bx + bz


class NumpyEngine:
    """Reference implementation of the operator pair on flat tables.

    Parameters
    ----------
    tabs_t, tabr_t : (ns, npts), (nr, npts) float32 traveltimes [s]
    hbin : (ns, nr) int32, offset bin per trace, -1 = trace not used
    nh, nt : gather bins, time samples
    idt : 1 / dt
    angle : bool, angle-domain gathers
    tabs_a, tabr_a : (ns, npts), (nr, npts) float32 emergence angles [rad]
    ihd, hmax_rad : nh / hmax_rad and hmax in radians (angle domain)
    aa : bool, anti-alias filtering
    tabs_d, tabr_d : (ns, npts), (nr, npts) float32 operator dip along the
        trace axis [s/m]
    aaf : (ns, nr) float32, ``drho / dt`` per trace (effective trace spacing
        of that source/receiver pair, in samples per s/m)
    aa_factor : dimensionless scale of the whole filter width
    aa_max : largest triangle half-width in samples
    tabs_g, tabr_g : (ns, npts, 2), (nr, npts, 2) float32 traveltime gradients
        (d/dx, d/dz) [s/m], or None to leave the image-cell term out
    dxdt, dzdt : ``dx / dt``, ``dz / dt``: cell size in samples per s/m
    tabs_w, tabr_w : (ns, npts), (nr, npts) float32 per-side amplitude
        weights, or None; every contribution is multiplied by
        ``float32(w_s * w_r)`` (converted to the accumulator type), exactly
        as the kernels do
    chunk_elems : receiver block size is chosen so that a (block, npts) float32
        work array has about this many elements
    """

    name = "numpy"

    def __init__(self, tabs_t, tabr_t, hbin, nh, nt, idt, *, angle=False,
                 tabs_a=None, tabr_a=None, ihd=0.0, hmax_rad=0.0,
                 aa=False, tabs_d=None, tabr_d=None, aaf=0.0, aa_factor=1.0, aa_max=32,
                 tabs_g=None, tabr_g=None, dxdt=0.0, dzdt=0.0,
                 tabs_w=None, tabr_w=None,
                 chunk_elems=1 << 22, **_ignored):
        self.tabs_t = np.ascontiguousarray(tabs_t, dtype=np.float32)
        self.tabr_t = np.ascontiguousarray(tabr_t, dtype=np.float32)
        self.hbin = np.ascontiguousarray(hbin, dtype=np.int32)
        self.nh, self.nt = int(nh), int(nt)
        self.ns, self.npts = self.tabs_t.shape
        self.nr = self.tabr_t.shape[0]
        self.idt32 = _F32(idt)
        self.angle = bool(angle)
        self.ihd = float(ihd)
        self.hmax_rad = float(hmax_rad)
        if self.angle:
            self.tabs_a = np.ascontiguousarray(tabs_a, dtype=np.float32)
            self.tabr_a = np.ascontiguousarray(tabr_a, dtype=np.float32)
        self.aa = bool(aa)
        if self.aa:
            self.tabs_d = np.ascontiguousarray(tabs_d, dtype=np.float32)
            self.tabr_d = np.ascontiguousarray(tabr_d, dtype=np.float32)
            self.aaf = np.ascontiguousarray(aaf, dtype=np.float32)
            self.aa_factor = float(aa_factor)
            self.aa_max = int(aa_max)
            self.stretch = tabs_g is not None
            if self.stretch:
                self.tabs_g = np.ascontiguousarray(tabs_g, dtype=np.float32)
                self.tabr_g = np.ascontiguousarray(tabr_g, dtype=np.float32)
                self.dxdt, self.dzdt = float(dxdt), float(dzdt)
            self.pad = self.aa_max + 1              # D index of time sample 0
            self.npadded = self.nt + 2 * self.aa_max + 1
        self.weighted = tabs_w is not None
        if self.weighted:
            self.tabs_w = np.ascontiguousarray(tabs_w, dtype=np.float32)
            self.tabr_w = np.ascontiguousarray(tabr_w, dtype=np.float32)
        self.rchunk = max(1, int(chunk_elems) // max(self.npts, 1))

    # ------------------------------------------------------------------ core
    def _pairs(self, s, r0, r1):
        """Valid (receiver, image point) pairs for source ``s`` and receivers
        ``r0:r1``. Returns local receiver index, image point, time sample,
        float32 weight of the second tap, the gather bin, the triangle
        half-width (``None`` when anti-aliasing is off) and the float64
        amplitude weight (``None`` when unweighted)."""
        t = (self.tabs_t[s][None, :] + self.tabr_t[r0:r1]) * self.idt32
        it = np.floor(t).astype(np.int32)
        w = t - it.astype(np.float32)
        valid = (it >= 0) & (it < self.nt - 1)
        hb = self.hbin[s, r0:r1]
        valid &= (hb >= 0)[:, None]
        if self.angle:
            h = angle_bin(self.tabs_a[s][None, :], self.tabr_a[r0:r1],
                          self.ihd, self.hmax_rad, self.nh)
            valid &= h >= 0
        else:
            h = np.broadcast_to(hb[:, None], t.shape)
        rr, pp = np.nonzero(valid)
        n = None
        if self.aa:
            dip = self.tabs_d[s][None, :] + self.tabr_d[r0:r1]
            cell = _F32(0.0)
            if self.stretch:
                cell = aa_cell(self.tabs_g[s][None, :], self.tabr_g[r0:r1],
                               self.dxdt, self.dzdt)
            n = aa_width(dip, self.aaf[s, r0:r1][:, None], cell,
                         self.aa_factor, self.aa_max)[rr, pp]
        wt = None
        if self.weighted:
            wt = (self.tabs_w[s][None, :] * self.tabr_w[r0:r1])[rr, pp].astype(np.float64)
        return rr, pp, it[rr, pp], w[rr, pp], h[rr, pp], n, wt

    @staticmethod
    def _taps(w32):
        """Float64 interpolation weights from the float32 fractional offset,
        formed exactly like the kernel: (double)(1.0f - w) and (double)w."""
        return (_F32(1.0) - w32).astype(np.float64), w32.astype(np.float64)

    def _integrate(self, traces):
        """Double cumulative sum of ``(rc, nt)`` traces onto the padded time
        axis, in float64. ``D[:, pad + i]`` holds the double integral up to
        time sample ``i``."""
        d = np.zeros((traces.shape[0], self.npadded), dtype=np.float64)
        d[:, self.pad:self.pad + self.nt] = traces
        np.cumsum(d, axis=1, out=d)
        np.cumsum(d, axis=1, out=d)
        return d

    @staticmethod
    def _reverse_integrate(p):
        """Transpose of :meth:`_integrate`: reverse double cumulative sum."""
        q = p[:, ::-1]
        q = np.cumsum(q, axis=1)
        np.cumsum(q, axis=1, out=q)
        return q[:, ::-1]

    # -------------------------------------------------------------- adjoint
    def adjoint(self, data):
        """data (ns, nr, nt) float32 -> model (nh, npts) float32."""
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.shape != (self.ns, self.nr, self.nt):
            raise ValueError(f"data has shape {data.shape}, expected {(self.ns, self.nr, self.nt)}")
        nbins = self.nh * self.npts
        acc = np.zeros(nbins, dtype=np.float64)
        for s in range(self.ns):
            ds = data[s]
            for r0 in range(0, self.nr, self.rchunk):
                r1 = min(self.nr, r0 + self.rchunk)
                rr, pp, itv, wv, hv, nv, wt = self._pairs(s, r0, r1)
                if rr.size == 0:
                    continue
                w1, w2 = self._taps(wv)
                rg = r0 + rr
                if self.aa:
                    dd = self._integrate(ds[r0:r1])
                    base = itv.astype(np.int64) + self.pad

                    def tap(off):
                        i = base + off
                        return dd[rr, i] * w1 + dd[rr, i + 1] * w2

                    n64 = nv.astype(np.float64)
                    v = (tap(nv - 1) - 2.0 * tap(-1) + tap(-nv - 1)) / (n64 * n64)
                else:
                    v = ds[rg, itv].astype(np.float64) * w1 \
                        + ds[rg, itv + 1].astype(np.float64) * w2
                if wt is not None:
                    v = wt * v
                idx = hv.astype(np.int64) * self.npts + pp
                acc += np.bincount(idx, weights=v, minlength=nbins)
        return acc.reshape(self.nh, self.npts).astype(np.float32)

    # -------------------------------------------------------------- forward
    def forward(self, model):
        """model (nh, npts) float32 -> data (ns, nr, nt) float32."""
        model = np.ascontiguousarray(model, dtype=np.float32)
        if model.shape != (self.nh, self.npts):
            raise ValueError(f"model has shape {model.shape}, expected {(self.nh, self.npts)}")
        mflat = model.astype(np.float64).ravel()
        out = np.zeros((self.ns, self.nr, self.nt), dtype=np.float32)
        for s in range(self.ns):
            for r0 in range(0, self.nr, self.rchunk):
                r1 = min(self.nr, r0 + self.rchunk)
                rc = r1 - r0
                rr, pp, itv, wv, hv, nv, wt = self._pairs(s, r0, r1)
                if rr.size == 0:
                    continue
                w1, w2 = self._taps(wv)
                v = mflat[hv.astype(np.int64) * self.npts + pp]
                if wt is not None:
                    v = v * wt
                if self.aa:
                    n64 = nv.astype(np.float64)
                    v = v / (n64 * n64)
                    base = rr.astype(np.int64) * self.npadded + itv + self.pad
                    n = rc * self.npadded
                    acc = np.zeros(n, dtype=np.float64)
                    for off, c in ((nv - 1, 1.0), (-1, -2.0), (-nv - 1, 1.0)):
                        i = base + off
                        acc += np.bincount(i, weights=v * w1 * c, minlength=n)
                        acc += np.bincount(i + 1, weights=v * w2 * c, minlength=n)
                    full = self._reverse_integrate(acc.reshape(rc, self.npadded))
                    out[s, r0:r1] = full[:, self.pad:self.pad + self.nt]
                else:
                    idx = rr.astype(np.int64) * self.nt + itv
                    n = rc * self.nt
                    acc = np.bincount(idx, weights=v * w1, minlength=n)
                    acc += np.bincount(idx + 1, weights=v * w2, minlength=n)
                    out[s, r0:r1] = acc.reshape(rc, self.nt)
        return out
