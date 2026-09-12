"""Half-order time derivative (the 2D Kirchhoff "rho" filter).

Kirchhoff demigration in 2D needs a half-order derivative along time: spreading
an image point along its traveltime curve and summing the spread points over a
reflector leaves, by stationary phase over the one lateral dimension, a factor
``|omega|^-1/2 exp(-i pi/4)`` behind. The filter

    H(omega) = sqrt(1 - rho * exp(-i omega))

puts it back. It is the spectrum of the causal, minimum-phase half difference
``sqrt(1 - rho z^-1)``: for ``rho = 1`` its magnitude is ``sqrt(2 sin(omega/2))``,
close to ``sqrt(omega)`` in the band, with a 45-degree phase. ``rho`` slightly
below 1 (default ``1 - 1/nt``) keeps the DC response finite. Same filter, same
default, as Madagascar's ``sf_halfint(inv=true)`` which ``sfmig2``, ``sfkirchnew``
and, through the order-2 Ricker, ``sfkirmod`` apply.

Being the root of the *backward* difference, its phase is ``pi/4 - omega/4``:
the 45 degrees plus a quarter-sample delay, the half of the half-sample delay
of ``1 - z^-1``. The adjoint advances by the same quarter sample, so round
trips are unshifted; a migrated reflector sits ``dt/4`` (in two-way time)
shallower than with ``halfderiv=False``.

``forward`` applies ``H`` to every trace of demigrated data; ``adjoint``
applies its exact transpose ``H^*`` (conjugate spectrum) to the data before
migration. Both are circular convolutions on traces zero-padded to at least
twice their length, so the operator pair stays an exact transpose: the
transpose of pad-convolve-truncate is pad-convolve-with-the-conjugate-truncate.
The FFTs run in float64 whatever the input dtype.
"""
from __future__ import annotations

import math

import numpy as np


def next_fast_len(n):
    """Smallest 5-smooth integer >= n (fast FFT length)."""
    n = int(n)
    if n <= 6:
        return max(n, 1)
    best = 1 << (n - 1).bit_length()               # power of two is always an option
    p5 = 1
    while p5 < best:
        p3 = p5
        while p3 < best:
            p2 = p3
            while p2 < n:
                p2 <<= 1
            best = min(best, p2)
            p3 *= 3
        p5 *= 5
    return best


class HalfDerivative:
    """Half-derivative filter on the last axis of ``(..., nt)`` arrays.

    Parameters
    ----------
    nt : trace length
    rho : leak, default ``1 - 1/nt``
    xp : array module, ``numpy`` or ``cupy``
    mem_budget : bytes of float64/complex128 work arrays per call; longer
        leading axes are processed in chunks
    """

    def __init__(self, nt, rho=None, xp=np, mem_budget=256 << 20):
        self.nt = int(nt)
        self.rho = float(1.0 - 1.0 / self.nt if rho is None else rho)
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("rho must be in (0, 1]")
        self.n = next_fast_len(2 * self.nt)             # >= 2 nt: no wrap-around of the tail
        self.xp = xp
        self.mem_budget = int(mem_budget)
        k = np.arange(self.n // 2 + 1, dtype=np.float64)
        z = 1.0 - self.rho * np.exp(-2j * np.pi * k / self.n)
        self.spec = xp.asarray(np.sqrt(z))                 # principal root; Re z >= 1 - rho > 0
        self.spec_adj = xp.asarray(np.conj(np.sqrt(z)))

    def __call__(self, x, adj=False):
        """Filter ``x`` (``(..., nt)``, any real dtype) along the last axis.
        Returns float32."""
        xp = self.xp
        x = xp.asarray(x)
        if x.shape[-1] != self.nt:
            raise ValueError(f"last axis must have length {self.nt}, got {x.shape[-1]}")
        lead = x.shape[:-1]
        flat = x.reshape(-1, self.nt)
        out = xp.empty(flat.shape, dtype=xp.float32)
        spec = self.spec_adj if adj else self.spec
        per_trace = 8 * self.n + 16 * (self.n // 2 + 1)    # padded float64 + complex128
        chunk = max(1, self.mem_budget // per_trace)
        for i in range(0, flat.shape[0], chunk):
            blk = flat[i:i + chunk].astype(xp.float64)
            f = xp.fft.rfft(blk, n=self.n, axis=-1)
            f *= spec
            out[i:i + chunk] = xp.fft.irfft(f, n=self.n, axis=-1)[:, :self.nt]
        return out.reshape(lead + (self.nt,))

    def kernel(self, nsamp=None):
        """Impulse response as a NumPy array, ``nsamp`` samples (default ``nt``)."""
        d = np.zeros(self.nt)
        d[0] = 1.0
        k = self(d)
        k = k.get() if hasattr(k, "get") else k
        return np.asarray(k)[:nsamp or self.nt]
