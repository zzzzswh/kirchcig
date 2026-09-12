"""CUDA engine: NVRTC-compiled kernels via CuPy.

Launch configuration
--------------------
Adjoint: one thread per image point, ``BLOCK`` threads per block, and
``NH * BLOCK * sizeof(ACC)`` bytes of dynamic shared memory per block. The
block size is chosen from {256, 128, 64, 32} as the largest that keeps the
shared-memory footprint at or below 32 KB, so several blocks fit per SM; very
large ``nh`` falls back to 32 threads and the opt-in shared-memory limit. When
``npts / BLOCK`` is too few blocks to fill the GPU, the source loop is split
across ``grid.y`` and the partial sums are reduced on the device (in float64).

Forward: one block per trace, ``FBLOCK`` threads, one shared-memory trace of
``tchunk * sizeof(ACC)`` bytes. ``tchunk`` is the trace length unless that does
not fit in the opt-in shared-memory limit, in which case the time axis is
windowed over ``grid.y`` (exactly; see the kernel comment).

Anti-aliasing (``aa=True``)
---------------------------
The kernels read/write the *double integral* of the traces instead of the
traces themselves (see ``_kernels.py``). Before the adjoint kernel the engine
zero-pads the data to ``npad = nt + 2*aa_max + 1`` samples per trace and takes
two cumulative sums in float64; after the forward kernel it takes two reverse
cumulative sums of the float64 output and cuts the padding off. Both are a few
CuPy calls on ``(ns, nr, npad)`` arrays and cost far less than the kernels, but
they do need a float64 copy of the data, ``ns * nr * npad * 8`` bytes, next to
the float32 input. The kernels themselves read three float64 taps per pair
instead of one float32 sample, so expect the adjoint to be a few times slower
than without anti-aliasing.

Everything a kernel needs per (side, image point) -- traveltime, emergence
angle, operator dip, amplitude weight, traveltime gradient -- is packed into
one table element of 1, 2, 4 or 8 floats (``_upload``), so a pair costs one or
two aligned loads per side. The adjoint processes sources in chunks of
``schunk`` (``SCH``), loading each receiver-table element once per chunk; the
receiver table is the dominant memory stream, so this divides its traffic by
``schunk``.

Compilation is keyed on ``(nh, block, fblock, acc, out, angle, aa, aas,
weight, schunk, device)`` and cached in-process, with CuPy's on-disk cache
underneath.
"""
from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - depends on the environment
    import cupy as cp
except Exception:  # cupy missing or unusable
    cp = None

from ._kernels import CUDA_SOURCE

_ACC_CTYPE = {"float64": "double", "float32": "float"}
_DEFAULT_SMEM = 48 * 1024
_KERNELS: dict = {}


def cuda_available() -> bool:
    """True when CuPy imports and at least one CUDA device is visible."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _device_limits():
    attrs = cp.cuda.Device().attributes
    sms = int(attrs.get("MultiProcessorCount", 16))
    optin = int(attrs.get("MaxSharedMemoryPerBlockOptin", 0) or 0)
    return sms, max(optin, _DEFAULT_SMEM)


def _kernel(name, *, nh, block, fblock, acc, out, angle, aa, aas, weight, schunk, smem):
    """Compile (or fetch from cache) one kernel specialisation."""
    key = (name, nh, block, fblock, acc, out, int(angle), int(aa), int(aas), int(weight),
           int(schunk), cp.cuda.Device().id)
    k = _KERNELS.get(key)
    if k is None:
        options = (
            f"-DNH={nh}",
            f"-DBLOCK={block}",
            f"-DFBLOCK={fblock}",
            f"-DACC={acc}",
            f"-DOUT={out}",
            f"-DANGLE={int(angle)}",
            f"-DAA={int(aa)}",
            f"-DAAS={int(aas)}",
            f"-DWEIGHT={int(weight)}",
            f"-DSCH={int(schunk)}",
        )
        k = cp.RawKernel(CUDA_SOURCE, name, options=options, backend="nvrtc")
        k.compile()
        _KERNELS[key] = k
    if smem > _DEFAULT_SMEM:
        # Opt in to more than 48 KB of dynamic shared memory (Volta and later).
        k.max_dynamic_shared_size_bytes = int(smem)
    return k


def choose_block(nh, acc_bytes, smem_optin, block=None):
    """Adjoint block size for ``nh`` bins and ``acc_bytes`` per accumulator."""
    if block is not None:
        block = int(block)
        if block <= 0 or block % 32 or block > 1024:
            raise ValueError("block must be a positive multiple of 32, at most 1024")
        if nh * block * acc_bytes > smem_optin:
            raise ValueError(
                f"nh={nh} with block={block} needs {nh * block * acc_bytes / 1024:.0f} KB "
                f"of shared memory, more than this device allows ({smem_optin // 1024} KB)")
        return block
    for b in (256, 128, 64, 32):
        if nh * b * acc_bytes <= 32 * 1024:
            return b
    if nh * 32 * acc_bytes <= smem_optin:
        return 32
    raise ValueError(
        f"nh={nh} needs {nh * 32 * acc_bytes / 1024:.0f} KB of shared memory per block even "
        f"at 32 threads; this device allows {smem_optin // 1024} KB. Use fewer bins or "
        f"acc='float32'.")


class CudaEngine:
    """The operator pair on the GPU. Works on CuPy arrays of flat tables.

    See :class:`kirchcig.NumpyEngine` for the meaning of the table arguments.

    Additional parameters
    ---------------------
    acc : 'float64' (default) or 'float32', accumulator type
    block : adjoint threads per block; ``None`` picks automatically
    fblock : forward threads per block (default 256)
    split : 'auto' or int, number of source chunks for the adjoint grid
    split_mem_budget : bytes allowed for the adjoint partial-sum buffer
    schunk : sources per register chunk in the adjoint kernel (default 4)
    """

    name = "cuda"

    def __init__(self, tabs_t, tabr_t, hbin, nh, nt, idt, *, angle=False,
                 tabs_a=None, tabr_a=None, ihd=0.0, hmax_rad=0.0,
                 aa=False, tabs_d=None, tabr_d=None, aaf=0.0, aa_factor=1.0, aa_max=32,
                 tabs_g=None, tabr_g=None, dxdt=0.0, dzdt=0.0,
                 tabs_w=None, tabr_w=None,
                 acc="float64", block=None, fblock=256, split="auto",
                 split_mem_budget=256 << 20, schunk=4):
        if not cuda_available():
            raise RuntimeError("engine='cuda' needs CuPy and a visible CUDA device")
        if acc not in _ACC_CTYPE:
            raise ValueError("acc must be 'float64' or 'float32'")
        tabs_t = np.asarray(tabs_t, dtype=np.float32)
        tabr_t = np.asarray(tabr_t, dtype=np.float32)
        self.nh, self.nt = int(nh), int(nt)
        self.ns, self.npts = tabs_t.shape
        self.nr = tabr_t.shape[0]
        self.angle = bool(angle)
        self.acc = acc
        self.acc_ctype = _ACC_CTYPE[acc]
        self.acc_dtype = np.dtype(acc)
        self.acc_bytes = self.acc_dtype.itemsize
        # forward trace accumulator: float64 under anti-aliasing (see _kernels.py)
        self.facc_bytes = 8 if aa else self.acc_bytes
        self.idt = np.float32(idt)
        self.ihd = np.float32(ihd)
        self.hmax_rad = np.float32(hmax_rad)
        self.sms, self.smem_optin = _device_limits()

        # -- anti-alias layout ------------------------------------------------
        self.aa = bool(aa)
        if self.aa:
            if tabs_d is None or tabr_d is None:
                raise ValueError("anti-aliasing needs the operator dip tables")
            self.aa_max = int(aa_max)
            self.pad = self.aa_max + 1                # row index of time sample 0
            self.npad = self.nt + 2 * self.aa_max + 1
            self.aaf = cp.asarray(np.ascontiguousarray(aaf, dtype=np.float32).ravel())
            if self.aaf.size != self.ns * self.nr:
                raise ValueError("aaf must be (ns, nr)")
        else:
            self.aa_max, self.pad, self.npad = 0, 0, self.nt
            self.aaf = cp.zeros(1, dtype=cp.float32)  # never dereferenced
        self.aa_factor = np.float32(aa_factor)
        self.aas = self.aa and tabs_g is not None
        if self.aas:
            self.dxdt, self.dzdt = np.float32(dxdt), np.float32(dzdt)
        else:
            self.dxdt = self.dzdt = np.float32(0.0)
        self.schunk = int(schunk)
        if not 1 <= self.schunk <= 16:
            raise ValueError("schunk must be in [1, 16]")

        # -- tables on the device -------------------------------------------
        self.weighted = tabs_w is not None
        self.tab_s = self._upload(tabs_t, tabs_a, tabs_d, tabs_w, tabs_g)
        self.tab_r = self._upload(tabr_t, tabr_a, tabr_d, tabr_w, tabr_g)
        self.hbin = cp.asarray(np.ascontiguousarray(hbin, dtype=np.int32).ravel())

        # -- adjoint configuration -------------------------------------------
        self.block = choose_block(self.nh, self.acc_bytes, self.smem_optin, block)
        self.smem_adj = self.nh * self.block * self.acc_bytes
        self.nblocks = math.ceil(self.npts / self.block)
        self.split = split
        self.split_mem_budget = int(split_mem_budget)

        # -- forward configuration -------------------------------------------
        self.fblock = int(fblock)
        if self.fblock <= 0 or self.fblock % 32:
            raise ValueError("fblock must be a positive multiple of 32")
        self.set_time_chunk(None)

        # -- kernels ----------------------------------------------------------
        common = dict(nh=self.nh, block=self.block, fblock=self.fblock,
                      acc=self.acc_ctype, angle=self.angle, aa=self.aa, aas=self.aas,
                      weight=self.weighted, schunk=self.schunk)
        self._k_adj = _kernel("kirch_adjoint", out="float", smem=self.smem_adj, **common)
        self._k_fwd = _kernel("kirch_forward", out="float", smem=self.smem_fwd, **common)
        self._k_adj_partial = None  # compiled lazily, only when a split is used
        self._common = common

    # -------------------------------------------------------------- helpers
    def _upload(self, t, a, d, w, g):
        """Pack one table row-set into the kernel's ``tab_t`` layout: the fields
        ``t, a, d, w, gx, gz`` that are in use, in that order, padded to 1, 2, 4
        or 8 floats (``KC_WIDTH`` in the kernel source)."""
        t = np.ascontiguousarray(t, dtype=np.float32)
        fields = [t]
        if self.angle:
            if a is None:
                raise ValueError("angle-domain gathers need the emergence-angle tables")
            fields.append(np.asarray(a, dtype=np.float32))
        if self.aa:
            fields.append(np.asarray(d, dtype=np.float32))
        if self.weighted:
            fields.append(np.asarray(w, dtype=np.float32))
        if self.aas:
            g = np.asarray(g, dtype=np.float32).reshape(t.shape + (2,))
            fields += [g[..., 0], g[..., 1]]
        n = len(fields)
        if n == 1:
            return cp.asarray(t)
        width = 2 if n == 2 else (4 if n <= 4 else 8)
        packed = np.zeros(t.shape + (width,), dtype=np.float32)
        for i, f in enumerate(fields):
            packed[..., i] = f
        return cp.asarray(packed)

    def set_time_chunk(self, tchunk=None):
        """Set the forward time window (samples of the padded trace).
        ``None`` = largest that fits."""
        max_fit = max(1, self.smem_optin // self.facc_bytes)
        if tchunk is None:
            tchunk = min(self.npad, max_fit)
        tchunk = int(tchunk)
        if tchunk < 1 or tchunk > max_fit:
            raise ValueError(f"tchunk must be in [1, {max_fit}]")
        self.tchunk = min(tchunk, self.npad)
        self.nchunk = math.ceil(self.npad / self.tchunk)
        self.smem_fwd = self.tchunk * self.facc_bytes
        if hasattr(self, "_k_fwd") and self.smem_fwd > _DEFAULT_SMEM:
            self._k_fwd.max_dynamic_shared_size_bytes = int(self.smem_fwd)

    def _nsplit(self):
        if isinstance(self.split, (int, np.integer)) and not isinstance(self.split, bool):
            n = int(self.split)
        else:  # 'auto': aim for a few blocks per SM in flight
            n = math.ceil(6 * self.sms / max(self.nblocks, 1))
        n = max(1, min(n, self.ns))
        per_split = self.nh * self.npts * self.acc_bytes
        return max(1, min(n, self.split_mem_budget // max(per_split, 1)))

    def _shape_args(self):
        return (np.int32(self.ns), np.int32(self.nr), np.int32(self.nt), np.int32(self.npts))

    def _aa_args(self):
        return (np.int32(self.npad), np.int32(self.pad), np.int32(self.aa_max))

    def _tail_args(self):
        return (self.idt, self.ihd, self.hmax_rad, self.aa_factor, self.dxdt, self.dzdt)

    def _integrate(self, data):
        """Double cumulative sum of ``(ns, nr, nt)`` float32 traces onto the
        padded axis, float64. Mirrors ``NumpyEngine._integrate``."""
        d = cp.zeros((self.ns, self.nr, self.npad), dtype=cp.float64)
        d[..., self.pad:self.pad + self.nt] = data
        cp.cumsum(d, axis=-1, out=d)
        cp.cumsum(d, axis=-1, out=d)
        return d

    def _reverse_integrate(self, p):
        """Transpose of :meth:`_integrate` (without the cut): reverse double
        cumulative sum of ``(ns, nr, npad)`` float64."""
        q = cp.cumsum(p[..., ::-1], axis=-1)
        cp.cumsum(q, axis=-1, out=q)
        return q[..., ::-1]

    # -------------------------------------------------------------- adjoint
    def adjoint(self, data):
        """data (ns, nr, nt) -> model (nh, npts), both CuPy float32."""
        data = cp.ascontiguousarray(cp.asarray(data, dtype=cp.float32))
        if data.shape != (self.ns, self.nr, self.nt):
            raise ValueError(f"data has shape {data.shape}, expected {(self.ns, self.nr, self.nt)}")
        if self.aa:
            data = self._integrate(data)
        nsplit = self._nsplit()
        tail = (*self._aa_args(), *self._tail_args())

        if nsplit == 1:
            out = cp.empty((self.nh, self.npts), dtype=cp.float32)
            self._k_adj(
                (self.nblocks, 1, 1), (self.block, 1, 1),
                (data, self.tab_s, self.tab_r, self.hbin, self.aaf, out,
                 *self._shape_args(), np.int32(self.ns), *tail),
                shared_mem=self.smem_adj)
            return out

        s_per = math.ceil(self.ns / nsplit)
        nsplit = math.ceil(self.ns / s_per)
        if self._k_adj_partial is None:
            self._k_adj_partial = _kernel("kirch_adjoint", out=self.acc_ctype,
                                          smem=self.smem_adj, **self._common)
        part = cp.empty((nsplit, self.nh, self.npts), dtype=self.acc_dtype)
        self._k_adj_partial(
            (self.nblocks, nsplit, 1), (self.block, 1, 1),
            (data, self.tab_s, self.tab_r, self.hbin, self.aaf, part,
             *self._shape_args(), np.int32(s_per), *tail),
            shared_mem=self.smem_adj)
        return part.sum(axis=0, dtype=cp.float64).astype(cp.float32)

    # -------------------------------------------------------------- forward
    def forward(self, model):
        """model (nh, npts) -> data (ns, nr, nt), both CuPy float32."""
        model = cp.ascontiguousarray(cp.asarray(model, dtype=cp.float32))
        if model.shape != (self.nh, self.npts):
            raise ValueError(f"model has shape {model.shape}, expected {(self.nh, self.npts)}")
        dtype = cp.float64 if self.aa else cp.float32
        out = cp.empty((self.ns, self.nr, self.npad), dtype=dtype)
        self._k_fwd(
            (self.ns * self.nr, self.nchunk, 1), (self.fblock, 1, 1),
            (model, self.tab_s, self.tab_r, self.hbin, self.aaf, out,
             *self._shape_args(), np.int32(self.tchunk),
             *self._aa_args(), *self._tail_args()),
            shared_mem=self.smem_fwd)
        if self.aa:
            full = self._reverse_integrate(out)
            out = cp.ascontiguousarray(full[..., self.pad:self.pad + self.nt]
                                       .astype(cp.float32))
        return out

    def __repr__(self):
        return (f"CudaEngine(nh={self.nh}, block={self.block}, fblock={self.fblock}, "
                f"acc={self.acc}, angle={self.angle}, aa={self.aa}, aas={self.aas}, "
                f"weighted={self.weighted}, schunk={self.schunk}, "
                f"smem_adj={self.smem_adj // 1024}KB, tchunk={self.tchunk}x{self.nchunk})")
