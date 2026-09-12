"""CUDA kernels for kirchcig.

The source is kept as a Python string so the package needs no build step and no
package-data handling. It is compiled at runtime by NVRTC through
``cupy.RawKernel``. All problem-shape constants that matter for register and
shared-memory allocation are injected as ``-D`` flags by ``_engine_cuda.py``:

    NH      number of gather bins (true compile-time constant)
    BLOCK   threads per block for the adjoint kernel (one thread per image point)
    FBLOCK  threads per block for the forward kernel (one block per trace)
    ACC     accumulator type, ``double`` (default) or ``float``
    OUT     element type written by the adjoint kernel: ``float`` when writing the
            final model directly, ``ACC`` when writing per-split partial sums
    ANGLE   0 = offset-domain gathers, 1 = angle-domain gathers
    AA      0 = plain summation, 1 = anti-alias triangle filtering
    AAS     1 = the filter width also covers the image cell's time footprint
            (anti-aliased stretch); needs the traveltime-gradient tables
    WEIGHT  1 = multiply every contribution by w_s(s, ip) * w_r(r, ip), the
            product of two per-side amplitude tables packed into the element

Both kernels compute the interpolation index and weights with the *same*
float32 expression, so the pair is an exact transpose to accumulator precision.

Anti-alias filtering (AA = 1)
-----------------------------
Every contribution is read through a normalised triangle filter whose
half-width ``n`` follows the local operator dip (Lumley, Claerbout and Bevc
1994) and, with AAS, the time footprint of the image cell (Madagascar's
``aastretch``: an image sample stands for a ``dx * dz`` cell whose traveltime
spans ``|dtau/dx| dx + |dtau/dz| dz``; spreading it over that span keeps a
coarse depth grid from demigrating into a comb of spikes). The two widths
combine as a root mean square. The triangle is applied with the
double-running-integration identity

    (1/n^2) * (D[i+n-1] - 2 D[i-1] + D[i-n-1])  =  (T_n * d)[i]

where ``D`` is the double cumulative sum of the trace. Three taps regardless of
``n``. The adjoint kernel therefore reads a *float64* ``D`` buffer of row length
``npad = nt + 2*aa_max + 1`` (time sample ``i`` sits at ``pad + i``,
``pad = aa_max + 1``) that the host prepares with two cumulative sums; the
forward kernel scatters the six taps into a float64 buffer of the same shape
that the host reverse-integrates twice afterwards. ``D`` has to be float64: it
grows like ``nt^2`` and the second difference cancels almost all of it.

With AA = 0 the host passes ``npad = nt`` and ``pad = 0`` and the kernels are
exactly the pre-0.2 kernels.
"""

CUDA_SOURCE = r"""
#if !defined(NH) || !defined(BLOCK) || !defined(FBLOCK) || !defined(ACC) || !defined(OUT) || !defined(ANGLE)
#error "kirchcig kernels need NH, BLOCK, FBLOCK, ACC, OUT and ANGLE defined"
#endif
#ifndef AA
#define AA 0
#endif
#ifndef AAS
#define AAS 0
#endif
#if AAS && !AA
#error "AAS needs AA"
#endif
#ifndef WEIGHT
#define WEIGHT 0
#endif

// ---------------------------------------------------------------------------
// double atomicAdd fallback for pre-Pascal devices (sm < 60). Only used in the
// forward kernel on shared memory.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ __forceinline__ double atomicAdd(double* address, double val)
{
    unsigned long long* p = (unsigned long long*)address;
    unsigned long long old = *p, assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// ---------------------------------------------------------------------------
// Traveltime table element. Whatever else a table row carries per image point
// (emergence angle for the angle domain, operator dip for anti-aliasing, an
// amplitude weight) is packed next to the traveltime, in the fixed order
// t, a, d, w, padded to 1, 2 or 4 floats so it always comes from a single
// aligned, coalesced load. The host packs the same order (_engine_cuda._upload).
// ---------------------------------------------------------------------------
#define KC_NFIELDS (1 + ANGLE + AA + WEIGHT)
#if KC_NFIELDS == 1
typedef float tab_t;
#define TAB_T(v) (v)
#else
#if KC_NFIELDS == 2
struct alignas(8) tab_t {
#else
struct alignas(16) tab_t {
#endif
    float t;
#if ANGLE
    float a;
#endif
#if AA
    float d;
#endif
#if WEIGHT
    float w;
#endif
#if KC_NFIELDS == 3
    float pad_;
#endif
};
#define TAB_T(v) ((v).t)
#define TAB_D(v) ((v).d)
#define TAB_W(v) ((v).w)
#endif

// Traveltime gradient (d/dx, d/dz) [s/m] per image point, read only with AAS.
struct alignas(8) grad_t { float gx; float gz; };

// Data-side element types. With anti-aliasing the adjoint reads the float64
// double integral and the forward writes float64 partial sums; otherwise both
// sides are the float32 traces. The forward's shared-memory trace accumulator
// is float64 whenever AA is on, whatever ACC is: its contents are integrated
// twice afterwards, which turns float32 rounding noise into a smooth error of
// order 1e-4 (measured) instead of the usual 1e-7.
#if AA
typedef double din_t;
typedef double dout_t;
typedef double facc_t;
#else
typedef float din_t;
typedef float dout_t;
typedef ACC facc_t;
#endif

__device__ __forceinline__ int kc_min(int a, int b) { return a < b ? a : b; }

#define KC_PI  3.14159265358979323846f
#define KC_2PI 6.28318530717958647692f

#if ANGLE
// Half opening angle gamma = |theta_s - theta_r| / 2 (angles measured from the
// vertical, wrapped into (-pi, pi]). Bins are [k*dg, (k+1)*dg) with the last bin
// closed at hmax. Returns -1 when out of range. Mirrored bit-for-bit by the
// numpy engine (float32 arithmetic, same operation order).
__device__ __forceinline__ int kc_angle_bin(float ths, float thr, float ihd, float hmax_rad)
{
    float d = ths - thr;
    if (d >  KC_PI) d -= KC_2PI;
    if (d < -KC_PI) d += KC_2PI;
    const float g = fabsf(d) * 0.5f;
    int h = (int)(g * ihd);
    if (h >= NH) h = (g > hmax_rad) ? -1 : (NH - 1);
    return h;
}
#endif

#if AA
// Triangle half-width in samples:
//     n = clip(round(aa_factor * sqrt((dip*aaf)^2 + cell^2) + 1), 1, nmax)
// dip is the summed source+receiver operator dip [s/m], aaf = drho/dt the
// trace's effective spacing (so dip*aaf is the time shift between neighbouring
// traces in samples), cell the image cell's time footprint in samples (0
// without AAS). Mirrored bit-for-bit by _engine_numpy.aa_width: the __*_rn
// intrinsics stop the compiler from fusing anything into an FMA with a
// different rounding, sqrtf is correctly rounded, and the final 1.5 is added
// in float64 (exact) before truncation.
__device__ __forceinline__ int kc_aa_width(float dip, float aaf, float cell, float aa_factor, int nmax)
{
    const float a = __fmul_rn(dip, aaf);
    const float w = __fmul_rn(aa_factor, sqrtf(__fadd_rn(__fmul_rn(a, a), __fmul_rn(cell, cell))));
    int n = (int)((double)w + 1.5);
    if (n < 1) n = 1;
    if (n > nmax) n = nmax;
    return n;
}
#endif
#if AAS
// |gx_s + gx_r| dx/dt + |gz_s + gz_r| dz/dt: gradients summed before the
// absolute value so the footprint vanishes at the specular point.
__device__ __forceinline__ float kc_aa_cell(grad_t gs, grad_t gr, float dxdt, float dzdt)
{
    const float bx = __fmul_rn(fabsf(__fadd_rn(gs.gx, gr.gx)), dxdt);
    const float bz = __fmul_rn(fabsf(__fadd_rn(gs.gz, gr.gz)), dzdt);
    return __fadd_rn(bx, bz);
}
#endif

// ---------------------------------------------------------------------------
// Adjoint (migration): data (ns, nr, npad) -> model (NH, npts)
//
// One thread per image point. The thread owns the whole gather axis of its
// image point, so every write is exclusive and no atomics are needed. The
// accumulators live in dynamic shared memory laid out [NH][BLOCK]: thread tid
// always touches word (h*BLOCK + tid), whose bank does not depend on h, so the
// access is bank-conflict free for any per-thread bin index.
//
// grid.y splits the source loop into blockIdx.y-th chunk of s_per_split
// sources. With grid.y == 1 the kernel writes the float32 model directly
// (OUT = float); with grid.y > 1 it writes ACC partials (OUT = ACC) that the
// host reduces. The split exists purely to fill large GPUs when npts/BLOCK is
// a small number of blocks.
//
// npad is the row length of one trace in `data` and pad the index of time
// sample 0 in that row (nt and 0 without anti-aliasing).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(BLOCK)
kirch_adjoint(const din_t*  __restrict__ data,
              const tab_t*  __restrict__ tab_s,
              const tab_t*  __restrict__ tab_r,
              const grad_t* __restrict__ grd_s,
              const grad_t* __restrict__ grd_r,
              const int*    __restrict__ hbin,
              const float*  __restrict__ aaf,
              OUT*          __restrict__ out,
              const int ns, const int nr, const int nt, const int npts,
              const int s_per_split,
              const int npad, const int pad, const int aa_max,
              const float idt, const float ihd, const float hmax_rad,
              const float aa_factor, const float dxdt, const float dzdt)
{
    extern __shared__ ACC acc[];                       // [NH][BLOCK]
    const int tid = threadIdx.x;
    const int ip  = blockIdx.x * BLOCK + tid;
    const int s0  = blockIdx.y * s_per_split;
    const int s1  = kc_min(s0 + s_per_split, ns);

    #pragma unroll 8
    for (int h = 0; h < NH; ++h) acc[h * BLOCK + tid] = (ACC)0;

    if (ip < npts) {
        for (int s = s0; s < s1; ++s) {
            const tab_t vs = tab_s[(size_t)s * npts + ip];
            const din_t* __restrict__ dsrc = data + (size_t)s * nr * npad;
            const int*   __restrict__ hrow = hbin + (size_t)s * nr;
#if AA
            const float* __restrict__ arow = aaf + (size_t)s * nr;
#endif
#if AAS
            const grad_t gs = grd_s[(size_t)s * npts + ip];
#endif

            for (int r = 0; r < nr; ++r) {
                const int hb = hrow[r];                // uniform across the block
                if (hb < 0) continue;

                const tab_t vr = tab_r[(size_t)r * npts + ip];   // coalesced
                const float t  = (TAB_T(vs) + TAB_T(vr)) * idt;
                const int   it = (int)floorf(t);
                if (it < 0 || it >= nt - 1) continue;
#if ANGLE
                const int h = kc_angle_bin(vs.a, vr.a, ihd, hmax_rad);
                if (h < 0) continue;
#else
                const int h = hb;
#endif
                const float w = t - (float)it;
                const din_t* __restrict__ tr = dsrc + (size_t)r * npad + (it + pad);
#if WEIGHT
                const ACC wt = (ACC)__fmul_rn(TAB_W(vs), TAB_W(vr));
#else
                const ACC wt = (ACC)1;
#endif
#if AA
                // tap(off) = D[it+off]*(1-w) + D[it+off+1]*w, all float64.
#if AAS
                const float cell = kc_aa_cell(gs, grd_r[(size_t)r * npts + ip], dxdt, dzdt);
#else
                const float cell = 0.0f;
#endif
                const int    n  = kc_aa_width(__fadd_rn(TAB_D(vs), TAB_D(vr)), arow[r], cell,
                                              aa_factor, aa_max);
                const double w1 = (double)(1.0f - w), w2 = (double)w;
                const double tp  = tr[n - 1]  * w1 + tr[n]      * w2;
                const double tm  = tr[-1]     * w1 + tr[0]      * w2;
                const double tmm = tr[-n - 1] * w1 + tr[-n]     * w2;
                const double nn  = (double)n * (double)n;
                acc[h * BLOCK + tid] += wt * (ACC)((tp - 2.0 * tm + tmm) / nn);
#else
                acc[h * BLOCK + tid] += wt * ((ACC)tr[0] * (ACC)(1.0f - w)
                                            + (ACC)tr[1] * (ACC)w);
#endif
            }
        }

        // Exclusive ownership: no __syncthreads needed before the writeback.
        OUT* __restrict__ o = out + (size_t)blockIdx.y * NH * npts + ip;
        #pragma unroll 8
        for (int h = 0; h < NH; ++h) o[(size_t)h * npts] = (OUT)acc[h * BLOCK + tid];
    }
}

// ---------------------------------------------------------------------------
// Forward (demigration): model (NH, npts) -> data (ns, nr, npad)
//
// One block per trace (blockIdx.x = s*nr + r). Threads stride over the image
// points, read the model coalesced and spread each value onto the time samples
// of a shared-memory trace with cheap shared atomics; the trace is written out
// once. blockIdx.y selects a window of tchunk samples of the (padded) trace so
// traces longer than the shared-memory budget are handled exactly: every tap
// is bounds-checked individually and each window adds only the taps it owns.
//
// Without anti-aliasing there are two taps (linear interpolation) and the
// output is the float32 trace. With it there are six: the transpose of the
// three-tap second difference of the double integral, each spread over two
// samples; the host reverse-integrates the float64 output twice.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(FBLOCK)
kirch_forward(const float*  __restrict__ model,
              const tab_t*  __restrict__ tab_s,
              const tab_t*  __restrict__ tab_r,
              const grad_t* __restrict__ grd_s,
              const grad_t* __restrict__ grd_r,
              const int*    __restrict__ hbin,
              const float*  __restrict__ aaf,
              dout_t*       __restrict__ data,
              const int ns, const int nr, const int nt, const int npts,
              const int tchunk,
              const int npad, const int pad, const int aa_max,
              const float idt, const float ihd, const float hmax_rad,
              const float aa_factor, const float dxdt, const float dzdt)
{
    extern __shared__ facc_t trace[];                  // [tchunk]
    const int tid  = threadIdx.x;
    const int pair = blockIdx.x;
    const int s    = pair / nr;
    const int r    = pair - s * nr;
    const int t0   = blockIdx.y * tchunk;
    const int nloc = kc_min(tchunk, npad - t0);
    dout_t* __restrict__ o = data + (size_t)pair * npad + t0;

    const int hb = hbin[pair];
    if (hb < 0) {                                      // inactive trace: zero fill
        for (int i = tid; i < nloc; i += FBLOCK) o[i] = (dout_t)0;
        return;
    }

    for (int i = tid; i < nloc; i += FBLOCK) trace[i] = (facc_t)0;
    __syncthreads();

    const tab_t* __restrict__ ts = tab_s + (size_t)s * npts;
    const tab_t* __restrict__ tr = tab_r + (size_t)r * npts;
#if !ANGLE
    const float* __restrict__ mh = model + (size_t)hb * npts;
#endif
#if AA
    const float aaf_pair = aaf[pair];
#endif
#if AAS
    const grad_t* __restrict__ gs_row = grd_s + (size_t)s * npts;
    const grad_t* __restrict__ gr_row = grd_r + (size_t)r * npts;
#endif

    for (int ip = tid; ip < npts; ip += FBLOCK) {
        const tab_t vs = ts[ip];
        const tab_t vr = tr[ip];
        const float t  = (TAB_T(vs) + TAB_T(vr)) * idt;
        const int   it = (int)floorf(t);
        if (it < 0 || it >= nt - 1) continue;
        const int i0 = it + pad - t0;                  // window-relative index of sample it
#if AA
#if AAS
        const float cell = kc_aa_cell(gs_row[ip], gr_row[ip], dxdt, dzdt);
#else
        const float cell = 0.0f;
#endif
        const int n = kc_aa_width(__fadd_rn(TAB_D(vs), TAB_D(vr)), aaf_pair, cell, aa_factor, aa_max);
        if (i0 + n < 0 || i0 - n - 1 >= nloc) continue;   // all six taps outside this window
#else
        if (i0 < -1 || i0 >= nloc) continue;          // neither tap in this window
#endif
#if ANGLE
        const int h = kc_angle_bin(vs.a, vr.a, ihd, hmax_rad);
        if (h < 0) continue;
        const float mv = model[(size_t)h * npts + ip];
#else
        const float mv = mh[ip];
#endif
        const float w = t - (float)it;
#if WEIGHT
        const ACC wt = (ACC)__fmul_rn(TAB_W(vs), TAB_W(vr));
#else
        const ACC wt = (ACC)1;
#endif
#if AA
        const double nn = (double)n * (double)n;
        const double v  = ((double)mv * (double)wt) / nn;
        const double w1 = (double)(1.0f - w), w2 = (double)w;
        const double a1 = v * w1, a2 = v * w2;
        // taps: +1 at it+n-1, -2 at it-1, +1 at it-n-1, each spread over (i, i+1)
        int i;
        i = i0 + n - 1;
        if (i >= 0     && i < nloc)     atomicAdd(&trace[i],     a1);
        if (i + 1 >= 0 && i + 1 < nloc) atomicAdd(&trace[i + 1], a2);
        i = i0 - 1;
        if (i >= 0     && i < nloc)     atomicAdd(&trace[i],     a1 * -2.0);
        if (i + 1 >= 0 && i + 1 < nloc) atomicAdd(&trace[i + 1], a2 * -2.0);
        i = i0 - n - 1;
        if (i >= 0     && i < nloc)     atomicAdd(&trace[i],     a1);
        if (i + 1 >= 0 && i + 1 < nloc) atomicAdd(&trace[i + 1], a2);
#else
        const ACC v = (ACC)mv * wt;
        if (i0 >= 0)        atomicAdd(&trace[i0],     v * (ACC)(1.0f - w));
        if (i0 + 1 < nloc)  atomicAdd(&trace[i0 + 1], v * (ACC)w);
#endif
    }
    __syncthreads();

    for (int i = tid; i < nloc; i += FBLOCK) o[i] = (dout_t)trace[i];
}
"""
