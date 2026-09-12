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

Both kernels compute the interpolation index and weights with the *same*
float32 expression, so the pair is an exact transpose to accumulator precision.
"""

CUDA_SOURCE = r"""
#if !defined(NH) || !defined(BLOCK) || !defined(FBLOCK) || !defined(ACC) || !defined(OUT) || !defined(ANGLE)
#error "kirchcig kernels need NH, BLOCK, FBLOCK, ACC, OUT and ANGLE defined"
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
// Traveltime table element. In the angle domain the emergence angle is packed
// next to the traveltime so both come from a single 8-byte coalesced load.
// ---------------------------------------------------------------------------
#if ANGLE
struct alignas(8) tab_t { float t; float a; };
#define TAB_T(v) ((v).t)
#else
typedef float tab_t;
#define TAB_T(v) (v)
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

// ---------------------------------------------------------------------------
// Adjoint (migration): data (ns, nr, nt) -> model (NH, npts)
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
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(BLOCK)
kirch_adjoint(const float* __restrict__ data,
              const tab_t* __restrict__ tab_s,
              const tab_t* __restrict__ tab_r,
              const int*   __restrict__ hbin,
              OUT*         __restrict__ out,
              const int ns, const int nr, const int nt, const int npts,
              const int s_per_split,
              const float idt, const float ihd, const float hmax_rad)
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
            const float* __restrict__ dsrc = data + (size_t)s * nr * nt;
            const int*   __restrict__ hrow = hbin + (size_t)s * nr;

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
                const float* __restrict__ tr = dsrc + (size_t)r * nt + it;
                acc[h * BLOCK + tid] += (ACC)tr[0] * (ACC)(1.0f - w)
                                      + (ACC)tr[1] * (ACC)w;
            }
        }

        // Exclusive ownership: no __syncthreads needed before the writeback.
        OUT* __restrict__ o = out + (size_t)blockIdx.y * NH * npts + ip;
        #pragma unroll 8
        for (int h = 0; h < NH; ++h) o[(size_t)h * npts] = (OUT)acc[h * BLOCK + tid];
    }
}

// ---------------------------------------------------------------------------
// Forward (demigration): model (NH, npts) -> data (ns, nr, nt)
//
// One block per trace (blockIdx.x = s*nr + r). Threads stride over the image
// points, read the model coalesced and spread each value onto two time samples
// of a shared-memory trace with cheap shared atomics; the trace is written out
// once. blockIdx.y selects a window of tchunk time samples so traces longer
// than the shared-memory budget are handled exactly (the two linear-
// interpolation taps may fall in different windows, each window adds only the
// tap it owns).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(FBLOCK)
kirch_forward(const float* __restrict__ model,
              const tab_t* __restrict__ tab_s,
              const tab_t* __restrict__ tab_r,
              const int*   __restrict__ hbin,
              float*       __restrict__ data,
              const int ns, const int nr, const int nt, const int npts,
              const int tchunk,
              const float idt, const float ihd, const float hmax_rad)
{
    extern __shared__ ACC trace[];                     // [tchunk]
    const int tid  = threadIdx.x;
    const int pair = blockIdx.x;
    const int s    = pair / nr;
    const int r    = pair - s * nr;
    const int t0   = blockIdx.y * tchunk;
    const int nloc = kc_min(tchunk, nt - t0);
    float* __restrict__ o = data + (size_t)pair * nt + t0;

    const int hb = hbin[pair];
    if (hb < 0) {                                      // inactive trace: zero fill
        for (int i = tid; i < nloc; i += FBLOCK) o[i] = 0.0f;
        return;
    }

    for (int i = tid; i < nloc; i += FBLOCK) trace[i] = (ACC)0;
    __syncthreads();

    const tab_t* __restrict__ ts = tab_s + (size_t)s * npts;
    const tab_t* __restrict__ tr = tab_r + (size_t)r * npts;
#if !ANGLE
    const float* __restrict__ mh = model + (size_t)hb * npts;
#endif

    for (int ip = tid; ip < npts; ip += FBLOCK) {
        const tab_t vs = ts[ip];
        const tab_t vr = tr[ip];
        const float t  = (TAB_T(vs) + TAB_T(vr)) * idt;
        const int   it = (int)floorf(t);
        if (it < 0 || it >= nt - 1) continue;
        const int i0 = it - t0;
        if (i0 < -1 || i0 >= nloc) continue;          // neither tap in this window
#if ANGLE
        const int h = kc_angle_bin(vs.a, vr.a, ihd, hmax_rad);
        if (h < 0) continue;
        const ACC v = (ACC)model[(size_t)h * npts + ip];
#else
        const ACC v = (ACC)mh[ip];
#endif
        const float w = t - (float)it;
        if (i0 >= 0)        atomicAdd(&trace[i0],     v * (ACC)(1.0f - w));
        if (i0 + 1 < nloc)  atomicAdd(&trace[i0 + 1], v * (ACC)w);
    }
    __syncthreads();

    for (int i = tid; i < nloc; i += FBLOCK) o[i] = (float)trace[i];
}
"""
