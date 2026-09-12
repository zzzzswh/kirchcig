// CPU emulation of the kirchcig CUDA kernels: one std::thread per CUDA thread,
// a generation barrier for __syncthreads, CAS-based atomics. Blocks run
// sequentially. Used only to validate kernel logic without a GPU.
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__
#define __restrict__ __restrict
#define __launch_bounds__(x)
#define __align__(n) alignas(n)
#define __CUDA_ARCH__ 500   // exercise the CAS fallback for double atomicAdd

struct dim3s { unsigned x, y, z; };
thread_local dim3s threadIdx;
static dim3s blockIdx, blockDim, gridDim;

struct Barrier {
    std::mutex m; std::condition_variable cv; unsigned n, count = 0, gen = 0;
    explicit Barrier(unsigned n_) : n(n_) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m);
        unsigned g = gen;
        if (++count == n) { count = 0; ++gen; cv.notify_all(); return; }
        cv.wait(lk, [&] { return g != gen; });
    }
};
static Barrier* g_barrier = nullptr;
static inline void __syncthreads() { g_barrier->wait(); }

static inline unsigned long long atomicCAS(unsigned long long* p, unsigned long long cmp, unsigned long long val) {
    __atomic_compare_exchange_n(p, &cmp, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return cmp;  // holds the old value either way
}
static inline float atomicAdd(float* p, float v) {
    uint32_t* u = (uint32_t*)p; uint32_t old = *u, assumed; float f;
    do { assumed = old; std::memcpy(&f, &assumed, 4); f += v; uint32_t nv; std::memcpy(&nv, &f, 4);
         old = assumed; __atomic_compare_exchange_n(u, &old, nv, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    } while (old != assumed);
    std::memcpy(&f, &old, 4); return f;
}
// round-to-nearest intrinsics: plain ops here (-std=c++17 => -ffp-contract=off, no FMA on baseline x86-64)
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline long long __double_as_longlong(double d) { long long l; std::memcpy(&l, &d, 8); return l; }
static inline double __longlong_as_double(long long l) { double d; std::memcpy(&d, &l, 8); return d; }
using std::min;

#include "kernel_body.inc"   // the CUDA_SOURCE string, verbatim

// dynamic shared memory definitions (extern __shared__ in the kernels)
ACC acc[NH * BLOCK];
facc_t trace[TCHUNK_MAX];

template <class F>
static void launch(unsigned gx, unsigned gy, unsigned block, F body) {
    gridDim = {gx, gy, 1}; blockDim = {block, 1, 1};
    for (unsigned by = 0; by < gy; ++by) for (unsigned bx = 0; bx < gx; ++bx) {
        blockIdx = {bx, by, 0};
        Barrier bar(block); g_barrier = &bar;
        std::vector<std::thread> th; th.reserve(block);
        for (unsigned t = 0; t < block; ++t)
            th.emplace_back([&, t] { threadIdx = {t, 0, 0}; body(); });
        for (auto& x : th) x.join();
    }
}

// Launchers: the Python side supplies the grid exactly as it would to CuPy.
extern "C" void launch_kirch_adjoint(unsigned gx, unsigned gy, unsigned block,
        const void* data, const void* tab_s, const void* tab_r, const void* grd_s, const void* grd_r,
        const int* hbin, const float* aaf, void* out, int ns, int nr, int nt, int npts, int s_per_split,
        int npad, int pad, int aa_max, float idt, float ihd, float hmax_rad,
        float aa_factor, float dxdt, float dzdt) {
    if (block != BLOCK) { fprintf(stderr, "block mismatch %u != %d\n", block, BLOCK); abort(); }
    launch(gx, gy, BLOCK, [&] {
        kirch_adjoint((const din_t*)data, (const tab_t*)tab_s, (const tab_t*)tab_r,
                      (const grad_t*)grd_s, (const grad_t*)grd_r, hbin, aaf,
                      (OUT*)out, ns, nr, nt, npts, s_per_split, npad, pad, aa_max,
                      idt, ihd, hmax_rad, aa_factor, dxdt, dzdt);
    });
}
extern "C" void launch_kirch_forward(unsigned gx, unsigned gy, unsigned block,
        const float* model, const void* tab_s, const void* tab_r, const void* grd_s, const void* grd_r,
        const int* hbin, const float* aaf, void* data, int ns, int nr, int nt, int npts, int tchunk,
        int npad, int pad, int aa_max, float idt, float ihd, float hmax_rad,
        float aa_factor, float dxdt, float dzdt) {
    if (block != FBLOCK) { fprintf(stderr, "fblock mismatch %u != %d\n", block, FBLOCK); abort(); }
    if (tchunk > TCHUNK_MAX) { fprintf(stderr, "tchunk too large\n"); abort(); }
    launch(gx, gy, FBLOCK, [&] {
        kirch_forward(model, (const tab_t*)tab_s, (const tab_t*)tab_r,
                      (const grad_t*)grd_s, (const grad_t*)grd_r, hbin, aaf,
                      (dout_t*)data, ns, nr, nt, npts, tchunk, npad, pad, aa_max,
                      idt, ihd, hmax_rad, aa_factor, dxdt, dzdt);
    });
}
