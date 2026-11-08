// src/prefill/wmma_gemm.cu - RDNA3 WMMA FP16 & BF16 GEMM for Strata prefill (gfx1100).
//
// Computes Y[t, n] = beta * Y[t, n] + sum_k W[n, k] * X[t, k] using RDNA3 WMMA intrinsics:
//   * X is T x K row-major (leading dim K), fp16 or bf16
//   * W is N x K row-major (leading dim K), fp16 or bf16
//   * Y is T x N row-major with leading dimension ldy >= N, fp32
//
// Hardware: AMD RDNA3 (gfx1100, e.g. RX 7900 XTX)
//   * v_wmma_f32_16x16x16_f16_w32 intrinsic (__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
//   * v_wmma_f32_16x16x16_bf16_w32 intrinsic (__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
//   * Wave32 doubled input fragment: lane t (lane_lo = t & 15) holds row lane_lo of A (16 elements along K)
//     and column lane_lo of B (16 elements along K). Lanes 16..31 duplicate lanes 0..15.
//   * Wave32 C output mapping: lane t holds column lane_lo of 16x16 output tile,
//     with 8 elements alternating rows: row m = 2*i + lane_hi (lane_hi = t >> 4).
//   * Tile variants:
//       1. gemm_wmma_64x64_4w: 4 waves (128 threads), 64T x 64N x 16K tile, double-buffered LDS for W.
//       2. gemm_wmma_16x16_1w: 1 wave (32 threads), 16T x 16N x 16K tile, zero LDS/barriers, for small T/N.

#include "wmma_gemm.h"
#include <algorithm>
#include <mutex>
#include <map>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <cstdlib>
#include "strata/kernels/gfx_arch.hpp"

#include <atomic>
#include <cstring>

// STRATA_WMMA_GFX11 is defined by the BUILD (CMakeLists.txt, from CMAKE_HIP_ARCHITECTURES), not inferred
// from compiler macros: measured on this toolchain the HOST pass of a HIP compile does not define
// __gfx1100__ but does define __HIP_DEVICE_COMPILE__, so a compiler-macro guard here silently selected the
// "return false" stub at the bottom of this file for the very symbol the engine links - the WMMA path then
// never ran, while the same file compiled with hipcc (as the probes do) took the real branch.  One
// build-defined macro is uniform across the host and device passes.
#if defined(STRATA_WMMA_GFX11)

using v8fp32 = float __attribute__((ext_vector_type(8)));

// Does the CURRENT device run the gfx11 WMMA kernels of this file?  Exactly the targets whose intrinsics are selected
// below (gfx1100 / 1101 / 1102 / 1150 / 1151): a gfx11 part outside the list (gfx1103, gfx1152) has no device code
// here, so it takes the hipBLASLt / hipBLAS paths instead of running a kernel that is not there.  Cached per device
// (hipGetDeviceProperties is not cheap, and a mixed-GPU box has more than one answer).
static bool strata_wmma_gfx11_device() {
    static std::atomic<int> cache[64];   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return false;
    int c = cache[dev].load(std::memory_order_acquire);
    if (c == 0) {
        hipDeviceProp_t prop{};
        const bool ok = hipGetDeviceProperties(&prop, dev) == hipSuccess && strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName);
        c = ok ? 1 : 2;
        cache[dev].store(c, std::memory_order_release);
    }
    return c == 1;
}

template <typename ElemT>
struct WmmaTraits;

template <>
struct WmmaTraits<_Float16> {
    using vec_t = _Float16 __attribute__((ext_vector_type(16)));
    __device__ static inline v8fp32 mma(vec_t a, vec_t b, v8fp32 c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
        (void) a; (void) b; (void) c;
        __builtin_trap();   // a device pass without the gfx11 intrinsic: the runtime gate never launches it, and a stray launch must not return zeros
#endif
    }
};

template <>
struct WmmaTraits<__bf16> {
    using vec_t = __bf16 __attribute__((ext_vector_type(16)));
    __device__ static inline v8fp32 mma(vec_t a, vec_t b, v8fp32 c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
        (void) a; (void) b; (void) c;
        __builtin_trap();   // see above
#endif
    }
};

// ===========================================================================
// Variant 1: 16x16_1w - Single-wave kernel for small T / N.
// Zero LDS, zero barrier synchronization, full register residency.
// ===========================================================================
template <typename ElemT>
__global__ void gemm_wmma_16x16_1w(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    float* __restrict__ Y,
    int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {

    using vec_t = typename WmmaTraits<ElemT>::vec_t;
    const int m_tile = blockIdx.y * 16;
    const int n_tile = blockIdx.x * 16;
    if (m_tile >= T || n_tile >= N) return;

    const int lane = threadIdx.x;   // 0..31
    const int lane_lo = lane & 15;  // 0..15
    const int lane_hi = lane >> 4;  // 0 or 1

    v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const int m_row = m_tile + lane_lo;
    const int n_row = n_tile + lane_lo;

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        vec_t a_frag, b_frag;

        if (m_row < T) {
            __builtin_memcpy(&a_frag, X + (int64_t)m_row * K + k_tile, sizeof(a_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a_frag[i] = 0;
        }

        if (n_row < N) {
            __builtin_memcpy(&b_frag, W + (int64_t)n_row * K + k_tile, sizeof(b_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b_frag[i] = 0;
        }

        c_acc = WmmaTraits<ElemT>::mma(a_frag, b_frag, c_acc);
    }

    const int out_n = n_tile + lane_lo;
    if (out_n < N) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_m = m_tile + 2 * i + lane_hi;
            if (out_m < T) {
                float* dst = Y + (int64_t)out_m * ldy + out_n;
                if (beta == 0.0f) {
                    *dst = c_acc[i];
                } else {
                    *dst = beta * (*dst) + c_acc[i];
                }
            }
        }
    }
}

// ===========================================================================
// Variant 2: 64x64_4w - 4 waves per block (128 threads), 64T x 64N tile.
// Double-buffered LDS tile for W (4 KB LDS total), cooperative 128-bit global loads.
// ===========================================================================
template <typename ElemT>
__global__ void gemm_wmma_64x64_4w(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    float* __restrict__ Y,
    int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {

    using vec_t = typename WmmaTraits<ElemT>::vec_t;
    const int m_tile = blockIdx.y * 64;
    const int n_tile = blockIdx.x * 64;
    if (m_tile >= T || n_tile >= N) return;

    const int tid = threadIdx.x;   // 0..127
    const int wave_id = tid >> 5;  // 0..3
    const int lane = tid & 31;     // 0..31
    const int lane_lo = lane & 15; // 0..15
    const int lane_hi = lane >> 4; // 0 or 1

    // 4 accumulators per wave covering 4 x 16 N-subtiles
    v8fp32 c_acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc3 = {0, 0, 0, 0, 0, 0, 0, 0};

    // Double-buffered LDS tile: 64 rows of N x 16 elements of K (2 KB per buffer)
    alignas(16) __shared__ ElemT b_lds[2][64][16];

    // Thread mapping for cooperative loading of W into LDS (128 threads load 64x16 elements)
    // Each thread loads 8 halfs (16 bytes = uint4)
    const int row_in_tile = tid >> 1;     // 0..63
    const int k_sub = (tid & 1) << 3;     // 0 or 8
    const int actual_n = n_tile + row_in_tile;

    auto load_w_into_lds = [&](int buf, int k_tile) {
        const int actual_k = k_tile + k_sub;
        uint4 val = {0, 0, 0, 0};
        if (actual_n < N && actual_k < K) {
            const void* ptr = reinterpret_cast<const void*>(W + (int64_t)actual_n * K + actual_k);
            val = *reinterpret_cast<const uint4*>(ptr);
        }
        *reinterpret_cast<uint4*>(&b_lds[buf][row_in_tile][k_sub]) = val;
    };

    // Pre-fill buffer 0
    load_w_into_lds(0, 0);
    __syncthreads();

    const int m_row = m_tile + wave_id * 16 + lane_lo;
    int cur_buf = 0;

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        const int next_buf = 1 - cur_buf;
        const int k_next = k_tile + 16;

        // Prefetch next K-tile of W into LDS
        if (k_next < K) {
            load_w_into_lds(next_buf, k_next);
        }

        // Load A fragment from X for current wave's 16 M-rows
        vec_t a_frag;
        if (m_row < T) {
            __builtin_memcpy(&a_frag, X + (int64_t)m_row * K + k_tile, sizeof(a_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a_frag[i] = 0;
        }

        // Read B fragments from LDS and compute WMMA
        vec_t b_frag0, b_frag1, b_frag2, b_frag3;
        __builtin_memcpy(&b_frag0, &b_lds[cur_buf][0 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag1, &b_lds[cur_buf][16 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag2, &b_lds[cur_buf][32 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag3, &b_lds[cur_buf][48 + lane_lo][0], sizeof(vec_t));

        c_acc0 = WmmaTraits<ElemT>::mma(a_frag, b_frag0, c_acc0);
        c_acc1 = WmmaTraits<ElemT>::mma(a_frag, b_frag1, c_acc1);
        c_acc2 = WmmaTraits<ElemT>::mma(a_frag, b_frag2, c_acc2);
        c_acc3 = WmmaTraits<ElemT>::mma(a_frag, b_frag3, c_acc3);

        __syncthreads();
        cur_buf = next_buf;
    }

    // Store C to Y (coalesced 64-byte writes per wave half)
    const int m_tile_wave = m_tile + wave_id * 16;
    auto store_acc = [&](const v8fp32& acc, int n_base) {
        const int out_n = n_base + lane_lo;
        if (out_n < N) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int out_m = m_tile_wave + 2 * i + lane_hi;
                if (out_m < T) {
                    float* dst = Y + (int64_t)out_m * ldy + out_n;
                    if (beta == 0.0f) {
                        *dst = acc[i];
                    } else {
                        *dst = beta * (*dst) + acc[i];
                    }
                }
            }
        }
    };

    store_acc(c_acc0, n_tile + 0);
    store_acc(c_acc1, n_tile + 16);
    store_acc(c_acc2, n_tile + 32);
    store_acc(c_acc3, n_tile + 48);
}

template <typename ElemT>
static inline bool strata_wmma_gemm_dispatch(const uint16_t* X, const uint16_t* W, float* Y,
                                             int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                                             void* stream) {
    if (!X || !W || !Y) return false;
    if (T <= 0 || N <= 0 || K <= 0) return false;
    // Runtime gate (review): run only on gfx11 devices, whatever this build compiled for.  The build-time
    // STRATA_WMMA_GFX11 macro controls whether the intrinsics compile; this check controls whether they run.
    if (!strata_wmma_gfx11_device()) return false;
    if (K % 16 != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    if (ldy <= 0) ldy = N;
    if (ldy < N) return false;

    hipStream_t s = static_cast<hipStream_t>(stream);

    if (T >= 32 && N >= 32) {
        dim3 block(128);
        dim3 grid((uint32_t)((N + 63) / 64), (uint32_t)((T + 63) / 64), 1);
        gemm_wmma_64x64_4w<ElemT><<<grid, block, 0, s>>>(X, W, Y, T, N, K, ldy, beta);
    } else {
        dim3 block(32);
        dim3 grid((uint32_t)((N + 15) / 16), (uint32_t)((T + 15) / 16), 1);
        gemm_wmma_16x16_1w<ElemT><<<grid, block, 0, s>>>(X, W, Y, T, N, K, ldy, beta);
    }
    return true;
}

bool strata_wmma_gemm_f16(const uint16_t* X, const uint16_t* W, float* Y,
                          int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                          void* stream) {
    return strata_wmma_gemm_dispatch<_Float16>(X, W, Y, T, N, K, ldy, beta, stream);
}

bool strata_wmma_gemm_bf16(const uint16_t* X, const uint16_t* W, float* Y,
                           int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                           void* stream) {
    return strata_wmma_gemm_dispatch<__bf16>(X, W, Y, T, N, K, ldy, beta, stream);
}

// ===========================================================================
// S23 (opt-in STRATA_PF_GEMM=1): the prompt projections' FP16 GEMM on gfx11 (gfx1151: 30-33 TFLOPS on the 16K-row
// projection shapes where the tuned hipBLASLt reaches 25-27; s23/gemm_probe4.hip).  Block 128 x BN (BN 256 or 128),
// BK 32, 8 waves of 64 x BN/4, LDS double buffer with the next stage prefetched into named registers (arrays there
// went to scratch), grouped tile order (8 row tiles sweep all column tiles: the weights stream from DRAM once per
// group).  FP32 accumulation, K in another order than hipBLASLt: rounding-level, quality-gated.
// ===========================================================================
namespace pfg {
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
constexpr int BM = 128, BK = 32, LDK = BK + 8, GM = 8;
__device__ __forceinline__ h16 frag(const _Float16* p) {
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(h16, w);
}
template <int BN, int WN>
__global__ void __launch_bounds__(256) kernel(const _Float16* __restrict__ X, const _Float16* __restrict__ W,
                                              float* __restrict__ Y, int M, int N, int K, int ldy, int accumulate,
                                              int ldx, int ldw) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int TN = WN / 16, NB = BN / 64;
    __shared__ __align__(16) _Float16 sA[2][BM][LDK];
    __shared__ __align__(16) _Float16 sB[2][BN][LDK];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 2, sq = tid & 3;
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(W + (size_t) min(n0 + r, N - 1) * ldw + k0 + 8 * sq);
    };
    f8 acc[4][TN];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < TN; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 ra0 = ldA(0, sr), ra1 = ldA(0, sr + 64), rb0 = ldB(0, sr), rb1 = ldB(0, sr + 64), rb2 = rb0, rb3 = rb0;
    if constexpr (NB > 2) { rb2 = ldB(0, sr + 128); rb3 = ldB(0, sr + 192); }
    int buf = 0;
    *reinterpret_cast<uint4*>(&sA[0][sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[0][sr + 64][8 * sq]) = ra1;
    *reinterpret_cast<uint4*>(&sB[0][sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[0][sr + 64][8 * sq]) = rb1;
    if constexpr (NB > 2) { *reinterpret_cast<uint4*>(&sB[0][sr + 128][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[0][sr + 192][8 * sq]) = rb3; }
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) {
            ra0 = ldA(k0 + BK, sr); ra1 = ldA(k0 + BK, sr + 64); rb0 = ldB(k0 + BK, sr); rb1 = ldB(k0 + BK, sr + 64);
            if constexpr (NB > 2) { rb2 = ldB(k0 + BK, sr + 128); rb3 = ldB(k0 + BK, sr + 192); }
        }
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            h16 a[4], bb[TN];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = frag(&sA[buf][64 * wm + 16 * i + l16][ks]);
#pragma unroll
            for (int j = 0; j < TN; ++j) bb[j] = frag(&sB[buf][WN * wn + 16 * j + l16][ks]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], bb[j], acc[i][j]);
        }
        if (more) {
            const int nb = buf ^ 1;
            *reinterpret_cast<uint4*>(&sA[nb][sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[nb][sr + 64][8 * sq]) = ra1;
            *reinterpret_cast<uint4*>(&sB[nb][sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[nb][sr + 64][8 * sq]) = rb1;
            if constexpr (NB > 2) { *reinterpret_cast<uint4*>(&sB[nb][sr + 128][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[nb][sr + 192][8 * sq]) = rb3; }
            __syncthreads();
            buf = nb;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + WN * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                if (m < M) {
                    float* y = Y + (size_t) m * ldy + n;
                    *y = accumulate ? *y + acc[i][j][e] : acc[i][j][e];
                }
            }
        }
#endif
}
// The 128 x 256 kernel with a 64-k LDS tile (single buffer, the next tile in 12 named staging registers, 2 barriers
// per tile: half the loop trips and barriers per k of the BK 32 double buffer). Each accumulator takes the same
// WMMAs in the same k order as `kernel`: the same bits (s23/gemm_probe11.hip, memcmp). gfx1151, M 16384, both
// operands padded: 38.0 -> 41.3 TFLOPS (N 2560 K 6144), 38.9 -> 40.9 (N 12288 K 2560), 40.3 -> 40.4 (N 10240).
// K a multiple of 64.
__global__ void __launch_bounds__(256) kernel_bk64(const _Float16* __restrict__ X, const _Float16* __restrict__ W,
                                                   float* __restrict__ Y, int M, int N, int K, int ldy, int accumulate,
                                                   int ldx, int ldw) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int BN = 256, WN = 64, BK64 = 64, LDK64 = BK64 + 8;
    __shared__ __align__(16) _Float16 sA[BM][LDK64];
    __shared__ __align__(16) _Float16 sB[BN][LDK64];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 3, sq = tid & 7;            // 32 rows per pass, 8 x 8 halves per row
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(W + (size_t) min(n0 + r, N - 1) * ldw + k0 + 8 * sq);
    };
    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7;
    auto load = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
        rb4 = ldB(k0, sr + 128); rb5 = ldB(k0, sr + 160); rb6 = ldB(k0, sr + 192); rb7 = ldB(k0, sr + 224);
    };
    auto store = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[sr + 96][8 * sq]) = rb3;
        *reinterpret_cast<uint4*>(&sB[sr + 128][8 * sq]) = rb4; *reinterpret_cast<uint4*>(&sB[sr + 160][8 * sq]) = rb5;
        *reinterpret_cast<uint4*>(&sB[sr + 192][8 * sq]) = rb6; *reinterpret_cast<uint4*>(&sB[sr + 224][8 * sq]) = rb7;
    };
    f8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    load(0);
    store();
    __syncthreads();
    const int ar = 64 * wm + l16, br = WN * wn + l16;
    for (int k0 = 0; k0 < K; k0 += BK64) {
        const bool more = k0 + BK64 < K;
        if (more) load(k0 + BK64);
#pragma unroll
        for (int ks = 0; ks < BK64; ks += 16) {
            const h16 a0 = frag(&sA[ar][ks]), a1 = frag(&sA[ar + 16][ks]), a2 = frag(&sA[ar + 32][ks]),
                      a3 = frag(&sA[ar + 48][ks]);
            const h16 b0 = frag(&sB[br][ks]), b1 = frag(&sB[br + 16][ks]), b2 = frag(&sB[br + 32][ks]),
                      b3 = frag(&sB[br + 48][ks]);
#define PFG_ROW(i, a)                                                                                                 \
    acc[i][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, acc[i][0]);                                         \
    acc[i][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, acc[i][1]);                                         \
    acc[i][2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b2, acc[i][2]);                                         \
    acc[i][3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b3, acc[i][3]);
            PFG_ROW(0, a0) PFG_ROW(1, a1) PFG_ROW(2, a2) PFG_ROW(3, a3)
#undef PFG_ROW
        }
        if (more) {
            __syncthreads();   // every wave is done reading the tile
            store();
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + WN * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                if (m < M) {
                    float* y = Y + (size_t) m * ldy + n;
                    *y = accumulate ? *y + acc[i][j][e] : acc[i][j][e];
                }
            }
        }
#endif
}
// S23 (STRATA_PF_HCDOWN=1): the hyper-connection read's down (N nd = 320) and inject (N ni = 4) projections of xn16
// (BF16, T x 10240, token stride ldx: 10240 + 64 avoids the 4 KB-multiple stride) as ONE GEMM - weight rows
// [w_down; w_inject] read from the two tensors, the epilogue splitting the columns into lo (T x nd) and inj (T x ni),
// so xn16 is read once.  128 x 128 blocks, the BK 64 single-buffer scheme of kernel_bk64.  s23/hcdown_probe.hip,
// 16K tokens: 3.60 / 3.77 ms (hipBLASLt in the engine: down 6.78 + inject 1.81).  FP32 accumulation in another k
// order than hipBLASLt: rounding-level.
typedef __bf16 hb16 __attribute__((ext_vector_type(16)));
__device__ __forceinline__ hb16 bfrag(const uint16_t* p) {
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(hb16, w);
}
__global__ void __launch_bounds__(256) kernel_hcdown(const uint16_t* __restrict__ X, int ldx,
                                                     const uint16_t* __restrict__ Wd, const uint16_t* __restrict__ Wi,
                                                     int nd, int ni, float* __restrict__ lo, float* __restrict__ inj,
                                                     int M, int K) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int BN = 128, WN = 32, TN = 2, BK64 = 64, LDK64 = BK64 + 8;
    __shared__ __align__(16) uint16_t sA[BM][LDK64];
    __shared__ __align__(16) uint16_t sB[BN][LDK64];
    const int N = nd + ni;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 3, sq = tid & 7;
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        const int n = min(n0 + r, N - 1);
        const uint16_t* row = n < nd ? Wd + (size_t) n * K : Wi + (size_t) (n - nd) * K;
        return *reinterpret_cast<const uint4*>(row + k0 + 8 * sq);
    };
    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3;
    auto load = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
    };
    auto store = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[sr + 96][8 * sq]) = rb3;
    };
    f8 acc[4][TN];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    load(0);
    store();
    __syncthreads();
    const int ar = 64 * wm + l16, br = WN * wn + l16;
    for (int k0 = 0; k0 < K; k0 += BK64) {
        const bool more = k0 + BK64 < K;
        if (more) load(k0 + BK64);
#pragma unroll
        for (int ks = 0; ks < BK64; ks += 16) {
            const hb16 a0 = bfrag(&sA[ar][ks]), a1 = bfrag(&sA[ar + 16][ks]), a2 = bfrag(&sA[ar + 32][ks]),
                       a3 = bfrag(&sA[ar + 48][ks]);
            const hb16 b0 = bfrag(&sB[br][ks]), b1 = bfrag(&sB[br + 16][ks]);
            acc[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b0, acc[0][0]);
            acc[1][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b0, acc[1][0]);
            acc[2][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a2, b0, acc[2][0]);
            acc[3][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a3, b0, acc[3][0]);
            acc[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b1, acc[0][1]);
            acc[1][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b1, acc[1][1]);
            acc[2][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a2, b1, acc[2][1]);
            acc[3][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a3, b1, acc[3][1]);
        }
        if (more) {
            __syncthreads();
            store();
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + WN * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                if (m < M) {
                    if (n < nd) lo[(size_t) m * nd + n] = acc[i][j][e];
                    else inj[(size_t) m * ni + (n - nd)] = acc[i][j][e];
                }
            }
        }
#endif
}

// S23/S (STRATA_HCD_EXACT=1): the HC down projection (N 320, K 10240, BF16 in, FP32 out) bitwise equal to hipBLASLt
// solution 1176 / 1177 (MT32x96 / MT96x96, StaggerU 32, stride 256 B): every output is the k tiles of 32 accumulated in
// order, 16 per WMMA, starting at k tile 4 * (token tile of 96 % 32) and wrapping (hcd_bit / hcd_perm probes: 0 of
// 5.2 M outputs differ at T 16384).  96 x 160 blocks (two per token tile), 6 waves of 32 x 80.
typedef uint32_t hx_u4 __attribute__((ext_vector_type(4)));
constexpr int HX_N = 320, HX_BN = 160, HX_BM = 96, HX_BK = 64, HX_LDK = HX_BK + 8, HX_NT = 192, HX_K = 10240;
__device__ __forceinline__ hb16 hx_frag(const uint16_t* p) {
    const hx_u4 a = *reinterpret_cast<const hx_u4*>(p), b = *reinterpret_cast<const hx_u4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(hb16, w);
}
__global__ void __launch_bounds__(HX_NT) kernel_hcd_exact(const uint16_t* __restrict__ X, int ldx,
                                                          const uint16_t* __restrict__ W, float* __restrict__ Y, int M) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int K = HX_K, BN = HX_BN, BMT = HX_BM, BK = HX_BK, NT = HX_NT;
    __shared__ __align__(16) uint16_t sA[BMT][HX_LDK];
    __shared__ __align__(16) uint16_t sB[BN][HX_LDK];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave % 3, wn = wave / 3, nbase = BN * (blockIdx.x & 1);
    const int tt = blockIdx.x >> 1, m0 = tt * BMT;
    constexpr int nslab = K / BK;
    const int rot = (2 * (tt & 31)) % nslab;   // 4 k tiles of 32 = 2 slabs of 64
    hx_u4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, rb4, rb5, rb6;
#define HX_LDA(v, i) { const int idx = tid + NT * (i), r = idx >> 3, q = idx & 7; v = *reinterpret_cast<const hx_u4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * q); }
#define HX_LDB(v, i) { const int idx = min(tid + NT * (i), BN * 8 - 1), r = idx >> 3, q = idx & 7; v = *reinterpret_cast<const hx_u4*>(W + (size_t) (nbase + r) * K + k0 + 8 * q); }
#define HX_STA(v, i) { const int idx = tid + NT * (i); *reinterpret_cast<hx_u4*>(&sA[idx >> 3][8 * (idx & 7)]) = v; }
#define HX_STB(v, i) { const int idx = tid + NT * (i); if (idx < BN * 8) *reinterpret_cast<hx_u4*>(&sB[idx >> 3][8 * (idx & 7)]) = v; }
#define HX_LOAD(k0v) { const int k0 = (k0v); HX_LDA(ra0, 0) HX_LDA(ra1, 1) HX_LDA(ra2, 2) HX_LDA(ra3, 3) HX_LDB(rb0, 0) HX_LDB(rb1, 1) HX_LDB(rb2, 2) HX_LDB(rb3, 3) HX_LDB(rb4, 4) HX_LDB(rb5, 5) HX_LDB(rb6, 6) }
#define HX_STORE() { HX_STA(ra0, 0) HX_STA(ra1, 1) HX_STA(ra2, 2) HX_STA(ra3, 3) HX_STB(rb0, 0) HX_STB(rb1, 1) HX_STB(rb2, 2) HX_STB(rb3, 3) HX_STB(rb4, 4) HX_STB(rb5, 5) HX_STB(rb6, 6) }
    f8 acc[2][5];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 5; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    HX_LOAD(rot * BK);
    HX_STORE();
    __syncthreads();
    const int ar = 32 * wm + l16, br = 80 * wn + l16;
    for (int s = 0; s < nslab; ++s) {
        const bool more = s + 1 < nslab;
        if (more) HX_LOAD(((s + 1 + rot) % nslab) * BK);
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            const hb16 a0 = hx_frag(&sA[ar][ks]), a1 = hx_frag(&sA[ar + 16][ks]);
            const hb16 b0 = hx_frag(&sB[br][ks]), b1 = hx_frag(&sB[br + 16][ks]), b2 = hx_frag(&sB[br + 32][ks]),
                       b3 = hx_frag(&sB[br + 48][ks]), b4 = hx_frag(&sB[br + 64][ks]);
#define HX_W2(j, bj) acc[0][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, bj, acc[0][j]); acc[1][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, bj, acc[1][j]);
            HX_W2(0, b0) HX_W2(1, b1) HX_W2(2, b2) HX_W2(3, b3) HX_W2(4, b4)
#undef HX_W2
        }
        if (more) {
            __syncthreads();
            HX_STORE();
            __syncthreads();
        }
    }
#undef HX_LDA
#undef HX_LDB
#undef HX_STA
#undef HX_STB
#undef HX_LOAD
#undef HX_STORE
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 5; ++j) {
            const int n = nbase + 80 * wn + 16 * j + l16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 32 * wm + 16 * i + 2 * e + hi;
                if (m < M) Y[(size_t) m * HX_N + n] = acc[i][j][e];
            }
        }
#endif
}
}  // namespace pfg

bool strata_pf_hcdown_bf16(const uint16_t* X, int64_t ldx, const uint16_t* Wd, const uint16_t* Wi, int64_t nd,
                           int64_t ni, float* lo, float* inj, int64_t T, int64_t K, void* stream) {
    if (!X || !Wd || !Wi || !lo || !inj || T < 64 || nd < 1 || ni < 0 || K % 64 != 0 || ldx < K || ldx % 8 != 0) return false;
    if (T > (1LL << 30) || K > (1LL << 30) || ldx > (1LL << 30)) return false;
    if (!strata_wmma_gfx11_device()) return false;
    const int64_t mt = (T + pfg::BM - 1) / pfg::BM;
    const unsigned grid = (unsigned) (mt * ((nd + ni + 127) / 128));
    pfg::kernel_hcdown<<<grid, 256, 0, static_cast<hipStream_t>(stream)>>>(X, (int) ldx, Wd, Wi, (int) nd, (int) ni,
                                                                            lo, inj, (int) T, (int) K);
    return hipGetLastError() == hipSuccess;
}

bool strata_pf_hcdown_exact_bf16(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N,
                                 int64_t K, void* stream) {
    if (!X || !W || !Y || T < 1 || N != pfg::HX_N || K != pfg::HX_K || ldx < K || ldx % 8 != 0 || ldx > (1LL << 30) ||
        T > (1LL << 30))
        return false;
    if (!strata_wmma_gfx11_device()) return false;
    const unsigned grid = (unsigned) (2 * ((T + pfg::HX_BM - 1) / pfg::HX_BM));
    pfg::kernel_hcd_exact<<<grid, pfg::HX_NT, 0, static_cast<hipStream_t>(stream)>>>(X, (int) ldx, W, Y, (int) T);
    return hipGetLastError() == hipSuccess;
}

bool strata_pf_gemm_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                        float beta, void* stream) {
    return strata_pf_gemm_f16_ld(X, K, W, K, Y, T, N, K, ldy, beta, stream);
}

bool strata_pf_gemm_f16_ld(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw, float* Y, int64_t T,
                           int64_t N, int64_t K, int64_t ldy, float beta, void* stream) {
    if (!X || !W || !Y || T < 64 || N < 512 || K % pfg::BK != 0 || K < pfg::BK) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    if (ldy <= 0) ldy = N;
    if (ldx < K || ldw < K || ldx % 8 != 0 || ldw % 8 != 0 || ldx > (1LL << 30) || ldw > (1LL << 30)) return false;
    if (ldy < N || T > (1LL << 30) || N > (1LL << 30) || K > (1LL << 30)) return false;
    if (!strata_wmma_gfx11_device()) return false;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const int acc = beta == 1.0f ? 1 : 0;
    const int64_t mt = (T + pfg::BM - 1) / pfg::BM;
    // STRATA_PF_BK64=0: the BK 32 kernel for the wide shapes too (the A/B; the same bits)
    static const bool bk64 = [] { const char* v = std::getenv("STRATA_PF_BK64"); return !v || v[0] != '0'; }();
    if (N >= 1024 && bk64 && K % 64 == 0) {
        const unsigned grid = (unsigned) (mt * ((N + 255) / 256));
        pfg::kernel_bk64<<<grid, 256, 0, s>>>((const _Float16*) X, (const _Float16*) W, Y, (int) T, (int) N, (int) K,
                                              (int) ldy, acc, (int) ldx, (int) ldw);
    } else if (N >= 1024) {
        const unsigned grid = (unsigned) (mt * ((N + 255) / 256));
        pfg::kernel<256, 64><<<grid, 256, 0, s>>>((const _Float16*) X, (const _Float16*) W, Y, (int) T, (int) N, (int) K,
                                                  (int) ldy, acc, (int) ldx, (int) ldw);
    } else {
        const unsigned grid = (unsigned) (mt * ((N + 127) / 128));
        pfg::kernel<128, 32><<<grid, 256, 0, s>>>((const _Float16*) X, (const _Float16*) W, Y, (int) T, (int) N, (int) K,
                                                  (int) ldy, acc, (int) ldx, (int) ldw);
    }
    return hipGetLastError() == hipSuccess;
}

// ===========================================================================
// SKINNY (T <= 128 rows, the short prompt read): Y[t, n] (+)= sum_k W[n, k] X[t, k] with the weights streamed ONCE.
// A short read's projections are weight-bound (every weight byte meets at most 128 tokens), and the tiled kernels
// above and hipBLASLt leave most of the bandwidth unused at a handful of rows - or launch 20 waves for a 320-column
// product with K = 10240.  Here a wave owns 16 output columns and every row (MT 16-row tiles, the B fragment reused
// for all of them), reads its W rows KU x 32 bytes at a time per lane, and the K range is split across blocks until
// the grid has enough waves to keep the memory busy; the splits' partial sums go to a workspace and one kernel adds
// them in split order (deterministic).  FP32 accumulation, K in its own order: rounding-level against hipBLASLt.
// ===========================================================================
namespace skinny {
constexpr int WPB = 4;     // waves per block
constexpr int KU = 4;      // 16-wide K steps per W load (128 bytes of a row per lane)

template <typename ElemT, int MT>
__global__ void __launch_bounds__(32 * WPB) kernel(const uint16_t* __restrict__ X, int ldx, const uint16_t* __restrict__ W,
                                                   int ldw, float* __restrict__ Y, int ldy, float* __restrict__ part,
                                                   int T, int N, int K, int kchunk, float beta) {
    using vec_t = typename WmmaTraits<ElemT>::vec_t;
    const int lane = threadIdx.x & 31, lane_lo = lane & 15, lane_hi = lane >> 4;
    const int n0 = (blockIdx.x * WPB + (threadIdx.x >> 5)) * 16;
    if (n0 >= N) return;
    const int kb = blockIdx.y * kchunk, ke = min(K, kb + kchunk);
    v8fp32 acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = v8fp32{0, 0, 0, 0, 0, 0, 0, 0};
    const int n_row = n0 + lane_lo;
    const bool n_ok = n_row < N;
    const uint16_t* wrow = W + (int64_t) (n_ok ? n_row : 0) * ldw;
    for (int k = kb; k < ke; k += 16 * KU) {
        vec_t b[KU];
#pragma unroll
        for (int u = 0; u < KU; ++u) {
            if (n_ok && k + 16 * u < ke) __builtin_memcpy(&b[u], wrow + k + 16 * u, sizeof(vec_t));
            else {
#pragma unroll
                for (int i = 0; i < 16; ++i) b[u][i] = 0;
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int t = m * 16 + lane_lo;
            const uint16_t* xrow = X + (int64_t) (t < T ? t : 0) * ldx;
#pragma unroll
            for (int u = 0; u < KU; ++u) {
                vec_t a;
                if (t < T && k + 16 * u < ke) __builtin_memcpy(&a, xrow + k + 16 * u, sizeof(vec_t));
                else {
#pragma unroll
                    for (int i = 0; i < 16; ++i) a[i] = 0;
                }
                acc[m] = WmmaTraits<ElemT>::mma(a, b[u], acc[m]);
            }
        }
    }
    if (!n_ok) return;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int t = m * 16 + 2 * i + lane_hi;
            if (t >= T) continue;
            if (part) {
                part[((int64_t) blockIdx.y * T + t) * N + n_row] = acc[m][i];
            } else {
                float* dst = Y + (int64_t) t * ldy + n_row;
                *dst = beta == 0.0f ? acc[m][i] : beta * *dst + acc[m][i];
            }
        }
    }
}

__global__ void reduce(const float* __restrict__ part, int splits, float* __restrict__ Y, int ldy, int T, int N,
                       float beta) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * N) return;
    const int t = (int) (i / N), n = (int) (i % N);
    float s = 0.0f;
    for (int k = 0; k < splits; ++k) s += part[((int64_t) k * T + t) * N + n];
    float* dst = Y + (int64_t) t * ldy + n;
    *dst = beta == 0.0f ? s : beta * *dst + s;
}

// the split partials' workspace, per stream (the prompt path's streams; grown once, never inside a capture)
float* workspace(hipStream_t s, size_t bytes) {
    static std::mutex mu;
    static std::map<hipStream_t, std::pair<float*, size_t>> ws;
    std::lock_guard<std::mutex> lk(mu);
    auto& e = ws[s];
    if (e.second < bytes) {
        if (e.first) { hipStreamSynchronize(s); hipFree(e.first); }
        e = {nullptr, 0};
        const size_t want = std::max<size_t>(bytes, (size_t) 16 << 20);
        if (hipMalloc((void**) &e.first, want) != hipSuccess) { e.first = nullptr; return nullptr; }
        e.second = want;
    }
    return e.first;
}

int max_t() {   // STRATA_SKINNY_MAX_T: the largest chunk this takes (0 = off); 128 by default
    static const int v = [] { const char* e = std::getenv("STRATA_SKINNY_MAX_T"); return e ? std::atoi(e) : 128; }();
    return std::min(v, 128);
}

template <typename ElemT>
bool run(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw, float* Y, int64_t T, int64_t N, int64_t K,
         int64_t ldy, float beta, void* stream) {
    if (!X || !W || !Y || T <= 0 || N <= 0 || K <= 0 || T > max_t() || K % 16 != 0 || ldx % 8 != 0 || ldw % 8 != 0 ||
        (beta != 0.0f && beta != 1.0f) || (reinterpret_cast<uintptr_t>(X) & 15) || (reinterpret_cast<uintptr_t>(W) & 15))
        return false;
    if (!strata_wmma_gfx11_device()) return false;
    if (ldy <= 0) ldy = N;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const int64_t waves_n = (N + 15) / 16, blocks_n = (waves_n + WPB - 1) / WPB;
    static const int64_t target = [] { const char* e = std::getenv("STRATA_SKINNY_WAVES"); return e ? std::atoll(e) : (int64_t) 640; }();
    const int64_t step = 16 * KU;
    int64_t splits = std::max<int64_t>(1, std::min<int64_t>((target + waves_n - 1) / waves_n, (K + step - 1) / step));
    int64_t kchunk = ((K + splits - 1) / splits + step - 1) / step * step;
    splits = (K + kchunk - 1) / kchunk;
    float* part = nullptr;
    if (splits > 1) {
        part = workspace(s, (size_t) (splits * T * N) * sizeof(float));
        if (!part) return false;
    }
    const dim3 grid((unsigned) blocks_n, (unsigned) splits);
    const int mt = (int) ((T + 15) / 16);
#define SK(MT) kernel<ElemT, MT><<<grid, 32 * WPB, 0, s>>>(X, (int) ldx, W, (int) ldw, Y, (int) ldy, part, (int) T, (int) N, \
                                                         (int) K, (int) kchunk, beta)
    switch (mt) {
        case 1: SK(1); break; case 2: SK(2); break; case 3: SK(3); break; case 4: SK(4); break;
        case 5: SK(5); break; case 6: SK(6); break; case 7: SK(7); break; default: SK(8); break;
    }
#undef SK
    if (splits > 1) {
        const int64_t n = T * N;
        reduce<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(part, (int) splits, Y, (int) ldy, (int) T, (int) N, beta);
    }
    return hipGetLastError() == hipSuccess;
}
}  // namespace skinny

// SKINNY on the GGUF blocks themselves (Q6_K, Q4_K, IQ4_XS, IQ4_NL - the dense projections of the IQ3_S pack): each
// lane dequantizes 64 values of its weight row per step into four FP16 B fragments, in registers (llama.cpp's
// formulas, the values the FP16 image holds), so a short read streams the native bytes - 1.9 GB for the dense set
// against 5.3 GB as FP16 - and needs no FP16 copy of the weights at all.
namespace skinny {
static __constant__ int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
using h16 = _Float16 __attribute__((ext_vector_type(16)));
__device__ __forceinline__ float ldh(const uint8_t* p) { _Float16 h; __builtin_memcpy(&h, p, 2); return (float) h; }

// Each type: kofs(g, k) - the K offsets of group g's four 16-value fragments; half(row, g, h, b) - fragments 2h and
// 2h + 1 of the group (the two halves of a wave dequantize one pair each and swap them: permlanex16 below).
struct Q6K {   // 210 bytes / 256: ql[128] qh[64] scales[16] d
    static constexpr int QK = 256, BYTES = 210;
    __device__ static void kofs(int g, int k[4]) {
        const int sb = g >> 2, sub = g & 3, n = sub >> 1, part = sub & 1;
#pragma unroll
        for (int f = 0; f < 4; ++f) k[f] = 256 * sb + 128 * n + 32 * (part + 2 * (f >> 1)) + 16 * (f & 1);
    }
    __device__ static void half(const uint8_t* row, int g, int h, h16 b[2]) {
        const int sb = g >> 2, sub = g & 3, n = sub >> 1, part = sub & 1, jj = part + 2 * h;
        const uint8_t* x = row + (int64_t) sb * BYTES;
        uint8_t ql[32], qh[32];
        int8_t sc[2];
        __builtin_memcpy(ql, x + 64 * n + 32 * part, 32);
        __builtin_memcpy(qh, x + 128 + 32 * n, 32);
        __builtin_memcpy(sc, x + 192 + 8 * n + 2 * jj, 2);
        const float d = ldh(x + 208);
        const int nsh = 4 * h, qsh = 2 * jj;
#pragma unroll
        for (int is = 0; is < 2; ++is) {
            const float ds = d * (float) sc[is];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int l = 16 * is + i;
                const int q = (int) (((ql[l] >> nsh) & 0xF) | (((qh[l] >> qsh) & 3) << 4)) - 32;
                b[is][i] = (_Float16) (ds * (float) q);
            }
        }
    }
};
struct Q4K {   // 144 bytes / 256: d dmin scales[12] qs[128]
    static constexpr int QK = 256, BYTES = 144;
    __device__ static void kofs(int g, int k[4]) {
        const int sb = g >> 2, c = g & 3;
#pragma unroll
        for (int f = 0; f < 4; ++f) k[f] = 256 * sb + 64 * c + 16 * f;
    }
    __device__ static void half(const uint8_t* row, int g, int h, h16 b[2]) {
        const int sb = g >> 2, c = g & 3, j = 2 * c + h;
        const uint8_t* x = row + (int64_t) sb * BYTES;
        uint8_t qs[32], scl[12];
        __builtin_memcpy(qs, x + 16 + 32 * c, 32);
        __builtin_memcpy(scl, x + 4, 12);
        const float d = ldh(x), dmin = ldh(x + 2);
        uint8_t sc, m;
        if (j < 4) { sc = scl[j] & 63; m = scl[j + 4] & 63; }
        else { sc = (scl[j + 4] & 0xF) | ((scl[j - 4] >> 6) << 4); m = (scl[j + 4] >> 4) | ((scl[j] >> 6) << 4); }
        const float d1 = d * sc, m1 = dmin * m;
        const int sh = 4 * h;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
            for (int i = 0; i < 16; ++i) b[hh][i] = (_Float16) (d1 * (float) ((qs[16 * hh + i] >> sh) & 0xF) - m1);
    }
};
struct Q5K {   // 176 bytes / 256: d dmin scales[12] qh[32] qs[128]
    static constexpr int QK = 256, BYTES = 176;
    __device__ static void kofs(int g, int k[4]) {
        const int sb = g >> 2, c = g & 3;
#pragma unroll
        for (int f = 0; f < 4; ++f) k[f] = 256 * sb + 64 * c + 16 * f;
    }
    __device__ static void half(const uint8_t* row, int g, int h, h16 b[2]) {
        const int sb = g >> 2, c = g & 3, j = 2 * c + h;
        const uint8_t* x = row + (int64_t) sb * BYTES;
        uint8_t qs[32], qh[32], scl[12];
        __builtin_memcpy(qs, x + 48 + 32 * c, 32);
        __builtin_memcpy(qh, x + 16, 32);
        __builtin_memcpy(scl, x + 4, 12);
        const float d = ldh(x), dmin = ldh(x + 2);
        uint8_t sc, m;
        if (j < 4) { sc = scl[j] & 63; m = scl[j + 4] & 63; }
        else { sc = (scl[j + 4] & 0xF) | ((scl[j - 4] >> 6) << 4); m = (scl[j + 4] >> 4) | ((scl[j] >> 6) << 4); }
        const float d1 = d * sc, m1 = dmin * m;
        const int sh = 4 * h, hb = 2 * c + h;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int l = 16 * hh + i;
                const int q = ((qs[l] >> sh) & 0xF) + (((qh[l] >> hb) & 1) << 4);
                b[hh][i] = (_Float16) (d1 * (float) q - m1);
            }
    }
};
struct Q80 {   // 34 bytes / 32: d qs[32]
    static constexpr int QK = 32, BYTES = 34;
    __device__ static void kofs(int g, int k[4]) {
#pragma unroll
        for (int f = 0; f < 4; ++f) k[f] = 64 * g + 16 * f;
    }
    __device__ static void half(const uint8_t* row, int g, int h, h16 b[2]) {
        const uint8_t* x = row + (int64_t) (2 * g + h) * BYTES;
        int8_t qs[32];
        __builtin_memcpy(qs, x + 2, 32);
        const float d = ldh(x);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
            for (int i = 0; i < 16; ++i) b[hh][i] = (_Float16) (d * (float) qs[16 * hh + i]);
    }
};
struct IQ4XS {   // 136 bytes / 256: d scales_h scales_l[4] qs[128]
    static constexpr int QK = 256, BYTES = 136;
    __device__ static void kofs(int g, int k[4]) {
        const int sb = g >> 2, sub = g & 3;
#pragma unroll
        for (int f = 0; f < 4; ++f) k[f] = 256 * sb + 64 * sub + 16 * f;
    }
    __device__ static void half(const uint8_t* row, int g, int h, h16 b[2]) {
        const int sb = g >> 2, sub = g & 3, ib = 2 * sub + h;
        const uint8_t* x = row + (int64_t) sb * BYTES;
        uint8_t qs[16];
        uint16_t sh;
        __builtin_memcpy(qs, x + 8 + 16 * ib, 16);
        __builtin_memcpy(&sh, x + 2, 2);
        const uint8_t sl = x[4 + sub];
        const int ls = ((sl >> (4 * h)) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = ldh(x) * (float) (ls - 32);
#pragma unroll
        for (int hi = 0; hi < 2; ++hi)
#pragma unroll
            for (int i = 0; i < 16; ++i) b[hi][i] = (_Float16) (dl * (float) kIq4nl[(qs[i] >> (4 * hi)) & 0xF]);
    }
};
struct IQ4NL {   // 18 bytes / 32: d qs[16]
    static constexpr int QK = 32, BYTES = 18;
    __device__ static void kofs(int g, int k[4]) {
#pragma unroll
        for (int f = 0; f < 4; ++f) k[f] = 64 * g + 16 * f;
    }
    __device__ static void half(const uint8_t* row, int g, int h, h16 b[2]) {
        const uint8_t* x = row + (int64_t) (2 * g + h) * BYTES;
        uint8_t qs[16];
        __builtin_memcpy(qs, x + 2, 16);
        const float d = ldh(x);
#pragma unroll
        for (int hi = 0; hi < 2; ++hi)
#pragma unroll
            for (int i = 0; i < 16; ++i) b[hi][i] = (_Float16) (d * (float) kIq4nl[(qs[i] >> (4 * hi)) & 0xF]);
    }
};

// the other half-wave's value of v (lane l <-> lane l ^ 16)
__device__ __forceinline__ uint32_t swap16(uint32_t v) {
    return (uint32_t) __builtin_amdgcn_permlanex16((int) v, (int) v, 0x76543210, 0xfedcba98, false, false);
}

template <typename Q, int MT>
__global__ void __launch_bounds__(32 * WPB) kernel_q(const uint16_t* __restrict__ X, int ldx, const uint8_t* __restrict__ W,
                                                     int64_t row_bytes, float* __restrict__ Y, int ldy,
                                                     float* __restrict__ part, int T, int N, int groups, int gchunk,
                                                     float beta) {
    const int lane = threadIdx.x & 31, lane_lo = lane & 15, lane_hi = lane >> 4;
    const int n0 = (blockIdx.x * WPB + (threadIdx.x >> 5)) * 16;
    if (n0 >= N) return;
    const int gb = blockIdx.y * gchunk, ge = min(groups, gb + gchunk);
    v8fp32 acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = v8fp32{0, 0, 0, 0, 0, 0, 0, 0};
    const int n_row = n0 + lane_lo;
    const bool n_ok = n_row < N;
    const uint8_t* wrow = W + (int64_t) (n_ok ? n_row : 0) * row_bytes;
    for (int g = gb; g < ge; ++g) {
        int k[4];
        Q::kofs(g, k);
        h16 mine[2], b[4];
        Q::half(wrow, g, lane_hi, mine);
        if (!n_ok) {
#pragma unroll
            for (int f = 0; f < 2; ++f)
#pragma unroll
                for (int i = 0; i < 16; ++i) mine[f][i] = 0;
        }
#pragma unroll
        for (int f = 0; f < 2; ++f) {
            uint32_t w[8], o[8];
            __builtin_memcpy(w, &mine[f], 32);
#pragma unroll
            for (int d = 0; d < 8; ++d) o[d] = swap16(w[d]);
            h16 other;
            __builtin_memcpy(&other, o, 32);
            b[f] = lane_hi ? other : mine[f];       // fragments 0, 1: the low half's
            b[2 + f] = lane_hi ? mine[f] : other;   // fragments 2, 3: the high half's
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int t = m * 16 + lane_lo;
            const uint16_t* xrow = X + (int64_t) (t < T ? t : 0) * ldx;
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                h16 a;
                if (t < T) __builtin_memcpy(&a, xrow + k[f], sizeof(h16));
                else {
#pragma unroll
                    for (int i = 0; i < 16; ++i) a[i] = 0;
                }
                acc[m] = WmmaTraits<_Float16>::mma(a, b[f], acc[m]);
            }
        }
    }
    if (!n_ok) return;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int t = m * 16 + 2 * i + lane_hi;
            if (t >= T) continue;
            if (part) {
                part[((int64_t) blockIdx.y * T + t) * N + n_row] = acc[m][i];
            } else {
                float* dst = Y + (int64_t) t * ldy + n_row;
                *dst = beta == 0.0f ? acc[m][i] : beta * *dst + acc[m][i];
            }
        }
    }
}

template <typename Q>
bool run_q(const uint16_t* X, int64_t ldx, const void* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
           float beta, void* stream) {
    if (!X || !W || !Y || T <= 0 || N <= 0 || T > max_t() || K % Q::QK != 0 || K % 64 != 0 || ldx % 8 != 0 ||
        (beta != 0.0f && beta != 1.0f) || (reinterpret_cast<uintptr_t>(X) & 15))
        return false;
    if (!strata_wmma_gfx11_device()) return false;
    if (ldy <= 0) ldy = N;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const int64_t row_bytes = K / Q::QK * Q::BYTES, groups = K / 64;
    const int64_t waves_n = (N + 15) / 16, blocks_n = (waves_n + WPB - 1) / WPB;
    static const int64_t target = [] { const char* e = std::getenv("STRATA_SKINNY_WAVES"); return e ? std::atoll(e) : (int64_t) 640; }();
    int64_t splits = std::max<int64_t>(1, std::min<int64_t>((target + waves_n - 1) / waves_n, groups));
    const int64_t gchunk = (groups + splits - 1) / splits;
    splits = (groups + gchunk - 1) / gchunk;
    float* part = nullptr;
    if (splits > 1) {
        part = workspace(s, (size_t) (splits * T * N) * sizeof(float));
        if (!part) return false;
    }
    const dim3 grid((unsigned) blocks_n, (unsigned) splits);
    const int mt = (int) ((T + 15) / 16);
#define SKQ(MT) kernel_q<Q, MT><<<grid, 32 * WPB, 0, s>>>(X, (int) ldx, (const uint8_t*) W, row_bytes, Y, (int) ldy, part, \
                                                         (int) T, (int) N, (int) groups, (int) gchunk, beta)
    switch (mt) {
        case 1: SKQ(1); break; case 2: SKQ(2); break; case 3: SKQ(3); break; case 4: SKQ(4); break;
        case 5: SKQ(5); break; case 6: SKQ(6); break; case 7: SKQ(7); break; default: SKQ(8); break;
    }
#undef SKQ
    if (splits > 1) {
        const int64_t n = T * N;
        reduce<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(part, (int) splits, Y, (int) ldy, (int) T, (int) N, beta);
    }
    return hipGetLastError() == hipSuccess;
}
}  // namespace skinny

bool strata_skinny_gemm_native(const uint16_t* X, int64_t ldx, int ggml_type, const void* W, float* Y, int64_t T,
                               int64_t N, int64_t K, int64_t ldy, float beta, void* stream) {
    switch (ggml_type) {
        case 14: return skinny::run_q<skinny::Q6K>(X, ldx, W, Y, T, N, K, ldy, beta, stream);
        case 12: return skinny::run_q<skinny::Q4K>(X, ldx, W, Y, T, N, K, ldy, beta, stream);
        case 13: return skinny::run_q<skinny::Q5K>(X, ldx, W, Y, T, N, K, ldy, beta, stream);
        case 8: return skinny::run_q<skinny::Q80>(X, ldx, W, Y, T, N, K, ldy, beta, stream);
        case 23: return skinny::run_q<skinny::IQ4XS>(X, ldx, W, Y, T, N, K, ldy, beta, stream);
        case 20: return skinny::run_q<skinny::IQ4NL>(X, ldx, W, Y, T, N, K, ldy, beta, stream);
        default: return false;
    }
}
bool strata_skinny_gemm_f16(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw, float* Y, int64_t T,
                            int64_t N, int64_t K, int64_t ldy, float beta, void* stream) {
    return skinny::run<_Float16>(X, ldx, W, ldw, Y, T, N, K, ldy, beta, stream);
}
bool strata_skinny_gemm_bf16(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw, float* Y, int64_t T,
                             int64_t N, int64_t K, int64_t ldy, float beta, void* stream) {
    return skinny::run<__bf16>(X, ldx, W, ldw, Y, T, N, K, ldy, beta, stream);
}
int strata_skinny_max_t() { return skinny::max_t(); }

#else
// Non-HIP / Non-GFX11 compilation fallback
bool strata_skinny_gemm_f16(const uint16_t*, int64_t, const uint16_t*, int64_t, float*, int64_t, int64_t, int64_t, int64_t,
                            float, void*) { return false; }
bool strata_skinny_gemm_bf16(const uint16_t*, int64_t, const uint16_t*, int64_t, float*, int64_t, int64_t, int64_t, int64_t,
                             float, void*) { return false; }
int strata_skinny_max_t() { return 0; }
bool strata_skinny_gemm_native(const uint16_t*, int64_t, int, const void*, float*, int64_t, int64_t, int64_t, int64_t,
                               float, void*) { return false; }
bool strata_pf_gemm_f16(const uint16_t*, const uint16_t*, float*, int64_t, int64_t, int64_t, int64_t, float, void*) {
    return false;
}
bool strata_pf_gemm_f16_ld(const uint16_t*, int64_t, const uint16_t*, int64_t, float*, int64_t, int64_t, int64_t,
                           int64_t, float, void*) {
    return false;
}
bool strata_pf_hcdown_bf16(const uint16_t*, int64_t, const uint16_t*, const uint16_t*, int64_t, int64_t, float*, float*,
                           int64_t, int64_t, void*) {
    return false;
}
bool strata_pf_hcdown_exact_bf16(const uint16_t*, int64_t, const uint16_t*, float*, int64_t, int64_t, int64_t, void*) {
    return false;
}
bool strata_wmma_gemm_f16(const uint16_t*, const uint16_t*, float*,
                          int64_t, int64_t, int64_t, int64_t, float,
                          void*) {
    return false;
}

bool strata_wmma_gemm_bf16(const uint16_t*, const uint16_t*, float*,
                           int64_t, int64_t, int64_t, int64_t, float,
                           void*) {
    return false;
}
#endif
