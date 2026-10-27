// FlashHead's stage 1 pick: the `probes` best clusters of a row of centroid scores, as the candidate token list
// stage 2 reads (include/strata/kernels/flashhead.hpp).
#include "strata/kernels/flashhead.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int THREADS = 1024;
constexpr int MAX_PROBES = 1024;

// a float's order as an unsigned key: larger float, larger key (negative floats flipped whole, positive ones get the
// sign bit), so the radix select below compares keys with integer arithmetic
__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// inclusive scan of v over the block's THREADS threads (s: THREADS ints of shared memory)
__device__ int block_scan(int v, int* s) {
    const int t = int(threadIdx.x);
    s[t] = v;
    __syncthreads();
    for (int off = 1; off < THREADS; off <<= 1) {
        const int add = t >= off ? s[t - off] : 0;
        __syncthreads();
        s[t] += add;
        __syncthreads();
    }
    const int r = s[t];
    __syncthreads();
    return r;
}

__launch_bounds__(THREADS)
__global__ void flashhead_select_kernel(const float* __restrict__ scores, int n_clusters, int probes,
                                        const int32_t* __restrict__ c2t, int b, const int32_t* __restrict__ stat,
                                        int n_static, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int scan[THREADS];
    __shared__ int pick[MAX_PROBES];
    __shared__ uint32_t s_prefix;
    __shared__ int s_k;
    const int t = int(threadIdx.x);
    const int m = n_static + probes * b;
    scores += std::size_t(blockIdx.x) * n_clusters;
    ids += std::size_t(blockIdx.x) * m;

    // the probes-th largest key, a byte at a time from the top: `prefix` holds the bytes fixed so far, `k` how many
    // keys with that prefix are still to take
    uint32_t prefix = 0, mask = 0;
    int k = probes;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (t < 256) hist[t] = 0;
        __syncthreads();
        for (int i = t; i < n_clusters; i += THREADS) {
            const uint32_t key = order_key(scores[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
        }
        __syncthreads();
        if (t == 0) {
            int above = 0;
            for (int d = 255; d >= 0; --d) {
                if (above + hist[d] >= k) {
                    s_prefix = prefix | (uint32_t(d) << shift);
                    s_k = k - above;
                    break;
                }
                above += hist[d];
            }
        }
        __syncthreads();
        prefix = s_prefix;
        k = s_k;
        mask |= 255u << shift;
    }
    // prefix is now the cut: every key above it is picked, and the first k keys equal to it (probes - k are above).
    // Each thread scans a contiguous slice so the picks come out in cluster order.
    const int per = (n_clusters + THREADS - 1) / THREADS;
    const int lo = min(n_clusters, t * per), hi = min(n_clusters, lo + per);
    int n_gt = 0, n_eq = 0;
    for (int i = lo; i < hi; ++i) {
        const uint32_t key = order_key(scores[i]);
        n_gt += key > prefix;
        n_eq += key == prefix;
    }
    int gt = block_scan(n_gt, scan) - n_gt;          // exclusive: this slice's first slot
    int eq = block_scan(n_eq, scan) - n_eq;
    for (int i = lo; i < hi; ++i) {
        const uint32_t key = order_key(scores[i]);
        if (key > prefix) {
            pick[gt++] = i;
        } else if (key == prefix) {
            if (eq < k) pick[probes - k + eq] = i;
            ++eq;
        }
    }
    __syncthreads();
    for (int j = t; j < m; j += THREADS) {
        if (j < n_static) {
            ids[j] = stat[j];
        } else {
            const int r = j - n_static;
            ids[j] = c2t[std::size_t(pick[r / b]) * b + r % b];
        }
    }
}

}  // namespace

void flashhead_select(const float* scores, int n_rows, int n_clusters, int probes, const int32_t* c2t, int b,
                      const int32_t* stat, int n_static, int32_t* ids, void* stream) {
    if (n_rows <= 0 || probes <= 0 || probes > MAX_PROBES || probes > n_clusters || b <= 0 || n_static < 0)
        throw std::invalid_argument("flashhead_select: bad shape");
    flashhead_select_kernel<<<unsigned(n_rows), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        scores, n_clusters, probes, c2t, b, stat, n_static, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("flashhead_select launch: ") + cudaGetErrorString(e));
}

}  // namespace strata::kernels
