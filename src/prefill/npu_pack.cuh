// (included by npu_split.cu; generated from the scratchpad npu_pack.hip - the layouts of strata-npu/w4gemm)
// GPU-side packing for the NPU w8 prefill GEMM (layouts = strata-npu/w4gemm/w4gemm.py pack_a / pack_w8 /
// unpack_y_rows, LOOP41 + PREFILL, YS=2, NCOL=8, NROW=4).  All index math follows the numpy transposes there.



namespace npk {
constexpr int NCOL = 8, NROW = 4, E0 = 133;

// smallest s with amax <= 127 * 2^s (amax > 0), as int8g_encode / bfp16_encode define it
__device__ __forceinline__ int shared_exp(float amax) {
    if (!(amax > 0.f)) return 0;
    int s = (int) ceilf(log2f(amax / 127.f));
    while (amax > 127.f * exp2f((float) s)) ++s;
    while (s > -120 && amax <= 127.f * exp2f((float) (s - 1))) --s;
    return s;
}
__device__ __forceinline__ int8_t q8(float x, float inv) {
    float v = rintf(x * inv);
    v = fminf(fmaxf(v, -128.f), 127.f);
    return (int8_t) (int) v;
}

// A: X f16 [T rows, Kc cols at k0] (row stride ldx) -> bytes [c][p][r][quad][half][kb][rb][i][9]; rows >= T are zero.
// One thread per 8-value block, threads in OUTPUT order (the host-memory writes coalesce; X reads hit the L2).
__global__ void pack_a_kernel(const _Float16* __restrict__ X, int ldx, int k0, int T, int rows_total, int KB, int RP,
                              uint8_t* __restrict__ out) {
    const long long nblk = (long long) rows_total * NCOL * KB;
    const long long ob = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (ob >= nblk) return;
    const int RPh = RP / 2, P = rows_total / (NROW * 16 * RP);
    long long x = ob;
    const int i = (int) (x % 8); x /= 8;
    const int rb = (int) (x % 2); x /= 2;
    const int kb = (int) (x % KB); x /= KB;
    const int half = (int) (x % 2); x /= 2;
    const int quad = (int) (x % RPh); x /= RPh;
    const int r = (int) (x % NROW); x /= NROW;
    const int p = (int) (x % P); const int c = (int) (x / P);
    const int m = ((((p * NROW + r) * RPh + quad) * 2 + half) * 2 + rb) * 8 + i;
    const int kbg = c * KB + kb;
    uint8_t* dst = out + ob * 9;
    float v[8]; float amax = 0.f;
    if (m < T) {
        const _Float16* src = X + (long long) m * ldx + k0 + (long long) kbg * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) { v[j] = (float) src[j]; amax = fmaxf(amax, fabsf(v[j])); }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = 0.f;
    }
    const int s = shared_exp(amax);
    const float inv = exp2f((float) -s);
    dst[0] = (uint8_t) (s + E0);
    #pragma unroll
    for (int j = 0; j < 8; ++j) dst[1 + j] = (uint8_t) q8(v[j], inv);
}

// W: f16 rows n in [0, Nn) of W (row stride ldw), k in [k0, k0+Kc) -> bytes per (c, obj) of W_OBJ:
//   mant [q][kb][j][nl][kl] (2*KB*2*64) then exps [q][g][j][nl] (2*G*2*8); one exponent per 32 k (4 k blocks).
// One thread per 8-byte mantissa unit in OUTPUT order; the kb % 4 == 0 thread also writes the group's exponent.
__global__ void pack_w_kernel(const _Float16* __restrict__ W, int ldw, int k0, int Nn, int KB, int W_OBJ,
                              uint8_t* __restrict__ out) {
    const int G = KB / 4, NW = Nn / 32;
    const long long nunits = (long long) NCOL * NW * 2 * KB * 2 * 8;
    const long long t = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (t >= nunits) return;
    long long x = t;
    const int nl = (int) (x % 8); x /= 8;
    const int j = (int) (x % 2); x /= 2;
    const int kb = (int) (x % KB); x /= KB;
    const int q = (int) (x % 2); x /= 2;
    const int obj = (int) (x % NW); const int c = (int) (x / NW);
    const int n = ((obj * 2 + q) * 2 + j) * 8 + nl, g = kb / 4, u = kb % 4;
    const _Float16* src = W + (long long) n * ldw + k0 + (long long) c * (KB * 8) + g * 32;
    float amax = 0.f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf((float) src[k]));
    const int s = shared_exp(amax);
    const float inv = exp2f((float) -s);
    uint8_t* base = out + ((long long) c * NW + obj) * W_OBJ;
    uint8_t* d = base + ((((q * KB + kb) * 2 + j) * 8 + nl) * 8);
    #pragma unroll
    for (int kl = 0; kl < 8; ++kl) d[kl] = (uint8_t) q8((float) src[u * 8 + kl], inv);
    if (u == 0) base[2 * KB * 2 * 64 + ((q * G + g) * 2 + j) * 8 + nl] = (uint8_t) (s + E0);
}

// Y: bf16 [h][chunk][p][obj][rr][quad][cb][rb][i][c] -> Y f32 [T, N] (row stride ldy) columns [n0, n0+Nn); acc adds.
// One thread per bf16 element in the NPU's layout order (coalesced host-memory reads); rows >= T are dropped.
__global__ void unpack_y_kernel(const uint16_t* __restrict__ y, int T, int rows_total, int Nn, int NC, int NWc, int RP,
                                float* __restrict__ Y, int ldy, int n0, int acc) {
    const int R = 16 * RP, P = rows_total / (NROW * R), YS = 2, RPh = RP / 2;
    const long long total = (long long) YS * NC * P * NWc * (NROW / YS) * RPh * 4 * 4 * 8 * 8;
    const long long o = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (o >= total) return;
    long long x = o;
    const int cc = (int) (x % 8); x /= 8;
    const int i = (int) (x % 8); x /= 8;
    const int rb = (int) (x % 4); x /= 4;
    const int cb = (int) (x % 4); x /= 4;
    const int quad = (int) (x % RPh); x /= RPh;
    const int rr = (int) (x % (NROW / YS)); x /= (NROW / YS);
    const int obj = (int) (x % NWc); x /= NWc;
    const int p = (int) (x % P); x /= P;
    const int chunk = (int) (x % NC); const int h = (int) (x / NC);
    const int r = h * (NROW / YS) + rr;
    const int m = p * (NROW * R) + r * R + quad * 32 + rb * 8 + i;
    if (m >= T) return;
    const int n = (chunk * NWc + obj) * 32 + cb * 8 + cc;
    const float v = __uint_as_float(((uint32_t) y[o]) << 16);
    float* dst = Y + (long long) m * ldy + n0 + n;
    *dst = acc ? *dst + v : v;
}
}  // namespace npk

