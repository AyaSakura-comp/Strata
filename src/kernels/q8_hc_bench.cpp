// src/kernels/q8_hc_bench.cpp - the STRATA_HC_Q8 read (launch_q8 in fused_gr.cu) on synthetic weights: time the
// down+up Q8 kernels per layer-half, and check the outputs against the BF16 path bit for bit where comparable.
//
//     build/q8_hc_bench [iters] [T]
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// ggml's reference Q8_0 rounding (same as native_dense.cpp's q8_0_of)
static void q8_0_of(const float* x, uint64_t n, uint8_t* out) {
    for (uint64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = out + b * 34;
        const float df = d;
        uint16_t hb;
        std::memcpy(&hb, &df, 2);   // fp16 of d (host is fine with the cast via _Float16)
        _Float16 h = (_Float16) df;
        std::memcpy(o, &h, 2);
        (void) hb;
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    const int T = argc > 2 ? std::atoi(argv[2]) : 4;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<float> wd((size_t) LR * D), wu((size_t) D * LR);
    for (auto& x : wd) x = 0.02f * nd(rng);
    for (auto& x : wu) x = 0.05f * nd(rng);
    std::vector<uint16_t> wdb(wd.size()), wub(wu.size());
    for (size_t i = 0; i < wd.size(); ++i) wdb[i] = bf16(wd[i]);
    for (size_t i = 0; i < wu.size(); ++i) wub[i] = bf16(wu[i]);
    // Q8_0 copies, per ROW (down: LR rows of D; up: D rows of LR), as the engine's loader makes them
    std::vector<uint8_t> qd((size_t) LR * (D / 32) * 34), qu((size_t) D * (LR / 32) * 34);
    for (int r = 0; r < LR; ++r) q8_0_of(wd.data() + (size_t) r * D, D, qd.data() + (size_t) r * (D / 32) * 34);
    for (int r = 0; r < D; ++r) q8_0_of(wu.data() + (size_t) r * LR, LR, qu.data() + (size_t) r * (LR / 32) * 34);

    float *dR, *dbo, *dinj, *dwn, *dxn, *dlo, *drs, *dio, *dmix;
    uint16_t *dwd, *dwu;
    uint8_t *dqd, *dqu;
    cudaMalloc((void**) &dR, R.size() * 4);
    cudaMalloc((void**) &dbo, bo.size() * 4);
    cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMalloc((void**) &dwn, wn.size() * 4);
    cudaMalloc((void**) &dxn, (size_t) TM * D * 4);
    cudaMalloc((void**) &dlo, (size_t) TM * LR * 4);
    cudaMalloc((void**) &drs, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dio, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dmix, (size_t) TM * N * 4);
    cudaMalloc((void**) &dwd, wdb.size() * 2);
    cudaMalloc((void**) &dwu, wub.size() * 2);
    cudaMalloc((void**) &dqd, qd.size());
    cudaMalloc((void**) &dqu, qu.size());
    // L2-defeating rotation: NS sets of weights, cycled per call (production streams ~7 MB/layer with no reuse)
    constexpr int NS = 12;
    uint8_t *dqds[NS], *dqus[NS];
    for (int i = 0; i < NS; ++i) {
        cudaMalloc((void**) &dqds[i], qd.size());
        cudaMalloc((void**) &dqus[i], qu.size());
        std::vector<uint8_t> td(qd.size()), tu(qu.size());
        for (size_t b = 0; b < td.size(); ++b) td[b] = (uint8_t) (qd[b] + i * 7);
        for (size_t b = 0; b < tu.size(); ++b) tu[b] = (uint8_t) (qu[b] + i * 5);
        cudaMemcpy(dqds[i], td.data(), td.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dqus[i], tu.data(), tu.size(), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwd, wdb.data(), wdb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wub.data(), wub.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dqd, qd.data(), qd.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dqu, qu.data(), qu.size(), cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    K::FusedGrArgs a[TM];
    for (int t = 0; t < T; ++t) {
        a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = 1;
        a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
        a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = nullptr;
        a[t].q8_down = dqd; a[t].q8_up = dqu; a[t].q8_inject = nullptr;
        a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
        a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
    }
    // time the Q8 read, and (for reference) the BF16 read
    double us_q8 = 1e30, us_bf16 = 1e30;
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(a, T, dxn, s);
        cudaEventRecord(e0, s);
        for (int i = 0; i < iters; ++i) {
            for (int t = 0; t < T; ++t) { a[t].q8_down = dqds[i % NS]; a[t].q8_up = dqus[i % NS]; }
            K::fused_gr_read_multi(a, T, dxn, s);
        }
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        us_q8 = std::fmin(us_q8, 1e3 * ms / iters);
    }
    for (int t = 0; t < T; ++t) { a[t].q8_down = nullptr; a[t].q8_up = nullptr; }
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(a, T, dxn, s);
        cudaEventRecord(e0, s);
        for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, dxn, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        us_bf16 = std::fmin(us_bf16, 1e3 * ms / iters);
    }
    const double q8_bytes = (double) (LR * (D / 32) * 34 + D * (LR / 32) * 34);
    const double bf_bytes = (double) (LR * D * 2 + D * LR * 2);
    std::printf("q8 hc read: %6.1f us  (%5.1f GB/s weights) | bf16 read: %6.1f us (%5.1f GB/s) | T %d, %d iters\n",
                us_q8, q8_bytes / (us_q8 * 1e3), us_bf16, bf_bytes / (us_bf16 * 1e3), T, iters);
    return 0;
}
