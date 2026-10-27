// src/kernels/flashhead_parity.cpp - FlashHead's two kernels against references (GPU, synthetic, no model).
//
//     build/flashhead_parity          the checks below; exit 1 on any difference
//     build/flashhead_parity --bench  one draft head evaluation's time: the subsets, the whole head, FlashHead
//
// 1. native_mmvq_rows (stage 2): random Q6_K rows (random codes and scales, fp16 d in a realistic range) and a random
//    activation.  y[i] for ids with repeats, the first and the last row must be the dense native_mmvq's y[ids[i]],
//    bitwise, from the GGUF layout and from a STRATA_Q6_PACKED copy - the draft's argmax over the candidates is then
//    the dense head's whenever its argmax is a candidate.
// 2. flashhead_select (stage 1's pick): random scores with many ties, negative ones included.  The picked clusters
//    must be the host's (score descending, ties to the lower cluster), listed in cluster order - those above the cut,
//    then the tied ones - after the static ids, each cluster's b ids from c2t.  One row and two rows; probes 1, 256
//    and every cluster.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/flashhead.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

namespace K = strata::kernels;

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc((void**) &d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T> std::vector<T> download(const T* d, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

void rows_parity(cudaStream_t s) {
    const int n_in = 2560, n_out = 5000, n_rows = 3000;
    std::mt19937 rng(7);
    std::vector<uint8_t> w((size_t) n_out * (n_in / 256) * 210);
    for (size_t b = 0; b < w.size() / 210; ++b) {
        uint8_t* blk = w.data() + b * 210;
        for (int i = 0; i < 192; ++i) blk[i] = (uint8_t) rng();                         // ql, qh
        for (int i = 0; i < 16; ++i) blk[192 + i] = (uint8_t) (int8_t) ((int) (rng() % 64) - 32);   // scales
        const uint16_t d = K::f16_from_f32(0.0005f + 0.002f * (float) (rng() % 1000) / 1000.0f);
        std::memcpy(blk + 208, &d, 2);
    }
    std::vector<float> x(n_in);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x) v = nd(rng);
    std::vector<int32_t> ids(n_rows);
    for (int32_t& v : ids) v = (int32_t) (rng() % n_out);
    ids[0] = 0;
    ids[1] = n_out - 1;
    ids[2] = ids[3];                                                                     // a repeat

    uint8_t* dw = upload(w);
    float* dx = upload(x);
    int32_t* did = upload(ids);
    void* xq = nullptr;
    float *dy = nullptr, *dr = nullptr;
    cudaMalloc(&xq, (size_t) (n_in / 32) * 36);
    cudaMalloc((void**) &dy, (size_t) n_out * 4);
    cudaMalloc((void**) &dr, (size_t) n_rows * 4);
    K::native_quantize_q8_1(dx, xq, n_in, 1, s);
    K::native_mmvq(14, dw, xq, dy, n_in, n_out, 1, s);
    const bool launched = K::native_mmvq_rows(14, dw, did, xq, dr, n_in, n_rows, s);
    cudaStreamSynchronize(s);
    const auto y = download(dy, n_out), r = download(dr, n_rows);
    int diff = 0;
    for (int i = 0; i < n_rows; ++i) diff += std::memcmp(&r[(size_t) i], &y[(size_t) ids[(size_t) i]], 4) != 0;
    double norm = 0;
    for (float v : y) norm += (double) v * v;
    check(launched, "native_mmvq_rows launches for Q6_K");
    check(diff == 0 && norm > 0, "native_mmvq_rows: every row bitwise the dense head's (n_in 2560, 3000 of 5000 rows)");
    check(!K::native_mmvq_rows(8, dw, did, xq, dr, n_in, n_rows, s), "native_mmvq_rows declines another type");
    // the same rows from a STRATA_Q6_PACKED copy (the shipped heads on gfx1151 are packed)
    check(K::native_q6_k_pack(dw, n_in, n_out, "parity matrix"), "native_q6_k_pack packs the matrix");
    cudaMemset(dr, 0xff, (size_t) n_rows * 4);
    K::native_mmvq_rows(14, dw, did, xq, dr, n_in, n_rows, s);
    cudaStreamSynchronize(s);
    const auto rp = download(dr, n_rows);
    int diff_p = 0;
    for (int i = 0; i < n_rows; ++i) diff_p += std::memcmp(&rp[(size_t) i], &y[(size_t) ids[(size_t) i]], 4) != 0;
    check(diff_p == 0, "native_mmvq_rows from the packed planes: every row bitwise the dense head's");
    K::native_q6_k_unpack(dw);
    cudaFree(dw); cudaFree(dx); cudaFree(did); cudaFree(xq); cudaFree(dy); cudaFree(dr);
}

// the host's pick: the clusters above the probes-th largest score in cluster order, then the first tied ones
std::vector<int> host_pick(const float* sc, int n, int probes) {
    std::vector<int> order((size_t) n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sc[a] > sc[b]; });
    const float cut = sc[order[(size_t) probes - 1]];
    std::vector<int> gt, eq;
    for (int i = 0; i < n; ++i) {
        if (sc[i] > cut) gt.push_back(i);
        else if (sc[i] == cut) eq.push_back(i);
    }
    eq.resize((size_t) probes - gt.size());
    gt.insert(gt.end(), eq.begin(), eq.end());
    return gt;
}

void select_parity(cudaStream_t s, int n_rows, int n_clusters, int probes, int b, int n_static, const char* what) {
    std::mt19937 rng(11 + probes);
    std::vector<float> sc((size_t) n_rows * n_clusters);
    for (float& v : sc) v = ((float) ((int) (rng() % 81) - 40) + 0.5f) * 0.25f;          // ties, no zeros
    std::vector<int32_t> c2t((size_t) n_clusters * b), stat((size_t) n_static);
    std::iota(c2t.begin(), c2t.end(), 1000);
    std::shuffle(c2t.begin(), c2t.end(), rng);
    std::iota(stat.begin(), stat.end(), 7);
    const int m = n_static + probes * b;
    float* dsc = upload(sc);
    int32_t* dc2t = upload(c2t);
    int32_t* dst = upload(stat);
    int32_t* dids = nullptr;
    cudaMalloc((void**) &dids, (size_t) n_rows * m * 4);
    K::flashhead_select(dsc, n_rows, n_clusters, probes, dc2t, b, dst, n_static, dids, s);
    cudaStreamSynchronize(s);
    const auto ids = download(dids, (size_t) n_rows * m);
    bool ok = true;
    for (int r = 0; r < n_rows && ok; ++r) {
        const auto pick = host_pick(sc.data() + (size_t) r * n_clusters, n_clusters, probes);
        std::vector<int32_t> want(stat);
        for (int c : pick)
            for (int j = 0; j < b; ++j) want.push_back(c2t[(size_t) c * b + j]);
        ok = std::equal(want.begin(), want.end(), ids.begin() + (ptrdiff_t) r * m);
    }
    check(ok, what);
    cudaFree(dsc); cudaFree(dc2t); cudaFree(dst); cudaFree(dids);
}

// --bench: one draft head evaluation, as the drafter runs it (packed Q6_K heads of 2560-wide rows): the subsets of
// 40,525 (en) and 106,299 (cjk) rows, the whole 248,320-row head, and FlashHead (the Q4_0 centroid scores, the pick,
// 12,288 rows of the whole head).  Median of 9 batches of 20 launches, in microseconds.  Synthetic data: the timing
// depends on the bytes read, not on the values.
void bench(cudaStream_t s) {
    const int n_in = 2560, n_vocab = 248320, clusters = 7632, b = 32, n_static = 4096, probes = 256;
    std::mt19937 rng(3);
    auto random_q6 = [&](int rows) {
        std::vector<uint8_t> w((size_t) rows * (n_in / 256) * 210);
        for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t) rng();
        for (size_t blk = 0; blk < w.size() / 210; ++blk) {
            const uint16_t d = K::f16_from_f32(0.001f);
            std::memcpy(w.data() + blk * 210 + 208, &d, 2);
        }
        uint8_t* dw = upload(w);
        K::native_q6_k_pack(dw, n_in, rows, "bench head");
        return dw;
    };
    std::vector<float> x(n_in, 0.5f);
    float* dx = upload(x);
    void* xq = nullptr;
    cudaMalloc(&xq, (size_t) (n_in / 32) * 36);
    K::native_quantize_q8_1(dx, xq, n_in, 1, s);
    float* dy = nullptr;
    cudaMalloc((void**) &dy, (size_t) n_vocab * 4);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto time_us = [&](auto&& launch) {
        for (int i = 0; i < 5; ++i) launch();
        std::vector<float> t;
        for (int k = 0; k < 9; ++k) {
            cudaEventRecord(e0, s);
            for (int i = 0; i < 20; ++i) launch();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            t.push_back(ms * 1000.0f / 20.0f);
        }
        std::sort(t.begin(), t.end());
        return t[4];
    };
    for (int rows : {40525, 106299, n_vocab}) {
        uint8_t* dw = random_q6(rows);
        const float us = time_us([&] { K::native_mmvq(14, dw, xq, dy, n_in, rows, 1, s); });
        std::printf("dense head %6d rows: %7.1f us  (%.1f MiB, %.0f GB/s)\n", rows, us, rows * 2100.0 / 1048576.0,
                    rows * 2100.0 / (us * 1e3));
        if (rows != n_vocab) {
            K::native_q6_k_unpack(dw);
            cudaFree(dw);
            continue;
        }
        std::vector<uint8_t> cent((size_t) clusters * (n_in / 32) * 18);
        for (size_t i = 0; i < cent.size(); ++i) cent[i] = (uint8_t) rng();
        for (size_t blk = 0; blk < cent.size() / 18; ++blk) {
            const uint16_t d = K::f16_from_f32(0.01f);
            std::memcpy(cent.data() + blk * 18, &d, 2);
        }
        std::vector<int32_t> c2t((size_t) clusters * b), stat((size_t) n_static);
        std::iota(c2t.begin(), c2t.end(), n_static);
        std::shuffle(c2t.begin(), c2t.end(), rng);          // a cluster's rows are scattered over the head
        std::iota(stat.begin(), stat.end(), 0);
        uint8_t* dc = upload(cent);
        int32_t* dc2t = upload(c2t);
        int32_t* dst = upload(stat);
        float* dsc = nullptr;
        int32_t* dids = nullptr;
        const int m = n_static + probes * b;
        cudaMalloc((void**) &dsc, (size_t) clusters * 4);
        cudaMalloc((void**) &dids, (size_t) m * 4);
        const float t1 = time_us([&] { K::native_mmvq(2, dc, xq, dsc, n_in, clusters, 1, s); });
        const float t2 = time_us([&] { K::flashhead_select(dsc, 1, clusters, probes, dc2t, b, dst, n_static, dids, s); });
        const float t3 = time_us([&] { K::native_mmvq_rows(14, dw, dids, xq, dy, n_in, m, s); });
        const float all = time_us([&] {
            K::native_mmvq(2, dc, xq, dsc, n_in, clusters, 1, s);
            K::flashhead_select(dsc, 1, clusters, probes, dc2t, b, dst, n_static, dids, s);
            K::native_mmvq_rows(14, dw, dids, xq, dy, n_in, m, s);
        });
        std::printf("FlashHead: centroids %.1f us + pick %.1f us + %d rows %.1f us = %.1f us back to back\n", t1, t2, m,
                    t3, all);
        cudaFree(dc); cudaFree(dc2t); cudaFree(dst); cudaFree(dsc); cudaFree(dids);
        K::native_q6_k_unpack(dw);
        cudaFree(dw);
    }
    cudaFree(dx); cudaFree(xq); cudaFree(dy);
}

}  // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no GPU: skipped\n");
        return 77;
    }
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        bench(s);
        cudaStreamDestroy(s);
        return 0;
    }
    rows_parity(s);
    select_parity(s, 1, 7632, 256, 32, 4096, "flashhead_select: 7632 clusters, 256 probes, the host's pick and order");
    select_parity(s, 2, 7632, 256, 4, 100, "flashhead_select: two rows, each its own pick");
    select_parity(s, 1, 7632, 1, 4, 0, "flashhead_select: one probe, no static tokens");
    select_parity(s, 1, 1000, 1000, 2, 3, "flashhead_select: every cluster");
    select_parity(s, 1, 300, 37, 8, 5, "flashhead_select: fewer clusters than threads");
    cudaStreamDestroy(s);
    std::printf("%s\n", failures ? "flashhead_parity: FAILED" : "flashhead_parity: all checks passed");
    return failures ? 1 : 0;
}
