// NPU N-split of the dense prefill GEMM (gfx1151 + XDNA2): the GPU computes columns [0, N_gpu) with its WMMA kernel
// while the NPU computes [N_gpu, N) with the strata-npu w8 design (bfp16 activations, int8 weights with a shared
// exponent per 32 k, bf16 out).  Zero copy: the GPU packs X and W straight into pinned host buffers the NPU reads
// (XRT userptr BOs) and unpacks the NPU's Y tiles from its output buffer into Y.  Opt-in: STRATA_NPU_SPLIT=1 with
// STRATA_NPU_MANIFEST=<flat text of designs>; shapes without a design fall through to the GPU path.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include "wmma_gemm.h"
#include "npu_pack.cuh"

namespace {
constexpr int RP = 4, BIG_MIN = 8192;   // rows left >= BIG_MIN take a 10240-row launch, else 2048-row ones

struct Pinned {
    void* host = nullptr; void* dev = nullptr; size_t n = 0;
    bool alloc(size_t bytes, std::string& err) {
        n = (bytes + (1u << 21) - 1) & ~((size_t) (1u << 21) - 1);
        host = std::aligned_alloc(1u << 21, n);
        if (!host) { err = "npu split: aligned_alloc failed"; return false; }
        std::memset(host, 0, n);
        if (hipHostRegister(host, n, hipHostRegisterMapped) != hipSuccess) { err = "npu split: hipHostRegister failed"; return false; }
        if (hipHostGetDevicePointer(&dev, host, 0) != hipSuccess) { err = "npu split: hipHostGetDevicePointer failed"; return false; }
        return true;
    }
};
struct Set {   // one in-flight launch: its A, W, Y buffers
    Pinned a, w, y; xrt::bo abo, wbo, ybo; xrt::run run; hipEvent_t ev = nullptr; bool ready = false;
};
struct Design {
    int N, K, Kc, Nn, NC, NWc, W_OBJ, rows, P, KB; std::string dir;
    size_t a_bytes, w_bytes, y_bytes, ninst;
    xrt::xclbin xb; xrt::hw_context ctx; xrt::kernel k; xrt::bo ibo; bool loaded = false;
    std::vector<std::unique_ptr<Set>> sets;
};
struct Job { Design* d; int m0, trows, k0, kc; Set* s; };

class NpuSplit {
public:
    bool init(const char* manifest, std::string& err) {
        std::ifstream f(manifest);
        if (!f) { err = std::string("npu split: cannot read ") + manifest; return false; }
        std::string ln;
        while (std::getline(f, ln)) {
            if (ln.empty() || ln[0] == '#') continue;
            std::istringstream is(ln); Design d{};
            if (!(is >> d.N >> d.K >> d.Kc >> d.Nn >> d.NC >> d.NWc >> d.W_OBJ >> d.rows >> d.P >> d.dir)) continue;
            d.KB = d.Kc / 64;
            d.a_bytes = (size_t) d.rows * d.Kc * 9 / 8;
            d.w_bytes = (size_t) d.NC * d.NWc * d.W_OBJ * npk::NCOL;
            d.y_bytes = (size_t) d.NC * d.P * d.NWc * npk::NROW * (RP * 2 * 4 * 64 * 2);
            designs_.push_back(std::make_unique<Design>(std::move(d)));
        }
        if (designs_.empty()) { err = "npu split: manifest has no designs"; return false; }
        try { dev_ = xrt::device(0); } catch (const std::exception& e) { err = std::string("npu split: xrt device: ") + e.what(); return false; }
        return true;
    }
    Design* find(int64_t N, int64_t K, int rows) {
        for (auto& d : designs_) if (d->N == N && d->K == K && d->rows == rows) return d.get();
        return nullptr;
    }
    bool has(int64_t N, int64_t K) { return find(N, K, 10240) && find(N, K, 2048); }
    bool load(Design& d, std::string& err) {
        if (d.loaded) return true;
        try {
            d.xb = xrt::xclbin(d.dir + "/final.xclbin");
            dev_.register_xclbin(d.xb);
            d.ctx = xrt::hw_context(dev_, d.xb.get_uuid());
            d.k = xrt::kernel(d.ctx, d.xb.get_kernels()[0].get_name());
            std::ifstream f(d.dir + "/insts.bin", std::ios::binary);
            std::vector<char> insts((std::istreambuf_iterator<char>(f)), {});
            if (insts.empty()) { err = "npu split: empty insts.bin in " + d.dir; return false; }
            d.ninst = insts.size();
            d.ibo = xrt::bo(dev_, insts.size(), XCL_BO_FLAGS_CACHEABLE, d.k.group_id(1));
            d.ibo.write(insts.data()); d.ibo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        } catch (const std::exception& e) { err = std::string("npu split: load ") + d.dir + ": " + e.what(); return false; }
        d.loaded = true;
        return true;
    }
    Set* set(Design& d, size_t idx, std::string& err) {
        while (d.sets.size() <= idx) d.sets.push_back(std::make_unique<Set>());
        Set& s = *d.sets[idx];
        if (s.ready) return &s;
        if (!s.a.alloc(d.a_bytes, err) || !s.w.alloc(d.w_bytes, err) || !s.y.alloc(d.y_bytes, err)) return nullptr;
        try {
            s.abo = xrt::bo(dev_, s.a.host, s.a.n, xrt::bo::flags::host_only, d.k.group_id(3));
            s.wbo = xrt::bo(dev_, s.w.host, s.w.n, xrt::bo::flags::host_only, d.k.group_id(4));
            s.ybo = xrt::bo(dev_, s.y.host, s.y.n, xrt::bo::flags::host_only, d.k.group_id(5));
        } catch (const std::exception& e) { err = std::string("npu split: userptr bo: ") + e.what(); return nullptr; }
        if (hipEventCreateWithFlags(&s.ev, hipEventDisableTiming) != hipSuccess) { err = "npu split: hipEventCreate"; return nullptr; }
        s.ready = true;
        pinned_bytes_ += s.a.n + s.w.n + s.y.n;
        return &s;
    }
    size_t pinned_bytes() const { return pinned_bytes_; }
private:
    xrt::device dev_;
    std::vector<std::unique_ptr<Design>> designs_;
    size_t pinned_bytes_ = 0;
};

NpuSplit* instance() {
    static NpuSplit* g = [] () -> NpuSplit* {
        const char* on = std::getenv("STRATA_NPU_SPLIT");
        if (!on || on[0] != '1') return nullptr;
        const char* man = std::getenv("STRATA_NPU_MANIFEST");
        if (!man) { std::fprintf(stderr, "npu split: STRATA_NPU_SPLIT=1 needs STRATA_NPU_MANIFEST\n"); return nullptr; }
        auto* p = new NpuSplit(); std::string err;
        if (!p->init(man, err)) { std::fprintf(stderr, "%s - NPU split off\n", err.c_str()); delete p; return nullptr; }
        std::fprintf(stderr, "npu split: on (%s)\n", man);
        return p;
    }();
    return g;
}
bool verbose() { static const bool v = [] { const char* e = std::getenv("STRATA_NPU_VERBOSE"); return e && e[0] == '1'; }(); return v; }
}  // namespace

bool strata_npu_split_gemm(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw, float* Y, int64_t T,
                           int64_t N, int64_t K, int64_t ldy, float beta, void* stream) {
    NpuSplit* g = instance();
    if (!g || !g->has(N, K) || T < 1024) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    if (ldy <= 0) ldy = N;
    hipStream_t st = static_cast<hipStream_t>(stream);
    static bool failed = false;
    if (failed) return false;
    std::string err;
    // ---- plan the launches: row chunks x K chunks
    std::vector<Job> jobs;
    const int Kc = g->find(N, K, 2048)->Kc, nk = (int) (K / Kc);
    std::vector<size_t> used;   // per design pointer: sets taken (indexed by design address below)
    for (int64_t m0 = 0; m0 < T;) {
        const int64_t rem = T - m0; const int rows = rem >= BIG_MIN ? 10240 : 2048;
        Design* d = g->find(N, K, rows);
        if (!g->load(*d, err)) { std::fprintf(stderr, "%s - NPU split off\n", err.c_str()); failed = true; return false; }
        for (int kc = 0; kc < nk; ++kc) jobs.push_back(Job{d, (int) m0, (int) std::min<int64_t>(rows, rem), kc * Kc, kc, nullptr});
        m0 += rows;
    }
    for (auto& j : jobs) {
        size_t idx = 0;
        for (auto& o : jobs) { if (&o == &j) break; if (o.d == j.d) ++idx; }
        j.s = g->set(*j.d, idx, err);
        if (!j.s) { std::fprintf(stderr, "%s - NPU split off\n", err.c_str()); failed = true; return false; }
    }
    const int Nn = jobs[0].d->Nn, n_gpu = (int) N - Nn;
    // ---- GPU: pack every job's operands, then its own columns
    for (auto& j : jobs) {
        const Design& d = *j.d; Set& s = *j.s;
        const long long nw = (long long) npk::NCOL * (d.Nn / 32) * 2 * d.KB * 2 * 8;
        npk::pack_w_kernel<<<(unsigned) ((nw + 255) / 256), 256, 0, st>>>((const _Float16*) (W + (int64_t) n_gpu * ldw), (int) ldw, j.k0, d.Nn, d.KB, d.W_OBJ, (uint8_t*) s.w.dev);
        const long long na = (long long) d.rows * npk::NCOL * d.KB;
        npk::pack_a_kernel<<<(unsigned) ((na + 255) / 256), 256, 0, st>>>((const _Float16*) (X + (int64_t) j.m0 * ldx), (int) ldx, j.k0, j.trows, d.rows, d.KB, RP, (uint8_t*) s.a.dev);
        hipEventRecord(s.ev, st);
    }
    if (!strata_pf_gemm_f16_ld(X, ldx, W, ldw, Y, T, n_gpu, K, ldy, beta, stream)) {
        std::fprintf(stderr, "npu split: GPU part refused T=%lld N=%d K=%lld - NPU split off\n", (long long) T, n_gpu, (long long) K);
        failed = true; (void) hipStreamSynchronize(st); return false;
    }
    // ---- NPU: start each launch once its operands are packed; unpack each result as it completes
    try {
        for (auto& j : jobs) {
            hipEventSynchronize(j.s->ev);
            j.s->run = j.d->k(3, j.d->ibo, (int) j.d->ninst, j.s->abo, j.s->wbo, j.s->ybo);
        }
        for (auto& j : jobs) {
            const Design& d = *j.d; Set& s = *j.s;
            auto state = s.run.wait();
            if (state != ERT_CMD_STATE_COMPLETED) { std::fprintf(stderr, "npu split: run state %d\n", (int) state); failed = true; return false; }
            const int acc = (j.kc > 0 || beta == 1.0f) ? 1 : 0;
            const long long ny = (long long) 2 * d.NC * d.P * d.NWc * (npk::NROW / 2) * (RP / 2) * 4 * 4 * 8 * 8;
            npk::unpack_y_kernel<<<(unsigned) ((ny + 255) / 256), 256, 0, st>>>((const uint16_t*) s.y.dev, j.trows, d.rows, d.Nn, d.NC, d.NWc, RP, Y + (int64_t) j.m0 * ldy, (int) ldy, n_gpu, acc);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "npu split: %s - NPU split off\n", e.what()); failed = true; return false;
    }
    if (verbose()) {
        static int n = 0;
        if (n++ < 8) std::fprintf(stderr, "npu split: T=%lld N=%lld K=%lld -> GPU %d cols, NPU %d cols in %zu launches (pinned %.0f MB)\n",
                                  (long long) T, (long long) N, (long long) K, n_gpu, Nn, jobs.size(), g->pinned_bytes() / 1e6);
    }
    return true;
}
