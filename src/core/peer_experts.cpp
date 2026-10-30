// src/core/peer_experts.cpp - see include/strata/core/peer_experts.hpp.
#include "strata/core/peer_experts.hpp"

#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE)
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/window_limits.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace strata::core {

static bool g_peer_portable = false;
void set_peer_portable(bool on) { g_peer_portable = on; }
bool peer_portable() { return g_peer_portable; }

namespace {

constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t CAP = strata::kernels::kWindowMaxT * 10;   // entries per layer: a window's rows x top-10

struct Meta {                           // one block, uploaded with the input rows behind it in one copy per layer
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count[4];                   // [0] groups, [1] entries
    int32_t out_row[CAP];               // multi-GPU: compact row -> entry row in the host `out`
};

// the plan, then the input rows (n_tok x H floats), in one host block and one device block: one copy per layer
constexpr size_t kMetaBytes = (sizeof(Meta) + 255) & ~(size_t) 255;

/// Switches the calling thread to `dev` and back.  The engine's own thread lives on device 0.
struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

/// #2666: `cudaDeviceCanAccessPeer` said yes and `cudaDeviceEnablePeerAccess` succeeded, and on a pair of GPUs behind a host bridge
/// (2x RTX A4000, "PHB" in nvidia-smi topo) `cudaMemcpyPeerAsync` then returned success having moved nothing: the peer's input rows
/// were zeros and the rows it sent back never arrived, so the first prompt's MoE output was garbage and the logits NaN.  Without
/// peer access enabled the same copies are right (the driver stages them).  So the pair is asked once, with a small copy each way,
/// whether the direct path really carries data.  true = both ways exact (or nothing could be tested: the old trust is kept).
bool p2p_copies_data(int device) {
    constexpr size_t kN = 1 << 14;   // floats
    std::vector<float> h(kN), back(kN);
    for (size_t i = 0; i < kN; ++i) h[i] = 1.0f + (float) ((i * 2654435761u) % 4093u);
    float* d[2] = {nullptr, nullptr};
    const int dev[2] = {0, device};
    bool ok = true, tested = true;
    for (int k = 0; k < 2 && tested; ++k) {
        On on(dev[k]);
        if (cudaMalloc((void**) &d[k], kN * 4) != cudaSuccess) { cudaGetLastError(); tested = false; }
    }
    for (int dir = 0; dir < 2 && tested && ok; ++dir) {   // 0: device 0 -> peer, 1: peer -> device 0
        const int from = dir == 0 ? 0 : 1, to = 1 - from;
        {
            On on(dev[from]);
            if (cudaMemcpy(d[from], h.data(), kN * 4, cudaMemcpyHostToDevice) != cudaSuccess) { tested = false; break; }
        }
        {
            On on(dev[to]);
            if (cudaMemset(d[to], 0, kN * 4) != cudaSuccess) { tested = false; break; }
        }
        if (cudaMemcpyPeer(d[to], dev[to], d[from], dev[from], kN * 4) != cudaSuccess) { tested = false; break; }
        {
            On on(dev[to]);
            if (cudaMemcpy(back.data(), d[to], kN * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { tested = false; break; }
        }
        ok = std::memcmp(back.data(), h.data(), kN * 4) == 0;
    }
    for (int k = 0; k < 2; ++k)
        if (d[k] != nullptr) { On on(dev[k]); cudaFree(d[k]); }
    if (!tested) { cudaGetLastError(); return true; }
    return ok;
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("peer experts: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

PeerExperts::~PeerExperts() { close(); }

void PeerExperts::close() {
    if (device_ < 0) return;
    {
        On on(device_);
        if (stream_) cudaStreamSynchronize(stream_);
        if (refill_) cudaStreamSynchronize(refill_);
        cache_.close();
        if (d_out_) cudaFree(d_out_);
        if (d_meta_) cudaFree(d_meta_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scratch_) cudaFree(d_scratch_);
        for (cudaEvent_t& e : tev_)
            if (e) { cudaEventDestroy(e); e = nullptr; }
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (h_jobs_) cudaFreeHost(h_jobs_);
        if (refill_ev_) cudaEventDestroy(refill_ev_);
        if (stream_) cudaStreamDestroy(stream_);
        if (refill_) cudaStreamDestroy(refill_);
    }
    d_x_ = d_out_ = h_x_ = h_out_ = nullptr;
    d_meta_ = h_meta_ = d_scratch_ = nullptr;
    h_jobs_ = d_jobs_ = nullptr;
    jobs_cap_ = 0;
    refill_sm_ = 0;
    d_q8_ = nullptr;
    d_sx_ = d_sh_ = nullptr;
    stream_ = refill_ = nullptr;
    refill_ev_ = nullptr;
    device_ = -1;
}

bool PeerExperts::open(int device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
                       ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots,
                       std::string& err) {
    close();
    int count = 0;
    if (!ck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err)) return false;
    if (device < 1 || device >= count) {
        err = "peer experts: CUDA device " + std::to_string(device) + " is not visible (" + std::to_string(count) +
              " devices; set CUDA_VISIBLE_DEVICES)";
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    src_ = &src;
    res_.assign((size_t) (n_layers * n_expert), kNotResident);
    incoming_.assign((size_t) (n_layers * n_expert), 0);
    {   // direct access both ways (NVLink or another P2P path): the prompt path copies activations and results over it
        int a = 0, b = 0;
        cudaDeviceCanAccessPeer(&a, primary_device(), device);
        cudaDeviceCanAccessPeer(&b, device, primary_device());
        if (const char* np = std::getenv("STRATA_PEER_NO_P2P"); np != nullptr && std::atoi(np) != 0) a = 0;   // test knob: pretend the pair has no P2P
        p2p_ = a && b;
        if (p2p_) {
            On on0(primary_device());
            cudaError_t e0 = cudaDeviceEnablePeerAccess(device, 0);
            if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
            On on1(device);
            cudaError_t e1 = cudaDeviceEnablePeerAccess(primary_device(), 0);
            if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
            p2p_ = e0 == cudaSuccess && e1 == cudaSuccess;
            // recommend, never force: the direct path is used only if a test copy each way arrives exact (#2666)
            if (p2p_ && !p2p_copies_data(device)) {
                std::fprintf(stderr, "strata: peer GPU %d: peer access is enabled but a test copy between the two GPUs arrived wrong "
                                     "(a host bridge between them, no real P2P): the prompt path takes the host route instead "
                                     "(STRATA_PEER_NO_P2P=1 does that by hand)\n", device);
                { On q0(0); cudaDeviceDisablePeerAccess(device); cudaGetLastError(); }
                { On q1(device); cudaDeviceDisablePeerAccess(0); cudaGetLastError(); }
                p2p_ = false;
            }
        }
    }
    On on(device);
    device_ = device;
    // scratch first, so the slots take what is really left
    int64_t ff = strata::kernels::cpu::FF;
    if (lay.native)
        for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
    const size_t scratch = std::max<size_t>((size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, ff),
                                            strata::kernels::native_expert_scratch_bytes(CAP, ff));
    const bool alloc_ok =
        ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) &&
        ck(cudaStreamCreateWithFlags(&refill_, cudaStreamNonBlocking), "refill stream", err) &&
        ck(cudaEventCreateWithFlags(&refill_ev_, cudaEventDisableTiming), "event", err) &&
        ck(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err) &&
        ck(cudaHostAlloc(&h_meta_, kMetaBytes + (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped),
           "plan and input staging", err) &&
        ck(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err) &&
        ck(cudaMalloc(&d_meta_, kMetaBytes + (size_t) CAP * H * sizeof(float)), "plan and input", err) &&
        ck(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activations", err) &&
        ck(cudaMalloc(&d_scratch_, scratch), "scratch", err);
    if (!alloc_ok) { close(); return false; }
    h_x_ = (float*) ((uint8_t*) h_meta_ + kMetaBytes);   // (inside the plan's blocks: freed with them)
    d_x_ = (float*) ((uint8_t*) d_meta_ + kMetaBytes);
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this device's copies of the sign vectors
        d_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) H), (int) H);
        d_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) ff), (int) ff);
    }

    // the pairs the primary does not hold, in rank order, as many as fit
    size_t free_b = 0, total_b = 0;
    if (!ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) { close(); return false; }
    const uint64_t reserve = (uint64_t) std::max(reserve_mib, 128) << 20;
    const uint64_t budget = free_b > reserve ? free_b - reserve : 0;
    std::vector<std::pair<int32_t, int32_t>> pick;
    std::vector<int64_t> sizes;
    uint64_t used = 0;
    for (const auto& pr : ranked) {
        if (primary.slot_of(pr.first, pr.second) >= 0) continue;
        const uint64_t b = lay.blob_bytes(pr.first);
        const uint64_t b256 = lay.native ? (b + 255) / 256 * 256 : lay.max_blob;
        if (used + b256 > budget) break;
        if (max_slots > 0 && (int64_t) pick.size() >= max_slots) break;
        used += b256;
        pick.push_back(pr);
        sizes.push_back((int64_t) b);
    }
    if (pick.empty()) { err = "peer experts: no room or no expert left for the peer"; close(); return false; }
    const bool opened = lay.native ? cache_.open_sized(sizes, n_layers, n_expert, err)
                                   : cache_.open((int64_t) pick.size(), n_layers, n_expert, (int64_t) lay.max_blob, err);
    if (!opened) { err = "peer experts: " + err; close(); return false; }
    // open() zeroes the arena with cudaMemset on the legacy stream, which is not ordered against the non-blocking
    // refill stream: wait for it, or the zeroing can land on top of the fills
    if (!ck(cudaDeviceSynchronize(), "arena zeroing", err)) { close(); return false; }
    // mapped reads: advise the next kAhead pairs so their reads overlap
    constexpr size_t kAhead = 256;
    const bool advise = src.advise_pairs(pick.data(), (int64_t) std::min(kAhead, pick.size()));
    for (size_t i = 0; i < pick.size(); ++i) {
        const auto& pr = pick[i];
        if (advise && i + kAhead < pick.size()) (void) src.advise_pairs(&pick[i + kAhead], 1);
        const int32_t slot = cache_.admit(pr.first, pr.second);
        const uint8_t* b = src.blob(pr.first, pr.second);
        if (slot < 0 || b == nullptr || !cache_.fill_slot(slot, b, refill_, err, (int64_t) lay.blob_bytes(pr.first))) {
            err = "peer experts: fill failed: " + err;
            close();
            return false;
        }
        src.note_async_read(b, refill_);   // #1237
        res_[(size_t) (pr.first * n_expert + pr.second)] = slot;
    }
    if (!ck(cudaStreamSynchronize(refill_), "fill", err)) { close(); return false; }
    const auto& f0 = pick.front();
    if (!cache_.verify_slot(res_[(size_t) (f0.first * n_expert + f0.second)], src.blob(f0.first, f0.second), err,
                            (int64_t) lay.blob_bytes(f0.first))) {
        err = "peer experts: " + err;
        close();
        return false;
    }
    resident_ = (int64_t) pick.size();
    // the adaptive refill by SM loads (set_refill_sm, --peer-refill-sm; 0 = the copy engine): only where this card
    // reads registered memory by its host address; tried once here on the first slot (rewritten with its own bytes)
    {
        int want = refill_sm_want_;
        int can = 0;
        if (want > 0 && (cudaDeviceGetAttribute(&can, cudaDevAttrCanUseHostPointerForRegisteredMem, device) != cudaSuccess || !can)) {
            (void) cudaGetLastError();
            std::fprintf(stderr, "strata: peer GPU %d: the adaptive refill stays on the copy engine (the card does not read "
                                 "registered memory by its host address)\n", device);
            want = 0;
        }
        refill_sm_ = want;
        const uint8_t* b = src.blob(f0.first, f0.second);
        if (refill_sm_ > 0 && src.device_alias(f0.first, f0.second) == b) {
            strata::kernels::CopyJob j{b, cache_.device_slot(res_[(size_t) (f0.first * n_expert + f0.second)]),
                                       (unsigned long long) lay.blob_bytes(f0.first)};
            if (!refill_jobs(&j, 1, err) || !ck(cudaStreamSynchronize(refill_), "refill test", err) ||
                !cache_.verify_slot(res_[(size_t) (f0.first * n_expert + f0.second)], b, err, (int64_t) lay.blob_bytes(f0.first))) {
                err = "peer experts: the SM refill test: " + err;
                close();
                return false;
            }
        }
    }
    return true;
}

bool PeerExperts::refill_jobs(const strata::kernels::CopyJob* jobs, size_t n, std::string& err) {
    if (n == 0) return true;
    if (n > jobs_cap_) {   // only between rounds: the round before has landed (pending_ was empty)
        if (h_jobs_ != nullptr) cudaFreeHost(h_jobs_);
        h_jobs_ = d_jobs_ = nullptr;
        jobs_cap_ = 0;
        const size_t cap = std::max<size_t>(n, 256);
        if (!ck(cudaHostAlloc(&h_jobs_, cap * sizeof(strata::kernels::CopyJob), cudaHostAllocPortable | cudaHostAllocMapped), "refill jobs", err) ||
            !ck(cudaHostGetDevicePointer(&d_jobs_, h_jobs_, 0), "refill jobs alias", err))
            return false;
        jobs_cap_ = cap;
    }
    std::memcpy(h_jobs_, jobs, n * sizeof(strata::kernels::CopyJob));
    strata::kernels::copy_jobs_sm((const strata::kernels::CopyJob*) d_jobs_, (int) n, refill_sm_, refill_);
    return ck(cudaGetLastError(), "refill kernel", err);
}

namespace {
bool peer_timing() {   // STRATA_PEER_TIMING=1: PeerExperts::take_timing's sums (a diagnostic, off by default)
    static const bool on = [] { const char* v = std::getenv("STRATA_PEER_TIMING"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}
bool peer_down_host() {   // STRATA_PEER_DOWN_HOST=0: the rows computed into the device, then scattered, as before
    static const bool on = [] { const char* v = std::getenv("STRATA_PEER_DOWN_HOST"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}
bool peer_host_q8_env() {   // STRATA_PEER_HOST_Q8=0: the input copied as floats and quantized on the second GPU
    static const bool on = [] { const char* v = std::getenv("STRATA_PEER_HOST_Q8"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}
bool cpu_avx2_f16c() {
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
    return ok;
}

// The input rows as q8_1 blocks, quantized on the host the way quantize_q8_1_kernel's q8_1_store does it on the device
// (iq_kernels.cu, no fast math): the warp's butterfly sum (lane i adds lane i^16, then ^8, ^4, ^2, ^1 - float addition
// is commutative, so every lane ends with this sum), amax, d = amax / 127 kept finite (q8_1_finite), the codes roundf(x
// / d) clamped to +-127 (0 for an all-zero block), ds = {d, sum} as fp16 rounded to nearest even.  The same bytes, so
// the second GPU gets 36 bytes a block instead of 128 and no quantization to run.  AVX2 + F16C only (else the device's).
constexpr size_t kQ8Block = 36;
__attribute__((target("avx2,f16c"))) void quantize_q8_1_host(const float* x, int64_t n_tok, int64_t n_cols, uint8_t* y) {
    const __m256 sign = _mm256_set1_ps(-0.0f), half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f);
    const __m256 lim = _mm256_set1_ps(127.0f), nlim = _mm256_set1_ps(-127.0f);
    const int64_t blocks = n_tok * (n_cols / 32);
    for (int64_t b = 0; b < blocks; ++b) {
        const float* p = x + b * 32;
        uint8_t* blk = y + b * kQ8Block;
        const __m256 a[4] = {_mm256_loadu_ps(p), _mm256_loadu_ps(p + 8), _mm256_loadu_ps(p + 16), _mm256_loadu_ps(p + 24)};
        const __m256 s8 = _mm256_add_ps(_mm256_add_ps(a[0], a[2]), _mm256_add_ps(a[1], a[3]));   // ^16, then ^8
        const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));   // ^4
        const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));                                   // ^2
        const float sum = _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));                // ^1
        const __m256 m8 = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, a[0]), _mm256_andnot_ps(sign, a[1])),
                                        _mm256_max_ps(_mm256_andnot_ps(sign, a[2]), _mm256_andnot_ps(sign, a[3])));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m8), _mm256_extractf128_ps(m8, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        const float amax = _mm_cvtss_f32(_mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1)));
        const float d = std::fabs(amax / 127.0f) > 65504.0f ? std::copysign(65504.0f, amax / 127.0f) : amax / 127.0f;
        int8_t* qs = (int8_t*) (blk + 4);
        if (amax == 0.0f) {
            std::memset(qs, 0, 32);
        } else {
            const __m256 dv = _mm256_set1_ps(d);
            alignas(32) int32_t q[32];
            for (int k = 0; k < 4; ++k) {
                const __m256 v = _mm256_div_ps(a[k], dv);                                    // x / d (IEEE)
                const __m256 t = _mm256_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                const __m256 frac = _mm256_andnot_ps(sign, _mm256_sub_ps(v, t));             // exact for |v| < 2^23
                const __m256 step = _mm256_and_ps(_mm256_cmp_ps(frac, half, _CMP_GE_OQ),
                                                  _mm256_or_ps(one, _mm256_and_ps(sign, v)));   // +-1 away from zero
                const __m256 r = _mm256_min_ps(lim, _mm256_max_ps(nlim, _mm256_add_ps(t, step)));   // roundf, clamped
                _mm256_store_si256((__m256i*) (q + 8 * k), _mm256_cvtps_epi32(r));
            }
            for (int i = 0; i < 32; ++i) qs[i] = (int8_t) q[i];
        }
        const float sf = std::fabs(sum) > 65504.0f ? std::copysign(65504.0f, sum) : sum;
        const uint16_t dh = _cvtss_sh(d, _MM_FROUND_TO_NEAREST_INT), sh = _cvtss_sh(sf, _MM_FROUND_TO_NEAREST_INT);
        std::memcpy(blk, &dh, 2);
        std::memcpy(blk + 2, &sh, 2);
    }
}

bool peer_check() {   // STRATA_PEER_CHECK=1: the rows compared bit for bit with the launches the faster path replaces
    static const bool on = [] { const char* v = std::getenv("STRATA_PEER_CHECK"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}
double us_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

std::string PeerExperts::take_timing() {
    if (t_calls_ == 0) return "";
    const double n = t_calls_;
    char buf[600];
    std::snprintf(buf, sizeof buf,
                  "peer timing: %.0f layer calls, per call: launch() on the host %.1f us, launch to done %.1f us; on the "
                  "GPU: copies %.1f us, input quantization %.1f us, experts %.1f us, scatter %.1f us; finish() waited %.1f us; "
                  "%.1f rows, %.1f experts%s",
                  n, t_host_launch_ / n, t_until_ / n, t_copies_ / n, t_quant_ / n, t_kernels_ / n, t_scatter_ / n,
                  t_wait_ / n, t_rows_ / n, t_groups_ / n,
                  checked_ ? (" (checked " + std::to_string(checked_) + " rows: " + std::to_string(mismatched_) +
                              " different)").c_str() : "");
    t_calls_ = t_host_launch_ = t_copies_ = t_quant_ = t_groups_ = t_kernels_ = t_scatter_ = t_wait_ = t_until_ = t_rows_ = 0;
    return buf;
}

bool PeerExperts::launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                         const int32_t* kind, std::string& err, float* out) {
    const bool timing = peer_timing();
    if (timing) t_launch0_ = std::chrono::steady_clock::now();
    static const bool direct_env = [] { const char* v = std::getenv("STRATA_PEER_DIRECT"); return v == nullptr || std::atoi(v) != 0; }();
    const bool direct = direct_env && out != nullptr;
    launched_direct_ = false;
    launched_rows_ = 0;
    row_of_.clear();
    const int64_t n = n_tok * k;
    if (n > CAP) { err = "peer experts: window too large"; return false; }
    Meta& m = *(Meta*) h_meta_;
    int groups = 0, rows = 0;
    // distinct experts in routing order; each one's entries become compact rows
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != 2) continue;
        bool seen = false;
        for (int64_t j = 0; j < i; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) { seen = true; break; }
        if (seen) continue;
        const int32_t slot = res_[(size_t) (layer * n_expert_ + ids[i])];
        if (slot < 0) { err = "peer experts: an entry was planned for the peer but is not resident"; return false; }
        m.ptr[groups] = (unsigned long long) cache_.device_slot(slot);
        m.start[groups] = rows;
        for (int64_t j = i; j < n; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) {
                m.dst[rows] = rows;
                m.tok[rows] = (int32_t) (j / k);
                m.out_row[rows] = (int32_t) j;
                row_of_.push_back((int32_t) j);
                ++rows;
            }
        ++groups;
        ++experts_;
    }
    if (groups == 0) return true;
    m.start[groups] = rows;
    m.count[0] = groups;
    m.count[1] = rows;
    entries_ += rows;
    launched_rows_ = rows;
    launched_groups_ = groups;
    On on(device_);
    const cudaStream_t s = stream_;
    if (timing && tev_[0] == nullptr)
        for (cudaEvent_t& e : tev_) cudaEventCreate(&e);
    Meta* dm = (Meta*) d_meta_;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) { err = "peer experts: only native packs are supported"; return false; }
    const auto& f = lay.fmt[(size_t) layer];
    auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
    if (f.had_d) { L.had_block = f.had_block; L.d_signs = d_sh_; }
    // the down rows straight into the host `out`, a block's 32 rows a 128-byte store, while the other blocks still
    // compute (no scatter after the experts; STRATA_PEER_DOWN_HOST=0: the scatter, as before)
    const bool down_host = direct && peer_down_host() && strata::kernels::native_expert_down_host_ok(f.d_type, f.n_ff);
    // the input quantized on the host (quantize_q8_1_host: the same bytes; APR's rotation before it stays the device's)
    const bool host_q8 = peer_host_q8_env() && cpu_avx2_f16c() && !f.had_gu;
    const size_t in_f32 = (size_t) (n_tok * H) * sizeof(float), in_q8 = (size_t) n_tok * (H / 32) * kQ8Block;
    if (peer_check() && (down_host || host_q8)) {
        // STRATA_PEER_CHECK: the path these replace, first - the input as floats, the device's quantization, the rows
        // into the device and back (h_out_) - for finish() to compare bit for bit
        std::memcpy(h_x_, x, in_f32);
        if (!ck(cudaMemcpyAsync(d_meta_, h_meta_, kMetaBytes + in_f32, cudaMemcpyHostToDevice, s), "check input", err))
            return false;
        if (f.had_gu) strata::kernels::hadamard_rows(d_x_, d_x_, n_tok, H, f.had_block, d_sx_, s);
        strata::kernels::quantize_q8_1_rows(d_x_, n_tok, H, d_q8_, s);
        strata::kernels::native_expert_grouped(L, dm->ptr, dm->start, dm->count, dm->dst, dm->tok, groups, rows, d_q8_,
                                               d_scratch_, d_out_, s);
        if (!ck(cudaMemcpyAsync(h_out_, d_out_, (size_t) rows * H * sizeof(float), cudaMemcpyDeviceToHost, s), "check rows", err) ||
            !ck(cudaStreamSynchronize(s), "check", err))
            return false;
        check_ref_.assign(h_out_, h_out_ + (size_t) rows * H);
    }
    const void* xq = d_q8_;
    if (host_q8) {   // behind the plan, one copy for both
        quantize_q8_1_host(x, n_tok, H, (uint8_t*) h_x_);
        if (timing) cudaEventRecord(tev_[0], s);
        if (!ck(cudaMemcpyAsync(d_meta_, h_meta_, kMetaBytes + in_q8, cudaMemcpyHostToDevice, s), "plan and input", err))
            return false;
        if (timing) { cudaEventRecord(tev_[1], s); cudaEventRecord(tev_[4], s); }
        xq = d_x_;
    } else {
        std::memcpy(h_x_, x, in_f32);
        if (timing) cudaEventRecord(tev_[0], s);
        if (!ck(cudaMemcpyAsync(d_meta_, h_meta_, kMetaBytes + in_f32, cudaMemcpyHostToDevice, s), "plan and input", err))
            return false;
        if (timing) cudaEventRecord(tev_[1], s);
        // APR: Hadamard-folded gate/up take H (s * x); d_x_ is this device's private copy, rotated in place
        if (f.had_gu) strata::kernels::hadamard_rows(d_x_, d_x_, n_tok, H, f.had_block, d_sx_, s);
        strata::kernels::quantize_q8_1_rows(d_x_, n_tok, H, d_q8_, s);
        if (timing) cudaEventRecord(tev_[4], s);
    }
    if (down_host) {
        L.out_host = true;
        strata::kernels::native_expert_grouped(L, dm->ptr, dm->start, dm->count, dm->out_row, dm->tok, groups, rows, xq,
                                               d_scratch_, out, s);
        if (timing) { cudaEventRecord(tev_[2], s); cudaEventRecord(tev_[3], s); t_host_launch_ += us_since(t_launch0_); }
        launched_direct_ = true;
        return true;
    }
    strata::kernels::native_expert_grouped(L, dm->ptr, dm->start, dm->count, dm->dst, dm->tok, groups, rows, xq,
                                           d_scratch_, d_out_, s);
    if (timing) cudaEventRecord(tev_[2], s);
    if (direct) {   // the rows straight into the host `out` (zero-copy, coalesced float4 writes)
        try {
            strata::kernels::scatter_rows_f32(d_out_, out, dm->out_row, rows, H, s);
        } catch (const std::exception& e) { err = std::string("peer experts: scatter: ") + e.what(); return false; }
        launched_direct_ = true;
        if (timing) { cudaEventRecord(tev_[3], s); t_host_launch_ += us_since(t_launch0_); }
        return true;
    }
    const bool ok = ck(cudaMemcpyAsync(h_out_, d_out_, (size_t) rows * H * sizeof(float), cudaMemcpyDeviceToHost, s),
                       "results", err);
    if (timing) { cudaEventRecord(tev_[3], s); t_host_launch_ += us_since(t_launch0_); }
    return ok;
}

bool PeerExperts::finish(float* out, std::string& err) {
    if (launched_rows_ == 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    {
        On on(device_);
        // spin on the stream (a blocking sync would sleep the pool thread and wake it late)
        cudaError_t e;
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!ck(e, "finish", err)) return false;
    }
    ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (peer_timing() && tev_[3] != nullptr) {
        On on(device_);
        float a = 0, b = 0, c = 0, q = 0;
        cudaEventElapsedTime(&a, tev_[0], tev_[1]);
        cudaEventElapsedTime(&q, tev_[1], tev_[4]);
        cudaEventElapsedTime(&b, tev_[4], tev_[2]);
        cudaEventElapsedTime(&c, tev_[2], tev_[3]);
        t_copies_ += 1000.0 * a, t_quant_ += 1000.0 * q, t_kernels_ += 1000.0 * b, t_scatter_ += 1000.0 * c;
        t_wait_ += us_since(t0), t_until_ += us_since(t_launch0_), t_rows_ += (double) launched_rows_, t_calls_ += 1;
        t_groups_ += (double) launched_groups_;
    }
    if (launched_direct_ && !check_ref_.empty() && out != nullptr) {   // STRATA_PEER_CHECK: these rows vs the launches'
        for (size_t r = 0; r < row_of_.size(); ++r) {
            ++checked_;
            if (std::memcmp(out + (size_t) row_of_[r] * H, check_ref_.data() + r * H, (size_t) H * sizeof(float)) != 0)
                ++mismatched_;
        }
        check_ref_.clear();
    }
    if (launched_direct_) { launched_rows_ = 0; return true; }
    for (size_t r = 0; r < row_of_.size(); ++r)
        std::memcpy(out + (size_t) row_of_[r] * H, h_out_ + r * H, (size_t) H * sizeof(float));
    launched_rows_ = 0;
    return true;
}

bool PeerExperts::adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err) {
    if (!pending_.empty() || max_swaps <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * n_expert_;
        const int32_t* r0 = res0 + l * n_expert_;
        const int32_t* r1 = res_.data() + l * n_expert_;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            if (r1[e] >= 0) vict.emplace_back(u[e], e);
            else if (r0[e] < 0 && u[e] >= 2.0f) cand.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    if (swaps.empty()) return true;
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    On on(device_);
    std::vector<strata::kernels::CopyJob> jobs;
    for (const Swap& s : swaps) {
        const size_t in = (size_t) (s.layer * n_expert_ + s.in), out = (size_t) (s.layer * n_expert_ + s.out);
        const int32_t slot = res_[out];
        const uint8_t* b = src_->blob(s.layer, s.in);
        if (slot < 0 || b == nullptr) continue;
        const size_t bytes = (size_t) lay.blob_bytes(s.layer);
        uint8_t* dst = cache_.device_slot(slot);
        if (refill_sm_ > 0 && src_->device_alias(s.layer, s.in) == b && (((uintptr_t) b | (uintptr_t) dst | bytes) & 15) == 0)
            jobs.push_back({b, dst, (unsigned long long) bytes});   // registered memory: SM loads (refill_jobs)
        else if (!ck(cudaMemcpyAsync(dst, b, bytes, cudaMemcpyHostToDevice, refill_), "refill", err))
            return false;
        res_[out] = kNotResident;              // evicted now: the CPU computes it meanwhile
        pending_.emplace_back((int32_t) in, slot);
        incoming_[in] = 1;
        ++swaps_;
    }
    if (!refill_jobs(jobs.data(), jobs.size(), err)) return false;
    return ck(cudaEventRecord(refill_ev_, refill_), "refill event", err);
}

void PeerExperts::apply_pending(bool wait) {
    if (pending_.empty()) return;
    {
        On on(device_);
        if (wait) cudaEventSynchronize(refill_ev_);
        else if (cudaEventQuery(refill_ev_) != cudaSuccess) return;
    }
    for (const auto& [i, slot] : pending_) {
        res_[(size_t) i] = slot;
        incoming_[(size_t) i] = 0;
    }
    pending_.clear();
}

}  // namespace strata::core
