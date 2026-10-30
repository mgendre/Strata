// include/strata/core/peer_experts.hpp - multi-GPU: a second expert tier on another GPU.
//
// The primary GPU (device 0) keeps everything it had: the dense weights, the KV cache, the MTP layer, its own
// adaptive expert tier and the captured verify graph.  This class adds an expert tier on a SECOND card.  From
// the verify graph's point of view nothing changes: for each layer the pool thread decides which routed experts
// the primary computes (its VRAM tier, the PCIe share), and every remaining expert is a "CPU row" the graph reads
// from mapped host memory after the pool raises its flag.  The peer takes the rows whose experts it holds: the
// pool thread launches them on the peer right after publishing the primary's plan, the CPU computes the rest at
// the same time, and the peer's rows are written into the same mapped rows before the pool returns.
//
// The peer computes with the SAME kernels as the primary's hit path (q8_1 activations + native_expert_grouped),
// from the same activation values, so an expert gives bit-identical rows on either card.
//
// The tier follows the conversation like the primary's: `adapt` swaps the most-routed experts that neither card
// holds into the peer's least-used slots (per layer, so a slot keeps its layer's blob size).
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::kernels { struct CopyJob; }

namespace strata::core {

/// multi-GPU: set once at startup when a peer device is configured (--peer-device N).  The mapped host buffers
/// the peer card writes into are then allocated with cudaHostAllocPortable; without a peer the flag stays off.
void set_peer_portable(bool on);
bool peer_portable();

class PeerExperts {
public:
    PeerExperts() = default;
    ~PeerExperts();
    PeerExperts(const PeerExperts&) = delete;
    PeerExperts& operator=(const PeerExperts&) = delete;

    /// Opens the tier on `device`: fills it with the first pairs of `ranked` that `primary` does not hold, as many
    /// as fit in the card's free memory minus `reserve_mib` (`max_slots` > 0 caps the count).
    bool open(int device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
              ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots,
              std::string& err);
    void close();
    bool valid() const { return device_ >= 0; }
    int device() const { return device_; }

    /// Whether the peer holds (layer, expert) right now.
    bool has(int64_t layer, int64_t expert) const { return res_[(size_t) (layer * n_expert_ + expert)] >= 0; }
    /// Whether (layer, expert) is on its way into the peer (an adaptive round's copy that `apply_pending` has not
    /// taken yet).  The primary's own round skips these: taking one too would leave both cards holding it.
    bool incoming(int64_t layer, int64_t expert) const {
        return !incoming_.empty() && incoming_[(size_t) (layer * n_expert_ + expert)] != 0;
    }

    /// The device address of (layer, expert)'s blob on the peer, or null when it is not resident.
    const uint8_t* slot_ptr(int64_t layer, int64_t expert) {
        const int32_t sl = res_[(size_t) (layer * n_expert_ + expert)];
        return sl >= 0 ? cache_.device_slot(sl) : nullptr;
    }
    /// Whether the primary (device 0) and the peer can read each other's memory directly (NVLink / PCIe P2P).
    bool p2p() const { return p2p_; }

    /// Starts the peer's share of one layer: the entries i with kind[i] == 2 (x: n_tok rows of n_embd floats).
    /// multi-GPU: with `out` (pinned, portable host rows) the peer writes its rows straight into
    /// out[entry] with a scatter kernel (no D2H copy + host memcpy in finish); STRATA_PEER_DIRECT=0: the old way.
    bool launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, const int32_t* kind,
                std::string& err, float* out = nullptr);
    /// Waits for the launched share and writes its rows into `out` (row i = entry i, n_embd floats).
    bool finish(float* out, std::string& err);

    /// Adaptive tier: swap up to `max_swaps` experts routed at least twice that neither the primary (`res0`, slot
    /// or -1) nor the peer holds into the peer's least-used slots.  Copies run on the peer's refill stream; a
    /// swapped-in expert becomes resident at `apply_pending`.  Call only between verify windows.
    bool adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err);
    void apply_pending(bool wait);

    int64_t resident() const { return resident_; }
    /// the last launch writes its rows into the host rows itself (the caller must not touch them)
    bool launched_direct() const { return launched_direct_; }
    double gib() const { return cache_.gib(); }
    int64_t entries() const { return entries_; }      ///< routed entries the peer computed
    int64_t experts() const { return experts_; }      ///< distinct (layer, expert) the peer computed
    int64_t swaps() const { return swaps_; }
    int refill_sm() const { return refill_sm_; }      ///< SM blocks of the adaptive refill (0 = the copy engine)
    /// Before `open`: the adaptive refill by `blocks` blocks of SM loads (0 = the copy engine, the default).
    void set_refill_sm(int blocks) { refill_sm_want_ = blocks; }
    /// --peer-slow-link-optimizations (off by default): for a peer on a narrow link, its input is quantized on the host (3.5x fewer
    /// bytes in) and its down rows go straight to the host rows while it computes (no scatter after); the same bits
    void set_slow_link(bool on) { slow_link_ = on; }
    bool slow_link() const { return slow_link_; }
    double ms_wait = 0;                               ///< host time spent in finish() waiting for the peer
    /// STRATA_PEER_TIMING=1 (a diagnostic): where a layer's peer call goes, summed since the last call and reset -
    /// "" when off or nothing was launched
    std::string take_timing();

private:
    bool refill_jobs(const strata::kernels::CopyJob* jobs, size_t n, std::string& err);
    int device_ = -1;
    bool p2p_ = false;
    int64_t n_layers_ = 0, n_expert_ = 0;
    ExpertCache cache_;
    ExpertSource* src_ = nullptr;
    std::vector<int32_t> res_;                        ///< [n_layers * n_expert] -> peer slot or -1
    std::vector<std::pair<int32_t, int32_t>> pending_;   ///< (residency index, slot) once the copies land
    std::vector<uint8_t> incoming_;                      ///< [n_layers * n_expert] -> 1 while in pending_
    int refill_sm_ = 0;                               ///< adaptive refill: SM copy blocks (0 = the copy engine)
    int refill_sm_want_ = 0;                          ///< asked for (set_refill_sm), kept across close()
    bool slow_link_ = false;                          ///< --peer-slow-link-optimizations (set_slow_link), kept across close()
    void* h_jobs_ = nullptr;                          ///< its job table (pinned, mapped), `jobs_cap_` entries
    void* d_jobs_ = nullptr;
    size_t jobs_cap_ = 0;
    int64_t resident_ = 0, entries_ = 0, experts_ = 0, swaps_ = 0;
    cudaStream_t stream_ = nullptr, refill_ = nullptr;
    cudaEvent_t refill_ev_ = nullptr;
    float* h_x_ = nullptr;   float* d_x_ = nullptr;   // n_tok x n_embd
    float* h_out_ = nullptr; float* d_out_ = nullptr; // compact rows
    void* h_meta_ = nullptr; void* d_meta_ = nullptr; // groups: ptr | start | dst | tok | count
    uint8_t* d_q8_ = nullptr;
    void* d_scratch_ = nullptr;
    const float* d_sx_ = nullptr;                     // APR: device sign vectors (x: H, h: FF); nullptr = identity
    const float* d_sh_ = nullptr;
    std::vector<int32_t> row_of_;                     ///< compact row -> original entry
    int64_t launched_rows_ = 0;
    bool launched_direct_ = false;
    // STRATA_PEER_CHECK=1: the rows of the faster path compared bit for bit with the launches it replaces (check_ref_:
    // those rows; checked_ / mismatched_: rows compared / different)
    std::vector<float> check_ref_;
    int64_t checked_ = 0, mismatched_ = 0;
    // STRATA_PEER_TIMING: device events around the copies, the kernels and the scatter; host clocks around launch()
    // and finish(); sums in microseconds
    cudaEvent_t tev_[5] = {};   // before the copies, after them, after the input's quantization, after the experts, after the scatter
    std::chrono::steady_clock::time_point t_launch0_{};
    double t_calls_ = 0, t_host_launch_ = 0, t_copies_ = 0, t_quant_ = 0, t_kernels_ = 0, t_scatter_ = 0, t_wait_ = 0, t_until_ = 0;
    double t_rows_ = 0, t_groups_ = 0;
    int64_t launched_groups_ = 0;
};

}  // namespace strata::core
