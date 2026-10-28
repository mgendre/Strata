// --conversation-cache-disk DIR: the conversation cache's two disk levels, under the RAM cache.
//
//   swap  short term: the conversations RAM pushes out (its oldest) or cannot hold (no free RAM); the least recently
//         used one leaves first; emptied at every start.
//   big   long term: the conversations that keep coming back.  A usage score counts every return (from RAM or disk)
//         and halves every `half_life_s`; a conversation that leaves RAM with a score of `promote_score` or more goes
//         straight here, one that comes back from swap moves here (its files stay where they are: no write), and when
//         it is full the lowest score goes back to swap.  Kept across restarts of the same engine build and model.
//
// A conversation is stored as its token ids, ONE running state (the deepest checkpoint near its end, else the live
// state: conversation_disk_state) and the K/V up to that point, in blocks of `block_tokens` tokens named by a hash
// of their bytes.  A block already on disk is never written again: a conversation that comes back and leaves again
// writes only its new blocks and its running state, and conversations that share a start share its blocks.  A copy
// that a conversation outgrew by fewer than `refresh_tokens` is kept as it is.  Nothing is flushed for durability
// (a cache: a file that a crash cut short fails its hash and is dropped).
//
// Pure host code (no CUDA): the K/V comes in through SessionKvSource readers and goes back out through SessionKvSink
// writers, one per K/V layer (the session's QSA layers, then the draft layer), which the engine provides.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {

/// One K/V layer written back from disk: `write(part, offset, src, n)` puts bytes [offset, offset+n) of part
/// 0..4 = k, v, k_scale, v_scale, pooled where the layer keeps them; `finish` runs once every block is in (the
/// residency refresh).  After a failure `error()` says why.
struct SessionKvSink {
    std::array<size_t, 5> sizes{};
    std::function<bool(size_t part, size_t offset, const void* src, size_t n)> write;
    std::function<bool()> finish;
    std::function<std::string()> error;
};

/// A parked image's K/V (host buffers) as sources, one per layer, for a disk write.
std::vector<SessionKvSource> conversation_image_sources(const SavedConversation& image);

/// The running state a disk copy keeps: the deepest checkpoint on the live state's path that is at most `near` tokens
/// short of it (a client that re-renders its last reply resumes there too), else the live state itself.
const ConversationCheckpoint& conversation_disk_state(const ConversationCheckpoint& live,
                                                      const std::vector<ConversationCheckpoint>& checkpoints,
                                                      int64_t near);

/// A usage score `score` stamped at `at` (unix seconds), seen at `now`: halved every `half_life_s`.
double conversation_score_at(double score, int64_t at, int64_t now, double half_life_s);

struct ConversationDiskOptions {
    std::string dir;
    uint64_t swap_bytes = 16ull << 30, big_bytes = 50ull << 30;
    int64_t min_tokens = 32768, refresh_tokens = 4096, block_tokens = 4096;
    double half_life_s = 86400.0, promote_score = 2.0;
    uint64_t day_guide_bytes = 200ull << 30;   ///< written per day above which the monitor warns (nothing stops)
};

/// What the files are bound to: the model inputs and the settings that change what the saved bytes mean (the session
/// files' identity), this engine binary, and the K/V layout (layer count and part sizes).
struct ConversationDiskIdentity {
    uint64_t model = 0, config = 0, build = 0, layout = 0;
    bool operator==(const ConversationDiskIdentity&) const = default;
};

/// The bytes of each K/V part (k, v, k_scale, v_scale, pooled) of layer `layer` holding `tokens` tokens.
using ConversationDiskSizes = std::function<bool(size_t layer, int64_t tokens, std::array<uint64_t, 5>& sizes)>;

struct ConversationDiskMatch {
    uint64_t id = 0;
    int64_t tokens = 0;   ///< the resume point the copy gives this prompt (its stored length)
    bool big = false;
    explicit operator bool() const { return tokens > 0; }
};

/// One write or read.
struct ConversationDiskStats {
    uint64_t written = 0, reused = 0, read = 0;   ///< bytes: written to disk, already on disk (not written), read
    size_t blocks = 0, new_blocks = 0;
    int64_t tokens = 0;
    bool skipped = false;                         ///< the copy on disk was kept as it is (nothing written)
    bool big = false;                             ///< the level the conversation is on now
    bool promoted = false;                        ///< moved from swap to big by this operation
    bool guide_crossed = false;                   ///< today's writes went past the day guide with this write
    double score = 0;                             ///< its usage score after this operation
    std::string why;                              ///< a skip's or a refusal's reason
};

class ConversationDisk {
public:
    /// Creates the folder, empties swap, keeps the big entries whose identity matches and drops every other file.
    /// `layers`: the K/V layer count.  `note` gets one line for the log.
    bool open(const ConversationDiskOptions& options, const ConversationDiskIdentity& identity, size_t layers,
              ConversationDiskSizes sizes, std::string& note, std::string& error);
    bool enabled() const { return enabled_; }
    const ConversationDiskOptions& options() const { return opt_; }
    size_t layers() const { return layers_; }

    /// The copy that resumes `prompt` furthest (its stored length, below the prompt's), from the token ids alone.
    template<class Token>
    ConversationDiskMatch best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                               bool cvec) const {
        ConversationDiskMatch m;
        for (const Entry& e : entries_) {
            if (e.cvec != cvec) continue;
            const int64_t n = conversation_prefix(e.meta, prompt, images);
            if (n > m.tokens) m = {e.id, n, e.big};
        }
        return m;
    }

    struct Write {
        const std::vector<int32_t>& ids;                   ///< the conversation up to its stored running state
        const std::vector<ConversationImageKey>& imgs;     ///< its pictures below that
        bool cvec;
        const ConversationCheckpoint& state;               ///< the running state after `ids`
        const std::vector<SessionKvSource>& kv;            ///< one per K/V layer, holding at least ids.size() tokens
        double score;                                      ///< its usage score now
    };
    /// Stores a conversation (false: an I/O or input error, `error` says it; a skip is true with stats.skipped).
    bool spill(const Write& w, ConversationDiskStats& stats, std::string& error);
    /// Streams copy `m` into `kv` (one sink per layer) and its running state into `state` (ids and pictures too).
    /// Counts the return: the score goes up by one and a swap copy moves to big.  On false the sinks may hold part
    /// of the copy: the caller must not continue from that session.
    bool load(const ConversationDiskMatch& m, ConversationCheckpoint& state, bool& cvec,
              const std::vector<SessionKvSink>& kv, ConversationDiskStats& stats, std::string& error);
    /// Drops every copy (#879: after non-finite logits nothing that may hold the poison is kept).
    size_t clear();

    struct Level { size_t entries = 0; uint64_t bytes = 0, budget = 0; };
    Level swap() const { return level(false); }
    Level big() const { return level(true); }
    uint64_t physical_bytes() const;                       ///< the folder's files: shared blocks counted once
    struct Counters {
        uint64_t written = 0, read = 0, reused = 0;        ///< bytes
        uint64_t spills = 0, skipped = 0, restores_swap = 0, restores_big = 0, tokens_swap = 0, tokens_big = 0;
        uint64_t promotions = 0, demotions = 0, deleted = 0;
    };
    const Counters& since_start() const { return start_; }
    uint64_t written_all_time() const { return total_written_; }
    uint64_t read_all_time() const { return total_read_; }
    uint64_t written_today() const { return day_written_; }
    /// `<dir>/status.json`: the levels, the counters and one row per copy, for the server's monitor.
    std::string status_json() const;

private:
    struct Hash { uint64_t a = 0, b = 0; bool operator==(const Hash&) const = default; };
    struct HashFn { size_t operator()(const Hash& h) const { return (size_t) (h.a ^ (h.b * 0x9E3779B97F4A7C15ull)); } };
    struct Entry {
        uint64_t id = 0;
        bool big = false, cvec = true;
        ConversationCheckpoint meta;                       ///< ids and imgs only
        std::vector<Hash> blocks;
        std::vector<uint64_t> block_bytes;
        Hash state_hash;
        uint64_t state_bytes = 0, written = 0, served_tokens = 0, returns = 0;
        double score = 0;
        int64_t score_at = 0, created = 0, last_used = 0;
        uint64_t bytes() const;
    };
    struct Block { uint64_t bytes = 0; size_t refs = 0; };

    Level level(bool big) const;
    std::string path(const char* sub, const std::string& name) const;
    std::string block_name(const Hash& h) const;
    std::string entry_name(uint64_t id, const char* ext) const;
    bool slices(int64_t t0, int64_t t1, std::vector<std::array<uint64_t, 5>>& from,
                std::vector<std::array<uint64_t, 5>>& to, std::string& error) const;
    bool write_block(const Write& w, int64_t t0, int64_t t1, Hash& hash, uint64_t& bytes, bool& fresh,
                     std::string& error);
    bool write_state(const Entry& e, const ConversationCheckpoint& state, Hash& hash, uint64_t& bytes, std::string& error);
    bool read_state(const Entry& e, ConversationCheckpoint& state, uint64_t& bytes, std::string& error);
    bool write_ids(const Entry& e, std::string& error);
    bool read_ids(Entry& e, std::string& error);
    bool write_manifest(const Entry& e, std::string& error);
    bool read_manifest(const std::string& file, Entry& e, ConversationDiskIdentity& id, std::string& error);
    void ref(const Entry& e);
    void unref(const Entry& e);
    void remove(size_t index);
    void fit(uint64_t keep_id);
    void count_written(uint64_t bytes, ConversationDiskStats& stats);
    void save_counters() const;
    void load_counters();
    void publish() const;
    int64_t now() const;

    bool enabled_ = false;
    ConversationDiskOptions opt_;
    ConversationDiskIdentity identity_;
    size_t layers_ = 0;
    ConversationDiskSizes sizes_;
    std::vector<Entry> entries_;
    std::unordered_map<Hash, Block, HashFn> blocks_;
    std::vector<uint8_t> buffer_;                          ///< one 16 MiB piece at a time
    uint64_t next_id_ = 1;
    Counters start_;
    uint64_t total_written_ = 0, total_read_ = 0, day_written_ = 0;
    int64_t day_ = 0;                                      ///< yyyymmdd (local time) of day_written_
};

} // namespace strata::core
