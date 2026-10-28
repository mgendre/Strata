// conversation_disk_test [folder]: the disk levels of the conversation cache (conversation_disk.hpp) on a fake K/V
// layout - two layers (a QSA layer with an indexer, a draft layer without), parts that grow by page - in a scratch
// folder (default: the system's temporary folder).  No GPU.
#include "strata/core/conversation_disk.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}

constexpr int64_t kPage = 16, kIdx = 32;
bool sizes(size_t layer, int64_t t, std::array<uint64_t, 5>& s) {
    if (t < 0 || layer > 1) return false;
    const uint64_t cells = (uint64_t) ((t + kPage - 1) / kPage * kPage);
    s = {cells * 8, cells * 8, cells * 2, cells * 2, layer == 0 && t > 0 ? (uint64_t) (t / kIdx + 1) * 4 : 0};
    return true;
}

// a conversation's K/V: every byte a function of the conversation's seed (its first token) and where it is, so a
// longer turn of the same conversation has the same bytes in front
uint8_t byte_at(int32_t seed, size_t layer, size_t part, size_t i) {
    uint64_t x = (uint64_t) seed * 0x9E3779B97F4A7C15ull ^ (layer * 1315423911u) ^ (part * 2654435761u) ^ (i * 0x100000001B3ull);
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    return (uint8_t) x;
}

struct Conv {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    ConversationCheckpoint state;
    std::vector<SessionKvSource> kv;
};
Conv conv(int32_t seed, int64_t tokens) {
    Conv c;
    for (int64_t i = 0; i < tokens; ++i) c.ids.push_back(i == 0 ? seed : (int32_t) (i * 7 + seed));
    c.state.ids = c.ids;
    c.state.gdn.assign(100, (uint8_t) (seed + tokens));
    c.state.ple.assign(10, 2);
    c.state.tails.assign(20, 3);
    c.state.dead.assign(4, 4);
    c.state.block_pos.assign(8, 5);
    for (size_t l = 0; l < 2; ++l) {
        SessionKvSource s;
        std::array<uint64_t, 5> z{};
        sizes(l, tokens, z);
        for (size_t p = 0; p < 5; ++p) s.sizes[p] = z[p];
        s.read = [seed, l, z](size_t part, size_t offset, void* dst, size_t n) {
            if (part >= 5 || offset + n > z[part]) return false;
            for (size_t i = 0; i < n; ++i) static_cast<uint8_t*>(dst)[i] = byte_at(seed, l, part, offset + i);
            return true;
        };
        c.kv.push_back(std::move(s));
    }
    return c;
}

struct Target {
    std::vector<std::array<std::vector<uint8_t>, 5>> parts{2};
    std::vector<SessionKvSink> sinks;
    int finished = 0;
    explicit Target(int64_t tokens) {
        for (size_t l = 0; l < 2; ++l) {
            std::array<uint64_t, 5> z{};
            sizes(l, tokens, z);
            SessionKvSink s;
            for (size_t p = 0; p < 5; ++p) { parts[l][p].assign(z[p], 0); s.sizes[p] = z[p]; }
            s.write = [this, l](size_t part, size_t offset, const void* src, size_t n) {
                if (part >= 5 || offset + n > parts[l][part].size()) return false;
                std::memcpy(parts[l][part].data() + offset, src, n);
                return true;
            };
            s.finish = [this] { ++finished; return true; };
            sinks.push_back(std::move(s));
        }
    }
    bool holds(int32_t seed) const {
        for (size_t l = 0; l < 2; ++l)
            for (size_t p = 0; p < 5; ++p)
                for (size_t i = 0; i < parts[l][p].size(); ++i)
                    if (parts[l][p][i] != byte_at(seed, l, p, i)) return false;
        return true;
    }
};

std::vector<int32_t> extend(const std::vector<int32_t>& ids, int n) {
    std::vector<int32_t> out = ids;
    for (int i = 0; i < n; ++i) out.push_back(99000 + i);
    return out;
}

ConversationDiskOptions options(const std::string& dir) {
    ConversationDiskOptions o;
    o.dir = dir;
    o.block_tokens = 32;
    o.min_tokens = 50;
    o.refresh_tokens = 8;
    o.budget_bytes = 1 << 20;
    o.day_guide_bytes = 1;
    return o;
}

ConversationDisk::Write write_of(const Conv& c, double score = 0) {
    return ConversationDisk::Write{c.ids, c.imgs, true, c.state, c.kv, score};
}
Conv longer(const Conv& base, int32_t seed, int n) {   // the same conversation, n tokens on
    Conv c = conv(seed, (int64_t) base.ids.size() + n);
    c.ids = extend(base.ids, n);
    c.state.ids = c.ids;
    return c;
}
} // namespace

int main(int argc, char** argv) {
    const fs::path root = fs::path(argc > 1 ? argv[1] : fs::temp_directory_path().string()) / "conversation_disk_test";
    std::error_code ec;
    fs::remove_all(root, ec);
    const ConversationDiskIdentity id{1, 2, 3, 4};
    std::string note, error;

    {   // helpers
        ConversationCheckpoint live;
        live.ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        live.gdn.assign(4, 1);
        std::vector<ConversationCheckpoint> chain(3);
        chain[0].ids = {1, 2, 3};
        chain[1].ids = {1, 2, 3, 4, 5, 6, 7, 8};
        chain[2].ids = {1, 2, 9, 9, 9, 9, 9, 9, 9};   // another branch
        for (auto& c : chain) c.gdn.assign(4, 2);
        check(&conversation_disk_state(live, chain, 4) == &chain[1], "the deepest checkpoint on the path near the end");
        check(&conversation_disk_state(live, chain, 1) == &live, "too far back: the live state");
        check(std::abs(conversation_score_at(4.0, 0, 100, 100.0) - 2.0) < 1e-9, "a score halves in one half-life");
        check(conversation_score_at(4.0, 50, 10, 100.0) == 4.0, "no decay backwards in time");
    }

    ConversationDisk disk;
    check(disk.open(options(root.string()), id, 2, sizes, note, error), "opens an empty folder");
    check(fs::exists(root / "status.json") && fs::exists(root / "blocks") && fs::exists(root / "conv"), "creates its files");

    // a conversation of 100 tokens: four blocks (32, 32, 32, 4), all new
    Conv a = conv(11, 100);
    ConversationDiskStats st;
    check(disk.save(write_of(a), st, error) && !st.skipped && st.blocks == 4 && st.new_blocks == 4 && st.written > 0 &&
          st.reused == 0, "a first copy writes every block");
    check(disk.usage().entries == 1 && disk.saved_tokens(extend(a.ids, 5), {}, true) == 100, "it is on disk");
    check(disk.saved_tokens(extend(a.ids, 5), {}, false) == 0 && disk.saved_tokens({1, 2, 3}, {}, true) == 0,
          "saved_tokens: another steering mode or another conversation has none");
    check(st.guide_crossed && disk.written_today() == st.written, "the day guide is crossed and counted, nothing stops");

    Conv short_one = conv(12, 40);
    check(disk.save(write_of(short_one), st, error) && st.skipped && disk.usage().entries == 1, "a short conversation is not stored");

    const auto prompt = extend(a.ids, 10);
    ConversationDiskMatch m = disk.best(prompt, {}, true);
    check(m && m.tokens == 100, "a prompt that continues it finds the copy");
    check(!disk.best(a.ids, {}, true), "the copy must be shorter than the prompt");
    check(!disk.best(prompt, {}, false), "another steering mode never matches");
    check(!disk.best(prompt, {{5, 1}}, true), "a picture the copy does not have never matches");

    Target t(100);
    ConversationCheckpoint state;
    bool cvec = false;
    check(disk.load(m, state, cvec, t.sinks, st, error), "loads the copy");
    check(t.holds(11) && t.finished == 2, "every K/V byte comes back, every layer refreshed");
    check(state.ids == a.ids && state.gdn == a.state.gdn && state.block_pos == a.state.block_pos && cvec,
          "the running state, ids and steering come back");
    check(st.read > 0 && std::abs(st.score - 1.0) < 1e-6 && disk.since_start().restores == 1 &&
          disk.since_start().tokens == 100, "the return is counted: score 1");

    // the same conversation, 40 tokens on (its next save): only the blocks past the full ones on disk are written
    Conv a2 = longer(a, 11, 40);
    check(disk.save(write_of(a2, 1.0), st, error) && !st.skipped, "a longer turn is stored");
    check(st.blocks == 5 && st.new_blocks == 2 && st.reused > 0, "the three full blocks are named, not written");
    check(disk.usage().entries == 1 && disk.best(extend(a2.ids, 5), {}, true).tokens == 140, "it replaces its copy");
    Conv a3 = longer(a2, 11, 5);
    const uint64_t before = disk.written_all_time();
    check(disk.save(write_of(a3), st, error) && st.skipped && disk.written_all_time() == before,
          "fewer than refresh_tokens new tokens: the copy on disk stays, no K/V written");
    check(disk.save(write_of(a2), st, error) && st.skipped, "the same state again: nothing written");

    // a restart keeps every copy; over the budget the lowest score leaves first, then the least recently used
    {
        Conv b = conv(21, 100), c = conv(31, 100), hot = conv(41, 100);
        check(disk.save(write_of(b), st, error) && disk.usage().entries == 2, "b stored (score 0)");
        const uint64_t one = disk.usage().bytes;   // (a2 and b)
        ConversationDiskOptions small = options(root.string());
        small.budget_bytes = one + 10;                     // room for what is there, not for one more
        ConversationDisk d2;
        check(d2.open(small, id, 2, sizes, note, error) && d2.usage().entries == 2, "a restart keeps both copies");
        check(d2.save(write_of(c), st, error) && d2.usage().entries == 2 && !d2.best(extend(b.ids, 3), {}, true) &&
              d2.best(extend(a2.ids, 3), {}, true) && d2.best(extend(c.ids, 3), {}, true),
              "c pushes out b (score 0, the older of the two with none), a2 (score 1) stays");
        check(d2.save(write_of(hot, 3.0), st, error) && !d2.best(extend(c.ids, 3), {}, true) &&
              d2.best(extend(hot.ids, 3), {}, true) && d2.best(extend(a2.ids, 3), {}, true),
              "a copy with a score pushes out the one without");
        check(d2.since_start().deleted >= 2, "the deletions are counted");
        Target t2(100);
        ConversationCheckpoint s2;
        bool cv = false;
        const auto mh = d2.best(extend(hot.ids, 1), {}, true);
        check(mh && d2.load(mh, s2, cv, t2.sinks, st, error) && t2.holds(41), "a copy loads after a restart");
    }

    // the oldest copy with the worst score leaves first; a copy just saved is not the first to go
    {
        int64_t clock = 1000000;
        ConversationDiskOptions probe = options((root / "probe").string());
        ConversationDisk dp;
        Conv p0 = conv(80, 100);
        check(dp.open(probe, id, 2, sizes, note, error) && dp.save(write_of(p0), st, error), "keep: measure one copy");
        const uint64_t one = dp.usage().bytes;
        ConversationDiskOptions o = options((root / "keep").string());
        o.clock = [&clock] { return clock; };
        o.budget_bytes = 3 * one + 10;                     // room for three copies of this size
        ConversationDisk d;
        Conv used = conv(81, 100), idle = conv(82, 100), fresh = conv(83, 100), newest = conv(84, 100), last = conv(85, 100);
        check(d.open(o, id, 2, sizes, note, error), "keep: open");
        check(d.save(write_of(used, 1.0), st, error) && d.save(write_of(idle), st, error), "day 0: one that came back once, one never");
        clock += 3 * 86400;
        check(d.save(write_of(fresh), st, error) && d.usage().entries == 3, "day 3: a fresh copy (score 0), three on disk");
        // keep values on day 3: used 1/8 + 1/8, idle 0 + 1/8, fresh 0 + 1
        check(d.save(write_of(newest), st, error) && d.usage().entries == 3 && !d.best(extend(idle.ids, 1), {}, true) &&
              d.best(extend(used.ids, 1), {}, true) && d.best(extend(fresh.ids, 1), {}, true),
              "a fourth pushes out the old copy that never came back, not the fresh one");
        check(d.save(write_of(last), st, error) && !d.best(extend(used.ids, 1), {}, true) &&
              d.best(extend(fresh.ids, 1), {}, true) && d.best(extend(newest.ids, 1), {}, true),
              "then the old one that came back once (its score faded), still not the fresh one");
        check(std::abs(conversation_disk_keep(1.0, 0, 0, 3 * 86400, 86400.0) - 0.25) < 1e-9 &&
              std::abs(conversation_disk_keep(0.0, 0, 3 * 86400, 3 * 86400, 86400.0) - 1.0) < 1e-9, "keep values");
        const std::string json = d.status_json();
        check(json.find("\"keep\":") != std::string::npos, "status.json rows carry the keep value");
    }

    // a damaged block: the load fails and the copy is not offered again
    {
        ConversationDisk d3;
        check(d3.open(options(root.string()), id, 2, sizes, note, error), "reopen");
        Conv e = conv(61, 64);
        check(d3.save(write_of(e), st, error) && !st.skipped, "e stored");
        for (const auto& item : fs::directory_iterator(root / "blocks")) {
            std::fstream f(item.path(), std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(-1, std::ios::end);
            f.put('\x5a');
        }
        Target t3(64);
        ConversationCheckpoint s3;
        bool cv = false;
        const auto me = d3.best(extend(e.ids, 1), {}, true);
        check(me && !d3.load(me, s3, cv, t3.sinks, st, error) && !d3.best(extend(e.ids, 1), {}, true),
              "a damaged copy fails to load and is dropped");
        const std::string json = d3.status_json();
        check(json.find("\"disk\":{") != std::string::npos && json.find("\"written_all_time\":") != std::string::npos &&
              json.find("\"entries\":[") != std::string::npos, "status.json carries usage, counters and rows");
        check(d3.clear() >= 1 && d3.usage().entries == 0, "clear drops every copy");
    }

    // another engine build: the copies are dropped, no stray file stays
    {
        ConversationDisk d4;
        Conv f = conv(71, 100);
        check(d4.open(options(root.string()), id, 2, sizes, note, error) && d4.save(write_of(f, 3.0), st, error),
              "f stored");
        ConversationDisk d5;
        ConversationDiskIdentity other = id;
        other.build = 99;
        check(d5.open(options(root.string()), other, 2, sizes, note, error) && d5.usage().entries == 0,
              "another build keeps nothing");
        size_t files = 0;
        for (const auto& sub : {"blocks", "conv"})
            for (const auto& item : fs::directory_iterator(root / sub)) { (void) item; ++files; }
        check(files == 0, "and leaves no file behind");
        check(d5.written_all_time() > 0, "the all-time counter survives");
    }
    fs::remove_all(root, ec);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
}
