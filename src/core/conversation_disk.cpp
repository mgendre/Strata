// --conversation-cache-disk: the conversation cache's long-term level on disk (conversation_disk.hpp).
//
// Files (little-endian, like the session files):
//   blocks/<32 hex>.kvb   one K/V block: "SCDB", version, layer count, each layer's five part sizes, then the bytes of
//                         every layer's parts for that token range; named by its 128-bit hash (header and bytes)
//   conv/<16 hex>.man     a copy's manifest: identity, level, length, block list, running-state hash, usage
//   conv/<16 hex>.ids     its token ids and pictures
//   conv/<16 hex>.state   its running state (gdn, ple, tails, dead, block_pos)
//   counters.bin          bytes written and read, all time and today (kept across starts)
//   status.json           what the server's monitor shows
// Every file is written under a hidden temporary name and renamed into place.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <system_error>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace fs = std::filesystem;

namespace {
constexpr uint32_t kVersion = 2;   // 2: one level (1 had swap and big)
constexpr uint64_t kSeedA = 0x5c0d15c0a11ull, kSeedB = 0xd15cb10cc0ffeeull, kSeedFile = 0xc0de5ca1eull;
constexpr size_t kPiece = 16u << 20;

struct File {
    std::FILE* f = nullptr;
    ~File() { if (f) std::fclose(f); }
};

// a written file's pages leave the page cache (RAM is what this tier saves), and a read file's too
void drop_pages(std::FILE* f, bool written) {
#ifndef _WIN32
    if (!f) return;
    std::fflush(f);
    const int fd = fileno(f);
    if (written) (void) fdatasync(fd);
    (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#else
    (void) f; (void) written;
#endif
}

bool close_written(File& file) {
    drop_pages(file.f, true);
    const bool ok = std::ferror(file.f) == 0 && std::fclose(file.f) == 0;
    file.f = nullptr;
    return ok;
}

std::string hex(uint64_t v) {
    char s[17];
    std::snprintf(s, sizeof s, "%016llx", (unsigned long long) v);
    return s;
}

std::string temp_beside(const std::string& path) {
    const fs::path p(path);
    static uint64_t serial = 0;
    return (p.parent_path() / (".tmp-" + p.filename().string() + "-" + hex(++serial))).string();
}

bool publish_file(const std::string& tmp, const std::string& path, std::string& error) {
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (!ec) return true;
    fs::remove(tmp, ec);
    error = "conversation disk: cannot rename " + tmp + " to " + path;
    return false;
}

// small structured files: built in memory, a trailing hash of everything before it
struct Out {
    std::vector<uint8_t> b;
    template<class T> void put(T v) { const auto* p = reinterpret_cast<const uint8_t*>(&v); b.insert(b.end(), p, p + sizeof v); }
    void bytes(const void* p, size_t n) { const auto* q = static_cast<const uint8_t*>(p); b.insert(b.end(), q, q + n); }
};
struct In {
    const std::vector<uint8_t>& b;
    size_t at = 0;
    bool ok = true;
    template<class T> T get() {
        T v{};
        if (!ok || b.size() - at < sizeof v) { ok = false; return v; }
        std::memcpy(&v, b.data() + at, sizeof v);
        at += sizeof v;
        return v;
    }
    bool bytes(void* p, size_t n) {
        if (!ok || b.size() - at < n) { ok = false; return false; }
        std::memcpy(p, b.data() + at, n);
        at += n;
        return true;
    }
};

bool write_small(const std::string& path, Out& out, uint64_t& written, std::string& error) {
    out.put<uint64_t>(session_hash64(out.b.data(), out.b.size(), kSeedFile));
    const std::string tmp = temp_beside(path);
    File file;
    file.f = std::fopen(tmp.c_str(), "wb");
    if (!file.f) { error = "conversation disk: cannot create " + tmp; return false; }
    const bool ok = std::fwrite(out.b.data(), 1, out.b.size(), file.f) == out.b.size();
    if (!close_written(file) || !ok) {
        std::error_code ec;
        fs::remove(tmp, ec);
        error = "conversation disk: cannot write " + path;
        return false;
    }
    written = out.b.size();
    return publish_file(tmp, path, error);
}

bool read_small(const std::string& path, std::vector<uint8_t>& b, std::string& error) {
    File file;
    file.f = std::fopen(path.c_str(), "rb");
    if (!file.f) { error = "conversation disk: cannot open " + path; return false; }
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec || size < sizeof(uint64_t) || size > (1ull << 32)) { error = "conversation disk: bad size " + path; return false; }
    b.resize((size_t) size);
    if (std::fread(b.data(), 1, b.size(), file.f) != b.size()) { error = "conversation disk: cannot read " + path; return false; }
    drop_pages(file.f, false);
    uint64_t want = 0;
    std::memcpy(&want, b.data() + b.size() - sizeof want, sizeof want);
    b.resize(b.size() - sizeof want);
    if (session_hash64(b.data(), b.size(), kSeedFile) != want) { error = "conversation disk: damaged " + path; return false; }
    return true;
}

int64_t local_day(int64_t t) {
    std::time_t tt = (std::time_t) t;
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    return (int64_t) (tm.tm_year + 1900) * 10000 + (tm.tm_mon + 1) * 100 + tm.tm_mday;
}

// `a` (ids, imgs) is a prefix of `b`: same tokens up to a's length, the same pictures below it
bool prefix_of(const std::vector<int32_t>& a, const std::vector<ConversationImageKey>& a_imgs,
               const std::vector<int32_t>& b, const std::vector<ConversationImageKey>& b_imgs) {
    if (a.size() > b.size() || !std::equal(a.begin(), a.end(), b.begin())) return false;
    size_t j = 0;
    for (const auto& image : b_imgs) {
        if (image.start >= (int64_t) a.size()) continue;
        if (j == a_imgs.size() || !(a_imgs[j++] == image)) return false;
    }
    return j == a_imgs.size();
}

} // namespace

std::vector<SessionKvSource> conversation_image_sources(const SavedConversation& image) {
    std::vector<SessionKvSource> out;
    out.reserve(image.kv.size());
    for (const ConversationKv& k : image.kv) {
        SessionKvSource s;
        s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
        s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
        const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (size_t p = 0; p < 5; ++p) s.sizes[p] = parts[p]->size();
        s.read = [parts](size_t part, size_t offset, void* dst, size_t n) {
            return part < 5 && parts[part]->read(dst, offset, n);
        };
        s.error = [] { return std::string("parked K/V read out of range"); };
        out.push_back(std::move(s));
    }
    return out;
}

const ConversationCheckpoint& conversation_disk_state(const ConversationCheckpoint& live,
                                                      const std::vector<ConversationCheckpoint>& checkpoints,
                                                      int64_t near) {
    const ConversationCheckpoint* best = nullptr;
    for (const auto& c : checkpoints) {
        if (c.gdn.empty() || c.ids.empty() || c.ids.size() > live.ids.size() ||
            (int64_t) (live.ids.size() - c.ids.size()) > near || !conversation_covers(live, c.ids, c.imgs)) continue;
        if (!best || c.ids.size() > best->ids.size()) best = &c;
    }
    return best ? *best : live;
}

double conversation_score_at(double score, int64_t at, int64_t now, double half_life_s) {
    if (half_life_s <= 0 || now <= at) return score;
    return score * std::exp2(-(double) (now - at) / half_life_s);
}

uint64_t ConversationDisk::Entry::bytes() const {
    uint64_t n = state_bytes;
    for (uint64_t b : block_bytes) n += b;
    return n;
}

int64_t ConversationDisk::now() const { return (int64_t) std::time(nullptr); }

std::string ConversationDisk::path(const char* sub, const std::string& name) const {
    return (fs::path(opt_.dir) / sub / name).string();
}
std::string ConversationDisk::block_name(const Hash& h) const { return hex(h.a) + hex(h.b) + ".kvb"; }
std::string ConversationDisk::entry_name(uint64_t id, const char* ext) const { return hex(id) + ext; }

ConversationDisk::Usage ConversationDisk::usage() const {
    Usage u;
    u.budget = opt_.budget_bytes;
    for (const Entry& e : entries_) { ++u.entries; u.bytes += e.bytes(); }
    return u;
}

int64_t ConversationDisk::saved_tokens(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                                       bool cvec) const {
    int64_t n = 0;
    for (const Entry& e : entries_)
        if (e.cvec == cvec && (int64_t) e.meta.ids.size() > n && prefix_of(e.meta.ids, e.meta.imgs, ids, images))
            n = (int64_t) e.meta.ids.size();
    return n;
}

uint64_t ConversationDisk::physical_bytes() const {
    uint64_t n = 0;
    for (const auto& [h, b] : blocks_) n += b.bytes;
    for (const Entry& e : entries_) n += e.state_bytes;
    return n;
}

bool ConversationDisk::slices(int64_t t0, int64_t t1, std::vector<std::array<uint64_t, 5>>& from,
                              std::vector<std::array<uint64_t, 5>>& to, std::string& error) const {
    from.assign(layers_, {});
    to.assign(layers_, {});
    for (size_t l = 0; l < layers_; ++l) {
        if ((t0 > 0 && !sizes_(l, t0, from[l])) || !sizes_(l, t1, to[l])) {
            error = "conversation disk: no K/V layout for layer " + std::to_string(l);
            return false;
        }
        for (size_t p = 0; p < 5; ++p)
            if (to[l][p] < from[l][p]) { error = "conversation disk: K/V layout shrinks"; return false; }
    }
    return true;
}

bool ConversationDisk::write_block(const Write& w, int64_t t0, int64_t t1, Hash& hash, uint64_t& bytes, bool& fresh,
                                   std::string& error) {
    std::vector<std::array<uint64_t, 5>> from, to;
    if (!slices(t0, t1, from, to, error)) return false;
    Out header;
    header.bytes("SCDB", 4);
    header.put<uint32_t>(kVersion);
    header.put<uint64_t>(layers_);
    uint64_t payload = 0;
    for (size_t l = 0; l < layers_; ++l)
        for (size_t p = 0; p < 5; ++p) {
            if (to[l][p] > w.kv[l].sizes[p]) { error = "conversation disk: the K/V source is shorter than the copy"; return false; }
            header.put<uint64_t>(to[l][p] - from[l][p]);
            payload += to[l][p] - from[l][p];
        }
    // every slice in order, one piece at a time, into the hashers and (pass 2) the file
    auto stream = [&](SessionHasher& ha, SessionHasher& hb, std::FILE* out) {
        ha.update(header.b.data(), header.b.size());
        hb.update(header.b.data(), header.b.size());
        if (out && std::fwrite(header.b.data(), 1, header.b.size(), out) != header.b.size()) return false;
        for (size_t l = 0; l < layers_; ++l)
            for (size_t p = 0; p < 5; ++p)
                for (uint64_t at = from[l][p]; at < to[l][p];) {
                    const size_t n = (size_t) std::min<uint64_t>(kPiece, to[l][p] - at);
                    if (!w.kv[l].read(p, (size_t) at, buffer_.data(), n)) {
                        error = "conversation disk: K/V read failed" + (w.kv[l].error ? ": " + w.kv[l].error() : std::string());
                        return false;
                    }
                    ha.update(buffer_.data(), n);
                    hb.update(buffer_.data(), n);
                    if (out && std::fwrite(buffer_.data(), 1, n, out) != n) return false;
                    at += n;
                }
        return true;
    };
    SessionHasher ha(kSeedA), hb(kSeedB);
    if (!stream(ha, hb, nullptr)) return false;
    hash = {ha.digest(), hb.digest()};
    bytes = header.b.size() + payload;
    fresh = blocks_.find(hash) == blocks_.end();
    if (!fresh) return true;
    const std::string final_path = path("blocks", block_name(hash));
    const std::string tmp = temp_beside(final_path);
    File file;
    file.f = std::fopen(tmp.c_str(), "wb");
    if (!file.f) { error = "conversation disk: cannot create " + tmp; return false; }
    SessionHasher ha2(kSeedA), hb2(kSeedB);
    const bool streamed = stream(ha2, hb2, file.f);
    const bool closed = close_written(file);
    std::error_code ec;
    if (!streamed || !closed) {
        fs::remove(tmp, ec);
        if (error.empty()) error = "conversation disk: cannot write " + final_path;
        return false;
    }
    if (Hash{ha2.digest(), hb2.digest()} != hash) {
        fs::remove(tmp, ec);
        error = "conversation disk: the K/V changed while it was written";
        return false;
    }
    if (!publish_file(tmp, final_path, error)) return false;
    blocks_[hash] = Block{bytes, 0};
    return true;
}

bool ConversationDisk::write_state(const Entry& e, const ConversationCheckpoint& s, Hash& hash, uint64_t& bytes,
                                   std::string& error) {
    const std::string final_path = path("conv", entry_name(e.id, ".state"));
    const std::string tmp = temp_beside(final_path);
    Out header;
    header.bytes("SCDS", 4);
    header.put<uint32_t>(kVersion);
    const std::array<const std::vector<uint8_t>*, 5> parts = {&s.gdn, &s.ple, &s.tails, &s.dead, &s.block_pos};
    for (const auto* v : parts) header.put<uint64_t>(v->size());
    File file;
    file.f = std::fopen(tmp.c_str(), "wb");
    if (!file.f) { error = "conversation disk: cannot create " + tmp; return false; }
    SessionHasher ha(kSeedA), hb(kSeedB);
    bool ok = std::fwrite(header.b.data(), 1, header.b.size(), file.f) == header.b.size();
    ha.update(header.b.data(), header.b.size());
    hb.update(header.b.data(), header.b.size());
    bytes = header.b.size();
    for (const auto* v : parts) {
        ok = ok && std::fwrite(v->data(), 1, v->size(), file.f) == v->size();
        ha.update(v->data(), v->size());
        hb.update(v->data(), v->size());
        bytes += v->size();
    }
    if (!close_written(file) || !ok) {
        std::error_code ec;
        fs::remove(tmp, ec);
        error = "conversation disk: cannot write " + final_path;
        return false;
    }
    hash = {ha.digest(), hb.digest()};
    return publish_file(tmp, final_path, error);
}

bool ConversationDisk::read_state(const Entry& e, ConversationCheckpoint& s, uint64_t& bytes, std::string& error) {
    const std::string file_path = path("conv", entry_name(e.id, ".state"));
    File file;
    file.f = std::fopen(file_path.c_str(), "rb");
    if (!file.f) { error = "conversation disk: cannot open " + file_path; return false; }
    char magic[4];
    uint32_t version = 0;
    std::array<uint64_t, 5> n{};
    if (std::fread(magic, 1, 4, file.f) != 4 || std::memcmp(magic, "SCDS", 4) != 0 ||
        std::fread(&version, sizeof version, 1, file.f) != 1 || version != kVersion ||
        std::fread(n.data(), sizeof(uint64_t), 5, file.f) != 5) {
        error = "conversation disk: not a running-state file " + file_path;
        return false;
    }
    uint64_t total = 0;
    for (uint64_t k : n) total += k;
    if (total + 4 + sizeof version + 5 * sizeof(uint64_t) != e.state_bytes) {
        error = "conversation disk: running-state size differs " + file_path;
        return false;
    }
    SessionHasher ha(kSeedA), hb(kSeedB);
    Out header;
    header.bytes("SCDS", 4);
    header.put<uint32_t>(version);
    for (uint64_t k : n) header.put<uint64_t>(k);
    ha.update(header.b.data(), header.b.size());
    hb.update(header.b.data(), header.b.size());
    const std::array<std::vector<uint8_t>*, 5> parts = {&s.gdn, &s.ple, &s.tails, &s.dead, &s.block_pos};
    for (size_t i = 0; i < 5; ++i) {
        parts[i]->resize((size_t) n[i]);
        if (std::fread(parts[i]->data(), 1, parts[i]->size(), file.f) != parts[i]->size()) {
            error = "conversation disk: cannot read " + file_path;
            return false;
        }
        ha.update(parts[i]->data(), parts[i]->size());
        hb.update(parts[i]->data(), parts[i]->size());
    }
    drop_pages(file.f, false);
    if (Hash{ha.digest(), hb.digest()} != e.state_hash) { error = "conversation disk: damaged " + file_path; return false; }
    bytes = e.state_bytes;
    return true;
}

bool ConversationDisk::write_ids(const Entry& e, std::string& error) {
    Out out;
    out.bytes("SCDI", 4);
    out.put<uint32_t>(kVersion);
    out.put<uint64_t>(e.meta.ids.size());
    out.bytes(e.meta.ids.data(), e.meta.ids.size() * sizeof(int32_t));
    out.put<uint64_t>(e.meta.imgs.size());
    for (const auto& im : e.meta.imgs) { out.put<int64_t>(im.start); out.put<uint64_t>(im.hash); }
    uint64_t n = 0;
    return write_small(path("conv", entry_name(e.id, ".ids")), out, n, error);
}

bool ConversationDisk::read_ids(Entry& e, std::string& error) {
    std::vector<uint8_t> b;
    if (!read_small(path("conv", entry_name(e.id, ".ids")), b, error)) return false;
    In in{b};
    char magic[4];
    in.bytes(magic, 4);
    const uint32_t version = in.get<uint32_t>();
    const uint64_t n = in.get<uint64_t>();
    if (!in.ok || std::memcmp(magic, "SCDI", 4) != 0 || version != kVersion || n > b.size() / sizeof(int32_t)) {
        error = "conversation disk: not an ids file";
        return false;
    }
    e.meta.ids.resize((size_t) n);
    in.bytes(e.meta.ids.data(), (size_t) n * sizeof(int32_t));
    const uint64_t m = in.get<uint64_t>();
    if (!in.ok || m > b.size()) { error = "conversation disk: bad ids file"; return false; }
    e.meta.imgs.resize((size_t) m);
    for (auto& im : e.meta.imgs) { im.start = in.get<int64_t>(); im.hash = in.get<uint64_t>(); }
    if (!in.ok || in.at != b.size()) { error = "conversation disk: bad ids file"; return false; }
    return true;
}

bool ConversationDisk::write_manifest(const Entry& e, std::string& error) {
    Out out;
    out.bytes("SCDM", 4);
    out.put<uint32_t>(kVersion);
    out.put<uint64_t>(identity_.model); out.put<uint64_t>(identity_.config);
    out.put<uint64_t>(identity_.build); out.put<uint64_t>(identity_.layout);
    out.put<uint64_t>(e.id);
    out.put<uint8_t>(e.cvec ? 1 : 0);
    out.put<uint64_t>(e.meta.ids.size());
    out.put<uint64_t>(e.state_hash.a); out.put<uint64_t>(e.state_hash.b); out.put<uint64_t>(e.state_bytes);
    out.put<uint64_t>(e.blocks.size());
    for (size_t k = 0; k < e.blocks.size(); ++k) {
        out.put<uint64_t>(e.blocks[k].a); out.put<uint64_t>(e.blocks[k].b); out.put<uint64_t>(e.block_bytes[k]);
    }
    out.put<double>(e.score); out.put<int64_t>(e.score_at);
    out.put<int64_t>(e.created); out.put<int64_t>(e.last_used);
    out.put<uint64_t>(e.returns); out.put<uint64_t>(e.served_tokens); out.put<uint64_t>(e.written);
    uint64_t n = 0;
    return write_small(path("conv", entry_name(e.id, ".man")), out, n, error);
}

bool ConversationDisk::read_manifest(const std::string& file, Entry& e, ConversationDiskIdentity& id, std::string& error) {
    std::vector<uint8_t> b;
    if (!read_small(file, b, error)) return false;
    In in{b};
    char magic[4];
    in.bytes(magic, 4);
    const uint32_t version = in.get<uint32_t>();
    if (!in.ok || std::memcmp(magic, "SCDM", 4) != 0 || version != kVersion) { error = "conversation disk: not a manifest"; return false; }
    id.model = in.get<uint64_t>(); id.config = in.get<uint64_t>(); id.build = in.get<uint64_t>(); id.layout = in.get<uint64_t>();
    e.id = in.get<uint64_t>();
    e.cvec = in.get<uint8_t>() != 0;
    const uint64_t tokens = in.get<uint64_t>();
    e.state_hash.a = in.get<uint64_t>(); e.state_hash.b = in.get<uint64_t>(); e.state_bytes = in.get<uint64_t>();
    const uint64_t nb = in.get<uint64_t>();
    if (!in.ok || nb > b.size() / 24) { error = "conversation disk: bad manifest"; return false; }
    e.blocks.resize((size_t) nb);
    e.block_bytes.resize((size_t) nb);
    for (size_t k = 0; k < nb; ++k) {
        e.blocks[k].a = in.get<uint64_t>(); e.blocks[k].b = in.get<uint64_t>(); e.block_bytes[k] = in.get<uint64_t>();
    }
    e.score = in.get<double>(); e.score_at = in.get<int64_t>();
    e.created = in.get<int64_t>(); e.last_used = in.get<int64_t>();
    e.returns = in.get<uint64_t>(); e.served_tokens = in.get<uint64_t>(); e.written = in.get<uint64_t>();
    if (!in.ok || in.at != b.size()) { error = "conversation disk: bad manifest"; return false; }
    e.meta.ids.assign((size_t) std::min<uint64_t>(tokens, 1ull << 31), 0);   // sized here, read from the ids file
    return true;
}

void ConversationDisk::ref(const Entry& e) {
    for (size_t k = 0; k < e.blocks.size(); ++k) {
        Block& b = blocks_[e.blocks[k]];
        b.bytes = e.block_bytes[k];
        ++b.refs;
    }
}

void ConversationDisk::unref(const Entry& e) {
    for (const Hash& h : e.blocks) {
        auto it = blocks_.find(h);
        if (it == blocks_.end()) continue;
        if (it->second.refs > 0) --it->second.refs;
        if (it->second.refs == 0) {
            std::error_code ec;
            fs::remove(path("blocks", block_name(h)), ec);
            blocks_.erase(it);
        }
    }
}

void ConversationDisk::remove(size_t index) {
    const Entry e = std::move(entries_[index]);
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
    std::error_code ec;
    for (const char* ext : {".man", ".ids", ".state"}) fs::remove(path("conv", entry_name(e.id, ext)), ec);
    unref(e);
    ++start_.deleted;
}

void ConversationDisk::fit(uint64_t keep_id) {
    // over the budget: the lowest usage score goes (a copy never seen again has none), the least recently used among
    // equals; the copy just stored or used last of all
    const int64_t t = now();
    while (usage().bytes > opt_.budget_bytes && !entries_.empty()) {
        size_t v = 0;
        auto worse = [&](const Entry& a, const Entry& b) {   // a leaves before b
            if ((a.id == keep_id) != (b.id == keep_id)) return b.id == keep_id;
            const double sa = conversation_score_at(a.score, a.score_at, t, opt_.half_life_s);
            const double sb = conversation_score_at(b.score, b.score_at, t, opt_.half_life_s);
            return sa != sb ? sa < sb : a.last_used < b.last_used;
        };
        for (size_t i = 1; i < entries_.size(); ++i)
            if (worse(entries_[i], entries_[v])) v = i;
        remove(v);
    }
}

void ConversationDisk::count_written(uint64_t bytes, ConversationDiskStats& stats) {
    const int64_t today = local_day(now());
    if (today != day_) { day_ = today; day_written_ = 0; }
    const uint64_t before = day_written_;
    day_written_ += bytes;
    total_written_ += bytes;
    start_.written += bytes;
    if (opt_.day_guide_bytes > 0 && before < opt_.day_guide_bytes && day_written_ >= opt_.day_guide_bytes)
        stats.guide_crossed = true;
}

void ConversationDisk::save_counters() const {
    Out out;
    out.bytes("SCDC", 4);
    out.put<uint32_t>(kVersion);
    out.put<uint64_t>(total_written_); out.put<uint64_t>(total_read_);
    out.put<int64_t>(day_); out.put<uint64_t>(day_written_);
    uint64_t n = 0;
    std::string error;
    (void) write_small((fs::path(opt_.dir) / "counters.bin").string(), out, n, error);
}

void ConversationDisk::load_counters() {
    std::vector<uint8_t> b;
    std::string error;
    if (!read_small((fs::path(opt_.dir) / "counters.bin").string(), b, error)) return;
    In in{b};
    char magic[4];
    in.bytes(magic, 4);
    const uint32_t version = in.get<uint32_t>();
    const uint64_t w = in.get<uint64_t>(), r = in.get<uint64_t>();
    const int64_t day = in.get<int64_t>();
    const uint64_t dw = in.get<uint64_t>();
    if (!in.ok || std::memcmp(magic, "SCDC", 4) != 0 || version != kVersion) return;
    total_written_ = w; total_read_ = r;
    day_ = day; day_written_ = day == local_day(now()) ? dw : 0;
}

bool ConversationDisk::open(const ConversationDiskOptions& options, const ConversationDiskIdentity& identity,
                            size_t layers, ConversationDiskSizes sizes, std::string& note, std::string& error) {
    enabled_ = false;
    entries_.clear();
    blocks_.clear();
    opt_ = options;
    identity_ = identity;
    layers_ = layers;
    sizes_ = std::move(sizes);
    if (opt_.dir.empty() || layers_ == 0 || !sizes_ || opt_.block_tokens <= 0) {
        error = "conversation disk: no folder or no K/V layout";
        return false;
    }
    std::error_code ec;
    for (const char* sub : {"", "blocks", "conv"}) {
        fs::create_directories(fs::path(opt_.dir) / sub, ec);
        if (ec) { error = "conversation disk: cannot create " + (fs::path(opt_.dir) / sub).string(); return false; }
    }
    buffer_.resize(kPiece);
    load_counters();
    size_t dropped = 0;
    std::vector<uint64_t> kept_ids;
    for (const auto& item : fs::directory_iterator(fs::path(opt_.dir) / "conv", ec)) {
        const fs::path p = item.path();
        if (p.extension() != ".man" || p.filename().string().rfind(".tmp-", 0) == 0) continue;
        Entry e;
        ConversationDiskIdentity id;
        std::string why;
        const bool ok = read_manifest(p.string(), e, id, why);
        const size_t want_tokens = e.meta.ids.size();
        bool keep = ok && id == identity_ && read_ids(e, why) && !e.meta.ids.empty() && e.meta.ids.size() == want_tokens;
        if (keep) {
            const auto state_size = fs::file_size(path("conv", entry_name(e.id, ".state")), ec);
            keep = !ec && state_size == e.state_bytes;
            for (size_t k = 0; keep && k < e.blocks.size(); ++k) {
                const auto bs = fs::file_size(path("blocks", block_name(e.blocks[k])), ec);
                keep = !ec && bs == e.block_bytes[k];
            }
        }
        if (!keep) { ++dropped; continue; }
        next_id_ = std::max(next_id_, e.id + 1);
        kept_ids.push_back(e.id);
        entries_.push_back(std::move(e));
    }
    for (const Entry& e : entries_) ref(e);
    // every file no kept copy names goes: other builds' and models', damaged ones, cut-short temporaries
    auto kept_name = [&](const std::string& stem) {
        for (uint64_t id : kept_ids) if (stem == hex(id)) return true;
        return false;
    };
    for (const auto& item : fs::directory_iterator(fs::path(opt_.dir) / "conv", ec))
        if (!kept_name(item.path().stem().string()) || item.path().filename().string().rfind(".tmp-", 0) == 0)
            fs::remove(item.path(), ec);
    size_t orphan_blocks = 0;
    for (const auto& item : fs::directory_iterator(fs::path(opt_.dir) / "blocks", ec)) {
        const std::string name = item.path().filename().string();
        bool used = false;
        if (name.size() == 36 && item.path().extension() == ".kvb") {
            Hash h;
            h.a = std::strtoull(name.substr(0, 16).c_str(), nullptr, 16);
            h.b = std::strtoull(name.substr(16, 16).c_str(), nullptr, 16);
            used = blocks_.count(h) != 0;
        }
        if (!used) { fs::remove(item.path(), ec); ++orphan_blocks; }
    }
    for (const auto& item : fs::directory_iterator(fs::path(opt_.dir), ec))
        if (item.path().filename().string().rfind(".tmp-", 0) == 0) fs::remove(item.path(), ec);
    enabled_ = true;
    fit(0);
    const Usage u = usage();
    char line[320];
    std::snprintf(line, sizeof line, "kept %zu conversations (%llu MiB), dropped %zu (another build or model, or damaged), "
                  "%zu stray blocks removed; written all time %llu MiB",
                  u.entries, (unsigned long long) (u.bytes >> 20), dropped, orphan_blocks,
                  (unsigned long long) (total_written_ >> 20));
    note = line;
    publish();
    return true;
}

bool ConversationDisk::save(const Write& w, ConversationDiskStats& stats, std::string& error) {
    stats = {};
    if (!enabled_) { error = "conversation disk: not open"; return false; }
    const int64_t tokens = (int64_t) w.ids.size();
    stats.tokens = tokens;
    if (tokens < opt_.min_tokens || tokens <= 0) { stats.skipped = true; stats.why = "shorter than the minimum"; return true; }
    if (w.kv.size() != layers_) { error = "conversation disk: K/V layer count differs"; return false; }
    if (w.state.gdn.empty()) { error = "conversation disk: no running state"; return false; }
    const int64_t t = now();
    // the copies of this conversation on disk: the same state already, or the longest one it extends (a longer copy
    // it is a prefix of is another branch: its running state is further on)
    size_t holds = entries_.size(), base = entries_.size();
    for (size_t i = 0; i < entries_.size(); ++i) {
        const Entry& e = entries_[i];
        if (e.cvec != w.cvec || !prefix_of(e.meta.ids, e.meta.imgs, w.ids, w.imgs)) continue;
        if (e.meta.ids.size() == w.ids.size()) holds = i;
        else if (base == entries_.size() || e.meta.ids.size() > entries_[base].meta.ids.size()) base = i;
    }
    auto touch = [&](Entry& e, const char* why) {
        e.score = std::max(conversation_score_at(e.score, e.score_at, t, opt_.half_life_s), w.score);
        e.score_at = t;
        e.last_used = t;
        stats.skipped = true;
        stats.why = why;
        stats.score = e.score;
        ++start_.skipped;
        if (!write_manifest(e, error)) return false;
        count_written(0, stats);
        fit(e.id);
        publish();
        save_counters();
        return true;
    };
    if (holds != entries_.size()) return touch(entries_[holds], "already on disk");
    if (base != entries_.size() && tokens - (int64_t) entries_[base].meta.ids.size() < opt_.refresh_tokens)
        return touch(entries_[base], "the copy on disk is close enough");

    Entry ne;
    const Entry* b = base != entries_.size() ? &entries_[base] : nullptr;
    ne.id = b ? b->id : next_id_++;
    ne.cvec = w.cvec;
    ne.meta.ids = w.ids;
    ne.meta.imgs = w.imgs;
    ne.created = b ? b->created : t;
    ne.returns = b ? b->returns : 0;
    ne.served_tokens = b ? b->served_tokens : 0;
    ne.score = std::max(b ? conversation_score_at(b->score, b->score_at, t, opt_.half_life_s) : 0.0, w.score);
    ne.score_at = t;
    ne.last_used = t;
    // the K/V blocks: the ones on disk already are only named
    std::vector<Hash> fresh_blocks;
    auto abandon = [&] {
        for (const Hash& h : fresh_blocks) {
            auto it = blocks_.find(h);
            if (it != blocks_.end() && it->second.refs == 0) {
                std::error_code ec;
                fs::remove(path("blocks", block_name(h)), ec);
                blocks_.erase(it);
            }
        }
    };
    uint64_t written = 0;
    for (int64_t t0 = 0; t0 < tokens; t0 += opt_.block_tokens) {
        const int64_t t1 = std::min<int64_t>(t0 + opt_.block_tokens, tokens);
        Hash h;
        uint64_t bytes = 0;
        bool fresh = false;
        if (!write_block(w, t0, t1, h, bytes, fresh, error)) { abandon(); return false; }
        ne.blocks.push_back(h);
        ne.block_bytes.push_back(bytes);
        ++stats.blocks;
        if (fresh) { fresh_blocks.push_back(h); written += bytes; ++stats.new_blocks; }
        else stats.reused += bytes;
    }
    if (!write_state(ne, w.state, ne.state_hash, ne.state_bytes, error) || !write_ids(ne, error)) { abandon(); return false; }
    written += ne.state_bytes + ne.meta.ids.size() * sizeof(int32_t);
    ne.written = (b ? b->written : 0) + written;
    if (!write_manifest(ne, error)) { abandon(); return false; }
    ref(ne);
    if (b) {
        Entry old = std::move(entries_[base]);
        entries_[base] = std::move(ne);
        unref(old);
    } else {
        entries_.push_back(std::move(ne));
    }
    const uint64_t id = b ? entries_[base].id : entries_.back().id;
    // older copies of this conversation that this one holds (a turn back, another branch's shorter start) go
    std::vector<uint64_t> superseded;
    for (const Entry& cur : entries_)
        if (cur.id == id)
            for (const Entry& e : entries_)
                if (e.id != id && e.cvec == cur.cvec && prefix_of(e.meta.ids, e.meta.imgs, cur.meta.ids, cur.meta.imgs))
                    superseded.push_back(e.id);
    for (uint64_t gone : superseded)
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].id == gone) { remove(i); break; }
    stats.written = written;
    count_written(written, stats);
    ++start_.saves;
    start_.reused += stats.reused;
    fit(id);
    for (const Entry& e : entries_)
        if (e.id == id) stats.score = e.score;
    publish();
    save_counters();
    return true;
}

bool ConversationDisk::load(const ConversationDiskMatch& m, ConversationCheckpoint& state, bool& cvec,
                            const std::vector<SessionKvSink>& kv, ConversationDiskStats& stats, std::string& error) {
    stats = {};
    size_t index = entries_.size();
    for (size_t i = 0; i < entries_.size(); ++i) if (entries_[i].id == m.id) index = i;
    if (!enabled_ || index == entries_.size()) { error = "conversation disk: the copy is gone"; return false; }
    if (kv.size() != layers_) { error = "conversation disk: K/V layer count differs"; return false; }
    auto drop = [&] { remove(index); publish(); };   // a copy that cannot be read is not offered again
    Entry& e = entries_[index];
    const int64_t tokens = (int64_t) e.meta.ids.size();
    uint64_t read = 0;
    ConversationCheckpoint s;
    if (!read_state(e, s, read, error)) { drop(); return false; }
    for (int64_t t0 = 0, k = 0; t0 < tokens; t0 += opt_.block_tokens, ++k) {
        const int64_t t1 = std::min<int64_t>(t0 + opt_.block_tokens, tokens);
        std::vector<std::array<uint64_t, 5>> from, to;
        if ((size_t) k >= e.blocks.size() || !slices(t0, t1, from, to, error)) {
            if (error.empty()) error = "conversation disk: missing block";
            drop();
            return false;
        }
        const std::string file_path = path("blocks", block_name(e.blocks[(size_t) k]));
        File file;
        file.f = std::fopen(file_path.c_str(), "rb");
        if (!file.f) { error = "conversation disk: cannot open " + file_path; drop(); return false; }
#ifndef _WIN32
        (void) posix_fadvise(fileno(file.f), 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
        Out header;
        header.bytes("SCDB", 4);
        header.put<uint32_t>(kVersion);
        header.put<uint64_t>(layers_);
        for (size_t l = 0; l < layers_; ++l)
            for (size_t p = 0; p < 5; ++p) {
                if (to[l][p] > kv[l].sizes[p]) { error = "conversation disk: the K/V target is shorter than the copy"; drop(); return false; }
                header.put<uint64_t>(to[l][p] - from[l][p]);
            }
        std::vector<uint8_t> on_disk(header.b.size());
        if (std::fread(on_disk.data(), 1, on_disk.size(), file.f) != on_disk.size() || on_disk != header.b) {
            error = "conversation disk: block layout differs " + file_path;
            drop();
            return false;
        }
        SessionHasher ha(kSeedA), hb(kSeedB);
        ha.update(header.b.data(), header.b.size());
        hb.update(header.b.data(), header.b.size());
        read += header.b.size();
        for (size_t l = 0; l < layers_; ++l)
            for (size_t p = 0; p < 5; ++p)
                for (uint64_t at = from[l][p]; at < to[l][p];) {
                    const size_t n = (size_t) std::min<uint64_t>(kPiece, to[l][p] - at);
                    if (std::fread(buffer_.data(), 1, n, file.f) != n) { error = "conversation disk: cannot read " + file_path; drop(); return false; }
                    ha.update(buffer_.data(), n);
                    hb.update(buffer_.data(), n);
                    if (!kv[l].write(p, (size_t) at, buffer_.data(), n)) {
                        error = "conversation disk: K/V write failed" + (kv[l].error ? ": " + kv[l].error() : std::string());
                        return false;
                    }
                    read += n;
                    at += n;
                }
        drop_pages(file.f, false);
        if (Hash{ha.digest(), hb.digest()} != e.blocks[(size_t) k]) { error = "conversation disk: damaged " + file_path; drop(); return false; }
    }
    for (const SessionKvSink& sink : kv)
        if (sink.finish && !sink.finish()) {
            error = "conversation disk: K/V refresh failed" + (sink.error ? ": " + sink.error() : std::string());
            return false;
        }
    s.ids = e.meta.ids;
    s.imgs = e.meta.imgs;
    state = std::move(s);
    cvec = e.cvec;
    const int64_t t = now();
    ++e.returns;
    e.served_tokens += (uint64_t) m.tokens;
    e.score = conversation_score_at(e.score, e.score_at, t, opt_.half_life_s) + 1.0;
    e.score_at = t;
    e.last_used = t;
    ++start_.restores;
    start_.tokens += (uint64_t) m.tokens;
    stats.read = read;
    stats.tokens = tokens;
    stats.score = e.score;
    start_.read += read;
    total_read_ += read;
    const uint64_t id = e.id;
    std::string merr;
    if (write_manifest(e, merr)) count_written(0, stats);
    fit(id);
    publish();
    save_counters();
    return true;
}

size_t ConversationDisk::clear() {
    const size_t n = entries_.size();
    while (!entries_.empty()) remove(entries_.size() - 1);
    if (enabled_) publish();
    return n;
}

std::string ConversationDisk::status_json() const {
    std::string s;
    char b[512];
    const Usage u = usage();
    std::snprintf(b, sizeof b,
                  "{\"version\":2,\"time\":%lld,\"block_tokens\":%lld,\"min_tokens\":%lld,\"day_guide_bytes\":%llu,"
                  "\"disk\":{\"entries\":%zu,\"bytes\":%llu,\"budget\":%llu},\"physical_bytes\":%llu,",
                  (long long) now(), (long long) opt_.block_tokens, (long long) opt_.min_tokens,
                  (unsigned long long) opt_.day_guide_bytes, u.entries, (unsigned long long) u.bytes,
                  (unsigned long long) u.budget, (unsigned long long) physical_bytes());
    s += b;
    std::snprintf(b, sizeof b,
                  "\"day\":%lld,\"written_today\":%llu,\"written_all_time\":%llu,\"read_all_time\":%llu,"
                  "\"since_start\":{\"written\":%llu,\"read\":%llu,\"reused\":%llu,\"saves\":%llu,\"skipped\":%llu,"
                  "\"restores\":%llu,\"tokens\":%llu,\"deleted\":%llu},\"entries\":[",
                  (long long) day_, (unsigned long long) day_written_, (unsigned long long) total_written_,
                  (unsigned long long) total_read_, (unsigned long long) start_.written, (unsigned long long) start_.read,
                  (unsigned long long) start_.reused, (unsigned long long) start_.saves,
                  (unsigned long long) start_.skipped, (unsigned long long) start_.restores,
                  (unsigned long long) start_.tokens, (unsigned long long) start_.deleted);
    s += b;
    std::vector<const Entry*> rows;
    for (const Entry& e : entries_) rows.push_back(&e);
    const int64_t t = now();
    std::sort(rows.begin(), rows.end(), [&](const Entry* x, const Entry* y) {
        const double sx = conversation_score_at(x->score, x->score_at, t, opt_.half_life_s);
        const double sy = conversation_score_at(y->score, y->score_at, t, opt_.half_life_s);
        return sx != sy ? sx > sy : x->last_used > y->last_used;
    });
    for (size_t i = 0; i < rows.size(); ++i) {
        const Entry& e = *rows[i];
        std::snprintf(b, sizeof b,
                      "%s{\"id\":\"%s\",\"tokens\":%zu,\"bytes\":%llu,\"returns\":%llu,\"score\":%.3f,"
                      "\"served_tokens\":%llu,\"written\":%llu,\"created\":%lld,\"last_used\":%lld}",
                      i ? "," : "", hex(e.id).c_str(), e.meta.ids.size(), (unsigned long long) e.bytes(),
                      (unsigned long long) e.returns, conversation_score_at(e.score, e.score_at, t, opt_.half_life_s),
                      (unsigned long long) e.served_tokens, (unsigned long long) e.written, (long long) e.created,
                      (long long) e.last_used);
        s += b;
    }
    s += "]}";
    return s;
}

void ConversationDisk::publish() const {
    const std::string p = (fs::path(opt_.dir) / "status.json").string();
    const std::string tmp = temp_beside(p);
    const std::string body = status_json();
    File file;
    file.f = std::fopen(tmp.c_str(), "wb");
    if (!file.f) return;
    const bool ok = std::fwrite(body.data(), 1, body.size(), file.f) == body.size();
    const bool closed = std::fclose(file.f) == 0;
    file.f = nullptr;
    std::string error;
    if (ok && closed) (void) publish_file(tmp, p, error);
    else { std::error_code ec; fs::remove(tmp, ec); }
}

} // namespace strata::core
