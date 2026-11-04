// peer_host_q8_parity: --peer-slow-link-optimizations quantizes the peer's input rows on the host
// (strata::core::quantize_q8_1_host).  Its q8_1 blocks must be the bytes the device's quantize_q8_1_rows makes, or the
// peer's experts would compute on other inputs.  Random rows and the edge cases of the rounding: zeros, an all-zero
// block, ties at .5, values far above fp16's range (d and the sum kept finite), denormals, signed zeros.  No model.
// Exit 77 (skipped) without a CUDA device or where the host path is not built (not x86, MSVC) or the CPU lacks it.
#include "strata/core/peer_host_q8.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {
int check(const std::vector<float>& x, int64_t rows, int64_t cols, const char* what) {
    const size_t blocks = (size_t) (rows * cols / 32), bytes = blocks * strata::core::kQ8_1Block;
    std::vector<uint8_t> host(bytes, 0xAA), dev(bytes, 0x55);
    strata::core::quantize_q8_1_host(x.data(), rows, cols, host.data());
    float* dx = nullptr;
    void* dy = nullptr;
    if (cudaMalloc(&dx, x.size() * sizeof(float)) != cudaSuccess || cudaMalloc(&dy, bytes) != cudaSuccess) {
        std::printf("FAIL %s: cudaMalloc\n", what);
        return 1;
    }
    cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    strata::kernels::quantize_q8_1_rows(dx, rows, cols, dy, nullptr);
    const bool ok = cudaDeviceSynchronize() == cudaSuccess &&
                    cudaMemcpy(dev.data(), dy, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dx);
    cudaFree(dy);
    if (!ok) {
        std::printf("FAIL %s: CUDA error\n", what);
        return 1;
    }
    size_t diff = 0, first = bytes;
    for (size_t i = 0; i < bytes; ++i)
        if (host[i] != dev[i]) {
            ++diff;
            if (first == bytes) first = i;
        }
    if (diff) {
        std::printf("FAIL %s: %zu of %zu bytes differ (first at block %zu, byte %zu)\n", what, diff, bytes,
                    first / strata::core::kQ8_1Block, first % strata::core::kQ8_1Block);
        return 1;
    }
    std::printf("ok   %s: %zu blocks, the same bytes\n", what, blocks);
    return 0;
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("peer_host_q8_parity: no CUDA device, skipped\n");
        return 77;
    }
    if (!strata::core::host_q8_available()) {
        std::printf("peer_host_q8_parity: no host quantization in this build or on this CPU, skipped\n");
        return 77;
    }
    constexpr int64_t H = 2560;   // Qwen3.8-Flash-Next's rows; any multiple of 32 works
    int fails = 0;
    std::mt19937 rng(1234);
    for (const int64_t rows : {1, 3, 8, 16}) {   // a decode window's rows
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> x((size_t) (rows * H));
        for (float& v : x) v = nd(rng) * 0.05f;
        char what[64];
        std::snprintf(what, sizeof what, "random rows (%lld x %lld)", (long long) rows, (long long) H);
        fails += check(x, rows, H, what);
    }
    {   // wide range, so every exponent and the fp16 rounding of d and the sum are exercised
        std::uniform_real_distribution<float> e(-20.0f, 20.0f), s(-1.0f, 1.0f);
        std::vector<float> x((size_t) (4 * H));
        for (float& v : x) v = s(rng) * std::exp2(e(rng));
        fails += check(x, 4, H, "random magnitudes 2^-20 .. 2^20");
    }
    {   // the edge cases, one block each
        std::vector<float> x((size_t) H, 0.0f);
        float* b = x.data();
        for (int i = 0; i < 32; ++i) b[32 + i] = (i - 16) * 0.5f;                        // ties at .5 after x / d
        for (int i = 0; i < 32; ++i) b[64 + i] = (i % 2 ? -1.0f : 1.0f) * 127.0f * (i + 1) / 32.0f;
        for (int i = 0; i < 32; ++i) b[96 + i] = 1e30f * (i % 3 - 1);                    // d and the sum far above fp16
        for (int i = 0; i < 32; ++i) b[128 + i] = std::numeric_limits<float>::denorm_min() * (i + 1);
        for (int i = 0; i < 32; ++i) b[160 + i] = (i % 2) ? -0.0f : 0.0f;                 // signed zeros
        b[192] = 1.0f;                                                                   // one value, the rest zero
        for (int i = 0; i < 32; ++i) b[224 + i] = 3.0f;                                  // a constant block
        fails += check(x, 1, H, "edge cases (an all-zero block, ties, overflow, denormals, signed zeros)");
    }
    std::printf(fails ? "peer_host_q8_parity: FAIL\n" : "peer_host_q8_parity: PASS\n");
    return fails ? 1 : 0;
}
