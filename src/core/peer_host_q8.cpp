// --peer-slow-link-optimizations: the peer's input rows quantized to q8_1 on the host (peer_host_q8.hpp).
#include "strata/core/peer_host_q8.hpp"

#include <cmath>
#include <cstring>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define STRATA_PEER_HOST_Q8 1
#include <immintrin.h>
#endif

namespace strata::core {

#ifdef STRATA_PEER_HOST_Q8
namespace {
// The input rows as q8_1 blocks, quantized on the host the way quantize_q8_1_kernel's q8_1_store does it on the device
// (iq_kernels.cu, no fast math): the warp's butterfly sum (lane i adds lane i^16, then ^8, ^4, ^2, ^1 - float addition is
// commutative, so every lane ends with this sum), amax, d = amax / 127 kept finite (q8_1_finite), the codes roundf(x / d)
// clamped to +-127 (0 for an all-zero block), ds = {d, sum} as fp16 rounded to nearest even.  The same bytes, so the peer
// gets 36 bytes a block instead of 128 and runs no quantization.  AVX2 + F16C only (else the device's).
__attribute__((target("avx2,f16c"))) void quantize_q8_1_host_avx2(const float* x, int64_t n_tok, int64_t n_cols, uint8_t* y) {
    const __m256 sign = _mm256_set1_ps(-0.0f), half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f);
    const __m256 lim = _mm256_set1_ps(127.0f), nlim = _mm256_set1_ps(-127.0f);
    const int64_t blocks = n_tok * (n_cols / 32);
    for (int64_t b = 0; b < blocks; ++b) {
        const float* p = x + b * 32;
        uint8_t* blk = y + b * kQ8_1Block;
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
}  // namespace

bool host_q8_available() {
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
    return ok;
}

void quantize_q8_1_host(const float* x, int64_t n_tok, int64_t n_cols, uint8_t* y) {
    quantize_q8_1_host_avx2(x, n_tok, n_cols, y);
}
#else
bool host_q8_available() { return false; }
void quantize_q8_1_host(const float*, int64_t, int64_t, uint8_t*) {}
#endif

}  // namespace strata::core
