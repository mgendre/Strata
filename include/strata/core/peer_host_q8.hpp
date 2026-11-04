// --peer-slow-link-optimizations: the peer's input rows quantized to q8_1 on the host, the same bytes the device's
// quantize_q8_1_rows makes (checked by peer_host_q8_parity).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::core {

constexpr size_t kQ8_1Block = 36;   ///< a q8_1 block: fp16 d, fp16 sum, 32 int8 codes

/// Whether this build and CPU quantize on the host: x86 with AVX2 + F16C, a GCC or Clang build.  Elsewhere false, and
/// quantize_q8_1_host does nothing (the device quantizes).
bool host_q8_available();
/// `n_tok` rows of `n_cols` floats (n_cols a multiple of 32) as q8_1 blocks into `y` (n_tok * n_cols / 32 blocks of
/// kQ8_1Block bytes), only where host_q8_available().
void quantize_q8_1_host(const float* x, int64_t n_tok, int64_t n_cols, uint8_t* y);

}  // namespace strata::core
