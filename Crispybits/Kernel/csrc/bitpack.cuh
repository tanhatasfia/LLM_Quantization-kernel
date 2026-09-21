#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t cb_mask(int bits) {
    return (1u << bits) - 1u;
}

// Continuous row-major bitstream extraction. This explicitly handles the
// W3 values that cross 32-bit word boundaries.
__device__ __forceinline__ uint32_t cb_unpack(
    const uint32_t* __restrict__ row,
    int k,
    int bits) {
    const int bitpos = k * bits;
    const int wi = bitpos >> 5;
    const int off = bitpos & 31;
    const uint32_t mask = cb_mask(bits);
    if (off + bits <= 32) {
        return (row[wi] >> off) & mask;
    }
    const int lo_bits = 32 - off;
    const int hi_bits = bits - lo_bits;
    const uint32_t lo = (row[wi] >> off) & ((1u << lo_bits) - 1u);
    const uint32_t hi = row[wi + 1] & ((1u << hi_bits) - 1u);
    return lo | (hi << lo_bits);
}

template <typename scalar_t>
__device__ __forceinline__ float cb_to_float(scalar_t x) {
    return static_cast<float>(x);
}
