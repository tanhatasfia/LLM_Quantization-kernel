#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>



__device__ __forceinline__ uint32_t cb_mask(int bits) { return (1u << bits) - 1u; }

__device__ __forceinline__ uint32_t cb_unpack(const uint32_t* __restrict__ row, int k, int bits) {
    const int bitpos = k * bits, wi = bitpos >> 5, off = bitpos & 31;
    const uint32_t mask = cb_mask(bits);
    if (off + bits <= 32) return (row[wi] >> off) & mask;
    const int lo_bits = 32 - off, hi_bits = bits - lo_bits;
    const uint32_t lo = (row[wi] >> off) & ((1u << lo_bits) - 1u);
    const uint32_t hi = row[wi + 1] & ((1u << hi_bits) - 1u);
    return lo | (hi << lo_bits);
}


template <int BITS>
__device__ __forceinline__ void cb_load_chunk(const uint32_t* __restrict__ p, uint32_t (&w)[BITS]) {
    if constexpr (BITS == 4) {
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(p));
        w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
    } else if constexpr (BITS == 2) {
        const uint2 v = __ldg(reinterpret_cast<const uint2*>(p));
        w[0] = v.x; w[1] = v.y;
    } else {
        w[0] = __ldg(p); w[1] = __ldg(p + 1); w[2] = __ldg(p + 2);
    }
}


template <int BITS>
__device__ __forceinline__ uint32_t cb_code(const uint32_t (&w)[BITS], int j) {
    const int pos = j * BITS, wi = pos >> 5, off = pos & 31;
    uint32_t v = w[wi] >> off;
    if (off + BITS > 32) v |= w[wi + 1] << (32 - off);
    return v & ((1u << BITS) - 1u);
}


__device__ __forceinline__ float cb_code_to_float(uint32_t q) {
    return __uint_as_float(q | 0x4B000000u) - 8388608.0f;
}
