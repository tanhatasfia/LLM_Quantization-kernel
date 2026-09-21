#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>


enum CbAct : int { CB_ACT_NONE = 0, CB_ACT_RELU = 1, CB_ACT_SILU = 2 };


enum CbMode : int { CB_MODE_GEMV = 0, CB_MODE_QKV = 1, CB_MODE_GATEUP = 2 };

__device__ __forceinline__ float cb_apply_act(float v, int act) {
    if (act == CB_ACT_RELU) return v > 0.f ? v : 0.f;
    if (act == CB_ACT_SILU) return v / (1.0f + __expf(-v));
    return v;
}


template <typename T> __device__ __forceinline__ float cb_to_float(T x);
template <> __device__ __forceinline__ float cb_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float cb_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float cb_to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T cb_from_float(float x);
template <> __device__ __forceinline__ float cb_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half cb_from_float<__half>(float x) { return __float2half_rn(x); }
template <> __device__ __forceinline__ __nv_bfloat16 cb_from_float<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }


__device__ __forceinline__ int cb_claim_job(int* __restrict__ ws, int* s_job) {
    if (threadIdx.x == 0) *s_job = atomicAdd(ws, 1);
    __syncthreads();
    return *s_job;
}


__device__ __forceinline__ void cb_finish(int* __restrict__ ws) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const int done = atomicAdd(ws + 1, 1);
        if (done == (int)gridDim.x - 1) {
            atomicExch(ws, 0);
            atomicExch(ws + 1, 0);
        }
    }
}


__device__ __forceinline__ void cb_decode_job(int job, int split_k, int n_tiles,
                                              int& split, int& tile, int& batch) {
    split = job % split_k; job /= split_k;
    tile = job % n_tiles;  job /= n_tiles;
    batch = job;
}


__device__ __forceinline__ float cb_rope(float v, float partner, int d, int head_dim, float c, float s) {
    return (d < head_dim / 2) ? (v * c - partner * s) : (v * c + partner * s);
}


__device__ __forceinline__ int cb_unperm(int r, int hd) {
    if (hd <= 0) return r;
    const int head = r / hd, p = r - head * hd;
    return head * hd + ((p & 1) ? (p >> 1) + hd / 2 : (p >> 1));
}
