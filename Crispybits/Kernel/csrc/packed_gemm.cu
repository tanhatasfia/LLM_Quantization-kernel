
#include "host_utils.h"
#include "bitpack.cuh"
#include "common.cuh"
#include <mma.h>

namespace {
constexpr int BM = 64, BNT = 64, BK = 32;
constexpr int LDA = BK + 8, LDB = BK + 8, LDC = BNT + 4;
constexpr int GEMM_THREADS = 128;             

template <typename T, int BITS>
__device__ __forceinline__ void cb_gemm_tc_body(
    const T* __restrict__ x, const uint32_t* __restrict__ packed,
    const T* __restrict__ scales, const T* __restrict__ zeros,
    const T* __restrict__ bias, T* __restrict__ out,
    int M, int N, int K, int wpr, int groups, int group_size, int act) {
    using namespace nvcuda;
    __shared__ __align__(32) T sA[BM * LDA];
    __shared__ __align__(32) T sB[BNT * LDB];  
    __shared__ __align__(32) float sC[BM * LDC];

    const int tid = threadIdx.x, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BNT;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

   
    const int brow = tid & 63, bhalf = tid >> 6;
    const int bn = n0 + brow;
    const int nchunks = (K + 31) / 32;

    for (int kc = 0; kc < nchunks; ++kc) {

        #pragma unroll
        for (int e = tid; e < BM * BK; e += GEMM_THREADS) {
            const int m = e >> 5, kk = e & 31;
            const int gm = m0 + m, gk = kc * 32 + kk;
            sA[m * LDA + kk] = (gm < M && gk < K) ? x[(size_t)gm * K + gk] : cb_from_float<T>(0.f);
        }
       
        if (bn < N) {
            uint32_t w[BITS];
            cb_load_chunk<BITS>(packed + (size_t)bn * wpr + (size_t)kc * BITS, w);
            const int g = (kc * 32) / group_size;
            const float s = cb_to_float(scales[(size_t)bn * groups + g]);
            const float z = cb_to_float(zeros[(size_t)bn * groups + g]);
            if (bhalf == 0) {
                #pragma unroll
                for (int j = 0; j < 16; ++j)
                    sB[brow * LDB + j] = cb_from_float<T>((cb_code_to_float(cb_code<BITS>(w, j)) - z) * s);
            } else {
                #pragma unroll
                for (int j = 16; j < 32; ++j)
                    sB[brow * LDB + j] = cb_from_float<T>((cb_code_to_float(cb_code<BITS>(w, j)) - z) * s);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 16; ++j) sB[brow * LDB + bhalf * 16 + j] = cb_from_float<T>(0.f);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], sA + (wm * 32 + i * 16) * LDA + kk, LDA);
            #pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], sB + (wn * 32 + j * 16) * LDB + kk, LDB);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(sC + (wm * 32 + i * 16) * LDC + wn * 32 + j * 16, acc[i][j], LDC, wmma::mem_row_major);
    __syncthreads();

   
    for (int e = tid; e < BM * BNT; e += GEMM_THREADS) {
        const int m = e / BNT, n = e - m * BNT;
        const int gm = m0 + m, gn = n0 + n;
        if (gm < M && gn < N) {
            float v = sC[m * LDC + n];
            if (bias) v += cb_to_float(bias[gn]);
            out[(size_t)gm * N + gn] = cb_from_float<T>(cb_apply_act(v, act));
        }
    }
}

template <typename T, int BITS>
__global__ void __launch_bounds__(GEMM_THREADS) cb_gemm_tc_kernel(
    const T* __restrict__ x, const uint32_t* __restrict__ packed,
    const T* __restrict__ scales, const T* __restrict__ zeros,
    const T* __restrict__ bias, T* __restrict__ out,
    int M, int N, int K, int wpr, int groups, int group_size, int act) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
    if constexpr (std::is_same<T, __nv_bfloat16>::value) { __trap(); } else   
#endif
    {
        cb_gemm_tc_body<T, BITS>(x, packed, scales, zeros, bias, out, M, N, K, wpr, groups, group_size, act);
    }
}


constexpr int TM = 16, TN = 16, TKS = 32;
template <typename T, int BITS>
__global__ void cb_gemm_scalar_kernel(
    const T* __restrict__ x, const uint32_t* __restrict__ packed,
    const T* __restrict__ scales, const T* __restrict__ zeros,
    const T* __restrict__ bias, T* __restrict__ out,
    int M, int N, int K, int wpr, int groups, int group_size, int act) {
    __shared__ float sx[TM][TKS + 1];
    __shared__ float sw[TN][TKS + 1];
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * TN + tx;
    const int m = blockIdx.y * TM + ty, n = blockIdx.x * TN + tx;
    float acc = 0.f;
    const int nchunks = (K + 31) / 32;
    for (int kc = 0; kc < nchunks; ++kc) {
        for (int e = tid; e < TM * TKS; e += TM * TN) {
            const int mm = e / TKS, kk = e % TKS, gm = blockIdx.y * TM + mm, gk = kc * 32 + kk;
            sx[mm][kk] = (gm < M && gk < K) ? cb_to_float(x[(size_t)gm * K + gk]) : 0.f;
        }

        for (int e = tid; e < TN * TKS; e += TM * TN) {
            const int nn = e / TKS, kk = e % TKS, gn = blockIdx.x * TN + nn;
            float wv = 0.f;
            if (gn < N) {
                const int g = (kc * 32) / group_size;
                const uint32_t q = cb_unpack(packed + (size_t)gn * wpr, kc * 32 + kk, BITS);
                wv = (float(q) - cb_to_float(zeros[(size_t)gn * groups + g])) * cb_to_float(scales[(size_t)gn * groups + g]);
            }
            sw[nn][kk] = wv;
        }
        __syncthreads();
        #pragma unroll 8
        for (int kk = 0; kk < TKS; ++kk) acc = fmaf(sx[ty][kk], sw[tx][kk], acc);
        __syncthreads();
    }
    if (m < M && n < N) {
        if (bias) acc += cb_to_float(bias[n]);
        out[(size_t)m * N + n] = cb_from_float<T>(cb_apply_act(acc, act));
    }
}

template <typename T, int BITS>
void cb_gemm_launch(const torch::Tensor& x, const torch::Tensor& packed, const torch::Tensor& scales,
                    const torch::Tensor& zeros, const T* bias, torch::Tensor& out,
                    int K, int group_size, int act, bool tensor_cores) {
    const int M = x.size(0), N = packed.size(0), wpr = packed.size(1), groups = scales.size(1);
    const T* xp = reinterpret_cast<const T*>(x.data_ptr());
    const uint32_t* pp = reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>());
    const T* sp = reinterpret_cast<const T*>(scales.data_ptr());
    const T* zp = reinterpret_cast<const T*>(zeros.data_ptr());
    T* op = reinterpret_cast<T*>(out.data_ptr());
    if constexpr (!std::is_same<T, float>::value) {
        if (tensor_cores) {
            dim3 grid((N + BNT - 1) / BNT, (M + BM - 1) / BM);
            cb_gemm_tc_kernel<T, BITS><<<grid, GEMM_THREADS, 0, cb_stream()>>>(xp, pp, sp, zp, bias, op, M, N, K, wpr, groups, group_size, act);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }
    dim3 block(TN, TM), grid((N + TN - 1) / TN, (M + TM - 1) / TM);
    cb_gemm_scalar_kernel<T, BITS><<<grid, block, 0, cb_stream()>>>(xp, pp, sp, zp, bias, op, M, N, K, wpr, groups, group_size, act);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} 

torch::Tensor packed_gemm_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, torch::Tensor zeros,
                               torch::Tensor bias, int64_t bits, int64_t in_features, int64_t group_size,
                               int64_t act, bool use_tensor_cores) {
    cb_check_x(x, in_features);
    cb_check_weight(x, packed, scales, zeros, bits, in_features, group_size, "packed_gemm");
    TORCH_CHECK(act >= 0 && act <= 2, "act must be 0 (none), 1 (relu) or 2 (silu)");
    const c10::cuda::CUDAGuard guard(x.device());
    const bool has_bias = cb_has_bias(bias, x, packed.size(0));
    auto out = torch::empty({x.size(0), packed.size(0)}, x.options());
    if (x.size(0) == 0) return out;
    const int major = cb_cc_major_current();
    const bool tc = use_tensor_cores &&
                    ((x.scalar_type() == at::kHalf && major >= 7) || (x.scalar_type() == at::kBFloat16 && major >= 8));
    CB_DISPATCH_FLOAT_TYPES(x.scalar_type(), "crispybits_packed_gemm", [&] {
        using T = typename CbNative<scalar_t>::type;
        const T* bp = has_bias ? reinterpret_cast<const T*>(bias.data_ptr()) : nullptr;
        if (bits == 2)      cb_gemm_launch<T, 2>(x, packed, scales, zeros, bp, out, in_features, group_size, act, tc);
        else if (bits == 3) cb_gemm_launch<T, 3>(x, packed, scales, zeros, bp, out, in_features, group_size, act, tc);
        else                cb_gemm_launch<T, 4>(x, packed, scales, zeros, bp, out, in_features, group_size, act, tc);
    });
    return out;
}
