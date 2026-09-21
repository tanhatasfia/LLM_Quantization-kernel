#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "bitpack.cuh"

namespace {
constexpr int TM = 16;
constexpr int TN = 16;
constexpr int TK = 16;

template <typename scalar_t, int BITS>
__global__ void packed_gemm_kernel(
    const scalar_t* __restrict__ x,
    const uint32_t* __restrict__ packed,
    const scalar_t* __restrict__ scales,
    const scalar_t* __restrict__ zeros,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int M, int N, int K, int words, int groups, int group_size, bool has_bias) {
    __shared__ scalar_t sx[TM][TK];
    const int m = blockIdx.y * TM + threadIdx.y;
    const int n = blockIdx.x * TN + threadIdx.x;
    float acc = 0.0f;
    const uint32_t* row = n < N ? packed + (size_t)n * words : nullptr;
    const scalar_t* srow = n < N ? scales + (size_t)n * groups : nullptr;
    const scalar_t* zrow = n < N ? zeros + (size_t)n * groups : nullptr;

    for (int kb = 0; kb < K; kb += TK) {
        int kload = kb + threadIdx.x;
        if (m < M && kload < K) sx[threadIdx.y][threadIdx.x] = x[(size_t)m*K + kload];
        else sx[threadIdx.y][threadIdx.x] = static_cast<scalar_t>(0.0f);
        __syncthreads();

        if (m < M && n < N) {
            #pragma unroll
            for (int j = 0; j < TK; ++j) {
                int k = kb + j;
                if (k < K) {
                    int g = k / group_size;
                    uint32_t q = cb_unpack(row, k, BITS);
                    float w = (float(q) - cb_to_float(zrow[g])) * cb_to_float(srow[g]);
                    acc = fmaf(cb_to_float(sx[threadIdx.y][j]), w, acc);
                }
            }
        }
        __syncthreads();
    }
    if (m < M && n < N) {
        if (has_bias) acc += cb_to_float(bias[n]);
        out[(size_t)m*N+n] = static_cast<scalar_t>(acc);
    }
}

template <typename scalar_t, int BITS>
void launch(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, torch::Tensor zeros, torch::Tensor bias, torch::Tensor out, int K, int group_size) {
    int M=x.size(0), N=packed.size(0), words=packed.size(1), groups=scales.size(1);
    dim3 block(TN, TM);
    dim3 grid((N+TN-1)/TN, (M+TM-1)/TM);
    bool has_bias = bias.defined() && bias.numel()==N;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    packed_gemm_kernel<scalar_t,BITS><<<grid,block,0,stream>>>(
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()),
        reinterpret_cast<const scalar_t*>(scales.data_ptr()),
        reinterpret_cast<const scalar_t*>(zeros.data_ptr()),
        has_bias ? reinterpret_cast<const scalar_t*>(bias.data_ptr()) : nullptr,
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        M,N,K,words,groups,group_size,has_bias);
}
}

torch::Tensor packed_gemm_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, torch::Tensor zeros, torch::Tensor bias, int64_t bits, int64_t in_features, int64_t group_size) {
    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && scales.is_cuda() && zeros.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(x.dim()==2, "x must be [M,K]");
    TORCH_CHECK(x.size(1)==in_features, "x K != in_features");
    TORCH_CHECK(packed.scalar_type()==at::kInt, "packed must be int32");
    TORCH_CHECK(scales.scalar_type()==x.scalar_type() && zeros.scalar_type()==x.scalar_type(), "metadata dtype must match x");
    auto out = torch::empty({x.size(0), packed.size(0)}, x.options());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "crispybits_packed_gemm", [&]{
        if(bits==2) launch<scalar_t,2>(x,packed,scales,zeros,bias,out,in_features,group_size);
        else if(bits==3) launch<scalar_t,3>(x,packed,scales,zeros,bias,out,in_features,group_size);
        else if(bits==4) launch<scalar_t,4>(x,packed,scales,zeros,bias,out,in_features,group_size);
        else TORCH_CHECK(false, "bits must be 2/3/4");
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
