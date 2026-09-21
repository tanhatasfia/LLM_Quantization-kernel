#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include "bitpack.cuh"

namespace {
constexpr int THREADS = 128;
constexpr int BN = 128;
constexpr int TK = 256;

template <typename scalar_t, int BITS>
__global__ void packed_gemv_persistent_kernel(
    const scalar_t* __restrict__ x,
    const uint32_t* __restrict__ packed,
    const scalar_t* __restrict__ scales,
    const scalar_t* __restrict__ zeros,
    float* __restrict__ partial,
    int* __restrict__ next_job,
    int B,
    int N,
    int K,
    int words_per_row,
    int groups_per_row,
    int group_size,
    int split_k,
    int n_tiles,
    int total_jobs) {

    extern __shared__ unsigned char smem_raw[];
    scalar_t* sx = reinterpret_cast<scalar_t*>(smem_raw);

    while (true) {
        int job = atomicAdd(next_job, 1);
        if (job >= total_jobs) break;

        int tmp = job;
        int split = tmp % split_k; tmp /= split_k;
        int tile = tmp % n_tiles; tmp /= n_tiles;
        int batch = tmp;

        const int n = tile * BN + threadIdx.x;
        const int k0 = (K * split) / split_k;
        const int k1 = (K * (split + 1)) / split_k;
        float acc = 0.0f;

        const uint32_t* row = (n < N) ? (packed + (size_t)n * words_per_row) : nullptr;
        const scalar_t* srow = (n < N) ? (scales + (size_t)n * groups_per_row) : nullptr;
        const scalar_t* zrow = (n < N) ? (zeros + (size_t)n * groups_per_row) : nullptr;

        for (int kb = k0; kb < k1; kb += TK) {
            const int chunk = min(TK, k1 - kb);
            for (int j = threadIdx.x; j < chunk; j += blockDim.x) {
                sx[j] = x[(size_t)batch * K + kb + j];
            }
            __syncthreads();

            if (n < N) {
                #pragma unroll 4
                for (int j = 0; j < chunk; ++j) {
                    const int k = kb + j;
                    const int g = k / group_size;
                    const uint32_t q = cb_unpack(row, k, BITS);
                    const float w = (float(q) - cb_to_float(zrow[g])) * cb_to_float(srow[g]);
                    acc = fmaf(cb_to_float(sx[j]), w, acc);
                }
            }
            __syncthreads();
        }

        if (n < N) {
            partial[((size_t)split * B + batch) * N + n] = acc;
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void reduce_partials_kernel(
    const float* __restrict__ partial,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int B, int N, int split_k, bool has_bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N;
    if (idx >= total) return;
    int n = idx % N;
    float v = 0.0f;
    #pragma unroll 4
    for (int s = 0; s < split_k; ++s) {
        v += partial[(size_t)s * total + idx];
    }
    if (has_bias) v += cb_to_float(bias[n]);
    out[idx] = static_cast<scalar_t>(v);
}

template <typename scalar_t, int BITS>
void launch_gemv_typed(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor bias,
    torch::Tensor partial,
    torch::Tensor counter,
    torch::Tensor out,
    int K,
    int group_size,
    int split_k,
    int rho,
    int sms) {
    const int B = x.size(0);
    const int N = packed.size(0);
    const int words = packed.size(1);
    const int groups = scales.size(1);
    const int n_tiles = (N + BN - 1) / BN;
    const int jobs = B * n_tiles * split_k;
    const int blocks = std::max(1, std::min(jobs, std::max(1, rho) * sms));
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    packed_gemv_persistent_kernel<scalar_t, BITS><<<blocks, THREADS, TK * sizeof(scalar_t), stream>>>(
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()),
        reinterpret_cast<const scalar_t*>(scales.data_ptr()),
        reinterpret_cast<const scalar_t*>(zeros.data_ptr()),
        partial.data_ptr<float>(),
        counter.data_ptr<int>(),
        B, N, K, words, groups, group_size, split_k, n_tiles, jobs);

    int total = B * N;
    int rb = (total + 255) / 256;
    const bool has_bias = bias.defined() && bias.numel() == N;
    reduce_partials_kernel<scalar_t><<<rb, 256, 0, stream>>>(
        partial.data_ptr<float>(),
        has_bias ? reinterpret_cast<const scalar_t*>(bias.data_ptr()) : nullptr,
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        B, N, split_k, has_bias);
}

} // namespace

torch::Tensor packed_gemv_cuda(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor bias,
    int64_t bits,
    int64_t in_features,
    int64_t group_size,
    int64_t split_k,
    int64_t rho) {

    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && scales.is_cuda() && zeros.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be [B,K]");
    TORCH_CHECK(packed.scalar_type() == at::kInt, "packed must be torch.int32");
    TORCH_CHECK(x.size(1) == in_features, "x K != in_features");
    TORCH_CHECK(scales.sizes() == zeros.sizes(), "scale/zero shape mismatch");
    TORCH_CHECK(scales.size(0) == packed.size(0), "metadata rows != output rows");
    TORCH_CHECK(split_k >= 1, "split_k must be >=1");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits must be 2/3/4");
    TORCH_CHECK(scales.scalar_type() == x.scalar_type() && zeros.scalar_type() == x.scalar_type(), "metadata dtype must match x");
    if (bias.defined() && bias.numel() > 0) TORCH_CHECK(bias.scalar_type() == x.scalar_type(), "bias dtype must match x");

    auto out = torch::empty({x.size(0), packed.size(0)}, x.options());
    auto partial = torch::empty({split_k, x.size(0), packed.size(0)}, x.options().dtype(torch::kFloat32));
    auto counter = torch::zeros({1}, x.options().dtype(torch::kInt32));

    int dev = x.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int sms = prop.multiProcessorCount;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "crispybits_packed_gemv", [&] {
        if (bits == 2) launch_gemv_typed<scalar_t,2>(x,packed,scales,zeros,bias,partial,counter,out,in_features,group_size,split_k,rho,sms);
        else if (bits == 3) launch_gemv_typed<scalar_t,3>(x,packed,scales,zeros,bias,partial,counter,out,in_features,group_size,split_k,rho,sms);
        else launch_gemv_typed<scalar_t,4>(x,packed,scales,zeros,bias,partial,counter,out,in_features,group_size,split_k,rho,sms);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

int64_t crispybits_sm_count() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    return prop.multiProcessorCount;
}

int64_t crispybits_max_rho(int64_t bits) {
    int blocks = 0;
    const size_t smem = TK * sizeof(at::Half);
    if (bits == 2) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, packed_gemv_persistent_kernel<at::Half,2>, THREADS, smem);
    } else if (bits == 3) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, packed_gemv_persistent_kernel<at::Half,3>, THREADS, smem);
    } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, packed_gemv_persistent_kernel<at::Half,4>, THREADS, smem);
    }
    return blocks;
}
