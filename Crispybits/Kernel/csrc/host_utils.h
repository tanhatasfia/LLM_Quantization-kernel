#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <string>

template <typename T> struct CbNative { using type = T; };
template <> struct CbNative<at::Half> { using type = __half; };
template <> struct CbNative<at::BFloat16> { using type = __nv_bfloat16; };

template <typename scalar_t>
inline const typename CbNative<scalar_t>::type* cb_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const typename CbNative<scalar_t>::type*>(t.data_ptr<scalar_t>());
}
template <typename scalar_t>
inline typename CbNative<scalar_t>::type* cb_mut_ptr(torch::Tensor& t) {
    return reinterpret_cast<typename CbNative<scalar_t>::type*>(t.data_ptr<scalar_t>());
}

inline int cb_sm_count_current() {           // cached by PyTorch per device
    return at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
}
inline int cb_cc_major_current() {
    return at::cuda::getCurrentDeviceProperties()->major;
}


inline cudaStream_t cb_stream() { return at::cuda::getCurrentCUDAStream(); }

inline int64_t cb_chunks(int64_t K) { return (K + 31) / 32; }


inline torch::Tensor cb_workspace(const torch::Tensor& ws, const torch::Tensor& like) {
    if (ws.defined() && ws.numel() >= 2) {
        TORCH_CHECK(ws.is_cuda() && ws.device() == like.device(), "workspace must be on x's device");
        TORCH_CHECK(ws.scalar_type() == at::kInt && ws.is_contiguous(), "workspace must be contiguous int32");
        return ws;
    }
    auto c = torch::empty({2}, like.options().dtype(torch::kInt32));
    C10_CUDA_CHECK(cudaMemsetAsync(c.data_ptr<int>(), 0, 2 * sizeof(int), cb_stream()));
    return c;
}

inline void cb_check_weight(const torch::Tensor& x, const torch::Tensor& packed,
                            const torch::Tensor& scales, const torch::Tensor& zeros,
                            int64_t bits, int64_t K, int64_t group_size, const char* name) {
    const std::string n(name);
    TORCH_CHECK(packed.is_cuda() && scales.is_cuda() && zeros.is_cuda(), n, ": tensors must be CUDA");
    TORCH_CHECK(packed.device() == x.device(), n, ": weight and activation on different devices");
    TORCH_CHECK(packed.scalar_type() == at::kInt, n, ": packed must be torch.int32");
    TORCH_CHECK(packed.dim() == 2 && scales.dim() == 2, n, ": packed/scales must be 2-D");
    TORCH_CHECK(packed.is_contiguous() && scales.is_contiguous() && zeros.is_contiguous(), n, ": weight tensors must be contiguous");
    TORCH_CHECK(scales.sizes() == zeros.sizes(), n, ": scale/zero shape mismatch");
    TORCH_CHECK(scales.size(0) == packed.size(0), n, ": metadata rows != output rows");
    TORCH_CHECK(scales.scalar_type() == x.scalar_type() && zeros.scalar_type() == x.scalar_type(), n, ": metadata dtype must match x");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, n, ": bits must be 2/3/4");
    TORCH_CHECK(group_size > 0 && group_size % 32 == 0, n, ": group_size must be a positive multiple of 32");
    TORCH_CHECK(packed.size(1) == cb_chunks(K) * bits,
                n, ": packed row must hold exactly ceil(K/32)*bits words (rows padded to 32 codes)");
    TORCH_CHECK(scales.size(1) == (K + group_size - 1) / group_size, n, ": expected ceil(K/group_size) groups");
    const int64_t align = bits == 4 ? 16 : (bits == 2 ? 8 : 4);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % align == 0,
                n, ": packed storage must be ", align, "-byte aligned for vector loads");
}

inline void cb_check_x(const torch::Tensor& x, int64_t K) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be [B,K]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(1) == K, "x K != in_features");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat,
                "x must be fp16/bf16/fp32");
}

inline bool cb_has_bias(const torch::Tensor& bias, const torch::Tensor& x, int64_t N) {
    if (!bias.defined() || bias.numel() == 0) return false;
    TORCH_CHECK(bias.is_cuda() && bias.device() == x.device(), "bias must be on x's device");
    TORCH_CHECK(bias.numel() == N, "bias length must equal out_features");
    TORCH_CHECK(bias.scalar_type() == x.scalar_type(), "bias dtype must match x");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    return true;
}


inline int cb_blocks(int jobs, int64_t rho, int sms) {
    if (rho <= 0) return std::max(1, jobs);
    return (int)std::max<int64_t>(1, std::min<int64_t>(jobs, rho * (int64_t)sms));
}


#define CB_DISPATCH_FLOAT_TYPES(TYPE, NAME, ...)                                  \
    AT_DISPATCH_SWITCH(TYPE, NAME,                                                \
                       AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)       \
                       AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)        \
                       AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))
