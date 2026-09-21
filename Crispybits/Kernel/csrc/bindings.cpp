#include <torch/extension.h>
#include <vector>

torch::Tensor packed_gemv_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);
torch::Tensor packed_gemm_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);
torch::Tensor fused_gate_up_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);
std::vector<torch::Tensor> fused_qkv_cuda(torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 int64_t,int64_t,int64_t,int64_t,int64_t);
int64_t crispybits_sm_count();
int64_t crispybits_max_rho(int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("packed_gemv", &packed_gemv_cuda, "Packed W2/W3/W4 persistent GEMV (CUDA)");
  m.def("packed_gemm", &packed_gemm_cuda, "Packed W2/W3/W4 tiled GEMM (CUDA)");
  m.def("fused_gate_up", &fused_gate_up_cuda, "Fused Gate/Up packed GEMV + SiLU gate (CUDA)");
  m.def("fused_qkv", &fused_qkv_cuda, "Fused Q/K/V packed GEMV (CUDA)");
  m.def("sm_count", &crispybits_sm_count, "Streaming multiprocessor count");
  m.def("max_rho", &crispybits_max_rho, "Occupancy-limited resident CTAs/SM for packed GEMV");
}
