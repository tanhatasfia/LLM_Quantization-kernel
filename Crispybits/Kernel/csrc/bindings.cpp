#include <torch/extension.h>
#include <vector>

torch::Tensor packed_gemv_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, torch::Tensor zeros,
                               torch::Tensor bias, int64_t bits, int64_t in_features, int64_t group_size,
                               int64_t split_k, int64_t rho, int64_t act, torch::Tensor workspace);
torch::Tensor packed_gemm_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, torch::Tensor zeros,
                               torch::Tensor bias, int64_t bits, int64_t in_features, int64_t group_size,
                               int64_t act, bool use_tensor_cores);
torch::Tensor fused_gate_up_cuda(torch::Tensor x, torch::Tensor wg, torch::Tensor sg, torch::Tensor zg,
                                 torch::Tensor wu, torch::Tensor su, torch::Tensor zu,
                                 int64_t bits, int64_t in_features, int64_t group_size, int64_t split_k,
                                 int64_t rho, torch::Tensor workspace);
std::vector<torch::Tensor> fused_qkv_cuda(torch::Tensor x,
    torch::Tensor wq, torch::Tensor sq, torch::Tensor zq, torch::Tensor bq,
    torch::Tensor wk, torch::Tensor sk, torch::Tensor zk, torch::Tensor bk,
    torch::Tensor wv, torch::Tensor sv, torch::Tensor zv, torch::Tensor bv,
    torch::Tensor rope_cos, torch::Tensor rope_sin, int64_t head_dim, int64_t perm_hd,
    int64_t bits, int64_t in_features, int64_t group_size, int64_t split_k, int64_t rho,
    torch::Tensor workspace);
int64_t crispybits_sm_count();
int64_t crispybits_tile_n();
int64_t crispybits_max_rho_mode(int64_t bits, int64_t mode);

int64_t crispybits_max_rho(int64_t bits, const std::string& kind) {
    if (kind == "gemv") return crispybits_max_rho_mode(bits, 0);
    if (kind == "qkv") return crispybits_max_rho_mode(bits, 1);
    if (kind == "gate_up") return crispybits_max_rho_mode(bits, 2);
    TORCH_CHECK(false, "kind must be 'gemv', 'qkv' or 'gate_up'");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("packed_gemv", &packed_gemv_cuda, "Packed W2/W3/W4 warp-per-row persistent GEMV with SM-underfill Split-K");
  m.def("packed_gemm", &packed_gemm_cuda, "Packed W2/W3/W4 prefill GEMM (WMMA tensor cores for fp16/bf16)");
  m.def("fused_gate_up", &fused_gate_up_cuda, "Fused Gate/Up packed GEMV + SiLU gating");
  m.def("fused_qkv", &fused_qkv_cuda, "Fused Q/K/V packed GEMV (GQA widths, biases, optional in-kernel RoPE)");
  m.def("sm_count", &crispybits_sm_count, "Streaming multiprocessor count of the current device");
  m.def("tile_n", &crispybits_tile_n, "Virtual output rows per decode CTA tile (B_N of the underfill rule)");
  m.def("max_rho", &crispybits_max_rho, "Occupancy-limited resident CTAs/SM", py::arg("bits"), py::arg("kind") = "gemv");
}
