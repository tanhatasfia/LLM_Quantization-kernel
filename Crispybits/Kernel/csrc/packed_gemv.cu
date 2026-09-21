
#include "host_utils.h"
#include "bitpack.cuh"
#include "common.cuh"
#include <vector>

namespace {
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;
constexpr int RPW = 8;                
constexpr int BN = WARPS * RPW;       
constexpr int TC = 64;                 
constexpr int XPITCH = TC + 1;         
static_assert(RPW % 2 == 0, "RPW must be even for paired epilogues");

struct Proj {
    const uint32_t* w; const void* s; const void* z; const void* b; void* out; int n;
};

struct Params {
    Proj p[3];
    int V, B, K, nchunks, wpr, groups, group_size;
    int split_k, ntiles, jobs, act;
    float* partial;                   
    int* ws;                          
    const float* rope_cos;            
    const float* rope_sin;
    int head_dim;                   
    int perm_hd;                    
};


template <int MODE>
__device__ __forceinline__ void cb_map(const Params& P, int v, int& proj, int& r) {
    if constexpr (MODE == CB_MODE_GATEUP) { proj = v & 1; r = v >> 1; }
    else if constexpr (MODE == CB_MODE_QKV) {
        const int n0 = P.p[0].n, n1 = P.p[1].n;
        if (v < n0)           { proj = 0; r = v; }
        else if (v < n0 + n1) { proj = 1; r = v - n0; }
        else                  { proj = 2; r = v - n0 - n1; }
    } else { proj = 0; r = v; }
}


#define CB_SEL(P, proj, field) \
    ((proj) == 0 ? (P).p[0].field : ((proj) == 1 ? (P).p[1].field : (P).p[2].field))

template <typename T>
__device__ __forceinline__ float cb_bias(const void* b, int r) {
    return b ? cb_to_float(reinterpret_cast<const T*>(b)[r]) : 0.f;
}


template <typename T, int MODE>
__device__ __forceinline__ void cb_epilogue(const Params& P, int b, int v, float mine, float partner) {
    int proj, r;
    cb_map<MODE>(P, v, proj, r);
    if constexpr (MODE == CB_MODE_GEMV) {
        const float y = cb_apply_act(mine + cb_bias<T>(P.p[0].b, r), P.act);
        reinterpret_cast<T*>(P.p[0].out)[(size_t)b * P.p[0].n + r] = cb_from_float<T>(y);
    } else if constexpr (MODE == CB_MODE_GATEUP) {
        if (proj != 0) return;                          
        const float y = cb_apply_act(mine, CB_ACT_SILU) * partner;
        reinterpret_cast<T*>(P.p[0].out)[(size_t)b * P.p[0].n + r] = cb_from_float<T>(y);
    } else {
        const void* bias = CB_SEL(P, proj, b);
        void* out = CB_SEL(P, proj, out);
        const int n = CB_SEL(P, proj, n);
        float y = mine + cb_bias<T>(bias, r);
        const int hd = proj < 2 ? P.perm_hd : 0;
        const int orig = cb_unperm(r, hd);
        if (P.rope_cos != nullptr && proj < 2) {
            const float yp = partner + cb_bias<T>(bias, r ^ 1);
            const int d = orig % P.head_dim;
            const float c = P.rope_cos[(size_t)b * P.head_dim + d];
            const float s = P.rope_sin[(size_t)b * P.head_dim + d];
            y = cb_rope(y, yp, d, P.head_dim, c, s);
        }
        reinterpret_cast<T*>(out)[(size_t)b * n + orig] = cb_from_float<T>(y);
    }
}

template <typename T, int BITS, int MODE>
__global__ void __launch_bounds__(THREADS) cb_gemv_kernel(const T* __restrict__ x, const Params P) {
    __shared__ float xs[32 * XPITCH];
    __shared__ int s_job;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    while (true) {
        const int job = cb_claim_job(P.ws, &s_job);       
        if (job >= P.jobs) break;
        int split, tile, b;
        cb_decode_job(job, P.split_k, P.ntiles, split, tile, b);
        const int c0 = (int)(((long long)P.nchunks * split) / P.split_k);
        const int c1 = (int)(((long long)P.nchunks * (split + 1)) / P.split_k);
        const int v0 = tile * BN + warp * RPW;          

        float acc[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

        const T* xb = x + (size_t)b * P.K;
        for (int cb = c0; cb < c1; cb += TC) {
            const int nc = min(TC, c1 - cb);
          
            for (int e = threadIdx.x; e < TC * 32; e += THREADS) {
                const int c = e >> 5, j = e & 31;
                const int k = (cb + c) * 32 + j;
                xs[j * XPITCH + c] = (c < nc && k < P.K) ? cb_to_float(xb[k]) : 0.f;
            }
            __syncthreads();

            for (int cc = lane; cc < nc; cc += 32) {
                float xv[32];
                float xsum = 0.f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) { xv[j] = xs[j * XPITCH + cc]; xsum += xv[j]; }
                const int chunk = cb + cc;
                const int g = (chunk * 32) / P.group_size;   

                #pragma unroll
                for (int r = 0; r < RPW; ++r) {
                    const int v = v0 + r;
                    if (v < P.V) {                         
                        int proj, lr;
                        cb_map<MODE>(P, v, proj, lr);
                        uint32_t w[BITS];
                        cb_load_chunk<BITS>(CB_SEL(P, proj, w) + (size_t)lr * P.wpr + (size_t)chunk * BITS, w);
                        const T* sp = reinterpret_cast<const T*>(CB_SEL(P, proj, s)) + (size_t)lr * P.groups;
                        const T* zp = reinterpret_cast<const T*>(CB_SEL(P, proj, z)) + (size_t)lr * P.groups;
                        float d = 0.f;
                        #pragma unroll
                        for (int j = 0; j < 32; ++j) d = fmaf(xv[j], cb_code_to_float(cb_code<BITS>(w, j)), d);
                        const float s = cb_to_float(sp[g]), z = cb_to_float(zp[g]);
                        acc[r] = fmaf(s, d - z * xsum, acc[r]);
                    }
                }
            }
            __syncthreads();
        }

     
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
        }

        float mine = 0.f, partner = 0.f;
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (r == lane) mine = acc[r];
            if ((r ^ 1) == lane) partner = acc[r];
        }
        const int v = v0 + lane;
        if (lane < RPW && v < P.V) {
            if (P.split_k == 1) cb_epilogue<T, MODE>(P, b, v, mine, partner);
            else P.partial[((size_t)split * P.B + b) * P.V + v] = mine;
        }
        __syncthreads();                              
    }
    cb_finish(P.ws);
}


template <typename T, int MODE>
__global__ void cb_reduce_kernel(const Params P) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = P.B * P.V;
    if (i >= total) return;
    const int b = i / P.V, v = i - b * P.V;
    float mine = 0.f, partner = 0.f;
    const bool paired = (MODE == CB_MODE_GATEUP) || (MODE == CB_MODE_QKV && P.rope_cos != nullptr);
    for (int s = 0; s < P.split_k; ++s) {
        const float* ps = P.partial + ((size_t)s * P.B + b) * P.V;
        mine += ps[v];
        if (paired && (v ^ 1) < P.V) partner += ps[v ^ 1];
    }
    cb_epilogue<T, MODE>(P, b, v, mine, partner);
}

template <typename T, int BITS, int MODE>
void cb_launch(const torch::Tensor& x, Params P, int64_t rho, const torch::Tensor& ws_in) {
    P.nchunks = (int)cb_chunks(P.K);
    P.split_k = std::max(1, std::min(P.split_k, P.nchunks));
    P.ntiles = (P.V + BN - 1) / BN;
    P.jobs = P.B * P.ntiles * P.split_k;
    const int blocks = cb_blocks(P.jobs, rho, cb_sm_count_current());
    cudaStream_t stream = cb_stream();

    torch::Tensor ws = cb_workspace(ws_in, x);
    P.ws = ws.data_ptr<int>();
    torch::Tensor partial;
    if (P.split_k > 1) {
        partial = torch::empty({P.split_k, P.B, P.V}, x.options().dtype(torch::kFloat32));
        P.partial = partial.data_ptr<float>();
    } else {
        P.partial = nullptr;
    }
    cb_gemv_kernel<T, BITS, MODE><<<blocks, THREADS, 0, stream>>>(reinterpret_cast<const T*>(x.data_ptr()), P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (P.split_k > 1) {
        const int total = P.B * P.V;
        cb_reduce_kernel<T, MODE><<<(total + 255) / 256, 256, 0, stream>>>(P);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <int MODE>
void cb_dispatch(const torch::Tensor& x, Params P, int64_t bits, int64_t rho, const torch::Tensor& ws) {
    CB_DISPATCH_FLOAT_TYPES(x.scalar_type(), "crispybits_gemv", [&] {
        using T = typename CbNative<scalar_t>::type;
        if (bits == 2)      cb_launch<T, 2, MODE>(x, P, rho, ws);
        else if (bits == 3) cb_launch<T, 3, MODE>(x, P, rho, ws);
        else                cb_launch<T, 4, MODE>(x, P, rho, ws);
    });
}

Proj cb_proj(const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& z,
             const torch::Tensor& bias, bool has_bias, torch::Tensor& out) {
    Proj p{};
    p.w = reinterpret_cast<const uint32_t*>(w.data_ptr<int32_t>());
    p.s = s.data_ptr(); p.z = z.data_ptr();
    p.b = has_bias ? bias.data_ptr() : nullptr;
    p.out = out.data_ptr();
    p.n = (int)w.size(0);
    return p;
}

Params cb_base(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s,
               int64_t K, int64_t group_size, int64_t split_k) {
    Params P{};
    P.B = (int)x.size(0); P.K = (int)K;
    P.wpr = (int)w.size(1); P.groups = (int)s.size(1); P.group_size = (int)group_size;
    P.split_k = (int)split_k; P.act = 0;
    P.rope_cos = nullptr; P.rope_sin = nullptr; P.head_dim = 0; P.perm_hd = 0;
    return P;
}

template <int MODE>
int64_t cb_occupancy(int64_t bits) {
    int blocks = 0;
    const void* f = bits == 2 ? (const void*)cb_gemv_kernel<__half, 2, MODE>
                  : bits == 3 ? (const void*)cb_gemv_kernel<__half, 3, MODE>
                              : (const void*)cb_gemv_kernel<__half, 4, MODE>;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, f, THREADS, 0));
    return blocks;
}
}


torch::Tensor packed_gemv_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, torch::Tensor zeros,
                               torch::Tensor bias, int64_t bits, int64_t in_features, int64_t group_size,
                               int64_t split_k, int64_t rho, int64_t act, torch::Tensor workspace) {
    cb_check_x(x, in_features);
    cb_check_weight(x, packed, scales, zeros, bits, in_features, group_size, "packed_gemv");
    TORCH_CHECK(split_k >= 1, "split_k must be >= 1");
    TORCH_CHECK(act >= 0 && act <= 2, "act must be 0 (none), 1 (relu) or 2 (silu)");
    const c10::cuda::CUDAGuard guard(x.device());
    const bool hb = cb_has_bias(bias, x, packed.size(0));
    auto out = torch::empty({x.size(0), packed.size(0)}, x.options());
    if (x.size(0) == 0) return out;
    Params P = cb_base(x, packed, scales, in_features, group_size, split_k);
    P.p[0] = cb_proj(packed, scales, zeros, bias, hb, out);
    P.p[1] = P.p[0]; P.p[2] = P.p[0];
    P.V = P.p[0].n; P.act = (int)act;
    cb_dispatch<CB_MODE_GEMV>(x, P, bits, rho, workspace);
    return out;
}

std::vector<torch::Tensor> fused_qkv_cuda(torch::Tensor x,
    torch::Tensor wq, torch::Tensor sq, torch::Tensor zq, torch::Tensor bq,
    torch::Tensor wk, torch::Tensor sk, torch::Tensor zk, torch::Tensor bk,
    torch::Tensor wv, torch::Tensor sv, torch::Tensor zv, torch::Tensor bv,
    torch::Tensor rope_cos, torch::Tensor rope_sin, int64_t head_dim, int64_t perm_hd,
    int64_t bits, int64_t in_features, int64_t group_size, int64_t split_k, int64_t rho,
    torch::Tensor workspace) {

    cb_check_x(x, in_features);
    cb_check_weight(x, wq, sq, zq, bits, in_features, group_size, "fused_qkv(q)");
    cb_check_weight(x, wk, sk, zk, bits, in_features, group_size, "fused_qkv(k)");
    cb_check_weight(x, wv, sv, zv, bits, in_features, group_size, "fused_qkv(v)");
    TORCH_CHECK(split_k >= 1, "split_k must be >= 1");
    const c10::cuda::CUDAGuard guard(x.device());
    const int64_t B = x.size(0);

    if (perm_hd > 0) {
        TORCH_CHECK(perm_hd % 2 == 0, "perm head_dim must be even");
        TORCH_CHECK(wq.size(0) % perm_hd == 0 && wk.size(0) % perm_hd == 0, "Q/K widths must be multiples of head_dim");
    }
    const bool rope = rope_cos.defined() && rope_cos.numel() > 0;
    torch::Tensor rc, rs;
    if (rope) {
        TORCH_CHECK(perm_hd > 0 && perm_hd == head_dim,
                    "fused RoPE needs Q/K weights stored pair-interleaved with the same head_dim "
                    "(ops.permute_for_rope / install_bitmap(fuse_rope=True))");
        TORCH_CHECK(rope_cos.sizes() == rope_sin.sizes(), "cos/sin shape mismatch");
        TORCH_CHECK(rope_cos.numel() == B * head_dim, "cos/sin must be [B, head_dim] (one row per x row)");
        rc = rope_cos.to(x.device(), torch::kFloat32).contiguous();
        rs = rope_sin.to(x.device(), torch::kFloat32).contiguous();
    }

    torch::Tensor W[3] = {wq, wk, wv}, S[3] = {sq, sk, sv}, Z[3] = {zq, zk, zv}, Bi[3] = {bq, bk, bv};
    std::vector<torch::Tensor> outs;
    for (int i = 0; i < 3; ++i) outs.push_back(torch::empty({B, W[i].size(0)}, x.options()));
    if (B == 0) return outs;

    Params P = cb_base(x, wq, sq, in_features, group_size, split_k);
    for (int i = 0; i < 3; ++i) P.p[i] = cb_proj(W[i], S[i], Z[i], Bi[i], cb_has_bias(Bi[i], x, W[i].size(0)), outs[i]);
    P.V = P.p[0].n + P.p[1].n + P.p[2].n;
    P.perm_hd = (int)perm_hd;
    if (rope) { P.rope_cos = rc.data_ptr<float>(); P.rope_sin = rs.data_ptr<float>(); P.head_dim = (int)head_dim; }
    cb_dispatch<CB_MODE_QKV>(x, P, bits, rho, workspace);
    return outs;
}

torch::Tensor fused_gate_up_cuda(torch::Tensor x, torch::Tensor wg, torch::Tensor sg, torch::Tensor zg,
                                 torch::Tensor wu, torch::Tensor su, torch::Tensor zu,
                                 int64_t bits, int64_t in_features, int64_t group_size, int64_t split_k,
                                 int64_t rho, torch::Tensor workspace) {
    cb_check_x(x, in_features);
    cb_check_weight(x, wg, sg, zg, bits, in_features, group_size, "fused_gate_up(gate)");
    cb_check_weight(x, wu, su, zu, bits, in_features, group_size, "fused_gate_up(up)");
    TORCH_CHECK(wg.sizes() == wu.sizes(), "gate/up packed shapes must match");
    TORCH_CHECK(split_k >= 1, "split_k must be >= 1");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty({x.size(0), wg.size(0)}, x.options());
    if (x.size(0) == 0) return out;
    torch::Tensor none;
    Params P = cb_base(x, wg, sg, in_features, group_size, split_k);
    P.p[0] = cb_proj(wg, sg, zg, none, false, out);
    P.p[1] = cb_proj(wu, su, zu, none, false, out);
    P.p[2] = P.p[0];
    P.V = 2 * P.p[0].n;
    cb_dispatch<CB_MODE_GATEUP>(x, P, bits, rho, workspace);
    return out;
}

int64_t crispybits_sm_count() { return cb_sm_count_current(); }
int64_t crispybits_tile_n() { return BN; }

// Occupancy-limited resident CTAs/SM of the fp16 instantiation of each decode kernel.
int64_t crispybits_max_rho_mode(int64_t bits, int64_t mode) {
    if (mode == CB_MODE_QKV) return cb_occupancy<CB_MODE_QKV>(bits);
    if (mode == CB_MODE_GATEUP) return cb_occupancy<CB_MODE_GATEUP>(bits);
    return cb_occupancy<CB_MODE_GEMV>(bits);
}
