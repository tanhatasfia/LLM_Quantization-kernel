#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "bitpack.cuh"

namespace {
constexpr int THREADS=128, BN=128, TK=256;

template <typename scalar_t, int BITS>
__global__ void gate_up_kernel(
    const scalar_t* __restrict__ x,
    const uint32_t* __restrict__ wg,
    const scalar_t* __restrict__ sg,
    const scalar_t* __restrict__ zg,
    const uint32_t* __restrict__ wu,
    const scalar_t* __restrict__ su,
    const scalar_t* __restrict__ zu,
    float* __restrict__ pg,
    float* __restrict__ pu,
    int* __restrict__ counter,
    int B,int N,int K,int words,int groups,int group_size,int split_k,int ntiles,int jobs) {
    extern __shared__ unsigned char raw[];
    scalar_t* sx = reinterpret_cast<scalar_t*>(raw);
    while(true){
        int job=atomicAdd(counter,1); if(job>=jobs) break;
        int t=job; int split=t%split_k; t/=split_k; int tile=t%ntiles; t/=ntiles; int batch=t;
        int n=tile*BN+threadIdx.x;
        int k0=(K*split)/split_k, k1=(K*(split+1))/split_k;
        float ag=0.f, au=0.f;
        const uint32_t* rg=n<N?wg+(size_t)n*words:nullptr;
        const uint32_t* ru=n<N?wu+(size_t)n*words:nullptr;
        const scalar_t* sgr=n<N?sg+(size_t)n*groups:nullptr;
        const scalar_t* zgr=n<N?zg+(size_t)n*groups:nullptr;
        const scalar_t* sur=n<N?su+(size_t)n*groups:nullptr;
        const scalar_t* zur=n<N?zu+(size_t)n*groups:nullptr;
        for(int kb=k0;kb<k1;kb+=TK){
            int chunk=min(TK,k1-kb);
            for(int j=threadIdx.x;j<chunk;j+=blockDim.x) sx[j]=x[(size_t)batch*K+kb+j];
            __syncthreads();
            if(n<N){
                for(int j=0;j<chunk;++j){
                    int k=kb+j,g=k/group_size;
                    float xv=cb_to_float(sx[j]);
                    uint32_t qg=cb_unpack(rg,k,BITS), qu=cb_unpack(ru,k,BITS);
                    float vg=(float(qg)-cb_to_float(zgr[g]))*cb_to_float(sgr[g]);
                    float vu=(float(qu)-cb_to_float(zur[g]))*cb_to_float(sur[g]);
                    ag=fmaf(xv,vg,ag); au=fmaf(xv,vu,au);
                }
            }
            __syncthreads();
        }
        if(n<N){ size_t off=((size_t)split*B+batch)*N+n; pg[off]=ag; pu[off]=au; }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void gate_up_reduce(const float* pg,const float* pu,scalar_t* out,int total,int split_k){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=total)return;
    float g=0.f,u=0.f; for(int s=0;s<split_k;++s){g+=pg[(size_t)s*total+i];u+=pu[(size_t)s*total+i];}
    float silu=g/(1.0f+expf(-g));
    out[i]=static_cast<scalar_t>(silu*u);
}

template <typename scalar_t,int BITS>
void launch(torch::Tensor x,torch::Tensor wg,torch::Tensor sg,torch::Tensor zg,torch::Tensor wu,torch::Tensor su,torch::Tensor zu,torch::Tensor pg,torch::Tensor pu,torch::Tensor ctr,torch::Tensor out,int K,int G,int split,int rho,int sms){
    int B=x.size(0),N=wg.size(0),words=wg.size(1),groups=sg.size(1),nt=(N+BN-1)/BN,jobs=B*nt*split;
    int blocks=std::max(1,std::min(jobs,std::max(1,rho)*sms));
    cudaStream_t stream=at::cuda::getDefaultCUDAStream();
    gate_up_kernel<scalar_t,BITS><<<blocks,THREADS,TK*sizeof(scalar_t),stream>>>(
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        reinterpret_cast<const uint32_t*>(wg.data_ptr<int32_t>()),reinterpret_cast<const scalar_t*>(sg.data_ptr()),reinterpret_cast<const scalar_t*>(zg.data_ptr()),
        reinterpret_cast<const uint32_t*>(wu.data_ptr<int32_t>()),reinterpret_cast<const scalar_t*>(su.data_ptr()),reinterpret_cast<const scalar_t*>(zu.data_ptr()),
        pg.data_ptr<float>(),pu.data_ptr<float>(),ctr.data_ptr<int>(),B,N,K,words,groups,G,split,nt,jobs);
    int total=B*N; gate_up_reduce<scalar_t><<<(total+255)/256,256,0,stream>>>(pg.data_ptr<float>(),pu.data_ptr<float>(),reinterpret_cast<scalar_t*>(out.data_ptr()),total,split);
}
}

torch::Tensor fused_gate_up_cuda(torch::Tensor x,torch::Tensor wg,torch::Tensor sg,torch::Tensor zg,torch::Tensor wu,torch::Tensor su,torch::Tensor zu,int64_t bits,int64_t in_features,int64_t group_size,int64_t split_k,int64_t rho){
    TORCH_CHECK(wg.sizes()==wu.sizes(),"gate/up packed shapes must match");
    TORCH_CHECK(sg.sizes()==su.sizes() && zg.sizes()==zu.sizes(),"gate/up metadata shapes must match");
    int B=x.size(0),N=wg.size(0); auto fopt=x.options().dtype(torch::kFloat32);
    auto pg=torch::empty({split_k,B,N},fopt), pu=torch::empty({split_k,B,N},fopt), ctr=torch::zeros({1},x.options().dtype(torch::kInt32)), out=torch::empty({B,N},x.options());
    int dev=x.get_device(); cudaDeviceProp prop; cudaGetDeviceProperties(&prop,dev);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,x.scalar_type(),"crispybits_gate_up",[&]{
        if(bits==2)launch<scalar_t,2>(x,wg,sg,zg,wu,su,zu,pg,pu,ctr,out,in_features,group_size,split_k,rho,prop.multiProcessorCount);
        else if(bits==3)launch<scalar_t,3>(x,wg,sg,zg,wu,su,zu,pg,pu,ctr,out,in_features,group_size,split_k,rho,prop.multiProcessorCount);
        else if(bits==4)launch<scalar_t,4>(x,wg,sg,zg,wu,su,zu,pg,pu,ctr,out,in_features,group_size,split_k,rho,prop.multiProcessorCount);
        else TORCH_CHECK(false,"bits must be 2/3/4");
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
