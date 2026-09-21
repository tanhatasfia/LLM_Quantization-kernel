#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include "bitpack.cuh"

namespace {
constexpr int THREADS=128,BN=128,TK=256;

template <typename scalar_t,int BITS>
__device__ __forceinline__ void accum_one(float &acc,const scalar_t* sx,int kb,int chunk,const uint32_t* row,const scalar_t* srow,const scalar_t* zrow,int G){
    if(!row)return;
    for(int j=0;j<chunk;++j){int k=kb+j,g=k/G;uint32_t q=cb_unpack(row,k,BITS);float w=(float(q)-cb_to_float(zrow[g]))*cb_to_float(srow[g]);acc=fmaf(cb_to_float(sx[j]),w,acc);}
}

template <typename scalar_t,int BITS>
__global__ void qkv_kernel(
 const scalar_t* x,
 const uint32_t* wq,const scalar_t* sq,const scalar_t* zq,int Nq,int Wq,int Gq,
 const uint32_t* wk,const scalar_t* sk,const scalar_t* zk,int Nk,int Wk,int Gk,
 const uint32_t* wv,const scalar_t* sv,const scalar_t* zv,int Nv,int Wv,int Gv,
 float* pq,float* pk,float* pv,int* counter,
 int B,int K,int group_size,int split_k,int ntiles,int jobs){
    extern __shared__ unsigned char raw[]; scalar_t* sx=reinterpret_cast<scalar_t*>(raw);
    int Nmax=max(Nq,max(Nk,Nv));
    while(true){
      int job=atomicAdd(counter,1);if(job>=jobs)break;int t=job,split=t%split_k;t/=split_k;int tile=t%ntiles;t/=ntiles;int batch=t;
      int n=tile*BN+threadIdx.x;int k0=(K*split)/split_k,k1=(K*(split+1))/split_k;float aq=0.f,ak=0.f,av=0.f;
      const uint32_t* rq=n<Nq?wq+(size_t)n*Wq:nullptr;const scalar_t* srq=n<Nq?sq+(size_t)n*Gq:nullptr;const scalar_t* zrq=n<Nq?zq+(size_t)n*Gq:nullptr;
      const uint32_t* rk=n<Nk?wk+(size_t)n*Wk:nullptr;const scalar_t* srk=n<Nk?sk+(size_t)n*Gk:nullptr;const scalar_t* zrk=n<Nk?zk+(size_t)n*Gk:nullptr;
      const uint32_t* rv=n<Nv?wv+(size_t)n*Wv:nullptr;const scalar_t* srv=n<Nv?sv+(size_t)n*Gv:nullptr;const scalar_t* zrv=n<Nv?zv+(size_t)n*Gv:nullptr;
      for(int kb=k0;kb<k1;kb+=TK){int chunk=min(TK,k1-kb);for(int j=threadIdx.x;j<chunk;j+=blockDim.x)sx[j]=x[(size_t)batch*K+kb+j];__syncthreads();
        if(n<Nmax){accum_one<scalar_t,BITS>(aq,sx,kb,chunk,rq,srq,zrq,group_size);accum_one<scalar_t,BITS>(ak,sx,kb,chunk,rk,srk,zrk,group_size);accum_one<scalar_t,BITS>(av,sx,kb,chunk,rv,srv,zrv,group_size);}__syncthreads();}
      if(n<Nq)pq[((size_t)split*B+batch)*Nq+n]=aq;if(n<Nk)pk[((size_t)split*B+batch)*Nk+n]=ak;if(n<Nv)pv[((size_t)split*B+batch)*Nv+n]=av;__syncthreads();
    }
}

template <typename scalar_t>
__global__ void reduce(const float* p,const scalar_t* bias,scalar_t* out,int total,int N,int split,bool has_bias){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;float a=0.f;for(int s=0;s<split;++s)a+=p[(size_t)s*total+i];if(has_bias)a+=cb_to_float(bias[i%N]);out[i]=static_cast<scalar_t>(a);}

template <typename scalar_t,int BITS>
void launch(torch::Tensor x,torch::Tensor wq,torch::Tensor sq,torch::Tensor zq,torch::Tensor bq,torch::Tensor wk,torch::Tensor sk,torch::Tensor zk,torch::Tensor bk,torch::Tensor wv,torch::Tensor sv,torch::Tensor zv,torch::Tensor bv,torch::Tensor pq,torch::Tensor pk,torch::Tensor pv,torch::Tensor ctr,torch::Tensor oq,torch::Tensor ok,torch::Tensor ov,int K,int G,int split,int rho,int sms){
 int B=x.size(0),Nq=wq.size(0),Nk=wk.size(0),Nv=wv.size(0),Nmax=std::max(Nq,std::max(Nk,Nv)),nt=(Nmax+BN-1)/BN,jobs=B*nt*split,blocks=std::max(1,std::min(jobs,std::max(1,rho)*sms));cudaStream_t stream=at::cuda::getDefaultCUDAStream();
 qkv_kernel<scalar_t,BITS><<<blocks,THREADS,TK*sizeof(scalar_t),stream>>>(reinterpret_cast<const scalar_t*>(x.data_ptr()),
 reinterpret_cast<const uint32_t*>(wq.data_ptr<int32_t>()),reinterpret_cast<const scalar_t*>(sq.data_ptr()),reinterpret_cast<const scalar_t*>(zq.data_ptr()),Nq,wq.size(1),sq.size(1),
 reinterpret_cast<const uint32_t*>(wk.data_ptr<int32_t>()),reinterpret_cast<const scalar_t*>(sk.data_ptr()),reinterpret_cast<const scalar_t*>(zk.data_ptr()),Nk,wk.size(1),sk.size(1),
 reinterpret_cast<const uint32_t*>(wv.data_ptr<int32_t>()),reinterpret_cast<const scalar_t*>(sv.data_ptr()),reinterpret_cast<const scalar_t*>(zv.data_ptr()),Nv,wv.size(1),sv.size(1),
 pq.data_ptr<float>(),pk.data_ptr<float>(),pv.data_ptr<float>(),ctr.data_ptr<int>(),B,K,G,split,nt,jobs);
 auto red=[&](torch::Tensor p,torch::Tensor bias,torch::Tensor out,int N){int total=B*N;bool hb=bias.defined()&&bias.numel()==N;reduce<scalar_t><<<(total+255)/256,256,0,stream>>>(p.data_ptr<float>(),hb?reinterpret_cast<const scalar_t*>(bias.data_ptr()):nullptr,reinterpret_cast<scalar_t*>(out.data_ptr()),total,N,split,hb);};red(pq,bq,oq,Nq);red(pk,bk,ok,Nk);red(pv,bv,ov,Nv);
}
}

std::vector<torch::Tensor> fused_qkv_cuda(torch::Tensor x,
 torch::Tensor wq,torch::Tensor sq,torch::Tensor zq,torch::Tensor bq,
 torch::Tensor wk,torch::Tensor sk,torch::Tensor zk,torch::Tensor bk,
 torch::Tensor wv,torch::Tensor sv,torch::Tensor zv,torch::Tensor bv,
 int64_t bits,int64_t in_features,int64_t group_size,int64_t split_k,int64_t rho){
 int B=x.size(0),Nq=wq.size(0),Nk=wk.size(0),Nv=wv.size(0);auto f=x.options().dtype(torch::kFloat32);auto pq=torch::empty({split_k,B,Nq},f),pk=torch::empty({split_k,B,Nk},f),pv=torch::empty({split_k,B,Nv},f),ctr=torch::zeros({1},x.options().dtype(torch::kInt32));auto oq=torch::empty({B,Nq},x.options()),ok=torch::empty({B,Nk},x.options()),ov=torch::empty({B,Nv},x.options());cudaDeviceProp prop;cudaGetDeviceProperties(&prop,x.get_device());
 AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,x.scalar_type(),"crispybits_qkv",[&]{if(bits==2)launch<scalar_t,2>(x,wq,sq,zq,bq,wk,sk,zk,bk,wv,sv,zv,bv,pq,pk,pv,ctr,oq,ok,ov,in_features,group_size,split_k,rho,prop.multiProcessorCount);else if(bits==3)launch<scalar_t,3>(x,wq,sq,zq,bq,wk,sk,zk,bk,wv,sv,zv,bv,pq,pk,pv,ctr,oq,ok,ov,in_features,group_size,split_k,rho,prop.multiProcessorCount);else if(bits==4)launch<scalar_t,4>(x,wq,sq,zq,bq,wk,sk,zk,bk,wv,sv,zv,bv,pq,pk,pv,ctr,oq,ok,ov,in_features,group_size,split_k,rho,prop.multiProcessorCount);else TORCH_CHECK(false,"bits must be 2/3/4");});C10_CUDA_KERNEL_LAUNCH_CHECK();return {oq,ok,ov};
}
