// 128x128 block quantization with per-K-block scaling and CUTLASS warp MMA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/block/block_reduce.cuh>
#include <cutlass/cutlass.h>
#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
namespace {
using Tensor = torch::Tensor;
constexpr int Q=128, Consumers=256, NT=Consumers, Stride=133;
constexpr int StageBytes=2*Q*Q, SharedBytes=Q*Stride*4;
struct MaxOp { __device__ float operator()(float a,float b) const { return fmaxf(a,b); } };
template<bool FP8,int WarpM=32,int WarpN=64> struct Types {
 using Element=typename std::conditional<FP8,cutlass::float_e4m3_t,int8_t>::type;
 using Acc=typename std::conditional<FP8,float,int32_t>::type;
 using Arch=typename std::conditional<FP8,cutlass::arch::Sm89,cutlass::arch::Sm80>::type;
 using Op=typename std::conditional<FP8,cutlass::arch::OpMultiplyAdd,cutlass::arch::OpMultiplyAddSaturate>::type;
 using Shape=cutlass::gemm::GemmShape<Q,Q,Q>;
 using Warp=cutlass::gemm::GemmShape<WarpM,WarpN,Q>;
 static_assert((Q/WarpM)*(Q/WarpN)*32==Consumers,"BQ requires eight consumer warps");
 using Instruction=cutlass::gemm::GemmShape<16,8,32>;
 using Default=cutlass::gemm::kernel::DefaultGemm<Element,cutlass::layout::RowMajor,16,
  Element,cutlass::layout::ColumnMajor,16,Acc,cutlass::layout::RowMajor,Acc,
  cutlass::arch::OpClassTensorOp,Arch,Shape,Warp,Instruction,
  cutlass::epilogue::thread::LinearCombination<Acc,4,Acc,Acc>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2,false,Op>;
 using Mma=typename Default::GemmKernel::Mma::Operator;
};
template<typename E> __device__ E quantize(float v,float multiplier) {
 float x=v*multiplier;
 if constexpr(std::is_same<E,int8_t>::value) return int8_t(max(-127,min(127,__float2int_rz(x))));
 else return E(fmaxf(-448.f,x)); // FP8 conversion already saturates the upper bound.
}
// Convert and store adjacent quantized values together. FP8 uses one packed
// conversion instruction instead of two scalar conversions.
template<typename E> __device__ uint16_t quantize_pair(float a,float b,float multiplier) {
 if constexpr(std::is_same<E,int8_t>::value) {
  return uint16_t(uint8_t(quantize<E>(a,multiplier))) |
         (uint16_t(uint8_t(quantize<E>(b,multiplier)))<<8);
 } else {
  cutlass::Array<float,2> values;
  values[0]=fmaxf(-448.f,a*multiplier);
  values[1]=fmaxf(-448.f,b*multiplier);
  auto pair=cutlass::NumericArrayConverter<E,float,2>()(values);
  uint16_t packed;memcpy(&packed,&pair,sizeof(packed));return packed;
 }
}
template<typename Input,bool FP8> __global__ void quant_kernel(Input const* input,float const* norm,
 typename Types<FP8>::Element* out,typename Types<FP8>::Element* out_t,float* scale,
 int rows,int cols,int m,int n,bool transpose,float eps) {
 using E=typename Types<FP8>::Element;
 using Reduce=cub::BlockReduce<float,256>;
 __shared__ typename Reduce::TempStorage reduction;
 __shared__ __align__(16) unsigned char quantized[Q][Q+4];
 int tm=blockIdx.x,tn=blockIdx.y,batch=blockIdx.z;
 float inv=1.f/fmaxf(norm[batch],eps);
 if(norm[batch]==0) inv=0;
 float values[64],maximum=1e-8f;
 #pragma unroll
 for(int j=0;j<32;++j) {
  #pragma unroll
  for(int e=0;e<2;++e) {
   int idx=2*(threadIdx.x+j*256)+e,r=tm*Q+idx/Q,c=tn*Q+idx%Q;
   int ir=transpose?c:r,ic=transpose?r:c;
   float v=(ir<rows&&ic<cols)?float(input[(int64_t(batch)*rows+ir)*cols+ic])*inv:0;
   values[2*j+e]=v;maximum=fmaxf(maximum,fabsf(v));
  }
 }
 float mx=Reduce(reduction).Reduce(maximum,MaxOp());
 __shared__ float multiplier;
 if(threadIdx.x==0) {
  multiplier=mx>0?(FP8?448.f:127.f)/mx:0;
  scale[(int64_t(batch)*(m/Q)+tm)*(n/Q)+tn]=mx>0?1.f/multiplier:0;
 }
 __syncthreads();
 #pragma unroll
 for(int j=0;j<32;++j) {
  int idx=2*(threadIdx.x+j*256),r=tm*Q+idx/Q,c=tn*Q+idx%Q;
  uint16_t value=quantize_pair<E>(values[2*j],values[2*j+1],multiplier);
  *reinterpret_cast<uint16_t*>(out+(int64_t(batch)*m+r)*n+c)=value;
  *reinterpret_cast<uint16_t*>(&quantized[idx/Q][idx%Q])=value;
 }
 __syncthreads();
 #pragma unroll
 for(int j=0;j<64;++j) {
  int idx=threadIdx.x+j*256,r=idx/Q,c=idx%Q;
  reinterpret_cast<unsigned char*>(out_t)[(int64_t(batch)*n+(tn*Q+r))*m+(tm*Q+c)]=quantized[c][r];
 }
}
// Mode 0: Gram. Mode 1: quadratic, separate diagonal. Mode 2: update.
template<bool FP8,int Mode,bool Final=false,typename Output=float,bool UseTma=false,int WarpM=32,int WarpN=64> __global__ __launch_bounds__(NT,1) void gemm_kernel(
 typename Types<FP8,WarpM,WarpN>::Element const* a,float const* sa,float const* da,
 typename Types<FP8,WarpM,WarpN>::Element const* b,float const* sb,typename Types<FP8,WarpM,WarpN>::Element const* row_b,
 void* output,void* output_t,float* so,float* dout,int m,int n,int k,
 int logical_m,int logical_n,float ca,float cb,float cc,bool transpose,
 __grid_constant__ const CUtensorMap map_a,__grid_constant__ const CUtensorMap map_b) {
 using T=Types<FP8,WarpM,WarpN>;using E=typename T::Element;using Acc=typename T::Acc;using Mma=typename T::Mma;
 using Reduce=cub::BlockReduce<float,NT>;using Barrier=cutlass::arch::ClusterTransactionBarrier;
 union Control {uint64_t barriers[4];typename Reduce::TempStorage reduction;};
 __shared__ __align__(16) Control control;
 __shared__ float multiplier;
 extern __shared__ __align__(1024) unsigned char raw[];
 int tm,tn,batch=blockIdx.y;
 if constexpr(Mode!=2) {
  int pid=blockIdx.x;tm=int((sqrtf(float(8*pid+1))-1)*.5f);
  while((tm+1)*(tm+2)/2<=pid)++tm;
  while(tm*(tm+1)/2>pid)--tm;
  tn=pid-tm*(tm+1)/2;
 } else {tm=blockIdx.x/(n/Q);tn=blockIdx.x%(n/Q);}
 int row0=tm*Q,col0=tn*Q,warp=threadIdx.x/32,lane=threadIdx.x%32;
 constexpr int Count=Mma::FragmentC::kElements;
 cutlass::Array<float,Count> total;total.clear();
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
 if constexpr(UseTma) {
 if(threadIdx.x==0) {
  Barrier::init(control.barriers,1);Barrier::init(control.barriers+1,1);
  Barrier::init(control.barriers+2,Consumers/32);Barrier::init(control.barriers+3,Consumers/32);
 }
 cutlass::arch::fence_barrier_init();
 }
 #endif
 __syncthreads();
 int iterations=k/Q;
 auto issue=[&](int it) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr(UseTma) {
  if(threadIdx.x==0) {
   int slot=it&1;auto bar=control.barriers+slot;
   E* ap=reinterpret_cast<E*>(raw)+slot*StageBytes;E* bp=ap+Q*Q;
   Barrier::arrive_and_expect_tx(bar,StageBytes);
   uint64_t hint=uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL);
   cute::SM90_TMA_LOAD_3D::copy(&map_a,bar,hint,ap,it*Q,row0,batch);
   cute::SM90_TMA_LOAD_3D::copy(&map_b,bar,hint,bp,it*Q,col0,batch);
  }
  } else
  #endif
  {
   int slot=it&1;
   E* ap=reinterpret_cast<E*>(raw)+slot*StageBytes;E* bp=ap+Q*Q;
   auto la=Mma::LayoutA::packed({Q,Q});auto lb=Mma::LayoutB::packed({Q,Q});
   for(int idx=threadIdx.x*16;idx<Q*Q;idx+=NT*16) {
    int r=idx/Q,c=idx%Q;
    cutlass::arch::cp_async<16>(ap+la({r,c}),a+(int64_t(batch)*m+row0+r)*k+it*Q+c);
    cutlass::arch::cp_async<16>(bp+lb({c,r}),b+(int64_t(batch)*n+col0+r)*k+it*Q+c);
   }
   cutlass::arch::cp_async_fence();
  }
 };
 issue(0);
 {
  Mma op;
  for(int it=0;it<iterations;++it) {
   if(it+1<iterations)issue(it+1);
   int slot=it&1;
   #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
   if constexpr(UseTma) Barrier::wait(control.barriers+slot,(it/2)&1);
   else
   #endif
   {
    if(it+1<iterations)cutlass::arch::cp_async_wait<1>();
    else cutlass::arch::cp_async_wait<0>();
    __syncthreads();
   }
   E* ap=reinterpret_cast<E*>(raw)+slot*StageBytes;E* bp=ap+Q*Q;
   typename Mma::IteratorA ia({ap,Mma::LayoutA::packed({Q,Q})},lane);
   typename Mma::IteratorB ib({bp,Mma::LayoutB::packed({Q,Q})},lane);
   ia.add_tile_offset({warp%(Q/WarpM),0});ib.add_tile_offset({0,warp/(Q/WarpM)});
   typename Mma::FragmentC accum;accum.clear();
   typename Mma::FragmentA fa[2];typename Mma::FragmentB fb[2];
   ia.set_kgroup_index(0);ib.set_kgroup_index(0);ia.load(fa[0]);ib.load(fb[0]);++ia;++ib;
   #pragma unroll
   for(int group=0;group<4;++group) {
    if(group+1<4) {ia.set_kgroup_index(group+1);ib.set_kgroup_index(group+1);
     ia.load(fa[(group+1)&1]);ib.load(fb[(group+1)&1]);++ia;++ib;}
    typename Mma::TransformedFragmentA ta;typename Mma::TransformedFragmentB tb;
    op.transform(ta,tb,fa[group&1],fb[group&1]);op(accum,ta,tb,accum);
   }
   float bs;
   if constexpr(Mode==2) bs=sb[(int64_t(batch)*(k/Q)+it)*(n/Q)+tn];
   else bs=sb[(int64_t(batch)*(n/Q)+tn)*(k/Q)+it];
   float scale=sa[(int64_t(batch)*(m/Q)+tm)*(k/Q)+it]*bs;
   #pragma unroll
   for(int i=0;i<Count;++i)total[i]=__fmaf_rn(float(accum[i]),scale,total[i]);
   __syncthreads();
  }
 }
 __syncthreads();
 float* tile=reinterpret_cast<float*>(raw);float mx=1e-8f;
 if(threadIdx.x<Consumers) {
  using FI=cutlass::epilogue::warp::FragmentIteratorTensorOp<typename T::Warp,typename T::Instruction,float,
   cutlass::Array<float,4>,cutlass::layout::RowMajor>;
  FI fi(total);
  #pragma unroll
  for(int it=0;it<WarpM/8;++it) {
   typename FI::Fragment frag;fi.load(frag);++fi;
   int r=(warp%(Q/WarpM))*WarpM+it*8+lane/4,row=row0+r;
   #pragma unroll
   for(int j=0;j<WarpN/8;++j) {
    int c=(warp/(Q/WarpM))*WarpN+j*8+(lane%4)*2,col=col0+c;
    float results[2];
    #pragma unroll
    for(int e=0;e<2;++e) {
     float value=frag[j*2+e];int column=col+e;
     if constexpr(Mode==1) {
      float v=float(a[(int64_t(batch)*m+row)*k+column])*sa[(int64_t(batch)*(m/Q)+tm)*(k/Q)+tn];
      float ai=da[int64_t(batch)*m+row],aj=da[int64_t(batch)*m+column];
      value+=(ai+aj)*v;value*=cc;value+=v*cb;
      if(row==column) {dout[int64_t(batch)*m+row]=row<logical_m?value+(ca+ai*(cb+ai*cc)):0;value=0;}
     } else if constexpr(Mode==0) {
      if(row==column){dout[int64_t(batch)*m+row]=row<logical_m?value:0;value=0;}
     } else {
      float v=float(row_b[(int64_t(batch)*m+row)*n+column]);
      value+=(v*da[int64_t(batch)*m+row])*sb[(int64_t(batch)*(k/Q)+tm)*(n/Q)+tn];
     }
     // Padding never contributes to a block scale.
     if(row>=logical_m||column>=logical_n)value=0;
     results[e]=value;
     if constexpr(!Final) {
      tile[r*Stride+c+e]=value;mx=fmaxf(mx,fabsf(value));
     } else if(transpose)tile[r*Stride+c+e]=value;
    }
    if constexpr(Final) {
     if(!transpose&&row<logical_m&&col<logical_n) {
      auto target=static_cast<Output*>(output)+(int64_t(batch)*logical_m+row)*logical_n+col;
      if constexpr(std::is_same<Output,float>::value) {
       if(logical_n%2==0&&col+1<logical_n)
        *reinterpret_cast<float2*>(target)=make_float2(results[0],results[1]);
       else {
        target[0]=Output(results[0]);
        if(col+1<logical_n)target[1]=Output(results[1]);
       }
      } else {
       target[0]=Output(results[0]);
       if(col+1<logical_n)target[1]=Output(results[1]);
      }
     }
    }
   }
  }
 }
 __syncthreads();
 if constexpr(Final)if(!transpose)return;
 if constexpr(!Final) {
  float maximum=Reduce(control.reduction).Reduce(mx,MaxOp());
  if(threadIdx.x==0) {
   multiplier=maximum>0?(FP8?448.f:127.f)/maximum:0;
   float scale=maximum>0?1.f/multiplier:0;
   so[(int64_t(batch)*(m/Q)+tm)*(n/Q)+tn]=scale;
   if constexpr(Mode!=2)so[(int64_t(batch)*(m/Q)+tn)*(n/Q)+tm]=scale;
  }
  __syncthreads();
 }
 for(int idx=threadIdx.x*2;idx<Q*Q;idx+=NT*2) {
  int r=idx/Q,c=idx%Q,row=row0+r,col=col0+c;
  if constexpr(Final) {
   if(!transpose&&row<logical_m&&col<logical_n) {
    auto target=static_cast<Output*>(output)+(int64_t(batch)*logical_m+row)*logical_n+col;
    if constexpr(std::is_same<Output,float>::value) {
     if(logical_n%2==0&&col+1<logical_n)
      *reinterpret_cast<float2*>(target)=make_float2(tile[r*Stride+c],tile[r*Stride+c+1]);
     else {
      target[0]=Output(tile[r*Stride+c]);
      if(col+1<logical_n)target[1]=Output(tile[r*Stride+c+1]);
     }
    } else {
     target[0]=Output(tile[r*Stride+c]);
     if(col+1<logical_n)target[1]=Output(tile[r*Stride+c+1]);
    }
   }
   if(transpose&&col0+r<logical_n) {
    #pragma unroll
    for(int e=0;e<2;++e)if(row0+c+e<logical_m)
     static_cast<Output*>(output)[(int64_t(batch)*logical_n+(col0+r))*logical_m+(row0+c+e)]=Output(tile[(c+e)*Stride+r]);
   }
  } else {
   float v0=(Mode!=2&&tm==tn&&r<c)?tile[c*Stride+r]:tile[r*Stride+c];
   float v1=(Mode!=2&&tm==tn&&r<c+1)?tile[(c+1)*Stride+r]:tile[r*Stride+c+1];
   *reinterpret_cast<uint16_t*>(static_cast<E*>(output)+(int64_t(batch)*m+row)*n+col)=quantize_pair<E>(v0,v1,multiplier);
   if constexpr(Mode!=2) {
    if(tm!=tn)*reinterpret_cast<uint16_t*>(static_cast<E*>(output)+(int64_t(batch)*m+(col0+r))*n+(row0+c))=
     quantize_pair<E>(tile[c*Stride+r],tile[(c+1)*Stride+r],multiplier);
   } else {
    *reinterpret_cast<uint16_t*>(static_cast<E*>(output_t)+(int64_t(batch)*n+(col0+r))*m+(row0+c))=
     quantize_pair<E>(tile[c*Stride+r],tile[(c+1)*Stride+r],multiplier);
   }
  }
 }
}
void quant(Tensor input,Tensor norm,Tensor out,Tensor out_t,Tensor scale,bool transpose,double eps) {
 c10::cuda::CUDAGuard guard(input.device());
 bool fp8=out.scalar_type()==at::kFloat8_e4m3fn;
 dim3 grid(out.size(1)/Q,out.size(2)/Q,out.size(0));
 auto stream=at::cuda::getCurrentCUDAStream();
 AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,input.scalar_type(),"bq_quant",[&]{
  if(fp8)quant_kernel<scalar_t,true><<<grid,256,0,stream>>>(input.data_ptr<scalar_t>(),norm.data_ptr<float>(),
    reinterpret_cast<cutlass::float_e4m3_t*>(out.data_ptr()),reinterpret_cast<cutlass::float_e4m3_t*>(out_t.data_ptr()),
    scale.data_ptr<float>(),input.size(1),input.size(2),out.size(1),out.size(2),transpose,eps);
  else quant_kernel<scalar_t,false><<<grid,256,0,stream>>>(input.data_ptr<scalar_t>(),norm.data_ptr<float>(),
    out.data_ptr<int8_t>(),out_t.data_ptr<int8_t>(),scale.data_ptr<float>(),input.size(1),input.size(2),out.size(1),out.size(2),transpose,eps);
 });C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<bool FP8,int Mode,bool Final=false,typename Output=float,bool UseTma=false,int WarpM=32,int WarpN=64> void launch(Tensor a,Tensor sa,Tensor da,
 Tensor b,Tensor sb,Tensor rb,Tensor out,Tensor ot,Tensor so,Tensor dout,int lm,int ln,float ca,float cb,float cc,bool transpose) {
 CUtensorMap ma{},mb{};int m=a.size(1),k=a.size(2),n=b.size(1);
 auto encode=[&](CUtensorMap* map,Tensor tensor,int height){
  uint64_t dims[3]={uint64_t(k),uint64_t(height),uint64_t(a.size(0))},strides[2]={uint64_t(k),uint64_t(k)*height};
  uint32_t box[3]={Q,Q,1},element[3]={1,1,1};
  TORCH_CHECK(cuTensorMapEncodeTiled(map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,tensor.data_ptr(),dims,strides,box,element,
   CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS,"Cannot encode TMA tensor map");
 };
 if constexpr(UseTma) {
  encode(&ma,a,m);encode(&mb,b,n);
 }
 auto kernel=gemm_kernel<FP8,Mode,Final,Output,UseTma,WarpM,WarpN>;

 C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
 int blocks=Mode==2?(m/Q)*(n/Q):(m/Q)*(m/Q+1)/2;
 kernel<<<dim3(blocks,a.size(0)),NT,SharedBytes,at::cuda::getCurrentCUDAStream()>>>(
  reinterpret_cast<typename Types<FP8,WarpM,WarpN>::Element const*>(a.data_ptr()),sa.data_ptr<float>(),da.numel()?da.data_ptr<float>():nullptr,
  reinterpret_cast<typename Types<FP8,WarpM,WarpN>::Element const*>(b.data_ptr()),sb.data_ptr<float>(),
  rb.numel()?reinterpret_cast<typename Types<FP8,WarpM,WarpN>::Element const*>(rb.data_ptr()):nullptr,
  out.data_ptr(),ot.numel()?ot.data_ptr():nullptr,so.numel()?so.data_ptr<float>():nullptr,dout.numel()?dout.data_ptr<float>():nullptr,
  m,n,k,lm,ln,ca,cb,cc,transpose,ma,mb);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<bool FP8,bool UseTma,int WarpM=32,int WarpN=64> void dispatch(Tensor a,Tensor sa,Tensor da,Tensor b,Tensor sb,Tensor rb,Tensor out,Tensor ot,
 Tensor so,Tensor dout,int mode,int lm,int ln,float ca,float cb,float cc,bool final,bool transpose) {
 if(mode==0)launch<FP8,0,false,float,UseTma,WarpM,WarpN>(a,sa,da,b,sb,rb,out,ot,so,dout,lm,ln,ca,cb,cc,transpose);
 else if(mode==1)launch<FP8,1,false,float,UseTma,WarpM,WarpN>(a,sa,da,b,sb,rb,out,ot,so,dout,lm,ln,ca,cb,cc,transpose);
 else if(!final)launch<FP8,2,false,float,UseTma,WarpM,WarpN>(a,sa,da,b,sb,rb,out,ot,so,dout,lm,ln,ca,cb,cc,transpose);
 else if(out.scalar_type()==at::kFloat)launch<FP8,2,true,float,UseTma,WarpM,WarpN>(a,sa,da,b,sb,rb,out,ot,so,dout,lm,ln,ca,cb,cc,transpose);
 else if(out.scalar_type()==at::kHalf)launch<FP8,2,true,at::Half,UseTma,WarpM,WarpN>(a,sa,da,b,sb,rb,out,ot,so,dout,lm,ln,ca,cb,cc,transpose);
 else if(out.scalar_type()==at::kBFloat16)launch<FP8,2,true,at::BFloat16,UseTma,WarpM,WarpN>(a,sa,da,b,sb,rb,out,ot,so,dout,lm,ln,ca,cb,cc,transpose);
 else TORCH_CHECK(false,"Unsupported output dtype");
}
void gemm(Tensor a,Tensor sa,Tensor da,Tensor b,Tensor sb,Tensor rb,Tensor out,Tensor ot,Tensor so,Tensor dout,
 int mode,int lm,int ln,double ca,double cb,double cc,bool final,bool transpose,int tile) {
 c10::cuda::CUDAGuard guard(a.device());
 #define DISPATCH(F,T,M,N) dispatch<F,T,M,N>(a,sa,da,b,sb,rb,out,ot,so,dout,mode,lm,ln,ca,cb,cc,final,transpose)
 if(a.scalar_type()==at::kFloat8_e4m3fn) {if(tile==1)DISPATCH(true,true,32,64);else if(tile==2)DISPATCH(true,false,64,32);else DISPATCH(true,false,32,64);}
 else {if(tile==1)DISPATCH(false,true,32,64);else if(tile==2)DISPATCH(false,false,64,32);else DISPATCH(false,false,32,64);}
 #undef DISPATCH
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("quant",&quant);m.def("gemm",&gemm,pybind11::arg("a"),pybind11::arg("sa"),pybind11::arg("da"),
 pybind11::arg("b"),pybind11::arg("sb"),pybind11::arg("rb"),pybind11::arg("out"),pybind11::arg("ot"),
 pybind11::arg("so"),pybind11::arg("dout"),pybind11::arg("mode"),pybind11::arg("lm"),pybind11::arg("ln"),
 pybind11::arg("ca"),pybind11::arg("cb"),pybind11::arg("cc"),pybind11::arg("final"),pybind11::arg("transpose"),pybind11::arg("tile")=0);}
