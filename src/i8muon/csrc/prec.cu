// Floating-point Newton-Schulz: CUTLASS cp.async mainloops and fused epilogues.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace {
using Tensor = torch::Tensor;

template <typename E, int BM, int BN, int WM, int WN>
struct Traits {
  static constexpr bool Simt = std::is_same<E, float>::value;
  static constexpr int BK = Simt ? 16 : (BN == 128 ? 32 : 64);
  static constexpr int Alignment = Simt ? 1 : 16 / sizeof(E);
  using Row = cutlass::layout::RowMajor;
  using Col = cutlass::layout::ColumnMajor;
  using Shape = cutlass::gemm::GemmShape<BM, BN, BK>;
  using Warp = cutlass::gemm::GemmShape<WM, WN, BK>;
  using Instruction = typename std::conditional<Simt,
      cutlass::gemm::GemmShape<1, 1, 1>, cutlass::gemm::GemmShape<16, 8, 16>>::type;
  using Class = typename std::conditional<Simt, cutlass::arch::OpClassSimt,
      cutlass::arch::OpClassTensorOp>::type;
  using Default = cutlass::gemm::kernel::DefaultGemm<E, Row, Alignment, E, Col, Alignment,
      float, Row, float, Class, cutlass::arch::Sm80, Shape, Warp, Instruction,
      cutlass::epilogue::thread::LinearCombination<float, Simt ? 1 : 4, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, false,
      cutlass::arch::OpMultiplyAdd>;
  using Mma = typename Default::GemmKernel::Mma;
  using Epilogue = typename Default::GemmKernel::Epilogue;
  using FI = typename Epilogue::AccumulatorFragmentIterator;
  using WI = typename Epilogue::WarpTileIterator;
  static constexpr int Stride = BN + WI::Padding::kColumn + 4;
  static constexpr int Threads = BM / WM * (BN / WN) * 32;
  union Storage {
    typename Mma::SharedStorage mma;
    float tile[BM * Stride];
  };
};

template <typename Input, typename E>
__global__ void normalize_kernel(Input const* input, float const* norm, E* out, E* out_t,
    int rows, int cols, int m, int n, bool transpose, float eps) {
  __shared__ E tile[32][33];
  int batch = blockIdx.z;
  float inv = norm[batch] == 0 ? 0 : 1.f / fmaxf(norm[batch], eps);
  #pragma unroll
  for (int j = 0; j < 32; j += 8) {
    int r = threadIdx.y + j, c = threadIdx.x;
    int row = blockIdx.y * 32 + r, col = blockIdx.x * 32 + c;
    int ir = transpose ? col : row, ic = transpose ? row : col;
    E value = E(ir < rows && ic < cols ? float(input[(int64_t(batch) * rows + ir) * cols + ic]) * inv : 0);
    tile[r][c] = value;
    if (row < m && col < n) out[(int64_t(batch) * m + row) * n + col] = value;
  }
  __syncthreads();
  #pragma unroll
  for (int j = 0; j < 32; j += 8) {
    int row = blockIdx.x * 32 + threadIdx.y + j, col = blockIdx.y * 32 + threadIdx.x;
    if (row < n && col < m)
      out_t[(int64_t(batch) * n + row) * m + col] = tile[threadIdx.x][threadIdx.y + j];
  }
}

// Right operand is stored as rows of B^T. Symmetric operations preserve the
// existing 64-row diagonal-block convention, including AB^T for Gram updates.
template <typename E, int BM, int BN, int WM, int WN, bool Quad, bool Final, typename Output>
__global__ void gemm_kernel(E const* a, E const* bt, void* output, E* output_t,
    int m, int n, int k, int lm, int ln, bool symmetric, bool transpose,
    float ca, float cb, float cc,
    typename Traits<E, BM, BN, WM, WN>::Mma::IteratorA::Params pa,
    typename Traits<E, BM, BN, WM, WN>::Mma::IteratorB::Params pb) {
  using T = Traits<E, BM, BN, WM, WN>;
  using Row = cutlass::layout::RowMajor;
  constexpr int R = BM / BN, WarpsM = BM / WM;
  extern __shared__ __align__(128) unsigned char raw[];
  auto& shared = *reinterpret_cast<typename T::Storage*>(raw);
  int tm, tn;
  if (symmetric) {
    int base = blockIdx.x / R;
    tm = int((sqrtf(float(8 * base + 1)) - 1) * .5f);
    while ((tm + 1) * (tm + 2) / 2 <= base) ++tm;
    while (tm * (tm + 1) / 2 > base) --tm;
    tn = (base - tm * (tm + 1) / 2) * R + blockIdx.x % R;
  } else {
    tm = blockIdx.x / ((n + BN - 1) / BN);
    tn = blockIdx.x % ((n + BN - 1) / BN);
  }
  int row0 = tm * BM, col0 = tn * BN, batch = blockIdx.y;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  typename T::Mma::IteratorA ia(pa, const_cast<E*>(a) + int64_t(batch) * m * k,
                              {m, k}, threadIdx.x, {row0, 0});
  typename T::Mma::IteratorB ib(pb, const_cast<E*>(bt) + int64_t(batch) * n * k,
                              {k, n}, threadIdx.x, {0, col0});
  typename T::Mma::FragmentC accum;
  accum.clear();
  typename T::Mma mma(shared.mma, threadIdx.x, warp, lane);
  mma((k + T::BK - 1) / T::BK, accum, ia, ib, accum);
  __syncthreads();
  if constexpr (T::Simt) {
    typename T::Mma::Operator::IteratorC ci({shared.tile, Row(T::Stride)}, lane);
    ci.add_pointer_offset((warp % WarpsM) * WM * T::Stride + (warp / WarpsM) * WN);
    ci.store(accum);
  } else {
  typename T::FI fi(accum);
  typename T::WI wi({shared.tile, Row(T::Stride)}, lane);
  wi.add_pointer_offset((warp % WarpsM) * WM * T::Stride + (warp / WarpsM) * WN);
  #pragma unroll
  for (int i = 0; i < T::FI::kIterations; ++i) {
    typename T::FI::Fragment fragment;
    fi.load(fragment); ++fi;
    wi.store(fragment);
    wi.add_pointer_offset(T::WI::Shape::kRow * T::Stride);
  }
  }
  __syncthreads();
  if (!Final || (!transpose && ln % 4 == 0)) {
    for (int idx = threadIdx.x * 4; idx < BM * BN; idx += T::Threads * 4) {
      int r = idx / BN, c = idx % BN, row = row0 + r, col = col0 + c;
      if (row >= m || col >= n || (symmetric && row / 64 < col / 64)) continue;
      cutlass::AlignedArray<Output, 4> values;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float value = shared.tile[r * T::Stride + c + j];
        if constexpr (Quad) {
          value = __fadd_rn(__fmul_rn(cb, float(a[(int64_t(batch) * m + row) * k + col + j])), __fmul_rn(cc, value));
          if (row == col + j) value = __fadd_rn(value, ca);
        }
        E rounded = E(row < lm && col + j < ln ? value : 0);
        values[j] = Output(float(rounded));
      }
      if constexpr (Final) {
        if (row < lm && col < ln) {
          int64_t index = (int64_t(batch) * lm + row) * ln + col;
          reinterpret_cast<cutlass::AlignedArray<Output, 4>*>(output)[index / 4] = values;
        }
      } else {
        int64_t index = (int64_t(batch) * m + row) * n + col;
        reinterpret_cast<cutlass::AlignedArray<Output, 4>*>(output)[index / 4] = values;
      }
    }
  } else {
  for (int idx = threadIdx.x; idx < BM * BN; idx += T::Threads) {
    int r = idx / BN, c = idx % BN, row = row0 + r, col = col0 + c;
    if constexpr (Final) {
      if (transpose) { r = idx % BM; c = idx / BM; row = row0 + r; col = col0 + c; }
    }
    if (row >= m || col >= n || (symmetric && row / 64 < col / 64)) continue;
    float value = shared.tile[r * T::Stride + c];
    if constexpr (Quad) {
      value = __fadd_rn(__fmul_rn(cb, float(a[(int64_t(batch) * m + row) * k + col])), __fmul_rn(cc, value));
      if (row == col) value = __fadd_rn(value, ca);
    }
    E rounded = E(row < lm && col < ln ? value : 0);
    if constexpr (Final) {
      if (row < lm && col < ln) {
        int64_t index = transpose ? (int64_t(batch) * ln + col) * lm + row
                                  : (int64_t(batch) * lm + row) * ln + col;
        static_cast<Output*>(output)[index] = Output(float(rounded));
      }
    } else {
      static_cast<E*>(output)[(int64_t(batch) * m + row) * n + col] = rounded;
    }
  }
  }
  if constexpr (!Final) {
    if (symmetric || output_t) {
      for (int idx = threadIdx.x * 4; idx < BM * BN; idx += T::Threads * 4) {
        int r = idx % BM, c = idx / BM, row = row0 + r, col = col0 + c;
        if (row >= m || col >= n || (symmetric && row / 64 <= col / 64)) continue;
        cutlass::AlignedArray<E, 4> values;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          float value = shared.tile[(r + j) * T::Stride + c];
          if constexpr (Quad) {
            value = __fadd_rn(__fmul_rn(cb, float(a[(int64_t(batch) * m + row + j) * k + col])), __fmul_rn(cc, value));
          }
          values[j] = E(row + j < lm && col < ln ? value : 0);
        }
        E* target = symmetric ? static_cast<E*>(output) : output_t;
        int64_t index = (int64_t(batch) * n + col) * m + row;
        reinterpret_cast<cutlass::AlignedArray<E, 4>*>(target)[index / 4] = values;
      }
    }
  }
}

template <typename E> void normalize_typed(Tensor x, Tensor norm, Tensor out, Tensor out_t,
                                          bool transpose, double eps) {
  dim3 grid((out.size(2) + 31) / 32, (out.size(1) + 31) / 32, x.size(0));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "prec_normalize", [&] {
    normalize_kernel<scalar_t, E><<<grid, dim3(32, 8), 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<scalar_t>(), norm.data_ptr<float>(), reinterpret_cast<E*>(out.data_ptr()),
        reinterpret_cast<E*>(out_t.data_ptr()), x.size(1), x.size(2), out.size(1), out.size(2), transpose, eps);
  });
}
void normalize(Tensor x, Tensor norm, Tensor out, Tensor out_t, bool transpose, double eps) {
  c10::cuda::CUDAGuard guard(x.device());
  if (out.scalar_type() == at::kHalf) normalize_typed<cutlass::half_t>(x, norm, out, out_t, transpose, eps);
  else if (out.scalar_type() == at::kBFloat16) normalize_typed<cutlass::bfloat16_t>(x, norm, out, out_t, transpose, eps);
  else normalize_typed<float>(x, norm, out, out_t, transpose, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename E, int BM, int BN, int WM, int WN, bool Quad, bool Final, typename Output>
void launch(Tensor a, Tensor bt, Tensor out, Tensor out_t, int lm, int ln, bool symmetric,
            bool transpose, float ca, float cb, float cc) {
  using T = Traits<E, BM, BN, WM, WN>;
  int m = a.size(1), k = a.size(2), n = bt.size(1);
  auto kernel = gemm_kernel<E, BM, BN, WM, WN, Quad, Final, Output>;
  if constexpr (sizeof(typename T::Storage) > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(typename T::Storage)));
  int u = (m + BM - 1) / BM;
  int blocks = symmetric ? u * (u - 1) / 2 * (BM / BN) + (m + BN - 1) / BN
                         : u * ((n + BN - 1) / BN);
  kernel<<<dim3(blocks, a.size(0)), T::Threads, sizeof(typename T::Storage), at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<E const*>(a.data_ptr()), reinterpret_cast<E const*>(bt.data_ptr()), out.data_ptr(),
      out_t.numel() ? reinterpret_cast<E*>(out_t.data_ptr()) : nullptr, m, n, k, lm, ln,
      symmetric, transpose, ca, cb, cc,
      typename T::Mma::IteratorA::Params(cutlass::layout::RowMajor(k)),
      typename T::Mma::IteratorB::Params(cutlass::layout::ColumnMajor(k)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename E, int BM, int BN, int WM, int WN>
void dispatch(Tensor a, Tensor bt, Tensor out, Tensor out_t, int mode, int lm, int ln,
              bool symmetric, bool final, bool transpose, float ca, float cb, float cc) {
  if (mode == 1) launch<E, BM, BN, WM, WN, true, false, E>(a, bt, out, out_t, lm, ln, symmetric, false, ca, cb, cc);
  else if (!final) launch<E, BM, BN, WM, WN, false, false, E>(a, bt, out, out_t, lm, ln, symmetric, false, ca, cb, cc);
  else {
    if (out.scalar_type() == at::kHalf) launch<E, BM, BN, WM, WN, false, true, at::Half>(a, bt, out, out_t, lm, ln, false, transpose, ca, cb, cc);
    else if (out.scalar_type() == at::kBFloat16) launch<E, BM, BN, WM, WN, false, true, at::BFloat16>(a, bt, out, out_t, lm, ln, false, transpose, ca, cb, cc);
    else launch<E, BM, BN, WM, WN, false, true, float>(a, bt, out, out_t, lm, ln, false, transpose, ca, cb, cc);
  }
}

template <typename E> void dispatch_tile(Tensor a, Tensor bt, Tensor out, Tensor out_t, int mode, int lm,
    int ln, bool symmetric, bool final, bool transpose, float ca, float cb, float cc, int tile) {
  if (tile == 0) dispatch<E, 64, 64, 32, 32>(a, bt, out, out_t, mode, lm, ln, symmetric, final, transpose, ca, cb, cc);
  else if (tile == 1) dispatch<E, 128, 64, 64, 32>(a, bt, out, out_t, mode, lm, ln, symmetric, final, transpose, ca, cb, cc);
  else dispatch<E, 128, 128, 64, 32>(a, bt, out, out_t, mode, lm, ln, symmetric, final, transpose, ca, cb, cc);
}

void gemm(Tensor a, Tensor bt, Tensor out, Tensor out_t, int mode, int lm, int ln,
          bool symmetric, bool final, bool transpose, double ca, double cb, double cc, int tile) {
  c10::cuda::CUDAGuard guard(a.device());
  if (a.scalar_type() == at::kHalf) dispatch_tile<cutlass::half_t>(a, bt, out, out_t, mode, lm, ln, symmetric, final, transpose, ca, cb, cc, tile);
  else if (a.scalar_type() == at::kBFloat16) dispatch_tile<cutlass::bfloat16_t>(a, bt, out, out_t, mode, lm, ln, symmetric, final, transpose, ca, cb, cc, tile);
  else dispatch_tile<float>(a, bt, out, out_t, mode, lm, ln, symmetric, final, transpose, ca, cb, cc, tile);
}

template <typename E> std::vector<int> shared_sizes() {
  return {int(sizeof(typename Traits<E, 64, 64, 32, 32>::Storage)),
          int(sizeof(typename Traits<E, 128, 64, 64, 32>::Storage)),
          int(sizeof(typename Traits<E, 128, 128, 64, 32>::Storage))};
}
std::vector<int> shared_memory(std::string dtype) {
  if (dtype == "float16") return shared_sizes<cutlass::half_t>();
  if (dtype == "bfloat16") return shared_sizes<cutlass::bfloat16_t>();
  TORCH_CHECK(dtype == "float32", "Unsupported floating-point precision");
  return shared_sizes<float>();
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("normalize", &normalize);
  m.def("gemm", &gemm);
  m.def("shared_memory", &shared_memory);
}
