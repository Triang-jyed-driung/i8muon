// Regular int8 Newton-Schulz: CUTLASS MMA mainloops and fused CUDA epilogues.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/block/block_reduce.cuh>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cutlass/epilogue/warp/tile_iterator_tensor_op.h>
#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

namespace {
using Tensor = torch::Tensor;
constexpr int Threads = 256;
constexpr int StatsElements = 32;
template <typename Value> struct MaxOp {
  __device__ Value operator()(Value a, Value b) const { return a > b ? a : b; }
};

// Address a per-batch reduction slot without constructing Tensor views.
void* reduction_pointer(Tensor const& t, int64_t batch, int64_t slot) {
  return static_cast<char*>(t.data_ptr()) + slot * batch * t.element_size();
}

__device__ int8_t quantize(float x, float multiplier) {
  return static_cast<int8_t>(max(-127, min(127, __float2int_rn(x * multiplier))));
}

__device__ void atomic_max_float(float* p, float value) {
  // All callers supply finite, nonnegative magnitudes; their float bits are ordered.
  atomicMax(reinterpret_cast<int*>(p), __float_as_int(value));
}

template <typename Input, bool Sum>
__global__ void stats_kernel(Input const* input, float* maximum, float* sum,
                             int64_t matrix_size) {
  using Reduce = cub::BlockReduce<float, Threads>;
  __shared__ typename Reduce::TempStorage storage;
  int64_t base = (int64_t(blockIdx.y) * matrix_size) + int64_t(blockIdx.x) * Threads * StatsElements;
  float vmax = 0, vsum = 0;
  #pragma unroll
  for (int i = 0; i < StatsElements; ++i) {
    int64_t offset = int64_t(blockIdx.x) * Threads * StatsElements + threadIdx.x + i * Threads;
    float value = offset < matrix_size ? float(input[base + threadIdx.x + i * Threads]) : 0;
    vmax = fmaxf(vmax, fabsf(value));
    if constexpr (Sum) vsum += value * value;
  }
  vmax = Reduce(storage).Reduce(vmax, MaxOp<float>());
  if (threadIdx.x == 0) atomic_max_float(maximum + blockIdx.y, vmax);
  if constexpr (Sum) {
    __syncthreads();
    vsum = Reduce(storage).Sum(vsum);
    if (threadIdx.x == 0) atomicAdd(sum + blockIdx.y, vsum);
  }
}

template <typename Input, bool Transpose>
__global__ void input_quant_kernel(Input const* input, float const* maximum,
    void const* norm_ptr, bool norm_is_float, bool use_norm, int8_t* output,
    int8_t* output_t, float* scale, int rows, int cols, int lp, int hp, float eps) {
  __shared__ int8_t tile[32][33];
  int batch = blockIdx.z;
  float amax = maximum[batch];
  float norm = norm_is_float ? static_cast<float const*>(norm_ptr)[batch]
                            : float(static_cast<Input const*>(norm_ptr)[batch]);
  if (!use_norm) norm = sqrtf(norm);
  float multiplier = amax > 0 ? 127.0f / amax : 0;
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    scale[batch] = amax > 0 ? amax / (127.0f * fmaxf(norm, eps)) : 0;
  int padded_rows = Transpose ? hp : lp, padded_cols = Transpose ? lp : hp;
  #pragma unroll
  for (int j = 0; j < 32; j += 8) {
    int row = blockIdx.y * 32 + threadIdx.y + j, col = blockIdx.x * 32 + threadIdx.x;
    int64_t src = int64_t(row) * cols + col;
    int8_t value = row < rows && col < cols
        ? quantize(float(input[int64_t(batch) * rows * cols + src]), multiplier) : 0;
    tile[threadIdx.y + j][threadIdx.x] = value;
    if (row < padded_rows && col < padded_cols) {
      int8_t* primary = Transpose ? output_t : output;
      if (primary) primary[(int64_t(batch) * padded_rows + row) * padded_cols + col] = value;
    }
  }
  if (!Transpose && !output_t) return;
  __syncthreads();
  #pragma unroll
  for (int j = 0; j < 32; j += 8) {
    int row = blockIdx.x * 32 + threadIdx.y + j, col = blockIdx.y * 32 + threadIdx.x;
    if (row < padded_cols && col < padded_rows) {
      int8_t* secondary = Transpose ? output : output_t;
      secondary[(int64_t(batch) * padded_cols + row) * padded_rows + col] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

template <int BM, int BN, int BK, int Stages, int WarpM, int WarpN>
struct GemmTraits {
  using Row = cutlass::layout::RowMajor;
  using Col = cutlass::layout::ColumnMajor;
  using BLayout = Col;
  using Shape = cutlass::gemm::GemmShape<BM, BN, BK>;
  using Warp = cutlass::gemm::GemmShape<WarpM, WarpN, BK>;
  using Instruction = cutlass::gemm::GemmShape<16, 8, 32>;
  using Default = cutlass::gemm::kernel::DefaultGemm<
      int8_t, Row, 16, int8_t, BLayout, 16, int32_t, Row, int32_t,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Shape, Warp, Instruction,
      cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, false,
      cutlass::arch::OpMultiplyAddSaturate>;
  using Mma = typename Default::GemmKernel::Mma;
  using ParamsA = typename Mma::IteratorA::Params;
  using ParamsB = typename Mma::IteratorB::Params;
};

// Mode 0: A A^T -> int32 lower triangle + off-diagonal max.
// Mode 1: b A + c A^2, with a float diagonal kept separately.
// Mode 2: Type2 x Type1, including its exact float diagonal correction.
// Mode 3: unscaled shifted square; Mode 4: update with gain folded into b_scale.
// Column-major B reinterprets row-major rows of a symmetric matrix / A^T.
template <int BM, int BN, int Mode, typename Output, bool Final, bool Transpose,
          int BK, int Stages, int WarpM, int WarpN, bool UseTma>
__global__ void gemm_kernel(int8_t const* a, int8_t const* b, int8_t const* row_b, void* output,
    void* maximum, float const* a_scale, float const* b_scale, float const* diag,
    int m, int n, int k, int logical_m, int logical_n, float alpha, float beta,
    __grid_constant__ const typename GemmTraits<BM, BN, BK, Stages, WarpM, WarpN>::ParamsA params_a,
    __grid_constant__ const typename GemmTraits<BM, BN, BK, Stages, WarpM, WarpN>::ParamsB params_b,
    __grid_constant__ const CUtensorMap tma_a, __grid_constant__ const CUtensorMap tma_b) {
  using Traits = GemmTraits<BM, BN, BK, Stages, WarpM, WarpN>;
  using Row = cutlass::layout::RowMajor;
  using Warp = typename Traits::Warp;
  using Instruction = typename Traits::Instruction;
  using Mma = typename Traits::Mma;
  using FragIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
      Warp, Instruction, int32_t, cutlass::Array<int32_t, 4>, Row>;
  using WarpIterator = cutlass::epilogue::warp::TileIteratorTensorOp<Warp, Instruction, int32_t, Row>;
  constexpr int WM = BM / Warp::kM, WN = BN / Warp::kN;
  constexpr int Consumers = WM * WN * 32;
  constexpr int NT = Consumers + (UseTma ? 32 : 0);
  constexpr int SliceRows = WM * 8, SmemStride = BN + 8;
  using MaxType = typename std::conditional<Mode == 0, int32_t, float>::type;
  using Reduce = cub::BlockReduce<MaxType, NT>;
  union Storage {
    typename Mma::SharedStorage mma;
    int32_t epilogue[SliceRows * SmemStride];
    typename Reduce::TempStorage reduction;
  };
  static_assert(sizeof(Storage) <= Stages * (BM + BN) * BK, "Insufficient shared memory allocation");
  extern __shared__ __align__(1024) unsigned char raw[];
  Storage& shared = *reinterpret_cast<Storage*>(raw);
  int tile_m, tile_n;
  if constexpr (Mode != 2 && Mode != 4) {
    constexpr int R = BM / BN;
    int base = blockIdx.x / R;
    tile_m = int((sqrtf(float(8 * base + 1)) - 1) * 0.5f);
    // Correct inverse-triangular rounding at boundaries.
    while ((tile_m + 1) * (tile_m + 2) / 2 <= base) ++tile_m;
    while (tile_m * (tile_m + 1) / 2 > base) --tile_m;
    tile_n = (base - tile_m * (tile_m + 1) / 2) * R + blockIdx.x % R;
  } else {
    tile_m = blockIdx.x / ((n + BN - 1) / BN);
    tile_n = blockIdx.x % ((n + BN - 1) / BN);
  }
  int row0 = tile_m * BM, col0 = tile_n * BN, batch = blockIdx.y;
  int64_t a_batch = int64_t(batch) * m * k;
  int64_t b_batch = int64_t(batch) * k * n;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  typename Mma::FragmentC accum;
  accum.clear();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (UseTma) {
    static_assert(Stages == 2, "TMA pipeline uses two stages");
    __shared__ __align__(16) uint64_t barriers[4];
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    if (threadIdx.x == 0) {
      Barrier::init(barriers, 1);
      Barrier::init(barriers + 1, 1);
      Barrier::init(barriers + 2, WM * WN);
      Barrier::init(barriers + 3, WM * WN);
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();
    constexpr int StageBytes = (BM + BN) * BK;
    auto issue = [&](int iteration) {
      if (threadIdx.x == Consumers) {
        int slot = iteration & 1;
        int8_t* ap = reinterpret_cast<int8_t*>(raw) + slot * StageBytes;
        int8_t* bp = ap + BM * BK;
        Barrier::arrive_and_expect_tx(barriers + slot, StageBytes);
        uint64_t hint = uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL);
        cute::SM90_TMA_LOAD_3D::copy(&tma_a, barriers + slot, hint, ap, iteration * BK, row0, batch);
        cute::SM90_TMA_LOAD_3D::copy(&tma_b, barriers + slot, hint, bp, iteration * BK, col0, batch);
      }
    };
    using WarpMma = typename Mma::Operator;
    WarpMma op;
    int iterations = (k + BK - 1) / BK;
    if (threadIdx.x >= Consumers) {
      if (threadIdx.x == Consumers) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
          if (iteration >= 2) Barrier::wait(barriers + 2 + (iteration & 1), ((iteration / 2) - 1) & 1);
          issue(iteration);
        }
      }
    } else {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      int slot = iteration & 1;
      Barrier::wait(barriers + slot, (iteration / 2) & 1);
      int8_t* ap = reinterpret_cast<int8_t*>(raw) + slot * StageBytes;
      int8_t* bp = ap + BM * BK;
      typename WarpMma::IteratorA wa({ap, WarpMma::LayoutA::packed({BM, BK})}, lane);
      typename WarpMma::IteratorB wb({bp, WarpMma::LayoutB::packed({BK, BN})}, lane);
      wa.add_tile_offset({warp % WM, 0});
      wb.add_tile_offset({0, warp / WM});
      typename WarpMma::FragmentA fa[2];
      typename WarpMma::FragmentB fb[2];
      wa.set_kgroup_index(0); wb.set_kgroup_index(0);
      wa.load(fa[0]); wb.load(fb[0]); ++wa; ++wb;
      #pragma unroll
      for (int group = 0; group < BK / 32; ++group) {
        if (group + 1 < BK / 32) {
          wa.set_kgroup_index(group + 1); wb.set_kgroup_index(group + 1);
          wa.load(fa[(group + 1) & 1]); wb.load(fb[(group + 1) & 1]); ++wa; ++wb;
        }
        typename WarpMma::TransformedFragmentA ta;
        typename WarpMma::TransformedFragmentB tb;
        op.transform(ta, tb, fa[group & 1], fb[group & 1]);
        op(accum, ta, tb, accum);
      }
      __syncwarp();
      if (lane == 0) Barrier::arrive(barriers + 2 + slot);
    }
    }
    __syncthreads();
  } else
#endif
  {
  typename Mma::IteratorA ia(params_a,
      const_cast<int8_t*>(a) + a_batch, {m, k}, threadIdx.x, {row0, 0});
  typename Mma::IteratorB ib(params_b,
      const_cast<int8_t*>(b) + b_batch, {k, n}, threadIdx.x, {0, col0});
  Mma mma(shared.mma, threadIdx.x, warp, lane);
  mma((k + BK - 1) / BK, accum, ia, ib, accum);
  __syncthreads();
  }

  FragIterator fi(accum);
  int warp_m = warp % WM, warp_n = warp / WM;
  MaxType block_max = 0;
  float sa = Mode == 0 ? 0 : a_scale[batch];
  float sb = 0;
  if constexpr (Mode == 2) sb = b_scale[batch];
  if constexpr (Mode == 4) sb = b_scale[batch] * alpha;
  float sqscale = sa * sa;
  if constexpr (Mode == 1) sqscale *= beta;
  #pragma unroll
  for (int iter = 0; iter < FragIterator::kIterations; ++iter) {
    typename FragIterator::Fragment frag;
    fi.load(frag); ++fi;
    int row = row0 + warp_m * Warp::kM + iter * 8 + lane / 4;
    #pragma unroll
    for (int j = 0; j < Warp::kN / 8; ++j) {
      int col = col0 + warp_n * Warp::kN + j * 8 + (lane % 4) * 2;
      if (row < m && col < n && warp < WM * WN) {
        if constexpr (Mode == 0) {
          int32_t v0 = frag[j * 2], v1 = frag[j * 2 + 1];
          int64_t index = (int64_t(batch) * m + row) * n + col;
          *reinterpret_cast<int2*>(static_cast<int32_t*>(output) + index) = make_int2(v0, v1);
          if (row < logical_m) {
            if (col < logical_n && row != col) block_max = max(block_max, abs(v0));
            if (col + 1 < logical_n && row != col + 1) block_max = max(block_max, abs(v1));
          }
        } else {
          float results[2];
          char2 input_pair;
          if constexpr (Mode == 2 || Mode == 4)
            input_pair = *reinterpret_cast<char2 const*>(row_b + (int64_t(batch) * m + row) * n + col);
          #pragma unroll
          for (int e = 0; e < 2; ++e) {
            int c = col + e;
            float value = float(frag[j * 2 + e]);
            if constexpr (Mode == 1) {
              float ri = (diag[int64_t(batch) * m + row] * beta + alpha) * sa;
              float cj = (diag[int64_t(batch) * m + c] * sa) * beta;
              results[e] = float(a[(int64_t(batch) * m + row) * k + c]) * (ri + cj) + value * sqscale;
            } else if constexpr (Mode == 3) {
              float ri = diag[int64_t(batch) * m + row] * sa;
              float cj = diag[int64_t(batch) * m + c] * sa;
              results[e] = float(a[(int64_t(batch) * m + row) * k + c]) * (ri + cj) + value * sqscale;
            } else {
              results[e] = (float(e == 0 ? input_pair.x : input_pair.y)
                  * diag[int64_t(batch) * m + row] + value * sa) * sb;
            }
            if constexpr (!Final) {
              if ((Mode == 2 || Mode == 4 || (row != c && row / 128 >= c / 128)) && row < logical_m && c < logical_n)
                block_max = fmaxf(block_max, fabsf(results[e]));
            }
          }
          if constexpr (!Final) {
            int64_t index = (int64_t(batch) * m + row) * n + col;
            *reinterpret_cast<float2*>(static_cast<float*>(output) + index) = make_float2(results[0], results[1]);
          } else if constexpr (Transpose) {
            int sr = warp_m * 8 + lane / 4;
            int sc = col - col0;
            shared.epilogue[sr * SmemStride + sc] = __float_as_int(results[0]);
            shared.epilogue[sr * SmemStride + sc + 1] = __float_as_int(results[1]);
          } else if (row < logical_m) {
            int64_t index = (int64_t(batch) * logical_m + row) * logical_n + col;
            if constexpr (std::is_same<Output, float>::value) {
              if (logical_n % 2 == 0 && col + 1 < logical_n)
                *reinterpret_cast<float2*>(static_cast<float*>(output) + index) = make_float2(results[0], results[1]);
              else {
                if (col < logical_n) static_cast<float*>(output)[index] = results[0];
                if (col + 1 < logical_n) static_cast<float*>(output)[index + 1] = results[1];
              }
            } else {
              #pragma unroll
              for (int e = 0; e < 2; ++e) {
                if (col + e < logical_n) static_cast<Output*>(output)[index + e] = Output(results[e]);
              }
            }
          }
        }
      }
    }
    if constexpr (Final && Transpose) {
      __syncthreads();
      for (int t = threadIdx.x; t < SliceRows * BN; t += NT) {
        int sr = t % SliceRows, sc = t / SliceRows;
        int row = row0 + (sr / 8) * Warp::kM + iter * 8 + sr % 8;
        int col = col0 + sc;
        if (row < logical_m && col < logical_n)
          static_cast<Output*>(output)[(int64_t(batch) * logical_n + col) * logical_m + row] =
              Output(__int_as_float(shared.epilogue[sr * SmemStride + sc]));
      }
      __syncthreads();
    }
  }
  if constexpr (!Final) {
    // The reduction reuses mainloop storage after all asynchronous loads drain.
    __syncthreads();
    MaxType value = Reduce(shared.reduction).Reduce(block_max, MaxOp<MaxType>());
    if (threadIdx.x == 0) {
      if constexpr (Mode == 0) atomicMax(static_cast<int32_t*>(maximum) + batch, value);
      else atomic_max_float(static_cast<float*>(maximum) + batch, value);
    }
  }
}

__device__ __forceinline__ void store_int8_pair(int8_t* output, int64_t index, char2 value, bool second) {
  if ((index & 1) == 0 && second) *reinterpret_cast<char2*>(output + index) = value;
  else {
    output[index] = value.x;
    if (second) output[index + 1] = value.y;
  }
}

template <typename Input, bool Quad, bool Factored = false>
__global__ void symmetric_quant_kernel(Input const* input, void const* maximum,
    float const* orig_scale, float const* orig_diag, int8_t* output, float* scale,
    float* diag, int m, float c0, float c1, float c2) {
  __shared__ __align__(16) int8_t tile[32][34];
  int batch = blockIdx.y;
  int pid = blockIdx.x;
  int tile_m = int((sqrtf(float(8 * pid + 1)) - 1) * 0.5f);
  while ((tile_m + 1) * (tile_m + 2) / 2 <= pid) ++tile_m;
  while (tile_m * (tile_m + 1) / 2 > pid) --tile_m;
  int tile_n = pid - tile_m * (tile_m + 1) / 2;
  float mx;
  if constexpr (Quad) mx = static_cast<float const*>(maximum)[batch];
  else mx = float(static_cast<int32_t const*>(maximum)[batch]);
  float multiplier = mx > 0 ? 127.0f / mx : 0;
  float sa = Quad ? 1 : orig_scale[batch] * orig_scale[batch];
  if (pid == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    scale[batch] = mx > 0 ? sa / multiplier : 0;
  #pragma unroll
  for (int j = 0; j < 32; j += 16) {
    int r = threadIdx.y + j, c = threadIdx.x * 2;
    int row = tile_m * 32 + r, col = tile_n * 32 + c;
    int8_t values[2];
    #pragma unroll
    for (int e = 0; e < 2; ++e) {
      float value = row < m && col + e < m ? float(input[(int64_t(batch) * m + row) * m + col + e]) : 0;
      values[e] = row == col + e ? 0 : quantize(value, multiplier);
      if (row == col + e && row < m) {
        if constexpr (Quad) {
          float d = orig_diag[int64_t(batch) * m + row];
          if constexpr (Factored) diag[int64_t(batch) * m + row] = (value + d * d) + c0;
          else diag[int64_t(batch) * m + row] = value + (c2 * d + c1) * d + c0;
        } else {
          if constexpr (Factored) diag[int64_t(batch) * m + row] = value * sa + c0;
          else diag[int64_t(batch) * m + row] = value * sa;
        }
      }
    }
    *reinterpret_cast<char2*>(&tile[r][c]) = make_char2(values[0], values[1]);
  }
  __syncthreads();
  #pragma unroll
  for (int j = 0; j < 32; j += 16) {
    int r = threadIdx.y + j, c = threadIdx.x * 2;
    int row = tile_m * 32 + r, col = tile_n * 32 + c;
    if (row < m && col < m) {
      char2 value = make_char2(tile_m == tile_n && r < c ? tile[c][r] : tile[r][c],
                              tile_m == tile_n && r < c + 1 ? tile[c + 1][r] : tile[r][c + 1]);
      store_int8_pair(output, (int64_t(batch) * m + row) * m + col, value, col + 1 < m);
    }
    if (tile_m != tile_n) {
      row = tile_n * 32 + r; col = tile_m * 32 + c;
      if (row < m && col < m)
        store_int8_pair(output, (int64_t(batch) * m + row) * m + col,
                        make_char2(tile[c][r], tile[c + 1][r]), col + 1 < m);
    }
  }
}

__global__ void output_quant_kernel(float const* input, float const* maximum,
    int8_t* output, int8_t* output_t, float* scale, int m, int n) {
  __shared__ __align__(16) int8_t tile[32][34];
  int batch = blockIdx.z;
  float mx = maximum[batch], multiplier = mx > 0 ? 127.0f / mx : 0;
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    scale[batch] = mx > 0 ? 1.0f / multiplier : 0;
  #pragma unroll
  for (int j = 0; j < 32; j += 16) {
    int r = threadIdx.y + j, c = threadIdx.x * 2;
    int row = blockIdx.y * 32 + r, col = blockIdx.x * 32 + c;
    char2 value = make_char2(
        row < m && col < n ? quantize(input[(int64_t(batch) * m + row) * n + col], multiplier) : 0,
        row < m && col + 1 < n ? quantize(input[(int64_t(batch) * m + row) * n + col + 1], multiplier) : 0);
    *reinterpret_cast<char2*>(&tile[r][c]) = value;
    if (row < m && col < n)
      store_int8_pair(output, (int64_t(batch) * m + row) * n + col, value, col + 1 < n);
  }
  if (!output_t) return;
  __syncthreads();
  #pragma unroll
  for (int j = 0; j < 32; j += 16) {
    int r = threadIdx.y + j, c = threadIdx.x * 2;
    int row = blockIdx.x * 32 + r, col = blockIdx.y * 32 + c;
    if (row < n && col < m)
      store_int8_pair(output_t, (int64_t(batch) * n + row) * m + col,
                      make_char2(tile[c][r], tile[c + 1][r]), col + 1 < m);
  }
}

template <int BM, int BN, int Mode, typename Output = float, bool Final = false, bool Transpose = false,
          int BK = 64, int Stages = 3, int WarpM = (BN == 128 ? 64 : 32), int WarpN = (BN == 128 ? 64 : 32), bool UseTma = false>
void launch_gemm(Tensor a, Tensor b, Tensor out, void* maximum, float const* sa, float const* sb, float const* diag,
    int logical_m, int logical_n, float alpha, float beta, Tensor row_b = {}) {
  int m = a.size(1), k = a.size(2), n = b.size(1);
  constexpr int NT = (BM / WarpM) * (BN / WarpN) * 32 + (UseTma ? 32 : 0);
  constexpr int SharedBytes = Stages * BK * (BM + BN);
  auto kernel = gemm_kernel<BM, BN, Mode, Output, Final, Transpose, BK, Stages, WarpM, WarpN, UseTma>;
  if constexpr (SharedBytes > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes));
  CUtensorMap tma_a{}, tma_b{};
  if constexpr (UseTma) {
    auto encode = [&](CUtensorMap* map, Tensor tensor, int height, int width, int box_rows) {
      uint64_t dims[3] = {uint64_t(width), uint64_t(height), uint64_t(tensor.size(0))};
      uint64_t strides[2] = {uint64_t(width), uint64_t(width) * height};
      uint32_t box[3] = {BK, uint32_t(box_rows), 1};
      uint32_t element_strides[3] = {1, 1, 1};
      CUresult status = cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, tensor.data_ptr(),
          dims, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
          BK == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TORCH_CHECK(status == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(status));
    };
    encode(&tma_a, a, m, k, BM);
    encode(&tma_b, b, n, k, BN);
  }
  int blocks;
  if constexpr (Mode == 2 || Mode == 4) blocks = ((m + BM - 1) / BM) * ((n + BN - 1) / BN);
  else {
    int u = (m + BM - 1) / BM;
    blocks = u * (u - 1) / 2 * (BM / BN) + (m + BN - 1) / BN;
  }
  kernel<<<dim3(blocks, a.size(0)), NT, SharedBytes, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), row_b.defined() ? row_b.data_ptr<int8_t>() : nullptr, out.data_ptr(), maximum,
      sa, sb, diag, m, n, k, logical_m, logical_n, alpha, beta,
      typename GemmTraits<BM, BN, BK, Stages, WarpM, WarpN>::ParamsA(cutlass::layout::RowMajor(k)),
      typename GemmTraits<BM, BN, BK, Stages, WarpM, WarpN>::ParamsB(cutlass::layout::ColumnMajor(k)), tma_a, tma_b);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void stats(Tensor input, Tensor maximum, Tensor sum, bool compute_sum, int max_slot, int sum_slot) {
  auto mx = static_cast<float*>(reduction_pointer(maximum, input.size(0), max_slot));
  auto total = static_cast<float*>(reduction_pointer(sum, input.size(0), sum_slot));

  c10::cuda::CUDAGuard guard(input.device());
  int64_t size = input.size(1) * input.size(2);
  dim3 grid((size + Threads * StatsElements - 1) / (Threads * StatsElements), input.size(0));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "stats", [&] {
    if (compute_sum) stats_kernel<scalar_t, true><<<grid, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(), mx, total, size);
    else stats_kernel<scalar_t, false><<<grid, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(), mx, total, size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void input_quant(Tensor input, Tensor maximum, Tensor norm, Tensor output, Tensor output_t, Tensor scale,
                 bool transpose, bool use_norm, double eps, int max_slot, int norm_slot) {
  auto mx = static_cast<float*>(reduction_pointer(maximum, input.size(0), max_slot));

  auto norm_ptr = reduction_pointer(norm, input.size(0), norm_slot);

  c10::cuda::CUDAGuard guard(input.device());
  int rows = input.size(1), cols = input.size(2), lp = output.size(1), hp = output.size(2);
  dim3 grid((cols + 31) / 32, (rows + 31) / 32, input.size(0));
  dim3 block(32, 8);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "input_quant", [&] {
    if (transpose) input_quant_kernel<scalar_t, true><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(), mx, norm_ptr, norm.scalar_type() == at::kFloat,
        use_norm, output.data_ptr<int8_t>(), output_t.data_ptr<int8_t>(), scale.data_ptr<float>(), rows, cols, lp, hp, eps);
    else input_quant_kernel<scalar_t, false><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(), mx, norm_ptr, norm.scalar_type() == at::kFloat,
        use_norm, output.data_ptr<int8_t>(), output_t.data_ptr<int8_t>(), scale.data_ptr<float>(), rows, cols, lp, hp, eps);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void aat(Tensor a, Tensor out, Tensor mx, int logical_m, int tile, int max_slot) {
  auto maximum = reduction_pointer(mx, a.size(0), max_slot);

  c10::cuda::CUDAGuard guard(a.device());
  if (tile == 0) launch_gemm<128, 64, 0>(a, a, out, maximum, nullptr, nullptr, nullptr, logical_m, logical_m, 0, 0);
  else if (tile == 1) launch_gemm<128, 128, 0>(a, a, out, maximum, nullptr, nullptr, nullptr, logical_m, logical_m, 0, 0);
  else if (tile == 2) launch_gemm<256, 128, 0>(a, a, out, maximum, nullptr, nullptr, nullptr, logical_m, logical_m, 0, 0);
  else if (tile == 3) launch_gemm<128, 128, 0, float, false, false, 128, 3, 64, 32>(a, a, out, maximum, nullptr, nullptr, nullptr, logical_m, logical_m, 0, 0);
  else launch_gemm<256, 128, 0, float, false, false, 128, 2, 64, 64, true>(a, a, out, maximum, nullptr, nullptr, nullptr, logical_m, logical_m, 0, 0);
}

template <bool Factored>
void square_impl(Tensor a, Tensor sa, Tensor diag, Tensor out, Tensor mx,
            int logical_m, double alpha, double beta, int tile, int max_slot) {
  auto maximum = reduction_pointer(mx, a.size(0), max_slot);

  c10::cuda::CUDAGuard guard(a.device());
  constexpr int Mode = Factored ? 3 : 1;
  if (tile == 0) launch_gemm<128, 64, Mode>(a, a, out, maximum, sa.data_ptr<float>(), sa.data_ptr<float>(), diag.data_ptr<float>(), logical_m, logical_m, alpha, beta);
  else if (tile == 1) launch_gemm<128, 128, Mode>(a, a, out, maximum, sa.data_ptr<float>(), sa.data_ptr<float>(), diag.data_ptr<float>(), logical_m, logical_m, alpha, beta);
  else if (tile == 2) launch_gemm<256, 128, Mode>(a, a, out, maximum, sa.data_ptr<float>(), sa.data_ptr<float>(), diag.data_ptr<float>(), logical_m, logical_m, alpha, beta);
  else if (tile == 3) launch_gemm<128, 128, Mode, float, false, false, 128, 3, 64, 32>(a, a, out, maximum, sa.data_ptr<float>(), sa.data_ptr<float>(), diag.data_ptr<float>(), logical_m, logical_m, alpha, beta);
  else launch_gemm<256, 128, Mode, float, false, false, 128, 2, 64, 64, true>(a, a, out, maximum, sa.data_ptr<float>(), sa.data_ptr<float>(), diag.data_ptr<float>(), logical_m, logical_m, alpha, beta);
}

void square(Tensor a, Tensor sa, Tensor diag, Tensor out, Tensor mx,
            int logical_m, double alpha, double beta, int tile, int max_slot) {
  square_impl<false>(a, sa, diag, out, mx, logical_m, alpha, beta, tile, max_slot);
}

void factored_square(Tensor a, Tensor sa, Tensor diag, Tensor out, Tensor mx,
                     int logical_m, int tile, int max_slot) {
  square_impl<true>(a, sa, diag, out, mx, logical_m, 0, 1, tile, max_slot);
}

template <bool Factored>
void symmetric_quant_impl(Tensor input, Tensor mx, Tensor orig_scale, Tensor orig_diag,
    Tensor output, Tensor scale, Tensor diag, bool quad, double c0, double c1, double c2, int max_slot) {
  auto maximum = reduction_pointer(mx, output.size(0), max_slot);

  c10::cuda::CUDAGuard guard(input.device());
  int m = input.size(1);
  int tiles = (m + 31) / 32;
  dim3 grid(tiles * (tiles + 1) / 2, input.size(0));
  if (quad) symmetric_quant_kernel<float, true, Factored><<<grid, dim3(16, 16), 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(), maximum, orig_scale.data_ptr<float>(), orig_diag.data_ptr<float>(),
      output.data_ptr<int8_t>(), scale.data_ptr<float>(), diag.data_ptr<float>(), m, c0, c1, c2);
  else symmetric_quant_kernel<int32_t, false, Factored><<<grid, dim3(16, 16), 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<int32_t>(), maximum, orig_scale.data_ptr<float>(), orig_diag.data_ptr<float>(),
      output.data_ptr<int8_t>(), scale.data_ptr<float>(), diag.data_ptr<float>(), m, c0, c1, c2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void symmetric_quant(Tensor input, Tensor mx, Tensor orig_scale, Tensor orig_diag,
    Tensor output, Tensor scale, Tensor diag, bool quad, double c0, double c1, double c2, int max_slot) {
  symmetric_quant_impl<false>(input, mx, orig_scale, orig_diag, output, scale, diag, quad, c0, c1, c2, max_slot);
}

void factored_quant(Tensor input, Tensor mx, Tensor orig_scale, Tensor orig_diag,
    Tensor output, Tensor scale, Tensor diag, bool quad, double shift_or_floor, int max_slot) {
  symmetric_quant_impl<true>(input, mx, orig_scale, orig_diag, output, scale, diag, quad, shift_or_floor, 0, 1, max_slot);
}

template <int BM, int BN, int BK, int Stages, int WarpM, int WarpN, bool UseTma = false, bool Factored = false>
void launch_update(Tensor a, Tensor sa, Tensor diag, Tensor b, Tensor sb, Tensor out, void* maximum,
                   int logical_m, int logical_n, bool final, bool transpose, Tensor row_b, float gain) {
  constexpr int Mode = Factored ? 4 : 2;
  if (!final) {
    launch_gemm<BM, BN, Mode, float, false, false, BK, Stages, WarpM, WarpN, UseTma>(a, b, out, maximum, sa.data_ptr<float>(), sb.data_ptr<float>(), diag.data_ptr<float>(), logical_m, logical_n, gain, 0, row_b);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, out.scalar_type(), "final_update", [&] {
      if (transpose) launch_gemm<BM, BN, Mode, scalar_t, true, true, BK, Stages, WarpM, WarpN, UseTma>(a, b, out, maximum, sa.data_ptr<float>(), sb.data_ptr<float>(), diag.data_ptr<float>(), logical_m, logical_n, gain, 0, row_b);
      else launch_gemm<BM, BN, Mode, scalar_t, true, false, BK, Stages, WarpM, WarpN, UseTma>(a, b, out, maximum, sa.data_ptr<float>(), sb.data_ptr<float>(), diag.data_ptr<float>(), logical_m, logical_n, gain, 0, row_b);
    });
  }
}

template <bool Factored>
void update_impl(Tensor a, Tensor sa, Tensor diag, Tensor b, Tensor sb, Tensor out, Tensor mx,
            int logical_m, int logical_n, bool final, bool transpose, int tile, Tensor row_b, double gain, int max_slot) {
  auto maximum = reduction_pointer(mx, a.size(0), max_slot);

  c10::cuda::CUDAGuard guard(a.device());
  if (tile == 0) launch_update<128, 64, 64, 3, 32, 32, false, Factored>(a, sa, diag, b, sb, out, maximum, logical_m, logical_n, final, transpose, row_b, gain);
  else if (tile == 1) launch_update<128, 128, 64, 3, 64, 64, false, Factored>(a, sa, diag, b, sb, out, maximum, logical_m, logical_n, final, transpose, row_b, gain);
  else if (tile == 2) launch_update<256, 128, 64, 3, 64, 64, false, Factored>(a, sa, diag, b, sb, out, maximum, logical_m, logical_n, final, transpose, row_b, gain);
  else if (tile == 3) launch_update<128, 128, 128, 3, 64, 32, false, Factored>(a, sa, diag, b, sb, out, maximum, logical_m, logical_n, final, transpose, row_b, gain);
  else launch_update<256, 128, 128, 2, 64, 64, true, Factored>(a, sa, diag, b, sb, out, maximum, logical_m, logical_n, final, transpose, row_b, gain);
}

void update(Tensor a, Tensor sa, Tensor diag, Tensor b, Tensor sb, Tensor out, Tensor mx,
            int logical_m, int logical_n, bool final, bool transpose, int tile, Tensor row_b, int max_slot) {
  update_impl<false>(a, sa, diag, b, sb, out, mx, logical_m, logical_n, final, transpose, tile, row_b, 1, max_slot);
}

void factored_update(Tensor a, Tensor sa, Tensor diag, Tensor b, Tensor sb, Tensor out, Tensor mx,
            int logical_m, int logical_n, bool final, bool transpose, int tile, Tensor row_b, double gain, int max_slot) {
  update_impl<true>(a, sa, diag, b, sb, out, mx, logical_m, logical_n, final, transpose, tile, row_b, gain, max_slot);
}

void output_quant(Tensor input, Tensor mx, Tensor output, Tensor output_t, Tensor scale, int max_slot) {
  auto maximum = static_cast<float*>(reduction_pointer(mx, input.size(0), max_slot));

   c10::cuda::CUDAGuard guard(input.device());
  dim3 grid((input.size(2) + 31) / 32, (input.size(1) + 31) / 32, input.size(0));
  output_quant_kernel<<<grid, dim3(16, 16), 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(), maximum, output.data_ptr<int8_t>(), output_t.data_ptr<int8_t>(),
      scale.data_ptr<float>(), input.size(1), input.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("stats", &stats,
        py::arg("input"), py::arg("maximum"), py::arg("sum"), py::arg("compute_sum"), py::arg("max_slot") = 0, py::arg("sum_slot") = 0);
  m.def("input_quant", &input_quant,
        py::arg("input"), py::arg("maximum"), py::arg("norm"), py::arg("output"), py::arg("output_t"), py::arg("scale"), py::arg("transpose"), py::arg("use_norm"), py::arg("eps"), py::arg("max_slot") = 0, py::arg("norm_slot") = 0);
  m.def("aat", &aat,
        py::arg("a"), py::arg("out"), py::arg("mx"), py::arg("logical_m"), py::arg("tile"), py::arg("max_slot") = 0);
  m.def("square", &square,
        py::arg("a"), py::arg("sa"), py::arg("diag"), py::arg("out"), py::arg("mx"), py::arg("logical_m"), py::arg("alpha"), py::arg("beta"), py::arg("tile"), py::arg("max_slot") = 0);
  m.def("symmetric_quant", &symmetric_quant,
        py::arg("input"), py::arg("mx"), py::arg("orig_scale"), py::arg("orig_diag"), py::arg("output"), py::arg("scale"), py::arg("diag"), py::arg("quad"), py::arg("c0"), py::arg("c1"), py::arg("c2"), py::arg("max_slot") = 0);
  m.def("update", &update,
        py::arg("a"), py::arg("sa"), py::arg("diag"), py::arg("b"), py::arg("sb"), py::arg("out"), py::arg("mx"), py::arg("logical_m"), py::arg("logical_n"), py::arg("final"), py::arg("transpose"), py::arg("tile"), py::arg("row_b"), py::arg("max_slot") = 0);
  m.def("output_quant", &output_quant,
        py::arg("input"), py::arg("mx"), py::arg("output"), py::arg("output_t"), py::arg("scale"), py::arg("max_slot") = 0);
  m.def("factored_square", &factored_square,
        py::arg("a"), py::arg("sa"), py::arg("diag"), py::arg("out"), py::arg("mx"), py::arg("logical_m"), py::arg("tile"), py::arg("max_slot") = 0);
  m.def("factored_quant", &factored_quant,
        py::arg("input"), py::arg("mx"), py::arg("orig_scale"), py::arg("orig_diag"), py::arg("output"), py::arg("scale"), py::arg("diag"), py::arg("quad"), py::arg("shift_or_floor"), py::arg("max_slot") = 0);
  m.def("factored_update", &factored_update,
        py::arg("a"), py::arg("sa"), py::arg("diag"), py::arg("b"), py::arg("sb"), py::arg("out"), py::arg("mx"), py::arg("logical_m"), py::arg("logical_n"), py::arg("final"), py::arg("transpose"), py::arg("tile"), py::arg("row_b"), py::arg("gain"), py::arg("max_slot") = 0);
}
