# CUTLASS/CUDA implementation

| File | Purpose |
|---|---|
| `src/i8muon/_coeff.py` | INT8 safety coefficients, Polar Express, iteration extensions |
| `src/i8muon/_optimizer.py` | Routing, precision/hardware checks, Muon, Nesterov, weight updates, state restoration, auxiliary optimizer, CUDA Graphs, `muon_step` |
| `src/i8muon/_cutlass.py` | Expanded INT8 NS and the FP16-to-INT8/BQ handoff |
| `src/i8muon/_cutlass_prec.py` | FP16/BF16/FP32 regular and Gram NS |
| `src/i8muon/_cutlass_bq.py` | 128×128 INT8 / FP8 E4M3 block quantization |
| `src/i8muon/_cuda_factory.py` | Resource filtering, operation autotuning, complete-pipeline comparison and caching |
| `src/i8muon/_cutlass_build.py` | Extension compilation, architecture flags, header discovery |
| `src/i8muon/csrc/{prec,int8,bq}.cu` | CUTLASS GEMM, fused epilogues, reductions, quantization kernels |
| `src/i8muon/_vendor/cutlass/` | Bundled CUTLASS headers and licenses |

The implementation does not import TileLang or use its JIT/autotuner. Direct calls use `CutlassNSInt8` from the package API.

## BMM convention

The nine main paths are FP32/BF16/FP16 regular and Gram NS, INT8 regular NS, and INT8/FP8 BQ NS. Native matrix entry points accept `(B,M,K)`, `(B,N,K)`, and `(B,M,N)`, including `B=1`. Multiple leading batch dimensions are folded and restored on return.

GEMMs in all three CUDA sources launch with `dim3(blocks, a.size(0))`. `blockIdx.y` selects the batch entry; batch strides address both inputs, outputs, quantization scales, and separate diagonals. Symmetric products use triangular tiles while retaining independent storage and computation for each batch entry. Normalization, transpose, and quantization helpers also use separate batch indices. These are native batched GEMMs, without Python loops that reduce execution to two-dimensional per-sample products.

## Arithmetic paths

The first FP16 stage normalizes the input once using an FP32 Frobenius norm. Subsequent INT8 stages quantize that result without normalizing again. All five stages use the expanded ABC polynomial with a separate FP32 diagonal. Regular INT8 uses global scales; BQ stores scales per 128×128 block.

Regular INT8 rounds to nearest even; INT8 BQ truncates toward zero; FP8 BQ converts to E4M3FN with saturation at ±448. Each path retains its own quantization rules, so different quantization paths need not be bitwise equal. Floating-point and FP8 paths also default to the INT8 safety coefficients; pass the corresponding `ns_coefficients` explicitly for Polar Express.

With `fp16_first=False`, all-INT8 execution normalizes the input using an FP32 norm. BQ input norms also use FP32, avoiding FP16 norm overflow beyond 65504. Regular INT8 retains its fused atomic reduction option when `deterministic=False`.

FP16/BF16 use Tensor Cores with FP32 accumulation; FP32 uses SIMT FMA. Gram defaults to `I8MUON_GRAM_RESET=00100`, with extra iterations continuing as 0. The reset string must start with 0 and contain only 0/1. Each call owns its workspace and reuses buffers only when their lifetimes do not overlap.

Regular INT8 and INT8 BQ support SM87/SM89/SM120. FP8 BQ dispatches only on SM89+; SM87 reports an error before compilation. `cp.async` is the general path. TMA is used only when both the compilation target and input GPU support it. Explicit tile 4 requires a corresponding TMA target; tiles exceeding shared-memory limits are excluded from tuning.

## Tuning and dispatch

`router(M,N,precision,use_gram,gram_aspect_threshold)` selects the algorithm and compute precision. `KernelFactory` caches tiles by input device, dtype, shape, operation signature, and architecture. Timings use batches of CUDA Graph replays and round medians. Operation improvements below 2% retain the default tile; complete-pipeline improvements below 1% retain the default combination. Explicit tile overrides apply only to regular INT8.

Routing lives in `_optimizer.py`. Auto and explicit precision requests first share the decision `use_gram and aspect_ratio >= gram_aspect_threshold`, with a default threshold of 5.0. When Gram is not selected, auto chooses FP16 or INT8 based on matrix element count.

`autotune=False` performs no timing. With `autotune=True`, new signatures are measured once and later calls use cached results. An untuned signature encountered during capture raises an instruction to warm up first.

Optimizer graph keys include the method, precision, device, dtype, complete shape, coefficients, eps, determinism, first-stage option, tiles, and Gram reset order. This prevents parameter groups with different configurations from sharing an incompatible graph. Loading optimizer state restores group settings and clears old graphs before recapture.

Graph keys also distinguish engine instances, so functional callers sharing a graph dictionary do not reuse another engine's captured graph. Complete-pipeline tuning caches distinguish explicit tile configurations as well.

## Explicit polynomial entry points

`_regular_i8_factored`, `_regular_i8_grouped_cubic`, and `_regular_i8_cubic_abc` remain available for explicit calls. These entry points do not participate in default coefficient selection or automatic dispatch; defaults use the expanded five-stage INT8 safety schedule.

## Build

Native extension cache names have the `_native` suffix. Shapes share the same extension; there is no per-shape recompilation. All CUDA sources use `--fmad=true`, allowing automatic floating-point multiply-add fusion. 

