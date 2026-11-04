# i8muon

A CUTLASS/CUDA Muon optimizer with floating-point, INT8, and block-quantized Newton–Schulz. The distribution includes the Python API, CUDA kernels, and CUTLASS headers.

## Features

| Path | Configuration | SM87 | SM89 | SM120 |
|---|---|---|---|---|
| FP32 regular NS | `precision="float32", use_gram=False` | Supported | Supported | Supported |
| FP32 Gram NS | `precision="float32", use_gram=True`, at the aspect threshold | Supported | Supported | Supported |
| BF16 regular NS | `precision="bfloat16", use_gram=False` | Supported | Supported | Supported |
| BF16 Gram NS | `precision="bfloat16", use_gram=True`, at the aspect threshold | Supported | Supported | Supported |
| FP16 regular NS | `precision="float16", use_gram=False` | Supported | Supported | Supported |
| FP16 Gram NS | `precision="float16", use_gram=True`, at the aspect threshold | Supported | Supported | Supported |
| Global INT8 NS | `precision="int8", use_gram=False` | Supported | Supported | Supported |
| 128×128 INT8 BQ NS | `precision="int8_bq", use_gram=False` | Supported | Supported | Supported |
| 128×128 E4M3 FP8 BQ NS | `precision="fp8_bq", use_gram=False` | Unsupported hardware | Supported | Supported |

All nine paths use native BMM with an explicit `B` dimension. An `(M,N)` input becomes `B=1`; `(B,M,N)` inputs and inputs with additional leading dimensions fold those dimensions into `B`. Gram products, polynomial squares, updates, and Gram accumulation products remain three-dimensional. They do not loop over samples in Python to call two-dimensional GEMM. Norms, quantization scales, and diagonals are stored separately for each batch entry.

Other features include automatic dispatch, operation and complete-pipeline autotuning, CUDA Graphs, batched and noncontiguous inputs, explicit tiles, an optimizer, and the functional `muon_step` API. Results preserve the input shape and dtype. INT8 retains a separate FP32 diagonal.

Both `int8` and `int8_bq` default to one FP16 iteration followed by INT8 iterations on each update; `fp16_first=False` selects all INT8. FP8 BQ uses FP8 for every iteration. Quantized Gram NS remains disabled; Gram selection dispatches to floating-point Gram NS.

## Installation

Requires Python 3.10+, CUDA-enabled PyTorch 2.2+, the CUDA Toolkit with `nvcc`, a C++ compiler, and Ninja. SM120 requires PyTorch 2.7+ with CUDA 12.8+. The PyTorch extension compiles on first use. These minimum versions follow source API requirements; execution on the minimum versions has not been validated. Target support does not imply execution validation on every GPU model.

```bash
cd /path/to/release
python -m pip install .
```

CUTLASS headers are bundled. `CUTLASS_PATH` can select another CUTLASS source or include directory. Compilation targets visible devices by default; targets can also be selected explicitly:

```bash
export TORCH_CUDA_ARCH_LIST="8.7;8.9;12.0a"
```

SM87/89 use `cp.async`. SM120 defaults to `sm_120a`, enabling applicable TMA tiles. An explicit `12.0` target uses `cp.async`. Native extensions use the `_native` cache namespace.

## Recommended coefficients

`recommend_coefficients(iters=5, *, method="int8_safe")` defaults to the INT8 safety coefficients. Each row evaluates `(a I + b G + c G²) X`, where `G=X Xᵀ`. Safety scaling is already included; do not apply it again. The function has no `precision` argument.

```python
from i8muon import recommend_coefficients

five = recommend_coefficients(method="int8_safe") # f1, f2, f3, f4, f5
six = recommend_coefficients(iters=6)   # f1, f2, f3, f3, f4, f5
seven = recommend_coefficients(iters=7) # f1, f2, f3, f3, f3, f4, f5
```

Seven steps therefore compose as `f5(f4(f3(f3(f3(f2(f1(A)))))))`. Fewer than five steps raise an error; pass custom `ns_coefficients` for shorter experiments. `Muon(ns_steps=N)` follows the same rule without appending other terminal coefficients.

`method="polar_express"` selects the supplied eight-stage Polar Express table. Its first seven stages are scaled once by `(a/1.01, b/1.01**3, c/1.01**5)`; stage eight, `(1.875,-1.25,0.375)`, stays unscaled. Fewer than eight steps select a prefix; extra steps repeat the unscaled eighth stage. Any positive integer step count is supported. Prefix selection preserves the scaling defined by the original eight-stage table.

```python
from i8muon import Muon

pe = recommend_coefficients(method="polar_express", iters=8)
pe12 = recommend_coefficients(method="polar_express", iters=12)
optimizer = Muon(matrices, ns_coefficients=pe, precision="float16")
```

The default method is `int8_safe`, also used by `Muon(ns_steps=N)`. Pass Polar Express explicitly through `ns_coefficients`. The optimizer's `precision` selects compute precision; the coefficient helper's `method` selects the table.

The complete safety table is in [coefficients/int8_safe.json](coefficients/int8_safe.json).

## Optimizer

```python
import torch
from i8muon import Muon

matrices = [p for p in model.parameters() if p.ndim >= 2]
vectors = [p for p in model.parameters() if p.ndim < 2]
auxiliary = torch.optim.AdamW(vectors, lr=2.5e-4) if vectors else None
optimizer = Muon(
    matrices,
    lr=2.5e-4,
    momentum=(0.95, 0.95),
    weight_decay=0.01,
    precision="int8",
    fp16_first=True,
    ns_steps=5,
    use_gram=False,
    adjust_lr_fn="match_rms_adamw",
    autotune=False,
    use_cuda_graph=True,
    scalar_optimizer=auxiliary,
)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Language-model embeddings and output heads can be assigned to AdamW according to the training configuration. This example only demonstrates grouping by dimensionality. `scalar_optimizer` can also own matrices, provided its parameters do not overlap with Muon's.

`momentum=0.95` selects a single momentum; `momentum=(0.95,0.95)` selects Nesterov. Learning-rate scaling accepts `spectral`, `original`, `match_rms_adamw`, `"none"`, a custom function, or `None`. Both `"none"` and `None` leave the learning rate unchanged.

`weight_decay_mode` selects which learning rate scales matrix weight decay. Let `lr` be the base learning rate, `adjusted_lr` the result of LR scaling, and `U` the NS output:

| Mode | Matrix update |
|---|---|
| `"absolute"` | `W <- (1 - weight_decay) * W - adjusted_lr * U` |
| `"base_lr"` (default) | `W <- (1 - lr * weight_decay) * W - adjusted_lr * U` |
| `"adjusted_lr"` | `W <- W - adjusted_lr * (U + weight_decay * W)` |

The default is base-LR decay, as in PyTorch Muon. With `"absolute"`, `weight_decay` is the per-step fractional decay, independent of both learning rates; it still applies when `lr=0`. Select `"adjusted_lr"` to preserve the original update and its operation order. Each mode can be set per parameter group and is also accepted by `muon_step`. All use the adjusted LR for the NS update. Auxiliary `scalar_optimizer` decay is unchanged. These are different optimizer conventions; none guarantees numerical stability. Report the mode when comparing runs with the same numeric WD. Checkpoints save the mode in each Muon parameter group; older checkpoints without it retain `"adjusted_lr"` behavior on load.

`param_groups` exposes both optimizers' groups to schedulers. `state_dict()` saves Muon state only; save and restore the auxiliary optimizer separately:

```python
checkpoint = {"muon": optimizer.state_dict()}
if auxiliary is not None:
    checkpoint["auxiliary"] = auxiliary.state_dict()
# Call each optimizer's load_state_dict separately when restoring.
```

Engine settings such as FP16-first, the Gram aspect threshold, autotuning, tiles, and CUDA Graph options are supplied through the constructor and are not saved in `state_dict()`. Recreate the optimizer with the intended settings before loading. Loading clears old graphs. Muon and auxiliary parameters must be disjoint. Matrix groups added through `add_param_group()` participate in subsequent updates.

CUDA parameters are checked against their selected route during `Muon` construction: native paths require SM80+, and FP8 BQ requires SM89+. Unsupported devices raise `ValueError`; training scripts need no duplicate hardware checks. CPU construction is allowed, but updates require CUDA parameters. The functional `muon_step` API also checks devices. Checks use each parameter's actual device, group precision, and Gram dispatch result.

## Automatic dispatch and autotuning

`precision="auto"` dispatches using the last two dimensions:

1. With `use_gram=True` and aspect ratio ≥ `gram_aspect_threshold` (default 5.0): FP16 Gram.
2. Otherwise, fewer than 1048576 matrix elements: FP16 regular.
3. Otherwise: global INT8, with an FP16 first stage by default.

Explicit precisions use the same Gram decision. Floating-point requests retain their precision; quantized requests use FP16 Gram. Gram selection has no additional element-count threshold. `use_gram=False` disables Gram, including in auto mode. Routing, precision aliases, and hardware checks live in `_optimizer.py`; coefficient tables and recommendations live in `_coeff.py`.

`autotune=True` times precompiled tiles, caches operation choices, then compares the complete pipeline. Tuned combinations are used only when the full pipeline improves. Autotuning does not refit coefficients or change precision routes. Initial tuning and compilation must precede CUDA Graph capture; `Muon(use_cuda_graph=True)` warms up before capture automatically. Autotuning and automatic dispatch are independent features.

## CIFAR-10 MLP example

[examples/cifar10.py](examples/cifar10.py) uses a three-layer MLP without biases: `3072 → 2048 → 2048 → 10`, with GELU hidden activations. Defaults are batch=128, 50 epochs, lr=1e-3, momentum=(0.95,0.95), weight_decay=0.01, and cosine learning-rate scheduling. Each epoch prints training loss, training/test accuracy, and elapsed time.

```bash
python -m pip install '.[examples]'
nvidia-smi
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python examples/cifar10.py
```

The example has no command-line arguments. Hyperparameters are at the top of the file; precision is set directly in `Muon(...)` and defaults to INT8. Running downloads data into `data/`. Importing the script neither downloads data nor starts training.

Source distributions contain `examples/`; wheel installations place examples under the environment's `share/i8muon/examples/`.

## Direct calls

```python
from i8muon import CutlassNSInt8

engine = CutlassNSInt8(autotune=False, fp16_first=True)
y = engine._regular_i8(x)                        # 1 FP16 + 4 INT8
y = engine._regular_prec(x, precision="float16") # FP16 NS
y = engine._gram_prec(x, precision="float16")    # FP16 Gram NS
y = engine._regular_bq(x, precision="int8")      # INT8 BQ
y = engine._regular_bq(x, precision="fp8_bq")    # FP8 BQ, SM89+
```

See [docs/CUTLASS.md](docs/CUTLASS.md) for layouts and implementation.
