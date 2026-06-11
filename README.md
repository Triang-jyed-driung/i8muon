# i8muon: Int8-Accelerated Muon Optimizer

Muon optimizer has recently attracted considerable attention. However, until now, Muon has relied on general matrix multiplication (GEMM), which is not especially efficient for symmetric matrix multiplication. Recently, Tri Dao et al. demonstrated in their blog post that fp16 symmetric-matrix operations combined with the Gram method can yield significant speedups. Here we show that int8-precision Muon is equally feasible.

A GPU-accelerated implementation of the **Muon optimizer** (Keller Jordan, 2024) that replaces the standard Newton-Schulz orthogonalization step with **int8 precision** written in **TileLang**, delivering faster per-step throughput while maintaining accuracy parity with the float reference.

## Training run
<img width="1641" height="871" alt="image" src="https://github.com/user-attachments/assets/fda52263-d210-4b1f-b2b7-48aeea7e92d4" />
A model with Qwen3 architecture, 8 layers, dimension 1024 trained under the int8, fp8 and fp16 Muon. All stable and no spikes.

## i8muon int8 vs [Gram-Newton-Schulz](https://github.com/Dao-AILab/gram-newton-schulz) — RTX 5090

Both methods use Newton-Schulz iterations to orthogonalize matrices.
- **i8muon** applies TileLang autotuned int8 kernels with CUDA graphs.
- **GNS** uses CuTeDSL symmetric GEMM kernels with `torch.compile` (reduce-overhead) and CUDA graphs.
- **GNS best** = `min(GNS gram, GNS standard)` per shape.
- **Speedup** = GNS best / i8muon int8 (>1 means i8muon is faster).
- All timings are CUDA-event min over 10 measurements after 3 warmup iterations.

**Summary:** i8muon int8 wins 16 of 22 shapes. Total time: 16.10 ms vs 22.02 ms — a **1.37×** overall speedup. GNS holds a slight edge on tall-skinny shapes where M ≫ N (e.g., 2048×512, 3072×1024).

| Shape | i8muon int8 (ms) | GNS best (ms) | GNS standard (ms) | Speedup |
|-------|-----------------:|--------------:|------------------:|-------------:|
| 512×512 | 0.1082 | **0.1055** | **0.1055** | 0.98x |
| 768×768 | **0.1287** | 0.1356 | 0.1364 | 1.05x |
| 512×1536 | **0.1257** | 0.1301 | 0.1367 | 1.04x |
| 1536×512 | 0.1430 | **0.1305** | 0.1359 | 0.91x |
| 512×2048 | **0.1353** | 0.1395 | 0.1543 | 1.03x |
| 2048×512 | 0.1560 | **0.1377** | 0.1541 | 0.88x |
| 1024×1024 | **0.1498** | 0.1750 | 0.1750 | 1.17x |
| 768×2048 | **0.1733** | 0.1870 | 0.2258 | 1.08x |
| 2048×768 | 0.2036 | **0.1868** | 0.2258 | 0.92x |
| 1536×1536 | **0.2561** | 0.4347 | 0.4347 | 1.70x |
| 1024×3072 | **0.2498** | 0.2628 | 0.3410 | 1.05x |
| 3072×1024 | 0.3101 | **0.2620** | 0.3408 | 0.84x |
| 2048×2048 | **0.4405** | 1.0492 | 1.0520 | 2.38x |
| 1536×4096 | **0.5096** | 0.6492 | 0.8704 | 1.27x |
| 4096×1536 | **0.6267** | 0.6553 | 0.8762 | 1.05x |
| 2560×2560 | **0.8700** | 1.6445 | 1.6486 | 1.89x |
| 2048×8192 | **1.1940** | 1.8760 | 2.9338 | 1.57x |
| 8192×2048 | **1.4848** | 1.8780 | 2.9348 | 1.26x |
| 2560×7168 | **1.6568** | 2.6460 | 3.8379 | 1.60x |
| 7168×2560 | **2.0070** | 2.6481 | 3.8308 | 1.32x |
| 2560×10240 | **2.3326** | 3.3444 | 5.2521 | 1.43x |
| 10240×2560 | **2.8416** | 3.3444 | 5.2449 | 1.18x |

**Key observations:**
- Near-tie on tiny matrices (512², 768²); both around 0.1 ms.
- i8muon pulls ahead at ≥1024² and dominates on large square shapes (2048²: 2.38×, 2560²: 1.89×).

## Background

### Muon Optimizer

Muon is a optimizer designed for the hidden-layer weight matrices of neural networks. Its core idea: after accumulating gradient momentum, apply **Newton-Schulz orthogonalization** to the update direction, then step. The orthogonalization enforces an approximate spectral norm of 1 on the update, which stabilizes training for large models.

A single Muon step (simplified) is:

```
g_t      = gradient
B_t      = mu * B_{t-1} + g_t                      (momentum)
O_t      = NewtonSchulz(B_t)                       (orthogonalize)
theta_t  = theta_{t-1} - adjusted_lr * O_t         (apply, with optional weight decay)
```

### Newton-Schulz Orthogonalization

Given a matrix X (shape M x N), Newton-Schulz iterates:

```
R = X @ X^T
Z = a*I + b*R + c*R^2            (polynomial in the Gram matrix)
X = Z @ X
```

After several iterations with specific coefficients (a, b, c), X converges to a matrix with orthonormal rows, satisfying `X @ X^T ≈ I`.

The Gram Newton-Schulz variant adds periodic "restart" steps where the Gram matrix is recomputed from the updated X, then the polynomial is applied to the new Gram. This reduces numerical error at the cost of recomputation.

### Int8 Acceleration

Both formulations are dominated by matrix multiplications (X @ X^T, Z @ X, etc.). This project quantizes the matrices to int8, uses **NVIDIA Tensor Cores** (`mma` instructions) for the GEMM operations, and manages scale factors to reconstruct float32 results. The core kernels are written in TileLang and JIT-compiled to CUDA.

## Architecture

```
                  Muon Optimizer (i8muon.py)
                              |
                     _single_tensor_muon()
                              |
            +-----------------+-----------------+
            |                 |                 |
       precision='int8'   precision='auto'   precision='float16'
       & use_gram=False   (auto-detect)      or bfloat16
            |                 |                 |
      _regular_i8()      selects path        _regular_prec()
      (int8 NS)          based on shape      (fp16 NS)
                              |
                    NSInt8 Engine (_NS.py)
                              |
              TileLang JIT Kernels (_kernels.py)
                         15 CUDA kernels
```

### Routing Logic

The optimizer selects the Newton-Schulz implementation at each step based on:

1. **Precision**: `int8`, `float16`, `bfloat16`, `float32` or `auto`. In `auto` mode, matrices with max(M, N) >= 256 use int8; smaller matrices use float16.

2. **Gram vs Regular**: When `use_gram=True` and the matrix aspect ratio `max(M,N)/min(M,N) >= gram_aspect_threshold` (default 4.0), the Gram formulation is selected. However, due to numerical stability concerns, the int8 Gram path is blocked: if both Gram and int8 apply, the system falls back to the float16 Gram path (`_gram_prec`). The regular int8 path (`_regular_i8`) is safe and actively used.

3. **CUDA Graph**: When `use_cuda_graph=True`, the NS kernel chain is captured into a CUDA graph on the first step. Subsequent steps replay the graph for reduced launch overhead. Each unique (method, precision, M, N) tuple gets its own graph.

## Key Components

### 1. `Muon` Optimizer (`i8muon.py`)

PyTorch `Optimizer` subclass. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `lr` | 0.01 | Learning rate |
| `weight_decay` | 0.0 | Weight decay (decoupled) |
| `momentum` | (0.95, 0.95) | Single float for plain momentum; 2-tuple enables Nesterov |
| `ns_coefficients` | None | Custom (a,b,c) triples; None uses defaults |
| `ns_steps` | None | Number of NS iterations; overrides default 5 when set |
| `eps` | 1e-7 | Epsilon for norm clamping |
| `adjust_lr_fn` | "spectral" | LR adjustment: "spectral", "original", "match_rms_adamw", or callable |
| `precision` | "auto" | "int8", "float16", "bfloat16", "float32", or "auto" |
| `autotune` | False | Run TileLang autotuner on kernel configs at init |
| `use_gram` | True | Enable Gram Newton-Schulz for high-aspect-ratio matrices |
| `use_cuda_graph` | False | Capture NS kernels into CUDA graph |
| `gram_aspect_threshold` | 4.0 | Aspect ratio above which Gram path is preferred |
| `deterministic` | True | Use PyTorch GPU norm for scale factor (reproducible). Set False for faster fused kernel |

**Interaction between `autotune` and `deterministic`.** When both are enabled (`autotune=True`, `deterministic=True`), the result is *relatively* deterministic: after tuning completes, the selected optimal kernel configurations remain fixed across runs, so subsequent steps produce bit-identical results for the same input. To guarantee *absolute* determinism independent of tuning, set `autotune=False`.

**Warning**: set `deterministic=False` in multi-GPU DDP training results in divergent behavior.

Only **2D parameters** are supported (typical for linear layer weight matrices). If necessary, you may create a different view on tensors with different numbers of dimensions.

### 2. `NSInt8` Engine (`i8muon_NS.py`)

The computational engine providing four orthogonalization methods:

| Method | Precision | Algorithm | Status |
|---|---|---|---|
| `_regular_i8` | int8 | Standard NS, int8 Tensor Cores | Primary production path |
| `_regular_prec` | fp16/bf16 | Standard NS, fp16 arithmetic | Fallback for small matrices |
| `_gram_prec` | fp16/bf16 | Gram NS, fp16 arithmetic | Used for high-aspect-ratio matrices |
| `_gram_i8` | int8 | Gram NS, int8 Tensor Cores | **Deprecated. Numerically unstable in long training. Blocked at routing level.** |

The engine uses lazy kernel instantiation via `__getattr__`: a kernel for a given shape is compiled only on first use, then cached. When `autotune=True`, the TileLang autotuner benchmarks all candidate tile configurations for that shape and selects the fastest.

### 3. TileLang Kernels (`i8muon_kernels.py`)

Fifteen custom CUDA kernels written in the TileLang DSL. They fall into several categories:

**Int8 quantization:**
- `_sumsq_maxabs`: Parallel reduction computing max absolute value and sum of squares of a float matrix.
- `_scale_int8`: Quantizes float to int8, computing the scale factor as `max / (127 * sqrt(sumsq))`.

**Symmetric int8 matrix operations (triangular block layout):**
- `_aat_int8_max`: Computes lower triangle of `A @ A^T` in int8 Tensor Cores, tracking the max absolute off-diagonal value. Only the lower triangle is computed and stored.
- `_int32_compl_symm_int8`: Completes a symmetric int8 matrix from its int32 lower triangle. Fills the upper triangle by transposition, extracts the diagonal as float32, and zeroes the diagonal in the int8 off-diagonal matrix.
- `_typeii_int8_sq`: Computes the polynomial `Z = a*I + b*A + c*A^2` directly from the symmetric int8 (Type2) representation, avoiding materialization to float.
- `_float32_compl_symm_int8_quad`: Completes a symmetric int8 matrix from a float32 lower triangle. Unlike `_int32_compl_symm_int8`, it simultaneously applies the quadratic polynomial (a,b,c) to the diagonal.
- `_typeii_int8_ab`: Computes `A @ B` for two **symmetric commutative** int8 matrices.
- `_float32_ab_to_int8`: Converts a float32 symmetric matrix to int8 symmetric representation.
- `_typeii_typei_int8`: Computes `A @ B` where A is symmetric int8 and B is general int8.
- `_float32_to_int8`: Converts a general float32 matrix to int8.

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/1f6d1300-a00f-4980-a32d-6be405cd298e" />

**Floating point precision kernels:**
- `_to_prec`: Normalizes and casts to fp16/bf16.
- `_ab_prec`: General matrix multiplication in fp16/bf16.
- `_aat_prec`: Symmetric `A @ A^T` in fp16/bf16.
- `_quad_prec`: Quadratic polynomial `Z = a*I + b*A + c*A^2` in fp16/bf16.
- `_ab_symm_prec`: Symmetric matrix multiplication `A @ B` in fp16/bf16.

All kernels take some of `M`, `N`, `K` as runtime arguments and are compiled on first use by TileLang's JIT.

### 4. Triangular Block Layout

For symmetric matrix operations (the Gram matrix `X @ X^T` and all subsequent Type2 operations), only the **lower triangle** of the result is computed. For an M x M matrix divided into `U = ceil(M / BLOCK_M)` blocks per dimension, the full grid would have `U * U` blocks. The triangular layout uses only:

```
total_blocks = (U * (U - 1) // 2) * R + ceil(M / BLOCK_N)
```

where `R = BLOCK_M / BLOCK_N` is the inner ratio when the block is rectangular. Each PID is mapped to a `(row, col)` position using the inverse triangular number formula:

```
pid_m = floor(sqrt(8*base_pid + 1) * 0.5 - 0.5)
pid_n = (base_pid - pid_m*(pid_m+1)/2) * R + (pid % R)
```

This reduces kernel launch count by approximately 50% for symmetric operations. Combined with int8 quantization (4x memory reduction vs float32), this layout significantly improves GPU occupancy and reduces memory bandwidth pressure.

When two kernels operate on the same symmetric int8 matrix (e.g., one computing the triangular GEMM and the next completing or reducing it), both kernels must agree on the block tiling. The second completion kernel must satisfy `BLOCK_M >= BLOCK_N`, and both block dimensions must be factors of `lcm(BLOCK_M, BLOCK_N)` of the first kernel. The autotuner respects these constraints automatically through the config filter conditions defined in `_make_configs`.


<img width="868" height="797" alt="image" src="https://github.com/user-attachments/assets/c5bbe118-9993-47e6-9719-160afa17d68f" />


### 5. CUDA Graph Support

When `use_cuda_graph=True`, the optimizer captures the NS kernel chain into a `torch.cuda.CUDAGraph` on the first step. The fixed-address input buffer is allocated and reused; each step copies the update into the input buffer then replays the graph. This eliminates per-step kernel launch overhead, which is most beneficial when the NS computation itself is fast (e.g., small matrices or few iterations). Each unique `(method, precision, M, N)` gets its own graph.

## File Layout

```
i8muon/
  i8muon.py               Muon optimizer class (main API)
  i8muon_NS.py             NSInt8 engine, autotune infra, coefficient utilities
  i8muon_kernels.py        TileLang CUDA kernel definitions (15 kernels)
  i8muon_unit_test.py      Unit tests: correctness, CUDA graphs, optimizer steps
  cifar10.py               CIFAR-10 demo: 3-layer MLP, 2048 hidden, Muon training
  mechanism/
    i8muon_naive.py        Pure-PyTorch int8 NS using torch._int_mm (mechanism reference)
    muon_coeff.py          Script that fits Newton-Schulz polynomial coefficients
  README.md
```

### `mechanism/i8muon_naive.py`

A pure-PyTorch reference implementation for understanding and accuracy comparison. Defines `Type1` (general int8 matrix with scale) and `Type2` (symmetric int8 matrix: diagonal float + off-diagonal int8 + scale) abstractions, along with matrix operations (`Type1_aat`, `Type2_sq`, `Type2_ab`, `Type2_typei`) using `torch._int_mm` for int8 GEMM. Not used in the production optimizer; the TileLang kernels manually manage buffers for performance.

### `mechanism/muon_coeff.py`

The script used to derive the default Newton-Schulz coefficients. It optimizes 15 parameters (5 triples of a,b,c) to make the composition of 5 quintic polynomials approximate the function `f(x) = sgn(x)` over the domain where Newton-Schulz operates (singular values near 1). The optimization uses AdamW with log-normal sampling of the input domain, and the resulting coefficients match `_DEFAULT_NS_COEFFS`.

## Newton-Schulz Coefficients

Default coefficients (5 iterations, int8-optimized):

```
Iter 0: (3.9274, -8.7643,  5.3095)
Iter 1: (3.4317, -5.4288,  2.3608)
Iter 2: (3.5403, -5.3366,  2.2324)
Iter 3: (3.6733, -4.8533,  1.8498)
Iter 4: (2.6731, -2.4447,  0.7695)
```

For more than 5 iterations, `recommend_coefficients(precision=..., iters=N)` inserts:
- Additional copies of `(3.486, -5.3827, 2.2966)` (iteration 2) for faster convergence.
- When `precision=True` and `iters >= 7`, an extra `(2.026, -1.513, 0.483)` step for higher precision (not recommended for int8 due to limited int8 dynamic range).

The `_gram_*` methods additionally scale all coefficients by 0.997 per iteration as a numerical stabilization measure for the Gram formulation. This scaling is not applied in the regular formulation.

## Installation

Requirements:
- Python >= 3.10 (tested and verified on 3.12, 3.13 and 3.14)
- PyTorch >= 2.8 with CUDA
- TileLang >= 0.1.10
- NVIDIA GPU with Compute Capability >= 7.5 (Turing or newer, for int8 Tensor Cores)

```
pip install torch
pip install tilelang
```

**Critical note on TileLang version.** During development, four distinct TileLang bugs (https://github.com/tile-ai/tilelang/issues/2053, https://github.com/tile-ai/tilelang/issues/2081, https://github.com/tile-ai/tilelang/issues/2172, https://github.com/tile-ai/tilelang/issues/2200) were encountered. Some could produce silently incorrect numerical results in int8 matrix kernels. Most were fixed in TileLang >= 0.1.10, though certain shapes for int8 (e.g., 128x257) remain to be solved. Using an older version will cause missing features as well as computation errors. Installing the latest version from PyPI or directly from the GitHub repository is strongly recommended.

## Usage

### Basic

```python
from i8muon import Muon

optimizer = Muon(
    model.parameters(),
    lr=0.01,
    momentum=(0.95, 0.95),
    weight_decay=0.01,
    precision="int8",
    autotune=False,        # Set to `True` to select best kernel configs at 10-minute one-time cost
    use_cuda_graph=True,   # reduce launch overhead
)
```

### With Learning Rate Scheduler

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

### Float-Only

```python
optimizer = Muon(
    model.parameters(),
    lr=0.01,
    precision="float16",   # or "bfloat16" or "float32"
)
```

### Custom Coefficients

```python
from i8muon_NS import recommend_coefficients

coeffs = recommend_coefficients(precision=True, iters=7)  # 7-iteration high-precision
optimizer = Muon(model.parameters(), lr=0.001, ns_coefficients=coeffs)
```

## Running the CIFAR-10 Demo

```bash
python cifar10.py
```

This trains a 3-layer MLP (3072 -> 2048 -> 2048 -> 10, no bias, GELU activation) on CIFAR-10 for 50 epochs using Muon with int8 precision and CUDA graphs. The model is intentionally simple to provide clear differentiation between optimizers: Muon reaches approximately 61% test accuracy while AdamW with identical hyperparameters achieves approximately 58.5%. This gap is a useful quick check that the optimizer is functioning correctly.

Expected output (RTX 4090, ~5s/epoch):

```
Device: cuda | NVIDIA GeForce RTX 4090
Batch size: 128 | Hidden dim: 2048
Model Params: 12,597,248

Round     Loss   Train%   Test%   Time
----------------------------------------
    1   2.0573   33.01%   39.73%    6.0s
    2   1.6556   45.14%   47.83%    5.1s
    ...
   50   0.0001  100.00%   60.xx%    5.1s
```

## Running Tests

```bash
python i8muon_unit_test.py
```

Tests cover:
- **Correctness**: int8 NS output vs float32 reference (across small/medium/large condition numbers and various shapes)
- **CUDA Graph consistency**: Graph replay output matches direct call
- **Optimizer steps**: Full step correctness, graph vs no-graph equivalence, multi-parameter graphs
- **Condition number scan**: Degradation across condition numbers from exp(0.2) to exp(2.0)

## Known Limitations

1. **2D parameters only.** Muon is designed for linear layer weight matrices. 1D parameters (biases, LayerNorm weights) and parameters with dimension > 2 are not supported.

2. **`_gram_i8` is blocked.** The int8 Gram Newton-Schulz path is numerically unstable in sustained training. In a 1GB-scale training run, it caused numerical explosion in the final layer. It is excluded from the routing logic and should not be enabled.

3. **Autotune overhead.** When `autotune=True`, the first step triggers TileLang's autotuner to benchmark candidate tile configurations for each kernel family. The total duration is approximately 10 minutes, depending on CPU speed and the number of distinct matrix shapes in the model. For LLM training where only 4 to 6 distinct weight-matrix shapes exist, enabling autotune is recommended because the one-time cost is amortized over millions of steps. For models with many different shapes (e.g., heterogeneous architectures with dozens of unique layer dimensions), the overhead may not be worth it. Set `autotune=False` to skip tuning and use default tile sizes instead.

4. **No tensor-parallel distributed support yet.** The optimizer itself is compatible with standard PyTorch DDP (each rank runs its own Muon step independently), but the CIFAR-10 demo and test suite are single-GPU.

## References

- Keller Jordan. **Muon: An optimizer for hidden layers in neural networks**. https://kellerjordan.github.io/posts/muon/, 2024
- **TileLang**: https://github.com/tile-ai/tilelang
- Tri Dao. **Gram Newton-Schulz: A Fast, Hardware-Aware Newton-Schulz Algorithm for Muon**. https://dao-ailab.github.io/blog/2026/gram-newton-schulz/
