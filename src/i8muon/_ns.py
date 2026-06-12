import os
import itertools
import warnings
import tilelang
from tilelang.autotuner import set_autotune_inputs
import torch
from ._kernels import *
from functools import cache, lru_cache

DISABLE_TMA = not int(os.environ.get("I8MUON_USE_TMA", '1'))
GRAM_ORDER = os.environ.get("I8MUON_GRAM_RESET", '00100')

# warnings.warn(f"I8MUON: disable TMA and warp specialization: {DISABLE_TMA}")
if not DISABLE_TMA:
    warnings.warn(
"""
TileLang's TMA and warp specialization features are experimental.

If you're using TileLang v0.1.11, it is strongly recommended to disable this option:
    export I8MUON_USE_TMA=0

Known issues by version:

  TileLang 0.1.11 on RTX 5090:
    - Incorrect results across nearly all int8 and block quantization kernels.
    - Kernel hangs (sync errors) under almost all block quantization configurations.
    - The consumer thread layout changes between versions, and TileLang does not
      expose any API for querying relative thread IDs within a consumer group.
    - Do not use TMA or warp specialization on this version. export I8MUON_USE_TMA=0
    - Disabling TMA and WS has a 10-15% performance penalty, but should run correctly.

  TileLang 0.1.10 on RTX 5090 (currently the most stable version):
    - Kernel hangs (sync errors) in some block quantization configurations.
      Setting autotune=False avoids most issues, but the default parameters may
      still fail on certain shapes.
    - Do not enable autotuning with block quantization kernels.

  TileLang 0.1.9:
    - Kernel hangs (sync errors) in some fp16/bf16 (*_prec) kernels on RTX 5090.
    - Incorrect int8 matrix multiplication results on RTX 4090 for certain shapes,
      caused by an off-by-one error in cp.async stage calculation.
    - Upgrade to TileLang 0.1.10.

  TileLang 0.1.8 and lower:
    - Some features like tile transpose are missing.
    - Upgrade to TileLang 0.1.10.
"""
    )

_CONF = {
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: DISABLE_TMA,
}


# pyright: reportAttributeAccessIssue=false

def _cdiv(a, b): return -(-a // b)

_DEFAULT_NS_COEFFS = [
    (3.9274, -8.7643,  5.3095),
    (3.4317, -5.4288,  2.3608),
    # (3.486, -5.3827, 2.2966), # (repeat this line for faster convergence)
    (3.5403, -5.3366,  2.2324),
    (3.6733, -4.8533,  1.8498),
    (2.6731, -2.4447,  0.7695),
    # (2.026, -1.513, 0.483)
    # (repeat this line to improve precision; not recommended for int8, because the precision is limited)
]

def recommend_coefficients(precision=True, iters=5):
    r = iters - 5 - (precision if iters >= 7 else 0)
    return [
        (3.9274, -8.7643,  5.3095),
        (3.4317, -5.4288,  2.3608)
    ] + [
        (3.486, -5.3827, 2.2966) 
    ] * max(r, 0) + [
        (3.5403, -5.3366,  2.2324),
        (3.6733, -4.8533,  1.8498),
        (2.6731, -2.4447,  0.7695),
    ] + ([
        (2.026, -1.513, 0.483)
    ] if precision and iters>=7 else [])

def _make_configs(condition=lambda x: 1, **kwargs):
    s = [dict(zip(kwargs.keys(), c)) for c in itertools.product(*kwargs.values())]
    return [c for c in s if condition(c)]


_MN_mem_configs = _make_configs(
    BLOCK_M=[8, 16, 32, 64],
    BLOCK_N=[16, 64],
    threads=[64, 128, 256, 512],
    condition=lambda x: (x["BLOCK_M"] * x["BLOCK_N"] >= x["threads"])
)

_MN_mem_configs_packed = _make_configs(
    BLOCK_M=[16, 32, 64],
    BLOCK_N=[16, 64],
    threads=[64, 128, 256, 512],
    condition=lambda x: (x["BLOCK_M"] * x["BLOCK_N"] >= x["threads"] and x["BLOCK_M"] >= x["BLOCK_N"])
)

_MNK_configs = _make_configs(
    BLOCK_M=[128, 256],
    BLOCK_N=[64, 128],
    BLOCK_K=[128, 256],
    threads=[128, 256],
    num_stages=[2, 3],
    condition=lambda x:
        max(x["BLOCK_M"], x["BLOCK_N"]) >= 64
        and (x["BLOCK_M"] + x["BLOCK_N"]) * x["BLOCK_K"] * x["num_stages"] <= 98304
)

_MNK_configs_prec = _make_configs(
    BLOCK_M=[32, 64, 128],
    BLOCK_N=[32, 64],
    BLOCK_K=[32, 64, 128],
    threads=[128, 256],
    num_stages=[2, 3],
    condition=lambda x:
        # max(x["BLOCK_M"], x["BLOCK_N"]) >= 64 and 
        (x["BLOCK_M"] + x["BLOCK_N"]) * x["BLOCK_K"] * x["num_stages"] <= 49152
)

_MNK_configs_packed = _make_configs(
    BLOCK_M=[128, 256],
    BLOCK_N=[64, 128],
    BLOCK_K=[128, 256],
    threads=[128, 256],
    num_stages=[2, 3],
    condition=lambda x:
        (x["BLOCK_M"] + x["BLOCK_N"]) * x["BLOCK_K"] * x["num_stages"] <= 98304
        and x["BLOCK_M"] >= x["BLOCK_N"]
)

_MNK_configs_packed_prec = _make_configs(
    BLOCK_M=[32, 64, 128],
    BLOCK_N=[32, 64],
    BLOCK_K=[64, 128],
    threads=[128, 256],
    num_stages=[2, 3],
    condition=lambda x:
        max(x["BLOCK_M"], x["BLOCK_N"]) >= 32
        and (x["BLOCK_M"] + x["BLOCK_N"]) * x["BLOCK_K"] * x["num_stages"] <= 49152
        and x["BLOCK_M"] >= x["BLOCK_N"]
)

_BLOCKQ_gemm_configs = _make_configs(
    threads=[64, 128, 256, 512],
    num_stages=[0, 1, 2, 3],
)

_BLOCKQ_mem_configs = _make_configs(
    threads=[128, 256, 512],
)

tune_mem = tilelang.autotune(
                configs=_MN_mem_configs, warmup=1, rep=3, timeout=1
            )

tune_mem_packed = tilelang.autotune(
                configs=_MN_mem_configs_packed, warmup=1, rep=3, timeout=1
            )

tune_gemm = tilelang.autotune(
                configs=_MNK_configs, warmup=1, rep=3, timeout=2
            )

tune_gemm_prec = tilelang.autotune(
                configs=_MNK_configs_prec, warmup=1, rep=3, timeout=4
            )

tune_gemm_packed = tilelang.autotune(
                configs=_MNK_configs_packed, warmup=1, rep=3, timeout=2
            )

tune_gemm_packed_prec = tilelang.autotune(
                configs=_MNK_configs_packed_prec, warmup=1, rep=3, timeout=4
            )

tune_gemm_blockq = tilelang.autotune(
                configs=_BLOCKQ_gemm_configs, warmup=1, rep=3, timeout=4
            )

tune_mem_blockq = tilelang.autotune(
                configs=_BLOCKQ_mem_configs, warmup=1, rep=3, timeout=4
            )

kernel_map = [
    (_sumsq_maxabs, tune_mem),
    (_scale_int8, tune_mem),
    (_scale_int8_transpose_out, tune_mem),
    (_aat_int8_max, tune_gemm_packed),
    (_int32_compl_symm_int8, tune_mem_packed),
    (_typeii_int8_sq, tune_gemm_packed),
    (_float32_compl_symm_int8_quad, tune_mem_packed),
    (_typeii_int8_ab, tune_gemm_packed),
    (_float32_ab_to_int8, tune_mem_packed),
    (_typeii_typei_int8, tune_gemm),
    (_typeii_typei_int8_transpose_out, tune_gemm),
    (_float32_to_int8, tune_mem),
    (_to_prec, tune_mem),
    (_to_prec_transpose_out, tune_mem),
    (_ab_prec, tune_gemm_prec),
    (_ab_prec_transpose_out, tune_gemm_prec),
    (_aat_prec, tune_gemm_packed_prec),
    (_quad_prec, tune_gemm_packed_prec),
    (_ab_symm_prec, tune_gemm_packed_prec),
    (_to_bq, tune_mem_blockq),
    (_to_bq_transpose_out, tune_mem_blockq),
    (_aat_bq, tune_gemm_blockq),
    (_quad_bq, tune_gemm_blockq),
    (_typeii_typei_bq, tune_gemm_blockq),
    (_typeii_typei_final_bq, tune_gemm_blockq),
    (_typeii_typei_final_bq_transpose_out, tune_gemm_blockq),
    (_ab_symm_bq, tune_gemm_blockq),
]

def _prec2dtype(prec: str):
    return getattr(torch, prec)

def _to_batched(X: torch.Tensor):
    """Reshape X to 3D (B,ROW,COL).  Returns (X, orig_shape)."""
    assert X.ndim >= 2
    X = X.contiguous()
    orig_shape = X.shape
    if X.ndim == 2:
        X = X.unsqueeze(0)
    elif X.ndim > 3:
        X = X.view(-1, X.shape[-2], X.shape[-1])
    return X, orig_shape


def _from_batched(result: torch.Tensor, orig_shape: torch.Size):
    """Reshape result back to orig_shape if needed."""
    if result.shape != orig_shape:
        return result.view(orig_shape)
    return result


class NSInt8:
    def __init__(self, autotune: bool = False):
        self.autotune = autotune
        for (fn, tuner) in kernel_map:
            setattr(
                self, '_Z_'+fn.__name__, 
                tuner(tilelang.jit(fn, pass_configs=_CONF)) 
                if autotune else 
                tilelang.jit(fn, pass_configs=_CONF)
            )

    def __getattr__(self, name):
        @cache
        def kernel_factory(*args, **kwargs):
            kernel = None
            def runner(*a1, **k1):
                nonlocal kernel
                if kernel is None:
                    if self.autotune:
                        with set_autotune_inputs(*a1, **k1):
                            kernel = getattr(self, '_Z_'+name)(*args, **kwargs)
                    else:
                        kernel = getattr(self, '_Z_'+name)(*args, **kwargs)
                return kernel(*a1, **k1)
            return runner
        setattr(self, name, kernel_factory)
        return kernel_factory
    
    def _gram_i8(
        self,
        X: torch.Tensor,
        coeffs: list | None = None,
        eps: float = 1e-7,
        deterministic: bool = True,
        precision: str = 'int8'
    ) -> torch.Tensor:
        r"""Gram-form int8 Newton-Schulz orthogonalisation."""

        raise RuntimeError("Int8 Gram Newton-Schulz is not precise enough. Temporarily disabled.")

    def _gram_prec(
        self,
        X: torch.Tensor,
        coeffs: list | None = None,
        eps: float = 1e-7,
        deterministic: bool = True,
        precision: str = 'float16'
    ) -> torch.Tensor:
        r"""Gram-form fp16 Newton-Schulz orthogonalisation."""
        if coeffs is None:
            coeffs = _DEFAULT_NS_COEFFS

        X, orig_shape = _to_batched(X)
        B, ROW, COL = X.shape
        L = min(ROW, COL)
        H = max(ROW, COL)
        dtype_str = str(X.dtype).split(".")[-1]
        dev = X.device
        prec = _prec2dtype(precision)

        transposed = ROW > COL
        A = torch.empty((B, L, H), device=dev, dtype=prec)
        R_mat = torch.empty((B, L, L), device=dev, dtype=prec)
        Y = torch.empty((B, L, H), device=dev, dtype=prec)
        Z = torch.empty((B, L, L), device=dev, dtype=prec)
        Q0 = torch.empty((B, L, L), device=dev, dtype=prec)
        Q1 = torch.empty((B, L, L), device=dev, dtype=prec)
        if transposed:
            self._to_prec_transpose_out(B=B, M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str,
                          dtype_out=precision, use_norm=True, eps=eps)(
                X, X.norm(dim=[1,2], keepdim=True).view(B), A)
        else:
            self._to_prec(B=B, M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str,
                          dtype_out=precision, use_norm=True, eps=eps)(
                X, X.norm(dim=[1,2], keepdim=True).view(B), A)

        n = len(coeffs)
        Ksym = dict(B=B, M=L, dtype=precision)
        Knsq = dict(B=B, M=L, N=H, K=L, dtype=precision)
        Kgram = dict(B=B, M=L, K=H, dtype=precision)
        
        for t, (a, b, c) in enumerate(coeffs):
            # a*=0.9995; b*=0.9995; c*=0.9995
            kind = GRAM_ORDER[t]
            nnr = t + 1 < n and GRAM_ORDER[t+1] == '0'
            if kind == '1':
                self._ab_prec(**Knsq)(Q0, A, Y)
                A, Y = Y, A
            if kind == '1' or t == 0:
                self._aat_prec(**Kgram)(A, R_mat)
                self._quad_prec(**Ksym)(R_mat, Q0, a, b, c)
                if nnr:
                    self._ab_symm_prec(**Ksym)(R_mat, Q0, Q1)
                    self._ab_symm_prec(**Ksym)(Q0, Q1, R_mat)
            else:
                self._quad_prec(**Ksym)(R_mat, Z, a, b, c)
                self._ab_symm_prec(**Ksym)(Z, Q0, Q1)
                Q0, Q1 = Q1, Q0
                if nnr:
                    self._ab_symm_prec(**Ksym)(R_mat, Z, Q1)
                    self._ab_symm_prec(**Ksym)(Z, Q1, R_mat)

        if transposed:
            Y_T = torch.empty((B, H, L), device=dev, dtype=prec)
            self._ab_prec_transpose_out(**Knsq)(Q0, A, Y_T)
            return _from_batched(Y_T, orig_shape).to(X.dtype)
        else:
            self._ab_prec(**Knsq)(Q0, A, Y)
            return _from_batched(Y, orig_shape).to(X.dtype)

    def _gram_bq(
        self,
        X: torch.Tensor,
        coeffs: list | None = None,
        eps: float = 1e-7,
        deterministic: bool = True,
        precision: str = 'float16'
    ) -> torch.Tensor:
        r"""Gram-form fp16 Newton-Schulz orthogonalisation."""

        raise RuntimeError("Block quantized Gram Newton-Schulz is not precise enough. Temporarily disabled.")

    # ═══════════════════════════════════════════════════════════════
    #  Public:  regular  (standard Newton-Schulz)
    # ═══════════════════════════════════════════════════════════════

    # @profile_calls()
    def _regular_i8(
        self,
        X: torch.Tensor,
        coeffs: list | None = None,
        eps: float = 1e-7,
        deterministic: bool = True,
        precision: str = 'int8'
    ) -> torch.Tensor:
        """Standard Newton-Schulz int8.  Accepts (M,N) or (B,M,N)."""
        if coeffs is None:
            coeffs = _DEFAULT_NS_COEFFS

        X, orig_shape = _to_batched(X)
        B, ROW, COL = X.shape
        dev = X.device
        dtype_str = str(X.dtype).split(".")[-1]
        L = min(ROW, COL)
        H = max(ROW, COL)

        atom = torch.zeros((8, B), device=dev)
        A_max = atom[0]
        A_square_sum = atom[1]
        AA_max = atom[2].view(torch.int32)
        B_max = atom[3]
        C_max = atom[4]
        transposed = ROW > COL
        A8 = torch.empty((B, L, H), device=dev, dtype=torch.int8)
        A_scale = torch.empty((B,), device=dev, dtype=torch.float32)
        AA32L = torch.empty((B, L, L), device=dev, dtype=torch.int32)
        AA8 = torch.empty((B, L, L), device=dev, dtype=torch.int8)
        AA_scale = torch.empty((B,), device=dev, dtype=torch.float32)
        AA_diag = torch.empty((B, L), device=dev, dtype=torch.float32)
        B32 = torch.empty((B, L, L), device=dev, dtype=torch.float32)
        B8 = torch.empty((B, L, L), device=dev, dtype=torch.int8)
        B_scale = torch.empty((B,), device=dev, dtype=torch.float32)
        B_diag = torch.empty((B, L), device=dev, dtype=torch.float32)
        C32 = torch.empty((B, L, H), device=dev, dtype=torch.float32)

        # ── prologue ──
        self._sumsq_maxabs(B=B, M=ROW, N=COL, dtype=dtype_str)(X, A_max, A_square_sum)
        A_frob_norm = X.norm(dim=[1, 2], keepdim=True).view(B) if deterministic else None
        if transposed:
            if deterministic:
                self._scale_int8_transpose_out(B=B, M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str,
                         use_norm=True, eps=eps)(X, A_max, A_frob_norm, A8, A_scale)
            else:
                self._scale_int8_transpose_out(B=B, M=ROW, N=COL, dtype=dtype_str)(
                    X, A_max, A_square_sum, A8, A_scale)
        else:
            if deterministic:
                self._scale_int8(B=B, M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str,
                         use_norm=True, eps=eps)(X, A_max, A_frob_norm, A8, A_scale)
            else:
                self._scale_int8(B=B, M=ROW, N=COL, dtype=dtype_str)(
                    X, A_max, A_square_sum, A8, A_scale)

        # ── main loop ──
        N_iter = len(coeffs)
        for i in range(N_iter):
            a, b, c = coeffs[i]
            self._aat_int8_max(B=B, M=L, K=H)(A8, AA32L, AA_max)
            self._int32_compl_symm_int8(B=B, M=L)(
                AA32L, AA_max, A_scale, AA8, AA_scale, AA_diag)
            self._typeii_int8_sq(B=B, M=L)(
                AA8, AA_scale, AA_diag, B32, B_max, b, c)
            self._float32_compl_symm_int8_quad(B=B, M=L)(
                B32, B_max, AA_diag, B8, B_scale, B_diag, a, b, c)
            self._typeii_typei_int8(B=B, M=L, N=H)(
                B8, B_scale, B_diag, A8, A_scale, C32, C_max)
            if i == N_iter - 1:
                if transposed:
                    C32_T = torch.empty((B, H, L), device=dev, dtype=torch.float32)
                    self._typeii_typei_int8_transpose_out(B=B, M=L, N=H)(
                        B8, B_scale, B_diag, A8, A_scale, C32_T, C_max)
                    result = C32_T.to(X.dtype)
                else:
                    result = C32.to(X.dtype)
                return _from_batched(result, orig_shape)
            else:
                self._float32_to_int8(B=B, M=L, N=H)(C32, C_max, A8, A_scale)
                atom.zero_()

        raise RuntimeError("unreachable")
        raise RuntimeError("unreachable")
    
    def _regular_prec(
        self,
        X: torch.Tensor,
        coeffs: list | None = None,
        eps: float = 1e-7,
        deterministic: bool = True,
        precision: str = 'float16'
    ) -> torch.Tensor:
        """Standard Newton-Schulz fp16.  Accepts (M,N) or (B,M,N)."""
        if coeffs is None:
            coeffs = _DEFAULT_NS_COEFFS

        X, orig_shape = _to_batched(X)
        B, ROW, COL = X.shape
        dev = X.device
        dtype_str = str(X.dtype).split(".")[-1]
        L = min(ROW, COL)
        H = max(ROW, COL)

        transposed = ROW > COL
        A = torch.empty((B, L, H), device=dev, dtype=_prec2dtype(precision))
        B_mat = torch.empty((B, L, L), device=dev, dtype=_prec2dtype(precision))
        C = torch.empty((B, L, H), device=dev, dtype=_prec2dtype(precision))
        AA = torch.as_strided(C, (B, L, L), (L * L, L, 1))
        AAA = torch.as_strided(A, (B, L, L), (L * L, L, 1))

        A_frob_norm = X.norm(dim=[1, 2], keepdim=True).view(B)
        if transposed:
            self._to_prec_transpose_out(B=B, M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str,
                dtype_out=precision, use_norm=True, eps=eps)(X, A_frob_norm, A)
        else:
            self._to_prec(B=B, M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str,
                dtype_out=precision, use_norm=True, eps=eps)(X, A_frob_norm, A)

        N_iter = len(coeffs)
        for i in range(N_iter):
            a, b, c = coeffs[i]
            self._aat_prec(B=B, M=L, K=H, dtype=precision)(A, AA)
            self._quad_prec(B=B, M=L, dtype=precision)(AA, B_mat, a, b, c)
            if i == N_iter - 1:
                if transposed:
                    C_T = torch.empty((B, H, L), device=dev, dtype=_prec2dtype(precision))
                    self._ab_prec_transpose_out(B=B, M=L, K=L, N=H, dtype=precision)(B_mat, A, C_T)
                    result = C_T.to(X.dtype)
                else:
                    self._ab_prec(B=B, M=L, K=L, N=H, dtype=precision)(B_mat, A, C)
                    result = C.to(X.dtype)
                return _from_batched(result, orig_shape)
            else:
                self._ab_prec(B=B, M=L, K=L, N=H, dtype=precision)(B_mat, A, C)
                A, C = C, A
                AA, AAA = AAA, AA
        raise RuntimeError("unreachable")
    
    def _regular_bq(
        self,
        X: torch.Tensor,
        coeffs: list | None = None,
        eps: float = 1e-7,
        deterministic: bool = True,
        precision: str = 'float8_e4m3fn'
    ) -> torch.Tensor:

        if coeffs is None:
            coeffs = _DEFAULT_NS_COEFFS

        X, orig_shape = _to_batched(X)
        B, ROW, COL = X.shape
        dev = X.device
        dtype_str = str(X.dtype).split(".")[-1]
        L = min(ROW, COL)
        H = max(ROW, COL)
        LQ = _cdiv(L, BLOCK_Q)
        HQ = _cdiv(H, BLOCK_Q)
        
        transposed = ROW > COL
        A = torch.empty((B, L, H), device=dev, dtype=_prec2dtype(precision))
        A_scale = torch.empty((B, LQ, HQ), device=dev, dtype=torch.float32)

        B_mat = torch.empty((B, L, L), device=dev, dtype=_prec2dtype(precision))
        B_scale = torch.empty((B, LQ, LQ), device=dev, dtype=torch.float32)
        B_diag = torch.empty((B, L), device=dev, dtype=torch.float32)
        
        C = torch.empty((B, L, H), device=dev, dtype=_prec2dtype(precision))
        C_scale = torch.empty((B, LQ, HQ), device=dev, dtype=torch.float32)

        AA = torch.as_strided(C, (B, L, L), (L * L, L, 1))
        AA_scale = torch.as_strided(C_scale, (B, LQ, LQ), (LQ * LQ, LQ, 1))
        AA_diag = torch.empty((B, L), device=dev, dtype=torch.float32)
        
        AAA = torch.as_strided(A, (B, L, L), (L * L, L, 1))
        AAA_scale = torch.as_strided(A_scale, (B, LQ, LQ), (LQ * LQ, LQ, 1))

        A_frob_norm = X.norm(dim=[1, 2], keepdim=True).view(B)
        if transposed:
            self._to_bq_transpose_out(B=B, M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str,
                dtype_out=precision, use_norm=True, eps=eps)(X, A_frob_norm, A, A_scale)
        else:
            self._to_bq(B=B, M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str,
                dtype_out=precision, use_norm=True, eps=eps)(X, A_frob_norm, A, A_scale)

        N_iter = len(coeffs)
        for i in range(N_iter):
            a, b, c = coeffs[i]
            self._aat_bq(B=B, M=L, K=H, dtype=precision)(A, A_scale, AA, AA_scale, AA_diag)
            self._quad_bq(B=B, M=L, dtype=precision)(AA, AA_scale, AA_diag, B_mat, B_scale, B_diag, a, b, c)
            if i == N_iter - 1:
                if transposed:
                    Z_T = torch.empty((B, H, L), device=dev, dtype=torch.float32)
                    self._typeii_typei_final_bq_transpose_out(B=B, M=L, N=H, dtype=precision)(
                        B_mat, B_scale, B_diag, A, A_scale, Z_T)
                    result = Z_T.to(X.dtype)
                else:
                    Z = torch.empty((B, L, H), device=dev, dtype=torch.float32)
                    self._typeii_typei_final_bq(B=B, M=L, N=H, dtype=precision)(
                        B_mat, B_scale, B_diag, A, A_scale, Z)
                    result = Z.to(X.dtype)
                return _from_batched(result, orig_shape)
            else:
                self._typeii_typei_bq(B=B, M=L, N=H, dtype=precision)(B_mat, B_scale, B_diag, A, A_scale, C, C_scale)
                A, A_scale, C, C_scale = C, C_scale, A, A_scale
                AA, AA_scale, AAA, AAA_scale, = AAA, AAA_scale, AA, AA_scale

        raise RuntimeError("unreachable")
    
    @cache
    def router(self, M: int, N: int, precision: str, use_gram: bool, gram_aspect_threshold: float):
        numel = M * N
        aspect_ratio = max(M, N) / min(M, N)
        if precision == 'auto':
            if numel >= 524288 and aspect_ratio >= 7.0:
                return self._gram_prec, 'float16'
            elif numel < 1048576:
                return self._regular_prec, 'float16'
            else:
                return self._regular_i8, 'int8'
            
        else:
            gram_method = (aspect_ratio >= gram_aspect_threshold and use_gram)
            
            use_bq = False
            if precision == "float8_e4m3fn":
                use_bq = True
            elif precision == "int8_bq":
                precision = "int8"
                use_bq = True
            
            if gram_method:
                if precision in ('float8_e4m3fn', 'int8'):
                    precision = 'float16'
                return self._gram_prec, precision
            else:
                if use_bq:
                    return self._regular_bq, precision
                else:
                    if precision == 'int8':
                        return self._regular_i8, 'int8'
                    else:
                        return self._regular_prec, precision
