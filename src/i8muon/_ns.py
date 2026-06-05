

import itertools
import warnings
import tilelang
from tilelang.autotuner import set_autotune_inputs
import torch
from ._kernels import *
from functools import cache, lru_cache



_CONF = {tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True}


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

_BLOCKQ_gemm_configs = [
    {'threads':  64, 'num_stages': 1},
    {'threads': 128, 'num_stages': 1},
    {'threads': 256, 'num_stages': 1},
    {'threads': 512, 'num_stages': 1},
    {'threads':  64, 'num_stages': 2},
    {'threads': 128, 'num_stages': 2},
    {'threads': 256, 'num_stages': 2},
    {'threads':  64, 'num_stages': 3},
]

_BLOCKQ_mem_configs = _make_configs(
    threads=[128, 256, 512],
    condition=lambda x: 1
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
    (_aat_int8_max, tune_gemm_packed),
    (_int32_compl_symm_int8, tune_mem_packed),
    (_typeii_int8_sq, tune_gemm_packed),
    (_float32_compl_symm_int8_quad, tune_mem_packed),
    (_typeii_int8_ab, tune_gemm_packed),
    (_float32_ab_to_int8, tune_mem_packed),
    (_typeii_typei_int8, tune_gemm),
    (_float32_to_int8, tune_mem),
    (_to_prec, tune_mem),
    (_ab_prec, tune_gemm_prec),
    (_aat_prec, tune_gemm_packed_prec),
    (_quad_prec, tune_gemm_packed_prec),
    (_ab_symm_prec, tune_gemm_packed_prec),
    (_to_bq, tune_mem_blockq),
    (_aat_bq, tune_gemm_blockq),
    (_quad_bq, tune_gemm_blockq),
    (_typeii_typei_bq, tune_gemm_blockq),
    (_typeii_typei_final_bq, tune_gemm_blockq),
    (_ab_symm_bq, tune_gemm_blockq),
]

def _prec2dtype(prec: str):
    return getattr(torch, prec)

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
        if coeffs is None:
            coeffs = _DEFAULT_NS_COEFFS
        
        warnings.warn("Int8 Gram Newton-Schulz is not precise enough yet. Use at your own risk.")

        X = X.contiguous()
        ROW, COL = X.shape
        L = min(ROW, COL)
        H = max(ROW, COL)
        dtype_str = str(X.dtype).split(".")[-1]
        dev = X.device

        # ── pre-allocate ──
        A8 = torch.empty((ROW, COL), device=dev, dtype=torch.int8)
        A_scale = torch.empty((1,), device=dev, dtype=torch.float32)
        atom = torch.zeros((8, 1), device=dev)
        A_max = atom[0]
        A_square_sum = atom[1]
        AA_max = atom[2].view(torch.int32)
        B_max = atom[3]
        C_max = atom[4]
        B_max_1 = atom[5]
        B_max_2 = atom[6]
        B_max_3 = atom[7]
        AA32L = torch.empty((L, L), device=dev, dtype=torch.int32)
        R8 = torch.empty((L, L), device=dev, dtype=torch.int8)
        R_scale = torch.empty((1,), device=dev, dtype=torch.float32)
        R_diag = torch.empty((L,), device=dev, dtype=torch.float32)
        B32 = torch.empty((L, L), device=dev, dtype=torch.float32)
        Z8 = torch.empty((L, L), device=dev, dtype=torch.int8)
        Z_scale = torch.empty((1,), device=dev, dtype=torch.float32)
        Z_diag = torch.empty((L,), device=dev, dtype=torch.float32)
        Q8 = torch.empty((L, L), device=dev, dtype=torch.int8)
        Q_scale = torch.empty((1,), device=dev, dtype=torch.float32)
        Q_diag = torch.empty((L,), device=dev, dtype=torch.float32)
        RZ8 = torch.empty((L, L), device=dev, dtype=torch.int8)
        RZ_scale = torch.empty((1,), device=dev, dtype=torch.float32)
        RZ_diag = torch.empty((L,), device=dev, dtype=torch.float32)
        C32 = torch.empty((L, H), device=dev, dtype=torch.float32)

        # ── Step 1: Frobenius norm + int8 quantise ──
        self._sumsq_maxabs(M=ROW, N=COL, dtype=dtype_str)(X, A_max, A_square_sum)
        if deterministic:
            A_frob_norm = torch.linalg.vector_norm(X).view(1)
            self._scale_int8(M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str,
                     use_norm=True, eps=eps)(X, A_max, A_frob_norm, A8, A_scale)
        else:
            self._scale_int8(M=ROW, N=COL, dtype=dtype_str)(
                X, A_max, A_square_sum, A8, A_scale
            )

        # ── Step 2: Transpose trick: make it L×H with L = min ──
        transposed = ROW > COL
        if transposed:
            A8 = A8.mT.contiguous()

        # ── Step 3: R = A @ A^T (Gram matrix in Type2) ──
        self._aat_int8_max(M=L, K=H)(A8, AA32L, AA_max)
        self._int32_compl_symm_int8(M=L)(
            AA32L, AA_max, A_scale, R8, R_scale, R_diag
        )

        # ── Step 4: unrolled iterations ──
        #  kind: "i"(init t=0), "r"(restart), "n"(normal), "e"(end)
        #  init/restart: Q = Z,  then RZ = R·Q,  R = Q·RZ
        #  normal/end:   Z → Z8,  Q = Q·Z,  (normal only) RZ = R·Z, R = Z·RZ
        for t in range(len(coeffs)):
            a, b, c = coeffs[t]
            a *= 0.997
            b *= 0.997
            c *= 0.997
            kind = ("i" if t == 0
                    else "e" if t == len(coeffs) - 1
                    else "r" if (t + len(coeffs)) % 2 == 0
                    else "n")

            if kind == "r":
                atom.zero_()
                # X = Q @ X  (Type2 × Type1)
                self._typeii_typei_int8(M=L, N=H)(
                    Q8, Q_scale, Q_diag, A8, A_scale, C32, C_max
                )
                self._float32_to_int8(M=L, N=H)(C32, C_max, A8, A_scale)
                # R = X @ X^T  (recompute Gram)
                self._aat_int8_max(M=L, K=H)(A8, AA32L, AA_max)
                self._int32_compl_symm_int8(M=L)(
                    AA32L, AA_max, A_scale, R8, R_scale, R_diag
                )

            if kind in ("i", "r"):
                self._typeii_int8_sq(M=L)(
                    R8, R_scale, R_diag, B32, B_max, b, c
                )
                self._float32_compl_symm_int8_quad(M=L)(
                    B32, B_max, R_diag, Q8, Q_scale, Q_diag, a, b, c
                )
                # RZ = R @ Q  (Q holds Z)
                self._typeii_int8_ab(M=L)(
                    R8, R_scale, R_diag, Q8, Q_scale, Q_diag, B32, B_max_1
                )
                self._float32_ab_to_int8(M=L)(B32, B_max_1, RZ8, RZ_scale, RZ_diag)
                # R = Q @ RZ
                self._typeii_int8_ab(M=L)(
                    Q8, Q_scale, Q_diag, RZ8, RZ_scale, RZ_diag, B32, B_max_2
                )
                self._float32_ab_to_int8(M=L)(B32, B_max_2, R8, R_scale, R_diag)
            else:  # "n" or "e"
                atom.zero_()
                self._typeii_int8_sq(M=L)(
                    R8, R_scale, R_diag, B32, B_max, b, c
                )
                self._float32_compl_symm_int8_quad(M=L)(
                    B32, B_max, R_diag, Z8, Z_scale, Z_diag, a, b, c
                )
                # Q = Q @ Z  (Type2 × Type2)
                self._typeii_int8_ab(M=L)(
                    Z8, Z_scale, Z_diag, Q8, Q_scale, Q_diag, B32, B_max_1
                )
                self._float32_ab_to_int8(M=L)(B32, B_max_1, Q8, Q_scale, Q_diag)
                if kind == "n":
                    # RZ = R @ Z
                    self._typeii_int8_ab(M=L)(
                        R8, R_scale, R_diag,
                        Z8, Z_scale, Z_diag, B32, B_max_2,
                    )
                    self._float32_ab_to_int8(M=L)(B32, B_max_2, RZ8, RZ_scale, RZ_diag)
                    # R = Z @ RZ
                    self._typeii_int8_ab(M=L)(
                        Z8, Z_scale, Z_diag,
                        RZ8, RZ_scale, RZ_diag, B32, B_max_3,
                    )
                    self._float32_ab_to_int8(M=L)(B32, B_max_3, R8, R_scale, R_diag)

        # ── Step 5: X_out = Q_final @ X ──
        self._typeii_typei_int8(M=L, N=H)(
            Q8, Q_scale, Q_diag, A8, A_scale, C32, C_max
        )
        X_out = C32.to(X.dtype)

        # Undo transpose trick
        if transposed:
            X_out = X_out.mT.contiguous()

        return X_out

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

        X = X.contiguous()
        ROW, COL = X.shape
        L = min(ROW, COL)
        H = max(ROW, COL)
        dtype_str = str(X.dtype).split(".")[-1]
        dev = X.device

        # ── pre-allocate ──
        A = torch.empty((ROW, COL), device=dev, dtype=_prec2dtype(precision))
        R = torch.empty((L, L), device=dev, dtype=_prec2dtype(precision))
        Y = torch.empty((L, H), device=dev, dtype=_prec2dtype(precision))
        QQ = torch.as_strided(Y, (L, L), (L, 1))
        # QQ = torch.empty((L, L), device=dev, dtype=_prec2dtype(precision))
        Z = torch.empty((L, L), device=dev, dtype=_prec2dtype(precision))
        Q = torch.empty((L, L), device=dev, dtype=_prec2dtype(precision))
        # RZ = torch.empty((L, L), device=dev, dtype=_prec2dtype(precision))

        # ── Step 1: Frobenius norm + int8 quantise ──
        # self._sumsq_maxabs(M=ROW, N=COL, dtype=dtype_str)(X, A_max, A_square_sum)
        A_frob_norm = torch.linalg.vector_norm(X).view(1)
        self._to_prec(
            M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str, dtype_out=precision, use_norm=True, eps=eps,
        )(X, A_frob_norm, A)

        # ── Step 2: Transpose trick: make it L×H with L = min ──
        transposed = ROW > COL
        if transposed:
            A = A.mT.contiguous()
        QQA = torch.as_strided(A, (L, L), (L, 1))

        # ── Step 3: R = A @ A^T (Gram matrix in Type2) ──
        self._aat_prec(M=L, K=H, dtype=precision)(A, R)

        # ── Step 4: unrolled iterations ──
        #  kind: "i"(init t=0), "r"(restart), "n"(normal), "e"(end)
        for t in range(len(coeffs)):
            a, b, c = coeffs[t]
            a *= 0.997
            b *= 0.997
            c *= 0.997
            kind = ("i" if t == 0
                    else "e" if t == len(coeffs) - 1
                    else "r" if (t + len(coeffs)) % 2 == 0
                    else "n")

            if kind == "r":
                # Y = Q @ X
                self._ab_prec(M=L, N=H, K=L, dtype=precision)(Q, A, Y)
                # R = Y @ Y^T  (recompute Gram)
                self._aat_prec(M=L, K=H, dtype=precision)(Y, R)
                
                A, Y = Y, A 
                QQ, QQA = QQA, QQ

            if kind in ("i", "r"):
                # Q = Quad(R)
                self._quad_prec(M=L, dtype=precision)(R, Q, a, b, c)
                # RZ = R @ Q  (Q holds Z)
                self._ab_symm_prec(M=L, dtype=precision)(R, Q, QQ)
                # R = Q @ RZ
                self._ab_symm_prec(M=L, dtype=precision)(Q, QQ, R)
            else:  # "n" or "e"
                # Z = Quad(R)
                self._quad_prec(M=L, dtype=precision)(R, Z, a, b, c)
                # Q = Z @ Q 
                self._ab_symm_prec(M=L, dtype=precision)(Z, Q, QQ)
                Q.copy_(QQ)
                if kind == "n":
                    # RZ = R @ Z
                    self._ab_symm_prec(M=L, dtype=precision)(R, Z, QQ)
                    # R = Z @ RZ
                    self._ab_symm_prec(M=L, dtype=precision)(Z, QQ, R)

        # ── Step 5: X_out = Q_final @ X ──
        self._ab_prec(M=L, N=H, K=L, dtype=precision)(Q, A, Y)
        if transposed:
            Y = Y.mT.contiguous()
        X_out = Y.to(X.dtype)

        return X_out
    


    def _gram_bq(
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

        X = X.contiguous()
        ROW, COL = X.shape
        L = min(ROW, COL)
        H = max(ROW, COL)
        LQ = _cdiv(L, BLOCK_Q)
        HQ = _cdiv(H, BLOCK_Q)
        dtype_str = str(X.dtype).split(".")[-1]
        dev = X.device

        # ── pre-allocate ──
        A = torch.empty((ROW, COL), device=dev, dtype=_prec2dtype(precision))
        A_scale = torch.empty((_cdiv(ROW, BLOCK_Q), _cdiv(COL, BLOCK_Q)), device=dev, dtype=torch.float32)

        R = torch.empty((L, L), device=dev, dtype=_prec2dtype(precision))
        R_scale = torch.empty((LQ, LQ), device=dev, dtype=torch.float32)
        R_diag = torch.empty((L,), device=dev, dtype=torch.float32)

        Y = torch.empty((L, H), device=dev, dtype=_prec2dtype(precision))
        Y_scale = torch.empty((LQ, HQ), device=dev, dtype=torch.float32)

        QQ = torch.as_strided(Y, (L, L), (L, 1))
        QQ_scale = torch.as_strided(Y_scale, (LQ, LQ), (LQ, 1))
        QQ_diag = torch.empty((L,), device=dev, dtype=torch.float32)


        # QQ = torch.empty((L, L), device=dev, dtype=_prec2dtype(precision))
        Z = torch.empty((L, L), device=dev, dtype=_prec2dtype(precision))
        Z_scale = torch.empty((LQ, LQ), device=dev, dtype=torch.float32)
        Z_diag = torch.empty((L,), device=dev, dtype=torch.float32)

        Q = torch.empty((L, L), device=dev, dtype=_prec2dtype(precision))
        Q_scale = torch.empty((LQ, LQ), device=dev, dtype=torch.float32)
        Q_diag = torch.empty((L,), device=dev, dtype=torch.float32)

        O = torch.empty((L, H), device=dev, dtype=torch.float32)
        # ── Step 1: Frobenius norm + int8 quantise ──
        # self._sumsq_maxabs(M=ROW, N=COL, dtype=dtype_str)(X, A_max, A_square_sum)
        A_frob_norm = torch.linalg.vector_norm(X).view(1)
        self._to_bq(
            M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str, dtype_out=precision, use_norm=True, eps=eps,
        )(X, A_frob_norm, A, A_scale)

        # ── Step 2: Transpose trick: make it L×H with L = min ──
        transposed = ROW > COL
        if transposed:
            A = A.mT.contiguous()
            A_scale = A_scale.mT.contiguous()
        
        QQA = torch.as_strided(A, (L, L), (L, 1))
        QQA_scale = torch.as_strided(A_scale, (LQ, LQ), (LQ, 1))

        # ── Step 3: R = A @ A^T (Gram matrix in Type2) ──
        self._aat_bq(M=L, K=H, dtype=precision)(A, A_scale, R, R_scale, R_diag)

        # ── Step 4: unrolled iterations ──
        #  kind: "i"(init t=0), "r"(restart), "n"(normal), "e"(end)
        for t in range(len(coeffs)):
            a, b, c = coeffs[t]
            a *= 0.997
            b *= 0.997
            c *= 0.997
            kind = ("i" if t == 0
                    else "e" if t == len(coeffs) - 1
                    else "r" if (t + len(coeffs)) % 2 == 0
                    else "n")

            if kind == "r":
                # Y = Q @ X
                self._typeii_typei_bq(M=L, N=H, dtype=precision)(Q, Q_scale, Q_diag, A, A_scale, Y, Y_scale)
                # R = Y @ Y^T  (recompute Gram)
                self._aat_bq(M=L, K=H, dtype=precision)(Y, Y_scale, R, R_scale, R_diag)
                
                A, A_scale, Y, Y_scale = Y, Y_scale, A, A_scale
                QQ, QQ_scale, QQA, QQA_scale = QQA, QQA_scale, QQ, QQ_scale

            if kind in ("i", "r"):
                # Q = Quad(R)
                self._quad_bq(M=L, dtype=precision)(R, R_scale, R_diag, Q, Q_scale, Q_diag, a, b, c)
                # RZ = R @ Q  (Q holds Z)
                self._ab_symm_bq(M=L, dtype=precision)(R, R_scale, R_diag, Q, Q_scale, Q_diag, QQ, QQ_scale, QQ_diag)
                # R = Q @ RZ
                self._ab_symm_bq(M=L, dtype=precision)(Q, Q_scale, Q_diag, QQ, QQ_scale, QQ_diag, R, R_scale, R_diag)
            else:  # "n" or "e"
                # Z = Quad(R)
                self._quad_bq(M=L, dtype=precision)(R, R_scale, R_diag, Z, Z_scale, Z_diag, a, b, c)
                # Q = Z @ Q 
                self._ab_symm_bq(M=L, dtype=precision)(Z, Z_scale, Z_diag, Q, Q_scale, Q_diag, QQ, QQ_scale, QQ_diag)
                torch._foreach_copy_((Q, Q_scale, Q_diag), (QQ, QQ_scale, QQ_diag))
                if kind == "n":
                    # RZ = R @ Z
                    self._ab_symm_bq(M=L, dtype=precision)(R, R_scale, R_diag, Z, Z_scale, Z_diag, QQ, QQ_scale, QQ_diag)
                    # R = Z @ RZ
                    self._ab_symm_bq(M=L, dtype=precision)(Z, Z_scale, Z_diag, QQ, QQ_scale, QQ_diag, R, R_scale, R_diag)

        # ── Step 5: X_out = Q_final @ X ──
        self._typeii_typei_final_bq(M=L, N=H, dtype=precision)(Q, Q_scale, Q_diag, A, A_scale, O)
        OO = O.to(X.dtype)
        if transposed:
            OO = OO.mT.contiguous()
        return OO
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
        r"""Standard Newton-Schulz int8 orthogonalisation.

        Each iteration:  R = X·Xᵀ,  Z = c0·I + c1·R + c2·R²,  X ← Z·X.

        Parameters
        ----------
        X : torch.Tensor  shape (M, N), float, cuda
        coeffs : list of (c0,c1,c2) tuples, optional
        eps : float
        deterministic : bool

        Returns
        -------
        torch.Tensor  shape (M, N), same dtype as input.
        """
        if coeffs is None:
            coeffs = _DEFAULT_NS_COEFFS

        assert len(X.shape) == 2
        X = X.contiguous()
        ROW, COL = X.shape
        dev = X.device
        dtype_str = str(X.dtype).split(".")[-1]
        L = min(ROW, COL)
        H = max(ROW, COL)

        atom = torch.zeros((8, 1), device=dev)
        A_max = atom[0]
        A_square_sum = atom[1]
        AA_max = atom[2].view(torch.int32)
        B_max = atom[3]
        C_max = atom[4]
        A8 = torch.empty((ROW, COL), device=dev, dtype=torch.int8)
        A_scale = torch.empty((1,), device=dev, dtype=torch.float32)
        AA32L = torch.empty((L, L), device=dev, dtype=torch.int32)
        AA8 = torch.empty((L, L), device=dev, dtype=torch.int8)
        AA_scale = torch.empty((1,), device=dev, dtype=torch.float32)
        AA_diag = torch.empty((L,), device=dev, dtype=torch.float32)
        B32 = torch.empty((L, L), device=dev, dtype=torch.float32)
        B8 = torch.empty((L, L), device=dev, dtype=torch.int8)
        B_scale = torch.empty((1,), device=dev, dtype=torch.float32)
        B_diag = torch.empty((L,), device=dev, dtype=torch.float32)
        C32 = torch.empty((L, H), device=dev, dtype=torch.float32)

        # ── prologue ──
        self._sumsq_maxabs(M=ROW, N=COL, dtype=dtype_str)(X, A_max, A_square_sum)
        if deterministic:
            A_frob_norm = torch.linalg.vector_norm(X).view(1)
            self._scale_int8(M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str,
                     use_norm=True, eps=eps)(X, A_max, A_frob_norm, A8, A_scale)
        else:
            self._scale_int8(M=ROW, N=COL, dtype=dtype_str)(
                X, A_max, A_square_sum, A8, A_scale
            )

        transposed = ROW > COL
        if transposed:
            A8 = A8.mT.contiguous()

        # ── main loop ──
        N_iter = len(coeffs)
        for i in range(N_iter):
            a, b, c = coeffs[i]
            self._aat_int8_max(M=L, K=H)(A8, AA32L, AA_max)
            self._int32_compl_symm_int8(M=L)(
                AA32L, AA_max, A_scale, AA8, AA_scale, AA_diag
            )
            self._typeii_int8_sq(M=L)(
                AA8, AA_scale, AA_diag, B32, B_max, b, c
            )
            self._float32_compl_symm_int8_quad(M=L)(
                B32, B_max, AA_diag, B8, B_scale, B_diag, a, b, c
            )
            self._typeii_typei_int8(M=L, N=H)(
                B8, B_scale, B_diag, A8, A_scale, C32, C_max
            )
            if i == N_iter - 1:
                C = C32.to(X.dtype)
                if transposed:
                    C = C.mT.contiguous()
                return C
            else:
                self._float32_to_int8(M=L, N=H)(C32, C_max, A8, A_scale)
                atom.zero_()

        raise RuntimeError("unreachable")
    
    def _regular_prec(
        self,
        X: torch.Tensor,
        coeffs: list | None = None,
        eps: float = 1e-7,
        deterministic: bool = True,
        precision: str = 'float16'
    ) -> torch.Tensor:
        r"""Standard Newton-Schulz fp16 orthogonalisation.

        Each iteration:  R = X·Xᵀ,  Z = c0·I + c1·R + c2·R²,  X ← Z·X.

        Parameters
        ----------
        X : torch.Tensor  shape (M, N), float, cuda
        coeffs : list of (c0,c1,c2) tuples, optional
        eps : float
        deterministic : bool

        Returns
        -------
        torch.Tensor  shape (M, N), same dtype as input.
        """
        if coeffs is None:
            coeffs = _DEFAULT_NS_COEFFS

        assert len(X.shape) == 2
        X = X.contiguous()
        ROW, COL = X.shape
        dev = X.device
        dtype_str = str(X.dtype).split(".")[-1]
        L = min(ROW, COL)
        H = max(ROW, COL)

        A = torch.empty((ROW, COL), device=dev, dtype=_prec2dtype(precision))
        B = torch.empty((L, L), device=dev, dtype=_prec2dtype(precision))
        C = torch.empty((L, H), device=dev, dtype=_prec2dtype(precision))
        AA = torch.as_strided(C, (L, L), (L, 1))
        AAA = torch.as_strided(A, (L, L), (L, 1))

        A_frob_norm = torch.linalg.vector_norm(X).view(1)
        self._to_prec(
            M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str, dtype_out=precision, use_norm=True, eps=eps
        )(X, A_frob_norm, A)

        transposed = ROW > COL
        if transposed:
            A = A.mT.contiguous()

        # ── main loop ──
        N_iter = len(coeffs)
        for i in range(N_iter):
            a, b, c = coeffs[i]
            self._aat_prec(M=L, K=H, dtype=precision)(A, AA)
            self._quad_prec(M=L, dtype=precision)(AA, B, a, b, c)
            self._ab_prec(M=L, K=L, N=H, dtype=precision)(B, A, C)
            if i == N_iter - 1:
                if transposed:
                    C = C.mT.contiguous()
                return C.to(X.dtype)
            else:
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

        assert len(X.shape) == 2
        X = X.contiguous()
        ROW, COL = X.shape
        dev = X.device
        dtype_str = str(X.dtype).split(".")[-1]
        L = min(ROW, COL)
        H = max(ROW, COL)
        LQ = _cdiv(L, BLOCK_Q)
        HQ = _cdiv(H, BLOCK_Q)
        
        A = torch.empty((ROW, COL), device=dev, dtype=_prec2dtype(precision))
        A_scale = torch.empty((_cdiv(ROW, BLOCK_Q), _cdiv(COL, BLOCK_Q)), device=dev, dtype=torch.float32)

        B = torch.empty((L, L), device=dev, dtype=_prec2dtype(precision))
        B_scale = torch.empty((LQ, LQ), device=dev, dtype=torch.float32)
        B_diag = torch.empty((L,), device=dev, dtype=torch.float32)
        
        C = torch.empty((L, H), device=dev, dtype=_prec2dtype(precision))
        C_scale = torch.empty((LQ, HQ), device=dev, dtype=torch.float32)

        AA = torch.as_strided(C, (L, L), (L, 1))
        AA_scale = torch.as_strided(C_scale, (LQ, LQ), (LQ, 1))
        AA_diag = torch.empty((L,), device=dev, dtype=torch.float32)
        
        AAA = torch.as_strided(A, (L, L), (L, 1))
        AAA_scale = torch.as_strided(A_scale, (LQ, LQ), (LQ, 1))

        Z = torch.empty((L, H), device=dev, dtype=torch.float32)

        A_frob_norm = torch.linalg.vector_norm(X).view(1)
        self._to_bq(
            M=ROW, N=COL, dtype=dtype_str, dtype2=dtype_str, dtype_out=precision, use_norm=True, eps=eps
        )(X, A_frob_norm, A, A_scale)

        transposed = ROW > COL
        if transposed:
            A = A.mT.contiguous()
            A_scale = A_scale.mT.contiguous()

        # ── main loop ──
        N_iter = len(coeffs)
        for i in range(N_iter):
            a, b, c = coeffs[i]
            self._aat_bq(M=L, K=H, dtype=precision)(A, A_scale, AA, AA_scale, AA_diag)
            self._quad_bq(M=L, dtype=precision)(AA, AA_scale, AA_diag, B, B_scale, B_diag, a, b, c)
            if i == N_iter - 1:
                self._typeii_typei_final_bq(M=L, N=H, dtype=precision)(B, B_scale, B_diag, A, A_scale, Z)
                Z = Z.to(X.dtype)
                if transposed:
                    Z = Z.mT.contiguous()
                return Z
            else:
                self._typeii_typei_bq(M=L, N=H, dtype=precision)(B, B_scale, B_diag, A, A_scale, C, C_scale)
                A, A_scale, C, C_scale = C, C_scale, A, A_scale
                AA, AA_scale, AAA, AAA_scale, = AAA, AAA_scale, AA, AA_scale

        raise RuntimeError("unreachable")
