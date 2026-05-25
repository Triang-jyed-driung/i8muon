"""
i8muon_naive — pure-PyTorch int8 Newton-Schulz using torch._int_mm.

No TileLang kernels.  Uses Type1/Type2 abstractions.
For accuracy comparison against the TileLang kernel path.
"""

import math
import torch


# ═══════════════════════════════════════════════════════════════
#  Type1 / Type2 — int8 matrix representations
# ═══════════════════════════════════════════════════════════════


class Type1:
    """General int8 matrix (M×N) with scale factor."""

    __slots__ = ("data", "scale")

    def __init__(self, data: torch.Tensor, scale: torch.Tensor):
        self.data = data    # int8  (M, N)
        self.scale = scale   # () or (1,) float

    @staticmethod
    def from_float(A: torch.Tensor) -> "Type1":
        amax = A.abs().max()
        if amax == 0:
            scale = torch.ones_like(amax)
            data = A.to(torch.int8)
        else:
            scale = amax / 127.0
            data = (A / scale).round().clamp(-128, 127).to(torch.int8)
        return Type1(data, scale)

    def to_float(self) -> torch.Tensor:
        return self.data.float() * self.scale

    @property
    def T(self) -> "Type1":
        return Type1(self.data.mT, self.scale)

    @property
    def shape(self):
        return self.data.shape


class Type2:
    """Symmetric int8 matrix (M×M) as diagonal + off-diagonal + scale."""

    __slots__ = ("diag", "data", "scale")

    def __init__(self, diag: torch.Tensor, data: torch.Tensor, scale: torch.Tensor):
        self.diag = diag   # (M,)  float32
        self.data = data    # (M,M) int8  (diagonal zero)
        self.scale = scale  # () or (1,) float

    @staticmethod
    def from_float(A: torch.Tensor) -> "Type2":
        diag = A.diag()
        off = A - torch.diag(diag)
        amax = off.abs().max()
        if amax == 0:
            scale = torch.tensor([1.0], device=A.device)
            data = off.to(torch.int8)
        else:
            scale = amax / 127.0
            data = (off / scale).round().clamp(-128, 127).to(torch.int8)
        data.fill_diagonal_(0)
        return Type2(diag, data, scale)

    def to_float(self) -> torch.Tensor:
        return torch.diag(self.diag) + self.data.float() * self.scale

    @property
    def shape(self):
        return self.data.shape


# ═══════════════════════════════════════════════════════════════
#  Pure-PyTorch int8 matrix operations
# ═══════════════════════════════════════════════════════════════


def Type1_mul_Type1(A: Type1, B: Type1) -> Type1:
    """A @ B  (int8 GEMM via torch._int_mm)."""
    C_int = torch._int_mm(A.data, B.data)        # (M, N) int32
    C_float = C_int.float() * (A.scale * B.scale)
    return Type1.from_float(C_float)


def Type1_aat(A: Type1) -> Type2:
    """A @ A^T → Type2."""
    C_int = torch._int_mm(A.data, A.data.t().contiguous())  # (M, M) int32
    C_float = C_int.float() * (A.scale ** 2)
    return Type2.from_float(C_float)


def Type2_sq(A: Type2, a: float, b: float, c: float) -> Type2:
    """Z = a·I + b·A + c·A²  (polynomial in Type2)."""
    A_f = A.to_float()                     # float32 (M, M)
    A2 = A_f @ A_f                          # A²
    Z_f = a * torch.eye(A.shape[0], device=A.data.device) + b * A_f + c * A2
    return Type2.from_float(Z_f)


def Type2_ab(A: Type2, B: Type2) -> Type2:
    """A @ B → Type2  (Type2 × Type2)."""
    A_f = A.to_float()
    B_f = B.to_float()
    C_f = A_f @ B_f
    return Type2.from_float(C_f)


def Type2_typei(A: Type2, B: Type1) -> Type1:
    """A @ B → Type1  (Type2 × Type1)."""
    A_f = A.to_float()
    B_f = B.to_float()
    C_f = A_f @ B_f
    return Type1.from_float(C_f)


# ═══════════════════════════════════════════════════════════════
#  NSInt8Naive
# ═══════════════════════════════════════════════════════════════

_DEFAULT_NS_COEFFS = [
    (3.9274, -8.7643,  5.3095),
    (3.4317, -5.4288,  2.3608),
    (3.5403, -5.3366,  2.2324),
    (3.6733, -4.8533,  1.8498),
    (2.6731, -2.4447,  0.7695),
]


class NSInt8Naive:
    """Pure-PyTorch int8 Newton-Schulz (no TileLang).

    Uses torch._int_mm for int8 GEMM, Type1/Type2 for scale management.
    """

    def gram(
        self,
        X: torch.Tensor,
        coeffs: list | None = None,
        eps: float = 1e-7,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Gram-form int8 Newton-Schulz (naive PyTorch)."""
        if coeffs is None:
            coeffs = _DEFAULT_NS_COEFFS

        X = X.contiguous()
        ROW, COL = X.shape
        L = min(ROW, COL)
        H = max(ROW, COL)

        # Step 1: norm + quantise
        A = Type1.from_float(X / X.norm().clamp(min=eps))

        # Step 2: transpose trick → L×H
        transposed = ROW > COL
        if transposed:
            A = A.T
            A = Type1(A.data.contiguous(), A.scale)

        # Step 3: initial Gram R = A @ A^T
        R = Type1_aat(A)

        # Step 4: unrolled iterations
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
                A = Type2_typei(Q, A)          # X = Q @ X
                A = Type1.from_float(A.to_float())
                R = Type1_aat(A)               # rebuild R

            if kind in ("i", "r"):
                Q = Type2_sq(R, a, b, c)        # Q = Z
                RZ = Type2_ab(R, Q)              # RZ = R·Q
                R = Type2_ab(Q, RZ)              # R = Q·RZ
            else:
                Z = Type2_sq(R, a, b, c)        # Z → Z8
                Q = Type2_ab(Z, Q)               # Q = Q·Z
                if kind == "n":
                    RZ = Type2_ab(R, Z)          # RZ = R·Z
                    R = Type2_ab(Z, RZ)          # R = Z·RZ

        # Step 5: X_out = Q_final @ X
        X_out = Type2_typei(Q, A).to_float().to(X.dtype)
        if transposed:
            X_out = X_out.mT.contiguous()
        return X_out

    def regular(
        self,
        X: torch.Tensor,
        coeffs: list | None = None,
        eps: float = 1e-7,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Standard int8 Newton-Schulz (naive PyTorch)."""
        if coeffs is None:
            coeffs = _DEFAULT_NS_COEFFS

        assert len(X.shape) == 2
        X = X.contiguous()
        ROW, COL = X.shape
        L = min(ROW, COL)
        H = max(ROW, COL)

        A = Type1.from_float(X / X.norm().clamp(min=eps))

        transposed = ROW > COL
        if transposed:
            A = A.T
            A = Type1(A.data.contiguous(), A.scale)

        for i, (a, b, c) in enumerate(coeffs):
            R = Type1_aat(A)                     # R = A @ A^T
            Z = Type2_sq(R, a, b, c)            # Z = aI + bR + cR²
            if i == len(coeffs) - 1:
                A_out = Type2_typei(Z, A)
                C = A_out.to_float().to(X.dtype)
                if transposed:
                    C = C.mT.contiguous()
                return C
            A = Type2_typei(Z, A)
            A = Type1.from_float(A.to_float())

        raise RuntimeError("unreachable")
