
import math
import os
from functools import cache
from typing import Optional

import torch
from torch import Tensor

from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
)

from typing import TYPE_CHECKING
from ._coeff import recommend_coefficients
from ._cutlass_build import device_properties

if TYPE_CHECKING:
    from ._cutlass import CutlassNSInt8

__all__ = ["Muon", "muon_step"]
EPS = 1e-7
_GRAM_ASPECT_THRESHOLD = 5.0
_PRECISIONS = {
    'auto': 'auto', 'default': 'auto',
    'int8': 'int8', 'i8': 'int8',
    'float16': 'float16', 'half': 'float16', 'f16': 'float16', 'fp16': 'float16',
    'bfloat16': 'bfloat16', 'bf16': 'bfloat16',
    'float32': 'float32', 'float': 'float32', 'fp32': 'float32',
    'fp8': 'float8_e4m3fn', 'fp8_bq': 'float8_e4m3fn', 'float8': 'float8_e4m3fn',
    'float8_e4m3': 'float8_e4m3fn', 'float8_e4m3fn': 'float8_e4m3fn',
    'int8_bq': 'int8_bq', 'int8_block': 'int8_bq', 'i8bq': 'int8_bq', 'i8block': 'int8_bq',
}


@cache
def _normalize_precision(precision):
    if not isinstance(precision, str) or precision.lower() not in _PRECISIONS:
        raise ValueError(f"Unsupported precision: {precision}")
    return _PRECISIONS[precision.lower()]


def _validate_group(group):
    # Apply the original optimizer argument rules to initial and added groups.
    lr = group["lr"]
    if isinstance(lr, Tensor) and lr.numel() != 1:
        raise ValueError("Tensor lr must be 1-element")
    for name in ("weight_decay", "eps"):
        if not 0.0 <= group[name]:
            raise ValueError(f"{name} must be >= 0")
    coeffs = group["ns_coefficients"]
    if coeffs is not None and (not coeffs or any(len(row) != 3 for row in coeffs)):
        raise ValueError("Expected a nonempty list of (a, b, c) coefficients")
    momentum = group["momentum"]
    if isinstance(momentum, (int, float)):
        momentum = (momentum,)
    if len(momentum) not in (1, 2) or any(not 0.0 <= m for m in momentum):
        raise ValueError("momentum must contain one or two nonnegative values")
    group["momentum"] = tuple(momentum)
    adjust = group["adjust_lr_fn"]
    if adjust is not None and not callable(adjust) and adjust not in (
            "spectral", "original", "match_rms_adamw", "none"):
        raise ValueError(f"adjust_lr_fn {adjust} is not supported")
    group["precision"] = _normalize_precision(group["precision"])
    if group["weight_decay_mode"] not in ("absolute", "base_lr", "adjusted_lr"):
        raise ValueError("weight_decay_mode must be 'absolute', 'base_lr' or 'adjusted_lr'")


class RoutingMixin:
    """Shared route selection for optimizer and direct engine calls."""

    @cache
    def router(self, M: int, N: int, precision: str, use_gram: bool, gram_aspect_threshold: float):
        if min(M, N) <= 0:
            raise ValueError("Matrix dimensions must be positive")
        precision = _normalize_precision(precision)
        aspect_ratio = max(M, N) / min(M, N)
        if use_gram and aspect_ratio >= gram_aspect_threshold:
            compute = ("float16" if precision in ("auto", "int8", "int8_bq", "float8_e4m3fn")
                       else precision)
            return self._gram_prec, compute
        if precision == "auto":
            if M * N < 1024 * 1024:
                return self._regular_prec, "float16"
            return self._regular_i8, "int8"
        if precision == "int8_bq":
            return self._regular_bq, "int8"
        if precision == "float8_e4m3fn":
            return self._regular_bq, precision
        if precision == "int8":
            return self._regular_i8, precision
        return self._regular_prec, precision


@cache
def _validate_ns_device(device, compute_precision):
    """Check the selected route on its actual device before native execution."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Muon requires CUDA parameters; move matrix parameters to a CUDA device.")
    properties = device_properties(device)
    capability = (properties.major, properties.minor)
    if capability < (8, 0):
        raise ValueError("Muon CUTLASS kernels require SM80 or newer.")
    if compute_precision == "float8_e4m3fn" and capability < (8, 9):
        raise ValueError("FP8 BQ requires SM89 or newer; select precision='int8_bq' on SM87.")


@cache
def _matrix_spec(shape, dtype, device, precision, alignment=16):
    """Validate a matrix specification once and cache its padded BMM geometry."""
    if len(shape) < 2 or device.type != "cuda":
        raise ValueError("Expected a CUDA tensor with at least two dimensions")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Input must be float32, float16 or bfloat16")
    if min(shape[-2:]) < 1:
        raise ValueError("Matrix dimensions must be positive")
    batch = math.prod(shape[:-2])
    if not 1 <= batch <= 65535:
        raise ValueError("Batch size must be between 1 and 65535")
    _validate_ns_device(device, precision)
    rows, cols = shape[-2:]
    l, h = min(rows, cols), max(rows, cols)
    lp, hp = ((l + alignment - 1) // alignment * alignment,
              (h + alignment - 1) // alignment * alignment)
    if alignment == 16 and precision == "int8" and hp * 127 * 127 > 2**31 - 1:
        raise ValueError("Inner dimension can overflow the int32 accumulator")
    if hp > 2**31 - 1:
        raise ValueError("Padded matrix dimensions exceed the CUDA integer range")
    return batch, rows, cols, l, h, lp, hp


def _ns_graph_key(ns_fn, update, precision, coefficients, eps, deterministic, engine):
    """A graph owns addresses and constants for one complete NS configuration."""
    schedule = recommend_coefficients() if coefficients is None else coefficients
    return (engine, ns_fn.__name__, update.device, update.dtype, tuple(update.shape),
            precision, tuple(tuple(row) for row in schedule), eps, deterministic,
            engine.fp16_first, tuple(sorted(engine.tiles.items())),
            os.environ.get("I8MUON_GRAM_RESET", "00100"))


def _adjust_lr(lr: float, adjust_lr_fn, param_shape: torch.Size) -> float:
    """Learning rate adjustment for Muon.

    If ``adjust_lr_fn`` is callable: ``fn(lr, A, B) -> float``.
    Built-in strings: ``"spectral"`` (default), ``"original"``,
    ``"match_rms_adamw"``, and ``"none"`` (equivalent to None).
    """
    A, B = param_shape[-2], param_shape[-1]

    if callable(adjust_lr_fn):
        return adjust_lr_fn(lr, A, B) # type: ignore

    if adjust_lr_fn == "spectral":
        adjusted_ratio = math.sqrt(A / B)
    elif adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio

# ═══════════════════════════════════════════════════════════════
#  Muon Optimizer
# ═══════════════════════════════════════════════════════════════

class Muon(Optimizer):
    """Muon with native CUTLASS orthogonalization of momentum updates.

    Matrix parameters use Newton–Schulz; scalar/vector parameters can be handed
    to ``scalar_optimizer``. A scalar momentum uses a single moving average;
    ``(mu, mu2)`` uses Nesterov blending with a buffer decay of ``mu2``.

    ``precision`` selects auto, int8, int8_bq, fp8, float16, bfloat16, or
    float32. Both int8 routes run one fp16 iteration before the int8 tail on
    every update when ``fp16_first=True`` (default). Set it False to run all
    iterations in int8, using the same coefficient schedule.
    ``use_gram`` permits floating-point Gram computation above
    ``gram_aspect_threshold``; set it False to keep the regular hybrid path.

    ``ns_coefficients`` supplies one (a, b, c) triple per iteration. Otherwise
    ``ns_steps`` selects the INT8 safety schedule (default five iterations);
    extra iterations repeat stage 3 before stages 4 and 5. Pass an explicit
    ``ns_coefficients`` schedule to select Polar Express.
    ``adjust_lr_fn`` accepts spectral, original, match_rms_adamw, "none", a
    callable taking (lr, rows, cols), or None. "none" and None disable scaling.
    ``weight_decay_mode="base_lr"`` (default) applies decoupled decay using the
    unadjusted learning rate. ``"adjusted_lr"`` scales decay with the adjusted
    learning rate, preserving the original update. ``"absolute"`` applies
    a per-step decay fraction of ``weight_decay``, independent of either LR.
    All modes scale the NS update with the adjusted learning rate.

    ``autotune`` measures precompiled kernel choices on first use.
    ``use_cuda_graph`` captures a warmed-up pipeline and replays it on later
    updates. ``deterministic`` controls fused int8 norm reduction; the hybrid
    first iteration always accumulates its normalization norm in fp32.

    CUDA parameters are checked on construction against the selected route:
    all kernels require SM80+, and FP8 BQ requires SM89+. CPU construction
    defers the device check until step. Unsupported devices raise ValueError.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        momentum: float | tuple[float, float] = (0.95, 0.95),
        ns_coefficients: list[tuple[float, float, float]] | None = None,
        ns_steps: int | None = None,
        eps: float = EPS,
        adjust_lr_fn="spectral",
        *,
        weight_decay_mode: str = "base_lr",
        precision: str = 'auto',
        autotune: bool = False,
        fp16_first: bool = True,
        use_gram: bool = True,
        use_cuda_graph: bool = False,
        gram_aspect_threshold: float = _GRAM_ASPECT_THRESHOLD,
        deterministic: bool = True,
        scalar_optimizer: Optional[Optimizer] = None,
    ) -> None:
        precision = _normalize_precision(precision)

        # Default coefficients
        if ns_coefficients is None:
            ns_coefficients = recommend_coefficients(iters=5 if ns_steps is None else ns_steps)

        from ._cutlass import CutlassNSInt8
        self._ns = CutlassNSInt8(autotune=autotune, fp16_first=fp16_first)
        # ── CUDA graph state ──
        self._use_cuda_graph = use_cuda_graph
        
        self._ns_graphs: dict[tuple, object] = {}  # key -> (graph, input, output)
        self._gram_aspect_threshold = gram_aspect_threshold
        self.scalar_optimizer = None  # Attach after Muon groups have been initialized.

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "weight_decay_mode": weight_decay_mode,
            "momentum": momentum,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "adjust_lr_fn": adjust_lr_fn,
            "deterministic": deterministic,
            "use_gram": use_gram,
            "precision": precision,
        }
        super().__init__(params, defaults)

        self._check_scalar_overlap(scalar_optimizer)
        self.scalar_optimizer = scalar_optimizer

    def _check_scalar_overlap(self, scalar):
        if scalar is not None:
            matrices = {id(p) for group in self._muon_param_groups for p in group["params"]}
            if any(id(p) in matrices for group in scalar.param_groups for p in group["params"]):
                raise ValueError("Muon and scalar_optimizer must have disjoint parameters")

    def _validate_param_group(self, group):
        _validate_group(group)
        for p in group["params"]:
            if p.ndim < 2 or min(p.shape[-2:]) < 1 or not p.is_floating_point():
                raise ValueError("Muon only supports nonempty real floating >=2D parameters")
            if p.is_cuda:
                method, compute = self._ns.router(*p.shape[-2:], group["precision"],
                                             group["use_gram"], self._gram_aspect_threshold)
                _matrix_spec(p.shape, p.dtype, p.device, compute,
                             128 if method.__name__ == "_regular_bq" else 16)

    def add_param_group(self, param_group):
        group = dict(param_group)
        params = group["params"]
        if isinstance(params, Tensor):
            params = [params]
        if isinstance(params, set):
            raise TypeError("Optimizer parameters must have a deterministic order")
        group["params"] = list(params)
        merged = dict(self.defaults, **group)
        self._validate_param_group(merged)
        scalar = self.scalar_optimizer
        if scalar is not None:
            auxiliary = {id(p) for g in scalar.param_groups for p in g["params"]}
            if any(id(p) in auxiliary for p in group["params"]):
                raise ValueError("Muon and scalar_optimizer must have disjoint parameters")
        self.scalar_optimizer = None
        try:
            super().add_param_group(merged)
        finally:
            self.scalar_optimizer = scalar


    @property
    def param_groups(self):
        """Expose both muon and scalar optimizer param groups for LR schedulers."""
        if self.scalar_optimizer is None:
            return self._muon_param_groups
        return self._muon_param_groups + self.scalar_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        """Set param_groups during initialization. Called by PyTorch's Optimizer.__init__."""
        self._muon_param_groups = value

    def __setstate__(self, state):
        # Optimizer.load_state_dict calls __setstate__, whose base implementation
        # updates __dict__ directly and bypasses the param_groups property.
        state = dict(state)
        if "param_groups" in state:
            state["_muon_param_groups"] = state.pop("param_groups")
        super().__setstate__(state)
        for group in self._muon_param_groups:
            # Checkpoints predating this option used adjusted-LR decay.
            group.setdefault("weight_decay_mode", "adjusted_lr")

    def state_dict(self):
        """Return Muon state only; scalar optimizer state is separate."""
        scalar = self.scalar_optimizer
        self.scalar_optimizer = None
        try:
            return super().state_dict()
        finally:
            self.scalar_optimizer = scalar

    def load_state_dict(self, state_dict):
        """Load state_dict for muon params only."""
        scalar = self.scalar_optimizer
        self.scalar_optimizer = None
        try:
            super().load_state_dict(state_dict)
            self._ns_graphs.clear()
        finally:
            self.scalar_optimizer = scalar

    # ── step ──────────────────────────────────────────────────────

    def step(self, closure=None):
        """Performs a single optimization step.

        For each 2D parameter:
        1. Update momentum buffer
        2. Compute Nesterov-adjusted update
        3. Apply Newton-Schulz orthogonalization (int8 or float fallback)
        4. Apply weight decay + scaled update

        If scalar_optimizer is provided, also calls scalar_optimizer.step().
        """
        with torch.no_grad():
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self._muon_param_groups:
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                ns_coefficients = group["ns_coefficients"]
                eps = group["eps"]
                adjust_lr_fn = group["adjust_lr_fn"]
                deterministic = group["deterministic"]
                precision = group["precision"]
                use_gram = group["use_gram"]

                params_with_grad: list[Tensor] = []
                grads: list[Tensor] = []
                momentum_bufs: list[Tensor] = []

                for p in group["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    state = self.state[p]
                    buffer = state.get("momentum_buffer")
                    if buffer is None:
                        buffer = state["momentum_buffer"] = torch.zeros_like(
                            grad, memory_format=torch.preserve_format)
                    params_with_grad.append(p)
                    grads.append(grad)
                    momentum_bufs.append(buffer)

                muon_step(
                    params_with_grad,
                    grads,
                    momentum_bufs,
                    lr=lr,
                    weight_decay=weight_decay,
                    weight_decay_mode=group["weight_decay_mode"],
                    momentum=momentum,
                    ns_coefficients=ns_coefficients,
                    eps=eps,
                    adjust_lr_fn=adjust_lr_fn,
                    deterministic=deterministic,
                    ns_engine=self._ns,
                    precision=precision,
                    use_gram=use_gram,
                    use_cuda_graph=self._use_cuda_graph,
                    ns_graphs=self._ns_graphs,
                    gram_aspect_threshold=self._gram_aspect_threshold,
                )

            if self.scalar_optimizer is not None:
                self.scalar_optimizer.step()

            return loss


# ═══════════════════════════════════════════════════════════════
#  Shared optimizer and functional update
# ═══════════════════════════════════════════════════════════════


def muon_step(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: tuple[float, ...],
    ns_coefficients: list[tuple[float, float, float]] | None,
    eps: float,
    adjust_lr_fn,
    deterministic: bool = True,
    ns_engine: "CutlassNSInt8",
    precision: str = 'auto',
    use_gram: bool = True,
    use_cuda_graph: bool = False,
    ns_graphs: dict | None = None,
    gram_aspect_threshold: float = _GRAM_ASPECT_THRESHOLD,
    weight_decay_mode: str = "base_lr",
) -> None:
    """Apply dense real Muon updates; shared by the optimizer and functional API.

    The optimizer validates group settings on construction. Direct callers
    supply the same settings and matching parameter/gradient/buffer lists.
    """
    if ns_graphs is None:
        ns_graphs = {}

    if isinstance(lr, Tensor) and lr.ndim != 0:
        lr = lr.squeeze()
    decay_factor = {"absolute": 1 - weight_decay,
                    "base_lr": 1 - lr * weight_decay,
                    "adjusted_lr": None}[weight_decay_mode]

    nesterov = len(momentum) == 2
    mu = momentum[0]
    mu2 = momentum[-1]
    precision = _normalize_precision(precision)
    prepared = {}
    custom_lr = callable(adjust_lr_fn)

    for i, param in enumerate(params):
        grad = grads[i]
        M, N = grad.shape[-2], grad.shape[-1]
        layout = grad if nesterov else momentum_bufs[i]
        signature = (layout.device, layout.dtype, tuple(layout.shape))
        settings = prepared.get(signature)
        if settings is None:
            ns_fn, actual_prec = ns_engine.router(M, N, precision, use_gram, gram_aspect_threshold)
            _validate_ns_device(grad.device, actual_prec)
            key = (_ns_graph_key(ns_fn, layout, actual_prec, ns_coefficients,
                                 eps, deterministic, ns_engine) if use_cuda_graph else None)
            adjusted_lr = None if custom_lr else _adjust_lr(lr, adjust_lr_fn, param.shape)
            settings = (ns_fn, actual_prec, key, adjusted_lr)
            prepared[signature] = settings
        ns_fn, actual_prec, key, adjusted_lr = settings
        buf = momentum_bufs[i]
        buf.lerp_(grad, 1 - mu2)
        update = grad.lerp(buf, mu) if nesterov else buf

        if use_cuda_graph:
            captured = ns_graphs.get(key)
            if captured is None:
                with torch.cuda.device(update.device):
                    # Preserve leading batch dimensions and select the input
                    # device even when it is not the current CUDA device.
                    in_buf = torch.empty(update.shape, device=update.device, dtype=update.dtype)
                    in_buf.copy_(update)
                    _ = ns_fn(in_buf, coeffs=ns_coefficients, eps=eps,
                              deterministic=deterministic, precision=actual_prec)
                    _ = ns_fn(in_buf, coeffs=ns_coefficients, eps=eps,
                              deterministic=deterministic, precision=actual_prec)
                    in_buf.copy_(update)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        out_buf = ns_fn(in_buf, coeffs=ns_coefficients, eps=eps,
                                        deterministic=deterministic, precision=actual_prec)
                captured = ns_graphs[key] = (g, in_buf, out_buf)
            g, in_buf, out_buf = captured
            in_buf.copy_(update)
            g.replay()
            update = out_buf
        else:
            update = ns_fn(update, coeffs=ns_coefficients, eps=eps,
                           deterministic=deterministic, precision=actual_prec)

        # ── weight decay + apply ──
        if custom_lr:
            adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        if weight_decay != 0:
            if decay_factor is None:
                update.add_(param, alpha=weight_decay)
            else:
                param.mul_(decay_factor)
        param.add_(update, alpha=-adjusted_lr)

