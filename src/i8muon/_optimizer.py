
import math
from collections.abc import MutableMapping

import torch
from torch import Tensor

from torch.optim.optimizer import (
    _disable_dynamo_if_unsupported,
    _to_scalar,
    Optimizer,
    ParamsT,
)

from ._ns import NSInt8, _DEFAULT_NS_COEFFS, recommend_coefficients

__all__ = ["Muon"]
EPS = 1e-7
_GRAM_ASPECT_THRESHOLD = 4.0

def _adjust_lr(lr: float, adjust_lr_fn, param_shape: torch.Size) -> float:
    """Learning rate adjustment for Muon.

    If ``adjust_lr_fn`` is callable: ``fn(lr, A, B) -> float``.
    Built-in strings: ``"spectral"`` (default), ``"original"``, ``"match_rms_adamw"``.
    """
    A, B = param_shape[:2]

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
    """Implements Muon algorithm with optional int8 acceleration.

    .. math::
       \\begin{aligned}
            &\\textbf{input} : \\gamma \\text{ (lr)},\\ \\lambda \\text{ (wd)},\\\
            &\\hspace{13mm}\\mu \\text{ (momentum)},\\ 
            C = [(a_0,b_0,c_0),\\dots]\\ (\\text{NS coefficients}),\\ \\
            &\\hspace{13mm}\\varepsilon \\text{ (epsilon)},\\ \\theta_0 \\text{ (params)}\\\\
            &\\textbf{initialize} : B_0 \\leftarrow 0 \\\\
            &\\textbf{for}\\ t=1\\ \\textbf{to}\\ \\ldots\\ \\textbf{do}\\\\
            &\\hspace{5mm} g_t \\leftarrow \\nabla_{\\theta} f_t(\\theta_{t-1})\\\\
            &\\hspace{5mm} B_t \\leftarrow \\mu B_{t-1} + g_t \\\\
            &\\hspace{5mm} \\widetilde{B}_t \\leftarrow
                \\begin{cases}
                   g_t + \\mu_0 B_t, & \\text{if } |\\mu|=2\\ (\\text{Nesterov})\\\\
                   B_t,            & \\text{if } |\\mu|=1
                \\end{cases}\\\\
            &\\hspace{5mm} O_t \\leftarrow \\mathrm{NS}^{C}(\\widetilde{B}_t;\\ \\varepsilon)\\\\
            &\\hspace{5mm} \\gamma \\leftarrow \\mathrm{AdjustLR}(\\gamma;\\ \\text{shape})\\\\
            &\\hspace{5mm} \\theta_t \\leftarrow \\theta_{t-1} - \\gamma\\, O_t
                    \\quad\\text{(with weight decay if } \\lambda \\neq 0\\text{)}
       \\end{aligned}

    Parameters
    ----------
    params : iterable
        Only 2D parameters are supported.
    lr : float, default 0.01
    weight_decay : float, default 0.0
    momentum : float | tuple[float, float], default 0.95
        Single float → no Nesterov.  2-tuple ``(mu, mu2)`` → Nesterov:
        ``buf.lerp_(grad, 1 - mu2)``, ``update = grad.lerp(buf, mu)``.
    ns_coefficients : list[tuple[float,float,float]] | None
        Newton-Schulz coefficients, one triple per iteration.
        Default: int8-optimised when ``use_int8=True`` else Keller's.
    eps : float, default 1e-7
    adjust_lr_fn : str | callable | None
        ``"spectral"`` (default), ``"original"``, ``"match_rms_adamw"``, or callable.
    use_int8 : bool, default False
        Use int8 Tensor-Core NS kernels.
    int8_autotune : bool, default False
        Autotune kernel configs at init (adds ~15 s once).
    use_cuda_graph : bool, default False
        Capture NS kernels into a CUDA graph for reduced launch overhead.
        Graph is captured lazily on the first step and replayed thereafter.
        Only meaningful with ``use_int8=True``.
    gram_aspect_threshold : float, default 4.0
        Aspect ratio above which ``gram`` is preferred over ``regular``
        (only meaningful with ``use_int8=True``).
    deterministic : bool, default True
        If True, use PyTorch norm for int8 scale factor (bit-exact,
        reproducible across runs).  If False, use fused GPU kernel norm
        (faster, non-deterministic atomics).  Only affects int8 path.
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
        precision: str = 'auto', # 'auto' | 'int8' | 'float16' | 'bfloat16' | 'fp16' | 'bf16' | 'fp8' | 'int8_block'
        autotune: bool = False,
        use_gram: bool = True,
        use_cuda_graph: bool = False,
        gram_aspect_threshold: float = _GRAM_ASPECT_THRESHOLD,
        deterministic: bool = True,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"eps should be >= 0 but is: {eps}")

        # Normalise momentum
        if isinstance(momentum, (int, float)):
            momentum = (momentum,)

        if len(momentum) not in (1, 2):
            raise ValueError(
                f"momentum must be a float or a tuple of 2 floats (Nesterov), got {momentum}"
            )
        for m in momentum:
            if not 0.0 <= m:
                raise ValueError(f"momentum should be >= 0 but is: {m}")

        if (
            adjust_lr_fn is not None
            and not callable(adjust_lr_fn)
            and adjust_lr_fn not in ["spectral", "original", "match_rms_adamw"]
        ):
            raise ValueError(f"adjust_lr_fn {adjust_lr_fn} is not supported")


        supported_precisions = {
            'auto': 'auto', 'default': 'auto', 
            'int8': 'int8', 'i8': 'int8',
            'float16': 'float16', 'half': 'float16', 'f16': 'float16', 'fp16': 'float16',
            'bfloat16': 'bfloat16', 'bf16': 'bfloat16', 
            'float32': 'float32', 'float': 'float32', 'fp32': 'float32',
            'fp8': 'float8_e4m3fn', 'float8': 'float8_e4m3fn', 'float8_e4m3': 'float8_e4m3fn', 'float8_e4m3fn': 'float8_e4m3fn',
            'int8_bq': 'int8_bq', 'int8_block': 'int8_bq', 'i8bq': 'int8_bq', 'i8block': 'int8_bq',
            # 'float64': 'float64', 'double': 'float64', 'fp64': 'float64',
        }
        if precision.lower() not in supported_precisions:
            raise ValueError(f"Supported precisions are {list(set(supported_precisions.values()))}, but got: {precision}")

        precision = supported_precisions[precision.lower()]

        # Default coefficients
        if ns_coefficients is None:
            if ns_steps is None:
                ns_coefficients = _DEFAULT_NS_COEFFS
            else:
                ns_coefficients = recommend_coefficients(
                    precision=(precision not in ("int8", "auto", 'float8_e4m3fn', 'int8_bq')), iters=ns_steps
                )

        self._ns = NSInt8(autotune=autotune)
        # ── CUDA graph state ──
        self._use_cuda_graph = use_cuda_graph
        
        self._ns_graphs: dict[tuple, object] = {}  # key → (graph, output_tensor)
        self._gram_aspect_threshold = gram_aspect_threshold

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "adjust_lr_fn": adjust_lr_fn,
            "deterministic": deterministic,
            "use_gram": use_gram,
            "precision": precision,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters, got {p.size()}"
                    )



    # ── helper ────────────────────────────────────────────────────

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
    ) -> bool:
        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )
            momentum_bufs.append(state["momentum_buffer"])

        return False  # has_complex

    # ── step ──────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        For each 2D parameter:
        1. Update momentum buffer
        2. Compute Nesterov-adjusted update
        3. Apply Newton-Schulz orthogonalization (int8 or float fallback)
        4. Apply weight decay + scaled update
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
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

            has_complex = self._init_group(
                group, params_with_grad, grads, momentum_bufs
            )

            muon_step(
                params_with_grad,
                grads,
                momentum_bufs,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                ns_coefficients=ns_coefficients,
                eps=eps,
                adjust_lr_fn=adjust_lr_fn,
                deterministic=deterministic,
                has_complex=has_complex,
                ns_engine=self._ns,
                precision=precision,
                use_gram=use_gram,
                use_cuda_graph=self._use_cuda_graph,
                ns_graphs=self._ns_graphs,
                gram_aspect_threshold=self._gram_aspect_threshold,
            )
        return loss


# ═══════════════════════════════════════════════════════════════
#  Single-tensor update (extracted for Dynamo/compile support)
# ═══════════════════════════════════════════════════════════════


def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: tuple[float, ...],
    ns_coefficients: list[tuple[float, float, float]],
    eps: float,
    adjust_lr_fn,
    deterministic: bool,
    has_complex: bool,
    ns_engine: NSInt8,
    precision: str,
    use_gram: bool,
    use_cuda_graph: bool,
    ns_graphs: dict,
    gram_aspect_threshold: float,
) -> None:
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    nesterov = len(momentum) == 2
    mu = momentum[0]
    mu2 = momentum[-1]

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = momentum_bufs[i]
        buf.lerp_(grad, 1 - mu2)
        update = grad.lerp(buf, mu) if nesterov else buf

        M, N = update.shape
        gram_method = (max(M, N) / min(M, N) >= gram_aspect_threshold and use_gram)
        actual_prec = (
            ('int8' if 256 <= max(M, N) else 'float16') if precision == 'auto' else precision
        )
        if gram_method and actual_prec == 'int8': 
            actual_prec = 'float16'

        use_bq = False
        if precision == "float8_e4m3fn":
            actual_prec = "float8_e4m3fn"
            use_bq = True
        if precision == "int8_bq":
            actual_prec = "int8"
            use_bq = True
        
        ns_fn = (
            (ns_engine._gram_bq if gram_method else ns_engine._regular_bq)
            if use_bq else
            ns_engine._gram_prec 
            if gram_method else 
            ns_engine._regular_i8
            if actual_prec == 'int8' else
            ns_engine._regular_prec
        )

        def call_ns(u):
            return ns_fn(
                u, coeffs=ns_coefficients, eps=eps, deterministic=deterministic, precision=actual_prec
            )
        
        if use_cuda_graph:
            key = (ns_fn.__name__, actual_prec, M, N)
            if key not in ns_graphs:
                # Allocate fixed-address buffers
                in_buf = torch.empty(M, N, device=update.device, dtype=update.dtype)
                in_buf.copy_(update)
                _ = call_ns(in_buf)
                _ = call_ns(in_buf)
                in_buf.copy_(update)
                # Capture
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out_buf = call_ns(in_buf)
                ns_graphs[key] = (g, in_buf, out_buf)
            g, in_buf, out_buf = ns_graphs[key]
            in_buf.copy_(update)
            g.replay()
            update = out_buf
        else:
            update = call_ns(update)

        # ── weight decay + apply ──
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        if weight_decay != 0:
            update.add_(param, alpha=weight_decay)
        param.add_(update, alpha=-adjusted_lr)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_muon)
def muon_step(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    *,
    foreach: bool | None = None,
    lr: float,
    weight_decay: float,
    momentum: tuple[float, ...],
    ns_coefficients: list[tuple[float, float, float]],
    eps: float,
    adjust_lr_fn,
    deterministic: bool = True,
    has_complex: bool = False,
    ns_engine: NSInt8,
    precision: str = 'auto',
    use_gram: bool = True,
    use_cuda_graph: bool = False,
    ns_graphs: dict | None = None,
    gram_aspect_threshold: float = _GRAM_ASPECT_THRESHOLD,
) -> None:
    r"""Functional API for Muon algorithm computation.

    See :class:`Muon` for details.
    """
    if foreach is not None and foreach:
        raise RuntimeError("Foreach is not supported for Muon yet")

    if ns_graphs is None:
        ns_graphs = {}

    _single_tensor_muon(
        params,
        grads,
        momentum_bufs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        ns_coefficients=ns_coefficients,
        eps=eps,
        adjust_lr_fn=adjust_lr_fn,
        deterministic=deterministic,
        has_complex=has_complex,
        ns_engine=ns_engine,
        precision=precision,
        use_gram=use_gram,
        use_cuda_graph=use_cuda_graph,
        ns_graphs=ns_graphs,
        gram_aspect_threshold=gram_aspect_threshold,
    )
