"""CUTLASS/CUDA floating-point, int8, and block-quantized engines."""

from functools import cache, cached_property, partial
import math
import os
import torch

from ._coeff import _DEFAULT_NS_COEFFS
from ._optimizer import RoutingMixin, _normalize_precision, _matrix_spec
from ._cutlass_build import load_extension, device_properties
from ._cuda_factory import KernelFactory, int8_candidates
from . import _cutlass_prec, _cutlass_bq


class CutlassNSInt8(RoutingMixin):
    """Native floating-point regular/Gram NS and regular int8/int8/FP8 BQ.

    ``fp16_first`` enables one FP16 iteration before INT8 (default True).
    Floating-point and FP8 routes are unaffected.
    Internal NS entry points run under the caller's gradient context;
    ``Muon.step`` disables gradients once for the complete update.
    The optimizer validates coefficients and eps at construction; internal
    NS methods assume those settings are valid.

    The extension is compiled once, then reused across shapes. ``autotune`` is
    used to benchmark precompiled CUDA specializations on the input device.
    ``tiles`` may override ``aat``, ``square``, and ``update`` choices.
    See docs/CUTLASS.md for the precompiled configurations. Call once before CUDA graph capture to build the extension.
    """

    def __init__(self, autotune=False, *, tiles=None, fp16_first=True):
        self.fp16_first = fp16_first
        self.autotune = autotune
        self._factory = KernelFactory(autotune)
        self.tiles = dict(tiles or {})
        if (set(self.tiles) - {"aat", "square", "update"}
                or any(v not in (0, 1, 2, 3, 4) for v in self.tiles.values())):
            raise ValueError("tiles accepts aat/square/update with values 0/1/2/3/4")

        self._regular_prec = partial(_cutlass_prec.floating, self)
        self._regular_prec.__name__ = "_regular_prec"
        self._gram_prec = partial(_cutlass_prec.floating, self, gram=True)
        self._gram_prec.__name__ = "_gram_prec"
        if autotune:
            for name in ("_regular_prec", "_gram_prec", "_regular_i8", "_regular_bq"):
                run = getattr(self, name)
                tuned = partial(self._native_pipeline, name, run)
                tuned.__name__ = name
                setattr(self, name, tuned)

    @cached_property
    def _extension(self):
        return load_extension()

    def _native_pipeline(self, family, run, x, coeffs=None, eps=1e-7, deterministic=True, precision=None, **kwargs):
        if precision is None:
            precision = {"_regular_i8": "int8", "_regular_bq": "float8_e4m3fn"}.get(family, "float16")
        defaults = _DEFAULT_NS_COEFFS
        signature = (tuple(tuple(c) for c in (defaults if coeffs is None else coeffs)),
                     eps, deterministic, precision, self.fp16_first, tuple(self.tiles.items()),
                     tuple(kwargs.items()), os.environ.get("I8MUON_GRAM_RESET", "00100"))
        def launch(tuned):
            previous = self._factory.autotune
            self._factory.autotune = tuned
            try:
                return run(x, coeffs, eps, deterministic, precision, **kwargs)
            finally:
                self._factory.autotune = previous
        return self._factory.select_pipeline(family, x, signature, launch)

    def _regular_bq(self, x, coeffs=None, eps=1e-7, deterministic=True, precision="float8_e4m3fn"):
        precision = _normalize_precision(precision)
        precision = "int8" if precision == "int8_bq" else precision
        if precision == "int8":
            return self._regular_i8(x, coeffs, eps, deterministic, precision, block=True)
        return _cutlass_bq.regular_bq(self, x, coeffs, eps, deterministic, precision)

    def _regular_i8_grouped_cubic(self, X, input_gain, cubic_gains, terminal_abc, eps=1e-7):
        """Opt-in all-INT8 grouped cubics followed by one expanded quintic.

        This explicit entry ignores fp16_first.
        """
        from ._cutlass_cubic import grouped_cubic
        return grouped_cubic(self, X, input_gain, cubic_gains, terminal_abc, eps)

    def _regular_i8_cubic_abc(self, X, coeffs, eps=1e-7, deterministic=True, precision='int8'):
        """Opt-in saved cubic ABC rows plus terminal quintic; all INT8 GEMMs."""
        if precision != 'int8':
            raise ValueError('Cubic ABC evaluation requires INT8')
        from ._cutlass_cubic import cubic_abc
        return cubic_abc(self, X, coeffs, eps)

    def _regular_i8_factored(self, X, factor_rows, eps=1e-7, deterministic=True):
        """Opt-in completed-square path; rows are stored (c,p,q), never recovered.

        INT8 square/quantization are unscaled; c is folded into the update's
        A scale. The FP32 diagonal and the existing default route are retained.
        """
        factors = tuple(tuple(float(v) for v in row) for row in factor_rows)
        if not factors or any(len(row) != 3 or not all(map(math.isfinite, row)) for row in factors):
            raise ValueError("Expected nonempty finite (c,p,q) factor rows")

        def run():
            value, tail = X, factors
            if self.fp16_first:
                from ._cutlass_prec import floating
                c, p, q = factors[0]
                # Keep the previously validated FP16 evaluation order. This
                # starter is separate from the new INT8 operator path.
                value = floating(self, X, [(c*(p*p+q), 2*c*p, c)], eps,
                                 deterministic, "float16", factor_rows=factors[:1])
                tail = factors[1:]
            if not tail:
                return value
            return self._regular_i8_impl(value, tail, eps, deterministic,
                                         normalize=not self.fp16_first, factored=True)

        if not self.autotune:
            return run()
        def launch(tuned):
            previous = self._factory.autotune
            self._factory.autotune = tuned
            try:
                return run()
            finally:
                self._factory.autotune = previous
        return self._factory.select_pipeline("int8_factored", X,
            (factors, eps, deterministic, self.fp16_first, tuple(self.tiles.items())), launch)

    def _regular_i8(self, x, coeffs=None, eps=1e-7, deterministic=True, precision="int8", *, block=False):
        """Run INT8 iterations, optionally warming up with one FP16 iteration."""
        coeffs = _DEFAULT_NS_COEFFS if coeffs is None else coeffs
        if not self.fp16_first:
            if block:
                return _cutlass_bq.regular_bq(self, x, coeffs, eps, deterministic, precision)
            return self._regular_i8_impl(x, coeffs, eps, deterministic, precision)
        first = self._regular_prec(x, coeffs[:1], eps, deterministic, "float16")
        if len(coeffs) == 1:
            return first
        if block:
            return _cutlass_bq.regular_bq(self, first, coeffs[1:], eps, deterministic, precision, normalize=False)
        return self._regular_i8_impl(first, coeffs[1:], eps, deterministic, precision, normalize=False)

    @cache
    def _int8_tiles(self, device, l, h, overrides):
        """Resolve resource limits and explicit tiles once per matrix specification."""
        properties = device_properties(device)
        tma = (properties.major, properties.minor) in self._extension._tma_arches
        candidates = tuple(int8_candidates(properties.shared_memory_per_block_optin, tma))
        default_tile = int(l == 2048)
        defaults = {"aat": default_tile, "square": default_tile, "update": 0}
        if l == h == 4096:
            defaults = {"aat": 4 if tma else 2, "square": 1, "update": 4 if tma else 2}
        for operation, tile in overrides:
            if tile == 4 and not tma:
                raise ValueError("TMA tile 4 requires a supported build target; use TORCH_CUDA_ARCH_LIST=12.0a on SM120")
            if tile not in candidates:
                raise ValueError(f"Tile {tile} for {operation} exceeds this GPU's shared-memory limit")
            defaults[operation] = tile
        return {op: tile if tile in candidates else candidates[0] for op, tile in defaults.items()}, candidates

    def _regular_i8_impl(self, X, coeffs=None, eps=1e-7, deterministic=True, precision="int8", *, normalize=True, factored=False):
        coeffs = _DEFAULT_NS_COEFFS if coeffs is None else coeffs
        orig_shape, dev = X.shape, X.device
        batch, rows, cols, l, h, lp, hp = _matrix_spec(orig_shape, X.dtype, dev, "int8")
        X = X.contiguous().view(batch, rows, cols)
        ext = self._extension
        defaults, candidates = self._int8_tiles(dev, l, h, tuple(self.tiles.items()))
        # Zero all reduction slots once; each iteration consumes fresh slots.
        # Four INT8 stages use 56 bytes per batch entry, including input stats.
        maxima = torch.zeros((2 + 3 * len(coeffs), batch), device=dev, dtype=torch.float32)
        a8 = torch.empty((batch, lp, hp), device=dev, dtype=torch.int8)
        a8_t = torch.empty((batch, hp, lp), device=dev, dtype=torch.int8)
        a_scale = torch.empty(batch, device=dev, dtype=torch.float32)
        # These outputs have disjoint lifetimes: Gram -> square -> update.
        # Sharing storage also avoids cycling three large fp32 buffers through L2.
        workspace = torch.empty(batch * lp * hp, device=dev, dtype=torch.float32)
        aa32 = workspace[:batch * lp * lp].view(torch.int32).view(batch, lp, lp)
        aa8 = torch.empty((batch, lp, lp), device=dev, dtype=torch.int8)
        aa_scale = torch.empty(batch, device=dev, dtype=torch.float32)
        aa_diag = torch.empty((batch, lp), device=dev, dtype=torch.float32)
        b32 = aa32.view(torch.float32)
        b8 = aa8
        b_scale = torch.empty(batch, device=dev, dtype=torch.float32)
        b_diag = torch.empty((batch, lp), device=dev, dtype=torch.float32)
        intermediate = workspace.view(batch, lp, hp)
        output = workspace.view(X.dtype)[:batch * rows * cols].view(batch, rows, cols)

        ext.stats(X, maxima, maxima, not deterministic, 0, 1)
        norm = ((X.float().norm(dim=(1, 2)) if deterministic else maxima) if normalize
                else torch.ones(batch, device=dev, dtype=torch.float32))
        norm_slot = int(normalize and not deterministic)
        ext.input_quant(X, maxima, norm, a8, a8_t, a_scale, rows > cols,
                        deterministic or not normalize, eps if normalize else 0., 0, norm_slot)
        for iteration, row in enumerate(coeffs):
            aa_max, bmax, cmax = 2 + 3 * iteration, 3 + 3 * iteration, 4 + 3 * iteration
            if factored:
                gain, p, q = row
            else:
                a, b, c = row
            tile = defaults["aat"]
            if self._factory.autotune and "aat" not in self.tiles:
                def run_aat(tile):
                    ext.aat(a8, aa32, maxima, l, tile, aa_max)
                tile = self._factory.select("int8_factored" if factored else "int8", "aat",
                                            a8, (l,), candidates, tile, run_aat)
            ext.aat(a8, aa32, maxima, l, tile, aa_max)
            if factored:
                ext.factored_quant(aa32, maxima, a_scale, aa_diag,
                                   aa8, aa_scale, aa_diag, False, p, aa_max)
            else:
                ext.symmetric_quant(aa32, maxima, a_scale, aa_diag,
                                    aa8, aa_scale, aa_diag, False, 0, 0, 0, aa_max)
            tile = defaults["square"]
            if self._factory.autotune and "square" not in self.tiles:
                def run_square(tile):
                    if factored:
                        ext.factored_square(aa8, aa_scale, aa_diag, b32, maxima, l, tile, bmax)
                    else:
                        ext.square(aa8, aa_scale, aa_diag, b32, maxima, l, b, c, tile, bmax)
                tile = self._factory.select("int8_factored" if factored else "int8", "square",
                                            a8, (l,), candidates, tile, run_square)
            if factored:
                ext.factored_square(aa8, aa_scale, aa_diag, b32, maxima, l, tile, bmax)
            else:
                ext.square(aa8, aa_scale, aa_diag, b32, maxima, l, b, c, tile, bmax)
            if factored:
                ext.factored_quant(b32, maxima, aa_scale, aa_diag,
                                   b8, b_scale, b_diag, True, q, bmax)
            else:
                ext.symmetric_quant(b32, maxima, aa_scale, aa_diag,
                                    b8, b_scale, b_diag, True, a, b, c, bmax)
            final = iteration == len(coeffs) - 1
            tile = defaults["update"]
            args = (b8, b_scale, b_diag, a8_t, a_scale,
                    output if final else intermediate, maxima, l, h, final, rows > cols)
            if self._factory.autotune and "update" not in self.tiles:
                def run_update(tile):
                    if factored:
                        ext.factored_update(*args, tile, a8, gain, cmax)
                    else:
                        ext.update(*args, tile, a8, cmax)
                tile = self._factory.select("int8_factored" if factored else "int8", "update",
                    a8, (l, h, final, rows > cols, X.dtype), candidates, tile, run_update)
            if factored:
                ext.factored_update(*args, tile, a8, gain, cmax)
            else:
                ext.update(*args, tile, a8, cmax)
            if not final:
                ext.output_quant(intermediate, maxima, a8, a8_t, a_scale, cmax)
        return output.reshape(orig_shape)


__all__ = ["CutlassNSInt8"]
