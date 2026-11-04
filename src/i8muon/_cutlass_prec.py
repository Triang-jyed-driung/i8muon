"""Native floating-point regular and Gram-form Newton-Schulz."""
import os
from functools import cache
import torch

from ._coeff import _DEFAULT_NS_COEFFS
from ._cutlass_build import load_extension, device_properties
from ._optimizer import _matrix_spec



@cache
def _floating_tiles(device, precision):
    properties = device_properties(device)
    ext = load_extension("prec.cu", "i8muon_cutlass_prec")
    limit = properties.shared_memory_per_block_optin
    candidates = tuple(tile for tile, size in enumerate(ext.shared_memory(precision)) if size <= limit)
    if not candidates:
        raise ValueError("Insufficient shared memory for native floating-point kernels")
    return candidates


@cache
def _gram_order(order, steps):
    if not order or order[0] != "0" or set(order) - {"0", "1"}:
        raise ValueError("I8MUON_GRAM_RESET must be a binary string starting with 0")
    return order + "0" * max(0, steps - len(order))


def floating(engine, x, coeffs=None, eps=1e-7, deterministic=True, precision="float16", *, gram=False, factor_rows=None):
    coeffs = _DEFAULT_NS_COEFFS if coeffs is None else coeffs
    shape, device = x.shape, x.device
    batch, rows, cols, l, h, lp, hp = _matrix_spec(shape, x.dtype, device, precision)
    x = x.contiguous().view(batch, rows, cols)
    order = _gram_order(os.environ.get("I8MUON_GRAM_RESET", "00100"), len(coeffs)) if gram else None
    dtype = getattr(torch, precision)
    ext = load_extension("prec.cu", "i8muon_cutlass_prec")
    candidates = _floating_tiles(device, precision)
    default = 1 if l >= 512 and 1 in candidates else 0
    empty = torch.empty(0, device=device, dtype=dtype)
    elements = batch * lp * hp

    def pair():
        storage = torch.empty(2 * elements, device=device, dtype=dtype)
        return storage, storage[:elements].view(batch, lp, hp), storage[elements:].view(batch, hp, lp)

    if not gram and len(coeffs) == 1:
        # A single stage only writes the final output, never its transpose.
        # Keep room for the Gram scratch and for a wider output dtype.
        element_size = torch.finfo(dtype).bits // 8
        output_bytes = batch * rows * cols * x.element_size()
        output_elements = (output_bytes + element_size - 1) // element_size
        destination = torch.empty(max(elements, output_elements), device=device, dtype=dtype)
        dest = destination[:elements].view(batch, lp, hp)
        dest_t = empty
    else:
        destination, dest, dest_t = pair()
    other, other_row, other_t = pair()
    if gram or len(coeffs) % 2:
        a, at, y, yt = other_row, other_t, dest, dest_t
    else:
        a, at, y, yt = dest, dest_t, other_row, other_t
    output = (torch.empty((batch, rows, cols), device=device, dtype=x.dtype) if gram
              else destination.view(x.dtype)[:batch * rows * cols].view(batch, rows, cols))
    # Accumulate before casting: an fp16 gradient norm can exceed 65504.
    if not gram and x.dtype != torch.float32:
        # Neither half of this pair is initialized yet. Its bytes also fit
        # the same FP32 conversion that x.float() would otherwise allocate.
        norm_input = other.view(torch.float32)[:x.numel()].view(x.shape)
        norm_input.copy_(x)
        norm = norm_input.norm(dim=(1, 2))
    else:
        norm = x.float().norm(dim=(1, 2))
    ext.normalize(x, norm, a, at, rows > cols, eps)
    q0 = torch.empty((batch, lp, lp), device=device, dtype=dtype) if gram else None

    gemm = ext.gemm
    if engine._factory.autotune:
        family = "gram_prec" if gram else ("regular_prec_factored" if factor_rows is not None else "regular_prec")
        def gemm(left, right_t, out, out_t, mode, logical_m, logical_n,
                 symmetric, final, transpose, ca, cb, cc, tile):
            signature = (logical_m, logical_n, tuple(right_t.shape), final, transpose, out.dtype, symmetric)
            def launch(tile):
                ext.gemm(left, right_t, out, out_t, mode, logical_m, logical_n,
                         symmetric, final, transpose, ca, cb, cc, tile)
            tile = engine._factory.select(family, mode, left, signature, candidates, tile, launch)
            launch(tile)

    if not gram:
        for iteration, (ca, cb, cc) in enumerate(coeffs):
            # The unused update destination can hold the preceding Gram output.
            r = y.reshape(-1)[:batch * lp * lp].view(batch, lp, lp)
            gemm(a, a, r, empty, 0, l, l, True, False, False, 0., 0., 0., default)
            # Gram consumes the row-major A; only its disjoint transpose is
            # needed by the update. Reuse the dead rows for the polynomial.
            q0 = a.reshape(-1)[:batch * lp * lp].view(batch, lp, lp)
            if factor_rows is not None:
                c, p, q = factor_rows[iteration]
                r.diagonal(dim1=-2, dim2=-1).add_(p)
                ca, cb, cc = c*q, 0., c
            gemm(r, r, q0, empty, 1, l, l, True, False, False, ca, cb, cc, default)
            final = iteration == len(coeffs) - 1
            gemm(q0, at, output if final else y, empty if final else yt,
                 0, l, h, False, final, final and rows > cols, 0., 0., 0., default)
            if not final:
                a, y, at, yt = y, a, yt, at
    else:
        r = torch.empty_like(q0)
        z = torch.empty_like(q0)
        q1 = torch.empty_like(q0)
        for iteration, (ca, cb, cc) in enumerate(coeffs):
            kind = order[iteration]
            refresh = iteration + 1 < len(coeffs) and order[iteration + 1] == "0"
            if kind == "1":
                gemm(q0, at, y, yt, 0, l, h, False, False, False, 0., 0., 0., default)
                a, y, at, yt = y, a, yt, at
            if kind == "1" or iteration == 0:
                gemm(a, a, r, empty, 0, l, l, True, False, False, 0., 0., 0., default)
                gemm(r, r, q0, empty, 1, l, l, True, False, False, ca, cb, cc, default)
                if refresh:
                    gemm(r, q0, q1, empty, 0, l, l, True, False, False, 0., 0., 0., default)
                    gemm(q0, q1, r, empty, 0, l, l, True, False, False, 0., 0., 0., default)
            else:
                gemm(r, r, z, empty, 1, l, l, True, False, False, ca, cb, cc, default)
                gemm(z, q0, q1, empty, 0, l, l, True, False, False, 0., 0., 0., default)
                q0, q1 = q1, q0
                if refresh:
                    gemm(r, z, q1, empty, 0, l, l, True, False, False, 0., 0., 0., default)
                    gemm(z, q1, r, empty, 0, l, l, True, False, False, 0., 0., 0., default)
        gemm(q0, at, output, empty, 0, l, h, False, True, rows > cols, 0., 0., 0., default)
    return output.reshape(shape)
