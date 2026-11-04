"""All-INT8 grouped cubic evaluation with a terminal quintic."""
import math

import torch
from ._cutlass_build import device_properties


def grouped_cubic(engine, x, input_gain, cubic_gains, terminal_abc, eps, *, diagonal_shifts=None):
    """T=b1*normalized(X); T <- -gain*(T@T.T-I)@T, then ABC quintic.

    Each cubic has exactly four native launches: aat, factored_quant,
    factored_update, output_quant. Each maximum has its own zero-initialized
    slot, so no resets are launched inside the loop. Graph replay reexecutes
    the single initialization before the loop. Setup is counted separately.
    """
    gains = tuple(float(g) for g in cubic_gains)
    abc = tuple(float(v) for v in terminal_abc)
    shifts = (-1.,)*len(gains) if diagonal_shifts is None else tuple(float(v) for v in diagonal_shifts)
    if (not gains or len(abc) != 3 or not all(map(math.isfinite, (*gains, *abc, input_gain, eps)))
            or len(shifts) != len(gains) or not all(map(math.isfinite, shifts))
            or input_gain <= 0 or eps <= 0):
        raise ValueError("Expected finite cubic gains, terminal ABC, positive input_gain and eps")
    if engine.autotune or engine.tiles:
        raise ValueError("Grouped cubic evaluation uses default tiles with autotune=False")
    if (x.ndim < 2 or not x.is_cuda or min(x.shape[-2:]) < 1
            or x.dtype not in (torch.float32, torch.float16, torch.bfloat16)):
        raise ValueError("Expected a nonempty FP32/FP16/BF16 CUDA matrix or matrix batch")
    if device_properties(x.device).major < 8:
        raise ValueError("CUTLASS requires compute capability >= 8.0")
    shape = x.shape
    x = x.contiguous().reshape(-1, *shape[-2:])
    batch, rows, cols = x.shape
    l, h = min(rows, cols), max(rows, cols)
    lp, hp = (l+15)//16*16, (h+15)//16*16
    if not 1 <= batch <= 65535 or hp*127*127 > 2**31-1:
        raise ValueError("Invalid batch or int32 accumulator overflow")
    ext, dev = engine._extension, x.device
    # Disjoint maxima prevent both stale reductions and intra-kernel reset races.
    maxima = torch.zeros((2*len(gains)+5, batch), dtype=torch.float32, device=dev)
    aq = torch.empty((batch, lp, hp), dtype=torch.int8, device=dev)
    at = torch.empty((batch, hp, lp), dtype=torch.int8, device=dev)
    sa = torch.empty(batch, dtype=torch.float32, device=dev)
    work = torch.empty((batch, lp, hp), dtype=torch.float32, device=dev)
    gram = work.flatten()[:batch*lp*lp].view(torch.int32).view(batch, lp, lp)
    gq = torch.empty((batch, lp, lp), dtype=torch.int8, device=dev)
    sg = torch.empty(batch, dtype=torch.float32, device=dev)
    dg = torch.empty((batch, lp), dtype=torch.float32, device=dev)
    sp = torch.empty_like(sg)
    dp = torch.empty_like(dg)
    out = work.view(x.dtype).flatten()[:batch*rows*cols].view(batch, rows, cols)
    ext.stats(x, maxima, maxima, False, 0, 1)
    # Fold b1 into the dequantization scale; no floating intermediate cast.
    norm = x.float().norm(dim=(1, 2)).clamp_min(eps)/input_gain
    ext.input_quant(x, maxima, norm, aq, at, sa, rows > cols, True, 0.)
    tile = int(l == 2048)
    aat_tile = 2 if l == h == 4096 else tile
    update_tile = 2 if l == h == 4096 else 0
    for i, gain in enumerate(gains):
        gm, ym = 2 + 2 * i, 3 + 2 * i
        ext.aat(aq, gram, maxima, l, aat_tile, gm)
        ext.factored_quant(gram, maxima, sa, dg, gq, sg, dg, False, shifts[i], gm)
        ext.factored_update(gq, sg, dg, at, sa, work, maxima, l, h,
                            False, False, update_tile, aq, -gain, ym)
        ext.output_quant(work, maxima, aq, at, sa, ym)
    gm, pm, ym = 2 + 2 * len(gains), 3 + 2 * len(gains), 4 + 2 * len(gains)
    a, b, c = abc
    ext.aat(aq, gram, maxima, l, aat_tile, gm)
    ext.symmetric_quant(gram, maxima, sa, dg, gq, sg, dg, False, 0., 0., 0., gm)
    ext.square(gq, sg, dg, gram.view(torch.float32), maxima, l, b, c, tile, pm)
    ext.symmetric_quant(gram.view(torch.float32), maxima, sg, dg, gq, sp, dp, True, a, b, c, pm)
    ext.update(gq, sp, dp, at, sa, out, maxima, l, h, True, rows > cols, update_tile, aq, ym)
    return out.reshape(shape)


def cubic_abc(engine, x, coefficients, eps):
    """Apply saved cubic ABC rows without an external input multiplier.

    (a I + b G) X = b (G + a/b I) X. This preserves two INT8 GEMMs
    and four launches per cubic, with an independent FP32 diagonal.
    Input gain may be absorbed into the first cubic coefficient row.
    """
    rows = tuple(tuple(float(v) for v in row) for row in coefficients)
    if (len(rows) < 2 or any(len(r) != 3 or not all(map(math.isfinite, r)) for r in rows)
            or any(c != 0. or a <= 0. or b >= 0. for a,b,c in rows[:-1])):
        raise ValueError('Expected finite cubic ABC rows (a>0,b<0,c=0), then one terminal quintic')
    return grouped_cubic(engine, x, 1., [-r[1] for r in rows[:-1]], rows[-1], eps,
                         diagonal_shifts=[r[0]/r[1] for r in rows[:-1]])
