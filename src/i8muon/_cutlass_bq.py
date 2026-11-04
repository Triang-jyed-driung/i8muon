"""Native 128x128 block-quantized Newton–Schulz for int8 and E4M3 FP8."""
import torch
from ._coeff import _DEFAULT_NS_COEFFS
from ._cutlass_build import load_extension, device_properties
from ._optimizer import _matrix_spec
from functools import cache


@cache
def _bq_tma(device):
    properties = device_properties(device)
    return (properties.major, properties.minor) in load_extension("bq.cu", "i8muon_cutlass_bq")._tma_arches


def regular_bq(engine, x, coeffs=None, eps=1e-7, deterministic=True, precision="float8_e4m3fn", *, normalize=True):
    coeffs = _DEFAULT_NS_COEFFS if coeffs is None else coeffs
    shape, device = x.shape, x.device
    batch, rows, cols, l, h, lp, hp = _matrix_spec(shape, x.dtype, device, precision, 128)
    x = x.contiguous().view(batch, rows, cols)
    ext = load_extension("bq.cu", "i8muon_cutlass_bq")
    tma = _bq_tma(device)
    dtype = getattr(torch, precision)
    matrix_bytes = batch * lp * hp
    output_bytes = batch * rows * cols * x.element_size()
    # The final update can overwrite its unused destination pair, but never
    # the source pair. Choose the initial pair using the iteration parity.
    destination = torch.empty(max(output_bytes, 2 * matrix_bytes), device=device, dtype=torch.uint8)
    other = torch.empty(2 * matrix_bytes, device=device, dtype=torch.uint8)
    initial_a, initial_c = (other, destination) if len(coeffs) % 2 else (destination, other)
    a = initial_a[:matrix_bytes].view(dtype).view(batch, lp, hp)
    at = initial_a[matrix_bytes:2 * matrix_bytes].view(dtype).view(batch, hp, lp)
    c = initial_c[:matrix_bytes].view(dtype).view(batch, lp, hp)
    ct = initial_c[matrix_bytes:2 * matrix_bytes].view(dtype).view(batch, hp, lp)
    sa = torch.empty((batch, lp // 128, hp // 128), device=device, dtype=torch.float32)
    sc = torch.empty_like(sa)
    b = torch.empty((batch, lp, lp), device=device, dtype=dtype)
    sb = torch.empty((batch, lp // 128, lp // 128), device=device, dtype=torch.float32)
    db = torch.empty((batch, lp), device=device, dtype=torch.float32)
    daa = torch.empty_like(db)
    empty = torch.empty(0, device=device, dtype=torch.float32)
    output = destination[:output_bytes].view(x.dtype).view(batch, rows, cols)
    norm = x.float().norm(dim=(1, 2)) if normalize else torch.ones(batch, device=device, dtype=torch.float32)
    ext.quant(x, norm, a, at, sa, rows > cols, eps if normalize else 0.)
    tile = int(tma)
    gemm = ext.gemm
    if engine._factory.autotune:
        def gemm(*args):
            default = args[-1]
            args = args[:-1]
            mode, final, transpose = args[10], args[16], args[17]
            tile = engine._factory.select("bq", mode, args[0],
                (args[11], args[12], final, transpose, args[6].dtype),
                [0, 2, 1] if tma else [0, 2], default, lambda tile: ext.gemm(*args, tile))
            ext.gemm(*args, tile)

    for iteration, (ca, cb, cc) in enumerate(coeffs):
        # Gram output lives in the update output's otherwise unused storage.
        aa = c.reshape(-1)[:batch * lp * lp].view(batch, lp, lp)
        saa = sc.reshape(-1)[:batch * (lp // 128)**2].view(batch, lp // 128, lp // 128)
        gemm(a, sa, empty, a, sa, empty, aa, empty, saa, daa,
                 0, l, l, 0, 0, 0, False, False, tile)
        gemm(aa, saa, daa, aa, saa, empty, b, empty, sb, db,
                 1, l, l, ca, cb, cc, False, False, tile)
        final = iteration == len(coeffs) - 1
        gemm(b, sb, db, at, sa, a,
                 output if final else c, empty if final else ct, empty if final else sc, empty,
                 2, l, h, 0, 0, 0, final, rows > cols, tile)
        if not final:
            a, c, at, ct, sa, sc = c, a, ct, at, sc, sa
    return output.reshape(shape)
