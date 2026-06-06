#!/usr/bin/env python3
"""Benchmark i8muon NS methods.

Usage:
    python tests/bench.py --method _regular_i8 --shape 2048 2048 --precision int8
    python tests/bench.py --method _gram_prec --shape 4096 1024 --precision float16 --autotune --cudagraph
"""

import argparse
import os
import time
from datetime import datetime

import torch

from i8muon._ns import NSInt8, _DEFAULT_NS_COEFFS

DEVICE = "cuda"
COEFFS = _DEFAULT_NS_COEFFS

VALID_METHODS = [
    "_gram_i8", "_gram_prec", "_gram_bq",
    "_regular_i8", "_regular_prec", "_regular_bq",
]
VALID_PRECISIONS = ["int8", "float16", "float8_e4m3fn", "bfloat16", "float32"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark i8muon NS methods")
    parser.add_argument("--method", required=True, choices=VALID_METHODS,
                        help="NS method to benchmark")
    parser.add_argument("--shape", type=int, nargs=2, required=True, metavar=("M", "N"),
                        help="Matrix shape (M N)")
    parser.add_argument("--precision", default="int8", choices=VALID_PRECISIONS,
                        help="Precision (default: int8)")
    parser.add_argument("--autotune", action="store_true",
                        help="Enable autotune")
    parser.add_argument("--cudagraph", action="store_true",
                        help="Measure CUDA graph replay time")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (default: 5)")
    parser.add_argument("--average", type=int, default=20,
                        help="Benchmark iterations (default: 20)")
    args = parser.parse_args()

    M, N = args.shape
    gpu_name = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu_name}")
    print(f"Method: {args.method}  Shape: {M}x{N}  Precision: {args.precision}")
    print(f"Autotune: {args.autotune}  CUDA Graph: {args.cudagraph}")
    print(f"Warmup: {args.warmup}  Average: {args.average}")
    print()

    torch.manual_seed(42)
    X = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    X = X / X.norm()

    ns = NSInt8(autotune=args.autotune)
    method = getattr(ns, args.method)

    # ── Warmup ──
    print(f"Warmup ({args.warmup} iters)...", flush=True)
    for i in range(args.warmup):
        t0 = time.perf_counter()
        _ = method(X.clone(), coeffs=COEFFS, precision=args.precision)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"  {i+1}: {elapsed*1000:.1f} ms", flush=True)

    # ── Benchmark ──
    if args.cudagraph:
        print(f"Capturing CUDA graph...", flush=True)
        in_buf = torch.empty(M, N, device=DEVICE, dtype=torch.float32)
        in_buf.copy_(X)
        _ = method(in_buf, coeffs=COEFFS, precision=args.precision)
        _ = method(in_buf, coeffs=COEFFS, precision=args.precision)
        torch.cuda.synchronize()
        in_buf.copy_(X)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_buf = method(in_buf, coeffs=COEFFS, precision=args.precision)
        torch.cuda.synchronize()

        times = []
        for i in range(args.average):
            in_buf.copy_(X)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - t0))

        mode_str = "cudagraph_replay"
    else:
        times = []
        for i in range(args.average):
            Xi = X.clone()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = method(Xi, coeffs=COEFFS, precision=args.precision)
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - t0))

        mode_str = "direct_call"

    t = torch.tensor(times)
    print(f"\nResults ({mode_str}):")
    print(f"  mean: {t.mean():.3f} ms")
    print(f"  std:  {t.std():.3f} ms")
    print(f"  min:  {t.min():.3f} ms")
    print(f"  max:  {t.max():.3f} ms")

    # ── Write to bench_results.txt ──
    result_path = os.path.join(os.path.dirname(__file__), "bench_results.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{timestamp} | GPU={gpu_name} | method={args.method} | "
        f"shape=({M},{N}) | precision={args.precision} | "
        f"autotune={args.autotune} | cudagraph={args.cudagraph} | "
        f"warmup={args.warmup} | average={args.average} | "
        f"mean={t.mean():.3f}ms | std={t.std():.3f}ms | "
        f"min={t.min():.3f}ms | max={t.max():.3f}ms"
    )
    with open(result_path, "a+", encoding="utf8") as f:
        f.write(line + "\n")
    print(f"\nAppended to {result_path}")


if __name__ == "__main__":
    main()
