"""A small, offline example of matrix Muon plus AdamW for vector parameters.

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python examples/optimizer.py --precision int8
"""
import argparse

import torch
from i8muon import Muon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", default="int8",
                        choices=("auto", "float16", "bfloat16", "float32", "int8", "int8_bq", "fp8_bq"))
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--all-int8", action="store_true")
    parser.add_argument("--gram", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.GELU(),
                                torch.nn.Linear(128, 32)).cuda()
    matrices = [p for p in model.parameters() if p.ndim >= 2]
    vectors = [p for p in model.parameters() if p.ndim < 2]
    auxiliary = torch.optim.AdamW(vectors, lr=1e-3)
    optimizer = Muon(matrices, lr=1e-3, precision=args.precision, ns_steps=args.ns_steps,
                     fp16_first=not args.all_int8, use_gram=args.gram,
                     autotune=args.autotune, use_cuda_graph=True,
                     scalar_optimizer=auxiliary)
    for step in range(3):
        inputs, targets = torch.randn(8, 128, device="cuda"), torch.randn(8, 32, device="cuda")
        loss = (model(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"step={step + 1} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
