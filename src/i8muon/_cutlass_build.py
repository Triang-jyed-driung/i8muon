"""Load native CUDA extensions for the CUTLASS backend."""

import os
from functools import cache
from pathlib import Path


@cache
def device_properties(device):
    """Cache immutable hardware properties once per explicit CUDA device."""
    import torch
    return torch.cuda.get_device_properties(device)


def _cutlass_include():
    root = os.environ.get("CUTLASS_PATH")
    candidates = [Path(root)] if root else []
    candidates += [Path(__file__).parent / "_vendor" / "cutlass"]
    for candidate in candidates:
        include = candidate if (candidate / "cutlass" / "cutlass.h").is_file() else candidate / "include"
        if (include / "cutlass" / "cutlass.h").is_file():
            return str(include)
    raise RuntimeError("CUTLASS headers not found. Set CUTLASS_PATH to a CUTLASS source checkout.")


@cache
def load_extension(source="int8.cu", name="i8muon_cutlass"):
    import torch
    from torch.utils.cpp_extension import load, _get_cuda_arch_flags

    if not torch.cuda.is_available():
        raise RuntimeError("The CUTLASS backend requires a CUDA GPU.")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up the CUTLASS backend before capturing a CUDA graph.")
    arch_flags = _get_cuda_arch_flags()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        # CUTLASS enables Blackwell TMA for architecture-specific targets.
        arch_flags = [f.replace("compute_120,code=sm_120", "compute_120a,code=sm_120a")
                      for f in arch_flags if "code=compute_120" not in f]
    extension = load(
        name=name + "_native",
        sources=[str(Path(__file__).parent / "csrc" / source)],
        extra_include_paths=[_cutlass_include()],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr",
                          "--fmad=true", "-lineinfo"] + arch_flags,
        verbose=bool(int(os.environ.get("I8MUON_BUILD_VERBOSE", "0"))),
    )
    extension._tma_arches = frozenset(
        (major, minor) for major, minor in ((9, 0), (10, 0), (10, 1), (10, 3), (12, 0), (12, 1))
        if any(f"code=sm_{major}{minor}a" in flag or
               ((major, minor) == (9, 0) and "code=sm_90" in flag)
               for flag in arch_flags)
    )
    return extension
