"""Select precompiled CUDA specializations using timings on the input device."""
import statistics
import torch


def int8_candidates(shared_memory, tma=False):
    """Return tiles whose dynamic shared storage fits the input GPU."""
    storage = {0: 36864, 1: 49152, 2: 73728, 3: 98304, 4: 98304}
    tiles = (0, 1, 2, 3)
    if tma:
        tiles += (4,)
    return [tile for tile in tiles if storage[tile] <= shared_memory]


class KernelFactory:
    """Per-engine tuning cache; explicit tiles take precedence over measurements."""

    def __init__(self, autotune=False):
        self.autotune = autotune
        self.cache = {}
        self.timings = {}
        self.pipeline_cache = {}
        self.pipeline_timings = {}

    def select_pipeline(self, family, tensor, signature, launch):
        """Validate operation choices against the complete default pipeline."""
        if not self.autotune:
            return launch(False)
        device = tensor.device
        key = (family, device.index,
               tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), signature)
        if key in self.pipeline_cache:
            return launch(self.pipeline_cache[key])
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up this pipeline before capturing autotuned CUDA kernels")
            launch(True)  # Populate every operation cache before capture.
            graphs, outputs = {}, {}
            for tuned in (False, True):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[tuned] = launch(tuned)
                for _ in range(5):
                    graph.replay()
                graphs[tuned] = graph
            samples = {False: [], True: []}
            for trial in range(3):
                for tuned in ((False, True) if trial % 2 == 0 else (True, False)):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        graphs[tuned].replay()
                    end.record()
                    end.synchronize()
                    samples[tuned].append(start.elapsed_time(end) / 10)
            scores = {tuned: statistics.median(values) for tuned, values in samples.items()}
            # Require a full-pipeline win, not merely a hot-cache operation win.
            self.pipeline_cache[key] = scores[True] < scores[False] * 0.99
            self.pipeline_timings[key] = scores
        return launch(self.pipeline_cache[key])

    def select(self, family, operation, tensor, signature, candidates, default, launch):
        if not self.autotune:
            return default
        device = tensor.device
        key = (family, operation, device.index,
               tensor.dtype, tuple(tensor.shape), signature, tuple(candidates))
        if key in self.cache:
            return self.cache[key]
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up this shape and dtype before capturing autotuned CUDA kernels")
            graphs = {}
            for tile in candidates:
                launch(tile)
                # Replay a batch so Python dispatch gaps do not decide which
                # CUDA specialization wins, especially for small matrices.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(5):
                        launch(tile)
                for _ in range(20):
                    graph.replay()
                graphs[tile] = graph
            samples = {tile: [] for tile in candidates}
            for trial in range(3):
                order = list(candidates[trial % len(candidates):]) + list(candidates[:trial % len(candidates)])
                for tile in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):
                        graphs[tile].replay()
                    end.record()
                    end.synchronize()
                    samples[tile].append(start.elapsed_time(end) / 100)
            scores = {tile: statistics.median(values) for tile, values in samples.items()}
            best = min(scores, key=scores.get)
            # Small timing differences do not justify abandoning a known
            # default; isolated operation timings also omit pipeline locality.
            self.cache[key] = default if scores[best] >= scores[default] * 0.98 else best
            self.timings[key] = scores
        return self.cache[key]
