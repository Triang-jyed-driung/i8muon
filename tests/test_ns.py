"""Unit tests for i8muon — int8 Newton-Schulz + Muon optimizer.

Run:
    python unittest.py

"""

import math
import os
import sys
import unittest
import warnings

# ── Prevent cwd shadowing of installed packages (tilelang/TVM type conflict) ──
if "" in sys.path:
    sys.path.remove("")

import tilelang
import torch
from torch import Tensor

def _zeropower_via_newtonschulz(
    grad: Tensor,
    coeffs: list[tuple[float, float, float]],
    eps: float,
    deterministic: bool = True
) -> Tensor:
    """Newton-Schulz orthogonalization in bf16 (pure PyTorch).

    Each element of ``coeffs`` is an (a, b, c) triple for one
    iteration; ``len(coeffs)`` determines the number of steps.
    """
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if not coeffs:
        raise ValueError("coeffs must be non-empty")
    for c in coeffs:
        if len(c) != 3:
            raise ValueError("Each entry must be a tuple of exactly 3 values")

    ortho_grad = grad.float()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for (a, b, c) in coeffs:
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from i8muon._ns import NSInt8, _DEFAULT_NS_COEFFS
    from i8muon import (
        Muon,
        _adjust_lr,
        # _zeropower_via_newtonschulz,
        _GRAM_ASPECT_THRESHOLD,
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_CUDA = DEVICE == "cuda"

# ═══════════════════════════════════════════════════════════════════
#  Test data generator
# ═══════════════════════════════════════════════════════════════════

COEFFS = [
    (3.9274, -8.7643, 5.3095),
    (3.4317, -5.4288, 2.3608),
    (3.5403, -5.3366, 2.2324),
    (3.6733, -4.8533, 1.8498),
    (2.6731, -2.4447, 0.7695),
]

CONDITION_U = [math.exp(x) for x in [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]]


def generate_rand_svd(M: int, N: int, u: float) -> torch.Tensor:
    """Generate a random matrix with log-normal singular values.

    Condition number ~ exp(|logsnr|) where logsnr ~ N(0, log(u)).
    """
    T = M > N
    if T:
        M, N = N, M
    U = torch.empty(M, M, device=DEVICE)
    V = torch.empty(N, N, device=DEVICE)
    torch.nn.init.orthogonal_(U.data)
    torch.nn.init.orthogonal_(V.data)
    S = torch.zeros(M, N, device=DEVICE)
    S.view(-1)[:: N + 1] = u ** torch.randn(M, device=DEVICE)
    R = U @ S @ V
    return R.mT if T else R


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════


def _orthogonality_error(X: torch.Tensor) -> float:
    """Frobenius norm of ``X @ X^T - I``, normalised by min(M,N)."""
    M, N = X.shape
    G = X @ X.mT
    I = torch.eye(M, device=X.device, dtype=X.dtype)
    return (G - I).norm().item() / math.sqrt(min(M, N))


def _spectral_error(X_est: torch.Tensor, X_ref: torch.Tensor) -> float:
    """Spectral norm of difference, relative to reference."""
    diff = X_est - X_ref
    s = torch.linalg.svdvals(diff.to(torch.float32))
    s_ref = torch.linalg.svdvals(X_ref.to(torch.float32))
    return (s[0] / (s_ref[0] + 1e-12)).item()


# ═══════════════════════════════════════════════════════════════════
#  Tests:  data generator
# ═══════════════════════════════════════════════════════════════════


class TestGenerateRandSVD(unittest.TestCase):
    def test_shapes(self):
        for M, N in [(64, 128), (128, 64), (256, 256)]:
            R = generate_rand_svd(M, N, u=math.exp(0.5))
            self.assertEqual(R.shape, (M, N))

    def test_finite(self):
        R = generate_rand_svd(192, 384, u=math.exp(1.0))
        self.assertTrue(R.isfinite().all())

    def test_condition_number_grows_with_u(self):
        """Larger u → larger spread of singular values."""
        c1 = torch.linalg.cond(generate_rand_svd(64, 64, math.exp(0.2)))
        c2 = torch.linalg.cond(generate_rand_svd(64, 64, math.exp(2.0)))
        self.assertLess(c1, c2)


# ═══════════════════════════════════════════════════════════════════
#  Tests:  NSInt8 — _gram_prec / _regular_i8
# ═══════════════════════════════════════════════════════════════════


@unittest.skipUnless(IS_CUDA, "requires CUDA")
class TestNSInt8_Correctness(unittest.TestCase):
    """Compare int8 NS output to float (pure PyTorch) reference."""

    @classmethod
    def setUpClass(cls):
        cls._ns = NSInt8(autotune=False)

    # ── helper ──

    def _compare(self, M, N, u, deterministic):
        X = generate_rand_svd(M, N, u)
        X_norm = X / X.norm()  # ensure spectral norm ≤ 1

        # Float reference (bf16)
        ref = _zeropower_via_newtonschulz(X_norm.clone(), COEFFS, eps=1e-7)
        ref = ref.to(torch.float32)

        # int8 _gram_prec
        out_g = self._ns._gram_prec(
            X_norm.clone(), coeffs=COEFFS, deterministic=deterministic
        )
        # int8 _regular_i8
        out_r = self._ns._regular_i8(
            X_norm.clone(), coeffs=COEFFS, deterministic=deterministic
        )

        return ref, out_g, out_r
    
    def _compare16(self, M, N, u, deterministic):
        X = generate_rand_svd(M, N, u)
        X_norm = X / X.norm()  # ensure spectral norm ≤ 1

        # Float reference (bf16)
        ref = _zeropower_via_newtonschulz(X_norm.clone(), COEFFS, eps=1e-7)
        ref = ref.to(torch.float32)

        out_g = self._ns._regular_i8(
            X_norm.clone(), coeffs=COEFFS, deterministic=deterministic
        )

        out_r = self._ns._gram_prec(
            X_norm.clone(), coeffs=COEFFS, deterministic=deterministic
        )
        return ref, out_g, out_r

    # ── tests ──

    def test_orthogonality_improves(self):
        """After NS, rows should be more orthonormal."""
        X = generate_rand_svd(192, 384, u=math.exp(0.2))
        X = X / X.norm()
        err_before = _orthogonality_error(X)
        out = self._ns._gram_prec(X, coeffs=COEFFS)
        err_after = _orthogonality_error(out)
        self.assertLess(err_after, err_before * 0.5)

    def test_small_condition(self):
        ref, g, r = self._compare16(192, 384, math.exp(0.2), deterministic=True)
        self.assertLess(_spectral_error(g, ref), 0.1)
        self.assertLess(_spectral_error(r, ref), 0.1)

    def test_medium_condition(self):
        ref, g, r = self._compare16(192, 384, math.exp(1.0), deterministic=True)
        self.assertLess(_spectral_error(g, ref), 0.4)
        self.assertLess(_spectral_error(r, ref), 0.4)

    def test_large_condition(self):
        ref, g, r = self._compare16(192, 384, math.exp(2.0), deterministic=True)
        # Large condition → ~40% error is expected
        self.assertLess(_spectral_error(g, ref), 0.7)
        self.assertLess(_spectral_error(r, ref), 0.7)

    def test_very_long(self):
        ref, g, r = self._compare16(8, 2048, math.exp(1.0), deterministic=True)
        self.assertLess(_spectral_error(g, ref), 0.3)
        self.assertLess(_spectral_error(r, ref), 0.3)
    
    def test_very_large(self):
        ref, g, r = self._compare16(7168, 7168, math.exp(0.8), deterministic=True)
        self.assertLess(_spectral_error(g, ref), 0.3)
        self.assertLess(_spectral_error(r, ref), 0.3)

    def test_very_small(self):
        ref, g, r = self._compare16(16, 16, math.exp(1.0), deterministic=True)
        self.assertLess(_spectral_error(g, ref), 0.2)
        self.assertLess(_spectral_error(r, ref), 0.2)

    def test_very_small_10(self):
        ref, g, r = self._compare16(10, 10, math.exp(1.0), deterministic=True)
        self.assertLess(_spectral_error(g, ref), 0.2)
        self.assertLess(_spectral_error(r, ref), 0.2)

    def test_gram_vs__regular_i8_similar(self):
        """_gram_prec and _regular_i8 should give similar outputs."""
        for u_exponent in [0.2, 0.5, 1.0]:
            u = math.exp(u_exponent)
            X = generate_rand_svd(128, 64, u)
            X = X / X.norm()
            g = self._ns._gram_prec(X.clone(), coeffs=COEFFS)
            r = self._ns._regular_i8(X.clone(), coeffs=COEFFS)
            # They use different algorithms but should be close
            self.assertLess(_spectral_error(g, r), 0.30,
                            f"_gram_prec vs _regular_i8 diverge at u=exp({u_exponent})")

    def test_tall_vs_wide(self):
        """Tall (M>N) and wide (M<N) should both work."""
        for shape in [(256, 64), (64, 256)]:
            X = generate_rand_svd(*shape, u=math.exp(1.0))
            X = X / X.norm()
            out = self._ns._gram_prec(X, coeffs=COEFFS)
            self.assertEqual(out.shape, shape)
            self.assertTrue(out.isfinite().all())

    def test_deterministic_vs_nondeterministic(self):
        """Non-deterministic should be close to deterministic."""
        X = generate_rand_svd(128, 64, u=math.exp(0.5))
        X = X / X.norm()
        det = self._ns._gram_prec(X.clone(), coeffs=COEFFS, deterministic=True)
        ndet = self._ns._gram_prec(X.clone(), coeffs=COEFFS, deterministic=False)
        self.assertLess(_spectral_error(ndet, det), 0.05)

    def test_cuda_graph_consistent(self):
        """CUDA graph replay must match direct call."""
        X = generate_rand_svd(256, 128, u=math.exp(0.5))
        X = X / X.norm()

        # Direct call
        direct = self._ns._gram_prec(X.clone(), coeffs=COEFFS)

        # Build fixed buffers + warmup + capture
        in_buf = torch.empty_like(X)
        in_buf.copy_(X)
        _ = self._ns._gram_prec(in_buf, coeffs=COEFFS)       # warmup 1
        _ = self._ns._gram_prec(in_buf, coeffs=COEFFS)       # warmup 2
        in_buf.copy_(X)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_buf = self._ns._gram_prec(in_buf, coeffs=COEFFS)
        g.replay()
        self.assertLess(_spectral_error(out_buf, direct), 1e-4)

    def test_cuda_graph__regular_i8(self):
        """Same graph-consistency test for _regular_i8 method."""
        X = generate_rand_svd(128, 128, u=math.exp(0.5))
        X = X / X.norm()

        direct = self._ns._regular_i8(X.clone(), coeffs=COEFFS)

        in_buf = torch.empty_like(X)
        in_buf.copy_(X)
        _ = self._ns._regular_i8(in_buf, coeffs=COEFFS)
        _ = self._ns._regular_i8(in_buf, coeffs=COEFFS)
        in_buf.copy_(X)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_buf = self._ns._regular_i8(in_buf, coeffs=COEFFS)
        g.replay()

        self.assertLess(_spectral_error(out_buf, direct), 1e-4)

    def test_cuda_graph_multiple_replays(self):
        """Graph should give same result across many replays with different input."""
        in_buf = torch.empty(128, 64, device=DEVICE, dtype=torch.float32)

        # Warmup + capture
        X0 = generate_rand_svd(128, 64, u=math.exp(0.5))
        X0 = X0 / X0.norm()
        in_buf.copy_(X0)
        _ = self._ns._gram_prec(in_buf, coeffs=COEFFS)
        _ = self._ns._gram_prec(in_buf, coeffs=COEFFS)
        in_buf.copy_(X0)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_buf = self._ns._gram_prec(in_buf, coeffs=COEFFS)

        for _ in range(5):
            X = generate_rand_svd(128, 64, u=math.exp(0.5))
            X = X / X.norm()
            direct = self._ns._gram_prec(X, coeffs=COEFFS)
            in_buf.copy_(X)
            g.replay()
            self.assertLess(_spectral_error(out_buf, direct), 1e-4,
                            "graph replay diverged from direct call")


# ═══════════════════════════════════════════════════════════════════
#  Tests:  _choose_ns_method
# ═══════════════════════════════════════════════════════════════════


# class TestChooseNSMethod(unittest.TestCase):
#     def test_gram_for_tall(self):
#         self.assertEqual(_choose_ns_method(4096, 1024), "_gram_prec")   # 4:1

#     def test_gram_for_wide(self):
#         self.assertEqual(_choose_ns_method(1024, 4096), "_gram_prec")   # 1:4

#     def test__regular_i8_for_square(self):
#         self.assertEqual(_choose_ns_method(1024, 1024), "_regular_i8")

#     def test__regular_i8_for_near_square(self):
#         self.assertEqual(_choose_ns_method(1500, 1000), "_regular_i8")  # 1.5:1 < 2

#     def test_boundary(self):
#         self.assertEqual(_choose_ns_method(4000, 1000), "_gram_prec")     # 2:1 ≥ 2
#         self.assertEqual(_choose_ns_method(3999, 1000), "_regular_i8")  # 1.999:1 < 2


# ═══════════════════════════════════════════════════════════════════
#  Tests:  _adjust_lr
# ═══════════════════════════════════════════════════════════════════


class TestAdjustLR(unittest.TestCase):
    def test_spectral_default(self):
        lr = _adjust_lr(0.01, "spectral", torch.Size([4096, 1024]))
        self.assertAlmostEqual(lr, 0.01 * math.sqrt(4096 / 1024), places=5)

    def test_original(self):
        lr = _adjust_lr(0.01, "original", torch.Size([4096, 1024]))
        self.assertAlmostEqual(lr, 0.01 * math.sqrt(4096 / 1024), places=5)

    def test_original_small_B(self):
        lr = _adjust_lr(0.01, "original", torch.Size([256, 1024]))
        self.assertAlmostEqual(lr, 0.01 * 1.0, places=5)  # max(1, 0.25) = 1

    def test_match_rms_adamw(self):
        lr = _adjust_lr(0.01, "match_rms_adamw", torch.Size([4096, 1024]))
        self.assertAlmostEqual(lr, 0.01 * 0.2 * math.sqrt(4096), places=5)

    def test_callable(self):
        fn = lambda lr, A, B: lr * (A + B)
        lr = _adjust_lr(0.01, fn, torch.Size([100, 200]))
        self.assertAlmostEqual(lr, 0.01 * 300)

    def test_spectral(self):
        lr2 = _adjust_lr(0.01, "spectral", torch.Size([4096, 1024]))
        self.assertEqual(lr2, 0.02)


# ═══════════════════════════════════════════════════════════════════
#  Tests:  Muon optimizer
# ═══════════════════════════════════════════════════════════════════


class _SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 32, bias=False)

    def forward(self, x):
        return self.fc(x)


@unittest.skipUnless(IS_CUDA, "requires CUDA")
class TestMuonOptimizer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = _SimpleModel().to(DEVICE)
        self.param = next(self.model.parameters())

    # ── construction ──

    def test_construct_default(self):
        opt = Muon([self.param])
        self.assertEqual(opt.defaults["lr"], 0.01)
        self.assertEqual(opt.defaults["weight_decay"], 0.0)

    def test_construct_nesterov(self):
        opt = Muon([self.param], momentum=(0.95, 0.9))
        self.assertEqual(opt.defaults["momentum"], (0.95, 0.9))

    def test_construct_scalar_momentum_normalised(self):
        opt = Muon([self.param], momentum=0.95)
        self.assertEqual(opt.defaults["momentum"], (0.95,))

    def test_rejects_non_2d(self):
        p = torch.nn.Parameter(torch.randn(64, 32, 3))
        with self.assertRaises(ValueError):
            Muon([p])

    # ── step: float path ──

    def test_step_float(self):
        opt = Muon([self.param], lr=0.01, momentum=0.0)
        x = torch.randn(8, 64, device=DEVICE)
        loss = self.model(x).sum()
        loss.backward()
        opt.step()
        self.assertTrue(self.param.isfinite().all())

    def test_step_nesterov_float(self):
        opt = Muon([self.param], lr=0.01, momentum=(0.95, 0.9))
        x = torch.randn(8, 64, device=DEVICE)
        for _ in range(5):
            opt.zero_grad()
            loss = self.model(x).sum()
            loss.backward()
            opt.step()
        self.assertTrue(self.param.isfinite().all())

    def test_step_weight_decay_float(self):
        opt = Muon([self.param], lr=0.01, weight_decay=0.1, momentum=0.0)
        p_before = self.param.clone()
        x = torch.randn(8, 64, device=DEVICE)
        loss = self.model(x).sum()
        loss.backward()
        opt.step()
        # With wd > 0, param should change more than without
        self.assertFalse(torch.allclose(self.param, p_before))

    # ── step: int8 path ──

    def test_step_int8(self):
        opt = Muon(
            [self.param], lr=0.01, momentum=0.0,
            precision="i8", autotune=False,
        )
        x = torch.randn(8, 64, device=DEVICE)
        loss = self.model(x).sum()
        loss.backward()
        opt.step()
        self.assertTrue(self.param.isfinite().all())

    def test_step_int8_nesterov(self):
        opt = Muon(
            [self.param], lr=0.01, momentum=(0.95, 0.9),
            precision="i8",
        )
        x = torch.randn(8, 64, device=DEVICE)
        for _ in range(5):
            opt.zero_grad()
            loss = self.model(x).sum()
            loss.backward()
            opt.step()
        self.assertTrue(self.param.isfinite().all())

    def test_int8_vs_float_close(self):
        """int8 and float paths should produce close parameter updates."""
        torch.manual_seed(42)
        # Use the same NS coefficients for fair comparison
        shared_coeffs = list(COEFFS)

        def _make_model():
            m = torch.nn.Linear(64, 32, bias=False).to(DEVICE)
            m.weight.data.normal_()
            return m

        # float
        m1 = _make_model()
        m2 = _make_model()
        with torch.no_grad():
            m2.load_state_dict(m1.state_dict())
        opt1 = Muon([m1.weight], lr=0.01, momentum=0.0,
                    ns_coefficients=shared_coeffs)
        x = torch.randn(8, 64, device=DEVICE)
        (m1(x).sum()).backward()
        opt1.step()

        # int8
        opt2 = Muon([m2.weight], lr=0.01, momentum=0.0,
                    ns_coefficients=shared_coeffs, precision="i8")
        m2.weight.grad = m1.weight.grad.clone()
        opt2.step()

        diff = (m1.weight - m2.weight).norm() / m1.weight.norm()
        self.assertLess(diff.item(), 0.10,
                        f"int8 vs float divergence: {diff.item():.4f}")

    # ── CUDA graph full-path test ──

    def test_step_cuda_graph_consistent(self):
        """Optimizer step with CUDA graph must match step without it."""
        torch.manual_seed(42)

        def _make_model():
            m = torch.nn.Linear(128, 64, bias=False).to(DEVICE)
            m.weight.data.normal_()
            return m

        grad = torch.randn(64, 128, device=DEVICE)

        # Without graph
        m1 = _make_model()
        m2 = _make_model()
        with torch.no_grad():
            m2.load_state_dict(m1.state_dict())
        opt1 = Muon([m1.weight], lr=1, momentum=0.0,
                    precision="i8")
        m1.weight.grad = grad.clone()
        opt1.step()

        # With graph
        opt2 = Muon([m2.weight], lr=1, momentum=0.0,
                    precision="i8", use_cuda_graph=True)
        m2.weight.grad = grad.clone()
        
        opt2.step()

        diff = (m1.weight - m2.weight).norm() / m1.weight.norm()
        self.assertLess(diff.item(), 1e-4,
                        f"graph step diverged: {diff.item():.6f}")

    def test_step_cuda_graph_multiple_params(self):
        """Multiple parameters: each shape gets its own graph."""
        class TwoLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(128, 64, bias=False)
                self.fc2 = torch.nn.Linear(64, 32, bias=False)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        model = TwoLayer().to(DEVICE)
        opt = Muon(
            [p for p in model.parameters() if p.ndim == 2],
            lr=0.01, momentum=0.0,
            precision="i8", use_cuda_graph=True,
        )
        x = torch.randn(8, 128, device=DEVICE)
        loss = model(x).sum()
        loss.backward()

        # Run a few steps — graphs are captured per-shape
        for _ in range(3):
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            opt.step()

        for p in model.parameters():
            self.assertTrue(p.isfinite().all())

    # ── custom coefficients ──

    def test_custom_coefficients(self):
        my_coeffs = [(3.0, -4.0, 2.0)] * 3
        opt = Muon([self.param], lr=0.01, momentum=0.0,
                   ns_coefficients=my_coeffs, precision="i8")
        x = torch.randn(8, 64, device=DEVICE)
        loss = self.model(x).sum()
        loss.backward()
        opt.step()
        self.assertTrue(self.param.isfinite().all())


# ═══════════════════════════════════════════════════════════════════
#  Tests:  condition number scan
# ═══════════════════════════════════════════════════════════════════


@unittest.skipUnless(IS_CUDA, "requires CUDA")
class TestConditionNumberScan(unittest.TestCase):
    """Sweep across condition numbers as specified."""

    @classmethod
    def setUpClass(cls):
        cls._ns = NSInt8(autotune=False)

    def test_scan_condition_numbers(self):
        """int8 NS should degrade gracefully as condition number grows."""
        results = []
        for u in CONDITION_U:
            X = generate_rand_svd(256, 128, u)
            X = X / X.norm()
            ref = _zeropower_via_newtonschulz(
                X.clone(), COEFFS, eps=1e-7
            ).to(torch.float32)
            out = self._ns._gram_prec(X.clone(), coeffs=COEFFS)
            err = _spectral_error(out, ref)
            results.append((math.log(u), err))

        # Verify monotonic degradation (not strictly monotonic,
        # but larger u should give larger error in general)
        log_u_small, err_small = results[0]   # exp(0.2)
        log_u_large, err_large = results[-1]  # exp(2.0)
        self.assertLess(err_small, err_large,
                        f"Expected err(u=exp(0.2)) < err(u=exp(2.0)), "
                        f"got {err_small:.4f} vs {err_large:.4f}")

        # Print results for inspection
        print("\n  Condition-number scan (spectral error vs float ref):")
        for log_u, err in results:
            print(f"    log(u)={log_u:5.2f}  cond≈exp(|N(0,{log_u:.2f})|)  "
                  f"error={err:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if IS_CUDA else 'CPU'})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Condition numbers to test: u ∈ {[f'{math.log(u):.1f}' for u in CONDITION_U]}")
    print()
    unittest.main(verbosity=2)
