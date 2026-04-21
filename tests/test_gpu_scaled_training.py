from fractions import Fraction

import pytest

from rkan_hz.rkan_gpu_scaled_training import (
    GpuScaledConfig,
    rationalize_coefficients,
    run_pair,
    summarize,
)


def test_rationalize_coefficients_recovers_exact_symbolic_target():
    coeffs = (0.99999994, 1.0000001, -0.166666761)
    assert rationalize_coefficients(coeffs, denominator_bound=1_024) == (
        Fraction(1),
        Fraction(1),
        Fraction(-1, 6),
    )


def test_gpu_scaled_training_smoke_if_cuda_available(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = GpuScaledConfig(
        samples=1,
        seed=20260420,
        feature_count=4096,
        steps=80,
        warmup_steps=2,
        batch_size=4096,
        validation_samples=4096,
        progress_every=40,
        coefficient_tolerance=0.05,
        validation_mse_threshold=0.02,
    )
    dense, sparse = run_pair(cfg, seed=cfg.seed, sample_index=0, progress_path=tmp_path / "progress.jsonl")
    summary = summarize([(dense, sparse)])
    assert dense.coefficients_rationalized == sparse.coefficients_rationalized
    assert dense.convergence_passed
    assert sparse.convergence_passed
    assert summary["dense_over_p4_training_speedup"]["ci95_lower"] > 1.0
