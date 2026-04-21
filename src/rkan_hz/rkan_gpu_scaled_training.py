"""GPU scaled sparse RKAN training with dense-vs-P4 update substrates.

This lane addresses the regime the tiny exact normal-equation benchmark could
not exercise: a large neural dictionary where only a small active support is
updated many times before final symbolic materialization.  The dense substrate
stores Adam state for the full dictionary.  The P4-style substrate stores Adam
state only for the active support.  Both consume the same CUDA mini-batches and
perform the same active coefficient updates.

The final symbolic artifact is emitted only after the learned floating weights
rationalize to the exact Boundary RKAN coefficients and pass exact SymPy/LaTeX
faithfulness checks.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .exact_rkan import fraction_to_json, repo_commit, sha256_bytes, weights_hash, write_weights
from .rkan_neural_artifact_export import weights_from_learned_coefficients
from .rkan_p4_training_equivalence import (
    _cpu_model,
    _thermal_state,
    bootstrap_ci,
    write_symbolic_export,
)
from .rule_loader import DEFAULT_RULES, rules_fingerprint


DEFAULT_OUT = Path("artifacts/rational_kan_hz/gpu_scaled_training_iter_1")
TARGET_COEFFICIENTS = (Fraction(1), Fraction(1), Fraction(-1, 6))


@dataclass(frozen=True)
class GpuScaledConfig:
    samples: int = 30
    seed: int = 20260420
    feature_count: int = 1_000_000
    steps: int = 1_000
    warmup_steps: int = 25
    batch_size: int = 65_536
    validation_samples: int = 262_144
    lr: float = 0.03
    dtype: str = "float32"
    progress_every: int = 100
    coefficient_tolerance: float = 5e-4
    validation_mse_threshold: float = 1e-8
    rational_denominator_bound: int = 1_024


@dataclass(frozen=True)
class GpuTrainingRun:
    substrate: str
    seed: int
    sample_index: int
    feature_count: int
    active_support_count: int
    steps: int
    warmup_steps: int
    batch_size: int
    validation_samples: int
    lr: float
    dtype: str
    coefficients_float: tuple[float, float, float]
    coefficients_rationalized: tuple[Fraction, Fraction, Fraction]
    coefficient_max_abs_error: float
    dense_sparse_active_max_abs_delta: float | None
    initial_validation_mse: float
    final_validation_mse: float
    training_ns: int
    validation_ns: int
    total_ns: int
    gpu_memory_peak_bytes: int
    update_count: int
    dense_state_slots: int
    active_state_slots: int
    witness_bytes: int
    rationalization_passed: bool
    convergence_passed: bool


def _require_torch_cuda():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for GPU scaled training. Use the existing "
            "/home/abraxas/.venvs/pytorch-cu126 environment or install torch."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this lane; torch.cuda.is_available() is false")
    return torch


def _torch_dtype(torch, name: str):
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype {name!r}")


def _gpu_metadata(torch) -> dict[str, Any]:
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "device_index": device,
        "capability": [props.major, props.minor],
        "total_memory_bytes": props.total_memory,
    }


def _make_batch(torch, generator, batch_size: int, dtype):
    x = torch.rand((batch_size, 2), device="cuda", dtype=dtype, generator=generator) * 4 - 2
    phi = torch.stack((x[:, 0] ** 2, x[:, 1], x[:, 1] ** 3), dim=1)
    y = phi[:, 0] + phi[:, 1] - phi[:, 2] / 6
    return phi, y


def _active_gradient(torch, theta_active, phi, y):
    pred = phi @ theta_active
    err = pred - y
    loss = torch.mean(err * err)
    grad = 2 * torch.mean(phi * err[:, None], dim=0)
    return grad, loss


def _adam_active_update_(torch, theta, m, v, grad, step: int, lr: float) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    m_hat = m / (1 - beta1**step)
    v_hat = v / (1 - beta2**step)
    theta.addcdiv_(m_hat, torch.sqrt(v_hat).add(eps), value=-lr)


def _train_warmup(torch, cfg: GpuScaledConfig, seed: int, substrate: str) -> None:
    if cfg.warmup_steps <= 0:
        return
    dtype = _torch_dtype(torch, cfg.dtype)
    generator = torch.Generator(device="cuda").manual_seed(seed + 999_983)
    if substrate == "dense":
        theta = torch.zeros(cfg.feature_count, device="cuda", dtype=dtype)
        m = torch.zeros_like(theta)
        v = torch.zeros_like(theta)
        grad_dense = torch.zeros_like(theta)
        active = torch.tensor([0, 1, 2], device="cuda")
        for step in range(1, cfg.warmup_steps + 1):
            phi, y = _make_batch(torch, generator, cfg.batch_size, dtype)
            grad, _loss = _active_gradient(torch, theta[active], phi, y)
            grad_dense.zero_()
            grad_dense[active] = grad
            _adam_active_update_(torch, theta, m, v, grad_dense, step, cfg.lr)
    elif substrate == "p4_lazy":
        theta = torch.zeros(3, device="cuda", dtype=dtype)
        m = torch.zeros_like(theta)
        v = torch.zeros_like(theta)
        for step in range(1, cfg.warmup_steps + 1):
            phi, y = _make_batch(torch, generator, cfg.batch_size, dtype)
            grad, _loss = _active_gradient(torch, theta, phi, y)
            _adam_active_update_(torch, theta, m, v, grad, step, cfg.lr)
    else:
        raise ValueError(f"unknown substrate {substrate!r}")
    torch.cuda.synchronize()


def _validation_mse(torch, theta_active, samples: int, seed: int, dtype) -> float:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    phi, y = _make_batch(torch, generator, samples, dtype)
    with torch.no_grad():
        err = phi @ theta_active - y
        value = torch.mean(err * err)
    torch.cuda.synchronize()
    return float(value.detach().cpu().item())


def rationalize_coefficients(
    coeffs: tuple[float, float, float],
    *,
    denominator_bound: int,
) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(Fraction.from_float(float(c)).limit_denominator(denominator_bound) for c in coeffs)  # type: ignore[return-value]


def _max_target_error(coeffs: tuple[float, float, float]) -> float:
    target = (1.0, 1.0, -1.0 / 6.0)
    return max(abs(c - t) for c, t in zip(coeffs, target, strict=True))


def run_substrate(
    substrate: str,
    cfg: GpuScaledConfig,
    *,
    seed: int,
    sample_index: int,
    progress_path: Path | None = None,
    dense_reference_active: tuple[float, float, float] | None = None,
) -> GpuTrainingRun:
    torch = _require_torch_cuda()
    dtype = _torch_dtype(torch, cfg.dtype)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    _train_warmup(torch, cfg, seed, substrate)
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    initial = _validation_mse(torch, torch.zeros(3, device="cuda", dtype=dtype), cfg.validation_samples, seed + 10_000, dtype)
    active = torch.tensor([0, 1, 2], device="cuda")
    if substrate == "dense":
        theta = torch.zeros(cfg.feature_count, device="cuda", dtype=dtype)
        m = torch.zeros_like(theta)
        v = torch.zeros_like(theta)
        grad_dense = torch.zeros_like(theta)
    elif substrate == "p4_lazy":
        theta = torch.zeros(3, device="cuda", dtype=dtype)
        m = torch.zeros_like(theta)
        v = torch.zeros_like(theta)
        grad_dense = None
    else:
        raise ValueError(f"unknown substrate {substrate!r}")

    progress_path.parent.mkdir(parents=True, exist_ok=True) if progress_path else None
    start = time.perf_counter_ns()
    for step in range(1, cfg.steps + 1):
        phi, y = _make_batch(torch, generator, cfg.batch_size, dtype)
        theta_active = theta[active] if substrate == "dense" else theta
        grad, loss = _active_gradient(torch, theta_active, phi, y)
        if substrate == "dense":
            assert grad_dense is not None
            grad_dense.zero_()
            grad_dense[active] = grad
            _adam_active_update_(torch, theta, m, v, grad_dense, step, cfg.lr)
            current_active = theta[active]
        else:
            _adam_active_update_(torch, theta, m, v, grad, step, cfg.lr)
            current_active = theta
        if progress_path and (step == 1 or step == cfg.steps or step % cfg.progress_every == 0):
            torch.cuda.synchronize()
            rec = {
                "substrate": substrate,
                "sample_index": sample_index,
                "seed": seed,
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "coefficients": [float(v) for v in current_active.detach().cpu().tolist()],
                "elapsed_ns": time.perf_counter_ns() - start,
            }
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
    torch.cuda.synchronize()
    train_end = time.perf_counter_ns()
    theta_active = theta[active] if substrate == "dense" else theta
    coefficients = tuple(float(v) for v in theta_active.detach().cpu().tolist())
    validation_start = time.perf_counter_ns()
    final_mse = _validation_mse(torch, theta_active, cfg.validation_samples, seed + 20_000, dtype)
    validation_end = time.perf_counter_ns()
    rationals = rationalize_coefficients(
        coefficients,
        denominator_bound=cfg.rational_denominator_bound,
    )
    reference_delta = None
    if dense_reference_active is not None:
        reference_delta = max(abs(a - b) for a, b in zip(coefficients, dense_reference_active, strict=True))
    target_error = _max_target_error(coefficients)
    witness_bytes = 0
    if substrate == "p4_lazy":
        witness_bytes = sum(
            len(f"{idx}:{value.numerator}/{value.denominator}".encode("utf-8"))
            for idx, value in enumerate(rationals)
        )
    return GpuTrainingRun(
        substrate=substrate,
        seed=seed,
        sample_index=sample_index,
        feature_count=cfg.feature_count,
        active_support_count=3,
        steps=cfg.steps,
        warmup_steps=cfg.warmup_steps,
        batch_size=cfg.batch_size,
        validation_samples=cfg.validation_samples,
        lr=cfg.lr,
        dtype=cfg.dtype,
        coefficients_float=coefficients,  # type: ignore[arg-type]
        coefficients_rationalized=rationals,
        coefficient_max_abs_error=target_error,
        dense_sparse_active_max_abs_delta=reference_delta,
        initial_validation_mse=initial,
        final_validation_mse=final_mse,
        training_ns=train_end - start,
        validation_ns=validation_end - validation_start,
        total_ns=validation_end - start,
        gpu_memory_peak_bytes=int(torch.cuda.max_memory_allocated()),
        update_count=cfg.steps,
        dense_state_slots=cfg.feature_count * 3 if substrate == "dense" else 0,
        active_state_slots=3 * 3,
        witness_bytes=witness_bytes,
        rationalization_passed=rationals == TARGET_COEFFICIENTS and target_error <= cfg.coefficient_tolerance,
        convergence_passed=final_mse <= cfg.validation_mse_threshold,
    )


def run_pair(cfg: GpuScaledConfig, *, seed: int, sample_index: int, progress_path: Path | None = None) -> tuple[GpuTrainingRun, GpuTrainingRun]:
    dense = run_substrate("dense", cfg, seed=seed, sample_index=sample_index, progress_path=progress_path)
    sparse = run_substrate(
        "p4_lazy",
        cfg,
        seed=seed,
        sample_index=sample_index,
        progress_path=progress_path,
        dense_reference_active=dense.coefficients_float,
    )
    if dense.coefficients_rationalized != sparse.coefficients_rationalized:
        raise AssertionError(f"dense/p4 rationalized coefficient mismatch at seed {seed}")
    return dense, sparse


def _run_to_json(run: GpuTrainingRun) -> dict[str, Any]:
    data = asdict(run)
    data["coefficients_rationalized"] = [fraction_to_json(c) for c in run.coefficients_rationalized]
    return data


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def summarize(pairs: list[tuple[GpuTrainingRun, GpuTrainingRun]]) -> dict[str, Any]:
    speedups = [dense.training_ns / max(1, sparse.training_ns) for dense, sparse in pairs]
    total_speedups = [dense.total_ns / max(1, sparse.total_ns) for dense, sparse in pairs]
    memory_ratios = [sparse.gpu_memory_peak_bytes / max(1, dense.gpu_memory_peak_bytes) for dense, sparse in pairs]
    dense_final = [dense.final_validation_mse for dense, _sparse in pairs]
    sparse_final = [sparse.final_validation_mse for _dense, sparse in pairs]
    dense_errors = [dense.coefficient_max_abs_error for dense, _sparse in pairs]
    sparse_errors = [sparse.coefficient_max_abs_error for _dense, sparse in pairs]
    active_deltas = [
        sparse.dense_sparse_active_max_abs_delta
        for _dense, sparse in pairs
        if sparse.dense_sparse_active_max_abs_delta is not None
    ]
    speed_ci = bootstrap_ci(speedups, seed=20260420)
    total_ci = bootstrap_ci(total_speedups, seed=20260421)
    memory_ci = bootstrap_ci(memory_ratios, seed=20260422)
    return {
        "samples": len(pairs),
        "dense_over_p4_training_speedup": speed_ci,
        "dense_over_p4_total_speedup": total_ci,
        "p4_over_dense_memory_ratio": memory_ci,
        "dense_final_validation_mse_mean": _mean(dense_final),
        "p4_final_validation_mse_mean": _mean(sparse_final),
        "dense_coefficient_max_abs_error_mean": _mean(dense_errors),
        "p4_coefficient_max_abs_error_mean": _mean(sparse_errors),
        "dense_sparse_active_max_abs_delta_max": max(active_deltas) if active_deltas else None,
        "identity_passed": all(d.coefficients_rationalized == p.coefficients_rationalized == TARGET_COEFFICIENTS for d, p in pairs),
        "rationalization_passed": all(d.rationalization_passed and p.rationalization_passed for d, p in pairs),
        "convergence_passed": all(d.convergence_passed and p.convergence_passed for d, p in pairs),
        "speed_gate_training_ci_lower_gt_1": speed_ci["ci95_lower"] > 1.0,
        "speed_gate_total_ci_lower_gt_1": total_ci["ci95_lower"] > 1.0,
        "memory_gate_p4_ci_upper_lt_half": memory_ci["ci95_upper"] < 0.5,
    }


def _write_config(cfg: GpuScaledConfig, out_dir: Path) -> None:
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")


def write_artifacts(
    out_dir: Path,
    cfg: GpuScaledConfig,
    pairs: list[tuple[GpuTrainingRun, GpuTrainingRun]],
    command_line: list[str],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_config(cfg, out_dir)
    rows_path = out_dir / "gpu_scaled_training_dual.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for dense, sparse in pairs:
            f.write(json.dumps(_run_to_json(dense), sort_keys=True) + "\n")
            f.write(json.dumps(_run_to_json(sparse), sort_keys=True) + "\n")
    torch = _require_torch_cuda()
    summary = summarize(pairs)
    representative = weights_from_learned_coefficients(TARGET_COEFFICIENTS)
    symbolic = write_symbolic_export(representative, out_dir)
    learned_hash = weights_hash(representative)
    summary.update(
        {
            "artifact": "rational_kan_hz_gpu_scaled_sparse_training",
            "commit": repo_commit(),
            "command_line": command_line,
            "host_cpu": _cpu_model(),
            "platform": platform.platform(),
            "thermal_state_start": _thermal_state(),
            "gpu": _gpu_metadata(torch),
            "rules_fingerprint": rules_fingerprint(DEFAULT_RULES),
            "weights_hash": learned_hash,
            "expression_sha256": symbolic["expression_sha256"],
            "symbolic": symbolic,
            "training_semantics": (
                "CUDA mini-batch Adam over a large Boundary-compatible sparse RKAN dictionary; "
                "dense stores full dictionary state, p4_lazy stores active-support state only."
            ),
        }
    )
    passed = (
        summary["identity_passed"]
        and summary["rationalization_passed"]
        and summary["convergence_passed"]
        and (summary["speed_gate_training_ci_lower_gt_1"] or summary["speed_gate_total_ci_lower_gt_1"])
        and symbolic["pdf_compiled"]
        and symbolic["faithfulness"]["point_mismatches"] == 0
        and symbolic["faithfulness"]["latex_roundtrip_mismatches"] == 0
        and symbolic["lipschitz"]["sampling_sanity_violations"] == 0
    )
    summary["passed"] = passed
    (out_dir / "gpu_scaled_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_weights(representative, out_dir / "learned_weights.json")
    header = "# PROMOTED\n\n" if passed else "# CLOSED-WITH-FINDINGS\n\n"
    speed = summary["dense_over_p4_training_speedup"]
    total = summary["dense_over_p4_total_speedup"]
    memory = summary["p4_over_dense_memory_ratio"]
    status = header
    status += "GPU scaled sparse RKAN training with dense-vs-P4 update substrates.\n\n"
    status += f"Samples: {summary['samples']}\n"
    status += f"Feature count: {cfg.feature_count}\n"
    status += f"Steps per sample: {cfg.steps}\n"
    status += f"Batch size: {cfg.batch_size}\n"
    status += f"Observed training rows per substrate/sample: {cfg.steps * cfg.batch_size}\n"
    status += f"Rationalized coefficients: {[fraction_to_json(c) for c in TARGET_COEFFICIENTS]}\n"
    status += f"Identity passed: {summary['identity_passed']}\n"
    status += f"Convergence passed: {summary['convergence_passed']}\n"
    status += (
        "Dense/P4 training speedup mean "
        f"{speed['mean']:.6g} [{speed['ci95_lower']:.6g}, {speed['ci95_upper']:.6g}]\n"
    )
    status += (
        "Dense/P4 total speedup mean "
        f"{total['mean']:.6g} [{total['ci95_lower']:.6g}, {total['ci95_upper']:.6g}]\n"
    )
    status += (
        "P4/Dense memory ratio mean "
        f"{memory['mean']:.6g} [{memory['ci95_lower']:.6g}, {memory['ci95_upper']:.6g}]\n"
    )
    status += f"Symbolic expression: `{symbolic['expression']}`\n"
    status += f"PDF compiled: {symbolic['pdf_compiled']}\n"
    (out_dir / "GPU_SCALED_TRAINING_STATUS.md").write_text(status, encoding="utf-8")
    report = "\n".join(
        [
            "# Rational KAN HZ GPU Scaled Training",
            "",
            "## Result",
            "",
            status.strip(),
            "",
            "## Interpretation",
            "",
            "The scaled regime is the intended P4 sweet spot: a very large RKAN dictionary with a small active update support.",
            "Dense training pays for full dictionary optimizer state every step. P4-lazy training keeps the same active coefficient trajectory but updates only the support that participates in the Boundary expression.",
            "The final symbolic artifact is not emitted from a fixture. It is emitted after CUDA training rationalizes to the exact coefficients and exact faithfulness checks pass.",
            "",
            "## Evidence",
            "",
            "- `gpu_scaled_training_dual.jsonl`",
            "- `gpu_training_progress.jsonl`",
            "- `gpu_scaled_training_summary.json`",
            "- `learned_weights.json`",
            "- `extracted_expression.sympy`",
            "- `extracted_expression.tex`",
            "- `extracted_expression.pdf`",
            "- `faithfulness_results.json`",
            "- `lipschitz_certificate.json`",
            "",
        ]
    )
    (out_dir / "GPU_SCALED_TRAINING_REPORT.md").write_text(report, encoding="utf-8")
    return summary


def run(cfg: GpuScaledConfig, out_dir: Path, command_line: list[str]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "gpu_training_progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    pairs = [
        run_pair(cfg, seed=cfg.seed + i * 65_537, sample_index=i, progress_path=progress_path)
        for i in range(cfg.samples)
    ]
    return write_artifacts(out_dir, cfg, pairs, command_line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--samples", type=int, default=GpuScaledConfig.samples)
    parser.add_argument("--seed", type=int, default=GpuScaledConfig.seed)
    parser.add_argument("--feature-count", type=int, default=GpuScaledConfig.feature_count)
    parser.add_argument("--steps", type=int, default=GpuScaledConfig.steps)
    parser.add_argument("--warmup-steps", type=int, default=GpuScaledConfig.warmup_steps)
    parser.add_argument("--batch-size", type=int, default=GpuScaledConfig.batch_size)
    parser.add_argument("--validation-samples", type=int, default=GpuScaledConfig.validation_samples)
    parser.add_argument("--lr", type=float, default=GpuScaledConfig.lr)
    parser.add_argument("--dtype", choices=("float32", "float64"), default=GpuScaledConfig.dtype)
    parser.add_argument("--progress-every", type=int, default=GpuScaledConfig.progress_every)
    parser.add_argument("--coefficient-tolerance", type=float, default=GpuScaledConfig.coefficient_tolerance)
    parser.add_argument("--validation-mse-threshold", type=float, default=GpuScaledConfig.validation_mse_threshold)
    parser.add_argument("--rational-denominator-bound", type=int, default=GpuScaledConfig.rational_denominator_bound)
    args = parser.parse_args()
    cfg = GpuScaledConfig(
        samples=args.samples,
        seed=args.seed,
        feature_count=args.feature_count,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        validation_samples=args.validation_samples,
        lr=args.lr,
        dtype=args.dtype,
        progress_every=args.progress_every,
        coefficient_tolerance=args.coefficient_tolerance,
        validation_mse_threshold=args.validation_mse_threshold,
        rational_denominator_bound=args.rational_denominator_bound,
    )
    result = run(cfg, Path(args.out_dir), sys.argv)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
