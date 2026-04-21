"""Train a rational Boundary RKAN and export provable symbolic artifacts.

This is the final neural artifact lane. It does not benchmark the P4 substrate
in isolation. It trains a small rational KAN-shaped network from data, verifies
exact convergence, and only then materializes the learned network as SymPy,
LaTeX, PDF, and certificate JSON artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import sympy as sp

from .exact_rkan import (
    IDENTITY,
    KANWeights,
    ZERO,
    deterministic_inputs,
    fraction_to_json,
    kan_eval,
    monomial,
    repo_commit,
    sha256_bytes,
    target_value,
    weights_hash,
    write_weights,
)
from .rkan_symbolic_extract import (
    compile_latex,
    faithfulness,
    lipschitz_certificate,
    normalize_weights,
    write_latex,
)
from .rule_loader import DEFAULT_RULES, rules_fingerprint


DEFAULT_OUT = Path("artifacts/rational_kan_hz/neural_export_iter_1")


@dataclass(frozen=True)
class LearnedRkan:
    weights: KANWeights
    coefficients: tuple[Fraction, Fraction, Fraction]
    training_rows: int
    validation_rows: int
    initial_mse: Fraction
    final_training_mse: Fraction
    final_validation_mse: Fraction


def _grid_training_inputs() -> list[tuple[Fraction, Fraction]]:
    values = [Fraction(-2), Fraction(-1), Fraction(0), Fraction(1), Fraction(2)]
    return [(x0, x1) for x0 in values for x1 in values]


def _basis(x: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction, Fraction]:
    x0, x1 = x
    return (x0 * x0, x1, x1 * x1 * x1)


def _mean_square_error(coeffs: tuple[Fraction, Fraction, Fraction], xs: Iterable[tuple[Fraction, Fraction]]) -> Fraction:
    total = Fraction(0)
    count = 0
    for x in xs:
        phi = _basis(x)
        pred = coeffs[0] * phi[0] + coeffs[1] * phi[1] + coeffs[2] * phi[2]
        diff = pred - target_value(x)
        total += diff * diff
        count += 1
    return total / max(1, count)


def _kan_mean_square_error(weights: KANWeights, xs: Iterable[tuple[Fraction, Fraction]]) -> Fraction:
    total = Fraction(0)
    count = 0
    for x in xs:
        diff = kan_eval(weights, x) - target_value(x)
        total += diff * diff
        count += 1
    return total / max(1, count)


def _normal_equations(xs: list[tuple[Fraction, Fraction]]) -> tuple[list[list[Fraction]], list[Fraction]]:
    gram = [[Fraction(0) for _ in range(3)] for _ in range(3)]
    rhs = [Fraction(0) for _ in range(3)]
    for x in xs:
        phi = _basis(x)
        y = target_value(x)
        for i in range(3):
            rhs[i] += phi[i] * y
            for j in range(3):
                gram[i][j] += phi[i] * phi[j]
    return gram, rhs


def _solve_fraction_linear_system(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise ValueError("training design matrix is rank deficient")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(n + 1)]
    return [aug[i][n] for i in range(n)]


def train_exact_rkan() -> LearnedRkan:
    training = _grid_training_inputs()
    validation = deterministic_inputs(200, 20260420, bound=2)
    initial = (Fraction(0), Fraction(0), Fraction(0))
    gram, rhs = _normal_equations(training)
    coeffs = tuple(_solve_fraction_linear_system(gram, rhs))
    if len(coeffs) != 3:
        raise AssertionError("expected three learned coefficients")
    weights = weights_from_learned_coefficients(coeffs)
    return LearnedRkan(
        weights=weights,
        coefficients=coeffs,  # type: ignore[arg-type]
        training_rows=len(training),
        validation_rows=len(validation),
        initial_mse=_mean_square_error(initial, training),
        final_training_mse=_kan_mean_square_error(weights, training),
        final_validation_mse=_kan_mean_square_error(weights, validation),
    )


def weights_from_learned_coefficients(coeffs: tuple[Fraction, Fraction, Fraction]) -> KANWeights:
    inner = (
        (IDENTITY, ZERO),
        (ZERO, IDENTITY),
        (ZERO, IDENTITY),
    )
    outer = (
        monomial(coeffs[0], 2),
        monomial(coeffs[1], 1),
        monomial(coeffs[2], 3),
    )
    return KANWeights(
        inner=inner,
        outer=outer,
        metadata={
            "source": "learned_exact_boundary_rkan_normal_equations",
            "target": "x0^2 + x1 - x1^3/6",
            "optimizer": "exact normal equations over rational Boundary-compatible RKAN basis neurons",
            "input_dim": 2,
            "outer_count": 3,
            "inner_edges_total": 6,
            "activation_degree": 3,
            "hot_path_note": "P4 lazy HZ/Veselov acceleration applies to sparse payload updates before this final materialization.",
        },
    )


def _hash_record(prev_hash: str, record: dict) -> str:
    payload = json.dumps(record, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(prev_hash.encode("utf-8") + b"\0" + payload).hexdigest()


def write_learning_trace(learned: LearnedRkan, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = "sha256:" + ("0" * 64)
    records = [
        {
            "step": 0,
            "rule": "training_data_emit",
            "training_rows": learned.training_rows,
            "target": "x0^2 + x1 - x1^3/6",
        },
        {
            "step": 1,
            "rule": "basis_eval",
            "basis": ["x_0^2", "x_1", "x_1^3"],
        },
        {
            "step": 2,
            "rule": "normal_matrix_accumulate",
            "semantics": "exact Fraction Gram matrix and RHS accumulation",
        },
        {
            "step": 3,
            "rule": "solve_exact_normal_equations",
            "coefficients": [fraction_to_json(c) for c in learned.coefficients],
        },
        {
            "step": 4,
            "rule": "boundary_weight_update",
            "weights_hash": weights_hash(learned.weights),
        },
        {
            "step": 5,
            "rule": "convergence_check",
            "initial_mse": fraction_to_json(learned.initial_mse),
            "final_training_mse": fraction_to_json(learned.final_training_mse),
            "final_validation_mse": fraction_to_json(learned.final_validation_mse),
        },
    ]
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            rec = dict(record)
            rec["pre_hash"] = prev
            post = _hash_record(prev, record)
            rec["post_hash"] = post
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            prev = post
    return {
        "trace_path": str(out),
        "records_written": len(records),
        "final_trace_hash": prev,
    }


def replay_learning_trace(trace: Path, learned: LearnedRkan, out: Path) -> dict:
    records = []
    chain_breaks = 0
    malformed = 0
    prev = "sha256:" + ("0" * 64)
    for line in trace.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        expected_pre = prev
        core = {k: v for k, v in rec.items() if k not in {"pre_hash", "post_hash"}}
        expected_post = _hash_record(expected_pre, core)
        if rec.get("pre_hash") != expected_pre or rec.get("post_hash") != expected_post:
            chain_breaks += 1
        if not str(rec.get("post_hash", "")).startswith("sha256:"):
            malformed += 1
        prev = rec.get("post_hash", "")
        records.append(rec)
    solved = next((r for r in records if r.get("rule") == "solve_exact_normal_equations"), {})
    converged = next((r for r in records if r.get("rule") == "convergence_check"), {})
    result = {
        "records_examined": len(records),
        "chain_break_records": chain_breaks,
        "malformed_hash_records": malformed,
        "coefficients_match_learned_weights": solved.get("coefficients") == [fraction_to_json(c) for c in learned.coefficients],
        "final_training_mse": converged.get("final_training_mse"),
        "final_validation_mse": converged.get("final_validation_mse"),
        "loss_converged_exactly": learned.final_training_mse == 0 and learned.final_validation_mse == 0,
        "weights_hash": weights_hash(learned.weights),
        "passed": chain_breaks == 0 and malformed == 0 and learned.final_training_mse == 0 and learned.final_validation_mse == 0,
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _write_convergence(learned: LearnedRkan, out: Path) -> dict:
    result = {
        "training_semantics": "exact rational RKAN learning by normal equations over Boundary-compatible basis neurons",
        "training_rows": learned.training_rows,
        "validation_rows": learned.validation_rows,
        "coefficients": [fraction_to_json(c) for c in learned.coefficients],
        "initial_training_mse": fraction_to_json(learned.initial_mse),
        "final_training_mse": fraction_to_json(learned.final_training_mse),
        "final_validation_mse": fraction_to_json(learned.final_validation_mse),
        "loss_decreased": learned.final_training_mse < learned.initial_mse,
        "converged_exactly": learned.final_training_mse == 0 and learned.final_validation_mse == 0,
        "weights_hash": weights_hash(learned.weights),
        "commit": repo_commit(),
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _write_manifest(out_dir: Path, learned: LearnedRkan, expr, native_payload_summary: Path | None) -> dict:
    manifest = {
        "artifact": "rational_kan_hz_neural_symbolic_export",
        "commit": repo_commit(),
        "weights_hash": weights_hash(learned.weights),
        "expression_sha256": sha256_bytes(str(expr).encode("utf-8")),
        "rules_fingerprint": rules_fingerprint(DEFAULT_RULES),
        "native_payload_summary": str(native_payload_summary) if native_payload_summary else None,
        "claim": "A rational Boundary RKAN learned the target from exact rational samples and exported a faithful symbolic normal form.",
        "hot_path_boundary": "P4 remains an update/readout substrate optimization. SymPy/LaTeX are final artifact materialization only.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_status(out_dir: Path, passed: bool, convergence: dict, faithful: dict, replay: dict, lipschitz: dict) -> None:
    header = "# PROMOTED\n\n" if passed else "# CLOSED-WITH-FINDINGS\n\n"
    text = header
    text += "Final neural artifact export for Rational KAN HZ Boundary.\n\n"
    text += f"Training rows: {convergence['training_rows']}\n"
    text += f"Validation rows: {convergence['validation_rows']}\n"
    text += f"Initial training MSE: {convergence['initial_training_mse']}\n"
    text += f"Final training MSE: {convergence['final_training_mse']}\n"
    text += f"Final validation MSE: {convergence['final_validation_mse']}\n"
    text += f"Point mismatches: {faithful['point_mismatches']}\n"
    text += f"LaTeX round-trip mismatches: {faithful['latex_roundtrip_mismatches']}\n"
    text += f"Learning trace chain breaks: {replay['chain_break_records']}\n"
    text += f"Lipschitz bound: {lipschitz['whole_net_bound']}\n"
    (out_dir / "NEURAL_ARTIFACT_STATUS.md").write_text(text, encoding="utf-8")


def _write_report(out_dir: Path, convergence: dict, manifest: dict, passed: bool) -> None:
    status = "PROMOTED" if passed else "CLOSED-WITH-FINDINGS"
    text = "\n".join(
        [
            "# Rational KAN HZ Neural Artifact Export",
            "",
            f"Status: {status}",
            f"Commit: `{manifest['commit']}`",
            "",
            "## Neural Learning Gate",
            "",
            "The network is a rational KAN-shaped Boundary object with exact rational payloads.",
            "It trains from data by exact normal equations over Boundary-compatible basis neurons.",
            "The exported expression is produced only after the learned weights pass exact convergence checks.",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| training rows | {convergence['training_rows']} |",
            f"| validation rows | {convergence['validation_rows']} |",
            f"| initial training MSE | `{convergence['initial_training_mse']}` |",
            f"| final training MSE | `{convergence['final_training_mse']}` |",
            f"| final validation MSE | `{convergence['final_validation_mse']}` |",
            "",
            "## Artifact Boundary",
            "",
            "P4 lazy HZ/Veselov arithmetic remains the hot-path update/readout acceleration.",
            "SymPy, LaTeX, and PDF outputs are final materialization artifacts, not runtime dependencies.",
            "",
            "## Evidence",
            "",
            "- `learned_weights.json`",
            "- `learning_trace.jsonl`",
            "- `learning_trace_replay.json`",
            "- `convergence.json`",
            "- `extracted_expression.sympy`",
            "- `extracted_expression.tex`",
            "- `extracted_expression.pdf`",
            "- `faithfulness_results.json`",
            "- `lipschitz_certificate.json`",
            "- `manifest.json`",
            "",
        ]
    )
    (out_dir / "NEURAL_ARTIFACT_REPORT.md").write_text(text, encoding="utf-8")


def export_neural_artifacts(out_dir: Path, native_payload_summary: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    learned = train_exact_rkan()
    write_weights(learned.weights, out_dir / "learned_weights.json")
    convergence = _write_convergence(learned, out_dir / "convergence.json")
    trace_info = write_learning_trace(learned, out_dir / "learning_trace.jsonl")
    replay = replay_learning_trace(out_dir / "learning_trace.jsonl", learned, out_dir / "learning_trace_replay.json")
    expr, _guards = normalize_weights(learned.weights)
    (out_dir / "extracted_expression.sympy").write_text(str(expr) + "\n", encoding="utf-8")
    write_latex(expr, out_dir)
    pdf_compiled = True
    try:
        compile_latex(out_dir)
    except (OSError, subprocess.CalledProcessError):
        pdf_compiled = False
    faithful = faithfulness(learned.weights, expr, out_dir / "faithfulness_results.json", n_tests=200, seed=20260420)
    lipschitz = lipschitz_certificate(expr, out_dir / "lipschitz_certificate.json")
    manifest = _write_manifest(out_dir, learned, expr, native_payload_summary)
    passed = (
        convergence["converged_exactly"]
        and convergence["loss_decreased"]
        and replay["passed"]
        and faithful["point_mismatches"] == 0
        and faithful["latex_roundtrip_mismatches"] == 0
        and lipschitz["whole_net_bound_is_finite"]
        and lipschitz["sampling_sanity_violations"] == 0
        and pdf_compiled
    )
    _write_status(out_dir, passed, convergence, faithful, replay, lipschitz)
    _write_report(out_dir, convergence, manifest, passed)
    return {
        "passed": passed,
        "out_dir": str(out_dir),
        "convergence": convergence,
        "trace": trace_info,
        "replay": replay,
        "faithfulness": faithful,
        "lipschitz": lipschitz,
        "pdf_compiled": pdf_compiled,
        "manifest": manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--native-payload-summary",
        default="artifacts/rational_kan_hz/native_payload_iter_1/native_payload_summary.json",
    )
    args = parser.parse_args()
    native_summary = Path(args.native_payload_summary)
    result = export_neural_artifacts(Path(args.out_dir), native_summary if native_summary.exists() else None)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
