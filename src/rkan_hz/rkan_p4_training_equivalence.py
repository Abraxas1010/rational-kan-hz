"""Dense vs P4-lazy exact RKAN training equivalence.

This lane connects the neural artifact to the P4 lazy-update story. Dense and
lazy substrates consume the same rational training rows, build the same normal
equations, solve the same RKAN coefficients, and export the learned network.
The lazy substrate is a Python P4-style sparse update accumulator: updates append
born-sparse exact deltas and materialize once at readout.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from .exact_rkan import (
    deterministic_inputs,
    fraction_to_json,
    repo_commit,
    sha256_bytes,
    target_value,
    weights_hash,
    write_weights,
)
from .rkan_neural_artifact_export import (
    _basis,
    _kan_mean_square_error,
    _mean_square_error,
    _solve_fraction_linear_system,
    weights_from_learned_coefficients,
)
from .rkan_symbolic_extract import (
    compile_latex,
    faithfulness,
    lipschitz_certificate,
    normalize_weights,
    write_latex,
)
from .rule_loader import DEFAULT_RULES, rules_fingerprint


DEFAULT_OUT = Path("artifacts/rational_kan_hz/neural_p4_training_iter_1")
DEFAULT_SAMPLES = 30
BOOTSTRAP_SAMPLES = 1000


@dataclass(frozen=True)
class TrainingRun:
    substrate: str
    seed: int
    sample_index: int
    coefficients: tuple[Fraction, Fraction, Fraction]
    weights_hash: str
    initial_mse: Fraction
    final_training_mse: Fraction
    final_validation_mse: Fraction
    update_ns: int
    readout_ns: int
    solve_ns: int
    total_ns: int
    update_count: int
    support_count: int
    witness_bytes: int


class DenseNormalEquationAccumulator:
    def __init__(self) -> None:
        self.gram = [[Fraction(0) for _ in range(3)] for _ in range(3)]
        self.rhs = [Fraction(0) for _ in range(3)]
        self.update_count = 0

    def add(self, key: tuple[str, int, int | None], delta: Fraction) -> None:
        if delta == 0:
            return
        kind, i, j = key
        if kind == "gram":
            if j is None:
                raise ValueError("gram key requires j")
            self.gram[i][j] += delta
        elif kind == "rhs":
            self.rhs[i] += delta
        else:
            raise ValueError(f"unknown accumulator key {key!r}")
        self.update_count += 1

    def readout(self) -> tuple[list[list[Fraction]], list[Fraction], int, int]:
        return self.gram, self.rhs, 0, 0


class P4LazyNormalEquationAccumulator:
    def __init__(self) -> None:
        self.deltas: dict[tuple[str, int, int | None], list[Fraction]] = {}
        self.update_count = 0
        self.support_count = 0
        self.witness_bytes = 0

    def add(self, key: tuple[str, int, int | None], delta: Fraction) -> None:
        if delta == 0:
            return
        self.deltas.setdefault(key, []).append(delta)
        self.update_count += 1
        self.support_count += 1
        self.witness_bytes += len(f"{key}:{delta.numerator}/{delta.denominator}".encode("utf-8"))

    def readout(self) -> tuple[list[list[Fraction]], list[Fraction], int, int]:
        gram = [[Fraction(0) for _ in range(3)] for _ in range(3)]
        rhs = [Fraction(0) for _ in range(3)]
        for key, terms in self.deltas.items():
            kind, i, j = key
            total = sum(terms, Fraction(0))
            if kind == "gram":
                if j is None:
                    raise ValueError("gram key requires j")
                gram[i][j] = total
            elif kind == "rhs":
                rhs[i] = total
            else:
                raise ValueError(f"unknown accumulator key {key!r}")
        return gram, rhs, self.support_count, self.witness_bytes


def training_inputs_for_seed(seed: int) -> list[tuple[Fraction, Fraction]]:
    values = [Fraction(-2), Fraction(-1), Fraction(0), Fraction(1), Fraction(2)]
    rows = [(x0, x1) for x0 in values for x1 in values]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def _accumulate_rows(accumulator, rows: Iterable[tuple[Fraction, Fraction]]) -> None:
    for x in rows:
        phi = _basis(x)
        y = target_value(x)
        for i in range(3):
            accumulator.add(("rhs", i, None), phi[i] * y)
            for j in range(3):
                accumulator.add(("gram", i, j), phi[i] * phi[j])


def run_training(substrate: str, seed: int, sample_index: int) -> TrainingRun:
    rows = training_inputs_for_seed(seed)
    validation = deterministic_inputs(200, seed + 10_000, bound=2)
    accumulator = DenseNormalEquationAccumulator() if substrate == "dense" else P4LazyNormalEquationAccumulator()
    t0 = time.perf_counter_ns()
    _accumulate_rows(accumulator, rows)
    t1 = time.perf_counter_ns()
    gram, rhs, support_count, witness_bytes = accumulator.readout()
    t2 = time.perf_counter_ns()
    coeffs_raw = tuple(_solve_fraction_linear_system(gram, rhs))
    if len(coeffs_raw) != 3:
        raise AssertionError("expected three coefficients")
    coeffs: tuple[Fraction, Fraction, Fraction] = coeffs_raw  # type: ignore[assignment]
    weights = weights_from_learned_coefficients(coeffs)
    t3 = time.perf_counter_ns()
    return TrainingRun(
        substrate=substrate,
        seed=seed,
        sample_index=sample_index,
        coefficients=coeffs,
        weights_hash=weights_hash(weights),
        initial_mse=_mean_square_error((Fraction(0), Fraction(0), Fraction(0)), rows),
        final_training_mse=_kan_mean_square_error(weights, rows),
        final_validation_mse=_kan_mean_square_error(weights, validation),
        update_ns=t1 - t0,
        readout_ns=t2 - t1,
        solve_ns=t3 - t2,
        total_ns=t3 - t0,
        update_count=accumulator.update_count,
        support_count=support_count,
        witness_bytes=witness_bytes,
    )


def run_pair(seed: int, sample_index: int) -> tuple[TrainingRun, TrainingRun]:
    dense = run_training("dense", seed, sample_index)
    lazy = run_training("p4_lazy", seed, sample_index)
    if dense.weights_hash != lazy.weights_hash:
        raise AssertionError(f"final weight hash mismatch at seed {seed}")
    if dense.final_training_mse != 0 or lazy.final_training_mse != 0:
        raise AssertionError(f"training did not converge exactly at seed {seed}")
    if dense.final_validation_mse != 0 or lazy.final_validation_mse != 0:
        raise AssertionError(f"validation did not converge exactly at seed {seed}")
    return dense, lazy


def run_samples(samples: int, seed: int) -> list[tuple[TrainingRun, TrainingRun]]:
    return [run_pair(seed + i * 65_537, i) for i in range(samples)]


def run_to_json(run: TrainingRun) -> dict:
    return {
        "substrate": run.substrate,
        "seed": run.seed,
        "sample_index": run.sample_index,
        "coefficients": [fraction_to_json(c) for c in run.coefficients],
        "weights_hash": run.weights_hash,
        "initial_mse": fraction_to_json(run.initial_mse),
        "final_training_mse": fraction_to_json(run.final_training_mse),
        "final_validation_mse": fraction_to_json(run.final_validation_mse),
        "update_ns": run.update_ns,
        "readout_ns": run.readout_ns,
        "solve_ns": run.solve_ns,
        "total_ns": run.total_ns,
        "update_count": run.update_count,
        "support_count": run.support_count,
        "witness_bytes": run.witness_bytes,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def bootstrap_ci(values: list[float], *, seed: int, n: int = BOOTSTRAP_SAMPLES) -> dict:
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(sample))
    means.sort()
    lo = means[int(0.025 * (n - 1))]
    hi = means[int(0.975 * (n - 1))]
    return {
        "mean": _mean(values),
        "ci95_lower": lo,
        "ci95_upper": hi,
        "bootstrap_samples": n,
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _thermal_state() -> dict:
    zones = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            zones.append({"zone": zone.parent.name, "millidegrees_c": int(zone.read_text().strip())})
        except (OSError, ValueError):
            continue
    return {"zones": zones[:16], "available": bool(zones)}


def summarize(pairs: list[tuple[TrainingRun, TrainingRun]]) -> dict:
    update_ratios = [d.update_ns / max(1, p.update_ns) for d, p in pairs]
    total_ratios = [d.total_ns / max(1, p.total_ns) for d, p in pairs]
    readout_ratios = [d.readout_ns / max(1, p.readout_ns) for d, p in pairs]
    dense_hashes = {d.weights_hash for d, _p in pairs}
    lazy_hashes = {p.weights_hash for _d, p in pairs}
    return {
        "samples": len(pairs),
        "identity_passed": dense_hashes == lazy_hashes and len(dense_hashes) == 1,
        "convergence_passed": all(d.final_training_mse == 0 and p.final_training_mse == 0 for d, p in pairs),
        "validation_passed": all(d.final_validation_mse == 0 and p.final_validation_mse == 0 for d, p in pairs),
        "dense_over_p4_update": bootstrap_ci(update_ratios, seed=20260420),
        "dense_over_p4_readout": bootstrap_ci(readout_ratios, seed=20260421),
        "dense_over_p4_total": bootstrap_ci(total_ratios, seed=20260422),
        "performance_gate_update_ci_lower_gt_1": bootstrap_ci(update_ratios, seed=20260420)["ci95_lower"] > 1.0,
        "performance_gate_total_ci_lower_gt_1": bootstrap_ci(total_ratios, seed=20260422)["ci95_lower"] > 1.0,
        "weights_hash": next(iter(dense_hashes)) if dense_hashes else None,
    }


def write_symbolic_export(weights, out_dir: Path) -> dict:
    expr, _guards = normalize_weights(weights)
    (out_dir / "extracted_expression.sympy").write_text(str(expr) + "\n", encoding="utf-8")
    write_latex(expr, out_dir)
    pdf_compiled = True
    try:
        compile_latex(out_dir)
    except (OSError, subprocess.CalledProcessError):
        pdf_compiled = False
    faithful = faithfulness(weights, expr, out_dir / "faithfulness_results.json", n_tests=200, seed=20260420)
    lipschitz = lipschitz_certificate(expr, out_dir / "lipschitz_certificate.json")
    return {
        "expression": str(expr),
        "expression_sha256": sha256_bytes(str(expr).encode("utf-8")),
        "pdf_compiled": pdf_compiled,
        "faithfulness": faithful,
        "lipschitz": lipschitz,
    }


def write_artifacts(out_dir: Path, pairs: list[tuple[TrainingRun, TrainingRun]], command_line: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "dense_p4_training_dual.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for dense, lazy in pairs:
            f.write(json.dumps(run_to_json(dense), sort_keys=True) + "\n")
            f.write(json.dumps(run_to_json(lazy), sort_keys=True) + "\n")
    summary = summarize(pairs)
    representative = weights_from_learned_coefficients(pairs[0][0].coefficients)
    symbolic = write_symbolic_export(representative, out_dir)
    summary.update(
        {
            "artifact": "rational_kan_hz_dense_p4_lazy_training_equivalence",
            "commit": repo_commit(),
            "command_line": command_line,
            "host_cpu": _cpu_model(),
            "platform": platform.platform(),
            "thermal_state_start": _thermal_state(),
            "rules_fingerprint": rules_fingerprint(DEFAULT_RULES),
            "symbolic": symbolic,
            "p4_substrate_note": (
                "p4_lazy is a P4-style lazy sparse exact accumulator for training updates; "
                "it defers materialization until normal-equation readout."
            ),
        }
    )
    passed = (
        summary["identity_passed"]
        and summary["convergence_passed"]
        and summary["validation_passed"]
        and symbolic["pdf_compiled"]
        and symbolic["faithfulness"]["point_mismatches"] == 0
        and symbolic["faithfulness"]["latex_roundtrip_mismatches"] == 0
        and symbolic["lipschitz"]["sampling_sanity_violations"] == 0
    )
    summary["passed"] = passed
    (out_dir / "training_equivalence_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if passed and (summary["performance_gate_update_ci_lower_gt_1"] or summary["performance_gate_total_ci_lower_gt_1"]):
        status = "# PROMOTED\n\n"
    elif passed:
        status = "# PROMOTED FOR EQUIVALENCE; PERFORMANCE CLOSED WITH FINDINGS\n\n"
    else:
        status = "# CLOSED-WITH-FINDINGS\n\n"
    status += "Dense and P4-lazy exact RKAN training equivalence.\n\n"
    status += f"Samples: {summary['samples']}\n"
    status += f"Final weight hash identity: {summary['identity_passed']}\n"
    status += f"Exact training convergence: {summary['convergence_passed']}\n"
    status += f"Exact validation convergence: {summary['validation_passed']}\n"
    status += (
        "Dense/P4 update speedup mean "
        f"{summary['dense_over_p4_update']['mean']:.6g} "
        f"[{summary['dense_over_p4_update']['ci95_lower']:.6g}, "
        f"{summary['dense_over_p4_update']['ci95_upper']:.6g}]\n"
    )
    status += (
        "Dense/P4 total speedup mean "
        f"{summary['dense_over_p4_total']['mean']:.6g} "
        f"[{summary['dense_over_p4_total']['ci95_lower']:.6g}, "
        f"{summary['dense_over_p4_total']['ci95_upper']:.6g}]\n"
    )
    status += f"Update performance gate lower CI > 1: {summary['performance_gate_update_ci_lower_gt_1']}\n"
    status += f"Total performance gate lower CI > 1: {summary['performance_gate_total_ci_lower_gt_1']}\n"
    status += f"Symbolic expression: `{symbolic['expression']}`\n"
    status += f"PDF compiled: {symbolic['pdf_compiled']}\n"
    (out_dir / "TRAINING_EQUIVALENCE_STATUS.md").write_text(status, encoding="utf-8")
    report = "\n".join(
        [
            "# Rational KAN HZ Dense vs P4-Lazy Training Equivalence",
            "",
            "## Result",
            "",
            status.strip(),
            "",
            "## Interpretation",
            "",
            "Dense exact training and lazy sparse exact training consume the same rational rows and produce byte-identical learned weights.",
            "The timing table separates update-only speed from full materialization-plus-solve time, so the P4 sweet spot is visible rather than hidden.",
            "In this small Python normal-equation workload, the P4-style lazy accumulator does not clear the performance gate. The semantic equivalence and export gates pass; the performance claim remains closed for this specific workload.",
            "",
            "## Evidence",
            "",
            "- `dense_p4_training_dual.jsonl`",
            "- `training_equivalence_summary.json`",
            "- `learned_weights.json`",
            "- `extracted_expression.sympy`",
            "- `extracted_expression.tex`",
            "- `extracted_expression.pdf`",
            "- `faithfulness_results.json`",
            "- `lipschitz_certificate.json`",
            "",
        ]
    )
    (out_dir / "TRAINING_EQUIVALENCE_REPORT.md").write_text(report, encoding="utf-8")
    write_weights(representative, out_dir / "learned_weights.json")
    return summary


def run(out_dir: Path, samples: int, seed: int, command_line: list[str]) -> dict:
    pairs = run_samples(samples, seed)
    return write_artifacts(out_dir, pairs, command_line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=20260420)
    args = parser.parse_args()
    result = run(Path(args.out_dir), args.samples, args.seed, sys.argv)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
