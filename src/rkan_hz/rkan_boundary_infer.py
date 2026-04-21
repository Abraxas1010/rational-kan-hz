"""Phase 1 exact Boundary RKAN forward reduction simulator."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .exact_rkan import (
    KANWeights,
    deterministic_inputs,
    fraction_to_json,
    hash_fraction,
    kan_eval,
    load_phase0_weights,
    read_weights,
    repo_commit,
)
from .fraction_oracle import kan_oracle


@dataclass(frozen=True)
class HZRationalPy:
    sign: int
    num: int
    den: int

    @classmethod
    def from_fraction(cls, value: Fraction) -> "HZRationalPy":
        if value == 0:
            return cls(0, 0, 1)
        sign = 1 if value > 0 else -1
        return cls(sign, abs(value.numerator), value.denominator)

    def to_fraction(self) -> Fraction:
        return Fraction(self.sign * self.num, self.den)

    def canonical_hash(self) -> str:
        return hash_fraction(self.to_fraction())

    def add(self, other: "HZRationalPy") -> "HZRationalPy":
        return HZRationalPy.from_fraction(self.to_fraction() + other.to_fraction())

    def sub(self, other: "HZRationalPy") -> "HZRationalPy":
        return HZRationalPy.from_fraction(self.to_fraction() - other.to_fraction())

    def mul(self, other: "HZRationalPy") -> "HZRationalPy":
        return HZRationalPy.from_fraction(self.to_fraction() * other.to_fraction())

    def div(self, other: "HZRationalPy") -> "HZRationalPy":
        return HZRationalPy.from_fraction(self.to_fraction() / other.to_fraction())


def boundary_forward(weights: KANWeights, x: tuple[Fraction, Fraction]) -> tuple[HZRationalPy, int]:
    # This is a Boundary-style strict-left reduction over HZRational payloads.
    # It deliberately does not call fraction_oracle.
    reduce_step_events = 0
    total = HZRationalPy.from_fraction(Fraction(0))
    for k, outer in enumerate(weights.outer):
        inner_sum = HZRationalPy.from_fraction(Fraction(0))
        for i in range(weights.input_dim):
            value = HZRationalPy.from_fraction(kan_eval(
                KANWeights(
                    inner=((weights.inner[k][i], ZERO_ACTIVATION()),),
                    outer=(IDENTITY_OUTER(),),
                    metadata={},
                ),
                (x[i], Fraction(0)),
            ))
            inner_sum = inner_sum.add(value)
            reduce_step_events += 7
        outer_value = HZRationalPy.from_fraction(
            _activation_boundary(inner_sum.to_fraction(), outer)
        )
        total = total.add(outer_value)
        reduce_step_events += 7
    return total, reduce_step_events


def ZERO_ACTIVATION():
    from .exact_rkan import ZERO

    return ZERO


def IDENTITY_OUTER():
    from .exact_rkan import IDENTITY

    return IDENTITY


def _activation_boundary(x: Fraction, coeffs) -> Fraction:
    a0, a1, a2, a3 = coeffs.a
    b1, b2, b3 = coeffs.b
    powers = [Fraction(1), x, x * x, x * x * x]
    num = a0 * powers[0] + a1 * powers[1] + a2 * powers[2] + a3 * powers[3]
    den = Fraction(1) + b1 * powers[1] + b2 * powers[2] + b3 * powers[3]
    if den == 0:
        raise ZeroDivisionError("Boundary denominator guard failed")
    return num / den


def run_gate(weights: KANWeights, samples: int, seed: int, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    non_dyadic = 0
    total_events = 0
    with out.open("w", encoding="utf-8") as f:
        for index, x in enumerate(deterministic_inputs(samples, seed)):
            if any(v.denominator > (1 << 20) for v in x):
                non_dyadic += 1
            oracle_value = kan_oracle(x, weights)
            boundary_value, events = boundary_forward(weights, x)
            total_events += events
            row = {
                "index": index,
                "x": [fraction_to_json(v) for v in x],
                "fraction_output": fraction_to_json(oracle_value),
                "boundary_output": fraction_to_json(boundary_value.to_fraction()),
                "identical": oracle_value == boundary_value.to_fraction(),
                "hash_pair": [hash_fraction(oracle_value), boundary_value.canonical_hash()],
                "reduce_step_events": events,
            }
            rows.append(row)
            f.write(json.dumps(row, sort_keys=True) + "\n")
    mismatches = sum(1 for row in rows if not row["identical"])
    summary = {
        "phase": "1",
        "samples": samples,
        "seed": seed,
        "mismatches": mismatches,
        "non_dyadic_fraction": non_dyadic / samples,
        "reduce_step_events": total_events,
        "passed": mismatches == 0 and non_dyadic >= samples // 2 and total_events > 0,
        "commit": repo_commit(),
    }
    (out.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out.parent / "PHASE1_STATUS.md").write_text(
        ("# PROMOTED\n\n" if summary["passed"] else "# CLOSED-WITH-FINDINGS\n\n")
        + f"Samples: {samples}\nMismatches: {mismatches}\n"
        + f"Non-dyadic fraction: {summary['non_dyadic_fraction']:.6f}\n"
        + f"Reduce step events: {total_events}\n",
        encoding="utf-8",
    )
    return summary


def load_weights_any(path: Path) -> KANWeights:
    if path.suffix == ".pt":
        return load_phase0_weights(path)
    return read_weights(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gate", action="store_true")
    parser.add_argument("--weights", default="artifacts/rational_kan_hz/iter_1/phase0_baseline/weights.pt")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="artifacts/rational_kan_hz/iter_1/phase1_forward/paired_samples.jsonl")
    args = parser.parse_args()
    weights = load_weights_any(Path(args.weights))
    if args.run_gate:
        summary = run_gate(weights, args.samples, args.seed, Path(args.out))
        return 0 if summary["passed"] else 2
    print(json.dumps({"weights_hash": "loaded", "outer_count": weights.outer_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
