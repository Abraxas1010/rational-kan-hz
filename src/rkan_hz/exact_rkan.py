"""Exact rational RKAN primitives used by Phases 1 through 4."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ARTIFACT_ROOT = Path("artifacts/rational_kan_hz/iter_1")


def repo_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fraction_to_json(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def fraction_from_json(raw: str | int | float | Fraction) -> Fraction:
    if isinstance(raw, Fraction):
        return raw
    if isinstance(raw, int):
        return Fraction(raw, 1)
    if isinstance(raw, float):
        return Fraction.from_float(raw).limit_denominator(1 << 24)
    if "/" in raw:
        n, d = raw.split("/", 1)
        return Fraction(int(n), int(d))
    return Fraction(int(raw), 1)


def hash_fraction(value: Fraction) -> str:
    return sha256_bytes(fraction_to_json(value).encode("utf-8"))


def deterministic_inputs(samples: int, seed: int, *, bound: int = 2) -> list[tuple[Fraction, Fraction]]:
    rng = random.Random(seed)
    out: list[tuple[Fraction, Fraction]] = []
    for _ in range(samples):
        row = []
        for _dim in range(2):
            raw = rng.uniform(-bound, bound)
            row.append(Fraction.from_float(raw).limit_denominator(1 << 24))
        out.append((row[0], row[1]))
    return out


@dataclass(frozen=True)
class ActivationCoeffs:
    a: tuple[Fraction, Fraction, Fraction, Fraction]
    b: tuple[Fraction, Fraction, Fraction]

    def to_json(self) -> dict[str, list[str]]:
        return {
            "a": [fraction_to_json(v) for v in self.a],
            "b": [fraction_to_json(v) for v in self.b],
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "ActivationCoeffs":
        return cls(
            a=tuple(fraction_from_json(v) for v in raw["a"]),  # type: ignore[arg-type]
            b=tuple(fraction_from_json(v) for v in raw["b"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class KANWeights:
    inner: tuple[tuple[ActivationCoeffs, ...], ...]
    outer: tuple[ActivationCoeffs, ...]
    metadata: dict[str, Any]

    @property
    def input_dim(self) -> int:
        return len(self.inner[0])

    @property
    def outer_count(self) -> int:
        return len(self.outer)

    def to_json(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "inner": [[act.to_json() for act in row] for row in self.inner],
            "outer": [act.to_json() for act in self.outer],
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "KANWeights":
        return cls(
            inner=tuple(tuple(ActivationCoeffs.from_json(act) for act in row) for row in raw["inner"]),
            outer=tuple(ActivationCoeffs.from_json(act) for act in raw["outer"]),
            metadata=dict(raw.get("metadata") or {}),
        )


IDENTITY = ActivationCoeffs(
    a=(Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
    b=(Fraction(0), Fraction(0), Fraction(0)),
)
ZERO = ActivationCoeffs(
    a=(Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
    b=(Fraction(0), Fraction(0), Fraction(0)),
)


def monomial(scale: Fraction, power: int) -> ActivationCoeffs:
    a = [Fraction(0), Fraction(0), Fraction(0), Fraction(0)]
    a[power] = scale
    return ActivationCoeffs(a=tuple(a), b=(Fraction(0), Fraction(0), Fraction(0)))  # type: ignore[arg-type]


def constructive_trained_weights() -> KANWeights:
    """Exact five-outer-unit RKAN approximating x0^2 + sin(x1)."""

    inner = (
        (IDENTITY, ZERO),
        (IDENTITY, ZERO),
        (ZERO, IDENTITY),
        (ZERO, IDENTITY),
        (ZERO, IDENTITY),
    )
    outer = (
        monomial(Fraction(1, 2), 2),
        monomial(Fraction(1, 2), 2),
        monomial(Fraction(1, 2), 1),
        monomial(Fraction(1, 2), 1),
        monomial(Fraction(-1, 6), 3),
    )
    return KANWeights(
        inner=inner,
        outer=outer,
        metadata={
            "source": "constructive_exact_boundary_training",
            "target": "x0^2 + sin(x1)",
            "sin_approximation": "x1 - x1^3/6",
            "input_dim": 2,
            "outer_count": 5,
            "inner_edges_total": 10,
            "activation_degree": 3,
        },
    )


def zero_weights() -> KANWeights:
    return KANWeights(
        inner=tuple(tuple(ZERO for _ in range(2)) for _ in range(5)),
        outer=tuple(ZERO for _ in range(5)),
        metadata={
            "source": "zero_initial_boundary_net",
            "input_dim": 2,
            "outer_count": 5,
            "inner_edges_total": 10,
            "activation_degree": 3,
        },
    )


def rational_activation(x: Fraction, coeffs: ActivationCoeffs) -> Fraction:
    a0, a1, a2, a3 = coeffs.a
    b1, b2, b3 = coeffs.b
    numerator = a0 + a1 * x + a2 * x**2 + a3 * x**3
    denominator = Fraction(1) + b1 * x + b2 * x**2 + b3 * x**3
    if denominator == 0:
        raise ZeroDivisionError("rational activation denominator vanished")
    return numerator / denominator


def kan_eval(weights: KANWeights, x: tuple[Fraction, Fraction]) -> Fraction:
    total = Fraction(0)
    for k, outer in enumerate(weights.outer):
        inner_sum = Fraction(0)
        for i in range(weights.input_dim):
            inner_sum += rational_activation(x[i], weights.inner[k][i])
        total += rational_activation(inner_sum, outer)
    return total


def target_value(x: tuple[Fraction, Fraction]) -> Fraction:
    # Phase 2 exact target uses the same polynomial approximation that the
    # constructive Boundary net can represent exactly.
    return x[0] ** 2 + x[1] - x[1] ** 3 / 6


def true_toy_target_float(x: tuple[Fraction, Fraction]) -> float:
    return float(x[0]) ** 2 + math.sin(float(x[1]))


def mse_against_true_target(weights: KANWeights, samples: int = 5000, seed: int = 42) -> float:
    total = 0.0
    for x in deterministic_inputs(samples, seed):
        prediction = float(kan_eval(weights, x))
        diff = prediction - true_toy_target_float(x)
        total += diff * diff
    return total / samples


def prediction_variance(weights: KANWeights, samples: int = 1000, seed: int = 1042) -> float:
    values = [float(kan_eval(weights, x)) for x in deterministic_inputs(samples, seed)]
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def write_weights(weights: KANWeights, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_weights(path: Path) -> KANWeights:
    return KANWeights.from_json(json.loads(path.read_text(encoding="utf-8")))


def weights_hash(weights: KANWeights) -> str:
    return sha256_bytes(json.dumps(weights.to_json(), sort_keys=True).encode("utf-8"))


def quantize_float(value: float, max_denominator: int = 1 << 24) -> Fraction:
    return Fraction.from_float(float(value)).limit_denominator(max_denominator)


def load_phase0_weights(path: Path) -> KANWeights:
    """Load Phase 0 PyTorch weights as exact Fractions for Phase 1 inference."""

    import torch

    state = torch.load(path, map_location="cpu")

    def activation(prefix: str) -> ActivationCoeffs:
        a_raw = state[f"{prefix}.a"].detach().cpu().numpy().tolist()
        b_raw = state[f"{prefix}.b"].detach().cpu().numpy().tolist()
        return ActivationCoeffs(
            a=tuple(quantize_float(v) for v in a_raw),  # type: ignore[arg-type]
            b=tuple(quantize_float(v) for v in b_raw),  # type: ignore[arg-type]
        )

    inner: list[list[ActivationCoeffs]] = []
    for k in range(5):
        row = []
        for i in range(2):
            row.append(activation(f"inner.{k}.{i}"))
        inner.append(row)
    outer = [activation(f"outer.{k}") for k in range(5)]
    return KANWeights(
        inner=tuple(tuple(row) for row in inner),
        outer=tuple(outer),
        metadata={
            "source": str(path),
            "quantization": "Fraction.from_float(...).limit_denominator(2^24)",
            "input_dim": 2,
            "outer_count": 5,
            "inner_edges_total": 10,
            "activation_degree": 3,
        },
    )


def count_coefficients(weights: KANWeights) -> int:
    return weights.outer_count * (weights.input_dim + 1) * 7


def all_coefficients(weights: KANWeights) -> Iterable[Fraction]:
    for row in weights.inner:
        for act in row:
            yield from act.a
            yield from act.b
    for act in weights.outer:
        yield from act.a
        yield from act.b


def zeckendorf_support(n: int) -> int:
    n = abs(n)
    if n == 0:
        return 0
    fibs = [1, 2]
    while fibs[-1] <= n:
        fibs.append(fibs[-1] + fibs[-2])
    support = 0
    remaining = n
    for fib in reversed(fibs):
        if fib <= remaining:
            remaining -= fib
            support += 1
        if remaining == 0:
            break
    return support


def support_count(value: Fraction) -> int:
    return zeckendorf_support(value.numerator) + zeckendorf_support(value.denominator)
