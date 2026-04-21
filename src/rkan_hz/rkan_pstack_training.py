"""Exact degree-8 Rational KAN with an in-the-loop Hybrid Zeckendorf accumulator.

The exact semantics live in Python's ``fractions.Fraction``. The P-stack path
uses the real Hybrid Zeckendorf substrate in the regime it actually helps:
append-only exact-rational delta accumulation with deferred readout. Fractions
are kept denominator-factored rather than being collapsed by a group-wide LCM
before HZ sees them. This preserves exactness while avoiding the adversarial
operand-magnitude blow-up that invalidated the earlier benchmark lane.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .exact_rkan import repo_commit, sha256_bytes
from .repo_layout import find_repo_root

try:
    import sys

    sys.set_int_max_str_digits(0)
except Exception:
    pass


REPO_ROOT = find_repo_root(Path(__file__))
DEFAULT_OUT = REPO_ROOT / "artifacts" / "rational_kan_hz" / "paper_run" / "pstack_in_loop"
ACCUM_WORKDIR = REPO_ROOT / "bench" / "hybrid_zeckendorf"
ACCUM_BIN = ACCUM_WORKDIR / "target" / "release" / "pstack_exact_accumulate"
DEFAULT_STEPS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEGREE = 8
DEFAULT_HIDDEN = 10
DEFAULT_INPUT_DIM = 4
DEFAULT_DEN_BOUND = 10**6
DEFAULT_LR = Fraction(1, 64)
DEFAULT_SEED = 20260420
BOOTSTRAP_SAMPLES = 1000
DEFAULT_MICROBATCH_GRID = (1, 4, 8, 16, 32, 64, 128)
CONJECTURE_STATEMENT = (
    "An exact-rational KAN with degree at least 8 and hidden width at least 10 can run "
    "the forward/backward/training loop through the Hybrid Zeckendorf P-stack substrate "
    "with bit-identical Fraction semantics and a measurable inference speed advantage at "
    "network scale."
)


def _fraction_string(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"num": value.numerator, "den": value.denominator}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Fraction):
        return _fraction_string(value)
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    return value


def _tensor_hash(value: Any) -> str:
    blob = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(blob)


def _sb_axis(max_denominator: int = 8) -> list[Fraction]:
    del max_denominator
    return [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]


def sb_grid_batch(batch_size: int, seed: int, *, input_dim: int = DEFAULT_INPUT_DIM, max_denominator: int = 8) -> list[tuple[Fraction, ...]]:
    rng = random.Random(seed)
    axis = _sb_axis(max_denominator)
    batch: list[tuple[Fraction, ...]] = []
    for _ in range(batch_size):
        batch.append(tuple(axis[rng.randrange(len(axis))] for _ in range(input_dim)))
    return batch


def target_degree8(x: tuple[Fraction, ...]) -> Fraction:
    x0, x1, x2, x3 = x
    return (
        x0**2
        + x1
        - x1**3 / 6
        + x1**5 / 120
        + x2**7 / 5040
        + x3**8 / 40320
    )


def _powers(x: Fraction, degree: int) -> tuple[Fraction, ...]:
    out = [Fraction(1)]
    for _ in range(degree):
        out.append(out[-1] * x)
    return tuple(out)


def _project(value: Fraction, denominator_bound: int) -> Fraction:
    return value.limit_denominator(denominator_bound) if denominator_bound > 0 else value


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return _mean([float(row[field]) for row in rows])


def bootstrap_ci(values: list[float], *, seed: int, n: int = BOOTSTRAP_SAMPLES) -> dict[str, float | int]:
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
    except Exception:
        pass
    return "unknown"


def _thermal_state() -> dict[str, Any]:
    zones = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            zones.append({"zone": zone.parent.name, "millidegrees_c": int(zone.read_text().strip())})
        except (OSError, ValueError):
            continue
    return {"zones": zones[:16], "available": bool(zones)}


def _mean_square_loss(predictions: list[Fraction], targets: list[Fraction]) -> Fraction:
    total = Fraction(0)
    for prediction, target in zip(predictions, targets, strict=True):
        diff = prediction - target
        total += diff * diff
    return total / len(predictions)


@dataclass(frozen=True)
class RationalKANDeg8:
    edge_coeffs: tuple[tuple[tuple[Fraction, ...], ...], ...]
    output_weights: tuple[Fraction, ...]
    bias: Fraction
    metadata: dict[str, Any]

    @property
    def hidden(self) -> int:
        return len(self.edge_coeffs)

    @property
    def input_dim(self) -> int:
        return len(self.edge_coeffs[0])

    @property
    def degree(self) -> int:
        return len(self.edge_coeffs[0][0]) - 1

    @property
    def param_count(self) -> int:
        return self.hidden * self.input_dim * (self.degree + 1) + self.hidden + 1

    def to_json(self) -> dict[str, Any]:
        return {
            "edge_coeffs": [[[ _fraction_string(v) for v in coeffs] for coeffs in row] for row in self.edge_coeffs],
            "output_weights": [_fraction_string(v) for v in self.output_weights],
            "bias": _fraction_string(self.bias),
            "metadata": self.metadata,
        }

    @classmethod
    def seeded(cls, *, seed: int, hidden: int = DEFAULT_HIDDEN, degree: int = DEFAULT_DEGREE, input_dim: int = DEFAULT_INPUT_DIM) -> "RationalKANDeg8":
        rng = random.Random(seed)
        edge_rows: list[list[tuple[Fraction, ...]]] = []
        target_terms = [
            (0, 2, Fraction(1)),
            (1, 1, Fraction(1)),
            (1, 3, Fraction(-1, 6)),
            (1, 5, Fraction(1, 120)),
            (2, 7, Fraction(1, 5040)),
            (3, 8, Fraction(1, 40320)),
        ]
        for hidden_index in range(hidden):
            row: list[tuple[Fraction, ...]] = []
            for input_index in range(input_dim):
                coeffs = [Fraction(0) for _ in range(degree + 1)]
                if hidden_index < len(target_terms):
                    target_input, target_power, target_coeff = target_terms[hidden_index]
                    if input_index == target_input:
                        noise = Fraction(rng.randint(-2, 2), 64)
                        coeffs[target_power] = target_coeff + noise
                        coeffs[0] = Fraction(rng.randint(-1, 1), 128)
                row.append(tuple(coeffs))
            edge_rows.append(row)
        output_weights = []
        for hidden_index in range(hidden):
            if hidden_index < len(target_terms):
                output_weights.append(Fraction(1) + Fraction(rng.randint(-2, 2), 64))
            else:
                output_weights.append(Fraction(0))
        return cls(
            edge_coeffs=tuple(tuple(coeffs for coeffs in row) for row in edge_rows),
            output_weights=tuple(output_weights),
            bias=Fraction(rng.randint(-1, 1), 128),
            metadata={
                "source": "seeded_sparse_exact_rkan",
                "seed": seed,
                "hidden": hidden,
                "degree": degree,
                "input_dim": input_dim,
                "target": "t6_degree8_extended_4d",
            },
        )

    def network_spec(self) -> dict[str, Any]:
        active_edge_terms = sum(
            1
            for row in self.edge_coeffs
            for coeffs in row
            for coeff in coeffs
            if coeff != 0
        )
        active_hidden = sum(
            1
            for h in range(self.hidden)
            if self.output_weights[h] != 0
            or any(coeff != 0 for row in (self.edge_coeffs[h],) for coeffs in row for coeff in coeffs)
        )
        return {
            "degree": self.degree,
            "hidden": self.hidden,
            "input_dim": self.input_dim,
            "param_count": self.param_count,
            "active_edge_terms": active_edge_terms,
            "active_hidden_units": active_hidden,
            "meets_scale_requirement": self.degree >= 8 and self.hidden >= 10,
            "metadata": self.metadata,
            "commit": repo_commit(),
        }

    def forward_fraction(self, batch: list[tuple[Fraction, ...]]) -> dict[str, Any]:
        edge_outputs: list[list[list[Fraction]]] = []
        pre_activations: list[list[Fraction]] = []
        activations: list[list[Fraction]] = []
        predictions: list[Fraction] = []
        for x in batch:
            sample_edges: list[list[Fraction]] = []
            sample_preacts: list[Fraction] = []
            x_powers = [_powers(x_i, self.degree) for x_i in x]
            for hidden_index in range(self.hidden):
                hidden_edges: list[Fraction] = []
                preactivation = Fraction(0)
                for input_index in range(self.input_dim):
                    coeffs = self.edge_coeffs[hidden_index][input_index]
                    edge_value = sum(
                        (
                            coeffs[power] * x_powers[input_index][power]
                            for power in range(self.degree + 1)
                            if coeffs[power] != 0
                        ),
                        Fraction(0),
                    )
                    hidden_edges.append(edge_value)
                    preactivation += edge_value
                sample_edges.append(hidden_edges)
                sample_preacts.append(preactivation)
            prediction = self.bias + sum(
                self.output_weights[hidden_index] * sample_preacts[hidden_index]
                for hidden_index in range(self.hidden)
                if self.output_weights[hidden_index] != 0
            )
            edge_outputs.append(sample_edges)
            pre_activations.append(sample_preacts)
            activations.append(list(sample_preacts))
            predictions.append(prediction)
        return {
            "predictions": predictions,
            "pre_activations": pre_activations,
            "activations": activations,
            "edge_outputs": edge_outputs,
        }

    def backward_fraction(
        self,
        batch: list[tuple[Fraction, ...]],
        targets: list[Fraction],
        forward: dict[str, Any],
    ) -> dict[str, Any]:
        predictions = forward["predictions"]
        pre_activations = forward["pre_activations"]
        batch_scale = Fraction(1, len(batch))
        diff_scales = [2 * (prediction - target) * batch_scale for prediction, target in zip(predictions, targets, strict=True)]

        output_weight_grads = []
        for hidden_index in range(self.hidden):
            total = Fraction(0)
            for sample_index in range(len(batch)):
                total += diff_scales[sample_index] * pre_activations[sample_index][hidden_index]
            output_weight_grads.append(total)

        bias_grad = sum(diff_scales, Fraction(0))

        edge_grads: list[list[list[Fraction]]] = []
        for hidden_index in range(self.hidden):
            row: list[list[Fraction]] = []
            for input_index in range(self.input_dim):
                coeff_grads: list[Fraction] = []
                for power in range(self.degree + 1):
                    total = Fraction(0)
                    for sample_index, x in enumerate(batch):
                        total += diff_scales[sample_index] * self.output_weights[hidden_index] * (x[input_index] ** power)
                    coeff_grads.append(total)
                row.append(coeff_grads)
            edge_grads.append(row)

        return {
            "loss": _mean_square_loss(predictions, targets),
            "output_weight_grads": output_weight_grads,
            "bias_grad": bias_grad,
            "edge_grads": edge_grads,
        }

    def apply_update(
        self,
        grads: dict[str, Any],
        *,
        lr: Fraction,
        denominator_bound: int,
    ) -> "RationalKANDeg8":
        new_edges: list[list[tuple[Fraction, ...]]] = []
        for hidden_index in range(self.hidden):
            row: list[tuple[Fraction, ...]] = []
            for input_index in range(self.input_dim):
                updated = tuple(
                    _project(
                        self.edge_coeffs[hidden_index][input_index][power]
                        - lr * grads["edge_grads"][hidden_index][input_index][power],
                        denominator_bound,
                    )
                    for power in range(self.degree + 1)
                )
                row.append(updated)
            new_edges.append(row)
        new_output_weights = tuple(
            _project(self.output_weights[hidden_index] - lr * grads["output_weight_grads"][hidden_index], denominator_bound)
            for hidden_index in range(self.hidden)
        )
        new_bias = _project(self.bias - lr * grads["bias_grad"], denominator_bound)
        return RationalKANDeg8(
            edge_coeffs=tuple(tuple(coeffs for coeffs in row) for row in new_edges),
            output_weights=new_output_weights,
            bias=new_bias,
            metadata=dict(self.metadata),
        )


class PStackAccumulatorWorker:
    _instance: "PStackAccumulatorWorker | None" = None

    def __init__(self) -> None:
        self.bin_path = self._ensure_binary()
        self.proc = subprocess.Popen(
            [str(self.bin_path), "--backend", "hz_lazy"],
            cwd=str(ACCUM_WORKDIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    @classmethod
    def instance(cls) -> "PStackAccumulatorWorker":
        if cls._instance is None:
            cls._instance = cls()
            atexit.register(cls._instance.close)
        return cls._instance

    def close(self) -> None:
        if getattr(self, "proc", None) and self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _ensure_binary(self) -> Path:
        if ACCUM_BIN.exists():
            return ACCUM_BIN
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "pstack_exact_accumulate"],
            cwd=str(ACCUM_WORKDIR),
            check=True,
        )
        return ACCUM_BIN

    def sum_integer_groups(self, groups: list[list[int | str]]) -> tuple[list[int], dict[str, int]]:
        request_groups = [{"values": [str(value) for value in group if int(value) != 0]} for group in groups]
        request = json.dumps({"groups": request_groups}, separators=(",", ":"))
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(request + "\n")
        self.proc.stdin.flush()
        response_line = self.proc.stdout.readline()
        if not response_line:
            stderr = ""
            if self.proc.stderr is not None:
                stderr = self.proc.stderr.read()
            raise RuntimeError(f"pstack worker exited unexpectedly: {stderr}")
        response = json.loads(response_line)
        results: list[int] = []
        support_card = 0
        active_levels = 0
        witness_bytes = 0
        nonzero_terms = 0
        for row in response["results"]:
            results.append(int(row["sum"]))
            support_card += int(row["support_card"])
            active_levels += int(row["active_levels"])
            witness_bytes += int(row["witness_bytes"])
            nonzero_terms += int(row["nonzero_terms"])
        return results, {
            "support_card": support_card,
            "active_levels": active_levels,
            "witness_bytes": witness_bytes,
            "nonzero_terms": nonzero_terms,
            "group_count": len(groups),
        }

    def sum_groups(self, groups: list[list[Fraction]]) -> tuple[list[Fraction], dict[str, int]]:
        results = [Fraction(0) for _ in groups]

        by_denominator: dict[int, list[list[int]]] = {}
        for group_index, group in enumerate(groups):
            for value in group:
                if value == 0:
                    continue
                denominator_groups = by_denominator.setdefault(value.denominator, [[] for _ in groups])
                denominator_groups[group_index].append(value.numerator)

        ordered_denominators = sorted(by_denominator)
        flattened_groups: list[list[int]] = []
        for denominator in ordered_denominators:
            flattened_groups.extend(by_denominator[denominator])

        integer_sums, stats = self.sum_integer_groups(flattened_groups)
        cursor = 0
        for denominator in ordered_denominators:
            for group_index in range(len(groups)):
                integer_sum = integer_sums[cursor]
                if integer_sum != 0:
                    results[group_index] += Fraction(integer_sum, denominator)
                cursor += 1

        return results, {
            "support_card": stats["support_card"] if ordered_denominators else 0,
            "active_levels": stats["active_levels"] if ordered_denominators else 0,
            "witness_bytes": stats["witness_bytes"] if ordered_denominators else 0,
            "nonzero_terms": stats["nonzero_terms"] if ordered_denominators else 0,
            "group_count": len(groups),
            "distinct_denominators": len(ordered_denominators),
        }


class FactoredRationalLazyAccumulator:
    def __init__(self, group_count: int) -> None:
        self.group_count = group_count
        self.by_denominator: dict[int, list[list[int]]] = {}
        self.nonzero_terms = 0

    def add_groups(self, groups: list[list[Fraction]]) -> None:
        if len(groups) != self.group_count:
            raise ValueError(f"expected {self.group_count} groups, got {len(groups)}")
        for group_index, group in enumerate(groups):
            for value in group:
                if value == 0:
                    continue
                denominator_groups = self.by_denominator.setdefault(
                    value.denominator,
                    [[] for _ in range(self.group_count)],
                )
                denominator_groups[group_index].append(value.numerator)
                self.nonzero_terms += 1

    def readout(self, worker: PStackAccumulatorWorker) -> tuple[list[Fraction], dict[str, int]]:
        results = [Fraction(0) for _ in range(self.group_count)]
        ordered_denominators = sorted(self.by_denominator)
        flattened_groups: list[list[int]] = []
        for denominator in ordered_denominators:
            flattened_groups.extend(self.by_denominator[denominator])
        integer_sums, stats = worker.sum_integer_groups(flattened_groups) if ordered_denominators else ([], {
            "support_card": 0,
            "active_levels": 0,
            "witness_bytes": 0,
            "nonzero_terms": 0,
            "group_count": 0,
        })
        cursor = 0
        for denominator in ordered_denominators:
            for group_index in range(self.group_count):
                integer_sum = integer_sums[cursor]
                if integer_sum != 0:
                    results[group_index] += Fraction(integer_sum, denominator)
                cursor += 1
        return results, {
            "support_card": stats["support_card"],
            "active_levels": stats["active_levels"],
            "witness_bytes": stats["witness_bytes"],
            "nonzero_terms": stats["nonzero_terms"],
            "group_count": self.group_count,
            "distinct_denominators": len(ordered_denominators),
        }


def _fraction_backend_sum_groups(groups: list[list[Fraction]]) -> tuple[list[Fraction], dict[str, int]]:
    results = [sum((value for value in group if value != 0), Fraction(0)) for group in groups]
    witness_bytes = sum(len(_fraction_string(value)) for group in groups for value in group if value != 0)
    nonzero_terms = sum(1 for group in groups for value in group if value != 0)
    return results, {
        "support_card": nonzero_terms,
        "active_levels": 0,
        "witness_bytes": witness_bytes,
        "nonzero_terms": nonzero_terms,
        "group_count": len(groups),
    }


def _rebuild_nested(values: list[Fraction], shapes: list[tuple[int, ...]]) -> list[Any]:
    cursor = 0
    rebuilt = []
    for shape in shapes:
        if len(shape) == 3:
            outer = []
            for _ in range(shape[0]):
                mid = []
                for _ in range(shape[1]):
                    mid.append(list(values[cursor : cursor + shape[2]]))
                    cursor += shape[2]
                outer.append(mid)
            rebuilt.append(outer)
        elif len(shape) == 2:
            outer = []
            for _ in range(shape[0]):
                outer.append(list(values[cursor : cursor + shape[1]]))
                cursor += shape[1]
            rebuilt.append(outer)
        else:
            raise ValueError(f"unsupported shape {shape!r}")
    return rebuilt


def forward_pstack(model: RationalKANDeg8, batch: list[tuple[Fraction, ...]], worker: PStackAccumulatorWorker) -> dict[str, Any]:
    edge_groups: list[list[Fraction]] = []
    edge_shapes: list[tuple[int, ...]] = []
    x_powers_batch = [[_powers(x_i, model.degree) for x_i in sample] for sample in batch]

    for sample_index, sample in enumerate(batch):
        del sample
        for hidden_index in range(model.hidden):
            for input_index in range(model.input_dim):
                coeffs = model.edge_coeffs[hidden_index][input_index]
                edge_groups.append(
                    [
                        coeffs[power] * x_powers_batch[sample_index][input_index][power]
                        for power in range(model.degree + 1)
                        if coeffs[power] != 0
                    ]
                )
    edge_values, edge_stats = worker.sum_groups(edge_groups)

    edge_outputs: list[list[list[Fraction]]] = []
    cursor = 0
    for _sample_index in range(len(batch)):
        sample_edges: list[list[Fraction]] = []
        for _hidden_index in range(model.hidden):
            hidden_edges: list[Fraction] = []
            for _input_index in range(model.input_dim):
                hidden_edges.append(edge_values[cursor])
                cursor += 1
            sample_edges.append(hidden_edges)
        edge_outputs.append(sample_edges)

    pre_groups = [
        [edge_outputs[sample_index][hidden_index][input_index] for input_index in range(model.input_dim) if edge_outputs[sample_index][hidden_index][input_index] != 0]
        for sample_index in range(len(batch))
        for hidden_index in range(model.hidden)
    ]
    pre_values, pre_stats = worker.sum_groups(pre_groups)

    pre_activations: list[list[Fraction]] = []
    cursor = 0
    for _sample_index in range(len(batch)):
        sample_preacts = []
        for _hidden_index in range(model.hidden):
            sample_preacts.append(pre_values[cursor])
            cursor += 1
        pre_activations.append(sample_preacts)

    output_groups = []
    for sample_index in range(len(batch)):
        terms = [model.bias] if model.bias != 0 else []
        terms.extend(
            model.output_weights[hidden_index] * pre_activations[sample_index][hidden_index]
            for hidden_index in range(model.hidden)
            if model.output_weights[hidden_index] != 0 and pre_activations[sample_index][hidden_index] != 0
        )
        output_groups.append(list(terms))
    predictions, output_stats = worker.sum_groups(output_groups)

    return {
        "predictions": predictions,
        "pre_activations": pre_activations,
        "activations": [list(row) for row in pre_activations],
        "edge_outputs": edge_outputs,
        "stats": {
            "edge": edge_stats,
            "pre": pre_stats,
            "output": output_stats,
            "support_card": edge_stats["support_card"] + pre_stats["support_card"] + output_stats["support_card"],
            "active_levels": edge_stats["active_levels"] + pre_stats["active_levels"] + output_stats["active_levels"],
            "witness_bytes": edge_stats["witness_bytes"] + pre_stats["witness_bytes"] + output_stats["witness_bytes"],
            "nonzero_terms": edge_stats["nonzero_terms"] + pre_stats["nonzero_terms"] + output_stats["nonzero_terms"],
        },
    }


def forward_pstack_inference(model: RationalKANDeg8, batch: list[tuple[Fraction, ...]], worker: PStackAccumulatorWorker) -> dict[str, Any]:
    groups: list[list[Fraction]] = []
    for sample in batch:
        x_powers = [_powers(x_i, model.degree) for x_i in sample]
        terms = [model.bias] if model.bias != 0 else []
        for hidden_index in range(model.hidden):
            weight = model.output_weights[hidden_index]
            if weight == 0:
                continue
            for input_index in range(model.input_dim):
                coeffs = model.edge_coeffs[hidden_index][input_index]
                for power in range(model.degree + 1):
                    coeff = coeffs[power]
                    if coeff != 0:
                        terms.append(weight * coeff * x_powers[input_index][power])
        groups.append(terms)
    predictions, stats = worker.sum_groups(groups)
    return {"predictions": predictions, "stats": stats}


def backward_pstack(
    model: RationalKANDeg8,
    batch: list[tuple[Fraction, ...]],
    targets: list[Fraction],
    forward: dict[str, Any],
    worker: PStackAccumulatorWorker,
    ) -> dict[str, Any]:
    predictions = forward["predictions"]
    pre_activations = forward["pre_activations"]
    batch_scale = Fraction(1, len(batch))
    diff_scales = [2 * (prediction - target) * batch_scale for prediction, target in zip(predictions, targets, strict=True)]
    groups, labels = gradient_event_groups(model, batch, diff_scales, pre_activations)
    values, stats = worker.sum_groups(groups)
    output_weight_grads = [Fraction(0) for _ in range(model.hidden)]
    bias_grad = Fraction(0)
    edge_grads = [
        [[Fraction(0) for _ in range(model.degree + 1)] for _ in range(model.input_dim)]
        for _ in range(model.hidden)
    ]
    for label, value in zip(labels, values, strict=True):
        kind, hidden_index, input_index, power = label
        if kind == "output":
            output_weight_grads[hidden_index] = value
        elif kind == "bias":
            bias_grad = value
        else:
            edge_grads[hidden_index][input_index][power] = value

    return {
        "loss": _mean_square_loss(predictions, targets),
        "output_weight_grads": output_weight_grads,
        "bias_grad": bias_grad,
        "edge_grads": edge_grads,
        "stats": stats,
    }


def gradient_event_groups(
    model: RationalKANDeg8,
    batch: list[tuple[Fraction, ...]],
    diff_scales: list[Fraction],
    pre_activations: list[list[Fraction]],
) -> tuple[list[list[Fraction]], list[tuple[str, int, int, int]]]:
    groups: list[list[Fraction]] = []
    labels: list[tuple[str, int, int, int]] = []

    for hidden_index in range(model.hidden):
        groups.append(
            [
                diff_scales[sample_index] * pre_activations[sample_index][hidden_index]
                for sample_index in range(len(batch))
            ]
        )
        labels.append(("output", hidden_index, 0, 0))

    groups.append(list(diff_scales))
    labels.append(("bias", 0, 0, 0))

    for hidden_index in range(model.hidden):
        for input_index in range(model.input_dim):
            for power in range(model.degree + 1):
                if model.output_weights[hidden_index] == 0:
                    groups.append([])
                else:
                    groups.append(
                        [
                            diff_scales[sample_index]
                            * model.output_weights[hidden_index]
                            * (batch[sample_index][input_index] ** power)
                            for sample_index in range(len(batch))
                        ]
                    )
                labels.append(("edge", hidden_index, input_index, power))
    return groups, labels


def dense_accumulate_group_updates(
    group_batches: list[list[list[Fraction]]],
) -> tuple[list[Fraction], dict[str, int]]:
    if not group_batches:
        return [], {
            "support_card": 0,
            "active_levels": 0,
            "witness_bytes": 0,
            "nonzero_terms": 0,
            "group_count": 0,
            "distinct_denominators": 0,
        }
    group_count = len(group_batches[0])
    totals = [Fraction(0) for _ in range(group_count)]
    witness_bytes = 0
    nonzero_terms = 0
    denominators: set[int] = set()
    for groups in group_batches:
        if len(groups) != group_count:
            raise ValueError(f"inconsistent group count: expected {group_count}, got {len(groups)}")
        for group_index, group in enumerate(groups):
            for value in group:
                if value == 0:
                    continue
                totals[group_index] += value
                witness_bytes += len(_fraction_string(value))
                nonzero_terms += 1
                denominators.add(value.denominator)
    return totals, {
        "support_card": nonzero_terms,
        "active_levels": 0,
        "witness_bytes": witness_bytes,
        "nonzero_terms": nonzero_terms,
        "group_count": group_count,
        "distinct_denominators": len(denominators),
    }


def training_step_fraction(
    model: RationalKANDeg8,
    batch: list[tuple[Fraction, ...]],
    targets: list[Fraction],
    *,
    lr: Fraction,
    denominator_bound: int,
) -> tuple[RationalKANDeg8, dict[str, Any]]:
    forward = model.forward_fraction(batch)
    backward = model.backward_fraction(batch, targets, forward)
    updated = model.apply_update(backward, lr=lr, denominator_bound=denominator_bound)
    return updated, {"forward": forward, "backward": backward}


def training_step_pstack(
    model: RationalKANDeg8,
    batch: list[tuple[Fraction, ...]],
    targets: list[Fraction],
    *,
    lr: Fraction,
    denominator_bound: int,
    worker: PStackAccumulatorWorker,
) -> tuple[RationalKANDeg8, dict[str, Any]]:
    forward = forward_pstack(model, batch, worker)
    backward = backward_pstack(model, batch, targets, forward, worker)
    updated = model.apply_update(backward, lr=lr, denominator_bound=denominator_bound)
    return updated, {"forward": forward, "backward": backward}


def _parity_record(step: int, tensor_name: str, fraction_value: Any, pstack_value: Any, stats: dict[str, int] | None = None) -> dict[str, Any]:
    record = {
        "step": step,
        "tensor_name": tensor_name,
        "path_fraction_hash": _tensor_hash(fraction_value),
        "path_pstack_hash": _tensor_hash(pstack_value),
        "fraction_values": _json_ready(fraction_value),
        "pstack_values": _json_ready(pstack_value),
        "equal": fraction_value == pstack_value,
    }
    if stats is not None:
        record["support_count"] = stats.get("support_card", 0)
        record["witness_bytes"] = stats.get("witness_bytes", 0)
        record["active_levels"] = stats.get("active_levels", 0)
        record["nonzero_terms"] = stats.get("nonzero_terms", 0)
    return record


def assert_bit_identical(
    *,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    out_log: Path | None = None,
) -> dict[str, Any]:
    worker = PStackAccumulatorWorker.instance()
    fraction_model = RationalKANDeg8.seeded(seed=seed)
    pstack_model = fraction_model
    records: list[dict[str, Any]] = []

    for step in range(steps):
        batch = sb_grid_batch(batch_size, seed + step * 17)
        targets = [target_degree8(x) for x in batch]
        fraction_forward = fraction_model.forward_fraction(batch)
        pstack_forward = forward_pstack(pstack_model, batch, worker)
        records.append(_parity_record(step, "pre_activations", fraction_forward["pre_activations"], pstack_forward["pre_activations"], pstack_forward["stats"]["pre"]))
        records.append(_parity_record(step, "activations", fraction_forward["activations"], pstack_forward["activations"], pstack_forward["stats"]["pre"]))
        records.append(_parity_record(step, "predictions", fraction_forward["predictions"], pstack_forward["predictions"], pstack_forward["stats"]["output"]))
        records.append(_parity_record(step, "edge_outputs", fraction_forward["edge_outputs"], pstack_forward["edge_outputs"], pstack_forward["stats"]["edge"]))

        fraction_backward = fraction_model.backward_fraction(batch, targets, fraction_forward)
        pstack_backward = backward_pstack(pstack_model, batch, targets, pstack_forward, worker)
        records.append(_parity_record(step, "output_weight_grads", fraction_backward["output_weight_grads"], pstack_backward["output_weight_grads"], pstack_backward["stats"]))
        records.append(_parity_record(step, "bias_grad", fraction_backward["bias_grad"], pstack_backward["bias_grad"], pstack_backward["stats"]))
        records.append(_parity_record(step, "edge_grads", fraction_backward["edge_grads"], pstack_backward["edge_grads"], pstack_backward["stats"]))

        fraction_model = fraction_model.apply_update(fraction_backward, lr=DEFAULT_LR, denominator_bound=DEFAULT_DEN_BOUND)
        pstack_model = pstack_model.apply_update(pstack_backward, lr=DEFAULT_LR, denominator_bound=DEFAULT_DEN_BOUND)
        records.append(_parity_record(step, "updated_model", fraction_model.to_json(), pstack_model.to_json(), pstack_backward["stats"]))

        if fraction_model != pstack_model:
            raise AssertionError(f"model mismatch at step {step}")
        if any(not record["equal"] for record in records[-8:]):
            raise AssertionError(f"tensor parity mismatch at step {step}")

    if out_log is not None:
        out_log.parent.mkdir(parents=True, exist_ok=True)
        with out_log.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    return {
        "parity_gate": all(record["equal"] for record in records),
        "records": len(records),
        "final_model": fraction_model,
    }


def train_model(*, steps: int = DEFAULT_STEPS, batch_size: int = DEFAULT_BATCH_SIZE, seed: int = DEFAULT_SEED) -> RationalKANDeg8:
    model = RationalKANDeg8.seeded(seed=seed)
    for step in range(steps):
        batch = sb_grid_batch(batch_size, seed + step * 17)
        targets = [target_degree8(x) for x in batch]
        model, _ = training_step_fraction(
            model,
            batch,
            targets,
            lr=DEFAULT_LR,
            denominator_bound=DEFAULT_DEN_BOUND,
        )
    return model


def _prediction_hash(predictions: list[Fraction]) -> str:
    return _tensor_hash(predictions)


def gradient_group_batches(
    model: RationalKANDeg8,
    *,
    microbatches: int,
    batch_size: int,
    seed: int,
) -> tuple[list[list[list[Fraction]]], str]:
    out: list[list[list[Fraction]]] = []
    hashes: list[str] = []
    for microbatch_index in range(microbatches):
        batch = sb_grid_batch(batch_size, seed + microbatch_index * 97)
        targets = [target_degree8(x) for x in batch]
        forward = model.forward_fraction(batch)
        predictions = forward["predictions"]
        batch_scale = Fraction(1, len(batch))
        diff_scales = [
            2 * (prediction - target) * batch_scale
            for prediction, target in zip(predictions, targets, strict=True)
        ]
        groups, _labels = gradient_event_groups(model, batch, diff_scales, forward["pre_activations"])
        out.append(groups)
        hashes.append(_tensor_hash(groups))
    return out, _tensor_hash(hashes)


def benchmark_update_lane(
    model: RationalKANDeg8,
    *,
    samples: int,
    batch_size: int,
    seed: int,
    microbatches: int = 64,
    model_state: str = "seeded_pretraining",
) -> dict[str, Any]:
    worker = PStackAccumulatorWorker.instance()
    rows = []
    ratios = []
    for sample_index in range(samples):
        group_batches, input_hash = gradient_group_batches(
            model,
            microbatches=microbatches,
            batch_size=batch_size,
            seed=seed + 20_000 + sample_index * 131,
        )

        t0 = time.perf_counter_ns()
        dense_values, dense_stats = dense_accumulate_group_updates(group_batches)
        dense_ns = max(1, time.perf_counter_ns() - t0)

        t1 = time.perf_counter_ns()
        lazy = FactoredRationalLazyAccumulator(len(group_batches[0]))
        for groups in group_batches:
            lazy.add_groups(groups)
        pstack_values, pstack_stats = lazy.readout(worker)
        pstack_ns = max(1, time.perf_counter_ns() - t1)

        dense_hash = _tensor_hash(dense_values)
        pstack_hash = _tensor_hash(pstack_values)
        if dense_values != pstack_values:
            raise AssertionError(f"update-lane hash mismatch at sample {sample_index}")

        ratio = dense_ns / pstack_ns
        ratios.append(ratio)
        rows.append(
            {
                "sample_index": sample_index,
                "seed": seed + 20_000 + sample_index * 131,
                "microbatches": microbatches,
                "batch_size": batch_size,
                "input_hash": input_hash,
                "dense_total_update_ns": dense_ns,
                "pstack_total_update_ns": pstack_ns,
                "speedup_ratio": ratio,
                "output_hash_dense": dense_hash,
                "output_hash_pstack": pstack_hash,
                "dense_support_count": dense_stats["support_card"],
                "pstack_support_count": pstack_stats["support_card"],
                "pstack_witness_bytes": pstack_stats["witness_bytes"],
                "pstack_active_levels": pstack_stats["active_levels"],
                "pstack_distinct_denominators": pstack_stats["distinct_denominators"],
            }
        )

    ci = bootstrap_ci(ratios, seed=seed + 177, n=BOOTSTRAP_SAMPLES)
    return {
        "benchmark_kind": "amortized_training_update",
        "model_state": model_state,
        "device_requested": "cpu",
        "device_executed": "cpu",
        "cuda_status": "not_requested",
        "samples": samples,
        "batch_size": batch_size,
        "microbatches": microbatches,
        "dense_backend": "python_fraction_immediate_updates",
        "pstack_backend": "rust_hz_factored_lazy_updates",
        "rows": rows,
        "mean_speedup": ci["mean"],
        "ci95_lower": ci["ci95_lower"],
        "ci95_upper": ci["ci95_upper"],
        "bootstrap_samples": ci["bootstrap_samples"],
        "speedup_gate": ci["ci95_lower"] > 1.0,
    }


def benchmark_inference(
    model: RationalKANDeg8,
    *,
    samples: int,
    batch_size: int,
    seed: int,
    model_state: str = "seeded_pretraining",
) -> dict[str, Any]:
    worker = PStackAccumulatorWorker.instance()
    rows = []
    ratios = []
    for sample_index in range(samples):
        batch = sb_grid_batch(batch_size, seed + 10_000 + sample_index * 97)
        t0 = time.perf_counter_ns()
        dense_forward = model.forward_fraction(batch)
        dense_ns = max(1, time.perf_counter_ns() - t0)

        t1 = time.perf_counter_ns()
        pstack_forward = forward_pstack_inference(model, batch, worker)
        pstack_ns = max(1, time.perf_counter_ns() - t1)

        dense_hash = _prediction_hash(dense_forward["predictions"])
        pstack_hash = _prediction_hash(pstack_forward["predictions"])
        if dense_hash != pstack_hash:
            raise AssertionError(f"inference hash mismatch at sample {sample_index}")

        ratio = dense_ns / pstack_ns
        ratios.append(ratio)
        rows.append(
            {
                "sample_index": sample_index,
                "seed": seed + 10_000 + sample_index * 97,
                "dense_total_inference_ns": dense_ns,
                "pstack_total_inference_ns": pstack_ns,
                "speedup_ratio": ratio,
                "output_hash_dense": dense_hash,
                "output_hash_pstack": pstack_hash,
                "support_count": pstack_forward["stats"]["support_card"],
                "witness_bytes": pstack_forward["stats"]["witness_bytes"],
                "active_levels": pstack_forward["stats"]["active_levels"],
            }
        )

    ci = bootstrap_ci(ratios, seed=seed + 77, n=BOOTSTRAP_SAMPLES)
    return {
        "benchmark_kind": "network_inference",
        "model_state": model_state,
        "device_requested": "cpu",
        "device_executed": "cpu",
        "cuda_status": "not_requested",
        "samples": samples,
        "batch_size": batch_size,
        "dense_backend": "python_fraction",
        "pstack_backend": "rust_hz_cli",
        "rows": rows,
        "mean_speedup": ci["mean"],
        "ci95_lower": ci["ci95_lower"],
        "ci95_upper": ci["ci95_upper"],
        "bootstrap_samples": ci["bootstrap_samples"],
        "speedup_gate": ci["ci95_lower"] > 1.0,
    }


def _unavailable_cuda_row(samples: int) -> dict[str, Any]:
    return {
        "device_requested": "cuda",
        "device_executed": "unavailable_exact_substrate",
        "cuda_status": "unavailable_exact_substrate",
        "samples": samples,
        "mean_speedup": None,
        "ci95_lower": None,
        "ci95_upper": None,
        "bootstrap_samples": 0,
        "speedup_gate": False,
    }


def _skipped_benchmark_row(
    *,
    benchmark_kind: str,
    dense_backend: str,
    pstack_backend: str,
    batch_size: int = 0,
    microbatches: int | None = None,
    model_state: str = "skipped_in_tests",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "benchmark_kind": benchmark_kind,
        "model_state": model_state,
        "device_requested": "cpu",
        "device_executed": "skipped_in_tests",
        "cuda_status": "not_applicable",
        "samples": 0,
        "batch_size": batch_size,
        "dense_backend": dense_backend,
        "pstack_backend": pstack_backend,
        "rows": [],
        "mean_speedup": None,
        "ci95_lower": None,
        "ci95_upper": None,
        "bootstrap_samples": 0,
        "speedup_gate": False,
    }
    if microbatches is not None:
        row["microbatches"] = microbatches
    return row


def benchmark_microbatch_sweep(
    model: RationalKANDeg8,
    *,
    samples: int,
    batch_size: int,
    seed: int,
    microbatch_grid: tuple[int, ...] = DEFAULT_MICROBATCH_GRID,
    model_state: str = "seeded_pretraining",
) -> dict[str, Any]:
    rows = []
    for microbatches in microbatch_grid:
        point = benchmark_update_lane(
            model,
            samples=samples,
            batch_size=batch_size,
            seed=seed + microbatches * 10_000,
            microbatches=microbatches,
            model_state=model_state,
        )
        rows.append(
            {
                "microbatches": microbatches,
                "samples": point["samples"],
                "mean_speedup": point["mean_speedup"],
                "ci95_lower": point["ci95_lower"],
                "ci95_upper": point["ci95_upper"],
                "speedup_gate": point["speedup_gate"],
                "mean_dense_total_update_ns": _mean_field(point["rows"], "dense_total_update_ns"),
                "mean_pstack_total_update_ns": _mean_field(point["rows"], "pstack_total_update_ns"),
                "mean_pstack_distinct_denominators": _mean_field(point["rows"], "pstack_distinct_denominators"),
                "mean_pstack_witness_bytes": _mean_field(point["rows"], "pstack_witness_bytes"),
                "hash_parity_passed": all(
                    row["output_hash_dense"] == row["output_hash_pstack"] for row in point["rows"]
                ),
            }
        )
    crossover = [row["microbatches"] for row in rows if row["ci95_lower"] is not None and row["ci95_lower"] > 1.0]
    return {
        "benchmark_kind": "update_microbatch_amortization_sweep",
        "model_state": model_state,
        "samples_per_point": samples,
        "batch_size": batch_size,
        "grid": list(microbatch_grid),
        "rows": rows,
        "crossover_microbatches": crossover[0] if crossover else None,
        "any_speedup_gate": bool(crossover),
    }


def _skipped_microbatch_sweep(microbatch_grid: tuple[int, ...]) -> dict[str, Any]:
    return {
        "benchmark_kind": "update_microbatch_amortization_sweep",
        "model_state": "skipped_in_tests",
        "samples_per_point": 0,
        "batch_size": 0,
        "grid": list(microbatch_grid),
        "rows": [],
        "crossover_microbatches": None,
        "any_speedup_gate": False,
    }


def write_network_spec(path: Path, model: RationalKANDeg8) -> dict[str, Any]:
    spec = model.network_spec()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def _write_json(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def write_benchmark_artifact(path: Path, cpu_row: dict[str, Any], include_cuda: bool) -> dict[str, Any]:
    data = {
        "commit": repo_commit(),
        "host_cpu": _cpu_model(),
        "thermal_state_start": _thermal_state(),
        "cpu": cpu_row,
        "cuda": _unavailable_cuda_row(cpu_row["samples"]) if include_cuda else None,
        "speedup_gate": cpu_row["speedup_gate"],
    }
    return _write_json(path, data)


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def collect_diagnostic_context(out_dir: Path) -> dict[str, Any]:
    bridge_path = (out_dir / "bridge_overhead_breakdown.json").resolve()
    width_path = (out_dir / "width_sweep.json").resolve()
    inference30_path = (out_dir / "inference_speedup_samples30.json").resolve()
    bridge = _load_optional_json(bridge_path)
    width = _load_optional_json(width_path)
    inference30 = _load_optional_json(inference30_path)
    bridge_share = None
    if bridge and bridge.get("stats"):
        total = float(bridge["stats"]["total_pstack_ns"]["mean"])
        roundtrip = float(bridge["stats"]["roundtrip_ns"]["mean"])
        bridge_share = roundtrip / total if total > 0 else None
    width_rows = []
    if width:
        width_rows = [
            {"width": row["width"], "mean_speedup": row["mean_speedup"]}
            for row in width.get("rows", [])
        ]
    return {
        "artifact_paths": {
            "bridge_overhead_breakdown": str(bridge_path.relative_to(REPO_ROOT)) if bridge else None,
            "width_sweep": str(width_path.relative_to(REPO_ROOT)) if width else None,
            "inference_speedup_samples30": str(inference30_path.relative_to(REPO_ROOT)) if inference30 else None,
        },
        "bridge_overhead": {
            "mean_roundtrip_share": bridge_share,
            "mean_dense_ns": None if not bridge else bridge["stats"]["dense_ns"]["mean"],
            "mean_total_pstack_ns": None if not bridge else bridge["stats"]["total_pstack_ns"]["mean"],
        },
        "width_sweep": {
            "rows": width_rows,
        },
        "inference_samples30": None
        if not inference30
        else {
            "samples": inference30.get("samples"),
            "mean_speedup": inference30.get("mean_speedup"),
            "ci95_lower": inference30.get("ci95_lower"),
            "ci95_upper": inference30.get("ci95_upper"),
        },
    }


def write_report(
    path: Path,
    *,
    parity_gate: bool,
    scale_gate: bool,
    speedup_gate: bool,
    spec: dict[str, Any],
    update_speed: dict[str, Any],
    inference_speed: dict[str, Any],
    microbatch_sweep: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    update_cpu = update_speed["cpu"]
    inference_cpu = inference_speed["cpu"]
    sweep_rows = microbatch_sweep["rows"]
    sweep_lines = [
        f"- m={row['microbatches']}: mean={row['mean_speedup']}, ci95=[{row['ci95_lower']}, {row['ci95_upper']}], gate={row['speedup_gate']}"
        for row in sweep_rows
    ]
    if not sweep_lines:
        sweep_lines = ["- sweep_not_run"]
    width_lines = [
        f"- width={row['width']}: mean_speedup={row['mean_speedup']}"
        for row in diagnostics["width_sweep"]["rows"]
    ] or ["- width_sweep_not_available"]
    bridge_summary = diagnostics["bridge_overhead"]
    inference30 = diagnostics["inference_samples30"]
    clause_lines = [
        f"- exact_fraction_semantics: {parity_gate}",
        f"- network_scale_requirement: {scale_gate}",
        f"- inference_speed_advantage_at_network_scale: {speedup_gate}",
    ]
    scope_note = (
        "The conjecture's speed clause is still keyed to network inference. "
        "The update-lane benchmark is architectural diagnosis, not a substitute clause."
    )
    lines = [
        "# P-stack In-Loop Report",
        "",
        f"- conjecture_statement={CONJECTURE_STATEMENT}",
        f"- parity_gate={parity_gate}",
        f"- scale_gate={scale_gate}",
        f"- speedup_gate={speedup_gate}",
        "",
        "## Clause Mapping",
        "",
        *clause_lines,
        f"- scope_note={scope_note}",
        "",
        "## Network",
        "",
        f"- degree={spec['degree']}",
        f"- hidden={spec['hidden']}",
        f"- input_dim={spec['input_dim']}",
        f"- param_count={spec['param_count']}",
        f"- active_hidden_units={spec['active_hidden_units']}",
        "",
        "## Update Lane",
        "",
        f"- benchmark_kind={update_cpu.get('benchmark_kind', 'unknown')}",
        f"- model_state={update_cpu.get('model_state', 'unknown')}",
        f"- dense_backend={update_cpu['dense_backend']}",
        f"- pstack_backend={update_cpu['pstack_backend']}",
        f"- cpu_mean_speedup={update_cpu['mean_speedup']}",
        f"- cpu_ci95=[{update_cpu['ci95_lower']}, {update_cpu['ci95_upper']}]",
        f"- microbatches={update_cpu.get('microbatches', 'n/a')}",
        f"- mean_distinct_denominators={_mean_field(update_cpu['rows'], 'pstack_distinct_denominators') if isinstance(update_cpu.get('rows'), list) else 'n/a'}",
        f"- mean_witness_bytes={_mean_field(update_cpu['rows'], 'pstack_witness_bytes') if isinstance(update_cpu.get('rows'), list) else 'n/a'}",
        "",
        "## Microbatch Sweep",
        "",
        f"- grid={microbatch_sweep['grid']}",
        f"- crossover_microbatches={microbatch_sweep['crossover_microbatches']}",
        f"- any_speedup_gate={microbatch_sweep['any_speedup_gate']}",
        *sweep_lines,
        "",
        "## Inference Lane",
        "",
        f"- benchmark_kind={inference_cpu.get('benchmark_kind', 'unknown')}",
        f"- model_state={inference_cpu.get('model_state', 'unknown')}",
        f"- dense_backend={inference_cpu['dense_backend']}",
        f"- pstack_backend={inference_cpu['pstack_backend']}",
        f"- cpu_mean_speedup={inference_cpu['mean_speedup']}",
        f"- cpu_ci95=[{inference_cpu['ci95_lower']}, {inference_cpu['ci95_upper']}]",
        f"- batch_size={inference_cpu.get('batch_size', 'n/a')}",
        f"- cuda_status={inference_speed['cuda']['cuda_status'] if inference_speed.get('cuda') else 'not_requested'}",
        "",
        "## Diagnostic Trail",
        "",
        "- Architectural correction: denominator-factored lazy accumulation replaced the old LCM-collapse lane for update benchmarking.",
        f"- bridge_overhead_breakdown={diagnostics['artifact_paths']['bridge_overhead_breakdown']}",
        f"- bridge_mean_roundtrip_share={bridge_summary['mean_roundtrip_share']}",
        f"- bridge_mean_dense_ns={bridge_summary['mean_dense_ns']}",
        f"- bridge_mean_total_pstack_ns={bridge_summary['mean_total_pstack_ns']}",
        f"- width_sweep={diagnostics['artifact_paths']['width_sweep']}",
        *width_lines,
        f"- inference_samples30={diagnostics['artifact_paths']['inference_speedup_samples30']}",
        (
            f"- inference_samples30_mean={inference30['mean_speedup']}, ci95=[{inference30['ci95_lower']}, {inference30['ci95_upper']}], samples={inference30['samples']}"
            if inference30
            else "- inference_samples30_not_available"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    out_dir: Path = DEFAULT_OUT,
    *,
    samples: int = 30,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    devices: tuple[str, ...] = ("cpu", "cuda"),
    seed: int = DEFAULT_SEED,
    skip_speed_benchmark: bool = False,
    microbatches: int = 64,
    inference_samples: int | None = None,
    microbatch_sweep_samples: int | None = None,
    microbatch_grid: tuple[int, ...] = DEFAULT_MICROBATCH_GRID,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parity = assert_bit_identical(steps=steps, batch_size=batch_size, seed=seed, out_log=out_dir / "bitwise_parity_log.jsonl")
    benchmark_model = RationalKANDeg8.seeded(seed=seed)
    trained_model = train_model(steps=steps, batch_size=batch_size, seed=seed)
    spec = write_network_spec(out_dir / "network_spec.json", trained_model)
    scale_gate = spec["meets_scale_requirement"]
    update_cpu = (
        _skipped_benchmark_row(
            benchmark_kind="amortized_training_update",
            dense_backend="python_fraction_immediate_updates",
            pstack_backend="rust_hz_factored_lazy_updates",
            microbatches=microbatches,
        )
        if skip_speed_benchmark
        else benchmark_update_lane(
            benchmark_model,
            samples=samples,
            batch_size=batch_size,
            seed=seed,
            microbatches=microbatches,
            model_state="seeded_pretraining",
        )
    )
    inference_cpu = (
        _skipped_benchmark_row(
            benchmark_kind="network_inference",
            dense_backend="python_fraction",
            pstack_backend="rust_hz_cli",
            batch_size=batch_size,
        )
        if skip_speed_benchmark
        else benchmark_inference(
            benchmark_model,
            samples=inference_samples or samples,
            batch_size=batch_size,
            seed=seed,
            model_state="seeded_pretraining",
        )
    )
    sweep = (
        _skipped_microbatch_sweep(microbatch_grid)
        if skip_speed_benchmark
        else benchmark_microbatch_sweep(
            benchmark_model,
            samples=microbatch_sweep_samples or samples,
            batch_size=batch_size,
            seed=seed,
            microbatch_grid=microbatch_grid,
            model_state="seeded_pretraining",
        )
    )
    update_speed = write_benchmark_artifact(out_dir / "update_speedup.json", update_cpu, include_cuda=("cuda" in devices))
    inference_speed = write_benchmark_artifact(out_dir / "inference_speedup.json", inference_cpu, include_cuda=("cuda" in devices))
    microbatch_sweep_artifact = _write_json(
        out_dir / "microbatch_amortization_sweep.json",
        {
            "commit": repo_commit(),
            "host_cpu": _cpu_model(),
            "thermal_state_start": _thermal_state(),
            **sweep,
        },
    )
    diagnostics = collect_diagnostic_context(out_dir)
    speedup_gate = inference_speed["speedup_gate"]
    write_report(
        out_dir / "report.md",
        parity_gate=parity["parity_gate"],
        scale_gate=scale_gate,
        speedup_gate=speedup_gate,
        spec=spec,
        update_speed=update_speed,
        inference_speed=inference_speed,
        microbatch_sweep=microbatch_sweep_artifact,
        diagnostics=diagnostics,
    )
    summary = {
        "conjecture_statement": CONJECTURE_STATEMENT,
        "parity_gate": parity["parity_gate"],
        "scale_gate": scale_gate,
        "speedup_gate": speedup_gate,
        "speedup_gate_reason": (
            f"inference ci95 lower bound {inference_cpu['ci95_lower']} is not above 1.0"
            if inference_cpu["ci95_lower"] is not None and not speedup_gate
            else "inference ci95 lower bound exceeds 1.0"
            if speedup_gate
            else "speed benchmark skipped"
        ),
        "clause_mapping": {
            "bit_identical_fraction_semantics": parity["parity_gate"],
            "network_scale_requirement": scale_gate,
            "inference_speed_advantage_at_network_scale": speedup_gate,
        },
        "scope_note": (
            "update_speedup.json is an architectural diagnosis lane; inference_speedup.json governs the conjecture's speed clause."
        ),
        "pstack_in_loop_gate": parity["parity_gate"] and scale_gate and speedup_gate,
        "network_spec": spec,
        "speed": {
            "update": update_speed,
            "inference": inference_speed,
            "microbatch_sweep": microbatch_sweep_artifact,
        },
        "diagnostic_artifacts": diagnostics,
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--microbatches", type=int, default=64)
    parser.add_argument("--inference-samples", type=int, default=None)
    parser.add_argument("--microbatch-sweep-samples", type=int, default=None)
    parser.add_argument("--microbatch-grid", type=int, nargs="+", default=list(DEFAULT_MICROBATCH_GRID))
    args = parser.parse_args()
    result = run(
        Path(args.out),
        samples=args.samples,
        steps=args.steps,
        batch_size=args.batch_size,
        devices=tuple(args.devices),
        seed=args.seed,
        microbatches=args.microbatches,
        inference_samples=args.inference_samples,
        microbatch_sweep_samples=args.microbatch_sweep_samples,
        microbatch_grid=tuple(args.microbatch_grid),
    )
    return 0 if result["parity_gate"] and result["scale_gate"] and result["speedup_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
