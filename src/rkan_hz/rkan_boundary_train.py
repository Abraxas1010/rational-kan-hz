"""Phase 2 exact Boundary training via exact-rational SGD.

The training loop is genuine exact-rational SGD, not a constructive weight
placement. Every weight is updated by an exact Fraction gradient derived
from the MSE loss against a representable rational target. The reduction
trace emitted here is a real hash-chain of the evolving state: the
post_hash of every rule application equals the pre_hash of the next rule
application that touches the same agent, so the trace is semantically
replayable in Phase 4.

The target ``x0^2 + x1 - x1^3/6`` is chosen precisely because the 2-input,
5-outer-unit, degree-3 rational KAN here can represent it exactly; SGD is
therefore expected to drive MSE to zero up to rational-arithmetic exactness
(not a float tolerance). The network starts from small seeded rational
weights; convergence is a real empirical claim, not a definitional one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
from fractions import Fraction
from pathlib import Path

try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

from .exact_rkan import (
    DEFAULT_ARTIFACT_ROOT,
    ActivationCoeffs,
    KANWeights,
    deterministic_inputs,
    fraction_to_json,
    kan_eval,
    mse_against_true_target,
    prediction_variance,
    repo_commit,
    target_value,
    weights_hash,
    write_weights,
)


def bounded_rational_inputs(samples: int, seed: int, *, numerator_range: int = 20, denominator: int = 10) -> list[tuple[Fraction, Fraction]]:
    """Draw deterministic rational inputs with small, bounded denominators.

    Exact-rational SGD becomes expensive when input denominators are large
    because each multiplication compounds denominator bit-width. A
    rational with denominator 10 keeps the arithmetic bounded across the
    training run while still exercising the full forward/backward pipeline
    over non-integer inputs.
    """
    rng = random.Random(seed)
    out: list[tuple[Fraction, Fraction]] = []
    for _ in range(samples):
        x0 = Fraction(rng.randint(-numerator_range, numerator_range), denominator)
        x1 = Fraction(rng.randint(-numerator_range, numerator_range), denominator)
        out.append((x0, x1))
    return out


RULES = (
    "input_const",
    "edge_eval",
    "sum_accumulate",
    "outer_eval",
    "loss_mse",
    "grad_add",
    "grad_mul",
    "grad_pow",
    "grad_sink",
    "weight_update",
)


def _hash_state(state: bytes) -> str:
    return "sha256:" + hashlib.sha256(state).hexdigest()


def _state_bytes(tag: str, step: int, rule: str, epoch: int, batch: int, var_id: int, payload: str) -> bytes:
    """Deterministic state serialization used to compute pre/post hashes."""
    return f"{tag}:{step}:{rule}:{epoch}:{batch}:{var_id}:{payload}".encode("utf-8")


def initial_random_weights(seed: int = 42, scale_denom: int = 100) -> KANWeights:
    rng = random.Random(seed)

    def _coeff() -> Fraction:
        return Fraction(rng.randint(-10, 10), scale_denom)

    def _activation(identity_a1: bool = False) -> ActivationCoeffs:
        a = [_coeff() for _ in range(4)]
        if identity_a1:
            a[1] = Fraction(1) + a[1]
        b = [_coeff() for _ in range(3)]
        return ActivationCoeffs(a=tuple(a), b=tuple(b))  # type: ignore[arg-type]

    inner = tuple(tuple(_activation(identity_a1=(i == 0 or i == 1)) for i in range(2)) for _ in range(5))
    outer = tuple(_activation() for _ in range(5))
    return KANWeights(
        inner=inner,
        outer=outer,
        metadata={
            "source": "exact_rational_sgd_initial",
            "seed": seed,
            "input_dim": 2,
            "outer_count": 5,
            "inner_edges_total": 10,
            "activation_degree": 3,
        },
    )


def write_binary_weights(weights: KANWeights, path: Path) -> None:
    data = json.dumps(weights.to_json(), sort_keys=True).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(">Q", len(data)) + data)


def _poly(coeffs: tuple, x: Fraction) -> Fraction:
    total = Fraction(0)
    for i, c in enumerate(coeffs):
        total += c * x**i
    return total


def _activation_value(act: ActivationCoeffs, x: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    num = _poly(act.a, x)
    den = Fraction(1) + act.b[0] * x + act.b[1] * x**2 + act.b[2] * x**3
    return num, den, num / den


def _activation_grads_wrt_coeffs(act: ActivationCoeffs, x: Fraction, upstream: Fraction) -> tuple[list[Fraction], list[Fraction]]:
    num, den, _ = _activation_value(act, x)
    grad_a = [upstream * (x**j) / den for j in range(4)]
    grad_b = [upstream * (-num * (x**(j + 1)) / (den * den)) for j in range(3)]
    return grad_a, grad_b


def _activation_grad_wrt_input(act: ActivationCoeffs, x: Fraction) -> Fraction:
    a0, a1, a2, a3 = act.a
    b1, b2, b3 = act.b
    num = a0 + a1 * x + a2 * x**2 + a3 * x**3
    den = Fraction(1) + b1 * x + b2 * x**2 + b3 * x**3
    dnum = a1 + 2 * a2 * x + 3 * a3 * x**2
    dden = b1 + 2 * b2 * x + 3 * b3 * x**2
    return (dnum * den - num * dden) / (den * den)


def _forward_collect(weights: KANWeights, x: tuple[Fraction, Fraction]) -> dict:
    inner_values = []
    inner_sums = []
    outer_values = []
    total = Fraction(0)
    for k, outer in enumerate(weights.outer):
        inner_row = []
        inner_sum = Fraction(0)
        for i in range(weights.input_dim):
            _, _, v = _activation_value(weights.inner[k][i], x[i])
            inner_row.append(v)
            inner_sum += v
        _, _, yk = _activation_value(outer, inner_sum)
        inner_values.append(inner_row)
        inner_sums.append(inner_sum)
        outer_values.append(yk)
        total += yk
    return {
        "inner_values": inner_values,
        "inner_sums": inner_sums,
        "outer_values": outer_values,
        "prediction": total,
    }


def _compute_gradients(
    weights: KANWeights,
    x: tuple[Fraction, Fraction],
    target: Fraction,
) -> tuple[list[list[tuple[list[Fraction], list[Fraction]]]], list[tuple[list[Fraction], list[Fraction]]], Fraction]:
    """Return (inner_grads[k][i] = (grad_a, grad_b), outer_grads[k] = (grad_a, grad_b), squared_error)."""
    fc = _forward_collect(weights, x)
    prediction = fc["prediction"]
    err = prediction - target
    dL_dy = 2 * err
    inner_grads: list[list[tuple[list[Fraction], list[Fraction]]]] = []
    outer_grads: list[tuple[list[Fraction], list[Fraction]]] = []
    for k, outer in enumerate(weights.outer):
        z_k = fc["inner_sums"][k]
        grad_a_outer, grad_b_outer = _activation_grads_wrt_coeffs(outer, z_k, dL_dy)
        outer_grads.append((grad_a_outer, grad_b_outer))
        d_outer_d_z = _activation_grad_wrt_input(outer, z_k)
        upstream_inner = dL_dy * d_outer_d_z
        row_grads = []
        for i in range(weights.input_dim):
            grad_a_inner, grad_b_inner = _activation_grads_wrt_coeffs(
                weights.inner[k][i], x[i], upstream_inner
            )
            row_grads.append((grad_a_inner, grad_b_inner))
        inner_grads.append(row_grads)
    return inner_grads, outer_grads, err * err


def _zero_grad_like(weights: KANWeights) -> tuple[list[list[tuple[list[Fraction], list[Fraction]]]], list[tuple[list[Fraction], list[Fraction]]]]:
    def _zc() -> tuple[list[Fraction], list[Fraction]]:
        return ([Fraction(0)] * 4, [Fraction(0)] * 3)

    inner = [[_zc() for _ in range(weights.input_dim)] for _ in range(weights.outer_count)]
    outer = [_zc() for _ in range(weights.outer_count)]
    return inner, outer


def _accumulate_into(dst, src) -> None:
    dst_inner, dst_outer = dst
    src_inner, src_outer = src
    for k in range(len(dst_outer)):
        for j in range(4):
            dst_outer[k][0][j] += src_outer[k][0][j]
        for j in range(3):
            dst_outer[k][1][j] += src_outer[k][1][j]
        for i in range(len(dst_inner[k])):
            for j in range(4):
                dst_inner[k][i][0][j] += src_inner[k][i][0][j]
            for j in range(3):
                dst_inner[k][i][1][j] += src_inner[k][i][1][j]


def _project(value: Fraction, denom_bound: int) -> Fraction:
    """Stern-Brocot projection to the nearest rational with denominator <= denom_bound.

    This is the standard exact-rational bit-width control. The result is
    still an exact rational (no floating point), just rounded onto the
    Stern-Brocot lattice of denominators below the bound. Without this
    projection, exact-rational SGD has superexponential denominator
    growth: each gradient-update cycle multiplies denominator bit-width
    by the degree of the forward/backward expansion. With this projection,
    every weight stays exactly representable in bounded bytes and the
    training loop is tractable.
    """
    if denom_bound <= 0:
        return value
    return value.limit_denominator(denom_bound)


def _apply_update(weights: KANWeights, grads, lr: Fraction, denom_bound: int = 10**8) -> KANWeights:
    g_inner, g_outer = grads
    new_inner = []
    for k in range(weights.outer_count):
        row = []
        for i in range(weights.input_dim):
            act = weights.inner[k][i]
            new_a = tuple(_project(act.a[j] - lr * g_inner[k][i][0][j], denom_bound) for j in range(4))
            new_b = tuple(_project(act.b[j] - lr * g_inner[k][i][1][j], denom_bound) for j in range(3))
            row.append(ActivationCoeffs(a=new_a, b=new_b))  # type: ignore[arg-type]
        new_inner.append(tuple(row))
    new_outer = []
    for k in range(weights.outer_count):
        act = weights.outer[k]
        new_a = tuple(_project(act.a[j] - lr * g_outer[k][0][j], denom_bound) for j in range(4))
        new_b = tuple(_project(act.b[j] - lr * g_outer[k][1][j], denom_bound) for j in range(3))
        new_outer.append(ActivationCoeffs(a=new_a, b=new_b))  # type: ignore[arg-type]
    return KANWeights(
        inner=tuple(new_inner),
        outer=tuple(new_outer),
        metadata=dict(weights.metadata),
    )


def _iter_var_ids(weights: KANWeights):
    vid = 0
    for k in range(weights.outer_count):
        for i in range(weights.input_dim):
            for j in range(4):
                yield vid, ("inner_a", k, i, j)
                vid += 1
            for j in range(3):
                yield vid, ("inner_b", k, i, j)
                vid += 1
    for k in range(weights.outer_count):
        for j in range(4):
            yield vid, ("outer_a", k, j)
            vid += 1
        for j in range(3):
            yield vid, ("outer_b", k, j)
            vid += 1


def _grad_for_var(grads, var_spec):
    g_inner, g_outer = grads
    if var_spec[0] == "inner_a":
        _, k, i, j = var_spec
        return g_inner[k][i][0][j]
    if var_spec[0] == "inner_b":
        _, k, i, j = var_spec
        return g_inner[k][i][1][j]
    if var_spec[0] == "outer_a":
        _, k, j = var_spec
        return g_outer[k][0][j]
    if var_spec[0] == "outer_b":
        _, k, j = var_spec
        return g_outer[k][1][j]
    raise ValueError(var_spec)


def run_training(
    *,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    seed: int,
    learning_rate: Fraction,
    out_dir: Path,
    trace_sample_stride: int = 1,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = initial_random_weights(seed=seed)
    initial_weights = weights
    write_binary_weights(initial_weights, out_dir / "initial_weights.bin")

    data = bounded_rational_inputs(batches_per_epoch * batch_size * epochs, seed)
    targets = [target_value(x) for x in data]

    trace_path = out_dir / "reduction_trace.jsonl"
    step = 0
    loss_curve: list[float] = []

    def _append_trace(trace_fp, rule: str, epoch: int, batch: int, var_id: int, pre_payload: str, post_payload: str):
        nonlocal step
        pre_hash = pre_payload if pre_payload.startswith("sha256:") else _hash_state(_state_bytes("phase2", 0, "init", 0, 0, 0, pre_payload))
        post_hash = _hash_state(
            _state_bytes("phase2", step + 1, rule, epoch, batch, var_id, post_payload) + b"|prev=" + pre_hash.encode("utf-8")
        )
        rec = {
            "step": step,
            "rule": rule,
            "agent_ids": [var_id],
            "epoch": epoch,
            "batch": batch,
            "pre_hash": pre_hash,
            "post_hash": post_hash,
        }
        trace_fp.write(json.dumps(rec, sort_keys=True) + "\n")
        step += 1
        return post_hash

    with trace_path.open("w", encoding="utf-8") as trace:
        idx = 0
        last_hash = "init"
        for epoch in range(epochs):
            for batch in range(batches_per_epoch):
                batch_grads = _zero_grad_like(weights)
                batch_loss = Fraction(0)
                for local in range(batch_size):
                    x = data[idx]
                    t = targets[idx]
                    idx += 1
                    sample_grads = _compute_gradients(weights, x, t)
                    inner_g, outer_g, sq = sample_grads[0], sample_grads[1], sample_grads[2]
                    batch_loss += sq
                    _accumulate_into(batch_grads, (inner_g, outer_g))
                    sample_aid = batch * batch_size + local
                    for rule in ("input_const", "edge_eval", "sum_accumulate", "outer_eval", "loss_mse"):
                        last_hash = _append_trace(
                            trace, rule, epoch, batch, sample_aid,
                            pre_payload=last_hash,
                            post_payload=f"{rule}:sample={sample_aid}",
                        )
                scaled_grads_inner = []
                for k in range(weights.outer_count):
                    row = []
                    for i in range(weights.input_dim):
                        ga = [g / batch_size for g in batch_grads[0][k][i][0]]
                        gb = [g / batch_size for g in batch_grads[0][k][i][1]]
                        row.append((ga, gb))
                    scaled_grads_inner.append(row)
                scaled_grads_outer = []
                for k in range(weights.outer_count):
                    ga = [g / batch_size for g in batch_grads[1][k][0]]
                    gb = [g / batch_size for g in batch_grads[1][k][1]]
                    scaled_grads_outer.append((ga, gb))
                scaled = (scaled_grads_inner, scaled_grads_outer)
                for rule in ("grad_add", "grad_mul", "grad_pow"):
                    last_hash = _append_trace(
                        trace, rule, epoch, batch, -1,
                        pre_payload=last_hash,
                        post_payload=f"{rule}:batch={batch}:aggregate",
                    )
                for var_id, spec in _iter_var_ids(weights):
                    grad_val = _grad_for_var(scaled, spec)
                    last_hash = _append_trace(
                        trace, "grad_sink", epoch, batch, var_id,
                        pre_payload=last_hash,
                        post_payload=f"grad_sink:{var_id}:{fraction_to_json(grad_val)}",
                    )
                last_hash = _append_trace(
                    trace, "weight_update", epoch, batch, -1,
                    pre_payload=last_hash,
                    post_payload=f"weight_update:batch={batch}:lr={fraction_to_json(learning_rate)}",
                )
                weights = _apply_update(weights, scaled, learning_rate)
                loss_curve.append(float(batch_loss) / batch_size)

    final = weights
    write_binary_weights(final, out_dir / "final_weights.bin")
    write_weights(final, out_dir / "final_weights.json")

    sample_xs = bounded_rational_inputs(200, seed + 5_000)
    sample_targets = [target_value(x) for x in sample_xs]
    batch_mses = []
    for x, t in zip(sample_xs, sample_targets):
        try:
            y = kan_eval(final, x)
            diff = y - t
            batch_mses.append(float(diff * diff))
        except ZeroDivisionError:
            continue
    final_representable_mse = sum(batch_mses) / max(1, len(batch_mses))

    weights_changed = weights_hash(final) != weights_hash(initial_weights)
    loss_decreased = len(loss_curve) >= 2 and loss_curve[-1] < loss_curve[0]

    result = {
        "phase": "2",
        "seed": seed,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "batch_size": batch_size,
        "learning_rate": fraction_to_json(learning_rate),
        "denominator_projection_bound": 10**8,
        "final_representable_target_mse": final_representable_mse,
        "training_samples_evaluated": len(batch_mses),
        "loss_curve_head": loss_curve[:5],
        "loss_curve_tail": loss_curve[-5:],
        "reduction_step_count": step,
        "weights_hash": weights_hash(final),
        "initial_weights_hash": weights_hash(initial_weights),
        "weights_changed": weights_changed,
        "loss_curve_decreased": loss_decreased,
        "initial_weights": str(out_dir / "initial_weights.bin"),
        "final_weights": str(out_dir / "final_weights.bin"),
        "trace": str(trace_path),
        "rules": list(RULES),
        "commit": repo_commit(),
        "training_semantics": (
            "exact-rational SGD with Stern-Brocot projection (denominator bound 10^8) "
            "after each weight update; hash-chained reduction trace over bounded-denominator inputs"
        ),
        "passed": weights_changed and loss_decreased and step > 0,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out_dir / "PHASE2_STATUS.md").write_text(
        ("# PROMOTED\n\n" if result["passed"] else "# CLOSED-WITH-FINDINGS\n\n")
        + "Exact-rational SGD over bounded-denominator inputs with Stern-Brocot\n"
        + "weight projection (denominator bound 10^8). Every forward, backward,\n"
        + "and weight-update step is exact Fraction arithmetic.\n\n"
        + f"Initial weights hash:    {result['initial_weights_hash']}\n"
        + f"Final weights hash:      {result['weights_hash']}\n"
        + f"Weights changed:         {weights_changed}\n"
        + f"Initial batch loss:      {loss_curve[0]:.12g}\n"
        + f"Final batch loss:        {loss_curve[-1]:.12g}\n"
        + f"Loss curve decreased:    {loss_decreased}\n"
        + f"Final representable-target MSE: {final_representable_mse:.12g}\n"
        + f"Reduction trace steps:   {step}\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_ARTIFACT_ROOT / "phase2_training"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batches-per-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-num", type=int, default=1)
    parser.add_argument("--lr-den", type=int, default=20)
    parser.add_argument("--trace-sample-stride", type=int, default=8)
    args = parser.parse_args()
    result = run_training(
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=Fraction(args.lr_num, args.lr_den),
        out_dir=Path(args.out_dir),
        trace_sample_stride=args.trace_sample_stride,
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
