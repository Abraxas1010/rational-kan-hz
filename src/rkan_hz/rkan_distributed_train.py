"""Phase 2.5 real two-host tally-merge bit-identity test.

The test partitions a single full-batch gradient computation between two
simulated hosts, serializes each host's per-VarId exact-rational gradient
tally to a JSONL file, merges the tallies in canonical sorted VarId order,
applies the merged gradient to the shared initial weights, and verifies
byte-equality against the sequential one-step run on the same initial
weights with the same batch. Every arithmetic operation is on exact
Fractions; there is no float rounding, and the bit-identity claim is a
genuine consequence of the associativity of exact rational addition, not
a tautology.

This is a single-host surrogate in the sense that both "hosts" run in the
same process. Cross-hardware bit identity is still a future evidence
upgrade (real two-machine rerun). What is not a surrogate: the partition
is disjoint, the tallies are genuinely different per host, and the merge
is a real sum in canonical VarId order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from .exact_rkan import (
    DEFAULT_ARTIFACT_ROOT,
    deterministic_inputs,
    fraction_to_json,
    repo_commit,
    target_value,
    weights_hash,
)
from .rkan_boundary_train import (
    _accumulate_into,
    _apply_update,
    _compute_gradients,
    _grad_for_var,
    _iter_var_ids,
    _zero_grad_like,
    initial_random_weights,
    write_binary_weights,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scale(grads, batch_size: int):
    g_inner, g_outer = grads
    scaled_inner = []
    for k in range(len(g_outer)):
        row = []
        for i in range(len(g_inner[k])):
            ga = [g / batch_size for g in g_inner[k][i][0]]
            gb = [g / batch_size for g in g_inner[k][i][1]]
            row.append((ga, gb))
        scaled_inner.append(row)
    scaled_outer = []
    for k in range(len(g_outer)):
        ga = [g / batch_size for g in g_outer[k][0]]
        gb = [g / batch_size for g in g_outer[k][1]]
        scaled_outer.append((ga, gb))
    return (scaled_inner, scaled_outer)


def run(out_dir: Path, *, seed: int = 42, full_batch_size: int = 16, lr: Fraction = Fraction(1, 20)) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = initial_random_weights(seed=seed)
    data = deterministic_inputs(full_batch_size, seed + 77)
    targets = [target_value(x) for x in data]

    # Sequential one-step full-batch gradient descent from initial weights.
    seq_grads = _zero_grad_like(weights)
    for x, t in zip(data, targets):
        inner_g, outer_g, _ = _compute_gradients(weights, x, t)
        _accumulate_into(seq_grads, (inner_g, outer_g))
    seq_scaled = _scale(seq_grads, full_batch_size)
    sequential_weights = _apply_update(weights, seq_scaled, lr)
    sequential_path = out_dir / "sequential_one_step_weights.bin"
    write_binary_weights(sequential_weights, sequential_path)

    # Partitioned host_a and host_b: disjoint halves of the batch.
    half = full_batch_size // 2
    partitions = {
        "host_a": list(zip(data[:half], targets[:half])),
        "host_b": list(zip(data[half:], targets[half:])),
    }

    partial_tallies: dict[str, object] = {}
    for host, pairs in partitions.items():
        host_grads = _zero_grad_like(weights)
        for x, t in pairs:
            inner_g, outer_g, _ = _compute_gradients(weights, x, t)
            _accumulate_into(host_grads, (inner_g, outer_g))
        partial_tallies[host] = host_grads
        with (out_dir / f"{host}_grad_tally.jsonl").open("w", encoding="utf-8") as f:
            for var_id, spec in _iter_var_ids(weights):
                grad_val = _grad_for_var(host_grads, spec)
                rec = {
                    "host": host,
                    "epoch": 0,
                    "batch_partition": host,
                    "var_id": var_id,
                    "var_spec": list(spec),
                    "partial_sum": fraction_to_json(grad_val),
                }
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    # Merge by reading both tally files back in canonical sorted VarId order
    # (the substantive test: we do not use the in-memory tallies here).
    merged_tally: dict[int, Fraction] = {}
    var_specs: dict[int, tuple] = {}
    for host in ("host_a", "host_b"):
        with (out_dir / f"{host}_grad_tally.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                var_id = rec["var_id"]
                frac = Fraction(rec["partial_sum"])
                merged_tally[var_id] = merged_tally.get(var_id, Fraction(0)) + frac
                var_specs[var_id] = tuple(rec["var_spec"])

    merged_grads = _zero_grad_like(weights)
    for var_id in sorted(merged_tally.keys()):
        spec = var_specs[var_id]
        v = merged_tally[var_id]
        if spec[0] == "inner_a":
            _, k, i, j = spec
            merged_grads[0][k][i][0][j] = v
        elif spec[0] == "inner_b":
            _, k, i, j = spec
            merged_grads[0][k][i][1][j] = v
        elif spec[0] == "outer_a":
            _, k, j = spec
            merged_grads[1][k][0][j] = v
        elif spec[0] == "outer_b":
            _, k, j = spec
            merged_grads[1][k][1][j] = v
    merged_scaled = _scale(merged_grads, full_batch_size)
    merged_weights = _apply_update(weights, merged_scaled, lr)

    # Both hosts independently apply the merged gradients to verify
    # deterministic cross-host replay.
    host_a_weights = _apply_update(weights, merged_scaled, lr)
    host_b_weights = _apply_update(weights, merged_scaled, lr)
    write_binary_weights(merged_weights, out_dir / "merged_weights.bin")
    write_binary_weights(host_a_weights, out_dir / "host_a_replay_weights.bin")
    write_binary_weights(host_b_weights, out_dir / "host_b_replay_weights.bin")

    merged_hash = sha(out_dir / "merged_weights.bin")
    sequential_hash = sha(sequential_path)
    host_a_hash = sha(out_dir / "host_a_replay_weights.bin")
    host_b_hash = sha(out_dir / "host_b_replay_weights.bin")

    tallies_differ = Fraction(0)
    per_var_diff_is_zero = 0
    per_var_count = 0
    for var_id, spec in _iter_var_ids(weights):
        va = _grad_for_var(partial_tallies["host_a"], spec)
        vb = _grad_for_var(partial_tallies["host_b"], spec)
        d = va - vb
        tallies_differ += abs(d)
        per_var_count += 1
        if d == 0:
            per_var_diff_is_zero += 1
    tallies_differ_is_zero = tallies_differ == 0
    tallies_differ_float = float(tallies_differ)

    report = {
        "phase": "2.5",
        "mode": "single-host surrogate with real disjoint partition and exact-rational tally merge",
        "full_batch_size": full_batch_size,
        "partition": {"host_a_samples": half, "host_b_samples": full_batch_size - half},
        "learning_rate": fraction_to_json(lr),
        "initial_weights_hash": weights_hash(weights),
        "host_a_hash": host_a_hash,
        "host_b_hash": host_b_hash,
        "merged_hash": merged_hash,
        "sequential_hash": sequential_hash,
        "trivial_consequence_of_construction": [
            "host_a_hash == host_b_hash == merged_hash (both hosts apply the same merged tally to the same initial weights)"
        ],
        "substantive_claims": [
            "merged_hash == sequential_hash (partitioned merge equals sequential full-batch)",
            "host_a and host_b partial tallies are genuinely different",
            "both partial tallies are loaded from disk and merged in canonical sorted VarId order",
        ],
        "hosts_byte_equal": host_a_hash == host_b_hash,
        "matches_sequential_one_step": merged_hash == sequential_hash,
        "partial_tallies_differ_is_zero": tallies_differ_is_zero,
        "partial_tallies_differ_l1_float": tallies_differ_float,
        "per_var_tally_comparison": {
            "total_vars": per_var_count,
            "vars_with_zero_diff": per_var_diff_is_zero,
            "vars_with_nonzero_diff": per_var_count - per_var_diff_is_zero,
        },
        "commit": repo_commit(),
    }
    (out_dir / "bit_identity_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ok = report["matches_sequential_one_step"] and report["hosts_byte_equal"] and tallies_differ != 0
    (out_dir / "PHASE2_5_STATUS.md").write_text(
        ("# PROMOTED\n\n" if ok else "# CLOSED-WITH-FINDINGS\n\n")
        + "Real partitioned tally merge in exact rationals.\n\n"
        + f"Sequential one-step hash: {sequential_hash}\n"
        + f"Merged tally hash:        {merged_hash}\n"
        + f"Host A / B hashes:        {host_a_hash}, {host_b_hash}\n"
        + f"Partial-tally L1 non-zero: {not tallies_differ_is_zero}\n"
        + f"Per-var non-zero diffs: {per_var_count - per_var_diff_is_zero} / {per_var_count}\n\n"
        + "Honest boundary: both hosts run in the same process. Cross-hardware\n"
        + "bit identity requires a future two-machine rerun; the exact-rational\n"
        + "merge protocol itself is fully exercised here.\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_ARTIFACT_ROOT / "phase2_5_distributed"))
    parser.add_argument("--run-bit-identity-check", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-batch-size", type=int, default=16)
    parser.add_argument("--lr-num", type=int, default=1)
    parser.add_argument("--lr-den", type=int, default=20)
    parser.add_argument("--out")
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if not args.out else Path(args.out).parent
    report = run(
        out_dir,
        seed=args.seed,
        full_batch_size=args.full_batch_size,
        lr=Fraction(args.lr_num, args.lr_den),
    )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if (report["matches_sequential_one_step"] and report["hosts_byte_equal"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
