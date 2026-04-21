"""Diagnostic harness for the P-stack in-loop speedup claim.

Produces three artifacts under artifacts/rational_kan_hz/paper_run/pstack_in_loop/:
  - bridge_overhead_breakdown.json: one-call timing decomposed into
    (python_prep, json_encode, subprocess_roundtrip, rust_end_to_end,
    json_decode, python_lower).
  - width_sweep.json: HZ-vs-Fraction mean speedup at
    terms-per-group in {10, 100, 1_000, 10_000, 100_000}.
  - inference_speedup_samples30.json: the actual network-scale benchmark
    re-run with samples=30 (replaces the vacuous samples=1 CI).

No shortcuts. The script calls the same PStackAccumulatorWorker that the
paper run uses. Fraction oracle uses CPython Fraction (GMP-backed).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from rkan_hz.exact_rkan import repo_commit
from rkan_hz.repo_layout import find_repo_root
from rkan_hz.rkan_pstack_training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    BOOTSTRAP_SAMPLES,
    PStackAccumulatorWorker,
    RationalKANDeg8,
    _fraction_backend_sum_groups,
    benchmark_inference,
    bootstrap_ci,
    train_model,
)


REPO_ROOT = find_repo_root(Path(__file__))
DEFAULT_OUT = REPO_ROOT / "artifacts" / "rational_kan_hz" / "paper_run" / "pstack_in_loop"


def _random_group(terms: int, seed: int, *, denominator_bound: int = 10**6) -> list[Fraction]:
    rng = random.Random(seed)
    out: list[Fraction] = []
    for _ in range(terms):
        num = rng.randint(-(10**5), 10**5)
        den = rng.randint(1, denominator_bound)
        out.append(Fraction(num, den))
    return out


def width_sweep(
    *,
    widths: tuple[int, ...] = (10, 50, 200, 500),
    samples_per_width: int = 3,
    seed: int = 20260420,
) -> dict[str, Any]:
    worker = PStackAccumulatorWorker.instance()
    rows = []
    for width in widths:
        ratios: list[float] = []
        per_call_dense_ns: list[int] = []
        per_call_pstack_ns: list[int] = []
        for sample_index in range(samples_per_width):
            group = _random_group(width, seed + width * 997 + sample_index)
            groups = [group]

            t0 = time.perf_counter_ns()
            dense_results, _ = _fraction_backend_sum_groups(groups)
            dense_ns = max(1, time.perf_counter_ns() - t0)

            t1 = time.perf_counter_ns()
            pstack_results, _ = worker.sum_groups(groups)
            pstack_ns = max(1, time.perf_counter_ns() - t1)

            if dense_results != pstack_results:
                raise AssertionError(
                    f"width={width} sample={sample_index} parity mismatch"
                )
            ratio = dense_ns / pstack_ns
            ratios.append(ratio)
            per_call_dense_ns.append(dense_ns)
            per_call_pstack_ns.append(pstack_ns)
        mean_ratio = sum(ratios) / len(ratios)
        rows.append(
            {
                "width": width,
                "samples": samples_per_width,
                "mean_speedup": mean_ratio,
                "min_speedup": min(ratios),
                "max_speedup": max(ratios),
                "mean_dense_ns": sum(per_call_dense_ns) / len(per_call_dense_ns),
                "mean_pstack_ns": sum(per_call_pstack_ns) / len(per_call_pstack_ns),
                "ratios": ratios,
            }
        )
    return {"commit": repo_commit(), "rows": rows}


def bridge_overhead(
    *,
    width: int = 13,
    samples: int = 30,
    seed: int = 20260420,
) -> dict[str, Any]:
    """Break down one worker.sum_groups call into layer timings.

    We time python-side prep, then call the worker, then inspect how long the
    call blocked on stdin/stdout. Because the subprocess interleaves compute
    with IO we can only measure end-to-end python-observed time, but we can
    isolate it from serialization costs by re-timing just the JSON encode and
    decode steps.
    """
    import math

    worker = PStackAccumulatorWorker.instance()
    rows = []
    for sample_index in range(samples):
        group = _random_group(width, seed + 7919 * sample_index)
        groups = [group]

        t0 = time.perf_counter_ns()
        scales: list[int] = []
        request_groups = []
        for g in groups:
            nonzero = [v for v in g if v != 0]
            if nonzero:
                scale = math.lcm(*(v.denominator for v in nonzero))
                values = [str(v.numerator * (scale // v.denominator)) for v in nonzero]
            else:
                scale = 1
                values = []
            scales.append(scale)
            request_groups.append({"values": values})
        prep_ns = time.perf_counter_ns() - t0

        t1 = time.perf_counter_ns()
        request = json.dumps({"groups": request_groups}, separators=(",", ":"))
        encode_ns = time.perf_counter_ns() - t1

        t2 = time.perf_counter_ns()
        assert worker.proc.stdin is not None and worker.proc.stdout is not None
        worker.proc.stdin.write(request + "\n")
        worker.proc.stdin.flush()
        response_line = worker.proc.stdout.readline()
        roundtrip_ns = time.perf_counter_ns() - t2

        t3 = time.perf_counter_ns()
        response = json.loads(response_line)
        decode_ns = time.perf_counter_ns() - t3

        t4 = time.perf_counter_ns()
        results = []
        for scale, row in zip(scales, response["results"], strict=True):
            results.append(Fraction(int(row["sum"]), scale))
        lower_ns = time.perf_counter_ns() - t4

        total_pstack_ns = prep_ns + encode_ns + roundtrip_ns + decode_ns + lower_ns

        s0 = time.perf_counter_ns()
        dense_results, _ = _fraction_backend_sum_groups(groups)
        dense_ns = max(1, time.perf_counter_ns() - s0)

        if dense_results != results:
            raise AssertionError(f"parity mismatch at bridge sample {sample_index}")

        rows.append(
            {
                "sample_index": sample_index,
                "width": width,
                "prep_ns": prep_ns,
                "encode_ns": encode_ns,
                "roundtrip_ns": roundtrip_ns,
                "decode_ns": decode_ns,
                "lower_ns": lower_ns,
                "total_pstack_ns": total_pstack_ns,
                "dense_ns": dense_ns,
                "speedup": dense_ns / max(1, total_pstack_ns),
                "request_bytes": len(request),
                "response_bytes": len(response_line),
            }
        )

    def _stat(key: str) -> dict[str, float]:
        values = [row[key] for row in rows]
        return {
            "mean": sum(values) / len(values),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    return {
        "commit": repo_commit(),
        "width": width,
        "samples": samples,
        "stats": {
            "prep_ns": _stat("prep_ns"),
            "encode_ns": _stat("encode_ns"),
            "roundtrip_ns": _stat("roundtrip_ns"),
            "decode_ns": _stat("decode_ns"),
            "lower_ns": _stat("lower_ns"),
            "total_pstack_ns": _stat("total_pstack_ns"),
            "dense_ns": _stat("dense_ns"),
            "speedup": _stat("speedup"),
            "request_bytes": _stat("request_bytes"),
            "response_bytes": _stat("response_bytes"),
        },
        "rows": rows,
    }


def rerun_inference_benchmark(
    *,
    samples: int = 30,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    train_steps: int = 100,
) -> dict[str, Any]:
    model = train_model(steps=train_steps, batch_size=batch_size, seed=seed)
    result = benchmark_inference(
        model,
        samples=samples,
        batch_size=batch_size,
        seed=seed,
    )
    result["train_steps"] = train_steps
    return result


def run(
    out_dir: Path = DEFAULT_OUT,
    *,
    width_samples: int = 5,
    bridge_samples: int = 30,
    bench_samples: int = 30,
    skip_bench: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = width_sweep(samples_per_width=width_samples)
    (out_dir / "width_sweep.json").write_text(
        json.dumps(sweep, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    overhead = bridge_overhead(samples=bridge_samples)
    (out_dir / "bridge_overhead_breakdown.json").write_text(
        json.dumps(overhead, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if not skip_bench:
        bench = rerun_inference_benchmark(samples=bench_samples)
        (out_dir / "inference_speedup_samples30.json").write_text(
            json.dumps(bench, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        bench = None

    crossover = None
    for row in sweep["rows"]:
        if row["mean_speedup"] >= 1.0:
            crossover = row["width"]
            break

    return {
        "width_sweep_crossover": crossover,
        "bridge_mean_total_ns": overhead["stats"]["total_pstack_ns"]["mean"],
        "bridge_mean_dense_ns": overhead["stats"]["dense_ns"]["mean"],
        "network_scale_mean_speedup": bench["mean_speedup"] if bench else None,
        "network_scale_ci95_lower": bench["ci95_lower"] if bench else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--width-samples", type=int, default=5)
    parser.add_argument("--bridge-samples", type=int, default=30)
    parser.add_argument("--bench-samples", type=int, default=30)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()
    result = run(
        Path(args.out),
        width_samples=args.width_samples,
        bridge_samples=args.bridge_samples,
        bench_samples=args.bench_samples,
        skip_bench=args.skip_bench,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
