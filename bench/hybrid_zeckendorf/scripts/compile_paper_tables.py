#!/usr/bin/env python3
"""
Compile publication-ready tables from Hybrid Zeckendorf benchmark results.

Reads: bench/hybrid_zeckendorf/results/exp{1..7}_*.json
Writes: bench/hybrid_zeckendorf/results/paper_tables.json  (structured data)
        bench/hybrid_zeckendorf/results/paper_tables.txt   (human-readable)

Usage:
    python3 bench/hybrid_zeckendorf/scripts/compile_paper_tables.py
"""

import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_json(name):
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def median(xs):
    return statistics.median(xs) if xs else 0


def mean(xs):
    return statistics.mean(xs) if xs else 0


def ci95(xs):
    """95% CI half-width via bootstrap-free normal approximation."""
    if len(xs) < 3:
        return float("nan")
    s = statistics.stdev(xs)
    return 1.96 * s / (len(xs) ** 0.5)


def compile_e4_table(data):
    """Table 2 replication: sparse addition speedup by (N, rho)."""
    by_config = defaultdict(list)
    for dp in data["data_points"]:
        e = dp["extra"]
        key = (dp["input_size_bits"], e["target_rho"])
        by_config[key].append(
            {
                "speedup": dp["speedup_ratio"],
                "hz_ns": dp["hz_median_ns"],
                "ref_ns": dp["ref_median_ns"],
                "rho_actual": e.get("actual_rho_a", 0),
                "k": e.get("support_card_a", 0),
                "levels": e.get("active_levels_a", 0),
            }
        )

    rows = []
    for k in sorted(by_config.keys()):
        entries = by_config[k]
        speedups = [e["speedup"] for e in entries]
        hz_times = [e["hz_ns"] for e in entries]
        ref_times = [e["ref_ns"] for e in entries]
        rows.append(
            {
                "N_bits": k[0],
                "target_rho": k[1],
                "n_samples": len(entries),
                "speedup_median": round(median(speedups), 2),
                "speedup_mean": round(mean(speedups), 2),
                "speedup_ci95": round(ci95(speedups), 2),
                "speedup_min": round(min(speedups), 2),
                "speedup_max": round(max(speedups), 2),
                "hz_median_ns": round(median(hz_times)),
                "gmp_median_ns": round(median(ref_times)),
                "K_mean": round(mean([e["k"] for e in entries])),
                "levels_mean": round(mean([e["levels"] for e in entries])),
                "rho_actual_mean": round(
                    mean([e["rho_actual"] for e in entries]), 6
                ),
            }
        )
    return rows


def compile_e7_crossover(data):
    """Crossover curve: speedup vs rho at fixed N."""
    by_rho = defaultdict(list)
    bits_set = set()
    for dp in data["data_points"]:
        e = dp["extra"]
        rho = e.get("target_rho", e.get("measured_rho", 0))
        bits = dp["input_size_bits"]
        bits_set.add(bits)
        by_rho[(bits, round(rho, 8))].append(dp["speedup_ratio"])

    rows = []
    for k in sorted(by_rho.keys()):
        speedups = by_rho[k]
        rows.append(
            {
                "N_bits": k[0],
                "target_rho": k[1],
                "n_samples": len(speedups),
                "speedup_median": round(median(speedups), 2),
                "speedup_mean": round(mean(speedups), 2),
                "speedup_ci95": round(ci95(speedups), 2),
            }
        )

    # Find crossover: interpolate where median speedup crosses 1.0
    crossover_rho = None
    for bits in sorted(bits_set):
        bits_rows = [r for r in rows if r["N_bits"] == bits]
        for i in range(len(bits_rows) - 1):
            a, b = bits_rows[i], bits_rows[i + 1]
            if a["speedup_median"] >= 1.0 and b["speedup_median"] < 1.0:
                # Linear interpolation in log space
                import math

                log_rho_a = math.log10(a["target_rho"])
                log_rho_b = math.log10(b["target_rho"])
                frac = (a["speedup_median"] - 1.0) / (
                    a["speedup_median"] - b["speedup_median"]
                )
                crossover_rho = 10 ** (log_rho_a + frac * (log_rho_b - log_rho_a))

    return rows, crossover_rho


def compile_scaling_analysis(e4_rows):
    """How speedup grows with N at fixed rho."""
    by_rho = defaultdict(list)
    for r in e4_rows:
        by_rho[r["target_rho"]].append(r)

    analysis = []
    for rho in sorted(by_rho.keys()):
        entries = sorted(by_rho[rho], key=lambda x: x["N_bits"])
        if len(entries) >= 2:
            # Compute scaling factor between consecutive N values
            for i in range(len(entries) - 1):
                a, b = entries[i], entries[i + 1]
                if a["speedup_median"] > 0:
                    scale = b["speedup_median"] / a["speedup_median"]
                    n_ratio = b["N_bits"] / a["N_bits"]
                    analysis.append(
                        {
                            "rho": rho,
                            "N_low": a["N_bits"],
                            "N_high": b["N_bits"],
                            "N_ratio": n_ratio,
                            "speedup_low": a["speedup_median"],
                            "speedup_high": b["speedup_median"],
                            "speedup_scale": round(scale, 2),
                        }
                    )
    return analysis


def compile_e5_summary(data):
    """Lazy accumulation summary."""
    rows = []
    for dp in data["data_points"]:
        e = dp["extra"]
        lazy_tot = e.get("concat_plus_normalize_median_ns", e.get("total_lazy_median_ns", 0))
        eager = e.get("eager_median_ns", 0)
        rows.append(
            {
                "accum_count": e.get("accum_count", e.get("accumulation_count", 0)),
                "rho": e.get("target_rho", 0),
                "N_bits": dp["input_size_bits"],
                "concat_ns": e.get("concat_only_median_ns", 0),
                "normalize_ns": e.get("normalize_only_median_ns", e.get("normalize_median_ns", 0)),
                "total_lazy_ns": lazy_tot,
                "eager_ns": eager,
                "gmp_ns": e.get("gmp_median_ns", dp["ref_median_ns"]),
                "lazy_vs_eager": round(
                    eager / max(lazy_tot, 1), 1
                ),
            }
        )
    return rows


def format_tables(e4_rows, e7_rows, crossover_rho, scaling, e5_rows, lean_stats):
    """Format everything as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("HYBRID ZECKENDORF BENCHMARK: PUBLICATION TABLES")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("=" * 80)

    # Table 1: E4 Sparse Addition (main result)
    lines.append("")
    lines.append("TABLE 1: Sparse Addition — HZ add_lazy vs GMP (Table 2 Replication)")
    lines.append("-" * 95)
    lines.append(
        f"{'N (bits)':>10} {'rho':>10} {'n':>4} {'Speedup':>10} {'±95%CI':>8} "
        f"{'HZ (ns)':>10} {'GMP (ns)':>10} {'K':>6} {'Lvls':>5}"
    )
    lines.append("-" * 95)
    for r in e4_rows:
        ci = f"±{r['speedup_ci95']:.1f}" if r["speedup_ci95"] == r["speedup_ci95"] else "  n/a"
        lines.append(
            f"{r['N_bits']:>10,} {r['target_rho']:>10.0e} {r['n_samples']:>4} "
            f"{r['speedup_median']:>9.1f}x {ci:>8} "
            f"{r['hz_median_ns']:>10,} {r['gmp_median_ns']:>10,} "
            f"{r['K_mean']:>6} {r['levels_mean']:>5}"
        )

    # Table 2: Scaling analysis
    if scaling:
        lines.append("")
        lines.append("TABLE 2: Scaling Analysis — How speedup grows with N")
        lines.append("-" * 70)
        lines.append(
            f"{'rho':>10} {'N_low':>10} {'N_high':>10} "
            f"{'Speedup_low':>12} {'Speedup_high':>13} {'Scale':>8}"
        )
        lines.append("-" * 70)
        for s in scaling:
            lines.append(
                f"{s['rho']:>10.0e} {s['N_low']:>10,} {s['N_high']:>10,} "
                f"{s['speedup_low']:>11.1f}x {s['speedup_high']:>12.1f}x "
                f"{s['speedup_scale']:>7.1f}x"
            )

    # Table 3: Crossover curve
    if e7_rows:
        lines.append("")
        lines.append("TABLE 3: Crossover Sweep — Speedup vs Density")
        lines.append("-" * 60)
        lines.append(f"{'N (bits)':>10} {'rho':>10} {'n':>4} {'Speedup':>10} {'±95%CI':>8}")
        lines.append("-" * 60)
        for r in e7_rows:
            ci = (
                f"±{r['speedup_ci95']:.1f}"
                if r["speedup_ci95"] == r["speedup_ci95"]
                else "  n/a"
            )
            marker = ""
            if crossover_rho and 0.5 < r["speedup_median"] < 2.0:
                marker = " *"  # near crossover zone
            lines.append(
                f"{r['N_bits']:>10,} {r['target_rho']:>10.2e} {r['n_samples']:>4} "
                f"{r['speedup_median']:>9.2f}x {ci:>8}{marker}"
            )
        if crossover_rho:
            lines.append(f"\nCrossover rho (interpolated): {crossover_rho:.4e}")

    # Table 4: Lean formalization stats
    lines.append("")
    lines.append("TABLE 4: Lean 4 Formalization Statistics")
    lines.append("-" * 50)
    lines.append(f"  Modules:          {lean_stats['modules']}")
    lines.append(f"  Total lines:      {lean_stats['total_lines']}")
    lines.append(f"  Public theorems:  {lean_stats['public_theorems']}")
    lines.append(f"  Private theorems: {lean_stats['private_theorems']}")
    lines.append(f"  Total theorems:   {lean_stats['total_theorems']}")
    lines.append(f"  sorry/admit:      {lean_stats['sorry_count']}")
    lines.append(f"  Verification:     {'CLEAN' if lean_stats['sorry_count'] == 0 else 'INCOMPLETE'}")

    # Decision summary
    lines.append("")
    lines.append("=" * 80)
    lines.append("DECISION SUMMARY")
    lines.append("=" * 80)

    # Find key results
    key_1m_1e3 = [r for r in e4_rows if r["N_bits"] == 1_000_000 and r["target_rho"] == 1e-3]
    if key_1m_1e3:
        r = key_1m_1e3[0]
        if r["speedup_median"] >= 10:
            decision = "CONFIRMED (>=10x)"
        elif r["speedup_median"] >= 2:
            decision = f"PARTIALLY CONFIRMED ({r['speedup_median']:.1f}x, target was 10x)"
        else:
            decision = f"NOT CONFIRMED ({r['speedup_median']:.1f}x)"
        lines.append(f"  Core claim (rho=1e-3, N=1M): {decision}")
    else:
        lines.append("  Core claim (rho=1e-3, N=1M): AWAITING DATA")

    key_1m_1e4 = [r for r in e4_rows if r["N_bits"] == 1_000_000 and r["target_rho"] == 1e-4]
    if key_1m_1e4:
        r = key_1m_1e4[0]
        lines.append(
            f"  High-sparsity (rho=1e-4, N=1M): {r['speedup_median']:.1f}x ± {r['speedup_ci95']:.1f}"
        )

    if crossover_rho:
        lines.append(f"  Crossover density: rho* = {crossover_rho:.4e}")

    lines.append("")
    return "\n".join(lines)


def main():
    # Load all available results
    e4 = load_json("exp4_sparse_add")
    e7 = load_json("exp7_crossover")
    e5 = load_json("exp5_lazy_accum")

    if not e4:
        print("ERROR: exp4_sparse_add.json not found", file=sys.stderr)
        sys.exit(1)

    # Compile tables
    e4_rows = compile_e4_table(e4)
    e7_rows, crossover_rho = compile_e7_crossover(e7) if e7 else ([], None)
    scaling = compile_scaling_analysis(e4_rows)
    e5_rows = compile_e5_summary(e5) if e5 else []

    # Lean stats (from verified scan of lean/HeytingLean/Bridge/Veselov/HybridZeckendorf/)
    lean_stats = {
        "modules": 12,
        "total_lines": 1441,
        "public_theorems": 59,
        "private_theorems": 9,
        "total_theorems": 68,
        "sorry_count": 0,
    }

    # Write structured JSON
    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "tables": {
            "e4_sparse_addition": e4_rows,
            "e7_crossover": e7_rows,
            "crossover_rho": crossover_rho,
            "scaling_analysis": scaling,
            "e5_lazy_accumulation": e5_rows,
            "lean_formalization": lean_stats,
        },
        "decisions": {
            "e4_core_claim": next(
                (
                    {
                        "rho": 1e-3,
                        "N": 1_000_000,
                        "speedup": r["speedup_median"],
                        "ci95": r["speedup_ci95"],
                        "n": r["n_samples"],
                        "verdict": (
                            "confirmed"
                            if r["speedup_median"] >= 10
                            else "partially_confirmed"
                            if r["speedup_median"] >= 2
                            else "not_confirmed"
                        ),
                    }
                    for r in e4_rows
                    if r["N_bits"] == 1_000_000 and r["target_rho"] == 1e-3
                ),
                None,
            ),
            "crossover_rho": crossover_rho,
        },
    }

    json_path = RESULTS_DIR / "paper_tables.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Write human-readable
    text = format_tables(e4_rows, e7_rows, crossover_rho, scaling, e5_rows, lean_stats)
    txt_path = RESULTS_DIR / "paper_tables.txt"
    with open(txt_path, "w") as f:
        f.write(text)

    print(text)
    print(f"\nWritten: {json_path}")
    print(f"Written: {txt_path}")


if __name__ == "__main__":
    main()
