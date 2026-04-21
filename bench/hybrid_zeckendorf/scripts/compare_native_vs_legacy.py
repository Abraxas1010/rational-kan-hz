#!/usr/bin/env python3
"""Render a compact markdown comparison from dual-path HZ benchmark JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_rows(path: Path):
    data = json.loads(path.read_text())
    rows = []
    for dp in data.get("data_points", []):
        extra = dp.get("extra", {})
        rows.append(
            {
                "input": dp.get("input_description", ""),
                "legacy_ns": extra.get("hz_lazy_median_ns", dp.get("hz_median_ns")),
                "native_ns": extra.get("hz_native_lazy_median_ns", dp.get("native_median_ns")),
                "gmp_ns": dp.get("ref_median_ns"),
                "legacy_speedup": extra.get("hz_lazy_speedup_ratio", dp.get("speedup_ratio")),
                "native_speedup": extra.get("hz_native_lazy_speedup_ratio", dp.get("native_speedup_ratio")),
            }
        )
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: compare_native_vs_legacy.py <result.json>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    rows = load_rows(path)
    print("| Input | Legacy median ns | Native median ns | GMP median ns | Legacy speedup | Native speedup |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows[:20]:
        print(
            f"| {row['input']} | {row['legacy_ns']} | {row['native_ns']} | {row['gmp_ns']} | "
            f"{row['legacy_speedup']:.3f} | {row['native_speedup']:.3f} |"
        )
    if len(rows) > 20:
        print(f"\n_Truncated to first 20 rows out of {len(rows)}._")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
