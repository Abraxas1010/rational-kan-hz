#!/usr/bin/env python3
"""
Merge multiple exp7_crossover result files into one combined file.

Usage:
    python3 scripts/merge_e7_runs.py results/exp7_crossover_100k.json results/exp7_crossover.json -o results/exp7_crossover_merged.json
"""

import argparse
import json
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description="Merge E7 result files")
    parser.add_argument("files", nargs="+", help="Input JSON files to merge")
    parser.add_argument("-o", "--output", required=True, help="Output merged file")
    args = parser.parse_args()

    all_data_points = []
    config = None

    for path in args.files:
        with open(path) as f:
            data = json.load(f)
        all_data_points.extend(data["data_points"])
        if config is None:
            config = data.get("config", {})

    # Deduplicate by key
    seen = set()
    unique_points = []
    for dp in all_data_points:
        e = dp["extra"]
        key = (
            dp["input_size_bits"],
            round(e.get("target_rho", e.get("measured_rho", 0)), 10),
            e.get("trial_idx", id(dp)),
            dp["hz_median_ns"],
        )
        if key not in seen:
            seen.add(key)
            unique_points.append(dp)

    merged = {
        "config": config,
        "data_points": sorted(
            unique_points,
            key=lambda dp: (
                dp["input_size_bits"],
                dp["extra"].get("target_rho", dp["extra"].get("measured_rho", 0)),
            ),
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_info": all_data_points[0].get("machine_info") if all_data_points else {},
        "merge_info": {
            "sources": args.files,
            "total_before_dedup": len(all_data_points),
            "total_after_dedup": len(unique_points),
        },
    }

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(args.files)} files → {len(unique_points)} data points → {args.output}")


if __name__ == "__main__":
    main()
