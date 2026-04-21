#!/usr/bin/env bash
# Post-run pipeline: merge E4 runs, recompile paper tables, show key results.
# Usage: bash scripts/post_run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS=results

echo "=== Post-Run Pipeline ==="
echo ""

# 1. Check which result files exist
echo "--- Available result files ---"
ls -lh "$RESULTS"/exp*.json 2>/dev/null || echo "No result files found!"
echo ""

# 2. Check if 1M E4 data exists (file updated since the 100k backup was made)
E4_MAIN="$RESULTS/exp4_sparse_add.json"
E4_100K="$RESULTS/exp4_sparse_add_100k.json"

if [ -f "$E4_MAIN" ] && [ -f "$E4_100K" ]; then
    MAIN_SIZE=$(stat -c %s "$E4_MAIN")
    BACKUP_SIZE=$(stat -c %s "$E4_100K")
    if [ "$MAIN_SIZE" != "$BACKUP_SIZE" ]; then
        echo "--- E4: 1M data detected (main file differs from 100k backup) ---"
        echo "  Main: $MAIN_SIZE bytes  Backup: $BACKUP_SIZE bytes"

        # Back up the 1M-only data
        cp "$E4_MAIN" "$RESULTS/exp4_sparse_add_1m.json"

        # Merge 100k + 1M
        echo "  Merging 100k + 1M E4 data..."
        python3 scripts/merge_e4_runs.py \
            "$E4_100K" "$E4_MAIN" \
            -o "$RESULTS/exp4_sparse_add_merged.json"

        # Use merged as the main file
        cp "$RESULTS/exp4_sparse_add_merged.json" "$E4_MAIN"
        echo "  Merged file written to $E4_MAIN"
    else
        echo "--- E4: No new data (main and backup are identical) ---"
    fi
fi
# 2b. Check if 1M E7 data exists
E7_MAIN="$RESULTS/exp7_crossover.json"
E7_100K="$RESULTS/exp7_crossover_100k.json"

if [ -f "$E7_MAIN" ] && [ -f "$E7_100K" ]; then
    MAIN_SIZE=$(stat -c %s "$E7_MAIN")
    BACKUP_SIZE=$(stat -c %s "$E7_100K")
    if [ "$MAIN_SIZE" != "$BACKUP_SIZE" ]; then
        echo "--- E7: 1M data detected (main file differs from 100k backup) ---"
        echo "  Main: $MAIN_SIZE bytes  Backup: $BACKUP_SIZE bytes"

        cp "$E7_MAIN" "$RESULTS/exp7_crossover_1m.json"
        echo "  Merging 100k + 1M E7 data..."
        python3 scripts/merge_e7_runs.py \
            "$E7_100K" "$E7_MAIN" \
            -o "$RESULTS/exp7_crossover_merged.json"
        cp "$RESULTS/exp7_crossover_merged.json" "$E7_MAIN"
        echo "  Merged file written to $E7_MAIN"
    else
        echo "--- E7: No new data (main and backup are identical) ---"
    fi
fi
echo ""

# 3. Quick analysis of E4 data
echo "--- E4 Summary ---"
python3 -c "
import json, statistics
d = json.load(open('$E4_MAIN'))
from collections import defaultdict
by_config = defaultdict(list)
for dp in d['data_points']:
    key = (dp['input_size_bits'], dp['extra']['target_rho'])
    by_config[key].append(dp['speedup_ratio'])
print(f'Total data points: {len(d[\"data_points\"])}')
print(f'{\"Config\":>25s}  {\"N\":>5s}  {\"Median\":>8s}  {\"±CI95\":>8s}')
for k in sorted(by_config.keys()):
    sp = by_config[k]
    med = statistics.median(sp)
    ci = 1.96 * statistics.stdev(sp) / len(sp)**0.5 if len(sp) >= 3 else float('nan')
    print(f'  N={k[0]:>8d} rho={k[1]:.0e}  {len(sp):>5d}  {med:>7.1f}x  ±{ci:.1f}')
"
echo ""

# 4. Compile paper tables
echo "--- Compiling paper tables ---"
python3 scripts/compile_paper_tables.py
echo ""

echo "=== Pipeline complete ==="
