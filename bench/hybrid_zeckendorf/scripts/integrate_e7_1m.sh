#!/usr/bin/env bash
# Quick integration of E7 1M data after the run completes.
# Usage: bash scripts/integrate_e7_1m.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS=results
E7_MAIN="$RESULTS/exp7_crossover.json"
E7_100K="$RESULTS/exp7_crossover_100k.json"

echo "=== E7 1M Integration ==="

# 1. Verify the file has changed
MAIN_SIZE=$(stat -c %s "$E7_MAIN")
BACKUP_SIZE=$(stat -c %s "$E7_100K")
if [ "$MAIN_SIZE" = "$BACKUP_SIZE" ]; then
    echo "ERROR: E7 main file is identical to 100k backup. Run may not be complete."
    exit 1
fi
echo "Main: $MAIN_SIZE bytes, Backup: $BACKUP_SIZE bytes — data differs, proceeding."

# 2. Back up 1M-only data
cp "$E7_MAIN" "$RESULTS/exp7_crossover_1m.json"
echo "Backed up 1M-only data to exp7_crossover_1m.json"

# 3. Merge
python3 scripts/merge_e7_runs.py \
    "$E7_100K" "$RESULTS/exp7_crossover_1m.json" \
    -o "$RESULTS/exp7_crossover_merged.json"
cp "$RESULTS/exp7_crossover_merged.json" "$E7_MAIN"
echo "Merged 100k + 1M data"

# 4. Show 1M crossover curve
echo ""
echo "--- 1M Crossover Curve ---"
python3 -c "
import json, statistics
from collections import defaultdict

d = json.load(open('$E7_MAIN'))
by_config = defaultdict(list)
for dp in d['data_points']:
    if dp['input_size_bits'] == 1_000_000:
        e = dp['extra']
        rho = e.get('target_rho', 0)
        by_config[round(rho, 8)].append(dp['speedup_ratio'])

print(f'{\"rho\":>12s}  {\"n\":>4s}  {\"Speedup\":>10s}  {\"±CI95\":>8s}')
for rho in sorted(by_config.keys()):
    sp = by_config[rho]
    med = statistics.median(sp)
    ci = 1.96 * statistics.stdev(sp) / len(sp)**0.5 if len(sp) >= 3 else float('nan')
    marker = ' *' if 0.5 < med < 2.0 else ''
    print(f'{rho:>12.2e}  {len(sp):>4d}  {med:>9.2f}x  ±{ci:.1f}{marker}')

# Find crossover
import math
rows = []
for rho in sorted(by_config.keys()):
    sp = by_config[rho]
    rows.append((rho, statistics.median(sp)))

for i in range(len(rows)-1):
    rho_a, sp_a = rows[i]
    rho_b, sp_b = rows[i+1]
    if sp_a >= 1.0 and sp_b < 1.0:
        log_rho_a = math.log10(rho_a)
        log_rho_b = math.log10(rho_b)
        frac = (sp_a - 1.0) / (sp_a - sp_b)
        crossover = 10 ** (log_rho_a + frac * (log_rho_b - log_rho_a))
        print(f'\nCrossover rho at 1M bits (interpolated): {crossover:.4e}')
        break
else:
    # Check if HZ is faster at ALL rho values
    if all(sp > 1.0 for _, sp in rows):
        print('\nNo crossover: HZ faster at all tested rho values')
    elif all(sp < 1.0 for _, sp in rows):
        print('\nNo crossover: GMP faster at all tested rho values')
    else:
        print('\nCrossover not cleanly detected')
"

# 5. Recompile paper tables
echo ""
echo "--- Recompiling paper tables ---"
python3 scripts/compile_paper_tables.py

echo ""
echo "=== Integration complete ==="
