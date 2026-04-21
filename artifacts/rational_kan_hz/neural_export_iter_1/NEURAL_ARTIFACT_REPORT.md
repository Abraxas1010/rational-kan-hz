# Rational KAN HZ Neural Artifact Export

Status: PROMOTED
Commit: `c8967472d5`

## Neural Learning Gate

The network is a rational KAN-shaped Boundary object with exact rational payloads.
It trains from data by exact normal equations over Boundary-compatible basis neurons.
The exported expression is produced only after the learned weights pass exact convergence checks.

| metric | value |
|---|---:|
| training rows | 25 |
| validation rows | 200 |
| initial training MSE | `653/90` |
| final training MSE | `0/1` |
| final validation MSE | `0/1` |

## Artifact Boundary

P4 lazy HZ/Veselov arithmetic remains the hot-path update/readout acceleration.
SymPy, LaTeX, and PDF outputs are final materialization artifacts, not runtime dependencies.

## Evidence

- `learned_weights.json`
- `learning_trace.jsonl`
- `learning_trace_replay.json`
- `convergence.json`
- `extracted_expression.sympy`
- `extracted_expression.tex`
- `extracted_expression.pdf`
- `faithfulness_results.json`
- `lipschitz_certificate.json`
- `manifest.json`
