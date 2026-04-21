# Rational KAN HZ Paper Run Report

# PIPELINE-CERTIFIED-TRAINING-PARTIAL

## Audit Pass

This report separates three orthogonal claims that were conflated in the prior draft:

1. **Pipeline correctness** (T1.3/T1.4, T1.6 `rkan_p4_sbr_oracle`): the symbolic
   extraction pipeline preserves an oracle rational configuration byte-exactly
   (canonicalization, simplifier, LaTeX emit, LaTeX roundtrip). This is a *pipeline
   identity* witness and is computed by reconstructing Sigma c_i * b_i from the
   spec's oracle coefficients. No training occurs in this layer.
2. **Training convergence** (T1.6 `rkan_p4_sbr_trained`): a real training run
   converges to the oracle rational configuration. This row is only populated
   where a real CPU/GPU sweep has been executed; evidence is joined from
   target-specific artifacts under `artifacts/rational_kan_hz/paper_run/trained/`
   plus the legacy `artifacts/rational_kan_hz/gpu_scaled_training_iter_1/` t1 lane.
3. **Scaling** (T1.5): real GPU dense-vs-P4 training-time measurements across
   10K/100K/1M/10M dictionary features, bootstrap CIs, alpha-fit.

The prior draft reported all three as a single 'SUBMISSION-READY' banner based on
pipeline-identity evidence alone. The remediation separates the claims, labels
oracle vs trained evidence, and makes the confluence residual computation
structural (independent callables for both reducts) rather than by assertion.

## T1.1 Confluence

critical_pairs_unclosed: 0
closure_verdict: symbolic_residuals_closed
closure_is_structural: True

## T1.2 Theorem Hook

Lean lemma: `HeytingLean.SternBrocot.stern_brocot_idempotent_on_bounded`.
Paper proof: `papers/rational_kan_hz/theorem_convergence.tex`.

## T1.3/T1.4 Targets (pipeline correctness, oracle mode)

- t1: oracle_reconstruction_exact residual=0 threshold=1e-12 mode=oracle_reconstruction training_executed=False pipeline_passed=True
- t2: oracle_reconstruction_exact residual=0 threshold=1e-12 mode=oracle_reconstruction training_executed=False pipeline_passed=True
- t3: oracle_reconstruction_exact residual=0 threshold=1e-12 mode=oracle_reconstruction training_executed=False pipeline_passed=True
- t4: oracle_reconstruction_exact residual=0 threshold=1e-12 mode=oracle_reconstruction training_executed=False pipeline_passed=True
- t5: oracle_reconstruction_exact residual=0 threshold=1e-08 mode=oracle_reconstruction training_executed=False pipeline_passed=True
- t6: best_polynomial_approximant residual=0 threshold=1e-06 mode=oracle_reconstruction training_executed=False pipeline_passed=True
- t7: residual_reported residual=0.0516 threshold=1e-06 mode=oracle_reconstruction training_executed=False boundary_recorded=True
- t8: residual_reported residual=0.25 threshold=1e-06 mode=oracle_reconstruction training_executed=False boundary_recorded=True

## T1.5 Scaling (real GPU measurements)

alpha: 0.519081 se=0.178064 passed=True
- 10000: speedup 1.16543 [1.16019, 1.16969]
- 100000: speedup 1.1679 [1.16333, 1.17235]
- 1000000: speedup 3.32786 [3.22891, 3.39761]
- 10000000: speedup 44.1743 [42.5903, 45.4147]

## T1.6 Baselines

blocked_baseline_rows: 0 (excluding rkan_p4_sbr_trained absent rows)
Execution, LaTeX presentation, LaTeX roundtrip, and symbolic faithfulness are separate gates.
rkan_p4_sbr_oracle reports pipeline-identity; rkan_p4_sbr_trained reports real SGD training where available.
- mlp_pysr: executed 6/6, latex_presentation 6/6, latex_roundtrip 3/6, symbolic_faithfulness 0/6
- pykan: executed 6/6, latex_presentation 0/6, latex_roundtrip 0/6, symbolic_faithfulness 0/6
- rkan_dense: executed 6/6, latex_presentation 0/6, latex_roundtrip 0/6, symbolic_faithfulness 0/6
- rkan_p4_sbr_oracle: executed 6/6, latex_presentation 6/6, latex_roundtrip 6/6, symbolic_faithfulness 6/6
- rkan_p4_sbr_trained: executed 6/6, latex_presentation 6/6, latex_roundtrip 6/6, symbolic_faithfulness 6/6

Training-convergence rows executed: t1, t2, t3, t4, t5, t6
Training-convergence rows pending: (none)

## Paper Submission Gating

- pipeline_gate: True
- scaling_gate: True
- training_gate_full (all targets have real trained evidence): True
- pstack_in_loop_parity_scale_gate: True (conjecture rational_kan_hz_pstack_in_loop_20260420 resolved genuine 2026-04-20)
- pstack_in_loop_speed_regime_gate: False (conjecture rational_kan_hz_pstack_speed_regime_20260420 partial - CI95 lower below 1.0 on both inference and update lanes; see pstack_in_loop/ diagnostics)
- status_tier: PIPELINE-CERTIFIED-TRAINING-PARTIAL
- tier_reason: speed-regime open; parity+scale closed; paper may cite bit-identical parity as a closed result and reference the speed regime as ongoing.

## Audit

commit: 6664a35489
host_cpu: aarch64
platform: Linux-6.14.0-1015-nvidia-aarch64-with-glibc2.39
thermal_state_start: {"available": true, "zones": [{"millidegrees_c": 73300, "zone": "thermal_zone0"}, {"millidegrees_c": 54600, "zone": "thermal_zone1"}, {"millidegrees_c": 57500, "zone": "thermal_zone2"}, {"millidegrees_c": 56000, "zone": "thermal_zone3"}, {"millidegrees_c": 73300, "zone": "thermal_zone4"}, {"millidegrees_c": 54800, "zone": "thermal_zone5"}, {"millidegrees_c": 56700, "zone": "thermal_zone6"}]}
