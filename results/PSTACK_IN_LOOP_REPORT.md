# P-stack In-Loop Report

- conjecture_statement=An exact-rational KAN with degree at least 8 and hidden width at least 10 can run the forward/backward/training loop through the Hybrid Zeckendorf P-stack substrate with bit-identical Fraction semantics at network scale.
- conjecture_id=rational_kan_hz_pstack_in_loop_20260420 (status=proved)
- split_speed_clause_into=rational_kan_hz_pstack_speed_regime_20260420 (status=partial)
- parity_gate=True
- scale_gate=True
- speedup_gate=False (tracked under rational_kan_hz_pstack_speed_regime_20260420)

## Clause Mapping

- exact_fraction_semantics: True
- network_scale_requirement: True
- inference_speed_advantage_at_network_scale: False (split out 2026-04-20)
- scope_note=Speed clause retired from rational_kan_hz_pstack_in_loop_20260420 on 2026-04-20 and carried into rational_kan_hz_pstack_speed_regime_20260420. The update-lane, inference-lane, microbatch sweep, and width sweep below are the evidence floor for the new conjecture; they remain co-located here because they are jointly diagnostic of the same substrate.

## Network

- degree=8
- hidden=10
- input_dim=4
- param_count=371
- active_hidden_units=6

## Update Lane

- benchmark_kind=amortized_training_update
- model_state=seeded_pretraining
- dense_backend=python_fraction_immediate_updates
- pstack_backend=rust_hz_factored_lazy_updates
- cpu_mean_speedup=0.07125798346957654
- cpu_ci95=[0.07012355597670272, 0.07240530049136648]
- microbatches=32
- mean_distinct_denominators=301.2
- mean_witness_bytes=699700.2

## Microbatch Sweep

- grid=[1, 4, 8, 16, 32, 64, 128]
- crossover_microbatches=None
- any_speedup_gate=False
- m=1: mean=0.02137005353529726, ci95=[0.02040663009668664, 0.022575237580946966], gate=False
- m=4: mean=0.038124547654295106, ci95=[0.036226706954909385, 0.039923262075082656], gate=False
- m=8: mean=0.0510986856540316, ci95=[0.05053264204471029, 0.05167301210452584], gate=False
- m=16: mean=0.06233830470615519, ci95=[0.06121058178757806, 0.06336694919627367], gate=False
- m=32: mean=0.07068113664001643, ci95=[0.0700256760716291, 0.07149026754986429], gate=False
- m=64: mean=0.07639167111145458, ci95=[0.07547153208318559, 0.07727393093241827], gate=False
- m=128: mean=0.0805408699556079, ci95=[0.07902096074864028, 0.08198795335162672], gate=False

## Inference Lane

- benchmark_kind=network_inference
- model_state=seeded_pretraining
- dense_backend=python_fraction
- pstack_backend=rust_hz_cli
- cpu_mean_speedup=0.9286179785770601
- cpu_ci95=[0.7743630172353385, 1.0739289613602867]
- batch_size=16
- cuda_status=not_requested

## Diagnostic Trail

- Architectural correction: denominator-factored lazy accumulation replaced the old LCM-collapse lane for update benchmarking.
- bridge_overhead_breakdown=artifacts/rational_kan_hz/paper_run/pstack_in_loop/bridge_overhead_breakdown.json
- bridge_mean_roundtrip_share=0.9928457154899806
- bridge_mean_dense_ns=23363.3
- bridge_mean_total_pstack_ns=6468431.6
- width_sweep=artifacts/rational_kan_hz/paper_run/pstack_in_loop/width_sweep.json
- width=10: mean_speedup=0.0027965195367254253
- width=50: mean_speedup=0.0006703360618863823
- width=200: mean_speedup=0.0001782598474998838
- width=500: mean_speedup=6.383660311444385e-05
- inference_samples30=artifacts/rational_kan_hz/paper_run/pstack_in_loop/inference_speedup_samples30.json
- inference_samples30_mean=0.007637970892587564, ci95=[0.0038295785527365394, 0.011688534450651157], samples=30
