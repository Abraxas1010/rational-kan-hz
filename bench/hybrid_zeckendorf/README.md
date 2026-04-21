# Hybrid Zeckendorf Benchmarks

This workspace implements seven computational experiments:
- `exp1_modexp`: modular exponentiation (`a^e mod n`) vs GMP
- `exp2_polymul`: schoolbook polynomial multiplication over giant coefficients vs GMP
- `exp3_density`: density scaling verification against the proved active-level bound
- `exp4_sparse_add`: sparse-regime addition sweep (Table-2 style replication)
- `exp5_lazy_accum`: eager vs lazy accumulation (`add` vs `add_lazy`)
- `exp6_sparse_modexp`: modular exponentiation with sparse HZ exponents
- `exp7_crossover`: density crossover sweep (speedup vs rho)

## Build

```bash
cargo build --release
```

## Smoke Runs

These are intentionally reduced inputs for quick validation; they do not match
the `--full` defaults.

```bash
cargo run --release --bin exp1_modexp -- --bit-sizes 512 --repeats 5
cargo run --release --bin exp2_polymul -- --degree 10 --coeff-digits 1000 --repeats 3
cargo run --release --bin exp3_density -- --digit-counts 3,5,10 --samples-per-size 5
cargo run --release --bin exp4_sparse_add -- --bit-sizes 100000 --rhos 1e-3,1e-2,1e-1 --samples 2 --repeats 2
cargo run --release --bin exp5_lazy_accum -- --counts 10 --rhos 1e-3 --bits 10000 --repeats 3
cargo run --release --bin exp6_sparse_modexp -- --bit-sizes 10000 --rhos 1e-3 --samples 3 --repeats 3
cargo run --release --bin exp7_crossover -- --bits 100000 --rho-steps 10 --rho-min-exp=-4 --rho-max-exp=-2 --trials 2 --repeats 2
```

## Full Runs

```bash
cargo run --release --bin exp1_modexp -- --full
cargo run --release --bin exp2_polymul -- --full
cargo run --release --bin exp3_density -- --full
cargo run --release --bin exp4_sparse_add -- --full
cargo run --release --bin exp5_lazy_accum -- --full
cargo run --release --bin exp6_sparse_modexp -- --full
cargo run --release --bin exp7_crossover -- --full
python3 scripts/generate_summary.py
```

Results are written to `results/*.json`.

## Notes on Formal Correspondence

- Lean formalization proves normalization properties on already-structured hybrid numbers.
- This benchmark additionally needs `from_integer`, so Rust seeds level `0` then runs inter-level carry with radix `weight(1)=1000` at level `0` to construct multi-level structure.
- Rust inter-level carry runs bottom-up to a fixed point (convergence), while Lean's proof artifact is expressed as top-down recursion; both preserve evaluation, but the implementation order differs.

## Current Scientific Outcome

Dense-regime artifacts (`exp1`-`exp3`) indicate a negative performance result relative to GMP at tested scales:

- `exp1_modexp`: HZ slower than GMP
- `exp2_polymul`: HZ much slower than GMP
- `exp3_density`: active-level bound holds, but random-integer density stays roughly flat (does not show the expected drop toward very small values)

Sparse-regime experiments (`exp4`-`exp7`) are added to directly test the claimed low-density regime. They are intentionally separated because runtime grows sharply at million-bit scale and high rho sweeps.

For `exp6_sparse_modexp`, each active HZ level currently delegates factor
exponentiation to GMP `pow_mod`; this is correctness-preserving but can erase
any theoretical sparse-level advantage and should be interpreted as a
conservative negative-result baseline.

This benchmark should be read as an honest overhead/feasibility study, not as demonstrated HZ speedup evidence.
