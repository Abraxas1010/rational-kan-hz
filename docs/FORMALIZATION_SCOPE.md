# Formalization Scope

This repository vendors a narrow Lean subset from `heyting-imm` rather than the whole monorepo.

## Included

- Hybrid Zeckendorf foundation under `lean/HeytingLean/Bridge/Veselov/HybridZeckendorf/`
- Boundary Zeckendorf canonicality surface under `lean/HeytingLean/Boundary/Homomorphic/`
- supporting local dependencies pulled by the import closure
- the Stern-Brocot idempotence hook at `lean/HeytingLean/SternBrocot/Idempotence.lean`
- Veselov sanity tests under `lean/HeytingLean/Tests/Bridge/Veselov/`

Total vendored Lean modules under `lean/HeytingLean/`: `113`.
Including the top-level `lean/HeytingLean.lean` umbrella, the total number of Lean source files is `114`.

## Not Included

- unrelated HeytingLean subsystems
- CI, MCP tooling, dashboards, ATP orchestration, or unrelated boundary/frontier lanes
- a claim that the empirical Python or CUDA benchmark results are themselves fully formalized in Lean

## Honest Claim Boundary

The formal layer in this repository supports the algebraic and proof-theoretic substrate:

- Hybrid Zeckendorf arithmetic structure
- Zeckendorf canonicality and supporting boundary lemmas
- the Stern-Brocot fixed-point lemma used by the paper theorem

The executable layer supports:

- exact-rational RKAN training and symbolic extraction
- Rust P-stack execution
- empirical speed/memory benchmarking

That split is deliberate. This package is meant to be reproducible and reviewable, not cosmetically overclaimed.
