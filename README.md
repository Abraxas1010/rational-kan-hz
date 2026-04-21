<img src="assets/Apoth3osis.webp" alt="Apoth3osis — Formal Mathematics and Verified Software" width="140"/>

<sub><strong>Our tech stack is ontological:</strong><br>
<strong>Hardware — Physics</strong><br>
<strong>Software — Mathematics</strong><br><br>
<strong>Our engineering workflow is simple:</strong> discover, build, grow, learn & teach</sub>

---

<sub>
<strong>Acknowledgment</strong><br>
We humbly thank the collective intelligence of humanity for providing the technology and culture we cherish. We do our best to properly reference the authors of the works utilized herein, though we may occasionally fall short. Our formalization acts as a reciprocal validation—confirming the structural integrity of their original insights while securing the foundation upon which we build. In truth, all creative work is derivative; we stand on the shoulders of those who came before, and our contributions are simply the next link in an unbroken chain of human ingenuity.
</sub>

---

[![License: Apoth3osis License Stack v1](https://img.shields.io/badge/License-Apoth3osis%20License%20Stack%20v1-blue.svg)](LICENSE.md)

# Rational KAN HZ: Exact Rational Learning with Hybrid Zeckendorf Arithmetic

Rational KAN HZ is a reproducible research package for exact-rational Kolmogorov-Arnold Networks, Hybrid Zeckendorf sparse arithmetic, Lean 4 proof surfaces, Rust P-stack acceleration, and symbolic extraction to SymPy and LaTeX. It packages the code, formal foundations, benchmark artifacts, and audit trail for the Rational KAN / Boundary / Hybrid Zeckendorf line without mirroring the whole `heyting-imm` monorepo.

## Paper

The accompanying paper, **Exact-Rational Kolmogorov–Arnold Networks with Hybrid Zeckendorf Arithmetic** (Apoth3osis Labs; R. Goodman and V. Veselov), is built from sources in [`papers/rational_kan_hz/`](papers/rational_kan_hz/):

- PDF: [`papers/rational_kan_hz/rational_kan_hz_paper.pdf`](papers/rational_kan_hz/rational_kan_hz_paper.pdf)
- Source: [`papers/rational_kan_hz/rational_kan_hz_paper.tex`](papers/rational_kan_hz/rational_kan_hz_paper.tex)
- Convergence theorem: [`papers/rational_kan_hz/theorem_convergence.tex`](papers/rational_kan_hz/theorem_convergence.tex)

Rebuild from source (TeX Live 2023 or later):

```bash
cd papers/rational_kan_hz
pdflatex rational_kan_hz_paper.tex
pdflatex rational_kan_hz_paper.tex
```

## What Is This?

This repository isolates the exact surfaces needed to study three linked questions:

1. Can a rational KAN be trained and symbolically exported without losing exactness?
2. When does the Hybrid Zeckendorf P-stack preserve bit-identical semantics against a `fractions.Fraction` oracle?
3. Where does sparse-active large-dictionary training actually gain speed, and where does it not?

## Included Surfaces

- `src/rkan_hz/`: exact-rational RKAN implementation, symbolic extraction, paper-run orchestration, in-loop P-stack parity lane, and GPU scaled sparse-training lane.
- `tests/`: the project-local regression suite for parity, symbolic export, confluence, and paper-run artifacts.
- `bench/hybrid_zeckendorf/`: the Rust source for the real Hybrid Zeckendorf backend, including `pstack_exact_accumulate`.
- `lean/`: a standalone Lean proof subset containing the Hybrid Zeckendorf foundation, Boundary Zeckendorf canonicality surface, supporting sanity tests, and the Stern-Brocot idempotence hook used by the paper theorem.
- `artifacts/rational_kan_hz/`: the packaged benchmark and symbolic-export outputs used in the reports.
- `conjectures/` and `Blueprint/proof_trees/`: the research ledger for the RKAN/HZ line.

## Key Results

- Paper status tier: `PIPELINE-CERTIFIED-TRAINING-PARTIAL`
- Lean proof subset: `113` modules
- GPU scaled sparse-training speedup: `3.7258` with 95% CI `[3.6802, 3.7679]`
- GPU dense-to-P4 memory ratio: `0.41773`
- P-stack in-loop parity gate: `True`
- P-stack in-loop network-scale gate: `True`
- P-stack inference speed gate: `False` with mean `0.92862` and 95% CI `[0.77436, 1.0739]`
- Dictionary-scaling exponent alpha: `0.51908`

Scaling sweep:
- 10000: speedup 1.1654 [1.1602, 1.1697]
- 100000: speedup 1.1679 [1.1633, 1.1724]
- 1000000: speedup 3.3279 [3.2289, 3.3976]
- 10000000: speedup 44.174 [42.59, 45.415]

## Reproduce

Python and Rust:

```bash
python3 -m pip install -e .
cargo build --release --manifest-path bench/hybrid_zeckendorf/Cargo.toml --bin pstack_exact_accumulate
PYTHONPATH=src python3 -m pytest tests/test_pstack_in_loop.py tests/test_neural_artifact_export.py tests/test_symbolic_normalization.py tests/test_paper_run.py -q
python3 scripts/verify_results.py
```

Lean:

```bash
lake update
lake build
```

One-command verification:

```bash
./scripts/verify_all.sh
```

## Formalization Boundary

The Lean package formalizes the Hybrid Zeckendorf arithmetic and supporting proof hooks used by this research package. It does **not** claim that every empirical Python benchmark is fully mechanized in Lean. The honest split is:

- Lean: arithmetic identities, canonicality surfaces, sanity tests, and the Stern-Brocot projection lemma used by the convergence theorem.
- Python/Rust: executable training, symbolic extraction, benchmark orchestration, and empirical performance characterization.

See [docs/FORMALIZATION_SCOPE.md](docs/FORMALIZATION_SCOPE.md) and [docs/RESULTS_INDEX.md](docs/RESULTS_INDEX.md).

## Repository Metadata For Publishing

Suggested GitHub description:
`Exact rational KAN research package with Lean 4 proofs, Rust Hybrid Zeckendorf arithmetic, symbolic extraction, and reproducible benchmark artifacts.`

Suggested topics:
`rational-kan`, `lean4`, `formal-verification`, `symbolic-regression`, `zeckendorf`, `sparse-arithmetic`, `rust`, `sympy`, `pytorch`, `reproducible-research`

## Citation

See [CITATION.cff](CITATION.cff).

## License

[Apoth3osis License Stack v1](LICENSE.md)
