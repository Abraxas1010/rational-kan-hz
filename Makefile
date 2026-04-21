.PHONY: verify verify-python verify-rust verify-lean verify-results

# Prefer the project-local venv if one exists; otherwise fall back to system python3.
PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

verify: verify-rust verify-python verify-lean verify-results

verify-python:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -q

verify-rust:
	cargo build --release --manifest-path bench/hybrid_zeckendorf/Cargo.toml --bin pstack_exact_accumulate

verify-lean:
	lake build

verify-results:
	$(PYTHON) scripts/verify_results.py
