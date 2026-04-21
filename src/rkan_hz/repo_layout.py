"""Repository layout helpers for RKAN/HZ package portability.

The RKAN/HZ code runs in two layouts:

1. Inside the main `heyting-imm` monorepo under `projects/rational_kan_hz/`
2. Inside the exported standalone public repo where `src/rkan_hz/` lives at the root

Hard-coding `Path(__file__).parents[n]` silently breaks one of those layouts.
This module resolves the repository root by searching upward for the real
project markers instead.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the RKAN/HZ repository root in either supported layout."""

    origin = (start or Path(__file__)).resolve()
    candidates = [origin] + list(origin.parents)
    for candidate in candidates:
        has_rust_backend = (candidate / "bench" / "hybrid_zeckendorf" / "Cargo.toml").is_file()
        has_standalone_python = (candidate / "src" / "rkan_hz").is_dir()
        has_monorepo_python = (candidate / "projects" / "rational_kan_hz" / "src" / "rkan_hz").is_dir()
        if has_rust_backend and (has_standalone_python or has_monorepo_python):
            return candidate
    raise FileNotFoundError(f"could not locate RKAN/HZ repository root from {origin}")
