"""Shared rule-table loader and fingerprint checker.

The rule table is vendored alongside this module so the loader works in both
the standalone public repo and the monorepo layout. The monorepo path remains
a secondary fallback so historical callers continue to work.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent

_VENDORED_RULES = _PACKAGE_DIR / "rules_table.toml"
_MONOREPO_RULES = Path("projects/boundary_lang/runtime/src/rkan/rules_table.toml")


def _resolve_default() -> Path:
    if _VENDORED_RULES.is_file():
        return _VENDORED_RULES
    return _MONOREPO_RULES


DEFAULT_RULES = _resolve_default()


def rules_fingerprint(path: Path = DEFAULT_RULES) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rules_table(path: Path = DEFAULT_RULES) -> dict:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    fingerprint_path = path.with_suffix(".fingerprint")
    if fingerprint_path.exists():
        expected = fingerprint_path.read_text(encoding="utf-8").strip()
        observed = rules_fingerprint(path)
        if expected != observed:
            raise ValueError(f"rule fingerprint mismatch: {observed} != {expected}")
    return data
