import json

import sympy as sp

from rkan_hz.rkan_symbolic_extract import confluence_certificate


def test_outer_eval_sum_accumulate_overlap_has_symbolic_zero_residual(tmp_path):
    out = tmp_path / "confluence_certificate.json"
    cert = confluence_certificate(out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == cert
    assert cert["critical_pairs_unclosed"] == 0
    pairs = cert["overlapping_pairs_detail"]
    assert pairs
    target = next(
        pair
        for pair in pairs
        if {pair["rule_left"], pair["rule_right"]} == {"outer_eval", "sum_accumulate"}
    )
    assert target["payload_algebra"] == "Q[c] with HZRConst payload c"
    assert sp.simplify(target["residual"]) == 0
    assert target["derivation_steps"]
    # Audit-hardening: the closure must be structural (reducts computed by
    # independent callables, with an independent fixture cross-check).
    assert target.get("closure_is_structural") is True
    assert target["closure_reducts_computed_by"]
    assert sp.simplify(target["fixture_residual"]) == 0
