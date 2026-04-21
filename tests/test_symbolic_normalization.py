from pathlib import Path

from rkan_hz.exact_rkan import constructive_trained_weights
from rkan_hz.rkan_symbolic_extract import normalize_weights


def test_symbolic_expression_contains_inputs():
    expr, guards = normalize_weights(constructive_trained_weights())
    assert "x_0" in str(expr)
    assert "x_1" in str(expr)
    assert guards
