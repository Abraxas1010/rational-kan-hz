from fractions import Fraction

from rkan_hz.rkan_p4_training_equivalence import (
    DenseNormalEquationAccumulator,
    P4LazyNormalEquationAccumulator,
    run_pair,
    summarize,
    training_inputs_for_seed,
    _accumulate_rows,
)


def test_dense_and_p4_lazy_accumulators_read_out_same_normal_equations():
    rows = training_inputs_for_seed(123)
    dense = DenseNormalEquationAccumulator()
    lazy = P4LazyNormalEquationAccumulator()
    _accumulate_rows(dense, rows)
    _accumulate_rows(lazy, rows)
    dense_gram, dense_rhs, _dense_support, _dense_witness = dense.readout()
    lazy_gram, lazy_rhs, support, witness = lazy.readout()
    assert lazy_gram == dense_gram
    assert lazy_rhs == dense_rhs
    assert support > 0
    assert witness > 0


def test_dense_and_p4_lazy_training_produce_identical_converged_weights():
    dense, lazy = run_pair(20260420, 0)
    assert dense.weights_hash == lazy.weights_hash
    assert dense.coefficients == (Fraction(1), Fraction(1), Fraction(-1, 6))
    assert lazy.coefficients == (Fraction(1), Fraction(1), Fraction(-1, 6))
    assert dense.final_training_mse == 0
    assert lazy.final_training_mse == 0
    assert dense.final_validation_mse == 0
    assert lazy.final_validation_mse == 0


def test_training_equivalence_summary_reports_ci_for_every_speedup():
    pairs = [run_pair(20260420 + i, i) for i in range(3)]
    summary = summarize(pairs)
    assert summary["identity_passed"]
    for key in ("dense_over_p4_update", "dense_over_p4_readout", "dense_over_p4_total"):
        assert "mean" in summary[key]
        assert "ci95_lower" in summary[key]
        assert "ci95_upper" in summary[key]
