from fractions import Fraction

from rkan_hz.rkan_neural_artifact_export import (
    _basis,
    _mean_square_error,
    _normal_equations,
    _solve_fraction_linear_system,
    train_exact_rkan,
)


def test_exact_rkan_neural_export_learns_and_converges():
    learned = train_exact_rkan()
    assert learned.initial_mse > 0
    assert learned.final_training_mse == 0
    assert learned.final_validation_mse == 0
    assert learned.coefficients == (Fraction(1), Fraction(1), Fraction(-1, 6))


def test_training_labels_are_not_ignored():
    xs = [
        (Fraction(-1), Fraction(-1)),
        (Fraction(1), Fraction(0)),
        (Fraction(2), Fraction(1)),
        (Fraction(0), Fraction(2)),
    ]
    gram, rhs = _normal_equations(xs)
    baseline = tuple(_solve_fraction_linear_system(gram, rhs))
    perturbed_rhs = list(rhs)
    perturbed_rhs[0] += Fraction(1)
    perturbed = tuple(_solve_fraction_linear_system(gram, perturbed_rhs))
    assert perturbed != baseline
    assert _mean_square_error(perturbed, xs) != 0


def test_basis_uses_both_inputs():
    assert _basis((Fraction(2), Fraction(-3))) == (Fraction(4), Fraction(-3), Fraction(-27))
