"""Independent Fraction oracle for rational KAN evaluation.

This file intentionally imports no HZ or Boundary runtime modules.
"""

from __future__ import annotations

from fractions import Fraction

from .exact_rkan import ActivationCoeffs, KANWeights, kan_eval, rational_activation


def rational_activation_oracle(
    x: Fraction,
    a: list[Fraction] | tuple[Fraction, Fraction, Fraction, Fraction],
    b: list[Fraction] | tuple[Fraction, Fraction, Fraction],
) -> Fraction:
    assert len(a) == 4 and len(b) == 3
    return rational_activation(x, ActivationCoeffs(a=tuple(a), b=tuple(b)))  # type: ignore[arg-type]


def kan_oracle(x: tuple[Fraction, Fraction], weights: KANWeights) -> Fraction:
    return kan_eval(weights, x)
