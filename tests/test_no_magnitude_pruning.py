from fractions import Fraction

from rkan_hz.rkan_boundary_infer import HZRationalPy


def test_tiny_rational_survives_canonicalization():
    tiny = HZRationalPy.from_fraction(Fraction(1, 2**30))
    assert tiny.to_fraction() == Fraction(1, 2**30)
