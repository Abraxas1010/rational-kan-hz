from fractions import Fraction


def test_ad_mul_rule_symbolic():
    x = Fraction(1, 3)
    y = Fraction(5, 7)
    upstream = Fraction(1)
    grad_x = upstream * y
    grad_y = upstream * x
    assert grad_x == Fraction(5, 7)
    assert grad_y == Fraction(1, 3)


def test_ad_div_rule_symbolic():
    x = Fraction(2, 5)
    y = Fraction(3, 7)
    upstream = Fraction(1)
    assert upstream / y == Fraction(7, 3)
    assert -upstream * x / (y * y) == Fraction(-98, 45)
