"""Symbolic payload algebra for Phase 4."""

from __future__ import annotations

import sympy as sp


class SymbolicPayload:
    def __init__(self) -> None:
        self.guards: list[sp.Expr] = []

    def zero(self):
        return sp.Integer(0)

    def one(self):
        return sp.Integer(1)

    def add(self, a, b):
        return sp.simplify(a + b)

    def sub(self, a, b):
        return sp.simplify(a - b)

    def mul(self, a, b):
        return sp.simplify(a * b)

    def div(self, a, b):
        self.guards.append(b)
        return sp.simplify(a / b)

    def neg(self, a):
        return sp.simplify(-a)

    def pow_u32(self, a, n: int):
        return sp.simplify(a**n)

    def materialize(self, a):
        return sp.factor(sp.together(a))
