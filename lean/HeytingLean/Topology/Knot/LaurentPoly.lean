import Std

/-!
# Knot theory: tiny Laurent polynomial type (executable)

Mathlib's `LaurentPolynomial` lives in `noncomputable` sections and is therefore not usable for
repository executables that compile to native code.

This module provides a small **executable** Laurent polynomial implementation over `Int`:
- representation: a canonical sorted `List` of `(exp, coeff)` with `coeff ≠ 0`,
- operations: `0/1`, monomials, `+`, unary `-`, `*`, `^` (Nat powers),
- deterministic serialization via `toList`.

Scope: sufficient for small state-sum invariants (e.g. the Kauffman bracket).
-/

namespace HeytingLean
namespace Topology
namespace Knot

/-- Laurent polynomials over `Int`, represented as a canonical sorted list `exp ↦ coeff`. -/
structure LaurentPoly where
  terms : List (Int × Int) := []
deriving Inhabited, Repr, DecidableEq

namespace LaurentPoly

def toList (p : LaurentPoly) : List (Int × Int) :=
  p.terms

def isZero (p : LaurentPoly) : Bool :=
  p.terms.isEmpty

/-- Insert a coefficient into a canonical Laurent-polynomial term list. -/
def insertTerm (exp : Int) (coeff : Int) : List (Int × Int) → List (Int × Int)
  | [] =>
      if coeff == 0 then [] else [(exp, coeff)]
  | (e, c) :: xs =>
      if exp < e then
        if coeff == 0 then (e, c) :: xs else (exp, coeff) :: (e, c) :: xs
      else if exp == e then
        let c' := c + coeff
        if c' == 0 then xs else (e, c') :: xs
      else
        (e, c) :: insertTerm exp coeff xs

def mon (exp : Int) (coeff : Int := 1) : LaurentPoly :=
  { terms := insertTerm exp coeff [] }

instance : Zero LaurentPoly := ⟨{ terms := [] }⟩

instance : One LaurentPoly := ⟨mon 0 1⟩

instance : Neg LaurentPoly where
  neg p :=
    { terms := p.terms.map (fun (e, c) => (e, -c)) }

instance : Add LaurentPoly where
  add p q :=
    { terms := q.terms.foldl (fun acc (e, c) => insertTerm e c acc) p.terms }

instance : Sub LaurentPoly where
  sub p q := p + (-q)

instance : Mul LaurentPoly where
  mul p q := Id.run do
    if p.isZero || q.isZero then
      return 0
    let mut acc : List (Int × Int) := []
    for (e1, c1) in p.terms do
      for (e2, c2) in q.terms do
        acc := insertTerm (e1 + e2) (c1 * c2) acc
    return { terms := acc }

def pow (p : LaurentPoly) : Nat → LaurentPoly
  | 0 => 1
  | Nat.succ n => (pow p n) * p

instance : Pow LaurentPoly Nat := ⟨pow⟩

instance : OfNat LaurentPoly n := ⟨mon 0 (Int.ofNat n)⟩

instance : ToString LaurentPoly where
  toString p :=
    if p.terms.isEmpty then "0"
    else
      let parts := p.terms.map (fun (e, c) => s!"{c}*A^{e}")
      String.intercalate " + " parts

end LaurentPoly

end Knot
end Topology
end HeytingLean
