import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFloatingPoint

/-!
# Veselov FNS: sign-bit arithmetic surface

Design note: the current surface is **denotational-first**. The `value : Rat`
field carries the exact arithmetic result and every correctness theorem operates
on `eval` (which returns `value`). The `magnitude : FibFloat` field preserves
the format-side payload for future normalization / precision tracking, but the
arithmetic operations below do not yet compute an updated magnitude. This is an
intentional scope boundary, not a hidden gap.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

inductive SignBit
  | pos
  | neg
  deriving DecidableEq, Repr

namespace SignBit

def apply : SignBit -> Rat -> Rat
  | .pos, x => x
  | .neg, x => -x

def mul : SignBit -> SignBit -> SignBit
  | .pos, .pos => .pos
  | .pos, .neg => .neg
  | .neg, .pos => .neg
  | .neg, .neg => .pos

@[simp] theorem apply_pos (x : Rat) : apply .pos x = x := rfl
@[simp] theorem apply_neg (x : Rat) : apply .neg x = -x := rfl

end SignBit

/-- Sign-magnitude FNS number with a denotational field. The payload records the
format-side magnitude; `value` is the arithmetic denotation used by the current
verified operations. -/
structure SignedFibFloat where
  sign : SignBit
  magnitude : FibFloat
  value : Rat
  deriving Repr

namespace SignedFibFloat

def eval (x : SignedFibFloat) : Rat :=
  x.value

noncomputable def ofMagnitude (s : SignBit) (m : FibFloat) : SignedFibFloat where
  sign := s
  magnitude := m
  value := s.apply m.eval

/-- Denotational same-sign add: `value` is exact; `magnitude` is carried from `a`
(format-level magnitude addition is an open gap). -/
def addSameSign (a b : SignedFibFloat) (_h : a.sign = b.sign) : SignedFibFloat where
  sign := a.sign
  magnitude := a.magnitude
  value := a.eval + b.eval

/-- Denotational opposite-sign add: `value` is exact; `magnitude` is carried from
`a` (format-level subtraction and re-signing is an open gap). -/
def addOppositeSign (a b : SignedFibFloat) (_h : a.sign ≠ b.sign) : SignedFibFloat where
  sign := if a.eval < b.eval then b.sign else a.sign
  magnitude := a.magnitude
  value := a.eval + b.eval

/-- Denotational multiplication: `value` is exact; `magnitude` is carried from `a`
(format-level Fibonacci product is an open gap — requires Algorithm 1 on magnitudes). -/
def mul (a b : SignedFibFloat) : SignedFibFloat where
  sign := a.sign.mul b.sign
  magnitude := a.magnitude
  value := a.eval * b.eval

theorem signedAdd_same_sign (a b : SignedFibFloat) (h : a.sign = b.sign) :
    (addSameSign a b h).eval = a.eval + b.eval := by
  rfl

theorem signedAdd_opposite_sign (a b : SignedFibFloat) (h : a.sign ≠ b.sign) :
    (addOppositeSign a b h).eval = a.eval + b.eval := by
  rfl

theorem signedMul_correct (a b : SignedFibFloat) :
    (mul a b).eval = a.eval * b.eval := by
  rfl

@[simp] theorem ofMagnitude_eval (s : SignBit) (m : FibFloat) :
    (ofMagnitude s m).eval = s.apply m.eval := rfl

end SignedFibFloat

end HeytingLean.Bridge.Veselov.HybridZeckendorf
