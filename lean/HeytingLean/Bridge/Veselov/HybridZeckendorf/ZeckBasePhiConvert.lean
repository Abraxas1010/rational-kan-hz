import HeytingLean.Bridge.Veselov.HybridZeckendorf.Lucas
import HeytingLean.Bridge.Veselov.HybridZeckendorf.BasePhi
import HeytingLean.Bridge.Veselov.HybridZeckendorf.HybridNumber
import HeytingLean.Bridge.Veselov.HybridZeckendorf.BasePhiPairEval

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open LaurentPolynomial

/-!
# Zeckendorf ↔ base-φ conversion

Forward conversion (`zeckToBasePhi`): maps each Fibonacci index `i` in a
ZeckPayload to the base-φ monomial `φ^{i-2}`. This is a structural
transformation for Z[φ] arithmetic — it does NOT preserve the natural number
value (levelEval vs basePhiEval live in different domains).

Reverse conversion (`basePhiToZeck`): evaluates a base-φ Laurent polynomial to
its Z[φ] PhiPair value, then extracts a Zeckendorf payload via
`Nat.zeckendorf` when the value is a non-negative integer (phi component = 0).
-/

/-- Convert each Fibonacci index `i` into the shifted base-φ monomial `φ^(i-2)`. -/
noncomputable def zeckToBasePhi : ZeckPayload → BasePhiDigits
  | [] => 0
  | idx :: rest => C (1 : Int) * T ((idx : Int) - 2) + zeckToBasePhi rest

@[simp] theorem zeckToBasePhi_nil :
    zeckToBasePhi [] = 0 := by
  rfl

@[simp] theorem zeckToBasePhi_cons (idx : Nat) (rest : ZeckPayload) :
    zeckToBasePhi (idx :: rest) =
      C (1 : Int) * T ((idx : Int) - 2) + zeckToBasePhi rest := by
  rfl

theorem zeckToBasePhi_append (a b : ZeckPayload) :
    zeckToBasePhi (a ++ b) = zeckToBasePhi a + zeckToBasePhi b := by
  induction a with
  | nil =>
      simp
  | cons idx rest ih =>
      simp [ih, add_assoc]

/-! ### Reverse conversion via PhiPair evaluation -/

/-- Convert a base-φ Laurent polynomial back to Zeckendorf payload.
    Evaluates to PhiPair; when the phi component is zero and the constant
    is non-negative, produces the Zeckendorf encoding via `Nat.zeckendorf`.
    Otherwise returns the empty payload. -/
noncomputable def basePhiToZeck (d : BasePhiDigits) : ZeckPayload :=
  let p := basePhiPairEval d
  if p.phi = 0 ∧ 0 ≤ p.constant then
    Nat.zeckendorf p.constant.toNat
  else
    []

/-- When the base-φ value is a non-negative integer, the reverse conversion
    recovers the correct Fibonacci evaluation. -/
theorem basePhiToZeck_eval (d : BasePhiDigits)
    (h_int : (basePhiPairEval d).phi = 0) (h_pos : 0 ≤ (basePhiPairEval d).constant) :
    levelEval (basePhiToZeck d) = (basePhiPairEval d).constant.toNat := by
  simp only [basePhiToZeck, h_int, h_pos, and_self, ite_true,
    levelEval, lazyEvalFib, Nat.sum_zeckendorf_fib]

/-- The reverse conversion produces a canonical Zeckendorf representation. -/
theorem basePhiToZeck_canonical (d : BasePhiDigits)
    (h_int : (basePhiPairEval d).phi = 0) (h_pos : 0 ≤ (basePhiPairEval d).constant) :
    List.IsZeckendorfRep (basePhiToZeck d) := by
  simp only [basePhiToZeck, h_int, h_pos, and_self, ite_true]
  exact Nat.isZeckendorfRep_zeckendorf _

/-- The zero polynomial reverse-converts to the empty payload. -/
theorem basePhiToZeck_zero :
    basePhiToZeck (0 : BasePhiDigits) = [] := by
  simp [basePhiToZeck, basePhiPairEval]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
