import Mathlib.Algebra.Polynomial.Laurent
import Mathlib.Data.Real.GoldenRatio
import HeytingLean.Bridge.Veselov.HybridZeckendorf.Shiftable

open LaurentPolynomial
open scoped goldenRatio

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

abbrev BasePhiDigits := LaurentPolynomial ℤ

/-- The unit corresponding to `φ`, used for Laurent evaluation. -/
noncomputable def goldenRatioUnit : ℝˣ :=
  Units.mk0 Real.goldenRatio Real.goldenRatio_ne_zero

/-- Real-valued base-phi semantics on finitely supported integer-indexed digits. -/
noncomputable def basePhiEval (d : BasePhiDigits) : Real :=
  LaurentPolynomial.eval₂ (Int.castRingHom ℝ) goldenRatioUnit d

/-- Local admissibility condition: no adjacent nonzero digits. -/
def basePhiCanonical (d : BasePhiDigits) : Prop :=
  ∀ i, d i * d (i + 1) = 0

/-- Initial semantic bridge from the exact shift-support carrier to base-phi digits. -/
noncomputable def shiftToBasePhi (s : ShiftSupport) : BasePhiDigits :=
  s

/-- Raw base-phi multiplication is Laurent multiplication on the digit carrier. -/
noncomputable def rawBasePhiMul (a b : BasePhiDigits) : BasePhiDigits :=
  a * b

@[simp] theorem basePhiEval_zero : basePhiEval 0 = 0 := by
  simp [basePhiEval]

theorem basePhiEval_add (a b : BasePhiDigits) :
    basePhiEval (a + b) = basePhiEval a + basePhiEval b := by
  simp [basePhiEval]

@[simp] theorem basePhiEval_C_mul_T (coeff : ℤ) (n : ℤ) :
    basePhiEval (C coeff * T n : BasePhiDigits) = (coeff : ℝ) * (Real.goldenRatio ^ n) := by
  simp [basePhiEval, goldenRatioUnit]

theorem basePhiEval_eq_sum (d : BasePhiDigits) :
    basePhiEval d = d.sum (fun i coeff => (coeff : Real) * (Real.goldenRatio ^ i)) := by
  induction d using LaurentPolynomial.induction_on' with
  | add p q hp hq =>
      rw [basePhiEval_add, hp, hq]
      symm
      exact Finsupp.sum_add_index
        (fun i => by simp)
        (fun i _ x y => by simp [add_mul])
  | C_mul_T n coeff =>
      rw [basePhiEval_C_mul_T]
      rw [← LaurentPolynomial.single_eq_C_mul_T]
      symm
      simpa using
        (Finsupp.sum_single_index
          (a := n)
          (b := coeff)
          (h := fun i c => (c : ℝ) * (Real.goldenRatio ^ i))
          (by simp : (((0 : ℤ) : ℝ) * (Real.goldenRatio ^ n)) = 0))

@[simp] theorem shiftToBasePhi_zero : shiftToBasePhi 0 = 0 := by
  rfl

theorem shiftToBasePhi_semantics (s : ShiftSupport) :
    basePhiEval (shiftToBasePhi s) = s.sum (fun i coeff => (coeff : Real) * (Real.goldenRatio ^ i)) := by
  simpa [shiftToBasePhi] using basePhiEval_eq_sum (shiftToBasePhi s)

theorem basePhiEval_mul (a b : BasePhiDigits) :
    basePhiEval (a * b) = basePhiEval a * basePhiEval b := by
  simp [basePhiEval]

theorem basePhiEval_rawBasePhiMul (a b : BasePhiDigits) :
    basePhiEval (rawBasePhiMul a b) = basePhiEval a * basePhiEval b := by
  simpa [rawBasePhiMul] using basePhiEval_mul a b

end HeytingLean.Bridge.Veselov.HybridZeckendorf
