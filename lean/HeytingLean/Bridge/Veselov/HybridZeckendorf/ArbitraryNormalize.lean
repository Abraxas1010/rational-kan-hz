import HeytingLean.Bridge.Veselov.HybridZeckendorf.ZeckBasePhiConvert
import HeytingLean.Bridge.Veselov.HybridZeckendorf.BasePhiPairEval

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open LaurentPolynomial

/-!
# Arbitrary-weight base-φ normalization

This module provides:
1. **Carry identities**: `2φ^k = φ^{k+1} + φ^{k-2}` and `φ^k + φ^{k+1} = φ^{k+2}`,
   proved in the real-valued golden ratio domain.
2. **Carry step definitions**: `carryDuplicateAt` and `eliminateConsecutiveAt` —
   single-position Laurent polynomial rewrites that preserve `basePhiEval`.
3. **Full normalizer** (`normalizeArbitrary`): reduces arbitrary-weight Laurent
   polynomials to the canonical Z[φ] form `a + bφ` via `basePhiPairEval`
   evaluation and reconstruction. This is the ring-theoretic normal form for
   Z[φ] — every element is uniquely `a + bφ` — and is provably sound.
-/

/-- Integer-weighted base-φ digits use the same Laurent carrier. -/
abbrev WeightedBasePhi := BasePhiDigits

/-- Evaluation of weighted base-φ digits. -/
noncomputable def weightedBasePhiEval (w : WeightedBasePhi) : Real :=
  basePhiEval w

/-! ### Golden ratio identities -/

/-- Golden ratio squared, with ℤ exponent. -/
theorem goldenRatio_sq_int :
    Real.goldenRatio ^ (2 : ℤ) = Real.goldenRatio + 1 := by
  exact_mod_cast Real.goldenRatio_sq

/-- The golden ratio satisfies `φ + φ⁻² = 2`. -/
theorem goldenRatio_add_inv_sq :
    Real.goldenRatio + Real.goldenRatio ^ (-2 : ℤ) = 2 := by
  have hne : Real.goldenRatio ≠ 0 := Real.goldenRatio_ne_zero
  have hne2 : Real.goldenRatio ^ (2 : ℤ) ≠ 0 := zpow_ne_zero 2 hne
  suffices h : (Real.goldenRatio + Real.goldenRatio ^ (-2 : ℤ)) * Real.goldenRatio ^ (2 : ℤ) =
               2 * Real.goldenRatio ^ (2 : ℤ) from
    mul_right_cancel₀ hne2 h
  have hcancel : Real.goldenRatio ^ (-2 : ℤ) * Real.goldenRatio ^ (2 : ℤ) = 1 := by
    rw [← zpow_add₀ hne]; norm_num
  have step1 : (Real.goldenRatio + Real.goldenRatio ^ (-2 : ℤ)) * Real.goldenRatio ^ (2 : ℤ) =
    Real.goldenRatio * Real.goldenRatio ^ (2 : ℤ) + 1 := by
    nlinarith [hcancel, mul_comm (Real.goldenRatio ^ (-2 : ℤ)) (Real.goldenRatio ^ (2 : ℤ)),
              add_mul Real.goldenRatio (Real.goldenRatio ^ (-2 : ℤ)) (Real.goldenRatio ^ (2 : ℤ))]
  rw [step1, goldenRatio_sq_int]
  have expand : Real.goldenRatio * (Real.goldenRatio + 1) =
                Real.goldenRatio ^ 2 + Real.goldenRatio := by ring
  rw [expand, Real.goldenRatio_sq]
  ring

/-- Carry identity: `2φ^k = φ^{k+1} + φ^{k-2}`. -/
theorem goldenRatio_carry (k : ℤ) :
    (2 : ℝ) * Real.goldenRatio ^ k =
      Real.goldenRatio ^ (k + 1) + Real.goldenRatio ^ (k - 2) := by
  have hne : Real.goldenRatio ≠ 0 := Real.goldenRatio_ne_zero
  have hfact : Real.goldenRatio ^ (k + 1) + Real.goldenRatio ^ (k - 2) =
               Real.goldenRatio ^ k * (Real.goldenRatio + Real.goldenRatio ^ (-2 : ℤ)) := by
    rw [show k - 2 = k + (-2 : ℤ) by omega]
    rw [zpow_add₀ hne k 1, zpow_add₀ hne k (-2), zpow_one]
    ring
  rw [hfact, goldenRatio_add_inv_sq, mul_comm]

/-- Consecutive identity: `φ^k + φ^{k+1} = φ^{k+2}`. -/
theorem goldenRatio_consecutive (k : ℤ) :
    Real.goldenRatio ^ k + Real.goldenRatio ^ (k + 1) =
      Real.goldenRatio ^ (k + 2) := by
  have hne : Real.goldenRatio ≠ 0 := Real.goldenRatio_ne_zero
  calc Real.goldenRatio ^ k + Real.goldenRatio ^ (k + 1)
      = Real.goldenRatio ^ k + Real.goldenRatio ^ k * Real.goldenRatio ^ (1 : ℤ) := by
          rw [zpow_add₀ hne]
    _ = Real.goldenRatio ^ k * (1 + Real.goldenRatio) := by
          rw [zpow_one]; ring
    _ = Real.goldenRatio ^ k * Real.goldenRatio ^ (2 : ℤ) := by
          rw [goldenRatio_sq_int]; ring
    _ = Real.goldenRatio ^ (k + 2) := by rw [← zpow_add₀ hne]

/-! ### Carry step definitions -/

/-- Apply the duplicate carry rule at position `k`:
    d(k) -= 2, d(k+1) += 1, d(k-2) += 1.
    Uses: `2φ^k = φ^{k+1} + φ^{k-2}`. -/
noncomputable def carryDuplicateAt (k : ℤ) (d : BasePhiDigits) : BasePhiDigits :=
  d + (C (-2 : ℤ) * T k + C (1 : ℤ) * T (k + 1) + C (1 : ℤ) * T (k - 2) : BasePhiDigits)

/-- The duplicate carry rule preserves `basePhiEval`. -/
theorem carryDuplicateAt_sound (k : ℤ) (d : BasePhiDigits) :
    basePhiEval (carryDuplicateAt k d) = basePhiEval d := by
  simp only [carryDuplicateAt, basePhiEval_add, basePhiEval_C_mul_T]
  push_cast
  linarith [goldenRatio_carry k]

/-- Apply the consecutive elimination rule at position `k`:
    d(k) -= 1, d(k+1) -= 1, d(k+2) += 1.
    Uses: `φ^k + φ^{k+1} = φ^{k+2}`. -/
noncomputable def eliminateConsecutiveAt (k : ℤ) (d : BasePhiDigits) : BasePhiDigits :=
  d + (C (-1 : ℤ) * T k + C (-1 : ℤ) * T (k + 1) + C (1 : ℤ) * T (k + 2) : BasePhiDigits)

/-- The consecutive elimination rule preserves `basePhiEval`. -/
theorem eliminateConsecutiveAt_sound (k : ℤ) (d : BasePhiDigits) :
    basePhiEval (eliminateConsecutiveAt k d) = basePhiEval d := by
  simp only [eliminateConsecutiveAt, basePhiEval_add, basePhiEval_C_mul_T]
  push_cast
  linarith [goldenRatio_consecutive k]

/-! ### Full normalizer via Z[φ] normal form -/

/-- Reconstruct a Laurent polynomial from a PhiPair.
    The output `a · T⁰ + b · T¹` is the Z[φ] ring normal form. -/
noncomputable def phiPairToBasePhi (p : PhiPair) : BasePhiDigits :=
  (C p.constant * T (0 : ℤ) : BasePhiDigits) + C p.phi * T 1

/-- The reconstruction preserves real-valued semantics. -/
theorem phiPairToBasePhi_eval (p : PhiPair) :
    basePhiEval (phiPairToBasePhi p) = p.eval := by
  unfold phiPairToBasePhi
  rw [basePhiEval_add]
  simp only [basePhiEval_C_mul_T, PhiPair.eval, zpow_zero, mul_one, zpow_one]

/-- Full normalizer: evaluate to PhiPair, then reconstruct.
    Reduces any Laurent polynomial to the canonical 2-term Z[φ] form `a + bφ`. -/
noncomputable def normalizeArbitrary (w : WeightedBasePhi) : BasePhiDigits :=
  phiPairToBasePhi (basePhiPairEval w)

/-- The normalizer preserves `basePhiEval`. -/
theorem normalizeArbitrary_sound (w : WeightedBasePhi) :
    basePhiEval (normalizeArbitrary w) = weightedBasePhiEval w := by
  simp [normalizeArbitrary, phiPairToBasePhi_eval, basePhiPairEval_eval, weightedBasePhiEval]

/-- The normalizer output represents the same Z[φ] element as the input. -/
theorem normalizeArbitrary_phiPair (w : WeightedBasePhi) :
    basePhiPairEval (normalizeArbitrary w) = basePhiPairEval w := by
  apply PhiPair.eval_injective
  rw [basePhiPairEval_eval, basePhiPairEval_eval, normalizeArbitrary_sound, weightedBasePhiEval]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
