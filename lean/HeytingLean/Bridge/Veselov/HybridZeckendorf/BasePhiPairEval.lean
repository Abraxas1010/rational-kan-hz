import HeytingLean.Bridge.Veselov.HybridZeckendorf.BasePhiPair
import HeytingLean.Bridge.Veselov.HybridZeckendorf.BasePhi
import Mathlib.RingTheory.Real.Irrational

open LaurentPolynomial
open scoped goldenRatio

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

@[ext] theorem PhiPair.ext (a b : PhiPair)
    (hconst : a.constant = b.constant) (hphi : a.phi = b.phi) : a = b := by
  cases a
  cases b
  cases hconst
  cases hphi
  rfl

/-- Exact addition on the `Z + Zφ` carrier. -/
def phiPairAdd (a b : PhiPair) : PhiPair :=
  { constant := a.constant + b.constant
    phi := a.phi + b.phi }

/-- Integer scaling on the exact `Z + Zφ` carrier. -/
def phiPairScale (coeff : Int) (p : PhiPair) : PhiPair :=
  { constant := coeff * p.constant
    phi := coeff * p.phi }

/-- Exact multiplication on the `Z + Zφ` carrier using `φ² = φ + 1`. -/
def phiPairMul (a b : PhiPair) : PhiPair :=
  { constant := a.constant * b.constant + a.phi * b.phi
    phi := a.constant * b.phi + a.phi * b.constant + a.phi * b.phi }

@[simp] theorem phiPairAdd_eval (a b : PhiPair) :
    (phiPairAdd a b).eval = a.eval + b.eval := by
  unfold phiPairAdd PhiPair.eval
  push_cast
  ring

@[simp] theorem phiPairScale_eval (coeff : Int) (p : PhiPair) :
    (phiPairScale coeff p).eval = (coeff : ℝ) * p.eval := by
  unfold phiPairScale PhiPair.eval
  push_cast
  ring

@[simp] theorem phiPairMul_eval (a b : PhiPair) :
    (phiPairMul a b).eval = a.eval * b.eval := by
  let x : ℝ := Real.goldenRatio
  have hx : x ^ 2 = x + 1 := by
    dsimp [x]
    linarith [Real.goldenRatio_sq]
  unfold phiPairMul PhiPair.eval
  change ((a.constant * b.constant + a.phi * b.phi : Int) : ℝ) +
      ((a.constant * b.phi + a.phi * b.constant + a.phi * b.phi : Int) : ℝ) * x =
    ((((a.constant : Int) : ℝ) + ((a.phi : Int) : ℝ) * x) *
      (((b.constant : Int) : ℝ) + ((b.phi : Int) : ℝ) * x))
  push_cast
  calc
    (a.constant : ℝ) * b.constant + a.phi * b.phi + (a.constant * b.phi + a.phi * b.constant + a.phi * b.phi) * x
        = (a.constant : ℝ) * b.constant + (a.constant * b.phi + a.phi * b.constant) * x + a.phi * b.phi * (x + 1) := by
            ring
    _ = (a.constant : ℝ) * b.constant + (a.constant * b.phi + a.phi * b.constant) * x + a.phi * b.phi * (x ^ 2) := by
            rw [hx]
    _ = (((a.constant : Int) : ℝ) + ((a.phi : Int) : ℝ) * x) *
          (((b.constant : Int) : ℝ) + ((b.phi : Int) : ℝ) * x) := by
            ring

theorem PhiPair.eval_injective : Function.Injective PhiPair.eval := by
  intro a b h
  rcases a with ⟨ac, ap⟩
  rcases b with ⟨bc, bp⟩
  dsimp [PhiPair.eval] at h
  by_cases hphi : ap = bp
  · rw [hphi] at h
    have hconstR : (ac : ℝ) = (bc : ℝ) := by linarith
    have hconst : ac = bc := by exact_mod_cast hconstR
    cases hphi
    cases hconst
    rfl
  · have hcoeff_ne : (ap - bp : Int) ≠ 0 := sub_ne_zero.mpr hphi
    have hmul : (((ap - bp : Int) : ℝ) * Real.goldenRatio) = -((ac - bc : Int) : ℝ) := by
      have h' : ((ac : ℝ) - bc) + (((ap : ℝ) - bp) * Real.goldenRatio) = 0 := by
        linarith
      have h'' : ((ac - bc : Int) : ℝ) + ((ap - bp : Int) : ℝ) * Real.goldenRatio = 0 := by
        norm_num [sub_eq_add_neg] at h' ⊢
        exact h'
      linarith
    have hirr_mul : Irrational (((ap - bp : Int) : ℝ) * Real.goldenRatio) := by
      rw [irrational_intCast_mul_iff]
      exact ⟨hcoeff_ne, Real.goldenRatio_irrational⟩
    have hrat : (((ap - bp : Int) : ℝ) * Real.goldenRatio) ∈ Set.range (Rat.cast : ℚ → ℝ) := by
      refine ⟨((-(ac - bc : Int)) : ℚ), ?_⟩
      rw [hmul]
      exact_mod_cast rfl
    exfalso
    exact hirr_mul hrat

/-- Exact base-`φ` pair evaluation mirroring the executable Rust fold. -/
noncomputable def basePhiPairEval (d : BasePhiDigits) : PhiPair :=
  let coeffs : Int × Int :=
    d.sum (fun i coeff => (coeff * signedFib (i - 1), coeff * signedFib i))
  { constant := coeffs.1
    phi := coeffs.2 }

@[simp] theorem basePhiPairEval_zero :
    basePhiPairEval 0 = { constant := 0, phi := 0 } := by
  simp [basePhiPairEval]

theorem basePhiPairEval_add (a b : BasePhiDigits) :
    basePhiPairEval (a + b) = phiPairAdd (basePhiPairEval a) (basePhiPairEval b) := by
  classical
  ext
  · unfold basePhiPairEval phiPairAdd
    dsimp
    have hsum :=
      (Finsupp.sum_add_index'
        (f := a) (g := b)
        (h := fun i coeff : Int => (coeff * signedFib (i - 1), coeff * signedFib i))
        (h_zero := by
          intro i
          simp)
        (h_add := by
          intro i x y
          simp [Int.add_mul]))
    exact congrArg Prod.fst hsum
  · unfold basePhiPairEval phiPairAdd
    dsimp
    have hsum :=
      (Finsupp.sum_add_index'
        (f := a) (g := b)
        (h := fun i coeff : Int => (coeff * signedFib (i - 1), coeff * signedFib i))
        (h_zero := by
          intro i
          simp)
        (h_add := by
          intro i x y
          simp [Int.add_mul]))
    exact congrArg Prod.snd hsum

@[simp] theorem basePhiPairEval_single (i coeff : Int) :
    basePhiPairEval (Finsupp.single i coeff : BasePhiDigits) = phiPairScale coeff (phiPairPow i) := by
  ext
  · unfold basePhiPairEval phiPairScale phiPairPow
    dsimp
    simpa using
      (congrArg Prod.fst <|
        Finsupp.sum_single_index (a := i) (b := coeff)
          (h := fun j c : Int => (c * signedFib (j - 1), c * signedFib j))
          (by simp : (0 * signedFib (i - 1), 0 * signedFib i) = (0, 0)))
  · unfold basePhiPairEval phiPairScale phiPairPow
    dsimp
    simpa using
      (congrArg Prod.snd <|
        Finsupp.sum_single_index (a := i) (b := coeff)
          (h := fun j c : Int => (c * signedFib (j - 1), c * signedFib j))
          (by simp : (0 * signedFib (i - 1), 0 * signedFib i) = (0, 0)))

@[simp] theorem basePhiPairEval_C_mul_T (coeff : Int) (n : Int) :
    basePhiPairEval (LaurentPolynomial.C coeff * LaurentPolynomial.T n : BasePhiDigits) =
      phiPairScale coeff (phiPairPow n) := by
  rw [← LaurentPolynomial.single_eq_C_mul_T]
  exact basePhiPairEval_single n coeff

theorem basePhiPairEval_eval (d : BasePhiDigits) :
    (basePhiPairEval d).eval = basePhiEval d := by
  classical
  induction d using LaurentPolynomial.induction_on' with
  | add p q hp hq =>
      rw [basePhiPairEval_add, phiPairAdd_eval, hp, hq, basePhiEval_add]
  | C_mul_T n coeff =>
      rw [basePhiPairEval_C_mul_T, phiPairScale_eval, phiPairPow_eval, basePhiEval_C_mul_T]

theorem basePhiPairEval_rawBasePhiMul (a b : BasePhiDigits) :
    basePhiPairEval (rawBasePhiMul a b) =
      phiPairMul (basePhiPairEval a) (basePhiPairEval b) := by
  apply PhiPair.eval_injective
  rw [basePhiPairEval_eval, phiPairMul_eval, basePhiPairEval_eval, basePhiPairEval_eval,
    basePhiEval_rawBasePhiMul]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
