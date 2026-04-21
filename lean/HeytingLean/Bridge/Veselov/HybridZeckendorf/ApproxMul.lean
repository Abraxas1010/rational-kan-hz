import HeytingLean.Bridge.Veselov.HybridZeckendorf.BasePhiPairEval
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFracConvergence
import Mathlib.Data.Real.GoldenRatio
import Mathlib.Tactic

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-!
# Approximate multiplication substrate

This file closes the exact Binet expansion needed for the Veselov index-addition
approximation. Tight real-inequality bounds remain a separate analytic step.
-/

open scoped goldenRatio

/-- The approximation target from the Veselov brief. -/
noncomputable def fibApprox (n m : Nat) : ℝ :=
  Real.goldenRatio ^ (n + m) / Real.sqrt 5

/-- The exact Fibonacci product expressed via Binet for both factors. -/
theorem fib_product_via_binet (n m : Nat) :
    (Nat.fib n : ℝ) * Nat.fib m =
      (Real.goldenRatio ^ n - Real.goldenConj ^ n) *
        (Real.goldenRatio ^ m - Real.goldenConj ^ m) / 5 := by
  rw [binet_nat n, binet_nat m]
  have hsqrt_ne : Real.sqrt 5 ≠ 0 := by positivity
  field_simp [hsqrt_ne]
  have hsqrt_sq : (Real.sqrt 5) ^ 2 = (5 : ℝ) := by
    rw [Real.sq_sqrt]
    norm_num
  rw [hsqrt_sq]
  ring

/-- Deterministic single-step relative-error package. -/
structure RelativeErrorStep where
  ε : ℝ
  bound : ℝ
  bound_nonneg : 0 ≤ bound
  bounded : |ε| ≤ bound

/-- A one-step bounded error factor is no larger than `1 + bound`. -/
theorem relativeError_factor_le (s : RelativeErrorStep) :
    1 + s.ε ≤ 1 + s.bound := by
  have hε : s.ε ≤ s.bound := by
    exact le_trans (le_abs_self s.ε) s.bounded
  linarith

end HeytingLean.Bridge.Veselov.HybridZeckendorf
