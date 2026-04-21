import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFracConvergence
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibSemigroup
import Mathlib.Data.Nat.Fib.Zeckendorf

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-!
# Fibonacci-coded rational surface

The full density theorem depends on the complete greedy convergence theorem.
This file provides the exact coded-rational surface and natural-number exactness
without claiming the unfinished density result.
-/

/-- A finite Fibonacci-coded rational: Zeckendorf integer payload plus fraction indices. -/
structure FibCodedRat where
  intPart : List Nat
  fracPart : List Nat
  deriving Repr

/-- Exact rational evaluation of a finite Fibonacci-coded rational. -/
noncomputable def FibCodedRat.eval (r : FibCodedRat) : ℚ :=
  (lazyEvalFib r.intPart : ℚ) + (r.fracPart.map fibFrac).sum

/-- Natural numbers embed exactly via their Zeckendorf payload and no fractional part. -/
theorem fibCoded_nat_exact (n : Nat) :
    ∃ r : FibCodedRat, r.fracPart = [] ∧ r.eval = n := by
  refine ⟨{ intPart := Nat.zeckendorf n, fracPart := [] }, rfl, ?_⟩
  simp [FibCodedRat.eval, lazyEvalFib, Nat.sum_zeckendorf_fib n]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
