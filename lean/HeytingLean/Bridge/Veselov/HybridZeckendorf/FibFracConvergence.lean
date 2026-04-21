import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFrac
import Mathlib.Data.Real.GoldenRatio
import Mathlib.Tactic

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-!
# Binet-backed convergence substrate for Fibonacci fractions

This file records the exact Binet bridge available from Mathlib and the exact
finite telescoping identity used by the greedy convergence argument. The full
infinite-series and density statements are intentionally not asserted here.
-/

open scoped goldenRatio

/-- Local name for Mathlib's Binet formula. -/
theorem binet_nat (n : Nat) :
    (Nat.fib n : ℝ) =
      (Real.goldenRatio ^ n - Real.goldenConj ^ n) / Real.sqrt 5 := by
  simpa using Real.coe_fib_eq n

/-- Finite tail sum for the in-repo Fibonacci fractions. -/
theorem fibFrac_tail_partial_sum (start len : Nat) :
    (Finset.range len).sum (fun i => fibFrac (start + i)) =
      (1 : ℚ) / Nat.fib (start + 1) -
        1 / Nat.fib (start + len + 1) := by
  induction len with
  | zero =>
      simp
  | succ len ih =>
      calc
        (Finset.range (len + 1)).sum (fun i => fibFrac (start + i))
            = (Finset.range len).sum (fun i => fibFrac (start + i)) +
                fibFrac (start + len) := by
              rw [Finset.sum_range_succ]
        _ = ((1 : ℚ) / Nat.fib (start + 1) -
                1 / Nat.fib (start + len + 1)) +
              ((1 : ℚ) / Nat.fib (start + len + 1) -
                1 / Nat.fib (start + len + 2)) := by
              rw [ih, fibFrac_telescoping (start + len)]
        _ = (1 : ℚ) / Nat.fib (start + 1) -
              1 / Nat.fib (start + (len + 1) + 1) := by
              ring_nf

/-- The finite tail is bounded by its first telescoping endpoint. -/
theorem fibFrac_tail_partial_le (start len : Nat) :
    (Finset.range len).sum (fun i => fibFrac (start + i)) ≤
      (1 : ℚ) / Nat.fib (start + 1) := by
  rw [fibFrac_tail_partial_sum]
  have hpos : 0 < (Nat.fib (start + len + 1) : ℚ) := by
    exact_mod_cast Nat.fib_pos.mpr (Nat.succ_pos (start + len))
  have hnonneg : 0 ≤ (1 : ℚ) / Nat.fib (start + len + 1) := by
    positivity
  linarith

end HeytingLean.Bridge.Veselov.HybridZeckendorf
