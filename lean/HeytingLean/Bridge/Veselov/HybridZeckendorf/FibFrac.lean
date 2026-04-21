import Mathlib.Data.Rat.Defs
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-!
Exact Fibonacci fractions from the HZAC-R brief.

This file also records a kernel-checked blocker: the PM instruction's
normalization rule `p_n + p_{n+1} = p_{n+2}` is false for the stated
definition `p_n = F_n / (F_{n+1} * F_{n+2})`.
-/

/-- Fibonacci fraction `p_n = F_n / (F_{n+1} * F_{n+2})`, exact over `Rat`. -/
def fibFrac (n : Nat) : Rat :=
  (Nat.fib n : Rat) / ((Nat.fib (n + 1) : Rat) * (Nat.fib (n + 2) : Rat))

/-- The stated fractions telescope exactly. -/
theorem fibFrac_telescoping (n : Nat) :
    fibFrac n = (1 : Rat) / Nat.fib (n + 1) - 1 / Nat.fib (n + 2) := by
  unfold fibFrac
  have h1 : ((Nat.fib (n + 1) : Rat) ≠ 0) := by
    exact_mod_cast (ne_of_gt (Nat.fib_pos.mpr (Nat.succ_pos n)))
  have h2 : ((Nat.fib (n + 2) : Rat) ≠ 0) := by
    exact_mod_cast (ne_of_gt (Nat.fib_pos.mpr (Nat.succ_pos (n + 1))))
  have hrec : ((Nat.fib (n + 2) : Rat) = Nat.fib n + Nat.fib (n + 1)) := by
    exact_mod_cast (Nat.fib_add_two (n := n))
  field_simp [h1, h2]
  rw [hrec]
  ring

/-- Finite telescoping sum for the fractional Fibonacci parts. -/
theorem fibFrac_partial_sum (n : Nat) :
    (Finset.range n).sum (fun i => fibFrac (i + 1)) =
      1 - (1 : Rat) / Nat.fib (n + 2) := by
  induction n with
  | zero =>
      norm_num
  | succ n ih =>
      rw [Finset.sum_range_succ, ih, fibFrac_telescoping (n + 1)]
      ring

/-- Counterexample to the PM sketch's adjacent-fraction normalization rule. -/
theorem fibFrac_adjacent_sum_counterexample :
    fibFrac 1 + fibFrac 2 ≠ fibFrac 3 := by
  norm_num [fibFrac, Nat.fib]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
