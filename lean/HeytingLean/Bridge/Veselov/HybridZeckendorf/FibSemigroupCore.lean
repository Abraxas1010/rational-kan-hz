import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic.Ring

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

noncomputable def partRatRaw (n : Nat) : ℚ :=
  1 / (Nat.fib (n + 1) : ℚ) - 1 / (Nat.fib (n + 2) : ℚ)

@[simp] theorem partRatRaw_eq_telescopic (n : Nat) :
    partRatRaw n = 1 / (Nat.fib (n + 1) : ℚ) - 1 / (Nat.fib (n + 2) : ℚ) := by
  rfl

structure FibPartCarrier where
  parts : List Nat
  deriving Repr

noncomputable def FibPartCarrier.eval (s : FibPartCarrier) : ℚ :=
  (s.parts.map partRatRaw).sum

@[simp] theorem FibPartCarrier.eval_nil : FibPartCarrier.eval ⟨[]⟩ = 0 := by
  simp [FibPartCarrier.eval]

@[simp] theorem FibPartCarrier.eval_cons (n : Nat) (xs : List Nat) :
    FibPartCarrier.eval ⟨n :: xs⟩ = partRatRaw n + FibPartCarrier.eval ⟨xs⟩ := by
  simp [FibPartCarrier.eval]

theorem telescopic_prefix_sum (start len : Nat) :
    Finset.sum (Finset.range len) (fun i => partRatRaw (start + i))
      = 1 / (Nat.fib (start + 1) : ℚ) - 1 / (Nat.fib (start + len + 1) : ℚ) := by
  induction len with
  | zero =>
      simp [partRatRaw]
  | succ len ih =>
      calc
        Finset.sum (Finset.range (len + 1)) (fun i => partRatRaw (start + i))
            = Finset.sum (Finset.range len) (fun i => partRatRaw (start + i)) + partRatRaw (start + len) := by
                rw [Finset.sum_range_succ]
        _ = (1 / (Nat.fib (start + 1) : ℚ) - 1 / (Nat.fib (start + len + 1) : ℚ))
              + (1 / (Nat.fib (start + len + 1) : ℚ) - 1 / (Nat.fib (start + len + 2) : ℚ)) := by
                rw [ih]
                simp [partRatRaw, Nat.add_assoc]
        _ = 1 / (Nat.fib (start + 1) : ℚ) - 1 / (Nat.fib (start + (len + 1) + 1) : ℚ) := by
                ring_nf

end HeytingLean.Bridge.Veselov.HybridZeckendorf
