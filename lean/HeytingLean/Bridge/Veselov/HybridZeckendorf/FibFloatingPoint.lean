import Mathlib.Algebra.Order.Ring.Rat
import HeytingLean.Bridge.Veselov.HybridZeckendorf.LinearTimeAdd
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFrac
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFracConvergence

/-!
# Veselov FNS: Fibonacci floating point format
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-- Fibonacci-style floating point payload:
integer Zeckendorf digits, fractional Fibonacci-weight bits, and the
normalization flag required for exceptional two-representation boundary cases. -/
structure FibFloat where
  intPart : ZeckPayload
  fracPart : List Bool
  normFlag : Bool
  deriving DecidableEq, Repr

namespace FibFloat

def zero : FibFloat where
  intPart := []
  fracPart := []
  normFlag := false

/-- Boolean fractional bit at list position `i` carries paper weight `a_(i+2)`. -/
noncomputable def fracBitEval (i : Nat) (b : Bool) : Rat :=
  if b then fibFrac (i + 2) else 0

noncomputable def fracEval.go : Nat -> List Bool -> Rat
  | _, [] => 0
  | i, b :: bs => fracBitEval i b + fracEval.go (i + 1) bs

noncomputable def fracEval (bits : List Bool) : Rat :=
  fracEval.go 0 bits

noncomputable def eval (f : FibFloat) : Rat :=
  (levelEval f.intPart : Rat) + fracEval f.fracPart

def truncate (f : FibFloat) (L : Nat) : FibFloat :=
  { f with fracPart := f.fracPart.take L }

/-- The paper's advertised tail budget at fractional precision `L`. -/
noncomputable def precisionBudget (L : Nat) : Rat :=
  1 / (Nat.fib (L + 1) : Rat)

theorem truncate_frac_length_le (f : FibFloat) (L : Nat) :
    (truncate f L).fracPart.length ≤ L := by
  simp [truncate]

theorem zero_eval : eval zero = 0 := by
  norm_num [zero, eval, fracEval, fracEval.go]

theorem fibFrac_nonneg (n : Nat) : 0 ≤ fibFrac n := by
  unfold fibFrac
  positivity

theorem fracBitEval_nonneg (i : Nat) (b : Bool) : 0 ≤ fracBitEval i b := by
  by_cases hb : b <;> simp [fracBitEval, hb, fibFrac_nonneg]

theorem fracBitEval_le (i : Nat) (b : Bool) : fracBitEval i b ≤ fibFrac (i + 2) := by
  by_cases hb : b <;> simp [fracBitEval, hb, fibFrac_nonneg]

theorem fracEval_go_nonneg (i : Nat) (bits : List Bool) :
    0 ≤ fracEval.go i bits := by
  induction bits generalizing i with
  | nil =>
      simp [fracEval.go]
  | cons b bs ih =>
      have hb := fracBitEval_nonneg i b
      have ht := ih (i + 1)
      simp [fracEval.go]
      linarith

theorem fracEval_go_le_tail (i : Nat) (bits : List Bool) :
    fracEval.go i bits ≤
      (Finset.range bits.length).sum (fun j => fibFrac (i + j + 2)) := by
  induction bits generalizing i with
  | nil =>
      simp [fracEval.go]
  | cons b bs ih =>
      have hb := fracBitEval_le i b
      have ht := ih (i + 1)
      have hshift :
          (Finset.range bs.length).sum (fun j => fibFrac (i + 1 + j + 2)) =
            (Finset.range bs.length).sum (fun j => fibFrac (i + (j + 1) + 2)) := by
        apply Finset.sum_congr rfl
        intro j _hj
        congr 1
        omega
      calc
        fracEval.go i (b :: bs)
            = fracBitEval i b + fracEval.go (i + 1) bs := rfl
        _ ≤ fibFrac (i + 2) +
              (Finset.range bs.length).sum (fun j => fibFrac (i + 1 + j + 2)) := by
              linarith
        _ = (Finset.range bs.length).sum (fun j => fibFrac (i + (j + 1) + 2)) + fibFrac (i + 2) := by
              rw [hshift, add_comm]
        _ = (Finset.range (b :: bs).length).sum (fun j => fibFrac (i + j + 2)) := by
              rw [List.length_cons, Finset.sum_range_succ']

theorem fracEval_go_append (i : Nat) (xs ys : List Bool) :
    fracEval.go i (xs ++ ys) =
      fracEval.go i xs + fracEval.go (i + xs.length) ys := by
  induction xs generalizing i with
  | nil =>
      simp [fracEval.go]
  | cons x xs ih =>
      calc
        fracEval.go i ((x :: xs) ++ ys)
            = fracBitEval i x + fracEval.go (i + 1) (xs ++ ys) := rfl
        _ = fracBitEval i x +
              (fracEval.go (i + 1) xs + fracEval.go (i + 1 + xs.length) ys) := by
              rw [ih]
        _ = fracEval.go i (x :: xs) + fracEval.go (i + (x :: xs).length) ys := by
              have hidx : i + 1 + xs.length = i + (xs.length + 1) := by omega
              simp [fracEval.go, hidx]
              ring

theorem fracEval_take_add_drop (bits : List Bool) (L : Nat) :
    fracEval bits =
      fracEval (bits.take L) + fracEval.go (bits.take L).length (bits.drop L) := by
  unfold fracEval
  have h := fracEval_go_append 0 (bits.take L) (bits.drop L)
  rw [List.take_append_drop] at h
  simpa using h

theorem truncate_error_eq_tail (f : FibFloat) (L : Nat) :
    eval f - eval (truncate f L) =
      fracEval.go (f.fracPart.take L).length (f.fracPart.drop L) := by
  rw [eval, eval, truncate, fracEval_take_add_drop f.fracPart L]
  ring

theorem truncate_error_nonneg (f : FibFloat) (L : Nat) :
    0 ≤ eval f - eval (truncate f L) := by
  rw [truncate_error_eq_tail]
  exact fracEval_go_nonneg (f.fracPart.take L).length (f.fracPart.drop L)

theorem truncate_error_bound_by_start (f : FibFloat) (L : Nat) :
    eval f - eval (truncate f L) ≤ precisionBudget ((f.fracPart.take L).length + 2) := by
  rw [truncate_error_eq_tail, precisionBudget]
  have htail := fracEval_go_le_tail (f.fracPart.take L).length (f.fracPart.drop L)
  have htail' :
      fracEval.go (f.fracPart.take L).length (f.fracPart.drop L) ≤
        (Finset.range (f.fracPart.drop L).length).sum
          (fun j => fibFrac ((f.fracPart.take L).length + 2 + j)) := by
    simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using htail
  have hsum :=
    fibFrac_tail_partial_le ((f.fracPart.take L).length + 2) (f.fracPart.drop L).length
  exact le_trans htail' hsum

theorem truncate_precision_bound (f : FibFloat) (L : Nat) (hL : L ≤ f.fracPart.length) :
    |eval f - eval (truncate f L)| ≤ precisionBudget (L + 2) := by
  have hlen : (f.fracPart.take L).length = L := List.length_take_of_le hL
  have hnonneg := truncate_error_nonneg f L
  rw [abs_of_nonneg hnonneg]
  simpa [hlen] using truncate_error_bound_by_start f L

theorem truncate_precision_bound_budget (f : FibFloat) (L : Nat) (hL : L ≤ f.fracPart.length) :
    |eval f - eval (truncate f L)| ≤ precisionBudget L := by
  have hbound := truncate_precision_bound f L hL
  have hpos : (0 : ℚ) < Nat.fib (L + 1) := by
    exact_mod_cast Nat.fib_pos.mpr (Nat.succ_pos L)
  have hmono : (Nat.fib (L + 1) : ℚ) ≤ Nat.fib (L + 3) := by
    exact_mod_cast Nat.fib_mono (by omega : L + 1 ≤ L + 3)
  have hbudget : precisionBudget (L + 2) ≤ precisionBudget L := by
    unfold precisionBudget
    exact one_div_le_one_div_of_le hpos hmono
  exact le_trans hbound hbudget

theorem truncate_precision_bound_budget_all (f : FibFloat) (L : Nat) :
    |eval f - eval (truncate f L)| ≤ precisionBudget L := by
  by_cases hL : L ≤ f.fracPart.length
  · exact truncate_precision_bound_budget f L hL
  · have hlen : f.fracPart.length ≤ L := Nat.le_of_not_ge hL
    have ht : f.fracPart.take L = f.fracPart := List.take_of_length_le hlen
    have hbudget_nonneg : 0 ≤ precisionBudget L := by
      unfold precisionBudget
      positivity
    simp [eval, truncate, ht, hbudget_nonneg]

end FibFloat

end HeytingLean.Bridge.Veselov.HybridZeckendorf
