import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic.Ring
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibIdentities
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibSemigroupCore

/-!
# Bridge.Veselov.HybridZeckendorf.FibSemigroup

Raw and canonical surfaces for Vladimir's Fibonacci semigroup over `(0,1]`.

The paper's additive structure has two distinct layers:

- a **raw additive carrier** where addition is just concatenation of Fibonacci
  payloads and telescopic-part payloads, and
- a **canonical carrier** imposing the Zeckendorf no-consecutive constraint on
  the fractional indices.

The raw layer is the mathematically easy part: concatenation already gives a
semigroup, and evaluation is additive. The hard part is the normalization map
from the raw layer back into canonical representatives. That normalization still
remains open here and is stated honestly below.

## Formalized here

- `FibSemigroupRaw` with exact rational evaluation
- raw addition by concatenation
- value preservation for raw addition
- canonical carrier `FibSemigroup` as a constrained subset of the raw carrier
- the finite telescopic identity `Σ_{i < len} p_{1+i} = 1 - 1/F_{len+2}`

## Still open

- canonical normalization `FibSemigroupRaw → FibSemigroup`
- proof that the chosen normalization preserves value
- uniqueness / completeness of canonical representatives for the fractional lane
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-- Raw semigroup carrier: integer Fibonacci payload plus fractional part payload. -/
structure FibSemigroupRaw where
  fibs : List Nat
  parts : List Nat
  deriving Repr

/-- Evaluate the integer Fibonacci payload. -/
def FibSemigroupRaw.evalFibs (s : FibSemigroupRaw) : Nat :=
  lazyEvalFib s.fibs

/-- Evaluate the fractional telescopic payload. -/
noncomputable def FibSemigroupRaw.evalParts (s : FibSemigroupRaw) : ℚ :=
  FibPartCarrier.eval ⟨s.parts⟩

/-- Full exact evaluation of the raw carrier. -/
noncomputable def FibSemigroupRaw.eval (s : FibSemigroupRaw) : ℚ :=
  (s.evalFibs : ℚ) + s.evalParts

instance : Zero FibSemigroupRaw where
  zero := ⟨[], []⟩

/-- Raw semigroup addition is concatenation of both payload lists. -/
def FibSemigroupRaw.add (a b : FibSemigroupRaw) : FibSemigroupRaw :=
  ⟨a.fibs ++ b.fibs, a.parts ++ b.parts⟩

instance : Add FibSemigroupRaw where
  add := FibSemigroupRaw.add

@[simp] theorem FibSemigroupRaw.zero_fibs : (0 : FibSemigroupRaw).fibs = [] := rfl

@[simp] theorem FibSemigroupRaw.zero_parts : (0 : FibSemigroupRaw).parts = [] := rfl

@[simp] theorem FibSemigroupRaw.evalFibs_zero : FibSemigroupRaw.evalFibs 0 = 0 := by
  simp [FibSemigroupRaw.evalFibs]

@[simp] theorem FibSemigroupRaw.evalParts_zero : FibSemigroupRaw.evalParts 0 = 0 := by
  simp [FibSemigroupRaw.evalParts]

@[simp] theorem FibSemigroupRaw.eval_zero : FibSemigroupRaw.eval 0 = 0 := by
  simp [FibSemigroupRaw.eval]

@[simp] theorem FibSemigroupRaw.fibs_add (a b : FibSemigroupRaw) :
    (a + b).fibs = a.fibs ++ b.fibs := by
  rfl

@[simp] theorem FibSemigroupRaw.parts_add (a b : FibSemigroupRaw) :
    (a + b).parts = a.parts ++ b.parts := by
  rfl

theorem FibSemigroupRaw.evalFibs_add (a b : FibSemigroupRaw) :
    FibSemigroupRaw.evalFibs (a + b) =
      FibSemigroupRaw.evalFibs a + FibSemigroupRaw.evalFibs b := by
  simp [FibSemigroupRaw.evalFibs, lazyEvalFib_append]

theorem FibSemigroupRaw.evalParts_add (a b : FibSemigroupRaw) :
    FibSemigroupRaw.evalParts (a + b) =
      FibSemigroupRaw.evalParts a + FibSemigroupRaw.evalParts b := by
  simp [FibSemigroupRaw.evalParts, FibPartCarrier.eval, List.map_append, List.sum_append]

theorem FibSemigroupRaw.eval_add (a b : FibSemigroupRaw) :
    FibSemigroupRaw.eval (a + b) =
      FibSemigroupRaw.eval a + FibSemigroupRaw.eval b := by
  rw [FibSemigroupRaw.eval, FibSemigroupRaw.eval, FibSemigroupRaw.eval, FibSemigroupRaw.evalFibs_add,
    FibSemigroupRaw.evalParts_add]
  simp [Nat.cast_add, add_assoc, add_left_comm]

/-- Canonical representatives enforce the no-consecutive constraint on the part payload. -/
structure FibSemigroup where
  fibs : List Nat
  parts : List Nat
  no_consec : ∀ i, i ∈ parts → (i + 1) ∉ parts
  deriving Repr

/-- Forget the canonical proof and view a canonical element as a raw one. -/
def FibSemigroup.toRaw (s : FibSemigroup) : FibSemigroupRaw :=
  ⟨s.fibs, s.parts⟩

/-- Evaluate the fractional part of a canonical element. -/
noncomputable def FibSemigroup.evalParts (s : FibSemigroup) : ℚ :=
  s.toRaw.evalParts

/-- Evaluate the integer Fibonacci payload of a canonical element. -/
def FibSemigroup.evalFibs (s : FibSemigroup) : Nat :=
  s.toRaw.evalFibs

/-- Full exact evaluation of a canonical element. -/
noncomputable def FibSemigroup.eval (s : FibSemigroup) : ℚ :=
  s.toRaw.eval

/-- Canonical empty element. -/
def FibSemigroup.empty : FibSemigroup where
  fibs := []
  parts := []
  no_consec := by simp

/-- Raw addition of canonical elements, leaving normalization as a separate obligation. -/
def FibSemigroup.addRaw (a b : FibSemigroup) : FibSemigroupRaw :=
  a.toRaw + b.toRaw

@[simp] theorem FibSemigroup.toRaw_fibs (s : FibSemigroup) : s.toRaw.fibs = s.fibs := rfl

@[simp] theorem FibSemigroup.toRaw_parts (s : FibSemigroup) : s.toRaw.parts = s.parts := rfl

@[simp] theorem FibSemigroup.evalParts_eq (s : FibSemigroup) :
    s.evalParts = FibPartCarrier.eval ⟨s.parts⟩ := by
  rfl

@[simp] theorem FibSemigroup.evalFibs_eq (s : FibSemigroup) :
    s.evalFibs = lazyEvalFib s.fibs := by
  rfl

@[simp] theorem FibSemigroup.eval_eq (s : FibSemigroup) :
    s.eval = (lazyEvalFib s.fibs : ℚ) + FibPartCarrier.eval ⟨s.parts⟩ := by
  rfl

@[simp] theorem FibSemigroup.eval_empty : FibSemigroup.empty.eval = 0 := by
  simp [FibSemigroup.eval, FibSemigroup.empty, FibSemigroup.toRaw, FibSemigroupRaw.eval,
    FibSemigroupRaw.evalParts]

@[simp] theorem FibSemigroup.evalFibs_singleton (n : Nat) :
    ({ fibs := [n], parts := [], no_consec := by simp } : FibSemigroup).evalFibs = Nat.fib n := by
  simp [FibSemigroup.evalFibs_eq, lazyEvalFib]

theorem FibSemigroup.eval_addRaw (a b : FibSemigroup) :
    FibSemigroupRaw.eval (a.addRaw b) = a.eval + b.eval := by
  simpa [FibSemigroup.addRaw, FibSemigroup.eval] using
    FibSemigroupRaw.eval_add a.toRaw b.toRaw

/-- Finite telescopic identity for the paper's fractional generators `p_n`, starting at `n = 1`. -/
theorem telescopic_sum_identity (len : Nat) :
    Finset.sum (Finset.range len) (fun i => partRatRaw (1 + i))
      = 1 - 1 / (Nat.fib (len + 2) : ℚ) := by
  simpa [partRatRaw, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
    telescopic_prefix_sum 1 len

/-- The adjacent rewrite `p_n + p_{n+1} = p_{n+2}` printed in `vlad12.pdf`
is false for the `partRatRaw` definition used in the paper. -/
theorem partRatRaw_adjacent_step_not_next :
    partRatRaw 1 + partRatRaw 2 ≠ partRatRaw 3 := by
  norm_num [partRatRaw]

/-- Consequently, the printed fractional normalization rule cannot be sound as
stated for these Fibonacci parts. -/
theorem not_fractional_adjacent_carry_rule :
    ¬ ∀ n : Nat, partRatRaw n + partRatRaw (n + 1) = partRatRaw (n + 2) := by
  intro h
  exact partRatRaw_adjacent_step_not_next (h 1)

end HeytingLean.Bridge.Veselov.HybridZeckendorf
