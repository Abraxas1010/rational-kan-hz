import HeytingLean.LoF.Nucleus
import HeytingLean.Logic.Dialectic

/-!
# Modal dial on the Heyting core

A modal dial packages a core re-entry nucleus together with breathing operators `□` and `◇`
respecting the ordering hierarchy required in the roadmap.
-/

namespace HeytingLean
namespace Logic
namespace Modal

open HeytingLean.LoF

universe u

/-- A modal dial enriches a core nucleus with interior/exterior companions. -/
structure Dial (α : Type u) [PrimaryAlgebra α] where
  core : Reentry α
  box : Reentry α
  diamond : Reentry α
  box_le_core : ∀ a : α, box a ≤ core a
  core_le_diamond : ∀ a : α, core a ≤ diamond a
  box_le_diamond : ∀ a : α, box a ≤ diamond a

namespace Dial

variable {α : Type u} [PrimaryAlgebra α] (D : Dial α)

@[simp] lemma box_le_core_apply (a : α) : D.box a ≤ D.core a :=
  D.box_le_core a

@[simp] lemma core_le_diamond_apply (a : α) : D.core a ≤ D.diamond a :=
  D.core_le_diamond a

/-- The breathing cycle collapses a proposition by interiorising after an exterior move. -/
def collapse (a : α) : α :=
  D.core (D.diamond a)

/-- The breathing cycle expands a proposition by exteriorising after an interior move. -/
def expand (a : α) : α :=
  D.diamond (D.box a)

/-- Collapse is monotone because it is a composition of monotone nuclei. -/
lemma collapse_monotone : Monotone D.collapse := by
  intro a b h
  unfold collapse
  have := D.diamond.monotone h
  exact D.core.monotone this

/-- Helper inequality form of `collapse_monotone` for automation. -/
@[mono] lemma collapse_le {a b : α} (h : a ≤ b) : D.collapse a ≤ D.collapse b :=
  D.collapse_monotone h

/-- Expansion is monotone because it is a composition of monotone nuclei. -/
lemma expand_monotone : Monotone D.expand := by
  intro a b h
  unfold expand
  have := D.box.monotone h
  exact D.diamond.monotone this

/-- Helper inequality form of `expand_monotone` for automation. -/
@[mono] lemma expand_le {a b : α} (h : a ≤ b) : D.expand a ≤ D.expand b :=
  D.expand_monotone h

/-- The breathing cycle remains below the next exterior pass. -/
lemma collapse_le_next (a : α) : D.collapse a ≤ D.diamond (D.diamond a) := by
  unfold collapse
  exact D.core_le_diamond_apply _

@[simp] lemma box_le_diamond_apply (a : α) : D.box a ≤ D.diamond a :=
  D.box_le_diamond a

/-- A breathing cycle absorbs the boxed contribution. -/
lemma box_le_collapse (a : α) : D.box a ≤ D.collapse a := by
  unfold collapse
  have h₁ : D.box a ≤ D.diamond a := D.box_le_diamond_apply a
  have h₂ := D.box.monotone h₁
  have h₂' : D.box a ≤ D.box (D.diamond a) := by
    have hId := Reentry.idempotent (R := D.box) (a := a)
    calc
      D.box a = D.box (D.box a) := hId.symm
      _ ≤ D.box (D.diamond a) := h₂
  have h₃ : D.box (D.diamond a) ≤ D.core (D.diamond a) :=
    D.box_le_core_apply _
  exact le_trans h₂' h₃

/-- Expansion always dominates the boxed contribution, so the breathing cycle
never decreases the boxed part. -/
lemma box_le_expand (a : α) : D.box a ≤ D.expand a := by
  unfold expand
  have h := Dial.box_le_diamond_apply (D := D) (a := D.box a)
  have hId := Reentry.idempotent (R := D.box) (a := a)
  exact (le_of_eq_of_le hId.symm h)

@[simp] lemma collapse_eq (a : α) : D.collapse a = D.core (D.diamond a) := rfl

@[simp] lemma expand_eq (a : α) : D.expand a = D.diamond (D.box a) := rfl

@[simp] def trivial (R : Reentry α) : Dial α where
  core := R
  box := R
  diamond := R
  box_le_core := by intro; rfl
  core_le_diamond := by intro; rfl
  box_le_diamond := by intro; rfl

end Dial

/-- Dial parameters annotate the dimensional ladder for modal breathing. -/
structure DialParam (α : Type u) [PrimaryAlgebra α] where
  dimension : ℕ
  dial : Dial α

namespace DialParam

variable {α : Type u} [PrimaryAlgebra α] (P : DialParam α)

/-- Collapse interpreted at the parameter level. -/
def collapse : α → α :=
  P.dial.collapse

/-- Expansion interpreted at the parameter level. -/
def expand : α → α :=
  P.dial.expand

lemma collapse_monotone : Monotone P.collapse :=
  P.dial.collapse_monotone

/-- Expansion is monotone for every dial parameter. -/
lemma expand_monotone : Monotone P.expand :=
  P.dial.expand_monotone

/-- Boxing at the parameter level is always bounded by the associated expansion. -/
lemma box_le_expand (a : α) : P.dial.box a ≤ P.expand a :=
  P.dial.box_le_expand a

/-- The breathing endpoints packaged as a pair. -/
def breathe (a : α) : α × α :=
  (P.dial.box a, P.dial.diamond a)

/-- Promote the parameter to the next dimensional layer. -/
def elevate : DialParam α :=
  { dimension := P.dimension + 1
    dial := P.dial }

@[simp] lemma elevate_dimension :
    P.elevate.dimension = P.dimension + 1 := rfl

@[simp] lemma elevate_collapse :
    P.elevate.collapse = P.collapse := rfl

@[simp] lemma elevate_expand :
    P.elevate.expand = P.expand := rfl

/-- Ordering on dial parameters: compare dimension and the breathing bounds. -/
def le (P Q : DialParam α) : Prop :=
  P.dimension ≤ Q.dimension ∧
    ∀ a, P.dial.box a ≤ Q.dial.box a ∧ P.dial.diamond a ≤ Q.dial.diamond a

@[simp] lemma le_refl : P.le P := by
  refine ⟨le_rfl, ?_⟩
  intro a
  exact ⟨le_rfl, le_rfl⟩

lemma le_trans {Q R' : DialParam α} (hPQ : P.le Q) (hQR : Q.le R') :
    P.le R' := by
  refine ⟨Nat.le_trans hPQ.1 hQR.1, ?_⟩
  intro a
  obtain ⟨hBoxPQ, hDiamondPQ⟩ := hPQ.2 a
  obtain ⟨hBoxQR, hDiamondQR⟩ := hQR.2 a
  exact ⟨_root_.le_trans hBoxPQ hBoxQR, _root_.le_trans hDiamondPQ hDiamondQR⟩

/-- Elevating a dial parameter is monotone with respect to this ordering. -/
lemma le_elevate : P.le P.elevate := by
  refine ⟨Nat.le_succ _, ?_⟩
  intro a
  exact ⟨le_rfl, le_rfl⟩

def base (R : Reentry α) : DialParam α :=
  { dimension := 0
    dial := Dial.trivial R }

@[simp] lemma base_dimension (R : Reentry α) :
    (base R).dimension = 0 := rfl

def ladder (R : Reentry α) : ℕ → DialParam α
  | 0 => base R
  | n + 1 => (ladder R n).elevate

@[simp] lemma ladder_zero (R : Reentry α) :
    ladder R 0 = base R := rfl

@[simp] lemma ladder_succ (R : Reentry α) (n : ℕ) :
    ladder R (n + 1) = (ladder R n).elevate := rfl

/-- Collapse associated with the arithmetic ladder stage `n`. -/
def collapseAt (R : Reentry α) (n : ℕ) : α → α :=
  (ladder R n).collapse

/-- Expansion associated with the arithmetic ladder stage `n`. -/
def expandAt (R : Reentry α) (n : ℕ) : α → α :=
  (ladder R n).expand

lemma collapseAt_monotone (R : Reentry α) (n : ℕ) :
    Monotone (collapseAt (R := R) n) :=
  (ladder R n).collapse_monotone

/-- Stage `expandAt` is monotone because each step combines monotone nuclei. -/
lemma expandAt_monotone (R : Reentry α) (n : ℕ) :
    Monotone (expandAt (R := R) n) :=
  (ladder R n).expand_monotone

@[simp] lemma collapseAt_zero (R : Reentry α) :
    collapseAt (R := R) 0 = fun a => R a := by
  funext a
  unfold collapseAt
  simp [ladder_zero, base, collapse, Dial.collapse, Dial.trivial]

@[simp] lemma expandAt_zero (R : Reentry α) :
    expandAt (R := R) 0 = fun a => R a := by
  funext a
  unfold expandAt
  simp [ladder_zero, base, expand, Dial.expand, Dial.trivial]

@[simp] lemma collapseAt_succ (R : Reentry α) (n : ℕ) :
    collapseAt (R := R) (n + 1) = collapseAt (R := R) n := by
  unfold collapseAt
  simp [ladder_succ]

@[simp] lemma expandAt_succ (R : Reentry α) (n : ℕ) :
    expandAt (R := R) (n + 1) = expandAt (R := R) n := by
  unfold expandAt
  simp [ladder_succ]

/-! Dial integration with dialectic synthesis. -/

lemma collapseAt_eq_core (R : Reentry α) :
    ∀ n, collapseAt (R := R) n = fun a => R a
  | 0 => by funext a; simp [collapseAt_zero]
  | n+1 => by
      have ih := collapseAt_eq_core (R := R) n
      simpa [collapseAt, ladder_succ, ih]

lemma expandAt_eq_core (R : Reentry α) :
    ∀ n, expandAt (R := R) n = fun a => R a
  | 0 => by funext a; simp [expandAt_zero]
  | n+1 => by
      have ih := expandAt_eq_core (R := R) n
      simpa [expandAt, ladder_succ, ih]

@[simp] lemma collapseAt_synth (R : Reentry α) (n : ℕ) (T A : α) :
    collapseAt (R := R) n (HeytingLean.Logic.Dialectic.synth (R := R) T A)
      = HeytingLean.Logic.Dialectic.synth (R := R) T A := by
  have h := collapseAt_eq_core (R := R) n
  -- rewrite collapseAt to core and use idempotence of R
  simp [h, HeytingLean.Logic.Dialectic.synth] -- reduces to R (R (T ⊔ A)) = _

@[simp] lemma expandAt_synth (R : Reentry α) (n : ℕ) (T A : α) :
    expandAt (R := R) n (HeytingLean.Logic.Dialectic.synth (R := R) T A)
      = HeytingLean.Logic.Dialectic.synth (R := R) T A := by
  have h := expandAt_eq_core (R := R) n
  simp [h, HeytingLean.Logic.Dialectic.synth]

/-- The primordial oscillation as a canonical birth element. -/
def birth (R : Reentry α) : α :=
  HeytingLean.Logic.Dialectic.synth (R := R) (R.primordial) (R.counter)

@[simp] lemma collapseAt_birth (R : Reentry α) (n : ℕ) :
    collapseAt (R := R) n (birth (R := R)) = birth (R := R) := by
  simp [birth]

@[simp] lemma expandAt_birth (R : Reentry α) (n : ℕ) :
    expandAt (R := R) n (birth (R := R)) = birth (R := R) := by
  simp [birth]

@[simp] lemma ladder_dimension (R : Reentry α) :
    ∀ n, (ladder R n).dimension = n
  | 0 => rfl
  | Nat.succ n =>
      have ih := ladder_dimension R n
      calc
        (ladder R (Nat.succ n)).dimension
            = ((ladder R n).elevate).dimension := by rfl
        _ = (ladder R n).dimension + 1 := (ladder R n).elevate_dimension
        _ = n + 1 := by rw [ih]
        _ = Nat.succ n := (Nat.succ_eq_add_one n).symm

def one (R : Reentry α) : DialParam α := ladder R 1
def two (R : Reentry α) : DialParam α := ladder R 2
def three (R : Reentry α) : DialParam α := ladder R 3

lemma base_le_one (R : Reentry α) :
    (base R).le (one R) := by
  simpa [one, ladder] using (base R).le_elevate

inductive Stage
  | boolean
  | heyting
  | mv
  | effect
  | orthomodular
  | beyond
  deriving DecidableEq, Repr

/-- Successor relation on modal stages describing the breathing hierarchy. -/
def Stage.next : Stage → Stage
  | Stage.boolean => Stage.heyting
  | Stage.heyting => Stage.mv
  | Stage.mv => Stage.effect
  | Stage.effect => Stage.orthomodular
  | Stage.orthomodular => Stage.beyond
  | Stage.beyond => Stage.beyond

/-- Classify a natural number by stage depth in the modal ladder. -/
def stageOfNat : ℕ → Stage
  | 0 => Stage.boolean
  | 1 => Stage.heyting
  | 2 => Stage.mv
  | 3 => Stage.effect
  | 4 => Stage.orthomodular
  | _ => Stage.beyond

/-- Advancing the natural index moves to the successor stage. -/
lemma stageOfNat_succ (n : ℕ) :
    stageOfNat (Nat.succ n) = (stageOfNat n).next := by
  classical
  cases n with
  | zero =>
      simp [stageOfNat, Stage.next]
  | succ n₁ =>
      cases n₁ with
      | zero =>
          simp [stageOfNat, Stage.next]
      | succ n₂ =>
          cases n₂ with
          | zero =>
              simp [stageOfNat, Stage.next]
          | succ n₃ =>
              cases n₃ with
              | zero =>
                  simp [stageOfNat, Stage.next]
              | succ n₄ =>
                  cases n₄ with
                  | zero =>
                      simp [stageOfNat, Stage.next]
                  | succ n₅ =>
                      cases n₅ with
                      | zero => simp [stageOfNat, Stage.next]
                      | succ _ => simp [stageOfNat, Stage.next]

/-- All natural indices ≥ 5 fall into the `beyond` stage of the ladder. -/
lemma stageOfNat_add_five (k : ℕ) :
    stageOfNat (5 + k) = Stage.beyond := by
  refine Nat.rec ?base ?step k
  · simp [stageOfNat]
  · intro k ih
    have h := stageOfNat_succ (n := 5 + k)
    have hsucc :
        stageOfNat (Nat.succ (5 + k)) = Stage.beyond :=
      by
        simpa [Nat.succ_eq_add_one, Nat.add_comm, Nat.add_left_comm,
          Nat.add_assoc, Stage.next, ih] using h
    simpa [Nat.succ_eq_add_one, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using hsucc

/-- Coarsely classify a dial parameter by its dimension. -/
def stage (P : DialParam α) : Stage :=
  stageOfNat P.dimension

/-- The core process of any dial parameter yields a strictly positive element of Ω. -/
lemma process_pos (P : DialParam α) :
    ⊥ < ((P.dial.core.process : P.dial.core.Omega) : α) :=
  Reentry.process_pos (R := P.dial.core)

/-- The counter-process likewise yields a strictly positive element. -/
lemma counter_pos (P : DialParam α) :
    ⊥ < ((P.dial.core.counterProcess : P.dial.core.Omega) : α) :=
  Reentry.counter_pos (R := P.dial.core)

@[simp] lemma euler_boundary_coe (P : DialParam α) :
    ((P.dial.core.eulerBoundary : P.dial.core.Omega) : α)
      = P.dial.core.primordial := by
  simp [Reentry.eulerBoundary_eq_process, Reentry.process_coe]

/-- The Euler boundary coincides with the process witness inside Ω. -/
lemma euler_boundary_process (P : DialParam α) :
    P.dial.core.eulerBoundary = P.dial.core.process :=
  Reentry.eulerBoundary_eq_process (R := P.dial.core)

@[simp] lemma stage_base (R : Reentry α) :
    (base R).stage = Stage.boolean := rfl

@[simp] lemma stage_elevate (P : DialParam α) :
    (P.elevate).stage = (P.stage).next := by
  unfold stage
  have := stageOfNat_succ (n := P.dimension)
  simpa [elevate_dimension, Nat.succ_eq_add_one] using this

end DialParam

end Modal
end Logic
end HeytingLean
