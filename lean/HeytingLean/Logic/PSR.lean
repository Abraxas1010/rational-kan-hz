import Mathlib.Logic.Relation
import HeytingLean.LoF.Nucleus
import HeytingLean.Epistemic.Occam

namespace HeytingLean
namespace Logic
namespace PSR

open HeytingLean.LoF Relation
open Epistemic
open scoped Classical

variable {α : Type u} [PrimaryAlgebra α]

/-- The Principle of Sufficient Reason: a proposition is sufficient precisely when it is invariant
under the re-entry nucleus. -/
def Sufficient (R : Reentry α) (a : α) : Prop :=
  R a = a

@[simp] lemma sufficient_iff (R : Reentry α) (a : α) :
    Sufficient R a ↔ R a = a := Iff.rfl

lemma sufficient_of_fixed (R : Reentry α) (a : α)
    (ha : R a = a) : Sufficient R a :=
  ha

lemma fixed_of_sufficient (R : Reentry α) {a : α}
    (ha : Sufficient R a) : R a = a :=
  ha

/-- Single-step evolution induced by the re-entry nucleus. -/
def Step (R : Reentry α) (x y : α) : Prop :=
  y = R x

/-- Reachability obtained by repeatedly applying the nucleus. -/
def reachable (R : Reentry α) (x y : α) : Prop :=
  Relation.ReflTransGen (Step (R := R)) x y

@[simp] lemma reachable_refl (R : Reentry α) (x : α) :
    reachable (R := R) x x :=
  Relation.ReflTransGen.refl

lemma reachable_step (R : Reentry α) {x y : α}
    (h : Step (R := R) x y) :
    reachable (R := R) x y :=
  Relation.ReflTransGen.single h

lemma reachable_trans (R : Reentry α) {x y z : α}
    (hxy : reachable (R := R) x y) (hyz : reachable (R := R) y z) :
    reachable (R := R) x z :=
  Relation.ReflTransGen.trans hxy hyz

/-- Every finite breath corresponds to a reachable state. -/
lemma reachable_of_breathe (R : Reentry α) (x : α) :
    ∀ n, reachable (R := R) x (Epistemic.breathe (R := R) n x)
  | 0 => reachable_refl (R := R) x
  | Nat.succ n =>
      by
        have h := reachable_of_breathe (R := R) (x := x) n
        have hstep :
            Step (R := R) (Epistemic.breathe (R := R) n x)
              (Epistemic.breathe (R := R) (n + 1) x) := rfl
        exact reachable_trans (R := R) h (reachable_step (R := R) hstep)

/-- Reachability is equivalent to breathing a finite number of times. -/
lemma reachable_iff_exists_breathe (R : Reentry α) (x y : α) :
    reachable (R := R) x y ↔
      ∃ n, Epistemic.breathe (R := R) n x = y := by
  constructor
  · intro h
    induction h with
    | refl => exact ⟨0, rfl⟩
    | tail hxy hyz ih =>
        obtain ⟨n, hn⟩ := ih
        cases hyz
        refine ⟨n + 1, ?_⟩
        simp [Epistemic.breathe, hn]
  · rintro ⟨n, hn⟩
    have := reachable_of_breathe (R := R) (x := x) n
    simpa [hn] using this

/-- A sufficient reason remains valid along every breath of the nucleus. -/
lemma breathe_le_of_sufficient (R : Reentry α) {a x : α}
    (ha : Sufficient R a) (hx : x ≤ a) :
    ∀ n, Epistemic.breathe (R := R) n x ≤ a
  | 0 => by simpa [Epistemic.breathe] using hx
  | Nat.succ n =>
      by
        have ih := breathe_le_of_sufficient (R := R) (a := a) (x := x) ha hx n
        have hmono := R.monotone ih
        have ha' : R a = a := ha
        simpa [Epistemic.breathe, ha'] using hmono

/-- Sufficient reasons are stable along the reachability relation. -/
lemma sufficient_reachable (R : Reentry α) {a x y : α}
    (ha : Sufficient R a) (hx : x ≤ a)
    (hy : reachable (R := R) x y) :
    y ≤ a := by
  obtain ⟨n, rfl⟩ :=
    (reachable_iff_exists_breathe (R := R) (x := x) (y := y)).mp hy
  exact breathe_le_of_sufficient (R := R) (a := a) (x := x) ha hx _

lemma sufficient_stable (R : Reentry α) {a x : α}
    (ha : Sufficient R a) (hx : x ≤ a) :
    R x ≤ a :=
  sufficient_reachable (R := R) (a := a)
    (x := x) (y := R x) ha hx (reachable_step (R := R) rfl)

/-- Minimal reasons exist at each dial: the Occam reduction is invariant. -/
lemma occam_sufficient (R : Reentry α) (a : α) :
    Sufficient R (Epistemic.occam (R := R) a) :=
  occam_idempotent (R := R) (a := a)

@[simp] lemma sufficient_eulerBoundary (R : Reentry α) :
    Sufficient R ((R.eulerBoundary : R.Omega) : α) :=
  Reentry.Omega.apply_coe (R := R) (a := R.eulerBoundary)

/-- Minimal reason witness: for any `x`, the invariant `R x` contains `x` and has birthday 0. -/
lemma minimal_reason_exists (R : Reentry α) (x : α) :
    ∃ u, Sufficient R u ∧ x ≤ u ∧ Epistemic.birth R u = 0 := by
  refine ⟨R x, ?_, ?_, ?_⟩
  · exact Reentry.idempotent (R := R) (a := x)
  · exact Reentry.le_apply (R := R) (a := x)
  · have : R (R x) = R x := Reentry.idempotent (R := R) (a := x)
    exact Epistemic.birth_eq_zero_of_fixed (R := R) (a := R x) this

end PSR
end Logic
end HeytingLean
