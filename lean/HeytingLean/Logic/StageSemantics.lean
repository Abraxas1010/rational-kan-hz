import HeytingLean.Logic.ModalDial
import HeytingLean.Logic.ResiduatedLadder
import HeytingLean.Logic.Triad

/-
# Stage semantics transport

This module packages staged (MV / effect / orthomodular) operations on the Heyting core together
with a reusable transport interface for bridges.  The intent is to keep a single source of truth
for the operations on the core `Ω_R` and to expose helpers that automatically commute with the
round-trip data supplied by bridges.
-/

namespace HeytingLean
namespace Logic
namespace Stage

open HeytingLean.LoF
open scoped Classical

universe u

/-- Łukasiewicz / MV-style structure available on the core carrier. -/
class MvCore (Ω : Type u) where
  mvAdd : Ω → Ω → Ω
  mvNeg : Ω → Ω
  zero  : Ω
  one   : Ω

/-- Effect-algebra style structure with partial addition. -/
class EffectCore (Ω : Type u) where
  effectAdd? : Ω → Ω → Option Ω
  compat     : Ω → Ω → Prop
  orthosupp  : Ω → Ω
  zero       : Ω
  one        : Ω
  compat_iff_defined :
    ∀ u v, compat u v ↔ (effectAdd? u v).isSome

/-- Orthomodular lattice façade (no laws recorded yet, only operations). -/
class OmlCore (Ω : Type u) where
  meet : Ω → Ω → Ω
  join : Ω → Ω → Ω
  compl : Ω → Ω
  bot : Ω
  top : Ω

variable {α : Type u} [PrimaryAlgebra α]

namespace DialParam

variable (P : Modal.DialParam α)

/-- MV-style addition realised via the Heyting join. -/
@[simp] def mvAdd (a b : P.dial.core.Omega) : P.dial.core.Omega :=
  a ⊔ b

/-- MV-style negation realised via implication into bottom. -/
@[simp] def mvNeg (a : P.dial.core.Omega) : P.dial.core.Omega :=
  a ⇨ (⊥ : P.dial.core.Omega)

/-- MV-stage zero. -/
@[simp] def mvZero : P.dial.core.Omega := ⊥
/-- MV-stage one. -/
@[simp] def mvOne : P.dial.core.Omega := ⊤

/-- Effect-compatibility predicate (disjointness). -/
def effectCompatible (a b : P.dial.core.Omega) : Prop :=
  a ⊓ b = ⊥

/-- Partial effect-style addition returning a value only on compatible arguments. -/
noncomputable def effectAdd? (a b : P.dial.core.Omega) :
    Option P.dial.core.Omega :=
  if _ : DialParam.effectCompatible (P := P) a b then
    some (DialParam.mvAdd (P := P) a b)
  else
    none

/-- Orthocomplement induced by Heyting negation. -/
@[simp] def orthocomplement (a : P.dial.core.Omega) :
    P.dial.core.Omega :=
  DialParam.mvNeg (P := P) a

/-- Orthomodular meet (plain Heyting meet). -/
@[simp] def omlMeet (a b : P.dial.core.Omega) : P.dial.core.Omega := a ⊓ b

/-- Orthomodular join (plain Heyting join). -/
@[simp] def omlJoin (a b : P.dial.core.Omega) : P.dial.core.Omega := a ⊔ b

/-- Orthomodular bottom element. -/
@[simp] def omlBot : P.dial.core.Omega := ⊥
/-- Orthomodular top element. -/
@[simp] def omlTop : P.dial.core.Omega := ⊤

instance instMvCore : MvCore P.dial.core.Omega where
  mvAdd := DialParam.mvAdd (P := P)
  mvNeg := DialParam.mvNeg (P := P)
  zero := DialParam.mvZero (P := P)
  one := DialParam.mvOne (P := P)

noncomputable instance instEffectCore : EffectCore P.dial.core.Omega where
  effectAdd? := DialParam.effectAdd? (P := P)
  compat := DialParam.effectCompatible (P := P)
  orthosupp := DialParam.orthocomplement (P := P)
  zero := DialParam.mvZero (P := P)
  one := DialParam.mvOne (P := P)
  compat_iff_defined := by
    intro u v
    classical
    unfold DialParam.effectAdd? DialParam.effectCompatible
    by_cases h : u ⊓ v = (⊥ : P.dial.core.Omega)
    · simp [h]
    · simp [h]

instance instOmlCore : OmlCore P.dial.core.Omega where
  meet := DialParam.omlMeet (P := P)
  join := DialParam.omlJoin (P := P)
  compl := DialParam.orthocomplement (P := P)
  bot := DialParam.omlBot (P := P)
  top := DialParam.omlTop (P := P)

@[simp] lemma mvAdd_zero_left (a : P.dial.core.Omega) :
    DialParam.mvAdd (P := P) (DialParam.mvZero (P := P)) a = a := by
  simp [DialParam.mvAdd, DialParam.mvZero]

@[simp] lemma mvAdd_zero_right (a : P.dial.core.Omega) :
    DialParam.mvAdd (P := P) a (DialParam.mvZero (P := P)) = a := by
  simp [DialParam.mvAdd, DialParam.mvZero]

@[simp] lemma mvAdd_comm (a b : P.dial.core.Omega) :
    DialParam.mvAdd (P := P) a b =
      DialParam.mvAdd (P := P) b a := by
  simp [DialParam.mvAdd, sup_comm]

lemma effectCompatible_orthocomplement
    (a : P.dial.core.Omega) :
    DialParam.effectCompatible (P := P) a
        (DialParam.orthocomplement (P := P) a) := by
  unfold DialParam.effectCompatible DialParam.orthocomplement DialParam.mvNeg
  apply le_antisymm
  · exact HeytingLean.Logic.double_neg_collapse (R := P.dial.core) (a := a)
  · exact bot_le

@[simp] lemma effectAdd?_orthocomplement
    (a : P.dial.core.Omega) :
    DialParam.effectAdd? (P := P) a
        (DialParam.orthocomplement (P := P) a) =
          some
            (DialParam.mvAdd (P := P) a
              (DialParam.orthocomplement (P := P) a)) := by
  classical
  unfold DialParam.effectAdd?
  simp [DialParam.effectCompatible,
    DialParam.orthocomplement, DialParam.mvNeg]

@[simp] lemma himp_closed (a b : P.dial.core.Omega) :
    P.dial.core ((a : α) ⇨ (b : α)) = (a : α) ⇨ (b : α) :=
  HeytingLean.Logic.Residuated.himp_closed
    (R := P.dial.core) (a := a) (b := b)

lemma map_himp_le (a b : α) :
    P.dial.core (a ⇨ b) ≤ a ⇨ P.dial.core b :=
  HeytingLean.Logic.Residuated.map_himp_le
    (R := P.dial.core) (a := a) (b := b)

/-- At every ladder stage the modal collapse coincides with the core re-entry. -/
@[simp] lemma collapseAt_eq_reentry (R : Reentry α) :
    ∀ n, HeytingLean.Logic.Modal.DialParam.collapseAt (R := R) n =
        fun a => R a
  | 0 =>
      by
        funext a
        simp [HeytingLean.Logic.Modal.DialParam.collapseAt_zero]
  | Nat.succ n =>
      by
        have ih := collapseAt_eq_reentry (R := R) n
        funext a
        simp [HeytingLean.Logic.Modal.DialParam.collapseAt_succ,
          ih]

/-- At every ladder stage the modal expansion collapses back to the core re-entry. -/
@[simp] lemma expandAt_eq_reentry (R : Reentry α) :
    ∀ n, HeytingLean.Logic.Modal.DialParam.expandAt (R := R) n =
        fun a => R a
  | 0 =>
      by
        funext a
        simp [HeytingLean.Logic.Modal.DialParam.expandAt_zero]
  | Nat.succ n =>
      by
        have ih := expandAt_eq_reentry (R := R) n
        funext a
        simp [HeytingLean.Logic.Modal.DialParam.expandAt_succ,
          ih]

/-- Ladder-level collapse promoted to the Heyting core `Ω_R`. -/
noncomputable def collapseAtOmega (R : Reentry α) (n : ℕ) :
    R.Omega → R.Omega :=
  fun a =>
    Reentry.Omega.mk (R := R)
      (HeytingLean.Logic.Modal.DialParam.collapseAt
        (α := α) (R := R) n (a : α))
      (by
        simp [collapseAt_eq_reentry])

/-- Ladder-level expansion promoted to the Heyting core `Ω_R`. -/
noncomputable def expandAtOmega (R : Reentry α) (n : ℕ) :
    R.Omega → R.Omega :=
  fun a =>
    Reentry.Omega.mk (R := R)
      (R
        (HeytingLean.Logic.Modal.DialParam.expandAt
          (α := α) (R := R) n (a : α)))
      (by
        simp)

@[simp] lemma collapseAtOmega_coe (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    ((collapseAtOmega (α := α) (R := R) n a : R.Omega) : α) =
      HeytingLean.Logic.Modal.DialParam.collapseAt
        (α := α) (R := R) n (a : α) := rfl

@[simp] lemma expandAtOmega_coe (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    ((expandAtOmega (α := α) (R := R) n a : R.Omega) : α) =
      R
        (HeytingLean.Logic.Modal.DialParam.expandAt
          (α := α) (R := R) n (a : α)) := rfl

lemma collapseAtOmega_monotone (R : Reentry α) (n : ℕ) :
    Monotone (collapseAtOmega (α := α) (R := R) n) := by
  intro a b h
  change
      HeytingLean.Logic.Modal.DialParam.collapseAt (α := α)
          (R := R) n (a : α)
        ≤
      HeytingLean.Logic.Modal.DialParam.collapseAt (α := α)
          (R := R) n (b : α)
  exact
    HeytingLean.Logic.Modal.DialParam.collapseAt_monotone
      (α := α) (R := R) n h

lemma expandAtOmega_monotone (R : Reentry α) (n : ℕ) :
    Monotone (expandAtOmega (α := α) (R := R) n) := by
  intro a b h
  change
      R
          (HeytingLean.Logic.Modal.DialParam.expandAt (α := α)
              (R := R) n (a : α))
        ≤
      R
          (HeytingLean.Logic.Modal.DialParam.expandAt (α := α)
              (R := R) n (b : α))
  exact
    Reentry.monotone (R := R)
      (HeytingLean.Logic.Modal.DialParam.expandAt_monotone
        (α := α) (R := R) n h)

@[simp] lemma collapseAtOmega_self (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    collapseAtOmega (α := α) (R := R) n a = a := by
  ext
  simp [collapseAtOmega, collapseAt_eq_reentry,
    Reentry.Omega.apply_coe]

@[simp] lemma expandAtOmega_self (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    expandAtOmega (α := α) (R := R) n a = a := by
  ext
  simp [expandAtOmega, expandAt_eq_reentry,
    Reentry.Omega.apply_coe]

/-- MV-stage collapse law on the Heyting core. -/
@[simp] lemma mvCollapse_self (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder
        (α := α) R 2).dial.core.Omega) :
    collapseAtOmega (α := α) (R := R) 2 a = a :=
  collapseAtOmega_self (α := α) (R := R) 2 a

/-- MV-stage expansion law on the Heyting core. -/
@[simp] lemma mvExpand_self (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder
        (α := α) R 2).dial.core.Omega) :
    expandAtOmega (α := α) (R := R) 2 a = a :=
  expandAtOmega_self (α := α) (R := R) 2 a

/-- Effect-stage collapse law on the Heyting core. -/
@[simp] lemma effectCollapse_self (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder
        (α := α) R 3).dial.core.Omega) :
    collapseAtOmega (α := α) (R := R) 3 a = a :=
  collapseAtOmega_self (α := α) (R := R) 3 a

/-- Effect-stage expansion law on the Heyting core. -/
@[simp] lemma effectExpand_self (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder
        (α := α) R 3).dial.core.Omega) :
    expandAtOmega (α := α) (R := R) 3 a = a :=
  expandAtOmega_self (α := α) (R := R) 3 a

/-- Orthomodular-stage collapse law on the Heyting core. -/
@[simp] lemma orthCollapse_self (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder
        (α := α) R 4).dial.core.Omega) :
    collapseAtOmega (α := α) (R := R) 4 a = a :=
  collapseAtOmega_self (α := α) (R := R) 4 a

/-- Orthomodular-stage expansion law on the Heyting core. -/
@[simp] lemma orthExpand_self (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder
        (α := α) R 4).dial.core.Omega) :
    expandAtOmega (α := α) (R := R) 4 a = a :=
  expandAtOmega_self (α := α) (R := R) 4 a

end DialParam

/-- Bridges expose shadow/lift data satisfying a round-trip contract. -/
structure Bridge (α Ω : Type u) [LE α] [LE Ω] where
  shadow : α → Ω
  lift : Ω → α
  rt₁ : ∀ u, shadow (lift u) = u
  rt₂ : ∀ x, lift (shadow x) ≤ x

namespace Bridge

variable {α Ω : Type u} [LE α] [LE Ω] (B : Bridge α Ω)

/-- Transport MV addition across a bridge. -/
def stageMvAdd [MvCore Ω] (x y : α) : α :=
  B.lift (MvCore.mvAdd (B.shadow x) (B.shadow y))

/-- Transport MV negation across a bridge. -/
def stageMvNeg [MvCore Ω] (x : α) : α :=
  B.lift (MvCore.mvNeg (B.shadow x))

/-- Transport MV zero. -/
def stageMvZero [MvCore Ω] : α :=
  B.lift (MvCore.zero (Ω := Ω))

/-- Transport MV one. -/
def stageMvOne [MvCore Ω] : α :=
  B.lift (MvCore.one (Ω := Ω))

/-- Transport effect compatibility across a bridge. -/
def stageEffectCompatible [EffectCore Ω] (x y : α) : Prop :=
  EffectCore.compat (B.shadow x) (B.shadow y)

/-- Transport partial effect addition across a bridge. -/
def stageEffectAdd? [EffectCore Ω] (x y : α) : Option α :=
  (EffectCore.effectAdd? (B.shadow x) (B.shadow y)).map B.lift

/-- Transport the effect orthosupplement across a bridge. -/
def stageOrthosupp [EffectCore Ω] (x : α) : α :=
  B.lift (EffectCore.orthosupp (B.shadow x))

/-- Transport orthocomplement across a bridge. -/
def stageOrthocomplement [OmlCore Ω] (x : α) : α :=
  B.lift (OmlCore.compl (B.shadow x))

/-- Transport Heyting implication across a bridge. -/
def stageHimp [HeytingAlgebra Ω] (x y : α) : α :=
  B.lift ((B.shadow x) ⇨ (B.shadow y))

/-- Transport a collapse operator across a bridge. -/
def stageCollapse (collapse : Ω → Ω) (x : α) : α :=
  B.lift (collapse (B.shadow x))

/-- Transport an expansion operator across a bridge. -/
def stageExpand (expand : Ω → Ω) (x : α) : α :=
  B.lift (expand (B.shadow x))

@[simp] theorem shadow_stageMvAdd [MvCore Ω] (x y : α) :
    B.shadow (B.stageMvAdd x y) =
      MvCore.mvAdd (B.shadow x) (B.shadow y) := by
  unfold stageMvAdd
  exact B.rt₁ _

@[simp] theorem shadow_stageMvNeg [MvCore Ω] (x : α) :
    B.shadow (B.stageMvNeg x) = MvCore.mvNeg (B.shadow x) := by
  unfold stageMvNeg
  exact B.rt₁ _

@[simp] theorem stageEffectAdd?_isSome [EffectCore Ω] (x y : α) :
    (B.stageEffectAdd? x y).isSome ↔
      B.stageEffectCompatible x y := by
  unfold stageEffectAdd? stageEffectCompatible
  have h := EffectCore.compat_iff_defined (Ω := Ω)
  specialize h (B.shadow x) (B.shadow y)
  cases h' : EffectCore.effectAdd? (B.shadow x) (B.shadow y) with
  | none =>
      simp [Option.isSome, h', Option.map, h] at *
  | some w =>
      simp [Option.isSome, h', Option.map, h] at *

@[simp] theorem shadow_stageEffectAdd?_map [EffectCore Ω] (x y : α) :
    (B.stageEffectAdd? x y).map B.shadow =
      EffectCore.effectAdd? (B.shadow x) (B.shadow y) := by
  unfold stageEffectAdd?
  cases h : EffectCore.effectAdd? (B.shadow x) (B.shadow y) with
  | none =>
      simp
  | some w =>
      simp [B.rt₁]

@[simp] theorem shadow_stageOrthosupp [EffectCore Ω] (x : α) :
    B.shadow (B.stageOrthosupp x) =
      EffectCore.orthosupp (B.shadow x) := by
  unfold stageOrthosupp
  exact B.rt₁ _

@[simp] theorem shadow_stageOrthocomplement [OmlCore Ω] (x : α) :
    B.shadow (B.stageOrthocomplement x) =
      OmlCore.compl (B.shadow x) := by
  unfold stageOrthocomplement
  exact B.rt₁ _

@[simp] theorem shadow_stageHimp [HeytingAlgebra Ω] (x y : α) :
    B.shadow (B.stageHimp x y) =
      B.shadow x ⇨ B.shadow y := by
  unfold stageHimp
  exact B.rt₁ _

@[simp] theorem shadow_stageCollapse (collapse : Ω → Ω) (x : α) :
    B.shadow (B.stageCollapse collapse x) =
      collapse (B.shadow x) := by
  unfold stageCollapse
  exact B.rt₁ _

@[simp] theorem shadow_stageExpand (expand : Ω → Ω) (x : α) :
    B.shadow (B.stageExpand expand x) =
      expand (B.shadow x) := by
  unfold stageExpand
  exact B.rt₁ _

end Bridge

end Stage
end Logic
end HeytingLean
