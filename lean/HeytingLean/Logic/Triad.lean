import HeytingLean.LoF.HeytingCore

/-!
# Logical triad operations on the Heyting core

This module provides small helper lemmas that exercise the generic Heyting API on the fixed-point
sublocale `Ω_R`.
-/

namespace HeytingLean
namespace Logic

open HeytingLean.LoF

variable {α : Type u} [PrimaryAlgebra α]

section Deduction

variable (R : Reentry α)

/-- Deduction is encoded as the Heyting adjunction specialised to the nucleus core. -/
def deduction (a b c : R.Omega) : Prop :=
  a ⊓ b ≤ c

/-- Abduction/implication equivalence inherited from the Heyting adjunction. -/
lemma deduction_iff (a b c : R.Omega) :
    deduction (R := R) a b c ↔ a ≤ b ⇨ c :=
  Reentry.heyting_adjunction (R := R) _ _ _

/-- Double negation witnesses a deduction into bottom, capturing collapse behaviour. -/
lemma double_neg_collapse (a : R.Omega) :
    deduction (R := R) a (a ⇨ (⊥ : R.Omega)) ⊥ :=
  (Reentry.heyting_adjunction (R := R) a (a ⇨ (⊥ : R.Omega)) ⊥).mpr
    (Reentry.double_neg (R := R) a)

end Deduction

end Logic
end HeytingLean
