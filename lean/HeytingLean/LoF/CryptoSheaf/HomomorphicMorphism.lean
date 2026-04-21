import HeytingLean.LoF.Nucleus
import HeytingLean.LoF.HeytingCore

/-
CryptoSheaf/HomomorphicMorphism

Homomorphic evaluation over the fixed-point Heyting algebra Ωᴿ for a LoF reentry nucleus.
The `Liftable` predicate captures closure-compatibility: `R ∘ f = R ∘ f ∘ R`.
-/

universe u

namespace HeytingLean
namespace LoF
namespace CryptoSheaf

open HeytingLean.LoF

variable {α : Type u} [PrimaryAlgebra α]

/-- Closure-compatibility of an operation with a reentry nucleus. -/
def Liftable (R : Reentry α) (f : α → α) : Prop :=
  ∀ x, R (f (R x)) = R (f x)

/-- A basic homomorphic scheme over Ωᴿ. -/
structure HomomorphicScheme (R : Reentry α) where
  /-- Encrypt: embed an ambient element by closing it and packaging as a fixed point. -/
  encrypt : α → R.Omega := fun a => Reentry.Omega.mk (R := R) (R a) (R.idempotent a)
  /-- Evaluate an operation on encrypted data (then re-close). -/
  eval : ∀ (f : α → α), Liftable R f → (R.Omega → R.Omega)
  /-- Correctness on plaintexts: decrypt ∘ eval f ∘ encrypt = R ∘ f. -/
  correct : ∀ (f : α → α) (hf : Liftable R f) (a : α),
    ((eval f hf (encrypt a) : R.Omega) : α) = R (f a)

namespace HomomorphicScheme

/-- Canonical homomorphic scheme induced by a reentry nucleus. -/
noncomputable def canonical (R : Reentry α) : HomomorphicScheme R where
  encrypt := fun a => Reentry.Omega.mk (R := R) (R a) (R.idempotent a)
  eval := fun f _ x => Reentry.Omega.mk (R := R) (R (f (x : α))) (R.idempotent _)
  correct := by
    intro f hf a
    -- eval f (encrypt a) = R (f (R a)) and use Liftable to rewrite to R (f a)
    change (R (f ((Reentry.Omega.mk (R := R) (R a) (R.idempotent a) : R.Omega) : α))) = R (f a)
    simp [Reentry.Omega.coe_mk]
    simpa using hf a

end HomomorphicScheme

/-- Implication on the right as an operation on the fixed-point algebra. -/
def himpRight (R : Reentry α) (b : R.Omega) : R.Omega → R.Omega :=
  fun a => a ⇨ b

/-- Evaluating `himpRight` on an encrypted input: the underlying element equals `(R a) ⇨ b`. -/
lemma himpRight_encrypt_coe (R : Reentry α) (a : α) (b : R.Omega) :
    ((himpRight R b ((HomomorphicScheme.canonical R).encrypt a)) : α)
      = (R a) ⇨ (b : α) := by
  -- By definitions and the `coe_himp` + `coe_mk` lemmas from HeytingCore.
  rfl

/-- If the right argument is closed, the map `x ↦ R x ⇨ y` is `Liftable`. -/
lemma liftable_himp_right_closed (R : Reentry α) {y : α} (hy : R y = y) :
    Liftable R (fun x => R x ⇨ y) := by
  intro x
  -- Show both sides reduce to `(R x) ⇨ y`.
  calc
    R ((R (R x)) ⇨ y) = (R (R x)) ⇨ y := by
      simpa [hy] using (Reentry.map_himp_apply (R := R) (a := R (R x)) (b := y))
    _ = (R x) ⇨ y := by simp
    _ = R (R x ⇨ y) := by
      symm
      simpa [hy] using (Reentry.map_himp_apply (R := R) (a := R x) (b := y))

/-- Meet with a fixed right argument is always liftable. -/
lemma liftable_inf_right (R : Reentry α) (y : α) :
    Liftable R (fun x => x ⊓ y) := by
  intro x
  -- R ((R x) ⊓ y) = R x ⊓ R y = R (x ⊓ y)
  simp [Reentry.map_inf (R := R) (a := R x) (b := y),
        Reentry.map_inf (R := R) (a := x) (b := y)]

/-- Join on the left is liftable (closure-compatibility). -/
lemma liftable_sup_left (R : Reentry α) {y : α} :
    Liftable R (fun x => x ⊔ y) := by
  intro x
  -- Left: R (R x ⊔ y) = R (x ⊔ y) by `map_sup_left_closed`.
  simp [Reentry.map_sup_left_closed (R := R) (a := x) (b := y)]

/-- Join on the right is also liftable. -/
lemma liftable_sup_right (R : Reentry α) {y : α} :
    Liftable R (fun x => y ⊔ x) := by
  intro x
  -- Right: R (y ⊔ R x) = R (y ⊔ x) by `map_sup_right_closed`.
  simp [Reentry.map_sup_right_closed (R := R) (a := y) (b := x)]

end CryptoSheaf
end LoF
end HeytingLean
