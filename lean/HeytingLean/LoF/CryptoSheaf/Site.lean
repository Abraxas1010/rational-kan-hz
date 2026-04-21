import Mathlib.Order.Heyting.Basic
import Mathlib.Order.Heyting.Hom

namespace HeytingLean
namespace LoF
namespace CryptoSheaf

/-- A simplified site whose objects form a poset of contexts. -/
structure PosetSite (C : Type _) [PartialOrder C] where
  /-- Logical carrier at each context. -/
  Logic : C → Type _
  /-- Heyting structure for each context. -/
  heytingInst : ∀ U : C, HeytingAlgebra (Logic U)
  /-- Restriction map along the order relation (forgetting information). -/
  restrict : ∀ {U V : C}, V ≤ U → (Logic U → Logic V)
  /-- Restriction is a Heyting homomorphism (preserves structure). -/
  restrictHom : ∀ {U V : C}, V ≤ U → HeytingHom (Logic U) (Logic V)
  /-- Consistency: the underlying function of `restrictHom` equals `restrict`. -/
  restrictHom_toFun : ∀ {U V : C} (h : V ≤ U),
    (restrictHom (U := U) (V := V) h : Logic U → Logic V) = restrict (U := U) (V := V) h
  /-- Identity law for restriction. -/
  restrict_id : ∀ U (x : Logic U), restrict (U := U) (V := U) (le_rfl : U ≤ U) x = x
  /-- Composition law for restriction. -/
  restrict_comp : ∀ {U V W : C} (hVU : V ≤ U) (hWV : W ≤ V) (x : Logic U),
    restrict (U := V) (V := W) hWV (restrict (U := U) (V := V) hVU x)
      = restrict (U := U) (V := W) (le_trans hWV hVU) x

-- Provide an instance from the structure field.
instance instHeytingLogic (C : Type _) [PartialOrder C]
    (S : PosetSite C) (U : C) : HeytingAlgebra (S.Logic U) :=
  S.heytingInst U

end CryptoSheaf
end LoF
end HeytingLean
