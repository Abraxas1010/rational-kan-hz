import HeytingLean.LoF.CryptoSheaf.Basic
import HeytingLean.LoF.CryptoSheaf.Site

namespace HeytingLean
namespace LoF
namespace CryptoSheaf

variable {C : Type _} [PartialOrder C]

/-- A presheaf of computations over a poset site. -/
structure CompPresheaf (S : PosetSite C) where
  /-- Computations available at each context. -/
  Sections : C → Type _
  /-- The logical value/meaning of a computation. -/
  value : ∀ U : C, Sections U → S.Logic U
  /-- Restriction of computations along `V ≤ U`. -/
  restrict : ∀ {U V : C}, V ≤ U → Sections U → Sections V
  /-- Identity law for restriction on computations. -/
  restrict_id : ∀ U (s : Sections U),
    restrict (U := U) (V := U) (le_rfl : U ≤ U) s = s
  /-- Composition law for restriction on computations. -/
  restrict_comp : ∀ {U V W : C} (hVU : V ≤ U) (hWV : W ≤ V) (s : Sections U),
    restrict (U := V) (V := W) hWV (restrict (U := U) (V := V) hVU s)
      = restrict (U := U) (V := W) (le_trans hWV hVU) s
  /-- Value commutes with restriction. -/
  value_restrict : ∀ {U V : C} (h : V ≤ U) (s : Sections U),
    S.restrict (U := U) (V := V) h (value U s)
      = value V (restrict (U := U) (V := V) h s)

/-- A family of local sections over a cover that agree on overlaps. -/
structure CompatibleFamily (S : PosetSite C) (F : CompPresheaf S)
    (U : C) (cover : List C) (cover_le : ∀ V ∈ cover, V ≤ U) where
  /-- Local sections for each element of the cover. -/
  local_sections : ∀ V, V ∈ cover → F.Sections V
  /-- Compatibility on overlaps (expressed using restrictions to any `Z ≤ V, Z ≤ W`). -/
  compatible : ∀ V W (hV : V ∈ cover) (hW : W ∈ cover)
      (Z : C) (hZV : Z ≤ V) (hZW : Z ≤ W),
      F.restrict (U := V) (V := Z) hZV (local_sections V hV)
        = F.restrict (U := W) (V := Z) hZW (local_sections W hW)

/-- The sheaf condition: compatible families glue uniquely. -/
structure IsSheaf (S : PosetSite C) (F : CompPresheaf S) where
  /-- Gluing: every compatible family admits a global section. -/
  glue : ∀ (U : C) (cover : List C) (cover_le : ∀ V ∈ cover, V ≤ U),
    CompatibleFamily S F U cover cover_le → F.Sections U
  /-- The glued section restricts to each local section. -/
  glue_restrict : ∀ (U : C) (cover : List C) (cover_le : ∀ V ∈ cover, V ≤ U)
      (fam : CompatibleFamily S F U cover cover_le) (V : C) (hV : V ∈ cover),
      F.restrict (U := U) (V := V) (cover_le V hV) (glue U cover cover_le fam)
        = fam.local_sections V hV
  /-- Uniqueness of gluing. -/
  glue_unique : ∀ (U : C) (cover : List C) (cover_le : ∀ V ∈ cover, V ≤ U)
      (fam : CompatibleFamily S F U cover cover_le) (s : F.Sections U),
      (∀ V (hV : V ∈ cover),
        F.restrict (U := U) (V := V) (cover_le V hV) s
          = fam.local_sections V hV)
        → s = glue U cover cover_le fam

/-- A computation sheaf is a presheaf satisfying the sheaf condition. -/
structure CompSheaf (S : PosetSite C) extends CompPresheaf S where
  sheaf : IsSheaf S toCompPresheaf

end CryptoSheaf
end LoF
end HeytingLean
