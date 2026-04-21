import HeytingLean.LoF.CryptoSheaf.HomomorphicMorphism
import HeytingLean.LoF.CryptoSheaf.Site
import HeytingLean.LoF.CryptoSheaf.Presheaf

/-
CryptoSheaf/Descent

Minimal data and a concrete "gluing" operation over Ωᴿ using finite meets/sups.
We keep this algebraic (poset-indexed site not required for the core definition).
-/

namespace HeytingLean
namespace LoF
namespace CryptoSheaf

open HeytingLean.LoF

variable {α : Type u} [PrimaryAlgebra α]

/-- Descent data: a finite multiset of local encrypted values in Ωᴿ. -/
structure NucleusDescentData (R : Reentry α) where
  locals : List R.Omega

namespace NucleusDescentData

variable (R : Reentry α)

/-- Glue by taking the meet (conservative aggregation). -/
def glueMeet (d : NucleusDescentData R) : R.Omega :=
  d.locals.foldl (· ⊓ ·) ⊤

/-- Glue by taking the join (liberal aggregation). -/
def glueJoin (d : NucleusDescentData R) : R.Omega :=
  d.locals.foldl (· ⊔ ·) ⊥

end NucleusDescentData

/-! ## Site-indexed descent wrappers

We provide small wrappers around the `IsSheaf` interface to expose existence and
uniqueness of glued global sections as explicit lemmas in a site-indexed form. -/

section SiteDescent

variable {C : Type _} [PartialOrder C]

variable (S : PosetSite C) (F : CompSheaf S)

/-- Existence of a global section gluing a compatible family over a cover. -/
theorem descent_exists (U : C) (cover : List C)
    (cover_le : ∀ V ∈ cover, V ≤ U)
    (fam : CompatibleFamily S F.toCompPresheaf U cover cover_le) :
    ∃ s : F.toCompPresheaf.Sections U,
      ∀ V (hV : V ∈ cover),
        F.toCompPresheaf.restrict (U := U) (V := V) (cover_le V hV) s
          = fam.local_sections V hV := by
  refine ⟨F.sheaf.glue U cover cover_le fam, ?_⟩
  intro V hV
  simpa using F.sheaf.glue_restrict U cover cover_le fam V hV

/-- Uniqueness of the glued global section under matching local restrictions. -/
theorem descent_unique (U : C) (cover : List C)
    (cover_le : ∀ V ∈ cover, V ≤ U)
    (fam : CompatibleFamily S F.toCompPresheaf U cover cover_le)
    (s : F.toCompPresheaf.Sections U)
    (h : ∀ V (hV : V ∈ cover),
      F.toCompPresheaf.restrict (U := U) (V := V) (cover_le V hV) s
        = fam.local_sections V hV) :
    s = F.sheaf.glue U cover cover_le fam := by
  simpa using F.sheaf.glue_unique U cover cover_le fam s h

end SiteDescent

end CryptoSheaf
end LoF
end HeytingLean
