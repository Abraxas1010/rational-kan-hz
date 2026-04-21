import HeytingLean.LoF.CryptoSheaf.Site
import HeytingLean.LoF.CryptoSheaf.Presheaf
import HeytingLean.LoF.CryptoSheaf.Descent

/-
CryptoSheaf/Examples/DescentTwoCover

Demonstrates `descent_exists` and `descent_unique` on a Bool poset site
with a trivial computation sheaf and a 2-element cover of `U = true`.
-/

namespace HeytingLean
namespace LoF
namespace CryptoSheaf
namespace Examples

namespace DescentTwoCover

def boolSite : PosetSite Bool where
  Logic := fun _ => Prop
  heytingInst := fun _ => inferInstance
  restrict := by intro U V h; exact id
  restrictHom := by intro U V h; exact HeytingHom.id _
  restrictHom_toFun := by intro U V h; rfl
  restrict_id := by intro U x; rfl
  restrict_comp := by intro U V W hVU hWV x; rfl

def trivialPresheaf : CompPresheaf boolSite where
  Sections := fun _ => Unit
  value := fun _ _ => True
  restrict := by intro U V h s; exact s
  restrict_id := by intro U s; cases s; rfl
  restrict_comp := by intro U V W hVU hWV s; cases s; rfl
  value_restrict := by intro U V h s; cases s; rfl

def trivialSheaf : CompSheaf boolSite :=
  { toCompPresheaf := trivialPresheaf,
    sheaf :=
      { glue := by intro U cover cover_le fam; exact (),
        glue_restrict := by intro U cover cover_le fam V hV; rfl,
        glue_unique := by intro U cover cover_le fam s hs; cases s; rfl } }

/-- Existence of a glued section over a 2-cover on Bool. -/
theorem descent_exists_twoCover :
    let U := true
    let cover := [true, false]
    let cover_le : ∀ V ∈ cover, V ≤ U := by intro V h; cases V <;> decide
    let fam : CompatibleFamily boolSite trivialPresheaf U cover cover_le :=
      { local_sections := fun _ _ => (),
        compatible := by intro _ _ _ _ _ _ _; rfl }
    ∃ s : trivialPresheaf.Sections U,
      ∀ V (hV : V ∈ cover), trivialPresheaf.restrict (U := U) (V := V) (cover_le V hV) s
        = fam.local_sections V hV := by
  intro U cover cover_le fam
  simpa using descent_exists (S := boolSite) (F := trivialSheaf)
    (U := U) (cover := cover) (cover_le := cover_le) fam

/-- Uniqueness of glued section over the 2-cover on Bool. -/
theorem descent_unique_twoCover :
    let U := true
    let cover := [true, false]
    let cover_le : ∀ V ∈ cover, V ≤ U := by intro V h; cases V <;> decide
    let fam : CompatibleFamily boolSite trivialPresheaf U cover cover_le :=
      { local_sections := fun _ _ => (),
        compatible := by intro _ _ _ _ _ _ _; rfl }
    ∀ (s : trivialPresheaf.Sections U),
      (∀ V (hV : V ∈ cover), trivialPresheaf.restrict (U := U) (V := V) (cover_le V hV) s
        = fam.local_sections V hV)
      → s = trivialSheaf.sheaf.glue U cover cover_le fam := by
  intro U cover cover_le fam s hs
  simpa using descent_unique (S := boolSite) (F := trivialSheaf)
    (U := U) (cover := cover) (cover_le := cover_le) fam s hs

end DescentTwoCover

end Examples
end CryptoSheaf
end LoF
end HeytingLean
