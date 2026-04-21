import HeytingLean.Bridge.Veselov.HybridZeckendorf.DialecticaExecutable
import HeytingLean.CategoryTheory.PredicatedPolynomial.Indexed

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.CategoryTheory.PredicatedPolynomial

/-- Common-kernel predicated view of the supported Zeckendorf attack family at index `n`. -/
def supportedAttackPredPoly (n : Nat) : PredPoly where
  pos := Witness n
  dir := fun w => SupportedAttack w
  pred := fun w a => zeckendorfSupportedRefinement.realize w a

/-- Indexed wrapper over naturals; transport is only along definitional equality of indices. -/
def supportedAttackIndexed : IndexedPredPoly where
  Ctx := Nat
  fiber := supportedAttackPredPoly
  Rel := Eq
  reindex := by
    intro i j h
    cases h
    exact PredPoly.id (supportedAttackPredPoly i)

/-- The existing supported-attack realization theorem lives directly on the common kernel. -/
theorem supportedAttackPredPoly_sound (w : Witness n) (a : SupportedAttack w) :
    (supportedAttackPredPoly n).pred w a := by
  simpa [supportedAttackPredPoly] using zeckendorfSupportedRefinement_realizes w a

/-- Canonical Zeckendorf witnesses realize the common-kernel predicate at every supported attack. -/
theorem canonicalWitness_predicated_sound (n : Nat) (a : SupportedAttack (canonicalWitness n)) :
    (supportedAttackIndexed.fiber n).pred (canonicalWitness n) a := by
  simpa [supportedAttackIndexed, supportedAttackPredPoly] using
    canonicalWitness_refines_supported n a

/-- The indexed transport law specializes trivially along the equality-only index relation. -/
theorem supportedAttack_indexed_transport (n : Nat) (w : Witness n) (a : SupportedAttack w) :
    (supportedAttackIndexed.fiber n).pred w
      ((supportedAttackIndexed.reindex rfl).onDir w a) →
        (supportedAttackIndexed.fiber n).pred ((supportedAttackIndexed.reindex rfl).onPos w) a := by
  exact IndexedPredPoly.indexedAlpha_transport supportedAttackIndexed rfl w a

/-- The old executable checker is now wrapper glue around the common-kernel predicate family. -/
theorem checkCanonicalWitness_refactored (n : Nat) :
    checkSupportedAttacks (canonicalWitness n) = true ∧
      (∀ a : SupportedAttack (canonicalWitness n),
        (supportedAttackIndexed.fiber n).pred (canonicalWitness n) a) := by
  refine ⟨checkCanonicalWitness_eq_true n, ?_⟩
  intro a
  exact canonicalWitness_predicated_sound n a

end HeytingLean.Bridge.Veselov.HybridZeckendorf
