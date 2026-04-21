import HeytingLean.Bridge.Veselov.HybridZeckendorf.DialecticaSemantics
import HeytingLean.Bridge.Veselov.HybridZeckendorf.DialecticaAttackClasses

/-!
# Bridge.Veselov.HybridZeckendorf.DialecticaMorphisms

Project-local witness/challenge refinement surface for the Zeckendorf Dialectica family.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Boundary.Homomorphic

/-- Minimal project-local refinement object: witnesses indexed by `Nat`, with
witness-dependent challenges and a realization predicate. -/
structure NatWitnessRefinement where
  carrier : Nat -> Type
  challenge : {n : Nat} -> carrier n -> Type
  realize : {n : Nat} -> (w : carrier n) -> challenge w -> Prop

/-- The supported Zeckendorf refinement uses canonical Zeckendorf witnesses and
supported detectable zero-to-one flips. -/
def zeckendorfSupportedRefinement : NatWitnessRefinement where
  carrier := Witness
  challenge := fun {_} w => SupportedAttack w
  realize := fun {_} w a => plainCanonicalBits (singleBitFlip w.bits a.pos) = false

/-- The refinement relation is genuinely compatible with the Dialectica matrix,
not merely a definitional alias. -/
theorem zeckendorfSupportedRefinement_matrix_compatible (w : Witness n) (a : SupportedAttack w) :
    zeckendorfSupportedRefinement.realize w a -> alpha n w a.toAttack := by
  intro hreal
  left
  simpa [zeckendorfSupportedRefinement, alpha, runAttack, supportedAttack_toAttack_apply] using hreal

/-- Supported attacks are realized by the Zeckendorf refinement. -/
theorem zeckendorfSupportedRefinement_realizes (w : Witness n) (a : SupportedAttack w) :
    zeckendorfSupportedRefinement.realize w a := by
  simpa [zeckendorfSupportedRefinement] using supportedAttack_global_rejection w a

/-- The canonical witness refines each natural number against all supported attacks. -/
theorem canonicalWitness_refines_supported (n : Nat) (a : SupportedAttack (canonicalWitness n)) :
    zeckendorfSupportedRefinement.realize (canonicalWitness n) a :=
  zeckendorfSupportedRefinement_realizes (canonicalWitness n) a

end HeytingLean.Bridge.Veselov.HybridZeckendorf
