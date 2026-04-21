import HeytingLean.Boundary.Language.GradedTyping
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFloatingPoint

/-!
# Veselov FNS: HNS level hierarchy as Boundary grades
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

inductive FibHNSLevel where
  | integer
  | fractional (n : Nat)
  deriving DecidableEq, Repr

namespace FibHNSLevel

/-- Embed FNS levels into the existing Boundary grade chain. Integer payloads are
transparent; low fractional levels expose boundary precision; high fractional
levels may be demoted/opaque. -/
def toBoundaryGrade : FibHNSLevel -> HeytingLean.Boundary.Language.Grade
  | .integer => .top
  | .fractional 0 => .bot
  | .fractional 1 => .bot
  | .fractional _ => .mid

instance : LE FibHNSLevel where
  le a b := toBoundaryGrade a ≤ toBoundaryGrade b

instance : DecidableRel (· ≤ · : FibHNSLevel -> FibHNSLevel -> Prop) := fun a b => by
  change Decidable (toBoundaryGrade a ≤ toBoundaryGrade b)
  infer_instance

theorem boundary_grade_sub_identity (l : FibHNSLevel) :
    HeytingLean.Boundary.Language.Grade.boundary (toBoundaryGrade l) ≤ toBoundaryGrade l := by
  exact HeytingLean.Boundary.Language.Grade.boundary_sub_identity (toBoundaryGrade l)

end FibHNSLevel

/-- Truncate a Fibonacci float to the first `L` fractional levels. -/
def truncateToLevel (f : FibFloat) (L : Nat) : FibFloat :=
  FibFloat.truncate f L

theorem truncateToLevel_length_le (f : FibFloat) (L : Nat) :
    (truncateToLevel f L).fracPart.length ≤ L := by
  exact FibFloat.truncate_frac_length_le f L

/-- HNS truncation inherits the finite Fibonacci tail bound from the fractional
format: discarding levels after `L` costs no more than `1/F_(L+1)`. -/
theorem truncation_error_bound (f : FibFloat) (L : Nat) :
    |FibFloat.eval f - FibFloat.eval (truncateToLevel f L)| ≤ FibFloat.precisionBudget L := by
  simpa [truncateToLevel] using FibFloat.truncate_precision_bound_budget_all f L

end HeytingLean.Bridge.Veselov.HybridZeckendorf
