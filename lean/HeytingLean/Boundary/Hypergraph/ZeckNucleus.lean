import HeytingLean.Boundary.Hypergraph.ZeckRewriteNet
import HeytingLean.Core.Nucleus

namespace HeytingLean.Boundary.Hypergraph

/-!
Closed-subnet nucleus boundary for the Zeckendorf rewrite carrier.

Important correction to the PM sketch: `polyadicCheckOperator core S = S ∩ core`
is contractive under ordinary subset order, so it is not a `Core.Nucleus`
operator there. The verified nucleus below uses the closure operator
`S ↦ S ∪ core`. We still expose the `polyadicCheck` meet-preservation theorem
as the true CHECK/intersection fact.
-/

namespace ClosedSubNetOrder

variable {AgentId : Type*} [DecidableEq AgentId] {N : HNet AgentId}

instance : LE (ClosedSubNet N) where
  le A B := A.carrier ⊆ B.carrier

theorem le_def (A B : ClosedSubNet N) :
    A ≤ B ↔ A.carrier ⊆ B.carrier :=
  Iff.rfl

instance : PartialOrder (ClosedSubNet N) where
  le := (·.carrier ⊆ ·.carrier)
  le_refl := by
    intro A x hx
    exact hx
  le_trans := by
    intro A B C hAB hBC x hx
    exact hBC (hAB hx)
  le_antisymm := by
    intro A B hAB hBA
    cases A with
    | mk A hA =>
      cases B with
      | mk B hB =>
        have hset : A = B := Finset.Subset.antisymm hAB hBA
        cases hset
        rfl

instance : SemilatticeInf (ClosedSubNet N) where
  inf := ClosedSubNet.inter
  inf_le_left := by
    intro A B x hx
    exact (Finset.mem_inter.mp hx).1
  inf_le_right := by
    intro A B x hx
    exact (Finset.mem_inter.mp hx).2
  le_inf := by
    intro A B C hAB hAC x hx
    exact Finset.mem_inter.mpr ⟨hAB hx, hAC hx⟩
  le_refl := le_refl
  le_trans := fun _ _ _ => le_trans
  le_antisymm := fun _ _ => le_antisymm

instance : OrderBot (ClosedSubNet N) where
  bot := ClosedSubNet.empty N
  bot_le := by
    intro A x hx
    simp [ClosedSubNet.empty, ClosedSubNet.carrier] at hx

end ClosedSubNetOrder

open ClosedSubNetOrder

/-- Closure by adjoining the chosen core subnet. -/
def zeckClosureOperator {maxPos : Nat}
    (core : ClosedSubNet (zeckRewriteNet maxPos))
    (S : ClosedSubNet (zeckRewriteNet maxPos)) :
    ClosedSubNet (zeckRewriteNet maxPos) :=
  ClosedSubNet.union S core

/-- A genuine `Core.Nucleus` under the ordinary subset order: `S ↦ S ∪ core`. -/
noncomputable def zeckClosedSubNetNucleus (maxPos : Nat)
    (core : ClosedSubNet (zeckRewriteNet maxPos) := zeckRewriteCore maxPos) :
    HeytingLean.Core.Nucleus (ClosedSubNet (zeckRewriteNet maxPos)) where
  R := zeckClosureOperator core
  extensive := by
    intro S x hx
    exact Finset.mem_union.mpr (Or.inl hx)
  idempotent := by
    intro S
    apply Subtype.ext
    ext x
    simp [zeckClosureOperator, ClosedSubNet.union, Finset.mem_union]
  meet_preserving := by
    intro A B
    change zeckClosureOperator core (ClosedSubNet.inter A B) =
      ClosedSubNet.inter (zeckClosureOperator core A) (zeckClosureOperator core B)
    apply Subtype.ext
    ext x
    change x ∈ ((A.carrier ∩ B.carrier) ∪ core.carrier) ↔
      x ∈ ((A.carrier ∪ core.carrier) ∩ (B.carrier ∪ core.carrier))
    simp [Finset.mem_union]
    tauto

/-- The existing CHECK operator still has its verified meet-preservation carrier theorem. -/
theorem zeckPolyadicCheck_meet_preserving_closed (maxPos : Nat)
    (core A B : ClosedSubNet (zeckRewriteNet maxPos)) :
    (polyadicCheckOperator core (ClosedSubNet.inter A B)).carrier =
      (ClosedSubNet.inter (polyadicCheckOperator core A)
        (polyadicCheckOperator core B)).carrier :=
  polyadicCheck_meet_preserving_closed core A B

end HeytingLean.Boundary.Hypergraph
