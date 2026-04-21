import Mathlib.Algebra.Star.Basic
import HeytingLean.Bridges.Clifford
import HeytingLean.Contracts.RoundTrip
import HeytingLean.Logic.StageSemantics

open HeytingLean.LoF
open HeytingLean.Logic Stage

/-
Projector scaffolding for the Clifford bridge upgrade. Projector nets will realise the orthomodular
target carrier while preserving the legacy round-trip interface.
-/

namespace HeytingLean
namespace Bridges
namespace Clifford
namespace Projector

universe u v

variable {β : Type u}

/-- Basic projector data: an element that is both idempotent and self-adjoint. -/
structure Carrier (β : Type u) [Mul β] [Star β] where
  axis : β
  idempotent : axis * axis = axis
  selfAdjoint : star axis = axis

namespace Carrier

variable [Mul β] [Star β]

@[simp] lemma star_axis (P : Carrier β) : star P.axis = P.axis :=
  P.selfAdjoint

@[simp] lemma axis_idempotent (P : Carrier β) : P.axis * P.axis = P.axis :=
  P.idempotent

end Carrier

section
variable {α : Type v} [PrimaryAlgebra α]
variable {β : Type u} [Mul β] [Star β]

/-- Clifford projector model couples the legacy bridge with a target projector carrier. -/
structure Model where
  core : Clifford.Model α
  projector : Carrier β

namespace Model

structure Carrier (M : Model (α := α) (β := β)) where
  bivector : Clifford.Pair α

namespace Carrier

variable {M : Model (α := α) (β := β)}

@[simp] lemma mk_bivector (p : Clifford.Pair α) :
    (Carrier.mk (M := M) p).bivector = p := rfl

def toPair (c : Carrier M) : Clifford.Pair α := c.bivector

def fromPair (p : Clifford.Pair α) : Carrier M := ⟨p⟩

@[simp] lemma toPair_fromPair (p : Clifford.Pair α) :
    (fromPair (M := M) p).toPair = p := rfl

end Carrier

variable [PrimaryAlgebra α]

/-- Encode into the projector carrier by reusing the core Clifford encode. -/
noncomputable def encode (M : Model (α := α) (β := β)) :
    M.core.R.Omega → Carrier M :=
  fun a =>
    Carrier.fromPair (M := M)
      (Clifford.Model.encode (M := M.core) a)

/-- Decode a projector carrier element via the core Clifford decode. -/
noncomputable def decode (M : Model (α := α) (β := β)) :
    Carrier M → M.core.R.Omega :=
  fun c =>
    Clifford.Model.decode (M := M.core) (Carrier.toPair c)

/-- Round-trip contract for the projector carrier. -/
noncomputable def contract (M : Model (α := α) (β := β)) :
    Contracts.RoundTrip (M.core.R) (Carrier M) :=
  { encode := M.encode
    decode := M.decode
    round := by
      intro a
      unfold encode decode
      simp [Carrier.toPair_fromPair]
      exact Clifford.Model.decode_encode (M := M.core) (a := a) }

/-- Logical shadow lifted through the projector carrier. -/
noncomputable def logicalShadow (M : Model (α := α) (β := β)) :
    Carrier M → α :=
  fun c =>
    Clifford.Model.logicalShadow (M := M.core) (Carrier.toPair c)

/-- Stage-style MV addition on projector carriers. -/
noncomputable def stageMvAdd (M : Model (α := α) (β := β)) :
    Carrier M → Carrier M → Carrier M :=
  fun v w =>
    Carrier.fromPair (M := M)
      (Clifford.Model.stageMvAdd (M := M.core)
        (Carrier.toPair v) (Carrier.toPair w))

/-- Stage-style effect compatibility on projector carriers. -/
def stageEffectCompatible (M : Model (α := α) (β := β))
    (v w : Carrier M) : Prop :=
  Clifford.Model.stageEffectCompatible (M := M.core)
    (Carrier.toPair v) (Carrier.toPair w)

/-- Stage-style partial effect addition on projector carriers. -/
noncomputable def stageEffectAdd? (M : Model (α := α) (β := β))
    (v w : Carrier M) : Option (Carrier M) :=
  (Clifford.Model.stageEffectAdd? (M := M.core)
      (Carrier.toPair v) (Carrier.toPair w)).map
    (Carrier.fromPair (M := M))

/-- Stage-style orthocomplement lifted to projector carriers. -/
noncomputable def stageOrthocomplement (M : Model (α := α) (β := β)) :
    Carrier M → Carrier M :=
  fun v =>
    Carrier.fromPair (M := M)
      (Clifford.Model.stageOrthocomplement (M := M.core)
        (Carrier.toPair v))

/-- Stage-style Heyting implication lifted to projector carriers. -/
noncomputable def stageHimp (M : Model (α := α) (β := β))
    (v w : Carrier M) : Carrier M :=
  Carrier.fromPair (M := M)
    (Clifford.Model.stageHimp (M := M.core)
      (Carrier.toPair v) (Carrier.toPair w))

/-- Stage collapse lifted to projector carriers. -/
noncomputable def stageCollapseAt (M : Model (α := α) (β := β))
    (n : ℕ) : Carrier M → Carrier M :=
  fun v =>
    Carrier.fromPair (M := M)
      (Clifford.Model.stageCollapseAt (M := M.core) n
        (Carrier.toPair v))

/-- Stage expansion lifted to projector carriers. -/
noncomputable def stageExpandAt (M : Model (α := α) (β := β))
    (n : ℕ) : Carrier M → Carrier M :=
  fun v =>
    Carrier.fromPair (M := M)
      (Clifford.Model.stageExpandAt (M := M.core) n
        (Carrier.toPair v))

/-- Stage Occam reduction on projector carriers. -/
noncomputable def stageOccam (M : Model (α := α) (β := β)) :
    Carrier M → Carrier M :=
  Contracts.stageOccam (R := M.core.R) (C := M.contract)

@[simp] lemma stageOccam_encode
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) (a : M.core.R.Omega) :
    M.stageOccam (M.contract.encode a) =
      M.contract.encode
        (Reentry.Omega.mk (R := M.core.R)
          (HeytingLean.Epistemic.occam (R := M.core.R) (a : α))
          (HeytingLean.Epistemic.occam_idempotent
            (R := M.core.R) (a := (a : α)))) := by
  classical
  unfold Model.stageOccam
  exact
    HeytingLean.Contracts.stageOccam_encode
      (R := M.core.R) (C := M.contract) a

end Model

end

namespace Model

@[simp] lemma decode_encode
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) (a : M.core.R.Omega) :
    M.decode (M.encode a) = a := by
  unfold Model.encode Model.decode
  change Clifford.Model.decode (M := M.core)
      (Clifford.Model.encode (M := M.core) a) = a
  exact Clifford.Model.decode_encode (M := M.core) (a := a)

@[simp] lemma logicalShadow_encode
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) (a : M.core.R.Omega) :
    M.logicalShadow (M.encode a) = M.core.R a := by
  unfold Model.logicalShadow Model.encode
  change Clifford.Model.logicalShadow (M := M.core)
      (Clifford.Model.encode (M := M.core) a) = M.core.R a
  exact Clifford.Model.logicalShadow_encode' (M := M.core) (a := a)

@[simp] lemma projector_axis_idem
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) :
    M.projector.axis * M.projector.axis = M.projector.axis :=
  M.projector.axis_idempotent

@[simp] lemma projector_axis_selfAdjoint
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) :
    star M.projector.axis = M.projector.axis :=
  M.projector.star_axis

@[simp] lemma core_project_encode
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) (a : M.core.R.Omega) :
    M.core.project (Carrier.toPair (M.encode a)) =
      Carrier.toPair (M.encode a) := by
  classical
  simp [Model.encode, Carrier.toPair, Carrier.fromPair]

@[simp] lemma core_project_stageCollapseAt
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) (n : ℕ) (c : Carrier M) :
    M.core.project (Carrier.toPair (M.stageCollapseAt n c)) =
      Carrier.toPair (M.stageCollapseAt n c) := by
  classical
  simp [Model.stageCollapseAt, Carrier.toPair, Carrier.fromPair]

@[simp] lemma core_project_stageExpandAt
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) (n : ℕ) (c : Carrier M) :
    M.core.project (Carrier.toPair (M.stageExpandAt n c)) =
      Carrier.toPair (M.stageExpandAt n c) := by
  classical
  simp [Model.stageExpandAt, Carrier.toPair, Carrier.fromPair]

@[simp] lemma core_project_stageOccam
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) (c : Carrier M) :
    M.core.project (Carrier.toPair (M.stageOccam c)) =
      Carrier.toPair (M.stageOccam c) := by
  classical
  unfold Model.stageOccam Contracts.stageOccam
  simp [Model.contract]

/-- Projected subspace determined by the core projector. -/
def projected
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) :
    Set (Clifford.Pair α) :=
  {p | M.core.project p = p}

/-- Closure invariants required for the enriched projector rollout. -/
structure Invariants
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) : Prop where
  collapseClosed :
    ∀ n (c : Carrier M),
      Carrier.toPair (M.stageCollapseAt n c) ∈ projected (M := M)
  expandClosed :
    ∀ n (c : Carrier M),
      Carrier.toPair (M.stageExpandAt n c) ∈ projected (M := M)
  occamClosed :
    ∀ c : Carrier M,
      Carrier.toPair (M.stageOccam c) ∈ projected (M := M)
  axisClosed :
    Carrier.toPair (M.encode M.core.R.eulerBoundary) ∈ projected (M := M)

/-- The current scaffold already satisfies the closure invariants thanks to the core projector. -/
def invariants
    {α : Type v} [PrimaryAlgebra α]
    {β : Type u} [Mul β] [Star β]
    (M : Model (α := α) (β := β)) : Invariants (M := M) :=
{ collapseClosed := by
    intro n c
    refine Set.mem_setOf_eq.mpr ?_
    exact
      core_project_stageCollapseAt (M := M) (n := n) (c := c)
  expandClosed := by
    intro n c
    refine Set.mem_setOf_eq.mpr ?_
    exact
      core_project_stageExpandAt (M := M) (n := n) (c := c)
  occamClosed := by
    intro c
    refine Set.mem_setOf_eq.mpr ?_
    exact
      core_project_stageOccam (M := M) (c := c)
  axisClosed := by
    refine Set.mem_setOf_eq.mpr ?_
    exact
      core_project_encode (M := M) (a := M.core.R.eulerBoundary) }


end Model

end Projector
end Clifford
end Bridges
end HeytingLean
