import HeytingLean.Boundary.Hypergraph.Rules

set_option linter.dupNamespace false

namespace HeytingLean
namespace Boundary
namespace Hypergraph

/-- Closed subnets are the only carrier on which Phase C makes CHECK closure claims. -/
abbrev ClosedSubNet {AgentId : Type*} [DecidableEq AgentId] (N : HNet AgentId) :=
  { S : Finset AgentId // ClosedUnderIncidentWires N S }

namespace ClosedSubNet

def carrier {AgentId : Type*} [DecidableEq AgentId] {N : HNet AgentId}
    (S : ClosedSubNet N) : Finset AgentId :=
  S.1

def inter {AgentId : Type*} [DecidableEq AgentId] {N : HNet AgentId}
    (A B : ClosedSubNet N) : ClosedSubNet N :=
  ⟨A.1 ∩ B.1, HNet.inter_closed A.2 B.2⟩

def union {AgentId : Type*} [DecidableEq AgentId] {N : HNet AgentId}
    (A B : ClosedSubNet N) : ClosedSubNet N :=
  ⟨A.1 ∪ B.1, HNet.union_closed A.2 B.2⟩

def empty {AgentId : Type*} [DecidableEq AgentId] (N : HNet AgentId) : ClosedSubNet N :=
  ⟨∅, HNet.empty_closed N⟩

@[simp] theorem carrier_inter {AgentId : Type*} [DecidableEq AgentId] {N : HNet AgentId}
    (A B : ClosedSubNet N) :
    (inter A B).carrier = A.carrier ∩ B.carrier :=
  rfl

@[simp] theorem carrier_union {AgentId : Type*} [DecidableEq AgentId] {N : HNet AgentId}
    (A B : ClosedSubNet N) :
    (union A B).carrier = A.carrier ∪ B.carrier :=
  rfl

end ClosedSubNet

/-- Polyadic CHECK constraints are joint constraints over a finite term set. This is
deliberately not a conjunction of independent unary checks. -/
structure PolyadicConstraint (AgentId : Type*) [DecidableEq AgentId] where
  terms : Finset AgentId
  boundary : Finset AgentId
  arity : Nat

def AllTermsLive {AgentId : Type*} [DecidableEq AgentId]
    (N : HNet AgentId) (terms : Finset AgentId) : Prop :=
  terms ⊆ N.live

def JointBoundarySatisfies {AgentId : Type*} [DecidableEq AgentId]
    (N : HNet AgentId) (terms boundary : Finset AgentId) : Prop :=
  terms.Nonempty ∧ boundary ⊆ N.live ∧
    ∃ wire ∈ N.wires, HNet.wireTouches wire terms ∧ HNet.wireTouches wire boundary

def NoErrorAgentGenerated {AgentId : Type*} [DecidableEq AgentId]
    (N : HNet AgentId) (terms : Finset AgentId) : Prop :=
  ∀ id ∈ terms, (N.agents id).IsError = false

def PolyadicCheckPasses {AgentId : Type*} [DecidableEq AgentId]
    (N : HNet AgentId) (c : PolyadicConstraint AgentId) : Prop :=
  c.terms.Nonempty ∧
    AllTermsLive N c.terms ∧
    JointBoundarySatisfies N c.terms c.boundary ∧
    NoErrorAgentGenerated N c.terms

/-- A closed CHECK core packages the agents that survive one joint polyadic CHECK
frontier. The operator below intersects with this closed core, so the meet theorem is
about closed subnets and does not assert a raw-subnet nucleus. -/
def polyadicCheckOperator {AgentId : Type*} [DecidableEq AgentId] {N : HNet AgentId}
    (core : ClosedSubNet N) (S : ClosedSubNet N) : ClosedSubNet N :=
  ClosedSubNet.inter S core

theorem closedSubNet_intersection_closed {AgentId : Type*} [DecidableEq AgentId]
    {N : HNet AgentId} (A B : ClosedSubNet N) :
    ClosedUnderIncidentWires N (A.1 ∩ B.1) :=
  HNet.inter_closed A.2 B.2

theorem closedSubNet_closure_union_closed {AgentId : Type*} [DecidableEq AgentId]
    {N : HNet AgentId} (A B : ClosedSubNet N) :
    ClosedUnderIncidentWires N (A.1 ∪ B.1) :=
  HNet.union_closed A.2 B.2

theorem polyadicCheck_meet_preserving_closed {AgentId : Type*} [DecidableEq AgentId]
    {N : HNet AgentId} (core A B : ClosedSubNet N) :
    (polyadicCheckOperator core (ClosedSubNet.inter A B)).carrier =
      (ClosedSubNet.inter (polyadicCheckOperator core A)
        (polyadicCheckOperator core B)).carrier := by
  ext id
  simp [polyadicCheckOperator, ClosedSubNet.carrier, ClosedSubNet.inter, Finset.mem_inter]

end Hypergraph
end Boundary
end HeytingLean
