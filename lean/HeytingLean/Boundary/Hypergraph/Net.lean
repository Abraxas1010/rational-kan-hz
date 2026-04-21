import Mathlib.Data.Finset.Card

import HeytingLean.Boundary.Hypergraph.Agent
import HeytingLean.Boundary.Hypergraph.Motif

set_option linter.dupNamespace false

namespace HeytingLean
namespace Boundary
namespace Hypergraph

/-- A hyperwire is one irreducible incident set of agent ports, not a pairwise clique. -/
structure Wire (AgentId : Type*) where
  endpoints : Finset (AgentId × Nat)
  deriving DecidableEq

namespace Wire

def arity {AgentId : Type*} [DecidableEq AgentId] (wire : Wire AgentId) : Nat :=
  wire.endpoints.card

def WellFormed {AgentId : Type*} [DecidableEq AgentId] (wire : Wire AgentId) : Prop :=
  2 <= wire.arity

def IncidentAgent {AgentId : Type*} [DecidableEq AgentId]
    (wire : Wire AgentId) (id : AgentId) : Prop :=
  ∃ port, (id, port) ∈ wire.endpoints

end Wire

/-- Boundary hypergraph interaction nets. `agents` is total so finite subnets can be
represented by the `live` set without partial lookup noise. -/
structure HNet (AgentId : Type*) [DecidableEq AgentId] where
  agents : AgentId -> HAgent
  live : Finset AgentId
  wires : Finset (Wire AgentId)
  freeports : Finset (AgentId × Nat)
  linear : Bool := true
  internal_connected : Bool := true

namespace HNet

def wireTouches {AgentId : Type*} [DecidableEq AgentId]
    (wire : Wire AgentId) (S : Finset AgentId) : Prop :=
  ∃ endpoint ∈ wire.endpoints, endpoint.1 ∈ S

def wireContained {AgentId : Type*} [DecidableEq AgentId]
    (wire : Wire AgentId) (S : Finset AgentId) : Prop :=
  ∀ endpoint ∈ wire.endpoints, endpoint.1 ∈ S

/-- A finite agent set is closed when every incident hyperwire is wholly contained in it. -/
def ClosedUnderIncidentWires {AgentId : Type*} [DecidableEq AgentId]
    (N : HNet AgentId) (S : Finset AgentId) : Prop :=
  ∀ wire ∈ N.wires, wireTouches wire S -> wireContained wire S

def WellFormed {AgentId : Type*} [DecidableEq AgentId] (N : HNet AgentId) : Prop :=
    (∀ id ∈ N.live, (N.agents id).Valid) ∧
    (∀ wire ∈ N.wires, wire.WellFormed) ∧
    (∀ wire ∈ N.wires, ∀ endpoint ∈ wire.endpoints, endpoint.1 ∈ N.live) ∧
    N.linear = true ∧ N.internal_connected = true

def liveHypergraph {AgentId : Type*} [DecidableEq AgentId] (N : HNet AgentId) :
    Hypergraph AgentId :=
  { edges := N.wires.image (fun wire => ({ nodes := wire.endpoints.image Prod.fst } : Hyperedge AgentId)) }

@[simp] theorem empty_closed {AgentId : Type*} [DecidableEq AgentId]
    (N : HNet AgentId) :
    ClosedUnderIncidentWires N (∅ : Finset AgentId) := by
  intro wire hwire htouch endpoint hendpoint
  rcases htouch with ⟨touched, _htouched, htouchedEmpty⟩
  simp at htouchedEmpty

theorem inter_closed {AgentId : Type*} [DecidableEq AgentId]
    {N : HNet AgentId} {A B : Finset AgentId}
    (hA : ClosedUnderIncidentWires N A)
    (hB : ClosedUnderIncidentWires N B) :
    ClosedUnderIncidentWires N (A ∩ B) := by
  intro wire hwire htouch endpoint hendpoint
  rcases htouch with ⟨touched, htouchedWire, htouched⟩
  have htouchedA : touched.1 ∈ A := (Finset.mem_inter.mp htouched).1
  have htouchedB : touched.1 ∈ B := (Finset.mem_inter.mp htouched).2
  exact Finset.mem_inter.mpr
    ⟨ hA wire hwire ⟨touched, htouchedWire, htouchedA⟩ endpoint hendpoint
    , hB wire hwire ⟨touched, htouchedWire, htouchedB⟩ endpoint hendpoint ⟩

theorem union_closed {AgentId : Type*} [DecidableEq AgentId]
    {N : HNet AgentId} {A B : Finset AgentId}
    (hA : ClosedUnderIncidentWires N A)
    (hB : ClosedUnderIncidentWires N B) :
    ClosedUnderIncidentWires N (A ∪ B) := by
  intro wire hwire htouch endpoint hendpoint
  rcases htouch with ⟨touched, htouchedWire, htouched⟩
  rcases Finset.mem_union.mp htouched with htouchedA | htouchedB
  · exact Finset.mem_union.mpr
      (Or.inl (hA wire hwire ⟨touched, htouchedWire, htouchedA⟩ endpoint hendpoint))
  · exact Finset.mem_union.mpr
      (Or.inr (hB wire hwire ⟨touched, htouchedWire, htouchedB⟩ endpoint hendpoint))

end HNet

export HNet (ClosedUnderIncidentWires)

end Hypergraph
end Boundary
end HeytingLean
