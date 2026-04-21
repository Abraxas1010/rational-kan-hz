import Mathlib.Data.Finset.Card

set_option linter.dupNamespace false

namespace HeytingLean
namespace Boundary
namespace Hypergraph

/-- A Boundary hyperedge is a single irreducible incident set, not a binary clique. -/
structure Hyperedge (V : Type*) where
  nodes : Finset V
  deriving DecidableEq

namespace Hyperedge

/-- Semantic well-formedness predicate for nonempty hyperedges. Kept as a predicate so
`Hyperedge` itself can be used inside `Finset` without proof-irrelevance equality debt. -/
def Nonempty {V : Type*} (edge : Hyperedge V) : Prop :=
  edge.nodes.Nonempty

/-- The arity/order of a hyperedge. This is intentionally separate from the total node dimension. -/
def arity {V : Type*} (edge : Hyperedge V) : Nat :=
  edge.nodes.card

/-- Incidence predicate for a vertex and a hyperedge. -/
def Incident {V : Type*} [DecidableEq V] (v : V) (edge : Hyperedge V) : Prop :=
  v ∈ edge.nodes

/-- Number of binary edges a clique encoding would need for the same node support. -/
def pairCountIfClique {V : Type*} (edge : Hyperedge V) : Nat :=
  edge.arity * (edge.arity - 1) / 2

end Hyperedge

/-- Boundary-local hypergraphs. Do not substitute the WPP/Wolfram hypergraph surface here;
Boundary needs this layer to support interaction-net ports and exact binary recovery later. -/
structure Hypergraph (V : Type*) [DecidableEq V] where
  edges : Finset (Hyperedge V)

namespace Hypergraph

/-- A well-formed hypergraph has no empty hyperedges. -/
def WellFormed {V : Type*} [DecidableEq V] (H : Hypergraph V) : Prop :=
  ∀ edge ∈ H.edges, edge.Nonempty

/-- Filter edges by exact hyperedge arity/order. -/
def atOrder {V : Type*} [DecidableEq V] (H : Hypergraph V) (arity : Nat) : Finset (Hyperedge V) :=
  H.edges.filter (fun edge => edge.arity = arity)

/-- The number of irreducible hyperedges, not the number of pairwise clique edges. -/
def edgeCount {V : Type*} [DecidableEq V] (H : Hypergraph V) : Nat :=
  H.edges.card

/-- Existence of an edge at a given irreducible arity. -/
def HasIrreducibleArity {V : Type*} [DecidableEq V] (H : Hypergraph V) (arity : Nat) : Prop :=
  ∃ edge ∈ H.edges, edge.arity = arity

@[simp] theorem mem_atOrder_iff {V : Type*} [DecidableEq V]
    (H : Hypergraph V) (arity : Nat) (edge : Hyperedge V) :
    edge ∈ H.atOrder arity ↔ edge ∈ H.edges ∧ edge.arity = arity := by
  simp [atOrder]

end Hypergraph

end Hypergraph
end Boundary
end HeytingLean
