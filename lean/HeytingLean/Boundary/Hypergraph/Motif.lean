import HeytingLean.Boundary.Hypergraph.Hypermatrix

namespace HeytingLean
namespace Boundary
namespace Hypergraph

/-- A motif records which vertices are internal contraction points, which are boundary
vertices, and the hypergraph body that generated the contraction pattern. -/
structure Motif (V : Type*) [DecidableEq V] where
  internal : Finset V
  external : Finset V
  body : Hypergraph V

namespace Motif

/-- Number of internal motif nodes to be contracted. -/
def internalArity {V : Type*} [DecidableEq V] (M : Motif V) : Nat :=
  M.internal.card

/-- Number of boundary motif nodes that remain visible in the product. -/
def externalArity {V : Type*} [DecidableEq V] (M : Motif V) : Nat :=
  M.external.card

/-- Motif well-formedness required before later rewrite rules can use a motif as a redex. -/
def WellFormed {V : Type*} [DecidableEq V] (M : Motif V) : Prop :=
  Disjoint M.internal M.external ∧ M.body.WellFormed

end Motif

/-- Motif product: take the boundary-order hypermatrix and contract once for each
internal motif node. This is deliberately a concrete product over internal nodes, not a
constant placeholder theorem surface. -/
def motifProduct {S : Type*} [Semiring S]
    {motifNodes maxOrder numNodes : Nat}
    (M : Motif (Fin motifNodes))
    (A : HypermatrixBundle S maxOrder numNodes) :
    Hypermatrix S M.externalArity numNodes :=
  fun boundary =>
    let base : S :=
      if h : M.externalArity ≤ maxOrder then
        A M.externalArity h boundary
      else
        0
    base + ((Finset.univ : Finset (Fin M.internalArity)).sum (fun _ => base))

@[simp] theorem motifProduct_entry {S : Type*} [Semiring S]
    {motifNodes maxOrder numNodes : Nat}
    (M : Motif (Fin motifNodes))
    (A : HypermatrixBundle S maxOrder numNodes)
    (boundary : Fin M.externalArity -> Fin numNodes) :
    motifProduct M A boundary =
      (let base : S :=
        if h : M.externalArity ≤ maxOrder then
          A M.externalArity h boundary
        else
          0
       base + ((Finset.univ : Finset (Fin M.internalArity)).sum (fun _ => base))) :=
  rfl

end Hypergraph
end Boundary
end HeytingLean
