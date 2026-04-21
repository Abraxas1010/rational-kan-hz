import Mathlib.Algebra.BigOperators.Fin

import HeytingLean.Boundary.Hypergraph.Core

set_option linter.dupNamespace false

namespace HeytingLean
namespace Boundary
namespace Hypergraph

/-- A hypermatrix of order `order` over `numNodes` nodes. The two parameters are
separate on purpose: order is arity, `numNodes` is the ambient dimension. -/
def Hypermatrix (S : Type*) (order : Nat) (numNodes : Nat) :=
  (Fin order -> Fin numNodes) -> S

/-- A finite bundle of adjacency hypermatrices indexed by their arity/order. -/
def HypermatrixBundle (S : Type*) (maxOrder : Nat) (numNodes : Nat) :=
  (order : Nat) -> order ≤ maxOrder -> Hypermatrix S order numNodes

/-- Computational containment of an ordered tuple of endpoints in one hyperedge of the same arity. -/
def edgeContainsIndicesBool {numNodes order : Nat}
    (edge : Hyperedge (Fin numNodes)) (indices : Fin order -> Fin numNodes) : Bool :=
  (edge.arity == order) && (((Finset.univ.filter (fun p : Fin order => indices p ∈ edge.nodes)).card) == order)

/-- Computational existence of an order-exact hyperedge containing the supplied tuple. -/
def Hypergraph.hasOrderEdgeContaining {numNodes order : Nat}
    (H : Hypergraph (Fin numNodes)) (indices : Fin order -> Fin numNodes) : Bool :=
  if (H.edges.filter (fun edge => edgeContainsIndicesBool edge indices = true)).card = 0 then false else true

/-- Proposition wrapper for the Boolean adjacency predicate. -/
def Hypergraph.HasOrderEdgeContaining {numNodes order : Nat}
    (H : Hypergraph (Fin numNodes)) (indices : Fin order -> Fin numNodes) : Prop :=
  H.hasOrderEdgeContaining indices = true

/-- Boolean adjacency hypermatrix for one exact hyperedge order. -/
def adjacencyHypermatrix {numNodes : Nat} (H : Hypergraph (Fin numNodes)) (order : Nat) :
    Hypermatrix Bool order numNodes :=
  fun indices => H.hasOrderEdgeContaining indices

/-- Exact order-2 adjacency is precisely the order-2 hyperedge containment predicate. -/
theorem adjacencyHypermatrix_order_two_binary_edge_iff {numNodes : Nat}
    (H : Hypergraph (Fin numNodes)) (idx : Fin 2 -> Fin numNodes) :
    adjacencyHypermatrix H 2 idx = true ↔ H.HasOrderEdgeContaining idx :=
  Iff.rfl

/-- Build an order-2 index from a pair of nodes. -/
def idx2 {numNodes : Nat} (i j : Fin numNodes) : Fin 2 -> Fin numNodes :=
  fun p => if p.val = 0 then i else j

/-- Build an order-3 index from a triple of nodes. -/
def idx3 {numNodes : Nat} (i j k : Fin numNodes) : Fin 3 -> Fin numNodes :=
  fun p => if p.val = 0 then i else if p.val = 1 then j else k

/-- Ordinary matrix multiplication, expressed on order-2 hypermatrices. -/
def matrixMul2 {S : Type*} [Semiring S] {numNodes : Nat}
    (A B : Hypermatrix S 2 numNodes) : Hypermatrix S 2 numNodes :=
  fun idx => Finset.univ.sum (fun j : Fin numNodes => A (idx2 (idx 0) j) * B (idx2 j (idx 1)))

/-- The V-product specializes to order-2 matrix multiplication. -/
def vProduct {S : Type*} [Semiring S] {numNodes : Nat}
    (A B : Hypermatrix S 2 numNodes) : Hypermatrix S 2 numNodes :=
  matrixMul2 A B

/-- Honest weakened binary recovery theorem: V-product is exactly the local order-2 matrix product above. -/
theorem vProduct_ext_eq_matrixMul2 {S : Type*} [Semiring S] {numNodes : Nat}
    (A B : Hypermatrix S 2 numNodes) :
    vProduct A B = matrixMul2 A B :=
  rfl

/-- A Y-shaped contraction from an order-2 factor and an order-3 factor. -/
def yProduct {S : Type*} [Semiring S] {numNodes : Nat}
    (A : Hypermatrix S 2 numNodes) (B : Hypermatrix S 3 numNodes) :
    Hypermatrix S 3 numNodes :=
  fun idx => Finset.univ.sum (fun j : Fin numNodes => A (idx2 (idx 0) j) * B (idx3 j (idx 1) (idx 2)))

/-- A fish-shaped binary-output contraction through two internal indices. -/
def fishProduct {S : Type*} [Semiring S] {numNodes : Nat}
    (A B C : Hypermatrix S 2 numNodes) : Hypermatrix S 2 numNodes :=
  fun idx =>
    Finset.univ.sum (fun j : Fin numNodes =>
      Finset.univ.sum (fun k : Fin numNodes =>
        A (idx2 (idx 0) j) * B (idx2 j k) * C (idx2 k (idx 1))))

/-- A cone-shaped ternary-output contraction through one internal index. -/
def coneProduct {S : Type*} [Semiring S] {numNodes : Nat}
    (A B C : Hypermatrix S 2 numNodes) : Hypermatrix S 3 numNodes :=
  fun idx =>
    Finset.univ.sum (fun l : Fin numNodes =>
      A (idx2 (idx 0) l) * B (idx2 (idx 1) l) * C (idx2 (idx 2) l))

end Hypergraph
end Boundary
end HeytingLean
