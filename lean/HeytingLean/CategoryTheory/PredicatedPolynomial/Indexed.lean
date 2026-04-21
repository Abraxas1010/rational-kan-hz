import HeytingLean.CategoryTheory.PredicatedPolynomial.Basic

universe u v w wRel

namespace HeytingLean.CategoryTheory.PredicatedPolynomial

/--
A predicated polynomial family indexed by an external context surface.
The index relation need not already be a category; P4 only needs explicit
transport of witness/challenge predicates along context changes.
-/
structure IndexedPredPoly where
  /-- External contexts indexing the predicated polynomial family. -/
  Ctx : Type w
  /-- The predicated polynomial at a given context. -/
  fiber : Ctx → PredPoly.{u, v}
  /-- Allowed context-to-context reindexing witnesses. -/
  Rel : Ctx → Ctx → Sort wRel
  /-- Predicate-preserving transport along a context witness. -/
  reindex : {i j : Ctx} → Rel i j → PredPoly.Hom (fiber i) (fiber j)

namespace IndexedPredPoly

/--
Transporting an admissible source direction along an indexed reindexing witness
preserves admissibility in the target context.
-/
theorem indexedAlpha_transport (P : IndexedPredPoly.{u, v, w, wRel})
    {i j : P.Ctx} (h : P.Rel i j) (x : (P.fiber i).pos)
    (d : (P.fiber j).dir ((P.reindex h).onPos x)) :
    (P.fiber i).pred x ((P.reindex h).onDir x d) →
      (P.fiber j).pred ((P.reindex h).onPos x) d := by
  intro hx
  exact (P.reindex h).pred_ok x d hx

end IndexedPredPoly

end HeytingLean.CategoryTheory.PredicatedPolynomial
