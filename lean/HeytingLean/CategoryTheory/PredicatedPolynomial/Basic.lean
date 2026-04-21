import Mathlib.CategoryTheory.Category.Basic

universe u v w

namespace HeytingLean.CategoryTheory.PredicatedPolynomial

/--
A predicated polynomial functor consists of positions, directions at each position,
and a predicate constraining which directions are admissible. This keeps the usual
container/lens shape while preserving the predicate payload erased by the current
Dialectica shadow bridge.
-/
structure PredPoly where
  /-- The type of positions (shapes). -/
  pos : Type u
  /-- The directions available at each position. -/
  dir : pos → Type v
  /-- The admissibility predicate on directions. -/
  pred : (i : pos) → dir i → Prop

namespace PredPoly

/--
A morphism of predicated polynomials is a polynomial lens together with a proof
that admissible pulled-back directions remain admissible after transport.
-/
@[ext] structure Hom (P Q : PredPoly.{u, v}) where
  /-- Forward map on positions. -/
  onPos : P.pos → Q.pos
  /-- Backward map on directions. -/
  onDir : (i : P.pos) → Q.dir (onPos i) → P.dir i
  /-- Predicate preservation for pulled-back directions. -/
  pred_ok : ∀ (i : P.pos) (d : Q.dir (onPos i)), P.pred i (onDir i d) → Q.pred (onPos i) d

namespace Hom

/-- Composition of predicated-polynomial morphisms. -/
def comp {P Q R : PredPoly.{u, v}} (f : Hom P Q) (g : Hom Q R) : Hom P R where
  onPos := g.onPos ∘ f.onPos
  onDir := fun i d => f.onDir i (g.onDir (f.onPos i) d)
  pred_ok := fun i d h => g.pred_ok (f.onPos i) d (f.pred_ok i (g.onDir (f.onPos i) d) h)

/-- The composite morphism preserves admissible directions. -/
theorem comp_pred_ok {P Q R : PredPoly.{u, v}} (f : Hom P Q) (g : Hom Q R)
    (i : P.pos) (d : R.dir ((g.onPos ∘ f.onPos) i)) :
    P.pred i ((comp f g).onDir i d) → R.pred ((comp f g).onPos i) d := by
  intro h
  exact g.pred_ok (f.onPos i) d (f.pred_ok i (g.onDir (f.onPos i) d) h)

end Hom

/-- Identity morphism on a predicated polynomial. -/
def id (P : PredPoly.{u, v}) : Hom P P where
  onPos := fun i => i
  onDir := fun _ d => d
  pred_ok := fun _ _ h => h

/-- The identity morphism preserves admissible directions. -/
theorem id_pred_ok (P : PredPoly.{u, v}) (i : P.pos) (d : P.dir i) :
    P.pred i ((id P).onDir i d) → P.pred ((id P).onPos i) d := by
  simp [PredPoly.id]

end PredPoly

end HeytingLean.CategoryTheory.PredicatedPolynomial
