import Mathlib.Order.Basic
import Mathlib.Data.Set.Lattice
import HeytingLean.Bridges.Graph
import HeytingLean.Contracts.RoundTrip
import HeytingLean.Logic.StageSemantics

/-
Minimal Alexandroff scaffolding for the graph bridge.  We keep a reference to the core graph model
and an auxiliary open set while reusing the existing transport.
-/

namespace HeytingLean
namespace Bridges
namespace Graph
namespace Alexandroff

open HeytingLean.Contracts
open HeytingLean.LoF
open HeytingLean.Logic Stage

universe u

variable {α : Type u}

section
variable [PrimaryAlgebra α]

/-- Alexandroff graph model pointing at the core bridge. -/
structure Model where
  core : Graph.Model α
  openSet : Set α
  openUpper :
    ∀ {x y : α}, x ≤ y → x ∈ openSet → y ∈ openSet
  openInfClosed :
    ∀ {x y : α}, x ∈ openSet → y ∈ openSet → x ⊓ y ∈ openSet
  openSupClosed :
    ∀ {x y : α}, x ∈ openSet → y ∈ openSet →
      core.R (x ⊔ y) ∈ openSet
  collapseClosed :
    ∀ {n : ℕ} {x : α},
      x ∈ openSet →
        Graph.Model.stageCollapseAt (M := core) n x ∈ openSet
  expandClosed :
    ∀ {n : ℕ} {x : α},
      x ∈ openSet →
        Graph.Model.stageExpandAt (M := core) n x ∈ openSet
  occamClosed :
    ∀ {x : α},
      x ∈ openSet →
        Graph.Model.stageOccam (M := core) x ∈ openSet

namespace Model

private lemma collapseAt_eq_reentry_aux (core : Graph.Model α) :
    ∀ n,
      HeytingLean.Logic.Modal.DialParam.collapseAt
        (α := α) (R := core.R) n = fun a : α => core.R a
  | 0 => by
      funext a
      simp
  | Nat.succ n => by
      funext a
      have ih := collapseAt_eq_reentry_aux (core := core) n
      have hsucc :=
        congrArg (fun f => f a)
          (HeytingLean.Logic.Modal.DialParam.collapseAt_succ
            (α := α) (R := core.R) n)
      have hprev :=
        congrArg (fun f => f a) ih
      exact hsucc.trans hprev

private lemma expandAt_eq_reentry_aux (core : Graph.Model α) :
    ∀ n,
      HeytingLean.Logic.Modal.DialParam.expandAt
        (α := α) (R := core.R) n = fun a : α => core.R a
  | 0 => by
      funext a
      simp
  | Nat.succ n => by
      funext a
      have ih := expandAt_eq_reentry_aux (core := core) n
      have hsucc :=
        congrArg (fun f => f a)
          (HeytingLean.Logic.Modal.DialParam.expandAt_succ
            (α := α) (R := core.R) n)
      have hprev :=
        congrArg (fun f => f a) ih
      exact hsucc.trans hprev

@[simp] lemma stageCollapseAt_eq (core : Graph.Model α)
    (n : ℕ) (x : α) :
    Graph.Model.stageCollapseAt (M := core) n x =
      core.R x := by
  classical
  dsimp [Graph.Model.stageCollapseAt, Graph.Model.encode,
    Graph.Model.decode, HeytingLean.Logic.Stage.DialParam.collapseAtOmega]
  simp

@[simp] lemma stageExpandAt_eq (core : Graph.Model α)
    (n : ℕ) (x : α) :
    Graph.Model.stageExpandAt (M := core) n x =
      core.R x := by
  classical
  dsimp [Graph.Model.stageExpandAt, Graph.Model.encode,
    Graph.Model.decode, HeytingLean.Logic.Stage.DialParam.expandAtOmega]
  simp

def Carrier (M : Model (α := α)) : Type u := M.core.Carrier

def memOpen (M : Model (α := α)) (x : α) : Prop :=
  x ∈ M.openSet

/-- Open sets are upwards-closed: membership of `x` extends to any `y ≥ x`. -/
lemma mem_of_le (M : Model (α := α)) {x y : α}
    (hxy : x ≤ y) (hx : M.memOpen x) : M.memOpen y :=
  M.openUpper hxy hx

/-- The open set is closed under finite infima. -/
lemma mem_inf (M : Model (α := α)) {x y : α}
    (hx : M.memOpen x) (hy : M.memOpen y) :
    M.memOpen (x ⊓ y) :=
  M.openInfClosed hx hy

/-- The open set is closed under `core.R` applied to finite suprema. -/
lemma mem_sup_project (M : Model (α := α)) {x y : α}
    (hx : M.memOpen x) (hy : M.memOpen y) :
    M.memOpen (M.core.R (x ⊔ y)) :=
  M.openSupClosed hx hy

/-- Collapsing at any stage keeps the state inside the open region. -/
lemma mem_stageCollapseAt (M : Model (α := α)) (n : ℕ)
    {x : α} (hx : M.memOpen x) :
    M.memOpen (Graph.Model.stageCollapseAt (M := M.core) n x) :=
  M.collapseClosed hx

/-- Stage expansions also preserve openness. -/
lemma mem_stageExpandAt (M : Model (α := α)) (n : ℕ)
    {x : α} (hx : M.memOpen x) :
    M.memOpen (Graph.Model.stageExpandAt (M := M.core) n x) :=
  M.expandClosed hx

/-- Applying the Occam stage keeps points inside the open set. -/
lemma mem_stageOccam (M : Model (α := α))
    {x : α} (hx : M.memOpen x) :
    M.memOpen (Graph.Model.stageOccam (M := M.core) x) :=
  M.occamClosed hx

@[simp] def univ (core : Graph.Model α) : Model (α := α) :=
  { core := core
    openSet := Set.univ
    openUpper := by
      intro x y _ _
      simp
    openInfClosed := by
      intro x y _ _
      simp
    openSupClosed := by
      intro x y _ _
      simp
    collapseClosed := by
      intro n x _
      simp
    expandClosed := by
      intro n x _
      simp
    occamClosed := by
      intro x _
      simp }

@[simp] lemma memOpen_univ (core : Graph.Model α)
    (x : α) :
    (Model.univ (α := α) core).memOpen x := by
  simp [Model.memOpen]

@[simp] def processUpper (core : Graph.Model α) : Model (α := α) :=
  { core := core
    openSet := {x | ((core.R.process : core.R.Omega) : α) ≤ x}
    openUpper := by
      intro x y hxy hx
      exact le_trans hx hxy
    openInfClosed := by
      intro x y hx hy
      exact le_inf hx hy
    openSupClosed := by
      intro x y hx hy
      have hx_sup :
          ((core.R.process : core.R.Omega) : α) ≤ x ⊔ y :=
        le_sup_of_le_left hx
      exact le_trans hx_sup (Reentry.le_apply (R := core.R) (a := x ⊔ y))
    collapseClosed := by
      intro n x hx
      change ((core.R.process : core.R.Omega) : α) ≤
          Graph.Model.stageCollapseAt (M := core) n x
      have hxR :
          ((core.R.process : core.R.Omega) : α) ≤ core.R x :=
        le_trans hx (Reentry.le_apply (R := core.R) (a := x))
      have hStage := stageCollapseAt_eq (core := core) (n := n) (x := x)
      exact hStage.symm ▸ hxR
    expandClosed := by
      intro n x hx
      change ((core.R.process : core.R.Omega) : α) ≤
          Graph.Model.stageExpandAt (M := core) n x
      have hxR :
          ((core.R.process : core.R.Omega) : α) ≤ core.R x :=
        le_trans hx (Reentry.le_apply (R := core.R) (a := x))
      have hStage := stageExpandAt_eq (core := core) (n := n) (x := x)
      exact hStage.symm ▸ hxR
    occamClosed := by
      intro x hx
      have hxR :
          ((core.R.process : core.R.Omega) : α) ≤ core.R x :=
        le_trans hx (Reentry.le_apply (R := core.R) (a := x))
      have hoccam :=
        HeytingLean.Epistemic.occam_contains_candidate
          (R := core.R) (a := core.R x)
          (u := ((core.R.process : core.R.Omega) : α))
          hxR
          (by
            simp)
      change ((core.R.process : core.R.Omega) : α) ≤
          Graph.Model.stageOccam (M := core) x
      have hoccam' :
          ((core.R.process : core.R.Omega) : α) ≤
            HeytingLean.Epistemic.occam (R := core.R) (core.R x) := hoccam
      have hstage :
          HeytingLean.Epistemic.occam (R := core.R) (core.R x) =
            Graph.Model.stageOccam (M := core) x := by
        unfold Graph.Model.stageOccam
        unfold Contracts.stageOccam
        simp [Graph.Model.contract, Graph.Model.encode, Graph.Model.decode]
      exact hstage ▸ hoccam' }

@[simp] lemma memOpen_processUpper (core : Graph.Model α) (x : α) :
    (processUpper (α := α) core).memOpen x ↔
      ((core.R.process : core.R.Omega) : α) ≤ x := Iff.rfl

@[simp] lemma process_mem_processUpper (core : Graph.Model α) :
    (processUpper (α := α) core).memOpen
        ((core.R.process : core.R.Omega) : α) := by
  change ((core.R.process : core.R.Omega) : α) ≤
      ((core.R.process : core.R.Omega) : α)
  exact le_rfl

noncomputable def encode (M : Model (α := α)) :
    M.core.R.Omega → M.Carrier :=
  Graph.Model.encode (M := M.core)

noncomputable def decode (M : Model (α := α)) :
    M.Carrier → M.core.R.Omega :=
  Graph.Model.decode (M := M.core)

noncomputable def contract (M : Model (α := α)) :
    Contracts.RoundTrip (M.core.R) M.Carrier :=
  Graph.Model.contract (M := M.core)

noncomputable def logicalShadow (M : Model (α := α)) :
    M.Carrier → α :=
  Graph.Model.logicalShadow (M := M.core)

@[simp] lemma logicalShadow_encode (M : Model (α := α))
    (a : M.core.R.Omega) :
    M.logicalShadow (M.contract.encode a) = M.core.R a :=
  Graph.Model.logicalShadow_encode (M := M.core) (a := a)

@[simp] lemma decode_encode (M : Model (α := α)) (a : M.core.R.Omega) :
    M.decode (M.contract.encode a) = a :=
  Graph.Model.decode_encode (M := M.core) (a := a)

noncomputable def stageMvAdd (M : Model (α := α)) :
    M.Carrier → M.Carrier → M.Carrier :=
  Graph.Model.stageMvAdd (M := M.core)

def stageEffectCompatible (M : Model (α := α))
    (x y : M.Carrier) : Prop :=
  Graph.Model.stageEffectCompatible (M := M.core) x y

noncomputable def stageEffectAdd? (M : Model (α := α))
    (x y : M.Carrier) : Option M.Carrier :=
  Graph.Model.stageEffectAdd? (M := M.core) x y

noncomputable def stageOrthocomplement (M : Model (α := α)) :
    M.Carrier → M.Carrier :=
  Graph.Model.stageOrthocomplement (M := M.core)

noncomputable def stageHimp (M : Model (α := α)) :
    M.Carrier → M.Carrier → M.Carrier :=
  Graph.Model.stageHimp (M := M.core)

noncomputable def stageCollapseAt (M : Model (α := α)) (n : ℕ) :
    M.Carrier → M.Carrier :=
  Graph.Model.stageCollapseAt (M := M.core) n

noncomputable def stageExpandAt (M : Model (α := α)) (n : ℕ) :
    M.Carrier → M.Carrier :=
  Graph.Model.stageExpandAt (M := M.core) n

noncomputable def stageOccam (M : Model (α := α)) :
    M.Carrier → M.Carrier :=
  Graph.Model.stageOccam (M := M.core)

end Model

end

end Alexandroff
end Graph
end Bridges
end HeytingLean
