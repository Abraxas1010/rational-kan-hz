import HeytingLean.Contracts.RoundTrip
import HeytingLean.Logic.StageSemantics
import HeytingLean.Logic.Dialectic
import HeytingLean.Epistemic.Occam

/-
# Graph bridge

The graph bridge reuses the ambient type `α` as its carrier while exposing the ladder operations via
the existing re-entry nucleus.
-/

namespace HeytingLean
namespace Bridges
namespace Graph

open HeytingLean.Contracts
open HeytingLean.LoF
open HeytingLean.Epistemic

universe u

section
variable (α : Type u) [PrimaryAlgebra α]

open scoped Classical

/-- Graph bridge model: adjacency inherits the order on `α`. -/
structure Model where
  R : Reentry α

namespace Model

variable {α : Type u} [PrimaryAlgebra α]

def adjacency (_M : Model α) : α → α → Prop :=
  (· ≤ ·)

def Carrier (_M : Model α) : Type u := α

noncomputable def encode (M : Model α) (a : M.R.Omega) : M.Carrier :=
  (a : α)

noncomputable def decode (M : Model α) (x : M.Carrier) : M.R.Omega :=
  Reentry.Omega.mk (R := M.R) (M.R x) (M.R.idempotent _)

noncomputable def contract (M : Model α) : RoundTrip (R := M.R) M.Carrier where
  encode := M.encode
  decode := M.decode
  round := by
    intro a
    apply Subtype.ext
    simp [encode, decode]

noncomputable def logicalShadow (M : Model α) : M.Carrier → α :=
  interiorized (R := M.R) M.contract

@[simp] lemma logicalShadow_encode (M : Model α) (a : M.R.Omega) :
    M.logicalShadow (M.contract.encode a) = M.R a := by
  unfold logicalShadow
  exact interiorized_id (R := M.R) (C := M.contract) a

@[simp] lemma logicalShadow_encode' (M : Model α) (a : M.R.Omega) :
    M.logicalShadow (M.encode a) = M.R a := by
  change M.logicalShadow (M.contract.encode a) = M.R a
  exact logicalShadow_encode (M := M) (a := a)

@[simp] lemma decode_encode (M : Model α) (a : M.R.Omega) :
    M.decode (M.contract.encode a) = a := by
  change (M.contract.decode (M.contract.encode a)) = a
  exact M.contract.round a

lemma encode_eulerBoundary (M : Model α) :
    M.encode M.R.eulerBoundary = M.R.primordial := by
  simp [Model.encode, Reentry.eulerBoundary_eq_process, Reentry.process_coe]

lemma adjacency_refl (M : Model α) (a : α) :
    M.adjacency a a := le_rfl

lemma adjacency_trans (M : Model α) {a b c : α}
    (hab : M.adjacency a b) (hbc : M.adjacency b c) :
    M.adjacency a c :=
  le_trans hab hbc

@[simp] lemma adjacency_iff_le (M : Model α) (a b : α) :
    M.adjacency a b ↔ a ≤ b := Iff.rfl

/-- Monotonicity of adjacency in the left argument. -/
lemma adjacency_mono_left (M : Model α) {a b c : α}
    (hab : a ≤ b) (hbc : M.adjacency b c) :
    M.adjacency a c :=
  le_trans hab hbc

/-- Monotonicity of adjacency in the right argument. -/
lemma adjacency_mono_right (M : Model α) {a b c : α}
    (hab : M.adjacency a b) (hbc : b ≤ c) :
    M.adjacency a c :=
  le_trans hab hbc

/-- Stage-style MV addition lifted to the graph carrier. -/
noncomputable def stageMvAdd (M : Model α) :
    M.Carrier → M.Carrier → M.Carrier :=
  fun x y =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.base M.R)
        (M.decode x) (M.decode y))

/-- Stage-style effect compatibility on the graph carrier. -/
def stageEffectCompatible (M : Model α) (x y : M.Carrier) : Prop :=
  HeytingLean.Logic.Stage.DialParam.effectCompatible
    (P := HeytingLean.Logic.Modal.DialParam.base M.R)
    (M.decode x) (M.decode y)

/-- Stage-style partial effect addition on the graph carrier. -/
noncomputable def stageEffectAdd?
    (M : Model α) (x y : M.Carrier) : Option M.Carrier :=
  (HeytingLean.Logic.Stage.DialParam.effectAdd?
      (P := HeytingLean.Logic.Modal.DialParam.base M.R)
      (M.decode x) (M.decode y)).map M.encode

/-- Stage-style orthocomplement lifted to the graph carrier. -/
noncomputable def stageOrthocomplement (M : Model α) :
    M.Carrier → M.Carrier :=
  fun x =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.orthocomplement
        (P := HeytingLean.Logic.Modal.DialParam.base M.R)
        (M.decode x))

/-- Stage-style Heyting implication lifted to the graph carrier. -/
noncomputable def stageHimp (M : Model α) :
    M.Carrier → M.Carrier → M.Carrier :=
  fun x y =>
    M.encode ((M.decode x) ⇨ (M.decode y))

/-- Stage-style collapse (at ladder index `n`) on the graph carrier. -/
noncomputable def stageCollapseAt (M : Model α) (n : ℕ) :
    M.Carrier → M.Carrier :=
  fun x =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.collapseAtOmega
        (α := α) (R := M.R) n (M.decode x))

/-- Stage-style expansion (at ladder index `n`) on the graph carrier. -/
noncomputable def stageExpandAt (M : Model α) (n : ℕ) :
    M.Carrier → M.Carrier :=
  fun x =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.expandAtOmega
        (α := α) (R := M.R) n (M.decode x))

/-- Stage-style Occam reduction lifted to the graph carrier (via the contract). -/
noncomputable def stageOccam (M : Model α) :
    M.Carrier → M.Carrier :=
  Contracts.stageOccam (R := M.R) (C := M.contract)

variable {α : Type u} [PrimaryAlgebra α]

@[simp] theorem stageMvAdd_encode (M : Model α) (a b : M.R.Omega) :
    M.stageMvAdd
        (M.contract.encode a) (M.contract.encode b)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b) := by
  classical
  simp [Model.stageMvAdd, Model.decode_encode]

@[simp] theorem stageEffectCompatible_encode (M : Model α) (a b : M.R.Omega) :
    M.stageEffectCompatible
        (M.contract.encode a) (M.contract.encode b) ↔
      HeytingLean.Logic.Stage.DialParam.effectCompatible
        (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b := by
  simp [Model.stageEffectCompatible, Model.decode_encode]

@[simp] theorem stageEffectAdd_encode (M : Model α) (a b : M.R.Omega) :
    M.stageEffectAdd?
        (M.contract.encode a) (M.contract.encode b)
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b).map
          M.encode := by
  classical
  simp [Model.stageEffectAdd?, Model.decode_encode]

@[simp] theorem stageOrthocomplement_encode (M : Model α) (a : M.R.Omega) :
    M.stageOrthocomplement (M.contract.encode a)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.orthocomplement
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a) := by
  classical
  simp [Model.stageOrthocomplement, Model.decode_encode]

@[simp] lemma stageHimp_encode (M : Model α) (a b : M.R.Omega) :
    M.stageHimp
        (M.contract.encode a) (M.contract.encode b)
      =
        M.encode (a ⇨ b) := by
  classical
  simp [Model.stageHimp, Model.decode_encode]

@[simp] lemma stageCollapseAt_encode (M : Model α) (n : ℕ)
    (a : M.R.Omega) :
    M.stageCollapseAt n (M.contract.encode a)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.collapseAtOmega
            (α := α) (R := M.R) n a) := by
  classical
  simp [Model.stageCollapseAt, Model.decode_encode]

@[simp] lemma stageExpandAt_encode (M : Model α) (n : ℕ)
    (a : M.R.Omega) :
    M.stageExpandAt n (M.contract.encode a)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.expandAtOmega
            (α := α) (R := M.R) n a) := by
  classical
  simp [Model.stageExpandAt, Model.decode_encode]

@[simp] lemma stageOccam_encode (M : Model α) (a : M.R.Omega) :
    M.stageOccam (M.contract.encode a) =
      M.contract.encode
        (Reentry.Omega.mk (R := M.R)
          (Epistemic.occam (R := M.R) (a : α))
          (Epistemic.occam_idempotent (R := M.R) (a := (a : α)))) := by
  classical
  unfold Model.stageOccam
  exact
    HeytingLean.Contracts.stageOccam_encode
      (R := M.R) (C := M.contract) a

@[simp] lemma logicalShadow_stageMvAdd_encode (M : Model α) (a b : M.R.Omega) :
    M.logicalShadow
        (M.stageMvAdd (M.contract.encode a) (M.contract.encode b))
      =
        M.R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b) := by
  classical
  simp [stageMvAdd_encode, Model.logicalShadow_encode']

@[simp] theorem logicalShadow_stageEffectAdd_encode (M : Model α) (a b : M.R.Omega) :
    (M.stageEffectAdd?
        (M.contract.encode a) (M.contract.encode b)).map M.logicalShadow
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b).map
          (fun u => (u : α)) := by
  classical
  unfold Model.stageEffectAdd?
  cases h :
      HeytingLean.Logic.Stage.DialParam.effectAdd?
        (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b with
  | none =>
      simp [h]
  | some u =>
      simp [h, Model.logicalShadow_encode', Reentry.Omega.apply_coe]

@[simp] lemma logicalShadow_stageOrthocomplement_encode
    (M : Model α) (a : M.R.Omega) :
    M.logicalShadow (M.stageOrthocomplement (M.contract.encode a)) =
      M.R
        (HeytingLean.Logic.Stage.DialParam.orthocomplement
          (P := HeytingLean.Logic.Modal.DialParam.base M.R) a) := by
  classical
  simp [stageOrthocomplement_encode, Model.logicalShadow_encode']

@[simp] lemma logicalShadow_stageHimp_encode
    (M : Model α) (a b : M.R.Omega) :
    M.logicalShadow
        (M.stageHimp (M.contract.encode a) (M.contract.encode b)) =
      M.R (a ⇨ b) := by
  classical
  simp [stageHimp_encode, Model.logicalShadow_encode']

@[simp] lemma logicalShadow_stageCollapseAt_encode
    (M : Model α) (n : ℕ) (a : M.R.Omega) :
    M.logicalShadow
        (M.stageCollapseAt n (M.contract.encode a)) =
      M.R
        (HeytingLean.Logic.Modal.DialParam.collapseAt
          (α := α) (R := M.R) n (a : α)) := by
  classical
  simp [Model.logicalShadow_encode']

  @[simp] lemma logicalShadow_stageExpandAt_encode
    (M : Model α) (n : ℕ) (a : M.R.Omega) :
    M.logicalShadow
        (M.stageExpandAt n (M.contract.encode a)) =
      M.R
        (HeytingLean.Logic.Modal.DialParam.expandAt
          (α := α) (R := M.R) n (a : α)) := by
  classical
  simp [Model.logicalShadow_encode']

  -- Dialectic synthesis: graph bridge hook
  @[simp] lemma encode_synthOmega (M : Model α) (T A : M.R.Omega) :
      M.encode (HeytingLean.Logic.Dialectic.synthOmega (R := M.R) T A) =
        M.R ((T : α) ⊔ (A : α)) := by
    classical
    simp [Model.encode, HeytingLean.Logic.Dialectic.synthOmega,
      HeytingLean.Logic.Dialectic.synth]

end Model

end

end Graph
end Bridges
end HeytingLean
