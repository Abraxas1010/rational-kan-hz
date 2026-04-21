import HeytingLean.Contracts.RoundTrip
import HeytingLean.Logic.StageSemantics
import HeytingLean.Logic.Dialectic
import HeytingLean.Epistemic.Occam

/-!
# Tensor bridge

Concrete tensor carriers modelled as finite tuples of `α`, equipped with a round-trip contract that
collapses to the Heyting core via coordinate-wise interiorisation.
-/

namespace HeytingLean
namespace Bridges
namespace Tensor

open HeytingLean.Contracts
open HeytingLean.LoF
open HeytingLean.Epistemic

universe u

/-- Finite tensor point with `n.succ` coordinates in `α`. -/
structure Point (α : Type u) (n : ℕ) where
  coords : Fin (n.succ) → α

namespace Point

variable {α : Type u} {n : ℕ}

instance : CoeFun (Point α n) (fun _ => Fin (n.succ) → α) :=
  ⟨Point.coords⟩

@[simp] lemma coe_mk (f : Fin (n.succ) → α) :
    ((Point.mk f : Point α n) : Fin (n.succ) → α) = f := rfl

@[simp, ext] lemma ext {x y : Point α n} (h : ∀ i, x.coords i = y.coords i) :
    x = y := by
  cases x
  cases y
  simpa using congrArg Point.mk (funext h)

@[simp] lemma eta (p : Point α n) : Point.mk p.coords = p := by cases p; rfl

end Point

section
variable (α : Type u) [PrimaryAlgebra α]

open scoped Classical

/-- Tensor bridge data: dimension together with the core nucleus. -/
structure Model where
  dim : ℕ
  R : Reentry α

namespace Model

open scoped Classical

variable {α : Type u} [PrimaryAlgebra α]

def Carrier (M : Model α) : Type u :=
  Point α M.dim

noncomputable def encode (M : Model α) (a : M.R.Omega) : M.Carrier :=
  Point.mk fun _ => (a : α)

noncomputable def decode (M : Model α) (v : M.Carrier) : M.R.Omega :=
  let value := ⨅ i, v.coords i
  Reentry.Omega.mk (R := M.R) (M.R value) (M.R.idempotent _)

noncomputable def contract (M : Model α) : RoundTrip (R := M.R) M.Carrier where
  encode := M.encode
  decode := M.decode
  round := by
    intro a
    ext
    classical
    simp [encode, decode]

noncomputable def interpret (M : Model α) (v : M.Carrier) : M.Carrier :=
  Point.mk fun i => M.R (v.coords i)

@[simp] lemma interpret_coords (M : Model α) (v : M.Carrier) (i : Fin (M.dim.succ)) :
    (M.interpret v).coords i = M.R (v.coords i) := rfl

lemma interpret_idem (M : Model α) (v : M.Carrier) :
    M.interpret (M.interpret v) = M.interpret v := by
  classical
  apply Point.ext
  intro i
  simp [interpret]

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
lemma eulerBoundary_vector (M : Model α) :
    M.encode M.R.eulerBoundary = Point.mk (fun _ => M.R.primordial) := by
  apply Point.ext; intro i
  simp [Model.encode, Reentry.eulerBoundary_eq_process, Reentry.process_coe]

def pointwiseMin (M : Model α) (v w : M.Carrier) : M.Carrier :=
  Point.mk fun i => v.coords i ⊓ w.coords i

def pointwiseMax (M : Model α) (v w : M.Carrier) : M.Carrier :=
  Point.mk fun i => v.coords i ⊔ w.coords i

@[simp] lemma encode_inf (M : Model α) (a b : M.R.Omega) :
    M.encode (a ⊓ b) = M.pointwiseMin (M.encode a) (M.encode b) := by
  classical
  apply Point.ext; intro i
  simp [encode, pointwiseMin]

/-- Stage-style MV addition lifted to the tensor carrier. -/
noncomputable def stageMvAdd (M : Model α) :
    M.Carrier → M.Carrier → M.Carrier :=
  fun v w =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.base M.R)
        (M.decode v) (M.decode w))

/-- Stage-style effect compatibility viewed on the tensor carrier. -/
def stageEffectCompatible (M : Model α) (v w : M.Carrier) : Prop :=
  HeytingLean.Logic.Stage.DialParam.effectCompatible
    (P := HeytingLean.Logic.Modal.DialParam.base M.R)
    (M.decode v) (M.decode w)

/-- Stage-style partial effect addition on the tensor carrier. -/
noncomputable def stageEffectAdd?
    (M : Model α) (v w : M.Carrier) : Option M.Carrier :=
  (HeytingLean.Logic.Stage.DialParam.effectAdd?
      (P := HeytingLean.Logic.Modal.DialParam.base M.R)
      (M.decode v) (M.decode w)).map M.encode

/-- Stage-style orthocomplement lifted to the tensor carrier. -/
noncomputable def stageOrthocomplement (M : Model α) :
    M.Carrier → M.Carrier :=
  fun v =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.orthocomplement
        (P := HeytingLean.Logic.Modal.DialParam.base M.R)
        (M.decode v))

/-- Stage-style Heyting implication lifted to the tensor carrier. -/
noncomputable def stageHimp (M : Model α) :
    M.Carrier → M.Carrier → M.Carrier :=
  fun v w =>
    M.encode ((M.decode v) ⇨ (M.decode w))

/-- Stage-style collapse (at ladder index `n`) on the tensor carrier. -/
noncomputable def stageCollapseAt (M : Model α) (n : ℕ) :
    M.Carrier → M.Carrier :=
  fun v =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.collapseAtOmega
        (α := α) (R := M.R) n (M.decode v))

/-- Stage-style expansion (at ladder index `n`) on the tensor carrier. -/
noncomputable def stageExpandAt (M : Model α) (n : ℕ) :
    M.Carrier → M.Carrier :=
  fun v =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.expandAtOmega
        (α := α) (R := M.R) n (M.decode v))

/-- Stage-style Occam reduction lifted to the tensor carrier (via the contract). -/
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

@[simp] lemma stageCollapseAt_apply (M : Model α) (n : ℕ)
    (v : M.Carrier) :
    M.stageCollapseAt n v =
      M.encode (M.decode v) := by
  classical
  simp [Model.stageCollapseAt]

@[simp] lemma stageExpandAt_apply (M : Model α) (n : ℕ)
    (v : M.Carrier) :
    M.stageExpandAt n v =
      M.encode (M.decode v) := by
  classical
  simp [Model.stageExpandAt]

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

  -- Dialectic synthesis: tensor bridge hooks
  @[simp] lemma encode_synthOmega (M : Model α) (T A : M.R.Omega) :
      M.encode (HeytingLean.Logic.Dialectic.synthOmega (R := M.R) T A)
        = Point.mk (fun _ => M.R ((T : α) ⊔ (A : α))) := by
    classical
    apply Point.ext; intro i
    simp [Model.encode, HeytingLean.Logic.Dialectic.synthOmega,
      HeytingLean.Logic.Dialectic.synth]

  @[simp] lemma logicalShadow_pointwiseMax_encode
      (M : Model α) (T A : M.R.Omega) :
      M.logicalShadow
          (M.pointwiseMax (M.encode T) (M.encode A))
        = M.R ((T : α) ⊔ (A : α)) := by
    classical
    -- decode takes an infimum across coordinates; for constant coords, it's the constant
    -- expand logicalShadow and the contract components
    simp [Model.logicalShadow, HeytingLean.Contracts.interiorized,
          Model.contract, Model.decode]
    have hconst :
        (fun i : Fin (M.dim.succ) =>
            (M.pointwiseMax (M.encode T) (M.encode A)).coords i)
          = (fun _ => (T : α) ⊔ (A : α)) := by
      funext i; simp [Model.pointwiseMax, Model.encode]
    have hval : (⨅ i, (M.pointwiseMax (M.encode T) (M.encode A)).coords i)
            = ((T : α) ⊔ (A : α)) := by
      simp [hconst]
    simp [hval]

end Model

end

end Tensor
end Bridges
end HeytingLean
