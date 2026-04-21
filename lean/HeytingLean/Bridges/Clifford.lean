import HeytingLean.Contracts.RoundTrip
import HeytingLean.Logic.StageSemantics
import HeytingLean.Logic.Dialectic
import HeytingLean.Epistemic.Occam

/-!
# Clifford bridge

Geometric bridge built from pairs of `α` together with a projector that collapses onto the Heyting
core.
-/

namespace HeytingLean
namespace Bridges
namespace Clifford

open HeytingLean.Contracts
open HeytingLean.LoF
open HeytingLean.Epistemic

universe u

/-- Bivector carrier for the Clifford bridge, storing two components of `α`. -/
structure Pair (α : Type u) where
  fst : α
  snd : α

namespace Pair

variable {α : Type u}

instance : Coe (Pair α) (α × α) := ⟨fun p => (p.fst, p.snd)⟩

@[simp] lemma coe_mk (a b : α) : ((Pair.mk a b : Pair α) : α × α) = (a, b) := rfl

@[simp, ext] lemma ext {x y : Pair α} (h₁ : x.fst = y.fst) (h₂ : x.snd = y.snd) :
    x = y := by
  cases x
  cases y
  cases h₁
  cases h₂
  rfl

@[simp] lemma eta (p : Pair α) : Pair.mk p.fst p.snd = p := by cases p; rfl

@[simp] lemma fst_mk (a b : α) : (Pair.mk a b).fst = a := rfl

@[simp] lemma snd_mk (a b : α) : (Pair.mk a b).snd = b := rfl

end Pair

section
variable (α : Type u) [PrimaryAlgebra α]

open scoped Classical

/-- Clifford bridge model carrying pairs of `α`. -/
structure Model where
  R : Reentry α

namespace Model

variable {α : Type u} [PrimaryAlgebra α]

def Carrier (_M : Model α) : Type u := Pair α

noncomputable def encode (M : Model α) (a : M.R.Omega) : M.Carrier :=
  Pair.mk (a : α) (a : α)

noncomputable def decode (M : Model α) (p : M.Carrier) : M.R.Omega :=
  Reentry.Omega.mk (R := M.R) (M.R p.fst) (M.R.idempotent _)

noncomputable def contract (M : Model α) : RoundTrip (R := M.R) M.Carrier where
  encode := M.encode
  decode := M.decode
  round := by
    intro a
    apply Subtype.ext
    simp [encode, decode]

noncomputable def project (M : Model α) (p : M.Carrier) : M.Carrier :=
  Pair.mk (M.R p.fst) (M.R p.fst)

lemma project_idem (M : Model α) (p : M.Carrier) :
    M.project (M.project p) = M.project p := by
  cases p
  simp [project]

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
lemma encode_eulerBoundary_fst (M : Model α) :
    (M.encode M.R.eulerBoundary).fst = M.R.primordial := by
  simp [Model.encode, Reentry.eulerBoundary_eq_process, Reentry.process_coe]

lemma encode_eulerBoundary_snd (M : Model α) :
    (M.encode M.R.eulerBoundary).snd = M.R.primordial := by
  simp [Model.encode, Reentry.eulerBoundary_eq_process, Reentry.process_coe]

/-- Stage-style MV addition lifted to the Clifford carrier. -/
noncomputable def stageMvAdd (M : Model α) :
    M.Carrier → M.Carrier → M.Carrier :=
  fun p q =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.base M.R)
        (M.decode p) (M.decode q))

/-- Stage-style effect compatibility on the Clifford carrier. -/
def stageEffectCompatible (M : Model α) (p q : M.Carrier) : Prop :=
  HeytingLean.Logic.Stage.DialParam.effectCompatible
    (P := HeytingLean.Logic.Modal.DialParam.base M.R)
    (M.decode p) (M.decode q)

/-- Stage-style partial effect addition on the Clifford carrier. -/
noncomputable def stageEffectAdd?
    (M : Model α) (p q : M.Carrier) : Option M.Carrier :=
  (HeytingLean.Logic.Stage.DialParam.effectAdd?
      (P := HeytingLean.Logic.Modal.DialParam.base M.R)
      (M.decode p) (M.decode q)).map M.encode

/-- Stage-style orthocomplement lifted to the Clifford carrier. -/
noncomputable def stageOrthocomplement (M : Model α) :
    M.Carrier → M.Carrier :=
  fun p =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.orthocomplement
        (P := HeytingLean.Logic.Modal.DialParam.base M.R)
        (M.decode p))

/-- Stage-style Heyting implication lifted to the Clifford carrier. -/
noncomputable def stageHimp (M : Model α) :
    M.Carrier → M.Carrier → M.Carrier :=
  fun p q =>
    M.encode ((M.decode p) ⇨ (M.decode q))

/-- Stage-style collapse (at ladder index `n`) on the Clifford carrier. -/
noncomputable def stageCollapseAt (M : Model α) (n : ℕ) :
    M.Carrier → M.Carrier :=
  fun p =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.collapseAtOmega
        (α := α) (R := M.R) n (M.decode p))

/-- Stage-style expansion (at ladder index `n`) on the Clifford carrier. -/
noncomputable def stageExpandAt (M : Model α) (n : ℕ) :
    M.Carrier → M.Carrier :=
  fun p =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.expandAtOmega
        (α := α) (R := M.R) n (M.decode p))

/-- Stage-style Occam reduction lifted to the Clifford carrier (via the contract). -/
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
          (HeytingLean.Epistemic.occam (R := M.R) (a : α))
          (HeytingLean.Epistemic.occam_idempotent (R := M.R) (a := (a : α)))) := by
  classical
  unfold Model.stageOccam
  exact
    HeytingLean.Contracts.stageOccam_encode
      (R := M.R) (C := M.contract) a

@[simp] lemma stageCollapseAt_apply (M : Model α) (n : ℕ)
    (p : M.Carrier) :
    M.stageCollapseAt n p =
      M.encode (M.decode p) := by
  classical
  simp [Model.stageCollapseAt]

@[simp] lemma stageExpandAt_apply (M : Model α) (n : ℕ)
    (p : M.Carrier) :
    M.stageExpandAt n p =
      M.encode (M.decode p) := by
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

@[simp] lemma project_encode (M : Model α) (a : M.R.Omega) :
    M.project (M.encode a) = M.encode a := by
  classical
  unfold Model.project Model.encode
  simp [Reentry.Omega.apply_coe]

@[simp] lemma project_stageCollapseAt (M : Model α) (n : ℕ)
    (p : M.Carrier) :
    M.project (M.stageCollapseAt n p) = M.stageCollapseAt n p := by
  classical
  unfold Model.stageCollapseAt
  simp [project_encode]

@[simp] lemma project_stageExpandAt (M : Model α) (n : ℕ)
    (p : M.Carrier) :
    M.project (M.stageExpandAt n p) = M.stageExpandAt n p := by
  classical
  unfold Model.stageExpandAt
  simp [project_encode]

@[simp] lemma project_stageOccam (M : Model α) (p : M.Carrier) :
    M.project (M.stageOccam p) = M.stageOccam p := by
  classical
  unfold Model.stageOccam Contracts.stageOccam
  simp [Model.contract, project_encode]

  -- Dialectic synthesis: Clifford bridge hooks
  @[simp] lemma encode_synthOmega_fst (M : Model α) (T A : M.R.Omega) :
      (M.encode (HeytingLean.Logic.Dialectic.synthOmega (R := M.R) T A)).fst =
        M.R ((T : α) ⊔ (A : α)) := by
    classical
    simp [Model.encode, HeytingLean.Logic.Dialectic.synthOmega,
      HeytingLean.Logic.Dialectic.synth]

  @[simp] lemma encode_synthOmega_snd (M : Model α) (T A : M.R.Omega) :
      (M.encode (HeytingLean.Logic.Dialectic.synthOmega (R := M.R) T A)).snd =
        M.R ((T : α) ⊔ (A : α)) := by
    classical
    simp [Model.encode, HeytingLean.Logic.Dialectic.synthOmega,
      HeytingLean.Logic.Dialectic.synth]

end Model

end

end Clifford
end Bridges
end HeytingLean
