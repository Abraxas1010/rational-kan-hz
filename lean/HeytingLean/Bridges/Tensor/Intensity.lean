import Mathlib.Data.Real.Basic
import HeytingLean.Bridges.Tensor

open HeytingLean.LoF

/-
Scaffolding for tensor bridge carrier upgrades. Intensity vectors will refine the existing tensor
point model with explicit norm bounds so transport contracts can target the upgraded structure.
-/

namespace HeytingLean
namespace Bridges
namespace Tensor
namespace Intensity

universe u

/-- Norm witnesses for intensity vectors. The bounds are stored for future proofs about collapse
and expand operators respecting the intended ℓ¹/ℓ² controls. -/
structure Bounds where
  ℓ1 : ℝ
  ℓ2 : ℝ
  ℓ1_nonneg : 0 ≤ ℓ1
  ℓ2_nonneg : 0 ≤ ℓ2

/-- Raw intensity profile: coordinates living over a fixed dimension together with their bounds.
The `normalised` field is an abstract predicate (no default) so that
concrete instances can record ℓ¹/ℓ²-style constraints without forcing
any particular numeric structure at this level. -/
structure Profile (α : Type u) where
  dim : ℕ
  coords : Fin (dim.succ) → α
  bounds : Bounds
  normalised : Prop

namespace Profile

variable {α : Type u}

/-- Build a profile directly from a tensor point and metadata. -/
def ofPoint (bounds : Bounds) (normalised : Prop)
    {n : ℕ} (v : Point α n) : Profile α :=
  { dim := n
    coords := v.coords
    bounds := bounds
    normalised := normalised }

/-- The profile viewed as the existing tensor point carrier. -/
def asPoint (p : Profile α) : Point α p.dim :=
  Point.mk p.coords

@[simp] lemma asPoint_apply (p : Profile α) (i : Fin (p.dim.succ)) :
    p.asPoint i = p.coords i := rfl

@[simp] lemma asPoint_ofPoint (bounds : Bounds) (normalised : Prop)
    {n : ℕ} (v : Point α n) :
    (ofPoint (α := α) bounds normalised v).asPoint =
      (v : Point α n) := by
  apply Point.ext
  intro i
  simp [ofPoint, asPoint]

end Profile

section
variable {α : Type u} [PrimaryAlgebra α]

/-- Intensity-aware tensor model: remembers the legacy tensor bridge together with the target
profile, ensuring the upgrade data stays in sync with the existing round-trip contract. -/
structure Model where
  core : Tensor.Model α
  profile : Profile α
  dim_consistent : profile.dim = core.dim
  stabilised :
    ∀ i : Fin (profile.dim.succ),
      core.R (profile.coords i) = profile.coords i

namespace Model

@[simp] lemma dim_consistent' (M : Model (α := α)) :
    M.profile.dim = M.core.dim :=
  M.dim_consistent

/-- Present the profile as a tensor point over the core model's dimension. -/
def intensityPoint (M : Model (α := α)) : Point α M.core.dim :=
  M.dim_consistent ▸ M.profile.asPoint

/-- Intensity-aware carrier bundling a profile compatible with the core dimension. -/
structure Carrier (M : Model (α := α)) where
  profile : Profile α
  dim_ok : profile.dim = M.core.dim
  coords_fixed :
    ∀ i : Fin (profile.dim.succ),
      M.core.R (profile.coords i) = profile.coords i

namespace Carrier

variable {M : Model (α := α)}

@[simp] lemma dim_eq (c : Carrier M) :
    c.profile.dim = M.core.dim :=
  c.dim_ok

/-- View the carrier as a point in the core tensor model. -/
def toPoint (c : Carrier M) : Point α M.core.dim :=
  c.dim_ok ▸ c.profile.asPoint

/-- Build a carrier by encoding an element of the fixed-point core. -/
noncomputable def fromOmega
    (bounds : Bounds)
    (normalised : Prop)
    (a : M.core.R.Omega) : Carrier M :=
  { profile :=
      Profile.ofPoint (α := α) bounds normalised
        (Tensor.Model.encode (M := M.core) a)
    dim_ok := rfl
    coords_fixed := by
      intro i
      classical
      simp [Profile.ofPoint, Tensor.Model.encode,
        Reentry.Omega.apply_coe] }

@[simp] lemma toPoint_fromOmega
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True)
    (a : M.core.R.Omega) :
    (fromOmega (M := M) bounds normalised a).toPoint =
      Tensor.Model.encode (M := M.core) a := by
  simp [fromOmega, toPoint]

end Carrier

variable (M : Model (α := α))

/-- Encode into the intensity carrier, reusing the core encode result. -/
noncomputable def encode
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True) :
    M.core.R.Omega → Carrier M :=
  fun a =>
    Carrier.fromOmega (M := M) bounds normalised a

/-- Decode an intensity carrier by delegating to the core bridge. -/
noncomputable def decode (c : Carrier M) : M.core.R.Omega :=
  Tensor.Model.decode (M := M.core) (Carrier.toPoint c)

/-- Round-trip decoding the encoded carrier recovers the original fixed point. -/
@[simp] lemma decode_encode
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True)
    (a : M.core.R.Omega) :
    M.decode (M.encode bounds normalised a) = a := by
  unfold Model.decode Model.encode
  simp [Carrier.toPoint_fromOmega]
  exact Tensor.Model.decode_encode (M := M.core) (a := a)

/-- Round-trip contract on the intensity carrier, powered by the core transport. -/
noncomputable def contract
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True) :
    Contracts.RoundTrip (M.core.R) (Carrier M) :=
  { encode := M.encode bounds normalised
    decode := M.decode
    round := by
      intro a
      unfold encode decode
      simp [Carrier.toPoint_fromOmega]
      exact Tensor.Model.decode_encode (M := M.core) (a := a) }

/-- Logical shadow lifted to the intensity carrier. -/
noncomputable def logicalShadow :
    Carrier M → α :=
  fun c =>
    Tensor.Model.logicalShadow (M := M.core) (Carrier.toPoint c)

@[simp] lemma logicalShadow_encode
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True)
    (a : M.core.R.Omega) :
    M.logicalShadow (M.encode bounds normalised a) =
      M.core.R a := by
  unfold Model.logicalShadow Model.encode
  simp [Carrier.toPoint_fromOmega, Tensor.Model.logicalShadow_encode']

/-- Stage-style MV addition lifted to the intensity carrier. -/
noncomputable def stageMvAdd
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True) :
    Carrier M → Carrier M → Carrier M :=
  fun v w =>
    M.encode bounds normalised
      (HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.base M.core.R)
        (M.decode v) (M.decode w))

/-- Stage-style effect compatibility on the intensity carrier. -/
def stageEffectCompatible (v w : Carrier M) : Prop :=
  Tensor.Model.stageEffectCompatible (M := M.core)
    (Carrier.toPoint v) (Carrier.toPoint w)

/-- Stage-style partial effect addition lifted to the intensity carrier. -/
noncomputable def stageEffectAdd?
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True)
    (v w : Carrier M) : Option (Carrier M) :=
  (HeytingLean.Logic.Stage.DialParam.effectAdd?
      (P := HeytingLean.Logic.Modal.DialParam.base M.core.R)
      (M.decode v) (M.decode w)).map
    (M.encode bounds normalised)

/-- Stage-style orthocomplement lifted to the intensity carrier. -/
noncomputable def stageOrthocomplement
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True) :
    Carrier M → Carrier M :=
  fun v =>
    M.encode bounds normalised
      (HeytingLean.Logic.Stage.DialParam.orthocomplement
        (P := HeytingLean.Logic.Modal.DialParam.base M.core.R)
        (M.decode v))

/-- Stage-style Heyting implication lifted to the intensity carrier. -/
noncomputable def stageHimp
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True)
    (v w : Carrier M) : Carrier M :=
  M.encode bounds normalised
    ((M.decode v) ⇨ (M.decode w))

/-- Stage-style collapse lifted to the intensity carrier. -/
noncomputable def stageCollapseAt
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True)
    (n : ℕ) :
    Carrier M → Carrier M :=
  fun v =>
    M.encode bounds normalised
      (HeytingLean.Logic.Stage.DialParam.collapseAtOmega
        (α := α) (R := M.core.R) n (M.decode v))

/-- Stage-style expansion lifted to the intensity carrier. -/
noncomputable def stageExpandAt
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True)
    (n : ℕ) :
    Carrier M → Carrier M :=
  fun v =>
    M.encode bounds normalised
      (HeytingLean.Logic.Stage.DialParam.expandAtOmega
        (α := α) (R := M.core.R) n (M.decode v))

/-- Stage-style Occam reduction lifted to the intensity carrier. -/
noncomputable def stageOccam
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True) :
    Carrier M → Carrier M :=
  Contracts.stageOccam (R := M.core.R)
    (C := M.contract bounds normalised)

end Model

variable {α : Type u} [PrimaryAlgebra α]

/-- Stage-style MV addition on the intensity carrier agrees with the core MV addition on Ω. -/
@[simp] lemma Model.stageMvAdd_encode
    (M : Model (α := α))
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True)
    (a b : M.core.R.Omega) :
    M.stageMvAdd bounds normalised
        (M.encode bounds normalised a)
        (M.encode bounds normalised b)
      =
        M.encode bounds normalised
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base M.core.R) a b) := by
  classical
  unfold Model.stageMvAdd
  -- Reduce to equality of the Ω-arguments inside `encode`.
  apply congrArg (M.encode bounds normalised)
  simp [Model.decode_encode]

/-- Occam reduction on the intensity carrier reduces to the core Occam operation on Ω. -/
@[simp] lemma Model.stageOccam_encode
    (M : Model (α := α))
    (bounds : Bounds := M.profile.bounds)
    (normalised : Prop := True)
    (a : M.core.R.Omega) :
    M.stageOccam bounds normalised (M.encode bounds normalised a) =
      M.encode bounds normalised
        (Reentry.Omega.mk (R := M.core.R)
          (Epistemic.occam (R := M.core.R) (a : α))
          (Epistemic.occam_idempotent (R := M.core.R) (a := (a : α)))) := by
  classical
  unfold Model.stageOccam
  exact
    HeytingLean.Contracts.stageOccam_encode
      (R := M.core.R)
      (C := M.contract bounds normalised)
      (a := a)

end

end Intensity
end Tensor
end Bridges
end HeytingLean
