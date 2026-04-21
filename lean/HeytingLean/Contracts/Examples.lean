import HeytingLean.Contracts.RoundTrip
import HeytingLean.Bridges.RoundTrip
import HeytingLean.Bridges.Tensor
import HeytingLean.Bridges.Tensor.Intensity
import HeytingLean.Bridges.Graph
import HeytingLean.Bridges.Graph.Alexandroff
import HeytingLean.Bridges.Clifford
import HeytingLean.Bridges.Clifford.Projector
import Mathlib.Algebra.Star.Basic
import Mathlib.Data.Complex.Basic
import HeytingLean.Logic.Dialectic

/-!
# Contract examples

Simple round-trip examples that exercise the identity bridge.
-/

namespace HeytingLean
namespace Contracts
namespace Examples

open HeytingLean.Bridges
open HeytingLean.Bridges.Tensor
open HeytingLean.Bridges.Graph
open HeytingLean.Bridges.Clifford
open HeytingLean.LoF

universe u

variable (α : Type u) [PrimaryAlgebra α] (R : Reentry α)

def identity : RoundTrip (R := R) α :=
  HeytingLean.Bridges.identityContract (R := R)

@[simp] lemma identity_round (a : R.Omega) :
    (identity α R).decode ((identity α R).encode a) = a := by
  change HeytingLean.Bridges.decode (R := R)
      (HeytingLean.Bridges.encode (R := R) a) = a
  exact HeytingLean.Bridges.decode_encode (R := R) a

@[simp] lemma identity_shadow (a : R.Omega) :
    interiorized (R := R) (identity α R) ((identity α R).encode a) = R a := by
  change interiorized (R := R)
      (HeytingLean.Bridges.identityContract (R := R)) ((HeytingLean.Bridges.identityContract (R := R)).encode a) = R a
  exact interiorized_id (R := R) (C := HeytingLean.Bridges.identityContract (R := R)) a

def tensor (n : ℕ) : HeytingLean.Bridges.Tensor.Model α :=
  ⟨n, R⟩

@[simp] lemma tensor_round (n : ℕ) (a : R.Omega) :
    (HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).decode
        ((HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).encode a) = a := by
  simpa [tensor] using
    (HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).round a

@[simp] lemma tensor_shadow (n : ℕ) (a : R.Omega) :
    (HeytingLean.Bridges.Tensor.Model.logicalShadow (tensor α R n))
        ((HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).encode a) = R a := by
  simp [HeytingLean.Bridges.Tensor.Model.logicalShadow, tensor]

/-- Convenience lemma: the tensor bridge's implication transport reduces via `simp`. -/
@[simp] lemma tensor_shadow_himp (n : ℕ) (a b : R.Omega) :
    (HeytingLean.Bridges.Tensor.Model.logicalShadow (tensor α R n))
      (HeytingLean.Bridges.Tensor.Model.stageHimp
        (tensor α R n)
        ((HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).encode a)
        ((HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).encode b))
      =
        R (a ⇨ b) := by
  classical
  simp [tensor]
def graph : HeytingLean.Bridges.Graph.Model α :=
  ⟨R⟩

@[simp] lemma graph_round (a : R.Omega) :
    (HeytingLean.Bridges.Graph.Model.contract (graph α R)).decode
        ((HeytingLean.Bridges.Graph.Model.contract (graph α R)).encode a) = a := by
  simpa [graph] using
    (HeytingLean.Bridges.Graph.Model.contract (graph α R)).round a

@[simp] lemma graph_shadow (a : R.Omega) :
    (HeytingLean.Bridges.Graph.Model.logicalShadow (graph α R))
        ((HeytingLean.Bridges.Graph.Model.contract (graph α R)).encode a) = R a := by
  simp [HeytingLean.Bridges.Graph.Model.logicalShadow, graph]

/-- Convenience lemma: the graph bridge's implication transport reduces via `simp`. -/
@[simp] lemma graph_shadow_himp (a b : R.Omega) :
    (HeytingLean.Bridges.Graph.Model.logicalShadow (graph α R))
      (HeytingLean.Bridges.Graph.Model.stageHimp
        (graph α R)
        ((HeytingLean.Bridges.Graph.Model.contract (graph α R)).encode a)
        ((HeytingLean.Bridges.Graph.Model.contract (graph α R)).encode b))
      =
        R (a ⇨ b) := by
  classical
  simp [graph]
def clifford : HeytingLean.Bridges.Clifford.Model α :=
  ⟨R⟩

@[simp] lemma clifford_round (a : R.Omega) :
    (HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).decode
        ((HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).encode a) = a := by
  simpa [clifford] using
    (HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).round a

@[simp] lemma clifford_shadow (a : R.Omega) :
    (HeytingLean.Bridges.Clifford.Model.logicalShadow (clifford α R))
        ((HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).encode a) = R a := by
  simp [HeytingLean.Bridges.Clifford.Model.logicalShadow, clifford]

/-- Convenience lemma: the Clifford bridge's implication transport reduces via `simp`. -/
@[simp] lemma clifford_shadow_himp (a b : R.Omega) :
    (HeytingLean.Bridges.Clifford.Model.logicalShadow (clifford α R))
      (HeytingLean.Bridges.Clifford.Model.stageHimp
        (clifford α R)
        ((HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).encode a)
        ((HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).encode b))
      =
        R (a ⇨ b) := by
  classical
  simp [clifford]

/-! ## Dialectic synthesis convenience lemmas across bridges -/

open HeytingLean.Logic.Dialectic

/-- Tensor: the logical shadow of the pointwise sup of two encoded core points
    coincides with `R (T ⊔ A)`; this matches the definition of `synth`. -/
@[simp] lemma tensor_shadow_pointwiseMax_synthesis
    (n : ℕ) (T A : R.Omega) :
    (HeytingLean.Bridges.Tensor.Model.logicalShadow (tensor α R n))
      (HeytingLean.Bridges.Tensor.Model.pointwiseMax
        (tensor α R n)
        (HeytingLean.Bridges.Tensor.Model.encode (tensor α R n) T)
        (HeytingLean.Bridges.Tensor.Model.encode (tensor α R n) A))
    =
      R ((T : α) ⊔ (A : α)) := by
  classical
  simp [tensor]

/-- Graph: encoding `synthOmega` reduces to the nucleus of the join. -/
@[simp] lemma graph_encode_synthOmega (T A : R.Omega) :
    (HeytingLean.Bridges.Graph.Model.encode (graph α R))
      (synthOmega (R := R) T A)
    = R ((T : α) ⊔ (A : α)) := by
  classical
  simpa [graph] using
    (HeytingLean.Bridges.Graph.Model.encode_synthOmega (M := graph α R) T A)

/-- Clifford: each coordinate of the encoded `synthOmega` reduces to the nucleus of the join. -/
@[simp] lemma clifford_encode_synthOmega_fst (T A : R.Omega) :
    (HeytingLean.Bridges.Clifford.Model.encode (clifford α R)
      (synthOmega (R := R) T A)).fst
    = R ((T : α) ⊔ (A : α)) := by
  classical
  simpa [clifford] using
    (HeytingLean.Bridges.Clifford.Model.encode_synthOmega_fst
      (M := clifford α R) T A)

@[simp] lemma clifford_encode_synthOmega_snd (T A : R.Omega) :
    (HeytingLean.Bridges.Clifford.Model.encode (clifford α R)
      (synthOmega (R := R) T A)).snd
    = R ((T : α) ⊔ (A : α)) := by
  classical
  simpa [clifford] using
    (HeytingLean.Bridges.Clifford.Model.encode_synthOmega_snd
      (M := clifford α R) T A)

/-! ## Feature flags and enriched bridge packs -/

/-- Feature toggles controlling which enriched bridge carriers are selected. -/
structure BridgeFlags where
  useTensorIntensity : Bool := False
  useGraphAlexandroff : Bool := False
  useCliffordProjector : Bool := False
  deriving DecidableEq, Repr, Inhabited

/-- Legacy bridge flags with all enrichments disabled. -/
def BridgeFlags.legacy : BridgeFlags := {}

/-- Runtime bridge flags enabling every enriched carrier documented in `Docs/Semantics.md`. -/
def BridgeFlags.runtime : BridgeFlags :=
  { BridgeFlags.legacy with
      useTensorIntensity := True
      useGraphAlexandroff := True
      useCliffordProjector := True }

/-- Default bridge flags (runtime rollout of the enriched carriers). -/
def BridgeFlags.default : BridgeFlags := BridgeFlags.runtime

/-- Toggle enabling only the Alexandroff graph carrier. -/
def alexandroffFlags : BridgeFlags :=
  { BridgeFlags.legacy with useGraphAlexandroff := True }

/-- Toggle enabling only the tensor intensity carrier. -/
def intensityFlags : BridgeFlags :=
  { BridgeFlags.legacy with useTensorIntensity := True }

/-- Bridge execution pack: a carrier together with its round-trip contract. -/
structure BridgePack (R : Reentry α) where
  Carrier : Type u
  contract : RoundTrip (R := R) Carrier

/-- Collection of bridge packs for the tensor, graph, and Clifford lenses. -/
structure BridgeSuite (R : Reentry α) where
  tensor : BridgePack (α := α) (R := R)
  graph  : BridgePack (α := α) (R := R)
  clifford : BridgePack (α := α) (R := R)

open scoped Classical

/-- Canonical projector model used when the projector feature flag is enabled. -/
noncomputable def projectorModel (R : Reentry α) :
    HeytingLean.Bridges.Clifford.Projector.Model (α := α) (β := ℂ) :=
  { core := clifford α R
    projector :=
      { axis := (0 : ℂ)
        idempotent := by simp
        selfAdjoint := by simp } }

@[simp] lemma projectorModel_invariants (R : Reentry α) :
    Bridges.Clifford.Projector.Model.Invariants
      (M := projectorModel (α := α) (R := R)) :=
  Bridges.Clifford.Projector.Model.invariants
    (M := projectorModel (α := α) (R := R))

/-- Convenience flag enabling the projector carrier. -/
def projectorFlags : BridgeFlags :=
  { BridgeFlags.legacy with useCliffordProjector := True }

/-- Default intensity model used when promoting the tensor bridge. -/
noncomputable def tensorIntensityModel (R : Reentry α) :
    Bridges.Tensor.Intensity.Model (α := α) :=
  { core := tensor α R 0
    profile :=
      Bridges.Tensor.Intensity.Profile.ofPoint (α := α)
        { ℓ1 := 0, ℓ2 := 0
          ℓ1_nonneg := le_of_eq rfl
          ℓ2_nonneg := le_of_eq rfl }
        True
        (Bridges.Tensor.Model.encode (M := tensor α R 0) R.process)
    dim_consistent := rfl
    stabilised := by
      intro _
      classical
      simp [Bridges.Tensor.Intensity.Profile.ofPoint,
        Bridges.Tensor.Model.encode, tensor,
        Reentry.process_coe, Reentry.primordial_apply (R := R)] }

/-- Select the graph bridge based on feature flags. -/
noncomputable def graphPack (R : Reentry α)
    (flags : BridgeFlags := BridgeFlags.default) :
    BridgePack (α := α) (R := R) :=
by
  classical
  by_cases h : flags.useGraphAlexandroff
  ·
    let model := Bridges.Graph.Alexandroff.Model.univ
        (α := α) (core := graph α R)
    have hR : model.core.R = R := by
      dsimp [model]
      rfl
    have hcontr : RoundTrip (R := model.core.R)
        (Bridges.Graph.Alexandroff.Model.Carrier (M := model)) :=
      Bridges.Graph.Alexandroff.Model.contract (M := model)
    refine
      { Carrier := Bridges.Graph.Alexandroff.Model.Carrier (M := model)
        contract := ?_ }
    simpa [hR] using hcontr
  · exact
      { Carrier := Bridges.Graph.Model.Carrier (graph α R)
        contract := Bridges.Graph.Model.contract (graph α R) }

/-- Select the tensor bridge based on feature flags. -/
noncomputable def tensorPack (R : Reentry α)
    (flags : BridgeFlags := BridgeFlags.default) :
    BridgePack (α := α) (R := R) :=
by
  classical
  by_cases h : flags.useTensorIntensity
  ·
    let model := tensorIntensityModel (α := α) (R := R)
    have hR : model.core.R = R := by
      dsimp [model, tensorIntensityModel]
      rfl
    have hcontr : RoundTrip (R := model.core.R)
        (Bridges.Tensor.Intensity.Model.Carrier (M := model)) :=
      Bridges.Tensor.Intensity.Model.contract
        (M := model)
        (bounds := model.profile.bounds)
        (normalised := True)
    refine
      { Carrier := Bridges.Tensor.Intensity.Model.Carrier (M := model)
        contract := ?_ }
    simpa [hR] using hcontr
  · exact
      { Carrier := Bridges.Tensor.Model.Carrier (tensor α R 0)
        contract := Bridges.Tensor.Model.contract (tensor α R 0) }

/-- Select the Clifford bridge based on feature flags, returning the appropriate pack. -/
noncomputable def cliffordPack (R : Reentry α)
    (flags : BridgeFlags := BridgeFlags.default) :
    BridgePack (α := α) (R := R) :=
by
  classical
  by_cases h : flags.useCliffordProjector
  ·
    let model := projectorModel (α := α) (R := R)
    have hR : model.core.R = R := by
      dsimp [model, projectorModel]
      rfl
    have hcontr : RoundTrip (R := model.core.R)
        (Bridges.Clifford.Projector.Model.Carrier (M := model)) :=
      Bridges.Clifford.Projector.Model.contract (M := model)
    refine
      { Carrier := Bridges.Clifford.Projector.Model.Carrier (M := model)
        contract := ?_ }
    simpa [hR] using hcontr
  · exact
      { Carrier := Bridges.Clifford.Model.Carrier (clifford α R)
        contract := Bridges.Clifford.Model.contract (clifford α R) }

/-- Assemble bridge packs according to the requested feature flags. -/
noncomputable def selectSuite (R : Reentry α)
    (flags : BridgeFlags := BridgeFlags.default) :
    BridgeSuite (α := α) (R := R) :=
  { tensor := tensorPack (α := α) (R := R) flags
    graph := graphPack (α := α) (R := R) flags
    clifford := cliffordPack (α := α) (R := R) flags }

/-- Legacy bridge suite retaining the pre-rollout carriers. -/
noncomputable def legacySuite (R : Reentry α) :
    BridgeSuite (α := α) (R := R) :=
  selectSuite (α := α) (R := R) BridgeFlags.legacy

/-- Default bridge suite used at runtime, selecting the enriched carriers. -/
noncomputable def defaultSuite (R : Reentry α) :
    BridgeSuite (α := α) (R := R) :=
  selectSuite (α := α) (R := R) BridgeFlags.default

end Examples
end Contracts
end HeytingLean
