import HeytingLean.Contracts.Examples
import HeytingLean.Logic.ModalDial
import HeytingLean.Logic.Triad
import HeytingLean.Logic.ResiduatedLadder
import HeytingLean.Logic.StageSemantics
import HeytingLean.Ontology.Primordial
import HeytingLean.Bridges.Tensor
import HeytingLean.Bridges.Graph
import HeytingLean.Bridges.Clifford
import HeytingLean.Bridges.Tensor.Intensity
import HeytingLean.Bridges.Graph.Alexandroff
import HeytingLean.Bridges.Clifford.Projector
import HeytingLean.LoF.HeytingCore
import HeytingLean.Epistemic.Occam
import HeytingLean.Logic.PSR
import HeytingLean.Logic.Dialectic
import HeytingLean.Runtime.BridgeSuite
import Aesop

open HeytingLean.LoF
open HeytingLean.Ontology
open HeytingLean.Bridges

namespace HeytingLean
namespace Tests

universe u

variable {α : Type u} [PrimaryAlgebra α]

/-- Encoding and decoding through the identity contract returns every state unchanged. -/
theorem identity_round_verified (R : Reentry α) (a : R.Omega) :
    (Contracts.Examples.identity (α := α) R).decode
        ((Contracts.Examples.identity (α := α) R).encode a) = a :=
  Contracts.Examples.identity_round (α := α) (R := R) a

/-- The logical shadow of the tensor example reproduces the ambient reentry on encoded inputs. -/
theorem tensor_shadow_verified (R : Reentry α) (n : ℕ) (a : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow (Contracts.Examples.tensor (α := α) (R := R) n))
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor (α := α) (R := R) n)).encode a)
        = R a :=
  Contracts.Examples.tensor_shadow (α := α) (R := R) n a

/-- Running the tensor contract's logical shadow on an encoded state yields the original process. -/
theorem tensor_rt2_verified (R : Reentry α) (n : ℕ) (a : R.Omega) :
    (Contracts.Examples.tensor (α := α) (R := R) n).logicalShadow
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode a)
      = R a :=
  tensor_shadow_verified (R := R) (n := n) (a := a)

/-- The tensor contract encodes and decodes to the identity on the underlying state space. -/
theorem tensor_round_verified (R : Reentry α) (n : ℕ) (a : R.Omega) :
    (Bridges.Tensor.Model.contract (Contracts.Examples.tensor (α := α) (R := R) n)).decode
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor (α := α) (R := R) n)).encode a)
      = a := by
  classical
  simpa [Contracts.Examples.tensor]
    using Contracts.Examples.tensor_round (α := α) (R := R) (n := n) (a := a)

/-- The tensor intensity model round-trips any state through its contract encoder/decoder. -/
theorem tensor_intensity_round_verified (R : Reentry α) (n : ℕ) (a : R.Omega) :
    let model : Bridges.Tensor.Intensity.Model (α := α) :=
      { core := Contracts.Examples.tensor (α := α) (R := R) n
        profile :=
          Bridges.Tensor.Intensity.Profile.ofPoint (α := α)
            { ℓ1 := 0, ℓ2 := 0, ℓ1_nonneg := le_of_eq rfl, ℓ2_nonneg := le_of_eq rfl }
            True
            (Bridges.Tensor.Model.encode (M := Contracts.Examples.tensor (α := α) (R := R) n)
              R.process)
        dim_consistent := rfl
        stabilised := by
          intro _
          classical
          simp [Contracts.Examples.tensor,
            Bridges.Tensor.Intensity.Profile.ofPoint,
            Bridges.Tensor.Model.encode,
            Reentry.process_coe,
            Reentry.primordial_apply (R := R)] }
    model.decode (model.contract.encode a) = a := by
  classical
  intro model
  change model.decode
      (model.encode (bounds := model.profile.bounds) (normalised := True) a) = a
  exact
    Bridges.Tensor.Intensity.Model.decode_encode
      (M := model) (bounds := model.profile.bounds) (normalised := True) (a := a)

/-- The runtime tensor contract within the bridge suite performs an exact encode/decode round trip. -/
theorem runtime_tensor_round_verified (R : Reentry α) (a : R.Omega) :
    let suite := HeytingLean.Runtime.bridgeSuite (α := α) (R := R)
    suite.tensor.contract.decode (suite.tensor.contract.encode a) = a := by
  classical
  dsimp [HeytingLean.Runtime.bridgeSuite, HeytingLean.Runtime.bridgeFlags,
    Contracts.Examples.selectSuite] -- inline the suite definition
  simp [Contracts.Examples.tensorPack, Contracts.Examples.tensorIntensityModel,
    Contracts.Examples.BridgeFlags.runtime,
    Bridges.Tensor.Intensity.Model.contract,
    Bridges.Tensor.Intensity.Model.decode_encode]

/-- The graph model's logical shadow agrees with the reentry on encoded states. -/
theorem graph_shadow_verified (R : Reentry α) (a : R.Omega) :
    (Bridges.Graph.Model.logicalShadow (Contracts.Examples.graph (α := α) (R := R)))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph (α := α) (R := R))).encode a)
        = R a :=
  Contracts.Examples.graph_shadow (α := α) (R := R) a

/-- Encoding through the graph contract and evaluating its logical shadow recovers the reentry action. -/
theorem graph_rt2_verified (R : Reentry α) (a : R.Omega) :
    (Contracts.Examples.graph (α := α) (R := R)).logicalShadow
        ((Contracts.Examples.graph (α := α) (R := R)).contract.encode a)
      = R a :=
  graph_shadow_verified (R := R) (a := a)

/-- The Alexandroff graph model decodes an encoded state back to the original element. -/
theorem graph_alexandroff_round_verified (R : Reentry α) (a : R.Omega) :
    let model :
        Bridges.Graph.Alexandroff.Model (α := α) :=
      Bridges.Graph.Alexandroff.Model.univ
        (α := α) (core := Contracts.Examples.graph (α := α) (R := R))
    model.decode (model.contract.encode a) = a := by
  classical
  intro model
  exact Bridges.Graph.Alexandroff.Model.decode_encode (M := model) (a := a)

/-- The runtime graph contract in the bridge suite round-trips any state exactly. -/
theorem runtime_graph_round_verified (R : Reentry α) (a : R.Omega) :
    let suite := HeytingLean.Runtime.bridgeSuite (α := α) (R := R)
    suite.graph.contract.decode (suite.graph.contract.encode a) = a := by
  classical
  dsimp [HeytingLean.Runtime.bridgeSuite, HeytingLean.Runtime.bridgeFlags,
    Contracts.Examples.selectSuite]
  simp [Contracts.Examples.graphPack, Contracts.Examples.BridgeFlags.runtime,
    Bridges.Graph.Alexandroff.Model.contract]

/-- Collapsing a stage in the Alexandroff process keeps the point inside the open process region. -/
theorem graph_alexandroff_process_collapse (R : Reentry α)
    (n : ℕ) (x : α)
    (hx :
      (Bridges.Graph.Alexandroff.Model.processUpper
          (α := α) (core := Contracts.Examples.graph (α := α) (R := R))).memOpen x) :
    (Bridges.Graph.Alexandroff.Model.processUpper
        (α := α) (core := Contracts.Examples.graph (α := α) (R := R))).memOpen
      (Bridges.Graph.Model.stageCollapseAt
        (M := Contracts.Examples.graph (α := α) (R := R)) n x) :=
  Bridges.Graph.Alexandroff.Model.mem_stageCollapseAt
    (M :=
      Bridges.Graph.Alexandroff.Model.processUpper
        (α := α) (core := Contracts.Examples.graph (α := α) (R := R)))
    (n := n) hx

/-- Every expansion step of the Alexandroff process preserves membership in the open region. -/
theorem graph_alexandroff_process_expand (R : Reentry α)
    (n : ℕ) (x : α)
    (hx :
      (Bridges.Graph.Alexandroff.Model.processUpper
          (α := α) (core := Contracts.Examples.graph (α := α) (R := R))).memOpen x) :
    (Bridges.Graph.Alexandroff.Model.processUpper
        (α := α) (core := Contracts.Examples.graph (α := α) (R := R))).memOpen
      (Bridges.Graph.Model.stageExpandAt
        (M := Contracts.Examples.graph (α := α) (R := R)) n x) :=
  Bridges.Graph.Alexandroff.Model.mem_stageExpandAt
    (M :=
      Bridges.Graph.Alexandroff.Model.processUpper
        (α := α) (core := Contracts.Examples.graph (α := α) (R := R)))
    (n := n) hx

/-- Applying the Occam stage in the Alexandroff process keeps the state in the open process region. -/
theorem graph_alexandroff_process_occam (R : Reentry α)
    (x : α)
    (hx :
      (Bridges.Graph.Alexandroff.Model.processUpper
          (α := α) (core := Contracts.Examples.graph (α := α) (R := R))).memOpen x) :
    (Bridges.Graph.Alexandroff.Model.processUpper
        (α := α) (core := Contracts.Examples.graph (α := α) (R := R))).memOpen
      (Bridges.Graph.Model.stageOccam
        (M := Contracts.Examples.graph (α := α) (R := R)) x) :=
  Bridges.Graph.Alexandroff.Model.mem_stageOccam
    (M :=
      Bridges.Graph.Alexandroff.Model.processUpper
        (α := α) (core := Contracts.Examples.graph (α := α) (R := R)))
    hx

/-- Triangle (TRI-1): deduction, abduction, and induction coincide on the Heyting core. -/
theorem residuated_triangle_verified (R : Reentry α)
    (a b c : R.Omega) :
    HeytingLean.Logic.Residuated.abduction (R := R) a b c ↔
      HeytingLean.Logic.Residuated.induction (R := R) a b c :=
  HeytingLean.Logic.Residuated.abduction_iff_induction (R := R) a b c

/-- Triangle (TRI-2) for the tensor bridge reduces to the core triangle via RT. -/
theorem tensor_triangle_lens_verified (R : Reentry α) (n : ℕ)
    (a b c : R.Omega) :
    HeytingLean.Logic.Residuated.abduction (R := R)
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode a))
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode b))
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode c)) ↔
      HeytingLean.Logic.Residuated.induction (R := R)
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode a))
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode b))
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode c)) :=
by
  classical
  set M := Contracts.Examples.tensor (α := α) (R := R) n
  have ha := M.contract.round a
  have hb := M.contract.round b
  have hc := M.contract.round c
  simpa [M, ha, hb, hc] using residuated_triangle_verified (R := R) a b c

/-- Triangle (TRI-2) for the graph bridge reduces to the core triangle via the bridge contract. -/
theorem graph_triangle_lens_verified (R : Reentry α)
    (a b c : R.Omega) :
    HeytingLean.Logic.Residuated.abduction (R := R)
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode a))
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode b))
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode c)) ↔
      HeytingLean.Logic.Residuated.induction (R := R)
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode a))
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode b))
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode c)) :=
by
  classical
  set M := Contracts.Examples.graph (α := α) (R := R)
  have ha := M.contract.round a
  have hb := M.contract.round b
  have hc := M.contract.round c
  simpa [M, ha, hb, hc] using residuated_triangle_verified (R := R) a b c

/-- Triangle (TRI-2) for the Clifford bridge reduces to the core triangle via the bridge contract. -/
theorem clifford_triangle_lens_verified (R : Reentry α)
    (a b c : R.Omega) :
    HeytingLean.Logic.Residuated.abduction (R := R)
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode a))
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode b))
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode c)) ↔
      HeytingLean.Logic.Residuated.induction (R := R)
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode a))
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode b))
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode c)) :=
by
  classical
  set M := Contracts.Examples.clifford (α := α) (R := R)
  have ha := M.contract.round a
  have hb := M.contract.round b
  have hc := M.contract.round c
  simpa [M, ha, hb, hc] using residuated_triangle_verified (R := R) a b c

/-- The graph contract encodes and decodes any state to itself. -/
theorem graph_round_verified (R : Reentry α) (a : R.Omega) :
    (Bridges.Graph.Model.contract (Contracts.Examples.graph (α := α) (R := R))).decode
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph (α := α) (R := R))).encode a)
      = a := by
  classical
  simpa [Contracts.Examples.graph]
    using Contracts.Examples.graph_round (α := α) (R := R) (a := a)

/-- Projecting twice with the Clifford model equals projecting once. -/
theorem clifford_project_idem (R : Reentry α) (p : Bridges.Clifford.Pair α) :
    Bridges.Clifford.Model.project (Contracts.Examples.clifford (α := α) (R := R))
        (Bridges.Clifford.Model.project (Contracts.Examples.clifford (α := α) (R := R)) p)
        =
      Bridges.Clifford.Model.project (Contracts.Examples.clifford (α := α) (R := R)) p :=
  Bridges.Clifford.Model.project_idem (M := Contracts.Examples.clifford (α := α) (R := R)) p

/-- The ladder construction at level three has dimension three. -/
theorem ladder_dimension_verified (R : Reentry α) :
    (Logic.Modal.DialParam.ladder (α := α) R 3).dimension = 3 :=
  Logic.Modal.DialParam.ladder_dimension (α := α) R 3

/-- The process and counter-process intersect only at bottom. -/
theorem process_counter_disjoint (R : Reentry α) :
    R.process ⊓ R.counterProcess = ⊥ :=
  Reentry.process_inf_counter (R := R)

/-- The Euler boundary of a reentry equals its process component. -/
theorem euler_boundary_equals_process (R : Reentry α) :
    R.eulerBoundary = R.process :=
  Reentry.eulerBoundary_eq_process (R := R)

/-- The Euler boundary is bounded above by the counter-process whenever the latter
    lies inside the support window. -/
theorem euler_boundary_le_counter (R : Reentry α)
    (hcounter_sup : (R.counter : α) ≤ R.support) :
    R.eulerBoundary ≤ R.counterProcess :=
  Reentry.eulerBoundary_le_counter (R := R) (hcounter_sup := hcounter_sup)

/-- The coordinates of the theta cycle add to zero in ℂ. -/
theorem theta_cycle_zero_sum (θ : ℝ) :
    (thetaCycle θ).1 + (thetaCycle θ).2 = (0 : ℂ) :=
  thetaCycle_zero_sum θ

/-- Encoding the Euler boundary via the tensor model yields the primordial point vector. -/
theorem tensor_encode_euler (R : Reentry α) (n : ℕ) :
    Bridges.Tensor.Model.encode (Contracts.Examples.tensor (α := α) (R := R) n) R.eulerBoundary
      = Bridges.Tensor.Point.mk (fun _ => R.primordial) :=
  Bridges.Tensor.Model.eulerBoundary_vector (Contracts.Examples.tensor (α := α) (R := R) n)

/-- Encoding the Euler boundary via the graph model returns the primordial element. -/
theorem graph_encode_euler (R : Reentry α) :
    Bridges.Graph.Model.encode (Contracts.Examples.graph (α := α) (R := R)) R.eulerBoundary
      = R.primordial :=
  Bridges.Graph.Model.encode_eulerBoundary (Contracts.Examples.graph (α := α) (R := R))

/-- Encoding the Euler boundary through the Clifford model produces a pair of primordial points. -/
theorem clifford_encode_euler (R : Reentry α) :
    Bridges.Clifford.Model.encode (Contracts.Examples.clifford (α := α) (R := R)) R.eulerBoundary
      = Bridges.Clifford.Pair.mk R.primordial R.primordial := by
  classical
  simp [Bridges.Clifford.Model.encode, Contracts.Examples.clifford,
    Reentry.eulerBoundary_eq_process, Reentry.process_coe]

/-- The Clifford projector model decodes any encoded state back to the original element. -/
theorem clifford_projector_round_verified (R : Reentry α) (a : R.Omega) :
    let model : Bridges.Clifford.Projector.Model (α := α) (β := ℂ) :=
      { core := Contracts.Examples.clifford (α := α) (R := R)
        projector :=
          { axis := (0 : ℂ)
            idempotent := by simp
            selfAdjoint := by simp } }
  model.decode (model.encode a) = a := by
  classical
  intro model
  exact Bridges.Clifford.Projector.Model.decode_encode (M := model) (a := a)

/-- The runtime Clifford contract in the bridge suite performs an exact encode/decode round trip. -/
theorem runtime_clifford_round_verified (R : Reentry α) (a : R.Omega) :
    let suite := HeytingLean.Runtime.bridgeSuite (α := α) (R := R)
    suite.clifford.contract.decode (suite.clifford.contract.encode a) = a := by
  classical
  dsimp [HeytingLean.Runtime.bridgeSuite, HeytingLean.Runtime.bridgeFlags,
    Contracts.Examples.selectSuite]
  simp [Contracts.Examples.cliffordPack, Contracts.Examples.projectorModel,
    Contracts.Examples.BridgeFlags.runtime,
    Bridges.Clifford.Projector.Model.contract,
    Bridges.Clifford.Projector.Model.decode_encode]

/-- The Clifford projector example satisfies all projector invariants. -/
theorem clifford_projector_invariants_verified (R : Reentry α) :
    let model := Contracts.Examples.projectorModel (α := α) (R := R)
    Bridges.Clifford.Projector.Model.Invariants (M := model) := by
  classical
  intro model
  simpa using Bridges.Clifford.Projector.Model.invariants (M := model)

/-- Stage collapse in the projector model keeps states within the projected carrier. -/
theorem clifford_projector_collapse_closed (R : Reentry α)
    (n : ℕ)
    (c :
      Bridges.Clifford.Projector.Model.Carrier
        (M := Contracts.Examples.projectorModel (α := α) (R := R))) :
    Bridges.Clifford.Projector.Model.projected
        (M := Contracts.Examples.projectorModel (α := α) (R := R))
        (Bridges.Clifford.Projector.Model.Carrier.toPair
          (Bridges.Clifford.Projector.Model.stageCollapseAt
            (M := Contracts.Examples.projectorModel (α := α) (R := R)) n c)) := by
  classical
  have h :=
    Bridges.Clifford.Projector.Model.invariants
      (M := Contracts.Examples.projectorModel (α := α) (R := R))
  exact h.collapseClosed n c

/-- Stage expansion in the projector model preserves membership in the projected carrier. -/
theorem clifford_projector_expand_closed (R : Reentry α)
    (n : ℕ)
    (c :
      Bridges.Clifford.Projector.Model.Carrier
        (M := Contracts.Examples.projectorModel (α := α) (R := R))) :
    Bridges.Clifford.Projector.Model.projected
        (M := Contracts.Examples.projectorModel (α := α) (R := R))
        (Bridges.Clifford.Projector.Model.Carrier.toPair
          (Bridges.Clifford.Projector.Model.stageExpandAt
            (M := Contracts.Examples.projectorModel (α := α) (R := R)) n c)) := by
  classical
  have h :=
    Bridges.Clifford.Projector.Model.invariants
      (M := Contracts.Examples.projectorModel (α := α) (R := R))
  exact h.expandClosed n c

/-- The projector model remains projected under the Occam stage. -/
theorem clifford_projector_occam_closed (R : Reentry α)
    (c :
      Bridges.Clifford.Projector.Model.Carrier
        (M := Contracts.Examples.projectorModel (α := α) (R := R))) :
    Bridges.Clifford.Projector.Model.projected
        (M := Contracts.Examples.projectorModel (α := α) (R := R))
        (Bridges.Clifford.Projector.Model.Carrier.toPair
          (Bridges.Clifford.Projector.Model.stageOccam
            (M := Contracts.Examples.projectorModel (α := α) (R := R)) c)) := by
  classical
  have h :=
    Bridges.Clifford.Projector.Model.invariants
      (M := Contracts.Examples.projectorModel (α := α) (R := R))
  exact h.occamClosed c

/-- The projector axis element is already projected. -/
theorem clifford_projector_axis_closed (R : Reentry α) :
    Bridges.Clifford.Projector.Model.projected
        (M := Contracts.Examples.projectorModel (α := α) (R := R))
        (Bridges.Clifford.Projector.Model.Carrier.toPair
          ((Contracts.Examples.projectorModel (α := α) (R := R)).encode
            (Contracts.Examples.projectorModel (α := α) (R := R)).core.R.eulerBoundary)) := by
  classical
  have h :=
    Bridges.Clifford.Projector.Model.invariants
      (M := Contracts.Examples.projectorModel (α := α) (R := R))
  simpa using h.axisClosed

/-- The packaged Clifford projector contract round-trips any state. -/
theorem clifford_pack_projector_round_verified (R : Reentry α) (a : R.Omega) :
    (Contracts.Examples.cliffordPack (α := α) (R := R)
        Contracts.Examples.projectorFlags).contract.decode
      ((Contracts.Examples.cliffordPack (α := α) (R := R)
          Contracts.Examples.projectorFlags).contract.encode a) = a := by
  classical
  unfold Contracts.Examples.cliffordPack Contracts.Examples.projectorFlags
  simp [Contracts.Examples.projectorModel,
    Bridges.Clifford.Projector.Model.contract,
    Bridges.Clifford.Projector.Model.decode_encode]

/-- The packaged tensor intensity contract encodes and decodes to the identity. -/
theorem tensor_pack_intensity_round_verified (R : Reentry α) (a : R.Omega) :
    (Contracts.Examples.tensorPack (α := α) (R := R)
        Contracts.Examples.intensityFlags).contract.decode
      ((Contracts.Examples.tensorPack (α := α) (R := R)
          Contracts.Examples.intensityFlags).contract.encode a) = a := by
  classical
  change (Contracts.Examples.tensorIntensityModel (α := α) (R := R)).decode
      ((Contracts.Examples.tensorIntensityModel (α := α) (R := R)).contract.encode a) = a
  simp [Contracts.Examples.tensorIntensityModel,
    Bridges.Tensor.Intensity.Model.contract,
    Bridges.Tensor.Intensity.Model.decode_encode]

/-- The packaged graph Alexandroff contract decodes every encoded state to itself. -/
theorem graph_pack_alexandroff_round_verified (R : Reentry α) (a : R.Omega) :
    (Contracts.Examples.graphPack (α := α) (R := R)
        Contracts.Examples.alexandroffFlags).contract.decode
      ((Contracts.Examples.graphPack (α := α) (R := R)
          Contracts.Examples.alexandroffFlags).contract.encode a) = a := by
  classical
  change (Bridges.Graph.Alexandroff.Model.univ
      (α := α) (core := Contracts.Examples.graph (α := α) (R := R))).decode
    ((Bridges.Graph.Alexandroff.Model.univ
        (α := α) (core := Contracts.Examples.graph (α := α) (R := R))).contract.encode a) = a
  exact Bridges.Graph.Alexandroff.Model.decode_encode
    (M := Bridges.Graph.Alexandroff.Model.univ
        (α := α) (core := Contracts.Examples.graph (α := α) (R := R))) (a := a)

/-- Applying `R` to a Heyting implication inside the core leaves it unchanged. -/
theorem residuation_himp_closed (R : Reentry α) (a b : R.Omega) :
    R ((a : α) ⇨ (b : α)) = (a : α) ⇨ (b : α) :=
  HeytingLean.Logic.Residuated.himp_closed (R := R) (a := a) (b := b)

/-- Mapping a Heyting implication through `R` yields an upper bound by implicating the image of the conclusion. -/
theorem residuation_himp_le (R : Reentry α) (a b : α) :
    R (a ⇨ b) ≤ a ⇨ R b :=
  HeytingLean.Logic.Residuated.map_himp_le (R := R) (a := a) (b := b)

/-- The boolean ladder base preserves Heyting implications inside the core. -/
theorem ladder_boolean_himp_closed (R : Reentry α)
    (a b : (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 0).dial.core.Omega) :
    R ((a : α) ⇨ (b : α)) = (a : α) ⇨ (b : α) :=
  residuation_himp_closed (R := R) (a := a) (b := b)

/-- Every ladder stage preserves Heyting implications on its core. -/
theorem ladder_himp_closed (R : Reentry α) (n : ℕ)
    (a b :
      (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R n).dial.core.Omega) :
    (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R n).dial.core
        ((a : α) ⇨ (b : α)) =
      (a : α) ⇨ (b : α) :=
  HeytingLean.Logic.Stage.DialParam.himp_closed
    (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R n)
    (a := a) (b := b)

/-- The Euler boundary of any reentry has Occam birth zero. -/
theorem occam_birth_euler_boundary (R : Reentry α) :
    HeytingLean.Epistemic.birth R ((R.eulerBoundary : R.Omega) : α) = 0 := by
  exact HeytingLean.Epistemic.birth_eulerBoundary (R := R)

/-- The Euler boundary element satisfies the PSR sufficiency predicate. -/
theorem psr_sufficient_euler_boundary (R : Reentry α) :
    HeytingLean.Logic.PSR.Sufficient R ((R.eulerBoundary : R.Omega) : α) := by
  exact HeytingLean.Logic.PSR.sufficient_eulerBoundary (R := R)

/-- Breathing from a PSR-sufficient bound stays below that bound. -/
theorem psr_breathe_le (R : Reentry α) (a x : α)
    (ha : HeytingLean.Logic.PSR.Sufficient R a) (hx : x ≤ a) (n : ℕ) :
    HeytingLean.Epistemic.breathe (R := R) n x ≤ a :=
  HeytingLean.Logic.PSR.breathe_le_of_sufficient
    (R := R) (a := a) (x := x) ha hx n

/-- Iterated breathing below the Euler boundary never escapes it. -/
theorem psr_breathe_euler_boundary (R : Reentry α) (x : α)
    (hx : x ≤ (R.eulerBoundary : α)) (n : ℕ) :
    HeytingLean.Epistemic.breathe (R := R) n x ≤ (R.eulerBoundary : α) := by
  have hsuff := psr_sufficient_euler_boundary (R := R)
  have hx' : x ≤ ((R.eulerBoundary : R.Omega) : α) := hx
  simpa using psr_breathe_le
    (R := R)
    (a := ((R.eulerBoundary : R.Omega) : α))
    (x := x)
    hsuff hx' n

/-- Once a PSR reachability path stays under a sufficient bound, it remains under that bound. -/
theorem psr_reachable_stable (R : Reentry α) (a x y : α)
    (ha : HeytingLean.Logic.PSR.Sufficient R a) (hx : x ≤ a)
    (hy : HeytingLean.Logic.PSR.reachable (R := R) x y) :
    y ≤ a :=
  HeytingLean.Logic.PSR.sufficient_reachable
    (R := R) (a := a) (x := x) (y := y) ha hx hy

/-- Dialectic synthesis with the Euler boundary returns that boundary. -/
theorem dialectic_synth_euler_boundary (R : Reentry α) :
    HeytingLean.Logic.Dialectic.synthOmega (R := R) R.eulerBoundary R.eulerBoundary =
      R.eulerBoundary :=
  HeytingLean.Logic.Dialectic.synthOmega_self (R := R)

/-- Under a boolean reentry, evaluating via the equivalence returns the original element. -/
theorem boolean_limit_verified (R : Reentry α) (h : ∀ a : α, R a = a) (a : α) :
    R (((_root_.HeytingLean.LoF.Reentry.booleanEquiv (R := R) h).symm a) : R.Omega) = a :=
  _root_.HeytingLean.LoF.Reentry.boolean_limit (R := R) h a

/-- Adding the MV bottom element leaves any dialect element unchanged. -/
theorem mv_add_bottom_verified (P : Logic.Modal.DialParam α)
    (a : P.dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.mvAdd (P := P) ⊥ a = a := by
  change
      HeytingLean.Logic.Stage.DialParam.mvAdd (P := P)
        (HeytingLean.Logic.Stage.DialParam.mvZero (P := P)) a = a
  exact HeytingLean.Logic.Stage.DialParam.mvAdd_zero_left (P := P) (a := a)

/-- Effect-level addition with bottom is defined and returns the original element. -/
theorem effect_add_bottom_verified (P : Logic.Modal.DialParam α)
    (a : P.dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.effectAdd? (P := P) ⊥ a = some a := by
  classical
  simp [HeytingLean.Logic.Stage.DialParam.effectAdd?,
    HeytingLean.Logic.Stage.DialParam.effectCompatible,
    HeytingLean.Logic.Stage.DialParam.mvAdd]

/-- Each element is compatible with its orthocomplement in the effect stage. -/
theorem orthocomplement_disjoint_verified
    (P : Logic.Modal.DialParam α) (a : P.dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.effectCompatible (P := P) a
        (HeytingLean.Logic.Stage.DialParam.orthocomplement (P := P) a) := by
  exact
    HeytingLean.Logic.Stage.DialParam.effectCompatible_orthocomplement
      (P := P) (a := a)

/-- Boolean-stage exemplar: MV addition with bottom is neutral. -/
theorem ladder_boolean_mv_zero (R : Reentry α)
    (a : (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 0).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 0)
        (⊥) a = a := by
  exact
    mv_add_bottom_verified
      (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 0)
      (a := a)

/-- MV-stage exemplar: addition commutes at the second ladder level. -/
theorem ladder_mv_comm (R : Reentry α)
    (a b : (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2) a b =
      HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2) b a := by
  classical
  exact
    HeytingLean.Logic.Stage.DialParam.mvAdd_comm
      (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2)
      (a := a) (b := b)

/-- Effect-stage exemplar: adding an element to its orthocomplement is defined. -/
theorem ladder_effect_add_orthocomplement (R : Reentry α)
    (a : (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.effectAdd?
        (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3)
        a
        (HeytingLean.Logic.Stage.DialParam.orthocomplement
          (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3) a)
      =
        some
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3) a
            (HeytingLean.Logic.Stage.DialParam.orthocomplement
              (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3) a)) := by
  classical
  exact
    HeytingLean.Logic.Stage.DialParam.effectAdd?_orthocomplement
      (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3)
      (a := a)

/-- Orthomodular-stage exemplar: elements are disjoint from their orthocomplements. -/
theorem ladder_orthomodular_disjoint (R : Reentry α)
    (a : (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.effectCompatible
        (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4) a
        (HeytingLean.Logic.Stage.DialParam.orthocomplement
          (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4) a) :=
  HeytingLean.Logic.Stage.DialParam.effectCompatible_orthocomplement
    (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4) (a := a)

/-- MV-stage collapse halves back to the Heyting core. -/
theorem ladder_mv_collapse (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.collapseAtOmega
        (α := α) (R := R) 2 a = a :=
  HeytingLean.Logic.Stage.DialParam.mvCollapse_self
    (α := α) (R := R) (a := a)

/-- MV-stage expansion also returns to the Heyting core. -/
theorem ladder_mv_expand (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.expandAtOmega
        (α := α) (R := R) 2 a = a :=
  HeytingLean.Logic.Stage.DialParam.mvExpand_self
    (α := α) (R := R) (a := a)

/-- Effect-stage collapse returns to the Heyting core. -/
theorem ladder_effect_collapse (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.collapseAtOmega
        (α := α) (R := R) 3 a = a :=
  HeytingLean.Logic.Stage.DialParam.effectCollapse_self
    (α := α) (R := R) (a := a)

/-- Effect-stage expansion returns to the Heyting core. -/
theorem ladder_effect_expand (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.expandAtOmega
        (α := α) (R := R) 3 a = a :=
  HeytingLean.Logic.Stage.DialParam.effectExpand_self
    (α := α) (R := R) (a := a)

/-- Orthomodular-stage collapse returns to the Heyting core. -/
theorem ladder_orth_collapse (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.collapseAtOmega
        (α := α) (R := R) 4 a = a :=
  HeytingLean.Logic.Stage.DialParam.orthCollapse_self
    (α := α) (R := R) (a := a)

/-- Orthomodular-stage expansion returns to the Heyting core. -/
theorem ladder_orth_expand (R : Reentry α)
    (a :
      (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.expandAtOmega
        (α := α) (R := R) 4 a = a :=
  HeytingLean.Logic.Stage.DialParam.orthExpand_self
    (α := α) (R := R) (a := a)

-- Bridge transport lemmas (`@[simp]`) so compliance proofs can defer to automation.
@[simp] lemma tensor_shadow_mv_add (R : Reentry α) (n : ℕ)
    (a b : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      (Bridges.Tensor.Model.stageMvAdd
        (Contracts.Examples.tensor (α := α) (R := R) n)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode a)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode b))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b) := by
  classical
  exact
    Bridges.Tensor.Model.logicalShadow_stageMvAdd_encode
      (M := Contracts.Examples.tensor (α := α) (R := R) n)
      (a := a) (b := b)

@[simp] lemma tensor_shadow_effect_add (R : Reentry α) (n : ℕ)
    (a b : R.Omega) :
    (Bridges.Tensor.Model.stageEffectAdd?
        (Contracts.Examples.tensor (α := α) (R := R) n)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode a)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode b)).map
      (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b).map
          (fun u => (u : α)) := by
  classical
  exact
    Bridges.Tensor.Model.logicalShadow_stageEffectAdd_encode
      (M := Contracts.Examples.tensor (α := α) (R := R) n)
      (a := a) (b := b)

@[simp] lemma tensor_shadow_collapseAt (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      (Bridges.Tensor.Model.stageCollapseAt
        (Contracts.Examples.tensor (α := α) (R := R) n) n
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode a))
      =
        R
          (HeytingLean.Logic.Modal.DialParam.collapseAt
            (α := α) (R := R) n (a : α)) := by
  classical
  exact
    Bridges.Tensor.Model.logicalShadow_stageCollapseAt_encode
      (M := Contracts.Examples.tensor (α := α) (R := R) n)
      (n := n) (a := a)

@[simp] lemma tensor_shadow_expandAt (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      (Bridges.Tensor.Model.stageExpandAt
        (Contracts.Examples.tensor (α := α) (R := R) n) n
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode a))
      =
        R
          (HeytingLean.Logic.Modal.DialParam.expandAt
            (α := α) (R := R) n (a : α)) := by
  classical
  exact
    Bridges.Tensor.Model.logicalShadow_stageExpandAt_encode
      (M := Contracts.Examples.tensor (α := α) (R := R) n)
      (n := n) (a := a)

@[simp] lemma tensor_shadow_occam (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      (Bridges.Tensor.Model.stageOccam
        (Contracts.Examples.tensor (α := α) (R := R) n)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode a))
      =
        Epistemic.occam (R := R) (a : α) := by
  classical
  have := Contracts.stageOccam_spec (R := R)
    (C := (Contracts.Examples.tensor (α := α) (R := R) n).contract)
    (b := (Bridges.Tensor.Model.contract (Contracts.Examples.tensor
      (α := α) (R := R) n)).encode a)
  simpa [Bridges.Tensor.Model.stageOccam,
    Bridges.Tensor.Model.logicalShadow,
    Contracts.stageOccam,
    Contracts.interiorized]

/-- Taking the tensor stage `himp` and then its logical shadow matches implicating inside `R`. -/
lemma tensor_shadow_himp (R : Reentry α) (n : ℕ)
    (a b : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      (Bridges.Tensor.Model.stageHimp
        (Contracts.Examples.tensor (α := α) (R := R) n)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode a)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode b))
      =
        R (a ⇨ b) := by
  classical
  exact
    Contracts.Examples.tensor_shadow_himp
      (α := α) (R := R) (n := n) (a := a) (b := b)

@[simp] lemma graph_shadow_mv_add (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Graph.Model.logicalShadow
        (Contracts.Examples.graph (α := α) (R := R)))
      (Bridges.Graph.Model.stageMvAdd
        (Contracts.Examples.graph (α := α) (R := R))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode a)
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode b))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b) := by
  classical
  exact
    Bridges.Graph.Model.logicalShadow_stageMvAdd_encode
      (M := Contracts.Examples.graph (α := α) (R := R))
      (a := a) (b := b)

@[simp] lemma graph_shadow_effect_add (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Graph.Model.stageEffectAdd?
        (Contracts.Examples.graph (α := α) (R := R))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode a)
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode b)).map
      (Bridges.Graph.Model.logicalShadow
        (Contracts.Examples.graph (α := α) (R := R)))
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b).map
          (fun u => (u : α)) := by
  classical
  exact
    Bridges.Graph.Model.logicalShadow_stageEffectAdd_encode
      (M := Contracts.Examples.graph (α := α) (R := R))
      (a := a) (b := b)

@[simp] lemma graph_shadow_collapseAt (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    (Bridges.Graph.Model.logicalShadow
        (Contracts.Examples.graph (α := α) (R := R)))
      (Bridges.Graph.Model.stageCollapseAt
        (Contracts.Examples.graph (α := α) (R := R)) n
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode a))
      =
        R
          (HeytingLean.Logic.Modal.DialParam.collapseAt
            (α := α) (R := R) n (a : α)) := by
  classical
  exact
    Bridges.Graph.Model.logicalShadow_stageCollapseAt_encode
      (M := Contracts.Examples.graph (α := α) (R := R))
      (n := n) (a := a)

@[simp] lemma graph_shadow_expandAt (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    (Bridges.Graph.Model.logicalShadow
        (Contracts.Examples.graph (α := α) (R := R)))
      (Bridges.Graph.Model.stageExpandAt
        (Contracts.Examples.graph (α := α) (R := R)) n
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode a))
      =
        R
          (HeytingLean.Logic.Modal.DialParam.expandAt
            (α := α) (R := R) n (a : α)) := by
  classical
  exact
    Bridges.Graph.Model.logicalShadow_stageExpandAt_encode
      (M := Contracts.Examples.graph (α := α) (R := R))
      (n := n) (a := a)

@[simp] lemma graph_shadow_occam (R : Reentry α)
    (a : R.Omega) :
    (Bridges.Graph.Model.logicalShadow
        (Contracts.Examples.graph (α := α) (R := R)))
      (Bridges.Graph.Model.stageOccam
        (Contracts.Examples.graph (α := α) (R := R))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode a))
      =
        Epistemic.occam (R := R) (a : α) := by
  classical
  have := Contracts.stageOccam_spec (R := R)
    (C := (Contracts.Examples.graph (α := α) (R := R)).contract)
    (b := (Bridges.Graph.Model.contract (Contracts.Examples.graph
      (α := α) (R := R))).encode a)
  simpa [Bridges.Graph.Model.stageOccam,
    Bridges.Graph.Model.logicalShadow,
    Contracts.stageOccam,
    Contracts.interiorized]

@[simp] theorem graph_shadow_himp (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Graph.Model.logicalShadow
        (Contracts.Examples.graph (α := α) (R := R)))
      (Bridges.Graph.Model.stageHimp
        (Contracts.Examples.graph (α := α) (R := R))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode a)
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode b))
      =
        R (a ⇨ b) :=
  Contracts.Examples.graph_shadow_himp
      (α := α) (R := R) (a := a) (b := b)

@[simp] theorem clifford_shadow_mv_add (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Clifford.Model.logicalShadow
        (Contracts.Examples.clifford (α := α) (R := R)))
      (Bridges.Clifford.Model.stageMvAdd
        (Contracts.Examples.clifford (α := α) (R := R))
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a)
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode b))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b) :=
  Bridges.Clifford.Model.logicalShadow_stageMvAdd_encode
      (M := Contracts.Examples.clifford (α := α) (R := R)) (a := a) (b := b)

@[simp] theorem clifford_shadow_effect_add (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Clifford.Model.stageEffectAdd?
        (Contracts.Examples.clifford (α := α) (R := R))
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a)
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode b)).map
      (Bridges.Clifford.Model.logicalShadow
        (Contracts.Examples.clifford (α := α) (R := R)))
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b).map
          (fun u => (u : α)) :=
  Bridges.Clifford.Model.logicalShadow_stageEffectAdd_encode
      (M := Contracts.Examples.clifford (α := α) (R := R)) (a := a) (b := b)

@[simp] lemma clifford_shadow_collapseAt (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    (Bridges.Clifford.Model.logicalShadow
        (Contracts.Examples.clifford (α := α) (R := R)))
      (Bridges.Clifford.Model.stageCollapseAt
        (Contracts.Examples.clifford (α := α) (R := R)) n
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a))
      =
        R
          (HeytingLean.Logic.Modal.DialParam.collapseAt
            (α := α) (R := R) n (a : α)) := by
  classical
  exact
    Bridges.Clifford.Model.logicalShadow_stageCollapseAt_encode
      (M := Contracts.Examples.clifford (α := α) (R := R))
      (n := n) (a := a)

@[simp] lemma clifford_shadow_expandAt (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    (Bridges.Clifford.Model.logicalShadow
        (Contracts.Examples.clifford (α := α) (R := R)))
      (Bridges.Clifford.Model.stageExpandAt
        (Contracts.Examples.clifford (α := α) (R := R)) n
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a))
      =
        R
          (HeytingLean.Logic.Modal.DialParam.expandAt
            (α := α) (R := R) n (a : α)) := by
  classical
  exact
    Bridges.Clifford.Model.logicalShadow_stageExpandAt_encode
      (M := Contracts.Examples.clifford (α := α) (R := R))
      (n := n) (a := a)

@[simp] lemma clifford_shadow_occam (R : Reentry α)
    (a : R.Omega) :
    (Bridges.Clifford.Model.logicalShadow
        (Contracts.Examples.clifford (α := α) (R := R)))
      (Bridges.Clifford.Model.stageOccam
        (Contracts.Examples.clifford (α := α) (R := R))
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a))
      =
        Epistemic.occam (R := R) (a : α) := by
  classical
  have := Contracts.stageOccam_spec (R := R)
    (C := (Contracts.Examples.clifford (α := α) (R := R)).contract)
    (b := (Bridges.Clifford.Model.contract (Contracts.Examples.clifford
      (α := α) (R := R))).encode a)
  simpa [Bridges.Clifford.Model.stageOccam,
    Bridges.Clifford.Model.logicalShadow,
    Contracts.stageOccam,
    Contracts.interiorized]

@[simp] lemma clifford_shadow_himp (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Clifford.Model.logicalShadow
        (Contracts.Examples.clifford (α := α) (R := R)))
      (Bridges.Clifford.Model.stageHimp
        (Contracts.Examples.clifford (α := α) (R := R))
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a)
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode b))
      =
        R (a ⇨ b) := by
  classical
  simpa using
    Contracts.Examples.clifford_shadow_himp
      (α := α) (R := R) (a := a) (b := b)

/-- The Clifford contract encodes and decodes any state to itself. -/
theorem clifford_round_verified (R : Reentry α) (a : R.Omega) :
    (Bridges.Clifford.Model.contract (Contracts.Examples.clifford (α := α) (R := R))).decode
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a)
      = a := by
  classical
  set M := Contracts.Examples.clifford (α := α) (R := R)
  change
      M.contract.decode (M.contract.encode a) = a
  exact Contracts.Examples.clifford_round (α := α) (R := R) (a := a)

/-! ### Cross-lens concurrency helpers -/

namespace TraceConcurrency

open Contracts.Examples
open List

inductive BridgeOp
  | tensor
  | graph
  | clifford
  deriving DecidableEq

variable {α : Type u} [PrimaryAlgebra α] (R : Reentry α)

structure BridgeState (suite : BridgeSuite (α := α) (R := R)) where
  tensor : suite.tensor.Carrier
  graph : suite.graph.Carrier
  clifford : suite.clifford.Carrier

@[simp] noncomputable def bridgeStep (suite : BridgeSuite (α := α) (R := R))
    : BridgeOp → BridgeState (R := R) suite → BridgeState (R := R) suite
  | BridgeOp.tensor, st =>
      { st with
        tensor :=
          Contracts.stageOccam (R := R)
            (C := suite.tensor.contract) st.tensor }
  | BridgeOp.graph, st =>
      { st with
        graph :=
          Contracts.stageOccam (R := R)
            (C := suite.graph.contract) st.graph }
  | BridgeOp.clifford, st =>
      { st with
        clifford :=
          Contracts.stageOccam (R := R)
            (C := suite.clifford.contract) st.clifford }

lemma bridgeStep_comm (suite : BridgeSuite (α := α) (R := R))
    {a b : BridgeOp} (h : a ≠ b)
    (st : BridgeState (R := R) suite) :
    bridgeStep (R := R) suite b (bridgeStep (R := R) suite a st)
      = bridgeStep (R := R) suite a (bridgeStep (R := R) suite b st) := by
  cases st with
  | mk tensor graph clifford =>
      cases a <;> cases b <;> simp [bridgeStep, *] at h ⊢

@[simp] noncomputable def bridgeActWord (suite : BridgeSuite (α := α) (R := R)) :
    List BridgeOp → BridgeState (R := R) suite → BridgeState (R := R) suite
  | [], st => st
  | op :: ops, st => bridgeActWord suite ops (bridgeStep (R := R) suite op st)

/-- Acting on a bridge word for `l ++ w` equals acting on `w` after `l`. -/
lemma bridgeActWord_append (suite : BridgeSuite (α := α) (R := R))
    (l w : List BridgeOp)
    (st : BridgeState (R := R) suite) :
    bridgeActWord (R := R) suite (l ++ w) st =
      bridgeActWord (R := R) suite w
        (bridgeActWord (R := R) suite l st) := by
  induction l generalizing st with
  | nil =>
      simp [bridgeActWord]
  | cons op l ih =>
      simp [bridgeActWord, List.cons_append, ih]

/-- Swapping two adjacent, distinct bridge operations leaves the action unchanged. -/
lemma bridgeActWord_swap_adjacent (suite : BridgeSuite (α := α) (R := R))
    {a b : BridgeOp} (h : a ≠ b)
    (tail : List BridgeOp)
    (st : BridgeState (R := R) suite) :
    bridgeActWord (R := R) suite (a :: b :: tail) st =
      bridgeActWord (R := R) suite (b :: a :: tail) st := by
  classical
  have h_step :=
    bridgeStep_comm (R := R) suite (a := a) (b := b) h st
  simpa [bridgeActWord] using
    congrArg (fun st' => bridgeActWord (R := R) suite tail st') h_step

/-- Swapping a distinct adjacent pair inside any prefix leaves `bridgeActWord` unchanged. -/
lemma bridgeActWord_swap_prefix (suite : BridgeSuite (α := α) (R := R))
    {a b : BridgeOp} (h : a ≠ b)
    (front : List BridgeOp) (tail : List BridgeOp)
    (st : BridgeState (R := R) suite) :
    bridgeActWord (R := R) suite (front ++ (a :: b :: tail)) st =
      bridgeActWord (R := R) suite (front ++ (b :: a :: tail)) st := by
  classical
  have := bridgeActWord_swap_adjacent
    (R := R) (suite := suite) (h := h) (tail := tail)
    (st := bridgeActWord (R := R) suite front st)
  simpa [bridgeActWord_append, List.append_assoc] using this

/-- The bridge action depends only on the permutation class of the word of operations. -/
lemma bridgeActWord_of_perm (suite : BridgeSuite (α := α) (R := R))
    {ops ops' : List BridgeOp}
    (h : ops ~ ops')
    (st : BridgeState (R := R) suite) :
    bridgeActWord (R := R) suite ops st =
      bridgeActWord (R := R) suite ops' st := by
  classical
  revert st
  induction h with
  | nil =>
      intro st
      simp [bridgeActWord]
  | cons a h ih =>
      intro st
      simpa [bridgeActWord] using ih (st := bridgeStep (R := R) suite a st)
  | swap a b tail =>
      intro st
      by_cases h' : a = b
      · subst h'
        simp [bridgeActWord]
      · exact (bridgeActWord_swap_adjacent
          (R := R) (suite := suite) (h := h') (tail := tail) (st := st)).symm
  | trans _ _ ih₁ ih₂ =>
      intro st
      exact (ih₁ st).trans (ih₂ st)

@[simp] lemma bridgeActWord_singleton_tensor
    (suite : BridgeSuite (α := α) (R := R))
    (st : BridgeState (R := R) suite) :
    bridgeActWord (R := R) suite [BridgeOp.tensor] st =
      bridgeStep (R := R) suite BridgeOp.tensor st := by
  simp [bridgeActWord]

/-- Applying tensor then graph steps equals the graph-then-tensor sequence. -/
lemma bridge_tensor_graph_commute
    (flags : BridgeFlags := BridgeFlags.default)
    (suite := selectSuite (α := α) (R := R) flags)
    (st : BridgeState (R := R) suite) :
    bridgeStep (R := R) suite BridgeOp.tensor
      (bridgeStep (R := R) suite BridgeOp.graph st)
      =
    bridgeStep (R := R) suite BridgeOp.graph
      (bridgeStep (R := R) suite BridgeOp.tensor st) := by
  classical
  exact bridgeStep_comm (R := R) suite (a := BridgeOp.tensor)
    (b := BridgeOp.graph) (st := st) (by decide)

/-- Graph and Clifford bridge steps commute when acting on the same state. -/
lemma bridge_graph_clifford_commute
    (flags : BridgeFlags := BridgeFlags.default)
    (suite := selectSuite (α := α) (R := R) flags)
    (st : BridgeState (R := R) suite) :
    bridgeStep (R := R) suite BridgeOp.graph
      (bridgeStep (R := R) suite BridgeOp.clifford st)
      =
    bridgeStep (R := R) suite BridgeOp.clifford
      (bridgeStep (R := R) suite BridgeOp.graph st) := by
  classical
  exact bridgeStep_comm (R := R) suite (a := BridgeOp.graph)
    (b := BridgeOp.clifford) (st := st) (by decide)

/-- Tensor and Clifford bridge steps commute when acting on any state. -/
lemma bridge_tensor_clifford_commute
    (flags : BridgeFlags := BridgeFlags.default)
    (suite := selectSuite (α := α) (R := R) flags)
    (st : BridgeState (R := R) suite) :
    bridgeStep (R := R) suite BridgeOp.tensor
      (bridgeStep (R := R) suite BridgeOp.clifford st)
      =
    bridgeStep (R := R) suite BridgeOp.clifford
      (bridgeStep (R := R) suite BridgeOp.tensor st) := by
  classical
  exact bridgeStep_comm (R := R) suite (a := BridgeOp.tensor)
    (b := BridgeOp.clifford) (st := st) (by decide)

end TraceConcurrency

open TraceConcurrency
open Contracts.Examples
open List

@[simp] lemma bridge_occam_swap_tensor_graph
    (flags : BridgeFlags := BridgeFlags.default)
    (suite := selectSuite (α := α) (R := R) flags)
    (st : TraceConcurrency.BridgeState (R := R) suite) :
    TraceConcurrency.bridgeActWord (R := R) suite
        [BridgeOp.tensor, BridgeOp.graph] st
      =
    TraceConcurrency.bridgeActWord (R := R) suite
        [BridgeOp.graph, BridgeOp.tensor] st := by
  classical
  have hperm : List.Perm [BridgeOp.tensor, BridgeOp.graph] [BridgeOp.graph, BridgeOp.tensor] :=
    (List.Perm.swap BridgeOp.tensor BridgeOp.graph ([] : List BridgeOp)).symm
  exact TraceConcurrency.bridgeActWord_of_perm
    (R := R) (suite := suite) hperm st

@[simp] lemma bridge_occam_rotate_tensor_graph_clifford
    (flags : BridgeFlags := BridgeFlags.default)
    (suite := selectSuite (α := α) (R := R) flags)
    (st : TraceConcurrency.BridgeState (R := R) suite) :
    TraceConcurrency.bridgeActWord (R := R) suite
        [BridgeOp.tensor, BridgeOp.graph, BridgeOp.clifford] st
      =
    TraceConcurrency.bridgeActWord (R := R) suite
        [BridgeOp.graph, BridgeOp.clifford, BridgeOp.tensor] st := by
  classical
  have h₁ :
      List.Perm [BridgeOp.tensor, BridgeOp.graph, BridgeOp.clifford]
        [BridgeOp.graph, BridgeOp.tensor, BridgeOp.clifford] :=
    (List.Perm.swap BridgeOp.tensor BridgeOp.graph [BridgeOp.clifford]).symm
  have h₂ :
      List.Perm [BridgeOp.graph, BridgeOp.tensor, BridgeOp.clifford]
        [BridgeOp.graph, BridgeOp.clifford, BridgeOp.tensor] :=
    List.Perm.cons BridgeOp.graph
      ((List.Perm.swap BridgeOp.tensor BridgeOp.clifford ([] : List BridgeOp)).symm)
  have hperm := h₁.trans h₂
  exact TraceConcurrency.bridgeActWord_of_perm
    (R := R) (suite := suite) hperm st

end Tests
end HeytingLean
attribute [simp] HeytingLean.Tests.TraceConcurrency.bridgeActWord_of_perm
attribute [simp] HeytingLean.Tests.TraceConcurrency.bridgeActWord_singleton_tensor
attribute [simp] HeytingLean.Bridges.Graph.Model.adjacency_iff_le

attribute [aesop safe apply]
  HeytingLean.Bridges.Graph.Model.adjacency_refl
  HeytingLean.Bridges.Graph.Model.adjacency_trans
  HeytingLean.Bridges.Graph.Model.adjacency_mono_left
  HeytingLean.Bridges.Graph.Model.adjacency_mono_right
