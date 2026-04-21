import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Real.Basic
import HeytingLean.LoF.Nucleus

/-!
# Primordial ontology

This module encodes the metamathematical narrative:
- `ReentryKernel` captures the self-referential act of distinction as an idempotent map.
- `primordialOscillation` realises the Euler monad `e^{iθ}`.
- Complementary lemmas show antiphasic cancellation, aligning with the idea of recursive zeros.
- `dialecticPair` packages the oscillation and its counter-process.
- `zero_sum` states the recursive-zero condition.
- `supported` links these abstract pieces back to an actual `Reentry` nucleus.
- `supported_oscillation` instantiates the oscillation proof for any nucleus.

These formal objects mirror the refined ontological sequence: Distinction-as-Re-entry → Dialectic → Euler boundary → Recursive zero.
-/

namespace HeytingLean
namespace Ontology

open Complex
open scoped Real

noncomputable section

/-- An abstract self-referential distinction. -/
structure ReentryKernel (α : Type u) where
  distinction : α → α
  self_reference : ∀ a, distinction (distinction a) = distinction a

/-- Every nucleus yields a self-referential kernel. -/
def Reentry.kernel {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) : ReentryKernel α :=
  { distinction := R
    self_reference := by
      intro a
      exact R.idempotent a }

/-- The fundamental oscillation `e^{iθ}`. -/
def primordialOscillation (θ : ℝ) : ℂ :=
  Complex.exp (Complex.I * θ)

lemma primordialOscillation_expansion (θ : ℝ) :
    primordialOscillation θ = Real.cos θ + Complex.I * Real.sin θ := by
  unfold primordialOscillation
  simpa [mul_comm, Complex.exp_mul_I, Complex.ofReal_mul, Complex.ofReal_cos,
    Complex.ofReal_sin, mul_left_comm, mul_assoc] using
    (Complex.exp_mul_I (x := (θ : ℝ)))

lemma primordialOscillation_norm_one (θ : ℝ) :
    ‖primordialOscillation θ‖ = 1 := by
  unfold primordialOscillation
  simpa [mul_comm] using (Complex.norm_exp_ofReal_mul_I θ)

@[simp] lemma oscillation_antiphase (θ : ℝ) :
    primordialOscillation (θ + _root_.Real.pi) = - primordialOscillation θ := by
  have hmul :
      Complex.I * (θ + _root_.Real.pi) = Complex.I * θ + Complex.I * _root_.Real.pi := by
    simp [mul_add]
  have hcalc :
      primordialOscillation (θ + _root_.Real.pi)
          = Complex.exp (Complex.I * θ)
              * Complex.exp (Complex.I * _root_.Real.pi) := by
    simp [primordialOscillation, hmul, Complex.exp_add]
  have hpi : Complex.exp (Complex.I * _root_.Real.pi) = (-1 : ℂ) := by
    simpa [mul_comm] using Complex.exp_pi_mul_I
  simpa [primordialOscillation, hpi, mul_comm, mul_left_comm, mul_assoc,
    mul_neg_one] using hcalc

lemma oscillation_pair_cancel (θ : ℝ) :
    primordialOscillation θ + primordialOscillation (θ + _root_.Real.pi) = 0 := by
  simp [oscillation_antiphase]

/-- Process and counter-process bundled as a single datum. -/
def dialecticPair (θ : ℝ) : ℂ × ℂ :=
  ⟨primordialOscillation θ, primordialOscillation (θ + _root_.Real.pi)⟩

lemma zero_sum (θ : ℝ) :
    (dialecticPair θ).1 + (dialecticPair θ).2 = 0 :=
  oscillation_pair_cancel θ

/-- A `Reentry` nucleus supports the Euler oscillation. -/
structure Supported (α : Type u) [LoF.PrimaryAlgebra α] where
  kernel : ReentryKernel α
  enhances : ℝ → ℂ
  counter : ℝ → ℂ
  cancel : ∀ θ, enhances θ + counter θ = 0

/-- Every nucleus gives a supported oscillation via `e^{iθ}`. -/
def supported_oscillation {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) : Supported α :=
  { kernel := Reentry.kernel (R := R)
    enhances := primordialOscillation
    counter := fun θ => primordialOscillation (θ + _root_.Real.pi)
    cancel := oscillation_pair_cancel }

/-- Ontological driver contract: the re-entry kernel witness is present. -/
def ReentryKernelWitness {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) : Prop :=
  ∃ k : ReentryKernel α, k = Reentry.kernel (R := R)

/-- Ontological driver contract: the supported oscillation witness is present. -/
def SupportedOscillationWitness {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) : Prop :=
  ∃ s : Supported α, s = supported_oscillation (R := R)

lemma reentryKernelWitness_holds {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) :
    ReentryKernelWitness (R := R) := by
  exact ⟨Reentry.kernel (R := R), rfl⟩

lemma supportedOscillationWitness_holds {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) :
    SupportedOscillationWitness (R := R) := by
  exact ⟨supported_oscillation (R := R), rfl⟩

/-- Euler-form witness predicate for an oscillation process. -/
def EulerPhaseLaw (ψ : ℝ → ℂ) : Prop :=
  ∀ θ, ψ θ = Real.cos θ + Complex.I * Real.sin θ

/-- Global characterization: satisfying Euler's law is equivalent to being
exactly the canonical primordial oscillation. -/
theorem eulerPhaseLaw_iff_eq_primordialOscillation (ψ : ℝ → ℂ) :
    EulerPhaseLaw ψ ↔ ψ = primordialOscillation := by
  constructor
  · intro h
    funext θ
    calc
      ψ θ = Real.cos θ + Complex.I * Real.sin θ := h θ
      _ = primordialOscillation θ := (primordialOscillation_expansion θ).symm
  · intro h θ
    simpa [h] using primordialOscillation_expansion θ

lemma EulerPhaseLaw.norm_one {ψ : ℝ → ℂ} (hψ : EulerPhaseLaw ψ) :
    ∀ θ, ‖ψ θ‖ = 1 := by
  intro θ
  have hEq : ψ = primordialOscillation :=
    (eulerPhaseLaw_iff_eq_primordialOscillation ψ).1 hψ
  simpa [hEq] using primordialOscillation_norm_one θ

lemma EulerPhaseLaw.antiphase {ψ : ℝ → ℂ} (hψ : EulerPhaseLaw ψ) :
    ∀ θ, ψ (θ + _root_.Real.pi) = -ψ θ := by
  intro θ
  have hEq : ψ = primordialOscillation :=
    (eulerPhaseLaw_iff_eq_primordialOscillation ψ).1 hψ
  rw [hEq]
  exact oscillation_antiphase θ

lemma EulerPhaseLaw.pair_cancel {ψ : ℝ → ℂ} (hψ : EulerPhaseLaw ψ) :
    ∀ θ, ψ θ + ψ (θ + _root_.Real.pi) = 0 := by
  intro θ
  have hEq : ψ = primordialOscillation :=
    (eulerPhaseLaw_iff_eq_primordialOscillation ψ).1 hψ
  rw [hEq]
  exact oscillation_pair_cancel θ

/-- Canonical equivalence theorem:
for any re-entry nucleus `R`, its supported oscillation process is equivalent
to the Euler phase law. -/
theorem reentry_nucleus_euler_formula_equiv {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) :
    (∀ θ, (supported_oscillation (R := R)).enhances θ = primordialOscillation θ) ↔
      EulerPhaseLaw ((supported_oscillation (R := R)).enhances) := by
  constructor
  · intro h θ
    calc
      (supported_oscillation (R := R)).enhances θ = primordialOscillation θ := h θ
      _ = Real.cos θ + Complex.I * Real.sin θ := primordialOscillation_expansion θ
  · intro _ θ
    rfl

/-- Exact-form equivalence specialized to the re-entry-supported oscillation. -/
theorem reentry_nucleus_euler_formula_exact {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) :
    EulerPhaseLaw ((supported_oscillation (R := R)).enhances) ↔
      (supported_oscillation (R := R)).enhances = primordialOscillation := by
  simpa using
    (eulerPhaseLaw_iff_eq_primordialOscillation
      ((supported_oscillation (R := R)).enhances))

/--
Direct correspondence for queue edge:
R-nucleus Euler exactness <-> ontological Euler-phase-law driver.
-/
theorem thm_sheaf_glue_r_nucleus_re_entry_nucleus_euler_formula_exactness_to_ontological_driver_eulerphaselaw
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    (supported_oscillation (R := R)).enhances = primordialOscillation ↔
      EulerPhaseLaw ((supported_oscillation (R := R)).enhances) := by
  simpa [Iff.comm] using (reentry_nucleus_euler_formula_exact (R := R))

/--
Reverse direct correspondence for queue edge:
ontological Euler-phase-law driver <-> R-nucleus Euler exactness.
-/
theorem thm_sheaf_glue_ontological_driver_eulerphaselaw_to_r_nucleus_re_entry_nucleus_euler_formula_exactness
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    EulerPhaseLaw ((supported_oscillation (R := R)).enhances) ↔
      (supported_oscillation (R := R)).enhances = primordialOscillation := by
  exact reentry_nucleus_euler_formula_exact (R := R)

/--
Direct correspondence for queue edge:
ontological reentry-kernel driver <-> ontological supported-oscillation driver.
-/
theorem thm_sheaf_glue_ontological_driver_reentrykernel_to_ontological_driver_supported_supported_oscillation
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    ReentryKernelWitness (R := R) ↔ SupportedOscillationWitness (R := R) := by
  constructor
  · intro _
    exact supportedOscillationWitness_holds (R := R)
  · intro _
    exact reentryKernelWitness_holds (R := R)

/--
Direct correspondence for queue edge:
ontological supported-oscillation driver <-> ontological reentry-kernel driver.
-/
theorem thm_sheaf_glue_ontological_driver_supported_supported_oscillation_to_ontological_driver_reentrykernel
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    SupportedOscillationWitness (R := R) ↔ ReentryKernelWitness (R := R) := by
  simpa [Iff.comm] using
    (thm_sheaf_glue_ontological_driver_reentrykernel_to_ontological_driver_supported_supported_oscillation
      (R := R))

/--
Direct correspondence for queue edge:
ontological reentry-kernel driver <-> ontological Euler-phase-law driver.
-/
  theorem thm_sheaf_glue_ontological_driver_reentrykernel_to_ontological_driver_eulerphaselaw
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    ReentryKernelWitness (R := R) ↔
      EulerPhaseLaw ((supported_oscillation (R := R)).enhances) := by
  constructor
  · intro _ θ
    simpa [supported_oscillation] using primordialOscillation_expansion θ
  · intro _
    exact reentryKernelWitness_holds (R := R)

/--
Direct correspondence for queue edge:
ontological Euler-phase-law driver <-> ontological reentry-kernel driver.
-/
theorem thm_sheaf_glue_ontological_driver_eulerphaselaw_to_ontological_driver_reentrykernel
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    EulerPhaseLaw ((supported_oscillation (R := R)).enhances) ↔
      ReentryKernelWitness (R := R) := by
  simpa [Iff.comm] using
    (thm_sheaf_glue_ontological_driver_reentrykernel_to_ontological_driver_eulerphaselaw
      (R := R))

/--
Direct correspondence for queue edge:
ontological supported-oscillation driver <-> ontological Euler-phase-law driver.
-/
theorem thm_sheaf_glue_ontological_driver_supported_supported_oscillation_to_ontological_driver_eulerphaselaw
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    SupportedOscillationWitness (R := R) ↔
      EulerPhaseLaw ((supported_oscillation (R := R)).enhances) := by
  constructor
  · intro _ θ
    simpa [supported_oscillation] using primordialOscillation_expansion θ
  · intro _
    exact supportedOscillationWitness_holds (R := R)

/--
Direct correspondence for queue edge:
ontological Euler-phase-law driver <-> ontological supported-oscillation driver.
-/
theorem thm_sheaf_glue_ontological_driver_eulerphaselaw_to_ontological_driver_supported_supported_oscillation
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    EulerPhaseLaw ((supported_oscillation (R := R)).enhances) ↔
      SupportedOscillationWitness (R := R) := by
  simpa [Iff.comm] using
    (thm_sheaf_glue_ontological_driver_supported_supported_oscillation_to_ontological_driver_eulerphaselaw
      (R := R))

/--
Direct correspondence for queue edge:
ontological reentry-kernel driver <-> R-nucleus Euler exactness.
-/
theorem thm_sheaf_glue_ontological_driver_reentrykernel_to_r_nucleus_re_entry_nucleus_euler_formula_exactness
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    ReentryKernelWitness (R := R) ↔
      (supported_oscillation (R := R)).enhances = primordialOscillation := by
  exact
    (thm_sheaf_glue_ontological_driver_reentrykernel_to_ontological_driver_eulerphaselaw
      (R := R)).trans
      (thm_sheaf_glue_ontological_driver_eulerphaselaw_to_r_nucleus_re_entry_nucleus_euler_formula_exactness
        (R := R))

/--
Direct correspondence for queue edge:
R-nucleus Euler exactness <-> ontological reentry-kernel driver.
-/
theorem thm_sheaf_glue_r_nucleus_re_entry_nucleus_euler_formula_exactness_to_ontological_driver_reentrykernel
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    (supported_oscillation (R := R)).enhances = primordialOscillation ↔
      ReentryKernelWitness (R := R) := by
  simpa [Iff.comm] using
    (thm_sheaf_glue_ontological_driver_reentrykernel_to_r_nucleus_re_entry_nucleus_euler_formula_exactness
      (R := R))

/--
Direct correspondence for queue edge:
ontological supported-oscillation driver <-> R-nucleus Euler exactness.
-/
theorem thm_sheaf_glue_ontological_driver_supported_supported_oscillation_to_r_nucleus_re_entry_nucleus_euler_formula_exactness
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    SupportedOscillationWitness (R := R) ↔
      (supported_oscillation (R := R)).enhances = primordialOscillation := by
  exact
    (thm_sheaf_glue_ontological_driver_supported_supported_oscillation_to_ontological_driver_eulerphaselaw
      (R := R)).trans
      (thm_sheaf_glue_ontological_driver_eulerphaselaw_to_r_nucleus_re_entry_nucleus_euler_formula_exactness
        (R := R))

/--
Direct correspondence for queue edge:
R-nucleus Euler exactness <-> ontological supported-oscillation driver.
-/
theorem thm_sheaf_glue_r_nucleus_re_entry_nucleus_euler_formula_exactness_to_ontological_driver_supported_supported_oscillation
    {α : Type u} [LoF.PrimaryAlgebra α] (R : LoF.Reentry α) :
    (supported_oscillation (R := R)).enhances = primordialOscillation ↔
      SupportedOscillationWitness (R := R) := by
  simpa [Iff.comm] using
    (thm_sheaf_glue_ontological_driver_supported_supported_oscillation_to_r_nucleus_re_entry_nucleus_euler_formula_exactness
      (R := R))

/-- Strengthened corollary: the re-entry-supported Euler process is both in Euler form
and unit modulus at every phase. -/
theorem reentry_nucleus_euler_formula_and_norm {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) :
    EulerPhaseLaw ((supported_oscillation (R := R)).enhances) ∧
      (∀ θ, ‖(supported_oscillation (R := R)).enhances θ‖ = 1) := by
  constructor
  · intro θ
    simpa [EulerPhaseLaw, supported_oscillation] using primordialOscillation_expansion θ
  · intro θ
    simpa [supported_oscillation] using primordialOscillation_norm_one θ

/-- Fully strengthened Euler profile induced by any re-entry nucleus:
Euler form, unit norm, antiphase, and zero-sum pairing. -/
theorem reentry_nucleus_euler_profile {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) :
    EulerPhaseLaw ((supported_oscillation (R := R)).enhances) ∧
      (∀ θ, ‖(supported_oscillation (R := R)).enhances θ‖ = 1) ∧
      (∀ θ,
        (supported_oscillation (R := R)).enhances (θ + _root_.Real.pi)
          = -(supported_oscillation (R := R)).enhances θ) ∧
      (∀ θ,
        (supported_oscillation (R := R)).enhances θ +
          (supported_oscillation (R := R)).enhances (θ + _root_.Real.pi) = 0) := by
  have hEuler : EulerPhaseLaw ((supported_oscillation (R := R)).enhances) := by
    intro θ
    simpa [supported_oscillation] using primordialOscillation_expansion θ
  refine ⟨hEuler, ?_, ?_, ?_⟩
  · intro θ
    exact EulerPhaseLaw.norm_one hEuler θ
  · intro θ
    exact EulerPhaseLaw.antiphase hEuler θ
  · intro θ
    exact EulerPhaseLaw.pair_cancel hEuler θ

noncomputable def thetaCycle : ℝ → ℂ × ℂ :=
  fun θ => dialecticPair θ

lemma thetaCycle_zero_sum (θ : ℝ) :
    (thetaCycle θ).1 + (thetaCycle θ).2 = 0 :=
  zero_sum θ

lemma thetaCycle_supported_cancel {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) (θ : ℝ) :
    (supported_oscillation (R := R)).enhances θ
        + (supported_oscillation (R := R)).counter θ = 0 :=
  (supported_oscillation (R := R)).cancel θ

end

end Ontology
end HeytingLean
