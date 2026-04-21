import HeytingLean.Bridge.Veselov.HybridZeckendorf.ErrorDetection
import HeytingLean.Bridge.Veselov.HybridZeckendorf.Uniqueness
import Mathlib.Data.List.Range

/-!
# Bridge.Veselov.HybridZeckendorf.StructuralECC

A narrow structural-ECC surface for Fibonacci / Zeckendorf-coded payloads.
This file does not claim universal single-bit detection. It formalizes the
local detector that already exists in the bridge and exposes a reusable set of
"detectable sites" where a zero-to-one flip is guaranteed to create an
adjacent-one violation.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf.StructuralECC

open HeytingLean.Bridge.Veselov.HybridZeckendorf

/-- Sites whose neighborhood already contains a one, so a zero-to-one flip will
enter the local detector surface. -/
def detectableFlipSites (bs : List Bool) : List Nat :=
  (List.range bs.length).filter (fun pos => adjacentToOneBool bs pos)

@[simp] theorem mem_detectableFlipSites_iff (bs : List Bool) (pos : Nat) :
    pos ∈ detectableFlipSites bs ↔ pos < bs.length ∧ adjacentToOne bs pos := by
  unfold detectableFlipSites
  simp [adjacentToOneBool_eq_true]

/-- In the canonical three-bit local window, flipping the centre zero is
structurally detected exactly when one of its neighbors is one. -/
theorem single_bit_flip_detected_local (left right : Bool) :
    hasConsecutiveOnes (singleBitFlip [left, false, right] 1) = true ↔
      left = true ∨ right = true := by
  simpa [adjacentToOne, or_comm] using
    (consecutive_detects_some_single_flips_local left false right rfl)

/-- Canonical payloads remain unique at fixed semantic value. Structural ECC is
layered on top of a unique canonical representation, not an ambiguous carrier. -/
theorem zeckendorf_payload_unique (z₁ z₂ : List Nat)
    (h₁ : List.IsZeckendorfRep z₁) (h₂ : List.IsZeckendorfRep z₂)
    (heq : lazyEvalFib z₁ = lazyEvalFib z₂) :
    z₁ = z₂ :=
  zeckendorf_unique z₁ z₂ h₁ h₂ heq

end HeytingLean.Bridge.Veselov.HybridZeckendorf.StructuralECC
