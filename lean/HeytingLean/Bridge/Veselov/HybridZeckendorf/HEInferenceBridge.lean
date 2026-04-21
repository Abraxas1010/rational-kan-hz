import HeytingLean.Bridge.Veselov.HybridZeckendorf.DensityBounds
import Mathlib.Tactic

/-!
# Bridge.Veselov.HybridZeckendorf.HEInferenceBridge

An abstract bridge from the existing active-level bound to a ciphertext-plaintext
multiplication count / depth budget for Zeckendorf-coded HE inference.

This file is intentionally narrow. It does not depend on any external HE backend
and does not claim SEAL/OpenFHE integration. It packages the theorem-side fact
that active Zeckendorf levels bound an abstract linear-layer work budget.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf.HEInferenceBridge

open HeytingLean.Bridge.Veselov.HybridZeckendorf

/-- Abstract number of plaintext-side active multiplies induced by a hybrid
Zeckendorf weight carrier. -/
def hePlainMulCount (X : HybridNumber) : Nat :=
  X.support.card

/-- A one-parameter abstract depth model: each active level costs a fixed number
of ct-pt depth units. -/
def heCircuitDepth (depthPerActiveLevel : Nat) (X : HybridNumber) : Nat :=
  hePlainMulCount X * depthPerActiveLevel

/-- Direct bridge: the existing active-level theorem already bounds the abstract
ct-pt multiply count. -/
theorem hePlainMulCount_bound (X : HybridNumber) (hX : IsCanonical X) (hpos : 0 < eval X) :
    (hePlainMulCount X : ℝ) ≤ Real.logb 1000 (eval X) + 2 :=
  active_levels_bound X hX hpos

/-- The abstract HE depth budget inherits the same logarithmic bound up to the
chosen per-level cost multiplier. -/
theorem heCircuitDepth_bound (depthPerActiveLevel : Nat)
    (X : HybridNumber) (hX : IsCanonical X) (hpos : 0 < eval X) :
    (heCircuitDepth depthPerActiveLevel X : ℝ) ≤
      (depthPerActiveLevel : ℝ) * (Real.logb 1000 (eval X) + 2) := by
  calc
    (heCircuitDepth depthPerActiveLevel X : ℝ)
        = (depthPerActiveLevel : ℝ) * (hePlainMulCount X : ℝ) := by
            simp [heCircuitDepth, hePlainMulCount, Nat.cast_mul, mul_comm]
    _ ≤ (depthPerActiveLevel : ℝ) * (Real.logb 1000 (eval X) + 2) := by
          gcongr
          exact hePlainMulCount_bound X hX hpos

end HeytingLean.Bridge.Veselov.HybridZeckendorf.HEInferenceBridge
