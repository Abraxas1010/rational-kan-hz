import HeytingLean.Bridge.Veselov.HybridZeckendorf.ConvolutionMul
import HeytingLean.Bridge.Veselov.HybridZeckendorf.DensityBounds

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-!
# Standard-ring Zeckendorf NTT bridge

This file does not claim a new lattice-HE parameter family. It packages the
existing Laurent-product semantics together with the active-pair telemetry used
by the Rust standard-ring support-pruned lane.
-/

structure StandardRingZeckIntermediate where
  product : BasePhiDigits
  activePairCount : Nat

noncomputable def standardRingZeckIntermediate (a b : BasePhiDigits) : StandardRingZeckIntermediate :=
  { product := rawBasePhiMul a b
    activePairCount := a.support.card * b.support.card }

@[simp] theorem standardRingZeckIntermediate_product (a b : BasePhiDigits) :
    (standardRingZeckIntermediate a b).product = rawBasePhiMul a b := by
  rfl

@[simp] theorem standardRingZeckIntermediate_activePairCount (a b : BasePhiDigits) :
    (standardRingZeckIntermediate a b).activePairCount = a.support.card * b.support.card := by
  rfl

theorem standardRingZeckIntermediate_sound (a b : BasePhiDigits) :
    basePhiEval (standardRingZeckIntermediate a b).product = basePhiEval a * basePhiEval b := by
  simpa [standardRingZeckIntermediate] using basePhiEval_rawBasePhiMul a b

theorem standardRingZeckIntermediate_phiPair (a b : BasePhiDigits) :
    basePhiPairEval (standardRingZeckIntermediate a b).product =
      phiPairMul (basePhiPairEval a) (basePhiPairEval b) := by
  simpa [standardRingZeckIntermediate] using basePhiPairEval_rawBasePhiMul a b

theorem standardRingZeckIntermediate_activePairCount_comm (a b : BasePhiDigits) :
    (standardRingZeckIntermediate a b).activePairCount =
      (standardRingZeckIntermediate b a).activePairCount := by
  simp [standardRingZeckIntermediate, Nat.mul_comm]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
