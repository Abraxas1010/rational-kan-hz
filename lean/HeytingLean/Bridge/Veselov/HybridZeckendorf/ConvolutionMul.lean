import HeytingLean.Bridge.Veselov.HybridZeckendorf.ArbitraryNormalize
import HeytingLean.Bridge.Veselov.HybridZeckendorf.QMatrixBridge
import HeytingLean.Boundary.Hypergraph.TensorCompile

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Boundary.Hypergraph

/-!
# Base-φ convolution multiplication

This file verifies the Laurent-polynomial multiplication lane composed with the
Z[φ] normalizer, and records the tensor compile shape.

The `multiplyConvolutionDigits` function computes the Z[φ] product of two
base-φ digit sequences: convolve via Laurent multiplication, then reduce to
the canonical Z[φ] form `a + bφ` via `normalizeArbitrary`.
-/

/-- Digit-level convolution through Laurent-polynomial multiplication,
    followed by Z[φ] normalization. -/
noncomputable def multiplyConvolutionDigits (a b : BasePhiDigits) : BasePhiDigits :=
  normalizeArbitrary (rawBasePhiMul a b)

/-- The convolution boundary preserves base-φ semantics. -/
theorem multiplyConvolutionDigits_sound (a b : BasePhiDigits) :
    basePhiEval (multiplyConvolutionDigits a b) = basePhiEval a * basePhiEval b := by
  unfold multiplyConvolutionDigits
  rw [normalizeArbitrary_sound]
  exact basePhiEval_rawBasePhiMul a b

/-- The convolution boundary preserves Z[φ] ring semantics. -/
theorem multiplyConvolutionDigits_phiPair (a b : BasePhiDigits) :
    basePhiPairEval (multiplyConvolutionDigits a b) =
      phiPairMul (basePhiPairEval a) (basePhiPairEval b) := by
  apply PhiPair.eval_injective
  rw [basePhiPairEval_eval, multiplyConvolutionDigits_sound,
      phiPairMul_eval, basePhiPairEval_eval, basePhiPairEval_eval]

/-- The Zeckendorf convolution compile shape. This is a plan, not an FFT theorem. -/
def zeckConvolutionPlan (n : Nat) : TensorCompilePlan :=
  compileTensorOp (.matmul n n (2 * n - 1))

@[simp] theorem zeckConvolutionPlan_outputOrder (n : Nat) :
    (zeckConvolutionPlan n).outputOrder = 2 := by
  rfl

end HeytingLean.Bridge.Veselov.HybridZeckendorf
