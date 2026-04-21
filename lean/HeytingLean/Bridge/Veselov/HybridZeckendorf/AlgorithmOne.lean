import HeytingLean.Bridge.Veselov.HybridZeckendorf.LinearTimeAdd
import HeytingLean.Bridge.Veselov.HybridZeckendorf.ConvolutionMul

/-!
# Veselov FNS: Algorithm 1 composition

The current theorem surface composes the existing Zeckendorf-to-φ conversion,
Laurent convolution, Z[φ] normalization, and φ-to-Zeckendorf extraction. Because
`zeckToBasePhi` is explicitly documented as a structural Z[φ] transform rather
than a natural-number value embedding, correctness is stated in the exposed
Z[φ]/reverse-conversion semantics rather than as an unsupported `Nat` product.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open LaurentPolynomial

/-- Algorithm 1 digit pipeline:
Zeckendorf payloads -> base-φ digits -> convolution -> normalization -> Zeckendorf. -/
noncomputable def algorithmOne (a b : ZeckPayload) : ZeckPayload :=
  basePhiToZeck (multiplyConvolutionDigits (zeckToBasePhi a) (zeckToBasePhi b))

/-- The convolution stage has the exact Z[φ] product semantics already proved. -/
theorem algorithmOne_phiPair (a b : ZeckPayload) :
    basePhiPairEval (multiplyConvolutionDigits (zeckToBasePhi a) (zeckToBasePhi b)) =
      phiPairMul (basePhiPairEval (zeckToBasePhi a)) (basePhiPairEval (zeckToBasePhi b)) := by
  exact multiplyConvolutionDigits_phiPair (zeckToBasePhi a) (zeckToBasePhi b)

/-- If the normalized product is a non-negative integer in Z[φ], the reverse
conversion evaluates to that integer. -/
theorem algorithmOne_correct_when_integer (a b : ZeckPayload)
    (h_int :
      (basePhiPairEval (multiplyConvolutionDigits (zeckToBasePhi a) (zeckToBasePhi b))).phi = 0)
    (h_pos :
      0 ≤ (basePhiPairEval (multiplyConvolutionDigits (zeckToBasePhi a) (zeckToBasePhi b))).constant) :
    levelEval (algorithmOne a b) =
      (basePhiPairEval (multiplyConvolutionDigits (zeckToBasePhi a) (zeckToBasePhi b))).constant.toNat := by
  simpa [algorithmOne] using
    basePhiToZeck_eval
      (multiplyConvolutionDigits (zeckToBasePhi a) (zeckToBasePhi b)) h_int h_pos

/-- Under the same integer-side condition, Algorithm 1 returns canonical
Zeckendorf digits. -/
theorem algorithmOne_canonical_when_integer (a b : ZeckPayload)
    (h_int :
      (basePhiPairEval (multiplyConvolutionDigits (zeckToBasePhi a) (zeckToBasePhi b))).phi = 0)
    (h_pos :
      0 ≤ (basePhiPairEval (multiplyConvolutionDigits (zeckToBasePhi a) (zeckToBasePhi b))).constant) :
    List.IsZeckendorfRep (algorithmOne a b) := by
  simpa [algorithmOne] using
    basePhiToZeck_canonical
      (multiplyConvolutionDigits (zeckToBasePhi a) (zeckToBasePhi b)) h_int h_pos

/-- The tensor compile plan for the convolution leg has output order two. -/
theorem algorithmOne_convolutionPlan_outputOrder (n : Nat) :
    (zeckConvolutionPlan n).outputOrder = 2 := by
  exact zeckConvolutionPlan_outputOrder n

end HeytingLean.Bridge.Veselov.HybridZeckendorf
