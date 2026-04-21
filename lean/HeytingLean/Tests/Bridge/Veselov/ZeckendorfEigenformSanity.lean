import HeytingLean.Bridge.Veselov.HybridZeckendorf.QMatrixBridge
import HeytingLean.Bridge.Veselov.HybridZeckendorf.GaloisConj
import HeytingLean.Bridge.Veselov.HybridZeckendorf.Lucas
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFrac

namespace HeytingLean.Tests.Bridge.Veselov.ZeckendorfEigenformSanity

open HeytingLean.Bridge.Veselov.HybridZeckendorf
open HeytingLean.Boundary.Hypergraph

example :
    fibQMatrix (idx2 (1 : Fin 2) (1 : Fin 2)) = 0 := by
  simp

example :
    phiPairConj { constant := 3, phi := 5 } = { constant := 8, phi := -5 } := by
  rfl

example :
    phiPairNorm { constant := 0, phi := 1 } = -1 := by
  norm_num [phiPairNorm]

example :
    lucasNum 0 = 2 ∧ lucasNum 1 = 1 := by
  simp

example :
    fibFrac 1 + fibFrac 2 ≠ fibFrac 3 :=
  fibFrac_adjacent_sum_counterexample

end HeytingLean.Tests.Bridge.Veselov.ZeckendorfEigenformSanity
