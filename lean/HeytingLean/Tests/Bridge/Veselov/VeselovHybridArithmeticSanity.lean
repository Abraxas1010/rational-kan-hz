import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFracIrreducible
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFracConvergence
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFracDensity
import HeytingLean.Bridge.Veselov.HybridZeckendorf.ApproxMul
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFracNormalize
import HeytingLean.Boundary.Homomorphic.ZeckNormFHEFrac
import HeytingLean.Boundary.Hypergraph.ZeckRewriteNetFrac

namespace HeytingLean.Tests.Bridge.Veselov.VeselovHybridArithmeticSanity

open HeytingLean.Bridge.Veselov.HybridZeckendorf
open HeytingLean.Boundary.Homomorphic
open HeytingLean.Boundary.Hypergraph
open scoped goldenRatio

example : Nat.Coprime (Nat.fib 1) (Nat.fib 2 * Nat.fib 3) := by native_decide
example : Nat.Coprime (Nat.fib 4) (Nat.fib 5 * Nat.fib 6) := by native_decide
example : Nat.Coprime (Nat.fib 9) (Nat.fib 10 * Nat.fib 11) := by native_decide

example : fibFrac 1 = 1/2 := by norm_num [fibFrac, Nat.fib]

example : greedySelect (1/2 : ℚ) 1 = true := by
  norm_num [greedySelect, fibFrac, Nat.fib]

example : fibFrac 1 + fibFrac 2 ≠ fibFrac 3 :=
  fibFrac_adjacent_sum_counterexample

example : ¬ ∀ n : Nat, partRatRaw n + partRatRaw (n + 1) = partRatRaw (n + 2) :=
  not_fractional_adjacent_carry_rule

example (n : Nat) :
    (Nat.fib n : ℝ) =
      (Real.goldenRatio ^ n - Real.goldenConj ^ n) / Real.sqrt 5 :=
  binet_nat n

example (bits : List (HomBoolBackend.plain).EncBool) :
    (homFracBitmap (B := HomBoolBackend.plain) bits).map (HomBoolBackend.plain).dec =
      plainFracSelect (bits.map (HomBoolBackend.plain).dec) 0 :=
  homFracBitmap_decrypt bits

example (maxInt maxFrac : Nat) :
    ClosedUnderIncidentWires (zeckCombinedNet maxInt maxFrac) Finset.univ :=
  zeckCombinedNet_univ_closed maxInt maxFrac

end HeytingLean.Tests.Bridge.Veselov.VeselovHybridArithmeticSanity
