import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFrac
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibSemigroup
import Mathlib.Data.Nat.Fib.Zeckendorf

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-!
# Greedy normalization for finite Fibonacci-fraction prefixes

The fuel-bounded greedy algorithm is exact when the remainder is retained. This
avoids the false claim that finite fuel always consumes an arbitrary rational
fractional value completely.
-/

/-- Fractional Zeckendorf payload candidate: a list of fraction indices. -/
abbrev FracZeckPayload := List Nat

/-- Exact fractional evaluation. -/
def fracEval (fz : FracZeckPayload) : Rat :=
  (fz.map fibFrac).sum

@[simp] theorem partRatRaw_eq_fibFrac (n : Nat) :
    partRatRaw n = fibFrac n := by
  rw [partRatRaw_eq_telescopic, fibFrac_telescoping]

@[simp] theorem map_partRatRaw_sum_eq_map_fibFrac_sum (xs : List Nat) :
    (xs.map partRatRaw).sum = (xs.map fibFrac).sum := by
  induction xs with
  | nil =>
      simp
  | cons x xs ih =>
      simp only [List.map_cons, List.sum_cons]
      rw [partRatRaw_eq_fibFrac, ih]

/-- Greedy selection as a Boolean, matching the paper's `δ_n` branch. -/
def greedySelect (x : ℚ) (n : Nat) : Bool :=
  decide (fibFrac n ≤ x)

/-- One step of the greedy algorithm: select or skip the current fraction. -/
def greedyStep (x : ℚ) (n : Nat) : ℚ × Bool :=
  if fibFrac n ≤ x then (x - fibFrac n, true) else (x, false)

/-- Fuel-bounded greedy normalization, retaining the final remainder. -/
def greedyFracNorm (x : ℚ) (startIdx fuel : Nat) : List Nat × ℚ :=
  match fuel with
  | 0 => ([], x)
  | fuel + 1 =>
      let step := greedyStep x startIdx
      let rest := greedyFracNorm step.1 (startIdx + 1) fuel
      if step.2 then (startIdx :: rest.1, rest.2) else rest

/-- The greedy algorithm preserves value when the final remainder is retained. -/
theorem greedyFracNorm_sound (x : ℚ) (startIdx fuel : Nat) :
    let out := greedyFracNorm x startIdx fuel
    (out.1.map fibFrac).sum + out.2 = x := by
  induction fuel generalizing x startIdx with
  | zero =>
      simp [greedyFracNorm]
  | succ fuel ih =>
      by_cases h : fibFrac startIdx ≤ x
      · simp [greedyFracNorm, greedyStep, h]
        have hih := ih (x - fibFrac startIdx) (startIdx + 1)
        linarith
      · simp [greedyFracNorm, greedyStep, h, ih]

/-- The greedy remainder stays non-negative for non-negative input. -/
theorem greedyFracNorm_remainder_nonneg (x : ℚ) (hx : 0 ≤ x)
    (startIdx fuel : Nat) :
    0 ≤ (greedyFracNorm x startIdx fuel).2 := by
  induction fuel generalizing x startIdx with
  | zero =>
      simpa [greedyFracNorm] using hx
  | succ fuel ih =>
      by_cases h : fibFrac startIdx ≤ x
      · have hx' : 0 ≤ x - fibFrac startIdx := by linarith
        simpa [greedyFracNorm, greedyStep, h] using ih (x - fibFrac startIdx) hx' (startIdx + 1)
      · simpa [greedyFracNorm, greedyStep, h] using ih x hx (startIdx + 1)

/-- The PM-sketched adjacent rule is not valid for the stated `fibFrac`. -/
theorem not_forall_fibFrac_adjacent_sum :
    ¬ ∀ n : Nat, 1 ≤ n → fibFrac n + fibFrac (n + 1) = fibFrac (n + 2) := by
  intro h
  exact fibFrac_adjacent_sum_counterexample (h 1 (by norm_num))

/-- Fuel-bounded normalization output with the explicit residual remainder. -/
noncomputable def normalizeFibSemigroupRawWithRemainder
    (s : FibSemigroupRaw) (fracFuel : Nat := 64) : FibSemigroupRaw × ℚ :=
  let intNorm := Nat.zeckendorf (lazyEvalFib s.fibs)
  let greedy := greedyFracNorm s.evalParts 1 fracFuel
  ({ fibs := intNorm, parts := greedy.1 }, greedy.2)

/-- Re-encoding the integer payload by `Nat.zeckendorf` preserves its value. -/
theorem lazyEvalFib_zeckendorf (n : Nat) :
    lazyEvalFib (Nat.zeckendorf n) = n := by
  simp [lazyEvalFib, Nat.sum_zeckendorf_fib n]

/-- The greedy normalizer is value-preserving when its residual remainder is kept. -/
theorem normalizeFibSemigroupRawWithRemainder_sound
    (s : FibSemigroupRaw) (fuel : Nat) :
    let out := normalizeFibSemigroupRawWithRemainder s fuel
    FibSemigroupRaw.eval out.1 + out.2 = FibSemigroupRaw.eval s := by
  dsimp [normalizeFibSemigroupRawWithRemainder]
  have hgreedy := greedyFracNorm_sound (FibSemigroupRaw.evalParts s) 1 fuel
  have hnew :=
    map_partRatRaw_sum_eq_map_fibFrac_sum
      (greedyFracNorm (FibSemigroupRaw.evalParts s) 1 fuel).1
  simp [FibSemigroupRaw.eval, FibSemigroupRaw.evalFibs, FibSemigroupRaw.evalParts,
    FibPartCarrier.eval, lazyEvalFib_zeckendorf] at hgreedy hnew ⊢
  linarith

end HeytingLean.Bridge.Veselov.HybridZeckendorf
