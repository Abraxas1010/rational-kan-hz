import HeytingLean.Bridge.Veselov.HybridZeckendorf

/-!
# Hybrid Zeckendorf Sanity
-/

namespace HeytingLean.Tests.Bridge.Veselov

open HeytingLean.Bridge.Veselov.HybridZeckendorf

#check weight
#check weight_closed
#check fib_double_identity
#check HybridNumber
#check LazyHybridNumber
#check eval
#check lazyEval
#check normalize
#check add
#check multiplyBinary
#check anchor_invariant_euler_reentry
#check transport_coherence_baseStabilizes_split
#check zeckendorf_unique
#check canonical_eval_injective
#check add_comm_repr
#check supportCard_single_level_bound
#check active_levels_bound
#check density_upper_bound
#check carryAt_idempotent
#check carryAt_monotone
#check normalize_is_closure_operator
#check partRatRaw
#check partRatRaw_eq_telescopic
#check FibPartCarrier
#check FibPartCarrier.eval
#check telescopic_prefix_sum
#check FibSemigroupRaw
#check FibSemigroupRaw.eval
#check FibSemigroupRaw.eval_add
#check FibSemigroup
#check FibSemigroup.eval_addRaw
#check telescopic_sum_identity
#check partRatRaw_adjacent_step_not_next
#check not_fractional_adjacent_carry_rule
#check signedFib
#check shiftEval
#check shiftSingle
#check shiftBy
#check shiftEval_shiftBy
#check basePhiEval
#check basePhiCanonical
#check shiftToBasePhi
#check shiftToBasePhi_semantics
#check neg_goldenConj_pow_eq_zpow_neg
#check PhiPair
#check PhiPair.eval
#check phiPairPow
#check phiPairPow_eval
#check phiPairAdd
#check phiPairAdd_eval
#check phiPairScale
#check phiPairScale_eval
#check phiPairMul
#check phiPairMul_eval
#check PhiPair.eval_injective
#check basePhiPairEval
#check basePhiPairEval_eval
#check basePhiPairEval_rawBasePhiMul
#check rawBasePhiMul
#check basePhiEval_mul
#check basePhiEval_rawBasePhiMul

example : weight 0 = 1 := by decide
example : weight 1 = 1000 := by decide
example : 0 < weight 4 := weight_pos 4

example : eval (fromNat 0) = 0 := eval_fromNat 0
example : eval (fromNat 123456789) = 123456789 := eval_fromNat 123456789

example : eval (add (fromNat 789) (fromNat 456)) = 1245 := by
  norm_num [add_fromNat]

example : eval (multiplyBinary (fromNat 12) (fromNat 34)) = 408 := by
  norm_num [multiplyBinary_correct]

example : eval (normalize (fromNat 99999)) = 99999 := by
  calc
    eval (normalize (fromNat 99999)) = lazyEval (fromNat 99999) := normalize_sound (fromNat 99999)
    _ = 99999 := lazyEval_fromNat 99999

example : FibPartCarrier.eval ⟨[]⟩ = 0 := FibPartCarrier.eval_nil

example : FibPartCarrier.eval ⟨[0]⟩ = partRatRaw 0 := by
  simp [FibPartCarrier.eval_cons]

example (a b : FibSemigroupRaw) :
    FibSemigroupRaw.eval (a + b) = FibSemigroupRaw.eval a + FibSemigroupRaw.eval b :=
  FibSemigroupRaw.eval_add a b

example : FibSemigroup.empty.eval = 0 := FibSemigroup.eval_empty

example (a b : FibSemigroup) :
    FibSemigroupRaw.eval (a.addRaw b) = a.eval + b.eval :=
  FibSemigroup.eval_addRaw a b

example : partRatRaw 1 + partRatRaw 2 ≠ partRatRaw 3 :=
  partRatRaw_adjacent_step_not_next

example : ¬ ∀ n : Nat, partRatRaw n + partRatRaw (n + 1) = partRatRaw (n + 2) :=
  not_fractional_adjacent_carry_rule

example : shiftEval (shiftSingle 3 2) = 2 * signedFib 3 := by
  simp

example : shiftBy 2 (shiftSingle 3 5) = shiftSingle 5 5 := by
  simp

example : basePhiCanonical (0 : BasePhiDigits) := by
  intro i
  simp

example : basePhiEval (shiftToBasePhi (shiftSingle 0 1)) = 1 := by
  rw [shiftToBasePhi_semantics]
  simpa [shiftSingle] using
    (Finsupp.sum_single_index
      (a := (0 : Int))
      (b := (1 : Int))
      (h := fun i coeff => (coeff : Real) * (Real.goldenRatio ^ i))
      (by simp : (((0 : Int) : Real) * (Real.goldenRatio ^ (0 : Int))) = 0))

example (k : Nat) :
    (-Real.goldenConj) ^ k = Real.goldenRatio ^ (-(k : Int)) :=
  neg_goldenConj_pow_eq_zpow_neg k

example (i : Int) : (phiPairPow i).eval = Real.goldenRatio ^ i :=
  phiPairPow_eval i

example (d : BasePhiDigits) : (basePhiPairEval d).eval = basePhiEval d :=
  basePhiPairEval_eval d

example (a b : PhiPair) : (phiPairMul a b).eval = a.eval * b.eval :=
  phiPairMul_eval a b

example : Function.Injective PhiPair.eval :=
  PhiPair.eval_injective

example (a b : BasePhiDigits) :
    basePhiPairEval (rawBasePhiMul a b) =
      phiPairMul (basePhiPairEval a) (basePhiPairEval b) :=
  basePhiPairEval_rawBasePhiMul a b

end HeytingLean.Tests.Bridge.Veselov
