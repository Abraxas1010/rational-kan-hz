import HeytingLean.Bridge.Veselov.HybridZeckendorf.Normalization
import HeytingLean.Bridge.Veselov.HybridZeckendorf.Uniqueness
import HeytingLean.Core.Nucleus

/-!
# Bridge.Veselov.HybridZeckendorf.NucleusBridge

Bridge lemmas from HybridZeckendorf normalization/carry to closure-style
(`Nucleus`) semantics.

`HybridNumber = Nat →₀ ZeckPayload` does not currently carry the lattice
instances (`SemilatticeInf`, `OrderBot`) required to instantiate
`HeytingLean.Core.Nucleus` directly on hybrid states. This module therefore
proves closure-style properties directly at representation and semantic level.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Core

private theorem sum_over_union_eq_eval (Y : HybridNumber) (S : Finset Nat) (hsub : Y.support ⊆ S) :
    (∑ j ∈ S, levelEval (Y j) * weight j) = eval Y := by
  let f : Nat → Nat := fun j => levelEval (Y j) * weight j
  have hsum : (∑ j ∈ Y.support, f j) = (∑ j ∈ S, f j) := by
    exact Finset.sum_subset (s₁ := Y.support) (s₂ := S) (h := hsub) (f := f) (by
      intro j _hjS hjY
      unfold f
      have hj0 : Y j = 0 := Finsupp.notMem_support_iff.mp hjY
      rw [hj0]
      change (List.map Nat.fib ([] : List Nat)).sum * weight j = 0
      simp)
  calc
    (∑ j ∈ S, levelEval (Y j) * weight j)
        = (∑ j ∈ S, f j) := by rfl
    _ = (∑ j ∈ Y.support, f j) := hsum.symm
    _ = eval Y := by
      unfold f
      simp [eval, Finsupp.sum]

private theorem eval_monotone_of_levelEval_monotone (X Y : HybridNumber)
    (hXY : ∀ j, levelEval (X j) ≤ levelEval (Y j)) :
    eval X ≤ eval Y := by
  classical
  let S : Finset Nat := X.support ∪ Y.support
  have hXsub : X.support ⊆ S := Finset.subset_union_left
  have hYsub : Y.support ⊆ S := Finset.subset_union_right
  have hXeq : eval X = ∑ j ∈ S, levelEval (X j) * weight j := by
    simpa [S] using (sum_over_union_eq_eval X S hXsub).symm
  have hYeq : eval Y = ∑ j ∈ S, levelEval (Y j) * weight j := by
    simpa [S] using (sum_over_union_eq_eval Y S hYsub).symm
  rw [hXeq, hYeq]
  exact Finset.sum_le_sum (fun j _hj => Nat.mul_le_mul_right (weight j) (hXY j))

/-- Carry at a fixed level is representationally idempotent on canonical states. -/
theorem carryAt_idempotent (i : Nat) (X : HybridNumber) (hX : IsCanonical X) :
    carryAt i (carryAt i X) = carryAt i X := by
  have hcanon1 : IsCanonical (carryAt i X) := carryAt_preserves_canonical i X hX
  have hcanon2 : IsCanonical (carryAt i (carryAt i X)) :=
    carryAt_preserves_canonical i (carryAt i X) hcanon1
  have hweighted :
      ∀ j : Nat,
        levelEval ((carryAt i (carryAt i X)) j) * weight j =
          levelEval ((carryAt i X) j) * weight j := by
    intro j
    cases i with
    | zero =>
        simp [carryAt]
    | succ k =>
        by_cases hj1 : j = k + 1
        · subst hj1
          simp [carryAt, levelEval, lazyEvalFib, Nat.sum_zeckendorf_fib]
        · by_cases hj2 : j = k + 2
          · subst hj2
            simp [carryAt, levelEval, lazyEvalFib, Nat.sum_zeckendorf_fib]
          · simp [carryAt, hj1, hj2]
  exact canonical_eval_injective (carryAt i (carryAt i X)) (carryAt i X) hcanon2 hcanon1 hweighted

/-- Semantic monotonicity of carry under pointwise per-level order assumptions.
This is the strongest monotonicity available on the current representation. -/
theorem carryAt_eval_monotone (i : Nat) :
    ∀ X Y : HybridNumber, (∀ j, levelEval (X j) ≤ levelEval (Y j)) →
      eval (carryAt i X) ≤ eval (carryAt i Y) := by
  intro X Y hXY
  have hXYeval : eval X ≤ eval Y := eval_monotone_of_levelEval_monotone X Y hXY
  simpa [carryAt_preserves_eval] using hXYeval

@[deprecated carryAt_eval_monotone (since := "2026-03-06")]
theorem carryAt_monotone (i : Nat) :
    ∀ X Y : HybridNumber, (∀ j, levelEval (X j) ≤ levelEval (Y j)) →
      eval (carryAt i X) ≤ eval (carryAt i Y) :=
  carryAt_eval_monotone i

/-- Pointwise payload-order monotonicity fails for `carryAt` (remainder folding
at the carried level can decrease one coordinate while increasing the next). -/
theorem carryAt_not_pointwise_monotone :
    ∃ i : Nat, ∃ X Y : HybridNumber,
      (∀ j, levelEval (X j) ≤ levelEval (Y j)) ∧
      ¬ (∀ j, levelEval ((carryAt i X) j) ≤ levelEval ((carryAt i Y) j)) := by
  let X : HybridNumber := Finsupp.single 1 (Nat.zeckendorf 999)
  let Y : HybridNumber := Finsupp.single 1 (Nat.zeckendorf 1000)
  have hX1 : levelEval (X 1) = 999 := by
    simpa [X, levelEval, lazyEvalFib] using (Nat.sum_zeckendorf_fib 999)
  have hY1 : levelEval (Y 1) = 1000 := by
    simpa [Y, levelEval, lazyEvalFib] using (Nat.sum_zeckendorf_fib 1000)
  refine ⟨1, X, Y, ?_, ?_⟩
  · intro j
    by_cases hj : j = 1
    · subst hj
      omega
    · simp [X, Y, hj, levelEval, lazyEvalFib]
  · intro hmono
    have hdivX : levelEval (X 1) / weight 1 = 0 := by
      simp [hX1, weight]
    have hmodX : levelEval (X 1) % weight 1 = 999 := by
      simp [hX1, weight]
    have hcarryXMod : (carryAt 1 X) 1 = Nat.zeckendorf (levelEval (X 1) % weight 1) := by
      simp [X, carryAt]
    have hleft : levelEval ((carryAt 1 X) 1) = 999 := by
      rw [hcarryXMod, hmodX]
      change lazyEvalFib (Nat.zeckendorf 999) = 999
      simpa [lazyEvalFib] using (Nat.sum_zeckendorf_fib 999)
    have hdivY : levelEval (Y 1) / weight 1 = 1 := by
      simp [hY1, weight]
    have hmodY : levelEval (Y 1) % weight 1 = 0 := by
      simp [hY1, weight]
    have hcarryYMod : (carryAt 1 Y) 1 = Nat.zeckendorf (levelEval (Y 1) % weight 1) := by
      simp [Y, carryAt]
    have hright : levelEval ((carryAt 1 Y) 1) = 0 := by
      rw [hcarryYMod, hmodY]
      change lazyEvalFib (Nat.zeckendorf 0) = 0
      simp [lazyEvalFib]
    have h1 : levelEval ((carryAt 1 X) 1) ≤ levelEval ((carryAt 1 Y) 1) := hmono 1
    omega

private theorem normalizeIntra_of_canonical (Y : HybridNumber) (hY : IsCanonical Y) :
    normalizeIntra (toLazy Y) = Y := by
  apply Finsupp.ext
  intro i
  change intraNormalize (toLazy Y i) = Y i
  have hcanon : List.IsZeckendorfRep (Y i) := hY i
  have hsound : levelEval (intraNormalize (toLazy Y i)) = levelEval (Y i) := by
    simpa [toLazy, levelEval] using intraNormalize_sound (toLazy Y i)
  exact zeckendorf_unique (intraNormalize (toLazy Y i)) (Y i)
    (intraNormalize_canonical (toLazy Y i)) hcanon hsound

/-- Representation-level closure law for normalization. -/
theorem normalize_is_closure_operator (X : LazyHybridNumber) :
    normalize (toLazy (normalize X)) = normalize X := by
  unfold normalize
  have hcanon0 : IsCanonical (normalizeIntra X) := normalizeIntra_canonical X
  have hcanon : IsCanonical (interCarry (normalizeIntra X)) :=
    interCarry_preserves_canonical (normalizeIntra X) hcanon0
  have hintra :
      normalizeIntra (toLazy (interCarry (normalizeIntra X))) = interCarry (normalizeIntra X) :=
    normalizeIntra_of_canonical (interCarry (normalizeIntra X)) hcanon
  calc
    interCarry (normalizeIntra (toLazy (interCarry (normalizeIntra X))))
        = interCarry (interCarry (normalizeIntra X)) := by rw [hintra]
    _ = interCarry (normalizeIntra X) := interCarry_idempotent_of_canonical (normalizeIntra X) hcanon0

end HeytingLean.Bridge.Veselov.HybridZeckendorf
