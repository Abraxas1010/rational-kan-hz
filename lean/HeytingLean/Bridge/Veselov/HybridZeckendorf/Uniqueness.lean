import HeytingLean.Bridge.Veselov.HybridZeckendorf.Addition
import Mathlib.Data.Nat.Fib.Zeckendorf

/-!
# Bridge.Veselov.HybridZeckendorf.Uniqueness

Uniqueness and representation-level commutativity results.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-- Zeckendorf representations are unique at fixed semantic value. -/
theorem zeckendorf_unique (z₁ z₂ : List Nat)
    (h₁ : List.IsZeckendorfRep z₁) (h₂ : List.IsZeckendorfRep z₂)
    (heq : lazyEvalFib z₁ = lazyEvalFib z₂) :
    z₁ = z₂ := by
  have hz₁ : Nat.zeckendorf (lazyEvalFib z₁) = z₁ := by
    simpa [lazyEvalFib] using (Nat.zeckendorf_sum_fib h₁)
  have hz₂ : Nat.zeckendorf (lazyEvalFib z₂) = z₂ := by
    simpa [lazyEvalFib] using (Nat.zeckendorf_sum_fib h₂)
  calc
    z₁ = Nat.zeckendorf (lazyEvalFib z₁) := hz₁.symm
    _ = Nat.zeckendorf (lazyEvalFib z₂) := by rw [heq]
    _ = z₂ := hz₂

/-- Canonical hybrid representations are injective from weighted per-level
equality (the mixed-radix decomposition used by `eval`). -/
theorem canonical_eval_injective (X Y : HybridNumber)
    (hX : IsCanonical X) (hY : IsCanonical Y)
    (hweighted : ∀ i : Nat,
      levelEval (X i) * weight i = levelEval (Y i) * weight i) :
    X = Y := by
  exact Finsupp.ext (fun i =>
    zeckendorf_unique (X i) (Y i) (hX i) (hY i)
      (Nat.eq_of_mul_eq_mul_right (weight_pos i) (by simpa [Nat.mul_comm] using hweighted i)))

private theorem intraNormalize_append_comm (x y : LazyPayload) :
    intraNormalize (x ++ y) = intraNormalize (y ++ x) := by
  unfold intraNormalize intraCore
  apply congrArg Nat.zeckendorf
  calc
    lazyEvalFib (intraIter (2 * (x ++ y).length + 1) (x ++ y))
        = lazyEvalFib (x ++ y) := intraIter_preserves_eval _ _
    _ = lazyEvalFib (y ++ x) := by
      simp [lazyEvalFib_append, Nat.add_comm]
    _ = lazyEvalFib (intraIter (2 * (y ++ x).length + 1) (y ++ x)) := by
      symm
      exact intraIter_preserves_eval _ _

/-- Canonical addition commutes on representations.
This is stronger than value-level commutativity: both sides normalize to the
same per-level canonical payloads before carry transport. -/
theorem add_comm_repr (A B : HybridNumber) :
    add A B = add B A := by
  unfold add normalize
  apply congrArg interCarry
  exact Finsupp.ext (fun i => by
    simpa [normalizeIntra, addLazy, toLazy] using intraNormalize_append_comm (A i) (B i))

end HeytingLean.Bridge.Veselov.HybridZeckendorf
