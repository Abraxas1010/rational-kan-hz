import HeytingLean.Bridge.Veselov.HybridZeckendorf.BasePhiPowers
import HeytingLean.Bridge.Veselov.HybridZeckendorf.Shiftable

open scoped goldenRatio

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-- Exact `Z + Zφ` carrier mirroring the executable Rust bridge surface. -/
structure PhiPair where
  constant : Int
  phi : Int
deriving DecidableEq, Repr

/-- Real-valued semantics for the exact `Z + Zφ` carrier. -/
noncomputable def PhiPair.eval (p : PhiPair) : ℝ :=
  p.constant + p.phi * Real.goldenRatio

/-- Rust-style exact carrier for `φ^i`, encoded as Fibonacci coefficients of `1` and `φ`. -/
def phiPairPow (i : Int) : PhiPair :=
  { constant := signedFib (i - 1)
    phi := signedFib i }

@[simp] theorem phiPairPow_eval (i : Int) :
    (phiPairPow i).eval = Real.goldenRatio ^ i := by
  cases i with
  | ofNat n =>
      cases n with
      | zero =>
          unfold phiPairPow PhiPair.eval
          change
            ((((if 0 % 2 = 0 then (Nat.fib 1 : Int) else -(Nat.fib 1 : Int)) : Int) : ℝ)
              + (((0 : Int) : ℝ) * Real.goldenRatio)) = 1
          norm_num
      | succ k =>
          have h := Real.goldenRatio_mul_fib_succ_add_fib k
          simpa [phiPairPow, PhiPair.eval, signedFib,
            add_comm, add_left_comm, add_assoc,
            mul_comm, mul_left_comm, mul_assoc] using h
  | negSucc n =>
      have hψ :
          (Nat.fib (n + 2) : ℝ) - Real.goldenRatio * Nat.fib (n + 1) =
            Real.goldenConj ^ (n + 1) := by
        simpa [add_comm, add_left_comm, add_assoc] using
          Real.fib_succ_sub_goldenRatio_mul_fib (n + 1)
      have hneg : (-Real.goldenConj) ^ (n + 1) = Real.goldenRatio ^ (Int.negSucc n) := by
        exact neg_goldenConj_pow_eq_zpow_neg (n + 1)
      have hinv : (-Real.goldenConj) ^ (n + 1) = (Real.goldenRatio ^ (n + 1))⁻¹ := by
        calc
          (-Real.goldenConj) ^ (n + 1) = Real.goldenRatio ^ (Int.negSucc n) := hneg
          _ = (Real.goldenRatio ^ (n + 1))⁻¹ := by simp
      rcases Nat.mod_two_eq_zero_or_one n with h | h
      · have hsucc : (n + 1) % 2 = 1 := by omega
        have hodd : Odd (n + 1) := by simpa [Nat.odd_iff] using hsucc
        simp [phiPairPow, PhiPair.eval, signedFib, h, hsucc]
        calc
          -↑(Nat.fib (n + 2)) + ↑(Nat.fib (n + 1)) * Real.goldenRatio
              = -((Nat.fib (n + 2) : ℝ) - Real.goldenRatio * Nat.fib (n + 1)) := by ring
          _ = -(Real.goldenConj ^ (n + 1)) := by rw [hψ]
          _ = (-Real.goldenConj) ^ (n + 1) := by
                simpa using (Odd.neg_pow hodd Real.goldenConj).symm
          _ = (Real.goldenRatio ^ (n + 1))⁻¹ := hinv
      · have hsucc : (n + 1) % 2 = 0 := by omega
        have heven : Even (n + 1) := by simpa [Nat.even_iff] using hsucc
        simp [phiPairPow, PhiPair.eval, signedFib, h, hsucc]
        calc
          ↑(Nat.fib (n + 2)) - ↑(Nat.fib (n + 1)) * Real.goldenRatio
              = (Nat.fib (n + 2) : ℝ) - Real.goldenRatio * Nat.fib (n + 1) := by ring
          _ = Real.goldenConj ^ (n + 1) := hψ
          _ = (-Real.goldenConj) ^ (n + 1) := by
                simpa using (Even.neg_pow heven Real.goldenConj).symm
          _ = (Real.goldenRatio ^ (n + 1))⁻¹ := hinv

end HeytingLean.Bridge.Veselov.HybridZeckendorf
