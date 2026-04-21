import HeytingLean.Bridge.Veselov.HybridZeckendorf.GaloisConj

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-!
Lucas-number bridge.

Lucas numbers are exposed here through the `PhiPair` trace coefficients, not
as a standalone recurrence. The stronger real trace theorem
`L_n = φ^n + ψ^n` is intentionally left out until the negative-index
`phiPairPow` conjugate evaluation lemma is available.
-/

/-- Lucas number as the trace coefficient of the unit `φ^n` in `Z[φ]`. -/
def lucasNum (n : Nat) : Int :=
  let p := phiPairPow (n : Int)
  2 * p.constant + p.phi

/-- The definition is the coefficient trace `p.constant + conj(p).constant`. -/
theorem lucasNum_eq_phiPair_trace_coeff (n : Nat) :
    lucasNum n =
      (phiPairPow (n : Int)).constant +
        (phiPairConj (phiPairPow (n : Int))).constant := by
  simp [lucasNum, phiPairConj]
  ring

@[simp] theorem lucasNum_zero : lucasNum 0 = 2 := by
  native_decide

@[simp] theorem lucasNum_one : lucasNum 1 = 1 := by
  native_decide

end HeytingLean.Bridge.Veselov.HybridZeckendorf
