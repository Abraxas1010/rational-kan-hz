import Mathlib.Data.Real.GoldenRatio

open scoped goldenRatio

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-- Exact negative-power bridge used by the base-`φ` integer-like carrier:
`(-ψ)^k = φ^{-k}` where `ψ` is the golden conjugate. -/
theorem neg_goldenConj_pow_eq_zpow_neg (k : Nat) :
    (-Real.goldenConj) ^ k = Real.goldenRatio ^ (-(k : Int)) := by
  rw [zpow_neg, zpow_natCast]
  have hbase : -Real.goldenConj = Real.goldenRatio⁻¹ := by
    rw [Real.inv_goldenRatio]
  rw [hbase, inv_pow]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
