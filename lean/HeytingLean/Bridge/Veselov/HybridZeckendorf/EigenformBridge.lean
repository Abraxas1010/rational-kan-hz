import HeytingLean.Bridge.Veselov.HybridZeckendorf.QMatrixBridge
import HeytingLean.Bridge.Veselov.HybridZeckendorf.GaloisConj
import HeytingLean.Boundary.Hypergraph.ZeckNucleus

open scoped goldenRatio

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Boundary.Hypergraph

/-!
Eigenform bridge for the verified Q-matrix facts.
-/

/-- The golden ratio is a fixed point of `x ↦ 1 + 1/x`. -/
theorem goldenRatio_eigenform :
    Real.goldenRatio = 1 + 1 / Real.goldenRatio := by
  rw [show (1 : Real) / Real.goldenRatio = Real.goldenRatio⁻¹ by rw [one_div]]
  rw [Real.inv_goldenRatio]
  linarith [Real.goldenRatio_add_goldenConj]

/-- Row 0 of `Q * [φ, 1]^T = φ * [φ, 1]^T`. -/
theorem phi_is_eigenvalue_of_Q_row0 :
    (fibQMatrix (idx2 (0 : Fin 2) (0 : Fin 2)) : Real) * Real.goldenRatio +
      (fibQMatrix (idx2 (0 : Fin 2) (1 : Fin 2)) : Real) * 1 =
        Real.goldenRatio * Real.goldenRatio := by
  norm_num [fibQMatrix, idx2]
  have hsq : Real.goldenRatio * Real.goldenRatio = Real.goldenRatio + 1 := by
    rw [← pow_two]
    exact Real.goldenRatio_sq
  exact hsq.symm

/-- Row 1 of `Q * [φ, 1]^T = φ * [φ, 1]^T`. -/
theorem phi_is_eigenvalue_of_Q_row1 :
    (fibQMatrix (idx2 (1 : Fin 2) (0 : Fin 2)) : Real) * Real.goldenRatio +
      (fibQMatrix (idx2 (1 : Fin 2) (1 : Fin 2)) : Real) * 1 =
        Real.goldenRatio * 1 := by
  norm_num [fibQMatrix, idx2]

/-- Row 0 of `Q * [ψ, 1]^T = ψ * [ψ, 1]^T`. -/
theorem phiConj_is_eigenvalue_of_Q_row0 :
    (fibQMatrix (idx2 (0 : Fin 2) (0 : Fin 2)) : Real) * Real.goldenConj +
      (fibQMatrix (idx2 (0 : Fin 2) (1 : Fin 2)) : Real) * 1 =
        Real.goldenConj * Real.goldenConj := by
  norm_num [fibQMatrix, idx2]
  have hsq : Real.goldenConj * Real.goldenConj = Real.goldenConj + 1 := by
    rw [← pow_two]
    exact Real.goldenConj_sq
  exact hsq.symm

/-- Row 1 of `Q * [ψ, 1]^T = ψ * [ψ, 1]^T`. -/
theorem phiConj_is_eigenvalue_of_Q_row1 :
    (fibQMatrix (idx2 (1 : Fin 2) (0 : Fin 2)) : Real) * Real.goldenConj +
      (fibQMatrix (idx2 (1 : Fin 2) (1 : Fin 2)) : Real) * 1 =
        Real.goldenConj * 1 := by
  norm_num [fibQMatrix, idx2]

/-- Galois conjugation is the coefficient-level spectral flip `φ ↦ ψ`. -/
theorem galois_conj_spectral_flip_eval (p : PhiPair) :
    (phiPairConj p).eval = (p.constant : Real) + (p.phi : Real) * Real.goldenConj :=
  phiPairConj_eval p

/-- Summary: the scalar fixed-point and Q-matrix eigenvalue facts agree at row 0. -/
theorem eigenform_eq_eigenvalue_row0 :
    Real.goldenRatio = 1 + 1 / Real.goldenRatio ∧
    (fibQMatrix (idx2 (0 : Fin 2) (0 : Fin 2)) : Real) * Real.goldenRatio +
      (fibQMatrix (idx2 (0 : Fin 2) (1 : Fin 2)) : Real) * 1 =
        Real.goldenRatio * Real.goldenRatio := by
  exact ⟨goldenRatio_eigenform, phi_is_eigenvalue_of_Q_row0⟩

end HeytingLean.Bridge.Veselov.HybridZeckendorf
