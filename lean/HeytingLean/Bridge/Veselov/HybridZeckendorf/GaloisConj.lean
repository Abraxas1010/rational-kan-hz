import HeytingLean.Bridge.Veselov.HybridZeckendorf.BasePhiPairEval

open scoped goldenRatio

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-!
Galois conjugation and norm on the existing exact `Z + Zφ` carrier.

The conjugation is the nontrivial algebraic map
`a + bφ ↦ a + b(1 - φ) = (a + b) - bφ`.
-/

/-- Galois conjugation on `Z[φ]`: `φ ↦ 1 - φ`. -/
def phiPairConj (p : PhiPair) : PhiPair :=
  { constant := p.constant + p.phi
    phi := -p.phi }

/-- Conjugation evaluates in the conjugate real embedding. -/
theorem phiPairConj_eval (p : PhiPair) :
    (phiPairConj p).eval = (p.constant : Real) + (p.phi : Real) * Real.goldenConj := by
  unfold phiPairConj PhiPair.eval
  push_cast
  have hψ : Real.goldenConj = 1 - Real.goldenRatio := by
    linarith [Real.goldenRatio_add_goldenConj]
  rw [hψ]
  ring

/-- Galois conjugation is an involution. -/
theorem phiPairConj_involutive : Function.Involutive phiPairConj := by
  intro p
  cases p
  ext <;> simp [phiPairConj]

/-- Galois conjugation preserves exact addition. -/
theorem phiPairConj_add (a b : PhiPair) :
    phiPairConj (phiPairAdd a b) = phiPairAdd (phiPairConj a) (phiPairConj b) := by
  cases a
  cases b
  ext <;> simp [phiPairConj, phiPairAdd] <;> ring

/-- Galois conjugation preserves exact multiplication. -/
theorem phiPairConj_mul (a b : PhiPair) :
    phiPairConj (phiPairMul a b) = phiPairMul (phiPairConj a) (phiPairConj b) := by
  cases a
  cases b
  ext <;> simp [phiPairConj, phiPairMul] <;> ring

/-- Integer norm form for `a + bφ`. -/
def phiPairNorm (p : PhiPair) : Int :=
  p.constant * p.constant + p.constant * p.phi - p.phi * p.phi

/-- Exact multiplication by the conjugate collapses into the integer norm. -/
theorem phiPairMul_conj (p : PhiPair) :
    phiPairMul p (phiPairConj p) = { constant := phiPairNorm p, phi := 0 } := by
  cases p
  ext <;> simp [phiPairConj, phiPairMul, phiPairNorm] <;> ring

/-- The integer norm is the real product of the two embeddings. -/
theorem phiPairNorm_eq_mul_conj (p : PhiPair) :
    (phiPairNorm p : Real) = p.eval * (phiPairConj p).eval := by
  calc
    (phiPairNorm p : Real) =
        ({ constant := phiPairNorm p, phi := 0 } : PhiPair).eval := by simp [PhiPair.eval]
    _ = (phiPairMul p (phiPairConj p)).eval := by rw [phiPairMul_conj]
    _ = p.eval * (phiPairConj p).eval := phiPairMul_eval p (phiPairConj p)

/-- The algebraic norm is multiplicative. -/
theorem phiPairNorm_mul (a b : PhiPair) :
    phiPairNorm (phiPairMul a b) = phiPairNorm a * phiPairNorm b := by
  cases a
  cases b
  simp [phiPairMul, phiPairNorm]
  ring

end HeytingLean.Bridge.Veselov.HybridZeckendorf
