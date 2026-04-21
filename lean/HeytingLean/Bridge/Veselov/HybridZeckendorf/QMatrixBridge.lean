import HeytingLean.Bridge.Veselov.HybridZeckendorf.BasePhiPairEval
import HeytingLean.Boundary.Hypergraph.Hypermatrix

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Boundary.Hypergraph

/-!
Q-matrix bridge for the existing exact `Z + Zφ` carrier.

The full Idziaszek convolution algorithm is not encoded here. This file pins
the structural bridge required by later work: the Fibonacci Q-matrix lives in
the project's `Hypermatrix` lane, its determinant is nontrivial, and
multiplication by `φ` on `PhiPair` is the same coefficient update.
-/

/-- The Fibonacci Q-matrix as a project hypermatrix: `Q = [[1, 1], [1, 0]]`. -/
def fibQMatrix : Hypermatrix Int 2 2 :=
  fun idx => if idx 0 = 0 then (if idx 1 = 0 then 1 else 1)
             else (if idx 1 = 0 then 1 else 0)

@[simp] theorem fibQMatrix_entry00 :
    fibQMatrix (idx2 (0 : Fin 2) (0 : Fin 2)) = 1 := by
  norm_num [fibQMatrix, idx2]

@[simp] theorem fibQMatrix_entry01 :
    fibQMatrix (idx2 (0 : Fin 2) (1 : Fin 2)) = 1 := by
  norm_num [fibQMatrix, idx2]

@[simp] theorem fibQMatrix_entry10 :
    fibQMatrix (idx2 (1 : Fin 2) (0 : Fin 2)) = 1 := by
  norm_num [fibQMatrix, idx2]

@[simp] theorem fibQMatrix_entry11 :
    fibQMatrix (idx2 (1 : Fin 2) (1 : Fin 2)) = 0 := by
  norm_num [fibQMatrix, idx2]

/-- Q-matrix multiplication via the existing order-2 hypermatrix product. -/
def fibQSquare : Hypermatrix Int 2 2 :=
  matrixMul2 fibQMatrix fibQMatrix

@[simp] theorem fibQSquare_entry00 :
    fibQSquare (idx2 (0 : Fin 2) (0 : Fin 2)) = 2 := by
  norm_num [fibQSquare, matrixMul2, fibQMatrix, idx2, Fin.sum_univ_two]

/-- The determinant of the Fibonacci Q-matrix is `-1`. -/
theorem fibQMatrix_det :
    fibQMatrix (idx2 (0 : Fin 2) (0 : Fin 2)) *
        fibQMatrix (idx2 (1 : Fin 2) (1 : Fin 2)) -
      fibQMatrix (idx2 (0 : Fin 2) (1 : Fin 2)) *
        fibQMatrix (idx2 (1 : Fin 2) (0 : Fin 2)) = -1 := by
  norm_num [fibQMatrix, idx2]

/-- Multiplication by `φ` sends `(a, b)` to `(b, a + b)`. -/
theorem phiPairMul_phi_eq_QT_action (p : PhiPair) :
    phiPairMul { constant := 0, phi := 1 } p =
      { constant := p.phi, phi := p.constant + p.phi } := by
  cases p
  ext <;> simp [phiPairMul]


/-- Identity hypermatrix for order-2 matrix multiplication. -/
def fibQId : Hypermatrix Int 2 2 :=
  fun idx => if idx 0 = idx 1 then 1 else 0

@[simp] theorem fibQId_entry00 :
    fibQId (idx2 (0 : Fin 2) (0 : Fin 2)) = 1 := by
  norm_num [fibQId, idx2]

@[simp] theorem fibQId_entry01 :
    fibQId (idx2 (0 : Fin 2) (1 : Fin 2)) = 0 := by
  norm_num [fibQId, idx2]

@[simp] theorem fibQId_entry10 :
    fibQId (idx2 (1 : Fin 2) (0 : Fin 2)) = 0 := by
  norm_num [fibQId, idx2]

@[simp] theorem fibQId_entry11 :
    fibQId (idx2 (1 : Fin 2) (1 : Fin 2)) = 1 := by
  norm_num [fibQId, idx2]

/-- Powers of the Fibonacci Q-matrix through the project's matrix product. -/
def fibQPow : Nat -> Hypermatrix Int 2 2
  | 0 => fibQId
  | n + 1 => matrixMul2 (fibQPow n) fibQMatrix

@[simp] theorem fibQPow_zero : fibQPow 0 = fibQId := rfl

@[simp] theorem fibQPow_succ (n : Nat) :
    fibQPow (n + 1) = matrixMul2 (fibQPow n) fibQMatrix := rfl

lemma fibQPow_succ_entry00 (n : Nat) :
    fibQPow (n + 1) (idx2 (0 : Fin 2) (0 : Fin 2)) =
      fibQPow n (idx2 (0 : Fin 2) (0 : Fin 2)) +
        fibQPow n (idx2 (0 : Fin 2) (1 : Fin 2)) := by
  norm_num [fibQPow, matrixMul2, fibQMatrix, idx2, Fin.sum_univ_two]

lemma fibQPow_succ_entry01 (n : Nat) :
    fibQPow (n + 1) (idx2 (0 : Fin 2) (1 : Fin 2)) =
      fibQPow n (idx2 (0 : Fin 2) (0 : Fin 2)) := by
  norm_num [fibQPow, matrixMul2, fibQMatrix, idx2, Fin.sum_univ_two]

lemma fibQPow_succ_entry10 (n : Nat) :
    fibQPow (n + 1) (idx2 (1 : Fin 2) (0 : Fin 2)) =
      fibQPow n (idx2 (1 : Fin 2) (0 : Fin 2)) +
        fibQPow n (idx2 (1 : Fin 2) (1 : Fin 2)) := by
  norm_num [fibQPow, matrixMul2, fibQMatrix, idx2, Fin.sum_univ_two]

lemma fibQPow_succ_entry11 (n : Nat) :
    fibQPow (n + 1) (idx2 (1 : Fin 2) (1 : Fin 2)) =
      fibQPow n (idx2 (1 : Fin 2) (0 : Fin 2)) := by
  norm_num [fibQPow, matrixMul2, fibQMatrix, idx2, Fin.sum_univ_two]

/-- Full entry theorem, including the `n = 0` bottom-right identity case. -/
lemma fibQPow_entries_full (n : Nat) :
    fibQPow n (idx2 (0 : Fin 2) (0 : Fin 2)) = (Nat.fib (n + 1) : Int) ∧
    fibQPow n (idx2 (0 : Fin 2) (1 : Fin 2)) = (Nat.fib n : Int) ∧
    fibQPow n (idx2 (1 : Fin 2) (0 : Fin 2)) = (Nat.fib n : Int) ∧
    fibQPow n (idx2 (1 : Fin 2) (1 : Fin 2)) =
      (if n = 0 then 1 else (Nat.fib (n - 1) : Int)) := by
  induction n with
  | zero =>
      norm_num [fibQPow, fibQId, idx2]
  | succ n ih =>
      rcases ih with ⟨h00, h01, h10, h11⟩
      refine ⟨?_, ?_, ?_, ?_⟩
      · rw [fibQPow_succ_entry00, h00, h01]
        rw [Nat.fib_add_two (n := n)]
        norm_num
        ring
      · rw [fibQPow_succ_entry01, h00]
      · rw [fibQPow_succ_entry10, h10, h11]
        by_cases hn : n = 0
        · subst n
          norm_num
        · rw [if_neg hn]
          rw [Nat.fib_add_one hn]
          norm_num
          ring
      · rw [fibQPow_succ_entry11, h10]
        simp

/-- `Q^n` has the classical Fibonacci entries. -/
theorem fibQPow_entries (n : Nat) :
    fibQPow n (idx2 (0 : Fin 2) (0 : Fin 2)) = (Nat.fib (n + 1) : Int) ∧
    fibQPow n (idx2 (0 : Fin 2) (1 : Fin 2)) = (Nat.fib n : Int) ∧
    fibQPow n (idx2 (1 : Fin 2) (0 : Fin 2)) = (Nat.fib n : Int) ∧
    (n ≥ 1 → fibQPow n (idx2 (1 : Fin 2) (1 : Fin 2)) =
      (Nat.fib (n - 1) : Int)) := by
  rcases fibQPow_entries_full n with ⟨h00, h01, h10, h11⟩
  refine ⟨h00, h01, h10, ?_⟩
  intro hn
  rw [h11]
  simp [Nat.ne_of_gt hn]

lemma idx2_eta (idx : Fin 2 -> Fin 2) : idx2 (idx 0) (idx 1) = idx := by
  funext p
  fin_cases p <;> simp [idx2]

lemma matrixMul2_right_fibQId (A : Hypermatrix Int 2 2) :
    matrixMul2 A fibQId = A := by
  funext idx
  simp [matrixMul2, fibQId, idx2, idx2_eta]

lemma matrixMul2_assoc_2 (A B C : Hypermatrix Int 2 2) :
    matrixMul2 (matrixMul2 A B) C = matrixMul2 A (matrixMul2 B C) := by
  funext idx
  simp [matrixMul2, idx2, Fin.sum_univ_two]
  ring

/-- Q-powers add exponents through the project hypermatrix product. -/
theorem fibQPow_add (m n : Nat) :
    fibQPow (m + n) = matrixMul2 (fibQPow m) (fibQPow n) := by
  induction n with
  | zero =>
      simp [matrixMul2_right_fibQId]
  | succ n ih =>
      rw [Nat.add_succ, fibQPow_succ, ih, fibQPow_succ, matrixMul2_assoc_2]

/-- The second column of `Q^n` is exactly the `PhiPair` representation of `φ^n`. -/
theorem fibQPow_column_eq_phiPairPow (n : Nat) :
    ({ constant := fibQPow n (idx2 (1 : Fin 2) (1 : Fin 2)),
       phi := fibQPow n (idx2 (0 : Fin 2) (1 : Fin 2)) } : PhiPair) =
      phiPairPow (n : Int) := by
  cases n with
  | zero =>
      change ({ constant := 1, phi := 0 } : PhiPair) =
        { constant := signedFib (Int.negSucc 0), phi := signedFib 0 }
      norm_num [signedFib]
  | succ n =>
      rcases fibQPow_entries_full (n + 1) with ⟨h00, h01, h10, h11⟩
      ext <;> simp [h01, h11, phiPairPow, signedFib]

/-- Equivalently, the first column of `Q^n` is the representation of `φ^(n+1)`. -/
theorem fibQPow_first_column_eq_phiPairPow_succ (n : Nat) :
    ({ constant := fibQPow n (idx2 (1 : Fin 2) (0 : Fin 2)),
       phi := fibQPow n (idx2 (0 : Fin 2) (0 : Fin 2)) } : PhiPair) =
      phiPairPow ((n + 1 : Nat) : Int) := by
  rcases fibQPow_entries n with ⟨h00, h01, h10, h11⟩
  ext <;> simp [h00, h10, phiPairPow, signedFib]

/-- Determinant of a two-by-two hypermatrix. -/
def det2 (A : Hypermatrix Int 2 2) : Int :=
  A (idx2 (0 : Fin 2) (0 : Fin 2)) * A (idx2 (1 : Fin 2) (1 : Fin 2)) -
    A (idx2 (0 : Fin 2) (1 : Fin 2)) * A (idx2 (1 : Fin 2) (0 : Fin 2))

lemma det2_fibQId : det2 fibQId = 1 := by
  norm_num [det2, fibQId, idx2]

lemma det2_matrixMul2 (A B : Hypermatrix Int 2 2) :
    det2 (matrixMul2 A B) = det2 A * det2 B := by
  simp [det2, matrixMul2, idx2, Fin.sum_univ_two]
  ring

/-- The determinant of `Q^n` is `(-1)^n`. -/
theorem fibQPow_det (n : Nat) : det2 (fibQPow n) = (-1 : Int) ^ n := by
  induction n with
  | zero =>
      norm_num [fibQPow, det2_fibQId]
  | succ n ih =>
      rw [fibQPow_succ, det2_matrixMul2, ih]
      have hq : det2 fibQMatrix = -1 := by
        norm_num [det2, fibQMatrix, idx2]
      rw [hq]
      ring

/-- Cassini identity, obtained from the determinant of `Q^n`. -/
theorem cassini_from_qmatrix (n : Nat) (hn : 1 ≤ n) :
    (Nat.fib (n + 1) : Int) * Nat.fib (n - 1) - (Nat.fib n : Int) ^ 2 =
      (-1 : Int) ^ n := by
  rcases fibQPow_entries n with ⟨h00, h01, h10, h11⟩
  have hdet := fibQPow_det n
  rw [det2] at hdet
  rw [h00, h01, h10, h11 hn] at hdet
  simpa [pow_two] using hdet

/-- Fibonacci addition formula extracted from `Q^(m+n) = Q^m * Q^n`. -/
theorem fib_addition_from_qmatrix (m n : Nat) (hn : 1 ≤ n) :
    (Nat.fib (m + n) : Int) =
      Nat.fib (m + 1) * Nat.fib n + Nat.fib m * Nat.fib (n - 1) := by
  rcases fibQPow_entries m with ⟨hm00, hm01, hm10, hm11⟩
  rcases fibQPow_entries n with ⟨hn00, hn01, hn10, hn11⟩
  have hleft : fibQPow (m + n) (idx2 (0 : Fin 2) (1 : Fin 2)) =
      (Nat.fib (m + n) : Int) := (fibQPow_entries (m + n)).2.1
  calc
    (Nat.fib (m + n) : Int) =
        fibQPow (m + n) (idx2 (0 : Fin 2) (1 : Fin 2)) := hleft.symm
    _ = matrixMul2 (fibQPow m) (fibQPow n) (idx2 (0 : Fin 2) (1 : Fin 2)) := by
      rw [fibQPow_add]
    _ = (Nat.fib (m + 1) : Int) * Nat.fib n + Nat.fib m * Nat.fib (n - 1) := by
      rw [matrixMul2]
      simp [idx2, Fin.sum_univ_two, hm00, hm01, hn01, hn11 hn]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
