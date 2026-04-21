import Mathlib.Data.Finsupp.Basic
import Mathlib.Data.Finsupp.Ext
import Mathlib.Algebra.Group.Finsupp
import Mathlib.Data.Int.Basic
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibIdentities

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

abbrev ShiftCoeff := Int
abbrev ShiftSupport := Int →₀ Int

/-- Fibonacci weights extended to negative indices by the standard negafibonacci rule. -/
def signedFib : Int → Int
  | .ofNat n => Nat.fib n
  | .negSucc n => if n % 2 = 0 then Nat.fib (n + 1) else -(Nat.fib (n + 1) : Int)

/-- Exact integer semantics for a finitely-supported shifted Fibonacci combination. -/
noncomputable def shiftEval (s : ShiftSupport) : Int :=
  s.sum (fun i coeff => coeff * signedFib i)

/-- Single shifted coefficient. -/
noncomputable def shiftSingle (i coeff : Int) : ShiftSupport :=
  Finsupp.single i coeff

@[simp] theorem shiftEval_zero : shiftEval 0 = 0 := by
  simp [shiftEval]

@[simp] theorem shiftEval_single (i coeff : Int) :
    shiftEval (shiftSingle i coeff) = coeff * signedFib i := by
  simpa [shiftSingle, shiftEval] using
    (Finsupp.sum_single_index (a := i) (b := coeff)
      (h := fun j c => c * signedFib j)
      (by simp : (0 : Int) * signedFib i = 0))

theorem shiftEval_add (a b : ShiftSupport) :
    shiftEval (a + b) = shiftEval a + shiftEval b := by
  classical
  simpa [shiftEval, Int.add_mul, Int.mul_add, Int.add_assoc, Int.add_left_comm, Int.add_comm] using
    (Finsupp.sum_add_index'
      (f := a) (g := b)
      (h := fun i coeff => coeff * signedFib i)
      (h_zero := by
        intro i
        simp)
      (h_add := by
        intro i x y
        simp [Int.add_mul]))

/-- Shift every support index by `k`. -/
noncomputable def shiftBy (k : Int) (s : ShiftSupport) : ShiftSupport :=
  Finsupp.mapDomain (fun i => i + k) s

@[simp] theorem shiftBy_zero (k : Int) : shiftBy k 0 = 0 := by
  simp [shiftBy]

@[simp] theorem shiftBy_single (k i coeff : Int) :
    shiftBy k (shiftSingle i coeff) = shiftSingle (i + k) coeff := by
  simpa [shiftBy, shiftSingle] using
    (Finsupp.mapDomain_single (f := fun j : Int => j + k) (a := i) (b := coeff))

/-- Shifting the support transports evaluation by shifting the signed Fibonacci index. -/
theorem shiftEval_shiftBy (k : Int) (s : ShiftSupport) :
    shiftEval (shiftBy k s) = s.sum (fun i coeff => coeff * signedFib (i + k)) := by
  classical
  induction s using Finsupp.induction_linear with
  | zero =>
      simp [shiftBy, shiftEval]
  | single i coeff =>
      have hLeft : shiftEval (shiftBy k (Finsupp.single i coeff)) = coeff * signedFib (i + k) := by
        change shiftEval (shiftBy k (shiftSingle i coeff)) = coeff * signedFib (i + k)
        rw [shiftBy_single]
        simp
      have hRight : (Finsupp.single i coeff : ShiftSupport).sum (fun j c => c * signedFib (j + k)) = coeff * signedFib (i + k) := by
        simpa using
          (Finsupp.sum_single_index (a := i) (b := coeff)
            (h := fun j c => c * signedFib (j + k))
            (by simp : (0 : Int) * signedFib (i + k) = 0))
      exact hLeft.trans hRight.symm
  | add a b ha hb =>
      have ha' : shiftEval (Finsupp.mapDomain (fun i : Int => i + k) a) = a.sum (fun i coeff => coeff * signedFib (i + k)) := by
        simpa [shiftBy] using ha
      have hb' : shiftEval (Finsupp.mapDomain (fun i : Int => i + k) b) = b.sum (fun i coeff => coeff * signedFib (i + k)) := by
        simpa [shiftBy] using hb
      have hsum : (a + b).sum (fun i coeff => coeff * signedFib (i + k)) =
          a.sum (fun i coeff => coeff * signedFib (i + k)) +
          b.sum (fun i coeff => coeff * signedFib (i + k)) := by
        simpa [Int.add_mul, Int.add_assoc, Int.add_left_comm, Int.add_comm] using
          (Finsupp.sum_add_index'
            (f := a) (g := b)
            (h := fun i coeff => coeff * signedFib (i + k))
            (h_zero := by intro i; simp)
            (h_add := by intro i x y; simp [Int.add_mul]))
      rw [shiftBy, Finsupp.mapDomain_add, shiftEval_add, ha', hb', ← hsum]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
