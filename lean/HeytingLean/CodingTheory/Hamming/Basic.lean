import Mathlib.InformationTheory.Hamming
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Tactic

namespace HeytingLean
namespace CodingTheory
namespace Hamming

/-!
# Hamming basics (Bit vectors, weight, distance, repetition code)
-/

abbrev Bit := ZMod 2
abbrev BitVec (n : Nat) := Fin n → Bit

def hammingDist {n : Nat} (x y : BitVec n) : Nat :=
  _root_.hammingDist x y

def hammingWeight {n : Nat} (x : BitVec n) : Nat :=
  _root_.hammingNorm x

def repetitionCode (n : Nat) : Set (BitVec n) :=
  {x | ∀ i j : Fin n, x i = x j}

def repetitionEncode (n : Nat) (b : Bit) : BitVec n := fun _ => b

def bitToNat (b : Bit) : Nat := if b = 0 then 0 else 1

def onesCount {n : Nat} (x : BitVec n) : Nat :=
  (Finset.univ.sum fun i : Fin n => bitToNat (x i))

def majorityBit {n : Nat} (x : BitVec n) : Bit :=
  if 2 * onesCount x > n then 1 else 0

def repetitionDecode {n : Nat} (x : BitVec n) : Bit :=
  majorityBit x

lemma repetition_encode_mem (n : Nat) (b : Bit) :
    repetitionCode n (repetitionEncode n b) := by
  intro i j
  rfl

@[simp] lemma bitToNat_zero : bitToNat (0 : Bit) = 0 := by
  simp [bitToNat]

@[simp] lemma bitToNat_one : bitToNat (1 : Bit) = 1 := by
  simp [bitToNat]

@[simp] lemma bit_add_one_add_one (b : Bit) : b + 1 + 1 = b := by
  fin_cases b <;> decide

@[simp] lemma bit_one_add_one : (1 : Bit) + 1 = 0 := by
  decide

@[simp] lemma bit_zero_add_one : (0 : Bit) + 1 = 1 := by
  decide

lemma bit_ne_iff_add_one_eq (a b : Bit) : a ≠ b ↔ b = a + 1 := by
  fin_cases a <;> fin_cases b <;> simp

lemma onesCount_const (n : Nat) (b : Bit) :
    onesCount (fun _ : Fin n => b) = n * bitToNat b := by
  classical
  simp [onesCount, Finset.sum_const, Fintype.card_fin]

lemma repetitionDecode_encode {n : Nat} (b : Bit) (hn : 0 < n) :
    repetitionDecode (repetitionEncode n b) = b := by
  classical
  fin_cases b
  ·
    have hle : ¬ 2 * onesCount (repetitionEncode n 0) > n := by
      have hcount : onesCount (repetitionEncode n 0) = 0 := by
        simpa using (onesCount_const n (0 : Bit))
      have : 2 * onesCount (repetitionEncode n 0) ≤ n := by
        simp [hcount]
      exact not_lt_of_ge this
    simp [repetitionDecode, majorityBit, hle]
  ·
    have hgt : 2 * onesCount (repetitionEncode n 1) > n := by
      have hcount : onesCount (repetitionEncode n 1) = n := by
        simpa using (onesCount_const n (1 : Bit))
      have : 2 * n > n := by omega
      simpa [hcount] using this
    simp [repetitionDecode, majorityBit, hgt]

end Hamming
end CodingTheory
end HeytingLean
