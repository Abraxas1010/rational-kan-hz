import HeytingLean.Boundary.Homomorphic.ZeckendorfCanonical
import HeytingLean.Bridge.Veselov.HybridZeckendorf.LinearTimeAdd

/-!
# Veselov FNS: consecutive-one error detector
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Boundary.Homomorphic

def hasConsecutiveOnes (bs : List Bool) : Bool :=
  plainAdjacentBad bs

def flipBit : Bool -> Bool
  | true => false
  | false => true

def singleBitFlip (bs : List Bool) (pos : Nat) : List Bool :=
  bs.set pos (flipBit (bs.getD pos false))

def doubleBitFlip (bs : List Bool) (pos1 pos2 : Nat) : List Bool :=
  singleBitFlip (singleBitFlip bs pos1) pos2

def canonicalBits (bs : List Bool) : Bool :=
  !hasConsecutiveOnes bs

def adjacentToOne (bs : List Bool) (pos : Nat) : Prop :=
  bs.getD (pos + 1) false = true ∨ (0 < pos ∧ bs.getD (pos - 1) false = true)

def adjacentToOneBool (bs : List Bool) (pos : Nat) : Bool :=
  bs.getD (pos + 1) false || (decide (0 < pos) && bs.getD (pos - 1) false)

theorem adjacentToOneBool_eq_true (bs : List Bool) (pos : Nat) :
    adjacentToOneBool bs pos = true ↔ adjacentToOne bs pos := by
  unfold adjacentToOneBool adjacentToOne
  by_cases hp : 0 < pos <;> simp [hp]

/-- Bidirectional characterization for the canonical two-bit local window:
flipping the right bit creates `11` exactly when the left bit was already one. -/
theorem consecutive_detects_right_flip_pair (b : Bool) :
    hasConsecutiveOnes (singleBitFlip [true, b] 1) = true ↔ b = false := by
  cases b <;> decide

/-- The general local form of the detector for a zero-to-one flip: the local
adjacency signal is present exactly when the flipped zero is adjacent to a one. -/
theorem zero_to_one_local_detector_iff (bs : List Bool) (pos : Nat)
    (hzero : bs.getD pos false = false) :
    adjacentToOneBool bs pos = true ↔
      (bs.getD pos false = false ∧ adjacentToOne bs pos) := by
  constructor
  · intro h
    exact ⟨hzero, (adjacentToOneBool_eq_true bs pos).mp h⟩
  · intro h
    exact (adjacentToOneBool_eq_true bs pos).mpr h.2

/-- For in-range zero-to-one flips in a canonical two-sided local window, the
paper's consecutive-one detector fires exactly when the flipped position is
adjacent to an existing one. This theorem is local because the paper's detector
is local: other independently corrupted windows require independent redundancy. -/
theorem consecutive_detects_some_single_flips_local (left bit right : Bool)
    (hbit : bit = false) :
    hasConsecutiveOnes (singleBitFlip [left, bit, right] 1) = true ↔
      adjacentToOne [left, bit, right] 1 := by
  cases left <;> cases bit <;> cases right <;> simp [hbit, hasConsecutiveOnes,
    singleBitFlip, flipBit, adjacentToOne, plainAdjacentBad]

/-- A 1-to-0 flip in the same local window removes, rather than creates, the
adjacent-one detector signal. -/
theorem one_to_zero_flip_pair_undetected :
    hasConsecutiveOnes (singleBitFlip [true, true] 1) = false := by
  decide

/-- Explicit multi-bit limitation: two flips can erase the detector signal. -/
example :
    canonicalBits [true, false, true] = true ∧
      hasConsecutiveOnes (doubleBitFlip [true, false, true] 0 2) = false := by
  decide

end HeytingLean.Bridge.Veselov.HybridZeckendorf
