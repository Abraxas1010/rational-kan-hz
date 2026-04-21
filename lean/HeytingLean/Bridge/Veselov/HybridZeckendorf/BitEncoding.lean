import Mathlib.Data.Nat.Fib.Zeckendorf
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibIdentities
import HeytingLean.Boundary.Homomorphic.ZeckendorfCanonical

/-!
# Bridge.Veselov.HybridZeckendorf.BitEncoding

Bit-level encoding bridge for canonical Zeckendorf payloads.
The recursive encoder follows the descending Zeckendorf index order directly,
so later proofs can reuse the local non-consecutivity invariant.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Boundary.Homomorphic

/-- Encode a descending canonical Zeckendorf index list as little-endian bits. -/
def encodeCanonicalBits : List Nat -> List Bool
  | [] => []
  | a :: zs =>
      let tail := encodeCanonicalBits zs
      tail ++ List.replicate (a - tail.length) false ++ [true]

/-- Decode little-endian Fibonacci bits starting from index `i`. -/
def decodeBitsFrom : Nat -> List Bool -> Nat
  | _, [] => 0
  | i, b :: bs => (if b then Nat.fib i else 0) + decodeBitsFrom (i + 1) bs

/-- Decode little-endian Fibonacci bits to their semantic value. -/
def decodeBits (bits : List Bool) : Nat :=
  decodeBitsFrom 0 bits

/-- Semantic encoding bridge from lazy Zeckendorf payloads to canonical bits. -/
def encodeBits (z : List Nat) : List Bool :=
  encodeCanonicalBits (Nat.zeckendorf (lazyEvalFib z))

@[simp] theorem decodeBitsFrom_nil (i : Nat) :
    decodeBitsFrom i [] = 0 := rfl

@[simp] theorem decodeBits_nil : decodeBits [] = 0 := rfl

@[simp] theorem decodeBitsFrom_cons (i : Nat) (b : Bool) (bs : List Bool) :
    decodeBitsFrom i (b :: bs) = (if b then Nat.fib i else 0) + decodeBitsFrom (i + 1) bs := rfl

@[simp] theorem decodeBits_cons (b : Bool) (bs : List Bool) :
    decodeBits (b :: bs) = (if b then Nat.fib 0 else 0) + decodeBitsFrom 1 bs := rfl

theorem decodeBitsFrom_append (i : Nat) (xs ys : List Bool) :
    decodeBitsFrom i (xs ++ ys) = decodeBitsFrom i xs + decodeBitsFrom (i + xs.length) ys := by
  induction xs generalizing i with
  | nil =>
      simp [decodeBitsFrom]
  | cons b bs ih =>
      simp [decodeBitsFrom, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

theorem decodeBitsFrom_replicate_false (i k : Nat) :
    decodeBitsFrom i (List.replicate k false) = 0 := by
  induction k generalizing i with
  | zero => simp
  | succ k ih =>
      simp [List.replicate]
      simpa using ih (i + 1)


theorem isZeckendorfRep_tail {a b : Nat} {bs : List Nat}
    (h : List.IsZeckendorfRep (a :: b :: bs)) : List.IsZeckendorfRep (b :: bs) := by
  simp [List.IsZeckendorfRep] at h ⊢
  exact h.2

theorem encodeCanonicalBits_length {z : List Nat} (h : List.IsZeckendorfRep z) :
    match z with
    | [] => (encodeCanonicalBits z).length = 0
    | a :: _ => (encodeCanonicalBits z).length = a + 1 := by
  induction z with
  | nil =>
      simp [encodeCanonicalBits]
  | cons a zs ih =>
      cases zs with
      | nil =>
          simp [encodeCanonicalBits]
      | cons b bs =>
          have htail : List.IsZeckendorfRep (b :: bs) := isZeckendorfRep_tail h
          have ih' := ih htail
          have hle : b + 1 ≤ a := by
            simp [List.IsZeckendorfRep] at h
            omega
          rw [encodeCanonicalBits]
          repeat rw [List.length_append]
          simp [ih', List.length_replicate]
          omega

theorem decodeBits_append_gap_true (tail : List Bool) (a : Nat) (hlen : tail.length ≤ a) :
    decodeBits (tail ++ List.replicate (a - tail.length) false ++ [true]) =
      decodeBits tail + Nat.fib a := by
  unfold decodeBits
  rw [decodeBitsFrom_append, decodeBitsFrom_append]
  simp [decodeBitsFrom_replicate_false, Nat.add_sub_of_le hlen]

theorem decode_encodeCanonicalBits {z : List Nat} (h : List.IsZeckendorfRep z) :
    decodeBits (encodeCanonicalBits z) = lazyEvalFib z := by
  induction z with
  | nil =>
      simp [encodeCanonicalBits, decodeBits, lazyEvalFib]
  | cons a zs ih =>
      cases zs with
      | nil =>
          have hlen : ([] : List Bool).length ≤ a := by simp
          simpa [encodeCanonicalBits, lazyEvalFib, decodeBits] using decodeBits_append_gap_true [] a hlen
      | cons b bs =>
          have htail : List.IsZeckendorfRep (b :: bs) := isZeckendorfRep_tail h
          have ih' := ih htail
          have hlen : (encodeCanonicalBits (b :: bs)).length ≤ a := by
            have htop : (encodeCanonicalBits (b :: bs)).length = b + 1 := by
              simpa using encodeCanonicalBits_length htail
            simp [List.IsZeckendorfRep] at h
            omega
          calc
            decodeBits (encodeCanonicalBits (a :: b :: bs))
                = decodeBits (encodeCanonicalBits (b :: bs)) + Nat.fib a := by
                    simpa [encodeCanonicalBits] using
                      decodeBits_append_gap_true (encodeCanonicalBits (b :: bs)) a hlen
            _ = lazyEvalFib (b :: bs) + Nat.fib a := by rw [ih']
            _ = lazyEvalFib (a :: b :: bs) := by
                  simp [lazyEvalFib, Nat.add_left_comm, Nat.add_comm]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
