import HeytingLean.Bridge.Veselov.HybridZeckendorf.DialecticaCore
import HeytingLean.Bridge.Veselov.HybridZeckendorf.StructuralECC
import Mathlib.Tactic

/-!
# Bridge.Veselov.HybridZeckendorf.DialecticaAttackClasses

Honest attack-class boundary for the current Zeckendorf Dialectica lane.
Supported attacks are the local zero-to-one flips already covered by the
structural ECC surface; unsupported examples are recorded explicitly.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Boundary.Homomorphic

/-- Detectable zero-to-one flip sites for a witness bitstring. -/
def Witness.detectableFlipSites (w : Witness n) : List Nat :=
  StructuralECC.detectableFlipSites w.bits

/-- Supported local attack class: zero-to-one flips at detectable sites. -/
structure SupportedAttack (w : Witness n) where
  pos : Nat
  memDetectable : pos ∈ w.detectableFlipSites
  zeroAt : w.bits.getD pos false = false

/-- Every supported attack carries the existing local adjacency signal. -/
theorem supportedAttack_adjacent_signal (w : Witness n) (a : SupportedAttack w) :
    adjacentToOneBool w.bits a.pos = true := by
  rcases (StructuralECC.mem_detectableFlipSites_iff w.bits a.pos).mp a.memDetectable with ⟨_, hadj⟩
  exact (adjacentToOneBool_eq_true w.bits a.pos).2 hadj

/-- Any actual adjacent `11` pair forces the global scan to reject. -/
theorem plainAdjacentBad_of_adjacent_true_pair :
    ∀ (bs : List Bool) (j : Nat),
      j + 1 < bs.length -> bs.getD j false = true -> bs.getD (j + 1) false = true ->
      plainAdjacentBad bs = true := by
  intro bs
  induction bs with
  | nil =>
      intro j hj
      simp at hj
  | cons a bs ih =>
      intro j hj h0 h1
      cases bs with
      | nil =>
          simp at hj
      | cons b bs =>
          cases j with
          | zero =>
              cases a <;> cases b <;> simp [plainAdjacentBad] at h0 h1 ⊢
          | succ j =>
              have hj' : j + 1 < (b :: bs).length := by simpa using hj
              have ih' := ih j hj' h0 h1
              simp [plainAdjacentBad, ih']

/-- At an in-range flipped position, the bit toggles exactly once. -/
theorem singleBitFlip_getD_self_of_lt (bs : List Bool) : ∀ {pos : Nat}, pos < bs.length ->
    (singleBitFlip bs pos).getD pos false = flipBit (bs.getD pos false) := by
  induction bs with
  | nil =>
      intro pos h
      simp at h
  | cons b bs ih =>
      intro pos h
      cases pos with
      | zero =>
          simp [singleBitFlip, flipBit]
      | succ pos =>
          have h' : pos < bs.length := by simpa using h
          simpa [singleBitFlip] using ih h'

/-- Off the flipped position, in-range bits are preserved. -/
theorem singleBitFlip_getD_of_ne (bs : List Bool) : ∀ {pos j : Nat}, j < bs.length -> pos ≠ j ->
    (singleBitFlip bs pos).getD j false = bs.getD j false := by
  induction bs with
  | nil =>
      intro pos j h
      simp at h
  | cons b bs ih =>
      intro pos j h hne
      cases pos with
      | zero =>
          cases j with
          | zero => contradiction
          | succ j => simp [singleBitFlip]
      | succ pos =>
          cases j with
          | zero => simp [singleBitFlip]
          | succ j =>
              have h' : j < bs.length := by simpa using h
              have hne' : pos ≠ j := by
                intro hEq
                apply hne
                simp [hEq]
              simpa [singleBitFlip] using ih h' hne'

/-- Supported local attacks globally reject: after the flip the attacked
bitstring contains an actual adjacent `11` pair, so the canonicality scan
returns `false`. -/
theorem supportedAttack_global_rejection (w : Witness n) (a : SupportedAttack w) :
    plainCanonicalBits (singleBitFlip w.bits a.pos) = false := by
  rcases (StructuralECC.mem_detectableFlipSites_iff w.bits a.pos).mp a.memDetectable with ⟨hpos, hadj⟩
  have hself : (singleBitFlip w.bits a.pos).getD a.pos false = true := by
    rw [singleBitFlip_getD_self_of_lt w.bits hpos, a.zeroAt]
    decide
  have hbad : plainAdjacentBad (singleBitFlip w.bits a.pos) = true := by
    cases hadj with
    | inl hright =>
        have hrightLt : a.pos + 1 < w.bits.length := by
          by_contra hnot
          have : w.bits.getD (a.pos + 1) false = false := by
            simp [List.getD, hnot]
          rw [this] at hright
          contradiction
        have hrightFlip : (singleBitFlip w.bits a.pos).getD (a.pos + 1) false = true := by
          rw [singleBitFlip_getD_of_ne w.bits hrightLt (Nat.ne_of_lt (Nat.lt_succ_self _))]
          exact hright
        have hpairLt : a.pos + 1 < (singleBitFlip w.bits a.pos).length := by
          simpa [singleBitFlip, List.length_set] using hrightLt
        exact plainAdjacentBad_of_adjacent_true_pair (singleBitFlip w.bits a.pos) a.pos hpairLt hself hrightFlip
    | inr hleft =>
        rcases hleft with ⟨hposPos, hleft⟩
        have hleftLt : a.pos - 1 < w.bits.length := by omega
        have hleftFlip : (singleBitFlip w.bits a.pos).getD (a.pos - 1) false = true := by
          rw [singleBitFlip_getD_of_ne w.bits hleftLt (by omega)]
          exact hleft
        have hposEq : a.pos - 1 + 1 = a.pos := by omega
        have hpairLt : a.pos - 1 + 1 < (singleBitFlip w.bits a.pos).length := by
          simpa [hposEq, singleBitFlip, List.length_set] using hpos
        exact plainAdjacentBad_of_adjacent_true_pair (singleBitFlip w.bits a.pos) (a.pos - 1) hpairLt hleftFlip (by simpa [hposEq] using hself)
  simp [plainCanonicalBits, hbad]

/-- Supported attacks are still ordinary bit-flip challenges in the core lane. -/
def SupportedAttack.toAttack (a : SupportedAttack w) : Attack :=
  .flip a.pos

@[simp] theorem supportedAttack_toAttack_apply (w : Witness n) (a : SupportedAttack w) :
    applyAttack a.toAttack w.bits = singleBitFlip w.bits a.pos := by
  rfl

/-- Supported attacks therefore realize the Dialectica matrix by rejection. -/
theorem supportedAttack_alpha (w : Witness n) (a : SupportedAttack w) :
    alpha n w a.toAttack := by
  left
  simpa [alpha, runAttack, supportedAttack_toAttack_apply] using supportedAttack_global_rejection w a

/-- Explicit unsupported counterexample from the current ECC surface: a local
one-to-zero flip removes the adjacent-one detector signal. -/
theorem unsupported_one_to_zero_pair_undetected :
    plainCanonicalBits (singleBitFlip [true, true] 1) = true := by
  decide

/-- Unsupported multi-bit example retained from the structural ECC surface. -/
theorem unsupported_double_flip_example :
    plainCanonicalBits (doubleBitFlip [true, false, true] 0 2) = true := by
  decide

end HeytingLean.Bridge.Veselov.HybridZeckendorf
