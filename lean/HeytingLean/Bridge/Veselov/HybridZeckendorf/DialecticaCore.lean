import HeytingLean.Bridge.Veselov.HybridZeckendorf.BitEncoding
import HeytingLean.Bridge.Veselov.HybridZeckendorf.ErrorDetection

/-!
# Bridge.Veselov.HybridZeckendorf.DialecticaCore

Project-local Dialectica-style witness/challenge carrier for Zeckendorf payloads.
The witness remains a canonical Zeckendorf representation; the challenge acts on
its emitted bitstring representation, and `alpha` evaluates the attacked state.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Boundary.Homomorphic

/-- Canonical Zeckendorf witness for the semantic value `n`. -/
structure Witness (n : Nat) where
  rep : List Nat
  canonicalRep : List.IsZeckendorfRep rep
  decodesTo : lazyEvalFib rep = n

/-- Bit-level attack surface. `none` is the identity challenge used for sanity
and realization lemmas; substantive attack classes are refined later. -/
inductive Attack where
  | none
  | flip (pos : Nat)
  deriving DecidableEq, Repr

/-- Apply a bit-level attack to an emitted Zeckendorf codeword. -/
def applyAttack : Attack -> List Bool -> List Bool
  | .none, bits => bits
  | .flip pos, bits => singleBitFlip bits pos

/-- Executable attacked-state summary used by the Dialectica matrix. -/
structure AttackResult where
  attackedBits : List Bool
  canonicalAfter : Bool
  decodedAfter : Nat
  deriving Repr

/-- Emit the witness into the bitstring carrier. -/
def Witness.bits (w : Witness n) : List Bool :=
  encodeCanonicalBits w.rep

/-- Run an attack and summarize the attacked state. -/
def runAttack (w : Witness n) (a : Attack) : AttackResult :=
  let attackedBits := applyAttack a w.bits
  { attackedBits := attackedBits
    canonicalAfter := plainCanonicalBits attackedBits
    decodedAfter := decodeBits attackedBits }

/-- Dialectica-style attacked-state matrix: after challenge `a`, either the
attacked witness is rejected as non-canonical, or it still decodes to `n`. -/
def alpha (n : Nat) (w : Witness n) (a : Attack) : Prop :=
  let result := runAttack w a
  result.canonicalAfter = false ∨ result.decodedAfter = n

@[simp] theorem applyAttack_none (bits : List Bool) :
    applyAttack .none bits = bits := rfl

@[simp] theorem applyAttack_flip (pos : Nat) (bits : List Bool) :
    applyAttack (.flip pos) bits = singleBitFlip bits pos := rfl

@[simp] theorem runAttack_none_attackedBits (w : Witness n) :
    (runAttack w .none).attackedBits = w.bits := rfl

@[simp] theorem runAttack_none_decodedAfter (w : Witness n) :
    (runAttack w .none).decodedAfter = decodeBits w.bits := rfl

@[simp] theorem witness_bits_decode (w : Witness n) :
    decodeBits w.bits = n := by
  simpa [Witness.bits, w.decodesTo] using decode_encodeCanonicalBits w.canonicalRep

/-- Every witness realizes the identity challenge: the clean emitted bitstring
still decodes to the indexed semantic value. -/
theorem alpha_none (w : Witness n) : alpha n w .none := by
  right
  simp [runAttack, witness_bits_decode]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
