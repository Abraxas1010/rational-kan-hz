import HeytingLean.Boundary.Homomorphic.ZeckNormFHE
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFrac

namespace HeytingLean.Boundary.Homomorphic

open HeytingLean.Bridge.Veselov.HybridZeckendorf

/-! # FHE pipeline for fractional Fibonacci parts -/

/-- Plaintext fractional selection is an identity pass for a finite bitmap. -/
def plainFracSelect (bits : List Bool) (_k : Nat) : List Bool :=
  bits

/-- Read a fractional selection bit at position `k` from the bitmap. -/
def getFracBit (bits : List Bool) (k : Nat) : Bool :=
  getBit bits k

/-- Plaintext fractional evaluation: sum of `fibFrac (i+1)` at set positions. -/
noncomputable def plainFracEval (bits : List Bool) : ℚ :=
  (List.range bits.length).foldl
    (fun acc i => acc + if getBit bits i then fibFrac (i + 1) else 0) 0

/-- Encrypted fractional bitmap: each bit is encrypted separately. -/
def homFracBitmap [B : HomBoolBackend] (bits : List B.EncBool) : List B.EncBool :=
  bits

/-- Decryption of the fractional bitmap gives the plaintext bitmap. -/
theorem homFracBitmap_decrypt [B : HomBoolBackend] (bits : List B.EncBool) :
    (homFracBitmap bits).map B.dec = plainFracSelect (bits.map B.dec) 0 := by
  rfl

/-- Combined integer + fractional FHE pipeline. -/
def homFullPipeline [B : HomBoolBackend] (intBits fracBits : List B.EncBool)
    (carrySteps : Nat) : List B.EncBool × List B.EncBool :=
  (iterateCarrySteps carrySteps intBits, homFracBitmap fracBits)

/-- The full pipeline decrypts correctly on the fractional lane and preserves
the selected integer pipeline result by definition. -/
theorem homFullPipeline_decrypt [B : HomBoolBackend]
    (intBits fracBits : List B.EncBool) (n : Nat) :
    let out := homFullPipeline intBits fracBits n
    out.1.map B.dec = (iterateCarrySteps n intBits).map B.dec ∧
      out.2.map B.dec = plainFracSelect (fracBits.map B.dec) 0 := by
  simp [homFullPipeline, homFracBitmap, plainFracSelect]

end HeytingLean.Boundary.Homomorphic
