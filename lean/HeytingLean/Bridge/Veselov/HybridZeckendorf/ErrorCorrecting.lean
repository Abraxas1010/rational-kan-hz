import Mathlib.Data.Vector.Basic
import HeytingLean.CodingTheory.Hamming.Basic
import HeytingLean.OpenCLAW.Crypto.ReedSolomonBound
import HeytingLean.Bridge.Veselov.HybridZeckendorf.ErrorDetection
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFloatingPoint

/-!
# Veselov FNS: block error-correcting code interface

This module defines the **abstract ECC interface** for the Fibonacci block layer.
`SystematicECC` axiomatizes encode/decode round-trip and correction up to `t`
errors without committing to a concrete code construction. The `ReedSolomonBlockParams`
record names the GF(2^m) parameter regime but no concrete RS instance is provided here.

Open gap: instantiating `SystematicECC` with a verified Reed-Solomon or Hamming
implementation. The current surface proves that *any* code satisfying the interface
yields correct block encode/decode and error correction — it does not prove that
such a code exists in this codebase.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-- Block partition of a fractional bit stream. -/
def blockPartition (bits : List Bool) (blockSize : Nat) : List (List Bool) :=
  if blockSize = 0 then [bits] else bits.toChunks blockSize

/-- Abstract systematic binary ECC used by the FNS block layer. The `distance`
field keeps the correction theorem honest without committing this file to one
particular bit-vector metric implementation. -/
structure SystematicECC (n k t : Nat) where
  encode : Vector Bool k -> Vector Bool n
  decode : Vector Bool n -> Option (Vector Bool k)
  distance : Vector Bool n -> Vector Bool n -> Nat
  encode_decode : ∀ msg, decode (encode msg) = some msg
  correct_up_to_t : ∀ msg received, distance (encode msg) received ≤ t -> decode received = some msg

/-- Reed-Solomon parameter record for the fractional block lane. This names the
GF(2^m)-style construction without axiomatizing a field implementation here. -/
structure ReedSolomonBlockParams where
  symbolBits : Nat
  n : Nat
  k : Nat
  t : Nat
  t_eq_bound : t = HeytingLean.OpenCLAW.Crypto.rsCorrectableSymbols n k
  length_fits_field : n ≤ 2 ^ symbolBits
  deriving Repr

structure FibCodeword where
  blocks : List (List Bool)
  intPart : ZeckPayload
  normFlag : Bool
  deriving Repr

/-- A single encoded fractional block carrying both the transmitted symbols and
the source vector used for the systematic-code correctness theorem. -/
structure FibECCBlock (n k : Nat) where
  original : Vector Bool k
  encoded : Vector Bool n

def fibECCBlockEncode {n k t : Nat} (code : SystematicECC n k t)
    (msg : Vector Bool k) : FibECCBlock n k where
  original := msg
  encoded := code.encode msg

def fibECCBlockDecode {n k t : Nat} (code : SystematicECC n k t)
    (received : Vector Bool n) : Option (Vector Bool k) :=
  code.decode received

theorem fibECCBlockDecode_encode {n k t : Nat} (code : SystematicECC n k t)
    (msg : Vector Bool k) :
    fibECCBlockDecode code (fibECCBlockEncode code msg).encoded = some msg := by
  exact code.encode_decode msg

theorem fibECCBlockDecode_correct_up_to_t {n k t : Nat} (code : SystematicECC n k t)
    (msg : Vector Bool k) (received : Vector Bool n)
    (herrors : code.distance (code.encode msg) received ≤ t) :
    fibECCBlockDecode code received = some msg := by
  exact code.correct_up_to_t msg received herrors

def fibBlockEncode (f : FibFloat) (blockSize : Nat) : FibCodeword where
  blocks := blockPartition f.fracPart blockSize
  intPart := f.intPart
  normFlag := f.normFlag

def fibBlockDecode (cw : FibCodeword) : FibFloat where
  intPart := cw.intPart
  fracPart := cw.blocks.flatten
  normFlag := cw.normFlag

theorem fibBlockDecode_encode_flatten (f : FibFloat) :
    (fibBlockDecode (fibBlockEncode f 0)).fracPart = f.fracPart := by
  simp [fibBlockEncode, fibBlockDecode, blockPartition]

/-- Conservative finite-precision exceptional-count budget. -/
def exceptionalCount (L : Nat) : Nat :=
  L + 1

theorem reedSolomon_correctable_symbols (p : ReedSolomonBlockParams) :
    p.t = HeytingLean.OpenCLAW.Crypto.rsCorrectableSymbols p.n p.k :=
  p.t_eq_bound

theorem reedSolomon_length_fits_field (p : ReedSolomonBlockParams) :
    p.n ≤ 2 ^ p.symbolBits :=
  p.length_fits_field

end HeytingLean.Bridge.Veselov.HybridZeckendorf
