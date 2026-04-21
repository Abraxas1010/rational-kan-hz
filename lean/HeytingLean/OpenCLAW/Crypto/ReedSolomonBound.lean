import Mathlib.Tactic

namespace HeytingLean.OpenCLAW.Crypto

/-!
# Reed-Solomon Arithmetic (Standard Coding Theory Background)

This module records textbook arithmetic identities used by the sprint.
It is not a paper-attributed novelty claim.

- source_type: textbook coding theory
- attribution_status: n/a (background lemma)
- claim_status: B-pass (background)
- proof_status: proved
-/

/-- Number of parity symbols in an `(n,k)` Reed-Solomon code. -/
def rsParitySymbols (n k : Nat) : Nat :=
  n - k

/-- Number of correctable symbols for `(n,k)` with bounded-distance decoding. -/
def rsCorrectableSymbols (n k : Nat) : Nat :=
  (n - k) / 2

/-- Background arithmetic identity: parity equals `n-k`. -/
theorem rs_parity_symbols (n k : Nat) :
    rsParitySymbols n k = n - k := rfl

/-- Background arithmetic identity: correctable symbols equal `(n-k)/2`. -/
theorem rs_correctable_symbols (n k : Nat) :
    rsCorrectableSymbols n k = (n - k) / 2 := rfl

/-- RS(255,223): parity symbols = 32. -/
example : rsParitySymbols 255 223 = 32 := by decide

/-- RS(255,223): correctable symbols = 16. -/
example : rsCorrectableSymbols 255 223 = 16 := by decide

end HeytingLean.OpenCLAW.Crypto
