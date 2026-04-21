import HeytingLean.LoF.Nucleus
import HeytingLean.Contracts.RoundTrip

/-!
# Basic round-trip identities across the Heyting core

This skeleton module prepares the ground for richer bridge implementations by
verifying the round-trip identity for the canonical inclusion.

Transport note: Lens transports preserve the underlying Heyting structure
because each lens carries a domain-specific stabilizer `R_domain` (an
interior/nucleus). Round-trip and logical-shadow contracts are the algebraic
expression of the reflector/inclusion pattern at the core of the project.
-/

namespace HeytingLean
namespace Bridges

open HeytingLean.LoF

variable {α : Type u} [PrimaryAlgebra α] (R : Reentry α)

/-- The default encoder sends a core element to its ambient representative. -/
def encode (a : R.Omega) : α := a

/-- The default decoder interiorises an ambient element along the nucleus. -/
def decode (a : α) : R.Omega :=
  Reentry.Omega.mk (R := R) (R a) (R.idempotent a)

@[simp] lemma decode_encode (a : R.Omega) :
    decode (R := R) (encode (R := R) a) = a := by
  change Reentry.Omega.mk (R := R) (R (a : α)) (R.idempotent _) = a
  ext
  simp [Reentry.Omega.apply_coe]

@[simp] lemma encode_decode (a : α) :
    encode (R := R) (decode (R := R) a) = R a := rfl

/-- The identity bridge realises the generic round-trip contract. -/
def identityContract : Contracts.RoundTrip (R := R) α where
  encode := encode (R := R)
  decode := decode (R := R)
  round := decode_encode (R := R)

end Bridges
end HeytingLean
