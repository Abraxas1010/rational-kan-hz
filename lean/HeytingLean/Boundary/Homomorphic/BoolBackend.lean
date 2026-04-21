namespace HeytingLean.Boundary.Homomorphic

universe u

/-- Abstract Boolean gate surface for encrypted Boolean wires.

The only semantic observation exposed to Lean is `dec`; concrete runtimes can implement
`EncBool` with plaintext booleans, TFHE ciphertext handles, or another backend, provided
the gate laws below hold. -/
class HomBoolBackend where
  EncBool : Type u
  dec : EncBool -> Bool
  htrue : EncBool
  hfalse : EncBool
  hnot : EncBool -> EncBool
  hand : EncBool -> EncBool -> EncBool
  hor : EncBool -> EncBool -> EncBool
  dec_htrue : dec htrue = true
  dec_hfalse : dec hfalse = false
  dec_hnot : forall x, dec (hnot x) = !dec x
  dec_hand : forall x y, dec (hand x y) = (dec x && dec y)
  dec_hor : forall x y, dec (hor x y) = (dec x || dec y)

namespace HomBoolBackend

/-- Plain Boolean backend used only as an executable oracle and sanity target. -/
instance plain : HomBoolBackend where
  EncBool := Bool
  dec x := x
  htrue := true
  hfalse := false
  hnot := Bool.not
  hand := fun x y => x && y
  hor := fun x y => x || y
  dec_htrue := rfl
  dec_hfalse := rfl
  dec_hnot := by intro x; rfl
  dec_hand := by intro x y; rfl
  dec_hor := by intro x y; rfl

end HomBoolBackend

end HeytingLean.Boundary.Homomorphic
