import HeytingLean.Boundary.Homomorphic.BoolBackend

namespace HeytingLean.Boundary.Homomorphic

/-- Plain Zeckendorf canonicality violation detector: true iff adjacent true bits occur. -/
def plainAdjacentBad : List Bool -> Bool
  | [] => false
  | [_] => false
  | a :: b :: rest => (a && b) || plainAdjacentBad (b :: rest)

/-- Plain fixed-width Zeckendorf canonicality check: no adjacent true bits. -/
def plainCanonicalBits (bits : List Bool) : Bool :=
  !plainAdjacentBad bits

/-- Homomorphic adjacent-bit violation detector over an abstract Boolean backend. -/
def homAdjacentBad [B : HomBoolBackend] : List B.EncBool -> B.EncBool
  | [] => B.hfalse
  | [_] => B.hfalse
  | a :: b :: rest => B.hor (B.hand a b) (homAdjacentBad (b :: rest))

/-- Homomorphic fixed-width Zeckendorf canonicality check. -/
def homCanonical [B : HomBoolBackend] (bits : List B.EncBool) : B.EncBool :=
  B.hnot (homAdjacentBad bits)

/-- The encrypted adjacent-violation circuit decrypts to the plaintext adjacent scan. -/
theorem homAdjacentBad_decrypt_eq_plain [B : HomBoolBackend] (bits : List B.EncBool) :
    B.dec (homAdjacentBad bits) = plainAdjacentBad (bits.map B.dec) := by
  induction bits with
  | nil =>
      simp [homAdjacentBad, plainAdjacentBad, HomBoolBackend.dec_hfalse]
  | cons a rest ih =>
      cases rest with
      | nil =>
          simp [homAdjacentBad, plainAdjacentBad, HomBoolBackend.dec_hfalse]
      | cons b rest =>
          simp [homAdjacentBad, plainAdjacentBad, HomBoolBackend.dec_hor,
            HomBoolBackend.dec_hand, ih]

/-- The homomorphic canonicality circuit decrypts to the plaintext no-adjacent-ones check. -/
theorem homCanonical_decrypt_eq_plain [B : HomBoolBackend] (bits : List B.EncBool) :
    B.dec (homCanonical bits) = plainCanonicalBits (bits.map B.dec) := by
  simp [homCanonical, plainCanonicalBits, HomBoolBackend.dec_hnot,
    homAdjacentBad_decrypt_eq_plain]

end HeytingLean.Boundary.Homomorphic
