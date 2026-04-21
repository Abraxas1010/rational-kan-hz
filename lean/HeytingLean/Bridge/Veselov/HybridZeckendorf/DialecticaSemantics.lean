import HeytingLean.Bridge.Veselov.HybridZeckendorf.DialecticaCore
import HeytingLean.Bridge.Veselov.HybridZeckendorf.Uniqueness

/-!
# Bridge.Veselov.HybridZeckendorf.DialecticaSemantics

Witness-family semantics for the Zeckendorf Dialectica carrier.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-- Canonical witness for the natural number `n`. -/
def canonicalWitness (n : Nat) : Witness n where
  rep := Nat.zeckendorf n
  canonicalRep := Nat.isZeckendorfRep_zeckendorf n
  decodesTo := by
    change ((Nat.zeckendorf n).map Nat.fib).sum = n
    exact Nat.sum_zeckendorf_fib n

/-- Every natural number admits a canonical witness. -/
theorem witness_exists (n : Nat) : Nonempty (Witness n) :=
  ⟨canonicalWitness n⟩

@[simp] theorem canonicalWitness_rep (n : Nat) :
    (canonicalWitness n).rep = Nat.zeckendorf n := rfl

@[simp] theorem canonicalWitness_bits_decode (n : Nat) :
    decodeBits (canonicalWitness n).bits = n := by
  exact witness_bits_decode (canonicalWitness n)

/-- Canonical witnesses are unique at fixed semantic value. -/
theorem witness_rep_unique (w : Witness n) :
    w.rep = (canonicalWitness n).rep := by
  apply zeckendorf_unique w.rep (canonicalWitness n).rep w.canonicalRep (canonicalWitness n).canonicalRep
  rw [w.decodesTo, (canonicalWitness n).decodesTo]

/-- Therefore the emitted bitstrings are also unique at fixed semantic value. -/
theorem witness_bits_unique (w : Witness n) :
    w.bits = (canonicalWitness n).bits := by
  simp [Witness.bits, witness_rep_unique w]

end HeytingLean.Bridge.Veselov.HybridZeckendorf
