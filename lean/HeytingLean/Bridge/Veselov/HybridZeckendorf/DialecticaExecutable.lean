import HeytingLean.Bridge.Veselov.HybridZeckendorf.DialecticaMorphisms

/-!
# Bridge.Veselov.HybridZeckendorf.DialecticaExecutable

Executable finite checker surface for the supported Zeckendorf Dialectica attack class.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Boundary.Homomorphic

/-- Finite supported positions: detectable sites whose current bit is zero, so
flipping them is a genuine zero-to-one attack. -/
def supportedAttackPositions (w : Witness n) : List Nat :=
  w.detectableFlipSites.filter (fun pos => !(w.bits.getD pos false))

/-- Boolean rejection test for one supported-position candidate. -/
def supportedAttackRejected (w : Witness n) (pos : Nat) : Bool :=
  !(plainCanonicalBits (singleBitFlip w.bits pos))

/-- Finite exhaustive check over the supported attack positions for `w`. The
enumeration follows the detectable-site carrier and discharges the zero-bit
side condition locally. -/
def checkSupportedAttacks (w : Witness n) : Bool :=
  w.detectableFlipSites.all (fun pos =>
    if w.bits.getD pos false then
      true
    else
      supportedAttackRejected w pos)

@[simp] theorem supportedAttackRejected_eq_true (w : Witness n) {pos : Nat}
    (hmem : pos ∈ w.detectableFlipSites) (hzero : w.bits.getD pos false = false) :
    supportedAttackRejected w pos = true := by
  let a : SupportedAttack w := { pos := pos, memDetectable := hmem, zeroAt := hzero }
  unfold supportedAttackRejected
  simpa [a] using supportedAttack_global_rejection w a

/-- The executable finite checker accepts every witness on its supported attack class. -/
theorem checkSupportedAttacks_eq_true (w : Witness n) :
    checkSupportedAttacks w = true := by
  unfold checkSupportedAttacks
  rw [List.all_eq_true]
  intro pos hmem
  cases hbit : w.bits.getD pos false with
  | true =>
      simp
  | false =>
      simpa [hbit] using supportedAttackRejected_eq_true w hmem hbit

/-- Specialization to the canonical witness for `n`. -/
@[simp] theorem checkCanonicalWitness_eq_true (n : Nat) :
    checkSupportedAttacks (canonicalWitness n) = true :=
  checkSupportedAttacks_eq_true (canonicalWitness n)

end HeytingLean.Bridge.Veselov.HybridZeckendorf
