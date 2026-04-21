import HeytingLean.Bridge.Veselov.HybridZeckendorf.DialecticaMorphisms
import HeytingLean.Bridge.Veselov.HybridZeckendorf.DialecticaExecutable

/-!
# Bridge.Veselov.HybridZeckendorf.DialecticaVeselov

Veselov's claim lifted from empirical arithmetic observation to a
proof-theoretic Dialectica theorem.

The canonical Zeckendorf witness of every `n : Nat` uniformly realizes the
Dialectica matrix `α` against every supported adversarial flip. The empirical
"detection rate = 1.0" measured in the Rust harness is therefore not a
statistic but a necessary consequence of a universal theorem: the count of
rejected attacks equals the trial count on any finite batch of canonical
witnesses, because each individual rejection is kernel-checked from
`supportedAttack_global_rejection`.

The headline statement `veselov_dialectica_theorem` is the proof-theoretic
anchor the closeout cert binds as its primary theorem. The legacy names
(`checkCanonicalWitness_eq_true`, `canonicalWitness_refines_supported`,
`supportedAttack_global_rejection`) are now proper specializations of this
headline rather than separate first-class claims.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Boundary.Homomorphic

/-- **Veselov's claim as a proof-theoretic Dialectica theorem.**
For every natural `n` and every supported-class adversarial flip `a` on the
canonical Zeckendorf witness of `n`, the Dialectica matrix `α n u a` is
realized — the attacked encoding is rejected as non-canonical. The result
is universally quantified and extracted from the witness/challenge
semantics rather than measured. -/
theorem veselov_dialectica_theorem :
    ∀ (n : Nat) (a : SupportedAttack (canonicalWitness n)),
      alpha n (canonicalWitness n) a.toAttack :=
  fun n a => supportedAttack_alpha (canonicalWitness n) a

/-- Bit-level form of Veselov's claim: canonical Zeckendorf bitstrings
reject every supported-class adversarial flip as non-canonical. -/
theorem veselov_universal_rejection :
    ∀ (n : Nat) (a : SupportedAttack (canonicalWitness n)),
      plainCanonicalBits (singleBitFlip (canonicalWitness n).bits a.pos) = false :=
  fun n a => supportedAttack_global_rejection (canonicalWitness n) a

/-- Count of executable supported-attack candidate positions for a witness. -/
def supportedAttackCount (w : Witness n) : Nat :=
  (supportedAttackPositions w).length

/-- Count of candidate positions the executable checker rejects. -/
def supportedAttackRejectedCount (w : Witness n) : Nat :=
  ((supportedAttackPositions w).filter (supportedAttackRejected w)).length

/-- Every supported-attack candidate on any witness is rejected by the
executable checker. This is the pointwise form of Veselov's claim at the
decision-procedure level. -/
theorem allSupportedAttacksRejected (w : Witness n) :
    ∀ pos ∈ supportedAttackPositions w, supportedAttackRejected w pos = true := by
  intro pos hmem
  unfold supportedAttackPositions at hmem
  rw [List.mem_filter] at hmem
  obtain ⟨hmemDet, hnot⟩ := hmem
  have hzero : w.bits.getD pos false = false := by simpa using hnot
  exact supportedAttackRejected_eq_true w hmemDet hzero

/-- **Veselov's claim lifted from "detection rate = 1.0" (empirical
observation over a finite sample) to "rejected count equals trial count"
(universal theorem).**

This is the proof-theoretic anchor that makes the Rust harness's `1.0`
detection rate a necessary consequence, not a measurement: on every
witness, the executable checker rejects every supported-attack candidate
by theorem. -/
theorem veselov_detection_rate_eq_one (w : Witness n) :
    supportedAttackRejectedCount w = supportedAttackCount w := by
  unfold supportedAttackRejectedCount supportedAttackCount
  congr 1
  exact List.filter_eq_self.mpr (allSupportedAttacksRejected w)

/-- Specialization to canonical witnesses. -/
theorem veselov_detection_rate_eq_one_canonical (n : Nat) :
    supportedAttackRejectedCount (canonicalWitness n)
      = supportedAttackCount (canonicalWitness n) :=
  veselov_detection_rate_eq_one (canonicalWitness n)

/-- Aggregate form: on any finite batch of natural numbers, the rejected
count summed over their canonical witnesses equals the attack count. The
empirical "100% detection in every scenario" becomes a counting identity. -/
theorem veselov_detection_rate_eq_one_batch (ns : List Nat) :
    (ns.map (fun n => supportedAttackRejectedCount (canonicalWitness n))).sum
      = (ns.map (fun n => supportedAttackCount (canonicalWitness n))).sum := by
  apply congrArg List.sum
  apply List.map_congr_left
  intro n _
  exact veselov_detection_rate_eq_one (canonicalWitness n)

end HeytingLean.Bridge.Veselov.HybridZeckendorf
