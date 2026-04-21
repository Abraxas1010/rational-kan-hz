import HeytingLean.LoF.Nucleus
import HeytingLean.Epistemic.Occam

/-!
# Round-trip contracts

Abstract interface describing the encode/decode guarantees required in the roadmap.
-/

namespace HeytingLean
namespace Contracts

open HeytingLean.LoF

variable {α : Type u} [PrimaryAlgebra α]

/-- A round-trip contract packages encoding/decoding data for a given nucleus core. -/
structure RoundTrip (R : Reentry α) (β : Type v) where
  encode : R.Omega → β
  decode : β → R.Omega
  round : ∀ a, decode (encode a) = a

/-- The encoded representation is faithful up to the nucleus applied after decoding. -/
def interiorized (R : Reentry α) {β : Type v} (C : RoundTrip (R := R) β) :
    β → α :=
  fun b => R ((C.decode b : R.Omega) : α)

@[simp] lemma interiorized_id (R : Reentry α) {β} (C : RoundTrip (R := R) β) (a) :
    interiorized (R := R) C (C.encode a) = R (a : α) := by
  unfold interiorized
  simpa using
    congrArg (fun x : R.Omega => R (x : α)) (C.round a)

/-- Recover the Occam reduction of a decoded element through a round trip. -/
noncomputable def stageOccam (R : Reentry α)
    {β : Type v} (C : RoundTrip (R := R) β) (b : β) : β :=
  let core : α := ((C.decode b : R.Omega) : α)
  C.encode
    (Reentry.Omega.mk (R := R)
      (Epistemic.occam (R := R) core)
      (Epistemic.occam_idempotent (R := R) (a := core)))

lemma stageOccam_spec (R : Reentry α) {β : Type v}
    (C : RoundTrip (R := R) β) (b : β) :
    interiorized (R := R) C (stageOccam (R := R) (C := C) b) =
      Epistemic.occam (R := R)
        ((C.decode b : R.Omega) : α) := by
  unfold stageOccam interiorized
  simp [Epistemic.occam_idempotent, C.round]

@[simp] lemma stageOccam_encode (R : Reentry α) {β : Type v}
    (C : RoundTrip (R := R) β) (a : R.Omega) :
    stageOccam (R := R) (C := C) (C.encode a) =
      C.encode
        (Reentry.Omega.mk (R := R)
          (Epistemic.occam (R := R) (a : α))
          (Epistemic.occam_idempotent (R := R) (a := (a : α)))) := by
  classical
  unfold stageOccam
  have hround := C.round a
  simp [hround]

end Contracts
end HeytingLean
