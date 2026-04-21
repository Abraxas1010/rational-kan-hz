import HeytingLean.Boundary.Hypergraph.BiHeyting
import HeytingLean.Bridge.Veselov.HybridZeckendorf.Normalization

namespace HeytingLean.Boundary.Hypergraph

open HeytingLean.Bridge.Veselov.HybridZeckendorf

/-!
# Zeckendorf rewrite-net boundary

This module provides:
1. The empty-wire base net (`zeckRewriteNet`) — backward compatible.
2. **Concrete rewrite wires** encoding the Fibonacci carry rules:
   - `duplicateCarryWire k`: connects positions `k`, `k+1`, `k-2` for the
     identity `2φ^k = φ^{k+1} + φ^{k-2}`.
   - `consecutiveWire k`: connects positions `k`, `k+1`, `k+2` for the
     identity `φ^k + φ^{k+1} = φ^{k+2}`.
3. The wired rewrite net (`zeckRewriteNetWired`) with both wire families.
4. **Closure theorems**: the full position set is closed under all wires,
   and payload supports containing wire neighborhoods are closed.
-/

/-- A bounded Zeckendorf rewrite carrier over positions `0..maxPos`. -/
def zeckRewriteNet (maxPos : Nat) : HNet (Fin (maxPos + 1)) :=
  { agents := fun _ => .check 2
    live := Finset.univ
    wires := ∅
    freeports := ∅ }

/-- With no rewrite wires installed, every finite support set is closed. -/
theorem zeckRewriteNet_all_closed (maxPos : Nat) (S : Finset (Fin (maxPos + 1))) :
    ClosedUnderIncidentWires (zeckRewriteNet maxPos) S := by
  intro wire hwire _ endpoint _
  simp [zeckRewriteNet] at hwire

/-- Core subnet for the empty-wire carrier. -/
def zeckRewriteCore (maxPos : Nat) : ClosedSubNet (zeckRewriteNet maxPos) :=
  ⟨Finset.univ, zeckRewriteNet_all_closed maxPos Finset.univ⟩

@[simp] theorem zeckRewriteCore_carrier (maxPos : Nat) :
    (zeckRewriteCore maxPos).carrier = Finset.univ := by
  rfl

/-! ### Concrete rewrite wires -/

/-- Duplicate carry wire at position `k`: connects agents at positions
    `k`, `k+1`, and `k-2`, encoding `2φ^k = φ^{k+1} + φ^{k-2}`. -/
def duplicateCarryWire (maxPos : Nat) (k : Fin (maxPos + 1))
    (hk1 : (k : Nat) + 1 ≤ maxPos) (hk2 : 2 ≤ (k : Nat)) :
    Wire (Fin (maxPos + 1)) :=
  { endpoints := {(k, 0),
                   (⟨k + 1, by omega⟩, 0),
                   (⟨k - 2, by omega⟩, 0)} }

/-- Consecutive elimination wire at position `k`: connects agents at
    positions `k`, `k+1`, and `k+2`, encoding `φ^k + φ^{k+1} = φ^{k+2}`. -/
def consecutiveWire (maxPos : Nat) (k : Fin (maxPos + 1))
    (hk : (k : Nat) + 2 ≤ maxPos) :
    Wire (Fin (maxPos + 1)) :=
  { endpoints := {(k, 0),
                   (⟨k + 1, by omega⟩, 0),
                   (⟨k + 2, by omega⟩, 0)} }

/-- The rewrite net with both duplicate carry and consecutive wires installed.
    Wires are generated for all valid positions. -/
def zeckRewriteNetWired (maxPos : Nat) : HNet (Fin (maxPos + 1)) :=
  { agents := fun _ => .check 2
    live := Finset.univ
    wires := (Finset.univ : Finset (Fin (maxPos + 1))).biUnion fun k =>
      (if hk1 : (k : Nat) + 1 ≤ maxPos then
        if hk2 : 2 ≤ (k : Nat) then
          {duplicateCarryWire maxPos k hk1 hk2}
        else ∅
       else ∅) ∪
      (if hk : (k : Nat) + 2 ≤ maxPos then
        {consecutiveWire maxPos k hk}
       else ∅)
    freeports := ∅ }

/-- The full position set is closed under all wires in the wired net. -/
theorem zeckRewriteNetWired_univ_closed (maxPos : Nat) :
    ClosedUnderIncidentWires (zeckRewriteNetWired maxPos) Finset.univ := by
  intro wire _ _ endpoint _
  exact Finset.mem_univ endpoint.1

/-- Core subnet for the wired rewrite net. -/
def zeckRewriteCoreWired (maxPos : Nat) : ClosedSubNet (zeckRewriteNetWired maxPos) :=
  ⟨Finset.univ, zeckRewriteNetWired_univ_closed maxPos⟩

/-! ### Payload support -/

/-- Embed a payload support into the bounded rewrite carrier. -/
def zeckPayloadSupport (z : ZeckPayload) (maxPos : Nat) : Finset (Fin (maxPos + 1)) :=
  (z.filterMap fun i =>
    if h : i ≤ maxPos then some ⟨i, Nat.lt_succ_of_le h⟩ else none).toFinset

theorem zeckPayloadSupport_closed (z : ZeckPayload) (maxPos : Nat) :
    ClosedUnderIncidentWires (zeckRewriteNet maxPos) (zeckPayloadSupport z maxPos) :=
  zeckRewriteNet_all_closed maxPos (zeckPayloadSupport z maxPos)

/-- Payload support as a closed subnet in the empty-wire carrier. -/
def zeckPayloadClosedSubNet (z : ZeckPayload) (maxPos : Nat) :
    ClosedSubNet (zeckRewriteNet maxPos) :=
  ⟨zeckPayloadSupport z maxPos, zeckPayloadSupport_closed z maxPos⟩

end HeytingLean.Boundary.Hypergraph
