import HeytingLean.Boundary.Hypergraph.ZeckRewriteNet
import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFrac

namespace HeytingLean.Boundary.Hypergraph

open HeytingLean.Bridge.Veselov.HybridZeckendorf

/-! # Fractional extension of the Zeckendorf rewrite net -/

/-- Last combined position for integer and fractional lanes. -/
def combinedMaxPos (maxInt maxFrac : Nat) : Nat :=
  maxInt + maxFrac + 1

/-- Telescoping wire for adjacent fractional positions in the combined carrier. -/
def fracTelescopingWire (maxInt maxFrac : Nat) (k : Fin (maxFrac + 1))
    (hk : (k : Nat) + 1 ≤ maxFrac) :
    Wire (Fin (combinedMaxPos maxInt maxFrac + 1)) :=
  { endpoints :=
      { (⟨maxInt + 1 + k, by
            simp [combinedMaxPos]
            omega⟩, 0),
        (⟨maxInt + 1 + k + 1, by
            simp [combinedMaxPos]
            omega⟩, 0) } }

/-- Fractional telescoping net over the combined position carrier. -/
def zeckCombinedNet (maxInt maxFrac : Nat) :
    HNet (Fin (combinedMaxPos maxInt maxFrac + 1)) :=
  { agents := fun _ => .check 2
    live := Finset.univ
    wires := (Finset.univ : Finset (Fin (maxFrac + 1))).biUnion fun k =>
      if hk : (k : Nat) + 1 ≤ maxFrac then
        {fracTelescopingWire maxInt maxFrac k hk}
      else ∅
    freeports := ∅ }

/-- The full position set is closed under all combined fractional wires. -/
theorem zeckCombinedNet_univ_closed (maxInt maxFrac : Nat) :
    ClosedUnderIncidentWires (zeckCombinedNet maxInt maxFrac) Finset.univ := by
  intro wire _ _ endpoint _
  exact Finset.mem_univ endpoint.1

/-- Closed subnet for the combined fractional net. -/
def zeckCombinedCore (maxInt maxFrac : Nat) :
    ClosedSubNet (zeckCombinedNet maxInt maxFrac) :=
  ⟨Finset.univ, zeckCombinedNet_univ_closed maxInt maxFrac⟩

end HeytingLean.Boundary.Hypergraph
