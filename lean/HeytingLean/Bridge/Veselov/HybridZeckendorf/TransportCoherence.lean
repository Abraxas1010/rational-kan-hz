import HeytingLean.Bridge.Veselov.HybridZeckendorf.DragStability
import HeytingLean.Ontology.SGI26StatementBridges

/-!
# Bridge.Veselov.HybridZeckendorf.TransportCoherence

Round-trip transport coherence theorems for selected driver→knot paths.
These are namespace-local re-exports of the SGI26 bridge theorems, and are
intentionally redundant with `Ontology.SGI26StatementBridges`.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Ontology.SGI26StatementBridges

/-- Local alias used by the PM blueprint.
The current SGI26 bridge surface exports this node as
`driver_plural_generation_step`. -/
abbrev driver_anchoredpregame_over_nothing : Prop := driver_plural_generation_step

theorem transport_coherence_baseStabilizes_split
    (_x _y : HeytingLean.Topology.Knot.LaurentPoly)
    : transportCoherent driver_basestabilizes_split knot →
      glueEq (drag_update driver_basestabilizes_split knot) knot :=
  bridge_sgi26_transport_coherence_e863599482a5 _x _y

theorem transport_coherence_eulerPhaseLaw
    (_x _y : HeytingLean.Topology.Knot.LaurentPoly)
    : transportCoherent driver_eulerphaselaw knot →
      glueEq (drag_update driver_eulerphaselaw knot) knot :=
  bridge_sgi26_transport_coherence_34715284626c _x _y

theorem transport_coherence_anchoredPregame
    (_x _y : HeytingLean.Topology.Knot.LaurentPoly)
    : transportCoherent driver_plural_generation_step knot →
      glueEq (drag_update driver_plural_generation_step knot) knot :=
  bridge_sgi26_transport_coherence_513e12852107 _x _y

end HeytingLean.Bridge.Veselov.HybridZeckendorf
