import HeytingLean.Bridge.Veselov.HybridZeckendorf.HybridNumber
import HeytingLean.Bridge.Veselov.HybridZeckendorf.Normalization
import HeytingLean.Ontology.SGI26StatementBridges

/-!
# Bridge.Veselov.HybridZeckendorf.DragStability

Anchor-invariant drag theorems for the SGI26 sheaf bridge vocabulary.
These are explicit re-exports of the canonical SGI26 bridge statements, to keep
the HybridZeckendorf namespace aligned with the shared ontology surface.
They are intentionally redundant with `Ontology.SGI26StatementBridges`.
-/

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

open HeytingLean.Ontology.SGI26StatementBridges

theorem anchor_invariant_euler_reentry
    {α : Type} [HeytingLean.LoF.PrimaryAlgebra α]
    (_R : HeytingLean.LoF.Reentry α)
    : invariant_anchor (drag_update euler r_nucleus_re_entry_nucleus_euler_formula_exactness) =
      invariant_anchor euler :=
  bridge_sgi26_anchor_invariant_drag_c9e07f6f54f5 (α := α) _R

theorem anchor_invariant_sheaf_glue_witness
    : invariant_anchor (drag_update sheaf r_nucleus_re_entry_nucleus_sheaf_glue_witness) =
      invariant_anchor sheaf :=
  bridge_sgi26_anchor_invariant_drag_133368319f1f

theorem anchor_invariant_primordial_sheaf
    : invariant_anchor (drag_update driver_primordial_sheaf_glue_witness sheaf) =
      invariant_anchor driver_primordial_sheaf_glue_witness :=
  bridge_sgi26_anchor_invariant_drag_a1d600bb0883

end HeytingLean.Bridge.Veselov.HybridZeckendorf
