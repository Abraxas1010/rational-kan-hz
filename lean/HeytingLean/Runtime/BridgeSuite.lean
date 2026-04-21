import HeytingLean.Contracts.Examples

namespace HeytingLean
namespace Runtime

open HeytingLean.Contracts
open HeytingLean.Contracts.Examples
open HeytingLean.LoF

universe u

variable {α : Type u} [PrimaryAlgebra α]

/-- Runtime bridge flags mirroring the documented rollout (`Docs/Semantics.md`). -/
def bridgeFlags : BridgeFlags := BridgeFlags.runtime

/-- Legacy bridge flags retained for regression and comparison scenarios. -/
def legacyBridgeFlags : BridgeFlags := BridgeFlags.legacy

/-- Bridge suite exposed to runtime components, assembled via `Contracts.Examples.selectSuite`. -/
noncomputable def bridgeSuite (R : Reentry α) :
    BridgeSuite (α := α) (R := R) :=
  selectSuite (α := α) (R := R) bridgeFlags

/-- Legacy bridge suite mirroring the pre-rollout carriers. -/
noncomputable def legacyBridgeSuite (R : Reentry α) :
    BridgeSuite (α := α) (R := R) :=
  selectSuite (α := α) (R := R) legacyBridgeFlags

end Runtime
end HeytingLean
