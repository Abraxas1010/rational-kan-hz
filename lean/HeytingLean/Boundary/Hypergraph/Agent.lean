import Mathlib.Data.Finset.Basic

import HeytingLean.Boundary.Hypergraph.Core
import HeytingLean.Boundary.Kernel

set_option linter.dupNamespace false

namespace HeytingLean
namespace Boundary
namespace Hypergraph

/-- A hypergraph interaction-net port. Index `0` is reserved for the principal port. -/
structure HPort where
  index : Nat
  isPrincipal : Bool
  deriving DecidableEq, Repr

namespace HPort

def principal : HPort :=
  { index := 0, isPrincipal := true }

def aux (index : Nat) : HPort :=
  { index := index + 1, isPrincipal := false }

end HPort

/-- Boundary hypergraph agent kinds. Arity validity is tracked by `HAgent.Valid` so the
carrier remains proof-irrelevance-friendly for finite sets and executable tests. -/
inductive HAgent where
  | notw
  | andw (arity : Nat)
  | nandw (arity : Nat)
  | check (arity : Nat)
  | error (arity : Nat)
  deriving DecidableEq, Repr

namespace HAgent

/-- Semantic arity validity for hypergraph agents. -/
def Valid : HAgent -> Prop
  | .notw => True
  | .andw arity => 2 <= arity
  | .nandw arity => 2 <= arity
  | .check arity => 1 <= arity
  | .error _ => True

/-- Number of ports including the principal port. -/
def portCount : HAgent -> Nat
  | .notw => 1
  | .andw arity => arity + 1
  | .nandw arity => arity + 1
  | .check arity => arity + 1
  | .error arity => arity + 1

/-- Hypergraph agents always expose principal port index `0`. -/
def principalPorts (_agent : HAgent) : Finset Nat :=
  {0}

def IsError : HAgent -> Bool
  | .error _ => true
  | _ => false

def IsCheck : HAgent -> Bool
  | .check _ => true
  | _ => false

/-- Binary Boundary kernel embedding, defined only for arities where the old binary
carrier has an exact counterpart. -/
def toBoundaryKind? : HAgent -> Option Boundary.AgentKind
  | .notw => some .nu
  | .andw 2 => some .alpha
  | .nandw 2 => some .alpha
  | .check 2 => some .tau
  | .error 0 => some .eps
  | _ => none

def ofBoundaryKind : Boundary.AgentKind -> HAgent
  | .nu => .notw
  | .alpha => .andw 2
  | .eps => .error 0
  | .tau => .check 2

@[simp] theorem portCount_notw :
    portCount .notw = 1 := rfl

@[simp] theorem portCount_andw (arity : Nat) :
    portCount (.andw arity) = arity + 1 := rfl

@[simp] theorem principalPorts_eq_singleton (agent : HAgent) :
    principalPorts agent = {0} := rfl

@[simp] theorem toBoundaryKind_ofBoundaryKind (kind : Boundary.AgentKind) :
    toBoundaryKind? (ofBoundaryKind kind) = some kind := by
  cases kind <;> rfl

end HAgent

end Hypergraph
end Boundary
end HeytingLean
