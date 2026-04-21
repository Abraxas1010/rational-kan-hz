import HeytingLean.Boundary.Hypergraph.Net
import HeytingLean.Boundary.Hypergraph.Hypermatrix
import HeytingLean.Boundary.Step

set_option linter.dupNamespace false

namespace HeytingLean
namespace Boundary
namespace Hypergraph

inductive HRuleName where
  | notw_notw_annihilate
  | notw_andw_distribute
  | check_polyadic
  | andw_andw_commute
  | error_inert
  | no_rule
  deriving DecidableEq, Repr

/-- A motif match is intentionally structural: a match carries the matched live agents
and touched hyperwires instead of reducing an n-ary wire to binary pairs. -/
structure MotifMatch (AgentId : Type*) [DecidableEq AgentId] where
  agents : Finset AgentId
  wires : Finset (Wire AgentId)

structure ActiveMotif (AgentId : Type*) [DecidableEq AgentId] (M : Motif AgentId)
    (N : HNet AgentId) where
  match_ : MotifMatch AgentId
  agents_live : match_.agents ⊆ N.live
  principal_engaged : Prop

structure RewriteRule (AgentId : Type*) [DecidableEq AgentId] where
  name : HRuleName
  pattern : Motif AgentId
  replacement : HNet AgentId
  interface_map : Prop

/-- Hypergraph dispatch over agent kinds. ERROR is inert and never an active redex. -/
def dispatchHRule : HAgent -> HAgent -> HRuleName
  | .error _, _ => .error_inert
  | _, .error _ => .error_inert
  | .notw, .notw => .notw_notw_annihilate
  | .notw, .andw _ => .notw_andw_distribute
  | .andw _, .notw => .notw_andw_distribute
  | .check _, _ => .check_polyadic
  | _, .check _ => .check_polyadic
  | .andw _, .andw _ => .andw_andw_commute
  | _, _ => .no_rule

def notw_notw_annihilate {AgentId : Type*} [DecidableEq AgentId]
    (pattern : Motif AgentId) (replacement : HNet AgentId) :
    RewriteRule AgentId :=
  { name := .notw_notw_annihilate
    pattern := pattern
    replacement := replacement
    interface_map := True }

def notw_andw_distribute {AgentId : Type*} [DecidableEq AgentId]
    (pattern : Motif AgentId) (replacement : HNet AgentId) :
    RewriteRule AgentId :=
  { name := .notw_andw_distribute
    pattern := pattern
    replacement := replacement
    interface_map := True }

def check_polyadic {AgentId : Type*} [DecidableEq AgentId]
    (pattern : Motif AgentId) (replacement : HNet AgentId) :
    RewriteRule AgentId :=
  { name := .check_polyadic
    pattern := pattern
    replacement := replacement
    interface_map := True }

def andw_andw_commute {AgentId : Type*} [DecidableEq AgentId]
    (pattern : Motif AgentId) (replacement : HNet AgentId) :
    RewriteRule AgentId :=
  { name := .andw_andw_commute
    pattern := pattern
    replacement := replacement
    interface_map := True }

/-- Exact binary bridge dispatch. The bridge deliberately delegates to the established
binary kernel after `HAgent.toBoundaryKind?` succeeds, instead of reimplementing it. -/
def binarySpecializationDispatch (lhs rhs : Boundary.AgentRef) : Boundary.RuleName :=
  Boundary.dispatchRule lhs rhs

/-- Exact binary bridge step. This is the stable boundary for Phase C; later runtime
code must reproduce this result exactly for arity-2 fixtures. -/
def binarySpecializationStep (net : Boundary.Net)
    (boundaryMatch : Boundary.BoundaryMatch := fun _ _ => true) : Boundary.StepResult :=
  Boundary.step? net boundaryMatch

theorem binary_specialization_dispatch_eq_boundary_dispatch
    (lhs rhs : Boundary.AgentRef) :
    binarySpecializationDispatch lhs rhs = Boundary.dispatchRule lhs rhs :=
  rfl

theorem binary_specialization_step_eq_boundary_step
    (net : Boundary.Net) (boundaryMatch : Boundary.BoundaryMatch) :
    binarySpecializationStep net boundaryMatch = Boundary.step? net boundaryMatch :=
  rfl

theorem error_inert_left (arity : Nat) (agent : HAgent) :
    dispatchHRule (.error arity) agent = .error_inert := by
  cases agent <;> rfl

theorem error_inert_right (arity : Nat) (agent : HAgent) :
    dispatchHRule agent (.error arity) = .error_inert := by
  cases agent <;> rfl

theorem notw_cycles_decompose_pairwise :
    dispatchHRule .notw .notw = .notw_notw_annihilate :=
  rfl

end Hypergraph
end Boundary
end HeytingLean
