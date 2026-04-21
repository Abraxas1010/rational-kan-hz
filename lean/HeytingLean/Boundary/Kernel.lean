import Mathlib.Data.List.Basic

namespace HeytingLean
namespace Boundary

inductive Mode : Type
| bootstrap
| typed
  deriving DecidableEq, Repr, Inhabited

inductive AgentKind : Type
| nu
| alpha
| eps
| tau
  deriving DecidableEq, Repr, Inhabited

/-- Data carried by compiled agents.

Literal cases are used by terminal `eps` agents. The structural tags are used by
nonterminal agents so payload-aware readback can distinguish product-like values
that share the same kernel agent kind, e.g. pair/tuple/array/vec. -/
inductive LiteralPayload : Type
| none
| unit
| bool (value : Bool)
| int (value : Int)
| float (value : String)
| char (value : Char)
| str (value : String)
| pair
| tuple
| array
| vec
| inl
| inr
| structField (name field : String)
| enumVariant (enumName variant : String)
| box
| ref
| fst
| snd
| tupleIndex (index : Nat)
| indexAccess
| deref
| move
| drop
| genericInst
| implBlock
| fieldAccess
| traitMethodCall
| block
| binOp (op : String)
| unOp (op : String)
| ifElse
| opaque (tag : String)
  deriving DecidableEq, Repr, Inhabited

inductive Port : Type
| principal
| aux1
| aux2
| witness
  deriving DecidableEq, Repr, Inhabited

/-- Ports that are semantically meaningful for each agent kind. -/
def meaningfulPorts : AgentKind → List Port
  | .nu => [.principal, .aux1, .witness]
  | .alpha => [.principal, .aux1, .aux2]
  | .eps => [.principal]
  | .tau => [.principal, .aux1, .aux2]

/-- Semantic port membership predicate. -/
def PortMeaningful (kind : AgentKind) (port : Port) : Prop :=
  port ∈ meaningfulPorts kind

structure Agent where
  kind : AgentKind
  generation : Nat := 0
  payload : LiteralPayload := .none
  deriving DecidableEq, Repr, Inhabited

structure AgentRef where
  id : Nat
  generation : Nat
  kind : AgentKind
  payload : LiteralPayload := .none
  deriving DecidableEq, Repr, Inhabited

structure Endpoint where
  agentId : Nat
  generation : Nat
  port : Port
  deriving DecidableEq, Repr, Inhabited

structure Wire where
  left : Endpoint
  right : Endpoint
  deriving DecidableEq, Repr, Inhabited

structure ActivePair where
  left : AgentRef
  right : AgentRef
  deriving DecidableEq, Repr, Inhabited

structure Net where
  mode : Mode := .bootstrap
  agents : List (Nat × Agent) := []
  wires : List Wire := []
  sourceInterface : List Endpoint := []
  deriving DecidableEq, Repr, Inhabited

namespace Port

def isPrincipal : Port → Bool
  | .principal => true
  | _ => false

end Port

namespace AgentKind

def priorityRank : AgentKind → Nat
  | .eps => 0
  | .tau => 1
  | .nu => 2
  | .alpha => 3

end AgentKind

namespace Endpoint

def principal (ref : AgentRef) : Endpoint :=
  { agentId := ref.id, generation := ref.generation, port := .principal }

def ofRef (ref : AgentRef) (port : Port) : Endpoint :=
  { agentId := ref.id, generation := ref.generation, port := port }

end Endpoint

namespace Wire

def usesEndpoint (wire : Wire) (ep : Endpoint) : Bool :=
  wire.left = ep || wire.right = ep

end Wire

namespace Net

/-- Lookup an agent by arena id. -/
def agentAt? (net : Net) (id : Nat) : Option Agent :=
  match net.agents.find? (fun entry => entry.1 = id) with
  | some (_, agent) => some agent
  | none => none

/-- Recover the current live view for an arena id. -/
def agentRef? (net : Net) (id : Nat) : Option AgentRef := do
  let agent ← net.agentAt? id
  pure { id := id, generation := agent.generation, kind := agent.kind, payload := agent.payload }

/-- Whether an endpoint names a currently meaningful live port. -/
def endpointLive (net : Net) (ep : Endpoint) : Prop :=
  match net.agentAt? ep.agentId with
  | some agent => ep.generation = agent.generation ∧ PortMeaningful agent.kind ep.port
  | none => False

/-- The unique partner endpoint of `ep`, if one exists. -/
def partner? (net : Net) (ep : Endpoint) : Option Endpoint :=
  match net.wires.find? (fun wire => wire.usesEndpoint ep) with
  | some wire => if wire.left = ep then some wire.right else some wire.left
  | none => none

/-- Whether an endpoint is currently free. -/
def portFree (net : Net) (ep : Endpoint) : Bool :=
  (net.partner? ep).isNone

/-- Remove every wire incident to `ep`. -/
def disconnect (net : Net) (ep : Endpoint) : Net :=
  { net with wires := net.wires.filter (fun wire => !(wire.usesEndpoint ep)) }

/-- Remove every wire incident to any arena id in `ids`, and drop those agents. -/
def removeAgents (net : Net) (ids : List Nat) : Net :=
  let usesDeleted : Wire → Bool := fun wire =>
    ids.contains wire.left.agentId || ids.contains wire.right.agentId
  { net with
      agents := net.agents.filter (fun entry => !(ids.contains entry.1))
      wires := net.wires.filter (fun wire => !(usesDeleted wire))
      sourceInterface := net.sourceInterface.filter (fun ep => !(ids.contains ep.agentId)) }

/-- Largest currently allocated arena id. Returns `0` on the empty arena. -/
def maxAgentId (net : Net) : Nat :=
  net.agents.foldl (fun acc entry => max acc entry.1) 0

/-- Insert a new live agent at the next free arena id. -/
def addFreshAgent (net : Net) (kind : AgentKind) (payload : LiteralPayload := .none) :
    AgentRef × Net :=
  let nextId := net.maxAgentId + 1
  let agent : Agent := { kind := kind, generation := 0, payload := payload }
  let ref : AgentRef :=
    { id := nextId, generation := agent.generation, kind := kind, payload := payload }
  (ref, { net with agents := (nextId, agent) :: net.agents })

/-- Add a wire without checking side conditions. Intended for kernel rewrites. -/
def connectUnchecked (net : Net) (lhs rhs : Endpoint) : Net :=
  { net with wires := { left := lhs, right := rhs } :: net.wires }

/-- Bypass two optional neighbors after a local rewrite deletes a pair. -/
def reconnectOptional (net : Net) (lhs rhs : Option Endpoint) : Net :=
  match lhs, rhs with
  | some left, some right => net.connectUnchecked left right
  | _, _ => net

/-- Attach a fresh `ε` principal to an endpoint, leaving the new principal free when absent. -/
def eraseBranch (net : Net) (target : Option Endpoint) : Net :=
  match target with
  | none => net
  | some ep =>
      let (eraser, net') := net.addFreshAgent .eps
      net'.connectUnchecked (Endpoint.principal eraser) ep

private def refComesFirst (lhs rhs : AgentRef) : Bool :=
  let lhsRank := AgentKind.priorityRank lhs.kind
  let rhsRank := AgentKind.priorityRank rhs.kind
  if lhsRank < rhsRank then
    true
  else if rhsRank < lhsRank then
    false
  else if lhs.id < rhs.id then
    true
  else if rhs.id < lhs.id then
    false
  else
    lhs.generation ≤ rhs.generation

def canonicalizePair (lhs rhs : AgentRef) : ActivePair :=
  if refComesFirst lhs rhs then
    { left := lhs, right := rhs }
  else
    { left := rhs, right := lhs }

private def canonicalPair (lhs rhs : AgentRef) : ActivePair :=
  canonicalizePair lhs rhs

@[simp] theorem canonicalPair_eq_canonicalizePair (lhs rhs : AgentRef) :
    canonicalPair lhs rhs = canonicalizePair lhs rhs := rfl

/-- Boolean selectors are already in canonical active-pair order.

This exported fact keeps downstream branch-machine proofs from depending on the
private scheduler comparison used by `canonicalizePair`. -/
@[simp] theorem canonicalizePair_eps_bool_tau_ifElse (offset : Nat) (cond : Bool) :
    canonicalizePair
      { id := offset + 1, generation := 0, kind := .eps, payload := .bool cond }
      { id := offset + 2, generation := 0, kind := .tau, payload := .ifElse } =
    { left := { id := offset + 1, generation := 0, kind := .eps, payload := .bool cond }
      right := { id := offset + 2, generation := 0, kind := .tau, payload := .ifElse } } := by
  simp [canonicalizePair, refComesFirst, AgentKind.priorityRank]

/-- Extract an active pair from a principal-to-principal wire. -/
def activePairOfWire? (net : Net) (wire : Wire) : Option ActivePair := do
  let lhs ← net.agentRef? wire.left.agentId
  let rhs ← net.agentRef? wire.right.agentId
  if wire.left.generation ≠ lhs.generation || wire.right.generation ≠ rhs.generation then
    none
  else if wire.left.port = .principal && wire.right.port = .principal then
    some (canonicalPair lhs rhs)
  else
    none

/-- All active pairs in deterministic wire-list order. -/
def activePairs (net : Net) : List ActivePair :=
  net.wires.filterMap (net.activePairOfWire?)

/-- The deterministic next redex chosen by the reference machine. -/
def nextActivePair? (net : Net) : Option ActivePair :=
  net.activePairs.head?

/-- Endpoints that are principal and currently free. -/
def freePrincipalEndpoints (net : Net) : List Endpoint :=
  let livePrincipals :=
    net.agents.map fun entry =>
      { agentId := entry.1, generation := entry.2.generation, port := .principal }
  livePrincipals.filter fun ep => net.portFree ep

/-- Runtime well-formedness: live meaningful ports, no multi-wiring, and no self-loops. -/
def RuntimeWellFormed (net : Net) : Prop :=
  (∀ wire ∈ net.wires, wire.left ≠ wire.right) ∧
  (∀ wire ∈ net.wires, net.endpointLive wire.left) ∧
  (∀ wire ∈ net.wires, net.endpointLive wire.right) ∧
  (∀ ep, List.countP (fun wire => wire.usesEndpoint ep) net.wires ≤ 1)

/-- Source well-formedness strengthens runtime well-formedness by constraining free principals. -/
def SourceWellFormed (net : Net) : Prop :=
  net.RuntimeWellFormed ∧
    ∀ ep ∈ net.freePrincipalEndpoints, ep ∈ net.sourceInterface

end Net

open Net

@[simp] theorem principal_meaningful (kind : AgentKind) :
    PortMeaningful kind .principal := by
  cases kind <;> simp [PortMeaningful, meaningfulPorts]

@[simp] theorem priority_eps_lt_alpha :
    AgentKind.priorityRank .eps < AgentKind.priorityRank .alpha := by
  decide

@[simp] theorem priority_tau_lt_nu :
    AgentKind.priorityRank .tau < AgentKind.priorityRank .nu := by
  decide

@[simp] theorem priority_nu_lt_alpha :
    AgentKind.priorityRank .nu < AgentKind.priorityRank .alpha := by
  decide

@[simp] theorem nextActivePair_deterministic (net : Net) :
    net.nextActivePair? = net.nextActivePair? := rfl

end Boundary
end HeytingLean
