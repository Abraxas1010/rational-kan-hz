import HeytingLean.Boundary.Kernel

namespace HeytingLean
namespace Boundary

open Net

inductive RuleName : Type
| annihilate
| commute
| erase
| check
| checkTauTau
| associate
| select
  deriving DecidableEq, Repr, Inhabited

inductive BlockReason : Type
| checkFailed (pair : ActivePair)
  deriving DecidableEq, Repr

inductive StepResult : Type
| halted
| blocked (reason : BlockReason)
| progress (rule : RuleName) (net : Net)
  deriving DecidableEq, Repr

abbrev BoundaryMatch := AgentRef → AgentRef → Bool

private def principalEp (ref : AgentRef) : Endpoint :=
  Endpoint.principal ref

private def portEp (ref : AgentRef) (port : Port) : Endpoint :=
  Endpoint.ofRef ref port

private def partnerAt? (net : Net) (ref : AgentRef) (port : Port) : Option Endpoint :=
  net.partner? (portEp ref port)

private def reconnectToPrincipal (net : Net) (target : Option Endpoint) (ref : AgentRef) : Net :=
  net.reconnectOptional target (some (principalEp ref))

private def deletePair (net : Net) (lhs rhs : AgentRef) : Net :=
  net.removeAgents [lhs.id, rhs.id]

private def applyAnnihilate (net : Net) (lhs rhs : AgentRef) : Net :=
  let aAux := partnerAt? net lhs .aux1
  let bAux := partnerAt? net rhs .aux1
  let aWit := partnerAt? net lhs .witness
  let bWit := partnerAt? net rhs .witness
  let net' := deletePair net lhs rhs
  let net'' := net'.reconnectOptional aAux bAux
  let net''' := net''.eraseBranch aWit
  net'''.eraseBranch bWit

private def applyCommute (net : Net) (nuRef alphaRef : AgentRef) : Net :=
  let aAux := partnerAt? net nuRef .aux1
  let aWit := partnerAt? net nuRef .witness
  let b1 := partnerAt? net alphaRef .aux1
  let b2 := partnerAt? net alphaRef .aux2
  let base := deletePair net nuRef alphaRef
  let (n1, net1) := base.addFreshAgent .nu
  let (n2, net2) := net1.addFreshAgent .nu
  let (x1, net3) := net2.addFreshAgent .alpha
  let (x2, net4) := net3.addFreshAgent .alpha
  let net5 := reconnectToPrincipal net4 b1 n1
  let net6 := reconnectToPrincipal net5 b2 n2
  let net7 := reconnectToPrincipal net6 aAux x1
  let net8 := reconnectToPrincipal net7 aWit x2
  let net9 := net8.connectUnchecked (portEp x1 .aux1) (portEp n1 .aux1)
  let net10 := net9.connectUnchecked (portEp x1 .aux2) (portEp n2 .aux1)
  let net11 := net10.connectUnchecked (portEp x2 .aux1) (portEp n1 .witness)
  net11.connectUnchecked (portEp x2 .aux2) (portEp n2 .witness)

private def applyErase (net : Net) (epsRef other : AgentRef) : Net :=
  match other.kind with
  | .eps => deletePair net epsRef other
  | .nu =>
      let aux := partnerAt? net other .aux1
      let wit := partnerAt? net other .witness
      let base := deletePair net epsRef other
      let base' := base.eraseBranch aux
      base'.eraseBranch wit
  | .alpha =>
      let a1 := partnerAt? net other .aux1
      let a2 := partnerAt? net other .aux2
      let base := deletePair net epsRef other
      let base' := base.eraseBranch a1
      base'.eraseBranch a2
  | .tau =>
      let termSide := partnerAt? net other .aux1
      let typeSide := partnerAt? net other .aux2
      let base := deletePair net epsRef other
      let base' := base.eraseBranch termSide
      base'.eraseBranch typeSide

private def applyCheckPass (net : Net) (tauRef other : AgentRef) : Net :=
  let termSide := partnerAt? net tauRef .aux1
  let typeSide := partnerAt? net tauRef .aux2
  let base := net.removeAgents [tauRef.id]
  let base' := base.reconnectOptional termSide (some (principalEp other))
  base'.eraseBranch typeSide

private def applyCheckTauTau (net : Net) (lhs rhs : AgentRef) : Net :=
  let termSide := partnerAt? net lhs .aux1
  let typeSide := partnerAt? net lhs .aux2
  let base := net.removeAgents [lhs.id]
  let base' := base.reconnectOptional termSide (some (principalEp rhs))
  base'.eraseBranch typeSide

private def applyAssociate (net : Net) (lhs rhs : AgentRef) : Net :=
  let a1 := partnerAt? net lhs .aux1
  let a2 := partnerAt? net lhs .aux2
  let b1 := partnerAt? net rhs .aux1
  let b2 := partnerAt? net rhs .aux2
  let base := deletePair net lhs rhs
  let base' := base.reconnectOptional a1 b1
  base'.reconnectOptional a2 b2

def promoteSource (net : Net) (target : Option Endpoint) : Net :=
  match target with
  | none => net
  | some ep => { net with sourceInterface := ep :: net.sourceInterface.erase ep }

def selectBoolTauNext (net : Net) (boolRef tauRef : AgentRef) (cond : Bool) : Net :=
  let thenSide := partnerAt? net tauRef .aux1
  let elseSide := partnerAt? net tauRef .aux2
  let selected := if cond then thenSide else elseSide
  let base := net.removeAgents [boolRef.id, tauRef.id]
  promoteSource base selected

/-- Public normal form for the selector contraction. This keeps downstream
metatheory from depending on the private `partnerAt?` helper generated inside
this module. -/
theorem selectBoolTauNext_eq_promoteSource
    (net : Net) (boolRef tauRef : AgentRef) (cond : Bool) :
    selectBoolTauNext net boolRef tauRef cond =
      promoteSource (net.removeAgents [boolRef.id, tauRef.id])
        (if cond then
          net.partner? (Endpoint.ofRef tauRef .aux1)
        else
          net.partner? (Endpoint.ofRef tauRef .aux2)) := by
  cases cond <;> rfl

/-- Public reference next-net for the `ν >< ν` annihilation rule. -/
def annihilateNext (net : Net) (lhs rhs : AgentRef) : Net :=
  let aAux := net.partner? (Endpoint.ofRef lhs .aux1)
  let bAux := net.partner? (Endpoint.ofRef rhs .aux1)
  let aWit := net.partner? (Endpoint.ofRef lhs .witness)
  let bWit := net.partner? (Endpoint.ofRef rhs .witness)
  let net' := net.removeAgents [lhs.id, rhs.id]
  let net'' := net'.reconnectOptional aAux bAux
  let net''' := net''.eraseBranch aWit
  net'''.eraseBranch bWit

/-- Public reference next-net for the `ν >< α` commute rule. -/
def commuteNext (net : Net) (nuRef alphaRef : AgentRef) : Net :=
  let aAux := net.partner? (Endpoint.ofRef nuRef .aux1)
  let aWit := net.partner? (Endpoint.ofRef nuRef .witness)
  let b1 := net.partner? (Endpoint.ofRef alphaRef .aux1)
  let b2 := net.partner? (Endpoint.ofRef alphaRef .aux2)
  let base := net.removeAgents [nuRef.id, alphaRef.id]
  let (n1, net1) := base.addFreshAgent .nu
  let (n2, net2) := net1.addFreshAgent .nu
  let (x1, net3) := net2.addFreshAgent .alpha
  let (x2, net4) := net3.addFreshAgent .alpha
  let net5 := net4.reconnectOptional b1 (some (Endpoint.principal n1))
  let net6 := net5.reconnectOptional b2 (some (Endpoint.principal n2))
  let net7 := net6.reconnectOptional aAux (some (Endpoint.principal x1))
  let net8 := net7.reconnectOptional aWit (some (Endpoint.principal x2))
  let net9 := net8.connectUnchecked (Endpoint.ofRef x1 .aux1) (Endpoint.ofRef n1 .aux1)
  let net10 := net9.connectUnchecked (Endpoint.ofRef x1 .aux2) (Endpoint.ofRef n2 .aux1)
  let net11 := net10.connectUnchecked (Endpoint.ofRef x2 .aux1) (Endpoint.ofRef n1 .witness)
  net11.connectUnchecked (Endpoint.ofRef x2 .aux2) (Endpoint.ofRef n2 .witness)

/-- Public reference next-net for the `ε` erasure rule. -/
def eraseNext (net : Net) (epsRef other : AgentRef) : Net :=
  match other.kind with
  | .eps => net.removeAgents [epsRef.id, other.id]
  | .nu =>
      let aux := net.partner? (Endpoint.ofRef other .aux1)
      let wit := net.partner? (Endpoint.ofRef other .witness)
      let base := net.removeAgents [epsRef.id, other.id]
      let base' := base.eraseBranch aux
      base'.eraseBranch wit
  | .alpha =>
      let a1 := net.partner? (Endpoint.ofRef other .aux1)
      let a2 := net.partner? (Endpoint.ofRef other .aux2)
      let base := net.removeAgents [epsRef.id, other.id]
      let base' := base.eraseBranch a1
      base'.eraseBranch a2
  | .tau =>
      let termSide := net.partner? (Endpoint.ofRef other .aux1)
      let typeSide := net.partner? (Endpoint.ofRef other .aux2)
      let base := net.removeAgents [epsRef.id, other.id]
      let base' := base.eraseBranch termSide
      base'.eraseBranch typeSide

/-- Public reference next-net for a successful `τ` check rule. -/
def checkPassNext (net : Net) (tauRef other : AgentRef) : Net :=
  let termSide := net.partner? (Endpoint.ofRef tauRef .aux1)
  let typeSide := net.partner? (Endpoint.ofRef tauRef .aux2)
  let base := net.removeAgents [tauRef.id]
  let base' := base.reconnectOptional termSide (some (Endpoint.principal other))
  base'.eraseBranch typeSide

/-- Public reference next-net for the `τ >< τ` check rule. -/
def checkTauTauNext (net : Net) (lhs rhs : AgentRef) : Net :=
  let termSide := net.partner? (Endpoint.ofRef lhs .aux1)
  let typeSide := net.partner? (Endpoint.ofRef lhs .aux2)
  let base := net.removeAgents [lhs.id]
  let base' := base.reconnectOptional termSide (some (Endpoint.principal rhs))
  base'.eraseBranch typeSide

/-- Public reference next-net for the `α >< α` associate rule. -/
def associateNext (net : Net) (lhs rhs : AgentRef) : Net :=
  let a1 := net.partner? (Endpoint.ofRef lhs .aux1)
  let a2 := net.partner? (Endpoint.ofRef lhs .aux2)
  let b1 := net.partner? (Endpoint.ofRef rhs .aux1)
  let b2 := net.partner? (Endpoint.ofRef rhs .aux2)
  let base := net.removeAgents [lhs.id, rhs.id]
  let base' := base.reconnectOptional a1 b1
  base'.reconnectOptional a2 b2

/-- Public reference next-net for a Boolean selector active pair.

The branch machine is intentionally narrow: it fires only for an `ε` Boolean payload
cut against a `τ.ifElse` selector. The Boolean is the condition carried by the active
pair; the selector's `aux1` branch is chosen on `true`, and `aux2` on `false`.
The selected branch is promoted to the observable source interface. The unchosen
branch is intentionally left unreachable by this local rule; decision-tree
compilation must use dormant branch carriers, or a separate garbage-collection
rule must erase unreachable subnets explicitly. -/
def selectNext (net : Net) (lhs rhs : AgentRef) : Net :=
  match lhs.kind, lhs.payload, rhs.kind, rhs.payload with
  | .eps, .bool cond, .tau, .ifElse => selectBoolTauNext net lhs rhs cond
  | .tau, .ifElse, .eps, .bool cond => selectBoolTauNext net rhs lhs cond
  | _, _, _, _ => net

/-- A selector cut whose chosen branch points at an already-compiled tail reduces
exactly to that tail, provided the tail does not reuse the two selector ids. -/
theorem selectBoolTauNext_spine
    (offset : Nat) (cond : Bool) (tail : Net) (tailRoot : AgentRef)
    (hMode : tail.mode = .bootstrap)
    (hAgents : ∀ entry ∈ tail.agents, entry.1 ≠ offset + 1 ∧ entry.1 ≠ offset + 2)
    (hWires : ∀ wire ∈ tail.wires,
      wire.left.agentId ≠ offset + 1 ∧ wire.left.agentId ≠ offset + 2 ∧
        wire.right.agentId ≠ offset + 1 ∧ wire.right.agentId ≠ offset + 2)
    (hSource : tail.sourceInterface = [Endpoint.principal tailRoot]) :
    selectBoolTauNext
      { agents :=
          (offset + 1, { kind := .eps, generation := 0, payload := .bool cond }) ::
          (offset + 2, { kind := .tau, generation := 0, payload := .ifElse }) ::
          tail.agents
        wires :=
          { left := { agentId := offset + 1, generation := 0, port := .principal },
            right := { agentId := offset + 2, generation := 0, port := .principal } } ::
          (if cond then
            { left := { agentId := offset + 2, generation := 0, port := .aux1 },
              right := Endpoint.principal tailRoot }
          else
            { left := { agentId := offset + 2, generation := 0, port := .aux2 },
              right := Endpoint.principal tailRoot }) ::
          tail.wires
        sourceInterface :=
          [{ agentId := offset + 2, generation := 0, port := .principal }] }
      { id := offset + 1, generation := 0, kind := .eps, payload := .bool cond }
      { id := offset + 2, generation := 0, kind := .tau, payload := .ifElse }
      cond = tail := by
  have hAgentsFilter :
      List.filter (fun entry => !decide (entry.1 = offset + 1) && !decide (entry.1 = offset + 2))
          tail.agents =
        tail.agents := by
    apply List.filter_eq_self.2
    intro entry hmem
    have h := hAgents entry hmem
    simp [h.1, h.2]
  have hWiresFilter :
      List.filter
          (fun wire =>
            !decide (wire.left.agentId = offset + 1) &&
              !decide (wire.left.agentId = offset + 2) &&
                (!decide (wire.right.agentId = offset + 1) &&
                  !decide (wire.right.agentId = offset + 2)))
          tail.wires =
        tail.wires := by
    apply List.filter_eq_self.2
    intro wire hmem
    have h := hWires wire hmem
    simp [h.1, h.2.1, h.2.2.1, h.2.2.2]
  cases tail with
  | mk mode agents wires sourceInterface =>
      simp at hMode hSource
      subst mode
      subst sourceInterface
      cases cond <;>
        simp [selectBoolTauNext, partnerAt?, portEp, promoteSource, Net.partner?,
          Net.removeAgents, Wire.usesEndpoint, Endpoint.ofRef, Endpoint.principal,
          hAgentsFilter, hWiresFilter]

def isSelectPair (lhs rhs : AgentRef) : Bool :=
  match lhs.kind, lhs.payload, rhs.kind, rhs.payload with
  | .eps, .bool _, .tau, .ifElse => true
  | .tau, .ifElse, .eps, .bool _ => true
  | _, _, _, _ => false

/-- Total unordered dispatch table for the kernel active pairs. -/
def dispatchRule (lhs rhs : AgentRef) : RuleName :=
  if isSelectPair lhs rhs then
    .select
  else
    match lhs.kind, rhs.kind with
    | .nu, .nu => .annihilate
    | .nu, .alpha => .commute
    | .eps, _ => .erase
    | .tau, .tau => .checkTauTau
    | .tau, _ => .check
    | .alpha, .alpha => .associate
    | _, _ => .erase

/-- Contract a specific active pair, independent of the reference scheduler order. -/
def contractPair (net : Net) (pair : ActivePair)
    (boundaryMatch : BoundaryMatch := fun _ _ => true) : StepResult :=
  match dispatchRule pair.left pair.right with
  | .annihilate => .progress .annihilate (annihilateNext net pair.left pair.right)
  | .commute => .progress .commute (commuteNext net pair.left pair.right)
  | .erase => .progress .erase (eraseNext net pair.left pair.right)
  | .check =>
      if net.mode = .typed && !(boundaryMatch pair.left pair.right) then
        .blocked (.checkFailed pair)
      else
        .progress .check (checkPassNext net pair.left pair.right)
  | .checkTauTau =>
      .progress .checkTauTau (checkTauTauNext net pair.left pair.right)
  | .associate => .progress .associate (associateNext net pair.left pair.right)
  | .select => .progress .select (selectNext net pair.left pair.right)

/-- Deterministic one-step Boundary reduction. -/
def step? (net : Net) (boundaryMatch : BoundaryMatch := fun _ _ => true) : StepResult :=
  match net.nextActivePair? with
  | none => .halted
  | some pair => contractPair net pair boundaryMatch

@[simp] theorem step_deterministic (net : Net) (boundaryMatch : BoundaryMatch) :
    step? net boundaryMatch = step? net boundaryMatch := rfl

@[simp] theorem dispatch_eps_left (rhs : AgentRef) :
    dispatchRule { id := 0, generation := 0, kind := .eps } rhs = .erase := by
  cases rhs.kind <;> rfl

@[simp] theorem dispatch_tau_tau :
    dispatchRule { id := 0, generation := 0, kind := .tau }
      { id := 1, generation := 0, kind := .tau } = .checkTauTau := by
  rfl

@[simp] theorem dispatch_nu_alpha :
    dispatchRule { id := 0, generation := 0, kind := .nu }
      { id := 1, generation := 0, kind := .alpha } = .commute := by
  rfl

@[simp] theorem dispatch_eps_bool_tau_ifElse (cond : Bool) :
    dispatchRule { id := 0, generation := 0, kind := .eps, payload := .bool cond }
      { id := 1, generation := 0, kind := .tau, payload := .ifElse } = .select := by
  cases cond <;> rfl

@[simp] theorem dispatch_tau_ifElse_eps_bool (cond : Bool) :
    dispatchRule { id := 1, generation := 0, kind := .tau, payload := .ifElse }
      { id := 0, generation := 0, kind := .eps, payload := .bool cond } = .select := by
  cases cond <;> rfl

end Boundary
end HeytingLean
