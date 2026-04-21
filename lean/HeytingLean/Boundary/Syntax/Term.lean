import HeytingLean.Boundary.Kernel

namespace HeytingLean
namespace Boundary
namespace Syntax

structure EndpointDecl where
  agentId : Nat
  port : Port
  deriving DecidableEq, Repr, Inhabited

structure AgentDecl where
  id : Nat
  kind : AgentKind
  payload : LiteralPayload := .none
  deriving DecidableEq, Repr, Inhabited

structure WireDecl where
  left : EndpointDecl
  right : EndpointDecl
  deriving DecidableEq, Repr, Inhabited

structure SurfaceNet where
  mode : Mode := .bootstrap
  agents : List AgentDecl := []
  wires : List WireDecl := []
  sourceInterface : List EndpointDecl := []
  deriving DecidableEq, Repr, Inhabited

namespace EndpointDecl

def toEndpoint (ep : EndpointDecl) : Endpoint :=
  { agentId := ep.agentId, generation := 0, port := ep.port }

end EndpointDecl

namespace AgentDecl

def toAgent (decl : AgentDecl) : Nat × Agent :=
  (decl.id, { kind := decl.kind, generation := 0, payload := decl.payload })

end AgentDecl

namespace WireDecl

def toWire (wire : WireDecl) : Wire :=
  { left := wire.left.toEndpoint, right := wire.right.toEndpoint }

end WireDecl

end Syntax
end Boundary
end HeytingLean
