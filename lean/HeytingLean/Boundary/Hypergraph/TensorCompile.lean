import HeytingLean.Boundary.Hypergraph.Hypermatrix

set_option linter.dupNamespace false

namespace HeytingLean
namespace Boundary
namespace Hypergraph

/-- Tensor operation descriptors for the hypergraph lane. These are executable compile
shapes, not correctness theorems for graph rewriting. -/
inductive TensorOp where
  | matmul (m n k : Nat)
  | contract3 (dims : Fin 3 -> Nat)
  | contractN (arity : Nat) (dims : Fin arity -> Nat)

namespace TensorOp

/-- Output tensor order for each descriptor. -/
def outputOrder : TensorOp -> Nat
  | .matmul _ _ _ => 2
  | .contract3 _ => 3
  | .contractN arity _ => arity

/-- Input tensor orders expected by the descriptor. -/
def inputOrders : TensorOp -> List Nat
  | .matmul _ _ _ => [2, 2]
  | .contract3 _ => [3]
  | .contractN arity _ => [arity]

/-- Output dimensions retained by the descriptor. Matrix multiplication exposes the
visible row/column dimensions and treats the third dimension as internal. -/
def outputDim (op : TensorOp) : Fin op.outputOrder -> Nat :=
  match op with
  | .matmul m n _ => fun i => if i.val = 0 then m else n
  | .contract3 dims => dims
  | .contractN _ dims => dims

end TensorOp

/-- A compile plan packages the tensor descriptor with the orders it exposes to the
hypermatrix runtime. -/
structure TensorCompilePlan where
  op : TensorOp
  inputOrders : List Nat
  outputOrder : Nat

/-- Record an operation-level compile plan without asserting semantic correctness. -/
def compileTensorOp (op : TensorOp) : TensorCompilePlan :=
  { op := op
    inputOrders := op.inputOrders
    outputOrder := op.outputOrder }

/-- Correctness obligations are named records for downstream proof work. They are not
implemented as `Prop := True` placeholders. -/
structure TensorCorrectnessObligation where
  op : TensorOp
  sourceSemantics : String
  targetSemantics : String

/-- The binary V-product compile shape is the matrix-multiplication descriptor. -/
def vProductPlan (m n k : Nat) : TensorCompilePlan :=
  compileTensorOp (.matmul m n k)

@[simp] theorem compileTensorOp_outputOrder (op : TensorOp) :
    (compileTensorOp op).outputOrder = op.outputOrder :=
  rfl

@[simp] theorem vProductPlan_outputOrder (m n k : Nat) :
    (vProductPlan m n k).outputOrder = 2 :=
  rfl

end Hypergraph
end Boundary
end HeytingLean
