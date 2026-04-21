import HeytingLean.Boundary.Syntax.Term

namespace HeytingLean
namespace Boundary
namespace Language

/-- Rust binary operators represented in the Boundary language surface. -/
inductive RustBinOp
  | add | sub | mul | div | rem
  | eq | ne | lt | le | gt | ge
  | and | or
  deriving DecidableEq, Repr, Inhabited

namespace RustBinOp

def toRust : RustBinOp → String
  | .add => "+"
  | .sub => "-"
  | .mul => "*"
  | .div => "/"
  | .rem => "%"
  | .eq => "=="
  | .ne => "!="
  | .lt => "<"
  | .le => "<="
  | .gt => ">"
  | .ge => ">="
  | .and => "&&"
  | .or => "||"

instance : ToString RustBinOp where
  toString := toRust

end RustBinOp

/-- Rust unary operators represented in the Boundary language surface. -/
inductive RustUnOp
  | neg | not
  deriving DecidableEq, Repr, Inhabited

namespace RustUnOp

def toRust : RustUnOp → String
  | .neg => "-"
  | .not => "!"

instance : ToString RustUnOp where
  toString := toRust

end RustUnOp

/-- Rust match patterns carried by surface-level match syntax. -/
inductive Pattern
  | wildcard
  | var (name : String)
  | boolLit (value : Bool)
  | intLit (value : Int)
  | charLit (value : Char)
  | strLit (value : String)
  | enumVariant (enumName variant : String) (binder? : Option String)
  deriving DecidableEq, Repr, Inhabited

mutual
/-- Source-level Boundary types. This is intentionally richer than the kernel IR. -/
inductive Ty where
  | unit
  | bool
  | u8 | u16 | u32 | u64 | u128 | u256
  | i8 | i16 | i32 | i64
  | f32 | f64
  | usize | isize
  | address | bytes32
  | char
  | str
  | arrow (param result : Ty)
  | tensor (lhs rhs : Ty)
  | sum (lhs rhs : Ty)
  | thunk (body : Ty)
  | search (inner : Ty)
  | boundary (input output : Ty)
  | struct (name : String) (fields : TyFields)
  | enum (name : String) (variants : List String) (payload : Ty)
  | tuple (arity : Nat)
  | array (elem : Ty) (size : Nat)
  | slice (elem : Ty)
  | option (inner : Ty)
  | result (ok err : Ty)
  | ref (inner : Ty)
  | mutRef (inner : Ty)
  | box (inner : Ty)
  | vec (inner : Ty)
  | string
  | rawPtr (inner : Ty)
  | generic (name : String) (arity : Nat)
  | mu (binder : String) (body : Ty)
  | traitObject (traitName : String)
  | impl (traitName : String) (ty : Ty)
  | fnTrait (params : TyList) (ret : Ty)
  deriving DecidableEq, Repr, Inhabited

inductive TyList where
  | nil
  | cons (head : Ty) (tail : TyList)
  deriving DecidableEq, Repr, Inhabited

inductive TyFields where
  | nil
  | cons (name : String) (ty : Ty) (tail : TyFields)
  deriving DecidableEq, Repr, Inhabited
end

namespace TyList

def ofList : List Ty → TyList
  | [] => .nil
  | ty :: rest => .cons ty (ofList rest)

def toList : TyList → List Ty
  | .nil => []
  | .cons ty rest => ty :: toList rest

def length (tys : TyList) : Nat :=
  tys.toList.length

end TyList

namespace TyFields

def singleton (name : String) (ty : Ty) : TyFields :=
  .cons name ty .nil

def names : TyFields → List String
  | .nil => []
  | .cons name _ rest => name :: names rest

def lookup? : TyFields → String → Option Ty
  | .nil, _ => none
  | .cons name ty rest, field =>
      if field = name then some ty else lookup? rest field

def contains (fields : TyFields) (field : String) : Bool :=
  (fields.lookup? field).isSome

end TyFields

/-- Typing contexts are name-to-type lists with most-recent bindings first. -/
abbrev Context := List (String × Ty)

namespace Context

def lookup (ctx : Context) (name : String) : Option Ty :=
  (ctx.find? (fun entry => entry.1 = name)).map Prod.snd

def bind (ctx : Context) (name : String) (ty : Ty) : Context :=
  (name, ty) :: ctx

end Context

namespace Ty

mutual

/-- Substitute zero-arity generic occurrences under a recursive binder. -/
def substGeneric (binder : String) (replacement : Ty) : Ty → Ty
  | .unit => .unit
  | .bool => .bool
  | .u8 => .u8
  | .u16 => .u16
  | .u32 => .u32
  | .u64 => .u64
  | .u128 => .u128
  | .u256 => .u256
  | .i8 => .i8
  | .i16 => .i16
  | .i32 => .i32
  | .i64 => .i64
  | .f32 => .f32
  | .f64 => .f64
  | .usize => .usize
  | .isize => .isize
  | .address => .address
  | .bytes32 => .bytes32
  | .char => .char
  | .str => .str
  | .arrow param retTy => .arrow (substGeneric binder replacement param) (substGeneric binder replacement retTy)
  | .tensor lhs rhs => .tensor (substGeneric binder replacement lhs) (substGeneric binder replacement rhs)
  | .sum lhs rhs => .sum (substGeneric binder replacement lhs) (substGeneric binder replacement rhs)
  | .thunk body => .thunk (substGeneric binder replacement body)
  | .search inner => .search (substGeneric binder replacement inner)
  | .boundary input output =>
      .boundary (substGeneric binder replacement input) (substGeneric binder replacement output)
  | .struct name fields => .struct name (substGenericFields binder replacement fields)
  | .enum name variants payload => .enum name variants (substGeneric binder replacement payload)
  | .tuple arity => .tuple arity
  | .array elem size => .array (substGeneric binder replacement elem) size
  | .slice elem => .slice (substGeneric binder replacement elem)
  | .option inner => .option (substGeneric binder replacement inner)
  | .result ok err => .result (substGeneric binder replacement ok) (substGeneric binder replacement err)
  | .ref inner => .ref (substGeneric binder replacement inner)
  | .mutRef inner => .mutRef (substGeneric binder replacement inner)
  | .box inner => .box (substGeneric binder replacement inner)
  | .vec inner => .vec (substGeneric binder replacement inner)
  | .string => .string
  | .rawPtr inner => .rawPtr (substGeneric binder replacement inner)
  | .generic name 0 => if name = binder then replacement else .generic name 0
  | .generic name arity => .generic name arity
  | .mu innerBinder body =>
      if innerBinder = binder then
        .mu innerBinder body
      else
        .mu innerBinder (substGeneric binder replacement body)
  | .traitObject traitName => .traitObject traitName
  | .impl traitName ty => .impl traitName (substGeneric binder replacement ty)
  | .fnTrait params ret => .fnTrait (substGenericList binder replacement params) (substGeneric binder replacement ret)
termination_by ty => sizeOf ty
decreasing_by
  all_goals (try simp_wf; try omega)

def substGenericList (binder : String) (replacement : Ty) : TyList → TyList
  | .nil => .nil
  | .cons head tail =>
      .cons (substGeneric binder replacement head) (substGenericList binder replacement tail)
termination_by tys => sizeOf tys
decreasing_by
  all_goals (try simp_wf; try omega)

def substGenericFields (binder : String) (replacement : Ty) : TyFields → TyFields
  | .nil => .nil
  | .cons name ty tail =>
      .cons name (substGeneric binder replacement ty) (substGenericFields binder replacement tail)
termination_by fields => sizeOf fields
decreasing_by
  all_goals (try simp_wf; try omega)

end

/-- Lookup the uniform payload type attached to an enum variant. -/
def enumPayload? (variants : List String) (payloadTy : Ty) (variant : String) : Option Ty :=
  if variants.contains variant then some payloadTy else none

private def pairTy (lhs rhs : Ty) : Ty := .tensor lhs rhs

private def sum5 (first second third fourth fifth : Ty) : Ty :=
  .sum first (.sum second (.sum third (.sum fourth fifth)))

def quotedAtomTy : Ty := .generic "BoundaryQuoted.Atom" 0
def quotedTreeTy : Ty := .generic "BoundaryQuoted.Tree" 0

def unfoldMu? : Ty → Option Ty
  | .mu binder body => some (substGeneric binder (.mu binder body) body)
  | _ => none

def quotedProgramBody : Ty :=
  .sum quotedAtomTy (.tensor (.generic "Q" 0) (.generic "Q" 0))

def quotedProgramTy : Ty :=
  .mu "Q" quotedProgramBody

def quotedSchemaBody? : String → Option Ty
  | "BoundaryQuoted.Atom" =>
      some (sum5 .unit .bool .i64 .char .string)
  | "BoundaryQuoted.Tree" =>
      some (.sum quotedAtomTy (pairTy quotedTreeTy quotedTreeTy))
  | _ => none

def quotedSchemaType? : Ty → Option Ty
  | .generic name 0 => quotedSchemaBody? name
  | _ => none

end Ty

/-- User-facing Boundary source expressions, prior to desugaring. -/
inductive SurfaceExpr
  | var (name : String)
  | unitLit
  | boolLit (value : Bool)
  | letE (name : String) (value body : SurfaceExpr)
  | lam (param : String) (paramTy : Ty) (body : SurfaceExpr)
  | app (fn arg : SurfaceExpr)
  | pair (lhs rhs : SurfaceExpr)
  | fst (arg : SurfaceExpr)
  | snd (arg : SurfaceExpr)
  | inl (arg : SurfaceExpr) (other : Ty)
  | inr (other : Ty) (arg : SurfaceExpr)
  | caseE
      (scrut : SurfaceExpr)
      (leftName : String) (leftBody : SurfaceExpr)
      (rightName : String) (rightBody : SurfaceExpr)
  | suspend (arg : SurfaceExpr)
  | force (arg : SurfaceExpr)
  | failureE (ty : Ty)
  | choiceE (lhs rhs : SurfaceExpr)
  | eqnE (lhs rhs : SurfaceExpr)
  | oneE (body : SurfaceExpr)
  | allE (body : SurfaceExpr)
  | intLit (value : Int) (ty : Ty)
  | floatLit (value : String) (ty : Ty)
  | charLit (value : Char)
  | strLit (value : String)
  | binOp (op : RustBinOp) (lhs rhs : SurfaceExpr)
  | unOp (op : RustUnOp) (arg : SurfaceExpr)
  | cast (arg : SurfaceExpr) (targetTy : Ty)
  | ifElse (cond thenBranch elseBranch : SurfaceExpr)
  | matchE (scrut : SurfaceExpr) (pattern : Pattern) (body fallback : SurfaceExpr)
  | whileE (cond body : SurfaceExpr)
  | loop (body : SurfaceExpr)
  | breakE (value : SurfaceExpr)
  | continueE
  | returnE (value : SurfaceExpr)
  | block (stmt tail : SurfaceExpr)
  | structLit (name field : String) (value : SurfaceExpr)
  | fieldAccess (obj : SurfaceExpr) (field : String)
  | tupleLit (first second : SurfaceExpr)
  | tupleIndex (obj : SurfaceExpr) (index : Nat)
  | enumVariant (enumName variant : String) (payload : SurfaceExpr)
  | arrayLit (first second : SurfaceExpr)
  | indexAccess (obj index : SurfaceExpr)
  | borrow (arg : SurfaceExpr)
  | borrowMut (arg : SurfaceExpr)
  | deref (arg : SurfaceExpr)
  | moveE (arg : SurfaceExpr)
  | dropE (arg : SurfaceExpr)
  | boxNew (arg : SurfaceExpr)
  | vecNew (first second : SurfaceExpr)
  | genericInst (fn : SurfaceExpr) (tyArgs : List Ty)
  | traitMethodCall (obj : SurfaceExpr) (method : String) (arg : SurfaceExpr)
  | implBlock (traitName : String) (forTy : Ty) (method : String) (body : SurfaceExpr)
  | closure (params : List (String × Ty)) (body : SurfaceExpr)
  | closureCall (fn arg : SurfaceExpr)
  /-- For-in loop: `forIn iterVar iterExpr body` desugars to a loop+match+next pattern.
  The iterator expression must produce a value whose `.next()` trait method returns
  `Option<iterVar_type>`. -/
  | forIn (iterVar : String) (iterExpr body : SurfaceExpr)
  deriving DecidableEq, Repr, Inhabited

/-- Core expressions after local source-level sugar has been removed. -/
inductive CoreExpr
  | var (name : String)
  | unitLit
  | boolLit (value : Bool)
  | lam (param : String) (paramTy : Ty) (body : CoreExpr)
  | app (fn arg : CoreExpr)
  | pair (lhs rhs : CoreExpr)
  | fst (arg : CoreExpr)
  | snd (arg : CoreExpr)
  | inl (arg : CoreExpr) (other : Ty)
  | inr (other : Ty) (arg : CoreExpr)
  | caseE
      (scrut : CoreExpr)
      (leftName : String) (leftBody : CoreExpr)
      (rightName : String) (rightBody : CoreExpr)
  | suspend (arg : CoreExpr)
  | force (arg : CoreExpr)
  | intLit (value : Int) (ty : Ty)
  | floatLit (value : String) (ty : Ty)
  | charLit (value : Char)
  | strLit (value : String)
  | binOp (op : RustBinOp) (lhs rhs : CoreExpr)
  | unOp (op : RustUnOp) (arg : CoreExpr)
  | cast (arg : CoreExpr) (targetTy : Ty)
  | ifElse (cond thenBranch elseBranch : CoreExpr)
  | matchE (scrut : CoreExpr) (pattern : Pattern) (body fallback : CoreExpr)
  | whileE (cond body : CoreExpr)
  | loop (body : CoreExpr)
  | breakE (value : CoreExpr)
  | continueE
  | returnE (value : CoreExpr)
  | block (stmt tail : CoreExpr)
  | structLit (name field : String) (value : CoreExpr)
  | fieldAccess (obj : CoreExpr) (field : String)
  | tupleLit (first second : CoreExpr)
  | tupleIndex (obj : CoreExpr) (index : Nat)
  | enumVariant (enumName variant : String) (payload : CoreExpr)
  | arrayLit (first second : CoreExpr)
  | indexAccess (obj index : CoreExpr)
  | borrow (arg : CoreExpr)
  | borrowMut (arg : CoreExpr)
  | deref (arg : CoreExpr)
  | moveE (arg : CoreExpr)
  | dropE (arg : CoreExpr)
  | boxNew (arg : CoreExpr)
  | vecNew (first second : CoreExpr)
  | genericInst (fn : CoreExpr) (tyArgs : List Ty)
  | traitMethodCall (obj : CoreExpr) (method : String) (arg : CoreExpr)
  | implBlock (traitName : String) (forTy : Ty) (method : String) (body : CoreExpr)
  | closure (params : List (String × Ty)) (body : CoreExpr)
  | closureCall (fn arg : CoreExpr)
  deriving DecidableEq, Repr, Inhabited

/-- Language-level metadata used by higher-order readback.

The kernel `LiteralPayload` intentionally remains source-language-free; this
sidecar records exactly the source body data needed to reconstruct closures and
thunks after lowering, plus the callable node ids needed to recover the
currently connected application slice. The `agentId` is the compiled kernel id
assigned to the lowered root. -/
inductive HigherOrderReadbackPayload
  | closure (agentId : Nat) (params : List (String × Ty)) (body : CoreExpr)
  | thunk (agentId : Nat) (body : CoreExpr)
  | closureCall (agentId : Nat)
  | traitMethodCall (agentId : Nat) (method : String)
  | resolvedTraitMethodCall
      (agentId : Nat) (traitName : String) (forTy : Ty) (method : String)
      (body : CoreExpr)
  deriving DecidableEq, Repr, Inhabited

namespace HigherOrderReadbackPayload

def agentId : HigherOrderReadbackPayload → Nat
  | .closure id _ _ => id
  | .thunk id _ => id
  | .closureCall id => id
  | .traitMethodCall id _ => id
  | .resolvedTraitMethodCall id _ _ _ _ => id

end HigherOrderReadbackPayload

structure Decl where
  name : String
  body : SurfaceExpr
  deriving DecidableEq, Repr, Inhabited

structure CoreDecl where
  name : String
  body : CoreExpr
  deriving DecidableEq, Repr, Inhabited

structure Program where
  decls : List Decl := []
  main : SurfaceExpr
  deriving DecidableEq, Repr, Inhabited

structure CoreProgram where
  decls : List CoreDecl := []
  main : CoreExpr
  deriving DecidableEq, Repr, Inhabited

/-- The kernel-facing `SurfaceNet` remains a verified IR, not the source language. -/
abbrev KernelSurface := HeytingLean.Boundary.Syntax.SurfaceNet

end Language
end Boundary
end HeytingLean
