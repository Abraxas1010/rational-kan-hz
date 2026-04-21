import HeytingLean.Boundary.Language.Syntax

namespace HeytingLean
namespace Boundary
namespace Language

inductive TypingError
  | unboundVar (name : String)
  | duplicateDecl (name : String)
  | expectedArrow (actual : Ty)
  | expectedTensor (actual : Ty)
  | expectedSum (actual : Ty)
  | expectedThunk (actual : Ty)
  | expectedSearch (actual : Ty)
  | typeMismatch (expected actual : Ty)
  | branchMismatch (left right : Ty)
  deriving DecidableEq, Repr

private def ensureTy (expected actual : Ty) : Except TypingError Unit :=
  if actual = expected then .ok () else .error (.typeMismatch expected actual)

def castQuotedAtomBodyTy : Ty :=
  .sum .unit (.sum .bool (.sum .i64 (.sum .char .string)))

def castCompatible? (source target : Ty) : Bool :=
  if source = target then
    true
  else if target = Ty.quotedProgramTy then
    true
  else if target = Ty.quotedProgramBody then
    true
  else if target = castQuotedAtomBodyTy then
    true
  else
    match target with
    | .mu binder body => source = Ty.substGeneric binder (.mu binder body) body
    | _ =>
    match source, target with
    | .unit, .unit => true
    | .bool, .bool => true
    | .bool, .u8 | .bool, .u16 | .bool, .u32 | .bool, .u64
    | .bool, .u128 | .bool, .u256 | .bool, .usize
    | .bool, .i8 | .bool, .i16 | .bool, .i32 | .bool, .i64
    | .bool, .isize => true
    | .u8, .bool | .u16, .bool | .u32, .bool | .u64, .bool
    | .u128, .bool | .u256, .bool | .usize, .bool
    | .i8, .bool | .i16, .bool | .i32, .bool | .i64, .bool
    | .isize, .bool => true
    | .u8, .f32 | .u16, .f32 | .u32, .f32 | .u64, .f32
    | .u128, .f32 | .u256, .f32 | .usize, .f32
    | .i8, .f32 | .i16, .f32 | .i32, .f32 | .i64, .f32
    | .isize, .f32 => true
    | .u8, .f64 | .u16, .f64 | .u32, .f64 | .u64, .f64
    | .u128, .f64 | .u256, .f64 | .usize, .f64
    | .i8, .f64 | .i16, .f64 | .i32, .f64 | .i64, .f64
    | .isize, .f64 => true
    | .u8, .u8 | .u16, .u16 | .u32, .u32 | .u64, .u64
    | .u128, .u128 | .u256, .u256 | .usize, .usize
    | .i8, .i8 | .i16, .i16 | .i32, .i32 | .i64, .i64
    | .isize, .isize => true
    | .f32, .f32 | .f64, .f64 => true
    | .char, .char => true
    | .str, .str | .str, .string | .string, .string => true
    | .arrow p r, .arrow p' r' => p = p' && r = r'
    | .fnTrait ps ret, .fnTrait ps' ret' => ps = ps' && ret = ret'
    | _, _ => false

private def numericResult? (lhs rhs : Ty) : Except TypingError Ty :=
  if lhs = rhs then .ok lhs else .error (.typeMismatch lhs rhs)

private def comparisonResult? (lhs rhs : Ty) : Except TypingError Ty :=
  if lhs = rhs then .ok .bool else .error (.typeMismatch lhs rhs)

private def boolResult? (lhs rhs : Ty) : Except TypingError Ty := do
  ensureTy .bool lhs
  ensureTy .bool rhs
  .ok .bool

private def inferBinOp? (op : RustBinOp) (lhs rhs : Ty) : Except TypingError Ty :=
  match op with
  | .add | .sub | .mul | .div | .rem => numericResult? lhs rhs
  | .eq | .ne | .lt | .le | .gt | .ge => comparisonResult? lhs rhs
  | .and | .or => boolResult? lhs rhs

private def inferUnOp? (op : RustUnOp) (arg : Ty) : Except TypingError Ty :=
  match op with
  | .neg => .ok arg
  | .not => ensureTy .bool arg *> .ok .bool

private def expectSearch? : Ty → Except TypingError Ty
  | .search inner => .ok inner
  | other => .error (.expectedSearch other)

private def unfoldCallableTy? : Ty → Option Ty
  | .mu binder body => some (Ty.substGeneric binder (.mu binder body) body)
  | _ => none

def inferTraitMethodCallTy? (objTy argTy : Ty) : Except TypingError Ty :=
  match objTy with
  | .arrow paramTy ret =>
      if argTy = paramTy then .ok ret else .error (.typeMismatch paramTy argTy)
  | .fnTrait (.cons paramTy rest) ret =>
      if argTy = paramTy then
        match rest with
        | .nil => .ok ret
        | _ => .ok (.fnTrait rest ret)
      else .error (.typeMismatch paramTy argTy)
  | .mu _ _ =>
      match unfoldCallableTy? objTy with
      | some (.arrow paramTy ret) =>
          if argTy = paramTy then .ok ret else .error (.typeMismatch paramTy argTy)
      | some (.fnTrait (.cons paramTy rest) ret) =>
          if argTy = paramTy then
            match rest with
            | .nil => .ok ret
            | _ => .ok (.fnTrait rest ret)
          else .error (.typeMismatch paramTy argTy)
      | some other => .error (.expectedArrow other)
      | none => .error (.expectedArrow objTy)
  | other => .error (.expectedArrow other)

def infer? : Context → SurfaceExpr → Except TypingError Ty
  | ctx, .var name =>
      match Context.lookup ctx name with
      | some ty => .ok ty
      | none => .error (.unboundVar name)
  | _, .unitLit => .ok .unit
  | _, .boolLit _ => .ok .bool
  | ctx, .letE name value body =>
      infer? ctx value >>= fun valueTy => infer? (Context.bind ctx name valueTy) body
  | ctx, .lam param paramTy body =>
      infer? (Context.bind ctx param paramTy) body |>.map fun bodyTy => .arrow paramTy bodyTy
  | ctx, .app fn arg => do
      let fnTy ← infer? ctx fn
      let argTy ← infer? ctx arg
      match fnTy with
      | .arrow paramTy resultTy =>
          ensureTy paramTy argTy
          .ok resultTy
      | .mu _ _ =>
          match unfoldCallableTy? fnTy with
          | some unfolded =>
              match unfolded with
              | .arrow paramTy resultTy =>
                  ensureTy paramTy argTy
                  .ok resultTy
              | other => .error (.expectedArrow other)
          | none => .error (.expectedArrow fnTy)
      | other => .error (.expectedArrow other)
  | ctx, .pair lhs rhs => do
      let lhsTy ← infer? ctx lhs
      let rhsTy ← infer? ctx rhs
      .ok (.tensor lhsTy rhsTy)
  | ctx, .fst arg => do
      let argTy ← infer? ctx arg
      match argTy with
      | .tensor lhsTy _ => .ok lhsTy
      | other => .error (.expectedTensor other)
  | ctx, .snd arg => do
      let argTy ← infer? ctx arg
      match argTy with
      | .tensor _ rhsTy => .ok rhsTy
      | other => .error (.expectedTensor other)
  | ctx, .inl arg other =>
      infer? ctx arg |>.map fun lhsTy => .sum lhsTy other
  | ctx, .inr other arg =>
      infer? ctx arg |>.map fun rhsTy => .sum other rhsTy
  | ctx, .caseE scrut leftName leftBody rightName rightBody => do
      let scrutTy ← infer? ctx scrut
      match scrutTy with
      | .sum lhsTy rhsTy =>
          let leftTy ← infer? (Context.bind ctx leftName lhsTy) leftBody
          let rightTy ← infer? (Context.bind ctx rightName rhsTy) rightBody
          if leftTy = rightTy then .ok leftTy else .error (.branchMismatch leftTy rightTy)
      | other => .error (.expectedSum other)
  | ctx, .suspend arg =>
      infer? ctx arg |>.map .thunk
  | ctx, .force arg => do
      let argTy ← infer? ctx arg
      match argTy with
      | .thunk bodyTy => .ok bodyTy
      | other => .error (.expectedThunk other)
  | _, .failureE ty => .ok (.search ty)
  | ctx, .choiceE lhs rhs => do
      let lhsTy ← infer? ctx lhs
      let rhsTy ← infer? ctx rhs
      match lhsTy, rhsTy with
      | .search lhsInner, .search rhsInner =>
          if lhsInner = rhsInner then .ok (.search lhsInner) else .error (.branchMismatch lhsInner rhsInner)
      | .search _, other => .error (.expectedSearch other)
      | other, _ => .error (.expectedSearch other)
  | ctx, .eqnE lhs rhs => do
      let lhsTy ← infer? ctx lhs
      let rhsTy ← infer? ctx rhs
      if lhsTy = rhsTy then .ok (.search .unit) else .error (.typeMismatch lhsTy rhsTy)
  | ctx, .oneE body => do
      let bodyTy ← infer? ctx body
      let inner ← expectSearch? bodyTy
      .ok (.option inner)
  | ctx, .allE body => do
      let bodyTy ← infer? ctx body
      let inner ← expectSearch? bodyTy
      .ok (.vec inner)
  | _, .intLit _ ty => .ok ty
  | _, .floatLit _ ty => .ok ty
  | _, .charLit _ => .ok .char
  | _, .strLit _ => .ok .str
  | ctx, .binOp op lhs rhs => do
      let lhsTy ← infer? ctx lhs
      let rhsTy ← infer? ctx rhs
      inferBinOp? op lhsTy rhsTy
  | ctx, .unOp op arg =>
      infer? ctx arg >>= inferUnOp? op
  | ctx, .cast arg targetTy => do
      let sourceTy ← infer? ctx arg
      if castCompatible? sourceTy targetTy then .ok targetTy
      else .error (.typeMismatch targetTy sourceTy)
  | ctx, .ifElse cond thenBranch elseBranch => do
      let condTy ← infer? ctx cond
      ensureTy .bool condTy
      let thenTy ← infer? ctx thenBranch
      let elseTy ← infer? ctx elseBranch
      if thenTy = elseTy then .ok thenTy else .error (.branchMismatch thenTy elseTy)
  | ctx, .matchE scrut pattern body fallback => do
      let scrutTy ← infer? ctx scrut
      let bodyCtx ←
        match pattern with
        | .var name => .ok (Context.bind ctx name scrutTy)
        | .enumVariant _ variant (some name) =>
            match scrutTy with
            | .enum _ variants payloadTy =>
                match Ty.enumPayload? variants payloadTy variant with
                | some payloadTy => .ok (Context.bind ctx name payloadTy)
                | none => .error (.expectedSum scrutTy)
            | other => .error (.expectedSum other)
        | _ => .ok ctx
      let bodyTy ← infer? bodyCtx body
      let fallbackTy ← infer? ctx fallback
      if bodyTy = fallbackTy then .ok bodyTy else .error (.branchMismatch bodyTy fallbackTy)
  | ctx, .whileE cond body => do
      let condTy ← infer? ctx cond
      ensureTy .bool condTy
      let _ ← infer? ctx body
      .ok .unit
  | ctx, .loop body => infer? ctx body
  | ctx, .breakE value => infer? ctx value
  | _, .continueE => .ok .unit
  | ctx, .returnE value => infer? ctx value
  | ctx, .block stmt tail =>
      infer? ctx stmt *> infer? ctx tail
  | ctx, .structLit name field value =>
      infer? ctx value |>.map fun valueTy => .struct name (TyFields.singleton field valueTy)
  | ctx, .fieldAccess obj field => do
      match ← infer? ctx obj with
      | .struct _ fields =>
          match fields.lookup? field with
          | some fieldTy => .ok fieldTy
          | none => .error (.expectedTensor (.struct "" fields))
      | other => .error (.expectedTensor other)
  | ctx, .tupleLit first second => do
      let firstTy ← infer? ctx first
      let secondTy ← infer? ctx second
      .ok (.tensor firstTy secondTy)
  | ctx, .tupleIndex obj index => do
      let objTy ← infer? ctx obj
      match objTy, index with
      | .tensor lhsTy _, 0 => .ok lhsTy
      | .tensor _ rhsTy, _ => .ok rhsTy
      | other, _ => .error (.expectedTensor other)
  | ctx, .enumVariant enumName variant payload =>
      infer? ctx payload |>.map fun payloadTy => .enum enumName [variant] payloadTy
  | ctx, .arrayLit first second => do
      let firstTy ← infer? ctx first
      let secondTy ← infer? ctx second
      if firstTy = secondTy then .ok (.array firstTy 2) else .error (.typeMismatch firstTy secondTy)
  | ctx, .indexAccess obj index => do
      let objTy ← infer? ctx obj
      ensureTy .usize =<< infer? ctx index
      match objTy with
      | .array elem _ => .ok elem
      | .slice elem => .ok elem
      | .vec elem => .ok elem
      | other => .error (.expectedTensor other)
  | ctx, .borrow arg =>
      infer? ctx arg |>.map .ref
  | ctx, .borrowMut arg =>
      infer? ctx arg |>.map .mutRef
  | ctx, .deref arg => do
      let argTy ← infer? ctx arg
      match argTy with
      | .ref inner | .mutRef inner | .box inner | .rawPtr inner => .ok inner
      | other => .error (.expectedTensor other)
  | ctx, .moveE arg => infer? ctx arg
  | ctx, .dropE arg =>
      infer? ctx arg *> .ok .unit
  | ctx, .boxNew arg =>
      infer? ctx arg |>.map .box
  | ctx, .vecNew first second => do
      let firstTy ← infer? ctx first
      let secondTy ← infer? ctx second
      if firstTy = secondTy then .ok (.vec firstTy) else .error (.typeMismatch firstTy secondTy)
  | ctx, .genericInst fn tyArgs =>
      infer? ctx fn |>.map fun _ => .generic "inst" tyArgs.length
  | ctx, .traitMethodCall obj _ arg => do
      let objTy ← infer? ctx obj
      let argTy ← infer? ctx arg
      inferTraitMethodCallTy? objTy argTy
  | ctx, .implBlock _ _ _ body =>
      infer? ctx body *> .ok .unit
  | ctx, .closure params body =>
      let closureCtx := params.foldl (fun acc entry => Context.bind acc entry.1 entry.2) ctx
      infer? closureCtx body |>.map fun bodyTy => .fnTrait (TyList.ofList (params.map Prod.snd)) bodyTy
  | ctx, .closureCall fn arg => do
      let fnTy ← infer? ctx fn
      let argTy ← infer? ctx arg
      match fnTy with
      | .fnTrait (.cons paramTy rest) ret =>
          if argTy = paramTy then
            match rest with
            | .nil => .ok ret
            | _ => .ok (.fnTrait rest ret)
          else .error (.typeMismatch paramTy argTy)
      | .arrow paramTy ret =>
          if argTy = paramTy then .ok ret else .error (.typeMismatch paramTy argTy)
      | .mu _ _ =>
          match unfoldCallableTy? fnTy with
          | some (.fnTrait (.cons paramTy rest) ret) =>
              if argTy = paramTy then
                match rest with
                | .nil => .ok ret
                | _ => .ok (.fnTrait rest ret)
              else .error (.typeMismatch paramTy argTy)
          | some (.arrow paramTy ret) =>
              if argTy = paramTy then .ok ret else .error (.typeMismatch paramTy argTy)
          | some other => .error (.expectedArrow other)
          | none => .error (.expectedArrow fnTy)
      | other => .error (.expectedArrow other)
  | ctx, .forIn _ iterExpr body =>
      infer? ctx iterExpr *> infer? ctx body *> .ok .unit

def inferDecls? : Context → List Decl → Except TypingError Context
  | ctx, [] => .ok ctx
  | ctx, decl :: rest =>
      match Context.lookup ctx decl.name with
      | some _ => .error (.duplicateDecl decl.name)
      | none =>
          infer? ctx decl.body >>= fun declTy =>
            inferDecls? (Context.bind ctx decl.name declTy) rest

def inferProgram? (program : Program) : Except TypingError Ty := do
  let ctx ← inferDecls? [] program.decls
  infer? ctx program.main

/-- Algorithmic typing judgment for Boundary source expressions. -/
def HasType (ctx : Context) (expr : SurfaceExpr) (ty : Ty) : Prop :=
  infer? ctx expr = .ok ty

/-- Algorithmic declaration typing relation. -/
def DeclsWellTyped (ctx : Context) (decls : List Decl) (out : Context) : Prop :=
  inferDecls? ctx decls = .ok out

def WellFormedProgram (program : Program) (mainTy : Ty) : Prop :=
  inferProgram? program = .ok mainTy

theorem infer_sound {ctx expr ty} :
    infer? ctx expr = .ok ty →
    HasType ctx expr ty := by
  intro h
  exact h

theorem inferDecls_sound {ctx decls out} :
    inferDecls? ctx decls = .ok out →
    DeclsWellTyped ctx decls out := by
  intro h
  exact h

theorem inferProgram_sound {program ty} :
    inferProgram? program = .ok ty →
    WellFormedProgram program ty := by
  intro h
  exact h

end Language
end Boundary
end HeytingLean
