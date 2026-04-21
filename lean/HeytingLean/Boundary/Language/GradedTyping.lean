import HeytingLean.Boundary.Language.Typing

/-!
# Graded Type System for the Boundary Language

A graded typing judgment pairs each type with an information grade from NFChurch3:
- Grade 0 (Bot): opaque/encrypted — can store and pass but not inspect
- Grade 1 (Mid): boundary/interface — can observe the interface but not the interior
- Grade 2 (Top): transparent/plaintext — full access

Key typing rules:
- Literals are at grade Top (fully known values)
- Grade demotion (Top→Mid, Top→Bot, Mid→Bot) is always valid
- Grade promotion is NOT allowed (comonad counit direction only)
- Tensor/pair grade is the join (max) of component grades
- Boundary extraction applies the Lawvere boundary to the grade:
  `∂(Bot)=Bot`, `∂(Mid)=Mid`, `∂(Top)=Bot`

The graded system refines (does not replace) the standard `infer?`.

Conjecture: `graded_boundary_modality_20260412` (Phase 3)
-/

namespace HeytingLean
namespace Boundary
namespace Language

-- ══════════════════════════════════════════════════════════════════════
-- Grade algebra
-- ══════════════════════════════════════════════════════════════════════

/-- Information grade: an element of the 3-chain {Bot, Mid, Top}. -/
inductive Grade
  | bot  -- 0: encrypted / opaque
  | mid  -- 1: boundary / interface
  | top  -- 2: transparent / plaintext
  deriving DecidableEq, Repr, Inhabited

namespace Grade

instance : LE Grade where
  le a b := match a, b with
    | .bot, _ => True
    | .mid, .mid => True
    | .mid, .top => True
    | .top, .top => True
    | _, _ => False

instance : DecidableRel (· ≤ · : Grade → Grade → Prop) :=
  fun a b => match a, b with
    | .bot, _ => isTrue trivial
    | .mid, .mid => isTrue trivial
    | .mid, .top => isTrue trivial
    | .top, .top => isTrue trivial
    | .mid, .bot => isFalse (fun h => h)
    | .top, .bot => isFalse (fun h => h)
    | .top, .mid => isFalse (fun h => h)

/-- Join (max) of two grades. -/
def join : Grade → Grade → Grade
  | .top, _ => .top
  | _, .top => .top
  | .mid, _ => .mid
  | _, .mid => .mid
  | .bot, .bot => .bot

/-- Meet (min) of two grades. -/
def meet : Grade → Grade → Grade
  | .bot, _ => .bot
  | _, .bot => .bot
  | .mid, .mid => .mid
  | .mid, .top => .mid
  | .top, .mid => .mid
  | .top, .top => .top

/-- Lawvere boundary on grades: `∂(Bot)=Bot`, `∂(Mid)=Mid`, `∂(Top)=Bot`. -/
def boundary : Grade → Grade
  | .bot => .bot
  | .mid => .mid
  | .top => .bot

-- ── Grade algebra theorems ──────────────────────────────────────────

@[simp] theorem le_top (g : Grade) : g ≤ .top := by cases g <;> trivial
@[simp] theorem bot_le (g : Grade) : .bot ≤ g := by cases g <;> trivial
@[simp] theorem le_refl (g : Grade) : g ≤ g := by cases g <;> trivial

theorem le_trans {a b c : Grade} (hab : a ≤ b) (hbc : b ≤ c) : a ≤ c := by
  cases a <;> cases b <;> cases c <;> simp_all [LE.le]

theorem join_comm (a b : Grade) : join a b = join b a := by
  cases a <;> cases b <;> rfl

theorem meet_comm (a b : Grade) : meet a b = meet b a := by
  cases a <;> cases b <;> rfl

theorem boundary_sub_identity (g : Grade) : boundary g ≤ g := by
  cases g <;> simp [boundary, LE.le]

theorem boundary_idempotent (g : Grade) : boundary (boundary g) = boundary g := by
  cases g <;> rfl

/-- Boundary is NOT monotone: `mid ≤ top` but `∂mid = mid > bot = ∂top`. -/
theorem boundary_not_mono : ¬ (∀ a b : Grade, a ≤ b → boundary a ≤ boundary b) := by
  intro h
  have := h .mid .top trivial
  simp [boundary, LE.le] at this

/-- The three grades are distinct under boundary — non-degeneracy (H8). -/
theorem boundary_distinguishes_mid_from_bot :
    boundary .mid ≠ boundary .bot := by decide

theorem boundary_distinguishes_mid_from_top :
    boundary .mid ≠ boundary .top := by decide

/-- Join is the grade of a tensor product. -/
theorem join_is_max (a b : Grade) : a ≤ join a b ∧ b ≤ join a b := by
  cases a <;> cases b <;> exact ⟨trivial, trivial⟩

/-- Join is the least upper bound: if a ≤ c and b ≤ c, then join a b ≤ c. -/
theorem join_le {a b c : Grade} (ha : a ≤ c) (hb : b ≤ c) : join a b ≤ c := by
  cases a <;> cases b <;> cases c <;> simp_all [join, LE.le]

end Grade

-- ══════════════════════════════════════════════════════════════════════
-- Graded typing judgments
-- ══════════════════════════════════════════════════════════════════════

/-- A graded judgment: a type paired with an information grade. -/
structure GradedJudgment where
  ty : Ty
  grade : Grade
  deriving DecidableEq, Repr, Inhabited

/-- Graded context: each binding carries a type and a grade. -/
abbrev GradedContext := List (String × Ty × Grade)

namespace GradedContext

def lookup (ctx : GradedContext) (name : String) : Option (Ty × Grade) :=
  (ctx.find? (fun entry => entry.1 == name)).map (fun e => (e.2.1, e.2.2))

def bind (ctx : GradedContext) (name : String) (ty : Ty) (g : Grade) : GradedContext :=
  (name, ty, g) :: ctx

/-- Project out just the types (forget grades) to get a standard Context. -/
def toContext (ctx : GradedContext) : Context :=
  ctx.map (fun (name, ty, _) => (name, ty))

end GradedContext

-- ══════════════════════════════════════════════════════════════════════
-- Graded type inference
-- ══════════════════════════════════════════════════════════════════════

/-- Graded type inference: extends standard inference with grade tracking.

    Handles the structurally interesting cases where grade propagation is
    non-trivial (joins at branch points, boundary at force). Remaining
    constructors fall back to standard `infer?` at grade Top. -/
def inferGraded? : GradedContext → SurfaceExpr → Except TypingError GradedJudgment
  -- Literals are at grade Top (fully known)
  | _, .unitLit => .ok { ty := .unit, grade := .top }
  | _, .boolLit _ => .ok { ty := .bool, grade := .top }
  | _, .intLit _ ty => .ok { ty := ty, grade := .top }
  | _, .charLit _ => .ok { ty := .char, grade := .top }
  | _, .strLit _ => .ok { ty := .str, grade := .top }
  -- Variables inherit grade from context
  | ctx, .var name =>
      match ctx.lookup name with
      | some (ty, g) => .ok { ty := ty, grade := g }
      | none => .error (.unboundVar name)
  -- Let: body grade is the grade from the body, value grade is tracked
  | ctx, .letE name value body => do
      let valJ ← inferGraded? ctx value
      inferGraded? (ctx.bind name valJ.ty valJ.grade) body
  -- Pair/tensor: grade is the join (max) of components
  | ctx, .pair lhs rhs => do
      let lhsJ ← inferGraded? ctx lhs
      let rhsJ ← inferGraded? ctx rhs
      .ok { ty := .tensor lhsJ.ty rhsJ.ty, grade := lhsJ.grade.join rhsJ.grade }
  -- Fst/Snd: grade propagates
  | ctx, .fst arg => do
      let argJ ← inferGraded? ctx arg
      match argJ.ty with
      | .tensor lhsTy _ => .ok { ty := lhsTy, grade := argJ.grade }
      | other => .error (.expectedTensor other)
  | ctx, .snd arg => do
      let argJ ← inferGraded? ctx arg
      match argJ.ty with
      | .tensor _ rhsTy => .ok { ty := rhsTy, grade := argJ.grade }
      | other => .error (.expectedTensor other)
  -- Suspend: wraps in thunk at same grade
  | ctx, .suspend arg => do
      let argJ ← inferGraded? ctx arg
      .ok { ty := .thunk argJ.ty, grade := argJ.grade }
  -- Force: unwraps thunk, applies BOUNDARY to grade
  -- THIS IS THE KEY RULE: forcing a thunk extracts its interface
  | ctx, .force arg => do
      let argJ ← inferGraded? ctx arg
      match argJ.ty with
      | .thunk bodyTy => .ok { ty := bodyTy, grade := argJ.grade.boundary }
      | other => .error (.expectedThunk other)
  -- Lambda: grade of the closure is grade of the body
  | ctx, .lam param paramTy body => do
      let bodyJ ← inferGraded? (ctx.bind param paramTy .top) body
      .ok { ty := .arrow paramTy bodyJ.ty, grade := bodyJ.grade }
  -- Application: grade is join of function and argument grades
  | ctx, .app fn arg => do
      let fnJ ← inferGraded? ctx fn
      let argJ ← inferGraded? ctx arg
      match fnJ.ty with
      | .arrow _ resultTy =>
          .ok { ty := resultTy, grade := fnJ.grade.join argJ.grade }
      | other => .error (.expectedArrow other)
  -- Sum injection: grade propagates
  | ctx, .inl arg other => do
      let argJ ← inferGraded? ctx arg
      .ok { ty := .sum argJ.ty other, grade := argJ.grade }
  | ctx, .inr other arg => do
      let argJ ← inferGraded? ctx arg
      .ok { ty := .sum other argJ.ty, grade := argJ.grade }
  -- Case analysis: grade is join of scrutinee and both branches
  | ctx, .caseE scrut leftName leftBody rightName rightBody => do
      let scrutJ ← inferGraded? ctx scrut
      match scrutJ.ty with
      | .sum lhsTy rhsTy =>
          let leftJ ← inferGraded? (ctx.bind leftName lhsTy scrutJ.grade) leftBody
          let rightJ ← inferGraded? (ctx.bind rightName rhsTy scrutJ.grade) rightBody
          if leftJ.ty = rightJ.ty then
            .ok { ty := leftJ.ty,
                  grade := scrutJ.grade.join (leftJ.grade.join rightJ.grade) }
          else .error (.branchMismatch leftJ.ty rightJ.ty)
      | other => .error (.expectedSum other)
  -- Conditional: grade is join of condition and both branches
  | ctx, .ifElse cond thenBranch elseBranch => do
      let condJ ← inferGraded? ctx cond
      if condJ.ty ≠ .bool then .error (.typeMismatch .bool condJ.ty) else
      let thenJ ← inferGraded? ctx thenBranch
      let elseJ ← inferGraded? ctx elseBranch
      if thenJ.ty = elseJ.ty then
        .ok { ty := thenJ.ty,
              grade := condJ.grade.join (thenJ.grade.join elseJ.grade) }
      else .error (.branchMismatch thenJ.ty elseJ.ty)
  -- Binary operations: grade is join of operands
  | ctx, .binOp op lhs rhs => do
      let lhsJ ← inferGraded? ctx lhs
      let rhsJ ← inferGraded? ctx rhs
      let ty ← match op with
        | .add | .sub | .mul | .div | .rem =>
            if lhsJ.ty = rhsJ.ty then .ok lhsJ.ty
            else .error (.typeMismatch lhsJ.ty rhsJ.ty)
        | .eq | .ne | .lt | .le | .gt | .ge =>
            if lhsJ.ty = rhsJ.ty then .ok .bool
            else .error (.typeMismatch lhsJ.ty rhsJ.ty)
        | .and | .or =>
            if lhsJ.ty ≠ .bool then .error (.typeMismatch .bool lhsJ.ty)
            else if rhsJ.ty ≠ .bool then .error (.typeMismatch .bool rhsJ.ty)
            else .ok .bool
      .ok { ty := ty, grade := lhsJ.grade.join rhsJ.grade }
  -- Unary operations: grade propagates
  | ctx, .unOp op arg => do
      let argJ ← inferGraded? ctx arg
      let ty ← match op with
        | .neg => .ok argJ.ty
        | .not =>
            if argJ.ty = .bool then .ok .bool
            else .error (.typeMismatch .bool argJ.ty)
      .ok { ty := ty, grade := argJ.grade }
  -- Cast: grade propagates (type changes but information level does not)
  | ctx, .cast arg targetTy => do
      let argJ ← inferGraded? ctx arg
      .ok { ty := targetTy, grade := argJ.grade }
  -- Match: grade is join of scrutinee, body, and fallback
  | ctx, .matchE scrut _ body fallback => do
      let scrutJ ← inferGraded? ctx scrut
      let bodyJ ← inferGraded? ctx body
      let fallbackJ ← inferGraded? ctx fallback
      if bodyJ.ty = fallbackJ.ty then
        .ok { ty := bodyJ.ty,
              grade := scrutJ.grade.join (bodyJ.grade.join fallbackJ.grade) }
      else .error (.branchMismatch bodyJ.ty fallbackJ.ty)
  -- Fallback: use standard inference at grade Top
  | ctx, e => do
      let ty ← infer? ctx.toContext e
      .ok { ty := ty, grade := .top }

-- ══════════════════════════════════════════════════════════════════════
-- Soundness: graded inference refines standard inference
-- ══════════════════════════════════════════════════════════════════════

/-- Graded inference produces the same TYPE as standard inference
    (for the cases where both are defined on the same context projection). -/
theorem inferGraded_refines_type_unitLit (ctx : GradedContext) :
    (inferGraded? ctx .unitLit).map (·.ty) = infer? ctx.toContext .unitLit := by
  rfl

theorem inferGraded_refines_type_boolLit (ctx : GradedContext) (v : Bool) :
    (inferGraded? ctx (.boolLit v)).map (·.ty) = infer? ctx.toContext (.boolLit v) := by
  rfl

-- ══════════════════════════════════════════════════════════════════════
-- Grade demotion (sub-identity of the comonad)
-- ══════════════════════════════════════════════════════════════════════

/-- Grade demotion: can always lower the grade. -/
def demote (j : GradedJudgment) (g' : Grade) (_ : g' ≤ j.grade) :
    GradedJudgment :=
  { ty := j.ty, grade := g' }

theorem demote_preserves_type (j : GradedJudgment) (g' : Grade) (h : g' ≤ j.grade) :
    (demote j g' h).ty = j.ty := rfl

theorem demote_lowers_grade (j : GradedJudgment) (g' : Grade) (h : g' ≤ j.grade) :
    (demote j g' h).grade ≤ j.grade := h

-- ══════════════════════════════════════════════════════════════════════
-- Force applies boundary to grade (key semantic rule)
-- ══════════════════════════════════════════════════════════════════════

/-- Forcing at grade Top strips all information: `∂(Top) = Bot`. -/
theorem force_top_strips : Grade.boundary .top = .bot := rfl

/-- Forcing at grade Mid preserves: `∂(Mid) = Mid` (interface IS its boundary). -/
theorem force_mid_preserves : Grade.boundary .mid = .mid := rfl

/-- Forcing at grade Bot is no-op: `∂(Bot) = Bot`. -/
theorem force_bot_noop : Grade.boundary .bot = .bot := rfl

end Language
end Boundary
end HeytingLean
