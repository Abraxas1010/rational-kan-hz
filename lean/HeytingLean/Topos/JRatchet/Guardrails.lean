import Mathlib.Order.Lattice
import Mathlib.Order.Nucleus

/-!
# JRatchet guardrails: why the frame/distributivity hypothesis matters

`WIP/j_ratchet.md` records an important “no overclaim” rule:

> A nucleus gives a Heyting fixed-point core **only** when the ambient truth values form a
> frame (complete Heyting algebra).

If one drops distributivity/frame hypotheses, it is false that “nucleus fixed points are
Heyting” in general. The identity nucleus is a nucleus on *any* `SemilatticeInf`; its
fixed points are the entire lattice. So any non-distributive lattice is an immediate
counterexample to unsafe generalizations.

This module implements the standard minimal witness: the 5-element diamond lattice `M3`.
-/

namespace HeytingLean
namespace Topos
namespace JRatchet
namespace Guardrails

/-- The 5-element diamond lattice `M3`: ⊥, three atoms, ⊤. -/
inductive M3
  | bot | a | b | c | top
  deriving DecidableEq, Repr

namespace M3

instance : Bot M3 := ⟨bot⟩
instance : Top M3 := ⟨top⟩

/-- Order relation for the diamond: `⊥ ≤ x`, `x ≤ ⊤`, and atoms only ≤ themselves. -/
def le : M3 → M3 → Prop
  | bot, _ => True
  | _, top => True
  | a, a => True
  | b, b => True
  | c, c => True
  | _, _ => False

instance : LE M3 := ⟨le⟩

instance : DecidableRel (fun x y : M3 => x ≤ y) := by
  intro x y
  cases x <;> cases y <;>
    first
    | exact isTrue trivial
    | exact isFalse (by intro h; cases h)

@[simp] lemma bot_le (x : M3) : (⊥ : M3) ≤ x := by
  cases x <;> simp [LE.le, le]

@[simp] lemma le_top (x : M3) : x ≤ (⊤ : M3) := by
  cases x <;> simp [LE.le, le]

@[simp] lemma le_refl (x : M3) : x ≤ x := by
  cases x <;> simp [LE.le, le]

lemma le_trans {x y z : M3} (hxy : x ≤ y) (hyz : y ≤ z) : x ≤ z := by
  cases x <;> cases y <;> cases z <;> simp [LE.le, le] at *

lemma le_antisymm {x y : M3} (hxy : x ≤ y) (hyx : y ≤ x) : x = y := by
  cases x <;> cases y <;> simp [LE.le, le] at *

instance : PartialOrder M3 where
  le := (· ≤ ·)
  lt := fun x y => x ≤ y ∧ ¬ y ≤ x
  le_refl := le_refl
  le_trans := by
    intro x y z
    exact le_trans
  le_antisymm := by
    intro x y
    exact le_antisymm

/-- Meet in the diamond. -/
def inf : M3 → M3 → M3
  | bot, _ => bot
  | _, bot => bot
  | top, x => x
  | x, top => x
  | a, a => a
  | b, b => b
  | c, c => c
  | _, _ => bot

/-- Join in the diamond. -/
def sup : M3 → M3 → M3
  | bot, x => x
  | x, bot => x
  | top, _ => top
  | _, top => top
  | a, a => a
  | b, b => b
  | c, c => c
  | _, _ => top

instance : Max M3 := ⟨sup⟩

instance : SemilatticeSup M3 :=
  SemilatticeSup.mk (sup := Max.max)
    (le_sup_left := by
      intro x y
      cases x <;> cases y <;> simp [Max.max, sup, LE.le, le])
    (le_sup_right := by
      intro x y
      cases x <;> cases y <;> simp [Max.max, sup, LE.le, le])
    (sup_le := by
      intro x y z hxz hyz
      cases x <;> cases y <;> cases z <;> simp [Max.max, sup, LE.le, le] at * )

instance : Min M3 := ⟨inf⟩

instance : Lattice M3 :=
  Lattice.mk (inf := Min.min)
    (inf_le_left := by
      intro x y
      cases x <;> cases y <;> simp [Min.min, inf, LE.le, le])
    (inf_le_right := by
      intro x y
      cases x <;> cases y <;> simp [Min.min, inf, LE.le, le])
    (le_inf := by
      intro x y z hxy hxz
      cases x <;> cases y <;> cases z <;>
        simp [Min.min, inf, LE.le, le] at * )

/-- A concrete counterexample to distributivity in `M3`. -/
theorem not_distrib :
    (M3.a : M3) ⊓ ((M3.b : M3) ⊔ (M3.c : M3))
      ≠ ((M3.a : M3) ⊓ (M3.b : M3)) ⊔ ((M3.a : M3) ⊓ (M3.c : M3)) := by
  simp [Min.min, inf, Max.max, sup]

/-!
Interpretation (guardrail):

* `⊥ : Nucleus M3` is the identity nucleus (available on any `SemilatticeInf`),
* its fixed points are all of `M3`, and
* `not_distrib` shows this fixed-point core is not distributive, hence cannot be a Heyting
  layer (Heyting algebras are distributive).

Therefore: any theorem claiming “nucleus fixed points are Heyting” must assume a frame/distrib
ambient structure.
-/

end M3
end Guardrails
end JRatchet
end Topos
end HeytingLean

