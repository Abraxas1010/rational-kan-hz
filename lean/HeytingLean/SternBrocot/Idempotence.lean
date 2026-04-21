import Mathlib.Data.Rat.Defs

/-!
# Stern-Brocot Projection Idempotence Surface

This module records the core formal hook used by the Rational KAN HZ paper run:
a bounded-denominator rational is fixed by the Stern-Brocot projection at the
same denominator bound. The projection interface is intentionally minimal: the
paper only consumes the identity-on-bounded half, while stronger nearest-rational
properties remain executable/Python-side evidence for this iteration.
-/

namespace HeytingLean
namespace SternBrocot

/--
Projection to the bounded-denominator rational surface.

For rationals already inside the denominator bound, this is definitionally the
identity. Outside the bound, the implementation keeps the numerator and uses the
requested denominator as a canonical bounded representative when `D` is nonzero.
The paper-run convergence lemma only depends on the bounded identity branch.
-/
def project (D : ℕ) (q : ℚ) : ℚ :=
  if q.den ≤ D then
    q
  else if hD : D = 0 then
    0
  else
    Rat.normalize q.num D hD

/--
Core paper-run lemma: Stern-Brocot projection at denominator bound `D` fixes every
rational whose canonical denominator is already at most `D`.
-/
theorem stern_brocot_idempotent_on_bounded
    (D : ℕ) (q : ℚ) (hq : q.den ≤ D) :
    project D q = q := by
  simp [project, hq]

/-- A named fixed-point spelling used by the paper text. -/
theorem bounded_rational_is_project_fixed_point
    (D : ℕ) (q : ℚ) (hq : q.den ≤ D) :
    project D q = q :=
  stern_brocot_idempotent_on_bounded D q hq

end SternBrocot
end HeytingLean
