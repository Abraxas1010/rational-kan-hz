import HeytingLean.Bridge.Veselov.HybridZeckendorf.FibFrac
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic

namespace HeytingLean.Bridge.Veselov.HybridZeckendorf

/-!
# Irreducibility of Fibonacci fractions

For the in-repo convention `fibFrac n = F_n / (F_{n+1} * F_{n+2})`,
irreducibility is the coprimality of `F_n` with the denominator.
-/

/-- Consecutive Fibonacci numbers are coprime. -/
theorem fib_coprime_succ (n : Nat) :
    Nat.Coprime (Nat.fib n) (Nat.fib (n + 1)) :=
  Nat.fib_coprime_fib_succ n

/-- A Fibonacci number is coprime to the term two steps ahead. -/
theorem fib_coprime_skip (n : Nat) :
    Nat.Coprime (Nat.fib n) (Nat.fib (n + 2)) := by
  have hsucc : Nat.Coprime (Nat.fib n) (Nat.fib (n + 1)) :=
    Nat.fib_coprime_fib_succ n
  have hrec : Nat.fib (n + 2) = Nat.fib n + Nat.fib (n + 1) := by
    exact Nat.fib_add_two (n := n)
  rw [hrec]
  simpa using hsucc

/-- The in-repo Fibonacci fraction `fibFrac n` is in lowest terms. -/
theorem fibFrac_irreducible (n : Nat) :
    Nat.Coprime (Nat.fib n) (Nat.fib (n + 1) * Nat.fib (n + 2)) := by
  exact Nat.Coprime.mul_right (fib_coprime_succ n) (fib_coprime_skip n)

/-- Paper-1 indexing: `a_n = F_{n-1}/(F_n F_{n+1})` is in lowest terms. -/
theorem fibFrac_paper_irreducible (n : Nat) (hn : 1 ≤ n) :
    Nat.Coprime (Nat.fib (n - 1)) (Nat.fib n * Nat.fib (n + 1)) := by
  have h := fibFrac_irreducible (n - 1)
  have hidx : n - 1 + 2 = n + 1 := by omega
  simpa [Nat.sub_add_cancel hn, hidx] using h

example : Nat.Coprime (Nat.fib 1) (Nat.fib 2 * Nat.fib 3) := by native_decide
example : Nat.Coprime (Nat.fib 4) (Nat.fib 5 * Nat.fib 6) := by native_decide
example : Nat.Coprime (Nat.fib 9) (Nat.fib 10 * Nat.fib 11) := by native_decide

end HeytingLean.Bridge.Veselov.HybridZeckendorf
