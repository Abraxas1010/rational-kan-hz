import HeytingLean.Boundary.Homomorphic.BoolBackend

namespace HeytingLean.Boundary.Homomorphic

universe v

/-!
# Homomorphic Zeckendorf normalization

This module provides:
1. **Plaintext duplicate carry step** (`plainCarryStep`): applies a
   circuit-shaped carry rewrite at position `k`, clearing position `k` and
   OR-ing its payload into `k+1` and, when available, `k-2`.
2. **Homomorphic carry step** (`homCarryStep`): the same operation over
   encrypted Boolean wires, parameterized by `HomBoolBackend`.
3. **Decryption correctness**: the homomorphic steps decrypt to the
   plaintext steps.
4. **Identity boundary** preserved from the prior phase for backward compat.
-/

/-! ### Plaintext carry step -/

/-- Set bit at position `i` in a list, extending with `false` if needed. -/
def setBit (bits : List Bool) (i : Nat) (v : Bool) : List Bool :=
  let padded := bits ++ List.replicate (max (i + 1) bits.length - bits.length) false
  padded.set i v

/-- Read bit at position `i`, returning `false` if out of bounds. -/
def getBit (bits : List Bool) (i : Nat) : Bool :=
  bits.getD i false

/-- Circuit output length for a duplicate carry step.
    If `k` is inside the input vector, the circuit exposes `k+1`; otherwise
    the shape remains unchanged. -/
def carryShapeLen (k inputLen : Nat) : Nat :=
  if k < inputLen then max inputLen (k + 2) else inputLen

/-- Circuit output length for a consecutive carry step.
    If `k+1` is inside the input vector, the circuit exposes `k+2`; otherwise
    the shape remains unchanged. -/
def consecutiveShapeLen (k inputLen : Nat) : Nat :=
  if k + 1 < inputLen then max inputLen (k + 3) else inputLen

/-- Plain duplicate carry as a fixed-shape Boolean circuit:
    position `k` is cleared and its payload is OR-ed into `k+1` and, for
    `k ≥ 2`, `k-2`. -/
def plainCarryStep (k : Nat) (bits : List Bool) : List Bool :=
  let bk := getBit bits k
  List.ofFn fun i : Fin (carryShapeLen k bits.length) =>
    let raw := getBit bits i
    if i.1 = k then
      raw && !bk
    else if i.1 = k + 1 then
      raw || bk
    else if k ≥ 2 && i.1 = k - 2 then
      raw || bk
    else
      raw

/-- Plain consecutive carry as a fixed-shape Boolean circuit: if positions
    `k` and `k+1` are both set, clear both and OR the payload into `k+2`. -/
def plainConsecutiveStep (k : Nat) (bits : List Bool) : List Bool :=
  let bk := getBit bits k
  let bk1 := getBit bits (k + 1)
  let both := bk && bk1
  List.ofFn fun i : Fin (consecutiveShapeLen k bits.length) =>
    let raw := getBit bits i
    if i.1 = k then
      raw && !both
    else if i.1 = k + 1 then
      raw && !both
    else if i.1 = k + 2 then
      raw || both
    else
      raw

/-- Plain boundary for a normalization pass (identity, for backward compat). -/
def plainNormalizeBoundary (bits : List Bool) : List Bool :=
  bits

/-! ### Homomorphic carry step -/

/-- Read encrypted bit at position `i`, returning encrypted `false` out of bounds. -/
def getEncBit [B : HomBoolBackend] (bits : List B.EncBool) (i : Nat) : B.EncBool :=
  bits.getD i B.hfalse

@[simp] theorem getBit_map_dec [B : HomBoolBackend] (bits : List B.EncBool) (i : Nat) :
    getBit (bits.map B.dec) i = B.dec (getEncBit bits i) := by
  induction bits generalizing i with
  | nil =>
      cases i <;> simp [getBit, getEncBit, HomBoolBackend.dec_hfalse]
  | cons x xs ih =>
      cases i with
      | zero => simp [getBit, getEncBit]
      | succ i => simpa [getBit, getEncBit] using ih i

/-- Encrypted duplicate carry at position `k`, shape-aligned with `plainCarryStep`. -/
def homCarryStep [B : HomBoolBackend] (k : Nat) (bits : List B.EncBool) : List B.EncBool :=
  let bk := getEncBit bits k
  List.ofFn fun i : Fin (carryShapeLen k bits.length) =>
    let raw := getEncBit bits i
    if i.1 = k then
      B.hand raw (B.hnot bk)
    else if i.1 = k + 1 then
      B.hor raw bk
    else if k ≥ 2 && i.1 = k - 2 then
      B.hor raw bk
    else
      raw

/-- Encrypted consecutive carry at position `k`, shape-aligned with `plainConsecutiveStep`. -/
def homConsecutiveStep [B : HomBoolBackend] (k : Nat) (bits : List B.EncBool) : List B.EncBool :=
  let bk := getEncBit bits k
  let bk1 := getEncBit bits (k + 1)
  let both := B.hand bk bk1
  List.ofFn fun i : Fin (consecutiveShapeLen k bits.length) =>
    let raw := getEncBit bits i
    if i.1 = k then
      B.hand raw (B.hnot both)
    else if i.1 = k + 1 then
      B.hand raw (B.hnot both)
    else if i.1 = k + 2 then
      B.hor raw both
    else
      raw

/-- Decrypting the homomorphic carry step gives the plaintext carry step. -/
theorem homCarryStep_decrypt_consistent [B : HomBoolBackend] (k : Nat)
    (bits : List B.EncBool) :
    (homCarryStep k bits).map B.dec =
      plainCarryStep k (bits.map B.dec) := by
  apply List.ext_getElem
  · simp [homCarryStep, plainCarryStep, carryShapeLen]
  · intro i h1 h2
    simp only [homCarryStep, plainCarryStep, List.getElem_map, List.getElem_ofFn,
      getBit_map_dec]
    by_cases hi : i = k
    · simp [hi, HomBoolBackend.dec_hnot, HomBoolBackend.dec_hand]
    · by_cases hi1 : i = k + 1
      · simp [hi1, HomBoolBackend.dec_hor]
      · by_cases hi2 : 2 ≤ k ∧ i = k - 2
        · have hk2ne : k - 2 ≠ k := by
            intro h
            exact hi (by rw [hi2.2, h])
          simp [hi2, hk2ne, HomBoolBackend.dec_hor]
        · simp [hi, hi1, hi2]

/-- Decrypting the homomorphic consecutive step gives the plaintext consecutive step. -/
theorem homConsecutiveStep_decrypt_consistent [B : HomBoolBackend] (k : Nat)
    (bits : List B.EncBool) :
    (homConsecutiveStep k bits).map B.dec =
      plainConsecutiveStep k (bits.map B.dec) := by
  apply List.ext_getElem
  · simp [homConsecutiveStep, plainConsecutiveStep, consecutiveShapeLen]
  · intro i h1 h2
    simp only [homConsecutiveStep, plainConsecutiveStep, List.getElem_map,
      List.getElem_ofFn, getBit_map_dec]
    by_cases hi : i = k
    · simp [hi, HomBoolBackend.dec_hnot, HomBoolBackend.dec_hand]
    · by_cases hi1 : i = k + 1
      · simp [hi1, HomBoolBackend.dec_hnot, HomBoolBackend.dec_hand]
      · by_cases hi2 : i = k + 2
        · simp [hi2, HomBoolBackend.dec_hand, HomBoolBackend.dec_hor]
        · simp [hi, hi1, hi2]

/-- Homomorphic boundary parameterized over the existing Boolean backend. -/
def homNormalizeBoundary [B : HomBoolBackend] (bits : List B.EncBool) : List B.EncBool :=
  bits

/-- Decrypting the homomorphic boundary gives the plain boundary. -/
theorem homNormalizeBoundary_decrypt [B : HomBoolBackend] (bits : List B.EncBool) :
    (homNormalizeBoundary bits).map B.dec =
      plainNormalizeBoundary (bits.map B.dec) := by
  rfl

/-- Iteration helper for encrypted rewrite steps. -/
def iterateHomNorm (α : Type v) (step : List α → List α) (n : Nat) (xs : List α) :
    List α :=
  match n with
  | 0 => xs
  | n + 1 => iterateHomNorm α step n (step xs)

@[simp] theorem iterateHomNorm_identity {α : Type v} (n : Nat) (xs : List α) :
    iterateHomNorm α (fun ys : List α => ys) n xs = xs := by
  induction n generalizing xs with
  | zero =>
      rfl
  | succ n ih =>
      simp [iterateHomNorm, ih]

/-- The boundary iteration decrypts to the same plaintext boundary. -/
theorem homNormalizeBoundary_as_iterated_identity [B : HomBoolBackend]
    (bits : List B.EncBool) (maxIter : Nat) :
    (iterateHomNorm B.EncBool (fun ys : List B.EncBool => ys) maxIter
        (homNormalizeBoundary bits)).map B.dec =
      plainNormalizeBoundary (bits.map B.dec) := by
  simp [homNormalizeBoundary, plainNormalizeBoundary]

/-- Iterated carry step application. -/
def iterateCarrySteps [B : HomBoolBackend] (n : Nat) (bits : List B.EncBool) : List B.EncBool :=
  iterateHomNorm B.EncBool (fun xs => homCarryStep 0 xs) n bits

/-- Iterated consecutive step application. -/
def iterateConsecutiveSteps [B : HomBoolBackend] (n : Nat) (bits : List B.EncBool) : List B.EncBool :=
  iterateHomNorm B.EncBool (fun xs => homConsecutiveStep 0 xs) n bits

end HeytingLean.Boundary.Homomorphic
