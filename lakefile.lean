import Lake
open Lake DSL

package «RationalKANHZ» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.24.0"

require aesop from git
  "https://github.com/leanprover-community/aesop" @ "725ac8cd67acd70a7beaf47c3725e23484c1ef50"

@[default_target]
lean_lib HeytingLean where
  srcDir := "lean"
