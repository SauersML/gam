import Lake
open Lake DSL

package calibration where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.20.1"

@[default_target]
lean_lib Calibrator where
  srcDir := "proofs"
