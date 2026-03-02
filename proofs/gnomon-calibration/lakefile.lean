import Lake
open Lake DSL

package calibration where

-- Pin to a specific Mathlib commit for reproducible builds.
  -- Using Aristotle's Mathlib version for compatibility
  require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "f897ebcf72cd16f89ab4577d0c826cd14afaafc7"

@[default_target]
lean_lib Calibrator where
  srcDir := "proofs"
