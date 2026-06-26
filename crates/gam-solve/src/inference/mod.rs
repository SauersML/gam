//! Inference-tier numerics that are genuine gam-solve criterion math
//! (descended #1521): leaf evidence/diagnostic computations whose dependencies
//! are all at or below the gam-solve tier (`estimate`, `pirls`, `sensitivity`,
//! `mixture_link`, `gam-linalg`, `gam-problem`). The monolith crate root
//! re-exports this subtree as `gam::inference::*`, so existing callers
//! (`gam::inference::alo`, …) resolve unchanged.

/// Approximate-leave-one-out (ALO) REML-evidence diagnostics. Descended from the
/// monolith `inference::alo` (#1521): its only dependencies are the gam-solve
/// `estimate`/`pirls`/`sensitivity`/`mixture_link` modules plus `gam-linalg`
/// and `gam-problem` — all at or below the gam-solve tier.
pub mod alo;
