//! Approximation ledger.
//!
//! A small classification scheme for the deliberate departures from "exact
//! arithmetic on the stated statistical model" that show up across the solver
//! and family code. The point is to make hand-wavy markers like "bandaid",
//! "hack", or "magic" carry a meaningful contract instead of just a vibe:
//! every such marker in the source must sit next to an [`ApproxKind`]
//! annotation that names which kind of approximation it actually is, so a
//! reader (or a downstream caller, or `build.rs`) can tell at a glance
//! whether the deviation is a bounded numerical detail, a controlled
//! statistical estimator, a surrogate objective, or a temporary damping that
//! goes away as the solver converges.
//!
//! The taxonomy is deliberately closed (five variants) so that adding a new
//! category is a deliberate act, not a drift. The annotation does not change
//! runtime behavior — it exists as a `const` marker that callers and tests
//! can read, and as a syntactic anchor that `build.rs` matches against.
//!
//! ## Marker convention
//!
//! Hand-wavy words ("bandaid", "hack", "magic", "FIXME") are allowed in
//! source comments only when the same comment block (within `LEDGER_WINDOW`
//! lines) also references one of the variants below by its bare identifier:
//! `Exact`, `NumericalApproximation`, `StatisticalApproximation`,
//! `SurrogateObjective`, `TemporarySolverDamping`. The build script
//! enforces this; see `build.rs` for the scanner.

#![allow(dead_code)]

/// One classification entry. Five variants, each with a short payload that
/// names the bound or contract that justifies the approximation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproxKind {
    /// No approximation: arithmetic matches the stated model up to IEEE-754
    /// rounding. Use this when annotating a site that *looks* like an
    /// approximation but is provably exact (e.g. an algebraic rewrite).
    Exact,

    /// Bounded numerical approximation: the deviation from the exact
    /// expression is governed by a backward error bound (typically a small
    /// multiple of `eps_machine`, or a clamp justified by floating-point
    /// range). Examples: `eta.clamp(-700, 700)` before `exp()`, condition-
    /// number floors on positive weights.
    NumericalApproximation {
        /// Free-form description of the backward error bound, e.g.
        /// `"|out - exp(eta)| <= eps for |eta| <= 700"`.
        backward_error_bound: &'static str,
    },

    /// Statistical approximation: the estimator is intentionally not the
    /// full-data MLE (or full-data score), but its sampling distribution
    /// is bounded relative to the full-data version. Examples: stratified
    /// Horvitz-Thompson outer-score subsamples.
    StatisticalApproximation {
        /// Free-form description of the variance bound, e.g.
        /// `"Var(score_K) / Var(score_n) = O(n / K) for K = auto_outer_subsample_k(n)"`.
        variance_bound: &'static str,
    },

    /// Surrogate objective: the function being optimized is not the target
    /// objective but a related one whose stationary points coincide
    /// (locally) with the target's. Examples: PQL-type Fisher-scoring
    /// inner steps that use expected information instead of observed.
    SurrogateObjective {
        /// Free-form description of the relationship to the true target.
        description: &'static str,
    },

    /// Temporary solver damping: a transient modification of the solver
    /// step (line-search backoff, LM ridge, trust-region cap) that does
    /// not change the fixed point — at convergence the damping is inactive
    /// and the result is identical to the undamped solver. Examples:
    /// LM `lambda` ridge on the curvature, schedule backoffs.
    TemporarySolverDamping,
}

/// Window (in lines) inside which a hand-wavy marker comment must reference
/// an `ApproxKind` variant to be considered annotated. Kept generous so
/// long comment blocks above a marker still satisfy the scanner.
pub const LEDGER_WINDOW: usize = 8;

/// Compile-time site annotation. Carries no runtime state; its presence as
/// a `const` in the source serves as both documentation and as a syntactic
/// anchor that mirrors the comment-level annotation.
#[derive(Debug, Clone, Copy)]
pub struct LedgerSite {
    pub site: &'static str,
    pub kind: ApproxKind,
}

impl LedgerSite {
    pub const fn new(site: &'static str, kind: ApproxKind) -> Self {
        Self { site, kind }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_are_distinct() {
        let a = ApproxKind::Exact;
        let b = ApproxKind::NumericalApproximation {
            backward_error_bound: "x",
        };
        assert_ne!(a, b);
    }

    #[test]
    fn ledger_site_carries_kind() {
        const S: LedgerSite =
            LedgerSite::new("test_site", ApproxKind::TemporarySolverDamping);
        assert_eq!(S.site, "test_site");
        assert!(matches!(S.kind, ApproxKind::TemporarySolverDamping));
    }
}
