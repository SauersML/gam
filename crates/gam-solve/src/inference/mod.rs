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

/// Margin-resolved [`Verdict`](gam_problem::topology_certificates::Verdict)
/// mappings for the two gam-solve-tier certificates consumed by
/// [`crate::topology_selector`]. Descended from the monolith
/// `inference::certificate_impls` (#1521): the two helpers here depend only on
/// gam-solve-tier types (`logdet_bounds`, `row_sampling_measure`) plus the
/// contracted-down certificate ladder — they do NOT pull `gam_sae`, so no
/// trait inversion is needed (the gam-sae `impl Certificate for …` blocks stay
/// in the monolith).
pub mod certificate_impls;

/// Structured residual-covariance estimator (#974) and the single producer of
/// [`MetricProvenance::WhitenedStructured`](gam_problem::MetricProvenance::WhitenedStructured).
/// Descended from the monolith `inference::residual_factor` (#1521): its only
/// dependencies are `gam_problem::RowMetric` plus `gam-linalg`
/// (`faer_ndarray::{FaerCholesky, FaerEigh}`) — all at or below the gam-solve
/// tier. Reached downward from `gam_sae::structure_harvest`; the monolith crate
/// root re-exports it so `gam::inference::residual_factor` resolves unchanged.
pub mod residual_factor;

/// Deterministic Pólya–Gamma gate-block evidence (#1016): Schur-eliminates a
/// logit gate sub-block with a true PG-augmented quadratic. Descended from the
/// monolith `inference::pg_gate_evidence` (#1521): depends only on
/// `gam-linalg` (`matrix::FactorizedSystem`, `faer_ndarray`) plus the in-crate
/// [`pg_moments`]. Reached downward from `gam_sae::structure_harvest`; the
/// monolith crate root re-exports it so `gam::inference::pg_gate_evidence`
/// resolves unchanged.
pub mod pg_gate_evidence;

/// Closed-form Pólya–Gamma moments (#1016). Pure functions of `(b, c)` with no
/// RNG. Descended from the monolith `inference::pg_moments` (#1521); the
/// monolith crate root re-exports it so `gam::inference::pg_moments` resolves
/// unchanged.
pub mod pg_moments;
