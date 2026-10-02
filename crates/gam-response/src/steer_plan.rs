//! The data a steering query hands to an intervention: which atom moved, where from
//! and to, the activation-space delta, and its Fisher dose. The SAE engine's
//! `inference::steering` produces it; [`crate::intervention_shard`] records and
//! replays it, so the type lives below both.

use gam_problem::MetricProvenance;
use ndarray::Array1;

/// Scientific status of the quadratic dose relative to the full output Fisher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FisherDoseKind {
    /// Euclidean/no-behavior metric: no nats dose exists.
    Unavailable,
    /// The supplied factor exactly represents the complete local Fisher.
    ExactFull,
    /// The producer certified the retained PSD operator as a lower bound.
    CertifiedPsdLowerBound,
    /// The factor is randomized, stochastic, truncated, or otherwise lacks an
    /// operator-order certificate.
    UncertifiedApproximation,
}

impl FisherDoseKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unavailable => "unavailable",
            Self::ExactFull => "exact_full",
            Self::CertifiedPsdLowerBound => "certified_psd_lower_bound",
            Self::UncertifiedApproximation => "uncertified_approximation",
        }
    }
}

/// The actionable output of a steering query over one atom.
#[derive(Clone, Debug, PartialEq)]
pub struct SteerPlan {
    /// Which atom was steered (index into the steered term's atoms).
    pub atom: usize,
    /// The atom's name (mirrors the atom's `name`).
    pub atom_name: String,
    /// The source latent coordinate `t_from` (length = atom's `latent_dim`).
    pub t_from: Vec<f64>,
    /// The target latent coordinate `t_to` (length = atom's `latent_dim`).
    pub t_to: Vec<f64>,
    /// The exact amplitude `a` the caller applied to the on-manifold move.
    pub amplitude: f64,
    /// The exact row whose output-Fisher metric prices the applied move.
    pub metric_row: usize,
    /// **The activation-space delta**: `δ = a · (g_k(t_to) − g_k(t_from))`, a
    /// length-`p` vector in the reconstruction/output space — the actual move to
    /// add to a hidden state.
    pub delta: Array1<f64>,
    /// **DOSIMETRY**: predicted output effect of the exact applied move in
    /// **nats** of KL, `0.5 * delta^T M_metric_row delta`.
    /// `None` when the metric carries no behavioral information (Euclidean
    /// provenance) — the dose is *not available*, not zero.
    pub predicted_nats: Option<f64>,
    /// Mathematical status of the factor used for `predicted_nats`.
    pub predicted_nats_kind: FisherDoseKind,
    /// Captured trace `tr(U_n U_n^T)` at `metric_row`, when behavior is present.
    pub fisher_mass_captured: Option<f64>,
    /// Non-negative omitted Fisher trace supplied by the harvest, when audited.
    pub fisher_mass_residual: Option<f64>,
    /// `residual / (captured + residual)`, when audited.
    pub fisher_mass_residual_fraction: Option<f64>,
    /// **OFF-MANIFOLD GUARD**: the norm of `δ`'s component outside the span of
    /// the atom's local decoder tangents `∂g_k/∂t` at `t_from`. `≈ 0` by
    /// construction (the move is a chord of the curve); a large value flags a
    /// move that left the learned surface.
    pub off_manifold_norm: f64,
    /// The provenance of the metric the dose was read through, echoed so a
    /// consumer can certify *why* `predicted_nats` is `None` when it is.
    pub metric_provenance: MetricProvenance,
}
