//! Laplace-correction / mode-posterior sampler contract (trait-inversion #1521).
//!
//! gam-solve's REML inner loop (`#784` block-local sampled marginal correction)
//! and the custom-family never-fail covariance path call into the
//! gam-inference-tier NUTS / importance-sampling engine (`inference::hmc_io`,
//! ~8k lines) — an UP-edge that keeps gam-solve in the inference SCC.
//!
//! The COMPUTATION (NUTS, importance sampling, the directional-cubic eigen
//! diagnostic) is irreducibly above gam-solve and STAYS UP in `hmc_io`. Only
//! the neutral surface is contract-downed here, mirroring the `rho_posterior`
//! data-down (#1521):
//!
//! * the plain-DATA result carriers gam-solve reads
//!   ([`BlockSampledMarginal`], [`BlockSampledMoments`], [`GaussianModePosterior`],
//!   [`LaplaceTrustworthiness`]);
//! * the caller-supplied [`BlockExcessTarget`] evaluator gam-solve IMPLEMENTS
//!   (its `Gam784BlockTarget`), so the trait must live below both;
//! * the two SAMPLER TRAITS ([`LaplaceMarginalSampler`],
//!   [`GaussianModePosteriorSampler`]) gam-solve calls THROUGH; the monolith /
//!   gam-inference implements them over `hmc_io` and injects the impl via the
//!   process-level registry below.
//!
//! The pure threshold math ([`laplace_skewness_threshold`],
//! [`laplace_trustworthiness_from_skewness`]) has no sampler dependency, so it is
//! moved down outright (gam-solve calls it directly).
//!
//! When no impl is registered (e.g. a build that never links the sampler tier)
//! the sampler getters return `None` and gam-solve degrades to its existing
//! decline paths — the `#784` correction returns zero (already a frequent
//! decline outcome) and the never-fail covariance path keeps the
//! optimizer-conditional covariance (already the `Err(reason)` fallback). The
//! contract therefore introduces no behavioral cliff and no stub.

use std::sync::OnceLock;

use gam_linalg::matrix::DesignMatrix;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ───────────────────────── data carriers (contract-down) ─────────────────────

/// Adaptive, block-local Laplace-trustworthiness verdict (issue #784): which
/// curvature directions are too non-Gaussian for the plain Laplace summary.
///
/// Field-for-field the monolith `hmc_io` type; that module re-exports this so
/// its construction sites name it unchanged.
#[derive(Clone, Debug)]
pub struct LaplaceTrustworthiness {
    /// Per-direction standardized skewness `γ_r`.
    pub directional_skewness: Array1<f64>,
    /// Indices of the directions whose skewness exceeds the auto-derived
    /// validity threshold (the curvature-heavy, non-Gaussian block).
    pub untrustworthy_directions: Vec<usize>,
    /// The auto-derived per-direction skewness threshold `τ(n)` actually used.
    pub threshold: f64,
    /// `max_r |γ_r|` across all directions (the global non-Gaussianity scale).
    pub max_abs_skewness: f64,
}

impl LaplaceTrustworthiness {
    /// Whether any curvature direction is too non-Gaussian for the plain
    /// Laplace summary, i.e. whether the higher-order correction / directional
    /// sampling fallback should engage at all.
    pub fn fallback_required(&self) -> bool {
        !self.untrustworthy_directions.is_empty()
    }
}

/// Self-normalized importance-weighted moments of the per-draw gradient channels
/// — the sampler-side half of the #784 exact-gradient seam. All expectations are
/// under `p ∝ q·e^{−ΔF}` over the SAME fixed-seed draws that produced the value,
/// so the spliced value and its assembled gradient can never desync (#901).
#[derive(Clone, Debug)]
pub struct BlockSampledMoments {
    /// `E_p[t]`, length `m`.
    pub e_t: Array1<f64>,
    /// `E_p[t tᵀ]`, shape `m × m`.
    pub e_tt: Array2<f64>,
    /// `E_p[ngs(η̂+s)]`, length n — the displaced per-row score moment.
    pub e_neg_score: Array1<f64>,
    /// Column `r` = `E_p[t_r · ngs(η̂+s)]`, shape `n × m`.
    pub e_t_neg_score: Array2<f64>,
}

/// Block-local sampled marginal correction (issue #784).
///
/// `value` is `Δ_b` (added to the block marginal log-likelihood, subtracted from
/// the REML/LAML cost); `rho_gradient` is the explicit penalty-score channel (a)
/// of the gradient exactness contract; `moments` carries the channels (b)–(d) the
/// gam-solve assembly contracts against fields it already owns.
#[derive(Clone, Debug)]
pub struct BlockSampledMarginal {
    /// `Δ_b`: additive correction to the block marginal log-likelihood.
    pub value: f64,
    /// `∂Δ_b/∂ρ`, length `rho_dim()` — explicit channel (a) ONLY.
    pub rho_gradient: Array1<f64>,
    /// Importance-sampling effective sample size (draws), for trust gating.
    pub importance_ess: f64,
    /// Number of draws used.
    pub n_draws: usize,
    /// Gradient-channel moments for the exact (b)–(d) assembly; `None` only when
    /// the block is empty (`m == 0`, where the correction is zero).
    pub moments: Option<BlockSampledMoments>,
}

/// Honest posterior summary from sampling the proper Gaussian posterior
/// `N(mode, H⁻¹)` — the terminal never-fail rung of the custom-family
/// covariance escalation. Field-for-field the monolith `hmc_io` type.
#[derive(Clone, Debug)]
pub struct GaussianModePosterior {
    /// Coefficient draws in original (un-whitened) space: `(n_draws, dim)`.
    pub samples: Array2<f64>,
    /// Posterior mean (≈ the seeded mode for a Gaussian target).
    pub posterior_mean: Array1<f64>,
    /// Per-coordinate posterior standard deviation (honest SEs).
    pub posterior_std: Array1<f64>,
    /// Split-chain R̂ mixing diagnostic.
    pub rhat: f64,
    /// Effective sample size.
    pub ess: f64,
}

// ───────────────────────── pure threshold math (moved down) ──────────────────

/// Auto-derive the per-direction skewness threshold `τ(n)` separating
/// Laplace-trustworthy directions from those that need the higher-order
/// correction / sampling fallback. Derived purely from the effective sample
/// size, no tunable flag: `(5/24)γ_r² > 1/n_eff ⇔ |γ_r| > sqrt((24/5)/n_eff)`.
pub fn laplace_skewness_threshold(n_eff: f64) -> f64 {
    if !(n_eff > 0.0) {
        return f64::INFINITY;
    }
    ((24.0 / 5.0) / n_eff).sqrt()
}

/// Adaptive, block-local Laplace-trustworthiness verdict (issue #784): flag the
/// directions whose standardized skewness exceeds [`laplace_skewness_threshold`].
/// No linear algebra of its own — consumes the directional cubic diagnostic.
pub fn laplace_trustworthiness_from_skewness(
    directional_skewness: &Array1<f64>,
    n_eff: f64,
) -> LaplaceTrustworthiness {
    let threshold = laplace_skewness_threshold(n_eff);
    let mut untrustworthy_directions = Vec::new();
    let mut max_abs_skewness = 0.0_f64;
    for (r, &gamma) in directional_skewness.iter().enumerate() {
        let abs_gamma = if gamma.is_finite() { gamma.abs() } else { 0.0 };
        max_abs_skewness = max_abs_skewness.max(abs_gamma);
        if abs_gamma > threshold {
            untrustworthy_directions.push(r);
        }
    }
    LaplaceTrustworthiness {
        directional_skewness: directional_skewness.clone(),
        untrustworthy_directions,
        threshold,
        max_abs_skewness,
    }
}

// ───────────────────────── caller-supplied excess evaluator ──────────────────

/// Caller-supplied evaluator for the non-Gaussian remainder `ΔF(t)` of the local
/// log-posterior, restricted to the curvature-heavy block subspace (issue #784).
///
/// Implemented by gam-solve's `Gam784BlockTarget`; consumed by
/// [`LaplaceMarginalSampler::block_sampled_marginal_correction`]. Lives in this
/// neutral crate so both the implementor (gam-solve) and the sampler impl (the
/// gam-inference monolith) name the same trait without an SCC edge.
pub trait BlockExcessTarget {
    /// Dimension `m` of the block subspace (number of untrustworthy directions
    /// being sampled).
    fn block_dim(&self) -> usize;
    /// Number of outer ρ coordinates the gradient is reported against.
    fn rho_dim(&self) -> usize;
    /// Block curvatures `λ_r` (the H-eigenvalues of the sampled directions),
    /// length `block_dim()`.
    fn block_curvatures(&self) -> &Array1<f64>;
    /// Non-Gaussian remainder `ΔF(t)` at whitened block displacement `t`
    /// (length `block_dim()`).
    fn excess(&self, t: &Array1<f64>) -> f64;
    /// ρ-gradient `∂ΔF/∂ρ_k` at the same `t`, length `rho_dim()` — the explicit
    /// penalty-score channel (a).
    fn excess_rho_gradient(&self, t: &Array1<f64>) -> Array1<f64>;
    /// Per-row displaced score `∂(D(η̂+s(t))/2φ)/∂η` evaluated at `η̂ + s(t)`
    /// (length = number of observation rows): the only per-draw ingredient of
    /// the exact-gradient channels (b)–(d) the assembly side cannot reconstruct.
    fn displaced_neg_score(&self, t: &Array1<f64>) -> Array1<f64>;
    /// The same per-row score channel at the undisplaced mode `η̂`.
    fn base_neg_score(&self) -> Array1<f64>;

    /// Fused `(excess(t), displaced_neg_score(t))`. The returned score is `None`
    /// exactly when the excess is non-finite (an infeasible draw the sampler
    /// discards before reading the score). The default preserves the two-call
    /// behavior; implementors override to share the displacement + jet.
    fn excess_with_displaced_neg_score(&self, t: &Array1<f64>) -> (f64, Option<Array1<f64>>) {
        let excess = self.excess(t);
        if excess.is_finite() {
            (excess, Some(self.displaced_neg_score(t)))
        } else {
            (excess, None)
        }
    }

    /// Batched [`Self::excess_with_displaced_neg_score`] over many whitened draws
    /// (one draw per COLUMN, shape `block_dim() × n_draws`). Batching may only
    /// change HOW the shared linear algebra is computed (one BLAS-3 product over
    /// all columns), never WHAT is computed. The default preserves the per-column
    /// behavior exactly; the GLM implementor overrides it.
    fn excess_with_displaced_neg_score_batch(
        &self,
        draws: &Array2<f64>,
    ) -> Vec<(f64, Option<Array1<f64>>)> {
        let n_draws = draws.ncols();
        let mut out = Vec::with_capacity(n_draws);
        let mut t = Array1::<f64>::zeros(draws.nrows());
        for s in 0..n_draws {
            t.assign(&draws.column(s));
            out.push(self.excess_with_displaced_neg_score(&t));
        }
        out
    }
}

// ───────────────────────── injected sampler traits ───────────────────────────

/// The gam-inference-tier sampler for the #784 block-local Laplace correction.
///
/// Implemented UP in the monolith over `hmc_io`
/// (`laplace_directional_cubic_diagnostic` + `block_sampled_marginal_correction`)
/// and injected DOWN via [`set_laplace_marginal_sampler`]. gam-solve calls
/// through [`laplace_marginal_sampler`].
pub trait LaplaceMarginalSampler: Send + Sync {
    /// Per-direction standardized cubic skewness `γ_r` of the local posterior:
    /// returns `(max_r |γ_r|, γ)`. Pure eigen-diagnostic (no sampling), but kept
    /// behind the trait because it lives in the sampler module up-tier.
    fn directional_cubic_diagnostic(
        &self,
        hessian: &Array2<f64>,
        design: &DesignMatrix,
        c_weights: &Array1<f64>,
        refine_supremum: bool,
    ) -> Result<(f64, Array1<f64>), String>;

    /// Estimate `Δ_b` and its ρ-gradient by importance sampling against the local
    /// Laplace Gaussian, contracting the caller-supplied [`BlockExcessTarget`].
    fn block_sampled_marginal_correction(
        &self,
        target: &dyn BlockExcessTarget,
    ) -> Result<BlockSampledMarginal, String>;
}

/// The gam-inference-tier sampler for the never-fail Gaussian mode posterior
/// (custom-family covariance escalation). Implemented UP over
/// `hmc_io::sample_gaussian_mode_posterior` (which auto-derives its
/// `NutsConfig::for_dimension(mode.len())` internally — that NUTS config never
/// crosses the contract) and injected DOWN via
/// [`set_gaussian_mode_posterior_sampler`].
pub trait GaussianModePosteriorSampler: Send + Sync {
    /// Sample `N(mode, precision⁻¹)`. `Err` only for a structurally impossible
    /// request (dimension mismatch, non-PSD precision after symmetrization) —
    /// never for "did not converge".
    fn sample_gaussian_mode_posterior(
        &self,
        mode: ArrayView1<f64>,
        precision: ArrayView2<f64>,
    ) -> Result<GaussianModePosterior, String>;
}

// ───────────────────────── process-level injection registry ──────────────────

static LAPLACE_MARGINAL_SAMPLER: OnceLock<Box<dyn LaplaceMarginalSampler>> = OnceLock::new();
static GAUSSIAN_MODE_POSTERIOR_SAMPLER: OnceLock<Box<dyn GaussianModePosteriorSampler>> =
    OnceLock::new();

/// Register the monolith's `hmc_io`-backed #784 Laplace-correction sampler.
/// Called once at process init by the gam-inference tier. First writer wins;
/// a later call is ignored (returns `Err` with the boxed value) so a re-init can
/// never swap a live sampler mid-run.
pub fn set_laplace_marginal_sampler(
    sampler: Box<dyn LaplaceMarginalSampler>,
) -> Result<(), Box<dyn LaplaceMarginalSampler>> {
    LAPLACE_MARGINAL_SAMPLER.set(sampler)
}

/// The registered #784 Laplace-correction sampler, or `None` when the sampler
/// tier is not linked / not yet initialized (gam-solve then declines the
/// correction, returning the zero contribution — a safe no-op).
pub fn laplace_marginal_sampler() -> Option<&'static dyn LaplaceMarginalSampler> {
    LAPLACE_MARGINAL_SAMPLER.get().map(|b| b.as_ref())
}

/// Register the monolith's `hmc_io`-backed never-fail Gaussian-mode-posterior
/// sampler. First writer wins (see [`set_laplace_marginal_sampler`]).
pub fn set_gaussian_mode_posterior_sampler(
    sampler: Box<dyn GaussianModePosteriorSampler>,
) -> Result<(), Box<dyn GaussianModePosteriorSampler>> {
    GAUSSIAN_MODE_POSTERIOR_SAMPLER.set(sampler)
}

/// The registered never-fail Gaussian-mode-posterior sampler, or `None` when the
/// sampler tier is not linked (the custom-family path then retains the
/// optimizer-conditional covariance — its existing fallback).
pub fn gaussian_mode_posterior_sampler() -> Option<&'static dyn GaussianModePosteriorSampler> {
    GAUSSIAN_MODE_POSTERIOR_SAMPLER.get().map(|b| b.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ── laplace_skewness_threshold ────────────────────────────────────────────

    #[test]
    fn threshold_is_infinity_for_zero_n_eff() {
        assert_eq!(laplace_skewness_threshold(0.0), f64::INFINITY);
    }

    #[test]
    fn threshold_is_infinity_for_negative_n_eff() {
        assert_eq!(laplace_skewness_threshold(-5.0), f64::INFINITY);
    }

    #[test]
    fn threshold_known_value() {
        // n_eff = 24/5 → sqrt((24/5) / (24/5)) = 1.0
        let n_eff = 24.0 / 5.0;
        let t = laplace_skewness_threshold(n_eff);
        assert!((t - 1.0).abs() < 1e-14, "threshold={t}");
    }

    #[test]
    fn threshold_decreases_as_n_eff_increases() {
        let t_small = laplace_skewness_threshold(10.0);
        let t_large = laplace_skewness_threshold(1000.0);
        assert!(t_large < t_small, "threshold should decrease with more data");
    }

    // ── laplace_trustworthiness_from_skewness ─────────────────────────────────

    #[test]
    fn all_small_skewness_gives_no_untrustworthy_directions() {
        // With n_eff=1000, threshold ≈ 0.069; all |γ| < that
        let skewness = array![0.01_f64, -0.02, 0.005];
        let result = laplace_trustworthiness_from_skewness(&skewness, 1000.0);
        assert!(result.untrustworthy_directions.is_empty());
        assert!(!result.fallback_required());
    }

    #[test]
    fn large_skewness_flagged_as_untrustworthy() {
        // With n_eff=10, threshold ≈ 0.693; γ=2.0 exceeds it
        let skewness = array![0.1_f64, 2.0];
        let result = laplace_trustworthiness_from_skewness(&skewness, 10.0);
        assert!(result.untrustworthy_directions.contains(&1));
        assert!(!result.untrustworthy_directions.contains(&0));
        assert!(result.fallback_required());
    }

    #[test]
    fn max_abs_skewness_is_largest_abs_value() {
        let skewness = array![1.5_f64, -3.0, 2.0];
        let result = laplace_trustworthiness_from_skewness(&skewness, 1.0);
        assert!((result.max_abs_skewness - 3.0).abs() < 1e-14);
    }

    #[test]
    fn non_finite_skewness_treated_as_zero_for_max_abs() {
        let skewness = array![f64::NAN, 1.0];
        let result = laplace_trustworthiness_from_skewness(&skewness, 1.0);
        // NaN is treated as 0 in the loop; max_abs comes from 1.0
        assert!((result.max_abs_skewness - 1.0).abs() < 1e-14);
    }

    // ── LaplaceTrustworthiness::fallback_required ─────────────────────────────

    #[test]
    fn fallback_required_true_when_directions_nonempty() {
        let lt = LaplaceTrustworthiness {
            directional_skewness: array![1.0_f64],
            untrustworthy_directions: vec![0],
            threshold: 0.5,
            max_abs_skewness: 1.0,
        };
        assert!(lt.fallback_required());
    }

    #[test]
    fn fallback_required_false_when_directions_empty() {
        let lt = LaplaceTrustworthiness {
            directional_skewness: array![0.1_f64],
            untrustworthy_directions: vec![],
            threshold: 0.5,
            max_abs_skewness: 0.1,
        };
        assert!(!lt.fallback_required());
    }
}
