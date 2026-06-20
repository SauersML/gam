use faer::Side;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::faer_ndarray::FaerCholesky;
use crate::linalg::utils::stack_offsets;
use crate::model_types::{Dispersion, EstimationError};
use crate::types::{
    GlmLikelihoodSpec, InverseLink, LatentCLogLogState, LikelihoodScaleMetadata, LikelihoodSpec,
    LogLikelihoodNormalization, MixtureLinkSpec, MixtureLinkState, ResponseFamily, SasLinkSpec,
    SasLinkState, StandardLink,
};

/// Strictly-positive floor on a reported dispersion / scale parameter `φ`.
/// Every GLM family resolves `φ` to a non-negative quantity, but downstream
/// consumers (covariance scaling, deviance ratios) divide by it, so it is
/// clamped to the smallest positive normal `f64` to keep those quotients
/// finite without perturbing any `φ` above the denormal range.
const DISPERSION_POSITIVE_FLOOR: f64 = 1e-300;

pub(crate) fn dispersion_from_likelihood(
    likelihood: &GlmLikelihoodSpec,
    standard_deviation: f64,
) -> Dispersion {
    match &likelihood.spec.response {
        ResponseFamily::Gaussian => Dispersion::Estimated(
            (standard_deviation * standard_deviation).max(DISPERSION_POSITIVE_FLOOR),
        ),
        ResponseFamily::Gamma => {
            let phi = likelihood.scale.fixed_phi().unwrap_or_else(|| {
                let shape = likelihood
                    .gamma_shape()
                    .unwrap_or(standard_deviation.max(DISPERSION_POSITIVE_FLOOR));
                1.0 / shape.max(DISPERSION_POSITIVE_FLOOR)
            });
            if likelihood.scale.gamma_shape_is_estimated() {
                Dispersion::Estimated(phi.max(DISPERSION_POSITIVE_FLOOR))
            } else {
                Dispersion::Known(phi.max(DISPERSION_POSITIVE_FLOOR))
            }
        }
        ResponseFamily::Tweedie { .. } => {
            let phi = likelihood
                .fixed_phi()
                .unwrap_or(1.0)
                .max(DISPERSION_POSITIVE_FLOOR);
            if likelihood.scale.tweedie_phi_is_estimated() {
                Dispersion::Estimated(phi)
            } else {
                Dispersion::Known(phi)
            }
        }
        ResponseFamily::NegativeBinomial { theta, .. } => Dispersion::Known(
            likelihood
                .fixed_phi()
                .unwrap_or(*theta)
                .max(DISPERSION_POSITIVE_FLOOR),
        ),
        ResponseFamily::Beta { phi } => {
            Dispersion::Known((1.0 / (1.0 + phi.max(1e-12))).max(DISPERSION_POSITIVE_FLOOR))
        }
        ResponseFamily::Binomial | ResponseFamily::Poisson | ResponseFamily::RoystonParmar => {
            Dispersion::Known(1.0)
        }
    }
}

#[cfg(test)]
mod per_term_edf_tests {
    use super::*;

    fn eye(n: usize) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((n, n));
        for j in 0..n {
            out[[j, j]] = 1.0;
        }
        out
    }

    fn fit_with_legacy_tensor_block_sum() -> UnifiedFitResult {
        let beta = Array1::zeros(36);
        UnifiedFitResult::new_for_test_unchecked(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: beta.clone(),
                role: BlockRole::Mean,
                edf: 28.0,
                lambdas: Array1::from_vec(vec![1.0, 1.0]),
            }],
            log_lambdas: Array1::zeros(2),
            lambdas: Array1::from_vec(vec![1.0, 1.0]),
            likelihood_family: Some(LikelihoodSpec::gaussian_identity()),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: Some(0.0),
            standard_deviation: 1.0,
            covariance_conditional: None,
            covariance_corrected: None,
            inference: Some(FitInference {
                edf_by_block: vec![20.0, 20.0],
                penalty_block_trace: Vec::new(),
                edf_total: 28.0,
                smoothing_correction: None,
                penalized_hessian: crate::inference::dispersion_cov::UnscaledPrecision::wrap(eye(
                    36,
                )),
                working_weights: Array1::ones(1),
                working_response: Array1::zeros(1),
                reparam_qs: None,
                dispersion: Dispersion::Estimated(1.0),
                beta_covariance: None,
                beta_standard_errors: None,
                beta_covariance_corrected: None,
                beta_standard_errors_corrected: None,
                beta_covariance_frequentist: None,
                coefficient_influence: None,
                weighted_gram: None,
                bias_correction_beta: None,
            }),
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: crate::pirls::PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts::default(),
            inner_cycles: 0,
        })
    }

    #[test]
    fn per_term_edf_legacy_block_sum_is_capped_by_model_total() {
        let fit = fit_with_legacy_tensor_block_sum();

        let edf = fit.per_term_edf(1..36, 0, 2);

        assert_eq!(edf, 28.0);
    }

    /// Single-smooth thin-plate model whose EDF channels agree: `edf_total`
    /// equals the influence-matrix trace `tr(F)` and is consistent with the
    /// single smooth term's own per-term EDF. Guards issue #1356, where the
    /// trace channel (`p − Σ tr_kk` from an over-ridged TRANSFORMED Hessian)
    /// collapsed `edf_total` onto its `1.0` floor while the per-term influence
    /// trace legitimately reported ~71 EDF — the total fell *below* a single
    /// term's EDF, which is structurally impossible for a sum of non-negative
    /// per-term contributions. After the optimizer reconciles both channels to
    /// the same rank-revealing inverse, `edf_total ≥ max per-term EDF` holds.
    fn fit_single_thinplate_consistent_edf() -> UnifiedFitResult {
        // p = 11: one intercept column (index 0, unpenalised, EDF 1) plus a
        // 10-coefficient thin-plate block that has spent 7 EDF (F diagonal 0.7).
        let p = 11usize;
        let mut influence = Array2::<f64>::zeros((p, p));
        influence[[0, 0]] = 1.0; // intercept: full degree of freedom
        for j in 1..p {
            influence[[j, j]] = 0.7; // smooth block: Σ = 7.0 EDF
        }
        let edf_total: f64 = (0..p).map(|j| influence[[j, j]]).sum(); // tr(F) = 8.0
        let beta = Array1::zeros(p);
        UnifiedFitResult::new_for_test_unchecked(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: beta.clone(),
                role: BlockRole::Mean,
                edf: edf_total,
                lambdas: Array1::from_vec(vec![1.0]),
            }],
            log_lambdas: Array1::zeros(1),
            lambdas: Array1::from_vec(vec![1.0]),
            likelihood_family: Some(LikelihoodSpec::gaussian_identity()),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: Some(0.0),
            standard_deviation: 1.0,
            covariance_conditional: None,
            covariance_corrected: None,
            inference: Some(FitInference {
                // tr_kk over the single penalty block = dim − edf = 10 − 7 = 3.
                edf_by_block: vec![7.0],
                penalty_block_trace: vec![3.0],
                edf_total,
                smoothing_correction: None,
                penalized_hessian: crate::inference::dispersion_cov::UnscaledPrecision::wrap(eye(
                    p,
                )),
                working_weights: Array1::ones(1),
                working_response: Array1::zeros(1),
                reparam_qs: None,
                dispersion: Dispersion::Estimated(1.0),
                beta_covariance: None,
                beta_standard_errors: None,
                beta_covariance_corrected: None,
                beta_standard_errors_corrected: None,
                beta_covariance_frequentist: None,
                coefficient_influence: Some(influence),
                weighted_gram: None,
                bias_correction_beta: None,
            }),
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: crate::pirls::PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts::default(),
            inner_cycles: 0,
        })
    }

    /// Build a fit that mirrors a factor `by=` smooth's penalty layout for the
    /// fallback (no influence matrix) `per_term_edf` channel that the persisted /
    /// FFI summary path uses. `n_levels` centred by-smooths each own ONE penalty
    /// block (`penalty_block_trace = 5` over a dim-20 block → EDF 15); the
    /// UNPENALISED treatment-coded factor main-effect block the `by=` expansion
    /// injects owns NO penalty block, so `penalty_block_trace.len() == n_levels`
    /// — strictly less than the number of summary rows (1 RE + n_levels smooths).
    fn fit_by_factor_penalty_layout(n_levels: usize) -> UnifiedFitResult {
        let dim = 20usize;
        let trace_per_block = 5.0_f64; // each by-smooth spends dim − trace = 15 EDF
        let traces = vec![trace_per_block; n_levels];
        let edf_per_smooth = dim as f64 - trace_per_block;
        // 1 unpenalised factor main effect (4 EDF, say) + n_levels × 15 smooth EDF.
        let edf_total = 4.0 + n_levels as f64 * edf_per_smooth;
        let p = 1 + dim * n_levels;
        let lambdas = Array1::from_vec(vec![1.0; n_levels]);
        UnifiedFitResult::new_for_test_unchecked(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: Array1::zeros(p),
                role: BlockRole::Mean,
                edf: edf_total,
                lambdas: lambdas.clone(),
            }],
            log_lambdas: Array1::zeros(n_levels),
            lambdas,
            likelihood_family: Some(LikelihoodSpec::gaussian_identity()),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: Some(0.0),
            standard_deviation: 1.0,
            covariance_conditional: None,
            covariance_corrected: None,
            inference: Some(FitInference {
                edf_by_block: vec![edf_per_smooth; n_levels],
                penalty_block_trace: traces,
                edf_total,
                smoothing_correction: None,
                penalized_hessian: crate::inference::dispersion_cov::UnscaledPrecision::wrap(eye(p)),
                working_weights: Array1::ones(1),
                working_response: Array1::zeros(1),
                reparam_qs: None,
                dispersion: Dispersion::Estimated(1.0),
                beta_covariance: None,
                beta_standard_errors: None,
                beta_covariance_corrected: None,
                beta_standard_errors_corrected: None,
                beta_covariance_frequentist: None,
                // No influence matrix: forces the `|coeff_range| − Σ tr_kk`
                // per-block-trace channel, where the `penalty_cursor` walk matters.
                coefficient_influence: None,
                weighted_gram: None,
                bias_correction_beta: None,
            }),
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: crate::pirls::PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts::default(),
            inner_cycles: 0,
        })
    }

    /// Regression for issue #1368. The summary smooth-term loop walks a
    /// `penalty_cursor` across the flat penalty-block layout. A factor `by=`
    /// smooth injects an UNPENALISED treatment-coded factor main-effect random
    /// block that owns NO penalty block; advancing the cursor for it slides every
    /// following by-level smooth's `penalty_cursor..+k` trace window one block
    /// down, so the LAST level's slice runs off the end of `penalty_block_trace`
    /// and `per_term_edf` returns 0 (then the EDF-0 term gets a NaN significance).
    ///
    /// This reproduces the exact cursor walk against the fallback per-block-trace
    /// channel: the BUGGY cursor (advance +1 for the unpenalised RE block) zeroes
    /// the last level; the FIXED cursor (advance +0 for an unpenalised block, the
    /// number of penalty blocks it actually owns) gives every level its EDF and
    /// recovers the per-term sum to within tolerance of `edf_total`.
    #[test]
    fn by_factor_unpenalised_main_effect_does_not_zero_last_level_edf() {
        let n_levels = 5usize;
        let dim = 20usize;
        let fit = fit_by_factor_penalty_layout(n_levels);
        let smooth_start = 1; // global layout: [unpenalised RE block(1) | smooths]
        let expected_per_smooth = 15.0_f64;
        let tol = 1e-9;

        // ── BUGGY walk: the unpenalised RE block advances the cursor by 1. ──
        let mut buggy_cursor = 0usize;
        buggy_cursor += 1; // RE block treated as owning one penalty block
        let mut buggy_edfs = Vec::new();
        for level in 0..n_levels {
            let start = smooth_start + level * dim;
            let edf = fit.per_term_edf(start..(start + dim), buggy_cursor, 1);
            buggy_cursor += 1;
            buggy_edfs.push(edf);
        }
        // The defect: the last by-level smooth's trace window runs off the end of
        // `penalty_block_trace` (len = n_levels) and collapses to 0 EDF.
        assert!(
            buggy_edfs.last().copied().unwrap() <= tol,
            "expected the buggy cursor to zero the last level's EDF, got {:?}",
            buggy_edfs
        );

        // ── FIXED walk: the unpenalised RE block owns 0 penalty blocks. ──
        let mut cursor = 0usize;
        let re_penalized = false; // the injected factor main effect is unpenalised
        cursor += usize::from(re_penalized); // advance by blocks actually owned (0)
        let mut edfs = Vec::new();
        for level in 0..n_levels {
            let start = smooth_start + level * dim;
            let edf = fit.per_term_edf(start..(start + dim), cursor, 1);
            cursor += 1;
            edfs.push(edf);
        }
        // Every level — including the last — now reports its honest EDF.
        for (level, &edf) in edfs.iter().enumerate() {
            assert!(
                (edf - expected_per_smooth).abs() < tol,
                "level {level} EDF {edf} != expected {expected_per_smooth} (set {edfs:?})"
            );
            assert!(edf > 1.0, "level {level} EDF {edf} must be well above 1");
        }
        // The per-term EDFs (smooths + the unpenalised main-effect dof) reconstruct
        // the model total — the dropped last level previously left a 15-EDF gap.
        let main_effect_edf = 4.0_f64; // the unpenalised RE block's full dof
        let reconstructed: f64 = edfs.iter().sum::<f64>() + main_effect_edf;
        let edf_total = fit.edf_total().expect("edf_total present");
        assert!(
            (reconstructed - edf_total).abs() < 1e-6,
            "Σ per-term EDF ({reconstructed}) must match edf_total ({edf_total})"
        );
    }

    #[test]
    fn edf_total_never_below_a_single_terms_edf() {
        let fit = fit_single_thinplate_consistent_edf();

        // The single thin-plate smooth term (coeff columns 1..11, one penalty
        // block at cursor 0) reports its influence-matrix EDF.
        let term_edf = fit.per_term_edf(1..11, 0, 1);
        let edf_total = fit.edf_total().expect("edf_total present");

        // #1356 invariant: a sum of non-negative per-term EDF contributions can
        // never be smaller than any one of them.
        assert!(
            edf_total + 1e-9 >= term_edf,
            "edf_total ({edf_total}) fell below a single term's EDF ({term_edf})"
        );
        // And both channels read the same influence matrix, so the smooth term
        // is the model total minus the intercept's one degree of freedom.
        assert!((term_edf - 7.0).abs() < 1e-9, "term_edf = {term_edf}");
        assert!((edf_total - 8.0).abs() < 1e-9, "edf_total = {edf_total}");
    }
}

/// Standardized-disagreement gate: the audit flags inconsistency when the
/// analytic and FD directional derivatives differ by more than this many FD
/// error bars (and also fail the relative gate).
pub(crate) const CERTIFICATE_Z_GATE: f64 = 4.0;

/// Relative agreement gate: differences below this fraction of the larger
/// directional derivative are consistent regardless of the (possibly
/// underestimated) FD error bar.
pub(crate) const CERTIFICATE_RELATIVE_GATE: f64 = 1e-3;

/// ρ margin (in log-λ units) within which an outer smoothing coordinate
/// counts as railed against its box bound.
pub(crate) const CERTIFICATE_RAIL_MARGIN: f64 = 0.5;

/// First-order optimality certificate: gradient-vs-objective FD audit at the
/// returned optimum (#934).
///
/// Answers, machine-checkably, the three questions every objective↔gradient
/// desync postmortem asks: does the analytic gradient match the actual
/// criterion value HERE ([`Self::first_order_consistent`]); is the outer
/// curvature positive definite HERE (`hessian_pd`); did any smoothing
/// coordinate rail to a box bound (`lambdas_railed`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CriterionCertificate {
    /// ‖∇F(θ̂)‖₂ from the analytic gradient path at the returned point.
    pub grad_norm: f64,
    /// Analytic directional derivative ∇F(θ̂)·v along the audit direction.
    pub analytic_directional: f64,
    // FD-OK: this certificate STORES a finite-difference oracle of the criterion
    // value solely to AUDIT the analytic directional derivative against it (the
    // analytic path is authoritative); the FD never feeds the optimizer.
    /// Richardson-extrapolated central difference of the criterion VALUE
    /// path along the same direction: (4·D_h − D_2h)/3 from the h and 2h
    /// central-difference pairs.
    pub fd_directional: f64, // fd-ok: FD-audit certificate, not in math path
    /// Error bar on `fd_directional`: the Richardson residual |D_h − D_2h|
    /// (which absorbs both truncation and inner-solve value noise) floored
    /// by the central-difference roundoff bound ε·|F|/h.
    pub fd_error: f64, // fd-ok: FD-audit certificate, not in math path
    /// |analytic − fd| / fd_error — standardized disagreement.
    pub agreement_z: f64,
    /// Base central-difference step h along the unit direction.
    pub fd_step: f64, // fd-ok: FD-audit certificate, not in math path
    // END-FD-OK
    /// Whether the final outer Hessian is positive definite at θ̂, when the
    /// solver tracked one (`None` when no final Hessian was available).
    pub hessian_pd: Option<bool>,
    /// Leading smoothing coordinates (ρ block) pinned within
    /// [`CERTIFICATE_RAIL_MARGIN`] of either box bound at the optimum.
    pub lambdas_railed: Vec<usize>,
}

impl CriterionCertificate {
    /// Whether the analytic directional derivative agrees with the finite
    /// difference of the actual criterion value at the optimum.
    pub fn first_order_consistent(&self) -> bool {
        // FD-OK: audit comparison of the analytic directional derivative against
        // the stored finite-difference oracle; this is the certificate check, not
        // a computational FD path.
        let diff = (self.analytic_directional - self.fd_directional).abs(); // fd-ok: FD-audit certificate, not in math path
        let scale = self
            .analytic_directional
            .abs()
            .max(self.fd_directional.abs()); // fd-ok: FD-audit certificate, not in math path
        diff <= (CERTIFICATE_Z_GATE * self.fd_error).max(CERTIFICATE_RELATIVE_GATE * scale) // fd-ok: FD-audit certificate, not in math path
        // END-FD-OK
    }

    /// Whether every audited fact is clean: gradient matches objective, no
    /// definiteness failure, no railed smoothing coordinate.
    pub fn is_clean(&self) -> bool {
        self.first_order_consistent()
            && self.hessian_pd != Some(false)
            && self.lambdas_railed.is_empty()
    }

    /// One-line human-readable rendering for logs and reports.
    pub fn summary(&self) -> String {
        format!(
            // FD-OK: human-readable summary of the audit certificate's stored
            // FD oracle fields; reporting only, no FD computation here.
            "grad·v={:.6e} fd·v={:.6e}±{:.1e} z={:.2} |g|={:.3e} hessian_pd={} railed={:?} → {}",
            self.analytic_directional,
            self.fd_directional, // fd-ok: FD-audit certificate, not in math path
            self.fd_error,       // fd-ok: FD-audit certificate, not in math path
            // END-FD-OK
            self.agreement_z,
            self.grad_norm,
            match self.hessian_pd {
                Some(true) => "yes",
                Some(false) => "NO",
                None => "n/a",
            },
            self.lambdas_railed,
            if self.first_order_consistent() {
                "consistent"
            } else {
                "GRADIENT-OBJECTIVE DESYNC"
            },
        )
    }
}

#[derive(Clone, Debug)]
pub struct FitOptions {
    pub latent_cloglog: Option<LatentCLogLogState>,
    pub mixture_link: Option<MixtureLinkSpec>,
    pub optimize_mixture: bool,
    pub sas_link: Option<SasLinkSpec>,
    pub optimize_sas: bool,
    pub compute_inference: bool,
    /// Internal lifecycle knob for fits whose result will be immediately
    /// superseded. Keeps ordinary inference work but skips the live-objective
    /// rho posterior certificate/escalation until the returned model is known.
    pub skip_rho_posterior_inference: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub nullspace_dims: Vec<usize>,
    pub linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    /// Use Jeffreys/Firth bias reduction for supported likelihoods.
    ///
    /// Model-fitting paths must pass this explicitly through every objective
    /// evaluator so baseline fits, spatial hyperparameter evaluations, outer
    /// line searches, final refits, and inference all optimize the same target.
    pub firth_bias_reduction: bool,
    pub adaptive_regularization: Option<AdaptiveRegularizationOptions>,
    /// Relative shrinkage floor for penalized block eigenvalues.
    ///
    /// When `Some(epsilon)`, a rho-independent ridge of magnitude
    /// `epsilon * max_balanced_eigenvalue` is added to each eigenvalue of the
    /// combined penalty on the penalized block. This acts as a weak proper
    /// complexity prior that prevents barely-penalized directions from causing
    /// pathological non-Gaussianity in the posterior (e.g., extreme skewness
    /// under logit link with high-dimensional spatial smooths).
    ///
    /// The ridge is rho-independent, so LAML gradients remain correct without
    /// modification (d(epsilon*I)/d(rho_k) = 0).
    ///
    /// Typical value: `Some(1e-6)`. Set to `None` or `Some(0.0)` to disable.
    /// Default: `Some(1e-6)`.
    pub penalty_shrinkage_floor: Option<f64>,
    /// Fixed prior on smoothing parameters for explicit joint HMC sampling
    /// flows.
    ///
    /// This prior is part of the sampled target itself, unlike `rho_mode`,
    /// which is only used to initialize chains near the REML solution.
    pub rho_prior: crate::types::RhoPrior,
    /// Kronecker-factored penalty system for tensor-product smooth terms.
    /// When set, the REML evaluator uses O(∏q_j) logdet and KroneckerMarginal
    /// penalty coordinates instead of O(p³) eigendecomposition.
    pub kronecker_penalty_system: Option<crate::smooth::KroneckerPenaltySystem>,
    /// Full Kronecker factored basis for P-IRLS factored reparameterization.
    pub kronecker_factored: Option<crate::basis::KroneckerFactoredBasis>,
    /// Engage the cross-process ON-DISK persistent warm-start layer.
    ///
    /// Default `false`: only the always-on in-memory warm start runs, so a
    /// single fit and throwaway/replicate/CI-coverage loops pay zero disk I/O
    /// (#1082). Set `true` (threaded from `FitConfig::persist_warm_start_disk`)
    /// to engage cross-process / repeat-fit resume; the standard `RemlState`
    /// then calls `enable_persistent_warm_start_disk()`.
    pub persist_warm_start_disk: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            skip_rho_posterior_inference: false,
            max_iter: 100,
            tol: 1e-6,
            nullspace_dims: Vec::new(),
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: Some(1e-6),
            rho_prior: crate::types::RhoPrior::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveRegularizationOptions {
    pub enabled: bool,
    pub max_mm_iter: usize,
    pub beta_rel_tol: f64,
    pub max_epsilon_outer_iter: usize,
    pub epsilon_log_step: f64,
    pub min_epsilon: f64,
    pub weight_floor: f64,
    pub weight_ceiling: f64,
}

impl Default for AdaptiveRegularizationOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            max_mm_iter: 10,
            beta_rel_tol: 1e-3,
            max_epsilon_outer_iter: 4,
            epsilon_log_step: std::f64::consts::LN_2,
            min_epsilon: 1e-8,
            weight_floor: 1e-8,
            weight_ceiling: 1e8,
        }
    }
}

/// Post-fit artifacts needed by downstream diagnostics/inference without
/// re-running PIRLS.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct FitArtifacts {
    #[serde(default, skip_serializing, skip_deserializing)]
    pub pirls: Option<crate::pirls::PirlsResult>,
    #[serde(default)]
    pub null_space_logdet: Option<f64>,
    #[serde(default)]
    pub null_space_dim: Option<usize>,
    #[serde(default)]
    pub survival_link_wiggle_knots: Option<Array1<f64>>,
    #[serde(default)]
    pub survival_link_wiggle_degree: Option<usize>,
    /// First-order optimality certificate from the outer smoothing-parameter
    /// optimization (#934): gradient-vs-objective FD audit at the returned
    /// optimum, Hessian-PD probe, λ-rail flags. `None` when the outer ran
    /// gradient-free or an audit probe could not evaluate.
    #[serde(default)]
    pub criterion_certificate: Option<CriterionCertificate>,
    /// Tier-0 marginal-smoothing (`ρ`-uncertainty) PSIS certificate (#938):
    /// the Pareto-`k̂` diagnostic that says whether the plug-in + first-order
    /// `V_ρ` correction is adequate or `ρ`-uncertainty needs a heavier
    /// quadrature/NUTS treatment. Computed against the live REML objective at
    /// the converged `ρ̂` (see `RemlState::rho_posterior_inference`). `None`
    /// when there are no smoothing parameters or the outer Hessian was
    /// unavailable. Re-derivable from the fit, so it is not serialized.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub rho_posterior_certificate: Option<crate::inference::rho_posterior::RhoPosteriorCertificate>,
    /// Escalation outcome (#938) when the Tier-0 certificate read `Escalate`:
    /// the Tier-1 quadrature mixture (`K ≤ 4`), the Tier-2 NUTS draws
    /// (`K ≤ 16`), or an honest `Unavailable` report. `None` whenever the
    /// certificate did not escalate (or is itself absent). Computed at the same
    /// live-objective seam as the certificate; re-derivable, not serialized.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub rho_posterior_escalation: Option<crate::inference::rho_posterior::RhoPosteriorEscalation>,
    /// Regularized inverse REML/LAML outer Hessian over `rho = log(lambda)`,
    /// aligned with [`UnifiedFitResult::lambdas`]. This is the narrow #740
    /// handoff consumed by estimated-lambda Lawley LR corrections; it is
    /// computed from the same path as smoothing-parameter uncertainty and is
    /// re-derivable, so it is not serialized.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub rho_covariance: Option<Array2<f64>>,
}

impl std::fmt::Debug for FitArtifacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FitArtifacts")
            .field("pirls", &self.pirls.as_ref().map(|_| "..."))
            .field("null_space_logdet", &self.null_space_logdet)
            .field("null_space_dim", &self.null_space_dim)
            .field(
                "survival_link_wiggle_knots",
                &self
                    .survival_link_wiggle_knots
                    .as_ref()
                    .map(|knots| knots.len()),
            )
            .field(
                "survival_link_wiggle_degree",
                &self.survival_link_wiggle_degree,
            )
            .field("criterion_certificate", &self.criterion_certificate)
            .field("rho_posterior_certificate", &self.rho_posterior_certificate)
            .field("rho_posterior_escalation", &self.rho_posterior_escalation)
            .field(
                "rho_covariance",
                &self.rho_covariance.as_ref().map(|m| m.dim()),
            )
            .finish()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitInference {
    pub edf_by_block: Vec<f64>,
    /// Raw per-penalty-block trace `tr_kk = λ_kk·tr(H⁻¹ S_kk)`, one entry per
    /// smoothing parameter (aligned 1:1 with `lambdas`, like `edf_by_block`).
    /// Unclamped, in either coefficient basis (the trace of a matrix product is
    /// basis-invariant). This is the quantity both the dense and survival EDF
    /// paths already form internally; storing it lets per-term EDF be assembled
    /// as `edf_term = |coeff_range| − Σ_{kk∈term} tr_kk`, which equals the trace
    /// of the influence matrix `F = H⁻¹X'WX` over the term's coefficient block,
    /// is additive across terms, and sums exactly to `edf_total`. The legacy
    /// `Σ_kk(rank(S_kk) − tr_kk)` block-sum over-counts whenever several
    /// penalties share one coefficient range (`te`/`ti`, anisotropic, adaptive),
    /// reporting a per-term EDF that can exceed the model total (issue #1219).
    /// May be empty for fits produced before this field existed or by paths that
    /// do not record traces; consumers fall back to `coefficient_influence`.
    #[serde(default)]
    pub penalty_block_trace: Vec<f64>,
    pub edf_total: f64,
    pub smoothing_correction: Option<Array2<f64>>,
    /// Raw penalised Hessian `H = X'W_HX + S(λ)` with NO dispersion scaling.
    /// Stored as [`UnscaledPrecision`] so callers that need the φ-scaled
    /// covariance `Vb` know they must pair this with [`Self::dispersion`].
    /// `#[serde(transparent)]` on the newtype keeps the on-disk encoding
    /// identical to the pre-newtype `Array2<f64>` storage.
    pub penalized_hessian: crate::inference::dispersion_cov::UnscaledPrecision,
    pub working_weights: Array1<f64>,
    pub working_response: Array1<f64>,
    pub reparam_qs: Option<Array2<f64>>,
    /// Dispersion/scale used to scale all coefficient covariance matrices.
    #[serde(default)]
    pub dispersion: Dispersion,
    /// Conditional Bayesian covariance under fixed smoothing parameters (mgcv
    /// `Vb`): `Vb = H^{-1} * phi`, where `H = X'W_HX + S(lambda)` and `phi`
    /// is [`dispersion`](Self::dispersion). Do not use an unscaled `H^{-1}`
    /// for standard errors when scale is estimated.
    pub beta_covariance: Option<crate::inference::dispersion_cov::PhiScaledCovariance>,
    /// Marginal SEs from `beta_covariance`.
    pub beta_standard_errors: Option<Array1<f64>>,
    /// Optional smoothing-parameter-corrected Bayesian covariance (mgcv `Vp`):
    /// `Vp = Vb + V_lambda`, on the same dispersion scale as `Vb`. Usually
    /// this is first-order: `Var*(β) ≈ Var(β|λ) + J Var(ρ) J^T`; high-risk
    /// regimes may use adaptive cubature for higher-order terms.
    pub beta_covariance_corrected: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance_corrected` (`Vp`).
    pub beta_standard_errors_corrected: Option<Array1<f64>>,
    /// Frequentist covariance Ve = H⁻¹ X'WX H⁻¹ * φ̂.
    #[serde(default)]
    pub beta_covariance_frequentist: Option<Array2<f64>>,
    /// Coefficient-space influence matrix F = H⁻¹ X'WX. Its trace is the total EDF.
    #[serde(default)]
    pub coefficient_influence: Option<Array2<f64>>,
    /// Weighted Gram `X'WX = H − S(λ)` in the original coefficient basis —
    /// symmetric PSD by construction. Stored directly (issue #1027) so the
    /// Wood–Pya–Säfken corrected-EDF correction `tr(X'WX·Σ_ρ)` pairs the true
    /// PSD Gram with `Σ_ρ`, rather than reconstructing it as `H·F` from a
    /// Hessian surface that need not satisfy `H·F = X'WX` (which made the
    /// correction indefinite and the corrected EDF drop below the conditional).
    #[serde(default)]
    pub weighted_gram: Option<Array2<f64>>,
    /// O(n⁻¹) frequentist bias-correction vector b̂ = H⁻¹ S(λ̂) β̂ in the
    /// original (untransformed) coefficient basis. Predictions apply
    /// η̂_BC(x) = η̂(x) + s_*(x)^T b̂ to remove first-order shrinkage bias.
    #[serde(default)]
    pub bias_correction_beta: Option<Array1<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FittedLinkState {
    Standard(Option<StandardLink>),
    LatentCLogLog {
        state: LatentCLogLogState,
    },
    Sas {
        state: SasLinkState,
        covariance: Option<Array2<f64>>,
    },
    BetaLogistic {
        state: SasLinkState,
        covariance: Option<Array2<f64>>,
    },
    Mixture {
        state: MixtureLinkState,
        covariance: Option<Array2<f64>>,
    },
}

impl Default for FittedLinkState {
    fn default() -> Self {
        FittedLinkState::Standard(None)
    }
}

pub fn saved_mixture_state_from_fit(fit: &UnifiedFitResult) -> Option<MixtureLinkState> {
    match &fit.fitted_link {
        FittedLinkState::Mixture { state, .. } => Some(state.clone()),
        _ => None,
    }
}

pub fn saved_latent_cloglog_state_from_fit(fit: &UnifiedFitResult) -> Option<LatentCLogLogState> {
    match &fit.fitted_link {
        FittedLinkState::LatentCLogLog { state } => Some(*state),
        _ => None,
    }
}

pub fn saved_sas_state_from_fit(fit: &UnifiedFitResult) -> Option<SasLinkState> {
    match &fit.fitted_link {
        FittedLinkState::Sas { state, .. } | FittedLinkState::BetaLogistic { state, .. } => {
            Some(*state)
        }
        _ => None,
    }
}

pub(crate) fn validate_fitted_link_estimation(
    fitted_link: &FittedLinkState,
) -> Result<(), EstimationError> {
    match fitted_link {
        FittedLinkState::Standard(_) => Ok(()),
        FittedLinkState::LatentCLogLog { state } => {
            ensure_finite_scalar_estimation("fit_result.latent_cloglog.latent_sd", state.latent_sd)
        }
        FittedLinkState::Mixture { state, covariance } => {
            validate_all_finite_estimation(
                "fit_result.mixture_link_rho",
                state.rho.iter().copied(),
            )?;
            validate_all_finite_estimation(
                "fit_result.mixture_linkweights",
                state.pi.iter().copied(),
            )?;
            if let Some(v) = covariance.as_ref() {
                validate_all_finite_estimation(
                    "fit_result.mixture_link_param_covariance",
                    v.iter().copied(),
                )?;
            }
            Ok(())
        }
        FittedLinkState::Sas { state, covariance }
        | FittedLinkState::BetaLogistic { state, covariance } => {
            ensure_finite_scalar_estimation("fit_result.sas_epsilon", state.epsilon)?;
            ensure_finite_scalar_estimation("fit_result.sas_log_delta", state.log_delta)?;
            ensure_finite_scalar_estimation("fit_result.sas_delta", state.delta)?;
            if let Some(v) = covariance.as_ref() {
                validate_all_finite_estimation(
                    "fit_result.sas_param_covariance",
                    v.iter().copied(),
                )?;
            }
            Ok(())
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Unified fit result — single type for all model families
// ═══════════════════════════════════════════════════════════════════════════

/// Role of a coefficient block within a multi-parameter model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockRole {
    /// Single-parameter GAM (standard GLM/GAM mean model).
    Mean,
    /// Location parameter in GAMLSS / survival location-scale.
    Location,
    /// Scale (log-sigma) parameter in GAMLSS / survival location-scale.
    Scale,
    /// Time/baseline hazard block in survival models.
    Time,
    /// Threshold block in survival models.
    Threshold,
    /// Link-wiggle correction block.
    LinkWiggle,
}

impl BlockRole {
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Location => "location",
            Self::Scale => "scale",
            Self::Time => "time",
            Self::Threshold => "threshold",
            Self::LinkWiggle => "link-wiggle",
        }
    }
}

/// Inference quantities for one coefficient block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FittedBlock {
    /// Coefficients at the converged mode.
    pub beta: Array1<f64>,
    /// Role of this block within the model.
    pub role: BlockRole,
    /// Effective degrees of freedom (sum of leverages).
    pub edf: f64,
    /// Smoothing parameters for this block.
    pub lambdas: Array1<f64>,
}

/// Working-set geometry at convergence needed by ALO and other post-fit
/// diagnostics. Only populated when the inner solver provides the data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitGeometry {
    /// Joint penalized Hessian `H = X'W_HX + S(λ)` at convergence.
    /// Stored as [`UnscaledPrecision`] so the dispersion-ownership invariant
    /// (this matrix is *not* φ-scaled) is enforced at the type level.
    pub penalized_hessian: crate::inference::dispersion_cov::UnscaledPrecision,
    /// Score-side Fisher IRLS weights paired with `working_response`.
    pub working_weights: Array1<f64>,
    /// IRLS working response at convergence.
    pub working_response: Array1<f64>,
}

pub struct UnifiedFitResultParts {
    pub blocks: Vec<FittedBlock>,
    pub log_lambdas: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub likelihood_family: Option<LikelihoodSpec>,
    pub likelihood_scale: LikelihoodScaleMetadata,
    pub log_likelihood_normalization: LogLikelihoodNormalization,
    pub log_likelihood: f64,
    pub deviance: f64,
    pub reml_score: f64,
    pub stable_penalty_term: f64,
    pub penalized_objective: f64,
    pub used_device: bool,
    pub outer_iterations: usize,
    pub outer_converged: bool,
    pub outer_gradient_norm: Option<f64>,
    pub standard_deviation: f64,
    pub covariance_conditional: Option<Array2<f64>>,
    pub covariance_corrected: Option<Array2<f64>>,
    pub inference: Option<FitInference>,
    pub fitted_link: FittedLinkState,
    pub geometry: Option<FitGeometry>,
    pub block_states: Vec<crate::families::custom_family::ParameterBlockState>,
    // Backward-compatible fields (all have sensible defaults).
    #[doc(hidden)]
    pub pirls_status: crate::pirls::PirlsStatus,
    #[doc(hidden)]
    pub max_abs_eta: f64,
    #[doc(hidden)]
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    #[doc(hidden)]
    pub artifacts: FitArtifacts,
    #[doc(hidden)]
    pub inner_cycles: usize,
}

/// Unified fit result for all model types (standard GAM, GAMLSS, survival).
///
/// Standard models have a single block; GAMLSS and survival models have
/// multiple blocks with different roles.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedFitResult {
    // ── canonical fields ──────────────────────────────────────────────────
    /// Coefficient blocks (1 for standard GAM, N for GAMLSS/survival).
    pub blocks: Vec<FittedBlock>,
    /// Log-smoothing parameters (all blocks concatenated in block order).
    pub log_lambdas: Array1<f64>,
    /// Smoothing parameters (exp of log_lambdas).
    pub lambdas: Array1<f64>,
    /// Explicit engine-level family, when the fit uses a built-in family.
    pub likelihood_family: Option<LikelihoodSpec>,
    /// Fixed-scale metadata for the fitted likelihood.
    pub likelihood_scale: LikelihoodScaleMetadata,
    /// Whether `log_likelihood` includes response-only normalization constants.
    pub log_likelihood_normalization: LogLikelihoodNormalization,
    /// Log-likelihood at the converged mode.
    pub log_likelihood: f64,
    /// Explicit deviance reported by the fitting path.
    pub deviance: f64,
    /// Complete REML/LAML objective value used for smoothing selection.
    pub reml_score: f64,
    /// Stable quadratic penalty term βᵀSβ, including any solver ridge quadratic.
    pub stable_penalty_term: f64,
    /// Public objective value reported for the fit. For REML/LAML fits this is
    /// the same complete objective as `reml_score`, not `-ℓ + penalty + reml_score`.
    pub penalized_objective: f64,
    /// Whether the converged fit used a GPU execution path for its final inner solve.
    #[serde(default)]
    pub used_device: bool,
    /// Number of outer (smoothing parameter) iterations.
    pub outer_iterations: usize,
    /// Whether the outer optimization converged.
    pub outer_converged: bool,
    /// Final gradient norm of the outer optimization. `None` when no
    /// gradient was measured at termination — cache-hit short-circuit
    /// (the prior fit's converged ρ was loaded from disk), gradient-free
    /// solver, or a degenerate early-exit path where no outer ran.
    /// `outer_converged` is the authoritative convergence signal.
    pub outer_gradient_norm: Option<f64>,
    /// Residual scale on the response scale.
    ///
    /// Contract: Gaussian identity models store residual standard deviation
    /// sigma here. Non-Gaussian families keep the response-scale summary used
    /// by their explicit likelihood-scale metadata.
    pub standard_deviation: f64,
    /// Vb: Bayesian/conditional covariance Var(β | λ) = H⁻¹ * φ̂ for the joint coefficient vector.
    pub covariance_conditional: Option<Array2<f64>>,
    /// Vp: Bayesian covariance with smoothing-parameter uncertainty correction.
    pub covariance_corrected: Option<Array2<f64>>,
    /// Inference quantities from the inner solver (EDF, Hessian, etc.).
    pub inference: Option<FitInference>,
    /// Fitted link parameters (SAS, BetaLogistic, Mixture).
    pub fitted_link: FittedLinkState,
    /// Working-set geometry at convergence (for ALO diagnostics and
    /// saved-model covariance reconstruction).
    pub geometry: Option<FitGeometry>,
    /// Internal block states from custom-family paths.
    #[serde(skip)]
    pub block_states: Vec<crate::families::custom_family::ParameterBlockState>,
    /// Joint coefficient vector (first block for standard GAMs, concatenated for multi-block).
    #[serde(default)]
    pub beta: Array1<f64>,
    /// Inner solver convergence status. Required at decode time: a missing
    /// field on an older-schema or corrupted saved model previously decoded
    /// as `Converged` via a default, silently promoting non-converged β̂
    /// through warm-start propagation, predict-time confidence intervals,
    /// and outer-loop convergence semantics. With the MODEL_PAYLOAD_VERSION
    /// gate in place, older schemas are rejected before this field is read,
    /// so requiring the field here is safe and strictly removes the silent
    /// default.
    pub pirls_status: crate::pirls::PirlsStatus,
    /// Maximum absolute linear predictor value at convergence.
    #[serde(default)]
    pub max_abs_eta: f64,
    /// Constraint KKT diagnostics (monotone-constrained fits).
    #[serde(default)]
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    /// Solver artifacts (e.g. cached PIRLS result for ALO).
    #[serde(default)]
    pub artifacts: FitArtifacts,
    /// Inner cycle count (blockwise path).
    #[serde(default)]
    pub inner_cycles: usize,
}

pub(crate) fn ensure_finite_scalar_estimation(
    name: &str,
    value: f64,
) -> Result<(), EstimationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{name} must be finite, got {value}"
        )))
    }
}

fn validate_likelihood_scale_estimation(
    scale: LikelihoodScaleMetadata,
) -> Result<(), EstimationError> {
    match scale {
        LikelihoodScaleMetadata::ProfiledGaussian | LikelihoodScaleMetadata::Unspecified => Ok(()),
        LikelihoodScaleMetadata::FixedDispersion { phi }
        | LikelihoodScaleMetadata::EstimatedBetaPhi { phi }
        | LikelihoodScaleMetadata::EstimatedTweediePhi { phi } => {
            ensure_finite_scalar_estimation("fit_result.likelihood_scale.phi", phi)?;
            if phi > 0.0 {
                Ok(())
            } else {
                Err(EstimationError::InvalidInput(format!(
                    "fit_result.likelihood_scale.phi must be > 0, got {phi}"
                )))
            }
        }
        LikelihoodScaleMetadata::FixedGammaShape { shape }
        | LikelihoodScaleMetadata::EstimatedGammaShape { shape } => {
            ensure_finite_scalar_estimation("fit_result.likelihood_scale.shape", shape)?;
            if shape > 0.0 {
                Ok(())
            } else {
                Err(EstimationError::InvalidInput(format!(
                    "fit_result.likelihood_scale.shape must be > 0, got {shape}"
                )))
            }
        }
        // A user-fixed θ (#983) carries the identical positivity contract as an
        // estimated one — only the PIRLS refresh gate differs, not the validity
        // of the recorded value.
        LikelihoodScaleMetadata::EstimatedNegBinTheta { theta }
        | LikelihoodScaleMetadata::FixedNegBinTheta { theta } => {
            ensure_finite_scalar_estimation("fit_result.likelihood_scale.theta", theta)?;
            if theta > 0.0 {
                Ok(())
            } else {
                Err(EstimationError::InvalidInput(format!(
                    "fit_result.likelihood_scale.theta must be > 0, got {theta}"
                )))
            }
        }
    }
}

pub(crate) fn validate_all_finite_estimation<I>(
    label: &str,
    values: I,
) -> Result<(), EstimationError>
where
    I: IntoIterator<Item = f64>,
{
    for (idx, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            crate::bail_invalid_estim!("{label}[{idx}] must be finite, got {value}");
        }
    }
    Ok(())
}

/// Public wrapper returning `String` errors for use outside the estimation module.
pub fn ensure_finite_scalar(name: &str, value: f64) -> Result<(), String> {
    ensure_finite_scalar_estimation(name, value).map_err(|e| e.to_string())
}

/// Public wrapper returning `String` errors for use outside the estimation module.
pub fn validate_all_finite<I: IntoIterator<Item = f64>>(
    label: &str,
    values: I,
) -> Result<(), String> {
    validate_all_finite_estimation(label, values).map_err(|e| e.to_string())
}

impl FitGeometry {
    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        validate_all_finite_estimation(
            "fit_result.geometry.penalized_hessian",
            self.penalized_hessian.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.geometry.working_weights",
            self.working_weights.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.geometry.working_response",
            self.working_response.iter().copied(),
        )?;
        Ok(())
    }
}

impl FitInference {
    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        ensure_finite_scalar_estimation("fit_result.edf_total", self.edf_total)?;
        validate_all_finite_estimation(
            "fit_result.edf_by_block",
            self.edf_by_block.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.penalty_block_trace",
            self.penalty_block_trace.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.working_weights",
            self.working_weights.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.working_response",
            self.working_response.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.penalized_hessian",
            self.penalized_hessian.iter().copied(),
        )?;
        if let Some(v) = self.beta_covariance.as_ref() {
            validate_all_finite_estimation("fit_result.beta_covariance", v.iter().copied())?;
        }
        if let Some(v) = self.beta_covariance_corrected.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_covariance_corrected",
                v.iter().copied(),
            )?;
        }
        if let Some(v) = self.beta_standard_errors.as_ref() {
            validate_all_finite_estimation("fit_result.beta_standard_errors", v.iter().copied())?;
        }
        if let Some(v) = self.beta_covariance_frequentist.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_covariance_frequentist",
                v.iter().copied(),
            )?;
        }
        if let Some(v) = self.coefficient_influence.as_ref() {
            validate_all_finite_estimation("fit_result.coefficient_influence", v.iter().copied())?;
        }
        if let Some(v) = self.weighted_gram.as_ref() {
            validate_all_finite_estimation("fit_result.weighted_gram", v.iter().copied())?;
        }
        if let Some(v) = self.bias_correction_beta.as_ref() {
            validate_all_finite_estimation("fit_result.bias_correction_beta", v.iter().copied())?;
        }
        if let Some(v) = self.beta_standard_errors_corrected.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_standard_errors_corrected",
                v.iter().copied(),
            )?;
        }
        if let Some(v) = self.beta_covariance_frequentist.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_covariance_frequentist",
                v.iter().copied(),
            )?;
        }
        if let Some(v) = self.smoothing_correction.as_ref() {
            validate_all_finite_estimation("fit_result.smoothing_correction", v.iter().copied())?;
        }
        if let Some(v) = self.reparam_qs.as_ref() {
            validate_all_finite_estimation("fit_result.reparam_qs", v.iter().copied())?;
        }
        Ok(())
    }
}

/// Validate the *structural integrity* of an exported penalized Hessian.
///
/// Checks shape, finiteness, non-zero (no placeholder), and symmetry. This is
/// the right gate for fit-export: every consumer (HMC, sampling, covariance
/// inversion, diagnostics) needs these invariants, and the cost is `O(p²)`
/// once at construction.
///
/// **Does not** check positive definiteness.  Strict-PD via bare Cholesky is
/// too narrow a gate for fit-export: legitimate fits can produce penalized
/// Hessians that are positive *semi*-definite — boundary-projected
/// coefficients in structurally constrained blocks lose curvature in active
/// directions; partially converged outer fits (small `outer_max_iter`) may
/// still have negative diagonal entries; rank-deficient penalty subspaces
/// require an LM δ-ridge that the inner solver applies during the fit but
/// that is not (and should not be) baked into the exported `H + Σ λ_k S_k`.
/// Whether strict-PD is required is a *consumer* property — see
/// [`validate_explicit_dense_hessian_for_whitening`] for the HMC-side gate.
pub fn validate_dense_hessian_export(
    label: &str,
    hessian: &Array2<f64>,
    expected_dim: usize,
) -> Result<(), EstimationError> {
    if hessian.nrows() != expected_dim || hessian.ncols() != expected_dim {
        crate::bail_invalid_estim!(
            "{label} shape mismatch: got {}x{}, expected {}x{}",
            hessian.nrows(),
            hessian.ncols(),
            expected_dim,
            expected_dim
        );
    }
    if expected_dim == 0 {
        return Ok(());
    }
    validate_all_finite_estimation(label, hessian.iter().copied())?;
    if !hessian.iter().any(|value| value.abs() > 0.0) {
        crate::bail_invalid_estim!(
            "{label} must be an explicit dense Hessian; zero placeholders are not allowed at fit export"
        );
    }
    let symmetry_tol = 1e-10;
    for i in 0..expected_dim {
        for j in 0..i {
            let a = hessian[[i, j]];
            let b = hessian[[j, i]];
            let scale = 1.0_f64.max(a.abs()).max(b.abs());
            if (a - b).abs() > symmetry_tol * scale {
                crate::bail_invalid_estim!(
                    "{label} must be symmetric at fit export; entries ({i},{j})={a} and ({j},{i})={b} differ"
                );
            }
        }
    }
    Ok(())
}

/// Validate that a saved penalized Hessian is an explicit dense precision
/// matrix suitable for HMC/NUTS whitening.
///
/// The HMC path whitens with a Cholesky factor of this matrix, so HMC's own
/// entry layer must reject placeholders, missing curvature hidden behind a
/// covariance, nonsymmetric, or non-SPD matrices. This check is intentionally
/// the strictest of the validation chain — it composes the structural gate
/// from [`validate_dense_hessian_export`] with a bare Cholesky that does not
/// add a δ-ridge (HMC's whitening Jacobian is sensitive to any artificial
/// floor).  Call this from the HMC entry, not from `try_from_parts`: not
/// every fit is consumed by HMC, and rejecting partially-converged or
/// boundary-projected fits at construction would block legitimate non-HMC
/// downstream uses.
pub fn validate_explicit_dense_hessian_for_whitening(
    label: &str,
    hessian: &Array2<f64>,
    expected_dim: usize,
) -> Result<(), EstimationError> {
    validate_dense_hessian_export(label, hessian, expected_dim)?;
    if expected_dim == 0 {
        return Ok(());
    }
    hessian
        .to_owned()
        .cholesky(Side::Lower)
        .map(|_| ())
        .map_err(|err| {
            EstimationError::InvalidInput(format!(
                "{label} must be positive definite for HMC/NUTS whitening; Cholesky failed: {err:?}"
            ))
        })
}

fn log_lambdas_match_lambdas(log_lambdas: &Array1<f64>, lambdas: &Array1<f64>) -> bool {
    if log_lambdas.len() != lambdas.len() {
        return false;
    }
    log_lambdas
        .iter()
        .zip(lambdas.iter())
        .all(|(&log_lam, &lam)| {
            let canonical = lam.max(1e-300).ln();
            let tol = 1e-12 * (1.0 + canonical.abs());
            (log_lam - canonical).abs() <= tol
        })
}

/// Vertically stack a per-block `Array1<f64>` field (selected by `field`) into
/// one contiguous vector, in block order. Single helper shared by the β and λ
/// flatteners, routed through the canonical [`stack_offsets`] concatenation.
fn flatten_blocks_field(
    blocks: &[FittedBlock],
    field: impl Fn(&FittedBlock) -> &Array1<f64>,
) -> Array1<f64> {
    let parts: Vec<&Array1<f64>> = blocks.iter().map(field).collect();
    stack_offsets(&parts)
}

fn flatten_block_betas(blocks: &[FittedBlock]) -> Array1<f64> {
    flatten_blocks_field(blocks, |b| &b.beta)
}

fn flatten_block_lambdas(blocks: &[FittedBlock]) -> Array1<f64> {
    flatten_blocks_field(blocks, |b| &b.lambdas)
}

impl UnifiedFitResult {
    pub fn try_from_parts(parts: UnifiedFitResultParts) -> Result<Self, EstimationError> {
        let UnifiedFitResultParts {
            blocks,
            log_lambdas,
            lambdas,
            likelihood_family,
            likelihood_scale,
            log_likelihood_normalization,
            log_likelihood,
            deviance,
            reml_score,
            stable_penalty_term,
            penalized_objective,
            used_device,
            outer_iterations,
            outer_converged,
            outer_gradient_norm,
            standard_deviation,
            covariance_conditional,
            covariance_corrected,
            inference,
            fitted_link,
            geometry,
            block_states,
            pirls_status,
            max_abs_eta,
            constraint_kkt,
            artifacts,
            inner_cycles,
        } = parts;

        if blocks.is_empty() {
            crate::bail_invalid_estim!("UnifiedFitResult requires at least one coefficient block");
        }
        if log_lambdas.len() != lambdas.len() {
            crate::bail_invalid_estim!(
                "UnifiedFitResult lambda mismatch: log_lambdas={}, lambdas={}",
                log_lambdas.len(),
                lambdas.len()
            );
        }
        for (idx, block) in blocks.iter().enumerate() {
            validate_all_finite_estimation(
                &format!("fit_result.blocks[{idx}].beta"),
                block.beta.iter().copied(),
            )?;
            ensure_finite_scalar_estimation(&format!("fit_result.blocks[{idx}].edf"), block.edf)?;
            validate_all_finite_estimation(
                &format!("fit_result.blocks[{idx}].lambdas"),
                block.lambdas.iter().copied(),
            )?;
        }
        let beta = flatten_block_betas(&blocks);
        let block_lambdas = flatten_block_lambdas(&blocks);
        if block_lambdas != lambdas {
            crate::bail_invalid_estim!("UnifiedFitResult top-level lambdas must match block lambdas concatenated in block order"
                    .to_string(),);
        }
        validate_all_finite_estimation("fit_result.log_lambdas", log_lambdas.iter().copied())?;
        validate_all_finite_estimation("fit_result.lambdas", lambdas.iter().copied())?;
        if !log_lambdas_match_lambdas(&log_lambdas, &lambdas) {
            crate::bail_invalid_estim!(
                "UnifiedFitResult log_lambdas must equal ln(lambdas) elementwise"
            );
        }
        validate_likelihood_scale_estimation(likelihood_scale)?;
        ensure_finite_scalar_estimation("fit_result.log_likelihood", log_likelihood)?;
        ensure_finite_scalar_estimation("fit_result.deviance", deviance)?;
        ensure_finite_scalar_estimation("fit_result.reml_score", reml_score)?;
        ensure_finite_scalar_estimation("fit_result.stable_penalty_term", stable_penalty_term)?;
        ensure_finite_scalar_estimation("fit_result.penalized_objective", penalized_objective)?;
        if let Some(g) = outer_gradient_norm {
            ensure_finite_scalar_estimation("fit_result.outer_gradient_norm", g)?;
        }
        ensure_finite_scalar_estimation("fit_result.standard_deviation", standard_deviation)?;
        if let Some(v) = covariance_conditional.as_ref() {
            validate_all_finite_estimation("fit_result.beta_covariance", v.iter().copied())?;
        }
        if let Some(v) = covariance_corrected.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_covariance_corrected",
                v.iter().copied(),
            )?;
        }
        if let Some(inf) = inference.as_ref() {
            inf.validate_numeric_finiteness()?;
        }
        if let Some(geom) = geometry.as_ref() {
            geom.validate_numeric_finiteness()?;
        }
        for (idx, state) in block_states.iter().enumerate() {
            validate_all_finite_estimation(
                &format!("fit_result.block_states[{idx}].beta"),
                state.beta.iter().copied(),
            )?;
            validate_all_finite_estimation(
                &format!("fit_result.block_states[{idx}].eta"),
                state.eta.iter().copied(),
            )?;
        }
        validate_fitted_link_estimation(&fitted_link)?;

        let p = beta.len();
        if let Some(cov) = covariance_conditional.as_ref()
            && (cov.nrows() != p || cov.ncols() != p)
        {
            crate::bail_invalid_estim!(
                "UnifiedFitResult conditional covariance shape mismatch: got {}x{}, expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                p,
                p
            );
        }
        if let Some(cov) = covariance_corrected.as_ref()
            && (cov.nrows() != p || cov.ncols() != p)
        {
            crate::bail_invalid_estim!(
                "UnifiedFitResult corrected covariance shape mismatch: got {}x{}, expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                p,
                p
            );
        }
        if let Some(inf) = inference.as_ref() {
            if !inf.edf_by_block.is_empty() && inf.edf_by_block.len() != lambdas.len() {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult EDF smoothing-parameter count mismatch: edf_by_block={}, lambdas={}",
                    inf.edf_by_block.len(),
                    lambdas.len()
                );
            }
            if !inf.penalty_block_trace.is_empty() && inf.penalty_block_trace.len() != lambdas.len()
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult EDF smoothing-parameter count mismatch: penalty_block_trace={}, lambdas={}",
                    inf.penalty_block_trace.len(),
                    lambdas.len()
                );
            }
            if inf.working_weights.len() != inf.working_response.len() {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult working vector length mismatch: working_weights={}, working_response={}",
                    inf.working_weights.len(),
                    inf.working_response.len()
                );
            }
            if inf.penalized_hessian.nrows() != p || inf.penalized_hessian.ncols() != p {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult penalized Hessian shape mismatch: got {}x{}, expected {}x{}",
                    inf.penalized_hessian.nrows(),
                    inf.penalized_hessian.ncols(),
                    p,
                    p
                );
            }
            validate_dense_hessian_export(
                "UnifiedFitResult inference penalized Hessian",
                &inf.penalized_hessian,
                p,
            )?;
            if let Some(cov) = inf.beta_covariance.as_ref() {
                if cov.nrows() != p || cov.ncols() != p {
                    crate::bail_invalid_estim!(
                        "UnifiedFitResult inference conditional covariance shape mismatch: got {}x{}, expected {}x{}",
                        cov.nrows(),
                        cov.ncols(),
                        p,
                        p
                    );
                }
                match covariance_conditional.as_ref() {
                    Some(top) if **cov == *top => {}
                    Some(_) => {
                        crate::bail_invalid_estim!("UnifiedFitResult inference conditional covariance must match top-level covariance_conditional"
                                .to_string(),);
                    }
                    None => {
                        crate::bail_invalid_estim!("UnifiedFitResult inference conditional covariance requires top-level covariance_conditional"
                                .to_string(),);
                    }
                }
            }
            if let Some(se) = inf.beta_standard_errors.as_ref()
                && se.len() != p
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult beta standard error length mismatch: got {}, expected {}",
                    se.len(),
                    p
                );
            }
            if let Some(cov) = inf.beta_covariance_corrected.as_ref() {
                if cov.nrows() != p || cov.ncols() != p {
                    crate::bail_invalid_estim!(
                        "UnifiedFitResult inference corrected covariance shape mismatch: got {}x{}, expected {}x{}",
                        cov.nrows(),
                        cov.ncols(),
                        p,
                        p
                    );
                }
                match covariance_corrected.as_ref() {
                    Some(top) if **cov == *top => {}
                    Some(_) => {
                        crate::bail_invalid_estim!("UnifiedFitResult inference corrected covariance must match top-level covariance_corrected"
                                .to_string(),);
                    }
                    None => {
                        crate::bail_invalid_estim!("UnifiedFitResult inference corrected covariance requires top-level covariance_corrected"
                                .to_string(),);
                    }
                }
            }
            if let Some(se) = inf.beta_standard_errors_corrected.as_ref()
                && se.len() != p
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult corrected beta standard error length mismatch: got {}, expected {}",
                    se.len(),
                    p
                );
            }
            if let Some(cov) = inf.beta_covariance_frequentist.as_ref()
                && (cov.nrows() != p || cov.ncols() != p)
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult frequentist covariance shape mismatch: got {}x{}, expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    p,
                    p
                );
            }
            if let Some(f_mat) = inf.coefficient_influence.as_ref()
                && (f_mat.nrows() != p || f_mat.ncols() != p)
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult coefficient influence shape mismatch: got {}x{}, expected {}x{}",
                    f_mat.nrows(),
                    f_mat.ncols(),
                    p,
                    p
                );
            }
            if let Some(corr) = inf.smoothing_correction.as_ref()
                && (corr.nrows() != p || corr.ncols() != p)
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult smoothing correction shape mismatch: got {}x{}, expected {}x{}",
                    corr.nrows(),
                    corr.ncols(),
                    p,
                    p
                );
            }
            if let Some(qs) = inf.reparam_qs.as_ref()
                && (qs.nrows() != p || qs.ncols() != p)
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult reparam_qs shape mismatch: got {}x{}, expected {}x{}",
                    qs.nrows(),
                    qs.ncols(),
                    p,
                    p
                );
            }
        }
        if let Some(geom) = geometry.as_ref() {
            if geom.penalized_hessian.nrows() != p || geom.penalized_hessian.ncols() != p {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult geometry penalized Hessian shape mismatch: got {}x{}, expected {}x{}",
                    geom.penalized_hessian.nrows(),
                    geom.penalized_hessian.ncols(),
                    p,
                    p
                );
            }
            validate_dense_hessian_export(
                "UnifiedFitResult geometry penalized Hessian",
                &geom.penalized_hessian,
                p,
            )?;
            if geom.working_weights.len() != geom.working_response.len() {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult geometry working vector length mismatch: working_weights={}, working_response={}",
                    geom.working_weights.len(),
                    geom.working_response.len()
                );
            }
            if let Some(inf) = inference.as_ref() {
                if geom.penalized_hessian != inf.penalized_hessian {
                    crate::bail_invalid_estim!("UnifiedFitResult geometry penalized Hessian must match inference.penalized_hessian"
                            .to_string(),);
                }
                if geom.working_weights != inf.working_weights {
                    crate::bail_invalid_estim!("UnifiedFitResult geometry working_weights must match inference.working_weights"
                            .to_string(),);
                }
                if geom.working_response != inf.working_response {
                    crate::bail_invalid_estim!("UnifiedFitResult geometry working_response must match inference.working_response"
                            .to_string(),);
                }
            }
        }
        if !block_states.is_empty() && block_states.len() != blocks.len() {
            crate::bail_invalid_estim!(
                "UnifiedFitResult block state count mismatch: blocks={}, block_states={}",
                blocks.len(),
                block_states.len()
            );
        }

        Ok(Self {
            blocks,
            log_lambdas,
            lambdas,
            likelihood_family,
            likelihood_scale,
            log_likelihood_normalization,
            log_likelihood,
            deviance,
            reml_score,
            stable_penalty_term,
            penalized_objective,
            used_device,
            outer_iterations,
            outer_converged,
            outer_gradient_norm,
            standard_deviation,
            covariance_conditional,
            covariance_corrected,
            inference,
            fitted_link,
            geometry,
            block_states,
            beta,
            pirls_status,
            max_abs_eta,
            constraint_kkt,
            artifacts,
            inner_cycles,
        })
    }
    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        let expected_beta = flatten_block_betas(&self.blocks);
        if self.beta != expected_beta {
            crate::bail_invalid_estim!("UnifiedFitResult decoded beta must match coefficient blocks concatenated in block order"
                    .to_string(),);
        }
        Self::try_from_parts(UnifiedFitResultParts {
            blocks: self.blocks.clone(),
            log_lambdas: self.log_lambdas.clone(),
            lambdas: self.lambdas.clone(),
            likelihood_family: self.likelihood_family.clone(),
            likelihood_scale: self.likelihood_scale,
            log_likelihood_normalization: self.log_likelihood_normalization,
            log_likelihood: self.log_likelihood,
            deviance: self.deviance,
            reml_score: self.reml_score,
            stable_penalty_term: self.stable_penalty_term,
            penalized_objective: self.penalized_objective,
            used_device: self.used_device,
            outer_iterations: self.outer_iterations,
            outer_converged: self.outer_converged,
            outer_gradient_norm: self.outer_gradient_norm,
            standard_deviation: self.standard_deviation,
            covariance_conditional: self.covariance_conditional.clone(),
            covariance_corrected: self.covariance_corrected.clone(),
            inference: self.inference.clone(),
            fitted_link: self.fitted_link.clone(),
            geometry: self.geometry.clone(),
            block_states: self.block_states.clone(),
            pirls_status: self.pirls_status,
            max_abs_eta: self.max_abs_eta,
            constraint_kkt: self.constraint_kkt.clone(),
            artifacts: self.artifacts.clone(),
            inner_cycles: self.inner_cycles,
        })
        .map(|_| ())
    }
}

impl UnifiedFitResult {
    pub fn new_for_test_unchecked(parts: UnifiedFitResultParts) -> Self {
        let beta = flatten_block_betas(&parts.blocks);
        Self {
            blocks: parts.blocks,
            log_lambdas: parts.log_lambdas,
            lambdas: parts.lambdas,
            likelihood_family: parts.likelihood_family,
            likelihood_scale: parts.likelihood_scale,
            log_likelihood_normalization: parts.log_likelihood_normalization,
            log_likelihood: parts.log_likelihood,
            deviance: parts.deviance,
            reml_score: parts.reml_score,
            stable_penalty_term: parts.stable_penalty_term,
            penalized_objective: parts.penalized_objective,
            used_device: parts.used_device,
            outer_iterations: parts.outer_iterations,
            outer_converged: parts.outer_converged,
            outer_gradient_norm: parts.outer_gradient_norm,
            standard_deviation: parts.standard_deviation,
            covariance_conditional: parts.covariance_conditional,
            covariance_corrected: parts.covariance_corrected,
            inference: parts.inference,
            fitted_link: parts.fitted_link,
            geometry: parts.geometry,
            block_states: parts.block_states,
            beta,
            pirls_status: parts.pirls_status,
            max_abs_eta: parts.max_abs_eta,
            constraint_kkt: parts.constraint_kkt,
            artifacts: parts.artifacts,
            inner_cycles: parts.inner_cycles,
        }
    }
}

impl UnifiedFitResult {
    /// Get the conditional Bayesian covariance matrix (`Vb`) if available.
    ///
    /// Contract: `Vb = H^{-1} * phi`, scaled by the fitted dispersion. This is
    /// the Wood/mgcv `Vb` (Bayesian/conditional) covariance.
    pub fn beta_covariance(&self) -> Option<&Array2<f64>> {
        self.covariance_conditional.as_ref()
    }

    /// Get the frequentist sandwich covariance (`Ve`) if available.
    ///
    /// Wood/mgcv `Ve = H⁻¹ X'WX H⁻¹ * φ̂`.
    pub fn beta_covariance_ve(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.beta_covariance_frequentist.as_ref())
    }

    /// Get coefficient-space influence matrix `F = H^{-1}X'WX` if available.
    pub fn coefficient_influence(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.coefficient_influence.as_ref())
    }

    /// Get the original-basis weighted Gram `X'WX = H − S(λ)` if available —
    /// the symmetric PSD matrix the Wood–Pya–Säfken corrected-EDF correction
    /// pairs with the smoothing-parameter uncertainty covariance (issue #1027).
    pub fn weighted_gram(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.weighted_gram.as_ref())
    }

    /// Dispersion used to scale covariance matrices.
    pub fn dispersion(&self) -> Option<Dispersion> {
        self.inference.as_ref().map(|inf| inf.dispersion)
    }

    /// Canonical residual dispersion `φ̂` — the response-level observation noise
    /// (Gaussian `σ̂²`, Gamma `1/shape`, Beta `1/(1+φ)`, fixed-scale families
    /// `1`). This is the predictive observation-noise scale used to widen
    /// prediction *observation* intervals; it is NOT the coefficient-covariance
    /// scale (see [`Self::coefficient_covariance_scale`]). For families whose
    /// IRLS working weight already carries `1/φ`, the two differ: the
    /// coefficient covariance is `H⁻¹` (scale `1`) while this dispersion stays
    /// `1/shape` (#679).
    ///
    /// Unlike [`Self::dispersion`], which reads the cached `inference` block,
    /// this is computed from fields that always survive serialization
    /// (`likelihood_family`, `likelihood_scale`, `standard_deviation`). That
    /// matters for deployment-time consumers operating on a saved model whose
    /// `inference` block was dropped (e.g. `core_saved_fit_result` stores
    /// `inference: None`): the cached `dispersion()` is then `None`, but the
    /// scale is still recoverable and identical to the value used at fit time.
    /// When the cached block is present its dispersion is preferred verbatim so
    /// the two paths never diverge.
    pub fn dispersion_phi(&self) -> f64 {
        if let Some(dispersion) = self.dispersion() {
            return dispersion.phi();
        }
        match &self.likelihood_family {
            Some(spec) => {
                let glm = GlmLikelihoodSpec {
                    spec: spec.clone(),
                    scale: self.likelihood_scale.clone(),
                };
                dispersion_from_likelihood(&glm, self.standard_deviation).phi()
            }
            // No engine-level family (custom/GAMLSS paths): no scalar
            // response-scale dispersion is defined, so fall back to the
            // fixed-scale convention `φ = 1`.
            None => 1.0,
        }
    }

    /// Multiplier that turns the stored unscaled inverse penalized Hessian
    /// `H⁻¹` into the reported coefficient covariance `Vb = H⁻¹·scale`.
    ///
    /// This is the deployment-time / serialized-model counterpart of
    /// `GlmLikelihoodSpec::coefficient_covariance_scale`, used wherever the full
    /// stored `beta_covariance()` is unavailable and `Vb` must be reconstructed
    /// from the factorized Hessian (large-model predict path). It returns the
    /// profiled residual variance `σ̂²` for the scale-free profiled Gaussian and
    /// `1.0` for every family whose IRLS working weight already carries the
    /// dispersion / full Fisher information (Gamma, Tweedie, Beta,
    /// Negative-Binomial, Poisson, Binomial) — see #679. For custom/GAMLSS
    /// paths with no engine-level family it falls back to `1.0`.
    pub fn coefficient_covariance_scale(&self) -> f64 {
        match &self.likelihood_family {
            Some(spec) => {
                let glm = GlmLikelihoodSpec {
                    spec: spec.clone(),
                    scale: self.likelihood_scale.clone(),
                };
                glm.coefficient_covariance_scale(self.standard_deviation * self.standard_deviation)
            }
            None => 1.0,
        }
    }

    /// Get the smoothing-parameter-corrected beta covariance (`Vp`) if available.
    ///
    /// Wood/mgcv name for the smoothing-parameter-corrected covariance `Vp`.
    pub fn beta_covariance_corrected(&self) -> Option<&Array2<f64>> {
        self.covariance_corrected.as_ref().or_else(|| {
            self.inference
                .as_ref()
                .and_then(|inf| inf.beta_covariance_corrected.as_ref())
        })
    }

    /// Get beta standard errors (conditional) if available.
    pub fn beta_standard_errors(&self) -> Option<&Array1<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.beta_standard_errors.as_ref())
    }

    /// Get smoothing-corrected beta standard errors if available.
    pub fn beta_standard_errors_corrected(&self) -> Option<&Array1<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.beta_standard_errors_corrected.as_ref())
    }

    /// Get the O(n⁻¹) bias-correction vector b̂ = H⁻¹ S(λ̂) β̂ in the
    /// original coefficient basis, if available.
    pub fn bias_correction_beta(&self) -> Option<&Array1<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.bias_correction_beta.as_ref())
    }

    /// Get the penalized Hessian if available.
    ///
    /// Boundary accessor: returns `&Array2<f64>` so out-of-scope consumers
    /// (CLI, GPU, families) keep their pre-newtype call shape. Use
    /// [`Self::penalized_hessian_unscaled`] when the caller wants the
    /// [`UnscaledPrecision`] newtype to enforce the dispersion-ownership
    /// invariant.
    pub fn penalized_hessian(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .map(|inf| inf.penalized_hessian.as_array())
            .or_else(|| {
                self.geometry
                    .as_ref()
                    .map(|geom| geom.penalized_hessian.as_array())
            })
    }

    /// Get the penalized Hessian as the [`UnscaledPrecision`] newtype if
    /// available. Use this when constructing newtype-aware APIs (HMC
    /// whitening, sampling) so the dispersion convention is enforced at
    /// the type level.
    pub fn penalized_hessian_unscaled(
        &self,
    ) -> Option<&crate::inference::dispersion_cov::UnscaledPrecision> {
        self.inference
            .as_ref()
            .map(|inf| &inf.penalized_hessian)
            .or_else(|| self.geometry.as_ref().map(|geom| &geom.penalized_hessian))
    }

    /// Get the φ-scaled posterior covariance as the [`PhiScaledCovariance`]
    /// newtype if available, sourced from `FitInference::beta_covariance`.
    ///
    /// Prefer this over [`Self::beta_covariance`] in inference-internal
    /// code so the φ-scaled invariant is type-enforced.
    pub fn beta_covariance_phi_scaled(
        &self,
    ) -> Option<&crate::inference::dispersion_cov::PhiScaledCovariance> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.beta_covariance.as_ref())
    }

    /// Get working weights if available.
    pub fn working_weights(&self) -> Option<&Array1<f64>> {
        self.inference.as_ref().map(|inf| &inf.working_weights)
    }

    /// Get working response if available.
    pub fn working_response(&self) -> Option<&Array1<f64>> {
        self.inference.as_ref().map(|inf| &inf.working_response)
    }

    /// Smoothing-parameter uncertainty covariance contribution `J·Var(ρ)·Jᵀ`
    /// in coefficient space, on the same dispersion scale as the conditional
    /// covariance `Vb = φ·H⁻¹`. This is the exact ρ-uncertainty term assembled
    /// from the IFT `dβ̂/dρ` and the outer Hessian at the fit optimum; the
    /// model-comparison machinery divides it by `φ` to recover the H⁻¹-scale
    /// ρ-covariance needed for the Wood–Pya–Säfken corrected EDF.
    pub fn smoothing_correction(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.smoothing_correction.as_ref())
    }

    /// Total effective degrees of freedom.
    pub fn edf_total(&self) -> Option<f64> {
        self.inference.as_ref().map(|inf| inf.edf_total)
    }

    /// EDF by block.
    pub fn edf_by_block(&self) -> &[f64] {
        self.inference
            .as_ref()
            .map(|inf| inf.edf_by_block.as_slice())
            .unwrap_or(&[])
    }

    /// Raw per-penalty-block trace `tr_kk = λ_kk·tr(H⁻¹ S_kk)`, aligned 1:1 with
    /// `lambdas`. Empty when the producing path did not record traces (issue
    /// #1219); callers must treat an empty slice as "unavailable".
    pub fn penalty_block_trace(&self) -> &[f64] {
        self.inference
            .as_ref()
            .map(|inf| inf.penalty_block_trace.as_slice())
            .unwrap_or(&[])
    }

    /// Per-term effective degrees of freedom over a smooth/random-effect term's
    /// coefficient block, defined as the trace of the linear-smoother influence
    /// matrix `F = H⁻¹X'WX` restricted to that block:
    ///
    /// ```text
    /// edf_term = Σ_{j ∈ coeff_range} F[j,j]
    ///          = |coeff_range| − Σ_{kk ∈ term} tr_kk,   tr_kk = λ_kk·tr(H⁻¹ S_kk).
    /// ```
    ///
    /// This is additive across terms and sums exactly to `edf_total = p − Σ_all
    /// tr_kk`, so a term's EDF can never exceed the model total or the design
    /// column count. The legacy per-block EDF sum `Σ_kk (rank(S_kk) − tr_kk)`
    /// double-counts shared tensor coefficients for `te`/`ti` (and anisotropic /
    /// adaptive) smooths, where several penalty blocks span the *same* coefficient
    /// range and `Σ_kk rank(S_kk) ≫ |coeff_range|` (#1219, #1277).
    ///
    /// `penalty_cursor` is the index of the term's first penalty block in the
    /// flat `lambdas` / `penalty_block_trace` / `edf_by_block` layout, and `k` is
    /// the number of penalty blocks the term owns (`0` for an unpenalised term).
    ///
    /// Resolution order, each exact when available: the influence-matrix trace
    /// (the model's own definition), then `|coeff_range| − Σ tr_kk` from the
    /// stored per-block traces (basis-invariant; exact even when `F` was never
    /// materialised for a large model), then — only when neither was recorded —
    /// the legacy block-sum as a last resort.
    pub fn per_term_edf(
        &self,
        coeff_range: std::ops::Range<usize>,
        penalty_cursor: usize,
        k: usize,
    ) -> f64 {
        let dim = coeff_range.len() as f64;
        // Primary: trace of the influence matrix over the term's coefficient block.
        if let Some(f) = self.coefficient_influence()
            && coeff_range.end <= f.nrows()
            && coeff_range.end <= f.ncols()
        {
            let tr = coeff_range.clone().map(|j| f[[j, j]]).sum::<f64>();
            return tr.clamp(0.0, dim);
        }
        // Fallback: |coeff_range| − Σ tr_kk from the stored per-block traces. Equal
        // to the influence-matrix trace and basis-invariant, so it is exact even
        // when `F` was never materialised (large models).
        if k == 0 {
            // Unpenalised term: every coefficient carries one full degree of freedom.
            return dim;
        }
        let traces = self.penalty_block_trace();
        if let Some(block) = traces.get(penalty_cursor..penalty_cursor + k) {
            let sum_trace = block.iter().sum::<f64>();
            return (dim - sum_trace).clamp(0.0, dim);
        }
        // Last resort: the legacy per-block EDF sum. Correct for disjoint penalties;
        // retained only for fits that recorded neither `F` nor per-block traces.
        // Clamp to the invariants that remain knowable without `F` or `tr_kk`:
        // a term sub-trace cannot exceed its coefficient count or the full-model
        // trace. Without this guard a `te`/`ti` block-sum reports e.g. 40 EDF for
        // a 36-coefficient model with total EDF 28 (#1277).
        let upper = self.edf_total().unwrap_or(dim).min(dim).max(0.0);
        self.edf_by_block()
            .get(penalty_cursor..penalty_cursor + k)
            .map(|block| block.iter().sum::<f64>().clamp(0.0, upper))
            .unwrap_or(0.0)
    }

    /// Find a block by role.
    pub fn block_by_role(&self, role: BlockRole) -> Option<&FittedBlock> {
        self.blocks.iter().find(|b| b.role == role)
    }

    /// Flat coefficient vector (all blocks concatenated).
    /// This is equivalent to `self.beta.clone()`.
    pub fn beta_flat(&self) -> Array1<f64> {
        self.beta.clone()
    }

    /// Time/baseline-hazard coefficients (survival location-scale).
    pub fn beta_time(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Time)
            .map(|b| b.beta.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Threshold coefficients (survival location-scale).
    pub fn beta_threshold(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Threshold)
            .map(|b| b.beta.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Log-sigma coefficients (survival location-scale).
    pub fn beta_log_sigma(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Scale)
            .map(|b| b.beta.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Link-wiggle coefficients (survival location-scale, optional).
    pub fn beta_link_wiggle(&self) -> Option<Array1<f64>> {
        self.block_by_role(BlockRole::LinkWiggle)
            .map(|b| b.beta.clone())
    }

    /// Smoothing parameters for time block.
    pub fn lambdas_time(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Time)
            .map(|b| b.lambdas.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Smoothing parameters for threshold block.
    pub fn lambdas_threshold(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Threshold)
            .map(|b| b.lambdas.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Smoothing parameters for log-sigma block.
    pub fn lambdas_log_sigma(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Scale)
            .map(|b| b.lambdas.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Smoothing parameters for link-wiggle block.
    pub fn lambdas_linkwiggle(&self) -> Option<Array1<f64>> {
        self.block_by_role(BlockRole::LinkWiggle)
            .map(|b| b.lambdas.clone())
    }

    /// Number of coefficient blocks.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Block roles.
    pub fn block_roles(&self) -> Vec<BlockRole> {
        self.blocks.iter().map(|b| b.role.clone()).collect()
    }

    /// Resolve the fitted link state for a given family.
    ///
    /// For standard (non-adaptive) link families, no extra state is fitted, so
    /// this returns the bare `FittedLinkState::Standard(None)` payload — the
    /// concrete `LinkFunction` lives on the family/spec and is not duplicated
    /// into the fitted-link record.  For adaptive links (SAS, BetaLogistic,
    /// Mixture, LatentCLogLog) it validates that the stored state matches the
    /// family and clones it out.
    pub fn fitted_link_state(
        &self,
        family: &crate::types::LikelihoodSpec,
    ) -> Result<FittedLinkState, EstimationError> {
        match (&family.response, &family.link) {
            (ResponseFamily::Gaussian, _) => Ok(FittedLinkState::Standard(None)),
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
                Ok(FittedLinkState::Standard(None))
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Probit)) => {
                Ok(FittedLinkState::Standard(None))
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::CLogLog)) => {
                Ok(FittedLinkState::Standard(None))
            }
            (ResponseFamily::Binomial, InverseLink::LatentCLogLog(_)) => match &self.fitted_link {
                FittedLinkState::LatentCLogLog { state } => {
                    Ok(FittedLinkState::LatentCLogLog { state: *state })
                }
                _ => Err(EstimationError::InvalidInput(
                    "BinomialLatentCLogLog requires fixed latent cloglog state".to_string(),
                )),
            },
            (ResponseFamily::Binomial, InverseLink::Sas(_)) => match &self.fitted_link {
                FittedLinkState::Sas { state, covariance } => Ok(FittedLinkState::Sas {
                    state: *state,
                    covariance: covariance.clone(),
                }),
                _ => Err(EstimationError::InvalidInput(
                    "BinomialSas requires fitted SAS link parameters".to_string(),
                )),
            },
            (ResponseFamily::Binomial, InverseLink::BetaLogistic(_)) => match &self.fitted_link {
                FittedLinkState::BetaLogistic { state, covariance } => {
                    Ok(FittedLinkState::BetaLogistic {
                        state: *state,
                        covariance: covariance.clone(),
                    })
                }
                _ => Err(EstimationError::InvalidInput(
                    "BinomialBetaLogistic requires fitted beta-logistic link parameters"
                        .to_string(),
                )),
            },
            (ResponseFamily::Binomial, InverseLink::Mixture(_)) => match &self.fitted_link {
                FittedLinkState::Mixture { state, covariance } => Ok(FittedLinkState::Mixture {
                    state: state.clone(),
                    covariance: covariance.clone(),
                }),
                _ => Err(EstimationError::InvalidInput(
                    "BinomialMixture requires fitted mixture link parameters".to_string(),
                )),
            },
            (ResponseFamily::Binomial, _) => Err(EstimationError::InvalidInput(
                "unsupported (binomial, link) combination".to_string(),
            )),
            (ResponseFamily::Poisson, _)
            | (ResponseFamily::Tweedie { .. }, _)
            | (ResponseFamily::NegativeBinomial { .. }, _)
            | (ResponseFamily::Gamma, _) => Ok(FittedLinkState::Standard(None)),
            (ResponseFamily::Beta { .. }, _) => Ok(FittedLinkState::Standard(None)),
            (ResponseFamily::RoystonParmar, _) => Ok(FittedLinkState::Standard(None)),
        }
    }
}
