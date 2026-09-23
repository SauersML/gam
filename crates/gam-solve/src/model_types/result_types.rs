use faer::Side;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::model_types::{Dispersion, EstimationError};
use gam_linalg::faer_ndarray::FaerCholesky;
use gam_linalg::utils::stack_offsets;
use gam_terms::inference::smooth_score_test::WorkingResidual;
use gam_problem::{
    FitStationarityEvidence, GlmLikelihoodSpec, InverseLink, LatentCLogLogState,
    LikelihoodScaleMetadata, LikelihoodSpec, LogLikelihoodNormalization, MixtureLinkSpec,
    MixtureLinkState, ResponseFamily, SasLinkSpec, SasLinkState, StandardLink, StationarityRung,
};

pub use gam_problem::ExecutionPath;

pub fn dispersion_from_likelihood(
    likelihood: &GlmLikelihoodSpec,
    profiled_gaussian_standard_deviation: Option<f64>,
) -> Result<Dispersion, EstimationError> {
    use gam_problem::ResolvedLikelihoodScale as Scale;

    let invalid = |reason: String| {
        EstimationError::InvalidInput(format!(
            "cannot resolve response dispersion for {}: {reason}",
            likelihood.spec.response.name()
        ))
    };
    let known = |phi| Dispersion::known(phi).map_err(|err| invalid(err.to_string()));
    let estimated_dispersion =
        |phi| Dispersion::estimated(phi).map_err(|err| invalid(err.to_string()));
    let reciprocal = |value, is_estimated| {
        Dispersion::from_reciprocal(value, is_estimated).map_err(|err| invalid(err.to_string()))
    };
    let resolved = likelihood
        .resolved_scale()
        .map_err(|error| invalid(error.to_string()))?;

    if !matches!(resolved, Scale::ProfiledGaussian)
        && profiled_gaussian_standard_deviation.is_some()
    {
        return Err(invalid(
            "a profiled Gaussian standard deviation was supplied for a non-profiled likelihood"
                .to_string(),
        ));
    }

    match resolved {
        Scale::ProfiledGaussian => {
            let standard_deviation = profiled_gaussian_standard_deviation.ok_or_else(|| {
                invalid(
                    "profiled Gaussian requires an explicit fitted standard deviation".to_string(),
                )
            })?;
            if !(standard_deviation.is_finite() && standard_deviation >= 0.0) {
                return Err(invalid(format!(
                    "profiled Gaussian standard deviation must be finite and non-negative, got {standard_deviation}"
                )));
            }
            let phi = standard_deviation * standard_deviation;
            if !phi.is_finite() || (standard_deviation > 0.0 && phi == 0.0) {
                return Err(invalid(format!(
                    "squared profiled Gaussian standard deviation is not representable: {standard_deviation}^2"
                )));
            }
            estimated_dispersion(phi)
        }
        Scale::FixedGaussian { phi } => known(phi.value()),
        Scale::Unit | Scale::NegativeBinomial { .. } => Ok(Dispersion::UNIT),
        Scale::Gamma {
            scale: gam_problem::ResolvedGammaScale::Shape(shape),
            estimated: is_estimated,
        } => reciprocal(shape.value(), is_estimated),
        Scale::Gamma {
            scale: gam_problem::ResolvedGammaScale::Dispersion(phi),
            estimated,
        } => {
            if estimated {
                estimated_dispersion(phi.value())
            } else {
                known(phi.value())
            }
        }
        Scale::Tweedie { phi, estimated } | Scale::Dispersion { phi, estimated } => {
            if estimated {
                estimated_dispersion(phi.value())
            } else {
                known(phi.value())
            }
        }
        Scale::BetaPrecision {
            precision,
            estimated: is_estimated,
        } => {
            let precision = precision.value();
            let beta_dispersion = if precision >= 1.0 {
                let inv_precision = 1.0 / precision;
                inv_precision / (1.0 + inv_precision)
            } else {
                1.0 / (1.0 + precision)
            };
            if is_estimated {
                estimated_dispersion(beta_dispersion)
            } else {
                known(beta_dispersion)
            }
        }
        Scale::Unspecified => Err(invalid(
            "Royston-Parmar has no GLM scalar response dispersion".to_string(),
        )),
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
        UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: beta.clone(),
                role: BlockRole::Mean,
                edf: 28.0,
                lambdas: Array1::from_vec(vec![1.0, 1.0]),
            }],
            training_sample_size: 64,
            training_response_fingerprint: None,
            log_lambdas: Array1::zeros(2),
            lambdas: Array1::from_vec(vec![1.0, 1.0]),
            likelihood_family: Some(LikelihoodSpec::gaussian_identity()),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
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
                edf_rank_bound: Vec::new(),
                edf_total: 28.0,
                smoothing_correction: None,
                smoothing_correction_method: None,
                smoothing_correction_first_order: None,
                smoothing_correction_method_first_order: None,
                smoothing_correction_absence: None,
                penalized_hessian: gam_problem::dispersion_cov::UnscaledPrecision::wrap(eye(36)),
                reparam_qs: None,
                dispersion: Dispersion::estimated(1.0)
                    .expect("1.0 is a valid estimated dispersion"),
                factorized_standard_errors: None,
                smoothing_correction_factorized: None,
                beta_covariance_frequentist: None,
                coefficient_influence: None,
                weighted_gram: None,
                identified_subspace: None,
                working_residual: None,
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
        .expect("test fixture carries fixed-outer convergence evidence")
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
    /// A custom-family fit has no dispersion to scale `H⁻¹` by: its objective
    /// is the complete negative log-likelihood, so the Laplace covariance is
    /// the inverse penalized Hessian as it stands (gam#2765). A Gaussian fit
    /// keeps its profiled `σ̂²`.
    #[test]
    fn a_fit_without_an_engine_level_family_scales_its_precision_by_one_2765() {
        let gaussian = fit_single_thinplate_consistent_edf();
        let gaussian_scale = gaussian
            .coefficient_covariance_scale()
            .expect("a Gaussian fit has a profiled scale");
        assert!(
            (gaussian_scale - gaussian.standard_deviation * gaussian.standard_deviation).abs()
                <= 1e-12 * (1.0 + gaussian_scale.abs()),
            "Gaussian scale must be σ̂²: got {gaussian_scale}, σ̂ = {}",
            gaussian.standard_deviation
        );
        let mut custom = fit_single_thinplate_consistent_edf();
        custom.likelihood_family = None;
        assert_eq!(
            custom.coefficient_covariance_scale().expect("a custom-family scale is defined"),
            1.0
        );
    }

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
        UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: beta.clone(),
                role: BlockRole::Mean,
                edf: edf_total,
                lambdas: Array1::from_vec(vec![1.0]),
            }],
            training_sample_size: 64,
            training_response_fingerprint: None,
            log_lambdas: Array1::zeros(1),
            lambdas: Array1::from_vec(vec![1.0]),
            likelihood_family: Some(LikelihoodSpec::gaussian_identity()),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
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
                edf_rank_bound: Vec::new(),
                edf_total,
                smoothing_correction: None,
                smoothing_correction_method: None,
                smoothing_correction_first_order: None,
                smoothing_correction_method_first_order: None,
                smoothing_correction_absence: None,
                penalized_hessian: gam_problem::dispersion_cov::UnscaledPrecision::wrap(eye(p)),
                reparam_qs: None,
                dispersion: Dispersion::estimated(1.0)
                    .expect("1.0 is a valid estimated dispersion"),
                factorized_standard_errors: None,
                smoothing_correction_factorized: None,
                beta_covariance_frequentist: None,
                coefficient_influence: Some(influence),
                weighted_gram: None,
                identified_subspace: None,
                working_residual: None,
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
        .expect("test fixture carries fixed-outer convergence evidence")
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
        UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: Array1::zeros(p),
                role: BlockRole::Mean,
                edf: edf_total,
                lambdas: lambdas.clone(),
            }],
            training_sample_size: 256,
            training_response_fingerprint: None,
            log_lambdas: Array1::zeros(n_levels),
            lambdas,
            likelihood_family: Some(LikelihoodSpec::gaussian_identity()),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
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
                edf_rank_bound: Vec::new(),
                edf_total,
                smoothing_correction: None,
                smoothing_correction_method: None,
                smoothing_correction_first_order: None,
                smoothing_correction_method_first_order: None,
                smoothing_correction_absence: None,
                penalized_hessian: gam_problem::dispersion_cov::UnscaledPrecision::wrap(eye(p)),
                reparam_qs: None,
                dispersion: Dispersion::estimated(1.0)
                    .expect("1.0 is a valid estimated dispersion"),
                factorized_standard_errors: None,
                smoothing_correction_factorized: None,
                beta_covariance_frequentist: None,
                // No influence matrix: forces the `|coeff_range| − Σ tr_kk`
                // per-block-trace channel, where the `penalty_cursor` walk matters.
                coefficient_influence: None,
                weighted_gram: None,
                identified_subspace: None,
                working_residual: None,
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
        .expect("test fixture carries fixed-outer convergence evidence")
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
            buggy_edfs
                .last()
                .copied()
                .expect("the loop above pushed at least one EDF")
                <= tol,
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

    /// Sibling of `by_factor_unpenalised_main_effect_does_not_zero_last_level_edf`
    /// for the OTHER design skip condition. `design_construction.rs` skips an RE
    /// penalty block when `range.is_empty() || !penalized` — so a PENALIZED random
    /// effect with zero kept groups (every level filtered → empty coefficient
    /// range) owns NO entry in the flat `penalty_block_trace` layout, exactly like
    /// an unpenalised block. The summary cursor must mirror BOTH conditions: a
    /// `k_pen = usize::from(penalized && !range.is_empty())` advance. A cursor that
    /// keys only on `penalized` (the partial fix) still advances by 1 for the
    /// empty penalized block and slides every following smooth's trace window one
    /// block down — the same #1368 desync that zeroes the last level's EDF.
    #[test]
    fn empty_range_penalised_re_does_not_zero_last_level_edf() {
        let n_levels = 5usize;
        let dim = 20usize;
        let fit = fit_by_factor_penalty_layout(n_levels);
        let smooth_start = 1; // global layout: [empty penalised RE block(0 cols) | smooths]
        let expected_per_smooth = 15.0_f64;
        let tol = 1e-9;
        // The leading RE block is PENALIZED but has an empty coefficient range,
        // so the design pushed no penalty block for it (penalty_block_trace has
        // only the n_levels smooth entries).
        let re_penalized = true;
        let re_range_empty = true;

        // ── BUGGY walk: cursor keys only on `penalized`, advances by 1. ──
        let mut buggy_cursor = usize::from(re_penalized);
        let mut buggy_edfs = Vec::new();
        for level in 0..n_levels {
            let start = smooth_start + level * dim;
            let edf = fit.per_term_edf(start..(start + dim), buggy_cursor, 1);
            buggy_cursor += 1;
            buggy_edfs.push(edf);
        }
        assert!(
            buggy_edfs
                .last()
                .copied()
                .expect("the loop above pushed at least one EDF")
                <= tol,
            "the penalized-only cursor must zero the last level's EDF, got {buggy_edfs:?}"
        );

        // ── FIXED walk: cursor mirrors BOTH design conditions. ──
        let mut cursor = usize::from(re_penalized && !re_range_empty); // = 0
        let mut edfs = Vec::new();
        for level in 0..n_levels {
            let start = smooth_start + level * dim;
            let edf = fit.per_term_edf(start..(start + dim), cursor, 1);
            cursor += 1;
            edfs.push(edf);
        }
        for (level, &edf) in edfs.iter().enumerate() {
            assert!(
                (edf - expected_per_smooth).abs() < tol,
                "level {level} EDF {edf} != expected {expected_per_smooth} (set {edfs:?})"
            );
        }
    }

    /// Build a fit whose GLOBAL penalty layout opens with a shared
    /// `LinearTermRidge` block (trace `lead_trace`) followed by `block_traces`
    /// further penalty blocks, with NO influence matrix recorded — so
    /// `per_term_edf` reads the `penalty_block_trace` fallback window, the exact
    /// path the persisted / column-conditioned summary builders hit.
    fn fit_with_leading_linear_ridge(lead_trace: f64, block_traces: &[f64]) -> UnifiedFitResult {
        let mut traces = vec![lead_trace];
        traces.extend_from_slice(block_traces);
        let n_blocks = traces.len();
        let p = 1 + 10 * block_traces.len().max(1);
        let lambdas = Array1::from_vec(vec![1.0; n_blocks]);
        UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: Array1::zeros(p),
                role: BlockRole::Mean,
                edf: p as f64,
                lambdas: lambdas.clone(),
            }],
            training_sample_size: 256,
            training_response_fingerprint: None,
            log_lambdas: Array1::zeros(n_blocks),
            lambdas,
            likelihood_family: Some(LikelihoodSpec::gaussian_identity()),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: Some(0.0),
            standard_deviation: 1.0,
            covariance_conditional: None,
            covariance_corrected: None,
            inference: Some(FitInference {
                edf_by_block: vec![0.0; n_blocks],
                penalty_block_trace: traces,
                edf_rank_bound: Vec::new(),
                edf_total: p as f64,
                smoothing_correction: None,
                smoothing_correction_method: None,
                smoothing_correction_first_order: None,
                smoothing_correction_method_first_order: None,
                smoothing_correction_absence: None,
                penalized_hessian: gam_problem::dispersion_cov::UnscaledPrecision::wrap(eye(p)),
                reparam_qs: None,
                dispersion: Dispersion::estimated(1.0)
                    .expect("1.0 is a valid estimated dispersion"),
                factorized_standard_errors: None,
                smoothing_correction_factorized: None,
                beta_covariance_frequentist: None,
                // No influence matrix: forces the per-block-trace fallback where
                // `penalty_cursor` keys into `penalty_block_trace`.
                coefficient_influence: None,
                weighted_gram: None,
                identified_subspace: None,
                working_residual: None,
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
        .expect("test fixture carries fixed-outer convergence evidence")
    }

    /// Regression for issue #1372. The fit's global penalty order is
    /// `[LinearTermRidge (trace 0.9), s(x) smooth (trace 3.0)]` (a `double_penalty`
    /// linear term emits the leading shared ridge). For the dim-10 smooth block the
    /// summary builders must seed `penalty_cursor` PAST that leading ridge: the
    /// CORRECT cursor=1 reads trace[1]=3.0 → EDF = 10 − 3.0 = 7.0, whereas the
    /// BUGGY cursor=0 reads trace[0]=0.9 → 9.1. With `coefficient_influence=None`
    /// the bug is unmasked (small dense fits route through the influence matrix and
    /// never hit this path).
    #[test]
    fn summary_penalty_cursor_skips_leading_linear_ridge() {
        let fit = fit_with_leading_linear_ridge(0.9, &[3.0]);
        let smooth_range = 1..11; // 10-column smooth block past the intercept
        let tol = 1e-9;

        // BUGGY cursor (0): reads the leading LinearTermRidge trace (0.9).
        let buggy = fit.per_term_edf(smooth_range.clone(), 0, 1);
        assert!(
            (buggy - 9.1).abs() < tol,
            "buggy cursor=0 should read trace[0]=0.9 → EDF 9.1, got {buggy}"
        );

        // FIXED cursor (1): seeded past the leading LinearTermRidge block.
        let leading_linear_ridge = 1usize;
        let fixed = fit.per_term_edf(smooth_range, leading_linear_ridge, 1);
        assert!(
            (fixed - 7.0).abs() < tol,
            "cursor seeded past LinearTermRidge should read trace[1]=3.0 → EDF 7.0, got {fixed}"
        );
    }

    /// Defect B coverage for issue #1372: with the global order
    /// `[LinearTermRidge (0.9), <unpenalised RE: 0 blocks>, s(x) (trace 3.0)]`, the
    /// summary cursor must (a) skip the leading ridge AND (b) NOT advance over the
    /// unpenalised RE term. Combined, the smooth's cursor lands at 1 (the ridge
    /// occupies index 0; the RE term owns no penalty block) → trace[1]=3.0 → EDF
    /// 7.0. Advancing for the RE term (cursor=2) would run off the 2-block trace and
    /// collapse the EDF to 0.
    #[test]
    fn summary_penalty_cursor_skips_leading_ridge_and_unpenalised_re() {
        let fit = fit_with_leading_linear_ridge(0.9, &[3.0]);
        let smooth_range = 11..21; // smooth block after the RE block's coefficients
        let tol = 1e-9;

        // Replay the summary walk: seed past the leading LinearTermRidge, then the
        // unpenalised RE term owns 0 penalty blocks (does not advance the cursor).
        let mut cursor = 1usize; // past LinearTermRidge
        let re_penalized = false;
        cursor += usize::from(re_penalized); // Defect B: +0 for unpenalised RE
        let edf = fit.per_term_edf(smooth_range.clone(), cursor, 1);
        assert!(
            (edf - 7.0).abs() < tol,
            "cursor past ridge + unpenalised RE should read trace[1]=3.0 → EDF 7.0, got {edf}"
        );

        // Sanity: the Defect-B-buggy walk (advance over the unpenalised RE) runs off
        // the end of the 2-block trace and `per_term_edf` collapses to 0.
        let buggy = fit.per_term_edf(smooth_range, 2, 1);
        assert!(
            buggy <= tol,
            "buggy cursor=2 should run off the trace and zero the EDF, got {buggy}"
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

/// ρ margin (in log-λ units) within which an outer smoothing coordinate
/// counts as railed against its box bound.
pub(crate) const CERTIFICATE_RAIL_MARGIN: f64 = 0.5;

/// Which exact test proved a rail face's first-order expansion positive.
///
/// Off a face `F` the criterion is `V_∞ + f(t) + O(|t|²)` with
/// `f(t) = ½tr((Σ_{j∈F} A_j/t_j)⁻¹C)`, `t_j = e^{−ρ_j} ≥ 0`. `f` is
/// homogeneous of degree one, so positivity on the closed orthant is
/// positivity on the simplex — and two of its cases are decided exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FacePositivityRoute {
    /// `C ≻ 0`: every compression of `C` is positive definite, so `f > 0` for
    /// every weighting. Sufficient, not necessary.
    PositiveForm,
    /// The released ranges of the face penalties are linearly independent
    /// (`Σ_j rank A_j = q`). A congruence then block-diagonalizes every `A_j`
    /// at once, `f(t) = Σ_j c_j t_j` is exactly LINEAR, and `f > 0` on the
    /// simplex iff every identified `c_j > 0` — the KKT test at the face.
    IndependentRanges,
    /// The released ranges OVERLAP, so `f` is genuinely nonlinear and neither
    /// the per-axis slopes nor `C ≻ 0` decides it (both axis laws can be
    /// positive while a joint release descends). The weighted parallel sum
    /// `M(t) = (Σ_j A_j/t_j)⁻¹` is matrix-concave, so on the spectral split
    /// `C = C₊ − C₋` the expansion is a difference of concave functions; a
    /// simplicial branch-and-bound bounds it below on every cell (vertex values
    /// of the concave part, a tangent plane of the subtracted one) and proves
    /// `f > 0` on the whole simplex, each cell's bound clearing its own
    /// rounding band.
    SimplexBound,
    /// The ZERO-smoothing face (`λ_j → 0`, `j ∈ L`) of penalties whose ranges
    /// the surviving penalties already cover. There the criterion is jointly
    /// analytic in `λ_L` — no logdet rank changes as they vanish — so
    /// `V = V(0) + Σ_{j∈L} c′_j λ_j + O(|λ|²)` on the closed orthant, exactly
    /// linear at first order, and the face is a strict minimizer iff every
    /// `c′_j` clears its rounding band `τ_j` (the KKT test at `λ = 0`).
    CoveredZeroSmoothing,
}

/// What established a rail coordinate's tail law, and the standard it cleared
/// (#2348 Inc 5 build-out).
///
/// The two routes are not two ways of measuring one quantity. They apply
/// different standards to different evidence, and the floors they clear are not
/// comparable: one is a settlement radius on a measured estimate of `ĉ`, the
/// other a backward-error threshold on the smallest eigenvalue of a matrix.
/// Carrying both in a single `noise_margin: f64` — documented as "the
/// pencil-constant noise floor for a measured tail, or the eigen-backward-error
/// margin of the analytic face form" — left every reader unable to tell which
/// number they held, including this module's own well-formedness guard.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RailTailEvidence {
    /// PROVEN at the face. The `λ = ∞` limit was formed exactly and the
    /// criterion's first-order expansion off the face,
    /// `V = V_∞ + ½tr((Σ_j A_j/t_j)⁻¹C) + O(|t|²)` with `t_j = e^{−ρ_j}`, is
    /// strictly positive for every way of coming off it — every finite
    /// smoothing parameter on the face and on every sub-face, at once.
    /// [`FacePositivityRoute`] names which exact positivity test decided it.
    ///
    /// A coordinate whose own penalty releases nothing once the REST of the
    /// face is at `λ = ∞` is unidentified there, and the proof derives
    /// `tail_constant = 0` for it exactly: `V` does not depend on its smoothing
    /// parameter at all. That zero is a result of the proof, not a missing
    /// measurement — which is precisely why the two routes cannot share one
    /// well-formedness rule.
    AnalyticFaceProof {
        /// Which exact positivity test proved the face.
        route: FacePositivityRoute,
        /// The route's decisive statistic: `λ_min(C)` on
        /// [`FacePositivityRoute::PositiveForm`], the binding coordinate's
        /// analytic pencil constant `c_j` on
        /// [`FacePositivityRoute::IndependentRanges`], the binding simplex
        /// cell's lower bound on `f` on [`FacePositivityRoute::SimplexBound`],
        /// the binding coordinate's `λ`-slope `c′_j` on
        /// [`FacePositivityRoute::CoveredZeroSmoothing`].
        statistic: f64,
        /// The rounding band that statistic had to clear, from the measured
        /// error of forming `C` in floating point (never a tuned margin).
        band: f64,
    },
    /// MEASURED by probing back from the rail: the pencil constant
    /// `ĉ = ∓e^{±ρ}·∂V/∂ρ`, read off analytic gradients that each cleared their
    /// own rounding band, settled within those bands to a limit `c` with a
    /// derived settlement radius. The evidence is a window at finite `λ`, so a
    /// constant the radius cannot separate from zero is the instrument
    /// reporting itself rather than a tail law.
    ProbedTail {
        /// The rounding bound on the rail-most probe's gradient, from the
        /// measured magnitudes of the parts that gradient was summed from.
        gradient_band: f64,
        /// The settlement radius `R`: `|c − ĉ| ≤ R` for the reported constant.
        extrapolation_radius: f64,
    },
}

impl RailTailEvidence {
    /// Whether `tail_constant` is well formed FOR THIS ROUTE.
    ///
    /// The routes disagree about zero, and both are right. A probed tail reads
    /// `ĉ` off a window of measured gradients, where a constant its settlement
    /// radius cannot separate from zero is noise being reported as a law. A proven face derives `ĉ` from the limit
    /// itself, and derives exactly zero for a coordinate the rest of the face
    /// has already pinned; refusing that would refuse a face the proof covers.
    /// Such a coordinate needs another face coordinate to pin its directions, so
    /// it can only arise on a face of two or more — which is why a
    /// single-coordinate certificate never exercised the disagreement.
    pub fn admits(&self, tail_constant: f64) -> bool {
        if !tail_constant.is_finite() {
            return false;
        }
        match self {
            Self::AnalyticFaceProof {
                statistic, band, ..
            } => {
                // Re-check the proof's own inequality: `certifies()` may be
                // asked of a DESERIALIZED certificate, where these are only
                // numbers someone supplied.
                tail_constant >= 0.0
                    && statistic.is_finite()
                    && band.is_finite()
                    && *band >= 0.0
                    && *statistic > *band
            }
            Self::ProbedTail {
                gradient_band,
                extrapolation_radius,
            } => {
                // Deserialized certificates carry only numbers someone
                // supplied, so the settlement inequality is re-checked here.
                gradient_band.is_finite()
                    && *gradient_band >= 0.0
                    && extrapolation_radius.is_finite()
                    && *extrapolation_radius >= 0.0
                    && tail_constant > *extrapolation_radius
            }
        }
    }

    /// A short label naming the route, for logs and refusal summaries.
    pub fn route(&self) -> &'static str {
        match self {
            Self::AnalyticFaceProof { .. } => "analytic face proof",
            Self::ProbedTail { .. } => "probed tail",
        }
    }
}

/// One outer smoothing coordinate certified stationary-at-asymptote: it has no
/// interior optimum (its gradient never clears the fixed bound), but the fitted
/// model has provably reached its rail limit to within tolerance along it
/// (#2348 Inc 1 / #2299 layer 3, #2337 Thm 2.1). The certified facts are the
/// per-coordinate outputs of
/// `crate::rho_optimizer::asymptote_certificate::assess_coordinate`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RailCoordinate {
    /// The ρ-block index of the railed coordinate.
    pub index: usize,
    /// Which rail (`λ → ∞` upper, `λ → 0` lower) it is approaching.
    pub side: crate::rho_optimizer::asymptote_certificate::AsymptoteSide,
    /// The pencil constant `ĉ` of the tail law `∂V/∂ρ = −ĉ·e^{−ρ}`: either the
    /// settled limit measured by probing back from the rail, or — when the
    /// objective can form its λ=∞ limit exactly — the analytic constant
    /// `½tr((QᵀS_kQ)⁻¹QᵀCQ)` of the face proof (#2348 Inc 5). A face
    /// coordinate the rest of the face has already pinned reports `0`: the
    /// criterion does not depend on its smoothing parameter at all.
    pub tail_constant: f64,
    /// The exact remaining criterion value-gap to the rail, `|∂V/∂ρ| = ĉ·e^{−ρ}`.
    pub value_gap: f64,
    /// The bound on remaining coefficient travel to the rail limit.
    pub estimand_travel_bound: f64,
    /// Which route established the tail, and the standard that route applied.
    /// The well-formedness rule for `tail_constant` comes from here, because
    /// the two routes genuinely differ about what a valid constant is.
    pub evidence: RailTailEvidence,
}

/// The stationarity equation that certified an outer optimum.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OuterStationarityCertificate {
    /// KKT-projected analytic objective gradient.
    AnalyticGradient {
        grad_norm: f64,
        projected_grad_norm: f64,
        bound: f64,
        /// Which rung produced `bound` (#2530). Beside it inside the variant,
        /// not as a sibling field on the certificate, so the two cannot be read
        /// apart — the bound alone spans nine orders across this subsystem and
        /// says nothing about which standard it came from.
        rung: CertifiedRung,
    },
    /// KKT-projected, root-equivalent analytic fixed-point equations.
    FixedPoint {
        residual_inf_norm: f64,
        projected_residual_inf_norm: f64,
        bound: f64,
        /// Which rung produced `bound` (#2530).
        rung: CertifiedRung,
        covered_coordinates: usize,
    },
    /// Stationary-at-asymptote (#2348 Inc 1): the interior (non-railed)
    /// coordinates are gradient-stationary, and every coordinate whose gradient
    /// stays above the fixed bound all the way to a box rail is certified on a
    /// confirmed exponential tail whose fitted model already equals the rail
    /// limit to within the estimand tolerance. `interior_projected_grad_norm`
    /// is the KKT-projected gradient with the railed coordinates removed; it is
    /// the quantity compared against `bound`.
    AsymptoteRail {
        interior_projected_grad_norm: f64,
        bound: f64,
        /// Which rung produced `bound` (#2530).
        rung: CertifiedRung,
        rails: Vec<RailCoordinate>,
    },
}

impl OuterStationarityCertificate {
    pub fn raw_norm(&self) -> f64 {
        match self {
            Self::AnalyticGradient { grad_norm, .. } => *grad_norm,
            Self::FixedPoint {
                residual_inf_norm, ..
            } => *residual_inf_norm,
            // The load-bearing residual is the interior (non-railed) projected
            // gradient; the railed coordinates are certified by their tails, not
            // by a vanishing residual.
            Self::AsymptoteRail {
                interior_projected_grad_norm,
                ..
            } => *interior_projected_grad_norm,
        }
    }

    pub fn projected_norm(&self) -> f64 {
        match self {
            Self::AnalyticGradient {
                projected_grad_norm,
                ..
            } => *projected_grad_norm,
            Self::FixedPoint {
                projected_residual_inf_norm,
                ..
            } => *projected_residual_inf_norm,
            Self::AsymptoteRail {
                interior_projected_grad_norm,
                ..
            } => *interior_projected_grad_norm,
        }
    }

    pub fn bound(&self) -> f64 {
        match self {
            Self::AnalyticGradient { bound, .. }
            | Self::FixedPoint { bound, .. }
            | Self::AsymptoteRail { bound, .. } => *bound,
        }
    }

    /// The rung that produced [`Self::bound`] (#2530).
    pub fn rung(&self) -> &CertifiedRung {
        match self {
            Self::AnalyticGradient { rung, .. }
            | Self::FixedPoint { rung, .. }
            | Self::AsymptoteRail { rung, .. } => rung,
        }
    }

    /// A stable label for which stationarity equation certified this point.
    ///
    /// Each variant has its own label, so an `AsymptoteRail` certificate is
    /// never reported as an analytic gradient in the evidence map a reader
    /// consults to find out which route ran.
    pub(crate) fn kind_label(&self) -> &'static str {
        match self {
            Self::AnalyticGradient { .. } => "analytic_gradient",
            Self::FixedPoint { .. } => "fixed_point",
            Self::AsymptoteRail { .. } => "asymptote_rail",
        }
    }

    /// The certified asymptote rails, when this is an `AsymptoteRail`.
    pub fn rails(&self) -> &[RailCoordinate] {
        match self {
            Self::AsymptoteRail { rails, .. } => rails,
            _ => &[],
        }
    }
}

/// Analytic optimality certificate at the returned optimum (#934).
///
/// Certifies stationarity from the ANALYTIC objective alone — no
/// finite-difference probes run in production (SPEC rule 2; derivative
/// cross-checks live only in focused tests). The certificate answers,
/// machine-checkably, the three questions every non-termination postmortem
/// asks: does the KKT-projected analytic stationarity equation vanish HERE
/// (`Self::is_stationary`); is the outer curvature admissible for a minimum
/// HERE (`Self::curvature_admissible`); did any smoothing coordinate rail
/// to a box bound (`lambdas_railed`). A failed certificate REJECTS the fit as
/// typed non-convergence in `run_outer`; it is never a warn-and-continue
/// diagnostic.
/// The gradient-residue floor's verdict on the interior curvature, recorded
/// BESIDE the raw [`OuterCriterionCertificate::hessian_psd`] measurement rather
/// than replacing it.
///
/// Two different questions are being asked of the same matrix, and they have
/// different right answers:
///
/// * *is the interior sub-block positive semidefinite as assembled?* — the
///   measured fact, which reporting surfaces and any consumer demanding a
///   genuine PSD certificate (`OwnedModeCurvatureRequirement::CertifiedLocalMinimum`)
///   must keep receiving unchanged;
/// * *is its most negative direction distinguishable from zero by the
///   instrument that produced it?* — the verdict, which is what may admit a
///   minted optimum.
///
/// Collapsing them into one flag would leave every consumer asking the first
/// question and silently receiving an answer to the second.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurvatureFloorClearance {
    /// Smallest eigenvalue of the interior (un-railed) sub-block, as assembled.
    pub interior_min_eigenvalue: f64,
    /// `max_k |g_k|` over the JUDGED coordinates. By Weyl this is the most
    /// negative curvature the floor can absorb:
    /// `λ_min(H) + min|g| ≤ λ_min(H + diag|g|) ≤ λ_min(H) + max|g|`.
    pub gradient_floor: f64,
    /// Smallest eigenvalue of `H + diag(|g|)` on the judged sub-block — the
    /// matrix whose sign actually decides `cleared` (#2748).
    ///
    /// The two fields above are the ENDS of the Weyl sandwich, not the
    /// quantity in it. A refusal reporting `interior lambda_min = -8.533e-7`
    /// beside `gradient_floor = 3.870e-5` reads as a curvature 45x INSIDE its
    /// own floor and yet refused, because the floor is `max_k |g_k|` while the
    /// binding end of the sandwich is `min_k |g_k|`: the deciding matrix's
    /// eigenvalue was never on the record at all. It is now.
    ///
    /// `#[serde(default)]` so a certificate serialized before this field
    /// existed still deserializes; those carry no measurement here, which is
    /// honest — none was taken.
    #[serde(default)]
    pub floored_min_eigenvalue: f64,
    /// The MEASURED curvature resolution `‖δH‖₂` the definiteness test was
    /// taken at (#2748): the maximum of the identities that are exactly zero in
    /// exact arithmetic (the symmetrization defect, the penalty-map invariance
    /// residual). `0.0` when neither could be taken, in which case the
    /// historical `√ε·max(1, max|H_ii|)` arithmetic shift is what decided.
    #[serde(default)]
    pub measured_resolution: f64,
    /// The shift the verdict was ACTUALLY decided at (#2748): the larger of
    /// [`Self::measured_resolution`] and the arithmetic shift
    /// `sqrt(eps)*max(max|H_ii|, 1)`.
    ///
    /// [`Self::measured_resolution`] is `0.0` whenever no identity could be
    /// taken, and on a ρ-Hessian whose largest diagonal is under 1 the shift
    /// that then decides is a flat `1.49e-8` — eight orders above the
    /// eigensolver backward error a second layer would derive for itself.
    /// Recording `0.0` for "the arithmetic shift decided" is the same defect
    /// [`Self::floored_min_eigenvalue`] exists to fix one field earlier: the
    /// quantity the decision was taken against was not on the record.
    ///
    /// Measured on gam#2748's `geo_disease` k=12 cell: `lambda_min(H+diag|g|)`
    /// `= -1.129942e-8` cleared at `1.490116e-8`, and the smoothing correction
    /// then refused the same direction at `2.191651e-16`.
    ///
    /// `#[serde(default)]` so a certificate serialized before this field
    /// existed still deserializes; those carry no measurement here.
    #[serde(default)]
    pub decided_at_resolution: f64,
    /// Whether `H + diag(|g|)` is positive semidefinite on that sub-block.
    pub cleared: bool,
}

/// One railed coordinate together with the interval and margin it was judged
/// against (#2530).
///
/// `lambdas_railed` is a list of indices, and an index is not a verdict anyone
/// can check: `ρ=29.994` is railed against `[-30, +30]` and interior to
/// `[-100, +100]`. Recovering the interval for a single coordinate cost a
/// thirteen-point seeding sweep on #2462, and that same issue then made the
/// margin `coordinate_rail_margin(lo, hi)` — **width-capped per coordinate** —
/// so two coordinates in one fit can be judged against different margins and
/// `railed=[1, 3]` is not even one statement.
///
/// Carried beside `lambdas_railed` rather than replacing it: that field is the
/// stable index report every consumer already reads, and widening its element
/// type would churn ~45 call sites to deliver the same facts.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RailedCoordinateFact {
    /// Index into the θ vector — the same index that appears in
    /// `lambdas_railed` when the coordinate is in the ρ block.
    pub index: usize,
    /// The coordinate's value at the certified point.
    pub theta: f64,
    /// Lower box bound it was tested against.
    pub lower: f64,
    /// Upper box bound it was tested against.
    pub upper: f64,
    /// The width-capped margin in force for THIS coordinate.
    pub margin: f64,
    /// What the bound it is railed at is (#2954). `#[serde(default)]` reads a
    /// fact stored before this field as [`RailFaceKind::Unrecorded`].
    #[serde(default)]
    pub face: RailFaceKind,
}

/// The kind of box face a railed coordinate sits on (#2954, #2627).
///
/// Box-KKT certifies only at a face that belongs to the model. A ρ bound does
/// when the route derives it from the term's limit model: past a #2812
/// resolvability edge every direction's effective degrees of freedom are at
/// their λ → ∞ (null-space fit) or λ → 0 (unpenalized fit) value to within the
/// criterion gradient's resolution, so the bound is where that limit holds,
/// not where a search was stopped. Any other bound (the representable
/// log-strength range, the precision box a term without penalty geometry falls
/// back to, a box a route declares without deriving it) is a representability
/// face, and projection-only stationarity there says nothing about the data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RailFaceKind {
    /// A bound the route derived from the term's limit model.
    LimitModel,
    /// A literal bound: box-KKT does not certify here.
    Representability,
    /// A fact stored before certificates recorded the face kind, so which face
    /// it sat on is unknown. No live certificate records it.
    #[default]
    Unrecorded,
}

impl std::fmt::Display for RailedCoordinateFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "#{} theta={:.6e} box=[{:.6e}, {:.6e}] margin={:.3e}",
            self.index, self.theta, self.lower, self.upper, self.margin
        )
    }
}

/// The rung a CERTIFICATE was decided against, in owned form (#2530).
///
/// [`StationarityRung::label`] is a `&'static str` — a fixed in-process
/// vocabulary — and a certificate is a PERSISTED artifact: it round-trips
/// through `Serialize`/`Deserialize` from stored fits, where a borrowing
/// deserializer cannot produce a `'static` label. So the certificate owns its
/// label while the in-process refusal carrier keeps the static form. One
/// allocation per minted certificate, which is once per fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertifiedRung {
    /// Stable rung label, e.g. `"curvature-resolvability"`.
    pub label: String,
    /// Whether this rung is the derived resolvability standard rather than a
    /// gradient-magnitude substitute.
    pub derived_standard: bool,
}

impl From<StationarityRung> for CertifiedRung {
    fn from(rung: StationarityRung) -> Self {
        Self {
            label: rung.label.to_string(),
            derived_standard: rung.derived_standard,
        }
    }
}

impl std::fmt::Display for CertifiedRung {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rung={} derived_standard={}",
            self.label, self.derived_standard
        )
    }
}

/// What curvature evidence a certificate actually has (#2561).
///
/// This was `Option<bool>`, and its `None` carried FOUR structurally different
/// meanings that `OuterCriterionCertificate::curvature_admissible` accepted
/// alike:
///
/// 1. a multi-start screening pass deliberately declined to spend the
///    order-four derivative ladder ([`Self::NotSpent`]);
/// 2. the route exposes no analytic Hessian at all ([`Self::NotAvailable`]);
/// 3. the EFS/fixed-point route, which has none by construction (same);
/// 4. there is no outer estimand to have curvature ([`Self::NoEstimand`]).
///
/// Case 1 is deliberate and documented — screening is a first-order gate, and
/// "the one order-four evaluation belongs to the winner". But that design also
/// promises the winner's verdict is the one that mints, and while `None` meant
/// all four things at once, nothing could check it: a Mint-fidelity refusal to
/// measure was byte-identical to a screening pass that chose not to. Naming the
/// states makes that promise assertable.
///
/// Acceptance is unchanged. Only [`Self::Measured`] with `psd: false` is a
/// negative verdict; every other state passes, exactly as `!= Some(false)` did.
///
/// # Serialized form
///
/// Serializes as the legacy `Option<bool>` under the legacy key, because
/// `hessian_psd` is a published Python contract (`gamfit/_summary.py`) whose
/// domain is `null | true | false`, and because stored model bytes carry this
/// field with no version tag. The round trip is therefore deliberately LOSSY:
/// `NotSpent` and `NoEstimand` both reload as `NotAvailable`. That is sound
/// because the three are acceptance-identical, and screening certificates are
/// never persisted — only the winner mints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "Option<bool>", into = "Option<bool>")]
pub enum CurvatureEvidence {
    /// A Hessian existed at the certified point and was tested.
    Measured { psd: bool },
    /// A screening pass declined to spend the order-four ladder. A certificate
    /// minted at `CertificationFidelity::Mint` must never carry this.
    NotSpent,
    /// The route exposes no analytic Hessian, so there was nothing to test.
    NotAvailable,
    /// A zero-dimensional outer problem: no estimand, so no curvature exists
    /// to be admissible or otherwise. Distinct from [`Self::NotAvailable`],
    /// which means a curvature question existed and could not be answered.
    NoEstimand,
    /// A Hessian was measured, reported a negative direction, and **the
    /// criterion itself was then asked about that direction and did not fall**
    /// (#2612).
    ///
    /// The escape steps `ρ ± αv` along the reported minimum eigenvector, in
    /// both signs, from one e-fold down to the step at which the quadratic
    /// model's own predicted decrease `½|λ_min|α²` reaches the criterion's
    /// resolution. Below that step the claim predicts nothing the criterion can
    /// represent, so that ladder covers the claim's WHOLE falsifiable range. A
    /// direction that lowers the objective nowhere in it is not a descent
    /// direction of this criterion, whatever the matrix says.
    ///
    /// This is a statement about the MATRIX, not about the point, and it is the
    /// reason the second-order conjunct does not refuse here: refusing would
    /// spend evidence the criterion has just contradicted. It is deliberately
    /// NOT [`Self::Measured`] with `psd: true` — nothing established that the
    /// point is a minimum either; what was established is that this Hessian's
    /// negative direction has no operational content.
    ///
    /// Reachable only where an analytic Hessian exists AND the objective can be
    /// re-evaluated at trial points, which is exactly where the escape runs.
    ///
    /// Serializes as `null` under the legacy `hessian_psd` key and reloads as
    /// [`Self::NotAvailable`] — the same deliberate lossiness the module doc
    /// already records for `NotSpent`/`NoEstimand`, and sound for the same
    /// reason: the four are acceptance-identical, and the adjudication is a
    /// statement about a run, not a property of a stored model.
    CriterionContradicted,
    /// A Hessian was measured and reported a negative direction whose claim
    /// the criterion **cannot resolve at any step the adjudication may take**
    /// (#3036): even the largest step predicts a decrease `½|λ_min|·α_max²`
    /// under the criterion's resolution, so no trial could confirm or falsify
    /// it and none was evaluated.
    ///
    /// Like [`Self::CriterionContradicted`] this withdraws the matrix's verdict
    /// without inverting it, and it is admissible: a negative direction below
    /// the instrument's own resolution is exactly what
    /// [`CurvatureAdmissibility::Admissible`] already admits through the
    /// gradient-residue floor. Serializes as `null` and reloads as
    /// [`Self::NotAvailable`], for the reason recorded on that variant.
    CriterionUnresolvable,
}

impl CurvatureEvidence {
    /// The raw PSD verdict when one was measured, `None` otherwise — the
    /// legacy projection, and what the published `hessian_psd` surface shows.
    pub fn psd(self) -> Option<bool> {
        match self {
            Self::Measured { psd } => Some(psd),
            Self::NotSpent
            | Self::NotAvailable
            | Self::NoEstimand
            | Self::CriterionContradicted
            | Self::CriterionUnresolvable => None,
        }
    }

    /// Whether the criterion's adjudication withdrew a measured negative
    /// direction: it contradicted the claim over its falsifiable range
    /// (#2612), or that range is empty (#3036). Either way the matrix is wrong
    /// along that direction by at least `|λ_min|` at the criterion's
    /// resolution.
    pub fn withdrawn_by_criterion(self) -> bool {
        match self {
            Self::CriterionContradicted | Self::CriterionUnresolvable => true,
            Self::Measured { .. } | Self::NotSpent | Self::NotAvailable | Self::NoEstimand => false,
        }
    }

    /// Build from a raw optional measurement: `Some` was measured, `None`
    /// means the route had no analytic Hessian to test. Sites that mean
    /// [`Self::NotSpent`] or [`Self::NoEstimand`] must say so explicitly —
    /// that is the point of the type.
    pub(crate) fn from_measurement(psd: Option<bool>) -> Self {
        match psd {
            Some(psd) => Self::Measured { psd },
            None => Self::NotAvailable,
        }
    }
}

impl From<Option<bool>> for CurvatureEvidence {
    fn from(psd: Option<bool>) -> Self {
        Self::from_measurement(psd)
    }
}

impl From<CurvatureEvidence> for Option<bool> {
    fn from(evidence: CurvatureEvidence) -> Self {
        evidence.psd()
    }
}

impl std::fmt::Display for CurvatureEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `yes`/`NO` are asserted on downstream (run_plan_tests) and read by
        // humans in refusal text; they must not drift.
        f.write_str(match self {
            Self::Measured { psd: true } => "yes",
            Self::Measured { psd: false } => "NO",
            Self::NotSpent => "not-spent",
            Self::NotAvailable => "n/a",
            Self::NoEstimand => "no-estimand",
            Self::CriterionContradicted => "criterion-contradicted",
            Self::CriterionUnresolvable => "criterion-unresolvable",
        })
    }
}

/// What second-order admissibility a certificate can actually claim (#2578).
///
/// The three variants are the three states of the question, and they are NOT
/// two passes and a failure: `Unevaluated` is the absence of evidence, and it
/// is reported as such rather than folded into the pass branch. It does not
/// refuse — refusing every route with no analytic Hessian (the EFS/fixed-point
/// route, the support-sparse grouped-LAML lane) would be wrong — but a consumer
/// that needs a real second-order guarantee can now distinguish it, which is
/// precisely what a bare `bool` made impossible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurvatureAdmissibility {
    /// A Hessian existed at the certified point and it is admissible: measured
    /// PSD, or measured indefinite with its most negative direction below the
    /// instrument's own resolution.
    Admissible,
    /// A Hessian existed and it is indefinite by more than the floor allows.
    /// `floor_consulted` records whether a floor verdict was available at all,
    /// so the refusal can say whether the negative direction was judged against
    /// a resolution or simply taken at face value.
    Inadmissible { floor_consulted: bool },
    /// No curvature question was answered here, and this is why.
    Unevaluated { evidence: CurvatureEvidence },
    /// A Hessian was measured, reported a negative direction, and the CRITERION
    /// was then asked about that direction and did not fall anywhere in the
    /// claim's falsifiable range (#2612). Distinct from every variant above:
    /// it is not admissible (nothing showed the point is a minimum), it is not
    /// inadmissible (the evidence that would refuse has been contradicted), and
    /// it is not unevaluated (a measurement was taken and then adjudicated).
    CriterionContradicted,
}

impl std::fmt::Display for CurvatureAdmissibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admissible => f.write_str("admissible"),
            Self::Inadmissible { floor_consulted } => write!(
                f,
                "INADMISSIBLE (floor {})",
                if *floor_consulted {
                    "consulted, not cleared"
                } else {
                    "not available"
                }
            ),
            Self::Unevaluated { evidence } => write!(f, "unevaluated ({evidence})"),
            Self::CriterionContradicted => f.write_str(
                "CONTRADICTED (the criterion does not fall along the reported negative direction)",
            ),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OuterCriterionCertificate {
    pub stationarity: OuterStationarityCertificate,
    /// What curvature evidence this certificate has, and why it has no more
    /// than that (#2561). Serializes as the legacy `hessian_psd` optional
    /// bool; read the raw verdict through [`Self::hessian_psd`].
    #[serde(rename = "hessian_psd")]
    pub curvature: CurvatureEvidence,
    /// Leading smoothing coordinates (ρ block) pinned within
    /// `CERTIFICATE_RAIL_MARGIN` of either box bound at the optimum.
    pub lambdas_railed: Vec<usize>,
    /// The interval and margin each railed coordinate was judged against
    /// (#2530). Empty when nothing is railed. `#[serde(default)]` so a
    /// certificate stored before this field existed still deserializes — those
    /// carry no facts, which is honest: none were recorded.
    #[serde(default)]
    pub railed_facts: Vec<RailedCoordinateFact>,
    /// The gradient-residue floor's verdict on that same curvature, when it was
    /// computed. `hessian_psd` above stays the raw measurement; this records
    /// whether the most negative interior direction is below the instrument's
    /// own resolution, and by how much.
    #[serde(default)]
    pub curvature_floor: Option<CurvatureFloorClearance>,
    /// The Newton steps the mint took from the point the search handed over
    /// before this certificate judged it (#2954). `None` when none were taken,
    /// and for a certificate stored before this field existed.
    #[serde(default)]
    pub newton_polish: Option<NewtonPolishRecord>,
    /// How far the reported criterion value can sit from the criterion's true
    /// minimum, when the certificate's own evidence bounds it (#3331). `Some`
    /// only when the Newton decrement certified the point: the decrement is the
    /// one verdict that bounds the decrease left to the minimum. A first-order
    /// (gradient-band) certificate bounds the slope, not the value, so it
    /// carries `None`, as does a certificate stored before this field existed.
    #[serde(default)]
    pub criterion_error: Option<CriterionErrorBound>,
}

/// The error the certified criterion value carries (#3331), in the
/// criterion's own units.
///
/// Two independent parts. `decrease_left` is `λ̂² + band_λ²`, the Newton
/// decrement's bound on `V(ρ̂) − min V` together with the decrement's own
/// propagated rounding: the reported value can sit this far ABOVE the minimum.
/// `value_band` is the rounding band of the evaluation of `V(ρ̂)` itself: the
/// reported value can sit this far from the exact `V(ρ̂)` in either direction.
/// Comparing two fits' criteria is decided only outside the sum of both fits'
/// bounds; inside it the arithmetic has not separated them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CriterionErrorBound {
    /// `λ̂² + band_λ²` at the certified point.
    pub decrease_left: f64,
    /// The rounding band of the criterion value at the certified point.
    pub value_band: f64,
}

impl CriterionErrorBound {
    /// The reported value's error below its exact minimum is at most
    /// `value_band`; above it, at most `decrease_left + value_band`.
    pub fn above_minimum(&self) -> f64 {
        self.decrease_left + self.value_band
    }
}

/// The Newton polish a mint took before certifying (#2954).
///
/// The mint's Newton-decrement verdict is stricter than any search stop, so the
/// point a search hands over can still buy a decrease the arithmetic resolves.
/// The mint takes Newton steps on the free coordinates while each lowers the
/// criterion by more than its band and the decrement contracts, and judges the
/// point they reach. Where only the decrease left is resolvable, not a step's,
/// the last step is a settling step whose point must certify. A coordinate
/// whose Newton steps contract slower than Newton's quadratic rate toward the
/// box bound they head to is railed there when that lowers the criterion by
/// more than its band (projected Newton), and the coordinates left free are
/// polished on their own face.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewtonPolishRecord {
    /// `λ̂²` at the point the search handed over.
    pub lambda_sq_before: f64,
    /// `λ̂²` at the point this certificate judged.
    pub lambda_sq_after: f64,
    /// The criterion decrease each accepted step bought, in order. A settling
    /// step's is within the band and may be negative.
    pub decreases: Vec<f64>,
    /// Whether the last step was a settling step (#3012): the decrease left to
    /// the minimum was resolvable, the full step's own model decrease was not.
    #[serde(default)]
    pub settled: bool,
    /// The coordinates the polish railed, in order. `#[serde(default)]` so a
    /// record stored before this field existed still deserializes.
    #[serde(default)]
    pub rails: Vec<NewtonPolishRail>,
    /// The θ point the search handed over, where the polish began.
    #[serde(default)]
    pub entry: Vec<f64>,
}

/// One coordinate a mint's polish moved to its box bound (#2954).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewtonPolishRail {
    /// The θ coordinate railed.
    pub index: usize,
    /// Its value before the rail, and the bound it was moved to.
    pub from: f64,
    pub to: f64,
    /// `V(from) − V(to)`, never negative.
    pub decrease: f64,
    /// How many Newton steps the polish had taken before this rail.
    pub steps_before: usize,
    /// The kind of face it was railed at. A polish rails only at a limit model.
    pub face: RailFaceKind,
}

impl OuterCriterionCertificate {
    /// First-order (KKT) stationarity: the projected analytic gradient or
    /// root-equivalent fixed-point residual clears its declared bound.
    pub fn is_stationary(&self) -> bool {
        self.stationarity.projected_norm().is_finite()
            && self.stationarity.projected_norm() <= self.stationarity.bound()
    }

    /// The raw PSD verdict, `None` when no curvature question was answered
    /// — the legacy `hessian_psd` projection every existing consumer wants.
    /// Ask [`Self::curvature`] directly to learn WHY there is no verdict.
    pub fn hessian_psd(&self) -> Option<bool> {
        self.curvature.psd()
    }

    /// Second-order admissibility: a certified optimum must not sit on
    /// genuinely indefinite analytic curvature. A nearby box rail is only a
    /// diagnostic; it cannot waive negative curvature in unrelated free
    /// directions. If a future certificate projects onto the exact critical
    /// cone, that projected result can be recorded here instead.
    ///
    /// Three-valued, because the question has three answers (#2578). The
    /// predicate this replaced returned `bool`, and its `true` was produced by
    /// two structurally different facts: a Hessian that was measured and found
    /// admissible, and a route that never measured one. A gate whose passing
    /// condition is satisfied by the ABSENCE of an observation cannot fail on
    /// the case it exists to catch — on the support-sparse grouped-LAML lane,
    /// which reports `hessian=Unavailable`, every consultation answered `true`
    /// without inspecting anything.
    ///
    /// Acceptance is unchanged. `CurvatureAdmissibility::Unevaluated` does
    /// not refuse, exactly as `!= Some(false)` did not. The difference is that
    /// a caller wanting a genuine second-order guarantee can now ask for
    /// `CurvatureAdmissibility::Admissible` and get one, instead of being
    /// handed a `true` that means "nobody looked".
    pub fn curvature_verdict(&self) -> CurvatureAdmissibility {
        match self.curvature {
            CurvatureEvidence::Measured { psd: true } => CurvatureAdmissibility::Admissible,
            CurvatureEvidence::Measured { psd: false } => {
                // Only a measured `false` consults the floor, and then only to
                // ask whether that negative direction was distinguishable from
                // zero at all.
                let cleared = self
                    .curvature_floor
                    .is_some_and(|clearance| clearance.cleared);
                if cleared {
                    CurvatureAdmissibility::Admissible
                } else {
                    CurvatureAdmissibility::Inadmissible {
                        floor_consulted: self.curvature_floor.is_some(),
                    }
                }
            }
            CurvatureEvidence::CriterionContradicted => {
                CurvatureAdmissibility::CriterionContradicted
            }
            CurvatureEvidence::CriterionUnresolvable => CurvatureAdmissibility::Admissible,
            evidence @ (CurvatureEvidence::NotSpent
            | CurvatureEvidence::NotAvailable
            | CurvatureEvidence::NoEstimand) => CurvatureAdmissibility::Unevaluated { evidence },
        }
    }

    /// Whether the second-order conjunct REFUSES this certificate — the shape
    /// [`Self::refusal`] needs. This is deliberately not named
    /// `curvature_admissible`: "did not refuse" and "was found admissible" are
    /// different claims, and conflating them is what #2578 was.
    pub(crate) fn curvature_not_refused(&self) -> bool {
        !matches!(
            self.curvature_verdict(),
            CurvatureAdmissibility::Inadmissible { .. }
        )
    }

    /// Whether the certificate accepts the returned point as a constrained
    /// minimum. This is the load-bearing verdict: a `false` here rejects the
    /// fit with typed non-convergence.
    ///
    /// Defined as "no conjunct refused" so that this and [`Self::refusal`]
    /// cannot disagree. Two renderings of one predicate is exactly what #2550
    /// was: the verdict string named a cause the predicate had not tested.
    pub fn certifies(&self) -> bool {
        self.refusal().is_none()
    }

    /// WHICH conjunct refused, or `None` when the certificate accepts.
    ///
    /// `certifies()` is a conjunction of six independent things and its verdict
    /// string used to be a three-way branch, so the middle branch had to GUESS
    /// among four surviving culprits — and guessed curvature every time. It
    /// printed `INDEFINITE CURVATURE AT INTERIOR OPTIMUM` four words after
    /// printing `hessian_psd=yes`, on a fit whose actual refusal was a rail
    /// with a zero pencil constant (#2550, observed at #2348's increment 2).
    /// A misdirecting refusal is worse than a vague one: it spends the reader's
    /// attention on the one component the same line proves is not at fault.
    ///
    /// The order is the conjunction's own. It is reported first-failure-wins
    /// because that is what a conjunction means; a certificate failing several
    /// conjuncts is not better described by listing them than by naming the
    /// first thing that was wrong with it.
    pub fn refusal(&self) -> Option<CertificationRefusal> {
        if !self.stationarity.raw_norm().is_finite()
            || self.stationarity.raw_norm() < 0.0
            || self.stationarity.projected_norm() < 0.0
        {
            return Some(CertificationRefusal::UnusableNorms {
                raw: self.stationarity.raw_norm(),
                projected: self.stationarity.projected_norm(),
            });
        }
        if !self.stationarity.bound().is_finite() || self.stationarity.bound() < 0.0 {
            return Some(CertificationRefusal::UnusableBound {
                bound: self.stationarity.bound(),
            });
        }
        if let Some(malformed) = self.first_malformed_rail() {
            return Some(malformed);
        }
        if !self.is_stationary() {
            return Some(CertificationRefusal::NotStationary {
                projected: self.stationarity.projected_norm(),
                bound: self.stationarity.bound(),
            });
        }
        if let CurvatureAdmissibility::Inadmissible { floor_consulted } = self.curvature_verdict() {
            return Some(CertificationRefusal::InadmissibleCurvature {
                floor_consulted,
                floor: self.curvature_floor,
            });
        }
        None
    }

    /// An `AsymptoteRail` certificate is only admissible when it carries at
    /// least one rail and every rail's certified facts are finite, with
    /// non-negative gaps and a pencil constant its own route admits. Non-rail
    /// certificates trivially pass. This closes the door on a deserialized rail
    /// certificate with an empty or non-finite `rails` list minting
    /// convergence.
    ///
    /// The pencil-constant rule is delegated to [`RailTailEvidence::admits`]
    /// rather than fixed here at `ĉ > 0`. That flat rule was the MEASURED
    /// route's, applied to both: it silently un-certified any analytically
    /// proven face carrying an unidentified coordinate, whose proven constant is
    /// exactly zero. The mint logged a successful proof and `certifies()` then
    /// returned false with nothing recording why — reachable only on a face of
    /// two or more coordinates, which no single-coordinate fixture produces.
    /// The first rail that fails its own route's admissibility, named.
    ///
    /// A boolean here was enough for `certifies()` and never enough for a
    /// reader: "some rail is malformed" among a list of rails, each with its
    /// own route and its own rule, is the shape of message that sends someone
    /// to read all of them. The route is carried because the pencil-constant
    /// rule is delegated to [`RailTailEvidence::admits`] and differs between
    /// routes — a proven face's constant of exactly zero is admissible on the
    /// analytic route and not on the measured one, which is precisely the
    /// distinction that produced #2550's observed misdirection.
    fn first_malformed_rail(&self) -> Option<CertificationRefusal> {
        let OuterStationarityCertificate::AsymptoteRail { rails, .. } = &self.stationarity else {
            return None;
        };
        if rails.is_empty() {
            return Some(CertificationRefusal::NoRailOnRailCertificate);
        }
        rails.iter().find_map(|rail| {
            let fault = if !rail.evidence.admits(rail.tail_constant) {
                RailFault::TailConstantRefusedByRoute
            } else if !(rail.value_gap.is_finite() && rail.value_gap >= 0.0) {
                RailFault::UnusableValueGap
            } else if !(rail.estimand_travel_bound.is_finite() && rail.estimand_travel_bound >= 0.0)
            {
                RailFault::UnusableTravelBound
            } else {
                return None;
            };
            Some(CertificationRefusal::MalformedRail {
                index: rail.index,
                route: rail.evidence.route(),
                tail_constant: rail.tail_constant,
                fault,
            })
        })
    }

    /// Whether every audited fact is clean (stationary, PSD-or-untracked
    /// curvature, no railed smoothing coordinate) — the report-level verdict.
    pub fn is_clean(&self) -> bool {
        self.certifies() && self.curvature.psd() != Some(false) && self.lambdas_railed.is_empty()
    }

    /// One-line human-readable rendering for logs and reports.
    pub fn summary(&self) -> String {
        let stationarity = match &self.stationarity {
            OuterStationarityCertificate::AnalyticGradient {
                grad_norm,
                projected_grad_norm,
                bound,
                rung,
            } => format!(
                "gradient |g|={grad_norm:.3e} |Pg|={projected_grad_norm:.3e} bound={bound:.3e} ({rung})"
            ),
            OuterStationarityCertificate::FixedPoint {
                residual_inf_norm,
                projected_residual_inf_norm,
                bound,
                rung,
                covered_coordinates,
            } => format!(
                "fixed-point |r|inf={residual_inf_norm:.3e} |Pr|inf={projected_residual_inf_norm:.3e} bound={bound:.3e} ({rung}) covered={covered_coordinates}"
            ),
            OuterStationarityCertificate::AsymptoteRail {
                interior_projected_grad_norm,
                bound,
                rung,
                rails,
            } => {
                let rail_summary = rails
                    .iter()
                    .map(|rail| {
                        format!(
                            "#{}{} ĉ={:.3e} gap={:.3e} travel={:.3e} via {}",
                            rail.index,
                            match rail.side {
                                crate::rho_optimizer::asymptote_certificate::AsymptoteSide::Upper =>
                                    "↑",
                                crate::rho_optimizer::asymptote_certificate::AsymptoteSide::Lower =>
                                    "↓",
                            },
                            rail.tail_constant,
                            rail.value_gap,
                            rail.estimand_travel_bound,
                            rail.evidence.route(),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "asymptote-rail |Pg_interior|={interior_projected_grad_norm:.3e} bound={bound:.3e} ({rung}) rails=[{rail_summary}]"
                )
            }
        };
        let railed = if self.railed_facts.is_empty() {
            format!("{:?}", self.lambdas_railed)
        } else {
            format!(
                "{:?} [{}]",
                self.lambdas_railed,
                self.railed_facts
                    .iter()
                    .map(|fact| fact.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let curvature_source = match self.curvature {
            CurvatureEvidence::Measured { .. } => "terminal-analytic",
            CurvatureEvidence::NotSpent => "screening-not-spent",
            CurvatureEvidence::NotAvailable => "unavailable",
            CurvatureEvidence::NoEstimand => "no-estimand",
            // The SOURCE is still the terminal analytic Hessian — that is where
            // the (withdrawn) verdict came from. Naming the adjudication here
            // keeps the source honest while `hessian_psd=criterion-contradicted`
            // beside it carries what happened to it.
            CurvatureEvidence::CriterionContradicted => "terminal-analytic-contradicted",
            CurvatureEvidence::CriterionUnresolvable => "terminal-analytic-unresolvable",
        };
        let verdict = match self.refusal() {
            None => "stationary".to_string(),
            Some(refusal) => refusal.to_string(),
        };
        // A `psd=false` that was nonetheless admitted was admitted BY the
        // gradient-residue floor, and the numbers that admitted it are already
        // on the certificate — they were simply never rendered (#2748). A line
        // reading `hessian_psd=NO ... → stationary` is unreadable without them,
        // and the downstream smoothing correction re-judges the same direction
        // and prints its OWN full ledger when it refuses; with both rendered the
        // two standards can be compared in one log instead of inferred from two.
        let curvature_detail = match (self.curvature, self.curvature_floor) {
            (CurvatureEvidence::Measured { psd: false }, Some(clearance)) => format!(
                " (cleared={} λ_min(H)={:.6e} λ_min(H+diag|g|)={:.6e} max_k|g_k|={:.6e} \
                 measured_resolution={:.6e} decided_at={:.6e})",
                clearance.cleared,
                clearance.interior_min_eigenvalue,
                clearance.floored_min_eigenvalue,
                clearance.gradient_floor,
                clearance.measured_resolution,
                clearance.decided_at_resolution,
            ),
            _ => String::new(),
        };
        format!(
            "{stationarity} hessian_psd={}{curvature_detail} curvature_source={curvature_source} railed={} → {verdict}",
            self.curvature, railed,
        )
    }
}

/// Which conjunct of [`OuterCriterionCertificate::certifies`] refused.
///
/// Exists so a refusal cannot name a cause that was not the one tested (#2550).
/// Every rendering of a non-certifying verdict is built from this, so there is
/// no second place for a verdict string to be inferred.
// No `Eq`: two variants carry the `f64` quantities that decided the refusal,
// and reporting them is the point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CertificationRefusal {
    /// A stationarity norm is non-finite or negative — no comparison is
    /// possible, so nothing downstream of it was tested.
    UnusableNorms { raw: f64, projected: f64 },
    /// The bound the norms would be compared against is non-finite or negative.
    UnusableBound { bound: f64 },
    /// A rail certificate carrying no rail at all. Deserialization can produce
    /// this; a mint cannot.
    NoRailOnRailCertificate,
    /// A rail whose certified facts its own route does not admit.
    MalformedRail {
        index: usize,
        route: &'static str,
        tail_constant: f64,
        fault: RailFault,
    },
    /// The projected norm exceeds its bound: the point is not a stationary
    /// point of the outer problem.
    NotStationary { projected: f64, bound: f64 },
    /// Measured indefinite curvature that no floor clearance excused. **The
    /// only variant permitted to speak about curvature.**
    InadmissibleCurvature {
        floor_consulted: bool,
        /// The clearance actually consulted, so the refusal can say by how much
        /// the floor missed rather than only that it did. `None` when no floor
        /// was consulted at all.
        floor: Option<CurvatureFloorClearance>,
    },
}

/// Which of a rail's certified facts its route refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RailFault {
    /// The pencil constant is outside what this rail's route admits. The rule
    /// is the route's, not a fixed `ĉ > 0`.
    TailConstantRefusedByRoute,
    UnusableValueGap,
    UnusableTravelBound,
}

impl std::fmt::Display for RailFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TailConstantRefusedByRoute => "pencil constant not admitted by its route",
            Self::UnusableValueGap => "non-finite or negative value gap",
            Self::UnusableTravelBound => "non-finite or negative estimand travel bound",
        })
    }
}

impl std::fmt::Display for CertificationRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnusableNorms { raw, projected } => write!(
                f,
                "UNUSABLE STATIONARITY NORMS (|g|={raw:.3e}, |Pg|={projected:.3e})"
            ),
            Self::UnusableBound { bound } => {
                write!(f, "UNUSABLE STATIONARITY BOUND ({bound:.3e})")
            }
            Self::NoRailOnRailCertificate => f.write_str("RAIL CERTIFICATE CARRIES NO RAIL"),
            Self::MalformedRail {
                index,
                route,
                tail_constant,
                fault,
            } => write!(
                f,
                "MALFORMED RAIL #{index} via {route}: {fault} (ĉ={tail_constant:.3e})"
            ),
            Self::NotStationary { projected, bound } => write!(
                f,
                "NOT STATIONARY (|Pg|={projected:.3e} > bound={bound:.3e})"
            ),
            // The historical wording, kept verbatim: downstream refusal text is
            // asserted on it, and this is now the ONLY branch that can emit it.
            Self::InadmissibleCurvature {
                floor_consulted,
                floor,
            } => {
                f.write_str("INDEFINITE CURVATURE AT INTERIOR OPTIMUM")?;
                if *floor_consulted {
                    f.write_str(" (curvature floor did not clear)")?;
                }
                // Both measured quantities, appended AFTER the historical
                // wording so the substrings downstream asserts on are
                // untouched. Without these a reader cannot tell a genuine
                // saddle from a negative eigenvalue at the noise floor of a
                // nearly-flat direction, which is the whole question at large
                // rho -- and the two call for opposite responses.
                if let Some(clearance) = floor {
                    write!(
                        f,
                        " [interior lambda_min={:.3e}, gradient_floor={:.3e}, \
                         floored lambda_min={:.3e} (this is the one the verdict was taken on), \
                         measured resolution={:.3e}]",
                        clearance.interior_min_eigenvalue,
                        clearance.gradient_floor,
                        clearance.floored_min_eigenvalue,
                        clearance.measured_resolution
                    )?;
                }
                Ok(())
            }
        }
    }
}

/// Sealed proof that both optimization layers supporting a fitted model
/// reached certified optima.
///
/// The fields are private and deserialization revalidates them, so downstream
/// code cannot manufacture this proof from status booleans.  A
/// [`UnifiedFitResult`] owns one of these proofs; therefore the existence of a
/// fitted result is itself the convergence verdict.
#[derive(Clone, Debug, Serialize)]
pub struct FitConvergenceEvidence {
    inner_status: crate::pirls::PirlsStatus,
    outer_iterations: usize,
    outer: FitOuterConvergenceEvidence,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum FitOuterConvergenceEvidence {
    /// No smoothing coordinate was optimized. There is no outer stationarity
    /// equation to solve, so a converged inner mode is the complete proof.
    Fixed,
    /// Analytic constrained-stationarity/curvature proof at the selected rho.
    Analytic(OuterCriterionCertificate),
}

#[derive(Deserialize)]
struct SerializedFitConvergenceEvidence {
    inner_status: crate::pirls::PirlsStatus,
    outer_iterations: usize,
    outer: FitOuterConvergenceEvidence,
}

impl FitConvergenceEvidence {
    fn from_serialized(raw: SerializedFitConvergenceEvidence) -> Result<Self, String> {
        // Deserialization cannot weaken the live fit-minting contract. A
        // stalled checkpoint remains diagnostic state, not a fitted model.
        if !raw.inner_status.is_converged() {
            return Err(format!(
                "inner optimizer status {:?} is not converged",
                raw.inner_status
            ));
        }
        match &raw.outer {
            FitOuterConvergenceEvidence::Fixed => {
                if raw.outer_iterations != 0 {
                    return Err(format!(
                        "fixed-outer convergence evidence cannot carry {} outer iterations",
                        raw.outer_iterations
                    ));
                }
            }
            FitOuterConvergenceEvidence::Analytic(certificate) => {
                if !certificate.certifies() {
                    return Err(format!(
                        "analytic outer convergence evidence does not certify: {}",
                        certificate.summary()
                    ));
                }
            }
        }
        Ok(Self {
            inner_status: raw.inner_status,
            outer_iterations: raw.outer_iterations,
            outer: raw.outer,
        })
    }

    /// The diagnostic terminal status of the certified inner solve.
    pub fn inner_status(&self) -> crate::pirls::PirlsStatus {
        self.inner_status
    }

    /// Number of outer iterations covered by this proof.
    pub fn outer_iterations(&self) -> usize {
        self.outer_iterations
    }

    /// Analytic outer certificate, or `None` when no outer coordinate existed.
    pub fn outer_certificate(&self) -> Option<&OuterCriterionCertificate> {
        match &self.outer {
            FitOuterConvergenceEvidence::Fixed => None,
            FitOuterConvergenceEvidence::Analytic(certificate) => Some(certificate),
        }
    }
}

impl<'de> Deserialize<'de> for FitConvergenceEvidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = SerializedFitConvergenceEvidence::deserialize(deserializer)?;
        Self::from_serialized(raw).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug)]
pub struct FitOptions {
    /// Resource contract used by every basis realization and spatial
    /// hyperparameter rebuild belonging to this fit. Keeping it on the fit
    /// lifecycle prevents a policy used during formula lowering from being
    /// silently replaced by the library default when the design is built.
    pub resource_policy: gam_runtime::resource::ResourcePolicy,
    pub latent_cloglog: Option<LatentCLogLogState>,
    pub mixture_link: Option<MixtureLinkSpec>,
    pub optimize_mixture: bool,
    pub sas_link: Option<SasLinkSpec>,
    pub optimize_sas: bool,
    pub compute_inference: bool,
    /// Internal lifecycle knob for fits whose result will be immediately
    /// superseded. Keeps ordinary inference work but skips the live-objective
    /// rho posterior adequacy diagnostic/escalation until the returned model is known.
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
    /// Fixed prior on smoothing parameters for explicit joint HMC sampling
    /// flows.
    ///
    /// This prior is part of the sampled target itself, unlike `rho_mode`,
    /// which is only used to initialize chains near the REML solution.
    pub rho_prior: gam_problem::RhoPrior,
    /// Explicit cross-process warm-start capability. `None` is disk-silent;
    /// `Some` carries one lazy/opened caller-configured store through every
    /// standard REML owner.
    pub persistent_warm_start_store: Option<gam_runtime::warm_start::ConfiguredWarmStartStore>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
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
            rho_prior: gam_problem::RhoPrior::default(),
            persistent_warm_start_store: None,
        }
    }
}

/// SPEC's criterion contract, stated executably.
///
/// SPEC: *"REML (or LAML) always used for fitting, never GCV"* and *"posterior
/// mean must always be the default (never MAP)"*. Those are boundaries on what
/// the shipped deterministic criterion IS, and a boundary stated only in prose
/// is not stated. `FitOptions::default()` is the single seam every
/// `gam-models`, CLI and Python fit passes through, so it is where the contract
/// is checkable.
///
/// #2450 is what happens without this gate: `RhoPrior::default()` was
/// `Normal { mean: 0, sd: 3 }`, so the shipped criterion was `REML + Σρ²/18` —
/// MAP in ρ, with an underived `sd = 3.0` — for as long as nobody re-read the
/// `Default` impl. The damage was not only statistical: a prior whose gradient
/// survives into the λ→∞ tail makes `ĉ = −e^ρ ∂V/∂ρ` divergent, and the
/// measured rail-reasoning paths (`try_certify_asymptote_rail`,
/// `detect_wrong_rail_pullback`) decide by testing that `ĉ` is CONSTANT. One
/// `Default` disabled the face certificate and the repair path for a
/// coordinate stuck on the wrong bound.
#[cfg(test)]
mod tests_certification_refusal_2550 {
    use super::{
        CertificationRefusal, CurvatureAdmissibility, CurvatureEvidence, OuterCriterionCertificate,
        FacePositivityRoute, OuterStationarityCertificate, RailCoordinate, RailFault,
        RailTailEvidence,
    };
    use crate::rho_optimizer::asymptote_certificate::AsymptoteSide;

    fn rung() -> super::CertifiedRung {
        gam_problem::StationarityRung {
            label: "solver-band",
            derived_standard: false,
        }
        .into()
    }

    /// A rail the ANALYTIC route admits: proven face, non-negative constant.
    fn proven_rail(index: usize, tail_constant: f64) -> RailCoordinate {
        RailCoordinate {
            index,
            side: AsymptoteSide::Upper,
            tail_constant,
            value_gap: 2.6e-5,
            estimand_travel_bound: 1.0e-9,
            evidence: RailTailEvidence::AnalyticFaceProof {
                route: FacePositivityRoute::PositiveForm,
                statistic: 3.5,
                band: 1.0e-12,
            },
        }
    }

    /// Stationary, admissible curvature, well-formed rails — the certificate
    /// every case below perturbs in EXACTLY ONE conjunct.
    fn certifying(rails: Vec<RailCoordinate>) -> OuterCriterionCertificate {
        OuterCriterionCertificate {
            stationarity: OuterStationarityCertificate::AsymptoteRail {
                interior_projected_grad_norm: 1.0e-9,
                bound: 1.0e-6,
                rung: rung(),
                rails,
            },
            curvature: CurvatureEvidence::Measured { psd: true },
            lambdas_railed: vec![0, 1],
            railed_facts: Vec::new(),
            newton_polish: None,
            curvature_floor: None,
            criterion_error: None,
        }
    }

    /// The baseline must CERTIFY, or every negative control below is vacuous:
    /// a fixture that already refuses proves nothing about which conjunct the
    /// verdict names.
    #[test]
    fn the_unperturbed_certificate_certifies() {
        let certificate = certifying(vec![proven_rail(0, 4.25), proven_rail(1, 0.0)]);
        assert_eq!(certificate.refusal(), None, "{}", certificate.summary());
        assert!(certificate.certifies());
        assert!(certificate.summary().contains("stationary"));
    }

    /// ⭐ #2550's observed case, as a gate.
    ///
    /// A rail whose pencil constant its own route refuses, with curvature
    /// MEASURED PSD and the point stationary. The old three-way verdict had
    /// only "INDEFINITE CURVATURE AT INTERIOR OPTIMUM" available for this
    /// branch, so it printed that four words after printing `hessian_psd=yes`
    /// — naming the one component the same line proves is not at fault.
    #[test]
    fn a_malformed_rail_is_not_blamed_on_curvature() {
        // ProbedTail refuses a non-positive constant; the analytic route does
        // not. Using the measured route with c = 0 fails EXACTLY the rails
        // conjunct.
        let mut rails = vec![proven_rail(0, 4.25), proven_rail(1, 0.0)];
        rails[1].evidence = RailTailEvidence::ProbedTail {
            gradient_band: 1.0e-12,
            extrapolation_radius: 0.0,
        };
        let certificate = certifying(rails);

        assert!(
            certificate.is_stationary(),
            "the fixture must be stationary, or it exercises the wrong branch"
        );
        assert_eq!(
            certificate.curvature_verdict(),
            CurvatureAdmissibility::Admissible,
            "the fixture's curvature must be MEASURED admissible, or the blame would be              earned -- an unevaluated curvature would satisfy a mere not-refused test              without exercising this branch at all (#2578)"
        );
        assert!(!certificate.certifies());

        let summary = certificate.summary();
        assert!(
            !summary.contains("CURVATURE"),
            "a refusal caused by a rail must not name curvature, least of all on a line \
             that also prints hessian_psd=yes: {summary}"
        );
        assert!(
            summary.contains("hessian_psd=yes"),
            "the contradicting evidence is what made this defect visible; keep it: {summary}"
        );
        assert!(
            matches!(
                certificate.refusal(),
                Some(CertificationRefusal::MalformedRail {
                    index: 1,
                    fault: RailFault::TailConstantRefusedByRoute,
                    ..
                })
            ),
            "the refusal must name WHICH rail and WHICH of its facts: {:?}",
            certificate.refusal()
        );
        assert!(
            summary.contains("MALFORMED RAIL #1"),
            "the rendered verdict must carry the rail's index: {summary}"
        );
    }

    /// The curvature arm still says what it always said, and is the only arm
    /// that may. Downstream refusal text is asserted on this wording.
    #[test]
    fn inadmissible_curvature_is_still_named_curvature() {
        let mut certificate = certifying(vec![proven_rail(0, 4.25)]);
        certificate.curvature = CurvatureEvidence::Measured { psd: false };

        assert!(certificate.is_stationary());
        assert!(certificate.first_malformed_rail().is_none());
        assert!(!certificate.certifies());
        assert!(matches!(
            certificate.refusal(),
            Some(CertificationRefusal::InadmissibleCurvature { .. })
        ));
        let summary = certificate.summary();
        assert!(
            summary.contains("INDEFINITE CURVATURE AT INTERIOR OPTIMUM"),
            "the historical wording must survive for the case that earns it: {summary}"
        );
        assert!(summary.contains("hessian_psd=NO"), "{summary}");
    }

    /// A non-stationary point names stationarity and carries the comparison
    /// that decided it, rather than the bare words.
    #[test]
    fn a_non_stationary_point_names_stationarity_with_its_comparison() {
        let certificate = OuterCriterionCertificate {
            stationarity: OuterStationarityCertificate::AsymptoteRail {
                interior_projected_grad_norm: 1.0e-2,
                bound: 1.0e-6,
                rung: rung(),
                rails: vec![proven_rail(0, 4.25)],
            },
            curvature: CurvatureEvidence::Measured { psd: true },
            lambdas_railed: vec![0],
            railed_facts: Vec::new(),
            newton_polish: None,
            curvature_floor: None,
            criterion_error: None,
        };
        assert!(matches!(
            certificate.refusal(),
            Some(CertificationRefusal::NotStationary { .. })
        ));
        let summary = certificate.summary();
        assert!(summary.contains("NOT STATIONARY"), "{summary}");
        assert!(
            !summary.contains("CURVATURE"),
            "curvature is admissible here and must not be named: {summary}"
        );
    }

    /// A rail certificate with no rail at all is its own refusal, not a
    /// malformed one — deserialization can produce it and a mint cannot.
    #[test]
    fn a_rail_certificate_with_no_rail_says_so() {
        let certificate = certifying(Vec::new());
        assert_eq!(
            certificate.refusal(),
            Some(CertificationRefusal::NoRailOnRailCertificate)
        );
        let summary = certificate.summary();
        assert!(summary.contains("CARRIES NO RAIL"), "{summary}");
        assert!(!summary.contains("CURVATURE"), "{summary}");
    }

    /// `certifies()` and `refusal()` are one predicate, so they cannot drift.
    /// This is the invariant the whole change rests on: the defect was two
    /// renderings of one question disagreeing.
    #[test]
    fn certifies_agrees_with_refusal_across_every_perturbation() {
        let mut cases = vec![
            certifying(vec![proven_rail(0, 4.25), proven_rail(1, 0.0)]),
            certifying(Vec::new()),
        ];
        let mut indefinite = certifying(vec![proven_rail(0, 4.25)]);
        indefinite.curvature = CurvatureEvidence::Measured { psd: false };
        cases.push(indefinite);
        let mut unmeasured = certifying(vec![proven_rail(0, 4.25)]);
        unmeasured.curvature = CurvatureEvidence::NotAvailable;
        cases.push(unmeasured);
        let mut bad_rail = certifying(vec![proven_rail(0, 4.25)]);
        if let OuterStationarityCertificate::AsymptoteRail { rails, .. } =
            &mut bad_rail.stationarity
        {
            rails[0].value_gap = f64::NAN;
        }
        cases.push(bad_rail);

        for certificate in cases {
            assert_eq!(
                certificate.certifies(),
                certificate.refusal().is_none(),
                "certifies() and refusal() must be one predicate: {}",
                certificate.summary()
            );
        }
    }
}

#[cfg(test)]
mod rail_tail_evidence_tests {
    use super::{
        CurvatureEvidence, FacePositivityRoute, OuterCriterionCertificate,
        OuterStationarityCertificate, RailCoordinate, RailTailEvidence,
    };
    use crate::rho_optimizer::asymptote_certificate::AsymptoteSide;

    fn proven_rail(index: usize, tail_constant: f64) -> RailCoordinate {
        RailCoordinate {
            index,
            side: AsymptoteSide::Upper,
            tail_constant,
            value_gap: tail_constant * (-12.0_f64).exp(),
            estimand_travel_bound: 1.0e-9,
            evidence: RailTailEvidence::AnalyticFaceProof {
                route: FacePositivityRoute::PositiveForm,
                statistic: 3.5,
                band: 1.0e-12,
            },
        }
    }

    fn certificate(rails: Vec<RailCoordinate>) -> OuterCriterionCertificate {
        OuterCriterionCertificate {
            stationarity: OuterStationarityCertificate::AsymptoteRail {
                interior_projected_grad_norm: 1.0e-9,
                bound: 1.0e-6,
                rung: gam_problem::StationarityRung {
                    label: "solver-band",
                    derived_standard: false,
                }
                .into(),
                rails,
            },
            curvature: CurvatureEvidence::Measured { psd: true },
            lambdas_railed: vec![0, 1],
            railed_facts: Vec::new(),
            newton_polish: None,
            curvature_floor: None,
            criterion_error: None,
        }
    }

    /// A PROVEN face carrying an UNIDENTIFIED coordinate must certify.
    ///
    /// `certify_rail_face` types a coordinate whose penalty releases nothing —
    /// once the rest of the face is at `λ = ∞` — as `Unidentified` and derives
    /// `c = 0` for it exactly: `V` does not depend on its smoothing parameter at
    /// all. The well-formedness guard used to demand `c > 0` of every rail,
    /// which is the MEASURED route's rule, so a face the analytic route had
    /// just proven came back un-certified with nothing recording why. Building
    /// the state needs a second face coordinate to do the pinning, which is why
    /// no single-coordinate fixture could reach it.
    #[test]
    fn a_proven_face_certifies_with_an_unidentified_coordinate_2348() {
        let certificate = certificate(vec![proven_rail(0, 4.25), proven_rail(1, 0.0)]);
        assert!(
            certificate.certifies(),
            "an analytically proven face must certify even though coordinate 1 is \
             unidentified there — c = 0 is the proof's answer, not a missing \
             measurement: {}",
            certificate.summary()
        );
    }

    /// The measured route keeps its own rule. A probed tail reads `ĉ` off a
    /// window of measured gradients, where a constant its settlement radius
    /// cannot separate from zero is the instrument reporting itself rather
    /// than a tail law — that must still refuse, or the fix above would have
    /// widened both routes at once.
    #[test]
    fn a_probed_tail_still_refuses_a_non_positive_constant_2348() {
        let probed = RailCoordinate {
            index: 0,
            side: AsymptoteSide::Upper,
            tail_constant: 0.0,
            value_gap: 0.0,
            estimand_travel_bound: 1.0e-9,
            evidence: RailTailEvidence::ProbedTail {
                gradient_band: 1.0e-6,
                extrapolation_radius: 1.0e-3,
            },
        };
        assert!(
            !certificate(vec![probed]).certifies(),
            "a probed tail with c = 0 carries no evidence of a tail law at all"
        );
    }

    /// A probed constant is evidence only once its settlement radius separates
    /// it from zero: `|c − ĉ| ≤ R` with `ĉ ≤ R` admits `c = 0`.
    #[test]
    fn a_probed_tail_inside_its_settlement_radius_is_refused_3565() {
        let evidence = RailTailEvidence::ProbedTail {
            gradient_band: 1.0e-9,
            extrapolation_radius: 1.0e-3,
        };
        assert!(!evidence.admits(1.0e-3));
        assert!(!evidence.admits(5.0e-4));
        assert!(evidence.admits(2.0e-3));
        let unbounded = RailTailEvidence::ProbedTail {
            gradient_band: 1.0e-9,
            extrapolation_radius: f64::INFINITY,
        };
        assert!(!unbounded.admits(1.0e6));
    }

    /// A DESERIALIZED certificate cannot claim a proof it does not carry: the
    /// analytic route stores the inequality it was minted under, and the guard
    /// re-checks it rather than trusting the label.
    #[test]
    fn a_claimed_face_proof_below_its_own_margin_does_not_certify_2348() {
        let mut rail = proven_rail(0, 4.25);
        rail.evidence = RailTailEvidence::AnalyticFaceProof {
            route: FacePositivityRoute::PositiveForm,
            statistic: 1.0e-14,
            band: 1.0e-12,
        };
        assert!(
            !certificate(vec![rail]).certifies(),
            "lambda_min(C) must exceed the margin the certificate claims it cleared"
        );
    }

    /// The route is visible to a reader of the certificate, not only to the
    /// guard: a rail line that does not say which standard produced its
    /// constant leaves the two routes indistinguishable in the run record.
    #[test]
    fn the_summary_names_the_route_that_established_each_rail_2348() {
        let analytic = certificate(vec![proven_rail(0, 4.25)]).summary();
        assert!(
            analytic.contains("analytic face proof"),
            "a proven rail must say so: {analytic}"
        );
    }
}

#[cfg(test)]
mod shipped_criterion_identity_tests {
    use super::FitOptions;

    /// The default deterministic criterion is prior-free in the λ→∞ tail — i.e.
    /// it is REML/LAML, not REML + a ρ-prior.
    #[test]
    fn shipped_default_criterion_is_reml_not_map_in_rho_2450() {
        let prior = FitOptions::default().rho_prior;
        assert!(
            prior.upper_tail_gradient_vanishes_everywhere(8),
            "FitOptions::default() ships {prior:?}, whose rho-gradient survives into the \
             lambda -> infinity tail on one of the first 8 coordinates. The shipped deterministic \
             criterion is then REML + a rho-prior (MAP in rho), which SPEC forbids, and no \
             lambda = infinity face exists for any rail certificate to prove. A caller that \
             wants a prior passes one explicitly; joint HMC gets a proper prior from \
             rho_prior_distribution_correction, which fills unset coordinates without touching the \
             criterion."
        );
    }
}

/// Why a fit that could have published a coefficient covariance deliberately
/// did not (gam#2718, gam#2484).
///
/// A bare `None` on [`UnifiedFitResult::covariance_conditional`] or
/// [`UnifiedFitResult::covariance_corrected`] is three states wearing one costume: *not
/// requested*, *not computed*, and *computed but not valid to publish*. A
/// consumer cannot tell them apart, so an absence that was a considered refusal
/// reads exactly like an absence nobody thought about. This enum is the third
/// state said out loud: it is `Some` only when a fit reached a point where a
/// covariance was expected, decided it could not stand behind one, and minted
/// the point estimates anyway.
///
/// It is deliberately not a log line. The behaviour this replaces was a hard
/// `Err` that took the point estimates down with the covariance; the obvious
/// alternative — publish the uncorrected covariance and warn — is worse than
/// either, because on the wire a too-narrow interval is indistinguishable from
/// a corrected one. Absence plus a typed reason is distinguishable, and the
/// consumer that reads standard errors already destructures the `Option` this
/// sits beside.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "kebab-case")]
pub enum CovarianceDeclined {
    /// Bernoulli marginal-slope, conditional latent-z calibration active, a
    /// second-stage latent measure that is not `StandardNormal`, AND no
    /// cross-row channel for it.
    ///
    /// This is no longer the whole non-StandardNormal class. gam#2484 derived
    /// the channel the correction was missing: a row's log-likelihood depends on
    /// `zeta_i` directly and through a grid every other `zeta_j` helped build,
    /// and the second dependence has a closed form, because the equal-mass bins
    /// are cut by cumulative WEIGHT and the allocation is therefore exactly
    /// constant in `zeta`. On the ordinary rigid `global-empirical` fit the
    /// correction is computed and the covariance IS published.
    ///
    /// What still declines are the shapes with no such channel, and the variant
    /// now carries which one: a `local-empirical` measure (per-row grids, only
    /// reachable by deserializing a saved model, so there is no fit-time
    /// allocation record), a score-warp / link-deviation block (the latent score
    /// enters the row a second time through a basis evaluated at it, so the
    /// rigid intercept's node derivative is not the row's), or data on which the
    /// compression is not differentiable at all (a tied `zeta` group that a bin
    /// boundary cuts, where the left and right derivatives genuinely differ).
    ///
    /// Point estimation is unaffected and IS published; only the second-stage
    /// covariance and its standard errors are withheld.
    BmsGeneratedRegressorLatentMeasureNotStandardNormal {
        /// The measure the adequacy gate selected instead, as named by
        /// `LatentMeasureKind`: `global-empirical` or `local-empirical`.
        latent_measure: String,
        /// Which channel this fit's shape or data could not supply. Not named
        /// `reason`: that is this enum's serde tag.
        #[serde(default)]
        unavailable_channel: String,
    },
    /// Survival marginal-slope, conditional latent-z calibration active, and a
    /// fit shape whose per-row `d(score_beta,i)/d(zeta_i)` channel the family
    /// cannot supply (gam#2768).
    ///
    /// Same contract as the variant above and for the same reason: the naive
    /// covariance treats the generated regressor `zeta` as known, so it omits
    /// the first-stage uncertainty and is too narrow. The two shapes that reach
    /// this are a score-warp / link-deviation block (the latent score enters the
    /// row a second time through a basis evaluated at it, so the rigid mixed
    /// derivative is not the row's) and `K > 1` (the shared-slope kernel sees
    /// only `z_sum`, so a per-coordinate sensitivity is not separable).
    ///
    /// Point estimation is unaffected and IS published.
    SurvivalMarginalSlopeGeneratedRegressorSensitivityUnavailable {
        /// The family's own account of which channel it could not produce.
        /// Not named `reason`: that is this enum's serde tag.
        unavailable_channel: String,
    },
    /// Bernoulli marginal-slope, conditional latent-z calibration active, and a
    /// gam#2924 residual repair block whose joint `(z, r)` covariance escalated
    /// to the conditional `Σ(a)` (gam#2985).
    ///
    /// Whatever the latent measure: `Σ(a)` is an M-estimate fitted on the
    /// calibrated score, so every `zeta_j` moves every row's anchor through it,
    /// and that implicit derivative is not supplied. Under the pooled joint
    /// covariance the block's channels are carried and the covariance IS
    /// corrected.
    ///
    /// Point estimation is unaffected and IS published.
    BmsGeneratedRegressorResidualRepairChannelUnavailable {
        /// Which channel is missing. Not named `reason`: that is this enum's
        /// serde tag.
        unavailable_channel: String,
    },
    /// Expectile (LAWS) fit whose inner solve published no dense covariance —
    /// the memory governor refused it and only a factorized diagonal of the
    /// Gaussian working-model `Vb` exists, or inference was not computed.
    ///
    /// The expectile's covariance is the penalized Newey–Powell sandwich
    /// `H⁻¹(c·Xᵀdiag(w²r²)X + φ̂S_λ)H⁻¹`, a correction of the FULL `Vb`; a
    /// diagonal alone cannot carry it, and publishing the working-model
    /// diagonal under the expectile's name is the under-coverage the sandwich
    /// exists to remove. Point estimation is unaffected and IS published.
    ExpectileSandwichRequiresDenseCovariance {
        /// Coefficient count whose dense covariance was not admitted.
        coefficients: usize,
    },
}

impl CovarianceDeclined {
    /// One-paragraph explanation suitable for a CLI line or an error surface.
    /// Kept beside the variant so every consumer renders the same sentence.
    pub fn explain(&self) -> String {
        match self {
            Self::BmsGeneratedRegressorLatentMeasureNotStandardNormal {
                latent_measure,
                unavailable_channel,
            } => {
                format!(
                    "no coefficient covariance was published for this bernoulli marginal-slope \
                     fit: the conditional latent-z location-scale calibration fired, its \
                     calibrated residual then failed the standard-normal adequacy gate, and the \
                     second-stage latent measure is therefore `{latent_measure}`. The \
                     Murphy-Topel generated-regressor correction needs the TOTAL derivative of \
                     the score in the latent coordinate, which for a measure built from the \
                     whole calibrated-residual vector includes a cross-row channel through the \
                     grid. gam#2484 derived that channel and the ordinary rigid global-empirical \
                     fit now publishes the corrected covariance; this fit cannot, because \
                     {unavailable_channel}. Publishing the UNCORRECTED covariance instead is not \
                     admissible: it omits the first-stage uncertainty the correction exists to \
                     add, so the intervals would be too narrow and, on the wire, \
                     indistinguishable from corrected ones. The point estimates are unaffected \
                     and are published. See gam#2484 for the channel and gam#2718 for this \
                     contract."
                )
            }
            Self::SurvivalMarginalSlopeGeneratedRegressorSensitivityUnavailable {
                unavailable_channel,
            } => {
                format!(
                    "no coefficient covariance was published for this survival marginal-slope \
                     fit: the conditional latent-z location-scale calibration fired, so the \
                     fitted score is a GENERATED regressor whose first stage was estimated from \
                     the same data, and the Murphy-Topel correction for that needs a per-row \
                     mixed derivative of the score in the latent coordinate which this fit's \
                     shape cannot supply -- {unavailable_channel}. Publishing the UNCORRECTED \
                     covariance instead is not admissible: it omits the first-stage uncertainty \
                     the correction exists to add, so the intervals would be too narrow and, on \
                     the wire, indistinguishable from corrected ones. The point estimates are \
                     unaffected and are published. See gam#2768."
                )
            }
            Self::BmsGeneratedRegressorResidualRepairChannelUnavailable {
                unavailable_channel,
            } => {
                format!(
                    "no coefficient covariance was published for this bernoulli marginal-slope \
                     fit: the conditional latent-z location-scale calibration fired, so the \
                     fitted score is a GENERATED regressor whose first stage was estimated from \
                     the same data, and the fit carries a gam#2924 residual repair block. The \
                     Murphy-Topel correction needs the total derivative of every coefficient's \
                     score in the latent coordinate, and this block's is missing one channel: \
                     {unavailable_channel}. Publishing the UNCORRECTED covariance \
                     instead is not admissible: it omits the first-stage uncertainty the \
                     correction exists to add, so the intervals would be too narrow and, on the \
                     wire, indistinguishable from corrected ones. The point estimates are \
                     unaffected and are published. See gam#2985."
                )
            }
            Self::ExpectileSandwichRequiresDenseCovariance { coefficients } => {
                format!(
                    "no coefficient covariance was published for this expectile fit: its inner \
                     solve published no dense {coefficients}-coefficient covariance (the memory \
                     governor did not admit it, or inference was not computed), and the \
                     expectile's Newey-Powell sandwich covariance is a correction of that full \
                     matrix which a factorized diagonal or a bare Hessian cannot carry. \
                     Publishing the Gaussian working-model standard errors instead is not \
                     admissible: the asymmetric weights are not inverse variances, so those \
                     intervals under-cover wherever the noise is large. The point estimates are \
                     unaffected and are published."
                )
            }
        }
    }
}

/// A check for a basin lower than the one a certified outer search reached
/// (#1561, #1464), run by [`crate::rho_optimizer::probe_face_for_a_lower_basin`].
///
/// A certified optimum is a local minimum of its criterion. The criterion is
/// read once, one inner solve, at a face of the search box: the standard REML
/// path's least-penalized face, or the face opposite a railed curvature κ̂. A
/// face value strictly below the certified optimum's is a point of the same
/// criterion lower than a local minimum, so a lower basin exists, and a second
/// certified search then runs from the face. The two certified optima are
/// compared by the multistart's keep-best rule, and the lower is published.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceProbe {
    /// The face the criterion was read at.
    pub face_rho: Vec<f64>,
    /// The criterion at the face, or why its inner solve refused it.
    pub face: FaceValue,
    /// The criterion at the first search's certified optimum.
    pub certified_criterion: f64,
    /// What the second search did.
    pub second_search: FaceSearch,
}

/// The criterion at the probed face.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FaceValue {
    Evaluated { criterion: f64 },
    /// The inner solve at the face refused; nothing is compared.
    Refused { reason: String },
}

/// The search a lower face triggers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FaceSearch {
    /// The face was not below the certified optimum, or was not evaluated.
    NotRun,
    /// The search from the face certified at `criterion`, at `rho`;
    /// `published` when it displaced the first search's optimum.
    Certified { criterion: f64, rho: Vec<f64>, published: bool },
    /// The search from the face did not certify; the first optimum stands.
    NotCertified { reason: String },
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
    /// Deviance `D₀` of the intercept-only model on the training rows, weights
    /// and family, the denominator of the reported deviance explained
    /// `1 − D/D₀`. Recorded at fit time because a saved model keeps no response.
    /// `None` when the fit has no single intercept or carries an offset (the
    /// intercept-only reference would then not be nested in the fitted model),
    /// or on a saved model that predates the field.
    #[serde(default)]
    pub null_deviance: Option<f64>,
    /// Whether this binomial fit read its prior weights as trial counts: the
    /// response carried a proportion strictly inside `(0, 1)`, so a row with
    /// weight `m` is `Binomial(m, p)/m` and a new row's weight is its own trial
    /// count. `false` for every other family and for a 0/1 response, whose
    /// weights are case weights on a Bernoulli row. Recorded at fit time because
    /// a saved model keeps no response; the observation band and generative
    /// draws read it to pick the predictive law.
    #[serde(default)]
    pub binomial_trial_counts: bool,
    #[serde(default)]
    pub survival_link_wiggle_knots: Option<Array1<f64>>,
    #[serde(default)]
    pub survival_link_wiggle_degree: Option<usize>,
    /// First-order optimality certificate from the outer smoothing-parameter
    /// optimization (#934): gradient-vs-objective FD audit at the returned
    /// optimum, Hessian-PD probe, λ-rail flags. `None` when the outer ran
    /// gradient-free or an audit probe could not evaluate.
    #[serde(default)]
    pub criterion_certificate: Option<OuterCriterionCertificate>,
    /// What the standard REML path's lower-face probe found after its first
    /// certified search (#1561): the criterion at the least-penalized face of the
    /// ρ domain, and whether a second search ran from there and which optimum was
    /// published. `None` on every other route, and on a model that predates it.
    #[serde(default)]
    pub lower_face_probe: Option<FaceProbe>,
    /// What the Tier-0 marginal-smoothing (`ρ`-uncertainty) PSIS adequacy seam
    /// concluded (#938, #2627): the Pareto-`k̂` grade that says whether the
    /// plug-in + first-order `V_ρ` correction is adequate, or the typed reason it
    /// was not formed or was refused. Computed against the live REML objective at
    /// the converged `ρ̂` (see `RemlState::rho_posterior_inference`); a route that
    /// never passes that seam carries `NotComputed(NotFormedOnThisRoute)`, and a
    /// fit run without inference `NotComputed(InferenceNotRequested)`. It persists
    /// with the fit, so a reloaded model carries the same outcome.
    pub rho_posterior: gam_problem::rho_posterior::RhoPosteriorOutcome,
    /// Escalation outcome (#938) when the Tier-0 grade read `Escalate`:
    /// the Tier-1 quadrature mixture or the Tier-2 NUTS draws (whichever tier
    /// needs fewer criterion evaluations), or an honest `Unavailable` report.
    /// `None` whenever the grade did not escalate (or was not formed). Computed at the same
    /// live-objective seam as the grade; re-derivable, not serialized.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub rho_posterior_escalation: Option<gam_problem::rho_posterior::RhoPosteriorEscalation>,
    /// Regularized inverse REML/LAML outer Hessian over `rho = log(lambda)`,
    /// aligned with [`UnifiedFitResult::lambdas`]. This is the narrow #740
    /// handoff consumed by estimated-lambda Lawley LR corrections; it is
    /// computed from the same path as smoothing-parameter uncertainty and is
    /// re-derivable, so it is not serialized.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub rho_covariance: Option<Array2<f64>>,
    /// Selected per-component log-smoothing parameters of the full-width JOINT
    /// penalty (gam#1587/#561). Families whose smoothing is carried by a joint
    /// penalty (the multinomial centered `Σ_t λ_t (M ⊗ S_t)` metric) leave their
    /// per-block penalty lists — and hence [`UnifiedFitResult::lambdas`] — empty,
    /// so the only place the converged `ρ_t` survives is here. `None` for every
    /// per-block-only family. Re-derivable from a refit, so not serialized; it is
    /// consumed by the multinomial reporting path to reconstruct per-(class,term)
    /// λ, per-class EDF, and the influence matrix `F = I − H⁻¹ S_λ`.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub joint_log_lambdas: Option<Array1<f64>>,
    /// Whether the fit optimized the Firth/Jeffreys-adjusted likelihood.
    /// Persisted (serialized) so saved-model posterior sampling reconstructs
    /// the SAME target the fit optimized — dropping the Jeffreys term Φ(β)
    /// from the sampled log-posterior silently samples a different model
    /// (#2245 finding 16). `false` for fits that never engaged Firth.
    #[serde(default)]
    pub firth_bias_reduction: bool,
    /// The typed evidence on which the custom-family arming lifecycle armed this
    /// fit's Jeffreys/Firth prior (#979). `None` when the lifecycle did not arm
    /// it, either because the unarmed objective certified or because the fit
    /// did not run through the lifecycle. Serialized, so a saved model says
    /// which objective its coefficients are the mode of, and why.
    #[serde(default)]
    pub jeffreys_arming_evidence: Option<gam_problem::jeffreys_arming::JeffreysArmingEvidence>,
    /// Set when this fit certified an unarmed, unconstrained mode whose
    /// penalized information is singular on the directions no smoothing
    /// parameter reaches (#3164): its Laplace posterior is improper, which is
    /// the evidence the custom-family arming lifecycle arms on. The terminal
    /// posterior assembly measures it from the same precision it publishes.
    /// Re-derivable from that precision and consumed only by the lifecycle, so
    /// not serialized.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub improper_penalty_null_posterior:
        Option<gam_problem::jeffreys_arming::JeffreysArmingEvidence>,
    /// Set when this fit could have published a coefficient covariance and
    /// deliberately did not (gam#2718). `None` is the ordinary case and carries
    /// NO claim either way: a covariance may be present, or absent because it
    /// was never requested. `Some` is a positive statement that one was
    /// withheld, with the reason attached — see [`CovarianceDeclined`].
    ///
    /// Serialized, because the reason has to survive to a consumer reading a
    /// saved model's standard errors; that consumer is exactly the one who
    /// would otherwise misread the absence.
    ///
    /// # This channel is ADVISORY, and that is only safe because absence is honest
    ///
    /// Nothing in the type system forces a consumer to read this field.
    /// `covariance_conditional` is an `Option` whose `None` was already a
    /// common, benign value long before this existed (the exact-interpolation
    /// Gaussian boundary, saved models reconstructed without inference, any fit
    /// that declined it), so a consumer looking only there still cannot tell
    /// "never computed" from "computed and withheld". Making that distinction
    /// structural would mean replacing the `Option` with a three-state enum
    /// across ~185 references in 52 files.
    ///
    /// That was not done, and the reason it is safe not to have done it is a
    /// MEASURED property rather than an assumption: **no consumer substitutes a
    /// value for an absent covariance.** Every production read propagates the
    /// absence — `.as_ref().map(..)`, `.filter(..)?`, `.and_then(..)`, `.clone()`
    /// into another `Option`, or an outright `Err`. None defaults to a zero
    /// matrix, an identity, or a zero standard error. Checked over
    /// `covariance_conditional` (185 occurrences) and the paired
    /// `beta_covariance` / `beta_standard_errors` /
    /// `beta_covariance_corrected` / `beta_standard_errors_corrected` (114
    /// production reads).
    ///
    /// The single fallback in the tree is
    /// [`UnifiedFitResult::beta_covariance_corrected`], which returns `Vb` for
    /// `Vp` when the fit has no smoothing coordinate, per-block or joint. It is
    /// guarded, documented, and exact — with no smoothing coordinates the
    /// correction `J Var(rho) Jᵀ` is identically zero; a joint-penalty family
    /// with an empty per-block `lambdas` still has joint coordinates (#2898) —
    /// and it `.flatten()`s to `None` when the conditional covariance is also
    /// absent, which is the state a declined fit is in.
    ///
    /// # The trigger, as a checkable condition
    ///
    /// Run:
    ///
    /// ```text
    /// git grep -n -E '\.(covariance_conditional|beta_covariance|beta_standard_errors)' -- crates/ src/
    /// ```
    ///
    /// and count the sites that meet an absence with `unwrap_or`,
    /// `unwrap_or_default`, `unwrap_or_else`, `map_or`, or a `None =>` arm
    /// yielding a matrix or a zero SE, rather than propagating.
    ///
    /// **Today that count is 0**, out of 25 machine-flagged candidates, out of
    /// 185 references in 52 files (plus 114 production reads of the paired
    /// `beta_*` fields, also 0). **If it is ever greater than 0, this field is
    /// insufficient and the `Option` must become an enum.** One command, two
    /// integers; no re-derivation of the judgement above.
    ///
    /// # What this does NOT establish: the quantity is still RECONSTRUCTIBLE
    ///
    /// The sweep above is over consumers of the fields that get cleared. It
    /// says nothing about what else on the artifact PRODUCES the same quantity,
    /// and something does:
    ///
    /// * [`UnifiedFitResult::penalized_hessian`] returns `H`, and it is
    ///   **non-`Option`** on both `FitInference` and `FitGeometry` — neither of
    ///   which this seam clears — so it survives every declined fit and is
    ///   persisted onto the saved model;
    /// * `FitInference::dispersion` supplies `phi`.
    ///
    /// `Vb_naive = phi * H^-1` is therefore one Cholesky away, and
    /// `beta_covariance_frequentist`, `coefficient_influence` and
    /// `weighted_gram` are further optional producers this seam also leaves
    /// alone. **A consumer can obtain a coefficient covariance without ever
    /// reading this field.**
    ///
    /// That is accepted rather than fixed, for a stated reason: what is
    /// reconstructible is the NAIVE covariance — exactly the object the BMS
    /// seam refuses to publish, because it omits the first-stage
    /// generated-regressor uncertainty and is therefore too narrow. Withholding
    /// it means not SHIPPING it under the name `beta_covariance`, where it would
    /// be indistinguishable from a corrected one. It does not mean, and cannot
    /// mean, destroying the curvature every other consumer of the fit needs:
    /// `penalized_hessian` is what EDF accounting, posterior whitening and
    /// prediction all read, and it is not `Option`, so it cannot be withheld
    /// without dropping `inference` and `geometry` wholesale.
    ///
    /// The route that removes this residual entirely is implementing the
    /// correction (`G_measure`, gam#2484), after which nothing is withheld.
    ///
    /// # The producer-set trigger, also checkable
    ///
    /// One persistence route drops this field: the compact saved-fit
    /// constructors in `gam-cli` (`model_build.rs`) take `beta_covariance` and
    /// assign `inf.penalized_hessian` from the geometry, but have no parameter
    /// that could carry a declination — so a fit persisted through them would
    /// ship curvature with the warning stripped off. No parameter was threaded,
    /// because nothing reaches them: this field has exactly ONE producer, and
    /// that producer persists whole-`UnifiedFitResult` through
    /// `assemble_bernoulli_marginal_slope_payload` instead.
    ///
    /// That defence is only as good as the producer count, so count it. Run:
    ///
    /// ```text
    /// git grep -n 'covariance_declined' -- crates/ src/ | grep -E 'covariance_declined\s*='
    /// ```
    ///
    /// **Outside tests, today that returns** `bms/block_specs.rs` (the BMS
    /// Murphy-Topel seam), `survival/marginal_slope/generated_regressor.rs`,
    /// and `fit_orchestration/entry.rs` (the expectile sandwich on a fit whose
    /// dense covariance was not admitted, persisted whole-fit through
    /// `assemble_standard_payload`). **If a second producer ever appears, check whether it
    /// persists through a compact constructor; if it does, the parameter must be
    /// threaded and `tests/bms_covariance_declined_2718.rs` extended to cover
    /// that route.** The round-trip test there pins the wire, not the routing,
    /// so it will not catch a new producer on its own.
    #[serde(default)]
    pub covariance_declined: Option<CovarianceDeclined>,
    /// The certified outer point of the published search, in the outer
    /// optimizer's own coordinates, when the route that fitted the model records
    /// one: `gamfit.fit(..., warm_start_from=model)` offers it to a new fit
    /// (gam#3002, `OuterProblem::with_warm_start`), whose searches accept it only
    /// where it is certified for their own criterion. `None` on a route that
    /// records none and on a model saved before it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outer_warm_start: Option<OuterWarmStartRecord>,
    /// Which rule selected the coefficient mode this fit reports (#2366,
    /// #2661). A payload written before the record existed carries
    /// [`CoefficientModeSelection::NotRecorded`], which claims nothing.
    #[serde(default)]
    pub coefficient_mode_selection: CoefficientModeSelection,
    /// The variance-component score test of every random-effect term, computed
    /// once on the training fit's own IRLS row state
    /// (`gam_terms::inference::random_effect_test`). The summary's
    /// random-effect rows read their p-value (or its typed absence) from here,
    /// so the CLI, Rust and persisted-model surfaces report the same number.
    /// Empty on a model with no such term and on a payload written before the
    /// test existed; a summary treats a term missing from it as not recorded.
    /// Ridged linear terms are recorded in `linear_term_tests`, never here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub random_effect_tests:
        Vec<gam_terms::inference::random_effect_test::RandomEffectTestRecord>,
    /// The same variance-component score test for every linear term carrying
    /// the null-recovery ridge (`TermCollectionDesign::ridged_linear_ranges`),
    /// computed on the same row state (#3573). The summary's ridged linear
    /// rows read their statistic and p-value from here instead of from the
    /// Wald ratio of the shrunk coefficient. Empty when no linear term is
    /// ridged; a summary treats a ridged term missing from it as not recorded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub linear_term_tests:
        Vec<gam_terms::inference::random_effect_test::RandomEffectTestRecord>,
}

/// A certified outer point (gam#3002): `theta`, the outer coordinates in the order
/// the fit's outer problem holds them (ρ, then the log length scales, then the
/// auxiliary coordinates), `beta`, the inner coefficient mode there flattened
/// across blocks, `value`, the criterion the search certified there, and the
/// fingerprint of the inputs the point is certified for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OuterWarmStartRecord {
    /// v25 wrote the ρ-only point as `rho`.
    #[serde(alias = "rho")]
    pub theta: Vec<f64>,
    pub beta: Vec<f64>,
    /// `value` and `input_fingerprint` are `None` in a v25 record, which predates
    /// them; such a point can only join a search, never resume as a certificate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_fingerprint: Option<String>,
}

/// Which rule selected a fit's coefficient mode (#2366, #2661).
///
/// A custom family whose inner coefficient objective is nonconvex can hold
/// several modes at one ρ, and the profiled criterion `V(ρ)` is a function of
/// ρ only once a rule names which mode the fit reports. The #2661 anchored
/// continuation from maximal smoothing is that rule, and a family-owned
/// objective homotopy is the stronger one where a family declares it. When
/// the continuation declines, the fit proceeds from the caller's seed and its
/// mode is a functional of that seed. This records which of these happened, so
/// a consumer reads it from the fit and can refuse a seed-selected mode
/// ([`Self::require_rule_selected`]) instead of finding it in a log.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "kebab-case")]
pub enum CoefficientModeSelection {
    /// No rule was recorded: a payload written before this record existed, or
    /// a route that selects its mode elsewhere and records nothing here.
    #[default]
    NotRecorded,
    /// The inner objective has one mode: the family's Hessian does not depend
    /// on β, or the family certifies global convexity.
    UniqueMode,
    /// The endpoint of the family's certified coefficient-objective homotopy.
    ObjectiveHomotopy { steps: usize },
    /// The endpoint of the #2661 anchored continuation, certified.
    AnchoredContinuation {
        steps: usize,
        endpoint_discrepancy: f64,
    },
    /// No rule selected the mode, so it is the one the caller's seed reached.
    /// `reason` says why no rule applied: the continuation's refusal, or the
    /// route having no smoothing parameter to anchor a continuation at.
    SeedSelected { reason: String },
}

impl CoefficientModeSelection {
    /// Refuse a mode no rule selected: a seed-selected mode, naming the
    /// continuation's refusal, and an unrecorded one.
    pub fn require_rule_selected(&self, context: &str) -> Result<(), String> {
        match self {
            Self::UniqueMode
            | Self::ObjectiveHomotopy { .. }
            | Self::AnchoredContinuation { .. } => Ok(()),
            Self::SeedSelected { reason } => Err(format!(
                "{context}: the coefficient mode is the one the caller's seed reached, because the \
                 selection rule did not apply: {reason}"
            )),
            Self::NotRecorded => Err(format!(
                "{context}: no rule selecting the coefficient mode was recorded"
            )),
        }
    }
}

#[cfg(test)]
mod coefficient_mode_selection_wire_2661_tests {
    use super::{CoefficientModeSelection, FitArtifacts};

    /// Every variant survives the payload wire, and an artifacts record saved
    /// before the field existed reads as `NotRecorded`, which claims nothing.
    #[test]
    fn the_mode_selection_record_round_trips_and_defaults_to_not_recorded() {
        for selection in [
            CoefficientModeSelection::NotRecorded,
            CoefficientModeSelection::UniqueMode,
            CoefficientModeSelection::ObjectiveHomotopy { steps: 3 },
            CoefficientModeSelection::AnchoredContinuation {
                steps: 4,
                endpoint_discrepancy: 2.5e-7,
            },
            CoefficientModeSelection::SeedSelected {
                reason: "the continuation declined".to_string(),
            },
        ] {
            let wire = serde_json::to_string(&selection).expect("serialize");
            let read: CoefficientModeSelection = serde_json::from_str(&wire).expect("deserialize");
            assert_eq!(read, selection, "{wire}");
        }
        let mut saved = serde_json::to_value(FitArtifacts::default()).expect("serialize artifacts");
        saved
            .as_object_mut()
            .expect("artifacts serialize as an object")
            .remove("coefficient_mode_selection")
            .expect("the record is on the wire");
        let read: FitArtifacts = serde_json::from_value(saved).expect("an older record reads");
        assert_eq!(
            read.coefficient_mode_selection,
            CoefficientModeSelection::NotRecorded
        );
    }
}

impl std::fmt::Debug for FitArtifacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FitArtifacts")
            .field("pirls", &self.pirls.as_ref().map(|_| "..."))
            .field("null_space_logdet", &self.null_space_logdet)
            .field("null_space_dim", &self.null_space_dim)
            .field("null_deviance", &self.null_deviance)
            .field("binomial_trial_counts", &self.binomial_trial_counts)
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
            .field("lower_face_probe", &self.lower_face_probe)
            .field("rho_posterior", &self.rho_posterior)
            .field("rho_posterior_escalation", &self.rho_posterior_escalation)
            .field(
                "rho_covariance",
                &self.rho_covariance.as_ref().map(|m| m.dim()),
            )
            .field(
                "joint_log_lambdas",
                &self.joint_log_lambdas.as_ref().map(|v| v.len()),
            )
            .field("covariance_declined", &self.covariance_declined)
            .field(
                "coefficient_mode_selection",
                &self.coefficient_mode_selection,
            )
            .field("jeffreys_arming_evidence", &self.jeffreys_arming_evidence)
            .field(
                "improper_penalty_null_posterior",
                &self.improper_penalty_null_posterior,
            )
            .field(
                "outer_warm_start",
                &self
                    .outer_warm_start
                    .as_ref()
                    .map(|seed| (seed.theta.len(), seed.beta.len())),
            )
            .field("random_effect_tests", &self.random_effect_tests)
            .field("linear_term_tests", &self.linear_term_tests)
            .finish()
    }
}

/// Serde default for `max_node_criterion_rise` on models written before the
/// field existed: the rise was not measured, which is not the same as zero.
fn unmeasured_criterion_rise() -> f64 {
    f64::NAN
}

/// Which posterior covariance definition an uncertainty surface was built from.
///
/// This is the library's ONE covariance-definition vocabulary, and it lives
/// here — beside [`SmoothingCorrectionMethod`], which names how the correction
/// was produced — rather than in a predict-layer crate, so that every consumer
/// of a fit can say which of the two matrices it read. A family that owns its
/// own predict surface (the multinomial) sits BELOW `gam-predict` in the crate
/// graph; when this enum lived up there, that family had no way to express the
/// distinction and quietly published conditional-only bands while every other
/// family defaulted to [`Self::SmoothingCorrected`] (gam#2612).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum InferenceCovarianceMode {
    /// Use conditional posterior covariance only:
    ///   Var(beta | lambda_hat) ~= H_{rho_hat}^{-1}.
    Conditional,
    /// Require first-order smoothing-corrected covariance:
    ///   Var(beta) ~= H_{rho_hat}^{-1} + J Var(rho_hat) J^T.
    /// Absence is an error; this mode never substitutes conditional covariance.
    SmoothingCorrected,
}

impl InferenceCovarianceMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conditional => "conditional",
            Self::SmoothingCorrected => "smoothing-corrected",
        }
    }
}

impl std::str::FromStr for InferenceCovarianceMode {
    type Err = String;

    /// The one public vocabulary for the covariance-mode knob across every
    /// surface (`gam predict --covariance-mode`, Python `covariance_mode`).
    /// The CLI historically said "corrected" and the Python bindings said
    /// "smoothing" for the SAME mode; both spellings (plus the enum's own
    /// canonical "smoothing-corrected") are accepted here so the vocabulary
    /// cannot drift per frontend again. Unknown strings are a hard error so a
    /// typo never silently degrades to a default covariance.
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "conditional" => Ok(Self::Conditional),
            "corrected" | "smoothing" | "smoothing-corrected" => Ok(Self::SmoothingCorrected),
            other => Err(format!(
                "covariance mode must be one of \"conditional\", \"corrected\", or \"smoothing\"; got \"{other}\""
            )),
        }
    }
}

/// Serialized provenance of a retained smoothing-uncertainty correction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SmoothingCorrectionMethod {
    /// First-order IFT propagation on the explicitly identified outer-Hessian
    /// subspace, with no perturbation of either inner or outer covariance.
    FirstOrderIdentifiedSubspace {
        active_rank: usize,
        rho_dimension: usize,
    },
    /// Sigma-point integration of the smoothing-parameter posterior over a
    /// finite node set. No fit mints this any more: the correction is the
    /// analytic first-order one for every smoothing dimension. The variant is
    /// kept only so models saved by earlier releases still deserialize, and it
    /// is never reported as exact WPS.
    SigmaPointCubature {
        rank: usize,
        n_points: usize,
        /// Worst criterion rise `V(node) − V(ρ̂)` over the evaluation nodes the
        /// correction was built from, against the `1/2` a one-sigma node is
        /// asserted to sit at.
        ///
        /// This is the honest characterisation of the approximation, and it
        /// replaces the perturbation ledger: what makes a sigma-point estimate
        /// trustworthy is not that the rho-Hessian was left unperturbed, it is
        /// that the nodes sit where the posterior actually has mass. A value of
        /// `3309` — measured, before #2728 — says a node with posterior weight
        /// `e^-3309` was carrying weight 1/2.
        ///
        /// `NaN` on a fit deserialized from before this field existed.
        #[serde(default = "unmeasured_criterion_rise")]
        max_node_criterion_rise: f64,
    },
}

/// Why a fit that selected smoothing parameters retains no smoothing-uncertainty correction.
///
/// Minted where the correction was attempted, at fit time, so a consumer asked for the corrected
/// covariance names the structural cause instead of reporting a bare absence (#2677).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SmoothingCorrectionAbsence {
    /// The outer search declared no analytic ρ-Hessian, so there is no `V_ρ` to propagate.
    OuterHessianUndeclared { reason: OuterHessianAbsence },
    /// The interior ρ-Hessian was formed but refused inversion on its identified subspace.
    InteriorRhoHessianRefused { refusal: String },
    /// The optimum is certified on an infinite-smoothing rail, where ρ has no finite variance.
    ///
    /// No longer produced: the correction excludes railed coordinates exactly as the outer
    /// certificate does (at a rail `∂β̂/∂ρ_k → 0`, so they contribute nothing). Kept so fits
    /// saved before that change still load.
    RailCertified { detail: String },
    /// The corrected covariance could not be truncated to the constrained feasible set.
    ConstrainedTruncationRefused { detail: String },
    /// A fit whose inference stayed factorized (the dense covariance bundle was not reserved)
    /// could not reserve the workspace the first-order correction is assembled in (#3283).
    CorrectionWorkspaceRefused { detail: String },
    /// The certified outer optimum moved family hyperparameters `ψ`, but the terminal evaluation
    /// that owns the coefficient mode assembled no ψ scores `∂_ψ∇_β F`, so the mode response
    /// `∂β̂/∂ψ` that carries their uncertainty into the coefficients is unknown (#2677).
    FamilyHyperScoresUnrecorded { psi_dimension: usize },
    /// A constrained fit's outer coordinates include `psi_dimension` design (ψ) axes. The
    /// θ-mixture moves each node's precision by the penalties' exact `λ` dependence, and a ψ
    /// axis's is the family's own second design derivative, which no family publishes (#3229).
    ConstrainedMixtureOverDesignAxes { psi_dimension: usize },
    /// A constrained fit whose posterior is not a truncation of an ambient Gaussian (a certified
    /// boundary-mode law) has no node laws for the θ-mixture to be built from (#3229).
    ConstrainedMixtureWithoutAmbientMoments,
}

/// Why a custom-family outer search declares no analytic ρ-Hessian.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OuterHessianAbsence {
    /// The fit's options turned the outer Hessian off.
    DisabledByOptions,
    /// The family's outer objective prices no `log|H|` term.
    ObjectiveWithoutLogdetH,
    /// The family's derivative policy exposes first-order calculus only.
    FirstOrderCapability,
    /// The armed Jeffreys term needs a third information derivative the family does not provide.
    ArmedJeffreysWithoutThirdInformationDerivative,
    /// The smoothing selector published no analytic ρ-Hessian at its terminal point.
    NotPublished,
}

impl std::fmt::Display for OuterHessianAbsence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::DisabledByOptions => "the fit's options turned the outer Hessian off",
            Self::ObjectiveWithoutLogdetH => "the family's outer objective prices no log|H| term",
            Self::FirstOrderCapability => "the family exposes first-order outer calculus only",
            Self::ArmedJeffreysWithoutThirdInformationDerivative => {
                "the armed Jeffreys term needs a third information derivative the family does not provide"
            }
            Self::NotPublished => {
                "the smoothing selector published no analytic rho-Hessian at its terminal point"
            }
        })
    }
}

impl std::fmt::Display for SmoothingCorrectionAbsence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OuterHessianUndeclared { reason } => {
                write!(f, "no analytic outer rho-Hessian was declared: {reason}")
            }
            Self::InteriorRhoHessianRefused { refusal } => {
                write!(f, "the interior rho-Hessian refused inversion: {refusal}")
            }
            Self::RailCertified { detail } => write!(
                f,
                "the optimum is certified on an infinite-smoothing rail with no finite rho-variance: {detail}"
            ),
            Self::ConstrainedTruncationRefused { detail } => write!(
                f,
                "the corrected covariance could not be truncated to the feasible set: {detail}"
            ),
            Self::CorrectionWorkspaceRefused { detail } => write!(
                f,
                "the factorized fit could not reserve the smoothing correction's workspace: {detail}"
            ),
            Self::FamilyHyperScoresUnrecorded { psi_dimension } => write!(
                f,
                "the evaluation that owns the coefficient mode recorded no scores for its \
                 {psi_dimension} family hyperparameter(s), so their mode response is unknown"
            ),
            Self::ConstrainedMixtureOverDesignAxes { psi_dimension } => write!(
                f,
                "the constrained smoothing-corrected mixture moves the precision along its \
                 smoothing axes only, and this fit has {psi_dimension} design axis/axes"
            ),
            Self::ConstrainedMixtureWithoutAmbientMoments => write!(
                f,
                "the constrained posterior is a boundary-mode law, not a truncation of an \
                 ambient Gaussian, so it has no node laws to mix"
            ),
        }
    }
}

/// The smoothing-parameter correction of a fit whose inference stayed factorized (#3283).
///
/// When the resource governor refuses the dense covariance bundle, the standard optimizer
/// publishes standard errors without a covariance (#2960). The correction `C = J·V_ρ·Jᵀ` is then
/// kept as the square-root factor it is assembled from, `C = B·Bᵀ` with `B` of shape `p × r`
/// (`r` the identified rank of `V_ρ`), so no `p × p` matrix is formed. A consumer applies
/// `Vp·x = Vb·x + B·(Bᵀ·x)` through the Hessian factor. A constrained fit's corrected law is
/// instead the θ-mixture of its truncated node laws its geometry carries (#3229): there
/// `standard_errors` are the mixture's, published through one factored node precision per node,
/// and `B` is only the first-order factor the corrected-EDF reads, never an ambient inflation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactorizedSmoothingCorrection {
    /// `B` in the frame of the published coefficients, `C = B·Bᵀ`.
    pub factor: Array2<f64>,
    /// `sqrt(diag(Vp))` of the published coefficients, truncated to the feasible set on a
    /// constrained fit, solved beside [`FitInference::factorized_standard_errors`].
    pub standard_errors: Array1<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "FitInferenceWire")]
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
    /// Why each block's trace is confined to `[0, rank_k]`, or that it is not
    /// certified (#2901), aligned 1:1 with `penalty_block_trace`. An
    /// `Uncertified` block publishes its raw trace and an unclamped
    /// `edf_by_block` entry, and `edf_total` is unclamped whenever any block is not
    /// certified. Empty for fits saved before this field existed and for paths
    /// that record no rank bound.
    #[serde(default)]
    pub edf_rank_bound: Vec<crate::estimate::EdfRankBound>,
    pub edf_total: f64,
    pub smoothing_correction: Option<Array2<f64>>,
    /// Method that produced `smoothing_correction`, or the factor in
    /// `smoothing_correction_factorized`. Required whenever either is present;
    /// `None` means no correction was retained.
    pub smoothing_correction_method: Option<SmoothingCorrectionMethod>,
    /// The exact first-order IFT smoothing-parameter-uncertainty correction
    /// read by the #946 WPS-corrected-EDF/AIC channel (see
    /// `model_comparison_from_unified`'s `method_certified_exact` gate). A fresh
    /// fit sets it equal to `smoothing_correction`; it differs only on models
    /// saved by earlier releases whose primary pair held a sigma-point
    /// cubature. `Some` exactly when `smoothing_correction_method_first_order`
    /// is `Some(FirstOrderIdentifiedSubspace{..})`; `None` when the first-order
    /// geometry itself was unavailable.
    #[serde(default)]
    pub smoothing_correction_first_order: Option<Array2<f64>>,
    /// Provenance for `smoothing_correction_first_order`. Always either `None`
    /// or `Some(FirstOrderIdentifiedSubspace{..})` — this field never holds
    /// `SigmaPointCubature`.
    #[serde(default)]
    pub smoothing_correction_method_first_order: Option<SmoothingCorrectionMethod>,
    /// The typed reason a fit that selected smoothing parameters publishes no
    /// `beta_covariance_corrected`. `None` whenever the corrected covariance is published, in
    /// full or as the factorized branch's `smoothing_correction_factorized` (#3283), or the fit
    /// has no smoothing coordinate.
    #[serde(default)]
    pub smoothing_correction_absence: Option<SmoothingCorrectionAbsence>,
    /// Penalised Hessian `H = X'W_HX + S(λ)` with NO dispersion scaling.
    /// When [`UnifiedFitResult::geometry`] is present, this matrix shares its
    /// exact active coefficient frame and therefore has dimension
    /// `geometry.coefficient_gauge.reduced_total()`. Without saved geometry it
    /// is in the saved/raw coefficient frame.
    /// Stored as `UnscaledPrecision` so callers that need the φ-scaled
    /// covariance `Vb` know they must pair this with [`Self::dispersion`].
    /// `#[serde(transparent)]` on the newtype keeps the on-disk encoding
    /// identical to the pre-newtype `Array2<f64>` storage.
    pub penalized_hessian: gam_problem::dispersion_cov::UnscaledPrecision,
    pub reparam_qs: Option<Array2<f64>>,
    /// Dispersion/scale used to scale all coefficient covariance matrices.
    /// [`Dispersion`] is a validated newtype with no meaningful default (its
    /// source tag and φ are always established by the fit), so this field is
    /// required on the wire — no `#[serde(default)]`, which would both demand a
    /// nonexistent `Default` impl and silently fabricate an unvalidated scale.
    pub dispersion: Dispersion,
    /// Marginal standard errors published WITHOUT a coefficient covariance.
    ///
    /// The standard optimizer's factorized branch solves `diag(M·Σ·Mᵀ)`
    /// through the Hessian factor when the resource governor refuses the dense
    /// bundle, and publishes only that diagonal (#2960). Every other fit
    /// publishes its covariance in [`UnifiedFitResult::covariance_conditional`]
    /// and [`UnifiedFitResult::covariance_corrected`], the only store of each,
    /// and [`UnifiedFitResult::beta_standard_errors`] derives the standard
    /// errors from it (#2955). [`UnifiedFitResult::try_from_parts`] refuses a
    /// fit that carries both, so a published diagonal has no second copy to go
    /// stale.
    #[serde(default)]
    pub factorized_standard_errors: Option<Array1<f64>>,
    /// The factorized branch's smoothing correction and corrected standard
    /// errors (#3283). `Some` only beside [`Self::factorized_standard_errors`]:
    /// a fit that publishes a covariance carries its correction in
    /// [`Self::smoothing_correction`] and its corrected covariance in
    /// [`UnifiedFitResult::covariance_corrected`].
    #[serde(default)]
    pub smoothing_correction_factorized: Option<FactorizedSmoothingCorrection>,
    /// Frequentist covariance Ve = H⁻¹ X'WX H⁻¹ * φ̂.
    #[serde(default)]
    pub beta_covariance_frequentist: Option<Array2<f64>>,
    /// Coefficient-space influence matrix F = H⁻¹ X'WX. Its trace is the total EDF.
    #[serde(default)]
    pub coefficient_influence: Option<Array2<f64>>,
    /// Weighted Gram `X'WX = H − S(λ)`, in the penalized Hessian's frame (the
    /// coefficient gauge's active frame when geometry is present; read it in
    /// the saved frame through [`UnifiedFitResult::saved_frame_weighted_gram`],
    /// gam#3346) —
    /// symmetric PSD by construction on the GLM lanes. Stored directly (issue #1027) so the
    /// Wood–Pya–Säfken corrected-EDF correction `tr(X'WX·Σ_ρ)` pairs the true
    /// PSD Gram with `Σ_ρ`, rather than reconstructing it as `H·F` from a
    /// Hessian surface that need not satisfy `H·F = X'WX` (which made the
    /// correction indefinite and the corrected EDF drop below the conditional).
    ///
    /// A blockwise custom-family fit publishes its likelihood curvature
    /// `H − S(λ)` here, the observed information of an arbitrary likelihood,
    /// which is PSD only at a likelihood maximum. Such a fit has no scalar
    /// covariance scale, so the corrected-EDF correction never reads it; the
    /// smooth score test refuses a term whose score covariance it leaves
    /// indefinite.
    #[serde(default)]
    pub weighted_gram: Option<Array2<f64>>,
    /// The penalized Hessian's identified coefficient subspace at the fitted
    /// smoothing parameters (#2901 V22): its rank, the directions no observation
    /// and no penalty pins down, and whether that rank was certified constant
    /// over the outer certificate's Newton step. `None` where the criterion did
    /// not price `½log|H|₊` on a band-identified subspace: a Firth fit's
    /// structural rank, a sparse Hessian's strict factorization, or a route that
    /// does not form this Hessian.
    #[serde(default)]
    pub identified_subspace: Option<IdentifiedCoefficientSubspace>,
    /// The working residual `‖z − Xβ̂‖²_W` over the `n⁺` rows that carry
    /// curvature, in the metric of [`Self::weighted_gram`]. The smooth score
    /// test's estimated scale reads the full model's unpenalized residual off
    /// it (gam#3832). `None` where the fit publishes no working model in that
    /// metric.
    #[serde(default)]
    pub working_residual: Option<WorkingResidual>,
}

/// The wire form [`FitInference`] deserializes through.
///
/// A payload written before the one-store schema (#2955) also carries
/// `beta_covariance`, `beta_standard_errors`, `beta_covariance_corrected` and
/// `beta_standard_errors_corrected`. The covariance copies were checked bit for
/// bit against the top-level stores when the model was saved, so they are
/// dropped. The standard errors of a fit that published no covariance, the
/// factorized branch's diagonal, lived only in `beta_standard_errors`, so they
/// are carried into [`FitInference::factorized_standard_errors`].
#[derive(Deserialize)]
struct FitInferenceWire {
    edf_by_block: Vec<f64>,
    #[serde(default)]
    penalty_block_trace: Vec<f64>,
    #[serde(default)]
    edf_rank_bound: Vec<crate::estimate::EdfRankBound>,
    edf_total: f64,
    smoothing_correction: Option<Array2<f64>>,
    smoothing_correction_method: Option<SmoothingCorrectionMethod>,
    #[serde(default)]
    smoothing_correction_first_order: Option<Array2<f64>>,
    #[serde(default)]
    smoothing_correction_method_first_order: Option<SmoothingCorrectionMethod>,
    #[serde(default)]
    smoothing_correction_absence: Option<SmoothingCorrectionAbsence>,
    penalized_hessian: gam_problem::dispersion_cov::UnscaledPrecision,
    reparam_qs: Option<Array2<f64>>,
    dispersion: Dispersion,
    #[serde(default)]
    factorized_standard_errors: Option<Array1<f64>>,
    #[serde(default)]
    smoothing_correction_factorized: Option<FactorizedSmoothingCorrection>,
    #[serde(default)]
    beta_covariance: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    beta_standard_errors: Option<Array1<f64>>,
    #[serde(default)]
    beta_covariance_frequentist: Option<Array2<f64>>,
    #[serde(default)]
    coefficient_influence: Option<Array2<f64>>,
    #[serde(default)]
    weighted_gram: Option<Array2<f64>>,
    #[serde(default)]
    identified_subspace: Option<IdentifiedCoefficientSubspace>,
    #[serde(default)]
    working_residual: Option<WorkingResidual>,
}

impl From<FitInferenceWire> for FitInference {
    fn from(wire: FitInferenceWire) -> Self {
        let factorized_standard_errors = match (wire.factorized_standard_errors, wire.beta_covariance)
        {
            (Some(standard_errors), _) => Some(standard_errors),
            (None, None) => wire.beta_standard_errors,
            (None, Some(_)) => None,
        };
        Self {
            edf_by_block: wire.edf_by_block,
            penalty_block_trace: wire.penalty_block_trace,
            edf_rank_bound: wire.edf_rank_bound,
            edf_total: wire.edf_total,
            smoothing_correction: wire.smoothing_correction,
            smoothing_correction_method: wire.smoothing_correction_method,
            smoothing_correction_first_order: wire.smoothing_correction_first_order,
            smoothing_correction_method_first_order: wire.smoothing_correction_method_first_order,
            smoothing_correction_absence: wire.smoothing_correction_absence,
            penalized_hessian: wire.penalized_hessian,
            reparam_qs: wire.reparam_qs,
            dispersion: wire.dispersion,
            factorized_standard_errors,
            smoothing_correction_factorized: wire.smoothing_correction_factorized,
            beta_covariance_frequentist: wire.beta_covariance_frequentist,
            coefficient_influence: wire.coefficient_influence,
            weighted_gram: wire.weighted_gram,
            identified_subspace: wire.identified_subspace,
            working_residual: wire.working_residual,
        }
    }
}

/// The coefficient directions the REML criterion scored `½log|H|₊` over at the
/// fitted smoothing parameters, and the ones it dropped (#2901 V22).
///
/// `H = XᵀWX + S_λ` is priced over its eigenvalues above the rounding band
/// `p·ε·‖H‖₂`, never fewer than `rank(S_λ)`. The dropped directions lie in
/// `null(X) ∩ null(S_λ)` to that resolution: no observation and no penalty pins
/// the coefficients along them, and the fit reports the minimum-norm solution
/// there.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdentifiedCoefficientSubspace {
    /// The number of identified coefficient directions.
    pub rank: usize,
    /// One column per unidentified direction, `p × (p − rank)`, in the same
    /// coefficient frame as [`FitInference::penalized_hessian`]; empty at full
    /// rank. The columns are orthonormal in the solver's internal frame and are
    /// carried to the original frame like β.
    pub unidentified_basis: Array2<f64>,
    /// Whether `rank` was certified constant over the outer certificate's step.
    pub rank_constancy: IdentifiedRankConstancy,
}

/// Whether the identified rank was certified constant over the outer
/// certificate's own Newton step (#2901 V22). A fit whose rank can change inside
/// that step is refused, so a published fit carries one of these two.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IdentifiedRankConstancy {
    /// Certified over the step whose largest coordinate is `step_radius`.
    Certified {
        step_radius: f64,
        /// The smallest identified eigenvalue `σ_r`.
        smallest_identified: f64,
        /// The largest unidentified eigenvalue `σ_{r+1}`; `None` at full rank.
        largest_unidentified: Option<f64>,
        /// `H`'s rounding band `p·ε·‖H‖₂`.
        band: f64,
    },
    /// Not evaluated, for the typed reason.
    NotEvaluated { reason: RankConstancyNotEvaluated },
}

/// Why the identified rank's constancy was not evaluated at the fitted smoothing
/// parameters (#2901 V22).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankConstancyNotEvaluated {
    /// The fit has no smoothing parameters, so there is no step to certify over.
    NoSmoothingParameters,
    /// The outer search also moved link coordinates, whose Hessian drift the
    /// bound does not model.
    LinkCoordinates,
    /// The criterion priced the Hessian on an active linear-constraint face.
    ActiveConstraintFace,
    /// The fit carries no row curvature derivative to bound the weight motion by.
    NoRowCurvatureDerivative,
    /// The outer search tracked no Hessian over the certified coordinates, so its
    /// certificate has no Newton step.
    NoOuterHessian,
    /// The criterion priced every mode from the Hessian's root (#2644, gam#2735,
    /// gam#2959). The root resolves modes below the assembled Hessian's rounding
    /// band, which step bounds taken on the assembled matrix cannot resolve.
    RootScalePricedRank,
    /// The criterion's builder published no rank decision at the fitted
    /// smoothing parameters: the sparse route prices a strict factorization.
    NoPublishedRankDecision,
}

impl RankConstancyNotEvaluated {
    /// The reason, in words.
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::NoSmoothingParameters => {
                "the fit has no smoothing parameters, so there is no step to certify over"
            }
            Self::LinkCoordinates => {
                "the outer search also moved link coordinates, whose Hessian drift this bound \
                 does not model"
            }
            Self::ActiveConstraintFace => {
                "the criterion priced the Hessian on an active linear-constraint face"
            }
            Self::NoRowCurvatureDerivative => {
                "the fit carries no row curvature derivative to bound the weight motion by"
            }
            Self::NoOuterHessian => {
                "the outer search tracked no Hessian over the certified coordinates, so its \
                 certificate has no Newton step"
            }
            Self::RootScalePricedRank => {
                "the criterion priced every mode from the Hessian's root, which resolves modes \
                 below the assembled Hessian's rounding band that step bounds taken on the \
                 assembled matrix cannot"
            }
            Self::NoPublishedRankDecision => {
                "the criterion's builder published no rank decision at the fitted smoothing \
                 parameters"
            }
        }
    }
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
///
/// This type now lives in the neutral `gam-problem` crate; re-exported here so
/// all existing `crate::model_types::BlockRole` / `gam::estimate::BlockRole`
/// references keep resolving.
pub use gam_problem::BlockRole;

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

/// Owned diagonal working-set evidence at convergence.
///
/// This evidence is distinct from coefficient geometry: Exact-Newton and
/// multi-parameter fits can retain an exact coefficient gauge and penalized
/// Hessian without having a single row-wise IRLS representation. Consumers
/// that mathematically require row evidence (currently ALO and constrained
/// Gaussian centering) must explicitly require this value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkingGeometry {
    /// Score-side Fisher IRLS weights paired with `response`.
    pub weights: Array1<f64>,
    /// IRLS working response at convergence.
    pub response: Array1<f64>,
}

/// Coefficient geometry retained at convergence for inference and post-fit
/// diagnostics.
///
/// The saved coefficient blocks are always in the raw reporting frame. The
/// geometry may occupy a smaller active frame; `coefficient_gauge` is the
/// required affine map `β_saved = T θ_active + a` connecting the two.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitGeometry {
    /// Exact affine lift from the active geometry coordinates (the columns of
    /// `T`) to the saved/raw coefficient blocks (the rows of `T`). This field is
    /// intentionally required on the wire: models saved before the active-frame
    /// schema must be regenerated rather than guessed as identity.
    pub coefficient_gauge: gam_problem::gauge::Gauge,
    /// Joint penalized Hessian `H = X'W_HX + S(λ)` at convergence, in the
    /// active coordinates of `coefficient_gauge` (dimension
    /// `coefficient_gauge.reduced_total()`).
    /// Stored as `UnscaledPrecision` so the dispersion-ownership invariant
    /// (this matrix is *not* φ-scaled) is enforced at the type level.
    pub penalized_hessian: gam_problem::dispersion_cov::UnscaledPrecision,
    /// Exact inequality-truncated posterior identity in the active coefficient
    /// frame, when linear inequality constraints are part of the fitted model.
    ///
    /// This is deliberately part of the required wire schema: old constrained
    /// models must be regenerated rather than silently treating their reported
    /// coefficient vector as both a posterior mean and an optimizer mode.
    pub constrained_posterior: Option<crate::constrained_posterior::ConstrainedPosteriorGeometry>,
    /// Optional owned row-wise diagonal IRLS evidence. `None` is a typed
    /// statement that the terminal solver geometry has no single diagonal
    /// row representation; it is never represented by empty or zero-filled
    /// placeholder vectors.
    ///
    /// It is in-memory fit evidence only and is never serialized: both vectors
    /// have one entry per training row, and a saved model carries no per-row
    /// training data (speed F6). A loaded fit therefore reads `None`, and a
    /// payload written at payload version 28 or earlier, which serialized it, has its
    /// `working` key read past.
    #[serde(skip)]
    pub working: Option<WorkingGeometry>,
}

#[derive(Clone)]
pub struct UnifiedFitResultParts {
    pub blocks: Vec<FittedBlock>,
    /// Number of original training rows / experimental units. This is row
    /// count, not positive-weight count and not row count multiplied by the
    /// number of response coordinates.
    pub training_sample_size: usize,
    /// [`fn@training_response_fingerprint`] of the rows this fit was trained on,
    /// when the construction site holds them (#4556 P3).
    ///
    /// `None` is "this construction has no training response to identify" — a
    /// fit rebuilt from a payload that predates the field, a fixture, a
    /// prediction-side reconstruction — and never "the data did not matter".
    /// A comparison reads it as an absence of established provenance and falls
    /// back to the necessary conditions it can test.
    pub training_response_fingerprint: Option<u64>,
    pub log_lambdas: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub likelihood_family: Option<LikelihoodSpec>,
    pub likelihood_scale: LikelihoodScaleMetadata,
    pub log_likelihood_normalization: LogLikelihoodNormalization,
    pub log_likelihood: f64,
    pub deviance: f64,
    /// The fit's REML/LAML criterion, or `None` when no finite criterion exists
    /// at this fit (the exact-fit Gaussian boundary). See
    /// [`UnifiedFitResult::reml_score`].
    pub reml_score: Option<f64>,
    pub stable_penalty_term: f64,
    /// Absent exactly when [`Self::reml_score`] is absent; the constructor
    /// rejects any other pairing.
    pub penalized_objective: Option<f64>,
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
    pub block_states: Vec<gam_problem::ParameterBlockState>,
    // Fields every construction site already fills. They were `#[doc(hidden)]`
    // as "backward-compatible" leftovers, which hid five REQUIRED fields from
    // the docs of a struct whose whole job is to be filled in completely.
    pub pirls_status: crate::pirls::PirlsStatus,
    pub max_abs_eta: f64,
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    pub artifacts: FitArtifacts,
    pub inner_cycles: usize,
}

impl FitConvergenceEvidence {
    /// The inner certificate's own evidence, rendered for a refusal that is
    /// otherwise unreadable (#2705 group B).
    ///
    /// `FitDidNotConverge` names the inner status and then quotes the OUTER
    /// stationarity residual, because that is the only certificate it holds.
    /// On a fit refused for `inner status StalledAtValidMinimum` those two
    /// numbers describe different problems, and the reader is left to guess why
    /// the inner mode fell short — which is exactly the question #2705 group B
    /// records as open. The inner certificate is
    ///
    /// ```text
    ///     ‖g‖ < tol·√n·√p        OR        ‖g‖ / (1 + natural scale) < tol
    /// ```
    ///
    /// and for a CONSTRAINED fit `‖g‖` is not a gradient norm at all: it is
    /// `max(primal feasibility, dual feasibility, complementarity, stationarity)`
    /// over the constraint-KKT channels. Printing that max without its parts
    /// hides which channel is deciding — and the four channels do not carry the
    /// same units, so which one binds is the whole diagnosis.
    fn inner_certificate_evidence(parts: &UnifiedFitResultParts) -> String {
        let Some(pirls) = parts.artifacts.pirls.as_ref() else {
            return String::new();
        };
        let n = pirls.final_eta.len().max(1) as f64;
        let p = (pirls.penalized_gradient_transformed.len() as f64).max(1.0);
        let relative_bound_scale = 1.0 + pirls.gradient_natural_scale;
        let tolerance_text = pirls
            .final_kkt_tolerance
            .map_or("not evaluated".to_string(), |value| format!("{value:.6e}"));
        let dimension_bound_text = pirls.final_kkt_tolerance.map_or_else(
            || "not evaluated".to_string(),
            |value| format!("{:.6e}", value * n.sqrt() * p.sqrt()),
        );
        let mut evidence = format!(
            " [inner certificate: kkt_tol={tolerance_text} \
             dimension_bound={dimension_bound_text} (tol·√{n:.0}·√{p:.0}) \
             natural_scale={:.6e} raw_gradient_l2={:.6e} iterations={} \
             last_deviance_change={:.6e}",
            pirls.gradient_natural_scale,
            pirls.lastgradient_norm,
            pirls.iteration,
            pirls.last_deviance_change,
        );
        if let Some(kkt) = pirls.constraint_kkt.as_ref() {
            let residual = kkt
                .primal_feasibility
                .max(kkt.dual_feasibility)
                .max(kkt.complementarity)
                .max(kkt.stationarity);
            let deciding = if residual == kkt.stationarity {
                "stationarity"
            } else if residual == kkt.complementarity {
                "complementarity"
            } else if residual == kkt.primal_feasibility {
                "primal_feasibility"
            } else {
                "dual_feasibility"
            };
            evidence.push_str(&format!(
                "; constrained residual={residual:.6e} decided by {deciding} \
                 (primal={:.6e} dual={:.6e} complementarity={:.6e} stationarity={:.6e}) \
                 active={}/{} rank_deficient={} gradient_scale={:.6e} \
                 relative={:.6e} vs tol={tolerance_text}",
                kkt.primal_feasibility,
                kkt.dual_feasibility,
                kkt.complementarity,
                kkt.stationarity,
                kkt.n_active,
                kkt.n_constraints,
                kkt.working_set_rank_deficient,
                kkt.gradient_scale,
                residual / relative_bound_scale,
            ));
        } else {
            evidence.push_str(&format!(
                "; unconstrained relative={:.6e} vs tol={tolerance_text}",
                pirls.lastgradient_norm / relative_bound_scale,
            ));
        }
        evidence.push(']');
        evidence
    }

    fn assembly_error(parts: &UnifiedFitResultParts, outer_status: String) -> EstimationError {
        let certificate = parts.artifacts.criterion_certificate.as_ref();
        EstimationError::FitDidNotConverge {
            inner_status: format!(
                "{:?}{}",
                parts.pirls_status,
                Self::inner_certificate_evidence(parts)
            ),
            outer_status,
            outer_iterations: parts.outer_iterations,
            final_value: parts.reml_score,
            // Both halves from the SAME certificate or neither. The residual
            // used to fall back to `parts.outer_gradient_norm` while the bound
            // had no fallback, so a run with no certificate reported a residual
            // against nothing -- `Some(0.0) against None`, recorded on #2471.
            // A norm of different provenance is not a degraded version of the
            // certified residual; it is a different quantity.
            stationarity: certificate.map_or(FitStationarityEvidence::NoComparison, |value| {
                FitStationarityEvidence::Certified {
                    residual: value.stationarity.projected_norm(),
                    bound: value.stationarity.bound(),
                }
            }),
            // The solver exports no accepted-step residual on this path, so
            // there is no step comparison to report either.
            step: FitStationarityEvidence::NoComparison,
            rho_checkpoint: parts.log_lambdas.to_vec(),
            resume_token: None,
        }
    }

    fn try_from_parts(parts: &UnifiedFitResultParts) -> Result<Self, EstimationError> {
        // Inner and outer stationarity are independent obligations. An outer
        // envelope certificate cannot prove that the coefficient mode
        // converged, so every diagnostic stalled/exhausted checkpoint remains
        // non-minting. A final-state gate that truly discharges convergence
        // must promote the status to `Converged` before assembly; constructors
        // never reinterpret a non-converged status as a model (SPEC rule 20).
        if !parts.pirls_status.is_converged() {
            return Err(Self::assembly_error(
                parts,
                "outer evidence was not considered because the inner mode did not report convergence"
                    .to_string(),
            ));
        }
        if !parts.outer_converged {
            return Err(Self::assembly_error(
                parts,
                "optimizer reported non-convergence".to_string(),
            ));
        }

        let no_smoothing_coordinate =
            has_no_smoothing_coordinate(&parts.log_lambdas, &parts.artifacts);
        let (outer_iterations, outer) = if no_smoothing_coordinate {
            // A zero-dimensional analytic certificate (|g|=|Pg|=bound=0)
            // proves no equation: there was no smoothing coordinate to
            // optimize. Some orchestration paths still report one
            // administrative pass through the outer driver; dimensionality,
            // not that implementation counter, is the semantic authority.
            // Canonicalize both artifacts to the exact `Fixed` representation.
            (0, FitOuterConvergenceEvidence::Fixed)
        } else {
            let outer = match parts.artifacts.criterion_certificate.as_ref() {
                Some(certificate) if certificate.certifies() => {
                    FitOuterConvergenceEvidence::Analytic(certificate.clone())
                }
                Some(certificate) => {
                    return Err(Self::assembly_error(
                        parts,
                        format!("analytic certificate failed: {}", certificate.summary()),
                    ));
                }
                None if parts.outer_iterations == 0 => FitOuterConvergenceEvidence::Fixed,
                None => {
                    return Err(Self::assembly_error(
                        parts,
                        "outer iterations ran without an analytic stationarity certificate"
                            .to_string(),
                    ));
                }
            };
            (parts.outer_iterations, outer)
        };

        Ok(Self {
            inner_status: parts.pirls_status,
            outer_iterations,
            outer,
        })
    }
}

/// Whether a fit had no smoothing coordinate to select: no per-block
/// log-precision and no joint-penalty log-precision.
///
/// A joint-penalty family attaches every smoothing coordinate to
/// `FitArtifacts::joint_log_lambdas` and leaves the per-block vector empty (the
/// multinomial per-class carrier, gam#1587), so an empty `log_lambdas` alone is
/// not a zero-dimensional outer problem. Reading it as one erased the certified
/// outer evidence of every such fit (#2898).
fn has_no_smoothing_coordinate(log_lambdas: &Array1<f64>, artifacts: &FitArtifacts) -> bool {
    log_lambdas.is_empty()
        && artifacts
            .joint_log_lambdas
            .as_ref()
            .is_none_or(|joint| joint.is_empty())
}

#[cfg(test)]
mod training_response_fingerprint_tests {
    use super::training_response_fingerprint;
    use ndarray::array;

    /// #4556 P3. The identity is of the VALUES: the same rows read twice agree,
    /// and a change to one observation, to one weight, or to the row count moves
    /// it. The last one is what makes a fingerprint stronger than the row count
    /// it travels beside — two responses of different lengths cannot collide,
    /// because the hash absorbs the shape before the values.
    #[test]
    fn the_fingerprint_is_the_response_and_its_weights_by_value_4556() {
        let response = array![0.5, -1.25, 3.0, 0.0];
        let weights = array![1.0, 1.0, 2.5, 1.0];
        let identity = training_response_fingerprint(&[response.view(), weights.view()]);
        assert_eq!(
            identity,
            training_response_fingerprint(&[response.view(), weights.view()]),
            "the same rows read twice are one experiment"
        );

        let moved_response = array![0.5, -1.25, 3.0, 1e-300];
        assert_ne!(
            identity,
            training_response_fingerprint(&[moved_response.view(), weights.view()]),
            "one observation moved, by any representable amount, is different data"
        );

        let moved_weights = array![1.0, 1.0, 2.5, 1.0 + f64::EPSILON];
        assert_ne!(
            identity,
            training_response_fingerprint(&[response.view(), moved_weights.view()]),
            "a weighted likelihood on one response under two weightings is two experiments"
        );

        let shorter = array![0.5, -1.25, 3.0];
        let shorter_weights = array![1.0, 1.0, 2.5];
        assert_ne!(
            identity,
            training_response_fingerprint(&[shorter.view(), shorter_weights.view()]),
            "the shape is absorbed before the values, so lengths cannot collide"
        );

        // The function is total: a set whose lengths disagree is not a fit's
        // columns, and it gets its own identity rather than a panic or a
        // dropped column.
        assert_ne!(
            training_response_fingerprint(&[response.view(), shorter_weights.view()]),
            training_response_fingerprint(&[shorter.view(), shorter_weights.view()]),
        );
        assert_ne!(
            training_response_fingerprint(&[response.view(), shorter_weights.view()]),
            identity,
        );
    }

    /// Refs #4556. A survival transformation fit's likelihood reads four
    /// columns, and each of them makes two datasets two experiments — including
    /// the event code, which a `(response, weights)` rule cannot see at all:
    /// two datasets with identical entry times, exit times and weights that
    /// differ only in which rows were events are not one experiment, and before
    /// this the survival route published no identity rather than a partial one.
    ///
    /// The column COUNT is absorbed before any value, so the four-column
    /// identity of a survival fit can never equal the two-column identity of a
    /// standard fit over the same numbers.
    #[test]
    fn a_survival_fits_identity_reads_all_four_of_its_columns_4556() {
        let entry = array![0.0, 0.0, 1.5, 2.0];
        let exit = array![3.0, 4.5, 6.0, 7.25];
        let event = array![1.0, 0.0, 1.0, 0.0];
        let weights = array![1.0, 1.0, 1.0, 2.0];
        let columns = [entry.view(), exit.view(), event.view(), weights.view()];
        let identity = training_response_fingerprint(&columns);
        assert_eq!(
            identity,
            training_response_fingerprint(&columns),
            "the same rows read twice are one experiment"
        );

        for (index, moved) in [
            array![1e-300, 0.0, 1.5, 2.0],
            array![3.0, 4.5, 6.0, 7.25 + f64::EPSILON * 8.0],
            array![0.0, 1.0, 1.0, 0.0],
            array![1.0, 1.0, 1.0, 2.0 + f64::EPSILON * 2.0],
        ]
        .into_iter()
        .enumerate()
        {
            let mut moved_columns = columns;
            moved_columns[index] = moved.view();
            assert_ne!(
                identity,
                training_response_fingerprint(&moved_columns),
                "column {index} is part of the experiment and must move the identity"
            );
        }

        // The event column is the one a `(response, weights)` rule cannot see:
        // pricing the exit time as the response and the weights as the weights
        // leaves two datasets that differ only in their events indistinguishable.
        let flipped_event = array![0.0, 1.0, 1.0, 0.0];
        assert_eq!(
            training_response_fingerprint(&[exit.view(), weights.view()]),
            training_response_fingerprint(&[exit.view(), weights.view()]),
        );
        assert_ne!(
            training_response_fingerprint(&columns),
            training_response_fingerprint(&[
                entry.view(),
                exit.view(),
                flipped_event.view(),
                weights.view(),
            ]),
        );

        // Different column sets are different experiments by construction.
        assert_ne!(
            identity,
            training_response_fingerprint(&[exit.view(), weights.view()])
        );
    }
}

#[cfg(test)]
mod assembly_inner_status_gate_tests {
    use super::*;
    use crate::pirls::PirlsStatus;

    /// Assemble a minimal, otherwise-valid fit whose only variable is the inner
    /// PIRLS status. `outer_iterations = 0` routes the outer obligation through
    /// the `Fixed` evidence branch (no criterion certificate needed), so the
    /// pass/fail turns entirely on the inner-status gate under test.
    fn parts_with_inner_status(status: PirlsStatus) -> UnifiedFitResultParts {
        let p = 2usize;
        let mut hessian = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            hessian[[j, j]] = 1.0;
        }
        UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: Array1::zeros(p),
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::from_vec(vec![1.0]),
            }],
            training_sample_size: 64,
            training_response_fingerprint: None,
            log_lambdas: Array1::zeros(1),
            lambdas: Array1::from_vec(vec![1.0]),
            likelihood_family: Some(LikelihoodSpec::gaussian_identity()),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: Some(0.0),
            standard_deviation: 1.0,
            covariance_conditional: None,
            covariance_corrected: None,
            inference: Some(FitInference {
                edf_by_block: vec![1.0],
                penalty_block_trace: vec![0.0],
                edf_rank_bound: Vec::new(),
                edf_total: 1.0,
                smoothing_correction: None,
                smoothing_correction_method: None,
                smoothing_correction_first_order: None,
                smoothing_correction_method_first_order: None,
                smoothing_correction_absence: None,
                penalized_hessian: gam_problem::dispersion_cov::UnscaledPrecision::wrap(hessian),
                reparam_qs: None,
                dispersion: Dispersion::estimated(1.0)
                    .expect("1.0 is a valid estimated dispersion"),
                factorized_standard_errors: None,
                smoothing_correction_factorized: None,
                beta_covariance_frequentist: None,
                coefficient_influence: None,
                weighted_gram: None,
                identified_subspace: None,
                working_residual: None,
            }),
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: status,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts::default(),
            inner_cycles: 0,
        }
    }

    fn assemble_with_inner_status(
        status: PirlsStatus,
    ) -> Result<UnifiedFitResult, EstimationError> {
        UnifiedFitResult::try_from_parts(parts_with_inner_status(status))
    }

    /// SPEC rule 20 is deliberately status-exact: only a `Converged` inner
    /// optimization may mint. Stalled, exhausted, and unstable states remain
    /// checkpoints even when separate diagnostics describe a promising point.
    #[test]
    fn only_converged_inner_status_mints_a_fit() {
        assert!(
            assemble_with_inner_status(PirlsStatus::Converged).is_ok(),
            "a strictly converged inner mode must mint a fit"
        );
        for status in [
            PirlsStatus::StalledAtValidMinimum,
            PirlsStatus::MaxIterationsReached,
            PirlsStatus::LmStepSearchExhausted,
            PirlsStatus::Unstable,
        ] {
            let err = assemble_with_inner_status(status)
                .expect_err("a non-converged inner status must not mint a fit");
            assert!(
                matches!(err, EstimationError::FitDidNotConverge { .. }),
                "expected a non-convergence assembly error for {status:?}, got {err:?}"
            );
        }
    }

    /// #2898: a joint-penalty fit keeps its outer evidence. The multinomial
    /// per-class carrier attaches every smoothing coordinate to
    /// `joint_log_lambdas` and none to the per-block vector, so its empty
    /// `log_lambdas` is not a zero-dimensional outer problem: the certificate,
    /// the iteration count and the gradient norm of the certified outer search
    /// must survive assembly.
    #[test]
    fn joint_penalty_outer_artifacts_keep_their_certificate_2898() {
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        parts.blocks[0].lambdas = Array1::zeros(0);
        parts.log_lambdas = Array1::zeros(0);
        parts.lambdas = Array1::zeros(0);
        if let Some(inference) = parts.inference.as_mut() {
            inference.edf_by_block.clear();
            inference.penalty_block_trace.clear();
        }
        parts.artifacts.joint_log_lambdas = Some(Array1::from_vec(vec![0.4, -1.3]));
        parts.artifacts.criterion_certificate = Some(OuterCriterionCertificate {
            stationarity: OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 2e-7,
                projected_grad_norm: 2e-7,
                bound: 1e-5,
                rung: CertifiedRung {
                    label: "solver-band".to_string(),
                    derived_standard: false,
                },
            },
            curvature: CurvatureEvidence::Measured { psd: true },
            lambdas_railed: Vec::new(),
            railed_facts: Vec::new(),
            newton_polish: None,
            curvature_floor: None,
            criterion_error: None,
        });
        parts.outer_gradient_norm = Some(2e-7);
        parts.outer_iterations = 7;

        let fit = UnifiedFitResult::try_from_parts(parts)
            .expect("a certified joint-penalty fit must mint");
        assert!(
            fit.convergence_evidence().outer_certificate().is_some(),
            "joint smoothing coordinates are an outer stationarity equation"
        );
        assert!(
            fit.artifacts.criterion_certificate.is_some(),
            "the certificate of a joint-coordinate outer search must survive as an artifact"
        );
        assert_eq!(fit.outer_iterations, 7);
        assert_eq!(fit.convergence_evidence().outer_iterations(), 7);
        assert_eq!(fit.outer_gradient_norm, Some(2e-7));
    }

    /// #2898: joint smoothing coordinates make the correction `J Var(ρ) Jᵀ`
    /// nonzero, so a certified joint-penalty fit that persisted no corrected
    /// covariance publishes conditional instead of substituting `Vb` for `Vp`.
    #[test]
    fn joint_penalty_fit_without_a_persisted_correction_publishes_conditional_2898() {
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        parts.blocks[0].lambdas = Array1::zeros(0);
        parts.log_lambdas = Array1::zeros(0);
        parts.lambdas = Array1::zeros(0);
        parts.covariance_conditional = Some(Array2::from_diag(&Array1::from_vec(vec![2.0, 3.0])));
        if let Some(inference) = parts.inference.as_mut() {
            inference.edf_by_block.clear();
            inference.penalty_block_trace.clear();
        }
        parts.artifacts.joint_log_lambdas = Some(Array1::from_vec(vec![0.4, -1.3]));
        parts.artifacts.criterion_certificate = Some(OuterCriterionCertificate {
            stationarity: OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 2e-7,
                projected_grad_norm: 2e-7,
                bound: 1e-5,
                rung: CertifiedRung {
                    label: "solver-band".to_string(),
                    derived_standard: false,
                },
            },
            curvature: CurvatureEvidence::Measured { psd: true },
            lambdas_railed: Vec::new(),
            railed_facts: Vec::new(),
            newton_polish: None,
            curvature_floor: None,
            criterion_error: None,
        });
        parts.outer_gradient_norm = Some(2e-7);
        parts.outer_iterations = 7;

        let fit = UnifiedFitResult::try_from_parts(parts)
            .expect("a certified joint-penalty fit must mint");
        assert!(fit.beta_covariance().is_some());
        assert!(
            fit.beta_covariance_corrected().is_none(),
            "with joint smoothing coordinates Vb is not Vp"
        );
        assert_eq!(
            fit.published_covariance_mode(),
            InferenceCovarianceMode::Conditional
        );
    }

    /// `beta_standard_errors()² == diag(beta_covariance())` for both published
    /// pairs: the property one store gives by construction (#2955).
    fn assert_standard_errors_are_the_published_diagonal(fit: &UnifiedFitResult) {
        for (label, standard_errors, covariance) in [
            ("conditional", fit.beta_standard_errors(), fit.beta_covariance()),
            ("corrected", fit.beta_standard_errors_corrected(), fit.beta_covariance_corrected()),
        ] {
            let standard_errors =
                standard_errors.unwrap_or_else(|| panic!("{label} standard errors are published"));
            let covariance = covariance.unwrap_or_else(|| panic!("{label} covariance is published"));
            assert_eq!(standard_errors.len(), covariance.nrows(), "{label} width");
            for (i, (se, diagonal)) in standard_errors.iter().zip(covariance.diag().iter()).enumerate() {
                assert!(
                    (se * se - diagonal).abs() <= 4.0 * f64::EPSILON * diagonal.abs(),
                    "{label} standard error {i} squares to {:.17e} against the diagonal {diagonal:.17e}",
                    se * se
                );
            }
        }
    }

    /// gam#2943: a covariance correction reaches the published covariances and
    /// their standard errors in one step. The marginal-slope Murphy–Topel
    /// correction used to add to the top-level matrices only, so the inference
    /// copies kept the uncorrected matrix and every saved-model load refused the
    /// fit; with one store (#2955) there is no second copy to miss.
    #[test]
    fn covariance_correction_reaches_every_published_copy_2943() {
        let conditional = Array2::from_shape_vec((2, 2), vec![2.0, 0.3, 0.3, 3.0]).expect("2x2");
        let corrected = Array2::from_shape_vec((2, 2), vec![2.5, 0.4, 0.4, 3.5]).expect("2x2");
        let correction = Array2::from_shape_vec((2, 2), vec![0.5, -0.1, -0.1, 0.25]).expect("2x2");
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        parts.covariance_conditional = Some(conditional.clone());
        parts.covariance_corrected = Some(corrected.clone());
        let mut fit = UnifiedFitResult::try_from_parts(parts).expect("a consistent fit mints");

        let wrong_shape = Array2::<f64>::zeros((3, 3));
        assert!(
            fit.add_coefficient_covariance_correction(&wrong_shape).is_err(),
            "a correction of the wrong shape must be refused"
        );
        fit.add_coefficient_covariance_correction(&correction)
            .expect("the correction applies");

        assert_eq!(fit.covariance_conditional, Some(&conditional + &correction));
        assert_eq!(fit.covariance_corrected, Some(&corrected + &correction));
        assert_standard_errors_are_the_published_diagonal(&fit);

        // The corrected state mints again under try_from_parts.
        let mut round_trip = parts_with_inner_status(PirlsStatus::Converged);
        round_trip.covariance_conditional = fit.covariance_conditional.clone();
        round_trip.covariance_corrected = fit.covariance_corrected.clone();
        UnifiedFitResult::try_from_parts(round_trip)
            .expect("the corrected covariances must mint again under try_from_parts");

        // A correction cannot reach a diagonal published without its covariance.
        let mut factorized = parts_with_inner_status(PirlsStatus::Converged);
        if let Some(inference) = factorized.inference.as_mut() {
            inference.factorized_standard_errors = Some(Array1::from_vec(vec![1.0, 2.0]));
        }
        let mut factorized =
            UnifiedFitResult::try_from_parts(factorized).expect("factorized standard errors mint");
        assert!(
            factorized
                .add_coefficient_covariance_correction(&correction)
                .is_err(),
            "a correction must refuse standard errors published without their covariance"
        );
    }

    /// #2955: the standard errors derive from the one covariance store for both
    /// pairs, survive a `try_from_parts` re-mint of the fit's own fields, and
    /// are not written as a second copy.
    #[test]
    fn standard_errors_derive_from_the_one_covariance_store_2955() {
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        parts.covariance_conditional =
            Some(Array2::from_shape_vec((2, 2), vec![2.0, 0.3, 0.3, 3.0]).expect("2x2"));
        parts.covariance_corrected =
            Some(Array2::from_shape_vec((2, 2), vec![2.5, 0.4, 0.4, 3.5]).expect("2x2"));
        let fit = UnifiedFitResult::try_from_parts(parts).expect("a consistent fit mints");
        assert_standard_errors_are_the_published_diagonal(&fit);
        fit.validate_numeric_finiteness()
            .expect("the minted fit re-mints from its own fields");

        let encoded = serde_json::to_string(&fit).expect("serialize the fit");
        assert!(
            !encoded.contains("beta_standard_errors") && !encoded.contains("\"beta_covariance\""),
            "the one-store schema writes no covariance or standard-error copy"
        );
        let decoded: UnifiedFitResult = serde_json::from_str(&encoded).expect("the fit parses back");
        assert_eq!(decoded.beta_standard_errors(), fit.beta_standard_errors());
        assert_eq!(
            decoded.beta_standard_errors_corrected(),
            fit.beta_standard_errors_corrected()
        );
    }

    /// #2901: each block's EDF rank-bound status survives the saved-fit wire, and a fit
    /// written before the field existed parses back with it empty.
    #[test]
    fn the_edf_rank_bound_status_round_trips_and_reads_empty_when_absent_2901() {
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        let bounds = vec![crate::estimate::EdfRankBound::Uncertified {
            smallest_pivot: -3.3509,
            band: 1.0e-15,
        }];
        parts
            .inference
            .as_mut()
            .expect("the fixture carries inference")
            .edf_rank_bound = bounds.clone();
        let fit = UnifiedFitResult::try_from_parts(parts).expect("a consistent fit mints");
        let encoded = serde_json::to_value(&fit).expect("serialize the fit");
        let decoded: UnifiedFitResult =
            serde_json::from_value(encoded.clone()).expect("the fit parses back");
        assert_eq!(decoded.edf_rank_bound(), bounds.as_slice());

        let mut absent = encoded;
        absent["inference"]
            .as_object_mut()
            .expect("the fit serializes its inference block")
            .remove("edf_rank_bound")
            .expect("the status was written");
        let decoded: UnifiedFitResult =
            serde_json::from_value(absent).expect("a fit without the status parses");
        assert!(decoded.edf_rank_bound().is_empty(), "{:?}", decoded.edf_rank_bound());
    }

    /// #2954: a certificate saves its Newton polish and each railed coordinate's
    /// face kind, and one saved before either existed parses back with no polish
    /// and every face `Unrecorded`, never as a face kind it did not record.
    #[test]
    fn a_certificate_saved_before_the_polish_and_face_kinds_reads_them_as_unrecorded_2954() {
        let certificate = OuterCriterionCertificate {
            stationarity: OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 2e-7,
                projected_grad_norm: 2e-7,
                bound: 1e-5,
                rung: CertifiedRung {
                    label: "newton-decrement".to_string(),
                    derived_standard: true,
                },
            },
            curvature: CurvatureEvidence::Measured { psd: true },
            lambdas_railed: vec![0],
            railed_facts: vec![RailedCoordinateFact {
                index: 0,
                theta: 20.0,
                lower: -20.0,
                upper: 20.0,
                margin: 0.5,
                face: RailFaceKind::LimitModel,
            }],
            newton_polish: Some(NewtonPolishRecord {
                lambda_sq_before: 1.0e-4,
                lambda_sq_after: 0.0,
                decreases: vec![6.3e-5],
                settled: false,
                rails: vec![NewtonPolishRail {
                    index: 0,
                    from: 17.0,
                    to: 20.0,
                    decrease: 1.2e-5,
                    steps_before: 1,
                    face: RailFaceKind::LimitModel,
                }],
                entry: vec![17.0],
            }),
            curvature_floor: None,
            criterion_error: None,
        };
        let encoded = serde_json::to_value(&certificate).expect("serialize the certificate");
        let decoded: OuterCriterionCertificate =
            serde_json::from_value(encoded.clone()).expect("the certificate parses back");
        assert_eq!(decoded.railed_facts[0].face, RailFaceKind::LimitModel);
        let polish = decoded.newton_polish.expect("the polish round-trips");
        assert_eq!(polish.rails.len(), 1);
        assert_eq!(polish.rails[0].face, RailFaceKind::LimitModel);

        let mut older = encoded;
        older
            .as_object_mut()
            .expect("the certificate serializes as an object")
            .remove("newton_polish")
            .expect("the polish was written");
        older["railed_facts"][0]
            .as_object_mut()
            .expect("a railed fact serializes as an object")
            .remove("face")
            .expect("the face kind was written");
        let decoded: OuterCriterionCertificate =
            serde_json::from_value(older).expect("an older certificate parses");
        assert!(decoded.newton_polish.is_none());
        assert_eq!(decoded.railed_facts[0].face, RailFaceKind::Unrecorded);
    }

    /// #2954: a coordinate the certificate's polish carried onto a face seeds
    /// another search from the interior value the search handed the polish; the
    /// published `λ` and the rest of the seed are the fit's own.
    #[test]
    fn a_polish_railed_coordinate_seeds_the_next_search_from_its_entry_2954() {
        let log_lambdas = ndarray::array![-19.63, 2.5];
        let mut certificate = OuterCriterionCertificate {
            stationarity: OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 2e-7,
                projected_grad_norm: 2e-7,
                bound: 1e-5,
                rung: CertifiedRung {
                    label: "newton-decrement".to_string(),
                    derived_standard: true,
                },
            },
            curvature: CurvatureEvidence::Measured { psd: true },
            lambdas_railed: vec![0],
            railed_facts: vec![RailedCoordinateFact {
                index: 0,
                theta: -19.63,
                lower: -19.63,
                upper: 20.0,
                margin: 0.5,
                face: RailFaceKind::LimitModel,
            }],
            newton_polish: None,
            curvature_floor: None,
            criterion_error: None,
        };
        assert_eq!(
            search_seed_from(log_lambdas.clone(), Some(&certificate)),
            log_lambdas,
            "without a polish the fit's own log λ seeds"
        );
        certificate.newton_polish = Some(NewtonPolishRecord {
            lambda_sq_before: 1.0e-4,
            lambda_sq_after: 0.0,
            decreases: vec![1.0e-6],
            settled: false,
            rails: Vec::new(),
            entry: vec![-17.05, 2.49],
        });
        assert_eq!(
            search_seed_from(log_lambdas, Some(&certificate)),
            ndarray::array![-17.05, 2.5],
            "the railed coordinate seeds from its entry, the free one from the fit"
        );
    }

    /// #2955: a covariance whose diagonal `se_from_covariance` refuses is refused
    /// at mint, by name, so the derived standard errors need no `Result`.
    #[test]
    fn a_materially_negative_covariance_diagonal_is_refused_at_mint_2955() {
        let negative = Array2::from_shape_vec((2, 2), vec![-1.0, 0.0, 0.0, 2.0]).expect("2x2");
        for label in ["conditional", "smoothing-corrected"] {
            let mut parts = parts_with_inner_status(PirlsStatus::Converged);
            if label == "conditional" {
                parts.covariance_conditional = Some(negative.clone());
            } else {
                parts.covariance_corrected = Some(negative.clone());
            }
            let message = UnifiedFitResult::try_from_parts(parts)
                .expect_err("a materially negative diagonal must be refused")
                .to_string();
            assert!(
                message.contains(&format!("{label} covariance has an invalid diagonal")),
                "refused for another reason: {message}"
            );
        }

        // A roundoff-sized negative diagonal is inside `se_from_covariance`'s own
        // band: it mints and derives a zero standard error (gam-2929's arm).
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        parts.covariance_conditional =
            Some(Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, -1.0e-17]).expect("2x2"));
        let snapped = UnifiedFitResult::try_from_parts(parts)
            .expect("a roundoff-sized negative diagonal mints");
        assert_eq!(
            snapped.beta_standard_errors().expect("standard errors")[1],
            0.0,
            "a roundoff-sized negative diagonal derives a zero standard error"
        );

        // A correction that drives a diagonal materially negative is refused
        // before either store changes.
        let unit = Array2::from_diag(&Array1::from_vec(vec![2.0, 3.0]));
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        parts.covariance_conditional = Some(unit.clone());
        let mut fit = UnifiedFitResult::try_from_parts(parts).expect("a consistent fit mints");
        let correction = Array2::from_diag(&Array1::from_vec(vec![0.0, -4.0]));
        let message = fit
            .add_coefficient_covariance_correction(&correction)
            .expect_err("a correction that drives a diagonal materially negative must be refused")
            .to_string();
        assert!(
            message.contains("corrected conditional covariance has an invalid diagonal"),
            "refused for another reason: {message}"
        );
        assert_eq!(
            fit.covariance_conditional.as_ref(),
            Some(&unit),
            "a refused correction leaves the fit as it was"
        );
    }

    /// #2955: standard errors beside a conditional covariance are a second copy
    /// of its diagonal, so the fit is refused. Without a covariance they are the
    /// factorized branch's published uncertainty (#2960).
    #[test]
    fn factorized_standard_errors_beside_a_covariance_are_refused_2955() {
        let standard_errors = Array1::from_vec(vec![1.0, 2.0]);
        let mut alone = parts_with_inner_status(PirlsStatus::Converged);
        if let Some(inference) = alone.inference.as_mut() {
            inference.factorized_standard_errors = Some(standard_errors.clone());
        }
        let fit = UnifiedFitResult::try_from_parts(alone).expect("factorized standard errors alone mint");
        assert_eq!(fit.beta_standard_errors(), Some(standard_errors.clone()));

        let mut beside = parts_with_inner_status(PirlsStatus::Converged);
        beside.covariance_conditional = Some(Array2::from_diag(&Array1::from_vec(vec![4.0, 9.0])));
        if let Some(inference) = beside.inference.as_mut() {
            inference.factorized_standard_errors = Some(standard_errors);
        }
        let message = UnifiedFitResult::try_from_parts(beside)
            .expect_err("a second diagonal beside the store must be refused")
            .to_string();
        assert!(message.contains("beside a conditional covariance"), "{message}");
    }

    /// #3283: the factorized branch's smoothing correction (the fixture has one
    /// smoothing coordinate) mints only beside the standard errors it was solved with, is what the corrected standard errors
    /// and the published covariance definition read on a fit with no covariance,
    /// and never rides beside a published corrected covariance.
    #[test]
    fn a_factorized_smoothing_correction_mints_only_beside_factorized_standard_errors_3283() {
        let standard_errors = Array1::from_vec(vec![1.0, 2.0]);
        let corrected = Array1::from_vec(vec![3.0, 4.0]);
        let factorized = FactorizedSmoothingCorrection {
            factor: Array2::from_shape_vec((2, 1), vec![0.5, -1.5]).expect("2x1"),
            standard_errors: corrected.clone(),
        };
        let mut alone = parts_with_inner_status(PirlsStatus::Converged);
        if let Some(inference) = alone.inference.as_mut() {
            inference.factorized_standard_errors = Some(standard_errors.clone());
            inference.smoothing_correction_factorized = Some(factorized.clone());
        }
        let fit = UnifiedFitResult::try_from_parts(alone)
            .expect("a factorized correction beside factorized standard errors mints");
        assert_eq!(fit.beta_standard_errors(), Some(standard_errors.clone()));
        assert_eq!(fit.beta_standard_errors_corrected(), Some(corrected.clone()));
        assert_eq!(
            fit.published_covariance_mode(),
            InferenceCovarianceMode::SmoothingCorrected,
            "a fit carrying its factorized correction publishes the corrected definition"
        );

        let mut without_errors = parts_with_inner_status(PirlsStatus::Converged);
        if let Some(inference) = without_errors.inference.as_mut() {
            inference.smoothing_correction_factorized = Some(factorized.clone());
        }
        let message = UnifiedFitResult::try_from_parts(without_errors)
            .expect_err("a factorized correction without its standard errors must be refused")
            .to_string();
        assert!(message.contains("factorized smoothing correction beside"), "{message}");

        let mut short = parts_with_inner_status(PirlsStatus::Converged);
        if let Some(inference) = short.inference.as_mut() {
            inference.factorized_standard_errors = Some(standard_errors);
            inference.smoothing_correction_factorized = Some(FactorizedSmoothingCorrection {
                factor: Array2::from_shape_vec((1, 1), vec![0.5]).expect("1x1"),
                standard_errors: corrected,
            });
        }
        let message = UnifiedFitResult::try_from_parts(short)
            .expect_err("a factor with the wrong row count must be refused")
            .to_string();
        assert!(message.contains("factorized smoothing correction shape mismatch"), "{message}");
    }

    /// #2955: an inference block written before the one-store schema carries the
    /// covariance copies. The reader drops them and the standard errors come from
    /// the top-level store, and a fit that published only standard errors keeps
    /// them as its factorized standard errors.
    #[test]
    fn a_covariance_copies_inference_block_reads_the_top_level_store_2955() {
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        parts.covariance_conditional = Some(Array2::from_diag(&Array1::from_vec(vec![4.0, 9.0])));
        let dense = UnifiedFitResult::try_from_parts(parts).expect("a dense fit mints");
        let mut encoded = serde_json::to_value(&dense).expect("serialize the dense fit");
        let inference = encoded["inference"]
            .as_object_mut()
            .expect("the fit serializes its inference block");
        inference.insert(
            "beta_covariance".to_string(),
            serde_json::to_value(dense.beta_covariance()).expect("encode the copy"),
        );
        inference.insert(
            "beta_standard_errors".to_string(),
            serde_json::to_value(Array1::from_vec(vec![7.0, 7.0])).expect("encode a copy"),
        );
        let decoded: UnifiedFitResult =
            serde_json::from_value(encoded).expect("the copies-schema fit parses");
        decoded
            .validate_numeric_finiteness()
            .expect("the copies-schema fit mints");
        assert_eq!(
            decoded.beta_standard_errors(),
            Some(Array1::from_vec(vec![2.0, 3.0])),
            "the standard errors come from the top-level store, not the copy"
        );

        let only_standard_errors = UnifiedFitResult::try_from_parts(parts_with_inner_status(
            PirlsStatus::Converged,
        ))
        .expect("a fit without a covariance mints");
        let mut encoded =
            serde_json::to_value(&only_standard_errors).expect("serialize the covariance-free fit");
        let inference = encoded["inference"]
            .as_object_mut()
            .expect("the fit serializes its inference block");
        inference.remove("factorized_standard_errors");
        inference.insert(
            "beta_standard_errors".to_string(),
            serde_json::to_value(Array1::from_vec(vec![1.5, 2.5])).expect("encode the diagonal"),
        );
        let decoded: UnifiedFitResult =
            serde_json::from_value(encoded).expect("the copies-schema fit parses");
        decoded
            .validate_numeric_finiteness()
            .expect("the copies-schema fit mints");
        assert_eq!(
            decoded.beta_standard_errors(),
            Some(Array1::from_vec(vec![1.5, 2.5])),
            "a fit that published only standard errors keeps them"
        );
    }

    #[test]
    fn zero_dimensional_outer_artifacts_canonicalize_to_fixed_evidence() {
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        parts.blocks[0].lambdas = Array1::zeros(0);
        parts.log_lambdas = Array1::zeros(0);
        parts.lambdas = Array1::zeros(0);
        parts.covariance_conditional = Some(Array2::from_diag(&Array1::from_vec(vec![2.0, 3.0])));
        if let Some(inference) = parts.inference.as_mut() {
            inference.edf_by_block.clear();
            inference.penalty_block_trace.clear();
        }
        parts.artifacts.criterion_certificate = Some(OuterCriterionCertificate {
            stationarity: OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 0.0,
                projected_grad_norm: 0.0,
                bound: 0.0,
                // No outer estimand on this path either: stationary by
                // construction, not by clearing a band (#2530).
                rung: StationarityRung::EMPTY_ESTIMAND.into(),
            },
            curvature: CurvatureEvidence::NoEstimand,
            lambdas_railed: Vec::new(),
            railed_facts: Vec::new(),
            newton_polish: None,
            curvature_floor: None,
            criterion_error: None,
        });
        parts.outer_gradient_norm = Some(0.0);
        parts.outer_iterations = 1;

        let fit = UnifiedFitResult::try_from_parts(parts)
            .expect("a converged fit with no outer coordinate must mint");
        assert!(
            fit.convergence_evidence().outer_certificate().is_none(),
            "no smoothing coordinate means no outer stationarity equation"
        );
        assert!(
            fit.artifacts.criterion_certificate.is_none(),
            "the vacuous zero-dimensional certificate must not survive as an artifact"
        );
        assert!(
            fit.outer_gradient_norm.is_none(),
            "a zero-length gradient is absence, not a measured zero norm"
        );
        assert_eq!(
            fit.outer_iterations, 0,
            "an administrative outer-driver pass is not an optimized coordinate"
        );
        assert_eq!(
            fit.convergence_evidence().outer_iterations(),
            0,
            "sealed fixed-outer evidence must use the canonical zero count"
        );
        assert_eq!(
            fit.beta_covariance_corrected(),
            fit.beta_covariance(),
            concat!(
                "with no smoothing coordinate Vp = Vb exactly; corrected prediction must consume ",
                "the persisted conditional covariance rather than refusing an algebraically complete fit",
            )
        );
    }

    #[test]
    fn training_sample_size_is_required_and_owns_the_wald_denominator() {
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        parts.geometry = None;
        parts.training_sample_size = 73;
        let fit = UnifiedFitResult::try_from_parts(parts)
            .expect("a fit does not need working-row evidence to retain its training row count");
        assert_eq!(fit.training_sample_size(), 73);
        assert_eq!(fit.wald_residual_degrees_of_freedom(), Some(72.0));

        let mut encoded = serde_json::to_value(&fit).expect("serialize fit");
        encoded
            .as_object_mut()
            .expect("fit serializes as an object")
            .remove("training_sample_size");
        let error = serde_json::from_value::<UnifiedFitResult>(encoded)
            .expect_err("pre-training-size wire data must not deserialize");
        assert!(
            error.to_string().contains("training_sample_size"),
            "missing required wire field reported an unrelated error: {error}"
        );

        let mut zero_encoded = serde_json::to_value(&fit).expect("serialize fit");
        zero_encoded
            .as_object_mut()
            .expect("fit serializes as an object")
            .insert("training_sample_size".to_string(), serde_json::json!(0));
        let error = serde_json::from_value::<UnifiedFitResult>(zero_encoded)
            .expect_err("zero training rows must not deserialize");
        assert!(
            error.to_string().contains("nonzero"),
            "zero-row wire rejection reported an unrelated error: {error}"
        );

        let mut zero = parts_with_inner_status(PirlsStatus::Converged);
        zero.training_sample_size = 0;
        let error = UnifiedFitResult::try_from_parts(zero)
            .expect_err("zero training rows cannot mint a fitted result");
        assert!(
            error.to_string().contains("training_sample_size"),
            "zero-row rejection reported an unrelated error: {error}"
        );
    }

    /// #4556 P3. The identity has to survive the fit result's own wire, or a
    /// saved model is compared on nothing: `compare_models` reads it from the
    /// summary, the summary reads it from the fit result, and the payload
    /// persists that fit result. A wire written before the field existed decodes
    /// as no provenance rather than failing or acquiring one.
    #[test]
    fn the_training_response_fingerprint_survives_the_fit_results_wire_4556() {
        let identity = 0x0123_4567_89ab_cdef_u64;
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        parts.geometry = None;
        parts.training_response_fingerprint = Some(identity);
        let fit = UnifiedFitResult::try_from_parts(parts).expect("a fit carries its provenance");
        assert_eq!(fit.training_response_fingerprint(), Some(identity));

        let encoded = serde_json::to_value(&fit).expect("serialize fit");
        let decoded: UnifiedFitResult =
            serde_json::from_value(encoded).expect("a fit round-trips its own wire");
        assert_eq!(
            decoded.training_response_fingerprint(),
            Some(identity),
            "the identity of the training rows must survive save and load"
        );

        let mut older = serde_json::to_value(&fit).expect("serialize fit");
        older
            .as_object_mut()
            .expect("fit serializes as an object")
            .remove("training_response_fingerprint");
        let older: UnifiedFitResult =
            serde_json::from_value(older).expect("a model saved before the field still loads");
        assert_eq!(
            older.training_response_fingerprint(),
            None,
            "a wire that carries no identity establishes none"
        );
    }

    /// A fit solved on a reduced second block (`X_fit = X_saved·T`, `T` 2×1)
    /// with its active geometry and covariance, plus the saved-frame lift
    /// `J = blockdiag(I₂, T)`.
    fn reduced_frame_fit() -> (UnifiedFitResult, gam_problem::Gauge, Array2<f64>) {
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        let active_beta = [
            Array1::from_vec(vec![0.5, -1.0]),
            Array1::from_vec(vec![2.0]),
        ];
        parts.blocks[0].beta = active_beta[0].clone();
        parts.blocks.push(FittedBlock {
            beta: active_beta[1].clone(),
            role: BlockRole::Scale,
            edf: 1.0,
            lambdas: Array1::zeros(0),
        });
        parts.block_states = active_beta
            .iter()
            .map(|beta| gam_problem::ParameterBlockState {
                beta: beta.clone(),
                eta: Array1::zeros(3),
            })
            .collect();
        let hessian = ndarray::array![[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]];
        let covariance = ndarray::array![[0.3, -0.1, 0.02], [-0.1, 0.4, -0.1], [0.02, -0.1, 0.6]];
        parts.covariance_conditional = Some(covariance.clone());
        if let Some(inference) = parts.inference.as_mut() {
            inference.penalized_hessian =
                gam_problem::dispersion_cov::UnscaledPrecision::wrap(hessian.clone());
        }
        parts.geometry = Some(FitGeometry {
            coefficient_gauge: gam_problem::Gauge::from_block_transforms(&[
                Array2::eye(2),
                Array2::eye(1),
            ]),
            penalized_hessian: hessian.into(),
            constrained_posterior: None,
            working: None,
        });
        let fit = UnifiedFitResult::try_from_parts(parts).expect("reduced-frame fixture assembles");
        let frame = gam_problem::Gauge::from_block_transforms(&[
            Array2::eye(2),
            ndarray::array![[0.6], [0.8]],
        ]);
        (fit, frame, covariance)
    }

    /// gam#3021: the saved-frame lift carries every coefficient-indexed
    /// quantity at once. β and the block states move to the raw widths, the
    /// covariance pushes forward as `JΣJᵀ`, and the Hessian stays in its active
    /// frame behind the composed gauge — the result passes the same invariants
    /// a saved model is decoded through.
    #[test]
    fn lift_to_saved_frame_moves_every_coefficient_quantity_together_3021() {
        let (mut fit, frame, covariance) = reduced_frame_fit();
        let active_hessian = fit.geometry.as_ref().unwrap().penalized_hessian.as_array().clone();
        fit.lift_to_saved_frame(&frame).expect("exact lift succeeds");

        let saved_slope = ndarray::array![1.2, 1.6];
        assert_eq!(fit.blocks[1].beta, saved_slope);
        assert_eq!(fit.block_states[1].beta, saved_slope);
        assert_eq!(fit.beta, ndarray::array![0.5, -1.0, 1.2, 1.6]);
        let lifted_covariance = fit.covariance_conditional.as_ref().unwrap();
        assert_eq!(lifted_covariance, &frame.lift_covariance(&covariance));
        assert!((lifted_covariance[[2, 3]] - 0.6 * 0.8 * 0.6).abs() < 1e-15);
        let geometry = fit.geometry.as_ref().unwrap();
        assert_eq!(geometry.coefficient_gauge.raw_widths(), vec![2, 2]);
        assert_eq!(geometry.coefficient_gauge.reduced_total(), 3);
        assert_eq!(geometry.penalized_hessian.as_array(), &active_hessian);
        fit.validate_numeric_finiteness()
            .expect("the lifted fit satisfies the decode invariants");
    }

    /// A quantity with no unique pushforward through a non-square lift is
    /// refused, and the refused lift leaves the fit untouched.
    #[test]
    fn lift_to_saved_frame_refuses_a_pullback_and_leaves_the_fit_3021() {
        let (mut fit, frame, _) = reduced_frame_fit();
        if let Some(inference) = fit.inference.as_mut() {
            inference.coefficient_influence = Some(Array2::eye(3));
        }
        let before = fit.beta.clone();
        let error = fit
            .lift_to_saved_frame(&frame)
            .expect_err("the influence matrix cannot be pushed forward");
        assert!(error.to_string().contains("coefficient influence"), "{error}");
        assert_eq!(fit.beta, before);
        assert_eq!(fit.blocks[1].beta.len(), 1);
    }

    /// gam#3346: `X'WX` is covariant like `H`, so the lift keeps it beside `H`
    /// in the active frame instead of refusing the fit. Under this rectangular
    /// lift neither has a unique saved-frame form, and the saved-frame
    /// accessors say so rather than hand back an active-frame matrix.
    #[test]
    fn lift_to_saved_frame_keeps_the_weighted_gram_beside_the_hessian_3346() {
        let (mut fit, frame, _) = reduced_frame_fit();
        let gram = ndarray::array![[3.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 2.0]];
        if let Some(inference) = fit.inference.as_mut() {
            inference.weighted_gram = Some(gram.clone());
        }
        fit.lift_to_saved_frame(&frame)
            .expect("a weighted Gram stays in its active frame");
        assert_eq!(fit.inference.as_ref().unwrap().weighted_gram.as_ref(), Some(&gram));
        assert!(fit.saved_frame_weighted_gram().is_err());
        assert!(fit.saved_frame_penalized_hessian().is_err());
    }

    /// A fit on blocks of widths `[2, 1]` whose active frame is the saved frame.
    fn identity_frame_fit(
        beta: &Array1<f64>,
        hessian: &Array2<f64>,
        gram: &Array2<f64>,
    ) -> UnifiedFitResult {
        let mut parts = parts_with_inner_status(PirlsStatus::Converged);
        let block_betas = [
            beta.slice(ndarray::s![0..2]).to_owned(),
            beta.slice(ndarray::s![2..3]).to_owned(),
        ];
        parts.blocks[0].beta = block_betas[0].clone();
        parts.blocks.push(FittedBlock {
            beta: block_betas[1].clone(),
            role: BlockRole::Scale,
            edf: 1.0,
            lambdas: Array1::zeros(0),
        });
        parts.block_states = block_betas
            .iter()
            .map(|beta| gam_problem::ParameterBlockState {
                beta: beta.clone(),
                eta: Array1::zeros(3),
            })
            .collect();
        if let Some(inference) = parts.inference.as_mut() {
            inference.penalized_hessian =
                gam_problem::dispersion_cov::UnscaledPrecision::wrap(hessian.clone());
            inference.weighted_gram = Some(gram.clone());
        }
        parts.geometry = Some(FitGeometry {
            coefficient_gauge: gam_problem::Gauge::identity(&[2, 1]),
            penalized_hessian: hessian.clone().into(),
            constrained_posterior: None,
            working: None,
        });
        UnifiedFitResult::try_from_parts(parts).expect("identity-frame fixture assembles")
    }

    fn score_test_p_value(fit: &UnifiedFitResult) -> f64 {
        let hessian = fit.saved_frame_penalized_hessian().unwrap().unwrap();
        let gram = fit.saved_frame_weighted_gram().unwrap().unwrap();
        let beta = fit.beta_from_gauge_shift().unwrap();
        gam_terms::inference::smooth_score_test::smooth_score_test(
            gam_terms::inference::smooth_score_test::SmoothScoreTestInput {
                beta: beta.view(),
                penalized_hessian: &hessian,
                weighted_gram: &gram,
                coeff_range: 0..2,
                structural_penalties: &[Array2::eye(2)],
                scale: gam_terms::inference::smooth_score_test::ScoreTestScale::Known {
                    covariance_scale: 1.0,
                },
            },
        )
        .expect("the term is testable")
        .p_value
    }

    /// gam#3346 acceptance: under a square gauge (the mean-wiggle de-alias
    /// shape `[[I, −A], [0, I]]`) the smooth score test of a lifted fit equals
    /// the test of the same fit expressed in the saved frame, and under the
    /// identity gauge the saved-frame curvatures are the stored ones.
    #[test]
    fn square_gauge_score_test_matches_the_saved_frame_fit_3346() {
        let saved_gram = ndarray::array![[3.0, 0.5, 0.2], [0.5, 2.0, 0.1], [0.2, 0.1, 1.5]];
        let mut saved_hessian = saved_gram.clone();
        saved_hessian[[0, 0]] += 0.7;
        saved_hessian[[1, 1]] += 0.7;
        let saved_beta = ndarray::array![0.3, -0.2, 1.0];
        let saved_fit = identity_frame_fit(&saved_beta, &saved_hessian, &saved_gram);
        assert!(matches!(
            saved_fit.saved_frame_weighted_gram(),
            Ok(Some(std::borrow::Cow::Borrowed(gram))) if gram == &saved_gram
        ));
        assert!(matches!(
            saved_fit.saved_frame_penalized_hessian(),
            Ok(Some(std::borrow::Cow::Borrowed(hessian))) if hessian == &saved_hessian
        ));

        // `β_saved = M θ`, so the active fit carries `θ = M⁻¹ β_saved` and the
        // pulled-back curvatures `MᵀAM`.
        let de_alias = ndarray::array![[1.0, 0.0, -0.4], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]];
        let frame = gam_problem::Gauge::from_t(de_alias, &[2, 1], &[2, 1]);
        let active_beta = ndarray::array![0.3 + 0.4, -0.2 - 0.3, 1.0];
        let mut active_fit = identity_frame_fit(
            &active_beta,
            &frame.restrict_penalty(&saved_hessian),
            &frame.restrict_penalty(&saved_gram),
        );
        active_fit
            .lift_to_saved_frame(&frame)
            .expect("a square lift carries the weighted Gram");
        // `M` has entries of magnitude ≤ 1 and κ(M) < 2, so the lifted β and
        // the two QR solves are exact to a few ulps of the entries (≤ 3.7).
        for (a, b) in active_fit.beta.iter().zip(saved_beta.iter()) {
            assert!((a - b).abs() <= 1e-15, "{a} vs {b}");
        }
        let lifted_gram = active_fit.saved_frame_weighted_gram().unwrap().unwrap();
        let lifted_hessian = active_fit.saved_frame_penalized_hessian().unwrap().unwrap();
        for (lifted, saved) in [(&lifted_gram, &saved_gram), (&lifted_hessian, &saved_hessian)] {
            for (a, b) in lifted.iter().zip(saved.iter()) {
                assert!((a - b).abs() <= 1e-14, "{a} vs {b}");
            }
        }
        let saved_p = score_test_p_value(&saved_fit);
        let lifted_p = score_test_p_value(&active_fit);
        assert!(saved_p > 0.0 && saved_p < 1.0);
        assert!(
            (saved_p - lifted_p).abs() <= 1e-12,
            "the score test is frame-invariant: {saved_p} vs {lifted_p}"
        );
    }

    /// gam#3346 review: under an affine gauge `β = Tθ + a` the active fit
    /// carries the offset `X·a`, so the lifted `β̂` is the offset fit's
    /// coefficients plus `a`. The score test measures `β̂` from the shift and
    /// equals the test of the saved-frame fit with that offset; read off the
    /// uncentred `H·β̂` it would test a different score.
    #[test]
    fn affine_gauge_score_test_measures_beta_from_the_shift_3346() {
        let saved_gram = ndarray::array![[3.0, 0.5, 0.2], [0.5, 2.0, 0.1], [0.2, 0.1, 1.5]];
        let mut saved_hessian = saved_gram.clone();
        saved_hessian[[0, 0]] += 0.7;
        saved_hessian[[1, 1]] += 0.7;
        let offset_beta = ndarray::array![0.3, -0.2, 1.0];
        let offset_fit = identity_frame_fit(&offset_beta, &saved_hessian, &saved_gram);

        let de_alias = ndarray::array![[1.0, 0.0, -0.4], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]];
        let mut frame = gam_problem::Gauge::from_t(de_alias, &[2, 1], &[2, 1]);
        let shift = ndarray::array![0.25, -0.5, 0.75];
        frame.affine_shift = shift.clone();
        let active_beta = ndarray::array![0.3 + 0.4, -0.2 - 0.3, 1.0];
        let mut active_fit = identity_frame_fit(
            &active_beta,
            &frame.restrict_penalty(&saved_hessian),
            &frame.restrict_penalty(&saved_gram),
        );
        active_fit
            .lift_to_saved_frame(&frame)
            .expect("a square affine lift carries the weighted Gram");
        let expected_beta = &offset_beta + &shift;
        for (a, b) in active_fit.beta.iter().zip(expected_beta.iter()) {
            assert!((a - b).abs() <= 1e-15, "{a} vs {b}");
        }
        let centred = active_fit.beta_from_gauge_shift().unwrap();
        for (a, b) in centred.iter().zip(offset_beta.iter()) {
            assert!((a - b).abs() <= 1e-15, "{a} vs {b}");
        }

        let offset_p = score_test_p_value(&offset_fit);
        let lifted_p = score_test_p_value(&active_fit);
        assert!(offset_p > 0.0 && offset_p < 1.0);
        assert!(
            (offset_p - lifted_p).abs() <= 1e-12,
            "the score test keeps the shift as the null's offset: {offset_p} vs {lifted_p}"
        );
        let shifted_fit = identity_frame_fit(&expected_beta, &saved_hessian, &saved_gram);
        let uncentred_p = score_test_p_value(&shifted_fit);
        assert!(
            (offset_p - uncentred_p).abs() > 1e-3,
            "an uncentred score tests a different statistic: {offset_p} vs {uncentred_p}"
        );
    }
}

/// Exact coefficient-covariance definition (#2296).
///
/// This is the canonical vocabulary for "which covariance was used": the
/// conditional-on-λ̂ Bayesian `Vb`, the smoothing-parameter-corrected `Vp`
/// (`Vb + J·Var(ρ̂)·Jᵀ`), or the frequentist sandwich `Ve`. Any surface that
/// reports coefficient uncertainty must carry one of these values resolved
/// from the matrices it actually consumed, never from the caller's request.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CoefficientCovarianceDefinition {
    /// Conditional Bayesian covariance with smoothing parameters fixed (`Vb`).
    Conditional,
    /// Bayesian covariance including smoothing-parameter uncertainty (`Vp`).
    SmoothingCorrected,
    /// Frequentist sandwich covariance (`Ve`).
    FrequentistSandwich,
}

impl CoefficientCovarianceDefinition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conditional => "conditional",
            Self::SmoothingCorrected => "smoothing-corrected",
            Self::FrequentistSandwich => "frequentist-sandwich",
        }
    }
}

impl std::fmt::Display for CoefficientCovarianceDefinition {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A definition-consistent coefficient-uncertainty view (#2296): standard
/// errors and (optional) covariance from the SAME definition, tagged with
/// that definition. Produced by
/// [`UnifiedFitResult::display_coefficient_uncertainty`].
#[derive(Clone, Debug)]
pub struct DisplayCoefficientUncertainty<'a> {
    pub definition: CoefficientCovarianceDefinition,
    /// Derived from the covariance of the same definition (#2955).
    pub standard_errors: Array1<f64>,
    /// The covariance matrix of the same definition, when the fit persists
    /// it. `None` here never falls back to another definition's matrix.
    pub covariance: Option<&'a Array2<f64>>,
}

/// `sqrt(diag)` of a published covariance, by `se_from_covariance`'s rule.
///
/// [`UnifiedFitResult::try_from_parts`] and
/// [`UnifiedFitResult::add_coefficient_covariance_correction`] refuse a store
/// whose diagonal that rule refuses, so it cannot refuse a minted store. A store
/// mutated past those checks reads back `NaN` on a materially negative
/// coordinate, never a fabricated standard error.
fn standard_errors_of_published_covariance(covariance: &Array2<f64>) -> Array1<f64> {
    match gam_problem::dispersion_cov::se_from_covariance(covariance) {
        Ok(standard_errors) => standard_errors,
        Err(_) => covariance.diag().mapv(f64::sqrt),
    }
}

/// The identity of the data a fit was trained on: the columns that define its
/// experiment, by value (#4556 P3).
///
/// A model comparison is an evidence ratio only between two fits of ONE
/// experiment. Equal row counts, equal families and equal intercept-only
/// deviances are NECESSARY for that and none of them establishes it: two
/// different datasets can agree on all three. This is what establishes it, up
/// to the hash's collision probability — two fits whose fingerprints agree read
/// the same response values in the same order under the same weights.
///
/// EVERY column the fit's likelihood reads is absorbed, because each of them
/// can make two fits two experiments. A standard fit passes its response and
/// its prior weights — a weighted likelihood on one response under two
/// weightings is two experiments, since `−2ℓ` is weighted. A survival
/// transformation fit passes the entry time, the exit time, the event code and
/// the weights, because its likelihood reads all four and two datasets that
/// differ only in which rows were events are not one experiment. One rule for
/// one quantity: the column SET is the argument, so a new response shape adds a
/// call site and not a second identity to keep in step with this one.
///
/// The hash is [`gam_linalg::matrix::array2_bits_fingerprint`], the one value
/// identity this repository builds every fingerprint from. It absorbs the
/// column count and the row count before any value, so identities over
/// different column sets can never collide: a survival fit and a Gaussian fit
/// over the same numbers are different experiments and get different
/// identities.
///
/// This is not the warm-start key that `fit_survival_transformation_model`
/// hashes over the same four columns. That key answers "is this the same
/// PROBLEM", so it also absorbs the design, the offsets, the penalty blocks and
/// `ρ`; two models of one dataset share this identity and must not share that
/// key.
///
/// It is a value identity, not a provenance token: two runs over the same file
/// agree, a re-read of the same rows agrees, and a change to one observation
/// does not. An empty column set identifies nothing and no caller passes one.
pub fn training_response_fingerprint(columns: &[ndarray::ArrayView1<'_, f64>]) -> u64 {
    // One matrix over every column, with each column's own length absorbed as
    // the matrix's first row. A fit's columns have one length, and a view that
    // does not is not one of them; absorbing every length makes such a set its
    // own identity rather than a panic on a total function or a column silently
    // dropped, and it is what keeps a shorter column padded with zeros from
    // colliding with a genuine zero value.
    let rows = columns.iter().map(|column| column.len()).max().unwrap_or(0);
    let mut stacked = Array2::<f64>::zeros((rows + 1, columns.len().max(1)));
    for (index, column) in columns.iter().enumerate() {
        stacked[[0, index]] = column.len() as f64;
        for (row, value) in column.iter().enumerate() {
            stacked[[row + 1, index]] = *value;
        }
    }
    gam_linalg::matrix::array2_bits_fingerprint(&stacked)
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
    /// Number of original training rows / experimental units.
    ///
    /// This required wire field is the sole authority for sample-size-based
    /// post-fit inference. It deliberately cannot be reconstructed from
    /// optional IRLS evidence, a synthetic prediction grid, nonzero weights,
    /// or the number of response coordinates.
    training_sample_size: std::num::NonZeroUsize,
    /// Value identity of the training response and weights
    /// ([`fn@training_response_fingerprint`], #4556 P3), when the fit was built
    /// where those rows were in hand.
    ///
    /// `#[serde(default)]`, so a model saved before this field existed loads as
    /// "no established provenance" rather than failing, which is exactly what
    /// it is.
    #[serde(default)]
    training_response_fingerprint: Option<u64>,
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
    /// Complete REML/LAML objective value used for smoothing selection, when
    /// the fit has one.
    ///
    /// `None` is a typed statement that **no finite criterion value exists at
    /// this fit** — it is never "not recorded" and never a stand-in for zero.
    /// The one state that produces it is the exact-fit Gaussian boundary: when
    /// the fitted mean reproduces the response to floating-point resolution the
    /// profiled scale `φ̂` is zero, the restricted likelihood is unbounded, and
    /// every quantity derived from the criterion (`Summary.raw_reml_score`, the
    /// Tierney-Kadane comparable score, `compare_models`, Bayes factors) is
    /// undefined rather than large. Consumers that rank, compare, or normalize
    /// must refuse `None` by name; the field is private precisely so that a
    /// consumer cannot read a criterion the fit never had (#2595).
    ///
    /// This mirrors the decline already carried one field up:
    /// `log_likelihood = 0.0` tagged [`LogLikelihoodNormalization::UserProvided`]
    /// at the same boundary. Wire-compatible in both directions — a payload
    /// written before this field became optional carries a bare number and
    /// still deserializes as `Some`. The key itself stays mandatory: a payload
    /// that omits it is malformed, not a fit whose criterion is absent.
    reml_score: Option<f64>,
    /// Stable quadratic penalty term βᵀSβ, including any solver ridge quadratic.
    pub stable_penalty_term: f64,
    /// Public objective value reported for the fit. For REML/LAML fits this is
    /// the same complete objective as `reml_score`, not `-ℓ + penalty + reml_score`,
    /// and it is absent on exactly the fits whose criterion is absent.
    penalized_objective: Option<f64>,
    /// Whether the converged fit used a GPU execution path for its final inner solve.
    #[serde(default)]
    pub used_device: bool,
    /// Number of outer (smoothing parameter) iterations.
    pub outer_iterations: usize,
    /// Sealed proof that the inner and outer optimization layers converged.
    /// Private so struct literals and deserialization cannot bypass the checked
    /// constructor; inspect it through [`Self::convergence_evidence`].
    convergence: FitConvergenceEvidence,
    /// Final gradient norm of the outer optimization. `None` when no
    /// gradient was measured at termination — cache-hit short-circuit
    /// (the prior fit's converged ρ was loaded from disk), gradient-free
    /// solver, or a degenerate early-exit path where no outer ran.
    /// Fit existence is the authoritative convergence signal.
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
    pub block_states: Vec<gam_problem::ParameterBlockState>,
    /// Joint coefficient vector (first block for standard GAMs, concatenated for multi-block).
    #[serde(default)]
    pub beta: Array1<f64>,
    /// Maximum absolute linear predictor value at convergence.
    #[serde(default)]
    pub max_abs_eta: f64,
    /// Constraint KKT diagnostics (monotone-constrained fits).
    #[serde(default)]
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    /// Solver artifacts (e.g. cached PIRLS result for ALO).
    #[serde(default)]
    pub artifacts: FitArtifacts,
    /// Inner iterations of the final certified inner solve at the reported
    /// smoothing parameters: P-IRLS iterations on the standard path, blockwise
    /// cycles on the custom-family path.
    #[serde(default)]
    pub inner_cycles: usize,
    /// Number of outer REML cost-only evaluations the fit executed (each
    /// trust-region / line-search probe drives one, paying an inner P-IRLS
    /// solve). Diagnostic only — guards regressions in outer work (#1575) and
    /// is not part of the statistical contract. Zero for paths that do not run
    /// the standard external REML optimizer.
    #[serde(default)]
    pub outer_cost_evals: usize,
    /// Number of *actual* full-n inner P-IRLS solves the fit performed (the
    /// cache-missing solves across the seed-grid prepass, screening, multistart,
    /// and finalize). This is the true #1575 cost metric — distinct from, and
    /// typically ~2× larger than, `outer_cost_evals`, which counts outer
    /// requests including single-slot cache hits. Diagnostic only; not part of
    /// the statistical contract. Zero for paths that do not run the standard
    /// external REML optimizer.
    #[serde(default)]
    pub inner_pirls_solves: usize,
}

pub(crate) use gam_problem::ensure_finite_scalar_estimation;

fn validate_likelihood_scale_estimation(
    scale: LikelihoodScaleMetadata,
) -> Result<(), EstimationError> {
    match scale {
        LikelihoodScaleMetadata::ProfiledGaussian | LikelihoodScaleMetadata::Unspecified => Ok(()),
        LikelihoodScaleMetadata::FixedDispersion { phi }
        | LikelihoodScaleMetadata::EstimatedBetaPhi { phi }
        | LikelihoodScaleMetadata::FixedBetaPhi { phi }
        | LikelihoodScaleMetadata::EstimatedTweediePhi { phi }
        | LikelihoodScaleMetadata::EstimatedDispersion { phi } => {
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

pub(crate) use gam_problem::validate_all_finite_estimation;
pub use gam_problem::{ensure_finite_scalar, validate_all_finite};

impl FitGeometry {
    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        self.coefficient_gauge.validate().map_err(|reason| {
            EstimationError::InvalidInput(format!(
                "fit_result.geometry.coefficient_gauge is invalid: {reason}"
            ))
        })?;
        validate_all_finite_estimation(
            "fit_result.geometry.penalized_hessian",
            self.penalized_hessian.iter().copied(),
        )?;
        if let Some(constrained) = self.constrained_posterior.as_ref() {
            constrained
                .validate_for_dimension(self.coefficient_gauge.reduced_total())
                .map_err(|reason| {
                    EstimationError::InvalidInput(format!(
                        "fit_result.geometry.constrained_posterior is invalid: {reason}"
                    ))
                })?;
        }
        if let Some(working) = self.working.as_ref() {
            working.validate_numeric_finiteness()?;
        }
        Ok(())
    }
}

impl WorkingGeometry {
    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        if self.weights.len() != self.response.len() {
            return Err(EstimationError::InvalidInput(format!(
                "fit_result.geometry working vector length mismatch: weights={}, response={}",
                self.weights.len(),
                self.response.len(),
            )));
        }
        validate_all_finite_estimation(
            "fit_result.geometry.working.weights",
            self.weights.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.geometry.working.response",
            self.response.iter().copied(),
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
            "fit_result.penalized_hessian",
            self.penalized_hessian.iter().copied(),
        )?;
        if let Some(v) = self.factorized_standard_errors.as_ref() {
            validate_all_finite_estimation(
                "fit_result.factorized_standard_errors",
                v.iter().copied(),
            )?;
        }
        if let Some(factorized) = self.smoothing_correction_factorized.as_ref() {
            validate_all_finite_estimation(
                "fit_result.smoothing_correction_factorized.factor",
                factorized.factor.iter().copied(),
            )?;
            validate_all_finite_estimation(
                "fit_result.smoothing_correction_factorized.standard_errors",
                factorized.standard_errors.iter().copied(),
            )?;
        }
        // These three are INFERENCE-ONLY objects derived from `H⁻¹` at the
        // fitted mode, not the fitted mean coefficients. They go non-finite
        // when the curvature at THIS rho is singular or unstable, which is a
        // property of the trial point, so they refuse with the variant that
        // says so (#2593). The message text is unchanged.
        if let Some(v) = self.beta_covariance_frequentist.as_ref() {
            gam_problem::validate_all_finite_trial_point(
                "fit_result.beta_covariance_frequentist",
                v.iter().copied(),
            )?;
        }
        if let Some(v) = self.coefficient_influence.as_ref() {
            gam_problem::validate_all_finite_trial_point(
                "fit_result.coefficient_influence",
                v.iter().copied(),
            )?;
        }
        if let Some(subspace) = self.identified_subspace.as_ref() {
            validate_all_finite_estimation(
                "fit_result.identified_subspace.unidentified_basis",
                subspace.unidentified_basis.iter().copied(),
            )?;
        }
        if let Some(v) = self.weighted_gram.as_ref() {
            gam_problem::validate_all_finite_trial_point(
                "fit_result.weighted_gram",
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
        if let Some(v) = self.smoothing_correction_first_order.as_ref() {
            validate_all_finite_estimation(
                "fit_result.smoothing_correction_first_order",
                v.iter().copied(),
            )?;
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
    hessian.to_owned().cholesky(Side::Lower).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "{label} must be positive definite for HMC/NUTS whitening; Cholesky failed: {err:?}"
        ))
    })?;
    Ok(())
}

fn log_lambdas_match_lambdas(log_lambdas: &Array1<f64>, lambdas: &Array1<f64>) -> bool {
    if log_lambdas.len() != lambdas.len() {
        return false;
    }
    log_lambdas
        .iter()
        .zip(lambdas.iter())
        .all(
            |(&log_lam, &lam)| match gam_problem::checked_exp_log_strength(log_lam) {
                Ok(expected) => lam.to_bits() == expected.to_bits(),
                Err(_) => false,
            },
        )
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

/// The one explanation every ranking surface gives for an absent criterion.
///
/// Kept as a single constant so the summary, `evidence`, `compare_models` and
/// the Bayes-factor path cannot drift into describing the same state three
/// different ways (#2595).
pub const NO_CRITERION_AT_EXACT_FIT: &str = "this fit has no REML/LAML criterion value: the fitted mean reproduces the response to \
     floating-point resolution, so the profiled Gaussian scale is exactly zero and the \
     restricted likelihood is unbounded. Evidence, Bayes factors and cross-model comparison \
     are undefined for an exactly-interpolating fit; compare it on predictive accuracy \
     instead, or refit on data whose response is not an exact function of the design";

/// The one explanation for a fit that has a raw REML/LAML criterion but no
/// comparable one: the Tierney-Kadane normalizer needs the penalty null space,
/// and this fit family does not produce it (#2627).
pub const NO_COMPARABLE_CRITERION_WITHOUT_NULL_SPACE: &str = "this fit reports its raw REML/LAML criterion (raw_reml_score) but no comparable one: \
     the Tierney-Kadane normalizer that makes criteria comparable across fits needs the \
     penalty null-space dimension and log-determinant, and this fit family does not produce \
     them. Compare its raw criterion only against fits of the same family and penalty \
     structure, or compare on predictive accuracy";

/// The exact-fit Gaussian boundary, from the three quantities that define it.
///
/// A free function rather than a method so the ONE definition serves the
/// accessor on a built fit, the constructor gate (which runs before a `self`
/// exists), and any reconstruction that assembles those three quantities from a
/// persisted payload. Two copies of this predicate is precisely how a fit could
/// come to report a criterion the accessor then refuses to read.
pub fn is_zero_dispersion_boundary(
    likelihood_family: Option<&LikelihoodSpec>,
    likelihood_scale: LikelihoodScaleMetadata,
    standard_deviation: f64,
) -> bool {
    likelihood_family.is_some_and(|family| family.is_gaussian_identity())
        && matches!(likelihood_scale, LikelihoodScaleMetadata::ProfiledGaussian)
        && standard_deviation == 0.0
}

/// Refuse a fit result that violates its own consistency contract. These
/// checks describe the engine's assembly, not the caller's data, so they carry
/// [`EstimationError::FitResultInvariantViolated`] rather than `InvalidInput`
/// (#2937).
macro_rules! bail_fit_result_invariant {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err(EstimationError::FitResultInvariantViolated(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err(EstimationError::FitResultInvariantViolated($msg))
    };
}

/// [`UnifiedFitResult::search_seed_log_lambdas`] over a fit's `log λ` and its
/// certificate (#2954).
fn search_seed_from(
    mut seed: Array1<f64>,
    certificate: Option<&OuterCriterionCertificate>,
) -> Array1<f64> {
    let Some(certificate) = certificate else {
        return seed;
    };
    let Some(polish) = certificate.newton_polish.as_ref() else {
        return seed;
    };
    for fact in &certificate.railed_facts {
        let entry = polish.entry.get(fact.index).copied();
        if let (Some(value), Some(entry)) = (seed.get_mut(fact.index), entry)
            && entry.is_finite()
        {
            *value = entry;
        }
    }
    seed
}

impl UnifiedFitResult {
    /// Proof carried by every fitted model. Callers never need to re-check a
    /// convergence boolean; construction has already consumed and validated
    /// the inner and outer evidence.
    pub fn convergence_evidence(&self) -> &FitConvergenceEvidence {
        &self.convergence
    }

    /// The log smoothing strengths another search seeded from this fit starts
    /// at (#2954). A coordinate the certificate's Newton polish carried onto a
    /// box face, by a rail or a clamped step, is a fact about this fit's
    /// criterion, not a place to start another one: it seeds from the interior
    /// value the search handed the polish. Every other coordinate seeds from
    /// the fit's own `λ`.
    pub fn search_seed_log_lambdas(&self) -> Array1<f64> {
        search_seed_from(
            self.lambdas.mapv(f64::ln),
            self.artifacts.criterion_certificate.as_ref(),
        )
    }

    /// Number of original training rows / experimental units.
    pub fn training_sample_size(&self) -> usize {
        self.training_sample_size.get()
    }

    /// Value identity of the rows this fit was trained on, when it has one
    /// ([`fn@training_response_fingerprint`], #4556 P3). `None` is an absence of
    /// established provenance, not a claim about the data.
    pub fn training_response_fingerprint(&self) -> Option<u64> {
        self.training_response_fingerprint
    }

    /// The fit's REML/LAML criterion, or `None` when no finite criterion exists
    /// at this fit.
    ///
    /// `None` is not "unavailable"; it is the statement that the criterion is
    /// unbounded — see the field documentation. Callers that rank, compare, or
    /// normalize must propagate the absence, not substitute a number for it —
    /// [`NO_CRITERION_AT_EXACT_FIT`] is the one explanation to refuse with.
    ///
    /// The answer is decided by [`Self::at_zero_dispersion_boundary`] and not
    /// by the stored number alone. Payloads written before the criterion could
    /// be absent carry `0.0` at that boundary — the placeholder, not a
    /// criterion — so a legacy model loaded from disk gets the same honest
    /// answer as a fresh fit, without a migration pass over saved artifacts.
    pub fn reml_score(&self) -> Option<f64> {
        self.reml_score
            .filter(|_| !self.at_zero_dispersion_boundary())
    }

    /// [`Self::reml_score`] made comparable across models by the Tierney-Kadane
    /// normalizer over this fit's penalty null space
    /// ([`crate::topology_selector::comparable_reml_score`]).
    ///
    /// `Ok(None)` when [`Self::reml_score`] is `None`, since the normalizer is a
    /// correction to a criterion and a fit without one has no comparable score,
    /// and when the fit carries no null-space metadata
    /// ([`NO_COMPARABLE_CRITERION_WITHOUT_NULL_SPACE`]).
    pub fn comparable_reml_score(&self) -> Result<Option<f64>, String> {
        let Some(raw) = self.reml_score() else {
            return Ok(None);
        };
        crate::topology_selector::comparable_reml_score(
            raw,
            self.artifacts.null_space_dim.map(|dim| dim as f64),
            self.artifacts.null_space_logdet,
        )
    }

    /// `true` at the exact-fit Gaussian boundary: a profiled Gaussian scale
    /// estimated as exactly zero.
    ///
    /// This is the single state in which a fit has neither a normalized
    /// Lebesgue density nor a REML/LAML criterion — the fitted mean reproduces
    /// the response, so `φ̂ = 0` and both the full likelihood and the restricted
    /// likelihood are unbounded. It is DERIVED from the fit's own persisted
    /// family, scale metadata and `σ̂`, so a model loaded from disk reaches the
    /// same verdict as the live one, and no separate flag can drift from it.
    pub fn at_zero_dispersion_boundary(&self) -> bool {
        is_zero_dispersion_boundary(
            self.likelihood_family.as_ref(),
            self.likelihood_scale,
            self.standard_deviation,
        )
    }

    /// The fit's log-likelihood, or `None` when the fit declined to claim one.
    ///
    /// The decline is the same boundary as the criterion's: `log_likelihood`
    /// carries `0.0` under [`LogLikelihoodNormalization::UserProvided`] there,
    /// which states that no normalized density exists — a fact a consumer that
    /// reads the bare `f64` cannot see. Ranking on that zero is how an
    /// exactly-interpolating fit scores `−2·0 + 2·edf` in a conditional-AIC
    /// comparison and wins on nothing.
    pub fn reported_log_likelihood(&self) -> Option<f64> {
        (!self.at_zero_dispersion_boundary()).then_some(self.log_likelihood)
    }

    /// The log-likelihood at the fitted coefficient mode (gam#2921).
    ///
    /// A constrained custom-family fit publishes its truncated posterior mean as
    /// the coefficients, and [`Self::log_likelihood`] is evaluated there, so the
    /// returned model reproduces it. The mode's value is kept on the
    /// constrained-posterior geometry for statistics that belong to the mode.
    /// Every fit that publishes its mode returns `log_likelihood` itself.
    pub fn log_likelihood_at_mode(&self) -> f64 {
        self.geometry
            .as_ref()
            .and_then(|geometry| geometry.constrained_posterior.as_ref())
            .and_then(|constrained| constrained.mode_log_likelihood)
            .unwrap_or(self.log_likelihood)
    }

    /// Public objective value reported for the fit; absent exactly when
    /// [`Self::reml_score`] is absent.
    pub fn penalized_objective(&self) -> Option<f64> {
        self.penalized_objective
            .filter(|_| !self.at_zero_dispersion_boundary())
    }

    /// Replace the criterion with the value a later outer solve produced at
    /// this same fit.
    ///
    /// Sets both faces of the objective at once: the constructor requires them
    /// to be present or absent together, and two independent setters would let
    /// a caller satisfy one and not the other on an already-built result, where
    /// nothing re-validates.
    pub fn set_criterion(&mut self, value: Option<f64>) {
        self.reml_score = value;
        self.penalized_objective = value;
    }

    /// Shift the criterion by an additive constant — a response-rescaling
    /// Jacobian, a normalizer, any term that moves the objective without
    /// re-solving it.
    ///
    /// Absent stays absent: a constant added to a criterion that does not
    /// exist still does not exist, and silently materializing one here is the
    /// exact substitution this type was made optional to prevent.
    pub fn shift_criterion(&mut self, delta: f64) {
        self.reml_score = self.reml_score.map(|value| value + delta);
        self.penalized_objective = self.penalized_objective.map(|value| value + delta);
    }

    /// Denominator degrees of freedom for estimated-scale Wald/F references.
    ///
    /// Both in-process and persisted-model summaries call this method so the
    /// definition cannot drift between presentation surfaces.
    pub fn wald_residual_degrees_of_freedom(&self) -> Option<f64> {
        self.edf_total().and_then(|edf| {
            let residual_df = self.training_sample_size.get() as f64 - edf;
            (edf.is_finite() && residual_df.is_finite() && residual_df > 0.0).then_some(residual_df)
        })
    }

    /// Typed reason the saved coefficient vector is a constrained optimizer
    /// mode rather than the posterior mean required by the public estimand.
    pub fn posterior_moment_decline(
        &self,
    ) -> Option<&crate::constrained_posterior::ConePosteriorMomentDecline> {
        self.geometry
            .as_ref()
            .and_then(|geometry| geometry.constrained_posterior.as_ref())
            .and_then(crate::constrained_posterior::ConstrainedPosteriorGeometry::decline)
    }

    /// Refuse any operation that would label a constrained mode as a posterior
    /// mean. Keeping the converged low-level fit is valid; building a predictive
    /// model from an estimand whose moments were declined is not.
    pub fn require_posterior_mean(&self, operation: &str) -> Result<(), EstimationError> {
        if let Some(decline) = self.posterior_moment_decline() {
            return Err(EstimationError::InvalidInput(format!(
                "{operation} requires posterior-mean coefficients, but this converged constrained fit stores its optimizer mode under a typed posterior-moment decline: {}",
                decline.summary(),
            )));
        }
        Ok(())
    }

    pub fn try_from_parts(parts: UnifiedFitResultParts) -> Result<Self, EstimationError> {
        let convergence = FitConvergenceEvidence::try_from_parts(&parts)?;
        let UnifiedFitResultParts {
            blocks,
            training_sample_size,
            training_response_fingerprint,
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
            outer_converged: _,
            outer_gradient_norm,
            standard_deviation,
            covariance_conditional,
            covariance_corrected,
            inference,
            fitted_link,
            geometry,
            block_states,
            pirls_status: _,
            max_abs_eta,
            constraint_kkt,
            artifacts,
            inner_cycles,
        } = parts;
        let mut artifacts = artifacts;
        let no_smoothing_coordinate = has_no_smoothing_coordinate(&log_lambdas, &artifacts);
        let outer_iterations = if no_smoothing_coordinate {
            0
        } else {
            outer_iterations
        };
        let outer_gradient_norm = if no_smoothing_coordinate {
            // Keep the stored artifacts in the same semantic frame as the
            // sealed evidence. A zero-length gradient/certificate is vacuous,
            // not a measured stationary outer equation.
            artifacts.criterion_certificate = None;
            None
        } else {
            outer_gradient_norm
        };

        let training_sample_size =
            std::num::NonZeroUsize::new(training_sample_size).ok_or_else(|| {
                EstimationError::FitResultInvariantViolated(
                    "UnifiedFitResult training_sample_size must be positive".to_string(),
                )
            })?;
        if blocks.is_empty() {
            bail_fit_result_invariant!("UnifiedFitResult requires at least one coefficient block");
        }
        if log_lambdas.len() != lambdas.len() {
            bail_fit_result_invariant!(
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
            bail_fit_result_invariant!("UnifiedFitResult top-level lambdas must match block lambdas concatenated in block order"
                    .to_string(),);
        }
        validate_all_finite_estimation("fit_result.log_lambdas", log_lambdas.iter().copied())?;
        validate_all_finite_estimation("fit_result.lambdas", lambdas.iter().copied())?;
        if !log_lambdas_match_lambdas(&log_lambdas, &lambdas) {
            bail_fit_result_invariant!(
                "UnifiedFitResult log_lambdas must equal ln(lambdas) elementwise"
            );
        }
        validate_likelihood_scale_estimation(likelihood_scale)?;
        if let Some(spec) = likelihood_family.as_ref() {
            GlmLikelihoodSpec {
                spec: spec.clone(),
                scale: likelihood_scale,
            }
            .resolved_scale()
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
        }
        ensure_finite_scalar_estimation("fit_result.log_likelihood", log_likelihood)?;
        ensure_finite_scalar_estimation("fit_result.deviance", deviance)?;
        // A criterion that is PRESENT must still be finite: `None` is the only
        // legal way to report "no criterion", and `inf`/`NaN` remain rejected.
        if let Some(reml_score) = reml_score {
            ensure_finite_scalar_estimation("fit_result.reml_score", reml_score)?;
        }
        ensure_finite_scalar_estimation("fit_result.stable_penalty_term", stable_penalty_term)?;
        if let Some(penalized_objective) = penalized_objective {
            ensure_finite_scalar_estimation("fit_result.penalized_objective", penalized_objective)?;
        }
        // The two are the same objective under different names for every REML /
        // LAML fit, so one cannot exist without the other. Allowing the pair to
        // disagree on presence would let a consumer that reads only one of them
        // silently recover the value the other declined to state.
        if reml_score.is_none() != penalized_objective.is_none() {
            bail_fit_result_invariant!(
                "UnifiedFitResult reml_score and penalized_objective must be present or absent \
                 together; got reml_score={reml_score:?}, penalized_objective={penalized_objective:?}"
            );
        }
        // The exact-fit Gaussian boundary is not a place a criterion can exist:
        // `φ̂ = 0` makes the profiled restricted likelihood unbounded, so any
        // number offered here is a stand-in for `−∞`. Enforcing it at the ONE
        // constructor means no present or future route can reintroduce the
        // placeholder that #2595 traced `Summary.raw_reml_score = 0.0` to,
        // whether it arrives from a fast path, a saved payload, or a re-stamp.
        let at_boundary = is_zero_dispersion_boundary(
            likelihood_family.as_ref(),
            likelihood_scale,
            standard_deviation,
        );
        if at_boundary && let Some(reml_score) = reml_score {
            bail_fit_result_invariant!(
                "UnifiedFitResult reports a REML/LAML criterion {reml_score} at the exact-fit \
                 Gaussian boundary (profiled scale sigma_hat = 0), where the restricted \
                 likelihood is unbounded and no finite criterion exists; report `None`"
            );
        }
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
        // The published standard errors derive from these stores (#2955), so a
        // store whose diagonal `se_from_covariance` refuses is refused here, by
        // name, and the derived standard errors need no `Result`.
        for (label, covariance) in [
            ("conditional", covariance_conditional.as_ref()),
            ("smoothing-corrected", covariance_corrected.as_ref()),
        ] {
            if let Some(covariance) = covariance {
                gam_problem::dispersion_cov::se_from_covariance(covariance).map_err(|reason| {
                    EstimationError::InvalidInput(format!(
                        "UnifiedFitResult {label} covariance has an invalid diagonal: {reason}"
                    ))
                })?;
            }
        }
        if let Some(inf) = inference.as_ref() {
            inf.validate_numeric_finiteness()?;
        }
        if let Some(geom) = geometry.as_ref() {
            geom.validate_numeric_finiteness()?;
            if let Some(working) = geom.working.as_ref()
                && working.response.len() != training_sample_size.get()
            {
                bail_fit_result_invariant!(
                    "UnifiedFitResult working row count {} must match training_sample_size {}",
                    working.response.len(),
                    training_sample_size.get()
                );
            }
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
            bail_fit_result_invariant!(
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
            bail_fit_result_invariant!(
                "UnifiedFitResult corrected covariance shape mismatch: got {}x{}, expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                p,
                p
            );
        }
        let penalized_hessian_dim = if let Some(geom) = geometry.as_ref() {
            let gauge = &geom.coefficient_gauge;
            if gauge.n_blocks() != blocks.len() {
                bail_fit_result_invariant!(
                    "UnifiedFitResult geometry coefficient gauge block count mismatch: gauge={}, fitted blocks={}",
                    gauge.n_blocks(),
                    blocks.len(),
                );
            }
            for (block_index, (raw_width, block)) in gauge
                .raw_widths()
                .into_iter()
                .zip(blocks.iter())
                .enumerate()
            {
                if raw_width != block.beta.len() {
                    bail_fit_result_invariant!(
                        "UnifiedFitResult geometry coefficient gauge raw block {block_index} has width {raw_width}, expected saved beta width {}",
                        block.beta.len(),
                    );
                }
            }
            let active_dim = gauge.reduced_total();
            validate_dense_hessian_export(
                "UnifiedFitResult geometry active-coordinate penalized Hessian",
                &geom.penalized_hessian,
                active_dim,
            )?;
            active_dim
        } else {
            p
        };
        if let Some(inf) = inference.as_ref() {
            if !inf.edf_by_block.is_empty() && inf.edf_by_block.len() != lambdas.len() {
                bail_fit_result_invariant!(
                    "UnifiedFitResult EDF smoothing-parameter count mismatch: edf_by_block={}, lambdas={}",
                    inf.edf_by_block.len(),
                    lambdas.len()
                );
            }
            if !inf.penalty_block_trace.is_empty() && inf.penalty_block_trace.len() != lambdas.len()
            {
                bail_fit_result_invariant!(
                    "UnifiedFitResult EDF smoothing-parameter count mismatch: penalty_block_trace={}, lambdas={}",
                    inf.penalty_block_trace.len(),
                    lambdas.len()
                );
            }
            validate_dense_hessian_export(
                "UnifiedFitResult inference penalized Hessian",
                &inf.penalized_hessian,
                penalized_hessian_dim,
            )?;
            if let Some(se) = inf.factorized_standard_errors.as_ref() {
                if se.len() != p {
                    bail_fit_result_invariant!(
                        "UnifiedFitResult factorized standard error length mismatch: got {}, expected {}",
                        se.len(),
                        p
                    );
                }
                if let Some((index, value)) = se.iter().copied().enumerate().find(|&(_, v)| v < 0.0) {
                    bail_fit_result_invariant!(
                        "UnifiedFitResult factorized standard error {index} is negative: {value:?}"
                    );
                }
                // The standard errors of a published covariance derive from it;
                // a second diagonal beside it is a copy that can go stale (#2955).
                if covariance_conditional.is_some() {
                    bail_fit_result_invariant!(
                        "UnifiedFitResult carries factorized standard errors beside a conditional \
                         covariance; the standard errors of a published covariance derive from it"
                    );
                }
            }
            if let Some(factorized) = inf.smoothing_correction_factorized.as_ref() {
                // #3283: the factorized branch's correction is solved with the
                // standard errors beside it; a fit that publishes a covariance
                // carries its correction and corrected covariance instead.
                if inf.factorized_standard_errors.is_none() || covariance_corrected.is_some() {
                    bail_fit_result_invariant!(
                        "UnifiedFitResult carries a factorized smoothing correction beside a \
                         published covariance; it exists only beside factorized standard errors"
                    );
                }
                if factorized.factor.nrows() != p || factorized.standard_errors.len() != p {
                    bail_fit_result_invariant!(
                        "UnifiedFitResult factorized smoothing correction shape mismatch: factor \
                         {}x{}, standard errors {}, expected {} coefficients",
                        factorized.factor.nrows(),
                        factorized.factor.ncols(),
                        factorized.standard_errors.len(),
                        p
                    );
                }
                if let Some((index, value)) = factorized
                    .standard_errors
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|&(_, v)| v < 0.0)
                {
                    bail_fit_result_invariant!(
                        "UnifiedFitResult factorized corrected standard error {index} is negative: \
                         {value:?}"
                    );
                }
            }
            if let Some(cov) = inf.beta_covariance_frequentist.as_ref()
                && (cov.nrows() != p || cov.ncols() != p)
            {
                bail_fit_result_invariant!(
                    "UnifiedFitResult frequentist covariance shape mismatch: got {}x{}, expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    p,
                    p
                );
            }
            // `X'WX` is a covariant quadratic form exactly like `H`, so it lives
            // in `H`'s active frame (gam#3346).
            if let Some(gram) = inf.weighted_gram.as_ref()
                && gram.dim() != (penalized_hessian_dim, penalized_hessian_dim)
            {
                bail_fit_result_invariant!(
                    "UnifiedFitResult weighted Gram shape mismatch: got {}x{}, expected the \
                     penalized Hessian's active frame {}x{}",
                    gram.nrows(),
                    gram.ncols(),
                    penalized_hessian_dim,
                    penalized_hessian_dim
                );
            }
            if let Some(f_mat) = inf.coefficient_influence.as_ref()
                && (f_mat.nrows() != p || f_mat.ncols() != p)
            {
                bail_fit_result_invariant!(
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
                bail_fit_result_invariant!(
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
                bail_fit_result_invariant!(
                    "UnifiedFitResult reparam_qs shape mismatch: got {}x{}, expected {}x{}",
                    qs.nrows(),
                    qs.ncols(),
                    p,
                    p
                );
            }
        }
        if let Some(geom) = geometry.as_ref() {
            if let Some(inf) = inference.as_ref() {
                if geom.penalized_hessian != inf.penalized_hessian {
                    bail_fit_result_invariant!("UnifiedFitResult geometry penalized Hessian must match inference.penalized_hessian"
                            .to_string(),);
                }
            }
        }
        if !block_states.is_empty() && block_states.len() != blocks.len() {
            bail_fit_result_invariant!(
                "UnifiedFitResult block state count mismatch: blocks={}, block_states={}",
                blocks.len(),
                block_states.len()
            );
        }

        Ok(Self {
            blocks,
            training_sample_size,
            training_response_fingerprint,
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
            convergence,
            outer_gradient_norm,
            standard_deviation,
            covariance_conditional,
            covariance_corrected,
            inference,
            fitted_link,
            geometry,
            block_states,
            beta,
            max_abs_eta,
            constraint_kkt,
            artifacts,
            inner_cycles,
            // Populated post-construction by the external REML fit path
            // (`fit.rs`) from the optimizer's outer cost-eval counter. The
            // parts builder does not carry it, so it defaults to 0 here.
            outer_cost_evals: 0,
            // Likewise populated post-construction from the optimizer's inner
            // P-IRLS solve counter (#1575); defaults to 0 here.
            inner_pirls_solves: 0,
        })
    }
    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        if self.outer_iterations != self.convergence.outer_iterations() {
            bail_fit_result_invariant!(
                "UnifiedFitResult outer iteration count does not match its sealed convergence evidence"
            );
        }
        let expected_beta = flatten_block_betas(&self.blocks);
        if self.beta != expected_beta {
            bail_fit_result_invariant!("UnifiedFitResult decoded beta must match coefficient blocks concatenated in block order"
                    .to_string(),);
        }
        let reconstructed = Self::try_from_parts(UnifiedFitResultParts {
            blocks: self.blocks.clone(),
            training_sample_size: self.training_sample_size.get(),
            training_response_fingerprint: self.training_response_fingerprint,
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
            outer_converged: true,
            outer_gradient_norm: self.outer_gradient_norm,
            standard_deviation: self.standard_deviation,
            covariance_conditional: self.covariance_conditional.clone(),
            covariance_corrected: self.covariance_corrected.clone(),
            inference: self.inference.clone(),
            fitted_link: self.fitted_link.clone(),
            geometry: self.geometry.clone(),
            block_states: self.block_states.clone(),
            pirls_status: self.convergence.inner_status(),
            max_abs_eta: self.max_abs_eta,
            constraint_kkt: self.constraint_kkt.clone(),
            artifacts: self.artifacts.clone(),
            inner_cycles: self.inner_cycles,
        })?;
        if self.convergence.outer_certificate().is_some()
            != reconstructed.convergence.outer_certificate().is_some()
        {
            bail_fit_result_invariant!(
                "UnifiedFitResult convergence evidence kind does not match its smoothing-coordinate geometry"
            );
        }
        Ok(())
    }
}

impl UnifiedFitResult {
    /// Get the conditional Bayesian covariance matrix (`Vb`) in the saved/raw
    /// coefficient frame, if available.
    ///
    /// Contract: for an active geometry gauge `β = Tθ + a`,
    /// `Vb_raw = T H_active^{-1} Tᵀ * phi`. For an identity gauge this
    /// reduces to `H^{-1} * phi`. This is the Wood/mgcv `Vb`
    /// (Bayesian/conditional) covariance.
    ///
    /// # Which of the two published covariances to use
    ///
    /// This one treats `λ̂` as KNOWN. It is the right object when the
    /// smoothing parameters are fixed by the caller rather than estimated, and
    /// it is the reference against which
    /// [`Self::beta_covariance_corrected`] is judged: for a Gaussian identity
    /// fit with `W = I` the trace identity `E_x[xᵀVb x] = φ·edf/n` pins its
    /// size exactly, with no Monte Carlo and no truth involved.
    ///
    /// It is NOT the same as the frequentist sampling covariance `Vf`: they
    /// differ by `Vb − Vf = φ·H⁻¹SH⁻¹ ⪰ 0`, the smoothing bias term, which is
    /// exactly why a Bayesian interval built from `Vb` attains its nominal
    /// frequentist coverage across the function (Nychka 1988, Marra & Wood
    /// 2012) while an interval built from `Vf` does not.
    ///
    /// When `λ̂` was ESTIMATED — the default — the interval a user wants is
    /// [`Self::beta_covariance_corrected`], which additionally propagates the
    /// uncertainty in `λ̂` itself. `predict()` defaults to that one.
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

    /// The weighted Gram `X'WX = H − S(λ)` in the saved coefficient frame, the
    /// frame of `beta`, the covariances and the smoothing corrections — the
    /// matrix the Wood–Pya–Säfken corrected-EDF correction pairs with the
    /// smoothing-parameter uncertainty covariance (issue #1027).
    ///
    /// `X'WX` is stored beside the penalized Hessian in its active frame, so
    /// it is pushed forward through the coefficient gauge exactly like
    /// [`Self::saved_frame_penalized_hessian`] (gam#3346). `Ok(None)` when the
    /// fit carries no Gram; `Err` when it does but the gauge is rectangular,
    /// so the Gram has no unique saved-frame form.
    pub fn saved_frame_weighted_gram(
        &self,
    ) -> Result<Option<std::borrow::Cow<'_, Array2<f64>>>, String> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.weighted_gram.as_ref())
            .map(|gram| self.active_quadratic_form_in_saved_frame(gram))
            .transpose()
    }

    /// The penalized Hessian in the saved coefficient frame. The stored
    /// Hessian ([`Self::penalized_hessian`]) lives in the active frame of the
    /// coefficient gauge; under a square gauge `β = Tθ + a` its saved-frame
    /// form is `T⁻ᵀ H_θ T⁻¹`, and under a rectangular one it has none, which
    /// is an `Err` rather than a Hessian of the wrong frame (gam#3346).
    pub fn saved_frame_penalized_hessian(
        &self,
    ) -> Result<Option<std::borrow::Cow<'_, Array2<f64>>>, String> {
        self.penalized_hessian()
            .map(|hessian| self.active_quadratic_form_in_saved_frame(hessian))
            .transpose()
    }

    /// `β̂ − a`: the saved-frame coefficients measured from the gauge's affine
    /// shift `a` of `β = Tθ + a`.
    ///
    /// The active penalty `θᵀS_θθ` is centred at `a` in the saved frame,
    /// `(β − a)ᵀ T⁻ᵀS_θT⁻¹ (β − a)`, so stationarity in θ pushes forward to
    /// `H_saved·(β̂ − a) = X_savedᵀW(z − X_saved·a)`: a score read off `H·β`
    /// must read it off `H·(β̂ − a)`, with the shift as the known offset every
    /// `τ = 0` null keeps (gam#3346).
    pub fn beta_from_gauge_shift(&self) -> Result<std::borrow::Cow<'_, Array1<f64>>, String> {
        match self.geometry.as_ref() {
            Some(geometry)
                if geometry
                    .coefficient_gauge
                    .affine_shift
                    .iter()
                    .any(|&shift| shift != 0.0) =>
            {
                let shift = &geometry.coefficient_gauge.affine_shift;
                if shift.len() != self.beta.len() {
                    return Err(format!(
                        "the coefficient gauge's affine shift has {} coordinates but beta has {}",
                        shift.len(),
                        self.beta.len()
                    ));
                }
                Ok(std::borrow::Cow::Owned(&self.beta - shift))
            }
            _ => Ok(std::borrow::Cow::Borrowed(&self.beta)),
        }
    }

    fn active_quadratic_form_in_saved_frame<'a>(
        &self,
        form: &'a Array2<f64>,
    ) -> Result<std::borrow::Cow<'a, Array2<f64>>, String> {
        match self.geometry.as_ref() {
            Some(geometry) if !geometry.coefficient_gauge.is_identity() => geometry
                .coefficient_gauge
                .push_forward_quadratic_form(form)
                .map(std::borrow::Cow::Owned),
            _ => Ok(std::borrow::Cow::Borrowed(form)),
        }
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
    /// A cached inference dispersion is accepted only when it agrees exactly
    /// with the scale reconstructed from the family contract. Families without
    /// a scalar response scale return an error instead of adopting a fictitious
    /// unit dispersion.
    pub fn dispersion_phi(&self) -> Result<f64, EstimationError> {
        let spec = self.likelihood_family.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "this fit has no engine-level family and therefore no scalar response dispersion"
                    .to_string(),
            )
        })?;
        let glm = GlmLikelihoodSpec {
            spec: spec.clone(),
            scale: self.likelihood_scale,
        };
        let profiled_standard_deviation = matches!(
            glm.resolved_scale()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
            gam_problem::ResolvedLikelihoodScale::ProfiledGaussian
        )
        .then_some(self.standard_deviation);
        let resolved = dispersion_from_likelihood(&glm, profiled_standard_deviation)?;
        if let Some(cached) = self.dispersion()
            && cached != resolved
        {
            return Err(EstimationError::InvalidInput(format!(
                "cached inference dispersion {:?} disagrees with family-resolved dispersion {:?}",
                cached, resolved
            )));
        }
        Ok(resolved.phi())
    }

    /// [`Self::dispersion_phi`] when the fit's scale contract has a scalar
    /// response dispersion, and `None` exactly when it has none: a fit with no
    /// engine-level family (a custom family), or a family whose resolved scale
    /// is [`gam_problem::ResolvedLikelihoodScale::Unspecified`] (Royston-Parmar
    /// survival). The answer is read from the resolved scale, so every other
    /// dispersion failure is still an error (gam#3297).
    pub fn scalar_dispersion_phi(&self) -> Result<Option<f64>, EstimationError> {
        let Some(spec) = self.likelihood_family.as_ref() else {
            return Ok(None);
        };
        let glm = GlmLikelihoodSpec {
            spec: spec.clone(),
            scale: self.likelihood_scale,
        };
        match glm
            .resolved_scale()
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?
        {
            gam_problem::ResolvedLikelihoodScale::Unspecified => Ok(None),
            _ => self.dispersion_phi().map(Some),
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
    /// Negative-Binomial, Poisson, Binomial) — see #679. A fit with no
    /// engine-level family is a custom-family fit whose objective is its
    /// complete negative log-likelihood — every scale it has is a coefficient —
    /// so its Laplace posterior precision is the penalized Hessian itself and
    /// the scale is `1.0`. Refusing it here left a saved custom-family model
    /// (survival marginal-slope, fitted through the library route) unable to
    /// reconstruct the covariance the load gate asks for (gam#2765).
    pub fn coefficient_covariance_scale(&self) -> Result<f64, EstimationError> {
        match &self.likelihood_family {
            Some(spec) => {
                let glm = GlmLikelihoodSpec {
                    spec: spec.clone(),
                    scale: self.likelihood_scale.clone(),
                };
                let profiled_standard_deviation = matches!(
                    glm.resolved_scale()
                        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
                    gam_problem::ResolvedLikelihoodScale::ProfiledGaussian
                )
                .then_some(self.standard_deviation);
                let dispersion =
                    dispersion_from_likelihood(&glm, profiled_standard_deviation)?;
                glm.coefficient_covariance_scale(dispersion.phi())
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))
            }
            None => Ok(1.0),
        }
    }

    /// Whether this fit selected any smoothing coordinate, per-block or joint.
    ///
    /// `lambdas` holds only the per-block precisions. A joint-penalty family
    /// carries its smoothing coordinates in `artifacts.joint_log_lambdas` and
    /// leaves `lambdas` empty (the multinomial per-class carrier), so
    /// `lambdas.is_empty()` does not answer this question (#2898).
    pub fn has_smoothing_coordinate(&self) -> bool {
        !has_no_smoothing_coordinate(&self.log_lambdas, &self.artifacts)
    }

    /// Get the smoothing-parameter-corrected beta covariance (`Vp`) if available.
    ///
    /// Wood/mgcv name for the smoothing-parameter-corrected covariance `Vp`.
    /// When there are no smoothing coordinates, `Var(rho)` has dimension zero
    /// and the correction `J Var(rho) Jᵀ` is identically zero. In that exact
    /// case the persisted conditional covariance is already `Vp`; requiring a
    /// duplicate corrected matrix would make ordinary parametric models lose
    /// posterior-mean intervals after serialization.
    ///
    /// # Which of the two published covariances to use
    ///
    /// **This one, whenever the smoothing parameters were estimated** — which
    /// is the default, and what `predict()` uses. It is
    /// [`Self::beta_covariance`] plus the propagated uncertainty in `λ̂`
    /// itself, by the law of total covariance
    ///
    /// ```text
    ///     Var(β|y) = E_ρ[φ·H(ρ)⁻¹] + Cov_ρ[β̂(ρ)],
    /// ```
    ///
    /// evaluated to first order at `ρ̂` (Wood, Pya & Säfken 2016):
    /// `Vp = Vβ + J V_ρ Jᵀ`, with `J = ∂β̂/∂ρ` from the implicit function
    /// theorem and `V_ρ` the inverse outer Hessian on its identified subspace.
    /// Interval calibration is checked against repeated-data coverage. Use
    /// [`Self::beta_covariance`] instead only when `λ` is fixed by the caller,
    /// or when you specifically want the conditional-on-`λ̂` object.
    ///
    /// The smoothing contribution is small where the outer criterion is sharply
    /// determined and can be large where it is broad.
    pub fn beta_covariance_corrected(&self) -> Option<&Array2<f64>> {
        self.covariance_corrected.as_ref().or_else(|| {
            has_no_smoothing_coordinate(&self.log_lambdas, &self.artifacts)
                .then(|| self.beta_covariance())
                .flatten()
        })
    }

    /// Conditional standard errors: `sqrt(diag)` of [`Self::beta_covariance`],
    /// the one store (#2955), or the factorized branch's solved diagonal when
    /// the fit publishes no covariance (#2960).
    pub fn beta_standard_errors(&self) -> Option<Array1<f64>> {
        match self.beta_covariance() {
            Some(covariance) => Some(standard_errors_of_published_covariance(covariance)),
            None => self
                .inference
                .as_ref()
                .and_then(|inf| inf.factorized_standard_errors.clone()),
        }
    }

    /// Smoothing-corrected standard errors: `sqrt(diag)` of
    /// [`Self::beta_covariance_corrected`], with its exact identity form for a
    /// fit that has no smoothing coordinate, whose standard errors are then the
    /// conditional ones.
    pub fn beta_standard_errors_corrected(&self) -> Option<Array1<f64>> {
        match self.covariance_corrected.as_ref() {
            Some(covariance) => Some(standard_errors_of_published_covariance(covariance)),
            None => match self.smoothing_correction_factorized() {
                // The factorized branch's corrected standard errors (#3283).
                Some(factorized) => Some(factorized.standard_errors.clone()),
                None => has_no_smoothing_coordinate(&self.log_lambdas, &self.artifacts)
                    .then(|| self.beta_standard_errors())
                    .flatten(),
            },
        }
    }

    /// The smoothing correction of a fit whose inference stayed factorized:
    /// its square-root factor `B` (`C = B·Bᵀ`) in the frame of the published
    /// coefficients, and the corrected standard errors (#3283). A fit that
    /// publishes a covariance returns `None` here and carries
    /// [`Self::beta_covariance_corrected`] instead.
    pub fn smoothing_correction_factorized(&self) -> Option<&FactorizedSmoothingCorrection> {
        self.inference
            .as_ref()
            .and_then(|inference| inference.smoothing_correction_factorized.as_ref())
    }

    /// Add one correction matrix to the published conditional and corrected
    /// covariances, the only store of each (#2955), whose standard errors then
    /// derive from the corrected matrices.
    ///
    /// A correction added to one copy of a covariance and not another (gam#2943:
    /// the marginal-slope Murphy–Topel generated-regressor correction) produced
    /// fits that every saved-model load refused, and published uncorrected
    /// standard errors in the meantime; with one store there is no other copy.
    /// The frequentist covariance and the smoothing-correction matrix are
    /// separate estimators and stay as they are. A fit that publishes standard
    /// errors without a covariance is refused, because no correction can reach
    /// a diagonal alone.
    pub fn add_coefficient_covariance_correction(
        &mut self,
        correction: &Array2<f64>,
    ) -> Result<(), EstimationError> {
        for (label, covariance) in [
            ("conditional", self.covariance_conditional.as_ref()),
            ("corrected", self.covariance_corrected.as_ref()),
        ] {
            if let Some(covariance) = covariance
                && covariance.dim() != correction.dim()
            {
                crate::bail_invalid_estim!(
                    "covariance correction of shape {:?} does not match the {label} covariance of shape {:?}",
                    correction.dim(),
                    covariance.dim()
                );
            }
        }
        if self
            .inference
            .as_ref()
            .is_some_and(|inf| inf.factorized_standard_errors.is_some())
        {
            crate::bail_invalid_estim!(
                "a covariance correction cannot reach standard errors published without their covariance"
            );
        }
        // Both corrected matrices are formed and judged before either is assigned,
        // so a refused correction leaves the fit as it was (gam-2929).
        let conditional = self
            .covariance_conditional
            .as_ref()
            .map(|covariance| covariance + correction);
        let corrected = self
            .covariance_corrected
            .as_ref()
            .map(|covariance| covariance + correction);
        for (label, covariance) in [
            ("conditional", conditional.as_ref()),
            ("smoothing-corrected", corrected.as_ref()),
        ] {
            if let Some(covariance) = covariance {
                gam_problem::dispersion_cov::se_from_covariance(covariance).map_err(|reason| {
                    EstimationError::InvalidInput(format!(
                        "corrected {label} covariance has an invalid diagonal: {reason}"
                    ))
                })?;
            }
        }
        self.covariance_conditional = conditional;
        self.covariance_corrected = corrected;
        Ok(())
    }

    /// Carry the whole fit from the solver's coefficient frame θ to the saved
    /// frame `β = J·θ + a` through one exact lift `frame` (gam#3021).
    ///
    /// A family that fits on a reparameterized design (`X_fit = X_saved·J`)
    /// reports in the saved frame, and every coefficient-indexed quantity moves
    /// together or the fit describes two models at once:
    ///
    /// * the block coefficients, their block states' copies and the flat β are
    ///   lifted by `J`; the linear predictors are unchanged, `X_saved·Jθ = X_fit·θ`;
    /// * the covariances (conditional, corrected, frequentist) and both
    ///   smoothing-parameter corrections push forward as `JΣJᵀ`;
    /// * the penalized Hessian, the weighted Gram `XᵀWX` and the identified
    ///   subspace are covariant and stay in their active frame, whose
    ///   coefficient gauge becomes `J∘gauge`; a consumer that pairs them with
    ///   saved-frame objects reads them through
    ///   [`Self::saved_frame_penalized_hessian`] and
    ///   [`Self::saved_frame_weighted_gram`] (gam#3346).
    ///
    /// A quantity with no unique lift through a non-square `J` is refused, not
    /// dropped: standard errors published without their covariance, the
    /// influence matrix `H⁻¹XᵀWX`, and a stored reparameterization basis. The fit
    /// is rebuilt through the constructor's invariants before it replaces
    /// `self`, so a refused lift leaves the fit as it was.
    pub fn lift_to_saved_frame(
        &mut self,
        frame: &gam_problem::gauge::Gauge,
    ) -> Result<(), EstimationError> {
        frame.validate().map_err(|reason| {
            EstimationError::InvalidInput(format!("saved-frame lift is invalid: {reason}"))
        })?;
        let active_widths: Vec<usize> = self.blocks.iter().map(|block| block.beta.len()).collect();
        let frame_active_widths: Vec<usize> = frame
            .block_starts_reduced
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect();
        if active_widths != frame_active_widths {
            crate::bail_invalid_estim!(
                "saved-frame lift expects fitted block widths {frame_active_widths:?}, the fit has {active_widths:?}"
            );
        }
        if self.block_states.len() != self.blocks.len() {
            crate::bail_invalid_estim!(
                "saved-frame lift needs one block state per fitted block, got {} for {}",
                self.block_states.len(),
                self.blocks.len()
            );
        }
        if self.beta != flatten_block_betas(&self.blocks) {
            crate::bail_invalid_estim!(
                "saved-frame lift: the flat coefficients disagree with the fitted blocks before the lift"
            );
        }
        let active_total = frame.reduced_total();
        let push_forward = |matrix: &Array2<f64>, label: &str| -> Result<Array2<f64>, EstimationError> {
            if matrix.dim() != (active_total, active_total) {
                crate::bail_invalid_estim!(
                    "saved-frame lift: the {label} is {}x{}, the solver frame is {active_total}x{active_total}",
                    matrix.nrows(),
                    matrix.ncols()
                );
            }
            Ok(frame.lift_covariance(matrix))
        };

        let mut lifted = self.clone();
        let block_betas: Vec<Array1<f64>> = self.blocks.iter().map(|b| b.beta.clone()).collect();
        for (block, beta) in lifted.blocks.iter_mut().zip(frame.lift_block_betas(&block_betas)) {
            block.beta = beta;
        }
        let state_betas: Vec<Array1<f64>> =
            self.block_states.iter().map(|s| s.beta.clone()).collect();
        if state_betas.iter().map(Array1::len).ne(active_widths.iter().copied()) {
            crate::bail_invalid_estim!(
                "saved-frame lift: the block states are not at the fitted block widths {active_widths:?}"
            );
        }
        for (state, beta) in lifted
            .block_states
            .iter_mut()
            .zip(frame.lift_block_betas(&state_betas))
        {
            state.beta = beta;
        }
        lifted.beta = flatten_block_betas(&lifted.blocks);
        lifted.covariance_conditional = self
            .covariance_conditional
            .as_ref()
            .map(|v| push_forward(v, "conditional covariance"))
            .transpose()?;
        lifted.covariance_corrected = self
            .covariance_corrected
            .as_ref()
            .map(|v| push_forward(v, "corrected covariance"))
            .transpose()?;
        if let Some(inference) = lifted.inference.as_mut() {
            for (present, label) in [
                (
                    inference.factorized_standard_errors.is_some(),
                    "standard errors published without their covariance",
                ),
                (
                    inference.coefficient_influence.is_some(),
                    "coefficient influence matrix",
                ),
                (inference.reparam_qs.is_some(), "reparameterization basis"),
            ] {
                if present {
                    crate::bail_invalid_estim!(
                        "saved-frame lift: the fit carries a {label}, which has no unique lift to the saved frame"
                    );
                }
            }
            for (slot, label) in [
                (
                    &mut inference.beta_covariance_frequentist,
                    "frequentist covariance",
                ),
                (&mut inference.smoothing_correction, "smoothing correction"),
                (
                    &mut inference.smoothing_correction_first_order,
                    "first-order smoothing correction",
                ),
            ] {
                if let Some(matrix) = slot.take() {
                    *slot = Some(push_forward(&matrix, label)?);
                }
            }
        }
        match lifted.geometry.as_mut() {
            Some(geometry) => {
                geometry.coefficient_gauge =
                    geometry.coefficient_gauge.left_compose(frame).map_err(|reason| {
                        EstimationError::InvalidInput(format!(
                            "saved-frame lift cannot compose with the active geometry: {reason}"
                        ))
                    })?;
            }
            None if lifted.inference.is_some() => {
                crate::bail_invalid_estim!(
                    "saved-frame lift: a penalized Hessian without its coefficient geometry is in the \
                     solver frame and has no pushforward to the saved frame"
                );
            }
            None => {}
        }
        lifted.validate_numeric_finiteness()?;
        *self = lifted;
        Ok(())
    }

    /// Corrected-preferred, definition-consistent coefficient uncertainty for
    /// summary/report display surfaces (#2296).
    ///
    /// Returns the smoothing-corrected standard errors (with the corrected
    /// covariance, when persisted) if the fit carries them, else the
    /// conditional pair. Standard errors and covariance are NEVER mixed
    /// across definitions: if the preferred definition has SEs but no matrix,
    /// the matrix slot is `None` rather than a different definition's matrix.
    /// The returned [`CoefficientCovarianceDefinition`] names what was
    /// actually selected so presenters serialize result-owned provenance —
    /// a display policy or request is never evidence of what was used.
    /// The covariance definition this fit PUBLISHES, and therefore the one
    /// every default uncertainty surface uses when the caller names none
    /// (gam#2779): smoothing-corrected whenever the fit carries that matrix,
    /// or its factorized form on a fit whose inference stayed factorized
    /// (#3283) — or its exact identity form for a fit with no smoothing coordinates,
    /// where the correction is the zero matrix — and conditional otherwise.
    /// A fit certified at an infinite-smoothing rail has a typed-unavailable
    /// correction and publishes conditional, which is what `summary()` prices
    /// its standard errors from; `predict`, `predict_conformal`, `diagnose`
    /// and `partial_dependence` resolve their default through this same
    /// method so one fitted object tells one story. An EXPLICIT request for
    /// the corrected definition is a requirement, not a policy, and still
    /// refuses when the matrix is absent.
    pub fn published_covariance_mode(&self) -> InferenceCovarianceMode {
        if self.beta_covariance_corrected().is_some()
            || self.smoothing_correction_factorized().is_some()
            || has_no_smoothing_coordinate(&self.log_lambdas, &self.artifacts)
        {
            InferenceCovarianceMode::SmoothingCorrected
        } else {
            InferenceCovarianceMode::Conditional
        }
    }

    pub fn display_coefficient_uncertainty(&self) -> Option<DisplayCoefficientUncertainty<'_>> {
        if let Some(standard_errors) = self.beta_standard_errors_corrected() {
            return Some(DisplayCoefficientUncertainty {
                definition: CoefficientCovarianceDefinition::SmoothingCorrected,
                standard_errors,
                covariance: self.beta_covariance_corrected(),
            });
        }
        self.beta_standard_errors()
            .map(|standard_errors| DisplayCoefficientUncertainty {
                definition: CoefficientCovarianceDefinition::Conditional,
                standard_errors,
                covariance: self.beta_covariance(),
            })
    }

    /// Get the penalized Hessian if available.
    ///
    /// The matrix is in the active geometry coordinate frame when
    /// [`Self::geometry`] is present, so it may be smaller than the saved beta
    /// vector. Pair it with `geometry.coefficient_gauge`; only an identity gauge
    /// makes this a saved/raw-coordinate Hessian.
    ///
    /// Boundary accessor: returns `&Array2<f64>` so out-of-scope consumers
    /// (CLI, GPU, families) keep their pre-newtype call shape. Use
    /// the `UnscaledPrecision` newtype directly when the caller wants the
    /// `UnscaledPrecision` newtype to enforce the dispersion-ownership
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

    pub fn smoothing_correction_method(&self) -> Option<SmoothingCorrectionMethod> {
        self.inference
            .as_ref()
            .and_then(|inference| inference.smoothing_correction_method)
    }

    /// The exact first-order IFT smoothing-parameter-uncertainty correction.
    /// This is the accessor the #946 WPS corrected-EDF/AIC channel reads: it is
    /// populated whenever the first-order geometry was computable, including on
    /// models saved by earlier releases whose primary correction was a
    /// sigma-point cubature.
    pub fn smoothing_correction_first_order(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.smoothing_correction_first_order.as_ref())
    }

    /// Provenance for [`Self::smoothing_correction_first_order`]. Always
    /// either `None` or `Some(FirstOrderIdentifiedSubspace{..})`.
    pub fn smoothing_correction_method_first_order(&self) -> Option<SmoothingCorrectionMethod> {
        self.inference
            .as_ref()
            .and_then(|inference| inference.smoothing_correction_method_first_order)
    }

    /// The typed reason this fit retains no smoothing correction, minted at fit time (#2677).
    pub fn smoothing_correction_absence(&self) -> Option<&SmoothingCorrectionAbsence> {
        self.inference
            .as_ref()
            .and_then(|inference| inference.smoothing_correction_absence.as_ref())
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

    /// Per-block rank-bound status aligned 1:1 with `penalty_block_trace` (#2901).
    /// Empty when the producing path recorded none.
    pub fn edf_rank_bound(&self) -> &[crate::estimate::EdfRankBound] {
        self.inference
            .as_ref()
            .map(|inf| inf.edf_rank_bound.as_slice())
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

    /// Number of coefficient blocks.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
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
        family: &gam_problem::LikelihoodSpec,
    ) -> Result<FittedLinkState, EstimationError> {
        match (&family.response, &family.link) {
            (ResponseFamily::Gaussian, _) | (ResponseFamily::StudentT { .. }, _) => {
                Ok(FittedLinkState::Standard(None))
            }
            // Every state-less binomial probability link decodes to the bare
            // `Standard(None)` payload — the concrete `StandardLink` lives on the
            // family/spec, not in the fitted-link record. LogLog and Cauchit
            // belong here alongside Logit/Probit/CLogLog: they are ordinary
            // state-less links (#2158), so omitting them dropped a fully-fitted
            // model into the `(Binomial, _)` "unsupported combination" error at
            // predict time, breaking the fit→predict round-trip. The
            // state-bearing links (SAS/BetaLogistic/Mixture/LatentCLogLog) are
            // handled by the arms below. The log link is the state-less
            // relative-risk cell (a generic variance × link cell with the
            // feasibility set η < 0); the identity link is not a legal
            // binomial cell and correctly falls to the catch-all.
            (
                ResponseFamily::Binomial,
                InverseLink::Standard(
                    StandardLink::Logit
                    | StandardLink::Probit
                    | StandardLink::CLogLog
                    | StandardLink::LogLog
                    | StandardLink::Cauchit
                    | StandardLink::Log,
                ),
            ) => Ok(FittedLinkState::Standard(None)),
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
            | (ResponseFamily::Gamma, _)
            | (ResponseFamily::InverseGaussian, _) => Ok(FittedLinkState::Standard(None)),
            (ResponseFamily::Beta { .. }, _) => Ok(FittedLinkState::Standard(None)),
            (ResponseFamily::RoystonParmar, _) => Ok(FittedLinkState::Standard(None)),
        }
    }
}

#[cfg(test)]
mod curvature_evidence_serialized_contract_2561_tests {
    use super::{
        CertifiedRung, CurvatureAdmissibility, CurvatureEvidence, OuterCriterionCertificate,
        OuterStationarityCertificate,
    };

    /// `hessian_psd` is a PUBLISHED contract with an external consumer
    /// (`gamfit/_summary.py`, domain `null | true | false`) and it is carried in
    /// stored model bytes that have no version tag. #2561 replaced the raw
    /// `Option<bool>` field with `CurvatureEvidence` and kept the wire form
    /// via `#[serde(from = "Option<bool>", into = "Option<bool>")]`.
    ///
    /// Nothing gated that. The type is `Serialize + Deserialize` by derive, so
    /// deleting the two serde attributes still compiles and still passes every
    /// test that exercises the enum in memory — while silently changing the
    /// published JSON from `null | true | false` to a Rust variant name and
    /// making every previously-stored certificate unloadable.
    ///
    /// The round trip is deliberately LOSSY and the doc says so: every unmeasured
    /// state reloads as `NotAvailable`. Asserting the loss rather than working
    /// around it is the point — it records that those states are
    /// acceptance-identical, which is the fact that makes the loss sound.
    #[test]
    fn curvature_evidence_serializes_as_the_legacy_optional_bool_2561() {
        // A stationary certificate whose only free conjunct is the evidence
        // under test, so what it publishes is about that evidence alone.
        let certificate_with = |curvature: CurvatureEvidence| -> OuterCriterionCertificate {
            let rung: CertifiedRung = gam_problem::StationarityRung {
                label: "solver-band",
                derived_standard: false,
            }
            .into();
            OuterCriterionCertificate {
                stationarity: OuterStationarityCertificate::AnalyticGradient {
                    grad_norm: 1.0e-9,
                    projected_grad_norm: 1.0e-9,
                    bound: 1.0e-6,
                    rung,
                },
                curvature,
                lambdas_railed: Vec::new(),
                railed_facts: Vec::new(),
                newton_polish: None,
                curvature_floor: None,
                criterion_error: None,
            }
        };
        for (evidence, expected) in [
            (CurvatureEvidence::Measured { psd: true }, "true"),
            (CurvatureEvidence::Measured { psd: false }, "false"),
            (CurvatureEvidence::NotAvailable, "null"),
            (CurvatureEvidence::NotSpent, "null"),
            (CurvatureEvidence::NoEstimand, "null"),
            (CurvatureEvidence::CriterionContradicted, "null"),
            (CurvatureEvidence::CriterionUnresolvable, "null"),
        ] {
            let wire = serde_json::to_string(&evidence).expect("evidence serializes");
            assert_eq!(
                wire, expected,
                "the published `hessian_psd` domain is `null | true | false`; \
                 {evidence:?} serialized as {wire}"
            );
            // The certificate publishes it under the legacy key.
            let published = serde_json::to_value(certificate_with(evidence))
                .expect("certificate serializes");
            assert_eq!(
                published.get("hessian_psd").map(|value| value.to_string()),
                Some(expected.to_string()),
                "the certificate must publish {evidence:?} under `hessian_psd`: {published}"
            );
        }

        // Legacy documents reload, and a `null` NEVER becomes a measurement.
        for (wire, expected) in [
            ("true", CurvatureEvidence::Measured { psd: true }),
            ("false", CurvatureEvidence::Measured { psd: false }),
            ("null", CurvatureEvidence::NotAvailable),
        ] {
            let parsed: CurvatureEvidence =
                serde_json::from_str(wire).expect("legacy hessian_psd reloads");
            assert_eq!(
                parsed, expected,
                "legacy {wire} must reload as {expected:?}"
            );
        }

        // The lossy half, asserted rather than hidden.
        for lossy in [
            CurvatureEvidence::NotSpent,
            CurvatureEvidence::NoEstimand,
            CurvatureEvidence::CriterionContradicted,
            CurvatureEvidence::CriterionUnresolvable,
        ] {
            let wire = serde_json::to_string(&lossy).expect("serializes");
            let back: CurvatureEvidence = serde_json::from_str(&wire).expect("reloads");
            assert_eq!(
                back,
                CurvatureEvidence::NotAvailable,
                "{lossy:?} is documented to reload as NotAvailable"
            );
            assert_eq!(
                back.psd(),
                None,
                "the whole point of #2561 is that an unmeasured curvature never \
                 presents as a measured one, on either side of the wire"
            );
        }
    }

    /// The defect #2561 names, stated as a test: an ABSENT measurement and a
    /// PRESENT admissible one must not be the same answer.
    ///
    /// The old predicate was `hessian_psd != Some(false) || floor cleared`, and
    /// its `true` was produced by two structurally different facts — a Hessian
    /// measured and found admissible, and a route that never measured one. A
    /// gate whose passing condition is satisfied by the ABSENCE of an
    /// observation cannot fail on the case it exists to catch.
    ///
    /// Acceptance is deliberately unchanged (neither refuses), so the only
    /// thing separating them is the verdict the production certificate reports.
    #[test]
    fn an_unmeasured_curvature_is_not_the_same_answer_as_an_admissible_one_2561() {
        // A stationary certificate whose only free conjunct is the curvature
        // evidence under test, so every verdict below is about that evidence.
        let certificate_with = |curvature: CurvatureEvidence| -> OuterCriterionCertificate {
            let rung: CertifiedRung = gam_problem::StationarityRung {
                label: "solver-band",
                derived_standard: false,
            }
            .into();
            OuterCriterionCertificate {
                stationarity: OuterStationarityCertificate::AnalyticGradient {
                    grad_norm: 1.0e-9,
                    projected_grad_norm: 1.0e-9,
                    bound: 1.0e-6,
                    rung,
                },
                curvature,
                lambdas_railed: Vec::new(),
                railed_facts: Vec::new(),
                newton_polish: None,
                curvature_floor: None,
                criterion_error: None,
            }
        };
        let measured = certificate_with(CurvatureEvidence::Measured { psd: true });
        assert_eq!(
            measured.hessian_psd(),
            Some(true),
            "a measured PSD Hessian reports its verdict"
        );
        assert_eq!(
            measured.curvature_verdict(),
            CurvatureAdmissibility::Admissible,
            "a measured PSD Hessian is admissible"
        );
        for unmeasured in [
            CurvatureEvidence::NotAvailable,
            CurvatureEvidence::NotSpent,
            CurvatureEvidence::NoEstimand,
        ] {
            let certificate = certificate_with(unmeasured);
            assert_eq!(
                certificate.hessian_psd(),
                None,
                "{unmeasured:?} answered no curvature question, so it has no verdict"
            );
            assert_eq!(
                certificate.curvature_verdict(),
                CurvatureAdmissibility::Unevaluated {
                    evidence: unmeasured
                },
                "{unmeasured:?} must be reported as unevaluated, naming why"
            );
            assert_ne!(
                certificate.curvature_verdict(),
                CurvatureAdmissibility::Admissible,
                "an unevaluated curvature must not compare equal to an admissible one"
            );
            assert!(
                certificate.curvature_not_refused(),
                "absence of a measurement does not refuse: acceptance is unchanged"
            );
        }
        // And the three unmeasured states stay distinguishable from each other
        // in memory, which is why the enum exists: `NoEstimand` (no curvature
        // to have) is a different fact from `NotAvailable` (a curvature
        // question that could not be answered) and from `NotSpent` (a
        // screening pass that declined to ask).
        assert_ne!(
            CurvatureEvidence::NoEstimand,
            CurvatureEvidence::NotAvailable
        );
        assert_ne!(CurvatureEvidence::NotSpent, CurvatureEvidence::NotAvailable);
        assert_ne!(CurvatureEvidence::NoEstimand, CurvatureEvidence::NotSpent);
    }
}
