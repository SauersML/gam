use super::*;

pub(crate) fn cli_frailty_kind(
    frailty_kind: Option<FrailtyKindArg>,
) -> Option<crate::config_resolve::CliFrailtyKind> {
    frailty_kind.map(|kind| match kind {
        FrailtyKindArg::GaussianShift => crate::config_resolve::CliFrailtyKind::GaussianShift,
        FrailtyKindArg::HazardMultiplier => crate::config_resolve::CliFrailtyKind::HazardMultiplier,
    })
}

pub(crate) fn cli_hazard_loading(
    hazard_loading: Option<HazardLoadingArg>,
) -> Option<crate::config_resolve::CliHazardLoading> {
    hazard_loading.map(|loading| match loading {
        HazardLoadingArg::Full => crate::config_resolve::CliHazardLoading::Full,
        HazardLoadingArg::LoadedVsUnloaded => {
            crate::config_resolve::CliHazardLoading::LoadedVsUnloaded
        }
    })
}

pub(crate) fn fit_frailty_spec_from_args(
    args: &FitArgs,
    context: &str,
) -> Result<gam::families::survival::lognormal_kernel::FrailtySpec, String> {
    crate::config_resolve::resolve_cli_frailty_spec(
        cli_frailty_kind(args.frailty_kind),
        args.frailty_sd,
        cli_hazard_loading(args.hazard_loading),
        context,
    )
}

pub(crate) fn fit_frailty_spec_from_survival_args(
    args: &SurvivalArgs,
    context: &str,
) -> Result<gam::families::survival::lognormal_kernel::FrailtySpec, String> {
    crate::config_resolve::resolve_cli_frailty_spec(
        cli_frailty_kind(args.frailty_kind),
        args.frailty_sd,
        cli_hazard_loading(args.hazard_loading),
        context,
    )
}

pub(crate) fn fixed_hazard_multiplier_from_saved_family(
    family: &FittedFamily,
) -> Result<
    (
        f64,
        gam::families::survival::lognormal_kernel::HazardLoading,
    ),
    String,
> {
    let frailty = family.frailty().ok_or_else(|| {
        "saved latent survival/binary model requires a fixed HazardMultiplier frailty specification"
            .to_string()
    })?;
    fixed_latent_hazard_frailty(frailty, "saved latent survival/binary model")
}

pub(crate) fn build_bernoulli_marginal_slope_saved_model(
    formula: String,
    data_schema: DataSchema,
    slope_formula: String,
    z_column: String,
    training_headers: Vec<String>,
    training_feature_ranges: Vec<(f64, f64)>,
    resolved_marginalspec: TermCollectionSpec,
    resolved_slopespec: TermCollectionSpec,
    fit_result: UnifiedFitResult,
    p_marginal: usize,
    baseline_marginal: f64,
    baseline_slope: f64,
    latent_z_normalization: SavedLatentZNormalization,
    latent_measure: LatentMeasureKind,
    latent_z_rank_int_calibration: Option<gam::families::bms::LatentZRankIntCalibration>,
    latent_z_conditional_calibration: Option<gam::families::bms::LatentZConditionalCalibration>,
    score_warp_runtime: Option<&DeviationRuntime>,
    link_dev_runtime: Option<&DeviationRuntime>,
    base_link: InverseLink,
    frailty: gam::families::survival::lognormal_kernel::FrailtySpec,
) -> Result<SavedModel, String> {
    // Thin adapter over the shared core assembler. Everything semantic — the
    // singular/vector mirror fields, the flex-runtime serialization, the
    // likelihood resolution — lives in
    // `gam::inference::model_payload_builders` so the CLI- and Python-created
    // payloads are identical by construction. The CLI's only source-specific
    // contribution here is per-feature training ranges (the FFI path persists
    // headers without them); the caller applies offset columns to the returned
    // model.
    let payload = assemble_bernoulli_marginal_slope_payload(
        BernoulliMarginalSlopeInputs {
            formula,
            data_schema,
            slope_formula,
            z_column,
            resolved_marginalspec,
            resolved_slopespec,
            fit_result,
            p_marginal,
            baseline_marginal,
            baseline_slope,
            latent_z_normalization,
            latent_measure,
            latent_z_rank_int_calibration,
            latent_z_conditional_calibration,
            score_warp_runtime,
            link_dev_runtime,
            base_link,
            frailty,
        },
        SavedModelSourceMetadata {
            training_headers,
            training_feature_ranges: Some(training_feature_ranges),
            offset_column: None,
            noise_offset_column: None,
        },
    )?;
    Ok(SavedModel::from_payload(payload))
}

pub(crate) fn resolve_bernoulli_marginal_slope_base_link(
    linkspec: Option<&LinkFormulaSpec>,
    context: &str,
) -> Result<InverseLink, String> {
    let Some(linkspec) = linkspec else {
        return Ok(InverseLink::Standard(StandardLink::Probit));
    };
    let choice = parse_link_choice(Some(&linkspec.link), false)?;
    let Some(choice) = choice else {
        return Ok(InverseLink::Standard(StandardLink::Probit));
    };
    if matches!(choice.mode, LinkMode::Flexible) {
        return Err(format!(
            "{context} does not accept flexible(...) inside link(); use link(type=<base-link>) plus linkwiggle(...) to learn anchored link deviations"
        ));
    }
    if choice.mixture_components.is_some() || choice.link != LinkFunction::Probit {
        return Err(format!(
            "{context} requires link(type=probit); non-probit marginal-slope links are not supported by the calibrated de-nested probit kernel"
        ));
    }
    if linkspec.sas_init.is_some() {
        return Err(
            "link(sas_init=...) requires link(type=sas), which marginal-slope does not support"
                .to_string(),
        );
    }
    if linkspec.beta_logistic_init.is_some() {
        return Err("link(beta_logistic_init=...) requires link(type=beta-logistic), which marginal-slope does not support".to_string());
    }
    if linkspec.mixture_rho.is_some() {
        return Err("link(rho=...) requires link(type=blended(...)/mixture(...)), which marginal-slope does not support".to_string());
    }
    Ok(InverseLink::Standard(StandardLink::Probit))
}

pub(crate) fn build_transformation_normal_saved_model(
    formula: String,
    data_schema: DataSchema,
    training_headers: Vec<String>,
    training_feature_ranges: Vec<(f64, f64)>,
    resolved_covariate_spec: TermCollectionSpec,
    fit_result: UnifiedFitResult,
    family: &gam::families::transformation_normal::TransformationNormalFamily,
    score_calibration: gam::inference::model::TransformationScoreCalibration,
) -> SavedModel {
    // Thin adapter over the shared core assembler; the CLI supplies per-feature
    // training ranges and no offset columns. See
    // `assemble_transformation_normal_payload`.
    let payload = assemble_transformation_normal_payload(
        TransformationNormalInputs {
            formula,
            data_schema,
            resolved_covariate_spec,
            fit_result,
            family,
            score_calibration,
        },
        SavedModelSourceMetadata {
            training_headers,
            training_feature_ranges: Some(training_feature_ranges),
            offset_column: None,
            noise_offset_column: None,
        },
    );
    SavedModel::from_payload(payload)
}

pub(crate) fn core_saved_fit_result(
    beta: Array1<f64>,
    lambdas: Array1<f64>,
    standard_deviation: f64,
    beta_covariance: Option<Array2<f64>>,
    beta_covariance_corrected: Option<Array2<f64>>,
    summary: SavedFitSummary,
) -> Result<UnifiedFitResult, String> {
    // Saved models are part of the stable inference contract. Reject non-finite
    // values at construction time so JSON cannot silently encode them as null.
    let summary = summary
        .validated()
        .map_err(|error| format!("saved fit summary metrics are invalid: {error}"))?;
    validate_all_finite("fit_result.beta", beta.iter().copied())
        .expect("core_saved_fit_result called with non-finite beta");
    validate_all_finite("fit_result.lambdas", lambdas.iter().copied())
        .expect("core_saved_fit_result called with non-finite lambdas");
    // Saved-model contract: fit_result.standard_deviation is residual
    // standard deviation sigma for Gaussian identity models and the
    // response-scale summary paired with explicit likelihood-scale metadata
    // for non-Gaussian models.
    ensure_finite_scalar("fit_result.standard_deviation", standard_deviation)
        .expect("core_saved_fit_result called with non-finite standard_deviation");
    if let Some(cov) = beta_covariance.as_ref() {
        validate_all_finite("fit_result.beta_covariance", cov.iter().copied())
            .expect("core_saved_fit_result called with non-finite beta_covariance");
    }
    if let Some(cov) = beta_covariance_corrected.as_ref() {
        validate_all_finite("fit_result.beta_covariance_corrected", cov.iter().copied())
            .expect("core_saved_fit_result called with non-finite beta_covariance_corrected");
    }
    {
        // rho IS the canonical coordinate; lambda is derived from it, not the
        // other way round.
        //
        // `UnifiedFitResult` validates `lambda == exp(rho)` BIT-FOR-BIT
        // (`log_lambdas_match_lambdas` compares `to_bits()`), which encodes the
        // REML convention: the outer optimizer moves rho and lambda follows.
        // Deriving `rho = ln(lambda)` and then handing back the ORIGINAL lambda
        // violates that for almost every value, because `exp(ln(x)) != x` in
        // floating point -- `ln(1e-3)` round-trips to `0.0009999999999999998`,
        // one ulp off. Every saved fit built through here therefore failed with
        //
        //   UnifiedFitResult log_lambdas must equal ln(lambdas) elementwise
        //
        // Canonicalising through rho makes the invariant hold by construction
        // instead of by luck. The cost is at most one ulp on a lambda that was
        // itself read back from a serialized decimal, and the alternative --
        // loosening the validator to a tolerance -- would drop the exactness
        // the rest of the pipeline relies on to round-trip rho.
        if let Some(bad) = lambdas.iter().copied().find(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(format!(
                "saved fit carries a smoothing strength {bad} that is not finite and positive; a \
                 strength is exp(rho), so no rho reproduces it — refusing to floor it"
            ));
        }
        let log_lambdas = lambdas.mapv(f64::ln);
        // `checked_exp_log_strength` is a domain check followed by plain
        // `rho.exp()`, so this is bit-identical to what the validator computes
        // -- and `gam-problem` is not a dependency of this crate.
        let lambdas = log_lambdas.mapv(f64::exp);
        // Do not export a synthetic/placeholder Hessian here. Saved fits built
        // from externally supplied summary/covariance data may provide covariance
        // for prediction, but HMC/NUTS whitening requires an explicit upstream
        // penalized Hessian from the fitter itself.
        let covariance_conditional = beta_covariance;
        let covariance_corrected = beta_covariance_corrected;
        // A payload written before the criterion could be absent carries `0.0`
        // here when the reconstruction lands on the exact-fit boundary. That is
        // the placeholder, not a criterion — drop it rather than hand it to a
        // constructor that (rightly) refuses one, which is exactly the
        // normalization `UnifiedFitResult::reml_score` performs at read time
        // for the same legacy state (#2595).
        let reml_score = summary.reml_score.filter(|_| {
            !gam::estimate::is_zero_dispersion_boundary(
                summary.likelihood_family.as_ref(),
                summary.likelihood_scale,
                standard_deviation,
            )
        });
        let penalized_objective = reml_score;
        UnifiedFitResult::try_from_parts(gam::estimate::UnifiedFitResultParts {
            blocks: vec![gam::estimate::FittedBlock {
                beta: beta.clone(),
                role: gam::estimate::BlockRole::Mean,
                edf: 0.0,
                lambdas: lambdas.clone(),
            }],
            training_sample_size: summary.training_sample_size,
            log_lambdas,
            lambdas,
            likelihood_family: summary.likelihood_family,
            likelihood_scale: summary.likelihood_scale,
            log_likelihood_normalization: summary.log_likelihood_normalization,
            log_likelihood: summary.log_likelihood,
            deviance: summary.deviance,
            reml_score,
            stable_penalty_term: summary.stable_penalty_term,
            penalized_objective,
            // A fit reconstructed from a saved-model summary performed no device
            // (GPU) execution in this process — it was deserialized from disk —
            // so the device-use flag is false. `SavedFitSummary` does not persist
            // this field; it is a property of the original fit run, not the saved
            // artifact.
            used_device: false,
            outer_iterations: summary.iterations,
            // Saved reconstruction obeys the same strict inner convergence
            // authority as live assembly. A stalled/exhausted status remains a
            // checkpoint and cannot be reinterpreted as a model while loading.
            outer_converged: summary.pirls_status.is_converged(),
            outer_gradient_norm: Some(summary.finalgrad_norm),
            standard_deviation,
            covariance_conditional,
            covariance_corrected,
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: summary.pirls_status,
            max_abs_eta: summary.max_abs_eta,
            constraint_kkt: None,
            artifacts: gam::estimate::FitArtifacts {
                pirls: None,
                // Restore the outer-stationarity certificate the live fit carried
                // so `UnifiedFitResult::try_from_parts` reconstructs the same
                // `Analytic` convergence evidence. Dropping it here made the gate
                // reject every location-scale fit that ran outer iterations.
                criterion_certificate: summary.criterion_certificate,
                ..Default::default()
            },
            inner_cycles: 0,
        })
        .map_err(|error| format!("saved fit metrics are invalid: {error}"))
    }
}

// The generative dispersion picker `family_noise_parameter` lived here as a
// third divergent copy; it now lives once in `gam::generative` and the live
// `gam generate` path (`run_sample_generate_report`) calls it directly. See the
// doc comment there for the per-family rationale (#1124).

#[derive(Clone)]
pub(crate) struct SavedFitSummary {
    /// Original training rows, carried through compact saved-fit
    /// materialization without consulting optional working geometry.
    pub(crate) training_sample_size: usize,
    pub(crate) likelihood_family: Option<LikelihoodSpec>,
    pub(crate) likelihood_scale: LikelihoodScaleMetadata,
    pub(crate) log_likelihood_normalization: LogLikelihoodNormalization,
    pub(crate) log_likelihood: f64,
    pub(crate) iterations: usize,
    pub(crate) finalgrad_norm: f64,
    pub(crate) pirls_status: gam::pirls::PirlsStatus,
    pub(crate) deviance: f64,
    pub(crate) stable_penalty_term: f64,
    pub(crate) max_abs_eta: f64,
    /// The fit's REML/LAML criterion, or `None` when the fit has none at all
    /// (an exactly-interpolating Gaussian fit; see
    /// `UnifiedFitResult::reml_score`). Serialized as `null` in that case, and
    /// the reconstructed fit carries the same absence rather than a zero
    /// (#2595).
    pub(crate) reml_score: Option<f64>,
    /// Analytic outer-stationarity certificate proving the smoothing coordinate
    /// converged. `None` only when the fit ran no outer iterations (fixed-λ);
    /// whenever the fit optimized ρ this must carry the live certificate so the
    /// reconstructed `UnifiedFitResult` clears the assembly gate in
    /// `result_types.rs` instead of being rejected as non-stationary.
    pub(crate) criterion_certificate: Option<gam::estimate::OuterCriterionCertificate>,
}

impl SavedFitSummary {
    fn validated(self) -> Result<Self, String> {
        if self.training_sample_size == 0 {
            return Err("fit_result.training_sample_size must be positive".to_string());
        }
        ensure_finite_scalar("fit_result.log_likelihood", self.log_likelihood)?;
        ensure_finite_scalar("fit_result.finalgrad_norm", self.finalgrad_norm)?;
        ensure_finite_scalar("fit_result.deviance", self.deviance)?;
        ensure_finite_scalar("fit_result.stable_penalty_term", self.stable_penalty_term)?;
        ensure_finite_scalar("fit_result.max_abs_eta", self.max_abs_eta)?;
        if let Some(reml_score) = self.reml_score {
            ensure_finite_scalar("fit_result.reml_score", reml_score)?;
        }
        Ok(self)
    }

    pub(crate) fn from_blockwise_fit(
        fit: &gam::estimate::UnifiedFitResult,
    ) -> Result<Self, String> {
        let stable_penalty_term = fit.stable_penalty_term;
        let max_abs_eta = fit
            .block_states
            .iter()
            .flat_map(|b| b.eta.iter())
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        Self {
            training_sample_size: fit.training_sample_size(),
            likelihood_family: fit.likelihood_family.clone(),
            likelihood_scale: fit.likelihood_scale,
            log_likelihood_normalization: fit.log_likelihood_normalization,
            log_likelihood: fit.log_likelihood,
            iterations: fit.outer_iterations,
            // FitInfo.finalgrad_norm is a hard f64 (its own validator
            // ensure_finite_scalar fires below); when the outer skipped
            // gradient measurement (cache hit / gradient-free), persist 0.0
            // and rely on `pirls_status` for convergence quality.
            finalgrad_norm: fit.outer_gradient_norm.unwrap_or(0.0),
            // Persist the *real* status the fit carries (set at construction,
            // see `UnifiedFitResultParts::pirls_status`). Deriving it from the
            // `outer_converged` bool here would collapse the five-way taxonomy
            // (MaxIterationsReached / LmStepSearchExhausted / Unstable / …) into
            // a single stalled bucket, silently relabeling broken checkpoints
            // as healthy for downstream consumers. Persist the authoritative
            // inner status; live and reconstructed fits can only carry
            // `Converged`, while diagnostic/checkpoint payloads retain the
            // other variants outside this model constructor.
            pirls_status: fit.convergence_evidence().inner_status(),
            deviance: fit.deviance,
            stable_penalty_term,
            max_abs_eta,
            reml_score: fit.reml_score(),
            // Carry the live analytic certificate forward. Deriving it from the
            // sealed convergence evidence keeps the reconstructed fit's outer
            // stationarity proof identical to the one the optimizer minted; a
            // fixed-λ fit (no outer iterations) legitimately yields `None`.
            criterion_certificate: fit.convergence_evidence().outer_certificate().cloned(),
        }
        .validated()
    }
}

use gam::estimate::{ensure_finite_scalar, validate_all_finite};

pub(crate) fn compact_saved_multiblock_fit_result(
    blocks: Vec<gam::estimate::FittedBlock>,
    lambdas: Array1<f64>,
    standard_deviation: f64,
    beta_covariance: Option<Array2<f64>>,
    beta_covariance_corrected: Option<Array2<f64>>,
    geometry: Option<gam::estimate::FitGeometry>,
    summary: SavedFitSummary,
) -> Result<UnifiedFitResult, String> {
    let total: usize = blocks.iter().map(|block| block.beta.len()).sum();
    let mut beta = Array1::zeros(total);
    let mut offset = 0;
    for block in &blocks {
        let width = block.beta.len();
        beta.slice_mut(s![offset..offset + width])
            .assign(&block.beta);
        offset += width;
    }
    let mut fit_result = core_saved_fit_result(
        beta,
        lambdas,
        standard_deviation,
        beta_covariance,
        beta_covariance_corrected,
        summary,
    )?;
    fit_result.blocks = blocks;
    if let Some(geom) = geometry {
        if let Some(inf) = fit_result.inference.as_mut() {
            inf.penalized_hessian = geom.penalized_hessian.clone();
        }
        fit_result.geometry = Some(geom);
    }
    fit_result
        .validate_numeric_finiteness()
        .expect("compact saved fit materialization violated fitted-result invariants");
    Ok(fit_result)
}

pub(crate) fn compact_saved_survival_location_scale_fit_result(
    fit: &UnifiedFitResult,
    inverse_link: &InverseLink,
) -> Result<UnifiedFitResult, String> {
    let mut fit_result = compact_saved_multiblock_fit_result(
        fit.blocks.clone(),
        fit.lambdas.clone(),
        1.0,
        fit.covariance_conditional.clone(),
        fit.covariance_corrected.clone(),
        fit.geometry.clone(),
        SavedFitSummary::from_blockwise_fit(fit)?,
    )?;
    apply_inverse_link_state_to_fit_result(&mut fit_result, inverse_link);
    fit_result.artifacts.survival_link_wiggle_knots =
        fit.artifacts.survival_link_wiggle_knots.clone();
    fit_result.artifacts.survival_link_wiggle_degree = fit.artifacts.survival_link_wiggle_degree;
    Ok(fit_result)
}

pub(crate) fn write_model_json(path: &Path, model: &SavedModel) -> Result<(), String> {
    model.save_to_path(path)?;
    cli_out!("saved model: {}", path.display());
    Ok(())
}

pub(crate) fn write_payload_json(path: &Path, payload: FittedModelPayload) -> Result<(), String> {
    let model = SavedModel::from_payload(payload);
    write_model_json(path, &model)
}

pub(crate) fn print_inference_summary(notes: &[String]) {
    if notes.is_empty() {
        return;
    }
    cli_err!("Auto-discovery summary:");
    for note in notes {
        cli_err!("  - {}", note);
    }
}

/// Persist the fit's case-weight column name on the saved model.
///
/// Offset columns are set directly by the fit routes; the weight column was
/// silently dropped, so
/// `gam diagnose` (which reloads the prior weights by name to reconstruct the
/// IRLS working weights for the ALO geometry path) could not recover the case
/// weights of a `--weights-column` fit and fell back to unit weights. Persist it
/// alongside the offset so every post-fit diagnostic sees the weights the model
/// was actually fit with.
pub(crate) fn set_saved_weight_column(
    payload: &mut FittedModelPayload,
    weight_column: Option<String>,
) {
    payload.weight_column = weight_column;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A live location-scale fit that optimized its smoothing coordinate: it ran
    /// one outer iteration and carries the analytic outer-stationarity
    /// certificate the optimizer minted. This is the state that reaches the CLI
    /// save path for `gam fit --predict-noise --out ...`.
    fn location_scale_fit_with_outer_certificate() -> UnifiedFitResult {
        let certificate = gam::estimate::OuterCriterionCertificate {
            stationarity: gam::estimate::OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 1e-8,
                projected_grad_norm: 1e-8,
                bound: 1e-4,
                rung: gam::model_types::CertifiedRung {
                    label: "solver-band".to_string(),
                    derived_standard: false,
                },
            },
            curvature: gam::model_types::CurvatureEvidence::Measured { psd: true },
            lambdas_railed: Vec::new(),
            railed_facts: Vec::new(),
            curvature_floor: None,
        };
        UnifiedFitResult::try_from_parts(gam::estimate::UnifiedFitResultParts {
            blocks: vec![gam::estimate::FittedBlock {
                beta: Array1::from_vec(vec![0.5, -0.25]),
                role: gam::estimate::BlockRole::Mean,
                edf: 2.0,
                lambdas: Array1::from_vec(vec![1.0]),
            }],
            training_sample_size: 32,
            log_lambdas: Array1::zeros(1),
            lambdas: Array1::from_vec(vec![1.0]),
            likelihood_family: None,
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
            used_device: false,
            outer_iterations: 1,
            outer_converged: true,
            outer_gradient_norm: Some(1e-8),
            standard_deviation: 1.0,
            covariance_conditional: None,
            covariance_corrected: None,
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: gam::pirls::PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: gam::estimate::FitArtifacts {
                criterion_certificate: Some(certificate),
                ..Default::default()
            },
            inner_cycles: 0,
        })
        .expect("fixture certificate certifies outer stationarity")
    }

    #[test]
    fn compact_saved_fit_retains_outer_certificate_and_training_sample_size() {
        let fit = location_scale_fit_with_outer_certificate();
        // Precondition: the live fit optimized ρ and holds an analytic proof.
        assert_eq!(fit.outer_iterations, 1);
        assert!(fit.convergence_evidence().outer_certificate().is_some());

        // The intermediate summary must carry the certificate forward — dropping
        // it here is exactly what made the compact reconstruction reject the fit.
        let summary =
            SavedFitSummary::from_blockwise_fit(&fit).expect("saved summary builds from live fit");
        assert!(
            summary.criterion_certificate.is_some(),
            "SavedFitSummary dropped the live outer certificate"
        );
        assert_eq!(
            summary.training_sample_size,
            fit.training_sample_size(),
            "SavedFitSummary must carry the authoritative training-row count"
        );

        // Reconstructing the compact saved fit must not panic and must preserve
        // the Analytic outer convergence evidence (before the fix this rebuilt
        // `FitArtifacts` with `criterion_certificate: None` and the assembly gate
        // rejected the 1-outer-iteration fit as non-stationary).
        let reconstructed = compact_saved_multiblock_fit_result(
            fit.blocks.clone(),
            fit.lambdas.clone(),
            1.0,
            fit.covariance_conditional.clone(),
            fit.covariance_corrected.clone(),
            fit.geometry.clone(),
            summary,
        ).expect("saved fit reconstruction");
        assert_eq!(reconstructed.outer_iterations, 1);
        assert_eq!(
            reconstructed.training_sample_size(),
            fit.training_sample_size(),
            "compact materialization changed the authoritative training-row count"
        );
        assert!(
            reconstructed
                .convergence_evidence()
                .outer_certificate()
                .is_some(),
            "compacted saved fit dropped the outer stationarity certificate"
        );
    }

    /// A valid outer certificate cannot bypass the sealed constructor's strict
    /// inner convergence gate. `MaxIterationsReached` remains a checkpoint.
    #[test]
    fn max_iterations_reached_status_is_still_rejected() {
        let certificate = gam::estimate::OuterCriterionCertificate {
            stationarity: gam::estimate::OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 1e-8,
                projected_grad_norm: 1e-8,
                bound: 1e-4,
                rung: gam::model_types::CertifiedRung {
                    label: "solver-band".to_string(),
                    derived_standard: false,
                },
            },
            curvature: gam::model_types::CurvatureEvidence::Measured { psd: true },
            lambdas_railed: Vec::new(),
            railed_facts: Vec::new(),
            curvature_floor: None,
        };
        let result = UnifiedFitResult::try_from_parts(gam::estimate::UnifiedFitResultParts {
            blocks: vec![gam::estimate::FittedBlock {
                beta: Array1::from_vec(vec![0.5, -0.25]),
                role: gam::estimate::BlockRole::Mean,
                edf: 2.0,
                lambdas: Array1::from_vec(vec![1.0]),
            }],
            training_sample_size: 32,
            log_lambdas: Array1::zeros(1),
            lambdas: Array1::from_vec(vec![1.0]),
            likelihood_family: None,
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
            used_device: false,
            outer_iterations: 1,
            outer_converged: true,
            outer_gradient_norm: Some(1e-8),
            standard_deviation: 1.0,
            covariance_conditional: None,
            covariance_corrected: None,
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: gam::pirls::PirlsStatus::MaxIterationsReached,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: gam::estimate::FitArtifacts {
                criterion_certificate: Some(certificate),
                ..Default::default()
            },
            inner_cycles: 0,
        });
        assert!(
            result.is_err(),
            "a non-converged inner status (MaxIterationsReached) must be rejected"
        );
    }
}
