
fn set_training_feature_metadata_from_dataset(payload: &mut FittedModelPayload, ds: &Dataset) {
    payload.set_training_feature_metadata(ds.headers.clone(), ds.feature_ranges());
}


fn deviation_block_config_from_formula_linkwiggle(
    wiggle: &LinkWiggleFormulaSpec,
) -> Result<DeviationBlockConfig, String> {
    // The score-warp / link-deviation block is realized by the structurally
    // *cubic* I-spline `DeviationRuntime` (see
    // `build_deviation_block_from_knots_and_design_seed`): its span tables,
    // C2-continuous construction, and derivative operators are all hard-wired
    // to cubic, so the only realizable `degree` is 3. The shared formula parser
    // intentionally stays general (it also feeds the arbitrary-degree
    // `timewiggle` / location-scale monotone basis), so the cubic-only contract
    // is enforced here, at the routing boundary that feeds this runtime —
    // up front, instead of failing deep inside the fit after expensive setup.
    if wiggle.degree != 3 {
        return Err(format!(
            "linkwiggle() degree must be 3 when routed into the score-warp / \
             link-deviation block: that runtime is a cubic I-spline and only \
             supports cubic splines; got degree={}",
            wiggle.degree
        ));
    }
    let defaults = WigglePenaltyConfig::cubic_triple_operator_default();
    Ok(DeviationBlockConfig {
        degree: wiggle.degree,
        num_internal_knots: wiggle.num_internal_knots,
        penalty_order: *wiggle.penalty_orders.iter().max().unwrap_or(&2),
        penalty_orders: wiggle.penalty_orders.clone(),
        double_penalty: wiggle.double_penalty,
        monotonicity_eps: defaults.monotonicity_eps,
    })
}


#[derive(Debug)]
struct MarginalSlopeDeviationRouting {
    score_warp: Option<DeviationBlockConfig>,
    link_dev: Option<DeviationBlockConfig>,
}


fn route_marginal_slope_deviation_blocks(
    main_linkwiggle: Option<&LinkWiggleFormulaSpec>,
    logslope_linkwiggle: Option<&LinkWiggleFormulaSpec>,
) -> Result<MarginalSlopeDeviationRouting, String> {
    Ok(MarginalSlopeDeviationRouting {
        score_warp: logslope_linkwiggle
            .map(deviation_block_config_from_formula_linkwiggle)
            .transpose()?,
        link_dev: main_linkwiggle
            .map(deviation_block_config_from_formula_linkwiggle)
            .transpose()?,
    })
}


fn cli_frailty_kind(
    frailty_kind: Option<FrailtyKindArg>,
) -> Option<gam::config_resolve::CliFrailtyKind> {
    frailty_kind.map(|kind| match kind {
        FrailtyKindArg::GaussianShift => gam::config_resolve::CliFrailtyKind::GaussianShift,
        FrailtyKindArg::HazardMultiplier => gam::config_resolve::CliFrailtyKind::HazardMultiplier,
    })
}


fn cli_hazard_loading(
    hazard_loading: Option<HazardLoadingArg>,
) -> Option<gam::config_resolve::CliHazardLoading> {
    hazard_loading.map(|loading| match loading {
        HazardLoadingArg::Full => gam::config_resolve::CliHazardLoading::Full,
        HazardLoadingArg::LoadedVsUnloaded => {
            gam::config_resolve::CliHazardLoading::LoadedVsUnloaded
        }
    })
}


fn latent_cloglog_state_from_frailty_spec(
    frailty: &gam::families::lognormal_kernel::FrailtySpec,
    context: &str,
) -> Result<gam::types::LatentCLogLogState, String> {
    let sigma = match frailty {
        gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(sigma),
            loading: gam::families::lognormal_kernel::HazardLoading::Full,
        } => *sigma,
        gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(_),
            loading,
        } => {
            return Err(format!(
                "{context} requires --hazard-loading full, got {loading:?}"
            ));
        }
        gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
            sigma_fixed: None,
            ..
        } => {
            return Err(format!("{context} currently requires a fixed --frailty-sd"));
        }
        gam::families::lognormal_kernel::FrailtySpec::GaussianShift { .. } => {
            return Err(format!(
                "{context} requires --frailty-kind hazard-multiplier"
            ));
        }
        gam::families::lognormal_kernel::FrailtySpec::None => {
            return Err(format!(
                "{context} requires an explicit frailty specification"
            ));
        }
    };
    gam::types::LatentCLogLogState::new(sigma)
        .map_err(|e| format!("invalid latent-cloglog frailty sigma: {e}"))
}


fn fit_frailty_spec_from_args(
    args: &FitArgs,
    context: &str,
) -> Result<gam::families::lognormal_kernel::FrailtySpec, String> {
    gam::config_resolve::resolve_cli_frailty_spec(
        cli_frailty_kind(args.frailty_kind),
        args.frailty_sd,
        cli_hazard_loading(args.hazard_loading),
        context,
    )
}


fn fit_frailty_spec_from_survival_args(
    args: &SurvivalArgs,
    context: &str,
) -> Result<gam::families::lognormal_kernel::FrailtySpec, String> {
    gam::config_resolve::resolve_cli_frailty_spec(
        cli_frailty_kind(args.frailty_kind),
        args.frailty_sd,
        cli_hazard_loading(args.hazard_loading),
        context,
    )
}


fn fixed_gaussian_shift_frailty_from_spec(
    frailty: &gam::families::lognormal_kernel::FrailtySpec,
    context: &str,
) -> Result<gam::families::lognormal_kernel::FrailtySpec, String> {
    match frailty {
        gam::families::lognormal_kernel::FrailtySpec::None => {
            Ok(gam::families::lognormal_kernel::FrailtySpec::None)
        }
        gam::families::lognormal_kernel::FrailtySpec::GaussianShift {
            sigma_fixed: Some(sigma),
        } => Ok(
            gam::families::lognormal_kernel::FrailtySpec::GaussianShift {
                sigma_fixed: Some(*sigma),
            },
        ),
        gam::families::lognormal_kernel::FrailtySpec::GaussianShift { sigma_fixed: None } => {
            Err(format!(
                "{context} currently requires a fixed GaussianShift sigma; learnable GaussianShift sigma is not implemented for the exact marginal-slope outer solver"
            ))
        }
        gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier { .. } => Err(format!(
            "{context} requires --frailty-kind gaussian-shift or no frailty"
        )),
    }
}


fn fixed_hazard_multiplier_from_saved_family(
    family: &FittedFamily,
) -> Result<(f64, gam::families::lognormal_kernel::HazardLoading), String> {
    match family.frailty() {
        Some(gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(sigma),
            loading,
        }) => Ok((*sigma, *loading)),
        Some(gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
            sigma_fixed: None,
            ..
        }) => Err("saved latent survival/binary model must store a concrete HazardMultiplier sigma in family_state.frailty".to_string()),
        Some(gam::families::lognormal_kernel::FrailtySpec::GaussianShift { .. })
        | Some(gam::families::lognormal_kernel::FrailtySpec::None)
        | None => Err(
            "saved latent survival/binary model requires a fixed HazardMultiplier frailty specification"
                .to_string(),
        ),
    }
}


fn build_bernoulli_marginal_slope_saved_model(
    formula: String,
    data_schema: DataSchema,
    logslope_formula: String,
    z_column: String,
    training_headers: Vec<String>,
    training_feature_ranges: Vec<(f64, f64)>,
    resolved_marginalspec: TermCollectionSpec,
    resolved_logslopespec: TermCollectionSpec,
    fit_result: UnifiedFitResult,
    p_marginal: usize,
    baseline_marginal: f64,
    baseline_logslope: f64,
    latent_z_normalization: SavedLatentZNormalization,
    latent_measure: LatentMeasureKind,
    latent_z_rank_int_calibration: Option<gam::families::bms::LatentZRankIntCalibration>,
    latent_z_conditional_calibration: Option<gam::families::bms::LatentZConditionalCalibration>,
    score_warp_runtime: Option<&DeviationRuntime>,
    link_dev_runtime: Option<&DeviationRuntime>,
    base_link: InverseLink,
    frailty: gam::families::lognormal_kernel::FrailtySpec,
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
            logslope_formula,
            z_column,
            resolved_marginalspec,
            resolved_logslopespec,
            fit_result,
            p_marginal,
            baseline_marginal,
            baseline_logslope,
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


fn resolve_bernoulli_marginal_slope_base_link(
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


fn build_transformation_normal_saved_model(
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


fn core_saved_fit_result(
    beta: Array1<f64>,
    lambdas: Array1<f64>,
    standard_deviation: f64,
    beta_covariance: Option<Array2<f64>>,
    beta_covariance_corrected: Option<Array2<f64>>,
    summary: SavedFitSummary,
) -> UnifiedFitResult {
    // Saved models are part of the stable inference contract. Reject non-finite
    // values at construction time so JSON cannot silently encode them as null.
    let summary = summary
        .validated()
        .expect("core_saved_fit_result called with non-finite summary metrics");
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
        let log_lambdas = lambdas.mapv(|v| v.max(1e-300).ln());
        // Do not export a synthetic/placeholder Hessian here. Saved fits built
        // from externally supplied summary/covariance data may provide covariance
        // for prediction, but HMC/NUTS whitening requires an explicit upstream
        // penalized Hessian from the fitter itself.
        let covariance_conditional = beta_covariance;
        let covariance_corrected = beta_covariance_corrected;
        let penalized_objective = summary.reml_score;
        UnifiedFitResult::try_from_parts(gam::estimate::UnifiedFitResultParts {
            blocks: vec![gam::estimate::FittedBlock {
                beta: beta.clone(),
                role: gam::estimate::BlockRole::Mean,
                edf: 0.0,
                lambdas: lambdas.clone(),
            }],
            log_lambdas,
            lambdas,
            likelihood_family: summary.likelihood_family,
            likelihood_scale: summary.likelihood_scale,
            log_likelihood_normalization: summary.log_likelihood_normalization,
            log_likelihood: summary.log_likelihood,
            deviance: summary.deviance,
            reml_score: summary.reml_score,
            stable_penalty_term: summary.stable_penalty_term,
            penalized_objective,
            outer_iterations: summary.iterations,
            outer_converged: matches!(summary.pirls_status, gam::pirls::PirlsStatus::Converged),
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
                ..Default::default()
            },
            inner_cycles: 0,
        })
        .expect("core_saved_fit_result called with invalid fit metrics")
    }
}


fn family_noise_parameter(fit: &UnifiedFitResult, family: LikelihoodSpec) -> Option<f64> {
    match family.response {
        // The generative `gaussian_scale` slot carries the *dispersion* φ for
        // Tweedie; the variance power `p` is already read from the family spec by
        // `NoiseModel::from_likelihood`, so emitting `p` here drew responses with
        // φ = p (≈1.5) regardless of the data. φ is estimated jointly with the
        // mean (issue #771), so the authoritative value is the fit's scale
        // metadata, falling back to a unit dispersion only if the fit recorded
        // none.
        ResponseFamily::Tweedie { .. } => fit.likelihood_scale.fixed_phi().or(Some(1.0)),
        ResponseFamily::NegativeBinomial { theta, .. } => Some(theta),
        // Beta precision φ is estimated jointly with the mean (issue #567), so
        // the authoritative value is the fit's scale metadata, not the seed φ on
        // the original family spec. Fall back to the spec φ only if the fit did
        // not record an estimated/fixed dispersion.
        ResponseFamily::Beta { phi } => fit.likelihood_scale.fixed_phi().or(Some(phi)),
        ResponseFamily::Gamma => fit
            .likelihood_scale
            .gamma_shape()
            .or(Some(fit.standard_deviation)),
        _ => Some(fit.standard_deviation),
    }
}


#[derive(Clone)]
struct SavedFitSummary {
    likelihood_family: Option<LikelihoodSpec>,
    likelihood_scale: LikelihoodScaleMetadata,
    log_likelihood_normalization: LogLikelihoodNormalization,
    log_likelihood: f64,
    iterations: usize,
    finalgrad_norm: f64,
    pirls_status: gam::pirls::PirlsStatus,
    deviance: f64,
    stable_penalty_term: f64,
    max_abs_eta: f64,
    reml_score: f64,
}


impl SavedFitSummary {
    fn validated(self) -> Result<Self, String> {
        ensure_finite_scalar("fit_result.log_likelihood", self.log_likelihood)?;
        ensure_finite_scalar("fit_result.finalgrad_norm", self.finalgrad_norm)?;
        ensure_finite_scalar("fit_result.deviance", self.deviance)?;
        ensure_finite_scalar("fit_result.stable_penalty_term", self.stable_penalty_term)?;
        ensure_finite_scalar("fit_result.max_abs_eta", self.max_abs_eta)?;
        ensure_finite_scalar("fit_result.reml_score", self.reml_score)?;
        Ok(self)
    }

    fn from_blockwise_fit(fit: &gam::estimate::UnifiedFitResult) -> Result<Self, String> {
        let stable_penalty_term = fit.stable_penalty_term;
        let max_abs_eta = fit
            .block_states
            .iter()
            .flat_map(|b| b.eta.iter())
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        Self {
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
            // a single "StalledAtValidMinimum" bucket, silently relabeling
            // genuinely broken fits as healthy for any downstream consumer that
            // gates on status. The bool is itself just a projection of this
            // field (`outer_converged == matches!(status, Converged)`), so the
            // status is strictly more informative.
            pirls_status: fit.pirls_status,
            deviance: fit.deviance,
            stable_penalty_term,
            max_abs_eta,
            reml_score: fit.reml_score,
        }
        .validated()
    }

    fn from_survivalworking_summary(
        summary: &gam::pirls::WorkingModelPirlsResult,
        state: &gam::pirls::WorkingState,
    ) -> Result<Self, String> {
        let reml_score = 0.5 * (state.deviance + state.penalty_term);
        Self {
            likelihood_family: Some(LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            )),
            likelihood_scale: LikelihoodScaleMetadata::Unspecified,
            log_likelihood_normalization: LogLikelihoodNormalization::UserProvided,
            log_likelihood: state.log_likelihood,
            iterations: summary.iterations,
            finalgrad_norm: summary.lastgradient_norm,
            pirls_status: summary.status,
            deviance: state.deviance,
            stable_penalty_term: state.penalty_term,
            max_abs_eta: summary.max_abs_eta,
            reml_score,
        }
        .validated()
    }
}


use gam::estimate::{ensure_finite_scalar, validate_all_finite};


fn termspec_has_bounded_terms(spec: &TermCollectionSpec) -> bool {
    spec.linear_terms.iter().any(|term| {
        matches!(
            term.coefficient_geometry,
            LinearCoefficientGeometry::Bounded { .. }
        )
    })
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AloRefitRoute {
    StandardGam,
    UnifiedTermCollection,
}


fn alo_refit_route_for_termspec(spec: &TermCollectionSpec) -> AloRefitRoute {
    if termspec_has_bounded_terms(spec) {
        AloRefitRoute::UnifiedTermCollection
    } else {
        AloRefitRoute::StandardGam
    }
}


fn spatial_basiswarning_family_and_cols(term: &SmoothTermSpec) -> Option<(&'static str, &[usize])> {
    spatial_basiswarning_family_and_cols_basis(&term.basis)
}


fn spatial_basiswarning_family_and_cols_basis(
    basis: &SmoothBasisSpec,
) -> Option<(&'static str, &[usize])> {
    match basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            spatial_basiswarning_family_and_cols_basis(inner)
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            spatial_basiswarning_family_and_cols_basis(smooth)
        }
        SmoothBasisSpec::ThinPlate { feature_cols, .. } => Some(("thinplate/tps", feature_cols)),
        SmoothBasisSpec::Sphere { feature_cols, .. } => Some(("sphere/sos", feature_cols)),
        SmoothBasisSpec::ConstantCurvature { feature_cols, .. } => {
            Some(("constant_curvature", feature_cols))
        }
        SmoothBasisSpec::Matern { feature_cols, .. } => Some(("matern", feature_cols)),
        SmoothBasisSpec::MeasureJet { feature_cols, .. } => Some(("measurejet", feature_cols)),
        SmoothBasisSpec::Duchon { feature_cols, .. } => Some(("duchon", feature_cols)),
        SmoothBasisSpec::BSpline1D { .. }
        | SmoothBasisSpec::Pca { .. }
        | SmoothBasisSpec::TensorBSpline { .. }
        | SmoothBasisSpec::FactorSmooth { .. } => None,
    }
}


fn collect_spatial_smooth_usagewarnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let mut grouped: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    for term in &spec.smooth_terms {
        let Some((family, feature_cols)) = spatial_basiswarning_family_and_cols(term) else {
            continue;
        };
        if feature_cols.len() != 1 {
            continue;
        }
        let col = feature_cols[0];
        let featurename = headers
            .get(col)
            .cloned()
            .unwrap_or_else(|| format!("#{col}"));
        grouped.entry(family).or_default().push(featurename);
    }

    grouped
        .into_iter()
        .filter_map(|(family, cols)| {
            if cols.len() < 2 {
                return None;
            }
            // `spatial_basiswarning_family_and_cols` returns one of these four
            // family strings; any other value is filtered out by returning None.
            let example = match family {
                "thinplate/tps" => format!("thinplate({})", cols.join(", ")),
                "matern" => format!("matern({})", cols.join(", ")),
                "duchon" => format!("duchon({})", cols.join(", ")),
                "sphere/sos" => format!("sphere({})", cols.join(", ")),
                _ => return None,
            };
            let bad_example = match family {
                "thinplate/tps" => cols
                    .iter()
                    .map(|col| format!("s({col}, type=tps)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                "matern" => cols
                    .iter()
                    .map(|col| format!("s({col}, type=matern)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                "duchon" => cols
                    .iter()
                    .map(|col| format!("s({col}, type=duchon)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                "sphere/sos" => cols
                    .iter()
                    .map(|col| format!("s({col}, type=sphere)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                _ => return None,
            };
            Some(format!(
                "{label}: detected {} separate 1D {family} spatial smooths over [{}]. These build unrelated additive 1D smooths, not one shared spatial manifold. TIP: if you intended one spatial surface, replace `{bad_example}` with one multivariate term such as `{example}`.",
                cols.len(),
                cols.join(", "),
            ))
        })
        .collect()
}


fn collect_linear_smooth_overlapwarnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let linear_by_col = spec
        .linear_terms
        .iter()
        .map(|term| (term.feature_col, term.name.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut warnings = Vec::new();
    for smooth in &spec.smooth_terms {
        let overlaps = smooth_term_feature_cols(smooth)
            .into_iter()
            .filter_map(|col| {
                linear_by_col.get(&col).map(|linearname| {
                    let featurename = headers
                        .get(col)
                        .cloned()
                        .unwrap_or_else(|| format!("#{col}"));
                    (featurename, (*linearname).to_string())
                })
            })
            .collect::<Vec<_>>();
        if overlaps.is_empty() {
            continue;
        }
        let overlap_features = overlaps
            .iter()
            .map(|(featurename, _)| featurename.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let linear_terms = overlaps
            .iter()
            .map(|(_, linearname)| format!("linear({linearname})"))
            .collect::<Vec<_>>()
            .join(" + ");
        warnings.push(format!(
            "{label}: feature(s) [{overlap_features}] appear both in smooth term `{}` and explicit linear term(s) `{linear_terms}`. The fit now residualizes the smooth against the intercept and those overlapping linear columns, so the smooth contributes only the nonlinear remainder on those variables. This changes the term decomposition and interpretation.",
            smooth.name
        ));
    }
    warnings
}


fn collect_hierarchical_smooth_overlapwarnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let feature_label = |col: usize| {
        headers
            .get(col)
            .cloned()
            .unwrap_or_else(|| format!("#{col}"))
    };
    let join_feature_labels = |cols: &[usize]| {
        cols.iter()
            .map(|&col| feature_label(col))
            .collect::<Vec<_>>()
            .join(", ")
    };

    let SmoothStructureAnalysis {
        ownership_order,
        term_feature_cols,
        term_owners,
        ..
    } = analyze_smooth_ownership(&spec.smooth_terms);

    let mut warnings = Vec::new();
    for &target_idx in &ownership_order {
        let owners = &term_owners[target_idx];
        if owners.is_empty() {
            continue;
        }
        let target = &spec.smooth_terms[target_idx];
        let target_features = join_feature_labels(&term_feature_cols[target_idx]);
        let owner_descriptions = owners
            .iter()
            .map(|&owner_idx| {
                format!(
                    "`{}` over [{}]",
                    spec.smooth_terms[owner_idx].name,
                    join_feature_labels(&term_feature_cols[owner_idx]),
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        warnings.push(format!(
            "{label}: smooth term `{}` over [{target_features}] overlaps nested or duplicate smooth term(s) {}. The fit uses automatic hierarchical ownership: those higher-priority smooth term(s) keep any shared realized subspace, and `{}` is residualized against that overlap before fitting.",
            target.name,
            owner_descriptions,
            target.name,
        ));
    }
    warnings
}


fn collect_smooth_structure_warnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let mut warnings = collect_spatial_smooth_usagewarnings(spec, headers, label);
    warnings.extend(collect_linear_smooth_overlapwarnings(spec, headers, label));
    warnings.extend(collect_hierarchical_smooth_overlapwarnings(
        spec, headers, label,
    ));
    warnings
}


fn emit_smooth_structure_warnings(stage: &str, warnings: &[String]) {
    for warning in warnings {
        cli_err!("WARNING [{stage}]: {warning}");
    }
}


/// Build anisotropic spatial-geometry report rows from an optional resolved spec.
fn build_anisotropic_scales_rows(
    spec: Option<&TermCollectionSpec>,
) -> Vec<report::AnisotropicScalesRow> {
    use gam::smooth::{get_spatial_aniso_log_scales, get_spatial_length_scale};
    let Some(spec) = spec else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let Some(eta) = get_spatial_aniso_log_scales(spec, term_idx) else {
            continue;
        };
        if eta.is_empty() {
            continue;
        }
        let ls = get_spatial_length_scale(spec, term_idx);
        let axes = eta
            .iter()
            .enumerate()
            .map(|(a, &eta_a)| {
                let (length_a, kappa_a) = if let Some(ls) = ls {
                    (Some(ls * (-eta_a).exp()), Some((1.0 / ls) * eta_a.exp()))
                } else {
                    (None, None)
                };
                (a, eta_a, length_a, kappa_a)
            })
            .collect();
        rows.push(report::AnisotropicScalesRow {
            term_name: term.name.clone(),
            global_length_scale: ls,
            axes,
        });
    }
    rows
}


/// Build measure-jet spectrum report rows from a saved (frozen) spec alone:
/// realized band + spec order, no per-scale λ̂s (those need the rebuilt
/// design's penalty layout). Used when the report runs without a dataset.
fn measure_jet_spectrum_rows_from_spec(
    spec: Option<&TermCollectionSpec>,
) -> Vec<report::MeasureJetSpectrumRow> {
    let Some(spec) = spec else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for term in &spec.smooth_terms {
        let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &term.basis else {
            continue;
        };
        let Some(frozen) = mj.frozen_quadrature.as_ref() else {
            continue;
        };
        let (Some(&eps_min), Some(&eps_max)) = (frozen.eps_band.first(), frozen.eps_band.last())
        else {
            continue;
        };
        rows.push(report::MeasureJetSpectrumRow {
            term_name: term.name.clone(),
            eps_min,
            eps_max,
            n_scales: frozen.eps_band.len(),
            length_scale: mj.length_scale,
            spec_order_s: mj.order_s,
            per_scale: Vec::new(),
            implied_order: None,
        });
    }
    rows
}


/// Implied continuous order from a measure-jet raw-form per-scale λ spectrum:
/// ŝ = −½ · (least-squares slope of ln λ̂_ℓ on ln ε_ℓ). `None` unless at
/// least two scales carry finite positive (ε_ℓ, λ̂_ℓ) and the band has
/// nonzero log-spread.
fn measure_jet_implied_order(per_scale: &[(f64, f64)]) -> Option<f64> {
    let pts: Vec<(f64, f64)> = per_scale
        .iter()
        .filter(|&&(eps, lam)| eps.is_finite() && eps > 0.0 && lam.is_finite() && lam > 0.0)
        .map(|&(eps, lam)| (eps.ln(), lam.ln()))
        .collect();
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let xbar = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let ybar = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx = pts.iter().map(|p| (p.0 - xbar).powi(2)).sum::<f64>();
    if sxx <= 0.0 {
        return None;
    }
    let sxy = pts.iter().map(|p| (p.0 - xbar) * (p.1 - ybar)).sum::<f64>();
    let s_hat = -0.5 * (sxy / sxx);
    s_hat.is_finite().then_some(s_hat)
}


/// Print learned per-axis spatial anisotropy for spatial terms to stdout.
fn print_spatial_aniso_scales(spec: &TermCollectionSpec) {
    use gam::smooth::{get_spatial_aniso_log_scales, get_spatial_length_scale};
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let Some(eta) = get_spatial_aniso_log_scales(spec, term_idx) else {
            continue;
        };
        if eta.is_empty() {
            continue;
        }
        let ls = get_spatial_length_scale(spec, term_idx);
        match ls {
            Some(ls) => cli_out!(
                "[spatial-kappa] term {} (\"{}\"): anisotropic length scales (global length_scale={:.4})",
                term_idx,
                term.name,
                ls
            ),
            None => cli_out!(
                "[spatial-kappa] term {} (\"{}\"): pure Duchon shape anisotropy",
                term_idx,
                term.name
            ),
        }
        for (a, &eta_a) in eta.iter().enumerate() {
            if let Some(ls) = ls {
                let length_a = ls * (-eta_a).exp();
                let kappa_a = (1.0 / ls) * eta_a.exp();
                cli_out!(
                    "  axis {}: eta={:+.4}, length={:.4}, kappa={:.4}",
                    a,
                    eta_a,
                    length_a,
                    kappa_a
                );
            } else {
                cli_out!("  axis {}: eta={:+.4}", a, eta_a);
            }
        }
    }
}


fn compact_saved_multiblock_fit_result(
    blocks: Vec<gam::estimate::FittedBlock>,
    lambdas: Array1<f64>,
    standard_deviation: f64,
    beta_covariance: Option<Array2<f64>>,
    beta_covariance_corrected: Option<Array2<f64>>,
    geometry: Option<gam::estimate::FitGeometry>,
    summary: SavedFitSummary,
) -> UnifiedFitResult {
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
    );
    fit_result.blocks = blocks;
    if let Some(geom) = geometry {
        if let Some(inf) = fit_result.inference.as_mut() {
            inf.penalized_hessian = geom.penalized_hessian.clone();
            inf.working_weights = geom.working_weights.clone();
            inf.working_response = geom.working_response.clone();
        }
        fit_result.geometry = Some(geom);
    }
    fit_result
}


fn compact_saved_survival_location_scale_fit_result(
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
    );
    apply_inverse_link_state_to_fit_result(&mut fit_result, inverse_link);
    fit_result.artifacts.survival_link_wiggle_knots =
        fit.artifacts.survival_link_wiggle_knots.clone();
    fit_result.artifacts.survival_link_wiggle_degree = fit.artifacts.survival_link_wiggle_degree;
    Ok(fit_result)
}


fn write_model_json(path: &Path, model: &SavedModel) -> Result<(), String> {
    model.save_to_path(path)?;
    cli_out!("saved model: {}", path.display());
    Ok(())
}


fn write_payload_json(path: &Path, payload: FittedModelPayload) -> Result<(), String> {
    let model = SavedModel::from_payload(payload);
    write_model_json(path, &model)
}


fn print_inference_summary(notes: &[String]) {
    if notes.is_empty() {
        return;
    }
    cli_err!("Auto-discovery summary:");
    for note in notes {
        cli_err!("  - {}", note);
    }
}


fn set_saved_offset_columns(
    payload: &mut FittedModelPayload,
    offset_column: Option<String>,
    noise_offset_column: Option<String>,
) {
    payload.offset_column = offset_column;
    payload.noise_offset_column = noise_offset_column;
}


fn collect_term_column_names(terms: &[ParsedTerm], out: &mut BTreeSet<String>) {
    // Delegate to the single shared authority on the formula→columns walk
    // (`s(x, by=g)`'s `by` column is included there) so the fit-time required
    // columns, the predict-time required columns, and the PyFFI surface all
    // agree.
    parsed_term_column_names(terms, out);
}


fn required_columns_for_formula(parsed: &ParsedFormula) -> Result<Vec<String>, String> {
    let mut out = BTreeSet::<String>::new();
    if let Some((entry, exit, event)) = parse_surv_response(&parsed.response)? {
        if let Some(entry) = entry {
            out.insert(entry);
        }
        out.insert(exit);
        out.insert(event);
    } else if let Some((left, right, event)) =
        parse_surv_interval_response(&parsed.response)?
    {
        out.insert(left);
        out.insert(right);
        out.insert(event);
    } else {
        out.insert(parsed.response.clone());
    }
    collect_term_column_names(&parsed.terms, &mut out);
    for surface in &parsed.logslope_surfaces {
        out.insert(surface.z_column.clone());
        collect_term_column_names(&surface.terms, &mut out);
    }
    Ok(out.into_iter().collect())
}


fn merge_required_columns(target: &mut BTreeSet<String>, cols: Vec<String>) {
    target.extend(cols);
}


fn required_columns_for_fit(args: &FitArgs, parsed: &ParsedFormula) -> Result<Vec<String>, String> {
    let mut required = BTreeSet::<String>::new();
    merge_required_columns(&mut required, required_columns_for_formula(parsed)?);

    if let Some(noise_formula_raw) = args.predict_noise.as_deref() {
        let (_, parsed_noise) = parse_matching_auxiliary_formula(
            noise_formula_raw,
            &parsed.response,
            "--predict-noise",
        )?;
        merge_required_columns(&mut required, required_columns_for_formula(&parsed_noise)?);
    }

    if let Some(logslope_formula_raw) = args.logslope_formula.as_deref() {
        let (_, parsed_logslope) = parse_matching_auxiliary_formula(
            logslope_formula_raw,
            &parsed.response,
            "--logslope-formula",
        )?;
        merge_required_columns(
            &mut required,
            required_columns_for_formula(&parsed_logslope)?,
        );
    }

    if let Some(z_column) = args.z_column.as_ref() {
        required.insert(z_column.clone());
    }
    if let Some(weights_column) = args.weights_column.as_ref() {
        required.insert(weights_column.clone());
    }
    if let Some(offset_column) = args.offset_column.as_ref() {
        required.insert(offset_column.clone());
    }
    if let Some(noise_offset_column) = args.noise_offset_column.as_ref() {
        required.insert(noise_offset_column.clone());
    }
    Ok(required.into_iter().collect())
}


/// Format a `Surv(...)` response expression, omitting the entry argument
/// when the right-censored shorthand `Surv(time, event)` is in use.
fn surv_response_expr(entry: Option<&str>, exit: &str, event: &str) -> String {
    match entry {
        Some(entry) => format!("Surv({entry}, {exit}, {event})"),
        None => format!("Surv({exit}, {event})"),
    }
}


fn required_columns_for_survival(
    args: &SurvivalArgs,
    parsed: &ParsedFormula,
) -> Result<Vec<String>, String> {
    let mut required = BTreeSet::<String>::new();
    if let Some(entry) = args.entry.as_deref() {
        required.insert(entry.to_string());
    }
    required.insert(args.exit.clone());
    required.insert(args.event.clone());
    merge_required_columns(&mut required, required_columns_for_formula(parsed)?);

    if let Some(noise_formula_raw) = args.predict_noise.as_deref() {
        let response_expr = surv_response_expr(args.entry.as_deref(), &args.exit, &args.event);
        let (_, parsed_noise) =
            parse_matching_auxiliary_formula(noise_formula_raw, &response_expr, "--predict-noise")?;
        merge_required_columns(&mut required, required_columns_for_formula(&parsed_noise)?);
    }

    if let Some(z_column) = args.z_column.as_ref() {
        required.insert(z_column.clone());
    }
    if let Some(weights_column) = args.weights_column.as_ref() {
        required.insert(weights_column.clone());
    }
    if let Some(offset_column) = args.offset_column.as_ref() {
        required.insert(offset_column.clone());
    }
    if let Some(noise_offset_column) = args.noise_offset_column.as_ref() {
        required.insert(noise_offset_column.clone());
    }
    Ok(required.into_iter().collect())
}


fn load_dataset_projected(
    path: &Path,
    requested_columns: &[String],
) -> Result<Dataset, gam::inference::data::DataError> {
    load_dataset_auto_projected(path, requested_columns)
}


fn load_datasetwith_model_schema(path: &Path, model: &SavedModel) -> Result<Dataset, String> {
    load_datasetwith_model_schema_extra(path, model, &[])
}


/// Load a dataset for a *post-fit diagnostic* command (diagnose / sample /
/// report) against a fitted model's schema.
///
/// Unlike prediction, diagnostics need the observed response column: residuals,
/// R², posterior likelihoods, and leave-one-out are all statements *about* it.
/// The prediction loader deliberately drops a standard GAM's bare response
/// (#840 / #864), so this variant folds the model's diagnostic-required
/// response back in via [`SavedModel::diagnostic_extra_columns`]. Routing every
/// diagnostic command through here makes it structurally impossible to silently
/// drop the response — the #864 / #882 / #883 failure mode — rather than relying
/// on each command to remember an `extra_required` argument.
fn load_datasetwith_model_schema_for_diagnostics(
    path: &Path,
    model: &SavedModel,
) -> Result<Dataset, String> {
    let extras = model.diagnostic_extra_columns()?;
    load_datasetwith_model_schema_extra(path, model, &extras)
}


/// Load a new-data file against a fitted model's schema, keeping only the
/// columns the model references (plus any `extra_required` ones a caller knows
/// it will resolve by name, e.g. a `--offset-column` override that differs from
/// the model's saved offset).
///
/// A prediction file commonly carries extra ID / label / grouping columns the
/// formula never names; encoding those against the training schema would
/// strict-validate an unrelated categorical and abort on a held-out level
/// (#840). The projected loader selects just the model's input columns (and the
/// extras), erroring only when a genuinely required one is absent and ignoring
/// the rest — matching mgcv / glm semantics and the PyFFI predict path.
fn load_datasetwith_model_schema_extra(
    path: &Path,
    model: &SavedModel,
    extra_required: &[String],
) -> Result<Dataset, String> {
    let schema = model.require_data_schema()?;
    let policy =
        UnseenCategoryPolicy::encode_unknown_for_columns(model.random_effect_group_columns());
    let mut requested: Vec<String> = model
        .prediction_required_columns()?
        .into_iter()
        .collect::<Vec<_>>();
    requested.extend(extra_required.iter().cloned());
    load_dataset_auto_with_schema_projected(path, schema, policy, &requested).map_err(String::from)
}


/// Canonical family name for a CLI `--family` selection.
///
/// This is the one place that maps the closed `FamilyArg` enum onto the
/// string vocabulary understood by the canonical resolver
/// (`gam::resolve_family` in `src/solver/workflow.rs`). `Auto` returns `None`
/// so the resolver runs response inference; every concrete variant returns the
/// exact name the resolver matches, preserving its pinned/unpinned link
/// semantics (e.g. `binomial-logit` pins the link, `gaussian`/`poisson`/`gamma`
/// leave it open to refinement by a `link(...)` choice).
fn family_arg_canonical_name(arg: FamilyArg) -> Option<&'static str> {
    match arg {
        FamilyArg::Auto => None,
        FamilyArg::Gaussian => Some("gaussian"),
        FamilyArg::BinomialLogit => Some("binomial-logit"),
        FamilyArg::BinomialProbit => Some("binomial-probit"),
        FamilyArg::BinomialCloglog => Some("binomial-cloglog"),
        FamilyArg::LatentCloglogBinomial => Some("latent-cloglog-binomial"),
        FamilyArg::PoissonLog => Some("poisson"),
        FamilyArg::NegativeBinomial => Some("negative-binomial"),
        FamilyArg::GammaLog => Some("gamma"),
        FamilyArg::Tweedie => Some("tweedie"),
        FamilyArg::Beta => Some("beta"),
        FamilyArg::RoystonParmar => Some("royston-parmar"),
        FamilyArg::TransformationNormal => Some("transformation-normal"),
    }
}


/// CLI adapter over the canonical family resolver.
///
/// The fit-routing contract — explicit family vs link-implied family, the
/// SAS/Beta-Logistic links, negative-binomial `theta`, and response
/// auto-inference — lives once in `gam::resolve_family`. The CLI keeps only the
/// surface-specific concerns: translating the typed `FamilyArg` into the
/// canonical name and enforcing the CLI flag rule that
/// `--negative-binomial-theta` is meaningful exclusively with
/// `--family negative-binomial`.
///
/// The user's `link(sas_init=...)` / `link(beta_logistic_init=...)` state is
/// not threaded through this resolver: family resolution produces the
/// link-only placeholder, and the standard fit picks up the actual initial
/// state from `FitOptions.sas_link` (see `effective_sas_link_for_family` in
/// `src/solver/estimate.rs`), which overrides the family-embedded link. Keeping
/// the resolver link-state-free leaves a single, narrow family-routing contract
/// shared verbatim with the workflow and PyFFI surfaces.
fn resolve_family(
    arg: FamilyArg,
    negative_binomial_theta: Option<f64>,
    link_choice: Option<LinkChoice>,
    y: ArrayView1<'_, f64>,
    y_kind: ResponseColumnKind,
    response_name: &str,
) -> Result<LikelihoodSpec, String> {
    if negative_binomial_theta.is_some() && !matches!(arg, FamilyArg::NegativeBinomial) {
        return Err("--negative-binomial-theta requires --family negative-binomial".to_string());
    }
    gam::resolve_family(
        family_arg_canonical_name(arg),
        negative_binomial_theta,
        link_choice.as_ref(),
        y,
        y_kind,
        response_name,
    )
}


fn inverse_link_from_fitted_link_state(state: &FittedLinkState) -> Option<InverseLink> {
    match state {
        FittedLinkState::Standard(Some(link)) => Some(InverseLink::Standard(*link)),
        FittedLinkState::Standard(None) => None,
        FittedLinkState::LatentCLogLog { state } => Some(InverseLink::LatentCLogLog(*state)),
        FittedLinkState::Sas { state, .. } => Some(InverseLink::Sas(*state)),
        FittedLinkState::BetaLogistic { state, .. } => Some(InverseLink::BetaLogistic(*state)),
        FittedLinkState::Mixture { state, .. } => Some(InverseLink::Mixture(state.clone())),
    }
}


fn resolve_binomial_inverse_link_for_fit(
    family: LikelihoodSpec,
    effective_link: LinkFunction,
    mixture_linkspec: Option<&MixtureLinkSpec>,
    context: &str,
) -> Result<InverseLink, String> {
    if !family.is_binomial() {
        return Err(format!(
            "{context} is only available for binomial links, got {}",
            family.name()
        ));
    }
    match &family.link {
        InverseLink::Standard(StandardLink::Logit) => {
            let spec = mixture_linkspec
                .ok_or_else(|| format!("{context} requires link(type=blended(...))"))?;
            let state = state_fromspec(spec)
                .map_err(|e| format!("invalid blended link configuration: {e}"))?;
            Ok(InverseLink::Mixture(state))
        }
        // `resolve_family` already upgrades Sas / BetaLogistic to their
        // state-bearing variants; we only need to forward them here.
        InverseLink::Sas(state) => Ok(InverseLink::Sas(*state)),
        InverseLink::BetaLogistic(state) => Ok(InverseLink::BetaLogistic(*state)),
        InverseLink::Standard(StandardLink::CLogLog) => Err(format!(
            "{context} does not construct latent-cloglog links directly; use the latent-cloglog family path with explicit frailty"
        )),
        InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::Identity)
        | InverseLink::Standard(StandardLink::Log)
        | InverseLink::LatentCLogLog(_)
        | InverseLink::Mixture(_) => Ok(InverseLink::Standard(
            gam::config_resolve::effective_link_to_standard(effective_link, context)?,
        )),
    }
}


fn binomial_mean_linkwiggle_supports_family(
    family: &LikelihoodSpec,
    link_choice: Option<&LinkChoice>,
) -> bool {
    let standard_binomial = family.is_binomial()
        && matches!(
            &family.link,
            InverseLink::Standard(StandardLink::Logit)
                | InverseLink::Standard(StandardLink::Probit)
                | InverseLink::Standard(StandardLink::CLogLog)
        );
    standard_binomial
        && !link_choice.is_some_and(|choice| matches!(choice.mode, LinkMode::Flexible))
}


fn is_binary_response(y: ArrayView1<'_, f64>) -> bool {
    if y.is_empty() {
        return false;
    }
    y.iter()
        .all(|v| (*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
}


/// Project the CLI's `EncodedDataset` column-kind tag onto the
/// [`ResponseColumnKind`] consumed by the family layer. Mirrors the helper
/// of the same name in `workflow.rs` — having two tiny copies (one per
/// crate-internal entry point) is cleaner than threading the ingest enum
/// itself into the types layer.
fn response_column_kind_for_dataset(ds: &Dataset, y_col: usize) -> ResponseColumnKind {
    match ds.column_kinds.get(y_col) {
        Some(ColumnKindTag::Categorical) => ResponseColumnKind::Categorical {
            levels: ds
                .schema
                .columns
                .get(y_col)
                .map(|sc| sc.levels.clone())
                .unwrap_or_default(),
        },
        Some(ColumnKindTag::Binary) => ResponseColumnKind::Binary,
        Some(ColumnKindTag::Continuous) | None => ResponseColumnKind::Numeric,
    }
}


fn build_model_summary(
    design: &gam::smooth::TermCollectionDesign,
    spec: &TermCollectionSpec,
    fit: &UnifiedFitResult,
    family: LikelihoodSpec,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> ModelSummary {
    const CONTINUOUS_ORDER_EPS: f64 = 1e-12;
    let se = fit
        .beta_standard_errors_corrected()
        .or(fit.beta_standard_errors());
    let cov_forwald = fit.beta_covariance_corrected().or(fit.beta_covariance());
    let scale_is_estimated = matches!(
        family.response,
        ResponseFamily::Gaussian | ResponseFamily::Gamma
    );
    let residual_df = (y.len() as f64 - fit.edf_total().unwrap_or(fit.beta.len() as f64)).max(1.0);
    let two_sided_parametric_p = |z: f64| -> Option<f64> {
        if !z.is_finite() {
            return None;
        }
        if scale_is_estimated {
            let dist = StudentsT::new(0.0, 1.0, residual_df).ok()?;
            Some((2.0 * (1.0 - dist.cdf(z.abs()))).clamp(0.0, 1.0))
        } else {
            Some((2.0 * (1.0 - normal_cdf(z.abs()))).clamp(0.0, 1.0))
        }
    };

    let nullmu = match family.response {
        ResponseFamily::Gaussian => {
            let wsum = weights.iter().copied().sum::<f64>().max(1e-12);
            let ybar = y
                .iter()
                .zip(weights.iter())
                .map(|(&yy, &ww)| yy * ww)
                .sum::<f64>()
                / wsum;
            Array1::from_elem(y.len(), ybar)
        }
        ResponseFamily::Binomial => {
            let wsum = weights.iter().copied().sum::<f64>().max(1e-12);
            let p = y
                .iter()
                .zip(weights.iter())
                .map(|(&yy, &ww)| yy * ww)
                .sum::<f64>()
                / wsum;
            Array1::from_elem(y.len(), p)
        }
        ResponseFamily::RoystonParmar => Array1::from_elem(y.len(), 0.0),
        ResponseFamily::Poisson
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Beta { .. }
        | ResponseFamily::Gamma => {
            let wsum = weights.iter().copied().sum::<f64>().max(1e-12);
            let mean = y
                .iter()
                .zip(weights.iter())
                .map(|(&yy, &ww)| yy * ww)
                .sum::<f64>()
                / wsum;
            let baseline = match family.response {
                ResponseFamily::Poisson => mean.max(0.0),
                ResponseFamily::Beta { .. } => {
                    mean.clamp(gam::pirls::BETA_MU_EPS, 1.0 - gam::pirls::BETA_MU_EPS)
                }
                _ => mean.max(1e-12),
            };
            Array1::from_elem(y.len(), baseline)
        }
    };
    let null_dev = {
        let null_likelihood = if family.is_royston_parmar() {
            gam::types::GlmLikelihoodSpec::canonical(gam::types::LikelihoodSpec::new(
                gam::types::ResponseFamily::Gaussian,
                gam::types::InverseLink::Standard(gam::types::StandardLink::Identity),
            ))
        } else {
            gam::types::GlmLikelihoodSpec::canonical(family.clone())
        };
        gam::pirls::calculate_deviance(y, &nullmu, &null_likelihood, weights)
    };
    let deviance_explained = if null_dev.is_finite() && null_dev > 0.0 {
        Some((1.0 - fit.deviance / null_dev).clamp(-9.0, 1.0))
    } else {
        None
    };

    let mut parametric_terms = Vec::<ParametricTermSummary>::new();
    let intercept_idx = design.intercept_range.start;
    let intercept_beta = fit.beta.get(intercept_idx).copied().unwrap_or(0.0);
    let intercept_se = se.and_then(|s| s.get(intercept_idx).copied());
    let interceptz = intercept_se.and_then(|s| (s > 0.0).then_some(intercept_beta / s));
    let intercept_p = interceptz.and_then(two_sided_parametric_p);
    parametric_terms.push(ParametricTermSummary {
        name: "Intercept".to_string(),
        estimate: intercept_beta,
        std_error: intercept_se,
        zvalue: interceptz,
        pvalue: intercept_p,
    });
    for (name, range) in &design.linear_ranges {
        let linear_meta = spec.linear_terms.iter().find(|term| term.name == *name);
        let geometry_label = match linear_meta {
            Some(LinearTermSpec {
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min,
                coefficient_max,
                ..
            }) => match (coefficient_min, coefficient_max) {
                (Some(lb), Some(ub)) => format!("{name} [coef in [{lb:.3}, {ub:.3}]]"),
                (Some(lb), None) => format!("{name} [coef >= {lb:.3}]"),
                (None, Some(ub)) => format!("{name} [coef <= {ub:.3}]"),
                (None, None) => name.clone(),
            },
            Some(LinearTermSpec {
                coefficient_geometry: LinearCoefficientGeometry::Bounded { min, max, prior },
                coefficient_min,
                coefficient_max,
                ..
            }) => {
                let prior_txt = match prior {
                    BoundedCoefficientPriorSpec::None => ", no-prior".to_string(),
                    BoundedCoefficientPriorSpec::Uniform => ", Uniform(log-Jacobian)".to_string(),
                    BoundedCoefficientPriorSpec::Beta { a, b } => {
                        format!(", Beta({a:.3},{b:.3})")
                    }
                };
                let constraint_txt = match (coefficient_min, coefficient_max) {
                    (Some(lb), Some(ub)) => format!(", coef in [{lb:.3}, {ub:.3}]"),
                    (Some(lb), None) => format!(", coef >= {lb:.3}"),
                    (None, Some(ub)) => format!(", coef <= {ub:.3}"),
                    (None, None) => String::new(),
                };
                format!("{name} [bounded {min:.3}..{max:.3}{prior_txt}{constraint_txt}]")
            }
            None => name.clone(),
        };
        for idx in range.start..range.end {
            let beta = fit.beta.get(idx).copied().unwrap_or(0.0);
            let se_i = se.and_then(|s| s.get(idx).copied());
            let z = se_i.and_then(|s| (s > 0.0).then_some(beta / s));
            let p = z.and_then(two_sided_parametric_p);
            let label = if range.end - range.start > 1 {
                format!("{geometry_label}[{}]", idx - range.start)
            } else {
                geometry_label.clone()
            };
            parametric_terms.push(ParametricTermSummary {
                name: label,
                estimate: beta,
                std_error: se_i,
                zvalue: z,
                pvalue: p,
            });
        }
    }

    let mut smooth_terms = Vec::<SmoothTermSummary>::new();
    let mut penalty_cursor = 0usize;
    for (name, _range) in &design.random_effect_ranges {
        let edf = fit
            .edf_by_block()
            .get(penalty_cursor)
            .copied()
            .unwrap_or(0.0);
        penalty_cursor += 1;
        // Random-effect smooths are variance-component tests on the boundary;
        // a naive coefficient Wald χ² p-value is anti-conservative, so only EDF is reported.
        let chi_sq_opt: Option<f64> = None;
        let ref_df = edf.max(0.0);
        let pvalue: Option<f64> = None;
        smooth_terms.push(SmoothTermSummary {
            name: name.clone(),
            edf,
            ref_df,
            chi_sq: chi_sq_opt,
            pvalue,
            continuous_order: None,
            basis_note: None,
        });
    }
    for term in &design.smooth.terms {
        let k = term.penalties_local.len();
        let term_penalty_start = penalty_cursor;
        let edf = fit
            .edf_by_block()
            .get(penalty_cursor..penalty_cursor + k)
            .map(|block| block.iter().sum::<f64>())
            .unwrap_or(0.0);
        penalty_cursor += k;
        let smooth_test = if term.shape == gam::smooth::ShapeConstraint::None {
            cov_forwald.and_then(|cov| {
                wood_smooth_test(SmoothTestInput {
                    beta: fit.beta.view(),
                    covariance: cov,
                    influence_matrix: fit.coefficient_influence(),
                    coeff_range: term.coeff_range.clone(),
                    edf,
                    nullspace_dim: term.nullspace_dims.iter().copied().sum::<usize>(),
                    residual_df,
                    scale: if scale_is_estimated {
                        SmoothTestScale::Estimated
                    } else {
                        SmoothTestScale::Known
                    },
                })
            })
        } else {
            None
        };
        let chi_sq_opt = smooth_test.as_ref().map(|test| test.statistic);
        let ref_df = smooth_test
            .as_ref()
            .map(|test| test.ref_df)
            .unwrap_or(edf.max(0.0));
        let pvalue = smooth_test.as_ref().map(|test| test.p_value);
        let continuous_order = if k == 3
            && term_penalty_start + 2 < fit.lambdas.len()
            && term_penalty_start + 2 < design.penaltyinfo.len()
        {
            // Unscaling identity for physical lambdas:
            //   S_tilde_k = S_k / c_k, and
            //   lambda_tilde_k * S_tilde_k = (lambda_tilde_k / c_k) * S_k.
            // Therefore physical lambda used by continuous-order diagnostics is
            //   lambda_k = lambda_tilde_k / c_k.
            let normalized_scale = |idx: usize| {
                let c = design.penaltyinfo[idx].penalty.normalization_scale;
                if c.is_finite() && c > 0.0 {
                    Some(c)
                } else {
                    None
                }
            };
            let lambda_tilde = [
                fit.lambdas[term_penalty_start],
                fit.lambdas[term_penalty_start + 1],
                fit.lambdas[term_penalty_start + 2],
            ];
            match (
                normalized_scale(term_penalty_start),
                normalized_scale(term_penalty_start + 1),
                normalized_scale(term_penalty_start + 2),
            ) {
                (Some(c0), Some(c1), Some(c2)) => Some(compute_continuous_smoothness_order(
                    lambda_tilde,
                    [c0, c1, c2],
                    CONTINUOUS_ORDER_EPS,
                )),
                _ => None,
            }
        } else {
            None
        };
        let basis_note = match &term.metadata {
            gam::basis::BasisMetadata::BSpline1D {
                auto_shrink_note, ..
            } => auto_shrink_note.clone(),
            _ => None,
        };
        smooth_terms.push(SmoothTermSummary {
            name: term.name.clone(),
            edf,
            ref_df,
            chi_sq: chi_sq_opt,
            pvalue,
            continuous_order,
            basis_note,
        });
    }

    ModelSummary {
        family: family.pretty_name().to_string(),
        deviance_explained,
        reml_score: Some(fit.reml_score),
        parametric_terms,
        smooth_terms,
    }
}


fn array2_to_nestedvec(a: &Array2<f64>) -> Vec<Vec<f64>> {
    a.axis_iter(Axis(0)).map(|row| row.to_vec()).collect()
}


fn covariance_from_model(
    model: &SavedModel,
    mode: CovarianceModeArg,
) -> Result<Array2<f64>, String> {
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| "model is missing canonical fit_result payload; refit".to_string())?;
    let cov = match mode {
        CovarianceModeArg::Corrected => fit.beta_covariance_corrected().or(fit.beta_covariance()),
        CovarianceModeArg::Conditional => fit.beta_covariance(),
    };
    if let Some(cov) = cov {
        return Ok(cov.clone());
    }
    if let Some(hessian) = fit.penalized_hessian() {
        let backend = PredictionCovarianceBackend::from_factorized_hessian(SymmetricMatrix::Dense(
            hessian.clone(),
        ))
        .map_err(|e| format!("failed to factor saved penalized Hessian for prediction: {e}"))?;
        let dim = backend.nrows();
        let mut eye = Array2::<f64>::zeros((dim, dim));
        for j in 0..dim {
            eye[[j, j]] = 1.0;
        }
        return backend.apply_columns(&eye).map_err(|e| {
            format!("failed to recover covariance from saved penalized Hessian: {e}")
        });
    }
    Err(
        "nonlinear posterior-mean prediction requires covariance or a saved penalized Hessian; refit"
            .to_string(),
    )
}


fn prediction_backend_from_model<'a>(
    model: &'a SavedModel,
    mode: CovarianceModeArg,
) -> Result<PredictionCovarianceBackend<'a>, String> {
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| "model is missing canonical fit_result payload; refit".to_string())?;
    let covariance = match mode {
        CovarianceModeArg::Corrected => fit.beta_covariance_corrected().or(fit.beta_covariance()),
        CovarianceModeArg::Conditional => fit.beta_covariance(),
    };
    if let Some(covariance) = covariance {
        return Ok(PredictionCovarianceBackend::from_dense(covariance.view()));
    }
    if let Some(hessian) = fit.penalized_hessian() {
        // Surface the factorization error directly rather than swallowing it
        // and reporting the generic "model is missing either ..." message.
        // When the saved Hessian exists but cannot be factored (indefinite,
        // numerically degenerate, etc.) the user needs to see *why*, not a
        // confused "refit" instruction that doesn't match the real fault.
        return PredictionCovarianceBackend::from_factorized_hessian(SymmetricMatrix::Dense(
            hessian.clone(),
        ))
        .map_err(|e| format!("failed to factor saved penalized Hessian for prediction: {e}"));
    }
    Err(
        "nonlinear posterior-mean prediction requires either covariance or a saved penalized Hessian; refit"
            .to_string(),
    )
}


fn infer_covariance_mode(mode: CovarianceModeArg) -> gam::estimate::InferenceCovarianceMode {
    match mode {
        CovarianceModeArg::Conditional => gam::estimate::InferenceCovarianceMode::Conditional,
        CovarianceModeArg::Corrected => {
            gam::estimate::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred
        }
    }
}


fn response_interval_from_mean_sd(
    mean: ArrayView1<'_, f64>,
    response_sd: ArrayView1<'_, f64>,
    z: f64,
    lo: f64,
    hi: f64,
) -> (Array1<f64>, Array1<f64>) {
    let lower = Array1::from_iter(
        mean.iter()
            .zip(response_sd.iter())
            .map(|(&m, &s)| (m - z * s).clamp(lo, hi)),
    );
    let upper = Array1::from_iter(
        mean.iter()
            .zip(response_sd.iter())
            .map(|(&m, &s)| (m + z * s).clamp(lo, hi)),
    );
    (lower, upper)
}


fn invert_symmetric_matrix(a: &Array2<f64>) -> Result<Array2<f64>, CliError> {
    if a.nrows() != a.ncols() {
        return Err(CliError::Internal {
            reason: format!(
                "matrix must be square for inversion; got {}x{}",
                a.nrows(),
                a.ncols()
            ),
        });
    }
    let n = a.nrows();
    let h = gam::faer_ndarray::FaerArrayView::new(a);
    let mut rhs = FaerMat::zeros(n, n);
    for i in 0..n {
        rhs[(i, i)] = 1.0;
    }
    let factor = gam::faer_ndarray::factorize_symmetricwith_fallback(h.as_ref(), Side::Lower)
        .map_err(|err| CliError::Internal {
            reason: format!("failed to factorize matrix for inversion: {err}"),
        })?;
    factor.solve_in_place(rhs.as_mut());
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = rhs[(i, j)];
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(CliError::Internal {
            reason: "inversion produced non-finite entries".to_string(),
        });
    }
    Ok(out)
}


fn fit_result_from_external(ext: ExternalOptimResult) -> UnifiedFitResult {
    let log_lambdas = ext.lambdas.mapv(|v| v.max(1e-300).ln());
    let edf = ext
        .inference
        .as_ref()
        .map(|inf| inf.edf_total)
        .unwrap_or(0.0);
    let geometry = ext
        .inference
        .as_ref()
        .map(|inf| gam::estimate::FitGeometry {
            penalized_hessian: inf.penalized_hessian.clone(),
            working_weights: inf.working_weights.clone(),
            working_response: inf.working_response.clone(),
        });
    // Boundary adapter: `inf.beta_covariance` is now the `PhiScaledCovariance`
    // newtype; unwrap to the raw `Array2<f64>` that the
    // `covariance_conditional` parts field still uses.
    let covariance_conditional = ext
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance.as_ref().map(|c| c.as_array().clone()));
    let covariance_corrected = ext
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance_corrected.clone());
    let penalized_objective = ext.reml_score;
    UnifiedFitResult::try_from_parts(gam::estimate::UnifiedFitResultParts {
        blocks: vec![gam::estimate::FittedBlock {
            beta: ext.beta.clone(),
            role: gam::estimate::BlockRole::Mean,
            edf,
            lambdas: ext.lambdas.clone(),
        }],
        log_lambdas,
        lambdas: ext.lambdas,
        likelihood_family: Some(ext.likelihood_family),
        likelihood_scale: ext.likelihood_scale,
        log_likelihood_normalization: ext.log_likelihood_normalization,
        log_likelihood: ext.log_likelihood,
        deviance: ext.deviance,
        reml_score: ext.reml_score,
        stable_penalty_term: ext.stable_penalty_term,
        penalized_objective,
        outer_iterations: ext.iterations,
        outer_converged: ext.outer_converged,
        outer_gradient_norm: Some(ext.finalgrad_norm),
        standard_deviation: ext.standard_deviation,
        covariance_conditional,
        covariance_corrected,
        inference: ext.inference,
        fitted_link: ext.fitted_link,
        geometry,
        block_states: Vec::new(),
        pirls_status: ext.pirls_status,
        max_abs_eta: ext.max_abs_eta,
        constraint_kkt: ext.constraint_kkt,
        artifacts: ext.artifacts,
        inner_cycles: 0,
    })
    .expect("external optimizer returned invalid fit metrics")
}


fn write_matrix_csv(path: &Path, mat: &Array2<f64>, prefix: &str) -> Result<(), CliError> {
    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| CliError::FileWriteFailed {
            reason: format!("failed to create output csv '{}': {e}", path.display()),
        })?;
    let headers = (0..mat.ncols())
        .map(|j| format!("{prefix}_{j}"))
        .collect::<Vec<_>>();
    wtr.write_record(headers)
        .map_err(|e| CliError::FileWriteFailed {
            reason: format!("failed to write csv header: {e}"),
        })?;
    for i in 0..mat.nrows() {
        let row = (0..mat.ncols())
            .map(|j| format!("{:.12}", mat[[i, j]]))
            .collect::<Vec<_>>();
        wtr.write_record(row)
            .map_err(|e| CliError::FileWriteFailed {
                reason: format!("failed to write csv row {i}: {e}"),
            })?;
    }
    wtr.flush().map_err(|e| CliError::FileWriteFailed {
        reason: format!("failed to flush csv writer: {e}"),
    })?;
    Ok(())
}


fn load_prediction_id_values(
    path: &Path,
    id_column: &str,
    expected_rows: usize,
) -> Result<Vec<String>, String> {
    if id_column.trim().is_empty() {
        return Err("--id-column must be a non-empty column name".to_string());
    }
    let projected = load_dataset_projected(path, &[id_column.to_string()])?;
    if projected.values.nrows() != expected_rows {
        return Err(format!(
            "id column '{id_column}' row count {} does not match prediction row count {expected_rows}",
            projected.values.nrows()
        ));
    }
    let col_idx = resolve_role_col(&projected.column_map(), id_column, "id")?;
    let schema_col = projected
        .schema
        .columns
        .iter()
        .find(|column| column.name == id_column)
        .ok_or_else(|| format!("id column '{id_column}' missing from inferred schema"))?;
    let mut out = Vec::<String>::with_capacity(projected.values.nrows());
    for row_idx in 0..projected.values.nrows() {
        let value = projected.values[[row_idx, col_idx]];
        if !value.is_finite() {
            return Err(format!(
                "id column '{id_column}' contains non-finite value at row {row_idx}"
            ));
        }
        let rendered = match schema_col.kind {
            ColumnKindTag::Categorical => {
                let level_idx = value.round() as usize;
                schema_col.levels.get(level_idx).cloned().ok_or_else(|| {
                    format!(
                        "id column '{id_column}' categorical code {level_idx} at row {row_idx} is out of bounds"
                    )
                })?
            }
            ColumnKindTag::Continuous | ColumnKindTag::Binary => format_id_number(value),
        };
        out.push(rendered);
    }
    Ok(out)
}


fn format_id_number(value: f64) -> String {
    if (value - value.round()).abs() <= 1e-9 {
        format!("{value:.0}")
    } else {
        format!("{value:.12}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}


fn prepend_id_column_to_prediction_csv(
    path: &Path,
    id_column: &str,
    id_values: &[String],
) -> Result<(), String> {
    let mut rdr = csv::Reader::from_path(path)
        .map_err(|e| format!("failed to read prediction csv '{}': {e}", path.display()))?;
    let headers = rdr
        .headers()
        .map_err(|e| format!("failed to read prediction csv header: {e}"))?
        .clone();
    if headers.iter().any(|name| name == id_column) {
        return Err(format!(
            "prediction output already contains id column '{id_column}'"
        ));
    }

    let tmp_path = path.with_extension("tmp-id-column.csv");
    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(&tmp_path)
        .map_err(|e| {
            format!(
                "failed to create temporary prediction csv '{}': {e}",
                tmp_path.display()
            )
        })?;
    let mut out_headers = Vec::<String>::with_capacity(headers.len() + 1);
    out_headers.push(id_column.to_string());
    out_headers.extend(headers.iter().map(str::to_string));
    wtr.write_record(&out_headers)
        .map_err(|e| format!("failed writing prediction csv header with id column: {e}"))?;

    let mut row_count = 0usize;
    for record in rdr.records() {
        let record = record.map_err(|e| format!("failed reading prediction csv row: {e}"))?;
        let id = id_values.get(row_count).ok_or_else(|| {
            format!(
                "prediction csv has more rows than id column '{id_column}' (first extra row index {row_count})"
            )
        })?;
        let mut out_record = Vec::<String>::with_capacity(record.len() + 1);
        out_record.push(id.clone());
        out_record.extend(record.iter().map(str::to_string));
        wtr.write_record(&out_record)
            .map_err(|e| format!("failed writing prediction csv row {row_count}: {e}"))?;
        row_count += 1;
    }
    if row_count != id_values.len() {
        return Err(format!(
            "prediction csv row count {row_count} does not match id column '{id_column}' row count {}",
            id_values.len()
        ));
    }
    wtr.flush()
        .map_err(|e| format!("failed to flush prediction csv with id column: {e}"))?;
    std::fs::rename(&tmp_path, path).map_err(|e| {
        format!(
            "failed to replace prediction csv '{}' with id-column version '{}': {e}",
            path.display(),
            tmp_path.display()
        )
    })?;
    Ok(())
}


/// Unified CSV prediction writer.  Each column is a `(name, data)` pair;
/// the function writes a header row from the names and one data row per
/// element, formatting every value to 12 decimal places.
///
/// All columns must have the same length.  An empty column list is an error.
fn write_prediction_csv_unified(path: &Path, columns: &[(&str, &[f64])]) -> CliResult<()> {
    if columns.is_empty() {
        return Err(CliError::Internal {
            reason: "internal error: write_prediction_csv_unified called with no columns"
                .to_string(),
        });
    }
    let n = columns[0].1.len();
    for (name, data) in columns.iter() {
        if data.len() != n {
            return Err(CliError::Internal {
                reason: format!(
                    "internal error: column '{}' has length {} but expected {}",
                    name,
                    data.len(),
                    n,
                ),
            });
        }
    }

    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| CliError::FileWriteFailed {
            reason: format!("failed to create output csv '{}': {e}", path.display()),
        })?;

    let headers: Vec<&str> = columns.iter().map(|(name, _)| *name).collect();
    wtr.write_record(&headers)
        .map_err(|e| CliError::FileWriteFailed {
            reason: format!("failed writing csv header: {e}"),
        })?;

    // Validate all prediction values are finite before writing.
    // NaN or Inf in clinical output would be dangerous.
    for (col_name, data) in columns {
        for (i, val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(CliError::Internal {
                    reason: format!(
                        "non-finite prediction value in column '{}' at row {}: {}",
                        col_name, i, val
                    ),
                });
            }
        }
    }

    for i in 0..n {
        let row: Vec<String> = columns
            .iter()
            .map(|(_, data)| format!("{:.12}", data[i]))
            .collect();
        wtr.write_record(&row)
            .map_err(|e| CliError::FileWriteFailed {
                reason: format!("failed writing csv row {i}: {e}"),
            })?;
    }

    wtr.flush().map_err(|e| CliError::FileWriteFailed {
        reason: format!("failed to flush csv writer: {e}"),
    })?;
    Ok(())
}


/// Convenience wrapper: builds a standard (non-survival, non-location-scale)
/// prediction column list and delegates to [`write_prediction_csv_unified`].
fn write_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    mean: ArrayView1<'_, f64>,
    eta_se: Option<ArrayView1<'_, f64>>,
    mean_lower: Option<ArrayView1<'_, f64>>,
    mean_upper: Option<ArrayView1<'_, f64>>,
) -> CliResult<()> {
    // Materialise views into contiguous vecs so we can pass &[f64] slices.
    let eta_v: Vec<f64> = eta.to_vec();
    let mean_v: Vec<f64> = mean.to_vec();

    let mut cols: Vec<(&str, &[f64])> = vec![("linear_predictor", &eta_v), ("mean", &mean_v)];

    let se_v: Vec<f64>;
    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(se) = eta_se {
        se_v = se.to_vec();
        lo_v = mean_lower
            .ok_or_else(|| {
                "internal error: mean_lower missing while std_error is present".to_string()
            })?
            .to_vec();
        hi_v = mean_upper
            .ok_or_else(|| {
                "internal error: mean_upper missing while std_error is present".to_string()
            })?
            .to_vec();
        cols.push(("std_error", &se_v));
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if let (Some(lo), Some(hi)) = (mean_lower, mean_upper) {
        lo_v = lo.to_vec();
        hi_v = hi.to_vec();
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if mean_lower.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: mean_upper missing while mean_lower is present".to_string(),
        });
    } else if mean_upper.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: mean_lower missing while mean_upper is present".to_string(),
        });
    }

    write_prediction_csv_unified(path, &cols)
}


/// Convenience wrapper for Gaussian location-scale predictions (always
/// includes a `sigma` column).
fn write_gaussian_location_scale_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    mean: ArrayView1<'_, f64>,
    sigma: ArrayView1<'_, f64>,
    mean_lower: Option<ArrayView1<'_, f64>>,
    mean_upper: Option<ArrayView1<'_, f64>>,
) -> CliResult<()> {
    let eta_v: Vec<f64> = eta.to_vec();
    let mean_v: Vec<f64> = mean.to_vec();
    let sigma_v: Vec<f64> = sigma.to_vec();

    let mut cols: Vec<(&str, &[f64])> = vec![
        ("linear_predictor", &eta_v),
        ("mean", &mean_v),
        ("sigma", &sigma_v),
    ];

    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(lo) = mean_lower {
        lo_v = lo.to_vec();
        hi_v = mean_upper
            .ok_or_else(|| CliError::Internal {
                reason: "internal error: mean_upper missing while mean_lower is present"
                    .to_string(),
            })?
            .to_vec();
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if mean_upper.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: gaussian location-scale output requires both mean_lower and mean_upper"
                .to_string(),
        });
    }

    write_prediction_csv_unified(path, &cols)
}


/// Convenience wrapper for survival predictions. Survival output uses explicit
/// probability semantics because the event probability is `1 - survival_prob`.
fn write_survival_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    survival_prob: ArrayView1<'_, f64>,
    eta_se: Option<ArrayView1<'_, f64>>,
    survival_lower: Option<ArrayView1<'_, f64>>,
    survival_upper: Option<ArrayView1<'_, f64>>,
) -> CliResult<()> {
    let eta_v: Vec<f64> = eta.to_vec();
    let surv_v: Vec<f64> = survival_prob.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    let risk_v: Vec<f64> = eta_v.clone();
    let fail_v: Vec<f64> = surv_v.iter().map(|&s| (1.0 - s).clamp(0.0, 1.0)).collect();

    let mut cols: Vec<(&str, &[f64])> = vec![
        ("linear_predictor", &eta_v),
        ("survival_prob", &surv_v),
        ("failure_prob", &fail_v),
        ("risk_score", &risk_v),
    ];

    let se_v: Vec<f64>;
    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(se) = eta_se {
        se_v = se.to_vec();
        lo_v = survival_lower
            .ok_or_else(|| {
                "internal error: survival_lower missing while std_error is present".to_string()
            })?
            .to_vec();
        hi_v = survival_upper
            .ok_or_else(|| {
                "internal error: survival_upper missing while std_error is present".to_string()
            })?
            .to_vec();
        cols.push(("std_error", &se_v));
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if let (Some(lo), Some(hi)) = (survival_lower, survival_upper) {
        lo_v = lo.to_vec();
        hi_v = hi.to_vec();
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if survival_lower.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: survival_upper missing while survival_lower is present"
                .to_string(),
        });
    } else if survival_upper.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: survival_lower missing while survival_upper is present"
                .to_string(),
        });
    }

    write_prediction_csv_unified(path, &cols)
}


/// Convenience wrapper for binary deployment predictions backed by a survival
/// hazard window (includes explicit `event_prob`, `failure_prob`, and
/// `survival_prob` columns).
fn write_survival_binary_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    event_prob: ArrayView1<'_, f64>,
    eta_se: Option<ArrayView1<'_, f64>>,
    event_lower: Option<ArrayView1<'_, f64>>,
    event_upper: Option<ArrayView1<'_, f64>>,
) -> CliResult<()> {
    let eta_v: Vec<f64> = eta.to_vec();
    let event_v: Vec<f64> = event_prob.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    let risk_v: Vec<f64> = eta_v.clone();
    let survival_v: Vec<f64> = event_v.iter().map(|&p| (1.0 - p).clamp(0.0, 1.0)).collect();

    let mut cols: Vec<(&str, &[f64])> = vec![
        ("linear_predictor", &eta_v),
        ("mean", &event_v),
        ("event_prob", &event_v),
        ("failure_prob", &event_v),
        ("survival_prob", &survival_v),
        ("risk_score", &risk_v),
    ];

    let se_v: Vec<f64>;
    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(se) = eta_se {
        se_v = se.to_vec();
        lo_v = event_lower
            .ok_or_else(|| CliError::Internal {
                reason: "internal error: event_lower missing while std_error is present"
                    .to_string(),
            })?
            .to_vec();
        hi_v = event_upper
            .ok_or_else(|| CliError::Internal {
                reason: "internal error: event_upper missing while std_error is present"
                    .to_string(),
            })?
            .to_vec();
        cols.push(("std_error", &se_v));
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if let (Some(lo), Some(hi)) = (event_lower, event_upper) {
        lo_v = lo.to_vec();
        hi_v = hi.to_vec();
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if event_lower.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: event_upper missing while event_lower is present".to_string(),
        });
    } else if event_upper.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: event_lower missing while event_upper is present".to_string(),
        });
    }

    write_prediction_csv_unified(path, &cols)
}
