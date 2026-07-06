use super::*;

/// Request-specific inputs to the canonical standard-fit `FitOptions`.
///
/// Everything in here varies per call (the link state extracted from the
/// formula/config, the linear constraints synthesized from `bounded()` /
/// shape-constrained terms, the Firth / adaptive-regularization toggles read
/// off the `FitConfig`). Every *policy* field of `FitOptions` — the ones that
/// decide HOW the outer REML optimization behaves (`compute_inference`,
/// `skip_rho_posterior_inference`, `tol`, the `max_iter` default, the penalty
/// shrinkage floor) — is filled in by [`canonical_standard_fit_options`] and is
/// NOT settable here, so the CLI binary and the Python/PyO3 path cannot resolve
/// a different optimization policy for the same model (#1196). Before this seam
/// existed the CLI hand-built `FitOptions` with `tol: 1e-6` /
/// `skip_rho_posterior_inference: false` while the formula path used
/// `tol: 1e-10` / `skip_rho_posterior_inference: true`, so the identical model
/// fit *differently* depending on which entry point you called it from — the
/// exact class of divergence #1191 surfaced.
#[derive(Default)]
pub struct StandardFitOptionsInputs {
    pub latent_cloglog: Option<LatentCLogLogState>,
    pub mixture_link: Option<MixtureLinkSpec>,
    pub optimize_mixture: bool,
    pub sas_link: Option<SasLinkSpec>,
    pub optimize_sas: bool,
    pub linear_constraints: Option<gam_solve::pirls::LinearInequalityConstraints>,
    pub firth_bias_reduction: bool,
    pub adaptive_regularization: Option<AdaptiveRegularizationOptions>,
    /// `Some` only when a caller (the forced-Firth CLI branch) overrides the
    /// canonical default. `None` keeps the single-source default `Some(1e-6)`.
    pub penalty_shrinkage_floor_override: Option<Option<f64>>,
    pub persist_warm_start_disk: bool,
}

/// The single source of truth for standard-fit `FitOptions` *policy*.
///
/// Both standard-fit entry points — `materialize_standard` (the formula /
/// Python / PyO3 path) and the `gam` CLI's `run_fit` — construct their
/// `StandardFitRequest` options through this function, so the outer REML
/// optimization policy (`compute_inference`, `skip_rho_posterior_inference`,
/// `tol`, `max_iter` default, `penalty_shrinkage_floor`) is identical by
/// construction. New policy fields must be set HERE, never re-derived at a call
/// site, which is what makes Python/CLI behavioral divergence structurally
/// impossible rather than enforced by parallel-but-equal code (#1196).
pub fn canonical_standard_fit_options(
    config: &FitConfig,
    inputs: StandardFitOptionsInputs,
) -> FitOptions {
    FitOptions {
        latent_cloglog: inputs.latent_cloglog,
        mixture_link: inputs.mixture_link,
        optimize_mixture: inputs.optimize_mixture,
        sas_link: inputs.sas_link,
        optimize_sas: inputs.optimize_sas,
        // Posterior covariance is always computed so `predict --uncertainty`
        // works for every family (the `COV_MAX_P` diagonal fallback caps cost).
        compute_inference: true,
        // Formula/CLI fits are the interactive/default path: keep coefficient
        // covariance and the smoothing correction, and emit the CHEAP Tier-0
        // live-rho posterior certificate (a handful of outer-criterion
        // evaluations), which the optimizer surfaces regardless of this flag
        // whenever it is cheaply available (#1810). This flag only suppresses the
        // EXPENSIVE escalation tiers (Tier-1 quadrature / Tier-2 NUTS over rho),
        // which could otherwise launch NUTS and turn ordinary fits into sampler
        // benchmarks. Lower-level callers that explicitly need the escalation opt
        // in elsewhere (`skip_rho_posterior_inference: false`).
        skip_rho_posterior_inference: true,
        max_iter: config.outer_max_iter.unwrap_or(200),
        // Outer REML/LAML smoothing-selection tolerance. `1e-10` (effective
        // projected-gradient threshold ≈ 1e-7) resolves λ̂ to optimiser
        // precision and restores the `w=c ⇔ c-fold replication` invariance in
        // smoothing selection (gam#893). The CLI previously used the stale
        // `1e-6`, which over-smoothed relative to the formula path.
        tol: 1e-10,
        nullspace_dims: vec![],
        linear_constraints: inputs.linear_constraints,
        firth_bias_reduction: inputs.firth_bias_reduction,
        adaptive_regularization: inputs.adaptive_regularization,
        penalty_shrinkage_floor: inputs
            .penalty_shrinkage_floor_override
            .unwrap_or(Some(1e-6)),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: inputs.persist_warm_start_disk,
    }
}

pub fn fit_model(request: FitRequest<'_>) -> Result<FitResult, WorkflowError> {
    // Disk warm-start persistence is opt-in. The always-on in-memory warm start
    // remains inside the fit engines, but the workflow dispatcher must not open
    // the shared WarmStartStore for ordinary formula fits: refit-heavy quality
    // tests get no cross-process reuse and previously paid cache lookup,
    // checkpoint, and eviction scans on every replicate (#1082/#1114).
    let request = request;
    // Each `fit_*_model` helper still returns `Result<_, String>` internally;
    // the boundary conversion happens here so the public API returns
    // `WorkflowError::IntegrationFailed` carrying the underlying solver text.
    let wrap_solver_err =
        |reason: String| -> WorkflowError { WorkflowError::IntegrationFailed { reason } };
    match request {
        FitRequest::Standard(request) => fit_standard_model(request)
            .map(FitResult::Standard)
            .map_err(wrap_solver_err),
        FitRequest::GaussianLocationScale(request) => fit_gaussian_location_scale_model(request)
            .map(FitResult::GaussianLocationScale)
            .map_err(wrap_solver_err),
        FitRequest::BinomialLocationScale(request) => fit_binomial_location_scale_model(request)
            .map(FitResult::BinomialLocationScale)
            .map_err(wrap_solver_err),
        FitRequest::DispersionLocationScale(request) => {
            fit_dispersion_location_scale_model(request)
                .map(FitResult::DispersionLocationScale)
                .map_err(wrap_solver_err)
        }
        FitRequest::SurvivalLocationScale(request) => fit_survival_location_scale_model(request)
            .map(FitResult::SurvivalLocationScale)
            .map_err(wrap_solver_err),
        FitRequest::SurvivalTransformation(request) => fit_survival_transformation_model(request)
            .map(FitResult::SurvivalTransformation)
            .map_err(wrap_solver_err),
        FitRequest::BernoulliMarginalSlope(request) => fit_bernoulli_marginal_slope_model(request)
            .map(FitResult::BernoulliMarginalSlope)
            .map_err(wrap_solver_err),
        FitRequest::SurvivalMarginalSlope(request) => fit_survival_marginal_slope_model(request)
            .map(FitResult::SurvivalMarginalSlope)
            .map_err(wrap_solver_err),
        FitRequest::LatentSurvival(request) => fit_latent_survival_model(request)
            .map(FitResult::LatentSurvival)
            .map_err(wrap_solver_err),
        FitRequest::LatentBinary(request) => fit_latent_binary_model(request)
            .map(FitResult::LatentBinary)
            .map_err(wrap_solver_err),
        FitRequest::TransformationNormal(request) => fit_transformation_normal_model(request)
            .map(FitResult::TransformationNormal)
            .map_err(wrap_solver_err),
    }
}
/// Resolve the [`gam_runtime::resource::ResourcePolicy`] backing term construction
/// for a given [`FitConfig`] + dataset.
///
/// If the caller hasn't supplied an explicit policy override, derive one from
/// the shape of the problem via
/// [`gam_runtime::resource::ResourcePolicy::for_problem`]. At large scale (n_rows
/// >= 100k or the marginal-slope large-scale path active) this returns
/// > `analytic_operator_required` so that any silent dense materialization in
/// > the term-construction layer fails fast rather than allocating tens of GiB;
/// > at small scale it falls through to the permissive default-library policy
/// > so that non-operator bases still build cleanly.
///
/// `p_estimate = 0` because the per-block coefficient count isn't known until
/// the spec has been built; the n_rows and hints triggers are sufficient to
/// flip strict mode for every shape that needs it.
pub(crate) fn resolved_resource_policy(
    config: &FitConfig,
    data: &Dataset,
    hints: gam_runtime::resource::ProblemHints,
) -> gam_runtime::resource::ResourcePolicy {
    if let Some(p) = config.resource_policy.clone() {
        return p;
    }
    gam_runtime::resource::ResourcePolicy::for_problem(data.values.nrows(), 0, hints)
}

pub(crate) fn marginal_slope_hints(config: &FitConfig) -> gam_runtime::resource::ProblemHints {
    gam_runtime::resource::ProblemHints {
        marginal_slope_large_scale_active: requests_bernoulli_marginal_slope(config),
    }
}
/// Parse, materialize, and fit a model in one call.
/// Resolve the expectile asymmetry `τ` requested by `config`, if any.
///
/// Returns `Ok(Some(τ))` when `config.family` is `"expectile"` (optionally with
/// an inline asymmetry, `"expectile(0.9)"`), `Ok(None)` for every other family,
/// and `Err` when an expectile request carries an out-of-range `τ`. The inline
/// form takes precedence over the explicit [`FitConfig::expectile_tau`] field
/// only when both are present and disagree is rejected as a contradiction; when
/// neither pins `τ`, the median expectile `τ = 0.5` (the ordinary mean fit) is
/// the default.
fn expectile_tau_for_config(config: &FitConfig) -> Result<Option<f64>, WorkflowError> {
    let Some(raw) = config.family.as_deref() else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !(lower == "expectile" || lower.starts_with("expectile(")) {
        return Ok(None);
    }
    let invalid = |reason: String| WorkflowError::InvalidConfig { reason };
    // Optional inline asymmetry: `expectile(0.9)`.
    let inline_tau = if let Some(rest) = lower.strip_prefix("expectile(") {
        let inner = rest.strip_suffix(')').ok_or_else(|| {
            invalid(format!(
                "expectile family asymmetry must be written as `expectile(τ)`; got `{trimmed}`"
            ))
        })?;
        let value: f64 = inner.trim().parse().map_err(|_| {
            invalid(format!(
                "expectile asymmetry `{}` is not a finite number",
                inner.trim()
            ))
        })?;
        Some(value)
    } else {
        None
    };
    let tau = match (inline_tau, config.expectile_tau) {
        (Some(a), Some(b)) if (a - b).abs() > 0.0 => {
            return Err(invalid(format!(
                "expectile asymmetry given both inline (`expectile({a})`) and via expectile_tau \
                 ({b}); supply exactly one"
            )));
        }
        (Some(a), _) => a,
        (None, Some(b)) => b,
        (None, None) => 0.5,
    };
    if !(tau.is_finite() && tau > 0.0 && tau < 1.0) {
        return Err(invalid(format!(
            "expectile asymmetry τ must be finite and strictly in (0, 1); got {tau}"
        )));
    }
    Ok(Some(tau))
}

/// Per-row asymmetric LAWS weight `wᵢ(τ) = τ` if `yᵢ > μᵢ` else `1 − τ`, scaled
/// by the base prior weight. At the boundary `yᵢ = μᵢ` the two half-weights
/// agree in the limit only at `τ = 0.5`; the convention `yᵢ > μᵢ ⇒ τ` (strict)
/// matches Newey–Powell's lower-closed asymmetric loss and is what `expectreg`
/// uses. The fixed point is independent of the tie convention because ties form
/// a measure-zero set under any continuous response.
fn expectile_row_weights(
    y: ArrayView1<f64>,
    mu: ArrayView1<f64>,
    base: ArrayView1<f64>,
    tau: f64,
) -> Array1<f64> {
    Array1::from_shape_fn(y.len(), |i| {
        let asym = if y[i] > mu[i] { tau } else { 1.0 - tau };
        base[i] * asym
    })
}

fn constant_gaussian_standard_fit(
    request: &StandardFitRequest<'_>,
) -> Result<StandardFitResult, WorkflowError> {
    if !request.family.is_gaussian_identity() || request.y.is_empty() {
        return Err(WorkflowError::InvalidConfig {
            reason: "constant Gaussian shortcut requires a non-empty Gaussian identity request"
                .to_string(),
        });
    }
    if request.y.iter().any(|value| !value.is_finite())
        || request.offset.iter().any(|value| !value.is_finite())
        || request
            .weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(WorkflowError::InvalidConfig {
            reason: "constant Gaussian shortcut requires finite response, offset, and non-negative weights"
                .to_string(),
        });
    }
    let weight_sum = request.weights.sum();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(WorkflowError::InvalidConfig {
            reason: "constant Gaussian shortcut requires positive total weight".to_string(),
        });
    }
    let mut centered_sum = 0.0_f64;
    for i in 0..request.y.len() {
        centered_sum += request.weights[i] * (request.y[i] - request.offset[i]);
    }
    let intercept = centered_sum / weight_sum;
    let design =
        build_term_collection_design(request.data.view(), &request.spec).map_err(|err| {
            WorkflowError::InvalidConfig {
                reason: format!("constant Gaussian shortcut could not rebuild design: {err}"),
            }
        })?;
    let p = design.design.ncols();
    let mut beta = Array1::<f64>::zeros(p);
    for col in design.intercept_range.clone() {
        if col < p {
            beta[col] = intercept;
        }
    }
    let lambdas = Array1::<f64>::ones(design.penalties.len());
    let log_lambdas = Array1::<f64>::zeros(design.penalties.len());
    let fit =
        gam_solve::estimate::UnifiedFitResult::try_from_parts(gam_solve::estimate::UnifiedFitResultParts {
            blocks: vec![gam_solve::estimate::FittedBlock {
                beta: beta.clone(),
                role: gam_problem::BlockRole::Mean,
                edf: design.intercept_range.len() as f64,
                lambdas: lambdas.clone(),
            }],
            log_lambdas,
            lambdas,
            likelihood_family: Some(request.family.clone()),
            likelihood_scale: gam_problem::LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: gam_problem::LogLikelihoodNormalization::UserProvided,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: Some(0.0),
            standard_deviation: 0.0,
            covariance_conditional: None,
            covariance_corrected: None,
            inference: None,
            fitted_link: gam_solve::estimate::FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: gam_solve::pirls::PirlsStatus::Converged,
            max_abs_eta: intercept.abs(),
            constraint_kkt: None,
            artifacts: gam_solve::estimate::FitArtifacts {
                pirls: None,
                ..Default::default()
            },
            inner_cycles: 0,
        })
        .map_err(|err| WorkflowError::IntegrationFailed {
            reason: format!("constant Gaussian shortcut produced invalid fit: {err}"),
        })?;
    let resolvedspec =
        freeze_term_collection_from_design(&request.spec, &design).map_err(|err| {
            WorkflowError::InvalidConfig {
                reason: format!("constant Gaussian shortcut could not freeze design: {err}"),
            }
        })?;
    Ok(StandardFitResult {
        fit,
        design,
        resolvedspec,
        adaptive_diagnostics: None,
        kappa_timing: None,
        saved_link_state: gam_solve::estimate::FittedLinkState::Standard(None),
        wiggle_knots: None,
        wiggle_degree: None,
        wiggle_saved_warp_beta: None,
    })
}

fn gaussian_response_is_constant(request: &StandardFitRequest<'_>) -> bool {
    if !request.family.is_gaussian_identity()
        || request.y.is_empty()
        || request.y.iter().any(|value| !value.is_finite())
    {
        return false;
    }
    let (lo, hi) = request
        .y
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &value| {
            (lo.min(value), hi.max(value))
        });
    (hi - lo).abs() <= 1.0e-12 * hi.abs().max(1.0)
}

pub fn fit_from_formula(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<FitResult, WorkflowError> {
    // Expectile regression (Newey–Powell asymmetric least squares): when the
    // family resolves to "expectile", the τ-expectile of `y | x` is the
    // minimizer of `Σ wᵢ(τ)·(yᵢ − μᵢ)²`, `wᵢ(τ) = τ` if `yᵢ > μᵢ` else `1 − τ`
    // — the smooth analogue of the τ-quantile. The minimizer is a Least
    // Asymmetrically Weighted Squares (LAWS) fixed point: iterate the penalized
    // Gaussian-identity GAM with `wᵢ(τ)` recomputed from the current `μᵢ` until
    // the residual-sign pattern stabilizes. REML λ-selection runs inside each
    // inner Gaussian solve, so every gam smooth/tensor/spatial basis becomes a
    // penalized expectile smooth with data-driven smoothing for free. This is a
    // genuine estimator route, not a silent swap: it fires only on the explicit
    // `family = "expectile"`. Every other family falls through unchanged.
    if let Some(result) = fit_expectile_if_requested(formula, data, config)? {
        return Ok(FitResult::Standard(result));
    }
    let mat = materialize(formula, data, config)?;
    // Exact O(n) spline-scan fast path (#1030): when the materialized request
    // is the single 1-D Gaussian-identity penalized-smooth shape the
    // state-space scan solves exactly, route through it and return the
    // scan-bearing model directly — the same penalized posterior at O(n) per
    // λ-trial instead of the dense design/Gram route. Detection is structural
    // and conservative (see `spline_scan_fast_path`); every other shape falls
    // through to the dense `fit_model` path unchanged. Mirrors the CLI
    // (main.rs run_fit) and FFI consumers, which build the persistence payload
    // from this same `SplineScanFit`.
    if let FitRequest::Standard(request) = &mat.request {
        if gaussian_response_is_constant(request) {
            return constant_gaussian_standard_fit(request).map(FitResult::Standard);
        }
        if let Some(inputs) = spline_scan_fast_path(request) {
            let scan = gam_solve::spline_scan::fit_spline_scan(
                &inputs.x,
                &inputs.y,
                &inputs.w,
                inputs.order,
            )
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })?;
            return Ok(FitResult::SplineScan(scan));
        }
        // O(n log n) multiresolution residual-cascade fast path (#1032): a
        // scattered low-d Gaussian-identity Duchon/Matérn smooth past the
        // dense-kernel cliff. UNLIKE the scan, the cascade is a DIFFERENT
        // posterior from the dense radial term, so it only ever fires as an
        // explicit alternative estimator on the exact structural signature
        // (`residual_cascade_fast_path`) AND when the in-cascade quasi-uniformity
        // guard certifies the metric — a rejected metric or any ineligible shape
        // falls through to the dense `fit_model` path (a genuine estimator
        // choice, never a silent swap). The save paths build the persistence
        // payload from this `ResidualCascadeFit`'s `to_state` snapshot.
        if let Some(inputs) = residual_cascade_fast_path(request) {
            let coord_refs: Vec<&[f64]> = inputs.coords.iter().map(Vec::as_slice).collect();
            if let Ok(fit) = gam_solve::residual_cascade::fit_residual_cascade(
                &coord_refs,
                &inputs.y,
                &inputs.w,
                &inputs.metric,
                inputs.sobolev_s,
            ) {
                return Ok(FitResult::ResidualCascade(fit));
            }
            // The quasi-uniformity guard (caveat 2) or any degenerate-design
            // signal surfaces as a build/solve error; fall through to the dense
            // kernel path rather than failing the fit outright.
        }
    }
    // `fit_model` already returns `WorkflowError` end-to-end; propagate it
    // directly instead of stringifying then re-wrapping.
    fit_model(mat.request)
}

/// THE single dispatch seam for the expectile (Newey–Powell LAWS) family.
///
/// Returns `Ok(Some(result))` with the converged τ-expectile as an ordinary
/// [`StandardFitResult`] when `config.family` selects the expectile family
/// (`"expectile"` or `"expectile(τ)"`, optionally pinned by
/// [`FitConfig::expectile_tau`]), `Ok(None)` for every other family — in which
/// case the caller runs its normal materialize/`fit_model` path — and `Err` on a
/// malformed expectile request or an inner-fit failure.
///
/// Every public entry point that resolves a family routes through this seam
/// *before* materializing: the in-process [`fit_from_formula`], the Python FFI
/// (`gam-pyffi`), and the `gam` CLI. Centralizing the dispatch here is what makes
/// the estimator reachable from every interface instead of only the library
/// call — and what prevents the class of bug where a newly-added outer estimator
/// is wired into one entry point and silently bypassed by the others (#1777).
/// The returned [`StandardFitResult`] carries the full design / resolved spec /
/// fit, so each caller builds its persistence payload from it exactly as it does
/// for any other standard fit.
pub fn fit_expectile_if_requested(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<Option<StandardFitResult>, WorkflowError> {
    match expectile_tau_for_config(config)? {
        Some(tau) => Ok(Some(fit_expectile_laws(formula, data, config, tau)?)),
        None => Ok(None),
    }
}

/// Least Asymmetrically Weighted Squares (LAWS) driver for expectile GAMs.
///
/// The τ-expectile surface minimizes `Σ wᵢ(τ)·(yᵢ − μᵢ)²` with the residual-
/// sign asymmetric weight `wᵢ(τ)`. Because that weight is piecewise-constant in
/// `sign(yᵢ − μᵢ)`, the objective is the supremum of a finite family of
/// weighted least-squares problems and its minimizer is the unique fixed point
/// of: *solve the penalized WLS with weights frozen at the current sign
/// pattern, then recompute the sign pattern from the new fit*. The asymmetric
/// loss is strictly convex (weights bounded in `[min(τ,1−τ), max(τ,1−τ)] > 0`),
/// so this monotone-descent iteration converges, and since the sign pattern
/// takes finitely many values it stabilizes in finitely many steps (Schnabel &
/// Eilers 2009; the same Newton/IRLS-for-expectiles `expectreg` runs).
///
/// Each inner solve is the FULL standard Gaussian-identity GAM: any basis,
/// tensor, spatial smooth, by-variable, random effect, plus REML λ-selection on
/// the current asymmetric weights. The returned fit is an ordinary
/// [`FitResult::Standard`] whose coefficients ARE the penalized τ-expectile —
/// every downstream consumer (predict, posterior bands, persistence) works
/// unchanged. The reported scale is the asymmetric working variance, so
/// expectile standard errors are the sandwich-free Gaussian-form bands of the
/// converged weighted problem (a deliberate first-rung choice; see #1100).
fn fit_expectile_laws(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
    tau: f64,
) -> Result<StandardFitResult, WorkflowError> {
    use gam_linalg::matrix::LinearOperator;

    if config.frailty.as_ref().is_some_and(FrailtySpec::is_active) {
        return Err(WorkflowError::InvalidConfig {
            reason: "expectile regression does not support frailty; use a survival/frailty-aware family instead"
                .to_string(),
        });
    }

    // Inner fits are ordinary Gaussian-identity GAMs; the τ asymmetry lives
    // entirely in the per-iteration prior weights this driver injects.
    let gaussian_config = FitConfig {
        family: Some("gaussian".to_string()),
        link: Some("identity".to_string()),
        expectile_tau: None,
        // The inner Gaussian-identity design carries no frailty. Normalize the
        // CLI/config-layer null value (`Some(FrailtySpec::None)`) to `None` so
        // the expectile driver does not leak survival-only plumbing into the
        // standard-family materializer, while the active-frailty guard above
        // still rejects unsupported frailty requests explicitly.
        frailty: None,
        ..config.clone()
    };

    // Materialize once to capture the fixed training design, response, offset,
    // and base prior weights. The design (basis, penalties, identifiability
    // transforms) does not depend on the prior weights, so it is reused across
    // every LAWS iteration; only the weight vector and the resulting β change.
    let base_mat = materialize(formula, data, &gaussian_config)?;
    let FitRequest::Standard(base_request) = base_mat.request else {
        return Err(WorkflowError::InvalidConfig {
            reason: "expectile regression is only defined for standard (non-survival, \
                     non-location-scale) responses"
                .to_string(),
        });
    };
    let StandardFitRequest {
        data: design_data,
        y,
        weights: base_weights,
        offset,
        spec,
        family: materialized_family,
        estimate_tweedie_p: _,
        options,
        kappa_options,
        wiggle,
        coefficient_groups,
        penalty_block_gamma_priors,
        latent_coord,
        _marker,
    } = base_request;
    // The materializer already resolved the inner family to Gaussian-identity
    // from `gaussian_config`; assert it so a future materializer change that
    // silently picked a different family for `"gaussian"` is caught here rather
    // than producing a non-expectile fit.
    if !materialized_family.is_gaussian_identity() {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "expectile LAWS requires a Gaussian-identity inner family; materializer produced {}",
                materialized_family.name()
            ),
        });
    }

    if wiggle.is_some() || latent_coord.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "expectile regression does not support flexible-link wiggle or latent \
                     coordinates"
                .to_string(),
        });
    }

    let n = y.len();
    let gaussian_family = LikelihoodSpec::gaussian_identity();
    // Cold start: τ = 0.5 (symmetric) weights ⇒ the first inner fit is the OLS
    // mean GAM, the natural warm start for any τ.
    let mut weights = base_weights.clone();
    let mut last_sign: Option<Vec<bool>> = None;
    let mut last_result: Option<StandardFitResult> = None;

    // The sign pattern has 2ⁿ values but LAWS visits a monotone-descent subset;
    // empirically a handful of iterations suffice. The cap is a safety guard:
    // on the rare oscillation between two equal-objective sign patterns (only
    // possible when rows sit exactly on the fitted surface) the last fit is a
    // valid τ-expectile of the perturbation-stable problem, so returning it is
    // correct rather than an error.
    const MAX_LAWS_ITERS: usize = 50;

    for _iter in 0..MAX_LAWS_ITERS {
        let request = StandardFitRequest {
            data: design_data.clone(),
            y: y.clone(),
            weights: weights.clone(),
            offset: offset.clone(),
            spec: spec.clone(),
            family: gaussian_family.clone(),
            // Expectile LAWS fits a Gaussian-identity inner family; no Tweedie
            // power to estimate (#2026).
            estimate_tweedie_p: false,
            options: options.clone(),
            kappa_options: kappa_options.clone(),
            wiggle: None,
            coefficient_groups: coefficient_groups.clone(),
            penalty_block_gamma_priors: penalty_block_gamma_priors.clone(),
            latent_coord: None,
            _marker,
        };
        let result = fit_standard_model(request)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })?;

        // Training-scale fitted mean μ = X·β (identity link, zero-checked
        // offset folded by the design path). The design columns match the
        // combined coefficient vector exactly (the same contract `predict`
        // and the safety tests rely on).
        let mu = result.design.design.apply(&result.fit.beta);
        if mu.len() != n {
            return Err(WorkflowError::IntegrationFailed {
                reason: format!(
                    "expectile LAWS: fitted mean length {} disagrees with response length {n}",
                    mu.len()
                ),
            });
        }
        let mut mu_off = mu;
        mu_off += &offset;

        let sign: Vec<bool> = (0..n).map(|i| y[i] > mu_off[i]).collect();
        let converged = last_sign.as_ref().is_some_and(|prev| prev == &sign);
        weights = expectile_row_weights(y.view(), mu_off.view(), base_weights.view(), tau);
        last_sign = Some(sign);
        last_result = Some(result);
        if converged {
            break;
        }
    }

    let result = last_result.ok_or_else(|| WorkflowError::IntegrationFailed {
        reason: "expectile LAWS produced no fit".to_string(),
    })?;
    Ok(result)
}
/// Detection seam for the exact O(n) cubic-smoothing-spline fast path.
///
/// This is the EARLIEST point in the standard workflow where a materialized
/// fit request carries everything needed to prove the model is exactly the
/// problem the scan solves: a Gaussian likelihood with identity link over
/// `intercept + one 1-D cubic-class penalized smooth` — i.e. the penalized
/// least-squares problem `min Σ w_i (y_i − f(x_i))² + λ∫f″²` with an
/// unpenalized `{1, x}` null space. The Kalman/RTS scan computes that
/// posterior (mean, pointwise variance, exact diffuse REML for λ) in O(n) per
/// λ-trial instead of the dense design/Gram O(n·k²) + O(k³) route.
///
/// Returns `Some` only when ALL of the following hold; everything else falls
/// through to the dense path:
/// - family is Gaussian + identity link;
/// - no link wiggle, no latent coordinates, no coefficient groups, no penalty
///   hyperpriors, no linear/box constraints, no Firth, no adaptive
///   regularization, no Kronecker systems, no externally injected null-space
///   dims;
/// - the term collection is exactly one smooth term — no linear terms, no
///   random effects, no by-variables / factor interactions;
/// - that smooth is a plain 1-D B-spline whose penalty order is compatible
///   with the exact scan and whose null space is unshrunk
///   (`double_penalty=false`). `double_penalty` (mgcv `select = TRUE`) on a free
///   B-spline emits a second REML coordinate — the Marra & Wood (2011) null-space
///   shrinkage block — that the scan cannot represent (its polynomial null space
///   is an improper diffuse prior it can never shrink); routing such a fit
///   through the scan would silently drop that penalty and select λ from the
///   bending penalty alone, which is exactly the EDF inflation #1266 reports.
///   Those fits fall through to the dense two-rho path, which owns both penalties
///   jointly. Natural cubic regression (`bs="cr"`/`"cs"`) terms also fall
///   through: their knot-value parameterization is a finite-rank regression
///   spline, not the scan's full smoothing-spline state-space posterior;
/// - the offset is identically zero and every weight is finite and positive;
/// - at least 3 distinct finite abscissae (the scan's diffuse rank plus one).
///
/// λ-mapping note: the scan's penalty is exactly `λ∫f″²` (state-space
/// `q = 1/λ` at unit σ²). The dense 1-D B-spline path penalizes the same
/// cubic class through a reduced-rank discrete-difference Gram whose
/// normalization differs by a basis-dependent constant, so a λ selected by
/// one parameterization does not transfer numerically to the other. The scan
/// therefore always re-selects λ by its own exact diffuse REML criterion
/// (the optimizer of the same restricted likelihood, expressed in the scan's
/// parameterization); user-pinned smoothing parameters are not representable
/// at this seam (the formula DSL exposes none for this term class), so no
/// pinned-λ mapping arises.
///
/// Identifiability transforms on the smooth (centering / linear-trend
/// removal / orthogonality-to-intercept) are accepted as eligible: they only
/// re-coordinate the unpenalized null space against the implicit intercept
/// and do not change the fitted posterior of `E[y|x]`, which is what the
/// scan returns directly.
pub fn spline_scan_fast_path(request: &StandardFitRequest<'_>) -> Option<SplineScanInputs> {
    if !request.family.is_gaussian_identity() {
        return None;
    }
    if request.wiggle.is_some()
        || request.latent_coord.is_some()
        || !request.coefficient_groups.is_empty()
        || !request.penalty_block_gamma_priors.is_empty()
    {
        return None;
    }
    let options = &request.options;
    if options.latent_cloglog.is_some()
        || options.mixture_link.is_some()
        || options.sas_link.is_some()
        || options.linear_constraints.is_some()
        || options.adaptive_regularization.is_some()
        || options.kronecker_penalty_system.is_some()
        || options.kronecker_factored.is_some()
        || options.firth_bias_reduction
        || !options.nullspace_dims.is_empty()
    {
        return None;
    }
    let spec = &request.spec;
    if !spec.linear_terms.is_empty()
        || !spec.random_effect_terms.is_empty()
        || spec.smooth_terms.len() != 1
    {
        return None;
    }
    let term = &spec.smooth_terms[0];
    if !matches!(term.shape, gam_terms::smooth::ShapeConstraint::None)
        || term.joint_null_rotation.is_some()
    {
        return None;
    }
    let gam_terms::smooth::SmoothBasisSpec::BSpline1D {
        feature_col,
        spec: bspec,
    } = &term.basis
    else {
        return None;
    };
    // Smoothing-spline order m = penalty_order ∈ {1, 2, 3}. The exact scan
    // integrates the order-m integrated-Wiener prior whose natural spline has
    // degree 2m−1 (m=1 → linear, m=2 → cubic, m=3 → quintic), so require that
    // degree to match user intent. The de Jong exact diffuse leading-block
    // smoother (#1044) handles the m−1 partially-diffuse leading nodes for all
    // m ≤ MAX_ORDER; m > MAX_ORDER falls through to the dense path.
    let order = bspec.penalty_order;
    // Double-penalty (mgcv `select = TRUE`) is NOT representable by the scan and
    // must fall through to the dense two-rho path (#1266). On a free B-spline the
    // double penalty emits a *second* REML coordinate — the Marra & Wood (2011)
    // null-space shrinkage block `Z Zᵀ` (see `bspline_penalty_candidates`) —
    // whose entire purpose is to let REML shrink the unpenalized `{1, x, …}`
    // polynomial null space toward `EDF → 0` for an unsupported term. The scan,
    // by construction, carries that null space as an *improper diffuse* prior it
    // can never shrink (its EDF floor is the null-space dimension `order`), so
    // routing a `double_penalty` fit through it silently DROPS the second penalty
    // and selects λ from the single bending penalty alone. The scan's own exact
    // diffuse REML then genuinely prefers a mildly wiggly fit at finite λ for
    // some noise realizations (an interior REML optimum, EDF ≈ 3–4), which is the
    // EDF inflation #1266 reports. The dense path owns both penalties jointly and
    // its outer REML, seeded into the over-smoothing basin, drives the null space
    // out (EDF → null-space dim) when the data are truly polynomial. Excluding
    // `double_penalty` here keeps such a fit on the dense path; single-penalty
    // and boundary-conditioned single-penalty B-splines keep the exact O(n) scan.
    if !(1..=3).contains(&order)
        || bspec.degree != 2 * order - 1
        || bspec.double_penalty
        || !bspec.boundary_conditions.is_free()
        || !matches!(bspec.boundary, gam_terms::basis::OneDimensionalBoundary::Open)
        || matches!(
            bspec.knotspec,
            gam_terms::basis::BSplineKnotSpec::PeriodicUniform { .. }
                | gam_terms::basis::BSplineKnotSpec::NaturalCubicRegression { .. }
        )
        // mgcv `bs="cr"`/`"cs"` materialise a `NaturalCubicRegression` value-knot
        // spec: a Lancaster–Salkauskas cubic-regression basis whose columns
        // index `f(x*_i)` at `k` quantile knots — a genuinely DIFFERENT finite
        // basis (and hence a different penalized posterior) from the free
        // integrated-Wiener natural spline the exact scan solves on the raw data
        // points. The scan builds its own knots from `x` and ignores this spec,
        // so routing a cr fit through it would silently solve the wrong model and
        // (per #1844) return a non-`Standard` `SplineScan` result the predict-time
        // design replay cannot reconstruct. Keep cr/cs on the dense path.
        || matches!(
            bspec.knotspec,
            gam_terms::basis::BSplineKnotSpec::NaturalCubicRegression { .. }
        )
    {
        return None;
    }
    if request.offset.iter().any(|&v| v != 0.0) {
        return None;
    }
    if request.weights.iter().any(|&v| !(v.is_finite() && v > 0.0)) {
        return None;
    }
    if *feature_col >= request.data.ncols() || request.y.len() != request.data.nrows() {
        return None;
    }
    let x: Vec<f64> = request.data.column(*feature_col).iter().copied().collect();
    let y: Vec<f64> = request.y.iter().copied().collect();
    let w: Vec<f64> = request.weights.iter().copied().collect();
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // The diffuse polynomial null space consumes `order` innovations; the scan
    // needs at least one proper innovation beyond them to profile σ².
    let mut sorted = x.clone();
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();
    if sorted.len() < order + 1 {
        return None;
    }
    Some(SplineScanInputs { x, y, w, order })
}

/// Formula-level entry for the exact O(n) cubic-smoothing-spline fast path.
///
/// Materializes the formula exactly like [`fit_from_formula`], then runs the
/// [`spline_scan_fast_path`] detection on the resulting standard request.
/// When detection fires the fit is routed through
/// [`gam_solve::spline_scan::fit_spline_scan`] — the exact diffuse
/// REML Kalman/RTS scan — and the full in-memory posterior
/// ([`gam_solve::spline_scan::SplineScanFit`]: knots, smoothed
/// states, pointwise variances, lag-one gains, σ², log λ, exact EDF, and an
/// exact `predict`) is returned. `Ok(None)` means the model is not the
/// scan-eligible shape and the caller should use the dense
/// [`fit_from_formula`] path; this keeps every persistence-bearing consumer
/// (model save, CLI, FFI) transparently on the dense fit, whose saved payload
/// the scan does not yet have a schema for.
pub fn fit_spline_scan_from_formula(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<Option<gam_solve::spline_scan::SplineScanFit>, WorkflowError> {
    let mat = materialize(formula, data, config)?;
    let FitRequest::Standard(request) = mat.request else {
        return Ok(None);
    };
    let Some(inputs) = spline_scan_fast_path(&request) else {
        return Ok(None);
    };
    gam_solve::spline_scan::fit_spline_scan(&inputs.x, &inputs.y, &inputs.w, inputs.order)
        .map(Some)
        .map_err(|reason| WorkflowError::IntegrationFailed { reason })
}

/// #1464 diagnostic entry point: evaluate the EXACT production fixed-κ
/// profiled-REML criterion (`fixed_kappa_profiled_reml_score`, the same one the
/// joint-fit κ-sign scan uses) at a list of pinned κ values for the first
/// constant-curvature term of `formula`, materialised from `data`/`config`
/// exactly like [`fit_from_formula`]. Returns `(κ, V_p(κ))` pairs.
///
/// This settles solver-vs-criterion for the railing bug: if `V_p(+κ) < V_p(−κ)`
/// for a genuinely HYPERBOLIC dataset, the criterion itself prefers the collapsed
/// +κ corner — the bug is in the constant-curvature REML/Occam term, not the
/// optimiser. If `V_p(−κ) < V_p(+κ)` yet the full fit still returns +κ, the bug
/// is in the solver/readback. The profiled fit pins κ and profiles only ρ
/// (κ-optimisation disabled), so each returned score is the negative-log-evidence
/// the outer loop minimises.
pub fn constant_curvature_profiled_reml_scores(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
    kappas: &[f64],
) -> Result<Vec<(f64, f64)>, WorkflowError> {
    let mat = materialize(formula, data, config)?;
    let FitRequest::Standard(request) = mat.request else {
        return Err(WorkflowError::IntegrationFailed {
            reason: "constant_curvature_profiled_reml_scores: formula did not materialise to a \
                     standard fit request"
                .to_string(),
        });
    };
    let term_idx = *crate::fit_orchestration::drivers::constant_curvature_term_indices(&request.spec)
        .first()
        .ok_or_else(|| WorkflowError::IntegrationFailed {
            reason: "constant_curvature_profiled_reml_scores: formula has no constant-curvature \
                     curv() term"
                .to_string(),
        })?;
    let mut out = Vec::with_capacity(kappas.len());
    for &kappa in kappas {
        let score = crate::fit_orchestration::drivers::fixed_kappa_profiled_reml_score(
            request.data.view(),
            request.y.view(),
            request.weights.view(),
            request.offset.view(),
            &request.spec,
            term_idx,
            kappa,
            request.family.clone(),
            &request.options,
        )
        .map_err(|e| WorkflowError::IntegrationFailed {
            reason: format!(
                "constant_curvature_profiled_reml_scores: fixed-κ fit at κ={kappa} failed: {e}"
            ),
        })?;
        out.push((kappa, score));
    }
    Ok(out)
}

/// Derived dense-kernel cliff: the cascade auto-route fires only once the dense
/// radial basis the smooth would otherwise use has SATURATED at its center cap
/// (`default_num_centers == K_MAX`), so the dense `O(n·K² + K³)` kernel solve
/// can no longer grow resolution with `n` and the streaming cascade's
/// `O(n·polylog)` is the only path that keeps improving. This is the structural
/// "past the dense-kernel cliff" condition the issue names — derived from the
/// dense sizing rule, NOT a magic n constant or a user flag.
fn past_dense_kernel_cliff(n: usize, d: usize) -> bool {
    // `default_num_centers` clamps to K_MAX = 2000; equality means the dense
    // basis is pinned at the cap and cannot densify further with n.
    const DENSE_CENTER_CAP: usize = 2000;
    gam_terms::basis::default_num_centers(n, d) >= DENSE_CENTER_CAP
}

/// Map a Duchon/Matérn smoothness order onto the cascade's Sobolev order,
/// clamped into the Wendland-(3,1) native window `(d/2, (d+3)/2]` (issue
/// caveat 1: the multilevel frame can only represent up to `H^{(d+3)/2}`).
fn cascade_sobolev_order(requested: f64, d: usize) -> f64 {
    let lo = d as f64 / 2.0;
    let hi = (d as f64 + 3.0) / 2.0;
    // Nudge strictly inside the open lower bound when the request lands on it.
    let eps = 1e-6 * (hi - lo);
    requested.clamp(lo + eps, hi)
}

/// Detection seam for the O(n log n) multiresolution residual-cascade fast path
/// (issue #1032).
///
/// This mirrors [`spline_scan_fast_path`] in shape but carries one CRITICAL
/// difference dictated by the issue: the cascade is **not** the same posterior
/// as the Duchon/Matérn term it stands in for (a different finite basis — the
/// multilevel Wendland frame, not the reduced-rank radial kernel). So unlike
/// the 1-D scan, which silently swaps an identical posterior, this path must
/// only fire as an explicit alternative estimator on the structural signature
/// the issue names, never as a transparent replacement. It returns `Some` only
/// when ALL of the following hold:
/// - family is Gaussian + identity link (the scattered low-d smooth the
///   cascade solves);
/// - none of the exotic-link / constraint / Firth / Kronecker / coefficient-
///   group / hyperprior machinery is engaged;
/// - the model is exactly one smooth term — no linear terms, no random
///   effects, no by-variables;
/// - that smooth is a scattered radial spatial smooth (`Duchon` or `Matern`)
///   over `d ∈ {2, 3}` coordinates with no shape constraint;
/// - the offset is identically zero and every weight is finite and positive;
/// - `n` is past the derived dense-kernel cliff
///   ([`past_dense_kernel_cliff`]) — below it the dense radial path is both
///   exact-posterior and cheap, so there is no reason to change estimators.
///
/// The returned [`ResidualCascadeInputs`] carry a unit per-axis metric (the
/// spec's isotropic radial distance); the quasi-uniformity guard inside
/// [`gam_solve::residual_cascade::fit_residual_cascade`] (issue caveat 2)
/// is the no-regression gate that refuses the iterative solve — and forces the
/// caller back to the dense path — when a near-degenerate metric would break
/// the BPX iteration bound.
pub fn residual_cascade_fast_path(
    request: &StandardFitRequest<'_>,
) -> Option<ResidualCascadeInputs> {
    if !request.family.is_gaussian_identity() {
        return None;
    }
    if request.wiggle.is_some()
        || request.latent_coord.is_some()
        || !request.coefficient_groups.is_empty()
        || !request.penalty_block_gamma_priors.is_empty()
    {
        return None;
    }
    let options = &request.options;
    if options.latent_cloglog.is_some()
        || options.mixture_link.is_some()
        || options.sas_link.is_some()
        || options.linear_constraints.is_some()
        || options.adaptive_regularization.is_some()
        || options.kronecker_penalty_system.is_some()
        || options.kronecker_factored.is_some()
        || options.firth_bias_reduction
        || !options.nullspace_dims.is_empty()
    {
        return None;
    }
    let spec = &request.spec;
    if !spec.linear_terms.is_empty()
        || !spec.random_effect_terms.is_empty()
        || spec.smooth_terms.len() != 1
    {
        return None;
    }
    let term = &spec.smooth_terms[0];
    if !matches!(term.shape, gam_terms::smooth::ShapeConstraint::None)
        || term.joint_null_rotation.is_some()
    {
        return None;
    }
    // Only scattered radial spatial smooths (Duchon / Matérn) over 2–3 axes.
    // The Duchon spectral power `p + s` and the Matérn order set the requested
    // Sobolev smoothness; both clamp into the Wendland native window.
    let (feature_cols, requested_s) = match &term.basis {
        gam_terms::smooth::SmoothBasisSpec::Duchon {
            feature_cols, spec, ..
        } => {
            // Pure-Duchon native order is `p + s` (kernel exponent 2(p+s)−d);
            // the multilevel frame targets the same continuum smoothness. `p`
            // is the polynomial nullspace degree, `s` the spectral power.
            let p = match spec.nullspace_order {
                gam_terms::basis::DuchonNullspaceOrder::Zero => 0.0,
                gam_terms::basis::DuchonNullspaceOrder::Linear => 1.0,
                gam_terms::basis::DuchonNullspaceOrder::Degree(k) => k as f64,
            };
            (feature_cols, spec.power + p)
        }
        gam_terms::smooth::SmoothBasisSpec::Matern {
            feature_cols, spec, ..
        } => {
            // Matérn smoothness ν sets native Sobolev order ν + d/2; the cascade
            // frame represents up to (d+3)/2, so the clamp below applies the
            // ceiling. (d is known just below from feature_cols.)
            let nu = spec.nu.half_integer_value();
            (feature_cols, nu + feature_cols.len() as f64 / 2.0)
        }
        _ => return None,
    };
    let d = feature_cols.len();
    if !(2..=3).contains(&d) {
        return None;
    }
    if request.offset.iter().any(|&v| v != 0.0) {
        return None;
    }
    if request.weights.iter().any(|&v| !(v.is_finite() && v > 0.0)) {
        return None;
    }
    let n = request.y.len();
    if n != request.data.nrows() || feature_cols.iter().any(|&c| c >= request.data.ncols()) {
        return None;
    }
    if !past_dense_kernel_cliff(n, d) {
        return None;
    }
    let coords: Vec<Vec<f64>> = feature_cols
        .iter()
        .map(|&c| request.data.column(c).iter().copied().collect())
        .collect();
    let y: Vec<f64> = request.y.iter().copied().collect();
    let w: Vec<f64> = request.weights.iter().copied().collect();
    if coords
        .iter()
        .any(|axis| axis.iter().any(|v| !v.is_finite()))
        || y.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let metric = vec![1.0_f64; d];
    let sobolev_s = cascade_sobolev_order(requested_s, d);
    Some(ResidualCascadeInputs {
        coords,
        y,
        w,
        metric,
        sobolev_s,
    })
}

/// Formula-level library entry for the O(n log n) residual-cascade fast path
/// (issue #1032).
///
/// Materializes the formula exactly like [`fit_from_formula`], runs the
/// [`residual_cascade_fast_path`] detection, and — when it fires AND the
/// quasi-uniformity guard inside the cascade certifies the metric — returns the
/// certified [`ResidualCascadeFit`](gam_solve::residual_cascade::ResidualCascadeFit).
/// `Ok(None)` means EITHER the model is not the cascade-eligible shape OR the
/// quasi-uniformity guard rejected the metric; in both cases the caller falls
/// back to the dense [`fit_from_formula`] path (the cascade is a different
/// posterior, so the fallback is a genuine estimator choice, never a silent
/// swap). This keeps every persistence-bearing consumer on the dense fit until
/// the cascade payload schema lands.
pub fn fit_residual_cascade_from_formula(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<Option<gam_solve::residual_cascade::ResidualCascadeFit>, WorkflowError> {
    let mat = materialize(formula, data, config)?;
    let FitRequest::Standard(request) = mat.request else {
        return Ok(None);
    };
    let Some(inputs) = residual_cascade_fast_path(&request) else {
        return Ok(None);
    };
    let coord_refs: Vec<&[f64]> = inputs.coords.iter().map(Vec::as_slice).collect();
    match gam_solve::residual_cascade::fit_residual_cascade(
        &coord_refs,
        &inputs.y,
        &inputs.w,
        &inputs.metric,
        inputs.sobolev_s,
    ) {
        Ok(fit) => Ok(Some(fit)),
        // The quasi-uniformity guard (caveat 2) and any degenerate-design
        // signal both surface as a build/solve error; treat them as "not
        // cascade-eligible" so the caller falls back to the dense kernel path
        // rather than failing the fit outright.
        Err(_) => Ok(None),
    }
}

/// Parse a formula, resolve it against a dataset, and produce a ready-to-fit `FitRequest`.
fn family_requests_transformation_normal(family: Option<&str>) -> bool {
    family
        .map(|name| name.trim().to_ascii_lowercase().replace('_', "-"))
        .as_deref()
        == Some("transformation-normal")
}

pub fn materialize<'a>(
    formula: &str,
    data: &'a Dataset,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    gam_gpu::configure_global_policy(config.gpu_policy);
    let parsed = parse_formula(formula)?;
    let col_map = data.column_map();
    let family_transformation_normal =
        family_requests_transformation_normal(config.family.as_deref());
    let transformation_normal_config;
    let effective_config = if family_transformation_normal && !config.transformation_normal {
        // `family="transformation-normal"` is a documented spelling of the CTN
        // model class, not a Gaussian identity likelihood. Normalize it into the
        // same orchestration flag used by `transformation_normal=true` before any
        // dispatch/validation branch can silently treat the request as standard.
        transformation_normal_config = FitConfig {
            transformation_normal: true,
            ..config.clone()
        };
        &transformation_normal_config
    } else {
        config
    };

    if let Some((left_col, right_col, event_col)) = parse_surv_interval_response(&parsed.response)?
    {
        if effective_config.transformation_normal {
            return Err(WorkflowError::InvalidConfig {
                reason:
                    "transformation_normal cannot be combined with a SurvInterval(...) response"
                        .to_string(),
            });
        }
        // Interval censoring `T ∈ (L, R]` is only defined for the latent
        // hazard-window survival likelihood, whose kernel carries the
        // `log[S(L) − S(R)]` interval contribution. Route the left boundary `L`
        // through the standard exit channel and the right boundary `R` through
        // the dedicated interval-right channel; `event_col` distinguishes
        // bracketed (interval) rows from right-censored rows beyond the last
        // inspection (which carry an infinite/sentinel `R`).
        materialize_survival(
            &parsed,
            data,
            &col_map,
            effective_config,
            None,
            &left_col,
            &event_col,
            Some(&right_col),
        )
    } else if let Some((entry_col, exit_col, event_col)) = parse_surv_response(&parsed.response)? {
        if effective_config.transformation_normal {
            return Err(WorkflowError::InvalidConfig {
                reason: "transformation_normal cannot be combined with a Surv(...) response"
                    .to_string(),
            });
        }
        // `materialize_*` now return `WorkflowError` directly so the typed
        // `ColumnNotFound` payload (and any future variant-typed leaf
        // errors) survive the dispatcher hop instead of being flattened
        // into `IntegrationFailed { reason: String }`.
        materialize_survival(
            &parsed,
            data,
            &col_map,
            effective_config,
            entry_col.as_deref(),
            &exit_col,
            &event_col,
            None,
        )
    } else {
        // Non-survival response: `timewiggle(...)` and `survmodel(...)` are
        // structurally meaningless (there is no baseline hazard / time axis to
        // wiggle and no survival likelihood to configure). They are parsed into
        // `ParsedFormula` but consumed *only* by `materialize_survival`; without
        // this guard every non-survival materializer below would silently drop
        // them, fitting an ordinary GAM while the user believes they requested a
        // time-varying / survival model (#371). Reject here — the single
        // chokepoint for all non-survival paths — mirroring the symmetric
        // auxiliary-formula rejection in `validate_auxiliary_formula_controls`.
        reject_survival_only_terms_for_nonsurvival(&parsed)?;
        // Symmetrically, the `config.survival_likelihood` *knob* selects a
        // survival likelihood mode read only by `materialize_survival`. On this
        // non-survival branch a non-default value (e.g. "weibull") would be
        // discarded and the fit would silently degrade to an ordinary GAM
        // (#1767). Reject it at the same chokepoint.
        reject_survival_likelihood_for_nonsurvival(effective_config)?;
        if effective_config.transformation_normal {
            // Issue #789A: a Bernoulli marginal-slope request with
            // `transformation_normal=true` used to dispatch as a CTN fit while
            // retaining marginal-slope controls, leaving the transformation path
            // in a non-advancing loop. CTN score calibration now uses the
            // explicit `ctn_stage1` recipe instead, so the legacy boolean is a
            // hard configuration error for marginal-slope requests.
            reject_marginal_slope_controls_for_transformation_normal(effective_config)?;
            if effective_config.noise_formula.is_some() {
                return Err(WorkflowError::InvalidConfig {
                    reason: "transformation_normal cannot be combined with noise_formula"
                        .to_string(),
                });
            }
            materialize_transformation_normal(&parsed, data, &col_map, effective_config)
        } else if requests_bernoulli_marginal_slope(effective_config) {
            materialize_bernoulli_marginal_slope(&parsed, data, &col_map, effective_config)
        } else if effective_config.noise_formula.is_some() {
            materialize_location_scale(&parsed, data, &col_map, effective_config)
        } else {
            materialize_standard(&parsed, data, &col_map, effective_config)
        }
    }
}

#[cfg(test)]
mod sz_factor_smooth_recovery_tests {
    // `super::*` brings in `Dataset` (= gam_data::EncodedDataset), `FitConfig`,
    // `FitResult`, `StandardFitResult`, and `fit_from_formula`.
    use super::*;

    const NOISE_SD: f64 = 0.20;
    const N: usize = 4000;
    const N_GROUPS: usize = 4;

    /// A simple deterministic LCG so the dataset is reproducible without pulling
    /// an RNG dependency into the test.
    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            // Numerical Recipes LCG constants.
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        /// Uniform in [0, 1).
        fn unif(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Standard normal via Box–Muller (one of the pair).
        fn normal(&mut self) -> f64 {
            let u1 = (self.unif()).max(1e-12);
            let u2 = self.unif();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// Data drawn from EXACTLY the `sz` model class: a shared smooth `f0(x)` plus
    /// zero-sum per-group deviations `d_g(x)` (phase-shifted sinusoids whose
    /// cross-group mean is removed at every `x`), plus observation noise. This
    /// mirrors the (blocked) Python bug-hunt test `tests/bug_hunt_sz_factor_
    /// smooth_underfits_own_model_class_test.py`.
    ///
    /// Written to a CSV and loaded through the real `load_dataset_projected`
    /// inferer so the grouping column `g` (string levels) is encoded as a genuine
    /// categorical exactly as production does — hand-built `EncodedDataset`s do
    /// not carry the categorical level map the factor-smooth level resolver needs.
    fn sz_class_dataset() -> (Dataset, tempfile::TempDir) {
        let mut rng = Lcg(0x5326_2026_0628_1605);
        let phases: Vec<f64> = (0..N_GROUPS)
            .map(|k| 1.2 * k as f64 / (N_GROUPS as f64 - 1.0))
            .collect();
        let deviations = |xi: f64| -> Vec<f64> {
            let vals: Vec<f64> = phases
                .iter()
                .map(|p| 0.6 * (std::f64::consts::TAU * xi + std::f64::consts::TAU * p).sin())
                .collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            vals.iter().map(|v| v - mean).collect()
        };

        let mut csv = String::from("y,x,g\n");
        for _ in 0..N {
            let x = rng.unif();
            // Use the HIGH bits (via `unif`) for the group draw — an LCG's low
            // bits have a tiny period and would collapse `% N_GROUPS` to a near
            // constant.
            let g = ((rng.unif() * N_GROUPS as f64) as usize).min(N_GROUPS - 1);
            let f0 = (std::f64::consts::TAU * x).sin();
            let mu = f0 + deviations(x)[g];
            let y = mu + NOISE_SD * rng.normal();
            csv.push_str(&format!("{y},{x},g{g}\n"));
        }
        let td = tempfile::tempdir().expect("tempdir");
        let path = td.path().join("sz_class.csv");
        std::fs::write(&path, csv).expect("write sz-class csv");
        // Force `g` into a categorical role exactly as the formula intends so the
        // factor-smooth level resolver sees all `N_GROUPS` distinct levels.
        let mut roles = std::collections::HashSet::new();
        roles.insert("g");
        let data = gam_data::load_dataset_projected_with_categorical_roles(
            &path,
            &["y".to_string(), "x".to_string(), "g".to_string()],
            &roles,
        )
        .expect("load sz-class dataset");
        (data, td)
    }

    fn gaussian_config() -> FitConfig {
        FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() }
    }

    /// In-sample residual sd of a fitted standard GAM: `sd(y − Xβ̂)`.
    fn residual_sd(fit: &StandardFitResult, data: &Dataset) -> f64 {
        let beta = &fit.fit.beta;
        let design = &fit.design.design;
        let n = design.nrows();
        assert_eq!(design.ncols(), beta.len(), "design/beta width mismatch");
        let mut fitted = vec![0.0f64; n];
        // `try_row_chunk` materializes contiguous row blocks of whatever design
        // storage the fit used (dense or block-lazy) — robust to the storage kind.
        const CHUNK: usize = 512;
        let mut start = 0usize;
        while start < n {
            let end = (start + CHUNK).min(n);
            let block = design
                .try_row_chunk(start..end)
                .expect("materialize design row chunk");
            for (r, row) in block.rows().into_iter().enumerate() {
                let mut acc = 0.0;
                for (c, &xv) in row.iter().enumerate() {
                    acc += xv * beta[c];
                }
                fitted[start + r] = acc;
            }
            start = end;
        }
        let y = data.values.column(0);
        let resid: Vec<f64> = y.iter().zip(fitted.iter()).map(|(&yi, &fi)| yi - fi).collect();
        let mean = resid.iter().sum::<f64>() / resid.len() as f64;
        let var = resid.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / resid.len() as f64;
        var.sqrt()
    }

    fn fit_standard(formula: &str, data: &Dataset) -> StandardFitResult {
        match fit_from_formula(formula, data, &gaussian_config())
            .unwrap_or_else(|e| panic!("fit `{formula}` failed: {e:?}"))
        {
            FitResult::Standard(r) => r,
            other => panic!("expected Standard fit for `{formula}`, got a different variant: {}",
                std::any::type_name_of_val(&other)),
        }
    }

    /// #1605 (gold standard, end-to-end REML fit): the sum-to-zero factor smooth
    /// `s(x) + s(g, x, bs="sz")` must RECOVER data drawn from its own model class
    /// to the observation-noise floor, exactly as the strictly-more-general
    /// `s(x, g, bs="fs")` superset provably does.
    ///
    /// The recovery gap (`sz` resid ≈ 0.43 ≈ 2.1× the 0.20 floor while `fs`
    /// reaches the floor) was closed by THREE mgcv-faithful corrections, each
    /// necessary, that this end-to-end fit jointly exercises:
    ///   1. marginal basis (baef17e): cr → curvature-capable B-spline, so a
    ///      deviation with non-zero boundary curvature is representable;
    ///   2. ownership/overlap residualization (b49bb5c): the `sz` deviation is
    ///      sum-to-zero ACROSS the grouping factor, hence orthogonal to a
    ///      factor-independent owner like the shared `s(x)`. Residualizing it
    ///      against `s(x)`'s realized span (the #978 chart) collapsed every
    ///      group's curve to a flat per-group contrast; skipping that ownership
    ///      (same family as the #1276 factor-`by` level gate) restores the curve
    ///      shape and stops REML railing the shared `s(x)` wiggliness λ;
    ///   3. null-space ridge (this change): the `sz` deviation blocks now carry
    ///      the per-null-dimension ridge structure of `fs`, mapped into the
    ///      zero-sum contrast space, so the {const, linear} null space is
    ///      shrinkable per dimension (the #700/#712/#713 partial-pooling form)
    ///      rather than left free — without breaking the zero-sum constraint.
    ///
    /// This is the gold-standard verification: it drives the real
    /// `fit_from_formula` REML λ-selection on data drawn from exactly the `sz`
    /// model class and asserts `sz` reaches the floor (and a `fs` control does
    /// too). It failed before the fixes and passes after.
    #[test]
    fn sz_factor_smooth_recovers_its_own_model_class_end_to_end() {
        let (data, _td) = sz_class_dataset();

        // Control: bs="fs", a strict superset of the sz span, must reach the
        // noise floor — proves the data is well-posed and pins the floor.
        let fs_fit = fit_standard("y ~ s(x, g, bs='fs')", &data);
        let fs_resid = residual_sd(&fs_fit, &data);
        assert!(
            fs_resid < 1.2 * NOISE_SD,
            "control bs='fs' did not reach the noise floor: resid_sd={fs_resid:.4} \
             vs noise_sd={NOISE_SD} (data/floor sanity check)",
        );

        // The documented sz idiom on data drawn from the sz model class.
        let sz_fit = fit_standard("y ~ s(x) + s(g, x, bs='sz')", &data);
        let sz_resid = residual_sd(&sz_fit, &data);

        // A smoother whose span contains the truth, fit at large n, must explain
        // the systematic structure and leave ~only observation noise.
        assert!(
            sz_resid < 1.4 * NOISE_SD,
            "bs='sz' under-fits its own model class: resid_sd={sz_resid:.4} \
             ({:.2}x the noise floor {NOISE_SD}); the bs='fs' superset reached \
             {fs_resid:.4}. The sz fit leaves systematic signal in the residual.",
            sz_resid / NOISE_SD,
        );

        // Comparative guard: sz must not be dramatically worse than the fs
        // superset that recovers the same data.
        assert!(
            sz_resid < 1.5 * fs_resid,
            "bs='sz' residual {sz_resid:.4} is {:.2}x the bs='fs' residual \
             {fs_resid:.4} on identical sz-class data",
            sz_resid / fs_resid,
        );
    }
}
