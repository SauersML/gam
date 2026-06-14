use super::*;

pub fn fit_model(request: FitRequest<'_>) -> Result<FitResult, WorkflowError> {
    // Single warm-start chokepoint: open a persistent cache session
    // keyed on the FitRequest's exact family-shape fingerprint, and
    // attach it to the variant's BlockwiseFitOptions (or top-level
    // slot). Family-specific fit functions then consult
    // `options.cache_session` to seed θ from the last accepted iterate
    // and checkpoint accepted iterates. This is what makes warm-start
    // uniform across every model class — adding a new family does NOT
    // require remembering to wire the cache.
    //
    // Hierarchical near-match: if the exact key has no entry, try a
    // data-independent seed prefix. Two fits that share the same family,
    // link, baseline, and column structure but differ in their row sets
    // (cross-validation folds, hyperparameter sweeps, anything refitting
    // structurally identical models on related data) get their first
    // fit's ρ as a seed instead of cold-starting. Save always goes to
    // the exact key so future identical refits get exact hits.
    let mut request = request;
    let exact_key = request.cache_key();
    let seed_key = request.cache_seed_key();
    if let Some(session) = crate::solver::persistent_warm_start::open_outer_session(&exact_key) {
        let exact_present = session.peek_load().is_some();
        if !exact_present
            && let Some(seed) =
                crate::solver::persistent_warm_start::lookup_outer_iterate_payload(&seed_key)
        {
            let prior_obj = seed.objective.unwrap_or(f64::NAN);
            log::info!(
                "[CACHE] seed key={}.. via prefix family={} prior_obj={:.6e}",
                &exact_key[..8.min(exact_key.len())],
                request.family_tag(),
                prior_obj,
            );
            session.preload(seed);
        }
        request.attach_cache_session(session);
    }
    // Mirror checkpoints and finalize to the seed-prefix keyspace so the
    // *next* fit with related-but-not-identical structure can pick this run's
    // ρ up via the prefix lookup above, even if the current process is killed
    // before convergence.
    let mirror_session = crate::solver::persistent_warm_start::open_outer_session(&seed_key);
    if let Some(mirror) = mirror_session.as_ref() {
        request.attach_cache_mirror(Arc::clone(mirror));
    }
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
        FitRequest::SurvivalLocationScale(request) => {
            // Outermost defensive catch: any path that surfaces empty
            // `block_states` to a family method (multiple possible —
            // `validate_joint_states`, `blockwise_fit_from_parts`, etc.)
            // produces a cryptic "expects N blocks, got 0" Python
            // exception. The v0.3.31 projected-kernel REML fix should make
            // the outer ρ-gradient cancel at large λ, preventing the
            // ARC stall that triggers the downstream empty-states path —
            // but as a last resort we convert any remaining error of that
            // shape into a user-actionable message so the Python caller
            // sees a clear cause instead of a Rust assertion artifact.
            match fit_survival_location_scale_model(request).map(FitResult::SurvivalLocationScale) {
                Ok(fit) => Ok(fit),
                Err(e)
                    if e.contains("expects 3 blocks, got 0")
                        || e.contains("expects 4 blocks, got 0")
                        || (e.contains("block_states") && e.contains("got 0"))
                        || e.contains("blockwise fit requires at least one block state") =>
                {
                    Err(WorkflowError::IntegrationFailed {
                        reason: format!(
                            "survival location-scale fit failed: the smoothing-parameter optimizer \
                             landed at a degenerate iterate where the inner solver's block state \
                             was empty. This is the symptom of an under-identified smooth driven \
                             to a numerically pathological λ (e.g. exp(20+)) on a small-data \
                             subsample. Try: (1) reducing covariate count, (2) increasing n_train, \
                             (3) `baseline_target=\"linear\"` to drop the parametric baseline, or \
                             (4) `noise_formula=\"1\"` to drop the noise GAM. Underlying error: {e}"
                        ),
                    })
                }
                Err(reason) => Err(wrap_solver_err(reason)),
            }
        }
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
/// Resolve the [`crate::resource::ResourcePolicy`] backing term construction
/// for a given [`FitConfig`] + dataset.
///
/// If the caller hasn't supplied an explicit policy override, derive one from
/// the shape of the problem via
/// [`crate::resource::ResourcePolicy::for_problem`]. At large scale (n_rows
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
    hints: crate::resource::ProblemHints,
) -> crate::resource::ResourcePolicy {
    if let Some(p) = config.resource_policy.clone() {
        return p;
    }
    crate::resource::ResourcePolicy::for_problem(data.values.nrows(), 0, hints)
}


pub(crate) fn marginal_slope_hints(config: &FitConfig) -> crate::resource::ProblemHints {
    crate::resource::ProblemHints {
        marginal_slope_large_scale_active: config.logslope_formula.is_some()
            || config.z_column.is_some(),
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
    if let Some(tau) = expectile_tau_for_config(config)? {
        return fit_expectile_laws(formula, data, config, tau);
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
        if let Some(inputs) = spline_scan_fast_path(request) {
            let scan = crate::solver::spline_scan::fit_spline_scan(
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
            if let Ok(fit) = crate::solver::residual_cascade::fit_residual_cascade(
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
) -> Result<FitResult, WorkflowError> {
    use crate::linalg::matrix::LinearOperator;

    // Inner fits are ordinary Gaussian-identity GAMs; the τ asymmetry lives
    // entirely in the per-iteration prior weights this driver injects.
    let gaussian_config = FitConfig {
        family: Some("gaussian".to_string()),
        link: Some("identity".to_string()),
        expectile_tau: None,
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
    Ok(FitResult::Standard(result))
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
/// - that smooth is a plain 1-D cubic (degree 3) B-spline with a single
///   order-2 penalty (`double_penalty=false` — the default `s(x)` adds a
///   second null-space shrinkage penalty, which the scan's diffuse `{1, x}`
///   null space deliberately does NOT shrink, so it is not the same
///   posterior and is excluded), open (non-cyclic) boundary, free endpoint
///   conditions, and no shape constraint;
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
    if !matches!(term.shape, crate::smooth::ShapeConstraint::None)
        || term.joint_null_rotation.is_some()
    {
        return None;
    }
    let crate::smooth::SmoothBasisSpec::BSpline1D {
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
    if !(1..=3).contains(&order)
        || bspec.degree != 2 * order - 1
        || bspec.double_penalty
        || !bspec.boundary_conditions.is_free()
        || !matches!(bspec.boundary, crate::basis::OneDimensionalBoundary::Open)
        || matches!(
            bspec.knotspec,
            crate::basis::BSplineKnotSpec::PeriodicUniform { .. }
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
/// [`crate::solver::spline_scan::fit_spline_scan`] — the exact diffuse
/// REML Kalman/RTS scan — and the full in-memory posterior
/// ([`crate::solver::spline_scan::SplineScanFit`]: knots, smoothed
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
) -> Result<Option<crate::solver::spline_scan::SplineScanFit>, WorkflowError> {
    let mat = materialize(formula, data, config)?;
    let FitRequest::Standard(request) = mat.request else {
        return Ok(None);
    };
    let Some(inputs) = spline_scan_fast_path(&request) else {
        return Ok(None);
    };
    crate::solver::spline_scan::fit_spline_scan(&inputs.x, &inputs.y, &inputs.w, inputs.order)
        .map(Some)
        .map_err(|reason| WorkflowError::IntegrationFailed { reason })
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
    crate::terms::basis::default_num_centers(n, d) >= DENSE_CENTER_CAP
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
/// [`crate::solver::residual_cascade::fit_residual_cascade`] (issue caveat 2)
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
    if !matches!(term.shape, crate::smooth::ShapeConstraint::None)
        || term.joint_null_rotation.is_some()
    {
        return None;
    }
    // Only scattered radial spatial smooths (Duchon / Matérn) over 2–3 axes.
    // The Duchon spectral power `p + s` and the Matérn order set the requested
    // Sobolev smoothness; both clamp into the Wendland native window.
    let (feature_cols, requested_s) = match &term.basis {
        crate::smooth::SmoothBasisSpec::Duchon {
            feature_cols, spec, ..
        } => {
            // Pure-Duchon native order is `p + s` (kernel exponent 2(p+s)−d);
            // the multilevel frame targets the same continuum smoothness. `p`
            // is the polynomial nullspace degree, `s` the spectral power.
            let p = match spec.nullspace_order {
                crate::basis::DuchonNullspaceOrder::Zero => 0.0,
                crate::basis::DuchonNullspaceOrder::Linear => 1.0,
                crate::basis::DuchonNullspaceOrder::Degree(k) => k as f64,
            };
            (feature_cols, spec.power + p)
        }
        crate::smooth::SmoothBasisSpec::Matern {
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
/// certified [`ResidualCascadeFit`](crate::solver::residual_cascade::ResidualCascadeFit).
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
) -> Result<Option<crate::solver::residual_cascade::ResidualCascadeFit>, WorkflowError> {
    let mat = materialize(formula, data, config)?;
    let FitRequest::Standard(request) = mat.request else {
        return Ok(None);
    };
    let Some(inputs) = residual_cascade_fast_path(&request) else {
        return Ok(None);
    };
    let coord_refs: Vec<&[f64]> = inputs.coords.iter().map(Vec::as_slice).collect();
    match crate::solver::residual_cascade::fit_residual_cascade(
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
pub fn materialize<'a>(
    formula: &str,
    data: &'a Dataset,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    crate::gpu::configure_global_policy(config.gpu_policy);
    let parsed = parse_formula(formula)?;
    let col_map = data.column_map();

    if let Some((left_col, right_col, event_col)) = parse_surv_interval_response(&parsed.response)? {
        if config.transformation_normal {
            return Err(WorkflowError::InvalidConfig {
                reason: "transformation_normal cannot be combined with a SurvInterval(...) response"
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
            config,
            None,
            &left_col,
            &event_col,
            Some(&right_col),
        )
    } else if let Some((entry_col, exit_col, event_col)) = parse_surv_response(&parsed.response)? {
        if config.transformation_normal {
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
            config,
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
        if config.transformation_normal {
            // Issue #789A: a Bernoulli marginal-slope request with
            // `transformation_normal=true` used to dispatch as a CTN fit while
            // retaining marginal-slope controls, leaving the transformation path
            // in a non-advancing loop. CTN score calibration now uses the
            // explicit `ctn_stage1` recipe instead, so the legacy boolean is a
            // hard configuration error for marginal-slope requests.
            reject_marginal_slope_controls_for_transformation_normal(config)?;
            if config.noise_formula.is_some() {
                return Err(WorkflowError::InvalidConfig {
                    reason: "transformation_normal cannot be combined with noise_formula"
                        .to_string(),
                });
            }
            materialize_transformation_normal(&parsed, data, &col_map, config)
        } else if config.logslope_formula.is_some() || config.z_column.is_some() {
            materialize_bernoulli_marginal_slope(&parsed, data, &col_map, config)
        } else if config.noise_formula.is_some() {
            materialize_location_scale(&parsed, data, &col_map, config)
        } else {
            materialize_standard(&parsed, data, &col_map, config)
        }
    }
}
