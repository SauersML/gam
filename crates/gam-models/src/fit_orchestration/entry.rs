use super::*;
use crate::inference::model_payload_builders::standard_fit_comparable_reml_score;
use gam_linalg::matrix::LinearOperator;
use gam_problem::FailureCategory;
use gam_solve::estimate::reml::reml_outer_engine::penalty_matrix_root;
use gam_terms::smooth::AdaptiveResolution;

/// Request-specific inputs to the canonical standard-fit `FitOptions`.
///
/// Everything in here varies per call (the link state extracted from the
/// formula/config, the linear constraints synthesized from `bounded()` /
/// shape-constrained terms, the Firth toggle read
/// off the `FitConfig`). Every *policy* field of `FitOptions` — the ones that
/// decide HOW the outer REML optimization behaves (`compute_inference`,
/// `skip_rho_posterior_inference`, `tol`, and the `max_iter` default) — is
/// filled in by [`canonical_standard_fit_options`] and is
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
}

/// The single source of truth for standard-fit `FitOptions` *policy*.
///
/// Both standard-fit entry points — `materialize_standard` (the formula /
/// Python / PyO3 path) and the `gam` CLI's `run_fit` — construct their
/// `StandardFitRequest` options through this function, so the outer REML
/// optimization policy (`compute_inference`, `skip_rho_posterior_inference`,
/// `tol` and `max_iter` default) is identical by
/// construction. New policy fields must be set HERE, never re-derived at a call
/// site, which is what makes Python/CLI behavioral divergence structurally
/// impossible rather than enforced by parallel-but-equal code (#1196).
pub fn canonical_standard_fit_options(
    config: &FitConfig,
    inputs: StandardFitOptionsInputs,
) -> FitOptions {
    FitOptions {
        resource_policy: resolved_resource_policy(
            config,
            gam_runtime::resource::ProblemHints::default(),
        ),
        latent_cloglog: inputs.latent_cloglog,
        mixture_link: inputs.mixture_link,
        optimize_mixture: inputs.optimize_mixture,
        sas_link: inputs.sas_link,
        optimize_sas: inputs.optimize_sas,
        // Posterior covariance is always computed so `predict --uncertainty`
        // works for every family (the `COV_MAX_P` diagonal fallback caps cost).
        compute_inference: true,
        // Formula/CLI fits are the interactive/default path: keep coefficient
        // covariance and the analytic first-order smoothing correction, which
        // the returned fit needs. The rho-posterior adequacy diagnostic (Tier-0
        // PSIS over dozens of refits, and its Tier-1/Tier-2 escalations) has no
        // reader on this path, so the fit publishes
        // `NotComputed(InferenceNotRequested)` and spends no criterion
        // evaluation on it (#3010); lower-level callers that read it request it
        // (`skip_rho_posterior_inference: false`).
        skip_rho_posterior_inference: true,
        // The count for the loops that still take one: the negative-binomial
        // alternation, the expectile LAWS iterations, the bounded-effect
        // custom-family search and the latent-coordinate joint search. Each
        // refuses with a typed error when it runs out without its certificate.
        // The standard REML/LAML search takes no count (#2817).
        max_iter: 200,
        // Outer REML/LAML smoothing-selection tolerance. `1e-10` (effective
        // projected-gradient threshold ≈ 1e-7) resolves λ̂ to optimiser
        // precision and restores the `w=c ⇔ c-fold replication` invariance in
        // smoothing selection (gam#893). The CLI previously used the stale
        // `1e-6`, which over-smoothed relative to the formula path.
        tol: 1e-10,
        nullspace_dims: vec![],
        linear_constraints: inputs.linear_constraints,
        firth_bias_reduction: inputs.firth_bias_reduction,
        rho_prior: Default::default(),
        // Explicit opt-in only. Clones share the same lazy/opened store handle;
        // no formula fit consults ambient process state.
        persistent_warm_start_store: config.persistent_warm_start_store.clone(),
    }
}

/// A fit failure raised in this module under the category it belongs to.
fn raised_fit_failure(category: FailureCategory, reason: impl Into<String>) -> WorkflowError {
    WorkflowError::Fit(FitFailure::raised(category, reason))
}

/// An exact smoothing-spline scan refusal, under its category.
fn spline_scan_failure(error: gam_solve::spline_scan::SplineScoreProofError) -> WorkflowError {
    use gam_solve::spline_scan::SplineScoreProofError as E;
    let category = match &error {
        E::InnovationContainsZero { .. }
        | E::NonPositiveInnovation { .. }
        | E::NonPositiveProfileResidual { .. }
        | E::InvalidArithmetic { .. }
        | E::AccumulatorDiverged { .. } => FailureCategory::Numerical,
        E::InvalidInput(_) => FailureCategory::Input,
        E::MissingEndpointCertificate { .. }
        | E::GlobalValueOrderingUnresolved { .. }
        | E::OptimumKktUncertified { .. }
        | E::Search(_) => FailureCategory::Convergence,
        // Prose from several kinds of computation.
        E::Computation(_) => FailureCategory::Unclassified,
    };
    raised_fit_failure(category, error.to_string())
}

/// A residual-cascade proof or convergence refusal, under its category.
fn residual_cascade_failure(
    error: gam_solve::residual_cascade::ResidualCascadeError,
) -> WorkflowError {
    use gam_solve::residual_cascade::ResidualCascadeError as E;
    let category = match &error {
        // "Invalid input or a numerical failure": prose from two kinds.
        E::Computation(_) => FailureCategory::Unclassified,
        E::RemlScoreProofUnavailable { .. }
        | E::RemlOptimumResolutionFlat { .. }
        | E::RemlScoreSearchUndecomposable { .. }
        | E::RemlValueOrderingUnresolved { .. }
        | E::Underresolved { .. } => FailureCategory::Convergence,
    };
    raised_fit_failure(category, error.to_string())
}

/// The REML fit of a standard request whose exact Gaussian boundary
/// (`try_deterministic_gaussian_standard_fit`) has already been refused. That
/// certificate builds its own dense design and normal equations, so a caller
/// that already ran it hands the request here rather than back through
/// [`fit_model`], which would build and refuse it a second time. Like
/// [`fit_model`], it runs on the process worker pool.
///
/// `realized_design` is the design that certificate realized, when it built one;
/// the REML fit starts from it instead of realizing the same design again.
fn fit_standard_past_exact_gaussian_boundary(
    request: StandardFitRequest<'_>,
    realized_design: Option<TermCollectionDesign>,
) -> Result<FitResult, WorkflowError> {
    gam_runtime::parallel::install(move || {
        fit_standard_model_on_design(request, realized_design)
            .map(FitResult::Standard)
            .map_err(|failure| WorkflowError::from(failure.ending_the_fit()))
    })
}

/// Fit a materialized request on the process worker pool
/// ([`gam_runtime::parallel::install`]), so every parallel operation of the fit
/// runs on the pool that belongs to this process.
pub fn fit_model(request: FitRequest<'_>) -> Result<FitResult, WorkflowError> {
    gam_runtime::parallel::install(move || fit_model_on_pool(request))
}

fn fit_model_on_pool(request: FitRequest<'_>) -> Result<FitResult, WorkflowError> {
    // Every arm hands back the helper's `FitFailure` whole. This boundary used
    // to wrap each helper's text as `IntegrationFailed`, so every solver
    // failure reached Python as `IntegrationError` whatever had failed (#2937).
    // A fit that ended holding an uncertified inner solve is named here, where
    // no outer search is left to step away from it (#2943).
    let wrap_solver_err =
        |failure: FitFailure| -> WorkflowError { WorkflowError::from(failure.ending_the_fit()) };
    match request {
        FitRequest::Standard(request) => match try_deterministic_gaussian_standard_fit(&request)? {
            GaussianStandardRoute::Exact(fitted) => Ok(FitResult::Standard(fitted)),
            GaussianStandardRoute::Iterative(design) => {
                fit_standard_past_exact_gaussian_boundary(request, design)
            }
        },
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
/// If the caller hasn't supplied an explicit policy override, delegate to
/// [`gam_runtime::resource::ResourcePolicy::for_problem`]. Non-structural paths
/// no longer switch mode at row/column thresholds: each planned allocation is
/// admitted from its checked live-byte footprint against the process-wide
/// memory governor. Consequently there is no speculative pre-spec coefficient
/// estimate to compute here (and no small-n/large-p classification cliff);
/// `ProblemHints` remains the structural signal for operator-only estimators.
pub(crate) fn resolved_resource_policy(
    config: &FitConfig,
    hints: gam_runtime::resource::ProblemHints,
) -> gam_runtime::resource::ResourcePolicy {
    if let Some(p) = config.resource_policy.clone() {
        return p;
    }
    gam_runtime::resource::ResourcePolicy::for_problem(hints)
}

/// Parse, materialize, and fit a model in one call.
/// Resolve the expectile levels requested by `config`, if any.
///
/// Thin typed-error wrapper over [`FitConfig::resolved_expectile_levels`],
/// the one rule [`FitConfig::resolve`] also enforces: `Some(levels)` for the
/// expectile family, `None` for every other family, and `Err` for a malformed
/// expectile request or an `expectile_tau` given with a non-expectile family.
pub fn expectile_levels_for_config(config: &FitConfig) -> Result<Option<Vec<f64>>, WorkflowError> {
    config
        .resolved_expectile_levels()
        .map_err(|reason| WorkflowError::InvalidConfig { reason })
}

/// Prior-weighted empirical `τ`-expectile of `z` in closed form.
///
/// The expectile is the unique root `c` of the strictly decreasing, piecewise
/// linear estimating function
/// `g(c) = τ·Σ_{zᵢ>c} pᵢ(zᵢ − c) − (1 − τ)·Σ_{zᵢ≤c} pᵢ(c − zᵢ)`.
/// Sorting `z` and carrying the prefix sums `A = Σ p`, `B = Σ p·z` over the
/// rows below the root's segment, `g` is linear on that segment and vanishes
/// exactly at `c = [τ(Z − B) + (1 − τ)B] / [τ(P − A) + (1 − τ)A]` with totals
/// `P`, `Z`. The segment is the first one whose right sorted endpoint has
/// `g ≤ 0`; no iteration or tolerance is involved.
fn weighted_empirical_expectile(z: &[f64], p: &[f64], tau: f64) -> Result<f64, String> {
    if z.len() != p.len() || z.is_empty() {
        return Err(format!(
            "weighted expectile needs matching non-empty inputs; got {} values and {} weights",
            z.len(),
            p.len()
        ));
    }
    if z.iter().any(|v| !v.is_finite()) || p.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(
            "weighted expectile needs finite values and finite non-negative weights".to_string(),
        );
    }
    let mut order: Vec<usize> = (0..z.len()).collect();
    order.sort_by(|&a, &b| z[a].total_cmp(&z[b]));
    let total_p: f64 = p.iter().sum();
    let total_z: f64 = z.iter().zip(p).map(|(v, w)| v * w).sum();
    if !(total_p > 0.0) {
        return Err("weighted expectile needs a positive total weight".to_string());
    }
    let root = |below_p: f64, below_z: f64| {
        (tau * (total_z - below_z) + (1.0 - tau) * below_z)
            / (tau * (total_p - below_p) + (1.0 - tau) * below_p)
    };
    let (mut below_p, mut below_z) = (0.0_f64, 0.0_f64);
    for &i in &order {
        let at = z[i];
        let (upto_p, upto_z) = (below_p + p[i], below_z + p[i] * z[i]);
        let g = tau * ((total_z - upto_z) - at * (total_p - upto_p))
            - (1.0 - tau) * (at * upto_p - upto_z);
        if g <= 0.0 {
            // The root lies on the segment ending at `at`, whose lower set is
            // the rows strictly before this one.
            return Ok(root(below_p, below_z).min(at));
        }
        below_p = upto_p;
        below_z = upto_z;
    }
    // Unreachable for positive total weight: g at the largest value is ≤ 0.
    Ok(root(below_p, below_z))
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

/// Constant-history cycle detector for the deterministic LAWS sign map.
///
/// Brent's power-of-two schedule detects a cycle of any length while retaining
/// one `Vec<bool>` checkpoint, rather than one sign vector per iteration.  That
/// keeps cycle detection O(n) in the number of observations even when a caller
/// grants a large iteration budget.
#[derive(Debug, Default)]
struct ExpectileSignCycle {
    anchor: Option<Vec<bool>>,
    power: usize,
    span: usize,
}

impl ExpectileSignCycle {
    /// Observe the next sign state. Returns the detected cycle length once the
    /// current state revisits Brent's anchor.
    fn observe(&mut self, sign: &[bool]) -> Option<usize> {
        let Some(anchor) = self.anchor.as_deref() else {
            self.anchor = Some(sign.to_vec());
            self.power = 1;
            return None;
        };

        self.span += 1;
        if anchor == sign {
            return Some(self.span);
        }
        if self.span == self.power {
            self.anchor = Some(sign.to_vec());
            self.power = self.power.saturating_mul(2);
            self.span = 0;
        }
        None
    }
}

/// Dimensionless KKT residual for the asymmetric objective at a frozen-weight
/// WLS solution.
///
/// For coefficient `j`, `d_j = x_j'((w_frozen - w_target) ⊙ r)` is the
/// gradient defect introduced by using the old residual signs.  Normalize it
/// by `sqrt((x_j' W_audit x_j) (r' W_audit r))`, its Cauchy–Schwarz scale with
/// `W_audit = max(W_frozen, W_target)`.  The maximum coordinate residual is
/// invariant to response scale, column scale, and a common rescaling of prior
/// weights; unlike a score-relative ratio, it remains meaningful when the
/// frozen unpenalized score cancels to zero.
fn expectile_kkt_residual(
    design: &gam_linalg::matrix::DesignMatrix,
    residual: ArrayView1<'_, f64>,
    frozen_weights: ArrayView1<'_, f64>,
    target_weights: ArrayView1<'_, f64>,
) -> Result<f64, String> {
    use gam_linalg::matrix::LinearOperator;

    let n = design.nrows();
    if residual.len() != n || frozen_weights.len() != n || target_weights.len() != n {
        return Err(format!(
            "expectile KKT dimension mismatch: design rows={n}, residual={}, frozen weights={}, \
             target weights={}",
            residual.len(),
            frozen_weights.len(),
            target_weights.len(),
        ));
    }
    if residual.iter().any(|v| !v.is_finite())
        || frozen_weights
            .iter()
            .chain(target_weights.iter())
            .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(
            "expectile KKT audit requires finite residuals and finite non-negative weights"
                .to_string(),
        );
    }

    let mut row_scratch =
        Array1::from_shape_fn(n, |i| (frozen_weights[i] - target_weights[i]) * residual[i]);
    let defect = design.apply_transpose(&row_scratch);
    for i in 0..n {
        row_scratch[i] = frozen_weights[i].max(target_weights[i]);
    }
    let energy = (0..n)
        .map(|i| row_scratch[i] * residual[i] * residual[i])
        .sum::<f64>();
    if !energy.is_finite() || energy < 0.0 {
        return Err(format!(
            "expectile KKT audit produced invalid residual energy {energy:?}"
        ));
    }
    let gram_diag = design.diag_gram(&row_scratch)?;
    if defect.len() != gram_diag.len()
        || defect.iter().any(|v| !v.is_finite())
        || gram_diag.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("expectile KKT audit produced invalid score/Gram evidence".to_string());
    }

    let mut max_scaled = 0.0_f64;
    for (&d, &q) in defect.iter().zip(gram_diag.iter()) {
        let denominator_squared = q * energy;
        let scaled = if denominator_squared > 0.0 {
            d.abs() / denominator_squared.sqrt()
        } else if d == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        max_scaled = max_scaled.max(scaled);
    }
    Ok(max_scaled)
}

/// Sensitivity `∂r_F/∂a_F` of the LAWS residuals on the rows `F` to their own
/// asymmetry levels `a_l` (row weight `w_l = base_l·a_l`), through both the
/// frozen-λ weighted solve and the REML-selected smoothing parameters.
///
/// With `H = XᵀWX + S_λ` and `u_k = λ_k S_k(β̂ − m_k)`, implicit
/// differentiation of the normal equations `XᵀW r = S_λ(β̂ − m)` gives
/// `∂β̂/∂w_l = H⁻¹x_l r_l` at fixed ρ and `∂β̂/∂ρ_k = −H⁻¹u_k`. The REML
/// optimum moves by `dρ̂/dw_l = −V_ρ g_l`, where `V_ρ` is the certified inverse
/// of the outer ρ-Hessian the fit published and `g_l = ∂_{w_l}∇_ρ V` is the
/// mixed partial of the negative log restricted likelihood:
///
///   `g_{l,k} = −½ λ_k (H⁻¹x_l)ᵀ S_k (H⁻¹x_l) + u_kᵀH⁻¹x_l · r_l / φ̂ + O(r_l²)`.
///
/// The first term is `∂_{w_l} ½∂_{ρ_k} log|H|`, exact because `H` is affine
/// in `w_l`; the second is the penalty-quadratic `u_kᵀ∂β̂/∂w_l / φ̂` shared
/// by the profiled and the joint-φ criteria (their `r_l²` terms differ and are
/// dropped). Every omitted term is `O(r_l²)`, so the Jacobian is exact at
/// the generalized fixed point `r_F = 0` it is used to locate. Assembled:
///
///   `∂r_j/∂a_l = base_l·(−x_jᵀH⁻¹x_l r_l + Σ_k (x_jᵀH⁻¹u_k)(dρ̂_k/dw_l))`.
///
/// `H⁻¹` is `Vb/φ̂`, which already carries any identifiability or active
/// constraint projection `Z(ZᵀHZ)⁻¹Zᵀ`. A railed or unidentified ρ direction
/// has zero `V_ρ` rows, i.e. it does not move with the weights.
fn expectile_free_row_jacobian(
    fit: &gam_solve::estimate::UnifiedFitResult,
    design: &TermCollectionDesign,
    residual: ArrayView1<'_, f64>,
    base_weights: ArrayView1<'_, f64>,
    rows: &[usize],
) -> Result<Array2<f64>, String> {
    let n = design.design.nrows();
    let p = fit.beta.len();
    let k_count = fit.lambdas.len();
    let vb = fit.covariance_conditional.as_ref().ok_or_else(|| {
        "the inner fit published no dense conditional covariance Vb = φ̂·H⁻¹".to_string()
    })?;
    let v_rho = fit.artifacts.rho_covariance.as_ref().ok_or_else(|| {
        "the inner fit published no certified inverse outer ρ-Hessian".to_string()
    })?;
    let phi = fit
        .coefficient_covariance_scale()
        .map_err(|error| error.to_string())?;
    if !(phi.is_finite() && phi > 0.0) {
        return Err(format!(
            "the inner fit's coefficient covariance scale must be finite and positive, got {phi:?}"
        ));
    }
    if design.design.ncols() != p
        || vb.dim() != (p, p)
        || design.penalties.len() != k_count
        || v_rho.dim() != (k_count, k_count)
        || residual.len() != n
        || base_weights.len() != n
        || rows.iter().any(|&row| row >= n)
    {
        return Err(format!(
            "dimension mismatch: design {n}x{}, beta {p}, Vb {:?}, penalties {}, lambdas \
             {k_count}, V_rho {:?}, residual {}, base weights {}",
            design.design.ncols(),
            vb.dim(),
            design.penalties.len(),
            v_rho.dim(),
            residual.len(),
            base_weights.len(),
        ));
    }
    let h_inverse = |vector: &Array1<f64>| -> Array1<f64> { vb.dot(vector) / phi };

    // u_k = λ_k S_k β̂ embedded in the global coefficient vector, and its
    // H⁻¹ image −∂β̂/∂ρ_k.
    let mut penalty_scores = Vec::with_capacity(k_count);
    for (k, block) in design.penalties.iter().enumerate() {
        let range = block.col_range.clone();
        if range.end > p || block.local.dim() != (range.len(), range.len()) {
            return Err(format!(
                "penalty {k} block {range:?} with local {:?} does not fit {p} coefficients",
                block.local.dim()
            ));
        }
        let beta_block = fit.beta.slice(ndarray::s![range.clone()]);
        let mut score = Array1::<f64>::zeros(p);
        score
            .slice_mut(ndarray::s![range])
            .assign(&(block.local.dot(&beta_block) * fit.lambdas[k]));
        let image = h_inverse(&score);
        penalty_scores.push((score, image));
    }

    let rows_x: Vec<Array1<f64>> = rows
        .iter()
        .map(|&row| {
            let mut unit = Array1::<f64>::zeros(n);
            unit[row] = 1.0;
            design.design.apply_transpose(&unit)
        })
        .collect();
    let rows_z: Vec<Array1<f64>> = rows_x.iter().map(h_inverse).collect();

    let m = rows.len();
    let mut jacobian = Array2::<f64>::zeros((m, m));
    for (l_slot, &l) in rows.iter().enumerate() {
        let z_l = &rows_z[l_slot];
        let r_l = residual[l];
        // g_l = ∂_{w_l}∇_ρ V, then dρ̂/dw_l = −V_ρ g_l.
        let mixed = Array1::from_shape_fn(k_count, |k| {
            let block = &design.penalties[k];
            let z_block = z_l.slice(ndarray::s![block.col_range.clone()]);
            let log_det_term = -0.5 * fit.lambdas[k] * z_block.dot(&block.local.dot(&z_block));
            let quadratic_term = penalty_scores[k].0.dot(z_l) * r_l / phi;
            log_det_term + quadratic_term
        });
        let rho_response = -v_rho.dot(&mixed);
        for (j_slot, x_j) in rows_x.iter().enumerate() {
            let through_rho = (0..k_count)
                .map(|k| x_j.dot(&penalty_scores[k].1) * rho_response[k])
                .sum::<f64>();
            jacobian[[j_slot, l_slot]] = base_weights[l] * (-x_j.dot(z_l) * r_l + through_rho);
        }
    }
    if jacobian.iter().any(|value| !value.is_finite()) {
        return Err("the residual-weight Jacobian is not finite".to_string());
    }
    Ok(jacobian)
}

/// Solve the square system `a·x = b` through the workspace's partial-pivoting
/// LU, [`gam_linalg::faer_ndarray::FaerLu`], which is the owner of this
/// operation: it names the first column with no usable pivot instead of
/// dividing by it, so a singular system is a typed refusal here and never a
/// regularized solve. An empty system has the empty solution and nothing to
/// factor.
fn solve_square_partial_pivot(a: Array2<f64>, b: Array1<f64>) -> Result<Array1<f64>, String> {
    let m = b.len();
    if a.dim() != (m, m) {
        return Err(format!(
            "system {:?} does not match right-hand side {m}",
            a.dim()
        ));
    }
    if m == 0 {
        return Ok(b);
    }
    let view = gam_linalg::faer_ndarray::FaerArrayView::new(&a);
    let lu = gam_linalg::faer_ndarray::FaerLu::new(view.as_ref())
        .map_err(|column| format!("the {m}x{m} system is singular at column {column}"))?;
    let solved = lu.solve(faer::Mat::from_fn(m, 1, |row, _| b[row]).as_ref());
    let x = Array1::from_shape_fn(m, |row| solved[(row, 0)]);
    if x.iter().any(|value| !value.is_finite()) {
        return Err("the solution is not finite".to_string());
    }
    Ok(x)
}

#[cfg(test)]
mod expectile_convergence_tests {
    use super::{ExpectileSignCycle, expectile_kkt_residual, weighted_empirical_expectile};
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use ndarray::array;

    #[test]
    fn brent_detector_finds_fixed_sign_state() {
        let mut detector = ExpectileSignCycle::default();
        let sign = vec![true, false, true, true];
        assert_eq!(detector.observe(&sign), None);
        assert_eq!(detector.observe(&sign), Some(1));
    }

    #[test]
    fn brent_detector_finds_longer_cycle_without_storing_history() {
        let mut detector = ExpectileSignCycle::default();
        let cycle = [
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ];
        let mut detected = None;
        for sign in cycle.iter().cycle().take(9) {
            detected = detector.observe(sign);
            if detected.is_some() {
                break;
            }
        }
        assert_eq!(detected, Some(3));
        assert_eq!(detector.anchor.as_ref().map(Vec::len), Some(3));
    }

    #[test]
    fn normalized_kkt_residual_handles_a_cancelling_frozen_score() {
        let design = DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [1.0]]));
        // The frozen intercept score is exactly zero. A score-relative ratio
        // would divide the tiny target defect by itself and report O(1); the
        // Cauchy–Schwarz normalization correctly recognizes a near-tie.
        let residual = array![-1.0, 1.0];
        let frozen = array![1.0, 1.0];
        let target = array![1.0, 1.0 + 1.0e-12];
        let kkt = expectile_kkt_residual(&design, residual.view(), frozen.view(), target.view())
            .expect("finite KKT audit");
        assert!(kkt < 1.0e-10, "normalized residual was {kkt:.3e}");
    }

    #[test]
    fn normalized_kkt_residual_is_column_and_weight_scale_invariant() {
        let residual = array![-2.0, 1.0, 1.0];
        let frozen = array![1.0, 1.0, 1.0];
        let target = array![1.0, 1.25, 0.75];
        let x = array![[1.0], [2.0], [-1.0]];
        let base = DesignMatrix::Dense(DenseDesignMatrix::from(x.clone()));
        let scaled = DesignMatrix::Dense(DenseDesignMatrix::from(x * 1.0e6));
        let base_kkt = expectile_kkt_residual(&base, residual.view(), frozen.view(), target.view())
            .expect("base KKT audit");
        let scaled_kkt = expectile_kkt_residual(
            &scaled,
            residual.view(),
            (frozen.clone() * 1.0e4).view(),
            (target.clone() * 1.0e4).view(),
        )
        .expect("scaled KKT audit");
        assert!((base_kkt - scaled_kkt).abs() <= f64::EPSILON.sqrt());
    }

    /// `g(c)` from the doc comment of `weighted_empirical_expectile`.
    fn expectile_estimating_function(z: &[f64], p: &[f64], tau: f64, c: f64) -> f64 {
        z.iter()
            .zip(p)
            .map(|(&v, &w)| {
                if v > c {
                    tau * w * (v - c)
                } else {
                    -(1.0 - tau) * w * (c - v)
                }
            })
            .sum()
    }

    #[test]
    fn weighted_expectile_is_the_exact_root_of_the_estimating_function() {
        let z = [0.3, -1.7, 2.4, 0.3, -0.2, 5.1, -3.3];
        let p = [1.0, 0.5, 2.0, 0.0, 1.5, 0.25, 1.0];
        for tau in [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98] {
            let c = weighted_empirical_expectile(&z, &p, tau).expect("expectile");
            let scale: f64 = z.iter().zip(&p).map(|(v, w)| w * v.abs()).sum();
            let g = expectile_estimating_function(&z, &p, tau, c);
            assert!(g.abs() <= 1.0e-13 * scale, "tau={tau}: g(c)={g:e}");
        }
    }

    #[test]
    fn weighted_expectile_at_one_half_is_the_weighted_mean() {
        let z = [4.0, -2.0, 1.0, 7.5];
        let p = [1.0, 3.0, 0.5, 2.0];
        let mean = z.iter().zip(&p).map(|(v, w)| v * w).sum::<f64>() / p.iter().sum::<f64>();
        let c = weighted_empirical_expectile(&z, &p, 0.5).expect("expectile");
        assert!((c - mean).abs() <= 1.0e-14 * mean.abs().max(1.0));
    }

    #[test]
    fn weighted_expectile_is_strictly_increasing_in_the_level() {
        let z = [0.9, -0.4, 1.3, -2.2, 0.1, 3.0];
        let p = [1.0; 6];
        let levels = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];
        let values: Vec<f64> = levels
            .iter()
            .map(|&tau| weighted_empirical_expectile(&z, &p, tau).expect("expectile"))
            .collect();
        assert!(
            values.windows(2).all(|pair| pair[0] < pair[1]),
            "expectiles not strictly increasing: {values:?}"
        );
        assert!(values[0] > -2.2 && values[values.len() - 1] < 3.0);
    }

    #[test]
    fn weighted_expectile_ignores_zero_weight_rows_and_rejects_bad_input() {
        let with_dead = weighted_empirical_expectile(&[1.0, 100.0, 3.0], &[1.0, 0.0, 1.0], 0.8)
            .expect("expectile");
        let without =
            weighted_empirical_expectile(&[1.0, 3.0], &[1.0, 1.0], 0.8).expect("expectile");
        assert!((with_dead - without).abs() <= 1.0e-15);
        assert!(weighted_empirical_expectile(&[], &[], 0.5).is_err());
        assert!(weighted_empirical_expectile(&[1.0], &[1.0, 1.0], 0.5).is_err());
        assert!(weighted_empirical_expectile(&[1.0, 2.0], &[0.0, 0.0], 0.5).is_err());
        assert!(weighted_empirical_expectile(&[1.0, f64::NAN], &[1.0, 1.0], 0.5).is_err());
        assert!(weighted_empirical_expectile(&[1.0, 2.0], &[1.0, -1.0], 0.5).is_err());
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeterministicPenaltyFace {
    /// The exact coefficient lies in this penalty's null space, so unsupported
    /// directions are removed at the infinite-precision face.
    Infinite,
    /// This penalty acts on the exact coefficient and must vanish for the
    /// zero-residual fit to remain attainable.
    Zero,
}

struct ExactGaussianBoundary {
    /// The design the certificate was proved on.
    design: TermCollectionDesign,
    beta: Array1<f64>,
    penalty_faces: Vec<DeterministicPenaltyFace>,
}

/// What the exact Gaussian boundary certificate found.
enum ExactGaussianVerdict {
    /// The request is not a candidate; no design was realized.
    Ineligible,
    /// The realized design does not reproduce the response exactly.
    Interior(TermCollectionDesign),
    /// The resource policy refused the dense certificate on the realized
    /// design, so exactness is not decided.
    Undecided(TermCollectionDesign),
    Boundary(ExactGaussianBoundary),
}

/// Who fits a Gaussian identity standard request.
enum GaussianStandardRoute {
    /// The deterministic zero-residual boundary fit.
    Exact(StandardFitResult),
    /// The iterative REML solver, handed the design the boundary check
    /// realized from this request's `spec`, `data` and `resource_policy`, when
    /// it realized one.
    Iterative(Option<TermCollectionDesign>),
}

/// The design every standard fit of `request` realizes, built with the fit's
/// own resource policy so that the REML fit can start from it.
fn realize_standard_design(
    request: &StandardFitRequest<'_>,
) -> Result<TermCollectionDesign, gam_terms::basis::BasisError> {
    gam_terms::smooth::build_term_collection_design_with_policy(
        request.data.view(),
        &request.spec,
        &request.options.resource_policy,
    )
}

/// `Iterative` means the shortcut does not apply and the iterative REML solver
/// owns the fit; it is returned when the exact boundary face's free directions
/// are not identified by the data (see the tangent-precision factorization
/// below). Every `Err` is a malformed request, not a declined shortcut.
fn deterministic_gaussian_standard_fit(
    request: &StandardFitRequest<'_>,
    exact_boundary: Option<ExactGaussianBoundary>,
) -> Result<GaussianStandardRoute, WorkflowError> {
    if !request.family.is_gaussian_identity() || request.y.is_empty() {
        return Err(WorkflowError::InvalidConfig {
            reason:
                "deterministic Gaussian shortcut requires a non-empty Gaussian identity request"
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
            reason: "deterministic Gaussian shortcut requires finite response, offset, and non-negative weights"
                .to_string(),
        });
    }
    let weight_sum = request.weights.sum();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(WorkflowError::InvalidConfig {
            reason: "deterministic Gaussian shortcut requires positive total weight".to_string(),
        });
    }
    let (design, exact_boundary) = match exact_boundary {
        Some(ExactGaussianBoundary {
            design,
            beta,
            penalty_faces,
        }) => (design, Some((beta, penalty_faces))),
        None => (
            realize_standard_design(request).map_err(|err| WorkflowError::InvalidConfig {
                reason: format!(
                    "deterministic Gaussian shortcut could not build its design: {err}"
                ),
            })?,
            None,
        ),
    };
    let p = design.design.ncols();
    let n_penalties = design.penalties.len();
    let (beta, penalty_faces) = match exact_boundary {
        Some((beta, penalty_faces)) => {
            if beta.len() != p || penalty_faces.len() != n_penalties {
                return Err(raised_fit_failure(
                    FailureCategory::Invariant,
                    format!(
                        "deterministic Gaussian boundary does not match its design: \
                         coefficients {} vs {p}, penalty faces {} vs {n_penalties}",
                        beta.len(),
                        penalty_faces.len(),
                    ),
                ));
            }
            (beta, penalty_faces)
        }
        None => {
            // Dispatch proved every represented `y - offset` value is
            // identical. Use that exact value instead of recomputing it as a
            // weighted mean: summation round-off could contradict the
            // residual≡0 invariant even though the mathematical mean is
            // unchanged.
            let intercept = request.y[0] - request.offset[0];
            // Dispatch only takes this branch for a spec with a global
            // intercept; the realized design must agree, or the constant
            // would land on no column (or on several) and the fit published
            // as exact would not reproduce `y`.
            if design.intercept_range.len() != 1 || design.intercept_range.end > p {
                return Err(raised_fit_failure(
                    FailureCategory::Invariant,
                    format!(
                        "constant-response Gaussian shortcut needs exactly one intercept \
                         column, got {:?} in a design of width {p}",
                        design.intercept_range,
                    ),
                ));
            }
            let mut beta = Array1::<f64>::zeros(p);
            beta[design.intercept_range.start] = intercept;
            (beta, vec![DeterministicPenaltyFace::Infinite; n_penalties])
        }
    };
    let fitted_eta = design.design.apply(&beta) + &design.affine_offset + request.offset.as_ref();
    let max_abs_eta = fitted_eta
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);

    // Dispatch has proved an exact fitted response (residual ≡ 0). For the
    // penalized constant-response case, every wiggle is unsupported and shrinks
    // out; exact parametric fits carry no penalty coordinate. A fit is usable
    // only if it carries a
    // complete inference bundle — the penalized Hessian, EDF, dispersion, and
    // covariance that null-space metadata, `edf_total()`, prediction bands, and
    // the persistence payload all read. The prior shortcut returned `inference:
    // None`/`geometry: None`, so the model builder then hard-failed with
    // "null-space Hessian logdet requires fitted penalized Hessian" (#2254) even
    // for `y ~ 1`. We assemble that bundle here at a fully-smoothed λ. Because the
    // residual is exactly zero the estimated dispersion φ̂ = 0, so every
    // coefficient covariance is exactly zero (no ill-conditioned inverse needed).
    let x_dense = design
        .design
        .try_to_dense_arc_with_policy(
            "deterministic Gaussian inference bundle",
            &request.options.resource_policy,
        )
        .map_err(|reason| raised_fit_failure(FailureCategory::Input, reason))?;
    let weights = request.weights.as_ref().clone();
    let xtwx = gam_linalg::faer_ndarray::fast_xt_diag_x(&x_dense, &weights);
    let mut infinite_face_penalty = Array2::<f64>::zeros((p, p));
    for (penalty_index, block) in design.penalties.iter().enumerate() {
        let r = block.col_range.clone();
        if r.is_empty()
            || r.end > p
            || block.local.nrows() != r.len()
            || block.local.ncols() != r.len()
        {
            return Err(raised_fit_failure(
                FailureCategory::Invariant,
                format!(
                    "deterministic Gaussian shortcut received malformed penalty {penalty_index}: \
                     range={r:?}, local={}x{}, design width={p}",
                    block.local.nrows(),
                    block.local.ncols()
                ),
            ));
        }
        if block.local.iter().any(|value| !value.is_finite()) {
            return Err(raised_fit_failure(
                FailureCategory::Numerical,
                format!(
                    "deterministic Gaussian shortcut received non-finite penalty {penalty_index}"
                ),
            ));
        }
        if penalty_faces[penalty_index] == DeterministicPenaltyFace::Infinite {
            infinite_face_penalty
                .slice_mut(ndarray::s![r.clone(), r])
                .scaled_add(1.0, &block.local);
        }
    }

    // Infinite-face penalties are hard constraints; zero-face penalties vanish.
    // The exact quotient geometry below does not approximate either endpoint.
    // `UnifiedFitResult` nevertheless has a finite smoothing-coordinate schema,
    // so serialize the upper endpoint at floating-point resolution and the lower
    // endpoint at the exact supported log-strength minimum. Derive the upper
    // proxy from the ACTUAL joint infinite-face penalty spectrum: the weakest
    // numerically non-null penalty direction must dominate the largest
    // data-information scale by 1/sqrt(ε). Unlike the former `1e10` multiplier,
    // this is invariant to rescaling either X'WX or S and contains no
    // model-specific tuning knob.
    let has_infinite_face = penalty_faces
        .iter()
        .any(|face| *face == DeterministicPenaltyFace::Infinite);
    let (lambda_infinite, infinite_range_basis, infinite_null_basis, range_eigenvalues) =
        if !has_infinite_face {
            (
                0.0,
                Array2::<f64>::zeros((p, 0)),
                Array2::<f64>::eye(p),
                Vec::new(),
            )
        } else {
            use gam_linalg::faer_ndarray::FaerEigh;
            let symmetric_penalty =
                (&infinite_face_penalty + &infinite_face_penalty.t().to_owned()) * 0.5;
            let (penalty_eigenvalues, penalty_eigenvectors) =
            symmetric_penalty.eigh(faer::Side::Lower).map_err(|error| {
                raised_fit_failure(
                    FailureCategory::Numerical,
                    format!(
                        "deterministic Gaussian shortcut could not resolve the penalty spectrum: {error}"
                    ),
                )
            })?;
            let largest_penalty = penalty_eigenvalues
                .iter()
                .fold(0.0_f64, |largest, &value| largest.max(value.abs()));
            if !(largest_penalty.is_finite() && largest_penalty > 0.0) {
                return Err(raised_fit_failure(
                    FailureCategory::Numerical,
                    "deterministic Gaussian shortcut received penalties with zero numerical rank",
                ));
            }
            let rank_floor = f64::EPSILON * (p.max(1) as f64) * largest_penalty;
            if let Some(&negative) = penalty_eigenvalues
                .iter()
                .filter(|&&value| value < -rank_floor)
                .min_by(|left, right| left.total_cmp(right))
            {
                return Err(raised_fit_failure(
                    FailureCategory::Numerical,
                    format!(
                        "deterministic Gaussian shortcut received a non-PSD penalty \
                     (minimum eigenvalue {negative:.6e}, numerical floor {rank_floor:.6e})"
                    ),
                ));
            }
            let weakest_penalty = penalty_eigenvalues
                .iter()
                .copied()
                .filter(|&value| value > rank_floor)
                .min_by(|left, right| left.total_cmp(right))
                .ok_or_else(|| {
                    raised_fit_failure(
                        FailureCategory::Numerical,
                        "deterministic Gaussian shortcut could not identify a penalized direction",
                    )
                })?;
            // The induced infinity norm bounds the spectral norm of symmetric
            // X'WX. A diagonal-only scale can underestimate a highly correlated
            // design by O(p), leaving some data-informed direction insufficiently
            // constrained at the purported λ→∞ boundary.
            let information_scale = xtwx
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
                .fold(0.0_f64, f64::max);
            if !(information_scale.is_finite() && information_scale > 0.0) {
                return Err(raised_fit_failure(
                    FailureCategory::Input,
                    format!(
                        "deterministic Gaussian shortcut: the weighted design carries no information \
                     (‖X'WX‖∞ = {information_scale:e}), so the λ→∞ boundary has no scale"
                    ),
                ));
            }
            let lambda = information_scale / (f64::EPSILON.sqrt() * weakest_penalty);
            if !(lambda.is_finite() && lambda > 0.0) {
                return Err(raised_fit_failure(
                    FailureCategory::Numerical,
                    format!(
                        "deterministic Gaussian shortcut produced invalid boundary precision {lambda}"
                    ),
                ));
            }
            let range_indices: Vec<usize> = penalty_eigenvalues
                .iter()
                .enumerate()
                .filter_map(|(index, &value)| (value > rank_floor).then_some(index))
                .collect();
            let null_indices: Vec<usize> = penalty_eigenvalues
                .iter()
                .enumerate()
                .filter_map(|(index, &value)| (value <= rank_floor).then_some(index))
                .collect();
            let range_basis = penalty_eigenvectors.select(ndarray::Axis(1), &range_indices);
            let null_basis = penalty_eigenvectors.select(ndarray::Axis(1), &null_indices);
            let range_eigenvalues = range_indices
                .iter()
                .map(|&index| penalty_eigenvalues[index])
                .collect();
            (lambda, range_basis, null_basis, range_eigenvalues)
        };
    // Canonicalize λ through its log-strength coordinate BEFORE anything reads
    // it. `UnifiedFitResult` requires `lambdas[i]` to be BITWISE equal to
    // `checked_exp_log_strength(log_lambdas[i])`, and ρ is the canonical
    // coordinate everywhere else in the engine (`gam_problem::log_strength`,
    // `joint_penalty.rs:317`) — λ is DERIVED from ρ, never the reverse. This
    // shortcut computes λ straight from the penalty spectrum, so deriving
    // `ρ = ln(λ)` afterwards cannot satisfy that invariant: `exp(ln(x))` differs
    // from `x` by an ulp for most `x` and the check is exact, which is why a
    // constant-response fit reported "log_lambdas must equal ln(lambdas)
    // elementwise" (#2254) despite converging.
    //
    // Deriving BOTH stored values from this one `ρ` — rather than round-tripping
    // and hoping it is idempotent — makes the pair consistent by construction,
    // and doing it here rather than at the reporting site gives every serialized
    // smoothing field one canonical value. The exact boundary geometry below
    // does not substitute these finite proxies into its Hessian.
    let log_lambda_infinite = if has_infinite_face {
        lambda_infinite.ln()
    } else {
        0.0
    };
    let lambda_infinite = if has_infinite_face {
        gam_problem::checked_exp_log_strength(log_lambda_infinite).map_err(|error| {
            raised_fit_failure(
                FailureCategory::Numerical,
                format!(
                    "deterministic Gaussian shortcut produced a boundary precision outside the \
                     log-strength domain: {error}"
                ),
            )
        })?
    } else {
        0.0
    };
    let log_lambda_zero = gam_problem::LOG_STRENGTH_MIN;
    let lambda_zero = gam_problem::checked_exp_log_strength(log_lambda_zero).map_err(|error| {
        raised_fit_failure(
            FailureCategory::Invariant,
            format!(
                "deterministic Gaussian shortcut could not represent the zero-strength \
                 boundary: {error}"
            ),
        )
    })?;
    let log_lambdas = Array1::from_iter(penalty_faces.iter().map(|face| match face {
        DeterministicPenaltyFace::Infinite => log_lambda_infinite,
        DeterministicPenaltyFace::Zero => log_lambda_zero,
    }));
    let lambdas = Array1::from_iter(penalty_faces.iter().map(|face| match face {
        DeterministicPenaltyFace::Infinite => lambda_infinite,
        DeterministicPenaltyFace::Zero => lambda_zero,
    }));

    // At the exact mixed face, beta lies in every infinite-face null space and
    // every zero-face strength vanishes. Thus beta' S(lambda) beta is exactly
    // zero. Multiplying finite endpoint proxies would turn certified
    // annihilation round-off into an artificial positive objective term.
    let penalty_quadratic = 0.0_f64;
    // Effective degrees of freedom come from the exact constrained influence
    // matrix `F = Z (Z'X'WX Z)⁻¹ Z'X'WX`, whose trace is `dim(Z)`. The joint
    // pseudoinverse of the infinite-face penalty allocates its removed rank
    // across overlapping penalty blocks: `tr_k = tr(S_infinite⁺ S_k)`. Producing
    // the WHOLE bundle here — not just the scalar total — keeps `edf_by_block`,
    // `penalty_block_trace`, and `coefficient_influence` descriptions of the same
    // mathematical boundary.
    let (
        penalized_hessian,
        coefficient_gauge,
        edf_total,
        edf_by_block,
        penalty_block_trace,
        edf_rank_bound,
        coefficient_influence,
    ) = {
        use gam_linalg::faer_ndarray::FaerCholesky;
        // A mixed zero/infinite face has no finite ambient Hessian: representing
        // both endpoints at once creates an impossible condition number. Work
        // on its exact tangent space Z = null(S_infinite) instead. Data identify
        // A = Z' X'WX Z there; the orthogonal normal space is a hard constraint.
        let z = &infinite_null_basis;
        let u = &infinite_range_basis;
        let free_dim = z.ncols();
        // A zero face is a REML optimum only because the criterion is
        // unbounded there: the response lies exactly in a column space of
        // dimension `free_dim` that the data could have missed. That needs more
        // supported rows than free directions. With `free_dim >= n+` the free
        // columns span every supported response, so the exact fit is automatic
        // interpolation, the profiled criterion stays bounded, and certifying
        // phi = 0 with zero covariance would be wrong. The square case
        // `free_dim == n+` has a nonsingular `A` and would pass the
        // factorization below, so it is declined here.
        let supported_rows = request.weights.iter().filter(|&&w| w > 0.0).count();
        if free_dim >= supported_rows {
            return Ok(GaussianStandardRoute::Iterative(Some(design)));
        }
        let raw_free_information = z.t().dot(&xtwx.dot(z));
        let free_information = (&raw_free_information + &raw_free_information.t().to_owned()) * 0.5;
        let influence = if free_dim == 0 {
            Array2::<f64>::zeros((p, p))
        } else {
            let (equilibrated, scale) = gam_linalg::decision::equilibrate_gram(&free_information);
            // `A = Z. X.WX Z` is PSD by construction and positive DEFINITE only
            // when the data identify every free direction, i.e.
            // `rank(X Z) = dim(Z)`. That factorization IS the shortcut's
            // applicability test: the exact face is a REML optimum only where
            // the data pin every direction the face leaves free. When they do
            // not -- `free_dim > n` makes `A` singular by construction, since
            // `rank(X Z) <= n`, and a double-penalized smooth deliberately
            // admits `p > n` (only `n > M_p` is required, see
            // `reject_prefit_unidentifiable_unpenalized_space`) -- the unpenalized
            // interpolant the boundary was built from is not the optimum at
            // all: with a penalty on those directions the criterion's
            // `log|X'WX + S_λ| - log|S_λ|₊` terms move the optimum off the
            // zero face, and the shrinkage penalty, not the data, sets the
            // effective rank. So an indefinite tangent precision is the
            // shortcut declining, not the fit failing: return `None` and let
            // the iterative solver, whose inner pivoted factorization owns the
            // n-vs-rank decision, fit the model (the n=30 wine-shaped fold of
            // #1089 is exactly this shape).
            let Ok(chol) = equilibrated.cholesky(faer::Side::Lower) else {
                return Ok(GaussianStandardRoute::Iterative(Some(design)));
            };
            let solve_free = |rhs: &Array2<f64>| {
                let mut scaled_rhs = rhs.clone();
                for row in 0..scaled_rhs.nrows() {
                    scaled_rhs
                        .row_mut(row)
                        .mapv_inplace(|value| value / scale[row]);
                }
                let mut solution = chol.solve_mat(&scaled_rhs);
                for row in 0..solution.nrows() {
                    solution
                        .row_mut(row)
                        .mapv_inplace(|value| value / scale[row]);
                }
                solution
            };
            let tangent_score = z.t().dot(&xtwx);
            z.dot(&solve_free(&tangent_score))
        };
        let boundary_gauge = gam_problem::gauge::Gauge::from_block_transforms(&[z.clone()]);

        {
            let mut raw_traces = vec![0.0_f64; n_penalties];
            let mut trace_bands = vec![0.0_f64; n_penalties];
            let mut block_ranks = vec![0_usize; n_penalties];
            let mut scaled_range = u.clone();
            for (column, &eigenvalue) in range_eigenvalues.iter().enumerate() {
                scaled_range
                    .column_mut(column)
                    .mapv_inplace(|value| value / eigenvalue);
            }
            let joint_penalty_pseudoinverse = scaled_range.dot(&u.t());
            let apply_pseudoinverse = |values: &mut [f64]| -> Result<(), WorkflowError> {
                let applied = joint_penalty_pseudoinverse.dot(&ndarray::ArrayView1::from(&*values));
                for (slot, value) in values.iter_mut().zip(applied.iter()) {
                    *slot = *value;
                }
                Ok(())
            };
            let inverse_one_norm = gam_linalg::condition::estimate_inverse_one_norm(
                p,
                apply_pseudoinverse,
                apply_pseudoinverse,
            )?;
            for (kk, block) in design.penalties.iter().enumerate() {
                let r = block.col_range.clone();
                // The per-block ceiling is `rank(S_k)`, NOT the block's column
                // count: they differ by `nullity(S_k)`, a whole integer of
                // reported complexity for every penalized block, and the rank is
                // what the REML criterion already prices as `rank(S_k)·ρ_k`
                // (#2470). This path previously measured against `block_cols`
                // and so reported each block with its penalty nullity added.
                let root = penalty_matrix_root(&block.local).map_err(|reason| {
                    raised_fit_failure(
                        FailureCategory::Numerical,
                        format!(
                            "deterministic Gaussian shortcut penalty {kk} rank factorization \
                             failed: {reason}"
                        ),
                    )
                })?;
                block_ranks[kk] = root.nrows();
                if penalty_faces[kk] == DeterministicPenaltyFace::Infinite {
                    // tr(S_∞⁺S_k) = Σ_c r_cᵀ S_∞⁺ r_c over the root's modes. The
                    // pseudoinverse reproduces the root's projection onto the
                    // joint range, so that projection is the right-hand side its
                    // residual is priced against.
                    let mut root_columns = Array2::<f64>::zeros((p, root.nrows()));
                    root_columns.slice_mut(ndarray::s![r, ..]).assign(&root.t());
                    let projected = u.dot(&u.t().dot(&root_columns));
                    let solution = joint_penalty_pseudoinverse.dot(&root_columns);
                    let (trace, band) = gam_linalg::roundoff::solved_penalty_trace(
                        1.0,
                        projected.view(),
                        solution.view(),
                        infinite_face_penalty.view(),
                        inverse_one_norm,
                    )
                    .map_err(|reason| {
                        raised_fit_failure(
                            FailureCategory::Numerical,
                            format!(
                                "deterministic Gaussian shortcut penalty {kk} trace band failed: \
                                 {reason}"
                            ),
                        )
                    })?;
                    raw_traces[kk] = trace;
                    trace_bands[kk] = band;
                }
            }
            // At the mixed boundary, only infinite-face penalties constrain a
            // coefficient direction. Zero-face blocks contribute no limiting
            // rank, even though their finite representation is the smallest
            // positive strength admitted by the solver's log domain, and they
            // publish a zero trace.
            //
            // The infinite-face traces add to the joint boundary rank exactly,
            // `Σ_k tr(S_∞⁺S_k) = tr(S_∞⁺S_∞) = rank(S_∞)`. A computed sum away from
            // that rank by more than the traces' bands means the pseudoinverse and
            // the penalties are not one operator. Normalizing the shares to the
            // rank used to hide that, so it refuses instead (#2901).
            let joint_penalty_rank = u.ncols();
            let mut measured_infinite_trace = gam_math::sparse_grid::CompensatedSum::default();
            let mut infinite_trace_band = gam_math::sparse_grid::CompensatedSum::default();
            for ((&trace, &band), face) in raw_traces
                .iter()
                .zip(trace_bands.iter())
                .zip(penalty_faces.iter())
            {
                if *face == DeterministicPenaltyFace::Infinite {
                    measured_infinite_trace.add(trace);
                    infinite_trace_band.add(band);
                }
            }
            let measured_infinite_trace = measured_infinite_trace.value();
            let infinite_trace_band = infinite_trace_band.value();
            if !((measured_infinite_trace - joint_penalty_rank as f64).abs() <= infinite_trace_band)
            {
                return Err(raised_fit_failure(
                    FailureCategory::Numerical,
                    format!(
                        "deterministic Gaussian shortcut: the infinite-face traces sum to \
                         {measured_infinite_trace:.6e}, away from the joint boundary rank \
                         {joint_penalty_rank} by more than their band {infinite_trace_band:.4e}"
                    ),
                ));
            }
            // #2901: `tr(S_∞⁺S_k) ≤ rank_k` holds because `S_k ⪯ S_∞`, whatever the
            // data, so every block is certified structurally.
            let rank_bounds = vec![
                gam_solve::estimate::EdfRankBound::Certified(
                    gam_solve::estimate::EdfRankCertificate::Structural
                );
                n_penalties
            ];
            let bundle = gam_solve::estimate::penalized_edf_bundle_within_bands(
                &raw_traces,
                &trace_bands,
                &rank_bounds,
                &block_ranks,
                p,
                free_dim as f64,
            )
            .map_err(|error| WorkflowError::Fit(FitFailure::from(error)))?;
            (
                free_information,
                boundary_gauge,
                bundle.edf_total,
                bundle.edf_by_block,
                bundle.penalty_block_trace,
                bundle.rank_bound,
                Some(influence),
            )
        }
    };
    // IRLS working response for the identity link is the raw response y (η
    // absorbs the offset); the working weights are the prior weights.
    let working_response = request.y.as_ref().clone();
    let penalized_hessian_precision =
        gam_problem::dispersion_cov::UnscaledPrecision::wrap(penalized_hessian.clone());
    let inference = gam_solve::estimate::FitInference {
        edf_by_block,
        penalty_block_trace,
        edf_rank_bound,
        edf_total,
        smoothing_correction: None,
        smoothing_correction_method: None,
        smoothing_correction_first_order: None,
        smoothing_correction_method_first_order: None,
        smoothing_correction_absence: None,
        penalized_hessian: penalized_hessian_precision.clone(),
        reparam_qs: None,
        // Exact fit ⇒ residual variance is exactly zero.
        dispersion: gam_solve::estimate::Dispersion::ZERO_ESTIMATE,
        factorized_standard_errors: None,
        smoothing_correction_factorized: None,
        beta_covariance_frequentist: None,
        coefficient_influence,
        // `X'WX` is stored beside `H` in the gauge's active frame (gam#3346).
        // Every penalty vanishes on the tangent face, so there the Gram
        // `Z'X'WX Z` is the penalized Hessian itself.
        weighted_gram: Some(penalized_hessian.clone()),
        identified_subspace: None,
        // Exact fit ⇒ no working residual on any row that carries weight.
        working_residual: Some(gam_terms::inference::smooth_score_test::WorkingResidual {
            weighted_norm: 0.0,
            rows: weights.iter().filter(|&&w| w > 0.0).count(),
        }),
    };
    let geometry = Some(gam_solve::estimate::FitGeometry {
        coefficient_gauge,
        penalized_hessian: penalized_hessian_precision,
        constrained_posterior: None,
        working: Some(gam_solve::estimate::WorkingGeometry {
            weights,
            response: working_response,
        }),
    });
    let fit = gam_solve::estimate::UnifiedFitResult::try_from_parts(
        gam_solve::estimate::UnifiedFitResultParts {
            blocks: vec![gam_solve::estimate::FittedBlock {
                beta: beta.clone(),
                role: gam_problem::BlockRole::Mean,
                edf: edf_total,
                lambdas: lambdas.clone(),
            }],
            training_sample_size: request.y.len(),
            log_lambdas,
            lambdas,
            likelihood_family: Some(request.family.clone()),
            likelihood_scale: gam_problem::LikelihoodScaleMetadata::ProfiledGaussian,
            // Dispatch has certified that the fitted mean reproduces the
            // response, so `φ̂ = 0` and the profiled Gaussian likelihood is
            // unbounded: neither a normalized log-density nor a REML/LAML
            // criterion exists here. Both are DECLINED — the log-likelihood by
            // the `UserProvided` tag it has always carried, the criterion by
            // the absence `UnifiedFitResult` gained in #2595. Writing `0.0`
            // into `reml_score` was the only option before that, and it is what
            // made `Summary.raw_reml_score` report a criterion of zero on every
            // exactly-interpolating fit.
            log_likelihood_normalization: gam_problem::LogLikelihoodNormalization::UserProvided,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: None,
            // βᵀS(λ)β at the exact boundary: every infinite-face penalty
            // annihilates beta by certification and every zero-face penalty has
            // exactly zero strength. This deliberately does not evaluate the
            // finite smoothing-coordinate proxies stored for serialization.
            stable_penalty_term: penalty_quadratic,
            penalized_objective: None,
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: Some(0.0),
            standard_deviation: 0.0,
            covariance_conditional: Some(ndarray::Array2::<f64>::zeros((p, p))),
            covariance_corrected: None,
            inference: Some(inference),
            fitted_link: gam_solve::estimate::FittedLinkState::Standard(None),
            geometry,
            block_states: Vec::new(),
            pirls_status: gam_solve::pirls::PirlsStatus::Converged,
            max_abs_eta,
            constraint_kkt: None,
            artifacts: gam_solve::estimate::FitArtifacts {
                pirls: None,
                ..Default::default()
            },
            inner_cycles: 0,
        },
    )
    .map_err(|err| {
        WorkflowError::Fit(
            FitFailure::from(err).context("deterministic Gaussian shortcut produced invalid fit"),
        )
    })?;
    let resolvedspec =
        freeze_term_collection_from_design(&request.spec, &design).map_err(|err| {
            WorkflowError::InvalidConfig {
                reason: format!("deterministic Gaussian shortcut could not freeze design: {err}"),
            }
        })?;
    Ok(GaussianStandardRoute::Exact(StandardFitResult {
        fit,
        design,
        resolvedspec,
        basis_adequacy: Vec::new(),
        adaptive_bases: adaptive_bases(&request.spec),
        kappa_timing: None,
        saved_link_state: gam_solve::estimate::FittedLinkState::Standard(None),
        wiggle_knots: None,
        wiggle_degree: None,
        wiggle_penalty_metadata: None,
        wiggle_saved_warp_beta: None,
        wiggle_saved_index_shift: None,
    }))
}

fn gaussian_response_is_constant(request: &StandardFitRequest<'_>) -> bool {
    if !request.family.is_gaussian_identity() || request.y.is_empty() {
        return false;
    }
    // An inhomogeneous anchor adds a data-dependent affine channel only when
    // the term collection is realized. The shortcut predicate intentionally
    // does not build that design, so it cannot prove `y - user_offset -
    // anchor_offset` is constant. Keep such models on the ordinary exact fit
    // path; treating the user offset alone as complete would mint a false
    // zero-residual fit.
    if gam_terms::smooth::term_collection_has_nonzero_anchor(&request.spec) {
        return false;
    }
    // The shortcut carries the constant on the global intercept column. A
    // design without one (`0 + x`, `0 + factor`, an anchored B-spline gauging
    // the level) has nowhere to put it: β = 0 would leave η = offset ≠ y and
    // still be certified exact with φ̂ = 0. Such models go to the exact-boundary
    // certificate, which proves exactness from the realized design instead.
    if !gam_terms::smooth::term_collection_has_global_intercept(&request.spec) {
        return false;
    }
    // The intercept-only shortcut is exact — residual ≡ 0 — precisely when the
    // OFFSET-ADJUSTED response `y − offset` is constant: then `η = offset +
    // intercept = y` at every row. Testing the raw `y` alone would (a) miss an
    // exact fit where a varying offset cancels a varying `y`, and (b) wrongly
    // fire on a constant `y` under a varying offset, where the fit is NOT exact
    // and the zero-dispersion inference the shortcut mints would be invalid.
    if request.y.len() != request.offset.len() {
        return false;
    }
    let mut adjusted = request.y.iter().zip(request.offset.iter());
    let Some((&first_y, &first_offset)) = adjusted.next() else {
        return false;
    };
    let first = first_y - first_offset;
    if !first.is_finite() {
        return false;
    }
    for (&yi, &oi) in adjusted {
        let value = yi - oi;
        if !value.is_finite() || value != first {
            return false;
        }
    }
    true
}

fn exact_gaussian_coefficients(
    x: &Array2<f64>,
    adjusted_response: &Array1<f64>,
    weights: &Array1<f64>,
    subspace: Option<(&Array2<f64>, f64)>,
) -> Option<Array1<f64>> {
    let p = x.ncols();
    let (reduced_x_storage, basis, rotation_radius) = match subspace {
        Some((z, radius)) => (
            std::borrow::Cow::Owned(gam_linalg::faer_ndarray::fast_ab(x, z)),
            Some(z),
            radius,
        ),
        None => (std::borrow::Cow::Borrowed(x), None, 0.0),
    };
    let reduced_x: &Array2<f64> = &reduced_x_storage;
    if !rotation_radius.is_finite() {
        return None;
    }
    if adjusted_response.len() != reduced_x.nrows() || weights.len() != reduced_x.nrows() {
        return None;
    }
    let reduced_p = reduced_x.ncols();
    let beta = if reduced_p == 0 {
        Array1::<f64>::zeros(p)
    } else {
        let gram = gam_linalg::faer_ndarray::fast_xt_diag_x(reduced_x, weights);
        let rhs_matrix = gam_linalg::faer_ndarray::fast_xt_diag_y(
            reduced_x,
            weights,
            &adjusted_response.view().insert_axis(ndarray::Axis(1)),
        );
        let rhs = rhs_matrix.column(0).to_owned();
        let reduced_beta = gam_linalg::utils::certified_symmetric_solve(
            &gram,
            &rhs,
            "deterministic Gaussian normal equations",
        )
        .ok()?
        .into_solution();
        match basis {
            Some(z) => z.dot(&reduced_beta),
            None => reduced_beta,
        }
    };
    let fitted = x.dot(&beta);
    let operations = (p + 1) as f64;
    let roundoff = operations * f64::EPSILON;
    if !(roundoff < 1.0) {
        return None;
    }
    let gamma = roundoff / (1.0 - roundoff);
    // A computed subspace is rotated away from the exact one by at most
    // `rotation_radius` (see `embedded_penalty_null_basis`). An exact coefficient
    // β* in the exact subspace therefore lies within `rotation_radius·‖β*‖₂` of the
    // computed span, and the weighted least-squares residual on that span is no
    // larger than the residual at that nearest point:
    //   ‖W^{1/2} r‖₂ ≤ ‖W^{1/2} X‖_F · rotation_radius · ‖β*‖₂,
    // evaluated at the computed coefficient. A row of positive weight w carries at
    // most that norm over √w, on top of the rounding of its own dot product.
    let rotation_residual = if rotation_radius > 0.0 {
        let weighted_frobenius = x
            .rows()
            .into_iter()
            .zip(weights.iter())
            .map(|(row, &weight)| weight * row.iter().map(|value| value * value).sum::<f64>())
            .sum::<f64>()
            .sqrt();
        let beta_norm = beta.iter().map(|value| value * value).sum::<f64>().sqrt();
        weighted_frobenius * rotation_radius * beta_norm
    } else {
        0.0
    };
    for row in 0..x.nrows() {
        if weights[row] == 0.0 {
            continue;
        }
        let operand_scale = adjusted_response[row].abs()
            + x.row(row)
                .iter()
                .zip(beta.iter())
                .map(|(&value, &coefficient)| (value * coefficient).abs())
                .sum::<f64>();
        let residual = (adjusted_response[row] - fitted[row]).abs();
        let allowed = gamma * operand_scale + rotation_residual / weights[row].sqrt();
        if !(residual.is_finite() && residual <= allowed) {
            return None;
        }
    }
    // A zero-residual coefficient defines a deterministic Gaussian law only
    // when it is unique on the positive-weight support. A certified solve
    // residual alone cannot establish that: when n < p an underdetermined
    // design can interpolate arbitrary responses while still admitting
    // infinitely many coefficient vectors. Certify the injectivity promised by
    // `exact_gaussian_boundary` directly on the reduced design's positive-weight
    // support. Positive row scaling cannot change exact rank, so omitting it
    // here also makes the structural certificate invariant to a uniform
    // rescaling of all positive likelihood weights. Every certificate here is
    // a conjunct, so the rank-revealing QR runs last: a noisy response is
    // refused by the residual bound above without paying for it.
    if reduced_p > 0 {
        let positive_rows: Vec<usize> = weights
            .iter()
            .enumerate()
            .filter_map(|(row, &weight)| (weight > 0.0).then_some(row))
            .collect();
        if positive_rows.len() < reduced_p {
            return None;
        }
        // Every row already on the support needs no copy of the design.
        let positive_weight_reduced_x = if positive_rows.len() == reduced_x.nrows() {
            std::borrow::Cow::Borrowed(reduced_x)
        } else {
            std::borrow::Cow::Owned(Array2::from_shape_fn(
                (positive_rows.len(), reduced_p),
                |(weighted_row, column)| {
                    let row = positive_rows[weighted_row];
                    reduced_x[[row, column]]
                },
            ))
        };
        let rank = gam_linalg::faer_ndarray::rrqr_with_permutation(&*positive_weight_reduced_x)
            .ok()?
            .rank;
        if rank != reduced_p {
            return None;
        }
    }
    Some(beta)
}

#[cfg(test)]
mod exact_gaussian_boundary_tests {
    use super::exact_gaussian_coefficients;
    use ndarray::array;

    #[test]
    fn underdetermined_exact_interpolator_is_not_a_deterministic_boundary() {
        let x = array![[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        let response = array![1.0, 2.0];
        let weights = array![1.0, 1.0];

        assert!(
            exact_gaussian_coefficients(&x, &response, &weights, None).is_none(),
            "an n < p interpolator does not identify a unique deterministic Gaussian law"
        );
    }

    #[test]
    fn full_rank_exact_interpolator_remains_a_deterministic_boundary() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let response = array![1.25, -0.75, 0.5];
        let weights = array![1.0, 2.0, 0.5];

        let beta = exact_gaussian_coefficients(&x, &response, &weights, None)
            .expect("a full-rank exact design has a unique deterministic coefficient");
        assert_eq!(beta.len(), 2);
        assert!((beta[0] - 1.25).abs() <= 16.0 * f64::EPSILON);
        assert!((beta[1] + 0.75).abs() <= 16.0 * f64::EPSILON);
    }
}

#[cfg(test)]
mod exact_gaussian_boundary_design_reuse_tests {
    use super::*;
    use csv::StringRecord;
    use gam_data::encode_recordswith_inferred_schema;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn noisy_surface() -> Dataset {
        let mut rng = StdRng::seed_from_u64(3261);
        let headers: Vec<String> = ["x0", "x1", "y"].iter().map(|h| h.to_string()).collect();
        let rows = (0..240)
            .map(|_| {
                let x0: f64 = rng.random();
                let x1: f64 = rng.random();
                let noise: f64 = rng.random::<f64>() - 0.5;
                let y = (3.0 * x0).sin() * (2.0 * x1).cos() + 0.2 * noise;
                StringRecord::from(vec![x0.to_string(), x1.to_string(), y.to_string()])
            })
            .collect();
        encode_recordswith_inferred_schema(headers, rows).expect("encode")
    }

    fn standard_request<'a>(data: &'a Dataset) -> StandardFitRequest<'a> {
        let config = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        match materialize("y ~ te(x0, x1)", data, &config)
            .expect("materialize")
            .request
        {
            FitRequest::Standard(request) => request,
            _ => panic!("a Gaussian te() formula materializes a standard request"),
        }
    }

    /// The boundary certificate refuses a noisy response after realizing the
    /// full design; the REML fit must start from that design, and doing so
    /// must be the same fit as realizing it again.
    #[test]
    fn refused_boundary_hands_the_fit_its_realized_design() {
        let data = noisy_surface();
        let request = standard_request(&data);
        let GaussianStandardRoute::Iterative(Some(design)) =
            try_deterministic_gaussian_standard_fit(&request).expect("boundary check")
        else {
            panic!("a noisy te() response is refused with its realized design");
        };
        let rebuilt = realize_standard_design(&request).expect("rebuild");
        assert_eq!(design.design.to_dense(), rebuilt.design.to_dense());
        assert_eq!(design.affine_offset, rebuilt.affine_offset);
        assert_eq!(design.penalties.len(), rebuilt.penalties.len());
        for (handed, fresh) in design.penalties.iter().zip(&rebuilt.penalties) {
            assert_eq!(handed.col_range, fresh.col_range);
            assert_eq!(handed.local, fresh.local);
        }

        let on_design = fit_standard_model_on_design(request, Some(design)).expect("reused fit");
        let fresh = fit_standard_model(standard_request(&data)).expect("fresh fit");
        assert_eq!(on_design.fit.log_lambdas, fresh.fit.log_lambdas);
        assert_eq!(on_design.fit.beta, fresh.fit.beta);
        assert_eq!(
            on_design.design.design.to_dense(),
            fresh.design.design.to_dense()
        );
    }
}

#[cfg(test)]
mod square_exact_gaussian_design_tests {
    use super::*;
    use csv::StringRecord;
    use gam_data::encode_recordswith_inferred_schema;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const N: usize = 8;

    fn noisy_curve() -> Dataset {
        let mut rng = StdRng::seed_from_u64(4127);
        let headers: Vec<String> = ["x", "y"].iter().map(|h| h.to_string()).collect();
        let rows = (0..N)
            .map(|i| {
                let x = i as f64 / (N - 1) as f64;
                let noise: f64 = rng.random::<f64>() - 0.5;
                let y = (2.0 * std::f64::consts::PI * x).sin() + 0.6 * noise;
                StringRecord::from(vec![x.to_string(), y.to_string()])
            })
            .collect();
        encode_recordswith_inferred_schema(headers, rows).expect("encode")
    }

    fn standard_request<'a>(data: &'a Dataset) -> StandardFitRequest<'a> {
        let config = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        match materialize("y ~ s(x, bs='cr', k=8)", data, &config)
            .expect("materialize")
            .request
        {
            FitRequest::Standard(request) => request,
            _ => panic!("a Gaussian s() formula materializes a standard request"),
        }
    }

    /// With as many coefficients as rows, the design interpolates any
    /// response, so the exact fit carries no evidence of a zero residual
    /// variance. The shortcut must decline it instead of reporting phi = 0.
    #[test]
    fn square_design_interpolation_is_not_a_deterministic_fit() {
        let data = noisy_curve();
        let request = standard_request(&data);
        assert!(
            matches!(
                exact_gaussian_boundary(&request).expect("boundary check"),
                ExactGaussianVerdict::Boundary(_)
            ),
            "precondition: a square full-rank design certifies an exact boundary"
        );
        match try_deterministic_gaussian_standard_fit(&request).expect("route") {
            GaussianStandardRoute::Iterative(Some(design)) => {
                assert_eq!(design.design.ncols(), N, "the design is square");
            }
            GaussianStandardRoute::Iterative(None) => {
                panic!("the declined route must hand over its realized design")
            }
            GaussianStandardRoute::Exact(_) => {
                panic!("square-design interpolation was certified as an exact fit")
            }
        }
    }
}

#[cfg(test)]
mod constant_response_intercept_tests {
    use super::*;
    use csv::StringRecord;
    use gam_data::encode_recordswith_inferred_schema;

    fn constant_response() -> Dataset {
        let headers: Vec<String> = ["x", "y"].iter().map(|h| h.to_string()).collect();
        let rows = (0..40)
            .map(|i| {
                let x = 0.1 + 0.05 * i as f64;
                StringRecord::from(vec![x.to_string(), "5".to_string()])
            })
            .collect();
        encode_recordswith_inferred_schema(headers, rows).expect("encode")
    }

    fn standard_request<'a>(formula: &str, data: &'a Dataset) -> StandardFitRequest<'a> {
        let config = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        match materialize(formula, data, &config)
            .expect("materialize")
            .request
        {
            FitRequest::Standard(request) => request,
            _ => panic!("a Gaussian linear formula materializes a standard request"),
        }
    }

    /// The constant-response shortcut places `y - offset` on the global
    /// intercept column. With the intercept removed (`0 + x`) there is no
    /// such column, so the shortcut would publish β = 0 — fitted η = 0 while
    /// y = 5 — as an exact fit with φ̂ = 0 and zero covariance. The shortcut
    /// must decline, and the exact-boundary certificate must refuse a design
    /// that cannot reproduce the response.
    #[test]
    fn no_intercept_constant_response_is_not_an_exact_fit() {
        let data = constant_response();
        let with_intercept = standard_request("y ~ x", &data);
        assert!(
            gaussian_response_is_constant(&with_intercept),
            "a constant response under an intercept is the exact intercept-only fit"
        );

        let request = standard_request("y ~ 0 + x", &data);
        assert!(
            !gaussian_response_is_constant(&request),
            "no intercept column can carry the constant"
        );
        let route = try_deterministic_gaussian_standard_fit(&request).expect("route");
        assert!(
            matches!(route, GaussianStandardRoute::Iterative(_)),
            "y = 5 is not in the span of 0 + x, so the fit is not exact"
        );
    }
}

/// Certify that a Gaussian design represents its adjusted response exactly and
/// identify the asymptotic face of every smoothing precision.
///
/// A profiled Gaussian likelihood has no finite-density interior optimum when
/// `y - offset = X beta` exactly: its residual variance is zero and the correct
/// fitted law is the same deterministic boundary used by the constant-response
/// route. For a penalized design the boundary can be mixed: a penalty whose
/// null space still represents the response goes to infinite precision, while
/// a penalty acting on the unique exact coefficient goes to zero. A default
/// double-penalty `s(x)` on an exact line is the canonical case: roughness goes
/// to infinity and null-space shrinkage goes to zero.
///
/// The full and every restricted normal equation must have a unique certified
/// solution, and each final row must pass the `gamma_(p+1)` dot-product bound.
/// Thus a merely small residual cannot enter this route.
fn exact_gaussian_boundary(
    request: &StandardFitRequest<'_>,
) -> Result<ExactGaussianVerdict, WorkflowError> {
    if !request.family.is_gaussian_identity()
        || request.y.is_empty()
        || !request.spec.random_effect_terms.is_empty()
        || request.options.linear_constraints.is_some()
        || request.wiggle.is_some()
        || request.latent_coord.is_some()
        || !request.penalty_block_gamma_priors.is_empty()
        || request.spec.linear_terms.iter().any(|term| {
            !matches!(
                &term.coefficient_geometry,
                gam_terms::smooth::LinearCoefficientGeometry::Unconstrained
            ) || term.coefficient_min.is_some()
                || term.coefficient_max.is_some()
        })
        || request.y.len() != request.offset.len()
        || request.y.len() != request.weights.len()
    {
        return Ok(ExactGaussianVerdict::Ineligible);
    }
    let design = realize_standard_design(request).map_err(|err| WorkflowError::InvalidConfig {
        reason: format!("deterministic Gaussian candidate could not build its design: {err}"),
    })?;
    if design.design.ncols() == 0
        || design.coefficient_lower_bounds.is_some()
        || design.linear_constraints.is_some()
    {
        return Ok(ExactGaussianVerdict::Interior(design));
    }
    let adjusted_response = request.y.as_ref() - request.offset.as_ref() - &design.affine_offset;
    if adjusted_response.iter().any(|value| !value.is_finite())
        || request
            .weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Ok(ExactGaussianVerdict::Interior(design));
    }
    // The certificate factors the dense design, so it is admitted through the
    // fit's own resource policy. A refusal leaves the question undecided; it
    // is not evidence either way, and the iterative solver owns the fit.
    let x = match design.design.try_to_dense_arc_with_policy(
        "deterministic Gaussian boundary",
        &request.options.resource_policy,
    ) {
        Ok(x) => x,
        Err(_) => return Ok(ExactGaussianVerdict::Undecided(design)),
    };
    let x: &Array2<f64> = &x;
    let Some(beta) =
        exact_gaussian_coefficients(x, &adjusted_response, request.weights.as_ref(), None)
    else {
        return Ok(ExactGaussianVerdict::Interior(design));
    };

    let p = x.ncols();
    let mut penalty_faces = vec![DeterministicPenaltyFace::Zero; design.penalties.len()];
    let mut infinite_face_penalty = Array2::<f64>::zeros((p, p));
    let mut restricted_certification = false;
    for (penalty_index, block) in design.penalties.iter().enumerate() {
        let r = block.col_range.clone();
        if r.is_empty()
            || r.end > p
            || block.local.nrows() != r.len()
            || block.local.ncols() != r.len()
            || block.local.iter().any(|value| !value.is_finite())
        {
            return Err(raised_fit_failure(
                FailureCategory::Invariant,
                format!(
                    "deterministic Gaussian candidate received malformed penalty \
                     {penalty_index}: range={r:?}, local={}x{}, design width={p}",
                    block.local.nrows(),
                    block.local.ncols(),
                ),
            ));
        }
        // Classify every face against the ONE exact coefficient certified by
        // the full design. Asking whether each penalty null space can reproduce
        // the response independently is not closed under intersection when X
        // has aliased coefficient representations: two different restricted
        // coefficients can each fit y even though no coefficient satisfies
        // both restrictions. The full-design coefficient is the common witness
        // every infinite face must annihilate.
        let beta_local = beta.slice(ndarray::s![r.clone()]);
        let penalty_beta = block.local.dot(&beta_local);
        let roundoff = (r.len().max(1) as f64) * f64::EPSILON;
        let gamma = roundoff / (1.0 - roundoff);
        let annihilates = penalty_beta.iter().enumerate().all(|(row, &value)| {
            let operand_scale = block
                .local
                .row(row)
                .iter()
                .zip(beta_local.iter())
                .map(|(&penalty, &coefficient)| (penalty * coefficient).abs())
                .sum::<f64>();
            value.is_finite() && value.abs() <= gamma * operand_scale
        });
        // The dot-product bound above prices only the rounding of `S_k β`, not
        // the error β carries from its normal-equation solve. On a square,
        // ill-conditioned design (`n = p`, #2355) that solve error alone can
        // exceed the bound, so an exact line reports a roughness penalty that
        // does not annihilate it and the fit interpolates at EDF = p. Uniqueness
        // of β is certified above (full column rank on the positive-weight
        // support), so the response is reproduced on `null(S_k)` if and only if
        // β itself lies there: a certified restricted solve is then an exact
        // annihilation witness, and aliasing cannot make it disagree with β.
        // The computed null basis is itself rotated by the eigensolver's rounding,
        // amplified by the penalty's spectral gap (#2355: on an exact line at
        // n = p the rotated span alone leaves a residual above the dot-product
        // bound), so the restricted solve prices that rotation.
        let reproduced_on_null_space = !annihilates && {
            let (null_basis, rotation_radius) =
                embedded_penalty_null_basis(p, r.clone(), &block.local).map_err(|reason| {
                    raised_fit_failure(
                        FailureCategory::Numerical,
                        format!(
                            "deterministic Gaussian candidate could not resolve penalty \
                             {penalty_index}'s null space: {reason}"
                        ),
                    )
                })?;
            exact_gaussian_coefficients(
                x,
                &adjusted_response,
                request.weights.as_ref(),
                Some((&null_basis, rotation_radius)),
            )
            .is_some()
        };
        restricted_certification |= reproduced_on_null_space;
        if annihilates || reproduced_on_null_space {
            penalty_faces[penalty_index] = DeterministicPenaltyFace::Infinite;
            infinite_face_penalty
                .slice_mut(ndarray::s![r.clone(), r])
                .scaled_add(1.0, &block.local);
        }
    }
    // A face certified only by its restricted solve leaves β carrying the
    // full-design solve error in that face's range, while the boundary geometry
    // treats β as lying in every infinite-face null space (its penalty
    // quadratic is exactly zero). Re-solve on the joint tangent space
    // `null(S_infinite)`: uniqueness makes it the same coefficient in exact
    // arithmetic, and a joint solve that does not certify declines the route.
    let beta = if restricted_certification {
        let (joint_null_basis, joint_rotation_radius) =
            embedded_penalty_null_basis(p, 0..p, &infinite_face_penalty).map_err(|reason| {
                raised_fit_failure(
                    FailureCategory::Numerical,
                    format!(
                        "deterministic Gaussian candidate could not resolve the joint \
                         infinite-face null space: {reason}"
                    ),
                )
            })?;
        let Some(tangent_beta) = exact_gaussian_coefficients(
            x,
            &adjusted_response,
            request.weights.as_ref(),
            Some((&joint_null_basis, joint_rotation_radius)),
        ) else {
            return Ok(ExactGaussianVerdict::Interior(design));
        };
        tangent_beta
    } else {
        beta
    };

    Ok(ExactGaussianVerdict::Boundary(ExactGaussianBoundary {
        design,
        beta,
        penalty_faces,
    }))
}

/// Columns spanning `null(S)` for one block-local PSD penalty embedded in the
/// full `p`-coefficient frame. Every coordinate outside `range` is free; the
/// block's own null directions come from its spectrum at the same relative
/// rank floor `deterministic_gaussian_standard_fit` uses for an infinite face.
///
/// Also returns the radius by which the computed span can be rotated away from
/// the exact null space, `+∞` when the spectrum does not separate the two.
fn embedded_penalty_null_basis(
    p: usize,
    range: std::ops::Range<usize>,
    local: &Array2<f64>,
) -> Result<(Array2<f64>, f64), String> {
    use gam_linalg::faer_ndarray::FaerEigh;
    let symmetric = (local + &local.t().to_owned()) * 0.5;
    let (eigenvalues, eigenvectors) = symmetric
        .eigh(faer::Side::Lower)
        .map_err(|error| format!("penalty spectrum: {error}"))?;
    let largest = eigenvalues
        .iter()
        .fold(0.0_f64, |largest, &value| largest.max(value.abs()));
    let rank_floor = f64::EPSILON * (range.len().max(1) as f64) * largest;
    let null_directions: Vec<usize> = eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (value <= rank_floor).then_some(index))
        .collect();
    // The eigensolver's backward error sits at the rank floor the null directions
    // are classified at, and by Weyl it moves the weakest penalized eigenvalue by
    // at most that floor. Davis–Kahan then bounds the rotation of the computed
    // null directions away from the exact null space by
    //   floor / (weakest penalized − 2·floor);
    // a gap that does not clear twice the floor leaves the rotation unbounded.
    let weakest_penalized = eigenvalues
        .iter()
        .copied()
        .filter(|&value| value > rank_floor)
        .fold(f64::INFINITY, f64::min);
    let rotation_radius = if weakest_penalized.is_infinite() {
        0.0
    } else if weakest_penalized > 2.0 * rank_floor {
        rank_floor / (weakest_penalized - 2.0 * rank_floor)
    } else {
        f64::INFINITY
    };
    let outside = p - range.len();
    let mut basis = Array2::<f64>::zeros((p, outside + null_directions.len()));
    let mut column = 0usize;
    for coordinate in (0..range.start).chain(range.end..p) {
        basis[[coordinate, column]] = 1.0;
        column += 1;
    }
    for &direction in &null_directions {
        for (offset, coordinate) in range.clone().enumerate() {
            basis[[coordinate, column]] = eigenvectors[[offset, direction]];
        }
        column += 1;
    }
    Ok((basis, rotation_radius))
}

fn try_deterministic_gaussian_standard_fit(
    request: &StandardFitRequest<'_>,
) -> Result<GaussianStandardRoute, WorkflowError> {
    if gaussian_response_is_constant(request) {
        return deterministic_gaussian_standard_fit(request, None);
    }
    match exact_gaussian_boundary(request)? {
        ExactGaussianVerdict::Ineligible => Ok(GaussianStandardRoute::Iterative(None)),
        ExactGaussianVerdict::Interior(design) | ExactGaussianVerdict::Undecided(design) => {
            Ok(GaussianStandardRoute::Iterative(Some(design)))
        }
        ExactGaussianVerdict::Boundary(boundary) => {
            deterministic_gaussian_standard_fit(request, Some(boundary))
        }
    }
}

/// The training table with every zero-weight row removed.
///
/// A prior weight of zero removes the row from the likelihood, and it must
/// remove it from everything else the fit derives from the rows as well —
/// knots, covariate ranges, identifiability constraints, standardization,
/// factor levels, column kinds — so that weight zero is exactly row deletion.
/// Every fitting entry point runs its data through this one seam. The table
/// is borrowed unchanged when no weight column is configured or no weight is
/// exactly zero; rows with a missing or negative weight are kept so the weight
/// validator still reports them.
pub fn drop_zero_weight_rows<'a>(
    data: &'a Dataset,
    config: &FitConfig,
) -> Result<std::borrow::Cow<'a, Dataset>, WorkflowError> {
    use std::borrow::Cow;
    let Some(name) = config.weight_column.as_deref().map(str::trim) else {
        return Ok(Cow::Borrowed(data));
    };
    let Some(column) = data.headers.iter().position(|header| header == name) else {
        return Ok(Cow::Borrowed(data));
    };
    let weights = data.values.column(column);
    let keep: Vec<usize> = (0..weights.len())
        .filter(|&row| weights[row] != 0.0)
        .collect();
    if keep.len() == weights.len() {
        return Ok(Cow::Borrowed(data));
    }
    if keep.is_empty() {
        return Err(no_positive_weight_error(name, weights.len()));
    }
    data.select_rows(&keep)
        .map(Cow::Owned)
        .map_err(|error| WorkflowError::InvalidConfig {
            reason: error.to_string(),
        })
}

pub fn fit_from_formula(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<FitResult, WorkflowError> {
    fit_from_formula_with_notes(formula, data, config).map(|outcome| outcome.result)
}

/// A fitted formula result together with advisories emitted by its one
/// authoritative materialization pass.
pub struct FormulaFitResult {
    pub result: FitResult,
    pub inference_notes: FitNotes,
    /// Scalar terms the training rows could not identify, removed before the fit.
    pub unidentified_scalar_terms: Vec<UnidentifiedScalarTerm>,
}

/// Resolve, materialize, and fit a formula without making front ends repeat any
/// model construction. Unlike `fit_from_formula`, this service also returns the
/// materializer's user-facing advisories for CLI/Python presentation.
///
/// An automatic `.` term is expanded against `data` first; its notes (the
/// first of which spells out the fitted formula) lead the returned notes.
pub fn fit_from_formula_with_notes(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<FormulaFitResult, WorkflowError> {
    gam_runtime::parallel::install(|| fit_from_formula_with_notes_on_pool(formula, data, config))
}

fn fit_from_formula_with_notes_on_pool(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<FormulaFitResult, WorkflowError> {
    let data = &*drop_zero_weight_rows(data, config)?;
    let automatic = expand_automatic_fit_formula(formula, data, config)?;
    if automatic.notes.is_empty() {
        return fit_expanded_formula_with_notes(formula, data, config);
    }
    let mut outcome = fit_expanded_formula_with_notes(&automatic.formula, data, config)?;
    // The expansion is an advisory: the fitted formula is not the literal one.
    let mut advisories = automatic.notes;
    advisories.append(&mut outcome.inference_notes.advisories);
    outcome.inference_notes.advisories = advisories;
    Ok(outcome)
}

fn fit_expanded_formula_with_notes(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<FormulaFitResult, WorkflowError> {
    if config.ctn_stage1.is_some() || config.frozen_ctn.is_some() {
        let payload = crate::inference::model_payload_builders::fit_formula_to_payload(
            formula.to_string(),
            data,
            config,
        )?;
        return Ok(FormulaFitResult {
            inference_notes: FitNotes {
                advisories: payload.inference_notes.clone(),
                informational: payload.informational_notes.clone(),
            },
            unidentified_scalar_terms: payload.unidentified_scalar_terms.clone(),
            result: FitResult::Ctn(Box::new(payload)),
        });
    }
    fit_formula_through_adaptive_resolution(formula, data, config)
}

/// Resolve `config`, fit `formula` at the adaptive structural start, and
/// continue through the saturation-driven resolution loop. This is the one
/// owner of that loop for a formula fit, so the library and the payload
/// service reach the same fitted basis for the same request (the expectile
/// payload route used to fit the fully provisioned basis with no loop, #4062).
pub(crate) fn fit_formula_through_adaptive_resolution(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<FormulaFitResult, WorkflowError> {
    let mut config = config
        .clone()
        .resolve()
        .map_err(|reason| WorkflowError::InvalidConfig { reason })?;
    // Only this entry point owns the fit→measure→expand loop. Raw public
    // `materialize()` callers receive the ordinary fully provisioned basis;
    // activating the structural start without an owner would strand them in an
    // under-resolved function space.
    config.adaptive_resolution = Some(Vec::new());
    let current = fit_from_formula_once_with_notes(formula, data, &config)?;
    finish_adaptive_spatial_fit(formula, data, config, current)
}

/// Fit an already-materialized standard request, then continue through the
/// canonical saturation-driven spatial-resolution loop.
///
/// Front ends that must inspect the request variant for payload dispatch use
/// this seam so the dispatch materialization is also the first estimator
/// materialization. Re-entering [`fit_from_formula_with_notes`] after matching a
/// `Standard` request would build and discard one complete spatial basis before
/// the real fit (#1689), duplicating construction work and peak memory on the
/// Python path.
pub(crate) fn fit_materialized_standard_with_notes(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
    request: StandardFitRequest<'_>,
    inference_notes: FitNotes,
) -> Result<FormulaFitResult, WorkflowError> {
    let mut config = config
        .clone()
        .resolve()
        .map_err(|reason| WorkflowError::InvalidConfig { reason })?;
    config.adaptive_resolution = Some(Vec::new());
    let current = fit_materialized_once_with_notes(MaterializedModel {
        request: FitRequest::Standard(request),
        inference_notes,
        unidentified_scalar_terms: Vec::new(),
        survival_time_basis: None,
    })?;
    finish_adaptive_spatial_fit(formula, data, config, current)
}

/// Grow every formula-default smooth basis while the converged fits' own
/// evidence certifies that the larger basis is better (#1689, #3078, #3331).
///
/// Every open adaptive term proposes the next level of its NESTED refinement
/// chain ([`gam_terms::smooth::refined_adaptive_resolution`]). There is no
/// trigger threshold: neither an EDF-saturation margin nor a lack-of-fit
/// statistic decides whether to try. The refit's REML/LAML evidence decides
/// whether to keep it, and only through a comparison that is valid:
///
/// * the two bases are nested with the same unpenalized null space and the
///   same penalty order (checked on the realized, frozen knots and centers,
///   not on the request), so the evidence difference is a Bayes factor between
///   two smoothing priors on nested spaces rather than a difference of two
///   unrelated improper-prior normalizers;
/// * the difference exceeds the error both fits certify for their own
///   criterion value ([`certified_evidence_gain`]); a difference inside that
///   error is not evidence and the smaller basis stands;
/// * a refit that does not CONVERGE certifies nothing either, so it closes the
///   attempt on that same rule and the certified incumbent stands (#4529). It
///   is the absence of evidence, not a verdict on the model the caller asked
///   for: this loop proposes every level unconditionally, so a refusal there is
///   the outcome of an experiment the engine chose to run.
///
/// Every refinement proposes the whole chain level at once, jointly for the
/// terms that fit the residual rank together, then alone in formula order. A
/// term whose single refinement is not certified better, whose refit did not
/// converge, or whose realized refinement is not nested, is closed and never
/// proposed again, so the loop terminates after at most one rejected refit per
/// term plus one accepted refit per level grown.
fn finish_adaptive_spatial_fit(
    formula: &str,
    data: &Dataset,
    mut config: FitConfig,
    mut current: FormulaFitResult,
) -> Result<FormulaFitResult, WorkflowError> {
    let mut open: Option<Vec<bool>> = None;
    let mut declined: Vec<String> = Vec::new();
    loop {
        let Some(current_standard) = standard_result(&current) else {
            return Ok(with_declined_refinements(current, &mut declined));
        };
        let open = open.get_or_insert_with(|| {
            current_standard
                .adaptive_bases
                .iter()
                .map(|basis| {
                    basis
                        .as_ref()
                        .is_some_and(gam_terms::smooth::adaptive_refinement_can_nest)
                })
                .collect()
        });
        let refinements = adaptive_refinements(current_standard, data, open)?;
        if refinements.is_empty() {
            return Ok(with_declined_refinements(current, &mut declined));
        }
        // Without a certified error bar on the current criterion there is no
        // honest comparison to make, so the basis stays where it is.
        let Some(current_evidence) = certified_evidence(current_standard)? else {
            return Ok(with_declined_refinements(current, &mut declined));
        };
        let term_count = current_standard.adaptive_bases.len();
        let spare_rank = spare_design_rank(current_standard, data);

        let mut attempts: Vec<Vec<&AdaptiveRefinement>> = Vec::new();
        let joint = joint_refinement_prefix(&refinements, spare_rank);
        if joint.len() > 1 {
            attempts.push(joint);
        }
        attempts.extend(refinements.iter().map(|refinement| vec![refinement]));

        let mut accepted = None;
        for attempt in attempts {
            if attempt
                .iter()
                .any(|refinement| !open[refinement.term_index])
            {
                continue;
            }
            let candidate_config = config_with_refinements(&config, term_count, &attempt);
            // A refit that does not converge certifies nothing, which is the
            // same state as one that converges and certifies no criterion gain:
            // the absence of evidence that the larger basis is better. It used
            // to be the one outcome that discarded the incumbent — a converged,
            // certified fit was thrown away because a refinement this loop
            // proposes UNCONDITIONALLY refused, and the refusal reported the
            // caller's own model as under-resolved although nothing had
            // measured it so (#4529). It now closes the attempt on the same
            // rule as a non-improving refit, records why, and leaves the
            // incumbent standing. Only a refit that breaks the loop's own
            // invariant — a different estimator representation for the same
            // formula — is still an error.
            let candidate = match fit_from_formula_once_with_notes(formula, data, &candidate_config)
            {
                Ok(candidate) => candidate,
                Err(error) => {
                    declined.push(refinement_declined_note(&attempt, &error.to_string()));
                    if let [single] = attempt.as_slice() {
                        open[single.term_index] = false;
                    }
                    continue;
                }
            };
            let Some(candidate_standard) = standard_result(&candidate) else {
                return Err(refinement_failure(
                    &attempt,
                    "the refinement refit changed estimator representation".to_string(),
                    None,
                ));
            };
            // Evidence is compared only between truly nested bases. A realized
            // refinement that is not nested (farthest-point tie orbits capped
            // differently, a knot that moved) is never compared, and its terms
            // are closed: there is no valid evidence statement about them.
            let not_nested =
                non_nested_refinements(current_standard, candidate_standard, &attempt)?;
            if !not_nested.is_empty() {
                for term_index in not_nested {
                    open[term_index] = false;
                }
                continue;
            }
            let improves = match certified_evidence(candidate_standard)? {
                Some(candidate_evidence) => {
                    certified_evidence_gain(current_evidence, candidate_evidence)
                }
                None => false,
            };
            if improves {
                accepted = Some((candidate_config, candidate));
                break;
            }
            if let [single] = attempt.as_slice() {
                open[single.term_index] = false;
            }
        }
        match accepted {
            Some((candidate_config, candidate)) => {
                config = candidate_config;
                current = candidate;
            }
            None => return Ok(with_declined_refinements(current, &mut declined)),
        }
    }
}

/// Carry the declined-refinement advisories out on the fit the loop returns.
///
/// The loop holds `current` borrowed while it reasons about the refinements, so
/// the notes are accumulated beside it and attached at the one point the fit
/// leaves the loop. They are advisories rather than informational notes because
/// the basis the caller receives is smaller than the one the engine would have
/// grown, which is the fitted model differing from the request's intent.
fn with_declined_refinements(
    mut current: FormulaFitResult,
    declined: &mut Vec<String>,
) -> FormulaFitResult {
    current.inference_notes.advisories.append(declined);
    current
}

/// A fit's comparable REML/LAML criterion (lower is better) together with the
/// error its own outer certificate bounds it by.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CertifiedEvidence {
    score: f64,
    error: gam_solve::rho_optimizer::CriterionErrorBound,
}

/// The certified evidence of a converged standard fit, or `None` when the fit
/// has no finite criterion or its outer certificate did not certify a
/// decrement bound (#3331). The Tierney-Kadane null-space normalizer added by
/// [`standard_fit_comparable_reml_score`] is a closed-form determinant of the
/// realized penalties and carries no optimization error of its own.
fn certified_evidence(
    result: &StandardFitResult,
) -> Result<Option<CertifiedEvidence>, WorkflowError> {
    let Some(error) = result
        .fit
        .artifacts
        .criterion_certificate
        .as_ref()
        .and_then(|certificate| certificate.criterion_error)
    else {
        return Ok(None);
    };
    let score = standard_fit_comparable_reml_score(result)
        .map_err(|reason| raised_fit_failure(FailureCategory::Invariant, reason))?;
    Ok(score
        .filter(|score| score.is_finite())
        .filter(|_| error.decrease_left.is_finite() && error.value_band.is_finite())
        .map(|score| CertifiedEvidence { score, error }))
}

/// Whether `candidate`'s criterion minimum is certified lower than
/// `current`'s.
///
/// Each fit returned the criterion value `S` at its converged `ρ̂`, evaluated
/// to within `value_band`, and certifies by its Newton decrement that the true
/// minimum `M` lies at most `decrease_left` below it:
/// `M ∈ [S − value_band − decrease_left, S + value_band]`. The candidate is
/// better only when its whole interval lies below the current one's, so a
/// difference inside the certified evaluation error is never read as evidence.
///
/// The comparison is meaningful only between nested bases with the same null
/// space and penalty order; callers establish that with
/// [`non_nested_refinements`] before asking.
fn certified_evidence_gain(current: CertifiedEvidence, candidate: CertifiedEvidence) -> bool {
    let candidate_upper = candidate.score + candidate.error.value_band;
    let current_lower = current.score - current.error.above_minimum();
    candidate_upper < current_lower
}

/// The terms of `attempt` whose realized refinement in `candidate` does not
/// nest the basis `current` realized: the frozen knots or centers of the
/// current fit must all reappear in the candidate's, with the same degree,
/// penalty order, penalty count and unpenalized null-space dimension.
fn non_nested_refinements(
    current: &StandardFitResult,
    candidate: &StandardFitResult,
    attempt: &[&AdaptiveRefinement],
) -> Result<Vec<usize>, WorkflowError> {
    let freeze = |result: &StandardFitResult| {
        gam_terms::smooth::freeze_term_collection_from_design(&result.resolvedspec, &result.design)
            .map_err(|error| {
                raised_fit_failure(
                    FailureCategory::Invariant,
                    format!("adaptive refinement could not freeze its realized basis: {error}"),
                )
            })
    };
    let penalty_count = |result: &StandardFitResult, term_index: usize| {
        result
            .design
            .smooth_term_penalty_range(term_index)
            .map(|range| range.map_or(0, |range| range.len()))
            .map_err(|reason| raised_fit_failure(FailureCategory::Invariant, reason))
    };
    let current_frozen = freeze(current)?;
    let candidate_frozen = freeze(candidate)?;
    let mut not_nested = Vec::new();
    for refinement in attempt {
        let t = refinement.term_index;
        let (Some(coarse), Some(fine), Some(coarse_term), Some(fine_term)) = (
            current_frozen.smooth_terms.get(t),
            candidate_frozen.smooth_terms.get(t),
            current.design.smooth.terms.get(t),
            candidate.design.smooth.terms.get(t),
        ) else {
            not_nested.push(t);
            continue;
        };
        let nests = gam_terms::smooth::realized_basis_nests(&coarse.basis, &fine.basis)
            && coarse_term.wald_unpenalized_dim() == fine_term.wald_unpenalized_dim()
            && penalty_count(current, t)? == penalty_count(candidate, t)?;
        if !nests {
            not_nested.push(t);
        }
    }
    Ok(not_nested)
}

/// The formula-order prefix of `refinements` whose added coefficients fit the
/// design's residual rank together.
fn joint_refinement_prefix(
    refinements: &[AdaptiveRefinement],
    spare_rank: usize,
) -> Vec<&AdaptiveRefinement> {
    let mut used = 0usize;
    let mut prefix = Vec::new();
    for refinement in refinements {
        match used.checked_add(refinement.added_width) {
            Some(total) if total <= spare_rank => {
                used = total;
                prefix.push(refinement);
            }
            _ => break,
        }
    }
    prefix
}

fn config_with_refinements(
    config: &FitConfig,
    term_count: usize,
    refinements: &[&AdaptiveRefinement],
) -> FitConfig {
    let mut candidate = config.clone();
    let plan = candidate.adaptive_resolution.get_or_insert_with(Vec::new);
    if plan.len() < term_count {
        plan.resize(term_count, None);
    }
    for refinement in refinements {
        plan[refinement.term_index] = Some(refinement.proposed.clone());
    }
    candidate
}

fn refinement_failure(
    refinements: &[&AdaptiveRefinement],
    reason: String,
    refit_failure: Option<WorkflowError>,
) -> WorkflowError {
    let join = |part: &dyn Fn(&AdaptiveRefinement) -> String| {
        refinements
            .iter()
            .map(|refinement| part(refinement))
            .collect::<Vec<_>>()
            .join(", ")
    };
    WorkflowError::SpatialUnderresolved {
        term: join(&|refinement| refinement.term_name.clone()),
        current_resolution: join(&|refinement| refinement.current.to_string()),
        attempted_resolution: join(&|refinement| refinement.proposed.to_string()),
        reason,
        refit_failure: refit_failure.map(Box::new),
    }
}

/// The advisory a declined refinement records: which terms were proposed, from
/// which resolution to which, and what the refit said.
///
/// The fit the caller receives is the certified incumbent, so this note is the
/// only place the refused experiment is visible; printing it is what keeps
/// "the engine stopped growing this basis" from being silent (#4529).
fn refinement_declined_note(refinements: &[&AdaptiveRefinement], reason: &str) -> String {
    let proposals = refinements
        .iter()
        .map(|refinement| {
            format!(
                "'{}' {} -> {}",
                refinement.term_name, refinement.current, refinement.proposed
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Adaptive resolution: kept the certified basis and closed the refinement {proposals} \
         because that refit did not converge, so it is no evidence that the larger basis is \
         better. The refit reported: {reason}"
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AdaptiveRefinement {
    term_index: usize,
    term_name: String,
    current: AdaptiveResolution,
    proposed: AdaptiveResolution,
    /// Raw coefficients the refinement adds to the design.
    added_width: usize,
}

/// The furthest point of `current`'s nested refinement chain one proposal
/// takes, or `None` when even one level is unidentifiable.
///
/// A level is taken only when it stays within the covariate `support` (a
/// clamped level would not be nested in the chain) and its added width fits
/// the `spare_rank` residual degrees of freedom. The chain stops after one
/// level unless the term's lack-of-fit alternative needs `directions`
/// coefficients: a harmonic basis is orthogonal across degrees, so a signal of
/// degree `L` is invisible to every span below `L` and a one-level step shows
/// no evidence gain. `directions` is the rank of the screen's enrichment
/// alternative, a structural dimension, not a significance threshold.
fn chain_proposal(
    current: &AdaptiveResolution,
    support: &AdaptiveResolution,
    directions: usize,
    spare_rank: usize,
    width: impl Fn(&AdaptiveResolution) -> usize,
) -> Option<(AdaptiveResolution, usize)> {
    let base = width(current);
    let mut point = current.clone();
    let mut proposal = None;
    loop {
        let next = gam_terms::smooth::refined_adaptive_resolution(&point);
        let added = width(&next).saturating_sub(base);
        if !next.exceeds(&point) || next.exceeds(support) || added > spare_rank {
            break;
        }
        point = next;
        proposal = Some((point.clone(), added));
        if added >= directions {
            break;
        }
    }
    proposal
}

fn standard_result(outcome: &FormulaFitResult) -> Option<&StandardFitResult> {
    match &outcome.result {
        FitResult::Standard(result) => Some(result),
        _ => None,
    }
}

/// Residual degrees of freedom the design leaves after one for the scale, so
/// every refit keeps `p < n` (a `p >= n` design is not identified by the data,
/// and the REML surface is flat along directions the data never see).
fn spare_design_rank(result: &StandardFitResult, data: &Dataset) -> usize {
    data.values
        .nrows()
        .saturating_sub(1)
        .saturating_sub(result.design.design.ncols())
}

/// The next nested chain level of every `open` adaptive term of the converged
/// `result`, in formula order. A term with no identifiable next level is
/// closed.
fn adaptive_refinements(
    result: &StandardFitResult,
    data: &Dataset,
    open: &mut [bool],
) -> Result<Vec<AdaptiveRefinement>, WorkflowError> {
    let term_count = result.resolvedspec.smooth_terms.len();
    if result.adaptive_bases.len() != term_count
        || result.design.smooth.terms.len() != term_count
        || open.len() != term_count
    {
        return Err(raised_fit_failure(
            FailureCategory::Invariant,
            format!(
                "adaptive resolution provenance mismatch: resolved terms={term_count}, \
                 adaptive bases={}, realized terms={}, open terms={}",
                result.adaptive_bases.len(),
                result.design.smooth.terms.len(),
                open.len(),
            ),
        ));
    }
    let values = data.values.view();
    let spare_rank = spare_design_rank(result, data);
    let mut refinements = Vec::new();
    for (term_index, basis) in result.adaptive_bases.iter().enumerate() {
        let Some(basis) = basis.as_ref().filter(|_| open[term_index]) else {
            continue;
        };
        let term_name = &result.resolvedspec.smooth_terms[term_index].name;
        let invariant = |what: &str| {
            raised_fit_failure(
                FailureCategory::Invariant,
                format!("adaptive smooth term '{term_name}' {what}"),
            )
        };
        let current = gam_terms::smooth::adaptive_resolution_of(basis)
            .ok_or_else(|| invariant("lost its adaptive provenance"))?;
        let support = gam_terms::smooth::adaptive_resolution_support(basis, values)
            .ok_or_else(|| invariant("references covariates the data does not carry"))?;
        let directions = match current {
            AdaptiveResolution::HarmonicDegree(_) => result
                .basis_adequacy
                .iter()
                .filter(|row| row.term_idx == term_index)
                .filter_map(|row| row.enrichment_rank)
                .max()
                .unwrap_or(0),
            _ => 0,
        };
        match chain_proposal(&current, &support, directions, spare_rank, |resolution| {
            gam_terms::smooth::adaptive_resolution_width(basis, values, resolution)
        }) {
            Some((proposed, added_width)) => refinements.push(AdaptiveRefinement {
                term_index,
                term_name: term_name.clone(),
                current,
                proposed,
                added_width,
            }),
            None => open[term_index] = false,
        }
    }
    Ok(refinements)
}

#[cfg(test)]
mod adaptive_spatial_resolution_tests {
    use super::{
        AdaptiveRefinement, AdaptiveResolution, CertifiedEvidence, certified_evidence_gain,
        chain_proposal, joint_refinement_prefix,
    };
    use gam_solve::rho_optimizer::CriterionErrorBound;

    fn width(resolution: &AdaptiveResolution) -> usize {
        match resolution {
            AdaptiveResolution::InternalKnots(k) => k + 4,
            AdaptiveResolution::Centers(c) | AdaptiveResolution::PeriodicBasis(c) => *c,
            AdaptiveResolution::HarmonicDegree(l) => l * (l + 2),
        }
    }

    fn refinement(term_index: usize, added_width: usize) -> AdaptiveRefinement {
        AdaptiveRefinement {
            term_index,
            term_name: format!("s{term_index}"),
            current: AdaptiveResolution::InternalKnots(8),
            proposed: AdaptiveResolution::InternalKnots(17),
            added_width,
        }
    }

    #[test]
    fn chain_takes_one_nested_level() {
        use AdaptiveResolution::InternalKnots;
        assert_eq!(
            chain_proposal(&InternalKnots(8), &InternalKnots(996), 0, 1000, width),
            Some((InternalKnots(17), 9))
        );
    }

    #[test]
    fn chain_level_beyond_support_or_rank_is_not_proposed() {
        use AdaptiveResolution::{Centers, InternalKnots};
        // Clamping 17 knots to a support of 12 would leave the chain and lose
        // nesting, so no level is proposed.
        assert_eq!(
            chain_proposal(&InternalKnots(8), &InternalKnots(12), 0, 1000, width),
            None
        );
        assert_eq!(
            chain_proposal(&InternalKnots(8), &InternalKnots(996), 0, 8, width),
            None
        );
        assert_eq!(
            chain_proposal(&InternalKnots(8), &InternalKnots(996), 0, 9, width),
            Some((InternalKnots(17), 9))
        );
        // Far beyond any fixed default: only the data's distinct rows bound it.
        assert_eq!(
            chain_proposal(&Centers(4096), &Centers(9000), 0, 100_000, width),
            Some((Centers(8192), 4096))
        );
    }

    #[test]
    fn harmonic_chain_spans_the_screen_alternative() {
        use AdaptiveResolution::HarmonicDegree;
        // Degree 3 spans 15 directions; the chain is 3 -> 5 -> 8. A 60-direction
        // alternative needs the second level.
        assert_eq!(
            chain_proposal(&HarmonicDegree(3), &HarmonicDegree(20), 60, 1000, width),
            Some((HarmonicDegree(8), 65))
        );
        // The residual rank stops the chain at the last level that fits.
        assert_eq!(
            chain_proposal(&HarmonicDegree(3), &HarmonicDegree(20), 60, 30, width),
            Some((HarmonicDegree(5), 20))
        );
    }

    #[test]
    fn joint_attempt_is_the_formula_order_prefix_that_fits() {
        let refinements = vec![refinement(0, 9), refinement(1, 9), refinement(2, 9)];
        let prefix = joint_refinement_prefix(&refinements, 20);
        assert_eq!(
            prefix.iter().map(|r| r.term_index).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(joint_refinement_prefix(&refinements, 8).is_empty());
    }

    fn evidence(score: f64, decrease_left: f64, value_band: f64) -> CertifiedEvidence {
        CertifiedEvidence {
            score,
            error: CriterionErrorBound {
                decrease_left,
                value_band,
            },
        }
    }

    #[test]
    fn growth_requires_a_gain_beyond_both_certified_errors() {
        let current = evidence(1000.0, 1.0e-6, 1.0e-8);
        // A clear gain is accepted.
        assert!(certified_evidence_gain(current, evidence(999.0, 1.0e-6, 1.0e-8)));
        // A gain smaller than what the current fit may still decrease is not.
        assert!(!certified_evidence_gain(current, evidence(1000.0 - 5.0e-7, 0.0, 0.0)));
        // A gain smaller than the candidate's own evaluation band is not.
        assert!(!certified_evidence_gain(current, evidence(1000.0 - 1.0e-5, 0.0, 2.0e-5)));
        // The candidate's unfinished decrease only helps it, so it is not
        // charged against the gain.
        assert!(certified_evidence_gain(current, evidence(1000.0 - 1.0e-5, 1.0, 1.0e-8)));
        // A loss is never a gain.
        assert!(!certified_evidence_gain(current, evidence(1000.5, 0.0, 0.0)));
    }
}

fn fit_from_formula_once_with_notes(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<FormulaFitResult, WorkflowError> {
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
    if let Some(outcome) = fit_expectile_if_requested(formula, data, &config)? {
        return Ok(FormulaFitResult {
            result: outcome.fit.into_fit_result(),
            inference_notes: outcome.materialized.inference_notes,
            unidentified_scalar_terms: outcome.materialized.unidentified_scalar_terms,
        });
    }
    let mat = materialize(formula, data, &config)?;
    fit_materialized_once_with_notes(mat)
}

fn fit_materialized_once_with_notes(
    mat: MaterializedModel<'_>,
) -> Result<FormulaFitResult, WorkflowError> {
    let inference_notes = mat.inference_notes;
    let unidentified_scalar_terms = mat.unidentified_scalar_terms;
    // The materialized numeric covariate frame, kept across the `fit_model`
    // move. `SmoothBasisSpec::structural_feature_cols` indexes THIS matrix, so
    // it is the only frame in which a smooth's covariates can be identified;
    // `fit_model` consumes the request and the fitted result does not carry it.
    // Cloning the handle is `O(1)` by construction — a `Copy` view or an `Arc`
    // bump, aliasing the same storage — and its lifetime is the caller's
    // dataset, not `mat`, so it outlives the move.
    //
    // The response side travels with it: the conditional reference for a
    // canonical binomial/Poisson fit conditions on `Xᵀ(w∘y)`, so it needs the
    // response and prior weights the fit consumed (both `Arc` handles, so this
    // is a refcount bump too).
    let standard_covariate_frame = match &mat.request {
        FitRequest::Standard(request) => Some(BasisAdequacyInputs {
            frame: request.data.clone(),
            y: request.y.clone(),
            prior_weights: request.weights.clone(),
            canonical_family: crate::fit_orchestration::drivers::basis_adequacy_canonical_family(
                &request.family,
                request.wiggle.is_some(),
                request.latent_coord.is_some(),
            ),
        }),
        _ => None,
    };
    // Exact O(n) spline-scan fast path (#1030): when the materialized request
    // is the single 1-D Gaussian-identity penalized-smooth shape the
    // state-space scan solves exactly, route through it and return the
    // scan-bearing model directly — the same penalized posterior at O(n) per
    // λ-trial instead of the dense design/Gram route. Detection is structural
    // and conservative (see `spline_scan_fast_path`); every other shape falls
    // through to the dense `fit_model` path unchanged. Mirrors the CLI
    // (main.rs run_fit) and FFI consumers, which build the persistence payload
    // from this same `SplineScanFit`.
    let mut realized_design = None;
    if let FitRequest::Standard(request) = &mat.request {
        // Route selection comes before the exact Gaussian boundary. The
        // residual cascade below is a different estimator from the dense
        // model, so whether the DENSE model reproduces `y` exactly is not its
        // question, and asking it would build and factor the n×p dense design
        // the cascade exists to avoid (#3472). Only a constant response, whose
        // exactness is read off `y - offset` with no certificate, still takes
        // the deterministic route ahead of the cascade.
        let cascade_inputs = residual_cascade_fast_path(request);
        if cascade_inputs.is_none() || gaussian_response_is_constant(request) {
            match try_deterministic_gaussian_standard_fit(request)? {
                GaussianStandardRoute::Exact(result) => {
                    return Ok(attach_basis_adequacy(
                        FitResult::Standard(result),
                        standard_covariate_frame,
                        inference_notes,
                        unidentified_scalar_terms,
                    ));
                }
                GaussianStandardRoute::Iterative(design) => realized_design = design,
            }
        }
        if let Some(inputs) = spline_scan_fast_path(request) {
            let scan = gam_solve::spline_scan::fit_spline_scan(
                &inputs.x,
                &inputs.y,
                &inputs.w,
                inputs.order,
            )
            .map_err(spline_scan_failure)?;
            return Ok(FormulaFitResult {
                result: FitResult::SplineScan(scan),
                inference_notes,
                unidentified_scalar_terms,
            });
        }
        // O(n log n) multiresolution residual-cascade fast path (#1032): a
        // scattered low-d Gaussian-identity Duchon/Matérn smooth past the
        // dense-kernel cliff. UNLIKE the scan, the cascade is a DIFFERENT
        // posterior from the dense radial term, so it only ever fires as an
        // explicit alternative estimator on the exact structural signature
        // (`residual_cascade_fast_path`). Once that explicit structural route
        // is selected, a proof or convergence refusal is propagated: silently
        // replacing it with the different dense-kernel estimator would erase
        // the typed reason automatic REML was unavailable. The save paths
        // build the persistence payload from this `ResidualCascadeFit`'s
        // `to_state` snapshot.
        if let Some(inputs) = cascade_inputs {
            let coord_refs: Vec<&[f64]> = inputs.coords.iter().map(Vec::as_slice).collect();
            let fit = gam_solve::residual_cascade::fit_residual_cascade(
                &coord_refs,
                &inputs.y,
                &inputs.w,
                &inputs.metric,
                inputs.sobolev_s,
            )
            .map_err(residual_cascade_failure)?;
            return Ok(FormulaFitResult {
                result: FitResult::ResidualCascade(fit),
                inference_notes,
                unidentified_scalar_terms,
            });
        }
    }
    // `fit_model` already returns `WorkflowError` end-to-end; propagate it
    // directly instead of stringifying then re-wrapping. A standard request
    // was refused by the exact Gaussian boundary above, so it skips that
    // certificate's second design build inside `fit_model` and fits on the
    // design the certificate realized.
    let result = match mat.request {
        FitRequest::Standard(request) => {
            fit_standard_past_exact_gaussian_boundary(request, realized_design)?
        }
        request => fit_model(request)?,
    };
    Ok(attach_basis_adequacy(
        result,
        standard_covariate_frame,
        inference_notes,
        unidentified_scalar_terms,
    ))
}

/// What [`attach_basis_adequacy`] needs from the standard request, kept across
/// the `fit_model` move.
struct BasisAdequacyInputs<'a> {
    frame: StandardFitData<'a>,
    y: std::sync::Arc<ndarray::Array1<f64>>,
    prior_weights: std::sync::Arc<ndarray::Array1<f64>>,
    canonical_family: Option<gam_terms::inference::basis_adequacy::CanonicalExponentialFamily>,
}

/// Measure each smooth's basis adequacy (#2774) and fold the verdict into the
/// fit result and its user-facing advisories.
///
/// This is the ONE seam where it happens, for the same reason the per-term
/// summary walk lives in one place: the report needs the materialized covariate
/// frame, the realized design and the converged fit at once, and exactly one
/// function in the engine holds all three. `fit_model` holds the first two but
/// not the frame's column meaning; the payload builders hold the last two but
/// have already dropped the request.
///
/// A missing verdict is never an error. `basis_adequacy_report` returns a typed
/// reason per term instead, and a fit is not refused, delayed, or altered by
/// what this finds — the only thing that changes is what the caller is told.
fn attach_basis_adequacy(
    result: FitResult,
    covariate_frame: Option<BasisAdequacyInputs<'_>>,
    mut inference_notes: FitNotes,
    unidentified_scalar_terms: Vec<UnidentifiedScalarTerm>,
) -> FormulaFitResult {
    let FitResult::Standard(mut standard) = result else {
        return FormulaFitResult {
            result,
            inference_notes,
            unidentified_scalar_terms,
        };
    };
    // The random-effect test needs only the design and the converged fit, so
    // it runs whether or not the covariate frame is available.
    let variance_component_tests =
        crate::fit_orchestration::drivers::variance_component_test_records(
            &standard.design,
            &standard.fit,
        );
    standard.fit.artifacts.random_effect_tests = variance_component_tests.random_effect;
    standard.fit.artifacts.linear_term_tests = variance_component_tests.linear_term;
    if let Some(inputs) = covariate_frame {
        standard.basis_adequacy = crate::fit_orchestration::drivers::basis_adequacy_report(
            inputs.frame.view(),
            &standard.design,
            &standard.resolvedspec,
            &standard.fit,
            &crate::fit_orchestration::drivers::BasisAdequacyResponse {
                y: inputs.y.view(),
                prior_weights: inputs.prior_weights.view(),
                canonical_family: inputs.canonical_family,
            },
        );
        inference_notes
            .advisories
            .extend(crate::fit_orchestration::drivers::basis_adequacy_notes(
                &standard.basis_adequacy,
            ));
    }
    FormulaFitResult {
        result: FitResult::Standard(standard),
        inference_notes,
        unidentified_scalar_terms,
    }
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
pub(crate) fn fit_expectile_if_requested(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<Option<ExpectileOutcome>, WorkflowError> {
    let Some(levels) = expectile_levels_for_config(config)? else {
        return Ok(None);
    };
    let mut materialized = ExpectileMaterializeNotes::default();
    let fit = match levels.as_slice() {
        [tau] => ExpectileFit::Single(fit_expectile_laws(
            formula,
            data,
            config,
            *tau,
            &mut materialized,
        )?),
        _ => ExpectileFit::Joint(fit_expectile_location_scale(
            formula,
            data,
            config,
            levels,
            &mut materialized,
        )?),
    };
    Ok(Some(ExpectileOutcome { fit, materialized }))
}

/// An expectile fit together with what its one inner materialization reported.
pub(crate) struct ExpectileOutcome {
    pub(crate) fit: ExpectileFit,
    pub(crate) materialized: ExpectileMaterializeNotes,
}

/// The advisories and removed-term records of the inner Gaussian
/// materialization an expectile driver runs. They describe the model that was
/// fitted (a capped basis, a structural warning, a pruned scalar term), so every
/// front end reports them exactly as it does for any other formula fit (#1543).
#[derive(Default)]
pub(crate) struct ExpectileMaterializeNotes {
    pub(crate) inference_notes: FitNotes,
    pub(crate) unidentified_scalar_terms: Vec<UnidentifiedScalarTerm>,
}

/// The two shapes an expectile request resolves to.
pub(crate) enum ExpectileFit {
    /// One level: the LAWS fit of that level alone.
    Single(StandardFitResult),
    /// Several levels: one joint non-crossing location-scale fit.
    Joint(ExpectileLocationScaleFitResult),
}

impl ExpectileFit {
    pub(crate) fn into_fit_result(self) -> FitResult {
        match self {
            Self::Single(result) => FitResult::Standard(result),
            Self::Joint(result) => FitResult::ExpectileLocationScale(result),
        }
    }
}

/// The log-σ formula of a joint expectile fit: the caller's `noise_formula`, or
/// else the mean formula's right-hand side, so `σ(x)` is as flexible as `μ(x)`.
pub(crate) fn expectile_noise_formula(
    formula: &str,
    config: &FitConfig,
) -> Result<String, WorkflowError> {
    match config.noise_formula.as_deref() {
        Some(noise) => Ok(noise.to_string()),
        None => formula
            .split_once('~')
            .map(|(_, rhs)| rhs.trim().to_string())
            .ok_or_else(|| WorkflowError::InvalidConfig {
                reason: format!("expectile formula `{formula}` has no `~`"),
            }),
    }
}

/// Joint non-crossing multi-level expectile fit.
///
/// Fitting each level on its own lets the curves cross: nothing ties the
/// separately penalized surfaces together, and under heteroscedasticity their
/// slopes differ, so they meet as soon as the data (or an extrapolation) is far
/// enough from the centre. The joint model is the location-scale expectile
///
/// ```text
///   e_τ(x) = μ(x) + c_τ·σ(x),     y = μ(x) + σ(x)·ε,  ε ⟂ x,
/// ```
///
/// under which every conditional expectile of `y | x` has exactly this form,
/// with `c_τ` the `τ`-expectile of `ε`. `μ` and `σ` are the Gaussian
/// location-scale GAM (`noise_formula`, defaulting to the mean formula's
/// right-hand side), so both surfaces carry their own function penalties and
/// REML/LAML-selected smoothing and come only from a certified fit. `c_τ` is the
/// prior-weighted empirical `τ`-expectile of the standardized residuals
/// `(yᵢ − μᵢ)/E[σᵢ]`, solved in closed form.
///
/// The expectile of a fixed sample is strictly increasing in `τ`, and
/// `σ(x) > 0` everywhere (the link has a positive floor), so for `τ₁ < τ₂`
/// `e_τ₂(x) − e_τ₁(x) = (c_τ₂ − c_τ₁)·σ(x) > 0` at *every* `x`, extrapolation
/// included. Ordering is a property of the construction, never a post-hoc sort.
///
/// `σ` enters as its posterior mean `E[σ] = f + exp(m + v/2)` under the log-σ
/// block's conditional Gaussian posterior `N(m, v)` — the same functional the
/// predictor evaluates — and `μ` is identity-linked, so its plug-in is its
/// posterior mean.
fn fit_expectile_location_scale(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
    levels: Vec<f64>,
    materialized: &mut ExpectileMaterializeNotes,
) -> Result<ExpectileLocationScaleFitResult, WorkflowError> {
    if config.frailty.is_active() {
        return Err(WorkflowError::InvalidConfig {
            reason: "expectile regression does not support frailty; use a survival/frailty-aware family instead"
                .to_string(),
        });
    }
    let noise_formula = expectile_noise_formula(formula, config)?;
    let location_scale_config = FitConfig {
        family: Some("gaussian".to_string()),
        link: Some("identity".to_string()),
        expectile_tau: None,
        frailty: FrailtySpec::None,
        noise_formula: Some(noise_formula),
        ..config.clone()
    };
    let mat = materialize(formula, data, &location_scale_config)?;
    materialized.inference_notes = mat.inference_notes;
    materialized.unidentified_scalar_terms = mat.unidentified_scalar_terms;
    let FitRequest::GaussianLocationScale(request) = mat.request else {
        return Err(WorkflowError::InvalidConfig {
            reason: "joint expectile regression is only defined for a Gaussian location-scale \
                     response (non-survival, non-latent)"
                .to_string(),
        });
    };
    if request.wiggle.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "expectile regression does not support flexible-link wiggle".to_string(),
        });
    }
    let y = request.spec.y.clone();
    let prior_weights = request.spec.weights.clone();
    let mean_offset = request.spec.mean_offset.clone();
    let log_sigma_offset = request.spec.log_sigma_offset.clone();
    let FitResult::GaussianLocationScale(location_scale) =
        fit_model(FitRequest::GaussianLocationScale(request))?
    else {
        return Err(raised_fit_failure(
            FailureCategory::Invariant,
            "joint expectile: the Gaussian location-scale request returned another fit kind"
                .to_string(),
        ));
    };

    let standardized_expectiles = joint_expectile_standardized_expectiles(
        &location_scale,
        y.view(),
        prior_weights.view(),
        mean_offset.view(),
        log_sigma_offset.view(),
        &levels,
    )?;
    Ok(ExpectileLocationScaleFitResult {
        location_scale,
        levels,
        standardized_expectiles,
    })
}

/// The level constants `c_τ` of a joint expectile fit: the prior-weighted
/// empirical `τ`-expectiles of the standardized residuals `(yᵢ − μᵢ)/E[σᵢ]`,
/// with `E[σᵢ] = f + exp(mᵢ + vᵢ/2)` under the log-σ block's conditional
/// Gaussian posterior `N(mᵢ, vᵢ)`.
///
/// `vᵢ` comes from the Scale block of the fit's joint conditional covariance
/// (coefficient layout `[mean | scale]`). That covariance is part of the
/// estimand, so a fit without it is refused with a typed error instead of
/// being standardized by the plug-in σ.
fn joint_expectile_standardized_expectiles(
    location_scale: &GaussianLocationScaleFitResult,
    y: ArrayView1<'_, f64>,
    prior_weights: ArrayView1<'_, f64>,
    mean_offset: ArrayView1<'_, f64>,
    log_sigma_offset: ArrayView1<'_, f64>,
    levels: &[f64],
) -> Result<Vec<f64>, WorkflowError> {
    use gam_linalg::matrix::DenseDesignOperator;
    use gam_problem::BlockRole;

    let invariant = |reason: String| {
        raised_fit_failure(
            FailureCategory::Invariant,
            format!("joint expectile: {reason}"),
        )
    };
    let fit = &location_scale.fit;
    let beta_mu = crate::inference::model::gaussian_location_scale_mean_beta(&fit.fit)
        .ok_or_else(|| invariant("fit has no location block".to_string()))?;
    let beta_sigma = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .map(|block| block.beta.clone())
        .ok_or_else(|| invariant("fit has no scale block".to_string()))?;
    let mu = fit
        .mean_design
        .apply(beta_mu.view())
        .map_err(|error| invariant(format!("could not evaluate the mean design: {error}")))?
        + &mean_offset;
    let eta_sigma = fit
        .noise_design
        .apply(beta_sigma.view())
        .map_err(|error| invariant(format!("could not evaluate the log-σ design: {error}")))?
        + &log_sigma_offset;
    let n = y.len();
    if mu.len() != n || eta_sigma.len() != n || prior_weights.len() != n {
        return Err(invariant(format!(
            "row counts disagree: y={n}, μ={}, η_σ={}, weights={}",
            mu.len(),
            eta_sigma.len(),
            prior_weights.len()
        )));
    }
    // Posterior variance of η_σ per row from the Scale block of the joint
    // conditional covariance (coefficient layout `[mean | scale]`). `c_τ`
    // integrates σ over this posterior, so a fit without it has no `c_τ`:
    // a typed constrained-posterior decline is refused with its reason, and a
    // missing covariance with no decline breaks the location-scale fit contract.
    // Neither is ever read as zero posterior variance (the plug-in σ).
    let p_mu = beta_mu.len();
    let p_sigma = beta_sigma.len();
    fit.fit
        .require_posterior_mean("joint expectile c_τ")
        .map_err(|error| {
            raised_fit_failure(FailureCategory::Input, format!("joint expectile: {error}"))
        })?;
    let covariance = fit.fit.beta_covariance().ok_or_else(|| {
        invariant(
            "c_τ integrates σ over the log-σ posterior, but the location-scale fit carries \
             neither its joint posterior covariance nor a typed posterior-moment decline"
                .to_string(),
        )
    })?;
    if covariance.nrows() < p_mu + p_sigma || covariance.ncols() < p_mu + p_sigma {
        return Err(invariant(format!(
            "covariance is {}x{}, smaller than the {} location-scale coefficients",
            covariance.nrows(),
            covariance.ncols(),
            p_mu + p_sigma
        )));
    }
    let scale_block = covariance
        .slice(ndarray::s![p_mu..p_mu + p_sigma, p_mu..p_mu + p_sigma])
        .to_owned();
    let log_sigma_variance = fit
        .noise_design
        .design
        .quadratic_form_diag(&scale_block)
        .map_err(|error| invariant(format!("log-σ posterior variance: {error}")))?;
    let sigma_floor = location_scale.response_scale * location_scale.sigma_floor;
    let standardized: Vec<f64> = (0..n)
        .map(|i| {
            let sigma = gam_model_kernels::sigma_link::logb_sigma_posterior_mean_with_floor_scalar(
                sigma_floor,
                eta_sigma[i],
                log_sigma_variance[i],
            );
            (y[i] - mu[i]) / sigma
        })
        .collect();
    let prior_weights = prior_weights.to_vec();
    let standardized_expectiles = levels
        .iter()
        .map(|&tau| weighted_empirical_expectile(&standardized, &prior_weights, tau))
        .collect::<Result<Vec<f64>, String>>()
        .map_err(|reason| {
            raised_fit_failure(FailureCategory::Input, format!("joint expectile: {reason}"))
        })?;
    if standardized_expectiles
        .windows(2)
        .any(|pair| !(pair[0] < pair[1]))
    {
        return Err(raised_fit_failure(
            FailureCategory::Input,
            format!(
                "joint expectile: the standardized residual expectiles {standardized_expectiles:?} \
                 at levels {levels:?} are not strictly increasing — the standardized residuals \
                 carry no spread to order the levels by"
            ),
        ));
    }
    Ok(standardized_expectiles)
}

/// Least Asymmetrically Weighted Squares (LAWS) driver for expectile GAMs.
///
/// The τ-expectile surface minimizes `Σ wᵢ(τ)·(yᵢ − μᵢ)²` with the residual-
/// sign asymmetric weight `wᵢ(τ)`. The asymmetric loss is convex and
/// continuously differentiable: each side of zero is a positive quadratic and
/// both one-sided derivatives agree at zero. LAWS solves the penalized WLS
/// problem with weights frozen at the current sign pattern, then recomputes the
/// pattern. A returned estimator must satisfy the KKT residual of the original
/// asymmetric objective; an iteration cap is only termination evidence, never
/// an estimator-selection rule.
///
/// Because λ̂ is re-selected on every weight vector, the sign map can have no
/// fixed point: a boundary row whose weight flip moves λ̂ enough to flip its
/// own residual back makes the map cycle (#3039). A proven cycle hands the
/// disagreeing rows to the generalized (Clarke) fixed point of the
/// subgradient weight map, where a row at `r = 0` takes a fractional
/// asymmetry in `[min(τ,1−τ), max(τ,1−τ)]`, solved by projected Newton with
/// the analytic residual-weight Jacobian through `dρ̂/dw`. The same KKT
/// certificate is the only acceptance.
///
/// Each inner solve is the FULL standard Gaussian-identity GAM: any basis,
/// tensor, spatial smooth, by-variable, random effect, plus REML λ-selection on
/// the current asymmetric weights. The returned fit is an ordinary
/// [`FitResult::Standard`] whose coefficients ARE the penalized τ-expectile —
/// every downstream consumer (predict, posterior bands, persistence) works
/// unchanged. Its published coefficient covariance is the penalized
/// Newey–Powell sandwich of [`publish_expectile_sandwich_covariance`], never
/// the Gaussian working-model `φ̂·H⁻¹` of the last inner solve.
fn fit_expectile_laws(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
    tau: f64,
    materialized: &mut ExpectileMaterializeNotes,
) -> Result<StandardFitResult, WorkflowError> {
    if config.frailty.is_active() {
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
        // The inner Gaussian-identity design carries no frailty.
        frailty: FrailtySpec::None,
        ..config.clone()
    };

    // Materialize once to capture the fixed training design, response, offset,
    // and base prior weights. The design (basis, penalties, identifiability
    // transforms) does not depend on the prior weights, so it is reused across
    // every LAWS iteration; only the weight vector and the resulting β change.
    let base_mat = materialize(formula, data, &gaussian_config)?;
    materialized.inference_notes = base_mat.inference_notes;
    materialized.unidentified_scalar_terms = base_mat.unidentified_scalar_terms;
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
    // Cold start: unweighted base weights ⇒ the first inner fit is the OLS
    // mean GAM, the natural warm start for any τ.
    let mut weights = Arc::clone(&base_weights);
    // The LAWS map is deterministic given a sign pattern. Brent detection
    // proves recurrence using one O(n) sign checkpoint; no iteration-count
    // multiple of the training data is retained.
    let mut sign_cycle = ExpectileSignCycle::default();
    // Asymmetry `a_i = w_i / base_i` of the weights the next inner fit uses;
    // `None` while the weights are still the cold-start base weights.
    let mut asymmetry: Option<Array1<f64>> = None;
    // Set once the sign map has provably cycled: from then on the rows whose
    // weight disagrees with their residual sign are resolved at the
    // generalized fixed point instead of by the (cycling) sign map. Holds the
    // iteration and cycle length that triggered it, for diagnostics.
    let mut generalized_since: Option<(usize, usize)> = None;
    let (asym_lo, asym_hi) = (tau.min(1.0 - tau), tau.max(1.0 - tau));
    // Evidence for the typed exhaustion error: (dimensionless KKT residual,
    // configured KKT bound) of the final uncertified iterate.
    let mut last_kkt = (f64::NAN, f64::NAN);
    let mut last_rho_checkpoint = Vec::new();

    // Reuse the request's explicit outer-work budget; LAWS does not introduce a
    // second hidden iteration knob. The budget is a safety guard only: hitting
    // it without the certificate below is typed nonconvergence (SPEC rule 20).
    let max_laws_iters = options.max_iter;
    if max_laws_iters == 0 || !(options.tol.is_finite() && options.tol > 0.0) {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "expectile LAWS requires a positive iteration budget and finite positive KKT \
                 tolerance; got max_iter={max_laws_iters}, tol={}",
                options.tol,
            ),
        });
    }

    for iteration in 1..=max_laws_iters {
        let request = StandardFitRequest {
            data: design_data.clone(),
            y: Arc::clone(&y),
            weights: Arc::clone(&weights),
            offset: Arc::clone(&offset),
            spec: spec.clone(),
            family: gaussian_family.clone(),
            // Expectile LAWS fits a Gaussian-identity inner family; no Tweedie
            // power to estimate (#2026).
            options: options.clone(),
            kappa_options: kappa_options.clone(),
            wiggle: None,
            coefficient_groups: coefficient_groups.clone(),
            penalty_block_gamma_priors: penalty_block_gamma_priors.clone(),
            latent_coord: None,
        };
        let result = fit_standard_model(request).map_err(WorkflowError::from)?;
        // Training-scale fitted mean μ = X·β (identity link, zero-checked
        // offset folded by the design path). The design columns match the
        // combined coefficient vector exactly (the same contract `predict`
        // and the safety tests rely on).
        let mu = result
            .design
            .apply(result.fit.beta.view())
            .map_err(|error| {
                raised_fit_failure(
                    FailureCategory::Invariant,
                    format!("expectile LAWS could not evaluate fitted design: {error}"),
                )
            })?;
        if mu.len() != n {
            return Err(raised_fit_failure(
                FailureCategory::Invariant,
                format!(
                    "expectile LAWS: fitted mean length {} disagrees with response length {n}",
                    mu.len()
                ),
            ));
        }
        // `design.apply` already folds the design's fixed affine channel
        // (non-zero endpoint anchor, #2297) into `X·β`, so only the user offset
        // is added; adding `affine_offset` again would double-count the pin and
        // bias every expectile working weight for an anchored smooth.
        let mut mu_off = mu;
        mu_off += offset.as_ref();

        let sign: Vec<bool> = (0..n).map(|i| y[i] > mu_off[i]).collect();
        let next_weights = expectile_row_weights(y.view(), mu_off.view(), base_weights.view(), tau);

        // KKT certificate for the CONVEX penalized asymmetric-least-squares
        // problem at the fit's own selected λ. The asymmetric loss
        // ρ_τ(r) = |τ − 1[r<0]|·r² is convex and continuously differentiable
        // (its derivative vanishes at r = 0 from both sides), so the true
        // penalized objective J(β) = Σ wᵢ(τ)·rᵢ² + βᵀS_λβ has a checkable
        // gradient at the returned β. The inner solve certifies stationarity
        // of the FROZEN-weight problem, Xᵀ(w_used ∘ r) = S_λ β, hence
        //   ∇J(β)/2 = Xᵀ((w_used − w_new) ∘ r),
        // supported exactly on rows whose residual sign disagrees with the
        // pattern the weights were frozen at. The production audit normalizes
        // each coefficient defect by its Cauchy–Schwarz score scale, making the
        // result invariant to column, response, and prior-weight scale while
        // remaining defined when an unpenalized frozen score cancels to zero.
        let residual = y.as_ref() - &mu_off;
        let kkt = expectile_kkt_residual(
            &result.design.design,
            residual.view(),
            weights.view(),
            next_weights.view(),
        )
        .map_err(|reason| {
            raised_fit_failure(
                FailureCategory::Numerical,
                format!(
                    "expectile LAWS KKT audit failed at iteration {iteration} \
                     (rho_checkpoint={:?}): {reason}",
                    result.fit.log_lambdas.to_vec(),
                ),
            )
        })?;
        let kkt_bound = options.tol;
        if kkt <= kkt_bound {
            let mut result = result;
            publish_expectile_sandwich_covariance(
                &mut result.fit,
                &result.design.design,
                residual.view(),
                weights.view(),
                tau,
            )?;
            return Ok(result);
        }
        last_kkt = (kkt, kkt_bound);
        last_rho_checkpoint = result.fit.log_lambdas.to_vec();
        let target_asymmetry = Array1::from_shape_fn(n, |i| if sign[i] { tau } else { 1.0 - tau });
        let (entered_at, cycle_length) = match generalized_since {
            Some(entered) => entered,
            None => match sign_cycle.observe(&sign) {
                None => {
                    weights = Arc::new(next_weights);
                    asymmetry = Some(target_asymmetry);
                    continue;
                }
                Some(cycle_length) => {
                    generalized_since = Some((iteration, cycle_length));
                    (iteration, cycle_length)
                }
            },
        };

        // Generalized (Clarke) fixed point. The REML-selected λ̂ depends on the
        // weights, so a row whose residual sits at the sign boundary can have
        // `r > 0` at one endpoint weight and `r < 0` at the other: the sign map
        // then has no fixed point and cycles. The LAWS weight is the
        // subgradient `|τ − 1[r < 0]|`, which at `r = 0` is the whole interval
        // `[min(τ,1−τ), max(τ,1−τ)]`; the fixed point of that set-valued map is
        // the complementarity problem
        //   a_i = τ-side endpoint if r_i > 0, other endpoint if r_i < 0,
        //   a_i ∈ [lo, hi] if r_i = 0,
        // which exists by the intermediate value theorem. It is solved by a
        // projected Newton step on the free rows (interior asymmetry, or an
        // endpoint that disagrees with the residual sign), driving `r_F → 0`
        // with the analytic Jacobian of `expectile_free_row_jacobian`. Rows at
        // a sign-consistent endpoint are complementary already and stay
        // fixed. Acceptance is still the KKT certificate above, whose defect on
        // a fractional row is `(a_i − target_i)·base_i·r_i → 0` as `r_i → 0`.
        let generalized_failure = |reason: String| {
            raised_fit_failure(
                FailureCategory::Convergence,
                format!(
                    "expectile LAWS sign map cycled (tau={tau}, detected at iteration \
                     {entered_at}, cycle_length={cycle_length}) and the generalized fixed \
                     point could not be advanced at iteration {iteration} (KKT residual \
                     {kkt:.3e} vs scaled tolerance {kkt_bound:.3e}, rho_checkpoint={:?}): \
                     {reason}; non-convergence is a typed error, never a best-effort fit",
                    result.fit.log_lambdas.to_vec(),
                ),
            )
        };
        let Some(current) = asymmetry.as_mut() else {
            return Err(generalized_failure(
                "the cycle was detected before any sign-derived weights were used".to_string(),
            ));
        };
        let free: Vec<usize> = (0..n)
            .filter(|&i| {
                base_weights[i] > 0.0
                    && ((current[i] > asym_lo && current[i] < asym_hi)
                        || current[i] != target_asymmetry[i])
            })
            .collect();
        let jacobian = expectile_free_row_jacobian(
            &result.fit,
            &result.design,
            residual.view(),
            base_weights.view(),
            &free,
        )
        .map_err(|reason| generalized_failure(format!("residual-weight Jacobian: {reason}")))?;
        let newton_rhs = Array1::from_iter(free.iter().map(|&i| -residual[i]));
        let step = solve_square_partial_pivot(jacobian, newton_rhs)
            .map_err(|reason| generalized_failure(format!("Newton system: {reason}")))?;
        log::debug!(
            "[expectile] generalized fixed point iteration {iteration}: {} free rows, \
             max |r_F| = {:.3e}, KKT = {kkt:.3e}, rho = {:?}",
            free.len(),
            free.iter().fold(0.0_f64, |worst, &i| worst.max(residual[i].abs())),
            result.fit.log_lambdas.to_vec(),
        );
        let mut moved = false;
        for (slot, &i) in free.iter().enumerate() {
            let projected = (current[i] + step[slot]).clamp(asym_lo, asym_hi);
            moved |= projected != current[i];
            current[i] = projected;
        }
        if !moved {
            return Err(generalized_failure(format!(
                "the projected Newton step on {} free rows is null at the box boundary",
                free.len()
            )));
        }
        weights = Arc::new(Array1::from_shape_fn(n, |i| base_weights[i] * current[i]));
    }

    Err(raised_fit_failure(
        FailureCategory::Convergence,
        format!(
            "expectile LAWS exhausted its {max_laws_iters}-iteration safety cap without a \
             KKT certificate for the convex asymmetric least-squares problem (tau={tau}, \
             final KKT residual={:.3e} vs scaled tolerance {:.3e}, \
             rho_checkpoint={last_rho_checkpoint:?}); the iteration cap \
             never selects the estimator — non-convergence is a typed error",
            last_kkt.0, last_kkt.1,
        ),
    ))
}

/// Replace the working-model covariance of a certified LAWS fixed point with
/// the penalized Newey–Powell sandwich.
///
/// The expectile is an M-estimator, not a likelihood fit: `β̂` solves
/// `ψ(β) = Xᵀ(w ∘ r) − S_λβ = 0` with `wᵢ = baseᵢ·|τ − 1[rᵢ < 0]|`, whose
/// Jacobian is `−H`, `H = XᵀWX + S_λ` — the unscaled penalized Hessian the
/// last inner solve already factored. The inner fit publishes
/// `Vb = φ̂·H⁻¹`, which is correct only if `Var(wᵢrᵢ) = φ̂·wᵢ`, i.e. only if
/// the asymmetric weights were inverse variances. They are not: they are the
/// loss asymmetry, and under heteroscedastic noise the working model
/// under-covers wherever the noise is large (τ = 0.05/0.95 bands covered
/// 0.81/0.73 at nominal 0.95).
///
/// Newey & Powell (1987, Thm 3) give the unpenalized law
/// `√n(β̂ − β) → N(0, A⁻¹BA⁻¹)`, `A = E[w xxᵀ]`, `B = E[w²r² xxᵀ]`, with no
/// dispersion factor anywhere: the scale lives in `r` itself. The penalized
/// analogue keeps the smoothing prior `β ~ N(0, φ̂·S_λ⁻)` that makes `Vb`
/// Bayesian (Wahba 1983; Nychka 1988), so the published covariance is the
/// prior-inclusive sandwich
///
///   `V = H⁻¹ (c·Xᵀ diag(w²r²) X + φ̂·S_λ) H⁻¹`,   `c = n₊ / (n₊ − edf)`,
///
/// the Bayesian ("penalty as prior") form of the Huber–White sandwich.
/// Three limits pin every constant:
///
/// * `S_λ → 0` recovers the Newey–Powell `A⁻¹BA⁻¹` exactly (up to `c`).
/// * Under the working model, `E[c·w²r²] = φ̂·w` row by row, so `V → Vb`: the
///   sandwich changes nothing where the Gaussian form was already right.
/// * `c` is the HC1 degrees-of-freedom correction, the same `n₊ − edf` the
///   inner fit's `φ̂ = Σwr² / (n₊ − edf)` divides by (mgcv `gam.scale`), so the
///   meat and the prior term are debiased on one scale; `n₊` counts the rows
///   with positive weight, exactly as `φ̂` does.
///
/// `φ̂` enters only through the prior term, where it is the posterior
/// variance's own scale; it never multiplies the meat.
///
/// Because `φ̂·S_λ = φ̂·H − φ̂·XᵀWX`, the sandwich is an exact rank-`n`
/// correction of the published `Vb` that needs neither `S_λ` nor `H`:
///
///   `V = Vb + Vb Xᵀ diag(d) X Vb`,   `dᵢ = c·wᵢ²rᵢ²/φ̂² − wᵢ/φ̂`.
///
/// It holds verbatim for a constraint-projected `Vb = φ̂·Z(ZᵀHZ)⁻¹Zᵀ`, so the
/// identifiability and active-constraint gauge the fit already chose carries
/// over. The correction is added to the conditional AND smoothing-corrected
/// stores through the one seam that keeps them consistent, so the
/// smoothing-parameter uncertainty term `Vp − Vb` survives and every consumer
/// — predict bands, posterior draws, summary SEs, the CLI, Python — reads the
/// same matrix. A fit on the zero-dispersion boundary (`φ̂ = 0`, every
/// residual zero) has a zero meat and nothing to correct.
///
/// A fit with no dense `Vb` (the memory governor refused it and published only
/// a factorized diagonal, or inference was off) cannot carry the sandwich, so
/// its covariance is declined with a typed reason instead of leaving the
/// working-model diagonal or Hessian to be read under the expectile's name.
fn publish_expectile_sandwich_covariance(
    fit: &mut gam_solve::estimate::UnifiedFitResult,
    design: &gam_linalg::matrix::DesignMatrix,
    residual: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    tau: f64,
) -> Result<(), WorkflowError> {
    let invariant = |reason: String| {
        raised_fit_failure(
            FailureCategory::Invariant,
            format!("expectile sandwich covariance (tau={tau}): {reason}"),
        )
    };
    let Some(vb) = fit.covariance_conditional.clone() else {
        // The declination is also what stops every Hessian reconstruction
        // (summary, predict, sampling) from rebuilding the working-model `Vb`.
        let declined =
            gam_solve::estimate::CovarianceDeclined::ExpectileSandwichRequiresDenseCovariance {
                coefficients: fit.beta.len(),
            };
        log::debug!("[expectile] {}", declined.explain());
        fit.covariance_corrected = None;
        if let Some(inference) = fit.inference.as_mut() {
            inference.factorized_standard_errors = None;
            inference.smoothing_correction_factorized = None;
        }
        fit.artifacts.covariance_declined = Some(declined);
        return Ok(());
    };
    let n = design.nrows();
    if residual.len() != n || weights.len() != n {
        return Err(invariant(format!(
            "design rows={n}, residual={}, weights={}",
            residual.len(),
            weights.len()
        )));
    }
    let phi = fit
        .coefficient_covariance_scale()
        .map_err(|error| invariant(error.to_string()))?;
    if !(phi.is_finite() && phi >= 0.0) {
        return Err(invariant(format!(
            "coefficient covariance scale must be finite and non-negative, got {phi:?}"
        )));
    }
    if phi == 0.0 {
        return Ok(());
    }
    let edf = fit.edf_total().ok_or_else(|| {
        invariant(
            "a fit that publishes a covariance must carry its effective degrees of freedom"
                .to_string(),
        )
    })?;
    let n_positive = weights.iter().filter(|&&w| w > 0.0).count() as f64;
    let residual_df = n_positive - edf;
    if !(residual_df.is_finite() && residual_df > 0.0) {
        return Err(invariant(format!(
            "residual degrees of freedom n₊ − edf = {n_positive} − {edf} must be positive"
        )));
    }
    let hc1 = n_positive / residual_df;
    let row_correction = Array1::from_shape_fn(n, |i| {
        let score = weights[i] * residual[i];
        hc1 * score * score / (phi * phi) - weights[i] / phi
    });
    let certified = gam_linalg::matrix::FiniteSignedWeightsView::try_from_array(&row_correction)
        .map_err(invariant)?;
    let middle = gam_linalg::matrix::xt_diag_x_signed(design, certified)
        .map_err(invariant)?
        .to_dense();
    if middle.dim() != vb.dim() {
        return Err(invariant(format!(
            "design Gram is {:?} but the published covariance is {:?}",
            middle.dim(),
            vb.dim()
        )));
    }
    let mut correction = vb.dot(&middle).dot(&vb);
    gam_linalg::matrix::symmetrize_in_place(&mut correction);
    fit.add_coefficient_covariance_correction(&correction)
        .map_err(|error| invariant(error.to_string()))
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
///   hyperpriors, no linear/box constraints, no Firth, no externally
///   injected null-space dims;
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
///   jointly. Natural cubic regression (`bs="cr"`) terms also fall
///   through: their knot-value parameterization is a finite-rank regression
///   spline, not the scan's full smoothing-spline state-space posterior;
/// - the offset is identically zero and every weight is finite and positive;
/// - at least `order + 1` distinct finite abscissae (the scan's `order`
///   diffuse innovations plus one proper innovation to profile σ²).
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
/// Only two identifiability policies are eligible: `none` and sum-to-zero
/// centring. Each removes at most the constant, and the intercept puts it
/// back, so the model spans the same `{1, x, …} ⊕ wiggle` the scan solves.
/// Linear-trend removal (`identifiability="linear"`) takes the `x` direction
/// out of the model. Design-column orthogonality and a frozen transform can
/// impose any constraint. The scan honours none of these, so they fall through
/// to the dense path (#3870).
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
    if !term.shape.is_none() || term.joint_null_rotation.is_some() {
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
    // null-space shrinkage block, in its function-space form `G Z (ZᵀGZ)⁻¹ ZᵀG`
    // with `G` the basis Gram (see `bspline_penalty_candidates`) —
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
        || !matches!(
            bspec.identifiability,
            gam_terms::basis::BSplineIdentifiability::None
                | gam_terms::basis::BSplineIdentifiability::WeightedSumToZero { .. }
        )
        || !bspec.boundary_conditions.is_free()
        || !matches!(bspec.boundary, gam_terms::basis::OneDimensionalBoundary::Open)
        || matches!(
            bspec.knotspec,
            gam_terms::basis::BSplineKnotSpec::PeriodicUniform { .. }
        )
        // `bs="cr"` materialises a `NaturalCubicRegression` value-knot
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

/// Whether a radial spec asks for geometry the cascade cannot represent. The
/// cascade fits in open Euclidean coordinates under one unit per-axis metric,
/// so a wrapped axis (`period=[…]`) or learned per-axis length scales
/// (`scale_dims=true`, `FitConfig::scale_dimensions`) would be silently
/// dropped. Such a term stays on the dense radial path, which honours both.
fn radial_geometry_leaves_cascade_metric(
    periodic: Option<&[Option<f64>]>,
    aniso_log_scales: Option<&[f64]>,
) -> bool {
    periodic.is_some_and(|axes| axes.iter().any(Option::is_some)) || aniso_log_scales.is_some()
}

/// Structural signature of a residual-cascade-eligible request: the scattered
/// radial smooth's coordinate columns and the Sobolev order it requests
/// (before the Wendland native-window clamp). Produced by
/// [`residual_cascade_structural_signature`]; carries no size information.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualCascadeSignature {
    pub feature_cols: Vec<usize>,
    pub requested_sobolev_order: f64,
}

/// Pure structural predicate for the O(n log n) multiresolution
/// residual-cascade fast path (issue #1032): every eligibility guard of
/// [`residual_cascade_fast_path`] EXCEPT the dense-kernel size gate. It returns
/// `Some` only when ALL of the following hold:
/// - family is Gaussian + identity link (the scattered low-d smooth the
///   cascade solves);
/// - none of the exotic-link / constraint / Firth / coefficient-group /
///   hyperprior machinery is engaged;
/// - the model is exactly one smooth term — no linear terms, no random
///   effects, no by-variables;
/// - that smooth is a scattered radial spatial smooth (`Duchon` or `Matern`)
///   over `d ∈ {2, 3}` coordinates with no shape constraint, no periodic axis
///   and no per-axis anisotropy (the cascade's open unit metric represents
///   neither);
/// - the offset is identically zero, every weight is finite and positive, and
///   every coordinate and response value is finite.
///
/// Kept separate from the size gate so each structural guard is observable on
/// its own at small `n` (issue #3550): below the cliff the full fast path is
/// `None` for every input, which would mask a mis-firing structural guard.
pub fn residual_cascade_structural_signature(
    request: &StandardFitRequest<'_>,
) -> Option<ResidualCascadeSignature> {
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
    if !term.shape.is_none() || term.joint_null_rotation.is_some() {
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
            if radial_geometry_leaves_cascade_metric(
                spec.periodic.as_deref(),
                spec.aniso_log_scales.as_deref(),
            ) {
                return None;
            }
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
            if radial_geometry_leaves_cascade_metric(
                spec.periodic.as_deref(),
                spec.aniso_log_scales.as_deref(),
            ) {
                return None;
            }
            // Matérn smoothness ν sets native Sobolev order ν + d/2; the cascade
            // frame represents up to (d+3)/2, so the fast path's clamp applies
            // the ceiling. (d is known just below from feature_cols.)
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
    if request.y.len() != request.data.nrows()
        || feature_cols.iter().any(|&c| c >= request.data.ncols())
    {
        return None;
    }
    if feature_cols
        .iter()
        .any(|&c| request.data.column(c).iter().any(|v| !v.is_finite()))
        || request.y.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    Some(ResidualCascadeSignature {
        feature_cols: feature_cols.to_vec(),
        requested_sobolev_order: requested_s,
    })
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
/// when the request carries the structural signature
/// ([`residual_cascade_structural_signature`]) AND `n` is past the derived
/// dense-kernel cliff (`past_dense_kernel_cliff`) — below it the dense radial
/// path is both exact-posterior and cheap, so there is no reason to change
/// estimators.
///
/// The returned [`ResidualCascadeInputs`] carry a unit per-axis metric, which
/// is the spec's own radial distance because the structural signature refuses
/// periodic and anisotropic specs; the quasi-uniformity guard inside
/// [`gam_solve::residual_cascade::fit_residual_cascade`] (issue caveat 2)
/// is the no-regression gate that refuses the selected route when a
/// near-degenerate metric would break the BPX iteration bound. That refusal is
/// propagated; it never silently changes the estimator.
pub fn residual_cascade_fast_path(
    request: &StandardFitRequest<'_>,
) -> Option<ResidualCascadeInputs> {
    let signature = residual_cascade_structural_signature(request)?;
    let d = signature.feature_cols.len();
    if !past_dense_kernel_cliff(request.y.len(), d) {
        return None;
    }
    let coords: Vec<Vec<f64>> = signature
        .feature_cols
        .iter()
        .map(|&c| request.data.column(c).iter().copied().collect())
        .collect();
    let y: Vec<f64> = request.y.iter().copied().collect();
    let w: Vec<f64> = request.weights.iter().copied().collect();
    let metric = vec![1.0_f64; d];
    let sobolev_s = cascade_sobolev_order(signature.requested_sobolev_order, d);
    Some(ResidualCascadeInputs {
        coords,
        y,
        w,
        metric,
        sobolev_s,
    })
}

/// Parse a formula, resolve it against a dataset, and produce a ready-to-fit `FitRequest`.
pub(crate) fn family_requests_transformation_normal(family: Option<&str>) -> bool {
    family
        .map(|name| name.trim().to_ascii_lowercase().replace('_', "-"))
        .as_deref()
        == Some("transformation-normal")
}

/// Refuse `firth=true` on a route whose fit reads no Firth setting.
///
/// Only the standard route passes `config.firth` to the solver; the survival,
/// transformation-normal and location-scale fits run with Firth off. The CLI
/// refused `--firth` on those routes itself, so `gamfit.fit(..., firth=True)`
/// and a Rust caller got a fit without Firth instead of the refusal.
fn refuse_unread_firth(config: &FitConfig, model: &str) -> Result<(), WorkflowError> {
    if config.firth {
        return Err(WorkflowError::InvalidConfig {
            reason: format!("firth is not supported for {model}; that fit reads no Firth setting"),
        });
    }
    Ok(())
}

/// Build the design/request geometry for a formula against a dataset. This is the
/// FIT path: for survival location-scale / latent modes it resolves the baseline
/// θ via a real inner fit. Use [`materialize_structural`] for formula validation,
/// which must not fit.
pub fn materialize<'a>(
    formula: &str,
    data: &'a Dataset,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    if config.ctn_stage1.is_some() || config.frozen_ctn.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "CTN composition requires fit_from_formula or fit_formula_to_payload".into(),
        });
    }
    gam_runtime::parallel::install(|| materialize_impl(formula, data, config, false))
}

/// Structural-only materialization for `validate_formula`: builds the same
/// request geometry/metadata but skips every inner fit (notably the survival
/// baseline-θ resolution), honoring validation's "without fitting" contract.
pub fn materialize_structural<'a>(
    formula: &str,
    data: &'a Dataset,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    gam_runtime::parallel::install(|| materialize_impl(formula, data, config, true))
}

fn materialize_impl<'a>(
    formula: &str,
    data: &'a Dataset,
    config: &FitConfig,
    structural_only: bool,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let config = config
        .clone()
        .resolve()
        .map_err(|reason| WorkflowError::InvalidConfig { reason })?;
    let config = &config;
    gam_gpu::configure_global_policy(config.gpu_policy);
    let parsed = parse_formula(formula)?;
    data.validate_fit_boundary(&fit_required_columns(&parsed, config)?)?;
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
            return Err(WorkflowError::TransformationNormalConflict {
                conflict: TransformationNormalConflict::SurvIntervalResponse,
            });
        }
        refuse_unread_firth(effective_config, "survival models")?;
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
            structural_only,
        )
    } else if let Some((entry_col, exit_col, event_col)) = parse_surv_response(&parsed.response)? {
        if effective_config.transformation_normal {
            return Err(WorkflowError::TransformationNormalConflict {
                conflict: TransformationNormalConflict::SurvResponse,
            });
        }
        refuse_unread_firth(effective_config, "survival models")?;
        if !effective_config.residual_columns.is_empty() {
            return Err(WorkflowError::InvalidConfig {
                reason: "residual_columns is a Bernoulli marginal-slope block (gam#2924); the \
                         survival marginal-slope family takes it once gam#2923 lands"
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
            structural_only,
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
        reject_survival_only_config_for_nonsurvival(effective_config)?;
        if effective_config.transformation_normal {
            // Issue #789A: a Bernoulli marginal-slope request with
            // `transformation_normal=true` used to dispatch as a CTN fit while
            // retaining marginal-slope controls, leaving the transformation path
            // in a non-advancing loop. CTN score calibration now uses the
            // explicit `ctn_stage1` recipe instead, so the legacy boolean is a
            // hard configuration error for marginal-slope requests.
            reject_marginal_slope_controls_for_transformation_normal(effective_config)?;
            if effective_config.noise_formula.is_some() {
                return Err(WorkflowError::TransformationNormalConflict {
                    conflict: TransformationNormalConflict::NoiseFormula,
                });
            }
            refuse_unread_firth(effective_config, "the transformation-normal family")?;
            // The transformation-normal fit has its own likelihood and reads no
            // other family, so `transformation_normal=true` beside another family
            // is a conflict, not a family to drop.
            if let Some(family) = effective_config.family.as_deref()
                && !family_requests_transformation_normal(Some(family))
            {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "transformation_normal conflicts with family `{family}`; the \
                         transformation-normal fit reads no other family"
                    ),
                });
            }
            materialize_transformation_normal(&parsed, data, &col_map, effective_config)
        } else if requests_bernoulli_marginal_slope(effective_config) {
            materialize_bernoulli_marginal_slope(&parsed, data, &col_map, effective_config)
        } else if effective_config.noise_formula.is_some() {
            refuse_unread_firth(effective_config, "noise_formula location-scale fits")?;
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
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        /// Uniform in [0, 1).
        fn unif(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Standard normal via Box–Muller (one of the pair).
        fn normal(&mut self) -> f64 {
            // `unif` is in [0, 1); its complement is in (0, 1], so `ln u1` is finite.
            let u1 = 1.0 - self.unif();
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
        FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        }
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
        let resid: Vec<f64> = y
            .iter()
            .zip(fitted.iter())
            .map(|(&yi, &fi)| yi - fi)
            .collect();
        let mean = resid.iter().sum::<f64>() / resid.len() as f64;
        let var = resid.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / resid.len() as f64;
        var.sqrt()
    }

    fn fit_standard(formula: &str, data: &Dataset) -> StandardFitResult {
        match fit_from_formula(formula, data, &gaussian_config())
            .unwrap_or_else(|e| panic!("fit `{formula}` failed: {e:?}"))
        {
            FitResult::Standard(r) => r,
            other => panic!(
                "expected Standard fit for `{formula}`, got a different variant: {}",
                std::any::type_name_of_val(&other)
            ),
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

/// Formula-level library entry for the O(n log n) residual-cascade fast path
/// (issue #1032).
///
/// Materializes the formula exactly like [`fit_from_formula`], runs the
/// [`residual_cascade_fast_path`] detection, and — when it fires and the
/// cascade supplies every required proof — returns the
/// certified [`ResidualCascadeFit`](gam_solve::residual_cascade::ResidualCascadeFit).
/// `Ok(None)` means only that the model is not the cascade-eligible shape.
/// Once the route is selected, a proof/convergence failure is returned rather
/// than silently changing estimators.
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
    gam_runtime::parallel::install(|| {
        gam_solve::residual_cascade::fit_residual_cascade(
            &coord_refs,
            &inputs.y,
            &inputs.w,
            &inputs.metric,
            inputs.sobolev_s,
        )
    })
    .map(Some)
    .map_err(residual_cascade_failure)
}

/// Formula-level direct entry for the exact O(n) smoothing-spline scan.
///
/// Materializes the formula exactly like [`fit_from_formula`], then runs the
/// [`spline_scan_fast_path`] detection on the resulting standard request.
/// This public entry point is for library callers that specifically need the
/// specialized [`gam_solve::spline_scan::SplineScanFit`] rather than the
/// [`FitResult::SplineScan`] sum-type returned by the canonical workflow. When
/// detection fires the fit is routed through
/// [`gam_solve::spline_scan::fit_spline_scan`] — the exact diffuse
/// REML Kalman/RTS scan — and the full in-memory posterior
/// ([`gam_solve::spline_scan::SplineScanFit`]: knots, smoothed
/// states, pointwise variances, lag-one gains, σ², log λ, exact EDF, and an
/// exact `predict`) is returned. `Ok(None)` means the model is not the
/// scan-eligible shape; the direct caller then chooses another estimator.
/// Persistence-bearing workflows do not call this probe: [`fit_from_formula`]
/// returns [`FitResult::SplineScan`], and the shared
/// [`crate::inference::model_payload_builders::assemble_spline_scan_payload`]
/// authority writes the exact scan state for both CLI and FFI consumers.
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
    gam_runtime::parallel::install(|| {
        gam_solve::spline_scan::fit_spline_scan(&inputs.x, &inputs.y, &inputs.w, inputs.order)
    })
    .map(Some)
    .map_err(spline_scan_failure)
}

#[cfg(test)]
mod joint_expectile_scale_posterior_tests {
    use super::*;

    const LEVELS: [f64; 3] = [0.1, 0.5, 0.9];

    /// Heteroscedastic `y = sin(3x) + (0.3 + 0.6x)·ε`, `ε ~ N(0, 1)` from a
    /// fixed LCG with Box–Muller, so the fixture is reproducible.
    fn heteroscedastic_dataset(n: usize) -> Dataset {
        let mut state: u64 = 0x3056_2026_0919_0001;
        let mut unif = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let rows = (0..n)
            .map(|i| {
                let x = i as f64 / (n as f64 - 1.0);
                let u1 = 1.0 - unif();
                let u2 = unif();
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                let y = (3.0 * x).sin() + (0.3 + 0.6 * x) * z;
                csv::StringRecord::from(vec![x.to_string(), y.to_string()])
            })
            .collect();
        let headers = ["x", "y"].into_iter().map(String::from).collect();
        gam_data::encode_recordswith_inferred_schema(headers, rows).expect("encode fixture")
    }

    /// `c_τ` standardizes each residual by the posterior mean of σ, which needs
    /// the Scale block of the joint covariance. The fitted `c_τ` is exactly the
    /// covariance-integrated value, and the same fit with its covariance
    /// removed is refused with a typed error — never standardized by the
    /// plug-in σ as if the log-σ posterior variance were zero (#3056).
    #[test]
    fn joint_expectile_c_tau_requires_the_scale_block_posterior() {
        let n = 200;
        let data = heteroscedastic_dataset(n);
        let config = FitConfig {
            family: Some("expectile".to_string()),
            expectile_tau: Some(LEVELS.to_vec()),
            ..FitConfig::default()
        };
        let mut result = fit_expectile_location_scale(
            "y ~ s(x)",
            &data,
            &config,
            LEVELS.to_vec(),
            &mut ExpectileMaterializeNotes::default(),
        )
        .expect("joint expectile fit");
        let y_index = data
            .headers
            .iter()
            .position(|h| h == "y")
            .expect("response column");
        let y = data.values.column(y_index).to_owned();
        let ones = Array1::<f64>::ones(n);
        let zeros = Array1::<f64>::zeros(n);
        let c_tau = |location_scale: &GaussianLocationScaleFitResult| {
            joint_expectile_standardized_expectiles(
                location_scale,
                y.view(),
                ones.view(),
                zeros.view(),
                zeros.view(),
                &LEVELS,
            )
        };

        assert!(
            result.location_scale.fit.fit.beta_covariance().is_some(),
            "a joint expectile fit carries its joint posterior covariance"
        );
        let integrated = c_tau(&result.location_scale).expect("c_τ with covariance");
        assert_eq!(integrated, result.standardized_expectiles);

        result.location_scale.fit.fit.covariance_conditional = None;
        match c_tau(&result.location_scale) {
            Err(error) => {
                let message = error.to_string();
                assert!(
                    message.contains("joint posterior covariance"),
                    "refusal must name the missing covariance: {message}"
                );
            }
            Ok(plug_in) => panic!(
                "c_τ without the scale-block covariance must be refused, got the plug-in \
                 {plug_in:?} (integrated {integrated:?})"
            ),
        }
    }
}

#[cfg(test)]
mod expectile_front_end_tests {
    use super::*;

    const CR_CAP: &str = "cubic-regression ('cr'/'sz') basis reduced from k=8 to k=5";

    /// `y = sin(3x) + (0.3 + 0.6x)·ε` on a covariate with 5 distinct values,
    /// from a fixed LCG with Box–Muller, so `s(x, bs='cr', k=8)` is capped.
    fn five_level_dataset(n: usize) -> Dataset {
        let mut state: u64 = 0x1543_2026_0920_0007;
        let mut unif = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let rows = (0..n)
            .map(|i| {
                let x = (i % 5) as f64 / 4.0;
                let u1 = 1.0 - unif();
                let u2 = unif();
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                let y = (3.0 * x).sin() + (0.3 + 0.6 * x) * z;
                csv::StringRecord::from(vec![x.to_string(), y.to_string()])
            })
            .collect();
        let headers = ["x", "y"].into_iter().map(String::from).collect();
        gam_data::encode_recordswith_inferred_schema(headers, rows).expect("encode fixture")
    }

    fn expectile_config(levels: &[f64]) -> FitConfig {
        FitConfig {
            family: Some("expectile".to_string()),
            expectile_tau: Some(levels.to_vec()),
            ..FitConfig::default()
        }
    }

    fn assert_cr_cap_reported(advisories: &[String], route: &str) {
        assert!(
            advisories.iter().any(|note| note.contains(CR_CAP)),
            "{route}: the inner materialization's k cap must reach the caller, got {advisories:?}"
        );
    }

    /// The expectile drivers materialize their own inner Gaussian design, and
    /// what that materialization reports describes the fitted model. A capped
    /// `k` must reach the library caller and the saved payload alike, as it
    /// does for every other family (#1543).
    #[test]
    fn expectile_fits_report_the_inner_materialization_advisories() {
        let data = five_level_dataset(200);
        let formula = "y ~ s(x, bs='cr', k=8)";

        let single = expectile_config(&[0.5]);
        let library = fit_from_formula_with_notes(formula, &data, &single)
            .expect("single-level expectile fit");
        assert_cr_cap_reported(&library.inference_notes.advisories, "library, one level");
        let payload = crate::inference::model_payload_builders::fit_formula_to_payload(
            formula.to_string(),
            &data,
            &single,
        )
        .expect("single-level expectile payload");
        assert_cr_cap_reported(&payload.inference_notes, "payload, one level");

        let joint = expectile_config(&[0.25, 0.75]);
        let library = fit_from_formula_with_notes(formula, &data, &joint)
            .expect("joint expectile fit");
        assert_cr_cap_reported(&library.inference_notes.advisories, "library, two levels");
        let payload = crate::inference::model_payload_builders::fit_formula_to_payload(
            formula.to_string(),
            &data,
            &joint,
        )
        .expect("joint expectile payload");
        assert_cr_cap_reported(&payload.inference_notes, "payload, two levels");
    }

    /// `z = sin(3x₁)·cos(2x₂) + 0.3ε` on a scattered 2-D design, so the
    /// default `s(x1, x2)` is a multivariate radial smooth whose adaptive start
    /// is smaller than its fully provisioned basis.
    fn scattered_surface_dataset(n: usize) -> Dataset {
        let mut state: u64 = 0x4062_2026_0920_0011;
        let mut unif = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let rows = (0..n)
            .map(|_| {
                let x1 = unif();
                let x2 = unif();
                let u1 = 1.0 - unif();
                let u2 = unif();
                let e = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                let z = (3.0 * x1).sin() * (2.0 * x2).cos() + 0.3 * e;
                csv::StringRecord::from(vec![x1.to_string(), x2.to_string(), z.to_string()])
            })
            .collect();
        let headers = ["x1", "x2", "z"].into_iter().map(String::from).collect();
        gam_data::encode_recordswith_inferred_schema(headers, rows).expect("encode fixture")
    }

    /// A one-level expectile fit is a standard fit, so the library and the
    /// saved payload must reach it through the same adaptive-resolution loop
    /// and land on the same basis and coefficients (#4062). The payload route
    /// used to fit the fully provisioned radial basis instead.
    #[test]
    fn single_level_expectile_payload_matches_the_library_fit() {
        let data = scattered_surface_dataset(200);
        let formula = "z ~ s(x1, x2)";
        let config = expectile_config(&[0.5]);
        let library = fit_from_formula_with_notes(formula, &data, &config)
            .expect("library expectile fit");
        let FitResult::Standard(library) = library.result else {
            panic!("a one-level expectile fit is a standard fit");
        };
        let payload = crate::inference::model_payload_builders::fit_formula_to_payload(
            formula.to_string(),
            &data,
            &config,
        )
        .expect("expectile payload");
        let saved = payload
            .fit_result
            .as_ref()
            .expect("standard expectile payload carries its fit result");
        assert_eq!(
            saved.beta.len(),
            library.fit.beta.len(),
            "Python / CLI --out and the library must fit the same expectile basis"
        );
        for (saved_coef, library_coef) in saved.beta.iter().zip(library.fit.beta.iter()) {
            assert!(
                (saved_coef - library_coef).abs() <= 1e-4 * (1.0 + library_coef.abs()),
                "coefficients differ between front ends: {saved_coef} vs {library_coef}"
            );
        }
    }
}

#[cfg(test)]
mod unread_firth_and_family_refusal_tests {
    //! A setting the selected fit does not read is refused by the library, the
    //! one place every front end reaches, rather than dropped for a fit without it.
    use super::*;
    use gam_data::{ColumnKindTag, DataSchema, SchemaColumn};
    use ndarray::Array2;

    /// `t` a positive exit time, `e` a {0,1} event, `y` a continuous response,
    /// `x` a covariate.
    fn dataset() -> Dataset {
        let names = ["t", "e", "y", "x"];
        let kinds = [
            ColumnKindTag::Continuous,
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ];
        let t = [1.2, 2.5, 0.8, 3.1, 1.9, 2.2, 4.0, 0.6, 2.8, 1.4];
        let e = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let y = [0.3, -0.2, 1.1, 0.7, -0.5, 0.2, 1.4, -0.9, 0.5, 0.0];
        let x = [-1.0, -0.7, -0.4, -0.2, 0.0, 0.1, 0.3, 0.5, 0.8, 1.0];
        let values = Array2::from_shape_fn((t.len(), 4), |(i, j)| [t[i], e[i], y[i], x[i]][j]);
        Dataset {
            headers: names.iter().map(|n| n.to_string()).collect(),
            values,
            schema: DataSchema {
                columns: names
                    .iter()
                    .zip(kinds)
                    .map(|(name, kind)| SchemaColumn {
                        name: name.to_string(),
                        kind,
                        levels: vec![],
                    })
                    .collect(),
            },
            column_kinds: kinds.to_vec(),
        }
    }

    fn refusal(formula: &str, config: FitConfig) -> String {
        match materialize(formula, &dataset(), &config) {
            Ok(_) => panic!("`{formula}` with {config:?} must be refused"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn firth_is_refused_where_the_fit_reads_no_firth_setting() {
        let cases = [
            (
                "y ~ x",
                FitConfig {
                    transformation_normal: true,
                    firth: true,
                    ..FitConfig::default()
                },
                "firth is not supported for the transformation-normal family",
            ),
            (
                "y ~ x",
                FitConfig {
                    family: Some("transformation-normal".to_string()),
                    firth: true,
                    ..FitConfig::default()
                },
                "firth is not supported for the transformation-normal family",
            ),
            (
                "y ~ x",
                FitConfig {
                    family: Some("gaussian".to_string()),
                    noise_formula: Some("1".to_string()),
                    firth: true,
                    ..FitConfig::default()
                },
                "firth is not supported for noise_formula location-scale fits",
            ),
            (
                "Surv(t, e) ~ x",
                FitConfig {
                    firth: true,
                    ..FitConfig::default()
                },
                "firth is not supported for survival models",
            ),
        ];
        for (formula, config, expected) in cases {
            let message = refusal(formula, config);
            assert!(
                message.contains(expected),
                "`{formula}`: expected `{expected}`, got: {message}"
            );
        }
    }

    #[test]
    fn transformation_normal_beside_another_family_is_refused() {
        let message = refusal(
            "y ~ x",
            FitConfig {
                transformation_normal: true,
                family: Some("poisson".to_string()),
                ..FitConfig::default()
            },
        );
        assert!(
            message.contains("transformation_normal conflicts with family `poisson`"),
            "got: {message}"
        );
    }

    #[test]
    fn the_routes_that_read_their_settings_still_materialize() {
        let data = dataset();
        let tn = materialize(
            "y ~ x",
            &data,
            &FitConfig {
                transformation_normal: true,
                family: Some("transformation_normal".to_string()),
                ..FitConfig::default()
            },
        )
        .expect("transformation_normal=true with its own family name");
        assert!(matches!(tn.request, FitRequest::TransformationNormal(_)));
        let firth = materialize(
            "e ~ x",
            &data,
            &FitConfig {
                family: Some("binomial".to_string()),
                firth: true,
                ..FitConfig::default()
            },
        )
        .expect("the standard binomial route reads firth");
        assert!(matches!(firth.request, FitRequest::Standard(_)));
    }
}

#[cfg(test)]
mod residual_cascade_geometry_tests {
    use super::*;

    /// Deterministic scattered 2-D sample on the unit square.
    fn scattered_2d(n: usize) -> Dataset {
        let golden = 0.618_033_988_749_894_9_f64;
        let root2 = std::f64::consts::SQRT_2.fract();
        let rows = (0..n)
            .map(|i| {
                let a = ((i + 1) as f64 * golden).fract();
                let b = ((i + 1) as f64 * root2).fract();
                let u = ((i + 3) as f64 * golden).fract();
                let y = (std::f64::consts::TAU * a).sin() * (std::f64::consts::TAU * b).cos()
                    + 0.1 * (u - 0.5);
                csv::StringRecord::from(vec![a.to_string(), b.to_string(), y.to_string()])
            })
            .collect();
        let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
        gam_data::encode_recordswith_inferred_schema(headers, rows).expect("encode fixture")
    }

    fn gaussian(scale_dimensions: bool) -> FitConfig {
        FitConfig {
            family: Some("gaussian".to_string()),
            scale_dimensions,
            ..FitConfig::default()
        }
    }

    fn standard_request<'a>(
        formula: &str,
        data: &'a Dataset,
        config: &FitConfig,
    ) -> StandardFitRequest<'a> {
        let mat = materialize(formula, data, config).expect("materialize");
        let FitRequest::Standard(request) = mat.request else {
            panic!("`{formula}` must materialize as a standard request");
        };
        request
    }

    /// The cascade fits open coordinates under a unit metric. A term that asks
    /// for learned per-axis length scales must stay on the dense radial path,
    /// whether it asks per term or through the global flag; the isotropic
    /// control on the same data stays eligible.
    #[test]
    fn anisotropic_radial_smooth_is_not_a_cascade_candidate() {
        let data = scattered_2d(600);
        for formula in ["y ~ duchon(x1, x2)", "y ~ matern(x1, x2)"] {
            let control = standard_request(formula, &data, &gaussian(false));
            assert!(
                residual_cascade_structural_signature(&control).is_some(),
                "`{formula}` (isotropic) is the eligible control"
            );
            let global = standard_request(formula, &data, &gaussian(true));
            assert!(
                residual_cascade_structural_signature(&global).is_none(),
                "`{formula}` with scale_dimensions must not take the unit-metric cascade"
            );
        }
        let per_term = standard_request(
            "y ~ duchon(x1, x2, scale_dims=true)",
            &data,
            &gaussian(false),
        );
        assert!(
            residual_cascade_structural_signature(&per_term).is_none(),
            "duchon(scale_dims=true) must not take the unit-metric cascade"
        );
    }

    /// A wrapped axis has no representation in the cascade's open Euclidean
    /// frame, so a periodic radial spec must not be structurally eligible.
    #[test]
    fn periodic_radial_smooth_is_not_a_cascade_candidate() {
        let data = scattered_2d(600);
        for formula in ["y ~ duchon(x1, x2)", "y ~ matern(x1, x2)"] {
            let mut request = standard_request(formula, &data, &gaussian(false));
            assert!(residual_cascade_structural_signature(&request).is_some());
            match &mut request.spec.smooth_terms[0].basis {
                gam_terms::smooth::SmoothBasisSpec::Duchon { spec, .. } => {
                    spec.periodic = Some(vec![Some(1.0), None]);
                }
                gam_terms::smooth::SmoothBasisSpec::Matern { spec, .. } => {
                    spec.periodic = Some(vec![Some(1.0), None]);
                }
                _ => panic!("`{formula}` must materialize a radial basis"),
            }
            assert!(
                residual_cascade_structural_signature(&request).is_none(),
                "`{formula}` with a periodic axis must not take the open-domain cascade"
            );
        }
    }
}

#[cfg(test)]
mod spline_scan_identifiability_routing_tests {
    use super::*;
    use csv::StringRecord;
    use gam_data::encode_recordswith_inferred_schema;

    fn trend_data() -> Dataset {
        let headers: Vec<String> = ["x", "y"].iter().map(|h| h.to_string()).collect();
        let rows = (0..60)
            .map(|i| {
                let x = i as f64 / 59.0;
                let y = 2.0 * x + 0.3 * (7.0 * x).sin() + 0.05 * ((i * 37 % 11) as f64 - 5.0);
                StringRecord::from(vec![x.to_string(), y.to_string()])
            })
            .collect();
        encode_recordswith_inferred_schema(headers, rows).expect("encode")
    }

    fn routes_to_scan(identifiability: &str) -> bool {
        let data = trend_data();
        let config = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let formula = format!(
            "y ~ s(x, bs=\"ps\", degree=3, penalty_order=2, double_penalty=False{identifiability})"
        );
        match materialize(&formula, &data, &config)
            .expect("materialize")
            .request
        {
            FitRequest::Standard(request) => spline_scan_fast_path(&request).is_some(),
            _ => panic!("a Gaussian s(x) formula materializes a standard request"),
        }
    }

    /// Sum-to-zero centring spans `{1, x} ⊕ wiggle` with the intercept, which
    /// is exactly the scan's model.
    #[test]
    fn centred_smooths_keep_the_scan() {
        assert!(routes_to_scan(""));
        assert!(routes_to_scan(", identifiability=\"sum_tozero\""));
    }

    /// `identifiability="linear"` removes the `x` direction, which the scan's
    /// unpenalized null space would fit back in (#3870).
    #[test]
    fn linear_trend_removal_falls_through_to_the_dense_path() {
        assert!(!routes_to_scan(", identifiability=\"linear\""));
    }
}
