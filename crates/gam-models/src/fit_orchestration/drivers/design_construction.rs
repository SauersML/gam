// #1521: `build_term_collection_design` and its term-design subgraph were
// relocated DOWN into `gam_terms::smooth` (see `gam_terms::smooth::term_design`).
// The joint-build variants below STAY here: they return a `gam_solve`
// `EstimationError` / call `freeze_term_collection_from_design`
// (`spatial_optimization.rs`), so they belong to the gam-models orchestration
// tier. They reach the relocated `build_term_collection_design_with_policy` /
// `build_term_collection_design` via the module's `use gam_terms::smooth::*`.
pub fn build_term_collection_designs_joint(
    data: ArrayView2<'_, f64>,
    specs: &[TermCollectionSpec],
) -> Result<Vec<TermCollectionDesign>, BasisError> {
    for spec in specs {
        validate_term_collection_finite_inputs(data, spec)?;
    }
    let smooth_blocks = specs
        .iter()
        .map(|spec| spec.smooth_terms.clone())
        .collect::<Vec<_>>();
    let planned_blocks = plan_joint_spatial_centers_for_term_blocks(data, &smooth_blocks)?;
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let mut out = Vec::with_capacity(specs.len());
    for (spec, planned_terms) in specs.iter().zip(planned_blocks.into_iter()) {
        let mut planned_spec = spec.clone();
        planned_spec.smooth_terms = planned_terms;
        out.push(build_planned_term_collection_design_inner_with_policy(
            data,
            &planned_spec,
            &policy,
        )?);
    }
    Ok(out)
}

pub fn build_term_collection_designs_and_freeze_joint(
    data: ArrayView2<'_, f64>,
    specs: &[TermCollectionSpec],
) -> Result<(Vec<TermCollectionDesign>, Vec<TermCollectionSpec>), EstimationError> {
    let designs = build_term_collection_designs_joint(data, specs)?;
    let mut resolved_specs = Vec::with_capacity(specs.len());
    for (spec, design) in specs.iter().zip(designs.iter()) {
        resolved_specs.push(freeze_term_collection_from_design(spec, design)?);
    }
    Ok((designs, resolved_specs))
}

pub fn fit_term_collection_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    fit_term_collection_forspecwith_heuristic_log_lambdas(
        data, y, weights, offset, spec, None, family, options,
    )
}

pub fn fit_term_collection_with_coefficient_groups(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    groups: &[CoefficientGroupSpec],
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    if groups.is_empty() {
        return fit_term_collection_forspec(data, y, weights, offset, spec, family, options);
    }
    let design = build_term_collection_design_with_policy(data, spec, &options.resource_policy)?;
    let base_fit_opts = adaptive_fit_options_base(options, &design);
    let realized = design
        .realize_coefficient_groups(groups, &base_fit_opts.rho_prior)
        .map_err(EstimationError::BasisError)?;
    let effective_offset = design
        .compose_offset(offset, "coefficient-group fit")
        .map_err(EstimationError::BasisError)?;
    let mut grouped_options = base_fit_opts.clone();
    grouped_options.rho_prior = realized.rho_prior;
    let fitted = FittedTermCollection {
        fit: gam_solve::estimate::fit_gam_with_penalty_specs(
            design.design.clone(),
            y,
            weights,
            effective_offset.view(),
            realized.penalty_specs,
            realized.nullspace_dims,
            family.clone(),
            &grouped_options,
        )?,
        design,
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;
    Ok(fitted)
}

pub fn fit_term_collection_with_penalty_block_gamma_priors(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    priors: &[(String, f64, f64)],
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let design = build_term_collection_design_with_policy(data, spec, &options.resource_policy)?;
    let effective_offset = design
        .compose_offset(offset, "penalty-prior fit")
        .map_err(EstimationError::BasisError)?;
    let mut fit_opts = adaptive_fit_options_base(options, &design);
    fit_opts.rho_prior = realize_keyed_penalty_block_gamma_priors(&design, priors)
        .map_err(EstimationError::BasisError)?;
    let fitted = FittedTermCollection {
        fit: fit_gamwith_heuristic_log_lambdas(
            design.design.clone(),
            y,
            weights,
            effective_offset.view(),
            &design.penalties,
            None,
            family.clone(),
            &fit_opts,
        )?,
        design,
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;
    Ok(fitted)
}

pub fn fit_term_collection_with_coefficient_groups_and_penalty_block_gamma_priors(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    groups: &[CoefficientGroupSpec],
    priors: &[(String, f64, f64)],
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    if groups.is_empty() {
        return fit_term_collection_with_penalty_block_gamma_priors(
            data, y, weights, offset, spec, priors, family, options,
        );
    }
    if priors.is_empty() {
        return fit_term_collection_with_coefficient_groups(
            data, y, weights, offset, spec, groups, family, options,
        );
    }

    // The base design already emits one term-named function-space ridge per
    // recoverable linear effect, so keyed priors and coefficient groups address
    // the same authoritative λ coordinates as every other fit path.
    let design = build_term_collection_design_with_policy(data, spec, &options.resource_policy)?;
    let base_fit_opts = adaptive_fit_options_base(options, &design);
    let base_rho_prior = realize_keyed_penalty_block_gamma_priors(&design, priors)
        .map_err(EstimationError::BasisError)?;
    let realized = design
        .realize_coefficient_groups(groups, &base_rho_prior)
        .map_err(EstimationError::BasisError)?;
    let effective_offset = design
        .compose_offset(offset, "coefficient-group and penalty-prior fit")
        .map_err(EstimationError::BasisError)?;
    let mut grouped_options = base_fit_opts.clone();
    grouped_options.rho_prior = realized.rho_prior;
    let fitted = FittedTermCollection {
        fit: gam_solve::estimate::fit_gam_with_penalty_specs(
            design.design.clone(),
            y,
            weights,
            effective_offset.view(),
            realized.penalty_specs,
            realized.nullspace_dims,
            family.clone(),
            &grouped_options,
        )?,
        design,
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;
    Ok(fitted)
}

fn fit_term_collection_forspecwith_heuristic_log_lambdas(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    heuristic_log_lambdas: Option<&[f64]>,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let base_design =
        build_term_collection_design_with_policy(data, spec, &options.resource_policy)?;
    fit_term_collection_on_realized_design(
        y,
        weights,
        offset,
        spec,
        &base_design,
        heuristic_log_lambdas,
        family,
        options,
    )
}

/// Refit `spec` starting from a converged fit's own smoothing coordinates.
///
/// The seed is `fitted.log_lambdas`, the coordinate the outer search reads, and
/// never the physical `lambdas` derived from it (#2902 row 9).
fn refit_term_collection_at_fitted_strengths(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    fitted: &UnifiedFitResult,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    fit_term_collection_forspecwith_heuristic_log_lambdas(
        data,
        y,
        weights,
        offset,
        spec,
        fitted.log_lambdas.as_slice(),
        family,
        options,
    )
}

fn fit_term_collection_on_realized_design(
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    heuristic_log_lambdas: Option<&[f64]>,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let effective_offset = design
        .compose_offset(offset, "term-collection fit")
        .map_err(EstimationError::BasisError)?;
    let offset = effective_offset.view();
    if spec.has_bounded_linear_terms() {
        return fit_bounded_term_collection_with_design(
            y,
            weights,
            offset,
            spec,
            design,
            heuristic_log_lambdas,
            family,
            options,
        );
    }
    let base_fit_opts = adaptive_fit_options_base(options, design);
    let fitted = FittedTermCollection {
        fit: fit_gamwith_heuristic_log_lambdas(
            design.design.clone(),
            y,
            weights,
            offset,
            &design.penalties,
            heuristic_log_lambdas,
            family.clone(),
            &base_fit_opts,
        )?,
        design: design.clone(),
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;
    Ok(fitted)
}

fn checked_fit_log_lambdas(
    lambdas: &Array1<f64>,
    context: &str,
) -> Result<Array1<f64>, EstimationError> {
    let values = lambdas
        .iter()
        .copied()
        .enumerate()
        .map(|(coordinate, lambda)| {
            gam_problem::checked_log_strength(lambda).map_err(|error| {
                EstimationError::InvalidInput(format!(
                    "{context} lambda coordinate {coordinate} is outside the canonical physical-strength domain: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Array1::from_vec(values))
}

fn adaptive_fit_options_base(options: &FitOptions, design: &TermCollectionDesign) -> FitOptions {
    FitOptions {
        resource_policy: options.resource_policy.clone(),
        latent_cloglog: options.latent_cloglog,
        mixture_link: options.mixture_link.clone(),
        optimize_mixture: options.optimize_mixture,
        sas_link: options.sas_link,
        optimize_sas: options.optimize_sas,
        compute_inference: options.compute_inference,
        skip_rho_posterior_inference: options.skip_rho_posterior_inference,
        max_iter: options.max_iter,
        tol: options.tol,
        nullspace_dims: design.nullspace_dims.clone(),
        linear_constraints: design.linear_constraints.clone(),
        firth_bias_reduction: options.firth_bias_reduction,
        // Propagate user-supplied rho_prior so the baseline/refit and the
        // joint optimizer minimize the same REML objective.
        rho_prior: options.rho_prior.clone(),
        persistent_warm_start_store: options.persistent_warm_start_store.clone(),
    }
}

fn superseded_fit_options(options: &FitOptions) -> FitOptions {
    let mut fit_options = options.clone();
    fit_options.skip_rho_posterior_inference = true;
    fit_options
}

#[derive(Clone)]
struct BoundedLinearTermMeta {
    col_idx: usize,
    min: f64,
    max: f64,
    prior: BoundedCoefficientPriorSpec,
}

/// β-dependent effective Jacobian for the bounded-linear fit block.
///
/// Each bounded coefficient enters the linear predictor non-linearly, as
/// `β = min + width·σ(θ)`, and is supplied to the solver through the family
/// adapter's offset rather than the linear design. To keep that contribution
/// out of the *linear* design the fit places a deliberately **zeroed**
/// placeholder column for every bounded term in the block design
/// (see `fit_bounded_term_collection_with_design`). The pre-fit
/// identifiability audit, however, assesses block rank by reading each block's
/// effective Jacobian — and a zeroed column reads as a structural rank
/// deficiency, so without this callback the audit refuses *every* bounded
/// model before fitting begins.
///
/// This callback reports the model's true Jacobian column for each bounded
/// term, `∂η_i/∂θ = (dβ/dθ)·x_i`, so the audit inspects the same geometry the
/// solver actually fits. Because `dβ/dθ = width·σ(θ)(1−σ(θ))` is strictly
/// positive for finite θ and `width > 0`, a bounded column is rank-deficient
/// in the audit exactly when its underlying covariate is genuinely collinear
/// with the rest of the design — never merely because the placeholder was
/// zeroed. The callback is consumed only by the identifiability audit /
/// canonicalisation; the inner PIRLS solve drives η through the
/// [`BoundedLinearFamily`] adapter, so reporting the non-zeroed Jacobian here
/// does not double-count the bounded contribution.
struct BoundedEffectiveJacobian {
    design: Array2<f64>,
    bounded_terms: Vec<BoundedLinearTermMeta>,
}

impl BlockEffectiveJacobian for BoundedEffectiveJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let p = self.design.ncols();
        let n = self.design.nrows();
        let rows = rows.start.min(n)..rows.end.min(n);
        if !state.beta.is_empty() {
            if state.beta.len() != p {
                return Err(format!(
                    "BoundedEffectiveJacobian::effective_jacobian_at: beta length {} != design \
                     ncols {p}",
                    state.beta.len(),
                ));
            }
            if state.beta.iter().any(|v| !v.is_finite()) {
                return Err(
                    "BoundedEffectiveJacobian::effective_jacobian_at: beta contains a non-finite value"
                        .to_string(),
                );
            }
        }
        let mut jac = self
            .design
            .slice(ndarray::s![rows.start..rows.end, ..])
            .to_owned();
        for term in &self.bounded_terms {
            if term.col_idx >= p {
                return Err(format!(
                    "BoundedEffectiveJacobian::effective_jacobian_at: bounded column {} is outside {p} columns",
                    term.col_idx
                ));
            }
            let theta = if state.beta.is_empty() {
                0.0
            } else {
                state.beta[term.col_idx]
            };
            let (_, _, db_dtheta, _, _, _, _) = bounded_latent_derivatives(theta, term.min, term.max);
            if !(db_dtheta.is_finite() && db_dtheta > 0.0) {
                return Err(format!(
                    "BoundedEffectiveJacobian::effective_jacobian_at: bounded column {} has unrepresentable derivative {db_dtheta} at theta={theta}",
                    term.col_idx
                ));
            }
            jac.column_mut(term.col_idx).mapv_inplace(|v| v * db_dtheta);
        }
        Ok(jac)
    }
}

#[derive(Clone)]
struct BoundedLinearFamily {
    likelihood: gam_spec::GlmLikelihoodSpec,
    latent_cloglog_state: Option<LatentCLogLogState>,
    mixture_link_state: Option<MixtureLinkState>,
    sas_link_state: Option<SasLinkState>,
    y: Array1<f64>,
    weights: Array1<f64>,
    design: Array2<f64>,
    designzeroed: Array2<f64>,
    offset: Array1<f64>,
    bounded_terms: Vec<BoundedLinearTermMeta>,
    /// Whether this member's Jeffreys/Firth prior is armed. A fit arms it only
    /// on the unarmed fit's own evidence, through
    /// `fit_custom_family_arming_on_evidence` (#979).
    jeffreys_armed: bool,
}

#[derive(Clone, Debug)]
struct StandardFamilyObservationState {
    eta: Array1<f64>,
    score: Array1<f64>,
    fisherweight: Array1<f64>,
    neghessian_eta: Array1<f64>,
    neghessian_eta_derivative: Array1<f64>,
    neghessian_eta_second_derivative: Array1<f64>,
    /// `W'''` on every row (#2903).
    neghessian_eta_third_derivative: Array1<f64>,
    log_likelihood: f64,
}

fn bounded_latent_to_user(theta: f64, min: f64, max: f64) -> (f64, f64, f64) {
    let jet = logit_inverse_link_jet5(theta);
    let z = jet.mu;
    let width = max - min;
    let beta = min + width * z;
    let db_dtheta = width * jet.d1;
    (beta, z, db_dtheta)
}

/// Invert the bounded interval transform: given a user-scale coefficient
/// `beta` in the open interval `(min, max)`, return the latent coordinate
/// `theta` with `bounded_latent_to_user(theta, min, max).0 == beta`.
///
/// This is the exact inverse of the logistic interval map used by the bounded
/// custom family.  The log-gap identity avoids first forming a normalized
/// position that can underflow or round to one:
/// `theta = log(beta - min) - log(max - beta)`.
fn bounded_user_to_latent(beta: f64, min: f64, max: f64) -> f64 {
    (beta - min).ln() - (max - beta).ln()
}

/// One bounded coefficient column for posterior sampling: its position in the
/// (internal, conditioned) coefficient vector and the interval bounds expressed
/// on that same internal scale.
#[derive(Debug, Clone, Copy)]
pub struct BoundedSampleColumn {
    /// Column index into the internal (conditioned) coefficient vector.
    pub col_idx: usize,
    /// Lower interval bound on the internal scale.
    pub min: f64,
    /// Upper interval bound on the internal scale.
    pub max: f64,
}

/// Exact posterior draws for a model with `bounded()` coefficients.
///
/// The bounded custom family fits each bounded coefficient as a smooth interval
/// transform `beta = min + (max - min)·sigmoid(theta)` of an unconstrained
/// latent `theta`. The Laplace approximation is *Gaussian on the latent scale*
/// — that is precisely the scale on which the fit treats the coefficient as an
/// unconstrained, locally-quadratic parameter. Sampling a Gaussian directly on
/// the user (bounded) scale is wrong twice over: it can place mass outside
/// `[min, max]`, and it discards the boundary-induced skew that the nonlinear
/// map produces. This routine instead draws `theta ~ N(theta_mode, H_latent^{-1})`
/// and pushes every draw through the *exact* interval map, so user-scale draws
/// always lie strictly inside the interval and carry the correct skew.
///
/// Coordinate bookkeeping. The caller supplies the user-scale mode `beta_user`
/// and the user-scale penalized Hessian `user_hessian` (both in *internal /
/// conditioned* coordinates — i.e. before `backtransform_*` to the original
/// data scale) together with the internal-scale bounds for each bounded column.
/// The user-scale Hessian relates to the latent-scale Hessian by the diagonal
/// delta-method Jacobian `J = diag(db/dtheta)`:
///   `H_user = J^{-1} H_latent J^{-1}`  ⇒  `H_latent = J H_user J`,
/// which is exactly the inverse of `transform_bounded_latent_precision_to_user_internal`.
/// Non-bounded columns have `J_ii = 1`, so they are sampled as the ordinary
/// Gaussian Laplace draw and returned unchanged.
///
/// Dispersion. `user_hessian` is the UNSCALED penalized Hessian `H_user`
/// (unit implicit dispersion). For a free-dispersion family the latent
/// posterior covariance is `φ̂·H_latent⁻¹`, so the caller passes
/// `sqrt_cov_scale = √φ̂` (the coefficient-covariance scale `√σ̂²` for a
/// profiled Gaussian, `1` for fixed-scale families like Binomial) and every
/// latent perturbation is multiplied by it. This makes the draw covariance
/// `sqrt_cov_scale² · H_latent⁻¹`, matching the fit's reported
/// `Vb = cov_scale·H_user⁻¹` exactly (gam#1514) — without it a Gaussian
/// bounded slope's draws were ~`1/σ̂` too wide.
///
/// Returns the draws as a `(n_draws, p)` matrix on the *internal* user scale
/// (still conditioned); the caller back-transforms to the original data scale
/// with the same conditioning it used for the point estimate.
pub fn sample_bounded_latent_posterior_internal(
    beta_user: &Array1<f64>,
    user_hessian: &Array2<f64>,
    bounded_columns: &[BoundedSampleColumn],
    n_draws: usize,
    sqrt_cov_scale: f64,
    base_seed: u64,
) -> Result<Array2<f64>, EstimationError> {
    let p = beta_user.len();
    if user_hessian.nrows() != p || user_hessian.ncols() != p {
        crate::bail_invalid_estim!(
            "bounded posterior sampling dimension mismatch: mode has {p} entries, user Hessian is {}x{}",
            user_hessian.nrows(),
            user_hessian.ncols()
        );
    }
    if beta_user.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_estim!("bounded posterior sampling requires a finite mode");
    }
    if user_hessian.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_estim!("bounded posterior sampling requires a finite Hessian");
    }
    if !(sqrt_cov_scale.is_finite() && sqrt_cov_scale >= 0.0) {
        crate::bail_invalid_estim!(
            "bounded posterior sampling covariance scale must be finite and non-negative, got {sqrt_cov_scale}"
        );
    }

    // Latent mode and delta-method Jacobian, column by column.
    let mut theta_mode = beta_user.clone();
    let mut jac_diag = Array1::<f64>::ones(p);
    for bc in bounded_columns {
        if bc.col_idx >= p {
            crate::bail_invalid_estim!(
                "bounded posterior sampling: bounded column index {} out of range for {p} coefficients",
                bc.col_idx
            );
        }
        if !(bc.min.is_finite()
            && bc.max.is_finite()
            && (bc.max - bc.min).is_finite()
            && bc.min < beta_user[bc.col_idx]
            && beta_user[bc.col_idx] < bc.max)
        {
            crate::bail_invalid_estim!(
                "bounded posterior sampling column {} requires finite bounds with a finite width and a mode strictly inside ({}, {}); got {}",
                bc.col_idx,
                bc.min,
                bc.max,
                beta_user[bc.col_idx]
            );
        }
        let theta_i = bounded_user_to_latent(beta_user[bc.col_idx], bc.min, bc.max);
        let (_, _, db_dtheta) = bounded_latent_to_user(theta_i, bc.min, bc.max);
        if !(theta_i.is_finite() && db_dtheta.is_finite() && db_dtheta > 0.0) {
            crate::bail_invalid_estim!(
                "bounded posterior sampling column {} has unrepresentable latent geometry: theta={theta_i}, d_beta/d_theta={db_dtheta}",
                bc.col_idx
            );
        }
        theta_mode[bc.col_idx] = theta_i;
        jac_diag[bc.col_idx] = db_dtheta;
    }

    // H_latent = J H_user J  (J diagonal). This is the exact inverse of the
    // user-scale precision transform applied at fit time.
    let mut h_latent = user_hessian.clone();
    for i in 0..p {
        let ji = jac_diag[i];
        if ji != 1.0 {
            h_latent.row_mut(i).mapv_inplace(|v| v * ji);
            h_latent.column_mut(i).mapv_inplace(|v| v * ji);
        }
    }

    // Draw theta ~ N(theta_mode, H_latent^{-1}) via the Cholesky of H_latent:
    // L Lᵀ = H_latent, solve Lᵀ δ = ε so Var(δ) = H_latent^{-1}.
    use gam_linalg::faer_ndarray::FaerCholesky as _;
    use rand::SeedableRng as _;
    let chol = h_latent.cholesky(faer::Side::Lower).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "bounded posterior sampling: Cholesky of the latent penalized Hessian failed: {err:?}"
        ))
    })?;
    let l = chol.lower_triangular();

    let mut draws = Array2::<f64>::zeros((n_draws, p));
    let mut eps = Array1::<f64>::zeros(p);
    let mut delta = Array1::<f64>::zeros(p);
    let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed);
    for k in 0..n_draws {
        for e in eps.iter_mut() {
            *e = standard_normal_draw(&mut rng);
        }
        solve_lower_transpose_into(&l, &eps, &mut delta)?;
        for i in 0..p {
            // δ has covariance `H_latent⁻¹`; scaling by √cov_scale lifts it to
            // the dispersion-correct posterior covariance `cov_scale·H_latent⁻¹`.
            draws[(k, i)] = theta_mode[i] + sqrt_cov_scale * delta[i];
        }
        // Push bounded columns through the exact interval map; leave
        // unconstrained columns untouched. In a far IEEE tail the closest
        // representable image can equal an endpoint even though the latent
        // coordinate and its derivative remain finite.
        for bc in bounded_columns {
            let (beta_draw, _, _) = bounded_latent_to_user(draws[(k, bc.col_idx)], bc.min, bc.max);
            draws[(k, bc.col_idx)] = beta_draw;
        }
    }

    Ok(draws)
}

/// Box-Muller standard-normal draw (kept local so the bounded sampler does not
/// depend on the HMC module's RNG plumbing).
#[inline]
fn standard_normal_draw<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    use rand::RngExt as _;
    let u1 = loop {
        let candidate = rng.random::<f64>();
        if candidate > 0.0 {
            break candidate;
        }
    };
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Solve `Lᵀ x = b` for a lower-triangular `L` (back substitution), writing the
/// result into `out`. Used to turn a standard-normal `b` into a draw with
/// covariance `(L Lᵀ)^{-1}`.
fn solve_lower_transpose_into(
    l: &Array2<f64>,
    b: &Array1<f64>,
    out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    let p = l.nrows();
    if l.ncols() != p || b.len() != p || out.len() != p {
        crate::bail_invalid_estim!(
            "bounded triangular solve dimension mismatch: L={}x{}, b={}, out={}",
            l.nrows(),
            l.ncols(),
            b.len(),
            out.len()
        );
    }
    for i in (0..p).rev() {
        let mut acc = b[i];
        for j in (i + 1)..p {
            acc -= l[(j, i)] * out[j];
        }
        let diag = l[(i, i)];
        if !(diag.is_finite() && diag > 0.0 && acc.is_finite()) {
            crate::bail_invalid_estim!(
                "bounded triangular solve has invalid row {i}: diagonal={diag}, residual={acc}"
            );
        }
        let value = acc / diag;
        if !value.is_finite() {
            crate::bail_invalid_estim!(
                "bounded triangular solve produced a non-finite value at row {i}: {acc}/{diag}"
            );
        }
        out[i] = value;
    }
    Ok(())
}

/// Largest latent magnitude at which `bounded()`'s interval map is still
/// injective in binary64.
///
/// `bounded()` fits `beta = min + width*sigma(theta)`, and `sigma(theta)`
/// rounds to exactly `1.0` as soon as `exp(-theta) < EPSILON/2` — that is, for
/// `theta > ln(2/EPSILON)` (and symmetrically to `0.0` below `-ln(2/EPSILON)`).
/// Past that point `beta` is frozen bit for bit while
/// `dbeta/dtheta = width*sigma*(1-sigma)` stays strictly positive all the way
/// out to the exponential's underflow near `|theta| = 745`, so the solver reads
/// live first and second derivatives along a direction that cannot move the
/// model at all. The bound is derived from the representation, not chosen.
fn bounded_latent_injective_limit() -> f64 {
    (2.0 / f64::EPSILON).ln()
}

fn bounded_latent_derivatives(
    theta: f64,
    min: f64,
    max: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let jet = logit_inverse_link_jet5(theta);
    let z = jet.mu;
    let width = max - min;
    let beta = min + width * z;
    let db_dtheta = width * jet.d1;
    let d2b_dtheta2 = width * jet.d2;
    let d3b_dtheta3 = width * jet.d3;
    let d4b_dtheta4 = width * jet.d4;
    let d5b_dtheta5 = width * jet.d5;
    (beta, z, db_dtheta, d2b_dtheta2, d3b_dtheta3, d4b_dtheta4, d5b_dtheta5)
}

/// The Beta shapes `(a, b)` of a bounded coefficient's prior on the normalized
/// user-scale coefficient, or `None` when the coefficient carries no prior term.
fn bounded_prior_shapes(prior: &BoundedCoefficientPriorSpec) -> Result<Option<(f64, f64)>, String> {
    let (a, b) = match prior {
        // `None` means constrained MLE with no extra prior term on the bounded coefficient.
        BoundedCoefficientPriorSpec::None => return Ok(None),
        // Uniform on the normalized user-scale coefficient z in (0, 1). In latent space this is
        // exactly the Jacobian term for the logistic transform, up to an additive width constant.
        BoundedCoefficientPriorSpec::Uniform => (1.0, 1.0),
        BoundedCoefficientPriorSpec::Beta { a, b } => (*a, *b),
    };
    if !(a.is_finite() && a > 0.0 && b.is_finite() && b > 0.0) {
        return Err(format!(
            "bounded coefficient Beta prior requires finite positive shapes, got ({a}, {b})"
        ));
    }
    Ok(Some((a, b)))
}

/// Third latent derivative of the prior's negative Hessian `(a+b)·σ'(θ)`, the
/// prior's fifth derivative that the third information derivative reads (#2903).
fn bounded_prior_neghess_third_derivative(
    theta: f64,
    prior: &BoundedCoefficientPriorSpec,
) -> Result<f64, String> {
    let Some((a, b)) = bounded_prior_shapes(prior)? else {
        return Ok(0.0);
    };
    let value = (a + b) * logit_inverse_link_jet5(theta).d4;
    if !value.is_finite() {
        return Err(format!(
            "bounded coefficient prior fifth derivative is not representable at theta={theta}: {value}"
        ));
    }
    Ok(value)
}

fn bounded_prior_terms(
    theta: f64,
    prior: &BoundedCoefficientPriorSpec,
) -> Result<(f64, f64, f64, f64, f64), String> {
    if !theta.is_finite() {
        return Err(format!(
            "bounded coefficient prior requires a finite latent coordinate, got {theta}"
        ));
    }
    let Some((a, b)) = bounded_prior_shapes(prior)? else {
        return Ok((0.0, 0.0, 0.0, 0.0, 0.0));
    };
    let jet = logit_inverse_link_jet5(theta);
    let z = jet.mu;
    // log(sigmoid(theta)) = -softplus(-theta) and
    // log(1-sigmoid(theta)) = -softplus(theta).  Evaluating the prior on
    // these natural-coordinate tails keeps its value and derivative tower on
    // one surface even after `z` itself rounds to an endpoint.
    let logp = -a * gam_linalg::utils::stable_softplus(-theta)
        - b * gam_linalg::utils::stable_softplus(theta);
    let grad = a - (a + b) * z;
    let neghess = (a + b) * jet.d1;
    let neghess_derivative = (a + b) * jet.d2;
    let neghess_second_derivative = (a + b) * jet.d3;
    let terms = (logp, grad, neghess, neghess_derivative, neghess_second_derivative);
    if [terms.0, terms.1, terms.2, terms.3, terms.4]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(format!(
            "bounded coefficient prior geometry is not representable at theta={theta}: {terms:?}"
        ));
    }
    Ok(terms)
}

#[derive(Clone, Copy)]
struct ExactStandardObservationRow {
    mu: f64,
    score: f64,
    fisherweight: f64,
    neghessian_eta: f64,
    neghessian_eta_derivative: f64,
    neghessian_eta_second_derivative: f64,
    /// `W'''`, the third η-derivative of the observed negative Hessian (#2903).
    neghessian_eta_third_derivative: f64,
    log_likelihood: f64,
}

impl ExactStandardObservationRow {
    #[inline]
    fn zero_weight(mu: f64) -> Self {
        Self {
            mu,
            score: 0.0,
            fisherweight: 0.0,
            neghessian_eta: 0.0,
            neghessian_eta_derivative: 0.0,
            neghessian_eta_second_derivative: 0.0,
            neghessian_eta_third_derivative: 0.0,
            log_likelihood: 0.0,
        }
    }
}


#[inline]
fn certify_bounded_row(
    row: usize,
    eta: f64,
    state: ExactStandardObservationRow,
) -> Result<ExactStandardObservationRow, EstimationError> {
    for (quantity, value) in [
        ("bounded-family mean", state.mu),
        ("bounded-family score", state.score),
        ("bounded-family Fisher weight", state.fisherweight),
        ("bounded-family observed Hessian", state.neghessian_eta),
        (
            "bounded-family observed Hessian derivative",
            state.neghessian_eta_derivative,
        ),
        (
            "bounded-family observed Hessian second derivative",
            state.neghessian_eta_second_derivative,
        ),
        ("bounded-family log likelihood", state.log_likelihood),
    ] {
        if !value.is_finite() {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(row, quantity, eta, value));
        }
    }
    if !state.neghessian_eta_third_derivative.is_finite() {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "bounded-family observed Hessian third derivative",
            eta,
            state.neghessian_eta_third_derivative,
        ));
    }
    if state.fisherweight < 0.0 {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "bounded-family Fisher weight",
            eta,
            state.fisherweight,
        ));
    }
    Ok(state)
}

#[inline]
fn weighted_positive_from_log(weight: f64, log_value: f64) -> f64 {
    if weight == 0.0 {
        return 0.0;
    }
    (weight.ln() + log_value).exp()
}

#[inline]
fn weighted_product3(a: f64, b: f64, c: f64) -> f64 {
    crate::gamlss::scaled_signed_product3(a, b, c)
}

fn resolved_bounded_binomial_link(
    family: &LikelihoodSpec,
    latent_cloglog_state: Option<&LatentCLogLogState>,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> InverseLink {
    match &family.link {
        InverseLink::LatentCLogLog(_) => latent_cloglog_state
            .copied()
            .map(InverseLink::LatentCLogLog)
            .unwrap_or_else(|| family.link.clone()),
        InverseLink::Mixture(_) => mixture_link_state
            .cloned()
            .map(InverseLink::Mixture)
            .unwrap_or_else(|| family.link.clone()),
        InverseLink::Sas(_) => sas_link_state
            .copied()
            .map(InverseLink::Sas)
            .unwrap_or_else(|| family.link.clone()),
        InverseLink::BetaLogistic(_) => sas_link_state
            .copied()
            .map(InverseLink::BetaLogistic)
            .unwrap_or_else(|| family.link.clone()),
        InverseLink::Standard(_) => family.link.clone(),
    }
}

fn exact_logit_observation_row(
    row: usize,
    y: f64,
    weight: f64,
    eta: f64,
) -> Result<ExactStandardObservationRow, EstimationError> {
    let tail = (-eta.abs()).exp();
    let (mu, one_minus_mu) = if eta >= 0.0 {
        let q = tail / (1.0 + tail);
        (1.0 - q, q)
    } else {
        let p = tail / (1.0 + tail);
        (p, 1.0 - p)
    };
    if weight == 0.0 {
        return Ok(ExactStandardObservationRow::zero_weight(mu));
    }
    let log_fisher =
        -gam_linalg::utils::stable_softplus(eta) - gam_linalg::utils::stable_softplus(-eta);
    let fisherweight = weighted_positive_from_log(weight, log_fisher);
    if !(fisherweight.is_finite() && fisherweight > 0.0) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "bounded logit Fisher weight",
            eta,
            fisherweight,
        ));
    }
    let residual = if eta >= 0.0 {
        if y == 1.0 {
            one_minus_mu
        } else {
            (y - 1.0) + one_minus_mu
        }
    } else {
        y - mu
    };
    let log_likelihood_unit = if eta >= 0.0 {
        -(1.0 - y) * eta - gam_linalg::utils::stable_softplus(-eta)
    } else {
        y * eta - gam_linalg::utils::stable_softplus(eta)
    };
    certify_bounded_row(
        row,
        eta,
        ExactStandardObservationRow {
            mu,
            score: weight * residual,
            fisherweight,
            neghessian_eta: fisherweight,
            neghessian_eta_derivative: fisherweight * (one_minus_mu - mu),
            neghessian_eta_second_derivative: fisherweight * (1.0 - 6.0 * mu * one_minus_mu),
            // d/dη of W·(1 − 6μ(1−μ)), with dμ(1−μ)/dη = μ(1−μ)(1−2μ).
            neghessian_eta_third_derivative: fisherweight
                * (one_minus_mu - mu)
                * (1.0 - 12.0 * mu * one_minus_mu),
            log_likelihood: weight * log_likelihood_unit,
        },
    )
}

fn exact_noncanonical_binomial_observation_row(
    row: usize,
    y: f64,
    weight: f64,
    eta: f64,
    link: &InverseLink,
) -> Result<ExactStandardObservationRow, EstimationError> {
    let observation = bernoulli_natural_observation(row, y, eta, link)?;
    if weight == 0.0 {
        return Ok(ExactStandardObservationRow::zero_weight(observation.mu));
    }
    let third_derivative = bernoulli_natural_negative_hessian_third_derivative(row, y, eta, link)?;
    let fisherweight = weighted_positive_from_log(weight, observation.log_fisher);
    if !(fisherweight.is_finite() && fisherweight > 0.0) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "bounded binomial Fisher weight",
            eta,
            fisherweight,
        ));
    }
    certify_bounded_row(
        row,
        eta,
        ExactStandardObservationRow {
            mu: observation.mu,
            score: weight * observation.score,
            fisherweight,
            neghessian_eta: weight * observation.negative_hessian,
            neghessian_eta_derivative: weight * observation.negative_hessian_derivative,
            neghessian_eta_second_derivative: weight
                * observation.negative_hessian_second_derivative,
            neghessian_eta_third_derivative: weight * third_derivative,
            log_likelihood: weight * observation.log_likelihood,
        },
    )
}

#[inline]
fn eta_exprel(rate: f64, eta: f64) -> f64 {
    (rate * eta).exp_m1() / rate
}

fn validate_bounded_observation_inputs(
    likelihood: &gam_spec::GlmLikelihoodSpec,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta: &Array1<f64>,
) -> Result<gam_spec::ResolvedLikelihoodScale, EstimationError> {
    let family = &likelihood.spec;
    if weights.len() != y.len() || eta.len() != y.len() {
        crate::bail_invalid_estim!(
            "bounded family observation size mismatch: y={}, weights={}, eta={}",
            y.len(),
            weights.len(),
            eta.len()
        );
    }
    if !LikelihoodSpec::is_legal_cell(&family.response, &family.link) {
        crate::bail_invalid_estim!(
            "bounded family received illegal likelihood cell response={} link={}",
            family.response.name(),
            family.link.link_function().name()
        );
    }
    let resolved_scale = likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    if let ResponseFamily::Tweedie { p } = &family.response
        && !(p.is_finite() && *p > 1.0 && *p < 2.0)
    {
        crate::bail_invalid_estim!(
            "bounded Tweedie power must be finite and strictly inside (1, 2), got {p}"
        );
    }
    if let ResponseFamily::NegativeBinomial { theta, .. } = &family.response
        && !(theta.is_finite() && *theta > 0.0)
    {
        crate::bail_invalid_estim!(
            "bounded negative-binomial theta must be finite and positive, got {theta}"
        );
    }
    // Atomic whole-vector preflight: an invalid later weight wins before any
    // response or predictor row is inspected.
    // The row index is 0-based, matching the row-geometry refusals below — which
    // reports the same rows of the same vectors through the typed
    // `PirlsRowGeometryUnrepresentable { row }` field a caller can use to index
    // straight back into `y`/`weights`/`eta`. This message used to emit `i + 1`,
    // so one function refused two kinds of bad row under two different index
    // bases and a reader could not tell which one a given "row N" meant.
    for (i, &wi) in weights.iter().enumerate() {
        if !(wi.is_finite() && wi >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "bounded-family row {i} has invalid prior weight {wi:?}; expected finite weight >= 0"
            )));
        }
    }
    for i in 0..y.len() {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        if !eta[i].is_finite() {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(i, "linear predictor", eta[i], eta[i]));
        }
        if !y[i].is_finite() {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "bounded-family response",
                eta[i],
                y[i],
            ));
        }
        let yi = y[i];
        let valid = match &family.response {
            ResponseFamily::Gaussian => yi.is_finite(),
            ResponseFamily::Binomial => yi.is_finite() && (0.0..=1.0).contains(&yi),
            // A count is an exact non-negative integer: a value near one is not a
            // count, and the likelihood would read the value it carries.
            ResponseFamily::Poisson | ResponseFamily::NegativeBinomial { .. } => {
                yi.is_finite() && yi >= 0.0 && yi == yi.round()
            }
            ResponseFamily::Tweedie { .. } => yi.is_finite() && yi >= 0.0,
            ResponseFamily::Gamma => yi.is_finite() && yi > 0.0,
            ResponseFamily::Beta { .. } | ResponseFamily::RoystonParmar => false,
        };
        if !valid {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(i, "bounded-family response", eta[i], yi));
        }
    }
    Ok(resolved_scale)
}

fn exact_standard_observation_row(
    likelihood: &gam_spec::GlmLikelihoodSpec,
    resolved_scale: gam_spec::ResolvedLikelihoodScale,
    binomial_link: &InverseLink,
    row: usize,
    y: f64,
    weight: f64,
    eta: f64,
) -> Result<ExactStandardObservationRow, EstimationError> {
    if weight == 0.0 {
        return Ok(ExactStandardObservationRow::zero_weight(0.0));
    }
    let family = &likelihood.spec;
    match &family.response {
        ResponseFamily::Gaussian => {
            let scaled_weight = match resolved_scale {
                gam_spec::ResolvedLikelihoodScale::ProfiledGaussian => weight,
                gam_spec::ResolvedLikelihoodScale::FixedGaussian { phi } => {
                    gam_math::special::scaled_positive_product_quotient(weight, 1.0, 1.0, phi.value())
                }
                _ => {
                    crate::bail_invalid_estim!(
                        "bounded Gaussian received a non-Gaussian resolved scale"
                    );
                }
            };
            if !(scaled_weight.is_finite() && scaled_weight > 0.0) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
                    row,
                    "bounded Gaussian dispersion-scaled weight",
                    eta,
                    scaled_weight,
                ));
            }
            let residual = y - eta;
            let loss = if residual == 0.0 {
                0.0
            } else {
                gam_math::special::scaled_positive_product_quotient(
                    scaled_weight,
                    residual.abs(),
                    residual.abs(),
                    2.0,
                )
            };
            certify_bounded_row(
                row,
                eta,
                ExactStandardObservationRow {
                    mu: eta,
                    score: scaled_weight * residual,
                    fisherweight: scaled_weight,
                    neghessian_eta: scaled_weight,
                    neghessian_eta_derivative: 0.0,
                    neghessian_eta_second_derivative: 0.0,
                    neghessian_eta_third_derivative: 0.0,
                    log_likelihood: -loss,
                },
            )
        }
        ResponseFamily::Binomial
            if matches!(binomial_link, InverseLink::Standard(StandardLink::Logit)) =>
        {
            exact_logit_observation_row(row, y, weight, eta)
        }
        ResponseFamily::Binomial => {
            exact_noncanonical_binomial_observation_row(row, y, weight, eta, binomial_link)
        }
        ResponseFamily::Poisson => {
            let mu = inverse_link_jet_for_inverse_link(&family.link, eta)?.mu;
            let fisherweight = weight * mu;
            let score = weight * (y - mu);
            let raw_log_likelihood = y.mul_add(eta, -mu);
            let log_likelihood = if raw_log_likelihood.is_finite() {
                weight * raw_log_likelihood
            } else {
                weighted_product3(weight, y, eta) - weight * mu
            };
            if !(fisherweight.is_finite() && fisherweight > 0.0) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
                    row,
                    "bounded Poisson Fisher weight",
                    eta,
                    fisherweight,
                ));
            }
            certify_bounded_row(
                row,
                eta,
                ExactStandardObservationRow {
                    mu,
                    score,
                    fisherweight,
                    neghessian_eta: fisherweight,
                    neghessian_eta_derivative: fisherweight,
                    neghessian_eta_second_derivative: fisherweight,
                    neghessian_eta_third_derivative: fisherweight,
                    log_likelihood,
                },
            )
        }
        ResponseFamily::Gamma => {
            let mu = inverse_link_jet_for_inverse_link(&family.link, eta)?.mu;
            let shape = resolved_scale
                .gamma_shape()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
            let weighted_shape = weight * shape;
            if !(weighted_shape.is_finite() && weighted_shape > 0.0) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
                    row,
                    "bounded Gamma shape-scaled weight",
                    eta,
                    weighted_shape,
                ));
            }
            let weighted_ratio =
                gam_math::special::scaled_positive_product_quotient(weight, y, shape, mu);
            if !(weighted_ratio.is_finite() && weighted_ratio > 0.0) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
                    row,
                    "bounded Gamma observed Hessian",
                    eta,
                    weighted_ratio,
                ));
            }
            certify_bounded_row(
                row,
                eta,
                ExactStandardObservationRow {
                    mu,
                    score: weighted_ratio - weighted_shape,
                    fisherweight: weighted_shape,
                    neghessian_eta: weighted_ratio,
                    neghessian_eta_derivative: -weighted_ratio,
                    neghessian_eta_second_derivative: weighted_ratio,
                    neghessian_eta_third_derivative: -weighted_ratio,
                    log_likelihood: -weighted_ratio - weighted_shape * eta,
                },
            )
        }
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            let mu = inverse_link_jet_for_inverse_link(&family.link, eta)?.mu;
            let phi = resolved_scale
                .tweedie_phi()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
            let weight = gam_math::special::scaled_positive_product_quotient(weight, 1.0, 1.0, phi);
            if !(weight.is_finite() && weight > 0.0) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
                    row,
                    "bounded Tweedie dispersion-scaled weight",
                    eta,
                    weight,
                ));
            }
            let a = ((1.0 - p) * eta).exp();
            let b = ((2.0 - p) * eta).exp();
            let score_unit = y.mul_add(a, -b);
            let score = if score_unit.is_finite() {
                weight * score_unit
            } else {
                weighted_product3(weight, y, a) - weight * b
            };
            let fisherweight = weight * b;
            let observed_unit = (p - 1.0) * y * a + (2.0 - p) * b;
            let neghessian_eta = if observed_unit.is_finite() {
                weight * observed_unit
            } else {
                weighted_product3(weight * (p - 1.0), y, a) + weight * (2.0 - p) * b
            };
            let observed_derivative_unit = -(p - 1.0).powi(2) * y * a + (2.0 - p).powi(2) * b;
            let neghessian_eta_derivative = if observed_derivative_unit.is_finite() {
                weight * observed_derivative_unit
            } else {
                -weighted_product3(weight * (p - 1.0).powi(2), y, a)
                    + weight * (2.0 - p).powi(2) * b
            };
            let observed_second_derivative_unit =
                (p - 1.0).powi(3) * y * a + (2.0 - p).powi(3) * b;
            let neghessian_eta_second_derivative = if observed_second_derivative_unit.is_finite() {
                weight * observed_second_derivative_unit
            } else {
                weighted_product3(weight * (p - 1.0).powi(3), y, a)
                    + weight * (2.0 - p).powi(3) * b
            };
            let observed_third_derivative_unit =
                -(p - 1.0).powi(4) * y * a + (2.0 - p).powi(4) * b;
            let neghessian_eta_third_derivative = if observed_third_derivative_unit.is_finite() {
                weight * observed_third_derivative_unit
            } else {
                -weighted_product3(weight * (p - 1.0).powi(4), y, a)
                    + weight * (2.0 - p).powi(4) * b
            };
            // Centering Q at eta=0 removes response-only poles as p approaches
            // 1 or 2 without changing any eta derivative.
            let q_left = eta_exprel(1.0 - p, eta);
            let q_right = eta_exprel(2.0 - p, eta);
            let q = y.mul_add(q_left, -q_right);
            let log_likelihood = if q.is_finite() {
                weight * q
            } else {
                weighted_product3(weight, y, q_left) - weight * q_right
            };
            if !(fisherweight.is_finite() && fisherweight > 0.0) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
                    row,
                    "bounded Tweedie Fisher weight",
                    eta,
                    fisherweight,
                ));
            }
            certify_bounded_row(
                row,
                eta,
                ExactStandardObservationRow {
                    mu,
                    score,
                    fisherweight,
                    neghessian_eta,
                    neghessian_eta_derivative,
                    neghessian_eta_second_derivative,
                    neghessian_eta_third_derivative,
                    log_likelihood,
                },
            )
        }
        ResponseFamily::NegativeBinomial { .. } => {
            let theta = resolved_scale
                .negative_binomial_theta()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
            let mu = inverse_link_jet_for_inverse_link(&family.link, eta)?.mu;
            let log_theta = theta.ln();
            let delta = eta - log_theta;
            let log_q = -gam_linalg::utils::stable_softplus(-delta);
            let log_r = -gam_linalg::utils::stable_softplus(delta);
            let q = log_q.exp();
            let r = log_r.exp();
            let y_r = if y == 0.0 {
                0.0
            } else {
                (y.ln() + log_r).exp()
            };
            let theta_q = (log_theta + log_q).exp();
            let score = weight * (y_r - theta_q);
            let fisherweight = weighted_positive_from_log(weight, log_theta + log_q);
            let log_qr = log_q + log_r;
            let observed_y = if y == 0.0 {
                0.0
            } else {
                weighted_positive_from_log(weight, y.ln() + log_qr)
            };
            let observed_theta = weighted_positive_from_log(weight, log_theta + log_qr);
            let neghessian_eta = observed_y + observed_theta;
            let neghessian_eta_derivative = neghessian_eta * (r - q);
            // d(qr)/dδ = qr(r - q) and d(r - q)/dδ = -2qr.
            let neghessian_eta_second_derivative = neghessian_eta * ((r - q) * (r - q) - 2.0 * q * r);
            // d/dδ of W·((r − q)² − 2qr) = W(r − q)((r − q)² − 2qr) − 6W·qr(r − q).
            let neghessian_eta_third_derivative =
                neghessian_eta * (r - q) * ((r - q) * (r - q) - 8.0 * q * r);
            let softplus_tail = if delta >= 0.0 {
                gam_linalg::utils::stable_softplus(-delta)
            } else {
                gam_linalg::utils::stable_softplus(delta)
            };
            let log_likelihood = if delta >= 0.0 {
                -weighted_product3(weight, theta, delta)
                    - weighted_product3(weight, y, softplus_tail)
                    - weighted_product3(weight, theta, softplus_tail)
            } else {
                weighted_product3(weight, y, delta)
                    - weighted_product3(weight, y, softplus_tail)
                    - weighted_product3(weight, theta, softplus_tail)
            };
            if !(fisherweight.is_finite() && fisherweight > 0.0) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
                    row,
                    "bounded negative-binomial Fisher weight",
                    eta,
                    fisherweight,
                ));
            }
            certify_bounded_row(
                row,
                eta,
                ExactStandardObservationRow {
                    mu,
                    score,
                    fisherweight,
                    neghessian_eta,
                    neghessian_eta_derivative,
                    neghessian_eta_second_derivative,
                    neghessian_eta_third_derivative,
                    log_likelihood,
                },
            )
        }
        ResponseFamily::Beta { .. } => {
            crate::bail_invalid_estim!("bounded linear terms are not supported for BetaLogit fits");
        }
        ResponseFamily::RoystonParmar => {
            crate::bail_invalid_estim!(
                "bounded linear terms are not supported for survival model fits"
            );
        }
    }
}

fn evaluate_resolved_standard_family_observations(
    likelihood: &gam_spec::GlmLikelihoodSpec,
    latent_cloglog_state: Option<&LatentCLogLogState>,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta: &Array1<f64>,
) -> Result<StandardFamilyObservationState, EstimationError> {
    let n = y.len();
    let resolved_scale = validate_bounded_observation_inputs(likelihood, y, weights, eta)?;
    let family = &likelihood.spec;
    let binomial_link = resolved_bounded_binomial_link(
        &family,
        latent_cloglog_state,
        mixture_link_state,
        sas_link_state,
    );

    let mut score = Array1::<f64>::zeros(n);
    let mut fisherweight = Array1::<f64>::zeros(n);
    let mut neghessian_eta = Array1::<f64>::zeros(n);
    let mut neghessian_eta_derivative = Array1::<f64>::zeros(n);
    let mut neghessian_eta_second_derivative = Array1::<f64>::zeros(n);
    let mut neghessian_eta_third_derivative = Array1::<f64>::zeros(n);
    let mut log_likelihood = 0.0;
    let mut log_likelihood_compensation = 0.0;

    for i in 0..n {
        let row = exact_standard_observation_row(
            likelihood,
            resolved_scale,
            &binomial_link,
            i,
            y[i],
            weights[i],
            eta[i],
        )?;
        score[i] = row.score;
        fisherweight[i] = row.fisherweight;
        neghessian_eta[i] = row.neghessian_eta;
        neghessian_eta_derivative[i] = row.neghessian_eta_derivative;
        neghessian_eta_second_derivative[i] = row.neghessian_eta_second_derivative;
        neghessian_eta_third_derivative[i] = row.neghessian_eta_third_derivative;
        let adjusted = row.log_likelihood - log_likelihood_compensation;
        let updated = log_likelihood + adjusted;
        log_likelihood_compensation = (updated - log_likelihood) - adjusted;
        log_likelihood = updated;
        if !log_likelihood.is_finite() {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "bounded-family cumulative log likelihood",
                eta[i],
                log_likelihood,
            ));
        }
    }

    Ok(StandardFamilyObservationState {
        eta: eta.clone(),
        score,
        fisherweight,
        neghessian_eta,
        neghessian_eta_derivative,
        neghessian_eta_second_derivative,
        neghessian_eta_third_derivative,
        log_likelihood,
    })
}

/// Canonical scale-resolution boundary for callers whose family has not yet
/// entered a fit and therefore has no independently fitted scale metadata.
/// Bounded fits carry a full `GlmLikelihoodSpec` and call the resolved variant
/// directly; this path derives the family-defined estimated/fixed seed once.
fn evaluate_standard_familyobservations(
    family: LikelihoodSpec,
    latent_cloglog_state: Option<&LatentCLogLogState>,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta: &Array1<f64>,
) -> Result<StandardFamilyObservationState, EstimationError> {
    let likelihood = gam_spec::GlmLikelihoodSpec::canonical(family);
    evaluate_resolved_standard_family_observations(
        &likelihood,
        latent_cloglog_state,
        mixture_link_state,
        sas_link_state,
        y,
        weights,
        eta,
    )
}

fn exact_standard_working_response(
    state: &StandardFamilyObservationState,
) -> Result<Array1<f64>, EstimationError> {
    let mut out = state.eta.clone();
    for i in 0..out.len() {
        let weight = state.fisherweight[i];
        let score = state.score[i];
        if weight == 0.0 {
            if score != 0.0 {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
                    i,
                    "zero-Fisher row with nonzero score",
                    state.eta[i],
                    score,
                ));
            }
            continue;
        }
        let increment = score / weight;
        let value = out[i] + increment;
        if !increment.is_finite() || !value.is_finite() {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "bounded-family working response",
                state.eta[i],
                value,
            ));
        }
        out[i] = value;
    }
    Ok(out)
}

fn expect_single_block_state<'a>(
    block_states: &'a [ParameterBlockState],
    family_name: &str,
) -> Result<&'a ParameterBlockState, String> {
    crate::block_layout::block_count::validate_block_count::<SmoothError>(
        family_name,
        1,
        block_states.len(),
    )?;
    Ok(&block_states[0])
}

fn expect_block_idx_zero(block_idx: usize, family_name: &str, context: &str) -> Result<(), String> {
    if block_idx != 0 {
        return Err(SmoothError::invalid_index(format!(
            "{family_name} expects block_idx 0{context}, got {block_idx}"
        ))
        .into());
    }
    Ok::<(), _>(())
}

impl BoundedLinearFamily {
    fn bounded_term_derivative_data(
        &self,
        latent_beta: &Array1<f64>,
    ) -> Result<
        (
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
        ),
        String,
    > {
        let p = latent_beta.len();
        if p != self.design.ncols() || latent_beta.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "bounded coefficient geometry requires {} finite latent coefficients, got {}",
                self.design.ncols(),
                p
            ));
        }
        let mut beta_user = latent_beta.clone();
        let mut jac_diag = Array1::<f64>::ones(p);
        let mut second_diag = Array1::<f64>::zeros(p);
        let mut third_diag = Array1::<f64>::zeros(p);
        let mut priorthird = Array1::<f64>::zeros(p);
        for term in &self.bounded_terms {
            let width = term.max - term.min;
            if term.col_idx >= p
                || !term.min.is_finite()
                || !term.max.is_finite()
                || !(width.is_finite() && width > 0.0)
            {
                return Err(format!(
                    "bounded coefficient geometry has invalid column/bounds: col={}, p={p}, bounds=({}, {})",
                    term.col_idx, term.min, term.max
                ));
            }
            let (beta, _, db_dtheta, d2b_dtheta2, d3b_dtheta3, _, _) =
                bounded_latent_derivatives(latent_beta[term.col_idx], term.min, term.max);
            if [beta, db_dtheta, d2b_dtheta2, d3b_dtheta3]
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(format!(
                    "bounded coefficient transform is not representable at column {} and theta={}",
                    term.col_idx, latent_beta[term.col_idx]
                ));
            }
            beta_user[term.col_idx] = beta;
            jac_diag[term.col_idx] = db_dtheta;
            second_diag[term.col_idx] = d2b_dtheta2;
            third_diag[term.col_idx] = d3b_dtheta3;
            let (_, _, _, prior_neghess_derivative, _) =
                bounded_prior_terms(latent_beta[term.col_idx], &term.prior)?;
            priorthird[term.col_idx] = prior_neghess_derivative;
        }
        Ok((beta_user, jac_diag, second_diag, third_diag, priorthird))
    }

    fn user_beta_and_jacobian(
        &self,
        latent_beta: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let (beta_user, jac_diag, _, _, _) = self.bounded_term_derivative_data(latent_beta)?;
        Ok((beta_user, jac_diag))
    }

    fn nonlinear_offset_from_latent(
        &self,
        latent_beta: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        self.bounded_term_derivative_data(latent_beta)?;
        let mut offset = self.offset.clone();
        for term in &self.bounded_terms {
            let (beta, _, _) =
                bounded_latent_to_user(latent_beta[term.col_idx], term.min, term.max);
            offset.scaled_add(beta, &self.design.column(term.col_idx));
        }
        if offset.iter().any(|value| !value.is_finite()) {
            return Err("bounded nonlinear offset is not representable".to_string());
        }
        Ok(offset)
    }

    fn effective_design_for_latent(&self, jac_diag: &Array1<f64>) -> Array2<f64> {
        let mut x_eff = self.design.clone();
        for term in &self.bounded_terms {
            x_eff
                .column_mut(term.col_idx)
                .mapv_inplace(|v| v * jac_diag[term.col_idx]);
        }
        x_eff
    }

    fn exacthessian_andgradient(
        &self,
        latent_beta: &Array1<f64>,
    ) -> Result<
        (
            StandardFamilyObservationState,
            Array2<f64>,
            Array1<f64>,
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
        ),
        String,
    > {
        let (_, jac_diag, second_diag, third_diag, priorthird) =
            self.bounded_term_derivative_data(latent_beta)?;
        let x_eff = self.effective_design_for_latent(&jac_diag);
        let eta =
            self.designzeroed.dot(latent_beta) + self.nonlinear_offset_from_latent(latent_beta)?;
        let obs = evaluate_resolved_standard_family_observations(
            &self.likelihood,
            self.latent_cloglog_state.as_ref(),
            self.mixture_link_state.as_ref(),
            self.sas_link_state.as_ref(),
            &self.y,
            &self.weights,
            &eta,
        )
        .map_err(|e| e.to_string())?;

        let mut priorgrad = Array1::<f64>::zeros(latent_beta.len());
        let mut prior_neghess = Array2::<f64>::zeros((latent_beta.len(), latent_beta.len()));
        let mut prior_loglik = 0.0;
        for term in &self.bounded_terms {
            let (logp, grad, neghess, _, _) =
                bounded_prior_terms(latent_beta[term.col_idx], &term.prior)?;
            prior_loglik += logp;
            priorgrad[term.col_idx] += grad;
            prior_neghess[[term.col_idx, term.col_idx]] += neghess;
        }

        let mut hessian = xt_diag_x_dense(x_eff.view(), obs.neghessian_eta.view())?;
        let mut gradient = fast_atv(&x_eff, &obs.score);
        for term in &self.bounded_terms {
            let score_beta = self.design.column(term.col_idx).dot(&obs.score);
            hessian[[term.col_idx, term.col_idx]] -= score_beta * second_diag[term.col_idx];
        }
        hessian += &prior_neghess;
        gradient += &priorgrad;

        Ok((
            obs,
            hessian,
            gradient,
            prior_loglik,
            second_diag,
            third_diag,
            priorthird,
        ))
    }

    fn evaluation_from_latent(
        &self,
        latent_beta: &Array1<f64>,
    ) -> Result<
        (
            StandardFamilyObservationState,
            Array2<f64>,
            Array1<f64>,
            f64,
        ),
        String,
    > {
        let (obs, hessian, gradient, prior_loglik, _, _, _) =
            self.exacthessian_andgradient(latent_beta)?;
        Ok((obs, hessian, gradient, prior_loglik))
    }

    /// Exact `{D³H[u, v, e_a]}` over every coefficient axis of the latent joint
    /// Hessian `H = Eᵀ·diag(W)·E − diag_b(x_bᵀ·score·b'') + diag_b(P'')` (#2903).
    /// It is the Leibniz expansion of `Eᵀ·diag(W)·E` through the moving bounded
    /// columns of `E` and the likelihood curvature `W(η)` up to `W'''`, plus the
    /// bounded diagonal through `b⁽⁵⁾` and the prior's fifth derivative.
    fn joint_hessian_third_directional_all_axes(
        &self,
        latent_beta: &Array1<f64>,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let p = latent_beta.len();
        if d_beta_u_flat.len() != p || d_beta_v_flat.len() != p {
            return Err(SmoothError::dimension_mismatch(format!(
                "bounded linear family third directional derivative length mismatch: got {} and {}, expected {p}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len()
            ))
            .into());
        }
        let (obs, _, _, _, second_diag, third_diag, _) =
            self.exacthessian_andgradient(latent_beta)?;
        let w3 = &obs.neghessian_eta_third_derivative;
        let (_, jac_diag, _, _, _) = self.bounded_term_derivative_data(latent_beta)?;
        let x_eff = self.effective_design_for_latent(&jac_diag);
        let mut fourth_diag = Array1::<f64>::zeros(p);
        let mut fifth_diag = Array1::<f64>::zeros(p);
        let mut prior_fifth = Array1::<f64>::zeros(p);
        for term in &self.bounded_terms {
            let col = term.col_idx;
            let (_, _, _, _, _, d4b_dtheta4, d5b_dtheta5) =
                bounded_latent_derivatives(latent_beta[col], term.min, term.max);
            fourth_diag[col] = d4b_dtheta4;
            fifth_diag[col] = d5b_dtheta5;
            prior_fifth[col] = bounded_prior_neghess_third_derivative(latent_beta[col], &term.prior)?;
        }
        let scaled_bounded_columns = |scale: &dyn Fn(usize) -> f64| {
            let mut out = Array2::<f64>::zeros(x_eff.raw_dim());
            for term in &self.bounded_terms {
                let factor = scale(term.col_idx);
                if factor != 0.0 {
                    let mut col = out.column_mut(term.col_idx);
                    col.assign(&self.design.column(term.col_idx));
                    col.mapv_inplace(|value| value * factor);
                }
            }
            out
        };

        let u = d_beta_u_flat;
        let v = d_beta_v_flat;
        let w = &obs.neghessian_eta;
        let w1 = &obs.neghessian_eta_derivative;
        let w2 = &obs.neghessian_eta_second_derivative;
        let deta_u = x_eff.dot(u);
        let deta_v = x_eff.dot(v);
        let deta_uv = &deta_u * &deta_v;
        // The moving effective design: dE[s] scales bounded columns by b''·s,
        // d²E[s, t] by b'''·s·t and d³E[u, v, t] by b''''·u·v·t.
        let dx_u = scaled_bounded_columns(&|col| second_diag[col] * u[col]);
        let dx_v = scaled_bounded_columns(&|col| second_diag[col] * v[col]);
        let ddx_uv = scaled_bounded_columns(&|col| third_diag[col] * u[col] * v[col]);
        let ddeta_uv = dx_v.dot(u);
        let dw_u = w1 * &deta_u;
        let dw_v = w1 * &deta_v;
        let d2w_uv = &(w2 * &deta_uv) + &(w1 * &ddeta_uv);
        let d_score_u = -(w * &deta_u);
        let d_score_v = -(w * &deta_v);
        let dd_score_uv = -(w1 * &deta_uv) - &(w * &ddeta_uv);

        let mut axes = Vec::with_capacity(p);
        for axis in 0..p {
            let axis_bounded = self.bounded_terms.iter().any(|term| term.col_idx == axis);
            let deta_w = x_eff.column(axis).to_owned();
            let dx_w =
                scaled_bounded_columns(&|col| if col == axis { second_diag[col] } else { 0.0 });
            let ddx_uw = scaled_bounded_columns(&|col| {
                if col == axis { third_diag[col] * u[col] } else { 0.0 }
            });
            let ddx_vw = scaled_bounded_columns(&|col| {
                if col == axis { third_diag[col] * v[col] } else { 0.0 }
            });
            let dddx_uvw = scaled_bounded_columns(&|col| {
                if col == axis { fourth_diag[col] * u[col] * v[col] } else { 0.0 }
            });
            let ddeta_uw = dx_w.dot(u);
            let ddeta_vw = dx_w.dot(v);
            let dddeta_uvw = ddx_vw.dot(u);
            let second_first_sum =
                &(&(&ddeta_uv * &deta_w) + &(&ddeta_uw * &deta_v)) + &(&ddeta_vw * &deta_u);
            let d3w = &(&(w3 * &(&deta_uv * &deta_w)) + &(w2 * &second_first_sum))
                + &(w1 * &dddeta_uvw);
            let d2w_vw = &(w2 * &(&deta_v * &deta_w)) + &(w1 * &ddeta_vw);
            let d2w_uw = &(w2 * &(&deta_u * &deta_w)) + &(w1 * &ddeta_uw);
            let dw_w = w1 * &deta_w;

            let mut d3h = xt_diag_x_dense(x_eff.view(), d3w.view())?;
            add_symmetric_weighted_cross(&mut d3h, &dx_u, &d2w_vw, &x_eff);
            add_symmetric_weighted_cross(&mut d3h, &dx_v, &d2w_uw, &x_eff);
            add_symmetric_weighted_cross(&mut d3h, &ddx_uv, &dw_w, &x_eff);
            add_symmetric_weighted_cross(&mut d3h, &dx_u, &dw_w, &dx_v);
            if axis_bounded {
                add_symmetric_weighted_cross(&mut d3h, &dx_w, &d2w_uv, &x_eff);
                add_symmetric_weighted_cross(&mut d3h, &ddx_uw, &dw_v, &x_eff);
                add_symmetric_weighted_cross(&mut d3h, &ddx_vw, &dw_u, &x_eff);
                add_symmetric_weighted_cross(&mut d3h, &dx_u, &dw_v, &dx_w);
                add_symmetric_weighted_cross(&mut d3h, &dx_v, &dw_u, &dx_w);
                add_symmetric_weighted_cross(&mut d3h, &dddx_uvw, w, &x_eff);
                add_symmetric_weighted_cross(&mut d3h, &ddx_uv, w, &dx_w);
                add_symmetric_weighted_cross(&mut d3h, &ddx_uw, w, &dx_v);
                add_symmetric_weighted_cross(&mut d3h, &ddx_vw, w, &dx_u);
            }

            // Bounded diagonal: the third derivative of −(x_bᵀ·score)·b'' plus the
            // prior's fifth derivative, with D score[s] = −W·dη[s].
            let d_score_w = -(w * &deta_w);
            let dd_score_uw = -(w1 * &(&deta_u * &deta_w)) - &(w * &ddeta_uw);
            let dd_score_vw = -(w1 * &(&deta_v * &deta_w)) - &(w * &ddeta_vw);
            let ddd_score = -(w2 * &(&deta_uv * &deta_w))
                - &(w1 * &second_first_sum)
                - &(w * &dddeta_uvw);
            for term in &self.bounded_terms {
                let col = term.col_idx;
                let x_b = self.design.column(col);
                let u_b = u[col];
                let v_b = v[col];
                let w_b = if col == axis { 1.0 } else { 0.0 };
                d3h[[col, col]] -= x_b.dot(&ddd_score) * second_diag[col]
                    + (x_b.dot(&dd_score_uv) * w_b
                        + x_b.dot(&dd_score_uw) * v_b
                        + x_b.dot(&dd_score_vw) * u_b)
                        * third_diag[col]
                    + (x_b.dot(&d_score_u) * v_b * w_b
                        + x_b.dot(&d_score_v) * u_b * w_b
                        + x_b.dot(&d_score_w) * u_b * v_b)
                        * fourth_diag[col]
                    + x_b.dot(&obs.score) * fifth_diag[col] * u_b * v_b * w_b;
                d3h[[col, col]] += prior_fifth[col] * u_b * v_b * w_b;
            }
            axes.push(d3h);
        }
        Ok(axes)
    }
}

/// `Aᵀ·diag(c)·B + Bᵀ·diag(c)·A` over the rows of two same-shape designs.
fn add_symmetric_weighted_cross(
    target: &mut Array2<f64>,
    a: &Array2<f64>,
    c: &Array1<f64>,
    b: &Array2<f64>,
) {
    let cols = a.ncols();
    for i in 0..a.nrows() {
        let ci = c[i];
        if ci == 0.0 {
            continue;
        }
        for r in 0..cols {
            let ar = a[[i, r]];
            let br = b[[i, r]];
            for s in 0..cols {
                target[[r, s]] += ci * (ar * b[[i, s]] + br * a[[i, s]]);
            }
        }
    }
}

impl crate::custom_family::JeffreysThirdInformationDerivative for BoundedLinearFamily {
    fn third_directional_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        let latent_beta = &expect_single_block_state(block_states, "bounded linear family")?.beta;
        if specs.len() != 1 || specs[0].design.ncols() != latent_beta.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "bounded linear family third information derivative expects one block spec of width {}, got {} spec(s)",
                latent_beta.len(),
                specs.len()
            ))
            .into());
        }
        self.joint_hessian_third_directional_all_axes(latent_beta, d_beta_u_flat, d_beta_v_flat)
            .map(Some)
    }
}

impl JeffreysArming for BoundedLinearFamily {
    fn with_jeffreys_armed(
        &self,
        evidence: Option<&gam_problem::jeffreys_arming::JeffreysArmingEvidence>,
    ) -> Self {
        Self {
            jeffreys_armed: evidence.is_some(),
            ..self.clone()
        }
    }
}

impl CustomFamily for BoundedLinearFamily {
    // A bounded coefficient can run to its bound on separated data. The
    // self-limiting Jeffreys/Firth curvature bounds it there, but it is armed
    // only when the unarmed fit proves it is needed (#979).
    fn joint_jeffreys_term_required(&self) -> bool {
        self.jeffreys_armed
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let latent_beta = &expect_single_block_state(block_states, "bounded linear family")?.beta;
        let (obs, hessian, gradient, prior_loglik) = self.evaluation_from_latent(latent_beta)?;
        Ok(FamilyEvaluation {
            log_likelihood: obs.log_likelihood + prior_loglik,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            }],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let latent_beta = &expect_single_block_state(block_states, "bounded linear family")?.beta;
        let (_, hessian, _, _) = self.evaluation_from_latent(latent_beta)?;
        Ok(Some(hessian))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        expect_block_idx_zero(block_idx, "bounded linear family", "")?;
        self.exact_newton_joint_hessian_directional_derivative(block_states, d_beta)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let latent_beta = &expect_single_block_state(block_states, "bounded linear family")?.beta;
        if d_beta_flat.len() != latent_beta.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "bounded linear family directional derivative length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                latent_beta.len()
            ))
            .into());
        }

        let (obs, _, _, _, second_diag, third_diag, priorthird) =
            self.exacthessian_andgradient(latent_beta)?;

        let (_, jac_diag, _, _, _) = self.bounded_term_derivative_data(latent_beta)?;
        let x_eff = self.effective_design_for_latent(&jac_diag);
        let deta = x_eff.dot(d_beta_flat);
        let d_neghess_eta = &obs.neghessian_eta_derivative * &deta;

        let mut dx_eff = Array2::<f64>::zeros(x_eff.raw_dim());
        for term in &self.bounded_terms {
            let scale = second_diag[term.col_idx] * d_beta_flat[term.col_idx];
            if scale != 0.0 {
                let mut col = dx_eff.column_mut(term.col_idx);
                col.assign(&self.design.column(term.col_idx));
                col.mapv_inplace(|v| v * scale);
            }
        }

        let mut dhessian = xt_diag_x_dense(x_eff.view(), d_neghess_eta.view())?;
        let mut wxdx = Array2::<f64>::zeros((x_eff.ncols(), x_eff.ncols()));
        for i in 0..x_eff.nrows() {
            let wi = obs.neghessian_eta[i];
            if wi == 0.0 {
                continue;
            }
            for a in 0..x_eff.ncols() {
                let xa = x_eff[[i, a]];
                for b in 0..x_eff.ncols() {
                    wxdx[[a, b]] += wi * (dx_eff[[i, a]] * x_eff[[i, b]] + xa * dx_eff[[i, b]]);
                }
            }
        }
        dhessian += &wxdx;

        let d_score = -&obs.neghessian_eta * &deta;
        for term in &self.bounded_terms {
            let score_beta = self.design.column(term.col_idx).dot(&obs.score);
            let d_score_beta = self.design.column(term.col_idx).dot(&d_score);
            dhessian[[term.col_idx, term.col_idx]] -= d_score_beta * second_diag[term.col_idx]
                + score_beta * third_diag[term.col_idx] * d_beta_flat[term.col_idx];
            dhessian[[term.col_idx, term.col_idx]] +=
                priorthird[term.col_idx] * d_beta_flat[term.col_idx];
        }

        Ok(Some(dhessian))
    }

    /// Exact `D²H[u, v]` of the latent joint Hessian
    /// `H = Eᵀ·diag(W)·E − diag_b(x_bᵀ·score·b'') + diag_b(P'')`, where `E` scales
    /// each bounded column by `b'` (#2903). The family keeps the β-independent
    /// default objective, so the trait's block default would return zeros for a
    /// Hessian that moves through both the bounded transform and the likelihood
    /// curvature `W(η)`.
    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let latent_beta = &expect_single_block_state(block_states, "bounded linear family")?.beta;
        let p = latent_beta.len();
        if d_beta_u_flat.len() != p || d_betav_flat.len() != p {
            return Err(SmoothError::dimension_mismatch(format!(
                "bounded linear family second directional derivative length mismatch: got {} and {}, expected {p}",
                d_beta_u_flat.len(),
                d_betav_flat.len()
            ))
            .into());
        }

        let (obs, _, _, _, second_diag, third_diag, _) =
            self.exacthessian_andgradient(latent_beta)?;
        let (_, jac_diag, _, _, _) = self.bounded_term_derivative_data(latent_beta)?;
        let x_eff = self.effective_design_for_latent(&jac_diag);
        let deta_u = x_eff.dot(d_beta_u_flat);
        let deta_v = x_eff.dot(d_betav_flat);

        // The moving effective design: dE[w] scales bounded columns by b''·w, and
        // d²E[u, v] by b'''·u·v.
        let scaled_bounded_columns = |scale: &dyn Fn(usize) -> f64| {
            let mut out = Array2::<f64>::zeros(x_eff.raw_dim());
            for term in &self.bounded_terms {
                let factor = scale(term.col_idx);
                if factor != 0.0 {
                    let mut col = out.column_mut(term.col_idx);
                    col.assign(&self.design.column(term.col_idx));
                    col.mapv_inplace(|value| value * factor);
                }
            }
            out
        };
        let dx_u = scaled_bounded_columns(&|col| second_diag[col] * d_beta_u_flat[col]);
        let dx_v = scaled_bounded_columns(&|col| second_diag[col] * d_betav_flat[col]);
        let ddx_uv =
            scaled_bounded_columns(&|col| third_diag[col] * d_beta_u_flat[col] * d_betav_flat[col]);
        // d²η[u, v] = Σ_b x_b·b''·u_b·v_b.
        let ddeta_uv = dx_v.dot(d_beta_u_flat);

        let w = &obs.neghessian_eta;
        let w1 = &obs.neghessian_eta_derivative;
        let w2 = &obs.neghessian_eta_second_derivative;
        let diag_eta = w2 * &(&deta_u * &deta_v) + w1 * &ddeta_uv;
        let mut d2h = xt_diag_x_dense(x_eff.view(), diag_eta.view())?;
        add_symmetric_weighted_cross(&mut d2h, &dx_v, &(w1 * &deta_u), &x_eff);
        add_symmetric_weighted_cross(&mut d2h, &dx_u, &(w1 * &deta_v), &x_eff);
        add_symmetric_weighted_cross(&mut d2h, &ddx_uv, w, &x_eff);
        add_symmetric_weighted_cross(&mut d2h, &dx_u, w, &dx_v);

        // Bounded diagonal: the second derivative of −(x_bᵀ·score)·b'' plus the
        // prior's fourth derivative, with D score[w] = −W·dη[w].
        let dd_score_uv = -(w1 * &(&deta_u * &deta_v)) - &(w * &ddeta_uv);
        let d_score_u = -(w * &deta_u);
        let d_score_v = -(w * &deta_v);
        for term in &self.bounded_terms {
            let col = term.col_idx;
            let x_b = self.design.column(col);
            let (_, _, _, _, _, d4b_dtheta4, _) =
                bounded_latent_derivatives(latent_beta[col], term.min, term.max);
            let (_, _, _, _, prior_neghess_second_derivative) =
                bounded_prior_terms(latent_beta[col], &term.prior)?;
            let u_b = d_beta_u_flat[col];
            let v_b = d_betav_flat[col];
            d2h[[col, col]] -= x_b.dot(&dd_score_uv) * second_diag[col]
                + x_b.dot(&d_score_u) * third_diag[col] * v_b
                + x_b.dot(&d_score_v) * third_diag[col] * u_b
                + x_b.dot(&obs.score) * d4b_dtheta4 * u_b * v_b;
            d2h[[col, col]] += prior_neghess_second_derivative * u_b * v_b;
        }

        Ok(Some(d2h))
    }

    /// Every observation branch the bounded family admits carries a closed-form
    /// `W'''`: the standard GLM rows directly, and every binomial link through the
    /// Bernoulli tail kernels or the inverse-link density's fourth derivative
    /// (#2903).
    fn jeffreys_third_information_derivative(
        &self,
    ) -> Option<&dyn crate::custom_family::JeffreysThirdInformationDerivative> {
        if matches!(
            self.likelihood.spec.response,
            ResponseFamily::Beta { .. } | ResponseFamily::RoystonParmar
        ) {
            None
        } else {
            Some(self)
        }
    }

    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if block_states.is_empty() {
            return Ok((
                DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                    self.designzeroed.clone(),
                )),
                self.offset.clone(),
            ));
        }
        let offset = self.nonlinear_offset_from_latent(
            &expect_single_block_state(block_states, "bounded linear family")?.beta,
        )?;
        let x = if spec.design.ncols() == self.designzeroed.ncols() {
            self.designzeroed.clone()
        } else {
            return Err(SmoothError::dimension_mismatch(
                "bounded linear family design column mismatch",
            )
            .into());
        };
        Ok((
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x)),
            offset,
        ))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    /// The latent chart is not a basis (#2748).
    ///
    /// `post_update_block_beta` below clamps `theta[term.col_idx]` PER
    /// COORDINATE, and `block_geometry` above refuses any spec whose width is
    /// not `designzeroed.ncols()`. Both statements are about these exact
    /// coordinates, and neither survives `beta -> V' beta`: a rotation mixes a
    /// bounded latent coordinate with an unbounded one, so the clamp would
    /// confine the wrong thing, and it narrows the width the rebuild demands.
    /// The derived default cannot see either hook — this family states its
    /// feasible set through the clamp rather than through
    /// `block_linear_constraints` — so the declaration is explicit here.
    fn block_coefficient_coordinate(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        block_spec: &ParameterBlockSpec,
    ) -> gam_problem::CoefficientCoordinate {
        // This family carries exactly ONE block and both hooks that make its
        // coordinate structural are indexed against that block, so the answer is
        // the same for every question it can be asked. Say so when the question
        // does not describe the block this family holds, rather than answering
        // silently about a coordinate that is not its own: a mismatched index or
        // width IS the desynchronisation this declaration exists to prevent, and
        // `Structural` is the safe answer to it too.
        if block_index != 0 || block_spec.design.ncols() != self.designzeroed.ncols() {
            log::debug!(
                "bounded linear family: coefficient coordinate asked for block {block_index} \
                 at spec width {} ({} block state(s) supplied) while this family carries one \
                 block of width {}; the coordinate is structural either way",
                block_spec.design.ncols(),
                block_states.len(),
                self.designzeroed.ncols(),
            );
        }
        gam_problem::CoefficientCoordinate::Structural
    }

    /// Confine every bounded coefficient's latent coordinate to the range
    /// where [`bounded_latent_injective_limit`] says the interval map is still
    /// invertible.
    ///
    /// Without this the inner solve cannot certify a fit whose constrained
    /// optimum sits ON a box bound. Once `theta` crosses the saturation point
    /// the proposal keeps moving `theta` by order one while `beta`, the
    /// linear predictor, the log-likelihood and the objective are all frozen —
    /// the line search accepts the step (it does not increase the objective),
    /// and the blockwise convergence certificate, which measures the step on
    /// `theta` against `inner_tol*(1 + |theta|_inf)`, never fires. The loop
    /// then spends its whole `inner_max_cycles` budget and the no-smoothing
    /// path returns `InnerSolveNotConverged` with that exit as its terminal
    /// reason (gam#2705 group C: three `bounded()` anchors whose optimum is a
    /// bound — `bounded(x,0,1)` on `y = 2 + 5x`, and two `bounded(x,0,·)`
    /// anchors whose slope binds at the lower bound).
    ///
    /// Clamping changes no fitted quantity: every clipped `theta` maps to the
    /// same `beta` the clamp maps to. What it changes is reachability of the
    /// fixed point — the next proposal is clipped to the same coordinate, the
    /// accepted step is exactly zero, and the loop certifies at the boundary
    /// optimum instead of grinding.
    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        expect_block_idx_zero(
            block_index,
            "bounded linear family",
            " for post-update beta",
        )?;
        // A clamp is only meaningful on coefficients that really are this
        // block's, under this block's spec — so the proposal has to agree with
        // both the spec's width and the width of the state it is replacing.
        let current = expect_single_block_state(block_states, "bounded linear family")?;
        if beta.len() != block_spec.design.ncols() || beta.len() != current.beta.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "bounded linear family post-update beta width mismatch: got {}, expected {} \
                 (spec) / {} (current state)",
                beta.len(),
                block_spec.design.ncols(),
                current.beta.len()
            ))
            .into());
        }
        let limit = bounded_latent_injective_limit();
        let mut clamped = beta;
        for term in &self.bounded_terms {
            let theta = clamped[term.col_idx];
            clamped[term.col_idx] = theta.clamp(-limit, limit);
        }
        Ok(clamped)
    }

    fn block_geometry_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
        d_beta: &Array1<f64>,
    ) -> Result<Option<BlockGeometryDirectionalDerivative>, String> {
        expect_block_idx_zero(
            block_idx,
            "bounded linear family",
            " for geometry derivative",
        )?;
        expect_single_block_state(block_states, "bounded linear family")?;
        if d_beta.len() != spec.design.ncols() {
            return Err(SmoothError::dimension_mismatch(format!(
                "bounded linear family geometry derivative direction mismatch: got {}, expected {}",
                d_beta.len(),
                spec.design.ncols()
            ))
            .into());
        }
        let (_, jac_diag, _, _, _) = self.bounded_term_derivative_data(&block_states[0].beta)?;
        let mut d_offset = Array1::<f64>::zeros(self.offset.len());
        let has_drift = self
            .bounded_terms
            .iter()
            .any(|term| jac_diag[term.col_idx] != 0.0 && d_beta[term.col_idx] != 0.0);
        if !has_drift {
            return Ok(Some(BlockGeometryDirectionalDerivative {
                d_design: None,
                d_offset,
            }));
        }
        for term in &self.bounded_terms {
            let col = term.col_idx;
            let drift = jac_diag[col] * d_beta[col];
            if drift != 0.0 {
                d_offset.scaled_add(drift, &self.design.column(col));
            }
        }
        Ok(Some(BlockGeometryDirectionalDerivative {
            d_design: None,
            d_offset,
        }))
    }
}

fn xt_diag_x_dense(x: ArrayView2<'_, f64>, w: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
    if x.nrows() != w.len() {
        return Err(SmoothError::dimension_mismatch("xt_diag_x_dense row mismatch").into());
    }
    let (n, p) = x.dim();
    if n == 0 || p == 0 {
        return Ok(Array2::<f64>::zeros((p, p)));
    }

    const STREAMING_BYTES_THRESHOLD: usize = 8 * 1024 * 1024;
    let dense_work_bytes = n
        .checked_mul(p)
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
        .unwrap_or(usize::MAX);
    if dense_work_bytes <= STREAMING_BYTES_THRESHOLD {
        let mut weighted = x.to_owned();
        ndarray::Zip::from(weighted.rows_mut())
            .and(w)
            .par_for_each(|mut row, wi| row *= *wi);
        return Ok(fast_atb(&x, &weighted));
    }

    let chunkrows = gam_runtime::resource::byte_balanced_row_chunk(p, n);
    let mut weighted_chunk = Array2::<f64>::zeros((chunkrows, p));
    let mut out = Array2::<f64>::zeros((p, p));
    for row_start in (0..n).step_by(chunkrows) {
        let rows = (n - row_start).min(chunkrows);
        let x_chunk = x.slice(s![row_start..row_start + rows, ..]);
        {
            let mut chunk = weighted_chunk.slice_mut(s![0..rows, ..]);
            for local_row in 0..rows {
                let scale = w[row_start + local_row];
                if scale == 0.0 {
                    chunk.row_mut(local_row).fill(0.0);
                    continue;
                }
                for col in 0..p {
                    chunk[[local_row, col]] = x_chunk[[local_row, col]] * scale;
                }
            }
        }
        out += &fast_atb(&x_chunk, &weighted_chunk.slice(s![0..rows, ..]));
    }
    Ok(out)
}

/// `tr(C·S)` for a posterior covariance `C` and a PSD penalty `S`, accumulated
/// as a sum of squares.
///
/// The obvious spelling — `Σ_ij C_ij·S_ji` — sums mixed-sign products of two
/// PSD matrices, and on a penalized fit it cancels catastrophically: measured
/// `Σ|terms| / |Σ terms|` on a second-difference penalty rose from 8.4e1 at
/// λ=1 to **3.9e13 at λ=1e12**, where the computed trace has no significant
/// digits left. That is not a tolerance problem, and no backward-error bound
/// repairs it — a forward-error bound is linear in `Σ|terms|`, so a *correct*
/// bound over a cancelling sum is correctly enormous (issue #2470).
///
/// Factor both operands instead. With `S = RᵀR` (`penalty_matrix_root`) and
/// `C = L·Lᵀ` (Cholesky; `C` is a principal submatrix of an SPD inverse, hence
/// SPD),
///
/// ```text
/// tr(C·S) = tr(C·RᵀR) = tr(R·C·Rᵀ) = ‖R·L‖_F²
/// ```
///
/// and every summand is a square. Measured ratio: **exactly 1.000 at every λ**.
///
/// Factoring only the penalty is NOT enough, and the near-miss is worth
/// recording: accumulating `Σ_i (R·C·Rᵀ)_ii` gives a clean *outer* sum, because
/// those diagonal entries are non-negative — but each one is itself the
/// mixed-sign quadratic form `rᵢᵀ·C·rᵢ`, which cancels by **6.4e13** at λ=1e12,
/// worse than the elementwise form. The cancellation moves one level down,
/// where an error bound taken over the outer sum cannot see it. A sum of
/// squares is the only shape with nothing to cancel at either level.
fn trace_of_factored_product(
    penalty_root: &Array2<f64>,
    covariance: &Array2<f64>,
) -> Result<f64, String> {
    let m = covariance.nrows();
    if m != covariance.ncols() {
        return Err(
            SmoothError::dimension_mismatch("trace_of_factored_product needs a square covariance")
                .into(),
        );
    }
    if penalty_root.ncols() != m {
        return Err(SmoothError::dimension_mismatch(
            "trace_of_factored_product penalty root and covariance disagree on the block width",
        )
        .into());
    }
    if penalty_root
        .iter()
        .chain(covariance.iter())
        .any(|value| !value.is_finite())
    {
        return Err("trace_of_factored_product requires finite factors".to_string());
    }
    if m == 0 || penalty_root.nrows() == 0 {
        return Ok(0.0);
    }
    // `C = V·W·Vᵀ` with `W ≥ 0` (the block is a principal submatrix of an SPD
    // inverse), so `L = V·W^{1/2}` is a PSD factor. A negative eigenvalue here
    // is roundoff on an exactly-zero one, not a direction of negative variance;
    // clamping at zero is what keeps `L` real, and a materially negative one
    // would have been refused when the covariance was certified.
    use gam_linalg::faer_ndarray::FaerEigh as _;
    let (eigenvalues, eigenvectors) = covariance.eigh(faer::Side::Lower).map_err(|err| {
        format!("trace_of_factored_product could not factor the covariance block: {err}")
    })?;
    let mut covariance_factor = eigenvectors;
    for (col, eigenvalue) in eigenvalues.iter().enumerate() {
        let scale = eigenvalue.max(0.0).sqrt();
        covariance_factor.column_mut(col).mapv_inplace(|v| v * scale);
    }
    let scaled = penalty_root.dot(&covariance_factor);
    let mut trace = gam_linalg::utils::KahanSum::default();
    for value in scaled.iter() {
        let term = value * value;
        if !term.is_finite() {
            return Err("trace_of_factored_product term is not representable".to_string());
        }
        trace.add(term);
    }
    let trace = trace.sum();
    if !trace.is_finite() {
        return Err("trace_of_factored_product sum is not representable".to_string());
    }
    Ok(trace)
}

fn certify_bounded_edf_interval(
    value: f64,
    lower: f64,
    upper: f64,
    dimension: usize,
    label: &str,
) -> Result<f64, EstimationError> {
    if !(value.is_finite() && lower.is_finite() && upper.is_finite() && lower <= upper) {
        crate::bail_invalid_estim!(
            "{label} has invalid EDF interval/value: value={value}, interval=[{lower}, {upper}]"
        );
    }
    let scale = 1.0_f64.max(value.abs()).max(lower.abs()).max(upper.abs());
    // A dense trace has p^2 rounded products/additions. This is a backward-
    // error allowance for that declared operation count, not a statistical
    // projection: values materially outside the mathematical interval fail.
    let allowed = 256.0 * f64::EPSILON * (dimension.max(1) as f64).powi(2) * scale;
    if value < lower {
        if lower - value <= allowed {
            return Ok(lower);
        }
    } else if value > upper {
        if value - upper <= allowed {
            return Ok(upper);
        }
    } else {
        return Ok(value);
    }
    crate::bail_invalid_estim!(
        "{label}={value} lies outside [{lower}, {upper}] by more than the dense-trace backward-error allowance {allowed}"
    )
}

fn exact_bounded_edf(
    penalties: &[PenaltySpec],
    lambdas: &Array1<f64>,
    latent_cov: &Array2<f64>,
) -> Result<(Vec<f64>, Vec<f64>, f64), EstimationError> {
    if penalties.len() != lambdas.len() {
        crate::bail_invalid_estim!(
            "bounded EDF penalty/lambda mismatch: {} penalties vs {} lambdas",
            penalties.len(),
            lambdas.len()
        );
    }
    if latent_cov.nrows() != latent_cov.ncols() {
        crate::bail_invalid_estim!("bounded EDF covariance must be square");
    }

    let p = latent_cov.nrows();
    let mut s_lambda = Array2::<f64>::zeros((p, p));
    let mut edf_by_block = Vec::with_capacity(penalties.len());
    // Raw per-block penalty trace tr_kk = λ_kk·tr(H⁻¹S_kk) (issue #1219).
    let mut penalty_block_trace = Vec::with_capacity(penalties.len());
    let mut trace_sum = gam_linalg::utils::KahanSum::default();

    for (k, ps) in penalties.iter().enumerate() {
        let lambda_k = lambdas[k];
        if !(lambda_k.is_finite() && lambda_k >= 0.0) {
            crate::bail_invalid_estim!(
                "bounded EDF smoothing strength at block {k} must be finite and non-negative, got {lambda_k}"
            );
        }
        match ps {
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                s_lambda
                    .slice_mut(ndarray::s![col_range.clone(), col_range.clone()])
                    .scaled_add(lambda_k, local);
                // Compute penalty rank from the block-local matrix directly.
                let penalty_rank =
                    local
                        .nrows()
                        .saturating_sub(estimate_penalty_nullity(local).map_err(|e| {
                            EstimationError::InvalidInput(format!("bounded EDF rank failed: {e}"))
                        })?);
                // Trace only involves the block slice of latent_cov.
                let cov_block = latent_cov.slice(ndarray::s![col_range.clone(), col_range.clone()]);
                // The rank oracle is deliberately left on `estimate_penalty_nullity`
                // above: this root is used for the ACCUMULATION only. The two
                // disagree — `penalty_matrix_root` cuts at `n·ε·λmax` and
                // `spectral_tolerance` at `n·1e-10·λmax`, a factor of 4.5e5 — so
                // adopting the root's rank here would silently move published
                // per-block EDF, which is a separate, measured increment (#2469).
                let penalty_root = gam_solve::estimate::reml::reml_outer_engine::penalty_matrix_root(local)
                    .map_err(EstimationError::InvalidInput)?;
                let trace_k = lambda_k
                    * trace_of_factored_product(&penalty_root, &cov_block.to_owned())
                        .map_err(EstimationError::InvalidInput)?;
                trace_sum.add(trace_k);
                penalty_block_trace.push(trace_k);
                let p_k = penalty_rank as f64;
                edf_by_block.push(certify_bounded_edf_interval(
                    p_k - trace_k,
                    0.0,
                    p_k,
                    p,
                    &format!("bounded EDF block {k}"),
                )?);
            }
            PenaltySpec::Dense(m) | PenaltySpec::DenseWithMean { matrix: m, .. } => {
                s_lambda.scaled_add(lambda_k, m);
                let penalty_rank = p.saturating_sub(estimate_penalty_nullity(m).map_err(|e| {
                    EstimationError::InvalidInput(format!("bounded EDF rank failed: {e}"))
                })?);
                let penalty_root = gam_solve::estimate::reml::reml_outer_engine::penalty_matrix_root(m)
                    .map_err(EstimationError::InvalidInput)?;
                let trace_k = lambda_k
                    * trace_of_factored_product(&penalty_root, latent_cov)
                        .map_err(EstimationError::InvalidInput)?;
                trace_sum.add(trace_k);
                penalty_block_trace.push(trace_k);
                let p_k = penalty_rank as f64;
                edf_by_block.push(certify_bounded_edf_interval(
                    p_k - trace_k,
                    0.0,
                    p_k,
                    p,
                    &format!("bounded EDF block {k}"),
                )?);
            }
        }
    }

    let nullity_total = estimate_penalty_nullity(&s_lambda)
        .map_err(|e| EstimationError::InvalidInput(format!("bounded EDF nullity failed: {e}")))?
        as f64;
    let trace_sum = trace_sum.sum();
    let edf_total = certify_bounded_edf_interval(
        p as f64 - trace_sum,
        nullity_total,
        p as f64,
        p,
        "bounded total EDF",
    )?;
    Ok((edf_by_block, penalty_block_trace, edf_total))
}

/// Certified, unperturbed posterior-precision inverse for a bounded fit.
/// A reported covariance exists only at a strict posterior maximum, hence the
/// precision must be SPD. Singular and indefinite modes are refused; projecting
/// them into a pseudo-covariance would silently report zero uncertainty in an
/// unidentified direction.
/// `σ̂ = sqrt(deviance / residual_dof)` for a profiled-Gaussian scale, with the
/// residual degrees of freedom certified rather than floored.
///
/// `residual_dof = n − edf` is the information left over once the fit has spent
/// its effective parameters. At `n − edf ≤ 0` the fit has at least as many
/// effective parameters as observations and there is no residual information to
/// estimate a scale from: the quantity does not exist, and saying so is the only
/// honest answer.
///
/// Flooring the denominator returns a number anyway — and because a floor only
/// ever *raises* the denominator, that number is biased **low**. Every standard
/// error, Wald statistic and interval width computed from it is then too narrow,
/// which is the direction nothing downstream can detect: a too-wide interval is
/// visibly useless, a too-narrow one just looks confident.
///
/// The caller states `residual_dof` rather than passing `n` and `edf`, because
/// the routes legitimately differ on what they subtract — a fit that computed no
/// inference has spent no measured effective parameters and uses `n`.
fn certified_profiled_gaussian_scale(
    deviance: f64,
    residual_dof: f64,
    label: &str,
) -> Result<f64, EstimationError> {
    if !(residual_dof.is_finite() && residual_dof > 0.0) {
        crate::bail_invalid_estim!(
            "{label} residual degrees of freedom must be finite and positive, got {residual_dof}; \
             a fit with no residual information cannot report a scale"
        );
    }
    if !(deviance.is_finite() && deviance >= 0.0) {
        crate::bail_invalid_estim!(
            "{label} deviance must be finite and non-negative, got {deviance}"
        );
    }
    let variance = deviance / residual_dof;
    if !variance.is_finite() {
        crate::bail_invalid_estim!(
            "{label} residual variance is not representable: {deviance}/{residual_dof}"
        );
    }
    Ok(variance.sqrt())
}

fn certified_bounded_posterior_covariance(
    precision: &Array2<f64>,
    label: &'static str,
) -> Result<Array2<f64>, EstimationError> {
    gam_linalg::utils::certified_spd_inverse(precision, label)
        .map(gam_linalg::utils::CertifiedSpdInverse::into_inverse)
        .map_err(|error| {
            EstimationError::InvalidInput(format!(
                "bounded posterior covariance requires an exact SPD precision: {error}"
            ))
        })
}

fn transform_bounded_latent_precision_to_user_internal(
    latent_precision: &Array2<f64>,
    jac_diag: &Array1<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let p = latent_precision.nrows();
    if latent_precision.ncols() != p || jac_diag.len() != p {
        crate::bail_invalid_estim!(
            "bounded precision transform dimension mismatch: precision is {}x{}, jacobian has {} entries",
            latent_precision.nrows(),
            latent_precision.ncols(),
            jac_diag.len()
        );
    }
    let mut out = latent_precision.clone();
    for i in 0..p {
        let scale = jac_diag[i];
        if !scale.is_finite() || scale <= 0.0 {
            crate::bail_invalid_estim!(
                "bounded precision transform requires a positive finite coefficient jacobian; column {i} has {scale}"
            );
        }
        if scale != 1.0 {
            out.row_mut(i).mapv_inplace(|v| v / scale);
            out.column_mut(i).mapv_inplace(|v| v / scale);
        }
    }
    Ok(out)
}

/// `M·J·C·J·Mᵀ` of a latent coefficient-space bilinear form, with `J = diag(dβ/dθ)`
/// the bounded transform's Jacobian and `M` the conditioning back-transform: the
/// congruence that takes the latent conditional covariance to user scale (#2903).
fn bounded_latent_bilinear_to_user(
    latent: &Array2<f64>,
    jac_diag: &Array1<f64>,
    conditioning: &LinearFitConditioning,
) -> Result<Array2<f64>, EstimationError> {
    let p = latent.nrows();
    if latent.ncols() != p || jac_diag.len() != p {
        crate::bail_invalid_estim!(
            "bounded bilinear transform dimension mismatch: form is {}x{}, jacobian has {} entries",
            latent.nrows(),
            latent.ncols(),
            jac_diag.len()
        );
    }
    let mut scaled = latent.clone();
    for i in 0..p {
        let scale = jac_diag[i];
        scaled.row_mut(i).mapv_inplace(|value| value * scale);
        scaled.column_mut(i).mapv_inplace(|value| value * scale);
    }
    let mut left = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        left.column_mut(j)
            .assign(&conditioning.backtransform_beta(&scaled.column(j).to_owned()));
    }
    let mut out = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        out.row_mut(i)
            .assign(&conditioning.backtransform_beta(&left.row(i).to_owned()));
    }
    Ok(out)
}

fn fit_bounded_term_collection_with_design(
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    heuristic_log_lambdas: Option<&[f64]>,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let conditioning_cols: Vec<usize> = spec
        .linear_terms
        .iter()
        .enumerate()
        .filter_map(|(j, linear)| {
            (!linear.double_penalty).then_some(design.intercept_range.end + j)
        })
        .collect();
    let conditioning = LinearFitConditioning::from_columns(design, &conditioning_cols);
    let dense_design = design.design.to_dense_cow();
    let fit_design = conditioning.apply_to_design(&dense_design);
    let fit_penalties = conditioning
        .transform_blockwise_penalties_to_internal(&design.penalties, design.design.ncols());
    if design.linear_constraints.is_some() {
        crate::bail_invalid_estim!(
            "bounded() terms are not yet compatible with explicit linear constraints"
        );
    }
    let mut bounded_terms = Vec::<BoundedLinearTermMeta>::new();
    for (j, term) in spec.linear_terms.iter().enumerate() {
        if term.double_penalty
            && matches!(
                term.coefficient_geometry,
                LinearCoefficientGeometry::Bounded { .. }
            )
        {
            crate::bail_invalid_estim!(
                "bounded linear term '{}' cannot also use double_penalty",
                term.name
            );
        }
        if let LinearCoefficientGeometry::Bounded { min, max, prior } =
            term.coefficient_geometry.clone()
        {
            let col_idx = design.intercept_range.end + j;
            let (min_internal, max_internal) = conditioning.internal_bounds_for(col_idx, min, max);
            bounded_terms.push(BoundedLinearTermMeta {
                col_idx,
                min: min_internal,
                max: max_internal,
                prior,
            });
        }
    }
    if bounded_terms.is_empty() {
        crate::bail_invalid_estim!("internal bounded fit path called with no bounded terms");
    }

    let mut designzeroed = fit_design.clone();
    let mut initial_beta = Array1::<f64>::zeros(fit_design.ncols());
    for term in &bounded_terms {
        designzeroed.column_mut(term.col_idx).fill(0.0);
        initial_beta[term.col_idx] = 0.0;
    }

    let initial_log_lambdas = heuristic_log_lambdas
        .map(|vals| Array1::from_vec(vals.to_vec()))
        .unwrap_or_else(|| Array1::zeros(fit_penalties.len()));
    if initial_log_lambdas.len() != fit_penalties.len() {
        crate::bail_invalid_estim!(
            "heuristic log-lambda length mismatch for bounded model: got {}, expected {}",
            initial_log_lambdas.len(),
            fit_penalties.len()
        );
    }

    let glm_likelihood = gam_spec::GlmLikelihoodSpec::canonical(family);
    let resolved_likelihood_scale = glm_likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let is_beta_logistic = glm_likelihood.spec.is_binomial_beta_logistic();
    let family_adapter = BoundedLinearFamily {
        likelihood: glm_likelihood.clone(),
        latent_cloglog_state: options.latent_cloglog,
        mixture_link_state: options
            .mixture_link
            .clone()
            .as_ref()
            .map(state_fromspec)
            .transpose()
            .map_err(EstimationError::InvalidInput)?,
        sas_link_state: options
            .sas_link
            .map(|spec| {
                if is_beta_logistic {
                    state_from_beta_logisticspec(spec)
                } else {
                    state_from_sasspec(spec)
                }
            })
            .transpose()
            .map_err(EstimationError::InvalidInput)?,
        y: y.to_owned(),
        weights: weights.to_owned(),
        design: fit_design.clone(),
        designzeroed: designzeroed.clone(),
        offset: offset.to_owned(),
        bounded_terms: bounded_terms.clone(),
        jeffreys_armed: true,
    };
    let blockspec = ParameterBlockSpec {
        name: "eta".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(designzeroed)),
        offset: offset.to_owned(),
        penalties: fit_penalties
            .iter()
            .map(|ps| match ps {
                PenaltySpec::Block {
                    local, col_range, ..
                } => PenaltyMatrix::Blockwise {
                    local: local.clone(),
                    col_range: col_range.clone(),
                    total_dim: design.design.ncols(),
                },
                PenaltySpec::Dense(m) | PenaltySpec::DenseWithMean { matrix: m, .. } => {
                    PenaltyMatrix::Dense(m.clone())
                }
            })
            .collect(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas,
        initial_beta: Some(initial_beta),
        gauge_priority: 100,
        // Report the true β-dependent Jacobian (bounded columns scaled by
        // dβ/dθ) to the identifiability audit so it does not mistake the
        // deliberately-zeroed placeholder columns for a structural rank
        // deficiency. The inner solve still drives η through the family
        // adapter, so this does not affect the fit geometry.
        jacobian_callback: Some(Arc::new(BoundedEffectiveJacobian {
            design: fit_design.clone(),
            bounded_terms: bounded_terms.clone(),
        })),
        stacked_design: None,
        stacked_offset: None,
    };
    let fit = fit_custom_family_arming_on_evidence(
        &family_adapter,
        &[blockspec],
        &BlockwiseFitOptions {
            inner_tol: options.tol,
            outer_max_iter: options.max_iter,
            outer_tol: options.tol,
            // The bounded path builds its own user-scale covariance below by
            // inverting the user-scale penalised Hessian (delta-method through
            // the bounded transform's Jacobian + the conditioning map), so it
            // does not consume the inner solver's optional canonical-space
            // `covariance_conditional`. Inverting the reported precision
            // directly guarantees `inv(penalized_hessian) == covariance` and
            // works on every bounded fit — including the common no-smoothing
            // path where the inner solve surfaces no covariance at all (the
            // gam#854 "bounded fit emits no user-scale covariance" symptom).
            // A penalized fit with inference on still requests the latent
            // conditional covariance: it is the `V_cond` through which
            // `fit_custom_family` mints the first-order smoothing correction from
            // the certified outer rho-Hessian, pushed to user scale below (#2903).
            compute_covariance: options.compute_inference && !fit_penalties.is_empty(),
            ..BlockwiseFitOptions::default()
        },
    )
    .map_err(EstimationError::CustomFamily)?;

    let latent_beta = fit.block_states[0].beta.clone();
    let (beta_user_internal, jac_diag) = family_adapter
        .user_beta_and_jacobian(&latent_beta)
        .map_err(EstimationError::InvalidInput)?;
    let beta_user = conditioning.backtransform_beta(&beta_user_internal);

    let (eta_state, h_data, _, _) = family_adapter
        .evaluation_from_latent(&latent_beta)
        .map_err(EstimationError::InvalidInput)?;
    let p_fit = fit_design.ncols();
    let mut s_lambda_internal = Array2::<f64>::zeros((p_fit, p_fit));
    for (k, penalty) in fit_penalties.iter().enumerate() {
        match penalty {
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                s_lambda_internal
                    .slice_mut(ndarray::s![col_range.clone(), col_range.clone()])
                    .scaled_add(fit.lambdas[k], local);
            }
            PenaltySpec::Dense(m) | PenaltySpec::DenseWithMean { matrix: m, .. } => {
                s_lambda_internal.scaled_add(fit.lambdas[k], m);
            }
        }
    }
    let mut latent_precision = h_data.clone();
    latent_precision += &s_lambda_internal;
    let user_precision_internal =
        transform_bounded_latent_precision_to_user_internal(&latent_precision, &jac_diag)?;
    let penalized_hessian =
        conditioning.transform_penalized_hessian_to_original(&user_precision_internal);

    // User-scale posterior covariance via the delta method. The reported
    // geometry precision `penalized_hessian` is the user-scale penalized
    // Hessian `H_user = C⁻ᵀ J⁻¹ (H_latent + S_λ) J⁻¹ C⁻¹` (latent precision
    // pushed through the bounded transform's Jacobian `J = diag(dβ_user/dθ)`
    // and the conditioning map `C`). Its exact inverse `H_user⁻¹` is the
    // delta-method pushforward of the latent posterior precision-inverse
    // `(H_latent + S_λ)⁻¹` — but on the UNSCALED (unit-dispersion) scale. For a
    // free-dispersion family (profiled Gaussian) the reported coefficient
    // covariance is `Vb = φ̂ · H_user⁻¹` with `φ̂ = σ̂²`, so the unscaled inverse
    // below is multiplied by the dispersion scale `cov_scale` once `σ̂²` is
    // known (after the EDF, which sets the residual d.f.). For fixed-scale
    // families (Binomial, `φ ≡ 1`) `cov_scale == 1` and `Vb = H_user⁻¹`
    // unchanged. Skipping this scale was gam#1514: an interior, well-identified
    // Gaussian bounded slope reported an SE ≈ 1/√Σ(xᵢ−x̄)² instead of
    // σ̂/√Σ(xᵢ−x̄)², i.e. ~`1/σ̂` (≈20×) too wide.
    //
    // Inverting the same matrix the geometry reports keeps
    // `inv(penalized_hessian) == cov_scale⁻¹ · covariance` and removes the
    // dependency on the inner solver's optional, canonical-space
    // `covariance_conditional` (which is `None` whenever the bounded blockspec
    // carries no smoothing parameters — the no-rho fit path — leaving a bounded
    // fit with a populated precision but no user-scale covariance, the gam#854
    // symptom). The latent precision is SPD at a strict posterior maximum; on a
    // singular or indefinite boundary Hessian no finite posterior covariance
    // exists, so inference is refused rather than projected onto a
    // pseudo-covariance.
    let beta_covariance_unscaled = if options.compute_inference {
        Some(certified_bounded_posterior_covariance(
            &penalized_hessian,
            "bounded user-scale posterior precision",
        )?)
    } else {
        None
    };
    // EDF `p − Σ_k λ_k tr(H_latent⁻¹ S_k)` is computed in the *latent*
    // (untransformed) coordinate system the penalties `fit_penalties` live in,
    // so it needs the latent posterior covariance `(H_latent + S_λ)⁻¹`, not the
    // user-scale one. Invert the same latent precision that produced the
    // reported user precision so the two are an exact transform pair.
    let latent_cov = if options.compute_inference {
        Some(certified_bounded_posterior_covariance(
            &latent_precision,
            "bounded latent posterior precision",
        )?)
    } else {
        None
    };
    let s_lambda_original = weighted_blockwise_penalty_sum(
        &design.penalties,
        fit.lambdas
            .as_slice()
            .expect("the fitted lambdas are a contiguous standard-layout array"),
        design.design.ncols(),
    );
    let penalty_term = beta_user.dot(&s_lambda_original.dot(&beta_user));
    let deviance = -2.0 * eta_state.log_likelihood;
    let (edf_by_block, penalty_block_trace, edf_total) = if let Some(cov) = latent_cov.as_ref() {
        exact_bounded_edf(&fit_penalties, &fit.lambdas, cov)?
    } else {
        (
            vec![0.0; fit_penalties.len()],
            vec![0.0; fit_penalties.len()],
            0.0,
        )
    };

    // Dispersion. The bounded fit's working weight is scale-free for a profiled
    // Gaussian (`W = priorweights`), so the unscaled penalized Hessian carries
    // unit implicit dispersion and the reported coefficient covariance must be
    // restored to `Vb = σ̂²·H_user⁻¹` with the REML residual variance
    // `σ̂² = RSS/(n − edf_total)` — identical to the ordinary GAM path
    // (`solver/estimate/optimizer.rs`). Fixed-scale families (Binomial here,
    // `φ ≡ 1`) keep their full Fisher information in `W`, so `cov_scale == 1`
    // and the covariance is `H_user⁻¹` unscaled. The single source of truth for
    // the per-family scale is `GlmLikelihoodSpec::coefficient_covariance_scale`
    // / `dispersion_from_likelihood`, reused verbatim so the bounded path can
    // never drift from the standard contract (gam#1514).
    let profiled_gaussian_standard_deviation = if matches!(
        resolved_likelihood_scale,
        gam_spec::ResolvedLikelihoodScale::ProfiledGaussian
    ) {
        let residual_dof = if options.compute_inference {
            y.len() as f64 - edf_total
        } else {
            y.len() as f64
        };
        Some(certified_profiled_gaussian_scale(
            deviance,
            residual_dof,
            "bounded Gaussian",
        )?)
    } else {
        None
    };
    let dispersion = gam_solve::estimate::dispersion_from_likelihood(
        &glm_likelihood,
        profiled_gaussian_standard_deviation,
    )?;
    let standard_deviation = dispersion.phi().sqrt();
    let cov_scale = glm_likelihood
        .coefficient_covariance_scale(dispersion.phi())
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    // Apply the dispersion scale to the unscaled inverse, producing the reported
    // `Vb = cov_scale · H_user⁻¹` and its diagonal standard errors. The stored
    // `penalized_hessian` stays UNSCALED (`H_user`) per the dispersion-ownership
    // contract in `inference::dispersion_cov`; the sampler re-applies `√cov_scale`
    // when it reconstructs the latent posterior (see `sample_standard_bounded`).
    let beta_covariance = beta_covariance_unscaled.map(|mut cov| {
        if cov_scale != 1.0 {
            cov.mapv_inplace(|v| v * cov_scale);
        }
        cov
    });
    if let Some(covariance) = beta_covariance.as_ref()
        && covariance.iter().any(|value| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(
            "bounded coefficient covariance scaling produced a non-finite value".to_string(),
        ));
    }
    // The published standard errors derive from this matrix (#2955); judging its
    // diagonal here names this lane in the refusal.
    beta_covariance
        .as_ref()
        .map(gam_problem::se_from_covariance)
        .transpose()
        .map_err(|err| {
            EstimationError::InvalidInput(format!(
                "bounded coefficient covariance cannot produce standard errors: {err}"
            ))
        })?;
    // #2903: with lambda selected, `fit_custom_family` minted the first-order
    // correction `C = A·V_ρ·Aᵀ`, `A = V_cond·U`, `U[:, k] = λ_k S_k θ̂`, on the latent
    // coefficients from the certified outer rho-Hessian. The user coefficients are
    // `M·b(θ)`, so `∂β_user/∂ρ = M·J·∂θ̂/∂ρ` and C reaches user scale through the
    // conditional covariance's own congruence. It carries no dispersion factor,
    // because `∂θ̂/∂ρ` has none. A fit that selects no lambda has `Vp = Vb`
    // exactly; a penalized fit whose correction was refused keeps the typed absence.
    let smoothing_corrected = match (fit.smoothing_correction(), fit.smoothing_correction_method())
    {
        (Some(latent_correction), Some(method)) => Some((
            bounded_latent_bilinear_to_user(latent_correction, &jac_diag, &conditioning)?,
            method,
        )),
        _ => None,
    };
    let covariance_corrected = match (smoothing_corrected.as_ref(), beta_covariance.as_ref()) {
        (Some((correction, _)), Some(v_cond)) => Some(v_cond + correction),
        (None, Some(v_cond)) if fit_penalties.is_empty() => Some(v_cond.clone()),
        _ => None,
    };
    // The latent fit's typed reason carries through the user-scale lift unchanged (#2677).
    let smoothing_correction_absence = if covariance_corrected.is_none() {
        fit.smoothing_correction_absence().cloned()
    } else {
        None
    };
    covariance_corrected
        .as_ref()
        .map(gam_problem::se_from_covariance)
        .transpose()
        .map_err(|err| {
            EstimationError::InvalidInput(format!(
                "bounded corrected coefficient covariance cannot produce standard errors: {err}"
            ))
        })?;
    let working_response = exact_standard_working_response(&eta_state)?;

    let geometry = Some(gam_solve::estimate::FitGeometry {
        coefficient_gauge: gam_problem::gauge::Gauge::identity(&[beta_user.len()]),
        penalized_hessian: penalized_hessian.clone().into(),
        constrained_posterior: None,
        working: Some(gam_solve::estimate::WorkingGeometry {
            weights: eta_state.fisherweight.clone(),
            response: working_response,
        }),
    });
    let max_abs_eta = eta_state
        .eta
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    Ok(FittedTermCollection {
        fit: {
            let log_lambdas =
                checked_fit_log_lambdas(&fit.lambdas, "final fitted term collection")?;
            let inf = FitInference {
                edf_by_block,
                penalty_block_trace,
                edf_rank_bound: Vec::new(),
                edf_total,
                // This lane publishes only the first-order correction, so its
                // retained first-order pair is its primary pair.
                smoothing_correction: smoothing_corrected
                    .as_ref()
                    .map(|(correction, _)| correction.clone()),
                smoothing_correction_method: smoothing_corrected.as_ref().map(|(_, method)| *method),
                smoothing_correction_first_order: smoothing_corrected
                    .as_ref()
                    .map(|(correction, _)| correction.clone()),
                smoothing_correction_method_first_order: smoothing_corrected
                    .as_ref()
                    .map(|(_, method)| *method),
                smoothing_correction_absence,
                // Boundary adapter: `penalized_hessian` storage is now
                // `UnscaledPrecision`.
                penalized_hessian: penalized_hessian.clone().into(),
                reparam_qs: None,
                dispersion,
                factorized_standard_errors: None,
                beta_covariance_frequentist: None,
                coefficient_influence: None,
                weighted_gram: None,
                identified_subspace: None,
            };
            let covariance_conditional = beta_covariance;
            // Sealed `UnifiedFitResult`: existence certifies inner+outer
            // convergence (see `try_from_parts`), so the status is Converged.
            let pirls_status_val = gam_solve::pirls::PirlsStatus::Converged;
            let fit_objective = fit.penalized_objective();
            UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
                blocks: vec![gam_solve::estimate::FittedBlock {
                    beta: beta_user.clone(),
                    role: gam_problem::BlockRole::Mean,
                    edf: edf_total,
                    lambdas: fit.lambdas.clone(),
                }],
                training_sample_size: y.len(),
                log_lambdas,
                lambdas: fit.lambdas,
                likelihood_scale: glm_likelihood.scale,
                likelihood_family: Some(glm_likelihood.spec),
                log_likelihood_normalization: gam_spec::LogLikelihoodNormalization::UserProvided,
                log_likelihood: eta_state.log_likelihood,
                deviance,
                reml_score: fit_objective,
                stable_penalty_term: penalty_term,
                penalized_objective: fit_objective,
                used_device: false,
                outer_iterations: fit.outer_iterations,
                // Sealed result ⇒ outer convergence was certified at assembly.
                outer_converged: true,
                outer_gradient_norm: fit.outer_gradient_norm,
                standard_deviation,
                covariance_conditional,
                covariance_corrected,
                inference: Some(inf),
                fitted_link: gam_solve::estimate::FittedLinkState::Standard(None),
                geometry,
                block_states: Vec::new(),
                pirls_status: pirls_status_val,
                max_abs_eta,
                constraint_kkt: None,
                artifacts: gam_solve::estimate::FitArtifacts {
                    pirls: None,
                    // Assembly accepts selected smoothing only with the outer
                    // search's stationarity certificate, which the custom-family
                    // fit already carries; it is threaded through, not dropped.
                    criterion_certificate: fit.artifacts.criterion_certificate.clone(),
                    ..Default::default()
                },
                inner_cycles: 0,
            })?
        },
        design: design.clone(),
    })
}

fn enforce_term_constraint_feasibility(
    design: &TermCollectionDesign,
    fit: &UnifiedFitResult,
) -> Result<(), EstimationError> {
    // Geometric (per-row-scaled) tolerance: the solver's own primal-feasibility
    // contract `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`, which the in-solver acceptance
    // gate and the `fit.constraint_kkt` diagnostic both read. Lower-bound rows are
    // unit-norm (a_i = e_i), so the scaled and raw violations coincide there, and
    // linear-inequality rows are divided by ‖a_i‖ below. A B-spline
    // endpoint-derivative clamp at k = 12 carries ‖a_i‖ ≈ 38, so a raw residual of
    // 1e-6 is 2.6e-8 in geometric units. Auditing a completed fit against a
    // different number than the one it was accepted at is the inconsistency that
    // made well-conditioned clamped fits get rejected after they completed cleanly.
    let tol = gam_solve::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;
    let smooth_start = design
        .design
        .ncols()
        .saturating_sub(design.smooth.total_smooth_cols());
    let mut violations: Vec<String> = Vec::new();
    for term in &design.smooth.terms {
        let gr = (smooth_start + term.coeff_range.start)..(smooth_start + term.coeff_range.end);
        let beta_local = fit.beta.slice(s![gr.clone()]).to_owned();
        if let Some(lb) = term.lower_bounds_local.as_ref() {
            let mut worst = 0.0_f64;
            let mut worst_idx = 0usize;
            for i in 0..lb.len().min(beta_local.len()) {
                if lb[i].is_finite() {
                    let viol = (lb[i] - beta_local[i]).max(0.0);
                    if viol > worst {
                        worst = viol;
                        worst_idx = i;
                    }
                }
            }
            if worst > tol {
                violations.push(format!(
                    "term='{}' kind=lower-bound maxviolation={:.3e} coeff_index={}",
                    term.name, worst, worst_idx
                ));
            }
        }
        if let Some(lin) = term.linear_constraints_local.as_ref() {
            let mut worst = 0.0_f64;
            let mut worstrow = 0usize;
            for i in 0..lin.a.nrows() {
                let norm = lin.a.row(i).dot(&lin.a.row(i)).sqrt();
                let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
                let s = (lin.a.row(i).dot(&beta_local) - lin.b[i]) * inv;
                let viol = (-s).max(0.0);
                if viol > worst {
                    worst = viol;
                    worstrow = i;
                }
            }
            if worst > tol {
                violations.push(format!(
                    "term='{}' kind=linear-inequality maxviolation={:.3e} row={}",
                    term.name, worst, worstrow
                ));
            }
        }
    }

    if !violations.is_empty() {
        let mut msg = format!(
            "constraint violation after fit ({} violating term constraints): {}",
            violations.len(),
            violations.join(" | ")
        );
        if let Some(kkt) = fit.constraint_kkt.as_ref() {
            msg.push_str(&format!(
                "; KKT[primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}]{}",
                kkt.primal_feasibility,
                kkt.dual_feasibility,
                kkt.complementarity,
                kkt.stationarity,
                // `stat` here is `‖grad − Aᵀλ‖∞`; on a refused cone projection
                // no λ was ever computed, so say so rather than letting the
                // number stand as a residual (#2601).
                kkt.cone_projection_note()
            ));
        }
        return Err(EstimationError::ParameterConstraintViolation(msg));
    }
    Ok(())
}

pub(crate) fn spatial_length_scale_term_indices(spec: &TermCollectionSpec) -> Vec<usize> {
    spec.smooth_terms
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| spatial_term_supports_hyper_optimization(spec, idx).then_some(idx))
        .collect()
}

/// The scalar a multi-start / candidate comparison ranks one realized fit by.
///
/// The REML/LAML score is the criterion the outer search itself minimizes, so
/// it is what a comparison between two realized fits must use when it exists.
/// A route that produced no outer score (a fixed-ρ path, or a family whose
/// criterion is not assembled) still has a penalized deviance, and
/// `½·deviance + ½·stable_penalty_term` is the same objective up to the terms
/// those routes do not form. A non-finite score is `+∞`, i.e. never preferred,
/// which is what makes this usable as a plain `total_cmp` key.
fn fit_score(fit: &UnifiedFitResult) -> f64 {
    if let Some(score) = fit.reml_score().filter(|value| value.is_finite()) {
        return score;
    }
    let score = 0.5 * fit.deviance + 0.5 * fit.stable_penalty_term;
    if score.is_finite() {
        score
    } else {
        f64::INFINITY
    }
}

/// Classify an outer-evaluation error as a *recoverable trial-point
/// infeasibility* versus a genuine fatal failure.
///
/// The spatial-κ / anisotropy outer optimizer probes a sequence of trial
/// hyperparameters. At an extreme trial point the realized kernel design or
/// its ψ-derivatives may simply be non-constructible — e.g. a learned
/// per-axis log-scale stretches the anisotropic distance `r = |Λh|` until the
/// Duchon polyharmonic blocks `r^(2m−d)` overflow, or a degenerate metric
/// collapses two centers onto a non-C² collision. Those points lie outside
/// the model's feasible domain; the principled response is to treat them like
/// the cost-only path already does (objective `+∞`) so the line-search /
/// trust-region solver retreats, rather than aborting the entire REML fit.
///
/// A `BasisError` is exactly this class: it means "the basis/design cannot be
/// built at this hyperparameter". The same retreat semantics also apply when a
/// trial reaches the inner solve but produces a singular/unstable curvature:
/// those cases are reported by the shared inner-solve retreat classifier, or
/// by the final fit validator when an inference-only matrix derived from
/// `H⁻¹` (not the fitted mean coefficients themselves) becomes non-finite.
/// Everything else (layout/topology invariants, over-parameterization, and
/// arbitrary invalid inputs) stays fatal so genuine bugs are never masked.
fn is_recoverable_trial_point_error(err: &EstimationError) -> bool {
    // The producer's own verdict, and nothing else. `is_inner_solve_retreat` is
    // a subset of it, and the non-finite inference-matrix case that used to be
    // recovered here by matching `"must be finite"` against three field names
    // now arrives as `TrialPointRefused` from
    // `validate_all_finite_trial_point` (#2593). A verdict carried in prose is
    // one `format!` away from silently changing meaning, and this driver was
    // the third place in the workspace maintaining its own answer to a question
    // the producer had already answered (#2531/#2590/#2553).
    matches!(err, EstimationError::BasisError(_)) || err.is_trial_point_infeasible()
}

#[cfg(test)]
mod spatial_trial_recovery_tests {
    use super::*;

    #[test]
    fn nonfinite_frequentist_covariance_is_recoverable_trial_point() {
        // Exactly what `validate_all_finite_trial_point` produces, including
        // the unchanged message text.
        let err = EstimationError::TrialPointRefused {
            reason: "fit_result.beta_covariance_frequentist[0] must be finite, got NaN".to_string(),
        };

        assert!(
            is_recoverable_trial_point_error(&err),
            "singular trial-point curvature should make spatial κ retreat, not abort"
        );
        // The same TEXT as an `InvalidInput` must NOT be recoverable: the
        // verdict now comes from the variant, so prose can no longer buy it.
        assert!(
            !is_recoverable_trial_point_error(&EstimationError::InvalidInput(
                "fit_result.beta_covariance_frequentist[0] must be finite, got NaN".to_string(),
            )),
            "recoverability must come from the producer's variant, not from the message"
        );
    }

    #[test]
    fn a_trial_inner_non_convergence_still_retreats_and_only_a_fit_ending_one_is_fatal_2943() {
        // A κ trial whose inner solve did not certify is valley escape: the
        // producer's refusal, now carrying its budget and carrying block, must
        // still make the spatial driver retreat to another κ.
        let refusal = gam_problem::CustomFamilyError::InnerSolveNotConverged {
            cycles: 8,
            terminal: None,
            kkt_residual: Some(1.081e3),
            kkt_tol: Some(5.352e-2),
            theta_dim: 2,
            rho_dim: 2,
            psi_dim: 0,
            cycle_budget: Some(8),
            carrying_block: Some("slope_surface".to_string()),
        };
        assert!(
            is_recoverable_trial_point_error(&EstimationError::CustomFamily(refusal.clone())),
            "an uncertified trial must let the spatial κ search retreat, not abort the fit"
        );
        // The fit-ending form is minted only where a fit is handed to its caller,
        // with no search left above it; the driver must not treat it as a trial.
        let ended = gam_problem::CustomFamilyError::fit_ended_without_certified_inner_mode(refusal);
        assert!(
            !is_recoverable_trial_point_error(&EstimationError::CustomFamily(ended)),
            "a fit that has ended is not a trial point"
        );
    }

    #[test]
    fn arbitrary_invalid_input_remains_fatal_trial_point_error() {
        let err = EstimationError::InvalidInput("outer rho bounds are invalid".to_string());

        assert!(
            !is_recoverable_trial_point_error(&err),
            "the spatial κ recovery gate must not mask unrelated invalid inputs"
        );
    }

    #[test]
    fn spatial_value_probe_classifier_matches_derivative_lane() {
        // The contract is an AGREEMENT: the value-probe lane must answer with a
        // typed refusal — a variant `is_trial_point_infeasible` recognizes, still
        // carrying its reason — on exactly the errors the derivative lane calls
        // recoverable, and propagate every other error unchanged. No +∞ cost: a
        // sentinel names no reason, so three walls read "non-finite cost" and
        // nothing else (#2735). Asserting the agreement over a table — rather
        // than pinning one hand-picked error per outcome — is what keeps this
        // gate honest when the producer's verdict moves.
        //
        // #2593 moved that verdict from the message text to the error VARIANT
        // (`TrialPointRefused`), which is what the sibling
        // `nonfinite_frequentist_covariance_is_recoverable_trial_point` pins.
        // This fixture still carried the pre-#2593 premise — it built the
        // non-finite-covariance refusal as an `InvalidInput` and demanded a
        // retreat — so it asserted the exact opposite of its own sibling on the
        // same input, and the two could not both hold. The table below carries
        // both variants with the SAME prose, so the "prose cannot buy
        // recoverability" rule is now part of this gate rather than only of the
        // sibling's.
        let nonfinite_covariance =
            "fit_result.beta_covariance_frequentist[0] must be finite, got NaN";
        let unrelated = "outer rho bounds are invalid";
        let cases = [
            EstimationError::TrialPointRefused {
                reason: nonfinite_covariance.to_string(),
            },
            // A design that cannot be built at this trial hyperparameter — the
            // other half of `is_recoverable_trial_point_error`'s verdict.
            EstimationError::BasisError(gam_problem::BasisError::DegenerateRange(8)),
            EstimationError::InvalidInput(nonfinite_covariance.to_string()),
            EstimationError::InvalidInput(unrelated.to_string()),
        ];

        for error in cases {
            let message = error.to_string();
            let derivative_lane_recovers = is_recoverable_trial_point_error(&error);
            let typed = classify_spatial_value_probe_failure(error);
            if derivative_lane_recovers {
                assert!(
                    typed.is_trial_point_infeasible(),
                    "the value probe must carry {message:?} as a typed refusal the outer \
                     consumers retreat from; got {typed}"
                );
                assert!(
                    typed.to_string().contains(&message),
                    "the refusal must keep its reason: {message:?} became {typed}"
                );
            } else {
                assert!(
                    !typed.is_trial_point_infeasible(),
                    "the value probe turned the fatal {message:?} into a refusal"
                );
                assert_eq!(
                    typed.to_string(),
                    message,
                    "a fatal failure must be propagated unchanged, not reworded"
                );
            }
        }
    }
}

/// Surface the two ways the exact spatial-κ route can fail to produce a
/// candidate at all: it was unavailable, or it errored.
///
/// # ⚠ It deliberately does NOT grade the candidate's score (#2748)
///
/// It used to. A candidate whose realized score was worse than the incumbent's
/// by more than `max(1e-6, |score|·1e-8)` was turned into a
/// `RemlOptimizationFailed` and **killed the whole fit** — measured on
/// `geo_disease_eas_matern_k6`, where a `1.267594e3 → 1.267595e3` regression
/// (`8e-7` relative, some eighty times that bar) took down all four of its
/// non-flexible benchmark lanes, `rust_gam` included.
///
/// Two things were wrong with that, and neither is the size of the bar.
///
/// 1. **The incumbent exists.** κ optimization is a refinement of a fit that
///    has already been produced and scored. "The refinement did not improve on
///    the incumbent" is an argument for shipping the incumbent, not for
///    destroying it — which is exactly what the sibling
///    `JointSpatialKappaOutcome::DeclinedKeepIncumbent` arm concluded when the
///    joint route graded its own candidate one level in. Two graders of the
///    same comparison reaching opposite responses is the defect, and the
///    response that keeps a valid fit is the right one.
/// 2. **A comparison between two measured numbers needs no tolerance.** The
///    caller now keeps whichever of the incumbent and the candidate has the
///    smaller score. `argmin` over two values is monotone by construction: it
///    cannot ship something worse than what it was handed, at any drift, so
///    there is nothing left for a bar to calibrate.
fn require_available_spatial_optimization_result<T>(
    result: Result<Option<T>, EstimationError>,
) -> Result<T, EstimationError> {
    match result {
        Ok(Some(value)) => Ok(value),
        Ok(None) => Err(EstimationError::RemlOptimizationFailed(
            "spatial kappa optimization is unavailable for one or more eligible spatial terms"
                .to_string(),
        )),
        Err(err) => Err(EstimationError::RemlOptimizationFailed(format!(
            "spatial kappa optimization failed: {err}"
        ))),
    }
}

fn external_opts_for_design(
    family: &LikelihoodSpec,
    design: &TermCollectionDesign,
    options: &FitOptions,
) -> ExternalOptimOptions {
    ExternalOptimOptions {
        family: family.clone(),
        latent_cloglog: options.latent_cloglog,
        mixture_link: options.mixture_link.clone(),
        optimize_mixture: options.optimize_mixture,
        sas_link: options.sas_link,
        optimize_sas: options.optimize_sas,
        compute_inference: options.compute_inference,
        skip_rho_posterior_inference: options.skip_rho_posterior_inference,
        max_iter: options.max_iter,
        tol: options.tol,
        nullspace_dims: design.nullspace_dims.clone(),
        linear_constraints: design.linear_constraints.clone(),
        firth_bias_reduction: Some(options.firth_bias_reduction),
        rho_prior: options.rho_prior.clone(),
        persistent_warm_start_store: options.persistent_warm_start_store.clone(),
    }
}

/// Evaluate the joint REML cost, gradient, and Hessian result at a given θ = [ρ, ψ]
/// for a single-block term collection with spatial hyperparameters.
///
/// This provides a direct evaluation of the profiled REML objective using the
/// external-caller interface, which exposes exact cost/gradient/Hessian without
/// running the full outer smoothing loop. The returned tuple is
/// `(cost, gradient, hessian)` in the joint [ρ, ψ] space.
fn evaluate_joint_reml_outer_eval_at_theta(
    evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>,
    design: &TermCollectionDesign,
    theta: &Array1<f64>,
    rho_dim: usize,
    hyper_dirs: Vec<gam_solve::estimate::reml::DirectionalHyperParam>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    order: gam_solve::rho_optimizer::OuterEvalOrder,
    design_revision: Option<u64>,
) -> Result<(f64, Array1<f64>, gam_problem::HessianValue), EstimationError> {
    evaluator.evaluate_with_order(
        &design.design,
        &design.penalties,
        &design.nullspace_dims,
        design.linear_constraints.clone(),
        theta,
        rho_dim,
        hyper_dirs,
        warm_start_beta,
        "evaluate_joint_reml_outer_eval_at_theta",
        order,
        design_revision,
    )
}

/// The error an incremental κ rebuild reports when the realizer cannot put a
/// term back into its collection's identifiability gauge (#2747).
///
/// A trial ψ at which the term's block lies numerically inside the constraint
/// span cannot be placed, so the model does not exist at that trial: the search
/// retreats, as for a penalty that is not PSD there, instead of aborting the fit
/// (gam#2959). Any other placement failure is a defect and stays fatal.
fn collection_gauge_placement_error(
    name: &str,
    trial_report: &str,
    error: gam_terms::basis::BasisError,
) -> EstimationError {
    if matches!(
        error,
        gam_terms::basis::BasisError::CollectionGaugeNotOrthogonal { .. }
    ) {
        EstimationError::TrialPointRefused {
            reason: format!(
                "term '{name}' cannot be placed in its collection's identifiability gauge at \
                 this psi ({trial_report}): {error}"
            ),
        }
    } else {
        EstimationError::InvalidInput(format!(
            "term '{name}' could not be returned to its collection's identifiability gauge \
             after an incremental rebuild: {error}"
        ))
    }
}

fn evaluate_joint_reml_efs_at_theta(
    evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>,
    design: &TermCollectionDesign,
    theta: &Array1<f64>,
    rho_dim: usize,
    hyper_dirs: Vec<gam_solve::estimate::reml::DirectionalHyperParam>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    design_revision: Option<u64>,
) -> Result<gam_problem::EfsEval, EstimationError> {
    evaluator.evaluate_efs(
        &design.design,
        &design.penalties,
        &design.nullspace_dims,
        design.linear_constraints.clone(),
        theta,
        rho_dim,
        hyper_dirs,
        warm_start_beta,
        "evaluate_joint_reml_efs_at_theta",
        design_revision,
    )
}

fn exact_joint_spatial_outer_hessian_available(
    family: &LikelihoodSpec,
    design: &TermCollectionDesign,
) -> bool {
    // Every `LikelihoodSpec` variant (Gaussian, Binomial-*, Poisson, Gamma,
    // Royston-Parmar) routes through the unified evaluator's outer-Hessian
    // path: Gaussian Identity uses the no-correction dense form, all GLM
    // variants supply scalar-GLM derivative ingredients consumed by
    // `compute_outer_hessian` / `build_outer_hessian_operator`, and
    // `outer_hessian_route_plan` chooses the matrix-free
    // `HessianValue::Operator` representation when the dense assembly exceeds
    // the materialization cap.  The previous `Identity || sparse_design`
    // gate predates that operator routing and forced binomial+logit+Matern
    // (and any other non-Gaussian dense-lazy spatial design) onto the
    // gradient-only BFGS path even though analytic Hessian is fully
    // available — capability check, not cost.  Match every variant
    // explicitly so any future family addition (which may not yet provide
    // outer-Hessian ingredients) forces an authoring decision here rather
    // than silently inheriting `true`.
    // Every supported response (Gaussian, Binomial-*, Poisson, Tweedie,
    // NegativeBinomial, Beta, Gamma, Royston-Parmar) routes through the
    // unified evaluator's outer-Hessian path; the spec-level capability
    // check therefore always succeeds. Match every response explicitly so
    // any future family addition (which may not yet provide outer-Hessian
    // ingredients) forces an authoring decision here rather than silently
    // inheriting `true`.
    let family_supported = match &family.response {
        ResponseFamily::Gaussian
        | ResponseFamily::Binomial
        | ResponseFamily::Poisson
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Beta { .. }
        | ResponseFamily::Gamma
        | ResponseFamily::RoystonParmar => true,
    };
    // A design with zero columns has no joint outer-Hessian to compute;
    // the analytic path is only meaningful for non-empty parameter blocks.
    family_supported && design.design.ncols() > 0
}

fn try_build_spatial_term_log_kappa_derivativeinfo(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    term_idx: usize,
) -> Result<Option<SpatialPsiDerivative>, EstimationError> {
    let Some((
        global_range,
        total_p,
        x_psi_local,
        s_psi_local_check,
        x_psi_psi_local,
        s_psi_psi_local,
        s_psi_components_local,
        s_psi_psi_components_local,
        implicit_operator,
    )) = try_build_spatial_term_log_kappa_derivative(data, resolvedspec, design, term_idx)?
    else {
        return Ok(None);
    };
    let Some(penalty_range) = design
        .smooth_term_penalty_range(term_idx)
        .map_err(EstimationError::InvalidInput)?
    else {
        return Ok(None);
    };
    let penalty_start = penalty_range.start;
    if s_psi_components_local.is_empty() || s_psi_psi_components_local.is_empty() {
        return Ok(None);
    }
    if s_psi_components_local.len() != s_psi_psi_components_local.len() {
        return Ok(None);
    }
    let penalty_indices = (0..s_psi_components_local.len())
        .map(|j| penalty_start + j)
        .collect::<Vec<_>>();
    let penalty_index = penalty_indices[0];
    if s_psi_local_check.nrows() == 0 || s_psi_psi_local.nrows() == 0 {
        return Ok(None);
    }
    Ok(Some(SpatialPsiDerivative {
        penalty_index,
        penalty_indices,
        global_range,
        total_p,
        x_psi_local,
        s_psi_components_local,
        x_psi_psi_local,
        s_psi_psi_components_local,
        aniso_group_id: None,
        aniso_cross_designs: None,
        aniso_cross_penalty_provider: None,
        implicit_operator,
        implicit_axis: 0,
    }))
}

pub(crate) fn try_build_spatial_log_kappa_derivativeinfo_list(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    spatial_terms: &[usize],
) -> Result<Option<Vec<SpatialPsiDerivative>>, EstimationError> {
    let mut out = Vec::new();
    let mut aniso_gid = 0usize;
    for &term_idx in spatial_terms {
        if spatial_term_uses_per_axis_psi(resolvedspec, term_idx) {
            if let Some(entries) = try_build_spatial_term_log_kappa_aniso_derivativeinfos(
                data,
                resolvedspec,
                design,
                term_idx,
                aniso_gid,
            )? {
                aniso_gid += 1;
                out.extend(entries);
                continue;
            } else {
                // The enrollment predicate said `d` axes and the producer could
                // not supply them. Falling back to the isotropic builder here
                // would answer a `d`-coordinate question with one coordinate,
                // so the whole list declines — and says which term did it,
                // because the caller turns this into "spatial kappa
                // optimization is unavailable for one or more eligible spatial
                // terms", which names none of them.
                log::warn!(
                    "[spatial-kappa] term {term_idx}: enrolled for per-axis ψ but its per-axis \
                     derivative producer declined; the joint κ route is unavailable for this fit"
                );
                return Ok(None);
            }
        }
        let Some(info) =
            try_build_spatial_term_log_kappa_derivativeinfo(data, resolvedspec, design, term_idx)?
        else {
            log::warn!(
                "[spatial-kappa] term {term_idx}: isotropic ψ derivative producer declined; the \
                 joint κ route is unavailable for this fit"
            );
            return Ok(None);
        };
        out.push(info);
    }
    Ok(Some(out))
}

/// For an aniso term with d axes, produce d `SpatialPsiDerivative` entries.
fn try_build_spatial_term_log_kappa_aniso_derivativeinfos(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    term_idx: usize,
    aniso_group_id: usize,
) -> Result<Option<Vec<SpatialPsiDerivative>>, EstimationError> {
    let Some(smooth_term) = design.smooth.terms.get(term_idx) else {
        return Ok(None);
    };
    let Some(termspec) = resolvedspec.smooth_terms.get(term_idx) else {
        return Ok(None);
    };
    let mut aniso_result = match &termspec.basis {
        SmoothBasisSpec::Sphere { .. } => return Ok(None),
        SmoothBasisSpec::Matern {
            feature_cols,
            spec,
            input_scale,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            let mut spec_operator = spec.clone();
            if let Some(scale) = input_scale {
                scale.standardize(&mut x);
                let length_scale = spec.length_scale.resolved().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "anisotropic Matérn Auto length_scale reached derivative construction \
                         unresolved"
                            .to_string(),
                    )
                })?;
                spec_operator.length_scale.set_resolved(
                    scale
                        .to_standardized_units(gam_terms::OriginalUnits::new(length_scale))
                        .standardized_value(),
                );
            }
            // #1122: the realized Matérn design always carries the operator
            // {mass, tension, stiffness} penalty triplet (`build_term` overrides
            // the `double_penalty` kernel penalty via
            // `matern_operator_penalty_triplet_from_metadata`). The per-axis
            // κ-gradient must differentiate that SAME triplet, not the kernel
            // double-penalty blocks, or the analytic `tr(S⁺ Ṡ)` desyncs from the
            // FD of the criterion's operator-triplet `log|Sλ|₊` (the iso-axis
            // analogue is handled in `try_build_spatial_term_log_kappa_derivative`).
            // The jet is built in the FITTED coefficient chart. The incremental
            // realizer hands this builder a spec whose identifiability has been
            // put back into the TERM-LOCAL chart `z_local`
            // (`restore_local_identifiability_chart`, gam#2760) so that a design
            // REBUILD can apply the collection gauge's fixed `T0` itself; the
            // design the criterion is built on lives in the composition
            // `z_local · T0` the realized term's metadata records. Built on
            // `z_local`, the jet has the fitted WIDTH whenever `T0` is square
            // (the Residualize arm), so nothing declines and the ψ-gradient is
            // silently wrong: measured on the #1379 seed-3 fixture, the same θ
            // and the same cost (−138.0325036) gave ∂V/∂ψ = −2.949e4 here
            // against +0.2253 from the composed chart (central differences
            // +0.2249), and the line search walked uphill for 50 attempts. The
            // Duchon arm below already replays its chart from metadata; the
            // measure-jet per-axis arm was fixed the same way (597003f2e).
            if let BasisMetadata::Matern {
                identifiability_transform: Some(transform),
                ..
            } = &smooth_term.metadata
            {
                spec_operator.identifiability = MaternIdentifiability::FrozenTransform {
                    transform: transform.clone(),
                };
            }
            spec_operator.double_penalty = false;
            build_matern_basis_log_kappa_aniso_derivatives(x.view(), &spec_operator)
                .map_err(EstimationError::from)?
        }
        // Measure-jet: the grouped dial coordinates ride the same per-axis
        // carrier. The producer runs on the FROZEN spec (the driver runs
        // post-freeze), so per-trial rebuilds move only the dials; the
        // coordinate layout, zero design drift, and shared candidate
        // normalization are owned by `build_measure_jet_basis_psi_derivatives`.
        //
        // The chart the jets are built in is the FITTED one. The incremental
        // realizer hands this builder a spec whose identifiability has been put
        // back into the TERM-LOCAL chart `z_local` (see
        // `restore_local_identifiability_chart`, gam#2760) so that a design
        // REBUILD can apply the collection gauge's fixed `T0` itself. A ψ-jet
        // must differentiate the design the criterion is built on, and that
        // design lives in the composition `z_local · T0` the realized term's
        // metadata records — the same frozen chart the Duchon arm below replays
        // from its metadata. Built on `z_local` alone, every jet came out
        // `local_columns` wide against a `coeff_range` one gauge rank narrower,
        // the width check below declined the term, and the whole measure-jet ψ
        // search aborted at revision 0 (`failed to build anisotropic hyper_dirs
        // at current psi` on the 3-D/5-D helix and the 1-D formula fixtures).
        // The collection's fixed row-space projection is applied afterwards,
        // where every arm's jets receive it.
        SmoothBasisSpec::MeasureJet {
            feature_cols,
            spec,
            input_scale,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            if let Some(scale) = input_scale {
                scale.standardize(&mut x);
            }
            let mut spec_fitted_chart = spec.clone();
            if let BasisMetadata::MeasureJet {
                constraint_transform: Some(transform),
                ..
            } = &smooth_term.metadata
            {
                spec_fitted_chart.identifiability =
                    gam_terms::basis::MeasureJetIdentifiability::FrozenTransform {
                        transform: transform.clone(),
                    };
            }
            build_measure_jet_basis_psi_derivatives(x.view(), &spec_fitted_chart)
                .map_err(EstimationError::from)?
        }
        // gam#2735 — the hybrid Duchon's per-axis η, on the SAME frozen chart
        // the isotropic arm replays (`try_build_spatial_term_log_kappa_derivative`
        // does the identical `radial_reparam` restore for the same reason,
        // #1355): centers, identifiability transform and collocation points all
        // come from the realized design's metadata, so the ψ-jet differentiates
        // exactly the design and penalties the criterion is built on.
        SmoothBasisSpec::Duchon {
            feature_cols,
            spec,
            input_scale,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            let mut spec_local = spec.clone();
            if let Some(scale) = input_scale {
                scale.standardize(&mut x);
                spec_local.length_scale = spec.length_scale.map(|length| {
                    scale
                        .to_standardized_units(gam_terms::OriginalUnits::new(length))
                        .standardized_value()
                });
            }
            let BasisMetadata::Duchon {
                centers,
                identifiability_transform,
                operator_collocation_points,
                radial_reparam,
                aniso_log_scales,
                ..
            } = &smooth_term.metadata
            else {
                log::warn!(
                    "[spatial-kappa] term {term_idx}: per-axis ψ declined -- a Duchon spec whose \
                     realized design does not carry Duchon metadata"
                );
                return Ok(None);
            };
            if spec_local.radial_reparam.is_none() {
                spec_local.radial_reparam = radial_reparam.clone();
            }
            // The realized anisotropy: the forward build centres the requested η
            // (`centered_aniso_contrasts`) and records the centred vector in the
            // metadata, so the derivative is taken at the design that shipped
            // rather than at an uncentred request.
            if let Some(resolved) = aniso_log_scales.as_ref() {
                spec_local.aniso_log_scales = Some(resolved.clone());
            }
            gam_terms::basis::build_duchon_basis_log_kappa_aniso_derivativeswith_collocationwithworkspace(
                x.view(),
                &spec_local,
                centers.view(),
                identifiability_transform.as_ref(),
                operator_collocation_points
                    .as_ref()
                    .map(|points| points.view()),
                &mut BasisWorkspace::default(),
            )
            .map_err(EstimationError::from)?
        }
        _ => return Ok(None),
    };
    // Get number of axes from the shared operator when available; otherwise
    // fall back to the dense design list.
    let d = if let Some(ref op) = aniso_result.implicit_operator {
        op.n_axes()
    } else if !aniso_result.design_first.is_empty() {
        aniso_result.design_first.len()
    } else {
        0
    };
    if d == 0 {
        log::warn!(
            "[spatial-kappa] term {term_idx}: per-axis ψ declined -- the producer reported zero \
             axes (no implicit operator and no dense design list)"
        );
        return Ok(None);
    }
    let Some(penalty_range) = design
        .smooth_term_penalty_range(term_idx)
        .map_err(EstimationError::InvalidInput)?
    else {
        log::warn!(
            "[spatial-kappa] term {term_idx}: per-axis ψ declined -- the realized design exposes \
             no penalty range for this term"
        );
        return Ok(None);
    };
    let penalty_start = penalty_range.start;
    let p_total = design.design.ncols();
    let smooth_start = p_total.saturating_sub(design.smooth.total_smooth_cols());
    let global_range = (smooth_start + smooth_term.coeff_range.start)
        ..(smooth_start + smooth_term.coeff_range.end);
    // CANDIDATE → FITTED alignment (#2750). The ψ producer emits one derivative
    // block per EMITTED penalty candidate, in the builder's original order; the
    // realized design keeps only the survivors of `filter_penalty_candidates`,
    // and `ActivePenaltyInfo::original_index` is the map between them. Zipping
    // the emitted blocks positionally against `penalty_start + j` was correct
    // only while nothing was ever dropped — a measure-jet term genuinely loses
    // its null-component candidate at a long representer range, and the fit
    // then aborted with `penalty_index for dir 0 out of bounds: 1 >= 1`.
    // Selecting the survivors here, once, keeps every downstream zip positional.
    //
    // The Duchon producer is the exception, and deliberately: its penalty
    // ψ-derivative builders run `filter_penalty_candidates` themselves, on the
    // same candidates the design filtered, so they emit the ACTIVE list already
    // — the identical convention its isotropic sibling
    // (`try_build_spatial_term_log_kappa_derivativeinfo`) relies on when it zips
    // positionally against `penalty_start + j`. Re-selecting by
    // `original_index` there would index an active list with emitted indices.
    // The equal-length check below is what makes the two conventions
    // distinguishable rather than merely asserted (gam#2735).
    let emitted = aniso_result.penalties_first[0].len();
    let producer_emits_active_list = matches!(&termspec.basis, SmoothBasisSpec::Duchon { .. });
    let keep: Vec<usize> = if producer_emits_active_list {
        if emitted != smooth_term.active_penalties.len() {
            log::warn!(
                "[spatial-kappa] term {term_idx}: per-axis ψ declined -- the Duchon producer \
                 emitted {emitted} active penalty block(s) but the realized design carries {} \
                 ({:?}); the term falls back to its isotropic axis",
                smooth_term.active_penalties.len(),
                smooth_term
                    .active_penalties
                    .iter()
                    .map(|active| active.info.source.clone())
                    .collect::<Vec<_>>()
            );
            return Ok(None);
        }
        (0..emitted).collect()
    } else {
        smooth_term
            .active_penalties
            .iter()
            .map(|active| active.info.original_index)
            .collect()
    };
    if keep.is_empty() || keep.iter().any(|&index| index >= emitted) {
        log::warn!(
            "[spatial-kappa] term {term_idx}: per-axis ψ declined -- candidate→fitted map \
             {keep:?} does not index the {emitted} emitted block(s)"
        );
        return Ok(None);
    }
    let penalty_indices: Vec<usize> = (0..keep.len()).map(|j| penalty_start + j).collect();
    let penalties_cross_provider = aniso_result.penalties_cross_provider.clone();
    /// Take the emitted per-candidate blocks and keep only the fitted ones, in
    /// fitted order. Consumes the source so no `(n × p)` block is cloned.
    fn select_fitted(blocks: Vec<Array2<f64>>, keep: &[usize]) -> Vec<Array2<f64>> {
        let mut slots: Vec<Option<Array2<f64>>> = blocks.into_iter().map(Some).collect();
        keep.iter()
            .filter_map(|&index| slots.get_mut(index).and_then(Option::take))
            .collect()
    }

    // The per-axis path has the same collection geometry as the isotropic
    // builder: keep the current coefficient chart fixed and left-project every
    // design jet into the collection's fixed row-space complement (gam#2760).
    // The shared implicit operator covers first, diagonal-second, and cross-
    // second axes; dense fallbacks are projected here by the identical object.
    // gam#2760 — the same stage-2 joint-null rotation the isotropic arm applies
    // in `try_build_spatial_term_log_kappa_derivative`. The realized design this
    // derivative is graded against is `X·Q`, so an unrotated per-axis block is a
    // derivative of a different model. Before the row-space projector: `Q` acts
    // on coefficients, the projector on rows.
    if let Some(rotation) = smooth_term.joint_null_rotation.as_ref() {
        let Some(rotated) = aniso_result
            .rotated_by_joint_null(rotation)
            .map_err(EstimationError::from)?
        else {
            log::warn!(
                "[spatial-kappa] term {term_idx}: per-axis ψ declined -- the realized design \
                 carries a joint-null rotation of {} coefficients that the per-axis derivative \
                 blocks do not admit",
                rotation.rotation.nrows()
            );
            return Ok(None);
        };
        aniso_result = rotated;
    }

    if let Some(gauge) = smooth_term.collection_gauge.as_ref() {
        let projector = FixedRowSpaceProjector::from_constraint_block(
            gauge.constraint_block.view(),
        )
        .map_err(EstimationError::from)?;
        for matrix in aniso_result
            .design_first
            .iter_mut()
            .chain(aniso_result.design_second_diag.iter_mut())
            .chain(aniso_result.design_second_cross.iter_mut())
        {
            if matrix.nrows() > 0 {
                projector
                    .project_matrix_in_place(matrix)
                    .map_err(EstimationError::from)?;
            }
        }
        if let Some(operator) = aniso_result.implicit_operator.take() {
            aniso_result.implicit_operator = Some(
                operator
                    .with_fixed_row_space_projection(projector)
                    .map_err(EstimationError::from)?,
            );
        }
    }

    // Dense first/diagonal-second matrices may be present even when the shared
    // operator is available. The operator remains the canonical source for
    // exact cross-axis second derivatives.
    let use_implicit_design = aniso_result.design_first.is_empty();
    let implicit_op_arc = aniso_result
        .implicit_operator
        .as_ref()
        .map(|op| std::sync::Arc::new(op.clone()));

    let mut entries = Vec::with_capacity(d);
    for a in 0..d {
        let (x_psi_local, x_psi_psi_local) = if use_implicit_design {
            // Implicit path: design-derivative matvecs will be dispatched through
            // the ImplicitDerivativeOp inside HyperDesignDerivative, so we do NOT
            // need to materialize the dense (n x p) matrices here.  Store empty
            // placeholders — they are never read when the implicit operator is
            // present (spatial_log_kappa_hyper_dirs_frominfo_list uses from_implicit).
            (Array2::<f64>::zeros((0, 0)), Array2::<f64>::zeros((0, 0)))
        } else {
            // Move the dense (n × p) matrices out of aniso_result instead of
            // cloning. Each axis index `a` is read exactly once across the
            // loop, and aniso_result is dropped at function exit, so leaving
            // empty placeholders behind in those vec slots is safe.
            let x_first = std::mem::take(&mut aniso_result.design_first[a]);
            let x_second = std::mem::take(&mut aniso_result.design_second_diag[a]);
            if x_first.ncols() != smooth_term.coeff_range.len() {
                return Ok(None);
            }
            (x_first, x_second)
        };
        let s_psi_components =
            select_fitted(std::mem::take(&mut aniso_result.penalties_first[a]), &keep);
        let s_psi_psi_components = select_fitted(
            std::mem::take(&mut aniso_result.penalties_second_diag[a]),
            &keep,
        );
        // Build cross-design entries for other axes b != a in this group.
        // These will be indexed by (b, cross_matrix) where b is the axis
        // offset within the d-entry block.
        // Cross-axis second derivatives are sourced from the shared operator,
        // so we only need placeholder entries to preserve the axis layout.
        let cross_designs = if implicit_op_arc.is_some() {
            let mut cd = Vec::with_capacity(d - 1);
            for b in 0..d {
                if b == a {
                    continue;
                }
                cd.push((b, Array2::<f64>::zeros((0, 0))));
            }
            cd
        } else if !aniso_result.design_second_cross.is_empty() {
            let mut cd = Vec::new();
            for (cross_idx, &(pa, pb)) in aniso_result.design_second_cross_pairs.iter().enumerate()
            {
                if pa == a {
                    cd.push((pb, aniso_result.design_second_cross[cross_idx].clone()));
                } else if pb == a {
                    cd.push((pa, aniso_result.design_second_cross[cross_idx].clone()));
                }
            }
            cd
        } else {
            Vec::new()
        };
        let cross_penalty_provider = if d > 1 {
            let penalties_cross_provider = penalties_cross_provider.clone();
            let keep_cross = keep.clone();
            Some(std::sync::Arc::new(
                move |b_axis: usize| -> Result<Vec<Array2<f64>>, EstimationError> {
                    if b_axis == a {
                        return Ok(Vec::new());
                    }
                    let (axis_lo, axis_hi) = if a < b_axis { (a, b_axis) } else { (b_axis, a) };
                    if let Some(provider) = penalties_cross_provider.as_ref() {
                        // Same candidate → fitted selection as the diagonal
                        // blocks above; the provider also answers in the
                        // builder's emitted order.
                        provider
                            .evaluate(axis_lo, axis_hi)
                            .map_err(EstimationError::from)
                            .map(|blocks| select_fitted(blocks, &keep_cross))
                    } else {
                        // No provider: either the pair is unregistered, or it
                        // was registered without data (early-return raw-operator
                        // paths). Both cases contribute no cross penalties.
                        Ok(Vec::new())
                    }
                },
            )
                as std::sync::Arc<
                    dyn Fn(usize) -> Result<Vec<Array2<f64>>, EstimationError>
                        + Send
                        + Sync
                        + 'static,
                >)
        } else {
            None
        };

        entries.push(SpatialPsiDerivative {
            penalty_index: penalty_indices[0],
            penalty_indices: penalty_indices.clone(),
            global_range: global_range.clone(),
            total_p: p_total,
            x_psi_local,
            s_psi_components_local: s_psi_components,
            x_psi_psi_local,
            s_psi_psi_components_local: s_psi_psi_components,
            aniso_group_id: Some(aniso_group_id),
            aniso_cross_designs: if cross_designs.is_empty() {
                None
            } else {
                Some(cross_designs)
            },
            aniso_cross_penalty_provider: cross_penalty_provider,
            implicit_operator: implicit_op_arc.clone(),
            implicit_axis: a,
        });
    }
    Ok(Some(entries))
}

#[cfg(test)]
mod glm_eta_observation_fd_tests {
    //! #1615/#1616: the non-Gaussian GLM arms of `evaluate_standard_familyobservations`
    //! (Poisson / Gamma / NegativeBinomial / Tweedie) must have a self-consistent
    //! derivative tower: `score = ∂ℓ/∂η`, `neghessian_eta = −∂(score)/∂η`, and
    //! `neghessian_eta_derivative = ∂(neghessian_eta)/∂η`. Pin each against central
    //! finite differences of the assembled log-likelihood / score.
    use super::*;
    use ndarray::array;

    fn one_obs_weight(
        spec: &LikelihoodSpec,
        y: f64,
        weight: f64,
        eta: f64,
    ) -> StandardFamilyObservationState {
        let yv = Array1::from_vec(vec![y]);
        let wv = Array1::from_vec(vec![weight]);
        let ev = Array1::from_vec(vec![eta]);
        evaluate_standard_familyobservations(spec.clone(), None, None, None, &yv, &wv, &ev)
            .expect("standard family observation state assembles")
    }

    fn one_obs(spec: &LikelihoodSpec, y: f64, eta: f64) -> StandardFamilyObservationState {
        one_obs_weight(spec, y, 1.0, eta)
    }

    #[test]
    fn bounded_zero_rows_are_dormant_and_weight_preflight_is_atomic() {
        let likelihood = gam_spec::GlmLikelihoodSpec::canonical(LikelihoodSpec::poisson_log());
        let dormant = evaluate_resolved_standard_family_observations(
            &likelihood,
            None,
            None,
            None,
            &array![f64::NAN, 2.0],
            &array![0.0, 1.0],
            &array![f64::NAN, 0.2],
        )
        .expect("zero-weight response and predictor are dormant");
        assert_eq!(dormant.score[0], 0.0);
        assert_eq!(dormant.fisherweight[0], 0.0);

        let error = evaluate_resolved_standard_family_observations(
            &likelihood,
            None,
            None,
            None,
            &array![f64::NAN, 2.0],
            &array![1.0, f64::NAN],
            &array![f64::NAN, 0.2],
        )
        .expect_err("later invalid weight must refuse before row evaluation");
        // The offending weight is `weights[1]`, and row indices here are 0-based
        // (same base as the typed `PirlsRowGeometryUnrepresentable { row }` this
        // function's other refusal carries). This expectation read `row 2` while
        // the preflight emitted `i + 1`; both have been put on the 0-based base.
        assert!(
            error.to_string().contains("row 1 has invalid prior weight"),
            "unexpected atomic preflight error: {error}"
        );
    }

    fn check_fd(label: &str, spec: &LikelihoodSpec, y: f64, eta: f64) {
        let h = 1e-5;
        let s0 = one_obs(spec, y, eta);
        let sp = one_obs(spec, y, eta + h);
        let sm = one_obs(spec, y, eta - h);

        // score = d(log_likelihood)/d(eta)
        let score_fd = (sp.log_likelihood - sm.log_likelihood) / (2.0 * h);
        let score = s0.score[0];
        assert!(
            (score - score_fd).abs() <= 1e-4 * (1.0 + score.abs()),
            "{label}: score {score} vs FD {score_fd}"
        );

        // neghessian_eta = -d(score)/d(eta)
        let neghess_fd = -(sp.score[0] - sm.score[0]) / (2.0 * h);
        let neghess = s0.neghessian_eta[0];
        assert!(
            (neghess - neghess_fd).abs() <= 1e-3 * (1.0 + neghess.abs()),
            "{label}: neghessian_eta {neghess} vs FD {neghess_fd}"
        );

        // neghessian_eta_derivative = d(neghessian_eta)/d(eta)
        let nhd_fd = (sp.neghessian_eta[0] - sm.neghessian_eta[0]) / (2.0 * h);
        let nhd = s0.neghessian_eta_derivative[0];
        assert!(
            (nhd - nhd_fd).abs() <= 1e-2 * (1.0 + nhd.abs()),
            "{label}: neghessian_eta_derivative {nhd} vs FD {nhd_fd}"
        );
    }

    #[test]
    fn poisson_gamma_nb_tweedie_arms_match_finite_differences_1615_1616() {
        let log = InverseLink::Standard(StandardLink::Log);
        let poisson = LikelihoodSpec {
            response: ResponseFamily::Poisson,
            link: log.clone(),
        };
        check_fd("poisson y=3", &poisson, 3.0, 0.4);
        check_fd("poisson y=0", &poisson, 0.0, -0.2);

        let gamma = LikelihoodSpec {
            response: ResponseFamily::Gamma,
            link: log.clone(),
        };
        check_fd("gamma y=2.5", &gamma, 2.5, 0.3);
        check_fd("gamma y=0.7", &gamma, 0.7, -0.1);

        let nb = LikelihoodSpec {
            response: ResponseFamily::NegativeBinomial {
                theta: 1.5,
                theta_fixed: true,
            },
            link: log.clone(),
        };
        check_fd("negbin y=4", &nb, 4.0, 0.5);
        check_fd("negbin y=0", &nb, 0.0, -0.3);

        let tweedie = LikelihoodSpec {
            response: ResponseFamily::Tweedie { p: 1.5 },
            link: log.clone(),
        };
        check_fd("tweedie y=2", &tweedie, 2.0, 0.25);
        check_fd("tweedie y=0.5", &tweedie, 0.5, -0.15);
    }

    #[test]
    fn log_link_tails_balance_tiny_weights_before_certification() {
        let poisson = one_obs_weight(&LikelihoodSpec::poisson_log(), 0.0, 1e-300, 700.0);
        assert!(poisson.fisherweight[0].is_finite() && poisson.fisherweight[0] > 1.0);
        assert!(poisson.score[0].is_finite());
        assert!(poisson.log_likelihood.is_finite());

        let gamma = one_obs_weight(&LikelihoodSpec::gamma_log(), 1.0, 1e-300, -700.0);
        assert!(gamma.neghessian_eta[0].is_finite() && gamma.neghessian_eta[0] > 1.0);
        assert!(gamma.score[0].is_finite());
        assert!(gamma.log_likelihood.is_finite());
    }

    #[test]
    fn invalid_weights_are_refused_in_row_order_and_zero_weight_rows_are_excluded() {
        let family = LikelihoodSpec::gaussian_identity();
        let y = array![1.0, 2.0];
        let eta = array![0.0, 0.0];
        for weights in [array![-1.0, 1.0], array![f64::NAN, 1.0]] {
            let err = evaluate_standard_familyobservations(
                family.clone(),
                None,
                None,
                None,
                &y,
                &weights,
                &eta,
            )
            .expect_err("invalid prior weight must be refused");
            assert!(err.to_string().contains("row 0"), "{err}");
        }

        // #2618: a non-finite response is refused exactly when the row carries
        // likelihood mass. A prior weight of exactly zero is this project's
        // "excluded row" convention — such a row must be equivalent to omitting
        // it entirely (`gam_solve::gaussian_reml::effective_observation_count`
        // states the convention; `validate_dispersion_family_data` exempts
        // zero-weight rows from the response-support check and its refusal tells
        // callers to "set the row's prior weight to 0 to exclude it"; the PIRLS
        // family state substitutes `z = eta` instead of reading `y` there). This
        // assertion used to demand the opposite of the dormant-row test above
        // for the same call, which no implementation could satisfy.
        let err = evaluate_standard_familyobservations(
            family.clone(),
            None,
            None,
            None,
            &array![f64::NAN],
            &array![1.0],
            &array![0.0],
        )
        .expect_err("a non-finite response on a weight-bearing row must be refused");
        assert!(err.to_string().contains("row 0"), "{err}");

        let excluded = evaluate_standard_familyobservations(
            family,
            None,
            None,
            None,
            &array![f64::NAN],
            &array![0.0],
            &array![0.0],
        )
        .expect("a zero-weight row is excluded, so its response is never inspected");
        assert_eq!(excluded.score[0], 0.0);
        assert_eq!(excluded.fisherweight[0], 0.0);
        assert_eq!(excluded.neghessian_eta[0], 0.0);
        assert_eq!(excluded.log_likelihood, 0.0);
    }

    #[test]
    fn bounded_covariance_requires_a_certified_strict_spd_precision() {
        let covariance = certified_bounded_posterior_covariance(
            &array![[4.0, 1.0], [1.0, 3.0]],
            "bounded covariance regression",
        )
        .expect("strict SPD precision");
        assert!((covariance[[0, 0]] - 3.0 / 11.0).abs() < 1e-14);
        assert!((covariance[[0, 1]] + 1.0 / 11.0).abs() < 1e-14);
        assert!((covariance[[1, 1]] - 4.0 / 11.0).abs() < 1e-14);

        for invalid in [
            array![[1.0, 1.0], [1.0, 1.0]],
            array![[1.0, 2.0], [2.0, 1.0]],
        ] {
            assert!(
                certified_bounded_posterior_covariance(
                    &invalid,
                    "invalid bounded covariance regression"
                )
                .is_err(),
                "singular/indefinite precision must not become a pseudo-covariance"
            );
        }
    }
}

#[cfg(test)]
mod refit_seed_2902_tests {
    use super::*;
    use gam_solve::estimate::outer_eval_capture::{
        OuterSeedLayout, OuterSeedProbe, observe_next_outer_seed,
    };
    use gam_terms::basis::{
        BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
        OneDimensionalBoundary,
    };
    use std::cell::RefCell;
    use std::rc::Rc;

    /// #2902 row 9: a refit seeded from a converged fit starts the outer search at
    /// that fit's own `ρ*`, bit for bit and strictly inside the declared domain.
    /// The spatial refits after κ selection passed the physical `lambdas`, which
    /// the outer search reads as log strengths, so the refit started at
    /// `clamp(e^{ρ*})` instead.
    #[test]
    fn a_refit_from_a_converged_fit_starts_at_its_log_strengths_2902() {
        let n = 120usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = x;
            y[i] = (0.4 + 1.2 * (2.0 * std::f64::consts::PI * x).sin())
                .exp()
                .round();
        }
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: "s".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 8,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
                        boundary: OneDimensionalBoundary::Open,
                        boundary_conditions: BSplineBoundaryConditions::default(),
                    },
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let family = LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        );
        let options = FitOptions::default();
        let converged = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            family.clone(),
            &options,
        )
        .expect("the Poisson s(x) fixture fits");
        let rho_star = converged.fit.log_lambdas.clone();

        let seen: Rc<RefCell<Option<OuterSeedLayout>>> = Rc::new(RefCell::new(None));
        let sink = Rc::clone(&seen);
        observe_next_outer_seed(
            0,
            Box::new(
                move |probe: &mut dyn OuterSeedProbe| -> Result<(), EstimationError> {
                    *sink.borrow_mut() = Some(probe.layout().clone());
                    Ok(())
                },
            ),
        );
        let refit = refit_term_collection_at_fitted_strengths(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            &converged.fit,
            family,
            &options,
        );
        assert!(
            refit.is_ok(),
            "the refit at the converged strengths must fit: {:?}",
            refit.err()
        );
        let layout = seen
            .borrow_mut()
            .take()
            .expect("the refit's outer search entered a seed");
        let bits = |values: &Array1<f64>| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&layout.seed),
            bits(&rho_star),
            "the refit's first seed {:?} is not the converged log strengths {:?}",
            layout.seed,
            rho_star
        );
        for index in 0..rho_star.len() {
            assert!(
                layout.lower[index] < rho_star[index] && rho_star[index] < layout.upper[index],
                "rho* {} must sit strictly inside the domain [{}, {}] on coordinate {index}",
                rho_star[index],
                layout.lower[index],
                layout.upper[index]
            );
        }
    }
}
