// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

#[derive(Clone, Copy)]
pub(crate) struct GamlssLambdaLayout {
    pub(crate) k_mean: usize,
    pub(crate) k_noise: usize,
    pub(crate) kwiggle: usize,
}

impl GamlssLambdaLayout {
    pub(crate) fn two_block(k_mean: usize, k_noise: usize) -> Self {
        Self {
            k_mean,
            k_noise,
            kwiggle: 0,
        }
    }

    pub(crate) fn withwiggle(k_mean: usize, k_noise: usize, kwiggle: usize) -> Self {
        Self {
            k_mean,
            k_noise,
            kwiggle,
        }
    }

    pub(crate) fn total(self) -> usize {
        self.k_mean + self.k_noise + self.kwiggle
    }

    pub(crate) fn noise_start(self) -> usize {
        self.k_mean
    }

    pub(crate) fn noise_end(self) -> usize {
        self.k_mean + self.k_noise
    }

    pub(crate) fn wiggle_start(self) -> usize {
        self.k_mean + self.k_noise
    }

    pub(crate) fn wiggle_end(self) -> usize {
        self.k_mean + self.k_noise + self.kwiggle
    }

    pub(crate) fn validate_theta_len(self, theta_len: usize, context: &str) -> Result<(), String> {
        let needed = self.total();
        if theta_len < needed {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "{context} theta too short: got {}, need at least {}",
                    theta_len, needed
                ),
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn mean_from(self, theta: &Array1<f64>) -> Array1<f64> {
        theta.slice(s![0..self.k_mean]).to_owned()
    }

    pub(crate) fn noise_from(self, theta: &Array1<f64>) -> Array1<f64> {
        theta
            .slice(s![self.noise_start()..self.noise_end()])
            .to_owned()
    }

    pub(crate) fn wiggle_from(self, theta: &Array1<f64>) -> Array1<f64> {
        theta
            .slice(s![self.wiggle_start()..self.wiggle_end()])
            .to_owned()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct GamlssBetaLayout {
    pub(crate) pt: usize,
    pub(crate) pls: usize,
    pub(crate) pw: usize,
}

impl GamlssBetaLayout {
    pub(crate) fn withwiggle(pt: usize, pls: usize, pw: usize) -> Self {
        Self { pt, pls, pw }
    }

    pub(crate) fn total(self) -> usize {
        self.pt + self.pls + self.pw
    }

    pub(crate) fn split_three(
        self,
        flat: &Array1<f64>,
        context: &str,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
        if flat.len() != self.total() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "{context} length mismatch: got {}, expected {}",
                    flat.len(),
                    self.total()
                ),
            }
            .into());
        }
        Ok((
            flat.slice(s![0..self.pt]).to_owned(),
            flat.slice(s![self.pt..self.pt + self.pls]).to_owned(),
            flat.slice(s![self.pt + self.pls..self.total()]).to_owned(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct FamilyMetadata {
    pub name: &'static str,
    pub parameternames: &'static [&'static str],
    pub parameter_links: &'static [ParameterLink],
}

pub(crate) const DEFAULT_GAUGE_PRIORITY: u8 = 100;

pub(crate) const LINK_WIGGLE_GAUGE_PRIORITY: u8 = 80;

pub(crate) fn initial_log_lambdas_orzeros(
    block: &ParameterBlockInput,
) -> Result<Array1<f64>, String> {
    let k = block.penalties.len();
    let lambdas = block
        .initial_log_lambdas
        .clone()
        .unwrap_or_else(|| Array1::<f64>::zeros(k));
    if lambdas.len() != k {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "initial_log_lambdas length mismatch: got {}, expected {}",
                lambdas.len(),
                k
            ),
        }
        .into());
    }
    Ok(lambdas)
}

pub(crate) fn build_two_block_exact_joint_setup(
    data: ArrayView2<'_, f64>,
    meanspec: &TermCollectionSpec,
    noisespec: &TermCollectionSpec,
    mean_penalties: usize,
    noise_penalties: usize,
    extra_rho0: &[f64],
    rho0_override: Option<&Array1<f64>>,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    // GAMLSS-specific part: assemble the rho seed in [mean | noise | extra]
    // penalty order, honoring a caller override when it matches the layout.
    let rho_dim = mean_penalties + noise_penalties + extra_rho0.len();
    let mut rho0vec = Array1::<f64>::zeros(rho_dim);
    if let Some(rho0) = rho0_override.filter(|rho0| rho0.len() == rho_dim) {
        rho0vec.assign(rho0);
    } else {
        for (i, &rho_init) in extra_rho0.iter().enumerate() {
            rho0vec[mean_penalties + noise_penalties + i] = rho_init;
        }
    }

    // Generic part: per-block log(kappa) seed/bounds and exact-joint assembly,
    // with the two linear predictors (mean, noise) in theta order.
    build_location_scale_exact_joint_setup(data, &[meanspec, noisespec], rho0vec, kappa_options)
}

pub(crate) fn solve_penalizedweighted_projection(
    design: &DesignMatrix,
    offset: &Array1<f64>,
    target_eta: &Array1<f64>,
    weights: &Array1<f64>,
    penalties: &[PenaltyMatrix],
    log_lambdas: &Array1<f64>,
    ridge_floor: f64,
) -> Result<Array1<f64>, String> {
    let n = design.nrows();
    let p = design.ncols();
    if offset.len() != n || target_eta.len() != n || weights.len() != n {
        return Err(GamlssError::DimensionMismatch {
            reason: "solve_penalizedweighted_projection dimension mismatch".to_string(),
        }
        .into());
    }
    if penalties.len() != log_lambdas.len() {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "solve_penalizedweighted_projection lambda mismatch: penalties={}, log_lambdas={}",
                penalties.len(),
                log_lambdas.len()
            ),
        }
        .into());
    }

    let y_star = target_eta - offset;
    let xtwy = design.compute_xtwy(weights, &y_star)?;
    let mut penalty_system = if penalties.is_empty() {
        None
    } else {
        Some(Array2::<f64>::zeros((p, p)))
    };
    for (k, s) in penalties.iter().enumerate() {
        let lambda = log_lambdas[k].exp();
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(GamlssError::NumericalFailure { reason: format!(
                "solve_penalizedweighted_projection encountered invalid lambda at index {k}: {}",
                log_lambdas[k]
            ) }.into());
        }
        if s.nrows() != p || s.ncols() != p {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "solve_penalizedweighted_projection penalty shape mismatch at index {k}: \
                 penalty is {}x{} but design has {} columns",
                    s.nrows(),
                    s.ncols(),
                    p
                ),
            }
            .into());
        }
        if let Some(system) = penalty_system.as_mut() {
            s.add_scaled_to(lambda, system);
        }
    }

    let beta = design.solve_systemwith_policy(
        weights,
        &xtwy,
        penalty_system.as_ref(),
        ridge_floor.max(1e-12),
        RidgePolicy::explicit_stabilization_pospart(),
    )?;
    if beta.iter().any(|v| !v.is_finite()) {
        return Err(
            "solve_penalizedweighted_projection produced non-finite coefficients".to_string(),
        );
    }
    Ok(beta)
}

pub(crate) fn gaussian_location_scalewarm_start(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    mu_block: &ParameterBlockSpec,
    log_sigma_block: &ParameterBlockSpec,
    ridge_floor: f64,
    mean_beta_hint: Option<&Array1<f64>>,
    noise_beta_hint: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array1<f64>, f64), String> {
    let betamu = if let Some(beta) = mean_beta_hint {
        beta.clone()
    } else {
        solve_penalizedweighted_projection(
            &mu_block.design,
            &mu_block.offset,
            y,
            weights,
            &mu_block.penalties,
            &mu_block.initial_log_lambdas,
            ridge_floor,
        )?
    };
    let mut mu_hat = mu_block.solver_design().matrixvectormultiply(&betamu);
    mu_hat += mu_block.solver_offset();
    let mut weighted_ss = 0.0;
    let mut weight_sum = 0.0;
    for i in 0..y.len() {
        let wi = weights[i].max(0.0);
        let resid = y[i] - mu_hat[i];
        weighted_ss += wi * resid * resid;
        weight_sum += wi;
    }
    if !weighted_ss.is_finite() || !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(
            "gaussian location-scale warm start could not estimate residual scale".to_string(),
        );
    }
    // Warm-start σ̂ must clear the logb floor so the inverse link
    //   η = log(σ − b)
    // is finite. Use a relative cushion above b so the warm-start is in the
    // smooth interior of the link domain.
    let sigma_hat = (weighted_ss / weight_sum)
        .sqrt()
        .max(LOGB_SIGMA_FLOOR * 1.5);
    let beta_log_sigma = if let Some(beta) = noise_beta_hint {
        beta.clone()
    } else {
        let eta_sigma = (sigma_hat - LOGB_SIGMA_FLOOR).ln();
        let sigma_target = Array1::from_elem(y.len(), eta_sigma);
        solve_penalizedweighted_projection(
            &log_sigma_block.design,
            &log_sigma_block.offset,
            &sigma_target,
            weights,
            &log_sigma_block.penalties,
            &log_sigma_block.initial_log_lambdas,
            ridge_floor,
        )?
    };
    Ok((betamu, beta_log_sigma, sigma_hat))
}

/// Total output count for every two-block location-scale family in this
/// module (mu/log_sigma or threshold/log_sigma). The wiggle variants add a
/// third zero-channel block but still drive only two output channels.
pub(crate) const LOCATION_SCALE_N_OUTPUTS: usize = 2;

/// Construct a fully wired location-scale parameter block.
///
/// This is the **only** way to build a LocationScale `ParameterBlockSpec` in
/// this module — by construction the `AdditiveBlockJacobian` callback is
/// always installed, so the channel-aware identifiability audit cannot be
/// silently bypassed by a future `build_blocks` impl that forgets to wire
/// the callback at the tail (re-introducing #319).
///
/// `own_output` is the zero-based output channel this block drives
/// (e.g. 0 for `mu`/`threshold`, 1 for `log_sigma`). `n_family_outputs` is
/// fixed at [`LOCATION_SCALE_N_OUTPUTS`] for every two-block family here
/// but is exposed so the helper composes cleanly with any future
/// k-block extension.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_location_scale_block(
    name: impl Into<String>,
    design: DesignMatrix,
    offset: Array1<f64>,
    penalties: Vec<PenaltyMatrix>,
    nullspace_dims: Vec<usize>,
    initial_log_lambdas: Array1<f64>,
    initial_beta: Option<Array1<f64>>,
    own_output: usize,
    n_family_outputs: usize,
    caller: &str,
) -> Result<ParameterBlockSpec, String> {
    if own_output >= n_family_outputs {
        return Err(format!(
            "{caller}: own_output={own_output} >= n_family_outputs={n_family_outputs}"
        ));
    }
    let mut spec = ParameterBlockSpec {
        name: name.into(),
        design,
        offset,
        penalties,
        nullspace_dims,
        initial_log_lambdas,
        initial_beta,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let dense = spec.effective_design(caller)?;
    spec.jacobian_callback = Some(std::sync::Arc::new(AdditiveBlockJacobian {
        design: dense,
        own_output,
        n_family_outputs,
    }));
    Ok(spec)
}

/// Construct the wiggle block that accompanies a two-block location-scale
/// family. The wiggle modulates the inverse link nonlinearly and
/// contributes no linear effective Jacobian — the installed callback
/// therefore exposes a zero `(n × p_w)` design under
/// `n_family_outputs = LOCATION_SCALE_N_OUTPUTS`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_location_scale_wiggle_block(
    name: impl Into<String>,
    design: DesignMatrix,
    offset: Array1<f64>,
    penalties: Vec<PenaltyMatrix>,
    nullspace_dims: Vec<usize>,
    initial_log_lambdas: Array1<f64>,
    initial_beta: Option<Array1<f64>>,
    n_rows: usize,
) -> Result<ParameterBlockSpec, String> {
    let p_w = design.ncols();
    let mut spec = ParameterBlockSpec {
        name: name.into(),
        design,
        offset,
        penalties,
        nullspace_dims,
        initial_log_lambdas,
        initial_beta,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    spec.jacobian_callback = Some(std::sync::Arc::new(AdditiveBlockJacobian {
        design: ndarray::Array2::<f64>::zeros((n_rows, p_w)),
        own_output: 0,
        n_family_outputs: LOCATION_SCALE_N_OUTPUTS,
    }));
    Ok(spec)
}

pub(crate) fn prepared_gaussian_log_sigma_design(
    mu_design: &DesignMatrix,
    log_sigma_design: &DesignMatrix,
) -> Result<DesignMatrix, String> {
    if mu_design.nrows() != log_sigma_design.nrows() {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "gaussian log-sigma design row mismatch: mean rows={}, log_sigma rows={}",
                mu_design.nrows(),
                log_sigma_design.nrows()
            ),
        }
        .into());
    }
    // Gaussian location-scale remains identifiable even when μ and log σ use
    // the same covariate basis:
    //
    //   L(μ, η) = 0.5 * Σ_i [ (y_i - μ_i)^2 exp(-2η_i) + 2η_i ],
    //   μ = X_μ β_μ,  η = X_σ β_σ.
    //
    // Shared columns are not a frame mismatch. β_μ and β_σ enter through
    // different sufficient statistics (residual and residual²), so replacing
    // X_σ with (I - P_{X_μ}) X_σ would impose an extra constraint and can
    // erase real heteroscedastic signal when the two blocks share a basis.
    Ok(log_sigma_design.clone())
}

pub(crate) fn identified_binomial_log_sigma_design(
    threshold_design: &TermCollectionDesign,
    log_sigma_design: &TermCollectionDesign,
    weights: &Array1<f64>,
) -> Result<DesignMatrix, String> {
    let non_intercept_start = log_sigma_design
        .intercept_range
        .end
        .min(log_sigma_design.design.ncols());
    let transform = build_scale_deviation_transform_design(
        &threshold_design.design,
        &log_sigma_design.design,
        weights,
        non_intercept_start,
    )?;
    build_scale_deviation_operator(
        threshold_design.design.clone(),
        log_sigma_design.design.clone(),
        &transform,
    )
}

pub(crate) fn identity_penalty(dim: usize) -> Array2<f64> {
    let mut penalty = Array2::<f64>::zeros((dim, dim));
    for i in 0..dim {
        penalty[[i, i]] = 1.0;
    }
    penalty
}

/// Orthogonal projector `P₀ = U₀U₀ᵀ` onto the joint null space of the supplied
/// penalty blocks over a `dim`-column coefficient space.
///
/// Used as the log-σ *shrinkage* penalty for the Gaussian location-scale scale
/// block. The smooth's own wiggliness penalty already governs its range space
/// (the curvature directions REML trades off against fit); its null space —
/// the constant + low-order polynomial log-σ trend that carries the dominant
/// heteroscedastic signal — is left unpenalized and is only weakly identified
/// in the coupled (μ, log σ) likelihood, which lets the inner Newton wander
/// (#1073's "flat/ill-conditioned surface"). A *full-space* identity ridge
/// fixed that instability but DOUBLE-penalized the range space: REML then drove
/// the shrinkage λ up, crushing the genuine heteroscedastic curve back to a
/// constant σ (the underfit this issue reports). Penalizing the null space
/// ALONE keeps the weakly-identified polynomial trend from blowing up without
/// touching the wiggliness directions the smooth penalty already controls —
/// exactly mgcv's `select = TRUE` null-space penalty.
///
/// When the supplied penalties already span the whole space (null space empty),
/// the projector is the zero matrix and the shrinkage term is inert; when there
/// are no penalties at all (e.g. a purely parametric log-σ design), the null
/// space is everything and this returns the identity — recovering the previous
/// full-space ridge exactly where it was the right thing to do.
pub(crate) fn penalty_nullspace_projector(penalties: &[PenaltyMatrix], dim: usize) -> Array2<f64> {
    use crate::faer_ndarray::FaerEigh;
    use faer::Side;

    if dim == 0 {
        return Array2::<f64>::zeros((0, 0));
    }
    // Combined penalty S = Σ_k S_k over the dim-column scale space. Each block
    // penalty is already expressed on this space (the scale design's columns).
    let mut combined = Array2::<f64>::zeros((dim, dim));
    for pen in penalties {
        let dense = pen.to_dense();
        assert_eq!(
            dense.nrows(),
            dim,
            "scale penalty block dim {} != scale design cols {dim}",
            dense.nrows()
        );
        if dense.nrows() == dim && dense.ncols() == dim {
            combined += &dense;
        }
    }
    // Symmetrize defensively (eigendecomposition assumes self-adjoint input).
    let combined_sym = 0.5 * (&combined + &combined.t());
    let (eigvals, eigvecs) = match combined_sym.eigh(Side::Lower) {
        Ok(decomp) => decomp,
        // A failed decomposition (degenerate / non-finite) should not silently
        // drop the stabilizing shrinkage; fall back to the full-space ridge,
        // which is the conservative (always-positive-definite) choice.
        Err(_) => return identity_penalty(dim),
    };
    // Null space = eigenvectors whose eigenvalue is ≈ 0 relative to the largest.
    // The combined wiggliness penalty's range-space eigenvalues are O(1) after
    // basis normalization, so a relative floor cleanly separates the genuine
    // null directions (constant / low-order polynomial) from the penalized
    // curvature directions.
    let max_eig = eigvals.iter().cloned().fold(0.0_f64, f64::max);
    let tol = (max_eig * 1e-8).max(1e-12);
    let mut projector = Array2::<f64>::zeros((dim, dim));
    for (j, &lambda) in eigvals.iter().enumerate() {
        if lambda <= tol {
            let v = eigvecs.column(j);
            // Accumulate v vᵀ into the projector.
            for a in 0..dim {
                let va = v[a];
                if va == 0.0 {
                    continue;
                }
                for b in 0..dim {
                    projector[[a, b]] += va * v[b];
                }
            }
        }
    }
    projector
}

pub(crate) fn append_binomial_log_sigma_shrinkage_penalty_design(
    design: &mut TermCollectionDesign,
) {
    let p = design.design.ncols();
    design
        .penalties
        .push(BlockwisePenalty::new(0..p, identity_penalty(p)));
    // Identity penalty penalizes the full space → nullspace dimension is 0.
    design.nullspace_dims.push(0);
    design.penaltyinfo.push(PenaltyBlockInfo {
        global_index: design.penaltyinfo.len(),
        termname: Some("log_sigma_shrinkage".to_string()),
        penalty: PenaltyInfo {
            source: PenaltySource::Other("shrinkage".to_string()),
            original_index: 0,
            active: true,
            effective_rank: p,
            dropped_reason: None,
            nullspace_dim_hint: 0,
            normalization_scale: 1.0,
            kronecker_factors: None,
        },
    });
}

/// Build the (mean, log-σ) parameter-block pair for a Gaussian location-scale
/// family. Shared verbatim by the non-wiggle and wiggle Gaussian builders so the
/// scale-block construction — prepared log-σ design, the REML-selected full-span
/// shrinkage penalty on the scale nullspace, and the joint Gaussian warm start —
/// lives in exactly one place. Callers supply the per-block log-λ vectors sliced
/// from their own layout (two-block vs with-wiggle) and append any extra blocks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_gaussian_mean_and_scale_blocks(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    mean_design: &TermCollectionDesign,
    noise_design: &TermCollectionDesign,
    mean_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
    mean_log_lambdas: Array1<f64>,
    noise_log_lambdas: Array1<f64>,
    mean_beta_hint: Option<Array1<f64>>,
    noise_beta_hint: Option<Array1<f64>>,
    context: &str,
) -> Result<(ParameterBlockSpec, ParameterBlockSpec), String> {
    let mut meanspec = build_location_scale_block(
        "mu",
        mean_design.design.clone(),
        mean_offset.clone(),
        mean_design.penalties_as_penalty_matrix(),
        mean_design.nullspace_dims.clone(),
        mean_log_lambdas,
        mean_beta_hint,
        0,
        LOCATION_SCALE_N_OUTPUTS,
        &format!("{context}: mu"),
    )?;
    let prepared_noise_design =
        prepared_gaussian_log_sigma_design(&mean_design.design, &noise_design.design)?;
    let p_noise = prepared_noise_design.ncols();
    let mut log_sigma_penalty_matrices = noise_design.penalties_as_penalty_matrix();
    // Shrinkage penalty on the scale block's *null space only* (mgcv
    // `select = TRUE`): it stabilizes the weakly-identified constant/polynomial
    // log-σ trend without double-penalizing the wiggliness directions the
    // smooth penalty already governs. A full-space identity here over-shrinks
    // the genuine heteroscedastic curve back to a constant σ (#1073).
    let shrinkage = penalty_nullspace_projector(&log_sigma_penalty_matrices, p_noise);
    // The rank of an orthogonal projector equals its trace (P = P² for a projector,
    // so trace(P) = trace(P²) = ||P||_F² = sum of squared singular values = rank).
    // The diagonal-threshold test `diag[i] > 0.5` used previously was wrong: for a
    // rank-d projector onto a low-dimensional subspace (e.g. d=2 null directions of
    // a TP spline with p=10 columns), each diagonal entry is O(d/p) << 0.5, so the
    // threshold always returned 0 — misreporting the shrinkage penalty as having
    // zero penalized dimensions. Trace-based rank is exact for a symmetric
    // idempotent matrix (rounded to the nearest integer to absorb floating-point
    // rounding in the eigendecomposition).
    let shrinkage_rank = (0..p_noise).map(|i| shrinkage[[i, i]]).sum::<f64>().round() as usize;
    log_sigma_penalty_matrices.push(PenaltyMatrix::Dense(shrinkage));
    let mut log_sigma_nullspace_dims = noise_design.nullspace_dims.clone();
    // The null-space projector penalizes a rank-`shrinkage_rank` subspace, so
    // the remaining unpenalized directions number `p_noise − shrinkage_rank`.
    log_sigma_nullspace_dims.push(p_noise.saturating_sub(shrinkage_rank));
    let mut noisespec = build_location_scale_block(
        "log_sigma",
        prepared_noise_design,
        noise_offset.clone(),
        log_sigma_penalty_matrices,
        log_sigma_nullspace_dims,
        noise_log_lambdas,
        noise_beta_hint,
        1,
        LOCATION_SCALE_N_OUTPUTS,
        &format!("{context}: log_sigma"),
    )?;
    if meanspec.initial_beta.is_none() || noisespec.initial_beta.is_none() {
        let (betamu0, beta_ls0, _) = gaussian_location_scalewarm_start(
            y,
            weights,
            &meanspec,
            &noisespec,
            1e-10,
            meanspec.initial_beta.as_ref(),
            noisespec.initial_beta.as_ref(),
        )?;
        if meanspec.initial_beta.is_none() {
            meanspec.initial_beta = Some(betamu0);
        }
        if noisespec.initial_beta.is_none() {
            noisespec.initial_beta = Some(beta_ls0);
        }
    }
    Ok((meanspec, noisespec))
}

/// Build the (threshold, log-σ) parameter-block pair for a Binomial
/// location-scale family. Shared by the non-wiggle and wiggle Binomial builders;
/// mirrors [`build_gaussian_mean_and_scale_blocks`] but with the binomial-
/// identified log-σ design, the link-aware joint warm start, and the same
/// REML-selected full-span scale shrinkage penalty.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_binomial_threshold_and_scale_blocks(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    link_kind: &InverseLink,
    mean_design: &TermCollectionDesign,
    noise_design: &TermCollectionDesign,
    mean_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
    mean_log_lambdas: Array1<f64>,
    noise_log_lambdas: Array1<f64>,
    mean_beta_hint: Option<Array1<f64>>,
    noise_beta_hint: Option<Array1<f64>>,
    context: &str,
) -> Result<(ParameterBlockSpec, ParameterBlockSpec), String> {
    let identifiednoise_design =
        identified_binomial_log_sigma_design(mean_design, noise_design, weights)?;
    let p_noise = identifiednoise_design.ncols();
    let mut log_sigma_penalty_matrices: Vec<PenaltyMatrix> =
        noise_design.penalties_as_penalty_matrix();
    log_sigma_penalty_matrices.push(PenaltyMatrix::Dense(identity_penalty(p_noise)));
    let mut thresholdspec = build_location_scale_block(
        "threshold",
        mean_design.design.clone(),
        mean_offset.clone(),
        mean_design.penalties_as_penalty_matrix(),
        vec![],
        mean_log_lambdas,
        mean_beta_hint,
        0,
        LOCATION_SCALE_N_OUTPUTS,
        &format!("{context}: threshold"),
    )?;
    let mut log_sigmaspec = build_location_scale_block(
        "log_sigma",
        identifiednoise_design,
        noise_offset.clone(),
        log_sigma_penalty_matrices,
        vec![],
        noise_log_lambdas,
        noise_beta_hint,
        1,
        LOCATION_SCALE_N_OUTPUTS,
        &format!("{context}: log_sigma"),
    )?;
    if thresholdspec.initial_beta.is_none() || log_sigmaspec.initial_beta.is_none() {
        let (beta_t0, beta_ls0) = binomial_location_scalewarm_start(
            y,
            weights,
            link_kind,
            &thresholdspec,
            &log_sigmaspec,
            thresholdspec.initial_beta.as_ref(),
            log_sigmaspec.initial_beta.as_ref(),
        )?;
        if thresholdspec.initial_beta.is_none() {
            thresholdspec.initial_beta = Some(beta_t0);
        }
        if log_sigmaspec.initial_beta.is_none() {
            log_sigmaspec.initial_beta = Some(beta_ls0);
        }
    }
    Ok((thresholdspec, log_sigmaspec))
}

/// Convert a wiggle block's `PenaltySpec`s into the `PenaltyMatrix` list the
/// location-scale wiggle block expects. Shared by the Gaussian and Binomial
/// wiggle builders, which previously inlined the identical match.
pub(crate) fn wiggle_block_penalty_matrices(
    wiggle_block: &ParameterBlockInput,
) -> Vec<PenaltyMatrix> {
    let p_wiggle = wiggle_block.design.ncols();
    wiggle_block
        .penalties
        .iter()
        .map(|spec| match spec {
            crate::model_types::PenaltySpec::Block {
                local, col_range, ..
            } => PenaltyMatrix::Blockwise {
                local: local.clone(),
                col_range: col_range.clone(),
                total_dim: p_wiggle,
            },
            crate::model_types::PenaltySpec::Dense(m)
            | crate::model_types::PenaltySpec::DenseWithMean { matrix: m, .. } => {
                PenaltyMatrix::Dense(m.clone())
            }
        })
        .collect()
}

pub(crate) fn binomial_location_scale_link_eta_from_probability(
    link_kind: &InverseLink,
    probability: f64,
) -> Result<f64, String> {
    let target = probability.clamp(1e-6, 1.0 - 1e-6);
    match link_kind {
        InverseLink::Standard(StandardLink::Logit) => Ok((target / (1.0 - target)).ln()),
        InverseLink::Standard(StandardLink::Probit) => standard_normal_quantile(target)
            .map_err(|err| format!("failed to invert probit warm-start probability: {err}")),
        InverseLink::Standard(StandardLink::CLogLog) => Ok((-((1.0 - target).ln())).ln()),
        other => Err(GamlssError::UnsupportedConfiguration { reason: format!(
            "binomial location-scale warm start requires logit, probit, or cloglog link, got {other:?}"
        ) }.into()),
    }
}

pub(crate) fn weighted_binomial_prevalence(
    y: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<f64, String> {
    if y.len() != weights.len() {
        return Err(GamlssError::DimensionMismatch { reason: format!(
            "binomial location-scale warm start dimension mismatch: y has length {}, weights have length {}",
            y.len(),
            weights.len()
        ) }.into());
    }
    let mut weight_sum = 0.0;
    let mut success_sum = 0.0;
    for (&yi, &wi) in y.iter().zip(weights.iter()) {
        if !yi.is_finite() {
            return Err(GamlssError::NonFinite {
                reason: format!(
                    "binomial location-scale warm start encountered non-finite response {yi}"
                ),
            }
            .into());
        }
        let weight = floor_positiveweight(wi, MIN_WEIGHT);
        if weight > 0.0 {
            weight_sum += weight;
            success_sum += weight * yi;
        }
    }
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(
            "binomial location-scale warm start requires positive total weight".to_string(),
        );
    }
    Ok(success_sum / weight_sum)
}

pub(crate) fn project_constant_eta_into_block(
    block: &ParameterBlockSpec,
    weights: &Array1<f64>,
    eta: f64,
) -> Result<Array1<f64>, String> {
    let target_eta = Array1::from_elem(block.design.nrows(), eta);
    solve_penalizedweighted_projection(
        &block.design,
        &block.offset,
        &target_eta,
        weights,
        &block.penalties,
        &block.initial_log_lambdas,
        1e-10,
    )
}

// Deterministic warm start for the binomial location-scale model. This stays
// out of the optimizer: it projects a prevalence-matched threshold and neutral
// log-sigma value into the actual penalized block spaces.
pub(crate) fn binomial_location_scalewarm_start(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    link_kind: &InverseLink,
    threshold_block: &ParameterBlockSpec,
    log_sigma_block: &ParameterBlockSpec,
    mean_beta_hint: Option<&Array1<f64>>,
    noise_beta_hint: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    if let (Some(mean_beta), Some(noise_beta)) = (mean_beta_hint, noise_beta_hint) {
        return Ok((mean_beta.clone(), noise_beta.clone()));
    }

    let beta_threshold = match mean_beta_hint {
        Some(beta) => beta.clone(),
        None => {
            let prevalence = weighted_binomial_prevalence(y, weights)?;
            let eta = binomial_location_scale_link_eta_from_probability(link_kind, prevalence)?;
            project_constant_eta_into_block(threshold_block, weights, eta)?
        }
    };
    let beta_log_sigma = match noise_beta_hint {
        Some(beta) => beta.clone(),
        None => project_constant_eta_into_block(log_sigma_block, weights, 0.0)?,
    };
    Ok((beta_threshold, beta_log_sigma))
}

#[derive(Clone)]
pub(crate) struct BinomialMeanWiggleSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    pub eta_block: ParameterBlockInput,
    pub wiggle_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct GaussianLocationScaleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub meanspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub mean_offset: Array1<f64>,
    pub log_sigma_offset: Array1<f64>,
}

#[derive(Clone)]
pub struct GaussianLocationScaleWiggleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub meanspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub mean_offset: Array1<f64>,
    pub log_sigma_offset: Array1<f64>,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    pub wiggle_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct BinomialLocationScaleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub thresholdspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub threshold_offset: Array1<f64>,
    pub log_sigma_offset: Array1<f64>,
}

#[derive(Clone)]
pub struct BinomialLocationScaleWiggleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub thresholdspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub threshold_offset: Array1<f64>,
    pub log_sigma_offset: Array1<f64>,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    pub wiggle_block: ParameterBlockInput,
}

#[derive(Clone, Debug)]
pub struct BlockwiseTermFitResult {
    pub fit: UnifiedFitResult,
    pub meanspec_resolved: TermCollectionSpec,
    pub noisespec_resolved: TermCollectionSpec,
    pub mean_design: TermCollectionDesign,
    pub noise_design: TermCollectionDesign,
}

pub(crate) struct BlockwiseTermFitResultParts {
    pub fit: UnifiedFitResult,
    pub meanspec_resolved: TermCollectionSpec,
    pub noisespec_resolved: TermCollectionSpec,
    pub mean_design: TermCollectionDesign,
    pub noise_design: TermCollectionDesign,
}

pub struct BlockwiseTermWiggleFitResult {
    pub fit: BlockwiseTermFitResult,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
}

pub struct BinomialMeanWiggleTermFitResult {
    pub fit: UnifiedFitResult,
    pub resolvedspec: TermCollectionSpec,
    pub design: TermCollectionDesign,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
}

pub(crate) struct BlockwiseTermWiggleFitResultParts {
    pub fit: BlockwiseTermFitResult,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
}

pub(crate) fn validate_term_collection_design(
    label: &str,
    design: &TermCollectionDesign,
) -> Result<(), String> {
    let p = design.design.ncols();
    let n = design.design.nrows();
    for rows in exact_design_row_chunks(n, p) {
        let chunk = design
            .design
            .try_row_chunk(rows)
            .map_err(|e| format!("{label}.design row chunk materialization failed: {e}"))?;
        validate_all_finite_estimation(&format!("{label}.design"), chunk.iter().copied())
            .map_err(|e| e.to_string())?;
    }
    if design.nullspace_dims.len() != design.penalties.len() {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "{label}.nullspace_dims length mismatch: got {}, expected {}",
                design.nullspace_dims.len(),
                design.penalties.len()
            ),
        }
        .into());
    }
    if design.penaltyinfo.len() != design.penalties.len() {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "{label}.penaltyinfo length mismatch: got {}, expected {}",
                design.penaltyinfo.len(),
                design.penalties.len()
            ),
        }
        .into());
    }
    for (idx, bp) in design.penalties.iter().enumerate() {
        validate_all_finite_estimation(
            &format!("{label}.penalties[{idx}]"),
            bp.local.iter().copied(),
        )
        .map_err(|e| e.to_string())?;
        if bp.col_range.end > p {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "{label}.penalties[{idx}] col_range {}..{} exceeds design width {}",
                    bp.col_range.start, bp.col_range.end, p
                ),
            }
            .into());
        }
    }
    if let Some(bounds) = design.coefficient_lower_bounds.as_ref() {
        if bounds.len() != p {
            return Err(GamlssError::ConstraintViolation {
                reason: format!(
                    "{label}.coefficient_lower_bounds length mismatch: got {}, expected {p}",
                    bounds.len()
                ),
            }
            .into());
        }
        for (idx, &bound) in bounds.iter().enumerate() {
            if !(bound.is_finite() || bound == f64::NEG_INFINITY) {
                return Err(GamlssError::NonFinite { reason: format!(
                    "{label}.coefficient_lower_bounds[{idx}] must be finite or -inf, got {bound}",
                ) }.into());
            }
        }
    }
    if let Some(constraints) = design.linear_constraints.as_ref() {
        validate_all_finite_estimation(
            &format!("{label}.linear_constraints.a"),
            constraints.a.iter().copied(),
        )
        .map_err(|e| e.to_string())?;
        validate_all_finite_estimation(
            &format!("{label}.linear_constraints.b"),
            constraints.b.iter().copied(),
        )
        .map_err(|e| e.to_string())?;
        if constraints.a.ncols() != p {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "{label}.linear_constraints.a column mismatch: got {}, expected {p}",
                    constraints.a.ncols()
                ),
            }
            .into());
        }
        if constraints.a.nrows() != constraints.b.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "{label}.linear_constraints row mismatch: a has {}, b has {}",
                    constraints.a.nrows(),
                    constraints.b.len()
                ),
            }
            .into());
        }
    }
    if design.intercept_range.start > design.intercept_range.end || design.intercept_range.end > p {
        return Err(GamlssError::ConstraintViolation {
            reason: format!(
                "{label}.intercept_range out of bounds: {:?} for {} columns",
                design.intercept_range, p
            ),
        }
        .into());
    }
    Ok(())
}

impl BlockwiseTermFitResult {
    pub(crate) fn try_from_parts(parts: BlockwiseTermFitResultParts) -> Result<Self, String> {
        let BlockwiseTermFitResultParts {
            fit,
            meanspec_resolved,
            noisespec_resolved,
            mean_design,
            noise_design,
        } = parts;

        fit.validate_numeric_finiteness()
            .map_err(|e| format!("{e}"))?;
        if fit.block_states.len() < 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BlockwiseTermFitResult requires at least 2 block states, got {}",
                    fit.block_states.len()
                ),
            }
            .into());
        }
        validate_term_collection_design("blockwise_term.mean_design", &mean_design)?;
        validate_term_collection_design("blockwise_term.noise_design", &noise_design)?;
        if mean_design.design.nrows() != noise_design.design.nrows() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BlockwiseTermFitResult row mismatch: mean_design={}, noise_design={}",
                    mean_design.design.nrows(),
                    noise_design.design.nrows()
                ),
            }
            .into());
        }
        if fit.block_states[0].beta.len() != mean_design.design.ncols() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BlockwiseTermFitResult mean beta length mismatch: got {}, expected {}",
                    fit.block_states[0].beta.len(),
                    mean_design.design.ncols()
                ),
            }
            .into());
        }
        if fit.block_states[1].beta.len() != noise_design.design.ncols() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BlockwiseTermFitResult noise beta length mismatch: got {}, expected {}",
                    fit.block_states[1].beta.len(),
                    noise_design.design.ncols()
                ),
            }
            .into());
        }
        if fit.block_states[0].eta.len() != mean_design.design.nrows() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BlockwiseTermFitResult mean eta length mismatch: got {}, expected {}",
                    fit.block_states[0].eta.len(),
                    mean_design.design.nrows()
                ),
            }
            .into());
        }
        if fit.block_states[1].eta.len() != noise_design.design.nrows() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BlockwiseTermFitResult noise eta length mismatch: got {}, expected {}",
                    fit.block_states[1].eta.len(),
                    noise_design.design.nrows()
                ),
            }
            .into());
        }

        Ok(Self {
            fit,
            meanspec_resolved,
            noisespec_resolved,
            mean_design,
            noise_design,
        })
    }

    pub(crate) fn validate_numeric_finiteness(&self) -> Result<(), String> {
        Self::try_from_parts(BlockwiseTermFitResultParts {
            fit: self.fit.clone(),
            meanspec_resolved: self.meanspec_resolved.clone(),
            noisespec_resolved: self.noisespec_resolved.clone(),
            mean_design: self.mean_design.clone(),
            noise_design: self.noise_design.clone(),
        })
        .map(|_| ())
    }
}

impl BlockwiseTermWiggleFitResult {
    pub(crate) fn try_from_parts(parts: BlockwiseTermWiggleFitResultParts) -> Result<Self, String> {
        let BlockwiseTermWiggleFitResultParts {
            fit,
            wiggle_knots,
            wiggle_degree,
        } = parts;

        fit.validate_numeric_finiteness()
            .map_err(|e| e.to_string())?;
        if fit.fit.block_states.len() < 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BlockwiseTermWiggleFitResult requires at least 3 block states, got {}",
                    fit.fit.block_states.len()
                ),
            }
            .into());
        }
        if wiggle_knots.is_empty() {
            return Err(GamlssError::UnsupportedConfiguration {
                reason: "BlockwiseTermWiggleFitResult requires non-empty wiggle_knots".to_string(),
            }
            .into());
        }
        validate_all_finite_estimation(
            "blockwise_term_wiggle.wiggle_knots",
            wiggle_knots.iter().copied(),
        )
        .map_err(|e| e.to_string())?;

        Ok(Self {
            fit,
            wiggle_knots,
            wiggle_degree,
        })
    }
}

pub struct BinomialLocationScaleFitResult {
    pub fit: BlockwiseTermFitResult,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
    pub beta_link_wiggle: Option<Vec<f64>>,
}

pub struct GaussianLocationScaleFitResult {
    pub fit: BlockwiseTermFitResult,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
    pub beta_link_wiggle: Option<Vec<f64>>,
    /// Response standardization factor applied internally during fitting.
    ///
    /// The Gaussian location-scale path fits on `y / response_scale` so the
    /// fixed log-σ soft floor `LOGB_SIGMA_FLOOR = 0.01` is *operationally*
    /// scale-relative (1 % of the response spread) rather than absolute,
    /// keeping κ = dlogσ/dη ≈ 1 across the realistic σ range and informing the
    /// scale block like gamlss. The returned coefficient `blocks`, `beta`, and
    /// link-wiggle knots/coefficients are already mapped back to **raw response
    /// units** (the Location/Mean block scaled by `response_scale`, the Scale
    /// block intercept shifted by `+ln(response_scale)`), so downstream
    /// reconstruction `μ = X_mean·β` comes out in raw units with no further
    /// rescaling.
    ///
    /// The σ reconstruction, however, **must scale the floor too** to stay
    /// response-scale-equivariant (#884):
    ///
    /// ```text
    /// σ = response_scale·LOGB_SIGMA_FLOOR + exp(X_scale·β)
    ///   = response_scale·(LOGB_SIGMA_FLOOR + exp(η_internal)).
    /// ```
    ///
    /// The intercept shift carries only the `exp(η)` term; reconstructing with a
    /// raw `LOGB_SIGMA_FLOOR` instead of `response_scale·LOGB_SIGMA_FLOOR` leaves
    /// the non-equivariant residual `LOGB_SIGMA_FLOOR·(1 − response_scale)`.
    ///
    /// This field records the factor that was applied for transparency,
    /// covariance bookkeeping, and the equivariant σ-floor reconstruction; it is
    /// `1.0` when no standardization was needed (degenerate constant response).
    pub response_scale: f64,
}

pub(crate) fn fit_binomial_mean_wiggle(
    spec: BinomialMeanWiggleSpec,
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validateweights(&spec.weights, "fit_binomial_mean_wiggle")?;
    validate_binomial_response(&spec.y, "fit_binomial_mean_wiggle")?;
    validate_blockrows("eta", n, &spec.eta_block)?;
    validate_blockrows("wiggle", n, &spec.wiggle_block)?;
    if matches!(
        spec.link_kind,
        InverseLink::Standard(StandardLink::Identity)
    ) {
        return Err(GamlssError::UnsupportedConfiguration {
            reason: "fit_binomial_mean_wiggle does not support identity link".to_string(),
        }
        .into());
    }
    crate::inference::formula_dsl::require_binomial_inverse_link_supports_joint_wiggle(
        &spec.link_kind,
        "fit_binomial_mean_wiggle",
    )?;
    if spec.wiggle_degree < 2 {
        return Err(GamlssError::ConstraintViolation {
            reason: format!(
                "fit_binomial_mean_wiggle: wiggle_degree must be >= 2, got {}",
                spec.wiggle_degree
            ),
        }
        .into());
    }
    let minimum_knots = minimum_monotone_wiggle_knot_count(spec.wiggle_degree)?;
    if spec.wiggle_knots.len() < minimum_knots {
        return Err(GamlssError::DimensionMismatch { reason: format!(
            "fit_binomial_mean_wiggle: wiggle_knots length {} is too short for degree {} (need at least {})",
            spec.wiggle_knots.len(),
            spec.wiggle_degree,
            minimum_knots
        ) }.into());
    }

    let family = BinomialMeanWiggleFamily {
        y: spec.y,
        weights: spec.weights,
        link_kind: spec.link_kind,
        wiggle_knots: spec.wiggle_knots,
        wiggle_degree: spec.wiggle_degree,
        policy: crate::solver::resource::ResourcePolicy::default_library(),
    };
    let blocks = vec![
        // The wiggle block is a DYNAMIC monotone I-spline basis that the
        // family regenerates at full (raw) width every inner iteration
        // (`block_geometry_is_dynamic` + the `x.ncols() == spec.design.ncols()`
        // assertion in `block_geometry`), so it cannot tolerate a physical
        // column drop. The level/intercept direction that the I-spline shares
        // with the eta block must therefore be yielded by the *eta* block,
        // whose static term-collection design is safely column-reducible (and
        // lifted back via the canonical per-block transform `T`). Give the eta
        // block the lower gauge priority so the canonical-gauge RRQR routes the
        // shared-level alias drop onto eta and leaves the dynamic wiggle basis
        // full-width.
        spec.eta_block
            .intospec_with_gauge_priority("eta", LINK_WIGGLE_GAUGE_PRIORITY)?,
        spec.wiggle_block.intospec("wiggle")?,
    ];
    fit_custom_family(&family, &blocks, options).map_err(|e| e.to_string())
}

pub(crate) trait LocationScaleFamilyBuilder {
    type Family: CustomFamily + Clone + Send + Sync + 'static;

    fn meanspec(&self) -> &TermCollectionSpec;
    fn noisespec(&self) -> &TermCollectionSpec;

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String>;

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family;

    fn extract_primary_betas(
        &self,
        fit: &UnifiedFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String>;

    fn mean_penalty_count(&self, mean_design: &TermCollectionDesign) -> usize {
        mean_design.penalties.len()
    }

    fn noise_penalty_count(&self, noise_design: &TermCollectionDesign) -> usize {
        noise_design.penalties.len()
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        false
    }

    fn require_exact_spatial_joint(&self) -> bool {
        false
    }

    fn exact_spatial_seed_risk_profile(&self) -> crate::seeding::SeedRiskProfile {
        crate::seeding::SeedRiskProfile::GeneralizedLinear
    }

    fn extra_rho0(&self) -> Result<Array1<f64>, String> {
        Ok(Array1::zeros(0))
    }

    fn build_psiderivative_blocks(
        &self,
        arr: ndarray::ArrayView2<'_, f64>,
        term_spec: &TermCollectionSpec,
        term_spec2: &TermCollectionSpec,
        term_design: &TermCollectionDesign,
        term_design2: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String>;
}

pub(crate) fn fit_location_scale_terms<B: LocationScaleFamilyBuilder>(
    data: ndarray::ArrayView2<'_, f64>,
    builder: B,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    // Large-n location-scale fits keep the caller's explicit Hessian request.
    // The unified REML evaluator chooses a dense or matrix-free exact
    // representation from the realized (n, p, K) work model, so there is no
    // large-scale downgrade to BFGS here.

    let mut mean_beta_hint: Option<Array1<f64>> = None;
    let mut noise_beta_hint: Option<Array1<f64>> = None;
    let extra_rho0 = builder.extra_rho0()?;

    let mean_boot_design =
        build_term_collection_design(data, builder.meanspec()).map_err(|e| e.to_string())?;
    let noise_boot_design =
        build_term_collection_design(data, builder.noisespec()).map_err(|e| e.to_string())?;
    let mean_bootspec = freeze_term_collection_from_design(builder.meanspec(), &mean_boot_design)
        .map_err(|e| e.to_string())?;
    let noise_bootspec =
        freeze_term_collection_from_design(builder.noisespec(), &noise_boot_design)
            .map_err(|e| e.to_string())?;

    let require_exact_spatial_joint = builder.require_exact_spatial_joint();
    let analytic_joint_derivatives_check = if builder.exact_spatial_joint_supported() {
        builder
            .build_psiderivative_blocks(
                data,
                &mean_bootspec,
                &noise_bootspec,
                &mean_boot_design,
                &noise_boot_design,
            )
            .map(|_| ())
    } else {
        Err(
            "analytic spatial psi derivatives are unavailable for this location-scale family"
                .to_string(),
        )
    };
    let analytic_joint_derivatives_available = analytic_joint_derivatives_check.is_ok();
    if require_exact_spatial_joint {
        analytic_joint_derivatives_check.map_err(|err| {
            format!("exact two-block spatial path requires analytic psi derivatives: {err}")
        })?;
    }
    let mean_penalty_count = builder.mean_penalty_count(&mean_boot_design);
    let noise_penalty_count = builder.noise_penalty_count(&noise_boot_design);

    // Honor an explicit user-supplied `length_scale=X` on every spatial term
    // in both the mean and noise blocks: when every term is κ-locked (no
    // anisotropy, no per-axis ψ contrasts), the joint-spatial outer optimizer
    // has nothing to optimize. Routing through it anyway wraps the full
    // two-block coefficient solve inside an unnecessary outer loop where
    // each evaluation runs the inner Newton from scratch. This is the same
    // short-circuit the Bernoulli marginal-slope entry point performs at
    // bernoulli_marginal_slope.rs:16432-16442; mirroring it here makes the
    // GAMLSS path skip straight to the `(!enabled || log_kappa_dim == 0)`
    // fast path in `optimize_spatial_length_scale_exact_joint`.
    let mut effective_kappa_options = kappa_options.clone();
    if effective_kappa_options.enabled
        && crate::smooth::all_spatial_terms_kappa_fixed(&mean_bootspec)
        && crate::smooth::all_spatial_terms_kappa_fixed(&noise_bootspec)
    {
        log::info!(
            "[GAMLSS spatial] disabling κ/ψ optimization: every spatial term in \
             both blocks has an explicit length_scale and no anisotropy; \
             user-supplied kernel scale is fixed"
        );
        effective_kappa_options.enabled = false;
    }
    let kappa_options: &SpatialLengthScaleOptimizationOptions = &effective_kappa_options;

    // Macro to invoke the exact-joint spatial optimizer with shared closures.
    // The exact path evaluates the full profiled/Laplace objective over
    // theta = [rho, psi] with the real joint Hessian required by NewtonTR/ARC.
    macro_rules! run_exact_joint_spatial {
        () => {{
            let joint_setup = build_two_block_exact_joint_setup(
                data,
                builder.meanspec(),
                builder.noisespec(),
                mean_penalty_count,
                noise_penalty_count,
                extra_rho0.as_slice().unwrap_or(&[]),
                None,
                kappa_options,
            );
            let mean_terms = spatial_length_scale_term_indices(builder.meanspec());
            let noise_terms = spatial_length_scale_term_indices(builder.noisespec());
            let mean_beta_hint_cell = std::cell::RefCell::new(mean_beta_hint.clone());
            let noise_beta_hint_cell = std::cell::RefCell::new(noise_beta_hint.clone());
            let hyper_warm_start_cell =
                std::cell::RefCell::new(None::<CustomFamilyWarmStart>);
            // Two-block GAMLSS/location-scale joint likelihoods have a
            // β-dependent cross-block Hessian (the (μ,log σ) / (t,log σ)
            // off-diagonal blocks involve residual/response scalars that
            // shift when β moves). The Wood-Fasiolo structural property
            // `H^{-1/2} B_k H^{-1/2} ≽ 0` plus parameter-independent
            // nullspace — the mathematical basis for EFS convergence —
            // fails here, so EFS/HybridEFS must be excluded at plan time
            // rather than retried as a silent first attempt that stalls
            // for hundreds of seconds before the runner falls back.
            let gamlss_disable_fixed_point = true;
            let outer_policy = {
                // GAMLSS spatial path: psi_dim = log_kappa_dim + auxiliary_dim,
                // matching the (theta_dim - rho_dim) decomposition the
                // optimizer uses internally. Build realized ParameterBlockSpecs
                // at the seed rho so the family's own cost model — which
                // multiplies coefficient-gradient / coefficient-Hessian
                // per-row cost by the joint outer-coordinate dimension and
                // total p — produces honest `predicted_*_work` estimates.
                // Previously this fed `predicted_*_work: 0` to the planner,
                // which then ungated dense outer Hessian work that costs
                // hundreds of seconds per eval at large scale (see
                // `OuterDerivativePolicy::OUTER_HESSIAN_WORK_BUDGET`).
                let theta_seed = joint_setup.theta0();
                let rho_dim = joint_setup.rho_dim();
                let psi_dim = theta_seed.len() - rho_dim;
                let rho_seed = theta_seed.slice(s![..rho_dim]).to_owned();
                let policy_blocks_res = builder.build_blocks(
                    &rho_seed,
                    &mean_boot_design,
                    &noise_boot_design,
                    mean_beta_hint_cell.borrow().clone(),
                    noise_beta_hint_cell.borrow().clone(),
                );
                let mut policy = match policy_blocks_res {
                    Ok(policy_blocks) => {
                        let policy_family =
                            builder.build_family(&mean_boot_design, &noise_boot_design);
                        crate::families::custom_family::CustomFamily::outer_derivative_policy(
                            &policy_family,
                            &policy_blocks,
                            psi_dim,
                            options,
                        )
                    }
                    Err(err) => {
                        // Block construction at the seed should not fail for
                        // any in-tree family, but if it does, fall back to a
                        // policy that names the capability honestly and
                        // declines to predict cost. Setting work to
                        // `u128::MAX` routes the planner through gradient-only
                        // BFGS (the universal Hessian-work budget is
                        // saturating, so a sentinel is fine here).
                        log::warn!(
                            "[GAMLSS spatial] failed to realize policy blocks at seed rho ({err}); \
                             routing outer optimizer through gradient-only BFGS"
                        );
                        let capability = if analytic_joint_derivatives_available {
                            crate::families::custom_family::ExactOuterDerivativeOrder::Second
                        } else {
                            crate::families::custom_family::ExactOuterDerivativeOrder::First
                        };
                        crate::families::custom_family::OuterDerivativePolicy {
                            capability,
                            predicted_gradient_work: u128::MAX,
                            predicted_hessian_work: u128::MAX,
                            // No GAMLSS family today overrides its
                            // outer-only `_with_options` hooks to consume
                            // `outer_score_subsample`; staged-κ would
                            // build pilot masks the family then ignores.
                            subsample_capable: false,
                        }
                    }
                };
                if !analytic_joint_derivatives_available {
                    // Capability must not exceed what the analytic derivatives
                    // path can supply — the macro's hyper evaluator returns
                    // an error otherwise.
                    policy.capability =
                        crate::families::custom_family::ExactOuterDerivativeOrder::First;
                }
                policy
            };
            optimize_spatial_length_scale_exact_joint(
                data,
                &[builder.meanspec().clone(), builder.noisespec().clone()],
                &[mean_terms, noise_terms],
                kappa_options,
                &joint_setup,
                builder.exact_spatial_seed_risk_profile(),
                analytic_joint_derivatives_available,
                analytic_joint_derivatives_available,
                gamlss_disable_fixed_point,
                None,
                outer_policy,
                |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
                    assert_eq!(
                        specs.len(),
                        2,
                        "joint spatial closure expects exactly two block specs (mean, noise); got {}",
                        specs.len(),
                    );
                    assert_eq!(
                        designs.len(),
                        2,
                        "joint spatial closure expects exactly two block designs (mean, noise); got {}",
                        designs.len(),
                    );
                    let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
                    let fit = {
                        let blocks = builder.build_blocks(
                            &rho,
                            &designs[0],
                            &designs[1],
                            mean_beta_hint_cell.borrow().clone(),
                            noise_beta_hint_cell.borrow().clone(),
                        )?;
                        if mean_beta_hint_cell.borrow().is_none()
                            && let Some(beta) = blocks.first().and_then(|block| block.initial_beta.clone())
                        {
                            *mean_beta_hint_cell.borrow_mut() = Some(beta);
                        }
                        if noise_beta_hint_cell.borrow().is_none()
                            && let Some(beta) =
                                blocks.get(1).and_then(|block| block.initial_beta.clone())
                        {
                            *noise_beta_hint_cell.borrow_mut() = Some(beta);
                        }
                        let family = builder.build_family(&designs[0], &designs[1]);
                        // Branch on whether the κ optimizer drives rho.
                        //
                        // * `log_kappa_dim() > 0 && kappa_options.enabled` ⇒
                        //   the outer (ρ, ψ) optimizer is active and
                        //   passes each candidate ρ to this closure;
                        //   the inner fit must hold log-lambdas fixed
                        //   at the supplied ρ so the outer derivative
                        //   has a well-defined directional gradient.
                        //
                        // * Otherwise (κ disabled via the locked-κ
                        //   short-circuit, or no spatial terms at all)
                        //   the fast path in
                        //   `optimize_spatial_length_scale_exact_joint`
                        //   calls this closure exactly once at
                        //   `theta = theta0`; ρ must still be optimized
                        //   from data because the user never pinned it.
                        //   `fit_custom_family` performs the joint
                        //   ρ + coefficient REML fit at the user's
                        //   (now-fixed) kernel scale, which is the
                        //   intended behaviour when `length_scale=…` is
                        //   set on every spatial term.
                        if joint_setup.log_kappa_dim() > 0 && kappa_options.enabled {
                            let warm_start = hyper_warm_start_cell.borrow().clone();
                            fit_custom_family_fixed_log_lambdas(
                                &family,
                                &blocks,
                                options,
                                warm_start.as_ref(),
                                0,
                                None,
                                true,
                            )?
                        } else {
                            fit_custom_family(&family, &blocks, options)?
                        }
                    };
                    let (mean_beta, noise_beta) = builder.extract_primary_betas(&fit)?;
                    mean_beta_hint = Some(mean_beta);
                    noise_beta_hint = Some(noise_beta);
                    *mean_beta_hint_cell.borrow_mut() = mean_beta_hint.clone();
                    *noise_beta_hint_cell.borrow_mut() = noise_beta_hint.clone();
                    Ok(fit)
                },
                |theta,
                 specs: &[TermCollectionSpec],
                 designs: &[TermCollectionDesign],
                 eval_mode,
                 row_set: &crate::families::row_kernel::RowSet| {
                    use crate::reml_contracts::EvalMode;
                    if !analytic_joint_derivatives_available {
                        return Err(
                            "analytic spatial psi derivatives are unavailable for this exact two-block path"
                                .to_string(),
                        );
                    }
                    let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
                    let blocks = builder.build_blocks(
                        &rho,
                        &designs[0],
                        &designs[1],
                        mean_beta_hint_cell.borrow().clone(),
                        noise_beta_hint_cell.borrow().clone(),
                    )?;
                    if mean_beta_hint_cell.borrow().is_none()
                        && let Some(beta) = blocks.first().and_then(|block| block.initial_beta.clone())
                    {
                        *mean_beta_hint_cell.borrow_mut() = Some(beta);
                    }
                    if noise_beta_hint_cell.borrow().is_none()
                        && let Some(beta) = blocks.get(1).and_then(|block| block.initial_beta.clone())
                    {
                        *noise_beta_hint_cell.borrow_mut() = Some(beta);
                    }
                    let family = builder.build_family(&designs[0], &designs[1]);
                    let psiderivative_blocks = builder.build_psiderivative_blocks(
                        data,
                        &specs[0],
                        &specs[1],
                        &designs[0],
                        &designs[1],
                    )?;
                    let warm_start = hyper_warm_start_cell.borrow().clone();
                    // Forward the κ-staging row set to the family by installing it
                    // on the canonical `outer_score_subsample` option. Inner-PIRLS
                    // and final covariance still run on full data (the per-row
                    // weight is consulted only by outer-only paths inside the
                    // family). When the staging schedule is full-data the option
                    // stays `None` and the call is equivalent to the prior path.
                    let eval_options = match row_set {
                        crate::families::row_kernel::RowSet::All => {
                            std::borrow::Cow::Borrowed(options)
                        }
                        crate::families::row_kernel::RowSet::Subsample {
                            rows,
                            n_full,
                        } => {
                            let subsample = crate::solver::outer_subsample::
                                OuterScoreSubsample::from_weighted_rows(
                                    (**rows).clone(),
                                    *n_full,
                                    *n_full as u64,
                                );
                            let mut cloned = options.clone();
                            cloned.outer_score_subsample =
                                Some(std::sync::Arc::new(subsample));
                            std::borrow::Cow::Owned(cloned)
                        }
                    };
                    let eval = evaluate_custom_family_joint_hyper(
                        &family,
                        &blocks,
                        eval_options.as_ref(),
                        &rho,
                        &psiderivative_blocks,
                        warm_start.as_ref(),
                        eval_mode,
                    )?;
                    *hyper_warm_start_cell.borrow_mut() = Some(eval.warm_start.clone());
                    if !eval.inner_converged {
                        return Err(
                            "exact two-block spatial inner solve did not converge".to_string(),
                        );
                    }
                    if matches!(eval_mode, EvalMode::ValueGradientHessian)
                        && !eval.outer_hessian.is_analytic()
                    {
                        return Err(
                            "exact two-block spatial objective requires a full joint [rho, psi] hessian"
                                .to_string(),
                        );
                    }
                    Ok((eval.objective, eval.gradient, eval.outer_hessian))
                },
                |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
                    if !analytic_joint_derivatives_available {
                        return Err(
                            "analytic spatial psi derivatives are unavailable for this exact two-block path"
                                .to_string(),
                        );
                    }
                    let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
                    let blocks = builder.build_blocks(
                        &rho,
                        &designs[0],
                        &designs[1],
                        mean_beta_hint_cell.borrow().clone(),
                        noise_beta_hint_cell.borrow().clone(),
                    )?;
                    if mean_beta_hint_cell.borrow().is_none()
                        && let Some(beta) = blocks.first().and_then(|block| block.initial_beta.clone())
                    {
                        *mean_beta_hint_cell.borrow_mut() = Some(beta);
                    }
                    if noise_beta_hint_cell.borrow().is_none()
                        && let Some(beta) = blocks.get(1).and_then(|block| block.initial_beta.clone())
                    {
                        *noise_beta_hint_cell.borrow_mut() = Some(beta);
                    }
                    let family = builder.build_family(&designs[0], &designs[1]);
                    let psiderivative_blocks = builder.build_psiderivative_blocks(
                        data,
                        &specs[0],
                        &specs[1],
                        &designs[0],
                        &designs[1],
                    )?;
                    let warm_start = hyper_warm_start_cell.borrow().clone();
                    let eval = evaluate_custom_family_joint_hyper_efs(
                        &family,
                        &blocks,
                        options,
                        &rho,
                        &psiderivative_blocks,
                        warm_start.as_ref(),
                    )?;
                    *hyper_warm_start_cell.borrow_mut() = Some(eval.warm_start.clone());
                    if !eval.inner_converged {
                        return Err(
                            "exact two-block spatial EFS inner solve did not converge".to_string(),
                        );
                    }
                    Ok(eval.efs_eval)
                },
                |_beta: &Array1<f64>| Ok(crate::solver::rho_optimizer::SeedOutcome::NoSlot),
            )
        }};
    }

    let mut solved = run_exact_joint_spatial!()
        .map_err(|err| format!("exact two-block spatial optimization failed: {err}"))?;

    let expected_noise_penalty_count = builder.noise_penalty_count(&solved.designs[1]);
    let actual_noise_penalty_count = solved.designs[1].penalties.len();
    if expected_noise_penalty_count > actual_noise_penalty_count {
        if expected_noise_penalty_count != actual_noise_penalty_count + 1 {
            return Err(GamlssError::UnsupportedConfiguration {
                reason: format!(
                    "location-scale result noise design expected {} penalties after augmentation, got {} before augmentation",
                    expected_noise_penalty_count, actual_noise_penalty_count
                ),
            }
            .into());
        }
        append_binomial_log_sigma_shrinkage_penalty_design(&mut solved.designs[1]);
    }

    BlockwiseTermFitResult::try_from_parts(BlockwiseTermFitResultParts {
        fit: solved.fit,
        meanspec_resolved: solved.resolved_specs.remove(0),
        noisespec_resolved: solved.resolved_specs.remove(0),
        mean_design: solved.designs.remove(0),
        noise_design: solved.designs.remove(0),
    })
}

pub(crate) struct GaussianLocationScaleTermBuilder {
    pub(crate) y: Array1<f64>,
    pub(crate) weights: Array1<f64>,
    pub(crate) meanspec: TermCollectionSpec,
    pub(crate) noisespec: TermCollectionSpec,
    pub(crate) mean_offset: Array1<f64>,
    pub(crate) noise_offset: Array1<f64>,
}

impl LocationScaleFamilyBuilder for GaussianLocationScaleTermBuilder {
    type Family = GaussianLocationScaleFamily;

    fn meanspec(&self) -> &TermCollectionSpec {
        &self.meanspec
    }

    fn noisespec(&self) -> &TermCollectionSpec {
        &self.noisespec
    }

    fn noise_penalty_count(&self, noise_design: &TermCollectionDesign) -> usize {
        // Mirror the Binomial location-scale path: the log-sigma (scale)
        // block carries an extra full-space shrinkage penalty so its
        // polynomial nullspace (constant log-sigma, plus the linear term for
        // tp/Duchon bases) is not left unpenalized. Without it, outer REML
        // optimizes lambda_sigma on a flat/ill-conditioned surface, which
        // over-smooths the scale envelope (bad Pearson/CRPS/PIT/NLL) and can
        // diverge the coupled inner Newton (log_sigma residual blows up,
        // beta -> infinity). The strength of this ridge is REML-selected.
        noise_design.penalties.len() + 1
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }

    fn exact_spatial_seed_risk_profile(&self) -> crate::seeding::SeedRiskProfile {
        crate::seeding::SeedRiskProfile::Gaussian
    }

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String> {
        let layout = GamlssLambdaLayout::two_block(
            mean_design.penalties.len(),
            self.noise_penalty_count(noise_design),
        );
        layout.validate_theta_len(theta.len(), "gaussian location-scale")?;
        let (meanspec, noisespec) = build_gaussian_mean_and_scale_blocks(
            &self.y,
            &self.weights,
            mean_design,
            noise_design,
            &self.mean_offset,
            &self.noise_offset,
            layout.mean_from(theta),
            layout.noise_from(theta),
            mean_beta_hint,
            noise_beta_hint,
            "GaussianLocationScale::build_blocks",
        )?;
        Ok(vec![meanspec, noisespec])
    }

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family {
        let preparednoise_design =
            prepared_gaussian_log_sigma_design(&mean_design.design, &noise_design.design)
                .expect("prepared Gaussian log-sigma design should match block construction");
        GaussianLocationScaleFamily {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: Some(mean_design.design.clone()),
            log_sigma_design: Some(preparednoise_design),
            policy: crate::solver::resource::ResourcePolicy::default_library(),
            cached_row_scalars: std::sync::RwLock::new(None),
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &UnifiedFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let mean_beta = fit
            .block_states
            .get(GaussianLocationScaleFamily::BLOCK_MU)
            .ok_or_else(|| "missing Gaussian mu block state".to_string())?
            .beta
            .clone();
        let noise_beta = fit
            .block_states
            .get(GaussianLocationScaleFamily::BLOCK_LOG_SIGMA)
            .ok_or_else(|| "missing Gaussian log_sigma block state".to_string())?
            .beta
            .clone();
        Ok((mean_beta, noise_beta))
    }

    fn build_psiderivative_blocks(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        meanspec_resolved: &TermCollectionSpec,
        noisespec_resolved: &TermCollectionSpec,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        let mean_derivs =
            build_block_spatial_psi_derivatives(data, meanspec_resolved, mean_design)?
                .ok_or_else(|| "missing Gaussian mean spatial psi derivatives".to_string())?;
        let noise_derivs =
            build_block_spatial_psi_derivatives(data, noisespec_resolved, noise_design)?
                .ok_or_else(|| "missing Gaussian log-sigma spatial psi derivatives".to_string())?;
        Ok(vec![mean_derivs, noise_derivs])
    }
}

pub(crate) struct GaussianLocationScaleWiggleTermBuilder {
    pub(crate) y: Array1<f64>,
    pub(crate) weights: Array1<f64>,
    pub(crate) meanspec: TermCollectionSpec,
    pub(crate) noisespec: TermCollectionSpec,
    pub(crate) mean_offset: Array1<f64>,
    pub(crate) noise_offset: Array1<f64>,
    pub(crate) wiggle_knots: Array1<f64>,
    pub(crate) wiggle_degree: usize,
    pub(crate) wiggle_block: ParameterBlockInput,
}

impl LocationScaleFamilyBuilder for GaussianLocationScaleWiggleTermBuilder {
    type Family = GaussianLocationScaleWiggleFamily;

    fn meanspec(&self) -> &TermCollectionSpec {
        &self.meanspec
    }

    fn noisespec(&self) -> &TermCollectionSpec {
        &self.noisespec
    }

    fn noise_penalty_count(&self, noise_design: &TermCollectionDesign) -> usize {
        // Same full-space log-sigma shrinkage penalty as the non-wiggle
        // Gaussian builder; see GaussianLocationScaleTermBuilder.
        noise_design.penalties.len() + 1
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }

    fn exact_spatial_seed_risk_profile(&self) -> crate::seeding::SeedRiskProfile {
        crate::seeding::SeedRiskProfile::Gaussian
    }

    fn require_exact_spatial_joint(&self) -> bool {
        true
    }

    fn extra_rho0(&self) -> Result<Array1<f64>, String> {
        initial_log_lambdas_orzeros(&self.wiggle_block)
    }

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String> {
        let layout = GamlssLambdaLayout::withwiggle(
            mean_design.penalties.len(),
            self.noise_penalty_count(noise_design),
            self.wiggle_block.penalties.len(),
        );
        layout.validate_theta_len(theta.len(), "gaussian location-scale wiggle")?;
        let (mut meanspec, mut noisespec) = build_gaussian_mean_and_scale_blocks(
            &self.y,
            &self.weights,
            mean_design,
            noise_design,
            &self.mean_offset,
            &self.noise_offset,
            layout.mean_from(theta),
            layout.noise_from(theta),
            mean_beta_hint,
            noise_beta_hint,
            "GaussianLocationScaleWiggle::build_blocks",
        )?;
        // Keep the dynamic full-width wiggle basis safe from a canonical-gauge
        // column drop: route the shared level/intercept alias onto the
        // column-reducible mean and log-sigma blocks by giving them a lower
        // gauge priority than the wiggle block's fixed 100 (see the binomial
        // wiggle path and `build_location_scale_wiggle_block`).
        meanspec.gauge_priority = LINK_WIGGLE_GAUGE_PRIORITY;
        noisespec.gauge_priority = LINK_WIGGLE_GAUGE_PRIORITY;
        let n_rows = meanspec.design.nrows();
        let wigglespec = build_location_scale_wiggle_block(
            "wiggle",
            self.wiggle_block.design.clone(),
            self.wiggle_block.offset.clone(),
            wiggle_block_penalty_matrices(&self.wiggle_block),
            self.wiggle_block.nullspace_dims.clone(),
            layout.wiggle_from(theta),
            self.wiggle_block.initial_beta.clone(),
            n_rows,
        )?;
        Ok(vec![meanspec, noisespec, wigglespec])
    }

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family {
        let preparednoise_design =
            prepared_gaussian_log_sigma_design(&mean_design.design, &noise_design.design).expect(
                "prepared Gaussian log-sigma design should match wiggle block construction",
            );
        GaussianLocationScaleWiggleFamily {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: Some(mean_design.design.clone()),
            log_sigma_design: Some(preparednoise_design),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
            policy: crate::solver::resource::ResourcePolicy::default_library(),
            cached_row_scalars: std::sync::RwLock::new(None),
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &UnifiedFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let mean_beta = fit
            .block_states
            .get(GaussianLocationScaleWiggleFamily::BLOCK_MU)
            .ok_or_else(|| "missing Gaussian wiggle mu block state".to_string())?
            .beta
            .clone();
        let noise_beta = fit
            .block_states
            .get(GaussianLocationScaleWiggleFamily::BLOCK_LOG_SIGMA)
            .ok_or_else(|| "missing Gaussian wiggle log_sigma block state".to_string())?
            .beta
            .clone();
        Ok((mean_beta, noise_beta))
    }

    fn build_psiderivative_blocks(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        meanspec_resolved: &TermCollectionSpec,
        noisespec_resolved: &TermCollectionSpec,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        let mean_derivs =
            build_block_spatial_psi_derivatives(data, meanspec_resolved, mean_design)?.ok_or_else(
                || "missing Gaussian wiggle mean spatial psi derivatives".to_string(),
            )?;
        let noise_derivs =
            build_block_spatial_psi_derivatives(data, noisespec_resolved, noise_design)?
                .ok_or_else(|| {
                    "missing Gaussian wiggle log-sigma spatial psi derivatives".to_string()
                })?;
        Ok(vec![mean_derivs, noise_derivs, Vec::new()])
    }
}

pub(crate) struct BinomialLocationScaleTermBuilder {
    pub(crate) y: Array1<f64>,
    pub(crate) weights: Array1<f64>,
    pub(crate) link_kind: InverseLink,
    pub(crate) meanspec: TermCollectionSpec,
    pub(crate) noisespec: TermCollectionSpec,
    pub(crate) mean_offset: Array1<f64>,
    pub(crate) noise_offset: Array1<f64>,
}

impl LocationScaleFamilyBuilder for BinomialLocationScaleTermBuilder {
    type Family = BinomialLocationScaleFamily;

    fn meanspec(&self) -> &TermCollectionSpec {
        &self.meanspec
    }

    fn noisespec(&self) -> &TermCollectionSpec {
        &self.noisespec
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }

    fn require_exact_spatial_joint(&self) -> bool {
        true
    }

    fn noise_penalty_count(&self, noise_design: &TermCollectionDesign) -> usize {
        noise_design.penalties.len() + 1
    }

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String> {
        let layout = GamlssLambdaLayout::two_block(
            mean_design.penalties.len(),
            self.noise_penalty_count(noise_design),
        );
        layout.validate_theta_len(theta.len(), "binomial location-scale")?;
        let (thresholdspec, log_sigmaspec) = build_binomial_threshold_and_scale_blocks(
            &self.y,
            &self.weights,
            &self.link_kind,
            mean_design,
            noise_design,
            &self.mean_offset,
            &self.noise_offset,
            layout.mean_from(theta),
            layout.noise_from(theta),
            mean_beta_hint,
            noise_beta_hint,
            "BinomialLocationScale::build_blocks",
        )?;
        Ok(vec![thresholdspec, log_sigmaspec])
    }

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family {
        let identifiednoise_design =
            identified_binomial_log_sigma_design(mean_design, noise_design, &self.weights)
                .expect("identified binomial log-sigma design");
        BinomialLocationScaleFamily {
            y: self.y.clone(),
            weights: self.weights.clone(),
            link_kind: self.link_kind.clone(),
            threshold_design: Some(mean_design.design.clone()),
            log_sigma_design: Some(identifiednoise_design),
            policy: crate::solver::resource::ResourcePolicy::default_library(),
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &UnifiedFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let mean_beta = fit
            .block_states
            .get(BinomialLocationScaleFamily::BLOCK_T)
            .ok_or_else(|| "missing Binomial threshold block state".to_string())?
            .beta
            .clone();
        let noise_beta = fit
            .block_states
            .get(BinomialLocationScaleFamily::BLOCK_LOG_SIGMA)
            .ok_or_else(|| "missing Binomial log_sigma block state".to_string())?
            .beta
            .clone();
        Ok((mean_beta, noise_beta))
    }

    fn build_psiderivative_blocks(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        meanspec_resolved: &TermCollectionSpec,
        noisespec_resolved: &TermCollectionSpec,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        let mean_derivs =
            build_block_spatial_psi_derivatives(data, meanspec_resolved, mean_design)?
                .ok_or_else(|| "missing threshold spatial psi derivatives".to_string())?;
        let noise_derivs =
            build_block_spatial_psi_derivatives(data, noisespec_resolved, noise_design)?
                .ok_or_else(|| "missing log_sigma spatial psi derivatives".to_string())?;
        Ok(vec![mean_derivs, noise_derivs])
    }
}

pub(crate) struct BinomialLocationScaleWiggleTermBuilder {
    pub(crate) y: Array1<f64>,
    pub(crate) weights: Array1<f64>,
    pub(crate) link_kind: InverseLink,
    pub(crate) meanspec: TermCollectionSpec,
    pub(crate) noisespec: TermCollectionSpec,
    pub(crate) mean_offset: Array1<f64>,
    pub(crate) noise_offset: Array1<f64>,
    pub(crate) wiggle_knots: Array1<f64>,
    pub(crate) wiggle_degree: usize,
    pub(crate) wiggle_block: ParameterBlockInput,
}

impl LocationScaleFamilyBuilder for BinomialLocationScaleWiggleTermBuilder {
    type Family = BinomialLocationScaleWiggleFamily;

    fn meanspec(&self) -> &TermCollectionSpec {
        &self.meanspec
    }

    fn noisespec(&self) -> &TermCollectionSpec {
        &self.noisespec
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }

    fn require_exact_spatial_joint(&self) -> bool {
        true
    }

    fn extra_rho0(&self) -> Result<Array1<f64>, String> {
        initial_log_lambdas_orzeros(&self.wiggle_block)
    }

    fn noise_penalty_count(&self, noise_design: &TermCollectionDesign) -> usize {
        noise_design.penalties.len() + 1
    }

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String> {
        let layout = GamlssLambdaLayout::withwiggle(
            mean_design.penalties.len(),
            self.noise_penalty_count(noise_design),
            self.wiggle_block.penalties.len(),
        );
        layout.validate_theta_len(theta.len(), "wiggle location-scale")?;
        let (mut thresholdspec, mut log_sigmaspec) = build_binomial_threshold_and_scale_blocks(
            &self.y,
            &self.weights,
            &self.link_kind,
            mean_design,
            noise_design,
            &self.mean_offset,
            &self.noise_offset,
            layout.mean_from(theta),
            layout.noise_from(theta),
            mean_beta_hint,
            noise_beta_hint,
            "BinomialLocationScaleWiggle::build_blocks",
        )?;
        // The dynamic monotone wiggle basis is regenerated at full raw width
        // every inner iteration and asserts `x.ncols() == spec.design.ncols()`
        // in `block_geometry`, so it cannot tolerate a canonical-gauge column
        // drop. The level/intercept direction the I-spline shares with the
        // threshold block must therefore be routed onto the threshold (and the
        // log-sigma) block, whose static designs are column-reducible and
        // lifted back via the canonical per-block transform `T`. Give both
        // non-wiggle blocks a lower gauge priority than the wiggle block (which
        // `build_location_scale_wiggle_block` fixes at 100) so the shared-level
        // alias drop lands on them and leaves the dynamic wiggle basis full
        // width — mirroring the binomial mean-wiggle path.
        thresholdspec.gauge_priority = LINK_WIGGLE_GAUGE_PRIORITY;
        log_sigmaspec.gauge_priority = LINK_WIGGLE_GAUGE_PRIORITY;
        let n_rows = thresholdspec.design.nrows();
        let wigglespec = build_location_scale_wiggle_block(
            "wiggle",
            self.wiggle_block.design.clone(),
            self.wiggle_block.offset.clone(),
            wiggle_block_penalty_matrices(&self.wiggle_block),
            vec![],
            layout.wiggle_from(theta),
            self.wiggle_block.initial_beta.clone(),
            n_rows,
        )?;
        Ok(vec![thresholdspec, log_sigmaspec, wigglespec])
    }

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family {
        let identifiednoise_design =
            identified_binomial_log_sigma_design(mean_design, noise_design, &self.weights)
                .expect("identified binomial log-sigma design should match block construction");
        BinomialLocationScaleWiggleFamily {
            y: self.y.clone(),
            weights: self.weights.clone(),
            link_kind: self.link_kind.clone(),
            threshold_design: Some(mean_design.design.clone()),
            log_sigma_design: Some(identifiednoise_design),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
            policy: crate::solver::resource::ResourcePolicy::default_library(),
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &UnifiedFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let mean_beta = fit
            .block_states
            .get(BinomialLocationScaleWiggleFamily::BLOCK_T)
            .ok_or_else(|| "missing Binomial wiggle threshold block state".to_string())?
            .beta
            .clone();
        let noise_beta = fit
            .block_states
            .get(BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA)
            .ok_or_else(|| "missing Binomial wiggle log_sigma block state".to_string())?
            .beta
            .clone();
        Ok((mean_beta, noise_beta))
    }

    fn build_psiderivative_blocks(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        meanspec_resolved: &TermCollectionSpec,
        noisespec_resolved: &TermCollectionSpec,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        let mean_derivs =
            build_block_spatial_psi_derivatives(data, meanspec_resolved, mean_design)?
                .ok_or_else(|| "missing threshold spatial psi derivatives".to_string())?;
        let noise_derivs =
            build_block_spatial_psi_derivatives(data, noisespec_resolved, noise_design)?
                .ok_or_else(|| "missing log_sigma spatial psi derivatives".to_string())?;
        // The wiggle block has no direct spatial design matrix of its own in the
        // term builder. Spatial psi moves the wiggle family only through the
        // realized threshold/log-sigma designs, which in turn perturb q0 and the
        // realized wiggle basis B(q0). The exact joint wiggle psi hooks consume
        // those threshold/log-sigma derivative payloads and reconstruct the full
        // flattened likelihood-side [rho, psi] calculus internally, so the
        // wiggle block intentionally contributes no direct CustomFamilyBlockPsiDerivative
        // entries here.
        Ok(vec![mean_derivs, noise_derivs, Vec::new()])
    }
}

pub(crate) fn fit_gaussian_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: GaussianLocationScaleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    validate_gaussian_location_scale_termspec(data, &spec, "fit_gaussian_location_scale_terms")?;
    fit_location_scale_terms(
        data,
        GaussianLocationScaleTermBuilder {
            y: spec.y,
            weights: spec.weights,
            meanspec: spec.meanspec,
            noisespec: spec.log_sigmaspec,
            mean_offset: spec.mean_offset,
            noise_offset: spec.log_sigma_offset,
        },
        options,
        kappa_options,
    )
}

pub(crate) fn fit_gaussian_location_scalewiggle_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: GaussianLocationScaleWiggleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    validate_gaussian_location_scalewiggle_termspec(
        data,
        &spec,
        "fit_gaussian_location_scalewiggle_terms",
    )?;
    fit_location_scale_terms(
        data,
        GaussianLocationScaleWiggleTermBuilder {
            y: spec.y,
            weights: spec.weights,
            meanspec: spec.meanspec,
            noisespec: spec.log_sigmaspec,
            mean_offset: spec.mean_offset,
            noise_offset: spec.log_sigma_offset,
            wiggle_knots: spec.wiggle_knots,
            wiggle_degree: spec.wiggle_degree,
            wiggle_block: spec.wiggle_block,
        },
        options,
        kappa_options,
    )
}

pub(crate) fn select_gaussian_location_scale_link_wiggle_basis_from_pilot(
    pilot: &BlockwiseTermFitResult,
    wiggle_cfg: &WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let q_seed = pilot
        .fit
        .block_states
        .first()
        .ok_or_else(|| "pilot Gaussian wiggle fit is missing mean block".to_string())?
        .eta
        .view();
    select_wiggle_basis_from_seed(q_seed, wiggle_cfg, wiggle_penalty_orders)
}

pub(crate) fn fit_gaussian_location_scale_terms_with_selected_wiggle(
    data: ndarray::ArrayView2<'_, f64>,
    spec: GaussianLocationScaleTermSpec,
    selected_wiggle_basis: SelectedWiggleBasis,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermWiggleFitResult, String> {
    let SelectedWiggleBasis {
        knots: wiggle_knots,
        degree: wiggle_degree,
        block: wiggle_block,
        ..
    } = selected_wiggle_basis;
    let solved = fit_gaussian_location_scalewiggle_terms(
        data,
        GaussianLocationScaleWiggleTermSpec {
            y: spec.y,
            weights: spec.weights,
            meanspec: spec.meanspec,
            log_sigmaspec: spec.log_sigmaspec,
            mean_offset: spec.mean_offset,
            log_sigma_offset: spec.log_sigma_offset,
            wiggle_knots: wiggle_knots.clone(),
            wiggle_degree,
            wiggle_block,
        },
        options,
        kappa_options,
    )?;

    BlockwiseTermWiggleFitResult::try_from_parts(BlockwiseTermWiggleFitResultParts {
        fit: solved,
        wiggle_knots,
        wiggle_degree,
    })
}

pub(crate) fn fit_binomial_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    validate_binomial_location_scale_termspec(data, &spec, "fit_binomial_location_scale_terms")?;
    fit_location_scale_terms(
        data,
        BinomialLocationScaleTermBuilder {
            y: spec.y,
            weights: spec.weights,
            link_kind: spec.link_kind,
            meanspec: spec.thresholdspec,
            noisespec: spec.log_sigmaspec,
            mean_offset: spec.threshold_offset,
            noise_offset: spec.log_sigma_offset,
        },
        options,
        kappa_options,
    )
}

pub(crate) fn fit_binomial_location_scalewiggle_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleWiggleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    validate_binomial_location_scalewiggle_termspec(
        data,
        &spec,
        "fit_binomial_location_scalewiggle_terms",
    )?;
    fit_location_scale_terms(
        data,
        BinomialLocationScaleWiggleTermBuilder {
            y: spec.y,
            weights: spec.weights,
            link_kind: spec.link_kind,
            meanspec: spec.thresholdspec,
            noisespec: spec.log_sigmaspec,
            mean_offset: spec.threshold_offset,
            noise_offset: spec.log_sigma_offset,
            wiggle_knots: spec.wiggle_knots,
            wiggle_degree: spec.wiggle_degree,
            wiggle_block: spec.wiggle_block,
        },
        options,
        kappa_options,
    )
}

pub(crate) fn select_binomial_location_scale_link_wiggle_basis_from_pilot(
    pilot: &BlockwiseTermFitResult,
    wiggle_cfg: &WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let eta_t = pilot
        .fit
        .block_states
        .first()
        .ok_or_else(|| "pilot fit is missing threshold block".to_string())?
        .eta
        .view();
    let eta_ls = pilot
        .fit
        .block_states
        .get(1)
        .ok_or_else(|| "pilot fit is missing log_sigma block".to_string())?
        .eta
        .view();
    let sigma = eta_ls.mapv(safe_exp);
    let q_seed = Array1::from_iter(eta_t.iter().zip(sigma.iter()).map(|(&t, &s)| -t / s));
    select_wiggle_basis_from_seed(q_seed.view(), wiggle_cfg, wiggle_penalty_orders)
}

pub(crate) fn fit_binomial_location_scale_terms_with_selected_wiggle(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleTermSpec,
    selected_wiggle_basis: SelectedWiggleBasis,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermWiggleFitResult, String> {
    let SelectedWiggleBasis {
        knots: wiggle_knots,
        degree: wiggle_degree,
        block: wiggle_block,
        ..
    } = selected_wiggle_basis;
    let solved = fit_binomial_location_scalewiggle_terms(
        data,
        BinomialLocationScaleWiggleTermSpec {
            y: spec.y,
            weights: spec.weights,
            link_kind: spec.link_kind,
            thresholdspec: spec.thresholdspec,
            log_sigmaspec: spec.log_sigmaspec,
            threshold_offset: spec.threshold_offset,
            log_sigma_offset: spec.log_sigma_offset,
            wiggle_knots: wiggle_knots.clone(),
            wiggle_degree,
            wiggle_block,
        },
        options,
        kappa_options,
    )?;

    BlockwiseTermWiggleFitResult::try_from_parts(BlockwiseTermWiggleFitResultParts {
        fit: solved,
        wiggle_knots,
        wiggle_degree,
    })
}

pub(crate) fn select_binomial_mean_link_wiggle_basis_from_pilot(
    pilot_design: &TermCollectionDesign,
    pilot_fit: &UnifiedFitResult,
    wiggle_cfg: &WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let q_seed = pilot_design.design.dot(&pilot_fit.beta);
    select_wiggle_basis_from_seed(q_seed.view(), wiggle_cfg, wiggle_penalty_orders)
}

pub(crate) fn fit_binomial_mean_wiggle_terms_with_selected_basis(
    data: ndarray::ArrayView2<'_, f64>,
    pilot_spec: &TermCollectionSpec,
    pilot_design: &TermCollectionDesign,
    pilot_fit: &UnifiedFitResult,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    link_kind: InverseLink,
    selected_wiggle_basis: SelectedWiggleBasis,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BinomialMeanWiggleTermFitResult, String> {
    const RHO_BOUND: f64 = 12.0;

    validate_term_weights(
        data,
        y.len(),
        weights,
        "fit_binomial_mean_wiggle_terms_with_selected_basis",
    )?;
    validate_binomial_response(y, "fit_binomial_mean_wiggle_terms_with_selected_basis")?;

    // Large-n binomial mean-wiggle fits keep the caller's explicit Hessian
    // request. The unified evaluator chooses the scalable exact representation
    // (dense for small work, operator HVP for large work) instead of routing to
    // gradient-only BFGS by observation count.

    let SelectedWiggleBasis {
        knots: wiggle_knots,
        degree: wiggle_degree,
        block: wiggle_block,
        ..
    } = selected_wiggle_basis;

    let spatial_terms = spatial_length_scale_term_indices(pilot_spec);
    if spatial_terms.is_empty() {
        let fit = fit_binomial_mean_wiggle(
            BinomialMeanWiggleSpec {
                y: y.clone(),
                weights: weights.clone(),
                link_kind,
                wiggle_knots: wiggle_knots.clone(),
                wiggle_degree,
                eta_block: ParameterBlockInput {
                    design: pilot_design.design.clone(),
                    offset: Array1::zeros(y.len()),
                    penalties: pilot_design
                        .penalties
                        .iter()
                        .map(crate::model_types::PenaltySpec::from_blockwise_ref)
                        .collect(),
                    nullspace_dims: vec![],
                    initial_log_lambdas: Some(
                        pilot_fit
                            .lambdas
                            .mapv(|v| v.max(WARMSTART_LOG_LAMBDA_FLOOR).ln()),
                    ),
                    initial_beta: Some(pilot_fit.beta.clone()),
                },
                wiggle_block,
            },
            options,
        )?;
        return Ok(BinomialMeanWiggleTermFitResult {
            fit,
            resolvedspec: pilot_spec.clone(),
            design: pilot_design.clone(),
            wiggle_knots,
            wiggle_degree,
        });
    }

    let dims_per_term = spatial_dims_per_term(pilot_spec, &spatial_terms);
    let log_kappa0 =
        SpatialLogKappaCoords::from_length_scales_aniso(pilot_spec, &spatial_terms, kappa_options)
            .reseed_from_data(data, pilot_spec, &spatial_terms, kappa_options);
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        pilot_spec,
        &spatial_terms,
        &dims_per_term,
        kappa_options,
    );
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        pilot_spec,
        &spatial_terms,
        &dims_per_term,
        kappa_options,
    );
    // Project seed onto bounds; spec.length_scale is a hint, not a constraint.
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);

    let eta_penalty_count = pilot_design.penalties.len();
    let wiggle_penalty_count = initial_log_lambdas_orzeros(&wiggle_block)?.len();
    let rho_dim = eta_penalty_count + wiggle_penalty_count;
    let baseline_resolvedspec = log_kappa0
        .apply_tospec(pilot_spec, &spatial_terms)
        .map_err(|e| e.to_string())?;
    let baseline_design =
        build_term_collection_design(data, &baseline_resolvedspec).map_err(|e| e.to_string())?;
    let baseline_fit = fit_binomial_mean_wiggle(
        BinomialMeanWiggleSpec {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: link_kind.clone(),
            wiggle_knots: wiggle_knots.clone(),
            wiggle_degree,
            eta_block: ParameterBlockInput {
                design: baseline_design.design.clone(),
                offset: Array1::zeros(y.len()),
                penalties: baseline_design
                    .penalties
                    .iter()
                    .map(crate::model_types::PenaltySpec::from_blockwise_ref)
                    .collect(),
                nullspace_dims: vec![],
                initial_log_lambdas: Some(
                    pilot_fit
                        .lambdas
                        .mapv(|v| v.max(WARMSTART_LOG_LAMBDA_FLOOR).ln()),
                ),
                initial_beta: Some(pilot_fit.beta.clone()),
            },
            wiggle_block: wiggle_block.clone(),
        },
        options,
    )?;
    let baseline_log_lambdas = baseline_fit
        .lambdas
        .mapv(|v| v.max(WARMSTART_LOG_LAMBDA_FLOOR).ln());
    if baseline_log_lambdas.len() != rho_dim {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "baseline binomial mean-wiggle fit returned {} log-lambdas, expected {rho_dim}",
                baseline_log_lambdas.len()
            ),
        }
        .into());
    }
    let baseline_eta_beta = baseline_fit
        .block_states
        .get(BinomialMeanWiggleFamily::BLOCK_ETA)
        .ok_or_else(|| "baseline binomial mean-wiggle fit missing eta block".to_string())?
        .beta
        .clone();
    let baseline_wiggle_beta = Some(
        baseline_fit
            .block_states
            .get(BinomialMeanWiggleFamily::BLOCK_WIGGLE)
            .ok_or_else(|| "baseline binomial mean-wiggle fit missing wiggle block".to_string())?
            .beta
            .clone(),
    );
    let theta_dim = rho_dim + log_kappa0.len();
    let mut theta0 = Array1::<f64>::zeros(theta_dim);
    theta0
        .slice_mut(s![0..rho_dim])
        .assign(&baseline_log_lambdas);
    theta0
        .slice_mut(s![rho_dim..theta_dim])
        .assign(log_kappa0.as_array());

    let mut lower = Array1::<f64>::from_elem(theta_dim, -RHO_BOUND);
    let mut upper = Array1::<f64>::from_elem(theta_dim, RHO_BOUND);
    lower
        .slice_mut(s![rho_dim..theta_dim])
        .assign(log_kappa_lower.as_array());
    upper
        .slice_mut(s![rho_dim..theta_dim])
        .assign(log_kappa_upper.as_array());

    let pilot_spec_cloned = pilot_spec.clone();
    let pilot_beta = baseline_eta_beta;
    let wiggle_design = wiggle_block.design.clone();
    let wiggle_offset = wiggle_block.offset.clone();
    let wiggle_penalties = wiggle_block.penalties.clone();
    let wiggle_initial_beta = baseline_wiggle_beta;
    let wiggle_knots_cloned = wiggle_knots.clone();
    let y_cloned = y.clone();
    let weights_cloned = weights.clone();
    let link_kind_cloned = link_kind.clone();
    let outer_family = BinomialMeanWiggleFamily {
        y: y_cloned.clone(),
        weights: weights_cloned.clone(),
        link_kind: link_kind_cloned.clone(),
        wiggle_knots: wiggle_knots_cloned.clone(),
        wiggle_degree,
        policy: crate::solver::resource::ResourcePolicy::default_library(),
    };
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let mut outer_options = options.clone();
    outer_options.screening_max_inner_iterations = Some(Arc::clone(&screening_cap));
    struct MeanWiggleOuterState {
        pub(crate) warm_cache: Option<crate::custom_family::CustomFamilyWarmStart>,
        pub(crate) last_eval: Option<(
            Array1<f64>,
            f64,
            Array1<f64>,
            crate::solver::rho_optimizer::HessianResult,
            crate::custom_family::CustomFamilyWarmStart,
        )>,
    }

    let build_realized_blocks = |theta: &Array1<f64>| -> Result<
        (
            TermCollectionSpec,
            TermCollectionDesign,
            Vec<ParameterBlockSpec>,
            Vec<CustomFamilyBlockPsiDerivative>,
        ),
        String,
    > {
        let log_kappa =
            SpatialLogKappaCoords::from_theta_tail_with_dims(theta, rho_dim, dims_per_term.clone());
        let resolvedspec = log_kappa
            .apply_tospec(&pilot_spec_cloned, &spatial_terms)
            .map_err(|e| e.to_string())?;
        let design =
            build_term_collection_design(data, &resolvedspec).map_err(|e| e.to_string())?;
        let eta_derivs = build_block_spatial_psi_derivatives(data, &resolvedspec, &design)?
            .ok_or_else(|| {
                "missing eta spatial psi derivatives for binomial mean wiggle".to_string()
            })?;
        let blocks = vec![
            ParameterBlockSpec {
                name: "eta".to_string(),
                design: design.design.clone(),
                offset: Array1::zeros(y_cloned.len()),
                penalties: design.penalties_as_penalty_matrix(),
                nullspace_dims: vec![],
                initial_log_lambdas: theta.slice(s![0..eta_penalty_count]).to_owned(),
                initial_beta: Some(pilot_beta.clone()),
                // Lower gauge priority on the static eta design: it yields the
                // shared level/intercept direction to the dynamic full-width
                // wiggle I-spline block (see fit_binomial_mean_wiggle).
                gauge_priority: LINK_WIGGLE_GAUGE_PRIORITY,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            },
            ParameterBlockSpec {
                name: "wiggle".to_string(),
                design: wiggle_design.clone(),
                offset: wiggle_offset.clone(),
                penalties: {
                    let p_wiggle = wiggle_design.ncols();
                    wiggle_penalties
                        .iter()
                        .map(|spec| match spec {
                            crate::model_types::PenaltySpec::Block {
                                local, col_range, ..
                            } => PenaltyMatrix::Blockwise {
                                local: local.clone(),
                                col_range: col_range.clone(),
                                total_dim: p_wiggle,
                            },
                            crate::model_types::PenaltySpec::Dense(m)
                            | crate::model_types::PenaltySpec::DenseWithMean {
                                matrix: m, ..
                            } => PenaltyMatrix::Dense(m.clone()),
                        })
                        .collect()
                },
                nullspace_dims: vec![],
                initial_log_lambdas: theta.slice(s![eta_penalty_count..rho_dim]).to_owned(),
                initial_beta: wiggle_initial_beta.clone(),
                gauge_priority: DEFAULT_GAUGE_PRIORITY,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            },
        ];
        Ok((resolvedspec, design, blocks, eta_derivs))
    };

    let build_eval = |theta: &Array1<f64>,
                      warm_cache: Option<&crate::custom_family::CustomFamilyWarmStart>,
                      need_hessian: bool|
     -> Result<
        (
            crate::custom_family::CustomFamilyJointHyperResult,
            TermCollectionSpec,
            TermCollectionDesign,
        ),
        String,
    > {
        let (resolvedspec, design, blocks, eta_derivs) = build_realized_blocks(theta)?;
        let eval = evaluate_custom_family_joint_hyper(
            &outer_family,
            &blocks,
            &outer_options,
            &theta.slice(s![0..rho_dim]).to_owned(),
            &[eta_derivs, Vec::new()],
            warm_cache,
            if need_hessian {
                crate::reml_contracts::EvalMode::ValueGradientHessian
            } else {
                crate::reml_contracts::EvalMode::ValueAndGradient
            },
        )?;
        Ok((eval, resolvedspec, design))
    };

    let build_efs = |theta: &Array1<f64>,
                     warm_cache: Option<&crate::custom_family::CustomFamilyWarmStart>|
     -> Result<crate::custom_family::CustomFamilyJointHyperEfsResult, String> {
        let (_, _, blocks, eta_derivs) = build_realized_blocks(theta)?;
        evaluate_custom_family_joint_hyper_efs(
            &outer_family,
            &blocks,
            &outer_options,
            &theta.slice(s![0..rho_dim]).to_owned(),
            &[eta_derivs, Vec::new()],
            warm_cache,
        )
        .map_err(|e| e.to_string())
    };

    use crate::model_types::EstimationError;
    use crate::solver::rho_optimizer::{
        DeclaredHessianForm, Derivative, OuterEval, OuterEvalOrder,
    };

    // Exact first-order AND second-order [rho, psi] calculus is available
    // for all inverse links via the shared jet formulas plus the generic
    // exact-Newton D_βH / D²_βH closures routed through
    // evaluate_custom_family_joint_hyper -> joint_outer_evaluate ->
    // BorrowedJointDerivProvider. This enables the analytic-Hessian outer
    // plan for REML optimization instead of the downgraded gradient-only
    // outer strategies.
    //
    // Spatial log-kappa coordinates are ψ (design-moving) dimensions because
    // they rebuild the spatial basis and penalties at each outer proposal.
    let analytic_outer_hessian_available = true;
    let mut seed_heuristic = theta0.to_vec();
    for value in &mut seed_heuristic[..rho_dim] {
        *value = value.exp();
    }
    let problem = crate::solver::rho_optimizer::OuterProblem::new(theta_dim)
        .with_gradient(Derivative::Analytic)
        .with_hessian(if analytic_outer_hessian_available {
            DeclaredHessianForm::Either
        } else {
            DeclaredHessianForm::Unavailable
        })
        .with_psi_dim(theta_dim - rho_dim)
        .with_tolerance(options.outer_tol)
        .with_max_iter(options.outer_max_iter)
        .with_bounds(lower.clone(), upper.clone())
        .with_initial_rho(theta0.clone())
        .with_seed_config(crate::seeding::SeedConfig {
            max_seeds: 4,
            seed_budget: 2,
            risk_profile: crate::seeding::SeedRiskProfile::GeneralizedLinear,
            num_auxiliary_trailing: theta_dim - rho_dim,
            ..Default::default()
        })
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_rho_bound(12.0)
        .with_heuristic_lambdas(seed_heuristic);

    let eval_outer = |state: &mut MeanWiggleOuterState,
                      theta: &Array1<f64>,
                      order: OuterEvalOrder|
     -> Result<OuterEval, EstimationError> {
        if let Some((cached_theta, cached_cost, cached_grad, cached_hess, cached_warm)) =
            &state.last_eval
            && cached_theta == theta
            && (!matches!(order, OuterEvalOrder::ValueGradientHessian)
                || matches!(
                    cached_hess,
                    crate::solver::rho_optimizer::HessianResult::Analytic(_)
                        | crate::solver::rho_optimizer::HessianResult::Operator(_)
                ))
        {
            state.warm_cache = Some(cached_warm.clone());
            return Ok(OuterEval {
                cost: *cached_cost,
                gradient: cached_grad.clone(),
                hessian: cached_hess.clone(),
                inner_beta_hint: None,
            });
        }
        let need_hessian = matches!(order, OuterEvalOrder::ValueGradientHessian)
            && analytic_outer_hessian_available;
        let (eval, _, _) = build_eval(theta, state.warm_cache.as_ref(), need_hessian)
            .map_err(EstimationError::InvalidInput)?;
        if !eval.inner_converged {
            state.warm_cache = Some(eval.warm_start);
            crate::bail_invalid_estim!(
                "binomial mean-wiggle exact spatial inner solve did not converge"
            );
        }
        let hessian_result = eval.outer_hessian.clone();
        state.last_eval = Some((
            theta.clone(),
            eval.objective,
            eval.gradient.clone(),
            eval.outer_hessian.clone(),
            eval.warm_start.clone(),
        ));
        state.warm_cache = Some(eval.warm_start);
        Ok(OuterEval {
            cost: eval.objective,
            gradient: eval.gradient,
            hessian: hessian_result,
            inner_beta_hint: None,
        })
    };

    let mut obj = problem.build_objective_with_screening_proxy(
        MeanWiggleOuterState {
            warm_cache: None,
            last_eval: None,
        },
        |state: &mut MeanWiggleOuterState, theta: &Array1<f64>| {
            if let Some((cached_theta, cached_cost, _, _, cached_warm)) = &state.last_eval
                && cached_theta == theta
            {
                state.warm_cache = Some(cached_warm.clone());
                return Ok(*cached_cost);
            }
            let (eval, _, _) = build_eval(theta, state.warm_cache.as_ref(), false)
                .map_err(EstimationError::InvalidInput)?;
            if !eval.inner_converged {
                state.warm_cache = Some(eval.warm_start);
                crate::bail_invalid_estim!(
                    "binomial mean-wiggle exact spatial cost inner solve did not converge"
                        .to_string(),
                );
            }
            state.warm_cache = Some(eval.warm_start);
            Ok(eval.objective)
        },
        |state: &mut MeanWiggleOuterState, theta: &Array1<f64>| {
            eval_outer(
                state,
                theta,
                if analytic_outer_hessian_available {
                    OuterEvalOrder::ValueGradientHessian
                } else {
                    OuterEvalOrder::ValueAndGradient
                },
            )
        },
        |state: &mut MeanWiggleOuterState, theta: &Array1<f64>, order: OuterEvalOrder| {
            eval_outer(state, theta, order)
        },
        Some(|state: &mut MeanWiggleOuterState| {
            state.warm_cache = None;
            state.last_eval = None;
        }),
        Some(|state: &mut MeanWiggleOuterState, theta: &Array1<f64>| {
            let eval = build_efs(theta, state.warm_cache.as_ref())
                .map_err(EstimationError::InvalidInput)?;
            if !eval.inner_converged {
                state.warm_cache = Some(eval.warm_start);
                crate::bail_invalid_estim!(
                    "binomial mean-wiggle exact spatial EFS inner solve did not converge"
                        .to_string(),
                );
            }
            state.warm_cache = Some(eval.warm_start);
            Ok(eval.efs_eval)
        }),
        // Seed-screening ranking proxy (#969). The cost closure above
        // hard-errors on a non-converged inner solve — correct for
        // line-search costs, but under the screening cap (wired into the
        // outer options and installed by the cascade) the inner solve is
        // truncated BY DESIGN, so screening through it rejects every seed
        // — the all-seeds-rejected front-door genus. Screening only RANKS
        // candidates: the truncated solve's penalized objective is the
        // ranking signal; convergence is demanded of the selected seed's
        // full-budget fit, not of capped probes.
        |state: &mut MeanWiggleOuterState, theta: &Array1<f64>| {
            if let Some((cached_theta, cached_cost, _, _, cached_warm)) = &state.last_eval
                && cached_theta == theta
            {
                state.warm_cache = Some(cached_warm.clone());
                return Ok(*cached_cost);
            }
            let (eval, _, _) = build_eval(theta, state.warm_cache.as_ref(), false)
                .map_err(EstimationError::InvalidInput)?;
            state.warm_cache = Some(eval.warm_start);
            Ok(eval.objective)
        },
    );

    let outer = problem
        .run(&mut obj, "binomial mean wiggle exact spatial hyper")
        .map_err(|e| e.to_string())?;
    if !outer.converged {
        return Err(GamlssError::NumericalFailure { reason: format!(
            "binomial mean wiggle exact spatial hyper did not converge after {} iterations (final_objective={:.6e}, final_grad_norm={})",
            outer.iterations,
            outer.final_value,
            outer.final_grad_norm_report(),
        ) }.into());
    }
    let theta_star = outer.rho;

    let log_kappa =
        SpatialLogKappaCoords::from_theta_tail_with_dims(&theta_star, rho_dim, dims_per_term);
    let resolvedspec = log_kappa
        .apply_tospec(&pilot_spec_cloned, &spatial_terms)
        .map_err(|e| e.to_string())?;
    let design = build_term_collection_design(data, &resolvedspec).map_err(|e| e.to_string())?;
    let resolvedspec =
        freeze_term_collection_from_design(&resolvedspec, &design).map_err(|e| e.to_string())?;
    let fit = fit_binomial_mean_wiggle(
        BinomialMeanWiggleSpec {
            y: y_cloned,
            weights: weights_cloned,
            link_kind: link_kind_cloned,
            wiggle_knots: wiggle_knots.clone(),
            wiggle_degree,
            eta_block: ParameterBlockInput {
                design: design.design.clone(),
                offset: Array1::zeros(y.len()),
                penalties: design
                    .penalties
                    .iter()
                    .map(crate::model_types::PenaltySpec::from_blockwise_ref)
                    .collect(),
                nullspace_dims: vec![],
                initial_log_lambdas: Some(theta_star.slice(s![0..eta_penalty_count]).to_owned()),
                initial_beta: Some(pilot_beta),
            },
            wiggle_block: ParameterBlockInput {
                design: wiggle_design,
                offset: wiggle_offset,
                penalties: wiggle_penalties,
                nullspace_dims: vec![],
                initial_log_lambdas: Some(
                    theta_star.slice(s![eta_penalty_count..rho_dim]).to_owned(),
                ),
                initial_beta: wiggle_initial_beta,
            },
        },
        options,
    )?;

    Ok(BinomialMeanWiggleTermFitResult {
        fit,
        resolvedspec,
        design,
        wiggle_knots,
        wiggle_degree,
    })
}
