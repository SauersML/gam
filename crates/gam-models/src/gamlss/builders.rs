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

/// The gauge level at which the frozen, observation-space-residualized warp
/// block yields to the mean block (#2748).
///
/// `audit_identifiability` calls a cross-block alias FATAL **only when the two
/// blocks carry the same priority**, on the stated grounds that "two blocks
/// contributing the same direction is only *unfittable* when no ordering exists
/// to pick which one to drop". On the binomial mean link-wiggle path an
/// ordering plainly exists, and the de-aliasing IS the proof of it: `B⊥` is by
/// construction the part of `B(η̂)` the mean block does not explain, so a
/// direction the two still share after that is one the mean block can represent
/// and the warp is echoing. The mean block stays intact; the warp yields.
///
/// Both blocks reached the audit at `DEFAULT_GAUGE_PRIORITY`, so every such
/// pair was unfittable-by-declaration. Measured on
/// `geo_disease_eas3_matern_k12`: `FATAL: alias pair 'eta'[11] ~ 'wiggle'[0]
/// overlap=0.6542 >= halt half-width 0.1739` at **joint rank 23 of 23 columns
/// and 0 dropped columns** — a full-rank, merely ill-conditioned design refused
/// outright.
///
/// Two properties make the demotion safe rather than merely permissive:
///
/// * at full rank nothing is dropped at all — the ordering only decides who
///   would yield, and the fit proceeds with both blocks whole;
/// * if a warp column IS dropped, monotonicity survives it. `dq/dη = 1 + Σ_j
///   β_j B'_j` with `β_w ≥ 0` and `B'` an M-spline basis, so every retained
///   term is still non-negative and the sum is still `≥ 1`. Dropping a column
///   from the penalized mean smooth instead would silently change that
///   smooth's span, which is the worse of the two.
///
/// The sibling `GaussianLocationScaleWiggleFamily` assigns
/// [`LINK_WIGGLE_GAUGE_PRIORITY`] the OTHER way round — mean and log-σ yield to
/// the wiggle — and that is not an inconsistency: its warp basis is dynamic and
/// full-width with no residualization establishing precedence, so there the
/// wiggle is the block that cannot afford to lose a column. Precedence follows
/// from which block is defined as the residual, and on this path that is the
/// warp.
pub(crate) const DEALIASED_WARP_GAUGE_PRIORITY: u8 = LINK_WIGGLE_GAUGE_PRIORITY;

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
    gam_problem::validate_log_strengths(lambdas.iter().copied())
        .map_err(|error| format!("initial_log_lambdas: {error}"))?;
    Ok(lambdas)
}

fn fitted_log_lambdas(lambdas: &Array1<f64>, context: &str) -> Result<Array1<f64>, String> {
    lambdas
        .iter()
        .copied()
        .enumerate()
        .map(|(coordinate, value)| {
            gam_problem::checked_log_strength(value)
                .map_err(|error| format!("{context} coordinate {coordinate}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Array1::from_vec)
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
) -> Result<ExactJointHyperSetup, gam_terms::basis::BasisError> {
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
        penalty: ActivePenaltyInfo {
            source: PenaltySource::Other("shrinkage".to_string()),
            original_index: 0,
            effective_rank: p,
            normalization_scale: 1.0,
            kronecker_factors: None,
            structural_null_frame: None,
        },
    });
}

/// Build the (mean, log-σ) parameter-block pair for a Gaussian location-scale
/// family. Shared verbatim by the non-wiggle and wiggle Gaussian builders so the
/// scale-block construction — prepared log-σ design, formula-native penalties,
/// and the joint Gaussian warm start — lives in exactly one place. Callers
/// supply the per-block log-λ vectors sliced from their own layout (two-block vs
/// with-wiggle) and append any extra blocks.
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
    let mean_offset = mean_design
        .compose_offset(mean_offset.view(), &format!("{context}: mu"))
        .map_err(|error| error.to_string())?;
    let noise_offset = noise_design
        .compose_offset(noise_offset.view(), &format!("{context}: log_sigma"))
        .map_err(|error| error.to_string())?;
    let mut meanspec = build_location_scale_block(
        "mu",
        mean_design.design.clone(),
        mean_offset,
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
    // The formula-native penalty topology is authoritative. Smooth terms carry
    // their own REML-selected null-space penalty when `double_penalty=true`
    // (the default), while an explicit `double_penalty=false` remains a real
    // opt-out. In particular the global log-σ intercept is likelihood-identified
    // and must stay unpenalized: adding a Gaussian-only projector over the joint
    // null space placed a data-scale-dependent prior on the overall σ level and
    // introduced an extra smoothing coordinate absent from the formula (#1561).
    let mut noisespec = build_location_scale_block(
        "log_sigma",
        prepared_noise_design,
        noise_offset,
        noise_design.penalties_as_penalty_matrix(),
        noise_design.nullspace_dims.clone(),
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
    let mean_offset = mean_design
        .compose_offset(mean_offset.view(), &format!("{context}: threshold"))
        .map_err(|error| error.to_string())?;
    let noise_offset = noise_design
        .compose_offset(noise_offset.view(), &format!("{context}: log_sigma"))
        .map_err(|error| error.to_string())?;
    let identifiednoise_design =
        identified_binomial_log_sigma_design(mean_design, noise_design, weights)?;
    let p_noise = identifiednoise_design.ncols();
    let mut log_sigma_penalty_matrices: Vec<PenaltyMatrix> =
        noise_design.penalties_as_penalty_matrix();
    log_sigma_penalty_matrices.push(PenaltyMatrix::Dense(identity_penalty(p_noise)));
    let mut thresholdspec = build_location_scale_block(
        "threshold",
        mean_design.design.clone(),
        mean_offset,
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
        noise_offset,
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
        if !wi.is_finite() || wi < 0.0 {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "binomial location-scale warm start requires finite non-negative weights; weight={wi}"
                ),
            }
            .into());
        }
        if wi > 0.0 {
            weight_sum += wi;
            success_sum += wi * yi;
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
    /// Standard I-spline warp coefficients `β_w` for the saved-model predict
    /// runtime when frozen-basis de-aliasing engaged (#1596). Observation-space
    /// residualization preserves this coefficient chart, so the fit and predict
    /// runtime consume the same non-negative vector.
    pub saved_warp_beta: Option<Vec<f64>>,
    /// Frozen-index mean-coordinate shift `s = β_frozen_source − β_saved` for the
    /// predict runtime (#2141). Predict evaluates the warp basis at
    /// `X·(β_saved + s) = η̂` (the frozen index the fit pinned `B` at) instead of
    /// the de-aliased base predictor `X·β_saved`, reproducing the fitted `q`.
    pub saved_index_shift: Option<Vec<f64>>,
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
        })?;
        Ok(())
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

/// Exact coefficient-frame map for the frozen-basis binomial mean-wiggle
/// de-aliasing step.
///
/// The joint solver sees `[X, B - XA]` and returns coordinates
/// `(beta_mean_solver, beta_w)`. Saved prediction deliberately uses `[X, B]`,
/// so the reported coordinates are
///
/// ```text
/// beta_mean_saved = beta_mean_solver - A beta_w
/// beta_w_saved    = beta_w.
/// ```
///
/// This is one linear section with the cross-block lift
/// `M = [[I, -A], [0, I]]`. Keeping it as a [`gam_problem::Gauge`] makes the
/// same map authoritative for coefficients, covariance, and the active
/// geometry lineage used by saved-model ALO.
fn binomial_mean_wiggle_saved_frame_gauge(
    alias: &Array2<f64>,
    mean_width: usize,
    wiggle_width: usize,
) -> Result<gam_problem::Gauge, String> {
    if alias.dim() != (mean_width, wiggle_width) {
        return Err(format!(
            "binomial mean-wiggle de-alias map is {}x{}, expected {mean_width}x{wiggle_width}",
            alias.nrows(),
            alias.ncols(),
        ));
    }
    let total_width = mean_width
        .checked_add(wiggle_width)
        .ok_or_else(|| "binomial mean-wiggle coefficient dimension overflows usize".to_string())?;
    let mut transform = Array2::<f64>::eye(total_width);
    for row in 0..mean_width {
        for column in 0..wiggle_width {
            transform[[row, mean_width + column]] = -alias[[row, column]];
        }
    }
    let gauge = gam_problem::Gauge::from_t(
        transform,
        &[mean_width, wiggle_width],
        &[mean_width, wiggle_width],
    );
    gauge.validate().map_err(|reason| {
        format!("binomial mean-wiggle saved coefficient gauge is invalid: {reason}")
    })?;
    Ok(gauge)
}

fn binomial_mean_wiggle_saved_geometry(
    geometry: &gam_solve::model_types::FitGeometry,
    saved_frame: &gam_problem::Gauge,
) -> Result<gam_solve::model_types::FitGeometry, String> {
    let mut saved_geometry = geometry.clone();
    saved_geometry.coefficient_gauge = geometry
        .coefficient_gauge
        .left_compose(saved_frame)
        .map_err(|reason| {
            format!(
                "binomial mean-wiggle active geometry cannot compose with its exact saved-result gauge: {reason}"
            )
        })?;
    Ok(saved_geometry)
}

fn binomial_mean_wiggle_saved_covariance(
    covariance: &Array2<f64>,
    saved_frame: &gam_problem::Gauge,
    label: &str,
) -> Result<Array2<f64>, String> {
    let expected = saved_frame.reduced_total();
    if covariance.dim() != (expected, expected) {
        return Err(format!(
            "binomial mean-wiggle {label} is {}x{}; exact saved-result gauge requires {expected}x{expected} solver-frame coordinates",
            covariance.nrows(),
            covariance.ncols(),
        ));
    }
    if let Some(((row, column), value)) = covariance
        .indexed_iter()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "binomial mean-wiggle {label} is non-finite at ({row}, {column}): {value}"
        ));
    }
    let saved = saved_frame.lift_covariance(covariance);
    if let Some(((row, column), value)) = saved.indexed_iter().find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "binomial mean-wiggle saved-frame {label} is non-finite at ({row}, {column}): {value}"
        ));
    }
    Ok(saved)
}

/// Atomically move a converged frozen-basis mean-wiggle fit from the solver's
/// residualized-design coordinates into the saved prediction coordinates.
///
/// The penalized Hessian remains in its exact active solver coordinates;
/// composing its coefficient gauge records how raw saved rows pull back into
/// that frame. Covariances, in contrast, push forward through the saved-frame
/// map. No dimension mismatch is ignorable: returning a partially transformed
/// fit would make point estimates and uncertainty describe different models.
fn finalize_binomial_mean_wiggle_saved_frame(
    fit: &mut UnifiedFitResult,
    alias: &Array2<f64>,
    mean_design: &Array2<f64>,
    mean_offset: &Array1<f64>,
) -> Result<(), String> {
    use gam_problem::BlockRole;

    if fit.blocks.len() != 2
        || fit.blocks[0].role != BlockRole::Mean
        || fit.blocks[1].role != BlockRole::LinkWiggle
    {
        return Err(format!(
            "binomial mean-wiggle saved-frame finalization requires fitted blocks [Mean, LinkWiggle], got {:?}",
            fit.blocks
                .iter()
                .map(|block| block.role)
                .collect::<Vec<_>>()
        ));
    }
    if fit.block_states.len() != 2 {
        return Err(format!(
            "binomial mean-wiggle saved-frame finalization requires two fitted block states, got {}",
            fit.block_states.len(),
        ));
    }
    if mean_offset.len() != mean_design.nrows() {
        return Err(format!(
            "binomial mean-wiggle mean offset has {} rows, expected {}",
            mean_offset.len(),
            mean_design.nrows(),
        ));
    }

    let mean_width = fit.blocks[0].beta.len();
    let wiggle_width = fit.blocks[1].beta.len();
    if mean_design.ncols() != mean_width {
        return Err(format!(
            "binomial mean-wiggle saved mean design has {} columns, expected fitted width {mean_width}",
            mean_design.ncols(),
        ));
    }
    for block_index in 0..2 {
        if fit.block_states[block_index].beta != fit.blocks[block_index].beta {
            return Err(format!(
                "binomial mean-wiggle fitted block {block_index} and block-state coefficients disagree before saved-frame finalization"
            ));
        }
    }
    let total_width = mean_width
        .checked_add(wiggle_width)
        .ok_or_else(|| "binomial mean-wiggle coefficient dimension overflows usize".to_string())?;
    if fit.beta.len() != total_width {
        return Err(format!(
            "binomial mean-wiggle flat coefficient vector has width {}, expected {total_width}",
            fit.beta.len(),
        ));
    }
    if fit.beta.slice(s![0..mean_width]) != fit.blocks[0].beta
        || fit.beta.slice(s![mean_width..total_width]) != fit.blocks[1].beta
    {
        return Err(
            "binomial mean-wiggle flat and block coefficient vectors disagree before saved-frame finalization"
                .to_string(),
        );
    }

    let saved_frame = binomial_mean_wiggle_saved_frame_gauge(alias, mean_width, wiggle_width)?;
    let saved_blocks =
        saved_frame.lift_block_betas(&[fit.blocks[0].beta.clone(), fit.blocks[1].beta.clone()]);
    let saved_mean_eta = mean_design.dot(&saved_blocks[0]) + mean_offset;
    let mut saved_beta = Array1::<f64>::zeros(total_width);
    saved_beta
        .slice_mut(s![0..mean_width])
        .assign(&saved_blocks[0]);
    saved_beta
        .slice_mut(s![mean_width..total_width])
        .assign(&saved_blocks[1]);

    let saved_conditional = fit
        .covariance_conditional
        .as_ref()
        .map(|covariance| {
            binomial_mean_wiggle_saved_covariance(
                covariance,
                &saved_frame,
                "conditional covariance",
            )
        })
        .transpose()?;
    let saved_corrected = fit
        .covariance_corrected
        .as_ref()
        .map(|covariance| {
            binomial_mean_wiggle_saved_covariance(covariance, &saved_frame, "corrected covariance")
        })
        .transpose()?;
    let saved_geometry = binomial_mean_wiggle_saved_geometry(
        fit.geometry.as_ref().ok_or_else(|| {
            "binomial mean-wiggle fit is missing its exact active geometry".to_string()
        })?,
        &saved_frame,
    )?;

    let mut saved_inference = fit.inference.clone();
    if let Some(inference) = saved_inference.as_mut() {
        if inference.beta_covariance.is_none() && inference.beta_standard_errors.is_some() {
            return Err(
                "binomial mean-wiggle inference has conditional standard errors without their covariance"
                    .to_string(),
            );
        }
        if inference.beta_covariance_corrected.is_none()
            && inference.beta_standard_errors_corrected.is_some()
        {
            return Err(
                "binomial mean-wiggle inference has corrected standard errors without their covariance"
                    .to_string(),
            );
        }
        if let Some(covariance) = inference.beta_covariance.take() {
            let covariance = binomial_mean_wiggle_saved_covariance(
                covariance.as_array(),
                &saved_frame,
                "inference conditional covariance",
            )?;
            if inference.beta_standard_errors.is_some() {
                inference.beta_standard_errors = Some(
                    gam_problem::se_from_covariance(&covariance).map_err(|reason| {
                        format!(
                            "binomial mean-wiggle saved conditional standard errors are invalid: {reason}"
                        )
                    })?,
                );
            }
            inference.beta_covariance = Some(covariance.into());
        }
        if let Some(covariance) = inference.beta_covariance_corrected.take() {
            let covariance = binomial_mean_wiggle_saved_covariance(
                &covariance,
                &saved_frame,
                "inference corrected covariance",
            )?;
            if inference.beta_standard_errors_corrected.is_some() {
                inference.beta_standard_errors_corrected = Some(
                    gam_problem::se_from_covariance(&covariance).map_err(|reason| {
                        format!(
                            "binomial mean-wiggle saved corrected standard errors are invalid: {reason}"
                        )
                    })?,
                );
            }
            inference.beta_covariance_corrected = Some(covariance);
        }
        if let Some(covariance) = inference.beta_covariance_frequentist.take() {
            inference.beta_covariance_frequentist = Some(binomial_mean_wiggle_saved_covariance(
                &covariance,
                &saved_frame,
                "frequentist covariance",
            )?);
        }
        if let Some(correction) = inference.smoothing_correction.take() {
            inference.smoothing_correction = Some(binomial_mean_wiggle_saved_covariance(
                &correction,
                &saved_frame,
                "smoothing covariance correction",
            )?);
        }
    }

    fit.blocks[0].beta = saved_blocks[0].clone();
    fit.blocks[1].beta = saved_blocks[1].clone();
    fit.block_states[0].beta = saved_blocks[0].clone();
    fit.block_states[0].eta = saved_mean_eta;
    fit.block_states[1].beta = saved_blocks[1].clone();
    fit.beta = saved_beta;
    fit.covariance_conditional = saved_conditional;
    fit.covariance_corrected = saved_corrected;
    fit.geometry = Some(saved_geometry);
    fit.inference = saved_inference;
    Ok(())
}

#[cfg(test)]
mod binomial_mean_wiggle_saved_frame_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cross_block_dealias_composes_non_square_geometry_and_pushes_covariance() {
        let alias = array![[2.0], [-0.5]];
        let saved_frame = binomial_mean_wiggle_saved_frame_gauge(&alias, 2, 1)
            .expect("valid cross-block de-alias map");

        // The canonical solver retained one of two Mean directions plus the
        // LinkWiggle direction: active(2) -> solver raw(3) is rectangular.
        let active_to_solver = gam_problem::Gauge::from_t(
            array![[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            &[2, 1],
            &[1, 1],
        );
        let active_hessian = array![[7.0, 1.5], [1.5, 4.0]];
        let geometry = gam_solve::model_types::FitGeometry {
            coefficient_gauge: active_to_solver,
            penalized_hessian: active_hessian.clone().into(),
            constrained_posterior: None,
            working: None,
        };
        let saved_geometry = binomial_mean_wiggle_saved_geometry(&geometry, &saved_frame)
            .expect("non-square active geometry composes through saved frame");

        assert_eq!(
            saved_geometry.coefficient_gauge.t_full,
            array![[1.0, -2.0], [0.0, 0.5], [0.0, 1.0]],
        );
        assert_eq!(
            saved_geometry.penalized_hessian.as_array(),
            &active_hessian,
            "precision stays in the canonical active frame",
        );

        let solver_covariance = Array2::<f64>::eye(3);
        let saved_covariance = binomial_mean_wiggle_saved_covariance(
            &solver_covariance,
            &saved_frame,
            "test covariance",
        )
        .expect("covariance pushes into saved frame");
        assert_eq!(
            saved_covariance,
            array![[5.0, -1.0, -2.0], [-1.0, 1.25, 0.5], [-2.0, 0.5, 1.0]],
            "the -A cross block must alter both Mean variance and Mean/Wiggle covariance",
        );
    }
}

/// Residualize a frozen warp basis against the mean block **in the metric the
/// frozen joint solve's own Hessian defines** (#2748).
///
/// Returns `(A, B⊥)` with
///
/// ```text
///     A  = (XᵀWX)⁺ XᵀWB,     B⊥ = B − X·A,     W = diag(curvature),
/// ```
///
/// so that `Xᵀ W B⊥ = 0` on the range the pseudo-inverse retains.
///
/// # ⚠ The metric is not decoration — it is the whole content of "identifiable"
///
/// "The part of `B` outside the mean column space" is meaningless until an
/// inner product says what *outside* means, and the only inner product the rest
/// of this solve uses is the one its own curvature defines. With the warp basis
/// frozen, `q = Xβ + B⊥β_w` is linear in `(β, β_w)`, so the joint
/// negative-log-likelihood Hessian the fit assembles is EXACTLY
///
/// ```text
///     [X B⊥]ᵀ diag(m₂) [X B⊥] + penalties,      m₂ = ∂²(−ℓ)/∂q²
/// ```
///
/// — see [`BinomialMeanWiggleFamily::bmw_static_hessian_operator`], whose
/// frozen arm has `dq_dq0 = 1` and `basis_d1 = 0`, collapsing its four row
/// coefficients to `m₂` in every block. Its cross block is `Xᵀ diag(m₂) B⊥`,
/// and `W = diag(m₂)` makes that **zero by construction**. The two blocks are
/// then exactly decoupled in the curvature the solve reads, which is what
/// de-aliasing was for.
///
/// Taking `W = I` instead orthogonalizes in a metric no part of the problem
/// uses. It leaves `Xᵀ diag(m₂) B⊥ ≠ 0`, and the residue is largest exactly
/// where binomial data are least informative, because `m₂ = w·μ(1−μ)` on a
/// logit spans orders of magnitude across rows near saturation. Two
/// consequences:
///
/// * **The outer frozen-index fixed point stops contracting.** Freeze at `η̂`
///   and perturb by `δ`: `B(η̂+δ) ≈ B + diag(δ)B'`, so the design moves by
///   `(I−P)diag(s)δ` in `q` with `s = B'β_w`, and the refit answers with
///   `Δη = −H(I−P)diag(s)δ`, where `H = X(XᵀWX + S)⁻¹XᵀW` is this fit's own
///   hat matrix. With `P` the `W`-projection, `H(I−P) = 0` identically — *for
///   any penalty `S`*, since `XᵀW(I−P) = XᵀW − XᵀWX(XᵀWX)⁻¹XᵀW = 0` — so the
///   leading term of the outer map's derivative vanishes. With `P` the
///   Euclidean projection it does not, and the map then contracts only while
///   `max_i s_i` is small, which nothing enforces: monotonicity bounds `β_w`
///   below at zero and not at all above.
/// * **The `no identifiable warp direction` refusal stops firing on rows that
///   carry no information.** An unweighted projection spends its budget
///   fitting `B` on saturated rows, where `μ(1−μ) ≈ 0` and the likelihood
///   cannot tell `B` from `X` anyway; the `W`-projection leaves those rows in
///   the residual, where they belong.
///
/// # Degenerate metrics
///
/// A row whose curvature is not a usable weight — non-finite, or negative,
/// which a non-canonical link's OBSERVED information can be — must be handed
/// in as `0` by the caller. A metric with zero rows is still a valid
/// semi-inner product, and the eigen pseudo-inverse below already handles the
/// rank deficiency it can produce: directions of `XᵀWX` under the cutoff are
/// dropped from `A`, which leaves their (equally negligible) alias in `B⊥`.
///
/// The cutoff is purely RELATIVE, deliberately. `XᵀWX`'s scale carries the
/// curvature's units, which on a binomial logit sit well below one; an
/// absolute floor would read the whole spectrum as numerical noise and
/// residualize nothing.
pub(crate) fn dealias_warp_against_mean_block(
    x: &Array2<f64>,
    b_full: &Array2<f64>,
    curvature: &Array1<f64>,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    let n = x.nrows();
    if b_full.nrows() != n || curvature.len() != n {
        return Err(format!(
            "frozen-basis warp de-aliasing row mismatch: mean block has {n} row(s), warp basis \
             has {}, curvature has {}",
            b_full.nrows(),
            curvature.len()
        ));
    }
    if let Some(bad) = curvature
        .iter()
        .position(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(format!(
            "frozen-basis warp de-aliasing metric is not a semi-inner product: curvature[{bad}] \
             = {}",
            curvature[bad]
        ));
    }
    let mut weighted_x = x.clone();
    for row in 0..n {
        let weight = curvature[row];
        weighted_x.row_mut(row).map_inplace(|value| *value *= weight);
    }
    let xtwx = x.t().dot(&weighted_x);
    let xtwb = weighted_x.t().dot(b_full);
    let (evals, evecs) = xtwx
        .eigh(Side::Lower)
        .map_err(|e| format!("frozen-basis warp de-aliasing mean QR failed: {e}"))?;
    let max_eval = evals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let cutoff = 1.0e3 * f64::EPSILON * (xtwx.nrows().max(1) as f64) * max_eval;
    let mut alias = Array2::<f64>::zeros((x.ncols(), b_full.ncols()));
    for k in 0..evals.len() {
        let lam = evals[k];
        if !lam.is_finite() || lam.abs() <= cutoff {
            continue;
        }
        let uk = evecs.column(k);
        let uk_xtwb = uk.t().dot(&xtwb);
        for i in 0..alias.nrows() {
            for j in 0..alias.ncols() {
                alias[[i, j]] += uk[i] * uk_xtwb[j] / lam;
            }
        }
    }
    let bda = b_full - &x.dot(&alias);
    Ok((alias, bda))
}

/// The dominant eigenvalue of the frozen-index map's Jacobian, measured from
/// two consecutive residuals and the relaxation that separated them (#2748).
///
/// Let `Φ` be the frozen-index map, `η*` its fixed point, `M = Φ'(η*)` and
/// `e_k = η_k − η*`. The residual available at each pass is
/// `d_k = Φ(η_k) − η_k = (M−I)e_k`. The relaxed advance `η_{k+1} = η_k + t_k d_k`
/// gives `e_{k+1} = ((1−t_k)I + t_k M) e_k`, and because `M` commutes with
/// `M − I`,
///
/// ```text
///     d_{k+1} = ((1−t_k)I + t_k M) d_k
/// ```
///
/// exactly. So `⟨d_{k+1}, d_k⟩ / ‖d_k‖²` is the Rayleigh quotient of the
/// RELAXED Jacobian on a vector the iteration has aligned with its dominant
/// eigenvector: it measures `ρ = (1−t_k) + t_k·mu`, and the map's own multiplier
/// is `mu = 1 + (ρ − 1)/t_k`. Reading `ρ` as `mu` is what locked the n=1000
/// `geo_disease_matern` flexible cell into a period-2 orbit: after the first
/// damped pass annihilated the dominant mode, the next quotient was the mixed
/// sub-dominant ratio `−0.229`, read as `mu` it set `t` back to `0.814`, that
/// re-excited the mode by `1 + 0.814·(mu−1) = −4.36`, which read as `mu` set
/// `t = 0.186`, and so on: `t_k·t_{k+1} = 1/(1−mu)` is an involution, so every
/// `t` other than its fixed point lies on a two-cycle. Sixty passes alternated
/// `delta = 1.97e-3 ↔ 4.51e-4` with `cos(step_k, step_{k−1}) = −1.000`.
///
/// At `previous_relaxation == 1` the quotient IS `mu` and is returned untouched,
/// so every undamped pass is bit-for-bit what it was.
///
/// Returns `NaN` when there is no previous residual, when either has zero
/// length, or when the previous relaxation is not a positive finite number — a
/// Rayleigh quotient of nothing is not a measurement.
pub(crate) fn fixed_point_dominant_multiplier(
    previous_residual: Option<&Array1<f64>>,
    previous_relaxation: f64,
    residual: &Array1<f64>,
) -> f64 {
    let Some(previous) = previous_residual else {
        return f64::NAN;
    };
    if previous.len() != residual.len() {
        return f64::NAN;
    }
    let denominator = previous.dot(previous);
    if !(denominator > 0.0) || !residual.dot(residual).is_finite() {
        return f64::NAN;
    }
    let quotient = residual.dot(previous) / denominator;
    if !quotient.is_finite() {
        return f64::NAN;
    }
    if previous_relaxation == 1.0 {
        return quotient;
    }
    if !(previous_relaxation.is_finite() && previous_relaxation > 0.0) {
        return f64::NAN;
    }
    let multiplier = 1.0 + (quotient - 1.0) / previous_relaxation;
    if multiplier.is_finite() { multiplier } else { f64::NAN }
}

/// The relaxation `t` that removes the dominant mode of a frozen-index
/// iteration whose measured multiplier is `mu` (#2748).
///
/// Relaxing the map to `η + t(Φ(η) − η)` replaces its Jacobian `M` by
/// `(1−t)I + tM`, so the dominant mode's multiplier becomes `(1−t) + t·mu`,
/// and the `t` that annihilates it is `1/(1 − mu)`. That is Aitken's Δ² for
/// this iteration, written as a relaxation.
///
/// **This function only ever damps.** `1/(1 − mu) ≥ 1` for `mu ∈ [0, 1)`, and
/// on that branch it returns `1` — the undamped step, bit for bit. Extrapolating
/// a monotone-but-slow mode would be sound for the mode it measures and could
/// destabilise a mode of equal magnitude and opposite sign that the Rayleigh
/// quotient cannot see, and the failing population here does not need it. So
/// the return value is `min(1, 1/(1 − mu))`, i.e.:
///
/// * `mu < 0` — an ALTERNATING mode, the divergence signature this issue's
///   cluster shows: `t = 1/(1 + |mu|) ∈ (0, 1)`, exactly enough damping to
///   cancel it. `mu = −4` gives `t = 0.2`.
/// * `0 ≤ mu` — `t = 1`. Every pass of every currently-converging fit takes
///   this branch, including the first pass of all of them, where `mu` is not
///   measurable at all.
///
/// `mu ≥ 1` is not a case this can repair and does not pretend to be: the
/// dominant mode is a monotone divergence, `(1−t) + t·mu ≥ 1` for every `t > 0`,
/// and no relaxation of any size stabilises it. The caller reports `mu` in the
/// refusal instead.
pub(crate) fn frozen_index_relaxation(dominant_multiplier: f64) -> f64 {
    if !dominant_multiplier.is_finite() || dominant_multiplier >= 0.0 {
        return 1.0;
    }
    let relaxation = 1.0 / (1.0 - dominant_multiplier);
    if relaxation.is_finite() && relaxation > 0.0 {
        relaxation
    } else {
        1.0
    }
}

/// Which step the frozen-index loop took on a pass (#2748).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrozenIndexAdvance {
    /// The multisecant step over `order` stored residual differences.
    Anderson { order: usize },
    /// The scalar relaxed step `t · (Φ(x) − x)`: the first pass, and the pass
    /// after a safeguard reset, which reseeds the history.
    Relaxed,
}

impl std::fmt::Display for FrozenIndexAdvance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Anderson { order } => write!(f, "anderson(order={order})"),
            Self::Relaxed => write!(f, "relaxed"),
        }
    }
}

/// Multi-mode acceleration of the frozen-index fixed point (#2748).
///
/// A scalar relaxation annihilates ONE mode. The `geo_disease_matern` flexible
/// cell at n=1000 showed what that costs once the multiplier is read correctly:
/// `t = 1/(1−mu)` kills the dominant alternating mode (`mu ≈ −5.6`), the
/// residual is then a mild mode whose own relaxation is `≈ 1`, and that step
/// regrows the dominant mode by `|1 + (mu − 1)| ≈ 6` per pass — a nine-pass
/// cycle with `delta` floored at `4.2e-4` against a tolerance of `2.5e-5`.
///
/// Anderson mixing is the multisecant generalisation of that relaxation: with
/// `k` stored residual differences it annihilates `k` modes at once (on a
/// linear map it is GMRES on the residual). The accelerator is
/// [`gam_linalg::anderson::AndersonAccelerator`]; its depth is the pass budget
/// (full memory — the Gram's derived roundoff floor retires the directions the
/// history cannot resolve, so no separate memory knob is chosen here), and the
/// safeguard is the map's own residual: a pass whose undamped residual norm is
/// not below the previous pass's means the last accelerated step was not a
/// contraction, so the history is discarded and that pass takes the scalar
/// relaxed step, which reseeds it (Walker & Ni §5; Toth & Kelley 2015).
///
/// The state mixed is `[β_source; η]`; both parts move by the same affine
/// combination, so `η = Xβ + offset` survives every step by construction.
pub(crate) struct FrozenIndexMixer {
    accelerator: gam_linalg::anderson::AndersonAccelerator,
    last_advance: Option<Vec<f64>>,
    last_residual_norm: Option<f64>,
}

impl FrozenIndexMixer {
    pub(crate) fn new(pass_budget: usize) -> Result<Self, String> {
        let accelerator = gam_linalg::anderson::AndersonAccelerator::new(pass_budget)
            .map_err(|error| format!("fit_binomial_mean_wiggle: {error}"))?;
        Ok(Self {
            accelerator,
            last_advance: None,
            last_residual_norm: None,
        })
    }

    /// The undamped residual norm recorded by the last [`Self::advance`].
    pub(crate) fn last_residual_norm(&self) -> Option<f64> {
        self.last_residual_norm
    }

    /// Given the undamped map residual `Φ(x_k) − x_k` and the scalar relaxation
    /// the pass would otherwise apply, return the step to take from `x_k`, which
    /// kind it is, and whether the history was reset on this pass.
    pub(crate) fn advance(
        &mut self,
        residual: &[f64],
        relaxation: f64,
    ) -> Result<(Vec<f64>, FrozenIndexAdvance, bool), String> {
        let norm = residual.iter().map(|value| value * value).sum::<f64>().sqrt();
        let mut reset = false;
        if let Some(previous) = self.last_residual_norm
            && !(norm < previous)
        {
            self.accelerator.reset();
            self.last_advance = None;
            reset = true;
        }
        self.last_residual_norm = Some(norm);
        let taken = self
            .last_advance
            .take()
            .unwrap_or_else(|| vec![0.0; residual.len()]);
        let proposal = self
            .accelerator
            .propose(residual, &taken)
            .map_err(|error| format!("fit_binomial_mean_wiggle: frozen-index mixing: {error}"))?;
        let (step, kind) = match proposal {
            Some(step) => (
                step,
                FrozenIndexAdvance::Anderson {
                    order: self.accelerator.history_len(),
                },
            ),
            None => (
                residual.iter().map(|value| value * relaxation).collect(),
                FrozenIndexAdvance::Relaxed,
            ),
        };
        self.last_advance = Some(step.clone());
        Ok((step, kind, reset))
    }
}

#[cfg(test)]
mod dealiased_warp_gauge_priority_tests {
    use super::*;

    /// Two blocks that share a direction, at the geometry
    /// `geo_disease_eas3_matern_k12` refused on: FULL joint rank, nothing to
    /// drop, and a cross-block column overlap far above the leverage-based halt
    /// band. At equal priorities `audit_identifiability` calls that FATAL — "no
    /// ordering exists to pick which one to drop". Declaring the ordering the
    /// de-aliasing already implements is the whole of the repair, so this gates
    /// exactly that difference and nothing else: same designs, same overlap,
    /// only the priorities move.
    fn overlapping_blocks(warp_priority: u8) -> Vec<gam_problem::ParameterBlockSpec> {
        let n = 400;
        let mut mean = Array2::<f64>::zeros((n, 2));
        let mut warp = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let t = row as f64 / (n as f64 - 1.0);
            mean[[row, 0]] = 1.0;
            mean[[row, 1]] = t;
            // ~0.65 overlap with `mean[.., 1]` by construction, plus an
            // independent direction so the joint design stays full rank.
            warp[[row, 0]] = t + 0.75 * (7.0 * t).sin();
            warp[[row, 1]] = (3.0 * t).cos();
        }
        let mean_block = ParameterBlockInput {
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(mean)),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Some(Array1::zeros(0)),
            initial_beta: None,
        };
        let warp_block = ParameterBlockInput {
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(warp)),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Some(Array1::zeros(0)),
            initial_beta: None,
        };
        vec![
            mean_block.intospec("eta").expect("eta spec"),
            warp_block
                .intospec_with_gauge_priority("wiggle", warp_priority)
                .expect("wiggle spec"),
        ]
    }

    #[test]
    fn equal_priorities_make_a_full_rank_overlap_fatal() {
        let specs = overlapping_blocks(DEFAULT_GAUGE_PRIORITY);
        let audit = gam_identifiability::audit::audit_identifiability(&specs).expect("audit");
        assert!(
            audit.dropped_columns.is_empty(),
            "the fixture must be full rank so the verdict is about the ORDERING, not a drop: {}",
            audit.summary
        );
        assert!(
            audit.fatal,
            "the pre-#2748 declaration must still read as unfittable: {}",
            audit.summary
        );
    }

    #[test]
    fn the_warp_yielding_to_the_mean_block_makes_the_same_overlap_fittable() {
        let specs = overlapping_blocks(DEALIASED_WARP_GAUGE_PRIORITY);
        let audit = gam_identifiability::audit::audit_identifiability(&specs).expect("audit");
        assert!(
            audit.dropped_columns.is_empty(),
            "declaring an ordering must not itself drop a column at full rank: {}",
            audit.summary
        );
        assert!(
            !audit.fatal,
            "an ordering exists on this path, so the same overlap is ill-conditioned rather \
             than unfittable: {}",
            audit.summary
        );
    }

    /// And the ordering points the right way: the warp is the block that yields.
    #[test]
    fn the_warp_is_the_lower_priority_block() {
        assert!(DEALIASED_WARP_GAUGE_PRIORITY < DEFAULT_GAUGE_PRIORITY);
    }
}

#[cfg(test)]
mod frozen_index_relaxation_tests {
    use super::*;
    use ndarray::array;

    /// The Rayleigh quotient is exact on a residual pair produced by an actual
    /// linear map, which is the situation the derivation claims.
    #[test]
    fn the_multiplier_recovers_a_planted_linear_map() {
        // A diagonal map with a dominant alternating mode.
        let previous = array![1.0_f64, 0.0, 0.0];
        let residual = array![-3.5_f64, 0.0, 0.0];
        let mu = fixed_point_dominant_multiplier(Some(&previous), 1.0, &residual);
        assert!((mu + 3.5).abs() <= 1.0e-14, "mu={mu}");
        // And the relaxation it implies annihilates that mode: the relaxed
        // multiplier `(1-t) + t*mu` is zero.
        let t = frozen_index_relaxation(mu);
        assert!(((1.0 - t) + t * mu).abs() <= 1.0e-14, "t={t} mu={mu}");
    }

    /// A component orthogonal to the previous residual does not contribute to
    /// the quotient, which is what makes it an estimate of the DOMINANT mode
    /// rather than of the step's length.
    #[test]
    fn an_orthogonal_component_does_not_move_the_multiplier() {
        let previous = array![1.0_f64, 0.0];
        let aligned = array![-2.0_f64, 0.0];
        let with_orthogonal = array![-2.0_f64, 9.0];
        let bare = fixed_point_dominant_multiplier(Some(&previous), 1.0, &aligned);
        let mixed = fixed_point_dominant_multiplier(Some(&previous), 1.0, &with_orthogonal);
        assert!((bare - mixed).abs() <= 1.0e-14, "{bare} vs {mixed}");
    }

    /// The whole no-regression claim: a non-alternating pass, and the first
    /// pass of every fit, take the undamped step bit for bit.
    #[test]
    fn a_non_alternating_pass_is_the_undamped_step_exactly() {
        for mu in [0.0_f64, 0.25, 0.9, 1.0, 4.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                frozen_index_relaxation(mu).to_bits(),
                1.0_f64.to_bits(),
                "mu={mu} must take the undamped branch"
            );
        }
        let residual = array![1.0_f64, -2.0];
        assert!(fixed_point_dominant_multiplier(None, 1.0, &residual).is_nan());
        assert_eq!(
            frozen_index_relaxation(fixed_point_dominant_multiplier(None, 1.0, &residual)).to_bits(),
            1.0_f64.to_bits(),
        );
    }

    /// Damping never overshoots into a step in the wrong direction and never
    /// grows the step: `t` stays in `(0, 1]` for every representable input.
    #[test]
    fn the_relaxation_never_leaves_the_unit_interval() {
        for mu in [
            -1.0e300_f64,
            -1.0e6,
            -12.0,
            -1.0,
            -1.0e-12,
            -0.0,
            0.5,
            1.0e300,
        ] {
            let t = frozen_index_relaxation(mu);
            assert!(t > 0.0 && t <= 1.0, "mu={mu} gave t={t}");
        }
    }

    /// The measured damping actually turns the archetypal diverging two-cycle
    /// into a convergent one: simulate the scalar iteration the derivation
    /// models, with a multiplier that makes the undamped map diverge.
    #[test]
    fn the_derived_relaxation_converges_a_map_the_undamped_iteration_diverges_on() {
        let multiplier = -2.5_f64; // |M| > 1: the undamped iteration blows up.
        let fixed_point = 0.75_f64;
        let map = |eta: f64| fixed_point + multiplier * (eta - fixed_point);

        let mut undamped = 1.0_f64;
        for _ in 0..40 {
            undamped = map(undamped);
        }
        assert!(
            (undamped - fixed_point).abs() > 1.0e6,
            "the control must diverge, got {undamped}"
        );

        let mut eta = 1.0_f64;
        let mut previous: Option<Array1<f64>> = None;
        let mut previous_t = 1.0_f64;
        for _ in 0..40 {
            let residual = Array1::from_elem(1, map(eta) - eta);
            let mu = fixed_point_dominant_multiplier(previous.as_ref(), previous_t, &residual);
            let t = frozen_index_relaxation(mu);
            previous = Some(residual.clone());
            previous_t = t;
            eta += t * residual[0];
        }
        assert!(
            (eta - fixed_point).abs() <= 1.0e-12,
            "the damped iteration must reach the fixed point, got {eta}"
        );
    }

    /// Run the relaxed iteration on a two-mode diagonal map, reading the
    /// multiplier either from the relaxed quotient (`corrected == true`) or from
    /// the quotient taken as `mu` outright (the estimator before this test).
    fn two_mode_relaxed_iteration(corrected: bool, passes: usize) -> (f64, Vec<f64>) {
        let multipliers = [-5.5_f64, -0.5];
        let fixed_point = [0.75_f64, -0.25];
        let map = |eta: &Array1<f64>| {
            Array1::from_shape_fn(2, |i| {
                fixed_point[i] + multipliers[i] * (eta[i] - fixed_point[i])
            })
        };
        let mut eta = array![1.0_f64, 1.0];
        let mut previous: Option<Array1<f64>> = None;
        let mut previous_t = 1.0_f64;
        let mut relaxations = Vec::with_capacity(passes);
        for _ in 0..passes {
            let residual = map(&eta) - &eta;
            let read_t = if corrected { previous_t } else { 1.0 };
            let mu = fixed_point_dominant_multiplier(previous.as_ref(), read_t, &residual);
            let t = frozen_index_relaxation(mu);
            relaxations.push(t);
            previous = Some(residual.clone());
            previous_t = t;
            eta = &eta + &residual.mapv(|value| value * t);
        }
        let error = (eta[0] - fixed_point[0])
            .abs()
            .max((eta[1] - fixed_point[1]).abs());
        (error, relaxations)
    }

    /// gam#2748, `geo_disease_matern` flexible at n=1000: a single scalar mode
    /// hides the defect, because the first damped step annihilates it exactly
    /// and every later residual is zero. With a second, milder mode present the
    /// quotient after that step is the sub-dominant mixture; read as `mu` it
    /// undoes the damping, re-excites the dominant mode, and the passes lock
    /// into `t_k·t_{k+1} = 1/(1−mu)`. Recovering `mu` from the relaxed quotient
    /// makes the same iteration converge.
    #[test]
    fn the_multiplier_is_read_through_the_relaxation_that_produced_the_residual_2748() {
        let (error, relaxations) = two_mode_relaxed_iteration(true, 60);
        assert!(
            error <= 1.0e-6,
            "the corrected iteration must converge on two modes, error={error} t={relaxations:?}"
        );
        let (control_error, control_relaxations) = two_mode_relaxed_iteration(false, 60);
        assert!(
            !(control_error <= 1.0e-2),
            "reading the relaxed quotient as mu must NOT converge here, or this test measures \
             nothing: error={control_error} t={control_relaxations:?}"
        );
        // The control is the involution the n=1000 trace showed: consecutive
        // relaxations multiply to 1/(1−mu) once the iteration is on the cycle.
        let product = control_relaxations[10] * control_relaxations[11];
        assert!(
            (product - 1.0 / 6.5).abs() <= 1.0e-2,
            "t_k·t_(k+1) must sit on the 1/(1−mu) involution, got {product}"
        );
    }

    /// gam#2748: one scalar relaxation cannot serve a `−5.5` mode and a mild
    /// mode at once (kill one, and the step that suits the other regrows it).
    /// The mixer the loop now uses annihilates several modes per pass: on a
    /// three-mode linear map it must reach the fixed point in a handful of
    /// passes, driven exactly as the loop drives it — the scalar relaxation as
    /// the fallback, the residual-norm safeguard resetting the history.
    #[test]
    fn the_frozen_index_mixer_converges_a_three_mode_map_in_a_few_passes_2748() {
        let multipliers = [-5.5_f64, -0.5, 0.7];
        let fixed_point = [0.75_f64, -0.25, 1.5];
        let map = |eta: &Array1<f64>| {
            Array1::from_shape_fn(3, |i| {
                fixed_point[i] + multipliers[i] * (eta[i] - fixed_point[i])
            })
        };
        let mut eta = array![1.0_f64, 1.0, 1.0];
        let mut mixer = FrozenIndexMixer::new(60).expect("a positive pass budget");
        let mut previous: Option<Array1<f64>> = None;
        let mut previous_t = 1.0_f64;
        let mut last_measured = f64::NAN;
        let mut kinds = Vec::new();
        let mut converged_at = None;
        for pass in 0..60 {
            let residual = map(&eta) - &eta;
            let error = (0..3)
                .map(|i| (eta[i] - fixed_point[i]).abs())
                .fold(0.0_f64, f64::max);
            if error <= 1.0e-10 {
                converged_at = Some(pass);
                break;
            }
            let mu = fixed_point_dominant_multiplier(previous.as_ref(), previous_t, &residual);
            if mu.is_finite() {
                last_measured = mu;
            }
            let t = frozen_index_relaxation(if mu.is_finite() { mu } else { last_measured });
            let (advance, kind, _) = mixer
                .advance(residual.as_slice().expect("contiguous"), t)
                .expect("finite residual");
            kinds.push(kind);
            previous = Some(residual.clone());
            previous_t = match kind {
                FrozenIndexAdvance::Relaxed => t,
                FrozenIndexAdvance::Anderson { .. } => f64::NAN,
            };
            eta = &eta + &Array1::from_iter(advance);
        }
        let converged_at = converged_at.unwrap_or_else(|| {
            panic!("the mixed iteration must converge on three modes; kinds={kinds:?}")
        });
        assert!(
            converged_at <= 10,
            "three linear modes must be annihilated within a few multisecant passes, took \
             {converged_at}: kinds={kinds:?}"
        );
        assert!(
            kinds
                .iter()
                .any(|kind| matches!(kind, FrozenIndexAdvance::Anderson { .. })),
            "the accelerated step must actually have been taken: kinds={kinds:?}"
        );
    }
}

#[cfg(test)]
mod binomial_mean_wiggle_dealias_metric_tests {
    use super::*;

    /// A deterministic mean block, warp basis and curvature vector with the
    /// shape the failing cells have: a curvature spanning six orders of
    /// magnitude, because that is the whole reason the metric matters
    /// (`m₂ = w·μ(1−μ)` on a logit near saturation).
    fn fixture() -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        let n = 40;
        let mut x = Array2::<f64>::zeros((n, 4));
        let mut b = Array2::<f64>::zeros((n, 3));
        let mut curvature = Array1::<f64>::zeros(n);
        for row in 0..n {
            let t = row as f64 / (n as f64 - 1.0);
            let eta = -6.0 + 12.0 * t;
            x[[row, 0]] = 1.0;
            x[[row, 1]] = eta;
            x[[row, 2]] = (1.7 * eta).sin();
            x[[row, 3]] = (0.9 * eta).cos();
            // A monotone-ish warp basis: partial sums of positive bumps, which
            // is the shape an I-spline has.
            b[[row, 0]] = t;
            b[[row, 1]] = t * t;
            b[[row, 2]] = (1.0 - (-3.0 * t).exp()) / (1.0 - (-3.0_f64).exp());
            let mu = 1.0 / (1.0 + (-eta).exp());
            curvature[row] = mu * (1.0 - mu);
        }
        (x, b, curvature)
    }

    fn cross_block(x: &Array2<f64>, curvature: &Array1<f64>, bda: &Array2<f64>) -> Array2<f64> {
        let mut weighted = bda.clone();
        for row in 0..bda.nrows() {
            let weight = curvature[row];
            weighted.row_mut(row).map_inplace(|value| *value *= weight);
        }
        x.t().dot(&weighted)
    }

    fn max_abs(matrix: &Array2<f64>) -> f64 {
        matrix.iter().map(|value| value.abs()).fold(0.0, f64::max)
    }

    /// The defining identity: the de-aliased warp block is orthogonal to the
    /// mean block in the metric the frozen joint solve's Hessian is assembled
    /// in, so the Hessian's cross block is zero.
    #[test]
    fn the_dealiased_warp_is_curvature_orthogonal_to_the_mean_block() {
        let (x, b, curvature) = fixture();
        let (_, bda) = dealias_warp_against_mean_block(&x, &b, &curvature).expect("de-alias");
        let cross = cross_block(&x, &curvature, &bda);
        // Scale the bar by what a single unweighted cross block would have
        // been, so this asserts "zero relative to the quantity being cancelled"
        // and cannot pass by everything being small.
        let reference = max_abs(&cross_block(&x, &curvature, &b));
        assert!(reference > 1.0e-3, "the fixture must have a real alias to cancel, got {reference}");
        assert!(
            max_abs(&cross) <= 1.0e-12 * reference,
            "X' W B_perp must vanish: {} against reference {reference}",
            max_abs(&cross),
        );
    }

    /// The negative control that makes the point of the fix falsifiable: the
    /// EUCLIDEAN residualization — what this site used before #2748 — leaves
    /// the solve's own cross block at the same order as the raw alias. It
    /// removes nothing the Hessian can see.
    #[test]
    fn the_euclidean_residualization_leaves_the_solve_coupled() {
        let (x, b, curvature) = fixture();
        let flat = Array1::<f64>::ones(curvature.len());
        let (_, euclidean) = dealias_warp_against_mean_block(&x, &b, &flat).expect("de-alias");
        let euclidean_cross = max_abs(&cross_block(&x, &curvature, &euclidean));
        let reference = max_abs(&cross_block(&x, &curvature, &b));
        assert!(
            euclidean_cross >= 0.05 * reference,
            "the Euclidean residualization is supposed to leave the curvature cross block \
             standing; if this ever becomes small the fixture stopped exercising the defect: \
             {euclidean_cross} against {reference}",
        );
        // And it does cancel the metric it WAS taken in — so the two are
        // genuinely different projections, not one of them being broken.
        let flat_cross = max_abs(&cross_block(&x, &flat, &euclidean));
        let flat_reference = max_abs(&cross_block(&x, &flat, &b));
        assert!(
            flat_cross <= 1.0e-12 * flat_reference,
            "X' B_perp must vanish for the flat metric: {flat_cross} against {flat_reference}",
        );
    }

    /// A constant curvature is the Euclidean projection, exactly. Any fit whose
    /// rows are equally informative therefore does not move by this change —
    /// which is what makes the change a repair of the ill-conditioned case
    /// rather than a new model everywhere.
    #[test]
    fn a_constant_metric_reproduces_the_euclidean_projection() {
        let (x, b, _) = fixture();
        let flat = Array1::<f64>::ones(x.nrows());
        let scaled = Array1::<f64>::from_elem(x.nrows(), 7.5);
        let (alias_flat, _) = dealias_warp_against_mean_block(&x, &b, &flat).expect("flat");
        let (alias_scaled, _) = dealias_warp_against_mean_block(&x, &b, &scaled).expect("scaled");
        let scale = max_abs(&alias_flat).max(1.0);
        assert!(
            max_abs(&(&alias_flat - &alias_scaled)) <= 1.0e-12 * scale,
            "a metric proportional to the identity is the identity's projection",
        );
    }

    /// Rows the likelihood cannot see get weight zero, and the identity still
    /// holds: those rows keep their raw warp in the residual instead of
    /// spending the projection's budget on them.
    #[test]
    fn zero_curvature_rows_leave_their_warp_in_the_residual() {
        let (x, b, mut curvature) = fixture();
        for row in 0..12 {
            curvature[row] = 0.0;
        }
        let (_, bda) = dealias_warp_against_mean_block(&x, &b, &curvature).expect("de-alias");
        assert!(bda.iter().all(|value| value.is_finite()));
        let cross = max_abs(&cross_block(&x, &curvature, &bda));
        let reference = max_abs(&cross_block(&x, &curvature, &b));
        assert!(
            cross <= 1.0e-12 * reference,
            "X' W B_perp must vanish with a rank-degraded metric too: {cross} against {reference}",
        );
    }

    /// A rank-deficient mean block: the pseudo-inverse drops only its own null
    /// directions, and the retained ones are still exactly cancelled.
    #[test]
    fn a_rank_deficient_mean_block_still_cancels_its_retained_range() {
        let (x, b, curvature) = fixture();
        let mut duplicated = Array2::<f64>::zeros((x.nrows(), x.ncols() + 1));
        duplicated.slice_mut(s![.., ..x.ncols()]).assign(&x);
        // An exact copy of column 1, so `X'WX` is singular by construction.
        let copied = x.column(1).to_owned();
        duplicated.column_mut(x.ncols()).assign(&copied);
        let (_, bda) =
            dealias_warp_against_mean_block(&duplicated, &b, &curvature).expect("de-alias");
        assert!(bda.iter().all(|value| value.is_finite()));
        let cross = max_abs(&cross_block(&duplicated, &curvature, &bda));
        let reference = max_abs(&cross_block(&duplicated, &curvature, &b));
        assert!(
            cross <= 1.0e-10 * reference,
            "a singular X'WX must not stop the retained range from cancelling: {cross} against \
             {reference}",
        );
    }

    /// The mechanism gate, stated on the quantity the outer loop actually
    /// rides on rather than on the cross block that implies it (#2748).
    ///
    /// Freezing the warp basis at `η̂` and perturbing by `δ` moves the design by
    /// `(I−P)·diag(s)·δ` in `q`, and the refit answers with
    /// `Δη = −H·(I−P)·diag(s)·δ` for this fit's own penalized hat matrix
    /// `H = X(XᵀWX + S)⁻¹XᵀW`. That product IS the leading term of the
    /// frozen-index map's derivative. With `P` the `W`-projection it is zero
    /// for ANY penalty `S`; with `P` Euclidean it is not, and the assertion
    /// below is what turns "the metric is the right one" into a measurement.
    #[test]
    fn the_curvature_metric_annihilates_the_outer_maps_leading_term() {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;

        let (x, b, curvature) = fixture();
        let n = x.nrows();
        let p = x.ncols();
        // A warp slope large enough that the undamped map would diverge if the
        // term below survived: `max_i s_i` well above one.
        let slope = Array1::from_shape_fn(n, |row| 0.4 + 3.0 * (row as f64 / n as f64));
        // An arbitrary but deterministic freezing-point perturbation.
        let delta = Array1::from_shape_fn(n, |row| ((row as f64) * 0.37).sin());
        let forcing = Array1::from_shape_fn(n, |row| slope[row] * delta[row]);

        let mut weighted_x = x.clone();
        for row in 0..n {
            let weight = curvature[row];
            weighted_x.row_mut(row).map_inplace(|value| *value *= weight);
        }
        // A non-trivial ridge penalty, to exercise the "for any S" half of the
        // claim: the cancellation must not depend on the smoothing.
        let mut normal = x.t().dot(&weighted_x);
        for index in 0..p {
            normal[[index, index]] += 0.37 * (index as f64 + 1.0);
        }
        let (normal_values, normal_vectors) = normal.eigh(Side::Lower).expect("penalized normal");
        let mut normal_inverse = Array2::<f64>::zeros((p, p));
        for k in 0..p {
            let scale = 1.0 / normal_values[k];
            let uk = normal_vectors.column(k);
            for i in 0..p {
                for j in 0..p {
                    normal_inverse[[i, j]] += scale * uk[i] * uk[j];
                }
            }
        }

        // `H·r` for a residual `r`, through the same normal equations the fit
        // solves.
        let hat_apply = |residual: &Array1<f64>| -> Array1<f64> {
            x.dot(&normal_inverse.dot(&weighted_x.t().dot(residual)))
        };

        let forcing_as_basis = forcing
            .clone()
            .into_shape_with_order((n, 1))
            .expect("column");
        let (_, metric_residual) =
            dealias_warp_against_mean_block(&x, &forcing_as_basis, &curvature).expect("metric");
        let flat = Array1::<f64>::ones(n);
        let (_, euclidean_residual) =
            dealias_warp_against_mean_block(&x, &forcing_as_basis, &flat).expect("euclidean");

        let metric_response = hat_apply(&metric_residual.column(0).to_owned());
        let euclidean_response = hat_apply(&euclidean_residual.column(0).to_owned());
        let reference = hat_apply(&forcing);
        let norm = |v: &Array1<f64>| v.dot(v).sqrt();

        assert!(
            norm(&reference) > 1.0e-2,
            "the fixture must have a leading term to cancel, got {}",
            norm(&reference)
        );
        assert!(
            norm(&metric_response) <= 1.0e-12 * norm(&reference),
            "H(I-P_W) must annihilate the forcing for any penalty: {} against {}",
            norm(&metric_response),
            norm(&reference),
        );
        assert!(
            norm(&euclidean_response) >= 0.05 * norm(&reference),
            "the Euclidean projector is supposed to leave the outer map's leading term \
             standing; if this ever becomes small the fixture stopped exercising the defect: \
             {} against {}",
            norm(&euclidean_response),
            norm(&reference),
        );
        // And with the leading term gone, the whole B block inherits it: the
        // same statement for the real warp basis rather than for one column.
        let (_, bda) = dealias_warp_against_mean_block(&x, &b, &curvature).expect("de-alias");
        for column in 0..bda.ncols() {
            let response = norm(&hat_apply(&bda.column(column).to_owned()));
            // Against what the mean block DID absorb from that column before
            // residualization — the quantity being cancelled. Grading against
            // the residual column's own norm would be grading against a
            // different, and smaller, thing.
            let absorbed = norm(&hat_apply(&b.column(column).to_owned()));
            assert!(
                absorbed > 1.0e-3,
                "column {column} must have something to cancel, got {absorbed}"
            );
            assert!(
                response <= 1.0e-12 * absorbed,
                "column {column}: H B_perp must vanish, got {response} against {absorbed}",
            );
        }
    }

    /// A metric that is not a semi-inner product is refused, not silently used:
    /// a negative weight would make `X'WX` indefinite and its "projection"
    /// would not be one.
    #[test]
    fn a_negative_curvature_is_refused() {
        let (x, b, mut curvature) = fixture();
        curvature[7] = -1.0e-3;
        let error = dealias_warp_against_mean_block(&x, &b, &curvature)
            .expect_err("a negative metric weight must refuse");
        assert!(error.contains("semi-inner product"), "{error}");
    }
}

/// Fit the binomial mean link-wiggle model. The observation-space de-aliasing
/// preserves the standard I-spline coefficient coordinate, which is returned
/// for the saved-model predict runtime.
/// What a converged frozen-index binomial mean link-wiggle fit IS, named rather
/// than positional.
///
/// The third and fourth fields exist because the fit is returned in the SAVED
/// frame (`finalize_binomial_mean_wiggle_saved_frame`), where the mean
/// coefficient has absorbed `−A·β_w`: `q = X·β_saved + B(η̂)·β_w`. So neither
/// `block_states[BLOCK_ETA].eta` nor the wiggle block's eta is the frozen index
/// the warp was pinned at, and a caller that needs the CRITERION this fit
/// realizes cannot reconstruct it from the states alone.
///
/// `frozen_warp_design` is that criterion's warp block: the accepted `B⊥` the
/// fixed point converged with. Returning it means the exact-joint spatial driver
/// consumes the very matrix the baseline fit used rather than re-deriving one
/// from the saved-frame states — which is how gam#2748's first attempt at this
/// went wrong, since `wiggle_design(X·β_saved)` is the warp basis at the
/// DE-ALIASED predictor and not at `η̂`.
pub(crate) struct BinomialMeanWiggleFrozenFit {
    pub(crate) fit: UnifiedFitResult,
    pub(crate) saved_warp_beta: Option<Vec<f64>>,
    /// `β_frozen_source − β_saved` in mean coordinates, so `X·(β_saved + shift)`
    /// is the frozen index `η̂` (#2141).
    pub(crate) saved_index_shift: Option<Vec<f64>>,
    /// The accepted `B⊥ = B(η̂) − X A`, in the row order of the training data.
    pub(crate) frozen_warp_design: std::sync::Arc<Array2<f64>>,
}

pub(crate) fn fit_binomial_mean_wiggle(
    spec: BinomialMeanWiggleSpec,
    options: &BlockwiseFitOptions,
) -> Result<BinomialMeanWiggleFrozenFit, String> {
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
    gam_terms::inference::formula_dsl::require_binomial_inverse_link_supports_joint_wiggle(
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

    // ----- Frozen-basis Gauss-Newton link-warp fit (#1596) -----
    //
    // The warp basis `B(η)` is frozen at the current index `η̂` so that
    // `q = η + B(η̂)·β_w` is linear in `(β_η, β_w)` (`∂q/∂η = 1`). To keep the
    // mean block `X` full and identifiable we fit the warp through the
    // observation-space residualized design `B⊥ = (I - P_X)B(η̂)`. We re-freeze
    // at the refit `η̂` until the caller's outer convergence policy certifies the
    // fixed point.
    let x_dense: Array2<f64> = spec.eta_block.design.to_dense();
    let (pilot_beta, pilot_eta): (Array1<f64>, Array1<f64>) = {
        let pilot_beta = spec.eta_block.initial_beta.clone().ok_or_else(|| {
            "fit_binomial_mean_wiggle: eta block carries no pilot β to seed the \
             frozen-basis warp index"
                .to_string()
        })?;
        if x_dense.ncols() != pilot_beta.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "fit_binomial_mean_wiggle: eta design has {} columns but pilot β has {} \
                     coefficients",
                    x_dense.ncols(),
                    pilot_beta.len()
                ),
            }
            .into());
        }
        let mut eta = x_dense.dot(&pilot_beta);
        eta += &spec.eta_block.offset;
        (pilot_beta, eta)
    };

    // Original (full-width) warp penalties / nullspace metadata, captured before
    // `spec.wiggle_block` is consumed. The residualized block keeps the same
    // coefficient coordinate and therefore the same penalties.
    let wiggle_penalties_full = spec.wiggle_block.penalties.clone();
    let wiggle_nullspace_dims = spec.wiggle_block.nullspace_dims.clone();
    if !wiggle_nullspace_dims.is_empty()
        && wiggle_nullspace_dims.len() != wiggle_penalties_full.len()
    {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "fit_binomial_mean_wiggle: wiggle block has {} penalties but {} nullspace dimensions",
                wiggle_penalties_full.len(),
                wiggle_nullspace_dims.len()
            ),
        }
        .into());
    }
    let wiggle_log_lambdas = spec.wiggle_block.initial_log_lambdas.clone();
    let wiggle_beta_initial = spec.wiggle_block.initial_beta.clone();
    let eta_block_input = spec.eta_block.clone();
    // The custom-family solver is family-agnostic, but the returned standard
    // fit is still binomial with this resolved inverse link. Persist that
    // authority at the producer so every caller retains the response scale
    // after the joint mean/warp refit (issue #2748).
    let likelihood_family = gam_spec::inverse_link_to_binomial_spec(&spec.link_kind)
        .map_err(|err| err.to_string())?;

    let family = BinomialMeanWiggleFamily {
        y: spec.y,
        weights: spec.weights,
        link_kind: spec.link_kind,
        wiggle_knots: spec.wiggle_knots,
        wiggle_degree: spec.wiggle_degree,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        frozen_warp_design: None,
        continuation: false,
    };

    // Build the de-aliased warp block at a frozen index.  The identifiable
    // warp is the part of `B(η̂)` outside the mean column space:
    //
    //     B⊥ = (I - P_X) B = B - X A,     A = (XᵀWX)^+ XᵀWB.
    //
    // The previous implementation used `Z = null(XᵀB)` and fitted `B Z`.
    // That is too strong: when the warp basis has no coefficient combination
    // exactly orthogonal to `X` (for example a two-column flexible link beside
    // an intercept+slope mean), it drops every warp coefficient even though
    // the nonlinear columns of `B(η̂)` have a nonzero residual after projection
    // onto `X`.  Residualizing in observation space removes only the truly
    // mean-aliased component and leaves the curved, identifiable link-shape
    // signal available to the joint solve.
    //
    // The returned `A` is used after fitting: because the inner problem used
    // `Xβ + (B - XA)β_w`, while prediction reconstructs the saved warp as
    // `Xβ_saved + Bβ_w`, we save `β_saved = β - Aβ_w`.
    //
    // `W` is the curvature `∂²(−ℓ)/∂q²` of the row loss at `working_q`, the
    // composite index this fit last realized (the pilot's `η̂` on the first
    // pass, where the warp is still zero). The metric is the whole content of
    // the word "identifiable" here; see
    // [`dealias_warp_against_mean_block`] for why it is that one and not `I`.
    let build_dealiased = |frozen: &Array1<f64>,
                           working_q: &Array1<f64>,
                           beta_hint: Option<&Array1<f64>>,
                           log_lambda_hint: Option<&Array1<f64>>|
     -> Result<
        (
            ParameterBlockInput,
            Array2<f64>,
            std::sync::Arc<Array2<f64>>,
        ),
        String,
    > {
        let b_full = family.wiggle_design(frozen.view())?;
        let mut curvature = Array1::<f64>::zeros(n);
        for row in 0..n {
            let (_, m2, _) =
                family.neglog_q_derivatives(family.y[row], family.weights[row], working_q[row])?;
            curvature[row] = if m2.is_finite() && m2 > 0.0 { m2 } else { 0.0 };
        }
        let (alias, bda) = dealias_warp_against_mean_block(&x_dense, &b_full, &curvature)?;
        let max_b = b_full.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let max_resid = bda.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let resid_tol =
            1.0e3 * f64::EPSILON * (bda.nrows().max(bda.ncols()).max(1) as f64) * max_b.max(1.0);
        if max_resid <= resid_tol {
            return Err("frozen-basis warp de-aliasing left no identifiable warp \
                        direction (the mean block already spans the warp in \
                        observation space)"
                .to_string());
        }
        let penalties: Vec<crate::model_types::PenaltySpec> = wiggle_penalties_full
            .iter()
            .map(|p| {
                let s = penalty_spec_to_dense(p, b_full.ncols())?;
                Ok(crate::model_types::PenaltySpec::Dense(s))
            })
            .collect::<Result<_, String>>()?;
        let q = bda.ncols();
        let initial_beta = match beta_hint {
            Some(beta) if beta.len() == q => Some(beta.clone()),
            Some(beta) => {
                return Err(format!(
                    "frozen-basis warp warm start has {} coefficients but the realized basis has {q}",
                    beta.len()
                ));
            }
            None => Some(Array1::zeros(q)),
        };
        let block = ParameterBlockInput {
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(bda.clone())),
            offset: Array1::zeros(frozen.len()),
            penalties,
            nullspace_dims: wiggle_nullspace_dims.clone(),
            initial_log_lambdas: log_lambda_hint
                .cloned()
                .or_else(|| wiggle_log_lambdas.clone()),
            initial_beta,
        };
        Ok((block, alias, std::sync::Arc::new(bda)))
    };

    // Outer Gauss-Newton / backfitting loop over the frozen warp index. The
    // smoothing parameters remain continuously REML/LAML-optimized at every
    // step. A fit is returned only after the frozen index is a certified fixed
    // point; exhaustion returns non-convergence evidence instead of minting a
    // fit from the last iterate. The exact coefficient source used to build the
    // accepted frozen basis is retained for prediction (#2141).
    if options.outer_max_iter == 0 || !options.outer_tol.is_finite() || options.outer_tol <= 0.0 {
        return Err(GamlssError::InvalidInput {
            reason: format!(
                "fit_binomial_mean_wiggle requires positive outer convergence policy; outer_max_iter={}, outer_tol={}",
                options.outer_max_iter, options.outer_tol
            ),
        }
        .into());
    }
    let mut frozen_source_beta = pilot_beta;
    let mut frozen_eta = pilot_eta;
    let mut eta_block_warm = eta_block_input.clone();
    let mut wiggle_beta_warm = wiggle_beta_initial;
    let mut wiggle_log_lambda_warm = wiggle_log_lambdas.clone();
    let mut converged: Option<(
        UnifiedFitResult,
        Array2<f64>,
        Array1<f64>,
        std::sync::Arc<Array2<f64>>,
    )> = None;
    let mut last_delta = f64::INFINITY;
    let mut last_scale = 1.0_f64;
    // #2748 instrumentation. The frozen-basis outer loop is a fixed-point
    // iteration on the warp index, and the quantity that decides whether it is
    // a contraction is the warp SLOPE `s = B'(eta)*beta_w` — the chain term the
    // inner solve drops on purpose (`BinomialMeanWiggleFamily::frozen_warp_design`
    // doc). `previous_step` carries the last `Phi(eta_k) - eta_k` so the ratio
    // and the alternation angle of successive steps are readable: an
    // alternating step whose ratio exceeds one IS the divergence, measured
    // rather than inferred from a terminal `delta`.
    let mut previous_step: Option<Array1<f64>> = None;
    // The relaxation applied in the advance that produced the current residual:
    // the quotient of consecutive residuals measures the RELAXED Jacobian, and
    // this is what recovers the map's own multiplier from it (#2748).
    let mut previous_relaxation = 1.0_f64;
    // The dominant multiplier is a property of the map, so the last finite
    // reading serves the scalar fallback on a pass whose own quotient is not
    // interpretable (the pass after an accelerated step).
    let mut last_measured_multiplier = f64::NAN;
    let mut mixer = FrozenIndexMixer::new(options.outer_max_iter)?;
    let mut dominant_multiplier = f64::NAN;
    // The operating point the de-aliasing metric is evaluated at: the composite
    // index `q = X·β + B⊥·β_w` of the previous pass. The pilot carries no warp,
    // so on the first pass it is the pilot index itself (#2748).
    let mut working_q = frozen_eta.clone();
    for _outer in 0..options.outer_max_iter {
        let (wiggle_block, alias, bda) = build_dealiased(
            &frozen_eta,
            &working_q,
            wiggle_beta_warm.as_ref(),
            wiggle_log_lambda_warm.as_ref(),
        )?;
        let eta_penalty_count = eta_block_warm.penalties.len();
        let wiggle_penalty_count = wiggle_block.penalties.len();
        // The warp yields to the mean block when the two still share a
        // direction after residualization; see
        // [`DEALIASED_WARP_GAUGE_PRIORITY`] for why an ordering exists here at
        // all, which is what decides whether such a pair is fatal (#2748).
        let blocks = vec![
            eta_block_warm.clone().intospec("eta")?,
            wiggle_block
                .intospec_with_gauge_priority("wiggle", DEALIASED_WARP_GAUGE_PRIORITY)?,
        ];
        let mut fam = family.clone();
        // The pass's own frozen warp block, retained so the accepted pass can
        // hand it to the caller: it IS the criterion this fixed point realizes,
        // and it is not reconstructible from the saved-frame states (#2748).
        let accepted_frozen_warp_design = std::sync::Arc::clone(&bda);
        fam.frozen_warp_design = Some(bda);
        // Every pass after the first is a CONTINUATION of the previous pass's
        // solve: one seed (the warm ρ the blocks carry) and no screening
        // cascade, so the frozen-index map is single-valued. See the
        // `continuation` field for the n=1000 basin flip this removes (#2748).
        fam.continuation = _outer > 0;
        let pass_options = if _outer > 0 {
            let mut continuation_options = options.clone();
            continuation_options.screen_initial_rho = false;
            continuation_options
        } else {
            options.clone()
        };
        let fit = fit_custom_family(&fam, &blocks, &pass_options).map_err(|e| e.to_string())?;
        let mean_state = fit
            .block_states
            .get(BinomialMeanWiggleFamily::BLOCK_ETA)
            .ok_or_else(|| {
                "fit_binomial_mean_wiggle: frozen-basis refit did not expose a fitted eta block"
                    .to_string()
            })?;
        if mean_state.eta.len() != frozen_eta.len()
            || mean_state.beta.len() != frozen_source_beta.len()
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "fit_binomial_mean_wiggle: frozen-basis refit returned an incompatible eta block"
                    .to_string(),
            }
            .into());
        }
        let new_eta = mean_state.eta.clone();
        let new_source_beta = mean_state.beta.clone();
        let wiggle_state = fit
            .block_states
            .get(BinomialMeanWiggleFamily::BLOCK_WIGGLE)
            .ok_or_else(|| {
                "fit_binomial_mean_wiggle: frozen-basis refit did not expose a fitted wiggle block"
                    .to_string()
            })?;
        let new_wiggle_beta = wiggle_state.beta.clone();
        // The composite index this pass actually fitted, which is the operating
        // point the next pass's de-aliasing metric is taken at (#2748).
        working_q = &new_eta + &wiggle_state.eta;
        last_scale = frozen_eta
            .iter()
            .chain(new_eta.iter())
            .map(|value| value.abs())
            .fold(1.0_f64, f64::max);
        last_delta = new_eta
            .iter()
            .zip(frozen_eta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        // #2748: the step, its relation to the previous one, and the warp slope
        // that predicts both.
        let step = &new_eta - &frozen_eta;
        let step_norm = step.dot(&step).sqrt();
        let (step_ratio, step_cosine) = match previous_step.as_ref() {
            Some(previous) => {
                let previous_norm = previous.dot(previous).sqrt();
                if previous_norm > 0.0 && step_norm > 0.0 {
                    (
                        step_norm / previous_norm,
                        step.dot(previous) / (step_norm * previous_norm),
                    )
                } else {
                    (f64::NAN, f64::NAN)
                }
            }
            None => (f64::NAN, f64::NAN),
        };
        dominant_multiplier =
            fixed_point_dominant_multiplier(previous_step.as_ref(), previous_relaxation, &step);
        if dominant_multiplier.is_finite() {
            last_measured_multiplier = dominant_multiplier;
        }
        let relaxation = frozen_index_relaxation(if dominant_multiplier.is_finite() {
            dominant_multiplier
        } else {
            last_measured_multiplier
        });
        let warp_slope = family.wiggle_dq_dq0(frozen_eta.view(), new_wiggle_beta.view())?;
        let max_slope = warp_slope
            .iter()
            .map(|value| value - 1.0)
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_slope =
            warp_slope.iter().map(|value| value - 1.0).sum::<f64>() / warp_slope.len() as f64;
        log::info!(
            "[WIGGLE-OUTER] #2748 pass {_outer}: delta={last_delta:.6e} scale={last_scale:.6e} \
             tol={:.6e} |step|={step_norm:.6e} |step_k|/|step_k-1|={step_ratio:.6e} \
             cos(step_k, step_k-1)={step_cosine:+.6} mu_hat={dominant_multiplier:+.6e} \
             relaxation={relaxation:.6e} max_warp_slope={max_slope:.6e} \
             mean_warp_slope={mean_slope:.6e} |beta_w|_1={:.6e}",
            options.outer_tol * last_scale,
            new_wiggle_beta.iter().map(|value| value.abs()).sum::<f64>(),
        );
        previous_step = Some(step.clone());
        if last_delta <= options.outer_tol * last_scale {
            converged = Some((fit, alias, frozen_source_beta, accepted_frozen_warp_design));
            break;
        }

        let expected_log_lambdas = eta_penalty_count + wiggle_penalty_count;
        if fit.log_lambdas.len() != expected_log_lambdas {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "fit_binomial_mean_wiggle: refit returned {} log-lambdas for {expected_log_lambdas} penalties",
                    fit.log_lambdas.len()
                ),
            }
            .into());
        }
        // Multi-mode advance on the mixed state `[β_source; η]` (#2748): the
        // Anderson step when the residual history supports one, the scalar
        // relaxed step otherwise (first pass, and the pass after a safeguard
        // reset). `relaxation == 1.0` on a relaxed pass reproduces the undamped
        // iteration exactly.
        let source_width = frozen_source_beta.len();
        let residual: Vec<f64> = new_source_beta
            .iter()
            .zip(frozen_source_beta.iter())
            .map(|(new, frozen)| new - frozen)
            .chain(step.iter().copied())
            .collect();
        let (advance, advance_kind, history_reset) = mixer.advance(&residual, relaxation)?;
        log::info!(
            "[WIGGLE-OUTER] #2748 pass {_outer}: advance={advance_kind} |residual|={:.6e} \
             history_reset={history_reset}",
            mixer.last_residual_norm().unwrap_or(f64::NAN),
        );
        // The quotient of the next pass is `(1−t) + t·mu` only after a scalar
        // relaxed step; after an accelerated step it is not a reading of `mu`.
        previous_relaxation = match advance_kind {
            FrozenIndexAdvance::Relaxed => relaxation,
            FrozenIndexAdvance::Anderson { .. } => f64::NAN,
        };
        let next_source_beta =
            &frozen_source_beta + &Array1::from_iter(advance[..source_width].iter().copied());
        let next_eta = &frozen_eta + &Array1::from_iter(advance[source_width..].iter().copied());
        eta_block_warm.initial_beta = Some(next_source_beta.clone());
        eta_block_warm.initial_log_lambdas =
            Some(fit.log_lambdas.slice(s![0..eta_penalty_count]).to_owned());
        wiggle_beta_warm = Some(new_wiggle_beta);
        wiggle_log_lambda_warm = Some(
            fit.log_lambdas
                .slice(s![eta_penalty_count..expected_log_lambdas])
                .to_owned(),
        );
        frozen_source_beta = next_source_beta;
        frozen_eta = next_eta;
    }
    let converged = converged.ok_or_else(|| {
        GamlssError::NumericalFailure {
            reason: format!(
                "fit_binomial_mean_wiggle frozen-index fixed point did not converge in {} outer \
                 iterations: delta={last_delta:.3e}, scale={last_scale:.3e}, \
                 tolerance={:.3e}; the fixed-point map's measured dominant multiplier is \
                 mu={dominant_multiplier:.3e} (successive residuals satisfy \
                 d_k = ((1-t)I + tM) d_(k-1) for the relaxation t of the previous advance, so \
                 this is M's multiplier recovered from their Rayleigh quotient) and the \
                 relaxation derived from it was \
                 {:.3e}. mu >= 1 means the frozen index is a REPELLING fixed point along a \
                 monotone direction, which no positive relaxation stabilises: the composite \
                 index the warp and the mean block are competing for is not pinned by this \
                 model at these smoothing parameters. mu <= -1 with the relaxation at 1 would \
                 mean the damping was declined and is a defect here, not a modelling limit",
                options.outer_max_iter,
                options.outer_tol * last_scale,
                frozen_index_relaxation(dominant_multiplier),
            ),
        }
        .to_string()
    })?;
    let (mut fit, last_alias, frozen_source_beta, frozen_warp_design) = converged;
    fit.likelihood_scale = likelihood_family.default_scale_metadata();
    fit.likelihood_family = Some(likelihood_family);
    // Capture the mean coefficients whose linear predictor is the *frozen index*
    // `η̂` the warp basis `B(η̂)` was pinned at (#2141). The reported deviance is
    // evaluated with `q = X·β_saved + B(η̂)·β_w`, so `predict` must re-evaluate the
    // warp basis at `X·β_frozen_source` (= η̂), NOT at the de-aliased base
    // predictor `X·β_saved`. On the failing data those differ by the identifiable
    // de-alias projection `X·A·β_w`, so predict-at-`β_saved` reconstructs a
    // *different* link than the fit used. We persist the shift
    // `s = β_frozen_source − β_saved` (a mean-coordinate vector) so predict can form the
    // frozen index `X·(β_saved + s) = η̂` and reproduce the fitted `q` exactly.
    // The solver coefficient is already the standard I-spline coefficient:
    // observation-space residualization changed the design, not its coefficient
    // chart. The family imposes β_w ≥ 0 during the continuously optimized
    // constrained REML/LAML fit. Since B' is an M-spline basis with non-negative
    // values, dq/dη = 1 + B'(η)·β_w ≥ 1 for every η, including between
    // knots; no post-fit sampling or smoothing-parameter ladder is needed.
    let saved_warp_beta = fit
        .block_states
        .get(BinomialMeanWiggleFamily::BLOCK_WIGGLE)
        .map(|state| state.beta.to_vec())
        .ok_or_else(|| {
            "fit_binomial_mean_wiggle: converged fit is missing its LinkWiggle block state"
                .to_string()
        })?;
    validate_monotone_wiggle_beta_nonnegative(
        &saved_warp_beta,
        "fit_binomial_mean_wiggle saved warp",
    )?;
    finalize_binomial_mean_wiggle_saved_frame(
        &mut fit,
        &last_alias,
        &x_dense,
        &eta_block_input.offset,
    )?;
    // The frozen-index shift `s = β_frozen_source − β_saved` for the predict
    // runtime (#2141). `β_saved` is the just-de-aliased mean block; adding
    // `X·s` to the predict base predictor recovers the frozen warp index `η̂`.
    // Only meaningful when a warp actually engaged (`saved_warp_beta` present).
    let saved_mean_state = fit
        .block_states
        .get(BinomialMeanWiggleFamily::BLOCK_ETA)
        .ok_or_else(|| {
            "fit_binomial_mean_wiggle: finalized fit is missing its Mean block state".to_string()
        })?;
    if frozen_source_beta.len() != saved_mean_state.beta.len() {
        return Err(format!(
            "fit_binomial_mean_wiggle: frozen-index source has {} coefficients, but saved Mean block has {}",
            frozen_source_beta.len(),
            saved_mean_state.beta.len(),
        ));
    }
    let saved_index_shift = Some((&frozen_source_beta - &saved_mean_state.beta).to_vec());
    Ok(BinomialMeanWiggleFrozenFit {
        fit,
        saved_warp_beta: Some(saved_warp_beta),
        saved_index_shift,
        frozen_warp_design,
    })
}

/// Densify a wiggle-block penalty spec to its full `p×p` matrix for the
/// observation-space de-aliasing path (#1596). The link-warp block carries only
/// `Dense`/`DenseWithMean` difference (and optional ridge) penalties.
fn penalty_spec_to_dense(
    spec: &crate::model_types::PenaltySpec,
    p: usize,
) -> Result<Array2<f64>, String> {
    use crate::model_types::PenaltySpec;
    match spec {
        PenaltySpec::Dense(m) | PenaltySpec::DenseWithMean { matrix: m, .. } => {
            if m.nrows() != p || m.ncols() != p {
                return Err(format!(
                    "frozen-basis warp penalty must be {p}x{p}, got {}x{}",
                    m.nrows(),
                    m.ncols()
                ));
            }
            Ok(m.clone())
        }
        PenaltySpec::Block {
            local, col_range, ..
        } => {
            let mut full = Array2::<f64>::zeros((p, p));
            if col_range.end > p || local.nrows() != col_range.len() {
                return Err("frozen-basis warp penalty block range out of bounds".to_string());
            }
            full.slice_mut(s![col_range.clone(), col_range.clone()])
                .assign(local);
            Ok(full)
        }
    }
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
            .and_then(|blocks| {
                if blocks.is_empty() {
                    Err("analytic psi derivative construction produced no parameter blocks"
                        .to_string())
                } else {
                    Ok(())
                }
            })
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
        && gam_terms::smooth::all_spatial_terms_kappa_fixed(&mean_bootspec)
        && gam_terms::smooth::all_spatial_terms_kappa_fixed(&noise_bootspec)
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
            )
            .map_err(|error| error.to_string())?;
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
                        crate::custom_family::CustomFamily::outer_derivative_policy(
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
                            crate::custom_family::ExactOuterDerivativeOrder::Second
                        } else {
                            crate::custom_family::ExactOuterDerivativeOrder::First
                        };
                        crate::custom_family::OuterDerivativePolicy {
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
                        crate::custom_family::ExactOuterDerivativeOrder::First;
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
                |theta,
                 specs: &[TermCollectionSpec],
                 designs: &[TermCollectionDesign],
                 provenance| {
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
                            let (certified_outer, mode) = match provenance {
                                SpatialFitProvenance::Certified { outer, mode } => (outer, mode),
                                SpatialFitProvenance::NoOuterOptimization => {
                                    return Err(
                                        "active GAMLSS spatial optimization returned no certified outer provenance"
                                            .to_string(),
                                    );
                                }
                            };
                            let exact_options =
                                crate::outer_subsample::exact_outer_options_for_row_set(
                                    options,
                                    &crate::row_kernel::RowSet::All,
                                );
                            fit_custom_family_fixed_log_lambdas_from_owned_mode(
                                &family,
                                &blocks,
                                &exact_options,
                                mode,
                                theta,
                                certified_outer,
                            ).map_err(|error| error.to_string())?
                        } else {
                            fit_custom_family(&family, &blocks, options).map_err(|error| error.to_string())?
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
                 row_set: &crate::row_kernel::RowSet,
                 _| {
                    use gam_problem::EvalMode;
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
                    let hyper_layout = crate::custom_family::CustomFamilyHyperLayout::new(
                        psiderivative_blocks,
                        Vec::new(),
                        theta.slice(s![joint_setup.rho_dim()..]).to_owned(),
                    )?;
                    let warm_start = hyper_warm_start_cell.borrow().clone();
                    // Forward the κ-staging row set to the family by installing it
                    // on the canonical `outer_score_subsample` option. Inner-PIRLS
                    // and final covariance still run on full data (the per-row
                    // weight is consulted only by outer-only paths inside the
                    // family). When the staging schedule is full-data the option
                    // stays `None` and the call is equivalent to the prior path.
                    let eval_options =
                        crate::outer_subsample::exact_outer_options_for_row_set(options, row_set);
                    let owned = evaluate_custom_family_joint_hyper_owned(
                        &family,
                        &blocks,
                        &eval_options,
                        &rho,
                        &hyper_layout,
                        warm_start.as_ref(),
                        eval_mode,
                    ).map_err(|error| error.to_string())?;
                    *hyper_warm_start_cell.borrow_mut() = Some(owned.result.warm_start.clone());
                    if !owned.result.inner_converged {
                        return Err(
                            "exact two-block spatial inner solve did not converge".to_string(),
                        );
                    }
                    if matches!(eval_mode, EvalMode::ValueGradientHessian)
                        && !owned.result.outer_hessian.is_analytic()
                    {
                        return Err(
                            "exact two-block spatial objective requires a full joint [rho, psi] hessian"
                            .to_string(),
                        );
                    }
                    Ok(ExactJointEvaluation {
                        objective: owned.result.objective,
                        gradient: owned.result.gradient,
                        hessian: owned.result.outer_hessian,
                        mode: owned.mode,
                    })
                },
                |theta,
                 specs: &[TermCollectionSpec],
                 designs: &[TermCollectionDesign],
                 row_set: &crate::row_kernel::RowSet| {
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
                    let hyper_layout = crate::custom_family::CustomFamilyHyperLayout::new(
                        psiderivative_blocks,
                        Vec::new(),
                        theta.slice(s![joint_setup.rho_dim()..]).to_owned(),
                    )?;
                    let warm_start = hyper_warm_start_cell.borrow().clone();
                    let eval_options =
                        crate::outer_subsample::exact_outer_options_for_row_set(options, row_set);
                    let owned = evaluate_custom_family_joint_hyper_efs_owned(
                        &family,
                        &blocks,
                        &eval_options,
                        &rho,
                        &hyper_layout,
                        warm_start.as_ref(),
                    ).map_err(|error| error.to_string())?;
                    *hyper_warm_start_cell.borrow_mut() = Some(owned.result.warm_start.clone());
                    if !owned.result.inner_converged {
                        return Err(
                            "exact two-block spatial EFS inner solve did not converge".to_string(),
                        );
                    }
                    Ok(ExactJointEfsEvaluation {
                        evaluation: owned.result.efs_eval,
                        mode: owned.mode,
                    })
                },
                |_: &Array1<f64>| Ok(gam_solve::rho_optimizer::SeedOutcome::NoSlot),
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

    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }

    fn exact_spatial_seed_risk_profile(&self) -> crate::seeding::SeedRiskProfile {
        crate::seeding::SeedRiskProfile::GaussianLocationScale
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
            policy: gam_runtime::resource::ResourcePolicy::default_library(),
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

    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }

    fn exact_spatial_seed_risk_profile(&self) -> crate::seeding::SeedRiskProfile {
        crate::seeding::SeedRiskProfile::GaussianLocationScale
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
            policy: gam_runtime::resource::ResourcePolicy::default_library(),
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
            policy: gam_runtime::resource::ResourcePolicy::default_library(),
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
            policy: gam_runtime::resource::ResourcePolicy::default_library(),
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
    let q_seed = pilot_design
        .apply(pilot_fit.beta.view())
        .map_err(|error| error.to_string())?;
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
    // The joint `[rho, psi]` box rule, shared with the standard spatial route
    // (`spatial_optimization.rs` is `include!`d into `drivers`, so it lives at
    // the module root).
    use crate::fit_orchestration::drivers::{JOINT_RHO_BOUND, joint_rho_search_box};

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
        let BinomialMeanWiggleFrozenFit {
            fit,
            saved_warp_beta,
            saved_index_shift,
            ..
        } = fit_binomial_mean_wiggle(
            BinomialMeanWiggleSpec {
                y: y.clone(),
                weights: weights.clone(),
                link_kind,
                wiggle_knots: wiggle_knots.clone(),
                wiggle_degree,
                eta_block: ParameterBlockInput {
                    design: pilot_design.design.clone(),
                    offset: pilot_design.affine_offset.clone(),
                    penalties: pilot_design
                        .penalties
                        .iter()
                        .map(crate::model_types::PenaltySpec::from_blockwise_ref)
                        .collect(),
                    nullspace_dims: vec![],
                    initial_log_lambdas: Some(fitted_log_lambdas(
                        &pilot_fit.lambdas,
                        "binomial mean-wiggle pilot lambda",
                    )?),
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
            saved_warp_beta,
            saved_index_shift,
        });
    }

    let dims_per_term = spatial_dims_per_term(pilot_spec, &spatial_terms);
    let log_kappa0 =
        SpatialLogKappaCoords::from_length_scales_aniso(pilot_spec, &spatial_terms, kappa_options)
            .reseed_from_data(data, pilot_spec, &spatial_terms, kappa_options)
            .map_err(|error| error.to_string())?;
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        pilot_spec,
        &spatial_terms,
        &dims_per_term,
        kappa_options,
    )
    .map_err(|error| error.to_string())?;
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        pilot_spec,
        &spatial_terms,
        &dims_per_term,
        kappa_options,
    )
    .map_err(|error| error.to_string())?;
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
    let baseline = fit_binomial_mean_wiggle(
        BinomialMeanWiggleSpec {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: link_kind.clone(),
            wiggle_knots: wiggle_knots.clone(),
            wiggle_degree,
            eta_block: ParameterBlockInput {
                design: baseline_design.design.clone(),
                offset: baseline_design.affine_offset.clone(),
                penalties: baseline_design
                    .penalties
                    .iter()
                    .map(crate::model_types::PenaltySpec::from_blockwise_ref)
                    .collect(),
                nullspace_dims: vec![],
                initial_log_lambdas: Some(fitted_log_lambdas(
                    &pilot_fit.lambdas,
                    "binomial mean-wiggle pilot lambda",
                )?),
                initial_beta: Some(pilot_fit.beta.clone()),
            },
            wiggle_block: wiggle_block.clone(),
        },
        options,
    )?;
    let baseline_fit = baseline.fit;
    let baseline_log_lambdas = fitted_log_lambdas(
        &baseline_fit.lambdas,
        "binomial mean-wiggle baseline lambda",
    )?;
    if baseline_log_lambdas.len() != rho_dim {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "baseline binomial mean-wiggle fit returned {} log-lambdas, expected {rho_dim}",
                baseline_log_lambdas.len()
            ),
        }
        .into());
    }
    // THE WARM START MUST BE IN THE FITTING FRAME, NOT THE SAVED FRAME (#2748).
    //
    // `fit_binomial_mean_wiggle` returns `beta_saved`, which satisfies
    // `q = X*beta_saved + B(eta_hat)*beta_w`: the mean coefficient has absorbed
    // `-A*beta_w`. Every consumer below seeds a frozen-index construction with
    // it, and `fit_binomial_mean_wiggle`'s own entry computes its FIRST frozen
    // index as `X*initial_beta + offset`. Seeded with `beta_saved` that index is
    // the DE-ALIASED predictor, so the warp basis is built at the one index the
    // mean block already spans — and the refit refuses its own warm start with
    // "no identifiable warp direction". MEASURED on `papuan_oce4_matern_k6`: the
    // outer search certified `theta*` (`|Pg| = 4.445e-3` against `1.968e-2`,
    // `hessian_psd=yes`) and the refit at that same `theta*` then died on exactly
    // that message.
    //
    // `beta_frozen_source = beta_saved + saved_index_shift` is the fitting-frame
    // coefficient whose predictor IS `eta_hat` (#2141 defines the shift as
    // `beta_frozen_source - beta_saved` for precisely this reconstruction). It is
    // also the right frame for the outer evaluator, whose wiggle block is
    // `B_perp`: there the model is `q = X*beta + B_perp*beta_w`, which is the
    // fitting frame and not the saved one.
    let baseline_saved_eta_beta = baseline_fit
        .block_states
        .get(BinomialMeanWiggleFamily::BLOCK_ETA)
        .ok_or_else(|| "baseline binomial mean-wiggle fit missing eta block".to_string())?
        .beta
        .clone();
    let baseline_eta_beta = match baseline.saved_index_shift.as_ref() {
        Some(shift) if shift.len() == baseline_saved_eta_beta.len() => {
            &baseline_saved_eta_beta + &Array1::from(shift.clone())
        }
        // No warp engaged (`saved_index_shift` is only produced alongside a
        // fitted warp), so the two frames coincide.
        None => baseline_saved_eta_beta,
        Some(shift) => {
            return Err(format!(
                "baseline binomial mean-wiggle fit reported a {}-coefficient frozen-index                  shift for a {}-coefficient mean block",
                shift.len(),
                baseline_saved_eta_beta.len(),
            ));
        }
    };
    let baseline_wiggle_beta = Some(
        baseline_fit
            .block_states
            .get(BinomialMeanWiggleFamily::BLOCK_WIGGLE)
            .ok_or_else(|| "baseline binomial mean-wiggle fit missing wiggle block".to_string())?
            .beta
            .clone(),
    );
    // THE OUTER SEARCH MUST OPTIMIZE THE CRITERION THE FIT IT GRADES REALIZES
    // (#2748).
    //
    // This driver used to hand the outer `[rho, psi]` evaluator the RAW warp
    // basis with `frozen_warp_design = None`, i.e. the DYNAMIC `B(eta)` rebuilt
    // at the moving index — while the baseline fit above and the final refit
    // below are both `fit_binomial_mean_wiggle`, whose whole construction is the
    // FROZEN, de-aliased `B_perp = B(eta_hat) - X A`. Three things follow, and
    // all three were measured on `papuan_oce4_matern_k6`'s `_flexible` cell:
    //
    // * **The two are not the same function.** At the identical
    //   `rho = [-3.996, -8.097, -1.575, -4.136, -2.873, -10.0]` the frozen path
    //   certifies in three cycles at `cost = 1.967e3`; the dynamic path reports
    //   `cost = 7.957e8` at `|beta|inf = 1.592e5`. The search was choosing
    //   `theta*` for a criterion the refit never evaluates.
    //
    // * **The frozen one is CONVEX and the dynamic one is not.** With `B` frozen,
    //   `q = X*beta + B_perp*beta_w` is linear in both blocks, so the joint
    //   Hessian is `[X B_perp]' diag(m2) [X B_perp] + S` with `m2 = w*mu*(1-mu) >= 0`
    //   — positive semidefinite by construction. The dynamic arm carries the
    //   `d2B/deta2` chain terms and is genuinely indefinite: every seed of this
    //   cell reported `resolvable_negative_curvature=true` and ground its full
    //   1200-cycle budget at `step_inf = 3.4e4` with the residual at `9.9e2`
    //   against a `9.9e-4` tolerance.
    //
    // * **The declared psi-derivative layout is only complete for the frozen
    //   arm.** `CustomFamilyHyperLayout::new(vec![eta_derivs, Vec::new()], ..)`
    //   states that the wiggle block's design does not move with psi. Frozen, that
    //   is exactly true — `B_perp` is a fixed matrix. Dynamic, it is false:
    //   `B(eta)` with `eta = X(psi)*beta` carries `dB/dpsi = B'(eta)*(dX/dpsi)*beta`,
    //   a direct design dependence nothing declared, so the analytic outer
    //   gradient was missing a term.
    //
    // The index is frozen ONCE, at the baseline fixed point, and not per outer
    // proposal. A per-proposal freeze would make `B_perp` a function of psi and
    // reintroduce exactly the undeclared psi dependence of the third point; the
    // baseline's own converged index is the only psi-independent choice, and it
    // is the point the search starts from. The final refit re-freezes iteratively
    // from the pilot, so the approximation lives entirely in which nuisance
    // profile the search minimises — not in the fit that is minted.
    // The criterion the baseline fixed point realized, taken from the fit rather
    // than re-derived: `fit_binomial_mean_wiggle` returns the ACCEPTED `B_perp`
    // (`BinomialMeanWiggleFrozenFit::frozen_warp_design`). Re-deriving it here
    // was the first attempt at this and it was wrong for a reason worth keeping
    // written down: the returned fit is in the SAVED frame, so
    // `block_states[BLOCK_ETA].eta` is `X*beta_saved = eta_hat - X*A*beta_w` and
    // NOT the frozen index. `wiggle_design` at that predictor is the warp basis
    // at the de-aliased index, whose residual against the mean block measured
    // `<= 1.33e-9` on `papuan_oce4_matern_k6` — the driver refused its own fit
    // with "no identifiable warp direction" while the loop that produced it had
    // just converged. One matrix, one owner (#2748).
    let frozen_warp_basis = baseline.frozen_warp_design;
    let theta_dim = rho_dim + log_kappa0.len();
    let mut theta0 = Array1::<f64>::zeros(theta_dim);
    theta0
        .slice_mut(s![0..rho_dim])
        .assign(&baseline_log_lambdas);
    theta0
        .slice_mut(s![rho_dim..theta_dim])
        .assign(log_kappa0.as_array());

    // The ρ half of the joint box, through the SAME rule the standard joint
    // `[ρ, ψ]` spatial route uses (#2454, corrected by #2760): a `±12` PRIOR
    // that falls back per coordinate to the engine's own `±RHO_BOUND` the
    // moment the incumbent it will be GRADED against is not strictly inside it.
    //
    // This site used to write a flat local `RHO_BOUND = 12.0` across every ρ
    // coordinate while grading the joint minimum against `baseline_fit` — whose
    // λ̂ was found by `fit_binomial_mean_wiggle`'s own outer search over the
    // engine's `±30`. When that λ̂ lands outside `±12`, `min` is taken over a set
    // that does not contain the point it is compared with, the seed is pasted
    // onto the wall, and the coordinate is an ACTIVE constraint from iteration
    // zero: `joint_rho_search_box`'s own doc states the mechanism and #2760
    // measured it. MEASURED here too, on `papuan_oce4_matern_k6` at
    // `5f6bddb16`: all six ρ terminate at `11.9971 … 11.99998` against this
    // `±12`, `|Pg| = |g| = 2.163` (so the projection removed NOTHING — every
    // railed coordinate still has feasible descent), and the search gives up
    // after TWO iterations on a cost stall.
    let (rho_lower, rho_upper) = joint_rho_search_box(baseline_log_lambdas.view(), JOINT_RHO_BOUND);
    let widened: Vec<usize> = (0..rho_dim)
        .filter(|&k| rho_lower[k] < -JOINT_RHO_BOUND || rho_upper[k] > JOINT_RHO_BOUND)
        .collect();
    if !widened.is_empty() {
        log::info!(
            "[binomial-mean-wiggle] joint rho box fell back to the engine's own \
             +/-RHO_BOUND on coordinate(s) {widened:?}: the baseline fit's own lambda-hat is \
             not strictly inside the joint +/-{JOINT_RHO_BOUND} prior, so the prior is \
             falsified there and the search region becomes the one the graded incumbent was \
             found in (gam#2760). seed={:?} box=[{:?}, {:?}]",
            baseline_log_lambdas.to_vec(),
            rho_lower.to_vec(),
            rho_upper.to_vec(),
        );
    }
    let mut lower = Array1::<f64>::zeros(theta_dim);
    let mut upper = Array1::<f64>::zeros(theta_dim);
    lower.slice_mut(s![0..rho_dim]).assign(&rho_lower);
    upper.slice_mut(s![0..rho_dim]).assign(&rho_upper);
    lower
        .slice_mut(s![rho_dim..theta_dim])
        .assign(log_kappa_lower.as_array());
    upper
        .slice_mut(s![rho_dim..theta_dim])
        .assign(log_kappa_upper.as_array());

    let pilot_spec_cloned = pilot_spec.clone();
    let pilot_beta = baseline_eta_beta;
    // The block the outer evaluator solves is the frozen de-aliased warp, so its
    // DESIGN is that matrix and not the raw basis (#2748). `block_geometry`'s
    // frozen arm serves `frozen_warp_design` and asserts it matches
    // `spec.design.ncols()`; `B_perp` has the same width as `B` by construction
    // (`B - X A`), so the two agree and the family's own width check is what
    // proves it.
    let wiggle_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        frozen_warp_basis.as_ref().clone(),
    ));
    let wiggle_offset = Array1::<f64>::zeros(frozen_warp_basis.nrows());
    let wiggle_penalties: Vec<crate::model_types::PenaltySpec> = wiggle_block
        .penalties
        .iter()
        .map(|penalty| {
            penalty_spec_to_dense(penalty, frozen_warp_basis.ncols())
                .map(crate::model_types::PenaltySpec::Dense)
        })
        .collect::<Result<_, String>>()?;
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
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        // The FROZEN, de-aliased basis the baseline fixed point realized — the
        // same object `fit_binomial_mean_wiggle` fits and the same object the
        // refit below re-derives. See `frozen_warp_basis` above for the three
        // measured reasons this is not the dynamic `B(eta)` it used to be
        // (#2748): the two are different functions, only this one is convex, and
        // only this one makes the declared ψ-derivative layout complete.
        frozen_warp_design: Some(std::sync::Arc::clone(&frozen_warp_basis)),
        continuation: false,
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
            gam_problem::HessianValue,
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
                offset: design.affine_offset.clone(),
                penalties: design.penalties_as_penalty_matrix(),
                nullspace_dims: vec![],
                initial_log_lambdas: theta.slice(s![0..eta_penalty_count]).to_owned(),
                initial_beta: Some(pilot_beta.clone()),
                // The warp block here is the RESIDUAL `B_perp`, so precedence
                // runs the same way it does in `fit_binomial_mean_wiggle`: a
                // direction the two still share after residualization is one the
                // mean block can represent and the warp is echoing, so the mean
                // block keeps its priority and the warp yields
                // ([`DEALIASED_WARP_GAUGE_PRIORITY`]). This used to be the other
                // way round because the warp was the DYNAMIC full-width basis,
                // residualized against nothing (#2748).
                gauge_priority: DEFAULT_GAUGE_PRIORITY,
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
                gauge_priority: DEALIASED_WARP_GAUGE_PRIORITY,
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
        let hyper_layout = crate::custom_family::CustomFamilyHyperLayout::new(
            vec![eta_derivs, Vec::new()],
            Vec::new(),
            theta.slice(s![rho_dim..]).to_owned(),
        )?;
        let eval = evaluate_custom_family_joint_hyper(
            &outer_family,
            &blocks,
            &outer_options,
            &theta.slice(s![0..rho_dim]).to_owned(),
            &hyper_layout,
            warm_cache,
            if need_hessian {
                gam_problem::EvalMode::ValueGradientHessian
            } else {
                gam_problem::EvalMode::ValueAndGradient
            },
        ).map_err(|error| error.to_string())?;
        Ok((eval, resolvedspec, design))
    };

    let build_efs = |theta: &Array1<f64>,
                     warm_cache: Option<&crate::custom_family::CustomFamilyWarmStart>|
     -> Result<crate::custom_family::CustomFamilyJointHyperEfsResult, String> {
        let (_, _, blocks, eta_derivs) = build_realized_blocks(theta)?;
        let hyper_layout = crate::custom_family::CustomFamilyHyperLayout::new(
            vec![eta_derivs, Vec::new()],
            Vec::new(),
            theta.slice(s![rho_dim..]).to_owned(),
        )?;
        evaluate_custom_family_joint_hyper_efs(
            &outer_family,
            &blocks,
            &outer_options,
            &theta.slice(s![0..rho_dim]).to_owned(),
            &hyper_layout,
            warm_cache,
        )
        .map_err(|e| e.to_string())
    };

    use crate::model_types::EstimationError;
    use gam_problem::{DeclaredHessianForm, Derivative, OuterEval};
    use gam_solve::rho_optimizer::OuterEvalOrder;

    // Exact first-order AND second-order [rho, psi] calculus is available
    // for all inverse links via the shared jet formulas plus the generic
    // exact-Newton D_βH / D²_βH closures routed through
    // evaluate_custom_family_joint_hyper -> joint_outer_evaluate ->
    // BorrowedJointDerivProvider. Search consumes the exact first-order lane;
    // the terminal certificate alone consumes the exact Hessian.
    //
    // Spatial log-kappa coordinates are ψ (design-moving) dimensions because
    // they rebuild the spatial basis and penalties at each outer proposal.
    let analytic_outer_hessian_available = true;
    let mut seed_heuristic = theta0.to_vec();
    for value in &mut seed_heuristic[..rho_dim] {
        *value = value.exp();
    }
    let problem = gam_solve::rho_optimizer::OuterProblem::new(theta_dim)
        .with_gradient(Derivative::Analytic)
        .with_hessian(if analytic_outer_hessian_available {
            DeclaredHessianForm::Either
        } else {
            DeclaredHessianForm::Unavailable
        })
        // #2359: the family gradient consumes derivatives through order three;
        // reserve its order-four Hessian for the terminal mint certificate.
        .with_prefer_gradient_only(true)
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
        // The saturation REFERENCE stays at the joint prior even where the box
        // above widened a coordinate to `±RHO_BOUND`, exactly as the standard
        // joint `[ρ, ψ]` route does: the box is the per-dimension `lower`/`upper`
        // pair, and this scalar only feeds the seed grid and the bound-free
        // fallback box.
        .with_rho_bound(JOINT_RHO_BOUND)
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
                    gam_problem::HessianValue::Dense(_) | gam_problem::HessianValue::Operator(_)
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
        // The evaluator's contract is "produce the criterion at THIS theta",
        // so its refusals are per-theta refusals. `InvalidInput` says the
        // opposite to `is_trial_point_infeasible`, and the outer boundary then
        // graded them Fatal and killed the fit (#2627).
        let (eval, _, _) = build_eval(theta, state.warm_cache.as_ref(), need_hessian)
            .map_err(|reason| EstimationError::TrialPointRefused { reason })?;
        if !eval.inner_converged {
            state.warm_cache = Some(eval.warm_start);
            return Err(EstimationError::TrialPointRefused {
                reason: "binomial mean-wiggle exact spatial inner solve did not converge"
                    .to_string(),
            });
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
                .map_err(|reason| EstimationError::TrialPointRefused { reason })?;
            if !eval.inner_converged {
                state.warm_cache = Some(eval.warm_start);
                return Err(EstimationError::TrialPointRefused {
                    reason: "binomial mean-wiggle exact spatial cost inner solve did not converge"
                        .to_string(),
                });
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
                .map_err(|reason| EstimationError::TrialPointRefused { reason })?;
            if !eval.inner_converged {
                state.warm_cache = Some(eval.warm_start);
                return Err(EstimationError::TrialPointRefused {
                    reason: "binomial mean-wiggle exact spatial EFS inner solve did not converge"
                        .to_string(),
                });
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
                .map_err(|reason| EstimationError::TrialPointRefused { reason })?;
            state.warm_cache = Some(eval.warm_start);
            Ok(eval.objective)
        },
    );

    let outer = problem
        .run(&mut obj, "binomial mean wiggle exact spatial hyper")
        .map_err(|e| e.to_string())?;
    if !outer.converged() {
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
                offset: design.affine_offset.clone(),
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
    let BinomialMeanWiggleFrozenFit {
        fit,
        saved_warp_beta,
        saved_index_shift,
        ..
    } = fit;

    Ok(BinomialMeanWiggleTermFitResult {
        fit,
        resolvedspec,
        design,
        wiggle_knots,
        wiggle_degree,
        saved_warp_beta,
        saved_index_shift,
    })
}
