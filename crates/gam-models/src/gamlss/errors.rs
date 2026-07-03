// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

/// Typed errors surfaced from this module's helpers and family
/// implementations. The `Display` impl writes the carried `reason` verbatim,
/// so callers that historically returned `Result<_, String>` keep their
/// user-visible text byte-for-byte identical after coercion via the
/// `From<GamlssError> for String` impl below.
#[derive(Debug)]
pub enum GamlssError {
    /// Shape, length, row, or column mismatches between matrices,
    /// vectors, specs, or block configurations.
    DimensionMismatch { reason: String },
    /// Generic input validation that doesn't fit a more specific
    /// variant (e.g. positivity-of-response checks, shape parameter
    /// must be finite > 0).
    InvalidInput { reason: String },
    /// Non-finite values discovered in inputs, coefficients, seeds,
    /// or intermediate quantities required to remain finite.
    NonFinite { reason: String },
    /// A model configuration or feature combination is not supported
    /// by the requested family / link / engine (e.g. identity link on
    /// a binomial mean-wiggle family, unexpected design-map variant).
    UnsupportedConfiguration { reason: String },
    /// Bound, range, monotonicity, or sign constraints violated by
    /// supplied parameters or coefficients.
    ConstraintViolation { reason: String },
    /// Numerical failures during inner solves, integration, or
    /// optimization (invalid probabilities, non-finite log-likelihood,
    /// invalid λ, divergence).
    NumericalFailure { reason: String },
}

impl_reason_error_boilerplate! {
    GamlssError {
        DimensionMismatch,
        InvalidInput,
        NonFinite,
        UnsupportedConfiguration,
        ConstraintViolation,
        NumericalFailure,
    }
}

impl From<crate::block_layout::block_count::BlockCountMismatch> for GamlssError {
    fn from(err: crate::block_layout::block_count::BlockCountMismatch) -> GamlssError {
        GamlssError::DimensionMismatch {
            reason: err.message(),
        }
    }
}

/// Numerical floor on μ ∈ (0, 1) used only for downstream `1/μ` and
/// `1/(1-μ)` divisions and for `μ.ln()` / `(1-μ).ln()` in the generic
/// composed-link binomial log-likelihood (where the logit-stable
/// `log_expit` form is unavailable because `q` is the composed link
/// argument, not the raw logit η). Pure numerical safety, NOT a model
/// assumption — when the optimizer pushes μ to the floor it indicates a
/// separated/saturated fit which is detected and surfaced upstream
/// (`detect_logit_instability`, `Unstable` PIRLS status). For
/// composed-link μ, derivatives `dμ/dq` etc. are NOT zeroed when the
/// floor is hit; they carry the legitimate gradient signal of the
/// outer link and zeroing them would create a phantom flat region that
/// the optimizer would converge to as a stationary point.
pub(crate) const MIN_PROB: f64 = 1e-10;

pub(crate) const MIN_DERIV: f64 = 1e-8;

/// Lower clamp on POSITIVE working weights `w_i = (dμ/dη)² / V(μ_i)`
/// to keep `Xᵀ W X` numerically representable. Strictly numerical:
/// `w` enters subsequent dense matrix products and a true zero (which
/// happens when `dμ/dη = 0` at saturation, e.g. logistic μ → 0 with
/// `dμ/dη = μ(1-μ)`) is harmless but a denormal `w` propagates as
/// inf/NaN through `XᵀWX` because `w * (x_i x_j)` underflows
/// non-uniformly. `floor_positiveweight` returns 0 for non-finite or
/// non-positive inputs (so saturation correctly drops the row from
/// the inner Newton system); the floor only fires for *strictly
/// positive* tiny weights. The 1e-12 magnitude is chosen so that
/// `1e-12 · max|x|² · n` stays comfortably above `f64::MIN_POSITIVE`
/// at large scale.
///
/// This is the canonical positive-weight floor (`1e-12`); the value is owned by
/// [`gam_problem::MIN_WEIGHT`] so every floored family shares one definition
/// rather than re-declaring it per module.
use gam_problem::MIN_WEIGHT;

/// Hard symmetric clamp on η used by the Poisson / Gaussian / Gamma working-
/// model log-likelihood loops to keep `exp(η)` and `log(σ)` finite under the
/// IRLS step. Hoisted out of each loop so all three families share the same
/// numerical regime.
pub(crate) const ETA_HARD_CLAMP: f64 = 30.0;

/// Saturated `exp(η)` used by every log-link mean reconstruction in this
/// module: clamp η into `[−ETA_HARD_CLAMP, ETA_HARD_CLAMP]` so `exp` stays
/// finite, then floor at `MIN_WEIGHT` so downstream divisions never see
/// exact zero. Centralising the formula here means a tolerance change
/// propagates to all three families (Poisson / Gaussian / Gamma) without
/// risk of one path drifting.
#[inline]
pub(crate) fn saturated_exp_eta(eta: f64) -> f64 {
    eta.clamp(-ETA_HARD_CLAMP, ETA_HARD_CLAMP)
        .exp()
        .max(MIN_WEIGHT)
}

/// Floor applied to a fitted smoothing parameter λ before `ln(λ)` is taken to
/// seed an outer-loop `initial_log_lambdas` warm start. A pilot fit can return
/// λ underflowed to exactly 0 for a deselected (effectively unpenalized) term;
/// `ln(0) = -inf` would poison the seed, so we floor at the smallest λ that is
/// still numerically distinguishable from zero in the log-domain rather than a
/// modelling-meaningful value. `ln(1e-12) ≈ -27.6` sits well below any λ the
/// outer optimizer would select, so a genuinely tiny pilot λ still seeds the
/// search near its lower edge.
pub(crate) const WARMSTART_LOG_LAMBDA_FLOOR: f64 = 1e-12;

pub(crate) const EXACT_DENSE_BLOCK_BUDGET_BYTES: usize = 512 * 1024 * 1024;

pub(crate) const EXACT_DENSE_TOTAL_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

pub(crate) const GAMLSS_ROWWISE_PAR_MIN_N: usize = 4096;

pub(crate) const GAMLSS_PROJECTED_TRACE_TARGET_BYTES: usize = 32 * 1024 * 1024;

pub(crate) const GAMLSS_PROJECTED_TRACE_MIN_CHUNK_ROWS: usize = 64;

pub(crate) const GAMLSS_PROJECTED_TRACE_MAX_CHUNK_ROWS: usize = 8192;

pub(crate) fn gamlss_projected_trace_chunk_rows(
    rank: usize,
    projected_channel_count: usize,
    gram_column_count: usize,
) -> usize {
    let per_row_values = rank
        .saturating_mul(projected_channel_count.max(1))
        .saturating_add(gram_column_count.max(1))
        .max(1);
    let per_row_bytes = per_row_values.saturating_mul(std::mem::size_of::<f64>());
    let rows = GAMLSS_PROJECTED_TRACE_TARGET_BYTES / per_row_bytes.max(1);
    rows.clamp(
        GAMLSS_PROJECTED_TRACE_MIN_CHUNK_ROWS,
        GAMLSS_PROJECTED_TRACE_MAX_CHUNK_ROWS,
    )
}

pub(crate) fn gamlss_rowwise_map<F>(n: usize, f: F) -> Array1<f64>
where
    F: Fn(usize) -> f64 + Sync,
{
    if n >= GAMLSS_ROWWISE_PAR_MIN_N {
        Array1::from((0..n).into_par_iter().map(&f).collect::<Vec<f64>>())
    } else {
        Array1::from_iter((0..n).map(f))
    }
}

pub(crate) fn gamlss_rowwise_map_result<F>(n: usize, f: F) -> Result<Array1<f64>, String>
where
    F: Fn(usize) -> Result<f64, String> + Sync,
{
    if n >= GAMLSS_ROWWISE_PAR_MIN_N {
        let values: Result<Vec<f64>, String> = (0..n).into_par_iter().map(&f).collect();
        Ok(Array1::from(values?))
    } else {
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            out[i] = f(i)?;
        }
        Ok(out)
    }
}

pub(crate) enum DenseOrOperator<'a> {
    Borrowed(&'a Array2<f64>),
    Owned(Array2<f64>),
    Operator(DesignMatrix),
}

impl DenseOrOperator<'_> {
    pub(crate) fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(dense) => dense.nrows(),
            Self::Owned(dense) => dense.nrows(),
            Self::Operator(design) => design.nrows(),
        }
    }

    pub(crate) fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(dense) => dense.ncols(),
            Self::Owned(dense) => dense.ncols(),
            Self::Operator(design) => design.ncols(),
        }
    }

    pub(crate) fn row_chunk(&self, rows: std::ops::Range<usize>) -> Result<Array2<f64>, String> {
        match self {
            Self::Borrowed(dense) => Ok(dense.slice(s![rows, ..]).to_owned()),
            Self::Owned(dense) => Ok(dense.slice(s![rows, ..]).to_owned()),
            Self::Operator(design) => design.try_row_chunk(rows).map_err(|e| e.to_string()),
        }
    }

    pub(crate) fn dot(&self, beta: ArrayView1<'_, f64>) -> Array1<f64> {
        let n = self.nrows();
        let p = self.ncols();
        assert_eq!(beta.len(), p);
        match self {
            Self::Borrowed(dense) => fast_av(*dense, &beta),
            Self::Owned(dense) => fast_av(dense, &beta),
            Self::Operator(design) => {
                let mut out = Array1::<f64>::zeros(n);
                for rows in exact_design_row_chunks(n, p) {
                    let chunk = design
                        .try_row_chunk(rows.clone())
                        .expect("gamlss DesignSlot::dot: design row chunk materialization failed");
                    out.slice_mut(s![rows]).assign(&fast_av(&chunk, &beta));
                }
                out
            }
        }
    }
}

/// Resolve a single dense block design from a `ParameterBlockSpec`, falling
/// back to materializing the sparse representation through the policy when
/// the dense form isn't already cached. Returns `Cow::Borrowed` whenever the
/// spec already holds a dense array; `Cow::Owned` only after a forced
/// materialization. The `materialization_label` string is woven into the
/// materializer's error so callers can pin which block failed.
pub(crate) fn dense_block_from_spec<'a>(
    spec: &'a ParameterBlockSpec,
    material_policy: &gam_runtime::resource::MaterializationPolicy,
    materialization_label: &str,
) -> Result<Cow<'a, Array2<f64>>, String> {
    match spec.design.as_dense_ref() {
        Some(d) => Ok(Cow::Borrowed(d)),
        None => Ok(Cow::Owned(
            spec.design
                .try_to_dense_with_policy(material_policy, "gamlss dense_block_from_spec")
                .map_err(|e| format!("{materialization_label}: {e}"))?
                .as_ref()
                .clone(),
        )),
    }
}

/// Resolve the (primary, log-σ) pair of dense block designs that every
/// LocationScale family's spec-aware exact path needs. The primary block is
/// the family-specific "mean" axis (μ for Gaussian, latent t for Binomial);
/// the `short_family_name` ("GaussianLocationScale", "BinomialLocationScale",
/// or their Wiggle siblings) and `primary_label` ("mu" / "threshold") are
/// woven into the per-block materialization label for diagnostics.
pub(crate) fn dense_locscale_block_designs_fromspecs<'a>(
    specs: &'a [ParameterBlockSpec],
    expected_count: usize,
    family_name: &str,
    short_family_name: &str,
    primary_block_idx: usize,
    log_sigma_block_idx: usize,
    primary_label: &str,
    material_policy: &gam_runtime::resource::MaterializationPolicy,
) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
    if specs.len() != expected_count {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "{family_name} expects {expected_count} specs, got {}",
                specs.len()
            ),
        }
        .into());
    }
    let primary = dense_block_from_spec(
        &specs[primary_block_idx],
        material_policy,
        &format!("{short_family_name} dense_block_designs_fromspecs {primary_label}"),
    )?;
    let log_sigma = dense_block_from_spec(
        &specs[log_sigma_block_idx],
        material_policy,
        &format!("{short_family_name} dense_block_designs_fromspecs log_sigma"),
    )?;
    Ok((primary, log_sigma))
}

/// Assemble the joint log-likelihood gradient `g = ∇_β log L` from a family's
/// per-block IRLS working sets, in the flattened `β = [β_0; β_1; …]` block
/// order sized from `specs`.
///
/// For a `Diagonal` block the exact coefficient-space score is
/// `X_bᵀ (w ⊙ (z − η))`: by the IRLS pseudo-response identity
/// `z_i = η_i + (∂ℓ/∂η_i)/w_i`, the row score is `w_i (z_i − η_i) = ∂ℓ/∂η_i`
/// **exactly** — independent of whether `w` is the Fisher or the observed
/// weight (the score/gradient is always the exact observed gradient; only the
/// Hessian differs between Fisher scoring and observed curvature). An
/// `ExactNewton` block carries its own analytic gradient. This is the same
/// single source of truth the inner joint-Newton RHS uses
/// (`exact_newton_joint_gradient_from_eval`), so a family whose `evaluate()`
/// emits these working sets gets a joint gradient guaranteed consistent with
/// its joint Hessian without a bespoke, possibly-disagreeing derivation.
pub(crate) fn gamlss_joint_gradient_from_working_sets(
    eval: &FamilyEvaluation,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
) -> Result<ExactNewtonJointGradientEvaluation, String> {
    if eval.blockworking_sets.len() != specs.len() || states.len() != specs.len() {
        return Err(GamlssError::DimensionMismatch { reason: format!(
            "gamlss joint gradient: block/spec/state count mismatch (working_sets={}, specs={}, states={})",
            eval.blockworking_sets.len(),
            specs.len(),
            states.len()
        ) }
        .into());
    }
    let total: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    let mut gradient = Array1::<f64>::zeros(total);
    let mut offset = 0usize;
    for ((spec, work), state) in specs
        .iter()
        .zip(eval.blockworking_sets.iter())
        .zip(states.iter())
    {
        let width = spec.design.ncols();
        let block_grad = match work {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                let n = working_response.len();
                if working_weights.len() != n || state.eta.len() != n || spec.design.nrows() != n {
                    return Err(GamlssError::DimensionMismatch { reason: format!(
                        "gamlss joint gradient: diagonal working-set length mismatch (z={}, w={}, η={}, X_rows={})",
                        n,
                        working_weights.len(),
                        state.eta.len(),
                        spec.design.nrows()
                    ) }
                    .into());
                }
                let mut weighted = Array1::<f64>::zeros(n);
                for i in 0..n {
                    weighted[i] = working_weights[i] * (working_response[i] - state.eta[i]);
                }
                spec.design.transpose_vector_multiply(&weighted)
            }
            BlockWorkingSet::ExactNewton {
                gradient: block_gradient,
                ..
            } => block_gradient.clone(),
        };
        if block_grad.len() != width {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "gamlss joint gradient: assembled block gradient length {} != design cols {width}",
                block_grad.len()
            ) }
            .into());
        }
        gradient
            .slice_mut(s![offset..offset + width])
            .assign(&block_grad);
        offset += width;
    }
    Ok(ExactNewtonJointGradientEvaluation {
        log_likelihood: eval.log_likelihood,
        gradient,
    })
}

/// Materialize a single location-scale family's two cached block designs
/// (`primary` = mu/threshold, plus `log_sigma`) into dense matrices, borrowing
/// when the design is already dense and owning a policy-materialized copy
/// otherwise. Every non-wiggle and wiggle location-scale family's
/// `dense_block_designs` method is identical bar the accessed field and the
/// diagnostic labels, so both bits are passed in.
pub(crate) fn dense_locscale_block_designs_cached<'a>(
    primary_design: Option<&'a DesignMatrix>,
    log_sigma_design: Option<&'a DesignMatrix>,
    family_name: &str,
    short_family_name: &str,
    primary_label: &str,
    material_policy: &gam_runtime::resource::MaterializationPolicy,
) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
    let primary_design = primary_design
        .ok_or_else(|| format!("{family_name} exact path is missing {primary_label} design"))?;
    let log_sigma_design = log_sigma_design
        .ok_or_else(|| format!("{family_name} exact path is missing log-sigma design"))?;
    let primary = match primary_design.as_dense_ref() {
        Some(d) => Cow::Borrowed(d),
        None => Cow::Owned(
            primary_design
                .try_to_dense_with_policy(material_policy, "gamlss dense_locscale_block_designs")
                .map_err(|e| {
                    format!("{short_family_name} dense_block_designs {primary_label}: {e}")
                })?
                .as_ref()
                .clone(),
        ),
    };
    let log_sigma = match log_sigma_design.as_dense_ref() {
        Some(d) => Cow::Borrowed(d),
        None => Cow::Owned(
            log_sigma_design
                .try_to_dense_with_policy(material_policy, "gamlss dense_locscale_block_designs")
                .map_err(|e| format!("{short_family_name} dense_block_designs log_sigma: {e}"))?
                .as_ref()
                .clone(),
        ),
    };
    Ok((primary, log_sigma))
}

/// One resolved ψ-direction for a two-axis (primary + log-σ) location-scale
/// family. Holds the neutral pieces shared by every such family's
/// `exact_newton_joint_psi_direction`; each family wraps these into its own
/// named struct (mu/threshold field renames only).
pub(crate) struct LocScalePsiDirectionParts {
    pub(crate) block_idx: usize,
    pub(crate) local_idx: usize,
    pub(crate) primary_psi: PsiDesignMap,
    pub(crate) log_sigma_psi: PsiDesignMap,
    pub(crate) primary_z: Array1<f64>,
    pub(crate) log_sigma_z: Array1<f64>,
}

/// Shared body of every two-axis location-scale family's
/// `exact_newton_joint_psi_direction`. Walks the flat ψ-derivative list,
/// resolves the ψ-design map for the selected block (primary = block 0, log-σ
/// = block 1; the off-axis map is the matching `Zero`), and applies each
/// block's β via `forward_mul`. The wiggle block (and any other index) yields
/// `None`, matching the per-family methods. The only per-family variation —
/// the column counts, the two block betas, the block-list length (2 or 3) and
/// the diagnostic label prefix — is passed in; the math is identical across
/// Gaussian/Binomial × wiggle/non-wiggle.
pub(crate) fn locscale_joint_psi_direction_parts(
    block_states: &[ParameterBlockState],
    derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    psi_index: usize,
    n: usize,
    p_primary: usize,
    p_log_sigma: usize,
    primary_block_idx: usize,
    log_sigma_block_idx: usize,
    expected_blocks: usize,
    family_name: &str,
    primary_label: &str,
    policy: &gam_runtime::resource::ResourcePolicy,
) -> Result<Option<LocScalePsiDirectionParts>, String> {
    validate_block_count::<GamlssError>(family_name, expected_blocks, block_states.len())?;
    if derivative_blocks.len() != expected_blocks {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "{family_name} joint psi direction expects {expected_blocks} derivative block lists, got {}",
                derivative_blocks.len()
            ),
        }
        .into());
    }
    let beta_primary = &block_states[primary_block_idx].beta;
    let beta_log_sigma = &block_states[log_sigma_block_idx].beta;

    let mut global = 0usize;
    for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
        for (local_idx, deriv) in block_derivs.iter().enumerate() {
            if global == psi_index {
                let primary_psi;
                let log_sigma_psi;
                let primary_z;
                let log_sigma_z;
                if block_idx == primary_block_idx {
                    primary_psi = resolve_custom_family_x_psi_map(
                        deriv,
                        n,
                        p_primary,
                        0..n,
                        &format!("{family_name} {primary_label}"),
                        policy,
                    )?;
                    primary_z = primary_psi
                        .forward_mul(beta_primary.view())
                        .map_err(|e| format!("{family_name} {primary_label} forward_mul: {e}"))?;
                    log_sigma_psi = PsiDesignMap::Zero {
                        nrows: n,
                        ncols: p_log_sigma,
                    };
                    log_sigma_z = Array1::<f64>::zeros(n);
                } else if block_idx == log_sigma_block_idx {
                    log_sigma_psi = resolve_custom_family_x_psi_map(
                        deriv,
                        n,
                        p_log_sigma,
                        0..n,
                        &format!("{family_name} log-sigma"),
                        policy,
                    )?;
                    log_sigma_z = log_sigma_psi
                        .forward_mul(beta_log_sigma.view())
                        .map_err(|e| format!("{family_name} log-sigma forward_mul: {e}"))?;
                    primary_psi = PsiDesignMap::Zero {
                        nrows: n,
                        ncols: p_primary,
                    };
                    primary_z = Array1::<f64>::zeros(n);
                } else {
                    return Ok(None);
                }
                return Ok(Some(LocScalePsiDirectionParts {
                    block_idx,
                    local_idx,
                    primary_psi,
                    log_sigma_psi,
                    primary_z,
                    log_sigma_z,
                }));
            }
            global += 1;
        }
    }
    Ok(None)
}

/// Shared second-derivative design drift assembly for two-axis location-scale
/// joint-ψ paths. The family-specific methods differ only by block constants,
/// labels, and field names; the ψψ map lookup and `X_{ab} β` action are the
/// same for Gaussian/Binomial and wiggle/non-wiggle variants.
pub(crate) struct LocScalePsiDriftConfig<'a> {
    pub(crate) n: usize,
    pub(crate) p_primary: usize,
    pub(crate) p_log_sigma: usize,
    pub(crate) primary_block_idx: usize,
    pub(crate) log_sigma_block_idx: usize,
    pub(crate) family_name: &'a str,
    pub(crate) primary_label: &'a str,
    pub(crate) policy: &'a gam_runtime::resource::ResourcePolicy,
}

pub(crate) fn locscale_joint_psisecond_design_drifts(
    block_states: &[ParameterBlockState],
    derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    psi_a: &LocationScaleJointPsiDirection,
    psi_b: &LocationScaleJointPsiDirection,
    cfg: LocScalePsiDriftConfig<'_>,
) -> Result<LocationScaleJointPsiSecondDrifts, String> {
    let beta_primary = &block_states[cfg.primary_block_idx].beta;
    let beta_log_sigma = &block_states[cfg.log_sigma_block_idx].beta;
    let mut primary_ab_action = None;
    let mut log_sigma_ab_action = None;
    let mut primary_ab = None;
    let mut log_sigma_ab = None;

    // Smooth ψ second derivatives are block-local. Cross-block ψ_a/ψ_b
    // design second derivatives are therefore zero unless the derivative
    // payload itself supplies them for the same moving block.
    if psi_a.block_idx == psi_b.block_idx {
        let deriv = &derivative_blocks[psi_a.block_idx][psi_a.local_idx];
        let deriv_b = &derivative_blocks[psi_b.block_idx][psi_b.local_idx];
        if psi_a.block_idx == cfg.primary_block_idx {
            let (action, matrix) = psi_psi_map_to_drift_slots(
                deriv,
                deriv_b,
                psi_b.local_idx,
                cfg.n,
                cfg.p_primary,
                &format!("{} {}", cfg.family_name, cfg.primary_label),
                cfg.policy,
            )?;
            primary_ab_action = action;
            primary_ab = matrix;
        } else if psi_a.block_idx == cfg.log_sigma_block_idx {
            let (action, matrix) = psi_psi_map_to_drift_slots(
                deriv,
                deriv_b,
                psi_b.local_idx,
                cfg.n,
                cfg.p_log_sigma,
                &format!("{} log-sigma", cfg.family_name),
                cfg.policy,
            )?;
            log_sigma_ab_action = action;
            log_sigma_ab = matrix;
        }
    }

    let z_primary_ab = second_psi_linear_map(
        primary_ab_action.as_ref(),
        primary_ab.as_ref(),
        cfg.n,
        cfg.p_primary,
    )
    .forward_mul(beta_primary.view());
    let z_ls_ab = second_psi_linear_map(
        log_sigma_ab_action.as_ref(),
        log_sigma_ab.as_ref(),
        cfg.n,
        cfg.p_log_sigma,
    )
    .forward_mul(beta_log_sigma.view());

    Ok(LocationScaleJointPsiSecondDrifts {
        x_primary_ab_action: primary_ab_action,
        x_ls_ab_action: log_sigma_ab_action,
        x_primary_ab: primary_ab,
        x_ls_ab: log_sigma_ab,
        z_primary_ab,
        z_ls_ab,
    })
}

pub(crate) fn psi_psi_map_to_drift_slots(
    deriv: &crate::custom_family::CustomFamilyBlockPsiDerivative,
    deriv_b: &crate::custom_family::CustomFamilyBlockPsiDerivative,
    local_idx_b: usize,
    n: usize,
    p: usize,
    label: &str,
    policy: &gam_runtime::resource::ResourcePolicy,
) -> Result<
    (
        Option<crate::custom_family::CustomFamilyPsiSecondDesignAction>,
        Option<Array2<f64>>,
    ),
    String,
> {
    match resolve_custom_family_x_psi_psi_map(
        deriv,
        deriv_b,
        local_idx_b,
        n,
        p,
        0..n,
        label,
        policy,
    )? {
        crate::custom_family::PsiDesignMap::Second { action } => Ok((Some(action), None)),
        crate::custom_family::PsiDesignMap::Dense { matrix } => Ok((None, Some((*matrix).clone()))),
        crate::custom_family::PsiDesignMap::Zero { .. } => Ok((None, None)),
        crate::custom_family::PsiDesignMap::First { .. } => {
            Err(GamlssError::UnsupportedConfiguration {
                reason: format!("{label}: unexpected First variant from _psi_psi_map"),
            }
            .into())
        }
    }
}

pub(crate) fn dense_block_or_operator<'a>(
    design: &'a DesignMatrix,
    n: usize,
    p: usize,
    budget_bytes: usize,
    policy: &gam_runtime::resource::ResourcePolicy,
) -> DenseOrOperator<'a> {
    if let Some(dense) = design.as_dense_ref() {
        return DenseOrOperator::Borrowed(dense);
    }

    let dense_bytes = 8usize.saturating_mul(n).saturating_mul(p);
    if dense_bytes <= budget_bytes
        && let Ok(arc) = design
            .try_to_dense_with_policy(&policy.material_policy(), "gamlss dense_block_or_operator")
    {
        return DenseOrOperator::Owned(arc.as_ref().clone());
    }

    DenseOrOperator::Operator(design.clone())
}

pub(crate) fn dense_blocks_planned_budget(blocks: &[&DesignMatrix]) -> Vec<usize> {
    let mut planned = vec![0; blocks.len()];
    let mut total = 0usize;
    for (idx, design) in blocks.iter().enumerate() {
        if design.as_dense_ref().is_some() {
            continue;
        }
        let bytes = 8usize
            .saturating_mul(design.nrows())
            .saturating_mul(design.ncols());
        if bytes <= EXACT_DENSE_BLOCK_BUDGET_BYTES
            && total.saturating_add(bytes) <= EXACT_DENSE_TOTAL_BUDGET_BYTES
        {
            planned[idx] = bytes;
            total += bytes;
        }
    }
    planned
}

pub(crate) fn exact_design_row_chunks(
    n: usize,
    p: usize,
) -> impl Iterator<Item = std::ops::Range<usize>> {
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_ROWS: usize = 512;
    const MAX_ROWS: usize = 131_072;
    let rows = (TARGET_BYTES / (p.max(1) * 8))
        .clamp(MIN_ROWS, MAX_ROWS)
        .min(n.max(1));
    (0..n)
        .step_by(rows)
        .map(move |start| start..(start + rows).min(n))
}

pub(crate) fn design_weighted_column_squares(
    design: &DesignMatrix,
    weights: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let n = design.nrows();
    let p = design.ncols();
    if weights.len() != n {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "design weighted column squares dimension mismatch: weights={}, rows={}",
                weights.len(),
                n
            ),
        }
        .into());
    }
    let mut out = Array1::<f64>::zeros(p);
    for rows in exact_design_row_chunks(n, p) {
        let chunk = design.try_row_chunk(rows.clone()).map_err(|e| {
            format!("design weighted column squares row chunk materialization failed: {e}")
        })?;
        for (local_i, row) in chunk.outer_iter().enumerate() {
            let w = weights[rows.start + local_i];
            if w == 0.0 {
                continue;
            }
            for j in 0..p {
                let x = row[j];
                out[j] += w * x * x;
            }
        }
    }
    Ok(out)
}

#[inline]
pub(crate) fn floor_positiveweight(rawweight: f64, minweight: f64) -> f64 {
    if !rawweight.is_finite() || rawweight <= 0.0 {
        0.0
    } else {
        rawweight.max(minweight)
    }
}

#[inline]
pub(crate) fn logb_dlog_sigma_deta(sigma: f64, d_sigma_deta: f64) -> f64 {
    if d_sigma_deta.is_infinite() {
        1.0
    } else {
        let value = d_sigma_deta / sigma;
        if value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

#[inline]
pub(crate) fn gaussian_log_sigma_irlsinfo_directional_derivative(
    weight: f64,
    sigma: f64,
    d_sigma_deta: f64,
    d_eta: f64,
) -> f64 {
    if weight == 0.0 || d_eta == 0.0 || !sigma.is_finite() || sigma <= 0.0 {
        return 0.0;
    }
    // Logb form mirrors gaussian_jointrow_scalars: κ = exp(η)/(b + exp(η)) ∈ [0, 1)
    // and dκ/dη = κ(1−κ). Use dσ/dη over σ directly so the η → −∞ tail
    // preserves subnormal information instead of cancelling in `1 − b/σ`;
    // the helper handles the η → +∞ inf/inf case by returning the analytic
    // limit 1.
    let g = logb_dlog_sigma_deta(sigma, d_sigma_deta);
    if !g.is_finite() || !(0.0..1.0).contains(&g) {
        return 0.0;
    }
    let rawinfo = 2.0 * weight * g * g;
    if !rawinfo.is_finite() || rawinfo <= MIN_WEIGHT {
        return 0.0;
    }
    let dg_deta = g * (1.0 - g);
    let dw = 4.0 * weight * g * dg_deta * d_eta;
    if dw.is_finite() { dw } else { 0.0 }
}

#[derive(Clone, Copy)]
pub(crate) struct GaussianDiagonalRowKernel {
    pub(crate) log_likelihood: f64,
    pub(crate) location_working_weight: f64,
    pub(crate) location_working_shift: f64,
    pub(crate) log_sigma_working_weight: f64,
    pub(crate) log_sigma_working_response: f64,
}

#[inline]
pub(crate) fn gaussian_diagonal_row_kernel(
    y: f64,
    location_eta: f64,
    eta_log_sigma: f64,
    obs_weight: f64,
    ln2pi: f64,
) -> GaussianDiagonalRowKernel {
    if obs_weight == 0.0 {
        return GaussianDiagonalRowKernel {
            log_likelihood: 0.0,
            location_working_weight: 0.0,
            location_working_shift: 0.0,
            log_sigma_working_weight: 0.0,
            log_sigma_working_response: eta_log_sigma,
        };
    }

    // logb noise link σ = b + exp(η) bounds σ ≥ b > 0 by construction, so the
    // Gaussian location-scale objective ½Σ(y−μ)²/σ² + Σlog σ is bounded below
    // for any finite data. Its working weight 1/σ² is bounded by 1/b², so
    // H_μμ has bounded condition number — no after-the-fact floor or cap is
    // needed (the previous (1e-12, 1e24) clamp was a numerical bandaid for the
    // pure-exp link's σ→0 singularity and is structurally unnecessary here).
    // ApproxKind: Exact — working weight analytically bounded in (0, 1/b²].
    let SigmaJet1 { sigma, d1 } = logb_sigma_jet1_scalar(eta_log_sigma);
    let inv_s2 = (sigma * sigma).recip();
    let residual = y - location_eta;
    let location_working_weight = floor_positiveweight(obs_weight * inv_s2, MIN_WEIGHT);
    // dlog σ/dη = (∂σ/∂η)/σ = exp(η)/(b + exp(η)) ∈ [0, 1).
    // Use dσ/dη over σ directly so the η→−∞ tail preserves subnormal
    // derivative information instead of cancelling in `1 − b/σ`; the helper
    // returns the analytic limit 1 for the η→+∞ inf/inf case.
    // Fisher info per obs = 2·(dσ/dη)²/σ² = 2·dlog_sigma_deta², matching the
    // formula for the pure-exp link (where dlog_sigma_deta ≡ 1).
    let dlog_sigma_deta = logb_dlog_sigma_deta(sigma, d1);
    let log_sigma_working_weight = floor_positiveweight(
        2.0 * obs_weight * dlog_sigma_deta * dlog_sigma_deta,
        MIN_WEIGHT,
    );
    let log_sigma_score = obs_weight * (residual * residual * inv_s2 - 1.0) * dlog_sigma_deta;
    let log_sigma_working_response = if log_sigma_working_weight == 0.0 {
        eta_log_sigma
    } else {
        eta_log_sigma + log_sigma_score / log_sigma_working_weight
    };

    GaussianDiagonalRowKernel {
        log_likelihood: obs_weight
            * (-0.5 * (residual * residual * inv_s2 + ln2pi + 2.0 * sigma.ln())),
        location_working_weight,
        location_working_shift: residual,
        log_sigma_working_weight,
        log_sigma_working_response,
    }
}

/// Link identifiers for distribution parameters in multi-parameter GAMLSS families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterLink {
    Identity,
    Log,
    Logit,
    Probit,
    InverseLink,
    /// Learnable smooth departure from a known base link.
    Wiggle,
}
