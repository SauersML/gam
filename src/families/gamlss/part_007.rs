use joint_packing::{
    binomial_pack_mean_wiggle_joint_score, binomial_pack_mean_wiggle_joint_symmetrichessian,
    gaussian_pack_joint_score, gaussian_pack_joint_symmetrichessian,
    gaussian_pack_wiggle_joint_score, gaussian_pack_wiggle_joint_symmetrichessian,
};


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


impl std::fmt::Display for GamlssError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GamlssError::DimensionMismatch { reason }
            | GamlssError::InvalidInput { reason }
            | GamlssError::NonFinite { reason }
            | GamlssError::UnsupportedConfiguration { reason }
            | GamlssError::ConstraintViolation { reason }
            | GamlssError::NumericalFailure { reason } => f.write_str(reason),
        }
    }
}


impl std::error::Error for GamlssError {}


impl From<GamlssError> for String {
    fn from(err: GamlssError) -> Self {
        err.to_string()
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
const MIN_PROB: f64 = 1e-10;

const MIN_DERIV: f64 = 1e-8;

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
/// This is the canonical PIRLS positive-weight floor (`1e-12`); the value is
/// owned by [`crate::solver::pirls::MIN_WEIGHT`] so every floored family shares
/// one definition rather than re-declaring it per module.
use crate::solver::pirls::MIN_WEIGHT;

/// Hard symmetric clamp on η used by the Poisson / Gaussian / Gamma working-
/// model log-likelihood loops to keep `exp(η)` and `log(σ)` finite under the
/// IRLS step. Hoisted out of each loop so all three families share the same
/// numerical regime.
const ETA_HARD_CLAMP: f64 = 30.0;


/// Saturated `exp(η)` used by every log-link mean reconstruction in this
/// module: clamp η into `[−ETA_HARD_CLAMP, ETA_HARD_CLAMP]` so `exp` stays
/// finite, then floor at `MIN_WEIGHT` so downstream divisions never see
/// exact zero. Centralising the formula here means a tolerance change
/// propagates to all three families (Poisson / Gaussian / Gamma) without
/// risk of one path drifting.
#[inline]
fn saturated_exp_eta(eta: f64) -> f64 {
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
const WARMSTART_LOG_LAMBDA_FLOOR: f64 = 1e-12;


const EXACT_DENSE_BLOCK_BUDGET_BYTES: usize = 512 * 1024 * 1024;

const EXACT_DENSE_TOTAL_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

const GAMLSS_ROWWISE_PAR_MIN_N: usize = 4096;

const GAMLSS_PROJECTED_TRACE_TARGET_BYTES: usize = 32 * 1024 * 1024;

const GAMLSS_PROJECTED_TRACE_MIN_CHUNK_ROWS: usize = 64;

const GAMLSS_PROJECTED_TRACE_MAX_CHUNK_ROWS: usize = 8192;


fn gamlss_projected_trace_chunk_rows(
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


fn gamlss_rowwise_map<F>(n: usize, f: F) -> Array1<f64>
where
    F: Fn(usize) -> f64 + Sync,
{
    if n >= GAMLSS_ROWWISE_PAR_MIN_N {
        Array1::from((0..n).into_par_iter().map(&f).collect::<Vec<f64>>())
    } else {
        Array1::from_iter((0..n).map(f))
    }
}


fn gamlss_rowwise_map_result<F>(n: usize, f: F) -> Result<Array1<f64>, String>
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


enum DenseOrOperator<'a> {
    Borrowed(&'a Array2<f64>),
    Owned(Array2<f64>),
    Operator(DesignMatrix),
}


impl DenseOrOperator<'_> {
    fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(dense) => dense.nrows(),
            Self::Owned(dense) => dense.nrows(),
            Self::Operator(design) => design.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(dense) => dense.ncols(),
            Self::Owned(dense) => dense.ncols(),
            Self::Operator(design) => design.ncols(),
        }
    }

    fn row_chunk(&self, rows: std::ops::Range<usize>) -> Result<Array2<f64>, String> {
        match self {
            Self::Borrowed(dense) => Ok(dense.slice(s![rows, ..]).to_owned()),
            Self::Owned(dense) => Ok(dense.slice(s![rows, ..]).to_owned()),
            Self::Operator(design) => design.try_row_chunk(rows).map_err(|e| e.to_string()),
        }
    }

    fn dot(&self, beta: ArrayView1<'_, f64>) -> Array1<f64> {
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
fn dense_block_from_spec<'a>(
    spec: &'a ParameterBlockSpec,
    material_policy: &crate::resource::MaterializationPolicy,
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
fn dense_locscale_block_designs_fromspecs<'a>(
    specs: &'a [ParameterBlockSpec],
    expected_count: usize,
    family_name: &str,
    short_family_name: &str,
    primary_block_idx: usize,
    log_sigma_block_idx: usize,
    primary_label: &str,
    material_policy: &crate::resource::MaterializationPolicy,
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


/// Materialize a single location-scale family's two cached block designs
/// (`primary` = mu/threshold, plus `log_sigma`) into dense matrices, borrowing
/// when the design is already dense and owning a policy-materialized copy
/// otherwise. Every non-wiggle and wiggle location-scale family's
/// `dense_block_designs` method is identical bar the accessed field and the
/// diagnostic labels, so both bits are passed in.
fn dense_locscale_block_designs_cached<'a>(
    primary_design: Option<&'a DesignMatrix>,
    log_sigma_design: Option<&'a DesignMatrix>,
    family_name: &str,
    short_family_name: &str,
    primary_label: &str,
    material_policy: &crate::resource::MaterializationPolicy,
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
struct LocScalePsiDirectionParts {
    block_idx: usize,
    local_idx: usize,
    primary_psi: PsiDesignMap,
    log_sigma_psi: PsiDesignMap,
    primary_z: Array1<f64>,
    log_sigma_z: Array1<f64>,
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
#[allow(clippy::too_many_arguments)]
fn locscale_joint_psi_direction_parts(
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
    policy: &crate::resource::ResourcePolicy,
) -> Result<Option<LocScalePsiDirectionParts>, String> {
    if block_states.len() != expected_blocks || derivative_blocks.len() != expected_blocks {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "{family_name} joint psi direction expects {expected_blocks} blocks and {expected_blocks} derivative block lists, got {} and {}",
                block_states.len(),
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
struct LocScalePsiDriftConfig<'a> {
    n: usize,
    p_primary: usize,
    p_log_sigma: usize,
    primary_block_idx: usize,
    log_sigma_block_idx: usize,
    family_name: &'a str,
    primary_label: &'a str,
    policy: &'a crate::resource::ResourcePolicy,
}


fn locscale_joint_psisecond_design_drifts(
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


fn psi_psi_map_to_drift_slots(
    deriv: &crate::custom_family::CustomFamilyBlockPsiDerivative,
    deriv_b: &crate::custom_family::CustomFamilyBlockPsiDerivative,
    local_idx_b: usize,
    n: usize,
    p: usize,
    label: &str,
    policy: &crate::resource::ResourcePolicy,
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


fn dense_block_or_operator<'a>(
    design: &'a DesignMatrix,
    n: usize,
    p: usize,
    budget_bytes: usize,
    policy: &crate::resource::ResourcePolicy,
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


fn dense_blocks_planned_budget(blocks: &[&DesignMatrix]) -> Vec<usize> {
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


pub(super) fn exact_design_row_chunks(
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


fn design_weighted_column_squares(
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
fn floor_positiveweight(rawweight: f64, minweight: f64) -> f64 {
    if !rawweight.is_finite() || rawweight <= 0.0 {
        0.0
    } else {
        rawweight.max(minweight)
    }
}


#[inline]
fn logb_dlog_sigma_deta(sigma: f64, d_sigma_deta: f64) -> f64 {
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
fn gaussian_log_sigma_irlsinfo_directional_derivative(
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
struct GaussianDiagonalRowKernel {
    log_likelihood: f64,
    location_working_weight: f64,
    location_working_shift: f64,
    log_sigma_working_weight: f64,
    log_sigma_working_response: f64,
}


#[inline]
fn gaussian_diagonal_row_kernel(
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


#[derive(Clone, Copy)]
struct GamlssLambdaLayout {
    k_mean: usize,
    k_noise: usize,
    kwiggle: usize,
}


impl GamlssLambdaLayout {
    fn two_block(k_mean: usize, k_noise: usize) -> Self {
        Self {
            k_mean,
            k_noise,
            kwiggle: 0,
        }
    }

    fn withwiggle(k_mean: usize, k_noise: usize, kwiggle: usize) -> Self {
        Self {
            k_mean,
            k_noise,
            kwiggle,
        }
    }

    fn total(self) -> usize {
        self.k_mean + self.k_noise + self.kwiggle
    }

    fn mean_end(self) -> usize {
        self.k_mean
    }

    fn noise_start(self) -> usize {
        self.k_mean
    }

    fn noise_end(self) -> usize {
        self.k_mean + self.k_noise
    }

    fn wiggle_start(self) -> usize {
        self.k_mean + self.k_noise
    }

    fn wiggle_end(self) -> usize {
        self.k_mean + self.k_noise + self.kwiggle
    }

    fn validate_theta_len(self, theta_len: usize, context: &str) -> Result<(), String> {
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

    fn mean_from(self, theta: &Array1<f64>) -> Array1<f64> {
        theta.slice(s![0..self.mean_end()]).to_owned()
    }

    fn noise_from(self, theta: &Array1<f64>) -> Array1<f64> {
        theta
            .slice(s![self.noise_start()..self.noise_end()])
            .to_owned()
    }

    fn wiggle_from(self, theta: &Array1<f64>) -> Array1<f64> {
        theta
            .slice(s![self.wiggle_start()..self.wiggle_end()])
            .to_owned()
    }
}


#[derive(Clone, Copy)]
struct GamlssBetaLayout {
    pt: usize,
    pls: usize,
    pw: usize,
}


impl GamlssBetaLayout {
    fn withwiggle(pt: usize, pls: usize, pw: usize) -> Self {
        Self { pt, pls, pw }
    }

    fn total(self) -> usize {
        self.pt + self.pls + self.pw
    }

    fn split_three(
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


const DEFAULT_GAUGE_PRIORITY: u8 = 100;

const LINK_WIGGLE_GAUGE_PRIORITY: u8 = 80;


fn initial_log_lambdas_orzeros(block: &ParameterBlockInput) -> Result<Array1<f64>, String> {
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


fn build_two_block_exact_joint_setup(
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


fn gaussian_location_scalewarm_start(
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
const LOCATION_SCALE_N_OUTPUTS: usize = 2;


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
fn build_location_scale_block(
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
fn build_location_scale_wiggle_block(
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


fn prepared_gaussian_log_sigma_design(
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


fn identified_binomial_log_sigma_design(
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


fn identity_penalty(dim: usize) -> Array2<f64> {
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
fn penalty_nullspace_projector(penalties: &[PenaltyMatrix], dim: usize) -> Array2<f64> {
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


fn append_binomial_log_sigma_shrinkage_penalty_design(design: &mut TermCollectionDesign) {
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
fn build_gaussian_mean_and_scale_blocks(
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
fn build_binomial_threshold_and_scale_blocks(
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
fn wiggle_block_penalty_matrices(wiggle_block: &ParameterBlockInput) -> Vec<PenaltyMatrix> {
    let p_wiggle = wiggle_block.design.ncols();
    wiggle_block
        .penalties
        .iter()
        .map(|spec| match spec {
            crate::solver::estimate::PenaltySpec::Block {
                local, col_range, ..
            } => PenaltyMatrix::Blockwise {
                local: local.clone(),
                col_range: col_range.clone(),
                total_dim: p_wiggle,
            },
            crate::solver::estimate::PenaltySpec::Dense(m)
            | crate::solver::estimate::PenaltySpec::DenseWithMean { matrix: m, .. } => {
                PenaltyMatrix::Dense(m.clone())
            }
        })
        .collect()
}


fn binomial_location_scale_link_eta_from_probability(
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


fn weighted_binomial_prevalence(y: &Array1<f64>, weights: &Array1<f64>) -> Result<f64, String> {
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


fn project_constant_eta_into_block(
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
fn binomial_location_scalewarm_start(
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
struct BinomialMeanWiggleSpec {
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


struct BlockwiseTermWiggleFitResultParts {
    pub fit: BlockwiseTermFitResult,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
}


fn validate_term_collection_design(
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

    fn validate_numeric_finiteness(&self) -> Result<(), String> {
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
    fn try_from_parts(parts: BlockwiseTermWiggleFitResultParts) -> Result<Self, String> {
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


fn fit_binomial_mean_wiggle(
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
        policy: crate::resource::ResourcePolicy::default_library(),
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


trait LocationScaleFamilyBuilder {
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


fn fit_location_scale_terms<B: LocationScaleFamilyBuilder>(
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
                    use crate::solver::estimate::reml::unified::EvalMode;
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
                            let subsample = crate::families::marginal_slope_shared::
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
                |_beta: &Array1<f64>| Ok(()),
            )
        }};
    }

    let mut solved = run_exact_joint_spatial!()
        .map_err(|err| format!("exact two-block spatial optimization failed: {err}"))?;

    // #1082 gaulss under-recovery diagnostic: surface the REML-selected per-block
    // smoothing parameter (log-λ = ρ) and the realized effective df, split into
    // the MEAN block's penalties and the SCALE (log-σ) block's penalties. If the
    // mean tensor is over-smoothed (RMSE(mu) too high), this prints a large
    // mean-block ρ and a small mean EDF — pinpointing whether the joint
    // location-scale REML criterion selects the mean-λ too high. Penalty order in
    // `lambdas` / `edf_by_block` is mean-block penalties first, then scale-block
    // (the build order in `build_two_block_exact_joint_setup`); we split at the
    // mean design's penalty count.
    {
        let n_mean_pen = solved.designs[0].penalties.len();
        let lambdas = solved.fit.lambdas.as_slice().unwrap_or(&[]);
        let log_lambdas = solved.fit.log_lambdas.as_slice().unwrap_or(&[]);
        let edf = solved
            .fit
            .inference
            .as_ref()
            .map(|inf| inf.edf_by_block.as_slice())
            .unwrap_or(&[]);
        let fmt = |s: &[f64]| -> String {
            s.iter()
                .map(|v| format!("{v:.4}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let split = |s: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let cut = n_mean_pen.min(s.len());
            (s[..cut].to_vec(), s[cut..].to_vec())
        };
        let (mean_rho, scale_rho) = split(log_lambdas);
        let (mean_lam, scale_lam) = split(lambdas);
        let (mean_edf, scale_edf) = split(edf);
        log::debug!(
            "[gaulss-lambda-diag] n_mean_pen={n_mean_pen} reml_score={:.6e}\n  \
             MEAN  rho(log-lambda)=[{}] lambda=[{}] edf=[{}]\n  \
             SCALE rho(log-lambda)=[{}] lambda=[{}] edf=[{}]",
            solved.fit.reml_score,
            fmt(&mean_rho),
            fmt(&mean_lam),
            fmt(&mean_edf),
            fmt(&scale_rho),
            fmt(&scale_lam),
            fmt(&scale_edf),
        );
    }

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


struct GaussianLocationScaleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    meanspec: TermCollectionSpec,
    noisespec: TermCollectionSpec,
    mean_offset: Array1<f64>,
    noise_offset: Array1<f64>,
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
            policy: crate::resource::ResourcePolicy::default_library(),
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


struct GaussianLocationScaleWiggleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    meanspec: TermCollectionSpec,
    noisespec: TermCollectionSpec,
    mean_offset: Array1<f64>,
    noise_offset: Array1<f64>,
    wiggle_knots: Array1<f64>,
    wiggle_degree: usize,
    wiggle_block: ParameterBlockInput,
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
            "GaussianLocationScaleWiggle::build_blocks",
        )?;
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
            policy: crate::resource::ResourcePolicy::default_library(),
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


struct BinomialLocationScaleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    link_kind: InverseLink,
    meanspec: TermCollectionSpec,
    noisespec: TermCollectionSpec,
    mean_offset: Array1<f64>,
    noise_offset: Array1<f64>,
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
            policy: crate::resource::ResourcePolicy::default_library(),
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


struct BinomialLocationScaleWiggleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    link_kind: InverseLink,
    meanspec: TermCollectionSpec,
    noisespec: TermCollectionSpec,
    mean_offset: Array1<f64>,
    noise_offset: Array1<f64>,
    wiggle_knots: Array1<f64>,
    wiggle_degree: usize,
    wiggle_block: ParameterBlockInput,
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
            "BinomialLocationScaleWiggle::build_blocks",
        )?;
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
            policy: crate::resource::ResourcePolicy::default_library(),
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
                        .map(crate::solver::estimate::PenaltySpec::from_blockwise_ref)
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
                    .map(crate::solver::estimate::PenaltySpec::from_blockwise_ref)
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
        policy: crate::resource::ResourcePolicy::default_library(),
    };
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let mut outer_options = options.clone();
    outer_options.screening_max_inner_iterations = Some(Arc::clone(&screening_cap));
    struct MeanWiggleOuterState {
        warm_cache: Option<crate::custom_family::CustomFamilyWarmStart>,
        last_eval: Option<(
            Array1<f64>,
            f64,
            Array1<f64>,
            crate::solver::outer_strategy::HessianResult,
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
                            crate::solver::estimate::PenaltySpec::Block {
                                local,
                                col_range,
                                ..
                            } => PenaltyMatrix::Blockwise {
                                local: local.clone(),
                                col_range: col_range.clone(),
                                total_dim: p_wiggle,
                            },
                            crate::solver::estimate::PenaltySpec::Dense(m)
                            | crate::solver::estimate::PenaltySpec::DenseWithMean {
                                matrix: m,
                                ..
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
                crate::solver::estimate::reml::unified::EvalMode::ValueGradientHessian
            } else {
                crate::solver::estimate::reml::unified::EvalMode::ValueAndGradient
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

    use crate::estimate::EstimationError;
    use crate::solver::outer_strategy::{
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
    let problem = crate::solver::outer_strategy::OuterProblem::new(theta_dim)
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
                    crate::solver::outer_strategy::HessianResult::Analytic(_)
                        | crate::solver::outer_strategy::HessianResult::Operator(_)
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
                    .map(crate::solver::estimate::PenaltySpec::from_blockwise_ref)
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


struct BinomialLocationScaleCore {
    sigma: Array1<f64>,
    dsigma_deta: Array1<f64>,
    q0: Array1<f64>,
    mu: Array1<f64>,
    dmu_dq: Array1<f64>,
    d2mu_dq2: Array1<f64>,
    d3mu_dq3: Array1<f64>,
    log_likelihood: f64,
}


#[derive(Clone, Copy)]
struct NonWiggleQDerivs {
    q_t: f64,
    q_ls: f64,
    q_tl: f64,
    q_ll: f64,
    q_tl_ls: f64,
    q_ll_ls: f64,
}


#[derive(Clone, Copy)]
struct NonWiggleQDirectional {
    delta_q: f64,
    delta_q_t: f64,
    delta_q_ls: f64,
    delta_q_tl: f64,
    delta_q_ll: f64,
}


#[derive(Clone, Copy)]
struct BinomialLocationScaleRow {
    sigma: f64,
    dsigma_deta: f64,
    q0: f64,
    inverse_link: crate::mixture_link::InverseLinkJet,
    ll: f64,
}


/// Non-wiggle location-scale map derivatives via shared scalar core.
fn nonwiggle_q_derivs(eta_t: f64, sigma: f64) -> NonWiggleQDerivs {
    let inv_sigma = sigma.recip();
    let q_t = -inv_sigma;
    let q_ls = eta_t * inv_sigma;
    let q_tl = inv_sigma;
    let q_ll = -eta_t * inv_sigma;
    let q_tl_ls = -inv_sigma;
    let q_ll_ls = eta_t * inv_sigma;
    NonWiggleQDerivs {
        q_t,
        q_ls,
        q_tl,
        q_ll,
        q_tl_ls,
        q_ll_ls,
    }
}


/// Directional derivatives along (d_eta_t, d_eta_ls):
/// delta_q = q_t d_eta_t + q_ls d_eta_ls
/// delta_q_t = q_tl d_eta_ls
/// delta_q_ls = q_tl d_eta_t + q_ll d_eta_ls
/// delta_q_tt = 0
/// delta_q_tl = q_tl_ls d_eta_ls
/// delta_q_ll = q_tl_ls d_eta_t + q_ll_ls d_eta_ls
fn nonwiggle_q_directional(
    q: NonWiggleQDerivs,
    d_eta_t: f64,
    d_eta_ls: f64,
) -> NonWiggleQDirectional {
    // Directional-chain derivation:
    //
    // For any scalar f(eta_t,eta_ls), directional derivative along
    // d eta = (d_eta_t, d_eta_ls) is
    //   dot{f} = f_t d_eta_t + f_ls d_eta_ls.
    //
    // Apply to q and its eta-partials:
    //   dot{q}      = q_t d_eta_t + q_ls d_eta_ls.
    //   dot{q_t}    = q_tt d_eta_t + q_tl d_eta_ls = q_tl d_eta_ls (q_tt=0).
    //   dot{q_ls}   = q_tl d_eta_t + q_ll d_eta_ls.
    //   dot{q_tt}   = 0.
    //   dot{q_tl}   = q_tl_ls d_eta_ls.
    //   dot{q_ll}   = q_tl_ls d_eta_t + q_ll_ls d_eta_ls.
    NonWiggleQDirectional {
        delta_q: q.q_t * d_eta_t + q.q_ls * d_eta_ls,
        delta_q_t: q.q_tl * d_eta_ls,
        delta_q_ls: q.q_tl * d_eta_t + q.q_ll * d_eta_ls,
        delta_q_tl: q.q_tl_ls * d_eta_ls,
        delta_q_ll: q.q_tl_ls * d_eta_t + q.q_ll_ls * d_eta_ls,
    }
}


#[inline]
fn log1mexp_neg_positive(z: f64) -> f64 {
    assert!(z >= 0.0);
    if z == 0.0 {
        f64::NEG_INFINITY
    } else if z <= std::f64::consts::LN_2 {
        (-(-z).exp_m1()).ln()
    } else {
        (1.0 - (-z).exp()).ln()
    }
}


#[inline]
fn bernoulli_log_likelihood_from_probability(y: f64, weight: f64, mu: f64) -> Result<f64, String> {
    if weight == 0.0 {
        return Ok(0.0);
    }
    if !mu.is_finite() || !(0.0..=1.0).contains(&mu) {
        return Err(GamlssError::NumericalFailure {
            reason: format!(
                "binomial location-scale inverse link returned invalid probability {mu}"
            ),
        }
        .into());
    }
    let log_mu = if mu == 0.0 {
        if y == 0.0 { 0.0 } else { f64::NEG_INFINITY }
    } else {
        mu.ln()
    };
    let log_one_minus = if mu == 1.0 {
        if y == 1.0 { 0.0 } else { f64::NEG_INFINITY }
    } else {
        (1.0 - mu).ln()
    };
    let ll = weight * (y * log_mu + (1.0 - y) * log_one_minus);
    if ll.is_finite() {
        Ok(ll)
    } else {
        Err(GamlssError::NonFinite {
            reason: format!(
                "binomial location-scale log likelihood is non-finite at y={y}, mu={mu}"
            ),
        }
        .into())
    }
}


#[inline]
fn binomial_location_scale_q0(eta_t: f64, sigma: f64) -> f64 {
    -eta_t / sigma
}


#[inline]
fn binomial_location_scale_log_likelihood(
    y: f64,
    weight: f64,
    q: f64,
    link_kind: &InverseLink,
    mu: f64,
) -> Result<f64, String> {
    if weight == 0.0 {
        return Ok(0.0);
    }
    match link_kind {
        InverseLink::Standard(StandardLink::Probit) => {
            Ok(weight * (y * normal_logcdf(q) + (1.0_f64 - y) * normal_logsf(q)))
        }
        InverseLink::Standard(StandardLink::Logit) => Ok(weight
            * (-y * crate::linalg::utils::stable_softplus(-q)
                - (1.0_f64 - y) * crate::linalg::utils::stable_softplus(q))),
        InverseLink::Standard(StandardLink::CLogLog) => {
            let z = q.exp();
            let log_p = if z == 0.0 {
                q
            } else if z.is_infinite() {
                0.0
            } else {
                log1mexp_neg_positive(z)
            };
            let log_survival = -z;
            let ll = weight * (y * log_p + (1.0_f64 - y) * log_survival);
            if ll.is_finite() {
                Ok(ll)
            } else {
                Err(GamlssError::NonFinite { reason: format!(
                    "binomial cloglog location-scale log likelihood is non-finite at y={y}, q={q}"
                ) }.into())
            }
        }
        _ => bernoulli_log_likelihood_from_probability(y, weight, mu),
    }
}


#[inline]
fn binomial_expected_q_information_derivatives(
    weight: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
) -> (f64, f64, f64) {
    if weight == 0.0
        || !mu.is_finite()
        || !d1.is_finite()
        || !d2.is_finite()
        || !d3.is_finite()
        || mu <= 0.0
        || mu >= 1.0
        || d1 == 0.0
    {
        return (0.0, 0.0, 0.0);
    }
    let var = mu * (1.0 - mu);
    if !var.is_finite() || var <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let var1 = d1 * (1.0 - 2.0 * mu);
    let var2 = d2 * (1.0 - 2.0 * mu) - 2.0 * d1 * d1;

    let f = weight * d1 * d1 / var;
    let num1 = 2.0 * d1 * d2 * var - d1 * d1 * var1;
    let f1 = weight * num1 / (var * var);
    let num1_prime = 2.0 * (d2 * d2 + d1 * d3) * var - d1 * d1 * var2;
    let f2 = weight * (num1_prime / (var * var) - 2.0 * num1 * var1 / (var * var * var));
    if f.is_finite() && f1.is_finite() && f2.is_finite() {
        (f, f1, f2)
    } else {
        (0.0, 0.0, 0.0)
    }
}


fn binomial_expected_location_scale_second_coefficients(
    q: NonWiggleQDerivs,
    f: f64,
    f1: f64,
    f2: f64,
    d_eta_t_u: f64,
    d_eta_ls_u: f64,
    d_eta_t_v: f64,
    d_eta_ls_v: f64,
) -> (f64, f64, f64) {
    let u = nonwiggle_q_directional(q, d_eta_t_u, d_eta_ls_u);
    let v = nonwiggle_q_directional(q, d_eta_t_v, d_eta_ls_v);
    let q_uv = q.q_tl * (d_eta_t_u * d_eta_ls_v + d_eta_t_v * d_eta_ls_u)
        + q.q_ll * d_eta_ls_u * d_eta_ls_v;
    let q_t_uv = q.q_tl_ls * d_eta_ls_u * d_eta_ls_v;
    let q_ls_uv = q.q_tl_ls * (d_eta_ls_u * d_eta_t_v + d_eta_ls_v * d_eta_t_u)
        + q.q_ll_ls * d_eta_ls_u * d_eta_ls_v;
    let scalar = f2 * u.delta_q * v.delta_q + f1 * q_uv;
    let tt = scalar * q.q_t * q.q_t
        + 2.0 * f1 * u.delta_q * q.q_t * v.delta_q_t
        + 2.0 * f1 * v.delta_q * q.q_t * u.delta_q_t
        + 2.0 * f * (q.q_t * q_t_uv + u.delta_q_t * v.delta_q_t);
    let tl = scalar * q.q_t * q.q_ls
        + f1 * u.delta_q * (v.delta_q_t * q.q_ls + q.q_t * v.delta_q_ls)
        + f1 * v.delta_q * (u.delta_q_t * q.q_ls + q.q_t * u.delta_q_ls)
        + f * (q_t_uv * q.q_ls
            + q.q_t * q_ls_uv
            + u.delta_q_t * v.delta_q_ls
            + v.delta_q_t * u.delta_q_ls);
    let ll = scalar * q.q_ls * q.q_ls
        + 2.0 * f1 * u.delta_q * q.q_ls * v.delta_q_ls
        + 2.0 * f1 * v.delta_q * q.q_ls * u.delta_q_ls
        + 2.0 * f * (q.q_ls * q_ls_uv + u.delta_q_ls * v.delta_q_ls);
    (tt, tl, ll)
}


fn binomial_location_scalerow(
    y: f64,
    weight: f64,
    eta_t: f64,
    eta_ls: f64,
    etawiggle: f64,
    link_kind: &InverseLink,
) -> Result<BinomialLocationScaleRow, String> {
    let SigmaJet1 {
        sigma,
        d1: dsigma_deta,
    } = exp_sigma_jet1_scalar(eta_ls);
    let q0 = binomial_location_scale_q0(eta_t, sigma);
    let q = q0 + etawiggle;
    let jet = inverse_link_jet_for_inverse_link(link_kind, q)
        .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
    let raw_mu = jet.mu;
    // μ is stored RAW (unclamped). The q-derivative tower built downstream
    // (binomial_neglog_q_derivatives_dispatch et al.) is the EXACT derivative
    // of the loss evaluated here, computed via the per-branch reciprocals in
    // `binomial_loglik_mu_derivatives` plus the saturation guard in the
    // `*_from_jet` consumers. Flooring μ at MIN_PROB here would replace every
    // representable sub-MIN_PROB tail probability with a 1e-10 surrogate,
    // corrupting the Fisher curvature throughout the saturated tail (#948).
    // The inverse-link derivatives d1/d2/d3 carry the legitimate gradient
    // signal and are likewise preserved.
    let inverse_link = jet;
    let ll = binomial_location_scale_log_likelihood(y, weight, q, link_kind, raw_mu)?;
    Ok(BinomialLocationScaleRow {
        sigma,
        dsigma_deta,
        q0,
        inverse_link,
        ll,
    })
}


/// Compute only the log-likelihood scalar for the binomial location-scale model.
/// This avoids allocating 7 n-vectors that `binomial_location_scale_core` would produce,
/// making backtracking line searches much cheaper at large scale.
fn binomial_location_scale_ll_only(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    etawiggle: Option<&Array1<f64>>,
    link_kind: &InverseLink,
) -> Result<f64, String> {
    let n = y.len();
    let y_slice = y.as_slice().expect("y must be contiguous");
    let w_slice = weights.as_slice().expect("weights must be contiguous");
    let et_slice = eta_t.as_slice().expect("eta_t must be contiguous");
    let el_slice = eta_ls.as_slice().expect("eta_ls must be contiguous");
    let ew_slice = etawiggle.map(|w| w.as_slice().expect("etawiggle must be contiguous"));
    (0..n)
        .into_par_iter()
        .try_fold(
            || 0.0_f64,
            |acc, i| -> Result<f64, String> {
                let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(el_slice[i]);
                let q0 = binomial_location_scale_q0(et_slice[i], sigma);
                let q = q0 + ew_slice.map_or(0.0, |w| w[i]);
                if matches!(link_kind, InverseLink::Standard(StandardLink::Probit)) {
                    return Ok(acc
                        + binomial_location_scale_log_likelihood(
                            y_slice[i], w_slice[i], q, link_kind, 0.5,
                        )?);
                }
                let jet = inverse_link_jet_for_inverse_link(link_kind, q)
                    .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
                Ok(acc
                    + binomial_location_scale_log_likelihood(
                        y_slice[i], w_slice[i], q, link_kind, jet.mu,
                    )?)
            },
        )
        .try_reduce(|| 0.0_f64, |a, b| Ok(a + b))
}


fn binomial_location_scale_core(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    etawiggle: Option<&Array1<f64>>,
    link_kind: &InverseLink,
) -> Result<BinomialLocationScaleCore, String> {
    let n = y.len();
    if weights.len() != n || eta_t.len() != n || eta_ls.len() != n {
        return Err(GamlssError::DimensionMismatch {
            reason: "binomial location-scale core size mismatch".to_string(),
        }
        .into());
    }
    if let Some(w) = etawiggle
        && w.len() != n
    {
        return Err(GamlssError::DimensionMismatch {
            reason: "binomial location-scale core wiggle size mismatch".to_string(),
        }
        .into());
    }

    // Parallel per-row probit/inverse-link evaluation. At large scale
    // (n = 320K) the sequential probit erfc loop was a major single-thread
    // hotspot called dozens of times per outer REML gradient evaluation.
    let y_slice = y.as_slice().expect("y must be contiguous");
    let w_slice = weights.as_slice().expect("weights must be contiguous");
    let et_slice = eta_t.as_slice().expect("eta_t must be contiguous");
    let el_slice = eta_ls.as_slice().expect("eta_ls must be contiguous");
    let ew_slice = etawiggle.map(|w| w.as_slice().expect("etawiggle must be contiguous"));

    // Write each row's seven scalar derivatives directly into preallocated
    // output buffers in parallel, reducing the per-row log-likelihood
    // alongside. The previous path collected a `Vec<BinomialLocationScaleRow>`
    // (8 scalar fields plus alignment) and then serially scattered into the
    // seven `Array1`s, which at large scale n=3e5 cost ~50 MB of transient
    // allocation and a single-threaded post-pass.
    let mut sigma = vec![0.0_f64; n];
    let mut dsigma_deta = vec![0.0_f64; n];
    let mut q0 = vec![0.0_f64; n];
    let mut mu = vec![0.0_f64; n];
    let mut dmu_dq = vec![0.0_f64; n];
    let mut d2mu_dq2 = vec![0.0_f64; n];
    let mut d3mu_dq3 = vec![0.0_f64; n];

    /// Wrapper to send raw pointers across threads for disjoint per-row writes.
    /// Each parallel iteration writes to a unique index `i`, and the caller
    /// ensures the pointers outlive the parallel region (see SAFETY: notes
    /// on each `unsafe` site below).
    #[derive(Clone, Copy)]
    struct SendPtr(*mut f64);
    // SAFETY: pointers are constructed from live writable buffers and used
    // only for disjoint per-row writes inside a bounded parallel region; the
    // owning `Vec`s outlive the region.
    unsafe impl Send for SendPtr {}
    // SAFETY: same live-buffer and disjoint-index invariants as `Send`; no
    // two threads write the same offset through any shared `SendPtr` value.
    unsafe impl Sync for SendPtr {}
    impl SendPtr {
        #[inline(always)]
        // SAFETY: `self.0` points to a live writable allocation with length
        // greater than `i`, and `i` is exclusively owned by the calling
        // parallel iteration.
        unsafe fn write(self, i: usize, v: f64) {
            // SAFETY: see `write`'s function-level note: `i` is in-bounds
            // and exclusively owned by this iteration.
            unsafe { *self.0.add(i) = v };
        }
    }

    let sigma_p = SendPtr(sigma.as_mut_ptr());
    let dsigma_p = SendPtr(dsigma_deta.as_mut_ptr());
    let q0_p = SendPtr(q0.as_mut_ptr());
    let mu_p = SendPtr(mu.as_mut_ptr());
    let dmu_p = SendPtr(dmu_dq.as_mut_ptr());
    let d2mu_p = SendPtr(d2mu_dq2.as_mut_ptr());
    let d3mu_p = SendPtr(d3mu_dq3.as_mut_ptr());

    let ll = (0..n)
        .into_par_iter()
        .map(move |i| {
            let row = binomial_location_scalerow(
                y_slice[i],
                w_slice[i],
                et_slice[i],
                el_slice[i],
                ew_slice.map_or(0.0, |w| w[i]),
                link_kind,
            )?;
            // SAFETY: `i` comes from `0..n`, so it is in-bounds for each
            // preallocated length-`n` buffer, and every index is produced once;
            // each pointer targets a distinct output buffer.
            unsafe {
                sigma_p.write(i, row.sigma);
                dsigma_p.write(i, row.dsigma_deta);
                q0_p.write(i, row.q0);
                mu_p.write(i, row.inverse_link.mu);
                dmu_p.write(i, row.inverse_link.d1);
                d2mu_p.write(i, row.inverse_link.d2);
                d3mu_p.write(i, row.inverse_link.d3);
            }
            Ok::<f64, String>(row.ll)
        })
        .try_reduce(|| 0.0_f64, |a, b| Ok(a + b))?;

    Ok(BinomialLocationScaleCore {
        sigma: Array1::from_vec(sigma),
        dsigma_deta: Array1::from_vec(dsigma_deta),
        q0: Array1::from_vec(q0),
        mu: Array1::from_vec(mu),
        dmu_dq: Array1::from_vec(dmu_dq),
        d2mu_dq2: Array1::from_vec(d2mu_dq2),
        d3mu_dq3: Array1::from_vec(d3mu_dq3),
        log_likelihood: ll,
    })
}


#[inline]
fn binomial_location_scale_nll_tower(
    y: f64,
    weight: f64,
    eta_t: f64,
    eta_ls: f64,
    q_value: f64,
    mu: f64,
    dmu_dq: f64,
    d2mu_dq2: f64,
    d3mu_dq3: f64,
    link_kind: &InverseLink,
    include_fourth: bool,
) -> Result<crate::families::jet_tower::Tower4<2>, String> {
    use crate::families::jet_tower::Tower4;
    let eta_t_tower = Tower4::<2>::variable(eta_t, 0);
    let eta_ls_tower = Tower4::<2>::variable(eta_ls, 1);
    let inv_sigma = (eta_ls_tower * -1.0).exp();
    let q = -eta_t_tower * inv_sigma;
    let ll = binomial_location_scale_log_likelihood(y, weight, q_value, link_kind, mu)?;
    let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
        y, weight, q_value, mu, dmu_dq, d2mu_dq2, d3mu_dq3, link_kind,
    );
    let m4 = if include_fourth {
        binomial_neglog_q_fourth_derivative_dispatch(
            y, weight, q_value, mu, dmu_dq, d2mu_dq2, d3mu_dq3, link_kind,
        )?
    } else {
        0.0
    };
    Ok(q.compose_unary([-ll, m1, m2, m3, m4]))
}


#[inline]
fn binomial_location_scale_nll_tower_from_core_row(
    y: f64,
    weight: f64,
    core: &BinomialLocationScaleCore,
    row: usize,
    link_kind: &InverseLink,
    include_fourth: bool,
) -> Result<crate::families::jet_tower::Tower4<2>, String> {
    let sigma = core.sigma[row];
    let eta_t = -core.q0[row] * sigma;
    let eta_ls = sigma.ln();
    binomial_location_scale_nll_tower(
        y,
        weight,
        eta_t,
        eta_ls,
        core.q0[row],
        core.mu[row],
        core.dmu_dq[row],
        core.d2mu_dq2[row],
        core.d3mu_dq3[row],
        link_kind,
        include_fourth,
    )
}


/// Pure row-coefficient builder for the binomial location-scale joint
/// directional derivative `D_β H_L[u]`. Returns `(c_tt, c_tl, c_ll)` such
/// that the resulting matrix is
///
///   X_t^T diag(c_tt) X_t + X_t^T diag(c_tl) X_ls (+ symmetric)
///   + X_ls^T diag(c_ll) X_ls.
///
/// Inputs `d_eta_t = X_t · u_t`, `d_eta_ls = X_ls · u_ls` are the linear
/// predictor perturbations along the joint direction `u = (u_t, u_ls)`.
fn binomial_location_scale_first_directional_coefficients(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    core: &BinomialLocationScaleCore,
    d_eta_t: &Array1<f64>,
    d_eta_ls: &Array1<f64>,
    link_kind: &InverseLink,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    let n = y.len();
    let triples: Result<Vec<(f64, f64, f64)>, String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let tower = binomial_location_scale_nll_tower_from_core_row(
                y[i], weights[i], core, i, link_kind, false,
            )?;
            let dir = [d_eta_t[i], d_eta_ls[i]];
            let contracted = tower.third_contracted(&dir);
            Ok((contracted[0][0], contracted[0][1], contracted[1][1]))
        })
        .collect();
    let mut coeff_tt = Array1::<f64>::zeros(n);
    let mut coeff_tl = Array1::<f64>::zeros(n);
    let mut coeff_ll = Array1::<f64>::zeros(n);
    for (i, (tt, tl, ll)) in triples?.into_iter().enumerate() {
        coeff_tt[i] = tt;
        coeff_tl[i] = tl;
        coeff_ll[i] = ll;
    }
    Ok((coeff_tt, coeff_tl, coeff_ll))
}


/// Pure row-coefficient builder for the binomial location-scale joint
/// second directional derivative `D²_β H_L[u, v]`. Returns
/// `(c_tt, c_tl, c_ll)` analogous to the first-order helper but built from
/// the four predictor perturbations `(d_eta_t_u, d_eta_ls_u, d_eta_t_v,
/// d_eta_ls_v)`.
fn binomial_location_scalesecond_directional_coefficients(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    core: &BinomialLocationScaleCore,
    d_eta_t_u: &Array1<f64>,
    d_eta_ls_u: &Array1<f64>,
    d_eta_t_v: &Array1<f64>,
    d_eta_ls_v: &Array1<f64>,
    link_kind: &InverseLink,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = y.len();
    // Per-row second-directional coefficient computation. m4 dispatch
    // can fail (Result), so collect a Result<Vec<(tt, tl, ll)>>.
    let triples: Result<Vec<(f64, f64, f64)>, String> = (0..n)
        .into_par_iter()
        .map(|i| -> Result<(f64, f64, f64), String> {
            let tower = binomial_location_scale_nll_tower_from_core_row(
                y[i], weights[i], core, i, link_kind, true,
            )?;
            let dir_u = [d_eta_t_u[i], d_eta_ls_u[i]];
            let dir_v = [d_eta_t_v[i], d_eta_ls_v[i]];
            let contracted = tower.fourth_contracted(&dir_u, &dir_v);
            Ok((contracted[0][0], contracted[0][1], contracted[1][1]))
        })
        .collect();
    let triples = triples?;
    let mut coeff_tt = Array1::<f64>::zeros(n);
    let mut coeff_tl = Array1::<f64>::zeros(n);
    let mut coeff_ll = Array1::<f64>::zeros(n);
    for (i, (tt, tl, ll)) in triples.into_iter().enumerate() {
        coeff_tt[i] = tt;
        coeff_tl[i] = tl;
        coeff_ll[i] = ll;
    }
    Ok((coeff_tt, coeff_tl, coeff_ll))
}


/// Built-in Gaussian location-scale family:
/// - Block 0: location μ(·) with identity link
/// - Block 1: log-scale log σ(·) with log link
pub struct GaussianLocationScaleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub mu_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
    /// Cached per-observation row scalars keyed by 6-element fingerprint
    /// (first, mid, last elements of both eta vectors).
    /// Avoids recomputing O(n) scalars K+ times per REML gradient/Hessian evaluation.
    pub cached_row_scalars:
        std::sync::RwLock<Option<(f64, f64, f64, f64, f64, f64, Arc<GaussianJointRowScalars>)>>,
}


impl Clone for GaussianLocationScaleFamily {
    fn clone(&self) -> Self {
        Self {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: self.mu_design.clone(),
            log_sigma_design: self.log_sigma_design.clone(),
            policy: self.policy.clone(),
            cached_row_scalars: std::sync::RwLock::new(
                self.cached_row_scalars
                    .read()
                    .expect("lock poisoned")
                    .clone(),
            ),
        }
    }
}


struct LocationScaleJointPsiDirection {
    block_idx: usize,
    local_idx: usize,
    x_primary_psi: PsiDesignMap,
    x_ls_psi: PsiDesignMap,
    z_primary_psi: Array1<f64>,
    z_ls_psi: Array1<f64>,
}


struct LocationScaleJointPsiSecondDrifts {
    x_primary_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_ls_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_primary_ab: Option<Array2<f64>>,
    x_ls_ab: Option<Array2<f64>>,
    z_primary_ab: Array1<f64>,
    z_ls_ab: Array1<f64>,
}


/// Shared interface that the Gaussian and Binomial location-scale families (and
/// their wiggle variants) expose to the unified joint ψ workspace.
///
/// The four families are structurally identical at the workspace level: each
/// owns two dense block designs (location + log-scale), produces a per-ψ
/// direction, and assembles second-order ψ terms and a ψ-Hessian directional
/// derivative from those parts. They differ only in (1) the concrete
/// [`Direction`](Self::Direction) struct produced (Gaussian vs Binomial field
/// names), (2) the family-name fragment in the dense-designs error message, and
/// (3) whether an optional Horvitz–Thompson outer-row subsample is threaded
/// into the per-row weight arrays (Gaussian does; Binomial ignores it and runs
/// the full-data exact path). This single trait gives the generic
/// [`LocationScaleJointPsiWorkspace`] one dispatch surface; each family's impl
/// is a thin delegation to inherent methods it already owns.
trait LocationScaleJointPsiFamily: Clone + Send + Sync + 'static {
    /// Per-ψ joint direction produced by this family.
    type Direction: Send + Sync + 'static;

    /// Family-name fragment used in the workspace's dense-designs error
    /// message so the originating family stays visible after unification.
    const LABEL: &'static str;

    fn ws_policy(&self) -> &crate::resource::ResourcePolicy;

    fn ws_exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String>;

    fn ws_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<Self::Direction>, String>;

    fn ws_psi_second_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &Self::Direction,
        psi_b: &Self::Direction,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String>;

    fn ws_psi_hessian_directional_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        psi_dir: &Self::Direction,
        d_beta_flat: &Array1<f64>,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<Array2<f64>, String>;
}


impl LocationScaleJointPsiFamily for GaussianLocationScaleFamily {
    type Direction = LocationScaleJointPsiDirection;
    const LABEL: &'static str = "GaussianLocationScaleFamily";

    fn ws_policy(&self) -> &crate::resource::ResourcePolicy {
        &self.policy
    }

    fn ws_exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        self.exact_joint_dense_block_designs(specs)
    }

    fn ws_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            design_loc,
            design_scale,
            policy,
        )
    }

    fn ws_psi_second_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String> {
        self.exact_newton_joint_psisecond_order_terms_from_parts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            design_loc,
            design_scale,
            subsample,
        )
    }

    fn ws_psi_hessian_directional_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        psi_dir: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_from_parts(
            block_states,
            psi_dir,
            d_beta_flat,
            design_loc,
            design_scale,
            subsample,
        )
    }
}


impl LocationScaleJointPsiFamily for GaussianLocationScaleWiggleFamily {
    type Direction = LocationScaleJointPsiDirection;
    const LABEL: &'static str = "GaussianLocationScaleWiggleFamily";

    fn ws_policy(&self) -> &crate::resource::ResourcePolicy {
        &self.policy
    }

    fn ws_exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        self.exact_joint_dense_block_designs(specs)
    }

    fn ws_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            design_loc,
            design_scale,
            policy,
        )
    }

    fn ws_psi_second_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        outer_rows: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String> {
        assert!(outer_rows.map_or(true, |r| r.len() <= isize::MAX as usize));
        // Wiggle ψ path: full-data exact (= trivially unbiased). The
        // wiggle-specific second-order from-parts function inlines 30+
        // per-row coefficient arrays (`coeff_mm{,_a,_b,_ab}`,
        // `coeff_ml{,_a,_b,_ab}`, `coeff_ll{,_a,_b,_ab}`, `a{,_a,_b,_ab}`,
        // `c{,_a,_b,_ab}`, `l{,_a,_b,_ab}`, `dw_{a,b,ab}`, `s_mu*`, `s_ls*`,
        // `s_w*`, ...) instead of packing them into a struct like the
        // non-wiggle GLS path's `GaussianJointPsi{First,Second}Weights`.
        // Each is row-linear in `rows.{w,m,n,kappa,...}` and the direction
        // vectors so HT masking is theoretically clean, but threading a mask
        // across that many call sites is brittle (any missed array silently
        // biases the estimator). The outer score remains unbiased without
        // touching the wiggle ψ path: HT-unbiased LL
        // (`log_likelihood_only_with_options`) + HT-unbiased ρ-Hessian
        // (`exact_newton_joint_hessian_workspace_with_options`) +
        // exact-unbiased ψ (this path) = unbiased. Broadening to the wiggle
        // ψ path is a follow-up that should refactor the inline arrays into
        // `WiggleJointPsi{First,Second}Weights` structs mirroring
        // `GaussianJointPsi{First,Second}Weights` so a single
        // `apply_ht_mask_wiggle*` helper can mask everything in one place.
        self.exact_newton_joint_psisecond_order_terms_from_parts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            design_loc,
            design_scale,
        )
    }

    fn ws_psi_hessian_directional_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        psi_dir: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        outer_rows: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<Array2<f64>, String> {
        assert!(outer_rows.map_or(true, |r| r.len() <= isize::MAX as usize));
        // Same rationale as `ws_psi_second_order_terms_from_parts` above:
        // the wiggle ψ-Hessian directional-derivative function also inlines
        // dozens of per-row arrays. Full-data is exact (= trivially
        // unbiased), so the total outer score remains unbiased.
        self.exact_newton_joint_psihessian_directional_derivative_from_parts(
            block_states,
            psi_dir,
            d_beta_flat,
            design_loc,
            design_scale,
        )
    }
}


/// Generic joint exact-Newton ψ workspace shared by every location-scale
/// family (Gaussian / Binomial, with or without a wiggle block) via the
/// [`LocationScaleJointPsiFamily`] trait.
///
/// The workspace owns the two dense block designs as `Arc<Array2<f64>>` (the
/// per-family `ws_exact_joint_dense_block_designs` hands back a `Cow`, which is
/// materialized once here), the per-ψ direction cache, and an optional
/// Horvitz–Thompson outer-row subsample. When the subsample is `Some`, every
/// per-row weight array produced inside the second-order ψ Hessian and the
/// ψ-Hessian directional-derivative computations is masked: each sampled row's
/// contribution is scaled by `WeightedOuterRow.weight = 1/π_i` and non-sampled
/// rows are zeroed. Because every downstream assembly is row-linear in those
/// arrays, the resulting ψ score and ψ Hessian remain unbiased estimators of
/// the full-data quantities. Families that do not thread the subsample (the
/// Binomial families) construct with `new` and the field stays `None`.
struct LocationScaleJointPsiWorkspace<F: LocationScaleJointPsiFamily> {
    family: F,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    design_loc: Arc<Array2<f64>>,
    design_scale: Arc<Array2<f64>>,
    psi_directions: ExactNewtonJointPsiDirectCache<F::Direction>,
    outer_score_subsample: Option<Arc<crate::families::marginal_slope_shared::OuterScoreSubsample>>,
}


impl<F: LocationScaleJointPsiFamily> LocationScaleJointPsiWorkspace<F> {
    fn new(
        family: F,
        block_states: Vec<ParameterBlockState>,
        specs: &[ParameterBlockSpec],
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        Self::new_with_subsample(family, block_states, specs, derivative_blocks, None)
    }

    fn new_with_subsample(
        family: F,
        block_states: Vec<ParameterBlockState>,
        specs: &[ParameterBlockSpec],
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
        outer_score_subsample: Option<
            Arc<crate::families::marginal_slope_shared::OuterScoreSubsample>,
        >,
    ) -> Result<Self, String> {
        let Some((design_loc, design_scale)) =
            family.ws_exact_joint_dense_block_designs(Some(specs))?
        else {
            return Err(GamlssError::UnsupportedConfiguration {
                reason: format!(
                    "{} exact joint psi workspace requires dense block designs",
                    F::LABEL,
                ),
            }
            .into());
        };
        let design_loc = shared_dense_arc(design_loc.as_ref());
        let design_scale = shared_dense_arc(design_scale.as_ref());
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum();
        Ok(Self {
            family,
            block_states,
            derivative_blocks,
            design_loc,
            design_scale,
            psi_directions: ExactNewtonJointPsiDirectCache::new(psi_dim),
            outer_score_subsample,
        })
    }

    fn psi_direction(&self, psi_index: usize) -> Result<Option<Arc<F::Direction>>, String> {
        self.psi_directions.get_or_try_init(psi_index, || {
            self.family.ws_psi_direction(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                self.design_loc.as_ref(),
                self.design_scale.as_ref(),
                self.family.ws_policy(),
            )
        })
    }

    fn subsample_rows(
        &self,
    ) -> Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]> {
        self.outer_score_subsample
            .as_ref()
            .map(|s| s.rows.as_ref().as_slice())
    }
}


impl<F> ExactNewtonJointPsiWorkspace for LocationScaleJointPsiWorkspace<F>
where
    F: LocationScaleJointPsiFamily,
{
    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.psi_direction(psi_i)? else {
            return Ok(None);
        };
        let Some(dir_j) = self.psi_direction(psi_j)? else {
            return Ok(None);
        };
        Ok(Some(self.family.ws_psi_second_order_terms_from_parts(
            &self.block_states,
            &self.derivative_blocks,
            dir_i.as_ref(),
            dir_j.as_ref(),
            self.design_loc.as_ref(),
            self.design_scale.as_ref(),
            self.subsample_rows(),
        )?))
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        let Some(dir) = self.psi_direction(psi_index)? else {
            return Ok(None);
        };
        Ok(Some(
            crate::solver::estimate::reml::unified::DriftDerivResult::Dense(
                self.family.ws_psi_hessian_directional_from_parts(
                    &self.block_states,
                    dir.as_ref(),
                    d_beta_flat,
                    self.design_loc.as_ref(),
                    self.design_scale.as_ref(),
                    self.subsample_rows(),
                )?,
            ),
        ))
    }
}


type GaussianLocationScaleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<GaussianLocationScaleFamily>;

type GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<GaussianLocationScaleWiggleFamily>;


#[derive(Clone)]
pub struct GaussianJointRowScalars {
    obs_weight: Array1<f64>,
    w: Array1<f64>,
    m: Array1<f64>,
    n: Array1<f64>,
    /// κ = (dσ/dη_ls)/σ for the active sigma link.
    /// The cross Hessian block H_{μ,ls} carries an overall κ factor and the
    /// scale-scale block H_{ls,ls} carries κ².
    kappa: Array1<f64>,
    /// κ' = dκ/dη_ls = κ(1−κ) for the logb link. The static H_{ls,ls} block
    /// carries a κ'·(a−n) term, so κ' threads through every dH directional
    /// weight via the chain rule.
    kappa_prime: Array1<f64>,
    /// κ'' = κ(1−κ)(1−2κ); appears in d²H_{ls,ls} via the second
    /// η-derivative of κ'·(a−n).
    kappa_dprime: Array1<f64>,
}


struct GaussianJointPsiFirstWeights {
    objective_psirow: Array1<f64>,
    scoremu: Array1<f64>,
    score_ls: Array1<f64>,
    dscoremu: Array1<f64>,
    dscore_ls: Array1<f64>,
    hmumu: Array1<f64>,
    hmu_ls: Array1<f64>,
    h_ls_ls: Array1<f64>,
    dhmumu: Array1<f64>,
    dhmu_ls: Array1<f64>,
    dh_ls_ls: Array1<f64>,
}


struct GaussianJointPsiSecondWeights {
    objective_psi_psirow: Array1<f64>,
    d2scoremu: Array1<f64>,
    d2score_ls: Array1<f64>,
    d2hmumu: Array1<f64>,
    d2hmu_ls: Array1<f64>,
    d2h_ls_ls: Array1<f64>,
}


struct GaussianJointPsiMixedDriftWeights {
    dhmumu_u: Array1<f64>,
    dhmu_ls_u: Array1<f64>,
    dh_ls_ls_u: Array1<f64>,
    d2hmumu: Array1<f64>,
    d2hmu_ls: Array1<f64>,
    d2h_ls_ls: Array1<f64>,
}


/// Apply a Horvitz–Thompson outer-row subsample mask to every per-row array
/// of a `GaussianJointPsiFirstWeights` in place: each sampled row's
/// contribution is multiplied by `WeightedOuterRow.weight = 1/π_i` and all
/// non-sampled rows are zeroed. Every downstream assembly
/// (`gaussian_joint_psi*_fromweights`,
/// `build_two_block_custom_family_joint_psi_operator_from_actions`) consumes
/// these arrays row-linearly via `Xᵀ diag(W) Y` and `weighted_crossprod_psi_maps`,
/// so the resulting first-order ψ score and Hessian remain unbiased estimators
/// of the full-data quantities.
fn apply_ht_mask_first(
    weights: &mut GaussianJointPsiFirstWeights,
    rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
) {
    let n = weights.objective_psirow.len();
    let mut obj = Array1::<f64>::zeros(n);
    let mut smu = Array1::<f64>::zeros(n);
    let mut sls = Array1::<f64>::zeros(n);
    let mut dsmu = Array1::<f64>::zeros(n);
    let mut dsls = Array1::<f64>::zeros(n);
    let mut hmm = Array1::<f64>::zeros(n);
    let mut hml = Array1::<f64>::zeros(n);
    let mut hll = Array1::<f64>::zeros(n);
    let mut dhmm = Array1::<f64>::zeros(n);
    let mut dhml = Array1::<f64>::zeros(n);
    let mut dhll = Array1::<f64>::zeros(n);
    for r in rows {
        let i = r.index;
        let w = r.weight;
        obj[i] = weights.objective_psirow[i] * w;
        smu[i] = weights.scoremu[i] * w;
        sls[i] = weights.score_ls[i] * w;
        dsmu[i] = weights.dscoremu[i] * w;
        dsls[i] = weights.dscore_ls[i] * w;
        hmm[i] = weights.hmumu[i] * w;
        hml[i] = weights.hmu_ls[i] * w;
        hll[i] = weights.h_ls_ls[i] * w;
        dhmm[i] = weights.dhmumu[i] * w;
        dhml[i] = weights.dhmu_ls[i] * w;
        dhll[i] = weights.dh_ls_ls[i] * w;
    }
    weights.objective_psirow = obj;
    weights.scoremu = smu;
    weights.score_ls = sls;
    weights.dscoremu = dsmu;
    weights.dscore_ls = dsls;
    weights.hmumu = hmm;
    weights.hmu_ls = hml;
    weights.h_ls_ls = hll;
    weights.dhmumu = dhmm;
    weights.dhmu_ls = dhml;
    weights.dh_ls_ls = dhll;
}


/// HT mask for `GaussianJointPsiSecondWeights`. Same semantics as
/// `apply_ht_mask_first`: each per-row contribution is scaled by 1/π_i and
/// non-sampled rows are zeroed. Consumed row-linearly by
/// `gaussian_joint_psisecondhessian_fromweights` and the `score_psi_psi`
/// `fast_atv(_, d2score_*)` reductions.
fn apply_ht_mask_second(
    weights: &mut GaussianJointPsiSecondWeights,
    rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
) {
    let n = weights.objective_psi_psirow.len();
    let mut obj = Array1::<f64>::zeros(n);
    let mut d2smu = Array1::<f64>::zeros(n);
    let mut d2sls = Array1::<f64>::zeros(n);
    let mut d2hmm = Array1::<f64>::zeros(n);
    let mut d2hml = Array1::<f64>::zeros(n);
    let mut d2hll = Array1::<f64>::zeros(n);
    for r in rows {
        let i = r.index;
        let w = r.weight;
        obj[i] = weights.objective_psi_psirow[i] * w;
        d2smu[i] = weights.d2scoremu[i] * w;
        d2sls[i] = weights.d2score_ls[i] * w;
        d2hmm[i] = weights.d2hmumu[i] * w;
        d2hml[i] = weights.d2hmu_ls[i] * w;
        d2hll[i] = weights.d2h_ls_ls[i] * w;
    }
    weights.objective_psi_psirow = obj;
    weights.d2scoremu = d2smu;
    weights.d2score_ls = d2sls;
    weights.d2hmumu = d2hmm;
    weights.d2hmu_ls = d2hml;
    weights.d2h_ls_ls = d2hll;
}


/// HT mask for `GaussianJointPsiMixedDriftWeights`. Same semantics as the
/// other `apply_ht_mask_*` helpers; consumed row-linearly by
/// `gaussian_joint_psi_mixedhessian_drift_fromweights`.
fn apply_ht_mask_mixed(
    weights: &mut GaussianJointPsiMixedDriftWeights,
    rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
) {
    let n = weights.dhmumu_u.len();
    let mut dhmm_u = Array1::<f64>::zeros(n);
    let mut dhml_u = Array1::<f64>::zeros(n);
    let mut dhll_u = Array1::<f64>::zeros(n);
    let mut d2hmm = Array1::<f64>::zeros(n);
    let mut d2hml = Array1::<f64>::zeros(n);
    let mut d2hll = Array1::<f64>::zeros(n);
    for r in rows {
        let i = r.index;
        let w = r.weight;
        dhmm_u[i] = weights.dhmumu_u[i] * w;
        dhml_u[i] = weights.dhmu_ls_u[i] * w;
        dhll_u[i] = weights.dh_ls_ls_u[i] * w;
        d2hmm[i] = weights.d2hmumu[i] * w;
        d2hml[i] = weights.d2hmu_ls[i] * w;
        d2hll[i] = weights.d2h_ls_ls[i] * w;
    }
    weights.dhmumu_u = dhmm_u;
    weights.dhmu_ls_u = dhml_u;
    weights.dh_ls_ls_u = dhll_u;
    weights.d2hmumu = d2hmm;
    weights.d2hmu_ls = d2hml;
    weights.d2h_ls_ls = d2hll;
}


fn gaussian_jointrow_scalars(
    y: &Array1<f64>,
    etamu: &Array1<f64>,
    eta_ls: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<GaussianJointRowScalars, String> {
    let nobs = y.len();
    if etamu.len() != nobs || eta_ls.len() != nobs || weights.len() != nobs {
        return Err(GamlssError::DimensionMismatch {
            reason: "Gaussian joint row scalar input size mismatch".to_string(),
        }
        .into());
    }
    let mut obs_weight = Array1::<f64>::uninit(nobs);
    let mut w = Array1::<f64>::uninit(nobs);
    let mut m = Array1::<f64>::uninit(nobs);
    let mut n = Array1::<f64>::uninit(nobs);
    let mut kappa = Array1::<f64>::uninit(nobs);
    let mut kappa_prime = Array1::<f64>::uninit(nobs);
    let mut kappa_dprime = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let jet = crate::families::sigma_link::logb_sigma_jet1_scalar(eta_ls[i]);
        let s = jet.sigma;
        // κ = exp(η)/(b + exp(η)). Use the direct exp(η)/σ form
        // when finite — it preserves the precision of exp(η) at very negative
        // η (where 1 − b/σ catastrophically cancels because b/σ → 1). The
        // η → +∞ branch returns 1 cleanly without hitting ∞/∞ NaN.
        let ki = logb_dlog_sigma_deta(s, jet.d1);
        let kp = ki * (1.0 - ki);
        let kdp = kp * (1.0 - 2.0 * ki);
        let wi = weights[i] / (s * s);
        let ri = y[i] - etamu[i];
        obs_weight[i].write(weights[i]);
        w[i].write(wi);
        m[i].write(ri * wi);
        n[i].write(ri * ri * wi);
        kappa[i].write(ki);
        kappa_prime[i].write(kp);
        kappa_dprime[i].write(kdp);
    }
    // SAFETY: every `MaybeUninit` slot in each of these arrays was written
    // exactly once in the `for i in 0..nobs` loop above; no slot is read,
    // moved, or dropped before this point.
    let (obs_weight, w, m, n, kappa, kappa_prime, kappa_dprime) = unsafe {
        (
            obs_weight.assume_init(),
            w.assume_init(),
            m.assume_init(),
            n.assume_init(),
            kappa.assume_init(),
            kappa_prime.assume_init(),
            kappa_dprime.assume_init(),
        )
    };
    Ok(GaussianJointRowScalars {
        obs_weight,
        w,
        m,
        n,
        kappa,
        kappa_prime,
        kappa_dprime,
    })
}


fn gaussian_joint_first_directionalweights(
    scalars: &GaussianJointRowScalars,
    dotmu: &Array1<f64>,
    dot_eta: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let nobs = scalars.w.len();
    let mut w_u = Array1::<f64>::uninit(nobs);
    let mut c_u = Array1::<f64>::uninit(nobs);
    let mut d_u = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let mi = scalars.m[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let ai = scalars.obs_weight[i];
        let dm = dotmu[i];
        let de = dot_eta[i];
        // κ-scaled log-sigma direction.
        let sde = ki * de;
        w_u[i].write(-2.0 * wi * sde);
        // + 2·κ'·m·de: dκ/dη chain-rule from σ = b + e^η.
        c_u[i].write(ki * (-2.0 * wi * dm - 4.0 * mi * sde) + 2.0 * mi * kpi * de);
        // Directional derivative of Fisher E[H_{ls,ls}]=2κ²a: 4κκ'a·de (#566).
        d_u[i].write(4.0 * ki * kpi * ai * de);
    }
    // SAFETY: every slot of `w_u`, `c_u`, `d_u` was written exactly once
    // inside the loop above (one `.write(...)` per index per array).
    let (w_u, c_u, d_u) = unsafe { (w_u.assume_init(), c_u.assume_init(), d_u.assume_init()) };
    (w_u, c_u, d_u)
}


fn gaussian_jointsecond_directionalweights(
    scalars: &GaussianJointRowScalars,
    dotmu_u: &Array1<f64>,
    dot_eta_u: &Array1<f64>,
    dotmuv: &Array1<f64>,
    dot_etav: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let nobs = scalars.w.len();
    let mut w_uv = Array1::<f64>::uninit(nobs);
    let mut c_uv = Array1::<f64>::uninit(nobs);
    let mut d_uv = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let mi = scalars.m[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        let ai = scalars.obs_weight[i];
        let dmu = dotmu_u[i];
        let dmv = dotmuv[i];
        let deu = dot_eta_u[i];
        let dev = dot_etav[i];
        // κ-scaled log-sigma directions.
        let sdeu = ki * deu;
        let sdev = ki * dev;
        let de_sym = dmu * dev + dmv * deu;
        let de_eta = deu * dev;
        // − 2·κ'·w·deu·dev: ∂²w/∂η² = 4wκ² − 2wκ'.
        w_uv[i].write(4.0 * wi * sdeu * sdev - 2.0 * wi * kpi * de_eta);
        // − 2·κ'·w·sym + 2·m·(κ''−6·κ·κ')·deu·dev from d²(2mκ).
        c_uv[i].write(
            ki * (4.0 * wi * (dmu * sdev + dmv * sdeu) + 8.0 * mi * sdeu * sdev)
                - 2.0 * wi * kpi * de_sym
                + 2.0 * mi * (kdpi - 6.0 * ki * kpi) * de_eta,
        );
        // d²/du dv of Fisher E[H_{ls,ls}]=2κ²a: bilinear in fixed directions
        // u,v, no μ dependence ⇒ 4a(κ'²+κκ'')·deu·dev (#566).
        d_uv[i].write(4.0 * ai * (kpi * kpi + ki * kdpi) * de_eta);
    }
    // SAFETY: every slot of `w_uv`, `c_uv`, `d_uv` was written exactly once
    // inside the loop above.
    let (w_uv, c_uv, d_uv) =
        unsafe { (w_uv.assume_init(), c_uv.assume_init(), d_uv.assume_init()) };
    (w_uv, c_uv, d_uv)
}


fn gaussian_joint_psi_firstweights(
    scalars: &GaussianJointRowScalars,
    mu_a: &Array1<f64>,
    eta_a: &Array1<f64>,
) -> GaussianJointPsiFirstWeights {
    let nobs = scalars.w.len();
    let mut objective_psirow = Array1::<f64>::uninit(nobs);
    let mut scoremu = Array1::<f64>::uninit(nobs);
    let mut score_ls = Array1::<f64>::uninit(nobs);
    let mut dscoremu = Array1::<f64>::uninit(nobs);
    let mut dscore_ls = Array1::<f64>::uninit(nobs);
    let mut hmumu = Array1::<f64>::uninit(nobs);
    let mut hmu_ls = Array1::<f64>::uninit(nobs);
    let mut h_ls_ls = Array1::<f64>::uninit(nobs);
    let mut dhmumu = Array1::<f64>::uninit(nobs);
    let mut dhmu_ls = Array1::<f64>::uninit(nobs);
    let mut dh_ls_ls = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let mi = scalars.m[i];
        let ni = scalars.n[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let ai = scalars.obs_weight[i];
        let ma = mu_a[i];
        let ea = eta_a[i];
        // κ-scaled log-sigma direction.
        let sea = ki * ea;
        let smu = -mi;
        let sls = ki * (ai - ni);
        let wi = scalars.w[i];
        scoremu[i].write(smu);
        score_ls[i].write(sls);
        dscoremu[i].write(wi * ma + 2.0 * mi * sea);
        // + κ'·(a−n)·η̇ chain-rule term (∂[κ(a−n)]/∂η = κ'(a−n) + 2κ²n).
        dscore_ls[i].write(ki * (2.0 * mi * ma + 2.0 * ni * sea) + kpi * (ai - ni) * ea);
        hmumu[i].write(wi);
        // Cross block: Fisher expectation E[H_{μ,ls}] = 2κ·E[m] = 0 (μ ⊥ σ;
        // see exact_newton_joint_hessian_from_designs / #684). The observed
        // 2mκ is mean-zero noise that would inject spurious μ↔σ coupling into
        // the REML determinant via the Schur complement and over-smooth log σ.
        hmu_ls[i].write(0.0);
        // Fisher/expected (log σ, log σ) information: E[H_{ls,ls}] = 2κ²a.
        // The observed curvature 2κ²n + κ'(a−n) collapses where the fitted
        // residual is small (n→0), under-counting the scale block's EDF and
        // letting REML over-smooth the scale predictor toward a flat constant
        // (#566). Using E[n]=a (true model) gives the residual-free expected
        // information 2κ²a, exactly as gamlss/mgcv gaulss Fisher-score the
        // scale channel and as the diagonal PIRLS kernel already does
        // (gaussian_diagonal_row_kernel: 2·obs_weight·κ²). The score
        // (score_ls/dscore_ls/d2score_ls) stays the exact observed gradient so
        // the joint Newton still converges to the true MLE stationary point;
        // only the (ls,ls) curvature feeding the REML determinant/EDF is the
        // expectation.
        h_ls_ls[i].write(2.0 * ki * ki * ai);
        dhmumu[i].write(-2.0 * wi * sea);
        // Cross block is Fisher 0 (μ ⊥ σ; #684), so its directional derivative
        // is identically 0.
        dhmu_ls[i].write(0.0);
        // Directional derivative of E[H_{ls,ls}]=2κ²a along (μ̇,η̇): no μ
        // dependence; ∂(2κ²a)/∂η = 4κκ'a, so dh_ls_ls = 4κκ'a·η̇.
        dh_ls_ls[i].write(4.0 * ki * kpi * ai * ea);
        objective_psirow[i].write(smu * ma + sls * ea);
    }
    // SAFETY: every `MaybeUninit` slot in each field array was written
    // exactly once inside the `for i in 0..nobs` loop above.
    unsafe {
        GaussianJointPsiFirstWeights {
            objective_psirow: objective_psirow.assume_init(),
            scoremu: scoremu.assume_init(),
            score_ls: score_ls.assume_init(),
            dscoremu: dscoremu.assume_init(),
            dscore_ls: dscore_ls.assume_init(),
            hmumu: hmumu.assume_init(),
            hmu_ls: hmu_ls.assume_init(),
            h_ls_ls: h_ls_ls.assume_init(),
            dhmumu: dhmumu.assume_init(),
            dhmu_ls: dhmu_ls.assume_init(),
            dh_ls_ls: dh_ls_ls.assume_init(),
        }
    }
}


fn gaussian_joint_psisecondweights(
    scalars: &GaussianJointRowScalars,
    mu_a: &Array1<f64>,
    eta_a: &Array1<f64>,
    mu_b: &Array1<f64>,
    eta_b: &Array1<f64>,
    mu_ab: &Array1<f64>,
    eta_ab: &Array1<f64>,
) -> GaussianJointPsiSecondWeights {
    let nobs = scalars.w.len();
    let mut objective_psi_psirow = Array1::<f64>::uninit(nobs);
    let mut d2scoremu = Array1::<f64>::uninit(nobs);
    let mut d2score_ls = Array1::<f64>::uninit(nobs);
    let mut d2hmumu = Array1::<f64>::uninit(nobs);
    let mut d2hmu_ls = Array1::<f64>::uninit(nobs);
    let mut d2h_ls_ls = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let mi = scalars.m[i];
        let ni = scalars.n[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        let ai = scalars.obs_weight[i];
        let amn = ai - ni;
        let ma = mu_a[i];
        let mb = mu_b[i];
        let mab = mu_ab[i];
        let ea = eta_a[i];
        let eb = eta_b[i];
        let eab = eta_ab[i];
        // κ-scaled log-sigma directions.
        let sea = ki * ea;
        let seb = ki * eb;
        let seab = ki * eab;
        let cross = ma * seb + mb * sea;
        // Bare-η symmetric form (no κ): needed for κ' chain-rule terms.
        let cross_eta = ma * eb + mb * ea;
        let sea_seb = sea * seb;
        let ea_eb = ea * eb;
        let ma_mb = ma * mb;
        // + κ'·(a−n)·ea·eb: dκ/dη chain-rule contribution from σ = b + e^η.
        objective_psi_psirow[i].write(
            wi * ma_mb + 2.0 * mi * cross + 2.0 * ni * sea_seb - mi * mab
                + ki * amn * eab
                + kpi * amn * ea_eb,
        );
        // + 2·m·κ'·ea·eb: ∂²(−m)/∂η² = −4mκ² + 2mκ'.
        d2scoremu[i].write(
            wi * mab - 2.0 * wi * cross - 4.0 * mi * sea_seb
                + 2.0 * mi * seab
                + 2.0 * mi * kpi * ea_eb,
        );
        // + 2·κ'·m·sym(μ_a η_b) + (κ''(a−n)+6κκ'n)·ea·eb + κ'(a−n)·eab.
        d2score_ls[i].write(
            ki * (-2.0 * wi * ma_mb - 4.0 * mi * cross - 4.0 * ni * sea_seb
                + 2.0 * mi * mab
                + 2.0 * ni * seab)
                + 2.0 * mi * kpi * cross_eta
                + (kdpi * amn + 6.0 * ki * kpi * ni) * ea_eb
                + kpi * amn * eab,
        );
        // − 2·κ'·w·ea·eb: ∂²w/∂η² = 4wκ² − 2wκ'.
        d2hmumu[i].write(4.0 * wi * sea_seb - 2.0 * wi * seab - 2.0 * wi * kpi * ea_eb);
        // Cross block is Fisher 0 (μ ⊥ σ; #684), so its second directional
        // derivative is identically 0.
        d2hmu_ls[i].write(0.0);
        // d²/dψ_a dψ_b of the Fisher (ls,ls) information E[H_{ls,ls}]=2κ²a (#566).
        // No μ dependence; ∂(2κ²a)/∂η=4κκ'a and ∂(4κκ'a)/∂η=4a(κ'²+κκ'')a, so
        // the second directional derivative is 4a(κ'²+κκ'')·ea·eb + 4aκκ'·eab.
        d2h_ls_ls[i].write(4.0 * ai * (kpi * kpi + ki * kdpi) * ea_eb + 4.0 * ai * ki * kpi * eab);
    }
    // SAFETY: every `MaybeUninit` slot in each field array was written
    // exactly once inside the `for i in 0..nobs` loop above.
    unsafe {
        GaussianJointPsiSecondWeights {
            objective_psi_psirow: objective_psi_psirow.assume_init(),
            d2scoremu: d2scoremu.assume_init(),
            d2score_ls: d2score_ls.assume_init(),
            d2hmumu: d2hmumu.assume_init(),
            d2hmu_ls: d2hmu_ls.assume_init(),
            d2h_ls_ls: d2h_ls_ls.assume_init(),
        }
    }
}


fn gaussian_joint_psi_mixed_driftweights(
    scalars: &GaussianJointRowScalars,
    // Only the log-σ–channel directions enter the surviving (μ,μ) and (ls,ls)
    // Fisher blocks; the μ-channel drift directions fed the observed cross
    // block, which is now Fisher 0 (μ ⊥ σ; #684) and no longer assembled.
    dot_eta: &Array1<f64>,
    eta_a: &Array1<f64>,
    dot_eta_a: &Array1<f64>,
) -> GaussianJointPsiMixedDriftWeights {
    let nobs = scalars.w.len();
    let mut dhmumu_u = Array1::<f64>::uninit(nobs);
    let mut dhmu_ls_u = Array1::<f64>::uninit(nobs);
    let mut dh_ls_ls_u = Array1::<f64>::uninit(nobs);
    let mut d2hmumu = Array1::<f64>::uninit(nobs);
    let mut d2hmu_ls = Array1::<f64>::uninit(nobs);
    let mut d2h_ls_ls = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        let ai = scalars.obs_weight[i];
        let de = dot_eta[i];
        let ea = eta_a[i];
        let dea = dot_eta_a[i];
        // κ-scaled log-sigma directions.
        let sde = ki * de;
        let sea = ki * ea;
        let sdea = ki * dea;
        let de_ea = de * ea;
        // First directional derivative of Hessian blocks (== Helper A).
        dhmumu_u[i].write(-2.0 * wi * sde);
        // Cross block is Fisher 0 (μ ⊥ σ; #684); its first directional and
        // second mixed directional derivatives are identically 0. The
        // observed-cross drift inputs (m, dotmu, μ_a, dotmu_a) are therefore
        // not read here.
        dhmu_ls_u[i].write(0.0);
        // Directional derivative of Fisher E[H_{ls,ls}]=2κ²a along (dm,de):
        // no μ dependence, ∂(2κ²a)/∂η=4κκ'a ⇒ 4κκ'a·de (#566).
        dh_ls_ls_u[i].write(4.0 * ki * kpi * ai * de);
        // − 2·κ'·w·de·ea: ∂²w/∂η² = 4wκ² − 2wκ'.
        d2hmumu[i].write(4.0 * wi * sde * sea - 2.0 * wi * sdea - 2.0 * wi * kpi * de_ea);
        d2hmu_ls[i].write(0.0);
        // d²/(drift × ψ) of Fisher E[H_{ls,ls}]=2κ²a: 4a(κ'²+κκ'')·de·ea +
        // 4aκκ'·dea (drift direction de, ψ direction ea, mixed dea) (#566).
        d2h_ls_ls[i].write(4.0 * ai * (kpi * kpi + ki * kdpi) * de_ea + 4.0 * ai * ki * kpi * dea);
    }
    // SAFETY: every `MaybeUninit` slot in each field array was written
    // exactly once inside the `for i in 0..nobs` loop above.
    unsafe {
        GaussianJointPsiMixedDriftWeights {
            dhmumu_u: dhmumu_u.assume_init(),
            dhmu_ls_u: dhmu_ls_u.assume_init(),
            dh_ls_ls_u: dh_ls_ls_u.assume_init(),
            d2hmumu: d2hmumu.assume_init(),
            d2hmu_ls: d2hmu_ls.assume_init(),
            d2h_ls_ls: d2h_ls_ls.assume_init(),
        }
    }
}


/// Canonical Gaussian location-scale Fisher (expected) joint-Hessian row
/// coefficients `(mm, ml, ll)` — the SINGLE source of truth for this curvature,
/// shared by every representation that assembles the value Hessian (the dense
/// `exact_newton_joint_hessian_from_designs` and the matrix-free
/// `GaussianLocationScaleHessianWorkspace`). The (μ, log σ) information is
/// block-diagonal because location and scale are information-orthogonal:
///   `ml = E[H_{μ,ls}] = 2κ·E[m] = 2κ·E[r]·w/σ² = 0`  (E[r]=0 at any β; #684),
/// and the (log σ, log σ) block is the residual-free Fisher form
///   `ll = E[H_{ls,ls}] = 2κ²a`  (a = obs_weight; #566).
/// Routing both paths through this one constructor makes the cross-block drift
/// that caused #684 — one representation using the observed `2κm`, another the
/// Fisher 0 — structurally impossible: they cannot disagree because they read
/// the same coefficients. The observed SCORE still drives the Newton step
/// (Fisher scoring → exact joint MLE); only the curvature feeding the REML
/// determinant / Newton metric is the orthogonal expectation.
fn gaussian_locscale_fisher_joint_row_coeffs(
    rows: &GaussianJointRowScalars,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mm = rows.w.clone();
    let ml = Array1::<f64>::zeros(rows.kappa.len());
    let ll = 2.0 * &rows.kappa * &rows.kappa * &rows.obs_weight;
    (mm, ml, ll)
}


fn gaussian_joint_hessian_from_designs(
    xmu: &DenseOrOperator<'_>,
    x_ls: &DenseOrOperator<'_>,
    hmumu_coeff: &Array1<f64>,
    hmu_ls_coeff: &Array1<f64>,
    h_ls_ls_coeff: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    if xmu.nrows() != hmumu_coeff.len()
        || xmu.nrows() != hmu_ls_coeff.len()
        || xmu.nrows() != h_ls_ls_coeff.len()
        || x_ls.nrows() != xmu.nrows()
    {
        return Err(GamlssError::DimensionMismatch { reason: format!(
            "gaussian_joint_hessian_from_designs dimension mismatch: xmu {}x{}, x_ls {}x{}, coeffs {}/{}/{}",
            xmu.nrows(),
            xmu.ncols(),
            x_ls.nrows(),
            x_ls.ncols(),
            hmumu_coeff.len(),
            hmu_ls_coeff.len(),
            h_ls_ls_coeff.len()
        ) }.into());
    }

    let n = xmu.nrows();
    let pmu = xmu.ncols();
    let p_ls = x_ls.ncols();
    let total = pmu + p_ls;
    let mut out = Array2::<f64>::zeros((total, total));
    for rows in exact_design_row_chunks(n, pmu.max(p_ls)) {
        let xmu_chunk = xmu.row_chunk(rows.clone())?;
        let xls_chunk = x_ls.row_chunk(rows.clone())?;
        let hmumu = hmumu_coeff.slice(s![rows.clone()]);
        let hmu_ls = hmu_ls_coeff.slice(s![rows.clone()]);
        let h_ls_ls = h_ls_ls_coeff.slice(s![rows.clone()]);
        let chunk_hessian =
            fast_joint_hessian_2x2(&xmu_chunk, &xls_chunk, &hmumu, &hmu_ls, &h_ls_ls);
        out += &chunk_hessian;
    }
    Ok(out)
}


fn gaussian_joint_psihessian_fromweights(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
    xmu_psi: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_psi: CustomFamilyPsiLinearMapRef<'_>,
    weights: &GaussianJointPsiFirstWeights,
) -> Result<Array2<f64>, String> {
    // For the symmetric blocks (hmumu, h_ls_ls), the pair
    //   X_psi^T D X  and  X^T D X_psi
    // are transposes of each other, so compute one and add its transpose.
    let a_mu = weighted_crossprod_psi_maps(
        xmu_psi,
        weights.hmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let hmumu = &a_mu + &a_mu.t() + &xt_diag_x_dense(xmu, &weights.dhmumu)?;
    let hmu_ls = weighted_crossprod_psi_maps(
        xmu_psi,
        weights.hmu_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )? + &weighted_crossprod_psi_maps(
        CustomFamilyPsiLinearMapRef::Dense(xmu),
        weights.hmu_ls.view(),
        x_ls_psi,
    )? + &xt_diag_y_dense(xmu, &weights.dhmu_ls, x_ls)?;
    let a_ls = weighted_crossprod_psi_maps(
        x_ls_psi,
        weights.h_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let h_ls_ls = &a_ls + &a_ls.t() + &xt_diag_x_dense(x_ls, &weights.dh_ls_ls)?;
    Ok(gaussian_pack_joint_symmetrichessian(
        &hmumu, &hmu_ls, &h_ls_ls,
    ))
}


fn build_two_block_custom_family_joint_psi_operator_from_actions(
    left_action: Option<CustomFamilyPsiDesignAction>,
    right_action: Option<CustomFamilyPsiDesignAction>,
    left_range: std::ops::Range<usize>,
    right_range: std::ops::Range<usize>,
    left_design: &Array2<f64>,
    right_design: &Array2<f64>,
    left_weights: &Array1<f64>,
    cross_weights: &Array1<f64>,
    right_weights: &Array1<f64>,
    left_drift_weights: &Array1<f64>,
    cross_drift_weights: &Array1<f64>,
    right_drift_weights: &Array1<f64>,
) -> Result<Option<std::sync::Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
{
    if left_action.is_none() && right_action.is_none() {
        return Ok(None);
    }

    let total = left_design.ncols() + right_design.ncols();
    let channels = vec![
        CustomFamilyJointDesignChannel::new(left_range, shared_dense_arc(left_design), left_action),
        CustomFamilyJointDesignChannel::new(
            right_range,
            shared_dense_arc(right_design),
            right_action,
        ),
    ];
    let pair_contributions = vec![
        CustomFamilyJointDesignPairContribution::new(
            0,
            0,
            left_weights.clone(),
            left_drift_weights.clone(),
        ),
        CustomFamilyJointDesignPairContribution::new(
            0,
            1,
            cross_weights.clone(),
            cross_drift_weights.clone(),
        ),
        CustomFamilyJointDesignPairContribution::new(
            1,
            0,
            cross_weights.clone(),
            cross_drift_weights.clone(),
        ),
        CustomFamilyJointDesignPairContribution::new(
            1,
            1,
            right_weights.clone(),
            right_drift_weights.clone(),
        ),
    ];

    Ok(Some(std::sync::Arc::new(
        CustomFamilyJointPsiOperator::new(total, channels, pair_contributions),
    )))
}


fn gaussian_joint_psisecondhessian_fromweights(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
    xmu_i: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_i: CustomFamilyPsiLinearMapRef<'_>,
    xmu_j: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_j: CustomFamilyPsiLinearMapRef<'_>,
    xmu_ab: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_ab: CustomFamilyPsiLinearMapRef<'_>,
    weights_i: &GaussianJointPsiFirstWeights,
    weights_j: &GaussianJointPsiFirstWeights,
    secondweights: &GaussianJointPsiSecondWeights,
) -> Result<Array2<f64>, String> {
    // Exploit transpose symmetry: X_a^T D X_b and X_b^T D X_a are transposes.
    // For each such pair in the symmetric blocks (hmumu, h_ls_ls), compute one
    // and add its transpose, halving the number of O(np²) products.
    let a_ab_mu = weighted_crossprod_psi_maps(
        xmu_ab,
        weights_i.hmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let a_ij_mu = weighted_crossprod_psi_maps(xmu_i, weights_i.hmumu.view(), xmu_j)?;
    let a_iwj_mu = weighted_crossprod_psi_maps(
        xmu_i,
        weights_j.dhmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let a_jwi_mu = weighted_crossprod_psi_maps(
        xmu_j,
        weights_i.dhmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let hmumu = &a_ab_mu
        + &a_ab_mu.t()
        + &a_ij_mu
        + a_ij_mu.t()
        + &a_iwj_mu
        + a_iwj_mu.t()
        + &a_jwi_mu
        + a_jwi_mu.t()
        + &xt_diag_x_dense(xmu, &secondweights.d2hmumu)?;
    let hmu_ls = weighted_crossprod_psi_maps(
        xmu_ab,
        weights_i.hmu_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )? + &weighted_crossprod_psi_maps(xmu_i, weights_i.hmu_ls.view(), x_ls_j)?
        + &weighted_crossprod_psi_maps(xmu_j, weights_i.hmu_ls.view(), x_ls_i)?
        + &weighted_crossprod_psi_maps(
            xmu_i,
            weights_j.dhmu_ls.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?
        + &weighted_crossprod_psi_maps(
            xmu_j,
            weights_i.dhmu_ls.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?
        + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            weights_i.dhmu_ls.view(),
            x_ls_j,
        )?
        + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            weights_j.dhmu_ls.view(),
            x_ls_i,
        )?
        + &xt_diag_y_dense(xmu, &secondweights.d2hmu_ls, x_ls)?
        + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            weights_i.hmu_ls.view(),
            x_ls_ab,
        )?;
    let a_ab_ls = weighted_crossprod_psi_maps(
        x_ls_ab,
        weights_i.h_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let a_ij_ls = weighted_crossprod_psi_maps(x_ls_i, weights_i.h_ls_ls.view(), x_ls_j)?;
    let a_iwj_ls = weighted_crossprod_psi_maps(
        x_ls_i,
        weights_j.dh_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let a_jwi_ls = weighted_crossprod_psi_maps(
        x_ls_j,
        weights_i.dh_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let h_ls_ls = &a_ab_ls
        + &a_ab_ls.t()
        + &a_ij_ls
        + a_ij_ls.t()
        + &a_iwj_ls
        + a_iwj_ls.t()
        + &a_jwi_ls
        + a_jwi_ls.t()
        + &xt_diag_x_dense(x_ls, &secondweights.d2h_ls_ls)?;
    Ok(gaussian_pack_joint_symmetrichessian(
        &hmumu, &hmu_ls, &h_ls_ls,
    ))
}


fn gaussian_joint_psi_mixedhessian_drift_fromweights(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
    xmu_psi: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_psi: CustomFamilyPsiLinearMapRef<'_>,
    mixedweights: &GaussianJointPsiMixedDriftWeights,
) -> Result<Array2<f64>, String> {
    let a_mu = weighted_crossprod_psi_maps(
        xmu_psi,
        mixedweights.dhmumu_u.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let hmumu = &a_mu + &a_mu.t() + &xt_diag_x_dense(xmu, &mixedweights.d2hmumu)?;
    let hmu_ls = weighted_crossprod_psi_maps(
        xmu_psi,
        mixedweights.dhmu_ls_u.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )? + &weighted_crossprod_psi_maps(
        CustomFamilyPsiLinearMapRef::Dense(xmu),
        mixedweights.dhmu_ls_u.view(),
        x_ls_psi,
    )? + &xt_diag_y_dense(xmu, &mixedweights.d2hmu_ls, x_ls)?;
    let a_ls = weighted_crossprod_psi_maps(
        x_ls_psi,
        mixedweights.dh_ls_ls_u.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let h_ls_ls = &a_ls + &a_ls.t() + &xt_diag_x_dense(x_ls, &mixedweights.d2h_ls_ls)?;
    Ok(gaussian_pack_joint_symmetrichessian(
        &hmumu, &hmu_ls, &h_ls_ls,
    ))
}


#[inline]
fn exp_sigma_derivs_up_to_fourth_array(
    eta: ArrayView1<'_, f64>,
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = eta.len();
    let tuples: Vec<(f64, f64, f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| exp_sigma_derivs_up_to_fourth_scalar(eta[i]))
        .collect();
    let mut sigma = Array1::<f64>::zeros(n);
    let mut d1 = Array1::<f64>::zeros(n);
    let mut d2 = Array1::<f64>::zeros(n);
    let mut d3 = Array1::<f64>::zeros(n);
    let mut d4 = Array1::<f64>::zeros(n);
    for (i, (s_i, d1_i, d2_i, d3_i, d4_i)) in tuples.into_iter().enumerate() {
        sigma[i] = s_i;
        d1[i] = d1_i;
        d2[i] = d2_i;
        d3[i] = d3_i;
        d4[i] = d4_i;
    }
    (sigma, d1, d2, d3, d4)
}


impl GaussianLocationScaleFamily {
    pub const BLOCK_MU: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    fn get_or_compute_row_scalars(
        &self,
        etamu: &Array1<f64>,
        eta_ls: &Array1<f64>,
    ) -> Result<Arc<GaussianJointRowScalars>, String> {
        Ok(Arc::new(gaussian_jointrow_scalars(
            &self.y,
            etamu,
            eta_ls,
            &self.weights,
        )?))
    }

    pub fn parameternames() -> &'static [&'static str] {
        &["mu", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gaussian_location_scale",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    fn exact_joint_supported(&self) -> bool {
        self.mu_design.is_some() && self.log_sigma_design.is_some()
    }

    fn exact_block_designs(&self) -> Result<(DenseOrOperator<'_>, DenseOrOperator<'_>), String> {
        let mu_design = self.mu_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleFamily exact path is missing mu design".to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleFamily exact path is missing log-sigma design".to_string()
        })?;
        let planned = dense_blocks_planned_budget(&[mu_design, log_sigma_design]);
        let xmu = dense_block_or_operator(
            mu_design,
            mu_design.nrows(),
            mu_design.ncols(),
            planned[0],
            &self.policy,
        );
        let x_ls = dense_block_or_operator(
            log_sigma_design,
            log_sigma_design.nrows(),
            log_sigma_design.ncols(),
            planned[1],
            &self.policy,
        );
        Ok((xmu, x_ls))
    }

    fn exact_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(DenseOrOperator<'a>, DenseOrOperator<'a>), String> {
        if specs.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily spec-aware exact path expects 2 specs, got {}",
                    specs.len()
                ),
            }
            .into());
        }
        let mu_design = &specs[Self::BLOCK_MU].design;
        let log_sigma_design = &specs[Self::BLOCK_LOG_SIGMA].design;
        let planned = dense_blocks_planned_budget(&[mu_design, log_sigma_design]);
        let xmu = dense_block_or_operator(
            mu_design,
            mu_design.nrows(),
            mu_design.ncols(),
            planned[0],
            &self.policy,
        );
        let x_ls = dense_block_or_operator(
            log_sigma_design,
            log_sigma_design.nrows(),
            log_sigma_design.ncols(),
            planned[1],
            &self.policy,
        );
        Ok((xmu, x_ls))
    }

    fn exact_joint_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(DenseOrOperator<'a>, DenseOrOperator<'a>)>, String> {
        if self.exact_joint_supported() {
            return self.exact_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.exact_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        let xmu = match xmu {
            DenseOrOperator::Borrowed(dense) => Cow::Borrowed(dense),
            DenseOrOperator::Owned(dense) => Cow::Owned(dense),
            DenseOrOperator::Operator(_) => {
                return Err(
                    "GaussianLocationScaleFamily exact psi path requires chunked operator support for oversized designs"
                        .to_string(),
                );
            }
        };
        let x_ls = match x_ls {
            DenseOrOperator::Borrowed(dense) => Cow::Borrowed(dense),
            DenseOrOperator::Owned(dense) => Cow::Owned(dense),
            DenseOrOperator::Operator(_) => {
                return Err(
                    "GaussianLocationScaleFamily exact psi path requires chunked operator support for oversized designs"
                        .to_string(),
                );
            }
        };
        Ok(Some((xmu, x_ls)))
    }

    fn exact_newton_joint_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_from_designs(block_states, &xmu, &x_ls)
    }

    fn exact_newton_joint_hessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            &xmu,
            &x_ls,
        )
    }

    fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &xmu,
            &x_ls,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &xmu,
            &x_ls,
        )
    }

    fn exact_newton_joint_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        // Block-diagonal Gaussian Fisher curvature (μ ⊥ σ ⇒ cross = 0, #684;
        // (ls,ls) = 2κ²a, #566), built from the shared single-source-of-truth
        // constructor so this dense path and the matrix-free workspace can never
        // disagree on the cross block. See `gaussian_locscale_fisher_joint_row_coeffs`.
        let (mm, cross, scale) = gaussian_locscale_fisher_joint_row_coeffs(&rows);
        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &mm, &cross, &scale,
        )?))
    }

    fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let total = pmu + p_ls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily joint d_beta length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let ximu = xmu.dot(d_beta_flat.slice(s![0..pmu]));
        let xi_ls = x_ls.dot(d_beta_flat.slice(s![pmu..pmu + p_ls]));
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let directional = gaussian_joint_first_directionalweights(&rows, &ximu, &xi_ls);
        let dhmumu = directional.0;
        let dh_ls_ls = directional.2;
        // Fisher cross block E[H_{μ,ls}] ≡ 0 (μ ⊥ σ; see
        // exact_newton_joint_hessian_from_designs / #684), so its directional
        // derivative is identically 0 — keep the Hessian's curvature object the
        // block-diagonal Gaussian Fisher information at every order. The
        // observed-cross directional weight (`directional.1`) is therefore not
        // assembled.
        let dhmu_ls = Array1::<f64>::zeros(dhmumu.len());

        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &dhmumu, &dhmu_ls, &dh_ls_ls,
        )?))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let total = pmu + p_ls;
        if d_beta_u_flat.len() != total || d_betav_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint second directional derivative length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_betav_flat.len(),
                total
            ) }.into());
        }
        let ximu_u = xmu.dot(d_beta_u_flat.slice(s![0..pmu]));
        let xi_ls_u = x_ls.dot(d_beta_u_flat.slice(s![pmu..pmu + p_ls]));
        let ximuv = xmu.dot(d_betav_flat.slice(s![0..pmu]));
        let xi_lsv = x_ls.dot(d_betav_flat.slice(s![pmu..pmu + p_ls]));
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let second =
            gaussian_jointsecond_directionalweights(&rows, &ximu_u, &xi_ls_u, &ximuv, &xi_lsv);
        let d2hmumu = second.0;
        let d2h_ls_ls = second.2;
        // Fisher cross block E[H_{μ,ls}] ≡ 0 (μ ⊥ σ; #684), so its second
        // directional derivative is identically 0; `second.1` (observed) is not
        // assembled, keeping the curvature object block-diagonal Fisher.
        let d2hmu_ls = Array1::<f64>::zeros(d2hmumu.len());

        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &d2hmumu, &d2hmu_ls, &d2h_ls_ls,
        )?))
    }

    fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        let Some(parts) = locscale_joint_psi_direction_parts(
            block_states,
            derivative_blocks,
            psi_index,
            self.y.len(),
            xmu.ncols(),
            x_ls.ncols(),
            Self::BLOCK_MU,
            Self::BLOCK_LOG_SIGMA,
            2,
            "GaussianLocationScaleFamily",
            "mu",
            policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LocationScaleJointPsiDirection {
            block_idx: parts.block_idx,
            local_idx: parts.local_idx,
            z_primary_psi: parts.primary_z,
            z_ls_psi: parts.log_sigma_z,
            x_primary_psi: parts.primary_psi,
            x_ls_psi: parts.log_sigma_psi,
        }))
    }

    fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<LocationScaleJointPsiSecondDrifts, String> {
        locscale_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            LocScalePsiDriftConfig {
                n: self.y.len(),
                p_primary: xmu.ncols(),
                p_log_sigma: x_ls.ncols(),
                primary_block_idx: Self::BLOCK_MU,
                log_sigma_block_idx: Self::BLOCK_LOG_SIGMA,
                family_name: "GaussianLocationScaleFamily",
                primary_label: "mu",
                policy: &self.policy,
            },
        )
    }

    fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        if specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi terms expect 2 specs and 2 derivative blocks, got {} and {}",
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        // Gaussian 2-block location-scale family in the unified flattened
        // coefficient space beta = [betamu; beta_sigma]:
        //
        //   mu_i = z_i^T betamu,
        //   ell_i = x_i^T beta_sigma,
        //   s_i = exp(ell_i),
        //   r_i = y_i - mu_i,
        //   q_i = r_i / s_i,
        //   w_i = s_i^{-2},
        //   alpha_i = r_i s_i^{-2},
        //   b_i = q_i^2.
        //
        // The first fixed-beta psi object returned here is likelihood-only:
        //
        //   D_a         = -alpha^T m_a + (1 - b)^T ell_a
        //   D_{beta a}  = [ -Xmu^T alpha_a - X_{mu,a}^T alpha ;
        //                   -X_sigma^T b_a + X_{sigma,a}^T (1-b) ]
        //   D_{bb a}    = [ Xmu^T W_a Xmu + X_{mu,a}^T W Xmu + Xmu^T W X_{mu,a},
        //                   2( Xmu^T A_a X_sigma + X_{mu,a}^T A X_sigma + Xmu^T A X_{sigma,a} );
        //                   sym,
        //                   2( X_sigma^T B_a X_sigma + X_{sigma,a}^T B X_sigma + X_sigma^T B X_{sigma,a} ) ]
        //
        // with m_a = X_{mu,a} betamu, ell_a = X_{sigma,a} beta_sigma and
        // rowwise scalar drifts
        //
        //   w_a     = -2 w * ell_a
        //   alpha_a = -w * m_a - 2 alpha * ell_a
        //   b_a     = -2 alpha * m_a - 2 b * ell_a.
        //
        // Generic code in custom_family.rs promotes these likelihood-only
        // objects to the full fixed-beta V_a / g_a / H_a by adding S_a.
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let weights_a =
            gaussian_joint_psi_firstweights(&rows, &dir_a.z_primary_psi, &dir_a.z_ls_psi);
        let objective_psi = weights_a.objective_psirow.sum();
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let score_mu =
            xmu_map.transpose_mul(weights_a.scoremu.view()) + fast_atv(xmu, &weights_a.dscoremu);
        let score_ls = x_ls_map.transpose_mul(weights_a.score_ls.view())
            + fast_atv(x_ls, &weights_a.dscore_ls);
        let score_psi = gaussian_pack_joint_score(&score_mu, &score_ls);
        let hessian_psi_operator = build_two_block_custom_family_joint_psi_operator_from_actions(
            dir_a.x_primary_psi.cloned_first_action(),
            dir_a.x_ls_psi.cloned_first_action(),
            0..xmu.ncols(),
            xmu.ncols()..xmu.ncols() + x_ls.ncols(),
            xmu,
            x_ls,
            &weights_a.hmumu,
            &weights_a.hmu_ls,
            &weights_a.h_ls_ls,
            &weights_a.dhmumu,
            &weights_a.dhmu_ls,
            &weights_a.dh_ls_ls,
        )?;
        let hessian_psi = if hessian_psi_operator.is_some() {
            Array2::zeros((0, 0))
        } else {
            gaussian_joint_psihessian_fromweights(xmu, x_ls, xmu_map, x_ls_map, &weights_a)?
        };

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_j) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_i,
                &dir_j,
                xmu,
                x_ls,
                None,
            )?,
        ))
    }

    fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_i: &LocationScaleJointPsiDirection,
        dir_j: &LocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_i,
            dir_j,
            xmu,
            x_ls,
        )?;
        let n = self.y.len();
        let xmu_i_map = dir_i.x_primary_psi.as_linear_map_ref();
        let x_ls_i_map = dir_i.x_ls_psi.as_linear_map_ref();
        let xmu_j_map = dir_j.x_primary_psi.as_linear_map_ref();
        let x_ls_j_map = dir_j.x_ls_psi.as_linear_map_ref();
        let xmu_ab_map = second_psi_linear_map(
            second_drifts.x_primary_ab_action.as_ref(),
            second_drifts.x_primary_ab.as_ref(),
            n,
            xmu.ncols(),
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            x_ls.ncols(),
        );
        // Second fixed-beta psi objects for the same Gaussian location-scale
        // kernel. Using the notation from the first-order comment, the rowwise
        // second psi drifts are
        //
        //   w_ab     = 4 w * ell_a * ell_b - 2 w * ell_ab
        //   alpha_ab = 2 w * (m_a * ell_b + m_b * ell_a)
        //              + 4 alpha * ell_a * ell_b
        //              - w * m_ab
        //              - 2 alpha * ell_ab
        //   b_ab     = 2 w * m_a * m_b
        //              + 4 alpha * (m_a * ell_b + m_b * ell_a)
        //              + 4 b * ell_a * ell_b
        //              - 2 alpha * m_ab
        //              - 2 b * ell_ab.
        //
        // The exact likelihood-only second-order objects are then:
        //
        //   D_ab,
        //   D_{beta ab},
        //   D_{beta beta ab},
        //
        // assembled from the usual product-rule expansion over realized
        // design motion X_{.,a}, X_{.,b}, X_{.,ab}. Generic code adds S_ab.
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let mut weights_i =
            gaussian_joint_psi_firstweights(&rows, &dir_i.z_primary_psi, &dir_i.z_ls_psi);
        let mut weights_j =
            gaussian_joint_psi_firstweights(&rows, &dir_j.z_primary_psi, &dir_j.z_ls_psi);
        let mut secondweights = gaussian_joint_psisecondweights(
            &rows,
            &dir_i.z_primary_psi,
            &dir_i.z_ls_psi,
            &dir_j.z_primary_psi,
            &dir_j.z_ls_psi,
            &second_drifts.z_primary_ab,
            &second_drifts.z_ls_ab,
        );
        if let Some(sub_rows) = subsample {
            // HT mask: every downstream consumer (gaussian_joint_psisecondhessian_fromweights,
            // weighted_crossprod_psi_maps with weights_*.{hmumu,hmu_ls,h_ls_ls},
            // fast_atv on d2score_* and dscore_*) is row-linear in these arrays, so
            // scaling sampled rows by 1/π_i and zeroing the rest yields an unbiased
            // estimator of the full-data second-order ψ Hessian and ψ score.
            apply_ht_mask_first(&mut weights_i, sub_rows);
            apply_ht_mask_first(&mut weights_j, sub_rows);
            apply_ht_mask_second(&mut secondweights, sub_rows);
        }
        let objective_psi_psi = secondweights.objective_psi_psirow.sum();

        let score_psi_psi = gaussian_pack_joint_score(
            &(xmu_ab_map.transpose_mul(weights_i.scoremu.view())
                + xmu_i_map.transpose_mul(weights_j.dscoremu.view())
                + xmu_j_map.transpose_mul(weights_i.dscoremu.view())
                + fast_atv(xmu, &secondweights.d2scoremu)),
            &(x_ls_ab_map.transpose_mul(weights_i.score_ls.view())
                + x_ls_i_map.transpose_mul(weights_j.dscore_ls.view())
                + x_ls_j_map.transpose_mul(weights_i.dscore_ls.view())
                + fast_atv(x_ls, &secondweights.d2score_ls)),
        );
        let hessian_psi_psi = gaussian_joint_psisecondhessian_fromweights(
            xmu,
            x_ls,
            xmu_i_map,
            x_ls_i_map,
            xmu_j_map,
            x_ls_j_map,
            xmu_ab_map,
            x_ls_ab_map,
            &weights_i,
            &weights_j,
            &secondweights,
        )?;

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        })
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                xmu,
                x_ls,
                None,
            )?,
        ))
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<Array2<f64>, String> {
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let total = pmu + p_ls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi hessian directional derivative length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                total
            ) }.into());
        }
        // Only the log-σ–channel direction enters the surviving Fisher blocks
        // of the mixed drift (the μ-channel direction fed the observed cross
        // block, now Fisher 0; μ ⊥ σ, #684).
        let u_ls = d_beta_flat.slice(s![pmu..pmu + p_ls]);
        let xi_ls = fast_av(x_ls, &u_ls);
        let uza_ls = x_ls_map.forward_mul(u_ls);
        // Mixed drift T_a[u] = D_beta H_a^{(D)}[u] for the Gaussian family.
        //
        // Along u = [umu; u_sigma], define xi = Xmu umu and zeta = X_sigma u_sigma.
        // The first beta-directional drifts of the Gaussian row scalars are
        //
        //   d_u w     = -2 w * zeta
        //   d_u alpha = -w * xi - 2 alpha * zeta
        //   d_u b     = -2 alpha * xi - 2 b * zeta.
        //
        // Differentiating the psi-a scalar drifts once more gives
        //
        //   d_u w_a     = 4 w * ell_a * zeta - 2 w * zeta_a
        //   d_u alpha_a = 2 w * (m_a * zeta + ell_a * xi)
        //                 - w * xi_a
        //                 + 4 alpha * ell_a * zeta
        //                 - 2 alpha * zeta_a
        //   d_u b_a     = 2 w * m_a * xi
        //                 + 4 alpha * (m_a * zeta + ell_a * xi)
        //                 + 4 b * ell_a * zeta
        //                 - 2 alpha * xi_a
        //                 - 2 b * zeta_a.
        //
        // The matrix drift returned here is the exact likelihood-only
        //
        //   T_a[u] = D_beta H_{psi_a}^{(D)}[u],
        //
        // assembled blockwise as
        //
        //   Kmumu,a[u]   = Xmu^T W_a[u] Xmu
        //                   + X_{mu,a}^T W[u] Xmu
        //                   + Xmu^T W[u] X_{mu,a}
        //   Kmusigma,a[u]= 2( Xmu^T A_a[u] X_sigma
        //                   + X_{mu,a}^T A[u] X_sigma
        //                   + Xmu^T A[u] X_{sigma,a} )
        //   K_sigmasigma,a[u]
        //                   = 2( X_sigma^T B_a[u] X_sigma
        //                   + X_{sigma,a}^T B[u] X_sigma
        //                   + X_sigma^T B[u] X_{sigma,a} ).
        //
        // Generic code then combines this with S(theta)-motion and the profile
        // mode responses to form ddot H_{ij}.
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let mut mixedweights =
            gaussian_joint_psi_mixed_driftweights(&rows, &xi_ls, &dir_a.z_ls_psi, &uza_ls);
        if let Some(sub_rows) = subsample {
            // HT mask: `gaussian_joint_psi_mixedhessian_drift_fromweights` is
            // row-linear in every `mixedweights.*` array via `xt_diag_*_dense`
            // and `weighted_crossprod_psi_maps`, so the masked Hessian-drift
            // remains an unbiased estimator of the full-data drift.
            apply_ht_mask_mixed(&mut mixedweights, sub_rows);
        }

        gaussian_joint_psi_mixedhessian_drift_fromweights(
            xmu,
            x_ls,
            xmu_map,
            x_ls_map,
            &mixedweights,
        )
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx` given the
    /// realised block specs.  Returns an [`AdditiveBlockJacobian`] encoding the
    /// linear map η_r[i] = X_r[i,:] · β_r:
    ///
    /// - block 0 (mu):       output 0 = design rows, output 1 = zeros
    /// - block 1 (log_sigma): output 0 = zeros, output 1 = design rows
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::util::block_jacobian::AdditiveWiggleBlockLayout {
            family: "GaussianLocationScaleFamily",
            n_outputs: 2,
            additive_blocks: &[Self::BLOCK_MU, Self::BLOCK_LOG_SIGMA],
            wiggle_block: None,
        }
        .block_effective_jacobian(specs, block_idx)
    }
}


/// Per-subject 2×2 channel Hessian `W_i` for Gaussian location-scale.
///
/// The row negative log-likelihood (with per-row weight `w_i`, response `y_i`,
/// mean predictor `μ_i`, log-scale predictor `s_i = log σ_i`) is
///
/// ```text
/// ρ_i(μ, s) = w_i [s + 0.5·(y_i − μ)²·exp(−2s)]
/// ```
///
/// The 2×2 Hessian in `(μ, s)` coordinates:
///
/// ```text
/// W_i[0,0] = w_i · exp(−2 s_i)                        ∂²ρ/∂μ²
/// W_i[1,1] = w_i · 2·(y_i − μ_i)²·exp(−2 s_i)        ∂²ρ/∂s²
/// W_i[0,1] = W_i[1,0] = w_i · 2·(y_i − μ_i)·exp(−2 s_i)  ∂²ρ/∂μ∂s
/// ```
///
/// The off-diagonal cross-channel term `∂²ρ/∂μ∂s` is nonzero whenever the
/// residual `(y_i − μ_i) ≠ 0`, i.e. away from the fitted mean.
pub struct GaussianLocationScaleChannelHessian {
    /// Row-major `(n × 2 × 2)` PSD-clamped per-subject Hessian.
    h: ndarray::Array3<f64>,
}


impl GaussianLocationScaleChannelHessian {
    /// Construct the raw (un-PSD-clamped) per-subject observed Hessian.
    ///
    /// For Gaussian location-scale the 2×2 observed Hessian
    /// `[[w·e^{-2s}, 2·w·r·e^{-2s}], [2·w·r·e^{-2s}, 2·w·r²·e^{-2s}]]`
    /// has determinant `-2·w²·r²·e^{-4s}` which is non-positive whenever
    /// the residual `r = y − μ ≠ 0`. Tests that finite-difference the row
    /// NLL must compare against this raw observed Hessian — PSD clamping
    /// alters the eigenvalues and the FD-versus-closed-form match fails.
    ///
    /// Production code that needs a PSD matrix (e.g. the canonicalize gate)
    /// must call [`Self::from_pilot`] which PSD-clamps via 2×2
    /// eigendecomposition.
    pub fn from_pilot_observed_unclamped(
        y: &ndarray::Array1<f64>,
        w: &ndarray::Array1<f64>,
        eta_mu: &ndarray::Array1<f64>,
        eta_log_sigma: &ndarray::Array1<f64>,
    ) -> Result<Self, String> {
        let n = y.len();
        if w.len() != n || eta_mu.len() != n || eta_log_sigma.len() != n {
            return Err(format!(
                "GaussianLocationScaleChannelHessian::from_pilot_observed_unclamped: \
                 length mismatch y={n} w={} eta_mu={} eta_log_sigma={}",
                w.len(),
                eta_mu.len(),
                eta_log_sigma.len(),
            ));
        }
        let mut h = ndarray::Array3::<f64>::zeros((n, 2, 2));
        for i in 0..n {
            let wi = w[i];
            let mu_i = eta_mu[i];
            let s_i = eta_log_sigma[i];
            let inv_sigma2 = (-2.0 * s_i).exp();
            let resid = y[i] - mu_i;
            h[[i, 0, 0]] = wi * inv_sigma2;
            h[[i, 1, 1]] = wi * 2.0 * resid * resid * inv_sigma2;
            h[[i, 0, 1]] = wi * 2.0 * resid * inv_sigma2;
            h[[i, 1, 0]] = h[[i, 0, 1]];
        }
        Ok(Self { h })
    }

    /// Construct from pilot predictors (μ and log σ at current β) and data,
    /// with PSD eigenvalue clamping applied per subject.
    ///
    /// `y` is the response, `w` the per-row sample weights, `eta_mu` and
    /// `eta_log_sigma` the current linear predictors. Negative eigenvalues
    /// are projected to zero (PSD clamp) before storage so the resulting
    /// matrix is a valid metric for the W-Gram identifiability compile.
    pub fn from_pilot(
        y: &ndarray::Array1<f64>,
        w: &ndarray::Array1<f64>,
        eta_mu: &ndarray::Array1<f64>,
        eta_log_sigma: &ndarray::Array1<f64>,
    ) -> Result<Self, String> {
        let n = y.len();
        if w.len() != n || eta_mu.len() != n || eta_log_sigma.len() != n {
            return Err(format!(
                "GaussianLocationScaleChannelHessian::from_pilot: \
                 length mismatch y={n} w={} eta_mu={} eta_log_sigma={}",
                w.len(),
                eta_mu.len(),
                eta_log_sigma.len(),
            ));
        }
        let mut h = ndarray::Array3::<f64>::zeros((n, 2, 2));
        for i in 0..n {
            let wi = w[i];
            let mu_i = eta_mu[i];
            let s_i = eta_log_sigma[i];
            let inv_sigma2 = (-2.0 * s_i).exp(); // exp(-2s) = 1/sigma^2
            let resid = y[i] - mu_i;
            // Hessian of w_i * ρ_i
            let h00 = wi * inv_sigma2;
            let h11 = wi * 2.0 * resid * resid * inv_sigma2;
            let h01 = wi * 2.0 * resid * inv_sigma2;
            // PSD clamp via eigendecomposition of 2×2 matrix.
            // psd_clamp_2x2 returns (λ1, λ2, u1[0], u1[1], u2[0], u2[1])
            // where u1 and u2 are unit eigenvectors for λ1 and λ2.
            // Reconstruction: H_psd = λ1·u1·u1ᵀ + λ2·u2·u2ᵀ
            let (e0, e1, u1_0, u1_1, u2_0, u2_1) = psd_clamp_2x2(h00, h01, h11);
            h[[i, 0, 0]] = e0 * u1_0 * u1_0 + e1 * u2_0 * u2_0;
            h[[i, 0, 1]] = e0 * u1_0 * u1_1 + e1 * u2_0 * u2_1;
            h[[i, 1, 0]] = h[[i, 0, 1]];
            h[[i, 1, 1]] = e0 * u1_1 * u1_1 + e1 * u2_1 * u2_1;
        }
        Ok(Self { h })
    }
}


impl FamilyChannelHessian for GaussianLocationScaleChannelHessian {
    fn n_outputs(&self) -> usize {
        2
    }

    fn n_subjects(&self) -> usize {
        self.h.shape()[0]
    }

    fn fill_subject(&self, i: usize, out: &mut [f64]) {
        assert_eq!(out.len(), 4);
        out[0] = self.h[[i, 0, 0]];
        out[1] = self.h[[i, 0, 1]];
        out[2] = self.h[[i, 1, 0]];
        out[3] = self.h[[i, 1, 1]];
    }

    fn evaluate_full(&self) -> ndarray::Array3<f64> {
        self.h.clone()
    }
}


impl CustomFamily for GaussianLocationScaleFamily {
    /// The Gaussian location-scale joint Hessian depends on β because the
    /// cross-block (μ,log σ) and (log σ,log σ) blocks contain the residual
    /// r = y − μ (via the row scalars m = r·w and n = r²·w), which changes
    /// when β_μ moves.  The (μ,μ) block weight w = 1/σ² also depends on
    /// β_{log σ}.  This override is essential for correct M_j[u] drift
    /// corrections when ψ hyperparameters move the design matrices.
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    /// Two independent linear predictors: block 0 → μ channel, block 1 → log σ
    /// channel. Declaring the channel topology lets `fit_custom_family` route
    /// the identifiability audit channel-aware even when a caller builds the
    /// blocks by hand (without `build_location_scale_block`'s callbacks), so a
    /// shared μ/log-σ covariate basis is recognised as block-diagonal rather
    /// than mistaken for cross-block intercept aliases (#558).
    fn output_channel_assignment(&self, specs: &[ParameterBlockSpec]) -> Option<Vec<usize>> {
        // Two-channel families: `[mu, log_sigma]`. The optional trailing
        // zero-channel wiggle block (when present) also drives channel 0.
        Some(
            (0..specs.len())
                .map(|i| usize::from(i == Self::BLOCK_LOG_SIGMA))
                .collect(),
        )
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware: when the unified evaluator picks the matrix-free
        // joint Hessian path (see `use_joint_matrix_free_path`), the workspace
        // applies the joint Hessian via row-streaming Khatri-Rao matvecs at
        // O(n · (p_t + p_ℓ)) per Hv, never building the dense (p_t + p_ℓ)²
        // matrix. Report the operator work model so diagnostics and
        // first-order-only policies reflect the representation that actually
        // runs.
        crate::families::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        // Diagonal IRLS weights for the inner solver.
        //
        // For the location block (identity link): wmu = pw / sigma^2. Since the
        // location link is identity, observed = Fisher --- no correction needed.
        //
        // For the log-sigma block (log link): w_ls = 2 * pw * (dsigma/deta)^2 / sigma^2.
        // This is the Fisher weight. For the outer REML, the joint
        // `exact_newton_joint_hessian` provides the full observed Hessian directly,
        // so these Diagonal weights are only used for the inner IRLS iteration
        // (where Fisher scoring is fine). See response.md Section 3.
        //
        let mut zmu = Array1::<f64>::zeros(n);
        let mut wmu = Array1::<f64>::zeros(n);
        let mut z_ls = Array1::<f64>::zeros(n);
        let mut w_ls = Array1::<f64>::zeros(n);
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;

        const CHUNK: usize = 1024;
        if let (
            Some(y_s),
            Some(w_s),
            Some(mu_s),
            Some(ls_s),
            Some(zmu_s),
            Some(wmu_s),
            Some(zls_s),
            Some(wls_s),
        ) = (
            self.y.as_slice_memory_order(),
            self.weights.as_slice_memory_order(),
            etamu.as_slice_memory_order(),
            eta_log_sigma.as_slice_memory_order(),
            zmu.as_slice_memory_order_mut(),
            wmu.as_slice_memory_order_mut(),
            z_ls.as_slice_memory_order_mut(),
            w_ls.as_slice_memory_order_mut(),
        ) {
            // Per-row Gaussian LS kernel writes 4 working arrays directly into
            // the output slices; ll is reduced via Rayon's sum. Independent
            // across rows.
            ll += zmu_s
                .par_chunks_mut(CHUNK)
                .zip(wmu_s.par_chunks_mut(CHUNK))
                .zip(zls_s.par_chunks_mut(CHUNK))
                .zip(wls_s.par_chunks_mut(CHUNK))
                .enumerate()
                .map(|(chunk_idx, (((zmu_c, wmu_c), zls_c), wls_c))| {
                    let start = chunk_idx * CHUNK;
                    let mut local_ll = 0.0;
                    for local in 0..zmu_c.len() {
                        let i = start + local;
                        let row =
                            gaussian_diagonal_row_kernel(y_s[i], mu_s[i], ls_s[i], w_s[i], ln2pi);
                        zmu_c[local] = mu_s[i] + row.location_working_shift;
                        wmu_c[local] = row.location_working_weight;
                        zls_c[local] = row.log_sigma_working_response;
                        wls_c[local] = row.log_sigma_working_weight;
                        local_ll += row.log_likelihood;
                    }
                    local_ll
                })
                .sum::<f64>();
        } else {
            // Fallback path: inputs are not contiguous. Outputs (just-allocated
            // Array1::zeros) always are. Reborrow input views into the closure.
            let y_view = self.y.view();
            let w_view = self.weights.view();
            let mu_view = etamu.view();
            let ls_view = eta_log_sigma.view();
            let zmu_s = zmu
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let wmu_s = wmu
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let zls_s = z_ls
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let wls_s = w_ls
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            ll += zmu_s
                .par_chunks_mut(CHUNK)
                .zip(wmu_s.par_chunks_mut(CHUNK))
                .zip(zls_s.par_chunks_mut(CHUNK))
                .zip(wls_s.par_chunks_mut(CHUNK))
                .enumerate()
                .map(|(chunk_idx, (((zmu_c, wmu_c), zls_c), wls_c))| {
                    let start = chunk_idx * CHUNK;
                    let mut local_ll = 0.0;
                    for local in 0..zmu_c.len() {
                        let i = start + local;
                        let row = gaussian_diagonal_row_kernel(
                            y_view[i], mu_view[i], ls_view[i], w_view[i], ln2pi,
                        );
                        zmu_c[local] = mu_view[i] + row.location_working_shift;
                        wmu_c[local] = row.location_working_weight;
                        zls_c[local] = row.log_sigma_working_response;
                        wls_c[local] = row.log_sigma_working_weight;
                        local_ll += row.log_likelihood;
                    }
                    local_ll
                })
                .sum::<f64>();
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::diagonal_checked(zmu, wmu)?,
                BlockWorkingSet::diagonal_checked(z_ls, w_ls)?,
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }
        // logb noise link: σ(η_ls) = LOGB_SIGMA_FLOOR + exp(η_ls). σ ≥ b > 0
        // bounds the loglik below (−Σlog σ ≥ −n log b) and bounds 1/σ² by 1/b²,
        // so the previous `inv_s2.min(1e24)` cap is structurally unnecessary.
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;
        if let (Some(y_s), Some(w_s), Some(mu_s), Some(ls_s)) = (
            self.y.as_slice_memory_order(),
            self.weights.as_slice_memory_order(),
            etamu.as_slice_memory_order(),
            eta_log_sigma.as_slice_memory_order(),
        ) {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            ll += (0..n)
                .into_par_iter()
                .map(|i| {
                    let wi = w_s[i];
                    if wi == 0.0 {
                        return 0.0;
                    }
                    let sigma_i = logb_sigma_from_eta_scalar(ls_s[i]);
                    let inv_s2 = (sigma_i * sigma_i).recip();
                    let r = y_s[i] - mu_s[i];
                    wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
                })
                .sum::<f64>();
        } else {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            ll += (0..n)
                .into_par_iter()
                .map(|i| {
                    let wi = self.weights[i];
                    if wi == 0.0 {
                        return 0.0;
                    }
                    let sigma_i = logb_sigma_from_eta_scalar(eta_log_sigma[i]);
                    let inv_s2 = (sigma_i * sigma_i).recip();
                    let r = self.y[i] - etamu[i];
                    wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
                })
                .sum::<f64>();
        }
        Ok(ll)
    }

    /// Outer-only log-likelihood with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `Some`, only the sampled rows
    /// contribute; each row's per-row log-likelihood term is multiplied by
    /// `WeightedOuterRow.weight`, the Horvitz–Thompson inverse-inclusion
    /// factor 1/π_i (uniform or stratified sampling both supported), so the
    /// partial sum is an unbiased estimator of the full-data log-likelihood.
    /// When `None`, this returns the full-data `log_likelihood_only`. Inner
    /// PIRLS line searches never install the subsample option, so they
    /// continue to score the exact full-data log-likelihood.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let Some(subsample) = options.outer_score_subsample.as_ref() else {
            return self.log_likelihood_only(block_states);
        };
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        use rayon::iter::ParallelIterator;
        let ll: f64 = subsample
            .rows
            .par_iter()
            .map(|row| {
                let i = row.index;
                let wi = self.weights[i];
                if wi == 0.0 {
                    return 0.0;
                }
                let sigma_i = logb_sigma_from_eta_scalar(eta_log_sigma[i]);
                let inv_s2 = (sigma_i * sigma_i).recip();
                let r = self.y[i] - etamu[i];
                row.weight * wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
            })
            .sum();
        Ok(ll)
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, None)
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    /// The Gaussian location-scale likelihood has no separation /
    /// under-identification regime that the full-span Jeffreys curvature `H_Φ`
    /// is meant to regularize: with the soft floor `σ ≥ b > 0` the per-row
    /// Fisher information `diag(a/σ², 2κ²a)` is bounded and `O(n)` on every
    /// identified direction at every working point, so the well-conditioned-`H`
    /// Jeffreys gate smooth-steps `H_Φ` to ~0 — yet the matching score `∇Φ`
    /// kept leaking a *phantom* penalized-stationarity residual into the inner
    /// joint-Newton (a nonzero `|∇L − Sβ|` paired with a numerically null `H_Φ`
    /// and a full-rank `H_pen`), so the KKT certificate refused every iterate
    /// and the outer REML rejected all seeds — aborting heteroscedastic
    /// location-scale fits (#684–#688). This is the same opt-out
    /// `TransformationNormalFamily` takes for the same structural reason
    /// (continuous response, `O(n)` Fisher information everywhere); it removes
    /// the phantom residual and drops the per-cycle `O(n·p²)` Jeffreys
    /// directional-derivative overhead.
    fn joint_jeffreys_term_required(&self) -> bool {
        false
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n || d_eta.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let sigma = eta_ls.mapv(logb_sigma_from_eta_scalar);
        let mut dw = Array1::<f64>::zeros(n);
        match block_idx {
            Self::BLOCK_MU => {
                // Gaussian location block:
                //
                //   wmu = weight / sigma^2.
                //
                // This depends only on the scale predictor, so along a
                // location-only direction d etamu the directional derivative is
                // identically zero.
                Ok(Some(dw))
            }
            Self::BLOCK_LOG_SIGMA => {
                // Gaussian log-sigma block:
                //
                // The PIRLS information weight is
                //
                //   w_ls = max(2 * weight * clamp(g, -1, 1)^2, MIN_WEIGHT),
                //   g    = sigma'(eta_ls) / sigma(eta_ls),
                // with the semantic rule that zero observation weights stay zero.
                //
                // Along a direction d eta_ls,
                //
                //   dw_ls is the directional derivative of that piecewise
                //   definition. On the active clamp branch or active MIN_WEIGHT
                //   floor branch, the returned derivative is zero to match the
                //   selected local piece of the evaluated weight.
                //
                // This is the exact directional derivative needed by the REML
                // trace term
                //
                //   0.5 tr(J^{-1} D_beta J[u])
                //   = 0.5 sum_i (x_i^T J^{-1} x_i) dw_i
                //
                // for diagonal working-set blocks.
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                let dw_vec: Vec<f64> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let d1 = crate::families::sigma_link::logb_sigma_jet1_scalar(eta_ls[i]).d1;
                        gaussian_log_sigma_irlsinfo_directional_derivative(
                            self.weights[i],
                            sigma[i],
                            d1,
                            d_eta[i],
                        )
                    })
                    .collect();
                for (i, v) in dw_vec.into_iter().enumerate() {
                    dw[i] = v;
                }
                Ok(Some(dw))
            }
            _ => Ok(None),
        }
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, Some(specs))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if block_states.len() != 2 || specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi workspace expects 2 states, 2 specs, and 2 derivative block lists, got {} / {} / {}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    /// Outer-aware joint ψ workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_psi_workspace`. When `Some`, the subsample is
    /// stored in the workspace and forwarded into every per-row weight array
    /// produced by `gaussian_joint_psi_firstweights`,
    /// `gaussian_joint_psisecondweights`, and
    /// `gaussian_joint_psi_mixed_driftweights`: each sampled row's
    /// contribution is multiplied by `WeightedOuterRow.weight = 1/π_i` and
    /// non-sampled rows are zeroed. Every downstream assembly
    /// (`gaussian_joint_psi*_fromweights`, `weighted_crossprod_psi_maps`,
    /// `xt_diag_*_dense`,
    /// `build_two_block_custom_family_joint_psi_operator_from_actions`) is
    /// row-linear in these arrays via `Xᵀ diag(W) Y`, so the resulting
    /// second-order ψ Hessian and ψ-Hessian directional derivative are
    /// unbiased Horvitz–Thompson estimators of the full-data quantities.
    /// Inner-PIRLS and final-covariance paths never install the option.
    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if block_states.len() != 2 || specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi workspace expects 2 states, 2 specs, and 2 derivative block lists, got {} / {} / {}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleExactNewtonJointPsiWorkspace::new_with_subsample(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
                options.outer_score_subsample.clone(),
            )?,
        )))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = GaussianLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-aware joint-Hessian workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_hessian_workspace`. When `Some`, the precomputed
    /// per-row coefficient arrays (`coeff_mm`, `coeff_ml`, `coeff_ll`) — which
    /// every downstream assembly (`hessian_dense`, `hessian_matvec`,
    /// `hessian_diagonal`) consumes row-linearly via `Xᵀ diag(W) X` — are
    /// replaced by a Horvitz–Thompson mask: each sampled row's coefficient is
    /// multiplied by `WeightedOuterRow.weight` (the inverse-inclusion factor
    /// 1/π_i; uniform or stratified sampling both supported), and non-sampled
    /// rows are zeroed. The resulting joint Hessian is an unbiased estimator
    /// of the full-data joint Hessian. Inner PIRLS never installs the option,
    /// so the inner solve continues to consume the exact full-data Hessian.
    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let mut workspace = GaussianLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // The Gaussian location-scale workspace is returned by
        // `exact_newton_joint_hessian_workspace` whenever
        // `exact_joint_dense_block_designs` succeeds, which itself depends on
        // both block designs being present. This is only a β-space operator
        // capability; outer θθ Hessian availability is declared separately.
        self.exact_joint_supported()
            && matches!(
                self.exact_joint_dense_block_designs(Some(specs)),
                Ok(Some(_))
            )
    }

    /// Outer-derivative policy: declare HT-subsample capability.
    ///
    /// GaussianLocationScaleFamily overrides
    /// `log_likelihood_only_with_options`,
    /// `exact_newton_joint_hessian_workspace_with_options`, and
    /// `exact_newton_joint_psi_workspace_with_options` to consume
    /// `options.outer_score_subsample` with per-row Horvitz–Thompson weights
    /// (each sampled row's contribution is multiplied by
    /// `WeightedOuterRow.weight = 1/π_i`; non-sampled rows are zeroed),
    /// yielding unbiased estimators of the full-data log-likelihood, joint
    /// Hessian, and second-order ψ Hessian / ψ-Hessian directional
    /// derivative. The ψ-workspace masking happens inside
    /// `apply_ht_mask_first`, `apply_ht_mask_second`, and
    /// `apply_ht_mask_mixed` on the `GaussianJointPsi{First,Second,
    /// MixedDrift}Weights` per-row arrays, immediately after the row-scalar
    /// reductions and before the row-linear `weighted_crossprod_psi_maps` /
    /// `xt_diag_*_dense` assemblies, so the masked outputs remain unbiased.
    /// First-order ψ terms remain full-data exact (= trivially unbiased), so
    /// the total outer score is still unbiased. Inner-PIRLS and final-
    /// covariance paths never install the option, so they continue to
    /// consume the exact full-data quantities.
    fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }
}


impl CustomFamilyGenerative for GaussianLocationScaleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let mu = block_states[Self::BLOCK_MU].eta.clone();
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let sigma = gamlss_rowwise_map(eta_log_sigma.len(), |i| {
            logb_sigma_from_eta_scalar(eta_log_sigma[i])
        });
        Ok(GenerativeSpec {
            mean: mu,
            noise: NoiseModel::Gaussian { sigma },
        })
    }
}


/// One channel of a `RowCoeffOperator`: a row-major `Arc<Array2<f64>>`
/// design matrix indexed by row coefficient pairs. Channels with the same
/// `block` value contribute their `X^T r` outputs into the same coefficient
/// block of the joint vector (e.g. wiggle's basis B and basis_d1 are two
/// channels that both contribute to the wiggle output block).
struct RowCoeffChannel {
    block: usize,
    design: Arc<Array2<f64>>,
}


/// Symmetric pair coefficients `c_{ab}` for `a ≤ b`. The operator adds
/// `X_a^T diag(c_{ab}) X_b` to block `block_a`'s output and the transpose
/// contribution `X_b^T diag(c_{ab}) X_a` to block `block_b` when `a != b`.
struct RowCoeffPair {
    a: usize,
    b: usize,
    coeff: Array1<f64>,
}


/// Pooled per-call scratch for `RowCoeffOperator::mul_vec`. Each call
/// pops a buffer set; if the pool is empty (parallel callers exhausted
/// it) we allocate fresh — the alloc is amortized as concurrent callers
/// recycle. The pool's `Mutex` is taken only for `pop`/`push` (constant
/// time), never during the matmul.
///
/// **Invariant**: every buffer in `pool[k].u[ch]` and `pool[k].r[ch]` has
/// length `nrows`. `mul_vec` overwrites `u` via `fast_av_into` and
/// zeroes-then-accumulates `r`, leaving both buffers in any state on
/// return — callers must not depend on residual content.
struct RowCoeffScratch {
    u: Vec<Array1<f64>>,
    r: Vec<Array1<f64>>,
}


/// Matrix-free operator for two-block-style joint-Hessian directional
/// derivatives that decompose as `H = sum_{a,b} X_a^T diag(c_{ab}) X_b`
/// with each `X_a` an `n × p_a` design and `c_{ab}` an `n` row coefficient
/// vector. `mul_vec` applies the operator in O(n · sum_a p_a) per call,
/// reusing pre-sized scratch buffers for `u`, `r` from a small lock-pool
/// so concurrent `mul_vec` callers do not serialize on the same scratch.
///
/// `block_offsets` gives the starting column of each output block; the
/// operator dimension is the sum of all block widths. Each channel's
/// `mul_vec` contribution is added into the slice for its output block.
struct RowCoeffOperator {
    channels: Vec<RowCoeffChannel>,
    block_offsets: Vec<usize>,
    block_widths: Vec<usize>,
    dim: usize,
    pair_coeffs: Vec<RowCoeffPair>,
    nrows: usize,
    scratch_pool: std::sync::Mutex<Vec<RowCoeffScratch>>,
}


impl RowCoeffOperator {
    /// One-line constructor for the standard (channels, pair-coeffs)
    /// recipe used by every GAMLSS LS workspace: pass the block widths,
    /// the channel list as `(block_id, design)` tuples, and the pair
    /// list as `(a, b, coeff)` tuples. Pre-allocates one scratch in the
    /// pool so the first warm `mul_vec` call skips allocation.
    fn from_directions(
        block_widths: Vec<usize>,
        channels: Vec<(usize, Arc<Array2<f64>>)>,
        pairs: Vec<(usize, usize, Array1<f64>)>,
        nrows: usize,
    ) -> Self {
        let channels: Vec<RowCoeffChannel> = channels
            .into_iter()
            .map(|(block, design)| RowCoeffChannel { block, design })
            .collect();
        let pair_coeffs: Vec<RowCoeffPair> = pairs
            .into_iter()
            .map(|(a, b, coeff)| RowCoeffPair { a, b, coeff })
            .collect();
        let mut block_offsets = Vec::with_capacity(block_widths.len());
        let mut acc = 0;
        for w in &block_widths {
            block_offsets.push(acc);
            acc += *w;
        }
        let n_ch = channels.len();
        let initial = RowCoeffScratch {
            u: (0..n_ch).map(|_| Array1::<f64>::zeros(nrows)).collect(),
            r: (0..n_ch).map(|_| Array1::<f64>::zeros(nrows)).collect(),
        };
        Self {
            channels,
            block_offsets,
            block_widths,
            dim: acc,
            pair_coeffs,
            nrows,
            scratch_pool: std::sync::Mutex::new(vec![initial]),
        }
    }

    fn acquire_scratch(&self) -> RowCoeffScratch {
        self.scratch_pool
            .lock()
            .expect("RowCoeffOperator scratch pool poisoned")
            .pop()
            .unwrap_or_else(|| {
                let n_ch = self.channels.len();
                RowCoeffScratch {
                    u: (0..n_ch)
                        .map(|_| Array1::<f64>::zeros(self.nrows))
                        .collect(),
                    r: (0..n_ch)
                        .map(|_| Array1::<f64>::zeros(self.nrows))
                        .collect(),
                }
            })
    }

    fn release_scratch(&self, scratch: RowCoeffScratch) {
        self.scratch_pool
            .lock()
            .expect("RowCoeffOperator scratch pool poisoned")
            .push(scratch);
    }

    fn projected_trace(&self, factor: &Array2<f64>) -> f64 {
        let grams = self.projected_pair_gram_table(factor);
        self.trace_from_pair_gram_table(grams.view())
    }

    fn projected_pair_gram_cache_id(&self) -> usize {
        let mut hasher = DefaultHasher::new();
        "RowCoeffOperator::projected_pair_gram_table".hash(&mut hasher);
        self.nrows.hash(&mut hasher);
        self.dim.hash(&mut hasher);
        self.block_widths.hash(&mut hasher);
        self.block_offsets.hash(&mut hasher);
        self.channels.len().hash(&mut hasher);
        self.pair_coeffs.len().hash(&mut hasher);
        for (idx, ch) in self.channels.iter().enumerate() {
            idx.hash(&mut hasher);
            (Arc::as_ptr(&ch.design) as usize).hash(&mut hasher);
            ch.block.hash(&mut hasher);
            ch.design.nrows().hash(&mut hasher);
            ch.design.ncols().hash(&mut hasher);
            self.block_widths[ch.block].hash(&mut hasher);
        }
        for (idx, pair) in self.pair_coeffs.iter().enumerate() {
            idx.hash(&mut hasher);
            pair.a.hash(&mut hasher);
            pair.b.hash(&mut hasher);
        }
        hasher.finish() as usize
    }

    fn projected_pair_gram_table(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            factor.nrows(),
            self.dim,
            "row-coefficient cached projected trace factor row mismatch: factor rows={} but dim={}",
            factor.nrows(),
            self.dim
        );
        let rank = factor.ncols();
        let pair_count = self.pair_coeffs.len();
        if self.nrows == 0 || rank == 0 || pair_count == 0 {
            return Array2::<f64>::zeros((self.nrows, pair_count));
        }
        let rows_per_chunk =
            gamlss_projected_trace_chunk_rows(rank, self.channels.len(), pair_count)
                .min(self.nrows.max(1));
        let mut grams = Array2::<f64>::zeros((self.nrows, pair_count));
        let fill_chunk = |start: usize, mut out_chunk: ndarray::ArrayViewMut2<'_, f64>| {
            let end = (start + rows_per_chunk).min(self.nrows);
            let rows = start..end;
            let mut projected: Vec<Array2<f64>> = Vec::with_capacity(self.channels.len());
            for ch in &self.channels {
                let block_start = self.block_offsets[ch.block];
                let width = self.block_widths[ch.block];
                let design_chunk = ch.design.slice(s![rows.clone(), ..]);
                let factor_block = factor.slice(s![block_start..block_start + width, ..]);
                projected.push(fast_ab(&design_chunk, &factor_block));
            }
            for (pair_idx, pair) in self.pair_coeffs.iter().enumerate() {
                let u_a = &projected[pair.a];
                let u_b = &projected[pair.b];
                for local_i in 0..u_a.nrows() {
                    let mut value = 0.0;
                    for col in 0..rank {
                        value += u_a[[local_i, col]] * u_b[[local_i, col]];
                    }
                    out_chunk[[local_i, pair_idx]] = value;
                }
            }
        };
        if rayon::current_thread_index().is_none() && self.nrows > rows_per_chunk {
            grams
                .axis_chunks_iter_mut(Axis(0), rows_per_chunk)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    fill_chunk(chunk_idx * rows_per_chunk, out_chunk)
                });
        } else {
            for start in (0..self.nrows).step_by(rows_per_chunk) {
                let end = (start + rows_per_chunk).min(self.nrows);
                let out_chunk = grams.slice_mut(s![start..end, ..]);
                fill_chunk(start, out_chunk);
            }
        }
        grams
    }

    fn trace_from_pair_gram_table(&self, grams: ArrayView2<'_, f64>) -> f64 {
        assert_eq!(grams.nrows(), self.nrows);
        assert_eq!(grams.ncols(), self.pair_coeffs.len());
        let mut trace = 0.0;
        for i in 0..self.nrows {
            for (pair_idx, pair) in self.pair_coeffs.iter().enumerate() {
                let multiplier = if pair.a == pair.b { 1.0 } else { 2.0 };
                trace += multiplier * pair.coeff[i] * grams[[i, pair_idx]];
            }
        }
        trace
    }
}
