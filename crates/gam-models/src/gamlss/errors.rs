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
    /// A finite predictor/input produced a row likelihood or fitting-geometry
    /// quantity that cannot be represented by `f64`.  This is deliberately a
    /// typed refusal: changing the predictor, response, or curvature with a
    /// clamp/floor would fit a different objective.
    RowGeometryUnrepresentable {
        row: usize,
        quantity: &'static str,
        eta: f64,
        value: f64,
    },
}

impl std::fmt::Display for GamlssError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { reason }
            | Self::InvalidInput { reason }
            | Self::NonFinite { reason }
            | Self::UnsupportedConfiguration { reason }
            | Self::ConstraintViolation { reason }
            | Self::NumericalFailure { reason } => f.write_str(reason),
            Self::RowGeometryUnrepresentable {
                row,
                quantity,
                eta,
                value,
            } => write!(
                f,
                "GAMLSS row geometry is not representable at row {row}: {quantity} evaluated from eta={eta:?} produced {value:?}"
            ),
        }
    }
}

impl std::error::Error for GamlssError {}

impl From<GamlssError> for String {
    fn from(err: GamlssError) -> String {
        err.to_string()
    }
}

impl From<crate::block_layout::block_count::BlockCountMismatch> for GamlssError {
    fn from(err: crate::block_layout::block_count::BlockCountMismatch) -> GamlssError {
        GamlssError::DimensionMismatch {
            reason: err.message(),
        }
    }
}

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
pub(crate) fn logb_dlog_sigma_deta(sigma: f64, d_sigma_deta: f64) -> f64 {
    d_sigma_deta / sigma
}

/// Exact power-of-two decomposition `x = mantissa * 2^exponent` for a positive
/// finite `f64`, including subnormals. The mantissa lies in `[1, 2)`.
#[inline]
pub(crate) fn positive_frexp(x: f64) -> (f64, i32) {
    assert!(x.is_finite() && x > 0.0);
    let bits = x.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if raw_exp != 0 {
        let mantissa = f64::from_bits((1023_u64 << 52) | fraction);
        (mantissa, raw_exp - 1023)
    } else {
        let leading = 63_i32 - fraction.leading_zeros() as i32;
        let shift = 52_i32 - leading;
        let normalized = fraction << shift;
        let mantissa =
            f64::from_bits((1023_u64 << 52) | (normalized & ((1_u64 << 52) - 1)));
        (mantissa, -1022 - shift)
    }
}

#[inline]
pub(crate) fn scale_normalized_power_of_two(mut mantissa: f64, mut exponent: i32) -> f64 {
    while mantissa >= 2.0 {
        mantissa *= 0.5;
        exponent += 1;
    }
    while mantissa < 1.0 {
        mantissa *= 2.0;
        exponent -= 1;
    }
    if exponent > 1023 {
        return f64::INFINITY;
    }
    if exponent >= -1022 {
        let power = f64::from_bits(((exponent + 1023) as u64) << 52);
        return mantissa * power;
    }
    if exponent < -1075 {
        return 0.0;
    }
    let units = mantissa * 2.0_f64.powi(exponent + 1074);
    units * f64::from_bits(1)
}

/// Compute `a*b*c/d` for positive finite inputs while carrying the binary
/// exponent separately. Intermediate overflow/underflow therefore cannot
/// change a representable final result.
#[inline]
pub(crate) fn scaled_positive_product_quotient(a: f64, b: f64, c: f64, d: f64) -> f64 {
    assert!(a.is_finite() && a > 0.0);
    assert!(b.is_finite() && b > 0.0);
    assert!(c.is_finite() && c > 0.0);
    assert!(d.is_finite() && d > 0.0);
    let (ma, ea) = positive_frexp(a);
    let (mb, eb) = positive_frexp(b);
    let (mc, ec) = positive_frexp(c);
    let (md, ed) = positive_frexp(d);
    scale_normalized_power_of_two((ma * mb) * (mc / md), ea + eb + ec - ed)
}

#[inline]
pub(crate) fn scaled_signed_product3(a: f64, b: f64, c: f64) -> f64 {
    if a == 0.0 || b == 0.0 || c == 0.0 {
        return 0.0;
    }
    let sign = a.signum() * b.signum() * c.signum();
    sign * scaled_positive_product_quotient(a.abs(), b.abs(), c.abs(), 1.0)
}

#[inline]
pub(crate) fn gaussian_log_sigma_irlsinfo_directional_derivative(
    row: usize,
    eta: f64,
    weight: f64,
    sigma: f64,
    d_sigma_deta: f64,
    d_eta: f64,
) -> Result<f64, String> {
    if weight == 0.0 || d_eta == 0.0 {
        return Ok(0.0);
    }
    let g = logb_dlog_sigma_deta(sigma, d_sigma_deta);
    if !g.is_finite() || g <= 0.0 || g > 1.0 {
        return Err(GamlssError::RowGeometryUnrepresentable {
            row,
            quantity: "Gaussian log-scale link derivative",
            eta,
            value: g,
        }
        .into());
    }
    let info = scaled_positive_product_quotient(weight, g, g, 0.5);
    if !info.is_finite() || info <= 0.0 {
        return Err(GamlssError::RowGeometryUnrepresentable {
            row,
            quantity: "Gaussian log-scale Fisher information",
            eta,
            value: info,
        }
        .into());
    }
    let dw = scaled_signed_product3(info, 2.0 * (1.0 - g), d_eta);
    if !dw.is_finite() {
        return Err(GamlssError::RowGeometryUnrepresentable {
            row,
            quantity: "Gaussian log-scale Fisher-information directional derivative",
            eta,
            value: dw,
        }
        .into());
    }
    Ok(dw)
}

#[derive(Clone, Copy)]
pub(crate) struct GaussianDiagonalRowKernel {
    pub(crate) log_likelihood: f64,
    pub(crate) location_working_weight: f64,
    pub(crate) log_sigma_working_weight: f64,
    pub(crate) log_sigma_working_response: f64,
    pub(crate) joint_w: f64,
    pub(crate) joint_m: f64,
    pub(crate) joint_n: f64,
    pub(crate) standardized_residual: f64,
    pub(crate) inv_sigma: f64,
    pub(crate) kappa: f64,
    pub(crate) kappa_prime: f64,
    pub(crate) kappa_dprime: f64,
}

#[inline]
pub(crate) fn gaussian_diagonal_row_kernel(
    row: usize,
    y: f64,
    location_eta: f64,
    eta_log_sigma: f64,
    obs_weight: f64,
    ln2pi: f64,
) -> Result<GaussianDiagonalRowKernel, String> {
    if !y.is_finite() || !location_eta.is_finite() || !eta_log_sigma.is_finite() {
        return Err(GamlssError::NonFinite {
            reason: format!(
                "Gaussian location-scale requires finite row inputs at row {row}: y={y}, eta_mu={location_eta}, eta_log_sigma={eta_log_sigma}"
            ),
        }
        .into());
    }
    if !obs_weight.is_finite() || obs_weight < 0.0 {
        return Err(GamlssError::InvalidInput {
            reason: format!(
                "Gaussian location-scale requires finite non-negative weights; weight[{row}]={obs_weight}"
            ),
        }
        .into());
    }
    if obs_weight == 0.0 {
        return Ok(GaussianDiagonalRowKernel {
            log_likelihood: 0.0,
            location_working_weight: 0.0,
            log_sigma_working_weight: 0.0,
            log_sigma_working_response: eta_log_sigma,
            joint_w: 0.0,
            joint_m: 0.0,
            joint_n: 0.0,
            standardized_residual: 0.0,
            inv_sigma: 0.0,
            kappa: 0.0,
            kappa_prime: 0.0,
            kappa_dprime: 0.0,
        });
    }

    // logb noise link σ = b + exp(η) bounds σ ≥ b > 0 by construction, so the
    // Gaussian location-scale objective ½Σ(y−μ)²/σ² + Σlog σ is bounded below
    // for any finite data. Its working weight 1/σ² is bounded by 1/b², so
    // H_μμ has bounded condition number — no after-the-fact floor or cap is
    // needed (the previous (1e-12, 1e24) clamp was a numerical bandaid for the
    // pure-exp link's σ→0 singularity and is structurally unnecessary here).
    // ApproxKind: Exact — working weight analytically bounded in (0, 1/b²].
    let SigmaJet1 { sigma, d1 } = logb_sigma_jet1_scalar(eta_log_sigma);
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(GamlssError::RowGeometryUnrepresentable {
            row,
            quantity: "Gaussian scale link",
            eta: eta_log_sigma,
            value: sigma,
        }
        .into());
    }
    let kappa = logb_dlog_sigma_deta(sigma, d1);
    if !kappa.is_finite() || kappa <= 0.0 || kappa > 1.0 {
        return Err(GamlssError::RowGeometryUnrepresentable {
            row,
            quantity: "Gaussian log-scale link derivative",
            eta: eta_log_sigma,
            value: kappa,
        }
        .into());
    }
    let inv_sigma = sigma.recip();
    let location_working_weight =
        scaled_positive_product_quotient(obs_weight, inv_sigma, inv_sigma, 1.0);
    if !location_working_weight.is_finite() || location_working_weight <= 0.0 {
        return Err(GamlssError::RowGeometryUnrepresentable {
            row,
            quantity: "Gaussian location Fisher information",
            eta: location_eta,
            value: location_working_weight,
        }
        .into());
    }

    // Form (y-mu)/sigma without overflowing the subtraction. Same-sign finite
    // inputs cannot overflow on subtraction; opposite-sign inputs can, and in
    // that case separately scaling both terms is exact enough to expose whether
    // the final quotient is representable.
    let residual = y - location_eta;
    let standardized_residual = if residual.is_finite() {
        residual / sigma
    } else {
        y / sigma - location_eta / sigma
    };
    let standardized_residual_sq = standardized_residual * standardized_residual;
    if !standardized_residual.is_finite() || !standardized_residual_sq.is_finite() {
        return Err(GamlssError::RowGeometryUnrepresentable {
            row,
            quantity: "Gaussian standardized residual squared",
            eta: location_eta,
            value: standardized_residual_sq,
        }
        .into());
    }
    let joint_n = if standardized_residual == 0.0 {
        0.0
    } else {
        scaled_positive_product_quotient(
            obs_weight,
            standardized_residual.abs(),
            standardized_residual.abs(),
            1.0,
        )
    };
    let joint_m = if standardized_residual == 0.0 {
        0.0
    } else {
        scaled_signed_product3(obs_weight, standardized_residual, inv_sigma)
    };
    let log_sigma_working_weight =
        scaled_positive_product_quotient(obs_weight, kappa, kappa, 0.5);
    if !log_sigma_working_weight.is_finite() || log_sigma_working_weight <= 0.0 {
        return Err(GamlssError::RowGeometryUnrepresentable {
            row,
            quantity: "Gaussian log-scale Fisher information",
            eta: eta_log_sigma,
            value: log_sigma_working_weight,
        }
        .into());
    }
    let log_sigma_step = (standardized_residual_sq - 1.0) / (2.0 * kappa);
    let log_sigma_working_response = eta_log_sigma + log_sigma_step;
    if !log_sigma_working_response.is_finite() {
        return Err(GamlssError::RowGeometryUnrepresentable {
            row,
            quantity: "Gaussian log-scale working response",
            eta: eta_log_sigma,
            value: log_sigma_working_response,
        }
        .into());
    }
    let likelihood_core = standardized_residual_sq + ln2pi + 2.0 * sigma.ln();
    let log_likelihood = if likelihood_core == 0.0 {
        0.0
    } else {
        -scaled_signed_product3(0.5, obs_weight, likelihood_core)
    };
    if !log_likelihood.is_finite() || !joint_m.is_finite() || !joint_n.is_finite() {
        let (quantity, value) = if !log_likelihood.is_finite() {
            ("Gaussian row log likelihood", log_likelihood)
        } else if !joint_m.is_finite() {
            ("Gaussian location score", joint_m)
        } else {
            ("Gaussian squared standardized residual weight", joint_n)
        };
        return Err(GamlssError::RowGeometryUnrepresentable {
            row,
            quantity,
            eta: eta_log_sigma,
            value,
        }
        .into());
    }
    let kappa_prime = kappa * (1.0 - kappa);
    let kappa_dprime = kappa_prime * (1.0 - 2.0 * kappa);

    Ok(GaussianDiagonalRowKernel {
        log_likelihood,
        location_working_weight,
        log_sigma_working_weight,
        log_sigma_working_response,
        joint_w: location_working_weight,
        joint_m,
        joint_n,
        standardized_residual,
        inv_sigma,
        kappa,
        kappa_prime,
        kappa_dprime,
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
