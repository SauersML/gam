//! Fit-setup plumbing: the per-surface blockspec builders, guard-constraint
//! construction, hyperparameter joint-setup and log-lambda seeding, penalty
//! transformations, per-z surface combination, the pooled survival baseline
//! solve, and the joint-design preflight diagnostic.

use super::*;



/// Overwrite the timewiggle tail columns of the dense time-channel
/// primary-Jacobian slots with their analytic value at the β=0 pilot primary
/// state, in place.
///
/// When `timewiggle(...)` is active the workflow disables the base time basis
/// and appends the wiggle coefficient slots as **zero placeholder** columns
/// (the design width is a coefficient-layout convention, not the Jacobian).
/// Feeding those zeros into the identifiability compiler makes the whole time
/// block look structurally zero ("block 0 fully aliased") — a bad-Jacobian
/// artifact, not a real alias. The true primary-channel derivative of the time
/// channels w.r.t. wiggle coefficient `j`, at the linearisation point β=0, is
/// the monotone-wiggle basis value at the pilot coordinate:
///
/// ```text
///   ∂q0  / ∂β_tw[j] = B_j(h0)          h0    = q0_pilot  = offset_entry
///   ∂q1  / ∂β_tw[j] = B_j(h1)          h1    = q1_pilot  = offset_exit + marginal_offset
///   ∂qd1 / ∂β_tw[j] = B'_j(h1)·d_raw   d_raw = qd1_pilot = derivative_offset_exit
/// ```
///
/// Base columns (`j < p_base`) reduce to the raw design at β=0 (dq/dh = 1,
/// d²q/dh² = 0) and are left untouched. The logslope-curvature factor `c_i` is
/// **not** applied here; it enters the Fisher Gram through
/// [`SurvivalRowHessian`], matching the raw-design convention for base columns.
pub(crate) fn overwrite_timewiggle_time_slots_at_pilot(
    dq0: &mut Array2<f64>,
    dq1: &mut Array2<f64>,
    dqd1: &mut Array2<f64>,
    timewiggle: &TimeWiggleBlockInput,
    h0: &Array1<f64>,
    h1: &Array1<f64>,
    d_raw: &Array1<f64>,
) -> Result<(), String> {
    let p_tw = timewiggle.ncols;
    if p_tw == 0 {
        return Ok(());
    }
    let p_time = dq0.ncols();
    let p_base = p_time.saturating_sub(p_tw);
    let n = dq0.nrows();
    let knots = &timewiggle.knots;
    let degree = timewiggle.degree;
    let b0 = monotone_wiggle_basis_with_derivative_order(h0.view(), knots, degree, 0)?;
    let b1 = monotone_wiggle_basis_with_derivative_order(h1.view(), knots, degree, 0)?;
    let b1d = monotone_wiggle_basis_with_derivative_order(h1.view(), knots, degree, 1)?;
    if b0.ncols() != p_tw || b1.ncols() != p_tw || b1d.ncols() != p_tw {
        return Err(format!(
            "overwrite_timewiggle_time_slots_at_pilot: basis width B/B/B'={}/{}/{} != p_tw={p_tw}",
            b0.ncols(),
            b1.ncols(),
            b1d.ncols(),
        ));
    }
    for i in 0..n {
        for j in 0..p_tw {
            let col = p_base + j;
            dq0[[i, col]] = b0[[i, j]];
            dq1[[i, col]] = b1[[i, j]];
            dqd1[[i, col]] = b1d[[i, j]] * d_raw[i];
        }
    }
    Ok(())
}


// ── Building block specs ──────────────────────────────────────────────

pub(crate) fn build_time_blockspec(
    time_block: &TimeBlockInput,
    design_exit: &DesignMatrix,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    // Share the three dense design matrices with the multi-output Jacobian
    // via `Arc` — `try_to_dense_arc` is zero-copy for materialized designs,
    // so the callback retains no duplicate `n × p` storage. Falls back to no
    // callback if densification fails.
    let jac_cb: Option<Arc<dyn crate::custom_family::BlockEffectiveJacobian>> = (|| {
        let d_entry = time_block
            .design_entry
            .try_to_dense_arc("build_time_blockspec::entry")
            .ok()?;
        let d_exit = design_exit
            .try_to_dense_arc("build_time_blockspec::exit")
            .ok()?;
        let d_deriv = time_block
            .design_derivative_exit
            .try_to_dense_arc("build_time_blockspec::deriv")
            .ok()?;
        if d_entry.dim() != d_exit.dim() || d_entry.dim() != d_deriv.dim() {
            return None;
        }
        Some(Arc::new(TimeBlockJacobian::new(d_entry, d_exit, d_deriv))
            as Arc<dyn crate::custom_family::BlockEffectiveJacobian>)
    })();

    ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: design_exit.clone(),
        offset: Array1::zeros(design_exit.nrows()),
        penalties: time_block
            .penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: time_block.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
        gauge_priority: 200,
        jacobian_callback: jac_cb,
        stacked_design: None,
        stacked_offset: None,
    }
}


pub(crate) fn build_logslope_blockspec(
    design: &TermCollectionDesign,
    baseline: f64,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
    z_scaling: Arc<[f64]>,
    probit_scale: f64,
) -> ParameterBlockSpec {
    let z_vec = z_scaling.to_vec();
    let jac_cb: Option<Arc<dyn crate::custom_family::BlockEffectiveJacobian>> = design
        .design
        .try_to_dense_arc("build_logslope_blockspec")
        .ok()
        .map(|d| {
            Arc::new(LogslopeBlockJacobian::new(d, z_vec, probit_scale))
                as Arc<dyn crate::custom_family::BlockEffectiveJacobian>
        });

    ParameterBlockSpec {
        name: "logslope_surface".to_string(),
        design: design.design.clone(),
        offset: offset + baseline,
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
        gauge_priority: 120,
        jacobian_callback: jac_cb,
        stacked_design: None,
        stacked_offset: None,
    }
}


pub(crate) fn build_marginal_blockspec(
    design: &TermCollectionDesign,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    let jac_cb: Option<Arc<dyn crate::custom_family::BlockEffectiveJacobian>> = design
        .design
        .try_to_dense_arc("build_marginal_blockspec")
        .ok()
        .map(|d| {
            Arc::new(MarginalBlockJacobian::new(d))
                as Arc<dyn crate::custom_family::BlockEffectiveJacobian>
        });

    ParameterBlockSpec {
        name: "marginal_surface".to_string(),
        design: design.design.clone(),
        offset: offset.clone(),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
        gauge_priority: 150,
        jacobian_callback: jac_cb,
        stacked_design: None,
        stacked_offset: None,
    }
}


pub(crate) fn inner_fit(
    family: &SurvivalMarginalSlopeFamily,
    blocks: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    fit_custom_family(family, blocks, options).map_err(|e| e.to_string())
}


/// Marginal-slope guard policy: the guard is required to be strictly positive
/// (`q'(t) ≥ guard > 0`), because the row-wise representation here is the *only*
/// place the monotonicity barrier lives — a zero guard would silently collapse
/// it. Coefficient-free row feasibility uses the family's epsilon-scaled slack
/// (`survival_derivative_guard_tolerance`).
pub(crate) const MARGINAL_SLOPE_GUARD_POLICY: GuardConstraintPolicy = GuardConstraintPolicy {
    guard_policy: GuardPolicy::Positive,
    feasibility: FeasibilityTolerance::EpsilonScaled,
};


pub(crate) fn time_derivative_guard_constraints(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    derivative_guard: f64,
) -> Result<Option<LinearInequalityConstraints>, String> {
    build_time_derivative_guard_constraints(
        design_derivative_exit,
        derivative_offset_exit,
        derivative_guard,
        MARGINAL_SLOPE_GUARD_POLICY,
    )
    .map_err(map_guard_constraint_failure)
}


/// Render a shared guard-constraint failure into the marginal-slope error
/// vocabulary, preserving the family's historical wording.
pub(crate) fn map_guard_constraint_failure(failure: GuardConstraintFailure) -> String {
    match failure {
        GuardConstraintFailure::RowOffsetMismatch { rows, offsets } => {
            SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope derivative guard constraints require matching rows/offsets: rows={rows}, offsets={offsets}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::GuardOutOfRange { guard, range } => {
            SurvivalMarginalSlopeError::InvalidInput {
                reason: format!(
                    "survival marginal-slope derivative guard must be finite and {range}, got {guard}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::NonFiniteOffset { row, offset } => {
            SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope derivative guard constraints require finite derivative offsets; found offset[{row}]={offset}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::NonFiniteDesign { row, col } => {
            SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope derivative guard constraints require finite derivative design entries; found row {row}, column {col}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::InfeasibleRow {
            row,
            offset,
            guard,
            no_time_coefficients,
        } => {
            let reason = if no_time_coefficients {
                format!(
                    "survival marginal-slope derivative guard is infeasible at row {row}: offset={offset:.3e} < guard={guard:.3e} with no time coefficients"
                )
            } else {
                format!(
                    "survival marginal-slope derivative guard is infeasible at row {row}: zero derivative design row with offset={offset:.3e} < guard={guard:.3e}"
                )
            };
            SurvivalMarginalSlopeError::MonotonicityViolation { reason }.into()
        }
    }
}


pub(crate) fn append_timewiggle_tail_nonnegative_constraints(
    base: Option<LinearInequalityConstraints>,
    p_total: usize,
    time_wiggle_ncols: usize,
) -> Result<Option<LinearInequalityConstraints>, String> {
    let p_wiggle = time_wiggle_ncols.min(p_total);
    if p_wiggle == 0 {
        return Ok(base);
    }
    if let Some(base_constraints) = base.as_ref() {
        if base_constraints.a.ncols() != p_total {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope time constraint width mismatch: constraints={}, time block={p_total}",
                    base_constraints.a.ncols()
                ),
            }
            .into());
        }
        if base_constraints.a.nrows() != base_constraints.b.len() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope time constraint row mismatch: A rows={}, b len={}",
                    base_constraints.a.nrows(),
                    base_constraints.b.len()
                ),
            }
            .into());
        }
    }

    let base_rows = base.as_ref().map_or(0, |constraints| constraints.a.nrows());
    let rows = base_rows + p_wiggle;
    let mut a = Array2::<f64>::zeros((rows, p_total));
    let mut b = Array1::<f64>::zeros(rows);

    if let Some(base_constraints) = base {
        a.slice_mut(s![..base_rows, ..]).assign(&base_constraints.a);
        b.slice_mut(s![..base_rows]).assign(&base_constraints.b);
    }

    let tail_start = p_total - p_wiggle;
    for (row_offset, col) in (tail_start..p_total).enumerate() {
        a[[base_rows + row_offset, col]] = 1.0;
    }
    Ok(Some(LinearInequalityConstraints { a, b }))
}


pub(crate) fn mean_abs(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for v in values {
        sum += v.abs();
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}


pub(crate) fn block_log_lambda_seeds<'a, I>(design: &DesignMatrix, penalty_locals: I) -> Vec<f64>
where
    I: IntoIterator<Item = &'a Array2<f64>>,
{
    let unit_weights = Array1::<f64>::ones(design.nrows());
    let likelihood_scale = match design.diag_gram(&unit_weights) {
        Ok(d) => mean_abs(d.iter().copied()).max(1.0e-8),
        Err(_) => 1.0,
    };
    penalty_locals
        .into_iter()
        .map(|s| {
            let penalty_scale = mean_abs(s.diag().iter().copied()).max(1.0e-8);
            (likelihood_scale / penalty_scale).ln().clamp(-12.0, 12.0)
        })
        .collect()
}


pub(crate) fn joint_setup(
    data: ArrayView2<'_, f64>,
    time_penalties: usize,
    marginalspec: &TermCollectionSpec,
    marginal_penalties: usize,
    logslopespec: &TermCollectionSpec,
    logslope_penalties: usize,
    core_rho0_seed: &[f64],
    extra_rho0: &[f64],
    pinned_rho_slots: &[(usize, f64)],
    initial_sigma: Option<f64>,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let marginal_terms = spatial_length_scale_term_indices(marginalspec);
    let logslope_terms = spatial_length_scale_term_indices(logslopespec);
    let core_len = time_penalties + marginal_penalties + logslope_penalties;
    let rho_dim = core_len + extra_rho0.len();
    let mut rho0vec = Array1::<f64>::zeros(rho_dim);
    assert_eq!(
        core_rho0_seed.len(),
        core_len,
        "core_rho0_seed length must equal time+marginal+logslope penalty count"
    );
    for (idx, value) in core_rho0_seed.iter().copied().enumerate().take(core_len) {
        rho0vec[idx] = value;
    }
    if !extra_rho0.is_empty() {
        let start = core_len;
        for (idx, value) in extra_rho0.iter().copied().enumerate() {
            rho0vec[start + idx] = value;
        }
    }
    let mut rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let mut rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    // Pin fixed-ridge penalty slots (e.g. the #461 influence absorber) to a
    // degenerate box so the outer REML optimizer can never move their log-λ:
    // the absorber ridge is a fixed training-time leakage absorber, not a
    // smooth/learned surface. Seed rho0 at the pinned value too so the start
    // point is feasible.
    for &(slot, value) in pinned_rho_slots {
        assert!(
            slot < rho_dim,
            "pinned rho slot {slot} out of range (rho_dim={rho_dim})"
        );
        rho0vec[slot] = value;
        rho_lower[slot] = value;
        rho_upper[slot] = value;
    }
    // Time block has no spatial length scales (pure B-spline on time)
    let empty_kappa = SpatialLogKappaCoords::new_with_dims(Array1::zeros(0), vec![]);
    let marginal_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        marginalspec,
        &marginal_terms,
        kappa_options,
    )
    .reseed_from_data(data, marginalspec, &marginal_terms, kappa_options);
    let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        logslopespec,
        &logslope_terms,
        kappa_options,
    )
    .reseed_from_data(data, logslopespec, &logslope_terms, kappa_options);
    let mut values = empty_kappa.as_array().to_vec();
    values.extend(marginal_kappa.as_array().iter());
    values.extend(logslope_kappa.as_array().iter());
    let marginal_dims = marginal_kappa.dims_per_term().to_vec();
    let logslope_dims = logslope_kappa.dims_per_term().to_vec();
    let mut dims = empty_kappa.dims_per_term().to_vec();
    dims.extend(marginal_dims.iter().copied());
    dims.extend(logslope_dims.iter().copied());
    let log_kappa0 =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(values.clone()), dims.clone());
    // Bounds: concatenate [empty | marginal data-aware | logslope data-aware]
    let marginal_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        marginalspec,
        &marginal_terms,
        &marginal_dims,
        kappa_options,
    );
    let logslope_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        logslopespec,
        &logslope_terms,
        &logslope_dims,
        kappa_options,
    );
    let mut lower_vals = Vec::with_capacity(dims.iter().sum());
    lower_vals.extend(marginal_lower.as_array().iter());
    lower_vals.extend(logslope_lower.as_array().iter());
    let log_kappa_lower =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(lower_vals), dims.clone());
    let marginal_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        marginalspec,
        &marginal_terms,
        &marginal_dims,
        kappa_options,
    );
    let logslope_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        logslopespec,
        &logslope_terms,
        &logslope_dims,
        kappa_options,
    );
    let mut upper_vals = Vec::with_capacity(dims.iter().sum());
    upper_vals.extend(marginal_upper.as_array().iter());
    upper_vals.extend(logslope_upper.as_array().iter());
    let log_kappa_upper = SpatialLogKappaCoords::new_with_dims(Array1::from_vec(upper_vals), dims);
    // Project seed onto bounds; spec.length_scale is a hint, not a constraint.
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);
    let setup = ExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    );
    if let Some(sigma) = initial_sigma.filter(|sigma| *sigma > 0.0) {
        setup.with_auxiliary(
            Array1::from_vec(vec![sigma.ln()]),
            Array1::from_vec(vec![-12.0]),
            Array1::from_vec(vec![6.0]),
        )
    } else {
        setup
    }
}





pub(crate) fn install_time_nullspace_shrinkage_penalty(
    time_block: &mut TimeBlockInput,
) -> Result<bool, String> {
    let p = time_block.design_exit.ncols();
    if p == 0 || time_block.penalties.is_empty() {
        return Ok(false);
    }
    if time_block.nullspace_dims.len() != time_block.penalties.len() {
        return Err(format!(
            "survival-marginal-slope time_block nullspace_dims length {} does not match penalties {}",
            time_block.nullspace_dims.len(),
            time_block.penalties.len(),
        ));
    }

    let mut aggregate = Array2::<f64>::zeros((p, p));
    for (idx, penalty) in time_block.penalties.iter().enumerate() {
        if penalty.nrows() != p || penalty.ncols() != p {
            return Err(format!(
                "survival-marginal-slope time_block penalty {idx} must be {p}x{p}, got {}x{}",
                penalty.nrows(),
                penalty.ncols(),
            ));
        }
        let scale = penalty
            .iter()
            .try_fold(0.0_f64, |acc, &value| {
                value.is_finite().then_some(acc.max(value.abs()))
            })
            .ok_or_else(|| {
                format!(
                    "survival-marginal-slope time_block penalty {idx} contains non-finite values"
                )
            })?;
        if scale > 0.0 {
            ndarray::Zip::from(&mut aggregate)
                .and(penalty)
                .for_each(|agg, &value| *agg += value / scale);
        }
    }

    let Some(shrinkage) = crate::terms::basis::build_nullspace_shrinkage_penalty(&aggregate)
        .map_err(|err| format!("survival-marginal-slope time_block nullspace shrinkage: {err}"))?
    else {
        return Ok(false);
    };
    if shrinkage.sym_penalty.nrows() != p || shrinkage.sym_penalty.ncols() != p {
        return Err(format!(
            "survival-marginal-slope time_block nullspace shrinkage penalty must be {p}x{p}, got {}x{}",
            shrinkage.sym_penalty.nrows(),
            shrinkage.sym_penalty.ncols(),
        ));
    }
    time_block.penalties.push(shrinkage.sym_penalty);
    time_block.nullspace_dims.push(0);
    log::info!(
        "[survival-marginal-slope] added time_block nullspace shrinkage penalty (p={p}, penalties={})",
        time_block.penalties.len(),
    );
    Ok(true)
}


pub(crate) fn concatenate_term_specs(specs: &[TermCollectionSpec]) -> TermCollectionSpec {
    let mut out = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: Vec::new(),
    };
    for spec in specs {
        out.linear_terms.extend(spec.linear_terms.clone());
        out.random_effect_terms
            .extend(spec.random_effect_terms.clone());
        out.smooth_terms.extend(spec.smooth_terms.clone());
    }
    out
}


pub(crate) fn shift_penalty(mut penalty: BlockwisePenalty, offset: usize) -> BlockwisePenalty {
    penalty.col_range = (penalty.col_range.start + offset)..(penalty.col_range.end + offset);
    penalty
}


pub(crate) fn combine_logslope_surface_designs(
    mut designs: Vec<TermCollectionDesign>,
    specs: &[TermCollectionSpec],
) -> Result<
    (
        TermCollectionDesign,
        TermCollectionSpec,
        Vec<std::ops::Range<usize>>,
    ),
    String,
> {
    if designs.is_empty() {
        return Err(
            "survival marginal-slope requires at least one logslope surface design".to_string(),
        );
    }
    if designs.len() == 1 {
        let design = designs.remove(0);
        let range = 0..design.design.ncols();
        let spec = specs
            .first()
            .cloned()
            .ok_or_else(|| "missing logslope surface spec".to_string())?;
        return Ok((design, spec, vec![range]));
    }
    if designs.iter().any(|design| {
        design.linear_constraints.is_some() || design.coefficient_lower_bounds.is_some()
    }) {
        return Err(
            "per-z logslope surface concatenation does not support coefficient bounds or linear constraints"
                .to_string(),
        );
    }

    let mut ranges = Vec::with_capacity(designs.len());
    let mut offset = 0usize;
    let mut blocks = Vec::with_capacity(designs.len());
    let mut penalties = Vec::new();
    let mut nullspace_dims = Vec::new();
    let mut penaltyinfo = Vec::new();
    let mut dropped_penaltyinfo = Vec::new();
    let mut linear_ranges = Vec::new();
    let mut random_effect_ranges = Vec::new();
    let mut random_effect_levels = Vec::new();
    let mut combined = designs[0].clone();
    combined.smooth.term_designs.clear();
    combined.smooth.penalties.clear();
    combined.smooth.nullspace_dims.clear();
    combined.smooth.penaltyinfo.clear();
    combined.smooth.dropped_penaltyinfo.clear();
    combined.smooth.terms.clear();
    combined.smooth.coefficient_lower_bounds = None;
    combined.smooth.linear_constraints = None;

    for (surface_idx, design) in designs.into_iter().enumerate() {
        let width = design.design.ncols();
        ranges.push(offset..offset + width);
        blocks.push(design.design.clone());
        for (local_penalty_idx, penalty) in design.penalties.iter().cloned().enumerate() {
            let global_index = penalties.len();
            penalties.push(shift_penalty(penalty, offset));
            if let Some(info) = design.penaltyinfo.get(local_penalty_idx) {
                let mut info = info.clone();
                info.global_index = global_index;
                if let Some(termname) = info.termname.as_mut() {
                    *termname = format!("logslope[z{surface_idx}]::{termname}");
                }
                penaltyinfo.push(info);
            }
        }
        nullspace_dims.extend(design.nullspace_dims.iter().copied());
        dropped_penaltyinfo.extend(design.dropped_penaltyinfo.iter().cloned());
        linear_ranges.extend(design.linear_ranges.iter().cloned().map(|(name, range)| {
            (
                format!("logslope[z{surface_idx}]::{name}"),
                (range.start + offset)..(range.end + offset),
            )
        }));
        random_effect_ranges.extend(design.random_effect_ranges.iter().cloned().map(
            |(name, range)| {
                (
                    format!("logslope[z{surface_idx}]::{name}"),
                    (range.start + offset)..(range.end + offset),
                )
            },
        ));
        random_effect_levels.extend(design.random_effect_levels.iter().cloned());
        offset += width;
    }
    combined.design = DesignMatrix::hstack(blocks)
        .map_err(|e| format!("survival marginal-slope logslope hstack: {e}"))?;
    combined.penalties = penalties;
    combined.nullspace_dims = nullspace_dims;
    combined.penaltyinfo = penaltyinfo;
    combined.dropped_penaltyinfo = dropped_penaltyinfo;
    combined.coefficient_lower_bounds = None;
    combined.linear_constraints = None;
    combined.intercept_range = 0..0;
    combined.linear_ranges = linear_ranges;
    combined.random_effect_ranges = random_effect_ranges;
    combined.random_effect_levels = random_effect_levels;
    Ok((combined, concatenate_term_specs(specs), ranges))
}


/// Compute a baseline slope from the actual survival marginal-slope likelihood,
/// using the baseline offsets alone as a time-only pilot q(t).
///
/// This is a safeguarded 1D Newton solve on the true row objective. It does not
/// use a coarse fixed grid scan.
pub(crate) fn pooled_survival_baseline(
    event: &Array1<f64>,
    weights: &Array1<f64>,
    z: &Array1<f64>,
    q0: &Array1<f64>,
    q1: &Array1<f64>,
    qd1: &Array1<f64>,
    probit_scale: f64,
) -> f64 {
    let n = event.len();
    if n == 0 {
        return 0.0;
    }
    let objective_grad_hess = |slope: f64| -> Option<(f64, f64, f64)> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let triples: Option<Vec<(f64, f64, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (row_obj, row_grad, row_hess) = row_primary_closed_form(
                    q0[i],
                    q1[i],
                    qd1[i],
                    slope,
                    z[i],
                    weights[i],
                    event[i],
                    0.0,
                    probit_scale,
                )
                .ok()?;
                Some((row_obj, row_grad[3], row_hess[3][3]))
            })
            .collect();
        let triples = triples?;
        Some(
            triples
                .into_iter()
                .fold((0.0_f64, 0.0_f64, 0.0_f64), |(o, g, h), (oi, gi, hi)| {
                    (o + oi, g + gi, h + hi)
                }),
        )
    };

    let Some(state0) = objective_grad_hess(0.0) else {
        return 0.0;
    };
    if !state0.0.is_finite() {
        return 0.0;
    }
    if state0.1.abs() < 1e-8 {
        return 0.0;
    }

    let mut best_slope = 0.0;
    let mut best = state0;

    let mut bracket_lo = if state0.1 <= 0.0 {
        Some((0.0, state0))
    } else {
        None
    };
    let mut bracket_hi = if state0.1 >= 0.0 {
        Some((0.0, state0))
    } else {
        None
    };
    let mut step = 0.5f64;
    for _ in 0..48 {
        for &candidate in &[-step, step] {
            if let Some(state) = objective_grad_hess(candidate) {
                if state.0 < best.0 {
                    best_slope = candidate;
                    best = state;
                }
                if state.1 <= 0.0 {
                    bracket_lo = Some((candidate, state));
                }
                if state.1 >= 0.0 {
                    bracket_hi = Some((candidate, state));
                }
                if let (Some((lo, lo_state)), Some((hi, hi_state))) = (bracket_lo, bracket_hi)
                    && lo < hi
                    && lo_state.1 <= 0.0
                    && hi_state.1 >= 0.0
                {
                    let mut slope = best_slope.clamp(lo, hi);
                    let mut state = if (slope - lo).abs() < f64::EPSILON {
                        lo_state
                    } else if (slope - hi).abs() < f64::EPSILON {
                        hi_state
                    } else {
                        match objective_grad_hess(slope) {
                            Some(s) => s,
                            None => {
                                slope = 0.5 * (lo + hi);
                                objective_grad_hess(slope).unwrap_or(best)
                            }
                        }
                    };

                    let mut bracket_lo = (lo, lo_state);
                    let mut bracket_hi = (hi, hi_state);
                    for _ in 0..60 {
                        if state.1.abs() < 1e-8 || (bracket_hi.0 - bracket_lo.0).abs() < 1e-8 {
                            break;
                        }
                        let mut candidate = 0.5 * (bracket_lo.0 + bracket_hi.0);
                        if state.2.is_finite() && state.2 > 0.0 {
                            let newton = slope - state.1 / state.2;
                            if newton > bracket_lo.0 && newton < bracket_hi.0 {
                                candidate = newton;
                            }
                        }
                        let Some(candidate_state) = objective_grad_hess(candidate) else {
                            candidate = 0.5 * (bracket_lo.0 + bracket_hi.0);
                            let Some(mid_state) = objective_grad_hess(candidate) else {
                                break;
                            };
                            if mid_state.0 < best.0 {
                                best_slope = candidate;
                                best = mid_state;
                            }
                            if mid_state.1 <= 0.0 {
                                bracket_lo = (candidate, mid_state);
                            } else {
                                bracket_hi = (candidate, mid_state);
                            }
                            slope = candidate;
                            state = mid_state;
                            continue;
                        };
                        if candidate_state.0 < best.0 {
                            best_slope = candidate;
                            best = candidate_state;
                        }
                        if candidate_state.1 <= 0.0 {
                            bracket_lo = (candidate, candidate_state);
                        } else {
                            bracket_hi = (candidate, candidate_state);
                        }
                        slope = candidate;
                        state = candidate_state;
                    }
                    return if best.0.is_finite() { best_slope } else { 0.0 };
                }
            }
        }
        step *= 2.0;
    }
    if best.0.is_finite() { best_slope } else { 0.0 }
}


// ── Public fitting function ───────────────────────────────────────────

/// Whether the optional score-warp / link-deviation flex blocks participate in
/// a family build. The rigid warm-start pilot must construct its family and
/// blocks with `OffForRigidPilot` so the cold-start coefficient solve cannot
/// silently activate the survival flex exact-Joint-Newton path.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum FlexActivation {
    OffForRigidPilot,
    On,
}


/// Which coordinate system the `marginal`/`logslope` designs handed to
/// `build_blocks` live in — and therefore whether the side-bound V+M-exact
/// pulled-back penalties (`*_penalties_vm`) apply.
///
/// The V+M-exact cutover compiles the marginal/logslope designs into a reduced
/// (column-dropped, reparameterised) coordinate system and pulls each block's
/// penalty back through that block's own `V_b` to a matching compiled-width
/// `Vᵀ S V`. Those pulled-back penalties are valid **only** against the
/// compiled designs they were derived from. Two distinct callers reach
/// `build_blocks` with structurally different designs:
///
/// * [`BlockDesignCoords::PostCutover`] — the construction-site calls (rigid
///   warm-start pilot, initial derivative probe) pass the post-cutover designs
///   the `*_penalties_vm` were pulled back through. The compiled penalties are
///   authoritative here and must replace the raw-width penalties that
///   `build_*_blockspec` derives from `TermCollectionDesign.penalties` (which
///   the cutover intentionally leaves at raw width for predict-time consumers).
/// * [`BlockDesignCoords::RematerializedRaw`] — the outer spatial-length-scale
///   optimizer re-materialises *raw*-width marginal/logslope designs from the
///   boot `TermCollectionSpec`s on every κ probe and routes them here. The raw
///   design-derived penalties are authoritative; the compiled `*_penalties_vm`
///   are at the wrong width/parametrisation and must NOT be installed (doing so
///   is the #788 `block i penalty 0 must be NxN, got KxK` shape mismatch — and,
///   in the no-column-drop-but-`V≠I` case where widths happen to coincide, a
///   silent installation of a `Vᵀ S V` penalty onto a raw design).
///
/// Distinguishing by an explicit, named coordinate tag — rather than guessing
/// from a `penalty.shape() == design.shape()` width coincidence — makes the
/// provenance an auditable property of each call site and lets the compiled
/// path assert width agreement loudly instead of relying on it holding.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum BlockDesignCoords {
    PostCutover,
    RematerializedRaw,
}


/// One block in the joint training-row design layout consumed by
/// [`joint_training_design_preflight`]. Holds a dense `(n x p_block)`
/// slice and the block tag used in the failure diagnostic.
pub(crate) struct JointPreflightSegment {
    pub block: JointPreflightBlock,
    pub columns: DesignMatrix,
}


/// W-metric joint training-row design preflight.
///
/// Stacks the per-block training-row designs horizontally into a single
/// `n x p_joint` matrix `J`, pre-scales by `sqrt(W)`, and runs a thin-SVD
/// via [`crate::faer_ndarray::FaerSvd`]. Always returns `Ok(())`; if any
/// singular value falls at or below the numerical-rank tolerance
/// `sigma_max * max(n, p_joint) * 16 * f64::EPSILON`, emits an info-level
/// log line localising each alias to its dominant `(block, local_col, weight)`
/// triple. The canonical-gauge pipeline in
/// `custom_family.rs::canonicalize_for_identifiability` attributes the alias
/// drops via `gauge_priority` and reduces the specs before any inner Newton
/// fires, so this preflight is diagnostic-only.
pub(crate) fn joint_training_design_preflight(
    segments: &[JointPreflightSegment],
    weights: &Array1<f64>,
) -> Result<(), SurvivalMarginalSlopeError> {
    use crate::faer_ndarray::{FaerEigh, fast_xt_diag_y};

    if segments.is_empty() {
        return Ok(());
    }
    let n = weights.len();
    let mut p_joint = 0usize;
    let mut block_ranges: Vec<(JointPreflightBlock, usize, usize)> =
        Vec::with_capacity(segments.len());
    for seg in segments {
        if seg.columns.nrows() != n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "joint preflight: block {} has {} rows, weights have {}",
                    seg.block,
                    seg.columns.nrows(),
                    n,
                ),
            });
        }
        let start = p_joint;
        let end = p_joint + seg.columns.ncols();
        block_ranges.push((seg.block, start, end));
        p_joint = end;
    }
    if p_joint == 0 {
        return Ok(());
    }

    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: format!("joint preflight: weights[{i}] = {w} is not finite/non-negative",),
            });
        }
    }

    // W-metric joint Gram `G = Jᵀ diag(w) J`, accumulated over fixed-height
    // row chunks of the operator-backed segments. The previous implementation
    // stacked a dense `(n, p_joint)` sqrt(W)-scaled copy of the joint design
    // and ran a thin-SVD over it — at biobank scale that transient (plus the
    // SVD's own workspace) was a multi-GiB contributor to the #979 survival
    // construction-phase OOM, and the budget guard that protected against it
    // silently skipped the diagnostic at exactly the scale where it matters.
    // The Gram route needs `O(chunk × p_joint + p_joint²)` memory at any n,
    // so the preflight always runs. Singular values come back as √eigenvalue;
    // squaring the spectrum coarsens the detectable near-alias floor from
    // ~`dim·ε·σ_max` to ~`sqrt(dim·ε)·σ_max`, which is immaterial here:
    // structural aliases (σ exactly 0) are detected identically, and this
    // preflight is observability-only — the canonical-gauge RRQR audit
    // downstream remains the fail-closed authority on borderline cases.
    const PREFLIGHT_GRAM_ROW_CHUNK: usize = 4096;
    let mut gram = Array2::<f64>::zeros((p_joint, p_joint));
    let mut chunk_start = 0usize;
    while chunk_start < n {
        let chunk_end = (chunk_start + PREFLIGHT_GRAM_ROW_CHUNK).min(n);
        let chunks: Vec<Array2<f64>> = segments
            .iter()
            .map(|seg| {
                seg.columns
                    .try_row_chunk(chunk_start..chunk_end)
                    .map_err(|e| SurvivalMarginalSlopeError::NumericalFailure {
                        reason: format!(
                            "joint preflight: block {} rows {chunk_start}..{chunk_end}: {e}",
                            seg.block
                        ),
                    })
            })
            .collect::<Result<_, _>>()?;
        let w_chunk = weights.slice(s![chunk_start..chunk_end]);
        for (s, (_, s0, s1)) in block_ranges.iter().enumerate() {
            for (t, (_, t0, t1)) in block_ranges.iter().enumerate().skip(s) {
                let g_st = fast_xt_diag_y(&chunks[s], &w_chunk, &chunks[t]);
                let mut dst = gram.slice_mut(s![*s0..*s1, *t0..*t1]);
                dst += &g_st;
                if t > s {
                    let mut dst_t = gram.slice_mut(s![*t0..*t1, *s0..*s1]);
                    dst_t += &g_st.t();
                }
            }
        }
        chunk_start = chunk_end;
    }

    let (eigvals, eigvecs) =
        gram.eigh(faer::Side::Lower)
            .map_err(|e| SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!("joint preflight: W-metric Gram eigh failed: {e:?}"),
            })?;
    // σ_i = sqrt(max(λ_i, 0)); roundoff can push numerically-zero Gram
    // eigenvalues slightly negative.
    let sigma: Vec<f64> = eigvals.iter().map(|&l| l.max(0.0).sqrt()).collect();
    let sigma_max = sigma.iter().copied().fold(0.0_f64, f64::max);
    let rank_dim = n.max(p_joint) as f64;
    // Gram-spectrum near-alias floor (see the block comment above): aliases
    // are directions whose singular value is at or below
    // `σ_max · sqrt(dim · 16ε)`.
    let rank_tol = sigma_max * (rank_dim * 16.0 * f64::EPSILON).sqrt();

    let alias_idx: Vec<usize> = sigma
        .iter()
        .enumerate()
        .filter_map(|(idx, &s)| (s <= rank_tol).then_some(idx))
        .collect();
    let rank = sigma.len() - alias_idx.len();

    if alias_idx.is_empty() {
        let sigma_min = sigma.iter().copied().fold(f64::INFINITY, f64::min);
        let condition = if sigma_min > 0.0 {
            sigma_max / sigma_min
        } else {
            f64::INFINITY
        };
        log::info!(
            "[survival-marginal-slope/preflight] joint design full-rank: n={n} p_joint={p_joint} \
             sigma_min={sigma_min:.3e} sigma_max={sigma_max:.3e} kappa={condition:.3e} tol={rank_tol:.3e}",
        );
        return Ok(());
    }

    let structural_alias = p_joint.saturating_sub(n.min(p_joint));

    let mut columns: Vec<(JointPreflightBlock, usize, f64)> = Vec::new();
    for &idx in alias_idx.iter() {
        // Eigenvector column `idx` of the Gram is the right singular vector
        // of the collapsing direction.
        let v_col = eigvecs.column(idx);
        let mut best_j = 0usize;
        let mut best_w = 0.0_f64;
        for j in 0..p_joint {
            let w = v_col[j].abs();
            if w > best_w {
                best_w = w;
                best_j = j;
            }
        }
        let (block, local_col) = block_ranges
            .iter()
            .find_map(|(b, start, end)| {
                (best_j >= *start && best_j < *end).then_some((*b, best_j - *start))
            })
            .ok_or_else(|| SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "joint preflight: alias column index {best_j} outside block ranges (p_joint={p_joint})",
                ),
            })?;
        columns.push((block, local_col, best_w));
    }

    let block_summary = block_ranges
        .iter()
        .map(|(b, start, end)| format!("{b}=[{start}..{end})"))
        .collect::<Vec<_>>()
        .join(", ");
    let dominant_summary = columns
        .iter()
        .map(|(b, c, w)| format!("{b}[{c}] (|v|={w:.3e})"))
        .collect::<Vec<_>>()
        .join("; ");
    // Informational: rank deficiency at this point is handled gracefully by
    // the canonical-gauge pipeline downstream (`canonicalize_for_identifiability`
    // in `custom_family.rs`). That pipeline runs a joint RRQR audit with
    // `gauge_priority` attribution and converts attributed alias drops into
    // per-block selection matrices `T_i`, then solves on reduced specs and
    // lifts coefficients back via `β_raw = T_i θ`. Aborting here would defeat
    // the canonical reduction.
    log::info!(
        "[survival-marginal-slope/preflight] joint design W-metric rank-deficient: \
         rank={rank}/{p_joint} (sigma_max={sigma_max:.3e}, rank_tol={rank_tol:.3e}, n={n}, \
         structural_alias={structural_alias}, alias_directions={alias_count}). \
         Block layout: {block_summary}. Dominant block x column per alias: {dominant_summary}. \
         Canonical-gauge pipeline (gauge_priority: time=200, marginal=150, logslope=120, \
         score_warp=80, link_dev=60) will attribute the alias to lower-priority blocks and \
         proceed with reduced specs.",
        alias_count = alias_idx.len(),
    );
    Ok(())
}

