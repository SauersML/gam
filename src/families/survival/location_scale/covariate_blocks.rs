use super::*;

pub(crate) fn validate_cov_block(
    name: &str,
    n: usize,
    b: &ParameterBlockInput,
) -> Result<(), SurvivalLocationScaleError> {
    if b.design.nrows() != n {
        crate::bail_dim_sls!(
            "{name} design row mismatch: got {}, expected {n}",
            b.design.nrows()
        );
    }
    if b.offset.len() != n {
        crate::bail_dim_sls!(
            "{name} offset length mismatch: got {}, expected {n}",
            b.offset.len()
        );
    }
    let p = b.design.ncols();
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        crate::bail_dim_sls!(
            "{name} initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        );
    }
    let k = b.penalties.len();
    if let Some(rho0) = &b.initial_log_lambdas
        && rho0.len() != k
    {
        crate::bail_dim_sls!(
            "{name} initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        );
    }
    for (idx, s) in b.penalties.iter().enumerate() {
        match s {
            crate::model_types::PenaltySpec::Block {
                local, col_range, ..
            } => {
                if col_range.end > p
                    || local.nrows() != col_range.len()
                    || local.ncols() != col_range.len()
                {
                    crate::bail_dim_sls!(
                        "{name} penalty {idx} block shape mismatch: col_range={}..{}, local={}x{}, total_dim={p}",
                        col_range.start,
                        col_range.end,
                        local.nrows(),
                        local.ncols()
                    );
                }
            }
            crate::model_types::PenaltySpec::Dense(m)
            | crate::model_types::PenaltySpec::DenseWithMean { matrix: m, .. } => {
                let (r, c) = m.dim();
                if r != p || c != p {
                    crate::bail_dim_sls!("{name} penalty {idx} must be {p}x{p}, got {r}x{c}");
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_cov_block_kind(
    name: &str,
    n: usize,
    bk: &CovariateBlockKind,
) -> Result<(), SurvivalLocationScaleError> {
    match bk {
        CovariateBlockKind::Static(b) => validate_cov_block(name, n, b),
        CovariateBlockKind::TimeVarying(tv) => {
            if tv.design_covariates.nrows() != n {
                crate::bail_dim_sls!(
                    "{name} time-varying covariate design row mismatch: got {}, expected {n}",
                    tv.design_covariates.nrows()
                );
            }
            if tv.time_basis_entry.nrows() != n || tv.time_basis_exit.nrows() != n {
                crate::bail_dim_sls!(
                    "{name} time-varying time basis row mismatch: entry={}, exit={}, expected {n}",
                    tv.time_basis_entry.nrows(),
                    tv.time_basis_exit.nrows()
                );
            }
            if tv.time_basis_derivative_exit.nrows() != n {
                crate::bail_dim_sls!(
                    "{name} time-varying derivative basis row mismatch: got {}, expected {n}",
                    tv.time_basis_derivative_exit.nrows()
                );
            }
            if tv.offset.len() != n {
                crate::bail_dim_sls!(
                    "{name} time-varying offset length mismatch: got {}, expected {n}",
                    tv.offset.len()
                );
            }
            let p_cov = tv.design_covariates.ncols();
            let p_time = tv.time_basis_exit.ncols();
            if tv.time_basis_entry.ncols() != p_time {
                crate::bail_dim_sls!(
                    "{name} time-varying time basis column mismatch: entry={}, exit={}",
                    tv.time_basis_entry.ncols(),
                    p_time
                );
            }
            if tv.time_basis_derivative_exit.ncols() != p_time {
                crate::bail_dim_sls!(
                    "{name} time-varying derivative basis column mismatch: derivative={}, exit={}",
                    tv.time_basis_derivative_exit.ncols(),
                    p_time
                );
            }
            let p_tensor = p_cov * p_time;
            let k = tv.penalties.len();
            if let Some(beta0) = &tv.initial_beta
                && beta0.len() != p_tensor
            {
                crate::bail_dim_sls!(
                    "{name} time-varying initial_beta length mismatch: got {}, expected {p_tensor}",
                    beta0.len()
                );
            }
            if let Some(rho0) = &tv.initial_log_lambdas
                && rho0.len() != k
            {
                crate::bail_dim_sls!(
                    "{name} time-varying initial_log_lambdas length mismatch: got {}, expected {k}",
                    rho0.len()
                );
            }
            for (idx, s) in tv.penalties.iter().enumerate() {
                let (r, c) = s.shape();
                if r != p_tensor || c != p_tensor {
                    crate::bail_dim_sls!(
                        "{name} time-varying penalty {idx} must be {p_tensor}x{p_tensor}, got {r}x{c}"
                    );
                }
            }
            Ok(())
        }
    }
}

/// Build row-wise Kronecker product: each row of the result is
/// kron(cov_row[i,:], time_row[i,:]).
pub(crate) fn assert_rowwise_kronecker_dimensions(
    n: usize,
    p_resp: usize,
    p_cov: usize,
    context: &str,
) {
    assert!(
        p_resp > 0 && p_cov > 0,
        "{context} rowwise Kronecker dimensions must be non-empty: n={n}, p_resp={p_resp}, p_cov={p_cov}"
    );
}

pub(crate) fn rowwise_kronecker(
    cov_design: &DesignMatrix,
    time_basis: &Array2<f64>,
) -> DesignMatrix {
    let n = cov_design.nrows();
    let p_cov = cov_design.ncols();
    let p_time = time_basis.ncols();
    assert_rowwise_kronecker_dimensions(n, p_time, p_cov, "survival");
    let op = RowwiseKroneckerOperator::new(cov_design.clone(), shared_dense_arc(time_basis))
        .expect("rowwise kronecker design should have matched row counts");
    DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(op)))
}

pub(crate) fn design_block_from_matrix(design: DesignMatrix) -> DesignBlock {
    match design {
        DesignMatrix::Dense(matrix) => DesignBlock::Dense(matrix),
        other => DesignBlock::Dense(DenseDesignMatrix::from(Arc::new(other))),
    }
}

pub(crate) fn design_column_tail(
    design: &DesignMatrix,
    first_col: usize,
    label: &str,
) -> Result<DesignMatrix, String> {
    let p = design.ncols();
    if first_col > p {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!("{label}: first retained column {first_col} exceeds design width {p}"),
        }
        .into());
    }
    if first_col == 0 {
        return Ok(design.clone());
    }
    let n = design.nrows();
    let active_p = p - first_col;
    if active_p == 0 {
        return Ok(DesignMatrix::from(Array2::<f64>::zeros((n, 0))));
    }

    let chunk_rows = (ROW_CHUNK_BYTE_BUDGET / (p.max(1) * std::mem::size_of::<f64>()))
        .max(1)
        .min(n.max(1));
    let mut out = Array2::<f64>::zeros((n, active_p));
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| format!("{label}: failed to materialize design rows: {e}"))?;
        out.slice_mut(s![start..end, ..])
            .assign(&chunk.slice(s![.., first_col..]));
    }
    Ok(DesignMatrix::from(out))
}

pub(crate) fn drop_leading_initial_beta(
    beta: Option<Array1<f64>>,
    fixed_cols: usize,
    full_dim: usize,
    label: &str,
) -> Result<Option<Array1<f64>>, String> {
    let Some(beta) = beta else {
        return Ok(None);
    };
    if beta.len() != full_dim {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "{label}: initial_beta length mismatch before identifiability reduction: got {}, expected {full_dim}",
            beta.len()
        ) }.into());
    }
    Ok(Some(beta.slice(s![fixed_cols..]).to_owned()))
}

pub(crate) fn expand_leading_fixed_beta(
    beta_active: &Array1<f64>,
    fixed_cols: usize,
    full_dim: usize,
    label: &str,
) -> Result<Array1<f64>, String> {
    if fixed_cols > full_dim {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "{label}: fixed column count {fixed_cols} exceeds full width {full_dim}"
            ),
        }
        .into());
    }
    let active_dim = full_dim - fixed_cols;
    if beta_active.len() != active_dim {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "{label}: active beta length mismatch: got {}, expected {active_dim}",
                beta_active.len()
            ),
        }
        .into());
    }
    if fixed_cols == 0 {
        return Ok(beta_active.clone());
    }
    let mut beta_full = Array1::<f64>::zeros(full_dim);
    beta_full.slice_mut(s![fixed_cols..]).assign(beta_active);
    Ok(beta_full)
}

pub(crate) fn drop_leading_penalty_columns(
    penalties: &[PenaltyMatrix],
    nullspace_dims: &[usize],
    initial_log_lambdas: Array1<f64>,
    fixed_cols: usize,
    full_dim: usize,
    label: &str,
) -> Result<(Vec<PenaltyMatrix>, Vec<usize>, Array1<f64>), String> {
    if fixed_cols > full_dim {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "{label}: fixed column count {fixed_cols} exceeds full penalty width {full_dim}"
            ),
        }
        .into());
    }
    if initial_log_lambdas.len() != penalties.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "{label}: initial log-lambda length {} does not match {} penalties",
                initial_log_lambdas.len(),
                penalties.len()
            ),
        }
        .into());
    }
    if fixed_cols == 0 {
        return Ok((
            penalties.to_vec(),
            nullspace_dims.to_vec(),
            initial_log_lambdas,
        ));
    }

    let active_dim = full_dim - fixed_cols;
    if active_dim == 0 {
        return Ok((Vec::new(), Vec::new(), Array1::zeros(0)));
    }

    let structural_nullspace_available = nullspace_dims.len() == penalties.len();
    let mut structural_nullspace_exact = structural_nullspace_available;
    let mut retained_penalties = Vec::new();
    let mut retained_nullspace_dims = Vec::new();
    let mut retained_log_lambdas = Vec::new();

    for (idx, penalty) in penalties.iter().enumerate() {
        if penalty.dim() != full_dim {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{label}: penalty {idx} has dimension {}, expected {full_dim}",
                    penalty.dim()
                ),
            }
            .into());
        }

        let reduced = match penalty {
            PenaltyMatrix::Blockwise {
                local,
                col_range,
                total_dim,
            } => {
                if *total_dim != full_dim {
                    return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                        "{label}: blockwise penalty {idx} total_dim {total_dim} does not match {full_dim}"
                    ) }.into());
                }
                if col_range.end <= fixed_cols {
                    None
                } else {
                    let active_start = col_range.start.max(fixed_cols);
                    let active_end = col_range.end;
                    let local_start = active_start - col_range.start;
                    let local_end = active_end - col_range.start;
                    if local_start != 0 || local_end != local.nrows() {
                        structural_nullspace_exact = false;
                    }
                    Some(PenaltyMatrix::Blockwise {
                        local: local
                            .slice(s![local_start..local_end, local_start..local_end])
                            .to_owned(),
                        col_range: (active_start - fixed_cols)..(active_end - fixed_cols),
                        total_dim: active_dim,
                    })
                }
            }
            PenaltyMatrix::Dense(matrix) => {
                structural_nullspace_exact = false;
                Some(PenaltyMatrix::Dense(
                    matrix
                        .slice(s![fixed_cols..full_dim, fixed_cols..full_dim])
                        .to_owned(),
                ))
            }
            PenaltyMatrix::KroneckerFactored { .. } => {
                structural_nullspace_exact = false;
                let dense = penalty.to_dense();
                Some(PenaltyMatrix::Dense(
                    dense
                        .slice(s![fixed_cols..full_dim, fixed_cols..full_dim])
                        .to_owned(),
                ))
            }
            PenaltyMatrix::Labeled { label, inner } => {
                structural_nullspace_exact = false;
                let dense = inner.to_dense();
                Some(
                    PenaltyMatrix::Dense(
                        dense
                            .slice(s![fixed_cols..full_dim, fixed_cols..full_dim])
                            .to_owned(),
                    )
                    .with_precision_label(label.clone()),
                )
            }
            PenaltyMatrix::Fixed { log_lambda, inner } => {
                structural_nullspace_exact = false;
                let dense = inner.to_dense();
                Some(
                    PenaltyMatrix::Dense(
                        dense
                            .slice(s![fixed_cols..full_dim, fixed_cols..full_dim])
                            .to_owned(),
                    )
                    .with_fixed_log_lambda(*log_lambda),
                )
            }
        };

        if let Some(reduced) = reduced {
            retained_penalties.push(reduced);
            retained_log_lambdas.push(initial_log_lambdas[idx]);
            if structural_nullspace_available {
                retained_nullspace_dims.push(nullspace_dims[idx]);
            }
        }
    }

    if !structural_nullspace_exact {
        retained_nullspace_dims.clear();
    }

    Ok((
        retained_penalties,
        retained_nullspace_dims,
        Array1::from_vec(retained_log_lambdas),
    ))
}

/// Prepared covariate block data for the family struct.
pub(crate) struct PreparedCovBlock {
    /// Exit design (used as the solver's primary).
    pub(crate) design_exit: DesignMatrix,
    /// Entry design, only for time-varying blocks.
    pub(crate) design_entry: Option<DesignMatrix>,
    /// Exit-time derivative design, only for time-varying blocks.
    pub(crate) design_derivative_exit: Option<DesignMatrix>,
    /// Offset (same for both entry/exit since it comes from other terms).
    pub(crate) offset: Array1<f64>,
    pub(crate) penalties: Vec<PenaltyMatrix>,
    pub(crate) nullspace_dims: Vec<usize>,
    pub(crate) initial_log_lambdas: Option<Array1<f64>>,
    pub(crate) initial_beta: Option<Array1<f64>>,
}

pub(crate) fn prepare_cov_block_kind(
    bk: &CovariateBlockKind,
) -> Result<PreparedCovBlock, SurvivalLocationScaleError> {
    match bk {
        CovariateBlockKind::Static(b) => Ok(PreparedCovBlock {
            design_exit: b.design.clone(),
            design_entry: None,
            design_derivative_exit: None,
            offset: b.offset.clone(),
            penalties: {
                let p = b.design.ncols();
                b.penalties
                    .iter()
                    .map(|spec| match spec {
                        crate::model_types::PenaltySpec::Block {
                            local, col_range, ..
                        } => PenaltyMatrix::Blockwise {
                            local: local.clone(),
                            col_range: col_range.clone(),
                            total_dim: p,
                        },
                        crate::model_types::PenaltySpec::Dense(m)
                        | crate::model_types::PenaltySpec::DenseWithMean {
                            matrix: m, ..
                        } => PenaltyMatrix::Dense(m.clone()),
                    })
                    .collect()
            },
            nullspace_dims: b.nullspace_dims.clone(),
            initial_log_lambdas: b.initial_log_lambdas.clone(),
            initial_beta: b.initial_beta.clone(),
        }),
        CovariateBlockKind::TimeVarying(tv) => {
            let design_exit = rowwise_kronecker(&tv.design_covariates, &tv.time_basis_exit);
            let design_entry = rowwise_kronecker(&tv.design_covariates, &tv.time_basis_entry);
            let design_derivative_exit =
                rowwise_kronecker(&tv.design_covariates, &tv.time_basis_derivative_exit);
            Ok(PreparedCovBlock {
                design_exit,
                design_entry: Some(design_entry),
                design_derivative_exit: Some(design_derivative_exit),
                offset: tv.offset.clone(),
                penalties: tv.penalties.clone(),
                nullspace_dims: vec![],
                initial_log_lambdas: tv.initial_log_lambdas.clone(),
                initial_beta: tv.initial_beta.clone(),
            })
        }
    }
}

pub(crate) fn build_survival_covariate_block_from_design(
    cov_design: &TermCollectionDesign,
    template: &SurvivalCovariateTermBlockTemplate,
    offset: &Array1<f64>,
    initial_log_lambdas: Option<Array1<f64>>,
    initial_beta: Option<Array1<f64>>,
) -> Result<CovariateBlockKind, String> {
    match template {
        SurvivalCovariateTermBlockTemplate::Static => {
            Ok(CovariateBlockKind::Static(ParameterBlockInput {
                design: cov_design.design.clone(),
                offset: offset.clone(),
                penalties: cov_design
                    .penalties
                    .iter()
                    .map(crate::model_types::PenaltySpec::from_blockwise_ref)
                    .collect(),
                nullspace_dims: cov_design.nullspace_dims.clone(),
                initial_log_lambdas,
                initial_beta,
            }))
        }
        SurvivalCovariateTermBlockTemplate::TimeVarying {
            time_basis_entry,
            time_basis_exit,
            time_basis_derivative_exit,
            time_penalties,
        } => {
            let p_cov = cov_design.design.ncols();
            let p_time = time_basis_exit.ncols();
            let design_covariates = cov_design.design.clone();
            let i_cov = Array2::<f64>::eye(p_cov);
            let i_time = Array2::<f64>::eye(p_time);
            let cov_dense_for_kronecker: Vec<Array2<f64>> = cov_design
                .penalties
                .iter()
                .map(|bp| bp.to_global(p_cov))
                .collect();
            let mut penalties =
                Vec::with_capacity(cov_dense_for_kronecker.len() + time_penalties.len());
            for s_cov in &cov_dense_for_kronecker {
                penalties.push(PenaltyMatrix::KroneckerFactored {
                    left: s_cov.clone(),
                    right: i_time.clone(),
                });
            }
            for s_time in time_penalties {
                penalties.push(PenaltyMatrix::KroneckerFactored {
                    left: i_cov.clone(),
                    right: s_time.clone(),
                });
            }
            Ok(CovariateBlockKind::TimeVarying(
                TimeDependentCovariateBlockInput {
                    design_covariates,
                    time_basis_entry: time_basis_entry.clone(),
                    time_basis_exit: time_basis_exit.clone(),
                    time_basis_derivative_exit: time_basis_derivative_exit.clone(),
                    penalties,
                    initial_log_lambdas,
                    initial_beta,
                    offset: offset.clone(),
                },
            ))
        }
    }
}

/// Survival time-varying tensorization adapter for the shared spatial-ψ engine.
///
/// A time-dependent survival covariate represents each spatial design row as the
/// rowwise-Kronecker of the (spatial) base row against three time bases — exit,
/// entry, and the exit-time derivative — stacked vertically, while each spatial
/// penalty is Kronecker-multiplied against the time identity. This is a *uniform*
/// coordinate change applied to every block the shared engine assembles, so we
/// invert the dependency: the engine owns the spatial-ψ block construction and
/// this adapter only supplies the tensorization via [`SpatialPsiBlockTransform`].
pub(crate) struct SurvivalTimeVaryingPsiTransform {
    pub(crate) time_basis_entry: Array2<f64>,
    pub(crate) time_basis_exit: Array2<f64>,
    pub(crate) time_basis_derivative_exit: Array2<f64>,
}

impl crate::families::spatial_psi_bridge::SpatialPsiBlockTransform
    for SurvivalTimeVaryingPsiTransform
{
    fn transform_operator(
        &self,
        op: Arc<dyn crate::custom_family::CustomFamilyPsiDerivativeOperator>,
    ) -> Result<Arc<dyn crate::custom_family::CustomFamilyPsiDerivativeOperator>, String> {
        build_rowwise_kronecker_psi_operator(
            op,
            vec![
                shared_dense_arc(&self.time_basis_exit),
                shared_dense_arc(&self.time_basis_entry),
                shared_dense_arc(&self.time_basis_derivative_exit),
            ],
        )
    }

    fn transform_design(&self, base: Array2<f64>) -> Array2<f64> {
        let base_dm = DesignMatrix::Dense(DenseDesignMatrix::from(base));
        let exit_design = rowwise_kronecker(&base_dm, &self.time_basis_exit);
        let entry_design = rowwise_kronecker(&base_dm, &self.time_basis_entry);
        let deriv_design = rowwise_kronecker(&base_dm, &self.time_basis_derivative_exit);
        let exit_cow = exit_design.to_dense_cow();
        let entry_cow = entry_design.to_dense_cow();
        let deriv_cow = deriv_design.to_dense_cow();
        let n = exit_cow.nrows();
        let p = exit_cow.ncols();
        let mut stacked = Array2::<f64>::zeros((3 * n, p));
        stacked.slice_mut(s![0..n, ..]).assign(&*exit_cow);
        stacked.slice_mut(s![n..2 * n, ..]).assign(&*entry_cow);
        stacked.slice_mut(s![2 * n..3 * n, ..]).assign(&*deriv_cow);
        stacked
    }

    fn transform_penalty(&self, base: Array2<f64>) -> Array2<f64> {
        let i_time = Array2::<f64>::eye(self.time_basis_exit.ncols());
        kronecker_product(&base, &i_time)
    }
}

/// Survival covariate spatial-ψ derivatives: a thin adapter over the shared
/// exact-derivative engine [`build_block_spatial_psi_derivatives_with_transform`].
/// The `Static` template emits blocks unchanged; the `TimeVarying` template
/// supplies a [`SurvivalTimeVaryingPsiTransform`] so the same engine produces the
/// time-tensorized blocks without re-implementing block assembly.
pub(crate) fn build_survival_covariate_block_psi_derivatives(
    data: ndarray::ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    template: &SurvivalCovariateTermBlockTemplate,
) -> Result<Option<Vec<CustomFamilyBlockPsiDerivative>>, String> {
    match template {
        SurvivalCovariateTermBlockTemplate::Static => {
            crate::families::spatial_psi_bridge::build_block_spatial_psi_derivatives(
                data,
                resolvedspec,
                design,
            )
        }
        SurvivalCovariateTermBlockTemplate::TimeVarying {
            time_basis_entry,
            time_basis_exit,
            time_basis_derivative_exit,
            ..
        } => {
            let transform = SurvivalTimeVaryingPsiTransform {
                time_basis_entry: time_basis_entry.clone(),
                time_basis_exit: time_basis_exit.clone(),
                time_basis_derivative_exit: time_basis_derivative_exit.clone(),
            };
            crate::families::spatial_psi_bridge::build_block_spatial_psi_derivatives_with_transform(
                data,
                resolvedspec,
                design,
                &transform,
            )
        }
    }
}

pub(crate) fn survival_psi_derivatives_support_exact_joint_hessian(
    derivs: &[CustomFamilyBlockPsiDerivative],
) -> bool {
    let psi_dim = derivs.len();
    derivs.iter().all(|deriv| {
        let design_ok = deriv.implicit_operator.is_some()
            || deriv
                .x_psi_psi
                .as_ref()
                .is_some_and(|rows| rows.len() == psi_dim);
        let penalty_ok = deriv
            .s_psi_psi_components
            .as_ref()
            .is_some_and(|rows| rows.len() == psi_dim)
            || deriv
                .s_psi_psi
                .as_ref()
                .is_some_and(|rows| rows.len() == psi_dim);
        design_ok && penalty_ok
    })
}

pub(crate) fn build_survival_two_block_exact_joint_setup(
    data: ndarray::ArrayView2<'_, f64>,
    thresholdspec: &TermCollectionSpec,
    log_sigmaspec: &TermCollectionSpec,
    rho0: Array1<f64>,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    // Survival location-scale uses the shared engine directly: the rho seed is
    // already assembled by the caller (penalty + link-wiggle layout), and the
    // two linear predictors (threshold, log sigma) supply the per-block
    // log(kappa) geometry in theta order.
    build_location_scale_exact_joint_setup(
        data,
        &[thresholdspec, log_sigmaspec],
        rho0,
        kappa_options,
    )
}

pub(crate) fn filtered_initial_beta(
    hint: Option<&Array1<f64>>,
    expected: usize,
) -> Option<Array1<f64>> {
    hint.filter(|beta| beta.len() == expected).cloned()
}
