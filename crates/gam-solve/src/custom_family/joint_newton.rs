//! The joint (cross-block) Newton machinery: the joint-Hessian source
//! (dense / matrix-free operator) and its symmetrized materialization, the
//! drift-derivative closure factories (`build_joint_hessian_closures`),
//! blockwise-logdet terms, the trust-region controller (dogleg / Cauchy /
//! whitened spectral step), the joint PCG solver, and the KKT refusal
//! diagnostics. Hosts the `whitened_spectrum` submodule and its trust-region
//! subproblem tests.
//!
//! The block-diagonal trait-default assemblers (`exact_newton_joint_hessian_*`
//! from exact blocks / working sets and the cross-block coupling probe) are
//! single-sourced in `gam_model_api::families::custom_family::joint_newton_defaults`,
//! where the `CustomFamily` trait that consumes them lives.

use super::*;

pub(crate) fn exact_newton_joint_hessian_symmetrized<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    total: usize,
    context: &str,
) -> Result<Option<Array2<f64>>, String> {
    let Some(mut h) = family.exact_newton_joint_hessian_with_specs(states, specs)? else {
        return Ok(None);
    };
    if h.nrows() != total || h.ncols() != total {
        return Err(format!(
            "{context}: got {}x{}, expected {}x{}",
            h.nrows(),
            h.ncols(),
            total,
            total
        ));
    }
    symmetrize_dense_in_place(&mut h);
    Ok(Some(h))
}

// `ExactNewtonOuterCurvature` is single-sourced in `gam-problem`
// (`custom_family_blockwise`, re-exported as `gam_problem::ExactNewtonOuterCurvature`).
// The carve left a byte-identical, never-referenced copy here; removed.

pub(crate) enum JointHessianSource {
    Dense(Array2<f64>),
    Operator {
        apply: Arc<dyn Fn(&Array1<f64>) -> Result<Array1<f64>, String> + Send + Sync>,
        /// Write-into matvec used by the inner-Newton PCG hot path so the
        /// matvec result no longer allocates an `Array1<f64>` per CG iter.
        /// At large scale (~6400 inner CG iters per outer iter, p~200) this
        /// removes thousands of small Vec<f64> allocations from the tightest
        /// loop. Wired from `workspace.hessian_matvec_into`.
        apply_into: Arc<dyn Fn(&Array1<f64>, &mut Array1<f64>) -> Result<(), String> + Send + Sync>,
        /// Batched multi-RHS apply: `out = H · V` for `(total, n_rhs)` `V`.
        /// Wired from `workspace.hessian_apply_mat`, which for the BMS tiled
        /// row-primary Hessian sweeps each row tile once and applies its `Hᵢ`
        /// to every column. Column-basis dense reconstruction below uses this
        /// to materialise the operator in one batched sweep (`H = H · I`)
        /// rather than `total` single-vector HVPs, each of which re-reads every
        /// row tile. Numerically identical to looping `apply_into`.
        apply_mat: Arc<dyn Fn(&Array2<f64>, &mut Array2<f64>) -> Result<(), String> + Send + Sync>,
        diagonal: Array1<f64>,
        /// Forced dense materialization that bypasses the workspace's
        /// `hessian_dense` amortization gate. Returns `Some` when the
        /// workspace can build dense via a structural direct path (e.g.
        /// CTN's `scop_gradient_and_negative_hessian`), `None` when the
        /// caller should fall back to column-basis HVP through `apply`.
        dense_forced: Arc<dyn Fn() -> Result<Option<Array2<f64>>, String> + Send + Sync>,
    },
}

pub(crate) const EXACT_JOINT_HESSIAN_DENSE_MAX_BYTES: usize = 512 * 1024 * 1024;

pub(crate) fn exact_joint_hessian_dense_bytes(total: usize) -> Result<usize, String> {
    total
        .checked_mul(total)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| format!("joint Hessian dense byte count overflow for dim={total}"))
}

pub(crate) fn ensure_exact_joint_hessian_dense_budget(
    total: usize,
    context: &str,
) -> Result<(), String> {
    let bytes = exact_joint_hessian_dense_bytes(total)?;
    if bytes > EXACT_JOINT_HESSIAN_DENSE_MAX_BYTES {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: format!(
                "{context}: exact dense joint Hessian requires {:.2} GiB for dim={total}, \
             exceeding the {:.2} GiB cap; refusing approximate determinant algebra",
                bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                EXACT_JOINT_HESSIAN_DENSE_MAX_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
            ),
        }
        .into());
    }
    Ok(())
}

pub(crate) struct JointHessianBundle<'a> {
    pub(crate) source: JointHessianSource,
    pub(crate) beta_flat: Array1<f64>,
    pub(crate) compute_dh: Box<DriftDerivFn<'a>>,
    pub(crate) compute_dh_many: Option<Box<DriftDerivManyFn<'a>>>,
    pub(crate) compute_d2h: Box<DriftSecondDerivFn<'a>>,
    /// Optional batched second-derivative callback. The unified evaluator's
    /// outer-Hessian ρ-ρ pair loop forwards the K(K+1)/2 (v_k, v_l) pairs
    /// here in one call when set, so families that fuse the per-row D²H walk
    /// (e.g. survival marginal-slope scanning n rows once per outer eval)
    /// amortise the row-walk across all pairs instead of paying it per pair.
    pub(crate) compute_d2h_many: Option<Box<DriftSecondDerivManyFn<'a>>>,
    pub(crate) owned_compute_dh:
        Option<Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>>,
    pub(crate) owned_compute_dh_many: Option<
        Arc<dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, String> + Send + Sync>,
    >,
    pub(crate) owned_compute_d2h: Option<
        Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
                + Send
                + Sync,
        >,
    >,
    /// Owned twin of `compute_d2h_many`. Threaded through to
    /// `OwnedJointDerivProvider` so the unified evaluator can share the
    /// callback across rayon worker threads when the outer Hessian routes
    /// through the parallel pair dispatch.
    pub(crate) owned_compute_d2h_many: Option<
        Arc<
            dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Option<DriftDerivResult>>, String>
                + Send
                + Sync,
        >,
    >,
    pub(crate) rho_curvature_scale: f64,
    pub(crate) hessian_logdet_correction: f64,
}

pub(crate) type DriftDerivFn<'a> =
    dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync + 'a;

pub(crate) type DriftDerivManyFn<'a> =
    dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, String> + Send + Sync + 'a;

pub(crate) type DriftSecondDerivFn<'a> = dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
    + Send
    + Sync
    + 'a;

pub(crate) type DriftSecondDerivManyFn<'a> = dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Option<DriftDerivResult>>, String>
    + Send
    + Sync
    + 'a;

/// Cheap, deterministic non-finite-curvature probe for the joint Hessian
/// source (gam#1088). A `NaN`/`Inf` in the penalized Hessian `H_pen = H +
/// S(λ)` makes its spectrum (`λ_max`, `λ_min`, `cond`) degenerate to `NaN`,
/// so the KKT certificate — which thresholds on that spectrum — can *never*
/// be satisfied, the spectral step solve returns garbage, and the projected
/// residual neither converges nor early-exits through the finite-comparison
/// guards. Left undetected the coupled joint-Newton loop then grinds the full
/// `inner_loop_hard_ceiling` (1200 cycles) on *every* outer ρ-eval / seed,
/// which is the multi-hour benchmark timeout (link-wiggle & location-scale
/// paths). The penalty `S(λ)` is finite by construction, so a non-finite
/// `H_pen` originates entirely in the family curvature `H` we probe here.
///
/// For the `Dense` variant the matrix is already materialized, so we scan it
/// directly. For the matrix-free `Operator` variant the full operator is never
/// formed; we scan its `diagonal`, which carries the per-row assembled
/// curvature (`XᵀWX`) — exactly where a collapsed/`0÷0` row weight injects the
/// `NaN`, corrupting the whole row block including its diagonal entry. Returns
/// `true` when every probed entry is finite.
pub(crate) fn joint_hessian_source_curvature_is_finite(source: &JointHessianSource) -> bool {
    match source {
        JointHessianSource::Dense(h_joint) => h_joint.iter().all(|v| v.is_finite()),
        JointHessianSource::Operator { diagonal, .. } => diagonal.iter().all(|v| v.is_finite()),
    }
}

pub(crate) fn materialize_joint_hessian_source(
    source: &JointHessianSource,
    total: usize,
    context: &str,
) -> Result<Array2<f64>, String> {
    match source {
        JointHessianSource::Dense(matrix) => Ok(matrix.clone()),
        JointHessianSource::Operator {
            apply_mat,
            dense_forced,
            ..
        } => {
            ensure_exact_joint_hessian_dense_budget(total, context)?;
            // Preferred path: the workspace exposes a structural direct-dense
            // build (e.g. SCOP's `scop_gradient_and_negative_hessian`). That
            // is `Θ(n·p²)` like column-basis HVP would be, but the constant
            // factor is much better because the structural build sweeps rows
            // once and uses BLAS-3 for the chain-rule pullback. Falling back
            // to column-basis HVP would re-walk all `n` rows once per column.
            if let Some(mut matrix) = dense_forced()? {
                if matrix.nrows() != total || matrix.ncols() != total {
                    return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                        "{context}: dense_forced shape mismatch: got {}x{}, expected {total}x{total}",
                        matrix.nrows(),
                        matrix.ncols()
                    ) }.into());
                }
                if matrix.iter().any(|value| !value.is_finite()) {
                    return Err(CustomFamilyError::NumericalFailure {
                        reason: format!("{context}: dense_forced returned non-finite values"),
                    }
                    .into());
                }
                symmetrize_dense_in_place(&mut matrix);
                return Ok(matrix);
            }
            // Column-basis reconstruction `H = H · I`. Driving it through the
            // batched multi-RHS apply lets a tiled/streamed operator sweep each
            // row tile exactly once for all `total` columns instead of once per
            // column (`total` full sweeps). The result is, column for column,
            // identical to applying the operator to each unit basis vector.
            let identity = Array2::<f64>::eye(total);
            let mut matrix = Array2::<f64>::zeros((total, total));
            apply_mat(&identity, &mut matrix)?;
            if matrix.iter().any(|value| !value.is_finite()) {
                return Err(CustomFamilyError::NumericalFailure {
                    reason: format!("{context}: operator matvec returned non-finite values"),
                }
                .into());
            }
            symmetrize_dense_in_place(&mut matrix);
            Ok(matrix)
        }
    }
}

pub(crate) fn exact_newton_joint_hessian_source_from_workspace(
    workspace: &Arc<dyn ExactNewtonJointHessianWorkspace>,
    total: usize,
    intent: MaterializationIntent,
    context: &str,
) -> Result<Option<JointHessianSource>, String> {
    if workspace.hessian_source_preference_for_intent(intent)
        == JointHessianSourcePreference::Operator
    {
        return exact_newton_joint_hessian_operator_source_from_workspace(
            workspace, total, intent, context,
        );
    }

    if let Some(mut hessian) = workspace.hessian_dense()? {
        if hessian.nrows() != total || hessian.ncols() != total {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "{context}: dense Hessian shape mismatch: got {}x{}, expected {total}x{total}",
                    hessian.nrows(),
                    hessian.ncols()
                ),
            }
            .into());
        }
        if hessian.iter().any(|value| !value.is_finite()) {
            return Err(CustomFamilyError::NumericalFailure {
                reason: format!("{context}: dense Hessian contains non-finite values"),
            }
            .into());
        }
        symmetrize_dense_in_place(&mut hessian);
        return Ok(Some(JointHessianSource::Dense(hessian)));
    }

    exact_newton_joint_hessian_operator_source_from_workspace(workspace, total, intent, context)
}

pub(crate) fn exact_newton_joint_hessian_operator_source_from_workspace(
    workspace: &Arc<dyn ExactNewtonJointHessianWorkspace>,
    total: usize,
    intent: MaterializationIntent,
    context: &str,
) -> Result<Option<JointHessianSource>, String> {
    let Some(diagonal) = workspace.hessian_diagonal()? else {
        if workspace.hessian_source_preference_for_intent(intent)
            == JointHessianSourcePreference::Operator
        {
            return Err(CustomFamilyError::UnsupportedConfiguration {
                reason: format!(
                    "{context}: operator-preferred Hessian workspace did not provide a diagonal"
                ),
            }
            .into());
        }
        return Ok(None);
    };
    if diagonal.len() != total {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{context}: operator diagonal length mismatch: got {}, expected {}",
                diagonal.len(),
                total
            ),
        }
        .into());
    }
    if diagonal.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::NumericalFailure {
            reason: format!("{context}: operator diagonal contains non-finite values"),
        }
        .into());
    }

    if !workspace.hessian_matvec_available() {
        if workspace.hessian_source_preference_for_intent(intent)
            == JointHessianSourcePreference::Operator
        {
            return Err(CustomFamilyError::UnsupportedConfiguration {
                reason: format!(
                    "{context}: operator-preferred Hessian workspace did not provide HVPs"
                ),
            }
            .into());
        }
        return Ok(None);
    }

    let workspace_apply = Arc::clone(workspace);
    let workspace_apply_into = Arc::clone(workspace);
    let workspace_apply_mat = Arc::clone(workspace);
    let workspace_dense_forced = Arc::clone(workspace);
    let context_apply: Arc<str> = Arc::from(context);
    let context_apply_into = Arc::clone(&context_apply);
    let context_apply_mat = Arc::clone(&context_apply);
    let context_dense_forced = Arc::clone(&context_apply);
    Ok(Some(JointHessianSource::Operator {
        apply: Arc::new(move |v: &Array1<f64>| {
            if v.len() != total {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "{}: operator input length mismatch: got {}, expected {total}",
                        &*context_apply,
                        v.len()
                    ),
                }
                .into());
            }
            let Some(out) = workspace_apply.hessian_matvec(v)? else {
                return Err(CustomFamilyError::UnsupportedConfiguration {
                    reason: "joint exact-newton operator matvec unavailable".to_string(),
                }
                .into());
            };
            if out.len() != total {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "{}: operator matvec length mismatch: got {}, expected {total}",
                        &*context_apply,
                        out.len()
                    ),
                }
                .into());
            }
            if out.iter().any(|value| !value.is_finite()) {
                return Err(CustomFamilyError::NumericalFailure {
                    reason: format!(
                        "{}: operator matvec returned non-finite values",
                        &*context_apply
                    ),
                }
                .into());
            }
            Ok(out)
        }),
        apply_into: Arc::new(move |v: &Array1<f64>, out: &mut Array1<f64>| {
            if v.len() != total || out.len() != total {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "{}: operator input/output length mismatch: v={} out={} expected={total}",
                        &*context_apply_into,
                        v.len(),
                        out.len()
                    ),
                }
                .into());
            }
            if !workspace_apply_into.hessian_matvec_into(v, out)? {
                return Err(CustomFamilyError::UnsupportedConfiguration {
                    reason: "joint exact-newton operator matvec unavailable".to_string(),
                }
                .into());
            }
            if out.iter().any(|value| !value.is_finite()) {
                return Err(CustomFamilyError::NumericalFailure {
                    reason: format!(
                        "{}: operator matvec returned non-finite values",
                        &*context_apply_into
                    ),
                }
                .into());
            }
            Ok(())
        }),
        apply_mat: Arc::new(move |v_cols: &Array2<f64>, out: &mut Array2<f64>| {
            if v_cols.nrows() != total || out.nrows() != total {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "{}: operator batched apply row mismatch: v_cols={}x{} out={}x{} expected rows={total}",
                        &*context_apply_mat,
                        v_cols.nrows(),
                        v_cols.ncols(),
                        out.nrows(),
                        out.ncols()
                    ),
                }
                .into());
            }
            if v_cols.ncols() != out.ncols() {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "{}: operator batched apply column mismatch: v_cols has {} columns, out has {}",
                        &*context_apply_mat,
                        v_cols.ncols(),
                        out.ncols()
                    ),
                }
                .into());
            }
            if !workspace_apply_mat.hessian_apply_mat(v_cols, out)? {
                return Err(CustomFamilyError::UnsupportedConfiguration {
                    reason: "joint exact-newton operator batched apply unavailable".to_string(),
                }
                .into());
            }
            if out.iter().any(|value| !value.is_finite()) {
                return Err(CustomFamilyError::NumericalFailure {
                    reason: format!(
                        "{}: operator batched apply returned non-finite values",
                        &*context_apply_mat
                    ),
                }
                .into());
            }
            Ok(())
        }),
        diagonal,
        dense_forced: Arc::new(move || -> Result<Option<Array2<f64>>, String> {
            match workspace_dense_forced.hessian_dense_forced()? {
                Some(mut matrix) => {
                    if matrix.nrows() != total || matrix.ncols() != total {
                        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                            "{}: hessian_dense_forced shape mismatch: got {}x{}, expected {total}x{total}",
                            &*context_dense_forced,
                            matrix.nrows(),
                            matrix.ncols()
                        ) }.into());
                    }
                    if matrix.iter().any(|value| !value.is_finite()) {
                        return Err(CustomFamilyError::NumericalFailure {
                            reason: format!(
                                "{}: hessian_dense_forced returned non-finite values",
                                &*context_dense_forced
                            ),
                        }
                        .into());
                    }
                    symmetrize_dense_in_place(&mut matrix);
                    Ok(Some(matrix))
                }
                None => Ok(None),
            }
        }),
    }))
}

pub(crate) fn symmetrized_square_matrix(
    mut matrix: Array2<f64>,
    expected: usize,
    context: &str,
) -> Result<Array2<f64>, String> {
    if matrix.nrows() != expected || matrix.ncols() != expected {
        return Err(format!(
            "{context}: got {}x{}, expected {}x{}",
            matrix.nrows(),
            matrix.ncols(),
            expected,
            expected
        ));
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::NumericalFailure {
            reason: format!("{context}: matrix contains non-finite values"),
        }
        .into());
    }
    symmetrize_dense_in_place(&mut matrix);
    Ok(matrix)
}

/// Try exact Newton joint Hessian first, then surrogate. Returns `None` if
/// neither path provides a joint Hessian. When successful, returns the joint
/// Hessian source, flat beta, and boxed closures for computing directional
/// derivatives dH[v] and d²H[u,v].
///
/// This eliminates the previously duplicated exact-Newton and surrogate
/// code blocks in `outerobjectivegradienthessian_internal`.
pub(crate) fn build_joint_hessian_closures<'a, F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &'a F,
    block_states: &'a [ParameterBlockState],
    specs: &'a [ParameterBlockSpec],
    total: usize,
    options: &BlockwiseFitOptions,
    preferred_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Result<Option<JointHessianBundle<'a>>, String> {
    // Path 1: exact Newton joint Hessian (preferred).
    let beta_flat = flatten_state_betas(block_states, specs);
    let synced = Arc::new(synchronized_states_from_flat_beta(
        family,
        specs,
        block_states,
        &beta_flat,
    )?);
    let hessian_workspace = match preferred_workspace {
        Some(workspace) => Some(workspace),
        None => family.exact_newton_joint_hessian_workspace_with_options(
            synced.as_ref(),
            specs,
            options,
        )?,
    };
    // Outer-eval entry: prime any per-row jet caches the workspace will hand
    // to the directional-derivative path. Runs at top-level rayon (we are
    // outside the ext-coord `par_iter` here), so the cache build's own
    // `par_iter` enjoys full thread-pool parallelism. PIRLS-side workspace
    // construction skips this priming because PIRLS never invokes
    // `directional_derivative_operator`.
    if let Some(workspace) = hessian_workspace.as_ref() {
        workspace.warm_up_outer_caches()?;
    }
    if let Some(curvature) = family.exact_newton_outer_curvature(block_states)? {
        let h_joint_unpen = JointHessianSource::Dense(symmetrized_square_matrix(
            curvature.hessian,
            total,
            "joint exact-newton Hessian shape mismatch in outer gradient (rescaled)",
        )?);
        let compute_dh = Box::new(exact_newton_dh_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            true,
            1.0,
            hessian_workspace.clone(),
        ));
        let compute_dh_many = None;
        let compute_d2h = Box::new(exact_newton_d2h_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            true,
            1.0,
            hessian_workspace.clone(),
        ));
        let owned_compute_dh = exact_newton_dh_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            true,
            1.0,
            hessian_workspace.clone(),
        );
        let owned_compute_dh_many = None;
        let owned_compute_d2h = exact_newton_d2h_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            true,
            1.0,
            hessian_workspace.clone(),
        );
        return Ok(Some(JointHessianBundle {
            source: h_joint_unpen,
            beta_flat,
            compute_dh,
            compute_dh_many,
            compute_d2h,
            compute_d2h_many: None,
            owned_compute_dh: Some(owned_compute_dh),
            owned_compute_dh_many,
            owned_compute_d2h: Some(owned_compute_d2h),
            owned_compute_d2h_many: None,
            rho_curvature_scale: curvature.rho_curvature_scale,
            hessian_logdet_correction: curvature.hessian_logdet_correction,
        }));
    }
    let exact_joint_source = if let Some(workspace) = hessian_workspace.as_ref() {
        exact_newton_joint_hessian_source_from_workspace(
            workspace,
            total,
            MaterializationIntent::OuterGradient,
            "joint exact-newton operator mismatch in outer gradient",
        )?
    } else {
        None
    };
    let exact_joint_source = match exact_joint_source {
        Some(source) => Some(source),
        None => exact_newton_joint_hessian_symmetrized(
            family,
            block_states,
            specs,
            total,
            "joint exact-newton Hessian shape mismatch in outer gradient",
        )
        .map(|source| source.map(JointHessianSource::Dense))?,
    };
    if let Some(h_joint_unpen) = exact_joint_source {
        let compute_dh = Box::new(exact_newton_dh_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            false,
            1.0,
            hessian_workspace.clone(),
        ));
        let compute_dh_many = exact_newton_dh_many_closure(1.0, hessian_workspace.clone());
        let compute_d2h = Box::new(exact_newton_d2h_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            false,
            1.0,
            hessian_workspace.clone(),
        ));
        let owned_compute_dh = exact_newton_dh_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            false,
            1.0,
            hessian_workspace.clone(),
        );
        let owned_compute_dh_many =
            exact_newton_dh_many_closure_owned(1.0, hessian_workspace.clone());
        let owned_compute_d2h = exact_newton_d2h_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            false,
            1.0,
            hessian_workspace.clone(),
        );
        let compute_d2h_many = exact_newton_d2h_many_closure(1.0, hessian_workspace.clone());
        let owned_compute_d2h_many =
            exact_newton_d2h_many_closure_owned(1.0, hessian_workspace.clone());
        return Ok(Some(JointHessianBundle {
            source: h_joint_unpen,
            beta_flat,
            compute_dh,
            compute_dh_many,
            compute_d2h,
            compute_d2h_many,
            owned_compute_dh: Some(owned_compute_dh),
            owned_compute_dh_many,
            owned_compute_d2h: Some(owned_compute_d2h),
            owned_compute_d2h_many,
            rho_curvature_scale: 1.0,
            hessian_logdet_correction: 0.0,
        }));
    }

    // Path 2: surrogate joint Hessian (fallback).
    if let Some(h_joint_unpen) = family
        .joint_outer_hyper_surrogate_hessian_with_specs(block_states, specs)?
        .map(|h| {
            symmetrized_square_matrix(
                h,
                total,
                "joint outer-hyper surrogate Hessian shape mismatch",
            )
        })
        .transpose()?
    {
        let beta_flat = flatten_state_betas(block_states, specs);

        let compute_dh = Box::new(
            move |v_k: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                let h_rho = family
                    .joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
                        block_states,
                        specs,
                        v_k,
                    )?;
                match h_rho {
                    Some(h) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        h,
                        total,
                        "joint surrogate dH shape mismatch",
                    )?))),
                    None => Err(CustomFamilyError::UnsupportedConfiguration {
                        reason: "joint surrogate dH unavailable for analytic outer gradient"
                            .to_string(),
                    }
                    .into()),
                }
            },
        );
        let compute_d2h = Box::new(
            move |u: &Array1<f64>, v: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                match family
                    .joint_outer_hyper_surrogate_hessian_second_directional_derivative_with_specs(
                        block_states,
                        specs,
                        u,
                        v,
                    )? {
                    Some(m) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        m,
                        total,
                        "joint surrogate d2H shape mismatch",
                    )?))),
                    None => Ok(None),
                }
            },
        );
        let family_owned = family.clone();
        let states_owned = block_states.to_vec();
        let specs_owned = specs.to_vec();
        let owned_compute_dh = Arc::new(
            move |v_k: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                match family_owned
                    .joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
                        &states_owned,
                        &specs_owned,
                        v_k,
                    )? {
                    Some(h) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        h,
                        total,
                        "joint surrogate dH shape mismatch",
                    )?))),
                    None => Err(CustomFamilyError::UnsupportedConfiguration {
                        reason: "joint surrogate dH unavailable for analytic outer gradient"
                            .to_string(),
                    }
                    .into()),
                }
            },
        );
        let family_owned = family.clone();
        let states_owned = block_states.to_vec();
        let specs_owned = specs.to_vec();
        let owned_compute_d2h = Arc::new(
            move |u: &Array1<f64>, v: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                match family_owned
                    .joint_outer_hyper_surrogate_hessian_second_directional_derivative_with_specs(
                        &states_owned,
                        &specs_owned,
                        u,
                        v,
                    )? {
                    Some(m) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        m,
                        total,
                        "joint surrogate d2H shape mismatch",
                    )?))),
                    None => Ok(None),
                }
            },
        );
        return Ok(Some(JointHessianBundle {
            source: JointHessianSource::Dense(h_joint_unpen),
            beta_flat,
            compute_dh,
            compute_dh_many: None,
            compute_d2h,
            compute_d2h_many: None,
            owned_compute_dh: Some(owned_compute_dh),
            owned_compute_dh_many: None,
            owned_compute_d2h: Some(owned_compute_d2h),
            owned_compute_d2h_many: None,
            rho_curvature_scale: 1.0,
            hessian_logdet_correction: 0.0,
        }));
    }

    Ok(None)
}

/// Build a closure computing dH[v] using exact Newton derivatives on synced states.
/// Non-finite derivative output is treated as a hard error.
/// Symmetrize-and-scale the dH Dense result, optionally rejecting non-finite
/// values first.  The borrowed factory (`exact_newton_dh_closure`) guards
/// against non-finite output (`check_finite = true`); the owned factory
/// (`exact_newton_dh_closure_owned`) historically does not (`check_finite =
/// false`).  Routing both through this helper keeps that behavioral
/// distinction explicit rather than silently divergent.
pub(crate) fn finalize_dh_dense(
    h: Array2<f64>,
    total: usize,
    scale: f64,
    check_finite: bool,
) -> Result<Option<DriftDerivResult>, String> {
    if check_finite && h.iter().any(|v| !v.is_finite()) {
        return Err(CustomFamilyError::NumericalFailure {
            reason: "joint exact-newton dH returned non-finite values".to_string(),
        }
        .into());
    }
    let mut sym = symmetrized_square_matrix(h, total, "joint exact-newton dH shape mismatch")?;
    if scale != 1.0 {
        sym *= scale;
    }
    Ok(Some(DriftDerivResult::Dense(sym)))
}

/// Single source of truth for the dH[v] three-way dispatch shared by the
/// borrowed (`exact_newton_dh_closure`) and owned
/// (`exact_newton_dh_closure_owned`) closure factories.  The `check_finite`
/// flag preserves the lone behavioral difference between the two (the borrowed
/// variant rejects non-finite dense output, the owned variant does not); all
/// other logic — outer-curvature path, workspace-operator fast path, and
/// joint-Hessian fallback — is identical and lives here once.
pub(crate) fn exact_newton_dh_apply<F: CustomFamily + Sync>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<&Arc<dyn ExactNewtonJointHessianWorkspace>>,
    check_finite: bool,
    v_k: &Array1<f64>,
) -> Result<Option<DriftDerivResult>, String> {
    // `v_k` is ALREADY the perturbation direction `δβ` the caller wants the
    // directional Hessian derivative evaluated along. The `HessianDerivativeProvider`s
    // (`BorrowedJointDerivProvider`/`OwnedJointDerivProvider`) own the implicit-
    // function-theorem sign `δβ = −H⁻¹(A_k β̂)` and negate before calling this
    // closure (matching `exact_newton_d2h_apply` and the owned `_many` closure,
    // which also pass the direction straight through). Re-negating here would
    // double-negate `D_β H[δβ]`, flipping the mode-response drift in the outer
    // LAML trace `½ tr(K · (B_i + D_β H[δβ_i]))` and desynchronising the analytic
    // outer gradient from its objective for every β-dependent-Hessian exact
    // family (spatial-adaptive, survival/bernoulli marginal-slope). Pass through.
    let mode_response = v_k.clone();
    if use_outer_curvature_derivatives {
        let h_rho = family.exact_newton_outer_curvature_directional_derivative_with_specs(
            synced_states,
            specs,
            &mode_response,
        )?;
        return match h_rho {
            Some(h) => finalize_dh_dense(h, total, scale, check_finite),
            None => Err(CustomFamilyError::UnsupportedConfiguration {
                reason: "joint exact-newton dH unavailable for analytic outer gradient".to_string(),
            }
            .into()),
        };
    }

    if let Some(workspace) = workspace
        && let Some(operator) = workspace.directional_derivative_operator(&mode_response)?
    {
        return Ok(Some(scale_drift_deriv_result(
            DriftDerivResult::Operator(operator),
            scale,
        )));
    }

    match family.exact_newton_joint_hessian_directional_derivative_with_specs(
        synced_states,
        specs,
        &mode_response,
    )? {
        Some(h) => finalize_dh_dense(h, total, scale, check_finite),
        None => Err(CustomFamilyError::UnsupportedConfiguration {
            reason: "joint exact-newton dH unavailable for analytic outer gradient".to_string(),
        }
        .into()),
    }
}

pub(crate) fn exact_newton_dh_closure<'a, F: CustomFamily + Sync>(
    family: &'a F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: &'a [ParameterBlockSpec],
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> impl Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync + 'a {
    move |v_k: &Array1<f64>| {
        exact_newton_dh_apply(
            family,
            synced_states.as_ref(),
            specs,
            total,
            use_outer_curvature_derivatives,
            scale,
            workspace.as_ref(),
            true,
            v_k,
        )
    }
}

pub(crate) fn exact_newton_dh_many_closure<'a>(
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Option<Box<DriftDerivManyFn<'a>>> {
    let workspace = workspace?;
    Some(Box::new(move |directions: &[Array1<f64>]| {
        // `directions` are already the perturbation directions `δβ`; the provider
        // owns the IFT sign and pre-negates (see `exact_newton_dh_apply`). The
        // owned `_many` counterpart passes them straight through, so this borrowed
        // path must too — re-negating here double-flips the mode-response drift.
        workspace
            .directional_derivative_operators(directions)?
            .into_iter()
            .map(|maybe_operator| {
                Ok(maybe_operator.map(|operator| {
                    scale_drift_deriv_result(DriftDerivResult::Operator(operator), scale)
                }))
            })
            .collect()
    }))
}

/// Single source of truth for the d²H[u,v] three-way dispatch shared by the
/// borrowed (`exact_newton_d2h_closure`) and owned
/// (`exact_newton_d2h_closure_owned`) closure factories.  Takes references for
/// `family`/`specs` so both ownership flavors can call it; the only difference
/// between the two factories is borrow-vs-own plumbing, which lives in the
/// wrappers, not here.
pub(crate) fn exact_newton_d2h_apply<F: CustomFamily + Sync>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<&Arc<dyn ExactNewtonJointHessianWorkspace>>,
    u: &Array1<f64>,
    v: &Array1<f64>,
) -> Result<Option<DriftDerivResult>, String> {
    if use_outer_curvature_derivatives {
        return match family.exact_newton_outer_curvature_second_directional_derivative_with_specs(
            synced_states,
            specs,
            u,
            v,
        )? {
            Some(m) => {
                let mut sym =
                    symmetrized_square_matrix(m, total, "joint exact-newton d2H shape mismatch")?;
                if scale != 1.0 {
                    sym *= scale;
                }
                Ok(Some(DriftDerivResult::Dense(sym)))
            }
            None => Ok(None),
        };
    }

    if let Some(workspace) = workspace
        && let Some(operator) = workspace.second_directional_derivative_operator(u, v)?
    {
        return Ok(Some(scale_drift_deriv_result(
            DriftDerivResult::Operator(operator),
            scale,
        )));
    }

    match family.exact_newton_joint_hessian_second_directional_derivative_with_specs(
        synced_states,
        specs,
        u,
        v,
    )? {
        Some(m) => {
            let mut sym =
                symmetrized_square_matrix(m, total, "joint exact-newton d2H shape mismatch")?;
            if scale != 1.0 {
                sym *= scale;
            }
            Ok(Some(DriftDerivResult::Dense(sym)))
        }
        None => Ok(None),
    }
}

/// Build a closure computing d²H[u,v] using exact Newton derivatives on synced states.
pub(crate) fn exact_newton_d2h_closure<'a, F: CustomFamily + Sync>(
    family: &'a F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: &'a [ParameterBlockSpec],
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> impl Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync + 'a
{
    move |u: &Array1<f64>, v: &Array1<f64>| {
        exact_newton_d2h_apply(
            family,
            synced_states.as_ref(),
            specs,
            total,
            use_outer_curvature_derivatives,
            scale,
            workspace.as_ref(),
            u,
            v,
        )
    }
}

pub(crate) fn exact_newton_d2h_many_closure<'a>(
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Option<Box<DriftSecondDerivManyFn<'a>>> {
    let workspace = workspace?;
    Some(Box::new(move |pairs: &[(Array1<f64>, Array1<f64>)]| {
        workspace
            .second_directional_derivative_operators(pairs)?
            .into_iter()
            .map(|maybe_operator| {
                Ok(maybe_operator.map(|operator| {
                    scale_drift_deriv_result(DriftDerivResult::Operator(operator), scale)
                }))
            })
            .collect()
    }))
}

pub(crate) fn exact_newton_dh_closure_owned<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: Vec<ParameterBlockSpec>,
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync> {
    Arc::new(move |v_k: &Array1<f64>| {
        exact_newton_dh_apply(
            &family,
            synced_states.as_ref(),
            &specs,
            total,
            use_outer_curvature_derivatives,
            scale,
            workspace.as_ref(),
            false,
            v_k,
        )
    })
}

pub(crate) fn exact_newton_dh_many_closure_owned(
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Option<
    Arc<dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, String> + Send + Sync>,
> {
    let workspace = workspace?;
    Some(Arc::new(move |directions: &[Array1<f64>]| {
        workspace
            .directional_derivative_operators(directions)?
            .into_iter()
            .map(|maybe_operator| {
                Ok(maybe_operator.map(|operator| {
                    scale_drift_deriv_result(DriftDerivResult::Operator(operator), scale)
                }))
            })
            .collect()
    }))
}

pub(crate) fn exact_newton_d2h_closure_owned<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: Vec<ParameterBlockSpec>,
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Arc<dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>
{
    Arc::new(move |u: &Array1<f64>, v: &Array1<f64>| {
        exact_newton_d2h_apply(
            &family,
            synced_states.as_ref(),
            &specs,
            total,
            use_outer_curvature_derivatives,
            scale,
            workspace.as_ref(),
            u,
            v,
        )
    })
}

pub(crate) fn exact_newton_d2h_many_closure_owned(
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Option<
    Arc<
        dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Option<DriftDerivResult>>, String>
            + Send
            + Sync,
    >,
> {
    let workspace = workspace?;
    Some(Arc::new(move |pairs: &[(Array1<f64>, Array1<f64>)]| {
        workspace
            .second_directional_derivative_operators(pairs)?
            .into_iter()
            .map(|maybe_operator| {
                Ok(maybe_operator.map(|operator| {
                    scale_drift_deriv_result(DriftDerivResult::Operator(operator), scale)
                }))
            })
            .collect()
    }))
}

pub(crate) fn include_exact_newton_logdet_h<F: CustomFamily + ?Sized>(
    family: &F,
    options: &BlockwiseFitOptions,
) -> bool {
    options.use_remlobjective
        && matches!(
            family.exact_newton_outerobjective(),
            ExactNewtonOuterObjective::RidgedQuadraticReml
                | ExactNewtonOuterObjective::StrictPseudoLaplace
        )
}

pub fn custom_family_outer_derivatives<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> (gam_problem::Derivative, gam_problem::DeclaredHessianForm) {
    use gam_problem::{DeclaredHessianForm, Derivative};

    // The capability-vs-policy split: capability tells us *what the family
    // can compute*; policy tells us *what we should ask for at this size*.
    //
    // For the outer-strategy declaration here we have only `specs` and
    // `options` (no resolved psi_dim), so policy is queried at
    // psi_dim = 0 — the gradient/Hessian forms returned here are the
    // pre-psi declarations consumed by the outer planner ladder. The
    // per-iter clamp in `optimize_spatial_length_scale_exact_joint`
    // consults `outer_derivative_policy` again with the realized
    // psi_dim for the κ optimizer.
    let policy = family.outer_derivative_policy(specs, 0, options);
    let gradient = if policy.capability.has_gradient() {
        Derivative::Analytic
    } else {
        Derivative::Unavailable
    };
    // The analytic outer Hessian is routed to ARC whenever the realized family
    // exposes second-order calculus. Matrix-free Hessian support is a
    // representation capability used by the evaluator; it must not be hidden
    // from the outer optimizer by a cost-based first-order policy.
    let hessian = if options.use_outer_hessian
        && include_exact_newton_logdet_h(family, options)
        && policy.capability.has_hessian()
    {
        DeclaredHessianForm::Either
    } else {
        DeclaredHessianForm::Unavailable
    };

    (gradient, hessian)
}

pub(crate) fn include_exact_newton_logdet_s<F: CustomFamily + ?Sized>(
    family: &F,
    options: &BlockwiseFitOptions,
) -> bool {
    family.exact_newton_outerobjective() == ExactNewtonOuterObjective::RidgedQuadraticReml
        && options.use_remlobjective
}

pub(crate) fn use_exact_newton_strict_spd<F: CustomFamily + ?Sized>(family: &F) -> bool {
    family.exact_newton_outerobjective() == ExactNewtonOuterObjective::StrictPseudoLaplace
}

pub(crate) fn blockwise_logdet_terms<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<(f64, f64), String> {
    blockwise_logdet_terms_with_workspace(family, specs, states, block_log_lambdas, options, None)
}

pub(crate) fn blockwise_logdet_terms_with_workspace<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    preferred_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Result<(f64, f64), String> {
    let include_logdet_h = include_exact_newton_logdet_h(family, options);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    if !include_logdet_h && !include_logdet_s {
        return Ok((0.0, 0.0));
    }
    let strict_spd = use_exact_newton_strict_spd(family);
    refresh_all_block_etas(family, specs, states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    // Universal full-span robustness: the outer REML logdet of the
    // penalized Hessian must use the SAME Jeffreys-augmented Hessian
    // `H + S_λ + H_Φ` the inner Newton converged on, or the LAML score and its
    // analytic derivatives describe a different objective. Compute `H_Φ` once
    // over the full-span basis `Z_J` and add it into whichever
    // logdet path runs below. `None` ⇒ no logdet-H contribution (logdet-S only).
    // Cheap matrix-free conditioning pre-check for the OUTER logdet H_Φ. When a
    // matrix-free workspace exposes the Hessian-vector product, bound the joint
    // information's spectrum from a few matvecs (no dense H, no O(p³) eigh): if it
    // certifies well-conditioned the exact gate is certain to return H_Φ = 0, so
    // we skip the whole dense formation and use `None` (no logdet-H Jeffreys
    // contribution), byte-identical to the gated-off path. This keeps the outer
    // LAML logdet consistent with the inner solve (which also gated the term off
    // on the same well-conditioned geometry) while preserving the matrix-free path
    // at outer-eval scale. Returns `false`/unsure ⇒ exact formation below.
    //
    // EXPECTED-INFORMATION GUARD (gam#1020): the matvec certifies the
    // OBSERVED Hessian's conditioning; when the family overrides the Jeffreys
    // information with the expected Fisher information the certificate does
    // not transfer (observed grows on saturated rows where expected decays),
    // so the pre-check is bypassed and the exact gate always runs.
    let outer_precheck_eligible = include_logdet_h
        && total > 0
        && family.joint_jeffreys_information_matches_observed_hessian();
    let outer_jeffreys_precheck_skips = match preferred_workspace.as_ref() {
        Some(ws) if outer_precheck_eligible && ws.hessian_matvec_available() => {
            let hv = |v: &Array1<f64>| -> Result<Array1<f64>, String> {
                match ws.hessian_matvec(v)? {
                    Some(out) if out.len() == total => Ok(out),
                    // Workspace declined this matvec ⇒ cannot certify ⇒ do not skip.
                    // Return a non-finite sentinel so the cheap estimator bails to
                    // the conservative `false` (never skip on an unresolved apply).
                    _ => Ok(Array1::from_elem(total, f64::NAN)),
                }
            };
            if total >= crate::estimate::reml::jeffreys_subspace::CHEAP_CONDITIONING_PRECHECK_MIN_DIM
            {
                // Wide joint system: bound the spectrum from a few matvecs (no dense
                // H, no O(p³) eigh).
                crate::estimate::reml::jeffreys_subspace::jeffreys_term_skippable_via_matvec(
                    hv, total,
                )
                .unwrap_or(false)
            } else {
                // Small joint system: the dense p×p eigh is itself cheap, so form H
                // once via `total` matvecs and run the EXACT conditioning gate — the
                // same #1389 fix as the inner-Newton skip, applied to the OUTER LAML
                // logdet H_Φ so the per-outer-eval Jeffreys all-axes sweep is also
                // skipped on a well-conditioned small fit (the constant-scale
                // survival location-scale non-termination paid this term on every
                // outer eval as well as every inner cycle). A non-finite matvec (the
                // declined-apply sentinel) is treated as "cannot certify ⇒ run the
                // exact term", preserving the conservative never-skip-on-unresolved
                // contract.
                (|| -> Result<bool, String> {
                    let mut h = Array2::<f64>::zeros((total, total));
                    let mut e_a = Array1::<f64>::zeros(total);
                    for a in 0..total {
                        e_a[a] = 1.0;
                        let col = hv(&e_a)?;
                        e_a[a] = 0.0;
                        if col.len() != total || col.iter().any(|v| !v.is_finite()) {
                            return Ok(false);
                        }
                        for r in 0..total {
                            h[[r, a]] = col[r];
                        }
                    }
                    crate::estimate::reml::jeffreys_subspace::jeffreys_term_skippable_dense(h.view())
                })()
                .unwrap_or(false)
            }
        }
        _ => false,
    };
    let logdet_jeffreys_hphi: Option<Array2<f64>> = if include_logdet_h
        && !outer_jeffreys_precheck_skips
        && !options.seed_screening
        && family.joint_jeffreys_term_required()
    {
        // Skipped during seed screening: this per-axis Jeffreys curvature
        // (O(p · per-axis-Hdot)) augments the outer LAML logdet `½ log|H+Sλ+H_Φ|`,
        // a refinement the screening SCORE does not need. Screening ranks seeds by
        // the un-augmented `½ log|H+Sλ|` plus the value-only Firth penalty already
        // in `penalty_value`; the load-bearing H_Φ is restored for the real fit
        // (gam#729/#808).
        match build_joint_jeffreys_subspace(specs, &ranges)? {
            Some(z_joint) => {
                custom_family_joint_jeffreys_term(family, states, specs, &ranges, &z_joint)?
                    .map(|(_phi, _grad, hphi)| hphi)
            }
            None => None,
        }
    } else {
        None
    };
    let compute_block_logdet_term = |b: usize| -> Result<(Array2<f64>, f64), String> {
        let spec = &specs[b];
        let (start, end) = ranges[b];
        let p = end - start;
        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s.add_scaled_to(lambdas[k], &mut s_lambda);
        }
        let block_logdet = if include_logdet_s {
            // Pseudo-logdet of S_λ on the positive eigenspace.
            //
            // CONSISTENCY REQUIREMENT (gam#752/#748/#808 class): this VALUE is
            // the `log|S_λ|₊` term of the outer REML/LAML objective, and its
            // ρ-gradient is supplied separately by
            // `compute_block_penalty_logdet_derivs`, which differentiates the
            // canonical `PenaltyPseudologdet`. If the value used a *different*
            // positive/null eigenspace split (e.g. structural-count `skip(m0)`
            // by COUNT, or a ridge-blind `positive_eigenvalue_threshold`) than
            // the gradient's by-magnitude `> ridge + noise_band` rule, the
            // outer optimizer would see an objective and a gradient that
            // describe different functions near the ridge boundary (a barely-
            // active mode `λ_k σ_k → 0` whose ridged eigenvalue dips below
            // `ridge + noise_band` is kept by the count rule but dropped by the
            // magnitude rule). To guarantee value↔gradient agree by
            // construction, compute the value from the SAME canonical
            // `PenaltyPseudologdet` the gradient differentiates, with the same
            // dense penalty components, the same λ, and the same ridge.
            let ridge = if options.ridge_policy.include_penalty_logdet {
                effective_solverridge(options.ridge_floor)
            } else {
                0.0
            };
            let penalties_dense: Vec<Array2<f64>> =
                spec.penalties.iter().map(|pen| pen.to_dense()).collect();
            let lambdas_vec: Vec<f64> = lambdas.to_vec();
            match crate::estimate::reml::penalty_logdet::PenaltyPseudologdet::from_components(
                &penalties_dense,
                &lambdas_vec,
                ridge,
            ) {
                Ok(pld) => pld.value(),
                Err(eigh_err_msg) => {
                    // `from_components` only fails when the single internal
                    // eigendecomposition fails, which for PSD penalties is
                    // purely numerical. Fall back to Cholesky on the ridged
                    // matrix (which should be SPD). The Cholesky logdet
                    // includes null-space contributions (~m₀ × ln(ridge)),
                    // a smooth bias that does not corrupt the REML gradient.
                    let mut s_for_logdet = s_lambda.clone();
                    if ridge > 0.0 {
                        for i in 0..p {
                            s_for_logdet[[i, i]] += ridge;
                        }
                    }
                    penalty_logdet_cholesky_fallback(&s_for_logdet, ridge, b, p, &eigh_err_msg)?
                }
            }
        } else {
            0.0
        };
        Ok((s_lambda, block_logdet))
    };

    // Per-block penalty assembly and eigendecomposition are independent.
    // Use rayon only from non-rayon callers so inner operator/eigendecomp work
    // does not nest under an existing worker. Collecting an indexed range into
    // a Vec preserves block order; totals are accumulated sequentially below
    // to keep floating-point summation deterministic.
    let block_terms: Vec<Result<(Array2<f64>, f64), String>> =
        if specs.len() > 1 && rayon::current_thread_index().is_none() {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..specs.len())
                .into_par_iter()
                .map(compute_block_logdet_term)
                .collect()
        } else {
            (0..specs.len()).map(compute_block_logdet_term).collect()
        };
    let mut s_lambdas = Vec::with_capacity(block_terms.len());
    let mut penalty_logdet_s_total = 0.0;
    for block_term in block_terms {
        let (s_lambda, block_logdet) = block_term?;
        s_lambdas.push(s_lambda);
        penalty_logdet_s_total += block_logdet;
    }
    if !include_logdet_h {
        return Ok((0.0, penalty_logdet_s_total));
    }
    // Try the shared scale-aware exact curvature path first.
    if let Some(curvature) = family.exact_newton_outer_curvature(states)? {
        let mut h_joint = symmetrized_square_matrix(
            curvature.hessian,
            total,
            "joint exact-newton Hessian validation in logdet terms (rescaled)",
        )?;
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            h_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(curvature.rho_curvature_scale, s_lambda);
        }
        if let Some(hphi) = logdet_jeffreys_hphi.as_ref() {
            h_joint.scaled_add(curvature.rho_curvature_scale, hphi);
        }
        let logdet_h_scaled = if strict_spd {
            strict_exact_pseudo_logdet(&h_joint, joint_observation_count(states))?
        } else {
            stable_logdet_with_ridge_policy(
                &h_joint,
                options.ridge_floor * curvature.rho_curvature_scale,
                options.ridge_policy,
            )?
        };
        let logdet_h_total = logdet_h_scaled + curvature.hessian_logdet_correction;
        return Ok((logdet_h_total, penalty_logdet_s_total));
    }
    let exact_joint_source = if let Some(workspace) = preferred_workspace.as_ref() {
        exact_newton_joint_hessian_source_from_workspace(
            workspace,
            total,
            MaterializationIntent::LogdetFactorization,
            "joint exact-newton operator mismatch in logdet terms",
        )?
    } else if !strict_spd && use_joint_matrix_free_path(total, joint_observation_count(states)) {
        family
            .exact_newton_joint_hessian_workspace_with_options(states, specs, options)?
            .as_ref()
            .map(|workspace| {
                exact_newton_joint_hessian_source_from_workspace(
                    workspace,
                    total,
                    MaterializationIntent::LogdetFactorization,
                    "joint exact-newton operator mismatch in logdet terms",
                )
            })
            .transpose()?
            .flatten()
    } else {
        None
    };
    if let Some(source) = exact_joint_source {
        // Exact determinant of H + S_λ for operator-backed coefficient Hessians.
        //
        // The REML gradient and Hessian use analytic trace identities such as
        // ∂ log|A(θ)| = tr(A⁻¹ A_θ).  Mixing an approximate determinant with
        // exact traces violates that identity and gives ARC a Hessian for a
        // different objective.  Materializing the coefficient Hessian by
        // canonical-basis HVPs keeps the objective/derivative pair exact.  At
        // large-scale CTN scale `total` is a few hundred, so this is sub-MiB; the
        // materializer below refuses oversized systems before allocation.
        let mut h_joint = materialize_joint_hessian_source(
            &source,
            total,
            "joint exact-newton operator dense logdet materialization",
        )?;
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            h_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(1.0, s_lambda);
        }
        if let Some(hphi) = logdet_jeffreys_hphi.as_ref() {
            h_joint.scaled_add(1.0, hphi);
        }
        let logdet_h_total = if strict_spd {
            strict_exact_pseudo_logdet(&h_joint, joint_observation_count(states))?
        } else {
            stable_logdet_with_ridge_policy(&h_joint, options.ridge_floor, options.ridge_policy)?
        };
        return Ok((logdet_h_total, penalty_logdet_s_total));
    }
    // Fallback: try the non-rescaled symmetrized path (for families that
    // don't implement exact_newton_outer_curvature but do provide
    // a plain joint Hessian).
    if let Some(mut h_joint) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
        total,
        "joint exact-newton Hessian validation in logdet terms",
    )? {
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            h_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(1.0, s_lambda);
        }
        if let Some(hphi) = logdet_jeffreys_hphi.as_ref() {
            h_joint.scaled_add(1.0, hphi);
        }
        let logdet_h_total = if strict_spd {
            strict_exact_pseudo_logdet(&h_joint, joint_observation_count(states))?
        } else {
            stable_logdet_with_ridge_policy(&h_joint, options.ridge_floor, options.ridge_policy)?
        };
        return Ok((logdet_h_total, penalty_logdet_s_total));
    }

    let eval = family.evaluate(states)?;
    if eval.blockworking_sets.len() != specs.len() {
        return Err(format!(
            "family returned {} block working sets, expected {}",
            eval.blockworking_sets.len(),
            specs.len()
        ));
    }

    let mut logdet_h_total = 0.0;
    let logdet_s_total = penalty_logdet_s_total;
    for b in 0..specs.len() {
        let spec = &specs[b];
        let work = &eval.blockworking_sets[b];
        let p = spec.design.ncols();
        let xtwx = match work {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => with_block_geometry(family, states, spec, b, |x_dyn, _| {
                let w = floor_positiveworking_weights(working_weights, options.minweight);
                let (xtwx, _) = weighted_normal_equations(x_dyn, &w, None)?;
                Ok(xtwx)
            })?,
            BlockWorkingSet::ExactNewton {
                gradient: _,
                hessian,
            } => {
                if hessian.nrows() != p || hessian.ncols() != p {
                    return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                        "block {b} exact-newton Hessian shape mismatch: got {}x{}, expected {}x{}",
                        hessian.nrows(),
                        hessian.ncols(),
                        p,
                        p
                    ) }.into());
                }
                hessian.to_dense()
            }
        };

        let s_lambda = &s_lambdas[b];

        let mut h = xtwx;
        h += s_lambda;
        logdet_h_total += if strict_spd {
            strict_exact_pseudo_logdet(&h, joint_observation_count(states))?
        } else {
            stable_logdet_with_ridge_policy(&h, options.ridge_floor, options.ridge_policy)?
        };
    }
    Ok((logdet_h_total, logdet_s_total))
}

/// Snapshot of a single block's eta for line-search rollback.
///
/// Created from a specific block's state; can only restore to or update
/// that same block.  There is no shared buffer across blocks, so
/// cross-block length confusion is structurally impossible.
pub(crate) struct BlockEtaCheckpoint {
    pub(crate) saved: Array1<f64>,
}

impl BlockEtaCheckpoint {
    /// Capture the current eta of `state`.
    pub(crate) fn capture(state: &ParameterBlockState) -> Self {
        Self {
            saved: state.eta.clone(),
        }
    }

    /// Capture into a pre-allocated buffer, returning the filled checkpoint.
    /// The buffer is taken (O(1) move) and filled with eta's data (O(n) copy).
    pub(crate) fn capture_reuse(state: &ParameterBlockState, buf: &mut Array1<f64>) -> Self {
        if buf.len() == state.eta.len() {
            buf.assign(&state.eta);
            Self {
                saved: std::mem::take(buf),
            }
        } else {
            Self::capture(state)
        }
    }

    /// Return the internal buffer for recycling.
    pub(crate) fn into_buffer(self) -> Array1<f64> {
        self.saved
    }

    /// Restore: `state.eta = saved`.
    pub(crate) fn restore_eta(&self, state: &mut ParameterBlockState) {
        state.eta.assign(&self.saved);
    }

    /// Incremental update: `state.eta = saved + alpha * direction`.
    pub(crate) fn restore_eta_with_step(
        &self,
        state: &mut ParameterBlockState,
        alpha: f64,
        direction: &Array1<f64>,
    ) {
        // In-place: eta = eta_backup + alpha * xd (zero allocations).
        state.eta.assign(&self.saved);
        state.eta.scaled_add(alpha, direction);
    }
}

/// Classification of which branch of the trust-region radius policy
/// fired on a single update — surfaced in cycle logs so it is possible
/// to tell at a glance whether the inner solver is being throttled by
/// the TR (e.g., `RejectFloor`/`ShrinkOnRejection`) or, conversely,
/// whether the step is sitting well inside the region (`HoldInside`)
/// so the slow convergence is NOT a TR-policy issue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JointTrustRegionDecision {
    /// `rho > 0.75` AND `step_norm >= 0.99 * old_radius` — model is good
    /// AND the step is at the TR boundary, so doubling reflects a real
    /// constraint that was just lifted.
    GrowAtBoundary,
    /// `rho > 0.75` but the step is well inside the region; radius held
    /// because no evidence the TR was constraining the step.  When the
    /// inner is converging linearly and this branch fires every cycle,
    /// the TR is NOT the bottleneck — Newton itself is finding short
    /// steps for a reason unrelated to the trust radius.
    HoldInside,
    /// `0.25 <= rho <= 0.75` (moderate model fidelity) — radius held.
    HoldModerate,
    /// `rho < 0.25` but step accepted (positive descent above noise).
    /// Radius shrunk to a quarter to be more conservative next cycle.
    ShrinkOnMarginalAccept,
    /// Step rejected — radius shrunk and capped at half the proposed
    /// step norm so a re-proposal is constrained inside the rejected
    /// region.
    ShrinkOnRejection,
    /// Radius was already at the floor before this update.  Persistent
    /// `RejectFloor` is the unambiguous signal of a degenerate ρ region.
    RejectFloor,
}

impl JointTrustRegionDecision {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::GrowAtBoundary => "grow_at_boundary",
            Self::HoldInside => "hold_inside",
            Self::HoldModerate => "hold_moderate",
            Self::ShrinkOnMarginalAccept => "shrink_marginal_accept",
            Self::ShrinkOnRejection => "shrink_reject",
            Self::RejectFloor => "reject_floor",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct JointTrustRegionUpdate {
    pub(crate) rho: f64,
    pub(crate) radius: f64,
    pub(crate) accepted: bool,
    pub(crate) decision: JointTrustRegionDecision,
}

pub(crate) fn update_joint_trust_region_radius(
    old_radius: f64,
    step_norm: f64,
    actual_reduction: f64,
    predicted_reduction: f64,
    objective_scale: f64,
) -> JointTrustRegionUpdate {
    // Floating-point noise floor relative to the objective magnitude.
    // When both the model-predicted and the realized reductions are at
    // this scale, their sign is dominated by round-off in the
    // log-likelihood evaluation rather than by genuine descent or
    // ascent; rejecting on that sign would discard a perfectly
    // converged step. Mirrors the noise-floor handling in
    // src/solver/pirls.rs (see the analogous `noise_floor` block).
    let noise_floor = objective_scale.abs().max(1.0) * 1e-14;
    let predicted_finite_positive =
        predicted_reduction > noise_floor && predicted_reduction.is_finite();
    let rho = if actual_reduction.abs() <= noise_floor {
        // The realized objective change is at the floating-point round-off floor:
        // the step neither helped nor hurt beyond noise, so it is a numerically
        // neutral (converged) step. Treat it as `rho = 1` REGARDLESS of whether
        // `predicted_reduction` happens to sit just above the floor. The previous
        // form only took this branch when `!predicted_finite_positive`; when a
        // tiny-but-valid Newton step near a flat-objective optimum produced
        // `predicted_reduction` marginally above `noise_floor` while
        // `actual_reduction` was within it, it divided two round-off-level
        // quantities and got a spurious negative `rho`, rejecting the step and
        // ratcheting the trust radius to the floor — pinning the solve far below
        // the (small, valid) Newton step it needed (gam#797 last-mile; same
        // pinning family observed on clustered bernoulli). Keying neutrality on the
        // *actual* reduction is the correct round-off guard.
        1.0
    } else if predicted_finite_positive {
        actual_reduction / predicted_reduction
    } else {
        f64::NEG_INFINITY
    };
    let accepted = rho.is_finite() && rho > 0.0 && actual_reduction >= -noise_floor;
    let mut radius = old_radius;
    let decision: JointTrustRegionDecision;
    if !accepted {
        radius *= 0.25;
        if step_norm.is_finite() && step_norm > 0.0 {
            radius = radius.min(0.5 * step_norm);
        }
        decision = JointTrustRegionDecision::ShrinkOnRejection;
    } else if rho < 0.25 {
        radius *= 0.25;
        decision = JointTrustRegionDecision::ShrinkOnMarginalAccept;
    } else if rho > 0.75 && step_norm >= 0.99 * old_radius {
        radius *= 2.0;
        decision = JointTrustRegionDecision::GrowAtBoundary;
    } else if rho > 0.75 {
        decision = JointTrustRegionDecision::HoldInside;
    } else {
        decision = JointTrustRegionDecision::HoldModerate;
    }
    if !radius.is_finite() || radius <= 0.0 {
        radius = 1.0e-12;
    }
    let clamped_radius = radius.clamp(1.0e-12, 1.0e6);
    // Promote to RejectFloor if we landed at the absolute floor.  The
    // base classification is preserved up to this final clamp; the
    // floor classification is just a stronger label that captures the
    // "no descent direction exists at this radius" signal.
    let final_decision = if clamped_radius <= 1.0e-12 + f64::EPSILON
        && matches!(
            decision,
            JointTrustRegionDecision::ShrinkOnRejection
                | JointTrustRegionDecision::ShrinkOnMarginalAccept
        ) {
        JointTrustRegionDecision::RejectFloor
    } else {
        decision
    };
    JointTrustRegionUpdate {
        rho,
        radius: clamped_radius,
        accepted,
        decision: final_decision,
    }
}

pub(crate) fn joint_objective_roundoff_slack(old_objective: f64, trial_objective: f64) -> f64 {
    (64.0 * f64::EPSILON * (1.0 + old_objective.abs() + trial_objective.abs())).max(1.0e-10)
}

// True iff the line search detected a noise-level realized reduction (i.e.
// the trial step neither helped nor hurt the objective beyond round-off)
// AND the local quadratic model agrees that no further descent is available
// within tolerance. `actual_reduction <= 0` is kept (not made sign-symmetric)
// because at rank-deficient optima (σ_min(H) ≲ ε_machine) the outer-gradient
// FD identity requires β trajectories to be CONSISTENT across λ probes —
// accepting positive-noise-level reductions exits the loop one attempt
// earlier than the negative case and decorrelates the null-space drift
// between consecutive REML evaluations. Concretely:
// `outer_lamlgradient_matches_finite_differencewhen_joint_exact_path_is_active`
// at HardPseudo σ_min ~ 1e-10 fails when symmetric. The asymmetric guard
// preserves the spin avoidance for the common (negative-noise) case at
// large scale while leaving the rank-deficient FD identity intact.
pub(crate) fn joint_objective_floor_reached(
    old_objective: f64,
    trial_objective: f64,
    actual_reduction: f64,
    predicted_reduction: f64,
    objective_tol: f64,
) -> bool {
    trial_objective.is_finite()
        && actual_reduction <= 0.0
        && actual_reduction.abs() <= joint_objective_roundoff_slack(old_objective, trial_objective)
        && predicted_reduction.is_finite()
        && predicted_reduction
            <= objective_tol.max(joint_objective_roundoff_slack(
                old_objective,
                trial_objective,
            ))
}

/// True iff the joint-Newton proposal is already at the step-tolerance floor —
/// the unclamped Newton step's inf-norm is within `STEP_FLOOR_CERT_FACTOR ×
/// step_tol` (the same round-off band the constrained-stationary certificate
/// uses for "a hair above tol"). At the floor the iterate is doing KKT polishing
/// on a flat objective, so a `predicted_reduction = rhs·δ − ½δᵀHδ ≤ 0` is the
/// SIGN of two near-equal O(step²) quantities (round-off), NOT a model-invalid
/// descent direction; the preconditioned-descent substitution must be suppressed
/// there or it replaces the tiny polishing step with an objective-descent step
/// that catapults the KKT residual off the near-converged iterate (gam#787 binary
/// matern centers=12: residual 1.7e-4 → 4.7e-1, never recovers).
pub(crate) fn joint_proposal_at_step_floor(proposal_step_inf: f64, step_tol: f64) -> bool {
    const STEP_FLOOR_CERT_FACTOR: f64 = 4.0;
    proposal_step_inf.is_finite()
        && step_tol.is_finite()
        && proposal_step_inf <= STEP_FLOOR_CERT_FACTOR * step_tol
}

/// The absolute trust-radius floor `update_joint_trust_region_radius` clamps to
/// (`radius.clamp(1.0e-12, 1.0e6)`), with a small multiplicative slack to absorb
/// the promote-to-`RejectFloor` boundary test there.
pub(crate) const JOINT_COLLAPSED_FLOOR_RADIUS_CEIL: f64 = 1.0e-12 * (1.0 + 1e-9);

/// Number of consecutive all-reject-at-floor cycles after which the coupled
/// joint-Newton loop is declared stuck-but-stationary and exited cleanly (gam#979).
pub(crate) const JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES: usize = 4;

/// Collapsed-trust-region all-reject-at-floor guard (gam#979).
///
/// Returns `true` iff the joint trust radius is finite and pinned at its
/// absolute `1e-12` floor (no smaller step is representable, so it cannot shrink
/// further) AND a fully-rejected cycle was just observed AND that pattern has now
/// persisted for `JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES` consecutive cycles.
/// When all three hold the inner loop is provably grinding to its budget on a
/// near-singular coupled marginal↔logslope system (the survival-hang root cause):
/// the step makes no progress and the radius cannot adapt, so every further cycle
/// reproduces this one. The caller exits through the existing identified-subspace /
/// fixed-point certificate path (converged if stationary, give-best otherwise).
///
/// This is a pure CONTROL-FLOW predicate — it changes no numerics. It cannot fire
/// on a genuinely-progressing fit: a descending solve keeps the radius well above
/// the floor (it grows on `rho>0.75`/boundary), so `at_absolute_floor` is false and
/// the consecutive counter is reset to zero on every non-floor or accepted cycle.
pub(crate) fn joint_collapsed_floor_all_reject_exit(
    consecutive_all_reject_at_floor_cycles: usize,
    all_attempts_rejected_at_floor_this_cycle: bool,
) -> bool {
    all_attempts_rejected_at_floor_this_cycle
        && consecutive_all_reject_at_floor_cycles >= JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES
}

/// True iff the joint trust radius has reached its absolute `1e-12` floor and can
/// shrink no further (gam#979). Used to gate the collapsed-floor all-reject streak.
pub(crate) fn joint_trust_radius_at_absolute_floor(joint_trust_radius: f64) -> bool {
    joint_trust_radius.is_finite() && joint_trust_radius <= JOINT_COLLAPSED_FLOOR_RADIUS_CEIL
}

pub(crate) fn joint_trust_region_metric_step_norm(
    delta: &Array1<f64>,
    metric_diag: &Array1<f64>,
) -> f64 {
    assert_eq!(delta.len(), metric_diag.len());
    joint_trust_region_metric_step_norm_view(delta.view(), metric_diag.view())
}

pub(crate) fn joint_trust_region_metric_step_norm_view(
    delta: ArrayView1<f64>,
    metric_diag: ArrayView1<f64>,
) -> f64 {
    assert_eq!(delta.len(), metric_diag.len());
    delta
        .iter()
        .zip(metric_diag.iter())
        .map(|(step, weight)| step * step * positive_joint_diagonal_entry(*weight))
        .sum::<f64>()
        .sqrt()
}

pub(crate) fn joint_trust_region_block_metric_norms(
    delta: &Array1<f64>,
    ranges: &[(usize, usize)],
    metric_diag: &Array1<f64>,
) -> Vec<f64> {
    assert_eq!(delta.len(), metric_diag.len());
    ranges
        .iter()
        .map(|(start, end)| {
            joint_trust_region_metric_step_norm_view(
                delta.slice(s![*start..*end]),
                metric_diag.slice(s![*start..*end]),
            )
        })
        .collect()
}

pub(crate) fn truncate_joint_step_to_block_metric_radii(
    delta: &mut Array1<f64>,
    ranges: &[(usize, usize)],
    metric_diag: &Array1<f64>,
    block_radii: &[f64],
) -> Vec<f64> {
    assert_eq!(ranges.len(), block_radii.len());
    assert_eq!(delta.len(), metric_diag.len());
    let mut norms = Vec::with_capacity(ranges.len());
    for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
        let metric_view = metric_diag.slice(s![start..end]);
        let mut block = delta.slice_mut(s![start..end]);
        let norm = joint_trust_region_metric_step_norm_view(block.view(), metric_view);
        let radius = block_radii[block_idx];
        if norm.is_finite() && norm > radius && radius > 0.0 {
            block.mapv_inplace(|v| v * (radius / norm));
            norms.push(radius);
        } else {
            norms.push(norm);
        }
    }
    norms
}

pub(crate) fn joint_block_step_hit_trust_boundary(step_norm: f64, radius: f64) -> bool {
    step_norm.is_finite() && radius > 0.0 && step_norm >= 0.99 * radius
}

/// Per-block dogleg step (Powell, blending the Cauchy and Newton points within
/// the block's M-metric trust radius). This is the principled globalization for
/// the coupled location-scale inner Newton (gam#826/#808): box-truncating the
/// Newton step alone freezes progress when the spectral solve is degenerate at
/// the oversmoothed seed — the high-curvature `log_sigma` block has
/// `λ ~ exp(2·ρ_bound)` so its Newton component is `O(g/λ) ≈ 5e-21`, the
/// mean/trend blocks get isotropically shrunk to the radius, and the residual
/// stalls while β barely moves. The dogleg always includes the Cauchy leg
/// (the model-minimizing steepest-descent step in the block metric), so the
/// realized decrease is at least the Cauchy decrease whenever the block
/// gradient is nonzero — progress is guaranteed even when the Newton step is
/// numerically frozen. Inside the radius the dogleg returns the exact Newton
/// step, so the converged β, the KKT certificate, and the well-conditioned /
/// #729 endgame are byte-identical to the undamped solve.
///
/// Inputs per block `b`:
///   * `newton[start..end]`  — Newton (spectral) step block `δ_N`.
///   * `cauchy[start..end]`  — the FULL (unconstrained) Cauchy block
///     `δ_C = τ·p_sd`, where `p_sd = M⁻¹·rhs` is the M-metric steepest-descent
///     direction of the model and `τ` minimizes the model along it; precomputed
///     once per cycle by `joint_cauchy_step` (the curvature `p_sd·H·p_sd` needs
///     a coupled Hessian-vector product, so it must be hoisted out of the
///     radius-shrink loop).
///   * `radius`              — the block's current M-metric trust radius.
///
/// Returns the block step norms in the M-metric (same contract as
/// `truncate_joint_step_to_block_metric_radii`) and overwrites `out` with the
/// dogleg blend per block.
pub(crate) fn joint_dogleg_step_to_block_metric_radii(
    newton: &Array1<f64>,
    cauchy: &Array1<f64>,
    ranges: &[(usize, usize)],
    metric_diag: &Array1<f64>,
    block_radii: &[f64],
    out: &mut Array1<f64>,
) -> Vec<f64> {
    assert_eq!(ranges.len(), block_radii.len());
    assert_eq!(newton.len(), metric_diag.len());
    assert_eq!(cauchy.len(), metric_diag.len());
    assert_eq!(out.len(), metric_diag.len());
    let mut norms = Vec::with_capacity(ranges.len());
    for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
        let metric_view = metric_diag.slice(s![start..end]);
        let newton_b = newton.slice(s![start..end]);
        let cauchy_b = cauchy.slice(s![start..end]);
        let radius = block_radii[block_idx];
        let newton_norm = joint_trust_region_metric_step_norm_view(newton_b, metric_view);
        let cauchy_norm = joint_trust_region_metric_step_norm_view(cauchy_b, metric_view);
        let mut out_b = out.slice_mut(s![start..end]);

        // Degenerate radius (non-finite or non-positive): nothing moves.
        if !radius.is_finite() || radius <= 0.0 {
            out_b.fill(0.0);
            norms.push(0.0);
            continue;
        }

        // Newton step (or a non-finite Cauchy fallback) inside the radius: take
        // the exact Newton step. This is the only branch a well-conditioned /
        // converging fit ever reaches near the optimum, so the endgame numerics
        // are unchanged.
        if newton_norm.is_finite() && newton_norm <= radius {
            out_b.assign(&newton_b);
            norms.push(newton_norm);
            continue;
        }

        // Cauchy leg longer than the radius (or Newton/Cauchy not comparable):
        // scale the Cauchy step to the boundary. When the Cauchy step itself is
        // unusable, fall back to scaling the Newton step (pre-dogleg behavior).
        if !(cauchy_norm.is_finite() && cauchy_norm > 0.0) {
            let scale = if newton_norm.is_finite() && newton_norm > 0.0 {
                radius / newton_norm
            } else {
                0.0
            };
            out_b.assign(&newton_b);
            out_b.mapv_inplace(|v| v * scale);
            norms.push(if scale > 0.0 { radius } else { 0.0 });
            continue;
        }
        if cauchy_norm >= radius {
            let scale = radius / cauchy_norm;
            out_b.assign(&cauchy_b);
            out_b.mapv_inplace(|v| v * scale);
            norms.push(radius);
            continue;
        }

        // Dogleg blend: δ(θ) = δ_C + θ·(δ_N − δ_C), θ ∈ [0,1], pick θ so
        // ‖δ(θ)‖_M = radius. Solve the quadratic ‖δ_C + θ·d‖²_M = radius² with
        // d = δ_N − δ_C, a = ‖d‖²_M, b = 2·⟨δ_C, d⟩_M, c = ‖δ_C‖²_M − radius².
        let mut a = 0.0_f64;
        let mut b = 0.0_f64;
        for ((cb, nb), w) in cauchy_b.iter().zip(newton_b.iter()).zip(metric_view.iter()) {
            let m = positive_joint_diagonal_entry(*w);
            let d = nb - cb;
            a += m * d * d;
            b += 2.0 * m * cb * d;
        }
        let c = cauchy_norm * cauchy_norm - radius * radius;
        // a > 0 because δ_N ≠ δ_C here (Newton outside, Cauchy inside the
        // radius). Largest root in [0,1] keeps the step on the dogleg path.
        let disc = (b * b - 4.0 * a * c).max(0.0);
        let theta = if a > 0.0 {
            ((-b + disc.sqrt()) / (2.0 * a)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        for ((o, cb), nb) in out_b.iter_mut().zip(cauchy_b.iter()).zip(newton_b.iter()) {
            *o = cb + theta * (nb - cb);
        }
        let norm = joint_trust_region_metric_step_norm_view(out_b.view(), metric_view);
        norms.push(norm);
    }
    norms
}

/// Unconstrained Cauchy point of the joint penalized quadratic model in the
/// block-diagonal M-metric: `δ_C = τ·p_sd` with `p_sd = M⁻¹·rhs` (the M-metric
/// steepest-descent direction of the model `m(δ) = −rhs·δ + ½·δᵀHδ` at δ=0)
/// and `τ = (rhs·p_sd)/(p_sd·H·p_sd)` minimizing the model along `p_sd`. When
/// the curvature `p_sd·H·p_sd ≤ 0` the model is unbounded below along `p_sd`,
/// so `δ_C` is just `p_sd` (the dogleg's boundary scaling then takes it to the
/// radius — a descent step on the indefinite/flat direction). `h_psd` must be
/// `H_pen·p_sd` for the SAME penalized (and Firth-augmented, when armed) Hessian
/// the trust-region model uses, so the dogleg path is consistent with the
/// accept/reject quadratic.
pub(crate) fn joint_cauchy_step(
    rhs: &Array1<f64>,
    p_sd: &Array1<f64>,
    h_psd: &Array1<f64>,
) -> Array1<f64> {
    let directional = rhs.dot(p_sd);
    if !directional.is_finite() || directional <= 0.0 {
        // `p_sd` is not an ascent direction of −m (no descent on the objective);
        // emit a zero Cauchy step so the dogleg falls back to the Newton leg.
        return Array1::zeros(p_sd.len());
    }
    let curvature = p_sd.dot(h_psd);
    let mut delta = p_sd.clone();
    if curvature.is_finite() && curvature > 0.0 {
        let tau = directional / curvature;
        if tau.is_finite() && tau > 0.0 {
            delta.mapv_inplace(|v| tau * v);
        }
    }
    // Non-positive curvature: leave δ_C = p_sd; the dogleg scales it to the
    // trust boundary (the model decreases without bound along p_sd there).
    delta
}

pub(crate) fn shrink_active_joint_block_trust_radii(
    block_radii: &mut [f64],
    block_step_norms: &[f64],
    factor: f64,
) -> f64 {
    assert_eq!(block_radii.len(), block_step_norms.len());
    // Joint-Newton step-rejection radius shrink. Must guarantee strict
    // monotone decrease of `max(block_radii)` until the floor, otherwise the
    // next trust-region attempt computes a step byte-identical to the rejected
    // one and the inner loop stalls forever (gam joint-Newton fully-rejected
    // cycles, root cause behind the 8-cycle bail at FULLY_REJECTED_STALL_MAX_CYCLES).
    //
    // Two cooperating mechanisms:
    //   * For every block that participates in the shrink, the new radius is
    //     pulled below the rejected step's magnitude (`0.5 · step_norm`),
    //     matching the analogous clamp in `update_joint_trust_region_radius`'s
    //     reject branch. This forces the next step to be strictly smaller
    //     than the current one even when `radius * factor` is still larger
    //     than `step_norm` (which happens whenever the dogleg/truncate path
    //     returned a Newton step shorter than the block's radius).
    //   * Block participation: by default only shrink blocks whose step hit
    //     the per-block trust boundary (the boundary block was the one the
    //     trust radius actually constrained — interior blocks took their
    //     natural Newton step and shrinking their radius is wasted). BUT when
    //     every boundary block already sits at the 1e-12 floor, further
    //     shrinking those blocks is a no-op (they'd just re-clamp to the
    //     floor), so we *must* shrink the interior blocks instead to actually
    //     change the joint step. Without this carve-out the deadlock was:
    //     boundary block pinned at 1e-12, interior block radius held at its
    //     pre-stall value, `max(block_radii)` held by the interior block, the
    //     dogleg/truncate produces an identical joint δ every cycle, every
    //     trust attempt rejects on the same objective check, the cycle burns
    //     to `inner_loop_hard_ceiling` (1200) cycles wasting ~120 s per
    //     outer ρ-evaluation — the Rust CI Test hang and the
    //     `rust_margslope_aniso_duchon16d_*` large-scale 2400 s timeout.
    const RADIUS_FLOOR: f64 = 1.0e-12;
    let any_boundary_block = block_radii
        .iter()
        .zip(block_step_norms)
        .any(|(radius, step_norm)| joint_block_step_hit_trust_boundary(*step_norm, *radius));
    let all_boundary_blocks_at_floor = any_boundary_block
        && block_radii
            .iter()
            .zip(block_step_norms)
            .filter(|(radius, step_norm)| {
                joint_block_step_hit_trust_boundary(**step_norm, **radius)
            })
            .all(|(radius, _)| *radius <= RADIUS_FLOOR * (1.0 + 1.0e-12));
    // Snapshot the joint max BEFORE the shrink loop so the max-holding
    // block(s) — boundary OR interior — always participate. The
    // Moré–Sorensen inner step uses the SCALAR
    // `joint_trust_radius = max(block_radii)` as its trust constraint
    // (`spectrum.trust_region_step(joint_trust_radius)`); if the max-holder
    // is an interior block whose step came in well below its per-block
    // radius, the original boundary-only rule left the joint max held, the
    // MS solve re-computed a byte-identical rejected step, and the inner
    // loop stalled at `inner_loop_hard_ceiling`. Surfaced as the ~2-hour
    // Rust CI test hang where cycles 117..305+ all logged
    // `r=1.562e-2 (held) decision=shrink_reject |δ|=1.562e-2 |δ|∞=1.052e-4`
    // identically: the boundary block's per-block radius collapsed toward
    // the floor without ever changing the scalar joint radius, and the
    // existing `all_boundary_blocks_at_floor` carve-out was unreachable
    // because the boundary block kept getting deeper than the interior
    // max-holder. Forcing the max-holder to participate makes the scalar
    // `max(block_radii)` strictly decrease on every rejected attempt until
    // the floor, after which the fully-rejected stall guard
    // (`FULLY_REJECTED_STALL_MAX_CYCLES`) takes over and bails the cycle
    // cleanly.
    let max_radius_before = block_radii.iter().copied().fold(0.0_f64, f64::max);
    for (radius, step_norm) in block_radii.iter_mut().zip(block_step_norms) {
        let at_boundary = joint_block_step_hit_trust_boundary(*step_norm, *radius);
        let holds_max = max_radius_before > 0.0
            && max_radius_before.is_finite()
            && *radius >= max_radius_before * (1.0 - 1.0e-12);
        let participates = if all_boundary_blocks_at_floor {
            // Boundary-at-floor stall: the boundary blocks cannot shrink any
            // further, so participate every block (including interior ones)
            // so the joint step magnitude actually changes.
            true
        } else if any_boundary_block {
            at_boundary || holds_max
        } else {
            true
        };
        if participates {
            let mut new_radius = *radius * factor;
            if step_norm.is_finite() && *step_norm > 0.0 {
                new_radius = new_radius.min(0.5 * *step_norm);
            }
            *radius = new_radius.clamp(RADIUS_FLOOR, 1.0e6);
        }
    }
    block_radii.iter().copied().fold(0.0_f64, f64::max)
}

pub(crate) fn apply_joint_feasibility_limit<F: CustomFamily + ?Sized>(
    family: &F,
    states: &[ParameterBlockState],
    ranges: &[(usize, usize)],
    trial_delta: &mut Array1<f64>,
) -> Result<bool, String> {
    // Collect each block's feasibility α and apply the *minimum* to the
    // JOINT trial step, not to each block in isolation.
    //
    // The joint Newton direction δ̂ = H⁻¹(−g) is the unique descent direction
    // for the local quadratic model up to a positive scalar; any α·δ̂ with
    // α ∈ (0, 1] is still a descent direction on the joint objective.
    // Scaling ONLY one block by α produces (α·δ̂_A, δ̂_B, …), which is
    // neither δ̂ nor α·δ̂ and is not, in general, a descent direction on
    // the joint quadratic.
    //
    // Production survival_marginal_slope failure mode at large scale:
    // the time block returned α ≈ 1e-4 (monotonicity guard); per-block
    // scaling crushed δ_time to ~2.3e-4 while logslope kept its full
    // unconstrained Newton step. The joint step was no longer a Newton
    // direction; the time-block gradient stayed at ‖g_time‖ ≈ 5.6e8 for
    // the next 15+ cycles, triggering the linearized-rate stall
    // early-exit on every outer seed.
    //
    // Scaling the joint step by min α preserves Newton direction; the
    // trust-region/line-search already chooses the appropriate step size
    // within direction, this barrier check just enforces feasibility on
    // top of that direction.
    let (joint_alpha, limiting_block) =
        compute_joint_feasibility_alpha(family, states, ranges, trial_delta)?;
    if joint_alpha < 1.0 {
        trial_delta.mapv_inplace(|v| joint_alpha * v);
        log::debug!(
            "[PIRLS/joint-Newton] feasibility scaled joint step by α={:.3e} (block {:?} binding)",
            joint_alpha,
            limiting_block,
        );
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Compute the joint fraction-to-boundary feasibility scalar `α ∈ (0, 1]` that
/// [`apply_joint_feasibility_limit`] would apply to `trial_delta`, WITHOUT
/// mutating the step. Returns `(α, limiting_block)`. `α = 1.0` means the step is
/// fully feasible (no binding constraint); `α < 1.0` is the min over blocks of
/// each block's `max_feasible_step_size`. Returns `Err` iff a block has no
/// positive feasible step (current iterate infeasible / degenerate).
///
/// Split out (gam#979) so the constrained joint-Newton path can DETECT the
/// pathological α-collapse — a binding monotonicity row at slack≈0 driving
/// `α → 0`, which globally crushes the whole joint step and freezes β — and
/// reroute feasibility through the magnitude-preserving cone projection in that
/// case ONLY, while keeping the exact existing α-scaling behaviour whenever `α`
/// is healthy. Every currently-converging arm sees byte-identical numerics off
/// the pathology.
pub(crate) fn compute_joint_feasibility_alpha<F: CustomFamily + ?Sized>(
    family: &F,
    states: &[ParameterBlockState],
    ranges: &[(usize, usize)],
    trial_delta: &Array1<f64>,
) -> Result<(f64, Option<usize>), String> {
    let mut joint_alpha = 1.0_f64;
    let mut limiting_block: Option<usize> = None;
    for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
        let block_delta = trial_delta.slice(s![start..end]).to_owned();
        if let Some(alpha_max) = family.max_feasible_step_size(states, block_idx, &block_delta)? {
            if !alpha_max.is_finite() || alpha_max <= 0.0 {
                return Err(format!(
                    "joint Newton block {block_idx} has no positive feasible step"
                ));
            }
            if alpha_max < joint_alpha {
                joint_alpha = alpha_max;
                limiting_block = Some(block_idx);
            }
        }
    }
    Ok((joint_alpha, limiting_block))
}

/// Below this joint fraction-to-boundary `α`, scaling the WHOLE joint step by
/// `α` crushes it so severely that β is effectively frozen for the cycle (the
/// gam#979 survival hang: `α ≈ 1e-4` on a binding monotone time-derivative row
/// reduced the step to noise while a huge time-block gradient persisted). At or
/// above it the step retains enough magnitude that the existing α-scaling is a
/// healthy globalization, so the constrained path keeps the legacy behaviour
/// unchanged. The threshold is deliberately small (`1e-2`): a step scaled by
/// `≥ 1%` still makes real progress, whereas the pathology drives `α` five-plus
/// orders below it.
pub(crate) const JOINT_FEASIBILITY_ALPHA_CRUSH_THRESHOLD: f64 = 1.0e-2;

pub(crate) fn joint_inner_kkt_converged(residual: f64, residual_tol: f64) -> bool {
    residual.is_finite() && residual_tol.is_finite() && residual <= residual_tol
}

/// Per-iterate diagnostic snapshot assembled when the joint Newton inner solve
/// refuses to certify constrained-stationarity. The report breaks the failure
/// down by block (so the offending smooth can be named), records the H_pen
/// eigenvalue spectrum (so rank-deficiency in the penalized Hessian is
/// detectable from logs), and classifies the refusal so downstream tooling
/// can act without re-deriving the cert math.
#[derive(Clone, Debug)]
pub(crate) struct KktRefusalReport {
    pub(crate) block_names: Vec<String>,
    pub(crate) block_widths: Vec<usize>,
    pub(crate) block_beta_inf: Vec<f64>,
    pub(crate) block_grad_inf: Vec<f64>,
    pub(crate) block_penalty_grad_inf: Vec<f64>,
    pub(crate) block_residual_inf: Vec<f64>,
    pub(crate) block_carrying_residual: Option<usize>,

    pub(crate) hpen_eigenvalues_sorted_desc: Vec<f64>,
    pub(crate) hpen_min_abs_eigenvalue: f64,
    pub(crate) hpen_max_abs_eigenvalue: f64,
    pub(crate) hpen_condition_number: f64,
    pub(crate) hpen_nullity_at_rank_tol: usize,
    pub(crate) hpen_rank_tol: f64,
    pub(crate) hpen_null_gradient_inf: f64,
    pub(crate) hpen_null_vector_block_inf: Vec<f64>,
    pub(crate) hpen_null_vector_carrying_block: Option<usize>,

    pub(crate) active_set_rows_total: usize,
    pub(crate) accepted_step_inf: f64,
    pub(crate) proposal_step_inf: f64,
    pub(crate) trust_radius: f64,
    pub(crate) cycle: usize,

    pub(crate) residual_tol: f64,
    pub(crate) obj_tol: f64,
    pub(crate) step_tol: f64,

    pub(crate) linearized_rel: f64,
    pub(crate) scalar_model_relerr: f64,
    pub(crate) objective_change: f64,
    pub(crate) projected_residual_inf: f64,

    pub(crate) diagnosis: KktRefusalDiagnosis,
}

// `KktRefusalDiagnosis` was relocated DOWN to `gam_problem::diagnostics`
// (issue #1521 crate carve): it is the neutral diagnostic carrier the
// gam-solve REML/PIRLS core consumes, so it must live below both the core
// and the extracted custom-family subsystem. Re-exported here under the
// historical module path so existing references keep resolving.
pub(crate) use gam_problem::diagnostics::KktRefusalDiagnosis;

/// Relative rank tolerance applied to `|λ|/λ_max` when counting the
/// nullity of `H_pen`. Matches the threshold the surrounding REML
/// penalty-rank machinery uses for "structurally zero".
pub(crate) const KKT_REFUSAL_RANK_TOL: f64 = 1e-10;

/// Residual band (as a multiple of the KKT residual tolerance) inside which
/// the inner joint Newton is considered to be in its convergence ENDGAME and
/// the exact Jeffreys second-order completion is added to the step model
/// (gam#979). Far from the mode the trust region globalizes any model, so
/// exactness buys nothing there; in the endgame it converts the
/// divided-difference model's linear sawtooth into quadratic convergence.
pub(crate) const JEFFREYS_COMPLETION_RESIDUAL_BAND: f64 = 300.0;

/// Self-vanishing Levenberg–Marquardt damping factor for the range-restricted
/// spectral Newton step (`solve_joint_newton_step_on_spectral_range`). The
/// caller forms the residual-scaled magnitude
/// `μ = JOINT_SPECTRAL_LEVENBERG_FACTOR · ‖∇L − Sβ‖∞`, which the solve converts
/// to a DIMENSIONLESS, scale-invariant Marquardt damping `ν = μ / λ_max` applied
/// MULTIPLICATIVELY to each range curvature (`curvature·(1 + ν)`), not added
/// (`curvature + μ`). The multiplicative form is essential on a coupled
/// location-scale joint Hessian whose spectrum spans the penalty scale
/// (`λ ~ e²⁴` at the oversmoothed seed) and the likelihood scale (the
/// mean/wiggle XᵀWX curvature): an ADDITIVE μ — set by the penalty-inflated
/// residual — swamps the small likelihood curvature and freezes that block
/// (#826), whereas the multiplicative `1/(1+ν)` throttle is identical across all
/// scales so no block stalls. Both forms cap the unbounded `component/λ` step
/// along near-singular (ill-conditioned but above-`KKT_REFUSAL_RANK_TOL`)
/// eigen-directions — the modes that make the undamped step oscillate — and both
/// vanish as the iterate converges (`ν → 0`), recovering the exact Moore–Penrose
/// Newton step so the KKT fixed point and the well-identified fast path are
/// unchanged. `1e-3` keeps the damping two to three orders below the dominant
/// curvature on a well-conditioned problem.
pub(crate) const JOINT_SPECTRAL_LEVENBERG_FACTOR: f64 = 1.0e-3;

/// Condition number above which a FULL-RANK (`nullity == 0`) penalized Hessian is
/// treated as ill-conditioned enough that a family opting into
/// [`CustomFamily::levenberg_on_ill_conditioning`] gets the self-vanishing
/// constrained-QP Levenberg floor (gam#1040). Below it the active-set QP minimiser
/// is well-determined and the EXACT undamped Newton/KKT solve keeps its quadratic
/// convergence; the survival marginal-slope joint sits at cond ≈ 5.8e6, far above
/// this gate, while a tiny well-conditioned constrained AFT sits well below it.
pub(crate) const LEVENBERG_ILL_CONDITIONING_THRESHOLD: f64 = 1.0e4;

#[derive(Clone, Debug)]
pub(crate) struct JointSpectralNewtonStep {
    pub(crate) delta: Array1<f64>,
    pub(crate) range_rhs_inf: f64,
    pub(crate) null_rhs_inf: f64,
    pub(crate) lambda_max_abs: f64,
    pub(crate) lambda_min_positive: f64,
    pub(crate) nullity: usize,
    pub(crate) rank_tol: f64,
    /// Number of eigen-directions whose curvature was negative (beyond the
    /// rank cutoff) and was reflected to `|λ|` to form a modified-Newton
    /// descent step. Zero for a genuinely positive-semidefinite model.
    pub(crate) reflected_negative_modes: usize,
    /// Most negative eigenvalue encountered (≤ 0); `0.0` when the model was
    /// positive-semidefinite within the rank cutoff.
    pub(crate) most_negative_eigenvalue: f64,
}

/// Production home for the exact trust-region engine ([`WhitenedHessianSpectrum`]),
/// wired into the unconstrained dense-spectral joint-Newton step in
/// `inner_blockwise_fit` (gam#979). Kept in its own module so the engine's
/// helpers stay namespaced; the parent reaches it via `whitened_spectrum::`.
pub(crate) mod whitened_spectrum {
    use super::super::*;

    /// Eigendecomposition of the metric-whitened penalized Hessian, retained so
    /// every trust-radius shrink within one Newton cycle re-solves the
    /// trust-region subproblem from the SAME `O(p³)` factorization at `O(p)` cost.
    ///
    /// # Why this exists (gam#979)
    ///
    /// The coupled marginal↔logslope inner Newton needs ONE globalization, not a
    /// stack of approximations. Historically the joint step was a *modified-Newton*
    /// (reflect indefinite eigenvalues to `|λ|`) wrapped in a *heuristically gated*
    /// multiplicative Marquardt damping (engaged on `nullity>0`, or condition number
    /// over a threshold, or after N non-improving cycles) and then a *dogleg* between
    /// that step and the Cauchy point, truncated to per-block step-norm trust radii.
    /// Each piece approximates a different facet of the one exact object below, and
    /// each had to be gated so it would not re-break the case another piece was added
    /// for (#826 vs #808 vs #733/#734 vs #787). When none of the gates matched the
    /// operating point — well-conditioned `H_pen`, yet a coupled near-aliased
    /// direction with a huge raw Newton component — the truncated direction made only
    /// Cauchy-sized progress, the gain ratio never justified growing the radius, and
    /// the residual crawled for hundreds of cycles (the #979 "phantom multiplier"
    /// grind / survival hang).
    ///
    /// [`Self::trust_region_step`] replaces all of that with the *exact* solution of
    /// the trust-region subproblem
    ///   minimize  `−rhsᵀδ + ½ δᵀ H_pen δ`   subject to  `‖δ‖_D ≤ r`,
    /// via the Moré–Sorensen characterization: the minimizer is `δ(λ) = (H_pen +
    /// λD)⁻¹ rhs` for the unique `λ ≥ max(0, −γ_min)` with `‖δ(λ)‖_D = r` (or `λ = 0`
    /// when the Newton step is interior and `H_pen ≻ 0`). Working in the `D`-metric
    /// generalized eigenbasis this is a scalar secular equation in `λ`, solved by a
    /// safeguarded Newton iteration on the already-computed spectrum. Properties that
    /// make it the right object:
    ///   * indefiniteness is handled exactly (`λ ≥ −γ_min` makes `H_pen+λD ⪰ 0` on
    ///     the boundary — no reflection heuristic, no negative-curvature special case
    ///     other than the rigorous hard case);
    ///   * the damping `λ` is determined by the trust radius, not by nullity /
    ///     condition / stall gates — those gates disappear;
    ///   * it self-vanishes: at the KKT fixed point `rhs → 0 ⇒ δ → 0`, and once the
    ///     iterate is in a region where `H_pen ≻ 0` the Newton step goes interior so
    ///     `λ = 0` and convergence is quadratic — the converged β, the KKT
    ///     certificate, and the REML/LAML the residual feeds are byte-identical to an
    ///     undamped exact-Newton solve;
    ///   * it is affine covariant in the `D` metric, so blocks at wildly different
    ///     curvature scales (the penalty `λ ~ e²⁴` modes vs the `XᵀWX` likelihood
    ///     modes at an oversmoothed seed) are damped uniformly by `1/(γ_k+λ)` — the
    ///     scale-invariance the per-block radii and the multiplicative-Marquardt form
    ///     were each hand-built to approximate.
    ///
    /// The genuine penalty null space (`|γ_k| ≤ null_cutoff`) is still projected out
    /// (the gam#553 Moore–Penrose range restriction): an unidentified gauge direction
    /// carries no finite Newton step and is left unchanged, its stationarity-residual
    /// component reported via [`JointSpectralNewtonStep::null_rhs_inf`].
    pub(crate) struct WhitenedHessianSpectrum {
        /// Generalized eigenvalues `γ_k` of `(H_pen, D)` = eigenvalues of the
        /// whitened matrix `A = D^{-1/2} H_pen D^{-1/2}`.
        pub(crate) gamma: Array1<f64>,
        /// Whitened eigenvectors `v_k` (columns) of `A`.
        pub(crate) evecs: Array2<f64>,
        /// rhs in the whitened eigenbasis: `c_k = v_kᵀ D^{-1/2} rhs`.
        pub(crate) c: Array1<f64>,
        /// `D^{-1/2}` diagonal, mapping a whitened step `η` back to `δ = D^{-1/2} η`.
        pub(crate) d_inv_sqrt: Array1<f64>,
        /// `max_k |γ_k|` (the curvature scale; `D`-whitened).
        pub(crate) lambda_max_abs: f64,
        /// Curvature magnitude at/below which a direction is treated as genuinely
        /// unidentified (penalty null space) and dropped from the step.
        pub(crate) null_cutoff: f64,
    }

    impl WhitenedHessianSpectrum {
        /// Eigendecompose the `D`-whitened penalized Hessian once. `metric_diag`
        /// supplies the positive trust-region metric `D` (each entry is passed
        /// through [`positive_joint_diagonal_entry`] so a non-positive curvature
        /// estimate becomes a safe positive scale). `rank_tol` is the relative
        /// near-singularity cutoff; the genuine numerical-rank floor is derived from
        /// the whitened spectrum exactly as the legacy spectral solve did.
        pub(crate) fn decompose(
            h_pen: &Array2<f64>,
            rhs: &Array1<f64>,
            metric_diag: &Array1<f64>,
            rank_tol: f64,
        ) -> Result<Self, String> {
            let p = h_pen.nrows();
            if h_pen.ncols() != p || rhs.len() != p || metric_diag.len() != p {
                return Err(format!(
                    "whitened trust-region decomposition dimension mismatch: H={}x{}, rhs={}, metric={}",
                    h_pen.nrows(),
                    h_pen.ncols(),
                    rhs.len(),
                    metric_diag.len()
                ));
            }
            let d_inv_sqrt = Array1::from_iter(
                metric_diag
                    .iter()
                    .map(|w| 1.0 / positive_joint_diagonal_entry(*w).sqrt()),
            );
            // A = D^{-1/2} H D^{-1/2}; symmetric since H is symmetric and D
            // diagonal. This runs once per joint-Newton cycle on a ~p²-element
            // matrix (p up to ~382 on the coupled multinomial / survival
            // marginal-slope inner, gam#1082), so it is on the per-cycle hot
            // path alongside the O(p³) eigh below. The prior code was a serial
            // `for i,j in 0..p` double loop followed by a full
            // `symmetrize_dense_in_place` sweep. Two math-identical wins:
            //   1. `A` is exactly symmetric because `H` is symmetric and `D`
            //      diagonal, so row `i` of `A` equals `d_inv_sqrt[i] · (h_pen
            //      row i) ⊙ d_inv_sqrt`. Filling each row from that closed form
            //      yields a fully symmetric matrix with NO separate
            //      `symmetrize_dense_in_place` pass (drops a whole O(p²) sweep
            //      while keeping both triangles consistent, which the rare
            //      `FaerEigh::eigh` repair branch — it averages `[i,j]`/`[j,i]`
            //      — relies on).
            //   2. Rows are independent and own disjoint output slices, so we
            //      fan the fill across the Rayon pool (the same pool faer's eigh
            //      uses) via `outer_iter_mut().into_par_iter()`. No unsafe, no
            //      cross-row writes. The eigh dominates, but this assembly was
            //      otherwise a purely serial bounds-checked scalar loop.
            let mut a = Array2::<f64>::zeros((p, p));
            {
                use rayon::iter::{
                    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
                };
                let d = d_inv_sqrt.as_slice().expect("contiguous d_inv_sqrt");
                a.outer_iter_mut()
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut a_row)| {
                        let di = d[i];
                        let h_row = h_pen.row(i);
                        let out = a_row.as_slice_mut().expect("contiguous A row");
                        for (j, out_j) in out.iter_mut().enumerate() {
                            *out_j = h_row[j] * di * d[j];
                        }
                    });
            }
            let (gamma, evecs) = FaerEigh::eigh(&a, Side::Lower)
                .map_err(|e| format!("whitened trust-region eigendecomposition failed: {e}"))?;
            // c = Vᵀ (D^{-1/2} rhs).
            let whitened_rhs = &d_inv_sqrt * rhs;
            let c = evecs.t().dot(&whitened_rhs);
            let lambda_max_abs = gamma.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let numerical_floor = lambda_max_abs * (p as f64).sqrt() * f64::EPSILON;
            let rank_cutoff = rank_tol * lambda_max_abs;
            let null_cutoff = rank_cutoff.max(numerical_floor);
            Ok(Self {
                gamma,
                evecs,
                c,
                d_inv_sqrt,
                lambda_max_abs,
                null_cutoff,
            })
        }

        /// `‖η(λ)‖²_2 = Σ_{identified k} c_k² / (γ_k + λ)²` — the squared `D`-metric
        /// norm of the trial step as a function of the Levenberg shift `λ`. Only
        /// identified (above-`null_cutoff`) modes participate; the null space carries
        /// no step.
        pub(crate) fn step_norm_sq(&self, lambda: f64) -> f64 {
            let mut acc = 0.0;
            for k in 0..self.gamma.len() {
                if self.gamma[k].abs() <= self.null_cutoff {
                    continue;
                }
                let denom = self.gamma[k] + lambda;
                if denom.abs() <= f64::MIN_POSITIVE {
                    return f64::INFINITY;
                }
                let t = self.c[k] / denom;
                acc += t * t;
            }
            acc
        }

        /// Predicted objective decrease of the *unconstrained* (infinite trust
        /// radius) modified-Newton step on the quadratic model — the Newton
        /// decrement `½ Σ_{identified k} c_k² / |γ_k|`.
        ///
        /// For the exact Newton step `δ = Σ c_k/γ_k v_k` (range modes only) the
        /// model decrease `m(0) − m(δ) = rhsᵀδ − ½δᵀH_pen δ` evaluates to
        /// `½ Σ c_k²/γ_k`; reflecting negative-curvature modes to `|γ_k|` (the
        /// modified-Newton descent step `trust_region_step` takes) makes every
        /// term a genuine decrease, so the unconstrained model can reduce the
        /// objective by AT MOST this amount. The null space (`|γ_k| ≤
        /// null_cutoff`) carries no step and no decrease and is skipped — its
        /// residual mass is a free gauge direction the outer IFT projects out
        /// (gam#553).
        ///
        /// This is the curvature-aware convergence quantity the coupled
        /// joint-Newton needs on a weakly-identified (near-flat) carrying block
        /// (survival marginal↔logslope, link-wiggle, location-scale — gam#1040 /
        /// gam#1088): a large penalized stationarity residual `‖∇L − Sβ‖∞` along
        /// a low-curvature direction (`g` large, `γ` tiny) gives an enormous raw
        /// Newton step that the trust region clamps, so the residual- and
        /// step-norm gates never close and the loop grinds to the cycle ceiling
        /// — yet the *achievable* objective improvement `g²/(2γ)` may be far
        /// below `objective_tol`. When it is, the iterate IS the penalized
        /// optimum to within tolerance (Conn–Gould–Toint, *Trust-Region
        /// Methods*, Thm 6.4.6): no step the model can resolve lowers the
        /// objective by more than `objective_tol`, so continuing is wall-clock
        /// waste. A genuine defect (real curvature AND large gradient) produces
        /// a LARGE decrement, so this never masks one.
        pub(crate) fn newton_decrement(&self) -> f64 {
            let mut acc = 0.0;
            for k in 0..self.gamma.len() {
                let abs_gamma = self.gamma[k].abs();
                if abs_gamma <= self.null_cutoff {
                    continue;
                }
                acc += self.c[k] * self.c[k] / abs_gamma;
            }
            0.5 * acc
        }

        /// Infinity-norm of the whitened RHS mass on the SKIPPED near-null modes
        /// (`|γ_k| ≤ null_cutoff`) — exactly the residual the Newton-decrement
        /// certificate excludes as free-gauge mass that the outer IFT projects
        /// out. Surfaced at the convergence certificate so the question "did the
        /// stopping rule discard a weakly-identified REAL direction?" is
        /// observable rather than silent: the certificate is sound only when this
        /// excluded mass genuinely lies on penalty-null gauge directions, so a
        /// certificate firing with a LARGE `null_residual_inf` is the diagnostic
        /// witness of the ill-conditioning edge (a real curved+signal mode
        /// mis-classified as null because `null_cutoff = rank_tol·λ_max` grew with
        /// the spectrum). It never participates in the stopping decision — it only
        /// makes the decision auditable.
        pub(crate) fn null_residual_inf(&self) -> f64 {
            let mut m = 0.0_f64;
            for k in 0..self.gamma.len() {
                if self.gamma[k].abs() <= self.null_cutoff {
                    m = m.max(self.c[k].abs());
                }
            }
            m
        }

        /// Achievable objective improvement `½ Σ c_k²/|γ_k|` over modes the raw
        /// [`newton_decrement`] EXCLUDES but that carry GENUINE (above
        /// machine-noise) curvature — the weakly-identified band
        /// `numerical_floor < |γ_k| ≤ null_cutoff` (gam#1449).
        ///
        /// [`newton_decrement`] drops every `|γ_k| ≤ null_cutoff` mode as gauge
        /// null, where `null_cutoff = max(rank_tol·λ_max, numerical_floor)`. That
        /// is correct for the numerical-rank null space (`|γ_k| ≤
        /// numerical_floor = λ_max·√p·ε`): such a mode has no resolvable curvature,
        /// so `c_k²/|γ_k|` is meaningless machine noise and the outer IFT projects
        /// it out. But on a BADLY-SCALED penalized Hessian `rank_tol·λ_max` can
        /// grow large enough to also swallow a mode with small-but-REAL curvature
        /// (`numerical_floor < |γ_k| ≤ rank_tol·λ_max`) AND real signal `c_k` — a
        /// genuinely weakly-identified direction, not gauge. Its achievable
        /// improvement `c_k²/(2|γ_k|)` is real and may exceed `objective_tol`, so
        /// silently excluding it from the convergence decision would certify a
        /// non-converged iterate.
        ///
        /// This sums that real-but-excluded improvement using the
        /// CONDITIONING-ROBUST floor (`numerical_floor`, the machine-rank cutoff)
        /// rather than `rank_tol·λ_max`. The decrement certificate fires only when
        /// it is ALSO `≤ objective_tol`, so a weakly-identified real mode BLOCKS
        /// premature certification while the genuine numerical null space (below
        /// `numerical_floor`) still contributes nothing. The STEP is unchanged
        /// (it keeps dropping the whole `≤ null_cutoff` band); only the stopping
        /// test is hardened.
        pub(crate) fn weakly_identified_decrement(&self) -> f64 {
            let p = self.gamma.len();
            // Reconstruct the genuine numerical-rank floor used by `decompose`.
            let numerical_floor = self.lambda_max_abs * (p as f64).sqrt() * f64::EPSILON;
            let mut acc = 0.0_f64;
            for k in 0..p {
                let abs_gamma = self.gamma[k].abs();
                if abs_gamma > numerical_floor && abs_gamma <= self.null_cutoff {
                    acc += self.c[k] * self.c[k] / abs_gamma;
                }
            }
            0.5 * acc
        }

        /// Assemble the whitened step `η(λ) = Σ c_k/(γ_k+λ) v_k` over identified
        /// modes and map it back to `δ = D^{-1/2} η`. Returns `(δ, range_rhs_inf,
        /// null_rhs_inf, nullity, lambda_min_positive, reflected_negative_modes,
        /// most_negative)` diagnostics consistent with the legacy spectral step.
        pub(crate) fn assemble(
            &self,
            lambda: f64,
            extra_min_mode: Option<(usize, f64)>,
        ) -> JointSpectralNewtonStep {
            let p = self.gamma.len();
            let mut eta = Array1::<f64>::zeros(p);
            let mut range_rhs_inf = 0.0_f64;
            let mut null_rhs_inf = 0.0_f64;
            let mut lambda_min_positive = f64::INFINITY;
            let mut nullity = 0usize;
            let mut reflected_negative_modes = 0usize;
            let mut most_negative = 0.0_f64;
            for k in 0..p {
                let g = self.gamma[k];
                if g.abs() <= self.null_cutoff {
                    nullity += 1;
                    null_rhs_inf = null_rhs_inf.max(self.c[k].abs());
                    continue;
                }
                range_rhs_inf = range_rhs_inf.max(self.c[k].abs());
                if g < 0.0 {
                    reflected_negative_modes += 1;
                    most_negative = most_negative.min(g);
                } else {
                    lambda_min_positive = lambda_min_positive.min(g);
                }
                let denom = g + lambda;
                if denom.abs() > f64::MIN_POSITIVE {
                    let coeff = self.c[k] / denom;
                    for i in 0..p {
                        eta[i] += coeff * self.evecs[[i, k]];
                    }
                }
            }
            // Hard case: add τ·v_min along a minimal-curvature eigenvector to reach
            // the trust boundary when rhs has no component there.
            if let Some((k_min, tau)) = extra_min_mode {
                for i in 0..p {
                    eta[i] += tau * self.evecs[[i, k_min]];
                }
            }
            // δ = D^{-1/2} η.
            let delta = &self.d_inv_sqrt * &eta;
            JointSpectralNewtonStep {
                delta,
                range_rhs_inf,
                null_rhs_inf,
                lambda_max_abs: self.lambda_max_abs,
                lambda_min_positive,
                nullity,
                rank_tol: KKT_REFUSAL_RANK_TOL,
                reflected_negative_modes,
                most_negative_eigenvalue: most_negative,
            }
        }

        /// Exact solution of the trust-region subproblem inside the `D`-metric ball
        /// of radius `trust_radius`. When `trust_radius` is non-finite or `≤ 0` the
        /// unconstrained (Moore–Penrose, range-restricted) Newton step is returned —
        /// i.e. the caller opted out of the trust region.
        pub(crate) fn trust_region_step(&self, trust_radius: f64) -> JointSpectralNewtonStep {
            // Smallest identified curvature (signed). Empty identified set ⇒ pure
            // null space ⇒ zero step.
            let mut gamma_min_id = f64::INFINITY;
            let mut any_identified = false;
            for k in 0..self.gamma.len() {
                if self.gamma[k].abs() <= self.null_cutoff {
                    continue;
                }
                any_identified = true;
                gamma_min_id = gamma_min_id.min(self.gamma[k]);
            }
            if !any_identified {
                return self.assemble(0.0, None);
            }

            let unconstrained_radius = !(trust_radius.is_finite() && trust_radius > 0.0);
            // Interior Newton step is admissible only when the model is convex on the
            // identified range (γ_min > 0); then λ = 0 gives the exact Newton step.
            if gamma_min_id > 0.0 {
                let newton_norm = self.step_norm_sq(0.0).sqrt();
                if unconstrained_radius || newton_norm <= trust_radius {
                    return self.assemble(0.0, None);
                }
            } else if unconstrained_radius {
                // No trust region but an indefinite/semidefinite model: the
                // unconstrained problem is unbounded below. Fall back to the
                // reflected modified-Newton step (|γ| curvature) so the caller still
                // receives a finite descent direction; the downstream accept/reject
                // validates it. This path is only hit when a caller explicitly
                // disables the trust region on an indefinite model.
                return self.assemble_reflected();
            }

            // Boundary solution: find λ ≥ λ_lo with ‖η(λ)‖ = trust_radius.
            let lambda_lo = (-gamma_min_id).max(0.0);
            // Hard case detection: is rhs orthogonal to the minimal-curvature
            // eigenspace? If so ‖η(λ_lo)‖ is finite and may be below the radius.
            let min_mode_tol = self.null_cutoff.max(self.lambda_max_abs * 1e-12);
            let mut hard_case_component_sq = 0.0;
            let mut k_min_witness = None;
            for k in 0..self.gamma.len() {
                if self.gamma[k].abs() <= self.null_cutoff {
                    continue;
                }
                if (self.gamma[k] - gamma_min_id).abs() <= min_mode_tol {
                    hard_case_component_sq += self.c[k] * self.c[k];
                    k_min_witness = Some(k);
                }
            }
            // Evaluate the norm just above the pole. With a real rhs component at the
            // minimal mode the norm diverges at λ_lo, so the secular root is interior
            // to (λ_lo, ∞) and a small relative offset brackets it. With no such
            // component (hard case) the norm at λ_lo is finite.
            let lambda_lo_eval = lambda_lo + self.lambda_max_abs.max(1.0) * 1e-12;
            if hard_case_component_sq <= (self.lambda_max_abs.max(1.0) * 1e-12).powi(2) {
                let norm_at_lo = self.step_norm_sq(lambda_lo_eval).sqrt();
                if norm_at_lo < trust_radius {
                    // Hard case: λ = λ_lo, then add τ·v_min to reach the boundary.
                    if let Some(k_min) = k_min_witness {
                        let deficit =
                            (trust_radius * trust_radius - norm_at_lo * norm_at_lo).max(0.0);
                        let tau = deficit.sqrt();
                        return self.assemble(lambda_lo, Some((k_min, tau)));
                    }
                    return self.assemble(lambda_lo, None);
                }
            }
            // Safeguarded Newton on φ(λ) = 1/‖η(λ)‖ − 1/r (well-behaved, ~linear),
            // bracketed in [lo, hi]. φ is increasing in λ (‖η‖ decreasing), φ(lo)<0,
            // and we grow hi until φ(hi)>0.
            let target = trust_radius;
            let mut lo = lambda_lo_eval;
            let mut hi = lambda_lo_eval.max(self.lambda_max_abs).max(1.0);
            let mut grow_guard = 0;
            while self.step_norm_sq(hi).sqrt() > target && grow_guard < 200 {
                hi *= 2.0;
                grow_guard += 1;
            }
            let mut lambda = 0.5 * (lo + hi);
            for _ in 0..100 {
                let q = self.step_norm_sq(lambda);
                let norm = q.sqrt();
                if !norm.is_finite() {
                    lo = lambda;
                    lambda = 0.5 * (lo + hi);
                    continue;
                }
                // Maintain the bracket on φ(λ) = 1/norm − 1/target.
                if norm > target {
                    lo = lambda;
                } else {
                    hi = lambda;
                }
                let phi = 1.0 / norm - 1.0 / target;
                if phi.abs() <= 1e-12 / target {
                    break;
                }
                // q'(λ) = -2 Σ c_k²/(γ_k+λ)³ ⇒ d/dλ (1/norm) = -½ q^{-3/2} q'.
                let mut q_prime = 0.0;
                for k in 0..self.gamma.len() {
                    if self.gamma[k].abs() <= self.null_cutoff {
                        continue;
                    }
                    let denom = self.gamma[k] + lambda;
                    if denom.abs() <= f64::MIN_POSITIVE {
                        continue;
                    }
                    q_prime += -2.0 * self.c[k] * self.c[k] / (denom * denom * denom);
                }
                let phi_prime = -0.5 * q.powf(-1.5) * q_prime;
                let next = if phi_prime.abs() > f64::MIN_POSITIVE {
                    lambda - phi / phi_prime
                } else {
                    0.5 * (lo + hi)
                };
                // Safeguard into the bracket.
                lambda = if next.is_finite() && next > lo && next < hi {
                    next
                } else {
                    0.5 * (lo + hi)
                };
                if (hi - lo) <= 1e-14 * (1.0 + hi.abs()) {
                    break;
                }
            }
            self.assemble(lambda, None)
        }

        /// Reflected modified-Newton step (`|γ_k|` curvature, no trust region). Only
        /// used when a caller disables the trust region on an indefinite model — the
        /// trust-region path proper never reflects.
        pub(crate) fn assemble_reflected(&self) -> JointSpectralNewtonStep {
            let p = self.gamma.len();
            let mut eta = Array1::<f64>::zeros(p);
            let mut range_rhs_inf = 0.0_f64;
            let mut null_rhs_inf = 0.0_f64;
            let mut lambda_min_positive = f64::INFINITY;
            let mut nullity = 0usize;
            let mut reflected_negative_modes = 0usize;
            let mut most_negative = 0.0_f64;
            for k in 0..p {
                let g = self.gamma[k];
                if g.abs() <= self.null_cutoff {
                    nullity += 1;
                    null_rhs_inf = null_rhs_inf.max(self.c[k].abs());
                    continue;
                }
                range_rhs_inf = range_rhs_inf.max(self.c[k].abs());
                let curvature = if g < 0.0 {
                    reflected_negative_modes += 1;
                    most_negative = most_negative.min(g);
                    g.abs()
                } else {
                    lambda_min_positive = lambda_min_positive.min(g);
                    g
                };
                let coeff = self.c[k] / curvature;
                for i in 0..p {
                    eta[i] += coeff * self.evecs[[i, k]];
                }
            }
            let delta = &self.d_inv_sqrt * &eta;
            JointSpectralNewtonStep {
                delta,
                range_rhs_inf,
                null_rhs_inf,
                lambda_max_abs: self.lambda_max_abs,
                lambda_min_positive,
                nullity,
                rank_tol: KKT_REFUSAL_RANK_TOL,
                reflected_negative_modes,
                most_negative_eigenvalue: most_negative,
            }
        }
    }
}

#[cfg(test)]
mod trust_region_subproblem_tests {
    use super::super::*;
    use super::whitened_spectrum::WhitenedHessianSpectrum;
    use ndarray::array;

    pub(crate) fn metric_norm(delta: &Array1<f64>, d: &Array1<f64>) -> f64 {
        delta
            .iter()
            .zip(d.iter())
            .map(|(x, w)| x * x * positive_joint_diagonal_entry(*w))
            .sum::<f64>()
            .sqrt()
    }

    /// Interior case: a positive-definite model with a generous trust radius
    /// must return the exact (full) Newton step `H⁻¹ rhs`, i.e. λ = 0.
    #[test]
    pub(crate) fn interior_returns_exact_newton_step() {
        let h = array![[3.0, 1.0], [1.0, 2.0]];
        let rhs = array![1.0, -2.0];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let step = spec.trust_region_step(1e6);
        // Exact Newton: H δ = rhs.
        let resid = h.dot(&step.delta) - &rhs;
        assert!(
            resid.iter().all(|v| v.abs() < 1e-10),
            "interior step must solve H δ = rhs exactly, residual {resid:?}"
        );
    }

    /// Boundary case: a tight radius forces `‖δ‖_D = r` and the KKT condition
    /// `(H + λD) δ = rhs` with `λ > 0`.
    #[test]
    pub(crate) fn boundary_satisfies_more_sorensen_kkt() {
        let h = array![[3.0, 1.0], [1.0, 2.0]];
        let rhs = array![1.0, -2.0];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let r = 0.3;
        let step = spec.trust_region_step(r);
        let norm = metric_norm(&step.delta, &d);
        assert!(
            (norm - r).abs() < 1e-8,
            "boundary step must lie on the trust boundary: ‖δ‖_D={norm} vs r={r}"
        );
        // Recover λ from one coordinate of (H+λD)δ = rhs and check the whole
        // system is satisfied at that λ.
        let hd = h.dot(&step.delta);
        // Solve λ minimizing ‖(H+λD)δ − rhs‖ in least squares over the single
        // scalar λ: λ* = (Dδ)·(rhs − Hδ) / (Dδ)·(Dδ).
        let dd = &d * &step.delta;
        let lam = dd.dot(&(&rhs - &hd)) / dd.dot(&dd);
        assert!(lam > 0.0, "boundary multiplier must be positive, got {lam}");
        let resid = &hd + &(lam * &dd) - &rhs;
        assert!(
            resid.iter().all(|v| v.abs() < 1e-7),
            "(H+λD)δ = rhs must hold at the recovered λ={lam}, residual {resid:?}"
        );
    }

    /// Indefinite model: the exact subproblem still returns a finite boundary
    /// step that is a descent direction (rhsᵀδ > 0) and lies on the boundary.
    #[test]
    pub(crate) fn indefinite_model_returns_descent_step_on_boundary() {
        // Eigenvalues +4 and -1: genuinely indefinite.
        let h = array![[1.5, 2.5], [2.5, 1.5]];
        let rhs = array![1.0, 0.4];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let r = 0.7;
        let step = spec.trust_region_step(r);
        assert!(step.reflected_negative_modes >= 1 || step.most_negative_eigenvalue < 0.0);
        let norm = metric_norm(&step.delta, &d);
        assert!(
            (norm - r).abs() < 1e-7,
            "indefinite boundary step ‖δ‖_D={norm} vs r={r}"
        );
        assert!(
            rhs.dot(&step.delta) > 0.0,
            "step must be a descent direction for −rhsᵀδ + ½δᵀHδ (rhsᵀδ>0)"
        );
        // (H+λD) must be PSD at the chosen λ (most negative eigenvalue ≥ -λ).
        let dd = &d * &step.delta;
        let lam = dd.dot(&(&rhs - &h.dot(&step.delta))) / dd.dot(&dd);
        assert!(lam >= 1.0 - 1e-6, "λ must dominate -γ_min=1, got {lam}");
    }

    /// Self-vanishing: as rhs → 0 the step → 0 regardless of the radius, so the
    /// converged β and the KKT fixed point are unchanged by the globalization.
    #[test]
    pub(crate) fn step_vanishes_as_rhs_vanishes() {
        let h = array![[3.0, 1.0], [1.0, 2.0]];
        let rhs = array![1e-13, -2e-13];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let step = spec.trust_region_step(0.5);
        assert!(
            step.delta.iter().all(|v| v.abs() < 1e-11),
            "near-zero rhs must give near-zero step, got {:?}",
            step.delta
        );
    }

    /// Null space: a genuinely zero-curvature direction is dropped from the step
    /// (Moore–Penrose range restriction) and reported via `null_rhs_inf`.
    #[test]
    pub(crate) fn null_direction_is_dropped_and_reported() {
        // Second coordinate has zero curvature; rhs has mass there.
        let h = array![[2.0, 0.0], [0.0, 0.0]];
        let rhs = array![1.0, 0.5];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let step = spec.trust_region_step(1e6);
        assert_eq!(step.nullity, 1, "one null direction expected");
        assert!(
            step.null_rhs_inf >= 0.5 - 1e-9,
            "null-space rhs component must be reported, got {}",
            step.null_rhs_inf
        );
        // The identified direction takes its exact Newton component (1/2).
        assert!((step.delta[0] - 0.5).abs() < 1e-10);
        assert!(step.delta[1].abs() < 1e-10, "null coordinate left at 0");
    }

    /// Near-null directions below the public rank tolerance must be treated the
    /// same way as exact null directions. Regresses the #1082 gauge-drift crawl:
    /// using `min(rank_cutoff, numerical_floor)` classified this mode as
    /// identified, so the step and Newton-decrement certificate chased a
    /// rank-tolerance-null gauge direction instead of dropping it.
    #[test]
    pub(crate) fn rank_tolerance_null_direction_is_dropped_from_step_and_decrement() {
        let h = array![[1.0, 0.0], [0.0, 1e-12]];
        let rhs = array![1.0, 0.5];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();

        assert!(
            spec.null_cutoff >= KKT_REFUSAL_RANK_TOL,
            "rank tolerance must set the null cutoff; got {}",
            spec.null_cutoff
        );

        let step = spec.trust_region_step(1e6);
        assert_eq!(step.nullity, 1, "near-null direction expected");
        assert!(
            step.null_rhs_inf >= 0.5 - 1e-9,
            "near-null rhs component must be reported, got {}",
            step.null_rhs_inf
        );
        assert!((step.delta[0] - 1.0).abs() < 1e-10);
        assert!(
            step.delta[1].abs() < 1e-10,
            "rank-tolerance-null coordinate must be dropped, got {}",
            step.delta[1]
        );

        let decrement = spec.newton_decrement();
        assert!(
            (decrement - 0.5).abs() < 1e-10,
            "decrement must exclude the near-null gauge mode; got {decrement}"
        );

        // The mass the decrement excluded must be reported by the audit witness:
        // the near-null mode here carries rhs component |c| = 0.5, so a
        // certificate firing on this spectrum is auditably discarding 0.5 of
        // gauge residual. `null_residual_inf` exposes exactly that, and must NOT
        // change the (decrement-only) stopping decision.
        let excluded = spec.null_residual_inf();
        assert!(
            (excluded - 0.5).abs() < 1e-10,
            "excluded near-null residual mass must be observable; got {excluded}"
        );
    }

    /// gam#1449: a mode the raw decrement excludes (`|γ| ≤ null_cutoff`) but that
    /// has GENUINE curvature (above the machine-rank `numerical_floor`) and real
    /// signal carries real achievable improvement `c²/(2|γ|)` — it must be
    /// surfaced by `weakly_identified_decrement` so the conditioning-robust gate
    /// blocks premature certification, while a mode below `numerical_floor` (true
    /// numerical null) must contribute nothing.
    #[test]
    pub(crate) fn weakly_identified_real_mode_blocks_premature_certification() {
        // λ_max = 1, p = 2 ⇒ numerical_floor = 1·√2·ε ≈ 3.1e-16,
        // null_cutoff = max(rank_tol·1, floor) = rank_tol (≈ 1e-7).
        // γ = 1e-10 is in the WEAK band: above numerical_floor, below null_cutoff.
        let h = array![[1.0, 0.0], [0.0, 1e-10]];
        let rhs = array![1.0, 0.5];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();

        // The weak mode is excluded from the step and the raw decrement.
        let raw = spec.newton_decrement();
        assert!(
            (raw - 0.5).abs() < 1e-9,
            "raw decrement excludes the weak mode; got {raw}"
        );

        // ...but its real achievable improvement c²/(2γ) = 0.25/(2·1e-10) is huge,
        // so the conditioning-robust decrement is large and blocks the certificate.
        let weak = spec.weakly_identified_decrement();
        assert!(
            weak > 1e8,
            "weakly-identified real mode must surface its achievable improvement; got {weak}"
        );

        // A genuine numerical-null mode (below numerical_floor) contributes
        // nothing: γ = 1e-300 ≪ floor, so it is true gauge, not weakly identified.
        let h_null = array![[1.0, 0.0], [0.0, 1e-300]];
        let spec_null =
            WhitenedHessianSpectrum::decompose(&h_null, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let weak_null = spec_null.weakly_identified_decrement();
        assert!(
            weak_null < 1e-12,
            "true numerical-null mode must not register as weakly identified; got {weak_null}"
        );
    }

    /// Non-identity metric: the boundary is measured in the `D` norm, so a step
    /// with a large lightly-weighted coordinate is admissible.
    #[test]
    pub(crate) fn respects_non_identity_metric() {
        let h = array![[2.0, 0.0], [0.0, 8.0]];
        let rhs = array![1.0, 1.0];
        let d = array![1.0, 16.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let r = 0.2;
        let step = spec.trust_region_step(r);
        let norm = metric_norm(&step.delta, &d);
        assert!(
            (norm - r).abs() < 1e-8,
            "step must lie on the D-metric boundary, ‖δ‖_D={norm} vs r={r}"
        );
    }

    /// Shrinking the radius re-solves the subproblem (the direction bends toward
    /// the gradient) rather than rescaling a fixed direction — the property the
    /// dogleg/truncation lacked. A halved radius must not merely halve the step.
    #[test]
    pub(crate) fn radius_shrink_bends_direction_not_just_scale() {
        let h = array![[50.0, 0.0], [0.0, 0.5]];
        let rhs = array![1.0, 1.0];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let big = spec.trust_region_step(1.0).delta;
        let small = spec.trust_region_step(0.25).delta;
        // Direction (unit vectors) must differ: a pure truncation keeps the
        // direction fixed; the exact subproblem rotates toward the steep mode.
        let big_u = &big / metric_norm(&big, &d);
        let small_u = &small / metric_norm(&small, &d);
        let cos = big_u.dot(&small_u);
        assert!(
            cos < 0.9999,
            "exact TR step must bend the direction under radius shrink (cos={cos})"
        );
    }

    /// gam#1449 — end-to-end validation in the BADLY-SCALED regime the issue
    /// names: `null_cutoff = max(rank_tol·λ_max, numerical_floor)` scales with
    /// `λ_max`, so on an ill-conditioned penalized Hessian `rank_tol·λ_max` can
    /// grow large enough to swallow a mode with small-but-REAL curvature AND real
    /// signal — a genuinely weakly-identified direction misclassified as
    /// gauge-null and excluded from BOTH the modified-Newton step and the raw
    /// Newton-decrement certificate.
    ///
    /// Concrete regime: `H = diag(1e12, 1.0)`, identity metric `D = I` (so the
    /// whitened spectrum equals `H`), `rhs = (1, 0.5)`. Then
    ///   λ_max          = 1e12
    ///   numerical_floor = 1e12·√2·ε ≈ 3.1e-4   (machine-rank cutoff)
    ///   null_cutoff     = max(1e-10·1e12, 3.1e-4) = 100.
    /// The mode `γ = 1.0` is genuinely curved (a quadrillion times above the
    /// machine-rank floor) and carries real signal `c = 0.5`, yet `1.0 ≤ 100`
    /// so it is classified as null.
    ///
    /// This test PROVES the discard is real and material — the truncated step
    /// zeroes a coordinate the un-truncated linear solve `H δ = rhs` resolves to
    /// `0.5` — and PINS that the conditioning-robust gate (`weakly_identified_
    /// decrement`, gam#1449) recovers the real mode's achievable improvement so
    /// the decrement certificate is blocked, where the raw decrement alone would
    /// have falsely certified convergence. Outcome (a): a real regime where the
    /// raw `null_cutoff` discards a real mode, caught by the hardened gate.
    #[test]
    pub(crate) fn badly_scaled_null_cutoff_discards_real_mode_but_robust_gate_catches_it() {
        // Badly-scaled: λ_max = 1e12 inflates rank_tol·λ_max to 100.
        let h = array![[1.0e12, 0.0], [0.0, 1.0]];
        let rhs = array![1.0, 0.5];
        let d = array![1.0, 1.0]; // identity metric ⇒ whitened spectrum == H
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();

        // 1. The cutoff is driven by rank_tol·λ_max (NOT the machine floor), and
        //    it is large enough to swallow the γ = 1.0 mode.
        let numerical_floor = 1.0e12 * (2.0_f64).sqrt() * f64::EPSILON;
        assert!(
            spec.null_cutoff >= 1.0,
            "badly-scaled cutoff must exceed the real mode's curvature; \
             null_cutoff={} (numerical_floor={})",
            spec.null_cutoff,
            numerical_floor,
        );
        assert!(
            spec.null_cutoff > numerical_floor * 1.0e3,
            "cutoff must be set by rank_tol·λ_max, not the machine-rank floor; \
             null_cutoff={} numerical_floor={}",
            spec.null_cutoff,
            numerical_floor,
        );

        // 2. The real, weakly-identified mode IS discarded from the step: the
        //    unconstrained modified-Newton step leaves coordinate 1 at zero.
        let truncated = spec.trust_region_step(f64::INFINITY);
        assert_eq!(
            truncated.nullity, 1,
            "the γ=1 real mode is misclassified as null"
        );
        assert!(
            truncated.delta[1].abs() < 1e-10,
            "the discarded real mode carries no step component; got δ[1]={}",
            truncated.delta[1],
        );

        // 3. ...but the un-truncated linear solve H δ = rhs resolves that
        //    coordinate to a MATERIALLY different value (c/γ = 0.5/1 = 0.5). So
        //    the truncation is not a harmless gauge drop — it is a real
        //    coefficient the certified solve would silently get wrong.
        let untruncated_delta1 = rhs[1] / h[[1, 1]]; // diagonal ⇒ exact
        assert!(
            (untruncated_delta1 - 0.5).abs() < 1e-12,
            "un-truncated solve along the weak mode = {untruncated_delta1}"
        );
        let discard_gap = (untruncated_delta1 - truncated.delta[1]).abs();
        assert!(
            discard_gap > 0.4,
            "the discarded mode must be MATERIALLY wrong vs the un-truncated \
             solve; gap={discard_gap}"
        );

        // 4. The raw Newton decrement IGNORES the discarded mode entirely, so on
        //    a residual-stall window it would falsely certify convergence: it
        //    only sees the γ=1e12 mode's tiny improvement c²/(2γ) ≈ 4.2e-25.
        // c²/(2γ) for the surviving γ=1e12 mode is 1/(2e12) ≈ 5e-13.
        let raw = spec.newton_decrement();
        assert!(
            raw < 1e-10,
            "raw decrement is below any realistic objective_tol and would \
             certify; raw={raw}"
        );

        // 5. The audit witness (already in the repo) makes the discard
        //    OBSERVABLE: the excluded near-null residual mass is the real
        //    signal c = 0.5, not gauge noise.
        let excluded = spec.null_residual_inf();
        assert!(
            (excluded - 0.5).abs() < 1e-12,
            "excluded near-null residual mass must surface the real signal; \
             got {excluded}"
        );

        // 6. THE FIX (gam#1449): the conditioning-robust decrement re-includes
        //    the genuinely-curved (above numerical_floor) excluded band, so it
        //    recovers the real achievable improvement c²/(2γ) = 0.25/2 = 0.125.
        //    The certificate gate requires THIS to be ≤ objective_tol too, so a
        //    weakly-identified real mode now BLOCKS premature certification.
        let weak = spec.weakly_identified_decrement();
        assert!(
            (weak - 0.125).abs() < 1e-9,
            "conditioning-robust decrement must recover the real mode's \
             achievable improvement c²/(2γ)=0.125; got {weak}"
        );

        // Contract for the certificate gate (inner_blockwise_fit.rs): with a
        // realistic objective_tol the raw decrement passes (would certify) but
        // the conditioning-robust decrement does NOT, so the combined gate
        // refuses to certify — the real mode is no longer silently discarded
        // from the convergence decision.
        let objective_tol = 1e-8;
        assert!(
            raw <= objective_tol,
            "raw decrement alone would have certified (raw={raw} ≤ {objective_tol})"
        );
        assert!(
            weak > objective_tol,
            "hardened gate must block certification on the real weak mode \
             (weak={weak} > {objective_tol})"
        );
    }
}

pub(crate) fn compute_kkt_refusal_report(
    cycle: usize,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    s_lambdas: &[Array2<f64>],
    ranges: &[(usize, usize)],
    cached_joint_gradient: Option<&Array1<f64>>,
    cached_active_sets: &[Option<Vec<usize>>],
    block_constraints: &[Option<LinearInequalityConstraints>],
    joint_hessian_source: Option<&JointHessianSource>,
    total_p: usize,
    ridge: f64,
    ridge_policy: RidgePolicy,
    accepted_step_inf: f64,
    proposal_step_inf: f64,
    trust_radius: f64,
    residual_tol: f64,
    obj_tol: f64,
    step_tol: f64,
    objective_change: f64,
    projected_residual_inf: f64,
    math: Option<&JointNewtonMathDiagnostic>,
) -> KktRefusalReport {
    let block_names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();
    let block_widths: Vec<usize> = states.iter().map(|s| s.beta.len()).collect();
    let block_beta_inf: Vec<f64> = states
        .iter()
        .map(|s| s.beta.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max))
        .collect();

    let block_grad_inf: Vec<f64> = match cached_joint_gradient {
        Some(joint_grad) => {
            let mut acc = 0usize;
            states
                .iter()
                .map(|s| {
                    let n = s.beta.len();
                    let end = (acc + n).min(joint_grad.len());
                    // A width-0 block (e.g. a constant-scale `noise_formula="1"`
                    // log_sigma channel collapsed to zero free coefficients,
                    // gam#553) has no gradient and a zero residual — report 0.0,
                    // not the NaN sentinel. The NaN sentinel is reserved for a
                    // genuine layout mismatch: a positive-width block whose
                    // coordinates fall past the end of the joint gradient.
                    let nrm = if n == 0 {
                        0.0
                    } else if acc < end {
                        joint_grad
                            .slice(ndarray::s![acc..end])
                            .iter()
                            .map(|x: &f64| x.abs())
                            .fold(0.0_f64, f64::max)
                    } else {
                        f64::NAN
                    };
                    acc += n;
                    nrm
                })
                .collect()
        }
        None => vec![f64::NAN; states.len()],
    };

    let block_penalty_grad_inf: Vec<f64> = ranges
        .iter()
        .enumerate()
        .map(|(b, _)| {
            let mut penalty_block = s_lambdas[b].dot(&states[b].beta);
            if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
                penalty_block += &states[b].beta.mapv(|v| ridge * v);
            }
            penalty_block
                .iter()
                .map(|x: &f64| x.abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();

    let residual_vec_opt = cached_joint_gradient.and_then(|joint_grad| {
        exact_newton_joint_projected_stationarity_vector_from_gradient(
            joint_grad,
            states,
            specs,
            s_lambdas,
            ridge,
            ridge_policy,
            block_constraints,
            Some(cached_active_sets),
        )
        .ok()
    });
    let block_residual_inf: Vec<f64> = match residual_vec_opt.as_ref() {
        Some(residual) => ranges
            .iter()
            .map(|(start, end)| {
                // A zero-width block (start == end) has no residual of its own;
                // an empty `fold` would report a spurious `0.0`. Mark it `NaN`
                // so the `is_finite()` filter below excludes it from the
                // carrying-block selection (it cannot carry residual it has no
                // parameters for).
                if start >= end {
                    f64::NAN
                } else {
                    residual
                        .slice(ndarray::s![*start..*end])
                        .iter()
                        .map(|x: &f64| x.abs())
                        .fold(0.0_f64, f64::max)
                }
            })
            .collect(),
        None => vec![f64::NAN; states.len()],
    };
    let block_carrying_residual = block_residual_inf
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);

    let mut hpen_eigenvalues_sorted_desc: Vec<f64> = Vec::new();
    let mut hpen_min_abs_eigenvalue = f64::NAN;
    let mut hpen_max_abs_eigenvalue = f64::NAN;
    let mut hpen_condition_number = f64::NAN;
    let mut hpen_nullity_at_rank_tol = 0usize;
    let mut hpen_null_gradient_inf = f64::NAN;
    let mut hpen_null_vector_block_inf = Vec::new();
    let mut hpen_null_vector_carrying_block = None;
    if total_p > 0
        && let Some(source) = joint_hessian_source
        && let Ok(mut h_joint) =
            materialize_joint_hessian_source(source, total_p, "KKT refusal diagnostic spectrum")
    {
        let model_diagonal_ridge = if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            ridge
        } else {
            0.0
        };
        add_joint_penalty_to_matrix(&mut h_joint, ranges, s_lambdas, model_diagonal_ridge, None);
        symmetrize_dense_in_place(&mut h_joint);
        if let Ok((evals, evecs)) = FaerEigh::eigh(&h_joint, Side::Lower) {
            let mut sorted: Vec<f64> = evals.iter().copied().collect();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let max_abs = sorted.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max);
            let min_abs = sorted
                .iter()
                .map(|x: &f64| x.abs())
                .fold(f64::INFINITY, f64::min);
            let cutoff = KKT_REFUSAL_RANK_TOL * max_abs;
            hpen_nullity_at_rank_tol = sorted.iter().filter(|x| x.abs() < cutoff).count();
            hpen_max_abs_eigenvalue = max_abs;
            hpen_min_abs_eigenvalue = if min_abs.is_finite() {
                min_abs
            } else {
                f64::NAN
            };
            hpen_condition_number = if min_abs > 0.0 && min_abs.is_finite() {
                max_abs / min_abs
            } else {
                f64::INFINITY
            };
            if let Some(residual) = residual_vec_opt.as_ref()
                && residual.len() == total_p
                && hpen_nullity_at_rank_tol > 0
            {
                let mut best_component = 0.0_f64;
                let mut best_block_inf = vec![0.0_f64; ranges.len()];
                for k in 0..evals.len() {
                    if evals[k].abs() >= cutoff {
                        continue;
                    }
                    let component = evecs.column(k).dot(residual).abs();
                    if component > best_component {
                        best_component = component;
                        best_block_inf.clear();
                        best_block_inf.extend(ranges.iter().map(|(start, end)| {
                            evecs
                                .slice(ndarray::s![*start..*end, k])
                                .iter()
                                .map(|x: &f64| x.abs())
                                .fold(0.0_f64, f64::max)
                        }));
                    }
                }
                hpen_null_gradient_inf = best_component;
                hpen_null_vector_block_inf = best_block_inf;
                hpen_null_vector_carrying_block = hpen_null_vector_block_inf
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i);
            }
            hpen_eigenvalues_sorted_desc = sorted;
        }
    }

    let active_set_rows_total: usize = cached_active_sets
        .iter()
        .map(|maybe_rows| maybe_rows.as_ref().map(|v| v.len()).unwrap_or(0))
        .sum();
    let any_block_has_constraints = block_constraints.iter().any(|c| c.is_some());

    let diagnosis = if hpen_nullity_at_rank_tol > 0 {
        KktRefusalDiagnosis::RankDeficientHPen
    } else if any_block_has_constraints
        && cached_active_sets.iter().any(|s| s.is_some())
        && projected_residual_inf > residual_tol
    {
        // Well-conditioned H_pen, the user has bound constraints, the current
        // active set already pinned some rows, yet the projected residual is
        // still many tolerances above the threshold. The cert refused
        // *because* the projection captured part of the multiplier but not
        // all of it — i.e. the active set is missing a row.
        KktRefusalDiagnosis::ActiveSetIncomplete
    } else {
        KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH
    };

    KktRefusalReport {
        block_names,
        block_widths,
        block_beta_inf,
        block_grad_inf,
        block_penalty_grad_inf,
        block_residual_inf,
        block_carrying_residual,
        hpen_eigenvalues_sorted_desc,
        hpen_min_abs_eigenvalue,
        hpen_max_abs_eigenvalue,
        hpen_condition_number,
        hpen_nullity_at_rank_tol,
        hpen_rank_tol: KKT_REFUSAL_RANK_TOL,
        hpen_null_gradient_inf,
        hpen_null_vector_block_inf,
        hpen_null_vector_carrying_block,
        active_set_rows_total,
        accepted_step_inf,
        proposal_step_inf,
        trust_radius,
        cycle,
        residual_tol,
        obj_tol,
        step_tol,
        linearized_rel: math
            .map(JointNewtonMathDiagnostic::linearized_rel)
            .unwrap_or(f64::NAN),
        scalar_model_relerr: math
            .map(JointNewtonMathDiagnostic::scalar_model_relative_error)
            .unwrap_or(f64::NAN),
        objective_change,
        projected_residual_inf,
        diagnosis,
    }
}

impl KktRefusalReport {
    pub(crate) fn carrying_block_label(&self) -> String {
        match self.block_carrying_residual {
            Some(idx) => format!(
                "{} (idx={}, |g|={:.3e}, |Sβ|={:.3e}, |∇L-Sβ|={:.3e}, |β|={:.3e}, width={})",
                self.block_names.get(idx).map(String::as_str).unwrap_or("?"),
                idx,
                self.block_grad_inf.get(idx).copied().unwrap_or(f64::NAN),
                self.block_penalty_grad_inf
                    .get(idx)
                    .copied()
                    .unwrap_or(f64::NAN),
                self.block_residual_inf
                    .get(idx)
                    .copied()
                    .unwrap_or(f64::NAN),
                self.block_beta_inf.get(idx).copied().unwrap_or(f64::NAN),
                self.block_widths.get(idx).copied().unwrap_or(0),
            ),
            None => "<no block carries finite residual>".to_string(),
        }
    }

    pub(crate) fn beta_inf(&self) -> f64 {
        self.block_beta_inf.iter().copied().fold(0.0_f64, f64::max)
    }

    pub(crate) fn null_direction_label(&self) -> String {
        match self.hpen_null_vector_carrying_block {
            Some(idx) => format!(
                "{} (idx={}, |u_block|∞={:.3e}, |uᵀg_proj|={:.3e})",
                self.block_names.get(idx).map(String::as_str).unwrap_or("?"),
                idx,
                self.hpen_null_vector_block_inf
                    .get(idx)
                    .copied()
                    .unwrap_or(f64::NAN),
                self.hpen_null_gradient_inf,
            ),
            None => format!("none (|uᵀg_proj|={:.3e})", self.hpen_null_gradient_inf),
        }
    }

    /// Multi-line structured log emitted at the cert REFUSED site. The
    /// per-block residual / eigenspectrum / diagnosis breakdown is what
    /// makes the failure actionable (vs the legacy one-liner that only
    /// reported aggregate residual + cert math).
    pub(crate) fn format_structured_log(&self, four_tol: f64) -> String {
        format!(
            "[PIRLS/joint-Newton convergence] cycle {:>3} | cert REFUSED: residual={:.3e} > tol={:.3e} (cert)\n  \
             carrying-block: {}\n  \
             block_names={:?}, block_widths={:?}, block_grad_inf={:?}, block_penalty_grad_inf={:?}, block_residual_inf={:?}\n  \
             H_pen spectrum: λ_max={:.3e}, λ_min={:.3e}, cond={:.3e}, nullity@{:.0e}={} (of {} eigenvalues)\n  \
             free-null diagnostic: {}\n  \
             cert math: linearized_rel={:.3e}, scalar_relerr={:.3e}, |Δobj|={:.3e} (tol={:.3e}), accepted_step_inf={:.3e} (tol={:.3e}), proposal_step_inf={:.3e}, trust_radius={:.3e}, |β|∞={:.3e}, active_set_rows_total={}\n  \
             diagnosis: {}",
            self.cycle,
            self.projected_residual_inf,
            four_tol,
            self.carrying_block_label(),
            self.block_names,
            self.block_widths,
            self.block_grad_inf,
            self.block_penalty_grad_inf,
            self.block_residual_inf,
            self.hpen_max_abs_eigenvalue,
            self.hpen_min_abs_eigenvalue,
            self.hpen_condition_number,
            self.hpen_rank_tol,
            self.hpen_nullity_at_rank_tol,
            self.hpen_eigenvalues_sorted_desc.len(),
            self.null_direction_label(),
            self.linearized_rel,
            self.scalar_model_relerr,
            self.objective_change,
            self.obj_tol,
            self.accepted_step_inf,
            self.step_tol,
            self.proposal_step_inf,
            self.trust_radius,
            self.beta_inf(),
            self.active_set_rows_total,
            self.diagnosis.as_str(),
        )
    }

    /// Single-string formatter used by the bubbled error returned from
    /// the inner solver, where the caller wants one self-contained line
    /// even though the data is structured.
    pub(crate) fn format_bubbled_error(&self) -> String {
        let carrying = self.carrying_block_label();
        format!(
            "cycle={} cert REFUSED: residual={:.3e} > tol={:.3e}; \
             carrying-block: {}; block_names={:?}, block_widths={:?}, \
             block_grad_inf={:?}, block_penalty_grad_inf={:?}, block_residual_inf={:?}; \
             H_pen spectrum: λ_max={:.3e}, λ_min={:.3e}, cond={:.3e}, nullity@{:.0e}={}/{}; \
             free-null diagnostic: {}; \
             cert math: linearized_rel={:.3e}, scalar_relerr={:.3e}, |Δobj|={:.3e}, \
             accepted_step_inf={:.3e}, proposal_step_inf={:.3e}, trust_radius={:.3e}, \
             |β|∞={:.3e}, active_set_rows_total={}; diagnosis: {}; {}",
            self.cycle,
            self.projected_residual_inf,
            4.0 * self.residual_tol,
            carrying,
            self.block_names,
            self.block_widths,
            self.block_grad_inf,
            self.block_penalty_grad_inf,
            self.block_residual_inf,
            self.hpen_max_abs_eigenvalue,
            self.hpen_min_abs_eigenvalue,
            self.hpen_condition_number,
            self.hpen_rank_tol,
            self.hpen_nullity_at_rank_tol,
            self.hpen_eigenvalues_sorted_desc.len(),
            self.null_direction_label(),
            self.linearized_rel,
            self.scalar_model_relerr,
            self.objective_change,
            self.accepted_step_inf,
            self.proposal_step_inf,
            self.trust_radius,
            self.beta_inf(),
            self.active_set_rows_total,
            self.diagnosis.as_str(),
            self.diagnosis.guidance(),
        )
    }
}

pub(crate) const JOINT_PCG_REL_TOL: f64 = 1e-8;

pub(crate) const PCG_ETA_MAX: f64 = 1.0e-1;

pub(crate) const PCG_ETA_MIN: f64 = 1.0e-8;

pub(crate) const PCG_GAMMA: f64 = 0.9;

pub(crate) const PCG_ALPHA: f64 = 1.618_033_988_749_895;

/// Eisenstat–Walker adaptive forcing term for the inner PCG tolerance:
/// when the previous outer KKT residual is known, scale the next inner
/// solve's relative tolerance by `γ·(‖r_cur‖/‖r_prev‖)^α`, clamped to
/// `[PCG_ETA_MIN, PCG_ETA_MAX]`. On the first cycle (no previous
/// residual) we use the loose `PCG_ETA_MAX` to avoid over-solving when
/// the iterate is far from the optimum.
pub(crate) fn joint_pcg_eisenstat_walker_forcing(
    prev_kkt_norm: Option<f64>,
    current_kkt_norm: f64,
) -> f64 {
    if !current_kkt_norm.is_finite() || current_kkt_norm < 0.0 {
        return JOINT_PCG_REL_TOL;
    }
    let Some(prev_kkt_norm) = prev_kkt_norm else {
        return PCG_ETA_MAX;
    };
    if !prev_kkt_norm.is_finite() || prev_kkt_norm <= 0.0 {
        return JOINT_PCG_REL_TOL;
    }
    let ratio = current_kkt_norm / prev_kkt_norm;
    if !ratio.is_finite() || ratio < 0.0 {
        return JOINT_PCG_REL_TOL;
    }
    (PCG_GAMMA * ratio.powf(PCG_ALPHA)).clamp(PCG_ETA_MIN, PCG_ETA_MAX)
}

pub(crate) fn apply_joint_penalized_hessian_into(
    source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    vector: &Array1<f64>,
    out: &mut Array1<f64>,
    joint_full_width: Option<&gam_problem::JointPenaltyBundle>,
) -> Result<(), String> {
    let mut penalty = Array1::<f64>::zeros(vector.len());
    apply_joint_penalized_hessian_into_with_workspace(
        source,
        ranges,
        s_lambdas,
        diagonal_ridge,
        vector,
        out,
        &mut penalty,
        joint_full_width,
    )
}

/// Variant of [`apply_joint_penalized_hessian_into`] that reuses a
/// caller-supplied scratch buffer for the penalty term instead of
/// allocating per call.  Use this in hot loops (e.g. the trust-region
/// trial loop) where `penalty_scratch` and the output `out` are hoisted
/// outside the loop and reused across attempts.
///
/// `penalty_scratch` must have the same length as `vector`; its contents
/// are overwritten on every call.
pub(crate) fn apply_joint_penalized_hessian_into_with_workspace(
    source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    vector: &Array1<f64>,
    out: &mut Array1<f64>,
    penalty_scratch: &mut Array1<f64>,
    joint_full_width: Option<&gam_problem::JointPenaltyBundle>,
) -> Result<(), String> {
    match source {
        JointHessianSource::Dense(h_joint) => {
            gam_linalg::faer_ndarray::fast_av_view_into(h_joint, vector, out.view_mut());
        }
        JointHessianSource::Operator { apply_into, .. } => {
            apply_into(vector, out)?;
        }
    }
    penalty_scratch.fill(0.0);
    apply_joint_block_penalty_into(
        ranges,
        s_lambdas,
        vector,
        diagonal_ridge,
        penalty_scratch,
        joint_full_width,
    );
    *out += &*penalty_scratch;
    Ok(())
}

pub(crate) fn stabilized_joint_solver_diagonal_ridge<F: CustomFamily + ?Sized>(
    family: &F,
    source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    base_diagonal_ridge: f64,
    ridge_floor: f64,
    joint_full_width: Option<&gam_problem::JointPenaltyBundle>,
) -> f64 {
    if use_exact_newton_strict_spd(family) {
        return base_diagonal_ridge;
    }
    let JointHessianSource::Dense(h_joint) = source else {
        return base_diagonal_ridge;
    };
    let mut lhs = h_joint.clone();
    add_joint_penalty_to_matrix(
        &mut lhs,
        ranges,
        s_lambdas,
        base_diagonal_ridge,
        joint_full_width,
    );
    // The penalty added above is positive-semidefinite by construction (each
    // block penalty is `λ_k·S_k` with `λ_k = exp(ρ_k) > 0` and `S_k ⪰ 0`, plus a
    // non-negative diagonal ridge). Indefiniteness of the penalized Hessian can
    // therefore only originate in the data Hessian `h_joint`, so the stabilizing
    // shift is bounded by the data Hessian's curvature — NOT the penalty's. Pass
    // `h_joint` as the Gershgorin source so the shift stays `O(data scale)` even
    // when the penalty is heavily over-smoothed (gam#979; see the function doc).
    let shift =
        exact_newton_stabilizing_shift_psd_penalized(&lhs, h_joint, ridge_floor).unwrap_or(0.0);
    if shift > 0.0 {
        log::debug!(
            "[PIRLS/joint-Newton] stabilized dense penalized Hessian with diagonal shift {:.3e}",
            shift
        );
    }
    base_diagonal_ridge + shift
}

pub(crate) fn joint_quadratic_predicted_reduction(
    rhs: &Array1<f64>,
    hpen_delta: &Array1<f64>,
    delta: &Array1<f64>,
) -> f64 {
    rhs.dot(delta) - 0.5 * delta.dot(hpen_delta)
}

pub(crate) fn joint_preconditioned_descent_delta(
    source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    rhs: &Array1<f64>,
    joint_full_width: Option<&gam_problem::JointPenaltyBundle>,
) -> Result<Array1<f64>, String> {
    let base_diagonal = match source {
        JointHessianSource::Dense(h_joint) => h_joint.diag().to_owned(),
        JointHessianSource::Operator { diagonal, .. } => diagonal.clone(),
    };
    let preconditioner = joint_penalty_preconditioner_diag(
        &base_diagonal,
        ranges,
        s_lambdas,
        diagonal_ridge,
        joint_full_width,
    );
    let mut delta = rhs / &preconditioner;
    if !delta.iter().all(|v| v.is_finite()) || rhs.dot(&delta) <= 0.0 {
        delta.assign(rhs);
    }
    let directional = rhs.dot(&delta);
    if directional.is_finite() && directional > 0.0 {
        let mut hpen_delta = Array1::<f64>::zeros(rhs.len());
        apply_joint_penalized_hessian_into(
            source,
            ranges,
            s_lambdas,
            diagonal_ridge,
            &delta,
            &mut hpen_delta,
            joint_full_width,
        )?;
        let curvature = delta.dot(&hpen_delta);
        if curvature.is_finite() && curvature > 0.0 {
            let alpha = (directional / curvature).clamp(1.0e-12, 1.0);
            delta.mapv_inplace(|v| alpha * v);
        }
    }
    Ok(delta)
}

pub(crate) fn joint_line_search_log_likelihood<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    line_search_options: &BlockwiseFitOptions,
    states: &[ParameterBlockState],
) -> Result<(f64, Option<Arc<dyn ExactNewtonJointHessianWorkspace>>), String> {
    family
        .log_likelihood_only_with_options(states, line_search_options)
        .map(|log_likelihood| (log_likelihood, None))
}

/// Fused accept-trial likelihood: build the joint-Newton Hessian workspace at
/// the trial β and read its log-likelihood, returning the workspace so the
/// post-accept `gradient_reload` reuses it instead of re-streaming all `n` rows.
///
/// Why this exists (gam#979 inner-loop `gradient_reload` cost). The trust-region
/// accept/reject decision needs only the scalar log-likelihood, and
/// [`joint_line_search_log_likelihood`] supplies it via the family's cheap
/// early-exit row sweep — the right tool for a *rejected* backtracking trial,
/// which short-circuits before the sweep finishes. But on the *accepted* trial
/// that sweep runs to completion (≈ one full row stream) and is then immediately
/// discarded: `gradient_reload` re-streams every row to build the gradient
/// workspace at the same β. At biobank scale (n≈194k, BMS flex) each stream
/// re-runs the per-row intercept Newton + cell-moment math, so the accepted
/// cycle pays the row stream *twice* (~5s of redundant `gradient_reload` work
/// per accepted cycle, the dominant inner cost once the operator-reused Hessian
/// drops to ~0.05s).
///
/// When the workspace can report a joint log-likelihood
/// ([`inner_joint_workspace_log_likelihood_available`]), building the workspace
/// once and reading `joint_log_likelihood_evaluation()` yields the SAME scalar
/// (both evaluate `Σ wᵢ log Φ` at the trial β) while leaving the per-row
/// gradient cache materialised. Threaded forward as the accepted workspace, the
/// reload then short-circuits through `joint_gradient_evaluation()` without a
/// second stream — collapsing the accepted cycle from two row passes to one.
///
/// The early-exit threshold is intentionally NOT applied here: this path is
/// taken only for a trial expected to accept (the caller gates on the first
/// attempt of a cycle), where the full sweep happens regardless, and the
/// workspace build has no monotone-lower-bound short-circuit to exploit. A
/// rejected first attempt simply discards the workspace, costing one full sweep
/// instead of an early-exited one — paid back many-fold on the common
/// accept-on-first-attempt path.
pub(crate) fn joint_line_search_log_likelihood_with_workspace<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    options: &BlockwiseFitOptions,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
) -> Result<Option<(f64, Arc<dyn ExactNewtonJointHessianWorkspace>)>, String> {
    if !family.inner_joint_workspace_log_likelihood_available(specs) {
        return Ok(None);
    }
    let Some(workspace) =
        family.exact_newton_joint_hessian_workspace_with_options(states, specs, options)?
    else {
        return Ok(None);
    };
    match workspace.joint_log_likelihood_evaluation()? {
        Some(log_likelihood) => Ok(Some((log_likelihood, workspace))),
        // The workspace advertised a log-likelihood but did not produce one;
        // fall back to the cheap scalar sweep rather than fabricating a value.
        None => Ok(None),
    }
}

pub(crate) fn coefficient_line_search_options(
    options: &BlockwiseFitOptions,
    early_exit_threshold: f64,
) -> BlockwiseFitOptions {
    let mut line_search_options = options.clone();
    // Preserve `outer_score_subsample` so the trial-objective and the
    // Hessian/gradient share a row measure: the trust-region ratio
    // ρ = [F(β) − F(β + δ)] / [−g·δ − ½·δᵀHδ] is only valid when
    // numerator and denominator evaluate the same measure. Disable
    // *auto*-install so no mid-iteration mask rebuild can occur, and
    // tag scope=InnerCoefficient so any sibling auto-install path that
    // somehow gets reached bails out (cf. `install_auto_outer_subsample_options`).
    line_search_options.auto_outer_subsample = false;
    line_search_options.outer_eval_context =
        options
            .outer_eval_context
            .as_ref()
            .map(|ctx| OuterEvalContext {
                rho: ctx.rho.clone(),
                eval_id: ctx.eval_id,
                scope: EvalScope::InnerCoefficient,
            });
    line_search_options.early_exit_threshold = Some(early_exit_threshold);
    line_search_options
}

pub(crate) type JointGradientLoad = (
    f64,
    Option<Array1<f64>>,
    Option<FamilyEvaluation>,
    Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
);

pub(crate) fn load_joint_gradient_evaluation<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    states: &[ParameterBlockState],
    prefer_workspace: bool,
    preferred_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Result<JointGradientLoad, String> {
    let workspace = match preferred_workspace {
        Some(workspace) => Some(workspace),
        None if prefer_workspace && family.inner_joint_workspace_gradient_available(specs) => {
            family.exact_newton_joint_hessian_workspace_with_options(states, specs, options)?
        }
        None => None,
    };
    if let Some(workspace_ref) = workspace.as_ref()
        && let Some(joint_eval) = workspace_ref.joint_gradient_evaluation()?
    {
        return Ok((
            joint_eval.log_likelihood,
            Some(joint_eval.gradient),
            None,
            Some(Arc::clone(workspace_ref)),
        ));
    }
    if let Some(joint_eval) = family.exact_newton_joint_gradient_evaluation(states, specs)? {
        return Ok((
            joint_eval.log_likelihood,
            Some(joint_eval.gradient),
            None,
            workspace,
        ));
    }
    let eval = family.evaluate(states)?;
    let log_likelihood = eval.log_likelihood;
    let gradient = exact_newton_joint_gradient_from_eval(&eval, specs, states)?;
    Ok((log_likelihood, gradient, Some(eval), workspace))
}

pub(crate) fn require_projected_kkt_residual(
    residual: Option<ProjectedKktResidual>,
    context: &str,
) -> Result<ProjectedKktResidual, String> {
    match residual {
        Some(residual) => Ok(residual),
        None => Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
            "{context}: converged joint-Newton exact inner solve did not produce a projected KKT \
             residual; refusing to assemble REML/LAML derivatives without the IFT correction input"
        ) }.into()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConstrainedStationaryCertificate {
    NotCandidate,
    Accept,
    RefusePhantomMultiplier,
}

#[derive(Clone, Debug)]
pub(crate) struct JointNewtonMathDiagnostic {
    pub(crate) old_kkt_inf: f64,
    pub(crate) linearized_next_kkt_inf: f64,
    pub(crate) predicted_reduction: f64,
    pub(crate) actual_reduction: f64,
    pub(crate) trust_ratio: f64,
    pub(crate) step_inf: f64,
    pub(crate) proposal_inf: f64,
}

impl JointNewtonMathDiagnostic {
    pub(crate) fn scalar_model_relative_error(&self) -> f64 {
        (self.actual_reduction - self.predicted_reduction).abs()
            / self.predicted_reduction.abs().max(1.0)
    }

    pub(crate) fn linearized_rel(&self) -> f64 {
        self.linearized_next_kkt_inf / (1.0 + self.old_kkt_inf)
    }
}

pub(crate) fn constrained_stationary_certificate_decision(
    math: &JointNewtonMathDiagnostic,
    objective_change: f64,
    objective_tol: f64,
    step_tol: f64,
    geometric_tail_bound: Option<f64>,
    residual: f64,
    residual_tol: f64,
) -> ConstrainedStationaryCertificate {
    let linearized_rel = math.linearized_rel();
    let scalar_model_relerr = math.scalar_model_relative_error();
    let objective_exhausted = objective_change <= objective_tol
        || geometric_tail_bound.is_some_and(|tail| tail <= objective_tol);
    let step_exhausted =
        math.step_inf.is_finite() && step_tol.is_finite() && math.step_inf <= step_tol;

    if !(objective_exhausted
        && step_exhausted
        && linearized_rel >= 0.5
        && scalar_model_relerr <= 1e-3)
    {
        return ConstrainedStationaryCertificate::NotCandidate;
    }

    // A large linearized residual can mean either an honest active-set
    // multiplier or an H-null/rank-deficient direction that Newton cannot
    // move. Only the projected KKT residual distinguishes those cases. This
    // small tolerance band is intentionally tied to the inner residual
    // tolerance, because this branch is allowed to certify convergence only
    // when the active-set projection has actually captured the multiplier.
    //
    // The band is a small MULTIPLE of `residual_tol`, not exactly `1x`: this
    // branch fires only once the iterate is already proven stationary (objective
    // exhausted, step exhausted, `linearized_rel >= 0.5` so the residual is
    // multiplier/null mass not a gradient defect, `scalar_relerr <= 1e-3` so the
    // quadratic model is exact). There the active-projected residual stalls at the
    // conditioning/round-off floor — for the survival baseline-hazard block
    // (well-conditioned after the data-seeded baseline, gam#797) it floors a hair
    // above the scale-relative `residual_tol`, so demanding exactly `<= tol` leaves
    // a fully-stationary iterate uncertified. A `4x` band certifies the genuinely
    // converged iterate while still rejecting a residual orders of magnitude above
    // tolerance (a real defect), the only case this guard must catch.
    let cert_residual_factor = 4.0;
    if residual.is_finite() && residual <= cert_residual_factor * residual_tol {
        ConstrainedStationaryCertificate::Accept
    } else {
        ConstrainedStationaryCertificate::RefusePhantomMultiplier
    }
}

/// True iff the recent KKT-residual tail (`history`, oldest→newest) shows STEADY
/// geometric descent: every consecutive pair strictly decreased by at least the
/// factor `(1 - min_drop)` over the whole window.
///
/// This distinguishes a still-converging Newton direction from a genuine
/// multiplier/null plateau at the certificate-refusal gate (gam#787 duchon
/// centers≥20). The constrained-stationary refusal fires on a flat objective +
/// `linearized_rel ≥ 0.5`, but those signals ALSO hold for a logslope block
/// whose residual is dropping by a steady factor each cycle (objective already
/// at its Φ-bounded floor while the KKT residual still polishes): refusing there
/// rejects the seed a few cycles short of `residual_tol`. Requiring a STEADY
/// drop over `≥ window` cycles (not a single lucky decrease) keeps a noisy
/// near-plateau from being falsely extended, and the inner cycle cap still
/// bounds the extra work.
pub(crate) fn residual_in_steady_geometric_descent(
    history: &std::collections::VecDeque<f64>,
) -> bool {
    let window = history.len();
    if window < 3 {
        return false;
    }
    let min_drop = 0.1; // each cycle must cut the residual by ≥ 10%.
    history
        .iter()
        .zip(history.iter().skip(1))
        .all(|(prev, next)| {
            prev.is_finite() && next.is_finite() && *prev > 0.0 && *next < (1.0 - min_drop) * *prev
        })
}

/// Inf-norm of the active-set-projected stationarity residual restricted to the
/// **range** of the joint penalized Hessian `H_pen = H + S(λ) + ridge·I`.
///
/// A penalized smooth whose penalty has a polynomial null space the censored /
/// location-scale data does not pin down (TP / Bernstein trend directions in a
/// survival `time_transform` or `log_sigma` channel, gam#553) leaves a residual
/// that lives entirely in `ker(H_pen)`: along that direction the objective has
/// neither curvature nor a constraint, so it is a genuinely *free* gauge
/// direction, not an unresolved KKT defect. The total residual inf-norm then
/// stays large forever and the phantom-multiplier refusal never clears, aborting
/// the fit at REML startup even though the iterate is stationary on the entire
/// identifiable (range) subspace.
///
/// The downstream outer IFT trace already removes the null-space component via
/// the projected pseudo-inverse `U_S·H_proj⁻¹·U_Sᵀ`, so only a *range-space*
/// residual component can bias the envelope gradient (see the "do NOT
/// soft-accept" investigation note at the certifier call site). This returns the
/// range-space inf-norm so the certifier can accept iff that — the only part
/// that matters for outer correctness — is at tolerance, while a real defect
/// (residual with mass in the curved subspace) still refuses.
///
/// Returns `None` when the penalized Hessian cannot be materialized or
/// eigendecomposed, or carries no numerical null space — in which case the
/// caller keeps the strict total-residual refusal (no null space ⇒ range = all).
/// Per-block inf-norms of the range-space (identified-subspace) component of the
/// projected stationarity residual (gam#979). Same construction as
/// [`projected_residual_range_space_inf`] — project the residual onto range(H_pen)
/// by dropping its ker(H_pen) coordinates — but return the inf-norm restricted to
/// EACH block's coordinate range instead of one global scalar.
///
/// The per-block stationarity gate (`all_block_stationarity_small`) must test the
/// residual on the IDENTIFIED subspace, not the raw active-set-projected residual.
/// On the survival I-spline time block the unpenalized affine baseline direction
/// is a genuine ker(H_pen) gauge mode: the raw per-block residual keeps the full
/// gradient component along it (large), so the raw gate falsely rejects a solve
/// that IS stationary on every identifiable direction. The range-projected
/// per-block residual drops exactly that gauge mass (the outer IFT pseudo-inverse
/// projects it out anyway, gam#553), so the gate sees the true identified-subspace
/// stationarity. Returns `None` when there is no null space (range == whole space)
/// — there the raw per-block residual already IS the range residual, so the caller
/// keeps its existing gate unchanged.
pub(crate) fn projected_residual_range_space_per_block_inf(
    projected_residual: &Array1<f64>,
    joint_hessian_source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    total_p: usize,
) -> Option<Vec<f64>> {
    if total_p == 0 || projected_residual.len() != total_p {
        return None;
    }
    let mut h_joint = materialize_joint_hessian_source(
        joint_hessian_source,
        total_p,
        "penalty-null-space per-block certificate spectrum",
    )
    .ok()?;
    let model_diagonal_ridge = if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
        ridge
    } else {
        0.0
    };
    add_joint_penalty_to_matrix(&mut h_joint, ranges, s_lambdas, model_diagonal_ridge, None);
    symmetrize_dense_in_place(&mut h_joint);
    let (evals, evecs) = FaerEigh::eigh(&h_joint, Side::Lower).ok()?;
    let max_abs = evals.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max);
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return None;
    }
    let cutoff = KKT_REFUSAL_RANK_TOL * max_abs;
    let nullity = evals.iter().filter(|x| x.abs() < cutoff).count();
    if nullity == 0 {
        return None;
    }
    let mut range_component = Array1::<f64>::zeros(total_p);
    for k in 0..evals.len() {
        if evals[k].abs() < cutoff {
            continue;
        }
        let coeff = evecs.column(k).dot(projected_residual);
        range_component.scaled_add(coeff, &evecs.column(k));
    }
    Some(
        ranges
            .iter()
            .map(|&(start, end)| {
                range_component
                    .slice(ndarray::s![start..end])
                    .iter()
                    .map(|x: &f64| x.abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect(),
    )
}

pub(crate) fn projected_residual_range_space_inf(
    projected_residual: &Array1<f64>,
    joint_hessian_source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    total_p: usize,
) -> Option<f64> {
    if total_p == 0 || projected_residual.len() != total_p {
        return None;
    }
    let mut h_joint = materialize_joint_hessian_source(
        joint_hessian_source,
        total_p,
        "penalty-null-space certificate spectrum",
    )
    .ok()?;
    let model_diagonal_ridge = if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
        ridge
    } else {
        0.0
    };
    add_joint_penalty_to_matrix(&mut h_joint, ranges, s_lambdas, model_diagonal_ridge, None);
    symmetrize_dense_in_place(&mut h_joint);
    let (evals, evecs) = FaerEigh::eigh(&h_joint, Side::Lower).ok()?;
    let max_abs = evals.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max);
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return None;
    }
    let cutoff = KKT_REFUSAL_RANK_TOL * max_abs;
    let nullity = evals.iter().filter(|x| x.abs() < cutoff).count();
    if nullity == 0 {
        // No data-unconstrained null space — the range is the whole space, so
        // the strict total-residual refusal already governs. Signal "no relief".
        return None;
    }
    // Range-space residual = residual minus its projection onto ker(H_pen).
    // Equivalently, accumulate the residual's coordinates along every
    // range-space (|λ| ≥ cutoff) eigenvector. The eigenbasis is orthonormal,
    // so ‖P_range r‖∞ is read off the reconstructed range component.
    let mut range_component = Array1::<f64>::zeros(total_p);
    for k in 0..evals.len() {
        if evals[k].abs() < cutoff {
            continue;
        }
        let coeff = evecs.column(k).dot(projected_residual);
        range_component.scaled_add(coeff, &evecs.column(k));
    }
    Some(
        range_component
            .iter()
            .map(|x: &f64| x.abs())
            .fold(0.0_f64, f64::max),
    )
}
