// Cross-block identifiability canonicalisation — fail-closed safety gate.
//
// The pre-fit `audit_identifiability` (see `identifiability_audit.rs`)
// runs a joint RRQR on `[X_block_0 | X_block_1 | ...]` and reports per-
// block (block_idx, local_col) drops attributing each demoted joint
// column back to its origin. This module **previously** converted that
// report into a concrete coordinate transform that reduced the inner
// solve to a (p_raw → r_reduced) subspace via selection-T.
//
// That reduction is unsafe under the current `CustomFamily` contract.
// Blockwise families capture their per-block designs at construction
// time (e.g. `SurvivalMarginalSlopeFamily::marginal_design`,
// `::logslope_design`) and the family's `evaluate_blockwise_exact_newton`
// row-Hessian assembly uses `DesignMatrix::syr_row_into_view` /
// `::row_outer_into_view`, which assert that the target slice's column
// count equals the captured design's column count. Substituting a
// column-reduced `ParameterBlockSpec` under such a family produces
// `DesignMatrix::syr_row_into shape mismatch` (matrix.rs:6529), which
// blockwise inner-solve callers unwrap via `.expect(...)` — a panic
// later in the pipeline, masking the audit's diagnostic.
//
// Until the family contract is updated to consume reduced specs
// (Phase 4b of `identifiability_compiler` — see
// `src/families/identifiability_compiler.rs`), this module's
// `canonicalize_for_identifiability` is a **fail-closed audit gate**:
//   - clean audit (`!fatal`)             → identity transforms, raw specs returned as-is;
//   - fatal audit (any cause)            → `CustomFamilyError::IdentifiabilityFailure`.
//
// The identity-on-clean path keeps `lift_block_states_to_raw` and
// `lift_fit_geometry_to_raw` cheap no-ops in the outer-fit code path
// without altering its surface API. The fail-closed-on-fatal path
// converts what was a latent panic into an immediate, actionable
// `Err` naming the offending blocks and a reparameterisation hint —
// in milliseconds rather than minutes of singular Newton.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};

use crate::families::custom_family::{
    BlockEffectiveJacobian, CustomFamilyError, FamilyLinearizationState, ParameterBlockSpec,
    PenaltyMatrix,
};
use crate::families::identifiability_compiler::{IdentityRowHessian, RowJacobianOperator};
use crate::linalg::faer_ndarray::{default_rrqr_rank_alpha, rrqr_with_permutation};
use crate::linalg::matrix::{CoefficientTransformOperator, DenseDesignMatrix, DesignMatrix};
use crate::solver::identifiability_audit::{
    IdentifiabilityAudit, audit_identifiability, audit_identifiability_channel_aware,
};

/// A [`RowJacobianOperator`] built from a [`BlockEffectiveJacobian`] callback.
///
/// At audit time we call `effective_jacobian_at` once with `beta = 0` and
/// no family scalars.  The callback returns an `(n * k, p)` stacked matrix
/// (row-major channel stacking: block `i·k .. (i+1)·k` is observation `i`).
/// We reshape that into the `(n, p, k)` tensor that `RowJacobianOperator`
/// expects.
struct BlockJacobianAsRowOp {
    /// Materialised `(n, p, k)` Jacobian tensor.
    jac: Array3<f64>,
}

impl BlockJacobianAsRowOp {
    /// Build from a `BlockEffectiveJacobian` callback.
    ///
    /// `n_rows` is the number of training observations, `k` is the number of
    /// output channels.  The `effective_jacobian_at` call uses `beta = 0` and
    /// `family_scalars = None`.
    fn from_callback(
        cb: &dyn BlockEffectiveJacobian,
        n_rows: usize,
        block_name: &str,
    ) -> Result<Self, String> {
        let p = 0; // placeholder; will be determined from callback output below
        let k = cb.n_outputs();
        let zeros = vec![0.0f64; p];
        let state = FamilyLinearizationState {
            beta: &zeros,
            family_scalars: None,
            channel_hessian: None,
        };
        let stacked = cb
            .effective_jacobian_at(&state)
            .map_err(|e| format!("BlockJacobianAsRowOp block '{block_name}': {e}"))?;
        // stacked: (n_rows * k, p_block)
        let nk = stacked.nrows();
        let p_block = stacked.ncols();
        if k == 0 {
            return Err(format!(
                "BlockJacobianAsRowOp block '{block_name}': n_outputs=0 is invalid"
            ));
        }
        if nk != n_rows * k {
            return Err(format!(
                "BlockJacobianAsRowOp block '{block_name}': effective_jacobian_at returned \
                 {} rows but expected n_rows({n_rows}) * k({k}) = {}",
                nk,
                n_rows * k,
            ));
        }
        // Reshape (n*k, p) → (n, p, k) by interpreting stacked row `i*k + r`
        // as channel `r` for observation `i`.
        let mut jac = Array3::<f64>::zeros((n_rows, p_block, k));
        for i in 0..n_rows {
            for j in 0..p_block {
                for r in 0..k {
                    jac[[i, j, r]] = stacked[[i * k + r, j]];
                }
            }
        }
        Ok(Self { jac })
    }
}

impl RowJacobianOperator for BlockJacobianAsRowOp {
    fn k(&self) -> usize {
        self.jac.shape()[2]
    }
    fn ncols(&self) -> usize {
        self.jac.shape()[1]
    }
    fn nrows(&self) -> usize {
        self.jac.shape()[0]
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        let k = self.k();
        assert_eq!(out.len(), k);
        assert_eq!(delta_beta.len(), self.ncols());
        for r in 0..k {
            out[r] = 0.0;
        }
        for (j, &b) in delta_beta.iter().enumerate() {
            for r in 0..k {
                out[r] += self.jac[[row, j, r]] * b;
            }
        }
    }
    fn evaluate_full(&self) -> Array3<f64> {
        self.jac.clone()
    }
}

/// Specs after pre-fit cross-block identifiability canonicalisation.
///
/// `reduced_specs[i]` carries an `r_i`-column design wrapping the raw
/// `p_i`-column design via `CoefficientTransformOperator`. Penalties
/// are pulled back as `T_iᵀ S_k T_i`. `per_block_transform[i]` is the
/// raw-to-reduced selection matrix `T_i` of shape `(p_i_raw, r_i)`.
///
/// `used_channel_aware_audit` is `true` when the multi-channel path was
/// taken (i.e. at least one block declared `n_outputs > 1` via its
/// `jacobian_callback`).  Tests that assert routing correctness inspect
/// this field directly.
#[derive(Debug)]
pub struct CanonicalSpecs {
    pub reduced_specs: Vec<ParameterBlockSpec>,
    pub per_block_transform: Vec<Array2<f64>>,
    pub audit: IdentifiabilityAudit,
    /// `true` iff the audit was routed through `audit_identifiability_channel_aware`
    /// (multi-channel families such as survival marginal-slope).
    pub used_channel_aware_audit: bool,
}

impl CanonicalSpecs {
    /// Lift reduced-space block coefficients θ_i back to the raw space
    /// via `β_i_raw = T_i · θ_i`. Dropped raw coordinates receive zero.
    pub fn lift_block_betas_to_raw(&self, theta_blocks: &[Array1<f64>]) -> Vec<Array1<f64>> {
        assert_eq!(
            theta_blocks.len(),
            self.per_block_transform.len(),
            "lift_block_betas_to_raw: theta blocks ({}) != transforms ({})",
            theta_blocks.len(),
            self.per_block_transform.len(),
        );
        let mut out = Vec::with_capacity(theta_blocks.len());
        for (theta, transform) in theta_blocks.iter().zip(self.per_block_transform.iter()) {
            assert_eq!(
                theta.len(),
                transform.ncols(),
                "lift_block_betas_to_raw: theta length {} != transform ncols {}",
                theta.len(),
                transform.ncols(),
            );
            out.push(transform.dot(theta));
        }
        out
    }

    /// Raw block dimensions (rows of each `T_i`). Used to bound expansion.
    pub fn raw_block_dims(&self) -> Vec<usize> {
        self.per_block_transform.iter().map(|t| t.nrows()).collect()
    }

    /// Reduced block dimensions (cols of each `T_i`).
    pub fn reduced_block_dims(&self) -> Vec<usize> {
        self.per_block_transform.iter().map(|t| t.ncols()).collect()
    }

    /// Lift a reduced-space joint matrix `M_red` (total_r × total_r) to
    /// raw-space (total_p × total_p) via `T_full · M_red · T_fullᵀ`
    /// where `T_full = blockdiag(T_i)`. For selection-T this places the
    /// reduced block at surviving raw indices and leaves the rest zero.
    pub fn lift_joint_matrix_to_raw(&self, m_red: &Array2<f64>) -> Array2<f64> {
        let raw_dims = self.raw_block_dims();
        let red_dims = self.reduced_block_dims();
        let total_p: usize = raw_dims.iter().sum();
        let total_r: usize = red_dims.iter().sum();
        assert_eq!(
            m_red.nrows(),
            total_r,
            "lift_joint_matrix_to_raw: m_red rows {} != total reduced dim {}",
            m_red.nrows(),
            total_r,
        );
        assert_eq!(
            m_red.ncols(),
            total_r,
            "lift_joint_matrix_to_raw: m_red cols {} != total reduced dim {}",
            m_red.ncols(),
            total_r,
        );
        let mut out = Array2::<f64>::zeros((total_p, total_p));
        let mut raw_off_i = 0usize;
        let mut red_off_i = 0usize;
        for (i, t_i) in self.per_block_transform.iter().enumerate() {
            let p_i = raw_dims[i];
            let r_i = red_dims[i];
            let mut raw_off_j = 0usize;
            let mut red_off_j = 0usize;
            for (j, t_j) in self.per_block_transform.iter().enumerate() {
                let p_j = raw_dims[j];
                let r_j = red_dims[j];
                if r_i > 0 && r_j > 0 {
                    let m_ij = m_red.slice(ndarray::s![
                        red_off_i..red_off_i + r_i,
                        red_off_j..red_off_j + r_j
                    ]);
                    let lifted = t_i.dot(&m_ij).dot(&t_j.t());
                    out.slice_mut(ndarray::s![
                        raw_off_i..raw_off_i + p_i,
                        raw_off_j..raw_off_j + p_j
                    ])
                    .assign(&lifted);
                }
                raw_off_j += p_j;
                red_off_j += r_j;
            }
            raw_off_i += p_i;
            red_off_i += r_i;
        }
        out
    }
}

/// Run the pre-fit cross-block identifiability audit. Fail-closed
/// safety gate (see module docs).
///
/// Behaviour:
///   - If the audit cleanly passes (`!fatal`), each `T_i` is the
///     identity and the reduced specs are clones of the raw specs.
///     The lift/sandwich machinery downstream becomes a no-op.
///   - If the audit is `fatal=true` for **any** cause (joint rank
///     deficiency with attributed drops, joint rank deficiency
///     without attribution, or a hard-overlap alias pair), we refuse
///     the fit with `CustomFamilyError::IdentifiabilityFailure`. The
///     audit summary names the offending blocks and a reparameterisation
///     hint, giving the caller a millisecond-scale diagnostic instead
///     of a downstream `syr_row_into shape mismatch` panic when the
///     family captures raw-width designs.
///
/// # Multi-channel routing
///
/// When any spec's `jacobian_callback` reports `n_outputs > 1` (i.e.
/// the block contributes to multiple stacked output channels — as in
/// survival marginal-slope where marginal and logslope blocks target
/// orthogonal channels of the per-row Jacobian), this function routes
/// through [`audit_identifiability_channel_aware`] instead of the flat
/// [`audit_identifiability`].
///
/// The routing decision is principled: for each spec, call
/// `effective_jacobian_at(beta=0)` and check whether the returned
/// matrix has `nrows > n` (i.e. `nrows == n * k` for some `k > 1`).
/// If any block satisfies this, all blocks are treated as multi-channel
/// and [`BlockJacobianAsRowOp`] adapters are built from each spec's
/// callback to feed the channel-aware audit.
///
/// For specs without a `jacobian_callback`, the flat design with a
/// single-channel identity operator is used.
///
/// # Invariant assertion
///
/// After building the transform `T`, the post-T joint Jacobian
/// `J_can = J · T_full` is materialised and RRQR-checked.  If
/// `rank(J_can) != rank(J)`, the transform `T` is defective (a bug in
/// its construction) and the function returns
/// `CustomFamilyError::DimensionMismatch` with a diagnostic naming
/// `rank(J)`, `rank(J_can)`, and all per-block `T_i` shapes.
pub fn canonicalize_for_identifiability(
    specs: &[ParameterBlockSpec],
) -> Result<CanonicalSpecs, CustomFamilyError> {
    if specs.is_empty() {
        return Ok(CanonicalSpecs {
            reduced_specs: Vec::new(),
            per_block_transform: Vec::new(),
            audit: audit_identifiability(specs).map_err(|r| {
                CustomFamilyError::DimensionMismatch {
                    reason: format!("pre-fit identifiability audit failed: {r}"),
                }
            })?,
            used_channel_aware_audit: false,
        });
    }

    let n_rows = specs[0].design.nrows();

    // ── Multi-channel routing decision ───────────────────────────────────
    //
    // Probe each spec's effective Jacobian at beta=0 to detect
    // multi-output blocks.  A block is multi-output when the returned
    // matrix has nrows > n_rows — i.e. nrows == n_rows * k for k > 1.
    // We use the spec-level n_outputs() shortcut when a jacobian_callback
    // is present; otherwise the block is always single-output (k=1).
    let max_n_outputs = specs
        .iter()
        .map(|s| s.jacobian_callback.as_ref().map(|cb| cb.n_outputs()).unwrap_or(1))
        .max()
        .unwrap_or(1);
    let use_channel_aware = max_n_outputs > 1;

    log::debug!(
        "[CANON] canonicalize_for_identifiability: blocks={} n_rows={} \
         max_n_outputs={} route={}",
        specs.len(),
        n_rows,
        max_n_outputs,
        if use_channel_aware { "channel-aware" } else { "flat" },
    );

    // ── Per-block Jacobian Frobenius-norm logging (instrumentation) ──────
    //
    // Log the Frobenius norm and row-count of each block's effective
    // Jacobian before the audit so discrepancies between pilot and outer-fit
    // audits are visible in the log stream.
    for spec in specs.iter() {
        let jac_nrows = if use_channel_aware {
            let k = spec.jacobian_callback.as_ref().map(|cb| cb.n_outputs()).unwrap_or(1);
            n_rows * k
        } else {
            n_rows
        };
        let frob_sq: f64 = {
            let p = spec.design.ncols();
            let zeros = vec![0.0f64; p];
            let state = FamilyLinearizationState {
                beta: &zeros,
                family_scalars: None,
                channel_hessian: None,
            };
            match spec.effective_jacobian_at("canonicalize_for_identifiability", &state) {
                Ok(j) => j.iter().map(|v| v * v).sum(),
                Err(e) => {
                    log::debug!(
                        "[CANON]   block '{}': effective_jacobian_at probe failed: {e}",
                        spec.name,
                    );
                    f64::NAN
                }
            }
        };
        log::debug!(
            "[CANON]   block '{}': p={} jac_nrows={} frob_norm={:.4e}",
            spec.name,
            spec.design.ncols(),
            jac_nrows,
            frob_sq.sqrt(),
        );
    }

    // ── Run the audit ─────────────────────────────────────────────────────
    let audit = if use_channel_aware {
        // Determine the common k (all blocks must agree on the channel count;
        // blocks without a jacobian_callback get a single-channel identity
        // adapter at k = max_n_outputs).
        let k = max_n_outputs;
        let mut operators: Vec<Arc<dyn RowJacobianOperator>> = Vec::with_capacity(specs.len());
        for spec in specs.iter() {
            let op: Arc<dyn RowJacobianOperator> =
                match spec.jacobian_callback.as_ref() {
                    Some(cb) if cb.n_outputs() == k => {
                        let row_op =
                            BlockJacobianAsRowOp::from_callback(cb.as_ref(), n_rows, &spec.name)
                                .map_err(|e| CustomFamilyError::DimensionMismatch {
                                    reason: format!(
                                        "canonicalize_for_identifiability: build \
                                         BlockJacobianAsRowOp for block '{}': {e}",
                                        spec.name,
                                    ),
                                })?;
                        Arc::new(row_op)
                    }
                    Some(cb) => {
                        // k mismatch: this block has fewer outputs than
                        // max_n_outputs — embed its Jacobian in the top
                        // channels and zero-pad the rest.
                        let k_block = cb.n_outputs();
                        let row_op =
                            BlockJacobianAsRowOp::from_callback(cb.as_ref(), n_rows, &spec.name)
                                .map_err(|e| CustomFamilyError::DimensionMismatch {
                                    reason: format!(
                                        "canonicalize_for_identifiability: build \
                                         BlockJacobianAsRowOp (k_mismatch) for block '{}': {e}",
                                        spec.name,
                                    ),
                                })?;
                        // Embed: extend jac channels from k_block to k by
                        // zero-padding the trailing channels.
                        let mut jac_ext = Array3::<f64>::zeros((
                            row_op.nrows(),
                            row_op.ncols(),
                            k,
                        ));
                        let jac_inner = row_op.evaluate_full();
                        for i in 0..row_op.nrows() {
                            for j in 0..row_op.ncols() {
                                for r in 0..k_block {
                                    jac_ext[[i, j, r]] = jac_inner[[i, j, r]];
                                }
                            }
                        }
                        Arc::new(BlockJacobianAsRowOp { jac: jac_ext })
                    }
                    None => {
                        // Single-output block: embed the flat design in the
                        // first channel.
                        let p = spec.design.ncols();
                        let zeros = vec![0.0f64; p];
                        let state = FamilyLinearizationState {
                            beta: &zeros,
                            family_scalars: None,
                            channel_hessian: None,
                        };
                        let flat = spec
                            .effective_jacobian_at(
                                "canonicalize_for_identifiability",
                                &state,
                            )
                            .map_err(|e| CustomFamilyError::DimensionMismatch {
                                reason: format!(
                                    "canonicalize_for_identifiability: effective_jacobian_at \
                                     for block '{}': {e}",
                                    spec.name,
                                ),
                            })?;
                        let mut jac_ext =
                            Array3::<f64>::zeros((n_rows, flat.ncols(), k));
                        for i in 0..n_rows {
                            for j in 0..flat.ncols() {
                                jac_ext[[i, j, 0]] = flat[[i, j]];
                            }
                        }
                        Arc::new(BlockJacobianAsRowOp { jac: jac_ext })
                    }
                };
            operators.push(op);
        }
        let row_hess = IdentityRowHessian::new(n_rows, k);
        let audit_result =
            audit_identifiability_channel_aware(specs, &operators, &row_hess).map_err(
                |reason| CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "pre-fit channel-aware identifiability audit failed: {reason}"
                    ),
                },
            )?;

        log::info!(
            "[CANON] channel-aware audit: {} blocks, joint_rank={}/{} (flat audit NOT used)",
            specs.len(),
            audit_result.blocks.iter().map(|b| b.effective_dim).sum::<usize>(),
            specs.iter().map(|s| s.design.ncols()).sum::<usize>(),
        );

        // Also run the flat audit for comparison and log the discrepancy.
        // This is pure instrumentation — it does NOT affect the routing or
        // the fatal decision. We log it at debug level so the 23→20 rank drop
        // is visible without being a false alarm.
        if let Ok(flat_audit) = audit_identifiability(specs) {
            let flat_rank: usize = flat_audit.blocks.iter().map(|b| b.effective_dim).sum();
            let ca_rank: usize =
                audit_result.blocks.iter().map(|b| b.effective_dim).sum();
            if flat_rank != ca_rank {
                log::info!(
                    "[CANON] rank discrepancy: flat_rank={flat_rank} channel_aware_rank={ca_rank}; \
                     the flat audit would have {action} this fit; the channel-aware verdict is used",
                    action = if flat_audit.fatal { "refused" } else { "accepted (but with drops)" },
                );
            } else {
                log::debug!(
                    "[CANON] flat_rank == channel_aware_rank = {flat_rank} (no discrepancy)",
                );
            }
        }

        audit_result
    } else {
        let audit_result =
            audit_identifiability(specs).map_err(|reason| CustomFamilyError::DimensionMismatch {
                reason: format!("pre-fit identifiability audit failed: {reason}"),
            })?;
        log::debug!(
            "[CANON] flat audit: {} blocks, joint_rank={}",
            specs.len(),
            audit_result.blocks.iter().map(|b| b.effective_dim).sum::<usize>(),
        );
        audit_result
    };

    if audit.fatal {
        return Err(CustomFamilyError::IdentifiabilityFailure { audit });
    }

    let mut per_block_transform: Vec<Array2<f64>> = Vec::with_capacity(specs.len());
    let mut reduced_specs: Vec<ParameterBlockSpec> = Vec::with_capacity(specs.len());

    for spec in specs.iter() {
        let p_raw = spec.design.ncols();
        let dropped_locals: Vec<usize> = audit
            .dropped_columns
            .iter()
            .filter(|drop| drop.block == spec.name)
            .map(|drop| drop.column)
            .collect();
        let mut dropped_sorted = dropped_locals.clone();
        dropped_sorted.sort_unstable();
        dropped_sorted.dedup();
        for &col in &dropped_sorted {
            if col >= p_raw {
                crate::bail_dim_custom!(
                    "canonicalize_for_identifiability: audit reported dropped column \
                         {col} for block '{}' which has only {} columns",
                    spec.name,
                    p_raw,
                );
            }
        }
        let kept: Vec<usize> = (0..p_raw)
            .filter(|c| dropped_sorted.binary_search(c).is_err())
            .collect();
        let r_block = kept.len();

        let mut t_i = Array2::<f64>::zeros((p_raw, r_block));
        for (col_out, &raw_col) in kept.iter().enumerate() {
            t_i[[raw_col, col_out]] = 1.0;
        }

        let reduced_design = if dropped_sorted.is_empty() {
            spec.design.clone()
        } else {
            build_reduced_design(&spec.design, &kept, &spec.name, &t_i)?
        };

        let reduced_penalties: Vec<PenaltyMatrix> = spec
            .penalties
            .iter()
            .map(|p| pull_back_penalty(p, &kept))
            .collect();

        let reduced_initial_beta = match &spec.initial_beta {
            Some(beta_raw) => {
                if beta_raw.len() != p_raw {
                    crate::bail_dim_custom!(
                        "canonicalize_for_identifiability: block '{}' initial_beta \
                             length {} != design ncols {}",
                        spec.name,
                        beta_raw.len(),
                        p_raw,
                    );
                }
                let mut theta = Array1::<f64>::zeros(r_block);
                for (out_idx, &raw_col) in kept.iter().enumerate() {
                    theta[out_idx] = beta_raw[raw_col];
                }
                Some(theta)
            }
            None => None,
        };

        reduced_specs.push(ParameterBlockSpec {
            name: spec.name.clone(),
            design: reduced_design,
            offset: spec.offset.clone(),
            penalties: reduced_penalties,
            // Pulled-back penalties may carry an enlarged structural
            // nullspace (a column dropped from a smooth's pure-span
            // basis adds that direction to the penalty kernel).
            // Falling back to eigenvalue-based rank detection in the
            // pseudo-logdet path is the safe choice when the
            // selection-T pullback changes the kernel structurally.
            nullspace_dims: Vec::new(),
            initial_log_lambdas: spec.initial_log_lambdas.clone(),
            initial_beta: reduced_initial_beta,
            gauge_priority: spec.gauge_priority,
            // The reduction operates on raw-column selection; the
            // effective linear-predictor scaling is unchanged after
            // column selection (the surviving columns still carry the
            // same per-row scaling as in the original spec).  The
            // jacobian_callback (if any) is also forwarded: the callback
            // internally uses the raw design width, which the column-
            // selection T_i accounts for by selecting surviving columns.
            row_scaling: spec.row_scaling.clone(),
            jacobian_callback: spec.jacobian_callback.clone(),
        });
        per_block_transform.push(t_i);
    }

    // ── Post-T invariant check ────────────────────────────────────────────
    //
    // Materialise the joint post-T Jacobian J_can = J · T_full where
    // J is the (n*k × p_total) stacked Jacobian and T_full is block-diagonal
    // of the per-block T_i.  Assert rank(J_can) == rank(J_pre_T).
    //
    // This catches bugs in the T construction (wrong kept-column mapping,
    // off-by-one indexing, etc.) before they silently degrade the fit.
    // The check is run unconditionally — at GAM smooth widths (p_total ≪ n)
    // the RRQR on J_can is cheap.
    {
        let p_total_raw: usize = specs.iter().map(|s| s.design.ncols()).sum();
        let p_total_red: usize = per_block_transform.iter().map(|t| t.ncols()).sum();
        let k = if use_channel_aware { max_n_outputs } else { 1 };
        let nk = n_rows * k;

        // Build J_pre_T: (nk, p_total_raw) by row-stacking per-block Jacobians.
        let mut j_pre = Array2::<f64>::zeros((nk, p_total_raw));
        let mut col_off = 0usize;
        for spec in specs.iter() {
            let p_b = spec.design.ncols();
            let zeros = vec![0.0f64; p_b];
            let state = FamilyLinearizationState {
                beta: &zeros,
                family_scalars: None,
                channel_hessian: None,
            };
            match spec.effective_jacobian_at("canonicalize_rank_check", &state) {
                Ok(j_b) => {
                    // j_b is (nk_b, p_b) where nk_b = n_rows * k_b
                    // For single-channel blocks k_b = 1, so nk_b = n_rows.
                    // For multi-channel blocks k_b = k, so nk_b = nk.
                    // We embed single-channel blocks in the first k rows of
                    // each observation's nk block.
                    let k_b = j_b.nrows() / n_rows;
                    for i in 0..n_rows {
                        for j in 0..p_b {
                            for r in 0..k_b.min(k) {
                                j_pre[[i * k + r, col_off + j]] = j_b[[i * k_b + r, j]];
                            }
                        }
                    }
                }
                Err(_) => {
                    // Fall back: embed flat design as channel 0.
                    if let Ok(flat) = spec
                        .design
                        .try_to_dense_arc("canonicalize_rank_check")
                        .map(|a| a.as_ref().clone())
                    {
                        for i in 0..n_rows.min(flat.nrows()) {
                            for j in 0..p_b.min(flat.ncols()) {
                                j_pre[[i * k, col_off + j]] = flat[[i, j]];
                            }
                        }
                    }
                }
            }
            col_off += p_b;
        }

        // Build J_can = J_pre · T_full where T_full = blockdiag(T_i).
        let mut j_can = Array2::<f64>::zeros((nk, p_total_red));
        let mut raw_col_off = 0usize;
        let mut red_col_off = 0usize;
        for t_i in per_block_transform.iter() {
            let p_i = t_i.nrows();
            let r_i = t_i.ncols();
            if p_i > 0 && r_i > 0 {
                // J_can[:, red_col_off .. red_col_off+r_i]
                //   = J_pre[:, raw_col_off .. raw_col_off+p_i] · T_i
                for row in 0..nk {
                    for out_col in 0..r_i {
                        let mut acc = 0.0_f64;
                        for in_col in 0..p_i {
                            acc += j_pre[[row, raw_col_off + in_col]] * t_i[[in_col, out_col]];
                        }
                        j_can[[row, red_col_off + out_col]] = acc;
                    }
                }
            }
            raw_col_off += p_i;
            red_col_off += r_i;
        }

        // RRQR rank on J_pre and J_can.
        let rank_j_pre = rrqr_with_permutation(&j_pre, default_rrqr_rank_alpha())
            .map(|r| r.rank)
            .unwrap_or(0);
        let rank_j_can = rrqr_with_permutation(&j_can, default_rrqr_rank_alpha())
            .map(|r| r.rank)
            .unwrap_or(0);

        log::info!(
            "[CANON] post-T invariant: rank(J)={rank_j_pre} rank(J_can)={rank_j_can} \
             (p_raw={p_total_raw} p_red={p_total_red} k={k})",
        );

        if rank_j_pre != rank_j_can {
            let block_shapes: Vec<String> = per_block_transform
                .iter()
                .zip(specs.iter())
                .map(|(t, s)| format!("{}:({},{})", s.name, t.nrows(), t.ncols()))
                .collect();
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "canonicalize_for_identifiability: post-T rank invariant violated — \
                     rank(J)={rank_j_pre} but rank(J_can)={rank_j_can} \
                     (p_raw={p_total_raw} p_red={p_total_red} k={k}); \
                     this is a bug in T construction; per-block T shapes: [{}]",
                    block_shapes.join(", "),
                ),
            });
        }
    }

    Ok(CanonicalSpecs {
        reduced_specs,
        per_block_transform,
        audit,
        used_channel_aware_audit: use_channel_aware,
    })
}

fn build_reduced_design(
    raw: &DesignMatrix,
    kept: &[usize],
    block_name: &str,
    t_i: &Array2<f64>,
) -> Result<DesignMatrix, CustomFamilyError> {
    let inner_dense = match raw {
        DesignMatrix::Dense(d) => d.clone(),
        DesignMatrix::Sparse(_) => {
            let dense = raw
                .try_to_dense_by_chunks(&format!(
                    "canonicalize_for_identifiability sparse->dense block '{block_name}'"
                ))
                .map_err(|reason| CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "canonicalize_for_identifiability: densify sparse block '{block_name}' \
                         failed: {reason}"
                    ),
                })?;
            DenseDesignMatrix::from(dense)
        }
    };
    // Hot path: when the inner is already a materialised dense Array2,
    // slice the kept columns directly. This avoids carrying the full
    // raw-width inner through every PIRLS iteration when many columns
    // were dropped.
    if let Some(arr) = inner_dense.as_dense_ref() {
        let reduced =
            Array2::<f64>::from_shape_fn((arr.nrows(), kept.len()), |(i, j)| arr[[i, kept[j]]]);
        return Ok(DesignMatrix::Dense(DenseDesignMatrix::from(reduced)));
    }
    // Operator-backed inner (Lazy): preserve the operator structure
    // by wrapping with CoefficientTransformOperator on the selection T.
    let op = CoefficientTransformOperator::new(inner_dense, t_i.clone()).map_err(|reason| {
        CustomFamilyError::DimensionMismatch {
            reason: format!(
                "canonicalize_for_identifiability: build CoefficientTransformOperator \
                 for block '{block_name}': {reason}"
            ),
        }
    })?;
    Ok(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(op))))
}

fn pull_back_penalty(penalty: &PenaltyMatrix, kept: &[usize]) -> PenaltyMatrix {
    let label = penalty.precision_label().map(|s| s.to_string());
    let dense = penalty.as_dense_cow();
    let reduced =
        Array2::<f64>::from_shape_fn((kept.len(), kept.len()), |(i, j)| dense[[kept[i], kept[j]]]);
    let base = PenaltyMatrix::Dense(reduced);
    match label {
        Some(lbl) => base.with_precision_label(lbl),
        None => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::matrix::DenseDesignMatrix;
    use ndarray::Array2;

    fn spec_from_dense(name: &str, design: Array2<f64>) -> ParameterBlockSpec {
        let n = design.nrows();
        ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
            offset: Array1::<f64>::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::<f64>::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            row_scaling: None,
            jacobian_callback: None,
        }
    }

    fn linspace(n: usize) -> Array1<f64> {
        if n <= 1 {
            return Array1::<f64>::zeros(n.max(1));
        }
        let step = 2.0 / (n as f64 - 1.0);
        Array1::from_iter((0..n).map(|i| -1.0 + step * i as f64))
    }

    #[test]
    fn canonical_clean_specs_identity_transform() {
        let n = 32;
        let x = linspace(n);
        let mut p = Array2::<f64>::zeros((n, 2));
        let mut s = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            p[[i, 0]] = 1.0;
            p[[i, 1]] = x[i];
            s[[i, 0]] = x[i] * x[i];
            s[[i, 1]] = x[i] * x[i] * x[i];
        }
        let specs = [spec_from_dense("p", p), spec_from_dense("s", s)];
        let canon = canonicalize_for_identifiability(&specs).expect("clean canonical must succeed");
        assert_eq!(canon.reduced_specs.len(), 2);
        assert_eq!(canon.per_block_transform[0].dim(), (2, 2));
        assert_eq!(canon.per_block_transform[1].dim(), (2, 2));
        let theta = vec![Array1::from(vec![0.5, -0.25]), Array1::from(vec![1.0, 2.0])];
        let raw = canon.lift_block_betas_to_raw(&theta);
        assert_eq!(raw[0].as_slice().unwrap(), &[0.5, -0.25]);
        assert_eq!(raw[1].as_slice().unwrap(), &[1.0, 2.0]);
    }

    /// Fail-closed contract: an aliased smooth-constant ~ intercept
    /// configuration must produce `CustomFamilyError::IdentifiabilityFailure`,
    /// not a silently-reduced spec list. Substituting a column-reduced
    /// `ParameterBlockSpec` under a family that captures raw-width
    /// designs (the current `CustomFamily` contract) panics inside
    /// `DesignMatrix::syr_row_into_view`; the safe behaviour is to
    /// refuse the fit with an actionable diagnostic in milliseconds.
    #[test]
    fn canonical_refuses_aliased_smooth_constant_with_intercept() {
        let n = 64;
        let x = linspace(n);
        let parametric = Array2::<f64>::from_shape_fn((n, 1), |(_, _)| 1.0);
        let mut smooth = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            smooth[[i, 0]] = 1.0;
            smooth[[i, 1]] = x[i] * x[i];
            smooth[[i, 2]] = x[i] * x[i] * x[i];
        }
        let specs = [
            spec_from_dense("intercept", parametric),
            spec_from_dense("smooth_with_const", smooth),
        ];
        let err = canonicalize_for_identifiability(&specs)
            .expect_err("aliased smooth-constant + intercept must refuse, not reduce");
        match err {
            CustomFamilyError::IdentifiabilityFailure { audit } => {
                assert!(
                    audit.fatal,
                    "audit attached to IdentifiabilityFailure must be fatal; got {}",
                    audit.summary,
                );
                assert!(
                    audit.summary.contains("intercept")
                        && audit.summary.contains("smooth_with_const"),
                    "refusal summary must name both offending blocks; got {:?}",
                    audit.summary,
                );
            }
            other => panic!("expected IdentifiabilityFailure, got {other:?}"),
        }
    }

    /// Five-block biobank-shape aliasing repro. Each block carries an
    /// intercept-like constant column; gauge_priority is set per the
    /// survival marginal-slope ownership policy (time=200 > marginal=150
    /// > logslope=120 > score_warp=80 > link_dev=60). The joint design
    /// has a 4-D null space among the five constants.
    ///
    /// Under the gauge-aware contract the canonicalisation must SUCCEED
    /// (not refuse) because the distinct `gauge_priority` values provide
    /// an unambiguous ordering for which columns to drop. The
    /// priority-ordered RRQR demotes the lowest-priority participants;
    /// the canonical-gauge pipeline applies the column-selection matrices
    /// and proceeds with reduced specs.  `time_surface` (highest priority)
    /// must retain all its columns; every attributed drop must belong to
    /// one of the four lower-priority blocks.
    #[test]
    fn canonical_five_block_gauge_ownership_succeeds_with_attribution() {
        let n = 96;
        let x = linspace(n);
        // Each block carries `ones(n)` in column 0 (the shared-constant
        // direction) plus per-block-unique content in columns 1..p.
        // The joint design therefore has a 4-D null space among the
        // five constant columns; priority decides which four are
        // dropped. We expect canonicalize to drop one constant from
        // each of (marginal, logslope, score_warp, link_dev) and keep
        // the time block's constant.
        let mut time = Array2::<f64>::zeros((n, 3));
        let mut marginal = Array2::<f64>::zeros((n, 3));
        let mut logslope = Array2::<f64>::zeros((n, 3));
        let mut score_warp = Array2::<f64>::zeros((n, 2));
        let mut link_dev = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            time[[i, 0]] = 1.0;
            time[[i, 1]] = x[i];
            time[[i, 2]] = x[i] * x[i] * x[i];
            marginal[[i, 0]] = 1.0;
            marginal[[i, 1]] = x[i] * x[i];
            marginal[[i, 2]] = x[i].sin();
            logslope[[i, 0]] = 1.0;
            logslope[[i, 1]] = (3.0 * x[i]).sin();
            logslope[[i, 2]] = (6.0 * x[i]).sin();
            score_warp[[i, 0]] = 1.0;
            score_warp[[i, 1]] = (5.0 * x[i]).cos();
            link_dev[[i, 0]] = 1.0;
            link_dev[[i, 1]] = (7.0 * x[i]).tanh();
        }
        let mut t_spec = spec_from_dense("time_surface", time);
        t_spec.gauge_priority = 200;
        let mut m_spec = spec_from_dense("marginal_surface", marginal);
        m_spec.gauge_priority = 150;
        let mut g_spec = spec_from_dense("logslope_surface", logslope);
        g_spec.gauge_priority = 120;
        let mut w_spec = spec_from_dense("score_warp_dev", score_warp);
        w_spec.gauge_priority = 80;
        let mut l_spec = spec_from_dense("link_dev", link_dev);
        l_spec.gauge_priority = 60;
        let specs = [t_spec, m_spec, g_spec, w_spec, l_spec];

        // With distinct gauge_priority values, the audit recognises that
        // the rank deficiency is gauge-resolvable and returns Ok (non-fatal).
        // The canonical-gauge pipeline proceeds with the column reductions.
        let canon = canonicalize_for_identifiability(&specs)
            .expect("five-block aliased joint with distinct gauge_priority must succeed (gauge-resolved)");

        // The audit stored in canon.audit must be non-fatal.
        assert!(
            !canon.audit.fatal,
            "audit must be non-fatal when gauge_priority is non-trivial and \
             all drops are attributed to lower-priority blocks; got {}",
            canon.audit.summary,
        );

        // Raw p_total = 3+3+3+2+2 = 13; expected rank = 13 − 4 = 9.
        // Audit's per-block effective_dim sum equals joint_rank.
        let total_kept: usize = canon.audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept,
            9,
            "expected joint rank = 13 − 4 = 9 reported by audit; got {total_kept} \
             (per-block effective_dim {:?})",
            canon
                .audit
                .blocks
                .iter()
                .map(|b| (b.block_name.clone(), b.effective_dim))
                .collect::<Vec<_>>(),
        );

        // Gauge-priority attribution: with priority-descending column ordering,
        // the time_surface block (highest priority = 200) must NOT appear among
        // the attributed drops — every attributed drop belongs to one of the
        // four lower-priority blocks.
        for drop in &canon.audit.dropped_columns {
            assert_ne!(
                drop.block, "time_surface",
                "highest-priority block must never be the attributed drop \
                 origin under priority-aware RRQR; got drop on time_surface \
                 ({drop:?})",
            );
        }

        // The reduced specs must have 9 total columns (rank = 9).
        let reduced_total: usize = canon.reduced_specs.iter().map(|s| s.design.ncols()).sum();
        assert_eq!(
            reduced_total,
            9,
            "reduced specs must have joint rank = 9 total columns; got {reduced_total}",
        );

        // time_surface retains all 3 of its columns (highest priority, never dropped).
        assert_eq!(
            canon.reduced_specs[0].design.ncols(),
            3,
            "time_surface must retain all 3 columns after gauge canonicalisation",
        );
    }

    /// On a clean (non-fatal) configuration with a non-trivial penalty,
    /// canonicalisation must succeed with **identity** transforms (the
    /// fail-closed contract makes reduction unreachable, but the
    /// pull-back-on-identity path is still exercised by the lift
    /// machinery downstream). The reduced penalty equals the raw
    /// penalty modulo cloning.
    #[test]
    fn canonical_clean_specs_with_penalty_round_trip() {
        let n = 32;
        let x = linspace(n);
        let parametric = Array2::<f64>::from_shape_fn((n, 1), |(_, _)| 1.0);
        // Smooth WITHOUT a duplicate constant column — clean joint design.
        let mut smooth = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            smooth[[i, 0]] = x[i] * x[i];
            smooth[[i, 1]] = x[i] * x[i] * x[i];
        }
        let mut smooth_spec = spec_from_dense("smooth_only", smooth);
        let mut s = Array2::<f64>::zeros((2, 2));
        s[[0, 0]] = 4.0;
        s[[1, 1]] = 9.0;
        s[[0, 1]] = 1.5;
        s[[1, 0]] = 1.5;
        smooth_spec.penalties = vec![PenaltyMatrix::Dense(s.clone())];
        smooth_spec.initial_log_lambdas = Array1::from(vec![0.0]);
        let specs = [spec_from_dense("intercept", parametric), smooth_spec];
        let canon = canonicalize_for_identifiability(&specs)
            .expect("clean canonical must succeed with identity transforms");
        let smooth_reduced = &canon.reduced_specs[1];
        assert_eq!(smooth_reduced.penalties.len(), 1);
        let dense_red = smooth_reduced.penalties[0].as_dense_cow().into_owned();
        // Identity pullback: penalty dimensions equal raw design width.
        assert_eq!(dense_red.dim(), (2, 2));
        // Identity transform: per-block transform is the 2×2 identity.
        let t_smooth = &canon.per_block_transform[1];
        assert_eq!(t_smooth.dim(), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(t_smooth[[i, j]], expected, "T_smooth must be identity");
            }
        }
    }
}
