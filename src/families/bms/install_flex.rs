use super::family::{
    append_deviation_function_penalty, require_probit_marginal_slope_link,
    resolve_deviation_operator_orders,
};
use super::*;

//      needed for both the audit gate and the compile step.
//   2. `audit_identifiability_channel_aware` — structural rank gate using
//      the BMS K=1 row Jacobian; catches full aliasing before any install.
//   3. `identifiability::compiler::compile` — W-metric Gram + eigendecomp,
//      produces the V selector and anchor-correction M.
//   4. Install V/M into the `DeviationRuntime` via `install_compiled_flex_block`,
//      rebuild the block's design + penalties, and return `FlexCompileOutcome`.
//
// The K=1 row-Jacobian math still runs through `identifiability::compiler::compile`,
// so there is exactly one cross-block residualisation math implementation in
// the codebase.

/// Tolerance (in constraint units) by which a structural-monotonicity slack
/// `A·β − b` may dip below zero before we treat it as a genuine violation
/// rather than floating-point round-off in the constraint inner products. The
/// constraint rows are O(1)-scaled deviation differences, so a few ulps of
/// accumulation sit comfortably under this bound while any real infeasibility
/// is orders of magnitude larger.
pub(crate) const MONOTONICITY_SLACK_TOL: f64 = -1e-10;

/// Assembled inputs for the BMS flex-block spec-builder → compile pipeline.
///
/// Produced by [`build_bms_flex_block_context`] and consumed by
/// [`install_compiled_flex_block_into_runtime`].
pub(crate) struct BmsFlexBlockContext {
    /// Densified anchor blocks in parametric-before-flex order.
    pub(super) anchor_dense_blocks: Vec<Array2<f64>>,
    /// Per-anchor predict-time tags (same order as `anchor_dense_blocks`).
    pub(super) anchor_components: Vec<super::deviation_runtime::AnchorComponentTag>,
    /// Horizontally stacked anchor matrix N_train (n × d_total).
    pub(super) n_train: Array2<f64>,
    /// `BernoulliDenseDesignOperator` per anchor, then one for the candidate
    /// (trailing). Indices align with `ordering`.
    pub(super) operators:
        Vec<std::sync::Arc<dyn crate::families::identifiability::compiler::RowJacobianOperator>>,
    /// Block-order tags parallel to `operators`.
    pub(super) ordering: Vec<crate::families::identifiability::compiler::BlockOrder>,
    /// W-metric row Hessian built from the validated `training_row_weights`.
    pub(super) row_hess:
        crate::families::identifiability::bernoulli::BernoulliRowHessian,
    /// Dense candidate basis at training rows (n × p_candidate), cached to
    /// avoid a second `design()` call after context construction.
    pub(super) candidate_design_dense: Array2<f64>,
    /// Number of training rows.
    pub(super) n: usize,
    /// Raw column count of the candidate block (= `candidate_design_dense.ncols()`).
    pub(super) p_candidate: usize,
    /// Total anchor columns (= `n_train.ncols()`).
    pub(super) d_total: usize,
}

/// Validate inputs, densify anchors, stack N_train, and assemble the
/// `BernoulliDenseDesignOperator` / `BlockOrder` / `BernoulliRowHessian`
/// vectors needed by both [`audit_identifiability_channel_aware`] and
/// [`identifiability::compiler::compile`].
///
/// Returns `Ok(None)` when the anchor union is empty (no-anchor fast path).
pub(crate) fn build_bms_flex_block_context(
    candidate: &DeviationPrepared,
    candidate_arg_at_training_rows: &Array1<f64>,
    parametric_anchors: &[(
        &DesignMatrix,
        super::deviation_runtime::ParametricAnchorBlock,
    )],
    flex_anchors: &[&Array2<f64>],
    training_row_weights: &Array1<f64>,
) -> Result<Option<BmsFlexBlockContext>, String> {
    use super::deviation_runtime::AnchorComponentTag;
    use crate::families::identifiability::bernoulli::{
        BernoulliDenseDesignOperator, BernoulliRowHessian,
    };
    use crate::families::identifiability::compiler::{BlockOrder, RowJacobianOperator};

    let candidate_design = candidate.runtime.design(candidate_arg_at_training_rows)?;
    let n = candidate_design.nrows();
    let p_candidate = candidate_design.ncols();

    if training_row_weights.len() != n {
        return Err(format!(
            "cross-block identifiability: training_row_weights length {} does not match candidate row count {}",
            training_row_weights.len(),
            n,
        ));
    }
    for (i, &w) in training_row_weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(format!(
                "cross-block identifiability: training_row_weights[{i}] = {w} is not finite/non-negative",
            ));
        }
    }

    // Densify parametric anchors (parametric-before-flex ordering).
    let mut anchor_dense_blocks: Vec<Array2<f64>> = Vec::new();
    let mut anchor_components: Vec<AnchorComponentTag> = Vec::new();
    let mut total_anchor_cols = 0usize;
    for (d, block_tag) in parametric_anchors {
        if d.nrows() != n {
            return Err(format!(
                "cross-block identifiability: parametric anchor has {} rows, candidate has {}",
                d.nrows(),
                n,
            ));
        }
        let p_a = d.ncols();
        if p_a == 0 {
            continue;
        }
        let dense = d
            .try_to_dense_arc("cross-block parametric anchor")?
            .as_ref()
            .clone();
        anchor_dense_blocks.push(dense);
        anchor_components.push(AnchorComponentTag::Parametric {
            block: *block_tag,
            ncols: p_a,
        });
        total_anchor_cols += p_a;
    }
    for a in flex_anchors {
        if a.nrows() != n {
            return Err(format!(
                "cross-block identifiability: flex anchor has {} rows, candidate has {}",
                a.nrows(),
                n,
            ));
        }
        let p_a = a.ncols();
        if p_a == 0 {
            continue;
        }
        anchor_dense_blocks.push((*a).clone());
        anchor_components.push(AnchorComponentTag::FlexEvaluation { ncols: p_a });
        total_anchor_cols += p_a;
    }
    if total_anchor_cols == 0 {
        return Ok(None);
    }

    let d_total = total_anchor_cols;
    let mut n_train = Array2::<f64>::zeros((n, d_total));
    {
        let mut col_offset = 0usize;
        for block in &anchor_dense_blocks {
            let bc = block.ncols();
            n_train
                .slice_mut(s![.., col_offset..col_offset + bc])
                .assign(block);
            col_offset += bc;
        }
    }

    // Build BernoulliDenseDesignOperator per anchor block, then one for the
    // candidate (trailing, BlockOrder::LinkDev).
    let mut operators: Vec<std::sync::Arc<dyn RowJacobianOperator>> =
        Vec::with_capacity(anchor_dense_blocks.len() + 1);
    let mut ordering: Vec<BlockOrder> = Vec::with_capacity(anchor_dense_blocks.len() + 1);
    for dense in &anchor_dense_blocks {
        operators.push(std::sync::Arc::new(BernoulliDenseDesignOperator::new(
            dense.clone(),
        )));
        ordering.push(BlockOrder::Marginal);
    }
    operators.push(std::sync::Arc::new(BernoulliDenseDesignOperator::new(
        candidate_design.clone(),
    )));
    ordering.push(BlockOrder::LinkDev);

    let row_hess = BernoulliRowHessian::from_row_weights(training_row_weights.clone());

    Ok(Some(BmsFlexBlockContext {
        anchor_dense_blocks,
        anchor_components,
        n_train,
        operators,
        ordering,
        row_hess,
        candidate_design_dense: candidate_design,
        n,
        p_candidate,
        d_total,
    }))
}

/// Outcome of [`install_compiled_flex_block_into_runtime`].
///
/// * `Reparameterised` — the candidate was reparameterised in place so
///   its column span at the n training rows is orthogonal to the anchor
///   union. Kept/dropped direction counts are emitted via the
///   `[BMS cross-block identifiability]` log at the construction site;
///   callers only need to know which branch they're in to decide whether
///   to keep the prepared block.
/// * `FullyAliased` — every direction in span(C) is reproducible by the
///   anchor union (`(I − P_A) C` has numerical rank zero). The candidate
///   carries no information independent of the anchors and the caller
///   should drop it from the design with a structured warning rather than
///   continue with a zero-rank block. The candidate is left in its
///   pre-call state (the reparameterisation is never applied) so the
///   caller can safely discard it.
#[derive(Debug)]
pub enum FlexCompileOutcome {
    Reparameterised,
    FullyAliased { reason: String },
}

/// Structured warning surfaced by the BMS family when a candidate flex
/// block is fully aliased by its anchor union and gets dropped.
#[derive(Clone, Debug)]
pub struct CrossBlockIdentifiabilityWarning {
    pub candidate_label: &'static str,
    pub anchor_summary: String,
    pub reason: String,
}

/// Enforce joint-design identifiability for a single flex block by
/// reparameterising its basis so its column span at the n training rows
/// is orthogonal to the union of every supplied anchor's column span.
///
/// This is the standard GAM `gam.side` convention generalised to multiple
/// anchor sources. After applying the resulting reparameterisation `T`,
/// the joint design `[anchor₁ | anchor₂ | … | candidate · T]` has full
/// numerical column rank, so `σ_min(joint H+S) ≥ λ_min(S) > 0` for every
/// β regardless of how the linear-predictor distribution shifts during
/// PIRLS. This eliminates the near-null direction in the joint penalised
/// Hessian that arises whenever the candidate flex block's column span
/// overlaps an anchor's column span (parametric span aliasing, flex-flex
/// aliasing, or both simultaneously).
///
/// # Math
///
/// Let `C ∈ ℝⁿˣᵖᶜ` be the candidate basis evaluated at the n training
/// rows, `A ∈ ℝⁿˣᵈ` the horizontally stacked parametric anchors, and
/// `W = diag(training_row_weights)`. The W-metric projector onto span(A)
/// is `P_A = A (AᵀWA)⁻¹ AᵀW`; the W-residualised candidate is
/// `C̃ = (I − P_A) C`. The joint reparameterisation
///
///   `Aβ_A + Cβ_C = A(β_A + Bβ_C) + (C − AB)β_C`     with `B = (AᵀWA)⁻¹AᵀWC`
///
/// is block-triangular, so dropping the columns of C̃ that have negligible
/// `C̃ᵀ W C̃` eigenvalues drops exactly the directions span(C) shares with
/// span(A) — under the actual Hessian row metric W = p(1−p), not the
/// uniform metric. Concretely: factor `AᵀWA = U Λ Uᵀ` and let
/// `R = U₊ Λ₊⁻½` so `Q_w = AR` is W-orthonormal under W. Then
/// `K_w = Q_wᵀ W C = Rᵀ AᵀW C` and `C̃ = C − AR · K_w`. After selecting
/// the kept eigenvector matrix V of `C̃ᵀ W C̃`, the residual
/// `M = R K_w V` is what each evaluated row subtracts:
/// `design_row(x) = pure_span_row(x) · V − n_row(x) · M`.
///
/// Why the old `null(AᵀC)` test is wrong: it asks "which candidate
/// directions are *already* exactly orthogonal to A?" rather than "what
/// remains after projecting A out?". `null(AᵀC) ≠ ∅` is NOT equivalent
/// to `span(C) ⊆ span(A)` — the equivalence is
/// `span(C) ⊆ span(A) ⇔ (I − P_A) C = 0`. Whenever d ≥ p_c (anchor
/// wider than candidate), `null(AᵀC)` is generically empty even if C
/// carries plenty of information independent of A.
///
/// # Cost
///
/// `AᵀWA` is `d × d` (d = total parametric anchor cols), built as one
/// matmul on the sqrt-W-scaled `A`. `K_w` is one `Q_wᵀ · (W^½ C)` matmul
/// of size `r × p_c`. `C̃ᵀ W C̃` is `p_c × p_c`. Two `eigh`s, both small
/// (`d ≲ a few dozen`, `p_c ≲ 50`); negligible against the per-cycle
/// dense Hessian build at large scale. `DesignMatrix` parametric
/// anchors are densified once into a contiguous `n × d` block (a few
/// dozen columns).
///
/// # `training_row_weights` (the W in the W-metric)
///
/// Callers **must** pass the IRLS Hessian row metric the joint Hessian
/// will see during PIRLS, not bare sample weights. For the probit-style
/// Bernoulli marginal slope family that is
/// `w[i] = sample_weights[i] · φ(η_i)² / (μ_i·(1−μ_i))` at a β-independent
/// pilot η. Passing uniform `spec.weights` instead makes A and C̃ merely
/// Euclidean-orthogonal: `Aᵀ W_pirls C̃` is nonzero at PIRLS time, the
/// joint Hessian carries a near-null direction along the W-metric alias,
/// and REML can drive the flex block's λ small enough that the alias
/// direction's joint Hessian eigenvalue collapses — manifesting as the
/// well-known runaway (rho≈2.0, constant `step_inf`, growing `beta_inf`,
/// inner loop hitting `inner_max_cycles` without satisfying the KKT
/// residual). See `pilot_irls_hessian_row_metric_at_eta`.
///
/// # No-op fast paths
///
/// * Anchor list is empty, or every anchor has zero parametric columns.
/// * `r = 0` — `AᵀWA` is numerically zero (degenerate weights).
///
/// # Hard error
///
/// `(I − P_A) C` has numerical rank zero — every direction in span(C) is
/// reproducible by the anchors up to tolerance. The candidate flex block
/// carries no information the parametric blocks do not already capture in
/// their unpenalised span; the diagnostic surfaces this explicitly rather
/// than letting the inner solver collide with the resulting rank-deficient
/// Hessian.
pub(crate) fn install_compiled_flex_block_into_runtime(
    candidate: &mut DeviationPrepared,
    candidate_arg_at_training_rows: &Array1<f64>,
    candidate_cfg: &DeviationBlockConfig,
    parametric_anchors: &[(
        &DesignMatrix,
        super::deviation_runtime::ParametricAnchorBlock,
    )],
    flex_anchors: &[&Array2<f64>],
    training_row_weights: &Array1<f64>,
) -> Result<FlexCompileOutcome, String> {
    use crate::families::identifiability::compiler::compile;
    use crate::solver::identifiability_audit::audit_identifiability_channel_aware;

    // Fast path: zero-column candidate carries nothing to residualise.
    let p_check = candidate
        .runtime
        .design(candidate_arg_at_training_rows)?
        .ncols();
    if p_check == 0 {
        return Ok(FlexCompileOutcome::Reparameterised);
    }

    // Step 1 — spec-builder: validate inputs, densify anchors, stack N_train,
    // assemble operators + row_hess. Returns None when the anchor union is
    // empty (no residualisation needed).
    let ctx = match build_bms_flex_block_context(
        candidate,
        candidate_arg_at_training_rows,
        parametric_anchors,
        flex_anchors,
        training_row_weights,
    )? {
        None => {
            // No anchors — the candidate's per-block smoothness-null-space
            // drop already handles intra-block aliasing.
            return Ok(FlexCompileOutcome::Reparameterised);
        }
        Some(c) => c,
    };
    let BmsFlexBlockContext {
        anchor_dense_blocks,
        anchor_components,
        n_train,
        operators,
        ordering,
        row_hess,
        candidate_design_dense,
        n,
        p_candidate,
        d_total,
    } = ctx;

    // Step 2 — audit gate: `audit_identifiability_channel_aware` uses the
    // structural BMS K=1 row Jacobian to detect full aliasing before any
    // install. A fatal audit with effective_dim == 0 for the trailing
    // (candidate) block means every direction in span(C) is reproducible by
    // the anchor union; return FullyAliased immediately without touching the
    // runtime.
    let audit = audit_identifiability_channel_aware(
        &{
            // Build minimal ParameterBlockSpec wrappers so the audit can record
            // block names and column counts. The specs are audit-only; no
            // penalties or log-lambdas are needed here.
            let mut specs = Vec::with_capacity(anchor_dense_blocks.len() + 1);
            for (idx, dense) in anchor_dense_blocks.iter().enumerate() {
                specs.push(crate::custom_family::ParameterBlockSpec {
                    name: format!("anchor_{idx}"),
                    design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                        dense.clone(),
                    )),
                    offset: Array1::<f64>::zeros(n),
                    penalties: Vec::new(),
                    nullspace_dims: Vec::new(),
                    initial_log_lambdas: Array1::<f64>::zeros(0),
                    initial_beta: None,
                    gauge_priority: super::block_specs::GAUGE_PRIORITY_ANCHOR,
                    jacobian_callback: None,
                    stacked_design: None,
                    stacked_offset: None,
                });
            }
            specs.push(crate::custom_family::ParameterBlockSpec {
                name: "candidate_flex".to_string(),
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    candidate_design_dense.clone(),
                )),
                offset: Array1::<f64>::zeros(n),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: Array1::<f64>::zeros(0),
                initial_beta: None,
                gauge_priority: super::block_specs::GAUGE_PRIORITY_CANDIDATE_FLEX,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            });
            specs
        },
        &operators,
        &row_hess,
    )
    .map_err(|e| format!("cross-block identifiability audit failed: {e}"))?;

    if audit.fatal {
        let candidate_block = audit.blocks.last();
        let effective = candidate_block.map(|b| b.effective_dim).unwrap_or(0);
        if effective == 0 {
            let reason = format!(
                "candidate flex basis ({p_candidate} cols) has zero directions remaining after \
                 W-metric residualisation against the anchor union ({d_total} anchor cols) at the \
                 {n} training rows. The channel-aware audit collapses every direction in \
                 span(C) — every direction in span(C) is reproducible by the anchor union up to \
                 numerical tolerance. Drop the flex block or remove the anchor term that reproduces \
                 its argument; knot count is NOT the relevant lever for this failure mode.",
            );
            return Ok(FlexCompileOutcome::FullyAliased { reason });
        }
    }

    // Step 3 — W-metric compile: Gram + eigendecomp → V selector (t_lw) and
    // anchor-correction M. The compiler runs at K=1 (BMS row primary state =
    // scalar η) using `BernoulliRowHessian` as the row metric. This is the
    // single math implementation of the cross-block W-metric residualisation.
    let compiled = compile(&operators, &row_hess, &ordering).map_err(|e| {
        format!(
            "cross-block identifiability: compile failed (n={n}, d_total={d_total}, p_c={p_candidate}): {e}",
        )
    })?;
    let candidate_compiled = compiled
        .blocks
        .last()
        .ok_or_else(|| "cross-block identifiability: compile returned no blocks".to_string())?;
    let k_kept = candidate_compiled.t_lw.ncols();
    if k_kept == 0 {
        let reason = format!(
            "candidate flex basis ({p_candidate} cols) has zero directions remaining after \
             W-metric residualisation against the anchor union ({d_total} anchor cols) at the \
             {n} training rows. The compiler's joint pre-fit audit collapses every direction in \
             span(C) — every direction in span(C) is reproducible by the anchor union up to \
             numerical tolerance. Drop the flex block or remove the anchor term that reproduces \
             its argument; knot count is NOT the relevant lever for this failure mode.",
        );
        return Ok(FlexCompileOutcome::FullyAliased { reason });
    }
    // Shape contract: compile() must emit (d_total × k_kept) anchor_correction
    // for the trailing candidate block.
    {
        let m = candidate_compiled
            .anchor_correction
            .as_ref()
            .ok_or_else(|| {
                "cross-block identifiability: compile returned no anchor_correction for the \
             candidate block (expected for trailing block with non-empty anchor union)"
                    .to_string()
            })?;
        if m.nrows() != d_total || m.ncols() != k_kept {
            return Err(format!(
                "cross-block identifiability: anchor_correction shape {}×{} does not match \
                 expected d_total={d_total} × k_kept={k_kept}",
                m.nrows(),
                m.ncols(),
            ));
        }
    }

    // Step 4 — install: wrap compiled output into the runtime as an
    // InstalledFlexBlock (anchor_correction M + anchor_components tags),
    // cache N_train, apply selector V to span_c{0..3} + boundary/monotonicity
    // rows, then rebuild the block's design + penalties in the new basis.
    candidate.runtime.install_compiled_flex_block(
        candidate_compiled,
        anchor_components,
        n_train,
    )?;
    let new_design = candidate
        .runtime
        .design_at_training_with_residual(candidate_arg_at_training_rows)?;
    let new_p = new_design.ncols();
    assert_eq!(new_p, k_kept);
    assert_eq!(new_design.nrows(), n);
    candidate.block.design =
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(new_design));
    candidate.block.penalties.clear();
    candidate.block.nullspace_dims.clear();
    let penalty_orders = resolve_deviation_operator_orders(candidate_cfg)?;
    for order in penalty_orders {
        append_deviation_function_penalty(&mut candidate.block, &candidate.runtime, order)?;
    }
    if candidate_cfg.double_penalty {
        append_deviation_function_penalty(&mut candidate.block, &candidate.runtime, 0)?;
    }
    candidate.block.initial_beta = Some(Array1::zeros(new_p));

    log::info!(
        "[BMS cross-block identifiability] flex block reparameterised via compiler: \
         kept {kept}/{p_candidate} directions (anchor union cols={d_total}, training rows={n}, \
         joint_rank={joint_rank}, dropped_by_audit={dropped})",
        kept = new_p,
        p_candidate = p_candidate,
        d_total = d_total,
        n = n,
        joint_rank = compiled.joint_rank,
        dropped = compiled.dropped.len(),
    );
    Ok(FlexCompileOutcome::Reparameterised)
}

pub(crate) fn project_monotone_feasible_beta(
    runtime: &DeviationRuntime,
    current: &Array1<f64>,
    proposed: &Array1<f64>,
    label: &str,
) -> Result<Array1<f64>, String> {
    if current.len() != runtime.basis_dim() {
        return Err(format!(
            "{label} monotone projection current length mismatch: current={}, expected={}",
            current.len(),
            runtime.basis_dim()
        ));
    }
    if proposed.len() != runtime.basis_dim() {
        return Err(format!(
            "{label} monotone projection length mismatch: proposed={}, expected={}",
            proposed.len(),
            runtime.basis_dim()
        ));
    }
    for (idx, value) in current.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label} current coefficient {idx} is non-finite"));
        }
    }
    for (idx, value) in proposed.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label} coefficient {idx} is non-finite"));
        }
    }
    runtime.monotonicity_feasible(current, &format!("{label} current beta"))?;
    if runtime
        .monotonicity_feasible(proposed, &format!("{label} proposed beta"))
        .is_ok()
    {
        return Ok(proposed.clone());
    }

    let constraints = runtime.structural_monotonicity_constraints();
    let alpha = max_linear_constraint_segment_alpha(current, proposed, &constraints, label)?;
    let direction = proposed - current;
    let candidate = current + &direction.mapv(|value| value * alpha);
    validate_monotone_structural_feasible(runtime, &candidate, &format!("{label} projected beta"))?;
    Ok(candidate)
}

pub(crate) fn validate_monotone_structural_feasible(
    runtime: &DeviationRuntime,
    beta: &Array1<f64>,
    label: &str,
) -> Result<(), String> {
    let constraints = runtime.structural_monotonicity_constraints();
    if beta.len() != constraints.a.ncols() {
        return Err(format!(
            "{label} structural monotonicity length mismatch: beta={}, expected={}",
            beta.len(),
            constraints.a.ncols()
        ));
    }
    if beta.iter().any(|value| !value.is_finite()) {
        let bad = beta
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
            .map(|(idx, value)| format!("{label} coefficient {idx} is non-finite ({value})"))
            .unwrap_or_else(|| format!("{label} coefficient is non-finite"));
        return Err(bad);
    }
    let slack = constraints.a.dot(beta) - &constraints.b;
    let mut min_slack = f64::INFINITY;
    let mut min_row = 0usize;
    for (row, &value) in slack.iter().enumerate() {
        if value < min_slack {
            min_slack = value;
            min_row = row;
        }
    }
    if min_slack < MONOTONICITY_SLACK_TOL {
        return Err(format!(
            "{label} violates structural monotonicity row {min_row}: slack={min_slack:.3e}; \
             deviation monotonicity must be enforced by analytic linear constraints, not post-update projection"
        ));
    }
    runtime.monotonicity_feasible(beta, label)
}

pub(crate) fn max_linear_constraint_segment_alpha(
    current: &Array1<f64>,
    proposed: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    label: &str,
) -> Result<f64, String> {
    if current.len() != proposed.len() || current.len() != constraints.a.ncols() {
        return Err(format!(
            "{label} linear-constraint segment dimension mismatch: current={}, proposed={}, constraints={}",
            current.len(),
            proposed.len(),
            constraints.a.ncols()
        ));
    }
    if constraints.a.nrows() != constraints.b.len() {
        return Err(format!(
            "{label} linear-constraint segment row mismatch: A rows={}, b len={}",
            constraints.a.nrows(),
            constraints.b.len()
        ));
    }
    let direction = proposed - current;
    let mut alpha = 1.0_f64;
    for row in 0..constraints.a.nrows() {
        let a_row = constraints.a.row(row);
        let slack = a_row.dot(current) - constraints.b[row];
        if slack < MONOTONICITY_SLACK_TOL {
            return Err(format!(
                "{label} current beta violates structural monotonicity row {row}: slack={slack:.3e}"
            ));
        }
        let drift = a_row.dot(&direction);
        if drift < 0.0 {
            alpha = alpha.min((slack / -drift).clamp(0.0, 1.0));
        }
    }
    Ok(alpha.clamp(0.0, 1.0))
}

pub(super) fn validate_spec(
    data: ArrayView2<'_, f64>,
    spec: &BernoulliMarginalSlopeTermSpec,
) -> Result<(), String> {
    let n = data.nrows();
    if spec.y.len() != n
        || spec.weights.len() != n
        || spec.z.len() != n
        || spec.marginal_offset.len() != n
        || spec.logslope_offset.len() != n
    {
        return Err(format!(
            "bernoulli-marginal-slope row mismatch: data={}, y={}, weights={}, z={}, marginal_offset={}, logslope_offset={}",
            n,
            spec.y.len(),
            spec.weights.len(),
            spec.z.len(),
            spec.marginal_offset.len(),
            spec.logslope_offset.len()
        ));
    }
    if spec
        .y
        .iter()
        .any(|&yi| !yi.is_finite() || ((yi - 0.0).abs() > 1e-9 && (yi - 1.0).abs() > 1e-9))
    {
        return Err("bernoulli-marginal-slope requires binary y in {0,1}".to_string());
    }
    if spec.weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err("bernoulli-marginal-slope requires finite non-negative weights".to_string());
    }
    if spec.z.iter().any(|&zi| !zi.is_finite()) {
        return Err("bernoulli-marginal-slope requires finite z values".to_string());
    }
    if spec.marginal_offset.iter().any(|&value| !value.is_finite()) {
        return Err("bernoulli-marginal-slope requires finite marginal offsets".to_string());
    }
    if spec.logslope_offset.iter().any(|&value| !value.is_finite()) {
        return Err("bernoulli-marginal-slope requires finite logslope offsets".to_string());
    }
    if let Some(jac) = spec.score_influence_jacobian.as_ref() {
        // #461: the absorbed Stage-1 influence Jacobian J = ∂z/∂θ₁ must be an
        // n×p₁ matrix of finite entries co-indexed with the training rows.
        if jac.nrows() != n {
            return Err(format!(
                "bernoulli-marginal-slope score_influence_jacobian has {} rows, expected {n}",
                jac.nrows()
            ));
        }
        if jac.iter().any(|&value| !value.is_finite()) {
            return Err(
                "bernoulli-marginal-slope score_influence_jacobian must be finite".to_string(),
            );
        }
    }
    require_probit_marginal_slope_link(&spec.base_link, "bernoulli-marginal-slope")?;
    spec.frailty.validate_for_marginal_slope()?;
    match &spec.frailty {
        FrailtySpec::None => {}
        FrailtySpec::GaussianShift { sigma_fixed } => {
            if let Some(sigma) = sigma_fixed
                && (!sigma.is_finite() || *sigma < 0.0)
            {
                return Err(format!(
                    "bernoulli-marginal-slope requires GaussianShift sigma >= 0, got {sigma}"
                ));
            }
        }
        FrailtySpec::HazardMultiplier { .. } => {
            return Err(
                "bernoulli-marginal-slope does not support FrailtySpec::HazardMultiplier"
                    .to_string(),
            );
        }
    }
    Ok(())
}
