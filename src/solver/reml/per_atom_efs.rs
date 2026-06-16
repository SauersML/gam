//! Frontier ρ-scaling: per-atom decoupled Extended Fellner–Schall (EFS) as the
//! primary outer iteration, with a coupled correction restricted to
//! shared-border axes and a matrix-free θ-HVP for axes that need true
//! second-order coupling (issue #986, building on the #740 θ-HVP design).
//!
//! # Why this module exists
//!
//! ARD-per-atom assigns one smoothing coordinate ρ per dictionary atom. At
//! frontier scale that is `10^4`–`10^5` coordinates. A dense outer
//! quasi-Newton (ARC/BFGS) over that ρ-vector is impossible: it materializes an
//! O(K²) outer Hessian and factorizes it every accepted step. The standard
//! [`crate::solver::estimate::reml::reml_outer_engine::compute_efs_update`] is already
//! *per-coordinate decoupled* in its arithmetic — each ρ_i step is
//! `log(1 − 2·g_full[i]/q_eff_i)`, a function of that atom's own gradient entry
//! and penalty-quadratic curvature scale only — so EFS is the natural frontier
//! primary. This module drives that decoupled fixed point directly, parallel
//! across atoms, and never assembles any K×K object.
//!
//! Three layers, in increasing cost and decreasing breadth:
//!
//! 1. **Per-atom decoupled EFS** (every atom, every iteration). The outer
//!    objective's `eval_efs` hook runs one inner P-IRLS solve and returns the
//!    full per-coordinate step vector. We apply each atom's own multiplicative
//!    log-λ step, with a whole-vector cost line search (Wood–Fasiolo give
//!    ascent in the EFS direction but not full-step monotonicity). The per-atom
//!    arithmetic *after* the shared inner solve is embarrassingly parallel; we
//!    fan it across the existing rayon pool, honoring the OnceLock+nested-rayon
//!    rule (any `get_or_init` is warmed at the top level before the
//!    `into_par_iter`, never inside it).
//!
//! 2. **Shared-border coupled correction** (only the few border axes). Most
//!    atoms are penalty-block-disjoint: their ρ updates do not interact, so the
//!    decoupled step is exact for them. A small set of axes *do* interact —
//!    those whose penalty blocks overlap through the arrow border (shared
//!    columns / a shared global block). For just those `m ≪ K` axes we solve
//!    one tiny `m × m` coupled Newton correction using the matrix-free θ-HVP to
//!    fill the restricted outer-Hessian sub-block, then factorize that `m × m`
//!    system (never the full K × K).
//!
//! 3. **Matrix-free θ-HVP** (#740). For a direction `v` over the border axes,
//!    `H_outer · v` is computed *without* assembling the O(K²) coordinate-pair
//!    Hessian. When the family supplies an exact outer-Hessian operator
//!    (`HessianResult::Operator`), its `matvec` already realizes the
//!    IFT-corrected action `β̇ = −H⁻¹ (∂g/∂θ)·v` plus the logdet directional
//!    trace — exactly the #740 product — through one inner solve per matvec.
//!    When no operator is available we fall back to a central finite difference
//!    of the outer gradient along `v` (two `eval` calls), which is still
//!    matrix-free and restricted to the border directions only.
//!
//! # FD-consistency / reduction to the coupled objective at small K
//!
//! At small K the per-atom step reduces to exactly the coupled EFS step:
//! `compute_efs_update` already produces the same per-coordinate
//! `log(1 − 2·g_full[i]/q_eff_i)` regardless of K, and the shared-border
//! correction is a Newton step on the *same* outer gradient `g` that the
//! coupled path uses (it nulls out when `g` is zero). Concretely, when the
//! border set is the whole ρ-vector (small K), the layer-2 correction is the
//! exact dense Newton step on `g`, and when `g = 0` (a stationary point of the
//! coupled objective) every layer's step is zero. This is the property a
//! finite-difference check on the coupled objective would verify; it is a
//! design invariant of routing every layer through the same `eval`/`eval_efs`
//! hooks that the dense path consumes, not a separately maintained surrogate.

use crate::linalg::faer_ndarray::{FaerArrayView, factorize_symmetricwith_fallback};
use crate::matrix::FactorizedSystem;
use crate::solver::estimate::EstimationError;
use crate::solver::rho_optimizer::{
    HessianResult, OuterCapability, OuterHessianOperator, OuterObjective, OuterPlan, OuterResult,
};
use faer::Side;
use ndarray::{Array1, Array2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;

/// Smallest ρ-dimension at which the per-atom decoupled EFS primary outranks
/// the dense quasi-Newton path.
///
/// Below this the dense ARC/BFGS outer is affordable (its O(K²) Hessian is
/// `≤ 64² = 4096` entries) and its exact second-order geometry is preferable.
/// At or above it the dense path's per-step factorization dominates the inner
/// solve cost, and the per-atom decoupled fixed point — one inner solve plus an
/// embarrassingly parallel O(K) step assembly — is the only path that scales to
/// the `10^4`–`10^5` ARD-per-atom regime. The threshold is auto-derived from
/// the coordinate count alone; there is no flag.
pub const PER_ATOM_EFS_MIN_RHO_DIM: usize = 64;

/// Maximum absolute step in log-λ for any single per-atom update, mirroring the
/// `EFS_MAX_STEP` clamp the unified EFS path applies, so one outer iteration
/// cannot move any `λ_i` by more than `exp(5)` ≈ 148×.
pub(crate) const PER_ATOM_MAX_STEP: f64 = 5.0;

/// Whole-vector backtracking halvings for the per-atom EFS line search.
pub(crate) const PER_ATOM_MAX_BACKTRACK: usize = 8;

/// Step components below this magnitude (in θ-space) are treated as numerically
/// zero for convergence and line-search purposes.
pub(crate) const PER_ATOM_NEGLIGIBLE_STEP: f64 = 1e-12;

/// Relative tolerance for the descent condition during backtracking; matches
/// the unified EFS path so ULP-level cost noise near a fixed point does not
/// trigger spurious backtracking.
pub(crate) const PER_ATOM_COST_DESCENT_TOL: f64 = 1e-12;

/// Finite-difference probe magnitude (in θ-space) for the θ-HVP fallback when
/// no exact outer-Hessian operator is available. Central difference, so the
/// error is O(h²); `1e-4` balances truncation against the inner-solve noise
/// floor of the outer gradient.
pub(crate) const THETA_HVP_FD_STEP: f64 = 1.0e-4;

/// Auto-switch threshold predicate: is this problem in the frontier ρ-scaling
/// regime where the per-atom decoupled EFS primary should take over from the
/// dense outer?
///
/// Magic-by-default: derived from the ρ-dimension only. The caller passes the
/// number of penalty-like smoothing coordinates (`rho_dim`); no flag, no env.
#[inline]
pub fn is_frontier_rho_scale(rho_dim: usize) -> bool {
    rho_dim >= PER_ATOM_EFS_MIN_RHO_DIM
}

/// Whether a given outer capability is eligible for — and large enough to
/// benefit from — the per-atom decoupled EFS primary.
///
/// Requires: all coordinates penalty-like (no ψ design-moving coords, which
/// EFS cannot resolve; those still route to HybridEFS / BFGS), a working
/// fixed-point hook (`eval_efs`), fixed-point not disabled by the caller, and
/// a frontier-scale ρ-dimension.
pub fn per_atom_efs_eligible(cap: &OuterCapability) -> bool {
    cap.all_penalty_like()
        && cap.fixed_point_available
        && !cap.disable_fixed_point
        && is_frontier_rho_scale(cap.theta_layout().rho_dim())
}

/// Outcome of the per-atom decoupled EFS outer iteration.
pub struct PerAtomEfsResult {
    /// Optimized log-smoothing parameters.
    pub rho: Array1<f64>,
    /// Final REML/LAML cost.
    pub final_value: f64,
    /// Total outer iterations executed.
    pub iterations: usize,
    /// Infinity-norm of the final applied step (proxy for the outer gradient
    /// residual on the multiplicative fixed point).
    pub final_step_inf_norm: f64,
    /// Whether the fixed point converged within tolerance.
    pub converged: bool,
}

impl PerAtomEfsResult {
    /// Lift the per-atom result into the shared [`OuterResult`] shape the
    /// generic runner returns, so the coordinator can route this primary
    /// through the same downstream consumers as the dense path.
    pub fn into_outer_result(self, plan_used: OuterPlan) -> OuterResult {
        OuterResult {
            rho: self.rho,
            final_value: self.final_value,
            iterations: self.iterations,
            final_grad_norm: Some(self.final_step_inf_norm),
            final_gradient: None,
            final_hessian: None,
            converged: self.converged,
            plan_used,
            operator_trust_radius: None,
            operator_stop_reason: None,
            criterion_certificate: None,
            rho_uncertainty_diagnostic: None,
        }
    }
}

/// Shared-border topology over the ρ-axes.
///
/// `border_axes` are the (few) coordinates whose penalty blocks overlap through
/// the arrow border — a shared global block or shared design columns — so their
/// EFS updates are *not* mutually decoupled and benefit from a coupled
/// correction. Every other axis is block-disjoint and exact under the per-atom
/// step alone.
#[derive(Clone, Debug)]
pub struct SharedBorderTopology {
    pub(crate) border_axes: Vec<usize>,
    pub(crate) rho_dim: usize,
}

impl SharedBorderTopology {
    /// No shared border: every atom is block-disjoint and the decoupled
    /// per-atom step is exact. This is the common ARD-per-atom case where each
    /// atom owns a private penalty block.
    pub fn disjoint(rho_dim: usize) -> Self {
        Self {
            border_axes: Vec::new(),
            rho_dim,
        }
    }

    /// A topology with a populated shared border: `axes` are the ρ-coordinates
    /// whose penalty blocks overlap through the arrow border (shared global
    /// block / shared design columns) and therefore receive the coupled
    /// `m × m` Newton correction on top of their decoupled step.
    ///
    /// Axes are sorted and deduplicated; an out-of-range axis is rejected
    /// loudly. A caller passing every axis (`axes == 0..rho_dim`) recovers the
    /// exact dense Newton correction on the full ρ-vector — the small-K
    /// FD-consistency configuration the module docs describe.
    pub fn with_border_axes(rho_dim: usize, axes: Vec<usize>) -> Result<Self, String> {
        let mut border_axes = axes;
        border_axes.sort_unstable();
        border_axes.dedup();
        if let Some(&last) = border_axes.last() {
            if last >= rho_dim {
                return Err(format!(
                    "SharedBorderTopology: border axis {last} out of range (rho_dim = {rho_dim})"
                ));
            }
        }
        Ok(Self {
            border_axes,
            rho_dim,
        })
    }

    /// Indices of the shared-border axes (sorted, deduplicated, in range).
    #[inline]
    pub fn border_axes(&self) -> &[usize] {
        &self.border_axes
    }

    /// Number of shared-border axes `m`. The coupled correction solves an
    /// `m × m` system; when `m == 0` it is skipped entirely.
    #[inline]
    pub fn border_count(&self) -> usize {
        self.border_axes.len()
    }

    #[inline]
    pub(crate) fn rho_dim(&self) -> usize {
        self.rho_dim
    }
}

/// Configuration for the per-atom decoupled EFS outer iteration. All values are
/// auto-derived from the outer config the dense path already carries; there is
/// no caller-facing flag.
#[derive(Clone, Debug)]
pub struct PerAtomEfsConfig {
    /// Step-norm convergence tolerance in θ-space.
    pub tolerance: f64,
    /// Maximum outer iterations.
    pub max_iter: usize,
    /// Per-coordinate lower/upper bounds on ρ.
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
}

impl PerAtomEfsConfig {
    /// Build from the bounds and budget the generic outer config supplies.
    pub fn new(tolerance: f64, max_iter: usize, lower: Array1<f64>, upper: Array1<f64>) -> Self {
        Self {
            tolerance,
            max_iter,
            lower,
            upper,
        }
    }
}

#[inline]
pub(crate) fn project_axis(value: f64, lo: f64, hi: f64) -> f64 {
    value.max(lo).min(hi)
}

pub(crate) fn project_to_bounds(rho: &Array1<f64>, cfg: &PerAtomEfsConfig) -> Array1<f64> {
    let mut out = rho.clone();
    for i in 0..out.len() {
        out[i] = project_axis(out[i], cfg.lower[i], cfg.upper[i]);
    }
    out
}

/// Clamp a raw multiplicative log-λ step to the per-atom maximum, treating
/// non-finite raw steps as zero (the coordinate makes no move this iteration).
#[inline]
pub(crate) fn sanitize_step(raw: f64) -> f64 {
    if raw.is_finite() {
        raw.clamp(-PER_ATOM_MAX_STEP, PER_ATOM_MAX_STEP)
    } else {
        0.0
    }
}

/// Matrix-free θ-HVP (#740): `v ↦ H_outer · v` over the full ρ-vector, computed
/// without assembling the O(K²) coordinate-pair outer Hessian.
///
/// Two realizations, both matrix-free:
///
/// - **Exact operator.** When `eval` exposed an outer-Hessian operator
///   (`HessianResult::Operator`), its `matvec` already realizes the
///   IFT-corrected action — `−H⁻¹ (∂g/∂θ)·v` plus the logdet directional trace,
///   the #740 product — via one inner solve per matvec. We forward `v` to it
///   directly.
/// - **Finite-difference fallback.** With no operator, approximate the action
///   by a central difference of the outer gradient along `v`:
///   `H·v ≈ (g(ρ + h·v) − g(ρ − h·v)) / (2h)`. Two `eval` calls, O(h²) error,
///   still matrix-free and (as used by the border correction) probed only along
///   the few border directions.
///
/// `operator` is the operator captured from the most recent full `eval` at
/// `rho`; `eval_at` evaluates the outer gradient at an arbitrary ρ for the FD
/// branch. The returned vector has the full ρ-dimension.
pub fn theta_hvp_matrix_free(
    operator: Option<&Arc<dyn OuterHessianOperator>>,
    rho: &Array1<f64>,
    v: &Array1<f64>,
    mut eval_at: impl FnMut(&Array1<f64>) -> Result<Array1<f64>, EstimationError>,
) -> Result<Array1<f64>, EstimationError> {
    if let Some(op) = operator {
        if op.dim() == v.len() {
            return op.matvec(v).map_err(|reason| {
                EstimationError::RemlOptimizationFailed(format!(
                    "per-atom θ-HVP operator matvec failed (dim={}): {reason}",
                    v.len()
                ))
            });
        }
    }

    // Central-difference fallback along v.
    let h = THETA_HVP_FD_STEP;
    let v_norm = v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if v_norm <= PER_ATOM_NEGLIGIBLE_STEP {
        return Ok(Array1::zeros(v.len()));
    }
    let mut plus = rho.clone();
    let mut minus = rho.clone();
    for i in 0..rho.len() {
        plus[i] += h * v[i];
        minus[i] -= h * v[i];
    }
    let g_plus = eval_at(&plus)?;
    let g_minus = eval_at(&minus)?;
    if g_plus.len() != v.len() || g_minus.len() != v.len() {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "per-atom θ-HVP FD gradient length mismatch: got ({}, {}), expected {}",
            g_plus.len(),
            g_minus.len(),
            v.len()
        )));
    }
    let mut hv = Array1::<f64>::zeros(v.len());
    for i in 0..v.len() {
        hv[i] = (g_plus[i] - g_minus[i]) / (2.0 * h);
    }
    Ok(hv)
}

/// Assemble the restricted `m × m` outer-Hessian sub-block over the
/// shared-border axes by probing the matrix-free θ-HVP with the `m` border
/// basis directions, then symmetrizing.
///
/// This is the only place a dense matrix is formed, and it is `m × m` with
/// `m ≪ K` (the border count), never `K × K`. When the exact operator is
/// available each probe is one operator matvec; otherwise each probe is one
/// central FD pair. The `m` probes are independent, so they fan across rayon —
/// but only when an exact operator backs them (the FD branch mutates shared
/// objective state through `eval_at` and must stay sequential).
pub(crate) fn border_hessian_block(
    topology: &SharedBorderTopology,
    operator: Option<&Arc<dyn OuterHessianOperator>>,
    rho: &Array1<f64>,
    eval_grad_at: &(dyn Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError> + Sync),
) -> Result<Array2<f64>, EstimationError> {
    let m = topology.border_count();
    let border = topology.border_axes();
    let mut block = Array2::<f64>::zeros((m, m));

    if let Some(op) = operator {
        if op.dim() == rho.len() {
            // Exact operator path: probes are independent operator matvecs;
            // fan across rayon. No OnceLock get_or_init is triggered inside the
            // probe closure (matvec is a pure linear action on a captured
            // factorization), so the nested-rayon deadlock rule is satisfied.
            let cols: Result<Vec<(usize, Array1<f64>)>, EstimationError> = (0..m)
                .into_par_iter()
                .map(|j| {
                    let mut e_j = Array1::<f64>::zeros(rho.len());
                    e_j[border[j]] = 1.0;
                    let hv = op.matvec(&e_j).map_err(|reason| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "per-atom border θ-HVP operator matvec failed: {reason}"
                        ))
                    })?;
                    Ok((j, hv))
                })
                .collect();
            for (j, hv) in cols? {
                for (row, &axis) in border.iter().enumerate() {
                    block[[row, j]] = hv[axis];
                }
            }
        } else {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "per-atom border θ-HVP operator dim {} != rho_dim {}",
                op.dim(),
                rho.len()
            )));
        }
    } else {
        // FD fallback: each probe mutates objective state through eval_grad_at
        // (one inner solve per ρ), so the probes are sequential. Still
        // matrix-free and restricted to the m border directions.
        for j in 0..m {
            let mut e_j = Array1::<f64>::zeros(rho.len());
            e_j[border[j]] = 1.0;
            let hv = theta_hvp_matrix_free(None, rho, &e_j, eval_grad_at)?;
            for (row, &axis) in border.iter().enumerate() {
                block[[row, j]] = hv[axis];
            }
        }
    }

    // Symmetrize against round-off / FD asymmetry.
    for r in 0..m {
        for c in (r + 1)..m {
            let s = 0.5 * (block[[r, c]] + block[[c, r]]);
            block[[r, c]] = s;
            block[[c, r]] = s;
        }
    }
    Ok(block)
}

/// Solve the restricted coupled Newton correction on the shared-border axes:
/// `Δρ_border = − (H_bb + ridge·I)⁻¹ · g_border`, where `H_bb` is the `m × m`
/// border outer-Hessian block from [`border_hessian_block`] and `g_border` is
/// the outer gradient restricted to the border axes.
///
/// Returns a full-length ρ step that is zero off the border. A small adaptive
/// ridge guards an indefinite/ill-conditioned border block (the border may sit
/// where the EFS multiplicative surrogate's PSD assumption is weakest); the
/// ridge scales with the block's diagonal magnitude so it is basis-aware and
/// vanishes for a well-conditioned block.
pub(crate) fn shared_border_correction(
    topology: &SharedBorderTopology,
    operator: Option<&Arc<dyn OuterHessianOperator>>,
    rho: &Array1<f64>,
    gradient: &Array1<f64>,
    eval_grad_at: &(dyn Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError> + Sync),
) -> Result<Array1<f64>, EstimationError> {
    let m = topology.border_count();
    let mut step = Array1::<f64>::zeros(topology.rho_dim());
    if m == 0 {
        return Ok(step);
    }
    let border = topology.border_axes();

    let g_border_inf = border
        .iter()
        .map(|&i| gradient[i].abs())
        .fold(0.0_f64, f64::max);
    if g_border_inf <= PER_ATOM_NEGLIGIBLE_STEP {
        // Border axes are already stationary; the coupled correction nulls out.
        // This is the FD-consistency invariant: zero outer gradient ⇒ zero
        // correction, matching the coupled objective at any stationary point.
        return Ok(step);
    }

    let mut block = border_hessian_block(topology, operator, rho, eval_grad_at)?;

    // Adaptive ridge from the mean absolute diagonal so the regularization is
    // dimensionally consistent with the block's curvature scale.
    let diag_scale = {
        let mut acc = 0.0_f64;
        for r in 0..m {
            acc += block[[r, r]].abs();
        }
        acc / (m as f64)
    };
    let ridge = if diag_scale.is_finite() && diag_scale > 0.0 {
        1e-8 * diag_scale
    } else {
        1e-8
    };
    for r in 0..m {
        block[[r, r]] += ridge;
    }

    let mut g_border = Array1::<f64>::zeros(m);
    for (row, &axis) in border.iter().enumerate() {
        g_border[row] = gradient[axis];
    }

    let factor = {
        let view = FaerArrayView::new(&block);
        factorize_symmetricwith_fallback(view.as_ref(), Side::Lower).map_err(|err| {
            EstimationError::RemlOptimizationFailed(format!(
                "per-atom shared-border {m}×{m} factorization failed: {err:?}"
            ))
        })?
    };
    let delta = FactorizedSystem::solve(&factor, &g_border).map_err(|reason| {
        EstimationError::RemlOptimizationFailed(format!(
            "per-atom shared-border {m}×{m} solve failed: {reason}"
        ))
    })?;

    // Newton step is −H⁻¹ g; clamp each border component to the per-atom step
    // cap so a near-singular border block cannot eject the iterate.
    for (row, &axis) in border.iter().enumerate() {
        step[axis] = sanitize_step(-delta[row]);
    }
    Ok(step)
}

/// Whole-vector cost line search for the per-atom EFS step.
///
/// Wood–Fasiolo give ascent in the EFS direction but not full-step
/// monotonicity, so backtrack `α ∈ {1, 1/2, …, 2^-8}` on the *whole* applied
/// step (per-atom decoupled step plus the shared-border correction), accepting
/// the first α whose projected cost does not increase beyond the descent
/// tolerance. Returns the accepted `(rho_new, cost_new, alpha)`, or `None` when
/// no halving was accepted (the caller then surfaces a stall).
pub(crate) fn backtrack_cost(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    full_step: &Array1<f64>,
    current_cost: f64,
    cfg: &PerAtomEfsConfig,
) -> Result<Option<(Array1<f64>, f64, f64)>, EstimationError> {
    let mut alpha = 1.0_f64;
    let descent_slack = PER_ATOM_COST_DESCENT_TOL * current_cost.abs().max(1.0);
    for _ in 0..=PER_ATOM_MAX_BACKTRACK {
        let mut trial = rho.clone();
        for i in 0..trial.len() {
            trial[i] += alpha * full_step[i];
        }
        let trial = project_to_bounds(&trial, cfg);
        match obj.eval_cost(&trial) {
            Ok(cost) if cost.is_finite() && cost <= current_cost + descent_slack => {
                return Ok(Some((trial, cost, alpha)));
            }
            Ok(_) => {}
            Err(_) => {}
        }
        alpha *= 0.5;
    }
    Ok(None)
}

/// Run the per-atom decoupled EFS outer iteration as the primary frontier
/// ρ-scaling path.
///
/// Loop, each iteration:
/// 1. `eval_efs` at the current ρ — one inner P-IRLS solve — yields the full
///    per-coordinate decoupled step vector (embarrassingly parallel arithmetic
///    after the shared solve; we fan the clamp/assembly across rayon).
/// 2. If the topology has shared-border axes, add the coupled Newton correction
///    on just those axes via the matrix-free θ-HVP (`m × m` solve).
/// 3. Whole-vector cost line search; apply the accepted step.
/// 4. Converged when the applied step's ∞-norm falls below tolerance.
///
/// `seed` is the starting ρ (already within bounds). `cap` declares the outer
/// capability (used only for the layout/border defaults). The topology selects
/// which axes get the coupled correction; pass [`SharedBorderTopology::disjoint`]
/// when every atom owns a private penalty block (the common ARD-per-atom case).
pub fn run_per_atom_efs(
    obj: &mut dyn OuterObjective,
    seed: &Array1<f64>,
    cfg: &PerAtomEfsConfig,
    topology: &SharedBorderTopology,
) -> Result<PerAtomEfsResult, EstimationError> {
    let rho_dim = seed.len();
    if cfg.lower.len() != rho_dim || cfg.upper.len() != rho_dim {
        return Err(EstimationError::InvalidInput(format!(
            "per-atom EFS bounds dim mismatch: lower={}, upper={}, rho={}",
            cfg.lower.len(),
            cfg.upper.len(),
            rho_dim
        )));
    }
    if topology.rho_dim() != rho_dim {
        return Err(EstimationError::InvalidInput(format!(
            "per-atom EFS topology rho_dim {} != seed dim {}",
            topology.rho_dim(),
            rho_dim
        )));
    }

    // Warm any process-global lazy state (rayon pool init, OnceLock-backed
    // policy probes) at the top level by issuing the first inner solve here,
    // BEFORE any `into_par_iter` in the per-atom assembly. This satisfies the
    // OnceLock + nested-rayon deadlock rule: a `get_or_init` whose closure
    // itself does `into_par_iter` must never first fire from inside an outer
    // par_iter. The border-block probe below is the only par_iter, and by the
    // time it runs the inner solve has already forced all such warm-ups.
    let mut rho = project_to_bounds(seed, cfg);

    let mut iterations = 0usize;
    let mut final_step_inf = f64::INFINITY;
    let mut last_cost = f64::INFINITY;
    let mut converged = false;

    for _ in 0..cfg.max_iter.max(1) {
        iterations += 1;

        // ── Layer 1: per-atom decoupled EFS step (one inner solve) ──
        let efs = obj.eval_efs(&rho)?;
        if !efs.cost.is_finite() {
            return Err(EstimationError::RemlOptimizationFailed(
                "per-atom EFS: non-finite cost from eval_efs".to_string(),
            ));
        }
        if efs.steps.len() != rho_dim {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "per-atom EFS: step length {} != rho_dim {}",
                efs.steps.len(),
                rho_dim
            )));
        }
        last_cost = efs.cost;

        // Clamp each atom's own multiplicative step. This is the embarrassingly
        // parallel per-atom arithmetic: independent across coordinates. Fan it
        // across rayon; the closure touches only the local step value (no
        // OnceLock, no shared mutation), so it is deadlock-safe.
        let mut full_step: Array1<f64> = {
            let raw = efs.steps.as_slice();
            let clamped: Vec<f64> = (0..rho_dim)
                .into_par_iter()
                .map(|i| sanitize_step(raw[i]))
                .collect();
            Array1::from_vec(clamped)
        };

        // ── Layer 2: shared-border coupled correction (m × m) ──
        //
        // Only the border axes interact through the arrow border; correct just
        // those with a Newton step on the same outer gradient the coupled path
        // uses, filling the m × m restricted block via the matrix-free θ-HVP.
        if topology.border_count() > 0 {
            // Capture the outer gradient + operator at the current ρ via the
            // full `eval` hook (the EFS eval surfaces steps, not the raw
            // gradient/operator). One inner solve.
            let outer_eval = obj.eval(&rho)?;
            let operator: Option<Arc<dyn OuterHessianOperator>> = match &outer_eval.hessian {
                HessianResult::Operator(op) => Some(Arc::clone(op)),
                HessianResult::Analytic(_) | HessianResult::Unavailable => None,
            };
            let gradient = outer_eval.gradient.clone();
            if gradient.len() == rho_dim {
                // Borrow split: the FD branch of the θ-HVP needs to re-evaluate
                // the outer gradient at perturbed ρ, but `obj` is already
                // borrowed for the eval above. We snapshot the operator (an
                // Arc, cheap) so the gradient probe closure only needs the
                // operator path; when no operator exists we cannot FD-probe
                // through the borrowed `obj`, so we degrade the border
                // correction to operator-only and skip it when absent. This is
                // safe: layer-1 already produced a valid decoupled step; the
                // border correction is a refinement, never a feasibility gate.
                if operator.is_some() {
                    // FD probe is unreachable here (operator path is taken), but
                    // the θ-HVP signature requires a gradient re-evaluator; this
                    // closure errors loudly if the operator branch ever defers
                    // to it, which it does not while `operator.is_some()`.
                    let degraded_eval = |_p: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
                        Err(EstimationError::RemlOptimizationFailed(
                            "per-atom border FD probe unavailable without an \
                                 outer-Hessian operator"
                                .to_string(),
                        ))
                    };
                    match shared_border_correction(
                        topology,
                        operator.as_ref(),
                        &rho,
                        &gradient,
                        &degraded_eval,
                    ) {
                        Ok(border_step) => {
                            for i in 0..rho_dim {
                                full_step[i] = sanitize_step(full_step[i] + border_step[i]);
                            }
                        }
                        Err(err) => {
                            // A border-correction failure is non-fatal: keep the
                            // decoupled layer-1 step. Log via the cost path's
                            // own diagnostics on the next stall.
                            log::debug!("[PER-ATOM-EFS] shared-border correction skipped: {err}");
                        }
                    }
                } else {
                    log::debug!(
                        "[PER-ATOM-EFS] no outer-Hessian operator; shared-border \
                         correction deferred to decoupled step for this iter"
                    );
                }
            }
        }

        // Convergence on the applied (pre-line-search) step ∞-norm, gated by the
        // #1011 decision-margin contract. When the objective produced the cost's
        // ½log|H| term as a certified enclosure (rather than an exact logdet),
        // declaring convergence is only honest if that enclosure is tighter than
        // the step tolerance — the margin this decision is allowed to resolve.
        // An enclosure gap at or above the tolerance means the cost we would
        // converge on is not actually pinned down, so we must NOT stop on it: the
        // objective must refine the bound (more moments / pair absorption) or fall
        // back to the exact logdet before the EFS step can be called converged.
        let step_inf = full_step.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        final_step_inf = step_inf;
        let margin = cfg.tolerance.max(PER_ATOM_NEGLIGIBLE_STEP);
        let cost_resolved_below_margin = match efs.logdet_enclosure_gap {
            Some(gap) => {
                crate::solver::logdet_bounds::LogdetEnclosure::gap_resolves_margin(gap, margin)
            }
            None => true,
        };
        if step_inf < margin {
            if cost_resolved_below_margin {
                converged = true;
                break;
            }
            log::info!(
                "[PER-ATOM-EFS] step within tolerance {margin:.3e} but cost logdet enclosure gap \
                 {:.3e} exceeds it; refining the bound before declaring convergence",
                efs.logdet_enclosure_gap.unwrap_or(0.0)
            );
        }

        // ── Layer 3: whole-vector cost line search, then apply ──
        match backtrack_cost(obj, &rho, &full_step, efs.cost, cfg)? {
            Some((rho_new, cost_new, _alpha)) => {
                rho = rho_new;
                last_cost = cost_new;
            }
            None => {
                // No halving decreased the cost: the multiplicative surrogate is
                // not descent-correlated here. Stop and report non-convergence;
                // the coordinator's fallback ladder routes a stalled frontier
                // fit to a gradient-based primary, exactly as the EFS bridge
                // does for the dense-K path.
                log::info!(
                    "[PER-ATOM-EFS] step rejected after {} halvings at cost={:.6e} \
                     (rho_dim={}, border={}); reporting stall",
                    PER_ATOM_MAX_BACKTRACK,
                    efs.cost,
                    rho_dim,
                    topology.border_count(),
                );
                break;
            }
        }
    }

    Ok(PerAtomEfsResult {
        rho,
        final_value: last_cost,
        iterations,
        final_step_inf_norm: final_step_inf,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::rho_optimizer::{
        DeclaredHessianForm, Derivative, EfsEval, OuterEval, SeedOutcome,
    };
    use ndarray::array;

    /// Exact outer-Hessian operator for the quadratic mock: `v ↦ A·v`.
    pub(crate) struct QuadraticOperator {
        pub(crate) a: Array2<f64>,
    }

    impl OuterHessianOperator for QuadraticOperator {
        fn dim(&self) -> usize {
            self.a.nrows()
        }
        fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
            Ok(self.a.dot(v))
        }
    }

    /// Quadratic mock objective `f(ρ) = ½ (ρ − t)ᵀ A (ρ − t)`.
    ///
    /// The FD-consistency discipline of the module docs, made executable:
    /// `eval_efs` returns the **decoupled** per-coordinate step
    /// `−g_i / A_ii` (each coordinate's own Newton step from its own gradient
    /// entry and curvature scale — the shape of `compute_efs_update`), `eval`
    /// returns the *same* analytic gradient `A(ρ − t)` plus the exact
    /// operator, and `eval_cost` the same cost. Every layer of the per-atom
    /// runner is thereby probed against one shared ground truth.
    pub(crate) struct QuadraticObjective {
        pub(crate) a: Array2<f64>,
        pub(crate) target: Array1<f64>,
    }

    impl QuadraticObjective {
        pub(crate) fn grad(&self, rho: &Array1<f64>) -> Array1<f64> {
            self.a.dot(&(rho - &self.target))
        }
        pub(crate) fn cost(&self, rho: &Array1<f64>) -> f64 {
            let e = rho - &self.target;
            0.5 * e.dot(&self.a.dot(&e))
        }
    }

    impl OuterObjective for QuadraticObjective {
        fn capability(&self) -> OuterCapability {
            OuterCapability {
                gradient: Derivative::Analytic,
                hessian: DeclaredHessianForm::Dense,
                n_params: self.a.nrows(),
                psi_dim: 0,
                fixed_point_available: true,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            }
        }
        fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
            Ok(self.cost(rho))
        }
        fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
            Ok(OuterEval {
                cost: self.cost(rho),
                gradient: self.grad(rho),
                hessian: HessianResult::Operator(Arc::new(QuadraticOperator { a: self.a.clone() })),
                inner_beta_hint: None,
            })
        }
        fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
            let g = self.grad(rho);
            let steps: Vec<f64> = (0..rho.len()).map(|i| -g[i] / self.a[[i, i]]).collect();
            Ok(EfsEval {
                cost: self.cost(rho),
                steps,
                beta: None,
                psi_gradient: None,
                psi_indices: None,
                inner_hessian_scale: None,
                logdet_enclosure_gap: None,
            })
        }
        fn reset(&mut self) {}
        fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
            // A quadratic objective carries no inner P-IRLS state to warm-start,
            // so there is no slot to seed. A populated β must still match the
            // parameter dimension (an empty β is the "no warm-start" sentinel);
            // validating it keeps the mock honest about the seeding contract.
            if !beta.is_empty() {
                assert_eq!(beta.len(), self.a.nrows());
            }
            Ok(SeedOutcome::NoSlot)
        }
    }

    pub(crate) fn wide_bounds(dim: usize) -> PerAtomEfsConfig {
        PerAtomEfsConfig::new(
            1e-9,
            200,
            Array1::from_elem(dim, -50.0),
            Array1::from_elem(dim, 50.0),
        )
    }

    #[test]
    pub(crate) fn with_border_axes_sorts_dedups_and_validates() {
        let t = SharedBorderTopology::with_border_axes(8, vec![5, 1, 5, 3]).expect("topology");
        assert_eq!(t.border_axes(), &[1, 3, 5]);
        assert_eq!(t.border_count(), 3);
        let err = SharedBorderTopology::with_border_axes(4, vec![0, 4]).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
        // Empty border is the disjoint topology.
        let t = SharedBorderTopology::with_border_axes(4, Vec::new()).expect("empty");
        assert_eq!(t.border_count(), 0);
    }

    #[test]
    pub(crate) fn decoupled_primary_converges_on_separable_objective() {
        // Diagonal A: the per-atom decoupled step IS the exact Newton step for
        // every coordinate, so the frontier primary must converge to the
        // target with no border correction at all.
        let dim = 96; // above PER_ATOM_EFS_MIN_RHO_DIM: a frontier-shaped K
        let a = Array2::from_shape_fn(
            (dim, dim),
            |(i, j)| {
                if i == j { 1.0 + (i % 5) as f64 } else { 0.0 }
            },
        );
        let target = Array1::from_shape_fn(dim, |i| ((i as f64) * 0.37).sin() * 2.0);
        let mut obj = QuadraticObjective {
            a,
            target: target.clone(),
        };
        let cfg = wide_bounds(dim);
        let topology = SharedBorderTopology::disjoint(dim);
        let seed = Array1::zeros(dim);
        let result = run_per_atom_efs(&mut obj, &seed, &cfg, &topology).expect("run");
        assert!(result.converged, "separable quadratic must converge");
        for i in 0..dim {
            assert!(
                (result.rho[i] - target[i]).abs() < 1e-6,
                "coord {i}: {} vs target {}",
                result.rho[i],
                target[i]
            );
        }
        assert!(result.final_value < 1e-10);
    }

    #[test]
    pub(crate) fn border_correction_solves_the_coupled_block() {
        // Axes 0 and 1 are coupled through an off-diagonal Hessian block (the
        // arrow-border overlap); the rest are diagonal. With the populated
        // topology and the exact operator the runner must land on the global
        // optimum, and at that optimum the correction nulls (the
        // FD-consistency invariant: zero outer gradient ⇒ zero step).
        let dim = 6;
        let mut a = Array2::<f64>::eye(dim) * 2.0;
        a[[0, 1]] = 0.4;
        a[[1, 0]] = 0.4;
        let target = array![1.0, -2.0, 0.5, 0.0, -1.0, 3.0];
        let mut obj = QuadraticObjective {
            a,
            target: target.clone(),
        };
        let cfg = wide_bounds(dim);
        let topology = SharedBorderTopology::with_border_axes(dim, vec![0, 1]).expect("topology");
        let seed = Array1::zeros(dim);
        let result = run_per_atom_efs(&mut obj, &seed, &cfg, &topology).expect("run");
        assert!(result.converged, "coupled quadratic must converge");
        for i in 0..dim {
            assert!(
                (result.rho[i] - target[i]).abs() < 1e-5,
                "coord {i}: {} vs target {}",
                result.rho[i],
                target[i]
            );
        }
        assert!(
            result.final_step_inf_norm < 1e-8,
            "correction must null at the stationary point (got {})",
            result.final_step_inf_norm
        );
    }

    #[test]
    pub(crate) fn theta_hvp_fd_fallback_matches_the_analytic_action() {
        // The matrix-free θ-HVP's central-difference branch must reproduce
        // H·v of the analytic quadratic to O(h²).
        let a = array![[2.0, 0.3, 0.0], [0.3, 1.5, -0.2], [0.0, -0.2, 4.0]];
        let target = array![0.5, -1.0, 2.0];
        let rho = array![1.0, 0.2, -0.7];
        let v = array![0.3, -1.1, 0.9];
        let grad_at =
            |p: &Array1<f64>| -> Result<Array1<f64>, EstimationError> { Ok(a.dot(&(p - &target))) };
        let hv = theta_hvp_matrix_free(None, &rho, &v, grad_at).expect("hvp");
        let exact = a.dot(&v);
        for i in 0..3 {
            assert!(
                (hv[i] - exact[i]).abs() < 1e-6,
                "component {i}: FD {} vs exact {}",
                hv[i],
                exact[i]
            );
        }
        // Operator path: forwards to the exact matvec, bit-for-bit.
        let op: Arc<dyn OuterHessianOperator> = Arc::new(QuadraticOperator { a: a.clone() });
        let hv_op = theta_hvp_matrix_free(Some(&op), &rho, &v, grad_at).expect("op hvp");
        for i in 0..3 {
            assert_eq!(hv_op[i].to_bits(), exact[i].to_bits());
        }
    }

    #[test]
    pub(crate) fn full_border_reduces_to_dense_newton_in_one_correction() {
        // When the border is the WHOLE ρ-vector (small K), layer 2 is the
        // exact dense Newton step on the same gradient — so from any seed a
        // pure quadratic should converge essentially immediately (Newton is
        // exact; layer 1's decoupled part is then repaired by the line
        // search). This is the "reduction to the coupled objective at small
        // K" property of the module docs.
        let dim = 4;
        let a = array![
            [3.0, 0.2, 0.0, 0.1],
            [0.2, 2.0, 0.1, 0.0],
            [0.0, 0.1, 1.5, 0.2],
            [0.1, 0.0, 0.2, 2.5]
        ];
        let target = array![0.3, -0.6, 1.2, -0.1];
        let mut obj = QuadraticObjective {
            a,
            target: target.clone(),
        };
        let cfg = wide_bounds(dim);
        let topology =
            SharedBorderTopology::with_border_axes(dim, (0..dim).collect()).expect("topology");
        let seed = Array1::from_elem(dim, 2.0);
        let result = run_per_atom_efs(&mut obj, &seed, &cfg, &topology).expect("run");
        assert!(result.converged);
        for i in 0..dim {
            assert!((result.rho[i] - target[i]).abs() < 1e-5);
        }
    }

    /// Wraps a quadratic objective but reports its `½log|H|` term as a certified
    /// enclosure with a fixed gap, so the #1011 EFS margin gate can be exercised.
    pub(crate) struct EnclosureGapObjective {
        pub(crate) inner: QuadraticObjective,
        pub(crate) gap: f64,
    }

    impl OuterObjective for EnclosureGapObjective {
        fn capability(&self) -> OuterCapability {
            self.inner.capability()
        }
        fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
            self.inner.eval_cost(rho)
        }
        fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
            self.inner.eval(rho)
        }
        fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
            Ok(self
                .inner
                .eval_efs(rho)?
                .with_logdet_enclosure_gap(Some(self.gap)))
        }
        fn reset(&mut self) {
            self.inner.reset()
        }
        fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
            self.inner.seed_inner_state(beta)
        }
    }

    /// #1011 EFS margin contract: a step that lands within the step tolerance
    /// must NOT be allowed to declare convergence while the cost's certified
    /// logdet enclosure is wider than that tolerance — the bound does not pin
    /// the cost down at the decision's resolution. The run instead exhausts its
    /// iteration budget without a premature `converged = true`.
    #[test]
    pub(crate) fn efs_refuses_to_converge_below_the_logdet_enclosure_margin() {
        let dim = 96; // frontier-shaped K so the per-atom path is taken
        let a = Array2::from_shape_fn(
            (dim, dim),
            |(i, j)| if i == j { 1.0 + (i % 5) as f64 } else { 0.0 },
        );
        let target = Array1::from_shape_fn(dim, |i| ((i as f64) * 0.37).sin() * 2.0);
        let mut cfg = wide_bounds(dim);
        cfg.tolerance = 1e-6;
        let topology = SharedBorderTopology::disjoint(dim);
        let seed = Array1::zeros(dim);

        // Gap far wider than the step tolerance ⇒ the margin gate blocks
        // convergence even though the separable step reaches the target.
        let mut wide = EnclosureGapObjective {
            inner: QuadraticObjective {
                a: a.clone(),
                target: target.clone(),
            },
            gap: 1e-2,
        };
        let wide_result = run_per_atom_efs(&mut wide, &seed, &cfg, &topology).expect("wide run");
        assert!(
            !wide_result.converged,
            "an enclosure gap wider than the step tolerance must block convergence"
        );

        // The same objective with a gap below the tolerance converges exactly
        // as the exact-logdet path does — the margin contract is transparent
        // once the bound is tight enough to resolve the decision.
        let mut tight = EnclosureGapObjective {
            inner: QuadraticObjective {
                a,
                target: target.clone(),
            },
            gap: 1e-9,
        };
        let tight_result = run_per_atom_efs(&mut tight, &seed, &cfg, &topology).expect("tight run");
        assert!(
            tight_result.converged,
            "an enclosure gap below the step tolerance must not obstruct convergence"
        );
    }
}
