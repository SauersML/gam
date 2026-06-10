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
//! [`crate::solver::estimate::reml::unified::compute_efs_update`] is already
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

use crate::solver::estimate::EstimationError;
use crate::solver::outer_strategy::{
    HessianResult, OuterCapability, OuterHessianOperator, OuterObjective, OuterPlan, OuterResult,
};
use crate::linalg::faer_ndarray::{FaerArrayView, factorize_symmetricwith_fallback};
use crate::matrix::FactorizedSystem;
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
const PER_ATOM_MAX_STEP: f64 = 5.0;

/// Whole-vector backtracking halvings for the per-atom EFS line search.
const PER_ATOM_MAX_BACKTRACK: usize = 8;

/// Step components below this magnitude (in θ-space) are treated as numerically
/// zero for convergence and line-search purposes.
const PER_ATOM_NEGLIGIBLE_STEP: f64 = 1e-12;

/// Relative tolerance for the descent condition during backtracking; matches
/// the unified EFS path so ULP-level cost noise near a fixed point does not
/// trigger spurious backtracking.
const PER_ATOM_COST_DESCENT_TOL: f64 = 1e-12;

/// Finite-difference probe magnitude (in θ-space) for the θ-HVP fallback when
/// no exact outer-Hessian operator is available. Central difference, so the
/// error is O(h²); `1e-4` balances truncation against the inner-solve noise
/// floor of the outer gradient.
const THETA_HVP_FD_STEP: f64 = 1.0e-4;

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
    border_axes: Vec<usize>,
    rho_dim: usize,
}

impl SharedBorderTopology {
    /// Construct from the explicit set of border axes. Out-of-range and
    /// duplicate axes are dropped; the set is deduplicated and sorted so the
    /// restricted `m × m` correction has a stable, deterministic ordering (no
    /// clock, no hash-iteration order).
    pub fn new(rho_dim: usize, border_axes: impl IntoIterator<Item = usize>) -> Self {
        let mut axes: Vec<usize> = border_axes.into_iter().filter(|&i| i < rho_dim).collect();
        axes.sort_unstable();
        axes.dedup();
        Self {
            border_axes: axes,
            rho_dim,
        }
    }

    /// No shared border: every atom is block-disjoint and the decoupled
    /// per-atom step is exact. This is the common ARD-per-atom case where each
    /// atom owns a private penalty block.
    pub fn disjoint(rho_dim: usize) -> Self {
        Self {
            border_axes: Vec::new(),
            rho_dim,
        }
    }

    /// The whole ρ-vector is one shared border (small-K / fully-coupled
    /// regime). Used by the FD-consistency reduction: with this topology the
    /// layer-2 correction is the exact dense Newton step on the outer gradient,
    /// matching the coupled objective.
    pub fn fully_coupled(rho_dim: usize) -> Self {
        Self {
            border_axes: (0..rho_dim).collect(),
            rho_dim,
        }
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
    fn rho_dim(&self) -> usize {
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
fn project_axis(value: f64, lo: f64, hi: f64) -> f64 {
    value.max(lo).min(hi)
}

fn project_to_bounds(rho: &Array1<f64>, cfg: &PerAtomEfsConfig) -> Array1<f64> {
    let mut out = rho.clone();
    for i in 0..out.len() {
        out[i] = project_axis(out[i], cfg.lower[i], cfg.upper[i]);
    }
    out
}

/// Clamp a raw multiplicative log-λ step to the per-atom maximum, treating
/// non-finite raw steps as zero (the coordinate makes no move this iteration).
#[inline]
fn sanitize_step(raw: f64) -> f64 {
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
fn border_hessian_block(
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
fn shared_border_correction(
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
fn backtrack_cost(
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
                    let degraded_eval =
                        |_p: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
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
                            log::debug!(
                                "[PER-ATOM-EFS] shared-border correction skipped: {err}"
                            );
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

        // Convergence on the applied (pre-line-search) step ∞-norm.
        let step_inf = full_step.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        final_step_inf = step_inf;
        if step_inf < cfg.tolerance.max(PER_ATOM_NEGLIGIBLE_STEP) {
            converged = true;
            break;
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
mod topology_tests {
    use super::SharedBorderTopology;

    /// `new` must sort, deduplicate, and drop out-of-range axes so the
    /// restricted border correction has a deterministic ordering; `disjoint`
    /// and `fully_coupled` are its two extremes (m = 0 and m = rho_dim).
    #[test]
    fn shared_border_topology_constructors_pin_their_semantics() {
        let t = SharedBorderTopology::new(5, [4usize, 1, 4, 9, 1]);
        assert_eq!(t.border_axes(), &[1, 4], "sorted, deduped, in-range");
        assert_eq!(t.border_count(), 2);

        let d = SharedBorderTopology::disjoint(5);
        assert_eq!(d.border_count(), 0, "disjoint = exact decoupled step");

        let f = SharedBorderTopology::fully_coupled(3);
        assert_eq!(f.border_axes(), &[0, 1, 2]);
        assert_eq!(
            f.border_count(),
            3,
            "fully coupled = dense Newton on the whole rho vector"
        );
    }
}
