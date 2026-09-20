//! Frontier ρ-scaling: per-atom decoupled Extended Fellner–Schall (EFS) as the
//! primary outer iteration (issue #986).
//!
//! # Why this module exists
//!
//! ARD-per-atom assigns one smoothing coordinate ρ per dictionary atom. At
//! frontier scale that is `10^4`–`10^5` coordinates. A dense outer
//! quasi-Newton (ARC/BFGS) over that ρ-vector is impossible: it materializes an
//! O(K²) outer Hessian and factorizes it every accepted step. The standard
//! [`crate::estimate::reml::reml_outer_engine::compute_efs_update`] is already
//! *per-coordinate decoupled* in its arithmetic — each ρ_i step is
//! `log(1 − 2·g_full[i]/q_eff_i)`, a function of that atom's own gradient entry
//! and penalty-quadratic curvature scale only — so EFS is the natural frontier
//! primary. This module drives that decoupled fixed point directly and never
//! assembles any K×K object.
//!
//! Each iteration, the outer objective's `eval_efs` hook runs one inner P-IRLS
//! solve and returns the full per-coordinate step vector. We apply each atom's
//! own multiplicative log-λ step, with a whole-vector cost line search
//! (Wood–Fasiolo give ascent in the EFS direction but not full-step
//! monotonicity). The step is taken as `eval_efs` produced it, with no
//! per-coordinate box: the line search alone sets its length.
//!
//! # Consistency / reduction to the coupled objective at small K
//!
//! At small K the per-atom step reduces to exactly the coupled EFS step:
//! `compute_efs_update` already produces the same per-coordinate
//! `log(1 − 2·g_full[i]/q_eff_i)` regardless of K, and when `g = 0` (a
//! stationary point of the coupled objective) the step is zero. This
//! stationarity property is a design invariant of routing the step through the
//! same `eval_efs` hook that the dense path consumes, not a separately
//! maintained surrogate.

use crate::estimate::EstimationError;
use crate::rho_optimizer::{
    OuterCapability, OuterObjective, OuterPlan, OuterResult, OuterResultOrigin,
};
use ndarray::Array1;
use opt::{BacktrackConfig, backtracking_line_search};

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
pub(crate) const PER_ATOM_EFS_MIN_RHO_DIM: usize = 64;

/// Auto-switch threshold predicate: is this problem in the frontier ρ-scaling
/// regime where the per-atom decoupled EFS primary should take over from the
/// dense outer?
///
/// Magic-by-default: derived from the ρ-dimension only. The caller passes the
/// number of penalty-like smoothing coordinates (`rho_dim`); no flag, no env.
#[inline]
pub(crate) fn is_frontier_rho_scale(rho_dim: usize) -> bool {
    rho_dim >= PER_ATOM_EFS_MIN_RHO_DIM
}

/// Whether a given outer capability is eligible for — and large enough to
/// benefit from — the per-atom decoupled EFS primary.
///
/// Requires: all coordinates penalty-like (no ψ design-moving coords, which
/// EFS cannot resolve; those still route to HybridEFS / BFGS), a working
/// fixed-point hook (`eval_efs`), fixed-point not disabled by the caller, and
/// a frontier-scale ρ-dimension.
pub(crate) fn per_atom_efs_eligible(cap: &OuterCapability) -> bool {
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
    //
    // #1521: `OuterPlan` / `OuterResult` live in the private `rho_optimizer::run`
    // module (re-exported only as `OuterProblem`), so they are not pub-reachable.
    // Keep this method `pub(crate)` so the now-`pub` `per_atom_efs` module does
    // not expose a private-in-public type (`private_interfaces` under
    // `warnings = "deny"`). Called only in-crate (`rho_optimizer::run`); the root
    // facade re-exports `run_per_atom_efs` / `PerAtomEfsConfig`, never this
    // convenience method.
    pub(crate) fn into_outer_result(self, plan_used: OuterPlan) -> OuterResult {
        let mut result = OuterResult::new(
            self.rho,
            self.final_value,
            self.iterations,
            self.converged,
            plan_used,
        );
        result.origin = OuterResultOrigin::PerAtomFellnerSchall;
        result.final_grad_norm = Some(self.final_step_inf_norm);
        result
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

/// The step vector exactly as its source produced it. Its length is globalised
/// by the whole-vector cost line search, not by a box on each coordinate (SPEC
/// 18-22). A non-finite component is refused: it has no direction to search, and
/// zeroing it would let the ∞-norm test certify a point the step never resolved.
pub(crate) fn finite_step(raw: &[f64], source: &str) -> Result<Array1<f64>, EstimationError> {
    if let Some((axis, value)) = raw.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "per-atom EFS: {source} step on axis {axis} is non-finite ({value})"
        )));
    }
    Ok(Array1::from_vec(raw.to_vec()))
}

/// Whole-vector cost line search for the per-atom EFS step.
///
/// Wood–Fasiolo give ascent in the EFS direction but not full-step
/// monotonicity, so halve α from 1 on the *whole* applied per-atom step,
/// accepting the first α whose projected cost does not resolvably exceed the
/// current cost. Halving ends where the step stops being resolvable: once
/// `α·‖step‖∞` falls below the step-norm tolerance, the move is one the convergence test would already call
/// zero, so the schedule is `{α = 2^-k : α·‖step‖∞ ≥ tolerance}` and carries no
/// count of its own. Two costs differ resolvably when they are further apart than
/// the sum of their rounding bands, `γ₁·(|f_cur| + |f_trial|)`. Returns the
/// accepted `(rho_new, cost_new, alpha)`, or `None` when no resolvable halving
/// was accepted (the caller then surfaces a stall).
pub(crate) fn backtrack_cost(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    full_step: &Array1<f64>,
    current_cost: f64,
    cfg: &PerAtomEfsConfig,
) -> Result<Option<(Array1<f64>, f64, f64)>, EstimationError> {
    if !(cfg.tolerance > 0.0 && cfg.tolerance.is_finite()) {
        return Err(EstimationError::InvalidInput(format!(
            "per-atom EFS: step-norm tolerance {} must be positive and finite",
            cfg.tolerance
        )));
    }
    let full_step = finite_step(&full_step.to_vec(), "line-search")?;
    let step_inf = full_step.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    let schedule = BacktrackConfig::default();
    let mut max_steps = 0usize;
    let mut alpha = schedule.initial_step;
    while alpha * step_inf >= cfg.tolerance {
        max_steps += 1;
        alpha *= schedule.contraction;
    }
    let band = gam_math::roundoff::accumulation_growth(1);
    // Recoverable domain refusals arrive as `Ok(+∞)` and keep halving. A typed
    // error means the objective artifact could not be built and must escape this
    // line search without being reinterpreted as another numerical point.
    let accepted = backtracking_line_search::<_, EstimationError>(
        BacktrackConfig {
            max_steps,
            ..schedule
        },
        |alpha| {
            let mut trial = rho.clone();
            for i in 0..trial.len() {
                trial[i] += alpha * full_step[i];
            }
            let trial = project_to_bounds(&trial, cfg);
            let cost = obj.eval_cost(&trial)?;
            Ok(Some((cost, trial)))
        },
        |_, cost| {
            cost.is_finite() && cost <= current_cost + band * (current_cost.abs() + cost.abs())
        },
    )?;
    Ok(accepted.map(|step| (step.payload, step.value, step.step)))
}

/// Run the per-atom decoupled EFS outer iteration as the primary frontier
/// ρ-scaling path.
///
/// Loop, each iteration:
/// 1. `eval_efs` at the current ρ — one inner P-IRLS solve — yields the full
///    per-coordinate decoupled step vector.
/// 2. Converged when the step's ∞-norm falls below tolerance.
/// 3. Otherwise whole-vector cost line search; apply the accepted step.
///
/// `seed` is the starting ρ; it is projected into the bounds.
pub fn run_per_atom_efs(
    obj: &mut dyn OuterObjective,
    seed: &Array1<f64>,
    cfg: &PerAtomEfsConfig,
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

    let mut rho = project_to_bounds(seed, cfg);

    let mut iterations = 0usize;
    let mut final_step_inf = f64::INFINITY;
    let mut last_cost = f64::INFINITY;
    let mut converged = false;
    // The progress certificate the dense fixed-point walk carries (#2817,
    // #3176): an evaluation that bought no resolved improvement and no smaller
    // step ends the walk as a stall, instead of the iteration count.
    let mut progress = crate::rho_optimizer::FixedPointProgress::new();

    for _ in 0..cfg.max_iter.max(1) {
        iterations += 1;

        // Per-atom decoupled EFS step (one inner solve).
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

        // Each atom's own multiplicative step, unboxed: its length is set by the
        // whole-vector line search below, and a non-finite component is refused.
        let full_step = finite_step(&efs.steps, "decoupled EFS")?;

        // Convergence on the applied (pre-line-search) step ∞-norm.
        let step_inf = full_step.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        final_step_inf = step_inf;
        if step_inf < cfg.tolerance {
            converged = true;
            break;
        }
        if progress.observe(efs.cost, step_inf) {
            log::debug!(
                "[PER-ATOM-EFS] stopping at an unprogressing walk after {iterations} \
                 iteration(s) at cost={:.6e}: the evaluation bought no resolved improvement \
                 and no smaller step; reporting stall (#2817, #3176)",
                efs.cost,
            );
            break;
        }

        // Whole-vector cost line search, then apply.
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
                log::debug!(
                    "[PER-ATOM-EFS] step rejected at every resolvable halving \
                     (step_inf={:.3e}, tolerance={:.3e}) at cost={:.6e} \
                     (rho_dim={}); reporting stall",
                    step_inf,
                    cfg.tolerance,
                    efs.cost,
                    rho_dim,
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
    use crate::rho_optimizer::SeedOutcome;
    use gam_problem::{DeclaredHessianForm, Derivative, EfsEval, HessianValue, OuterEval};
    use ndarray::{Array2, array};

    /// Quadratic mock objective `f(ρ) = ½ (ρ − t)ᵀ A (ρ − t)`.
    ///
    /// The consistency discipline of the module docs, made executable:
    /// `eval_efs` returns the **decoupled** per-coordinate step
    /// `−g_i / A_ii` (each coordinate's own Newton step from its own gradient
    /// entry and curvature scale — the shape of `compute_efs_update`), `eval`
    /// returns the *same* analytic gradient `A(ρ − t)` plus the exact
    /// Hessian, and `eval_cost` the same cost. The per-atom runner is
    /// thereby probed against one shared ground truth.
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
                hessian: HessianValue::Dense(self.a.clone()),
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
                consecutive_restored_incumbents: None,
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
    fn per_atom_backtracking_propagates_typed_value_failure() {
        const SENTINEL: &str = "#2481 per-atom value artifact";

        struct FailingCostObjective;

        impl OuterObjective for FailingCostObjective {
            fn capability(&self) -> OuterCapability {
                OuterCapability {
                    gradient: Derivative::Analytic,
                    hessian: DeclaredHessianForm::Unavailable,
                    n_params: 1,
                    psi_dim: 0,
                    fixed_point_available: true,
                    barrier_config: None,
                    prefer_gradient_only: false,
                    disable_fixed_point: false,
                }
            }

            fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
                // The failure is unconditional, but the caller must still hand
                // this objective a point of its declared dimension — otherwise
                // the test would pass on a mis-wired line search.
                assert_eq!(rho.len(), self.capability().n_params);
                Err(EstimationError::InvalidInput(SENTINEL.to_string()))
            }

            fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
                assert_eq!(rho.len(), self.capability().n_params);
                Err(EstimationError::InvalidInput(SENTINEL.to_string()))
            }

            fn reset(&mut self) {}

            fn seed_inner_state(
                &mut self,
                beta: &Array1<f64>,
            ) -> Result<SeedOutcome, EstimationError> {
                // No inner state to seed, but an empty β is the documented
                // "no warm-start" sentinel; anything else must be a real
                // coefficient vector for the single declared parameter.
                if !beta.is_empty() {
                    assert_eq!(beta.len(), self.capability().n_params);
                }
                Ok(SeedOutcome::NoSlot)
            }
        }

        let mut objective = FailingCostObjective;
        let error = backtrack_cost(
            &mut objective,
            &array![0.0],
            &array![1.0],
            1.0,
            &wide_bounds(1),
        )
        .expect_err("a typed value failure must escape per-atom backtracking");
        assert!(matches!(error, EstimationError::InvalidInput(_)));
        assert!(error.to_string().contains(SENTINEL));
    }

    #[test]
    pub(crate) fn decoupled_primary_converges_on_separable_objective() {
        // Diagonal A: the per-atom decoupled step IS the exact Newton step for
        // every coordinate, so the frontier primary must converge to the
        // target.
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
        let seed = Array1::zeros(dim);
        let result = run_per_atom_efs(&mut obj, &seed, &cfg).expect("run");
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
    fn decoupled_step_is_realised_without_a_per_coordinate_box() {
        // Every coordinate sits 20 log-units from its target, well inside the
        // ±50 bounds. The decoupled step is the exact Newton step, so the walk
        // lands in one step and certifies it on the next: two iterations. A
        // ±5 box on each coordinate needs four boxed steps before the fifth
        // iteration even sees a step below tolerance.
        let dim = 96;
        let a = Array2::from_shape_fn(
            (dim, dim),
            |(i, j)| {
                if i == j { 1.0 + (i % 5) as f64 } else { 0.0 }
            },
        );
        let target = Array1::from_shape_fn(dim, |i| if i % 2 == 0 { 20.0 } else { -20.0 });
        let mut obj = QuadraticObjective {
            a,
            target: target.clone(),
        };
        let cfg = wide_bounds(dim);
        let result = run_per_atom_efs(&mut obj, &Array1::zeros(dim), &cfg).expect("run");
        assert!(result.converged, "separable quadratic must converge");
        assert_eq!(
            result.iterations, 2,
            "the exact Newton step must be taken whole, not boxed"
        );
        for i in 0..dim {
            assert_eq!(result.rho[i], target[i], "coord {i}");
        }
    }

    #[test]
    fn backtracking_halves_until_the_step_is_unresolvable() {
        // f(ρ) = ½(ρ − 1)², ρ = 0, step 768: the cost at ρ = 768α does not
        // exceed f(0) = ½ only for 768α ≤ 2, first reached at α = 2^-9. A
        // schedule stopped at a fixed count of eight halvings (α = 2^-8,
        // ρ = 3, f = 2) reports a stall on a step that halving still resolves.
        let mut obj = QuadraticObjective {
            a: array![[1.0]],
            target: array![1.0],
        };
        let cfg = PerAtomEfsConfig::new(
            1e-9,
            200,
            Array1::from_elem(1, -1e4),
            Array1::from_elem(1, 1e4),
        );
        let (rho_new, cost_new, alpha) =
            backtrack_cost(&mut obj, &array![0.0], &array![768.0], 0.5, &cfg)
                .expect("line search")
                .expect("a resolvable halving decreases the cost");
        assert_eq!(alpha, 2f64.powi(-9));
        assert_eq!(rho_new[0], 1.5);
        assert_eq!(cost_new, 0.125);

        // Once α·‖step‖∞ is below the tolerance there is nothing left to try:
        // a step already under it is refused without evaluating anything.
        let none =
            backtrack_cost(&mut obj, &array![0.0], &array![1e-10], 0.5, &cfg).expect("line search");
        assert!(none.is_none());
    }

    #[test]
    fn backtracking_refuses_a_resolvable_cost_increase() {
        // Every trial point costs 5e-13 more than the current 1.0: far below
        // any fixed relative slack of 1e-12, but more than two thousand times
        // the rounding band γ₁·(|1| + |1 + 5e-13|) ≈ 2.2e-16 of the two values,
        // so it is a real increase and no step may be accepted.
        struct RisingObjective;
        impl OuterObjective for RisingObjective {
            fn capability(&self) -> OuterCapability {
                OuterCapability {
                    gradient: Derivative::Analytic,
                    hessian: DeclaredHessianForm::Unavailable,
                    n_params: 1,
                    psi_dim: 0,
                    fixed_point_available: true,
                    barrier_config: None,
                    prefer_gradient_only: false,
                    disable_fixed_point: false,
                }
            }
            fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
                assert_eq!(rho.len(), 1);
                Ok(1.0 + 5e-13)
            }
            fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
                assert_eq!(rho.len(), 1);
                Err(EstimationError::InvalidInput("unused".to_string()))
            }
            fn reset(&mut self) {}
            fn seed_inner_state(
                &mut self,
                beta: &Array1<f64>,
            ) -> Result<SeedOutcome, EstimationError> {
                if !beta.is_empty() {
                    assert_eq!(beta.len(), 1);
                }
                Ok(SeedOutcome::NoSlot)
            }
        }
        let accepted = backtrack_cost(
            &mut RisingObjective,
            &array![0.0],
            &array![1.0],
            1.0,
            &wide_bounds(1),
        )
        .expect("line search");
        assert!(
            accepted.is_none(),
            "a cost increase above rounding must be refused"
        );
    }

    #[test]
    fn non_finite_decoupled_step_is_refused_not_zeroed() {
        // One coordinate's step is NaN. Zeroing it lets the ∞-norm test call the
        // walk converged at a point that coordinate never resolved.
        struct NanStepObjective(QuadraticObjective);
        impl OuterObjective for NanStepObjective {
            fn capability(&self) -> OuterCapability {
                self.0.capability()
            }
            fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
                self.0.eval_cost(rho)
            }
            fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
                self.0.eval(rho)
            }
            fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
                let mut efs = self.0.eval_efs(rho)?;
                efs.steps[0] = f64::NAN;
                Ok(efs)
            }
            fn reset(&mut self) {}
            fn seed_inner_state(
                &mut self,
                beta: &Array1<f64>,
            ) -> Result<SeedOutcome, EstimationError> {
                self.0.seed_inner_state(beta)
            }
        }
        let dim = 2;
        let mut obj = NanStepObjective(QuadraticObjective {
            a: Array2::eye(dim),
            target: array![3.0, 0.0],
        });
        let error = run_per_atom_efs(&mut obj, &Array1::zeros(dim), &wide_bounds(dim))
            .err()
            .expect("a non-finite EFS step must fail the run");
        assert!(error.to_string().contains("non-finite"), "{error}");
    }
}
