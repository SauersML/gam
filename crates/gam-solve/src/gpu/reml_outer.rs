//! GPU-backed outer REML optimization.
//!
//! Expensive objective work (inner P-IRLS, trace estimation, Arrow-Schur
//! factorization, and derivative assembly) stays in the device-backed evaluator.
//! Solver policy is shared with the host path through [`opt::Bfgs`], so both
//! paths use the same bounds, projected-gradient stopping, direction scaling,
//! hybrid line search, and robustness fixes.
//!
//! The device evaluator naturally returns value and gradient from one kernel
//! sequence. [`opt::FusedObjective`] preserves that fusion across the solver's
//! split cost/gradient requests, so accepted line-search points are not
//! evaluated twice.

use ndarray::Array1;
use opt::{
    Bfgs, BfgsError, Bounds, FirstOrderSample, FusedObjective, GradientTolerance, InitialMetric,
    MaxIterations, ObjectiveEvalError, Profile,
};

use crate::estimate::EstimationError;
use gam_gpu::policy::RemlOuterAdmission;

/// Input bundle handed to [`run_reml_outer_on_device`] by the host
/// outer-strategy dispatch site. Everything needed to seed the device-resident
/// BFGS driver and reconstruct the outer `OuterResult` after convergence.
///
/// The driver intentionally takes views rather than owned arrays for the
/// design / penalty / response so the host orchestrator's allocations remain
/// canonical — the device side keeps its own resident uploads and the host
/// arrays are only consulted for the seed evaluation and the post-loop
/// `OuterResult` payload.
#[derive(Clone, Debug)]
pub struct RemlOuterGpuInput {
    /// Initial ρ to start BFGS from. Same convention as the host BFGS branch
    /// in `rho_optimizer::run_outer_with_plan` — already projected onto the
    /// bounds box at the dispatch site.
    pub seed_rho: Array1<f64>,
    /// Per-axis lower / upper bounds on ρ. Same `(lo, hi)` shape the host
    /// branch consumes via `outer_bounds`.
    pub bounds: (Array1<f64>, Array1<f64>),
    /// Outer projected-gradient convergence rule. Mirrors
    /// `outer_gradient_tolerance(config)`.
    pub gradient_tolerance: GradientTolerance,
    /// Hard cap on outer BFGS iterations.
    pub max_iterations: usize,
    /// Per-axis step caps applied to BFGS line-search trial points. `None`
    /// disables axis-wise capping (matches the default opt::Bfgs behaviour).
    pub axis_step_caps: Option<Array1<f64>>,
    /// Admission descriptor used by the predicate. The driver keeps it on
    /// hand so it can re-check on each outer step that the inner family /
    /// curvature / device-availability still hold; a flip mid-run (e.g. the
    /// CUDA context dying) routes the next eval to the host fallback.
    pub admission: RemlOuterAdmission,
    /// REML log-evidence value at the seed (`−ℓ + ½ log|H_λ| − ½ log|S_λ|`
    /// up to the engine's sign convention). Used to seed the BFGS sample so
    /// the optimiser's first cached eval matches the host bridge.
    pub seed_objective: f64,
    /// Gradient paired with `seed_objective`. Passing the complete sample avoids
    /// repeating the expensive device-backed evaluation inside BFGS.
    pub seed_gradient: Array1<f64>,
}

/// Result of the device-resident outer driver. Shaped to feed
/// `rho_optimizer::solution_into_outer_result` so the host dispatch site can
/// surface the device-resident path through the same `OuterResult` envelope as
/// the host BFGS branch.
#[derive(Clone, Debug)]
pub struct RemlOuterGpuOutcome {
    /// Final ρ after BFGS convergence (or last accepted ρ at max-iter).
    pub rho: Array1<f64>,
    /// Final REML objective value at `rho`.
    pub objective: f64,
    /// Number of outer BFGS iterations consumed.
    pub iterations: usize,
    /// Final projected-gradient L2 norm; `None` when the optimiser exited before
    /// populating it.
    pub final_grad_norm: Option<f64>,
    /// Full final gradient vector (length `num_rho`); `None` on the same
    /// exits that leave `final_grad_norm` empty.
    pub final_gradient: Option<Array1<f64>>,
    /// True when the outer BFGS satisfied `gradient_tolerance`
    /// before exhausting the iteration budget.
    pub converged: bool,
}

/// Per-step evaluation handed back by the unified outer objective closure.
/// The driver feeds `objective` + `gradient` into the BFGS state update.
#[derive(Clone, Debug)]
pub struct RemlOuterDeviceEval {
    /// Penalised REML objective at the trial ρ. Single scalar download.
    pub objective: f64,
    /// Per-ρ gradient assembled from `evidence_derivatives_gpu` on the
    /// cached Cholesky factor of the penalised Hessian.
    pub gradient: Array1<f64>,
}

/// Drive the outer REML BFGS optimisation over ρ entirely through the
/// device-resident kernels.
///
/// ## Per-iteration flow
///
/// 1. Inner P-IRLS at the current ρ via `pirls_gpu::pirls_loop_on_stream`,
///    leaving the penalised Hessian factor resident on the device stream.
/// 2. Hutchinson trace estimator for `tr(H_λ⁻¹ ∂H/∂ρⱼ)` via the existing
///    on-device probes, batched across `num_rho` derivatives.
/// 3. Arrow-Schur batched Cholesky / log-determinant on the device through
///    `evidence_derivatives_gpu`, which keeps `H_λ` factored exactly once
///    per outer step and back-solves the per-ρ derivative slabs in one
///    `potrs` of width `p · num_rho`.
/// 4. Shared `opt::Bfgs` policy consumes the fused sample and chooses the next
///    bounded trial.
///
/// `evaluator` is the per-step bridge supplied by the outer-strategy
/// dispatch site: it is responsible for invoking the inner P-IRLS loop and
/// returning the objective and gradient from the unified REML evaluator.
/// Threading the evaluator in as a closure
/// keeps this driver agnostic of how the host wires up the REML evaluator
/// while still letting every per-step kernel run on the device.
pub fn run_reml_outer_on_device<E>(
    input: RemlOuterGpuInput,
    mut evaluator: E,
) -> Result<RemlOuterGpuOutcome, EstimationError>
where
    E: FnMut(&Array1<f64>) -> Result<RemlOuterDeviceEval, EstimationError>,
{
    if !matches!(input.admission.family, Some(_)) {
        return Err(EstimationError::RemlOptimizationFailed(
            "device-resident REML outer driver requires a JIT-cached PIRLS family".to_string(),
        ));
    }
    if !input.admission.gpu_available {
        return Err(EstimationError::RemlOptimizationFailed(
            "device-resident REML outer driver dispatched without GPU runtime".to_string(),
        ));
    }

    let num_rho = input.seed_rho.len();
    if num_rho == 0 {
        return Ok(RemlOuterGpuOutcome {
            rho: Array1::<f64>::zeros(0),
            objective: input.seed_objective,
            iterations: 0,
            final_grad_norm: Some(0.0),
            final_gradient: Some(Array1::<f64>::zeros(0)),
            converged: true,
        });
    }
    if input.bounds.0.len() != num_rho || input.bounds.1.len() != num_rho {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "device-resident REML outer driver: bounds shape mismatch (num_rho={num_rho}, \
             lower={}, upper={})",
            input.bounds.0.len(),
            input.bounds.1.len(),
        )));
    }

    if input.seed_gradient.len() != num_rho {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "device-backed REML outer driver: seed gradient has length {}, expected {num_rho}",
            input.seed_gradient.len(),
        )));
    }
    if !input.seed_objective.is_finite() || input.seed_gradient.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::RemlOptimizationFailed(
            "device-backed REML outer driver received a non-finite seed sample".to_string(),
        ));
    }

    let max_iterations = MaxIterations::new(input.max_iterations).map_err(|err| {
        EstimationError::InvalidInput(format!("outer max_iter is invalid: {err}"))
    })?;
    let bounds = Bounds::new(input.bounds.0, input.bounds.1, 1.0e-6).map_err(|err| {
        EstimationError::InvalidInput(format!("outer rho bounds are invalid: {err}"))
    })?;
    let seed_sample = FirstOrderSample {
        value: input.seed_objective,
        gradient: input.seed_gradient,
    };
    let initial_grad_norm = seed_sample.gradient.dot(&seed_sample.gradient).sqrt();
    let initial_scale = if initial_grad_norm.is_finite() && initial_grad_norm > 0.0 {
        (1.0 / initial_grad_norm).clamp(1.0e-3, 1.0e3)
    } else {
        1.0
    };

    let objective = FusedObjective::new(move |rho: &Array1<f64>| {
        evaluator(rho)
            .map(|eval| FirstOrderSample {
                value: eval.objective,
                gradient: eval.gradient,
            })
            .map_err(|err| ObjectiveEvalError::recoverable(err.to_string()))
    });
    let mut optimizer = Bfgs::new(input.seed_rho.clone(), objective)
        .with_initial_sample(input.seed_rho, seed_sample)
        .with_bounds(bounds)
        .with_gradient_tolerance(input.gradient_tolerance)
        .with_max_iterations(max_iterations)
        .with_initial_metric(InitialMetric::Scalar(initial_scale))
        .with_profile(Profile::Robust);
    if let Some(caps) = input.axis_step_caps {
        optimizer = optimizer.with_axis_step_caps(caps);
    }

    let (solution, converged) = match optimizer.run() {
        Ok(solution) => (solution, true),
        Err(BfgsError::MaxIterationsReached { last_solution })
        | Err(BfgsError::LineSearchFailed { last_solution, .. }) => (*last_solution, false),
        Err(err) => {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "device-backed opt::Bfgs failed: {err}"
            )));
        }
    };

    Ok(RemlOuterGpuOutcome {
        rho: solution.final_point,
        objective: solution.final_value,
        iterations: solution.iterations,
        final_grad_norm: solution.final_gradient_norm,
        final_gradient: solution.final_gradient,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_gpu::policy::{PirlsLoopCurvatureKind, PirlsLoopFamilyKind};

    fn dummy_admission(num_rho: usize) -> RemlOuterAdmission {
        RemlOuterAdmission {
            n: 200_000,
            p: 64,
            num_rho,
            family: Some(PirlsLoopFamilyKind::BernoulliLogit),
            curvature: PirlsLoopCurvatureKind::Fisher,
            gpu_available: true,
        }
    }

    #[test]
    fn empty_rho_returns_seed_objective() {
        let input = RemlOuterGpuInput {
            seed_rho: Array1::<f64>::zeros(0),
            bounds: (Array1::<f64>::zeros(0), Array1::<f64>::zeros(0)),
            gradient_tolerance: GradientTolerance::absolute(1.0e-6),
            max_iterations: 10,
            axis_step_caps: None,
            admission: dummy_admission(0),
            seed_objective: 42.0,
            seed_gradient: Array1::zeros(0),
        };
        let evaluator = |_: &Array1<f64>| -> Result<RemlOuterDeviceEval, EstimationError> {
            Ok(RemlOuterDeviceEval {
                objective: 0.0,
                gradient: Array1::<f64>::zeros(0),
            })
        };
        let out = run_reml_outer_on_device(input, evaluator).expect("empty path");
        assert_eq!(out.iterations, 0);
        assert!(out.converged);
        assert_eq!(out.objective, 42.0);
    }

    #[test]
    fn converges_on_quadratic() {
        // φ(ρ) = ½‖ρ − ρ*‖² has gradient (ρ − ρ*) and a constant Hessian
        // identity; BFGS hits the optimum in a couple of iterations.
        let target = Array1::from(vec![0.5_f64, -0.25, 1.0, -0.75]);
        let target_owned = target.clone();
        let seed = Array1::from(vec![2.0, 2.0, 2.0, 2.0]);
        let seed_diff = &seed - &target;
        let input = RemlOuterGpuInput {
            seed_rho: seed,
            bounds: (Array1::from_elem(4, -10.0), Array1::from_elem(4, 10.0)),
            gradient_tolerance: GradientTolerance::absolute(1.0e-8),
            max_iterations: 100,
            axis_step_caps: None,
            admission: dummy_admission(4),
            seed_objective: 0.5 * seed_diff.dot(&seed_diff),
            seed_gradient: seed_diff,
        };
        let evaluator = move |rho: &Array1<f64>| -> Result<RemlOuterDeviceEval, EstimationError> {
            let diff: Array1<f64> = rho - &target_owned;
            let value = 0.5 * diff.iter().map(|v| v * v).sum::<f64>();
            Ok(RemlOuterDeviceEval {
                objective: value,
                gradient: diff,
            })
        };
        let out = run_reml_outer_on_device(input, evaluator).expect("quadratic path");
        assert!(out.converged, "BFGS should converge on a quadratic");
        for (got, want) in out.rho.iter().zip(target.iter()) {
            assert!((*got - *want).abs() < 1.0e-4_f64, "got {got} want {want}");
        }
    }

    #[test]
    fn stationary_seed_is_not_evaluated_again() {
        let input = RemlOuterGpuInput {
            seed_rho: Array1::from(vec![0.25]),
            bounds: (Array1::from(vec![-1.0]), Array1::from(vec![1.0])),
            gradient_tolerance: GradientTolerance::absolute(1.0e-8),
            max_iterations: 10,
            axis_step_caps: None,
            admission: dummy_admission(1),
            seed_objective: 3.0,
            seed_gradient: Array1::zeros(1),
        };
        let evaluator = |_rho: &Array1<f64>| -> Result<RemlOuterDeviceEval, EstimationError> {
            panic!("the precomputed stationary seed must satisfy the first solver evaluation")
        };

        let out = run_reml_outer_on_device(input, evaluator).expect("stationary seed");

        assert!(out.converged);
        assert_eq!(out.iterations, 0);
        assert_eq!(out.objective, 3.0);
    }
}
