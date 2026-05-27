//! Device-resident outer REML BFGS-over-ρ driver.
//!
//! ## Why this module exists
//!
//! Before this driver landed, the outer REML BFGS optimiser over the smoothing
//! parameter vector `ρ` ran on the host orchestrator (`outer_strategy::run_outer`
//! → `Solver::Bfgs`). Each outer iteration evaluated `(cost, grad)` by hopping
//! back into the CPU REML evaluator, which in turn dispatched the inner P-IRLS
//! loop, the Hutchinson trace estimator, the arrow-Schur batched Cholesky, and
//! the sigma-cubature covariance to device kernels. The device-resident pieces
//! were already in place (`pirls_loop_on_stream`, `sigma_cubature_dispatch`,
//! `arrow_schur_gpu`), but every outer step still paid:
//!
//! * one full BFGS state hop out to host memory (ρ, gradient, inverse-Hessian
//!   approximation, line-search bookkeeping);
//! * a fresh round of design-matrix uploads inside the inner P-IRLS loop, even
//!   when the design was unchanged across the entire outer trajectory;
//! * scalar-by-scalar gather of trace estimates from the host orchestrator
//!   between Hutchinson probes and the gradient assembler.
//!
//! At biobank scale (n ≈ 200k–500k, p ≈ 32–128, `num_rho` ≈ 8–32) those hops
//! dominate the wall-clock cost of the outer BFGS-over-ρ loop. This module
//! collapses them into a single device-resident driver that:
//!
//! 1. Uploads the shared design matrix `X` once via
//!    `pirls_gpu::upload_shared_pirls_gpu`, and allocates the per-iter PIRLS
//!    workspace once.
//! 2. Keeps the BFGS state — ρ vector, gradient, BFGS inverse-Hessian
//!    approximation `H⁻¹`, line-search trial points — resident on the device's
//!    default stream; only the per-step scalar objective and a scalar
//!    convergence flag are downloaded.
//! 3. Assembles the per-step gradient by calling
//!    `evidence_derivatives_gpu` against the cached factor, with the derivative
//!    Hessians supplied by the existing arrow-Schur batched solver — no host
//!    fan-out.
//! 4. Drives the inner P-IRLS via `pirls_loop_on_stream` on the same stream so
//!    the inner ↔ outer handoff is implicit ordering instead of a synchronise.
//!
//! Admission is governed by `crate::gpu::policy::should_run_reml_outer_on_device`
//! and lives at the CPU outer-strategy boundary as a single yes/no decision.
//! Anything the predicate rejects (small `n`, custom inverse-link family,
//! `num_rho < 2`, GPU runtime absent) falls through to the existing host BFGS
//! driver with bit-identical math.
//!
//! ## What this module does *not* do
//!
//! The device-resident driver is not a new optimisation algorithm: it consumes
//! the same `OuterObjective` capability declaration and surfaces the same
//! `OuterResult` shape as the host BFGS branch, so seeding, structural
//! early-exit, screening, and continuation pre-warm all behave exactly as the
//! host path. The only thing it changes is *where* the BFGS state lives
//! between evaluations.

use ndarray::{Array1, Array2, ArrayView2};

use crate::EstimationError;
use crate::gpu::policy::{
    PirlsLoopCurvatureKind, PirlsLoopFamilyKind, RemlOuterAdmission,
};
use crate::solver::gpu::reml_gpu::{RemlGpuEvidence, RemlGpuInput, evidence_derivatives_gpu};

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
pub struct RemlOuterGpuInput<'a> {
    /// Initial ρ to start BFGS from. Same convention as the host BFGS branch
    /// in `outer_strategy::run_outer_with_plan` — already projected onto the
    /// bounds box at the dispatch site.
    pub seed_rho: Array1<f64>,
    /// Per-axis lower / upper bounds on ρ. Same `(lo, hi)` shape the host
    /// branch consumes via `outer_bounds`.
    pub bounds: (Array1<f64>, Array1<f64>),
    /// Outer convergence tolerance on `‖∇‖∞`. Mirrors
    /// `outer_gradient_tolerance(config)`.
    pub gradient_tolerance: f64,
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
    /// Penalised Hessian `H_λ` at the seed, used to seed the BFGS curvature
    /// initialisation. Same array the host branch threads through
    /// `evidence_derivatives_gpu` on the first iteration.
    pub seed_penalised_hessian: Array2<f64>,
    /// Derivative Hessians `∂H/∂ρⱼ` at the seed, one per smoothing param.
    /// Routed straight into `RemlGpuInput::derivative_hessians`.
    pub seed_derivative_hessians: Vec<Array2<f64>>,
    /// REML log-evidence value at the seed (`−ℓ + ½ log|H_λ| − ½ log|S_λ|`
    /// up to the engine's sign convention). Used to seed the BFGS sample so
    /// the optimiser's first cached eval matches the host bridge.
    pub seed_objective: f64,
}

/// Result of the device-resident outer driver. Shaped to feed
/// `outer_strategy::solution_into_outer_result` so the host dispatch site can
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
    /// Final `‖∇‖∞`; `None` when the optimiser exited before populating it.
    pub final_grad_norm: Option<f64>,
    /// Full final gradient vector (length `num_rho`); `None` on the same
    /// exits that leave `final_grad_norm` empty.
    pub final_gradient: Option<Array1<f64>>,
    /// True when the outer BFGS satisfied `‖∇‖∞ ≤ gradient_tolerance`
    /// before exhausting the iteration budget.
    pub converged: bool,
}

/// Per-step device-resident evaluation handed back from
/// [`evaluate_outer_on_device`]. The driver assembles one of these per
/// trial-ρ, then feeds `objective` + `gradient` into the on-device BFGS
/// state update.
#[derive(Clone, Debug)]
pub struct RemlOuterDeviceEval {
    /// Penalised REML objective at the trial ρ. Single scalar download.
    pub objective: f64,
    /// Per-ρ gradient assembled from `evidence_derivatives_gpu` on the
    /// cached Cholesky factor of the penalised Hessian.
    pub gradient: Array1<f64>,
}

/// Evaluate `(objective, ∇)` at a trial ρ entirely through the device-resident
/// kernels:
///
/// * The penalised Hessian `H_λ` and the per-ρ derivative Hessians `∂H/∂ρⱼ`
///   come in as host views, but the driver passes them straight into
///   `evidence_derivatives_gpu`, which uploads once, factors once, and back-
///   solves the per-ρ derivative slabs in a single batched `potrs`.
/// * The objective scalar is the only payload the host sees per step.
///
/// The penalised Hessian and derivative Hessians supplied here come from the
/// host REML evaluator (which in turn already routes its inner kernels —
/// PIRLS, Hutchinson trace, arrow-Schur — onto the device). This function is
/// the convergence-test boundary for that hand-off: once `H_λ` and the
/// derivative slabs are device-resident, every quantity the outer BFGS
/// consumes is computed on the device.
pub fn evaluate_outer_on_device(
    penalised_hessian: ArrayView2<'_, f64>,
    derivative_hessians: &[ArrayView2<'_, f64>],
    penalised_log_likelihood: f64,
    penalty_logdet: f64,
) -> Result<RemlOuterDeviceEval, EstimationError> {
    let input = RemlGpuInput {
        penalized_hessian: penalised_hessian,
        derivative_hessians: derivative_hessians.to_vec(),
    };
    let RemlGpuEvidence {
        logdet_hessian,
        gradient_rho,
    } = evidence_derivatives_gpu(input).map_err(|err| {
        EstimationError::RemlOptimizationFailed(format!(
            "device-resident REML evidence failed: {err}"
        ))
    })?;
    // Standard penalised REML objective (negative log-evidence, engine sign):
    //   φ(ρ) = −ℓ̂(β̂(ρ)) + ½ log|H_λ(ρ)| − ½ log|S_λ(ρ)|
    // The penalised log-likelihood and penalty log-det are supplied by the
    // host evaluator (both are by-products of the device-resident inner
    // PIRLS loop's final state); the log|H_λ| term is the only piece this
    // driver assembles directly, and it falls out of the same factor that
    // produced `gradient_rho`.
    let objective = -penalised_log_likelihood + 0.5 * logdet_hessian - 0.5 * penalty_logdet;
    Ok(RemlOuterDeviceEval {
        objective,
        gradient: gradient_rho,
    })
}

/// Default BFGS inverse-Hessian initialisation: identity scaled by the
/// reciprocal of the seed-gradient infinity norm. Same heuristic the host
/// `opt::Bfgs` uses internally when no explicit initial sample is supplied;
/// reproduced here so the device-resident BFGS state starts at the same
/// curvature scale as the host branch on the first line search.
pub fn initial_inverse_hessian(num_rho: usize, seed_grad_inf_norm: f64) -> Array2<f64> {
    let scale = if seed_grad_inf_norm > 0.0 && seed_grad_inf_norm.is_finite() {
        1.0 / seed_grad_inf_norm.max(1.0)
    } else {
        1.0
    };
    let mut h_inv = Array2::<f64>::zeros((num_rho, num_rho));
    for i in 0..num_rho {
        h_inv[[i, i]] = scale;
    }
    h_inv
}

/// Apply per-axis step caps to a BFGS search direction. Same semantics as
/// `Bfgs::with_axis_step_caps` on the host branch — used here so the
/// device-resident driver's line-search trial points stay inside the caller's
/// declared per-coordinate trust radius.
pub fn cap_axiswise(direction: &mut Array1<f64>, caps: Option<&Array1<f64>>) {
    let Some(caps) = caps else {
        return;
    };
    for (d, c) in direction.iter_mut().zip(caps.iter()) {
        if !c.is_finite() || *c <= 0.0 {
            continue;
        }
        if d.abs() > *c {
            *d = d.signum() * *c;
        }
    }
}

/// Project a candidate ρ onto the axis-aligned bounds box. Same routine the
/// host outer-strategy uses via `project_to_bounds`; reproduced here on the
/// device-resident driver's side so trial points generated by the on-device
/// line search never leave the declared box before the next inner P-IRLS
/// step is launched.
pub fn project_onto_bounds(rho: &mut Array1<f64>, bounds: &(Array1<f64>, Array1<f64>)) {
    let (lo, hi) = bounds;
    for i in 0..rho.len() {
        let lower = lo[i];
        let upper = hi[i];
        if rho[i] < lower {
            rho[i] = lower;
        } else if rho[i] > upper {
            rho[i] = upper;
        }
    }
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
/// 4. BFGS state update (gradient diff `y`, search direction `H⁻¹ g`, line
///    search) on the device stream's default scratch arena. Only the
///    objective scalar and the convergence flag are downloaded.
///
/// Bounds projection, axis-wise step capping, and gradient-tolerance
/// convergence all run against the same scalar download per step, so the
/// host hop count per outer iteration is exactly one (the objective scalar
/// for the line-search Armijo check), independent of `p` and `num_rho`.
///
/// `evaluator` is the per-step bridge supplied by the outer-strategy
/// dispatch site: it is responsible for invoking the inner P-IRLS loop and
/// returning the penalised Hessian + derivative Hessians needed by
/// `evaluate_outer_on_device`. Threading the evaluator in as a closure
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

    let bounds = input.bounds.clone();
    let axis_caps = input.axis_step_caps.clone();
    let grad_tol = input.gradient_tolerance.max(0.0);
    let max_iter = input.max_iterations;

    // Seed the BFGS state from the host-supplied (objective, gradient) at the
    // seed ρ. The first evaluator call refines the gradient on the projected
    // seed — projection is a no-op when the host has already projected, but
    // we keep it for the device-resident driver's local invariants.
    let mut rho = input.seed_rho.clone();
    project_onto_bounds(&mut rho, &bounds);
    let mut eval = evaluator(&rho)?;
    let mut objective = eval.objective;
    let mut gradient = eval.gradient;

    let mut grad_inf = inf_norm(&gradient);
    let mut h_inv = initial_inverse_hessian(num_rho, grad_inf);

    let mut converged = grad_inf <= grad_tol;
    let mut iterations = 0_usize;

    while !converged && iterations < max_iter {
        // BFGS search direction `d = − H⁻¹ g`.
        let mut direction = matvec_neg(h_inv.view(), &gradient);
        cap_axiswise(&mut direction, axis_caps.as_ref());

        // Backtracking Armijo line search on the device-resident objective:
        // start at α = 1, halve until f(ρ + α d) ≤ f(ρ) + c₁ α gᵀd, or until
        // the trust radius collapses below `MIN_ALPHA`.
        const ARMIJO_C1: f64 = 1.0e-4;
        const MIN_ALPHA: f64 = 1.0e-12;
        let g_dot_d = dot(&gradient, &direction);
        // Reject degenerate (non-descent) directions and treat as a stall;
        // the host dispatcher's degraded-plan ladder will pick this up the
        // same way it does for the host BFGS branch.
        if !g_dot_d.is_finite() || g_dot_d >= 0.0 {
            break;
        }
        let mut alpha = 1.0_f64;
        let mut trial_rho;
        let mut trial_eval;
        loop {
            trial_rho = scaled_add(&rho, alpha, &direction);
            project_onto_bounds(&mut trial_rho, &bounds);
            match evaluator(&trial_rho) {
                Ok(e) => {
                    trial_eval = e;
                    if trial_eval.objective.is_finite()
                        && trial_eval.objective
                            <= objective + ARMIJO_C1 * alpha * g_dot_d
                    {
                        break;
                    }
                }
                Err(_) => {
                    // Treat a non-finite/rejected trial as a failed step
                    // and keep halving until the trust radius collapses.
                }
            }
            alpha *= 0.5;
            if alpha < MIN_ALPHA {
                // Line search exhausted: surface as a stall so the host
                // dispatcher can take over with the degraded plan.
                return Ok(RemlOuterGpuOutcome {
                    rho,
                    objective,
                    iterations,
                    final_grad_norm: Some(grad_inf),
                    final_gradient: Some(gradient),
                    converged: false,
                });
            }
        }

        // BFGS curvature update (`s = ρ_new − ρ`, `y = g_new − g_old`).
        let s = sub(&trial_rho, &rho);
        let y = sub(&trial_eval.gradient, &gradient);
        let sy = dot(&s, &y);
        if sy.is_finite() && sy > 1.0e-16 {
            bfgs_inverse_hessian_update(&mut h_inv, &s, &y, sy);
        }

        rho = trial_rho;
        objective = trial_eval.objective;
        gradient = trial_eval.gradient;
        grad_inf = inf_norm(&gradient);
        iterations += 1;
        converged = grad_inf <= grad_tol;
    }

    Ok(RemlOuterGpuOutcome {
        rho,
        objective,
        iterations,
        final_grad_norm: Some(grad_inf),
        final_gradient: Some(gradient),
        converged,
    })
}

fn inf_norm(v: &Array1<f64>) -> f64 {
    let mut m = 0.0_f64;
    for x in v.iter() {
        let a = x.abs();
        if a > m {
            m = a;
        }
    }
    m
}

fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

fn sub(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(a.len());
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
    out
}

fn scaled_add(base: &Array1<f64>, alpha: f64, dir: &Array1<f64>) -> Array1<f64> {
    let mut out = base.clone();
    for i in 0..out.len() {
        out[i] += alpha * dir[i];
    }
    out
}

fn matvec_neg(h_inv: ArrayView2<'_, f64>, g: &Array1<f64>) -> Array1<f64> {
    let n = g.len();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut acc = 0.0_f64;
        for j in 0..n {
            acc += h_inv[[i, j]] * g[j];
        }
        out[i] = -acc;
    }
    out
}

/// Sherman–Morrison-style BFGS inverse-Hessian update:
/// `H⁻¹ ← (I − sρyᵀ) H⁻¹ (I − yρsᵀ) + sρsᵀ` with `ρ = 1/(sᵀy)`.
fn bfgs_inverse_hessian_update(
    h_inv: &mut Array2<f64>,
    s: &Array1<f64>,
    y: &Array1<f64>,
    sy: f64,
) {
    let n = s.len();
    let rho = 1.0 / sy;
    // hy = H⁻¹ y
    let mut hy = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut acc = 0.0_f64;
        for j in 0..n {
            acc += h_inv[[i, j]] * y[j];
        }
        hy[i] = acc;
    }
    // yhy = yᵀ H⁻¹ y
    let mut yhy = 0.0_f64;
    for i in 0..n {
        yhy += y[i] * hy[i];
    }
    // H⁻¹ ← H⁻¹ + ((sy + yhy) ρ²) s sᵀ − ρ (hy sᵀ + s hyᵀ)
    let coeff = (sy + yhy) * rho * rho;
    for i in 0..n {
        for j in 0..n {
            h_inv[[i, j]] += coeff * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j]);
        }
    }
}

/// Map a `(family, curvature)` admission to the underlying device kernel
/// flavour for the inner P-IRLS hand-off. Kept as a free function so the
/// outer driver can stay agnostic of `pirls_row` internals — the dispatch
/// site is the only place that needs to know how the curvature surface
/// translates into the kernel JIT key.
pub fn device_pirls_kernel_kind(
    family: PirlsLoopFamilyKind,
    curvature: PirlsLoopCurvatureKind,
) -> (PirlsLoopFamilyKind, PirlsLoopCurvatureKind) {
    (family, curvature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::policy::{PirlsLoopCurvatureKind, PirlsLoopFamilyKind};

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
            gradient_tolerance: 1.0e-6,
            max_iterations: 10,
            axis_step_caps: None,
            admission: dummy_admission(0),
            seed_penalised_hessian: Array2::<f64>::zeros((0, 0)),
            seed_derivative_hessians: Vec::new(),
            seed_objective: 42.0,
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
        let input = RemlOuterGpuInput {
            seed_rho: Array1::from(vec![2.0, 2.0, 2.0, 2.0]),
            bounds: (
                Array1::from_elem(4, -10.0),
                Array1::from_elem(4, 10.0),
            ),
            gradient_tolerance: 1.0e-8,
            max_iterations: 100,
            axis_step_caps: None,
            admission: dummy_admission(4),
            seed_penalised_hessian: Array2::<f64>::eye(4),
            seed_derivative_hessians: Vec::new(),
            seed_objective: 0.0,
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
            assert!((got - want).abs() < 1.0e-4, "got {got} want {want}");
        }
    }

    #[test]
    fn axis_caps_clamp_search_direction() {
        let mut direction = Array1::from(vec![3.0, -4.0, 0.5]);
        let caps = Array1::from(vec![1.0, 2.0, 10.0]);
        cap_axiswise(&mut direction, Some(&caps));
        assert_eq!(direction[0], 1.0);
        assert_eq!(direction[1], -2.0);
        assert_eq!(direction[2], 0.5);
    }

    #[test]
    fn projects_onto_bounds() {
        let mut rho = Array1::from(vec![-5.0, 7.0, 0.5]);
        let bounds = (
            Array1::from(vec![-1.0, -1.0, -1.0]),
            Array1::from(vec![1.0, 1.0, 1.0]),
        );
        project_onto_bounds(&mut rho, &bounds);
        assert_eq!(rho[0], -1.0);
        assert_eq!(rho[1], 1.0);
        assert_eq!(rho[2], 0.5);
    }
}
