//! The inner Gauss-Newton step: the public `solve_arrow_newton_step_*` entry
//! points, LM/proximal escalation, mixed-precision iterative refinement, and
//! the Schur reduced-RHS / back-substitution kernels they drive.

use super::*;


/// Schur-eliminate the per-row latent block and solve with an explicit BA
/// mode, returning the factor cache alongside the increments.
///
/// This is the BA-grade entry point. Direct and Square-Root BA form the dense
/// reduced camera/shared system; InexactPCG applies the same Schur operator by
/// matvec and uses Jacobi-preconditioned Steihaug-CG, following Agarwal et al.
pub fn solve_arrow_newton_step_with_options(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<(Array1<f64>, Array1<f64>, ArrowFactorCache), ArrowSchurError> {
    if options.streaming_chunk_size.is_some() {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "streaming Arrow-Schur solve does not materialize the factor cache required by this entry point".to_string(),
        });
    }
    // #1038 cross-row IBP: when the system carries the exact rank-`R` source, the
    // evidence base must be the NO-SELF `H₀'` (per-row logit-slot self term
    // `d_k·z'_ik²` downdated), so the full rank-one outer product `U D Uᵀ` — which
    // re-adds the `i=j` diagonal — does not double-count. We factor against `H₀'`,
    // then layer the exact Woodbury correction (value + logdet + adjoint) onto the
    // resulting cache. The Newton step is corrected to `H_full⁻¹(−g)` below so the
    // returned step and the reported curvature describe the SAME `H_full`.
    let downdated_owner;
    let (sys, ibp_source): (&ArrowSchurSystem, Option<&IbpCrossRowSource>) =
        match sys.ibp_cross_row.as_ref() {
            Some(source) => {
                let mut downdated = sys.clone();
                let total_len = downdated.row_offsets[downdated.rows.len()];
                let down = source.self_term_downdate(total_len);
                let offsets = Arc::clone(&downdated.row_offsets);
                for (i, row) in downdated.rows.iter_mut().enumerate() {
                    let base = offsets[i];
                    let di = row.htt.nrows();
                    for j in 0..di {
                        row.htt[[j, j]] -= down[base + j];
                    }
                }
                // The downdated rows carry a new curvature fingerprint.
                downdated.refresh_row_hessian_fingerprint();
                downdated_owner = downdated;
                (&downdated_owner, Some(source))
            }
            None => (sys, None),
        };
    let step = solve_arrow_newton_step_artifacts(sys, ridge_t, ridge_beta, options)?;
    let backend = CpuBatchedBlockSolver;

    let htbeta_estimated_bytes =
        estimated_htbeta_bytes(sys.rows.len(), sys.d, sys.k).unwrap_or(usize::MAX);
    let htbeta = if let Some(op) = sys.htbeta_matvec.as_ref() {
        ArrowHtbetaCache::Matvec {
            op: Arc::clone(op),
            estimated_bytes: htbeta_estimated_bytes,
        }
    } else if htbeta_estimated_bytes <= ARROW_FACTOR_CACHE_HTBETA_BUDGET_BYTES {
        ArrowHtbetaCache::Dense {
            blocks: sys
                .rows
                .iter()
                .map(|r| r.htbeta.clone())
                .collect::<Vec<_>>()
                .into(),
            estimated_bytes: htbeta_estimated_bytes,
        }
    } else {
        ArrowHtbetaCache::Disabled {
            estimated_bytes: htbeta_estimated_bytes,
        }
    };
    // Factor the UNDAMPED per-row blocks for the IFT predictor. When
    // ridge_t was zero the damped and undamped factors coincide and we
    // can alias htt_factors directly; otherwise pay a second per-row
    // Cholesky (O(N d³), same complexity class as the Newton solve).
    let htt_factors = step.htt_factors;
    let (htt_factors_undamped, gauge_deflated_directions) = if ridge_t == 0.0 {
        (
            ArrowUndampedFactors::SameAsDamped,
            step.gauge_deflated_directions,
        )
    } else {
        let undamped = factor_blocks_for_system(sys, 0.0, options, &backend)?;
        (
            ArrowUndampedFactors::Owned(undamped.factors),
            undamped.gauge_deflated_directions,
        )
    };
    let mut cache = ArrowFactorCache {
        htt_factors,
        htt_factors_undamped,
        schur_factor: step.schur_factor,
        solver_mode: options.mode,
        ridge_t,
        ridge_beta,
        htbeta,
        d: sys.d,
        row_dims: Arc::clone(&sys.row_dims),
        row_offsets: Arc::clone(&sys.row_offsets),
        k: sys.k,
        manifold_mode_fingerprint: sys.manifold_mode_fingerprint,
        row_hessian_fingerprint: sys.current_row_hessian_fingerprint(),
        pcg_diagnostics: step.pcg_diagnostics,
        gauge_deflated_directions,
        cross_row_woodbury: None,
    };
    let mut delta_t = step.delta_t;
    let mut delta_beta = step.delta_beta;
    if let Some(source) = ibp_source {
        // The cache's per-row factors are now `H₀'`; build the exact rank-`R`
        // Woodbury (one back-solve per atom column + the `R×R` capacitance LU)
        // and store it so the logdet/inverse/adjoint all read the same
        // `H_full = H₀' + U D Uᵀ`.
        if let Some(woodbury) = CrossRowWoodbury::build(&cache, source)? {
            // Correct the Newton step from `H₀'⁻¹(−g)` to `H_full⁻¹(−g)`. The base
            // `step.delta_t/β` solve `H₀' Δ₀ = −g`, so `delta_t` is the `t`-block
            // of `H₀'⁻¹(−g)`; the rank-`R` Woodbury inverse correction reads
            // `Uᵀ Δ₀ₜ` and writes both the `t` and `β` blocks of `H_full⁻¹(−g)`.
            let h0inv_neg_g_t = delta_t.clone();
            woodbury.apply_inverse_correction(
                h0inv_neg_g_t.view(),
                &source.entries,
                &mut delta_t,
                &mut delta_beta,
            )?;
            cache.cross_row_woodbury = Some(woodbury);
        }
    }
    Ok((delta_t, delta_beta, cache))
}


pub(crate) fn estimated_htbeta_bytes(n: usize, d: usize, k: usize) -> Option<usize> {
    n.checked_mul(d)?
        .checked_mul(k)?
        .checked_mul(std::mem::size_of::<f64>())
}


/// Schur-eliminate the per-row latent block and solve with explicit options,
/// returning `(Δt, Δβ, PcgDiagnostics)`.
///
/// The diagnostics are zero-valued (default) when the selected mode is
/// `Direct` or `SqrtBA` — use them to monitor `InexactPCG` iteration counts
/// and preconditioner escalation in production solves. Callers that do not
/// need diagnostics may pattern-match only the first two tuple elements.
pub fn solve_arrow_newton_step_core(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<(Array1<f64>, Array1<f64>, PcgDiagnostics), ArrowSchurError> {
    if let Some(chunk_size) = options.streaming_chunk_size {
        // #1014: the streaming/residency path is the memory-bound assembly wall,
        // so its reduced dense Schur solve runs certified mixed precision by
        // default (κ-gated f32 factor + f64 residual refinement, automatic f64
        // fallback). The reduced-Schur f64 factor — and therefore every evidence
        // log-determinant — is unaffected: only the Δβ solve drops to f32. An
        // explicit caller policy is honored as-is.
        let streaming_options = options.with_streaming_mixed_precision_default();
        let mut streaming = StreamingArrowSchur::from_system(sys, chunk_size);
        return streaming
            .solve(ridge_t, ridge_beta, &streaming_options)
            .map(|(delta_t, delta_beta, _)| (delta_t, delta_beta, PcgDiagnostics::default()));
    }
    // #1017 phase-3 production seam: when a device is present and the dense
    // Schur work clears the work-based dispatch threshold (LLM/SAE shapes —
    // few rows, thousands of border columns), route the Direct-mode point solve
    // through the fully device-resident batched Arrow-Schur sequence. The host
    // never sees the factors here (this `_core` entry discards the IFT cache),
    // so the device's scalars-only `(Δt, Δβ)` readback is exactly the contract.
    // Magic-by-default: no flag — the predicate fires from the shape. Any
    // non-admission or device failure falls through to the bit-identical CPU
    // path below, so the numbers are unchanged when the device declines.
    if let Some(device_step) = try_device_arrow_direct(sys, ridge_t, ridge_beta, options) {
        return device_step;
    }
    // #1017 production seam for the matrix-free SAE path: the real SAE decoder
    // β-block is the Kronecker operator (`htbeta_matvec`), never a dense slab,
    // so the dense device-resident solve above declines and the mode is
    // `InexactPCG`. The reduced-Schur matvec `Σ_i Y_i^T(Y_i x)` is the PCG hot
    // loop and is exactly what `gpu_schur_matvec_backend` offloads (dense rows)
    // or the row-procedural Kronecker apply handles (matrix-free). When the
    // device admits and the caller did not already supply a matvec, build one
    // and inject it through a cloned options so the existing InexactPCG branch
    // consumes it. On any device decline the original (CPU) options are used
    // unchanged, so results are bit-identical.
    if let Some(device_options) = maybe_inject_gpu_schur_matvec(sys, ridge_t, ridge_beta, options) {
        return solve_arrow_newton_step_artifacts(sys, ridge_t, ridge_beta, &device_options).map(
            |step| {
                let mut diagnostics = step.pcg_diagnostics;
                diagnostics.used_device_arrow = true;
                (step.delta_t, step.delta_beta, diagnostics)
            },
        );
    }
    solve_arrow_newton_step_artifacts(sys, ridge_t, ridge_beta, options)
        .map(|step| (step.delta_t, step.delta_beta, step.pcg_diagnostics))
}


/// Build and inject the GPU reduced-Schur matvec backend for an admitted
/// `InexactPCG` solve, returning a cloned `ArrowSolveOptions` carrying it.
///
/// Returns `None` (caller keeps the original CPU options) when: the mode is not
/// `InexactPCG`; the caller already supplied a `gpu_matvec`; no device is
/// present; the work-based predicate declines the shape; or the backend build
/// fails for any reason. The PCG numerics are identical whether the matvec runs
/// on host or device (same reduced Schur operator, same f64 accumulation), so
/// injecting it changes only where the `Σ_i Y_i^T(Y_i x)` flops execute.
pub(crate) fn maybe_inject_gpu_schur_matvec(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Option<ArrowSolveOptions> {
    if options.mode != ArrowSolverMode::InexactPCG || options.gpu_matvec.is_some() {
        return None;
    }
    if !sys.cross_row_penalties.is_empty() || options.streaming_chunk_size.is_some() {
        return None;
    }
    let runtime = crate::gpu::runtime::GpuRuntime::global()?;
    // #1017 Phase-1 call-site re-key: the reduced-Schur matvec is `O(n · d · k)`
    // per apply and the PCG runs `cg_iters` applies over device-resident frames,
    // so the offload becomes profitable on the CG-AMORTISED batched work — the
    // exact `n × k × d` arithmetic the dense-Direct `(n, k)` floor misses (it
    // ignores the per-row frame depth `d` and the `1/cg_iters` staging
    // amortisation). The CG budget here is the same `max_iterations` the PCG loop
    // launches with (`pcg.max_iterations.min(trust_region.max_iterations)`).
    // `try_device_arrow_direct` deliberately keeps the dense gate — that path is
    // one large factorization, not the amortised matvec.
    let cg_iters = options
        .pcg
        .max_iterations
        .min(options.trust_region.max_iterations);
    if !runtime
        .policy()
        .reduced_schur_matvec_should_offload(sys.rows.len(), sys.k, sys.d, cg_iters)
    {
        return None;
    }
    let matvec =
        crate::gpu::arrow_schur::gpu_schur_matvec_backend(sys, ridge_t, ridge_beta).ok()?;
    let mut device_options = options.clone();
    device_options.gpu_matvec = Some(matvec);
    Some(device_options)
}


/// Admission + dispatch for the device-resident Direct Arrow-Schur point solve.
///
/// Returns `Some(Ok(..))` when the device path produced a step, `Some(Err(..))`
/// only for a genuine numerical failure the device surfaced that the caller
/// must see (a non-PD pivot the LM escalation should respond to), and `None`
/// when the device declined for any reason — shape below threshold, no CUDA,
/// matrix-free operators present, or a transient device-unavailable — so the
/// caller transparently falls back to the CPU path.
///
/// The predicate is the same work-based gate the device-resident PIRLS loop
/// uses (`dense_hessian_work_target_is_gpu`) keyed on `(n_rows, border_k)`:
/// the reduced Schur assembly is `O(n · d · k²)`, dominated by the `k²` border,
/// so `k` is the dense-Hessian width. Below `DEVICE_LOOP_MIN_P` border columns
/// or below the flop floor the launch/staging overhead loses to the CPU dense
/// Cholesky, so the device is not engaged.
pub(crate) fn try_device_arrow_direct(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Option<Result<(Array1<f64>, Array1<f64>, PcgDiagnostics), ArrowSchurError>> {
    // Only the dense Direct mode maps onto the device dense-Schur sequence.
    // SqrtBA / InexactPCG have distinct numerics (square-root factors,
    // truncated-CG trust region) and must stay on their CPU implementations so
    // results are unchanged.
    if options.mode != ArrowSolverMode::Direct {
        return None;
    }
    // Cross-row penalties, streaming, and matrix-free H_ββ / H_tβ operators are
    // all outside the dense device path; the GPU entry itself rejects the
    // matrix-free cases, but short-circuit here so we never pay a device probe
    // for a system that cannot route.
    if !sys.cross_row_penalties.is_empty()
        || options.streaming_chunk_size.is_some()
        || sys.hbb_matvec.is_some()
        || sys.htbeta_matvec.is_some()
    {
        return None;
    }
    let runtime = crate::gpu::runtime::GpuRuntime::global()?;
    let admitted = runtime
        .policy()
        .dense_hessian_work_target_is_gpu(sys.rows.len(), sys.k);
    if !admitted {
        return None;
    }
    match crate::gpu::arrow_schur::solve_arrow_newton_step(sys, ridge_t, ridge_beta) {
        Ok(solution) => {
            let diagnostics = PcgDiagnostics {
                used_device_arrow: true,
                ..PcgDiagnostics::default()
            };
            Some(Ok((solution.delta_t, solution.delta_beta, diagnostics)))
        }
        // A non-PD per-row block or Schur pivot is a real numerical condition
        // the LM escalation around this solve must respond to; surface it as the
        // matching CPU error variant so `solve_with_lm_escalation_inner` bumps
        // the ridge and retries (it re-enters here and may route to device again
        // at the larger ridge, or fall to CPU if the device keeps declining).
        Err(crate::gpu::arrow_schur::ArrowSchurGpuFailure::RidgeBumpRequired { row, bump }) => {
            Some(Err(ArrowSchurError::PerRowFactorFailed {
                row,
                reason: format!("device per-row block non-PD; suggested ridge bump {bump:e}"),
            }))
        }
        // A non-PD reduced Schur is a real numerical condition the LM escalation
        // must respond to (bump the β-ridge and retry); surface it as the
        // matching CPU error rather than re-running the same factorisation on
        // the CPU only to fail identically.
        Err(crate::gpu::arrow_schur::ArrowSchurGpuFailure::SchurFactorFailed { reason }) => {
            Some(Err(ArrowSchurError::SchurFactorFailed { reason }))
        }
        // Unavailable (transient / below device policy) and
        // GpuRequiresDenseSystem (matrix-free, already filtered above) both mean
        // "device declined" — fall back to CPU transparently.
        Err(_) => None,
    }
}


/// LM-style ridge escalation around `solve_arrow_newton_step_core`.
///
/// On `PerRowFactorFailed` / `PerRowFactorIllConditioned` /
/// `SchurFactorFailed` (the factorization-level failure modes triggered
/// when a per-row `H_tt + ridge_t·I` block is non-PD, barely-PD with a
/// condition estimate above the safe Schur threshold, or the reduced
/// Schur complement has a non-PD pivot at the nominal ridge),
/// geometrically grow a `proximal_ridge` on top of the caller-supplied
/// `ridge_t` / `ridge_beta` and retry, exactly as the Ceres-style proximal
/// correction the Newton driver in `run_joint_fit_arrow_schur` does around
/// `solve`. Adaptive-correction exhaustion surfaces immediately because it is
/// not recoverable by shifting the diagonal.
///
/// A `PcgFailed` is likewise treated as recoverable: when the inexact-PCG
/// path stalls (all preconditioner tiers hit `MaxIter`, negative curvature on
/// an unbounded solve, or a non-PD preconditioned residual), shifting the
/// diagonal improves both conditioning and curvature, so a ridge bump is the
/// right response. Only `AdaptiveCorrectionFailed` surfaces immediately, since
/// it is an option-validation / line-search failure that a ridge shift cannot
/// repair.
///
/// Returns `(Δt, Δβ, PcgDiagnostics)` from `solve_arrow_newton_step_core`,
/// computed with the smallest escalated ridge that produced a successful factor.
/// `PcgDiagnostics::ridge_escalations` records how many ridge bumps were needed.
pub fn solve_with_lm_escalation_inner(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<(Array1<f64>, Array1<f64>, PcgDiagnostics), ArrowSchurError> {
    let mut proximal_ridge = 0.0_f64;
    let mut escalations: usize = 0;
    let mut last_err: Option<ArrowSchurError> = None;
    for attempt in 0..=DEFAULT_PROXIMAL_MAX_ATTEMPTS {
        let damped_ridge_t = ridge_t + proximal_ridge;
        let damped_ridge_beta = ridge_beta + proximal_ridge;
        match solve_arrow_newton_step_artifacts(sys, damped_ridge_t, damped_ridge_beta, options) {
            Ok(mut step) => {
                step.pcg_diagnostics.ridge_escalations = escalations;
                return Ok((step.delta_t, step.delta_beta, step.pcg_diagnostics));
            }
            Err(err) => {
                let recoverable = matches!(
                    err,
                    ArrowSchurError::PerRowFactorFailed { .. }
                        | ArrowSchurError::PerRowFactorIllConditioned { .. }
                        | ArrowSchurError::SchurFactorFailed { .. }
                        | ArrowSchurError::PcgFailed { .. }
                );
                last_err = Some(err);
                if !recoverable {
                    break;
                }
                if attempt == DEFAULT_PROXIMAL_MAX_ATTEMPTS {
                    break;
                }
                proximal_ridge = if proximal_ridge == 0.0 {
                    DEFAULT_PROXIMAL_INITIAL_RIDGE
                } else {
                    proximal_ridge * DEFAULT_PROXIMAL_RIDGE_GROWTH
                };
                escalations += 1;
            }
        }
    }
    Err(last_err.expect("escalation loop set last_err on failure"))
}


/// Solve a non-convex arrow-Schur step with adaptive proximal damping.
///
/// `trial_objective` receives the proposed `(delta_t, delta_beta)` and must
/// return the true nonlinear objective after applying that step. The function
/// increases a common proximal ridge until factorization succeeds, the
/// direction is descent, and Armijo decrease holds.
pub fn solve_arrow_newton_step_with_proximal_correction<F>(
    sys: &ArrowSchurSystem,
    base_ridge_t: f64,
    base_ridge_beta: f64,
    current_objective_value: f64,
    options: &ArrowSolveOptions,
    correction: &ArrowProximalCorrectionOptions,
    mut trial_objective: F,
) -> Result<ArrowAcceptedProximalStep, ArrowSchurError>
where
    F: for<'a, 'b> FnMut(ArrayView1<'a, f64>, ArrayView1<'b, f64>) -> f64,
{
    if !current_objective_value.is_finite() {
        return Err(ArrowSchurError::AdaptiveCorrectionFailed {
            reason: "current objective is not finite".to_string(),
        });
    }
    if !(correction.ridge_growth.is_finite() && correction.ridge_growth > 1.0) {
        return Err(ArrowSchurError::AdaptiveCorrectionFailed {
            reason: format!(
                "ridge_growth must be finite and > 1; got {}",
                correction.ridge_growth
            ),
        });
    }
    if !(correction.armijo_c1.is_finite()
        && correction.armijo_c1 > 0.0
        && correction.armijo_c1 < 1.0)
    {
        return Err(ArrowSchurError::AdaptiveCorrectionFailed {
            reason: format!("armijo_c1 must be in (0, 1); got {}", correction.armijo_c1),
        });
    }

    let grad_norm = arrow_gradient_norm(sys);
    if grad_norm <= correction.gradient_tolerance.max(0.0) {
        return Ok(ArrowAcceptedProximalStep {
            delta_t: Array1::<f64>::zeros(sys.row_offsets[sys.rows.len()]),
            delta_beta: Array1::<f64>::zeros(sys.k),
            ridge_t: base_ridge_t,
            ridge_beta: base_ridge_beta,
            proximal_ridge: 0.0,
            objective_value: current_objective_value,
            trial_objective_value: current_objective_value,
            gradient_dot_step: 0.0,
            attempts: 0,
        });
    }

    // Objective-scale resolution: the floating-point granularity of the
    // penalised objective at the incumbent value. Decreases smaller than this
    // are indistinguishable from rounding noise; increases smaller than this
    // are pure rounding and indicate the incumbent is already a (numerical)
    // stationary point.
    let objective_resolution =
        correction.convergence_objective_rel_tol.max(0.0) * (current_objective_value.abs() + 1.0);

    let mut proximal_ridge = correction.initial_ridge.max(0.0);
    let mut last_reason = String::from("no attempts were made");
    // Best strictly-decreasing trial seen across all ridge attempts. The Armijo
    // sufficient-decrease test can reject a step that nonetheless lowers the
    // objective; in the heavily-damped near-stationary regime, banking any
    // genuine decrease is a valid (relaxed) globalisation, so we retain the
    // best such candidate as a fallback to the strict Armijo accept.
    // Tuple: (delta_t, delta_beta, trial_value, g_dot_p, ridge_t, ridge_beta,
    // proximal_ridge) — the full step record for the best attempt so the
    // returned damping metadata matches the step actually banked.
    let mut best_decrease: Option<(Array1<f64>, Array1<f64>, f64, f64, f64, f64, f64)> = None;
    // Smallest objective INCREASE observed (over attempts that produced a
    // finite trial value but did not decrease). If even the best attempt only
    // raises the objective, but by no more than the objective resolution, the
    // incumbent is numerically stationary and we converge in place.
    let mut smallest_increase = f64::INFINITY;
    for attempt in 0..correction.max_attempts {
        let ridge_t = base_ridge_t + proximal_ridge;
        let ridge_beta = base_ridge_beta + proximal_ridge;
        match solve_arrow_newton_step_core(sys, ridge_t, ridge_beta, options) {
            Ok((delta_t, delta_beta, _diag)) => {
                let g_dot_p = arrow_gradient_dot_step(sys, delta_t.view(), delta_beta.view());
                if !(g_dot_p.is_finite() && g_dot_p < 0.0) {
                    last_reason =
                        format!("candidate was not a finite descent direction: g·p={g_dot_p}");
                } else {
                    let trial_value = trial_objective(delta_t.view(), delta_beta.view());
                    let armijo_bound = current_objective_value + correction.armijo_c1 * g_dot_p;
                    if trial_value.is_finite() && trial_value <= armijo_bound {
                        return Ok(ArrowAcceptedProximalStep {
                            delta_t,
                            delta_beta,
                            ridge_t,
                            ridge_beta,
                            proximal_ridge,
                            objective_value: current_objective_value,
                            trial_objective_value: trial_value,
                            gradient_dot_step: g_dot_p,
                            attempts: attempt + 1,
                        });
                    }
                    if trial_value.is_finite() {
                        let delta_obj = trial_value - current_objective_value;
                        if delta_obj < -objective_resolution {
                            // Genuine (Armijo-failing) decrease: keep the best.
                            let improves = best_decrease.as_ref().is_none_or(
                                |(_, _, best_value, _, _, _, _)| trial_value < *best_value,
                            );
                            if improves {
                                best_decrease = Some((
                                    delta_t.clone(),
                                    delta_beta.clone(),
                                    trial_value,
                                    g_dot_p,
                                    ridge_t,
                                    ridge_beta,
                                    proximal_ridge,
                                ));
                            }
                        } else if delta_obj < smallest_increase {
                            smallest_increase = delta_obj;
                        }
                    }
                    last_reason = {
                        let step_norm = (delta_t.iter().map(|v| v * v).sum::<f64>()
                            + delta_beta.iter().map(|v| v * v).sum::<f64>())
                        .sqrt();
                        format!(
                            "Armijo rejected trial objective {trial_value}; bound {armijo_bound}; \
                             |g|={grad_norm:.4e} g.p={g_dot_p:.4e} |step|={step_norm:.4e} ridge={proximal_ridge:.3e}"
                        )
                    };
                }
            }
            Err(err) => {
                last_reason = err.to_string();
            }
        }
        proximal_ridge = next_proximal_ridge(proximal_ridge, correction.ridge_growth);
    }

    // ── Fallback 1: bank the best genuine (Armijo-failing) decrease ──────────
    // Re-apply the best decreasing step so `self` (the caller's state, mutated
    // through the `trial_objective` closure) is left exactly at that step; the
    // returned deltas describe the move from the incumbent.
    if let Some((delta_t, delta_beta, trial_value, g_dot_p, ridge_t, ridge_beta, best_ridge)) =
        best_decrease
    {
        let reapplied = trial_objective(delta_t.view(), delta_beta.view());
        // The closure is deterministic (restore-then-apply), so `reapplied`
        // matches the recorded `trial_value` up to rounding; trust the live
        // value to keep the returned record consistent with `self`'s state.
        let final_value = if reapplied.is_finite() {
            reapplied
        } else {
            trial_value
        };
        return Ok(ArrowAcceptedProximalStep {
            delta_t,
            delta_beta,
            ridge_t,
            ridge_beta,
            proximal_ridge: best_ridge,
            objective_value: current_objective_value,
            trial_objective_value: final_value,
            gradient_dot_step: g_dot_p,
            attempts: correction.max_attempts,
        });
    }

    // ── Fallback 2: near-stationary convergence exit ─────────────────────────
    // No attempt decreased the objective, but the best attempt raised it by no
    // more than the objective's own resolution. The damped Newton model cannot
    // make distinguishable progress: the incumbent is a numerical stationary
    // point. Return a zero step at the incumbent state so the caller accepts it
    // as converged instead of failing. (`smallest_increase` is finite only if
    // at least one descent direction produced a finite trial value.)
    if smallest_increase.is_finite() && smallest_increase <= objective_resolution {
        return Ok(ArrowAcceptedProximalStep {
            delta_t: Array1::<f64>::zeros(sys.row_offsets[sys.rows.len()]),
            delta_beta: Array1::<f64>::zeros(sys.k),
            ridge_t: base_ridge_t,
            ridge_beta: base_ridge_beta,
            proximal_ridge: 0.0,
            objective_value: current_objective_value,
            trial_objective_value: current_objective_value,
            gradient_dot_step: 0.0,
            attempts: correction.max_attempts,
        });
    }

    Err(ArrowSchurError::AdaptiveCorrectionFailed {
        reason: format!(
            "failed after {} attempts; last rejection: {last_reason}",
            correction.max_attempts
        ),
    })
}


/// Predicted reduction of the *damped* joint Arrow-Schur quadratic model.
///
/// Includes the LM ridge terms in the quadratic:
///
/// `m(δ) - m(0) = gᵀδ + 0.5 δᵀ(H + ridge)δ`
///
/// Use this only for internal LM rejection logic that needs the damped model
/// (e.g. checking whether a candidate step satisfies a trust-region condition
/// against the augmented quadratic). For gain-ratio computations against the
/// bare penalized objective, use [`arrow_bare_quadratic_model_reduction`].
pub fn arrow_damped_quadratic_model_reduction(
    sys: &ArrowSchurSystem,
    delta_t: ArrayView1<'_, f64>,
    delta_beta: ArrayView1<'_, f64>,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<f64, ArrowSchurError> {
    let total_len = sys.row_offsets[sys.rows.len()];
    assert_eq!(delta_t.len(), total_len);
    assert_eq!(delta_beta.len(), sys.k);
    let mut lin = sys.gb.dot(&delta_beta);
    let mut quad = ridge_beta * delta_beta.dot(&delta_beta);

    // Route H_ββ · Δβ through penalty_matvec_add (#296):
    // no Arc-clone; dispatches inline to penalty_op or hbb.
    let mut hbb_delta = Array1::<f64>::zeros(sys.k);
    {
        let x_slice = delta_beta
            .as_slice()
            .expect("delta_beta must be contiguous");
        let y_slice = hbb_delta
            .as_slice_mut()
            .expect("hbb_delta must be contiguous");
        sys.penalty_matvec_add(x_slice, y_slice);
    }
    quad += delta_beta.dot(&hbb_delta);

    // Allocate scratch at max_d; per-row slice is ..di.
    let mut htbeta_x = Array1::<f64>::zeros(sys.d);
    for (i, row) in sys.rows.iter().enumerate() {
        let di = sys.row_dims[i];
        let row_base = sys.row_offsets[i];
        // H_tβ^(i) · Δβ via helper (routes through htbeta_matvec when present).
        let mut htbeta_x_i = htbeta_x.slice_mut(ndarray::s![..di]).to_owned();
        htbeta_x_i.fill(0.0);
        sys_htbeta_apply_row(sys, i, row, delta_beta, &mut htbeta_x_i);
        for c in 0..di {
            let dt_c = delta_t[row_base + c];
            lin += row.gt[c] * dt_c;
            quad += ridge_t * dt_c * dt_c;
            for r in 0..di {
                quad += dt_c * row.htt[[c, r]] * delta_t[row_base + r];
            }
            quad += 2.0 * dt_c * htbeta_x_i[c];
        }
    }

    Ok(-(lin + 0.5 * quad))
}


/// Predicted reduction of the *bare* joint Arrow-Schur quadratic model.
///
/// Drops the LM ridge contributions from the quadratic so the predicted
/// reduction is measured against the same bare penalized objective that the
/// actual reduction is measured against:
///
/// `m_bare(δ) - m_bare(0) = gᵀδ + 0.5 δᵀH δ`
///
/// Implemented as:
///   damped_quad − 0.5·(ridge_beta·‖δβ‖² + ridge_t·‖δt‖²)
///
/// When #282 lands and damping becomes diagonal (`λD²` instead of scalar `λI`),
/// replace the scalar `ridge_beta` / `ridge_t` correction with
/// `0.5 · δβᵀ(D_beta²)δβ` and `0.5 · δtᵀ(D_t²)δt` respectively — the
/// structure of this function already accepts per-scalar corrections; passing
/// a per-coordinate D² diagonal merely requires looping over coordinates
/// instead of multiplying by the squared norm.
///
/// Use this for PIRLS gain-ratio computations and any other place where the
/// accept/reject criterion compares against the bare (non-augmented) objective.
pub fn arrow_bare_quadratic_model_reduction(
    sys: &ArrowSchurSystem,
    delta_t: ArrayView1<'_, f64>,
    delta_beta: ArrayView1<'_, f64>,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<f64, ArrowSchurError> {
    // Compute the damped version first, then subtract the ridge contributions
    // to recover the bare-H quadratic.  This mirrors the beta-only PIRLS path:
    //     δ'(H+λI)δ − λ‖δ‖² = δ'Hδ
    let damped =
        arrow_damped_quadratic_model_reduction(sys, delta_t, delta_beta, ridge_t, ridge_beta)?;
    // Subtract 0.5 * (ridge_beta * ‖δβ‖² + ridge_t * ‖δt‖²).
    // The sign convention: arrow_damped returns -(lin + 0.5*quad), so the
    // ridge terms enter with a negative sign there.  To remove them we add
    // back 0.5 * (ridge_beta * ‖δβ‖² + ridge_t * ‖δt‖²).
    let ridge_beta_contrib = 0.5 * ridge_beta * delta_beta.dot(&delta_beta);
    let ridge_t_contrib = {
        let mut acc = 0.0_f64;
        for v in delta_t.iter() {
            acc += v * v;
        }
        0.5 * ridge_t * acc
    };
    Ok(damped + ridge_beta_contrib + ridge_t_contrib)
}


pub(crate) fn next_proximal_ridge(current: f64, growth: f64) -> f64 {
    if current > 0.0 {
        current * growth
    } else {
        DEFAULT_PROXIMAL_INITIAL_RIDGE
    }
}


pub(crate) fn arrow_gradient_norm(sys: &ArrowSchurSystem) -> f64 {
    let mut sum = 0.0;
    for row in sys.rows.iter() {
        for &v in row.gt.iter() {
            sum += v * v;
        }
    }
    for &v in sys.gb.iter() {
        sum += v * v;
    }
    sum.sqrt()
}


pub(crate) fn arrow_gradient_dot_step(
    sys: &ArrowSchurSystem,
    delta_t: ArrayView1<'_, f64>,
    delta_beta: ArrayView1<'_, f64>,
) -> f64 {
    assert_eq!(delta_t.len(), sys.row_offsets[sys.rows.len()]);
    assert_eq!(delta_beta.len(), sys.k);
    let mut out = 0.0;
    for (i, row) in sys.rows.iter().enumerate() {
        let di = sys.row_dims[i];
        let row_base = sys.row_offsets[i];
        for c in 0..di {
            out += row.gt[c] * delta_t[row_base + c];
        }
    }
    for a in 0..sys.k {
        out += sys.gb[a] * delta_beta[a];
    }
    out
}


pub(crate) struct ArrowNewtonStepArtifacts {
    pub(crate) delta_t: Array1<f64>,
    pub(crate) delta_beta: Array1<f64>,
    pub(crate) htt_factors: ArrowFactorSlab,
    pub(crate) schur_factor: Option<Array2<f64>>,
    pub(crate) pcg_diagnostics: PcgDiagnostics,
    pub(crate) gauge_deflated_directions: usize,
}


pub(crate) struct ArrowBlockFactorization {
    pub(crate) factors: ArrowFactorSlab,
    pub(crate) gauge_deflated_directions: usize,
}


pub(crate) fn factor_blocks_for_system<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    options: &ArrowSolveOptions,
    backend: &B,
) -> Result<ArrowBlockFactorization, ArrowSchurError> {
    let Some(deflation) = sys.row_gauge_deflation.as_ref() else {
        return Ok(ArrowBlockFactorization {
            factors: backend.factor_blocks(
                &sys.rows,
                ridge_t,
                sys.d,
                options.tolerate_ill_conditioning,
            )?,
            gauge_deflated_directions: 0,
        });
    };
    let mut blocks = Vec::with_capacity(sys.rows.len());
    let mut count = 0usize;
    for (row_idx, row) in sys.rows.iter().enumerate() {
        let result = factor_one_row_result(
            row,
            ridge_t,
            sys.row_dims[row_idx],
            row_idx,
            options.tolerate_ill_conditioning,
            deflation.row(row_idx),
        )?;
        count += result.gauge_deflated_directions;
        blocks.push(result.factor);
    }
    Ok(ArrowBlockFactorization {
        factors: ArrowFactorSlab::from_blocks(blocks),
        gauge_deflated_directions: count,
    })
}


pub(crate) enum MixedPrecisionAttempt {
    Certified {
        delta_t: Array1<f64>,
        delta_beta: Array1<f64>,
        schur_factor: Array2<f64>,
        refinement_steps: usize,
    },
    Fallback {
        reason: String,
    },
}


pub(crate) fn back_substitute_delta_t<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    delta_beta: ArrayView1<'_, f64>,
    backend: &B,
) -> Array1<f64> {
    let n = sys.rows.len();
    let total_dt_len = sys.row_offsets[n];
    let mut delta_t = Array1::<f64>::zeros(total_dt_len);
    // `Δt_i = -(H_tt^(i))⁻¹ (g_t^(i) + H_tβ^(i) Δβ)` is row-block-independent:
    // each row writes only its own contiguous `delta_t[row_offsets[i]..]`
    // segment. Fan out over the SAE LLM row count with the same nesting guard
    // (`rayon::current_thread_index()`) and row-min gate the `schur_matvec` hot
    // loop uses (#1017), so the topology race's outer candidate fan-out is not
    // oversubscribed. Disjoint writes ⇒ no reduction, no run-to-run drift.
    let parallel = n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
    let solve_row = |i: usize, out: &mut [f64]| {
        let di = sys.row_dims[i];
        assert!(
            sys.rows[i].gt.len() == di,
            "back_substitute_delta_t: row {i} gt len {} != row dim {di}",
            sys.rows[i].gt.len()
        );
        let mut htbeta_slice = Array1::<f64>::zeros(di);
        sys_htbeta_apply_row(sys, i, &sys.rows[i], delta_beta, &mut htbeta_slice);
        let mut rhs = Array1::<f64>::zeros(di);
        for c in 0..di {
            rhs[c] = sys.rows[i].gt[c] + htbeta_slice[c];
        }
        let dt_i = backend.solve_block_vector(htt_factors.factor(i), rhs.view());
        for c in 0..di {
            out[c] = -dt_i[c];
        }
    };
    if parallel {
        use rayon::prelude::*;
        pub(crate) const CHUNK: usize = 64;
        let row_offsets = &sys.row_offsets;
        // `par_chunks_mut` over uniform chunks does not align with variable row
        // dims, so partition by row chunk and hand each chunk its own contiguous
        // output segment via `split_at_mut` keyed on `row_offsets`.
        let dt_slice = delta_t.as_slice_mut().expect("delta_t contiguous");
        let n_chunks = n.div_ceil(CHUNK);
        let mut remaining = dt_slice;
        let mut segments: Vec<(usize, &mut [f64])> = Vec::with_capacity(n_chunks);
        let mut prev_end = 0usize;
        for chunk in 0..n_chunks {
            let start = chunk * CHUNK;
            let end = (start + CHUNK).min(n);
            let seg_len = row_offsets[end] - row_offsets[start];
            assert!(
                prev_end == row_offsets[start],
                "back_substitute_delta_t: non-contiguous row segment at chunk start {start} \
                 (prev_end={prev_end}, row_offset={})",
                row_offsets[start]
            );
            let (seg, rest) = remaining.split_at_mut(seg_len);
            remaining = rest;
            segments.push((start, seg));
            prev_end = row_offsets[end];
        }
        segments.into_par_iter().for_each(|(start, seg)| {
            let end = (start + CHUNK).min(n);
            let mut local = 0usize;
            for i in start..end {
                let di = sys.row_dims[i];
                solve_row(i, &mut seg[local..local + di]);
                local += di;
            }
        });
    } else {
        for i in 0..n {
            let row_base = sys.row_offsets[i];
            let di = sys.row_dims[i];
            solve_row(
                i,
                delta_t
                    .as_slice_mut()
                    .expect("delta_t contiguous")
                    .get_mut(row_base..row_base + di)
                    .expect("row segment in bounds"),
            );
        }
    }
    delta_t
}


pub(crate) fn try_mixed_precision_arrow_solve(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    htt_factors: &ArrowFactorSlab,
    schur: &Array2<f64>,
    options: &ArrowSolveOptions,
) -> Result<Option<MixedPrecisionAttempt>, ArrowSchurError> {
    let MixedPrecisionPolicy::Certified {
        max_refinement_steps,
        residual_relative_tolerance,
        kappa_unit_roundoff_margin,
    } = options.mixed_precision
    else {
        return Ok(None);
    };

    if options.trust_region.radius.is_finite() {
        return Ok(Some(MixedPrecisionAttempt::Fallback {
            reason: "trust-region-truncated dense solves are not certified by the mixed-precision refinement path".to_string(),
        }));
    }

    let schur_factor =
        cholesky_lower(schur).map_err(|e| ArrowSchurError::SchurFactorFailed { reason: e })?;
    if !options.tolerate_ill_conditioning {
        let schur_kappa = cholesky_factor_kappa_estimate(&schur_factor);
        if !schur_kappa.is_finite() || schur_kappa > safe_spd_kappa_max(schur.nrows()) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "reduced Schur complement Cholesky succeeded but is ill-conditioned \
                     (kappa_estimate={schur_kappa:e}); accumulated per-row \
                     (H_tt)^-1 contamination would yield an inaccurate delta_beta"
                ),
            });
        }
    }

    if let Some(reason) =
        mixed_precision_kappa_gate_failure(htt_factors, &schur_factor, kappa_unit_roundoff_margin)
    {
        return Ok(Some(MixedPrecisionAttempt::Fallback { reason }));
    }

    let row_factors_f32 = arrow_factor_slab_to_f32(htt_factors);
    let schur_factor_f32 = schur_factor.mapv(|v| v as f32);
    let (rhs_t, rhs_beta) = arrow_rhs(sys);
    let mut x = solve_arrow_system_f32(
        sys,
        &row_factors_f32,
        &schur_factor_f32,
        rhs_t.view(),
        rhs_beta.view(),
    )?;
    let certificate_tol = residual_relative_tolerance
        .max(MIXED_PRECISION_CERTIFICATE_EPSILON_MULTIPLIER * f64::EPSILON);
    for refinement_steps in 0..=max_refinement_steps {
        let (res_t, res_beta) = arrow_residual(
            sys,
            ridge_t,
            ridge_beta,
            x.0.view(),
            x.1.view(),
            rhs_t.view(),
            rhs_beta.view(),
        );
        let certificate = arrow_backward_error_certificate(
            sys,
            ridge_t,
            ridge_beta,
            x.0.view(),
            x.1.view(),
            rhs_t.view(),
            rhs_beta.view(),
            res_t.view(),
            res_beta.view(),
        )?;
        if certificate <= certificate_tol {
            return Ok(Some(MixedPrecisionAttempt::Certified {
                delta_t: x.0,
                delta_beta: x.1,
                schur_factor,
                refinement_steps,
            }));
        }
        if refinement_steps == max_refinement_steps {
            return Ok(Some(MixedPrecisionAttempt::Fallback {
                reason: format!(
                    "f64 residual certificate did not converge after {max_refinement_steps} refinement steps \
                     (backward_error={certificate:e}, tolerance={certificate_tol:e})"
                ),
            }));
        }
        let correction = solve_arrow_system_f32(
            sys,
            &row_factors_f32,
            &schur_factor_f32,
            res_t.view(),
            res_beta.view(),
        )?;
        if !correction
            .0
            .iter()
            .chain(correction.1.iter())
            .all(|v| v.is_finite())
        {
            return Ok(Some(MixedPrecisionAttempt::Fallback {
                reason: "f32 refinement correction produced a non-finite value".to_string(),
            }));
        }
        for i in 0..x.0.len() {
            x.0[i] += correction.0[i];
        }
        for i in 0..x.1.len() {
            x.1[i] += correction.1[i];
        }
    }

    Ok(Some(MixedPrecisionAttempt::Fallback {
        reason: "mixed refinement loop exhausted without certification".to_string(),
    }))
}


pub(crate) fn mixed_precision_kappa_gate_failure(
    htt_factors: &ArrowFactorSlab,
    schur_factor: &Array2<f64>,
    margin: f64,
) -> Option<String> {
    let mut max_kappa = cholesky_factor_kappa_estimate(schur_factor);
    let mut min_pivot = lower_cholesky_min_pivot(schur_factor.view());
    let mut max_pivot = lower_cholesky_max_pivot(schur_factor.view());
    for factor in htt_factors.iter() {
        let owned = factor.to_owned();
        max_kappa = max_kappa.max(cholesky_factor_kappa_estimate(&owned));
        if let Some(pivot) = lower_cholesky_min_pivot(owned.view()) {
            min_pivot = Some(match min_pivot {
                Some(current) => current.min(pivot),
                None => pivot,
            });
        }
        if let Some(pivot) = lower_cholesky_max_pivot(owned.view()) {
            max_pivot = Some(match max_pivot {
                Some(current) => current.max(pivot),
                None => pivot,
            });
        }
    }
    if let (Some(min_pivot), Some(max_pivot)) = (min_pivot, max_pivot) {
        if min_pivot > 0.0 && max_pivot.is_finite() {
            max_kappa = max_kappa.max(max_pivot / min_pivot);
        } else {
            max_kappa = f64::INFINITY;
        }
    }
    let kappa_u = max_kappa * F32_UNIT_ROUNDOFF;
    let threshold = margin
        .min(MIXED_PRECISION_KAPPA_MARGIN_CEILING)
        .max(F32_UNIT_ROUNDOFF);
    if !(max_kappa.is_finite() && kappa_u < threshold) {
        Some(format!(
            "kappa gate refused f32 refinement: kappa_estimate={max_kappa:e}, \
             kappa*u_f32={kappa_u:e}, required < {threshold:e}"
        ))
    } else {
        None
    }
}


pub(crate) fn arrow_factor_slab_to_f32(htt_factors: &ArrowFactorSlab) -> Vec<Array2<f32>> {
    htt_factors
        .iter()
        .map(|factor| factor.mapv(|v| v as f32))
        .collect()
}


pub(crate) fn arrow_rhs(sys: &ArrowSchurSystem) -> (Array1<f64>, Array1<f64>) {
    let n = sys.rows.len();
    let mut rhs_t = Array1::<f64>::zeros(sys.row_offsets[n]);
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        for c in 0..di {
            rhs_t[base + c] = -sys.rows[i].gt[c];
        }
    }
    let mut rhs_beta = Array1::<f64>::zeros(sys.k);
    for c in 0..sys.k {
        rhs_beta[c] = -sys.gb[c];
    }
    (rhs_t, rhs_beta)
}


pub(crate) fn solve_arrow_system_f32(
    sys: &ArrowSchurSystem,
    row_factors: &[Array2<f32>],
    schur_factor: &Array2<f32>,
    rhs_t: ArrayView1<'_, f64>,
    rhs_beta: ArrayView1<'_, f64>,
) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
    let n = sys.rows.len();
    let mut y_rows = Vec::<Array1<f32>>::with_capacity(n);
    let mut reduced_beta = rhs_beta.mapv(|v| v as f32);
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        let rhs_i = rhs_t.slice(ndarray::s![base..base + di]).mapv(|v| v as f32);
        let y_i = cholesky_solve_lower_f32(&row_factors[i], &rhs_i);
        let htbeta = sys_htbeta_materialize_row(sys, i, &sys.rows[i])?.mapv(|v| v as f32);
        for beta_col in 0..sys.k {
            let mut acc = 0.0_f32;
            for row_axis in 0..di {
                acc += htbeta[[row_axis, beta_col]] * y_i[row_axis];
            }
            reduced_beta[beta_col] -= acc;
        }
        y_rows.push(y_i);
    }

    let x_beta_f32 = cholesky_solve_lower_f32(schur_factor, &reduced_beta);
    let mut x_t = Array1::<f64>::zeros(sys.row_offsets[n]);
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        let htbeta = sys_htbeta_materialize_row(sys, i, &sys.rows[i])?.mapv(|v| v as f32);
        let mut cross = Array1::<f32>::zeros(di);
        for row_axis in 0..di {
            let mut acc = 0.0_f32;
            for beta_col in 0..sys.k {
                acc += htbeta[[row_axis, beta_col]] * x_beta_f32[beta_col];
            }
            cross[row_axis] = acc;
        }
        let correction = cholesky_solve_lower_f32(&row_factors[i], &cross);
        for row_axis in 0..di {
            x_t[base + row_axis] = (y_rows[i][row_axis] - correction[row_axis]) as f64;
        }
    }
    let x_beta = x_beta_f32.mapv(|v| v as f64);
    Ok((x_t, x_beta))
}


pub(crate) fn cholesky_solve_lower_f32(l: &Array2<f32>, b: &Array1<f32>) -> Array1<f32> {
    let n = l.nrows();
    // Precondition: positive, finite factor diagonals (see
    // `cholesky_solve_vector_fixed`). The certified mixed-precision streaming
    // path refines in f64 and falls back when this f32 solve is not usable, but
    // guard the precondition loudly — always, release included — so a future
    // factor source that skips that refinement cannot divide by a
    // zero/non-finite pivot silently.
    assert!(
        (0..n).all(|i| l[[i, i]].is_finite() && l[[i, i]].abs() >= f32::MIN_POSITIVE),
        "cholesky_solve_lower_f32: factor diagonal must be finite and non-subnormal"
    );
    let mut y = Array1::<f32>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f32>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}


pub(crate) fn arrow_residual(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    x_t: ArrayView1<'_, f64>,
    x_beta: ArrayView1<'_, f64>,
    rhs_t: ArrayView1<'_, f64>,
    rhs_beta: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    let (ax_t, ax_beta) = arrow_operator_apply(sys, ridge_t, ridge_beta, x_t, x_beta);
    let mut res_t = rhs_t.to_owned();
    let mut res_beta = rhs_beta.to_owned();
    for i in 0..res_t.len() {
        res_t[i] -= ax_t[i];
    }
    for i in 0..res_beta.len() {
        res_beta[i] -= ax_beta[i];
    }
    (res_t, res_beta)
}


pub(crate) fn arrow_operator_apply(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    x_t: ArrayView1<'_, f64>,
    x_beta: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = sys.rows.len();
    let mut y_t = Array1::<f64>::zeros(sys.row_offsets[n]);
    let mut y_beta = Array1::<f64>::zeros(sys.k);
    {
        let x_slice = x_beta.as_slice().expect("x_beta contiguous");
        let y_slice = y_beta.as_slice_mut().expect("y_beta contiguous");
        sys.penalty_matvec_add(x_slice, y_slice);
    }
    for beta_col in 0..sys.k {
        y_beta[beta_col] += ridge_beta * x_beta[beta_col];
    }
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        let row = &sys.rows[i];
        for a in 0..di {
            let mut acc = ridge_t * x_t[base + a];
            for b in 0..di {
                acc += row.htt[[a, b]] * x_t[base + b];
            }
            y_t[base + a] = acc;
        }
        let mut htbeta_xb = Array1::<f64>::zeros(di);
        sys_htbeta_apply_row(sys, i, row, x_beta, &mut htbeta_xb);
        for a in 0..di {
            y_t[base + a] += htbeta_xb[a];
        }
        let x_ti = x_t.slice(ndarray::s![base..base + di]).to_owned();
        sys_htbeta_accumulate_transpose(sys, i, row, x_ti.view(), &mut y_beta);
    }
    (y_t, y_beta)
}


pub(crate) fn arrow_backward_error_certificate(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    x_t: ArrayView1<'_, f64>,
    x_beta: ArrayView1<'_, f64>,
    rhs_t: ArrayView1<'_, f64>,
    rhs_beta: ArrayView1<'_, f64>,
    res_t: ArrayView1<'_, f64>,
    res_beta: ArrayView1<'_, f64>,
) -> Result<f64, ArrowSchurError> {
    let residual_norm = infinity_norm_pair(res_t, res_beta);
    let operator_norm = arrow_operator_infinity_norm(sys, ridge_t, ridge_beta)?;
    let solution_norm = infinity_norm_pair(x_t, x_beta);
    let rhs_norm = infinity_norm_pair(rhs_t, rhs_beta);
    let denom = operator_norm * solution_norm + rhs_norm;
    if denom > 0.0 {
        Ok(residual_norm / denom)
    } else {
        Ok(residual_norm)
    }
}


pub(crate) fn infinity_norm_pair(lhs: ArrayView1<'_, f64>, rhs: ArrayView1<'_, f64>) -> f64 {
    let mut out = 0.0_f64;
    for &v in lhs.iter().chain(rhs.iter()) {
        out = out.max(v.abs());
    }
    out
}


pub(crate) fn arrow_operator_infinity_norm(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<f64, ArrowSchurError> {
    let mut out = 0.0_f64;
    for i in 0..sys.rows.len() {
        let di = sys.row_dims[i];
        let row = &sys.rows[i];
        let htbeta = sys_htbeta_materialize_row(sys, i, row)?;
        for a in 0..di {
            let mut row_sum = 0.0_f64;
            for b in 0..di {
                row_sum += row.htt[[a, b]].abs();
            }
            row_sum += ridge_t;
            for beta_col in 0..sys.k {
                row_sum += htbeta[[a, beta_col]].abs();
            }
            out = out.max(row_sum);
        }
    }
    let hbb = sys.effective_penalty_op().to_dense();
    for beta_row in 0..sys.k {
        let mut row_sum = 0.0_f64;
        for beta_col in 0..sys.k {
            row_sum += hbb[[beta_row, beta_col]].abs();
        }
        row_sum += ridge_beta;
        for i in 0..sys.rows.len() {
            let di = sys.row_dims[i];
            let htbeta = sys_htbeta_materialize_row(sys, i, &sys.rows[i])?;
            for a in 0..di {
                row_sum += htbeta[[a, beta_row]].abs();
            }
        }
        out = out.max(row_sum);
    }
    Ok(out)
}


pub(crate) fn solve_arrow_newton_step_artifacts(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<ArrowNewtonStepArtifacts, ArrowSchurError> {
    // Auto-select the cross-row path: when any registered Psi penalty couples
    // distinct latent rows, the exact one-shot Schur elimination (which assumes
    // each H_tt^(i) is independent) cannot represent the off-row Hessian blocks.
    // Route the FULL (t, β) Newton system through matrix-free preconditioned CG
    // with the exact arrow block-diagonal inverse as the preconditioner. No
    // flag: the route is implied by the captured cross-row penalty set.
    if !sys.cross_row_penalties.is_empty() {
        return solve_arrow_newton_step_cross_row(sys, ridge_t, ridge_beta, options);
    }
    if let Some(chunk_size) = options.streaming_chunk_size {
        let mut streaming = StreamingArrowSchur::from_system(sys, chunk_size);
        let (delta_t, delta_beta, schur_factor) = streaming.solve(ridge_t, ridge_beta, options)?;
        return Ok(ArrowNewtonStepArtifacts {
            delta_t,
            delta_beta,
            htt_factors: ArrowFactorSlab::from_blocks(Vec::new()),
            schur_factor,
            pcg_diagnostics: PcgDiagnostics::default(),
            gauge_deflated_directions: 0,
        });
    }
    let backend = CpuBatchedBlockSolver;

    // 1. BA point elimination: per-row Cholesky factors of
    // (H_tt^(i) + ridge_t · I).  `factor_blocks` reads the actual row
    // dimension from `row.htt.nrows()` so heterogeneous systems work.
    let block_factorization = factor_blocks_for_system(sys, ridge_t, options, &backend)?;
    let htt_factors = block_factorization.factors;
    let gauge_deflated_directions = block_factorization.gauge_deflated_directions;

    // 2. Reduced RHS r_β = -g_β + Σ_i H_βt^(i) (H_tt^(i))⁻¹ g_t^(i).
    let rhs_beta = reduced_rhs_beta(sys, &htt_factors, &backend);
    // The Schur solve is over the reduced β vector. Latent manifold metric
    // weights live on each d-dimensional t_i block, so the induced metric for
    // this β-only Steihaug problem is Euclidean.
    let trust_metric_weights = None;

    // 3. Solve reduced shared system using the selected BA mode.
    let mut mixed_precision_status = MixedPrecisionStatus::Off;
    let (delta_beta, schur_factor, mut pcg_diagnostics) = match options.mode {
        ArrowSolverMode::Direct => {
            let schur = build_dense_schur_direct(sys, &htt_factors, ridge_beta, &backend)?;
            if let Some(attempt) = try_mixed_precision_arrow_solve(
                sys,
                ridge_t,
                ridge_beta,
                &htt_factors,
                &schur,
                options,
            )? {
                match attempt {
                    MixedPrecisionAttempt::Certified {
                        delta_t,
                        delta_beta,
                        schur_factor,
                        refinement_steps,
                    } => {
                        let mut pcg_diagnostics = PcgDiagnostics::default();
                        pcg_diagnostics.mixed_precision_status =
                            MixedPrecisionStatus::Certified { refinement_steps };
                        return Ok(ArrowNewtonStepArtifacts {
                            delta_t,
                            delta_beta,
                            htt_factors,
                            schur_factor: Some(schur_factor),
                            pcg_diagnostics,
                            gauge_deflated_directions,
                        });
                    }
                    MixedPrecisionAttempt::Fallback { reason } => {
                        log::info!("arrow-Schur mixed precision fallback to f64: {reason}");
                        mixed_precision_status = MixedPrecisionStatus::F64Fallback;
                    }
                }
            }
            let (db, sf, diag) =
                solve_dense_reduced_system(&schur, &rhs_beta, options, trust_metric_weights)?;
            (db, sf, diag)
        }
        ArrowSolverMode::SqrtBA => {
            let schur = build_dense_schur_sqrt_ba(sys, &htt_factors, ridge_beta, &backend)?;
            if let Some(attempt) = try_mixed_precision_arrow_solve(
                sys,
                ridge_t,
                ridge_beta,
                &htt_factors,
                &schur,
                options,
            )? {
                match attempt {
                    MixedPrecisionAttempt::Certified {
                        delta_t,
                        delta_beta,
                        schur_factor,
                        refinement_steps,
                    } => {
                        let mut pcg_diagnostics = PcgDiagnostics::default();
                        pcg_diagnostics.mixed_precision_status =
                            MixedPrecisionStatus::Certified { refinement_steps };
                        return Ok(ArrowNewtonStepArtifacts {
                            delta_t,
                            delta_beta,
                            htt_factors,
                            schur_factor: Some(schur_factor),
                            pcg_diagnostics,
                            gauge_deflated_directions,
                        });
                    }
                    MixedPrecisionAttempt::Fallback { reason } => {
                        log::info!("arrow-Schur mixed precision fallback to f64: {reason}");
                        mixed_precision_status = MixedPrecisionStatus::F64Fallback;
                    }
                }
            }
            let (db, sf, diag) =
                solve_dense_reduced_system(&schur, &rhs_beta, options, trust_metric_weights)?;
            (db, sf, diag)
        }
        ArrowSolverMode::InexactPCG => {
            if options.mixed_precision.is_enabled() {
                log::info!(
                    "arrow-Schur mixed precision fallback to f64: InexactPCG does not expose a dense Schur factor for certified f32 refinement"
                );
                mixed_precision_status = MixedPrecisionStatus::F64Fallback;
            }
            // Auto-select preconditioner level: starts with JacobiPreconditioner
            // (Diagonal / BetaBlockJacobi) and escalates to ClusterJacobi or
            // AdditiveSchwarz when K > 100 and PCG exhausts max_iterations.
            if options.trust_region.radius == f64::INFINITY {
                if let Some(device_data) = sys.device_sae_pcg.as_ref() {
                    let max_iterations = options
                        .pcg
                        .max_iterations
                        .min(options.trust_region.max_iterations);
                    let relative_tolerance = options
                        .pcg
                        .relative_tolerance
                        .max(options.trust_region.steihaug_relative_tolerance);
                    if let Ok((delta, mut diag)) =
                        crate::gpu::arrow_schur::solve_sae_matrix_free_pcg(
                            sys,
                            device_data.as_ref(),
                            ridge_t,
                            ridge_beta,
                            &rhs_beta,
                            max_iterations,
                            relative_tolerance,
                        )
                    {
                        diag.used_device_arrow = true;
                        return Ok(ArrowNewtonStepArtifacts {
                            delta_t: back_substitute_delta_t(
                                sys,
                                &htt_factors,
                                delta.view(),
                                &backend,
                            ),
                            delta_beta: delta,
                            htt_factors,
                            schur_factor: None,
                            pcg_diagnostics: diag,
                            gauge_deflated_directions,
                        });
                    }
                }
            }
            let (delta, diag) = steihaug_pcg_auto(
                sys,
                &htt_factors,
                ridge_beta,
                &rhs_beta,
                &options.pcg,
                &options.trust_region,
                &backend,
                options.gpu_matvec.as_ref(),
                trust_metric_weights,
            )?;
            (delta, None, diag)
        }
    };
    if mixed_precision_status != MixedPrecisionStatus::Off {
        pcg_diagnostics.mixed_precision_status = mixed_precision_status;
    }

    // 4. Back-substitute Δt_i = -(H_tt^(i))⁻¹ (g_t^(i) + H_tβ^(i) Δβ).
    let delta_t = back_substitute_delta_t(sys, &htt_factors, delta_beta.view(), &backend);

    Ok(ArrowNewtonStepArtifacts {
        delta_t,
        delta_beta,
        htt_factors,
        schur_factor,
        pcg_diagnostics,
        gauge_deflated_directions,
    })
}


/// Exact inverse of the block-diagonal arrow operator `K0 + ridge`, used as
/// the preconditioner for the cross-row full-system CG.
///
/// Holds the per-row `H_tt^(i) + ridge_t·I` Cholesky factors and the dense
/// Schur-complement factor `S = (H_ββ + ridge_β·I) − Σ_i H_tβ^(i)ᵀ
/// (H_tt^(i))⁻¹ H_tβ^(i)`, so applying `M⁻¹` to an arbitrary RHS is a single
/// Schur back/forward substitution — exactly the algebra
/// [`solve_arrow_newton_step_artifacts`] performs, generalized to a free RHS.
pub(crate) struct ArrowBlockDiagInverse<'a, B: BatchedBlockSolver> {
    pub(crate) sys: &'a ArrowSchurSystem,
    pub(crate) backend: &'a B,
    pub(crate) htt_factors: ArrowFactorSlab,
    pub(crate) schur_factor: Array2<f64>,
}


impl<'a, B: BatchedBlockSolver> ArrowBlockDiagInverse<'a, B> {
    pub(crate) fn build(
        sys: &'a ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
        tolerate_ill_conditioning: bool,
        backend: &'a B,
    ) -> Result<Self, ArrowSchurError>
    where
        B: Sync,
    {
        let htt_factors =
            backend.factor_blocks(&sys.rows, ridge_t, sys.d, tolerate_ill_conditioning)?;
        let schur = build_dense_schur_direct(sys, &htt_factors, ridge_beta, backend)?;
        let schur_factor =
            cholesky_lower(&schur).map_err(|e| ArrowSchurError::SchurFactorFailed { reason: e })?;
        Ok(Self {
            sys,
            backend,
            htt_factors,
            schur_factor,
        })
    }

    /// Solve `(K0 + ridge) · [x_t; x_β] = [r_t; r_β]` exactly.
    ///
    /// `r_t` is flat row-major (`Σ_i row_dims[i]`); `r_β` is length `K`. The
    /// outputs `x_t` / `x_β` use the same layout.
    pub(crate) fn apply(
        &self,
        r_t: ArrayView1<'_, f64>,
        r_beta: ArrayView1<'_, f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let sys = self.sys;
        let n = sys.rows.len();
        let k = sys.k;
        // Reduced β RHS: r_β − Σ_i H_βt^(i) (H_tt^(i))⁻¹ r_t,i.
        let mut rhs_beta = r_beta.to_owned();
        for i in 0..n {
            let di = sys.row_dims[i];
            let base = sys.row_offsets[i];
            let r_ti = r_t.slice(ndarray::s![base..base + di]).to_owned();
            let u_i = self
                .backend
                .solve_block_vector(self.htt_factors.factor(i), r_ti.view());
            let mut acc = Array1::<f64>::zeros(k);
            sys_htbeta_accumulate_transpose(sys, i, &sys.rows[i], u_i.view(), &mut acc);
            for a in 0..k {
                rhs_beta[a] -= acc[a];
            }
        }
        // x_β = S⁻¹ rhs_β.
        let x_beta = cholesky_solve_lower(&self.schur_factor, &rhs_beta);
        // x_t,i = (H_tt^(i))⁻¹ (r_t,i − H_tβ^(i) x_β).
        let total_dt = sys.row_offsets[n];
        let mut x_t = Array1::<f64>::zeros(total_dt);
        let mut htbeta_xb = Array1::<f64>::zeros(sys.d);
        for i in 0..n {
            let di = sys.row_dims[i];
            let base = sys.row_offsets[i];
            for c in 0..di {
                htbeta_xb[c] = 0.0;
            }
            let mut slab = htbeta_xb.slice_mut(ndarray::s![..di]).to_owned();
            sys_htbeta_apply_row(sys, i, &sys.rows[i], x_beta.view(), &mut slab);
            let mut rhs_i = Array1::<f64>::zeros(di);
            for c in 0..di {
                rhs_i[c] = r_t[base + c] - slab[c];
            }
            let xi = self
                .backend
                .solve_block_vector(self.htt_factors.factor(i), rhs_i.view());
            for c in 0..di {
                x_t[base + c] = xi[c];
            }
        }
        (x_t, x_beta)
    }
}


/// Apply the full cross-row Newton operator `A = (K0 + ridge) + P_cross` to
/// `[x_t; x_β]`, writing `[y_t; y_β]`.
///
/// `(K0 + ridge)` is the block-diagonal arrow operator: per row
/// `y_t,i = (H_tt^(i) + ridge_t·I) x_t,i + H_tβ^(i) x_β`, and
/// `y_β = Σ_i H_βt^(i) x_t,i + (H_ββ + ridge_β·I) x_β`. `P_cross` adds the
/// captured cross-row penalty Hessian to the latent block only:
/// `y_t += P_cross · x_t`.
pub(crate) fn arrow_cross_row_matvec(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    x_t: ArrayView1<'_, f64>,
    x_beta: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = sys.rows.len();
    let k = sys.k;
    let total_dt = sys.row_offsets[n];
    let mut y_t = Array1::<f64>::zeros(total_dt);
    let mut y_beta = Array1::<f64>::zeros(k);
    let mut htbeta_xb = Array1::<f64>::zeros(sys.d);
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        let row = &sys.rows[i];
        // H_tt^(i) x_t,i + ridge_t x_t,i.
        for a in 0..di {
            let mut acc = ridge_t * x_t[base + a];
            for b in 0..di {
                acc += row.htt[[a, b]] * x_t[base + b];
            }
            y_t[base + a] = acc;
        }
        // + H_tβ^(i) x_β.
        for c in 0..di {
            htbeta_xb[c] = 0.0;
        }
        let mut slab = htbeta_xb.slice_mut(ndarray::s![..di]).to_owned();
        sys_htbeta_apply_row(sys, i, row, x_beta, &mut slab);
        for c in 0..di {
            y_t[base + c] += slab[c];
        }
        // y_β += H_βt^(i) x_t,i.
        let x_ti = x_t.slice(ndarray::s![base..base + di]).to_owned();
        sys_htbeta_accumulate_transpose(sys, i, row, x_ti.view(), &mut y_beta);
    }
    // y_β += (H_ββ + ridge_β·I) x_β.
    {
        let x_beta_slice = x_beta.as_slice().expect("x_beta contiguous");
        let y_beta_slice = y_beta.as_slice_mut().expect("y_beta contiguous");
        sys.penalty_matvec_add(x_beta_slice, y_beta_slice);
    }
    for a in 0..k {
        y_beta[a] += ridge_beta * x_beta[a];
    }
    // y_t += P_cross · x_t (cross-row penalty Hessian, latent block only).
    sys.apply_cross_row_penalty_hessian(x_t, &mut y_t);
    (y_t, y_beta)
}


/// Solve the full bordered Newton system when one or more registered Psi
/// penalties couple distinct latent rows.
///
/// The operator is `A = (K0 + ridge) + P_cross`, SPD whenever the arrow
/// block-diagonal `K0 + ridge` is PD (enforced by the per-row factor checks)
/// and every cross-row penalty contributes a PSD `psd_majorizer_hvp`. We solve
/// `A · [Δt; Δβ] = −[g_t; g_β]` by preconditioned conjugate gradients, using
/// the exact arrow block-diagonal inverse `M⁻¹ = (K0 + ridge)⁻¹` as the
/// preconditioner — the same Schur elimination the row-block-diagonal path
/// uses, here applied to the CG residual rather than the negated gradient.
/// Because `M⁻¹` inverts everything except the (small, structured) `P_cross`
/// coupling, the preconditioned operator `M⁻¹ A = I + M⁻¹ P_cross` has a
/// tightly clustered spectrum and CG converges in a handful of iterations.
pub(crate) fn solve_arrow_newton_step_cross_row(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<ArrowNewtonStepArtifacts, ArrowSchurError> {
    let backend = CpuBatchedBlockSolver;
    let precond = ArrowBlockDiagInverse::build(
        sys,
        ridge_t,
        ridge_beta,
        options.tolerate_ill_conditioning,
        &backend,
    )?;

    let n = sys.rows.len();
    let k = sys.k;
    let total_dt = sys.row_offsets[n];

    // RHS b = −g = [−g_t; −g_β].
    let mut b_t = Array1::<f64>::zeros(total_dt);
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        for c in 0..di {
            b_t[base + c] = -sys.rows[i].gt[c];
        }
    }
    let mut b_beta = Array1::<f64>::zeros(k);
    for a in 0..k {
        b_beta[a] = -sys.gb[a];
    }

    // Preconditioned CG on the full (t, β) system.
    // x = 0; r = b − A·0 = b; z = M⁻¹ r; p = z.
    let mut x_t = Array1::<f64>::zeros(total_dt);
    let mut x_beta = Array1::<f64>::zeros(k);
    let mut r_t = b_t.clone();
    let mut r_beta = b_beta.clone();
    let (mut z_t, mut z_beta) = precond.apply(r_t.view(), r_beta.view());
    let mut p_t = z_t.clone();
    let mut p_beta = z_beta.clone();
    let mut rz = dot2(&r_t, &r_beta, &z_t, &z_beta);

    let b_norm = (dot2(&b_t, &b_beta, &b_t, &b_beta)).sqrt();
    // Solve the linear Newton system to tight relative accuracy. The cross-row
    // path is exact-CG (no trust region), so we drive the residual to machine-
    // scale relative tolerance; the spectrum I + M⁻¹P_cross makes this cheap.
    // Absolute floor guards b_norm → 0; relative term tracks the RHS scale.
    pub(crate) const CROSS_ROW_CG_ABS_TOL: f64 = 1e-12;
    pub(crate) const CROSS_ROW_CG_REL_TOL: f64 = 1e-13;
    // CG converges in at most (dim) iterations; allow a few passes over the
    // dimension to absorb round-off, with a small floor for tiny systems.
    pub(crate) const CROSS_ROW_CG_MIN_ITER_BUDGET: usize = 64;
    pub(crate) const CROSS_ROW_CG_ITER_MULTIPLE: usize = 4;
    let tol = CROSS_ROW_CG_ABS_TOL.max(CROSS_ROW_CG_REL_TOL * b_norm);
    let max_iter = (total_dt + k).max(CROSS_ROW_CG_MIN_ITER_BUDGET) * CROSS_ROW_CG_ITER_MULTIPLE;

    let mut iters = 0usize;
    let mut converged = b_norm == 0.0;
    while iters < max_iter && !converged {
        let (ap_t, ap_beta) =
            arrow_cross_row_matvec(sys, ridge_t, ridge_beta, p_t.view(), p_beta.view());
        let pap = dot2(&p_t, &p_beta, &ap_t, &ap_beta);
        if !(pap.is_finite() && pap > 0.0) {
            return Err(ArrowSchurError::PcgFailed {
                reason: format!(
                    "cross-row full-system CG hit non-positive curvature pᵀAp={pap:e}; \
                     the cross-row penalty Hessian or arrow block is not PD at this iterate"
                ),
            });
        }
        let alpha = rz / pap;
        for i in 0..total_dt {
            x_t[i] += alpha * p_t[i];
            r_t[i] -= alpha * ap_t[i];
        }
        for a in 0..k {
            x_beta[a] += alpha * p_beta[a];
            r_beta[a] -= alpha * ap_beta[a];
        }
        let r_norm = (dot2(&r_t, &r_beta, &r_t, &r_beta)).sqrt();
        iters += 1;
        if r_norm <= tol {
            converged = true;
            break;
        }
        let (nz_t, nz_beta) = precond.apply(r_t.view(), r_beta.view());
        z_t = nz_t;
        z_beta = nz_beta;
        let rz_new = dot2(&r_t, &r_beta, &z_t, &z_beta);
        let beta_cg = rz_new / rz;
        for i in 0..total_dt {
            p_t[i] = z_t[i] + beta_cg * p_t[i];
        }
        for a in 0..k {
            p_beta[a] = z_beta[a] + beta_cg * p_beta[a];
        }
        rz = rz_new;
    }

    if !converged {
        let r_norm = (dot2(&r_t, &r_beta, &r_t, &r_beta)).sqrt();
        return Err(ArrowSchurError::PcgFailed {
            reason: format!(
                "cross-row full-system CG did not converge in {iters} iters \
                 (‖r‖={r_norm:e}, tol={tol:e})"
            ),
        });
    }

    let final_residual = (dot2(&r_t, &r_beta, &r_t, &r_beta)).sqrt();
    let diag = PcgDiagnostics {
        iterations: iters,
        matvec_calls: iters,
        precond_apply_calls: iters + 1,
        ridge_escalations: 0,
        final_relative_residual: if b_norm > 0.0 {
            final_residual / b_norm
        } else {
            0.0
        },
        stopping_reason: PcgStopReason::Converged,
        mixed_precision_status: MixedPrecisionStatus::Off,
        used_device_arrow: false,
    };

    Ok(ArrowNewtonStepArtifacts {
        delta_t: x_t,
        delta_beta: x_beta,
        htt_factors: precond.htt_factors,
        schur_factor: Some(precond.schur_factor),
        pcg_diagnostics: diag,
        gauge_deflated_directions: 0,
    })
}


/// `⟨[a_t; a_β], [b_t; b_β]⟩` over the stacked latent/β vector.
pub(crate) fn dot2(a_t: &Array1<f64>, a_beta: &Array1<f64>, b_t: &Array1<f64>, b_beta: &Array1<f64>) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..a_t.len() {
        acc += a_t[i] * b_t[i];
    }
    for a in 0..a_beta.len() {
        acc += a_beta[a] * b_beta[a];
    }
    acc
}


/// Solve `L Lᵀ x = b` given the lower Cholesky factor `L`.
pub(crate) fn cholesky_solve_lower(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = l.nrows();
    // Precondition: positive, finite factor diagonals (see
    // `cholesky_solve_vector_fixed`). Guard loudly — always, release included —
    // so a future caller supplying an unvalidated factor cannot divide by a
    // zero/non-finite pivot and leak a silent `NaN` into the Schur β-solve
    // (#1038).
    assert!(
        (0..n).all(|i| l[[i, i]].is_finite() && l[[i, i]].abs() >= f64::MIN_POSITIVE),
        "cholesky_solve_lower: factor diagonal must be finite and non-subnormal"
    );
    // Forward solve L y = b.
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }
    // Back solve Lᵀ x = y.
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}


pub(crate) fn reduced_rhs_beta<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
) -> Array1<f64> {
    // Numerical invariant: each per-row `H_tt^(i)` factor must be PD
    // (already enforced by the adaptive-ridge `factor_blocks`).
    let k = sys.k;
    let n = sys.rows.len();
    let mut rhs_beta = Array1::<f64>::zeros(k);
    // The reduced RHS sum `Σ_i H_βt^(i) (H_tt^(i))⁻¹ g_t^(i)` is the same
    // embarrassingly-parallel per-row reduction the `schur_matvec` hot loop
    // already fans out (#1017): each row contributes an independent length-`K`
    // vector. Reuse the identical deterministic chunk-fold so the f64 reduction
    // is bit-identical run-to-run (the topology-candidate ranking gate must not
    // move), and the identical nesting guard (`rayon::current_thread_index()`)
    // so the topology race's outer fan-out is not oversubscribed.
    let parallel = n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
    if parallel {
        use rayon::prelude::*;
        pub(crate) const CHUNK: usize = 64;
        let partials: Vec<Array1<f64>> = (0..n)
            .into_par_iter()
            .chunks(CHUNK)
            .map(|idxs| {
                let mut acc = Array1::<f64>::zeros(k);
                for i in idxs {
                    let row = &sys.rows[i];
                    let v = backend.solve_block_vector(htt_factors.factor(i), row.gt.view());
                    sys_htbeta_accumulate_transpose(sys, i, row, v.view(), &mut acc);
                }
                acc
            })
            .collect();
        for acc in &partials {
            for j in 0..k {
                rhs_beta[j] += acc[j];
            }
        }
    } else {
        for (i, row) in sys.rows.iter().enumerate() {
            let v = backend.solve_block_vector(htt_factors.factor(i), row.gt.view());
            // H_βt^(i) · v accumulates into rhs_beta.  Routes through
            // sys.htbeta_matvec when the dense block is absent.
            sys_htbeta_accumulate_transpose(sys, i, row, v.view(), &mut rhs_beta);
        }
    }
    for j in 0..k {
        rhs_beta[j] -= sys.gb[j];
    }
    rhs_beta
}


/// Which Square-Root / direct factorization the per-row Schur contribution
/// uses. `Direct` forms `H_tβᵀ (H_tt)⁻¹ H_tβ` via a full block solve; `SqrtBa`
/// forms the equivalent `(L⁻¹ H_tβ)ᵀ (L⁻¹ H_tβ)` from the lower triangular
/// solve only. The reduction `Σ_i contribution_i` is identical in both axes.
#[derive(Clone, Copy)]
pub(crate) enum SchurReductionKind {
    Direct,
    SqrtBa,
}


/// Form one row block's `(left, right)` Schur contribution factors so that the
/// contribution is `leftᵀ · right` (`k×k`). `Direct` solves the full block,
/// `SqrtBa` uses only the lower-triangular whitening; both give the same
/// `H_tβᵀ (H_tt)⁻¹ H_tβ` because `H_tt = L Lᵀ`.
#[inline]
pub(crate) fn row_schur_contribution_factors<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
    htt_factor: ArrayView2<'_, f64>,
    backend: &B,
    kind: SchurReductionKind,
) -> Result<(Array2<f64>, Array2<f64>), ArrowSchurError> {
    // Materialize the (d, k) cross-block, probing via the matvec when the
    // dense slab is absent.
    let htbeta = sys_htbeta_materialize_row(sys, row_idx, row)?;
    match kind {
        SchurReductionKind::Direct => {
            let solved = backend.solve_block_matrix(htt_factor, htbeta.view());
            Ok((htbeta, solved))
        }
        SchurReductionKind::SqrtBa => {
            let whitened = backend.sqrt_solve_block_matrix(htt_factor, htbeta.view());
            Ok((whitened.clone(), whitened))
        }
    }
}


/// Subtract one row block's Schur contribution from `schur` using the selected
/// reduction kind. Identical algebra to the inline loop bodies the dense
/// builders used; factored out so the serial and multi-GPU partition paths
/// share one definition.
#[inline]
pub(crate) fn subtract_row_schur_contribution<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
    htt_factor: ArrayView2<'_, f64>,
    backend: &B,
    kind: SchurReductionKind,
    schur: &mut Array2<f64>,
) -> Result<(), ArrowSchurError> {
    let (left, right) =
        row_schur_contribution_factors(sys, row_idx, row, htt_factor, backend, kind)?;
    backend.block_gemm_subtract(schur, &left, &right);
    Ok(())
}
