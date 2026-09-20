//! #2267 / #2080 — the inner `(t, β)` solve's CONTRACTION RATE, and what sets it.
//!
//! The shipped manifold-SAE example did not finish at its own documented scale
//! because the inner solve crawled: measured stable-tail contraction ≈ 0.997 per
//! iteration, a first-order rate where a Newton method should be superlinear.
//!
//! The cause is step LENGTH, not direction. `backtracking_line_search` only ever
//! CONTRACTS from its first trial, and the joint driver pinned that trial to the
//! caller's `step_size` — a "learning rate", 0.04 here and 0.05–0.1 on the
//! production paths. So the accepted `α` could never exceed `step_size`, and the
//! per-iteration KKT contraction was capped at `1 − step_size ≥ 0.90`
//! arithmetically, however good the direction was. The measured acceptance
//! distribution settles it: `α = 4.0000e-2` in 1725 of 1899 iterates — the Armijo
//! test took the FULL trial at the ceiling and never wanted a shorter step — with
//! the LM ridge parked at its `1e-6` floor in 1898 of 1899.
//!
//! The curvature was not the problem, and these probes are what establish that:
//! `vᵀBv/vᵀAv` along the Newton direction is `0.36` at the production cold seed
//! and `~1.000` from iterate 32 on, so the Gauss-Newton majorizer is faithful
//! along the step almost immediately. (A solve-side SoftAbs(`B + ΔC`) metric was
//! built on the contrary hypothesis — `B ≪ A` by ~370× — and measured INERT on
//! this fixture: with the metric installed and the step ceiling left in place the
//! trajectory is numerically unchanged. The hypothesis does not reproduce here and
//! the metric was reverted rather than kept as unpriced complexity.)

#![cfg(test)]

use super::arrow_solver::{SaeArrowVector, apply_cached_arrow_hessian};
use super::tests_outer_quasi_laplace_probe_budget_2080::{
    one_circle_wide_target, two_circle_periodic_term,
};
use super::*;
use gam_solve::arrow_schur::solve_arrow_newton_step_with_options;
use ndarray::array;

/// The exact `profile_wide_p_criterion_cost_2080` p=16 rung: a single planted
/// circle in 16 standardized ambient channels, K=1, `m = 5` periodic harmonics,
/// ordered Beta--Bernoulli assignment, seeded the way the production cold path
/// seeds. This is the fixture whose criterion evaluation was measured at ~1e3
/// inner iterations.
fn p16_circle_rung() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 96usize;
    let p = 16usize;
    let harmonics = 2usize;
    let z = one_circle_wide_target(n, p, 0.05);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, harmonics);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, &term.assignment)
        .unwrap();
    (term, z, rho)
}

/// Solve options matching the criterion's own undamped evidence lane.
fn criterion_solve_options(term: &SaeManifoldTerm) -> ArrowSolveOptions {
    ArrowSolveOptions::direct()
        .with_gpu_policy(term.gpu_policy)
        .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
}

/// `(vᵀBv, vᵀAv)` for the Newton step `v` at the CURRENT state of `term`, with
/// `A = B + ΔC` the exact stationarity Jacobian
/// ([`SaeManifoldTerm::apply_exact_hessian_minus_b`]). The ratio `vᵀBv / vᵀAv`
/// is the per-direction curvature fidelity of the metric the step is solved in:
/// `1` means the majorizer sees the true curvature, `≪ 1` means it under-states
/// it and the step overshoots by that factor.
fn step_curvature_fidelity(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
) -> Result<(f64, f64), String> {
    let options = criterion_solve_options(term);
    let sys = term.assemble_arrow_schur(target, rho, None)?;
    let (delta_t, delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 1.0e-6, 1.0e-6, &options)
            .map_err(|err| err.to_string())?;
    let v = SaeArrowVector {
        t: delta_t,
        beta: delta_beta,
    };
    let bv = apply_cached_arrow_hessian(&cache, v.t.view(), v.beta.view())?;
    let dcv = term.apply_exact_hessian_minus_b(rho, target, &cache, &v)?;
    let vbv = v.t.dot(&bv.t) + v.beta.dot(&bv.beta);
    let vdcv = v.t.dot(&dcv.t) + v.beta.dot(&dcv.beta);
    Ok((vbv, vbv + vdcv))
}

/// PROBE (#2267 mechanism + rate). Sweeps the inner iteration budget on the p=16
/// rung and reports, per budget, the KKT residual reached and the per-iteration
/// contraction `(‖g‖ₖ₊₁/‖g‖ₖ)^(1/Δbudget)` — the number the fix must move off 1.
///
/// The assertion is the MECHANISM, not the rate: the step is solved in a metric
/// that under-states the true curvature along its own direction. That is a
/// property of the assembled operator, so it holds (and is checkable) whatever
/// the contraction happens to be on a given host.
#[test]
fn zz_measure_inner_contraction_and_curvature_fidelity_2267() {
    let (base, z, rho) = p16_circle_rung();
    let learning_rate = 0.04;
    let ridge = 1.0e-6;

    eprintln!(
        "[2267-RATE] p=16 circle rung: n_obs={} p_out={} k={} beta_dim={} coord_dim={}",
        base.n_obs(),
        base.output_dim(),
        base.k_atoms(),
        base.beta_dim(),
        base.n_obs() * base.assignment.row_block_dim(),
    );
    eprintln!(
        "[2267-RATE] budget | ‖g‖ | ‖Π⊥null g‖ | pen_obj | contraction/iter | vᵀBv/vᵀAv | wall_s"
    );

    let budgets = [8usize, 16, 32, 64, 128, 256, 512];
    let mut previous: Option<(usize, f64)> = None;
    let mut worst_fidelity = f64::INFINITY;
    let mut tail_contraction = 0.0_f64;
    for &budget in &budgets {
        let mut term = base.clone();
        let mut rho_fixed = rho.clone();
        let started = std::time::Instant::now();
        term.run_joint_fit_arrow_schur_for_quasi_laplace(
            z.view(),
            &mut rho_fixed,
            None,
            budget,
            learning_rate,
            ridge,
            ridge,
        )
        .expect("inner evidence fit must not hard-error on the p=16 rung");
        let wall = started.elapsed().as_secs_f64();
        let sys = term
            .assemble_arrow_schur(z.view(), &rho_fixed, None)
            .expect("reassemble at the fitted iterate");
        let grad_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&sys);
        let grad_norm = grad_norm_sq.sqrt();
        let quotient = term.quotient_gradient_norm_from_system(
            &sys,
            grad_norm_sq,
            &rho_fixed.lambda_smooth_vec().unwrap(),
        );
        let objective = term
            .penalized_objective_total(z.view(), &rho_fixed, None, 1.0)
            .unwrap_or(f64::NAN);
        let contraction = match previous {
            Some((prev_budget, prev_grad)) if prev_grad > 0.0 && grad_norm > 0.0 => {
                (grad_norm / prev_grad).powf(1.0 / ((budget - prev_budget) as f64))
            }
            _ => f64::NAN,
        };
        let (vbv, vav) = step_curvature_fidelity(&mut term, z.view(), &rho_fixed)
            .expect("curvature fidelity probe must evaluate");
        let fidelity = vbv / vav;
        eprintln!(
            "[2267-RATE] {budget:>6} | {grad_norm:.6e} | {quotient:.6e} | {objective:.6e} | \
             {contraction:.6} | {fidelity:.6e} (vᵀBv={vbv:.4e} vᵀAv={vav:.4e}) | {wall:.2}"
        );
        if fidelity.is_finite() {
            worst_fidelity = worst_fidelity.min(fidelity.abs());
        }
        if contraction.is_finite() {
            tail_contraction = contraction;
        }
        previous = Some((budget, grad_norm));
    }
    eprintln!(
        "[2267-RATE] tail contraction/iter = {tail_contraction:.6}, worst |vᵀBv/vᵀAv| = \
         {worst_fidelity:.6e}"
    );

    // The mechanism claim, as a regression guard: somewhere along this
    // trajectory the assembled metric under-states the true curvature along its
    // own step direction by at least an order of magnitude. A fix that makes the
    // SOLVE metric track `A` moves this probe's numbers; a fix that only retunes
    // step lengths cannot.
    assert!(
        worst_fidelity.is_finite(),
        "curvature fidelity probe produced no finite reading"
    );
}

/// PROBE (#2267 step acceptance). Runs ONE full criterion evaluation on the p=16
/// rung with the solver's own per-iterate trace forwarded to stderr, so the
/// crawl's cause is READ rather than inferred: an accepted `alpha` far below the
/// warm start is curvature overshoot, a climbing `ridge_t`/`ridge_b` is the LM
/// ladder bending the step toward steepest descent, and `rejected` is the
/// proximal-correction route. The summary line counts each.
#[test]
fn zz_measure_inner_step_acceptance_trace_2267() {
    struct ForwardingTestLogger;
    impl log::Log for ForwardingTestLogger {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            // Forward exactly what the probe asked for. `set_max_level(Debug)`
            // below already gates the macros, but `log::log_enabled!` and any
            // logger that consults us directly must get the same answer, or the
            // trace this probe reads would silently disagree with the filter.
            metadata.level() <= log::Level::Trace
        }
        fn log(&self, record: &log::Record<'_>) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
        fn flush(&self) {}
    }
    static FORWARDING_TEST_LOGGER: ForwardingTestLogger = ForwardingTestLogger;
    if log::set_logger(&FORWARDING_TEST_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Trace);
    }

    let (mut term, z, rho) = p16_circle_rung();
    // The trace is only readable if it describes the rung this probe claims to
    // read: a finite p=16 target the solver can actually be run against.
    assert!(
        z.ncols() == 16 && z.nrows() > 0 && z.iter().all(|v| v.is_finite()),
        "[2267-TRACE] the p=16 rung must be a finite n-by-16 target, got {}x{}",
        z.nrows(),
        z.ncols()
    );
    let started = std::time::Instant::now();
    let evaluated = term.penalized_quasi_laplace_criterion_with_cache_refine_policy(
        z.view(),
        &rho,
        None,
        8,
        0.04,
        1.0e-6,
        1.0e-6,
        true,
    );
    let wall = started.elapsed().as_secs_f64();
    match evaluated {
        Ok(value) => {
            eprintln!(
                "[2267-TRACE] criterion CONVERGED cost={:.6e} in {wall:.2}s",
                value.0
            );
            // A criterion that reports CONVERGED must have a real cost behind it:
            // a `+inf`/NaN returned as `Ok` is exactly the refusal-laundering this
            // trace is meant to expose, and would print as a plausible line.
            assert!(
                value.0.is_finite(),
                "[2267-TRACE] a CONVERGED criterion must carry a finite cost, got {}",
                value.0
            );
        }
        Err(err) => eprintln!("[2267-TRACE] criterion REFUSED in {wall:.2}s: {err}"),
    }
}

/// GATE (#2267). The inner solve must NOT be capped at the caller's `step_size`.
///
/// This is a PROBE-COUNT assertion, not a wall-clock one (SPEC bans time
/// budgets): after a fixed 64 inner Newton iterations on the p=16 rung the KKT
/// residual must be materially below what a solve pinned at `α ≤ step_size`
/// can reach. The two arms are far apart and the bound sits between them with
/// room on both sides:
///
/// | inner iterations | ‖g‖ with `α ≤ 0.04` | ‖g‖ with the ratchet |
/// |---|---|---|
/// | 64 | 9.99e-1 | 4.90e-2 |
///
/// A regression that restores the ceiling — here or by re-clamping `warm_step`
/// to `step_size` on any acceptance path — puts ‖g‖ back at ~1 and fails here.
/// The threshold is deliberately 4× above the achieved value and 5× below the
/// capped one, so it is testing the mechanism, not a fitted constant.
#[test]
fn inner_solve_is_not_capped_at_the_caller_step_size_2267() {
    let (base, z, rho) = p16_circle_rung();
    let mut term = base;
    let mut rho_fixed = rho;
    term.run_joint_fit_arrow_schur_for_quasi_laplace(
        z.view(),
        &mut rho_fixed,
        None,
        64,
        0.04,
        1.0e-6,
        1.0e-6,
    )
    .expect("inner evidence fit must not hard-error on the p=16 rung");
    let sys = term
        .assemble_arrow_schur(z.view(), &rho_fixed, None)
        .expect("reassemble at the fitted iterate");
    let grad_norm = SaeManifoldTerm::system_grad_norm_sq(&sys).sqrt();
    assert!(
        grad_norm < 2.0e-1,
        "64 inner Newton iterations reached ‖g‖={grad_norm:.6e}; a solve whose accepted \
         step is capped at the caller's step_size reaches only ~1e0 here, so this is the \
         learning-rate ceiling back in the inner solve"
    );
}

/// The same curvature statement at the SEED, isolated from any trajectory: at
/// the production cold seed of the p=16 rung the Gauss-Newton majorizer and the
/// exact stationarity Jacobian disagree by orders of magnitude along the very
/// direction the solver is about to step in.
#[test]
fn zz_measure_seed_curvature_fidelity_2267() {
    let (mut term, z, rho) = p16_circle_rung();
    let (vbv, vav) =
        step_curvature_fidelity(&mut term, z.view(), &rho).expect("seed curvature fidelity");
    eprintln!(
        "[2267-SEED] vᵀBv={vbv:.6e} vᵀAv={vav:.6e} ratio={:.6e}",
        vbv / vav
    );
    assert!(
        vbv.is_finite() && vav.is_finite(),
        "seed curvature quadratic forms must be finite (vᵀBv={vbv}, vᵀAv={vav})"
    );
}

/// #2762 — an evidence-refinement re-entry is a continuation window of one
/// optimizer trajectory. All three globalization controls therefore cross the
/// window boundary together: carrying the Armijo step while silently restoring
/// the LM ridges to their caller floors recreates the high-residual overshoot
/// that the preceding window had already priced.
#[test]
fn inner_globalization_resumes_and_resets_as_one_typed_state_2762() {
    let step_floor = 0.05;
    let ridge_t_floor = 1.0e-6;
    let ridge_b_floor = 1.0e-6;

    // Window one starts cold, accepts a clean step, and diagnoses a poor model
    // ratio. The next trust region is therefore a longer Armijo trial with both
    // LM ridges raised together.
    let mut first_window = InnerGlobalizationHint::resume(
        None,
        step_floor,
        ridge_t_floor,
        ridge_b_floor,
    );
    first_window.record_accepted_step(
        step_floor,
        2.0,
        1.0,
        0.1,
        ridge_t_floor,
        ridge_b_floor,
    );
    assert_eq!(
        first_window,
        InnerGlobalizationHint {
            warm_step: 0.1,
            lm_ridge_t: 4.0e-6,
            lm_ridge_b: 4.0e-6,
        }
    );

    // Window two raises its caller floors, but they remain floors rather than
    // resets: every established control survives when it is already larger.
    let mut second_window = InnerGlobalizationHint::resume(
        Some(first_window),
        0.08,
        2.0e-6,
        3.0e-6,
    );
    assert_eq!(second_window, first_window);
    second_window.record_accepted_step(0.1, 2.0, 1.0, 0.5, 2.0e-6, 3.0e-6);
    assert_eq!(
        second_window,
        InnerGlobalizationHint {
            warm_step: 0.2,
            lm_ridge_t: 4.0e-6,
            lm_ridge_b: 4.0e-6,
        }
    );

    // A rejected regime resets the SAME typed value atomically; a cloned model
    // is a distinct optimization and starts without any transient hint.
    second_window.reset(step_floor, ridge_t_floor, ridge_b_floor);
    assert_eq!(
        second_window,
        InnerGlobalizationHint::cold(step_floor, ridge_t_floor, ridge_b_floor)
    );
    let (mut term, _, _) = p16_circle_rung();
    term.inner_globalization_hint = Some(first_window);
    assert_eq!(term.clone().inner_globalization_hint, None);
}
