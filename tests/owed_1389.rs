//! Owed-work regression gate for issue #1389 — survival location-scale fit
//! hangs with steady RSS growth on constant-scale (inert-qdot) geometry.
//!
//! ## The defect (now fixed)
//!
//! The survival location-scale joint-Hessian directional-derivative path runs a
//! velocity (event-Jacobian `g`) pullback pass. That pass contributes EXACTLY
//! zero unless some weighted row carries live qdot-derivative mass
//! (`d1_qdot1` / `d2_qdot1` / `d_h_d`) — i.e. unless the scale channel is
//! genuinely time-varying. A constant-scale lognormal AFT has no such mass on
//! any row, so the term is identically zero.
//!
//! Before the fix (`family_solver.rs`, the `any_live_qdot` guard added in
//! 2df17f736) the per-row fold still paid a fresh `(p_total, p_total)`
//! allocation per Rayon task and per reduction even when EVERY row was inert.
//! The outer REML optimizer probes the joint Hessian directional derivative
//! repeatedly with no inner-PIRLS / `ensure_theta` stage line, so across many
//! such silent evaluations this `p² × threads` transient is a steady
//! contributor to the monotonic RSS growth and 100%-CPU "hang" reported on the
//! ICU location-scale benchmarks (`icu_survival_los`, `icu_survival_death`).
//!
//! ## What this test pins
//!
//! It drives the PUBLIC fit path (`fit_from_formula` →
//! `FitResult::SurvivalLocationScale`) on exactly the constant-scale lognormal
//! AFT geometry that triggers the hang: a single covariate-independent log-scale
//! channel (no `noise_formula`), so every row is qdot-inert and the velocity
//! pass is skipped on every one of the outer optimizer's repeated joint-Hessian
//! directional-derivative evaluations. The fit must:
//!
//!   1. RETURN (no unbounded hang) — guarded by a generous wall-clock bound that
//!      only a true non-terminating/quadratically-thrashing run can blow; and
//!   2. produce finite, converged location and log-scale coefficients.
//!
//! The exactness of the skip itself (that it drops only a provably-zero term)
//! is pinned in-crate by
//! `families::survival::location_scale::tests::joint_dh_velocity_skip_is_exact_on_all_censored_rows`,
//! which FD-checks the directional derivative on an all-censored fixture. This
//! file is the end-to-end complement: the real outer-REML loop on the hang
//! geometry terminates with a sane fit.

use std::time::Instant;

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

/// A tiny deterministic LCG. This is a hang/finiteness regression, not a
/// truth-recovery test, so we only need a reproducible, fixed data stream — not
/// a particular reference RNG.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        // Numerical Recipes LCG constants.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Standard normal via Box–Muller (one draw; the cosine branch is enough).
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[test]
fn survival_location_scale_constant_scale_fit_terminates_1389() {
    init_parallelism();

    // ---- constant-scale lognormal AFT data (the inert-qdot hang geometry) ----
    // log T = eta_location + sigma * eps, with a SINGLE constant sigma (no
    // time-varying scale → every row is qdot-inert → the velocity pass is
    // skipped on every outer-REML joint-Hessian directional-derivative eval).
    //
    // Sizing (n, the spline `k`, and the wall bound below) is chosen so the
    // healthy fit finishes comfortably inside CI's UNOPTIMIZED-DEBUG budget on a
    // shared runner, while a regressed outer loop is caught structurally — by
    // `outer_converged == false` — rather than by a wall-clock SIGKILL. See the
    // `outer_max_iter` / wall-bound comments below for why this matters: a
    // per-iteration cost small enough that even the worst case (the optimizer
    // crawling all the way to `outer_max_iter`) returns in seconds turns a
    // recurrence into a fast, NAMED assertion failure instead of a 600s
    // bulk-kill that hides which test hung and blocks the whole CI shard.
    let n = 120usize;
    let mut rng = Lcg::new(0x1389_2026);
    let sigma_true = 0.7_f64;

    let headers: Vec<String> = ["t", "event", "x", "z"]
        .into_iter()
        .map(str::to_string)
        .collect();

    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    let mut n_censored = 0usize;
    for _ in 0..n {
        let x = 2.0 * rng.unit() - 1.0;
        let z = 2.0 * rng.unit() - 1.0;
        let eps = rng.normal();
        let s_z = (std::f64::consts::PI * z).sin();
        let eta_loc = -0.5 + 0.8 * x + s_z;
        let t_event = (eta_loc + sigma_true * eps).exp();
        // Independent exponential censoring on the same scale as the event time
        // (mean ~0.9) → roughly 30–45% right-censoring, exercising both the
        // event and censored row kernels.
        let c = -rng.unit().max(1e-12).ln() * 0.9;
        let (t, event) = if t_event <= c {
            (t_event, 1.0)
        } else {
            n_censored += 1;
            (c.max(1e-6), 0.0)
        };
        rows.push(StringRecord::from(vec![
            format!("{t:.17e}"),
            format!("{event:.17e}"),
            format!("{x:.17e}"),
            format!("{z:.17e}"),
        ]));
    }
    let cens_frac = n_censored as f64 / n as f64;
    assert!(
        (0.15..=0.60).contains(&cens_frac),
        "fixture should have a real censored/event mix, got cens_frac={cens_frac:.3}"
    );

    let ds = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode constant-scale survival-LS data");

    // ---- fit: lognormal location-scale AFT with a constant log-scale ---------
    // No `noise_formula` ⇒ a single covariate-independent log-sigma channel ⇒
    // the qdot-inert geometry. `time_num_internal_knots: 2` keeps the
    // gauge-degenerate cold-start time block small (the estimand is a parametric
    // AFT), matching the lognormal-LS quality fixture.
    //
    // `outer_max_iter` is left deliberately generous (80): the point of this
    // gate is to give a regressed outer optimizer room to crawl so the crawl is
    // OBSERVABLE. A healthy fit on this well-posed geometry certifies outer
    // stationarity in a handful of iterations and never approaches the cap; a
    // regressed one that cannot certify on the flat constant-scale time ridge
    // exhausts the cap and returns `outer_converged == false`. With the small
    // per-iteration cost (n=120, k=4) even that worst case is seconds, so the
    // recurrence shows up as the fast, named convergence assertion below rather
    // than a wall-clock kill.
    let cfg = FitConfig {
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        time_num_internal_knots: 2,
        outer_max_iter: Some(80),
        ..FitConfig::default()
    };

    // The defect manifested as a non-terminating / quadratically-thrashing run
    // (steady RSS growth, 100% CPU, no stage lines) whose only end state was the
    // outer iteration cap. The PRIMARY teeth of this gate is the
    // `outer_converged` assertion further down — a regression cannot certify and
    // trips it. This wall bound is a secondary backstop for a genuine
    // never-returns hang (e.g. an unbounded inner solve), set well under nextest's
    // 300s SLOW notice / 600s terminate so that, if it ever fires, it fires as
    // THIS test's own named panic rather than as an anonymous shard bulk-kill.
    let started = Instant::now();
    let result = fit_from_formula(r#"Surv(t, event) ~ x + s(z, bs="tp", k=4)"#, &ds, &cfg)
        .expect("constant-scale survival location-scale AFT fit");
    let elapsed = started.elapsed();
    assert!(
        elapsed.as_secs() < 240,
        "constant-scale survival-LS fit took {elapsed:?}; the #1389 inert-qdot \
         velocity-pass hang appears to have regressed"
    );

    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;

    // A terminating fit on this well-posed constant-scale geometry must
    // converge with finite coefficients in both channels.
    assert!(
        unified.outer_converged,
        "constant-scale survival-LS outer optimizer did not converge: \
         iters={} grad_norm={:?}",
        unified.outer_iterations, unified.outer_gradient_norm
    );
    let beta_location = unified.beta_threshold();
    let beta_log_sigma = unified.beta_log_sigma();
    assert!(
        beta_location.iter().all(|v| v.is_finite()),
        "non-finite location coefficients: {beta_location:?}"
    );
    assert!(
        beta_log_sigma.iter().all(|v| v.is_finite()),
        "non-finite log-sigma coefficients: {beta_log_sigma:?}"
    );
}
