//! Regression for gam#1110, from the FIT-COEFFICIENT angle (no external
//! reference required, so it runs in every CI environment).
//!
//! A reduced, fully-parametric constant-scale log-logistic AFT
//! (`survival_likelihood = "location-scale"`, `survival_distribution =
//! "logistic"`, `noise_formula = "1"`) collapses its monotone I-spline time warp
//! to the affine log-t baseline. When the time-penalty null space had rank > 1
//! (e.g. `time_num_internal_knots = 2`), the prior code kept that reduced warp as
//! a FREE monotone block carrying a per-row derivative-guard inequality. Its
//! cold-start `β = 0` produced a degenerate warp with a ~1e7 gradient and a
//! constraint active at the boundary; the direct-MLE joint Newton's single global
//! step length was then capped to 0 by that one binding time row, FREEZING every
//! block — including the fully unconstrained location covariate — at its
//! cold-start 0. The whole fit returned all-zero coefficients, so the recovered
//! location slope was EXACTLY 0 (RMSE 0.267 vs the truth; the lifelines reference
//! test `quality_vs_lifelines_loglogistic_aft` checks the same failure from the
//! survivor-surface angle).
//!
//! The fix collapses the parametric AFT to the canonical σ-scaled log-t LOCATION
//! offset for ANY null-space rank (not just rank 1): the time block becomes empty
//! (no free warp, no constraint, no aliased location constant), so the joint
//! Newton is never frozen and the covariate is recovered.
//!
//! This test asserts the recovered location slope is (a) NOT pinned to zero and
//! (b) close to the known data-generating slope, and that the whole coefficient
//! vector is not the cold-start zero — the precise signatures of the bug.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

/// Deterministic SplitMix64 — no RNG crate, no seed drift.
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_open_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }
}

#[test]
fn loglogistic_aft_recovers_location_slope_not_pinned_to_zero() {
    init_parallelism();

    // ---- known log-logistic AFT data-generating process ----
    //   log T = b0 + b_x * x_c + sigma * eps,   eps ~ standard Logistic.
    let n = 320usize;
    let b0 = 2.4_f64;
    let b_x = -0.025_f64; // the location (acceleration) slope we must recover
    let true_sigma = 0.55_f64;

    // Covariate spread comparable to the Haberman ages the reference test uses.
    let x_vals: Vec<f64> = (0..n)
        .map(|i| 30.0 + 53.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    let x_mean = x_vals.iter().sum::<f64>() / n as f64;

    let mut rng = SplitMix64::new(0x5EED_1110_C0DE_A17F);
    let mut time = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    for &x in &x_vals {
        let x_c = x - x_mean;
        let mu = b0 + b_x * x_c;
        let u = rng.next_open_unit();
        let eps = (u / (1.0 - u)).ln();
        let t_event = (mu + true_sigma * eps).exp();
        // Independent log-logistic censoring time → partial censoring.
        let uc = rng.next_open_unit();
        let epsc = (uc / (1.0 - uc)).ln();
        let t_cens = (b0 + 0.9 + true_sigma * epsc).exp();
        let observed = t_event.min(t_cens);
        let d = if t_event <= t_cens { 1.0 } else { 0.0 };
        time.push(observed);
        event.push(d);
    }
    let observed_events: f64 = event.iter().sum();
    assert!(
        observed_events > 0.5 * n as f64 && observed_events < 0.95 * n as f64,
        "synthetic data should be partially censored, got {observed_events} of {n}"
    );

    let headers = ["time", "event", "x"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", time[i]),
                format!("{:.0}", event[i]),
                format!("{:.17e}", x_vals[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode survival data");

    let cfg = FitConfig {
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "logistic".to_string(),
        noise_formula: Some("1".to_string()),
        // The load-bearing setting for gam#1110: 2 internal knots makes the
        // time-penalty null space rank > 1, so the reduced warp took the GENERAL
        // (free-warp + derivative-guard) path rather than the clean rank-1 gauge.
        time_num_internal_knots: 2,
        outer_max_iter: Some(80),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("Surv(time, event) ~ x", &ds, &cfg).expect("gam log-logistic AFT fit");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };

    // The whole coefficient vector must not collapse to the cold-start zero — the
    // direct signature of the joint-Newton freeze (gam#1110).
    let beta_flat = fit.fit.fit.beta_flat();
    assert!(
        beta_flat.iter().any(|&b| b.abs() > 1e-6),
        "fit returned the all-zero cold start (gam#1110 joint-Newton freeze): beta_flat={beta_flat:?}"
    );

    // The reduced parametric AFT collapses the warp onto the σ-scaled log-t
    // location offset, so the free time block is empty: its lifted β is all-zero.
    // (Asserted as the fix's mechanism; the covariate recovery below is the
    // primary, representation-agnostic guard.)
    let beta_time = fit.fit.fit.beta_time();
    assert!(
        beta_time.iter().all(|&b| b == 0.0),
        "reduced parametric AFT must carry the log-t baseline on the location \
         channel (empty free time block); got beta_time={beta_time:?}"
    );

    // PRIMARY: the location covariate slope is recovered, NOT pinned to 0. The
    // threshold (location) design is `[intercept, x]`, so the last coefficient is
    // the `x` slope. The bug pinned it to EXACTLY 0.0.
    let beta_threshold = fit.fit.fit.beta_threshold();
    assert_eq!(
        beta_threshold.len(),
        2,
        "threshold block should be [intercept, x]; got {beta_threshold:?}"
    );
    let x_slope = beta_threshold[1];
    assert!(
        x_slope.abs() > 1e-3,
        "location slope pinned to ~0 — the gam#1110 regression: x_slope={x_slope}"
    );
    // The recovered slope must be close to the true data-generating slope. Right
    // censoring at this n attenuates the MLE slightly (the reference test shows
    // lifelines lands at ~-0.0207 for the true -0.025), so allow a generous band
    // that still excludes both 0 and a wrong sign/scale.
    assert!(
        (x_slope - b_x).abs() < 0.01,
        "location slope does not recover the truth: x_slope={x_slope}, true b_x={b_x}"
    );

    // Sanity: the constant scale recovers the right order of magnitude
    // (log σ ≈ log 0.55 ≈ -0.6), i.e. the scale block is not stuck either.
    let beta_log_sigma = fit.fit.fit.beta_log_sigma();
    assert_eq!(
        beta_log_sigma.len(),
        1,
        "constant-scale block is a single intercept"
    );
    let sigma_hat = beta_log_sigma[0].exp();
    assert!(
        sigma_hat > 0.3 && sigma_hat < 0.9,
        "recovered scale σ̂={sigma_hat} is implausible for true σ={true_sigma}"
    );
}
