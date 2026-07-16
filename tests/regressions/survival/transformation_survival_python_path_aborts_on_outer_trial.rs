//! Regression for gam#1123: the library survival-transformation fit (the path the
//! Python `gamfit.fit(df, "Surv(entry,exit,event) ~ x",
//! survival_likelihood="transformation")` wrapper drives) must NOT abort with an
//! IntegrationError on ordinary right-censored Weibull proportional-hazards data.
//!
//! ## The bug
//! The `gam` CLI fits this exact dataset/formula/mode to convergence in ~18 inner
//! PIRLS iterations and recovers the truth (β_x ≈ 0.8). The CLI fits at the SEED
//! smoothing parameter and stops — it has no outer λ-selection loop. The library
//! path (used by pyffi) additionally runs an outer BFGS over ρ = log λ to
//! REML/LAML-select the I-spline time penalty (issue #563). The inner PIRLS at the
//! seed converges IDENTICALLY for both paths; the two DIVERGE only in that outer
//! loop. On ~40% of seeds the outer BFGS marched into a pathological large-λ
//! region where the constrained I-spline PIRLS can no longer converge
//! (`MaxIterationsReached`, gradient plateaued), and that inner non-convergence at
//! an outer TRIAL ρ was escalated to a FATAL IntegrationError — discarding the
//! perfectly-good converged seed-λ fit that demonstrably exists (the CLI saves it).
//!
//! ## The contract
//! The CLI and the library "share one engine" (per the README). An inner
//! non-convergence (or any non-finite LAML) at an outer TRIAL ρ is not a valid
//! evaluation of the objective there — the envelope theorem that makes ∂LAML/∂ρ
//! exact needs the inner β-optimum — so it must be a high FINITE cost the outer
//! optimizer steps away from, NEVER a fatal error. The outer selector may only
//! ever IMPROVE on the seed; in the worst case it falls back to the seed λ, which
//! is exactly what the CLI fits. So the library fit must succeed and recover the
//! covariate effect.
//!
//! This test drives the real public `fit_from_formula` + `FitConfig` API (the same
//! entry the Python wrapper uses) on seed 0 — the seed the issue reports as a
//! hard failure — and asserts (a) no abort and (b) covariate recovery.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

/// Deterministic, dependency-free PRNG (SplitMix64) for bit-reproducible data.
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
    fn next_uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[test]
fn transformation_survival_library_path_does_not_abort_on_outer_trial() {
    init_parallelism();

    // ---- ordinary right-censored Weibull PH data, single covariate x --------
    // x ~ N(0, 1); Weibull(shape = exp(0.4), scale = exp(-β_x * x / shape))
    // gives a proportional-hazards covariate effect of β_x = 0.8 on the
    // log-cumulative-hazard scale. ~35% censoring via an independent uniform
    // censoring time, the textbook generative convention the issue describes.
    // Seed 0 (0x0) is the seed the issue reports as a hard failure.
    let n = 300usize;
    let beta_x_true = 0.8_f64;
    let shape_true = 0.4_f64.exp();
    let mut rng = SplitMix64::new(0x0000_0000_0000_0000);

    let mut entry = Vec::with_capacity(n);
    let mut exit = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = rng.next_normal();
        // PH: log Λ(t|x) = log Λ_0(t) + β_x·x  ⇒  scale_i = exp(-β_x·x/shape).
        let scale_i = (-(beta_x_true * xi) / shape_true).exp();
        let u = rng.next_uniform();
        let event_time = scale_i * (-u.ln()).powf(1.0 / shape_true);
        // Independent right-censoring time (~35% censoring on this design).
        let cens = 1.4_f64 * (-rng.next_uniform().ln());
        let (obs, ev) = if event_time <= cens {
            (event_time, 1.0)
        } else {
            (cens, 0.0)
        };
        entry.push(0.0_f64);
        exit.push(obs.max(1e-6));
        event.push(ev);
        x.push(xi);
    }
    let n_events = event.iter().filter(|&&e| e > 0.5).count();
    assert!(
        n_events > n / 4 && n_events < n,
        "sanity: design should be ordinary right-censored (events={n_events}/{n})"
    );

    let headers = vec![
        "entry".to_string(),
        "exit".to_string(),
        "event".to_string(),
        "x".to_string(),
    ];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                entry[i].to_string(),
                exit[i].to_string(),
                event[i].to_string(),
                x[i].to_string(),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode survival data");

    // The general transformation (Royston-Parmar I-spline) baseline — the mode
    // that builds REML-selected time-smoothing blocks and therefore runs the
    // outer λ-selection loop the bug lives in.
    let cfg = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        ..FitConfig::default()
    };

    // The crux of #1123: this MUST NOT return Err (the IntegrationError abort).
    let result =
        fit_from_formula("Surv(entry, exit, event) ~ x", &data, &cfg).unwrap_or_else(|err| {
            panic!(
                "gam#1123: transformation survival fit aborted on ordinary right-censored data \
                 (seed 0, ~ x) — the outer λ-selector escalated an inner trial non-convergence to \
                 a fatal error instead of stepping away / falling back to the convergent seed λ \
                 that the CLI fits. err = {err}"
            )
        });
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!(
            "expected a SurvivalTransformation fit result for survival_likelihood=transformation"
        );
    };

    // The fit must be finite and the covariate effect must be ordered/recovered:
    // β_x ≈ 0.8 (the CLI recovers 0.847 on the issue's data; a transformation
    // I-spline baseline on n=300 lands comfortably within a generous band).
    let beta = &fit.fit.beta;
    assert!(
        beta.iter().all(|v| v.is_finite()),
        "all fitted coefficients must be finite; got {beta:?}"
    );
    let p_time = fit.time_base_ncols;
    assert!(
        p_time < beta.len(),
        "the RP time block must be a strict prefix of beta: p_time={p_time}, p={}",
        beta.len()
    );
    // beta = [time block | covariate block]; the covariate block for `~ x` is
    // [intercept, β_x], so β_x is the LAST coefficient.
    let beta_x_hat = beta[beta.len() - 1];
    assert!(
        beta_x_hat > 0.4 && beta_x_hat < 1.2,
        "gam#1123: transformation survival must recover the covariate effect \
         β_x ≈ {beta_x_true} (the CLI gets 0.847); got β_x_hat = {beta_x_hat:.4} \
         (full beta = {beta:?})"
    );
}
