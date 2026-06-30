//! Diagnostic + regression harness for #1690 — Poisson (~28x) / Gamma (~7x)
//! 1-D `s(x)` REML fits much slower than mgcv at equal accuracy.
//!
//! This file reproduces the issue's exact data-generating process at the Rust
//! `fit_from_formula` level (n=600, single P-spline smooth, log link, REML +
//! double penalty) and records the outer-loop work metrics (`outer_cost_evals`,
//! `inner_pirls_solves`) for both families. The Poisson/Gamma path is
//! structurally identical (same closed-form log-link inner kernel, same
//! `DispersionHandling::Fixed` objective, no Firth), so any per-family
//! wall-clock gap is an outer-iteration-count / surface-conditioning artifact,
//! not extra coded work.
//!
//! The asserted contract is correctness + a coarse outer-work upper bound that
//! trips if the count-family outer loop blows up. It does NOT depend on
//! R / mgcv.

use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use csv::StringRecord;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

/// Mirror the issue's repro: `mu = exp(1 + sin(5x))`, x ~ U(0,1), n=600.
/// Poisson draws counts; Gamma draws shape=4 positive-continuous with the same
/// mean surface.
fn build_data(family: &str, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let n = 600usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform 0..1");
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi: f64 = ux.sample(&mut rng);
        let mu = (1.0 + (5.0 * xi).sin()).exp();
        let yi = if family == "poisson" {
            let p = Poisson::new(mu).expect("poisson mean > 0");
            p.sample(&mut rng)
        } else {
            // Gamma(shape=4, scale=mu/4) => mean = mu.
            let g = Gamma::new(4.0, mu / 4.0).expect("gamma(shape,scale)");
            g.sample(&mut rng)
        };
        x.push(xi);
        y.push(yi);
    }
    (x, y)
}

#[test]
fn poisson_gamma_single_smooth_outer_work_1690() {
    gam::init_parallelism();
    gam::progress_log::init_logging();

    for family in ["poisson", "gamma"] {
        let (x, y) = build_data(family, 0);
        let n = x.len();
        let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
        let rows: Vec<StringRecord> = (0..n)
            .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
            .collect();
        let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");

        let cfg = FitConfig {
            family: Some(family.to_string()),
            ..FitConfig::default()
        };
        // k=12 P-spline, matching the mgcv arm `s(x, bs="ps", k=12)`.
        let result = fit_from_formula("y ~ s(x, k=12)", &ds, &cfg)
            .expect("single-smooth REML fit must succeed");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit for {family}");
        };
        let fit = fit.fit;

        assert!(fit.reml_score.is_finite(), "{family}: reml_score finite");
        assert!(
            fit.beta.iter().all(|b| b.is_finite()),
            "{family}: beta finite"
        );
        let edf = fit
            .edf_total()
            .unwrap_or_else(|| panic!("{family}: inference EDF present"));
        assert!(
            edf.is_finite() && edf > 0.0,
            "{family}: edf finite positive (got {edf})"
        );

        eprintln!(
            "RECORD_1690 family={family} reml_score={:.10} edf={:.6} \
             outer_cost_evals={} inner_pirls_solves={} grad_norm={:?} converged={}",
            fit.reml_score,
            edf,
            fit.outer_cost_evals,
            fit.inner_pirls_solves,
            fit.outer_gradient_norm,
            fit.outer_converged,
        );

        assert!(
            fit.outer_converged,
            "{family}: outer REML optimizer must certify convergence"
        );

        // Coarse outer-work upper bound. A plain 1-parameter smooth must not
        // explode the outer loop. The seed-grid prepass for k=1 evaluates ~10-20
        // grid points; with screening + multistart + ARC Newton the genuine
        // full-n inner solves should stay well under 120. This trips if the
        // count-family outer loop regresses toward the binomial-class blow-up.
        assert!(
            fit.inner_pirls_solves > 0 && fit.inner_pirls_solves < 400,
            "{family}: inner_pirls_solves={} outside (0,400) — outer-work regression",
            fit.inner_pirls_solves
        );
    }
}
