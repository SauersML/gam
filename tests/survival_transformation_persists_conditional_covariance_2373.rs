//! Regression (#2373 defect C): a converged single-cause survival
//! transformation / weibull fit MUST persist the conditional Bayesian
//! covariance `Vb = H⁻¹`, so `predict()` can serve every covariance-requiring
//! mode instead of refusing with "fit result does not contain conditional
//! covariance".
//!
//! Root cause the fix closes: the single-cause assembler
//! `survival_unified_fit_result` (crates/gam-models/src/fit_orchestration/fit.rs)
//! hard-coded `covariance_conditional: None` and `inference.beta_covariance:
//! None`, even though it already held the converged observed-information
//! penalized Hessian `H = X'W_H X + S(λ)` — the very matrix its EDF trace-solves
//! successfully factor. Every converged transformation/weibull fit therefore had
//! predict broken for the `Conditional` covariance mode. The fix inverts that
//! Hessian (identity coefficient gauge → no lift) and persists `Vb`, mirroring
//! the location-scale reduced-parametric path
//! (crates/gam-models/src/survival/location_scale/fit.rs).
//!
//! DGP: deterministic right-censored Weibull (shape 1.5, log-hazard slope 0.7),
//! the well-posed #898 repro shape, so the RP fit converges cleanly (shared with
//! `bug_hunt_survival_outer_converged_honest.rs`).

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

const N: usize = 300;

/// Deterministic LCG uniform stream (no RNG crate dependency).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u01(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_u01().max(1e-12);
        let u2 = self.next_u01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn build_survival_frame() -> gam::data::EncodedDataset {
    const SHAPE: f64 = 1.5;
    const ETA0: f64 = -1.5;
    const SLOPE: f64 = 0.7;
    let mut rng = Lcg::new(0x1234_5678_9ABC_DEF0);

    let headers = ["t", "event", "x"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut rows = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = rng.next_normal();
        let eta = ETA0 + SLOPE * xi;
        let u = rng.next_u01().clamp(1e-9, 1.0 - 1e-12);
        let t_lat = (-eta / SHAPE).exp() * (-u.ln()).powf(1.0 / SHAPE);
        let cens = (-rng.next_u01().max(1e-12).ln() * 12.0).min(20.0);
        let exit = t_lat.min(cens);
        let event = if t_lat <= cens { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![
            format!("{exit:.17e}"),
            format!("{event:.1}"),
            format!("{xi:.17e}"),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode survival train frame")
}

/// Assert the minted transformation-survival fit exposes a well-formed
/// conditional covariance both on `covariance_conditional` (what the persisted
/// payload carries to disk) and through `beta_covariance()` (the exact accessor
/// `select_survival_prediction_covariance` consults at predict time — see
/// survival/predict.rs).
fn assert_fit_serves_conditional_covariance(
    fit: &gam::solver::estimate::UnifiedFitResult,
    label: &str,
) {
    let p = fit.beta.len();

    // (1) Persisted field is present with the right shape.
    let cov = fit.covariance_conditional.as_ref().unwrap_or_else(|| {
        panic!(
            "{label}: converged survival fit must persist covariance_conditional so predict \
             can serve uncertainty (#2373 defect C)"
        )
    });
    assert_eq!(
        cov.dim(),
        (p, p),
        "{label}: covariance shape {:?} does not match coefficient count {p}",
        cov.dim()
    );

    // (2) The predict-facing accessor returns the SAME matrix: this is the
    // exact call `survival_prediction_posterior_factor` makes
    // (`fit.beta_covariance()`), so a `Some` here is what stops predict from
    // erroring "fit result does not contain conditional covariance".
    let predict_cov = fit.beta_covariance().unwrap_or_else(|| {
        panic!("{label}: fit.beta_covariance() must be Some for survival predict to serve")
    });
    assert_eq!(
        predict_cov.dim(),
        (p, p),
        "{label}: beta_covariance() shape mismatch"
    );

    // (3) A valid posterior covariance: finite, symmetric, nonnegative diagonal.
    for i in 0..p {
        let v = cov[[i, i]];
        assert!(
            v.is_finite() && v >= 0.0,
            "{label}: covariance diagonal[{i}] = {v} is not finite-and-nonneg",
        );
        for j in (i + 1)..p {
            let sij = cov[[i, j]];
            let sji = cov[[j, i]];
            assert!(sij.is_finite(), "{label}: covariance[{i},{j}] non-finite");
            assert!(
                (sij - sji).abs() <= 1e-9 * (1.0 + sij.abs()),
                "{label}: covariance not symmetric at ({i},{j}): {sij} vs {sji}",
            );
        }
    }
    // Not the trivially-zero matrix: at least one identified direction has real
    // posterior spread.
    let positive_diag = (0..p).filter(|&i| cov[[i, i]] > 0.0).count();
    assert!(
        positive_diag >= 1,
        "{label}: covariance has no positive diagonal ({positive_diag}/{p}); fit skipped cov"
    );
}

#[test]
fn survival_transformation_fit_persists_conditional_covariance_for_predict() {
    let ds = build_survival_frame();
    let cfg = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 2,
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(t, event) ~ x + survmodel(spec=net)", &ds, &cfg)
        .expect("Royston-Parmar net-survival transformation fit on the synthetic Weibull cohort");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };
    assert_fit_serves_conditional_covariance(&fit.fit, "transformation");
}

#[test]
fn survival_weibull_fit_persists_conditional_covariance_for_predict() {
    let ds = build_survival_frame();
    let cfg = FitConfig {
        survival_likelihood: Some("weibull".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(t, event) ~ x + survmodel(spec=net)", &ds, &cfg)
        .expect("parametric Weibull survival fit on the synthetic Weibull cohort");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Weibull) fit result");
    };
    assert_fit_serves_conditional_covariance(&fit.fit, "weibull");
}
