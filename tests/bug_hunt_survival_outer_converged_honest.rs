//! Regression: a Royston-Parmar (survival "transformation") fit must carry an
//! HONEST convergence verdict — historically `survival_unified_fit_result` set
//! `outer_converged: true` unconditionally while carrying the REAL
//! `pirls_status`, so a `MaxIterationsReached` / `LmStepSearchExhausted` /
//! `Unstable` optimum was silently mislabelled as converged (#1426-class).
//!
//! The contract has since been strengthened past the boolean entirely (SPEC
//! 20): `UnifiedFitResult` seals its convergence proof in a private
//! `FitConvergenceEvidence` that the checked constructor mints only from a
//! `Converged` inner status plus outer evidence — the survival fit path gates
//! every exhausted/stalled PIRLS status behind
//! `survival_pirls_status_is_certified` and returns a typed error instead of a
//! fit. The mislabelling this file was written against is therefore
//! unrepresentable: there is no flag to disagree with the status, because a
//! fit that exists IS the verdict.
//!
//! What remains worth pinning through the public `fit_from_formula` path:
//! (1) the well-posed cohort still converges (a regression that starts
//! refusing it is loud), and (2) the minted fit's sealed evidence reports the
//! certified `Converged` inner status and a finite residual gradient when one
//! was measured.
//!
//! DGP: deterministic right-censored Weibull (shape 1.5, log-hazard slope 0.7),
//! the well-posed #898 repro shape, so the RP fit converges cleanly.

use csv::StringRecord;
use gam::pirls::PirlsStatus;
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

#[test]
fn survival_fit_existence_is_the_certified_convergence_verdict() {
    let ds = build_survival_frame();
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 2,
        ..FitConfig::default()
    };
    // A non-converged solve now surfaces here as a typed error; the well-posed
    // cohort regressing into refusal is exactly as loud as the old
    // mislabelling assertion.
    let result = fit_from_formula("Surv(t, event) ~ x + survmodel(spec=net)", &ds, &cfg)
        .expect("Royston-Parmar net-survival fit on the synthetic Weibull cohort");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };

    let evidence = fit.fit.convergence_evidence();
    let grad = fit.fit.outer_gradient_norm;

    eprintln!(
        "[survival-honest] inner_status={:?} outer_iterations={} outer_gradient_norm={grad:?}",
        evidence.inner_status(),
        evidence.outer_iterations(),
    );

    // The sealed constructor only certifies a `Converged` inner mode; the
    // exhausted/stalled statuses (`MaxIterationsReached` /
    // `LmStepSearchExhausted` / `Unstable` / `StalledAtValidMinimum`) are
    // rejected by `survival_pirls_status_is_certified` before assembly.
    assert_eq!(
        evidence.inner_status(),
        PirlsStatus::Converged,
        "a minted survival fit must carry the certified Converged inner status; \
         any other value means the sealed evidence constructor regressed"
    );

    // A measured residual gradient on the shipped fit must be finite (the
    // #1426 contract): a certified fit with a non-finite gradient is the
    // inverse mislabelling.
    if let Some(g) = grad {
        assert!(
            g.is_finite(),
            "certified survival fit reported a non-finite outer_gradient_norm ({g})"
        );
    }
}
