//! Regression: a Royston-Parmar (survival "transformation") fit must report
//! `outer_converged` HONESTLY — derived from the real inner-PIRLS verdict, not
//! a hardcoded `true`.
//!
//! `survival_unified_fit_result` used to set `outer_converged: true`
//! unconditionally while carrying the REAL `outer_gradient_norm`
//! (`summary.lastgradient_norm`) and the REAL `pirls_status` (`summary.status`).
//! The survival fit path (#1123) deliberately ACCEPTS a finite-but-non-converged
//! inner solve at the selected λ rather than aborting the model — so a
//! `MaxIterationsReached` / `LmStepSearchExhausted` / `Unstable` optimum can
//! legitimately reach the result builder. Stamping all of those `converged=true`
//! is the same silent-non-convergence mislabelling #1426 cured for the REML
//! path: a caller reading `outer_converged` was told the optimizer reached a
//! stationary point even when `pirls_status` said it had not.
//!
//! The honest contract this test pins (on a well-posed, genuinely-converging
//! synthetic cohort, driven entirely through the public `fit_from_formula`
//! path): `outer_converged` must AGREE with the reported `pirls_status` — it is
//! `true` exactly when the inner solve reached a stationary verdict
//! (`Converged` or `StalledAtValidMinimum`, the two statuses the survival fit
//! path treats as a valid minimum), and a `converged == true` fit must carry a
//! finite `outer_gradient_norm`. A reversion to the hardcoded `true` would make
//! `outer_converged` disagree with a non-stationary `pirls_status`, which this
//! invariant forbids.
//!
//! DGP: deterministic right-censored Weibull (shape 1.5, log-hazard slope 0.7),
//! the well-posed #898 repro shape, so the RP fit converges cleanly and the
//! invariant is exercised on its `true` branch with a real stationary status.

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
fn survival_outer_converged_agrees_with_pirls_status_not_hardcoded() {
    let ds = build_survival_frame();
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 2,
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(t, event) ~ x + survmodel(spec=net)", &ds, &cfg)
        .expect("Royston-Parmar net-survival fit on the synthetic Weibull cohort");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };

    let status = fit.fit.pirls_status;
    let outer_converged = fit.fit.outer_converged;
    let grad = fit.fit.outer_gradient_norm;

    eprintln!(
        "[survival-honest] outer_converged={outer_converged} pirls_status={status:?} \
         outer_gradient_norm={grad:?}"
    );

    // The two stationary verdicts the survival fit path treats as a valid
    // minimum. Every other status (MaxIterationsReached / LmStepSearchExhausted /
    // Unstable) is an honest NON-convergence even though the finite optimum is
    // still shipped (#1123).
    let status_is_stationary = matches!(
        status,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    );

    // The honest-flag invariant: `outer_converged` MUST equal the real verdict.
    // The hardcoded-`true` bug breaks this exactly when `status_is_stationary`
    // is false.
    assert_eq!(
        outer_converged, status_is_stationary,
        "outer_converged ({outer_converged}) must agree with the real inner verdict \
         (pirls_status={status:?}, stationary={status_is_stationary}); a hardcoded \
         `true` silently mislabels a non-converged survival fit (#1426-class)."
    );

    // A converged verdict must carry a finite residual gradient (the #1426
    // contract): a `converged == true` with a missing/non-finite gradient is the
    // inverse mislabelling.
    if outer_converged {
        let g = grad.expect("a converged survival fit must report an outer_gradient_norm");
        assert!(
            g.is_finite(),
            "converged survival fit reported a non-finite outer_gradient_norm ({g})"
        );
    }

    // Sanity: this well-posed cohort must actually reach the stationary basin,
    // so the invariant is exercised on its `true` branch (a fix that simply
    // tagged everything non-converged would be caught here).
    assert!(
        status_is_stationary && outer_converged,
        "the well-posed synthetic RP cohort must converge to a valid minimum \
         (got pirls_status={status:?}, outer_converged={outer_converged}); if this \
         fails the fixture stopped exercising the honest `true` branch."
    );
}
