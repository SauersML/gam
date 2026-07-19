//! Regression (#2373 defect A): a transformation-likelihood ("Royston-Parmar")
//! survival fit whose gradient-only outer BFGS bails on `opt::Bfgs`'s
//! flat-valley `StallPolicy` must still be minted.
//!
//! Root cause: this path declares no analytic outer Hessian, so it runs
//! `opt::Bfgs`, whose flat-stall exit gates on `‖g‖∞ ≤ tol·(1 + ‖ρ‖∞)`. In
//! log-λ space a baseline penalty railed near its box edge drives `‖ρ‖∞ ≈ 10`,
//! inflating that gate ~10× — so BFGS reports "converged (flat/stalled)" at a
//! checkpoint whose projected gradient still exceeds the un-inflated terminal
//! certificate bound, and the certificate (correctly) refuses with real
//! interior descent remaining. `optimize_survival_transformation_smoothing` now
//! resumes the outer search from the printed `rho_checkpoint` with a fresh BFGS
//! inverse-Hessian metric (bounded), which recovers the descent the accumulated
//! metric stalled on.
//!
//! DGP: the reporter's two-smooth Weibull cohort (shape 1.6, log-hazard slopes
//! +0.7 / −0.5). With two smooth terms plus the baseline I-spline penalties the
//! outer is 4–5-dimensional and one baseline coordinate rails, which is exactly
//! the regime that stalls the flat-valley gate. Before the fix this cohort
//! refuses with `RemlDidNotConverge`; a revert makes this test loud.

use csv::StringRecord;
use gam::pirls::PirlsStatus;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

const N: usize = 800;

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
}

fn build_two_smooth_survival_frame() -> gam::data::EncodedDataset {
    // Reporter DGP (#2373): T = exp(-eta/k) * Weibull(k), eta = 0.7 x1 - 0.5 x2,
    // right-censored by an exponential(mean 3). Weibull(k) is sampled from a
    // uniform via the inverse CDF (-ln u)^(1/k).
    const SHAPE: f64 = 1.6;
    const SLOPE1: f64 = 0.7;
    const SLOPE2: f64 = -0.5;
    let mut rng = Lcg::new(0x2373_A5EED_C0FFEE);

    let headers = ["time", "event", "x1", "x2"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut rows = Vec::with_capacity(N);
    for _ in 0..N {
        let x1 = -2.0 + 4.0 * rng.next_u01();
        let x2 = -2.0 + 4.0 * rng.next_u01();
        let eta = SLOPE1 * x1 + SLOPE2 * x2;
        let scale = (-eta / SHAPE).exp();
        let u = rng.next_u01().clamp(1e-12, 1.0 - 1e-12);
        let t_lat = scale * (-u.ln()).powf(1.0 / SHAPE);
        let cens = -3.0 * rng.next_u01().max(1e-12).ln();
        let exit = t_lat.min(cens);
        let event = if t_lat <= cens { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![
            format!("{exit:.17e}"),
            format!("{event:.1}"),
            format!("{x1:.17e}"),
            format!("{x2:.17e}"),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode two-smooth survival frame")
}

#[test]
fn survival_transformation_two_smooth_fit_resumes_past_flat_valley_stall() {
    let ds = build_two_smooth_survival_frame();
    let cfg = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 2,
        ..FitConfig::default()
    };
    // Before #2373A this cohort refused with a `RemlDidNotConverge` flat-valley
    // stall; the caller-level checkpoint resume must now mint a certified fit.
    let result = fit_from_formula("Surv(time, event) ~ s(x1) + s(x2)", &ds, &cfg)
        .expect("two-smooth Royston-Parmar fit must certify via the flat-stall checkpoint resume");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };

    let evidence = fit.fit.convergence_evidence();
    // A minted fit seals a certified `Converged` inner mode; the outer stall
    // that produced it can only ship as a fit once the resume reaches a
    // certified stationary rho.
    assert_eq!(
        evidence.inner_status(),
        PirlsStatus::Converged,
        "a minted survival-transformation fit must carry the certified Converged inner status"
    );
    if let Some(g) = fit.fit.outer_gradient_norm {
        assert!(
            g.is_finite(),
            "certified survival fit reported a non-finite outer_gradient_norm ({g})"
        );
    }
}
