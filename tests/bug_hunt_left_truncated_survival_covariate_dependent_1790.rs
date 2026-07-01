//! Regression #1790 — left-truncated survival `Surv(entry, exit, event)` with a
//! positive delayed entry must NOT collapse to a covariate-independent fit.
//!
//! The default `family="survival"` transformation (Royston-Parmar) model centers
//! its baseline-hazard I-spline time basis at a chosen anchor. Under left
//! truncation the earliest entry age is a genuine positive left-tail point far
//! below the exit mass; anchoring there leaves the unpenalized linear-trend
//! column `X(exit) − X(anchor)` large and one-signed across every row. That
//! column is the null space of the 2nd-difference time penalty, so the inflated
//! one-signed column rails the transformation smoothing-parameter selection and
//! collapses the baseline to a degenerate, cumulative-hazard-inflated surface
//! whose covariate smooth is annihilated — the predicted survival became
//! byte-identical across covariate values (#1790). This is the transformation
//! analogue of the marginal-slope #751 defect; the fix anchors the transformation
//! baseline at the robust interior median exit age under left truncation, an
//! exact affine reparameterization that only improves conditioning.
//!
//! DGP mirrors the issue: proportional-hazards `λ(x) = 0.4·exp(0.9·x)` with
//! `x ∈ [-1, 1]`, exponential censoring, and a constant delayed entry. The true
//! covariate effect on the log-cumulative-hazard scale is the log-hazard-ratio
//! `0.9·(x_hi − x_lo) = 0.9·1.6 = 1.44` (proportional hazards ⇒ the covariate is
//! additive on `log H`). The pin asserts truth recovery: the fitted covariate
//! linear predictor must differ substantially between two covariate values under
//! left truncation, and must barely change versus the `entry = 0` control (small
//! delayed entry is a small perturbation of the same cohort). Under the bug the
//! left-truncated covariate effect collapses to ≈ 0 while the control recovers
//! ≈ 1.4, so both assertions fail hard.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const N: usize = 400;
const X_LO: f64 = -0.8;
const X_HI: f64 = 0.8;
/// True covariate log-hazard-ratio between `X_HI` and `X_LO` on the
/// log-cumulative-hazard scale: `0.9 · (X_HI − X_LO)`.
const TRUE_COVARIATE_DELTA: f64 = 0.9 * (X_HI - X_LO);

/// Deterministic LCG uniform stream — byte-identical data run-to-run.
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
    fn next_exp(&mut self, mean: f64) -> f64 {
        let u = self.next_u01().clamp(1e-12, 1.0 - 1e-12);
        -mean * u.ln()
    }
}

/// Build a left-truncated survival frame with a CONSTANT delayed entry. The
/// covariate signal (`x`, latent event/censoring times) is generated from a
/// fixed seed so the two entry settings differ ONLY in the delayed-entry column
/// and the exit floor — everything else is the same cohort, exactly as the
/// issue's repro varies `entry` alone.
fn build_frame(entry: f64) -> gam::data::EncodedDataset {
    let mut rng = Lcg::new(0xABCD_1234_5678_9EF0);
    let headers = ["entry", "exit", "event", "x"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut rows = Vec::with_capacity(N);
    for _ in 0..N {
        let x = -1.0 + 2.0 * rng.next_u01();
        let lam = 0.4 * (0.9 * x).exp();
        let t_lat = rng.next_exp(1.0 / lam);
        let cens = rng.next_exp(5.0);
        let exit = t_lat.min(cens).max(entry + 0.11);
        let event = if t_lat <= cens { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![
            format!("{entry:.17e}"),
            format!("{exit:.17e}"),
            format!("{event:.1}"),
            format!("{x:.17e}"),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode left-truncated survival frame")
}

/// Fitted-quality summary of a left-truncated transformation survival fit.
struct FitSummary {
    /// Fitted covariate linear-predictor difference `η(X_HI) − η(X_LO)` on the
    /// log-cumulative-hazard scale (covariate-dependence probe).
    cov_delta: f64,
    /// Largest-magnitude fitted time-baseline coefficient. Under the #1791
    /// degenerate fit the unpenalized I-spline linear-trend column rails to a
    /// huge value (cumulative hazard inflated ~10³×, `S(t) ≡ 0`); a well-posed
    /// fit keeps the baseline coefficients O(1–10). This pins non-degeneracy of
    /// the baseline time-block directly, independent of the covariate probe.
    time_base_max_abs: f64,
}

/// Fit the transformation survival model and return the fitted covariate linear
/// predictor difference `η(X_HI) − η(X_LO)` on the log-cumulative-hazard scale
/// together with the largest-magnitude fitted time-baseline coefficient.
fn fitted_summary(entry: f64) -> FitSummary {
    let ds = build_frame(entry);
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 2,
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(entry, exit, event) ~ s(x)", &ds, &cfg)
        .expect("left-truncated Royston-Parmar transformation fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation (Royston-Parmar) fit result");
    };

    assert_eq!(
        fit.fit.blocks.len(),
        1,
        "single-cause survival fit must have exactly one coefficient block"
    );
    let beta = &fit.fit.blocks[0].beta;
    let time_base = fit.time_base_ncols;
    assert!(
        beta.len() > time_base,
        "beta must carry covariate columns beyond the {time_base} time-basis columns (got {})",
        beta.len()
    );
    let time_base_max_abs = beta
        .slice(ndarray::s![..time_base])
        .iter()
        .fold(0.0_f64, |acc, &b| acc.max(b.abs()));
    let cov_beta = beta.slice(ndarray::s![time_base..]).to_owned();

    // Rebuild the FITTED covariate smooth design at the two probe covariate
    // values using the model's own resolved term spec (identical to the predict
    // path), then contract with the fitted covariate coefficients.
    let x_idx = ["entry", "exit", "event", "x"]
        .iter()
        .position(|h| *h == "x")
        .expect("x column index");
    let mut covgrid = Array2::<f64>::zeros((2, 4));
    covgrid[[0, x_idx]] = X_LO;
    covgrid[[1, x_idx]] = X_HI;
    let design = build_term_collection_design(covgrid.view(), &fit.resolvedspec)
        .expect("rebuild covariate design at probe x values");
    assert_eq!(
        design.design.ncols(),
        cov_beta.len(),
        "covariate design width must match fitted covariate coefficient count"
    );
    let eta = design.design.apply(&cov_beta);
    FitSummary {
        cov_delta: eta[1] - eta[0],
        time_base_max_abs,
    }
}

#[test]
fn left_truncated_survival_fit_stays_covariate_dependent_1790() {
    init_parallelism();

    let control = fitted_summary(0.0);
    let truncated = fitted_summary(0.05);
    let delta_control = control.cov_delta;
    let delta_truncated = truncated.cov_delta;

    eprintln!(
        "[#1790] true_delta={TRUE_COVARIATE_DELTA:.4} control(entry=0)={delta_control:.4} \
         truncated(entry=0.05)={delta_truncated:.4} \
         time_base_max_abs: control={:.4} truncated={:.4}",
        control.time_base_max_abs, truncated.time_base_max_abs
    );

    // Sanity: the well-posed right-censored control recovers a clearly positive
    // covariate effect (so any failure below is a truncation defect, not a broken
    // DGP). Threshold is well below the true 1.44 log-HR to allow smooth shrinkage.
    assert!(
        delta_control > 0.6,
        "control (entry=0) must recover the positive covariate effect \
         (true log-HR={TRUE_COVARIATE_DELTA:.3}); got η(x_hi)-η(x_lo)={delta_control:.4}"
    );

    // THE #1790 PIN: left truncation must NOT annihilate the covariate smooth.
    // Under the bug the transformation smoothing selection rails and this
    // collapses to ≈ 0 (predicted survival becomes byte-identical across x).
    assert!(
        delta_truncated > 0.6,
        "left-truncated (entry=0.05) fit collapsed the covariate effect \
         (true log-HR={TRUE_COVARIATE_DELTA:.3}); got η(x_hi)-η(x_lo)={delta_truncated:.4} \
         — the degenerate covariate-independent surface #1790 forbids"
    );

    // A small delayed entry is a small perturbation of the same cohort, so the
    // recovered covariate effect must be close to the control's — not railed away.
    assert!(
        (delta_truncated - delta_control).abs() < 0.6,
        "left truncation should barely change the covariate effect: \
         control={delta_control:.4} truncated={delta_truncated:.4} \
         (|Δ|={:.4} exceeds tolerance 0.6)",
        (delta_truncated - delta_control).abs()
    );

    // THE #1791 NON-DEGENERACY PIN. #1790 could pass the covariate probe while the
    // baseline time-block silently railed (the transformation LAML under-smoothing
    // that inflates the cumulative hazard ~10³× and drives `S(t) ≡ 0`). Pin the
    // baseline directly: under left truncation the fitted time-baseline
    // coefficients must stay bounded and comparable to the well-posed control's,
    // not blow up into the huge, flat, covariate-independent offset #1791 reports.
    // A small delayed entry is a small perturbation of the same cohort, so a
    // generous 4× factor (plus an absolute floor for tiny control baselines)
    // catches the ~10³× inflation without over-constraining ordinary shrinkage.
    let baseline_ceiling = (4.0 * control.time_base_max_abs).max(20.0);
    assert!(
        truncated.time_base_max_abs <= baseline_ceiling,
        "left-truncated baseline time-block inflated (#1791): \
         truncated max|β_time|={:.4} exceeds ceiling {:.4} \
         (control max|β_time|={:.4}) — the degenerate cumulative-hazard-inflated \
         baseline that drives S(t)≡0 is forbidden",
        truncated.time_base_max_abs,
        baseline_ceiling,
        control.time_base_max_abs
    );
}
