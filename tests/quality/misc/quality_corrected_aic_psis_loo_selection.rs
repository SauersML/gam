//! End-to-end OBJECTIVE quality for the #946 model-comparison channels: exact
//! Wood–Pya–Säfken corrected conditional AIC and zero-refit PSIS-LOO, computed
//! from the fit-retained exact pieces (smoothing-parameter covariance Σ_ρ and
//! the ALO leave-one-out predictions).
//!
//! TRUTH IS CONSTRUCTED, NOT BORROWED. Each arm simulates data from a *known*
//! generative structure and asserts the comparison channels recover the right
//! ordering / sign — a peer tool is never the oracle here.
//!
//!   A. PREDICTIVE SELECTION (PSIS-LOO). Generate data where one of two nested
//!      models is the genuine generator (a real smooth signal vs an
//!      over-flexible competitor on pure noise). The paired Δelpd from
//!      `compare` must point at the model that actually predicts held-out data
//!      better, with the sign the issue defines (positive Δelpd favours `a`).
//!
//!   B. CORRECTED-EDF DIRECTION (Wood–Pya–Säfken). The corrected effective df
//!      must account for smoothing-parameter uncertainty: `τ ≥ tr(F)` exactly
//!      (the ρ-uncertainty contribution `tr(X'WX·Σ_ρ)` is a PSD trace), and the
//!      corrected AIC must therefore penalise complexity at least as much as the
//!      naive conditional AIC. On a fit with a genuinely penalised smooth the
//!      correction must be strictly positive — there IS λ-uncertainty — so the
//!      corrected AIC must exceed the conditional one.
//!
//! Bounds are not weakened to force a pass; a genuine shortfall failing is the
//! intended behaviour.

use csv::StringRecord;
use gam::inference::alo::compute_alo_diagnostics_from_fit;
use gam::inference::model_comparison::{compare, model_comparison_from_unified};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array1;

/// Deterministic standard-normal stream (splitmix64 + Box–Muller). No RNG / env
/// dependency — identical every run, satisfying the determinism spec.
struct DetNormal {
    state: u64,
}
impl DetNormal {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (((z >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Build a 2-column dataset (predictor `x`, response `y`) from raw vectors.
fn make_dataset(x: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn corrected_aic_penalizes_at_least_as_much_as_conditional() {
    init_parallelism();

    // Genuine smooth signal: y = sin(2πx) + noise. The fit is a penalised
    // smooth, so there is real smoothing-parameter uncertainty and the WPS
    // correction term must be strictly positive.
    let n = 200usize;
    let mut rng = DetNormal::new(0xC0FFEE_u64);
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (2.0 * std::f64::consts::PI * xi).sin() + 0.25 * rng.normal())
        .collect();
    let ds = make_dataset(&x, &y);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    let alo =
        compute_alo_diagnostics_from_fit(&fit.fit, ds.values.column(1)).expect("ALO diagnostics");

    // Fitted linear predictor (identity link, no offset): the in-sample mean.
    // It belongs to the converged fit geometry; ALO owns only the deleted-row
    // predictor and uncertainty diagnostics.
    let eta_hat = Array1::from(
        fit.fit
            .artifacts
            .pirls
            .as_ref()
            .expect("Gaussian comparison fixture retains converged PIRLS geometry")
            .final_eta
            .to_vec(),
    );
    let weights = Array1::<f64>::ones(n);
    let comparison = model_comparison_from_unified(
        &fit.fit,
        ds.values.column(1),
        eta_hat.view(),
        weights.view(),
        Some(alo.eta_tilde.view()),
    )
    .expect("construct comparison channels for the exact Gaussian fixture");
    let corrected_edf = comparison
        .edf
        .corrected
        .expect("exact Gaussian smooth fixture retains corrected EDF");
    let rho_uncertainty_df = comparison
        .edf
        .rho_uncertainty_df()
        .expect("exact Gaussian smooth fixture retains the WPS correction");
    let aic_corrected = comparison
        .aic_corrected
        .expect("exact Gaussian smooth fixture retains corrected AIC");

    eprintln!(
        "corrected-AIC arm: edf_cond={:.4} edf_corr={:.4} rho_df={:.4} \
         aic_cond={:.4} aic_corr={:.4}",
        comparison.edf.conditional,
        corrected_edf,
        rho_uncertainty_df,
        comparison.aic_conditional,
        aic_corrected,
    );

    // The conditional edf is the usual tr(F); on a wiggly smooth it must be
    // clearly above the 2-df null (intercept + linear).
    assert!(
        comparison.edf.conditional > 2.0,
        "penalised smooth should use > 2 effective df, got {:.4}",
        comparison.edf.conditional
    );

    // WPS correction direction: the ρ-uncertainty contribution tr(X'WX·Σ_ρ) is a
    // trace of PSD operators and must be >= 0; with a genuinely penalised smooth
    // it must be strictly positive (there is real λ-uncertainty).
    assert!(
        rho_uncertainty_df >= 0.0,
        "WPS ρ-uncertainty df must be non-negative, got {:.6}",
        rho_uncertainty_df
    );
    assert!(
        rho_uncertainty_df > 1e-6,
        "a penalised smooth must carry strictly positive smoothing-parameter \
         uncertainty in its corrected edf, got {:.6}",
        rho_uncertainty_df
    );

    // Therefore the corrected AIC penalises complexity at least as much as the
    // naive conditional AIC (same log-lik, larger edf).
    assert!(
        aic_corrected >= comparison.aic_conditional,
        "corrected AIC {:.4} must be >= conditional AIC {:.4}",
        aic_corrected,
        comparison.aic_conditional
    );
    assert!(
        aic_corrected.is_finite() && comparison.aic_conditional.is_finite(),
        "AIC values must be finite"
    );

    // The ALO predictive channel must be populated with one finite contribution
    // per row. Its cross-observation tail fit is an influence diagnostic, not a
    // PSIS reliability estimand, so it is deliberately not used as a pass bar.
    let loo = comparison.loo.expect("PSIS-LOO channel populated");
    assert_eq!(loo.pointwise.len(), n, "pointwise elpd length");
    assert!(loo.elpd.is_finite(), "elpd finite");
    assert!(
        loo.pointwise.iter().all(|value| value.is_finite()),
        "pointwise elpd contributions must be finite"
    );
}

#[test]
fn psis_loo_paired_comparison_prefers_the_true_generator() {
    init_parallelism();

    // Truth: y depends on a SMOOTH signal in x. Model A fits that smooth; model
    // B fits only an intercept+linear term (mis-specified, cannot capture the
    // curvature). PSIS-LOO Δelpd must favour the correctly-specified smooth.
    let n = 250usize;
    let mut rng = DetNormal::new(0x1234_5678_u64);
    let x: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * i as f64 / (n as f64 - 1.0))
        .collect();
    // A clearly nonlinear mean: a double bump. Linear-in-x cannot fit it.
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (3.0 * xi).sin() + 0.5 * xi * xi + 0.20 * rng.normal())
        .collect();
    let ds = make_dataset(&x, &y);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let fit_smooth = match fit_from_formula("y ~ s(x)", &ds, &cfg).expect("smooth fit") {
        FitResult::Standard(f) => f,
        _ => panic!("expected standard fit"),
    };
    let fit_linear = match fit_from_formula("y ~ x", &ds, &cfg).expect("linear fit") {
        FitResult::Standard(f) => f,
        _ => panic!("expected standard fit"),
    };

    let cmp_of = |fit: &gam::StandardFitResult| {
        let alo = compute_alo_diagnostics_from_fit(&fit.fit, ds.values.column(1))
            .expect("ALO diagnostics");
        let eta_hat = Array1::from(
            fit.fit
                .artifacts
                .pirls
                .as_ref()
                .expect("Gaussian comparison fixture retains converged PIRLS geometry")
                .final_eta
                .to_vec(),
        );
        let weights = Array1::<f64>::ones(n);
        model_comparison_from_unified(
            &fit.fit,
            ds.values.column(1),
            eta_hat.view(),
            weights.view(),
            Some(alo.eta_tilde.view()),
        )
        .expect("construct comparison channels for the exact Gaussian fixture")
    };

    let cmp_smooth = cmp_of(&fit_smooth);
    let cmp_linear = cmp_of(&fit_linear);

    // a = smooth (true generator), b = linear (mis-specified).
    let report = compare(&cmp_smooth, &cmp_linear)
        .expect("compare two finite fits evaluated on the same response rows");
    let delta_elpd = report
        .delta_elpd
        .expect("aligned Gaussian fixtures retain paired elpd");
    let delta_elpd_se = report
        .delta_elpd_se
        .expect("multi-row Gaussian fixtures retain paired elpd uncertainty");
    let delta_aic_corrected = report
        .delta_aic_corrected
        .expect("both exact Gaussian fixtures retain corrected AIC");
    let smooth_elpd = cmp_smooth
        .loo
        .as_ref()
        .expect("smooth Gaussian fixture retains its ALO predictive channel")
        .elpd;
    let linear_elpd = cmp_linear
        .loo
        .as_ref()
        .expect("linear Gaussian fixture retains its ALO predictive channel")
        .elpd;

    eprintln!(
        "paired-comparison arm: rows_aligned={} Δelpd={:.4} (se {:.4}) ΔAIC_corr={:.4} \
         elpd_smooth={:.4} elpd_linear={:.4}",
        report.rows_aligned,
        delta_elpd,
        delta_elpd_se,
        delta_aic_corrected,
        smooth_elpd,
        linear_elpd,
    );

    assert!(
        report.rows_aligned,
        "both fits are on the same response — paired comparison must align"
    );
    // Positive Δelpd favours `a` (the smooth). The true generator must win the
    // predictive comparison, and by a margin larger than its own SE (a genuine,
    // not within-noise, preference).
    assert!(
        delta_elpd > 0.0,
        "PSIS-LOO must prefer the true smooth generator: Δelpd={:.4}",
        delta_elpd
    );
    assert!(
        delta_elpd > delta_elpd_se,
        "the predictive preference for the true model must exceed its SE: \
         Δelpd={:.4} se={:.4}",
        delta_elpd,
        delta_elpd_se
    );
    // The corrected-AIC channel must agree: smaller corrected AIC for the smooth
    // (ΔAIC_corrected = AIC_corr(smooth) − AIC_corr(linear) < 0 favours smooth).
    assert!(
        delta_aic_corrected < 0.0,
        "corrected AIC must also favour the true smooth: ΔAIC_corrected={:.4}",
        delta_aic_corrected
    );
}
