//! Bug hunt: the Wood–Pya–Säfken (WPS) smoothing-parameter-uncertainty
//! correction to the effective degrees of freedom comes out NEGATIVE on an
//! ordinary penalized Gaussian smooth, so the "corrected" EDF is SMALLER than
//! the conditional EDF and the corrected AIC penalizes complexity LESS than the
//! conditional AIC — the exact reverse of what the correction is for.
//!
//! ## The math (why this must be non-negative)
//!
//! `model_comparison.rs` defines the corrected EDF as
//!
//!   τ = tr(F) + tr(X'WX · Σ_ρ),     Σ_ρ = smoothing_correction / φ,
//!
//! and `CorrectedEdf::rho_uncertainty_df() = τ − tr(F) = tr(X'WX · Σ_ρ)`.
//! Both factors are symmetric positive-semidefinite:
//!   * `X'WX` is a weighted Gram matrix (PSD by construction), and
//!   * `Σ_ρ = J·Var(ρ̂)·Jᵀ` is a covariance contribution that
//!     `compute_smoothing_correction` (solver/estimate.rs ~1747) explicitly
//!     eigenvalue-floors to PSD before storing.
//! For two symmetric PSD matrices `tr(A·B) = tr(A^{1/2} B A^{1/2}) ≥ 0`, so the
//! ρ-uncertainty EDF inflation is necessarily `≥ 0`. The unit tests in
//! `model_comparison.rs` assert exactly this (`rho_uncertainty_df()` of `6.0`
//! and `0.0`), and the field doc calls it how much λ-uncertainty is *inflating*
//! the complexity penalty.
//!
//! ## What actually happens
//!
//! `wps_correction_term` reconstructs `X'WX` as `H · F` where
//! `H = penalized_hessian` and `F = coefficient_influence = H⁻¹X'WX`. But the
//! fit's stored `H` and `F` do NOT satisfy `H·F = X'WX`: their product is
//! grossly *asymmetric* and indefinite (observed max off-symmetry ≈ 2.4e2 and
//! min eigenvalue ≈ −3.6e2 on a 24-coefficient `s(x)` fit), because `F` is built
//! from the covariance inverse (`I − Vb_unscaled·S`) while `H` is a different
//! stored Hessian surface. The "PSD × PSD ⇒ non-negative trace" guarantee then
//! does not apply, and `tr(H·F·Σ_ρ)` lands negative.
//!
//! This test fits a plain Gaussian `y ~ s(x)`, builds the model-comparison
//! payload through the public `model_comparison_from_unified`, and asserts the
//! WPS-corrected EDF is at least the conditional EDF (the non-negativity of the
//! ρ-uncertainty df, up to genuine round-off). It currently fails because the
//! correction is a sizeable negative number.

use csv::StringRecord;
use gam::inference::model_comparison::model_comparison_from_unified;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn build_dataset(n: usize, seed: u64) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 10.0).expect("uniform");
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let mut ys = Vec::with_capacity(n);
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x: f64 = ux.sample(&mut rng);
            let y = x.sin() + noise.sample(&mut rng);
            ys.push(y);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        ys,
    )
}

#[test]
fn wps_corrected_edf_is_not_below_conditional_edf() {
    init_parallelism();
    let (data, ys) = build_dataset(400, 7);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let FitResult::Standard(std_fit) = fit_from_formula("y ~ s(x)", &data, &cfg)
        .expect("ordinary Gaussian s(x) fit should succeed")
    else {
        panic!("expected a standard fit for y ~ s(x)");
    };
    let fit = &std_fit.fit;

    // Sanity: this fit actually carries the inputs the WPS correction needs, so
    // the assertion below exercises the certified correction.
    assert!(
        fit.penalized_hessian().is_some()
            && fit.coefficient_influence().is_some()
            && fit.weighted_gram().is_some()
            && fit.smoothing_correction().is_some(),
        "fit is missing the H / F / X'WX / smoothing-correction inputs required \
         for the WPS regression"
    );

    let n = ys.len();
    let y = Array1::from(ys);
    let eta_hat = Array1::from(
        fit.artifacts
            .pirls
            .as_ref()
            .expect("Gaussian WPS fixture retains converged PIRLS geometry")
            .final_eta
            .to_vec(),
    );
    let cmp = model_comparison_from_unified(
        fit,
        y.view(),
        eta_hat.view(),
        Array1::ones(n).view(), // prior weights
        None,                   // no ALO ⇒ corrected-EDF/AIC channel only
    )
    .expect("construct WPS comparison from a finite converged Gaussian fit");

    let conditional = cmp.edf.conditional;
    let corrected = cmp
        .edf
        .corrected
        .expect("WPS fixture must retain certified corrected EDF");
    let rho_df = cmp
        .edf
        .rho_uncertainty_df()
        .expect("WPS fixture must retain its smoothing-uncertainty EDF");
    let aic_corrected = cmp
        .aic_corrected
        .expect("WPS fixture must retain certified corrected AIC");

    // Allow only genuine eigensolver round-off (relative to the conditional EDF).
    let tol = 1e-6 * conditional.abs().max(1.0);

    assert!(
        rho_df >= -tol,
        "Wood–Pya–Säfken ρ-uncertainty EDF correction is negative: \
         rho_uncertainty_df = tr(X'WX · Σ_ρ) = {rho_df:.6} (tol {tol:.2e}). \
         Both X'WX and Σ_ρ are symmetric PSD, so this trace must be ≥ 0; a \
         negative value means the corrected EDF ({corrected:.4}) dropped BELOW \
         the conditional EDF ({conditional:.4}), making the WPS-corrected AIC \
         under-penalize complexity. Root cause: the stored penalized_hessian (H) \
         and coefficient_influence (F) do not satisfy H·F = X'WX, so \
         wps_correction_term's reconstruction is not the PSD weighted Gram it \
         assumes."
    );
    assert!(
        aic_corrected >= cmp.aic_conditional - tol,
        "corrected AIC {aic_corrected:.4} must not under-penalize relative to \
         conditional AIC {:.4}",
        cmp.aic_conditional
    );
}
