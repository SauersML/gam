//! Regression for #1504 — a Gaussian location-scale (gaulss) fit with a
//! by-group smooth in **both** the mean and the log-σ block must not crash on a
//! joint-Hessian shape mismatch.
//!
//! Root cause (fixed in `GaussianLocationScaleFamily::exact_joint_block_designs`):
//! the joint exact-Newton path reached for the family's STORED pre-audit block
//! designs whenever the exact joint solve was supported, ignoring the
//! identifiability-CONSTRAINED `specs` it was handed. A by-group smooth in both
//! blocks aliases per-group columns that the identifiability audit drops, so the
//! constrained specs are NARROWER than the stored designs. The inner joint-Newton
//! solve and the consumer's dense Hessian `total` are sized from the post-audit
//! specs, so building the Hessian against the wider stored designs tripped the
//! shape check ("joint Newton inner exact-newton operator mismatch: dense Hessian
//! shape mismatch: got 36x36, expected 32x32") and aborted EVERY such fit.
//!
//! This guards the user-visible symptom — the fit must SUCCEED and return finite,
//! correctly-sized location and log-σ coefficient blocks — with a fast, R-free
//! fixture that exercises the exact joint-Newton path the bug lived on. It is the
//! cheap counterpart to the reference-quality
//! `quality_vs_gamlss_gaussian_location_scale_by_group` gate, which proves the
//! same fit also RECOVERS the truth but requires R/gamlss.

use csv::StringRecord;
use gam::solver::estimate::BlockRole;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_PER_GROUP: usize = 80;

/// Group-A truth: smooth mean, smooth heteroscedastic σ (a hump).
fn mean_a(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin()
}
fn sigma_a(x: f64) -> f64 {
    0.10 + 0.10 * (std::f64::consts::PI * x).sin()
}
/// Group-B truth: a DIFFERENT mean shape and a near-linear σ ramp, so the two
/// per-group smooths are genuinely distinct and the by-group split is real.
fn mean_b(x: f64) -> f64 {
    0.5 + 0.3 * (3.0 * std::f64::consts::PI * x).sin()
}
fn sigma_b(x: f64) -> f64 {
    0.12 + 0.08 * x
}

#[test]
fn gaulss_by_group_smooth_in_both_blocks_fits_without_hessian_shape_mismatch() {
    init_parallelism();

    // Two groups, each with its own mean and σ function. Group A rows first, then
    // group B, so the inferred categorical levels are ["A", "B"]. A by-group tp
    // smooth in BOTH blocks is exactly the structure whose aliased per-group
    // columns the identifiability audit drops (the #1504 narrowing).
    let mut rng = StdRng::seed_from_u64(1504);
    let ux = Uniform::new(0.0_f64, 1.0_f64).expect("uniform x");
    let std_normal = Normal::new(0.0_f64, 1.0_f64).expect("standard normal");

    let headers = vec!["y".to_string(), "x".to_string(), "group".to_string()];
    let mut rows: Vec<StringRecord> = Vec::with_capacity(2 * N_PER_GROUP);

    for _ in 0..N_PER_GROUP {
        let x = ux.sample(&mut rng);
        let y = mean_a(x) + sigma_a(x) * std_normal.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "A".to_string(),
        ]));
    }
    for _ in 0..N_PER_GROUP {
        let x = ux.sample(&mut rng);
        let y = mean_b(x) + sigma_b(x) * std_normal.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "B".to_string(),
        ]));
    }

    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode by-group data");

    // Gaussian location-scale: by-group tp smooth on BOTH μ and log σ. Before
    // #1504 this `.expect(...)` fired with the dense-Hessian shape mismatch.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x, bs='tp', by=group)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='tp', by=group)", &data, &cfg)
        .expect("#1504: gaulss by-group fit in both blocks must not crash on a Hessian shape mismatch");

    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected a GaussianLocationScale fit result for a noise_formula model");
    };

    // The fit must carry finite, non-empty location (μ) and log-σ coefficient
    // blocks — a converged joint solve, not a degenerate empty fit.
    let beta_mean = &fit
        .fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location-scale fit must carry a Location (mean) block")
        .beta;
    let beta_logsig = &fit
        .fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("location-scale fit must carry a Scale (log-σ) block")
        .beta;

    assert!(!beta_mean.is_empty(), "mean block must have coefficients");
    assert!(!beta_logsig.is_empty(), "log-σ block must have coefficients");
    assert!(
        beta_mean.iter().all(|b| b.is_finite()),
        "mean coefficients must all be finite: {beta_mean:?}"
    );
    assert!(
        beta_logsig.iter().all(|b| b.is_finite()),
        "log-σ coefficients must all be finite: {beta_logsig:?}"
    );
}
