//! Regression for gam#2152 at the fit-orchestration layer.
//!
//! The constant-curvature smooth `curv(x, z, kappa=K)` documents `kappa=` as a
//! FIXED sectional curvature (the mgcv-`sp=` convention): an explicit value
//! selects the geometry (`Sᵈ` for κ>0, `ℝᵈ` for κ=0, `Hᵈ` for κ<0) and the fit
//! must build and KEEP the design/penalty at exactly that κ. Before the fix the
//! pure-`curv()` fit treated κ as an outer hyperparameter to ESTIMATE (#944 /
//! #1464), overwriting the user's `kappa=` at three sites (the κ-fair baseline
//! sign-scan pin, the `all_spatial_are_cc` κ-fair argmin override, and the joint
//! [ρ, κ] solver where κ is the ψ coordinate), so every `kappa=` seed landed on
//! the same κ̂ — spherical, flat, and hyperbolic fits were bit-for-bit identical.
//!
//! The top-level `gam` crate cannot build in this environment (a `build.rs`
//! author tripwire), so the issue's `fit_from_formula` path is exercised here in
//! `gam-models`, which builds standalone. This inspects the REALIZED geometry
//! (the frozen κ the design was actually built at) — a different angle than the
//! Python repro's prediction-equality check, so it catches the regression even
//! if two fits' predictions happened to coincide.

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_terms::smooth::SmoothBasisSpec;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

/// A smooth 2-D surface on the small ball `‖(x, z)‖² < 1/3`, strictly inside
/// BOTH the κ = +K and κ = −K stereographic charts for K ≤ 3, so a pinned fit at
/// either sign is well posed rather than chart-rejected.
fn small_ball_dataset(n: usize, seed: u64) -> gam_data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(-0.3_f64, 0.3).unwrap();
    let headers: Vec<String> = ["x", "z", "y"].iter().map(|s| s.to_string()).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = unif.sample(&mut rng);
        let z = unif.sample(&mut rng);
        let y = (4.0 * x).sin() * (4.0 * z).cos();
        rows.push(StringRecord::from(vec![
            x.to_string(),
            z.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// The frozen sectional curvature and its pin flag after a `curv(...)` fit — the
/// value/geometry the design was actually built and kept at.
fn fitted_kappa_and_pin(formula: &str, ds: &gam_data::EncodedDataset) -> (f64, bool) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, ds, &cfg).expect("curv() fit");
    let StandardFitResult { resolvedspec, .. } = match result {
        FitResult::Standard(s) => s,
        _ => panic!("expected Standard fit"),
    };
    let SmoothBasisSpec::ConstantCurvature { spec, .. } = &resolvedspec.smooth_terms[0].basis
    else {
        panic!("expected a ConstantCurvature term after fit");
    };
    (spec.kappa, spec.kappa_fixed)
}

/// The core contract: a pinned `kappa=` is kept verbatim. Spherical +3 freezes at
/// +3, hyperbolic −3 at −3, and a pinned flat 0 stays exactly 0 — never
/// re-derived to a common κ̂.
#[test]
fn pinned_kappa_is_kept_verbatim() {
    let ds = small_ball_dataset(600, 2152);

    let (k_pos, pinned_pos) = fitted_kappa_and_pin("y ~ curv(x, z, kappa=3, centers=20)", &ds);
    assert!(
        pinned_pos,
        "explicit kappa=3 must mark the term kappa_fixed"
    );
    assert!(
        (k_pos - 3.0).abs() < 1e-9,
        "pinned kappa=+3 was re-derived: fit kept κ = {k_pos} (want +3)"
    );

    let (k_neg, pinned_neg) = fitted_kappa_and_pin("y ~ curv(x, z, kappa=-3, centers=20)", &ds);
    assert!(
        pinned_neg,
        "explicit kappa=-3 must mark the term kappa_fixed"
    );
    assert!(
        (k_neg + 3.0).abs() < 1e-9,
        "pinned kappa=-3 was re-derived: fit kept κ = {k_neg} (want -3)"
    );

    let (k_flat, pinned_flat) = fitted_kappa_and_pin("y ~ curv(x, z, kappa=0, centers=20)", &ds);
    assert!(
        pinned_flat,
        "explicit kappa=0 must mark the term kappa_fixed"
    );
    assert!(
        k_flat.abs() < 1e-9,
        "pinned kappa=0 drifted off flat: fit kept κ = {k_flat} (want 0)"
    );

    assert!(
        (k_pos - k_neg).abs() > 1.0,
        "pinned +3 and -3 collapsed to the same κ ({k_pos} vs {k_neg})"
    );
}

/// Guard against over-correction: omitting `kappa=` must leave κ FREE for the
/// #944/#1464 outer estimation (`kappa_fixed = false`); the pin path must not
/// swallow the estimation path.
#[test]
fn omitted_kappa_stays_free_for_estimation() {
    let ds = small_ball_dataset(400, 7);
    let (_k, pinned) = fitted_kappa_and_pin("y ~ curv(x, z, centers=20)", &ds);
    assert!(
        !pinned,
        "omitted kappa= must leave the term free to estimate κ (kappa_fixed=false)"
    );
}

/// The issue's independent tell: because a pinned κ genuinely builds the design
/// at that κ, an out-of-chart pinned κ is now REJECTED by `validate_chart_points`
/// (`1 + κ‖x‖² > 0`) — under the old silently-re-derive-κ behaviour such a fit
/// was wrongly ACCEPTED at a different, in-chart estimated κ. Here ‖x‖² reaches
/// ≈0.18 and κ = −50 needs ‖x‖² < 1/50 = 0.02, so far rows are out of chart.
#[test]
fn pinned_out_of_chart_kappa_is_rejected() {
    let ds = small_ball_dataset(300, 99);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ curv(x, z, kappa=-50, centers=20)", &ds, &cfg);
    assert!(
        result.is_err(),
        "a pinned out-of-chart kappa=-50 must be rejected (design genuinely built \
         at κ=-50, so validate_chart_points must fire); got Ok — κ was silently re-derived"
    );
}
