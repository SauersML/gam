//! Regression for #501: a Gaussian location-scale fit must size the *scale*
//! (log-σ) predictor's spatial smooth parsimoniously, not with the generous
//! basis the mean gets. The held-out quality test
//! `quality_vs_gamlss_gagurine_location_scale` checks the *consequence*
//! (predictive NLL/CRPS); this test checks the *mechanism* directly and from a
//! different angle so a future change that re-inflates the scale basis (or that
//! starts ignoring an explicit user `k`) is caught even if the quality metrics
//! happen to stay within tolerance.
use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use std::path::Path;

const GAGURINE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/gagurine.csv");

fn fit_location_scale(noise_formula: &str) -> GaussianLocationScaleFitResult {
    init_parallelism();
    let ds = load_csvwith_inferred_schema(Path::new(GAGURINE_CSV)).expect("load gagurine.csv");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(noise_formula.to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula("GAG ~ s(Age, bs='tp')", &ds, &cfg).expect("fit location-scale") {
        FitResult::GaussianLocationScale(r) => r,
        _ => panic!("expected GaussianLocationScale fit result"),
    }
}

fn block_ncoef(result: &GaussianLocationScaleFitResult, role: BlockRole) -> usize {
    let label = format!("{role:?}");
    result
        .fit
        .fit
        .block_by_role(role)
        .unwrap_or_else(|| panic!("missing {label} block"))
        .beta
        .len()
}

/// With no explicit basis size on the scale smooth, the mean keeps the generous
/// thin-plate default while the scale is held to the conservative
/// secondary-predictor default — strictly smaller, and modest in absolute terms.
#[test]
fn scale_smooth_basis_is_parsimonious_relative_to_mean() {
    let result = fit_location_scale("1 + s(Age, bs='tp')");
    let mean_ncoef = block_ncoef(&result, BlockRole::Location);
    let scale_ncoef = block_ncoef(&result, BlockRole::Scale);

    // Mean keeps the generous spatial default (was ~40 pre-fix; assert it is
    // not collapsed by the parsimony pass that targets only the scale block).
    assert!(
        mean_ncoef >= 25,
        "mean smooth must keep its generous basis, got ncoef={mean_ncoef}"
    );
    // Scale is held to the conservative default (centers≈10 + small nullspace).
    assert!(
        scale_ncoef <= 18,
        "scale smooth must be parsimonious, got ncoef={scale_ncoef}"
    );
    assert!(
        scale_ncoef < mean_ncoef,
        "scale basis ({scale_ncoef}) must be strictly smaller than mean basis ({mean_ncoef})"
    );
}

/// An explicit `k=` on the scale smooth must be honored verbatim — the
/// parsimony pass only fills in a default when the user gave none.
#[test]
fn explicit_scale_k_is_respected() {
    let parsimonious = fit_location_scale("1 + s(Age, bs='tp')");
    let explicit = fit_location_scale("1 + s(Age, bs='tp', k=25)");

    let default_scale = block_ncoef(&parsimonious, BlockRole::Scale);
    let explicit_scale = block_ncoef(&explicit, BlockRole::Scale);

    // k=25 sizes the thin-plate basis to ~25 centers (minus an identifiability
    // constraint), well above the injected conservative default; if the default
    // were overriding the user's k these would be equal.
    assert!(
        explicit_scale >= 20,
        "explicit k=25 must yield ~25 scale coefficients, got {explicit_scale}"
    );
    assert!(
        explicit_scale > default_scale,
        "explicit k=25 ({explicit_scale}) must exceed the conservative default ({default_scale})"
    );
}
