//! Regression for issue #1507: `model.sample()` must respect hard *box*
//! constraints on a linear coefficient (`nonnegative()` / `nonpositive()` /
//! `linear(min=, max=)` / `constrain(...)`).
//!
//! These bounds are enforced at fit time as KKT inequality rows
//! (`src/terms/smooth/design_construction.rs` reads
//! `LinearTermSpec.coefficient_min` / `coefficient_max`), so the point estimate
//! pins to the active boundary. The posterior must live in the same admissible
//! region — a coefficient that is not allowed to be negative cannot have ~half
//! its posterior mass below zero. Before the fix, `sample_standard` drew a plain
//! unconstrained Gaussian `N(mode, φ·H⁻¹)` centred on the boundary and returned
//! it with `rhat ≈ 1` / `converged = true` — a confidently-wrong posterior with
//! roughly half the draws on the forbidden side. The sampler now routes box
//! constraints (and shape constraints, #1509) through a truncated-Gaussian Gibbs
//! sampler whose every draw is feasible by construction.
//!
//! Both subtests fit a strongly-negative true slope so the bound is *active*
//! (the fitted slope pins to the boundary), then sample and assert essentially
//! all draws of that coefficient stay inside the constraint.

use gam::hmc::NutsConfig;
use gam::inference::model::{
    FittedFamily, FittedModel, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind,
};
use gam::sample::sample_saved_model;
use gam::smooth::{build_term_collection_design, freeze_term_collection_from_design};
use gam::types::{LikelihoodSpec, StandardLink};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use std::io::Write;

/// Deterministic data with a strongly NEGATIVE slope of `x` on `y`, so any
/// `β_x ≥ a` lower bound with `a ≥ -1` binds (the unconstrained slope is ≈ -3).
fn negative_slope_csv(n: usize) -> String {
    let mut csv = String::from("x,y\n");
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    for i in 0..n {
        let k = (i + 1) as f64;
        let x = (k * phi).fract() * 2.0 - 1.0; // x in [-1, 1)
        // Smooth deterministic "noise" so the fit has real residual scale.
        let noise = 0.5 * (k * 1.234).sin();
        let y = -3.0 * x + noise;
        csv.push_str(&format!("{x:.17e},{y:.17e}\n"));
    }
    csv
}

fn write_temp_csv(contents: &str, tag: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("gam_issue_1507_{tag}_{}.csv", std::process::id()));
    let mut file = std::fs::File::create(&path).expect("create #1507 synthetic csv");
    file.write_all(contents.as_bytes())
        .expect("write #1507 synthetic csv");
    path
}

fn saved_standard_gaussian_model(
    formula: &str,
    ds: &gam::inference::data::EncodedDataset,
) -> (FittedModel, usize) {
    let cfg = FitConfig::default();
    let result = fit_from_formula(formula, ds, &cfg).expect("fit constrained Gaussian model");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian identity fit");
    };

    let design = build_term_collection_design(ds.values.view(), &fit.resolvedspec)
        .expect("rebuild training design");
    let p = design.design.ncols();
    let frozenspec = freeze_term_collection_from_design(&fit.resolvedspec, &design)
        .expect("freeze training term collection");

    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        formula.to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::gaussian_identity(),
            link: Some(StandardLink::Identity),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "gaussian".to_string(),
    );
    payload.fit_result = Some(fit.fit.clone());
    payload.unified = Some(fit.fit);
    payload.data_schema = Some(ds.schema.clone());
    payload.resolved_termspec = Some(frozenspec);
    payload.set_training_feature_metadata(ds.headers.clone(), ds.feature_ranges());

    (FittedModel::from_payload(payload), p)
}

/// The slope is the single non-intercept coefficient (column index 1).
fn slope_draws(
    model: &FittedModel,
    ds: &gam::inference::data::EncodedDataset,
    p: usize,
) -> Vec<f64> {
    let col = ds.column_map();
    let cfg = NutsConfig {
        n_samples: 1500,
        nwarmup: 300,
        n_chains: 2,
        seed: 1507,
        ..NutsConfig::for_dimension(p)
    };
    let result = sample_saved_model(
        model,
        ds.values.view(),
        &col,
        model.training_headers.as_ref(),
        &cfg,
    )
    .expect("sample constrained saved model");
    assert_eq!(result.samples.ncols(), p);
    result.samples.column(1).to_vec()
}

#[test]
fn posterior_respects_nonnegative_coefficient_bound() {
    init_parallelism();
    let csv = negative_slope_csv(200);
    let path = write_temp_csv(&csv, "nonneg");
    let ds = load_csvwith_inferred_schema(&path).expect("load #1507 csv");
    std::fs::remove_file(&path).ok();

    let (model, p) = saved_standard_gaussian_model("y ~ nonnegative(x)", &ds);

    // Sanity: the constraint is active — the fitted slope pins to ~0.
    let fitted_slope = model.unified.as_ref().expect("unified fit").beta[1];
    assert!(
        fitted_slope.abs() < 1e-4,
        "expected an active nonnegative bound (slope≈0); got {fitted_slope}"
    );

    let draws = slope_draws(&model, &ds, p);
    let frac_neg = draws.iter().filter(|&&d| d < -1e-8).count() as f64 / draws.len() as f64;
    let min = draws.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        frac_neg < 0.01,
        "posterior of a nonnegative()-constrained coefficient must stay in [0,∞); \
         got {:.1}% of draws below 0 (min draw {min:.4})",
        frac_neg * 100.0
    );
}

#[test]
fn posterior_respects_two_sided_linear_coefficient_bounds() {
    init_parallelism();
    let csv = negative_slope_csv(200);
    let path = write_temp_csv(&csv, "linbox");
    let ds = load_csvwith_inferred_schema(&path).expect("load #1507 csv");
    std::fs::remove_file(&path).ok();

    let (model, p) = saved_standard_gaussian_model("y ~ linear(x, min=-1, max=1)", &ds);

    // True slope -3 < lower bound -1, so the box constraint is active and the
    // fitted slope pins to ~-1.
    let fitted_slope = model.unified.as_ref().expect("unified fit").beta[1];
    assert!(
        (fitted_slope - (-1.0)).abs() < 1e-4,
        "expected an active lower bound (slope≈-1); got {fitted_slope}"
    );

    let draws = slope_draws(&model, &ds, p);
    let frac_outside = draws
        .iter()
        .filter(|&&d| d < -1.0 - 1e-8 || d > 1.0 + 1e-8)
        .count() as f64
        / draws.len() as f64;
    let (min, max) = draws
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &d| {
            (lo.min(d), hi.max(d))
        });
    assert!(
        frac_outside < 0.01,
        "posterior of a linear(min=-1,max=1)-constrained coefficient must stay in [-1,1]; \
         got {:.1}% outside (range [{min:.4}, {max:.4}])",
        frac_outside * 100.0
    );
}
