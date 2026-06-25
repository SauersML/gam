//! Regression for issue #1509: posterior draws from a
//! `s(x, shape='monotone_increasing')` smooth must stay in the monotone cone.
//!
//! A shape-constrained smooth restricts its spline coefficients to a cone
//! enforced at fit time as linear inequality rows
//! (`src/terms/smooth/shape_constraints.rs` — assembled into
//! `LinearInequalityConstraints` and merged into the global `A·β ≥ b`). The
//! fitted point estimate is therefore monotone, and the parameter space of the
//! model IS the monotone cone, so every posterior draw must live there too.
//! Before the fix, `model.sample()` drew a plain Gaussian on the raw spline
//! coefficients with no awareness of the shape inequalities, so ~30% of drawn
//! curves decreased while the sampler still reported `rhat ≈ 1` /
//! `converged = true`. The sampler now routes shape constraints (and box
//! constraints, #1507) through a truncated-Gaussian Gibbs sampler whose every
//! draw satisfies `A·β ≥ b` by construction.
//!
//! The constraint rows `A` are exactly the monotonicity differences of the
//! design evaluated at sorted unique covariate locations: `A·β ≥ 0` is the
//! statement "the curve is non-decreasing across the grid". This test fits a
//! clean monotone-increasing signal, confirms the fitted curve is in the cone,
//! samples the posterior, and asserts that essentially every drawn coefficient
//! vector satisfies the same monotone-cone inequalities (a tiny tolerance
//! absorbs round-off) — the faithful, coordinate-exact form of "every drawn
//! curve is monotone".

use gam::hmc::NutsConfig;
use gam::inference::model::{
    FittedFamily, FittedModel, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind,
};
use gam::sample::sample_saved_model;
use gam::smooth::{
    TermCollectionDesign, build_term_collection_design, freeze_term_collection_from_design,
};
use gam::types::{LikelihoodSpec, StandardLink};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array1;
use std::io::Write;

/// Deterministic monotone-NON-DECREASING signal with a long flat plateau, so the
/// monotone constraint *binds* (many cone faces `A·β = 0` are active at the
/// fitted mode). On an active face the unconstrained Gaussian posterior places
/// ~half its mass on the forbidden side, which is exactly the defect: the signal
/// rises on `[0, 0.4]`, is flat on `[0.4, 1.0]`, with enough noise that the
/// posterior is wide relative to the cone.
fn monotone_csv(n: usize) -> String {
    let mut csv = String::from("x,y\n");
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0); // x in [0, 1]
        // Pseudo-random but deterministic noise with non-trivial scale.
        let noise = 0.30 * ((i as f64 * 12.9898).sin() * 43758.547).fract();
        let signal = if x < 0.4 { 2.5 * x } else { 1.0 };
        let y = signal + noise;
        csv.push_str(&format!("{x:.17e},{y:.17e}\n"));
    }
    csv
}

fn write_temp_csv(contents: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "gam_issue_1509_monotone_{}.csv",
        std::process::id()
    ));
    let mut file = std::fs::File::create(&path).expect("create #1509 synthetic csv");
    file.write_all(contents.as_bytes())
        .expect("write #1509 synthetic csv");
    path
}

fn saved_monotone_model(
    formula: &str,
    ds: &gam::inference::data::EncodedDataset,
) -> (FittedModel, TermCollectionDesign) {
    let cfg = FitConfig::default();
    let result = fit_from_formula(formula, ds, &cfg).expect("fit monotone smooth model");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian identity fit");
    };

    let design = build_term_collection_design(ds.values.view(), &fit.resolvedspec)
        .expect("rebuild training design");
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

    (FittedModel::from_payload(payload), design)
}

#[test]
fn posterior_draws_of_monotone_smooth_stay_in_the_cone() {
    init_parallelism();
    let csv = monotone_csv(300);
    let path = write_temp_csv(&csv);
    let ds = load_csvwith_inferred_schema(&path).expect("load #1509 csv");
    std::fs::remove_file(&path).ok();

    let (model, design) = saved_monotone_model("y ~ s(x, shape='monotone_increasing')", &ds);
    let p = design.design.ncols();

    // The monotone-cone inequality rows `A·β ≥ b` (b = 0). Present iff the smooth
    // carried a shape constraint, which is the whole point of this fit.
    let constraints = design
        .linear_constraints
        .as_ref()
        .expect("monotone smooth must assemble shape inequality constraints");
    let a = &constraints.a;
    let b = &constraints.b;
    assert_eq!(
        a.ncols(),
        p,
        "constraint columns must match coefficient count"
    );
    assert!(a.nrows() > 0, "expected at least one monotonicity row");

    // Sanity: the fitted mode is in the cone (the fit honours the constraint).
    let beta = &model.unified.as_ref().expect("unified fit").beta;
    let ax_mode = a.dot(beta);
    let worst_mode = (0..a.nrows())
        .map(|r| ax_mode[r] - b[r])
        .fold(f64::INFINITY, f64::min);
    assert!(
        worst_mode >= -1e-6,
        "fitted mode must satisfy the monotone cone; worst slack {worst_mode}"
    );

    let col = ds.column_map();
    let cfg = NutsConfig {
        n_samples: 600,
        nwarmup: 300,
        n_chains: 2,
        seed: 1509,
        ..NutsConfig::for_dimension(p)
    };
    let result = sample_saved_model(
        &model,
        ds.values.view(),
        &col,
        model.training_headers.as_ref(),
        &cfg,
    )
    .expect("sample monotone saved model");
    assert_eq!(result.samples.ncols(), p);

    // Every drawn coefficient vector must satisfy A·β ≥ 0 (the monotone cone),
    // which is exactly "the drawn curve is non-decreasing across the grid". A
    // tiny relative tolerance absorbs round-off; before the fix ~30% of draws
    // violated this by a wide margin.
    let n_draws = result.samples.nrows();
    let tol = 1e-6;
    let mut violating = 0usize;
    let mut worst_drop = 0.0_f64;
    for k in 0..n_draws {
        let beta_k: Array1<f64> = result.samples.row(k).to_owned();
        let axk = a.dot(&beta_k);
        let mut row_violated = false;
        for r in 0..a.nrows() {
            let slack = axk[r] - b[r];
            if slack < -tol {
                row_violated = true;
                worst_drop = worst_drop.min(slack);
            }
        }
        if row_violated {
            violating += 1;
        }
    }
    let frac = violating as f64 / n_draws as f64;
    assert!(
        frac < 0.05,
        "posterior draws of a monotone_increasing smooth must stay in the monotone cone; \
         {:.1}% of {n_draws} draws violate A·β ≥ 0 (worst slack {worst_drop:.4e})",
        frac * 100.0
    );
}
