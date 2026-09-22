//! The standard REML path's lower-face probe (#1561).
//!
//! A certified outer optimum is a local minimum of the REML criterion. On
//! `y = sin(16πx) + N(0, 0.05²)`, `n = 200`, `duchon(x, k=20)` the criterion has
//! two basins: the signal lives in the basis's most-penalized directions, so
//! from the least-penalized face the criterion falls to a basin that fits it,
//! rises over a ridge, and falls again to a `λ → ∞` plateau where the smooth is
//! flat. The analytic `initial.sp` start sits on the plateau's side, so the one
//! search certified the plateau and the fit predicted a constant (truth RMSE
//! 0.7063 against the signal's own RMS 0.7071, reference-quality
//! `gam_duchon_1d_matches_mgcv_ds`). mgcv's REML stops on the same plateau; on
//! its scale the criterion there is 216.28 and in the lower basin −147.7.
//!
//! The probe reads the criterion once at the domain's least-penalized face. A
//! value below the certified optimum's proves a lower basin, a second certified
//! search runs from the face, and keep-best publishes the lower optimum. The
//! control is a smooth signal whose optimum is interior, where the face value
//! is above the optimum and nothing runs.
//!
//! The single declaration of this module in `fit_orchestration.rs` is
//! `#[cfg(test)] mod ...;`, so this file is test-only.
#![cfg(test)]

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_solve::model_types::{FaceProbe, FaceSearch, FaceValue};
use gam_terms::smooth::build_term_collection_design;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// A Gaussian `duchon(x, k=20)` fit of `truth` on `n = 200` evenly spaced `x`
/// with `N(0, noise_sd²)` noise, the probe record it published, and the truth
/// RMSE of the fitted smooth on an interior grid with the truth's own RMS there
/// (the error of a flat predictor).
fn fit_duchon(truth: fn(f64) -> f64, noise_sd: f64) -> (FaceProbe, f64, f64) {
    let n = 200usize;
    let mut rng = StdRng::seed_from_u64(123);
    let noise = Normal::new(0.0, noise_sd).expect("normal");
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let rows = x
        .iter()
        .map(|&t| {
            let y = truth(t) + noise.sample(&mut rng);
            StringRecord::from(vec![t.to_string(), y.to_string()])
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let x_idx = ds.column_map()["x"];
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let StandardFitResult {
        fit, resolvedspec, ..
    } = match fit_from_formula("y ~ duchon(x, k=20)", &ds, &cfg).expect("gam duchon fit") {
        FitResult::Standard(standard) => standard,
        _ => panic!("expected a standard fit for a Gaussian Duchon smooth"),
    };
    let probe = fit
        .artifacts
        .lower_face_probe
        .clone()
        .expect("the standard REML path's first search is probed at the lower face");

    let m = 201usize;
    let grid_x: Vec<f64> = (0..m).map(|i| 0.005 + 0.99 * i as f64 / (m as f64 - 1.0)).collect();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &t) in grid_x.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let design = build_term_collection_design(grid.view(), &resolvedspec).expect("grid design");
    let fitted = design.design.matrixvectormultiply(&fit.beta);
    let squared_error: f64 =
        grid_x.iter().zip(fitted.iter()).map(|(&t, &f)| (f - truth(t)).powi(2)).sum();
    let squared_truth: f64 = grid_x.iter().map(|&t| truth(t).powi(2)).sum();
    eprintln!(
        "[1561-LOWER-FACE] probe {probe:?} reml {:?} log_lambdas {:?} edf {:?}",
        fit.reml_score(),
        fit.log_lambdas.to_vec(),
        fit.edf_total(),
    );
    (
        probe,
        (squared_error / m as f64).sqrt(),
        (squared_truth / m as f64).sqrt(),
    )
}

fn eight_periods(x: f64) -> f64 {
    (16.0 * std::f64::consts::PI * x).sin()
}

fn one_period(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin()
}

/// The derived start certifies the `λ → ∞` plateau; the face is below it, the
/// search from the face certifies the lower basin, and that optimum is the one
/// published, with a smooth that recovers the signal rather than a constant.
#[test]
fn a_face_below_the_certified_optimum_publishes_the_lower_basin_1561() {
    let (probe, rmse, flat) = fit_duchon(eight_periods, 0.05);
    let FaceValue::Evaluated { criterion: face } = probe.face else {
        panic!("the face's inner solve refused: {:?}", probe.face);
    };
    assert!(
        face < probe.certified_criterion,
        "the fixture's first search must certify the plateau, with the face below it: face \
         {face} against the certified {}",
        probe.certified_criterion
    );
    let FaceSearch::Certified {
        criterion,
        published: displaced,
        ..
    } = probe.second_search
    else {
        panic!("the search from the face did not certify: {:?}", probe.second_search);
    };
    assert!(
        displaced && criterion < probe.certified_criterion,
        "the lower basin's optimum {criterion} must displace the plateau's {} and be published",
        probe.certified_criterion
    );
    assert!(
        rmse < flat,
        "the published smooth must recover the signal, not the plateau's constant: truth RMSE \
         {rmse} against the flat predictor's {flat}"
    );
}

/// A smooth signal has an interior optimum and no lower basin: the face is
/// above the certified optimum, no second search runs, and the fit is the first
/// search's.
#[test]
fn a_face_above_the_certified_optimum_changes_nothing_1561() {
    let (probe, _, _) = fit_duchon(one_period, 0.3);
    let FaceValue::Evaluated { criterion: face } = probe.face else {
        panic!("the face's inner solve refused: {:?}", probe.face);
    };
    assert!(
        face >= probe.certified_criterion,
        "the control's face {face} must not be below its certified optimum {}",
        probe.certified_criterion
    );
    assert_eq!(probe.second_search, FaceSearch::NotRun);
}
