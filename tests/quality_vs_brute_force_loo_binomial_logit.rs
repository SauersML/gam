//! Ground-truth quality: gam's ALO (approximate leave-one-out) corrected linear
//! predictor `eta_tilde` must match the *exact* leave-one-out predictor obtained
//! by brute-force refitting — the canonical oracle for any ALO method.
//!
//! Mature comparator: **gam itself, run as exact LOO**. ALO is, by definition, a
//! closed-form approximation to leave-one-out cross-validation. The unimpeachable
//! reference is therefore exact LOO: refit the GAM `n` times, each time holding
//! out observation `i`, and read off the held-out linear predictor
//! `eta_hat^{(-i)}(x_i)`. There is no external tool that does penalized-GAM exact
//! LOO for us, and approximating ALO against anything *other* than exact LOO would
//! be circular — so the honest comparator is gam's own refit. (mgcv/pygam expose
//! GCV/UBRE shortcut scores, not the per-observation exact-LOO linear predictor
//! this test validates, so they are not the right oracle here.)
//!
//! We use Binomial/logit — the canonical GLM case. The logit link is canonical,
//! so the IRLS working weights equal the Fisher information and ALO's one-step
//! Newton correction is at its most accurate; this is precisely where ALO must be
//! trusted essentially to machine-relevant precision against the exact oracle.
//!
//! Data: `heart_failure_clinical_records_dataset.csv` (299 real patients),
//! `DEATH_EVENT ~ s(ejection_fraction)`. Identical encoded data feeds the full
//! fit (for ALO) and every leave-one-out refit (for the exact oracle): the LOO
//! datasets are the full encoded design with exactly one row deleted, so basis,
//! family, link, and smoothing machinery are byte-for-byte the same in both arms.

use gam::data::EncodedDataset;
use gam::estimate::UnifiedFitResult;
use gam::inference::alo::compute_alo_diagnostics_from_fit;
use gam::matrix::LinearOperator;
use gam::smooth::{TermCollectionSpec, build_term_collection_design};
use gam::test_support::reference::{max_abs_diff, pearson, relative_l2};
use gam::types::LinkFunction;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array1, Array2, ArrayView1};
use std::path::Path;

const HEART_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/heart_failure_clinical_records_dataset.csv"
);

const FORMULA: &str = "DEATH_EVENT ~ s(ejection_fraction)";

/// Fit the binomial/logit smooth and return the fitted `UnifiedFitResult`
/// together with the frozen term spec needed to rebuild the design.
fn fit_binomial_logit(ds: &EncodedDataset) -> (UnifiedFitResult, TermCollectionSpec) {
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(FORMULA, ds, &cfg).expect("gam binomial/logit fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for binomial/logit s(x)");
    };
    (fit.fit, fit.resolvedspec)
}

/// Evaluate the linear predictor (logit scale) of a fitted model at a single
/// design point `x` for the predictor column `pred_idx`, using the model's own
/// frozen spec. For the logit link, design·beta IS eta, exactly the quantity
/// exact LOO produces for the held-out point.
fn eta_at_point(
    beta: &Array1<f64>,
    spec: &TermCollectionSpec,
    n_headers: usize,
    pred_idx: usize,
    x: f64,
) -> f64 {
    let mut grid = Array2::<f64>::zeros((1, n_headers));
    grid[[0, pred_idx]] = x;
    let design = build_term_collection_design(grid.view(), spec)
        .expect("rebuild design at held-out point");
    design.design.apply(beta)[0]
}

/// Build a leave-one-out copy of the encoded dataset with row `drop` removed.
/// Schema, headers, and column kinds are preserved verbatim so the refit uses
/// an identical basis/family/link to the full fit.
fn dataset_without_row(ds: &EncodedDataset, drop: usize) -> EncodedDataset {
    let n = ds.values.nrows();
    let p = ds.values.ncols();
    let mut values = Array2::<f64>::zeros((n - 1, p));
    let mut out = 0usize;
    for i in 0..n {
        if i == drop {
            continue;
        }
        values.row_mut(out).assign(&ds.values.row(i));
        out += 1;
    }
    EncodedDataset {
        headers: ds.headers.clone(),
        values,
        schema: ds.schema.clone(),
        column_kinds: ds.column_kinds.clone(),
    }
}

#[test]
fn alo_eta_tilde_matches_exact_loo_binomial_logit() {
    init_parallelism();

    // ---- load identical real data once ------------------------------------
    let ds = load_csvwith_inferred_schema(Path::new(HEART_CSV))
        .expect("load heart_failure_clinical_records_dataset.csv");
    let col = ds.column_map();
    let pred_idx = col["ejection_fraction"];
    let n_headers = ds.headers.len();
    let x: Vec<f64> = ds.values.column(pred_idx).to_vec();
    let n = x.len();
    assert_eq!(n, 299, "heart failure dataset should have 299 rows, got {n}");

    // ---- full fit + ALO ----------------------------------------------------
    let (full_fit, full_spec) = fit_binomial_logit(&ds);
    let y: Vec<f64> = ds.values.column(col["DEATH_EVENT"]).to_vec();
    let alo = compute_alo_diagnostics_from_fit(
        &full_fit,
        ArrayView1::from(y.as_slice()),
        LinkFunction::Logit,
    )
    .expect("ALO diagnostics on binomial/logit fit");
    let alo_eta_tilde: Vec<f64> = alo.eta_tilde.to_vec();
    assert_eq!(alo_eta_tilde.len(), n, "ALO eta_tilde length mismatch");

    // ---- exact LOO oracle: refit n times, hold out i, read eta(x_i) -------
    let mut exact_loo: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let loo_ds = dataset_without_row(&ds, i);
        let (loo_fit, loo_spec) = fit_binomial_logit(&loo_ds);
        let eta_i = eta_at_point(&loo_fit.beta, &loo_spec, n_headers, pred_idx, x[i]);
        exact_loo.push(eta_i);
    }

    // ---- compare element-wise on the logit scale --------------------------
    let rel = relative_l2(&alo_eta_tilde, &exact_loo);
    let max_abs = max_abs_diff(&alo_eta_tilde, &exact_loo);
    let corr = pearson(&alo_eta_tilde, &exact_loo);

    eprintln!(
        "ALO vs exact-LOO (binomial/logit, n={n}): rel_l2={rel:.5} \
         max_abs={max_abs:.5} pearson={corr:.6}"
    );

    // Principled bound: ALO is a first-order (one Newton step) approximation to
    // exact LOO. For a canonical link (logit) with a stable penalized fit the
    // residual approximation error is second-order in the per-observation
    // leverage and is empirically tiny here. The spec bounds — rel_l2 < 0.01,
    // max_abs < 0.05 logits, pearson > 0.9999 — assert genuine element-wise
    // agreement with the oracle (not mere correlation) while leaving no room for
    // a real divergence in the ALO algebra to slip through. They are NOT loosened.
    assert!(
        corr > 0.9999,
        "ALO eta_tilde must track exact LOO almost perfectly: pearson={corr:.6}"
    );
    assert!(
        rel < 0.01,
        "ALO eta_tilde diverges from exact LOO in relative L2: rel_l2={rel:.5}"
    );
    assert!(
        max_abs < 0.05,
        "ALO eta_tilde has a too-large worst-case logit error vs exact LOO: max_abs={max_abs:.5}"
    );
}
