//! Regression for #500: `s(x, bc=clamped)` must not abort the whole REML
//! fit on a zero-offset sine.
//!
//! The clamped endpoint-derivative equality is split into two opposing
//! inequalities (`row·β ≥ 0` and `−row·β ≥ 0`), so the constraint KKT
//! primal-feasibility residual is exactly `|row·β|` — the magnitude of the
//! smooth's first derivative at the endpoint. At the heavily-oversmoothed
//! continuation pre-warm start (ρ₀ = ρ* + offset, λ₀ ≫ λ*), the penalized
//! saddle-point system `[H+λS, Aᵀ; A, 0]` is ill-conditioned by λ₀, so the
//! linear solve cannot drive that residual below the 1e-7 primal tolerance;
//! it floors at ~λ·ε ≈ 1e-5. That is a numerical artifact of the
//! oversmoothing offset, NOT a real infeasibility — the constant function
//! satisfies a zero-slope endpoint constraint exactly.
//!
//! Before the fix every candidate seed was disqualified by this pre-warm
//! refusal ("no candidate seeds passed outer startup validation"), even
//! though the seed eval at ρ* (moderate λ) solves fine. The same clamped
//! model already fits on `sin(2πx) + 0.3` (see `bc_fit_quality_sanity.rs`);
//! a constant vertical offset cannot make a zero-first-derivative endpoint
//! constraint genuinely infeasible.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn truth(x: f64) -> f64 {
    // Zero vertical offset — the only difference from the passing
    // `bc_fit_quality_sanity.rs` dataset, and the trigger for #500.
    (2.0 * std::f64::consts::PI * x).sin()
}

fn make_data(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|t| truth(*t) + noise.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_and_probe(formula: &str, data: &gam::data::EncodedDataset, eval_xs: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg)
        .unwrap_or_else(|e| panic!("fit failed for `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit for `{formula}`")
    };
    let n = eval_xs.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &x) in eval_xs.iter().enumerate() {
        m[[i, 0]] = x;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("predict design failed for `{formula}`: {e:?}"));
    design.design.apply(&fit.fit.beta).to_vec()
}

fn rmse(pred: &[f64], eval_xs: &[f64]) -> f64 {
    let mut sumsq = 0.0_f64;
    for (p, &x) in pred.iter().zip(eval_xs.iter()) {
        let d = p - truth(x);
        sumsq += d * d;
    }
    (sumsq / pred.len() as f64).sqrt()
}

#[test]
fn clamped_zero_offset_sine_fits_and_tracks_interior() {
    init_parallelism();
    let data = make_data(300, 0.1, 7);
    // Interior probe grid (avoid the boundary where the clamp lives).
    let eval_xs: Vec<f64> = (0..101).map(|i| 0.10 + 0.80 * (i as f64) / 100.0).collect();

    // In-test reference: the unconstrained fit always succeeds on this data.
    let pred_free = fit_and_probe("y ~ s(x, k=12)", &data, &eval_xs);
    let rmse_free = rmse(&pred_free, &eval_xs);

    // The clamped fit on the SAME data must also succeed (#500). Before the
    // fix this panicked: "no candidate seeds passed outer startup validation".
    let pred_clamped = fit_and_probe("y ~ s(x, k=12, bc=clamped)", &data, &eval_xs);
    let rmse_clamped = rmse(&pred_clamped, &eval_xs);

    eprintln!("[bc-500] free={rmse_free:.4} clamped={rmse_clamped:.4}");

    // A successful-but-degenerate (≈constant) fit would also "not panic".
    // Guard against that: the clamped fit must track the interior signal,
    // within a small multiple of the unconstrained fit's RMSE.
    let budget = (5.0 * rmse_free).max(0.10);
    assert!(
        rmse_clamped <= budget,
        "clamped interior RMSE {rmse_clamped:.4} exceeds budget {budget:.4} \
         (free {rmse_free:.4}) — fit returned but did not recover the signal"
    );
}
