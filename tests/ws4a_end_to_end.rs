//! WS4a end-to-end integration test: staged outer-score subsampling
//! converges to the same optimum as a full-data fit.
//!
//! Math claim under test
//! ---------------------
//! When `BlockwiseFitOptions::outer_score_subsample` carries Horvitz-Thompson
//! per-row weights w_i = N_h / k_h (here, uniform sampling with w_i = N / m),
//! the subsampled outer score is an UNBIASED estimator of the full-data outer
//! score: E[Σ_{i in mask} w_i · s_i] = Σ_i s_i. The outer optimizer
//! (BFGS/ARC) is consistent under unbiased noisy gradients — at the true
//! optimum E[∇] = 0, and the iterate converges modulo single-sample noise.
//!
//! Test design
//! -----------
//! Build a Gaussian-LS fixture (mean smooth + constant noise), materialize
//! the formula twice, and on the second pass install a uniform 50% subsample
//! directly into the family's `BlockwiseFitOptions::outer_score_subsample`
//! slot. Compare:
//!   - final REML score      (relative match within 1e-2)
//!   - per-block edf         (absolute match within 0.5)
//!   - β coefficient vector  (relative-with-floor match within 5e-2)
//!
//! Tolerances are deliberately LOOSE because a single-sample subsampled fit
//! only converges in expectation; tighter tolerances would flake on the
//! finite-sample noise. A relative REML deviation above 5% would indicate a
//! bug in the HT reweighting, not a math limit.
//!
//! API path
//! --------
//! 1. `materialize(formula, &data, &cfg)` returns a `MaterializedModel`
//!    whose `request: FitRequest` carries a pub `options: BlockwiseFitOptions`
//!    on the `GaussianLocationScale` variant.
//! 2. We pattern-match-and-mutate that field to install an
//!    `OuterScoreSubsample::new(mask, n_full, seed)`, which sets the uniform
//!    HT weight w = n_full / m on every retained row.
//! 3. `fit_model(request)` returns `FitResult::GaussianLocationScale(fit)`,
//!    from which we read `fit.fit.reml_score`, `fit.fit.beta`, and
//!    `fit.fit.inference.unwrap().edf_by_block`.

use csv::StringRecord;
use gam::custom_family::BlockwiseFitOptions;
use gam::families::marginal_slope_shared::OuterScoreSubsample;
use gam::{
    FitConfig, FitRequest, FitResult, encode_recordswith_inferred_schema, fit_model,
    init_parallelism, materialize,
};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::sync::Arc;

const SEED: u64 = 0x_54A_E2E0_2026_5EEDu64;

fn truth(x: f64) -> f64 {
    // Smooth nonlinearity with enough structure to require a real fit.
    (2.0 * std::f64::consts::PI * x).sin() + 0.4 * (3.0 * x).cos()
}

fn make_data(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let xs: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|t| truth(*t) + noise.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = xs
        .iter()
        .zip(ys.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Materialize a Gaussian-LS (mean smooth + intercept-only log-sigma) fit
/// request. Returns the request so the caller can mutate
/// `BlockwiseFitOptions::outer_score_subsample` before calling `fit_model`.
fn materialize_gls<'a>(data: &'a gam::data::EncodedDataset) -> gam::MaterializedModel<'a> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1".to_string()),
        ..FitConfig::default()
    };
    materialize("y ~ s(x, k=10)", data, &cfg).expect("materialize Gaussian-LS")
}

struct FitSummary {
    reml: f64,
    edf_by_block: Vec<f64>,
    beta: Vec<f64>,
}

fn run_fit(req: FitRequest<'_>) -> FitSummary {
    let result = fit_model(req).expect("fit_model converges");
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected GaussianLocationScale fit result");
    };
    // `inference` is optional on the public path; both fits in this test
    // come from the same materializer so they're either both Some or both
    // None. When both are None we still validate REML + β and skip edf.
    let edf_by_block = fit
        .fit
        .fit
        .inference
        .as_ref()
        .map(|inf| inf.edf_by_block.clone())
        .unwrap_or_default();
    FitSummary {
        reml: fit.fit.fit.reml_score,
        edf_by_block,
        beta: fit.fit.fit.beta.to_vec(),
    }
}

/// Build a uniform 50% subsample over `n_full` rows with deterministic seed.
/// Uses `OuterScoreSubsample::new`, which assigns w_i = n_full / m to every
/// retained row — the HT inverse-inclusion weight under uniform sampling, so
/// Σ_{i in mask} w_i · s_i is unbiased for Σ_i s_i.
fn uniform_half_subsample(n_full: usize, seed: u64) -> OuterScoreSubsample {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n_full).collect();
    // Fisher-Yates partial shuffle: pick first n_full / 2.
    let m = n_full / 2;
    for i in 0..m {
        let j = i + rng.random_range(0..(n_full - i));
        indices.swap(i, j);
    }
    let mut mask: Vec<usize> = indices[..m].to_vec();
    mask.sort_unstable();
    OuterScoreSubsample::new(mask, n_full, seed)
}

#[test]
fn ws4a_subsampled_fit_matches_full_data_fit() {
    init_parallelism();

    // n=2000, p=20 (k=10 mean smooth + 1 intercept on log-sigma + intercept on mean).
    // Single noisy realisation, deterministic seed.
    let n = 2000usize;
    let sigma = 0.3;
    let data = make_data(n, sigma, SEED);

    // --- Pass A: full-data fit -------------------------------------------
    let mat_full = materialize_gls(&data);
    let full = run_fit(mat_full.request);

    // --- Pass B: 50% uniform subsample on the outer score ---------------
    let mut mat_sub = materialize_gls(&data);
    let sub = uniform_half_subsample(n, SEED.wrapping_add(0x9E37_79B9));
    if let FitRequest::GaussianLocationScale(ref mut req) = mat_sub.request {
        // Install the per-row HT-weighted subsample on the canonical slot.
        // Confirms the field is `pub` on BlockwiseFitOptions and on the
        // GaussianLocationScale variant of FitRequest.
        let opts: &mut BlockwiseFitOptions = &mut req.options;
        opts.outer_score_subsample = Some(Arc::new(sub));
    } else {
        panic!("expected GaussianLocationScale materialization");
    }
    let subbed = run_fit(mat_sub.request);

    eprintln!(
        "[ws4a-e2e] reml full={:.6e} sub={:.6e} (rel diff {:.3e})",
        full.reml,
        subbed.reml,
        (full.reml - subbed.reml).abs() / full.reml.abs().max(1.0),
    );
    eprintln!(
        "[ws4a-e2e] edf full={:?} sub={:?}",
        full.edf_by_block, subbed.edf_by_block,
    );

    // --- Assertions ------------------------------------------------------
    // 1. Final REML: relative match within 1%. The subsampled outer score
    //    is unbiased, so the converged objective tracks the full-data one
    //    up to single-sample noise. >5% relative deviation here would
    //    indicate a HT-weighting bug, not a finite-sample limit.
    let rel_reml = (full.reml - subbed.reml).abs() / full.reml.abs().max(1.0);
    assert!(
        rel_reml < 1.0e-2,
        "REML mismatch: full={:.6e}, sub={:.6e}, rel={:.3e}",
        full.reml,
        subbed.reml,
        rel_reml,
    );

    // 2. Per-block effective df: absolute match within 0.5.
    assert_eq!(
        full.edf_by_block.len(),
        subbed.edf_by_block.len(),
        "edf block count must match",
    );
    for (k, (ef, es)) in full
        .edf_by_block
        .iter()
        .zip(subbed.edf_by_block.iter())
        .enumerate()
    {
        assert!(
            (ef - es).abs() < 0.5,
            "edf mismatch on block {k}: full={ef:.3}, sub={es:.3}",
        );
    }

    // 3. β coefficient vector: relative L2 match within 5% (with +1 floor).
    assert_eq!(
        full.beta.len(),
        subbed.beta.len(),
        "β dimensions must match",
    );
    let mut num = 0.0;
    let mut den_sq = 0.0;
    for (a, b) in full.beta.iter().zip(subbed.beta.iter()) {
        let d = a - b;
        num += d * d;
        den_sq += a * a;
    }
    let rel_beta = num.sqrt() / (1.0 + den_sq.sqrt());
    eprintln!("[ws4a-e2e] ‖β_full - β_sub‖ / (1 + ‖β_full‖) = {rel_beta:.3e}");
    assert!(
        rel_beta < 5.0e-2,
        "β mismatch: rel L2 {rel_beta:.3e} > 5e-2",
    );
}
