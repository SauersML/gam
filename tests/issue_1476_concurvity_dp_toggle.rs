//! Regression test for #1476 (R-free): a textbook additive model
//! `y ~ s(x1) + s(x2)` (gaussian) on two MODERATELY CORRELATED covariates
//! (corr ≈ 0.90) must recover the known component functions under the DEFAULT
//! basis with `double_penalty = True` (mgcv `select = TRUE`), not over-shrink one
//! smooth's genuinely-supported component.
//!
//! This is the R-free companion to `issue_1476_concurvity_double_penalty_collapse`.
//! Rather than compare to mgcv, it uses gam-vs-itself: the SAME formula, SAME
//! data, SAME default basis, toggling ONLY `double_penalty`. mgcv's
//! `select=TRUE` vs `select=FALSE` ratio on this data is ≈1.00× (a correct
//! selection penalty leaves a supported smooth alone), so a healthy
//! `double_penalty` must recover the truth no worse than `double_penalty=False`
//! by more than a small margin. The pre-fix `double_penalty=True` default lands
//! ~2–3.6× worse than `double_penalty=False`, the live #1476 defect (the
//! catastrophic λ-rail collapse was fixed, but a residual over-shrink of the
//! supported null-space component survives on the large default basis).
//!
//! Because it needs no external reference tool, this gate runs in the autonomous
//! container that has no R.

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::TAU;

const N: usize = 400;
const RHO: f64 = 0.90;
const NOISE: f64 = 0.30;

fn encode(cols: &[(&str, &[f64])]) -> EncodedDataset {
    let n = cols[0].1.len();
    let headers: Vec<String> = cols.iter().map(|(h, _)| (*h).to_string()).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(
                cols.iter()
                    .map(|(_, c)| c[i].to_string())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode concurvity dataset")
}

/// Rank-transform to exact Uniform[0,1] marginals: o = argsort(argsort(a)).
fn rank_to_unit(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).expect("finite covariate"));
    let mut rank = vec![0usize; n];
    for (r, &i) in idx.iter().enumerate() {
        rank[i] = r;
    }
    rank.iter().map(|&r| (r as f64 + 0.5) / n as f64).collect()
}

/// Fit `y ~ s(x1) + s(x2)` (optionally with `double_penalty=False`) on the
/// supplied rows and return (rmse-vs-truth, total EDF).
fn fit_and_score(
    x1: &[f64],
    x2: &[f64],
    y: &[f64],
    mu: &[f64],
    double_penalty: bool,
) -> (f64, f64) {
    let ds = encode(&[("x1", x1), ("x2", x2), ("y", y)]);
    let col = ds.column_map();
    let (i1, i2) = (col["x1"], col["x2"]);
    let width = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = if double_penalty {
        "y ~ s(x1) + s(x2)".to_string()
    } else {
        "y ~ s(x1, double_penalty=False) + s(x2, double_penalty=False)".to_string()
    };
    let result = fit_from_formula(&formula, &ds, &cfg).expect("gam concurvity fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for s(x1)+s(x2)");
    };
    let edf = fit.fit.edf_total().expect("gam reports total edf");

    let n = x1.len();
    let mut pts = Array2::<f64>::zeros((n, width));
    for r in 0..n {
        pts[[r, i1]] = x1[r];
        pts[[r, i2]] = x2[r];
    }
    let design = build_term_collection_design(pts.view(), &fit.resolvedspec)
        .expect("rebuild concurvity design at training points");
    let fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    (rmse(&fitted, mu), edf)
}

fn make_data(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let nrm = Normal::new(0.0, 1.0).expect("normal");
    let mut g0 = Vec::with_capacity(N);
    let mut g1 = Vec::with_capacity(N);
    for _ in 0..N {
        let z0: f64 = nrm.sample(&mut rng);
        let z1: f64 = nrm.sample(&mut rng);
        g0.push(z0);
        g1.push(RHO * z0 + (1.0 - RHO * RHO).sqrt() * z1);
    }
    let x1 = rank_to_unit(&g0);
    let x2 = rank_to_unit(&g1);
    let f2: Vec<f64> = x2.iter().map(|&v| v * v).collect();
    let f2_mean = f2.iter().sum::<f64>() / N as f64;
    let mu: Vec<f64> = (0..N)
        .map(|i| (TAU * x1[i]).sin() + (f2[i] - f2_mean))
        .collect();
    let y: Vec<f64> = mu
        .iter()
        .map(|&m| m + NOISE * nrm.sample(&mut rng))
        .collect();
    (x1, x2, y, mu)
}

/// dp-toggle: across seeds, the default `double_penalty=True` fit must recover
/// the truth essentially as well as `double_penalty=False` on the SAME data and
/// basis (mgcv's select=TRUE/FALSE ratio is ≈1.00×). Pre-fix #1476 lands the
/// median dp=True/dp=False ratio at ~2.17× (worst-seed up to 5×).
#[test]
fn double_penalty_does_not_over_shrink_supported_smooth_under_concurvity() {
    init_parallelism();

    let seeds: [u64; 6] = [
        1_476_011, 1_476_012, 1_476_013, 1_476_014, 1_476_015, 1_476_016,
    ];
    let mut ratios = Vec::new();
    let mut worst = (0u64, 0.0f64);
    for &seed in &seeds {
        let (x1, x2, y, mu) = make_data(seed);
        let (err_dp, edf_dp) = fit_and_score(&x1, &x2, &y, &mu, true);
        let (err_sp, edf_sp) = fit_and_score(&x1, &x2, &y, &mu, false);
        let ratio = err_dp / err_sp.max(1e-12);
        eprintln!(
            "seed {seed}: dp=True rmse={err_dp:.5} (edf {edf_dp:.2}) | \
             dp=False rmse={err_sp:.5} (edf {edf_sp:.2}) | ratio={ratio:.2}"
        );
        ratios.push(ratio);
        if ratio > worst.1 {
            worst = (seed, ratio);
        }
    }
    let mut sorted = ratios.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0;
    eprintln!(
        "dp-toggle ratios: {ratios:?}  median={median:.2}  worst seed {} = {:.2}×",
        worst.0, worst.1
    );

    // A correct selection penalty leaves a supported smooth alone: mgcv's
    // select=TRUE / select=FALSE ratio on this data is ≈1.00×. Allow gam a
    // modest 25% median slack and a 1.6× worst-seed ceiling — the pre-fix
    // default lands median ~2.17× and worst ~5×.
    assert!(
        median <= 1.25,
        "double_penalty over-shrinks a supported smooth under concurvity: median \
         dp=True/dp=False rmse ratio = {median:.2}× (want ≤ 1.25×). The default \
         selection penalty is damaging a genuinely-supported null-space component \
         (#1476, residual after the projector + λ-rail fixes)."
    );
    assert!(
        worst.1 <= 1.6,
        "double_penalty over-shrinks a supported smooth on the worst seed {}: \
         dp=True/dp=False rmse ratio = {:.2}× (want ≤ 1.6×) (#1476).",
        worst.0,
        worst.1
    );
}
