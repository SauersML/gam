//! Regression test for #1476: a textbook additive model `y ~ s(x1) + s(x2)`
//! (gaussian) on two MODERATELY CORRELATED covariates (corr ≈ 0.90) must recover
//! the known component functions, not collapse one smooth.
//!
//! BUG (gam 0.1.222, current `main`): under moderate concurvity at the DEFAULT
//! basis, the double-penalty nullspace selection mis-allocates — REML drives one
//! smooth's nullspace ridge to λ ≈ 1e13, flattening it to ~zero EDF, while the
//! other inflates toward the full basis. The fit-to-truth RMSE lands ~2.8× worse
//! than mgcv, whose identical `s(x1)+s(x2)` formula splits the EDF sensibly and
//! recovers both components. On the SAME failing data, forcing `bs='cr'` OR
//! `double_penalty=False` restores recovery to mgcv parity → the defect is the
//! double-penalty nullspace, not basis size.
//!
//! This is the same mechanism as the CLOSED single-smooth issues #1371 (nullspace
//! ridge λ→∞ flat-collapse) and #1266 (nullspace EDF mis-shrinkage); their fixes
//! were single-smooth/linear and do not cover the concurvity-triggered case, which
//! is still live here. The open #1471 concurvity arm misses it (it pins k=10, uses
//! standard-normal support, and asserts only the additive-SUM R²).
//!
//! METRIC (truth recovery): `mu = f1(x1) + (f2(x2) - mean)` is the known mean; fit
//! gam and mgcv on byte-identical rows and compare fitted-vs-true-mean RMSE. We
//! assert gam recovers the truth and is no worse than mgcv by more than 25%. This
//! test FAILS on current `main` and is the gate the nullspace fix must turn green.

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, rmse, run_r};
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
const SEED: u64 = 1_476_011; // a moderate-concurvity draw that triggers the collapse

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

#[test]
fn gam_concurvity_recovers_both_smooths_not_nullspace_collapse() {
    init_parallelism();

    // ---- correlated design (Gaussian copula, corr ≈ RHO) on Uniform[0,1] -----
    let mut rng = StdRng::seed_from_u64(SEED);
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

    // truth: f1 = sin(2π x1), f2 = x2² (centered); mu = f1 + f2c. A low-order
    // quadratic is nullspace-heavy on [0,1] — exactly what the ridge over-shrinks.
    let f2: Vec<f64> = x2.iter().map(|&v| v * v).collect();
    let f2_mean = f2.iter().sum::<f64>() / N as f64;
    let mu: Vec<f64> = (0..N)
        .map(|i| (TAU * x1[i]).sin() + (f2[i] - f2_mean))
        .collect();
    let y: Vec<f64> = mu
        .iter()
        .map(|&m| m + NOISE * nrm.sample(&mut rng))
        .collect();

    // ---- fit gam: y ~ s(x1) + s(x2), gaussian/identity, REML -----------------
    let ds = encode(&[("x1", &x1), ("x2", &x2), ("y", &y)]);
    let col = ds.column_map();
    let (i1, i2) = (col["x1"], col["x2"]);
    let width = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x1) + s(x2)", &ds, &cfg).expect("gam concurvity fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for s(x1)+s(x2)");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted mean at the training points (identity link).
    let mut pts = Array2::<f64>::zeros((N, width));
    for r in 0..N {
        pts[[r, i1]] = x1[r];
        pts[[r, i2]] = x2[r];
    }
    let design = build_term_collection_design(pts.view(), &fit.resolvedspec)
        .expect("rebuild concurvity design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv on the identical rows (baseline) --------
    let r = run_r(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x1) + s(x2), data = df, family = gaussian(), method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        st <- summary(m)$s.table
        emit("edf1", as.numeric(st["s(x1)", "edf"]))
        emit("edf2", as.numeric(st["s(x2)", "edf"]))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    let (mgcv_edf1, mgcv_edf2) = (r.scalar("edf1"), r.scalar("edf2"));
    assert_eq!(mgcv_fitted.len(), N, "mgcv fitted length mismatch");

    // ---- recovery error vs TRUE mean -----------------------------------------
    let gam_err = rmse(&gam_fitted, &mu);
    let mgcv_err = rmse(mgcv_fitted, &mu);

    eprintln!(
        "concurvity recovery (seed {SEED}, corr≈{RHO}): n={N} \
         gam_rmse_vs_truth={gam_err:.5} mgcv_rmse_vs_truth={mgcv_err:.5} ratio={:.2} \
         gam_edf_total={gam_edf:.2} mgcv_edf[s(x1)={mgcv_edf1:.2}, s(x2)={mgcv_edf2:.2}, tot={mgcv_edf:.2}]",
        gam_err / mgcv_err.max(1e-12)
    );

    // PRIMARY: gam recovers the additive truth under moderate concurvity. mgcv
    // lands ~0.05 here; the pre-fix gam collapses one smooth and lands ~0.13–0.16.
    assert!(
        gam_err < 0.09,
        "s(x1)+s(x2) fails to recover the truth under concurvity (corr≈{RHO}): \
         rmse_vs_truth={gam_err:.5} (mgcv {mgcv_err:.5}); a correlated smooth is \
         collapsed by the double-penalty nullspace ridge (#1476)"
    );

    // MATCH-OR-BEAT: gam no worse than mgcv by more than 25% on truth recovery.
    assert!(
        gam_err <= 1.25 * mgcv_err,
        "s(x1)+s(x2) under concurvity: gam is far less accurate than mgcv at \
         recovering the truth: gam_rmse={gam_err:.5} > 1.25 * mgcv_rmse={mgcv_err:.5} \
         (≈{:.1}× worse) — nullspace flat-collapse of a correlated smooth (#1476; \
         same class as closed #1371/#1266, still live under concurvity)",
        gam_err / mgcv_err.max(1e-12)
    );
}
