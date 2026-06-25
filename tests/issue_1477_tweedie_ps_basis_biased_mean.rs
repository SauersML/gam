//! Regression test for #1477: a Tweedie GAM with the DEFAULT P-spline smooth
//! (`s(x)`, `bs='ps'`) must recover the true mean curve, not a biased one.
//!
//! BUG (gam 0.1.222, current `main`): on a Tweedie compound-Poisson-gamma response
//! with a curved log-mean, gam's default `bs='ps'` smooth ships a SYSTEMATICALLY
//! BIASED mean (EDF is sane ~5–6, so it is NOT an overfit) with a right-boundary
//! blow-up. On the SAME data, gam's OWN `bs='cr'` basis and mgcv's IDENTICAL
//! `bs='ps'` basis (matched fixed p=1.5) both recover the truth → the defect is
//! specific to gam's non-Gaussian P-spline mean path. The Gaussian PS failures in
//! this cluster are already fixed on this wheel (#1392/#1401/#1364); this shows the
//! fixes were not extended to the non-Gaussian path.
//!
//! Note: gam does not estimate the Tweedie power (it fits at a fixed p=1.5); both
//! engines use fixed p=1.5 here so the comparison is apples-to-apples on the mean.
//! This is the sharper sibling of the open #1471 Tweedie arm, which used a gentle
//! low-amplitude curve over a wide support and mgcv `tw()` (power estimated) and so
//! does not surface the boundary bias.
//!
//! Truth: mu(x) = exp(0.5 + sin(2πx)), x ~ U(0,1); Tweedie(p=1.5, phi=1), n=400.
//! METRIC (truth recovery): fitted-mean-vs-true-mean RMSE on a dense grid. Assert
//! (1) the default `ps` fit matches-or-beats mgcv's `ps` (≤15%), and (2) `ps` is no
//! worse than gam's own `cr` by more than 40% (isolating the ps path). This test
//! FAILS on current `main` and is the gate the non-Gaussian PS fix must turn green.

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, rmse, run_r};
use gam::{
    FitConfig, FitResult, StandardFitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::TAU;

const N: usize = 400;
const P_TRUE: f64 = 1.5;
const PHI: f64 = 1.0;
const SEED: u64 = 1_477_003;
const NG: usize = 120;

fn true_mu(x: f64) -> f64 {
    (0.5 + (TAU * x).sin()).exp()
}

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
    encode_recordswith_inferred_schema(headers, rows).expect("encode tweedie dataset")
}

/// gam linear predictor η = Xβ at arbitrary covariate rows (here a 1-D x grid).
fn gam_eta(fit: &StandardFitResult, width: usize, x_idx: usize, xs: &[f64]) -> Vec<f64> {
    let m = xs.len();
    let mut pts = Array2::<f64>::zeros((m, width));
    for (r, &v) in xs.iter().enumerate() {
        pts[[r, x_idx]] = v;
    }
    let d = build_term_collection_design(pts.view(), &fit.resolvedspec)
        .expect("rebuild tweedie design at eval rows");
    d.design.apply(&fit.fit.beta).to_vec()
}

/// Knuth Poisson sampler — adequate for the small-λ Tweedie DGP.
fn poisson_sample(lambda: f64, rng: &mut StdRng, unif: &Uniform<f64>) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let mut k = 0u32;
    let mut p = 1.0;
    loop {
        p *= unif.sample(rng);
        if p <= l {
            return k;
        }
        k += 1;
        if k > 10_000 {
            return k;
        }
    }
}

/// Marsaglia–Tsang gamma sampler (shape > 0) with the given scale.
fn gamma_sample(shape: f64, scale: f64, rng: &mut StdRng) -> f64 {
    let normal = Normal::new(0.0, 1.0).expect("normal");
    let unif = Uniform::new(0.0_f64, 1.0).expect("uniform");
    if shape < 1.0 {
        let u: f64 = unif.sample(rng);
        return gamma_sample(shape + 1.0, scale, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z: f64 = normal.sample(rng);
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = unif.sample(rng);
        if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
            return d * v * scale;
        }
    }
}

fn fit_mu(formula: &str, ds: &EncodedDataset, x_idx: usize, grid: &[f64]) -> (Vec<f64>, f64) {
    let cfg = FitConfig {
        family: Some("tweedie".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula(formula, ds, &cfg).unwrap_or_else(|e| panic!("gam fit {formula}: {e:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("Tweedie(log) is a scalar GLM => expected FitResult::Standard");
    };
    let edf = fit.fit.edf_total().expect("gam reports total edf");
    let width = ds.headers.len();
    let mu: Vec<f64> = gam_eta(&fit, width, x_idx, grid)
        .iter()
        .map(|e| e.exp())
        .collect();
    (mu, edf)
}

#[test]
fn gam_tweedie_ps_basis_recovers_unbiased_mean() {
    init_parallelism();

    // ---- Tweedie compound-Poisson-gamma DGP (Jørgensen), fixed p=1.5 ---------
    let mut rng = StdRng::seed_from_u64(SEED);
    let unif01 = Uniform::new(0.0_f64, 1.0).expect("uniform x");
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let xi: f64 = unif01.sample(&mut rng);
        let mu = true_mu(xi);
        let lambda = mu.powf(2.0 - P_TRUE) / (PHI * (2.0 - P_TRUE));
        let shape = (2.0 - P_TRUE) / (P_TRUE - 1.0);
        let scale = PHI * (P_TRUE - 1.0) * mu.powf(P_TRUE - 1.0);
        let n_jumps = poisson_sample(lambda, &mut rng, &unif01);
        let mut yi = 0.0;
        for _ in 0..n_jumps {
            yi += gamma_sample(shape, scale, &mut rng);
        }
        x.push(xi);
        y.push(yi);
    }
    let zeros = y.iter().filter(|&&v| v == 0.0).count();
    assert!(
        zeros > 0,
        "Tweedie 1<p<2 must be zero-inflated; got {zeros} zeros"
    );

    let ds = encode(&[("x", &x), ("y", &y)]);
    let x_idx = ds.column_map()["x"];
    let grid: Vec<f64> = (0..NG).map(|i| i as f64 / (NG as f64 - 1.0)).collect();
    let truth: Vec<f64> = grid.iter().map(|&xg| true_mu(xg)).collect();

    // ---- gam: default ps vs gam's own cr, both at k=10, family=tweedie -------
    let (gam_ps, edf_ps) = fit_mu("y ~ s(x, k=10)", &ds, x_idx, &grid);
    let (gam_cr, edf_cr) = fit_mu("y ~ s(x, bs=cr, k=10)", &ds, x_idx, &grid);

    // ---- mgcv: SAME ps basis, fixed p=1.5 (apples-to-apples) ----------------
    let r = run_r(
        &[Column::new("x", &x), Column::new("y", &y)],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            m <- gam(y ~ s(x, bs = "ps", k = 10), data = df,
                     family = Tweedie(p = 1.5, link = "log"), method = "REML")
            xg <- seq(0, 1, length.out = {ng})
            emit("mu", as.numeric(predict(m, newdata = data.frame(x = xg), type = "response")))
            emit("edf", sum(m$edf))
            "#,
            ng = NG,
        ),
    );
    let mgcv_ps = r.vector("mu");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_ps.len(), NG, "mgcv mu length mismatch");

    let err_ps = rmse(&gam_ps, &truth);
    let err_cr = rmse(&gam_cr, &truth);
    let err_mgcv = rmse(mgcv_ps, &truth);

    eprintln!(
        "tweedie ps-bias (seed {SEED}, p={P_TRUE}): n={N} \
         gam_ps_rmse={err_ps:.5} (edf {edf_ps:.2}) gam_cr_rmse={err_cr:.5} (edf {edf_cr:.2}) \
         mgcv_ps_rmse={err_mgcv:.5} (edf {mgcv_edf:.2}) \
         ps/mgcv={:.2} ps/cr={:.2}",
        err_ps / err_mgcv.max(1e-12),
        err_ps / err_cr.max(1e-12)
    );

    // (1) MATCH-OR-BEAT mgcv on the SAME ps basis: gam's default ps must not be
    //     materially worse than mgcv's ps. Pre-fix gam_ps lands ~2.9× mgcv_ps.
    assert!(
        err_ps <= 1.15 * err_mgcv,
        "Tweedie default ps smooth fails to recover the mean: gam_ps_rmse={err_ps:.5} > 1.15 * \
         mgcv_ps_rmse={err_mgcv:.5} (≈{:.1}× worse) on the IDENTICAL ps basis — biased \
         non-Gaussian P-spline mean (#1477)",
        err_ps / err_mgcv.max(1e-12)
    );

    // (2) ISOLATION: gam's default ps must not be far worse than gam's OWN cr on
    //     the same data — proves the bias is the ps path, not the problem. Pre-fix
    //     ps/cr ≈ 2.4×.
    assert!(
        err_ps <= 1.40 * err_cr,
        "Tweedie ps is far worse than gam's own cr on identical data: \
         gam_ps_rmse={err_ps:.5} > 1.40 * gam_cr_rmse={err_cr:.5} (≈{:.1}× worse) — \
         the bias is specific to the default P-spline basis under a non-Gaussian family (#1477)",
        err_ps / err_cr.max(1e-12)
    );
}
