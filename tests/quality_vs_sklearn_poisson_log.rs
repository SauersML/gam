//! End-to-end quality: gam's Poisson(log) GLM with smooth terms must recover the
//! same linear predictor as a mature count-data engine on identical data.
//!
//! Poisson(log) with smooth covariates is the quintessential count-data GAM. We
//! generate a fixed-seed synthetic dataset where the true linear predictor is
//!
//!     eta = 0.5 + 0.3*sin(x1*pi/5) + 0.2*cos(x2*pi/5),   y ~ Poisson(exp(eta)),
//!
//! and fit `y ~ s(x1, k=5) + s(x2, k=5)` with gam (REML smoothing-parameter
//! selection, log link). The mature reference is **statsmodels**
//! `GLMGam(family=Poisson(link=Log()))` with the SAME penalized B-spline smooths
//! fed the IDENTICAL data — the standard Python implementation of penalized
//! additive Poisson regression. statsmodels exposes both the additive smooth
//! structure and the canonical log link, so it is a far better reference than a
//! plain linear `PoissonRegressor` (which cannot represent the sinusoidal
//! covariate effects at all).
//!
//! Both engines fit the same penalized Poisson log-likelihood, so they must
//! recover essentially the same fitted function. We assert:
//!   1. gam's fitted linear predictor `eta_hat` tracks the known truth on the
//!      data grid (relative L2 over `eta`), and
//!   2. gam's fitted mean `exp(eta_hat)` and statsmodels' fitted mean are
//!      near-perfectly correlated (Pearson on the exp-scale).
//! A genuine divergence here is a real bug in gam's inverse-link / PIRLS logic.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};

const N: usize = 200;
const SEED: u64 = 42;

fn truth_eta(x1: f64, x2: f64) -> f64 {
    let pi = std::f64::consts::PI;
    0.5 + 0.3 * (x1 * pi / 5.0).sin() + 0.2 * (x2 * pi / 5.0).cos()
}

#[test]
fn gam_poisson_log_matches_statsmodels_glm() {
    init_parallelism();

    // ---- synthetic count data (identical bytes feed gam and statsmodels) ---
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 10.0).expect("uniform 0..10");
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut eta_truth = Vec::with_capacity(N);
    for _ in 0..N {
        let a = ux.sample(&mut rng);
        let b = ux.sample(&mut rng);
        let eta = truth_eta(a, b);
        let pois = Poisson::new(eta.exp()).expect("poisson mean > 0");
        let count: f64 = pois.sample(&mut rng);
        x1.push(a);
        x2.push(b);
        eta_truth.push(eta);
        y.push(count);
    }

    // ---- fit with gam: y ~ s(x1, k=5) + s(x2, k=5), Poisson(log), REML ------
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| {
            StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x1, k=5) + s(x2, k=5)", &ds, &cfg).expect("gam poisson fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the Poisson(log) family");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Rebuild the design at the observed covariates: for a log link,
    // design*beta is the linear predictor eta_hat; exp(eta_hat) is the fitted
    // mean. (build_term_collection_design freezes the same basis/penalty.)
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x1_idx]] = x1[i];
        grid[[i, x2_idx]] = x2[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_mean: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model with statsmodels (the mature reference) --------
    // GLMGam with two penalized cubic B-spline smooths (df=5 each, matching
    // k=5) under Poisson(Log). statsmodels selects the smoothing penalty via its
    // own criterion; both engines target the same penalized Poisson likelihood.
    let r = run_python(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("y", &y),
        ],
        r#"
import numpy as np
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines

x1 = np.asarray(df["x1"], dtype=float)
x2 = np.asarray(df["x2"], dtype=float)
y  = np.asarray(df["y"],  dtype=float)

X = np.column_stack([x1, x2])
# Cubic B-spline smooth basis, 5 basis functions per covariate (matches k=5).
bs = BSplines(X, df=[5, 5], degree=[3, 3])
model = GLMGam(y, smoother=bs, family=sm.families.Poisson(link=sm.families.links.Log()))
res = model.fit()

eta_hat = np.asarray(res.predict(which="linear"), dtype=float)
mu_hat  = np.asarray(res.predict(), dtype=float)
emit("eta", eta_hat)
emit("mu", mu_hat)
"#,
    );
    let sm_eta = r.vector("eta");
    let sm_mu = r.vector("mu");
    assert_eq!(sm_eta.len(), N, "statsmodels eta length mismatch");
    assert_eq!(sm_mu.len(), N, "statsmodels mu length mismatch");

    // ---- compare -----------------------------------------------------------
    // (a) gam's eta_hat vs the known truth: both engines recover the smooth
    //     linear predictor, so gam should track eta_truth tightly.
    let rel_truth = relative_l2(&gam_eta, &eta_truth);
    // (b) cross-engine fitted-mean agreement on the exp scale: the inverse link
    //     and PIRLS must produce the same fitted means as statsmodels.
    let corr_mean = pearson(&gam_mean, sm_mu);
    // Supplementary: cross-engine eta agreement.
    let rel_eta_cross = relative_l2(&gam_eta, sm_eta);

    eprintln!(
        "poisson s(x1)+s(x2): n={N} gam_edf={gam_edf:.3} \
         rel_l2(eta,truth)={rel_truth:.4} pearson(mean)={corr_mean:.5} \
         rel_l2(eta,statsmodels)={rel_eta_cross:.4}"
    );

    // Both gam and statsmodels fit the same penalized Poisson(log) additive
    // model, and the truth is a smooth low-frequency signal these k=5 bases
    // resolve exactly, so eta_hat must track eta_truth to within 5% relative L2
    // and the two engines' fitted means must be >0.99 Pearson-correlated.
    assert!(
        rel_truth < 0.05,
        "gam linear predictor diverges from the truth: rel_l2={rel_truth:.4}"
    );
    assert!(
        corr_mean > 0.99,
        "gam vs statsmodels fitted means disagree on the exp scale: pearson={corr_mean:.5}"
    );
}
