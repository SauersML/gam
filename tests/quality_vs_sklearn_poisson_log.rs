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
//! covariate effects at all). Crucially we let statsmodels pick its per-smoother
//! penalty by GCV via `select_penweight()` and refit at the optimum — a bare
//! `GLMGam(...).fit()` uses the default `alpha=0` (unpenalized, overfitting) and
//! would NOT target a comparable smoothing objective to gam's REML.
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
        .map(|i| StringRecord::from(vec![x1[i].to_string(), x2[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x1, k=5) + s(x2, k=5)", &ds, &cfg).expect("gam poisson fit");
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
    // k=5) under Poisson(Log). select_penweight() picks the per-smoother penalty
    // by GCV, then we refit at that optimum, so statsmodels actually performs
    // smoothing-parameter selection (comparable to gam's REML) rather than
    // fitting unpenalized at the alpha=0 default — both engines then target the
    // same penalized Poisson(log) likelihood.
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
fam = sm.families.Poisson(link=sm.families.links.Log())

alpha0 = [1.0, 1.0]
gam = GLMGam(y, smoother=bs, alpha=alpha0, family=fam)
# GCV search over the per-smoother penalty weights, then refit at the optimum.
alpha_opt, _ = gam.select_penweight()
gam = GLMGam(y, smoother=bs, alpha=alpha_opt, family=fam)
res = gam.fit()

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
    // (a) gam's eta_hat vs the known truth, as SHAPE recovery. The truth lives
    //     on a +0.5 pedestal with only ±0.3/±0.2 sinusoidal variation, so a
    //     relative-L2 against eta_truth would be dominated by the intercept and
    //     would assert almost nothing about the smooth signal. Pearson is
    //     offset/scale-invariant and directly measures whether gam recovered the
    //     sinusoidal structure under Poisson noise.
    let corr_truth = pearson(&gam_eta, &eta_truth);
    // (b) cross-engine fitted-mean agreement on the exp scale: the inverse link
    //     and PIRLS must produce the same fitted means as statsmodels.
    let corr_mean = pearson(&gam_mean, sm_mu);
    // (c) cross-engine eta agreement. Both engines see the SAME Poisson draws,
    //     so sampling noise is shared and cancels — their fitted linear
    //     predictors should nearly coincide. This is the tight apples-to-apples
    //     check (relative_l2 here is NOT pedestal-dominated because the engines
    //     share the +0.5 offset, so the residual is pure cross-engine drift).
    let rel_eta_cross = relative_l2(&gam_eta, sm_eta);

    eprintln!(
        "poisson s(x1)+s(x2): n={N} gam_edf={gam_edf:.3} \
         pearson(eta,truth)={corr_truth:.5} pearson(mean)={corr_mean:.5} \
         rel_l2(eta,statsmodels)={rel_eta_cross:.4}"
    );

    // (a) gam must recover the smooth low-frequency truth that a k=5 cubic basis
    //     resolves exactly; >0.9 Pearson on eta vs the noise-free truth is the
    //     floor a correct Poisson(log) PIRLS clears at n=200 (Poisson noise on a
    //     small-amplitude signal caps perfect recovery, but a broken inverse
    //     link or design would collapse this well below 0.9).
    assert!(
        corr_truth > 0.9,
        "gam linear predictor fails to recover the smooth truth: pearson={corr_truth:.5}"
    );
    // (b) Both engines fit the same penalized Poisson(log) model, so their fitted
    //     means must be >0.99 Pearson-correlated.
    assert!(
        corr_mean > 0.99,
        "gam vs statsmodels fitted means disagree on the exp scale: pearson={corr_mean:.5}"
    );
    // (c) Sharing the data, the two engines' linear predictors should agree to
    //     within ~10% relative L2 (GCV vs REML penalty selection on the same
    //     k=5 bases differs only in smoothing weight, not in resolvable
    //     structure). A real inverse-link/PIRLS/design bug in gam blows past this.
    assert!(
        rel_eta_cross < 0.10,
        "gam vs statsmodels linear predictors diverge: rel_l2={rel_eta_cross:.4}"
    );
}
