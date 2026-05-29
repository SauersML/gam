//! End-to-end quality: gam's Binomial(probit) GLM must match statsmodels —
//! the standard reference GLM implementation — on the linear predictor.
//!
//! This is a *cross-family* check: the logit link is gam's default binomial
//! link, so a separate probit test verifies that gam's link-family dispatch
//! (`family="binomial-probit"` -> `InverseLink::Standard(StandardLink::Probit)`)
//! is wired correctly for a non-logit binomial. Probit is the second-most
//! common binomial link (the Gaussian-CDF inverse, heavier-tailed than logit),
//! and statsmodels `GLM(family=Binomial(link=Probit()))` is the canonical
//! reference.
//!
//! We fit the *same* synthetic data with gam (`y ~ s(x1, k=4)`, probit) and
//! statsmodels (a probit GLM with a matched cubic-spline basis on x1), then
//! compare the fitted linear predictor eta = design*beta on a held-out
//! evaluation grid. Because both engines fit the same profile-likelihood model
//! (penalized IRLS / PIRLS for gam, IRLS for statsmodels on a comparable
//! spline basis), their linear predictors must coincide up to small
//! penalty/basis differences.
//!
//! Bound: max_abs_diff on eta, divided by the spread of eta (max-min), must be
//! below 0.03 — both engines target the same probit profile likelihood, so the
//! only legitimate gap is the modest smoothing/basis-convention difference, and
//! a larger gap signals a real link-dispatch bug.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use statrs::distribution::{ContinuousCDF, Normal};

const N: usize = 250;
const SEED: u64 = 789;
const NGRID: usize = 30;

fn truth_eta(x1: f64) -> f64 {
    -0.1 + x1.sin()
}

#[test]
fn gam_binomial_probit_matches_statsmodels() {
    init_parallelism();

    // ---- synthetic data: x1~U(-3,3); eta=-0.1+sin(x1); y~Bernoulli(Phi(eta)) ----
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(-3.0, 3.0).expect("uniform x1");
    let uunit = Uniform::new(0.0, 1.0).expect("uniform unit");
    let std_normal = Normal::new(0.0, 1.0).expect("standard normal");

    let x1: Vec<f64> = (0..N).map(|_| ux.sample(&mut rng)).collect();
    let y: Vec<f64> = x1
        .iter()
        .map(|&x| {
            let p = std_normal.cdf(truth_eta(x)); // Phi(eta)
            if uunit.sample(&mut rng) < p { 1.0 } else { 0.0 }
        })
        .collect();

    // ---- evaluation grid: 30 points x1~U(-3,3) (drawn after the data) -----
    let xgrid: Vec<f64> = (0..NGRID).map(|_| ux.sample(&mut rng)).collect();

    // ---- fit with gam: y ~ s(x1, k=4), Binomial(probit) -------------------
    let headers = ["x1", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x1
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode binomial dataset");
    let col = ds.column_map();
    let x1_idx = col["x1"];

    let cfg = FitConfig {
        family: Some("binomial-probit".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x1, k=4)", &ds, &cfg).expect("gam probit fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for binomial-probit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam linear predictor on the evaluation grid: design*beta == eta (the
    // link is applied on top of eta, so the design times coefficients is the
    // linear predictor itself, independent of which inverse link was chosen).
    let mut grid = Array2::<f64>::zeros((NGRID, ds.headers.len()));
    for (i, &x) in xgrid.iter().enumerate() {
        grid[[i, x1_idx]] = x;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at evaluation grid");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with statsmodels (the mature reference) -------
    // statsmodels GLM with a Binomial(probit) family on a natural-cubic-spline
    // basis of x1 (df=4 columns, matching gam's k=4 cubic smooth), then predict
    // the linear predictor at the evaluation grid. patsy's `cr(x1, df=4)`
    // builds a centered cubic regression spline, the closest statsmodels-side
    // analogue of gam's penalized cubic s(x1, k=4); both engines land on the
    // same probit profile likelihood.
    let grid_literal = xgrid
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(", ");
    let body = format!(
        r#"
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

xgrid = np.array([{grid_literal}], dtype=float)

# Probit GLM with a cubic regression spline on x1 (df=4, matching s(x1, k=4)).
model = smf.glm(
    "y ~ cr(x1, df=4)",
    data=df,
    family=sm.families.Binomial(link=sm.families.links.Probit()),
)
res = model.fit()

# Linear predictor (eta) at the evaluation grid. `linear=True` returns the
# untransformed predictor (design*beta) rather than the probit mean Phi(eta).
import pandas as pd
eta = res.predict(pd.DataFrame({"x1": xgrid}), linear=True)
emit("eta", np.asarray(eta, dtype=float))
emit("df_model", [float(res.df_model)])
"#
    );

    let r = run_python(
        &[Column::new("x1", &x1), Column::new("y", &y)],
        &body,
    );
    let sm_eta = r.vector("eta");
    assert_eq!(sm_eta.len(), NGRID, "statsmodels eta length mismatch");

    // ---- compare on the linear predictor ----------------------------------
    let mad = max_abs_diff(&gam_eta, sm_eta);
    let eta_min = sm_eta.iter().cloned().fold(f64::INFINITY, f64::min);
    let eta_max = sm_eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eta_scale = (eta_max - eta_min).abs().max(1e-6);
    let rel_mad = mad / eta_scale;

    eprintln!(
        "binomial-probit s(x1,k=4): n={N} ngrid={NGRID} gam_edf={gam_edf:.3} \
         eta_scale={eta_scale:.4} max_abs_diff={mad:.5} rel_mad={rel_mad:.5}"
    );

    // Both PIRLS (gam) and statsmodels IRLS fit the same probit profile
    // likelihood on a comparable df=4 cubic basis, so the linear predictors
    // must agree to well within 3% of eta's range; a larger gap means gam's
    // probit link dispatch diverges from the standard GLM.
    assert!(
        rel_mad < 0.03,
        "gam probit linear predictor diverges from statsmodels: \
         max_abs_diff={mad:.5}, eta_scale={eta_scale:.4}, rel_mad={rel_mad:.5}"
    );
}
