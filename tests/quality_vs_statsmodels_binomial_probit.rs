//! End-to-end quality: gam's Binomial(probit) GLM must match statsmodels —
//! the standard reference GLM implementation — on the fitted probability curve.
//!
//! This is a *cross-link* check. The logit link is gam's default binomial link,
//! so a separate probit test verifies that gam's link-family dispatch
//! (`family="binomial-probit"` -> `InverseLink::Standard(StandardLink::Probit)`)
//! is wired correctly for a non-logit binomial. Probit is the second-most
//! common binomial link (the standard-normal-CDF inverse, lighter-tailed than
//! logit), and statsmodels `GLM(family=Binomial(link=Probit()))` is the
//! canonical reference implementation.
//!
//! We fit the *same* synthetic data with gam (`y ~ s(x1, k=4)`, probit) and
//! statsmodels (a probit GLM on a cubic regression spline `cr(x1, df=4)` of x1)
//! and compare the fitted **mean** mu = Phi(eta) on a held-out evaluation grid.
//! The comparison is deliberately on the probability scale rather than on eta:
//!   * gam REML-*penalizes* its smooth toward a lower-rank fit, while patsy's
//!     `cr` basis enters statsmodels *unpenalized*, and the two cubic bases use
//!     different knot/centering conventions. So eta need not coincide
//!     coefficient-for-coefficient even though both encode the same probit
//!     model — but the fitted probability curve they imply (the only
//!     link-identified, basis-invariant quantity) must.
//!
//! To make this a genuine *link-dispatch* test and not merely "two cubic fits
//! look similar", we additionally fit the identical data under a Binomial
//! *logit* link in statsmodels and assert that gam's probit mean is strictly
//! closer to the statsmodels probit mean than to the statsmodels logit mean.
//! If gam's "probit" dispatch were silently running logit, that ordering would
//! invert — so this comparison fails loudly on a real link-wiring bug.
//!
//! Bounds (both justified by the small k=4 basis on n=250 binary observations):
//!   * Pearson correlation of gam-probit mu vs statsmodels-probit mu > 0.99 and
//!     relative-L2 of the mean curve < 0.05 — the penalty/basis gap on a 3-df
//!     smooth moves probabilities by at most a few percent of their spread.
//!   * gam-vs-(statsmodels probit) relative-L2 must be < 0.5x gam-vs-(logit),
//!     a strict separation that only holds if gam genuinely used the probit
//!     inverse link.

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

    // gam fitted mean on the grid: mu = Phi(eta) (probit inverse link).
    let gam_mu: Vec<f64> = gam_eta.iter().map(|&e| std_normal.cdf(e)).collect();

    // ---- fit the SAME data with statsmodels (the mature reference) --------
    // statsmodels GLM with a cubic regression spline of x1 (`cr(x1, df=4)`,
    // df=4 columns, matching gam's k=4 cubic smooth), under TWO binomial links:
    // probit (the model gam claims to fit) and logit (the discriminator). We
    // return the fitted MEAN curve mu at the evaluation grid for each link —
    // the mean is the link-identified, basis-invariant quantity to compare,
    // whereas eta would depend on each engine's (different) spline basis.
    let grid_literal = xgrid
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(", ");
    let body = format!(
        r#"
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

xgrid = pd.DataFrame({{"x1": np.array([{grid_literal}], dtype=float)}})

def fit_mean(link):
    model = smf.glm(
        "y ~ cr(x1, df=4)",
        data=df,
        family=sm.families.Binomial(link=link),
    )
    res = model.fit()
    # default predict() returns the mean mu = g^{{-1}}(eta) on the grid.
    return np.asarray(res.predict(xgrid), dtype=float)

emit("mu_probit", fit_mean(sm.families.links.Probit()))
emit("mu_logit", fit_mean(sm.families.links.Logit()))
"#
    );

    let r = run_python(&[Column::new("x1", &x1), Column::new("y", &y)], &body);
    let sm_mu_probit = r.vector("mu_probit");
    let sm_mu_logit = r.vector("mu_logit");
    assert_eq!(
        sm_mu_probit.len(),
        NGRID,
        "statsmodels probit mean length mismatch"
    );
    assert_eq!(
        sm_mu_logit.len(),
        NGRID,
        "statsmodels logit mean length mismatch"
    );

    // ---- compare on the fitted probability curve --------------------------
    let corr = pearson(&gam_mu, sm_mu_probit);
    let rel_probit = relative_l2(&gam_mu, sm_mu_probit);
    let rel_logit = relative_l2(&gam_mu, sm_mu_logit);

    eprintln!(
        "binomial-probit s(x1,k=4): n={N} ngrid={NGRID} gam_edf={gam_edf:.3} \
         pearson={corr:.5} rel_l2(probit)={rel_probit:.4} rel_l2(logit)={rel_logit:.4}"
    );

    // gam (penalized PIRLS, probit) and statsmodels (unpenalized IRLS, probit
    // on a cr df=4 basis) encode the same probit model; the only legitimate gap
    // is the REML penalty + differing cubic-basis convention on a 3-df smooth,
    // which moves the fitted probabilities by at most a few percent of their
    // spread. So the curves must be near-collinear and L2-close.
    assert!(
        corr > 0.99,
        "gam probit mean not collinear with statsmodels: pearson={corr:.5}"
    );
    assert!(
        rel_probit < 0.05,
        "gam probit mean curve diverges from statsmodels probit: rel_l2={rel_probit:.4}"
    );

    // Link-dispatch discriminator: if gam genuinely uses the probit inverse
    // link, its mean must sit strictly closer to the statsmodels *probit* mean
    // than to the statsmodels *logit* mean fit to identical data. A 0.5x margin
    // is comfortably satisfied when the right link is used and is violated only
    // if "binomial-probit" silently dispatches to logit (or any wrong link).
    assert!(
        rel_probit < 0.5 * rel_logit,
        "gam 'probit' fit is not closer to statsmodels probit than to logit \
         (rel_l2 probit={rel_probit:.4} vs logit={rel_logit:.4}) — link dispatch is suspect"
    );
}
