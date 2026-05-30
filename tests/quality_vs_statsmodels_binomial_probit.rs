//! End-to-end quality: gam's Binomial(probit) GLM must RECOVER THE TRUTH on a
//! synthetic dataset drawn from a known probit mean curve.
//!
//! OBJECTIVE METRIC (the pass/fail claim): the data are generated from a known
//! function `eta_true(x1) = -0.1 + sin(x1)` with `y ~ Bernoulli(Phi(eta_true))`.
//! The link-identified, basis-invariant quantity is the fitted mean
//! `mu(x) = Phi(eta(x))`, whose ground truth is `mu_true(x) = Phi(eta_true(x))`.
//! We therefore assert that gam's fitted probability curve recovers `mu_true`:
//!   * RMSE(gam_mu, mu_true) on a held-out grid is below a principled bar tied
//!     to the achievable precision of a 3-df smooth on n=250 binary draws.
//!
//! This is a *truth-recovery* assertion, not a "matches statsmodels" assertion:
//! reproducing another tool's noisy fit proves nothing, but recovering the
//! data-generating function is objective quality.
//!
//! BASELINE TO MATCH-OR-BEAT: statsmodels `GLM(Binomial(link=Probit))` on a
//! cubic regression spline `cr(x1, df=4)` is fit to the identical data. It is
//! the mature reference, so we additionally require gam's truth-recovery error
//! to be no worse than statsmodels' truth-recovery error (within a 10% margin):
//!   RMSE(gam_mu, mu_true) <= 1.10 * RMSE(sm_probit_mu, mu_true).
//! statsmodels is demoted to an accuracy baseline; it is never the ground truth.
//!
//! LINK-DISPATCH DISCRIMINATOR (still objective, now phrased on accuracy):
//! the same data are also fit with a Binomial *logit* link in statsmodels. The
//! logit inverse link is the wrong inverse link for probit-generated data, so a
//! correctly-dispatched probit gam must recover `mu_true` at least as well as
//! the logit fit does — and the probit reference must in turn beat (or tie) the
//! logit reference. If gam's "binomial-probit" silently ran logit, gam's error
//! would track the logit fit's error rather than improving on it, and this
//! ordering would fail. We assert gam recovers truth no worse than the
//! statsmodels logit fit (within a small margin), which only holds when the
//! probit inverse link is genuinely used.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_python};
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
fn gam_binomial_probit_recovers_truth() {
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

    // Ground-truth mean curve at the evaluation grid: mu_true = Phi(eta_true).
    // This is the data-generating function and the objective target of the fit.
    let mu_true: Vec<f64> = xgrid.iter().map(|&x| std_normal.cdf(truth_eta(x))).collect();

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

    // ---- fit the SAME data with statsmodels (the mature ACCURACY baseline) -
    // statsmodels GLM with a cubic regression spline of x1 (`cr(x1, df=4)`,
    // df=4 columns, matching gam's k=4 cubic smooth), under TWO binomial links:
    // probit (the model gam claims to fit) and logit (the wrong-link control).
    // We return the fitted MEAN curve mu at the evaluation grid for each link
    // and score each against the ground-truth mean.
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

    // ---- OBJECTIVE METRIC: recovery of the ground-truth mean curve --------
    let gam_err = rmse(&gam_mu, &mu_true);
    let sm_probit_err = rmse(sm_mu_probit, &mu_true);
    let sm_logit_err = rmse(sm_mu_logit, &mu_true);

    // Context-only diagnostics (NOT pass criteria): how close gam tracks the
    // mature reference's own (noisy) probit fit, printed for triage.
    let corr_ref = pearson(&gam_mu, sm_mu_probit);
    let rel_ref = relative_l2(&gam_mu, sm_mu_probit);

    eprintln!(
        "binomial-probit s(x1,k=4): n={N} ngrid={NGRID} gam_edf={gam_edf:.3} \
         rmse_truth(gam)={gam_err:.4} rmse_truth(sm_probit)={sm_probit_err:.4} \
         rmse_truth(sm_logit)={sm_logit_err:.4} | ctx: pearson(gam,sm_probit)={corr_ref:.5} \
         rel_l2(gam,sm_probit)={rel_ref:.4}"
    );

    // PRIMARY CLAIM: gam recovers the data-generating probability curve. With a
    // 3-df smooth on n=250 Bernoulli draws over x1~U(-3,3), the binomial sampling
    // noise on the mean is the dominant error floor; an RMSE of 0.06 on the
    // probability scale (mu spans roughly Phi(-1.1)..Phi(0.9), about a 0.6 range)
    // is well inside what a correct probit fit achieves and far above its floor,
    // while loudly failing a fit that does not track the truth.
    assert!(
        gam_err < 0.06,
        "gam probit fit does not recover the truth: rmse(gam, mu_true)={gam_err:.4} (bar 0.06)"
    );

    // MATCH-OR-BEAT the mature reference on truth-recovery ACCURACY: gam's error
    // must be no worse than statsmodels' probit error within a 10% margin. This
    // demotes statsmodels to a baseline — gam must be at least as accurate at
    // recovering the truth, not merely "similar" to statsmodels.
    assert!(
        gam_err <= 1.10 * sm_probit_err,
        "gam probit is less accurate than the statsmodels probit baseline at recovering truth: \
         rmse(gam)={gam_err:.4} > 1.10 * rmse(sm_probit)={sm_probit_err:.4}"
    );

    // LINK-DISPATCH DISCRIMINATOR (objective, on accuracy): the logit inverse
    // link is the wrong inverse link for probit-generated data. A correctly
    // dispatched probit gam must recover the truth at least as well as the
    // statsmodels *logit* fit (within a small margin); were gam's "probit"
    // silently running logit, its error would track the logit fit instead of
    // improving on it. The probit reference must itself be no worse than the
    // logit reference, confirming the link genuinely helps on this data.
    assert!(
        sm_probit_err <= sm_logit_err + 1e-9,
        "sanity: statsmodels probit should recover probit-generated truth at least as well as \
         logit (probit={sm_probit_err:.4}, logit={sm_logit_err:.4})"
    );
    assert!(
        gam_err <= sm_logit_err + 0.01,
        "gam 'probit' recovers truth no better than a wrong-link (logit) fit — link dispatch is \
         suspect: rmse(gam)={gam_err:.4} vs rmse(sm_logit)={sm_logit_err:.4}"
    );
}
