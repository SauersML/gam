//! End-to-end OBJECTIVE quality: gam's Poisson(log) GLM with smooth terms must
//! RECOVER the known truth used to generate the data.
//!
//! Poisson(log) with smooth covariates is the quintessential count-data GAM. We
//! generate a fixed-seed synthetic dataset where the true linear predictor is
//!
//!     eta = 0.5 + 0.3*sin(x1*pi/5) + 0.2*cos(x2*pi/5),   y ~ Poisson(exp(eta)),
//!
//! and fit `y ~ s(x1, k=5) + s(x2, k=5)` with gam (REML smoothing-parameter
//! selection, log link).
//!
//! OBJECTIVE METRIC (the pass/fail claim): truth recovery on the linear-predictor
//! scale. We assert that gam's fitted `eta_hat` reconstructs the noise-free
//! generating `eta` with small RMSE. Because both the centered smooth basis and
//! the data-generating expression carry an arbitrary additive offset (the smooth
//! terms are mean-centered for identifiability while the intercept absorbs the
//! level), the meaningful, units-preserving error is the RMSE between the
//! mean-centered fitted eta and the mean-centered truth — this measures whether
//! gam recovered the SHAPE of the sinusoidal signal in eta-units, not merely its
//! correlation. The signal (centered) has standard deviation ~0.26 and a
//! peak-to-peak range ~1.0; we require the recovery RMSE to be a small fraction
//! of that range, which a correct k=5 cubic-spline Poisson PIRLS clears easily at
//! n=200 while a broken inverse link / design / PIRLS would not.
//!
//! BASELINE TO MATCH-OR-BEAT: statsmodels `GLMGam(family=Poisson(link=Log()))`
//! with the SAME penalized B-spline smooths and GCV penalty selection, fed the
//! IDENTICAL data, is fit and scored on the SAME truth-recovery metric. We assert
//! gam's recovery RMSE is no worse than 1.10x statsmodels' — i.e. gam is at least
//! as ACCURATE at recovering the truth as the mature reference. Matching
//! statsmodels' fitted output is NOT a pass criterion; cross-engine agreement is
//! computed and printed for context only.

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
use rand_distr::{Distribution, Poisson, Uniform};

const N: usize = 200;
const SEED: u64 = 42;

fn truth_eta(x1: f64, x2: f64) -> f64 {
    let pi = std::f64::consts::PI;
    0.5 + 0.3 * (x1 * pi / 5.0).sin() + 0.2 * (x2 * pi / 5.0).cos()
}

/// Subtract the mean so two predictors are compared on a common (offset-free)
/// scale — eta is identifiable only up to an additive constant split between the
/// centered smooths and the intercept, so the SHAPE lives in the centered vector.
fn centered(v: &[f64]) -> Vec<f64> {
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    v.iter().map(|x| x - mean).collect()
}

#[test]
fn gam_poisson_log_recovers_truth() {
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

    // ---- fit the SAME model with statsmodels (the match-or-beat baseline) --
    // GLMGam with two penalized cubic B-spline smooths (df=5 each, matching
    // k=5) under Poisson(Log). select_penweight() picks the per-smoother penalty
    // by GCV, then we refit at that optimum, so statsmodels actually performs
    // smoothing-parameter selection (comparable to gam's REML) rather than
    // fitting unpenalized at the alpha=0 default. We then score statsmodels on
    // the SAME truth-recovery metric as gam.
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

    // ---- OBJECTIVE metric: truth recovery on the (centered) eta scale ------
    // eta is identifiable only up to an additive constant, so center both the
    // fit and the truth before measuring RMSE: this isolates SHAPE recovery in
    // eta-units and is independent of either engine's centering convention.
    let truth_c = centered(&eta_truth);
    let gam_eta_c = centered(&gam_eta);
    let sm_eta_c = centered(sm_eta);

    let gam_recovery_rmse = rmse(&gam_eta_c, &truth_c);
    let sm_recovery_rmse = rmse(&sm_eta_c, &truth_c);

    // Scale of the signal we are trying to recover (centered truth).
    let signal_sd = {
        let mss: f64 = truth_c.iter().map(|v| v * v).sum::<f64>() / truth_c.len() as f64;
        mss.sqrt()
    };
    let signal_range = {
        let lo = truth_c.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = truth_c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };

    // Context-only cross-engine diagnostics (NOT pass criteria).
    let corr_truth = pearson(&gam_eta, &eta_truth);
    let corr_mean = pearson(&gam_mean, sm_mu);
    let rel_eta_cross = relative_l2(&gam_eta, sm_eta);

    eprintln!(
        "poisson s(x1)+s(x2): n={N} gam_edf={gam_edf:.3} signal_sd={signal_sd:.4} \
         signal_range={signal_range:.4} \
         recovery_rmse: gam={gam_recovery_rmse:.4} statsmodels={sm_recovery_rmse:.4} \
         | context: pearson(eta,truth)={corr_truth:.5} pearson(mean)={corr_mean:.5} \
         rel_l2(eta,statsmodels)={rel_eta_cross:.4}"
    );

    // (1) PRIMARY OBJECTIVE CLAIM: gam recovers the noise-free truth. The
    //     centered signal has sd ~0.26 and range ~1.0; we require the recovery
    //     RMSE below 0.12 eta-units (< ~half a signal sd, ~12% of the range).
    //     This is an absolute accuracy bar a correct Poisson(log) k=5 PIRLS
    //     clears at n=200; a broken inverse link/design/PIRLS overshoots it.
    assert!(
        gam_recovery_rmse < 0.12,
        "gam fails to recover the smooth truth on the eta scale: rmse={gam_recovery_rmse:.4} \
         (signal_sd={signal_sd:.4}, signal_range={signal_range:.4})"
    );

    // (2) MATCH-OR-BEAT on ACCURACY: gam's truth-recovery error must be no worse
    //     than 1.10x the mature reference's on the SAME metric. gam is at least
    //     as accurate as statsmodels at reconstructing the generating function.
    assert!(
        gam_recovery_rmse <= sm_recovery_rmse * 1.10,
        "gam is less accurate at recovering the truth than statsmodels: \
         gam_rmse={gam_recovery_rmse:.4} > 1.10 * statsmodels_rmse={sm_recovery_rmse:.4}"
    );
}
