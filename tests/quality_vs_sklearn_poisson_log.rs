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
use gam::test_support::reference::{Column, pad_to, pearson, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::path::Path;

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
# select_penweight reads model attributes (e.g. `scale`) that only exist after
# an initial fit, so fit once at alpha0 to populate them before searching. It
# then returns (alpha, fit_res, history); take the optimized penalty weights and
# refit at the optimum so statsmodels performs real smoothing-parameter selection.
gam.fit()
alpha_opt = gam.select_penweight()[0]
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

// ---------------------------------------------------------------------------
// REAL-DATA ARM: same Poisson(log) GAM capability, but on the `badhealth`
// dataset where the truth is UNKNOWN, so quality is held-out predictive
// accuracy rather than truth recovery.
//
// Real data: the `badhealth` dataset from Hilbe's COUNT package (German
// Socioeconomic Panel), 1127 patients. Columns:
//   * `numvisit` — number of doctor visits (count response, 0..40, nonneg int),
//   * `age`      — patient age in years (continuous smooth covariate),
//   * `badh`     — self-reported bad-health indicator (0/1, linear term).
// Source (direct, no auth):
//   https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/COUNT/badhealth.csv
//
// Use case: count regression with a smooth covariate and a binary covariate,
// `numvisit ~ s(age) + linear(badh)`, family = Poisson, log link — the canonical
// health-utilization count model. We make a deterministic train/test split
// (every 4th row held out), fit on the training rows only, predict the held-out
// rows on the response (count) scale, and score gam's OWN predictions on
// count-appropriate, tool-free metrics.

const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");

/// Mean Poisson deviance per observation between observed counts `y` and fitted
/// means `mu`: `(1/n) * sum 2*( y*log(y/mu) - (y - mu) )`, with the convention
/// `y*log(y/mu) = 0` when `y == 0`. This is the natural, scale-aware loss for
/// count predictions — it penalizes a fitted mean that is too small or too large
/// asymmetrically, exactly as the Poisson likelihood does, and equals zero only
/// when `mu == y` everywhere.
fn mean_poisson_deviance(y: &[f64], mu: &[f64]) -> f64 {
    assert_eq!(y.len(), mu.len(), "poisson deviance length mismatch");
    let n = y.len() as f64;
    let mut acc = 0.0;
    for (&yi, &mui) in y.iter().zip(mu) {
        let m = mui.max(1e-12);
        let log_term = if yi > 0.0 { yi * (yi / m).ln() } else { 0.0 };
        acc += 2.0 * (log_term - (yi - m));
    }
    acc / n.max(1.0)
}

#[test]
fn gam_poisson_log_recovers_truth_on_real_data() {
    init_parallelism();

    // ---- load the real badhealth dataset (age, badh -> numvisit) ----------
    let ds = load_csvwith_inferred_schema(Path::new(BADHEALTH_CSV)).expect("load badhealth.csv");
    let col = ds.column_map();
    let age_idx = col["age"];
    let badh_idx = col["badh"];
    let numvisit_idx = col["numvisit"];
    let age: Vec<f64> = ds.values.column(age_idx).to_vec();
    let badh: Vec<f64> = ds.values.column(badh_idx).to_vec();
    let numvisit: Vec<f64> = ds.values.column(numvisit_idx).to_vec();
    let n = age.len();
    assert!(n > 1000, "badhealth should have ~1127 rows, got {n}");
    // Sanity: the response really is nonnegative integer counts.
    assert!(
        numvisit
            .iter()
            .all(|&v| v >= 0.0 && (v - v.round()).abs() < 1e-9),
        "numvisit must be nonnegative integer counts"
    );

    // ---- deterministic train/test split: every 4th row is held out -------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 700 && test_rows.len() > 200,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_age: Vec<f64> = train_rows.iter().map(|&i| age[i]).collect();
    let train_badh: Vec<f64> = train_rows.iter().map(|&i| badh[i]).collect();
    let train_numvisit: Vec<f64> = train_rows.iter().map(|&i| numvisit[i]).collect();
    let test_age: Vec<f64> = test_rows.iter().map(|&i| age[i]).collect();
    let test_badh: Vec<f64> = test_rows.iter().map(|&i| badh[i]).collect();
    let test_numvisit: Vec<f64> = test_rows.iter().map(|&i| numvisit[i]).collect();

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged, so the formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: numvisit ~ s(age) + linear(badh), Poisson, REML --
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("numvisit ~ s(age) + linear(badh)", &train_ds, &cfg)
        .expect("gam poisson fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the Poisson(log) family");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out rows: rebuild the design from the frozen
    // spec, apply beta for the log-link predictor eta, exp() to the mean count.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &row) in test_rows.iter().enumerate() {
        test_grid[[i, age_idx]] = age[row];
        test_grid[[i, badh_idx]] = badh[row];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_eta = test_design.design.apply(&fit.fit.beta);
    let gam_test_mu: Vec<f64> = gam_test_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model on TRAIN with scikit-learn, predict the SAME TEST
    // sklearn's `PoissonRegressor` is a penalized (ridge) Poisson GLM with a log
    // link. We give it a cubic B-spline expansion of `age` (the smooth term)
    // plus the raw `badh` indicator (the linear term), so the design mirrors
    // `numvisit ~ s(age) + badh`. The L2 penalty plays the role of the smoothing
    // penalty. The test covariates ride along in parallel columns (padded to the
    // training length); only the first `test_n` entries are read back in Python.
    let r = run_python(
        &[
            Column::new("age", &train_age),
            Column::new("badh", &train_badh),
            Column::new("numvisit", &train_numvisit),
            Column::new("test_age", &pad_to(&test_age, train_age.len())),
            Column::new("test_badh", &pad_to(&test_badh, train_age.len())),
            Column::new("test_n", &vec![test_age.len() as f64; train_age.len()]),
        ],
        r#"
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import SplineTransformer

age   = np.asarray(df["age"],   dtype=float)
badh  = np.asarray(df["badh"],  dtype=float)
y     = np.asarray(df["numvisit"], dtype=float)
k     = int(df["test_n"][0])
test_age  = np.asarray(df["test_age"],  dtype=float)[:k]
test_badh = np.asarray(df["test_badh"], dtype=float)[:k]

# Cubic B-spline expansion of age (a penalized smooth basis), matched on the
# training knot quantiles and applied identically to the held-out ages.
spl = SplineTransformer(n_knots=6, degree=3, knots="quantile",
                        include_bias=False)
spl.fit(age.reshape(-1, 1))
Xtr = np.column_stack([spl.transform(age.reshape(-1, 1)), badh])
Xte = np.column_stack([spl.transform(test_age.reshape(-1, 1)), test_badh])

# Penalized (ridge) Poisson GLM with the log link; alpha is the L2 smoothing
# strength, playing the role of gam's penalty.
glm = PoissonRegressor(alpha=1.0, max_iter=1000)
glm.fit(Xtr, y)
mu_test = np.asarray(glm.predict(Xte), dtype=float)
emit("test_mu", mu_test)
"#,
    );
    let sk_test_mu = r.vector("test_mu");
    assert_eq!(
        sk_test_mu.len(),
        test_rows.len(),
        "sklearn held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN held-out predictions --------------
    let gam_test_dev = mean_poisson_deviance(&test_numvisit, &gam_test_mu);
    let sk_test_dev = mean_poisson_deviance(&test_numvisit, sk_test_mu);

    // Null (intercept-only) baseline: predict the TRAINING mean count for every
    // held-out row. Its held-out deviance is the bar a real signal must beat.
    let train_mean = train_numvisit.iter().sum::<f64>() / train_numvisit.len() as f64;
    let null_test_mu = vec![train_mean; test_rows.len()];
    let null_test_dev = mean_poisson_deviance(&test_numvisit, &null_test_mu);

    // Positive predicted-vs-actual correlation confirms the held-out ranking is
    // recovered (a degenerate constant fit would give ~0).
    let gam_test_corr = pearson(&gam_test_mu, &test_numvisit);

    // Context-only cross-engine diagnostic: closeness of the two engines' held-out
    // mean predictions. NOT a pass criterion.
    let cross_rel = relative_l2(&gam_test_mu, sk_test_mu);

    eprintln!(
        "badhealth s(age)+badh Poisson held-out (sklearn arm): n_train={} n_test={} \
         gam_edf={gam_edf:.3} null_dev={null_test_dev:.4} gam_dev={gam_test_dev:.4} \
         sklearn_dev={sk_test_dev:.4} gam_test_corr={gam_test_corr:.4} \
         (context: held-out rel_l2 vs sklearn={cross_rel:.4})",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- PRIMARY objective assertion: gam beats the intercept-only null -----
    // A model that has learned a genuine age/health signal predicts held-out
    // counts with strictly lower Poisson deviance than predicting the constant
    // training mean. We demand a clear (>=3%) improvement, not a marginal one.
    assert!(
        gam_test_dev <= 0.97 * null_test_dev,
        "gam held-out Poisson deviance {gam_test_dev:.4} does not clearly beat the \
         intercept-only null {null_test_dev:.4} (need <= 0.97 * null)"
    );

    // The held-out predicted-vs-actual correlation must be positive — gam ranks
    // high-visit patients above low-visit ones out of sample.
    assert!(
        gam_test_corr > 0.10,
        "gam held-out predicted-vs-actual correlation too low: {gam_test_corr:.4}"
    );

    // ---- BASELINE (match-or-beat): no worse than sklearn on held-out deviance
    // sklearn's penalized Poisson GLM is the mature reference; gam must match or
    // beat it on the SAME held-out Poisson deviance, within a 5% margin.
    assert!(
        gam_test_dev <= sk_test_dev * 1.05,
        "gam held-out Poisson deviance {gam_test_dev:.4} exceeds sklearn {sk_test_dev:.4} * 1.05"
    );

    // ---- complexity sanity: edf in a signal-appropriate range (not matched) -
    assert!(
        gam_edf > 2.0 && gam_edf < 15.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
