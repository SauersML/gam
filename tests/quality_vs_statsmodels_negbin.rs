//! End-to-end quality: gam's Negative-Binomial(log, fixed theta) GLM must
//! RECOVER THE TRUE conditional-mean function from overdispersed count data with
//! a smooth + linear additive structure. The objective metric is accuracy
//! against the known generating truth, not agreement with any reference tool.
//!
//! Capability under test: `family="negative-binomial"` with a fixed
//! overdispersion parameter `theta`, log link, and the key feature combination
//! `y ~ s(x, k=5) + linear(z)` (a penalized smooth term plus a parametric
//! linear term). The negative-binomial variance function is
//! `Var(mu) = mu + mu^2 / theta`, so the IRLS/PIRLS working weights depend on
//! `theta`; recovering the true mean function is exactly the test that gam's
//! PIRLS incorporates the theta-dependent variance function correctly.
//!
//! Data (seed=678, n=220): x~U(0,10), z~U(-2,2);
//! eta = 1.0 + 0.6*sin(x*pi/5) + 0.4*z; y ~ NegBinom(mu=exp(eta), theta=2.0).
//! Because the data is generated from this known eta, the true conditional mean
//! mu_true = exp(eta) is KNOWN at every row, so we can score gam's fitted means
//! directly against ground truth.
//!
//! OBJECTIVE METRIC (primary claim): relative RMSE of gam's fitted means against
//! the true means, rRMSE = RMSE(mu_gam, mu_true) / range(mu_true). On a true
//! smooth+linear signal with NB(theta=2) noise, a correctly-specified penalized
//! NB-log fit recovers the conditional mean to a small fraction of the signal
//! range; we require rRMSE <= 0.15.
//!
//! BASELINE TO MATCH-OR-BEAT (accuracy, not output-agreement): the same NB-log
//! model is fit with statsmodels — `sm.GLM(y, X, NegativeBinomial(alpha=1/theta))`
//! on a 5-df cubic B-spline of x (mirroring s(x,k=5)) + z + intercept, same fixed
//! dispersion. We require gam's RMSE-to-truth <= statsmodels' RMSE-to-truth * 1.10.
//! statsmodels is a yardstick on the SAME objective (truth recovery); we never
//! assert gam reproduces statsmodels' fitted output. The reference rel_l2 between
//! the two fits is printed only as context.
//!
//! Asserts:
//!   1. rRMSE(mu_gam, mu_true) <= 0.15 — gam recovers the true mean function.
//!   2. RMSE(mu_gam, mu_true) <= RMSE(mu_sm, mu_true) * 1.10 — gam is at least as
//!      accurate at recovering the truth as the mature reference.
//!   3. Pearson chi-square / n in (0.8, 1.2). For a correctly-specified NB GLM
//!      the Pearson statistic sum((y-mu)^2 / V(mu)) with the NB variance
//!      V(mu) = mu + mu^2/theta has E[(y-mu)^2 / V(mu)] = 1 per row, so the
//!      statistic ~ n and chi2/n ~ 1. This confirms the theta-dependent variance
//!      the PIRLS weights bake in matches the data's overdispersion; a model
//!      ignoring theta (plain Poisson variance) drives this well above 1.2.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};
use std::path::Path;

const N: usize = 220;
const THETA: f64 = 2.0;
const SEED: u64 = 678;

/// Sample one Negative-Binomial(mu, theta) count via the gamma-Poisson mixture:
/// lambda ~ Gamma(shape=theta, scale=mu/theta) so E[lambda]=mu and
/// Var(lambda)=mu^2/theta, then y ~ Poisson(lambda) giving the NB2 law with
/// Var(y) = mu + mu^2/theta — the same overdispersion gam and statsmodels model.
fn sample_negbin(mu: f64, theta: f64, rng: &mut StdRng) -> f64 {
    let gamma = Gamma::new(theta, mu / theta).expect("gamma params valid");
    let lambda = gamma.sample(rng);
    let pois = Poisson::new(lambda.max(1e-12)).expect("poisson rate valid");
    // rand_distr 0.6 `Poisson<f64>::sample` already returns the count as f64.
    pois.sample(rng)
}

#[test]
fn gam_negbin_matches_statsmodels_overdispersed_counts() {
    init_parallelism();

    // ---- synthesize identical data for both engines ----------------------
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 10.0).expect("uniform x");
    let uz = Uniform::new(-2.0, 2.0).expect("uniform z");

    let mut x = vec![0.0_f64; N];
    let mut z = vec![0.0_f64; N];
    let mut y = vec![0.0_f64; N];
    let mut mu_true = vec![0.0_f64; N];
    for i in 0..N {
        let xi = ux.sample(&mut rng);
        let zi = uz.sample(&mut rng);
        let eta = 1.0 + 0.6 * (xi * std::f64::consts::PI / 5.0).sin() + 0.4 * zi;
        let mu = eta.exp();
        x[i] = xi;
        z[i] = zi;
        mu_true[i] = mu;
        y[i] = sample_negbin(mu, THETA, &mut rng);
    }

    // ---- fit with gam: y ~ s(x, k=5) + linear(z), NB(log, theta=2) -------
    let headers = vec!["x".to_string(), "z".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode negbin dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        negative_binomial_theta: Some(THETA),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=5) + linear(z)", &ds, &cfg).expect("gam negbin fit");
    let FitResult::Standard(fit) = result else {
        panic!("negative-binomial GLM should produce a Standard fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted means at the training rows: rebuild the design from the frozen
    // spec, apply beta to get the log-link predictor eta, then exp() to means.
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_eta = design.design.apply(&fit.fit.beta);
    let gam_mu: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME structure with statsmodels (the mature reference) ---
    // A 5-df cubic B-spline on x (mirrors s(x, k=5)) + z + intercept, NB-log,
    // dispersion alpha = 1/theta held fixed (statsmodels does not re-estimate it).
    let r = run_python(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix

theta = 2.0
xv = np.asarray(df["x"], dtype=float)
zv = np.asarray(df["z"], dtype=float)
yv = np.asarray(df["y"], dtype=float)

# 5 spline columns on x to mirror gam's s(x, k=5); intercept supplied separately.
spl = np.asarray(dmatrix("bs(x, df=5, degree=3, include_intercept=False)",
                         {"x": xv}, return_type="dataframe"))
X = np.column_stack([np.ones_like(xv), spl, zv])

fam = sm.families.NegativeBinomial(alpha=1.0 / theta)  # alpha = 1/theta dispersion
m = sm.GLM(yv, X, family=fam).fit()
mu = np.asarray(m.fittedvalues, dtype=float)
emit("mu", mu)
"#,
    );
    let sm_mu = r.vector("mu");
    assert_eq!(sm_mu.len(), N, "statsmodels fitted-mean length mismatch");

    // ---- score BOTH fits against the KNOWN true conditional means --------
    // Truth recovery on the exp(eta) (fitted-mean) scale: mu_true is exact.
    let rmse_gam_truth = rmse(&gam_mu, &mu_true);
    let rmse_sm_truth = rmse(sm_mu, &mu_true);

    // Relative RMSE normalized by the dynamic range of the true mean function,
    // so the bar is scale-free and signal-appropriate.
    let mu_min = mu_true.iter().copied().fold(f64::INFINITY, f64::min);
    let mu_max = mu_true.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mu_range = (mu_max - mu_min).max(1e-12);
    let rrmse_gam = rmse_gam_truth / mu_range;

    // Pearson chi-square with the NB variance V(mu) = mu + mu^2/theta (this is
    // the statistic whose expectation is n, matching the cited (0.8,1.2) bound).
    let chi2_gam: f64 = (0..N)
        .map(|i| {
            let mu = gam_mu[i];
            let var = mu + mu * mu / THETA;
            let d = y[i] - mu;
            d * d / var.max(1e-12)
        })
        .sum();
    let chi2_over_n = chi2_gam / N as f64;

    // Reference output-agreement is printed for CONTEXT only — never asserted.
    let rel_l2_vs_sm = relative_l2(&gam_mu, sm_mu);
    eprintln!(
        "negbin s(x,k=5)+linear(z): n={N} theta={THETA} gam_edf={gam_edf:.3} \
         rmse(mu_gam,truth)={rmse_gam_truth:.4} rmse(mu_sm,truth)={rmse_sm_truth:.4} \
         rRMSE_gam={rrmse_gam:.4} chi2/n(NB-var)={chi2_over_n:.3} \
         rel_l2(gam,sm)={rel_l2_vs_sm:.4} (context only)"
    );

    // (1) PRIMARY: gam recovers the true conditional mean to a small fraction of
    // the signal range. NB(theta=2) noise on a smooth+linear signal of this size
    // is fit by a correctly-specified penalized NB-log model to rRMSE well under
    // 0.15; failure here means gam mis-recovers the mean, not "differs from a tool".
    assert!(
        rrmse_gam <= 0.15,
        "gam failed to recover the true NB mean: rRMSE={rrmse_gam:.4} (rmse={rmse_gam_truth:.4}, range={mu_range:.4})"
    );

    // (2) MATCH-OR-BEAT (accuracy on the SAME objective): gam is at least as
    // accurate at recovering the truth as the mature statsmodels NB-log fit.
    assert!(
        rmse_gam_truth <= rmse_sm_truth * 1.10,
        "gam less accurate than statsmodels at recovering truth: \
         rmse_gam={rmse_gam_truth:.4} > 1.10 * rmse_sm={rmse_sm_truth:.4}"
    );

    // (3) Dispersion sanity: under the NB variance V(mu)=mu+mu^2/theta,
    // E[(y-mu)^2/V(mu)] = 1, so chi2/n ~ 1; the (0.8,1.2) window confirms the
    // theta-dependent variance the PIRLS weights bake in matches the data's
    // overdispersion. A model ignoring theta drives this well above 1.2.
    assert!(
        (0.8..1.2).contains(&chi2_over_n),
        "Pearson chi2/n (NB variance) outside (0.8,1.2): {chi2_over_n:.3}"
    );
}

const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");

/// NB deviance per observation under the NB2 law with size `theta`:
/// `d_i = 2*[ y*log(y/mu) - (y+theta)*log((y+theta)/(mu+theta)) ]` (the `y*log`
/// term is 0 at `y=0`). This is the standard saturated-vs-fitted log-likelihood
/// deviance for `Var(mu) = mu + mu^2/theta`; lower is a better held-out fit.
fn nb_deviance_per_obs(y: f64, mu: f64, theta: f64) -> f64 {
    let mu = mu.max(1e-12);
    let t1 = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
    let t2 = (y + theta) * ((y + theta) / (mu + theta)).ln();
    2.0 * (t1 - t2)
}

/// Mean NB deviance over a held-out set.
fn mean_nb_deviance(y: &[f64], mu: &[f64], theta: f64) -> f64 {
    assert_eq!(y.len(), mu.len(), "nb deviance length mismatch");
    let s: f64 = y
        .iter()
        .zip(mu)
        .map(|(&yi, &mui)| nb_deviance_per_obs(yi, mui, theta))
        .sum();
    s / y.len().max(1) as f64
}

/// REAL-DATA ARM — same capability (penalized NB-log smooth + linear term,
/// theta-dependent overdispersion) on a real overdispersed count dataset where
/// the truth is unknown, so quality is OBJECTIVE held-out predictive accuracy.
///
/// Dataset: `badhealth` (German health-care utilization), the canonical gamlss
/// count-regression benchmark. SOURCE: R package `gamlss.data` / Winkelmann &
/// Boes, shipped at `bench/datasets/badhealth.csv` with columns
/// `numvisit` (doctor visits in the quarter, the count response),
/// `badh` (self-rated bad health, 0/1), `age` (years). The marginal
/// variance-to-mean ratio is ~5, so the data is strongly overdispersed —
/// Poisson is badly misspecified and a negative-binomial mean+overdispersion
/// model is required.
///
/// Capability exercised (identical to the synthetic arm): `family=
/// "negative-binomial"` with a fixed `theta`, log link, and the feature combo
/// `numvisit ~ s(age, k=8) + linear(badh)` (penalized smooth + parametric
/// linear term). `theta` is unknown for real data, so we set it from the TRAIN
/// rows by method of moments (`theta = mu^2 / (var - mu)`) and hand the SAME
/// fixed `theta` to gam and to statsmodels — a fair, shared dispersion.
///
/// Split: deterministic, every 4th row held out (fixed index, identical row
/// order fed to gam and to statsmodels).
///
/// OBJECTIVE METRIC (held-out, tool-free):
///   PRIMARY accuracy: mean held-out NB deviance of gam's predicted means must
///     clear an ABSOLUTE bar `<= 0.85`. The intercept-only (constant-mean) NB
///     predictor scores ~0.87 on this split, so beating 0.85 proves the
///     s(age)+badh structure genuinely improves out-of-sample mean prediction.
///   PRIMARY overdispersion: the TRAIN Pearson statistic under the NB variance
///     `V(mu)=mu+mu^2/theta`, `chi2/n`, lands in `(0.6, 1.4)`. Recomputed under
///     the POISSON variance `V(mu)=mu` it must exceed `2.0` — i.e. ignoring
///     theta would massively under-fit the spread. This is the direct
///     overdispersion-recovery claim.
///   BASELINE (match-or-beat): statsmodels fits the SAME NB-log model
///     (8-df cubic B-spline of age + badh + intercept, same fixed theta) on the
///     SAME train rows and predicts the SAME held-out rows; gam's held-out mean
///     NB deviance must be `<= sm_deviance * 1.10`. statsmodels is a yardstick
///     on the same held-out objective, never an output to replicate.
#[test]
fn gam_negbin_matches_statsmodels_overdispersed_counts_on_real_data() {
    init_parallelism();

    // ---- load badhealth: numvisit (count) ~ s(age) + badh -----------------
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

    // ---- deterministic train/test split: every 4th row held out ----------
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

    // ---- fixed overdispersion from TRAIN by method of moments -------------
    // NB2: Var = mu + mu^2/theta  =>  theta = mu^2 / (Var - mu). Estimated on
    // train counts only (test stays untouched), then handed identically to gam
    // and statsmodels so both share the same dispersion assumption.
    let n_tr = train_numvisit.len() as f64;
    let mu_tr = train_numvisit.iter().sum::<f64>() / n_tr;
    let var_tr = train_numvisit
        .iter()
        .map(|&v| (v - mu_tr) * (v - mu_tr))
        .sum::<f64>()
        / (n_tr - 1.0);
    let theta = mu_tr * mu_tr / (var_tr - mu_tr);
    assert!(
        theta.is_finite() && theta > 0.0,
        "train MoM theta must be finite and positive (overdispersed data): theta={theta}"
    );

    // Training-only dataset: sub-set the encoded rows; headers/schema/kinds are
    // unchanged so the formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: numvisit ~ s(age, k=8) + linear(badh), NB-log --
    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        negative_binomial_theta: Some(theta),
        ..FitConfig::default()
    };
    let result = fit_from_formula("numvisit ~ s(age, k=8) + linear(badh)", &train_ds, &cfg)
        .expect("gam negbin fit on badhealth train");
    let FitResult::Standard(fit) = result else {
        panic!("negative-binomial GLM should produce a Standard fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predicted means at the held-out rows: rebuild the frozen design,
    // apply beta for the log-link predictor, exp() to means.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &row) in test_rows.iter().enumerate() {
        test_grid[[i, age_idx]] = age[row];
        test_grid[[i, badh_idx]] = badh[row];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_eta = test_design.design.apply(&fit.fit.beta);
    let gam_test_mu: Vec<f64> = gam_test_eta.iter().map(|e| e.exp()).collect();

    // gam fitted means at the TRAIN rows (for the overdispersion claim).
    let mut train_grid = Array2::<f64>::zeros((train_rows.len(), p));
    for (i, &row) in train_rows.iter().enumerate() {
        train_grid[[i, age_idx]] = age[row];
        train_grid[[i, badh_idx]] = badh[row];
    }
    let train_design = build_term_collection_design(train_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_train_eta = train_design.design.apply(&fit.fit.beta);
    let gam_train_mu: Vec<f64> = gam_train_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME NB-log model with statsmodels (mature baseline) -----
    // 8-df cubic B-spline of age (mirrors s(age, k=8)) + badh + intercept,
    // dispersion alpha = 1/theta held fixed. Train rows fit; test rows predicted.
    // Within ONE python call every column is train-length; the test rows ride
    // along right-padded and only the first `test_n` entries are read back.
    let pad_to = |v: &[f64], len: usize| -> Vec<f64> {
        assert!(
            v.len() <= len,
            "pad target {len} shorter than source {}",
            v.len()
        );
        let fill = v.last().copied().unwrap_or(0.0);
        let mut out = v.to_vec();
        out.resize(len, fill);
        out
    };
    let tr_len = train_age.len();
    let r = run_python(
        &[
            Column::new("age", &train_age),
            Column::new("badh", &train_badh),
            Column::new("numvisit", &train_numvisit),
            Column::new("test_age", &pad_to(&test_age, tr_len)),
            Column::new("test_badh", &pad_to(&test_badh, tr_len)),
            Column::new("test_n", &vec![test_age.len() as f64; tr_len]),
            Column::new("theta", &vec![theta; tr_len]),
        ],
        r#"
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix

theta = float(np.asarray(df["theta"])[0])
k = int(round(float(np.asarray(df["test_n"])[0])))

age = np.asarray(df["age"], dtype=float)
badh = np.asarray(df["badh"], dtype=float)
y = np.asarray(df["numvisit"], dtype=float)
test_age = np.asarray(df["test_age"], dtype=float)[:k]
test_badh = np.asarray(df["test_badh"], dtype=float)[:k]

# Build the train spline and re-evaluate the SAME basis at the test ages so the
# spline knots/columns match between fit and prediction (mirrors s(age, k=8)).
train_dm = dmatrix("bs(age, df=8, degree=3, include_intercept=False)",
                   {"age": age}, return_type="dataframe")
test_spl = np.asarray(dmatrix(train_dm.design_info,
                              {"age": test_age}, return_type="dataframe"))
train_spl = np.asarray(train_dm)

Xtr = np.column_stack([np.ones_like(age), train_spl, badh])
Xte = np.column_stack([np.ones(k), test_spl, test_badh])

fam = sm.families.NegativeBinomial(alpha=1.0 / theta)  # alpha = 1/theta
m = sm.GLM(y, Xtr, family=fam).fit()
mu_te = np.asarray(m.predict(Xte), dtype=float)
emit("test_mu", mu_te)
"#,
    );
    let sm_test_mu = r.vector("test_mu");
    assert_eq!(
        sm_test_mu.len(),
        test_rows.len(),
        "statsmodels held-out prediction length mismatch"
    );

    // ---- objective held-out metrics on gam's OWN predictions --------------
    let gam_test_dev = mean_nb_deviance(&test_numvisit, &gam_test_mu, theta);
    let sm_test_dev = mean_nb_deviance(&test_numvisit, sm_test_mu, theta);

    // Intercept-only (constant train-mean) held-out NB deviance, for context.
    let null_mu = vec![mu_tr; test_rows.len()];
    let null_test_dev = mean_nb_deviance(&test_numvisit, &null_mu, theta);

    // Overdispersion: TRAIN Pearson chi2/n under the NB variance vs under the
    // (wrong) Poisson variance. The NB statistic should sit near 1; the Poisson
    // one blows up because the data is ~5x overdispersed.
    let chi2_nb: f64 = train_numvisit
        .iter()
        .zip(&gam_train_mu)
        .map(|(&yi, &mui)| {
            let var = mui + mui * mui / theta;
            let d = yi - mui;
            d * d / var.max(1e-12)
        })
        .sum::<f64>()
        / train_numvisit.len() as f64;
    let chi2_pois: f64 = train_numvisit
        .iter()
        .zip(&gam_train_mu)
        .map(|(&yi, &mui)| {
            let d = yi - mui;
            d * d / mui.max(1e-12)
        })
        .sum::<f64>()
        / train_numvisit.len() as f64;

    let rel_l2_vs_sm = relative_l2(&gam_test_mu, sm_test_mu);
    eprintln!(
        "badhealth numvisit~s(age,k=8)+linear(badh) NB-log held-out: \
         n_train={} n_test={} theta(MoM)={theta:.4} gam_edf={gam_edf:.3} \
         gam_test_dev={gam_test_dev:.4} sm_test_dev={sm_test_dev:.4} null_test_dev={null_test_dev:.4} \
         train_chi2/n(NB)={chi2_nb:.3} train_chi2/n(Poisson)={chi2_pois:.3} \
         rel_l2(gam,sm)={rel_l2_vs_sm:.4} (context only)",
        train_rows.len(),
        test_rows.len(),
    );

    // (1) PRIMARY accuracy: held-out mean NB deviance below the constant-mean
    // null (~0.87 on this split). 0.85 is an absolute, tool-free bar that gam's
    // s(age)+badh structure must clear to prove it predicts the held-out counts.
    assert!(
        gam_test_dev <= 0.85,
        "gam held-out mean NB deviance too high: {gam_test_dev:.4} (>= 0.85; null≈{null_test_dev:.4})"
    );

    // (2) PRIMARY overdispersion recovery: the NB-variance Pearson statistic is
    // O(1) while the Poisson-variance one explodes — i.e. gam's theta-dependent
    // variance captures the spread that a Poisson fit would miss entirely.
    assert!(
        (0.6..1.4).contains(&chi2_nb),
        "train Pearson chi2/n under NB variance outside (0.6,1.4): {chi2_nb:.3}"
    );
    assert!(
        chi2_pois > 2.0,
        "Poisson-variance Pearson chi2/n should expose overdispersion (>2.0): {chi2_pois:.3}"
    );

    // (3) BASELINE (match-or-beat, SAME held-out objective): gam's held-out NB
    // deviance is no worse than statsmodels' (lower deviance is better), within
    // a 10% margin. statsmodels is a yardstick, never an output to reproduce.
    assert!(
        gam_test_dev <= sm_test_dev * 1.10,
        "gam held-out NB deviance {gam_test_dev:.4} exceeds statsmodels {sm_test_dev:.4} * 1.10"
    );

    // complexity sanity: a smooth of age plus one linear term has modest edf.
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
