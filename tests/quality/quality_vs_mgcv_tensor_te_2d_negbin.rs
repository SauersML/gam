//! End-to-end quality: gam's tensor-product 2-D smooth `te(x, z)` under the
//! **Negative-Binomial** family (log link, fixed theta) must RECOVER THE TRUE
//! log-mean surface from overdispersed counts — not merely reproduce a peer fit.
//!
//! Why this combination earns its own test. gam's NB family was previously tested
//! only *additively* (`quality_vs_statsmodels_negbin`: `s(x)+linear(z)` against a
//! fixed-basis statsmodels GLM) — never against **mgcv**, and never on a genuine
//! tensor-product INTERACTION surface. The NB2 variance `Var(y)=mu+mu^2/theta`
//! makes the PIRLS working weights theta-dependent, so recovering a curved
//! interaction is a real test that gam's IRLS bakes the overdispersion in
//! correctly. mgcv's native `negbin(theta)` family (penalised REML, theta fixed
//! to match gam) is the canonical peer, demoted to a match-or-beat accuracy
//! baseline on the SAME objective, never an output to reproduce.
//!
//! Synthetic arm (seed=20260605, n=300): x, z ~ U(0,1); truth on the log-mean
//! scale `eta_true = 2.0 + sin(pi*x)*cos(pi*z)` => mu = exp(eta) in ~[2.7,20.1];
//! y ~ NegBinom(mu, theta=3) via the gamma-Poisson mixture. Asserts truth
//! recovery on the linear predictor, match-or-beat mgcv, and an NB-variance
//! Pearson chi2/n near 1.
//!
//! Real arm: the SAME capability on `badhealth` (overdispersed doctor visits),
//! `numvisit ~ te(age, badh)`, theta fixed from train by method of moments and
//! shared with mgcv; held-out NB deviance beats the constant-mean null and
//! match-or-beats mgcv.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pad_to, pearson, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};
use std::f64::consts::PI;
use std::path::Path;
use std::time::Instant;

const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");

const N: usize = 300;
const THETA: f64 = 3.0;

/// Sample one Negative-Binomial(mu, theta) count via the gamma-Poisson mixture:
/// lambda ~ Gamma(shape=theta, scale=mu/theta) so E[lambda]=mu and
/// Var(lambda)=mu^2/theta, then y ~ Poisson(lambda), giving the NB2 law with
/// Var(y) = mu + mu^2/theta.
fn sample_negbin(mu: f64, theta: f64, rng: &mut StdRng) -> f64 {
    let gamma = Gamma::new(theta, mu / theta).expect("gamma params valid");
    let lambda = gamma.sample(rng);
    let pois = Poisson::new(lambda.max(1e-12)).expect("poisson rate valid");
    pois.sample(rng)
}

/// NB unit deviance under the NB2 law with size `theta`:
/// `2*[ y*log(y/mu) - (y+theta)*log((y+theta)/(mu+theta)) ]` (the `y*log` term
/// is 0 at `y=0`). Lower is a better fit.
fn nb_deviance_per_obs(y: f64, mu: f64, theta: f64) -> f64 {
    let mu = mu.max(1e-12);
    let t1 = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
    let t2 = (y + theta) * ((y + theta) / (mu + theta)).ln();
    2.0 * (t1 - t2)
}

fn mean_nb_deviance(y: &[f64], mu: &[f64], theta: f64) -> f64 {
    assert_eq!(y.len(), mu.len(), "nb deviance length mismatch");
    let s: f64 = y
        .iter()
        .zip(mu)
        .map(|(&yi, &mui)| nb_deviance_per_obs(yi, mui, theta))
        .sum();
    s / y.len().max(1) as f64
}

#[test]
fn gam_tensor_te_2d_negbin_matches_mgcv() {
    init_parallelism();

    // ---- synthetic overdispersed-count truth on the unit square ------------
    // eta_true = 2.0 + sin(pi*x)*cos(pi*z) (a genuine interaction); mu = exp(eta);
    // y ~ NegBinom(mu, theta=3). A fixed seed feeds the SAME draws to gam and mgcv.
    let mut rng = StdRng::seed_from_u64(20260605);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x = Vec::with_capacity(N);
    let mut z = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut eta_true = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        let eta = 2.0 + (PI * xi).sin() * (PI * zi).cos();
        let mu = eta.exp();
        x.push(xi);
        z.push(zi);
        y.push(sample_negbin(mu, THETA, &mut rng));
        eta_true.push(eta);
    }

    // ---- fit with gam: y ~ te(x, z, k=7), NB(log, theta=3), REML -----------
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
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
    let gam_t0 = Instant::now();
    let result = fit_from_formula("y ~ te(x, z, k=7)", &ds, &cfg).expect("gam negbin te fit");
    let gam_elapsed = gam_t0.elapsed();
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for NB te(x, z)");
    };

    // gam linear predictor (log scale) at the training points, then mu = exp(eta).
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at training points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_mu: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model with mgcv (the mature reference) ---------------
    // family = negbin(theta = 3) fixes the overdispersion to match gam.
    let r_t0 = Instant::now();
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ te(x, z, k = 7), data = df,
                 family = negbin(theta = 3), method = "REML")
        emit("eta", as.numeric(predict(m, type = "link")))
        emit("edf", sum(m$edf))
        "#,
    );
    let r_elapsed = r_t0.elapsed();
    let mgcv_eta = r.vector("eta");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_eta.len(), N, "mgcv linear-predictor length mismatch");

    // ---- OBJECTIVE METRICS -------------------------------------------------
    let gam_err = rmse(&gam_eta, &eta_true);
    let mgcv_err = rmse(mgcv_eta, &eta_true);

    // NB-variance Pearson statistic with the TRUE theta: under Var=mu+mu^2/theta,
    // E[(y-mu)^2/Var] = 1, so this lands near 1 when gam's means and the family's
    // variance function are correct.
    let chi2_over_n: f64 = (0..N)
        .map(|i| {
            let mu = gam_mu[i];
            let var = mu + mu * mu / THETA;
            let d = y[i] - mu;
            d * d / var.max(1e-12)
        })
        .sum::<f64>()
        / N as f64;

    let rel_to_mgcv = relative_l2(&gam_eta, mgcv_eta);
    let corr_to_mgcv = pearson(&gam_eta, mgcv_eta);

    eprintln!(
        "te(x,z) NB/log theta={THETA}: n={N} gam_wall={:.2}s r_mgcv_wall={:.2}s \
         mgcv_edf={mgcv_edf:.3} \
         rmse_to_truth(gam)={gam_err:.4} rmse_to_truth(mgcv)={mgcv_err:.4} \
         chi2/n(NB-var)={chi2_over_n:.3} \
         [context] rel_l2(gam,mgcv)={rel_to_mgcv:.4} pearson(gam,mgcv)={corr_to_mgcv:.5}",
        gam_elapsed.as_secs_f64(),
        r_elapsed.as_secs_f64(),
    );

    // PRIMARY: gam recovers the true log-mean interaction surface. The wiggly part
    // sin(pi*x)*cos(pi*z) spans [-1,1] (range 2.0); with n=300 and NB(theta=3)
    // overdispersion a correct tensor / log-link fit stays well inside a fraction
    // of that span. We require RMSE(gam_eta, eta_true) < 0.35 (under 18% of the
    // signal range; NB noise is heavier than Poisson, hence slightly above the
    // Poisson-tensor 0.30 bar).
    assert!(
        gam_err < 0.35,
        "gam should recover the true NB log-mean surface: rmse_to_truth={gam_err:.4} (bar 0.35)"
    );

    // MATCH-OR-BEAT: no worse than mgcv by more than 10% on the objective metric.
    assert!(
        gam_err <= mgcv_err * 1.10,
        "gam's truth-recovery error must match-or-beat mgcv: \
         rmse_to_truth(gam)={gam_err:.4} vs mgcv*1.10={:.4}",
        mgcv_err * 1.10
    );

    // VARIANCE FUNCTION: the NB-variance Pearson statistic sits near 1. A model
    // ignoring theta (plain Poisson variance) drives this well above 1.2.
    assert!(
        (0.8..1.2).contains(&chi2_over_n),
        "NB-variance Pearson chi2/n outside (0.8,1.2): {chi2_over_n:.3}"
    );
}

/// REAL-DATA arm: `numvisit ~ te(age, badh)` under NB(log) on the `badhealth`
/// survey counts (strongly overdispersed). Quality is OBJECTIVE held-out NB
/// deviance on a deterministic split, with both a beat-the-null bar and a
/// match-or-beat against mgcv on that SAME metric.
#[test]
fn gam_tensor_te_2d_negbin_matches_mgcv_on_real_data() {
    init_parallelism();

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

    // ---- deterministic train/test split: every 4th row held out -----------
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
    // NB2: Var = mu + mu^2/theta => theta = mu^2/(Var - mu). Estimated on train
    // counts only, then handed identically to gam and mgcv.
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

    // ---- fit gam on TRAIN: numvisit ~ te(age, badh, k=5), NB-log ----------
    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        negative_binomial_theta: Some(theta),
        ..FitConfig::default()
    };
    let gam_t0 = Instant::now();
    let result = fit_from_formula("numvisit ~ te(age, badh, k=5)", &train_ds, &cfg)
        .expect("gam negbin te fit on badhealth train");
    let gam_elapsed = gam_t0.elapsed();
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for NB te(age, badh)");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out rows.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &row) in test_rows.iter().enumerate() {
        test_grid[[i, age_idx]] = age[row];
        test_grid[[i, badh_idx]] = badh[row];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at held-out points");
    let gam_test_eta: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    let gam_test_mu: Vec<f64> = gam_test_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model on TRAIN with mgcv, predict the SAME TEST ------
    let tr_len = train_age.len();
    let r_t0 = Instant::now();
    let r = run_r(
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
        suppressPackageStartupMessages(library(mgcv))
        th <- df$theta[1]
        # badh is binary {0,1}: a tensor te(age, badh) margin on badh is not
        # constructible (mgcv resets its k=2 up to a default > 2 unique values
        # and then errors / fails the inner loop). The mgcv-idiomatic encoding of
        # the SAME age x badh interaction that gam's te(age, badh) represents over
        # a binary margin is a smooth-by-factor: a separate s(age) curve per badh
        # level plus the badh main effect.
        df$badhf <- factor(df$badh)
        m <- gam(numvisit ~ s(age, by = badhf) + badhf, data = df,
                 family = negbin(theta = th), method = "REML")
        k <- df$test_n[1]
        newd <- data.frame(age = df$test_age[1:k],
                           badhf = factor(df$test_badh[1:k], levels = levels(df$badhf)))
        emit("test_mu", as.numeric(predict(m, newdata = newd, type = "response")))
        emit("edf", sum(m$edf))
        "#,
    );
    let r_elapsed = r_t0.elapsed();
    let mgcv_test_mu = r.vector("test_mu");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_test_mu.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- OBJECTIVE METRIC: held-out NB deviance (per held-out row) ---------
    let gam_dev = mean_nb_deviance(&test_numvisit, &gam_test_mu, theta);
    let mgcv_dev = mean_nb_deviance(&test_numvisit, mgcv_test_mu, theta);

    let null_mu = vec![mu_tr; test_rows.len()];
    let null_dev = mean_nb_deviance(&test_numvisit, &null_mu, theta);

    let rel_to_mgcv = relative_l2(&gam_test_mu, mgcv_test_mu);
    let corr_to_mgcv = pearson(&gam_test_mu, mgcv_test_mu);

    eprintln!(
        "badhealth te(age,badh) NB held-out: n_train={} n_test={} theta(MoM)={theta:.4} \
         gam_wall={:.2}s r_mgcv_wall={:.2}s \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         dev_gam={gam_dev:.4} dev_mgcv={mgcv_dev:.4} dev_null={null_dev:.4} \
         [context] rel_l2(gam,mgcv)={rel_to_mgcv:.4} pearson(gam,mgcv)={corr_to_mgcv:.5}",
        train_rows.len(),
        test_rows.len(),
        gam_elapsed.as_secs_f64(),
        r_elapsed.as_secs_f64(),
    );

    // PRIMARY accuracy: the te(age,badh) structure must explain held-out visits
    // better than the constant mean. Require at least 5% below the null deviance.
    assert!(
        gam_dev < null_dev * 0.95,
        "gam held-out NB deviance {gam_dev:.4} must beat the constant-mean \
         baseline {null_dev:.4} (bar: < 95% of null)"
    );

    // BASELINE (match-or-beat, SAME held-out objective): no worse than mgcv by
    // more than 10%.
    assert!(
        gam_dev <= mgcv_dev * 1.10,
        "gam held-out NB deviance {gam_dev:.4} exceeds mgcv {mgcv_dev:.4} * 1.10"
    );

    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
