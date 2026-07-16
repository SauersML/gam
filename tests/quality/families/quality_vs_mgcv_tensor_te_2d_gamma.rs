//! End-to-end quality: gam's tensor-product 2-D smooth `te(x, z)` under the
//! **Gamma** family (log link) must RECOVER THE TRUE log-mean surface from
//! strictly-positive, multiplicative-error data — not merely reproduce a peer
//! tool's fit.
//!
//! Why this combination earns its own test. gam already has a Gamma *additive*
//! test (`quality_vs_statsmodels_gamma_log`: `y ~ s(x)+s(z)` scored against a
//! fixed-basis statsmodels GLM) and mgcv *tensor* tests under Gaussian/Poisson —
//! but NO Gamma test against mgcv, and NO Gamma test on a genuine tensor-product
//! INTERACTION surface (`te(x,z)`, a non-additive function an `s(x)+s(z)` model
//! cannot represent). The Gamma log link has a notable property: its IRLS weights
//! are constant, `w = 1/(V(mu)·g'(mu)^2) = 1/(mu^2 · mu^-2) = 1`, so the family's
//! correctness lives entirely in (a) the log-link mean gradient that drives the
//! tensor-penalised surface and (b) the Gamma variance function `V(mu)=mu^2/shape`
//! that governs dispersion. This test exercises both, with **mgcv** — the
//! canonical penalised-REML GAM peer — demoted to a match-or-beat accuracy
//! baseline on the SAME objective (truth recovery / held-out deviance), never an
//! output to reproduce.
//!
//! Synthetic arm (truth recovery): data are drawn from a KNOWN log-mean
//! interaction surface `eta_true(x, z) = 2.0 + sin(pi*x)*cos(pi*z)` with
//! `y ~ Gamma(shape, scale = exp(eta_true)/shape)` so `E[y] = exp(eta_true)` and
//! `Var(y) = mu^2/shape` (multiplicative error). The objective metric is
//! `RMSE(eta_hat, eta_true)` on the linear-predictor (log-mean) scale — the scale
//! on which the tensor penalty acts and where a PIRLS / link-gradient bug shows.
//! A second assertion checks the Gamma variance function directly via the Pearson
//! statistic `mean( shape·(y-mu)^2/mu^2 )`, whose expectation is 1.
//!
//! Real-data arm (held-out predictive accuracy): the SAME capability on the
//! classic MASS `crabs` morphometrics (n=200, all measurements strictly positive
//! mm). Carapace width is predicted from frontal-lobe length and rear width,
//! `CW ~ te(FL, RW)`, a strongly allometric (multiplicative) relationship that
//! Gamma(log) is the textbook model for. Quality is OBJECTIVE held-out Gamma
//! deviance on a deterministic split, with both an absolute beat-the-null bar and
//! a match-or-beat against mgcv on that same metric.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Uniform};
use std::f64::consts::PI;
use std::path::Path;

use gam::test_support::reference::{
    Column, QualityPair, pad_to, pearson, relative_l2, rmse, run_r,
};

// MASS `crabs`: Leptograpsus crab morphometrics. Columns we use are FL (frontal
// lobe size, mm), RW (rear width, mm), CW (carapace width, mm) — all strictly
// positive continuous measurements. Vendored at bench/datasets/crabs.csv.
const CRABS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/crabs.csv");

/// Mean unit-deviance for a Gamma GLM: `d_i = 2*[ (y-mu)/mu - log(y/mu) ]`
/// (the saturated-vs-fitted log-likelihood deviance for `Var ∝ mu^2`). Both
/// `y` and `mu` are strictly positive here; lower is a better fit.
fn gamma_deviance_mean(y: &[f64], mu: &[f64]) -> f64 {
    assert_eq!(y.len(), mu.len(), "gamma deviance length mismatch");
    let s: f64 = y
        .iter()
        .zip(mu)
        .map(|(&yi, &mui)| {
            let mui = mui.max(1e-12);
            2.0 * ((yi - mui) / mui - (yi / mui).ln())
        })
        .sum();
    s / y.len().max(1) as f64
}

const SHAPE: f64 = 4.0;

#[test]
fn gam_tensor_te_2d_gamma_matches_mgcv() {
    init_parallelism();

    // ---- synthetic Gamma truth on the unit square --------------------------
    // eta_true = 2.0 + sin(pi*x)*cos(pi*z) (a genuine interaction, not additive);
    // mu = exp(eta_true) in [exp(1), exp(3)] ~ [2.7, 20.1] is strictly positive.
    // y ~ Gamma(shape=4, scale=mu/4) => E[y]=mu, Var=mu^2/4 (CV=0.5). A fixed seed
    // feeds the SAME draws to gam and mgcv, so disagreement is in the fit, not the
    // data.
    let n = 300usize;
    let mut rng = StdRng::seed_from_u64(20260602);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut eta_true = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        let eta = 2.0 + (PI * xi).sin() * (PI * zi).cos();
        let mu = eta.exp();
        let draw: f64 = Gamma::new(SHAPE, mu / SHAPE)
            .expect("valid Gamma params")
            .sample(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(draw.max(1e-12));
        eta_true.push(eta);
    }

    // ---- fit with gam: y ~ te(x, z, k=7), Gamma / log link, REML ------------
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode gamma dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=7)", &ds, &cfg).expect("gam gamma te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Gamma te(x, z)");
    };

    // gam linear predictor (log scale) at the training points: rebuilding the
    // frozen design and applying beta yields eta = design*beta directly, BEFORE
    // the log-link inverse — the scale on which the tensor penalty acts.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at training points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_mu: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model with mgcv (the mature reference) ---------------
    // family = Gamma(link = "log"), method = "REML"; emit the linear predictor.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ te(x, z, k = 7), data = df,
                 family = Gamma(link = "log"), method = "REML")
        emit("eta", as.numeric(predict(m, type = "link")))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_eta = r.vector("eta");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_eta.len(), n, "mgcv linear-predictor length mismatch");

    // ---- OBJECTIVE METRIC: truth recovery on the log (linear-predictor) scale
    let gam_err = rmse(&gam_eta, &eta_true);
    let mgcv_err = rmse(mgcv_eta, &eta_true);

    // Gamma-variance Pearson statistic with the TRUE shape: under Var=mu^2/shape,
    // E[ shape*(y-mu)^2/mu^2 ] = 1, so this lands near 1 when gam's fitted means
    // and the family's variance function are correct.
    let chi2_over_n: f64 = (0..n)
        .map(|i| {
            let mu = gam_mu[i].max(1e-12);
            let d = y[i] - mu;
            SHAPE * d * d / (mu * mu)
        })
        .sum::<f64>()
        / n as f64;

    // Context only (NOT a pass criterion): closeness of the two fitted surfaces.
    let rel_to_mgcv = relative_l2(&gam_eta, mgcv_eta);
    let corr_to_mgcv = pearson(&gam_eta, mgcv_eta);

    eprintln!(
        "te(x,z) Gamma/log: n={n} shape={SHAPE} mgcv_edf={mgcv_edf:.3} \
         rmse_to_truth(gam)={gam_err:.4} rmse_to_truth(mgcv)={mgcv_err:.4} \
         chi2/n(Gamma-var)={chi2_over_n:.3} \
         [context] rel_l2(gam,mgcv)={rel_to_mgcv:.4} pearson(gam,mgcv)={corr_to_mgcv:.5}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_mgcv_tensor_te_2d_gamma",
            "eta_rmse_to_truth",
            gam_err,
            "mgcv",
            mgcv_err,
        )
        .line()
    );

    // PRIMARY: gam recovers the true log-mean interaction surface. The wiggly part
    // sin(pi*x)*cos(pi*z) spans [-1,1] (range 2.0); with n=300 and CV=0.5 Gamma
    // noise a correct tensor / log-link fit stays well inside a fraction of that
    // span. We require RMSE(gam_eta, eta_true) < 0.30 — under 15% of the signal
    // range — mirroring the Poisson-tensor bound. A per-iteration penalty
    // mis-application or a botched link gradient distorts the surface past this.
    assert!(
        gam_err < 0.30,
        "gam should recover the true Gamma log-mean surface: rmse_to_truth={gam_err:.4} (bar 0.30)"
    );

    // MATCH-OR-BEAT: gam's truth-recovery error is no worse than mgcv's by more
    // than 10%, holding mgcv as an accuracy baseline on the objective metric.
    assert!(
        gam_err <= mgcv_err * 1.10,
        "gam's truth-recovery error must match-or-beat mgcv: \
         rmse_to_truth(gam)={gam_err:.4} vs mgcv*1.10={:.4}",
        mgcv_err * 1.10
    );

    // VARIANCE FUNCTION: the Gamma-variance Pearson statistic sits near 1. A
    // family that mis-modelled the mu^2 variance (e.g. treated the data as
    // constant-variance Gaussian) would drive this far from 1.
    assert!(
        (0.7..1.4).contains(&chi2_over_n),
        "Gamma-variance Pearson chi2/n outside (0.7,1.4): {chi2_over_n:.3}"
    );
}

/// REAL-DATA arm: `CW ~ te(FL, RW)` under Gamma(log) on the MASS `crabs`
/// morphometrics, where the truth is unknown. Quality is judged out-of-sample on
/// a deterministic split (every 4th row held out): fit on train, predict the
/// held-out carapace widths, and assert OBJECTIVE held-out Gamma deviance — both
/// an absolute beat-the-null bar and a match-or-beat against mgcv on that SAME
/// metric. mgcv is a baseline to beat, never a fit to reproduce.
#[test]
fn gam_tensor_te_2d_gamma_matches_mgcv_on_real_data() {
    init_parallelism();

    // ---- load crabs: FL, RW -> CW (all strictly positive mm) ---------------
    let ds = load_csvwith_inferred_schema(Path::new(CRABS_CSV)).expect("load crabs.csv");
    let col = ds.column_map();
    let fl_idx = col["FL"];
    let rw_idx = col["RW"];
    let cw_idx = col["CW"];
    let fl: Vec<f64> = ds.values.column(fl_idx).to_vec();
    let rw: Vec<f64> = ds.values.column(rw_idx).to_vec();
    let cw: Vec<f64> = ds.values.column(cw_idx).to_vec();
    let n = fl.len();
    assert!(n > 150, "crabs should have ~200 rows, got {n}");

    // ---- deterministic train/test split: every 4th row held out ------------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 130 && test_rows.len() > 40,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_fl: Vec<f64> = train_rows.iter().map(|&i| fl[i]).collect();
    let train_rw: Vec<f64> = train_rows.iter().map(|&i| rw[i]).collect();
    let train_cw: Vec<f64> = train_rows.iter().map(|&i| cw[i]).collect();
    let test_fl: Vec<f64> = test_rows.iter().map(|&i| fl[i]).collect();
    let test_rw: Vec<f64> = test_rows.iter().map(|&i| rw[i]).collect();
    let test_cw: Vec<f64> = test_rows.iter().map(|&i| cw[i]).collect();

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

    // ---- fit gam on TRAIN: CW ~ te(FL, RW, k=5), Gamma / log ---------------
    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("CW ~ te(FL, RW, k=5)", &train_ds, &cfg)
        .expect("gam gamma te fit on crabs train");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Gamma te(FL, RW)");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out rows: rebuild the frozen design, apply beta
    // for the log-mean predictor, exp() to the predicted carapace widths.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &row) in test_rows.iter().enumerate() {
        test_grid[[i, fl_idx]] = fl[row];
        test_grid[[i, rw_idx]] = rw[row];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at held-out points");
    let gam_test_eta: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    let gam_test_mu: Vec<f64> = gam_test_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model on TRAIN with mgcv, predict the SAME TEST -------
    // The harness exposes one data.frame per call, so the held-out FL/RW ride
    // along padded to the train length plus a count of how many to read back.
    let tr_len = train_fl.len();
    let r = run_r(
        &[
            Column::new("FL", &train_fl),
            Column::new("RW", &train_rw),
            Column::new("CW", &train_cw),
            Column::new("test_FL", &pad_to(&test_fl, tr_len)),
            Column::new("test_RW", &pad_to(&test_rw, tr_len)),
            Column::new("test_n", &vec![test_fl.len() as f64; tr_len]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(CW ~ te(FL, RW, k = 5), data = df,
                 family = Gamma(link = "log"), method = "REML")
        k <- df$test_n[1]
        newd <- data.frame(FL = df$test_FL[1:k], RW = df$test_RW[1:k])
        emit("test_mu", as.numeric(predict(m, newdata = newd, type = "response")))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_test_mu = r.vector("test_mu");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_test_mu.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- OBJECTIVE METRIC: held-out Gamma deviance (per held-out row) -------
    let gam_dev = gamma_deviance_mean(&test_cw, &gam_test_mu);
    let mgcv_dev = gamma_deviance_mean(&test_cw, mgcv_test_mu);

    // Absolute reference: deviance of the constant-mean (intercept-only) predictor
    // — predict the TRAIN mean carapace width everywhere.
    let train_mean = train_cw.iter().sum::<f64>() / train_cw.len() as f64;
    let null_mu = vec![train_mean; test_rows.len()];
    let null_dev = gamma_deviance_mean(&test_cw, &null_mu);

    // Context only (NOT a pass criterion): closeness of the two fitted surfaces.
    let rel_to_mgcv = relative_l2(&gam_test_mu, mgcv_test_mu);
    let corr_to_mgcv = pearson(&gam_test_mu, mgcv_test_mu);

    eprintln!(
        "crabs CW~te(FL,RW) Gamma/log held-out: n_train={} n_test={} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         dev_gam={gam_dev:.5} dev_mgcv={mgcv_dev:.5} dev_null={null_dev:.5} \
         [context] rel_l2(gam,mgcv)={rel_to_mgcv:.4} pearson(gam,mgcv)={corr_to_mgcv:.5}",
        train_rows.len(),
        test_rows.len(),
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_mgcv_tensor_te_2d_gamma::real_data",
            "held_out_gamma_deviance",
            gam_dev,
            "mgcv",
            mgcv_dev,
        )
        .line()
    );

    // PRIMARY accuracy: the FL+RW allometric structure must explain held-out
    // carapace width far better than the constant mean. We require gam's held-out
    // Gamma deviance to be at least 50% below the intercept-only baseline (the
    // signal is strong; this is a generous, tool-free bar that still fails loudly
    // if the tensor surface or link inversion is broken).
    assert!(
        gam_dev < null_dev * 0.5,
        "gam's held-out Gamma deviance {gam_dev:.5} must beat the constant-mean \
         baseline {null_dev:.5} (bar: < 50% of null)"
    );

    // BASELINE (match-or-beat, SAME held-out objective): gam's held-out Gamma
    // deviance is no worse than mgcv's by more than 10%. mgcv is a yardstick on
    // the same objective, never an output to reproduce.
    assert!(
        gam_dev <= mgcv_dev * 1.10,
        "gam held-out Gamma deviance {gam_dev:.5} exceeds mgcv {mgcv_dev:.5} * 1.10"
    );

    // complexity sanity: a single 2-D tensor of two covariates has modest edf.
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
