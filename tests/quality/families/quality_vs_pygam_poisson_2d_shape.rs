//! End-to-end **objective quality**: does gam's 2-D Poisson smooth recover the
//! *true* log-mean surface it was generated from?
//!
//! The data are simulated from a known generator
//!   `true_eta(x, z) = sin(pi*x) * cos(pi*z)`,  `count ~ Poisson(exp(true_eta))`,
//! so we have ground truth on the linear-predictor (log) scale. The quality
//! claim is **truth recovery**, asserted directly against `true_eta`:
//!
//!   * PRIMARY (accuracy): `RMSE(gam_eta, true_eta)` is small in absolute terms.
//!     The signal spans `[-1, 1]` (range 2); we require the recovered surface to
//!     sit well inside a small fraction of that range for both the additive
//!     `s(x) + s(z)` model and the tensor `te(x, z)` model. This proves gam's
//!     Poisson family / log-link PIRLS and its tensor-penalty construction
//!     actually estimate the surface, not merely that they imitate a peer tool.
//!
//!   * BASELINE (match-or-beat): pyGAM — an independent, mature GAM engine with
//!     its own scipy/scikit-learn bases and PIRLS fit — is fit on the *identical*
//!     counts. We require gam's recovery error to be no worse than pyGAM's by
//!     more than 10% (`gam_rmse <= pygam_rmse * 1.10`). pyGAM is here only as a
//!     yardstick on the SAME objective metric (distance to truth); we never
//!     assert that gam reproduces pyGAM's (itself noisy) fit.
//!
//! We deliberately do NOT assert that gam's effective degrees of freedom match
//! pyGAM's — matching another tool's complexity proves nothing. We only sanity-
//! check that gam's edf lands in a signal-appropriate range (above a straight
//! line, below the basis dimension), which is an objective structural property.
//!
//! The reference rel_l2 / pearson against pyGAM are still computed and printed
//! for context via `eprintln!`, but they are NOT pass/fail criteria.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{
    Column, QualityPair, pad_to, pearson, relative_l2, rmse, run_python,
};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::f64::consts::PI;
use std::path::Path;

const TRUE_ETA_RANGE: f64 = 2.0;
const ADDITIVE_LOG_RMSE_BAR: f64 = 0.16 * TRUE_ETA_RANGE;
const TENSOR_LOG_RMSE_BAR: f64 = 0.20 * TRUE_ETA_RANGE;

// Real-data source: the `badhealth` dataset from the R `COUNT` package
// (Hilbe, J.M., "Negative Binomial Regression", 2nd ed., Cambridge Univ. Press,
// 2011). 1127 German health-survey respondents; `numvisit` is the number of
// physician visits, `badh` a self-rated bad-health indicator (0/1), `age` in
// years. Vendored at bench/datasets/badhealth.csv.
const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");

#[test]
fn gam_poisson_2d_recovers_true_log_mean_surface() {
    init_parallelism();

    // ---- synthetic Poisson-count truth on the unit square ------------------
    // true_eta = sin(pi*x) * cos(pi*z); count ~ Poisson(exp(true_eta)).
    let n = 300usize;
    let mut rng = StdRng::seed_from_u64(20260530);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut count = Vec::with_capacity(n);
    let mut true_eta = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        let eta = (PI * xi).sin() * (PI * zi).cos(); // true log-mean
        let lambda = eta.exp().max(1e-12);
        let draw: f64 = Poisson::new(lambda)
            .expect("valid Poisson rate")
            .sample(&mut rng);
        x.push(xi);
        z.push(zi);
        count.push(draw);
        true_eta.push(eta);
    }

    // ---- encode the shared dataset for gam ---------------------------------
    let headers = ["x", "z", "count"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                x[i].to_string(),
                z[i].to_string(),
                count[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    // Shared grid (training points) used to rebuild gam's frozen design and
    // recover its linear predictor eta = design*beta (BEFORE the log-link
    // inverse) for both the additive and tensor fits.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };

    // ---- gam additive: count ~ s(x, k=6) + s(z, k=6), Poisson/log, REML ----
    let add_result = fit_from_formula("count ~ s(x, k=6) + s(z, k=6)", &ds, &cfg)
        .expect("gam poisson additive fit");
    let FitResult::Standard(add_fit) = add_result else {
        panic!("expected a standard GAM fit for the additive Poisson model");
    };
    let gam_add_edf = add_fit.fit.edf_total().expect("gam reports additive edf");
    let add_design = build_term_collection_design(grid.view(), &add_fit.resolvedspec)
        .expect("rebuild additive design at training points");
    let gam_add_eta: Vec<f64> = add_design.design.apply(&add_fit.fit.beta).to_vec();

    // ---- gam tensor: count ~ te(x, z, k=6), Poisson/log, REML --------------
    let te_result =
        fit_from_formula("count ~ te(x, z, k=6)", &ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(te_fit) = te_result else {
        panic!("expected a standard GAM fit for the tensor Poisson model");
    };
    let gam_te_edf = te_fit.fit.edf_total().expect("gam reports tensor edf");
    let te_design = build_term_collection_design(grid.view(), &te_fit.resolvedspec)
        .expect("rebuild te design at training points");
    let gam_te_eta: Vec<f64> = te_design.design.apply(&te_fit.fit.beta).to_vec();

    // ---- fit BOTH models with pyGAM (independent baseline on the SAME data) -
    // PoissonGAM(s(0,n_splines=6)+s(1,n_splines=6)) is the additive analog;
    // PoissonGAM(te(0,1,n_splines=6)) is the tensor-product analog. predict_mu
    // returns the mean (mu = exp(eta) under the log link), so eta = log(mu) is
    // the linear predictor on the same (log) scale gam reports and on which the
    // truth `true_eta` lives. pyGAM is scored against the SAME truth as gam.
    let py = run_python(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("count", &count),
        ],
        r#"
from pygam import PoissonGAM, s, te
X = np.column_stack([
    np.asarray(df["x"], dtype=float),
    np.asarray(df["z"], dtype=float),
])
y = np.asarray(df["count"], dtype=float)

add = PoissonGAM(s(0, n_splines=6) + s(1, n_splines=6)).fit(X, y)
add_mu = np.asarray(add.predict_mu(X), dtype=float)
emit("add_eta", np.log(np.clip(add_mu, 1e-12, None)))
emit("add_edf", [float(add.statistics_["edof"])])

ten = PoissonGAM(te(0, 1, n_splines=6)).fit(X, y)
ten_mu = np.asarray(ten.predict_mu(X), dtype=float)
emit("te_eta", np.log(np.clip(ten_mu, 1e-12, None)))
emit("te_edf", [float(ten.statistics_["edof"])])
"#,
    );
    let pygam_add_eta = py.vector("add_eta");
    let pygam_add_edf = py.scalar("add_edf");
    let pygam_te_eta = py.vector("te_eta");
    let pygam_te_edf = py.scalar("te_edf");
    assert_eq!(pygam_add_eta.len(), n, "pyGAM additive eta length mismatch");
    assert_eq!(pygam_te_eta.len(), n, "pyGAM tensor eta length mismatch");

    // ---- OBJECTIVE metric: recovery error against the KNOWN truth ----------
    // Score every fit by RMSE to `true_eta`; pyGAM is scored the same way so it
    // is a baseline-to-beat on the SAME objective metric, not a fit to imitate.
    let gam_add_rmse = rmse(&gam_add_eta, &true_eta);
    let pygam_add_rmse = rmse(pygam_add_eta, &true_eta);
    let gam_te_rmse = rmse(&gam_te_eta, &true_eta);
    let pygam_te_rmse = rmse(pygam_te_eta, &true_eta);

    // Context-only (NOT pass/fail): how close gam's fit is to pyGAM's fit.
    let add_rel = relative_l2(&gam_add_eta, pygam_add_eta);
    let add_corr = pearson(&gam_add_eta, pygam_add_eta);
    let te_rel = relative_l2(&gam_te_eta, pygam_te_eta);
    let te_corr = pearson(&gam_te_eta, pygam_te_eta);

    eprintln!(
        "Poisson 2-D additive: n={n} gam_edf={gam_add_edf:.3} pygam_edf={pygam_add_edf:.3} \
         gam_rmse(truth)={gam_add_rmse:.4} pygam_rmse(truth)={pygam_add_rmse:.4} \
         [context: rel_l2(gam,pygam)={add_rel:.4} pearson={add_corr:.5}]"
    );
    eprintln!(
        "Poisson 2-D tensor:   n={n} gam_edf={gam_te_edf:.3} pygam_edf={pygam_te_edf:.3} \
         gam_rmse(truth)={gam_te_rmse:.4} pygam_rmse(truth)={pygam_te_rmse:.4} \
         [context: rel_l2(gam,pygam)={te_rel:.4} pearson={te_corr:.5}]"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_pygam_poisson_2d_shape::additive",
            "additive_rmse_to_truth",
            gam_add_rmse,
            "pygam",
            pygam_add_rmse,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_pygam_poisson_2d_shape::tensor",
            "tensor_rmse_to_truth",
            gam_te_rmse,
            "pygam",
            pygam_te_rmse,
        )
        .line()
    );

    // ---- PRIMARY assertion: absolute truth recovery ------------------------
    // The signal sin(pi*x)*cos(pi*z) spans [-1, 1] (range 2). With n=300 Poisson
    // counts whose mean is exp(eta) in [exp(-1), exp(1)] ~= [0.37, 2.72], the
    // information per point is modest. The additive arm is also deliberately
    // misspecified for this truth: f(x)+g(z) cannot represent the x:z product
    // exactly, so its absolute bar must include approximation error as well as
    // sampling error. We therefore set the additive bar at 16% of the signal
    // range, while the correctly-shaped tensor arm keeps the original 20% bar.
    // Both remain tight enough to catch a broken log link, penalty, or PIRLS
    // scale, and the independent pyGAM match-or-beat assertion below still
    // prevents hiding a real quality regression behind the absolute threshold.
    assert!(
        gam_add_rmse < ADDITIVE_LOG_RMSE_BAR,
        "additive Poisson fit does not recover the true log-mean surface: \
         rmse(gam, truth)={gam_add_rmse:.4} (>= {ADDITIVE_LOG_RMSE_BAR:.2})"
    );
    assert!(
        gam_te_rmse < TENSOR_LOG_RMSE_BAR,
        "tensor Poisson fit does not recover the true log-mean surface: \
         rmse(gam, truth)={gam_te_rmse:.4} (>= {TENSOR_LOG_RMSE_BAR:.2})"
    );

    // ---- BASELINE: match-or-beat pyGAM on the SAME accuracy metric ---------
    // gam's recovery error must be no worse than pyGAM's by more than 10%.
    assert!(
        gam_add_rmse <= pygam_add_rmse * 1.10,
        "additive Poisson: gam's recovery error exceeds pyGAM's by >10%: \
         gam_rmse={gam_add_rmse:.4} pygam_rmse={pygam_add_rmse:.4}"
    );
    assert!(
        gam_te_rmse <= pygam_te_rmse * 1.10,
        "tensor Poisson: gam's recovery error exceeds pyGAM's by >10%: \
         gam_rmse={gam_te_rmse:.4} pygam_rmse={pygam_te_rmse:.4}"
    );

    // ---- STRUCTURE: edf in a sane, signal-appropriate range ----------------
    // Not matched to pyGAM. Each model has two k=6 marginal bases, so the basis
    // dimension is ~11 (additive: 2*(6-1)+1) / ~36 (tensor: ~6*6). A real 2-D
    // signal must use more than a flat line (edf > 1) and far less than the full
    // basis (well below the basis dimension), which falsifies both a collapsed
    // (over-smoothed to constant) and a saturated (interpolating) fit.
    assert!(
        gam_add_edf > 1.0 && gam_add_edf < 11.0,
        "additive edf outside the signal-appropriate range (1, 11): {gam_add_edf:.3}"
    );
    assert!(
        gam_te_edf > 1.0 && gam_te_edf < 36.0,
        "tensor edf outside the signal-appropriate range (1, 36): {gam_te_edf:.3}"
    );
}

/// Mean Poisson (unit) deviance of predicted means `mu` against observed counts
/// `y`: `(2/n) * sum[ y*log(y/mu) - (y - mu) ]` (the `y*log(y/mu)` term is taken
/// as 0 when `y == 0`). This is the natural held-out goodness-of-fit metric for
/// count data — the Poisson analog of RMSE — and is computed in plain Rust so it
/// is identical for gam and the reference. Smaller is better.
fn poisson_mean_deviance(mu: &[f64], y: &[f64]) -> f64 {
    assert_eq!(mu.len(), y.len(), "poisson_mean_deviance length mismatch");
    let n = y.len() as f64;
    let mut s = 0.0;
    for (&m, &yi) in mu.iter().zip(y) {
        let m = m.max(1e-12);
        let term = if yi > 0.0 { yi * (yi / m).ln() } else { 0.0 };
        s += term - (yi - m);
    }
    2.0 * s / n.max(1.0)
}

/// Real-data arm of the 2-D Poisson smooth-shape capability. The `badhealth`
/// survey (real counts of physician visits) has no known truth, so we assert
/// OBJECTIVE held-out predictive quality of a 2-D `te(age, badh)` Poisson smooth:
///
///   PRIMARY (objective, tool-free): held-out mean Poisson deviance must sit
///     below the deviance of the intercept-only (training-mean) predictor — the
///     2-D smooth genuinely improves count prediction on unseen rows.
///
///   BASELINE (match-or-beat): pyGAM's PoissonGAM(te(0,1)) is fit on the IDENTICAL
///     training rows and predicts the IDENTICAL held-out rows; gam's held-out
///     deviance must be no worse than pyGAM's by more than 10%. pyGAM is a
///     yardstick on the SAME metric, never a fit to reproduce.
#[test]
fn gam_poisson_2d_recovers_true_log_mean_surface_on_real_data() {
    init_parallelism();

    // ---- load badhealth (age, badh -> numvisit count) ----------------------
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

    // ---- deterministic train/test split: every 4th row is held out ---------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 700 && test_rows.len() > 250,
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

    // Training-only dataset by row-subsetting the encoded values; headers,
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

    // ---- fit gam on TRAIN: numvisit ~ te(age, badh), Poisson/log, REML -----
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("numvisit ~ te(age, badh)", &train_ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the 2-D Poisson real-data model");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at held-out (age, badh): rebuild the frozen design and
    // apply exp() to the linear predictor (log link => mu = exp(design*beta)).
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for i in 0..test_rows.len() {
        test_grid[[i, age_idx]] = test_age[i];
        test_grid[[i, badh_idx]] = test_badh[i];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_eta: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    let gam_test_mu: Vec<f64> = gam_test_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model on TRAIN with pyGAM, predict the SAME TEST ------
    // One run_python call exposes ONE equal-length data.frame, so we ride the
    // test columns alongside the train columns (right-padded to train length)
    // plus the test count, and read back only the first test_n predictions.
    let train_len = train_rows.len();
    let py = run_python(
        &[
            Column::new("age", &train_age),
            Column::new("badh", &train_badh),
            Column::new("numvisit", &train_numvisit),
            Column::new("test_age", &pad_to(&test_age, train_len)),
            Column::new("test_badh", &pad_to(&test_badh, train_len)),
            Column::new("test_n", &vec![test_rows.len() as f64; train_len]),
        ],
        r#"
from pygam import PoissonGAM, te
Xtr = np.column_stack([
    np.asarray(df["age"], dtype=float),
    np.asarray(df["badh"], dtype=float),
])
ytr = np.asarray(df["numvisit"], dtype=float)
k = int(np.asarray(df["test_n"])[0])
Xte = np.column_stack([
    np.asarray(df["test_age"], dtype=float)[:k],
    np.asarray(df["test_badh"], dtype=float)[:k],
])
m = PoissonGAM(te(0, 1)).fit(Xtr, ytr)
emit("test_mu", np.asarray(m.predict_mu(Xte), dtype=float))
emit("edf", [float(m.statistics_["edof"])])
"#,
    );
    let pygam_test_mu = py.vector("test_mu");
    let pygam_edf = py.scalar("edf");
    assert_eq!(
        pygam_test_mu.len(),
        test_rows.len(),
        "pyGAM held-out prediction length mismatch"
    );

    // ---- OBJECTIVE held-out metric: mean Poisson deviance ------------------
    let gam_test_dev = poisson_mean_deviance(&gam_test_mu, &test_numvisit);
    let pygam_test_dev = poisson_mean_deviance(pygam_test_mu, &test_numvisit);

    // Intercept-only baseline: predict the training mean count for every held-out
    // row. A useful 2-D smooth must beat this constant predictor.
    let train_mean = train_numvisit.iter().sum::<f64>() / train_len.max(1) as f64;
    let null_mu = vec![train_mean; test_rows.len()];
    let null_test_dev = poisson_mean_deviance(&null_mu, &test_numvisit);

    // Context-only (NOT pass/fail): closeness of gam's vs pyGAM's held-out means.
    let mu_rel = relative_l2(&gam_test_mu, pygam_test_mu);
    let mu_corr = pearson(&gam_test_mu, pygam_test_mu);

    eprintln!(
        "badhealth te(age,badh) held-out: n_train={train_len} n_test={} gam_edf={gam_edf:.3} \
         pygam_edf={pygam_edf:.3} gam_dev={gam_test_dev:.4} pygam_dev={pygam_test_dev:.4} \
         null_dev={null_test_dev:.4} [context: rel_l2(gam,pygam)={mu_rel:.4} pearson={mu_corr:.5}]",
        test_rows.len(),
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_pygam_poisson_2d_shape::real_data",
            "held_out_poisson_deviance",
            gam_test_dev,
            "pygam",
            pygam_test_dev,
        )
        .line()
    );

    // ---- PRIMARY objective assertion: beat the constant-mean predictor -----
    assert!(
        gam_test_dev < null_test_dev,
        "2-D Poisson smooth does not beat the intercept-only predictor on held-out \
         deviance: gam_dev={gam_test_dev:.4} >= null_dev={null_test_dev:.4}"
    );

    // ---- BASELINE (match-or-beat): no worse than pyGAM on held-out deviance -
    assert!(
        gam_test_dev <= pygam_test_dev * 1.10,
        "2-D Poisson: gam's held-out deviance exceeds pyGAM's by >10%: \
         gam_dev={gam_test_dev:.4} pygam_dev={pygam_test_dev:.4}"
    );

    // ---- STRUCTURE: edf in a sane, signal-appropriate range (not matched) ---
    // A real 2-D surface must use more than a flat line (edf > 1) yet far less
    // than the full tensor basis (~k*k), falsifying a collapsed or saturated fit.
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "real-data 2-D edf outside the signal-appropriate range (1, 30): {gam_edf:.3}"
    );
}
