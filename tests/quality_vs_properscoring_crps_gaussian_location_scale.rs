//! End-to-end quality: gam's Gaussian *location-scale* predictive distribution
//! is judged by its **held-out continuous ranked probability score (CRPS)** — a
//! strictly-proper scoring rule that rewards a predictive distribution for being
//! both calibrated and sharp. This test asserts gam's OBJECTIVE distributional
//! quality on held-out data, not that gam reproduces any reference tool's fit.
//!
//! Objective metric asserted (CRPS is "lower is better"):
//!   1. NEAR-OPTIMALITY vs the ORACLE. The data are generated from a *known*
//!      heteroscedastic law `y ~ N(mu_true(x), sigma_true(x)^2)`. The smallest
//!      achievable expected CRPS is the ORACLE CRPS — the score of the true
//!      data-generating distribution `N(mu_true, sigma_true)`, the irreducible
//!      floor no estimator can beat in expectation. We assert gam's held-out mean
//!      CRPS is within a principled multiple of that floor: it has actually
//!      learned the right location-scale law, not merely "agreed with a peer".
//!   2. MATCH-OR-BEAT a HOMOSCEDASTIC baseline. A model that fits the mean but
//!      ignores the scale channel (constant sigma = RMS of the true sigmas) is
//!      the natural thing gam's location-scale machinery must beat. We assert
//!      gam's held-out mean CRPS is strictly below that baseline — proving the
//!      log-sigma smooth earns its keep on out-of-sample data.
//!
//! Role of `properscoring`. CRPS for a Gaussian `N(mu, sigma^2)` at observation
//! `y` has the exact closed form
//!
//!     CRPS(N(mu, sigma), y) = sigma * [ w*(2*Phi(w) - 1) + 2*phi(w) - 1/sqrt(pi) ],
//!     w = (y - mu) / sigma,
//!
//! and `properscoring.crps_gaussian` evaluates exactly this analytic quantity.
//! Here properscoring is used as the GROUND-TRUTH SCORING ENGINE: it scores all
//! three predictive distributions (gam's, the oracle's, the homoscedastic
//! baseline's) at the identical held-out (y, x), so the three CRPS numbers are
//! directly comparable on one independent, trusted ruler. gam is judged by where
//! its number lands relative to the oracle floor and the baseline — not by
//! whether it equals properscoring's recomputation of its own formula.
//!
//! For context we also recompute gam's CRPS from gam's OWN standard-normal CDF/PDF
//! primitives (`gam::inference::probability::{normal_cdf, normal_pdf}`, backed by
//! `statrs::erfc`) and `eprintln!` the cross-backend gap; that is diagnostic only
//! and is never a pass/fail criterion.
//!
//! What it verifies. The synthetic data is heteroscedastic (both mu(x) and
//! sigma(x) move with x), so gam must exercise *both* smooth channels — the mean
//! `s(x, k=8)` and the log-sigma `1 + s(x, k=8)` — to produce sensible predictive
//! distributions. We fit on 100 training rows, predict (mu, sigma) on a held-out
//! 50-row test set, and score generalization. The hold-out is INTERLEAVED (every
//! 3rd row of sorted x) so the test rows span the whole support [0, 1] and the
//! near-optimality verdict measures in-support calibration — not the irreducible
//! extrapolation error a contiguous-tail split would impose on any smoother.
//!
//! gam-side API (pinned by reading the source):
//!   * `fit_from_formula(.., FitConfig{ noise_formula: Some(..), .. })` produces a
//!     `FitResult::GaussianLocationScale`; the mean/log-sigma coefficient blocks
//!     carry `BlockRole::Location` / `BlockRole::Scale`, and the resolved designs
//!     live in `fit.meanspec_resolved` / `fit.noisespec_resolved`.
//!   * The noise link is `sigma = LOGB_SIGMA_FLOOR + exp(eta_scale)` with
//!     `LOGB_SIGMA_FLOOR = 0.01` (mirrors `families::sigma_link`, mgcv `gaulss(b=0.01)`).
//!   * The in-Rust path does NOT rescale `y`, so reconstructed mu/sigma are in raw
//!     response units — directly comparable to the y we generated.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::inference::probability::{normal_cdf, normal_pdf};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::path::Path;

/// MASS `gagurine` dataset (concentration of the glycosaminoglycan GAG in the
/// urine of `n = 314` children vs `Age` in years). Source: Venables & Ripley,
/// "Modern Applied Statistics with S" (the MASS R package). Mirrored at
/// `bench/datasets/gagurine.csv` (columns: rownames, Age, GAG).
const GAGURINE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/gagurine.csv");

/// gam's location-scale noise link floor: sigma = 0.01 + exp(eta_scale).
/// Mirrors `families::sigma_link::LOGB_SIGMA_FLOOR` (and mgcv `gaulss(b=0.01)`).
const LOGB_SIGMA_FLOOR: f64 = 0.01;

/// Closed-form CRPS of a Gaussian predictive `N(mu, sigma)` at observation `y`,
/// computed from gam's OWN standard-normal CDF/PDF primitives. Used here only to
/// `eprintln!` a diagnostic cross-backend gap against `properscoring.crps_gaussian`
/// (gam's `statrs::erfc` vs scipy/numpy erf); it is NOT part of any pass/fail
/// assertion. The objective quality verdict is rendered by properscoring scoring
/// gam, the oracle, and the homoscedastic baseline on one common ruler.
fn gam_crps_gaussian(y: f64, mu: f64, sigma: f64) -> f64 {
    const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3; // 1 / sqrt(pi)
    let w = (y - mu) / sigma;
    sigma * (w * (2.0 * normal_cdf(w) - 1.0) + 2.0 * normal_pdf(w) - INV_SQRT_PI)
}

#[test]
fn gam_gaussian_location_scale_crps_matches_properscoring() {
    init_parallelism();

    // ---- synthetic heteroscedastic recipe, seed = 1337 --------------------
    // n = 150, x ~ Uniform(0,1); y ~ N(sin(2*pi*x), [0.1 + 0.2*sin(2*pi*x)]^2).
    // A deterministic seeded LCG draws the uniforms and the Box-Muller normals so
    // the exact same y is reproducible and the (mu, sigma, y) we hand to Python
    // are byte-identical to gam's internal values.
    let n = 150usize;
    let two_pi = 2.0 * std::f64::consts::PI;

    let mut state: u64 = 1337;
    let mut next_unit = || -> f64 {
        // Numerical Recipes LCG; take the high bits for a uniform in [0,1).
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = (0..n).map(|_| next_unit()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Box-Muller standard normals from the same LCG stream (seed continues).
    let mut z: Vec<f64> = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit().max(1e-300);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }

    let mu_true = |t: f64| (two_pi * t).sin();
    let sigma_true = |t: f64| 0.1 + 0.2 * (two_pi * t).sin();
    let y: Vec<f64> = (0..n)
        .map(|i| mu_true(x[i]) + sigma_true(x[i]) * z[i])
        .collect();

    // ---- split: interleaved 100 train / 50 hold-out test ------------------
    // x is sorted, so the hold-out is taken as every 3rd row (i % 3 == 2) — 50
    // of 150 — which spreads the test set evenly across the WHOLE support
    // [0, 1] rather than a single tail. This judges held-out CALIBRATION
    // everywhere the heteroscedasticity lives, in-support, the same way the
    // sibling PIT and GAGurine tests interleave their splits. A contiguous
    // tail of sorted x is pure extrapolation (test x entirely outside the
    // training range): the scale spline can only extend its penalty-nullspace
    // trend beyond the data, so the oracle-relative CRPS there measures
    // irreducible extrapolation error that no smoother (gam, gamlss, or mgcv)
    // can beat — not predictive calibration. The near-optimality bar below is
    // a calibration claim, so the split must keep the test rows in-support.
    let is_test = |i: usize| i % 3 == 2;
    let n_test = (0..n).filter(|&i| is_test(i)).count(); // 50
    let n_train = n - n_test; // 100
    let x_train: Vec<f64> = (0..n).filter(|&i| !is_test(i)).map(|i| x[i]).collect();
    let y_train: Vec<f64> = (0..n).filter(|&i| !is_test(i)).map(|i| y[i]).collect();
    let x_test: Vec<f64> = (0..n).filter(|&i| is_test(i)).map(|i| x[i]).collect();
    let y_test: Vec<f64> = (0..n).filter(|&i| is_test(i)).map(|i| y[i]).collect();
    let x_train = x_train.as_slice();
    let y_train = y_train.as_slice();
    let x_test = x_test.as_slice();
    let y_test = y_test.as_slice();

    // ---- build the TRAINING dataset (column 0 = x, column 1 = y) ----------
    let headers: Vec<String> = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n_train)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x_train[i]),
                format!("{:.17e}", y_train[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode train data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    // ---- fit with gam: mu ~ s(x, k=8), log-sigma ~ 1 + s(x, k=8) ----------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(x, k=8)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=8)", &ds, &cfg).expect("gam location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result
    else {
        panic!("expected a Gaussian location-scale fit");
    };

    let beta_location = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location (mean) block present")
        .beta
        .clone();
    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale (log-sigma) block present")
        .beta
        .clone();

    // ---- predict (mu, sigma) on the held-out TEST points -------------------
    // Rebuild the frozen mean / log-sigma designs at the test x and apply each
    // block's coefficients. mu = X_mean*beta; sigma = floor + exp(X_scale*beta).
    let mut grid = Array2::<f64>::zeros((n_test, ncols));
    for (i, &t) in x_test.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let mean_design = build_term_collection_design(grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at test points");
    let scale_design = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma design at test points");

    let mu_test: Vec<f64> = mean_design.design.apply(&beta_location).to_vec();
    let eta_sigma_test: Vec<f64> = scale_design.design.apply(&beta_scale).to_vec();
    let sigma_test: Vec<f64> = eta_sigma_test
        .iter()
        .map(|&e| LOGB_SIGMA_FLOOR + e.exp())
        .collect();

    assert_eq!(mu_test.len(), n_test);
    assert_eq!(sigma_test.len(), n_test);
    for &s in &sigma_test {
        assert!(
            s > 0.0 && s.is_finite(),
            "predicted sigma must be positive/finite: {s}"
        );
    }

    // ---- ORACLE predictive: the true data-generating distribution ----------
    // N(mu_true(x_test), sigma_true(x_test)). Its expected CRPS is the irreducible
    // floor; no estimator can beat it in expectation. This is the optimality
    // yardstick for gam's distributional fit.
    let mu_oracle: Vec<f64> = x_test.iter().map(|&t| mu_true(t)).collect();
    let sigma_oracle: Vec<f64> = x_test
        .iter()
        .map(|&t| sigma_true(t).max(LOGB_SIGMA_FLOOR))
        .collect();

    // ---- HOMOSCEDASTIC baseline: right mean, scale channel switched off -----
    // Same true mean, but a single constant sigma = RMS of the true sigmas (the
    // best constant-variance Gaussian for these data). This is the natural model
    // gam's location-scale machinery must out-score on held-out CRPS to justify
    // fitting the log-sigma smooth at all.
    let sigma_const = {
        let ms: f64 = x_test.iter().map(|&t| sigma_true(t).powi(2)).sum::<f64>() / n_test as f64;
        ms.sqrt()
    };
    let mu_base: Vec<f64> = mu_oracle.clone();
    let sigma_base: Vec<f64> = vec![sigma_const; n_test];

    // ---- gam-side CRPS via gam's own normal_cdf / normal_pdf (diagnostic) ---
    let gam_crps_self: Vec<f64> = (0..n_test)
        .map(|i| gam_crps_gaussian(y_test[i], mu_test[i], sigma_test[i]))
        .collect();

    // ---- properscoring as the GROUND-TRUTH scorer for all three predictives -
    // One independent, trusted CRPS ruler scores gam, the oracle, and the
    // homoscedastic baseline at the IDENTICAL held-out (y, x); the three returned
    // means are directly comparable.
    let ref_py = run_python(
        &[
            Column::new("y_test", y_test),
            Column::new("mu_gam", &mu_test),
            Column::new("sigma_gam", &sigma_test),
            Column::new("mu_oracle", &mu_oracle),
            Column::new("sigma_oracle", &sigma_oracle),
            Column::new("mu_base", &mu_base),
            Column::new("sigma_base", &sigma_base),
        ],
        r#"
import properscoring as ps
y = np.asarray(df["y_test"], dtype=float)
def crps(mu, sig):
    return ps.crps_gaussian(y,
                            mu=np.asarray(df[mu], dtype=float),
                            sig=np.asarray(df[sig], dtype=float))
emit("crps_gam", crps("mu_gam", "sigma_gam"))
emit("crps_oracle", crps("mu_oracle", "sigma_oracle"))
emit("crps_base", crps("mu_base", "sigma_base"))
"#,
    );
    let ps_crps_gam = ref_py.vector("crps_gam");
    let ps_crps_oracle = ref_py.vector("crps_oracle");
    let ps_crps_base = ref_py.vector("crps_base");
    assert_eq!(ps_crps_gam.len(), n_test, "properscoring gam CRPS length");
    assert_eq!(
        ps_crps_oracle.len(),
        n_test,
        "properscoring oracle CRPS length"
    );
    assert_eq!(
        ps_crps_base.len(),
        n_test,
        "properscoring baseline CRPS length"
    );

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mean_gam = mean(ps_crps_gam);
    let mean_oracle = mean(ps_crps_oracle);
    let mean_base = mean(ps_crps_base);

    // Diagnostic only: cross-backend agreement of gam's own CRPS vs properscoring,
    // at the identical (y, mu, sigma). Never a pass/fail criterion.
    let mean_gam_self = mean(&gam_crps_self);
    let self_vs_ps = max_abs_diff(&gam_crps_self, ps_crps_gam);

    eprintln!(
        "crps location-scale held-out: n_train={n_train} n_test={n_test} \
         mean_crps(gam)={mean_gam:.6} mean_crps(oracle)={mean_oracle:.6} \
         mean_crps(homoscedastic)={mean_base:.6} sigma_const={sigma_const:.4} \
         | diagnostic gam_self_crps={mean_gam_self:.6} self_vs_ps_maxabs={self_vs_ps:.3e}"
    );

    // OBJECTIVE 1 — near-optimality. gam's held-out mean CRPS must be within a
    // principled multiple of the irreducible oracle floor. With both smooth
    // channels fit on 100 rows the realized ratio is comfortably under 1.25; a
    // larger gap means gam's predictive distribution is mis-calibrated or
    // over/under-dispersed, i.e. a genuine quality shortfall.
    assert!(
        mean_gam <= 1.25 * mean_oracle,
        "gam held-out mean CRPS not near the oracle floor: \
         gam={mean_gam:.6} oracle={mean_oracle:.6} ratio={:.3} (bar=1.25)",
        mean_gam / mean_oracle
    );

    // OBJECTIVE 2 — beat the homoscedastic baseline. Modelling the scale channel
    // must pay off out-of-sample: gam's mean CRPS must be strictly below that of
    // the best constant-sigma Gaussian. If gam cannot beat ignoring sigma(x), the
    // location-scale fit added nothing of value.
    assert!(
        mean_gam < mean_base,
        "gam location-scale did not beat the homoscedastic baseline on held-out CRPS: \
         gam={mean_gam:.6} homoscedastic={mean_base:.6}"
    );
}

/// Real-data arm of the SAME Gaussian location-scale + CRPS capability, on the
/// MASS `gagurine` dataset (GAG concentration in children's urine vs `Age`).
///
/// Truth is unknown here, so the objective verdict is HELD-OUT predictive CRPS.
/// The GAG ~ Age relationship is strongly heteroscedastic: at young ages GAG is
/// both higher and far more variable than in older children, so a location-scale
/// model that fits BOTH a mean smooth `s(Age)` and a log-sigma smooth
/// `1 + s(Age)` should out-predict any model that ignores the scale channel.
///
/// What this arm asserts (all on a deterministic, every-4th-row held-out split,
/// scored by `properscoring.crps_gaussian` as the one independent CRPS ruler):
///   PRIMARY (objective, calibration). gam's held-out mean CRPS must clear an
///     ABSOLUTE bar tied to the held-out response spread — gam's predictive
///     distributions are genuinely sharp+calibrated, not merely "near a peer".
///   BASELINE (match-or-beat). A homoscedastic Gaussian (the held-out-data mean
///     of gam's location predictions, with a single constant sigma = RMS of
///     gam's predicted sigmas) is the natural thing the scale channel must beat.
///     gam's location-scale mean CRPS must be <= that baseline minus a margin,
///     proving modelling sigma(Age) pays off out-of-sample. The homoscedastic
///     reference is a baseline to beat, never an output to replicate.
#[test]
fn gam_gaussian_location_scale_crps_matches_properscoring_on_real_data() {
    init_parallelism();

    // ---- load gagurine (Age -> GAG), drop the rownames index column --------
    let ds = load_csvwith_inferred_schema(Path::new(GAGURINE_CSV)).expect("load gagurine.csv");
    let colmap = ds.column_map();
    let age_idx = colmap["Age"];
    let gag_idx = colmap["GAG"];
    let age_all: Vec<f64> = ds.values.column(age_idx).to_vec();
    let gag_all: Vec<f64> = ds.values.column(gag_idx).to_vec();
    let n = age_all.len();
    assert!(n > 300, "gagurine should have ~314 rows, got {n}");

    // ---- deterministic train/test split: every 4th row held out -----------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    let n_train = train_rows.len();
    let n_test = test_rows.len();
    assert!(
        n_train > 200 && n_test > 60,
        "split sizes: train={n_train} test={n_test}"
    );

    let age_test: Vec<f64> = test_rows.iter().map(|&i| age_all[i]).collect();
    let gag_test: Vec<f64> = test_rows.iter().map(|&i| gag_all[i]).collect();

    // ---- build a TRAINING dataset (columns Age, GAG) from the encoded rows -
    let headers: Vec<String> = vec!["Age".to_string(), "GAG".to_string()];
    let rows: Vec<csv::StringRecord> = train_rows
        .iter()
        .map(|&i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", age_all[i]),
                format!("{:.17e}", gag_all[i]),
            ])
        })
        .collect();
    let train_ds =
        encode_recordswith_inferred_schema(headers, rows).expect("encode gagurine train data");
    let train_col = train_ds.column_map();
    let train_age_idx = train_col["Age"];
    let ncols = train_ds.headers.len();

    // ---- fit gam: GAG ~ s(Age, k=10), log-sigma ~ 1 + s(Age, k=10) ---------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(Age, k=10)".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("GAG ~ s(Age, k=10)", &train_ds, &cfg).expect("gam location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result
    else {
        panic!("expected a Gaussian location-scale fit");
    };

    let beta_location = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location (mean) block present")
        .beta
        .clone();
    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale (log-sigma) block present")
        .beta
        .clone();

    // ---- predict (mu, sigma) at the held-out Age points --------------------
    let mut grid = Array2::<f64>::zeros((n_test, ncols));
    for (i, &t) in age_test.iter().enumerate() {
        grid[[i, train_age_idx]] = t;
    }
    let mean_design = build_term_collection_design(grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at test points");
    let scale_design = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma design at test points");

    let mu_test: Vec<f64> = mean_design.design.apply(&beta_location).to_vec();
    let eta_sigma_test: Vec<f64> = scale_design.design.apply(&beta_scale).to_vec();
    let sigma_test: Vec<f64> = eta_sigma_test
        .iter()
        .map(|&e| LOGB_SIGMA_FLOOR + e.exp())
        .collect();

    assert_eq!(mu_test.len(), n_test);
    assert_eq!(sigma_test.len(), n_test);
    for &s in &sigma_test {
        assert!(
            s > 0.0 && s.is_finite(),
            "predicted sigma must be positive/finite: {s}"
        );
    }

    // ---- HOMOSCEDASTIC baseline: gam's mean predictions, but a single ------
    // constant sigma = RMS of gam's predicted sigmas. This isolates the value
    // of the SCALE channel: same locations, scale knowledge thrown away.
    let sigma_const = {
        let ms: f64 = sigma_test.iter().map(|s| s * s).sum::<f64>() / n_test as f64;
        ms.sqrt()
    };
    let mu_base: Vec<f64> = mu_test.clone();
    let sigma_base: Vec<f64> = vec![sigma_const; n_test];

    // ---- diagnostic: gam's own-backend CRPS at the held-out (y, mu, sigma) -
    let gam_crps_self: Vec<f64> = (0..n_test)
        .map(|i| gam_crps_gaussian(gag_test[i], mu_test[i], sigma_test[i]))
        .collect();

    // ---- properscoring scores gam + the homoscedastic baseline on one ruler
    let ref_py = run_python(
        &[
            Column::new("y_test", &gag_test),
            Column::new("mu_gam", &mu_test),
            Column::new("sigma_gam", &sigma_test),
            Column::new("mu_base", &mu_base),
            Column::new("sigma_base", &sigma_base),
        ],
        r#"
import properscoring as ps
y = np.asarray(df["y_test"], dtype=float)
def crps(mu, sig):
    return ps.crps_gaussian(y,
                            mu=np.asarray(df[mu], dtype=float),
                            sig=np.asarray(df[sig], dtype=float))
emit("crps_gam", crps("mu_gam", "sigma_gam"))
emit("crps_base", crps("mu_base", "sigma_base"))
"#,
    );
    let ps_crps_gam = ref_py.vector("crps_gam");
    let ps_crps_base = ref_py.vector("crps_base");
    assert_eq!(ps_crps_gam.len(), n_test, "properscoring gam CRPS length");
    assert_eq!(
        ps_crps_base.len(),
        n_test,
        "properscoring baseline CRPS length"
    );

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mean_gam = mean(ps_crps_gam);
    let mean_base = mean(ps_crps_base);

    // Held-out response spread sets the ABSOLUTE calibration bar. The
    // information-free predictor is the MARGINAL Gaussian N(mean(y), sd(y)) — it
    // knows nothing about Age. Its expected CRPS, for y itself drawn ~ N(m, s),
    // is exactly E[CRPS(N(0,1), Z)] * sd(y) with Z ~ N(0,1), and that constant is
    // the closed form 1/sqrt(pi) = 0.5641895835. A genuinely informative
    // location-scale fit must come in well under that marginal-predictor score.
    let y_mean = mean(&gag_test);
    let y_sd = {
        let ss: f64 = gag_test.iter().map(|v| (v - y_mean) * (v - y_mean)).sum();
        (ss / n_test as f64).sqrt()
    };
    let crps_naive = 0.564_189_583_547_756_3 * y_sd; // E[CRPS(N(0,1), Z)] = 1/sqrt(pi)

    let mean_gam_self = mean(&gam_crps_self);
    let self_vs_ps = max_abs_diff(&gam_crps_self, ps_crps_gam);

    eprintln!(
        "gagurine GAG~s(Age) location-scale held-out: n_train={n_train} n_test={n_test} \
         mean_crps(gam)={mean_gam:.6} mean_crps(homoscedastic)={mean_base:.6} \
         sigma_const={sigma_const:.4} y_sd={y_sd:.4} crps_naive={crps_naive:.4} \
         | diagnostic gam_self_crps={mean_gam_self:.6} self_vs_ps_maxabs={self_vs_ps:.3e}"
    );

    // PRIMARY — absolute held-out calibration. gam's mean CRPS must beat the
    // information-free predictor (constant mean + marginal sd) by a clear margin:
    // it has learned a sharp, calibrated Age-conditional law, not the marginal.
    assert!(
        mean_gam <= 0.75 * crps_naive,
        "gam held-out mean CRPS not sharp vs the information-free predictor: \
         gam={mean_gam:.6} crps_naive={crps_naive:.6} (bar=0.75*crps_naive={:.6})",
        0.75 * crps_naive
    );

    // BASELINE — beat the homoscedastic model. Modelling sigma(Age) must pay off
    // out-of-sample: gam's location-scale CRPS must beat the same-mean,
    // constant-sigma Gaussian by a margin. If not, the log-sigma smooth is dead
    // weight on this real, heteroscedastic dataset.
    assert!(
        mean_gam <= mean_base - 1e-3 * crps_naive,
        "gam location-scale did not beat the homoscedastic baseline on held-out CRPS: \
         gam={mean_gam:.6} homoscedastic={mean_base:.6}"
    );
}
