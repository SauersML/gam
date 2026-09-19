//! End-to-end quality: gam's Gaussian location-scale fit with a *cyclic*
//! (periodic) smooth in BOTH the mean (mu) and the log-scale (sigma) block must
//! RECOVER the known generating functions on a periodic, heteroscedastic signal.
//!
//! Objective metric asserted (TRUTH RECOVERY)
//! ------------------------------------------
//! The data are generated from a KNOWN truth,
//!   x = seq(0, 2*pi, length = 150),  y ~ N(mu*(x), sigma*(x)^2),
//! with x circular (0 and 2*pi identified). The PRIMARY claim is that gam's
//! constrained cyclic smooths recover those generating functions, measured on
//! 50 equally-spaced grid points in [0, 2*pi) as an absolute, reference-free
//! accuracy bar. Passing means gam fit the TRUE periodic mean and the TRUE
//! periodic (log-)scale, not that it imitated another tool's (possibly equally
//! wrong) fit.
//!
//! TWO TRUTHS, and why the second one exists (#1561)
//! -------------------------------------------------
//! The original fixture used
//!   mu*(x)    = sin(x),
//!   sigma*(x) = 0.15 + 0.1*cos(x),
//! and gam lost the mu channel to gamlss by 2.4x.
//!
//! `gamlss::pbc()` builds its second-order circular difference operator with the
//! stencil `c(-1, 2*cos(2*pi/n), -1)` (see `gamlss:::pbc`, the `sin = TRUE`
//! branch of its `Cdiff`) — the HARMONIC recurrence, not the ordinary
//! `c(-1, 2, -1)`. Its penalty therefore annihilates the FUNDAMENTAL sinusoid
//! rather than the constant: as lambda grows, a pbc fit converges onto a 2-df
//! pure first-harmonic model instead of onto a constant. Measured on this
//! fixture, pbc selects `mu.df = 2.004` on every one of 25 seeds and leaves
//! 8e-7 of energy above mode 1. `mu*(x) = sin(x)` IS that null space (its
//! amplitude above k=1 is 5e-17).
//!
//! gam's cyclic roughness used to be the plain derivative seminorm `∮(f'')²`,
//! which charges the fundamental, so the fundamental arm compared a smoother
//! against a model whose null space is the truth. Paired over 25 seeds before
//! the change, gamlss's mu advantage was 1.832x on that truth and 1.063x once
//! the truth carries content above the fundamental (commit `e9fa87f5f`); against
//! `mgcv gaulss`, whose `bs="cc"` penalty also charges the fundamental, gam was
//! 3.2% / 9.0% behind. Since `7ebbacd3d` gam's cyclic roughness is the harmonic
//! seminorm `∮(f'' + ω²(f − f̄))²`, whose null space is `{1, sin, cos}`, with its
//! own ridge shrinking that null component. Both engines now leave the
//! fundamental unpenalized, so the fundamental arm is a like-for-like
//! comparison.
//!
//! The derivative-roughness mechanism also predicted the SIGN of the other
//! channel: `log sigma*(x) = log(0.15 + 0.1 cos x)` is a nonlinear function of a
//! fundamental, so it carries cos2 = -0.146, cos3 = +0.037, ... — 11% of its
//! energy above k=1, outside pbc's null space — and gam won that channel.
//!
//! So this test runs BOTH truths: a pure fundamental, and a truth carrying
//! content above the fundamental, which no cyclic penalty's null space contains.
//!
//! PAIRED over seeds, not one draw (#2395)
//! ---------------------------------------
//! The joint gam/gamlss recovery ratio on a SINGLE draw of this fixture ranges
//! over 0.90..1.23 across seeds, so a one-draw 1.10x ceiling is a coin flip
//! whichever truth it uses. Both arms therefore run `K_SEEDS` seeds and compare
//! the two engines on the SAME noise draw seed by seed, so the common draw
//! cancels; the decision is `assert_paired_match_or_beat`, which additionally
//! requires that gam not be RESOLVED worse across seeds. At K=10 the joint
//! channel is an unresolved tie on both truths; K=25 resolves it in gam's
//! favour on both, which is why K is 25.
//!
//! gamlss as a match-or-beat ACCURACY baseline (not a target)
//! ----------------------------------------------------------
//! `gamlss::gamlss(family = NO())` is the mature distributional-regression
//! engine; fed the IDENTICAL (x, y) it produces its own cyclic mu- and
//! log-sigma fits, and we measure ITS error against the same truth. The
//! match-or-beat runs on the COMBINED location-scale recovery rather than each
//! block separately: the two engines use different cyclic penalties and trade
//! error BETWEEN the mean and log-scale blocks, and the joint object is what a
//! location-scale model estimates. Both blocks' absolute truth-recovery bars
//! remain the primary objective claim; per-block QUALITY_PAIR telemetry is
//! emitted for both channels so neither trade is hidden from the #1561 gate.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::solver::estimate::BlockRole;
use gam::test_support::reference::{
    Column, PairedFoldComparison, QualityPair, assert_paired_match_or_beat, held_out_r2, pad_to,
    relative_l2, rmse, run_r,
};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use std::path::Path;

/// Paired seeds per arm. See the "PAIRED over seeds" note above: 10 leaves the
/// joint channel unresolved, 25 resolves it.
const K_SEEDS: usize = 25;
/// Training points per seed.
const N_TRAIN: usize = 150;
/// Evaluation grid points in [0, 2*pi).
const N_GRID: usize = 50;

/// Which generating truth an arm uses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Truth {
    /// The original fixture: a pure fundamental in both blocks, which is exactly
    /// `gamlss::pbc`'s penalty null space.
    Fundamental,
    /// The same kind of object — periodic, smooth, heteroscedastic — carrying
    /// content ABOVE the fundamental, so no cyclic penalty's null space contains
    /// it and the comparison measures smoothing rather than a coincidence.
    AboveFundamental,
}

impl Truth {
    fn label(self) -> &'static str {
        match self {
            Truth::Fundamental => "fundamental",
            Truth::AboveFundamental => "above_fundamental",
        }
    }

    /// True mean function on the circle.
    fn mu(self, x: f64) -> f64 {
        match self {
            Truth::Fundamental => x.sin(),
            Truth::AboveFundamental => x.sin() + 0.4 * (2.0 * x + 0.7).sin(),
        }
    }

    /// True standard deviation on the circle (strictly positive on [0, 2*pi]).
    fn sigma(self, x: f64) -> f64 {
        match self {
            Truth::Fundamental => 0.15 + 0.1 * x.cos(),
            // min over the circle is 0.15 - 0.07 - 0.04 = 0.04 > 0.
            Truth::AboveFundamental => 0.15 + 0.07 * x.cos() + 0.04 * (2.0 * x - 0.4).cos(),
        }
    }
}

/// Deterministic standard-normal draws via Box–Muller from a tiny LCG, so the
/// data handed to gam and to gamlss is identical and reproducible without
/// pulling an RNG-crate dependency that could drift between versions.
fn standard_normals(n: usize, seed: u64) -> Vec<f64> {
    // 64-bit LCG (Numerical Recipes constants).
    let mut state = seed;
    let mut next_unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // top 53 bits -> (0,1)
        let bits = state >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_unit();
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out.truncate(n);
    out
}

/// The `x` grid every seed shares.
fn training_x() -> Vec<f64> {
    let period = 2.0 * PI;
    (0..N_TRAIN)
        .map(|i| period * (i as f64) / ((N_TRAIN - 1) as f64))
        .collect()
}

/// One seed's response vector under `truth`.
fn training_y(truth: Truth, xs: &[f64], seed: u64) -> Vec<f64> {
    let z = standard_normals(N_TRAIN, seed);
    xs.iter()
        .zip(z.iter())
        .map(|(&x, &zi)| truth.mu(x) + truth.sigma(x) * zi)
        .collect()
}

/// gam's cyclic location-scale fit on one seed; returns (mu-RMSE, log-sigma-RMSE)
/// against the truth on the shared evaluation grid.
fn gam_arm_scores(truth: Truth, xs: &[f64], ys: &[f64], grid_x: &[f64]) -> (f64, f64) {
    let headers = vec!["y".to_string(), "x".to_string()];
    let rows: Vec<StringRecord> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| StringRecord::from(vec![format!("{y:.17e}"), format!("{x:.17e}")]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode cyclic dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    // Pin the period explicitly to [0, 2*pi] so gam's cyclic boundary matches the
    // data range gamlss's pbc infers below.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(
            "1 + s(x, bs='cc', period_start=0, period_end=6.283185307179586)".to_string(),
        ),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x, bs='cc', period_start=0, period_end=6.283185307179586)",
        &ds,
        &cfg,
    )
    .expect("gam cyclic location-scale fit");
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected a GaussianLocationScale fit for a Gaussian noise_formula model");
    };

    let beta_mu = fit
        .fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location-scale fit carries a Location (mu) block")
        .beta
        .clone();
    let beta_noise = fit
        .fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("location-scale fit carries a Scale (log-sigma) block")
        .beta
        .clone();
    // A smooth sigma must materialize a multi-column basis (intercept + cc),
    // otherwise the cyclic structure never reached the scale block.
    assert!(
        beta_noise.len() >= 2,
        "cyclic noise_formula must materialize a multi-coefficient scale basis, got {}",
        beta_noise.len()
    );

    // Rebuild the mean and noise designs from the FROZEN resolved specs at the
    // evaluation grid, so the comparison is on the smooth SHAPE off the training
    // points rather than in-sample fitted values.
    let mut eval_grid = Array2::<f64>::zeros((N_GRID, ds.headers.len()));
    for (i, &gx) in grid_x.iter().enumerate() {
        eval_grid[[i, x_idx]] = gx;
    }
    let mean_design = build_term_collection_design(eval_grid.view(), &fit.fit.meanspec_resolved)
        .expect("rebuild mean design at eval grid");
    let noise_design = build_term_collection_design(eval_grid.view(), &fit.fit.noisespec_resolved)
        .expect("rebuild noise design at eval grid");
    assert_eq!(
        mean_design.design.ncols(),
        beta_mu.len(),
        "mean design columns ({}) must match mu coefficient count ({})",
        mean_design.design.ncols(),
        beta_mu.len()
    );
    assert_eq!(
        noise_design.design.ncols(),
        beta_noise.len(),
        "noise design columns ({}) must match log-sigma coefficient count ({})",
        noise_design.design.ncols(),
        beta_noise.len()
    );

    let gam_mu: Vec<f64> = mean_design.design.apply(&beta_mu).to_vec();
    let eta_noise: Array1<f64> = noise_design.design.apply(&beta_noise);
    // Raw-unit σ = response_scale·sigma_floor + exp(η): the fit's own floor
    // (recording-grid bound of the standardized response) mapped to raw units.
    let raw_sigma_floor = fit.response_scale * fit.sigma_floor;
    let gam_log_sigma: Vec<f64> = eta_noise
        .iter()
        .map(|&e| {
            gam::families::sigma_link::logb_sigma_from_eta_scalar(raw_sigma_floor, e).ln()
        })
        .collect();

    let truth_mu: Vec<f64> = grid_x.iter().map(|&gx| truth.mu(gx)).collect();
    let truth_log_sigma: Vec<f64> = grid_x.iter().map(|&gx| truth.sigma(gx).ln()).collect();
    (
        rmse(&gam_mu, &truth_mu),
        rmse(&gam_log_sigma, &truth_log_sigma),
    )
}

/// Run one truth's whole paired panel: K seeds through gam and through gamlss on
/// the SAME per-seed data, then the paired decision.
fn run_cyclic_location_scale_arm(truth: Truth) {
    init_parallelism();

    let period = 2.0 * PI;
    let xs = training_x();
    let grid_x: Vec<f64> = (0..N_GRID)
        .map(|i| period * (i as f64) / (N_GRID as f64))
        .collect();
    let truth_mu: Vec<f64> = grid_x.iter().map(|&gx| truth.mu(gx)).collect();
    let truth_log_sigma: Vec<f64> = grid_x.iter().map(|&gx| truth.sigma(gx).ln()).collect();

    // ---- gam on every seed, and the long-format data the reference replays ---
    let mut gam_mu_rmses = Vec::with_capacity(K_SEEDS);
    let mut gam_ls_rmses = Vec::with_capacity(K_SEEDS);
    let mut long_seed = Vec::with_capacity(K_SEEDS * N_TRAIN);
    let mut long_x = Vec::with_capacity(K_SEEDS * N_TRAIN);
    let mut long_y = Vec::with_capacity(K_SEEDS * N_TRAIN);
    for k in 0..K_SEEDS {
        let seed = 1 + k as u64;
        let ys = training_y(truth, &xs, seed);
        let (mu_rmse, ls_rmse) = gam_arm_scores(truth, &xs, &ys, &grid_x);
        gam_mu_rmses.push(mu_rmse);
        gam_ls_rmses.push(ls_rmse);
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            long_seed.push(seed as f64);
            long_x.push(x);
            long_y.push(y);
        }
    }

    // ---- the SAME K datasets through gamlss, in ONE R session ---------------
    // family = NO() (normal, identity mu, log sigma); mu and sigma each get
    // gamlss's native penalized CYCLIC P-spline `pbc()`, the cyclic analogue of
    // `pb()`. x spans exactly [0, 2*pi], so pbc's data-range period matches gam's
    // explicit [0, 2*pi] cyclic boundary.
    let r = run_r(
        &[
            Column::new("seed", &long_seed),
            Column::new("x", &long_x),
            Column::new("y", &long_y),
        ],
        r#"
        suppressPackageStartupMessages(library(gamlss))
        xg <- seq(0, 2*pi, length.out = 51)[1:50]
        nd <- data.frame(x = xg)
        mu_all <- c(); ls_all <- c()
        for (s in sort(unique(df$seed))) {
            d <- df[df$seed == s, c("x", "y")]
            m <- gamlss(
                y ~ pbc(x),
                sigma.formula = ~ pbc(x),
                family = NO(),
                data = d,
                control = gamlss.control(n.cyc = 200, trace = FALSE)
            )
            mu_all <- c(mu_all,
                as.numeric(predict(m, what = "mu", newdata = nd, type = "response", data = d)))
            ls_all <- c(ls_all,
                as.numeric(predict(m, what = "sigma", newdata = nd, type = "link", data = d)))
        }
        emit("mu", mu_all)
        emit("log_sigma", ls_all)
        "#,
    );
    let gamlss_mu_flat = r.vector("mu");
    // gamlss NO() uses a log link for sigma, so the "link"-scale sigma predictor
    // is exactly log(sigma): directly comparable to gam's log-sigma curve.
    let gamlss_ls_flat = r.vector("log_sigma");
    assert_eq!(
        gamlss_mu_flat.len(),
        K_SEEDS * N_GRID,
        "gamlss mu panel length mismatch"
    );
    assert_eq!(
        gamlss_ls_flat.len(),
        K_SEEDS * N_GRID,
        "gamlss log-sigma panel length mismatch"
    );

    let mut gamlss_mu_rmses = Vec::with_capacity(K_SEEDS);
    let mut gamlss_ls_rmses = Vec::with_capacity(K_SEEDS);
    for k in 0..K_SEEDS {
        let lo = k * N_GRID;
        let hi = lo + N_GRID;
        gamlss_mu_rmses.push(rmse(&gamlss_mu_flat[lo..hi], &truth_mu));
        gamlss_ls_rmses.push(rmse(&gamlss_ls_flat[lo..hi], &truth_log_sigma));
    }

    // ---- paired panels: same seed, same bytes, seed by seed -----------------
    let mu_panel = PairedFoldComparison::new(&gam_mu_rmses, &gamlss_mu_rmses, true);
    let ls_panel = PairedFoldComparison::new(&gam_ls_rmses, &gamlss_ls_rmses, true);
    // The joint location-scale object: what a location-scale model estimates, and
    // the channel the match-or-beat decision runs on.
    let gam_joint: Vec<f64> = gam_mu_rmses
        .iter()
        .zip(gam_ls_rmses.iter())
        .map(|(m, l)| (m * m + l * l).sqrt())
        .collect();
    let gamlss_joint: Vec<f64> = gamlss_mu_rmses
        .iter()
        .zip(gamlss_ls_rmses.iter())
        .map(|(m, l)| (m * m + l * l).sqrt())
        .collect();
    let joint_panel = PairedFoldComparison::new(&gam_joint, &gamlss_joint, true);

    let label = truth.label();
    eprintln!(
        "cyclic location-scale [{label}] K={K_SEEDS}-seed paired: n={N_TRAIN} grid={N_GRID} \
         gam_mu={:.5} gamlss_mu={:.5} gam_logsigma={:.5} gamlss_logsigma={:.5} \
         gam_joint={:.5} gamlss_joint={:.5}",
        mu_panel.gam_mean,
        mu_panel.reference_mean,
        ls_panel.gam_mean,
        ls_panel.reference_mean,
        joint_panel.gam_mean,
        joint_panel.reference_mean,
    );
    eprintln!("{}", mu_panel.report(&format!("cyclic_ls::{label}::mu")));
    eprintln!(
        "{}",
        ls_panel.report(&format!("cyclic_ls::{label}::log_sigma"))
    );
    eprintln!(
        "{}",
        joint_panel.report(&format!("cyclic_ls::{label}::joint"))
    );
    eprintln!(
        "{}",
        QualityPair::paired(
            "families",
            &format!("quality_vs_gamlss_gaussian_location_scale_cyclic::{label}::mu"),
            "mu_rmse_to_truth",
            "gamlss",
            &mu_panel,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::paired(
            "families",
            &format!("quality_vs_gamlss_gaussian_location_scale_cyclic::{label}::log_sigma"),
            "log_sigma_rmse_to_truth",
            "gamlss",
            &ls_panel,
        )
        .line()
    );

    // PRIMARY: gam recovers the true cyclic mean. The mean's signal SD is ~1/sqrt(2)
    // and per-point noise sigma runs 0.04..0.26, so 0.06 RMSE means the recovered
    // mean tracks the truth to a small fraction of the signal range. Measured
    // fold-mean at `7ebbacd3d`: 0.0210 (fundamental) / 0.0363 (above-fundamental),
    // against gamlss's 0.0210 / 0.0355; before the harmonic cyclic roughness it
    // was 0.0342 / 0.0376.
    assert!(
        mu_panel.gam_mean <= 0.06,
        "cyclic mu [{label}] does not recover the truth: fold-mean RMSE={:.4} (bound 0.06)",
        mu_panel.gam_mean
    );
    // PRIMARY: gam recovers the true cyclic log-scale. The log-scale block is
    // identified one likelihood-derivative removed from the data, so its absolute
    // bar is looser, yet 0.30 still requires the recovered curve to track the true
    // log sigma. Measured fold-mean at `7ebbacd3d`: 0.136 / 0.148, against gamlss's
    // 0.146 / 0.158.
    assert!(
        ls_panel.gam_mean <= 0.30,
        "cyclic log-sigma [{label}] does not recover the truth: fold-mean RMSE={:.4} (bound 0.30)",
        ls_panel.gam_mean
    );

    // MATCH-OR-BEAT on the JOINT location-scale object, paired across the K shared
    // noise draws. See the header: the two engines trade error between the mean and
    // the log-scale block. Measured paired effect before the harmonic cyclic
    // roughness (`7ebbacd3d`): -0.118 (fundamental) / -0.066 (above-fundamental),
    // both resolved in gam's favour. Joint fold-mean at `7ebbacd3d`: gam 0.1384 vs
    // gamlss 0.1480 (fundamental), gam 0.1534 vs gamlss 0.1627 (above-fundamental).
    assert_paired_match_or_beat(&format!("cyclic_ls::{label}::joint"), &joint_panel, 1.10);
}

#[test]
fn gam_cyclic_location_scale_recovers_truth() {
    run_cyclic_location_scale_arm(Truth::Fundamental);
}

#[test]
fn gam_cyclic_location_scale_recovers_truth_above_the_fundamental() {
    run_cyclic_location_scale_arm(Truth::AboveFundamental);
}

// ===========================================================================
// REAL-DATA ARM
// ===========================================================================
//
// Same gam capability (Gaussian location-scale with a CYCLIC smooth in BOTH
// the mean mu(month) and the log-scale log sigma(month)) exercised on a real,
// strongly periodic, heteroscedastic series. On real data the truth is
// unknown, so the assertions are OUT-OF-SAMPLE predictive quality, not
// truth recovery.
//
// Dataset SOURCE: `nottem` — average monthly air temperatures (deg F) at
// Nottingham Castle, Jan 1920 .. Dec 1939, 240 observations. Shipped with base
// R's `datasets` package (`datasets::nottem`); the classic seasonal-decomposition
// teaching series (Anderson 1976, "Time Series Analysis and Forecasting").
// Vendored here as bench/datasets/nottem_monthly_temp.csv with columns
// year, month (1..12), temp.
//
// The seasonal mean cycle is very strong (summer ~60F, winter ~38F) and the
// month-to-month spread is itself seasonal (mild months vary more than the
// settled deep-winter / mid-summer months), so month -> temp is a textbook
// periodic, heteroscedastic location-scale problem. Because month is circular
// (December 12 abuts January 1) the natural smooth is cyclic with the periodic
// boundary halfway outside the 1..12 integer grid, i.e. the period spans
// [0.5, 12.5].

/// Per-point Gaussian negative log-likelihood of held-out observations under a
/// predicted (mean, sigma) for each row: the natural objective score for a
/// heteroscedastic location-scale predictor (it rewards calibrated sigma, not
/// just an accurate mean). Lower is better.
fn gaussian_nll(mean: &[f64], sigma: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(mean.len(), truth.len(), "nll mean/truth length mismatch");
    assert_eq!(sigma.len(), truth.len(), "nll sigma/truth length mismatch");
    let half_log_2pi = 0.5 * (2.0 * PI).ln();
    let n = truth.len() as f64;
    let total: f64 = mean
        .iter()
        .zip(sigma.iter())
        .zip(truth.iter())
        .map(|((&m, &s), &y)| {
            assert!(s > 0.0, "predicted sigma must be positive, got {s}");
            let z = (y - m) / s;
            half_log_2pi + s.ln() + 0.5 * z * z
        })
        .sum();
    total / n
}

#[test]
fn gam_cyclic_location_scale_recovers_truth_on_real_data() {
    init_parallelism();

    // ---- load the real Nottingham monthly-temperature series --------------
    const NOTTEM_CSV: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/nottem_monthly_temp.csv"
    );
    let ds =
        load_csvwith_inferred_schema(Path::new(NOTTEM_CSV)).expect("load nottem_monthly_temp.csv");
    let col = ds.column_map();
    let month_idx = col["month"];
    let temp_idx = col["temp"];
    let month: Vec<f64> = ds.values.column(month_idx).to_vec();
    let temp: Vec<f64> = ds.values.column(temp_idx).to_vec();
    let n = month.len();
    assert!(n > 200, "nottem should have 240 rows, got {n}");

    // ---- deterministic train/test split: every 5th row held out ----------
    // Months 1..12 cycle every 12 rows, so stride 5 keeps all 12 months in
    // both folds (5 and 12 are coprime).
    let is_test = |i: usize| i % 5 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 150 && test_rows.len() > 40,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_month: Vec<f64> = train_rows.iter().map(|&i| month[i]).collect();
    let train_temp: Vec<f64> = train_rows.iter().map(|&i| temp[i]).collect();
    let test_month: Vec<f64> = test_rows.iter().map(|&i| month[i]).collect();
    let test_temp: Vec<f64> = test_rows.iter().map(|&i| temp[i]).collect();

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

    // ---- fit gam on TRAIN: Gaussian location-scale, cyclic in BOTH blocks --
    // Period pinned to [0.5, 12.5] so the cyclic boundary lands halfway between
    // December (12) and January (1) — the natural seam of a monthly calendar —
    // and matches the `knots = c(0.5, 12.5)` we hand mgcv inside gamlss below.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(month, bs='cc', period_start=0.5, period_end=12.5)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "temp ~ s(month, bs='cc', period_start=0.5, period_end=12.5)",
        &train_ds,
        &cfg,
    )
    .expect("gam cyclic location-scale fit on nottem");
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected a GaussianLocationScale fit for a Gaussian noise_formula model");
    };

    let beta_mu = fit
        .fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location-scale fit carries a Location (mu) block")
        .beta
        .clone();
    let beta_noise = fit
        .fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("location-scale fit carries a Scale (log-sigma) block")
        .beta
        .clone();
    assert!(
        beta_noise.len() >= 2,
        "cyclic noise_formula must materialize a multi-coefficient scale basis, got {}",
        beta_noise.len()
    );

    // gam standardizes the response internally (it fits y / sample_std(y_train)
    // so the log-σ floor is scale-relative) and then maps the fitted blocks
    // BACK to raw response units before returning them: the Location block is
    // scaled by response_scale and the log-σ block intercept is shifted by
    // +ln(response_scale). So the returned `beta_mu` / `beta_noise` are already
    // in response (deg F) units; only the σ floor, which sits outside the
    // exponential, is carried by response_scale:
    //   mu_response    = X_mu @ beta_mu
    //   sigma_response = response_scale·sigma_floor + exp(X_noise @ beta_noise)

    // ---- gam predictions at the HELD-OUT months ---------------------------
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &mo) in test_month.iter().enumerate() {
        test_grid[[i, month_idx]] = mo;
    }
    let mean_design = build_term_collection_design(test_grid.view(), &fit.fit.meanspec_resolved)
        .expect("rebuild mean design at held-out months");
    let noise_design = build_term_collection_design(test_grid.view(), &fit.fit.noisespec_resolved)
        .expect("rebuild noise design at held-out months");
    assert_eq!(
        mean_design.design.ncols(),
        beta_mu.len(),
        "mean design columns must match mu coefficient count"
    );
    assert_eq!(
        noise_design.design.ncols(),
        beta_noise.len(),
        "noise design columns must match log-sigma coefficient count"
    );

    let gam_test_mean: Vec<f64> = mean_design.design.apply(&beta_mu).to_vec();
    let eta_noise: Array1<f64> = noise_design.design.apply(&beta_noise);
    let raw_sigma_floor = fit.response_scale * fit.sigma_floor;
    let gam_test_sigma: Vec<f64> = eta_noise
        .iter()
        .map(|&e| gam::families::sigma_link::logb_sigma_from_eta_scalar(raw_sigma_floor, e))
        .collect();

    // ---- fit the SAME model on TRAIN with gamlss, predict the SAME TEST ----
    // family = NO() (normal, identity mu, log sigma); mu and sigma each get a
    // cyclic cubic smooth via mgcv's ga(~ s(month, bs="cc")), cyclic knots pinned
    // to [0.5, 12.5] to match gam's explicit period. We pass the training rows
    // plus the held-out months padded into a parallel column (the harness exposes
    // one equal-length data.frame per call) and predict on the first `test_n`.
    let r = run_r(
        &[
            Column::new("month", &train_month),
            Column::new("temp", &train_temp),
            Column::new("test_month", &pad_to(&test_month, train_month.len())),
            Column::new("test_n", &vec![test_month.len() as f64; train_month.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(gamlss))
        # gamlss's native penalized CYCLIC P-spline `pbc()` (auto smoothing-
        # parameter selection) replaces the gamlss.add/mgcv `ga(~ s(., bs="cc"))`
        # bridge, which is unavailable here. month is circular over 1..12, so the
        # cyclic boundary lands at the December/January seam, matching gam's
        # explicit [0.5, 12.5] period.
        m <- gamlss(
            temp ~ pbc(month),
            sigma.formula = ~ pbc(month),
            family = NO(),
            data = df,
            control = gamlss.control(n.cyc = 200, trace = FALSE)
        )
        k <- df$test_n[1]
        nd <- data.frame(month = df$test_month[1:k])
        mu <- as.numeric(predict(m, what = "mu", newdata = nd, type = "response", data = df))
        ls <- as.numeric(predict(m, what = "sigma", newdata = nd, type = "link", data = df))
        emit("mu", mu)
        emit("log_sigma", ls)
        "#,
    );
    let gamlss_mean = r.vector("mu").to_vec();
    // NO() uses a log link on sigma, so the link-scale sigma predictor is exactly
    // log(sigma): exponentiate to get gamlss's held-out response-unit sigma.
    let gamlss_sigma: Vec<f64> = r.vector("log_sigma").iter().map(|&v| v.exp()).collect();
    assert_eq!(
        gamlss_mean.len(),
        test_rows.len(),
        "gamlss mu length mismatch"
    );
    assert_eq!(
        gamlss_sigma.len(),
        test_rows.len(),
        "gamlss sigma length mismatch"
    );

    // ---- objective held-out metrics ---------------------------------------
    let gam_nll = gaussian_nll(&gam_test_mean, &gam_test_sigma, &test_temp);
    let gamlss_nll = gaussian_nll(&gamlss_mean, &gamlss_sigma, &test_temp);
    let gam_r2 = held_out_r2(&gam_test_mean, &test_temp);

    // Context-only diagnostic: gam-vs-gamlss agreement of the held-out mean and
    // sigma curves. NOT a pass criterion.
    let mean_rel = relative_l2(&gam_test_mean, &gamlss_mean);
    let sigma_rel = relative_l2(&gam_test_sigma, &gamlss_sigma);
    let response_scale = fit.response_scale;

    eprintln!(
        "nottem cyclic location-scale held-out: n_train={} n_test={} \
         response_scale={response_scale:.4} gam_nll={gam_nll:.4} gamlss_nll={gamlss_nll:.4} \
         gam_test_R2={gam_r2:.4} (context: mean_rel_l2={mean_rel:.4} sigma_rel_l2={sigma_rel:.4}) \
         beta_mu={} beta_sigma={}",
        train_rows.len(),
        test_rows.len(),
        beta_mu.len(),
        beta_noise.len(),
    );

    // ---- PRIMARY (objective, tool-free): calibrated held-out density -------
    // The held-out per-point Gaussian NLL scores BOTH the mean and the sigma
    // calibration. A competent location-scale fit of this series resolves the
    // seasonal mean to a few deg F and predicts a residual spread of ~2..4 F,
    // giving NLL ~ 0.5*log(2*pi) + log(sigma) + 0.5 ~ 2.3. We require NLL <= 3.2:
    // comfortably below the homoscedastic constant-mean baseline (sd(temp) ~ 8 F
    // => NLL ~ 0.5*log(2*pi) + log(8) + 0.5 ~ 3.5) while leaving ample headroom.
    assert!(
        gam_nll <= 3.2,
        "gam held-out Gaussian NLL too high: {gam_nll:.4} (> 3.2)"
    );

    // ---- PRIMARY (objective): the cyclic mean explains held-out variance ---
    // The seasonal cycle is overwhelmingly strong, so a faithful cyclic mean
    // smooth explains the vast majority of held-out variance.
    assert!(
        gam_r2 >= 0.90,
        "gam held-out mean R2 too low: {gam_r2:.4} (< 0.90)"
    );

    // ---- BASELINE (match-or-beat): no worse than gamlss on held-out NLL ----
    // gamlss is the mature distributional-regression baseline fed the IDENTICAL
    // train/test rows and the SAME cyclic basis; NLL is a log-scale score, so an
    // additive 0.10-nat slack is the principled match-or-beat margin (gamlss is a
    // floor to match-or-beat on predictive density, never a fit to reproduce).
    assert!(
        gam_nll <= gamlss_nll + 0.10,
        "gam held-out NLL {gam_nll:.4} worse than gamlss {gamlss_nll:.4} + 0.10"
    );
}
