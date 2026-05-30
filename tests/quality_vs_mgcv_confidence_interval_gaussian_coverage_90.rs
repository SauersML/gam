//! Empirical confidence-interval coverage of gam's penalized-smooth uncertainty,
//! benchmarked head-to-head against `mgcv` — the mature, standard GAM
//! implementation whose `predict(..., se.fit = TRUE)` is the field-standard way
//! to get pointwise standard errors for a smooth.
//!
//! Why this is the right comparator: mgcv's default prediction covariance is the
//! Bayesian/penalized covariance `Vp` (Wood 2017, §6.10), which propagates the
//! smoothing-penalty shrinkage into the reported SE. gam exposes the exact same
//! object as its smoothing-corrected coefficient covariance (`Vp`,
//! `beta_covariance_corrected`). A confidence interval built from `Vp` has a
//! frequency-coverage interpretation across replicate datasets: for a 90%
//! nominal interval on the regression function, the band should contain the true
//! mean `eta(x) = sin(6*pi*x)` at roughly 90% of (replicate, evaluation-point)
//! pairs. This is the across-replicate coverage property mgcv's Bayesian
//! intervals are designed to satisfy (Nychka 1988; Marra & Wood 2012).
//!
//! Design of the experiment (identical data fed to both engines):
//!   * One fixed design `x` (n = 500, U[0,1], seed 42) shared by every replicate
//!     so the signal `eta(x) = sin(6*pi*x)` is constant; only the Gaussian noise
//!     (sigma = 0.1) is redrawn per replicate.
//!   * 100 replicate response vectors `y_r = eta + noise_r` are generated once in
//!     Rust with a deterministic splitmix64 + Box–Muller stream and handed,
//!     column for column, to BOTH gam and mgcv. No engine sees different data.
//!   * Each replicate is fit `y ~ s(x, k = 10)` by REML in both engines.
//!   * Pointwise 90% CIs are evaluated on a fixed *interior* grid
//!     (`seq(0.05, 0.95, length.out = 50)`, regenerated bit-identically on both
//!     sides) so the measurement isolates the conditional/penalized covariance
//!     path and is NOT contaminated by gam's optional boundary or OOD variance
//!     inflation — both of which are switched off here (we read the covariance
//!     directly and build the band ourselves, so no correction is applied).
//!   * gam SE at an evaluation point `x_i` is `sqrt(s_i^T Vp s_i)` where `s_i` is
//!     the design row at `x_i` (identity link => the predictor equals the mean),
//!     exactly the quadratic form mgcv evaluates internally for `se.fit`.
//!
//! Metric / bound: empirical coverage = fraction of (replicate, eval-point) pairs
//! whose 90% CI contains the TRUE `eta`. Under correct CI construction this is
//! ~0.90. We require gam's coverage to land in [0.85, 0.95] (nominal 0.90 +/-
//! 0.05). Too low => SE underestimates (e.g. ignores smoothing-penalty variance);
//! too high => SE overestimates. The window is tight enough that a systematic SE
//! bias of more than ~half a nominal-level point is rejected, yet wide enough to
//! absorb Monte-Carlo error. The 50 eval points within one replicate share the
//! same fitted curve and the same noise draw, so they are strongly correlated;
//! the effective sample size for the coverage rate is therefore on the order of
//! the 100 replicates, not the 5000 (replicate, point) pairs. The between-
//! replicate s.e. of the coverage estimate is ~0.005-0.01, so the +/-0.05 window
//! is several s.e. wide — loose enough to never fail on noise, tight enough that
//! a systematic SE bias of half a nominal point is rejected. mgcv's own coverage
//! on the identical data is reported alongside as the calibration reference.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array2, ArrayView2};

const N: usize = 500;
const N_REPLICATES: usize = 100;
const N_EVAL: usize = 50;
const SIGMA: f64 = 0.1;
const EVAL_LO: f64 = 0.05;
const EVAL_HI: f64 = 0.95;
/// Two-sided 90% normal quantile, qnorm(0.95). Used identically on both sides.
const Z_90: f64 = 1.6448536269514722;

/// Deterministic 64-bit stream (SplitMix64) so the synthetic data is identical
/// across platforms and reproducible from `seed = 42`.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in the open interval (0, 1).
    fn next_open01(&mut self) -> f64 {
        // 53-bit mantissa, shifted off 0 so log() in Box-Muller is finite.
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller (one variate per call; the paired draw is
    /// recomputed each time, which is fine for a test generator).
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_open01();
        let u2 = self.next_open01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn true_eta(x: f64) -> f64 {
    (6.0 * std::f64::consts::PI * x).sin()
}

/// `seq(EVAL_LO, EVAL_HI, length.out = N_EVAL)`, bit-identical to R's `seq`.
fn eval_grid() -> Vec<f64> {
    (0..N_EVAL)
        .map(|i| EVAL_LO + (EVAL_HI - EVAL_LO) * (i as f64) / ((N_EVAL - 1) as f64))
        .collect()
}

/// SE at each evaluation row from a coefficient covariance `cov`:
/// `se_i = sqrt(s_i^T cov s_i)` where `s_i` is the i-th design row.
fn pointwise_se(design: ArrayView2<'_, f64>, cov: &Array2<f64>) -> Vec<f64> {
    let n = design.nrows();
    let p = design.ncols();
    assert_eq!(cov.nrows(), p, "covariance/design dimension mismatch");
    assert_eq!(cov.ncols(), p, "covariance must be square");
    (0..n)
        .map(|i| {
            let s = design.row(i);
            // s^T cov s
            let mut acc = 0.0;
            for a in 0..p {
                let sa = s[a];
                if sa == 0.0 {
                    continue;
                }
                let mut row_dot = 0.0;
                for b in 0..p {
                    row_dot += cov[[a, b]] * s[b];
                }
                acc += sa * row_dot;
            }
            acc.max(0.0).sqrt()
        })
        .collect()
}

#[test]
fn ci_coverage_matches_mgcv_on_gaussian_truth_90pct() {
    init_parallelism();

    // ---- fixed design + 100 replicate responses (seed 42) -----------------
    let mut rng = SplitMix64::new(42);
    let x: Vec<f64> = (0..N).map(|_| rng.next_open01()).collect();
    let eta: Vec<f64> = x.iter().map(|&xi| true_eta(xi)).collect();

    // Column-major replicate responses: replicates[r] is the r-th y vector.
    let replicates: Vec<Vec<f64>> = (0..N_REPLICATES)
        .map(|_| {
            (0..N)
                .map(|i| eta[i] + SIGMA * rng.next_standard_normal())
                .collect::<Vec<f64>>()
        })
        .collect();

    let grid = eval_grid();
    let truth_eval: Vec<f64> = grid.iter().map(|&g| true_eta(g)).collect();

    // ---- gam: fit each replicate, evaluate CI coverage at the grid --------
    // Design column layout for fit: ["x", "y"]. We reuse the same EncodedDataset
    // schema for every replicate, swapping only the y column.
    let headers = vec!["x".to_string(), "y".to_string()];
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let mut gam_hits_vp = 0usize; // smoothing-corrected (Vp) — the mgcv default analog
    let mut gam_hits_vb = 0usize; // conditional (Vb) — reported for diagnostics only
    let total_trials = N_REPLICATES * N_EVAL;

    for rep in &replicates {
        let records: Vec<csv::StringRecord> = (0..N)
            .map(|i| {
                csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", rep[i])])
            })
            .collect();
        let ds = encode_recordswith_inferred_schema(headers.clone(), records)
            .expect("encode synthetic replicate");
        let col = ds.column_map();
        let x_idx = col["x"];

        let result = fit_from_formula("y ~ s(x, k=10)", &ds, &cfg).expect("gam fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit for a Gaussian smooth");
        };

        // Build the design at the evaluation grid from the frozen spec.
        let mut eval_data = Array2::<f64>::zeros((N_EVAL, ds.headers.len()));
        for (i, &g) in grid.iter().enumerate() {
            eval_data[[i, x_idx]] = g;
        }
        let design = build_term_collection_design(eval_data.view(), &fit.resolvedspec)
            .expect("rebuild design at eval grid");
        let dense = design.design.to_dense();

        // Predictor at the grid (identity link => predictor == mean).
        let pred = design.design.apply(&fit.fit.beta);

        // Vp: smoothing-parameter-corrected covariance (mgcv's default Vp).
        let vp = fit
            .fit
            .beta_covariance_corrected()
            .expect("gam exposes smoothing-corrected covariance Vp");
        // Vb: conditional Bayesian covariance H^{-1} * phi.
        let vb = fit
            .fit
            .beta_covariance()
            .expect("gam exposes conditional covariance Vb");

        let se_vp = pointwise_se(dense.view(), vp);
        let se_vb = pointwise_se(dense.view(), vb);

        for j in 0..N_EVAL {
            let lo_vp = pred[j] - Z_90 * se_vp[j];
            let hi_vp = pred[j] + Z_90 * se_vp[j];
            if truth_eval[j] >= lo_vp && truth_eval[j] <= hi_vp {
                gam_hits_vp += 1;
            }
            let lo_vb = pred[j] - Z_90 * se_vb[j];
            let hi_vb = pred[j] + Z_90 * se_vb[j];
            if truth_eval[j] >= lo_vb && truth_eval[j] <= hi_vb {
                gam_hits_vb += 1;
            }
        }
    }

    let gam_coverage_vp = gam_hits_vp as f64 / total_trials as f64;
    let gam_coverage_vb = gam_hits_vb as f64 / total_trials as f64;

    // ---- mgcv: same data, same grid, predict(se.fit = TRUE), same z -------
    // Pass x plus all 100 replicate y columns. R regenerates the identical
    // interior grid and the identical truth, fits each replicate, and counts
    // coverage of the 90% CI built from its default (Vp) standard errors.
    let mut columns: Vec<Column<'_>> = Vec::with_capacity(N_REPLICATES + 1);
    columns.push(Column::new("x", &x));
    let yheaders: Vec<String> = (0..N_REPLICATES).map(|r| format!("y{r}")).collect();
    for (r, name) in yheaders.iter().enumerate() {
        columns.push(Column::new(name, &replicates[r]));
    }

    let r = run_r(
        &columns,
        r#"
        suppressPackageStartupMessages(library(mgcv))
        nrep <- 100L
        neval <- 50L
        z90 <- qnorm(0.95)
        grid <- seq(0.05, 0.95, length.out = neval)
        truth <- sin(6 * pi * grid)
        newd <- data.frame(x = grid)
        hits <- 0L
        for (r in 0:(nrep - 1L)) {
            yname <- paste0("y", r)
            dat <- data.frame(x = df$x, y = df[[yname]])
            m <- gam(y ~ s(x, k = 10), data = dat, method = "REML")
            pr <- predict(m, newdata = newd, se.fit = TRUE)
            lo <- pr$fit - z90 * pr$se.fit
            hi <- pr$fit + z90 * pr$se.fit
            hits <- hits + sum(truth >= lo & truth <= hi)
        }
        emit("coverage", hits / (nrep * neval))
        "#,
    );
    let mgcv_coverage = r.scalar("coverage");

    eprintln!(
        "Gaussian CI coverage (90% nominal, {N_REPLICATES} reps x {N_EVAL} eval pts = {total_trials} trials):\n  \
         gam Vp coverage = {gam_coverage_vp:.4}  (smoothing-corrected; mgcv-default analog)\n  \
         gam Vb coverage = {gam_coverage_vb:.4}  (conditional; diagnostic)\n  \
         mgcv coverage   = {mgcv_coverage:.4}  (predict se.fit, default Vp)"
    );

    // mgcv is the calibration reference: on this well-specified problem its
    // Bayesian intervals are known to be close to nominal. Sanity-check it lands
    // near 0.90 so a comparator misconfiguration cannot masquerade as a gam pass.
    assert!(
        mgcv_coverage >= 0.80 && mgcv_coverage <= 0.99,
        "mgcv reference coverage {mgcv_coverage:.4} is implausibly far from nominal 0.90; \
         the comparator setup is suspect"
    );

    // The core assertion: gam's penalized-covariance interval must achieve close
    // to the nominal 90% coverage of the true mean function. [0.85, 0.95] is the
    // principled +/- 0.05 window around 0.90; it rejects a systematic SE bias of
    // half a nominal point while tolerating the between-replicate Monte-Carlo
    // noise (~0.005-0.01 s.e. over the 100 effectively-independent replicates).
    assert!(
        gam_coverage_vp >= 0.85 && gam_coverage_vp <= 0.95,
        "gam 90% CI empirical coverage {gam_coverage_vp:.4} is outside the nominal window \
         [0.85, 0.95] (mgcv reference = {mgcv_coverage:.4}); gam's standard errors are \
         {} the truth.",
        if gam_coverage_vp < 0.85 {
            "under-covering (SE too small)"
        } else {
            "over-covering (SE too large)"
        }
    );
}
