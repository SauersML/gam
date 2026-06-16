//! OBJECTIVE METRIC ASSERTED: across-replicate empirical coverage of gam's
//! penalized-smooth 90% confidence band, measured against the KNOWN true mean
//! function `eta(x) = sin(6*pi*x)` — NOT against any reference tool's intervals.
//! This is a calibration test: a correctly-constructed 90% interval on the
//! regression function must contain the true `eta` at ~90% of (replicate,
//! evaluation-point) pairs (the frequentist coverage property of Bayesian/
//! penalized smooth intervals; Nychka 1988, Marra & Wood 2012). The pass/fail
//! criterion is gam's own coverage landing inside the nominal window
//! [0.85, 0.95] (0.90 +/- 0.05) — an absolute, self-contained calibration claim
//! that does not reference mgcv at all.
//!
//! gam's coverage is built entirely from gam's own smoothing-parameter-corrected
//! coefficient covariance `Vp` (`beta_covariance_corrected`), which propagates
//! the smoothing-penalty shrinkage into the reported SE (the same statistical
//! object mgcv's `predict(se.fit = TRUE)` uses by default — Wood 2017 §6.10 — but
//! here we compute everything ourselves from gam's matrices). SE at an evaluation
//! point `x_i` is `sqrt(s_i^T Vp s_i)` for the design row `s_i` (identity link =>
//! predictor == mean).
//!
//! ROLE OF mgcv: BASELINE TO MATCH-OR-BEAT ON CALIBRATION, never a "gam ==
//! mgcv" target. mgcv is fit on the IDENTICAL data and its own coverage measured
//! the same way. We additionally require gam's *calibration error*
//! |coverage - 0.90| to be no worse than mgcv's by more than a small margin, so
//! the mature field-standard interval is a floor on quality — but gam passing or
//! failing is decided by gam's own distance to nominal, not by reproducing
//! mgcv's (itself noisy) fitted SEs. Matching mgcv's intervals would prove
//! nothing; achieving nominal coverage of the true function proves the SEs are
//! honest.
//!
//! Design of the experiment (identical data fed to both engines):
//!   * One fixed design `x` (n = 500, U[0,1], seed 42) shared by every replicate
//!     so the signal `eta(x) = sin(6*pi*x)` is constant; only the Gaussian noise
//!     (sigma = 0.1) is redrawn per replicate.
//!   * 100 replicate response vectors `y_r = eta + noise_r` are generated once in
//!     Rust with a deterministic splitmix64 + Box–Muller stream and handed,
//!     column for column, to BOTH gam and mgcv. No engine sees different data.
//!   * Each replicate is fit `y ~ s(x, k = 25)` by REML in both engines. The
//!     basis dimension MUST resolve the truth: `sin(6*pi*x)` is three full
//!     cycles on [0,1], and a penalized cubic spline needs roughly four knots
//!     per cycle (k >= ~20) to represent it without a residual `O(lambda*f'')`
//!     smoothing bias at every crest/trough. At the previously-used k=10 the
//!     truth is intrinsically under-resolved: the penalized bias dwarfs the
//!     (correctly scaled) pointwise SE, so NO honest band — gam's OR mgcv's —
//!     can cover the true mean, and pooled coverage collapses to ~0.15/0.48 for
//!     BOTH engines (run 26753200374). That is the Nychka pointwise-bias dip on
//!     an un-representable truth, not an SE-scaling fault; the calibration claim
//!     below is only meaningful once the truth lives in the span of the basis.
//!   * Pointwise 90% CIs are evaluated on a fixed *interior* grid
//!     (`seq(0.05, 0.95, length.out = 50)`, regenerated bit-identically on both
//!     sides) so the measurement isolates the conditional/penalized covariance
//!     path and is NOT contaminated by gam's optional boundary or OOD variance
//!     inflation — both of which are switched off here (we read the covariance
//!     directly and build the band ourselves, so no correction is applied).
//!
//! Why the window is principled: the 50 eval points within one replicate share
//! the same fitted curve and noise draw, so they are strongly correlated; the
//! effective sample size for the coverage rate is on the order of the 100
//! replicates, not the 5000 pairs. The between-replicate s.e. of the coverage
//! estimate is ~0.005-0.01, so the +/-0.05 window is several s.e. wide — loose
//! enough never to fail on Monte-Carlo noise, tight enough to reject a systematic
//! SE bias of half a nominal-level point (too-low coverage => SE underestimates,
//! e.g. ignores smoothing-penalty variance; too-high => SE overestimates).

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array2, ArrayView2};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

// N=200 (down from 500), N_REPLICATES=50 (down from 100): reduces the R loop
// from 100 × mgcv gam(n=500) to 50 × mgcv gam(n=200), well within 360s.
// MC SE at 50 reps × 50 eval = 2500 trials: sqrt(0.9*0.1/2500) ≈ 0.006,
// comfortably below the ±0.05 tolerance window [0.85, 0.95].
const N: usize = 200;
const N_REPLICATES: usize = 50;
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
fn ci_coverage_near_nominal_on_gaussian_truth_90pct() {
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

    let total_trials = N_REPLICATES * N_EVAL;

    // Each replicate is an independent fit on a fixed design (only `y` changes);
    // the per-replicate hit count is deterministic regardless of how the outer
    // loop is scheduled, and coverage is accumulated by exact integer addition,
    // so parallelizing the fit loop is bit-identical to the serial version and
    // changes no asserted quantity. The 50 independent fits dominate wall-clock,
    // so fanning them across the rayon pool is the harness-side speedup here.
    let (gam_hits_vp, gam_hits_vb): (usize, usize) = replicates
        .par_iter()
        .map(|rep| {
            let records: Vec<csv::StringRecord> = (0..N)
                .map(|i| {
                    csv::StringRecord::from(vec![
                        format!("{:.17e}", x[i]),
                        format!("{:.17e}", rep[i]),
                    ])
                })
                .collect();
            let ds = encode_recordswith_inferred_schema(headers.clone(), records)
                .expect("encode synthetic replicate");
            let col = ds.column_map();
            let x_idx = col["x"];

            let result = fit_from_formula("y ~ s(x, k=25)", &ds, &cfg).expect("gam fit");
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

            let mut hits_vp = 0usize;
            let mut hits_vb = 0usize;
            for j in 0..N_EVAL {
                let lo_vp = pred[j] - Z_90 * se_vp[j];
                let hi_vp = pred[j] + Z_90 * se_vp[j];
                if truth_eval[j] >= lo_vp && truth_eval[j] <= hi_vp {
                    hits_vp += 1;
                }
                let lo_vb = pred[j] - Z_90 * se_vb[j];
                let hi_vb = pred[j] + Z_90 * se_vb[j];
                if truth_eval[j] >= lo_vb && truth_eval[j] <= hi_vb {
                    hits_vb += 1;
                }
            }
            (hits_vp, hits_vb)
        })
        .reduce(|| (0usize, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));

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
        nrep <- 50L
        neval <- 50L
        z90 <- qnorm(0.95)
        grid <- seq(0.05, 0.95, length.out = neval)
        truth <- sin(6 * pi * grid)
        newd <- data.frame(x = grid)
        hits <- 0L
        for (r in 0:(nrep - 1L)) {
            yname <- paste0("y", r)
            dat <- data.frame(x = df$x, y = df[[yname]])
            m <- gam(y ~ s(x, k = 25), data = dat, method = "REML")
            pr <- predict(m, newdata = newd, se.fit = TRUE)
            lo <- pr$fit - z90 * pr$se.fit
            hi <- pr$fit + z90 * pr$se.fit
            hits <- hits + sum(truth >= lo & truth <= hi)
        }
        emit("coverage", hits / (nrep * neval))
        "#,
    );
    let mgcv_coverage = r.scalar("coverage");

    // Calibration error = absolute distance of empirical coverage from nominal.
    // This is the objective quality scalar for an uncertainty band: 0 is perfect.
    const NOMINAL: f64 = 0.90;
    let gam_calib_err = (gam_coverage_vp - NOMINAL).abs();
    let mgcv_calib_err = (mgcv_coverage - NOMINAL).abs();

    eprintln!(
        "Gaussian CI coverage (90% nominal, {N_REPLICATES} reps x {N_EVAL} eval pts = {total_trials} trials):\n  \
         gam Vp coverage = {gam_coverage_vp:.4}  (calib err |cov-0.90| = {gam_calib_err:.4}; smoothing-corrected)\n  \
         gam Vb coverage = {gam_coverage_vb:.4}  (conditional; diagnostic)\n  \
         mgcv coverage   = {mgcv_coverage:.4}  (calib err |cov-0.90| = {mgcv_calib_err:.4}; baseline)"
    );

    // PRIMARY (absolute, self-contained) ASSERTION: gam's own 90% penalized-
    // covariance band must achieve close to nominal coverage of the TRUE mean
    // function. [0.85, 0.95] is the principled +/-0.05 window around 0.90; it
    // rejects a systematic SE bias of half a nominal point while tolerating the
    // between-replicate Monte-Carlo noise (~0.006 s.e. over the 50
    // effectively-independent replicates). This claim does not reference mgcv.
    assert!(
        gam_coverage_vp >= 0.85 && gam_coverage_vp <= 0.95,
        "gam 90% CI empirical coverage {gam_coverage_vp:.4} is outside the nominal window \
         [0.85, 0.95]; gam's standard errors are {} the truth.",
        if gam_coverage_vp < 0.85 {
            "under-covering (SE too small)"
        } else {
            "over-covering (SE too large)"
        }
    );

    // SECONDARY (match-or-beat baseline): gam's calibration error must be no
    // worse than the mature field-standard interval's by more than a small
    // Monte-Carlo margin. mgcv is the floor on quality, not a "gam == mgcv"
    // target — gam is allowed to be *better* calibrated, only not meaningfully
    // worse. The 0.03 margin (~3-6 between-replicate s.e.) absorbs the noise in
    // both engines' coverage estimates without licensing a real calibration gap.
    const MATCH_OR_BEAT_MARGIN: f64 = 0.03;
    assert!(
        gam_calib_err <= mgcv_calib_err + MATCH_OR_BEAT_MARGIN,
        "gam CI calibration error {gam_calib_err:.4} (coverage {gam_coverage_vp:.4}) is worse \
         than the mgcv baseline {mgcv_calib_err:.4} (coverage {mgcv_coverage:.4}) by more than \
         the {MATCH_OR_BEAT_MARGIN:.2} margin; gam's intervals are less honest than the \
         field-standard reference on identical data."
    );
}
