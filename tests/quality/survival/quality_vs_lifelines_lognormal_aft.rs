//! End-to-end quality: gam's parametric AFT survival with a *covariate
//! interaction* must RECOVER THE KNOWN GENERATING lognormal accelerated-failure-
//! time model from synthetic right-censored data. The data is drawn from a fully
//! specified lognormal AFT with KNOWN parameters
//!   log T = a + b0*x0 + b1*x1 + g*x0*x1 + sigma * W,   W ~ Normal(0, 1),
//! so the ground truth (b0, b1, g, sigma, and the survival surface S(t|x)) is
//! known exactly. This test asserts an OBJECTIVE accuracy claim against that
//! truth, NOT closeness to any reference tool's noisy fit.
//!
//! OBJECTIVE METRIC ASSERTED (truth recovery, no reference dependence):
//!   * RMSE of gam's recovered covariate *slopes* (b0, b1, g — including the
//!     x0:x1 interaction) against the TRUE generating slopes <= a principled
//!     finite-sample MLE bar.
//!   * gam's recovered constant scale `sigma` within a principled relative
//!     tolerance of the TRUE generating `sigma`.
//!   * relative-L2 of gam's reconstructed survival surface S(t|x) (built from
//!     gam's recovered slopes + sigma, gauge-anchored to the true intercept)
//!     against the TRUE survival surface <= a principled bar.
//! All bars are derived from the data-generating process (signal scale, n=150,
//! ~40% censoring), not from how close gam lands to lifelines.
//!
//! BASELINE TO MATCH-OR-BEAT (does not define pass/fail on its own):
//! `lifelines.LogNormalAFTFitter` — the de-facto standard Python parametric-AFT
//! reference — is fit on the IDENTICAL data and its slope-recovery error against
//! the SAME truth is computed. We additionally assert gam's slope error is no
//! worse than lifelines' by more than 10% (match-or-beat on ACCURACY vs the
//! ground truth). The primary claim is truth recovery; lifelines is the bar gam
//! must be competitive with, never the thing gam must reproduce.
//!
//! gam-side mapping (verified against the source). gam's Gaussian-residual
//! survival location-scale family IS a normal AFT: reading the predictor
//! assembly in `families::survival::location_scale`
//! (`survival_location_scale_response_from_predictors`), the standardized
//! survival index is
//!   z(t, x) = h(t) - eta_t(x) * exp(-eta_ls(x)),   S(t|x) = 1 - Phi(z),
//! a *location* channel `eta_t(x)` (role `BlockRole::Threshold`,
//! `beta_threshold()`) and a constant *log-scale* channel `eta_ls(x)` (role
//! `BlockRole::Scale`, link `sigma = exp(eta_ls)`), composed with a learned
//! monotone transform `h(t)` of the time axis. With a Gaussian residual this is
//! the lognormal AFT on the `h(t)`-warped clock. The in-Rust path is selected by
//! `FitConfig{ survival_likelihood: "location-scale", survival_distribution:
//! "gaussian" }` with a `Surv(...)` response (`materialize_survival` routes the
//! RHS to the threshold/location `thresholdspec`); the `survmodel(...)` term in
//! the formula is parsed and carried for documentary fidelity. The fit returns
//! `FitResult::SurvivalLocationScale`; the converged `UnifiedFitResult` exposes
//! the location coefficients via `beta_threshold()` and the log-scale via
//! `beta_log_sigma()`, and the frozen location design is rebuildable at
//! arbitrary covariate rows from `resolved_thresholdspec` through
//! `build_term_collection_design` (the canonical mgcv/gamlss rebuild pattern).
//!
//! The gauge. gam learns the time transform `h(t)` flexibly while the truth (and
//! lifelines) fixes it at `log t`; gam's location channel therefore differs from
//! the truth by an unknown additive *gauge offset* (the absolute location anchor
//! lives entirely in the intercept) and the scale by the (approximately unit)
//! local slope of `h` vs `log t`. The covariate *slope* coefficients `b0, b1, g`
//! are pure differences of `mu(x)` so the additive gauge cancels — they are
//! directly comparable to the true generating slopes. For the survival-surface
//! comparison we anchor gam's intercept to the true intercept `a` (removing only
//! the gauge offset, exactly as the gamlss survival-LS quality test mean-centers
//! its surfaces) so the surface metric measures recovered covariate dependence
//! and scale against truth, not the time-axis gauge.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// Standard normal CDF via erfc (matches scipy / lifelines `norm.cdf`).
fn norm_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// Complementary error function (Numerical-Recipes rational approximation,
/// ~1e-7 absolute — far below any survival-curve tolerance asserted here).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    if x >= 0.0 { ans } else { 2.0 - ans }
}

/// Lognormal-AFT survival `S(t|x) = 1 - Phi((log t - mu) / sigma)`.
fn lognormal_survival(t: f64, mu: f64, sigma: f64) -> f64 {
    1.0 - norm_cdf((t.ln() - mu) / sigma)
}

#[test]
fn gam_lognormal_aft_interaction_recovers_truth() {
    init_parallelism();

    // ---- synthetic recipe, fed IDENTICALLY to gam (in-Rust) and lifelines ----
    // Spec: fixed-seed (seed 1234), n=150,
    //   t ~ LogNormal(mu = a + x0*b0 + x1*b1 + x0*x1*g, sigma = 0.4),
    //   event ~ Bernoulli(0.6), a=-0.2, b0=0.5, b1=-0.3, g=0.2.
    // The data is drawn ONCE here (fixed-seed MT19937, deterministic across runs)
    // and the *same* (t, d, x0, x1) arrays are handed both to gam (in-Rust) and,
    // via the reference harness `Column`s, to the lifelines body as a pandas
    // `df` — a single source of truth, so the two engines fit byte-identical
    // rows. (The draw order x0, x1, z, u mirrors the canonical NumPy recipe.)
    let n = 150usize;
    let a = -0.2_f64;
    let b0 = 0.5_f64;
    let b1 = -0.3_f64;
    let g = 0.2_f64;
    let sigma_true = 0.4_f64;

    let mut rng = NumpyMt19937::new(1234);
    // Match the Python draw order below: x0 (uniform), x1 (uniform),
    // z (standard normal for the lognormal), u (uniform for the event).
    let x0: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    let x1: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    let z: Vec<f64> = (0..n).map(|_| rng.next_standard_normal()).collect();
    let u: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();

    let mut t = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for i in 0..n {
        let mu = a + x0[i] * b0 + x1[i] * b1 + x0[i] * x1[i] * g;
        t.push((mu + sigma_true * z[i]).exp());
        d.push(if u[i] < 0.6 { 1.0 } else { 0.0 });
    }

    // ---- build the gam dataset (columns: t, d, x0, x1) ---------------------
    let headers: Vec<String> = ["t", "d", "x0", "x1"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", t[i]),
                format!("{:.17e}", d[i]),
                format!("{:.17e}", x0[i]),
                format!("{:.17e}", x1[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode lognormal-AFT data");
    let col = ds.column_map();
    let x0_idx = col["x0"];
    let x1_idx = col["x1"];
    let ncols = ds.headers.len();

    // ---- fit with gam: lognormal AFT with the x0:x1 interaction ------------
    // Gaussian-residual survival location-scale == lognormal AFT (see module
    // doc). The `survmodel(...)` term mirrors the spec verbatim; the
    // location-scale path + Gaussian residual is what makes this the lognormal
    // family. No noise_formula => a single constant log-scale (sigma) channel,
    // matching lifelines' constant `sigma_`.
    let cfg = FitConfig {
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        // Default flexible time warp (8 internal knots). An earlier commit scoped
        // this to 2 knots claiming it let gam "recover the affine surface and beat
        // lifelines" — that was false: gam still fails this test badly. Measured,
        // gam recovers slopes (0.21, -0.11, 0.00) vs truth (0.5, -0.3, 0.2) and
        // sigma 0.19 vs 0.40. Two real defects are exposed, neither a knot-count
        // artifact:
        //   * a multiplicative warp-gauge scale (h -> c*h rescales BOTH the slopes
        //     and sigma by c; here c ~= sigma_gam/sigma_true ~= 0.48) that this
        //     test's "gauge-free" subtraction does NOT remove — it cancels only the
        //     additive intercept offset, not the multiplicative scale; and
        //   * the x0:x1 interaction collapsing to exactly 0 even after gauge
        //     normalization (slope/sigma ratios: gam (1.08, -0.58, 0.0) vs truth
        //     (1.25, -0.75, 0.5)) — genuine over-shrinkage of the interaction.
        // Restored to the honest default so the failure reflects gam's real
        // behaviour, not a hand-tuned basis. The fix is location-scale survival
        // fitting work (warp-scale identification + interaction shrinkage).
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        r#"Surv(t, d) ~ x0 + x1 + x0:x1 + survmodel(spec="transformation", distribution="lognormal")"#,
        &ds,
        &cfg,
    )
    .expect("gam lognormal-AFT fit");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;
    assert!(
        unified.outer_converged,
        "gam lognormal-AFT outer optimizer did not converge: iters={} grad_norm={:?}",
        unified.outer_iterations, unified.outer_gradient_norm
    );

    let beta_location = unified.beta_threshold();
    let beta_log_sigma = unified.beta_log_sigma();
    assert!(
        beta_location
            .iter()
            .chain(beta_log_sigma.iter())
            .all(|v| v.is_finite()),
        "non-finite gam location / log-sigma coefficients"
    );

    // The four covariate combinations the spec evaluates on.
    let combos: [(f64, f64); 4] = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)];

    // Rebuild the frozen location (threshold) design at the four combos and
    // apply the converged location coefficients: this is the AFT location
    // predictor mu_gam(x) on gam's learned-time-transform gauge. Same rebuild
    // pattern as the mgcv / gamlss quality tests.
    let mut combo_grid = Array2::<f64>::zeros((combos.len(), ncols));
    for (i, &(c0, c1)) in combos.iter().enumerate() {
        combo_grid[[i, x0_idx]] = c0;
        combo_grid[[i, x1_idx]] = c1;
    }
    let loc_design =
        build_term_collection_design(combo_grid.view(), &fit.fit.resolved_thresholdspec)
            .expect("rebuild location (threshold) design at covariate combos");
    let gam_mu_combo: Vec<f64> = loc_design.design.apply(&beta_location).to_vec();
    assert_eq!(gam_mu_combo.len(), combos.len());

    // Constant log-scale channel: sigma = exp(eta_ls) at any x (intercept-only).
    let ls_design =
        build_term_collection_design(combo_grid.view(), &fit.fit.resolved_log_sigmaspec)
            .expect("rebuild log-sigma design at covariate combos");
    let gam_eta_ls: Vec<f64> = ls_design.design.apply(&beta_log_sigma).to_vec();
    let gam_sigma = gam_eta_ls[0].exp();
    assert!(
        gam_eta_ls.iter().all(|&v| (v - gam_eta_ls[0]).abs() < 1e-9),
        "expected a constant (x-independent) log-scale channel, got {gam_eta_ls:?}"
    );

    // Recover gam's location *slopes* (b0, b1, g) and intercept from mu at the
    // four combos by inverting the 4x4 design [1, x0, x1, x0*x1]. The slopes are
    // the gauge-free AFT effects (the additive time-gauge offset lives entirely
    // in the intercept). With x in {-1,+1} the inverse is the orthogonal
    // Hadamard contrast (each coefficient is a simple sign-weighted average).
    let mu = &gam_mu_combo;
    let gam_intercept = 0.25 * (mu[0] + mu[1] + mu[2] + mu[3]);
    // x0 contrast: +1 on combos (1,-1),(1,1) [idx 2,3]; -1 on idx 0,1.
    let gam_b0 = 0.25 * (-mu[0] - mu[1] + mu[2] + mu[3]);
    // x1 contrast: +1 on combos (-1,1),(1,1) [idx 1,3]; -1 on idx 0,2.
    let gam_b1 = 0.25 * (-mu[0] + mu[1] - mu[2] + mu[3]);
    // x0*x1 contrast: +1 on (-1,-1),(1,1) [idx 0,3]; -1 on (-1,1),(1,-1).
    let gam_g = 0.25 * (mu[0] - mu[1] - mu[2] + mu[3]);

    // ---- BASELINE: fit the SAME data with lifelines.LogNormalAFT ------------
    // lifelines is the match-or-beat baseline (not the truth). The harness hands
    // the body the *exact* (t, d, x0, x1) columns gam was fit on as a pandas
    // `df` (one source of truth for the synthetic data, drawn once on the Rust
    // side from the fixed seed), so the rows are byte-identical to gam's.
    // LogNormalAFT fits log T = mu(x) + sigma W under right-censoring; mu_ holds
    // the location coefficients (Intercept, x0, x1, x0:x1) and sigma_ the scale.
    let body = r#"
import numpy as np
import pandas as pd
from lifelines import LogNormalAFTFitter

frame = pd.DataFrame({
    "t": np.asarray(df["t"], dtype=float),
    "d": np.asarray(df["d"], dtype=float),
    "x0": np.asarray(df["x0"], dtype=float),
    "x1": np.asarray(df["x1"], dtype=float),
})
frame["x0x1"] = frame["x0"] * frame["x1"]
aft = LogNormalAFTFitter()
aft.fit(frame, duration_col="t", event_col="d",
        formula="x0 + x1 + x0x1")

# Location ('mu_') coefficients on the natural-log time scale.
mu_params = aft.params_.loc["mu_"]
emit("intercept", [float(mu_params["Intercept"])])
emit("b0", [float(mu_params["x0"])])
emit("b1", [float(mu_params["x1"])])
emit("g", [float(mu_params["x0x1"])])

# Constant scale sigma = exp(sigma_ intercept on the log scale).
sigma_params = aft.params_.loc["sigma_"]
emit("sigma", [float(np.exp(sigma_params["Intercept"]))])
"#;
    let r = run_python(
        &[
            Column::new("t", &t),
            Column::new("d", &d),
            Column::new("x0", &x0),
            Column::new("x1", &x1),
        ],
        body,
    );
    let ref_b0 = r.scalar("b0");
    let ref_b1 = r.scalar("b1");
    let ref_g = r.scalar("g");
    let ref_sigma = r.scalar("sigma");

    // ---- OBJECTIVE METRIC: recovery of the TRUE generating parameters -------
    // The covariate slopes are gauge-free (pure differences of mu(x)), so they
    // are directly comparable to the TRUE generating slopes (b0, b1, g). This is
    // the primary, reference-free accuracy claim.
    let true_slopes = [b0, b1, g];
    let gam_slopes = [gam_b0, gam_b1, gam_g];
    let ref_slopes = [ref_b0, ref_b1, ref_g];
    // gam's slope-recovery error vs TRUTH (the metric we assert).
    let gam_slope_rmse = rmse(&gam_slopes, &true_slopes);
    // lifelines' slope-recovery error vs the SAME truth (match-or-beat baseline).
    let ref_slope_rmse = rmse(&ref_slopes, &true_slopes);
    // gam's scale recovery vs TRUTH.
    let gam_sigma_rel = (gam_sigma - sigma_true).abs() / sigma_true;
    let ref_sigma_rel = (ref_sigma - sigma_true).abs() / sigma_true;

    // ---- OBJECTIVE METRIC: survival surface vs the TRUE surface -------------
    // gam learns h(t) so its absolute location anchor (intercept) lives on a
    // different gauge than log-t; we anchor gam's intercept to the TRUE
    // intercept `a` (removing only the gauge offset, exactly as the gamlss
    // survival-LS quality test mean-centers its surfaces) and reconstruct the
    // lognormal survival from gam's recovered (slopes, sigma). This measures how
    // well gam recovers the TRUE covariate dependence and scale, not the gauge.
    let grid_n = 20usize;
    let (t_lo, t_hi) = (0.1_f64, 10.0_f64);
    let time_grid: Vec<f64> = (0..grid_n)
        .map(|i| t_lo + (t_hi - t_lo) * i as f64 / (grid_n as f64 - 1.0))
        .collect();

    let mut gam_surv = Vec::with_capacity(grid_n * combos.len());
    let mut true_surv = Vec::with_capacity(grid_n * combos.len());
    for &(c0, c1) in &combos {
        // gam location with the gauge offset removed (anchored to the truth `a`).
        let gam_mu = a + gam_b0 * c0 + gam_b1 * c1 + gam_g * c0 * c1;
        let truth_mu = a + b0 * c0 + b1 * c1 + g * c0 * c1;
        for &tt in &time_grid {
            gam_surv.push(lognormal_survival(tt, gam_mu, gam_sigma));
            true_surv.push(lognormal_survival(tt, truth_mu, sigma_true));
        }
    }
    let surv_rel_vs_truth = relative_l2(&gam_surv, &true_surv);
    // lifelines' reconstructed surface vs the SAME truth — the match-or-beat
    // baseline for the surface arm, mirroring the slope and scale arms. Built
    // from lifelines' recovered (slopes, sigma) on the same anchor `a` and grid.
    // The surface is a deterministic function of the recovered slopes+sigma, so
    // its truth-error is bounded below by the finite-sample slope/scale error —
    // at n=150 the low-leverage x0:x1 interaction MLE (var(x0*x1) smallest) has
    // large sampling variance, so even the gold-standard lifelines MLE
    // reconstructs the true surface only to ~0.14 rel_l2. The absolute bar below
    // is set to that finite-sample floor; the match-or-beat assert is the real
    // gate (gam must be no worse than the mature reference on the same truth).
    let mut ref_surv = Vec::with_capacity(grid_n * combos.len());
    for &(c0, c1) in &combos {
        let ref_mu = a + ref_b0 * c0 + ref_b1 * c1 + ref_g * c0 * c1;
        for &tt in &time_grid {
            ref_surv.push(lognormal_survival(tt, ref_mu, ref_sigma));
        }
    }
    let ref_surv_rel = relative_l2(&ref_surv, &true_surv);

    eprintln!(
        "lognormal AFT truth recovery: n={n} \
         truth(b0,b1,g,sigma)=({b0:.4},{b1:.4},{g:.4},{sigma_true:.4}) \
         gam(b0,b1,g,sigma)=({gam_b0:.4},{gam_b1:.4},{gam_g:.4},{gam_sigma:.4}) \
         lifelines(b0,b1,g,sigma)=({ref_b0:.4},{ref_b1:.4},{ref_g:.4},{ref_sigma:.4}) \
         gam_slope_rmse_vs_truth={gam_slope_rmse:.4} (baseline lifelines={ref_slope_rmse:.4}) \
         gam_sigma_rel_vs_truth={gam_sigma_rel:.4} (baseline lifelines={ref_sigma_rel:.4}) \
         gam_intercept={gam_intercept:.4} S_rel_l2_vs_truth={surv_rel_vs_truth:.4}"
    );

    // ---- PRIMARY: gam recovers the true generating slopes ------------------
    //  Bar 0.12: with n=150 under ~40% censoring the MLE standard error on each
    //  slope is ~0.07-0.10; an RMSE over the three slopes at or below this scale
    //  means gam has genuinely recovered the covariate effects (incl. the x0:x1
    //  interaction), within sampling noise of the truth. Not weakened to pass:
    //  this is the finite-sample noise floor of the generating process.
    assert!(
        gam_slope_rmse <= 0.12,
        "gam did not recover the true AFT location slopes: rmse_vs_truth={gam_slope_rmse:.4} \
         gam=({gam_b0:.4},{gam_b1:.4},{gam_g:.4}) truth=({b0:.4},{b1:.4},{g:.4})"
    );
    //  Match-or-beat: gam's accuracy must be competitive with the mature
    //  reference on the SAME truth (no worse than 10% beyond lifelines' error).
    assert!(
        gam_slope_rmse <= ref_slope_rmse * 1.10 + 1e-9,
        "gam slope recovery worse than lifelines baseline: gam={gam_slope_rmse:.4} \
         lifelines={ref_slope_rmse:.4} (allowed {:.4})",
        ref_slope_rmse * 1.10
    );

    // ---- PRIMARY: gam recovers the true scale -------------------------------
    //  Bar 12%: the lognormal scale MLE has finite-sample relative SE ~ 1/sqrt(2 n
    //  d) ~ 9% at n=150 with ~40% events; 12% admits sampling noise without
    //  letting a biased scale through.
    assert!(
        gam_sigma_rel <= 0.12,
        "gam did not recover the true AFT scale: gam_sigma={gam_sigma:.4} \
         sigma_true={sigma_true:.4} rel={gam_sigma_rel:.4}"
    );
    //  Match-or-beat on the scale too.
    assert!(
        gam_sigma_rel <= ref_sigma_rel * 1.10 + 1e-9,
        "gam scale recovery worse than lifelines baseline: gam_rel={gam_sigma_rel:.4} \
         lifelines_rel={ref_sigma_rel:.4} (allowed {:.4})",
        ref_sigma_rel * 1.10
    );

    // ---- PRIMARY: gam's survival surface matches the TRUE surface -----------
    //  The surface is reconstructed deterministically from the recovered
    //  (slopes, sigma) — whose own bars above are 0.12 — so its truth-error
    //  inherits that finite-sample noise (and then some, through the nonlinear
    //  S = 1 - Phi map at the grid corners where the x0:x1 interaction lands).
    //  Absolute bar 0.18: the n=150 surface floor that even the gold-standard
    //  lifelines MLE incurs (its g-MLE ~0.37 vs true 0.2 on this draw moves the
    //  corner surfaces); 0.05 was internally inconsistent with the 0.12 slope
    //  bar and unachievable by ANY correct estimator here.
    assert!(
        surv_rel_vs_truth <= 0.18,
        "gam survival surface diverges from the TRUE surface: rel_l2={surv_rel_vs_truth:.4}"
    );
    //  Match-or-beat: gam's surface must be no worse than lifelines' on the same
    //  truth (the real gate, mirroring the slope and scale arms).
    assert!(
        surv_rel_vs_truth <= ref_surv_rel * 1.10 + 1e-9,
        "gam survival surface worse than lifelines baseline: gam={surv_rel_vs_truth:.4} \
         lifelines={ref_surv_rel:.4} (allowed {:.4})",
        ref_surv_rel * 1.10
    );
}

/// Minimal fixed-seed MT19937 generator (MT19937 core, 53-bit uniform in
/// [0, 1) matching NumPy's `random_sample`, and a polar-Marsaglia Gaussian
/// matching NumPy's legacy `gauss`). It only needs to be *deterministic* across
/// runs: the drawn arrays are the single source of truth fed to BOTH engines,
/// so reproducing NumPy's exact stream is unnecessary for identical-data parity.
struct NumpyMt19937 {
    mt: [u32; 624],
    idx: usize,
    has_gauss: bool,
    gauss: f64,
}

impl NumpyMt19937 {
    fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            mt,
            idx: 624,
            has_gauss: false,
            gauss: 0.0,
        }
    }

    fn generate(&mut self) {
        const MATRIX_A: u32 = 0x9908b0df;
        const UPPER: u32 = 0x80000000;
        const LOWER: u32 = 0x7fffffff;
        for i in 0..624 {
            let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % 624] & LOWER);
            let mut next = self.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.mt[i] = next;
        }
        self.idx = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.generate();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    /// NumPy `random_sample`: 53-bit double in [0, 1).
    fn next_f64(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as u64; // 27 bits
        let b = (self.next_u32() >> 6) as u64; // 26 bits
        (a as f64 * 67108864.0 + b as f64) / 9007199254740992.0
    }

    /// NumPy legacy `standard_normal`: polar Marsaglia, matching `RandomState`.
    fn next_standard_normal(&mut self) -> f64 {
        if self.has_gauss {
            self.has_gauss = false;
            return self.gauss;
        }
        loop {
            let x1 = 2.0 * self.next_f64() - 1.0;
            let x2 = 2.0 * self.next_f64() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.gauss = f * x1;
                self.has_gauss = true;
                return f * x2;
            }
        }
    }
}
