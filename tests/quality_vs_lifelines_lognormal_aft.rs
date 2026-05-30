//! End-to-end quality: gam's parametric AFT survival with a *covariate
//! interaction* must recover the same lognormal accelerated-failure-time model
//! that `lifelines.LogNormalAFT` — the de-facto standard Python parametric-AFT
//! reference — recovers on the *identical* synthetic right-censored data.
//!
//! Capability under test: AFT lognormal distribution with covariate
//! interaction, requested via the survival formula
//!   `Surv(t, d) ~ x0 + x1 + x0:x1 + survmodel(spec="transformation",
//!                                             distribution="lognormal")`.
//!
//! Why `lifelines.LogNormalAFT` is the right reference. A lognormal AFT models
//!   log T = mu(x) + sigma * W,   W ~ Normal(0, 1),
//! i.e. `S(t|x) = 1 - Phi((log t - mu(x)) / sigma)` with `mu(x)` linear in the
//! covariates (here `mu(x) = a + b0*x0 + b1*x1 + g*x0*x1`) and a constant log
//! scale `sigma`. lifelines fits exactly this by maximum likelihood under
//! right-censoring and exposes `mu_` (the location coefficients, including the
//! `Intercept`) and `sigma_` (the constant scale).
//!
//! gam-side mapping (verified against the source). gam's Gaussian-residual
//! survival location-scale family IS a normal AFT: reading the predictor
//! assembly in `families::survival_location_scale`
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
//! What IS and is NOT directly comparable (the gauge). gam learns the time
//! transform `h(t)` flexibly while lifelines fixes it at `log t`; the two
//! location channels therefore differ by an unknown additive *gauge offset*
//! (the absolute location anchor) and the scale by the (approximately unit)
//! local slope of `h` vs `log t`. The engine-agnostic invariants are (a) the
//! covariate *slope* coefficients `b0, b1, g` (the interaction!) — pure
//! differences of `mu(x)`, so the additive gauge cancels — and (b) the constant
//! scale `sigma`. We therefore assert agreement on the location *slopes* and on
//! `sigma`, and we reconstruct `S(t|x)` from each engine's recovered lognormal
//! parameters with the *gauge offset removed* (gam's location intercept matched
//! to lifelines', exactly as the gamlss survival-LS quality test mean-centers
//! its surfaces) so the survival comparison measures the recovered covariate
//! dependence and scale rather than the time-axis gauge.

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
fn gam_lognormal_aft_interaction_matches_lifelines() {
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

    // ---- fit the SAME data with lifelines.LogNormalAFT (mature reference) ---
    // The harness hands the body the *exact* (t, d, x0, x1) columns gam was fit
    // on as a pandas `df` (one source of truth for the synthetic data, drawn
    // once on the Rust side from the fixed seed), so the rows are byte-identical
    // to gam's. LogNormalAFT fits log T = mu(x) + sigma W under right-censoring;
    // mu_ holds the location coefficients (Intercept, x0, x1, x0:x1) and sigma_
    // the constant scale.
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
    let ref_intercept = r.scalar("intercept");
    let ref_b0 = r.scalar("b0");
    let ref_b1 = r.scalar("b1");
    let ref_g = r.scalar("g");
    let ref_sigma = r.scalar("sigma");

    // ---- compare the AFT location SLOPE coefficients (gauge-free) ----------
    let gam_slopes = [gam_b0, gam_b1, gam_g];
    let ref_slopes = [ref_b0, ref_b1, ref_g];
    let coef_rmse = rmse(&gam_slopes, &ref_slopes);
    let sigma_rel = (gam_sigma - ref_sigma).abs() / ref_sigma.abs().max(1e-12);

    // ---- compare S(t|x) on the 20 x 4 grid (gauge offset removed) ----------
    // gam learns h(t) while lifelines fixes log t, so the absolute location
    // anchor (intercept) lives on different gauges; we re-anchor gam's intercept
    // to lifelines' (subtracting the gauge offset, exactly as the gamlss
    // survival-LS quality test mean-centers its surfaces) and reconstruct the
    // lognormal survival from each engine's recovered (intercept+slopes, sigma)
    // with the identical closed form. This measures whether gam recovers the
    // same covariate dependence and scale as lifelines, not the time gauge.
    let grid_n = 20usize;
    let (t_lo, t_hi) = (0.1_f64, 10.0_f64);
    let time_grid: Vec<f64> = (0..grid_n)
        .map(|i| t_lo + (t_hi - t_lo) * i as f64 / (grid_n as f64 - 1.0))
        .collect();

    let mut gam_surv = Vec::with_capacity(grid_n * combos.len());
    let mut ref_surv = Vec::with_capacity(grid_n * combos.len());
    for &(c0, c1) in &combos {
        // gam location with the gauge offset removed (re-anchored to lifelines).
        let gam_mu = ref_intercept + gam_b0 * c0 + gam_b1 * c1 + gam_g * c0 * c1;
        let ref_mu = ref_intercept + ref_b0 * c0 + ref_b1 * c1 + ref_g * c0 * c1;
        for &tt in &time_grid {
            gam_surv.push(lognormal_survival(tt, gam_mu, gam_sigma));
            ref_surv.push(lognormal_survival(tt, ref_mu, ref_sigma));
        }
    }
    let surv_rel = relative_l2(&gam_surv, &ref_surv);

    eprintln!(
        "lognormal AFT vs lifelines: n={n} \
         gam(b0,b1,g)=({gam_b0:.4},{gam_b1:.4},{gam_g:.4}) \
         ref(b0,b1,g)=({ref_b0:.4},{ref_b1:.4},{ref_g:.4}) \
         coef_rmse={coef_rmse:.4} \
         gam_sigma={gam_sigma:.4} ref_sigma={ref_sigma:.4} sigma_rel={sigma_rel:.4} \
         gam_intercept={gam_intercept:.4} ref_intercept={ref_intercept:.4} \
         S_rel_l2={surv_rel:.4}"
    );

    // Bounds (spec-derived, principled):
    //  * S(t|x) rel_l2 <= 0.02: both engines reconstruct the SAME lognormal
    //    survival from their recovered (slopes, sigma) once the time gauge is
    //    removed; AFT is fully parametric so this is tight. Looser than a
    //    single-smooth mgcv bound (~0.005) because lognormal scale estimation is
    //    mildly asymmetric under censoring.
    assert!(
        surv_rel <= 0.02,
        "lognormal-AFT survival surface diverges from lifelines: rel_l2={surv_rel:.4}"
    );
    //  * AFT location-slope rmse <= 0.08: the covariate effects (incl. the
    //    x0:x1 interaction) are gauge-free and must coincide. With n=150 under
    //    40% censoring this is the principled MLE-agreement margin.
    assert!(
        coef_rmse <= 0.08,
        "AFT location slope coefficients diverge from lifelines: rmse={coef_rmse:.4} \
         gam=({gam_b0:.4},{gam_b1:.4},{gam_g:.4}) ref=({ref_b0:.4},{ref_b1:.4},{ref_g:.4})"
    );
    //  * log-scale within 5% relative: lognormal scale is the second AFT
    //    parameter; the time-axis warp is locally ~unit-slope so sigma is
    //    preserved. 5% reflects the spec's allowance for scale asymmetry.
    assert!(
        sigma_rel <= 0.05,
        "AFT log-scale parameter diverges from lifelines: gam_sigma={gam_sigma:.4} \
         ref_sigma={ref_sigma:.4} rel={sigma_rel:.4}"
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
