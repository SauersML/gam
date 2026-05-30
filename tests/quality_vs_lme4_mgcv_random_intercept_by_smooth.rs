//! End-to-end quality: gam's COMBINATION of a shared smooth, a by-factor smooth,
//! and a random intercept — `y ~ s(x) + s(x, by=g) + group(g)`. The data is
//! synthesized from a KNOWN generative law, so the test asserts OBJECTIVE TRUTH
//! RECOVERY, not agreement with any reference tool.
//!
//! Generative truth (per row i in group g):
//!   μ_i  = base_smooth(x_i) + 0.15·slope_g·x_i + intercept_g      (the true mean)
//!   y_i  = μ_i + ε_i,   ε_i ~ N(0, σ_ε²),  σ_ε² = RESID_VAR
//! with one slope_g and one intercept_g drawn per group. The model must PARTITION
//! this without double-counting: the per-group vertical offset belongs to
//! `group(g)`, the per-group curvature/tilt to `s(x,by=g)`, the shared shape to
//! `s(x)`, and the leftover scatter to σ_ε². A double-penalization or
//! mis-attribution bug corrupts the recovered mean, the recovered intercepts, or
//! the recovered noise scale — all three are measured against the TRUE values.
//!
//! OBJECTIVE assertions (truth recovery; bars are principled, not tool-matched):
//!   1. MEAN RECOVERY: RMSE(gam_fitted, true μ) over all n rows must sit below the
//!      irreducible noise floor — RMSE ≤ 0.5·σ_ε. A fit cannot do better than the
//!      noise it cannot see; landing well under σ_ε proves the structured signal
//!      (shared smooth + per-group tilt + per-group offset) was recovered, not the
//!      noise overfit.
//!   2. INTERCEPT RECOVERY: the per-group offsets gam attributes to `group(g)`
//!      (read at a common x and centered across groups) must track the TRUE
//!      centered intercepts intercept_g with small absolute RMSE
//!      (≤ 0.10, i.e. ≤ 25% of the intercept SD 0.4). Misattributing the offset to
//!      the smooth's null space inflates this error.
//!   3. NOISE RECOVERY: gam's estimated residual variance σ̂_ε² must be within 20%
//!      of the TRUE σ_ε² = RESID_VAR. Double-penalization inflates it; overfitting
//!      deflates it.
//!
//! BASELINE-TO-MATCH-OR-BEAT (mature tools fit on the IDENTICAL data, then held to
//! the same objective metrics — gam must be at least as accurate, never merely
//! "close to" them):
//!   * `mgcv::gam(y ~ s(x) + s(x, by=g))` — its fitted mean error vs true μ is the
//!     accuracy bar for (1): gam's mean RMSE ≤ 1.10·mgcv's mean RMSE.
//!   * `lme4::lmer(y ~ ns(x) + (1|g))` — its σ̂_ε² error vs true σ_ε² is the bar
//!     for (3): gam's |σ̂² − σ²| ≤ 1.10·lme4's |σ̂² − σ²|.
//! The reference rel-L2 / curve correlations are still COMPUTED and printed for
//! context, but no pass/fail criterion is "gam reproduces the reference output".
//! A genuine recovery shortfall failing these bars is a real bug, not something to
//! loosen.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_GROUPS: usize = 5;
const PER_GROUP: usize = 60;
const SEED: u64 = 88;
// Residual noise ε ~ N(0, 0.2²): rand_distr::Normal takes (mean, std_dev), so we
// pass 0.2; the residual *variance* σ_ε² we compare against is therefore 0.04.
const RESID_SD: f64 = 0.2;
const RESID_VAR: f64 = RESID_SD * RESID_SD;

/// Shared base smooth: sin(3πx).
fn base_smooth(x: f64) -> f64 {
    (3.0 * std::f64::consts::PI * x).sin()
}

#[test]
fn gam_random_intercept_by_smooth_recovers_truth() {
    init_parallelism();

    // ---- synthesize identical data for all engines -----------------------
    // n=300, 5 groups (60 ea), x~U(0,1), g∈{0..4}.
    // y = base_smooth(x) + 0.15*slope_g*x + intercept_g + ε.
    // Per-group slope_g ~ N(0,0.5) tilts the curve (the by-smooth deviation);
    // intercept_g ~ N(0,0.4) is the vertical offset (the random intercept).
    // Rows are emitted group-blocked with string labels "g0".."g4" so the schema
    // inferrer treats `g` as categorical and first-appearance order makes the
    // encoded level index equal the group number.
    let n = N_GROUPS * PER_GROUP;
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let slope_dist = Normal::new(0.0, 0.5).expect("slope normal");
    let intercept_dist = Normal::new(0.0, 0.4).expect("intercept normal");
    let noise = Normal::new(0.0, RESID_SD).expect("noise normal");

    // Draw the per-group effects up front (one slope, one intercept per group).
    let slope_g: Vec<f64> = (0..N_GROUPS).map(|_| slope_dist.sample(&mut rng)).collect();
    let intercept_g: Vec<f64> = (0..N_GROUPS)
        .map(|_| intercept_dist.sample(&mut rng))
        .collect();

    let mut x = Vec::<f64>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    let mut true_mean = Vec::<f64>::with_capacity(n); // noise-free generative μ per row
    let mut g_code = Vec::<f64>::with_capacity(n); // numeric group index for R
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for grp in 0..N_GROUPS {
        for _ in 0..PER_GROUP {
            let xi = ux.sample(&mut rng);
            let mui = base_smooth(xi) + 0.15 * slope_g[grp] * xi + intercept_g[grp];
            let yi = mui + noise.sample(&mut rng);
            x.push(xi);
            y.push(yi);
            true_mean.push(mui);
            g_code.push(grp as f64);
            rows.push(StringRecord::from(vec![
                format!("{xi}"),
                format!("g{grp}"),
                format!("{yi}"),
            ]));
        }
    }

    // True per-group intercepts, centered across groups (the offset estimand). The
    // shared s(x) + global intercept is a per-group constant only when read at a
    // common x, so centering the TRUE intercept_g the same way gam's anchor is
    // centered makes the two directly comparable.
    let true_intercept_mean = intercept_g.iter().sum::<f64>() / N_GROUPS as f64;
    let true_intercept_dev: Vec<f64> = intercept_g.iter().map(|v| v - true_intercept_mean).collect();

    let headers = vec!["x".to_string(), "g".to_string(), "y".to_string()];
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode combined dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];

    // ---- fit with gam: y ~ s(x) + s(x, by=g) + group(g), REML -------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x) + s(x, by=g) + group(g)", &ds, &cfg).expect("gam combined fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian RE + by-smooth combination");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");
    // Gaussian identity stores σ_ε in standard_deviation per UnifiedFitResult.
    let gam_resid_var = fit.fit.standard_deviation * fit.fit.standard_deviation;

    // gam fitted mean at the training points: rebuild the frozen design at the
    // observed (x, g) and apply beta (identity link => design*beta = mean). Each
    // row carries its own encoded level index, so it is evaluated against its own
    // by-smooth block AND its own random intercept.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, g_idx]] = g_code[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_fitted.len(), n, "gam fitted length mismatch");

    // Per-group predicted intercept: evaluate at a single common x reference for
    // every group, then center across groups. Reading every group at the SAME x
    // makes the shared s(x) + global-intercept contribution a per-group CONSTANT,
    // which the across-group centering removes exactly. What survives is
    // group(g)'s BLUP-equivalent offset plus the by-smooth's value at x_ref,
    // s(x,by=g)(x_ref); the latter is the small linear deviation 0.15*slope_g*x_ref
    // (|x_ref|≈0.5, slope_g~N(0,0.5) ⇒ SD≈0.04 vs the intercept SD 0.4, i.e. ~10%
    // contamination), so the centered vector is dominated by — and tracks the SHAPE
    // of — the random intercept that we compare to lme4's conditional modes.
    let x_ref = x.iter().sum::<f64>() / n as f64;
    let mut anchor = Array2::<f64>::zeros((N_GROUPS, ds.headers.len()));
    for grp in 0..N_GROUPS {
        anchor[[grp, x_idx]] = x_ref;
        anchor[[grp, g_idx]] = grp as f64;
    }
    let anchor_design = build_term_collection_design(anchor.view(), &fit.resolvedspec)
        .expect("rebuild design at group anchors");
    let gam_group_pred: Vec<f64> = anchor_design.design.apply(&fit.fit.beta).to_vec();
    let gam_anchor_mean = gam_group_pred.iter().sum::<f64>() / N_GROUPS as f64;
    let gam_intercept_dev: Vec<f64> = gam_group_pred.iter().map(|v| v - gam_anchor_mean).collect();

    // ---- mgcv: the per-group fixed smooth s(x) + s(x, by=g) ----------------
    // mgcv's by=<factor> adds a level-specific deviation to the shared s(x); the
    // per-group fitted smooth is the sum. We DON'T model the random intercept in
    // mgcv (it would soak the offset into the smooth's null space); instead we
    // remove the offset from BOTH engines by within-group centering below, so the
    // comparison is purely on the smooth SHAPE that s(x,by=g) is responsible for.
    let r_mgcv = run_r(
        &[
            Column::new("x", &x),
            Column::new("g", &g_code),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        df$g <- as.factor(df$g)
        m <- gam(y ~ s(x) + s(x, by = g), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r_mgcv.vector("fitted");
    let mgcv_edf = r_mgcv.scalar("edf");
    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");

    // ---- lme4: the random-intercept variance component + conditional modes -
    // Fixed natural-spline main effect of x (df matched to a typical s(x) edf)
    // plus (1|g). VarCorr gives σ_g²/σ_ε²; ranef gives per-group BLUPs g0..g4.
    let r_lme4 = run_r(
        &[
            Column::new("x", &x),
            Column::new("g", &g_code),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(lme4))
        suppressPackageStartupMessages(library(splines))
        df$g <- factor(df$g, levels = as.character(sort(unique(df$g))))
        m <- lmer(y ~ ns(x, df = 6) + (1 | g), data = df, REML = TRUE)
        vc <- as.data.frame(VarCorr(m))
        sigma_e2 <- vc$vcov[vc$grp == "Residual"]
        re <- ranef(m)$g[, "(Intercept)"]
        emit("sigma_e2", sigma_e2)
        emit("ranef", as.numeric(re))
        "#,
    );
    let lme4_sigma_e2 = r_lme4.scalar("sigma_e2");
    let lme4_ranef = r_lme4.vector("ranef");
    assert_eq!(
        lme4_ranef.len(),
        N_GROUPS,
        "lme4 returned {} conditional modes, expected {N_GROUPS}",
        lme4_ranef.len()
    );

    // ---- OBJECTIVE METRIC (1): mean recovery vs TRUE μ ---------------------
    // RMSE of each engine's fitted mean against the noise-free generative μ. This
    // is the direct accuracy quantity: how close the recovered structured signal
    // (shared smooth + per-group tilt + per-group offset) is to the truth. mgcv's
    // fitted mean is its s(x)+s(x,by=g) prediction on the same rows — its error vs
    // the SAME true μ is the match-or-beat accuracy bar.
    let gam_mean_rmse = rmse(&gam_fitted, &true_mean);
    let mgcv_mean_rmse = rmse(mgcv_fitted, &true_mean);

    // ---- OBJECTIVE METRIC (2): intercept recovery vs TRUE offsets ---------
    // gam's per-group offset (anchor read at common x, centered across groups)
    // against the TRUE centered intercept_g. Absolute RMSE in the units of the
    // intercept (SD 0.4); the small by-smooth-at-x_ref contamination (~10%) lives
    // inside gam's value, so a 0.10 bar (25% of the intercept SD) is principled.
    let intercept_rmse = rmse(&gam_intercept_dev, &true_intercept_dev);

    // ---- OBJECTIVE METRIC (3): noise-scale recovery vs TRUE σ_ε² ----------
    // gam's estimated residual variance vs the TRUE σ_ε² = RESID_VAR. lme4's
    // estimate vs the same truth is the match-or-beat bar.
    let gam_var_err = (gam_resid_var - RESID_VAR).abs();
    let lme4_var_err = (lme4_sigma_e2 - RESID_VAR).abs();

    // Context-only reference closeness (printed, never asserted as pass/fail): how
    // near gam's fitted mean is to mgcv's, and gam's group offsets to lme4's BLUPs.
    let gam_vs_mgcv_rel_l2 = relative_l2(&gam_fitted, mgcv_fitted);
    let gam_vs_lme4_blup_rel_l2 = relative_l2(&gam_intercept_dev, lme4_ranef);

    eprintln!(
        "RE + by-smooth TRUTH RECOVERY: n={n} groups={N_GROUPS} σ_ε={RESID_SD:.3} (σ_ε²={RESID_VAR:.5})\n  \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3}\n  \
         mean RMSE vs truth: gam={gam_mean_rmse:.5} mgcv={mgcv_mean_rmse:.5} (bar ≤ {:.5}; gam ≤ 1.10·mgcv?)\n  \
         intercept RMSE vs true offsets: gam={intercept_rmse:.5} (bar ≤ 0.10)\n  \
         resid-var err vs truth: gam={gam_var_err:.5} (σ̂²={gam_resid_var:.5}) lme4={lme4_var_err:.5} (σ̂²={lme4_sigma_e2:.5})\n  \
         [context only] gam-vs-mgcv mean rel-L2={gam_vs_mgcv_rel_l2:.4}  gam-vs-lme4-BLUP rel-L2={gam_vs_lme4_blup_rel_l2:.4}",
        0.5 * RESID_SD
    );

    // (1) MEAN RECOVERY. A fit cannot beat the noise it cannot observe; landing
    // well under σ_ε proves the structured signal was recovered rather than the
    // noise overfit. Bar: RMSE ≤ 0.5·σ_ε.
    assert!(
        gam_mean_rmse <= 0.5 * RESID_SD,
        "gam mean fit does not recover the true μ: RMSE={gam_mean_rmse:.5} > {:.5} (0.5·σ_ε)",
        0.5 * RESID_SD
    );
    // ... and gam must be at least as accurate as mgcv on the same truth.
    assert!(
        gam_mean_rmse <= 1.10 * mgcv_mean_rmse,
        "gam mean fit is less accurate than the mgcv baseline vs truth: gam={gam_mean_rmse:.5} > 1.10·{mgcv_mean_rmse:.5}"
    );

    // (2) INTERCEPT RECOVERY. The per-group offset gam attributes to group(g) must
    // track the TRUE intercept_g. Misattributing the offset to the smooth's null
    // space inflates this error. Bar: RMSE ≤ 0.10 (≤ 25% of the intercept SD 0.4).
    assert!(
        intercept_rmse <= 0.10,
        "gam per-group intercepts do not recover the true offsets: RMSE={intercept_rmse:.5} > 0.10"
    );

    // (3) NOISE-SCALE RECOVERY. The residual variance is the best-determined
    // component (≈295 residual d.f.); a double-penalization bug (smooth + RE both
    // soaking signal) inflates it, overfitting deflates it. Bar: within 20% of the
    // TRUE σ_ε², and at least as accurate as lme4 on the same truth.
    assert!(
        gam_var_err <= 0.20 * RESID_VAR,
        "gam residual variance does not recover true σ_ε²: |{gam_resid_var:.5} − {RESID_VAR:.5}|={gam_var_err:.5} > {:.5} (20%)",
        0.20 * RESID_VAR
    );
    assert!(
        gam_var_err <= 1.10 * lme4_var_err.max(1e-6),
        "gam residual variance is less accurate than the lme4 baseline vs truth: gam_err={gam_var_err:.5} > 1.10·{lme4_var_err:.5}"
    );
}
