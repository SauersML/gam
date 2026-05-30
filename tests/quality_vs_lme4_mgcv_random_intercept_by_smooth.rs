//! End-to-end quality: gam's COMBINATION of a shared smooth, a by-factor smooth,
//! and a random intercept — `y ~ s(x) + s(x, by=g) + group(g)` — benchmarked
//! against the two mature standards for the two estimands it must partition.
//!
//!   * `mgcv::gam(y ~ s(x) + s(x, by = as.factor(g)))` — the de-facto GAM
//!     reference — for the *group-specific smooth shape*. mgcv's `s(x)` is a
//!     shared main-effect curve and each `s(x, by=level)` adds a level-specific
//!     deviation, so the per-group fitted smooth is `s(x) + s(x, by=g)`. This is
//!     exactly the fixed-smooth structure gam expresses with `s(x) + s(x,by=g)`.
//!   * `lme4::lmer(y ~ ns(x) + (1 | g))` — the de-facto mixed-model reference —
//!     for the *random-intercept variance component* and the per-group conditional
//!     modes (BLUPs). gam's `group(g)` is a REML-penalized random intercept, the
//!     same estimand lme4 optimizes.
//!
//! The point of the combination is that gam must correctly PARTITION variance
//! between the fixed by-smooth and the random intercept without double-counting:
//! the per-group vertical offset must land in `group(g)` (matching lme4's BLUP),
//! while the per-group *curvature* must land in `s(x,by=g)` (matching mgcv's
//! smooth). A double-penalization or mis-attribution bug shows up as either the
//! group curves disagreeing with mgcv or the intercepts disagreeing with lme4.
//!
//! Identical data is handed to all three engines. We assert:
//!   1. Pearson r on each group's fitted smooth curve (gam vs mgcv `s(x)+s(x,by=g)`,
//!      with the random intercept removed by within-group centering on both sides).
//!   2. Pearson r on the per-group intercepts (gam `group(g)` effect vs lme4 BLUP).
//!   3. Relative RMSE on the residual variance σ_ε² (gam vs lme4).
//! Bounds (pearson > 0.97 curves, > 0.96 intercepts, rel-RMSE < 0.12 on σ_ε²) are
//! the spec's principled tolerances; a genuine divergence failing them is a real
//! bug in gam's RE×by-smooth partitioning, not something to loosen.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, run_r};
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
fn gam_random_intercept_by_smooth_matches_lme4_and_mgcv() {
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
    let mut g_code = Vec::<f64>::with_capacity(n); // numeric group index for R
    let mut grp_of_row = Vec::<usize>::with_capacity(n);
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for grp in 0..N_GROUPS {
        for _ in 0..PER_GROUP {
            let xi = ux.sample(&mut rng);
            let yi = base_smooth(xi)
                + 0.15 * slope_g[grp] * xi
                + intercept_g[grp]
                + noise.sample(&mut rng);
            x.push(xi);
            y.push(yi);
            g_code.push(grp as f64);
            grp_of_row.push(grp);
            rows.push(StringRecord::from(vec![
                format!("{xi}"),
                format!("g{grp}"),
                format!("{yi}"),
            ]));
        }
    }

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

    // ---- (1) per-group smooth SHAPE: gam vs mgcv, intercept removed --------
    // Split fitted values by group and within-group center both engines. Centering
    // subtracts the per-group mean, killing the vertical offset (gam's random
    // intercept; mgcv's group-level constant absorbed into the smooth), so what
    // remains is the smooth shape s(x)+s(x,by=g) is responsible for. We correlate
    // the two centered curves per group element-wise on the SAME rows: both engines
    // were handed the identical row order, and within a group we collect rows by the
    // shared grp_of_row index in that order, so gam_c[k] and mgcv_c[k] are the same
    // observation (same x) on both sides — no sorting or re-gridding is needed.
    let mut corr_per_group = Vec::<f64>::with_capacity(N_GROUPS);
    for grp in 0..N_GROUPS {
        let mut gam_c = Vec::<f64>::new();
        let mut mgcv_c = Vec::<f64>::new();
        for i in 0..n {
            if grp_of_row[i] == grp {
                gam_c.push(gam_fitted[i]);
                mgcv_c.push(mgcv_fitted[i]);
            }
        }
        let gam_m = gam_c.iter().sum::<f64>() / gam_c.len() as f64;
        let mgcv_m = mgcv_c.iter().sum::<f64>() / mgcv_c.len() as f64;
        let gam_cc: Vec<f64> = gam_c.iter().map(|v| v - gam_m).collect();
        let mgcv_cc: Vec<f64> = mgcv_c.iter().map(|v| v - mgcv_m).collect();
        corr_per_group.push(pearson(&gam_cc, &mgcv_cc));
    }
    let min_curve_corr = corr_per_group.iter().copied().fold(f64::INFINITY, f64::min);

    // ---- (2) per-group intercepts: gam group effect vs lme4 BLUP -----------
    let intercept_corr = pearson(&gam_intercept_dev, lme4_ranef);

    // ---- (3) residual variance: relative RMSE (single component) -----------
    // With one scalar per engine, the relative RMSE collapses to the relative
    // absolute difference |σ²_gam − σ²_lme4| / σ²_lme4.
    let resid_rel_rmse = (gam_resid_var - lme4_sigma_e2).abs() / lme4_sigma_e2.abs().max(1e-12);

    eprintln!(
        "RE + by-smooth combo: n={n} groups={N_GROUPS} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3}\n  \
         per-group smooth pearson (gam vs mgcv, centered): {corr_per_group:?} min={min_curve_corr:.5}\n  \
         intercept pearson (gam group(g) vs lme4 BLUP): {intercept_corr:.5}\n  \
         residual var gam={gam_resid_var:.5} lme4={lme4_sigma_e2:.5} (truth {RESID_VAR:.5}) rel-RMSE={resid_rel_rmse:.4}"
    );

    // (1) Both engines REML-fit the same shared+by-factor smooth structure; with
    // the vertical offset removed, each group's smooth shape must coincide. The
    // by-smooth deviation here is mild (0.15*slope_g*x), so the curves are
    // dominated by the shared sin(3πx) plus a small per-group tilt — pearson > 0.97
    // on every group is the spec bound and catches any mis-attribution that would
    // distort the recovered shape.
    assert!(
        min_curve_corr > 0.97,
        "a group smooth shape diverges from mgcv: per-group pearson={corr_per_group:?}"
    );

    // (2) gam's group(g) random intercept must match lme4's conditional modes.
    // The 5 group offsets (intercept_g ~ N(0,0.4)) are well separated relative to
    // the per-group noise floor (σ_ε²/60 ≈ 7e-4), so both REML engines recover the
    // same BLUP ordering and magnitudes. pearson (not a tighter element-wise bound)
    // is the right metric here because gam's anchor vector carries the ~10% by-smooth
    // contamination quantified above plus a basis-dependent BLUP shrinkage scale that
    // need not equal lme4's, so we test agreement of the per-group ORDERING/SHAPE.
    // pearson > 0.96 is the spec bound; a lower value means gam soaked the offset
    // into the smooth instead of group(g).
    assert!(
        intercept_corr > 0.96,
        "gam per-group intercepts disagree with lme4 conditional modes: pearson={intercept_corr:.5}"
    );

    // (3) Residual variance is the best-determined component (≈295 residual d.f.).
    // Both engines fit it by REML on identical data, so it must match tightly; a
    // double-penalization bug (smooth + RE both soaking signal) would inflate gam's
    // σ_ε². rel-RMSE < 0.12 is the spec bound.
    assert!(
        resid_rel_rmse < 0.12,
        "residual variance disagrees with lme4: gam={gam_resid_var:.5} lme4={lme4_sigma_e2:.5} (rel-RMSE={resid_rel_rmse:.4})"
    );
}
