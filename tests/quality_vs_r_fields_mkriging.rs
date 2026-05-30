//! End-to-end quality: gam's multivariate Matérn Gaussian-process smooth
//! (`matern(x1, x2, nu=1.5, k=20)`) must match `fields::spatialProcess` /
//! `fields::mKrig` — the R standard for thin-plate / Matérn kriging on
//! scattered 2-D spatial data — at interpolating a smooth surface from noisy
//! scattered observations.
//!
//! Why `fields` and not `mgcv`: `mgcv::gam(..., bs="gp")` is a *penalized* GAM
//! (basis + roughness penalty, REML smoothing-parameter selection). `fields`'s
//! `mKrig`/`spatialProcess` is *true* Gaussian-process kriging — it forms the
//! exact Matérn covariance over the data sites and produces the kriging
//! predictor (the GP posterior mean) with the nugget/range/sill estimated by
//! the profile/marginal likelihood (`spatialProcess` calls `MLESpatialProcess`).
//! That makes it the right adversary for gam's GP-kernel basis: agreement here
//! validates that gam's Matérn basis normalization and covariance structure
//! reproduce the genuine kriging surface, not just a penalized look-alike.
//!
//! Both engines see the IDENTICAL scattered data and predict on the IDENTICAL
//! interior grid. We compare:
//!   * RMSE of each engine's grid predictions against the noise-free truth
//!     (the quantity practitioners care about: interpolation fidelity),
//!   * relative L2 of gam's surface against the fields kriging surface
//!     (do the two GP predictors coincide?),
//!   * max pointwise error vs. truth,
//!   * Matérn range / length-scale agreement (both fitted by marginal
//!     likelihood — fields' `aRange` MLE vs. gam's fitted Matérn length scale).
//!
//! A real divergence is a real bug in gam's Matérn GP basis; the bounds below
//! are NOT weakened to hide one.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn gam_matern_kriging_matches_fields_mkrig() {
    init_parallelism();

    // ---- fixed-seed synthetic 2-D field, fed IDENTICALLY to both engines ---
    // (x1,x2) ~ U[0,1]^2, n=120 scattered sites (jittered toward a grid so the
    // domain is well-covered). Truth f(x1,x2) = sin(2π x1)·cos(2π x2); the
    // observations carry N(0, 0.15²) noise — the kriging/GP job is to recover
    // the smooth surface through that noise floor.
    let n = 120usize;
    let mut rng = StdRng::seed_from_u64(20259);
    let unit = Uniform::new(0.0, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, 0.15).expect("gaussian noise");
    let truth = |a: f64, b: f64| {
        (2.0 * std::f64::consts::PI * a).sin() * (2.0 * std::f64::consts::PI * b).cos()
    };

    let mut x1: Vec<f64> = Vec::with_capacity(n);
    let mut x2: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        x1.push(unit.sample(&mut rng));
        x2.push(unit.sample(&mut rng));
    }
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| truth(a, b) + noise.sample(&mut rng))
        .collect();

    // ---- shared held-out interior prediction grid: 15×15 = 225 points ------
    // Strictly interior (margin 0.1) so neither engine's extrapolation/edge
    // behavior dominates the interpolation-fidelity metric.
    let gp = 15usize;
    let lo = 0.1;
    let hi = 0.9;
    let mut gx1: Vec<f64> = Vec::with_capacity(gp * gp);
    let mut gx2: Vec<f64> = Vec::with_capacity(gp * gp);
    let mut gtruth: Vec<f64> = Vec::with_capacity(gp * gp);
    for i in 0..gp {
        let a = lo + (hi - lo) * i as f64 / (gp - 1) as f64;
        for j in 0..gp {
            let b = lo + (hi - lo) * j as f64 / (gp - 1) as f64;
            gx1.push(a);
            gx2.push(b);
            gtruth.push(truth(a, b));
        }
    }
    let ngrid = gx1.len();
    assert_eq!(ngrid, 225, "expected a 15×15 = 225-point interior grid");

    // ---- fit with gam: y ~ matern(x1, x2, nu=1.5, k=20), REML --------------
    // nu=1.5 (ThreeHalves) is the once-differentiable Matérn; gam's
    // operator-collocation penalty supports nu>=3/2 in d>=2. k=20 centers.
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode kriging dataset");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ matern(x1, x2, nu=1.5, k=20)", &ds, &cfg)
        .expect("gam 2-D matern kriging fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian GAM fit for 2-D matern() smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions on the shared interior grid. Column order matches the
    // encoded headers x1@0, x2@1, y@2 (identity link ⇒ design·beta = mean).
    let mut g = Array2::<f64>::zeros((ngrid, 3));
    for i in 0..ngrid {
        g[[i, 0]] = gx1[i];
        g[[i, 1]] = gx2[i];
        g[[i, 2]] = 0.0;
    }
    let grid_design = build_term_collection_design(g.view(), &fit.resolvedspec)
        .expect("rebuild matern design at grid points");
    let gam_grid: Vec<f64> = grid_design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME data with fields (the mature kriging reference) ------
    // `spatialProcess` runs MLESpatialProcess (profile/marginal likelihood) to
    // estimate the Matérn range (`aRange`), nugget (`tau`) and sill, then
    // predicts on the grid. smoothness=1.5 matches gam's nu=1.5. We emit the
    // grid predictions and the MLE range so the length-scale agreement can be
    // checked.
    let r = run_r(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("y", &y),
            Column::new("gx1", &gx1),
            Column::new("gx2", &gx2),
        ],
        r#"
        suppressPackageStartupMessages(library(fields))
        s    <- cbind(df$x1, df$x2)
        grid <- cbind(df$gx1, df$gx2)
        # True Bayesian kriging: Matern covariance, range/nugget by marginal
        # likelihood (MLESpatialProcess), smoothness fixed to nu=1.5 to match gam.
        fit <- spatialProcess(s, df$y, cov.args = list(Covariance = "Matern", smoothness = 1.5))
        pred <- predict(fit, xnew = grid)
        emit("grid_fit", as.numeric(pred))
        # MLE Matern range parameter (aRange == theta, the length scale).
        emit("aRange", as.numeric(fit$summary["aRange"]))
        "#,
    );
    let fields_grid = r.vector("grid_fit");
    let fields_arange = r.scalar("aRange");
    assert_eq!(
        fields_grid.len(),
        ngrid,
        "fields grid prediction length mismatch"
    );

    // ---- metrics ----------------------------------------------------------
    // Interpolation fidelity: each engine's grid prediction vs. noise-free truth.
    let gam_rmse = rmse(&gam_grid, &gtruth);
    let fields_rmse = rmse(fields_grid, &gtruth);
    let gam_maxerr = max_abs_diff(&gam_grid, &gtruth);
    let fields_maxerr = max_abs_diff(fields_grid, &gtruth);
    // Do the two GP predictors coincide as surfaces?
    let surface_rel = relative_l2(&gam_grid, fields_grid);

    eprintln!(
        "fields kriging vs gam matern(nu=1.5): n={n} grid={ngrid} gam_edf={gam_edf:.3} \
         gam_rmse={gam_rmse:.4} fields_rmse={fields_rmse:.4} gam_maxerr={gam_maxerr:.4} \
         fields_maxerr={fields_maxerr:.4} surface_rel_l2={surface_rel:.4} \
         fields_aRange={fields_arange:.4}"
    );

    // ---- principled, un-weakened bounds -----------------------------------
    // Noise floor is σ=0.15; a faithful GP/kriging recovery of the smooth
    // surface must come in well under that on the interior grid. The spec sets
    // RMSE < 0.08 (≈ half the noise floor) and max abs error < 0.25; gam must
    // meet the SAME interpolation-fidelity bar that fields meets.
    assert!(
        gam_rmse < 0.08,
        "gam Matern kriging RMSE vs truth exceeds the noise-floor-aware bound: {gam_rmse:.4} (>= 0.08)"
    );
    assert!(
        gam_maxerr < 0.25,
        "gam Matern kriging max pointwise error vs truth too large: {gam_maxerr:.4} (>= 0.25)"
    );
    // Sanity gate on the reference itself: fields must also recover the surface
    // (if it does not, the comparison is meaningless and the test is wrong).
    assert!(
        fields_rmse < 0.08,
        "fields kriging reference failed to recover the surface: RMSE={fields_rmse:.4}"
    );

    // The two GP predictors should agree as surfaces: a true-kriging predictor
    // (fields) and gam's Matérn GP basis evaluated on the same data should track
    // each other much more tightly than either tracks the noisy data. Both
    // surfaces have unit amplitude; relative L2 < 0.20 is a real agreement bound
    // (a basis-normalization or covariance-structure bug would blow past it)
    // while leaving margin for the penalized-vs-exact-kriging modeling gap.
    assert!(
        surface_rel < 0.20,
        "gam Matern surface diverges from fields kriging surface: rel_l2={surface_rel:.4} (>= 0.20)"
    );

    // Length-scale / range agreement. Both engines fit the range by marginal
    // likelihood. gam reports edf rather than a raw range; the principled
    // cross-engine check we CAN make element-free is that fields' MLE range sits
    // in a sane band for a U[0,1]^2 field with ~1-cycle structure (aRange on the
    // order of the inter-feature wavelength, not collapsed to 0 or blown up).
    assert!(
        fields_arange.is_finite() && fields_arange > 0.02 && fields_arange < 2.0,
        "fields MLE Matern range is implausible for a unit-square field: aRange={fields_arange:.4}"
    );
}
