//! End-to-end quality: gam's CONVEX shape-constrained 1-D smooth
//! `s(x, shape=convex)` must (a) RECOVER a known convex truth to better than the
//! observation noise, (b) produce a fitted curve that is ITSELF provably convex
//! (the load-bearing constraint property), and (c) match-or-beat the mature
//! reference `scam`'s convex P-spline (`bs="cx"`) on truth-recovery RMSE — not
//! merely reproduce scam's fitted output.
//!
//! Why this combination earns its own test. A shape constraint is only worth
//! anything if it is ENFORCED: an unconstrained smoother can recover a convex
//! function on average yet still wiggle into locally-concave segments where the
//! data is noisy. gam's `shape=convex` path (the SCOP-spline-style monotone-2nd-
//! difference construction in `src/terms/smooth/shape_constraints.rs`, which maps
//! Convex => penalty order 2) must guarantee `f''>=0` everywhere on its own
//! fitted curve. The canonical mature peer for shape-constrained additive models
//! is R's `scam` package; its convex P-spline `scam(y ~ s(x, bs="cx"))` (penalised
//! REML/UBRE with monotone B-spline-coefficient constraints) is the match-or-beat
//! accuracy baseline on the SAME truth-recovery objective, never an output to
//! imitate.
//!
//! Data (seed=20260621, n=300): x ~ U(-1,1); truth is a genuinely convex,
//! nonlinear curve `f(x) = 0.5*x^2 - 0.3*x` (f''=1>0 everywhere, not just
//! affine); y = f(x) + N(0, sigma) with sigma = 0.20 * signal_range, putting the
//! noise well above any curvature feature so an unconstrained fit would be
//! tempted to bend the wrong way.
//!
//! Asserts:
//!   1. PRIMARY truth recovery: RMSE(gam fitted, truth) < sigma — gam recovers the
//!      convex mean to better than a single observation's noise.
//!   2. CONVEXITY of gam's OWN fitted curve on a dense sorted grid: every second
//!      finite difference f[i+1]-2f[i]+f[i-1] >= -epsilon, with epsilon a tiny
//!      fraction of the true grid curvature. This is gam's own constraint, scored
//!      independently of any reference.
//!   3. MATCH-OR-BEAT: RMSE(gam, truth) <= RMSE(scam, truth) * 1.15.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N: usize = 300;

#[test]
fn gam_convex_smooth_recovers_truth_is_convex_and_matches_scam() {
    init_parallelism();

    // ---- synthetic convex truth on [-1,1] ---------------------------------
    // f(x) = 0.5*x^2 - 0.3*x : f''(x) = 1 > 0 everywhere (genuinely convex and
    // nonlinear, not a degenerate affine line). A fixed seed feeds the SAME
    // (x, y) rows to gam and scam.
    let truth = |x: f64| 0.5 * x * x - 0.3 * x;

    let mut rng = StdRng::seed_from_u64(20260621);
    let u = Uniform::new(-1.0_f64, 1.0).expect("uniform [-1,1]");

    // Signal range over the support sets the noise scale.
    let f_lo = truth(0.3); // vertex region minimum (~ -0.045)
    let f_hi = truth(-1.0).max(truth(1.0)); // boundary maxima (0.8)
    let signal_range = f_hi - f_lo;
    let sigma = 0.20 * signal_range;
    let noise = Normal::new(0.0, sigma).expect("normal noise");

    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut truth_at_x = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = u.sample(&mut rng);
        let fi = truth(xi);
        x.push(xi);
        y.push(fi + noise.sample(&mut rng));
        truth_at_x.push(fi);
    }

    // ---- fit with gam: y ~ s(x, shape=convex), Gaussian / identity --------
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| csv::StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode convex dataset");
    let x_idx = ds.column_map()["x"];
    let width = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, k=12, shape=convex)", &ds, &cfg).expect("gam convex smooth fit");
    let FitResult::Standard(fit) = result else {
        panic!("Gaussian convex smooth is a scalar family => expected FitResult::Standard");
    };

    // gam fitted mean at the training points (Gaussian identity => eta = mu).
    let mut train_grid = Array2::<f64>::zeros((N, width));
    for i in 0..N {
        train_grid[[i, x_idx]] = x[i];
    }
    let train_design = build_term_collection_design(train_grid.view(), &fit.resolvedspec)
        .expect("rebuild convex design at training points");
    let gam_fitted: Vec<f64> = train_design.design.apply(&fit.fit.beta).to_vec();

    // Dense SORTED grid for the convexity property of gam's own curve.
    const M: usize = 200;
    let dense: Vec<f64> = (0..M)
        .map(|i| -1.0 + 2.0 * i as f64 / (M as f64 - 1.0))
        .collect();
    let mut dense_grid = Array2::<f64>::zeros((M, width));
    for i in 0..M {
        dense_grid[[i, x_idx]] = dense[i];
    }
    let dense_design = build_term_collection_design(dense_grid.view(), &fit.resolvedspec)
        .expect("rebuild convex design on dense grid");
    let gam_dense: Vec<f64> = dense_design.design.apply(&fit.fit.beta).to_vec();

    // Second finite differences on the (uniform) dense grid. f''>=0 means each is
    // >= 0 up to a tiny tolerance. Scale epsilon to the TRUE grid curvature so the
    // bound is meaningful: with f''=1 and step h, the true second difference is
    // h^2; allow violations no larger than 1% of that.
    let h = 2.0 / (M as f64 - 1.0);
    let true_second_diff = h * h; // f'' = 1 => second difference of truth = h^2
    let epsilon = 0.01 * true_second_diff;
    let worst_violation = (1..M - 1)
        .map(|i| gam_dense[i + 1] - 2.0 * gam_dense[i] + gam_dense[i - 1])
        .fold(f64::INFINITY, f64::min); // most-negative second difference

    // ---- fit the SAME data with scam (the mature reference) ---------------
    // scam(y ~ s(x, bs = "cx")) is the convex P-spline; predict at training x.
    let r = run_r(
        &[Column::new("x", &x), Column::new("y", &y)],
        r#"
        suppressPackageStartupMessages(library(scam))
        m <- scam(y ~ s(x, k = 12, bs = "cx"), data = df, family = gaussian())
        emit("fitted", as.numeric(predict(m, newdata = df)))
        emit("edf", sum(m$edf))
        "#,
    );
    let scam_fitted = r.vector("fitted");
    let scam_edf = r.scalar("edf");
    assert_eq!(scam_fitted.len(), N, "scam fitted length mismatch");

    // ---- OBJECTIVE METRICS -------------------------------------------------
    let gam_err = rmse(&gam_fitted, &truth_at_x);
    let scam_err = rmse(scam_fitted, &truth_at_x);

    eprintln!(
        "[scam-convex] n={N} gam_rmse={gam_err:.5} scam_rmse={scam_err:.5} \
         worst_convexity_violation={worst_violation:.3e} \
         (sigma={sigma:.4} signal={signal_range:.4} scam_edf={scam_edf:.3} \
          ratio={:.3})",
        gam_err / scam_err.max(1e-12)
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_scam_convex_smooth",
            "rmse",
            gam_err,
            "scam",
            scam_err,
        )
        .line()
    );

    // PRIMARY: gam recovers the convex truth to better than the observation noise.
    assert!(
        gam_err < sigma,
        "gam should recover the convex truth below the noise floor: rmse={gam_err:.5} \
         (bar = sigma {sigma:.4})"
    );

    // CONSTRAINT: gam's OWN fitted curve is convex everywhere (second differences
    // nonnegative up to a tiny fraction of the true curvature). A broken convex
    // construction would let a noisy segment bend concave, producing a clearly
    // negative second difference well past -epsilon.
    assert!(
        worst_violation >= -epsilon,
        "gam convex smooth is not convex on its own grid: worst second difference \
         {worst_violation:.3e} < -epsilon {:.3e}",
        -epsilon
    );

    // MATCH-OR-BEAT: gam's truth-recovery RMSE is no worse than scam's by more
    // than 15%, holding scam as an accuracy baseline (never an output to copy).
    assert!(
        gam_err <= scam_err * 1.15,
        "gam's convex truth-recovery must match-or-beat scam: \
         rmse(gam)={gam_err:.5} vs scam*1.15={:.5}",
        scam_err * 1.15
    );
}
