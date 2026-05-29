//! End-to-end quality: gam's 1-D Duchon spline (`duchon(x, k=20, order=2)`)
//! must match mgcv — the mature, standard GAM implementation — when both fit
//! the SAME low-noise synthetic data by REML.
//!
//! Reference: `mgcv::gam(y ~ s(x, bs="ds", k=20, m=c(2,0)), method="REML")`.
//! mgcv's `bs="ds"` is the canonical Duchon/thin-plate-with-Duchon-penalty
//! basis; `m=c(s,p)` selects the derivative order `s` of the penalty and the
//! extra kernel power `p`. gam's `duchon(x, k=20, order=2)` requests a Duchon
//! smooth with nullspace order 2 (`m=c(2,0)` in mgcv parlance) and 20 basis
//! functions, so the two engines target the same penalized objective on the
//! same grid.
//!
//! At 1-D and low noise (σ=0.05) the Duchon smoother is a flexible penalized
//! spline that should recover sin(8π·x) cleanly; mgcv is the ground-truth
//! reference for the fitted function and effective degrees of freedom. We
//! assert pointwise agreement of the fitted smooth on a dense test grid plus
//! same-ballpark complexity. A genuine divergence here is a real bug, not a
//! reason to loosen the bound.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

#[test]
fn gam_duchon_1d_matches_mgcv_ds() {
    init_parallelism();

    // ---- synthetic data: x in [0,1], y = sin(8π·x) + N(0, 0.05), n=200 -----
    // Fixed seed (123) so gam and mgcv see byte-identical data; sorted x keeps
    // the design well-conditioned and the test grid interpretable.
    let n = 200usize;
    let mut rng = StdRng::seed_from_u64(123);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let two_pi_f = 2.0 * std::f64::consts::PI * 8.0;
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (two_pi_f * t).sin() + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    // ---- fit with gam: y ~ duchon(x, k=20, order=2), REML ------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ duchon(x, k=20, order=2)", &ds, &cfg).expect("gam duchon fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian Duchon smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // ---- dense test grid interior to [0,1] (avoid extrapolation edges) -----
    let m = 201usize;
    let x_test: Vec<f64> = (0..m).map(|i| 0.005 + 0.99 * i as f64 / (m as f64 - 1.0)).collect();
    let y_truth: Vec<f64> = x_test.iter().map(|&t| (two_pi_f * t).sin()).collect();

    // gam fitted values at the test grid: rebuild the design from the frozen
    // spec (identity link => design*beta = mean).
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &t) in x_test.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild Duchon design at test grid");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv bs="ds" (the mature reference) -------
    // We pass both the training data and the test grid (as trailing rows with a
    // sentinel y) so mgcv predicts on exactly x_test. `m=c(2,0)` = derivative
    // order 2 / kernel power 0, the Duchon analogue of gam's order=2.
    let mut x_all = x.clone();
    x_all.extend_from_slice(&x_test);
    let mut y_all = y.clone();
    y_all.extend(std::iter::repeat_n(0.0, m));
    let mut is_train = vec![1.0; n];
    is_train.extend(std::iter::repeat_n(0.0, m));

    let r = run_r(
        &[
            Column::new("x", &x_all),
            Column::new("y", &y_all),
            Column::new("is_train", &is_train),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- df[df$is_train > 0.5, ]
        grid  <- df[df$is_train < 0.5, ]
        m <- gam(y ~ s(x, bs = "ds", k = 20, m = c(2, 0)), data = train, method = "REML")
        emit("fitted", as.numeric(predict(m, newdata = grid)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_fitted.len(), m, "mgcv prediction length mismatch");

    // ---- compare on the test grid -----------------------------------------
    let rel = relative_l2(&gam_fitted, mgcv_fitted);
    let max_abs = max_abs_diff(&gam_fitted, mgcv_fitted);
    // sanity: how well each engine recovers the truth (diagnostic only)
    let gam_truth_rel = relative_l2(&gam_fitted, &y_truth);
    let mgcv_truth_rel = relative_l2(mgcv_fitted, &y_truth);
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

    eprintln!(
        "duchon-vs-mgcv-1d: n={n} grid={m} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.4} max_abs={max_abs:.4} edf_rel={edf_rel:.4} \
         (truth-rel gam={gam_truth_rel:.4} mgcv={mgcv_truth_rel:.4})"
    );

    // Both engines REML-fit identical low-noise data with a Duchon penalty, so
    // their fitted smooths must essentially coincide. The truth peak is 1
    // (peak-to-peak 2); the bounds below are the spec's principled targets:
    //   - relative L2 < 0.05: the two fitted functions agree to within 5% in
    //     energy — tight for a sin8 of unit amplitude, yet leaves margin for
    //     differing center/null-space conventions between the bases.
    //   - max_abs_diff < 0.25: ~12.5% of peak-to-peak, catching any localized
    //     phase/amplitude divergence at the sin8 peaks.
    assert!(
        rel < 0.05,
        "Duchon fitted smooth diverges from mgcv bs=ds: rel_l2={rel:.4} (bound 0.05)"
    );
    assert!(
        max_abs < 0.25,
        "Duchon fitted smooth has a large pointwise gap vs mgcv: max_abs={max_abs:.4} (bound 0.25)"
    );
    // EDF is basis/null-space-convention sensitive; same-ballpark complexity
    // (within 15%) confirms the penalty selected matching smoothness.
    assert!(
        edf_rel < 0.15,
        "effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.4}, bound 0.15)"
    );
}
