//! End-to-end quality: gam's *doubly* periodic tensor smooth on the torus
//! S¹ × S¹ must RECOVER THE KNOWN TRUTH and satisfy the intrinsic continuity
//! property a cyclic basis exists to guarantee.
//!
//! OBJECTIVE METRIC ASSERTED (truth recovery, not tool mimicry):
//!   1. RMSE(gam_fitted, true_surface) on a dense INTERPOLATION grid (distinct
//!      from the training grid) must be <= the noise level σ. The data is
//!      generated from the exact analytic surface f(θ,φ), so the noise floor is
//!      the principled bar: a smoother that recovers the truth cannot do better
//!      than σ on noisy samples, and recovering it to within σ on held-out
//!      interpolation points proves the periodic-Kronecker fit is correct, not
//!      merely "the same as a peer tool".
//!   2. ACCURACY MATCH-OR-BEAT: gam's RMSE-to-truth <= mgcv's RMSE-to-truth ×
//!      1.10. mgcv (`te(bs=c("cc","cc"))`) is the mature de-facto-standard torus
//!      smoother; it is the BASELINE gam must equal or beat ON ACCURACY-TO-TRUTH,
//!      never a fitted-output target gam must reproduce. (Matching a peer tool's
//!      noisy fit proves nothing — both could overfit alike.)
//!   3. SEAM / PERIODIC CONTINUITY (structural correctness property): the fitted
//!      surface must be identical at θ=0 and θ=2π for every φ (and symmetrically
//!      at φ=0 vs φ=2π), exact up to float error. This is the defining contract
//!      of a cyclic basis; a sign/threshold bug in gam's periodic-basis closure
//!      surfaces here as a wrap-discontinuity invisible to an interior RMSE check.
//!
//! mgcv's fitted surface is still computed and its rel_l2-to-gam printed for
//! context, but "close to mgcv" is NOT a pass criterion anywhere.
//!
//! The torus is the canonical doubly-periodic manifold: a tensor product of two
//! cyclic (periodic) B-spline margins. mgcv builds it with
//! `te(theta, phi, bs = c("cc", "cc"))` — the row-wise Kronecker product of two
//! `bs="cc"` cyclic-cubic marginal bases, each wrapping continuously across its
//! period. gam exposes the same construction through
//! `te(theta, phi, boundary=['periodic','periodic'], period=[2*pi, 2*pi])`,
//! which builds two periodic B-spline marginals and forms their tensor product.
//!
//! Data: deterministic 20×20 grid (n=400), (θ,φ) uniform on [0,2π)² (last grid
//! point stops short of 2π so the seam is not duplicated in training), truth
//! f(θ,φ)=sin(2θ)·cos(3φ)+sin(θ+φ), Gaussian noise σ=0.05 from a fixed seed.
//! The identical (θ,φ,y) rows are handed to both gam and mgcv. Accuracy-to-truth
//! is evaluated on a 13×13 INTERPOLATION grid offset off the training nodes.

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
use rand_distr::{Distribution, Normal};
use std::f64::consts::TAU;

/// The exact analytic surface the data is sampled from. Truth recovery is
/// measured against this, not against any fitted tool output.
fn truth(theta: f64, phi: f64) -> f64 {
    (2.0 * theta).sin() * (3.0 * phi).cos() + (theta + phi).sin()
}

#[test]
fn gam_torus_tensor_cc_cc_recovers_truth_and_wraps_at_both_seams() {
    init_parallelism();

    // ---- deterministic near-separable periodic truth on a 20x20 grid -------
    // f(θ,φ) = sin(2θ)·cos(3φ) + sin(θ+φ) over [0,2π)². The grid stops short of
    // 2π in each margin so the seam is never duplicated in training. Gaussian
    // noise σ=0.05 from a fixed seed makes the rows reproducible and identical
    // across both engines.
    const G: usize = 20;
    let n = G * G;
    let sigma = 0.05_f64;
    let mut rng = StdRng::seed_from_u64(20240529);
    let noise = Normal::new(0.0, sigma).expect("normal");

    let mut theta: Vec<f64> = Vec::with_capacity(n);
    let mut phi: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for i in 0..G {
        let th = TAU * (i as f64) / (G as f64);
        for j in 0..G {
            let ph = TAU * (j as f64) / (G as f64);
            let f = truth(th, ph);
            theta.push(th);
            phi.push(ph);
            y.push(f + noise.sample(&mut rng));
        }
    }

    // ---- fit with gam: doubly-periodic tensor smooth, REML -----------------
    // `boundary=['periodic','periodic']` + `period=[2*pi, 2*pi]` is gam's exact
    // analog of mgcv's te(bs=c('cc','cc')): two cyclic B-spline marginals tensor
    // -producted on the torus.
    let headers = ["theta", "phi", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|r| {
            StringRecord::from(vec![
                theta[r].to_string(),
                phi[r].to_string(),
                y[r].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode torus dataset");
    let col = ds.column_map();
    let theta_idx = col["theta"];
    let phi_idx = col["phi"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = "y ~ te(theta, phi, boundary=['periodic','periodic'], period=[2*pi, 2*pi], k=8)";
    let result = fit_from_formula(formula, &ds, &cfg).expect("gam torus tensor fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the torus tensor smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Helper: evaluate gam's fitted surface at arbitrary (θ,φ) rows by rebuilding
    // the design from the frozen spec (identity link => design·beta = mean).
    let gam_predict = |ths: &[f64], phs: &[f64]| -> Vec<f64> {
        assert_eq!(ths.len(), phs.len());
        let m = ths.len();
        let mut pts = Array2::<f64>::zeros((m, ds.headers.len()));
        for r in 0..m {
            pts[[r, theta_idx]] = ths[r];
            pts[[r, phi_idx]] = phs[r];
        }
        let d = build_term_collection_design(pts.view(), &fit.resolvedspec)
            .expect("rebuild torus design");
        d.design.apply(&fit.fit.beta).to_vec()
    };

    // ---- dense INTERPOLATION grid offset off the training nodes ------------
    // Truth recovery is judged on points the model did NOT see: a 13×13 grid
    // whose nodes fall strictly between the 20×20 training nodes. Evaluating
    // accuracy here (rather than on the training fit) measures genuine function
    // recovery, not interpolation of the noise at the data points.
    const GI: usize = 13;
    let ni = GI * GI;
    let mut grid_theta: Vec<f64> = Vec::with_capacity(ni);
    let mut grid_phi: Vec<f64> = Vec::with_capacity(ni);
    let mut grid_truth: Vec<f64> = Vec::with_capacity(ni);
    for i in 0..GI {
        // offset by half a cell so we never land on a training node
        let th = TAU * (i as f64 + 0.5) / (GI as f64);
        for j in 0..GI {
            let ph = TAU * (j as f64 + 0.5) / (GI as f64);
            grid_theta.push(th);
            grid_phi.push(ph);
            grid_truth.push(truth(th, ph));
        }
    }

    // gam's recovered surface on the held-out interpolation grid.
    let gam_grid = gam_predict(&grid_theta, &grid_phi);

    // ---- fit the SAME model with mgcv te(bs=c("cc","cc")) (baseline) -------
    // mgcv needs explicit cyclic knot ranges [0, 2π] per margin so its cyclic
    // closure matches the [0, 2π) data support. mgcv is the mature baseline gam
    // must match-or-beat ON ACCURACY-TO-TRUTH; its predictions on the SAME
    // interpolation grid (appended as rows GI*GI..) are scored against `truth`,
    // never used as a target gam must reproduce.
    let mut theta_all = theta.clone();
    theta_all.extend_from_slice(&grid_theta);
    let mut phi_all = phi.clone();
    phi_all.extend_from_slice(&grid_phi);
    let mut y_all = y.clone();
    y_all.extend(std::iter::repeat_n(0.0, ni)); // placeholders; rows fit only on first n via weights
    let mut wts: Vec<f64> = std::iter::repeat_n(1.0, n).collect();
    wts.extend(std::iter::repeat_n(0.0, ni));

    let r = run_r(
        &[
            Column::new("theta", &theta_all),
            Column::new("phi", &phi_all),
            Column::new("y", &y_all),
            Column::new("w", &wts),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- df[df$w > 0, ]
        m <- gam(y ~ te(theta, phi, bs = c("cc", "cc"), k = c(8, 8)),
                 data = train, method = "REML",
                 knots = list(theta = c(0, 2 * pi), phi = c(0, 2 * pi)))
        grid <- df[df$w == 0, ]
        emit("grid_pred", as.numeric(predict(m, newdata = grid)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_grid = r.vector("grid_pred");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_grid.len(),
        ni,
        "mgcv interpolation-grid prediction length mismatch"
    );

    // ---- accuracy-to-truth on the held-out interpolation grid --------------
    let gam_rmse = rmse(&gam_grid, &grid_truth);
    let mgcv_rmse = rmse(mgcv_grid, &grid_truth);
    // For context only: how close the two fitted surfaces are (NOT a pass gate).
    let rel_gam_vs_mgcv = relative_l2(&gam_grid, mgcv_grid);

    // ---- intrinsic seam/periodic continuity (the load-bearing property) ----
    // Evaluate at a dense set of φ values, comparing θ=0 vs θ=2π (the θ-seam),
    // then symmetrically θ values comparing φ=0 vs φ=2π (the φ-seam). A genuine
    // doubly-periodic basis has identical design rows — hence identical fitted
    // values — at coordinates separated by exactly one period in either margin.
    let seam_grid: Vec<f64> = (0..40).map(|k| TAU * (k as f64) / 40.0).collect();
    let zeros: Vec<f64> = std::iter::repeat_n(0.0, seam_grid.len()).collect();
    let taus: Vec<f64> = std::iter::repeat_n(TAU, seam_grid.len()).collect();

    // θ-seam: f(0, φ) vs f(2π, φ).
    let theta_seam_0 = gam_predict(&zeros, &seam_grid);
    let theta_seam_tau = gam_predict(&taus, &seam_grid);
    let theta_seam_gap = theta_seam_0
        .iter()
        .zip(theta_seam_tau.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // φ-seam: f(θ, 0) vs f(θ, 2π).
    let phi_seam_0 = gam_predict(&seam_grid, &zeros);
    let phi_seam_tau = gam_predict(&seam_grid, &taus);
    let phi_seam_gap = phi_seam_0
        .iter()
        .zip(phi_seam_tau.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    eprintln!(
        "torus te(cc,cc): n={n} ni={ni} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         gam_rmse_to_truth={gam_rmse:.5} mgcv_rmse_to_truth={mgcv_rmse:.5} \
         (sigma={sigma:.3}) rel_l2_gam_vs_mgcv={rel_gam_vs_mgcv:.5} \
         theta_seam_gap={theta_seam_gap:.3e} phi_seam_gap={phi_seam_gap:.3e}"
    );

    // (1) TRUTH RECOVERY — the primary objective claim. The surface is sampled
    // from the exact analytic f(θ,φ) with noise σ; a correct doubly-periodic
    // tensor smooth recovers it on the held-out interpolation grid to within the
    // noise level. RMSE-to-truth <= σ is the principled bar: you cannot beat the
    // noise floor, and reaching it proves the periodic-Kronecker fit is right.
    assert!(
        gam_rmse <= sigma,
        "gam did not recover the torus truth: RMSE-to-truth={gam_rmse:.5} > σ={sigma:.3} \
         on the held-out interpolation grid"
    );
    // (2) MATCH-OR-BEAT mgcv ON ACCURACY-TO-TRUTH (not on fitted output). gam's
    // interpolation error must be no worse than the mature standard's by more
    // than 10%. This demotes mgcv to an accuracy baseline, never a target.
    assert!(
        gam_rmse <= mgcv_rmse * 1.10,
        "gam's accuracy-to-truth lags the mgcv baseline: gam_rmse={gam_rmse:.5} \
         mgcv_rmse={mgcv_rmse:.5} (allowed gam <= mgcv*1.10)"
    );
    // EDF sanity (NOT matched to mgcv): the recovered surface must be genuinely
    // wiggly (truth has 2θ/3φ harmonics) yet far below the k=8×8 saturation cap.
    assert!(
        gam_edf > 4.0 && gam_edf < 60.0,
        "gam edf outside a signal-appropriate range for this torus truth: {gam_edf:.3}"
    );
    // The defining contract of a cyclic basis: value continuity across the wrap,
    // exact up to float error. Both torus seams must close to < 1e-6; any larger
    // gap is a sign/threshold bug in gam's periodic-basis closure.
    assert!(
        theta_seam_gap < 1e-6,
        "θ-seam not closed: max |f(0,φ) - f(2π,φ)| = {theta_seam_gap:.3e}"
    );
    assert!(
        phi_seam_gap < 1e-6,
        "φ-seam not closed: max |f(θ,0) - f(θ,2π)| = {phi_seam_gap:.3e}"
    );
}
