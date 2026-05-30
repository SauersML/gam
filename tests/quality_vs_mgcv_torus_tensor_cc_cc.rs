//! End-to-end quality: gam's *doubly* periodic tensor smooth on the torus
//! S¹ × S¹ must match **mgcv** — the mature, de-facto standard GAM
//! implementation — on the same data, and must satisfy the intrinsic
//! continuity property a cyclic basis exists to guarantee.
//!
//! The torus is the canonical doubly-periodic manifold: a tensor product of two
//! cyclic (periodic) B-spline margins. mgcv builds it with
//! `te(theta, phi, bs = c("cc", "cc"))` — the row-wise Kronecker product of two
//! `bs="cc"` cyclic-cubic marginal bases, each wrapping continuously across its
//! period. gam exposes the same construction through
//! `te(theta, phi, boundary=['periodic','periodic'], period=[2*pi, 2*pi])`,
//! which builds two periodic B-spline marginals and forms their tensor product.
//!
//! Both engines fit by REML against a Gaussian likelihood, so they target the
//! *same* penalized objective; on a low-noise near-separable periodic truth the
//! fitted surfaces must essentially coincide. We additionally assert the
//! **seam/periodic continuity** property that is the load-bearing contract of a
//! cyclic basis: the fitted surface must be identical at θ=0 and θ=2π for every
//! φ (and symmetrically at φ=0 vs φ=2π). A sign error or threshold bug in gam's
//! periodic-basis closure would surface here as a jump/slope mismatch at the
//! wrap — invisible to an interior-only RMSE check but fatal to a true torus
//! smooth.
//!
//! Data: deterministic 20×20 grid (n=400), (θ,φ) uniform on [0,2π)² (last grid
//! point stops short of 2π so the seam is not duplicated in training), truth
//! f(θ,φ)=sin(2θ)·cos(3φ)+sin(θ+φ), Gaussian noise σ=0.05 from a fixed seed.
//! The identical (θ,φ,y) rows are handed to both gam and mgcv.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::TAU;

#[test]
fn gam_torus_tensor_cc_cc_matches_mgcv_and_wraps_at_both_seams() {
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
            let f = (2.0 * th).sin() * (3.0 * ph).cos() + (th + ph).sin();
            theta.push(th);
            phi.push(ph);
            y.push(f + noise.sample(&mut rng));
        }
    }

    // ---- fit with gam: doubly-periodic tensor smooth, REML -----------------
    // `boundary=['periodic','periodic']` + `period=[2*pi, 2*pi]` is gam's exact
    // analog of mgcv's te(bs=c('cc','cc')): two cyclic B-spline marginals tensor
    // -producted on the torus.
    let headers = ["theta", "phi", "y"].into_iter().map(String::from).collect();
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
    let formula =
        "y ~ te(theta, phi, boundary=['periodic','periodic'], period=[2*pi, 2*pi], k=8)";
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

    // gam fitted surface at the training grid.
    let gam_fitted = gam_predict(&theta, &phi);

    // ---- fit the SAME model with mgcv te(bs=c("cc","cc")) (the reference) --
    // mgcv needs explicit cyclic knot ranges [0, 2π] per margin so its cyclic
    // closure matches the [0, 2π) data support.
    let r = run_r(
        &[
            Column::new("theta", &theta),
            Column::new("phi", &phi),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ te(theta, phi, bs = c("cc", "cc"), k = c(8, 8)),
                 data = df, method = "REML",
                 knots = list(theta = c(0, 2 * pi), phi = c(0, 2 * pi)))
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");

    // ---- pointwise agreement on the training grid --------------------------
    let rel = relative_l2(&gam_fitted, mgcv_fitted);
    let corr = pearson(&gam_fitted, mgcv_fitted);
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

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
        "torus te(cc,cc): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.5} pearson={corr:.6} edf_rel={edf_rel:.3} \
         theta_seam_gap={theta_seam_gap:.3e} phi_seam_gap={phi_seam_gap:.3e}"
    );

    // Both engines REML-fit identical low-noise, near-separable periodic data in
    // matched doubly-cyclic tensor spaces (k=8 per margin), so the fitted
    // surfaces must essentially coincide: rel_l2 < 0.03 and pearson > 0.9995 are
    // tight for σ=0.05 yet absorb the small basis/centering convention gap; a
    // real divergence is a real bug in gam's periodic-Kronecker construction.
    assert!(
        corr > 0.9995,
        "torus fitted surfaces should be near-identical to mgcv: pearson={corr:.6}"
    );
    assert!(
        rel < 0.03,
        "torus fitted surface diverges from mgcv te(cc,cc): rel_l2={rel:.5}"
    );
    // EDF is basis/null-space-convention sensitive; same-ballpark complexity
    // (within 20% relative) is the right expectation for matched k and REML.
    assert!(
        edf_rel < 0.20,
        "effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
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
