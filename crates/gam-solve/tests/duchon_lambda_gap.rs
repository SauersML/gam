//! MEASUREMENT (diagnostic, not a pass/fail gate): trace gam's own Gaussian REML
//! score surface as a function of the Duchon smoothing parameter and compare its
//! minimizer to mgcv's REML-selected sp on the exact `duchon_sin8` fixture
//! (n=240, σ=0.10, sin(2π·8x), seed 11).
//!
//! For a single Gaussian smooth, gam's REML objective IS the closed-form
//! `gaussian_reml` score. We build the production Duchon basis + tension penalty,
//! find gam's REML-optimal rho (global grid), and print the whole score/edf/
//! amplitude curve. mgcv `bs="ds", m=c(2,0)` REML gives the reference sp/edf.
//! Fit at the training rows (amplitude attenuation shows there); truth-amplitude
//! peak-to-peak is 2.0.
//!
//! Mechanism read-off:
//!   - gam's REML min sits at high rho / low edf with attenuated amplitude while
//!     the score is genuinely LOWER there than at the rho matching mgcv's edf
//!     ⇒ gam's REML SCORE prefers over-smoothing (score-assembly, mechanism a).
//!   - gam's REML min recovers amplitude (edf ≈ mgcv edf) ⇒ the score is fine and
//!     any production over-smoothing is the outer OPTIMIZER (mechanism b).

use gam_solve::gaussian_reml::{gaussian_reml_closed_form, gaussian_reml_point_eval_at_rho};
use gam_terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability, build_duchon_basis,
};
use gam_test_support::reference::{Column, run_r};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn sin8_dataset(n: usize, sigma: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tp = 2.0 * std::f64::consts::PI * 8.0;
    let y: Vec<f64> = x.iter().map(|&t| (tp * t).sin() + noise.sample(&mut rng)).collect();
    (x, y)
}

fn duchon_spec(k: usize) -> DuchonBasisSpec {
    DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: k },
        periodic: None,
        length_scale: None,
        power: 0.5,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::default(),
    }
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(u, v)| (u - v).abs()).fold(0.0, f64::max)
}
fn amp(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
}

fn mgcv_ds(x: &[f64], y: &[f64], k: usize) -> (f64, f64, Vec<f64>) {
    let r = run_r(
        &[Column::new("x", x), Column::new("y", y)],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            m <- gam(y ~ s(x, bs = "ds", k = {k}, m = c(2, 0)), data = df, method = "REML")
            emit("edf", as.numeric(sum(m$edf)))
            emit("sp",  as.numeric(m$sp))
            emit("fit", as.numeric(fitted(m)))
            "#
        ),
    );
    (r.vector("edf")[0], r.vector("sp")[0], r.vector("fit").to_vec())
}

#[test]
fn zz_measure_duchon_sin8_reml_surface() {
    let n = 240usize;
    let (x, y) = sin8_dataset(n, 0.10, 11);
    let truth: Vec<f64> = x
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * 8.0 * t).sin())
        .collect();
    let y_arr = Array1::from(y.clone());

    for k in [50usize, 40] {
        let mut data = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            data[[i, 0]] = x[i];
        }
        let basis = build_duchon_basis(data.view(), &duchon_spec(k)).expect("duchon basis");
        let xd = basis
            .design
            .try_to_dense_arc("duchon-lambda-gap")
            .expect("dense design")
            .as_ref()
            .clone();
        let p = xd.ncols();
        let mut s = Array2::<f64>::zeros((p, p));
        for pen in &basis.penalties {
            // active penalties are full-size p×p here (single smooth term)
            if pen.nrows() == p && pen.ncols() == p {
                s += pen;
            }
        }
        let nulldim: usize = basis.nullspace_dims.iter().sum();

        // gam's REML global optimum on this basis
        let cf = gaussian_reml_closed_form(xd.view(), y_arr.view(), s.view(), None, None)
            .expect("closed form");
        let cf_fit = xd.dot(&cf.coefficients).to_vec();
        assert!(
            cf.rho.is_finite()
                && cf.lambda.is_finite()
                && cf.edf.is_finite()
                && cf.reml_score.is_finite(),
            "closed-form REML diagnostics must be finite"
        );

        // mgcv reference
        let (mgcv_edf, mgcv_sp, mgcv_fit) = mgcv_ds(&x, &y, k);
        assert!(
            mgcv_edf.is_finite() && mgcv_sp.is_finite(),
            "mgcv reference diagnostics must be finite"
        );

        eprintln!(
            "\n===== duchon sin8 k={k}: p={p} nulldim={nulldim} n_pen={} truth_amp_pp=2.0 =====",
            basis.penalties.len()
        );
        eprintln!(
            "  gam REML-opt : rho={:.3} lambda={:.4e} edf={:.3} score={:.5} \
             train_max_err={:.4} amp_pp={:.4}",
            cf.rho,
            cf.lambda,
            cf.edf,
            cf.reml_score,
            max_abs(&cf_fit, &truth),
            amp(&cf_fit)
        );
        eprintln!(
            "  mgcv ds      : sp={mgcv_sp:.4e} edf={mgcv_edf:.3} \
             train_max_err={:.4} amp_pp={:.4}",
            max_abs(&mgcv_fit, &truth),
            amp(&mgcv_fit)
        );

        // gam REML score surface: score/edf/amplitude across rho
        eprintln!("  gam REML score surface (rho -> score, edf, train_max_err, amp_pp):");
        let mut best_at_mgcv_edf: Option<(f64, f64, f64)> = None; // (rho,score,edf) closest to mgcv_edf
        for i in 0..=28 {
            let rho = -6.0 + (24.0) * (i as f64) / 28.0;
            let pe =
                gaussian_reml_point_eval_at_rho(xd.view(), y_arr.view(), s.view(), None, None, rho)
                    .expect("point eval");
            let fit = xd.dot(&pe.coefficients).to_vec();
            eprintln!(
                "    rho={rho:6.2} score={:.5} edf={:7.3} max_err={:.4} amp_pp={:.4}",
                pe.reml_score,
                pe.edf,
                max_abs(&fit, &truth),
                amp(&fit)
            );
            let d = (pe.edf - mgcv_edf).abs();
            if best_at_mgcv_edf.map(|(_, _, e)| (e - mgcv_edf).abs() > d).unwrap_or(true) {
                best_at_mgcv_edf = Some((rho, pe.reml_score, pe.edf));
            }
        }
        if let Some((rho_m, score_m, edf_m)) = best_at_mgcv_edf {
            eprintln!(
                "  SCORE-AT-BOTH: gam@gam-opt score={:.5} (edf={:.3}) vs gam@mgcv-edf score={:.5} \
                 (rho={:.2}, edf={:.3}). If gam-opt score < mgcv-edf score => gam's REML score \
                 prefers over-smoothing (mechanism a).",
                cf.reml_score, cf.edf, score_m, rho_m, edf_m
            );
        }
    }
}
