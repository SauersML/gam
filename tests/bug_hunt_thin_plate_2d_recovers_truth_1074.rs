//! #1074 regression (reference-free): gam's isotropic 2-D thin-plate smooth
//! `s(x, z, bs="tp", k=10)` must RECOVER a known smooth surface after REML
//! denoising — its truth-recovery RMSE must fall BELOW the observation-noise
//! level, not sit several times above it.
//!
//! This pins the gam-side "primary claim" of the mgcv comparison test
//! `gam_thin_plate_2d_matches_mgcv_gaussian`
//! (`tests/quality/families/quality_vs_mgcv_tensor_tp_2d_gaussian.rs`) WITHOUT
//! needing R/mgcv: the under-recovery reported in #1074 (rmse_vs_truth ≈ 0.147
//! vs noise σ = 0.05, ~3×) is a property of the gam fit ALONE — a broken
//! kernel/penalty (over- or under-smoothing, wrong null space, inflated EDF)
//! leaves the fitted surface far from the true f, and that is measurable
//! against the generating function with no external baseline.
//!
//! Data are the EXACT replica of the quality test's deterministic grid:
//! f(x,z) = sin(πx)·cos(πz) on a 20×20 grid of [0,1]², plus a fixed-seed
//! splitmix64 → Box–Muller Gaussian noise stream (σ = 0.05). A correct
//! REML-penalized thin-plate smooth shrinks the noise away and lands near the
//! truth (RMSE well under σ); the #1074 defect did not.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;
use std::f64::consts::PI;

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

/// Deterministic standard-normal stream (splitmix64 → Box–Muller), identical to
/// the quality test's `GaussianStream`, so the noisy `y` is bit-for-bit the same.
struct GaussianStream {
    state: u64,
    spare: Option<f64>,
}

impl GaussianStream {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let u = ((z >> 11) as f64) / ((1u64 << 53) as f64);
        u.max(f64::MIN_POSITIVE)
    }

    fn next_standard_normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * PI * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

#[test]
fn thin_plate_2d_recovers_truth_under_noise_floor() {
    init_parallelism();

    let side = 20usize;
    let n = side * side;
    let axis: Vec<f64> = (0..side).map(|i| i as f64 / (side as f64 - 1.0)).collect();
    let noise_sigma = 0.05_f64;
    let mut rng = GaussianStream::new(0x5eed_2d_7f_a1c0_u64);
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for &xi in &axis {
        for &zj in &axis {
            x.push(xi);
            z.push(zj);
            let f = (PI * xi).sin() * (PI * zj).cos();
            truth.push(f);
            y.push(f + noise_sigma * rng.next_standard_normal());
        }
    }

    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", x[i]),
                format!("{:.17e}", z[i]),
                format!("{:.17e}", y[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode 2-D tp grid");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, z, bs=\"tp\", k=10)", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian 2-D thin-plate smooth");
    };

    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild thin-plate 2-D design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    let gam_rmse = rmse(&gam_fitted, &truth);
    let truth_max = truth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let truth_min = truth.iter().cloned().fold(f64::INFINITY, f64::min);
    let signal_range = truth_max - truth_min;

    eprintln!(
        "[#1074-tp2d ref-free] n={n} sigma={noise_sigma:.3} signal_range={signal_range:.3} \
         gam_rmse_vs_truth={gam_rmse:.5} edf_total={:.3} log_lambdas={:?}",
        fit.fit.edf_total().unwrap_or(f64::NAN),
        fit.fit
            .log_lambdas
            .iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>(),
    );

    // PRIMARY: after REML shrinkage the denoising error must sit below the noise
    // level. The #1074 defect left gam_rmse ≈ 0.147 (≈ 3× σ); a correct
    // thin-plate smooth comfortably clears σ = 0.05.
    assert!(
        gam_rmse <= noise_sigma,
        "gam 2-D thin-plate did not recover the true surface: \
         rmse_vs_truth={gam_rmse:.5} exceeds noise sigma={noise_sigma:.3}"
    );
    // ... and is a small fraction of the signal range.
    assert!(
        gam_rmse <= 0.03 * signal_range,
        "gam 2-D thin-plate recovery error is large relative to the signal: \
         rmse_vs_truth={gam_rmse:.5}, signal_range={signal_range:.3}"
    );
}
