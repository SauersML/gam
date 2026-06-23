//! End-to-end quality: gam's isotropic 2-D thin-plate smooth (`s(x, z, bs="tp")`)
//! must RECOVER a known smooth surface, and do so at least as accurately as
//! mgcv — the mature, standard GAM implementation and the *origin* of Wood's
//! (2003) low-rank thin-plate regression spline.
//!
//! OBJECTIVE METRIC (primary, pass/fail): truth recovery. The data are sampled
//! from the *known* surface f(x,z) = sin(πx)·cos(πz). We assert that gam's
//! fitted surface reconstructs that true surface with small absolute error —
//! RMSE(gam_fit, truth) bounded by a tiny fraction of the signal range. This is
//! an objective accuracy claim about gam, not "gam looks like mgcv".
//!
//! BASELINE TO MATCH-OR-BEAT (secondary): mgcv is fit on the *identical* data
//! and we compute its own truth-recovery RMSE. gam must be at least as accurate
//! as mgcv (within a 10% slack): RMSE(gam, truth) <= 1.10 · RMSE(mgcv, truth).
//! mgcv is therefore an accuracy baseline, never the ground truth — matching the
//! reference's fitted output is no longer a pass criterion.
//!
//! Data: deterministic 20×20 regular grid on [0,1]² of f(x,z)=sin(πx)·cos(πz) —
//! 400 points. A small deterministic (fixed-seed, reproducible) Gaussian noise
//! term is added so truth recovery is a genuine denoising test rather than pure
//! interpolation: a smoother that simply overfits the observations would *not*
//! recover the truth, whereas a correct REML-penalized thin-plate smooth shrinks
//! the noise away and lands near the true surface. Both engines see the SAME
//! noisy y.
//!
//! Both engines are pinned to the same basis dimension `k=10` (mgcv's default
//! 2-D thin-plate rank) so the comparison is on accuracy, not basis size.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

#[test]
fn gam_thin_plate_2d_matches_mgcv_gaussian() {
    init_parallelism();

    // ---- deterministic 20×20 grid on [0,1]² of f(x,z)=sin(πx)·cos(πz) ------
    // 400 points. We record the noise-free TRUE surface `truth[i]` (the known
    // generating function) and a noisy observation `y[i] = truth[i] + ε[i]`. The
    // noise is a fully deterministic, reproducible draw (a fixed-seed splitmix64
    // → Box–Muller stream) so the test is bit-for-bit repeatable while still
    // turning truth recovery into a real denoising problem: a smoother must
    // shrink ε away to land back on `truth`, which pure interpolation cannot do.
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

    // ---- fit with gam: isotropic 2-D thin-plate smooth, REML ---------------
    // `s(x, z, bs="tp")` routes the two-variable smooth through the thin-plate
    // (`tps`) radial kernel — the exact analogue of mgcv's `s(x, z, bs="tp")`.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, z, bs=\"tp\", k=10)", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian 2-D thin-plate smooth");
    };
    eprintln!(
        "[#1074-tp2d] edf_total={:.3} edf_by_block={:?} log_lambdas={:?} reml={:.4} converged={} iters={}",
        fit.fit.edf_total().unwrap_or(f64::NAN),
        fit.fit.edf_by_block().iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        fit.fit.log_lambdas.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        fit.fit.reml_score, fit.fit.outer_converged, fit.fit.outer_iterations,
    );

    // gam fitted values at the training grid: rebuild the design from the
    // frozen spec at the observed (x, z) (identity link => design*beta = mean).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild thin-plate 2-D design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv (the mature reference) --------------
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x, z, bs = "tp", k = 10), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");

    // ---- OBJECTIVE METRIC: truth recovery ---------------------------------
    // Compare each engine's fitted surface to the KNOWN true surface `truth`.
    // This measures real accuracy: how well the penalized thin-plate smooth
    // denoises the observations back onto the generating function.
    let gam_rmse = rmse(&gam_fitted, &truth);
    let mgcv_rmse = rmse(mgcv_fitted, &truth);

    // Signal range of the true surface, for a scale-aware absolute bar.
    let truth_max = truth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let truth_min = truth.iter().cloned().fold(f64::INFINITY, f64::min);
    let signal_range = truth_max - truth_min;

    // Context-only: how close gam's fit is to mgcv's fit (NOT a pass criterion).
    let rel_to_mgcv = relative_l2(&gam_fitted, mgcv_fitted);

    eprintln!(
        "tp-2d s(x,z,bs=tp): n={n} sigma={noise_sigma:.3} signal_range={signal_range:.3} \
         gam_rmse_vs_truth={gam_rmse:.5} mgcv_rmse_vs_truth={mgcv_rmse:.5} \
         rel_l2_gam_vs_mgcv={rel_to_mgcv:.5}"
    );

    // PRIMARY claim: gam recovers the truth. After REML shrinkage the fitted
    // surface should sit far closer to the true f than the noisy observations
    // do: its denoising error must be well below the noise level itself. We
    // require RMSE(gam, truth) <= noise_sigma (and, equivalently, a small
    // fraction — under ~3% — of the signal range), which a correct thin-plate
    // smooth comfortably achieves while a broken kernel/penalty (over- or
    // under-smoothing, wrong nullspace) would not.
    assert!(
        gam_rmse <= noise_sigma,
        "gam 2-D thin-plate did not recover the true surface: \
         rmse_vs_truth={gam_rmse:.5} exceeds noise sigma={noise_sigma:.3}"
    );
    assert!(
        gam_rmse <= 0.03 * signal_range,
        "gam 2-D thin-plate recovery error is large relative to the signal: \
         rmse_vs_truth={gam_rmse:.5}, signal_range={signal_range:.3}"
    );

    // SECONDARY claim: match-or-beat mgcv on ACCURACY. gam must recover the
    // truth at least as well as the mature reference, within a 10% slack.
    assert!(
        gam_rmse <= 1.10 * mgcv_rmse,
        "gam recovers the true surface less accurately than mgcv: \
         gam_rmse={gam_rmse:.5} > 1.10 * mgcv_rmse={mgcv_rmse:.5}"
    );
}

/// Deterministic standard-normal stream: a splitmix64 bit generator feeding the
/// Box–Muller transform. Fully reproducible from its seed, so the test's added
/// noise is identical on every run while still being a genuine random-looking
/// perturbation that turns truth recovery into a denoising problem.
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

    /// splitmix64 → uniform in (0, 1).
    fn next_uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        // Map the top 53 bits into [0, 1), then nudge off 0 for the log() below.
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
