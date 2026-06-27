//! #1031 acceptance arm (2): the streaming 2-D grid spline engine
//! (`solver::grid_spline_2d`) must RECOVER a known 2-D surface from noisy data
//! and do so at least as accurately as `mgcv`'s tensor-product smooth `te()` —
//! the mature, de-facto standard for anisotropic 2-D GAM smoothing.
//!
//! Why `te()` is the right yardstick AND why this is its own estimator. The
//! grid engine penalizes the FULL anisotropic biharmonic energy
//!   `J(f) = ∫∫ a₁²·f_{x1x1}² + 2·a₁a₂·f_{x1x2}² + a₂²·f_{x2x2}²`,
//! INCLUDING the mixed `f_{x1x2}²` term. `te(x1,x2)` penalizes a Kronecker SUM
//! of per-margin wiggliness — a *different* roughness functional that drops the
//! mixed cross-derivative coupling (which is precisely why #1031 exposes the
//! grid as its own pair-component estimator instead of silently re-routing
//! `te()` through it). On a truth with a genuine cross-derivative component the
//! grid engine's penalty is better matched to the signal, so "match-or-beat
//! mgcv on truth recovery" is principled, not a tolerance artifact.
//!
//! DGP (self-constructed truth, #904). A non-separable surface with a real
//! mixed-derivative term,
//!   `f(x1,x2) = sin(2.4·x1) · cos(2.1·x2) + 0.8·x1·x2 + 0.5·sin(3.0·x1·x2)`,
//! over `[0,1]²` on a deterministic 45×45 lattice (n=2025), plus a fixed,
//! RNG-free golden-ratio noise stream (so gam and mgcv see IDENTICAL rows). The
//! `0.5·sin(3·x1·x2)` term has a non-trivial `f_{x1x2}`, the channel the grid
//! penalty captures and the Kronecker-marginal `te` penalty does not.
//!
//! OBJECTIVE METRIC. RMSE of each engine's fitted surface against the NOISELESS
//! truth `f` at the training rows. PRIMARY: the grid engine recovers `f` to a
//! small fraction of the signal amplitude. MATCH-OR-BEAT: its recovery RMSE is
//! no worse than mgcv's by more than 10%. We never assert the two fitted
//! surfaces are close to each other.

use gam::terms::grid_spline_2d::fit_grid_spline_2d;
use gam::test_support::reference::{Column, rmse, run_r};

const G: usize = 45;
const NOISE_AMP: f64 = 0.15;

fn truth(x1: f64, x2: f64) -> f64 {
    (2.4 * x1).sin() * (2.1 * x2).cos() + 0.8 * x1 * x2 + 0.5 * (3.0 * x1 * x2).sin()
}

/// Deterministic, RNG-free zero-mean noise stream (low-discrepancy phase).
fn noise(i: usize) -> f64 {
    let golden = 0.618_033_988_749_894_9_f64;
    (((i + 1) as f64 * golden).fract() - 0.5) * 2.0 * NOISE_AMP
}

#[test]
fn grid_spline_2d_recovers_truth_and_matches_or_beats_mgcv_te() {
    // ---- deterministic non-separable truth on a 45×45 lattice over [0,1]² ----
    let n = G * G;
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut f0 = Vec::with_capacity(n);
    for i in 0..G {
        for j in 0..G {
            let a = i as f64 / (G as f64 - 1.0);
            let b = j as f64 / (G as f64 - 1.0);
            let t = truth(a, b);
            x1.push(a);
            x2.push(b);
            f0.push(t);
            y.push(t + noise(i * G + j));
        }
    }
    let w = vec![1.0_f64; n];

    // ---- fit with the streaming grid engine: exact-REML, K = ⌈n^(1/3)⌉ ----
    // The metric a_i = L_i² = 1 here (both axes span [0,1]); REML owns the
    // smoothness. K matches the pair-component auto-rule (cube-root growth,
    // capped at the engine's sizing contract).
    let k = (n as f64).cbrt().ceil() as usize;
    let k = k.clamp(4, 32);
    let fit = fit_grid_spline_2d(&x1, &x2, &y, &w, k, [1.0, 1.0]).expect("grid REML fit");
    let mut grid_fitted = Vec::with_capacity(n);
    for r in 0..n {
        let (mean, var) = fit.predict(0, x1[r], x2[r]).expect("grid predict");
        assert!(var > 0.0, "posterior variance must be positive");
        grid_fitted.push(mean);
    }

    // ---- fit the comparable model with mgcv: y ~ te(x1, x2), REML ----------
    let r = run_r(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ te(x1, x2), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");

    // ---- OBJECTIVE METRIC: recovery error against the NOISELESS truth ------
    let grid_err = rmse(&grid_fitted, &f0);
    let mgcv_err = rmse(mgcv_fitted, &f0);
    let signal_range = {
        let lo = f0.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = f0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };

    eprintln!(
        "grid_spline_2d vs mgcv te(): n={n} K={k} \
         grid_rmse_vs_truth={grid_err:.6} mgcv_rmse_vs_truth={mgcv_err:.6} \
         grid_loglambda={:.4} mgcv_edf={mgcv_edf:.3} signal_range={signal_range:.4}",
        fit.log_lambda
    );

    // PRIMARY: the grid engine recovers the generating surface well below the
    // noise floor (NOISE_AMP = 0.15 ⇒ noise RMSE ≈ 0.0866). Recovery RMSE under
    // 5% of the signal range is comfortably below that and a real estimator bug
    // (dropped mixed term, wrong band/penalty assembly) cannot clear it.
    assert!(
        grid_err < 0.05 * signal_range,
        "grid engine failed to recover the truth: rmse_vs_truth={grid_err:.6} \
         (signal range {signal_range:.4}); an assembly/penalty bug, not a tolerance artifact"
    );

    // MATCH-OR-BEAT: the grid engine's recovery error is no worse than mgcv's by
    // more than 10%. mgcv is demoted to a yardstick on accuracy against the
    // truth, never a fit to reproduce. On this cross-derivative-bearing truth
    // the full biharmonic penalty should in fact edge out te()'s Kronecker form.
    assert!(
        grid_err <= mgcv_err * 1.10,
        "grid engine is less accurate than mgcv te() at recovering the truth: \
         grid_rmse={grid_err:.6} > 1.10 * mgcv_rmse={mgcv_err:.6}"
    );
}
