//! #1031 acceptance arm (2), through the LIVE backend: `fit_pair_surface`
//! — THE first-class pair-component estimator the #975 ANOVA carve consumes —
//! must recover a known 2-D surface and match-or-beat `mgcv`'s tensor-product
//! `te()`/`ti()` on truth recovery.
//!
//! Why this is distinct from `quality_vs_mgcv_grid_spline_2d_truth_recovery`.
//! That sibling test drives the bare `fit_grid_spline_2d` engine on a perfect
//! lattice. THIS test drives the production entry `fit_pair_surface` on
//! SCATTERED (non-lattice) raw coordinates — the shape the carve actually sees
//! — exercising the full live backend: the axis-rescaling-invariant metric
//! `a_i = L_i²`, the cube-root `K` sizing rule, the exact-REML λ-selection with
//! the dense-ridge degeneracy fallback, and the carve-facing posterior via the
//! consumer's own `predict`. We additionally assert the live path resolved to
//! the exact grid backend (`PairSurfaceBackend::GridExact`), certifying the
//! grid engine is the one actually reached on a well-posed scattered pair — not
//! the dense fallback.
//!
//! Why `te()` is the right yardstick AND why match-or-beat is principled. The
//! grid engine penalizes the FULL anisotropic biharmonic energy
//!   `J(f) = ∫∫ a₁²·f_{x1x1}² + 2·a₁a₂·f_{x1x2}² + a₂²·f_{x2x2}²`,
//! INCLUDING the mixed `f_{x1x2}²` term that `te()`'s Kronecker-marginal
//! penalty drops. On a truth carrying a genuine cross-derivative component the
//! biharmonic penalty is better matched to the signal, so "match-or-beat mgcv
//! on truth recovery" is principled, not a tolerance artifact. We never assert
//! the two fitted surfaces are close to each other (different posteriors — the
//! whole reason the grid is its own estimator, not a `te()` back-end).
//!
//! DGP (self-constructed truth, #904). A non-separable surface with a real
//! mixed-derivative term,
//!   `f(x1,x2) = sin(2.4·x1)·cos(2.1·x2) + 0.8·x1·x2 + 0.5·sin(3.0·x1·x2)`,
//! over `[0,1]²` at SCATTERED low-discrepancy (golden-ratio / √2) abscissae
//! (n=2000), plus a fixed, RNG-free zero-mean noise stream so gam and mgcv see
//! IDENTICAL rows.
//!
//! OBJECTIVE METRIC. RMSE of each engine's fitted surface against the NOISELESS
//! truth `f` at the training rows. PRIMARY: the live backend recovers `f` to a
//! small fraction of the signal range. MATCH-OR-BEAT: its recovery RMSE is no
//! worse than mgcv's by more than 10%.

use gam::terms::anova_atom::{PairSurfaceBackend, fit_pair_surface};
use gam::test_support::reference::{Column, rmse, run_r};
use ndarray::Array2;

const N: usize = 2000;
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
fn fit_pair_surface_recovers_truth_and_matches_or_beats_mgcv_te() {
    // ---- deterministic SCATTERED truth over [0,1]² (n=2000) -----------------
    // Two coprime irrational phases give a quasi-uniform but non-lattice cloud
    // — the carve's real input shape, distinct from the sibling lattice test.
    let golden = 0.618_033_988_749_894_9_f64;
    let sqrt2_frac = std::f64::consts::SQRT_2.fract();
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut f0 = Vec::with_capacity(N);
    for i in 0..N {
        let a = ((i + 1) as f64 * golden).fract();
        let b = ((i + 1) as f64 * sqrt2_frac).fract();
        let t = truth(a, b);
        x1.push(a);
        x2.push(b);
        f0.push(t);
        y.push(t + noise(i));
    }

    // ---- fit through the LIVE backend: fit_pair_surface (one response dim) ---
    // Raw scattered coordinates in, carve-facing posterior out. The backend
    // auto-selects the metric (a_i = L_i²), K (cube-root rule), and λ (exact
    // REML), with the dense-ridge fallback only on degeneracy.
    let responses = Array2::from_shape_fn((N, 1), |(r, _)| y[r]);
    let fit = fit_pair_surface(&x1, &x2, responses.view()).expect("live pair surface fit");

    // The well-posed scattered pair must reach the EXACT grid engine, not the
    // dense fallback — this certifies the grid backend is the one actually wired
    // live, the whole point of #1031's live-consumption gate.
    assert_eq!(
        fit.backend,
        PairSurfaceBackend::GridExact,
        "a well-posed scattered pair must route through the exact grid engine, \
         not the dense-ridge degeneracy fallback"
    );

    let mut gam_fitted = Vec::with_capacity(N);
    for r in 0..N {
        let (mean, var) = fit.predict(0, x1[r], x2[r]).expect("pair surface predict");
        assert!(var > 0.0, "posterior variance must be positive");
        gam_fitted.push(mean);
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
    assert_eq!(mgcv_fitted.len(), N, "mgcv fitted length mismatch");

    // ---- OBJECTIVE METRIC: recovery error against the NOISELESS truth ------
    let gam_err = rmse(&gam_fitted, &f0);
    let mgcv_err = rmse(mgcv_fitted, &f0);
    let signal_range = {
        let lo = f0.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = f0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };

    eprintln!(
        "fit_pair_surface (live backend) vs mgcv te(): n={N} \
         gam_rmse_vs_truth={gam_err:.6} mgcv_rmse_vs_truth={mgcv_err:.6} \
         lambda={:.6} edf={:.3} mgcv_edf={mgcv_edf:.3} signal_range={signal_range:.4} \
         backend={:?}",
        fit.surface.lambda, fit.surface.edf, fit.backend
    );

    // PRIMARY: the live backend recovers the generating surface well below the
    // noise floor (NOISE_AMP = 0.15 ⇒ noise RMSE ≈ 0.0866). Recovery RMSE under
    // 6% of the signal range is comfortably below that and a real estimator bug
    // (dropped mixed term, wrong band/penalty assembly, wrong λ-selection)
    // cannot clear it. The scattered cloud's local density is a touch lower than
    // the lattice sibling's, hence 6% vs 5%.
    assert!(
        gam_err < 0.06 * signal_range,
        "live pair-surface backend failed to recover the truth: \
         rmse_vs_truth={gam_err:.6} (signal range {signal_range:.4}); \
         an assembly/penalty/λ bug, not a tolerance artifact"
    );

    // MATCH-OR-BEAT: the live backend's recovery error is no worse than mgcv's
    // by more than 10%. mgcv is demoted to a yardstick on accuracy against the
    // truth, never a fit to reproduce.
    assert!(
        gam_err <= mgcv_err * 1.10,
        "live pair-surface backend is less accurate than mgcv te() at recovering \
         the truth: gam_rmse={gam_err:.6} > 1.10 * mgcv_rmse={mgcv_err:.6}"
    );
}
