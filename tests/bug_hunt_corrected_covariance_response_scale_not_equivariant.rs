//! Regression: the smoothing-parameter-corrected Bayesian covariance `Vp`
//! (mgcv's `Vp = Vb + J·V_ρ·Jᵀ`, the matrix `predict(..., interval=...)` turns
//! into prediction standard errors) must be response-scale equivariant.
//!
//! For a Gaussian identity-link GAM, replacing `y` by `c·y` (`c > 0`) is an
//! exact rescaling of the model:
//!   - penalized LS estimate  β̂ → c·β̂,
//!   - REML-optimal λ          unchanged (the REML cost gains only a
//!                              ρ-independent (n/2)·log(c²) offset),
//!   - effective df (EDF)       unchanged,
//!   - dispersion              φ̂ → c²·φ̂.
//! So every second-moment object scales by exactly `c²`. The conditional
//! covariance `Vb = φ̂·H⁻¹` does, and `Vp` — documented to be "on the same
//! dispersion scale as `Vb`" — must scale by the same `c²`.
//!
//! The smoothing correction `J·V_ρ·Jᵀ` is built from `J = dβ̂/dρ` (linear in β̂,
//! so `J → c·J`) and the dispersion-free curvature `V_ρ`, hence it already
//! scales as `c²` — exactly like `Vb`. It must be added to `Vb` directly. The
//! bug (#582) multiplied it by the dispersion φ̂ (≈ c²) a second time, so the
//! correction block scaled as `c⁴`, silently inflating every `predict()`
//! interval for large-magnitude responses (~1500× too wide at c = 1000).
//!
//! This test fits the SAME deterministic dataset at response scales 1 and 1000
//! and asserts, in order:
//!   1. premise  — λ and EDF are equivariant,
//!   2. premise  — `Vb` diagonals scale by exactly `c²`,
//!   3. property — `Vp` diagonals scale by the same `c²` (was `c⁴`).

use gam::estimate::{FitOptions, fit_gam};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};

fn fit_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 120,
        tol: 1e-10,
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

/// Deterministic, RNG-free `y ~ s(x)` design with a genuine interior REML
/// optimum (spectrally rich signal + structured residual so RSS > 0). The
/// penalty rides the non-intercept spline columns so smoothing actually
/// engages and the smoothing correction `J·V_ρ·Jᵀ` is non-negligible.
fn design(scale: f64) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let n = 200usize;
    let p = 8usize;
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        let tau = std::f64::consts::TAU;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = t * t;
        x[[i, 3]] = (tau * t).sin();
        x[[i, 4]] = (tau * t).cos();
        x[[i, 5]] = (2.0 * tau * t).sin();
        x[[i, 6]] = (2.0 * tau * t).cos();
        x[[i, 7]] = (3.0 * tau * t).sin();
        // Mean signal plus a deterministic structured "noise" term so the
        // residual variance is strictly positive (interior REML optimum).
        let base = (tau * t).sin() + 0.5 * t + 0.3 * ((i as f64) * 2.399_963).sin();
        y[i] = scale * base;
    }
    // Identity ridge on every non-intercept column (the spline coefficients).
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    (x, y, s)
}

fn fit_at(scale: f64) -> gam::estimate::UnifiedFitResult {
    fit_design(design, scale)
}

fn fit_design(
    design_fn: fn(f64) -> (Array2<f64>, Array1<f64>, Array2<f64>),
    scale: f64,
) -> gam::estimate::UnifiedFitResult {
    let (x, y, s) = design_fn(scale);
    let w = Array1::<f64>::ones(x.nrows());
    let offset = Array1::<f64>::zeros(x.nrows());
    let penalty = BlockwisePenalty::new(0..x.ncols(), s);
    fit_gam(
        x,
        y.view(),
        w.view(),
        offset.view(),
        &[penalty],
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &fit_options(),
    )
    .expect("Gaussian identity fit should succeed")
}

#[test]
fn corrected_covariance_is_response_scale_equivariant() {
    let c = 1000.0_f64;
    let c2 = c * c;

    let fit1 = fit_at(1.0);
    let fitc = fit_at(c);

    // ── Premise 1: the FIT is equivariant — λ and EDF are unchanged. ───────
    let lam1 = fit1.lambdas[0];
    let lamc = fitc.lambdas[0];
    let lam_rel = (lam1 - lamc).abs() / lam1.abs().max(1e-300);
    assert!(
        lam_rel < 1e-6,
        "premise failed: λ is not response-scale invariant (λ@1={lam1:.10e}, \
         λ@{c}={lamc:.10e}, rel diff {lam_rel:.3e}); the model is not genuinely \
         equivariant and the Vp test below would be ill-posed"
    );
    let edf1 = fit1.edf_total().expect("edf at scale 1");
    let edfc = fitc.edf_total().expect("edf at scale c");
    let edf_rel = (edf1 - edfc).abs() / edf1.abs().max(1e-300);
    assert!(
        edf_rel < 1e-6,
        "premise failed: EDF is not response-scale invariant \
         (edf@1={edf1:.10e}, edf@{c}={edfc:.10e}, rel diff {edf_rel:.3e})"
    );

    // ── Premise 2: Vb scales by exactly c² to ~machine precision. ──────────
    let vb1 = fit1.beta_covariance().expect("Vb at scale 1").clone();
    let vbc = fitc.beta_covariance().expect("Vb at scale c").clone();
    assert_eq!(vb1.dim(), vbc.dim(), "Vb shape mismatch across scales");
    let p = vb1.nrows();
    for i in 0..p {
        let expected = c2 * vb1[[i, i]];
        let got = vbc[[i, i]];
        let rel = (got - expected).abs() / expected.abs().max(1e-300);
        assert!(
            rel < 1e-8,
            "premise failed: Vb[{i},{i}] is not c²-equivariant (got {got:.6e}, \
             expected {expected:.6e}, rel {rel:.3e}); the conditional covariance \
             must scale exactly as c² for the Vp contract to be testable"
        );
    }

    // ── Property under test: Vp must scale by the SAME c² as Vb. ───────────
    // Before the fix the smoothing-correction block scaled as c⁴, so the worst
    // diagonal came out ~1500× too large at c = 1000.
    let vp1 = fit1
        .beta_covariance_corrected()
        .expect("Vp at scale 1")
        .clone();
    let vpc = fitc
        .beta_covariance_corrected()
        .expect("Vp at scale c")
        .clone();
    assert_eq!(vp1.dim(), vpc.dim(), "Vp shape mismatch across scales");

    // The correction must be materially present (otherwise Vp == Vb and the
    // test degenerates to the premise). Require at least one diagonal where the
    // correction contributes a non-trivial fraction of Vp at scale 1.
    let mut max_corr_frac = 0.0_f64;
    for i in 0..p {
        let corr = (vp1[[i, i]] - vb1[[i, i]]).abs();
        let frac = corr / vp1[[i, i]].abs().max(1e-300);
        max_corr_frac = max_corr_frac.max(frac);
    }
    assert!(
        max_corr_frac > 1e-6,
        "smoothing correction is negligible (max fraction {max_corr_frac:.3e}); \
         the dataset does not exercise the J·V_ρ·Jᵀ path so the equivariance \
         test would be vacuous"
    );

    let mut worst_rel = 0.0_f64;
    let mut worst_idx = 0usize;
    for i in 0..p {
        let expected = c2 * vp1[[i, i]];
        let got = vpc[[i, i]];
        let rel = (got - expected).abs() / expected.abs().max(1e-300);
        if rel > worst_rel {
            worst_rel = rel;
            worst_idx = i;
        }
    }
    assert!(
        worst_rel < 1e-6,
        "Vp is not response-scale equivariant: Vp[{worst_idx},{worst_idx}] scales \
         by {factor:.3e}·c² instead of c² (Vp@1={vp1v:.6e}, Vp@{c}={vpcv:.6e}, \
         expected {expected:.6e}, rel error {worst_rel:.3e}). A `c⁴` here is the \
         smoothing correction being multiplied by the dispersion φ̂ twice (#582).",
        factor = vpc[[worst_idx, worst_idx]] / (c2 * vp1[[worst_idx, worst_idx]]).max(1e-300),
        vp1v = vp1[[worst_idx, worst_idx]],
        vpcv = vpc[[worst_idx, worst_idx]],
        expected = c2 * vp1[[worst_idx, worst_idx]],
    );
}
