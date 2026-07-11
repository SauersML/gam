//! Regression: a Gaussian REML fit must be INVARIANT to a global rescaling of
//! the prior weights by a positive constant `c`.
//!
//! `gamfit` treats `weights` as inverse-variance weights with a *profiled*
//! dispersion: `Var(yᵢ) = φ / wᵢ` with `φ` estimated, exactly as mgcv does.
//! Under that convention the absolute magnitude of the weights carries no
//! information — only their ratios do. Multiplying every weight by the same
//! `c > 0` is absorbed entirely by the profiled scale `φ̂ → c·φ̂`, so the
//! selected smoothing parameters, the effective degrees of freedom, and the
//! fitted coefficients (hence every prediction) must be unchanged. (The
//! library already honours this for the *uncertainty*: the conditional SEs are
//! invariant to weight rescaling, because `φ̂·(XᵀWX)⁻¹` is — `XᵀWX` scales by
//! `c`, `φ̂` scales by `c`, the product is fixed.)
//!
//! The same invariance is visible directly in the Gaussian REML objective the
//! engine minimises (`src/solver/reml/unified.rs:7248`, the
//! `DispersionHandling::ProfiledGaussian` arm):
//!
//!   V(ρ) = D_p/(2φ̂) + ½·log|H| − ½·log|S|₊ + ((n−M_p)/2)·log(2π φ̂),
//!   with  D_p = deviance + penalty,  φ̂ = D_p/(n − M_p).
//!
//! Send `W → c·W` and the invariance-preserving smoothing `λ → c·λ` (so the
//! penalised Hessian `H = XᵀWX + λS → c·H` and β̂ is fixed). Then:
//!   • `D_p → c·D_p`, `φ̂ → c·φ̂`  ⇒  `D_p/(2φ̂)` is unchanged;
//!   • `½(log|H| − log|S|₊) → ½(p − r)·log c = ½·M_p·log c`;
//!   • `((n−M_p)/2)·log(2π φ̂) → +((n−M_p)/2)·log c`.
//! The two `log c` pieces sum to `(n/2)·log c`, a constant **independent of ρ**.
//! A constant offset cannot move an argmin, so the minimiser obeys
//! `λ̂(c·W) = c·λ̂(W)` exactly and the fit (β̂, EDF, predictions) is invariant.
//!
//! Observed (this fixture, debug build): rescaling all weights from 1 to
//! `c = 1000` moves the selected λ by a factor ≈ 810 (not 1000), shifts the
//! effective dof by ≈ 1.6e-2, and moves the fitted coefficients by ≈ 4e-3 —
//! a genuine change in the fitted function from a no-op rescaling. The
//! `(n−M_p)/2·log(2π φ̂)` term grows like `(n/2)·log c` with the weight scale,
//! inflating the absolute objective value while leaving its shape fixed; the
//! outer optimiser then stops short of the (shifted) optimum, so the
//! invariance that holds in exact arithmetic is broken in the implementation.
//!
//! This test fits the SAME `y ~ s(x)` Gaussian model twice — once with all
//! weights `1`, once with all weights `c = 1000` — and asserts the selected
//! smoothing parameter scales exactly by `c` and that the EDF and the fitted
//! coefficients are invariant. It fails today and will pass once the Gaussian
//! REML smoothing selection is made weight-scale invariant.

use gam::estimate::{FitOptions, fit_gam};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};

fn fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 200,
        tol: 1e-11,
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

/// Deterministic, RNG-free `y ~ s(x)` design: an intercept plus a ridge-
/// penalised Fourier/polynomial basis on `t ∈ [0, 1]`. The intercept (column 0)
/// is the only unpenalised column (`nullspace_dim = 1`); the rest carry a unit
/// ridge penalty, so REML has a single non-trivial smoothing parameter to pick.
fn base_design() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let n = 400usize;
    let p = 10usize;
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
        x[[i, 8]] = (3.0 * tau * t).cos();
        x[[i, 9]] = (4.0 * tau * t).sin();
        // Smooth signal + a deterministic, reproducible wiggle so REML lands at
        // an interior λ (neither fully smoothed nor interpolating).
        y[i] = (tau * t).sin() + 0.5 * t + 0.3 * ((i as f64) * 2.399_963).sin();
    }
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    (x, y, s)
}

fn spec() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

#[test]
fn gaussian_reml_fit_is_invariant_to_global_weight_rescaling() {
    let (x, y, s) = base_design();
    let n = x.nrows();
    let p = x.ncols();
    let offset = Array1::<f64>::zeros(n);
    let penalty = || BlockwisePenalty::new(0..p, s.clone());

    // ── Fit A: every weight 1. ─────────────────────────────────────────────
    let fit1 = fit_gam(
        x.clone(),
        y.view(),
        Array1::<f64>::ones(n).view(),
        offset.view(),
        &[penalty()],
        spec(),
        &fit_options(),
    )
    .expect("unit-weight fit should succeed");

    // ── Fit B: every weight c. Same data, same model — a pure rescale. ─────
    let c = 1000.0_f64;
    let fitc = fit_gam(
        x.clone(),
        y.view(),
        Array1::<f64>::from_elem(n, c).view(),
        offset.view(),
        &[penalty()],
        spec(),
        &fit_options(),
    )
    .expect("rescaled-weight fit should succeed");

    // ── λ̂: the selected smoothing parameter must scale EXACTLY by c. ───────
    let lam1 = fit1.lambdas[0];
    let lamc = fitc.lambdas[0];
    let lam_ratio = lamc / lam1;
    let lam_ratio_rel = (lam_ratio - c).abs() / c;
    assert!(
        lam_ratio_rel < 1e-4,
        "REML smoothing parameter does not scale with the weight magnitude: \
         λ̂(w=1)={lam1:.10e}, λ̂(w={c})={lamc:.10e}, ratio={lam_ratio:.6e} \
         (must equal c={c}); the profiled-Gaussian objective is argmin-invariant \
         to a global weight rescale, so λ̂ must scale by exactly c \
         (src/solver/reml/unified.rs:7248)"
    );

    // ── EDF: effective degrees of freedom must be invariant. ───────────────
    let edf1 = fit1.edf_total().expect("edf for unit-weight fit");
    let edfc = fitc.edf_total().expect("edf for rescaled-weight fit");
    assert!(
        (edf1 - edfc).abs() < 1e-3,
        "effective dof changed under a no-op weight rescale: \
         EDF(w=1)={edf1:.8}, EDF(w={c})={edfc:.8}, diff={:.3e}",
        (edf1 - edfc).abs()
    );

    // ── Fitted coefficients: identical basis ⇒ identical β̂ ⇒ identical
    //    predictions. This is the user-visible fitted function. ─────────────
    let beta1 = &fit1.blocks[0].beta;
    let betac = &fitc.blocks[0].beta;
    assert_eq!(
        beta1.len(),
        betac.len(),
        "coefficient vectors have different lengths"
    );
    let max_abs = (0..beta1.len())
        .map(|j| (beta1[j] - betac[j]).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs < 1e-5,
        "fitted coefficients changed under a global weight rescale by c={c}: \
         max|β̂(w=1) − β̂(w={c})| = {max_abs:.3e}. A Gaussian REML fit with a \
         profiled dispersion must be invariant to scaling all weights by a \
         positive constant — the change is a smoothing-selection artefact, not \
         a statistical effect (the conditional SEs are already invariant)."
    );

    // ── σ̂²: the profiled scale is the ONLY quantity that may move, and it must
    //    absorb the whole rescale exactly: φ̂(c·w) = c·φ̂(w). Under inverse-
    //    variance weights `Var(yᵢ) = φ/wᵢ`, the weighted RSS `Σ wᵢ rᵢ²` scales by
    //    c, so φ̂ = RSS/(n−edf) → c·φ̂. That leaves `Var(yᵢ) = φ̂/wᵢ = c·φ̂/(c·wᵢ)`
    //    — and therefore the SEs and every fitted/predicted quantity —
    //    invariant. This is the dual of the λ̂ → c·λ̂ scaling above. ────────────
    let phi1 = fit1.dispersion_phi();
    let phic = fitc.dispersion_phi();
    let phi_ratio = phic / phi1; // expected: c
    let phi_ratio_rel = (phi_ratio - c).abs() / c;
    assert!(
        phi_ratio_rel < 1e-4,
        "profiled dispersion did not absorb the weight rescale: \
         φ̂(w=1)={phi1:.10e}, φ̂(w={c})={phic:.10e}, ratio={phi_ratio:.6e} \
         (must equal c={c}); under inverse-variance weights only σ̂² may move, \
         and it must scale by exactly c so Var(yᵢ)=φ̂/wᵢ — and thus the fit and \
         the SEs — stay invariant."
    );
}
