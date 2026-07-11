//! Regression: a prior weight of exactly `0` must make an observation
//! equivalent to its absence (R's `glm` uses `n.ok = nobs − Σ[w==0]`; mgcv
//! drops zero-weight observations). A zero-weighted row contributes exactly
//! zero to every weighted cross-product (`XᵀWX`, `XᵀWy`) and to the weighted
//! RSS (`w_i·r_i² = 0`), so the ONLY channel by which it could still perturb
//! the fit is an explicit observation count `n`. If `n` counts zero-weight
//! rows, the dispersion denominator `weighted_rss / (n − edf)` puts a numerator
//! that already excludes them over a denominator that doesn't, biasing φ̂ low,
//! shrinking every SE, and (via the REML `n` term) shifting the selected λ.
//!
//! This test isolates the count from every other moving part:
//!   - Reference fit B: the deterministic base dataset (n=200), no weights.
//!   - Augmented fit A: the SAME base dataset with a byte-identical second copy
//!     of every row stacked underneath, the copies weighted 0 and the
//!     originals weighted 1.
//! A's positive-weight observations are exactly B, so a correct implementation
//! makes A and B identical. The test asserts λ, EDF, dispersion φ̂, and the
//! fitted coefficients all match to machine precision (#584).

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

/// Deterministic, RNG-free `y ~ s(x)` base dataset (the n=200 reference B).
fn base_design() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
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
        y[i] = (tau * t).sin() + 0.5 * t + 0.3 * ((i as f64) * 2.399_963).sin();
    }
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    (x, y, s)
}

fn penalty(p: usize, s: &Array2<f64>) -> BlockwisePenalty {
    BlockwisePenalty::new(0..p, s.clone())
}

fn spec() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

#[test]
fn zero_weight_rows_are_equivalent_to_absent_rows() {
    let (x, y, s) = base_design();
    let n = x.nrows();
    let p = x.ncols();

    // ── Reference fit B: base dataset, all weights 1. ──────────────────────
    let w_b = Array1::<f64>::ones(n);
    let offset_b = Array1::<f64>::zeros(n);
    let fit_b = fit_gam(
        x.clone(),
        y.view(),
        w_b.view(),
        offset_b.view(),
        &[penalty(p, &s)],
        spec(),
        &fit_options(),
    )
    .expect("reference fit B should succeed");

    // ── Augmented fit A: base rows (weight 1) + byte-identical copies
    //    (weight 0). The covariate multiset is exactly B's doubled, and the
    //    positive-weight rows are exactly B. ────────────────────────────────
    let mut x_a = Array2::<f64>::zeros((2 * n, p));
    let mut y_a = Array1::<f64>::zeros(2 * n);
    let mut w_a = Array1::<f64>::zeros(2 * n);
    for i in 0..n {
        for j in 0..p {
            x_a[[i, j]] = x[[i, j]];
            x_a[[n + i, j]] = x[[i, j]];
        }
        y_a[i] = y[i];
        y_a[n + i] = y[i];
        w_a[i] = 1.0;
        w_a[n + i] = 0.0;
    }
    let offset_a = Array1::<f64>::zeros(2 * n);
    let fit_a = fit_gam(
        x_a,
        y_a.view(),
        w_a.view(),
        offset_a.view(),
        &[penalty(p, &s)],
        spec(),
        &fit_options(),
    )
    .expect("augmented fit A should succeed");

    // ── λ: REML smoothing selection must be identical. ─────────────────────
    let lam_b = fit_b.lambdas[0];
    let lam_a = fit_a.lambdas[0];
    let lam_rel = (lam_a - lam_b).abs() / lam_b.abs().max(1e-300);
    assert!(
        lam_rel < 1e-8,
        "λ differs between zero-weight-padded and base fits \
         (λ_base={lam_b:.10e}, λ_pad={lam_a:.10e}, rel diff {lam_rel:.3e}); \
         zero-weight rows are being counted in the REML sample size n (#584)"
    );

    // ── EDF: effective degrees of freedom must be identical. ───────────────
    let edf_b = fit_b.edf_total().expect("edf base");
    let edf_a = fit_a.edf_total().expect("edf pad");
    let edf_rel = (edf_a - edf_b).abs() / edf_b.abs().max(1e-300);
    assert!(
        edf_rel < 1e-8,
        "EDF differs (edf_base={edf_b:.10e}, edf_pad={edf_a:.10e}, \
         rel diff {edf_rel:.3e})"
    );

    // ── Dispersion φ̂: must be identical (was biased low by ≈ (n_pos−edf)/
    //    (n_total−edf) when zero-weight rows inflated the denominator). ──────
    let phi_b = fit_b.dispersion().expect("dispersion base").phi();
    let phi_a = fit_a.dispersion().expect("dispersion pad").phi();
    let phi_rel = (phi_a - phi_b).abs() / phi_b.abs().max(1e-300);
    assert!(
        phi_rel < 1e-8,
        "dispersion φ̂ differs (φ̂_base={phi_b:.10e}, φ̂_pad={phi_a:.10e}, \
         rel diff {phi_rel:.3e}); zero-weight rows inflate the scale denominator \
         (n − edf), biasing φ̂ low and shrinking every SE (#584)"
    );

    // ── Fitted function: coefficients share the same basis columns, so they
    //    must match to machine precision (⇒ identical predictions). ─────────
    let beta_b = &fit_b.blocks[0].beta;
    let beta_a = &fit_a.blocks[0].beta;
    assert_eq!(
        beta_b.len(),
        beta_a.len(),
        "coefficient vectors have different lengths"
    );
    for j in 0..beta_b.len() {
        let rel = (beta_a[j] - beta_b[j]).abs() / beta_b[j].abs().max(1e-6);
        assert!(
            (beta_a[j] - beta_b[j]).abs() < 1e-8 || rel < 1e-8,
            "coefficient {j} differs (β_base={:.10e}, β_pad={:.10e})",
            beta_b[j],
            beta_a[j]
        );
    }

    // ── Standard errors: identical scale ⇒ identical conditional SEs. ──────
    let vb_b = fit_b.beta_covariance().expect("Vb base").clone();
    let vb_a = fit_a.beta_covariance().expect("Vb pad").clone();
    for j in 0..p {
        let se_b = vb_b[[j, j]].max(0.0).sqrt();
        let se_a = vb_a[[j, j]].max(0.0).sqrt();
        let rel = (se_a - se_b).abs() / se_b.abs().max(1e-300);
        assert!(
            rel < 1e-7,
            "SE for coefficient {j} differs (se_base={se_b:.10e}, \
             se_pad={se_a:.10e}, rel {rel:.3e}); anti-conservative when zero-weight \
             rows shrink φ̂ (#584)"
        );
    }
}
