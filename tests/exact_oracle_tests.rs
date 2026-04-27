use ndarray::{Array1, Array2, array};

use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient,
};
use gam::smooth::BlockwisePenalty;
use gam::types::LikelihoodFamily;

// LAML-vs-quadrature-oracle gradient comparisons were removed: the impl
// returns dV_LAML/dρ for the standard mgcv Bayesian Laplace ML score
// (V_LAML = -ℓ + 0.5·β'Sβ + 0.5·log|H| - 0.5·log|S|+, Wood 2011), and a
// kernel-quadrature integral over exp(ℓ - 0.5·λ·β'Sβ) returns
// d log Z/dρ for the un-normalized integral. These differ by the prior
// Jacobian -0.5·d log|λS|+/dρ plus a Laplace approximation residual. Any
// fixed relative-error threshold conflates "implementation correct" with
// "Laplace approximation tight" and cannot distinguish the two.
//
// Replacement: finite-difference self-consistency. We compute the cost
// at ρ ± δ using the same `evaluate_externalcost_andridge` path that
// `evaluate_externalgradient` differentiates, take the central FD, and
// compare to the analytic gradient. This tests the gradient code itself
// against the cost code itself; both sides use the SAME Laplace
// approximation, so the approximation cancels exactly and any non-trivial
// FD-vs-analytic disagreement is a pure implementation bug (sign error,
// missing term, indexing mistake, stale Hessian, etc.).

/// Build the standard Logit options the FD tests use. `firth` toggles the
/// Firth bias-reduction path so we exercise the Jeffreys-prior addition to
/// V_LAML.
fn logit_opts(firth: bool) -> ExternalOptimOptions {
    ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodFamily::BinomialLogit,
        compute_inference: true,
        max_iter: 200,
        // Inner P-IRLS must converge tightly so cost(ρ ± δ) and grad(ρ) are
        // both evaluated at converged β̂; otherwise the FD captures β̂'s
        // residual change as well as the explicit ρ-dependence.
        tol: 1e-12,
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: if firth { Some(true) } else { None },
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    }
}

/// Central FD estimate of dV/dρ at the given ρ, using the SAME path that
/// `evaluate_externalgradient` differentiates.
fn fd_grad_at(
    y: &Array1<f64>,
    x: &Array2<f64>,
    opts: &ExternalOptimOptions,
    rho: f64,
    delta: f64,
) -> f64 {
    let w = Array1::ones(y.len());
    let offset = Array1::zeros(y.len());
    let s_list = vec![BlockwisePenalty::new(0..2, Array2::eye(2))];
    let cost_at = |r: f64| -> f64 {
        let rho_arr = array![r];
        evaluate_externalcost_andridge(
            y.view(),
            w.view(),
            x.clone(),
            offset.view(),
            &s_list,
            opts,
            &rho_arr,
        )
        .expect("external cost evaluation should succeed")
        .0
    };
    let cp = cost_at(rho + delta);
    let cm = cost_at(rho - delta);
    (cp - cm) / (2.0 * delta)
}

/// Analytic dV/dρ via the unified evaluator.
fn analytic_grad_at(
    y: &Array1<f64>,
    x: &Array2<f64>,
    opts: &ExternalOptimOptions,
    rho: f64,
) -> f64 {
    let w = Array1::ones(y.len());
    let offset = Array1::zeros(y.len());
    let s_list = vec![BlockwisePenalty::new(0..2, Array2::eye(2))];
    let rho_arr = array![rho];
    evaluate_externalgradient(
        y.view(),
        w.view(),
        x.clone(),
        offset.view(),
        &s_list,
        opts,
        &rho_arr,
    )
    .expect("external gradient evaluation should succeed")[0]
}

#[test]
fn test_lamlgradient_finite_difference_self_consistency() {
    // n=3 well-conditioned (non-separable) toy logit. FD self-consistency
    // is a pure implementation check; the fixture size doesn't affect the
    // FD-vs-analytic agreement so we keep this small for speed.
    let x = array![[1.0, -0.3], [1.0, 0.6], [1.0, 1.2]];
    let y = array![0.0, 1.0, 1.0];
    let opts = logit_opts(false);
    // δ = 1e-4: small enough that O(δ²) FD truncation error stays below
    // ~1e-8 on the cost (smooth in ρ), large enough that the cost
    // difference cp-cm is far above f64 round-off in the cost magnitudes.
    let delta = 1e-4_f64;
    let mut worst_rel = 0.0_f64;
    for &rho in &[-0.6_f64, -0.3, 0.0, 0.3, 0.6] {
        let fd = fd_grad_at(&y, &x, &opts, rho, delta);
        let an = analytic_grad_at(&y, &x, &opts, rho);
        // Relative error normalized by max(|fd|, 1e-3): 1e-3 floor avoids
        // dividing by ~0 if the cost happens to be near a stationary point
        // in ρ, where FD and analytic should both be ~0 but the ratio of
        // tiny numbers is dominated by FD truncation noise rather than any
        // implementation bug.
        let denom = fd.abs().max(1e-3);
        let rel = (fd - an).abs() / denom;
        eprintln!(
            "[fd_self] rho={:+.3} fd={:+.6e} analytic={:+.6e} abs_err={:.3e} rel_err={:.3e}",
            rho,
            fd,
            an,
            (fd - an).abs(),
            rel
        );
        worst_rel = worst_rel.max(rel);
        assert!(
            fd.is_finite() && an.is_finite(),
            "non-finite at rho={rho:+.3}: fd={fd:+.6e}, analytic={an:+.6e}"
        );
        assert!(
            rel < 1e-4,
            "FD-vs-analytic disagreement at rho={rho:+.3}: fd={fd:+.6e}, analytic={an:+.6e}, abs_err={:.3e}, rel_err={:.3e} (>= 1e-4 threshold)",
            (fd - an).abs(),
            rel
        );
    }
    eprintln!("[fd_self] worst_rel={worst_rel:.3e}");
}

#[test]
fn test_lamlgradient_firth_finite_difference_self_consistency() {
    // Near-separable n=3 design; turning on Firth bias-reduction adds the
    // exact Jeffreys-prior log-determinant correction to V_LAML. The FD
    // self-consistency check verifies that the analytic gradient correctly
    // differentiates the firth-corrected cost as well.
    let x = array![[1.0, -6.0], [1.0, 0.2], [1.0, 5.8]];
    let y = array![0.0, 0.0, 1.0];
    let opts = logit_opts(true);
    let delta = 1e-4_f64;
    let mut worst_rel = 0.0_f64;
    for &rho in &[-0.6_f64, -0.3, 0.0, 0.3, 0.6] {
        let fd = fd_grad_at(&y, &x, &opts, rho, delta);
        let an = analytic_grad_at(&y, &x, &opts, rho);
        let denom = fd.abs().max(1e-3);
        let rel = (fd - an).abs() / denom;
        eprintln!(
            "[fd_firth] rho={:+.3} fd={:+.6e} analytic={:+.6e} abs_err={:.3e} rel_err={:.3e}",
            rho,
            fd,
            an,
            (fd - an).abs(),
            rel
        );
        worst_rel = worst_rel.max(rel);
        assert!(
            fd.is_finite() && an.is_finite(),
            "non-finite at rho={rho:+.3}: fd={fd:+.6e}, analytic={an:+.6e}"
        );
        assert!(
            rel < 1e-4,
            "Firth FD-vs-analytic disagreement at rho={rho:+.3}: fd={fd:+.6e}, analytic={an:+.6e}, abs_err={:.3e}, rel_err={:.3e} (>= 1e-4 threshold)",
            (fd - an).abs(),
            rel
        );
    }
    eprintln!("[fd_firth] worst_rel={worst_rel:.3e}");
}
