//! #938 escalation tiers over the smoothing-parameter posterior `π(ρ|y)`.
//!
//! Reference-as-truth: every assertion here is against self-constructed
//! closed-form truth. On an exactly Gaussian criterion
//! `−log π(ρ|y) = ½(ρ−ρ̂)ᵀH(ρ−ρ̂)` the Laplace posterior IS the truth:
//! Tier-1 Gauss-Hermite quadrature must reproduce its moments to quadrature
//! precision, and Tier-2 NUTS (exact gradient, Hessian-whitened) must recover
//! them within Monte-Carlo error under a fixed seed. The auto-selection seam
//! must route an `Escalate` verdict to the right tier by `K`, and the
//! mixture-corrected coefficient covariance must reduce to the plug-in
//! `Vb(ρ̂)` when all mixture weight concentrates at `ρ̂`.

use gam::inference::rho_posterior::{rho_posterior_nuts, rho_posterior_quadrature};
use ndarray::{Array1, Array2, array};

/// `½ (ρ−ρ̂)ᵀ H (ρ−ρ̂)` — the criterion whose exact posterior is `N(ρ̂, H⁻¹)`.
fn gaussian_quadratic(rho: &Array1<f64>, rho_hat: &Array1<f64>, h: &Array2<f64>) -> f64 {
    let d = rho - rho_hat;
    let mut q = 0.0;
    for i in 0..d.len() {
        for j in 0..d.len() {
            q += d[i] * h[[i, j]] * d[j];
        }
    }
    0.5 * q
}

/// `∇ = H (ρ−ρ̂)` for the quadratic criterion above.
fn gaussian_quadratic_grad(
    rho: &Array1<f64>,
    rho_hat: &Array1<f64>,
    h: &Array2<f64>,
) -> Array1<f64> {
    let d = rho - rho_hat;
    let k = d.len();
    Array1::from_shape_fn(k, |i| (0..k).map(|j| h[[i, j]] * d[j]).sum())
}

fn inverse_2x2(h: &Array2<f64>) -> Array2<f64> {
    let det = h[[0, 0]] * h[[1, 1]] - h[[0, 1]] * h[[1, 0]];
    array![
        [h[[1, 1]] / det, -h[[0, 1]] / det],
        [-h[[1, 0]] / det, h[[0, 0]] / det]
    ]
}

/// (a) Tier-1 quadrature on an exact Gaussian quadratic criterion reproduces
/// the Laplace posterior moments to quadrature precision: mean = ρ̂ and
/// covariance = H⁻¹ (the GH rule integrates degree-2 polynomials exactly, and
/// the importance reweighting is identically the GH weight for this target).
#[test]
fn quadrature_reproduces_laplace_moments_on_gaussian_quadratic() {
    let rho_hat = array![0.4, -1.2];
    let h = array![[2.0, 0.5], [0.5, 1.5]];
    let truth_cov = inverse_2x2(&h);

    let mixture = rho_posterior_quadrature(
        &rho_hat,
        &h,
        |rho| Some(gaussian_quadratic(rho, &rho_hat, &h)),
        None,
    )
    .expect("tier-1 quadrature on a Gaussian quadratic must succeed");

    // K = 2 auto-selects 5 nodes per axis -> 25 nodes.
    assert_eq!(mixture.nodes.len(), 25);
    let total: f64 = mixture.nodes.iter().map(|n| n.weight).sum();
    assert!(
        (total - 1.0).abs() < 1e-10,
        "mixture weights must sum to 1, got {total}"
    );
    for i in 0..2 {
        assert!(
            (mixture.mean[i] - rho_hat[i]).abs() < 1e-8,
            "posterior mean component {i} must equal rho_hat: {} vs {}",
            mixture.mean[i],
            rho_hat[i]
        );
        for j in 0..2 {
            assert!(
                (mixture.covariance[[i, j]] - truth_cov[[i, j]]).abs() < 1e-8,
                "posterior covariance [{i},{j}] must equal H^-1: {} vs {}",
                mixture.covariance[[i, j]],
                truth_cov[[i, j]]
            );
        }
    }
    // The exact-Gaussian target makes the importance correction a no-op, so
    // the node weights ARE the GH weights and the ESS is near the node count.
    assert!(
        mixture.effective_sample_size > 5.0,
        "Gaussian target must keep a healthy quadrature ESS, got {}",
        mixture.effective_sample_size
    );
}

/// (b) Tier-2 NUTS on the same Gaussian quadratic recovers mean and covariance
/// within Monte-Carlo error with a fixed seed, and the run is deterministic
/// (same seed -> bit-identical moments).
#[test]
fn nuts_recovers_gaussian_quadratic_moments_with_fixed_seed() {
    let rho_hat = array![0.4, -1.2];
    let h = array![[2.0, 0.5], [0.5, 1.5]];
    let truth_cov = inverse_2x2(&h);
    let seed = 0x938_0002_u64;

    let run = || {
        rho_posterior_nuts(
            &rho_hat,
            &h,
            |rho: &Array1<f64>| {
                Some((
                    gaussian_quadratic(rho, &rho_hat, &h),
                    gaussian_quadratic_grad(rho, &rho_hat, &h),
                ))
            },
            512,
            seed,
        )
        .expect("tier-2 NUTS on a Gaussian quadratic must succeed")
    };
    let samples = run();

    assert!(samples.converged, "rhat = {} must be < 1.1", samples.rhat);
    assert!(samples.samples.nrows() >= 512);
    for i in 0..2 {
        assert!(
            (samples.mean[i] - rho_hat[i]).abs() < 0.12,
            "NUTS mean component {i} outside MC error: {} vs {}",
            samples.mean[i],
            rho_hat[i]
        );
        for j in 0..2 {
            assert!(
                (samples.covariance[[i, j]] - truth_cov[[i, j]]).abs() < 0.2,
                "NUTS covariance [{i},{j}] outside MC error: {} vs {}",
                samples.covariance[[i, j]],
                truth_cov[[i, j]]
            );
        }
    }

    // Deterministic seeding: a second run with the same seed is bit-identical.
    let again = run();
    for i in 0..2 {
        assert_eq!(
            samples.mean[i].to_bits(),
            again.mean[i].to_bits(),
            "fixed-seed NUTS must be deterministic in mean[{i}]"
        );
    }
}

