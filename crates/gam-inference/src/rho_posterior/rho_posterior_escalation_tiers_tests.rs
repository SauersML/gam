#![cfg(test)]
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

use super::{
    escalate_rho_posterior, rho_posterior_nuts, rho_posterior_quadrature, RhoPosteriorEscalation,
};
use ndarray::{array, Array1, Array2};

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
        |rho| Ok(gaussian_quadratic(rho, &rho_hat, &h)),
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
                Ok((
                    gaussian_quadratic(rho, &rho_hat, &h),
                    gaussian_quadratic_grad(rho, &rho_hat, &h),
                ))
            },
            seed,
        )
        .expect("tier-2 NUTS on a Gaussian quadratic must succeed")
    };
    let samples = run();

    assert!(samples.converged, "rhat = {} must be < 1.1", samples.rhat);
    // #3187: the draw count is derived from the run, not requested. The ESS
    // estimator never credits more than one effective draw per draw.
    assert!(
        samples.samples.nrows() as f64 >= samples.ess,
        "{} draws, ess = {}",
        samples.samples.nrows(),
        samples.ess
    );
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

/// The #3293 geometry, measured on the block-corrected Poisson `te` fit: the
/// LAML at `ρ̂` is nearly flat in its last two coordinates, and the sampled
/// density adds the default PC distribution correction (#2450) to every
/// coordinate, `c(r) = r/2 + θe^{−r/2}` with `θ = −ln(0.01)/10`.
struct PriorDominatedDensity {
    rho_hat: Array1<f64>,
    laml_diagonal: Array1<f64>,
    theta: f64,
}

impl PriorDominatedDensity {
    fn issue_3293() -> Self {
        Self {
            rho_hat: array![7.80, 7.93, 1.85, 4.33, 4.90],
            laml_diagonal: array![6.08, 6.13, 0.209, 0.00957, 0.00268],
            theta: -(0.01f64).ln() / 10.0,
        }
    }

    /// Cost, gradient and curvature of coordinate `i` of the separable density.
    fn coordinate(&self, i: usize, r: f64) -> (f64, f64, f64) {
        let h = self.laml_diagonal[i];
        let d = r - self.rho_hat[i];
        let e = (-0.5 * r).exp();
        (
            0.5 * h * d * d + 0.5 * r + self.theta * e,
            h * d + 0.5 - 0.5 * self.theta * e,
            h + 0.25 * self.theta * e,
        )
    }

    fn cost_and_gradient(&self, rho: &Array1<f64>) -> (f64, Array1<f64>) {
        let mut cost = 0.0;
        let mut gradient = Array1::zeros(rho.len());
        for i in 0..rho.len() {
            let (c, g, _) = self.coordinate(i, rho[i]);
            cost += c;
            gradient[i] = g;
        }
        (cost, gradient)
    }

    /// The density's mode and Hessian there: per-coordinate Newton, which the
    /// strictly convex coordinate costs make globally convergent from `ρ̂`
    /// once each step is halved until it lowers the cost.
    fn laplace_geometry(&self) -> (Array1<f64>, Array2<f64>) {
        let k = self.rho_hat.len();
        let mut mode = self.rho_hat.clone();
        let mut curvature = Array1::zeros(k);
        for i in 0..k {
            let mut r = self.rho_hat[i];
            loop {
                let (c, g, h) = self.coordinate(i, r);
                if g * g / h <= f64::EPSILON * c.abs() {
                    break;
                }
                let mut step = -g / h;
                while self.coordinate(i, r + step).0 > c {
                    step *= 0.5;
                }
                r += step;
            }
            mode[i] = r;
            curvature[i] = self.coordinate(i, r).2;
        }
        (mode, Array2::from_diag(&curvature))
    }

    /// Exact mean, variance and fourth central moment of coordinate `i`, by
    /// the trapezoid rule, which converges geometrically for a smooth
    /// integrand that has decayed to rounding at both ends of the grid.
    fn coordinate_moments(&self, i: usize, centre: f64, scale: f64) -> (f64, f64, f64) {
        let nodes = 40_001;
        // Below the mode the density falls double-exponentially; above it
        // no slower than `e^{−r/2}`: 150 units carry it to `e^{−75}`.
        let (lo, hi) = (centre - 40.0 * scale, centre + 150.0);
        let dx = (hi - lo) / (nodes - 1) as f64;
        let c0 = self.coordinate(i, centre).0;
        let grid: Vec<(f64, f64)> = (0..nodes)
            .map(|n| {
                let r = lo + n as f64 * dx;
                (r, (c0 - self.coordinate(i, r).0).exp())
            })
            .collect();
        let integral = |f: &dyn Fn(f64) -> f64| -> f64 {
            grid.iter()
                .enumerate()
                .map(|(n, &(r, w))| {
                    let edge = if n == 0 || n == nodes - 1 { 0.5 } else { 1.0 };
                    edge * w * f(r)
                })
                .sum::<f64>()
                * dx
        };
        let mass = integral(&|_| 1.0);
        let mean = integral(&|r| r) / mass;
        let variance = integral(&|r| (r - mean).powi(2)) / mass;
        let fourth = integral(&|r| (r - mean).powi(4)) / mass;
        (mean, variance, fourth)
    }
}

/// #3293. NUTS placed on the sampled density's own Laplace geometry recovers
/// that density's exact moments, and does so for fewer density evaluations
/// than NUTS placed on the criterion's geometry `(ρ̂, H_LAML)`, which is what
/// the escalation was handed before and which saturated the sampler's tree
/// depth on the real fit.
#[test]
fn nuts_on_the_sampled_density_geometry_recovers_prior_dominated_moments_3293() {
    let density = PriorDominatedDensity::issue_3293();
    let run = |centre: &Array1<f64>, hessian: &Array2<f64>| {
        let mut evaluations = 0usize;
        let samples = rho_posterior_nuts(
            centre,
            hessian,
            |rho: &Array1<f64>| {
                evaluations += 1;
                Ok(density.cost_and_gradient(rho))
            },
            super::ESCALATION_NUTS_SEED,
        )
        .expect("rho-posterior NUTS");
        (samples, evaluations)
    };
    let (mode, hessian) = density.laplace_geometry();
    let (samples, evaluations) = run(&mode, &hessian);
    let (_, criterion_evaluations) =
        run(&density.rho_hat, &Array2::from_diag(&density.laml_diagonal));
    assert!(
        samples.converged,
        "rhat {} ess {}",
        samples.rhat, samples.ess
    );
    assert!(
        evaluations < criterion_evaluations,
        "the sampled density's geometry took {evaluations} evaluations, the criterion's \
         {criterion_evaluations}"
    );

    // Monte-Carlo standard errors from the chain's own effective sample size:
    // `σ/√n` for the mean and, by the delta method on `Var(s²) = (μ₄−σ⁴)/n`,
    // `√((μ₄−σ⁴)/n) / (2σ)` for the standard deviation. Four of them bound a
    // deterministic run's error far outside Monte-Carlo chance.
    let n = samples.ess;
    for i in 0..mode.len() {
        let (mean, variance, fourth) =
            density.coordinate_moments(i, mode[i], hessian[[i, i]].recip().sqrt());
        let sd = variance.sqrt();
        let mean_se = sd / n.sqrt();
        let sd_se = ((fourth - variance * variance) / n).sqrt() / (2.0 * sd);
        let sampled_sd = samples.covariance[[i, i]].sqrt();
        assert!(
            (samples.mean[i] - mean).abs() <= 4.0 * mean_se,
            "coordinate {i}: sampled mean {} vs exact {mean} (se {mean_se})",
            samples.mean[i]
        );
        assert!(
            (sampled_sd - sd).abs() <= 4.0 * sd_se,
            "coordinate {i}: sampled sd {sampled_sd} vs exact {sd} (se {sd_se})"
        );
    }
}

/// (c) #3187: the seam routes by cost, not by a dimension cap. The `3^4 = 81`
/// node grid is cheaper than any converged NUTS run, so `K = 4` runs quadrature;
/// the `3^5 = 243` node grid is not, so `K = 5` runs NUTS; and `K = 17`, past the
/// old cap, still runs NUTS instead of reporting escalation unavailable.
#[test]
fn escalation_routes_to_the_tier_with_fewer_criterion_evaluations() {
    for (k, quadrature) in [(4, true), (5, false), (17, false)] {
        let rho_hat = Array1::from_shape_fn(k, |i| 0.1 * i as f64);
        let h = Array2::from_shape_fn((k, k), |(i, j)| if i == j { 2.0 } else { 0.0 });
        let escalation = escalate_rho_posterior(
            &rho_hat,
            &h,
            |rho: &Array1<f64>| Ok(gaussian_quadratic(rho, &rho_hat, &h)),
            |rho: &Array1<f64>| {
                Ok((
                    gaussian_quadratic(rho, &rho_hat, &h),
                    gaussian_quadratic_grad(rho, &rho_hat, &h),
                ))
            },
        );
        match escalation {
            RhoPosteriorEscalation::Quadrature(mixture) => {
                assert!(quadrature, "K = {k} must not run quadrature");
                assert_eq!(mixture.nodes.len(), 3usize.pow(k as u32));
            }
            RhoPosteriorEscalation::Nuts(samples) => {
                assert!(!quadrature, "K = {k} must not run NUTS");
                assert!(
                    samples.converged,
                    "K = {k}: rhat = {}, ess = {}",
                    samples.rhat, samples.ess
                );
                assert_eq!(samples.mean.len(), k);
            }
            RhoPosteriorEscalation::Unavailable { reason, .. } => {
                panic!("K = {k} escalation must run, got Unavailable: {reason}")
            }
        }
    }
}
