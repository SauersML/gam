use gam::inference::hmc::{NUTSMassMatrixConfig, NutsConfig};
use general_mcmc::generic_hmc::HamiltonianTarget;
use general_mcmc::generic_nuts::GenericNUTS;
use ndarray::{Array1, Array2, Axis, arr1, arr2};

#[derive(Clone)]
struct GaussianTarget {
    mean: Array1<f64>,
    precision: Array2<f64>,
}

impl HamiltonianTarget<Array1<f64>> for GaussianTarget {
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        let delta = position - &self.mean;
        let q = self.precision.dot(&delta);
        grad.assign(&(-&q));
        -0.5 * delta.dot(&q)
    }
}

fn leapfrog_step(
    target: &GaussianTarget,
    q: &Array1<f64>,
    p: &Array1<f64>,
    eps: f64,
) -> (Array1<f64>, Array1<f64>) {
    let mut g0 = Array1::<f64>::zeros(q.len());
    target.logp_and_grad(q, &mut g0);
    let p_half = p + &(g0 * (0.5 * eps));
    let q_new = q + &(p_half.clone() * eps);
    let mut g1 = Array1::<f64>::zeros(q.len());
    target.logp_and_grad(&q_new, &mut g1);
    let p_new = p_half + &(g1 * (0.5 * eps));
    (q_new, p_new)
}

#[test]
fn nuts_leapfrog_identity_and_gaussian_posterior_recovery() {
    let mean = arr1(&[1.5, -0.5]);
    let cov = arr2(&[[1.2, 0.3], [0.3, 0.8]]);
    let det = cov[[0, 0]] * cov[[1, 1]] - cov[[0, 1]] * cov[[1, 0]];
    let precision = arr2(&[
        [cov[[1, 1]] / det, -cov[[0, 1]] / det],
        [-cov[[1, 0]] / det, cov[[0, 0]] / det],
    ]);
    let target = GaussianTarget {
        mean: mean.clone(),
        precision,
    };

    let eps = 0.1;
    let k = 50;
    let q0 = arr1(&[0.2, -1.0]);
    let p0 = arr1(&[0.3, 0.7]);
    let mut qf = q0.clone();
    let mut pf = p0.clone();
    for _ in 0..k {
        (qf, pf) = leapfrog_step(&target, &qf, &pf, eps);
    }
    for _ in 0..k {
        (qf, pf) = leapfrog_step(&target, &qf, &pf, -eps);
    }
    let rev_err = (&qf - &q0).mapv(f64::abs).sum() + (&pf - &p0).mapv(f64::abs).sum();
    assert!(
        rev_err < 1e-10,
        "leapfrog reversibility violated: {}",
        rev_err
    );

    let mut qd = q0.clone();
    let mut pd = p0.clone();
    let mut h_vals = Vec::new();
    for _ in 0..50 {
        let dq = &qd - &mean;
        let u = 0.5 * dq.dot(&target.precision.dot(&dq));
        let k_e = 0.5 * pd.dot(&pd);
        h_vals.push(u + k_e);
        (qd, pd) = leapfrog_step(&target, &qd, &pd, eps);
    }
    let h0 = h_vals[0];
    let max_drift = h_vals
        .into_iter()
        .map(|h| (h - h0).abs())
        .fold(0.0_f64, f64::max);
    // Leapfrog is symplectic and reversible, but for a Gaussian target it
    // exactly conserves a nearby modified Hamiltonian, not the continuous-time
    // Hamiltonian itself. The true-energy error is therefore bounded at O(eps²)
    // over stable trajectories rather than driven to roundoff. This constant is
    // deliberately tight for the anisotropic covariance above while respecting
    // the correct leapfrog backward-error identity.
    assert!(
        max_drift < 0.15 * eps * eps,
        "energy drift too large: {}",
        max_drift
    );

    let theta_minus = arr1(&[-2.0, -2.0]);
    let theta_plus = arr1(&[2.0, 2.0]);
    let r_minus = arr1(&[1.0, 1.0]);
    let r_plus = arr1(&[1.0, 1.0]);
    let delta = &theta_plus - &theta_minus;
    let should_uturn = delta.dot(&r_minus) <= 0.0 || delta.dot(&r_plus) <= 0.0;
    assert!(!should_uturn, "constructed non-U-turn state misclassified");
    let r_minus_uturn = arr1(&[-1.0, -1.0]);
    let should_uturn_now = delta.dot(&r_minus_uturn) <= 0.0 || delta.dot(&r_plus) <= 0.0;
    assert!(
        should_uturn_now,
        "U-turn condition did not fire when expected"
    );

    // Use enough draws for the covariance assertion to test sampler bias rather
    // than ordinary Monte Carlo variability.
    let n = 500_000usize;
    let initial = vec![arr1(&[0.0, 0.0])];
    let mut sampler = GenericNUTS::new_with_mass_matrix(
        target,
        initial,
        NutsConfig::default().target_accept,
        NUTSMassMatrixConfig::disabled(),
    )
    .set_seed(123);
    let draws = sampler.run(n, 1000);
    let chain = draws.index_axis(Axis(0), 0).to_owned();
    let sample_mean = chain.mean_axis(Axis(0)).unwrap();
    let centered = &chain - &sample_mean;
    let sample_cov = centered.t().dot(&centered) / ((n - 1) as f64);

    for d in 0..2 {
        let sigma = (cov[[d, d]] / n as f64).sqrt();
        let z = (sample_mean[d] - mean[d]).abs() / sigma;
        assert!(z < 3.0, "mean recovery failed on dim {d}: z={z}");
    }

    let fro = (&sample_cov - &cov).mapv(|v| v * v).sum().sqrt();
    assert!(
        fro < 0.01,
        "covariance recovery failed: frobenius distance={fro}"
    );
}
