//! Batch-native HMC using the BatchVector trait for zero GPU regression.
//!
//! This module provides a truly batch-native HMC implementation where the entire
//! batch of chains is treated as a single vector in phase space.

use crate::diag_mass::{DiagMass, RunningVariance};
use crate::euclidean::BatchVector;
use ndarray::Array3;
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use rand::SeedableRng;
use rand::distr::Distribution as RandDistribution;
use rand::rngs::SmallRng;
use rand_distr::StandardNormal;

/// A batched target density that returns per-chain log-probabilities.
///
/// This trait enables GPU-parallel execution by computing gradients for
/// all chains simultaneously.
pub trait BatchedHamiltonianTarget<V: BatchVector> {
    /// Returns per-chain log-densities and writes per-chain gradients into `grad`.
    /// For Tensor<B, 2>: position is [n_chains, dim], returns [n_chains] log-probs.
    fn logp_and_grad(&self, position: &V, grad: &mut V) -> V::Energy;
}

/// Batch-native HMC engine where "The Batch IS the Particle".
///
/// This struct stores all chains as a single batch vector `V`. For GPU backends,
/// this enables parallel execution without serialization.
#[derive(Debug)]
pub struct BatchedGenericHMC<V, Target>
where
    V: BatchVector,
    Target: BatchedHamiltonianTarget<V>,
{
    target: Target,
    step_size: V::Scalar,
    step_size_bar: V::Scalar,
    n_leapfrog: usize,
    target_accept_p: V::Scalar,
    gamma: V::Scalar,
    t_0: usize,
    kappa: V::Scalar,
    mu: V::Scalar,
    h_bar: V::Scalar,
    mass: DiagMass<V::Scalar>,
    /// ALL chains stored as single batch [n_chains, dim]
    position: V,
    /// Momentum buffer [n_chains, dim]
    momentum: V,
    /// Gradient buffer [n_chains, dim]
    grad: V,
    /// Proposal position buffer [n_chains, dim]
    proposal_pos: V,
    /// Proposal momentum buffer [n_chains, dim]
    proposal_mom: V,
    rng: SmallRng,
    n_chains: usize,
    dim: usize,
}

impl<V, Target> BatchedGenericHMC<V, Target>
where
    V: BatchVector,
    V::Scalar: Float + FromPrimitive + ToPrimitive + Zero,
    Target: BatchedHamiltonianTarget<V>,
    StandardNormal: RandDistribution<V::Scalar>,
{
    /// Create a new batch-native HMC sampler.
    ///
    /// `initial_position` is a batch tensor of shape [n_chains, dim].
    pub fn new(
        target: Target,
        initial_position: V,
        step_size: V::Scalar,
        n_leapfrog: usize,
    ) -> Self {
        let n_chains = initial_position.n_chains();
        let dim = initial_position.dim_per_chain();

        Self {
            target,
            step_size,
            step_size_bar: step_size,
            n_leapfrog,
            target_accept_p: V::Scalar::from_f64(0.8).unwrap(),
            gamma: V::Scalar::from_f64(0.05).unwrap(),
            t_0: 10,
            kappa: V::Scalar::from_f64(0.75).unwrap(),
            mu: (V::Scalar::from_f64(10.0).unwrap() * step_size).ln(),
            h_bar: V::Scalar::zero(),
            mass: DiagMass::identity(dim),
            momentum: initial_position.zeros_like(),
            grad: initial_position.zeros_like(),
            proposal_pos: initial_position.zeros_like(),
            proposal_mom: initial_position.zeros_like(),
            position: initial_position,
            rng: SmallRng::from_rng(&mut rand::rng()),
            n_chains,
            dim,
        }
    }

    /// Set the random seed for reproducibility.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    pub fn set_target_accept(mut self, target_accept_p: V::Scalar) -> Self {
        assert!(
            target_accept_p > V::Scalar::zero() && target_accept_p < V::Scalar::one(),
            "target_accept must be in (0, 1)"
        );
        self.target_accept_p = target_accept_p;
        self
    }

    /// Run the sampler, collecting `n_collect` samples after discarding `n_discard`.
    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Array3<V::Scalar> {
        self.warmup(n_discard);

        // Collect samples
        let mut out = Array3::<V::Scalar>::zeros((self.n_chains, n_collect, self.dim));
        let mut scratch = vec![V::Scalar::zero(); self.n_chains * self.dim];

        for step_idx in 0..n_collect {
            self.step();
            self.position.write_to_slice(&mut scratch);
            // Copy from flat slice to [n_chains, n_collect, dim]
            for chain_idx in 0..self.n_chains {
                for d in 0..self.dim {
                    out[[chain_idx, step_idx, d]] = scratch[chain_idx * self.dim + d];
                }
            }
        }
        out
    }

    /// Run the sampler and return device-native samples without host readback.
    pub fn run_positions(&mut self, n_collect: usize, n_discard: usize) -> Vec<V> {
        self.warmup(n_discard);
        let mut samples = Vec::with_capacity(n_collect);
        for _ in 0..n_collect {
            self.step();
            samples.push(self.position.clone());
        }
        samples
    }

    pub(crate) fn warmup(&mut self, n_discard: usize) {
        if n_discard == 0 {
            return;
        }
        self.step_size = self.find_reasonable_step_size(self.step_size);
        self.reset_step_size_adaptation();
        let mass_update_iter = Self::mass_adaptation_iter(n_discard);
        let mut adapt_iter = 0;
        let mut running = RunningVariance::new(self.dim);
        let mut scratch = vec![V::Scalar::zero(); self.n_chains * self.dim];
        for iter in 1..=n_discard {
            let accept_p = self.step();
            adapt_iter += 1;
            self.adapt_step_size(accept_p, adapt_iter);
            if iter <= mass_update_iter {
                self.position.write_to_slice(&mut scratch);
                running.update_batch(&scratch);
            }
            if iter == mass_update_iter && self.update_mass_from_running(&running) {
                self.step_size = self.find_reasonable_step_size(self.step_size_bar);
                self.reset_step_size_adaptation();
                adapt_iter = 0;
            }
        }
        self.step_size = self.step_size_bar;
    }

    fn mass_adaptation_iter(n_discard: usize) -> usize {
        let iter = n_discard / 2;
        if iter < 5 || iter >= n_discard {
            0
        } else {
            iter
        }
    }

    fn update_mass_from_running(&mut self, running: &RunningVariance<V::Scalar>) -> bool {
        if running.sample_count() < 5 {
            return false;
        }
        let regularize = V::Scalar::from_f64(0.05).unwrap();
        let jitter = V::Scalar::from_f64(1e-6).unwrap();
        let Some(var) = running.regularized_variance(regularize, jitter) else {
            return false;
        };
        self.mass = DiagMass::from_variance(var, jitter);
        true
    }

    fn find_reasonable_step_size(&mut self, mut step_size: V::Scalar) -> V::Scalar {
        let two = V::Scalar::from_f64(2.0).unwrap();
        let half = V::Scalar::from_f64(0.5).unwrap();
        let min_step = V::Scalar::epsilon();
        let max_iters = 32;

        let mut accept_p = self.mean_acceptance_for_step_size(step_size);
        while (!accept_p.is_finite() || accept_p <= V::Scalar::zero()) && step_size > min_step {
            step_size = (step_size / two).max(min_step);
            accept_p = self.mean_acceptance_for_step_size(step_size);
        }
        if !accept_p.is_finite() {
            return min_step;
        }

        let grow = accept_p > half;
        for _ in 0..max_iters {
            let candidate = if grow {
                step_size * two
            } else {
                step_size / two
            };
            if candidate <= min_step {
                break;
            }
            let candidate_accept = self.mean_acceptance_for_step_size(candidate);
            if !candidate_accept.is_finite() || candidate_accept <= V::Scalar::zero() {
                if grow {
                    break;
                }
                step_size = candidate.max(min_step);
                continue;
            }
            if (candidate_accept > half) != grow {
                break;
            }
            step_size = candidate;
        }
        step_size.max(min_step)
    }

    fn reset_step_size_adaptation(&mut self) {
        self.step_size_bar = self.step_size;
        self.mu = (V::Scalar::from_f64(10.0).unwrap() * self.step_size).ln();
        self.h_bar = V::Scalar::zero();
    }

    fn adapt_step_size(&mut self, accept_p: V::Scalar, iter: usize) {
        let m = V::Scalar::from_usize(iter).unwrap();
        let eta = V::Scalar::one()
            / V::Scalar::from_usize(iter + self.t_0).expect("iteration converts to scalar");
        self.h_bar =
            (V::Scalar::one() - eta) * self.h_bar + eta * (self.target_accept_p - accept_p);
        self.step_size = (self.mu - m.sqrt() / self.gamma * self.h_bar).exp();
        let eta_bar = m.powf(-self.kappa);
        self.step_size_bar = ((V::Scalar::one() - eta_bar) * self.step_size_bar.ln()
            + eta_bar * self.step_size.ln())
        .exp();
    }

    /// Perform one HMC step on ALL chains simultaneously.
    ///
    /// This is GPU-parallel: no loops over chains, uses device-native RNG,
    /// and vectorized acceptance via masking.
    pub fn step(&mut self) -> V::Scalar {
        self.step_with_step_size(self.step_size, true)
    }

    fn mean_acceptance_for_step_size(&mut self, step_size: V::Scalar) -> V::Scalar {
        self.step_with_step_size(step_size, false)
    }

    fn step_with_step_size(&mut self, step_size: V::Scalar, apply: bool) -> V::Scalar {
        let mass_inv = self.mass.inv();
        let mass_sqrt = self.mass.sqrt();

        // 1. Sample momentum for all chains (device-native RNG)
        self.momentum.fill_random_normal(&mut self.rng);
        self.momentum.scale_diag_assign(mass_sqrt);

        // 2. Compute current kinetic energy [n_chains]
        let ke_current = self.momentum.kinetic_energy_diag(mass_inv);

        // 3. Compute current potential energy [n_chains]
        self.grad.fill_zero();
        let logp_current = self.target.logp_and_grad(&self.position, &mut self.grad);

        // 4. Copy current state to proposal buffers
        self.proposal_pos.assign(&self.position);
        self.proposal_mom.assign(&self.momentum);

        // 5. Leapfrog integration (all chains in parallel)
        let logp_proposed = Self::leapfrog(
            &self.target,
            &mut self.proposal_pos,
            &mut self.proposal_mom,
            &mut self.grad,
            step_size,
            self.n_leapfrog,
            mass_inv,
        );

        // 6. Compute proposed kinetic energy [n_chains]
        let ke_proposed = self.proposal_mom.kinetic_energy_diag(mass_inv);

        // 7. Compute log acceptance probability (element-wise for batch)
        // log_accept = (logp_proposed - logp_current) + (ke_current - ke_proposed)
        let delta_logp = V::energy_sub(&logp_proposed, &logp_current);
        let delta_ke = V::energy_sub(&ke_current, &ke_proposed);
        let log_accept = V::energy_add(&delta_logp, &delta_ke);
        let mean_accept = V::mean_acceptance(&log_accept);

        // 8. Sample uniform [0,1] for acceptance test [n_chains]
        if apply {
            let u = self.position.sample_uniform(&mut self.rng);
            let ln_u = V::energy_ln(&u);

            // 9. Create acceptance mask and update positions (GPU-friendly masking)
            let mask = V::accept_mask(&log_accept, &ln_u);
            self.position.masked_assign(&self.proposal_pos, &mask);
        }
        mean_accept
    }

    /// Leapfrog integration on the entire batch.
    fn leapfrog(
        target: &Target,
        proposal_pos: &mut V,
        proposal_mom: &mut V,
        grad: &mut V,
        step_size: V::Scalar,
        n_leapfrog: usize,
        inv_mass: &[V::Scalar],
    ) -> V::Energy {
        let half = V::Scalar::from_f64(0.5).unwrap() * step_size;

        let mut logp = target.logp_and_grad(proposal_pos, grad);

        for _ in 0..n_leapfrog {
            // Half momentum update
            proposal_mom.add_scaled_assign(grad, half);

            // Full position update
            proposal_pos.add_diag_scaled_assign(proposal_mom, inv_mass, step_size);

            // Recompute gradient at new position
            logp = target.logp_and_grad(proposal_pos, grad);

            // Half momentum update
            proposal_mom.add_scaled_assign(grad, half);
        }
        logp
    }

    /// Get a reference to the current positions.
    pub fn positions(&self) -> &V {
        &self.position
    }

    /// Get a reference to the target distribution.
    pub fn target(&self) -> &Target {
        &self.target
    }

    /// Get a reference to the step size.
    pub fn step_size(&self) -> &V::Scalar {
        &self.step_size
    }

    /// Get the number of leapfrog steps.
    pub fn n_leapfrog(&self) -> usize {
        self.n_leapfrog
    }

    /// Clone the RNG state.
    pub fn rng_clone(&self) -> SmallRng {
        self.rng.clone()
    }

    #[cfg(test)]
    pub(crate) fn mass_diag(&self) -> Vec<V::Scalar> {
        self.mass
            .inv()
            .iter()
            .map(|&inv| V::Scalar::one() / inv)
            .collect()
    }
}
