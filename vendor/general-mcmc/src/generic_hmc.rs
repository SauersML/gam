use crate::diag_mass::{DiagMass, RunningVariance};
use crate::euclidean::EuclideanVector;
use crate::stats::{MultiChainTracker, RunStats};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array3, ArrayView1, s};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use rand::distr::Distribution as RandDistribution;
// rand_distr types implement rand::distr::Distribution for rand 0.9; use this trait to avoid conflicts.
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{StandardNormal, StandardUniform};
use std::error::Error;

/// A target density that can write its gradient in-place for a given position.
pub trait HamiltonianTarget<V: EuclideanVector> {
    /// Returns the log-density at `position` and writes the gradient into `grad`.
    fn logp_and_grad(&self, position: &V, grad: &mut V) -> V::Scalar;
}

/// Backend-agnostic, in-place Hamiltonian Monte Carlo engine.
#[derive(Debug)]
pub struct GenericHMC<V, Target>
where
    V: EuclideanVector,
    Target: HamiltonianTarget<V>,
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
    positions: Vec<V>,
    rng: SmallRng,
    grad_buffers: Vec<V>,
    momentum_buffers: Vec<V>,
    proposal_positions: Vec<V>,
    proposal_momenta: Vec<V>,
    dim: usize,
}

type RunResult<T> = Result<(Array3<T>, RunStats), Box<dyn Error>>;

impl<V, Target> GenericHMC<V, Target>
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive + ToPrimitive,
    Target: HamiltonianTarget<V>,
    StandardNormal: RandDistribution<V::Scalar>,
    StandardUniform: RandDistribution<V::Scalar>,
{
    pub fn new(
        target: Target,
        initial_positions: Vec<V>,
        step_size: V::Scalar,
        n_leapfrog: usize,
    ) -> Self {
        assert!(
            !initial_positions.is_empty(),
            "initial_positions must not be empty"
        );
        let dim = initial_positions[0].len();
        let template = initial_positions[0].zeros_like();
        let n_chains = initial_positions.len();
        let grad_buffers = (0..n_chains)
            .map(|_| template.zeros_like())
            .collect::<Vec<_>>();
        let momentum_buffers = (0..n_chains)
            .map(|_| template.zeros_like())
            .collect::<Vec<_>>();
        let proposal_positions = initial_positions
            .iter()
            .map(|p| p.zeros_like())
            .collect::<Vec<_>>();
        let proposal_momenta = (0..n_chains)
            .map(|_| template.zeros_like())
            .collect::<Vec<_>>();
        let mut thread_rng = rand::rng();
        let rng = SmallRng::from_rng(&mut thread_rng);

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
            positions: initial_positions,
            rng,
            grad_buffers,
            momentum_buffers,
            proposal_positions,
            proposal_momenta,
            dim,
        }
    }

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

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Array3<V::Scalar> {
        self.warmup(n_discard);
        let n_chains = self.positions.len();
        let mut out = Array3::<V::Scalar>::zeros((n_chains, n_collect, self.dim));
        let mut scratch = vec![V::Scalar::zero(); self.dim];

        for step_idx in 0..n_collect {
            self.step();
            for (chain_idx, pos) in self.positions.iter().enumerate() {
                pos.write_to_slice(&mut scratch);
                let view = ArrayView1::from(&scratch);
                out.slice_mut(s![chain_idx, step_idx, ..]).assign(&view);
            }
        }
        out
    }

    pub fn run_progress(&mut self, n_collect: usize, n_discard: usize) -> RunResult<V::Scalar> {
        self.warmup(n_discard);

        let n_chains = self.positions.len();
        let mut out = Array3::<V::Scalar>::zeros((n_chains, n_collect, self.dim));
        let mut scratch = vec![V::Scalar::zero(); self.dim];
        let mut flattened = vec![V::Scalar::zero(); n_chains * self.dim];

        let mut tracker = MultiChainTracker::new(n_chains, self.dim);
        self.flatten_positions(&mut flattened);
        tracker.step(&flattened)?;

        let pb = ProgressBar::new(n_collect as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:8} {bar:40.cyan/blue} {pos}/{len} ({eta}) | {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_prefix("HMC");

        for step_idx in 0..n_collect {
            self.step();
            for (chain_idx, pos) in self.positions.iter().enumerate() {
                pos.write_to_slice(&mut scratch);
                let view = ArrayView1::from(&scratch);
                out.slice_mut(s![chain_idx, step_idx, ..]).assign(&view);
            }
            self.flatten_positions(&mut flattened);
            tracker.step(&flattened)?;
            if let Ok(max_rhat) = tracker.max_rhat() {
                pb.set_message(format!(
                    "p(accept)≈{:.2} max(rhat)≈{:.2}",
                    tracker.p_accept, max_rhat
                ));
            }
            pb.inc(1);
        }
        pb.finish_with_message("Done!");

        let stats = RunStats::from(out.view());
        Ok((out, stats))
    }

    fn warmup(&mut self, n_discard: usize) {
        if n_discard == 0 {
            return;
        }
        self.step_size = self.find_reasonable_step_size(self.step_size);
        self.reset_step_size_adaptation();
        let mass_update_iter = Self::mass_adaptation_iter(n_discard);
        let mut adapt_iter = 0;
        let mut running = RunningVariance::new(self.dim);
        let mut scratch = vec![V::Scalar::zero(); self.positions.len() * self.dim];
        for iter in 1..=n_discard {
            let accept_p = self.step();
            adapt_iter += 1;
            self.adapt_step_size(accept_p, adapt_iter);
            if iter <= mass_update_iter {
                self.flatten_positions(&mut scratch);
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

    fn flatten_positions(&self, out: &mut [V::Scalar]) {
        let dim = self.dim;
        for (i, pos) in self.positions.iter().enumerate() {
            let start = i * dim;
            let end = start + dim;
            pos.write_to_slice(&mut out[start..end]);
        }
    }

    pub(crate) fn step(&mut self) -> V::Scalar {
        self.step_with_step_size(self.step_size, true)
    }

    fn mean_acceptance_for_step_size(&mut self, step_size: V::Scalar) -> V::Scalar {
        self.step_with_step_size(step_size, false)
    }

    fn step_with_step_size(&mut self, step_size: V::Scalar, apply: bool) -> V::Scalar {
        let n_chains = self.positions.len();
        // Kinetic energy uses 0.5 only (NOT step_size * 0.5)
        let ke_half = V::Scalar::from_f64(0.5).unwrap();
        let mut accept_sum = V::Scalar::zero();
        let mass_inv = self.mass.inv();
        let mass_sqrt = self.mass.sqrt();

        for i in 0..n_chains {
            let grad = &mut self.grad_buffers[i];
            grad.fill_zero();
            let logp_current = self.target.logp_and_grad(&self.positions[i], grad);

            let momentum = &mut self.momentum_buffers[i];
            momentum.fill_standard_normal(&mut self.rng);
            momentum.scale_diag_assign(mass_sqrt);
            let ke_current = momentum.quad_form_diag(mass_inv) * ke_half;

            let proposal_pos = &mut self.proposal_positions[i];
            proposal_pos.assign(&self.positions[i]);
            let proposal_mom = &mut self.proposal_momenta[i];
            proposal_mom.assign(momentum);

            let logp_proposed = Self::leapfrog_chain(
                &self.target,
                proposal_pos,
                proposal_mom,
                grad,
                mass_inv,
                step_size,
                self.n_leapfrog,
                logp_current,
            );

            let ke_proposed = proposal_mom.quad_form_diag(mass_inv) * ke_half;
            let log_accept = (logp_proposed - logp_current) + (ke_current - ke_proposed);
            let accept_p = if !log_accept.is_finite() {
                V::Scalar::zero()
            } else if log_accept >= V::Scalar::zero() {
                V::Scalar::one()
            } else {
                log_accept.exp()
            };
            accept_sum = accept_sum + accept_p;
            if apply {
                let ln_u: V::Scalar = self.rng.sample(StandardUniform).ln();
                if ln_u <= log_accept {
                    self.positions[i].assign(proposal_pos);
                }
            }
        }
        accept_sum / V::Scalar::from_usize(n_chains).unwrap()
    }

    fn leapfrog_chain(
        target: &Target,
        position: &mut V,
        momentum: &mut V,
        grad: &mut V,
        inv_mass: &[V::Scalar],
        step_size: V::Scalar,
        n_leapfrog: usize,
        mut logp: V::Scalar,
    ) -> V::Scalar {
        let half = V::Scalar::from_f64(0.5).unwrap() * step_size;
        for _ in 0..n_leapfrog {
            momentum.add_scaled_assign(grad, half);
            position.add_diag_scaled_assign(momentum, inv_mass, step_size);
            logp = target.logp_and_grad(position, grad);
            momentum.add_scaled_assign(grad, half);
        }
        logp
    }

    pub(crate) fn positions(&self) -> &[V] {
        &self.positions
    }

    pub(crate) fn rng_clone(&self) -> SmallRng {
        self.rng.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{GenericHMC, HamiltonianTarget};
    use ndarray::{Array1, arr1};

    #[derive(Clone, Copy, Debug)]
    struct AnisotropicGaussian2D;

    impl HamiltonianTarget<Array1<f64>> for AnisotropicGaussian2D {
        fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
            let dx = position[0];
            let dy = position[1];
            grad[0] = -4.0 * dx;
            grad[1] = -(1.0 / 9.0) * dy;
            -0.5 * (4.0 * dx * dx + (dy * dy) / 9.0)
        }
    }

    #[test]
    fn test_generic_warmup_adapts_diagonal_mass_for_anisotropic_target() {
        let initial_positions = vec![
            arr1(&[-1.5_f64, -9.0]),
            arr1(&[-0.5, -3.0]),
            arr1(&[0.5, 3.0]),
            arr1(&[1.5, 9.0]),
        ];
        let mut sampler =
            GenericHMC::new(AnisotropicGaussian2D, initial_positions, 0.3, 8).set_seed(42);

        sampler.run(0, 120);

        let mass: Vec<f64> = sampler.mass.inv().iter().map(|&inv| 1.0 / inv).collect();
        assert_eq!(mass.len(), 2);
        assert!(mass[0].is_finite() && mass[0] > 0.0);
        assert!(mass[1].is_finite() && mass[1] > 0.0);
        assert!(
            mass[1] > mass[0] * 4.0,
            "Expected warmup to learn a larger mass for the broader axis, got {:?}",
            mass
        );
    }
}
