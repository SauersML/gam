//! Importance integration of one history's latent path at fixed coefficients
//! (#2961): the numerical check on the joint law, and the history-conditioned
//! particles forecasts average over. It is not the fitting route, which reads
//! the structured Laplace posterior (`posterior.rs`) inside the joint Laplace
//! evidence (`strength_fit.rs`).
//!
//! Latent coordinates are the missing genetic scores and the standardized OU
//! innovations of `transport.rs`. There the entry and OU densities are the
//! standard normal law and the transport Jacobian cancels, so the target is
//!
//! ```text
//! log p(y, g, e | θ) = path_density(θ, h, (g, x(g, e)), m, dynamics = false) + log N(e; 0, I),
//! ```
//!
//! whose integral is the history's likelihood with its observed genetic scores.
//! `Mode::log_marginal` is the Laplace approximation of the same integral.
//!
//! The proposal is the equal Gaussian/t₃ mixture of `coefficient_integral.rs`
//! about the Laplace mode ẑ with covariance Q⁻¹. A structured draw is affine in
//! its standard normals, so `draw(z) − draw(0)` is an N(0, Q⁻¹) displacement,
//! and the t₃ component rescales it. Its distance in the metric Q comes from
//! the Gaussian component's normalized density.
//!
//! A bank is data with a reported resolution: the estimated log integral, its
//! delta-method standard error and the effective sample size. Each refinement
//! doubles the bank from the caller's random stream, and the caller stops once
//! its own quantity's error no longer dominates. No sample count is set here.
use super::coefficient_integral::{mixture_log_density, mixture_scale};
use super::law::{JointHistory, JointLikelihood, invalid, numerical};
use super::posterior::Mode;
use super::precision::StateLaw;
use crate::EventHistoryError;
use crate::chain::log_sum_exp;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// One draw in latent coordinates with its complete densities.
pub(super) struct LatentParticle {
    pub(super) genes: Array1<f64>,
    pub(super) innovations: Array2<f64>,
    /// `log p(y, g, e | θ)`.
    pub(super) log_target: f64,
    /// `log q(g, e)` under the mixture.
    pub(super) log_proposal: f64,
}

/// What a bank reports as it stands. These are estimates, not bounds.
pub(super) struct LatentIntegral {
    pub(super) log_integral: f64,
    /// Delta-method standard error of `log_integral`, a Monte Carlo spread.
    pub(super) log_standard_error: f64,
    pub(super) effective_samples: f64,
    pub(super) largest_normalized_weight: f64,
    /// The bank size the spread came from.
    pub(super) samples: usize,
}

/// The history-conditioned particles at the history's last node s, in the
/// layout the reference kernel's population reads.
pub(super) struct ConditionedParticles {
    pub(super) integral: LatentIntegral,
    /// Particles × K pre-softplus signature states at s, after the summed jumps
    /// of every mark tied at s.
    pub(super) states: Vec<Vec<f64>>,
    /// Particles × G full genomes in specification order: the observed scores
    /// and the particle's sampled missing scores.
    pub(super) genomes: Vec<Vec<f64>>,
    /// Normalized log importance weights, kept when they underflow.
    pub(super) log_weights: Vec<f64>,
}

/// Latent particles of one history at fixed coefficients and reference moments.
pub(super) struct LatentBank<'m> {
    model: &'m JointLikelihood,
    history: &'m JointHistory,
    theta: Vec<f64>,
    reference: Vec<f64>,
    mode: Mode,
    /// `draw(0)`, the origin every displacement is measured from.
    origin: (Array1<f64>, Array2<f64>),
    particles: Vec<LatentParticle>,
}

impl JointLikelihood {
    /// A bank about the history's Laplace mode, holding no particles yet.
    pub(super) fn latent_bank<'m>(
        &'m self,
        theta: &[f64],
        h: &'m JointHistory,
        reference: &[f64],
    ) -> Result<LatentBank<'m>, EventHistoryError> {
        let mode = self.mode(theta, h, reference, None)?;
        let (nodes, k) = (mode.law.nodes(), mode.law.signatures());
        let standard = Array1::<f64>::zeros(mode.law.genes() + 2 * nodes * k);
        let origin =
            mode.factor
                .draw(&mode.law, Array2::<f64>::zeros((nodes, k)).view(), standard.view())?;
        Ok(LatentBank {
            model: self,
            history: h,
            theta: theta.to_vec(),
            reference: reference.to_vec(),
            mode,
            origin,
            particles: Vec::new(),
        })
    }

    /// `log p(y, g, e | θ)` at one latent point.
    fn latent_target(
        &self,
        theta: &[f64],
        h: &JointHistory,
        law: &StateLaw,
        genes: &Array1<f64>,
        innovations: &Array2<f64>,
        reference: &[f64],
    ) -> Result<f64, EventHistoryError> {
        let states = law.transport(genes.view(), innovations.view(), true);
        let path: Vec<f64> = genes.iter().chain(states.iter()).copied().collect();
        let innovation_law = -0.5 * innovations.iter().map(|e| e * e).sum::<f64>()
            - 0.5 * innovations.len() as f64 * (2.0 * std::f64::consts::PI).ln();
        Ok(self.path_density(theta, h, &path, reference, false)? + innovation_law)
    }
}

impl LatentBank<'_> {
    /// Double the bank, starting from the two draws a standard error needs. A
    /// draw whose density is not representable is an error, never replaced,
    /// which would truncate the proposal.
    pub(super) fn refine<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<(), EventHistoryError> {
        let law = &self.mode.law;
        let factor = &self.mode.factor;
        let (nodes, k, g) = (law.nodes(), law.signatures(), law.genes());
        let dimension = g + nodes * k;
        let half_log_determinant = 0.5 * factor.log_determinant;
        let centre = &self.mode.point;
        let target = 2 * self.particles.len().max(1);
        while self.particles.len() < target {
            let standard: Array1<f64> = std::iter::repeat_with(|| StandardNormal.sample(&mut *rng))
                .take(g + 2 * nodes * k)
                .collect();
            let (genes, innovations) =
                factor.draw(law, Array2::<f64>::zeros((nodes, k)).view(), standard.view())?;
            let scale = mixture_scale(rng);
            let genes = &centre.genes + &((&genes - &self.origin.0) * scale);
            let innovations = &centre.innovations + &((&innovations - &self.origin.1) * scale);
            let gaussian = factor.log_density(
                law,
                (centre.genes.view(), centre.innovations.view()),
                (genes.view(), innovations.view()),
            )?;
            let squared_distance = 2.0
                * (half_log_determinant
                    - 0.5 * dimension as f64 * (2.0 * std::f64::consts::PI).ln()
                    - gaussian);
            let (log_proposal, _) =
                mixture_log_density(dimension, half_log_determinant, squared_distance.max(0.0).sqrt())?;
            let log_target = self.model.latent_target(
                &self.theta,
                self.history,
                law,
                &genes,
                &innovations,
                &self.reference,
            )?;
            self.particles.push(LatentParticle {
                genes,
                innovations,
                log_target,
                log_proposal,
            });
        }
        Ok(())
    }

    /// The log integral with its standard error and effective sample size.
    /// With self-normalized weights ŵ, `Var(log Î) ≈ n/(n − 1) Σ (ŵ − 1/n)²`.
    pub(super) fn estimate(&self) -> Result<LatentIntegral, EventHistoryError> {
        let count = self.particles.len();
        if count < 2 {
            return Err(invalid(
                "a latent integral needs at least two particles: refine the bank first",
            ));
        }
        let log_weights: Vec<f64> = self
            .particles
            .iter()
            .map(|p| p.log_target - p.log_proposal)
            .collect();
        let total = log_sum_exp(&log_weights);
        let weights: Vec<f64> = log_weights.iter().map(|w| (w - total).exp()).collect();
        let uniform = (count as f64).recip();
        let spread: f64 = weights.iter().map(|w| (w - uniform).powi(2)).sum();
        let log_standard_error = (spread * count as f64 / (count - 1) as f64).sqrt();
        let effective_samples = weights.iter().map(|w| w * w).sum::<f64>().recip();
        let largest_normalized_weight = weights.iter().fold(0.0_f64, |a, &b| a.max(b));
        let log_integral = total - (count as f64).ln();
        if !log_integral.is_finite() || !log_standard_error.is_finite() {
            return Err(numerical("latent importance integral is not representable"));
        }
        Ok(LatentIntegral {
            log_integral,
            log_standard_error,
            effective_samples,
            largest_normalized_weight,
            samples: count,
        })
    }

    /// The particles at the history's last node s with their normalized log
    /// weights, for a forecast started from s. A node's transported state is
    /// its pre-jump state, so every mark tied at s adds its jump here.
    pub(super) fn conditioned_particles(&self) -> Result<ConditionedParticles, EventHistoryError> {
        let integral = self.estimate()?;
        let law = &self.mode.law;
        let last = law
            .nodes()
            .checked_sub(1)
            .ok_or_else(|| invalid("a history without nodes has no state at s"))?;
        let fired = &self.history.events[last];
        let log_weights: Vec<f64> = self
            .particles
            .iter()
            .map(|p| p.log_target - p.log_proposal)
            .collect();
        let total = log_sum_exp(&log_weights);
        let mut states = Vec::with_capacity(self.particles.len());
        let mut genomes = Vec::with_capacity(self.particles.len());
        for particle in &self.particles {
            let path = law.transport(particle.genes.view(), particle.innovations.view(), true);
            states.push(
                (0..law.signatures())
                    .map(|axis| path[[last, axis]] + self.model.jump(&self.theta, fired, axis))
                    .collect(),
            );
            let mut missing = particle.genes.iter().copied();
            let genome: Option<Vec<f64>> = self
                .history
                .genetics
                .iter()
                .map(|gene| gene.or_else(|| missing.next()))
                .collect();
            genomes.push(genome.ok_or_else(|| {
                numerical("particle genes do not fill the history's missing genetic scores")
            })?);
        }
        Ok(ConditionedParticles {
            integral,
            states,
            genomes,
            log_weights: log_weights.iter().map(|w| w - total).collect(),
        })
    }

    /// The Laplace integral of the same target.
    pub(super) fn laplace_log_integral(&self) -> f64 {
        self.mode.log_marginal
    }

    pub(super) fn particles(&self) -> &[LatentParticle] {
        &self.particles
    }
}
