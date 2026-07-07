//! Standing simulation-based-calibration (SBC) harness (issue #1891).
//!
//! The library's core value proposition is *calibrated* uncertainty. The
//! open #1869–#1878 bug cluster showed that miscalibration keeps being caught
//! one instance at a time by ad-hoc bug-hunts. This module is the reusable core
//! of a standing gate that converts that recurring genus into a permanent
//! invariant.
//!
//! The mechanism is Simulation-Based Calibration (Talts, Betancourt, Simpson,
//! Vehtari, Gelman, 2018, arXiv:1804.06788). For a *correctly* calibrated
//! posterior the following procedure produces ranks that are exactly uniform:
//!
//! 1. Draw a true parameter from the prior, `θ ~ π(θ)`.
//! 2. Simulate data from the likelihood, `y ~ p(y | θ)`.
//! 3. Fit and draw `L` posterior samples, `θ*_1..θ*_L ~ p(θ | y)`.
//! 4. Record the rank of the truth among the posterior samples,
//!    `r = #{ l : θ*_l < θ } ∈ {0, …, L}`.
//!
//! Over many replications, `r` must be uniform on `{0, …, L}`; any departure is
//! evidence of a miscalibrated posterior (over-/under-dispersed, biased, or
//! shape-wrong — the class coverage-at-three-levels misses). This module owns
//! (a) the generic draw→simulate→fit→rank loop over an [`SbcModel`], (b) the
//! rank primitive, and (c) a chi-square uniformity verdict with a principled
//! fixed false-positive rate.
//!
//! The first live consumer is the analytically-calibrated conjugate Gaussian
//! model ([`ConjugateGaussianModel`]), whose closed-form posterior makes SBC
//! ranks *provably* uniform — so the module's own tests are a real end-to-end
//! run (not a fabricated rank sequence), and a planted-miscalibration variant
//! (a deliberately over-confident posterior) is required to *fail* the gate,
//! which proves the rank + uniformity logic has teeth.

/// Posterior draws per SBC replication.
///
/// One less than a round power of ten so the reachable rank set `{0, …, L}` has
/// exactly `L + 1 = 100` outcomes, which partitions evenly into the decile bins
/// of the uniformity verdict (no ragged final bin biasing the chi-square).
pub const SBC_POSTERIOR_DRAWS_PER_REP: usize = 99;

/// Rank-histogram bin count for the uniformity verdict.
///
/// Deciles evenly partition the 100 reachable ranks and are the coarsest
/// symmetric partition that still resolves the two-tailed pile-up produced by an
/// over-dispersed or under-dispersed posterior.
pub const SBC_BINS: usize = 10;

/// Replications per SBC gate.
///
/// Chosen so each of the [`SBC_BINS`] bins has an expected count of
/// `SBC_REPLICATIONS / SBC_BINS = 100` under the uniform null — twenty times the
/// conventional five-per-cell minimum for the chi-square approximation, so the
/// planted-miscalibration control trips far above the critical value while the
/// correctly-calibrated reference sits far below it.
pub const SBC_REPLICATIONS: usize = 1000;

/// A posterior-sampling model auditable by [`run_sbc`].
///
/// The trait draws the prior, then — from a prior draw — simulates data and
/// returns samples from the *fitted* posterior for the same scalar parameter.
/// Bundling "simulate" and "fit" is deliberate: a miscalibrated fit is exactly
/// the defect SBC exists to catch, so the fit lives inside the object under test
/// rather than in the harness.
pub trait SbcModel {
    /// Draw one true parameter value from the prior `π(θ)`.
    fn draw_prior(&self, rng: &mut CalibrationRng) -> f64;

    /// Simulate `y ~ p(y | truth)`, fit, and return `n_draws` samples from the
    /// fitted posterior `p(θ | y)` for the scalar parameter.
    fn simulate_and_posterior_draws(
        &self,
        truth: f64,
        n_draws: usize,
        rng: &mut CalibrationRng,
    ) -> Vec<f64>;
}

/// Rank of `truth` among `posterior_draws`: the count of draws strictly below
/// it, an integer in `{0, …, posterior_draws.len()}`. For a continuous
/// posterior ties occur with probability zero, so the strict comparison is the
/// canonical SBC rank statistic.
pub fn sbc_rank(truth: f64, posterior_draws: &[f64]) -> usize {
    posterior_draws
        .iter()
        .filter(|&&draw| draw < truth)
        .count()
}

/// Run the full SBC loop and return one rank per replication.
///
/// The returned ranks each lie in `{0, …, posterior_draws_per_rep}`. A single
/// deterministic RNG stream threads prior draws, simulation, and posterior
/// sampling so the whole gate is reproducible from `seed` (the harness is a
/// replicate-null consumer and inherits the repo determinism requirement).
pub fn run_sbc<M: SbcModel>(
    model: &M,
    seed: u64,
    replications: usize,
    posterior_draws_per_rep: usize,
) -> Vec<usize> {
    let mut rng = CalibrationRng::new(seed);
    (0..replications)
        .map(|_| {
            let truth = model.draw_prior(&mut rng);
            let posterior =
                model.simulate_and_posterior_draws(truth, posterior_draws_per_rep, &mut rng);
            assert_eq!(posterior.len(), posterior_draws_per_rep);
            sbc_rank(truth, &posterior)
        })
        .collect()
}

/// Verdict of the chi-square uniformity test on an SBC rank histogram.
#[derive(Clone, Debug)]
pub struct SbcUniformityVerdict {
    /// Rank-histogram bin counts (length `bins`).
    pub counts: Vec<usize>,
    /// Pearson chi-square statistic against the uniform null.
    pub chi_square: f64,
    /// Fixed critical value at a 1% false-positive rate.
    pub critical_value: f64,
    /// `true` iff `chi_square <= critical_value`.
    pub passed: bool,
}

/// Bin SBC ranks into `bins` equal-width cells and test uniformity by Pearson
/// chi-square against a fixed critical value.
///
/// The critical value is the Laurent–Massart (2000) upper tail bound for a
/// chi-square with `bins - 1` degrees of freedom:
/// `k + 2·√(k·x) + 2·x` with `x = ln(100)`, which bounds
/// `P(χ²_k ≥ value) ≤ e^{-x} = 0.01`. This is a principled fixed 1%
/// false-positive rate for a standing gate, not a tuned threshold.
///
/// # Panics
/// Panics if `bins < 2`, or if any rank exceeds `posterior_draws_per_rep`
/// (which would mean the caller mixed histograms of different resolutions).
pub fn audit_sbc_uniformity(
    ranks: &[usize],
    posterior_draws_per_rep: usize,
    bins: usize,
) -> SbcUniformityVerdict {
    assert!(bins > 1, "uniformity test needs at least two bins");
    assert!(!ranks.is_empty(), "SBC produced no ranks");
    let reachable_ranks = posterior_draws_per_rep + 1;
    let mut counts = vec![0usize; bins];
    for &rank in ranks {
        assert!(
            rank <= posterior_draws_per_rep,
            "rank {rank} exceeds posterior draw count {posterior_draws_per_rep}"
        );
        // Equal-width binning of the reachable rank set {0, …, L} into `bins`
        // cells: bin = floor(rank · bins / (L + 1)), clamped to the last cell.
        let bin = (rank * bins / reachable_ranks).min(bins - 1);
        counts[bin] += 1;
    }

    let expected = ranks.len() as f64 / bins as f64;
    let chi_square = counts
        .iter()
        .map(|&count| {
            let residual = count as f64 - expected;
            residual * residual / expected
        })
        .sum::<f64>();

    let degrees = (bins - 1) as f64;
    let x = 100.0_f64.ln();
    let critical_value = degrees + 2.0 * (degrees * x).sqrt() + 2.0 * x;

    SbcUniformityVerdict {
        counts,
        chi_square,
        critical_value,
        passed: chi_square <= critical_value,
    }
}

/// Conjugate normal-normal location model: the first live SBC consumer.
///
/// Prior `θ ~ Normal(prior_mean, prior_sd²)`, likelihood
/// `y_i ~ Normal(θ, obs_sd²)` for `n_obs` observations with `obs_sd` known. The
/// posterior is available in closed form, so SBC ranks are provably uniform when
/// [`posterior_sd_scale`](Self::posterior_sd_scale) is `1.0`. Setting the scale
/// away from `1.0` plants a miscalibration (over-confident below `1.0`,
/// over-dispersed above) that SBC must detect.
#[derive(Clone, Copy, Debug)]
pub struct ConjugateGaussianModel {
    /// Prior mean `μ₀`.
    pub prior_mean: f64,
    /// Prior standard deviation `τ` (`> 0`).
    pub prior_sd: f64,
    /// Known observation standard deviation `σ` (`> 0`).
    pub obs_sd: f64,
    /// Number of simulated observations per replication.
    pub n_obs: usize,
    /// Multiplier on the true posterior standard deviation. `1.0` is the exact
    /// conjugate posterior; other values plant a deliberate miscalibration.
    pub posterior_sd_scale: f64,
}

impl ConjugateGaussianModel {
    /// The exact conjugate posterior `(mean, sd)` for the parameter given the
    /// simulated sufficient statistic `sum_y = Σ yᵢ`.
    fn posterior_mean_sd(&self, sum_y: f64) -> (f64, f64) {
        let prior_precision = self.prior_sd.powi(-2);
        let data_precision = self.n_obs as f64 * self.obs_sd.powi(-2);
        let posterior_precision = prior_precision + data_precision;
        let posterior_var = 1.0 / posterior_precision;
        let posterior_mean = posterior_var
            * (self.prior_mean * prior_precision + sum_y * self.obs_sd.powi(-2));
        (posterior_mean, posterior_var.sqrt())
    }
}

impl SbcModel for ConjugateGaussianModel {
    fn draw_prior(&self, rng: &mut CalibrationRng) -> f64 {
        self.prior_mean + self.prior_sd * rng.standard_normal()
    }

    fn simulate_and_posterior_draws(
        &self,
        truth: f64,
        n_draws: usize,
        rng: &mut CalibrationRng,
    ) -> Vec<f64> {
        let sum_y = (0..self.n_obs)
            .map(|_| truth + self.obs_sd * rng.standard_normal())
            .sum::<f64>();
        let (posterior_mean, posterior_sd) = self.posterior_mean_sd(sum_y);
        let sampling_sd = posterior_sd * self.posterior_sd_scale;
        (0..n_draws)
            .map(|_| posterior_mean + sampling_sd * rng.standard_normal())
            .collect()
    }
}

/// Small deterministic RNG for the calibration harness.
///
/// A PCG-family 64-bit linear-congruential engine with Box–Muller normals. The
/// multiplier/increment are the reference generator's definition; the harness
/// owns only the seed, so gates are reproducible across machines.
pub struct CalibrationRng {
    state: u64,
    spare_normal: Option<f64>,
}

impl CalibrationRng {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    /// A uniform draw on the open interval `(0, 1)`.
    pub fn uniform_open01(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = self.state >> 11;
        (bits as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    /// A standard-normal draw via the Box–Muller transform.
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform_open01().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform_open01();
        let first = radius * angle.cos();
        self.spare_normal = Some(radius * angle.sin());
        first
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CalibrationRng, ConjugateGaussianModel, SBC_BINS, SBC_POSTERIOR_DRAWS_PER_REP,
        SBC_REPLICATIONS, audit_sbc_uniformity, run_sbc, sbc_rank,
    };

    /// A fixed, reproducible seed for the calibrated reference run.
    const REFERENCE_SEED: u64 = 0x1891_5BC_C0DE;
    /// A distinct fixed seed for the planted-miscalibration control.
    const PLANTED_SEED: u64 = 0x1891_DEAD_5BC;

    fn conjugate_model(posterior_sd_scale: f64) -> ConjugateGaussianModel {
        ConjugateGaussianModel {
            prior_mean: 0.0,
            prior_sd: 2.0,
            obs_sd: 1.0,
            n_obs: 8,
            posterior_sd_scale,
        }
    }

    #[test]
    fn sbc_rank_counts_draws_strictly_below_truth() {
        // Direct primitive check: three of the five draws are below 0.5.
        let draws = [-1.0, 0.0, 0.25, 0.75, 2.0];
        assert_eq!(sbc_rank(0.5, &draws), 3);
        // Truth below every draw -> rank 0; above every draw -> rank == len.
        assert_eq!(sbc_rank(-5.0, &draws), 0);
        assert_eq!(sbc_rank(5.0, &draws), draws.len());
    }

    #[test]
    fn calibrated_conjugate_gaussian_passes_sbc_uniformity() {
        // Real end-to-end SBC: prior -> simulate -> closed-form posterior draws
        // -> rank -> uniformity. The exact conjugate posterior makes ranks
        // provably uniform, so this passes with a wide margin.
        let model = conjugate_model(1.0);
        let ranks = run_sbc(
            &model,
            REFERENCE_SEED,
            SBC_REPLICATIONS,
            SBC_POSTERIOR_DRAWS_PER_REP,
        );
        assert_eq!(ranks.len(), SBC_REPLICATIONS);
        let verdict = audit_sbc_uniformity(&ranks, SBC_POSTERIOR_DRAWS_PER_REP, SBC_BINS);
        assert!(
            verdict.passed,
            "correctly-calibrated conjugate Gaussian must pass SBC uniformity: \
             chi_square={:.3} > critical_value={:.3}, counts={:?}",
            verdict.chi_square, verdict.critical_value, verdict.counts
        );
    }

    #[test]
    fn overconfident_posterior_fails_sbc_uniformity() {
        // Teeth: the identical loop with a posterior standard deviation halved
        // (posterior_sd_scale = 0.5) is over-confident, so the true parameter
        // repeatedly lands in the posterior tails and ranks pile up at both
        // extremes. SBC must reject it. A bug that made the rank or uniformity
        // logic vacuous would let this through.
        let model = conjugate_model(0.5);
        let ranks = run_sbc(
            &model,
            PLANTED_SEED,
            SBC_REPLICATIONS,
            SBC_POSTERIOR_DRAWS_PER_REP,
        );
        let verdict = audit_sbc_uniformity(&ranks, SBC_POSTERIOR_DRAWS_PER_REP, SBC_BINS);
        assert!(
            !verdict.passed,
            "over-confident posterior must fail SBC uniformity but passed: \
             chi_square={:.3} <= critical_value={:.3}, counts={:?}",
            verdict.chi_square, verdict.critical_value, verdict.counts
        );
        // The failure signature is two-tailed pile-up, not a middle bulge:
        // the extreme bins each hold more than the uniform expectation.
        let expected = SBC_REPLICATIONS as f64 / SBC_BINS as f64;
        assert!(
            verdict.counts[0] as f64 > expected && verdict.counts[SBC_BINS - 1] as f64 > expected,
            "over-confident posterior should overfill the extreme rank bins: counts={:?}",
            verdict.counts
        );
    }

    #[test]
    fn overdispersed_posterior_fails_sbc_uniformity() {
        // The complementary teeth check: an over-dispersed posterior
        // (posterior_sd_scale = 2.0) piles ranks in the centre and must also be
        // rejected, so the gate is not one-sided.
        let model = conjugate_model(2.0);
        let ranks = run_sbc(
            &model,
            PLANTED_SEED,
            SBC_REPLICATIONS,
            SBC_POSTERIOR_DRAWS_PER_REP,
        );
        let verdict = audit_sbc_uniformity(&ranks, SBC_POSTERIOR_DRAWS_PER_REP, SBC_BINS);
        assert!(
            !verdict.passed,
            "over-dispersed posterior must fail SBC uniformity but passed: \
             chi_square={:.3} <= critical_value={:.3}, counts={:?}",
            verdict.chi_square, verdict.critical_value, verdict.counts
        );
    }

    #[test]
    fn rng_normals_are_reproducible_and_reasonable() {
        // Determinism guard: same seed reproduces the same stream.
        let mut a = CalibrationRng::new(0x51EED);
        let mut b = CalibrationRng::new(0x51EED);
        for _ in 0..64 {
            assert_eq!(a.standard_normal().to_bits(), b.standard_normal().to_bits());
        }
        // Sanity on the moments so a broken normal generator can't silently
        // calibrate everything to pass.
        let mut rng = CalibrationRng::new(7);
        let n = 20_000;
        let mean = (0..n).map(|_| rng.standard_normal()).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "standard-normal mean drifted: {mean}");
    }
}
