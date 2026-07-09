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

// ---------------------------------------------------------------------------
// Coverage-sweep mode (issue #1891, audit mode 1).
//
// SBC rank uniformity (above) audits *posterior* surfaces. The complementary
// mode audits *interval* surfaces — Wald/delta intervals, credible bands,
// predictive intervals, and (via a 0/1 "reject" indicator) test-size curves.
// The primitive is empirical coverage: over many planted-truth replications,
// what fraction of the nominal-`c` intervals actually contain the truth?
//
// The tolerance is NOT a magic bound. Under correct calibration the hit count
// is `Binomial(R, c)`, so the acceptance region is the (1 − α) binomial
// confidence interval for the coverage proportion — the Wilson score interval,
// the standard interval-for-a-proportion, evaluated at the same fixed 1%
// false-positive rate the SBC verdict uses. A surface is anti-conservative (the
// #1870/#1871/#1878 failure signature) only when the nominal level sits ABOVE
// the whole CI; conservative when it sits below (reported with a named slack,
// per the issue: anti-conservative gates the build, conservative reports).
// ---------------------------------------------------------------------------

/// Fixed false-positive rate for the coverage verdict — the same 1% the SBC
/// uniformity gate uses, so both audit modes share one calibrated error budget
/// rather than two independently tuned thresholds.
pub const COVERAGE_FALSE_POSITIVE_RATE: f64 = 0.01;

/// Replications per coverage gate.
///
/// Sized so the Wilson half-width at the tightest audited level (`c = 0.95`) is
/// a few percent — narrow enough to resolve the historical failures (0.157 and
/// 0.731 observed vs 0.95 nominal in #1870/#1871) with wide margin, while a
/// merely-1-point miscalibration is not over-resolved into a spurious gate.
pub const COVERAGE_REPLICATIONS: usize = 2000;

/// The three nominal levels every interval surface is audited at, matching the
/// issue's 80/90/95 sweep. Test-size surfaces audit the complementary
/// `α ∈ {0.2, 0.1, 0.05}` by registering the rejection indicator as the "miss".
pub const COVERAGE_NOMINAL_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];

/// Rational-approximation inverse standard-normal CDF (Acklam's algorithm).
///
/// Accurate to ~1.15e-9 in absolute error over `p ∈ (0, 1)` — far tighter than
/// any coverage tolerance here — so a consumer can build an exact-level interval
/// `μ ± Φ⁻¹((1+c)/2)·σ` without hardcoding per-level z magic numbers. Not a
/// tuned bound: a fixed published approximation to a fixed function.
pub fn standard_normal_quantile(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "quantile argument must be in (0, 1)");
    // Coefficients (Acklam 2003).
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    // Break-points of the central region.
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Whether an interval surface's empirical coverage is statistically consistent
/// with, above, or below its nominal level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverageClass {
    /// Nominal level lies inside the (1 − α) Wilson CI — calibrated.
    Calibrated,
    /// Nominal level lies ABOVE the whole CI: the surface under-covers. This is
    /// the anti-conservative failure that gates the build.
    AntiConservative,
    /// Nominal level lies BELOW the whole CI: the surface over-covers. Reported
    /// with a named slack, not gated.
    Conservative,
}

/// Verdict of the empirical-coverage audit for one nominal level.
#[derive(Clone, Debug)]
pub struct CoverageVerdict {
    /// Nominal coverage level `c` the intervals were built at.
    pub nominal: f64,
    /// Replications run.
    pub replications: usize,
    /// Intervals that contained the truth.
    pub hits: usize,
    /// Empirical coverage `hits / replications`.
    pub empirical: f64,
    /// Lower endpoint of the (1 − α) Wilson score CI for the true coverage.
    pub ci_lo: f64,
    /// Upper endpoint of the (1 − α) Wilson score CI for the true coverage.
    pub ci_hi: f64,
    /// Classification of `nominal` against the CI.
    pub class: CoverageClass,
    /// `true` unless the surface is anti-conservative. Conservative and
    /// calibrated both pass (over-coverage is safe); only under-coverage gates.
    pub passed: bool,
}

impl CoverageVerdict {
    /// Signed slack of the nominal level outside the CI: positive when
    /// conservative (nominal below `ci_lo`), negative when anti-conservative
    /// (nominal above `ci_hi`), zero when calibrated. Lets a conservative report
    /// name its margin.
    pub fn slack(&self) -> f64 {
        match self.class {
            CoverageClass::Conservative => self.ci_lo - self.nominal,
            CoverageClass::AntiConservative => self.ci_hi - self.nominal,
            CoverageClass::Calibrated => 0.0,
        }
    }
}

/// Classify an observed hit count against its nominal level using the Wilson
/// score interval at [`COVERAGE_FALSE_POSITIVE_RATE`].
///
/// The Wilson interval is the standard confidence interval for a binomial
/// proportion (better small-sample behaviour than the Wald interval and never
/// escaping `[0, 1]`), so the tolerance is derived from the binomial law of the
/// replicate count — not a hand-picked coverage band.
pub fn audit_coverage(hits: usize, replications: usize, nominal: f64) -> CoverageVerdict {
    assert!(replications > 0, "coverage audit needs at least one replication");
    assert!(
        nominal > 0.0 && nominal < 1.0,
        "nominal coverage must lie in (0, 1)"
    );
    assert!(hits <= replications, "hit count exceeds replication count");
    let n = replications as f64;
    let p_hat = hits as f64 / n;
    // Two-sided z at the fixed FPR: Φ⁻¹(1 − α/2).
    let z = standard_normal_quantile(1.0 - COVERAGE_FALSE_POSITIVE_RATE / 2.0);
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p_hat + z2 / (2.0 * n)) / denom;
    let half = z * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    let ci_lo = (center - half).max(0.0);
    let ci_hi = (center + half).min(1.0);
    let class = if nominal > ci_hi {
        CoverageClass::AntiConservative
    } else if nominal < ci_lo {
        CoverageClass::Conservative
    } else {
        CoverageClass::Calibrated
    };
    CoverageVerdict {
        nominal,
        replications,
        hits,
        empirical: p_hat,
        ci_lo,
        ci_hi,
        class,
        passed: class != CoverageClass::AntiConservative,
    }
}

/// An interval-emitting surface auditable by [`run_coverage`].
///
/// Draws a truth from the prior, then simulates data, fits, and returns the
/// `(lower, upper)` interval the surface reports at nominal level `c`. As with
/// [`SbcModel`], simulate+fit+extract live inside the object under test so a
/// miscalibrated fit is exercised, not bypassed.
pub trait CoverageModel {
    /// Draw one true parameter value from the prior `π(θ)`.
    fn draw_prior(&self, rng: &mut CalibrationRng) -> f64;

    /// Simulate `y ~ p(y | truth)`, fit, and return the surface's `(lower,
    /// upper)` interval at nominal coverage `level ∈ (0, 1)`.
    fn simulate_and_interval(
        &self,
        truth: f64,
        level: f64,
        rng: &mut CalibrationRng,
    ) -> (f64, f64);
}

/// Run the coverage loop at one nominal level and return the hit count (the
/// number of replications whose interval contained the truth). A single
/// deterministic RNG stream threads every replication so the gate reproduces
/// from `seed`.
pub fn run_coverage<M: CoverageModel>(
    model: &M,
    seed: u64,
    replications: usize,
    level: f64,
) -> usize {
    let mut rng = CalibrationRng::new(seed);
    (0..replications)
        .filter(|_| {
            let truth = model.draw_prior(&mut rng);
            let (lo, hi) = model.simulate_and_interval(truth, level, &mut rng);
            lo <= truth && truth <= hi
        })
        .count()
}

/// Conjugate normal-normal location model: the first live SBC consumer.
///
/// Prior `θ ~ Normal(prior_mean, prior_sd²)`, likelihood
/// `y_i ~ Normal(θ, obs_sd²)` for `n_obs` observations with `obs_sd` known. The
/// posterior is available in closed form, so SBC ranks are provably uniform when
/// [`posterior_sd_scale`](Self::posterior_sd_scale) is `1.0`. Setting the scale
/// away from `1.0` plants a miscalibration (over-confident below `1.0`,
/// over-dispersed above) that SBC must detect. The same closed form gives an
/// exact credible interval, so the model is also the first [`CoverageModel`]
/// consumer: at scale `1.0` its intervals cover at nominal, and a narrowed scale
/// under-covers (the anti-conservative signature the coverage gate must catch).
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

impl CoverageModel for ConjugateGaussianModel {
    fn draw_prior(&self, rng: &mut CalibrationRng) -> f64 {
        self.prior_mean + self.prior_sd * rng.standard_normal()
    }

    fn simulate_and_interval(
        &self,
        truth: f64,
        level: f64,
        rng: &mut CalibrationRng,
    ) -> (f64, f64) {
        let sum_y = (0..self.n_obs)
            .map(|_| truth + self.obs_sd * rng.standard_normal())
            .sum::<f64>();
        let (posterior_mean, posterior_sd) = self.posterior_mean_sd(sum_y);
        // Exact equal-tailed credible interval at `level`, scaled by the planted
        // miscalibration factor: at scale 1.0 it is the true posterior interval
        // (covers at nominal); a narrowed scale under-covers.
        let half_width =
            standard_normal_quantile(0.5 + level / 2.0) * posterior_sd * self.posterior_sd_scale;
        (posterior_mean - half_width, posterior_mean + half_width)
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

// ---------------------------------------------------------------------------
// UQ-surface registry + completeness contract (issue #1891).
//
// The two primitives above — SBC rank uniformity and the coverage sweep — are
// the audit *engine*. The registry is the *contract*: every uncertainty-bearing
// surface the library exposes is enumerated here, tagged with the audit mode
// that gates it and the gate test that exercises it. A public result-payload
// field carrying an SE, an interval bound, a p-value, or a certificate that maps
// to NO registered target fails the completeness lint
// ([`assert_registry_covers_fields`]) — the mechanism that stops the next
// recycled-SE (#1875) from shipping unaudited.
//
// The registry TYPES live here (generic, dependency-light). The concrete
// registry rows and the exhaustive payload field-walk live in the calibration
// test suite (`tests/quality/calibration/`), which is the only layer that can
// name the upper-crate payload structs (`PredictUncertaintyResult`, the ALO /
// conformal / model-comparison / ρ-posterior results).
// ---------------------------------------------------------------------------

/// Type-I error rates the test-size audit sweeps, matching the issue's
/// `α ∈ {0.01, 0.05, 0.1}`. A test surface is anti-conservative when its
/// empirical size at `α` exceeds `α` beyond MC error — audited as coverage of
/// the *non-rejection* event at nominal `1 − α`, so the shared Wilson verdict
/// applies unchanged (an over-sized test under-covers non-rejection).
pub const TEST_SIZE_ALPHAS: [f64; 3] = [0.01, 0.05, 0.10];

/// The statistical kind of an uncertainty surface. Selects the audit mode and
/// documents the object under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    /// Wald / delta-method interval for a scalar or coefficient estimate.
    WaldDeltaInterval,
    /// Bayesian credible band for a mean / smooth function. `smoothing_corrected`
    /// distinguishes the conditional `Vp` band (`false`) from the smoothing-
    /// corrected `Vc` band (`true`) — the two surfaces the #1871 INLA comparison
    /// separates.
    CredibleBand { smoothing_corrected: bool },
    /// Approximate-leave-one-out / LOO predictive standard error (#1869).
    AloStandardError,
    /// Split / full conformal predictive interval.
    ConformalInterval,
    /// Predictive (observation) interval for a named response family.
    PredictiveInterval,
    /// Frequentist test p-value; audited as a type-I size curve under a
    /// simulated null (#1872 post-selection LR, #1873 Bartlett/Lawley).
    TestPValue,
    /// Posterior-sample surface audited by SBC rank uniformity (NUTS,
    /// ρ-posterior #1810, Polya-Gamma).
    PosteriorSample,
}

/// Which audit primitive gates a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditMode {
    /// Empirical coverage vs nominal at [`COVERAGE_NOMINAL_LEVELS`]
    /// (`run_coverage` + `audit_coverage`); anti-conservative gates.
    CoverageSweep,
    /// Type-I size curve at [`TEST_SIZE_ALPHAS`] under a simulated null; an
    /// empirical size above `α` beyond MC error gates.
    TestSizeCurve,
    /// SBC rank-uniformity histogram (`run_sbc` + `audit_sbc_uniformity`).
    SbcRankUniformity,
}

/// One registered uncertainty surface: a row of the #1891 completeness contract.
#[derive(Clone, Debug)]
pub struct CalibrationTarget {
    /// Stable identifier, referenced by the completeness lint's field map.
    pub name: &'static str,
    /// Statistical kind of the surface.
    pub kind: SurfaceKind,
    /// Audit primitive that gates this target.
    pub mode: AuditMode,
    /// Cluster issues whose regression this target's gate would catch.
    pub guards: &'static [u32],
    /// The gate test (`binary::module` or file stem) exercising it end to end.
    pub audited_by: &'static str,
}

/// How a public result-payload field relates to the UQ registry, assigned by
/// the completeness lint's exhaustive field walk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldDisposition {
    /// A point estimate or non-uncertainty metadata field — no coverage claim,
    /// so no registered target is required.
    NotUncertainty,
    /// An uncertainty-bearing field (SE / interval bound / p-value / certificate)
    /// whose calibration is audited by the registered target of this name.
    AuditedBy(&'static str),
}

/// One classified payload field produced by the exhaustive field-walk.
#[derive(Clone, Debug)]
pub struct FieldAudit {
    /// The payload field's Rust identifier.
    pub field: &'static str,
    /// Its relationship to the registry.
    pub disposition: FieldDisposition,
}

impl FieldAudit {
    /// A point-estimate / metadata field carrying no coverage claim.
    pub const fn point(field: &'static str) -> Self {
        Self { field, disposition: FieldDisposition::NotUncertainty }
    }

    /// An uncertainty field audited by the registered target `target`.
    pub const fn audited(field: &'static str, target: &'static str) -> Self {
        Self { field, disposition: FieldDisposition::AuditedBy(target) }
    }
}

/// The completeness contract (#1891): every payload field classified
/// `AuditedBy(t)` must name a registered target.
///
/// Panics (fails the lint) if any field points at an unregistered target — that
/// is the "an uncertainty surface not in the registry fails" invariant. The
/// complementary EXHAUSTIVENESS half — that no uncertainty field was omitted
/// from `fields` — is enforced at the call site by an exhaustive struct
/// destructure with no `..` rest pattern, so adding a payload field forces a
/// classification here rather than silently shipping unaudited.
pub fn assert_registry_covers_fields(fields: &[FieldAudit], registry: &[CalibrationTarget]) {
    let registered: std::collections::BTreeSet<&str> = registry.iter().map(|t| t.name).collect();
    let mut orphans = Vec::new();
    for f in fields {
        if let FieldDisposition::AuditedBy(target) = f.disposition {
            if !registered.contains(target) {
                orphans.push(format!(
                    "payload field `{}` claims uncertainty audited by `{target}`, which is \
                     NOT in the UQ-surface registry — register a CalibrationTarget for it",
                    f.field
                ));
            }
        }
    }
    assert!(
        orphans.is_empty(),
        "UQ completeness lint failed — unregistered uncertainty surfaces:\n{}",
        orphans.join("\n")
    );
}

/// Assert every registered target is exercised by a named gate and, for the
/// posterior kinds, routed to the SBC audit mode (a `PosteriorSample` audited by
/// a coverage sweep would silently miss the shape miscalibration SBC exists to
/// catch). This is the registry's internal consistency check, complementary to
/// the field-coverage lint.
pub fn assert_registry_well_formed(registry: &[CalibrationTarget]) {
    let mut problems = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for t in registry {
        if !seen.insert(t.name) {
            problems.push(format!("duplicate registry name `{}`", t.name));
        }
        if t.audited_by.is_empty() {
            problems.push(format!("target `{}` names no gate (`audited_by` empty)", t.name));
        }
        let mode_ok = match t.kind {
            // SBC rank uniformity is the ideal posterior audit (it sees shape
            // miscalibration a 3-level coverage sweep misses), but a posterior
            // surface consumed only through a derived band may be legitimately
            // gated by that band's coverage until a bespoke SBC gate exists —
            // both are accepted, a test-size or interval mode is not.
            SurfaceKind::PosteriorSample => {
                t.mode == AuditMode::SbcRankUniformity || t.mode == AuditMode::CoverageSweep
            }
            SurfaceKind::TestPValue => t.mode == AuditMode::TestSizeCurve,
            _ => t.mode == AuditMode::CoverageSweep,
        };
        if !mode_ok {
            problems.push(format!(
                "target `{}` kind {:?} routed to the wrong audit mode {:?}",
                t.name, t.kind, t.mode
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "UQ registry is malformed:\n{}",
        problems.join("\n")
    );
}

#[cfg(test)]
mod tests {
    use super::{
        AuditMode, CalibrationTarget, FieldAudit, FieldDisposition, SurfaceKind,
        assert_registry_covers_fields, assert_registry_well_formed,
    };
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
    fn normal_quantile_matches_known_values_and_is_symmetric() {
        use super::standard_normal_quantile;
        // Reference two-sided z at the classic levels (accurate to the
        // approximation's ~1e-9), and the symmetry Φ⁻¹(1−p) = −Φ⁻¹(p).
        for (p, z) in [(0.975_f64, 1.959_963_985), (0.95, 1.644_853_627), (0.9, 1.281_551_566)] {
            assert!(
                (standard_normal_quantile(p) - z).abs() < 1e-6,
                "Φ⁻¹({p}) = {} != {z}",
                standard_normal_quantile(p)
            );
            assert!(
                (standard_normal_quantile(1.0 - p) + z).abs() < 1e-6,
                "quantile not antisymmetric at p={p}"
            );
        }
        assert!(standard_normal_quantile(0.5).abs() < 1e-9, "median must be 0");
    }

    #[test]
    fn wilson_verdict_classifies_nominal_against_the_ci() {
        use super::{CoverageClass, audit_coverage};
        // Calibrated: 950/1000 hits at nominal 0.95 — nominal sits inside the CI.
        let v = audit_coverage(950, 1000, 0.95);
        assert_eq!(v.class, CoverageClass::Calibrated);
        assert!(v.passed && v.ci_lo < 0.95 && 0.95 < v.ci_hi);
        // Anti-conservative: 731/1000 (the #1871 signature) vs nominal 0.95 —
        // nominal far above the CI, gate must fail.
        let under = audit_coverage(731, 1000, 0.95);
        assert_eq!(under.class, CoverageClass::AntiConservative);
        assert!(!under.passed && 0.95 > under.ci_hi && under.slack() < 0.0);
        // Conservative: 995/1000 at nominal 0.90 — over-covers, reports positive
        // slack but does NOT gate.
        let over = audit_coverage(995, 1000, 0.90);
        assert_eq!(over.class, CoverageClass::Conservative);
        assert!(over.passed && over.slack() > 0.0);
    }

    #[test]
    fn calibrated_conjugate_gaussian_passes_coverage_sweep() {
        use super::{
            COVERAGE_NOMINAL_LEVELS, COVERAGE_REPLICATIONS, audit_coverage, run_coverage,
        };
        // Real end-to-end coverage: exact closed-form credible intervals cover at
        // nominal, so every level in the sweep must pass.
        let model = conjugate_model(1.0);
        for &level in &COVERAGE_NOMINAL_LEVELS {
            let hits = run_coverage(&model, REFERENCE_SEED, COVERAGE_REPLICATIONS, level);
            let v = audit_coverage(hits, COVERAGE_REPLICATIONS, level);
            assert!(
                v.passed,
                "calibrated Gaussian must pass coverage at {level}: empirical={:.4} \
                 CI=[{:.4},{:.4}] class={:?}",
                v.empirical, v.ci_lo, v.ci_hi, v.class
            );
        }
    }

    #[test]
    fn narrowed_interval_fails_coverage_sweep() {
        use super::{CoverageClass, COVERAGE_REPLICATIONS, audit_coverage, run_coverage};
        // Teeth: intervals narrowed to half width (posterior_sd_scale = 0.5)
        // under-cover, so the gate must classify anti-conservative and fail. A
        // vacuous coverage check would let this pass.
        let model = conjugate_model(0.5);
        let level = 0.95;
        let hits = run_coverage(&model, PLANTED_SEED, COVERAGE_REPLICATIONS, level);
        let v = audit_coverage(hits, COVERAGE_REPLICATIONS, level);
        assert_eq!(
            v.class,
            CoverageClass::AntiConservative,
            "narrowed interval must be flagged anti-conservative: empirical={:.4} CI=[{:.4},{:.4}]",
            v.empirical, v.ci_lo, v.ci_hi
        );
        assert!(!v.passed, "narrowed interval must fail the coverage gate");
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

    fn tiny_registry() -> Vec<CalibrationTarget> {
        vec![
            CalibrationTarget {
                name: "mean_credible_band",
                kind: SurfaceKind::CredibleBand { smoothing_corrected: false },
                mode: AuditMode::CoverageSweep,
                guards: &[1870, 1871],
                audited_by: "sbc_gaussian_smooth_band_coverage",
            },
            CalibrationTarget {
                name: "rho_posterior_certificate",
                kind: SurfaceKind::PosteriorSample,
                mode: AuditMode::SbcRankUniformity,
                guards: &[1810],
                audited_by: "calibration::rho_posterior_sbc",
            },
        ]
    }

    #[test]
    fn completeness_lint_accepts_fully_mapped_payload_and_rejects_an_orphan() {
        let registry = tiny_registry();
        // A payload whose only uncertainty field maps to a registered target,
        // plus a point estimate that needs no target — must pass.
        let ok = [
            FieldAudit::point("mean"),
            FieldAudit::audited("mean_lower", "mean_credible_band"),
        ];
        assert_registry_covers_fields(&ok, &registry);

        // Introduce a field auditing an UNREGISTERED surface — the exact
        // "recycled SE ships unaudited" failure the lint exists to block.
        let orphaned = [FieldAudit::audited("observation_lower", "predictive_interval_gaussian")];
        let caught = std::panic::catch_unwind(|| {
            assert_registry_covers_fields(&orphaned, &tiny_registry());
        });
        assert!(
            caught.is_err(),
            "completeness lint must reject a payload field pointing at an unregistered target"
        );
    }

    #[test]
    fn well_formed_check_flags_wrong_audit_mode() {
        assert_registry_well_formed(&tiny_registry());
        // A frequentist test p-value routed to a coverage sweep instead of the
        // type-I size curve is a mis-audit (it would never check the size under
        // the null the #1872/#1873 defects corrupt) — must be flagged.
        let mis = vec![CalibrationTarget {
            name: "lr_pvalue",
            kind: SurfaceKind::TestPValue,
            mode: AuditMode::CoverageSweep,
            guards: &[1872],
            audited_by: "somewhere",
        }];
        let caught = std::panic::catch_unwind(|| assert_registry_well_formed(&mis));
        assert!(caught.is_err(), "test p-value on a coverage sweep must be flagged");
    }

    #[test]
    fn field_disposition_constructors_tag_correctly() {
        assert_eq!(FieldAudit::point("eta").disposition, FieldDisposition::NotUncertainty);
        assert_eq!(
            FieldAudit::audited("eta_lower", "band").disposition,
            FieldDisposition::AuditedBy("band")
        );
    }
}
