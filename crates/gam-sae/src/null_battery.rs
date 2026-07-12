//! Standing null battery and spike-in calibration for topology/nerve claims.
//!
//! A Betti, nerve, or conditionality statistic is only interpretable together
//! with negative controls that preserve mundane structure while destroying the
//! claimed one. This module supplies the reusable harness: generate standing
//! nulls, run an arbitrary scalar audit on observed and null data, and attach a
//! spike-in power curve for a manufactured circle inside real residual noise.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Maximum row count of one covariance-exact Hadamard block.
///
/// Each active row-block/column-band tile owns one float64 transform workspace
/// with at most
/// `COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS * p` scalars.
pub const COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS: usize = 1_024;

/// Hard cap on concurrently transformed covariance-exact Hadamard tiles.
///
/// Tall inputs use independent row blocks; inputs with fewer row blocks than
/// workers split every block into deterministic column bands as well. Actual
/// concurrency never exceeds this cap or the active Rayon pool's thread count,
/// and the bands partition `p`, so transform memory remains bounded both by
/// `8 * 1024 * p * sizeof(f64)` and by the 128-MiB active-workspace budget,
/// independently of the corpus row count. Binary tail blocks reuse the same
/// workspaces in later waves.
pub const COVARIANCE_EXACT_HADAMARD_MAX_PARALLEL_WORKSPACES: usize = 8;

/// Maximum aggregate float64 workspace actively transformed in parallel.
///
/// An exclusive, internally warmed MSI sweep across `32768×1024`,
/// `8192×4096`, and `4096×7168` found the stable throughput knee at eight
/// 8-MiB tiles or four 28–32-MiB tiles. More active state increased RSS and
/// tail latency without increasing useful bandwidth. Column banding makes this
/// a hard portable memory bound rather than a shape-specific dispatch table.
pub const COVARIANCE_EXACT_HADAMARD_PARALLEL_WORKSPACE_BUDGET_BYTES: usize = 128 * 1024 * 1024;

/// Target upper bound for one cache-local Hadamard tile.
pub const COVARIANCE_EXACT_HADAMARD_TARGET_TILE_BYTES: usize = 32 * 1024 * 1024;

const HADAMARD_PERMUTATION_SEED_DOMAIN: u64 = 0x4841_4441_5045_524D;
const HADAMARD_SIGN_SEED_DOMAIN: u64 = 0x4841_4441_5349_474E;
const PER_DIMENSION_SHUFFLE_SEED_DOMAIN: u64 = 0x5045_5244_494D_5348;

/// Direction of the claim statistic under the null.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tail {
    Larger,
    Smaller,
}

impl Tail {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Larger => "larger",
            Self::Smaller => "smaller",
        }
    }
}

/// Plus-one-corrected Monte Carlo tail probability and its sampling uncertainty.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmpiricalPValue {
    pub p_value: f64,
    pub monte_carlo_standard_error: f64,
    pub extreme_draws: usize,
    pub draws: usize,
}

/// Evaluate a scalar statistic against an explicit Monte Carlo null distribution.
///
/// This is the single plus-one correction used by every native null-calibrated
/// report. Ties count as extreme in the requested tail, so a null statistic equal
/// to the observation is evidence against rejection rather than a free win.
pub fn empirical_p_value(
    observed: f64,
    null_samples: &[f64],
    tail: Tail,
) -> Result<EmpiricalPValue, String> {
    require_finite(observed, "observed statistic")?;
    if null_samples.is_empty() {
        return Err("empirical p-value requires at least one null draw".to_string());
    }
    for (draw, &sample) in null_samples.iter().enumerate() {
        require_finite(sample, &format!("null statistic draw {draw}"))?;
    }
    let extreme_draws = null_samples
        .iter()
        .filter(|&&sample| match tail {
            Tail::Larger => sample >= observed,
            Tail::Smaller => sample <= observed,
        })
        .count();
    let draws = null_samples.len();
    let p_value = (extreme_draws as f64 + 1.0) / (draws as f64 + 1.0);
    let monte_carlo_standard_error = (p_value * (1.0 - p_value) / (draws as f64 + 1.0)).sqrt();
    Ok(EmpiricalPValue {
        p_value,
        monte_carlo_standard_error,
        extreme_draws,
        draws,
    })
}

/// Marchenko–Pastur reconstruction-rank edge (#2262): the per-observation
/// reconstruction energy `R·(1+√(p/n_eff))²` that the production rank charge
/// uses to count a decoder direction in the hard reconstruction rank at
/// `(n_eff, p, R)`. This is the
/// identical closed-form noise edge the production rank charge thresholds on
/// ([`crate::manifold::construction::realised_rank_charge_dof`], and its
/// audit twin
/// [`crate::manifold::wbic_audit::ReconSpectrum::mp_reconstruction_rank_edge`]) —
/// surfaced standalone so a caller can report the rank-charge diagnostic
/// alongside a shape verdict without needing a fitted decoder Gram. It is not
/// an information-theoretic detection limit and the predictive 2-D shape race
/// does not threshold on it: a direction below this edge is omitted from the
/// hard reconstruction-rank count, but that fact neither negates nor overrides
/// a shape verdict.
pub fn mp_reconstruction_rank_edge(n_eff: f64, p: f64, r_floor: f64) -> Result<f64, String> {
    if !n_eff.is_finite() || n_eff <= 0.0 {
        return Err(format!(
            "mp_reconstruction_rank_edge: n_eff must be finite and positive; got {n_eff}"
        ));
    }
    if !p.is_finite() || p < 0.0 {
        return Err(format!(
            "mp_reconstruction_rank_edge: p must be finite and non-negative; got {p}"
        ));
    }
    if !r_floor.is_finite() || r_floor < 0.0 {
        return Err(format!(
            "mp_reconstruction_rank_edge: r_floor must be finite and non-negative; got {r_floor}"
        ));
    }
    // Zero residual dispersion has an exactly zero edge. Handle it before the
    // aspect ratio so valid zero-noise inputs cannot manufacture `0 * inf =
    // NaN` when `p / n_eff` exceeds the representable range.
    if r_floor == 0.0 {
        return Ok(0.0);
    }
    let edge = r_floor * (1.0 + (p / n_eff).sqrt()).powi(2);
    if edge.is_finite() {
        return Ok(edge);
    }

    // The aspect ratio or its square can overflow even when multiplication by
    // a small, positive dispersion brings the final edge back into range. The
    // logarithmic reconstruction evaluates that same product without losing a
    // representable answer:
    //   log edge = log R + 2 log(1 + exp(½(log p - log n))).
    // Keep the direct path above so ordinary inputs retain their exact rounding.
    let log_sqrt_aspect = 0.5 * (p.ln() - n_eff.ln());
    let log_multiplier = if log_sqrt_aspect > 0.0 {
        log_sqrt_aspect + (-log_sqrt_aspect).exp().ln_1p()
    } else {
        log_sqrt_aspect.exp().ln_1p()
    };
    let log_edge = r_floor.ln() + 2.0 * log_multiplier;
    if !log_edge.is_finite() || log_edge > f64::MAX.ln() {
        return Err(format!(
            "mp_reconstruction_rank_edge: edge is not representable for n_eff={n_eff}, p={p}, r_floor={r_floor}"
        ));
    }
    let recovered_edge = log_edge.exp();
    if !recovered_edge.is_finite() || recovered_edge == 0.0 {
        return Err(format!(
            "mp_reconstruction_rank_edge: edge is not representable for n_eff={n_eff}, p={p}, r_floor={r_floor}"
        ));
    }
    Ok(recovered_edge)
}

/// One member of the standing null battery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NullKind {
    /// Randomize each column's Fourier phases along token order, preserving its
    /// one-dimensional power spectrum while destroying coherent ordered shape.
    PhaseRandomized,
    /// Apply a Haar-like random orthogonal basis change. Coordinate-dependent
    /// claims should fail; basis-invariant claims should be unchanged.
    RandomRotation,
    /// Permute token rows, preserving the activation cloud and column marginals
    /// while destroying row-order structure.
    TokenShuffle,
    /// Permute every column independently. This preserves each one-dimensional
    /// empirical marginal exactly while destroying cross-column geometry such
    /// as an unordered circle or ring of clusters.
    PerDimensionShuffle,
    /// Draw independent Gaussian principal-component scores with the same mean
    /// and variance parameters in every supplied PC column. The finite random
    /// draw fluctuates around, rather than exactly preserving, that target
    /// eigenspectrum while destroying cyclic/non-Gaussian structure.
    MatchedSpectrumGaussian,
    /// Apply a seeded, mean-fixing orthogonal Hadamard transform to globally
    /// permuted rows. This preserves the empirical mean and full covariance
    /// while destroying rowwise nonlinear geometry in bounded memory.
    CovarianceExactHadamard,
    /// Use separately supplied random-weight activations from the same
    /// architecture, moment-matched to the observed matrix shape.
    ArchitectureMatchedRandomWeight,
}

impl NullKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PhaseRandomized => "phase_randomized",
            Self::RandomRotation => "random_rotation",
            Self::TokenShuffle => "token_shuffle",
            Self::PerDimensionShuffle => "per_dimension_shuffle",
            Self::MatchedSpectrumGaussian => "matched_spectrum_gaussian",
            Self::CovarianceExactHadamard => "covariance_exact_hadamard",
            Self::ArchitectureMatchedRandomWeight => "architecture_matched_random_weight",
        }
    }
}

/// Configuration for [`run_null_battery`].
#[derive(Clone, Debug)]
pub struct NullBatteryConfig {
    pub replicates: usize,
    pub seed: u64,
    pub kinds: Vec<NullKind>,
    pub tail: Tail,
}

impl NullBatteryConfig {
    pub fn standing(replicates: usize, seed: u64) -> Self {
        Self {
            replicates,
            seed,
            kinds: vec![
                NullKind::PhaseRandomized,
                NullKind::RandomRotation,
                NullKind::TokenShuffle,
                NullKind::PerDimensionShuffle,
                NullKind::MatchedSpectrumGaussian,
                NullKind::CovarianceExactHadamard,
                NullKind::ArchitectureMatchedRandomWeight,
            ],
            tail: Tail::Larger,
        }
    }
}

/// Null distribution summary for a single null kind.
#[derive(Clone, Debug, PartialEq)]
pub struct NullSummary {
    pub kind: NullKind,
    pub tail: Tail,
    pub observed: f64,
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    pub min: f64,
    pub q25: f64,
    pub median: f64,
    pub q75: f64,
    pub max: f64,
    pub z: f64,
    pub p_value: f64,
    pub monte_carlo_standard_error: f64,
    pub extreme_draws: usize,
    /// Statistic values in draw order. Keeping seed order (rather than sorting
    /// this ledger for quantiles) makes a persisted null artifact replayable.
    pub samples: Vec<f64>,
}

/// Complete null-battery result for one scalar audit statistic.
#[derive(Clone, Debug)]
pub struct NullBatteryReport {
    pub observed: f64,
    pub summaries: Vec<NullSummary>,
}

/// Synthetic topology planted into residual activations for spike-in calibration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpikeInShape {
    Circle,
    Torus,
}

impl SpikeInShape {
    fn signal_rank(self) -> usize {
        match self {
            SpikeInShape::Circle => 2,
            SpikeInShape::Torus => 4,
        }
    }

    fn expected_betti(self) -> (usize, usize, usize) {
        match self {
            SpikeInShape::Circle => (1, 1, 0),
            SpikeInShape::Torus => (1, 2, 1),
        }
    }
}

/// How trial noise is drawn before a synthetic topology is injected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpikeInNoiseMode {
    /// Resample centered residual rows with replacement. This keeps the actual
    /// post-sink-peel residual covariance and heavy-tailed row distribution in
    /// the calibration loop.
    EmpiricalResidualBootstrap,
    /// Preserve each residual channel's one-dimensional spectrum while
    /// destroying coherent topology.
    PhaseRandomizedResidual,
    /// Treat the supplied matrix as a moment donor and draw a covariance-
    /// matched heavy-tailed surrogate with the donor's average marginal excess
    /// kurtosis. Use this when real residual activations are unavailable.
    CovarianceHeavyTailSurrogate,
}

/// Configuration for [`spike_in_roc_curve`].
#[derive(Clone, Debug)]
pub struct SpikeInRocConfig {
    pub shape: SpikeInShape,
    pub snrs: Vec<f64>,
    pub trials: usize,
    pub seed: u64,
    pub fpr_levels: Vec<f64>,
    pub noise_mode: SpikeInNoiseMode,
}

impl SpikeInRocConfig {
    pub fn circle(snrs: Vec<f64>, trials: usize, seed: u64) -> Self {
        Self {
            shape: SpikeInShape::Circle,
            snrs,
            trials,
            seed,
            // reporting-only: FPR operating points at which spike-in ROC TPR is reported.
            fpr_levels: vec![0.01, 0.05, 0.10],
            noise_mode: SpikeInNoiseMode::PhaseRandomizedResidual,
        }
    }

    pub fn torus(snrs: Vec<f64>, trials: usize, seed: u64) -> Self {
        Self {
            shape: SpikeInShape::Torus,
            snrs,
            trials,
            seed,
            // reporting-only: FPR operating points at which spike-in ROC TPR is reported.
            fpr_levels: vec![0.01, 0.05, 0.10],
            noise_mode: SpikeInNoiseMode::EmpiricalResidualBootstrap,
        }
    }
}

/// Explicit covariance/heavy-tail moments for generating a surrogate residual
/// activation matrix when real post-sink-peel activations are unavailable.
#[derive(Clone, Debug)]
pub struct ResidualMomentSpec {
    pub rows: usize,
    pub mean: Array1<f64>,
    pub covariance: Array2<f64>,
    pub excess_kurtosis: f64,
}

/// Block-chart promotion plus topology-audit verdict for one detection run.
#[derive(Clone, Debug)]
pub struct DetectionPipelineReport {
    pub shape: SpikeInShape,
    pub statistic: f64,
    pub promoted: bool,
    pub topology: TopologyAuditReport,
    pub detected: bool,
}

/// Lightweight topology audit payload carried by spike-in ROC points. The
/// measured Betti numbers are the detector's auditable claim for the supplied
/// residual matrix, not a latched topology label.
#[derive(Clone, Debug)]
pub struct TopologyAuditReport {
    pub expected_betti0: usize,
    pub expected_betti1: usize,
    pub expected_betti2: usize,
    pub measured_betti0: usize,
    pub measured_betti1: usize,
    pub measured_betti2: usize,
    pub rank_energy: f64,
    pub spectral_balance: f64,
    pub residual_tail_energy: f64,
    pub accepted: bool,
}

/// One ROC operating point at a fixed false-positive-rate threshold.
#[derive(Clone, Debug)]
pub struct SpikeInRocThreshold {
    pub false_positive_rate: f64,
    pub threshold: f64,
    pub true_positive_rate: f64,
}

/// ROC payload for one injected SNR.
#[derive(Clone, Debug)]
pub struct SpikeInRocPoint {
    pub snr: f64,
    pub trials: usize,
    pub mean_stat: f64,
    pub promoted_fraction: f64,
    pub topology_accept_fraction: f64,
    pub roc: Vec<SpikeInRocThreshold>,
}

/// Claim payload carrying the full spike-in ROC calibration. This is the shape
/// topology, Betti, and conditionality claims should ship when the detector has
/// an explicit operating FPR.
#[derive(Clone, Debug)]
pub struct CalibratedRocClaimReport {
    pub claim: String,
    pub claimed_snr: f64,
    pub claimed_false_positive_rate: f64,
    pub nulls: NullBatteryReport,
    pub spike_in_roc: Vec<SpikeInRocPoint>,
    pub claimed_snr_power: f64,
}

/// Compact null-calibrated audit payload attached directly to emitted claims.
#[derive(Clone, Debug)]
pub struct ClaimNullCalibration {
    pub claim: String,
    pub observed_statistic: f64,
    pub null_pvalue: f64,
    pub null_z: f64,
    pub claimed_snr: f64,
    pub claimed_false_positive_rate: f64,
    pub spikein_power: f64,
    pub null_distribution: Vec<NullSummary>,
    pub spike_in_roc: Vec<SpikeInRocPoint>,
}

impl ClaimNullCalibration {
    pub fn from_calibrated_roc(report: CalibratedRocClaimReport) -> Result<Self, String> {
        let (null_pvalue, null_z) = primary_null_metrics(&report.nulls)?;
        Ok(Self {
            claim: report.claim,
            observed_statistic: report.nulls.observed,
            null_pvalue,
            null_z,
            claimed_snr: report.claimed_snr,
            claimed_false_positive_rate: report.claimed_false_positive_rate,
            spikein_power: report.claimed_snr_power,
            null_distribution: report.nulls.summaries,
            spike_in_roc: report.spike_in_roc,
        })
    }
}

/// Conservative headline across every available structure-destroying control.
/// Returns `(maximum p-value, minimum z-score)` in one allocation-free pass.
pub fn primary_null_metrics(report: &NullBatteryReport) -> Result<(f64, f64), String> {
    if report.summaries.is_empty() {
        return Err("primary null calibration requires at least one null summary".to_string());
    }
    let has_primary = report
        .summaries
        .iter()
        .any(|summary| is_primary_structure_destroying_null(summary.kind));
    if !has_primary {
        return Err(
            "primary null calibration requires at least one structure-destroying control"
                .to_string(),
        );
    }
    let mut p_value = 0.0_f64;
    let mut z = f64::INFINITY;
    for summary in &report.summaries {
        if !(summary.p_value.is_finite() && (0.0..=1.0).contains(&summary.p_value)) {
            return Err(format!(
                "primary null calibration received invalid {} p-value {}",
                summary.kind.as_str(),
                summary.p_value
            ));
        }
        if !summary.z.is_finite() {
            return Err(format!(
                "primary null calibration received non-finite {} z-score {}",
                summary.kind.as_str(),
                summary.z
            ));
        }
        if is_primary_structure_destroying_null(summary.kind) {
            p_value = p_value.max(summary.p_value);
            z = z.min(summary.z);
        }
    }
    Ok((p_value, z))
}

/// Controls that destroy the claimed cross-row/cross-coordinate structure
/// while retaining a matched nuisance property. A claim must clear every
/// available member, hence the conservative maximum p-value / minimum z-score.
/// Token shuffles and rotations are deliberately excluded: they preserve an
/// unordered point cloud or a basis-invariant topology, respectively.
const fn is_primary_structure_destroying_null(kind: NullKind) -> bool {
    matches!(
        kind,
        NullKind::PhaseRandomized
            | NullKind::PerDimensionShuffle
            | NullKind::MatchedSpectrumGaussian
            | NullKind::CovarianceExactHadamard
            | NullKind::ArchitectureMatchedRandomWeight
    )
}

/// Run an arbitrary scalar audit on the observed matrix and each requested null.
///
/// If [`NullKind::ArchitectureMatchedRandomWeight`] is requested,
/// `random_weight` must be a real activation matrix harvested from a random-
/// weight model with the same architecture family. The harness only resamples
/// and moment-matches it; it does not fabricate an architecture null.
pub fn run_null_battery<F>(
    data: ArrayView2<'_, f64>,
    random_weight: Option<ArrayView2<'_, f64>>,
    config: &NullBatteryConfig,
    mut audit: F,
) -> Result<NullBatteryReport, String>
where
    F: FnMut(ArrayView2<'_, f64>) -> Result<f64, String>,
{
    validate_matrix(data, "observed")?;
    if config.replicates == 0 {
        return Err("null battery requires at least one replicate".to_string());
    }
    if config.kinds.is_empty() {
        return Err("null battery requires at least one null kind".to_string());
    }
    for index in 0..config.kinds.len() {
        if config.kinds[..index].contains(&config.kinds[index]) {
            return Err(format!(
                "null battery contains duplicate {} control",
                config.kinds[index].as_str()
            ));
        }
    }
    let observed = audit(data)?;
    require_finite(observed, "observed statistic")?;
    let mut summaries = Vec::with_capacity(config.kinds.len());

    for &kind in &config.kinds {
        let mut samples = Vec::with_capacity(config.replicates);
        for rep in 0..config.replicates {
            let rep_seed = mix_seed(config.seed, kind_seed(kind), rep as u64);
            let null = match kind {
                NullKind::PhaseRandomized => phase_randomized_surrogate(data, rep_seed)?,
                NullKind::RandomRotation => random_rotation_null(data, rep_seed)?,
                NullKind::TokenShuffle => token_shuffle_null(data, rep_seed)?,
                NullKind::PerDimensionShuffle => per_dimension_shuffle_null(data, rep_seed)?,
                NullKind::MatchedSpectrumGaussian => {
                    matched_spectrum_gaussian_null(data, rep_seed)?
                }
                NullKind::CovarianceExactHadamard => {
                    covariance_exact_hadamard_null(data, rep_seed)?
                }
                NullKind::ArchitectureMatchedRandomWeight => {
                    let Some(rw) = random_weight else {
                        return Err(
                            "architecture-matched random-weight null requested without random_weight activations"
                                .to_string(),
                        );
                    };
                    architecture_matched_random_weight_null(data, rw, rep_seed)?
                }
            };
            let stat = audit(null.view())?;
            require_finite(stat, "null statistic")?;
            samples.push(stat);
        }
        summaries.push(summarize_null_distribution(
            kind,
            observed,
            samples,
            config.tail,
        )?);
    }

    Ok(NullBatteryReport {
        observed,
        summaries,
    })
}

/// Fourier phase-randomized surrogate, independently per activation channel.
pub fn phase_randomized_surrogate(
    data: ArrayView2<'_, f64>,
    seed: u64,
) -> Result<Array2<f64>, String> {
    validate_matrix(data, "phase-randomized input")?;
    let n = data.nrows();
    let p = data.ncols();
    let mut out = Array2::<f64>::zeros((n, p));
    let mut rng = StdRng::seed_from_u64(seed);
    for col in 0..p {
        let mut re = vec![0.0_f64; n];
        let mut im = vec![0.0_f64; n];
        for k in 0..n {
            for t in 0..n {
                let angle = -2.0 * PI * (k as f64) * (t as f64) / (n as f64);
                let x = data[[t, col]];
                re[k] += x * angle.cos();
                im[k] += x * angle.sin();
            }
        }
        let last_paired = (n.saturating_sub(1)) / 2;
        for k in 1..=last_paired {
            let amp = (re[k] * re[k] + im[k] * im[k]).sqrt();
            let phase = rng.random_range(0.0..(2.0 * PI));
            re[k] = amp * phase.cos();
            im[k] = amp * phase.sin();
            let mirror = n - k;
            re[mirror] = re[k];
            im[mirror] = -im[k];
        }
        if n % 2 == 0 {
            let nyquist = n / 2;
            im[nyquist] = 0.0;
        }
        for t in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..n {
                let angle = 2.0 * PI * (k as f64) * (t as f64) / (n as f64);
                acc += re[k] * angle.cos() - im[k] * angle.sin();
            }
            out[[t, col]] = acc / (n as f64);
        }
    }
    Ok(out)
}

/// Multiply the activation matrix by a seeded random orthogonal matrix.
pub fn random_rotation_null(data: ArrayView2<'_, f64>, seed: u64) -> Result<Array2<f64>, String> {
    validate_matrix(data, "random-rotation input")?;
    let p = data.ncols();
    let q = random_orthogonal(p, seed)?;
    let rotated = data.dot(&q);
    validate_matrix(rotated.view(), "random-rotation output")?;
    Ok(rotated)
}

/// Deterministically seeded token-row shuffle.
pub fn token_shuffle_null(data: ArrayView2<'_, f64>, seed: u64) -> Result<Array2<f64>, String> {
    validate_matrix(data, "token-shuffle input")?;
    let mut order: Vec<usize> = (0..data.nrows()).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    order.shuffle(&mut rng);
    let mut out = Array2::<f64>::zeros(data.raw_dim());
    for (dst, &src) in order.iter().enumerate() {
        out.row_mut(dst).assign(&data.row(src));
    }
    Ok(out)
}

/// Independently permute each dimension of an activation/coordinate matrix.
///
/// Every output column is a bit-for-bit permutation of the corresponding input
/// column, so all empirical marginal moments and tails are preserved exactly.
/// The independently seeded permutations break cross-dimensional geometry; in
/// particular, this is the matched shuffle control required for an unordered
/// circle census, whereas [`token_shuffle_null`] leaves that point cloud intact.
/// Column seeds depend only on `(seed, column)`, so cache-line column bands can
/// execute in parallel with bit-identical output under every Rayon schedule.
pub fn per_dimension_shuffle_null(
    data: ArrayView2<'_, f64>,
    seed: u64,
) -> Result<Array2<f64>, String> {
    validate_matrix(data, "per-dimension-shuffle input")?;
    per_dimension_shuffle_null_impl(data, seed)
}

/// Float32-preserving form of [`per_dimension_shuffle_null`].
///
/// The output is allocated directly as `n × p` float32 and values are copied
/// bit-for-bit from the input. No float64 data matrix is materialized.
pub fn per_dimension_shuffle_null_f32(
    data: ArrayView2<'_, f32>,
    seed: u64,
) -> Result<Array2<f32>, String> {
    validate_matrix_f32(data, "float32 per-dimension-shuffle input")?;
    per_dimension_shuffle_null_impl(data, seed)
}

trait NativeControlScalar: Copy + Default + Send + Sync {
    const TYPE_NAME: &'static str;

    fn widen(self) -> f64;
    fn narrow(value: f64) -> Option<Self>;
}

impl NativeControlScalar for f64 {
    const TYPE_NAME: &'static str = "float64";

    fn widen(self) -> f64 {
        self
    }

    fn narrow(value: f64) -> Option<Self> {
        value.is_finite().then_some(value)
    }
}

impl NativeControlScalar for f32 {
    const TYPE_NAME: &'static str = "float32";

    fn widen(self) -> f64 {
        f64::from(self)
    }

    fn narrow(value: f64) -> Option<Self> {
        let narrowed = value as f32;
        (value.is_finite() && narrowed.is_finite()).then_some(narrowed)
    }
}

fn fallible_native_output<T: NativeControlScalar>(
    n: usize,
    p: usize,
    control_name: &str,
) -> Result<Array2<T>, String> {
    let output_scalars = n.checked_mul(p).ok_or_else(|| {
        format!(
            "{} {control_name} output shape overflowed: {n} rows × {p} columns",
            T::TYPE_NAME
        )
    })?;
    let mut output_storage = Vec::<T>::new();
    output_storage
        .try_reserve_exact(output_scalars)
        .map_err(|error| {
            format!(
                "could not allocate {} {control_name} output ({n} × {p} native scalars): {error}",
                T::TYPE_NAME
            )
        })?;
    output_storage.resize(output_scalars, T::default());
    Array2::<T>::from_shape_vec((n, p), output_storage).map_err(|error| {
        format!(
            "could not construct {} {control_name} output with shape {n} × {p}: {error}",
            T::TYPE_NAME
        )
    })
}

fn per_dimension_shuffle_null_impl<T: NativeControlScalar>(
    data: ArrayView2<'_, T>,
    seed: u64,
) -> Result<Array2<T>, String> {
    let n = data.nrows();
    let p = data.ncols();
    let mut out = fallible_native_output::<T>(n, p, "per-dimension-shuffle")?;
    // Give each Rayon task a cache-line-wide band of columns. Parallelizing
    // single strided columns would make workers repeatedly write the same cache
    // lines in every C-order output row. A band retains independent column
    // permutations while eliminating almost all false sharing and reusing one
    // Fisher-Yates index buffer across the task's columns.
    let column_band_width = (64 / std::mem::size_of::<T>()).max(1);
    out.axis_chunks_iter_mut(Axis(1), column_band_width)
        .into_par_iter()
        .enumerate()
        .try_for_each(|(band, mut output_band)| -> Result<(), String> {
            let first_col = band.checked_mul(column_band_width).ok_or_else(|| {
                format!(
                    "per-dimension-shuffle column-band index overflowed: band {band} × width {column_band_width}"
                )
            })?;
            let mut order = Vec::<usize>::new();
            order.try_reserve_exact(n).map_err(|error| {
                format!(
                    "could not allocate per-dimension-shuffle row order for column band starting at {first_col} ({n} indices): {error}"
                )
            })?;
            order.extend(0..n);
            for local_col in 0..output_band.ncols() {
                let col = first_col + local_col;
                for (row, slot) in order.iter_mut().enumerate() {
                    *slot = row;
                }
                let mut rng = StdRng::seed_from_u64(mix_seed(
                    seed,
                    PER_DIMENSION_SHUFFLE_SEED_DOMAIN,
                    col as u64,
                ));
                order.shuffle(&mut rng);
                let mut output_column = output_band.column_mut(local_col);
                for (destination, &source) in order.iter().enumerate() {
                    output_column[destination] = data[[source, col]];
                }
            }
            Ok(())
        })?;
    Ok(out)
}

fn largest_power_of_two_at_most(value: usize) -> Result<usize, String> {
    if value == 0 {
        return Err("Hadamard block size must be positive".to_string());
    }
    Ok(1usize << (usize::BITS - value.leading_zeros() - 1))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HadamardBlock {
    start: usize,
    rows: usize,
    ordinal: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HadamardTile {
    block: HadamardBlock,
    first_col: usize,
    cols: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HadamardParallelPlan {
    tasks: usize,
    column_bands: usize,
    workspace_cols: usize,
}

fn covariance_exact_hadamard_blocks(nrows: usize) -> Result<Vec<HadamardBlock>, String> {
    if nrows == 0 {
        return Err("covariance-exact Hadamard block plan requires rows".to_string());
    }
    let capacity = (nrows / COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS)
        .checked_add(usize::BITS as usize)
        .ok_or_else(|| "covariance-exact Hadamard block-plan capacity overflowed".to_string())?;
    let mut blocks = Vec::<HadamardBlock>::new();
    blocks.try_reserve_exact(capacity).map_err(|error| {
        format!(
            "could not allocate covariance-exact Hadamard block plan ({capacity} descriptors): {error}"
        )
    })?;
    let mut start = 0usize;
    while start < nrows {
        let rows = largest_power_of_two_at_most(
            (nrows - start).min(COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS),
        )?;
        let ordinal = blocks.len();
        blocks.push(HadamardBlock {
            start,
            rows,
            ordinal,
        });
        start = start.checked_add(rows).ok_or_else(|| {
            "covariance-exact Hadamard block-plan row offset overflowed".to_string()
        })?;
    }
    Ok(blocks)
}

fn covariance_exact_hadamard_parallel_plan(
    nrows: usize,
    ncols: usize,
) -> Result<HadamardParallelPlan, String> {
    if nrows == 0 || ncols == 0 {
        return Err(
            "covariance-exact Hadamard parallel plan requires rows and columns".to_string(),
        );
    }
    let row_groups = nrows.div_ceil(COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS);
    let available_tasks =
        COVARIANCE_EXACT_HADAMARD_MAX_PARALLEL_WORKSPACES.min(rayon::current_num_threads().max(1));
    // A transform is separable across columns. When there are too few row
    // blocks to occupy the pool, split each one into the smallest number of
    // balanced column bands that can expose all available workers. Besides
    // filling the pool, this keeps ultra-wide B×p workspaces out of the
    // last-level-cache streaming regime without adding a stage barrier.
    let workspace_rows = covariance_exact_hadamard_workspace_rows(nrows)?;
    let scalar_bytes = std::mem::size_of::<f64>();
    let maximum_workspace_scalars = COVARIANCE_EXACT_HADAMARD_TARGET_TILE_BYTES / scalar_bytes;
    let maximum_workspace_cols = (maximum_workspace_scalars / workspace_rows).max(1);
    let worker_bands = available_tasks.div_ceil(row_groups);
    let budget_bands = ncols.div_ceil(maximum_workspace_cols);
    let column_bands = worker_bands.max(budget_bands).min(ncols).max(1);
    let workspace_cols = ncols.div_ceil(column_bands);
    let workspace_bytes = workspace_rows
        .checked_mul(workspace_cols)
        .and_then(|scalars| scalars.checked_mul(scalar_bytes))
        .ok_or_else(|| "covariance-exact Hadamard tile workspace size overflowed".to_string())?;
    let budget_tasks =
        (COVARIANCE_EXACT_HADAMARD_PARALLEL_WORKSPACE_BUDGET_BYTES / workspace_bytes).max(1);
    let tile_groups = row_groups.checked_mul(column_bands).ok_or_else(|| {
        "covariance-exact Hadamard parallel tile-group count overflowed".to_string()
    })?;
    Ok(HadamardParallelPlan {
        tasks: available_tasks.min(tile_groups).min(budget_tasks),
        column_bands,
        workspace_cols,
    })
}

fn covariance_exact_hadamard_tiles(
    blocks: &[HadamardBlock],
    ncols: usize,
    column_bands: usize,
) -> Result<Vec<HadamardTile>, String> {
    if blocks.is_empty() || ncols == 0 || column_bands == 0 || column_bands > ncols {
        return Err("covariance-exact Hadamard tile plan is invalid".to_string());
    }
    let tile_count = blocks
        .len()
        .checked_mul(column_bands)
        .ok_or_else(|| "covariance-exact Hadamard tile count overflowed".to_string())?;
    let mut tiles = Vec::<HadamardTile>::new();
    tiles.try_reserve_exact(tile_count).map_err(|error| {
        format!(
            "could not allocate covariance-exact Hadamard tile plan ({tile_count} descriptors): {error}"
        )
    })?;
    let base_cols = ncols / column_bands;
    let extra_bands = ncols % column_bands;
    for &block in blocks {
        let mut first_col = 0usize;
        for band in 0..column_bands {
            let cols = base_cols + usize::from(band < extra_bands);
            tiles.push(HadamardTile {
                block,
                first_col,
                cols,
            });
            first_col = first_col.checked_add(cols).ok_or_else(|| {
                "covariance-exact Hadamard tile column offset overflowed".to_string()
            })?;
        }
        if first_col != ncols {
            return Err(
                "covariance-exact Hadamard tile columns do not partition input".to_string(),
            );
        }
    }
    Ok(tiles)
}

fn covariance_exact_hadamard_workspace_rows(nrows: usize) -> Result<usize, String> {
    largest_power_of_two_at_most(nrows.min(COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS))
}

/// In-place, unnormalised Sylvester fast Walsh-Hadamard transform over rows.
///
/// The transform is deliberately sequential inside a row-block/column-band
/// tile. Independent tiles are the parallel unit, avoiding a Rayon barrier at
/// every butterfly stage while every scalar retains one fixed operation
/// sequence.
fn fwht_rows_in_place(workspace: &mut [f64], rows: usize, cols: usize) -> Result<(), String> {
    if !rows.is_power_of_two() || cols == 0 {
        return Err(format!(
            "Hadamard transform shape must have positive power-of-two rows and positive columns; got {rows} × {cols}"
        ));
    }
    let expected_scalars = rows.checked_mul(cols).ok_or_else(|| {
        format!("Hadamard transform shape overflowed: {rows} rows × {cols} columns")
    })?;
    if workspace.len() != expected_scalars {
        return Err(format!(
            "Hadamard transform workspace has {} scalars, expected {expected_scalars} for {rows} × {cols}",
            workspace.len()
        ));
    }
    let mut half = 1usize;
    while half < rows {
        let slab_rows = half
            .checked_mul(2)
            .ok_or_else(|| "Hadamard stage row count overflowed".to_string())?;
        let slab_scalars = slab_rows.checked_mul(cols).ok_or_else(|| {
            format!("Hadamard stage shape overflowed: {slab_rows} rows × {cols} columns")
        })?;
        let half_scalars = half.checked_mul(cols).ok_or_else(|| {
            format!("Hadamard half-stage shape overflowed: {half} rows × {cols} columns")
        })?;
        if workspace.len() % slab_scalars != 0 {
            return Err(format!(
                "Hadamard stage slab of {slab_scalars} scalars does not divide workspace length {}",
                workspace.len()
            ));
        }
        for slab in workspace.chunks_exact_mut(slab_scalars) {
            let (upper, lower) = slab.split_at_mut(half_scalars);
            for paired_row in 0..half {
                let row_offset = paired_row * cols;
                for col in 0..cols {
                    let index = row_offset + col;
                    let left = upper[index];
                    let right = lower[index];
                    upper[index] = left + right;
                    lower[index] = left - right;
                }
            }
        }
        half = slab_rows;
    }
    Ok(())
}

/// Unit vector orthogonal to the all-ones direction. Its Householder
/// reflection `Q = I - 2vv'` fixes the empirical mean, is dense for every
/// `n >= 3`, and costs one projection plus one rank-one update instead of an
/// `n x n` matrix. A seeded row permutation randomizes which observation is
/// the distinguished final coordinate.
fn dense_fixed_mean_householder_vector(n: usize) -> Result<Vec<f64>, String> {
    if n < 3 {
        return Err(format!(
            "covariance-exact geometry destruction requires at least three rows; with n={n}, every mean-fixing orthogonal map is only a row permutation"
        ));
    }
    let n_float = n as f64;
    let small = (n_float * (n_float - 1.0)).sqrt().recip();
    let large = -((n - 1) as f64) * small;
    if !(small.is_finite() && small > 0.0 && large.is_finite()) {
        return Err("could not construct the dense fixed-mean Householder vector".to_string());
    }
    let mut vector = Vec::new();
    vector.try_reserve_exact(n).map_err(|error| {
        format!(
            "could not allocate dense fixed-mean Householder vector ({n} float64 scalars): {error}"
        )
    })?;
    vector.resize(n, small);
    vector[n - 1] = large;
    Ok(vector)
}

fn is_walsh_character(signs: &[f64]) -> bool {
    if signs.len() < 2 || !signs.len().is_power_of_two() || signs[0] != 1.0 {
        return false;
    }
    for index in 0..signs.len() {
        let mut expected = 1.0;
        let mut remaining = index;
        let mut bit = 0usize;
        while remaining != 0 {
            if remaining & 1 == 1 {
                expected *= signs[1usize << bit];
            }
            remaining >>= 1;
            bit += 1;
        }
        if signs[index] != expected {
            return false;
        }
    }
    true
}

fn fill_hadamard_signs(
    signs: &mut [f64],
    seed: u64,
    block: HadamardBlock,
) -> Result<(), String> {
    if signs.len() != block.rows || !block.rows.is_power_of_two() {
        return Err("Hadamard sign workspace does not match its block".to_string());
    }
    let block_ordinal = u64::try_from(block.ordinal).map_err(|_| {
        format!(
            "covariance-exact Hadamard block ordinal {} exceeds the seed domain",
            block.ordinal
        )
    })?;
    let mut sign_rng = StdRng::seed_from_u64(mix_seed(
        seed ^ HADAMARD_SIGN_SEED_DOMAIN,
        block_ordinal,
        block.rows as u64,
    ));
    signs[0] = 1.0;
    for sign in &mut signs[1..] {
        *sign = if sign_rng.random_range(0..2) == 0 {
            -1.0
        } else {
            1.0
        };
    }
    // H diag(d) H / B is a signed row permutation exactly when `d` is a
    // Walsh character. Exclude that finite bad set for every block where a
    // genuinely mixing fixed-mean Hadamard map exists. Flipping one entry of a
    // character cannot produce another character for B >= 4 (distinct
    // characters differ on B/2 entries).
    if block.rows >= 4 && is_walsh_character(signs) {
        signs[1] = -signs[1];
    }
    Ok(())
}

fn transform_covariance_exact_hadamard_tile<T: NativeControlScalar>(
    data: &ArrayView2<'_, T>,
    permutation: &[usize],
    origin: &[f64],
    column_scale: &[f64],
    householder_projection: &[f64],
    householder_vector: &[f64],
    seed: u64,
    tile: HadamardTile,
    workspace: &mut [f64],
    signs: &mut [f64],
) -> Result<(), String> {
    let p = data.ncols();
    let block = tile.block;
    let block_end = block.start.checked_add(block.rows).ok_or_else(|| {
        format!(
            "covariance-exact Hadamard block {} row range overflowed",
            block.ordinal
        )
    })?;
    let column_end = tile.first_col.checked_add(tile.cols).ok_or_else(|| {
        format!(
            "covariance-exact Hadamard block {} column range overflowed",
            block.ordinal
        )
    })?;
    if !block.rows.is_power_of_two()
        || block.rows > COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS
        || block_end > permutation.len()
        || tile.cols == 0
        || column_end > p
        || origin.len() != p
        || column_scale.len() != p
        || householder_projection.len() != p
        || householder_vector.len() != permutation.len()
    {
        return Err(format!(
            "covariance-exact Hadamard block {} has an invalid internal plan",
            block.ordinal
        ));
    }
    let block_scalars = block.rows.checked_mul(tile.cols).ok_or_else(|| {
        format!(
            "{} covariance-exact Hadamard tile shape overflowed: {} rows × {} columns",
            T::TYPE_NAME,
            block.rows,
            tile.cols
        )
    })?;
    if workspace.len() != block_scalars || signs.len() != block.rows {
        return Err(format!(
            "covariance-exact Hadamard block {} received invalid workspace lengths",
            block.ordinal
        ));
    }
    let inverse_rows = 1.0 / block.rows as f64;

    for local_row in 0..block.rows {
        let permuted_row = block.start + local_row;
        let source_row = permutation[permuted_row];
        if source_row >= data.nrows() {
            return Err(format!(
                "covariance-exact Hadamard permutation row {source_row} is out of bounds"
            ));
        }
        for local_col in 0..tile.cols {
            let col = tile.first_col + local_col;
            // Transform in the first-row residual chart. This prevents a large
            // absolute origin from erasing small covariance-bearing residuals
            // while Q·1 = 1 restores the same origin exactly.
            let relative = data[[source_row, col]].widen() - origin[col];
            let normalized_relative = if column_scale[col] > 0.0 {
                relative / column_scale[col]
            } else {
                0.0
            };
            let mixed_relative = (-2.0 * householder_vector[permuted_row]).mul_add(
                householder_projection[col],
                normalized_relative,
            );
            let scaled = mixed_relative * inverse_rows;
            if !scaled.is_finite() {
                return Err(format!(
                    "{} covariance-exact Hadamard residual overflowed at input row {source_row}, column {col}",
                    T::TYPE_NAME
                ));
            }
            workspace[local_row * tile.cols + local_col] = scaled;
        }
    }

    fwht_rows_in_place(workspace, block.rows, tile.cols)?;
    fill_hadamard_signs(signs, seed, block)?;
    for local_row in 0..block.rows {
        let sign = signs[local_row];
        for local_col in 0..tile.cols {
            workspace[local_row * tile.cols + local_col] *= sign;
        }
    }
    fwht_rows_in_place(workspace, block.rows, tile.cols)
}

fn covariance_exact_hadamard_null_impl<T: NativeControlScalar>(
    data: ArrayView2<'_, T>,
    seed: u64,
) -> Result<Array2<T>, String> {
    let n = data.nrows();
    let p = data.ncols();
    if n == 0 || p == 0 {
        return Err("covariance-exact Hadamard input must be nonempty".to_string());
    }
    let blocks = covariance_exact_hadamard_blocks(n)?;
    let parallel_plan = covariance_exact_hadamard_parallel_plan(n, p)?;
    let tiles = covariance_exact_hadamard_tiles(&blocks, p, parallel_plan.column_bands)?;
    let parallel_tasks = parallel_plan.tasks.min(tiles.len());
    let workspace_rows = covariance_exact_hadamard_workspace_rows(n)?;
    let workspace_scalars = workspace_rows
        .checked_mul(parallel_plan.workspace_cols)
        .ok_or_else(|| {
        format!(
            "{} covariance-exact Hadamard workspace shape overflowed: {workspace_rows} rows × {} columns",
            T::TYPE_NAME,
            parallel_plan.workspace_cols
        )
    })?;

    let mut origin = Vec::<f64>::new();
    origin.try_reserve_exact(p).map_err(|error| {
        format!(
            "could not allocate covariance-exact Hadamard origin ({p} float64 scalars): {error}"
        )
    })?;
    origin.extend((0..p).map(|col| data[[0, col]].widen()));
    let mut permutation = Vec::<usize>::new();
    permutation.try_reserve_exact(n).map_err(|error| {
        format!(
            "could not allocate covariance-exact Hadamard row permutation ({n} indices): {error}"
        )
    })?;
    permutation.extend(0..n);
    let mut permutation_rng = StdRng::seed_from_u64(seed ^ HADAMARD_PERMUTATION_SEED_DOMAIN);
    permutation.shuffle(&mut permutation_rng);

    let householder_vector = dense_fixed_mean_householder_vector(n)?;
    let mut column_scale = Vec::<f64>::new();
    column_scale.try_reserve_exact(p).map_err(|error| {
        format!(
            "could not allocate covariance-exact Hadamard column scales ({p} float64 scalars): {error}"
        )
    })?;
    column_scale.resize(p, 0.0);
    for &source_row in &permutation {
        for col in 0..p {
            let relative = data[[source_row, col]].widen() - origin[col];
            if !relative.is_finite() {
                return Err(format!(
                    "{} covariance-exact Hadamard residual range is not representable at input row {source_row}, column {col}",
                    T::TYPE_NAME
                ));
            }
            column_scale[col] = column_scale[col].max(relative.abs());
        }
    }
    let mut householder_projection = Vec::<f64>::new();
    householder_projection
        .try_reserve_exact(p)
        .map_err(|error| {
            format!(
                "could not allocate covariance-exact Hadamard Householder projection ({p} float64 scalars): {error}"
            )
        })?;
    householder_projection.resize(p, 0.0);
    let mut projection_correction = Vec::<f64>::new();
    projection_correction
        .try_reserve_exact(p)
        .map_err(|error| {
            format!(
                "could not allocate covariance-exact Hadamard projection correction ({p} float64 scalars): {error}"
            )
        })?;
    projection_correction.resize(p, 0.0);
    for (permuted_row, &source_row) in permutation.iter().enumerate() {
        for col in 0..p {
            let normalized_relative = if column_scale[col] > 0.0 {
                (data[[source_row, col]].widen() - origin[col]) / column_scale[col]
            } else {
                0.0
            };
            neumaier_add(
                &mut householder_projection[col],
                &mut projection_correction[col],
                householder_vector[permuted_row] * normalized_relative,
            )?;
        }
    }
    for col in 0..p {
        householder_projection[col] += projection_correction[col];
        if !householder_projection[col].is_finite() {
            return Err(format!(
                "covariance-exact Hadamard Householder projection is non-finite on column {col}"
            ));
        }
    }

    let mut out = fallible_native_output::<T>(n, p, "covariance-exact Hadamard")?;
    let total_workspace_scalars = workspace_scalars
        .checked_mul(parallel_tasks)
        .ok_or_else(|| {
            format!(
                "{} covariance-exact Hadamard parallel workspace size overflowed: {parallel_tasks} × {workspace_rows} × {}",
                T::TYPE_NAME,
                parallel_plan.workspace_cols
            )
        })?;
    let mut workspaces = Vec::<f64>::new();
    workspaces
        .try_reserve_exact(total_workspace_scalars)
        .map_err(|error| {
            format!(
                "could not allocate {} covariance-exact Hadamard parallel workspaces ({parallel_tasks} × {workspace_rows} × {} float64 scalars): {error}",
                T::TYPE_NAME,
                parallel_plan.workspace_cols
            )
        })?;
    workspaces.resize(total_workspace_scalars, 0.0);

    let total_signs = workspace_rows
        .checked_mul(parallel_tasks)
        .ok_or_else(|| {
            format!(
                "covariance-exact Hadamard parallel sign workspace size overflowed: {parallel_tasks} × {workspace_rows}"
            )
        })?;
    let mut signs = Vec::<f64>::new();
    signs.try_reserve_exact(total_signs).map_err(|error| {
        format!(
            "could not allocate covariance-exact Hadamard parallel signs ({parallel_tasks} × {workspace_rows} float64 scalars): {error}"
        )
    })?;
    signs.resize(total_signs, 1.0);

    for wave_start in (0..tiles.len()).step_by(parallel_tasks) {
        let wave_end = wave_start
            .checked_add(parallel_tasks)
            .ok_or_else(|| "covariance-exact Hadamard parallel wave range overflowed".to_string())?
            .min(tiles.len());
        let wave = &tiles[wave_start..wave_end];
        // All three iterators are indexed, so workspace slot i always belongs
        // to wave tile i regardless of Rayon scheduling. Collecting indexed
        // results also makes a simultaneous error report deterministic.
        let results = wave
            .par_iter()
            .zip(workspaces.par_chunks_mut(workspace_scalars))
            .zip(signs.par_chunks_mut(workspace_rows))
            .map(|((tile, workspace), block_signs)| {
                let block_scalars = tile.block.rows.checked_mul(tile.cols).ok_or_else(|| {
                    format!(
                        "{} covariance-exact Hadamard tile shape overflowed: {} rows × {} columns",
                        T::TYPE_NAME,
                        tile.block.rows,
                        tile.cols
                    )
                })?;
                transform_covariance_exact_hadamard_tile(
                    &data,
                    &permutation,
                    &origin,
                    &column_scale,
                    &householder_projection,
                    &householder_vector,
                    seed,
                    *tile,
                    &mut workspace[..block_scalars],
                    &mut block_signs[..tile.block.rows],
                )
            })
            .collect::<Vec<_>>();
        for result in results {
            result?;
        }

        // Narrowing is one linear output pass after the expensive transforms.
        // Keeping it ordered avoids extra native block buffers and preserves the
        // exact output layout while the two O(log B) FWHTs run tile-parallel.
        for (slot, tile) in wave.iter().enumerate() {
            let workspace_start = slot.checked_mul(workspace_scalars).ok_or_else(|| {
                "covariance-exact Hadamard workspace slot offset overflowed".to_string()
            })?;
            let block_scalars = tile.block.rows.checked_mul(tile.cols).ok_or_else(|| {
                format!(
                    "{} covariance-exact Hadamard tile shape overflowed: {} rows × {} columns",
                    T::TYPE_NAME,
                    tile.block.rows,
                    tile.cols
                )
            })?;
            let workspace_end = workspace_start.checked_add(block_scalars).ok_or_else(|| {
                "covariance-exact Hadamard workspace tile range overflowed".to_string()
            })?;
            let workspace = &workspaces[workspace_start..workspace_end];
            for local_row in 0..tile.block.rows {
                let output_row = tile.block.start + local_row;
                for local_col in 0..tile.cols {
                    let col = tile.first_col + local_col;
                    let value = workspace[local_row * tile.cols + local_col]
                        .mul_add(column_scale[col], origin[col]);
                    out[[output_row, col]] = T::narrow(value).ok_or_else(|| {
                        format!(
                            "{} covariance-exact Hadamard output overflowed at row {output_row}, column {col}",
                            T::TYPE_NAME
                        )
                    })?;
                }
            }
        }
    }
    Ok(out)
}

/// Seeded structureless control preserving empirical mean and full covariance.
///
/// Rows are globally permuted and first reflected by the dense mean-fixing
/// Householder map `Q0 = I - 2vv'`, where `v'1 = 0`. They are then partitioned
/// greedily into power-of-two blocks no larger than
/// [`COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS`] and transformed by
/// `Qb = H D H / B` in each block. `H` is the unnormalised Sylvester Hadamard
/// matrix and the first Rademacher sign in `D` is fixed to +1; Walsh-character
/// sign patterns that would reduce `Qb` to a row permutation are excluded for
/// `B >= 4`. Thus the composite `Qb Q0` is genuinely mixing even for binary
/// tail blocks, while `Q'Q = I` and `Q·1 = 1` preserve the mean and centered
/// cross-product in exact arithmetic. With fewer than three rows no
/// non-permutation mean-fixing orthogonal map exists, so the control refuses.
/// Row-block/column-band tiles run independently with sequential butterfly
/// arithmetic. Peak transform storage is at most
/// `min(8, Rayon threads) × B × p` float64 scalars and never exceeds the
/// 128-MiB active-workspace budget, plus `O(n + p)` for the Householder vector
/// and its column projections; column banding makes each tile at most 32 MiB.
pub fn covariance_exact_hadamard_null(
    data: ArrayView2<'_, f64>,
    seed: u64,
) -> Result<Array2<f64>, String> {
    validate_matrix(data, "covariance-exact Hadamard input")?;
    covariance_exact_hadamard_null_impl(data, seed)
}

/// Float32-output form of [`covariance_exact_hadamard_null`].
///
/// The bounded per-task transform workspaces use float64 arithmetic, but no
/// corpus-sized float64 matrix is ever allocated; input scalars are widened
/// block by block and output scalars are narrowed directly into the `n × p`
/// float32 result.
pub fn covariance_exact_hadamard_null_f32(
    data: ArrayView2<'_, f32>,
    seed: u64,
) -> Result<Array2<f32>, String> {
    validate_matrix_f32(data, "float32 covariance-exact Hadamard input")?;
    covariance_exact_hadamard_null_impl(data, seed)
}

/// Gaussian matched-spectrum null for a matrix already expressed in principal-
/// component coordinates.
///
/// Each output PC column is an independent normal draw with the observed
/// column's mean and population standard deviation. Consequently the generated
/// population matches the per-PC variance targets while a finite draw has the
/// corresponding Monte Carlo fluctuation; cyclic ordering, higher moments, and
/// cross-row manifold structure are destroyed. The explicit PC-coordinate
/// contract avoids a hidden eigen-decomposition or a Python-side reimplementation.
pub fn matched_spectrum_gaussian_null(
    pc_scores: ArrayView2<'_, f64>,
    seed: u64,
) -> Result<Array2<f64>, String> {
    validate_matrix(pc_scores, "matched-spectrum PC scores")?;
    let location = stable_column_location(pc_scores)?;
    let sd = stable_column_sd(pc_scores, &location)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Array2::<f64>::zeros(pc_scores.raw_dim());
    for row in 0..out.nrows() {
        for pc in 0..out.ncols() {
            let centered_draw = sd[pc] * standard_normal(&mut rng);
            out[[row, pc]] = location
                .compose_centered(centered_draw, pc)
                .map_err(|error| {
                    format!(
                        "matched-spectrum Gaussian draw failed at row {row}, column {pc}: {error}"
                    )
                })?;
        }
    }
    Ok(out)
}

struct StableColumnLocation {
    mean: Array1<f64>,
    origin: Array1<f64>,
    offset_scale: Array1<f64>,
    normalized_offset: Array1<f64>,
}

impl StableColumnLocation {
    fn absolute_mean(&self) -> Result<Array1<f64>, String> {
        if self.mean.iter().any(|value| !value.is_finite()) {
            return Err("column mean overflowed in absolute coordinates".to_string());
        }
        Ok(self.mean.clone())
    }

    fn centered_value(&self, value: f64, axis: usize) -> Result<f64, String> {
        let relative = value - self.origin[axis];
        let centered = (-self.normalized_offset[axis]).mul_add(self.offset_scale[axis], relative);
        if !centered.is_finite() {
            return Err(format!(
                "centered value is not representable on column {axis}"
            ));
        }
        Ok(centered)
    }

    fn compose_centered(&self, centered: f64, axis: usize) -> Result<f64, String> {
        if !centered.is_finite() {
            return Err(format!(
                "centered draw is non-finite on column {axis}: {centered}"
            ));
        }
        let composition_scale = self.offset_scale[axis].max(centered.abs());
        let value = if composition_scale == 0.0 {
            self.origin[axis]
        } else {
            let normalized_total = self.normalized_offset[axis]
                * (self.offset_scale[axis] / composition_scale)
                + centered / composition_scale;
            normalized_total.mul_add(composition_scale, self.origin[axis])
        };
        if !value.is_finite() {
            return Err(format!(
                "absolute value is not representable on column {axis}"
            ));
        }
        Ok(value)
    }
}

fn neumaier_add(sum: &mut f64, correction: &mut f64, term: f64) -> Result<(), String> {
    let next = *sum + term;
    if !next.is_finite() {
        return Err("compensated accumulation overflowed".to_string());
    }
    if (*sum).abs() >= term.abs() {
        *correction += (*sum - next) + term;
    } else {
        *correction += (term - next) + *sum;
    }
    *sum = next;
    if !(*correction).is_finite() {
        return Err("compensated accumulation correction overflowed".to_string());
    }
    Ok(())
}

/// Shared range-safe column location. When subtraction from the first row is
/// representable, a compensated residual-chart average preserves small
/// variations around a large translation. Otherwise a compensated convex
/// average in a max-absolute unit chart handles antipodal finite values without
/// ever forming their unrepresentable difference. Downstream centering fails
/// only when the physical-coordinate residual itself is not representable.
fn stable_column_location(data: ArrayView2<'_, f64>) -> Result<StableColumnLocation, String> {
    validate_matrix(data, "stable-column-location input")?;
    let count = data.nrows() as f64;
    let mut mean = Array1::<f64>::zeros(data.ncols());
    let mut chart_origin = Array1::<f64>::zeros(data.ncols());
    let mut offset_scale = Array1::<f64>::zeros(data.ncols());
    let mut normalized_offset = Array1::<f64>::zeros(data.ncols());
    for axis in 0..data.ncols() {
        let origin = data[[0, axis]];
        let relative_chart_is_representable = data
            .column(axis)
            .iter()
            .all(|&value| (value - origin).is_finite());
        let mut sum = 0.0_f64;
        let mut correction = 0.0_f64;
        if relative_chart_is_representable {
            let residual_scale = data
                .column(axis)
                .iter()
                .map(|&value| (value - origin).abs())
                .fold(0.0_f64, f64::max);
            if residual_scale == 0.0 {
                mean[axis] = origin;
                chart_origin[axis] = origin;
                continue;
            }
            for &value in data.column(axis).iter() {
                neumaier_add(&mut sum, &mut correction, (value - origin) / residual_scale)?;
            }
            let axis_normalized_offset = (sum + correction) / count;
            // Fusing the scale restoration with the translated origin avoids
            // double-rounding a subnormal offset before it can change the
            // correctly rounded absolute mean.
            mean[axis] = axis_normalized_offset.mul_add(residual_scale, origin);
            chart_origin[axis] = origin;
            offset_scale[axis] = residual_scale;
            normalized_offset[axis] = axis_normalized_offset;
        } else {
            let absolute_scale = data
                .column(axis)
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            if absolute_scale == 0.0 {
                mean[axis] = 0.0;
                continue;
            }
            for &value in data.column(axis).iter() {
                neumaier_add(&mut sum, &mut correction, value / absolute_scale)?;
            }
            let axis_normalized_offset = (sum + correction) / count;
            mean[axis] = axis_normalized_offset * absolute_scale;
            offset_scale[axis] = absolute_scale;
            normalized_offset[axis] = axis_normalized_offset;
        }
    }
    if mean.iter().any(|value| !value.is_finite()) {
        return Err("stable column-mean accumulation overflowed".to_string());
    }
    Ok(StableColumnLocation {
        mean,
        origin: chart_origin,
        offset_scale,
        normalized_offset,
    })
}

struct StablePopulationMoments {
    location: StableColumnLocation,
    covariance: Array2<f64>,
}

/// Range- and translation-stable two-pass population moments. Coordinates are
/// centered through the range-safe location above, preventing both antipodal-
/// range overflow and a large location from erasing the residual covariance.
fn stable_population_moments(data: ArrayView2<'_, f64>) -> Result<StablePopulationMoments, String> {
    let location = stable_column_location(data)?;
    let n = data.nrows();
    let p = data.ncols();
    let mut residual_scales = vec![0.0_f64; p];
    for row in 0..n {
        for axis in 0..p {
            residual_scales[axis] =
                residual_scales[axis].max(location.centered_value(data[[row, axis]], axis)?.abs());
        }
    }
    let mut covariance = Array2::<f64>::zeros((p, p));
    let mut covariance_correction = Array2::<f64>::zeros((p, p));
    let mut normalized_residuals = vec![0.0_f64; p];
    for row in 0..n {
        for axis in 0..p {
            normalized_residuals[axis] = if residual_scales[axis] > 0.0 {
                location.centered_value(data[[row, axis]], axis)? / residual_scales[axis]
            } else {
                0.0
            };
        }
        for left in 0..p {
            for right in 0..=left {
                neumaier_add(
                    &mut covariance[[left, right]],
                    &mut covariance_correction[[left, right]],
                    normalized_residuals[left] * normalized_residuals[right],
                )?;
            }
        }
    }
    for left in 0..p {
        for right in 0..=left {
            let normalized_covariance =
                (covariance[[left, right]] + covariance_correction[[left, right]]) / n as f64;
            let value = balanced_finite_product(
                normalized_covariance,
                residual_scales[left],
                residual_scales[right],
                "population covariance entry",
            )?;
            covariance[[left, right]] = value;
            covariance[[right, left]] = value;
        }
    }
    if covariance.iter().any(|value| !value.is_finite()) {
        return Err("stable mean/covariance accumulation overflowed".to_string());
    }
    Ok(StablePopulationMoments {
        location,
        covariance,
    })
}

/// Certify that a symmetric covariance eigenspectrum is positive semidefinite
/// up to the backward error of its eigendecomposition. A negative eigenvalue
/// no larger than `64 * p * eps * trace` in magnitude is numerical zero; a
/// value beyond that scale-relative envelope is a material indefiniteness and
/// must not be silently projected into a different covariance.
fn certified_covariance_spectrum(
    eigenvalues: ArrayView1<'_, f64>,
    covariance_trace: f64,
) -> Result<Vec<f64>, String> {
    if !covariance_trace.is_finite() || covariance_trace < 0.0 {
        return Err(format!(
            "covariance trace must be finite and non-negative; got {covariance_trace}"
        ));
    }
    let tolerance = 64.0 * eigenvalues.len() as f64 * f64::EPSILON * covariance_trace;
    let mut certified = Vec::with_capacity(eigenvalues.len());
    for (axis, &eigenvalue) in eigenvalues.iter().enumerate() {
        if !eigenvalue.is_finite() {
            return Err(format!(
                "covariance eigenvalue {axis} is non-finite: {eigenvalue}"
            ));
        }
        if eigenvalue < -tolerance {
            return Err(format!(
                "covariance is materially indefinite: eigenvalue {axis}={eigenvalue:.6e} is below the numerical PSD tolerance -{tolerance:.6e}"
            ));
        }
        certified.push(if eigenvalue < 0.0 { 0.0 } else { eigenvalue });
    }
    Ok(certified)
}

/// Gaussian null matched to the observed mean and full covariance.
///
/// The covariance square root is formed from the symmetric eigendecomposition,
/// not by adding a ridge: zero empirical eigenvalues stay exactly zero, so a
/// rank-deficient coordinate cloud produces a rank-deficient matched null
/// rather than a different full-rank distribution. A scale-relative backward-
/// error certificate distinguishes numerical negative zero from a materially
/// indefinite spectrum before any projection is allowed. The eigensystem is
/// evaluated after dividing by the largest covariance entry, so a finite
/// covariance remains admissible even when its trace or leading eigenvalue is
/// larger than `f64::MAX`; the square-root scale is restored only after the
/// decomposition.
pub fn covariance_matched_gaussian_null(
    data: ArrayView2<'_, f64>,
    seed: u64,
) -> Result<Array2<f64>, String> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    validate_matrix(data, "covariance-matched Gaussian input")?;
    let n = data.nrows();
    let p = data.ncols();
    let StablePopulationMoments {
        location,
        covariance: mut normalized_covariance,
    } = stable_population_moments(data)?;
    let covariance_scale = normalized_covariance
        .iter()
        .fold(0.0_f64, |scale, &value| scale.max(value.abs()));
    if covariance_scale > 0.0 {
        normalized_covariance.mapv_inplace(|value| value / covariance_scale);
    }
    let normalized_trace = normalized_covariance.diag().sum();
    let (eigenvalues, eigenvectors) = normalized_covariance.eigh(Side::Lower).map_err(|error| {
        format!("covariance-matched Gaussian eigendecomposition failed: {error}")
    })?;
    let eigenvalues = certified_covariance_spectrum(eigenvalues.view(), normalized_trace)?;
    let covariance_root_scale = covariance_scale.sqrt();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Array2::<f64>::zeros((n, p));
    let mut scaled_normal = vec![0.0_f64; p];
    for row in 0..n {
        for axis in 0..p {
            scaled_normal[axis] =
                eigenvalues[axis].sqrt() * covariance_root_scale * standard_normal(&mut rng);
        }
        for col in 0..p {
            let mut centered_draw = 0.0_f64;
            for axis in 0..p {
                centered_draw += eigenvectors[[col, axis]] * scaled_normal[axis];
            }
            let draw = location
                .compose_centered(centered_draw, col)
                .map_err(|error| {
                    format!(
                        "covariance-matched Gaussian draw failed at row {row}, column {col}: {error}"
                    )
                })?;
            out[[row, col]] = draw;
        }
    }
    Ok(out)
}

/// Resample real random-weight activations after mapping donor columns to the
/// observed mean/scale parameters. Finite resampling still has ordinary Monte
/// Carlo fluctuation around those targets.
pub fn architecture_matched_random_weight_null(
    observed: ArrayView2<'_, f64>,
    random_weight: ArrayView2<'_, f64>,
    seed: u64,
) -> Result<Array2<f64>, String> {
    validate_matrix(observed, "observed")?;
    validate_matrix(random_weight, "random_weight")?;
    if random_weight.ncols() != observed.ncols() {
        return Err(format!(
            "random_weight ncols {} != observed ncols {}",
            random_weight.ncols(),
            observed.ncols()
        ));
    }
    let n = observed.nrows();
    let p = observed.ncols();
    let mut rng = StdRng::seed_from_u64(seed);
    let observed_location = stable_column_location(observed)?;
    let obs_sd = stable_column_sd(observed, &observed_location)?;
    let random_weight_location = stable_column_location(random_weight)?;
    let rw_sd = stable_column_sd(random_weight, &random_weight_location)?;
    for axis in 0..p {
        if rw_sd[axis] == 0.0 && obs_sd[axis] > 0.0 {
            return Err(format!(
                "architecture-matched donor column {axis} has zero scale but observed scale is {}",
                obs_sd[axis]
            ));
        }
    }
    let mut out = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let src = rng.random_range(0..random_weight.nrows());
        for j in 0..p {
            let centered = random_weight_location.centered_value(random_weight[[src, j]], j)?;
            let scaled = if rw_sd[j] > 0.0 {
                (centered / rw_sd[j]) * obs_sd[j]
            } else {
                // Both target and donor are structurally constant on this
                // axis, as certified above.
                0.0
            };
            out[[i, j]] = observed_location
                .compose_centered(scaled, j)
                .map_err(|error| {
                    format!("architecture-matched draw failed at row {i}, column {j}: {error}")
                })?;
        }
    }
    Ok(out)
}

/// Bootstrap a realistic residual-noise matrix from post-sink-peel activations.
///
/// Rows are sampled with replacement after centering, so the calibration sees the
/// empirical residual covariance, marginal tails, and row-level outliers rather
/// than an idealized Gaussian null.
pub fn empirical_residual_bootstrap(
    residuals: ArrayView2<'_, f64>,
    seed: u64,
) -> Result<Array2<f64>, String> {
    validate_matrix(residuals, "empirical residual bootstrap input")?;
    let n = residuals.nrows();
    let p = residuals.ncols();
    let location = stable_column_location(residuals)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let src = rng.random_range(0..n);
        for j in 0..p {
            out[[i, j]] = location.centered_value(residuals[[src, j]], j)?;
        }
    }
    Ok(out)
}

/// Build a covariance- and kurtosis-matched residual surrogate from explicit
/// moments when real residual activations cannot be supplied to the harness.
pub fn residual_surrogate_from_moments(
    spec: &ResidualMomentSpec,
    seed: u64,
) -> Result<Array2<f64>, String> {
    if spec.rows == 0 {
        return Err("residual surrogate requires at least one row".to_string());
    }
    if spec.mean.is_empty() {
        return Err("residual surrogate mean must be non-empty".to_string());
    }
    if spec.covariance.nrows() != spec.mean.len() || spec.covariance.ncols() != spec.mean.len() {
        return Err(format!(
            "residual surrogate covariance shape {}x{} does not match mean length {}",
            spec.covariance.nrows(),
            spec.covariance.ncols(),
            spec.mean.len()
        ));
    }
    if !spec.excess_kurtosis.is_finite() || spec.excess_kurtosis < 0.0 {
        return Err(format!(
            "residual surrogate excess_kurtosis must be finite and non-negative, got {}",
            spec.excess_kurtosis
        ));
    }
    for (idx, &value) in spec.mean.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "residual surrogate mean[{idx}] is not finite: {value}"
            ));
        }
    }
    validate_matrix(spec.covariance.view(), "residual surrogate covariance")?;

    let chol = cholesky_lower(spec.covariance.view())?;
    let p = spec.mean.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Array2::<f64>::zeros((spec.rows, p));
    for i in 0..spec.rows {
        let radial = standardized_lognormal_radial(spec.excess_kurtosis, &mut rng);
        let mut z = Array1::<f64>::zeros(p);
        for j in 0..p {
            z[j] = standard_normal(&mut rng);
        }
        let correlated = chol.dot(&z);
        for j in 0..p {
            out[[i, j]] = spec.mean[j] + radial * correlated[j];
        }
    }
    Ok(out)
}

/// Draw a surrogate matrix using moments estimated from a residual donor.
pub fn residual_surrogate_matching(
    residuals: ArrayView2<'_, f64>,
    seed: u64,
) -> Result<Array2<f64>, String> {
    validate_matrix(residuals, "residual surrogate donor")?;
    let spec = residual_moment_spec(residuals)?;
    residual_surrogate_from_moments(&spec, seed)
}

/// Estimate the moment specification used by [`residual_surrogate_from_moments`].
pub fn residual_moment_spec(residuals: ArrayView2<'_, f64>) -> Result<ResidualMomentSpec, String> {
    validate_matrix(residuals, "residual moment donor")?;
    let StablePopulationMoments {
        location,
        covariance: mut covariance,
    } = stable_population_moments(residuals)?;
    let mean = location.absolute_mean()?;
    let n = residuals.nrows();
    if n > 1 {
        let sample_factor = n as f64 / (n - 1) as f64;
        covariance.mapv_inplace(|value| value * sample_factor);
        if covariance.iter().any(|value| !value.is_finite()) {
            return Err("residual sample covariance is not representable in float64".to_string());
        }
    }
    let mut kurtosis_sum = 0.0_f64;
    let mut kurtosis_count = 0usize;
    for j in 0..residuals.ncols() {
        let mut scale = 0.0_f64;
        for &value in residuals.column(j) {
            scale = scale.max(location.centered_value(value, j)?.abs());
        }
        if scale == 0.0 {
            continue;
        }
        let mut second = 0.0_f64;
        let mut second_correction = 0.0_f64;
        let mut fourth = 0.0_f64;
        let mut fourth_correction = 0.0_f64;
        for i in 0..residuals.nrows() {
            let v = location.centered_value(residuals[[i, j]], j)? / scale;
            let v2 = v * v;
            neumaier_add(&mut second, &mut second_correction, v2)?;
            neumaier_add(&mut fourth, &mut fourth_correction, v2 * v2)?;
        }
        second = (second + second_correction) / n as f64;
        fourth = (fourth + fourth_correction) / n as f64;
        if second > 0.0 {
            kurtosis_sum += (fourth / (second * second) - 3.0).max(0.0);
            kurtosis_count += 1;
        }
    }
    let excess_kurtosis = if kurtosis_count > 0 {
        kurtosis_sum / kurtosis_count as f64
    } else {
        0.0
    };
    Ok(ResidualMomentSpec {
        rows: residuals.nrows(),
        mean,
        covariance,
        excess_kurtosis,
    })
}

/// Inject a circle into a random two-plane of the supplied residual/noise matrix.
///
/// `snr` is signal RMS divided by the input matrix RMS. The returned matrix keeps
/// the original residuals and adds the synthetic ground-truth circle.
pub fn inject_circle_spike(
    noise: ArrayView2<'_, f64>,
    snr: f64,
    seed: u64,
) -> Result<Array2<f64>, String> {
    validate_matrix(noise, "spike-in noise")?;
    if !snr.is_finite() || snr < 0.0 {
        return Err(format!("snr must be finite and non-negative, got {snr}"));
    }
    let n = noise.nrows();
    let p = noise.ncols();
    if p < 2 {
        return Err("circle spike-in requires at least two columns".to_string());
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let basis = random_orthogonal(p, seed ^ 0xA53A_9E1D_2C6B_7F40)?;
    let rms = matrix_rms(noise)?;
    let amp = balanced_nonnegative_product(snr, rms, (p as f64).sqrt(), "circle spike amplitude")?;
    let phase0 = rng.random_range(0.0..(2.0 * PI));
    let mut out = noise.to_owned();
    for i in 0..n {
        let theta = phase0 + 2.0 * PI * (i as f64) / (n as f64);
        for j in 0..p {
            let direction = theta.cos() * basis[[j, 0]] + theta.sin() * basis[[j, 1]];
            let value = amp.mul_add(direction, noise[[i, j]]);
            if !value.is_finite() {
                return Err(format!(
                    "circle spike output is not representable at row {i}, column {j}"
                ));
            }
            out[[i, j]] = value;
        }
    }
    Ok(out)
}

/// Inject a product torus into a random four-plane of the supplied residual
/// matrix at controlled RMS SNR.
pub fn inject_torus_spike(
    noise: ArrayView2<'_, f64>,
    snr: f64,
    seed: u64,
) -> Result<Array2<f64>, String> {
    validate_matrix(noise, "torus spike-in noise")?;
    if !snr.is_finite() || snr < 0.0 {
        return Err(format!("snr must be finite and non-negative, got {snr}"));
    }
    let n = noise.nrows();
    let p = noise.ncols();
    if p < 4 {
        return Err("torus spike-in requires at least four columns".to_string());
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let basis = random_orthogonal(p, seed ^ 0x7055_1A7E_4D2C_A11B)?;
    let rms = matrix_rms(noise)?;
    let amp =
        balanced_nonnegative_product(snr, rms, ((p as f64) / 2.0).sqrt(), "torus spike amplitude")?;
    let phase0 = rng.random_range(0.0..(2.0 * PI));
    let phase1 = rng.random_range(0.0..(2.0 * PI));
    let winding = coprime_winding(n);
    let mut out = noise.to_owned();
    for i in 0..n {
        let theta = phase0 + 2.0 * PI * (i as f64) / (n as f64);
        let wound_index = ((winding as u128 * i as u128) % n as u128) as usize;
        let phi = phase1 + 2.0 * PI * wound_index as f64 / n as f64;
        for j in 0..p {
            let direction = theta.cos() * basis[[j, 0]]
                + theta.sin() * basis[[j, 1]]
                + phi.cos() * basis[[j, 2]]
                + phi.sin() * basis[[j, 3]];
            let value = amp.mul_add(direction, noise[[i, j]]);
            if !value.is_finite() {
                return Err(format!(
                    "torus spike output is not representable at row {i}, column {j}"
                ));
            }
            out[[i, j]] = value;
        }
    }
    Ok(out)
}

/// Inject the configured synthetic topology at controlled RMS SNR.
pub fn inject_spike(
    noise: ArrayView2<'_, f64>,
    shape: SpikeInShape,
    snr: f64,
    seed: u64,
) -> Result<Array2<f64>, String> {
    match shape {
        SpikeInShape::Circle => inject_circle_spike(noise, snr, seed),
        SpikeInShape::Torus => inject_torus_spike(noise, snr, seed),
    }
}

/// Default detector used by the calibration harness: a block-chart promotion
/// gate followed by a topology audit for the requested planted shape.
pub fn default_spike_in_detection_pipeline(
    data: ArrayView2<'_, f64>,
    shape: SpikeInShape,
) -> Result<DetectionPipelineReport, String> {
    validate_matrix(data, "spike-in detection input")?;
    let rank = shape.signal_rank();
    if data.ncols() < rank {
        return Err(format!(
            "{shape:?} detection requires at least {rank} columns, got {}",
            data.ncols()
        ));
    }
    let (eigenvalues, total_energy) = centered_covariance_spectrum(data, rank + 1)?;
    let statistic = match shape {
        SpikeInShape::Circle => harmonic_circle_detector_stat(data)?,
        SpikeInShape::Torus => {
            let rank_sum = eigenvalues.iter().take(rank).sum::<f64>();
            if total_energy > 0.0 {
                certify_unit_interval(
                    rank_sum / total_energy,
                    data.ncols().saturating_mul(128),
                    "torus rank-energy fraction",
                )?
            } else {
                0.0
            }
        }
    };
    // reporting-only: heuristic promotion floors on the shape detector statistic;
    // they gate the reported promote/hold verdict, not any estimated quantity.
    let promotion_floor = match shape {
        SpikeInShape::Circle => 0.10, // reporting-only
        SpikeInShape::Torus => 0.55,  // reporting-only
    };
    let promoted = statistic >= promotion_floor;
    let topology = match shape {
        SpikeInShape::Circle => circle_topology_audit_from_harmonic_stat(statistic, &eigenvalues),
        SpikeInShape::Torus => topology_audit_from_spectrum(shape, statistic, &eigenvalues),
    };
    let detected = promoted && topology.accepted;
    Ok(DetectionPipelineReport {
        shape,
        statistic,
        promoted,
        topology,
        detected,
    })
}

/// Run the spike-in ROC calibration on residual noise with a caller-supplied
/// detector. The detector should execute the same block-chart promotion and
/// topology audit used by the production claim.
pub fn spike_in_roc_curve<F>(
    residual_noise: ArrayView2<'_, f64>,
    config: &SpikeInRocConfig,
    mut detector: F,
) -> Result<Vec<SpikeInRocPoint>, String>
where
    F: FnMut(ArrayView2<'_, f64>) -> Result<DetectionPipelineReport, String>,
{
    validate_spike_in_config(config)?;
    validate_matrix(residual_noise, "spike-in residual noise")?;
    if residual_noise.ncols() < config.shape.signal_rank() {
        return Err(format!(
            "{:?} spike-in requires at least {} residual columns, got {}",
            config.shape,
            config.shape.signal_rank(),
            residual_noise.ncols()
        ));
    }

    let mut null_stats = Vec::with_capacity(config.trials);
    for trial in 0..config.trials {
        let noise = draw_spike_in_noise(
            residual_noise,
            config.noise_mode,
            mix_seed(config.seed, 0xA17D_7E57, trial as u64),
        )?;
        let report = detector(noise.view())?;
        require_finite(report.statistic, "null detection statistic")?;
        null_stats.push(report.statistic);
    }
    null_stats.sort_by(|a, b| a.total_cmp(b));
    let thresholds = roc_thresholds(&null_stats, &config.fpr_levels)?;

    let mut curve = Vec::with_capacity(config.snrs.len());
    for &snr in &config.snrs {
        let mut reports = Vec::with_capacity(config.trials);
        let mut stat_sum = 0.0_f64;
        let mut promoted = 0usize;
        let mut topology_accepted = 0usize;
        for trial in 0..config.trials {
            let trial_seed = mix_seed(config.seed, 0, trial as u64);
            let noise = draw_spike_in_noise(residual_noise, config.noise_mode, trial_seed)?;
            let spiked = inject_spike(noise.view(), config.shape, snr, trial_seed ^ 0x5F1E_51A5)?;
            let report = detector(spiked.view())?;
            require_finite(report.statistic, "spike-in detection statistic")?;
            if report.promoted {
                promoted += 1;
            }
            if report.topology.accepted {
                topology_accepted += 1;
            }
            stat_sum += report.statistic;
            reports.push(report);
        }
        let mut roc = Vec::with_capacity(thresholds.len());
        for &(fpr, threshold) in &thresholds {
            let hits = reports
                .iter()
                .filter(|report| {
                    report.promoted && report.topology.accepted && report.statistic > threshold
                })
                .count();
            roc.push(SpikeInRocThreshold {
                false_positive_rate: fpr,
                threshold,
                true_positive_rate: hits as f64 / config.trials as f64,
            });
        }
        curve.push(SpikeInRocPoint {
            snr,
            trials: config.trials,
            mean_stat: stat_sum / config.trials as f64,
            promoted_fraction: promoted as f64 / config.trials as f64,
            topology_accept_fraction: topology_accepted as f64 / config.trials as f64,
            roc,
        });
    }
    Ok(curve)
}

/// Convenience wrapper for the default block-chart/topology detector.
pub fn default_spike_in_roc_curve(
    residual_noise: ArrayView2<'_, f64>,
    config: &SpikeInRocConfig,
) -> Result<Vec<SpikeInRocPoint>, String> {
    let shape = config.shape;
    spike_in_roc_curve(residual_noise, config, |x| {
        default_spike_in_detection_pipeline(x, shape)
    })
}

/// Build a calibrated claim report from a spike-in ROC curve at the requested
/// false-positive-rate operating point.
pub fn calibrated_roc_claim_report(
    claim: impl Into<String>,
    claimed_snr: f64,
    claimed_false_positive_rate: f64,
    nulls: NullBatteryReport,
    spike_in_roc: Vec<SpikeInRocPoint>,
) -> Result<CalibratedRocClaimReport, String> {
    if !claimed_snr.is_finite() || claimed_snr < 0.0 {
        return Err(format!(
            "claimed_snr must be finite and non-negative, got {claimed_snr}"
        ));
    }
    if !claimed_false_positive_rate.is_finite()
        || claimed_false_positive_rate <= 0.0
        || claimed_false_positive_rate >= 1.0
    {
        return Err(format!(
            "claimed_false_positive_rate must be in (0, 1), got {claimed_false_positive_rate}"
        ));
    }

    let mut best_point = None;
    let mut best_distance = f64::INFINITY;
    for point in &spike_in_roc {
        let distance = (point.snr - claimed_snr).abs();
        if distance < best_distance {
            best_distance = distance;
            best_point = Some(point);
        }
    }
    let Some(point) = best_point else {
        return Err("calibrated ROC claim report requires at least one spike-in point".to_string());
    };
    let mut best_power = None;
    let mut best_fpr_distance = f64::INFINITY;
    for threshold in &point.roc {
        let distance = (threshold.false_positive_rate - claimed_false_positive_rate).abs();
        if distance < best_fpr_distance {
            best_fpr_distance = distance;
            best_power = Some(threshold.true_positive_rate);
        }
    }
    let Some(claimed_snr_power) = best_power else {
        return Err("calibrated ROC claim report requires at least one ROC threshold".to_string());
    };
    Ok(CalibratedRocClaimReport {
        claim: claim.into(),
        claimed_snr,
        claimed_false_positive_rate,
        nulls,
        spike_in_roc,
        claimed_snr_power,
    })
}

/// Coordinate-dependent ordered-circle statistic on columns 0 and 1.
pub fn first_two_ordered_circle_stat(data: ArrayView2<'_, f64>) -> Result<f64, String> {
    validate_matrix(data, "first-two circle input")?;
    if data.ncols() < 2 {
        return Err("first-two circle statistic requires at least two columns".to_string());
    }
    let n = data.nrows();
    if n < 4 {
        return Err("first-two circle statistic requires at least four rows".to_string());
    }
    let centered = centered_unit_chart(data.slice(ndarray::s![.., 0..2]))?;
    let x = centered.column(0);
    let y = centered.column(1);
    let mut pos_re = 0.0_f64;
    let mut pos_im = 0.0_f64;
    let mut neg_re = 0.0_f64;
    let mut neg_im = 0.0_f64;
    let mut energy = 0.0_f64;
    let mut signed_area = 0.0_f64;
    for i in 0..n {
        let theta = 2.0 * PI * i as f64 / n as f64;
        let c = theta.cos();
        let s = theta.sin();
        // Frequency-1 coefficient of z = x + i y. A coherent ordered circle puts
        // its energy in one winding direction; independent per-channel phase
        // randomization leaks comparable energy into the opposite winding.
        pos_re += x[i] * c + y[i] * s;
        pos_im += y[i] * c - x[i] * s;
        neg_re += x[i] * c - y[i] * s;
        neg_im += y[i] * c + x[i] * s;
        energy += x[i] * x[i] + y[i] * y[i];
        let j = (i + 1) % n;
        signed_area += x[i] * y[j] - y[i] * x[j];
    }
    if energy == 0.0 {
        return Ok(0.0);
    }
    let pos = pos_re * pos_re + pos_im * pos_im;
    let neg = neg_re * neg_re + neg_im * neg_im;
    let harmonic_energy = pos + neg;
    let winding_balance = if harmonic_energy > 0.0 {
        (pos - neg).max(0.0) / harmonic_energy
    } else {
        0.0
    };
    // Parseval-normalized energy in the ordered frequency-1 mode. A coherent
    // circle attains one; unstructured architecture-matched activations put only
    // O(1/n) of their energy in this single Fourier mode. The previous statistic
    // computed these coefficients but omitted their concentration, allowing a
    // chance high-area noise polygon to outrank the planted circle.
    let harmonic_concentration = certify_unit_interval(
        harmonic_energy / (n as f64 * energy),
        n.saturating_mul(16),
        "ordered-circle harmonic concentration",
    )?;
    let area_scale = certify_unit_interval(
        signed_area.abs() / energy,
        n.saturating_mul(8),
        "ordered-circle signed-area scale",
    )?;
    Ok(winding_balance * harmonic_concentration * area_scale)
}

/// Basis-dependent energy in the first two coordinates.
pub fn first_two_energy_fraction(data: ArrayView2<'_, f64>) -> Result<f64, String> {
    validate_matrix(data, "first-two energy input")?;
    if data.ncols() < 2 {
        return Err("first-two energy requires at least two columns".to_string());
    }
    let centered = centered_unit_chart(data)?;
    let mut total = ScaledSumSquares::default();
    let mut first_two = ScaledSumSquares::default();
    for i in 0..centered.nrows() {
        for j in 0..centered.ncols() {
            total.add(centered[[i, j]])?;
            if j < 2 {
                first_two.add(centered[[i, j]])?;
            }
        }
    }
    if total.scale == 0.0 {
        return Ok(0.0);
    }
    certify_unit_interval(
        scaled_square_ratio(first_two, total)?,
        data.len(),
        "first-two energy fraction",
    )
}

/// Basis-invariant rank-two covariance energy fraction.
pub fn top_two_energy_fraction(data: ArrayView2<'_, f64>) -> Result<f64, String> {
    validate_matrix(data, "top-two energy input")?;
    let (eigenvalues, total) = centered_covariance_spectrum(data, 2)?;
    if total == 0.0 {
        return Ok(0.0);
    }
    certify_unit_interval(
        eigenvalues.iter().sum::<f64>() / total,
        data.ncols().saturating_mul(128),
        "top-two covariance energy fraction",
    )
}

/// Frequency-1 circle detector calibrated by the phase-randomized null.
///
/// The statistic finds the ambient two-plane carrying the strongest ordered
/// first harmonic, then scores whether that plane is dominated by frequency-1
/// energy with balanced quadrature components. Selecting a top-2 variance plane
/// is not enough to score well.
pub fn harmonic_circle_detector_stat(data: ArrayView2<'_, f64>) -> Result<f64, String> {
    validate_matrix(data, "harmonic-circle detector input")?;
    let n = data.nrows();
    let p = data.ncols();
    if n < 4 || p < 2 {
        return Err(
            "harmonic-circle detector requires at least four rows and two columns".to_string(),
        );
    }
    let centered = centered_unit_chart(data)?;
    let mut cos_coeff = Array1::<f64>::zeros(p);
    let mut sin_coeff = Array1::<f64>::zeros(p);
    for i in 0..n {
        let theta = 2.0 * PI * i as f64 / n as f64;
        let c = theta.cos();
        let s = theta.sin();
        for j in 0..p {
            let value = centered[[i, j]];
            cos_coeff[j] += value * c;
            sin_coeff[j] += value * s;
        }
    }

    let mut plane = Vec::with_capacity(2);
    append_unit_residual(&mut plane, cos_coeff)?;
    append_unit_residual(&mut plane, sin_coeff)?;
    if plane.len() < 2 {
        return Ok(0.0);
    }

    let mut total_plane_energy = ScaledSumSquares::default();
    let mut plane_cos = vec![0.0_f64; plane.len()];
    let mut plane_sin = vec![0.0_f64; plane.len()];
    for i in 0..n {
        let theta = 2.0 * PI * i as f64 / n as f64;
        let c = theta.cos();
        let s = theta.sin();
        for (axis, direction) in plane.iter().enumerate() {
            let mut score = 0.0_f64;
            for j in 0..p {
                score += centered[[i, j]] * direction[j];
            }
            total_plane_energy.add(score)?;
            plane_cos[axis] += score * c;
            plane_sin[axis] += score * s;
        }
    }
    if total_plane_energy.scale == 0.0 {
        return Ok(0.0);
    }

    let mut coefficient_energy = ScaledSumSquares::default();
    for &value in plane_cos.iter().chain(plane_sin.iter()) {
        coefficient_energy.add(value)?;
    }
    let scale_ratio = coefficient_energy.scale / total_plane_energy.scale;
    let normalized_scale_ratio = scale_ratio * (2.0 / n as f64).sqrt();
    let harmonic_fraction = certify_unit_interval(
        normalized_scale_ratio
            * normalized_scale_ratio
            * (coefficient_energy.scaled_sum / total_plane_energy.scaled_sum),
        n.saturating_mul(p).saturating_mul(32),
        "harmonic-circle Fourier energy fraction",
    )?;
    let circle_balance = quadrature_balance(&plane_cos, &plane_sin)?;
    Ok(harmonic_fraction * circle_balance)
}

fn validate_spike_in_config(config: &SpikeInRocConfig) -> Result<(), String> {
    if config.trials == 0 {
        return Err("spike-in ROC calibration requires at least one trial".to_string());
    }
    if config.snrs.is_empty() {
        return Err("spike-in ROC calibration requires at least one SNR".to_string());
    }
    for (idx, &snr) in config.snrs.iter().enumerate() {
        if !snr.is_finite() || snr < 0.0 {
            return Err(format!(
                "snr[{idx}] must be finite and non-negative, got {snr}"
            ));
        }
    }
    if config.fpr_levels.is_empty() {
        return Err("spike-in ROC calibration requires at least one FPR level".to_string());
    }
    for (idx, &fpr) in config.fpr_levels.iter().enumerate() {
        if !fpr.is_finite() || fpr <= 0.0 || fpr >= 1.0 {
            return Err(format!("fpr_levels[{idx}] must be in (0, 1), got {fpr}"));
        }
    }
    Ok(())
}

fn draw_spike_in_noise(
    residual_noise: ArrayView2<'_, f64>,
    mode: SpikeInNoiseMode,
    seed: u64,
) -> Result<Array2<f64>, String> {
    match mode {
        SpikeInNoiseMode::EmpiricalResidualBootstrap => {
            empirical_residual_bootstrap(residual_noise, seed)
        }
        SpikeInNoiseMode::PhaseRandomizedResidual => {
            phase_randomized_surrogate(residual_noise, seed)
        }
        SpikeInNoiseMode::CovarianceHeavyTailSurrogate => {
            residual_surrogate_matching(residual_noise, seed)
        }
    }
}

fn roc_thresholds(
    sorted_null_stats: &[f64],
    fpr_levels: &[f64],
) -> Result<Vec<(f64, f64)>, String> {
    if sorted_null_stats.is_empty() {
        return Err("ROC thresholding requires null statistics".to_string());
    }
    let n = sorted_null_stats.len();
    let mut thresholds = Vec::with_capacity(fpr_levels.len());
    for &fpr in fpr_levels {
        let rank_from_top = ((fpr * n as f64).ceil() as usize).max(1);
        let idx = n.saturating_sub(rank_from_top);
        thresholds.push((fpr, sorted_null_stats[idx]));
    }
    Ok(thresholds)
}

fn circle_topology_audit_from_harmonic_stat(
    statistic: f64,
    eigenvalues: &[f64],
) -> TopologyAuditReport {
    let expected = SpikeInShape::Circle.expected_betti();
    let leading = eigenvalues.first().copied().unwrap_or(0.0).max(0.0);
    let second = eigenvalues.get(1).copied().unwrap_or(0.0).max(0.0);
    let tail = eigenvalues.get(2).copied().unwrap_or(0.0).max(0.0);
    let spectral_balance = nonnegative_ratio(second, leading);
    let residual_tail_energy = nonnegative_ratio(tail, second);
    let accepted = statistic >= 0.10; // reporting-only: heuristic acceptance floor for the reported Betti verdict.
    let measured = if accepted { expected } else { (1, 0, 0) };
    TopologyAuditReport {
        expected_betti0: expected.0,
        expected_betti1: expected.1,
        expected_betti2: expected.2,
        measured_betti0: measured.0,
        measured_betti1: measured.1,
        measured_betti2: measured.2,
        rank_energy: statistic,
        spectral_balance,
        residual_tail_energy,
        accepted,
    }
}

fn topology_audit_from_spectrum(
    shape: SpikeInShape,
    rank_energy: f64,
    eigenvalues: &[f64],
) -> TopologyAuditReport {
    let rank = shape.signal_rank();
    let expected = shape.expected_betti();
    let leading = eigenvalues.first().copied().unwrap_or(0.0).max(0.0);
    let rank_tail = eigenvalues.get(rank).copied().unwrap_or(0.0).max(0.0);
    let weakest_signal = eigenvalues
        .get(rank.saturating_sub(1))
        .copied()
        .unwrap_or(0.0)
        .max(0.0);
    let spectral_balance = nonnegative_ratio(weakest_signal, leading);
    let residual_tail_energy = nonnegative_ratio(rank_tail, weakest_signal);
    // reporting-only: heuristic spectral operating points gating the reported
    // topology-audit verdict; not derived and not used as numeric estimates.
    let accepted = match shape {
        SpikeInShape::Circle => {
            // reporting-only
            rank_energy >= 0.45 && spectral_balance >= 0.25 && residual_tail_energy <= 0.75
        }
        SpikeInShape::Torus => {
            // reporting-only
            rank_energy >= 0.55 && spectral_balance >= 0.12 && residual_tail_energy <= 0.80
        }
    };
    let measured = if accepted { expected } else { (1, 0, 0) };
    TopologyAuditReport {
        expected_betti0: expected.0,
        expected_betti1: expected.1,
        expected_betti2: expected.2,
        measured_betti0: measured.0,
        measured_betti1: measured.1,
        measured_betti2: measured.2,
        rank_energy,
        spectral_balance,
        residual_tail_energy,
        accepted,
    }
}

/// Leading eigenvalues and total energy of the centered covariance Gram in a
/// shared unit chart. Both quantities carry the same omitted physical scale,
/// which cancels in every detector ratio.
fn centered_covariance_spectrum(
    data: ArrayView2<'_, f64>,
    count: usize,
) -> Result<(Vec<f64>, f64), String> {
    validate_matrix(data, "leading covariance eigenvalue input")?;
    let centered = centered_unit_chart(data)?;
    let mut total_accumulator = ScaledSumSquares::default();
    for &value in &centered {
        total_accumulator.add(value)?;
    }
    let total = if total_accumulator.scale == 0.0 {
        0.0
    } else {
        total_accumulator.scale * total_accumulator.scale * total_accumulator.scaled_sum
    };
    if !total.is_finite() {
        return Err("unit-chart centered energy is not representable".to_string());
    }
    if count == 0 || total == 0.0 {
        return Ok((Vec::new(), total));
    }
    let mut cov = centered.t().dot(&centered);
    if cov.iter().any(|value| !value.is_finite()) {
        return Err("unit-chart covariance Gram is non-finite".to_string());
    }
    let mut eigenvalues = Vec::with_capacity(count.min(cov.nrows()));
    for idx in 0..count.min(cov.nrows()) {
        let seed = mix_seed(0x5151_0000, idx as u64, cov.nrows() as u64);
        let v = dominant_eigenvector(cov.view(), seed)?;
        let mv = cov.dot(&v);
        let lambda = v.dot(&mv).max(0.0);
        eigenvalues.push(lambda);
        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                cov[[i, j]] -= lambda * v[i] * v[j];
            }
        }
    }
    Ok((eigenvalues, total))
}

fn cholesky_lower(matrix: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "cholesky requires square covariance, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    let p = matrix.nrows();
    let mut lower = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..=i {
            let mut sum = matrix[[i, j]];
            for k in 0..j {
                sum -= lower[[i, k]] * lower[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "covariance is not positive definite at diagonal {i}: pivot {sum}"
                    ));
                }
                lower[[i, j]] = sum.sqrt();
            } else {
                lower[[i, j]] = sum / lower[[j, j]];
            }
        }
    }
    Ok(lower)
}

fn standardized_lognormal_radial(excess_kurtosis: f64, rng: &mut StdRng) -> f64 {
    if excess_kurtosis <= f64::MIN_POSITIVE {
        1.0
    } else {
        // Moment match: for a unit-mean log-normal radius exp(sigma*z - sigma^2/2),
        // the excess kurtosis is a monotone function of sigma^2; this inverts its
        // leading term, so sigma^2 = ln(1 + excess_kurtosis/3)/4 reproduces the
        // requested tail weight (the /3 and /4 are the leading kurtosis coefficients).
        let sigma2 = (1.0 + excess_kurtosis / 3.0).ln() / 4.0;
        let sigma = sigma2.sqrt();
        let z = standard_normal(rng);
        (sigma * z - sigma2).exp()
    }
}

fn coprime_winding(n: usize) -> usize {
    let mut winding = 3usize.min(n.saturating_sub(1).max(1));
    while gcd(winding, n) != 1 {
        winding += 1;
        if winding >= n {
            return 1;
        }
    }
    winding
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn append_unit_residual(
    plane: &mut Vec<Array1<f64>>,
    mut candidate: Array1<f64>,
) -> Result<(), String> {
    // Two modified-Gram-Schmidt passes retain orthogonality when the two
    // harmonic coefficient vectors are nearly collinear.
    for _ in 0..2 {
        for direction in plane.iter() {
            let dot = candidate.dot(direction);
            if !dot.is_finite() {
                return Err(
                    "harmonic-plane orthogonalization produced a non-finite dot product"
                        .to_string(),
                );
            }
            for j in 0..candidate.len() {
                candidate[j] -= dot * direction[j];
            }
        }
    }
    let mut norm_accumulator = ScaledSumSquares::default();
    for &value in &candidate {
        norm_accumulator.add(value)?;
    }
    let norm = norm_accumulator.norm()?;
    if norm > 0.0 {
        candidate.mapv_inplace(|v| v / norm);
        plane.push(candidate);
    }
    Ok(())
}

fn quadrature_balance(cos_coeff: &[f64], sin_coeff: &[f64]) -> Result<f64, String> {
    if cos_coeff.len() != sin_coeff.len() {
        return Err("quadrature coefficient lengths differ".to_string());
    }
    let scale = cos_coeff
        .iter()
        .chain(sin_coeff)
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok(0.0);
    }
    let mut aa = 0.0_f64;
    let mut bb = 0.0_f64;
    for (&c, &s) in cos_coeff.iter().zip(sin_coeff.iter()) {
        let c = c / scale;
        let s = s / scale;
        aa += c * c;
        bb += s * s;
    }
    let trace = aa + bb;
    if trace == 0.0 {
        return Ok(0.0);
    }
    // Lagrange's identity evaluates the Gram determinant as a sum of squares,
    // avoiding the catastrophic `aa*bb - ab^2` subtraction near collinearity.
    let mut determinant = 0.0_f64;
    for left in 0..cos_coeff.len() {
        for right in 0..left {
            let cross = (cos_coeff[left] / scale) * (sin_coeff[right] / scale)
                - (cos_coeff[right] / scale) * (sin_coeff[left] / scale);
            determinant += cross * cross;
        }
    }
    certify_unit_interval(
        2.0 * determinant.sqrt() / trace,
        cos_coeff.len().saturating_mul(32),
        "harmonic quadrature balance",
    )
}

fn nonnegative_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else if numerator == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

fn scaled_square_ratio(
    numerator: ScaledSumSquares,
    denominator: ScaledSumSquares,
) -> Result<f64, String> {
    if denominator.scale == 0.0 {
        return if numerator.scale == 0.0 {
            Ok(0.0)
        } else {
            Err("square ratio has a zero denominator and positive numerator".to_string())
        };
    }
    if numerator.scale == 0.0 {
        return Ok(0.0);
    }
    let scale_ratio = numerator.scale / denominator.scale;
    let ratio = scale_ratio * scale_ratio * numerator.scaled_sum / denominator.scaled_sum;
    if ratio.is_finite() {
        Ok(ratio)
    } else {
        Err("square ratio is not representable in float64".to_string())
    }
}

/// Project a theoretically unit-interval quantity only inside a forward-error
/// envelope derived from the number of accumulated floating-point terms.
/// Values outside that envelope are a broken invariant, not something a clamp
/// may conceal.
fn certify_unit_interval(value: f64, term_count: usize, name: &str) -> Result<f64, String> {
    if !value.is_finite() {
        return Err(format!("{name} is non-finite: {value}"));
    }
    let accumulated = term_count.max(1) as f64 * (0.5 * f64::EPSILON);
    if accumulated >= 0.25 {
        return Err(format!(
            "{name} cannot be certified at float64 precision for {term_count} accumulated terms"
        ));
    }
    let tolerance = 64.0 * accumulated / (1.0 - accumulated);
    if value < -tolerance || value > 1.0 + tolerance {
        return Err(format!(
            "{name} left [0, 1] beyond its {tolerance:.3e} roundoff envelope: {value}"
        ));
    }
    Ok(value.clamp(0.0, 1.0))
}

fn validate_matrix(data: ArrayView2<'_, f64>, name: &str) -> Result<(), String> {
    if data.nrows() == 0 || data.ncols() == 0 {
        return Err(format!("{name} matrix must be non-empty"));
    }
    for ((i, j), &value) in data.indexed_iter() {
        if !value.is_finite() {
            return Err(format!("{name}[{i},{j}] is not finite: {value}"));
        }
    }
    Ok(())
}

fn validate_matrix_f32(data: ArrayView2<'_, f32>, name: &str) -> Result<(), String> {
    if data.nrows() == 0 || data.ncols() == 0 {
        return Err(format!("{name} matrix must be non-empty"));
    }
    for ((i, j), &value) in data.indexed_iter() {
        if !value.is_finite() {
            return Err(format!("{name}[{i},{j}] is not finite: {value}"));
        }
    }
    Ok(())
}

fn require_finite(value: f64, name: &str) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} is not finite: {value}"))
    }
}

fn kind_seed(kind: NullKind) -> u64 {
    match kind {
        NullKind::PhaseRandomized => 0x91E4_11CE,
        NullKind::RandomRotation => 0x807A_7100,
        NullKind::TokenShuffle => 0x70CE_514F,
        NullKind::PerDimensionShuffle => 0xD1AE_510F,
        NullKind::MatchedSpectrumGaussian => 0x5EEC_7A11,
        NullKind::CovarianceExactHadamard => 0x4841_DA4D,
        NullKind::ArchitectureMatchedRandomWeight => 0xA2C4_177E,
    }
}

fn mix_seed(a: u64, b: u64, c: u64) -> u64 {
    let mut x = a ^ b.rotate_left(17) ^ c.rotate_left(41);
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Summarize an explicit Monte Carlo null distribution with the same native
/// plus-one tail calibration used by [`run_null_battery`].
///
/// `samples` remains in draw order in the returned artifact. Quantiles are
/// computed from a separate sorted copy so a seed and draw index can reproduce
/// every persisted statistic exactly.
pub fn summarize_null_distribution(
    kind: NullKind,
    observed: f64,
    samples: Vec<f64>,
    tail: Tail,
) -> Result<NullSummary, String> {
    let calibration = empirical_p_value(observed, &samples, tail)?;
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    let sample_matrix = ArrayView2::from_shape((n, 1), samples.as_slice())
        .map_err(|error| format!("could not construct null-summary sample view: {error}"))?;
    let location = stable_column_location(sample_matrix)?;
    let mean = location.absolute_mean()?[0];
    let mut residual_squares = ScaledSumSquares::default();
    for &sample in &samples {
        residual_squares.add(location.centered_value(sample, 0)?)?;
    }
    let sd = if n > 1 {
        residual_squares.rms(n - 1)?
    } else {
        0.0
    };
    let observed_residual = location.centered_value(observed, 0)?;
    let z = if sd > 0.0 {
        observed_residual / sd
    } else if observed_residual == 0.0 {
        0.0
    } else {
        return Err(
            "null z-score is undefined because a constant null differs from the observation"
                .to_string(),
        );
    };
    if !(mean.is_finite() && sd.is_finite() && z.is_finite()) {
        return Err("null summary moments are not representable in float64".to_string());
    }
    Ok(NullSummary {
        kind,
        tail,
        observed,
        n,
        mean,
        sd,
        min: sorted[0],
        q25: sorted[n / 4],
        median: sorted[n / 2],
        q75: sorted[(3 * n) / 4],
        max: sorted[n - 1],
        z,
        p_value: calibration.p_value,
        monte_carlo_standard_error: calibration.monte_carlo_standard_error,
        extreme_draws: calibration.extreme_draws,
        samples,
    })
}

fn random_orthogonal(p: usize, seed: u64) -> Result<Array2<f64>, String> {
    if p == 0 {
        return Err("random orthogonal matrix needs positive dimension".to_string());
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut q = Array2::<f64>::zeros((p, p));
    for col in 0..p {
        let mut v = Array1::<f64>::zeros(p);
        for i in 0..p {
            v[i] = standard_normal(&mut rng);
        }
        // Reorthogonalized modified Gram-Schmidt (the DGKS remedy). A second
        // pass preserves the Haar construction in exact arithmetic while
        // preventing a nearly dependent residual from leaking covariance
        // under the advertised orthogonal rotation.
        for _ in 0..2 {
            for prev in 0..col {
                let mut dot = 0.0_f64;
                for i in 0..p {
                    dot += v[i] * q[[i, prev]];
                }
                for i in 0..p {
                    v[i] -= dot * q[[i, prev]];
                }
            }
        }
        let mut norm_accumulator = ScaledSumSquares::default();
        for &value in &v {
            norm_accumulator.add(value)?;
        }
        let norm = norm_accumulator.norm()?;
        if norm == 0.0 {
            return Err(format!(
                "random orthogonal construction produced a numerically zero residual at column {col}"
            ));
        }
        for i in 0..p {
            q[[i, col]] = v[i] / norm;
        }
    }
    Ok(q)
}

fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn stable_column_mean(data: ArrayView2<'_, f64>) -> Result<Array1<f64>, String> {
    stable_column_location(data)?.absolute_mean()
}

fn stable_column_sd(
    data: ArrayView2<'_, f64>,
    location: &StableColumnLocation,
) -> Result<Array1<f64>, String> {
    if location.mean.len() != data.ncols() {
        return Err("column-scale location dimension mismatch".to_string());
    }
    let mut accumulators = vec![ScaledSumSquares::default(); data.ncols()];
    for row in data.rows() {
        for j in 0..data.ncols() {
            accumulators[j].add(location.centered_value(row[j], j)?)?;
        }
    }
    let mut sd = Array1::<f64>::zeros(data.ncols());
    for axis in 0..data.ncols() {
        sd[axis] = accumulators[axis].rms(data.nrows())?;
    }
    if sd.iter().any(|value| !value.is_finite()) {
        return Err("stable column-scale accumulation overflowed".to_string());
    }
    Ok(sd)
}

/// LAPACK-LASSQ-style sum of squares represented as
/// `scale^2 * scaled_sum`. No raw square is formed before division by the
/// current scale.
#[derive(Clone, Copy, Debug, Default)]
struct ScaledSumSquares {
    scale: f64,
    scaled_sum: f64,
}

impl ScaledSumSquares {
    fn add(&mut self, value: f64) -> Result<(), String> {
        if !value.is_finite() {
            return Err(format!(
                "scaled sum-of-squares received non-finite value {value}"
            ));
        }
        let magnitude = value.abs();
        if magnitude == 0.0 {
            return Ok(());
        }
        if self.scale < magnitude {
            let ratio = self.scale / magnitude;
            self.scaled_sum = 1.0 + self.scaled_sum * ratio * ratio;
            self.scale = magnitude;
        } else {
            let ratio = magnitude / self.scale;
            self.scaled_sum += ratio * ratio;
        }
        if self.scaled_sum.is_finite() {
            Ok(())
        } else {
            Err("scaled sum-of-squares accumulation overflowed".to_string())
        }
    }

    fn norm(self) -> Result<f64, String> {
        if self.scale == 0.0 {
            return Ok(0.0);
        }
        let norm = self.scale * self.scaled_sum.sqrt();
        if norm.is_finite() {
            Ok(norm)
        } else {
            Err("Euclidean norm is not representable in float64".to_string())
        }
    }

    fn rms(self, count: usize) -> Result<f64, String> {
        if count == 0 {
            return Err("root-mean-square requires at least one value".to_string());
        }
        if self.scale == 0.0 {
            return Ok(0.0);
        }
        let rms = self.scale * (self.scaled_sum / count as f64).sqrt();
        if rms.is_finite() {
            Ok(rms)
        } else {
            Err("root-mean-square is not representable in float64".to_string())
        }
    }
}

/// Center every column and divide the entire residual matrix by its largest
/// absolute entry. Scale-invariant diagnostics can then form products and
/// Gram matrices without ever squaring a physical-coordinate magnitude.
fn centered_unit_chart(data: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let location = stable_column_location(data)?;
    let mut scale = 0.0_f64;
    for row in 0..data.nrows() {
        for col in 0..data.ncols() {
            scale = scale.max(location.centered_value(data[[row, col]], col)?.abs());
        }
    }
    let mut out = Array2::<f64>::zeros(data.raw_dim());
    if scale == 0.0 {
        return Ok(out);
    }
    for row in 0..data.nrows() {
        for col in 0..data.ncols() {
            out[[row, col]] = location.centered_value(data[[row, col]], col)? / scale;
        }
    }
    Ok(out)
}

fn matrix_rms(data: ArrayView2<'_, f64>) -> Result<f64, String> {
    let count = data
        .nrows()
        .checked_mul(data.ncols())
        .ok_or_else(|| "matrix RMS element count overflowed usize".to_string())?;
    let mut accumulator = ScaledSumSquares::default();
    for &value in data {
        accumulator.add(value)?;
    }
    accumulator.rms(count)
}

/// Multiply three finite nonnegative factors in an order that cannot overflow
/// or underflow before a representable final product. Pairing the smallest and
/// largest factor first balances exponents; with `a <= b <= c`, an overflow in
/// `a*c` implies `a >= 1` and hence the final product is also unrepresentable,
/// while an underflow to zero implies `c < 1` and later multiplication cannot
/// restore a representable value.
fn balanced_nonnegative_product(
    first: f64,
    second: f64,
    third: f64,
    name: &str,
) -> Result<f64, String> {
    let factors = [first, second, third];
    if factors
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(format!(
            "{name} factors must be finite and nonnegative; got {factors:?}"
        ));
    }
    let product = balanced_finite_product(first, second, third, name)?;
    if product == 0.0 && factors.iter().all(|value| *value > 0.0) {
        Err(format!(
            "{name} is positive but underflows the float64 output domain"
        ))
    } else {
        Ok(product)
    }
}

fn balanced_finite_product(first: f64, second: f64, third: f64, name: &str) -> Result<f64, String> {
    let factors = [first, second, third];
    if factors.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} factors must be finite; got {factors:?}"));
    }
    let negative = factors
        .iter()
        .filter(|value| value.is_sign_negative())
        .count()
        % 2
        == 1;
    let mut magnitudes = factors.map(f64::abs);
    magnitudes.sort_by(f64::total_cmp);
    let magnitude = (magnitudes[0] * magnitudes[2]) * magnitudes[1];
    if !magnitude.is_finite() {
        return Err(format!("{name} is not representable in float64"));
    }
    Ok(if negative { -magnitude } else { magnitude })
}

fn dominant_eigenvector(matrix: ArrayView2<'_, f64>, seed: u64) -> Result<Array1<f64>, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err("dominant eigenvector requires a square matrix".to_string());
    }
    let p = matrix.nrows();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v = Array1::<f64>::zeros(p);
    for i in 0..p {
        v[i] = standard_normal(&mut rng);
    }
    normalize(&mut v);
    for _iter in 0..64 {
        let mut next = matrix.dot(&v);
        let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= f64::MIN_POSITIVE {
            return Ok(v);
        }
        next.mapv_inplace(|x| x / norm);
        v = next;
    }
    Ok(v)
}

fn normalize(v: &mut Array1<f64>) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > f64::MIN_POSITIVE {
        v.mapv_inplace(|x| x / norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary_with_primary_metrics(kind: NullKind, p_value: f64, z: f64) -> NullSummary {
        NullSummary {
            kind,
            tail: Tail::Larger,
            observed: 1.0,
            n: 1,
            mean: 0.0,
            sd: 1.0,
            min: 0.0,
            q25: 0.0,
            median: 0.0,
            q75: 0.0,
            max: 0.0,
            z,
            p_value,
            monte_carlo_standard_error: 0.0,
            extreme_draws: 0,
            samples: vec![0.0],
        }
    }

    fn assert_population_moments_close(
        expected: ArrayView2<'_, f64>,
        actual: ArrayView2<'_, f64>,
        mean_relative_tolerance: f64,
        covariance_relative_tolerance: f64,
    ) {
        let expected = stable_population_moments(expected).unwrap();
        let actual = stable_population_moments(actual).unwrap();
        let expected_mean = expected.location.absolute_mean().unwrap();
        let actual_mean = actual.location.absolute_mean().unwrap();
        for axis in 0..expected_mean.len() {
            let scale = expected_mean[axis].abs().max(1.0);
            assert!(
                (actual_mean[axis] - expected_mean[axis]).abs() <= mean_relative_tolerance * scale,
                "mean {axis} mismatch: expected={} actual={} scale={scale}",
                expected_mean[axis],
                actual_mean[axis]
            );
        }
        for left in 0..expected.covariance.nrows() {
            for right in 0..expected.covariance.ncols() {
                let scale = (expected.covariance[[left, left]].sqrt()
                    * expected.covariance[[right, right]].sqrt())
                .max(1.0);
                assert!(
                    (actual.covariance[[left, right]] - expected.covariance[[left, right]]).abs()
                        <= covariance_relative_tolerance * scale,
                    "covariance ({left},{right}) mismatch: expected={} actual={} scale={scale}",
                    expected.covariance[[left, right]],
                    actual.covariance[[left, right]]
                );
            }
        }
    }

    #[test]
    fn mp_reconstruction_rank_edge_matches_recon_spectrum() {
        use ndarray::array;

        let gram = array![[3.0, 0.5], [0.5, 2.0]];
        let decoder = array![[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]];
        let n_eff = 40.0;
        let p_out = 3.0;
        let r_floor = 1.25;

        let spectrum =
            crate::manifold::recon_spectrum(&gram, &decoder, n_eff, p_out, r_floor, 0.0, None)
                .expect("recon_spectrum should succeed on a well-posed Gram");
        let edge = mp_reconstruction_rank_edge(n_eff, p_out, r_floor)
            .expect("mp_reconstruction_rank_edge should succeed on valid inputs");

        assert!(
            (spectrum.mp_reconstruction_rank_edge() - edge).abs() < 1.0e-12,
            "standalone reconstruction-rank edge must match production byte-for-byte: \
             spectrum.edge={} floor={}",
            spectrum.mp_reconstruction_rank_edge(),
            edge
        );
    }

    #[test]
    fn primary_claim_calibration_requires_the_per_dimension_shuffle() {
        let report = NullBatteryReport {
            observed: 1.0,
            summaries: vec![
                summary_with_primary_metrics(NullKind::PhaseRandomized, 0.01, 4.0),
                summary_with_primary_metrics(NullKind::PerDimensionShuffle, 0.40, 0.25),
                // These preserve the relevant unordered/basis-invariant shape
                // and therefore must not determine the primary claim metric.
                summary_with_primary_metrics(NullKind::TokenShuffle, 0.90, -3.0),
                summary_with_primary_metrics(NullKind::RandomRotation, 0.80, -2.0),
            ],
        };
        assert_eq!(primary_null_metrics(&report).unwrap(), (0.40, 0.25));

        let empty = NullBatteryReport {
            observed: 1.0,
            summaries: Vec::new(),
        };
        assert!(primary_null_metrics(&empty).is_err());

        let non_destroying_only = NullBatteryReport {
            observed: 1.0,
            summaries: vec![
                summary_with_primary_metrics(NullKind::TokenShuffle, 0.01, 4.0),
                summary_with_primary_metrics(NullKind::RandomRotation, 0.02, 3.0),
            ],
        };
        assert!(
            primary_null_metrics(&non_destroying_only).is_err(),
            "controls that preserve the unordered or basis-invariant claim cannot silently become its primary null"
        );
    }

    #[test]
    fn null_summary_moments_do_not_overflow_on_large_finite_location() {
        let samples = vec![1.0e307, 1.0e307 + 2.0e292, 1.0e307 - 2.0e292];
        let summary = summarize_null_distribution(
            NullKind::PerDimensionShuffle,
            1.0e307,
            samples,
            Tail::Larger,
        )
        .unwrap();
        assert!(summary.mean.is_finite());
        assert!(summary.sd.is_finite());
        assert!(summary.z.is_finite());
    }

    #[test]
    fn mp_reconstruction_rank_edge_grows_with_ambient_dimension_and_dispersion() {
        let base = mp_reconstruction_rank_edge(50.0, 10.0, 1.0).unwrap();
        let wider = mp_reconstruction_rank_edge(50.0, 200.0, 1.0).unwrap();
        let noisier = mp_reconstruction_rank_edge(50.0, 10.0, 4.0).unwrap();
        assert!(
            wider > base,
            "a larger ambient dimension should raise the reconstruction-rank edge: base={base} wider={wider}"
        );
        assert!(
            noisier > base,
            "higher dispersion should raise the reconstruction-rank edge: base={base} noisier={noisier}"
        );
    }

    #[test]
    fn mp_reconstruction_rank_edge_rejects_invalid_inputs() {
        assert!(mp_reconstruction_rank_edge(0.0, 10.0, 1.0).is_err());
        assert!(mp_reconstruction_rank_edge(-1.0, 10.0, 1.0).is_err());
        assert!(mp_reconstruction_rank_edge(f64::INFINITY, 10.0, 1.0).is_err());
        assert!(mp_reconstruction_rank_edge(50.0, -1.0, 1.0).is_err());
        assert!(mp_reconstruction_rank_edge(50.0, 10.0, -1.0).is_err());
        assert!(mp_reconstruction_rank_edge(50.0, f64::NAN, 1.0).is_err());
        assert!(mp_reconstruction_rank_edge(50.0, 10.0, f64::INFINITY).is_err());
        assert!(mp_reconstruction_rank_edge(f64::MIN_POSITIVE, f64::MAX, 1.0).is_err());
        assert_eq!(
            mp_reconstruction_rank_edge(f64::MIN_POSITIVE, f64::MAX, 0.0).unwrap(),
            0.0
        );
    }

    #[test]
    fn mp_reconstruction_rank_edge_recovers_after_intermediate_overflow() {
        let n_eff = f64::MIN_POSITIVE;
        let p = f64::MAX;
        let r_floor = 0.25 * n_eff;
        let edge = mp_reconstruction_rank_edge(n_eff, p, r_floor).unwrap();
        let asymptotic = 0.25 * p;
        assert!(edge.is_finite() && edge > 0.0);
        assert!(
            ((edge - asymptotic) / asymptotic).abs() < 1.0e-12,
            "log-domain MP edge reconstruction drifted: edge={edge:e}, expected≈{asymptotic:e}"
        );
    }

    #[test]
    fn structured_signal_beats_required_nulls_but_noise_does_not() {
        let signal = ordered_circle_fixture(96, 6, 0.04);
        let random_weight = noise_fixture(128, 6, 1717);
        let alpha = 0.05_f64;
        // With the plus-one correction, zero exceedances attain 1 / (R + 1).
        // Derive the smallest battery that can strictly clear the declared
        // level instead of weakening alpha to accommodate too few draws.
        let replicates = (1.0_f64 / alpha).floor() as usize;
        let config = NullBatteryConfig {
            replicates,
            seed: 17,
            kinds: vec![
                NullKind::PhaseRandomized,
                NullKind::RandomRotation,
                NullKind::ArchitectureMatchedRandomWeight,
            ],
            tail: Tail::Larger,
        };
        let report = run_null_battery(
            signal.view(),
            Some(random_weight.view()),
            &config,
            first_two_ordered_circle_stat,
        )
        .expect("structured null battery should run");
        assert!(
            report
                .summaries
                .iter()
                .all(|s| s.z > 2.0 && s.p_value < alpha),
            "structured ordered circle should separate from nulls: {:?}",
            report.summaries
        );

        let noise = noise_fixture(96, 6, 99);
        let random_weight_noise = noise_fixture(128, 6, 199);
        let noise_report = run_null_battery(
            noise.view(),
            Some(random_weight_noise.view()),
            &config,
            first_two_ordered_circle_stat,
        )
        .expect("noise null battery should run");
        assert!(
            noise_report
                .summaries
                .iter()
                .all(|s| s.z.abs() < 2.0 || s.p_value >= alpha),
            "pure noise must not look separated from nulls: {:?}",
            noise_report.summaries
        );
    }

    #[test]
    fn random_rotation_kills_basis_stat_but_preserves_invariant_stat() {
        let signal = ordered_circle_fixture(80, 7, 0.02);
        let rotated = random_rotation_null(signal.view(), 44).expect("rotation should succeed");
        let basis_before = first_two_energy_fraction(signal.view()).expect("basis stat");
        let basis_after = first_two_energy_fraction(rotated.view()).expect("rotated basis stat");
        assert!(
            (basis_before - basis_after).abs() > 0.20,
            "basis-dependent statistic should move under rotation: before={basis_before} after={basis_after}"
        );

        let invariant_before = top_two_energy_fraction(signal.view()).expect("invariant stat");
        let invariant_after = top_two_energy_fraction(rotated.view()).expect("rotated invariant");
        assert!(
            (invariant_before - invariant_after).abs() < 1.0e-8,
            "top-two energy should survive rotation: before={invariant_before} after={invariant_after}"
        );
    }

    #[test]
    fn random_rotation_basis_is_numerically_orthogonal() {
        let q = random_orthogonal(64, 0xA11C_E5E5).unwrap();
        let gram = q.t().dot(&q);
        for row in 0..gram.nrows() {
            for col in 0..gram.ncols() {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (gram[[row, col]] - expected).abs() <= 128.0 * f64::EPSILON,
                    "Q'Q mismatch at ({row},{col}): {}",
                    gram[[row, col]]
                );
            }
        }
    }

    #[test]
    fn matched_spectrum_gaussian_preserves_pc_scales_and_is_seeded_2250() {
        let rows = 8_192;
        let mut scores = Array2::<f64>::zeros((rows, 3));
        for row in 0..rows {
            let t = row as f64 / rows as f64;
            scores[[row, 0]] = 2.0 + 3.0 * (2.0 * PI * t).cos();
            scores[[row, 1]] = -1.0 + 1.5 * (4.0 * PI * t).sin();
            scores[[row, 2]] = 0.25 + 0.4 * (6.0 * PI * t).cos();
        }
        let first = matched_spectrum_gaussian_null(scores.view(), 72).expect("matched null");
        let repeated = matched_spectrum_gaussian_null(scores.view(), 72).expect("matched null");
        assert_eq!(first, repeated, "seeded null draws must be reproducible");

        let observed_location = stable_column_location(scores.view()).unwrap();
        let observed_mean = observed_location.absolute_mean().unwrap();
        let observed_sd = stable_column_sd(scores.view(), &observed_location).unwrap();
        let null_location = stable_column_location(first.view()).unwrap();
        let null_mean = null_location.absolute_mean().unwrap();
        let null_sd = stable_column_sd(first.view(), &null_location).unwrap();
        for pc in 0..scores.ncols() {
            let mean_se = observed_sd[pc] / (rows as f64).sqrt();
            assert!(
                (null_mean[pc] - observed_mean[pc]).abs() <= 4.0 * mean_se,
                "PC {pc} mean mismatch: observed={} null={} se={mean_se}",
                observed_mean[pc],
                null_mean[pc]
            );
            assert!(
                (null_sd[pc] / observed_sd[pc] - 1.0).abs() < 0.05,
                "PC {pc} scale mismatch: observed={} null={}",
                observed_sd[pc],
                null_sd[pc]
            );
        }
    }

    #[test]
    fn matched_control_scales_survive_large_and_tiny_representable_units() {
        let observed = ndarray::array![
            [1.0e300, -1.0e300],
            [-1.0e300, 1.0e300],
            [0.5e300, -0.5e300],
            [-0.5e300, 0.5e300],
        ];
        let donor = ndarray::array![
            [1.0e-300, -1.0e-300],
            [-1.0e-300, 1.0e-300],
            [0.5e-300, -0.5e-300],
            [-0.5e-300, 0.5e-300],
        ];
        let observed_location = stable_column_location(observed.view()).unwrap();
        let observed_sd = stable_column_sd(observed.view(), &observed_location).unwrap();
        assert!(
            observed_sd
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
        let matched =
            architecture_matched_random_weight_null(observed.view(), donor.view(), 2262).unwrap();
        assert!(matched.iter().all(|value| value.is_finite()));

        let covariance_scale = 1.0e154_f64;
        let covariance_input = ndarray::array![
            [covariance_scale, 0.0],
            [-covariance_scale, 0.0],
            [0.5 * covariance_scale, 0.0],
            [-0.5 * covariance_scale, 0.0],
        ];
        let moments = stable_population_moments(covariance_input.view()).unwrap();
        assert!(moments.covariance[[0, 0]].is_finite());
        assert!(moments.covariance[[0, 0]] > 0.0);

        let huge_scale = 1.1e154_f64;
        let trace_overflow = ndarray::array![
            [huge_scale, huge_scale],
            [huge_scale, -huge_scale],
            [-huge_scale, huge_scale],
            [-huge_scale, -huge_scale],
        ];
        let huge_moments = stable_population_moments(trace_overflow.view()).unwrap();
        assert!(
            huge_moments
                .covariance
                .diag()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(huge_moments.covariance.diag().sum().is_infinite());
        let huge_draw = covariance_matched_gaussian_null(trace_overflow.view(), 979).unwrap();
        assert!(huge_draw.iter().all(|value| value.is_finite()));

        let finite_limit = 1.7e308_f64;
        let antipodal =
            ndarray::array![[finite_limit, -finite_limit], [-finite_limit, finite_limit],];
        let antipodal_location = stable_column_location(antipodal.view()).unwrap();
        let antipodal_mean = antipodal_location.absolute_mean().unwrap();
        assert!(antipodal_mean.iter().all(|value| value.is_finite()));
        assert!(antipodal_mean.iter().all(|value| value.abs() <= 1.0e292));
        for row in antipodal.rows() {
            for axis in 0..antipodal.ncols() {
                assert!(
                    antipodal_location
                        .centered_value(row[axis], axis)
                        .unwrap()
                        .is_finite()
                );
            }
        }
    }

    #[test]
    fn stable_location_retains_fractional_ulp_offsets_and_composes_in_its_chart() {
        let translated = ndarray::array![[1.0e16], [1.0e16 + 2.0], [1.0e16 + 2.0]];
        let location = stable_column_location(translated.view()).unwrap();
        let residuals = translated
            .column(0)
            .iter()
            .map(|&value| location.centered_value(value, 0).unwrap())
            .collect::<Vec<_>>();
        assert!((residuals[0] + 4.0 / 3.0).abs() <= 4.0 * f64::EPSILON);
        assert!((residuals[1] - 2.0 / 3.0).abs() <= 4.0 * f64::EPSILON);
        assert!((residuals.iter().sum::<f64>()).abs() <= 8.0 * f64::EPSILON);
        let moments = stable_population_moments(translated.view()).unwrap();
        assert!((moments.covariance[[0, 0]] - 8.0 / 9.0).abs() <= 16.0 * f64::EPSILON);

        // A centered draw of -2 crosses to the adjacent representable value;
        // reconstruction through rounded `mean + draw` would lose the chart's
        // fractional offset and land one ULP too low.
        assert_eq!(location.compose_centered(-2.0 / 3.0, 0).unwrap(), 1.0e16);

        let subnormal = f64::from_bits(1);
        let tiny = ndarray::array![[subnormal], [2.0 * subnormal]];
        let tiny_location = stable_column_location(tiny.view()).unwrap();
        assert_eq!(tiny_location.absolute_mean().unwrap()[0].to_bits(), 2);

        let root_subnormal = f64::from_bits(1).sqrt();
        let tiny_variation = ndarray::array![
            [root_subnormal],
            [-root_subnormal],
            [root_subnormal],
            [-root_subnormal],
        ];
        let tiny_moments = stable_population_moments(tiny_variation.view()).unwrap();
        assert!(
            tiny_moments.covariance[[0, 0]] > 0.0,
            "normalizing each residual by sqrt(n) would underflow every term even though the final covariance is representable"
        );
    }

    #[test]
    fn architecture_match_refuses_positive_target_scale_from_constant_donor() {
        let observed = ndarray::array![[-1.0], [0.0], [1.0], [2.0]];
        let donor = ndarray::array![[7.0], [7.0], [7.0], [7.0]];
        let error =
            architecture_matched_random_weight_null(observed.view(), donor.view(), 9).unwrap_err();
        assert!(error.contains("zero scale"), "{error}");
    }

    #[test]
    fn balanced_product_refuses_silent_positive_amplitude_underflow() {
        let subnormal = f64::from_bits(1);
        let error =
            balanced_nonnegative_product(subnormal, 0.5, 0.5, "test amplitude").unwrap_err();
        assert!(error.contains("underflows"), "{error}");
        assert_eq!(
            balanced_nonnegative_product(0.0, f64::MAX, f64::MAX, "zero amplitude").unwrap(),
            0.0
        );
        assert_eq!(
            balanced_nonnegative_product(1.0e-300, 1.0e300, 2.0, "balanced amplitude").unwrap(),
            2.0
        );
    }

    #[test]
    fn per_dimension_shuffle_preserves_marginals_but_breaks_geometry_2262() {
        let rows = 2_048;
        let mut paired = Array2::<f64>::zeros((rows, 2));
        for row in 0..rows {
            let value = row as f64 - (rows as f64 - 1.0) / 2.0;
            paired[[row, 0]] = value;
            paired[[row, 1]] = value;
        }
        let shuffled = per_dimension_shuffle_null(paired.view(), 2262).expect("shuffle null");
        let repeated = per_dimension_shuffle_null(paired.view(), 2262).expect("shuffle null");
        assert_eq!(shuffled, repeated, "seeded shuffle must be reproducible");

        for col in 0..paired.ncols() {
            let mut observed = paired.column(col).to_vec();
            let mut null = shuffled.column(col).to_vec();
            observed.sort_by(f64::total_cmp);
            null.sort_by(f64::total_cmp);
            assert_eq!(observed, null, "column {col} marginal changed");
        }
        let original_cross = paired
            .rows()
            .into_iter()
            .map(|row| row[0] * row[1])
            .sum::<f64>();
        let shuffled_cross = shuffled
            .rows()
            .into_iter()
            .map(|row| row[0] * row[1])
            .sum::<f64>();
        assert!(
            shuffled_cross.abs() < original_cross.abs() / 10.0,
            "independent permutations did not destroy the paired geometry: original={original_cross} shuffled={shuffled_cross}"
        );
    }

    #[test]
    fn covariance_matched_gaussian_preserves_full_second_moment_2262() {
        let rows = 16_384;
        let mut observed = Array2::<f64>::zeros((rows, 3));
        for row in 0..rows {
            let t = 2.0 * PI * row as f64 / rows as f64;
            observed[[row, 0]] = 1.0 + 2.0 * t.cos();
            observed[[row, 1]] = -2.0 + 0.8 * t.cos() + 1.2 * t.sin();
            observed[[row, 2]] = 0.5 - 0.6 * t.cos() + 0.3 * t.sin();
        }
        let draw = covariance_matched_gaussian_null(observed.view(), 2262)
            .expect("covariance-matched Gaussian null");
        let repeated = covariance_matched_gaussian_null(observed.view(), 2262)
            .expect("covariance-matched Gaussian null");
        assert_eq!(draw, repeated, "seeded Gaussian null must be reproducible");

        let empirical_covariance =
            |data: ArrayView2<'_, f64>| stable_population_moments(data).unwrap().covariance;
        let target = empirical_covariance(observed.view());
        let realized = empirical_covariance(draw.view());
        for a in 0..observed.ncols() {
            for b in 0..observed.ncols() {
                let scale = target[[a, a]].sqrt() * target[[b, b]].sqrt();
                assert!(
                    (realized[[a, b]] - target[[a, b]]).abs() < 0.04 * scale,
                    "covariance ({a},{b}) mismatch: target={} realized={} scale={scale}",
                    target[[a, b]],
                    realized[[a, b]],
                );
            }
        }
    }

    #[test]
    fn float32_per_dimension_shuffle_is_seeded_and_preserves_exact_marginals() {
        let rows = 2_048usize;
        let mut observed = Array2::<f32>::zeros((rows, 3));
        for row in 0..rows {
            observed[[row, 0]] = row as f32 - 1_024.0;
            observed[[row, 1]] = (row as f32 * 0.25).sin();
            observed[[row, 2]] = (row % 17) as f32;
        }
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| per_dimension_shuffle_null_f32(observed.view(), 2262).unwrap());
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| per_dimension_shuffle_null_f32(observed.view(), 2262).unwrap());
        assert_eq!(
            one_thread, four_threads,
            "float32 shuffle must be bit-identical across thread counts"
        );
        for col in 0..observed.ncols() {
            let mut expected = observed.column(col).to_vec();
            let mut actual = one_thread.column(col).to_vec();
            expected.sort_by(f32::total_cmp);
            actual.sort_by(f32::total_cmp);
            assert_eq!(actual, expected, "float32 shuffle changed column {col}");
        }
    }

    #[test]
    fn covariance_exact_hadamard_handles_odd_rows_and_is_thread_invariant() {
        let rows = 2_051usize;
        let mut observed = Array2::<f64>::zeros((rows, 4));
        for row in 0..rows {
            let phase = 2.0 * PI * row as f64 / rows as f64;
            observed[[row, 0]] = 3.0 + 2.0 * phase.cos() + 0.1 * (7.0 * phase).sin();
            observed[[row, 1]] = -4.0 + 0.8 * phase.cos() + 1.2 * phase.sin();
            observed[[row, 2]] = 0.5 - 0.6 * phase.cos() + 0.3 * (3.0 * phase).sin();
            observed[[row, 3]] = 7.25;
        }
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| covariance_exact_hadamard_null(observed.view(), 2262).unwrap());
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| covariance_exact_hadamard_null(observed.view(), 2262).unwrap());
        assert_eq!(
            one_thread, four_threads,
            "Hadamard control must be bit-identical across thread counts"
        );
        let different_seed = covariance_exact_hadamard_null(observed.view(), 2263).unwrap();
        assert_ne!(one_thread, different_seed, "seed must change the control");
        assert_eq!(one_thread.dim(), observed.dim());
        assert!(one_thread.column(3).iter().all(|&value| value == 7.25));
        assert_population_moments_close(observed.view(), one_thread.view(), 2.0e-13, 2.0e-12);

        assert_eq!(
            covariance_exact_hadamard_workspace_rows(rows).unwrap(),
            COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS
        );
        let blocks = covariance_exact_hadamard_blocks(rows)
            .unwrap()
            .iter()
            .map(|block| block.rows)
            .collect::<Vec<_>>();
        assert_eq!(blocks, vec![1_024, 1_024, 2, 1]);
    }

    #[test]
    fn covariance_exact_hadamard_parallel_workspace_count_is_bounded() {
        let rows = 32 * COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS + 3;
        let blocks = covariance_exact_hadamard_blocks(rows).unwrap();
        assert_eq!(blocks.len(), 34);
        assert_eq!(blocks[32].rows, 2);
        assert_eq!(blocks[33].rows, 1);

        let columns = 7_000usize;
        let one_plan = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| covariance_exact_hadamard_parallel_plan(rows, columns).unwrap());
        let four_plan = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| covariance_exact_hadamard_parallel_plan(rows, columns).unwrap());
        let sixteen_thread_plan = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap()
            .install(|| covariance_exact_hadamard_parallel_plan(rows, columns).unwrap());
        assert_eq!(one_plan.tasks, 1);
        assert_eq!(four_plan.tasks, 4);
        assert_eq!(sixteen_thread_plan.tasks, 4);
        assert_eq!(one_plan.column_bands, 2);
        assert_eq!(four_plan.column_bands, 2);
        assert_eq!(sixteen_thread_plan.column_bands, 2);

        let per_workspace =
            COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS * sixteen_thread_plan.workspace_cols;
        let bounded_scalars = sixteen_thread_plan.tasks * per_workspace;
        assert_eq!(
            bounded_scalars,
            4 * COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS * columns.div_ceil(2)
        );
        assert!(
            bounded_scalars * std::mem::size_of::<f64>()
                <= COVARIANCE_EXACT_HADAMARD_PARALLEL_WORKSPACE_BUDGET_BYTES
        );
        assert!(bounded_scalars < rows * columns);

        let binary_tail_rows = COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS + 476;
        assert!(
            covariance_exact_hadamard_blocks(binary_tail_rows)
                .unwrap()
                .len()
                > 2
        );
        let binary_tail_plan = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap()
            .install(|| {
                covariance_exact_hadamard_parallel_plan(binary_tail_rows, columns).unwrap()
            });
        assert_eq!(binary_tail_plan.tasks, 8);
        assert_eq!(binary_tail_plan.column_bands, 4);
        assert_eq!(binary_tail_plan.workspace_cols, columns.div_ceil(4));
        let binary_tail_scalars = binary_tail_plan.tasks
            * COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS
            * binary_tail_plan.workspace_cols;
        assert_eq!(
            binary_tail_scalars,
            2 * COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS * columns,
            "column bands must fill workers without exceeding the two-row-group workspace budget"
        );

        let wide_short_plan = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap()
            .install(|| covariance_exact_hadamard_parallel_plan(4_096, 7_168).unwrap());
        assert_eq!(wide_short_plan.tasks, 4);
        assert_eq!(wide_short_plan.column_bands, 2);
        assert_eq!(wide_short_plan.workspace_cols, 3_584);

        let extreme_width_plan = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap()
            .install(|| covariance_exact_hadamard_parallel_plan(1_024, 100_000).unwrap());
        let extreme_tile_bytes = COVARIANCE_EXACT_HADAMARD_MAX_BLOCK_ROWS
            * extreme_width_plan.workspace_cols
            * std::mem::size_of::<f64>();
        assert!(extreme_tile_bytes <= COVARIANCE_EXACT_HADAMARD_TARGET_TILE_BYTES);
        assert!(
            extreme_width_plan.tasks * extreme_tile_bytes
                <= COVARIANCE_EXACT_HADAMARD_PARALLEL_WORKSPACE_BUDGET_BYTES
        );

        let one_block = covariance_exact_hadamard_blocks(17).unwrap();
        let uneven_tiles = covariance_exact_hadamard_tiles(&one_block[..1], 7, 3).unwrap();
        assert_eq!(
            uneven_tiles
                .iter()
                .map(|tile| (tile.first_col, tile.cols))
                .collect::<Vec<_>>(),
            vec![(0, 3), (3, 2), (5, 2)],
            "balanced column bands must cover every column exactly once"
        );
    }

    #[test]
    fn float32_covariance_exact_hadamard_is_native_and_matches_moments() {
        let rows = 4_099usize;
        let cols = 4usize;
        let input = Array2::<f32>::from_shape_fn((rows, cols), |(row, col)| {
            let phase = (2.0 * PI * row as f64 / rows as f64) as f32;
            match col {
                0 => 3.0 + 2.0 * phase.cos(),
                1 => -4.0 + 0.8 * phase.cos() + 1.2 * phase.sin(),
                2 => 0.5 - 0.6 * phase.cos() + 0.3 * (3.0 * phase).sin(),
                _ => 7.25,
            }
        });
        let output = covariance_exact_hadamard_null_f32(input.view(), 17).unwrap();
        let repeated = covariance_exact_hadamard_null_f32(input.view(), 17).unwrap();
        assert_eq!(
            output, repeated,
            "float32 Hadamard seed must replay exactly"
        );
        assert_eq!(output.dim(), input.dim());
        assert!(output.column(3).iter().all(|&value| value == 7.25));
        assert_eq!(
            std::mem::size_of_val(output.as_slice().unwrap()),
            rows * cols * std::mem::size_of::<f32>(),
            "the only n×p output allocation must be native float32"
        );
        let input_f64 = input.mapv(f64::from);
        let output_f64 = output.mapv(f64::from);
        assert_population_moments_close(input_f64.view(), output_f64.view(), 3.0e-6, 3.0e-5);
    }

    #[test]
    fn covariance_exact_hadamard_destroys_constant_radius() {
        let rows = 4_096usize;
        let circle = Array2::<f64>::from_shape_fn((rows, 2), |(row, col)| {
            let phase = 2.0 * PI * row as f64 / rows as f64;
            if col == 0 { phase.cos() } else { phase.sin() }
        });
        let control = covariance_exact_hadamard_null(circle.view(), 2262).unwrap();
        assert_population_moments_close(circle.view(), control.view(), 1.0e-13, 2.0e-12);

        let location = stable_column_location(control.view()).unwrap();
        let radii = control
            .rows()
            .into_iter()
            .map(|row| {
                let x = location.centered_value(row[0], 0).unwrap();
                let y = location.centered_value(row[1], 1).unwrap();
                x.hypot(y)
            })
            .collect::<Vec<_>>();
        let mean_radius = radii.iter().sum::<f64>() / rows as f64;
        let radius_sd = (radii
            .iter()
            .map(|radius| (radius - mean_radius).powi(2))
            .sum::<f64>()
            / rows as f64)
            .sqrt();
        assert!(
            radius_sd / mean_radius > 0.2,
            "Hadamard control retained the circle's constant radius: CV={}",
            radius_sd / mean_radius
        );
    }

    #[test]
    fn covariance_matched_moments_obey_translation_rotation_and_scale_laws() {
        let rows = 4_096usize;
        let mut base = Array2::<f64>::zeros((rows, 3));
        for row in 0..rows {
            let phase = std::f64::consts::TAU * row as f64 / rows as f64;
            base[[row, 0]] = 2.0 * phase.cos() + 0.2 * (3.0 * phase).sin();
            base[[row, 1]] = 0.7 * phase.sin() + 0.4 * base[[row, 0]];
            base[[row, 2]] = 0.3 * (2.0 * phase).cos() - 0.2 * base[[row, 1]];
        }
        let reference = stable_population_moments(base.view()).unwrap();

        let translation = [1.0e12, -2.0e12, 0.5e12];
        let mut translated = base.clone();
        for mut row in translated.rows_mut() {
            for axis in 0..3 {
                row[axis] += translation[axis];
            }
        }
        let shifted = stable_population_moments(translated.view()).unwrap();
        let reference_mean = reference.location.absolute_mean().unwrap();
        let shifted_mean = shifted.location.absolute_mean().unwrap();
        for axis in 0..3 {
            assert!(
                (shifted_mean[axis] - translation[axis] - reference_mean[axis]).abs() < 3.0e-4,
                "translation destabilized mean axis {axis}: reference={} shifted={}",
                reference_mean[axis],
                shifted_mean[axis]
            );
        }
        for left in 0..3 {
            for right in 0..3 {
                assert!(
                    (shifted.covariance[[left, right]] - reference.covariance[[left, right]]).abs()
                        < 3.0e-4,
                    "translation changed covariance ({left},{right})"
                );
            }
        }

        let scale = 1.0e-7_f64;
        let scaled = base.mapv(|value| scale * value);
        let scaled_moments = stable_population_moments(scaled.view()).unwrap();
        let covariance_scale = reference
            .covariance
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        for left in 0..3 {
            for right in 0..3 {
                let expected = scale * scale * reference.covariance[[left, right]];
                let error = (scaled_moments.covariance[[left, right]] - expected).abs();
                assert!(error < 2.0e-13 * scale * scale * covariance_scale);
            }
        }

        let angle = 0.713_f64;
        let (sin_angle, cos_angle) = angle.sin_cos();
        let rotation = ndarray::array![
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let rotated = base.dot(&rotation.t());
        let rotated_moments = stable_population_moments(rotated.view()).unwrap();
        let expected_rotated = rotation.dot(&reference.covariance).dot(&rotation.t());
        for left in 0..3 {
            for right in 0..3 {
                assert!(
                    (rotated_moments.covariance[[left, right]] - expected_rotated[[left, right]])
                        .abs()
                        < 2.0e-12,
                    "orthogonal covariance law failed at ({left},{right})"
                );
            }
        }
    }

    #[test]
    fn production_null_generators_share_stable_large_origin_centering() {
        let rows = 2_048usize;
        let mut observed = Array2::<f64>::zeros((rows, 2));
        let mut donor = Array2::<f64>::zeros((rows, 2));
        for row in 0..rows {
            let phase = std::f64::consts::TAU * row as f64 / rows as f64;
            observed[[row, 0]] = 1.7 * phase.cos() + 0.2 * (3.0 * phase).sin();
            observed[[row, 1]] = 0.5 * observed[[row, 0]] + 0.8 * phase.sin();
            donor[[row, 0]] = 0.9 * (2.0 * phase).cos() + 0.3 * phase.sin();
            donor[[row, 1]] = -0.4 * donor[[row, 0]] + 0.6 * (3.0 * phase).cos();
        }
        let shift = [1.0e12_f64, -2.0e12_f64];
        let donor_shift = [-0.75e12_f64, 1.25e12_f64];
        let mut shifted_observed = observed.clone();
        let mut shifted_donor = donor.clone();
        for row in 0..rows {
            for axis in 0..2 {
                shifted_observed[[row, axis]] += shift[axis];
                shifted_donor[[row, axis]] += donor_shift[axis];
            }
        }

        let spectrum = matched_spectrum_gaussian_null(observed.view(), 71).unwrap();
        let shifted_spectrum = matched_spectrum_gaussian_null(shifted_observed.view(), 71).unwrap();
        let architecture =
            architecture_matched_random_weight_null(observed.view(), donor.view(), 73).unwrap();
        let shifted_architecture = architecture_matched_random_weight_null(
            shifted_observed.view(),
            shifted_donor.view(),
            73,
        )
        .unwrap();
        let bootstrap = empirical_residual_bootstrap(observed.view(), 79).unwrap();
        let shifted_bootstrap = empirical_residual_bootstrap(shifted_observed.view(), 79).unwrap();
        for row in 0..rows {
            for axis in 0..2 {
                assert!(
                    (shifted_spectrum[[row, axis]] - shift[axis] - spectrum[[row, axis]]).abs()
                        < 2.0e-3,
                    "matched-spectrum translation law failed at ({row},{axis})"
                );
                assert!(
                    (shifted_architecture[[row, axis]] - shift[axis] - architecture[[row, axis]])
                        .abs()
                        < 2.0e-3,
                    "architecture-matched translation law failed at ({row},{axis})"
                );
                assert!(
                    (shifted_bootstrap[[row, axis]] - bootstrap[[row, axis]]).abs() < 3.0e-4,
                    "residual-bootstrap centering failed at ({row},{axis})"
                );
            }
        }

        let base_circle = first_two_ordered_circle_stat(observed.view()).unwrap();
        let shifted_circle = first_two_ordered_circle_stat(shifted_observed.view()).unwrap();
        assert!((shifted_circle - base_circle).abs() < 3.0e-4);
        let base_spec = residual_moment_spec(observed.view()).unwrap();
        let shifted_spec = residual_moment_spec(shifted_observed.view()).unwrap();
        for left in 0..2 {
            for right in 0..2 {
                assert!(
                    (shifted_spec.covariance[[left, right]] - base_spec.covariance[[left, right]])
                        .abs()
                        < 3.0e-4
                );
            }
        }
    }

    #[test]
    fn covariance_psd_certificate_rejects_material_negative_spectrum() {
        let trace = 3.0_f64;
        let dimension = 3usize;
        let tolerance = 64.0 * dimension as f64 * f64::EPSILON * trace;
        let roundoff = ndarray::array![-0.5 * tolerance, 1.0, 2.0];
        let certified = certified_covariance_spectrum(roundoff.view(), trace).unwrap();
        assert_eq!(certified[0], 0.0);
        assert_eq!(certified[1..], [1.0, 2.0]);

        let indefinite = ndarray::array![-2.0 * tolerance, 1.0, 2.0];
        let error = certified_covariance_spectrum(indefinite.view(), trace)
            .expect_err("material indefiniteness must not be silently projected");
        assert!(error.contains("materially indefinite"), "{error}");

        let scale_squared = 1.0e-18_f64;
        let scaled = roundoff.mapv(|value| scale_squared * value);
        let scaled_certified =
            certified_covariance_spectrum(scaled.view(), scale_squared * trace).unwrap();
        for axis in 0..dimension {
            assert_eq!(
                scaled_certified[axis],
                scale_squared * certified[axis],
                "PSD certificate must commute with covariance scaling"
            );
        }
    }

    #[test]
    fn topology_statistics_are_invariant_at_extreme_finite_scale() {
        let rows = 128usize;
        let mut base = Array2::<f64>::zeros((rows, 4));
        for row in 0..rows {
            let phase = std::f64::consts::TAU * row as f64 / rows as f64;
            base[[row, 0]] = phase.cos();
            base[[row, 1]] = phase.sin();
            base[[row, 2]] = 0.3 * (3.0 * phase).cos();
            base[[row, 3]] = 0.2 * (5.0 * phase).sin();
        }
        let scale = 1.0e200_f64;
        let shifts = [4.0e200, -3.0e200, 2.0e200, -1.0e200];
        let transformed = Array2::from_shape_fn(base.raw_dim(), |(row, col)| {
            base[[row, col]].mul_add(scale, shifts[col])
        });

        for (name, original, extreme) in [
            (
                "ordered circle",
                first_two_ordered_circle_stat(base.view()).unwrap(),
                first_two_ordered_circle_stat(transformed.view()).unwrap(),
            ),
            (
                "first-two energy",
                first_two_energy_fraction(base.view()).unwrap(),
                first_two_energy_fraction(transformed.view()).unwrap(),
            ),
            (
                "top-two energy",
                top_two_energy_fraction(base.view()).unwrap(),
                top_two_energy_fraction(transformed.view()).unwrap(),
            ),
            (
                "harmonic circle",
                harmonic_circle_detector_stat(base.view()).unwrap(),
                harmonic_circle_detector_stat(transformed.view()).unwrap(),
            ),
        ] {
            assert!(
                (extreme - original).abs() <= 2.0e-12 * (1.0 + original.abs()),
                "{name} changed under a representable translation/scale: original={original}, extreme={extreme}"
            );
        }
    }

    #[test]
    fn spike_in_roc_power_rises_monotonically_and_null_stays_quiet() {
        let noise = noise_fixture(128, 12, 1234);
        let mut config = SpikeInRocConfig::circle(vec![0.0, 1.0, 2.0], 16, 91);
        config.fpr_levels = vec![0.05];
        let curve = default_spike_in_roc_curve(noise.view(), &config).expect("spike-in ROC curve");
        let power: Vec<f64> = curve
            .iter()
            .map(|point| point.roc[0].true_positive_rate)
            .collect();
        assert!(
            power[0] <= 0.125,
            "pure-null spike-in row should stay near zero detection power: {:?}",
            curve
        );
        for pair in curve.windows(2) {
            let prev_power = pair[0].roc[0].true_positive_rate;
            let next_power = pair[1].roc[0].true_positive_rate;
            assert!(
                next_power + 1.0e-12 >= prev_power,
                "spike-in ROC power should be monotone: {:?}",
                curve
            );
            assert!(
                pair[1].mean_stat + 1.0e-12 >= pair[0].mean_stat,
                "spike-in mean statistic should be monotone: {:?}",
                curve
            );
        }
        assert!(
            curve
                .last()
                .map(|point| point.roc[0].true_positive_rate)
                .unwrap_or(0.0)
                >= 0.80,
            "high-SNR spike-in should be detected: {:?}",
            curve
        );
    }

    #[test]
    fn torus_spike_in_detection_reports_expected_betti_payload() {
        let noise = noise_fixture(128, 10, 4321);
        let spiked = inject_torus_spike(noise.view(), 2.0, 88).expect("torus injection should run");
        let report = default_spike_in_detection_pipeline(spiked.view(), SpikeInShape::Torus)
            .expect("torus detector should run");
        assert!(
            report.detected,
            "high-SNR torus should be detected: {report:?}"
        );
        assert_eq!(report.topology.measured_betti0, 1);
        assert_eq!(report.topology.measured_betti1, 2);
        assert_eq!(report.topology.measured_betti2, 1);
    }

    fn ordered_circle_fixture(n: usize, p: usize, noise_scale: f64) -> Array2<f64> {
        let mut data = noise_fixture(n, p, 555);
        for i in 0..n {
            let theta = 2.0 * PI * i as f64 / n as f64;
            data[[i, 0]] += theta.cos();
            data[[i, 1]] += theta.sin();
            for j in 2..p {
                data[[i, j]] *= noise_scale;
            }
        }
        data
    }

    fn noise_fixture(n: usize, p: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                data[[i, j]] = 0.2 * standard_normal(&mut rng);
            }
        }
        data
    }
}
