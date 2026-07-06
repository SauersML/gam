//! Standing null battery and spike-in calibration for topology/nerve claims.
//!
//! A Betti, nerve, or conditionality statistic is only interpretable together
//! with negative controls that preserve mundane structure while destroying the
//! claimed one. This module supplies the reusable harness: generate standing
//! nulls, run an arbitrary scalar audit on observed and null data, and attach a
//! spike-in power curve for a manufactured circle inside real residual noise.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::RngExt;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::f64::consts::PI;

/// Direction of the claim statistic under the null.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tail {
    Larger,
    Smaller,
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
    /// Use separately supplied random-weight activations from the same
    /// architecture, moment-matched to the observed matrix shape.
    ArchitectureMatchedRandomWeight,
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
                NullKind::ArchitectureMatchedRandomWeight,
            ],
            tail: Tail::Larger,
        }
    }
}

/// Null distribution summary for a single null kind.
#[derive(Clone, Debug)]
pub struct NullSummary {
    pub kind: NullKind,
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
    pub samples: Vec<f64>,
}

/// Complete null-battery result for one scalar audit statistic.
#[derive(Clone, Debug)]
pub struct NullBatteryReport {
    pub observed: f64,
    pub summaries: Vec<NullSummary>,
}

/// Spike-in detection power at one signal-to-noise ratio.
#[derive(Clone, Debug)]
pub struct SpikeInPowerPoint {
    pub snr: f64,
    pub trials: usize,
    pub power: f64,
    pub mean_stat: f64,
}

/// Claim payload expected by downstream report writers: observed statistic,
/// null summaries, and spike-in power at the claimed SNR.
#[derive(Clone, Debug)]
pub struct CalibratedClaimReport {
    pub claim: String,
    pub claimed_snr: f64,
    pub nulls: NullBatteryReport,
    pub spike_in: Vec<SpikeInPowerPoint>,
    pub claimed_snr_power: f64,
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
        summaries.push(summarize_null(kind, observed, samples, config.tail));
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
    Ok(data.dot(&q))
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

/// Resample real random-weight activations and match observed column moments.
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
    let obs_mean = column_mean(observed);
    let obs_sd = column_sd(observed, obs_mean.view());
    let rw_mean = column_mean(random_weight);
    let rw_sd = column_sd(random_weight, rw_mean.view());
    let mut out = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let src = rng.random_range(0..random_weight.nrows());
        for j in 0..p {
            let centered = random_weight[[src, j]] - rw_mean[j];
            let scaled = if rw_sd[j] > 0.0 {
                centered * (obs_sd[j] / rw_sd[j])
            } else {
                0.0
            };
            out[[i, j]] = obs_mean[j] + scaled;
        }
    }
    Ok(out)
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
    let rms = matrix_rms(noise);
    let amp = snr * rms * (p as f64).sqrt();
    let phase0 = rng.random_range(0.0..(2.0 * PI));
    let mut out = noise.to_owned();
    for i in 0..n {
        let theta = phase0 + 2.0 * PI * (i as f64) / (n as f64);
        for j in 0..p {
            out[[i, j]] += amp * (theta.cos() * basis[[j, 0]] + theta.sin() * basis[[j, 1]]);
        }
    }
    Ok(out)
}

/// Calibrate default circle-detection power with a phase-randomized threshold.
pub fn circle_spike_in_power_curve(
    noise: ArrayView2<'_, f64>,
    snrs: &[f64],
    trials: usize,
    seed: u64,
) -> Result<Vec<SpikeInPowerPoint>, String> {
    validate_matrix(noise, "spike-in noise")?;
    if trials == 0 {
        return Err("spike-in calibration requires at least one trial".to_string());
    }
    let mut null_stats = Vec::with_capacity(trials);
    for t in 0..trials {
        let surrogate =
            phase_randomized_surrogate(noise, mix_seed(seed, 0xB007_51A7, t as u64))?;
        null_stats.push(top_two_energy_fraction(surrogate.view())?);
    }
    null_stats.sort_by(|a, b| a.total_cmp(b));
    let threshold_idx = ((trials.saturating_sub(1)) * 19) / 20;
    let threshold = null_stats[threshold_idx];
    let mut curve = Vec::with_capacity(snrs.len());
    for (sidx, &snr) in snrs.iter().enumerate() {
        if !snr.is_finite() || snr < 0.0 {
            return Err(format!("snr[{sidx}] must be finite and non-negative, got {snr}"));
        }
        let mut hits = 0usize;
        let mut total = 0.0_f64;
        for t in 0..trials {
            let spiked = inject_circle_spike(noise, snr, mix_seed(seed, sidx as u64, t as u64))?;
            let stat = top_two_energy_fraction(spiked.view())?;
            total += stat;
            if stat > threshold {
                hits += 1;
            }
        }
        curve.push(SpikeInPowerPoint {
            snr,
            trials,
            power: hits as f64 / trials as f64,
            mean_stat: total / trials as f64,
        });
    }
    Ok(curve)
}

/// Build the report shape downstream claim code should carry.
pub fn calibrated_claim_report(
    claim: impl Into<String>,
    claimed_snr: f64,
    nulls: NullBatteryReport,
    spike_in: Vec<SpikeInPowerPoint>,
) -> Result<CalibratedClaimReport, String> {
    if !claimed_snr.is_finite() || claimed_snr < 0.0 {
        return Err(format!(
            "claimed_snr must be finite and non-negative, got {claimed_snr}"
        ));
    }
    let mut best_power = None;
    let mut best_distance = f64::INFINITY;
    for point in &spike_in {
        let distance = (point.snr - claimed_snr).abs();
        if distance < best_distance {
            best_distance = distance;
            best_power = Some(point.power);
        }
    }
    let Some(claimed_snr_power) = best_power else {
        return Err("calibrated claim report requires at least one spike-in point".to_string());
    };
    Ok(CalibratedClaimReport {
        claim: claim.into(),
        claimed_snr,
        nulls,
        spike_in,
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
    let x = centered_column(data.column(0));
    let y = centered_column(data.column(1));
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
    if energy <= 0.0 {
        return Ok(0.0);
    }
    let pos = pos_re * pos_re + pos_im * pos_im;
    let neg = neg_re * neg_re + neg_im * neg_im;
    let winding_balance = (pos - neg).max(0.0) / (pos + neg).max(f64::MIN_POSITIVE);
    let area_scale = signed_area.abs() / energy.max(f64::MIN_POSITIVE);
    Ok(winding_balance * area_scale)
}

/// Basis-dependent energy in the first two coordinates.
pub fn first_two_energy_fraction(data: ArrayView2<'_, f64>) -> Result<f64, String> {
    validate_matrix(data, "first-two energy input")?;
    if data.ncols() < 2 {
        return Err("first-two energy requires at least two columns".to_string());
    }
    let centered = centered_matrix(data);
    let total = centered.iter().map(|v| v * v).sum::<f64>();
    if total <= 0.0 {
        return Ok(0.0);
    }
    let mut first_two = 0.0_f64;
    for i in 0..centered.nrows() {
        first_two += centered[[i, 0]] * centered[[i, 0]] + centered[[i, 1]] * centered[[i, 1]];
    }
    Ok(first_two / total)
}

/// Basis-invariant rank-two covariance energy fraction.
pub fn top_two_energy_fraction(data: ArrayView2<'_, f64>) -> Result<f64, String> {
    validate_matrix(data, "top-two energy input")?;
    let centered = centered_matrix(data);
    let total = centered.iter().map(|v| v * v).sum::<f64>();
    if total <= 0.0 {
        return Ok(0.0);
    }
    let cov = centered.t().dot(&centered);
    let lambda1 = dominant_eigenvalue(cov.view(), 0x5151_0001)?;
    let mut deflated = cov.to_owned();
    let v1 = dominant_eigenvector(cov.view(), 0x5151_0001)?;
    for i in 0..deflated.nrows() {
        for j in 0..deflated.ncols() {
            deflated[[i, j]] -= lambda1 * v1[i] * v1[j];
        }
    }
    let lambda2 = dominant_eigenvalue(deflated.view(), 0x5151_0002)?.max(0.0);
    Ok((lambda1.max(0.0) + lambda2) / total)
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

fn summarize_null(kind: NullKind, observed: f64, mut samples: Vec<f64>, tail: Tail) -> NullSummary {
    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        samples
            .iter()
            .map(|x| {
                let d = *x - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1) as f64
    } else {
        0.0
    };
    let sd = var.sqrt();
    let z = if sd > 0.0 {
        (observed - mean) / sd
    } else {
        0.0
    };
    let extreme = samples
        .iter()
        .filter(|&&x| match tail {
            Tail::Larger => x >= observed,
            Tail::Smaller => x <= observed,
        })
        .count();
    let p_value = (extreme + 1) as f64 / (n + 1) as f64;
    NullSummary {
        kind,
        observed,
        n,
        mean,
        sd,
        min: samples[0],
        q25: samples[n / 4],
        median: samples[n / 2],
        q75: samples[(3 * n) / 4],
        max: samples[n - 1],
        z,
        p_value,
        samples,
    }
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
        for prev in 0..col {
            let mut dot = 0.0_f64;
            for i in 0..p {
                dot += v[i] * q[[i, prev]];
            }
            for i in 0..p {
                v[i] -= dot * q[[i, prev]];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= f64::MIN_POSITIVE {
            return random_orthogonal(p, seed ^ ((col as u64 + 1) * 0xD1B5_4A32_D192_ED03));
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

fn column_mean(data: ArrayView2<'_, f64>) -> Array1<f64> {
    let mut mean = Array1::<f64>::zeros(data.ncols());
    for row in data.rows() {
        mean += &row;
    }
    mean / data.nrows() as f64
}

fn column_sd(data: ArrayView2<'_, f64>, mean: ArrayView1<'_, f64>) -> Array1<f64> {
    let mut var = Array1::<f64>::zeros(data.ncols());
    for row in data.rows() {
        for j in 0..data.ncols() {
            let d = row[j] - mean[j];
            var[j] += d * d;
        }
    }
    var.mapv(|v| (v / data.nrows() as f64).sqrt())
}

fn centered_column(col: ArrayView1<'_, f64>) -> Array1<f64> {
    let mean = col.iter().sum::<f64>() / col.len() as f64;
    col.mapv(|v| v - mean)
}

fn centered_matrix(data: ArrayView2<'_, f64>) -> Array2<f64> {
    let mean = column_mean(data);
    let mut out = data.to_owned();
    for mut row in out.rows_mut() {
        row -= &mean;
    }
    out
}

fn matrix_rms(data: ArrayView2<'_, f64>) -> f64 {
    let ss = data.iter().map(|v| v * v).sum::<f64>();
    (ss / (data.nrows() * data.ncols()) as f64).sqrt()
}

fn dominant_eigenvalue(matrix: ArrayView2<'_, f64>, seed: u64) -> Result<f64, String> {
    let v = dominant_eigenvector(matrix, seed)?;
    let mv = matrix.dot(&v);
    Ok(v.dot(&mv))
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

    #[test]
    fn structured_signal_beats_phase_and_rotation_nulls_but_noise_does_not() {
        let signal = ordered_circle_fixture(96, 6, 0.04);
        let config = NullBatteryConfig {
            replicates: 16,
            seed: 17,
            kinds: vec![NullKind::PhaseRandomized, NullKind::RandomRotation],
            tail: Tail::Larger,
        };
        let report = run_null_battery(signal.view(), None, &config, first_two_ordered_circle_stat)
            .expect("structured null battery should run");
        assert!(
            report.summaries.iter().all(|s| s.z > 2.0 && s.p_value <= 0.12),
            "structured ordered circle should separate from nulls: {:?}",
            report.summaries
        );

        let noise = noise_fixture(96, 6, 99);
        let noise_report =
            run_null_battery(noise.view(), None, &config, first_two_ordered_circle_stat)
                .expect("noise null battery should run");
        assert!(
            noise_report
                .summaries
                .iter()
                .all(|s| s.z.abs() < 2.0 || s.p_value > 0.12),
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
    fn spike_in_power_rises_monotonically_with_snr() {
        let noise = noise_fixture(96, 8, 1234);
        let snrs = [0.0, 0.4, 0.8, 1.2];
        let curve =
            circle_spike_in_power_curve(noise.view(), &snrs, 12, 91).expect("spike-in curve");
        for pair in curve.windows(2) {
            assert!(
                pair[1].power + 1.0e-12 >= pair[0].power,
                "spike-in power should be monotone: {:?}",
                curve
            );
            assert!(
                pair[1].mean_stat + 1.0e-12 >= pair[0].mean_stat,
                "spike-in mean statistic should be monotone: {:?}",
                curve
            );
        }
        assert!(
            curve.last().map(|p| p.power).unwrap_or(0.0) >= 0.75,
            "high-SNR spike-in should be detected: {:?}",
            curve
        );
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
