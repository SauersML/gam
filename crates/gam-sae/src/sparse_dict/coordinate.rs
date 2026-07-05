//! Per-firing coordinate readout with closed-form standard errors, off the
//! block-sparse lane's already-computed within-block codes ([`super::block`]).
//!
//! # What this reads
//!
//! A [`BlockSparseFit`] stores, per firing, the signed within-block code
//! `z_g ∈ ℝᵇ` (the `codes[N,k,b]` field) alongside the block it fired on
//! (`blocks[N,k]`) and its group ℓ₂ gate `‖z_g‖₂` (`gates[N,k]`). When a block's
//! `b`-dimensional subspace hosts a *circle* feature, that code is the read-out
//! we turn into an interpretable **phase** `t ∈ [0,1)` (mod 1) and **amplitude**
//! `a`, each with a standard error computed in closed form (no finite
//! differences, per `SPEC.md`).
//!
//! # b = 2: the circle model and the delta-method SE
//!
//! For a block of size `b = 2` whose subspace hosts a circle, the within-block
//! code decomposes as
//!
//! ```text
//!   z = a · u(t₀) + ε,   u(t) = (cos 2πt, sin 2πt),   ε ~ N(0, σ² I₂).
//! ```
//!
//! **Phase.** The estimator is `t̂ = atan2(z₂, z₁) / (2π)  (mod 1)`. Write
//! `θ = atan2(z₂, z₁)` (so `t = θ/2π`). The angular gradient is
//!
//! ```text
//!   ∂θ/∂z₁ = −z₂/‖z‖²,   ∂θ/∂z₂ =  z₁/‖z‖²,
//!   ‖∇θ‖²  = (z₁² + z₂²)/‖z‖⁴ = 1/‖z‖².
//! ```
//!
//! With isotropic noise `cov(z) = σ² I₂`, the delta method gives
//! `Var(θ̂) = σ² ‖∇θ‖² = σ²/‖z‖²`, hence
//!
//! ```text
//!   SD(t̂) = SD(θ̂)/(2π) = σ / (2π ‖z‖).
//! ```
//!
//! **Amplitude.** The estimator is `â = ‖z‖`. Its gradient is the unit radial
//! `∂‖z‖/∂z = z/‖z‖` (norm 1), so `Var(â) = σ²·1` and `SD(â) ≈ σ`.
//!
//! # Estimating σ (it is not a knob)
//!
//! `σ` is estimated per block from the firings' codes, never supplied. For an
//! isotropic-noise circle of unknown per-firing amplitude, the phase is a pure
//! rotation and the radius carries `signal + radial-noise`: with
//! `z = r·e(θ) + ε`, the observed radius is `‖z‖ ≈ r + (ε·e)` where the radial
//! projection `ε·e ~ N(0, σ²)`. The radial component of an isotropic Gaussian is
//! one-dimensional with variance `σ²` **regardless of the ambient block size**,
//! so the scatter of `‖z‖` about the block's mean radius estimates the
//! per-component `σ²` directly:
//!
//! ```text
//!   r̄  = mean_i ‖z_i‖,
//!   σ̂² = (1/(n−1)) Σ_i (‖z_i‖ − r̄)²      (one dof spent on r̄).
//! ```
//!
//! **Assumption + one-sided guarantee.** This assumes isotropic within-block
//! noise, so radial scatter estimates the per-component `σ`. If the amplitude
//! genuinely varies across firings (the subspace is a *cone*/*cylinder* rather
//! than a fixed-radius circle), the radial scatter absorbs that amplitude
//! spread and **over**estimates `σ`; the reported SE is then conservative. This
//! is a one-sided guarantee: the SE never silently understates phase
//! uncertainty from amplitude heterogeneity.
//!
//! # Weak-signal regime (never NaN)
//!
//! The linearised phase SE `σ̂/(2π‖z‖)` diverges as `‖z‖ → 0`, but phase
//! uncertainty is bounded: the least informative posterior is the uniform
//! distribution on the unit-circumference parameter `t ∈ [0,1)`, whose standard
//! deviation is `√(Var U(0,1)) = √(1/12)`. We therefore clamp
//! `t_se = min(σ̂/(2π‖z‖), √(1/12))` and set [`FiringCoordinate::t_se_clamped`]
//! whenever the raw SE reaches that uniform ceiling — i.e. once
//! `‖z‖ ≤ σ̂·√12/(2π)`, the *derived* radius at which the linearised SE meets
//! the uniform SD (no separate magic threshold). A zero-norm firing maps to the
//! uniform SD, never a NaN.
//!
//! # b = 2H: harmonic charts
//!
//! When a block carries `H` harmonics (`b = 2H`), the code splits into per-
//! harmonic 2-vectors `ρ_h = (z_{2h}, z_{2h+1})` and the phase maximises the
//! trigonometric matched filter
//!
//! ```text
//!   f(t) = Σ_h ρ_h · u_h(t),   u_h(t) = (cos ω_h t, sin ω_h t),   ω_h = 2π h.
//! ```
//!
//! `f` is a real trigonometric polynomial of degree `H`; `f'` is likewise degree
//! `H` and has at most `2H` zeros per period, so `f` has at most `2H` critical
//! points and at most `H` local maxima. Evaluating `f` on the `4H`-point
//! equispaced grid (grid spacing `1/(4H)`, four times the degree) places a grid
//! node inside the global maximum's basin — a **degree-determined** localisation,
//! not a tuned grid — and a Newton polish on `f'(t) = 0` from the best node
//! converges to the maximiser. The polish reverts to the grid node if it fails to
//! increase `f`, so the reported phase is never worse than the grid argmax.
//!
//! The SE is the M-estimator (sandwich) delta method for the root `f'(t̂) = 0`.
//! `f'(t) = Σ_h ω_h(−ρ_{h,1} sin ω_h t + ρ_{h,2} cos ω_h t)` is linear in `ρ`
//! with `Var(f'(t)) = σ² Σ_h ω_h²` under `cov(ρ) = σ² I`, and the local slope is
//! `f''(t̂)`, giving
//!
//! ```text
//!   Var(t̂) ≈ Var(f'(t̂)) / [f''(t̂)]² = σ² (Σ_h ω_h²) / [f''(t̂)]².
//! ```
//!
//! For `H = 1` this collapses to the `b = 2` formula: at the peak
//! `f''(t̂) = −(2π)² ‖z‖`, so `Var(t̂) = σ²(2π)² / (2π)⁴‖z‖² = σ²/(2π‖z‖)²`.

use super::block::BlockSparseFit;
use crate::super_resolution::{recover_spikes, separation_limit};
use ndarray::{Array2, ArrayView2};
use std::f64::consts::TAU;

/// The phase and amplitude of one firing on a coordinate (circle/harmonic)
/// block, each with a closed-form standard error.
#[derive(Clone, Copy, Debug)]
pub struct FiringCoordinate {
    /// Block this firing fired on.
    pub block: usize,
    /// Row (token) index in the fit's routing.
    pub row: usize,
    /// Phase estimate `t̂ ∈ [0,1)` (mod 1).
    pub t: f64,
    /// Amplitude estimate `â = ‖z‖`.
    pub amplitude: f64,
    /// Standard error of `t̂` (clamped to the uniform-phase SD `√(1/12)` in the
    /// weak-signal regime — see [`Self::t_se_clamped`]).
    pub t_se: f64,
    /// Standard error of `â` (`≈ σ̂`).
    pub amplitude_se: f64,
    /// Set when `t_se` was clamped to the uniform-phase ceiling `√(1/12)`
    /// because the raw linearised SE met or exceeded it (weak signal, phase
    /// effectively unidentified). When set, `t` is reported but uninformative.
    pub t_se_clamped: bool,
}

/// Per-block coordinate report: the shared noise scale, the mean radius, and the
/// per-firing coordinates.
#[derive(Clone, Debug)]
pub struct BlockCoordinateReport {
    /// Estimated isotropic per-component noise `σ̂` (radial scatter of `‖z‖`).
    pub sigma_hat: f64,
    /// Mean firing radius `r̄ = mean ‖z‖`.
    pub mean_radius: f64,
    /// Number of firings on this block.
    pub n_firings: usize,
    /// One [`FiringCoordinate`] per firing, in ascending row order.
    pub firings: Vec<FiringCoordinate>,
}

/// One point mass in a measure-valued harmonic firing.
#[derive(Clone, Copy, Debug)]
pub struct MeasureSpikeCoordinate {
    /// Physical spike amplitude `a` in
    /// `z_h = Σ_j a_j (cos 2πh t_j, sin 2πh t_j)`.
    pub amplitude: f64,
    /// Circle coordinate `t ∈ [0, 1)`.
    pub coordinate: f64,
    /// Delta-method standard error of [`Self::coordinate`].
    pub coordinate_se: f64,
}

/// Variable-length code for one fired harmonic block in one row.
#[derive(Clone, Debug)]
pub struct MeasureValuedCode {
    /// Block this measure lives on.
    pub block: usize,
    /// Row (token) index.
    pub row: usize,
    /// Point masses on the block's circle. A live firing always has at least one.
    pub spikes: Vec<MeasureSpikeCoordinate>,
    /// Dual-polynomial birth ratio for the single-spike residual. Values above
    /// one are the threshold-free BLASSO multiplicity trigger.
    pub dual_eta: f64,
    /// Whether matrix-pencil super-resolution supplied the returned support.
    pub used_super_resolution: bool,
}

/// Measure-valued readout for a block: one variable-length code per firing.
#[derive(Clone, Debug)]
pub struct BlockMeasureCoordinateReport {
    /// Estimated isotropic per-component coefficient noise.
    pub sigma_hat: f64,
    /// Mean firing radius in the stored block-code coordinates.
    pub mean_radius: f64,
    /// Number of firings on this block.
    pub n_firings: usize,
    /// One measure-valued code per firing, in ascending row order.
    pub firings: Vec<MeasureValuedCode>,
}

/// SD of the uniform distribution on the unit-circumference phase `t ∈ [0,1)`:
/// `√(Var U(0,1)) = √(1/12)`. The maximum-entropy fallback SE when the phase is
/// unidentified.
fn uniform_phase_sd() -> f64 {
    (1.0f64 / 12.0).sqrt()
}

/// Collect one block's firings from a fit as `(row, z)` with `z` the signed
/// within-block code lifted to f64. A genuine firing is one whose stored gate
/// `‖z_g‖₂` is non-zero (padded routing slots carry a zero gate).
fn collect_firings(fit: &BlockSparseFit, block: usize, b: usize) -> Vec<(usize, Vec<f64>)> {
    let n = fit.blocks.nrows();
    let k = fit.blocks.ncols();
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..k {
            if fit.blocks[[i, j]] as usize != block {
                continue;
            }
            if fit.gates[[i, j]] == 0.0 {
                continue; // padded slot
            }
            let mut z = Vec::with_capacity(b);
            for r in 0..b {
                z.push(fit.codes[[i, j, r]] as f64);
            }
            out.push((i, z));
            break; // a block fires at most once per row
        }
    }
    out
}

/// Mean radius `r̄` and unbiased radial-scatter noise `σ̂` from the firing codes.
fn radius_and_sigma(firings: &[(usize, Vec<f64>)]) -> (f64, f64) {
    let n = firings.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut norms = Vec::with_capacity(n);
    let mut sum = 0.0f64;
    for (_, z) in firings {
        let nrm = z.iter().map(|v| v * v).sum::<f64>().sqrt();
        sum += nrm;
        norms.push(nrm);
    }
    let mean = sum / n as f64;
    if n < 2 {
        return (mean, 0.0);
    }
    let ss: f64 = norms.iter().map(|&r| (r - mean) * (r - mean)).sum();
    let sigma = (ss / (n - 1) as f64).sqrt();
    (mean, sigma)
}

/// Assemble the phase SE for a `b = 2` firing, clamping to the uniform ceiling in
/// the weak-signal regime. `raw_se = σ̂/(2π‖z‖)`; a zero-norm firing yields the
/// uniform SD (never NaN).
fn phase_se_b2(sigma: f64, norm: f64) -> (f64, bool) {
    let ceiling = uniform_phase_sd();
    let raw = if norm > 0.0 {
        sigma / (TAU * norm)
    } else {
        f64::INFINITY
    };
    if raw >= ceiling {
        (ceiling, true)
    } else {
        (raw, false)
    }
}

fn coeffs_from_code(z: &[f64]) -> Vec<(f64, f64)> {
    z.chunks_exact(2).map(|pair| (pair[0], pair[1])).collect()
}

fn code_from_spikes(spikes: &[MeasureSpikeCoordinate], h_count: usize) -> Vec<f64> {
    let mut z = vec![0.0; 2 * h_count];
    for spike in spikes {
        for h in 1..=h_count {
            let phase = TAU * h as f64 * spike.coordinate;
            let (s, c) = phase.sin_cos();
            z[2 * (h - 1)] += spike.amplitude * c;
            z[2 * (h - 1) + 1] += spike.amplitude * s;
        }
    }
    z
}

fn single_harmonic_spike(z: &[f64], sigma: f64) -> (MeasureSpikeCoordinate, Vec<f64>, f64) {
    let h_count = z.len() / 2;
    let (coordinate, _curvature) = harmonic_argmax(z);
    let matched = harmonic_f(z, coordinate);
    let amplitude = (matched / h_count.max(1) as f64).max(0.0);
    let spike = MeasureSpikeCoordinate {
        amplitude,
        coordinate,
        coordinate_se: spike_coordinate_se(sigma, amplitude, h_count),
    };
    let fitted = code_from_spikes(&[spike], h_count);
    let residual: Vec<f64> = z
        .iter()
        .zip(fitted.iter())
        .map(|(&observed, &pred)| observed - pred)
        .collect();
    let residual_norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
    (spike, residual, residual_norm)
}

fn spike_coordinate_se(sigma: f64, amplitude: f64, h_count: usize) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }
    let ceiling = uniform_phase_sd();
    let h_sq_sum = (1..=h_count).map(|h| (h * h) as f64).sum::<f64>();
    let slope = amplitude * TAU * h_sq_sum.sqrt();
    if slope <= 0.0 {
        ceiling
    } else {
        (sigma / slope).min(ceiling)
    }
}

fn local_dual_eta(residual: &[f64], active_mass: f64) -> f64 {
    let lambda = active_mass.max(f64::MIN_POSITIVE);
    let off_support_gate = harmonic_f(residual, harmonic_argmax(residual).0).max(0.0);
    off_support_gate / lambda
}

fn circle_dist(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(1.0 - d)
}

fn separated_from_all(t: f64, accepted: &[MeasureSpikeCoordinate], min_sep: f64) -> bool {
    accepted
        .iter()
        .all(|spike| circle_dist(t, spike.coordinate) + f64::EPSILON >= min_sep)
}

fn count_separated_positive_modes(z: &[f64], min_sep: f64) -> usize {
    let h_count = z.len() / 2;
    let grid = 4 * h_count.max(1);
    let mut candidates = Vec::new();
    for idx in 0..grid {
        let prev = (idx + grid - 1) % grid;
        let next = (idx + 1) % grid;
        let t = idx as f64 / grid as f64;
        let val = harmonic_f(z, t);
        if val <= 0.0 {
            continue;
        }
        if val >= harmonic_f(z, prev as f64 / grid as f64)
            && val >= harmonic_f(z, next as f64 / grid as f64)
        {
            candidates.push((t, val));
        }
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut accepted: Vec<MeasureSpikeCoordinate> = Vec::new();
    for (t, val) in candidates {
        if separated_from_all(t, &accepted, min_sep) {
            accepted.push(MeasureSpikeCoordinate {
                amplitude: val,
                coordinate: t,
                coordinate_se: 0.0,
            });
        }
    }
    accepted.len()
}

fn maybe_super_resolve(z: &[f64], sigma: f64) -> (Vec<MeasureSpikeCoordinate>, f64, bool) {
    let h_count = z.len() / 2;
    let (single, single_residual, single_residual_norm) = single_harmonic_spike(z, sigma);
    if h_count < 2 {
        return (vec![single], 0.0, false);
    }

    let min_sep = separation_limit(h_count);
    let eta = local_dual_eta(&single_residual, single.amplitude);
    let residual_is_multimodal = count_separated_positive_modes(&single_residual, min_sep) > 1;
    let code_is_multimodal = count_separated_positive_modes(z, min_sep) > 1;
    if eta <= 1.0 && !residual_is_multimodal && !code_is_multimodal {
        return (vec![single], eta, false);
    }

    let coeffs = coeffs_from_code(z);
    let recovery = match recover_spikes(&coeffs, sigma) {
        Ok(recovery) => recovery,
        Err(_err) => return (vec![single], eta, false),
    };
    if recovery.spikes.len() <= 1 || recovery.residual >= single_residual_norm {
        return (vec![single], eta, false);
    }

    let max_by_separation = if min_sep.is_finite() && min_sep > 0.0 {
        (1.0 / min_sep).floor().max(1.0) as usize
    } else {
        recovery.spikes.len()
    };
    let mut sorted = recovery.spikes;
    sorted.sort_by(|a, b| b.amplitude.total_cmp(&a.amplitude));
    let mut accepted = Vec::new();
    for spike in sorted {
        if spike.amplitude <= 0.0 {
            continue;
        }
        if accepted.len() >= max_by_separation {
            break;
        }
        if !separated_from_all(spike.t, &accepted, min_sep) {
            continue;
        }
        accepted.push(MeasureSpikeCoordinate {
            amplitude: spike.amplitude,
            coordinate: spike.t,
            coordinate_se: spike_coordinate_se(sigma, spike.amplitude, h_count),
        });
    }
    accepted.sort_by(|a, b| a.coordinate.total_cmp(&b.coordinate));
    if accepted.len() <= 1 {
        (vec![single], eta, false)
    } else {
        (accepted, eta, true)
    }
}

/// Per-firing coordinate readout for a `b = 2` circle block: phase `t̂`,
/// amplitude `‖z‖`, and their closed-form SEs, with `σ̂` estimated from the
/// block's radial scatter. See the module doc for the derivation.
pub fn block_firing_coordinates(
    fit: &BlockSparseFit,
    block: usize,
) -> Result<BlockCoordinateReport, String> {
    let b = fit.block_size;
    if b != 2 {
        return Err(format!(
            "block_firing_coordinates: circle readout requires block_size b = 2, got b = {b}; \
             use harmonic_firing_coordinates for b = 2H"
        ));
    }
    let g_total = fit.decoder.nrows() / b;
    if block >= g_total {
        return Err(format!(
            "block_firing_coordinates: block {block} out of range 0..{g_total}"
        ));
    }
    let firings = collect_firings(fit, block, b);
    let (mean_radius, sigma_hat) = radius_and_sigma(&firings);

    let mut coords = Vec::with_capacity(firings.len());
    for (row, z) in &firings {
        let norm = (z[0] * z[0] + z[1] * z[1]).sqrt();
        let t = (z[1].atan2(z[0]) / TAU).rem_euclid(1.0);
        let (t_se, t_se_clamped) = phase_se_b2(sigma_hat, norm);
        coords.push(FiringCoordinate {
            block,
            row: *row,
            t,
            amplitude: norm,
            t_se,
            amplitude_se: sigma_hat,
            t_se_clamped,
        });
    }

    Ok(BlockCoordinateReport {
        sigma_hat,
        mean_radius,
        n_firings: firings.len(),
        firings: coords,
    })
}

// ---------------------------------------------------------------------------
// Harmonic (b = 2H) matched-filter phase.
// ---------------------------------------------------------------------------

/// Matched-filter value `f(t) = Σ_h ρ_h·u_h(t)` for `H = rho.len()/2` harmonics.
fn harmonic_f(rho: &[f64], t: f64) -> f64 {
    let h_count = rho.len() / 2;
    let mut acc = 0.0;
    for h in 0..h_count {
        let w = TAU * (h + 1) as f64;
        let (s, c) = (w * t).sin_cos();
        acc += rho[2 * h] * c + rho[2 * h + 1] * s;
    }
    acc
}

/// First derivative `f'(t) = Σ_h ω_h(−ρ_{h,1} sin ω_h t + ρ_{h,2} cos ω_h t)`.
fn harmonic_fp(rho: &[f64], t: f64) -> f64 {
    let h_count = rho.len() / 2;
    let mut acc = 0.0;
    for h in 0..h_count {
        let w = TAU * (h + 1) as f64;
        let (s, c) = (w * t).sin_cos();
        acc += w * (-rho[2 * h] * s + rho[2 * h + 1] * c);
    }
    acc
}

/// Second derivative `f''(t) = Σ_h ω_h²(−ρ_{h,1} cos ω_h t − ρ_{h,2} sin ω_h t)`.
fn harmonic_fpp(rho: &[f64], t: f64) -> f64 {
    let h_count = rho.len() / 2;
    let mut acc = 0.0;
    for h in 0..h_count {
        let w = TAU * (h + 1) as f64;
        let (s, c) = (w * t).sin_cos();
        acc += w * w * (-rho[2 * h] * c - rho[2 * h + 1] * s);
    }
    acc
}

/// Locate the global maximiser of `f` on `[0,1)`: `4H`-point equispaced grid
/// scan (degree-determined localisation), then a Newton polish on `f'(t) = 0`
/// from the best node. Returns `(t̂, f''(t̂))`. The polish reverts to the grid
/// node if it fails to increase `f`, so `t̂` is never worse than the grid argmax.
fn harmonic_argmax(rho: &[f64]) -> (f64, f64) {
    let h_count = rho.len() / 2;
    let grid = 4 * h_count.max(1);
    let mut best_t = 0.0;
    let mut best_f = f64::NEG_INFINITY;
    for m in 0..grid {
        let t = m as f64 / grid as f64;
        let val = harmonic_f(rho, t);
        if val > best_f {
            best_f = val;
            best_t = t;
        }
    }

    // Newton polish on f'(t) = 0 from the best grid node. The step tolerance is
    // derived from f64 machine epsilon (Newton doubles correct digits each step,
    // so √eps reaches full precision); the iteration cap is a non-convergence
    // safety bound (a genuine peak converges in a handful of steps), not a tuned
    // budget — a flat/failed polish falls back to the grid node below.
    let tol = f64::EPSILON.sqrt();
    let iter_cap = 64;
    let mut t = best_t;
    let mut converged = false;
    for _ in 0..iter_cap {
        let fpp = harmonic_fpp(rho, t);
        if fpp.abs() <= f64::MIN_POSITIVE {
            break;
        }
        let step = harmonic_fp(rho, t) / fpp;
        t -= step;
        if step.abs() <= tol * (1.0 + t.abs()) {
            converged = true;
            break;
        }
    }
    let t_polished = t.rem_euclid(1.0);
    let (t_hat, fpp) = if converged && harmonic_f(rho, t_polished) >= best_f {
        (t_polished, harmonic_fpp(rho, t_polished))
    } else {
        (best_t, harmonic_fpp(rho, best_t))
    };
    (t_hat, fpp)
}

/// Per-firing coordinate readout for a harmonic block (`b = 2H`, `H ≥ 1`): the
/// phase `t̂` maximising the trig matched filter `Σ_h ρ_h·u_h(t)` via a
/// `4H`-grid scan plus Newton polish, with the delta-method SE
/// `√(σ̂² Σ_h ω_h²)/|f''(t̂)|`. Amplitude is `‖z‖` with SE `≈ σ̂`. See the module
/// doc for the derivation; `H = 1` reproduces [`block_firing_coordinates`].
pub fn harmonic_firing_coordinates(
    fit: &BlockSparseFit,
    block: usize,
) -> Result<BlockCoordinateReport, String> {
    let b = fit.block_size;
    if b < 2 || b % 2 != 0 {
        return Err(format!(
            "harmonic_firing_coordinates: harmonic readout requires block_size b = 2H (even, \
             ≥ 2), got b = {b}"
        ));
    }
    let g_total = fit.decoder.nrows() / b;
    if block >= g_total {
        return Err(format!(
            "harmonic_firing_coordinates: block {block} out of range 0..{g_total}"
        ));
    }
    let h_count = b / 2;
    // Σ_h ω_h² with ω_h = 2π h.
    let omega_sq_sum: f64 = (1..=h_count).map(|h| TAU * h as f64).map(|w| w * w).sum();

    let firings = collect_firings(fit, block, b);
    let (mean_radius, sigma_hat) = radius_and_sigma(&firings);
    let ceiling = uniform_phase_sd();

    let mut coords = Vec::with_capacity(firings.len());
    for (row, z) in &firings {
        let norm = z.iter().map(|v| v * v).sum::<f64>().sqrt();
        let (t_hat, fpp) = harmonic_argmax(z);
        // Delta-method phase SE. A non-negative curvature (no genuine peak, e.g.
        // a zero code) leaves the phase unidentified → uniform ceiling.
        let raw = if fpp < 0.0 {
            (sigma_hat * sigma_hat * omega_sq_sum).sqrt() / (-fpp)
        } else {
            f64::INFINITY
        };
        let (t_se, t_se_clamped) = if raw >= ceiling {
            (ceiling, true)
        } else {
            (raw, false)
        };
        coords.push(FiringCoordinate {
            block,
            row: *row,
            t: t_hat,
            amplitude: norm,
            t_se,
            amplitude_se: sigma_hat,
            t_se_clamped,
        });
    }

    Ok(BlockCoordinateReport {
        sigma_hat,
        mean_radius,
        n_firings: firings.len(),
        firings: coords,
    })
}

/// Measure-valued readout for a harmonic block (`b = 2H`). Each live
/// `(row, block)` firing returns one or more point masses
/// `(amplitude, coordinate, coordinate_se)`. The single-spike path is retained
/// unless the single-spike residual has a BLASSO dual birth ratio `η > 1` or the
/// harmonic profile/residual has multiple separated modes; a matrix-pencil
/// recovery is accepted only when it reduces the harmonic coefficient residual.
pub fn harmonic_measure_coordinates(
    fit: &BlockSparseFit,
    block: usize,
) -> Result<BlockMeasureCoordinateReport, String> {
    let b = fit.block_size;
    if b < 2 || b % 2 != 0 {
        return Err(format!(
            "harmonic_measure_coordinates: harmonic readout requires block_size b = 2H (even, \
             >= 2), got b = {b}"
        ));
    }
    let g_total = fit.decoder.nrows() / b;
    if block >= g_total {
        return Err(format!(
            "harmonic_measure_coordinates: block {block} out of range 0..{g_total}"
        ));
    }

    let firings = collect_firings(fit, block, b);
    let (mean_radius, sigma_hat) = radius_and_sigma(&firings);
    let mut measures = Vec::with_capacity(firings.len());
    for (row, z) in &firings {
        let (spikes, dual_eta, used_super_resolution) = maybe_super_resolve(z, sigma_hat);
        measures.push(MeasureValuedCode {
            block,
            row: *row,
            spikes,
            dual_eta,
            used_super_resolution,
        });
    }

    Ok(BlockMeasureCoordinateReport {
        sigma_hat,
        mean_radius,
        n_firings: firings.len(),
        firings: measures,
    })
}

/// Measure-valued readout for every fired harmonic block in a fit.
pub fn block_measure_valued_codes(fit: &BlockSparseFit) -> Result<Vec<MeasureValuedCode>, String> {
    let b = fit.block_size;
    if b < 2 || b % 2 != 0 {
        return Err(format!(
            "block_measure_valued_codes: harmonic readout requires block_size b = 2H (even, \
             >= 2), got b = {b}"
        ));
    }
    let g_total = fit.decoder.nrows() / b;
    let mut all = Vec::new();
    for block in 0..g_total {
        let mut report = harmonic_measure_coordinates(fit, block)?;
        all.append(&mut report.firings);
    }
    all.sort_by(|a, b| a.row.cmp(&b.row).then(a.block.cmp(&b.block)));
    Ok(all)
}

/// Reconstruct dense rows by integrating the decoder against variable-length
/// harmonic measures. This is the measure-valued analogue of
/// [`crate::sparse_dict::reconstruct_block_sparse_rows`].
pub fn reconstruct_measure_valued_rows(
    decoder: ArrayView2<'_, f32>,
    measures: &[MeasureValuedCode],
    n_rows: usize,
    block_size: usize,
) -> Result<Array2<f32>, String> {
    let b = block_size;
    if b < 2 || b % 2 != 0 {
        return Err(format!(
            "reconstruct_measure_valued_rows: block_size must be even and >= 2, got {b}"
        ));
    }
    if decoder.nrows() % b != 0 {
        return Err(format!(
            "reconstruct_measure_valued_rows: decoder rows {} not divisible by block_size {b}",
            decoder.nrows()
        ));
    }
    let g_total = decoder.nrows() / b;
    let h_count = b / 2;
    let p = decoder.ncols();
    let mut out = Array2::<f32>::zeros((n_rows, p));
    for measure in measures {
        if measure.row >= n_rows {
            return Err(format!(
                "reconstruct_measure_valued_rows: row {} out of range 0..{n_rows}",
                measure.row
            ));
        }
        if measure.block >= g_total {
            return Err(format!(
                "reconstruct_measure_valued_rows: block {} out of range 0..{g_total}",
                measure.block
            ));
        }
        let code = code_from_spikes(&measure.spikes, h_count);
        for r in 0..b {
            let coeff = code[r] as f32;
            if coeff == 0.0 {
                continue;
            }
            let atom = decoder.row(measure.block * b + r);
            for c in 0..p {
                out[[measure.row, c]] += coeff * atom[c];
            }
        }
    }
    Ok(out)
}

/// Reconstruct dense rows from the single-coordinate harmonic readout, retaining
/// exactly one spike per fired block. This is useful as the explicit baseline
/// the measure-valued readout must dominate.
pub fn reconstruct_single_coordinate_rows(fit: &BlockSparseFit) -> Result<Array2<f32>, String> {
    let b = fit.block_size;
    if b < 2 || b % 2 != 0 {
        return Err(format!(
            "reconstruct_single_coordinate_rows: harmonic readout requires block_size b = 2H \
             (even, >= 2), got b = {b}"
        ));
    }
    let mut measures = Vec::new();
    for block in 0..fit.decoder.nrows() / b {
        for (row, z) in collect_firings(fit, block, b) {
            let (spike, residual, residual_norm) = single_harmonic_spike(&z, 0.0);
            assert_eq!(residual.len(), b);
            assert!(residual_norm.is_finite());
            measures.push(MeasureValuedCode {
                block,
                row,
                spikes: vec![spike],
                dual_eta: local_dual_eta(&residual, spike.amplitude),
                used_super_resolution: false,
            });
        }
    }
    reconstruct_measure_valued_rows(fit.decoder.view(), &measures, fit.blocks.nrows(), b)
}

/// Held-in explained variance of a dense reconstruction against `x`.
pub fn explained_variance_from_reconstruction(
    x: ArrayView2<'_, f32>,
    reconstruction: ArrayView2<'_, f32>,
) -> Result<f64, String> {
    if x.dim() != reconstruction.dim() {
        return Err(format!(
            "explained_variance_from_reconstruction: X shape {:?} != reconstruction shape {:?}",
            x.dim(),
            reconstruction.dim()
        ));
    }
    let (n, p) = x.dim();
    let mut means = vec![0.0; p];
    for i in 0..n {
        for c in 0..p {
            means[c] += x[[i, c]] as f64;
        }
    }
    for mean in &mut means {
        *mean /= n.max(1) as f64;
    }
    let mut rss = 0.0;
    let mut tss = 0.0;
    for i in 0..n {
        for c in 0..p {
            let r = x[[i, c]] as f64 - reconstruction[[i, c]] as f64;
            rss += r * r;
            let centered = x[[i, c]] as f64 - means[c];
            tss += centered * centered;
        }
    }
    if tss <= f64::MIN_POSITIVE {
        Ok(if rss <= f64::MIN_POSITIVE { 1.0 } else { 0.0 })
    } else {
        Ok(1.0 - rss / tss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    /// Deterministic LCG uniform in `[0,1)`.
    fn u01(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller from the LCG.
    fn gauss(state: &mut u64) -> f64 {
        let u1 = u01(state).max(f64::MIN_POSITIVE);
        let u2 = u01(state);
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    /// Circular distance between two phases in `[0,1)`.
    fn circ_err(a: f64, b: f64) -> f64 {
        let d = (a - b).rem_euclid(1.0);
        d.min(1.0 - d)
    }

    /// Wrap planted `(row, code)` firings on a single block into a minimal
    /// `BlockSparseFit`. Only the fields the coordinate readout touches
    /// (`decoder` nrows for the block-count guard, `blocks`, `gates`, `codes`,
    /// `block_size`) carry meaning; the rest are inert placeholders.
    fn fit_from_codes(codes_rows: &[Vec<f32>], b: usize) -> BlockSparseFit {
        let n = codes_rows.len();
        let mut blocks = Array2::<u32>::zeros((n, 1));
        let mut gates = Array2::<f32>::zeros((n, 1));
        let mut codes = Array3::<f32>::zeros((n, 1, b));
        for (i, z) in codes_rows.iter().enumerate() {
            blocks[[i, 0]] = 0;
            let mut nrm = 0.0f32;
            for r in 0..b {
                codes[[i, 0, r]] = z[r];
                nrm += z[r] * z[r];
            }
            gates[[i, 0]] = nrm.sqrt();
        }
        let mut decoder = Array2::<f32>::zeros((b, b));
        for i in 0..b {
            decoder[[i, i]] = 1.0;
        }
        BlockSparseFit {
            decoder,
            blocks,
            gates,
            codes,
            gamma: 1.0,
            block_utilization: vec![1.0],
            block_stable_rank: vec![1.0],
            explained_variance: 1.0,
            epochs: 1,
            converged: true,
            block_topk: 1,
            block_size: b,
        }
    }

    /// Plant a b=2 circle: zᵢ = aᵢ·u(tᵢ) + σ·noise. Returns rows, true phases,
    /// and true amplitudes. Amplitude is drawn uniformly on `[amp_lo, amp_hi]`;
    /// pass a degenerate range for the fixed-amplitude calibration regime.
    fn plant_circle(
        n: usize,
        sigma: f64,
        amp_lo: f64,
        amp_hi: f64,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<f64>, Vec<f64>) {
        let mut s = seed;
        let mut rows = Vec::with_capacity(n);
        let mut phase = Vec::with_capacity(n);
        let mut amp = Vec::with_capacity(n);
        for _ in 0..n {
            let t = u01(&mut s);
            let a = amp_lo + (amp_hi - amp_lo) * u01(&mut s);
            let z0 = a * (TAU * t).cos() + sigma * gauss(&mut s);
            let z1 = a * (TAU * t).sin() + sigma * gauss(&mut s);
            rows.push(vec![z0 as f32, z1 as f32]);
            phase.push(t);
            amp.push(a);
        }
        (rows, phase, amp)
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len();
        if n == 0 {
            return 0.0;
        }
        if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    }

    #[test]
    fn circle_phase_se_is_calibrated_and_covers() {
        // Fixed unit amplitude — the regime where the radial-scatter σ̂ is exact
        // (with varying amplitude, amplitude spread and isotropic noise are
        // non-identifiable from the radial marginal, so σ̂ is deliberately
        // conservative; see `amplitude_readout_and_conservative_sigma`).
        let sigma = 0.05;
        let n = 4000;
        let (rows, phase, _amp) = plant_circle(n, sigma, 1.0, 1.0, 0x9E3779B97F4A7C15);
        let fit = fit_from_codes(&rows, 2);
        let report = block_firing_coordinates(&fit, 0).expect("circle readout");
        assert_eq!(report.n_firings, n);

        // (3) σ̂ recovers the planted σ within 30%.
        let sig_err = (report.sigma_hat - sigma).abs() / sigma;
        assert!(
            sig_err < 0.30,
            "σ̂ = {} should recover planted σ = {sigma} within 30% (rel err {sig_err})",
            report.sigma_hat
        );

        // (1) median circular error matches the predicted median SE (the
        // normal median-|deviation| factor Φ⁻¹(3/4) ≈ 0.6745 times median t_se)
        // within a factor of ~1.5.
        let mut errs = Vec::with_capacity(n);
        let mut ses = Vec::with_capacity(n);
        let mut covered = 0usize;
        for fc in &report.firings {
            let e = circ_err(fc.t, phase[fc.row]);
            errs.push(e);
            ses.push(fc.t_se);
            if e < 2.0 * fc.t_se {
                covered += 1;
            }
        }
        let normal_mad = 0.6744897501960817; // Φ⁻¹(3/4): median |N(0,1)|.
        let predicted = normal_mad * median(ses.clone());
        let measured = median(errs);
        let ratio = measured / predicted;
        assert!(
            (1.0 / 1.5..=1.5).contains(&ratio),
            "median error {measured} vs predicted median SE {predicted} (ratio {ratio}) \
             must agree within ~1.5×"
        );

        // (2) empirical 2·SE coverage ≥ ~90%.
        let coverage = covered as f64 / n as f64;
        assert!(
            coverage >= 0.90,
            "|t̂−t| < 2·t_se coverage {coverage} must be ≥ 0.90"
        );
    }

    #[test]
    fn amplitude_readout_and_conservative_sigma() {
        // Varying amplitude aᵢ ∈ [0.5, 2]: the amplitude readout â = ‖z‖ tracks
        // the planted aᵢ, and the radial-scatter σ̂ is CONSERVATIVE — it absorbs
        // the amplitude spread and over-estimates the true noise (the documented
        // one-sided guarantee), never under-states it.
        let sigma = 0.05;
        let n = 4000;
        let (rows, _phase, amp) = plant_circle(n, sigma, 0.5, 2.0, 0x243F6A8885A308D3);
        let fit = fit_from_codes(&rows, 2);
        let report = block_firing_coordinates(&fit, 0).expect("circle readout");

        // â = ‖z‖ recovers the planted amplitude to within a few noise SDs.
        let mut amp_err = Vec::with_capacity(n);
        for fc in &report.firings {
            amp_err.push((fc.amplitude - amp[fc.row]).abs());
        }
        let med_amp_err = median(amp_err);
        assert!(
            med_amp_err < 3.0 * sigma,
            "median |â − a| = {med_amp_err} should track the planted amplitude (~σ = {sigma})"
        );

        // σ̂ is conservative: it exceeds the true σ (amplitude spread inflates it),
        // so the reported phase SE never under-states uncertainty here.
        assert!(
            report.sigma_hat > sigma,
            "radial-scatter σ̂ = {} must be conservative (> true σ = {sigma}) under \
             amplitude variation",
            report.sigma_hat
        );
    }

    #[test]
    fn weak_signal_phase_se_clamps_to_uniform_never_nan() {
        // A firing deep inside the noise is phase-unidentified: its SE clamps to
        // the uniform SD and the flag is set, never NaN. (A genuinely zero code
        // carries a zero gate and is dropped as a padded slot, so we plant a
        // non-zero but negligible code instead.)
        let mut rows = vec![vec![1.0f32, 0.0]; 8];
        rows.push(vec![1.0e-6f32, 1.0e-6]);
        let fit = fit_from_codes(&rows, 2);
        let report = block_firing_coordinates(&fit, 0).expect("readout");
        let weak = report.firings.last().unwrap();
        assert!(weak.t_se.is_finite(), "weak-signal t_se must be finite");
        assert!(weak.t_se_clamped, "weak-signal firing must flag the clamp");
        assert!(
            (weak.t_se - uniform_phase_sd()).abs() < 1.0e-12,
            "clamped t_se must equal the uniform-phase SD"
        );
        // Sanity: the strong firings are not clamped.
        assert!(!report.firings[0].t_se_clamped);
    }

    #[test]
    fn harmonic_h2_recovers_phase() {
        // b = 2H with H = 2: zᵢ = aᵢ·U(tᵢ) + σ·noise,
        // U(t) = (cos2πt, sin2πt, cos4πt, sin4πt).
        let sigma = 0.05;
        let n = 2000;
        let mut s = 0xD1B54A32D192ED03u64;
        let mut rows = Vec::with_capacity(n);
        let mut truth = Vec::with_capacity(n);
        for _ in 0..n {
            let t = u01(&mut s);
            let a = 0.5 + 1.5 * u01(&mut s);
            let mut z = [0.0f64; 4];
            for h in 0..2 {
                let w = TAU * (h + 1) as f64;
                z[2 * h] = a * (w * t).cos();
                z[2 * h + 1] = a * (w * t).sin();
            }
            for zc in z.iter_mut() {
                *zc += sigma * gauss(&mut s);
            }
            rows.push(z.iter().map(|&v| v as f32).collect::<Vec<_>>());
            truth.push(t);
        }
        let fit = fit_from_codes(&rows, 4);
        let report = harmonic_firing_coordinates(&fit, 0).expect("harmonic readout");
        assert_eq!(report.n_firings, n);

        let mut errs = Vec::with_capacity(n);
        let mut covered = 0usize;
        for fc in &report.firings {
            let e = circ_err(fc.t, truth[fc.row]);
            errs.push(e);
            if e < 2.0 * fc.t_se {
                covered += 1;
            }
        }
        let med = median(errs);
        assert!(
            med < 0.02,
            "H=2 harmonic phase median error {med} must be well under 0.02"
        );
        let coverage = covered as f64 / n as f64;
        assert!(
            coverage >= 0.85,
            "harmonic 2·SE coverage {coverage} must be ≥ 0.85"
        );
    }

    fn harmonic_code(spikes: &[(f64, f64)], h_count: usize) -> Vec<f32> {
        let measure_spikes: Vec<MeasureSpikeCoordinate> = spikes
            .iter()
            .map(|&(coordinate, amplitude)| MeasureSpikeCoordinate {
                amplitude,
                coordinate,
                coordinate_se: 0.0,
            })
            .collect();
        code_from_spikes(&measure_spikes, h_count)
            .into_iter()
            .map(|v| v as f32)
            .collect()
    }

    fn row_matrix(rows: &[Vec<f32>]) -> Array2<f32> {
        let n = rows.len();
        let p = rows[0].len();
        let mut out = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            for c in 0..p {
                out[[i, c]] = rows[i][c];
            }
        }
        out
    }

    #[test]
    fn measure_readout_recovers_two_separated_spikes_on_one_circle() {
        let h_count = 8;
        let t1 = 0.20;
        let t2 = 0.55;
        assert!(
            circ_err(t1, t2) > separation_limit(h_count),
            "planted spikes must be beyond the super-resolution separation guarantee"
        );
        let rows = vec![
            harmonic_code(&[(t1, 1.0), (t2, 0.7)], h_count),
            harmonic_code(&[(0.05, 0.9)], h_count),
        ];
        let fit = fit_from_codes(&rows, 2 * h_count);

        let single = harmonic_firing_coordinates(&fit, 0).expect("single coordinate readout");
        assert_eq!(
            single.firings.iter().filter(|f| f.row == 0).count(),
            1,
            "legacy harmonic readout emits exactly one coordinate for the two-spike row"
        );

        let report = harmonic_measure_coordinates(&fit, 0).expect("measure readout");
        let row0 = report
            .firings
            .iter()
            .find(|firing| firing.row == 0)
            .expect("row 0 firing");
        assert!(
            row0.used_super_resolution,
            "two separated modes should invoke matrix-pencil recovery"
        );
        assert_eq!(
            row0.spikes.len(),
            2,
            "measure readout must recover both spikes"
        );
        assert!(
            row0.spikes
                .iter()
                .any(|spike| circ_err(spike.coordinate, t1) < 1.0e-8),
            "missing spike at t1={t1}"
        );
        assert!(
            row0.spikes
                .iter()
                .any(|spike| circ_err(spike.coordinate, t2) < 1.0e-8),
            "missing spike at t2={t2}"
        );
    }

    #[test]
    fn measure_readout_keeps_single_spike_single() {
        let h_count = 8;
        let t = 0.37;
        let rows = vec![
            harmonic_code(&[(t, 1.25)], h_count),
            harmonic_code(&[(0.62, 0.8)], h_count),
        ];
        let fit = fit_from_codes(&rows, 2 * h_count);
        let report = harmonic_measure_coordinates(&fit, 0).expect("measure readout");
        for firing in &report.firings {
            assert_eq!(
                firing.spikes.len(),
                1,
                "single-spike harmonic code must not receive spurious multiplicity"
            );
            assert!(
                !firing.used_super_resolution,
                "exact one-spike code should stay on the single-coordinate path"
            );
        }
    }

    #[test]
    fn measure_reconstruction_improves_two_spike_ev_and_preserves_one_spike_ev() {
        let h_count = 8;
        let rows = vec![
            harmonic_code(&[(0.20, 1.0), (0.55, 0.7)], h_count),
            harmonic_code(&[(0.05, 0.9)], h_count),
        ];
        let x = row_matrix(&rows);
        let fit = fit_from_codes(&rows, 2 * h_count);
        let single_recon = reconstruct_single_coordinate_rows(&fit).expect("single recon");
        let measures = block_measure_valued_codes(&fit).expect("measure codes");
        let measure_recon =
            reconstruct_measure_valued_rows(fit.decoder.view(), &measures, rows.len(), 2 * h_count)
                .expect("measure recon");
        let single_ev =
            explained_variance_from_reconstruction(x.view(), single_recon.view()).expect("ev");
        let measure_ev =
            explained_variance_from_reconstruction(x.view(), measure_recon.view()).expect("ev");
        assert!(
            measure_ev > single_ev,
            "measure reconstruction EV {measure_ev} must improve over single-coordinate EV {single_ev}"
        );
        assert!(
            (1.0 - measure_ev).abs() < 1.0e-10,
            "exact two-spike measure reconstruction should recover the harmonic code"
        );

        let one_rows = vec![
            harmonic_code(&[(0.20, 1.0)], h_count),
            harmonic_code(&[(0.55, 0.7)], h_count),
        ];
        let one_x = row_matrix(&one_rows);
        let one_fit = fit_from_codes(&one_rows, 2 * h_count);
        let one_single = reconstruct_single_coordinate_rows(&one_fit).expect("single recon");
        let one_measures = block_measure_valued_codes(&one_fit).expect("measure codes");
        let one_measure = reconstruct_measure_valued_rows(
            one_fit.decoder.view(),
            &one_measures,
            one_rows.len(),
            2 * h_count,
        )
        .expect("measure recon");
        let one_single_ev =
            explained_variance_from_reconstruction(one_x.view(), one_single.view()).expect("ev");
        let one_measure_ev =
            explained_variance_from_reconstruction(one_x.view(), one_measure.view()).expect("ev");
        assert!(
            (one_measure_ev - one_single_ev).abs() < 1.0e-10,
            "one-spike EV must be unchanged: measure {one_measure_ev}, single {one_single_ev}"
        );
    }
}
