//! Multi-spike super-resolution on a single harmonic-circle atom — the
//! closed-form solution to the **binding problem**.
//!
//! # The problem
//!
//! An H-harmonic circle atom occupies a `b = 2H` block whose within-block code
//! `z ∈ ℝ^{2H}` splits into per-harmonic 2-vectors `(c_h, s_h)`, `h = 1..H`
//! (see [`crate::sparse_dict::block`] for the block/frame vocabulary). Reading
//! each pair as one complex Fourier coefficient `y_h = c_h + i·s_h`, the atom's
//! *generative* model when the same feature fires **m times** in one context
//! (two instances of one concept = two spikes on one circle) is a sum of `m`
//! point masses on the circle:
//!
//! ```text
//!     y_h  =  Σ_{j=1}^{m}  a_j · e^{2πi·h·t_j},   a_j > 0,  t_j ∈ [0, 1),  h = 1..H.
//! ```
//!
//! `{(a_j, t_j)}` are the amplitudes and circle positions of the spikes; the
//! block only ever stores their superposition `{y_h}`. Un-superposing them —
//! recovering `m` point masses from `H` complex Fourier coefficients — is the
//! classical **line-spectral / super-resolution** problem. This module solves
//! it in closed form (no grid, no iteration, no tuned constants) by the
//! **matrix-pencil / Prony method**: one SVD for the model order, one small
//! complex eigenproblem for the positions, one least-squares solve for the
//! amplitudes.
//!
//! # Method
//!
//! Write `z_j = e^{2πi·t_j}` (the unknowns, on the unit circle). With only
//! `h = 1..H` present (no DC term `h = 0` in this interface — the block carries
//! the harmonics, not the mean), the samples are
//! `y_h = Σ_j a_j z_j^{h}`, `h = 1..H`, an exponential sum of `H` terms.
//!
//! **1. Hankel + SVD (Hua–Sarkar matrix pencil, 1990).** With `N = H` samples
//! we build the `(N − L) × (L + 1)` Hankel matrix `Y[i, k] = y_{i+k+1}` at the
//! balanced pencil parameter `L = ⌊N/2⌋`. In the noise-free case `rank(Y) = m`,
//! so the number of exponentials equals the number of non-zero singular values.
//! Balancing `L = ⌊N/2⌋` makes both Hankel dimensions ≈ `N/2`, so the largest
//! resolvable order is `m ≤ ⌊N/2⌋ = ⌊H/2⌋` — the classical Prony identifiability
//! count. (Including the DC term would give `N = H + 1` samples and lift this to
//! `⌊(H+1)/2⌋`; this interface deliberately excludes it.)
//!
//! **2. Order selection from singular values.** The singular values of `Y`
//! separate into `m` signal values and a noise floor. When the per-component
//! coefficient noise level `sigma` is known, the split point is the
//! **Gavish–Donoho (2014) optimal hard threshold** `τ = λ(β)·σ_entry·√n`,
//! where `n = max(rows, cols)`, `β = min/max ∈ (0, 1]`, `σ_entry = √2·sigma`
//! (the standard deviation of a complex entry whose real and imaginary parts are
//! each `N(0, sigma²)`), and
//! `λ(β) = √(2(β+1) + 8β / ((β+1) + √(β²+14β+1)))` is the closed-form coefficient
//! that minimises asymptotic MSE for a rank-truncated denoiser. This is a
//! *derived* threshold — no tuned knob. (`λ(β)` is the real-matrix constant;
//! for our complex Hankel it sits safely above the complex bulk edge
//! `(1+√β)√n`, and the Hankel's anti-diagonal repetition makes the entry noise
//! correlated rather than i.i.d., so `τ` is used as a principled — not exact —
//! separator. Both approximations are conservative for *counting* the order.)
//! When `sigma ≤ 0` the estimator falls back to the pure numerical rank
//! `τ = σ_1·max(rows,cols)·ε` (the noiseless / machine-precision path).
//!
//! **3. Positions from a complex eigenproblem.** Let `V_m` hold the first `m`
//! right singular vectors (columns of `V`, each of length `L + 1`). Deleting the
//! **last** row gives `V₁` and deleting the **first** row gives `V₂`, both
//! `L × m`. Shift invariance of the Vandermonde structure makes the `z_j` the
//! eigenvalues of `Φ = V₁⁺ V₂` (a total-least-squares matrix pencil; `V₁⁺` is
//! the least-squares pseudo-inverse via QR). `Φ` is `m × m` and complex, so its
//! eigenvalues are computed by a complex `evd`. Then
//! `t_j = arg(z_j) / (2π) mod 1`.
//!
//! **4. Amplitudes by least squares.** With the `z_j` fixed, `a` solves the
//! `N × m` Vandermonde system `Σ_j a_j (z_j)^{h} = y_h`, `h = 1..H`, in the
//! least-squares sense (QR). The positions carry the phase, so each `a_j` is
//! real up to noise; the reported amplitude is its real part. The residual is
//! the ℓ₂ norm of `y − ŷ` for the recovered physical model (real amplitudes on
//! unit-modulus phasors).
//!
//! # Resolution guarantee
//!
//! [`separation_limit`] returns the Candès–Fernández-Granda (2014) super-
//! resolution threshold `Δ ≈ 2/H`: their convex-programming theory guarantees
//! exact recovery when the minimum circle separation `min_{j≠k} |t_j − t_k|`
//! (wrap-around) is at least `2/f_c` with cutoff `f_c = H`. The constant `2` is
//! their original sufficient condition (later sharpened to ≈ `1.26` by
//! Fernández-Granda 2016). It is exposed as the *theoretical guarantee*
//! threshold; the matrix-pencil estimator here routinely resolves spikes below
//! it in low noise (it is a super-resolution, not a Rayleigh-limited, method).

use faer::prelude::*;

/// A single recovered point mass on the harmonic circle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spike {
    /// Circle position `t ∈ [0, 1)` (fraction of a full turn); the underlying
    /// phasor is `e^{2πi·t}`.
    pub t: f64,
    /// Amplitude `a > 0` of the spike (the real part of the least-squares
    /// Vandermonde coefficient).
    pub amplitude: f64,
}

/// The full result of a multi-spike recovery.
#[derive(Clone, Debug)]
pub struct SpikeRecovery {
    /// Recovered spikes, sorted by circle position `t` ascending.
    pub spikes: Vec<Spike>,
    /// Selected model order `m` (number of spikes), from the numerical rank of
    /// the Hankel matrix against the noise-derived threshold.
    pub model_order: usize,
    /// ℓ₂ norm of `y − ŷ` over the `H` coefficients for the recovered physical
    /// model (real amplitudes on unit-modulus phasors).
    pub residual: f64,
    /// Singular values of the Hankel matrix, descending — the spectrum the order
    /// selection thresholded (useful for diagnostics and for auditing the split
    /// against the noise floor).
    pub hankel_singular_values: Vec<f64>,
}

/// Candès–Fernández-Granda super-resolution separation threshold `Δ ≈ 2/H` for
/// an atom carrying `n_harmonics = H` harmonics.
///
/// This is the *theoretical guarantee*: with cutoff frequency `f_c = H`, exact
/// recovery by convex programming is guaranteed once the minimum wrap-around
/// separation between spikes is at least this value. The constant `2` is the
/// original Candès–Fernández-Granda (2014) sufficient condition; Fernández-Granda
/// (2016) sharpened it to ≈ `1.26`. The matrix-pencil estimator in
/// [`recover_spikes`] typically resolves separations below this threshold when
/// noise is low.
pub fn separation_limit(n_harmonics: usize) -> f64 {
    /// Candès–Fernández-Granda (2014) sufficient-separation constant.
    const CFG_CONSTANT: f64 = 2.0;
    if n_harmonics == 0 {
        return f64::INFINITY;
    }
    CFG_CONSTANT / (n_harmonics as f64)
}

/// Recover the `m` spikes `{(a_j, t_j)}` underlying a harmonic atom's Fourier
/// coefficients by the matrix-pencil / Prony method.
///
/// `fourier_coeffs[h] = (c_{h+1}, s_{h+1})` is the within-block per-harmonic pair
/// for harmonic `h + 1`, so `fourier_coeffs` covers harmonics `1..H` with
/// `H = fourier_coeffs.len()`. `sigma` is the per-component standard deviation of
/// the additive coefficient noise (each of `Re`, `Im` distributed `N(0, sigma²)`);
/// pass `sigma ≤ 0` to select the noiseless numerical-rank path.
///
/// At most `⌊H/2⌋` spikes are identifiable (the classical Prony count); the model
/// order is chosen from the Hankel singular spectrum and clamped to this bound.
pub fn recover_spikes(
    fourier_coeffs: &[(f64, f64)],
    sigma: f64,
) -> Result<SpikeRecovery, String> {
    let n = fourier_coeffs.len();
    if n < 2 {
        return Err(format!(
            "super-resolution needs at least 2 harmonics to resolve a spike; got {n}"
        ));
    }
    for (h, &(c, s)) in fourier_coeffs.iter().enumerate() {
        if !c.is_finite() || !s.is_finite() {
            return Err(format!(
                "fourier_coeffs[{h}] = ({c}, {s}) is not finite"
            ));
        }
    }

    // Complex sample sequence y_h = c_h + i s_h for h = 1..H, stored 0-based:
    // samples[k] is harmonic h = k + 1.
    let samples: Vec<c64> = fourier_coeffs
        .iter()
        .map(|&(c, s)| c64::new(c, s))
        .collect();

    // Balanced Hankel pencil parameter L = ⌊N/2⌋; the (N−L)×(L+1) Hankel has
    // both dimensions ≈ N/2, giving the maximal identifiable order ⌊N/2⌋.
    let l = n / 2;
    let rows = n - l;
    let cols = l + 1;
    let max_order = l; // = ⌊N/2⌋

    // Hankel matrix Y[i, k] = y_{i+k+1} = samples[i + k].
    let hankel = Mat::<c64>::from_fn(rows, cols, |i, k| samples[i + k]);

    let svd = hankel
        .thin_svd()
        .map_err(|e| format!("Hankel SVD failed to converge: {e:?}"))?;
    let singular_values: Vec<f64> = svd
        .S()
        .column_vector()
        .iter()
        .map(|c| c.re)
        .collect();

    let threshold = order_threshold(&singular_values, rows, cols, sigma);
    let threshold_order = singular_values
        .iter()
        .filter(|&&s| s > threshold)
        .count()
        .min(max_order);

    // Scale-invariant rank refinement. When the singular spectrum shows an
    // unambiguous *relative* collapse `σ_{k+1}/σ_k < dramatic_gap`, that gap pins
    // the true model order regardless of the supplied `sigma`, making the decoder
    // robust to a mis-specified noise level in either direction: a caller passing a
    // conservative (over-large) population sigma would otherwise under-select and
    // drop genuine spikes, while f32-quantised inputs leave a numerical shelf that
    // sits far above the f64 numerical floor and would otherwise be over-selected.
    //
    // The threshold is derived, not tuned: it is `√(ε_f32)`, the geometric mean
    // between the f32 quantisation shelf and unity. When the codes are f32 the
    // Hankel's round-off/quantisation singular values sit at `≈ ε_f32·σ_1` relative
    // to the top singular value (`ε_f32 ≈ 1.19e-7`), so a relative drop of `√ε_f32
    // ≈ 3.4e-4` is a full ~3.5 orders of magnitude above that shelf yet far below
    // any signal structure (distinct spike amplitudes differ by O(1) ratios) or a
    // genuinely noisy spectrum (whose signal→noise relative drop equals the passed
    // noise level, which the Gavish–Donoho path already handles and which is `≫
    // √ε_f32` for any noise a caller would bother modelling). Only the quantisation
    // shelf collapses below `√ε_f32`, so honest GD selection is preserved whenever
    // no such gap exists. `√ε` also matches the Newton step tolerance used in the
    // argmax polish elsewhere in this module.
    let dramatic_gap = (f32::EPSILON as f64).sqrt();
    let gap_order = {
        let mut order = None;
        for k in 1..max_order.min(singular_values.len()) {
            if singular_values[k - 1] <= 0.0 {
                break;
            }
            if singular_values[k] / singular_values[k - 1] < dramatic_gap {
                order = Some(k);
                break;
            }
        }
        order
    };
    let model_order = gap_order.unwrap_or(threshold_order);

    if model_order == 0 {
        // Pure noise / no resolvable spike: the whole signal is the residual.
        let residual = samples.iter().map(|y| y.norm_sqr()).sum::<f64>().sqrt();
        return Ok(SpikeRecovery {
            spikes: Vec::new(),
            model_order: 0,
            residual,
            hankel_singular_values: singular_values,
        });
    }

    // Signal-subspace right singular vectors V_m (first m columns of V, each of
    // length L+1 = cols). Shift-invariance pencil: V1 = drop-last-row,
    // V2 = drop-first-row, both L×m; Φ = V1⁺ V2.
    let v = svd.V();
    let v_m = v.submatrix(0, 0, cols, model_order);
    let v1 = v_m.submatrix(0, 0, l, model_order);
    let v2 = v_m.submatrix(1, 0, l, model_order);
    let phi = v1.qr().solve_lstsq(v2);

    let roots = phi
        .eigenvalues()
        .map_err(|e| format!("matrix-pencil eigenproblem failed: {e:?}"))?;

    // Unit-modulus phasors and circle positions t = arg(z)/(2π) mod 1. The pencil
    // is built from `svd.V()`, whose columns span the conjugate of the Hankel row
    // space (faer returns `V` for the decomposition `A = U S Vᴴ`), so the pencil
    // eigenvalues come out as the conjugate phasors `z̄ = 1/z` — arg-negated, i.e.
    // reflected positions `1 − t`. Conjugating here recovers the generating phasor
    // `z = e^{2πi t}` for both the position read AND the downstream Vandermonde
    // amplitude solve (which otherwise fits the wrong frequencies and leaves a
    // large residual). See `exact_recovery_two_spikes_no_noise`.
    let phasors: Vec<c64> = roots
        .iter()
        .map(|z| {
            let norm = z.norm();
            if norm > 0.0 { z.conj() / norm } else { c64::new(1.0, 0.0) }
        })
        .collect();
    let positions: Vec<f64> = phasors
        .iter()
        .map(|z| {
            let t = z.arg() / std::f64::consts::TAU;
            if t < 0.0 { t + 1.0 } else { t }
        })
        .collect();

    // Amplitudes: least-squares Vandermonde solve of Σ_j a_j z_j^{k+1} = y_k.
    let vander = Mat::<c64>::from_fn(n, model_order, |k, j| {
        phasors[j].powu((k + 1) as u32)
    });
    let rhs = Mat::<c64>::from_fn(n, 1, |k, _| samples[k]);
    let amps = vander.qr().solve_lstsq(&rhs);

    let mut spikes: Vec<Spike> = (0..model_order)
        .map(|j| Spike {
            t: positions[j],
            amplitude: amps[(j, 0)].re,
        })
        .collect();
    spikes.sort_by(|a, b| a.t.total_cmp(&b.t));

    // Residual of the recovered physical model (real amplitudes, unit phasors).
    let mut residual_sq = 0.0;
    for (k, y) in samples.iter().enumerate() {
        let mut fit = c64::new(0.0, 0.0);
        for spike in &spikes {
            let phasor = c64::new(
                (std::f64::consts::TAU * spike.t).cos(),
                (std::f64::consts::TAU * spike.t).sin(),
            );
            fit += phasor.powu((k + 1) as u32) * spike.amplitude;
        }
        residual_sq += (y - fit).norm_sqr();
    }

    Ok(SpikeRecovery {
        spikes,
        model_order,
        residual: residual_sq.sqrt(),
        hankel_singular_values: singular_values,
    })
}

/// Gavish–Donoho (2014) optimal-hard-threshold coefficient `λ(β)` for known
/// noise level, `β ∈ (0, 1]` the matrix aspect ratio (short/long dimension).
fn optimal_hard_threshold_coefficient(beta: f64) -> f64 {
    (2.0 * (beta + 1.0)
        + 8.0 * beta / ((beta + 1.0) + (beta * beta + 14.0 * beta + 1.0).sqrt()))
    .sqrt()
}

/// Singular-value threshold separating signal from noise for order selection.
///
/// With known per-component noise `sigma > 0` this is the Gavish–Donoho optimal
/// hard threshold `λ(β)·σ_entry·√n` (`σ_entry = √2·sigma` for a complex entry).
/// Otherwise it is the numerical rank floor `σ_1·max(rows,cols)·ε`. The larger
/// of the noise threshold and the numerical floor is always used, so a perfectly
/// clean signal never over-selects on round-off.
fn order_threshold(singular_values: &[f64], rows: usize, cols: usize, sigma: f64) -> f64 {
    let sigma_1 = singular_values.first().copied().unwrap_or(0.0);
    let n_big = rows.max(cols) as f64;
    let numerical_floor = sigma_1 * n_big * f64::EPSILON;
    if sigma > 0.0 {
        let beta = rows.min(cols) as f64 / n_big;
        let sigma_entry = std::f64::consts::SQRT_2 * sigma;
        let gd = optimal_hard_threshold_coefficient(beta) * sigma_entry * n_big.sqrt();
        gd.max(numerical_floor)
    } else {
        numerical_floor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt as _;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Build exact Fourier coefficients `y_h = Σ_j a_j e^{2πi h t_j}`, `h=1..H`.
    fn coeffs_from_spikes(spikes: &[(f64, f64)], n_harmonics: usize) -> Vec<(f64, f64)> {
        (1..=n_harmonics)
            .map(|h| {
                let mut c = 0.0;
                let mut s = 0.0;
                for &(t, a) in spikes {
                    let phase = std::f64::consts::TAU * (h as f64) * t;
                    c += a * phase.cos();
                    s += a * phase.sin();
                }
                (c, s)
            })
            .collect()
    }

    /// Add independent N(0, sigma²) noise to each real and imaginary component.
    fn add_noise(coeffs: &mut [(f64, f64)], sigma: f64, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        for coeff in coeffs.iter_mut() {
            coeff.0 += sigma * gaussian(&mut rng);
            coeff.1 += sigma * gaussian(&mut rng);
        }
    }

    fn gaussian(rng: &mut StdRng) -> f64 {
        let u1 = rng.random::<f64>().max(1e-16);
        let u2 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Wrap-around circle distance between two positions in `[0, 1)`.
    fn circle_dist(a: f64, b: f64) -> f64 {
        let d = (a - b).abs();
        d.min(1.0 - d)
    }

    /// Greedy match of recovered spikes to planted (both by position), returning
    /// max position error and max amplitude error. Requires equal counts.
    fn match_error(recovered: &[Spike], planted: &[(f64, f64)]) -> (f64, f64) {
        assert_eq!(recovered.len(), planted.len(), "spike-count mismatch");
        let mut planted_sorted = planted.to_vec();
        planted_sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut max_t = 0.0_f64;
        let mut max_a = 0.0_f64;
        for (rec, &(t, a)) in recovered.iter().zip(planted_sorted.iter()) {
            max_t = max_t.max(circle_dist(rec.t, t));
            max_a = max_a.max((rec.amplitude - a).abs());
        }
        (max_t, max_a)
    }

    #[test]
    fn exact_recovery_two_spikes_no_noise() {
        // m = 2, H = 8, separation 2/H = 0.25, no noise.
        let h = 8;
        let planted = [(0.20, 1.0), (0.45, 0.7)];
        let coeffs = coeffs_from_spikes(&planted, h);
        let rec = recover_spikes(&coeffs, 0.0).expect("recovery");
        assert_eq!(rec.model_order, 2, "order from clean singular values");
        let (t_err, a_err) = match_error(&rec.spikes, &planted);
        assert!(t_err < 1e-9, "position error {t_err:.3e}");
        assert!(a_err < 1e-9, "amplitude error {a_err:.3e}");
        assert!(rec.residual < 1e-9, "residual {:.3e}", rec.residual);
    }

    #[test]
    fn separation_law_brackets_the_limit() {
        // H = 8: sweep planted separations {0.5, 1.0, 2.0}·(1/H) at small noise.
        let h = 8;
        let sigma = 1e-3;
        let base = 1.0 / (h as f64);
        let mut errors = Vec::new();
        for factor in [0.5_f64, 1.0, 2.0] {
            let sep = factor * base;
            let planted = [(0.30, 1.0), (0.30 + sep, 1.0)];
            let mut coeffs = coeffs_from_spikes(&planted, h);
            add_noise(&mut coeffs, sigma, 0xC0FFEE + (factor * 1000.0) as u64);
            let rec = recover_spikes(&coeffs, sigma).expect("recovery");
            // Order may collapse to 1 when spikes are unresolvable; treat that as
            // a large (merged) position error.
            let err = if rec.model_order == 2 {
                match_error(&rec.spikes, &planted).0
            } else {
                sep.max(base)
            };
            errors.push(err);
        }
        let (err_half, _err_one, err_two) = (errors[0], errors[1], errors[2]);
        // At 2/H (the guarantee threshold) recovery is accurate...
        assert!(
            err_two < 0.02,
            "position error at 2/H should be small, got {err_two:.3e}"
        );
        // ...and the sub-limit 0.5/H case is markedly worse — the limit brackets
        // success and failure sensibly (a constant-factor law, not a hard edge).
        assert!(
            err_half > err_two,
            "0.5/H error {err_half:.3e} should exceed 2/H error {err_two:.3e}"
        );
    }

    #[test]
    fn model_order_selection_from_singular_values() {
        // One vs three spikes, order chosen from singular values at sigma = 1e-2.
        let h = 8;
        let sigma = 1e-2;

        let planted1 = [(0.35, 1.0)];
        let mut c1 = coeffs_from_spikes(&planted1, h);
        add_noise(&mut c1, sigma, 11);
        let rec1 = recover_spikes(&c1, sigma).expect("recovery m=1");
        assert_eq!(rec1.model_order, 1, "should select order 1");

        let planted3 = [(0.10, 1.0), (0.40, 0.9), (0.75, 1.1)];
        let mut c3 = coeffs_from_spikes(&planted3, h);
        add_noise(&mut c3, sigma, 22);
        let rec3 = recover_spikes(&c3, sigma).expect("recovery m=3");
        assert_eq!(rec3.model_order, 3, "should select order 3");
        let (t_err, _) = match_error(&rec3.spikes, &planted3);
        assert!(t_err < 0.05, "order-3 position error {t_err:.3e}");
    }

    #[test]
    fn noise_robustness_two_spikes() {
        // m = 2, well separated, sigma = 0.05; positions within a few times the
        // sigma-scaled bound.
        let h = 8;
        let sigma = 0.05;
        let planted = [(0.20, 1.0), (0.65, 1.0)];
        let mut coeffs = coeffs_from_spikes(&planted, h);
        add_noise(&mut coeffs, sigma, 777);
        let rec = recover_spikes(&coeffs, sigma).expect("recovery");
        assert_eq!(rec.model_order, 2, "order under moderate noise");
        let (t_err, a_err) = match_error(&rec.spikes, &planted);
        // Perturbation bound for a well-separated pencil is O(sigma); allow a
        // small constant factor.
        assert!(t_err < 10.0 * sigma / (h as f64), "position error {t_err:.3e}");
        assert!(a_err < 5.0 * sigma, "amplitude error {a_err:.3e}");
    }

    #[test]
    fn separation_limit_is_two_over_h() {
        assert!((separation_limit(8) - 0.25).abs() < 1e-15);
        assert!((separation_limit(16) - 0.125).abs() < 1e-15);
        assert!(separation_limit(0).is_infinite());
    }

    #[test]
    fn too_few_harmonics_errors() {
        assert!(recover_spikes(&[(1.0, 0.0)], 0.0).is_err());
        assert!(recover_spikes(&[], 0.0).is_err());
    }
}
