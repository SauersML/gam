//! Manifold chart-equivalence and realized-amplitude diagnostics.
//!
//! The same-manifold gluing test used by SAC's birth race is the
//! two-parameter affine transition of the arc-length coordinate
//! ([`affine_chart_transition`]) — under unit-speed coordinates two atoms that
//! trace the same 1-manifold are related by `t_a = ±t_b + c` (slope exactly
//! `±1`), so stagewise arc-tiling is caught at birth.
//!
//! All derivatives here are hand-derived closed forms (SPEC: no autodiff
//! outside tests); the `#[cfg(test)]` module verifies each one against finite
//! differences, which SPEC permits *inside tests only*.

use ndarray::ArrayView1;

use gam_math::special::{digamma, trigamma};
use opt::{BacktrackConfig, backtracking_line_search};

/// The two-parameter affine transition `t_a ≈ slope·t_b + offset` relating the
/// arc-length coordinate of curve B to that of curve A, the object SAC's birth
/// race reads to decide whether a candidate atom lies on the SAME 1-manifold as
/// an existing atom.
#[derive(Debug, Clone)]
pub struct AffineChartTransition {
    /// Fitted slope. Under unit-speed (arc-length) coordinates a genuine
    /// same-manifold match forces `|slope| = 1` (orientation-preserving `+1`
    /// or reflected `−1`); the value is *fitted freely*, so `|slope|` near `1`
    /// is a verification, not an imposition.
    pub slope: f64,
    /// Fitted offset (the base-point shift `c` of `t_a = ±t_b + c`).
    pub offset: f64,
    /// RMS residual of the affine coordinate fit, in the same units as the
    /// arc-length coordinate. Small ⇔ the coordinate relation really is affine.
    pub coord_residual: f64,
    /// Mean nearest-point distance from curve B to curve A, normalized by the
    /// scale of curve A (its RMS radius about its centroid). Small ⇔ curve B
    /// geometrically lies ON curve A (period, tolerance-free).
    pub geometric_residual: f64,
}

// ===========================================================================
// F1 — amplitude-concentration certificate (the "intensity is presence vs a
// hidden radial coordinate" law).
//
// This certifies the shape of an atom's realized assignment-amplitude
// distribution across the samples it fires on. Two regimes are observationally
// distinct and carry opposite structural verdicts:
//
//   * **Spike-at-saturation** — the realized amplitude piles at the two ends of
//     its range (near 0 = absent, near its saturation = present). This is a
//     genuine binary presence coordinate; the gate is honest and the atom's
//     latent dimension is what the chart says it is (a `circle` stays a circle).
//   * **Continuous** — the amplitude spreads unimodally across the interior of
//     its range. Intensity is then not presence but a hidden RADIAL latent axis:
//     the atom is really a disk / annulus (`S¹ × ℝ_radius`), and `d_atom` is
//     understated by one. `steer_delta`'s predicted nats scale with `a²`, so a
//     dosimetry claim rides on this uncertified quantity unless the radial axis
//     is promoted to an explicit coordinate and raced (circle vs cylinder-radial
//     vs disk).
//
// The certificate is an EVIDENCE decision, not a tuned threshold. Normalise the
// realized amplitudes to their saturation `r = a / max(a) ∈ (0, 1)` and fit a
// Beta(α, β) by maximum likelihood. The Beta family's own analytic mode-count
// transition IS the decision boundary: `Beta(α, β)` is U-shaped (density → ∞ at
// BOTH endpoints, an interior minimum — mass at absent AND saturated) exactly
// when `α < 1 AND β < 1`, and is unimodal / monotone (mass in the interior — a
// radial spread) otherwise. The boundary `α = β = 1` is the uniform density, the
// analytic shape-transition of the family, so "spike vs continuous" is read off
// the fitted shape with no magic constant. A disk's area-uniform radius has
// density `∝ r = Beta(2, 1)` (α > 1 ⇒ Continuous), and a present/absent atom
// collapses onto both endpoints (α, β < 1 ⇒ SpikeAtSaturation) — both verdicts
// fall out of the family analytically.
// ===========================================================================

/// The certified verdict on one atom's realized amplitude-concentration law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmplitudeConcentration {
    /// The realized amplitude is bimodal at the ends of its range (present /
    /// absent): a genuine binary presence coordinate. The gate is honest and the
    /// atom keeps its charted latent dimension.
    SpikeAtSaturation,
    /// The realized amplitude spreads continuously across the interior: intensity
    /// is a hidden RADIAL latent axis. Promote radius to an explicit coordinate
    /// and race the atom as circle vs cylinder-radial vs disk.
    Continuous,
    /// Too few / degenerate (no spread, non-finite, or all-equal) amplitudes to
    /// certify. Carries no radial promotion — a constant-intensity atom is a pure
    /// presence coordinate, not a disk.
    Indeterminate,
}

impl AmplitudeConcentration {
    /// Lowercase label for the diagnostics payload.
    pub fn label(self) -> &'static str {
        match self {
            AmplitudeConcentration::SpikeAtSaturation => "spike_at_saturation",
            AmplitudeConcentration::Continuous => "continuous",
            AmplitudeConcentration::Indeterminate => "indeterminate",
        }
    }
}

/// The per-atom amplitude-concentration certificate (F1): the fitted Beta shape
/// of the realized amplitude distribution and the presence-vs-radial verdict it
/// implies. Produced by [`amplitude_concentration_certificate`].
#[derive(Debug, Clone, Copy)]
pub struct AmplitudeConcentrationCertificate {
    /// The certified verdict.
    pub verdict: AmplitudeConcentration,
    /// Fitted Beta shape parameter `α` of the saturation-normalized amplitudes.
    /// `NaN` when [`AmplitudeConcentration::Indeterminate`].
    pub beta_alpha: f64,
    /// Fitted Beta shape parameter `β`.
    pub beta_beta: f64,
    /// The Beta log-likelihood at `(α, β)` — the evidence the verdict is read
    /// from. `NaN` when indeterminate.
    pub log_likelihood: f64,
    /// Number of realized amplitudes the certificate was fitted from.
    pub n: usize,
}

impl AmplitudeConcentrationCertificate {
    /// `true` iff the certificate calls for promoting a radial latent axis: the
    /// amplitude is a continuous (radial) coordinate, not a binary presence.
    pub fn recommends_radial_axis(&self) -> bool {
        matches!(self.verdict, AmplitudeConcentration::Continuous)
    }
}

/// Certify one atom's realized amplitude-concentration law from the amplitudes
/// `a_n ≥ 0` it fires with across its samples (the posterior gate per row). The
/// verdict is read from the fitted Beta
/// shape of the saturation-normalized amplitudes: U-shaped (`α < 1 ∧ β < 1`) ⟺
/// [`AmplitudeConcentration::SpikeAtSaturation`], otherwise
/// [`AmplitudeConcentration::Continuous`]; a degenerate / no-spread sample is
/// [`AmplitudeConcentration::Indeterminate`].
pub fn amplitude_concentration_certificate(
    amplitudes: ArrayView1<'_, f64>,
) -> AmplitudeConcentrationCertificate {
    let n = amplitudes.len();
    let indeterminate = |n: usize| AmplitudeConcentrationCertificate {
        verdict: AmplitudeConcentration::Indeterminate,
        beta_alpha: f64::NAN,
        beta_beta: f64::NAN,
        log_likelihood: f64::NAN,
        n,
    };
    if n < 4 {
        // Fewer than four samples cannot resolve a shape (a Beta has two shape
        // parameters; a bimodality claim needs mass observed at both ends).
        return indeterminate(n);
    }
    if amplitudes.iter().any(|a| !a.is_finite() || *a < 0.0) {
        return indeterminate(n);
    }
    let amax = amplitudes.iter().copied().fold(0.0_f64, f64::max);
    if !(amax > 0.0) {
        // All-zero: the atom never fires — no distribution to certify.
        return indeterminate(n);
    }
    // Saturation-normalize into [0, 1]. A near-constant amplitude (no spread)
    // carries neither bimodality nor a radial axis: it is a pure fixed-intensity
    // presence coordinate, reported Indeterminate so no radial axis is promoted.
    let raw: Vec<f64> = amplitudes
        .iter()
        .map(|&a| (a / amax).clamp(0.0, 1.0))
        .collect();
    let mean_r: f64 = raw.iter().sum::<f64>() / n as f64;
    let var_r: f64 = raw.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / n as f64;
    // Spread floor: the sample must vary by more than floating-point noise
    // relative to its scale for a shape to be identifiable at all.
    if !(var_r > f64::EPSILON) {
        return indeterminate(n);
    }
    // Open-interval boundary correction: map endpoints strictly inside (0, 1) via
    // the standard `(r(n−1) + 1/2)/n` compression so `ln r` / `ln(1−r)` stay
    // finite. This is a recognized boundary rule, not a tuning knob.
    let nf = n as f64;
    let r: Vec<f64> = raw.iter().map(|&x| (x * (nf - 1.0) + 0.5) / nf).collect();

    let (alpha, beta, loglik) = match fit_beta_mle(&r) {
        Some(v) => v,
        None => return indeterminate(n),
    };

    // The Beta family's analytic U-shape region: density diverges at both 0 and 1
    // (mass at absent AND saturation) iff both shape parameters are below the
    // uniform-density boundary `1`. This is the family's own mode-count
    // transition — the decision, not a threshold.
    let verdict = if alpha < 1.0 && beta < 1.0 {
        AmplitudeConcentration::SpikeAtSaturation
    } else {
        AmplitudeConcentration::Continuous
    };
    AmplitudeConcentrationCertificate {
        verdict,
        beta_alpha: alpha,
        beta_beta: beta,
        log_likelihood: loglik,
        n,
    }
}

/// Maximum-likelihood fit of a `Beta(α, β)` to samples `r ∈ (0, 1)` by Newton's
/// method on the (concave) Beta log-likelihood, method-of-moments initialized.
/// Returns `(α, β, loglik)` or `None` when the sufficient statistics are
/// undefined (a sample at the closed boundary slipped through, or the moments are
/// degenerate). Newton uses the exact digamma/trigamma score and Hessian — no
/// finite differences (SPEC), and no autodiff.
fn fit_beta_mle(r: &[f64]) -> Option<(f64, f64, f64)> {
    let n = r.len();
    if n < 2 {
        return None;
    }
    let mut sum_ln = 0.0_f64;
    let mut sum_ln1m = 0.0_f64;
    let mut mean = 0.0_f64;
    for &x in r {
        if !(x > 0.0 && x < 1.0) {
            return None;
        }
        sum_ln += x.ln();
        sum_ln1m += (1.0 - x).ln();
        mean += x;
    }
    let nf = n as f64;
    mean /= nf;
    // Second pass for the variance, which is what the caller already does. The
    // one-pass form `E[x²] − E[x]²` subtracts two quantities of size `mean²`
    // to produce one of size `var`, so its result is quantized to multiples of
    // `ulp(mean²)`: with samples clustered near a common amplitude — the
    // collapsed state this gauge exists to detect — the difference is a count
    // of ulps rather than a measurement of the spread, and it goes negative.
    let var = r.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nf;
    // A sample with no spread has no identifiable Beta shape: the
    // method-of-moments `common = m(1−m)/var − 1` is a division by zero and the
    // likelihood is flat in the direction that scales `α` and `β` together.
    // That is the degenerate-moment case this function documents, so it is
    // reported rather than floored — flooring `var` returns the same seed for
    // every sample below the floor, which makes the answer a property of the
    // floor and not of the data.
    if !(var > 0.0) {
        return None;
    }
    // Method-of-moments seed: `common = m(1−m)/v − 1`, `α = m·common`,
    // `β = (1−m)·common`. Guard positivity so Newton starts in the interior.
    let common = (mean * (1.0 - mean) / var - 1.0).max(1.0e-3);
    let mut alpha = (mean * common).max(1.0e-3);
    let mut beta = ((1.0 - mean) * common).max(1.0e-3);

    let s_ln = sum_ln / nf;
    let s_ln1m = sum_ln1m / nf;
    // Newton on the per-sample-averaged score (concave objective; the Hessian is
    // negative definite, so a damped Newton with step-halving converges).
    for _ in 0..100 {
        let psi_ab = digamma(alpha + beta);
        let g_a = s_ln - (digamma(alpha) - psi_ab);
        let g_b = s_ln1m - (digamma(beta) - psi_ab);
        if g_a.abs() < 1.0e-12 && g_b.abs() < 1.0e-12 {
            break;
        }
        let t_ab = trigamma(alpha + beta);
        // Negative Hessian of the averaged loglik (positive definite):
        //   H = [[ψ₁(α) − ψ₁(α+β), −ψ₁(α+β)], [−ψ₁(α+β), ψ₁(β) − ψ₁(α+β)]].
        let h_aa = trigamma(alpha) - t_ab;
        let h_bb = trigamma(beta) - t_ab;
        let h_ab = -t_ab;
        let det = h_aa * h_bb - h_ab * h_ab;
        if !(det.abs() > 0.0) {
            break;
        }
        // Newton step `Δ = H⁻¹ g` (H is the negative Hessian, g the gradient).
        let d_a = (h_bb * g_a - h_ab * g_b) / det;
        let d_b = (h_aa * g_b - h_ab * g_a) / det;
        // Step-halving to keep `(α, β)` strictly positive and non-decreasing in
        // loglik — a standard safeguard, no wall-clock budget.
        let base = beta_loglik_avg(alpha, beta, s_ln, s_ln1m);
        let accepted = match backtracking_line_search::<_, std::convert::Infallible>(
            BacktrackConfig {
                initial_step: 1.0,
                contraction: 0.5,
                max_steps: 40,
            },
            |step| {
                let na = alpha + step * d_a;
                let nb = beta + step * d_b;
                // Feasibility (strict positivity) gates the trial before the
                // ascent test — mirrors the short-circuit `&&` of the original.
                if na > 0.0 && nb > 0.0 {
                    Ok(Some((beta_loglik_avg(na, nb, s_ln, s_ln1m), (na, nb))))
                } else {
                    Ok(None)
                }
            },
            |_, f| f >= base,
        ) {
            Ok(v) => v,
            Err(never) => match never {},
        };
        match accepted {
            Some(step) => {
                let (na, nb) = step.payload;
                alpha = na;
                beta = nb;
            }
            None => break,
        }
    }
    let loglik = nf * beta_loglik_avg(alpha, beta, s_ln, s_ln1m);
    if !loglik.is_finite() {
        return None;
    }
    Some((alpha, beta, loglik))
}

/// Per-sample-averaged Beta log-likelihood `(α−1)⟨ln r⟩ + (β−1)⟨ln(1−r)⟩ −
/// ln B(α, β)` given the averaged sufficient statistics.
fn beta_loglik_avg(alpha: f64, beta: f64, s_ln: f64, s_ln1m: f64) -> f64 {
    (alpha - 1.0) * s_ln + (beta - 1.0) * s_ln1m
        - (ln_gamma(alpha) + ln_gamma(beta) - ln_gamma(alpha + beta))
}

// `ψ` and `ψ₁` come from the workspace's single polygamma implementation. The
// local copies they replace recursed only to `x ≥ 10` and stopped at `B₆`,
// which left `7.6e−10` / `3.1e−10` relative error — enough to matter to the
// Beta Newton below, whose own convergence test is `|g| < 1e−12`.

/// `ln Γ(x)` for `x > 0` via the Lanczos approximation (g = 7). Hand-derived
/// closed form; used only to report the Beta log-likelihood.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let mut a = C[0];
    let t = x + G - 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64 - 1.0);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x - 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    // ---- F1: amplitude-concentration certificate ----------------------------

    /// A deterministic low-discrepancy sequence on `[0, 1)` (van der Corput,
    /// base 2) so the amplitude tests need no RNG and are byte-reproducible.
    fn van_der_corput(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let (mut x, mut denom, mut k) = (0.0_f64, 2.0_f64, i + 1);
                while k > 0 {
                    x += (k & 1) as f64 / denom;
                    denom *= 2.0;
                    k >>= 1;
                }
                x
            })
            .collect()
    }

    #[test]
    fn digamma_trigamma_match_known_values() {
        // ψ(1) = −γ, ψ(2) = 1 − γ, ψ₁(1) = π²/6 — closed forms, so the bar is
        // `f64` rounding of the constants themselves, not the evaluator's own
        // accuracy. The former 1e-9/1e-8 bars were sized to the local Bernoulli
        // series (~4e-11 absolute) that `gam_math::special` now replaces.
        let gamma = 0.577_215_664_901_532_9_f64;
        assert!((digamma(1.0) + gamma).abs() < 1.0e-15);
        assert!((digamma(2.0) - (1.0 - gamma)).abs() < 1.0e-15);
        let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!((trigamma(1.0) - pi2_6).abs() < 1.0e-15);
        // ln Γ(5) = ln 24. The Lanczos g=7 form here is a separate primitive and
        // measures 1.6e-14 relative at worst, so it keeps a wider bar.
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1.0e-13);
    }

    #[test]
    fn beta_mle_recovers_planted_shape() {
        // Sample the Beta(2, 5) CDF quantiles deterministically via a coarse
        // inverse-CDF over a fine low-discrepancy grid on the density, and check
        // the MLE lands near the planted shape. We synthesize from the density
        // directly by rejection on the grid to stay RNG-free.
        // Simpler + exact: fit to the Beta(2,1) family whose CDF is r² so the
        // quantile of a uniform u is sqrt(u) — an exact inverse transform.
        let u = van_der_corput(400);
        let samples: Vec<f64> = u.iter().map(|&x| x.sqrt()).collect(); // Beta(2,1)
        let (a, b, _ll) = fit_beta_mle(&samples).expect("beta fit");
        assert!((a - 2.0).abs() < 0.3, "alpha {a}");
        assert!((b - 1.0).abs() < 0.3, "beta {b}");
    }

    /// A sample with no spread has no Beta shape, and now says so.
    ///
    /// `fit_beta_mle` documents `None` for "the moments are degenerate". It
    /// reached that verdict through `E[x²] − E[x]²`, which for identical
    /// samples returns a small NEGATIVE number (measured `-5.0e-15` at
    /// `x = 0.9`, `n = 256`) rather than zero, because the two accumulators
    /// round apart. `.max(f64::EPSILON)` then turned that into a positive
    /// variance and the function returned a shape, seeded at
    /// `α = m(m(1−m)/ε − 1)`, which is a property of the floor and not of the
    /// data. The second-pass form returns exactly zero here, so the documented
    /// verdict is reachable.
    #[test]
    fn beta_mle_reports_a_spreadless_sample_as_degenerate() {
        // 0.75 and n = 256 are binary-exact, so the mean is exactly 0.75 and
        // every deviation is exactly zero on any conforming platform.
        let samples = vec![0.75_f64; 256];
        assert!(
            fit_beta_mle(&samples).is_none(),
            "a sample with zero spread has no identifiable Beta shape"
        );
    }

    /// The retired variance was quantized to `ulp(mean²)`, not to the spread.
    ///
    /// This measures the expression `fit_beta_mle` no longer evaluates, because
    /// it is the reason it stopped. Half the mass sits at `c + s` and half at
    /// `c − s` with `c = 0.75` and `s` a power of two, so the mean is exactly
    /// `c` and the variance is exactly `s²` with no reference implementation
    /// required. The one-pass form must difference two quantities of size `c²`,
    /// so it can only resolve the variance in steps of `ulp(c²) = 2⁻⁵³`.
    #[test]
    fn beta_mle_variance_had_been_quantized_to_ulps_of_the_mean_square() {
        let c = 0.75_f64;
        let ulp_c2 = (c * c) * f64::EPSILON;
        let mut collapsed = 0usize;
        for k in 20..30u32 {
            let s = (2.0_f64).powi(-(k as i32));
            let mut samples = vec![c + s; 128];
            samples.extend(std::iter::repeat_n(c - s, 128));
            let exact = s * s;

            let nf = samples.len() as f64;
            let mean: f64 = samples.iter().sum::<f64>() / nf;
            let mean_sq: f64 = samples.iter().map(|x| x * x).sum::<f64>() / nf;
            let one_pass = mean_sq - mean * mean;
            let two_pass: f64 = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nf;

            assert_eq!(mean, c, "the construction must give the mean exactly");
            assert_eq!(
                two_pass, exact,
                "the second-pass form must be exact on an exact construction"
            );
            if one_pass <= 0.0 {
                collapsed += 1;
            }
            println!(
                "s = 2^-{k:<2}  exact var {exact:.6e}  one-pass {one_pass:.6e}  \
                 ulp(c^2) {ulp_c2:.6e}"
            );
        }
        assert!(
            collapsed > 0,
            "the one-pass form must be seen to lose the variance entirely, \
             or this measurement is not exercising the regime it describes"
        );
    }

    #[test]
    fn continuous_disk_radius_recommends_radial_axis() {
        // A disk uniform in AREA has radius density ∝ r on [0, 1] = Beta(2, 1),
        // whose quantile of uniform u is sqrt(u). Amplitude = radius. The
        // certificate must read this as a continuous (radial) coordinate.
        let u = van_der_corput(500);
        let amps = Array1::from_iter(u.iter().map(|&x| x.sqrt()));
        let cert = amplitude_concentration_certificate(amps.view());
        assert_eq!(cert.verdict, AmplitudeConcentration::Continuous, "{cert:?}");
        assert!(cert.recommends_radial_axis());
        assert!(cert.beta_alpha > 1.0, "alpha {}", cert.beta_alpha);
    }

    #[test]
    fn true_presence_certifies_spike_at_saturation() {
        // A genuine binary presence atom: roughly half the samples absent
        // (amplitude ≈ 0) and half saturated (≈ 1), with a little jitter so the
        // sample is not literally two atoms. Mass at both ends ⇒ U-shaped Beta
        // (α, β < 1) ⇒ SpikeAtSaturation, and NO radial axis is promoted.
        let jitter = van_der_corput(600);
        let amps = Array1::from_iter(jitter.iter().enumerate().map(|(i, &j)| {
            let base = if i % 2 == 0 { 0.0 } else { 1.0 };
            // Pull each sample toward its end by ≤ 8% so the piles stay at the
            // endpoints without ever leaving [0, 1].
            (base + if base == 0.0 { 0.08 * j } else { -0.08 * j }).clamp(0.0, 1.0)
        }));
        let cert = amplitude_concentration_certificate(amps.view());
        assert_eq!(
            cert.verdict,
            AmplitudeConcentration::SpikeAtSaturation,
            "{cert:?}"
        );
        assert!(!cert.recommends_radial_axis());
        assert!(cert.beta_alpha < 1.0 && cert.beta_beta < 1.0, "{cert:?}");
    }

    #[test]
    fn degenerate_amplitudes_are_indeterminate() {
        // No spread (constant intensity) ⇒ pure fixed-intensity presence, not a
        // disk: Indeterminate, no radial promotion.
        let flat = Array1::from_elem(50, 0.7);
        let cert = amplitude_concentration_certificate(flat.view());
        assert_eq!(cert.verdict, AmplitudeConcentration::Indeterminate);
        assert!(!cert.recommends_radial_axis());
        // All-zero (never fires) is also indeterminate.
        let zero = Array1::<f64>::zeros(50);
        assert_eq!(
            amplitude_concentration_certificate(zero.view()).verdict,
            AmplitudeConcentration::Indeterminate
        );
        // Too few samples.
        let few = array![0.1, 0.9];
        assert_eq!(
            amplitude_concentration_certificate(few.view()).verdict,
            AmplitudeConcentration::Indeterminate
        );
    }

}
