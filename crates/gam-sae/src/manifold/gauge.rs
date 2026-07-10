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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::{SaeBasisEvaluator, Side};
use opt::{BacktrackConfig, backtracking_line_search};

pub fn sample_decoded_curve(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    coords: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let n = coords.len();
    let mut coords2 = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        coords2[[i, 0]] = coords[i];
    }
    let (phi, _jet) = evaluator.evaluate(coords2.view())?;
    if phi.ncols() != decoder.nrows() {
        return Err(format!(
            "sample_decoded_curve: basis width {} != decoder rows {}",
            phi.ncols(),
            decoder.nrows()
        ));
    }
    Ok(phi.dot(&decoder))
}

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

impl AffineChartTransition {
    /// Same-manifold verdict at an explicit relative tolerance. Requires (i) the
    /// fitted slope to be within `rel_tol` of `±1` (arc-length rigidity), (ii)
    /// the affine coordinate residual to be within `rel_tol` of the coordinate
    /// scale `coord_scale` (the span of curve B's parameter), and (iii) the
    /// geometric residual within `rel_tol` (curve B lies on curve A).
    ///
    /// `rel_tol` is the caller's salience dial (SAC_PLAN Part 2: salience is a
    /// separate, explicit dial) — deliberately NOT hard-coded here, so this file
    /// carries no acceptance magic constant.
    pub fn same_manifold(&self, coord_scale: f64, rel_tol: f64) -> bool {
        let slope_ok = (self.slope.abs() - 1.0).abs() <= rel_tol;
        let coord_ok = coord_scale > 0.0 && self.coord_residual <= rel_tol * coord_scale;
        let geom_ok = self.geometric_residual <= rel_tol;
        slope_ok && coord_ok && geom_ok
    }
}

/// Fit the two-parameter affine transition between two arc-length-parameterized
/// curves. `points_a`/`points_b` are `(n_a × p)`/`(n_b × p)` point sets sampled
/// along the two decoded curves; `coords_a`/`coords_b` are their (arc-length)
/// latent coordinates. `period_a`, when `Some(P)`, unwraps the matched
/// `coord_a` sequence across the `S¹` branch cut so the regression is not
/// corrupted by the wrap (pass `None` for an interval/line chart).
///
/// Method (deterministic, closed-form, no autodiff): for each point of curve B
/// find its nearest point on curve A, giving a correspondence `(coord_b_j,
/// coord_a_j)` plus the point-to-point distance. Ordinary least squares on the
/// (branch-unwrapped) correspondences yields `slope`/`offset`; the RMS fit
/// residual is `coord_residual`; the mean matched distance normalized by curve
/// A's scale is `geometric_residual`.
pub fn affine_chart_transition(
    points_a: ArrayView2<'_, f64>,
    coords_a: ArrayView1<'_, f64>,
    points_b: ArrayView2<'_, f64>,
    coords_b: ArrayView1<'_, f64>,
    period_a: Option<f64>,
) -> Result<AffineChartTransition, String> {
    let (na, p) = points_a.dim();
    let (nb, pb) = points_b.dim();
    if p != pb {
        return Err(format!(
            "affine_chart_transition: output dims differ (a: {p}, b: {pb})"
        ));
    }
    if na != coords_a.len() || nb != coords_b.len() {
        return Err(format!(
            "affine_chart_transition: point/coord length mismatch (a: {na} vs {}, b: {nb} vs {})",
            coords_a.len(),
            coords_b.len()
        ));
    }
    if na < 2 || nb < 2 {
        return Err("affine_chart_transition: need at least two samples per curve".into());
    }

    // Curve A scale: RMS radius about its centroid, the normalizer for the
    // geometric residual (period-agnostic, tolerance-free).
    let mut centroid = vec![0.0_f64; p];
    for i in 0..na {
        for j in 0..p {
            centroid[j] += points_a[[i, j]];
        }
    }
    for c in centroid.iter_mut() {
        *c /= na as f64;
    }
    let mut scale_sq = 0.0_f64;
    for i in 0..na {
        for j in 0..p {
            let d = points_a[[i, j]] - centroid[j];
            scale_sq += d * d;
        }
    }
    let curve_scale = (scale_sq / na as f64).sqrt();

    // Nearest-A correspondence for every B point.
    let mut xs = Vec::with_capacity(nb); // coord_b
    let mut ys = Vec::with_capacity(nb); // coord_a of nearest A point
    let mut dist_sum = 0.0_f64;
    for jb in 0..nb {
        let mut best = f64::INFINITY;
        let mut best_i = 0usize;
        for ia in 0..na {
            let mut d = 0.0_f64;
            for c in 0..p {
                let diff = points_b[[jb, c]] - points_a[[ia, c]];
                d += diff * diff;
            }
            if d < best {
                best = d;
                best_i = ia;
            }
        }
        dist_sum += best.sqrt();
        xs.push(coords_b[jb]);
        ys.push(coords_a[best_i]);
    }
    let geometric_residual = if curve_scale > 0.0 {
        (dist_sum / nb as f64) / curve_scale
    } else {
        f64::INFINITY
    };

    // Order correspondences by coord_b and branch-unwrap coord_a so a circle
    // atom whose arc-length coordinate wraps modulo the period does not inject a
    // spurious `±P` jump into the regression.
    let mut order: Vec<usize> = (0..nb).collect();
    order.sort_by(|&i, &j| {
        xs[i]
            .partial_cmp(&xs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let xo: Vec<f64> = order.iter().map(|&i| xs[i]).collect();
    let mut yo: Vec<f64> = order.iter().map(|&i| ys[i]).collect();
    if let Some(pp) = period_a {
        if pp > 0.0 {
            for idx in 1..yo.len() {
                let mut d = yo[idx] - yo[idx - 1];
                while d > 0.5 * pp {
                    yo[idx] -= pp;
                    d -= pp;
                }
                while d < -0.5 * pp {
                    yo[idx] += pp;
                    d += pp;
                }
            }
        }
    }

    // Ordinary least squares slope/offset on the unwrapped correspondences.
    let m = xo.len() as f64;
    let mean_x = xo.iter().sum::<f64>() / m;
    let mean_y = yo.iter().sum::<f64>() / m;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    for idx in 0..xo.len() {
        let dx = xo[idx] - mean_x;
        sxx += dx * dx;
        sxy += dx * (yo[idx] - mean_y);
    }
    if !(sxx > 0.0) {
        return Err(
            "affine_chart_transition: curve B coordinate has zero spread; slope undefined".into(),
        );
    }
    let slope = sxy / sxx;
    let offset = mean_y - slope * mean_x;
    let mut resid_sq = 0.0_f64;
    for idx in 0..xo.len() {
        let pred = slope * xo[idx] + offset;
        let e = yo[idx] - pred;
        resid_sq += e * e;
    }
    let coord_residual = (resid_sq / m).sqrt();

    Ok(AffineChartTransition {
        slope,
        offset,
        coord_residual,
        geometric_residual,
    })
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
/// `a_n ≥ 0` it fires with across its samples (the gated intensity per row, e.g.
/// `exp(s_k)` times the per-row gate). The verdict is read from the fitted Beta
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
    let mut mean_sq = 0.0_f64;
    for &x in r {
        if !(x > 0.0 && x < 1.0) {
            return None;
        }
        sum_ln += x.ln();
        sum_ln1m += (1.0 - x).ln();
        mean += x;
        mean_sq += x * x;
    }
    let nf = n as f64;
    mean /= nf;
    let var = (mean_sq / nf - mean * mean).max(f64::EPSILON);
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
            |_step, f| f >= base,
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

/// Digamma `ψ(x) = d/dx ln Γ(x)` for `x > 0`: recurrence up to `x ≥ 6` then the
/// standard asymptotic (Bernoulli) series. Hand-derived closed form.
fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0_f64;
    // Recurse up to x ≥ 10 so the truncated Bernoulli tail is ~1e-11 (the x ≥ 6
    // cutoff leaves ~1e-6, too coarse for the Beta Newton and its own test).
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

/// Trigamma `ψ₁(x) = d²/dx² ln Γ(x)` for `x > 0`: recurrence up to `x ≥ 6` then
/// the asymptotic series `1/x + 1/(2x²) + Σ B₂ₖ/x^{2k+1}`.
fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0_f64;
    // Same x ≥ 10 recurrence cutoff as `digamma` for ~1e-11 accuracy.
    while x < 10.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + inv * (1.0 + inv * (0.5 + inv * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0))))
}

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
    use ndarray::{Array3, array};

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
        // ψ(1) = −γ ≈ −0.5772156649; ψ(2) = 1 − γ; ψ₁(1) = π²/6.
        let gamma = 0.577_215_664_901_532_9_f64;
        assert!((digamma(1.0) + gamma).abs() < 1.0e-9);
        assert!((digamma(2.0) - (1.0 - gamma)).abs() < 1.0e-9);
        let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!((trigamma(1.0) - pi2_6).abs() < 1.0e-8);
        // ln Γ(5) = ln 24.
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1.0e-9);
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

    /// A trivial `d = 1` evaluator whose basis is the monomial patch
    /// `Φ(t) = [1, t]` — enough to build straight-line and circle-arc decoders
    /// for the gluing tests without pulling in the production evaluators.
    #[derive(Debug)]
    struct AffineLineEvaluator;

    impl SaeBasisEvaluator for AffineLineEvaluator {
        fn evaluate(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Result<(Array2<f64>, Array3<f64>), String> {
            let n = coords.nrows();
            let mut phi = Array2::<f64>::zeros((n, 2));
            let mut jet = Array3::<f64>::zeros((n, 2, 1));
            for i in 0..n {
                let t = coords[[i, 0]];
                phi[[i, 0]] = 1.0;
                phi[[i, 1]] = t;
                jet[[i, 0, 0]] = 0.0;
                jet[[i, 1, 0]] = 1.0;
            }
            Ok((phi, jet))
        }

        fn second_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array4<f64>, String>> {
            if coords.ncols() != 1 {
                return Some(Err(format!(
                    "AffineLineEvaluator::second_jet_dyn: d = 1 evaluator got {} coords",
                    coords.ncols()
                )));
            }
            None
        }

        fn third_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array5<f64>, String>> {
            if coords.ncols() != 1 {
                return Some(Err(format!(
                    "AffineLineEvaluator::third_jet_dyn: d = 1 evaluator got {} coords",
                    coords.ncols()
                )));
            }
            None
        }
    }
    #[test]
    fn affine_transition_detects_same_line_with_reflection_and_offset() {
        // Curve A: straight segment through the origin, arc-length t ∈ [0, 1].
        // Curve B: the SAME line, reflected and offset — its arc-length coord is
        // t_a = -t_b + 1, i.e. slope -1, offset 1.
        let ev = AffineLineEvaluator;
        // Decoder makes γ(t) = [t·d] with unit-speed d (‖d‖ = 1) so t is arc length.
        let d = array![[0.0_f64, 0.0], [0.6, 0.8]]; // γ(t) = (0,0) + t·(0.6,0.8), speed 1
        let ca = Array1::linspace(0.0, 1.0, 11);
        let pts_a = sample_decoded_curve(&ev, d.view(), ca.view()).unwrap();
        // B samples the same physical points but parameterized as t_b with
        // t_a = -t_b + 1  ⇒  physical point = (1 - t_b)·d. Matched grid so every
        // B point coincides with an A grid point (nearest-match is exact and the
        // reflected transition is recovered to machine precision).
        let cb = Array1::linspace(0.0, 1.0, 11);
        let db = array![[0.6_f64, 0.8], [-0.6, -0.8]]; // γ_b(t_b) = (0.6,0.8) + t_b·(-0.6,-0.8)
        let pts_b = sample_decoded_curve(&ev, db.view(), cb.view()).unwrap();
        let tr = affine_chart_transition(pts_a.view(), ca.view(), pts_b.view(), cb.view(), None)
            .unwrap();
        assert!(
            (tr.slope + 1.0).abs() < 1e-6,
            "slope must be -1, got {}",
            tr.slope
        );
        assert!(
            (tr.offset - 1.0).abs() < 1e-6,
            "offset must be 1, got {}",
            tr.offset
        );
        assert!(
            tr.coord_residual < 1e-6,
            "coord residual {}",
            tr.coord_residual
        );
        assert!(
            tr.geometric_residual < 1e-6,
            "geometric residual {}",
            tr.geometric_residual
        );
        assert!(tr.same_manifold(1.0, 1e-3), "must be flagged same-manifold");
    }

    #[test]
    fn affine_transition_rejects_disjoint_curve() {
        // Curve B is a parallel line displaced far off curve A: the coordinate
        // regression may still fit a slope, but the GEOMETRIC residual is large,
        // so same_manifold must reject.
        let ev = AffineLineEvaluator;
        let da = array![[0.0_f64, 0.0], [1.0, 0.0]]; // A along x-axis
        let db = array![[0.0_f64, 5.0], [1.0, 0.0]]; // B parallel, y = 5 away
        let ca = Array1::linspace(0.0, 1.0, 11);
        let cb = Array1::linspace(0.0, 1.0, 11);
        let pts_a = sample_decoded_curve(&ev, da.view(), ca.view()).unwrap();
        let pts_b = sample_decoded_curve(&ev, db.view(), cb.view()).unwrap();
        let tr = affine_chart_transition(pts_a.view(), ca.view(), pts_b.view(), cb.view(), None)
            .unwrap();
        assert!(
            tr.geometric_residual > 1.0,
            "disjoint curve must have large geometric residual"
        );
        assert!(
            !tr.same_manifold(1.0, 1e-2),
            "disjoint curve must be rejected"
        );
    }
}
