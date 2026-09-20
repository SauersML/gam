//! O(2) classification of circle transports — the Fourier-rigidity classifier.
//!
//! **Theorem (Fourier rigidity).** Any linear map carrying an elliptical atom
//! bijectively onto an elliptical atom induces an angle map `h(θ) = ±θ + φ`:
//! writing `e(h(θ)) = u + M e(θ)` with `M = A′⁺ W A`, the identity `‖e(h)‖ ≡ 1`
//! forces (frequency-2 part) `MᵀM = λI` and (frequency-1 part) `Mᵀu = 0`, hence
//! `u = 0`, `λ = 1`, `M ∈ O(2)`. Rotary "clock arithmetic" is forced by the
//! geometry, not discovered by training.
//!
//! The classifier inverts this: from matched `(θ_in, θ_out)` samples (e.g.
//! [`FittedTransport::eval`] at the fit's own source coordinates), `S₊ = |Σ e^{i(θ_out − θ_in)}|`,
//! `S₋ = |Σ e^{i(θ_out + θ_in)}|`; the larger resultant selects the winding, its
//! argument is `φ`, and `defect = 1 − max(S₊, S₋)/n` is the circular variance
//! about the fitted rigid map — the O(2) departure. A large defect on a pair
//! whose composition defect is small localizes harmonic mixing. All angles in
//! radians.
//!
//! **Class probabilities.** `Shift`, `Reflect` and `Mixing` are three models of
//! the target angle given the source angle: von Mises about `θ_in + φ`, von
//! Mises about `−θ_in + φ`, and uniform. The phase `φ` is uniform on the circle
//! and integrates out exactly, which leaves each rigid model's likelihood
//! relative to `Mixing` as the resultant length's own sampling-density ratio
//! `I₀(nκR)/I₀(κ)ⁿ`. The concentration `κ` is integrated against a uniform prior
//! on the population resultant length `ρ = I₁(κ)/I₀(κ) ∈ [0, 1)`, the rigidity
//! the defect `1 − R` estimates. With the three classes equally probable a
//! priori, the report publishes each class's posterior probability, and the
//! class it names is the most probable one. No separation threshold enters.

use gam_math::special::bessel_i0_centered_terms;
use ndarray::ArrayView1;

use crate::inference::layer_transport::{ChartTopology, FittedTransport};

/// Discrete class of a fitted circle transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircleTransportClass {
    /// `θ ↦ θ + φ` (winding +1).
    Shift,
    /// `θ ↦ −θ + φ` (winding −1).
    Reflect,
    /// No rigid structure: the target angle is uniform given the source angle.
    Mixing,
}

/// Posterior probabilities of the three classes; they sum to one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircleTransportClassProbabilities {
    pub mixing: f64,
    pub shift: f64,
    pub reflect: f64,
}

impl CircleTransportClassProbabilities {
    /// The posterior probability of `class`.
    pub fn of(&self, class: CircleTransportClass) -> f64 {
        match class {
            CircleTransportClass::Mixing => self.mixing,
            CircleTransportClass::Shift => self.shift,
            CircleTransportClass::Reflect => self.reflect,
        }
    }

    /// The most probable class. An exact tie keeps `Mixing`, then `Shift`: the
    /// null of no rigid structure stands unless the data prefer otherwise.
    fn most_probable(&self) -> CircleTransportClass {
        let mut class = CircleTransportClass::Mixing;
        if self.shift > self.of(class) {
            class = CircleTransportClass::Shift;
        }
        if self.reflect > self.of(class) {
            class = CircleTransportClass::Reflect;
        }
        class
    }
}

/// The Fourier-rigidity report for one circle transport.
#[derive(Debug, Clone)]
pub struct CircleTransportReport {
    /// Layers this map connects (for legibility in a ladder report).
    pub layer_from: usize,
    pub layer_to: usize,
    pub n_samples: usize,
    /// `+1` shift, `−1` reflection (winding of the recovered map).
    pub winding: i8,
    /// Phase `φ` in radians, in `(−π, π]`.
    pub phase: f64,
    /// `1 − max(S₊, S₋)/n ∈ [0, 1]`: `0` = exact O(2) element, `≈ 1` = no rigid
    /// structure.
    pub defect: f64,
    /// Standard error of `defect`. For paired samples, from each pair's empirical
    /// influence on the winning resultant, `√Σ(cos(ψ_k − φ) − R)²/n`; for a fitted
    /// map, by the delta method through the fit's coefficient covariance.
    pub defect_se: f64,
    /// `max_k |θ_out,k − (winding·θ_in,k + φ)|`, wrapped to `[0, π]`: the largest
    /// angular gap in radians between the sampled transport and its `O(2)`
    /// element. This is the sup-norm defect that a
    /// [`Contract`](crate::inference::contracts::Contract) chain and
    /// [`loop_holonomy`](crate::inference::contracts::loop_holonomy) add up.
    /// [`Self::defect`] is a circular variance and is not an angle. With
    /// residuals `δ_k`, `1 − R = mean(1 − cos δ_k)` is about `δ²/2`, far below
    /// the gap it would stand in for. The gap is taken over the samples, so it
    /// bounds the map only where it was sampled.
    pub max_angle_gap: f64,
    /// Resultants for both hypotheses (diagnostics).
    pub resultant_shift: f64,
    pub resultant_reflect: f64,
    /// Posterior probability of each class.
    pub class_probabilities: CircleTransportClassProbabilities,
    /// The most probable class under [`Self::class_probabilities`].
    pub class: CircleTransportClass,
}

impl CircleTransportReport {
    /// Phase in degrees, wrapped to `(−180, 180]` — the ladder's report line.
    pub fn phase_degrees(&self) -> f64 {
        self.phase * 180.0 / std::f64::consts::PI
    }

    /// Posterior probability of the reported class.
    pub fn class_probability(&self) -> f64 {
        self.class_probabilities.of(self.class)
    }
}

/// `ln(a + b)` from `ln a` and `ln b`, without forming either exponential.
fn log_add_exp(log_a: f64, log_b: f64) -> f64 {
    if log_a == f64::NEG_INFINITY {
        return log_b;
    }
    if log_b == f64::NEG_INFINITY {
        return log_a;
    }
    let (high, low) = if log_a >= log_b {
        (log_a, log_b)
    } else {
        (log_b, log_a)
    };
    high + (low - high).exp().ln_1p()
}

/// `ln` of one rigid class's marginal likelihood relative to `Mixing`, for `n`
/// pairs whose resultant length under that class is `r`:
/// `ln ∫₀¹ I₀(nκR)/I₀(κ)ⁿ dρ` with `ρ = A(κ) = I₁(κ)/I₀(κ)`.
///
/// The integral is taken over `s = ln κ`, where `dρ = κ·A′(κ) ds` and the
/// integrand is analytic and decays at both ends, by the trapezoidal rule. The
/// first step is `1/√n`: at large `κ` the integrand's width in `s` is `√(2/n)`,
/// and it is wider at small `κ`, so no mode can fall between two nodes. The step
/// is halved until two successive sums differ by no more than the rounding both
/// accumulate over their nodes. That bound grows as the nodes double while the
/// discretization error of an analytic integrand shrinks faster, so the halving
/// ends.
///
/// A sum walks outward from `s = 0` in both directions and stops at a node that
/// is falling and below `ε` of the running total. For `R < 1` it also keeps
/// walking right while `A(κ) ≤ R`: below the likelihood's maximizer `A(κ̂) = R`
/// the integrand can still rise, beyond it and beyond the maximum of `κ·A′(κ)`
/// it only falls, so a sum never stops short of the peak. For `R = 1` the
/// likelihood rises without end, and the walk rests on the integrand itself,
/// which falls for `n ≤ 2`.
///
/// `R = 1` with `n ≥ 3` makes the integral diverge: the pairs fix the map
/// exactly, so `+∞` is returned and the class takes all the probability.
fn log_rigid_bayes_factor(n: usize, r: f64) -> Result<f64, String> {
    let r = r.clamp(0.0, 1.0);
    if r == 1.0 && n >= 3 {
        return Ok(f64::INFINITY);
    }
    let nf = n as f64;
    let log_integrand = |s: f64| -> (f64, bool) {
        let kappa = s.exp();
        let (centered_kappa, ratio, scaled_derivative) = bessel_i0_centered_terms(kappa);
        let (centered_scaled, _, _) = bessel_i0_centered_terms(nf * kappa * r);
        // ln I₀(nκR) − n·ln I₀(κ), with each leading exponential cancelled
        // analytically rather than in floating point.
        let log_likelihood_ratio =
            -nf * kappa * (1.0 - r) + centered_scaled - nf * centered_kappa;
        // κ·A′(κ) = κ(1 − A²) − A. The small-κ form reads A directly; the
        // large-κ form reads κ(1 − A) = −`scaled_derivative`, which carries no
        // cancellation. Each is used where its rounding bound is the smaller
        // one; where both bounds swamp the value, the asymptotic expansion
        // 1/(2κ) + 1/(4κ²) is exact to rounding.
        let small_form = kappa * (1.0 - ratio * ratio) - ratio;
        let small_bound = f64::EPSILON * (kappa + ratio + kappa * ratio * ratio);
        let large_form = -scaled_derivative * (1.0 + ratio) - ratio;
        let large_bound = f64::EPSILON * (scaled_derivative.abs() * (1.0 + ratio) + ratio);
        let (form, bound) = if small_bound / small_form.abs() <= large_bound / large_form.abs() {
            (small_form, small_bound)
        } else {
            (large_form, large_bound)
        };
        let jacobian = if form > bound {
            form
        } else {
            0.5 / kappa + 0.25 / (kappa * kappa)
        };
        (log_likelihood_ratio + jacobian.ln(), ratio <= r)
    };
    let ln_epsilon = f64::EPSILON.ln();
    let trapezoid = |h: f64| -> Result<(f64, usize), String> {
        let (at_zero, _) = log_integrand(0.0);
        let mut log_sum = at_zero;
        let mut nodes = 1usize;
        for direction in [1.0_f64, -1.0] {
            let mut previous = at_zero;
            let mut k = 1.0_f64;
            loop {
                let (value, below_maximizer) = log_integrand(direction * k * h);
                if value.is_nan() {
                    return Err(format!(
                        "circle transport class probability: the rigid-class integrand is NaN at \
                         s = {} (n = {n}, R = {r})",
                        direction * k * h
                    ));
                }
                nodes += 1;
                log_sum = log_add_exp(log_sum, value);
                let falling = value < previous;
                let negligible = value < log_sum + ln_epsilon;
                let peak_ahead = direction > 0.0 && r < 1.0 && below_maximizer;
                if falling && negligible && !peak_ahead {
                    break;
                }
                previous = value;
                k += 1.0;
            }
        }
        Ok((log_sum + h.ln(), nodes))
    };
    let mut h = 1.0 / nf.sqrt();
    let (mut estimate, mut previous_nodes) = trapezoid(h)?;
    loop {
        h *= 0.5;
        let (refined, nodes) = trapezoid(h)?;
        let settled = (refined - estimate).abs() <= (nodes + previous_nodes) as f64 * f64::EPSILON;
        estimate = refined;
        previous_nodes = nodes;
        if settled {
            return Ok(estimate);
        }
    }
}

/// Posterior class probabilities for `n` pairs with resultant lengths
/// `r_shift` and `r_reflect`, with the three classes equally probable a priori.
fn posterior_class_probabilities(
    n: usize,
    r_shift: f64,
    r_reflect: f64,
) -> Result<CircleTransportClassProbabilities, String> {
    let log_shift = log_rigid_bayes_factor(n, r_shift)?;
    let log_reflect = log_rigid_bayes_factor(n, r_reflect)?;
    if log_shift == f64::INFINITY || log_reflect == f64::INFINITY {
        let shift = if log_shift == f64::INFINITY { 1.0 } else { 0.0 };
        let reflect = if log_reflect == f64::INFINITY { 1.0 } else { 0.0 };
        let total = shift + reflect;
        return Ok(CircleTransportClassProbabilities {
            mixing: 0.0,
            shift: shift / total,
            reflect: reflect / total,
        });
    }
    let top = log_shift.max(log_reflect).max(0.0);
    let mixing = (-top).exp();
    let shift = (log_shift - top).exp();
    let reflect = (log_reflect - top).exp();
    let total = mixing + shift + reflect;
    Ok(CircleTransportClassProbabilities {
        mixing: mixing / total,
        shift: shift / total,
        reflect: reflect / total,
    })
}

/// Classify a circle transport from paired angle samples (radians).
///
/// The winding and phase come from the larger resultant. The class is the most
/// probable of `Mixing`, `Shift` and `Reflect` under the posterior the module
/// documentation derives, and every class's probability is published alongside
/// it. A single pair carries no information about the class, and each class
/// then has probability one third.
pub(crate) fn classify_circle_transport(
    theta_in: &[f64],
    theta_out: &[f64],
    layer_from: usize,
    layer_to: usize,
) -> Result<CircleTransportReport, String> {
    if theta_in.len() != theta_out.len() {
        return Err("classify_circle_transport: length mismatch".to_string());
    }
    let n = theta_in.len();
    if n == 0 {
        return Err("classify_circle_transport: need at least one sample, got none".to_string());
    }
    let nf = n as f64;
    let (mut cp, mut sp, mut cm, mut sm) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (&a, &b) in theta_in.iter().zip(theta_out.iter()) {
        let d = b - a;
        cp += d.cos();
        sp += d.sin();
        let s = b + a;
        cm += s.cos();
        sm += s.sin();
    }
    let r_shift = (cp * cp + sp * sp).sqrt() / nf;
    let r_reflect = (cm * cm + sm * sm).sqrt() / nf;
    let (winding, phase, best) = if r_shift >= r_reflect {
        (1i8, sp.atan2(cp), r_shift)
    } else {
        (-1i8, sm.atan2(cm), r_reflect)
    };
    let defect = 1.0 - best;
    let winning_sign = f64::from(winding);
    let influence_sq: f64 = theta_in
        .iter()
        .zip(theta_out.iter())
        .map(|(&a, &b)| {
            let influence = (b - winning_sign * a - phase).cos() - best;
            influence * influence
        })
        .sum();
    let defect_se = influence_sq.sqrt() / nf;
    let max_angle_gap = theta_in
        .iter()
        .zip(theta_out.iter())
        .map(|(&a, &b)| {
            let residual = b - winning_sign * a - phase;
            residual.sin().atan2(residual.cos()).abs()
        })
        .fold(0.0_f64, f64::max);
    let class_probabilities = posterior_class_probabilities(n, r_shift, r_reflect)?;
    Ok(CircleTransportReport {
        layer_from,
        layer_to,
        n_samples: n,
        winding,
        phase,
        defect,
        defect_se,
        max_angle_gap,
        resultant_shift: r_shift,
        resultant_reflect: r_reflect,
        class_probabilities,
        class: class_probabilities.most_probable(),
    })
}

/// Classify a fitted transport between two CIRCLE charts at the fit's own
/// observed source coordinates `coords_from`. These are the rows the map was
/// estimated from, so `n_samples` is the fit's sample size rather than an
/// evaluation density, and `defect_se` comes from the fit's coefficient
/// covariance. The fitted values are smoothed, not independent samples. Returns
/// `None` when either endpoint is not a circle (winding is an O(2) notion; on
/// intervals there is no phase).
pub fn classify_circle_transport_fit(
    fit: &FittedTransport,
    coords_from: ArrayView1<'_, f64>,
    from: ChartTopology,
    to: ChartTopology,
    layer_from: usize,
    layer_to: usize,
) -> Option<CircleTransportReport> {
    if !matches!(from, ChartTopology::Circle) || !matches!(to, ChartTopology::Circle) {
        return None;
    }
    let theta_in: Vec<f64> = coords_from.to_vec();
    let theta_out: Vec<f64> = fit.eval(coords_from).ok()?.to_vec();
    let mut report = classify_circle_transport(&theta_in, &theta_out, layer_from, layer_to).ok()?;
    report.defect_se = fit.circle_resultant_se(coords_from, report.winding).ok()?;
    Some(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::special::bessel_i0_log_and_ratio;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn scrambled_pairs() -> (Vec<f64>, Vec<f64>) {
        let mut s = 9u64;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for _ in 0..512 {
            a.push(std::f64::consts::TAU * lcg(&mut s) - std::f64::consts::PI);
            b.push(std::f64::consts::TAU * lcg(&mut s) - std::f64::consts::PI);
        }
        (a, b)
    }

    /// Independent oracle for one rigid class's Bayes factor: composite Simpson
    /// over `ρ` itself, with `κ = A⁻¹(ρ)` recovered by bisection on the Bessel
    /// ratio. It shares none of production's substitution, Jacobian or node walk.
    fn oracle_bayes_factor(n: usize, r: f64) -> f64 {
        let nf = n as f64;
        let inverse_ratio = |rho: f64| -> f64 {
            let mut hi = 1.0_f64;
            while bessel_i0_log_and_ratio(hi).1 < rho {
                hi *= 2.0;
            }
            let mut lo = 0.0_f64;
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if bessel_i0_log_and_ratio(mid).1 < rho {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };
        let integrand = |rho: f64| -> f64 {
            if rho == 0.0 {
                return 1.0;
            }
            let kappa = inverse_ratio(rho);
            let log_scaled = bessel_i0_log_and_ratio(nf * kappa * r).0;
            let log_single = bessel_i0_log_and_ratio(kappa).0;
            (log_scaled - nf * log_single).exp()
        };
        let intervals = 20_000usize;
        let upper = 0.999_f64;
        let h = upper / intervals as f64;
        let mut sum = integrand(0.0) + integrand(upper);
        for i in 1..intervals {
            let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
            sum += weight * integrand(i as f64 * h);
        }
        sum * h / 3.0
    }

    #[test]
    fn recovers_shift_phase_and_low_defect() {
        let mut s = 3u64;
        let phi = 0.9_f64;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for _ in 0..512 {
            let th = std::f64::consts::TAU * lcg(&mut s) - std::f64::consts::PI;
            a.push(th);
            b.push(th + phi + 0.01 * (lcg(&mut s) - 0.5));
        }
        let r = classify_circle_transport(&a, &b, 17, 18).unwrap();
        assert_eq!(r.class, CircleTransportClass::Shift);
        assert_eq!(r.winding, 1);
        assert!((r.phase - phi).abs() < 0.01);
        assert!(r.defect < 1e-3);
        assert!(r.class_probabilities.shift > r.class_probabilities.mixing);
        assert!(r.class_probabilities.shift > r.class_probabilities.reflect);
    }

    /// The sup-norm gap is the largest residual angle, not the circular variance
    /// `1 − R`, which for residuals `±δ` is `1 − cos δ ≈ δ²/2`.
    #[test]
    fn max_angle_gap_is_the_largest_residual_angle() {
        let phi = 0.4_f64;
        let delta = 0.1_f64;
        let a = [0.0, 1.0, 2.0, 3.0];
        let b: Vec<f64> = a
            .iter()
            .enumerate()
            .map(|(k, &th)| th + phi + if k % 2 == 0 { delta } else { -delta })
            .collect();
        let r = classify_circle_transport(&a, &b, 0, 1).unwrap();
        assert_eq!(r.winding, 1);
        assert!((r.phase - phi).abs() < 1e-12, "phase {}", r.phase);
        assert!((r.max_angle_gap - delta).abs() < 1e-12, "gap {}", r.max_angle_gap);
        assert!((r.defect - (1.0 - delta.cos())).abs() < 1e-12, "defect {}", r.defect);
        assert!(r.defect < r.max_angle_gap / 10.0);
    }

    #[test]
    fn recovers_reflection() {
        let mut s = 5u64;
        let phi = -1.3_f64;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for _ in 0..512 {
            let th = std::f64::consts::TAU * lcg(&mut s) - std::f64::consts::PI;
            a.push(th);
            b.push(-th + phi);
        }
        let r = classify_circle_transport(&a, &b, 0, 1).unwrap();
        assert_eq!(r.class, CircleTransportClass::Reflect);
        assert_eq!(r.winding, -1);
        let mut dphi = r.phase - phi;
        while dphi > std::f64::consts::PI {
            dphi -= std::f64::consts::TAU;
        }
        while dphi < -std::f64::consts::PI {
            dphi += std::f64::consts::TAU;
        }
        assert!(dphi.abs() < 1e-9);
        assert!(r.defect < 1e-12);
        assert!(r.class_probabilities.reflect > r.class_probabilities.shift);
        assert!(r.class_probabilities.reflect > r.class_probabilities.mixing);
    }

    #[test]
    fn scrambled_map_reports_mixing_defect() {
        let (a, b) = scrambled_pairs();
        let r = classify_circle_transport(&a, &b, 0, 2).unwrap();
        assert!(r.defect > 0.8);
        assert_eq!(r.class, CircleTransportClass::Mixing);
    }

    /// #2234 — a scrambled map lands on `Mixing` because its derived posterior
    /// probability is the largest, not because a separation crossed a cut. The
    /// production probabilities are checked against the Simpson-over-`ρ` oracle.
    #[test]
    fn scrambled_map_lands_on_mixing_by_derived_probability() {
        let (a, b) = scrambled_pairs();
        let r = classify_circle_transport(&a, &b, 0, 2).unwrap();
        let p = r.class_probabilities;
        assert!((p.mixing + p.shift + p.reflect - 1.0).abs() < 1e-12);
        let shift_bf = oracle_bayes_factor(512, r.resultant_shift);
        let reflect_bf = oracle_bayes_factor(512, r.resultant_reflect);
        let total = 1.0 + shift_bf + reflect_bf;
        let oracle_mixing = 1.0 / total;
        println!(
            "scrambled: production p = {p:?}; oracle p_mixing = {oracle_mixing}, shift BF = \
             {shift_bf}, reflect BF = {reflect_bf}, R+ = {}, R- = {}",
            r.resultant_shift, r.resultant_reflect
        );
        assert!((p.mixing - oracle_mixing).abs() < 1e-6);
        assert!((p.shift - shift_bf / total).abs() < 1e-6);
        assert!((p.reflect - reflect_bf / total).abs() < 1e-6);
        assert_eq!(r.class, CircleTransportClass::Mixing);
        assert_eq!(r.class_probability(), p.mixing);
    }

    /// A single pair says nothing about the class: every class keeps its prior
    /// probability of one third.
    #[test]
    fn single_pair_leaves_the_prior_class_probabilities() {
        let r = classify_circle_transport(&[0.4], &[-1.1], 0, 1).unwrap();
        let p = r.class_probabilities;
        assert!((p.mixing - 1.0 / 3.0).abs() < 1e-12, "{p:?}");
        assert!((p.shift - 1.0 / 3.0).abs() < 1e-12, "{p:?}");
        assert!((p.reflect - 1.0 / 3.0).abs() < 1e-12, "{p:?}");
    }
}
