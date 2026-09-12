//! Scalar special-function primitives shared across the workspace.
//!
//! These are pure (`std`/`libm`-only) numeric kernels with no upward crate
//! dependencies, so they live in the lowest crate (`gam-math`) and can be
//! consumed by any term/basis/inference code without inducing an SCC edge.

/// Numerically stable `C(n,k) = n! / (k!·(n−k)!)` as `f64`.  Uses the
/// symmetry `C(n,k) = C(n, n−k)` to keep the loop count `min(k, n−k)`
/// and the multiplicative recurrence `C(n,j+1) = C(n,j)·(n−j)/(j+1)`,
/// avoiding the overflow of separate factorial evaluations.  Returns
/// `0.0` for `k > n` and exact integer results within `2^53`.
#[inline]
pub fn binomial_coefficient_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k_eff = k.min(n - k);
    // Carry the recurrence in u128, not f64. At step `j` the running product
    // equals the integer `C(n, j)`. After multiplying by `n-j`, the product
    // is divisible by the next denominator `(j + 1)` (the product of `(j+1)` consecutive
    // integers `(n−j)…(n)` is divisible by `(j+1)!`), so each integer division
    // is exact and no rounding accumulates. The earlier all-`f64` recurrence
    // divided in floating point, where `(n−j)/(j+1)` is generally inexact, and
    // the drift pushed results off the true integer well below `2^53`
    // (e.g. `C(54,24)` came back one short). Converting the exact `u128` at the
    // end is bit-exact for every value at or below `2^53`.
    let mut num: u128 = 1;
    for j in 0..k_eff {
        match num.checked_mul((n - j) as u128) {
            Some(scaled) => num = scaled / (j as u128 + 1),
            None => {
                // The intermediate product overflows u128; the coefficient
                // is already above `2^53`, where the exactness contract no longer applies.
                // Finish the (now necessarily inexact) recurrence in f64.
                let mut out = num as f64;
                for jj in j..k_eff {
                    // Divide the integer factors first: the unscaled product
                    // can overflow even when the coefficient is finite.
                    out *= (n - jj) as f64 / (jj + 1) as f64;
                    if out.is_infinite() {
                        // Coefficients increase up to n/2, so every remaining
                        // step would also be infinite.
                        return out;
                    }
                }
                return out;
            }
        }
    }
    num as f64
}

#[inline]
fn horner_polynomial(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Evaluate `(Σ_k coeffs[k]·x^k) · exp(−x)` without overflow.  For moderate
/// `x ≤ 600` uses Horner + `exp(−x)` directly; for very large `x` rewrites
/// `xᵈ · exp(−x) = exp(d·ln x − x)` and runs Horner in `1/x`, which keeps
/// both the polynomial sum and its multiplier inside double range.  Returns
/// `0.0` for non-finite `x` or empty `coeffs`.
#[inline]
pub fn stable_polynomial_times_exp_neg(x: f64, coeffs: &[f64]) -> f64 {
    if coeffs.is_empty() || !x.is_finite() {
        return 0.0;
    }
    // Below this argument `(-x).exp()` is still well-resolved, so the direct
    // Horner-times-exp form is both accurate and cheapest. Above it the factor
    // underflows toward zero and we switch to the convergent asymptotic tail
    // series to retain the leading significant digits.
    const DIRECT_EXP_SWITCH: f64 = 600.0;
    if x <= DIRECT_EXP_SWITCH {
        return horner_polynomial(x, coeffs) * (-x).exp();
    }

    let inv_x = x.recip();
    let mut tail = 0.0;
    for &c in coeffs {
        tail = tail * inv_x + c;
    }
    let degree = (coeffs.len() - 1) as f64;
    let scale = (degree * x.ln() - x).exp();
    scale * tail
}

/// Argument at which the modified-Bessel evaluation switches from the ascending
/// power series to the large-argument (Hankel) asymptotic expansion.
///
/// Both branches are accurate to a few ulp here, which is what leaves the
/// crossover free of a visible seam. Below it the ascending series is exact up
/// to rounding, because every one of its terms is positive and nothing cancels
/// — with the one exception of the `I0 − I1` difference the branch also carries,
/// whose sign change costs a documented `√x`. Above it the asymptotic
/// expansion's optimal-truncation error is `O(e^{−2x})` — below `5e−18` already
/// at `x = 20`, and shrinking from there.
///
/// That `O(e^{−2x})` is the VALUE channel's floor, and only its. Differentiating
/// an asymptotic series term by term multiplies the `k`-th term by `k`, and both
/// derivative accumulators additionally carry the `x²` factored out of their
/// powers, so their own smallest term is larger by `≈ k·x² ≈ 2x³`. Measured, the
/// curvature channel's optimal truncation is `1.6e−14` absolute at the crossover
/// — against a numerator of `1/8` — so `c''` cannot be better than `≈ 1e−13`
/// relative there no matter how the loop is truncated. It achieves `2.5e−13`,
/// within a factor of two of that floor. Anyone tightening `CURVATURE_TOL`
/// further is chasing a bound the expansion itself does not admit; the fix would
/// have to be a different expansion, not a different stopping rule.
///
/// The former implementation used the single-precision Abramowitz & Stegun
/// 9.8.1–9.8.4 minimax polynomials (crossover 3.75), whose stated accuracy is
/// `|ε| < 2e−7`. That is seven digits short of `f64` and it was the accuracy
/// floor of everything derived from them: `I1/I0` carried `1e−6` relative
/// error, the von-Mises ARD gradient channel `x·(I1/I0 − 1)` carried `4e−6`,
/// and the ARD log-precision curvature carried `6e−3` — a 0.6% error in a
/// quantity an outer Newton step consumes as an exact second derivative, with
/// visible jumps across both the 3.75 and the 30 branch seams.
const BESSEL_ASYMPTOTIC_THRESHOLD: f64 = 20.0;

/// Loop bound for the ascending series. It converges for every argument; below
/// the crossover 36 terms always suffice, so the cap only bounds the loop for a
/// non-finite argument that slipped past the guards.
const BESSEL_SERIES_MAX_TERMS: usize = 128;

/// Loop bound for the asymptotic expansion. The expansion is divergent, so it
/// is truncated at its own smallest term long before this for every argument it
/// is used at; the cap also keeps the coefficient recurrence itself in range.
const BESSEL_ASYMPTOTIC_MAX_TERMS: usize = 64;

/// Ascending power series `(I0(x) − 1, I1(x), I0(x) − I1(x))` for finite `x ≥ 0`.
///
/// `I0(x) = Σ_k (x/2)^{2k}/(k!)²` and `I1(x) = (x/2)·Σ_k (x/2)^{2k}/(k!(k+1)!)`,
/// both carried by the ratio recurrence rather than by separate factorials.
/// Every term is positive, so neither sum cancels and each is correct to within
/// the accumulated rounding of its own additions.
///
/// `I0 − 1` is returned instead of `I0` so the caller can take `ln_1p`: as
/// `x → 0` the wanted `log I0(x) ≈ x²/4` falls below the resolution of
/// `1 + x²/4`, and forming `I0` first would round it away entirely.
///
/// `I0 − I1` is returned as a sum in its OWN right, for the same reason the
/// large-argument branch carries its `N = Σ (b_k − c_k) x^{−(k−1)}`: the caller
/// wants `d1 = x(I1/I0 − 1) = −x·(I0 − I1)/I0`, and `I0 − I1 ≈ I0/(2x)` is
/// smaller than either sum by the whole factor `2x`, so forming it by
/// subtracting the two finished sums throws away `log₂(2x)` bits — five of them
/// by the top of this branch's range. Writing `z = x/2`, the two series share a
/// term ratio, `u_k/t_k = z/(k+1)` with `t_k = z^{2k}/(k!)²`, so the difference
/// is summed directly as
///
/// `I0 − I1 = Σ_k t_k·(1 − z/(k+1)) = Σ_k t_k·(k+1−z)/(k+1)`.
///
/// That sum does change sign, at `k+1 = z`, so it is not cancellation-free the
/// way `I0` and `I1` individually are; but its terms are damped by the very
/// factor that vanishes there, and its condition number `Σ|terms|/|Σ terms|`
/// grows only like `√x` — 6.8 at `x = 20`, against the 40 of the naive
/// difference. Pairing termwise is what turns those five lost bits into three.
fn bessel_ascending_series(ax: f64) -> BesselAscending {
    let half = 0.5 * ax;
    let quarter_square = half * half;
    let mut term_i0 = 1.0_f64;
    let mut i0_minus_one = 0.0_f64;
    let mut term_i1 = 1.0_f64;
    let mut sum_i1 = 1.0_f64;
    // `k = 0`: `t_0 = 1` and the pairing factor is `(0+1−z)/(0+1)`.
    let mut i0_minus_i1 = 1.0 - half;
    for k in 1..=BESSEL_SERIES_MAX_TERMS {
        let kf = k as f64;
        term_i0 *= quarter_square / (kf * kf);
        term_i1 *= quarter_square / (kf * (kf + 1.0));
        i0_minus_one += term_i0;
        sum_i1 += term_i1;
        i0_minus_i1 += term_i0 * (kf + 1.0 - half) / (kf + 1.0);
        // The difference sum sets the stopping rule, because it is the smallest
        // of the three: cutting off at `ε·I0` would leave IT with a relative
        // error of `2x·ε`, which is exactly the error this pairing exists to
        // remove. Past the peak the terms fall factorially, so demanding the
        // extra `log₂(2x)` bits costs only a couple of iterations.
        if term_i0 <= f64::EPSILON * i0_minus_i1.abs()
            && term_i0 <= f64::EPSILON * (1.0 + i0_minus_one)
            && term_i1 <= f64::EPSILON * sum_i1
        {
            break;
        }
    }
    BesselAscending {
        i0_minus_one,
        i1: half * sum_i1,
        i0_minus_i1,
    }
}

/// The three ascending-series sums, each accumulated in its own right.
struct BesselAscending {
    /// `I0(x) − 1`.
    i0_minus_one: f64,
    /// `I1(x)`.
    i1: f64,
    /// `I0(x) − I1(x)`, summed termwise rather than by subtracting the two.
    i0_minus_i1: f64,
}

/// One evaluation of the large-argument (Hankel) asymptotic expansions of `I0`
/// and `I1`, kept in the combinations the callers need so that every leading
/// term cancels ANALYTICALLY here instead of in floating point.
///
/// With `I_ν(x) ~ e^x/√(2πx) · Σ_k (−1)^k a_k(ν) x^{−k}` and
/// `a_k(ν) = ∏_{j=1}^{k} (4ν² − (2j−1)²) / (k!·8^k)`, write `c_k = (−1)^k a_k(0)`
/// and `b_k = (−1)^k a_k(1)`; then `c_0 = b_0 = 1` and both families obey a
/// two-term ratio recurrence, so no coefficient table is needed.
struct BesselAsymptotic {
    /// `S0 = Σ_{k≥0} c_k x^{−k}`, so `I0(x) = e^x S0 / √(2πx)`.
    s0: f64,
    /// `S1 = Σ_{k≥0} b_k x^{−k}`, so `I1/I0 = S1/S0`.
    s1: f64,
    /// `N = Σ_{k≥1} (b_k − c_k) x^{−(k−1)}`, so `d1 = x(I1/I0 − 1) = N/S0`.
    ///
    /// The `k = 0` terms of `S1` and `S0` are both exactly `1`, so they are
    /// dropped symbolically and the difference series starts at its own leading
    /// term `b_1 − c_1 = −1/2` — which is precisely the `d1 → −½` limit. No
    /// near-equal quantities are ever subtracted at run time.
    n: f64,
    /// `x²·S0′ = Σ_{k≥1} (−k) c_k x^{−(k−1)}`.
    s0_scaled_derivative: f64,
    /// `x²·N′ = Σ_{k≥2} −(k−1)(b_k − c_k) x^{−(k−2)}`.
    ///
    /// Both derivative accumulators carry the common `x²` factored out, which
    /// is what keeps `c″(log x) = (x²N′·S0 − N·x²S0′)/(x·S0²)` representable —
    /// and non-zero — out to the largest finite argument, where the unscaled
    /// `x^{−k}` factors would have underflowed to zero.
    n_scaled_derivative: f64,
}

fn bessel_asymptotic_series(ax: f64) -> BesselAsymptotic {
    let inverse = 1.0 / ax;
    let mut c = 1.0_f64;
    let mut b = 1.0_f64;
    let mut acc = BesselAsymptotic {
        s0: 1.0,
        s1: 1.0,
        n: 0.0,
        s0_scaled_derivative: 0.0,
        n_scaled_derivative: 0.0,
    };
    // `x^{−(k−2)}` and `x^{−(k−1)}` at the current `k`, carried as their own
    // running products so that a power which has overflowed is never multiplied
    // by one which has underflowed.
    let mut power_two_back = ax;
    let mut power_one_back = 1.0_f64;
    let mut smallest = f64::INFINITY;
    for k in 1..=BESSEL_ASYMPTOTIC_MAX_TERMS {
        let kf = k as f64;
        let odd = 2.0 * kf - 1.0;
        c *= odd * odd / (8.0 * kf);
        b *= (odd * odd - 4.0) / (8.0 * kf);
        let power = power_one_back * inverse;
        let term_c = c * power;
        // The expansion is asymptotic, not convergent: past its smallest term
        // every further term makes the answer worse. Stopping there is what
        // realises the `O(e^{−2x})` optimal-truncation error. The negated
        // comparison also stops on a NaN argument.
        if !(term_c.abs() <= smallest) {
            break;
        }
        smallest = term_c.abs();
        let difference = b - c;
        let curvature_term = (kf - 1.0) * difference * power_two_back;
        acc.s0 += term_c;
        acc.s1 += b * power;
        acc.n += difference * power_one_back;
        acc.s0_scaled_derivative -= kf * c * power_one_back;
        if k >= 2 {
            acc.n_scaled_derivative -= curvature_term;
        }
        // `n_scaled_derivative` carries the largest power of the four sums, so
        // once ITS increment is negligible every other one is too.
        let scale = acc.n_scaled_derivative.abs().max(acc.n.abs());
        if k >= 3 && curvature_term.abs() <= f64::EPSILON * scale {
            break;
        }
        power_two_back = power_one_back;
        power_one_back = power;
    }
    acc
}

/// Overflow-free centered Bessel value, ratio, and log-scale derivative.
///
/// For `x = |eta|`, returns
/// `(log I0(x) - x, I1(x) / I0(x), x d/dx[log I0(x) - x])`. The third term is
/// the stable form of `x·(I1/I0 - 1)`: it approaches `-½` instead of becoming
/// `x·0` after the ordinary ratio rounds to one. Centering the logarithm by its
/// leading `x` term likewise prevents catastrophic cancellation.
pub fn bessel_i0_centered_terms(eta: f64) -> (f64, f64, f64) {
    let ax = eta.abs();
    if ax.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if ax.is_infinite() {
        // `−½ log(2πx) → −∞`, `I1/I0 → 1`, and the centered log-derivative
        // holds its exact `−½` limit.
        return (f64::NEG_INFINITY, 1.0, -0.5);
    }
    if ax < BESSEL_ASYMPTOTIC_THRESHOLD {
        let series = bessel_ascending_series(ax);
        let i0 = 1.0 + series.i0_minus_one;
        // `d1 = x(I1/I0 − 1) = −x·(I0 − I1)/I0`, taken from the difference the
        // series accumulated itself. Forming `ax * (ratio - 1.0)` here instead
        // would reintroduce the very `2x` cancellation the large-argument branch
        // is careful to avoid, and would leave a visible accuracy seam at the
        // crossover: `1 − ratio` is `0.025` at `x = 20`, so a correctly rounded
        // `ratio` still pins `d1` no tighter than `4e−15`.
        return (
            series.i0_minus_one.ln_1p() - ax,
            series.i1 / i0,
            -ax * (series.i0_minus_i1 / i0),
        );
    }
    let series = bessel_asymptotic_series(ax);
    (
        // `log I0(x) − x = −½ log(2πx) + log S0`. The `2πx` product is split so
        // it cannot overflow just short of the largest finite argument.
        series.s0.ln() - 0.5 * (std::f64::consts::TAU.ln() + ax.ln()),
        series.s1 / series.s0,
        series.n / series.s0,
    )
}

/// Stable centered Bessel terms when only `log(|eta|)` is representable.
///
/// For a finite representable `|eta|`, this is exactly
/// [`bessel_i0_centered_terms`]. Beyond the float range, inverse-`eta`
/// corrections are themselves below float resolution, so the limiting terms
/// `log I0(eta)-eta = -½ log(2 pi eta)` and
/// `eta d/deta[log I0(eta)-eta] = -½` are the correctly rounded result.
pub fn bessel_i0_centered_terms_from_log_abs(log_abs_eta: f64) -> (f64, f64, f64) {
    if log_abs_eta.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if log_abs_eta == f64::NEG_INFINITY {
        return (0.0, 0.0, 0.0);
    }
    if log_abs_eta <= f64::MAX.ln() {
        return bessel_i0_centered_terms(log_abs_eta.exp());
    }
    (-0.5 * (std::f64::consts::TAU.ln() + log_abs_eta), 1.0, -0.5)
}

/// Second log-scale derivative of the centered Bessel primitive:
/// `d²/d(log η)²[log I0(η) − η]`, i.e. the derivative of the third term `d1`
/// returned by [`bessel_i0_centered_terms`] (`d1 = η d/dη[log I0(η) − η]`).
///
/// Writing `s = log η`, `r = I1(η)/I0(η)`, and `c(s) = log I0(η) − η`, the first
/// log-derivative is `c'(s) = d1 = η(r − 1)`. Differentiating again and using
/// the modified-Bessel ratio ODE `r'(η) = 1 − r/η − r²` gives the exact closed
/// form `c''(s) = −η + η²(1 − r²)`. That direct form is numerically unusable
/// for moderate/large `η`: its two terms each grow like `η` and cancel to
/// `O(1/η)`, so the ratio's `~ε_poly` approximation error is amplified by `η²`.
/// The algebraically identical rearrangement in terms of the STABLE third term
///
/// `c''(s) = −η(2·d1 + 1) − d1²`
///
/// cancels safely instead: `d1 → −½` with `2·d1 + 1 → 0`, so the amplification
/// drops to `η·δd1`. It is also, by construction, the exact derivative of the
/// SAME `d1` the outer gradient's periodic-ARD normalizer channel reports, so
/// gradient and Hessian differentiate one quantity. Beyond the float range
/// `c'(s) → −½` (constant) so `c''(s) → 0`; likewise `η → 0` gives `c''(s) → 0`.
/// The von-Mises ARD log-precision normalizer `n[−η + log I0(η)]` therefore has
/// `∂²/∂(log α)² = n · c''(log η)` up to the affine `log η = log α + const` shift.
///
/// Three regimes, each chosen so that nothing cancels in it:
///
/// * `η ≥ 20`: read `c''(s) = η(N′S0 − N S0′)/S0²` straight off the asymptotic
///   expansion (`BesselAsymptotic`). Its two products are `3/16` and `1/16` at
///   leading order — a benign ratio, where the `−η(2d1+1)` and `d1²` of the
///   closed form both approach `¼` and cancel down to `1/(8η)`.
/// * `1 ≤ η < 20`: the closed form, with that `¼` removed symbolically. Writing
///   `q = d1 + ½` (which decays like `−1/(8η)`), `d1² = q² − q + ¼` and
///   `−2ηq − ¼ = −2η(q + 1/(8η))`, leaving `c''(s) = −2η(q + 1/(8η)) + q − q²`
///   with no constant term for the answer to be dwarfed by.
///
///   Removing the constant is not the same as removing the amplification, and
///   this branch keeps the latter. `q` is only ever known to `d1`'s own absolute
///   error, so `δc'' ≈ 2η·δd1`, while the answer it sits on is `|c''| ≈ 1/(8η)`
///   — a RELATIVE amplification of `16η²·δd1` that grows quadratically across
///   the branch. `d1` in turn carries `|d1|·κ(η)·ε` from the `I0 − I1` sum whose
///   condition number `κ = Σ|terms|/|Σ terms|` grows like `√η` (3.1 at `η = 5`,
///   6.8 at `η = 20`), so the floor of this representation is `≈ 8η²·κ(η)·ε`:
///
///   ```text
///     η          5        10        15        19       20⁻
///     floor   3.9e−14   3.3e−13   1.0e−12   2.0e−12   2.2e−12
///     worst   1.9e−13   1.2e−12   3.3e−12   7.4e−12   8.9e−12
///   ```
///
///   Measured against an 80-digit reference over 24000 points, the branch holds
///   a uniform 3−5x of that floor across its whole range, peaking at `8.9e−12`
///   just under the crossover; the asymptotic branch resumes at `1.6e−13` on
///   the far side.
///   That step is a property of the two representations, not a mis-placed
///   threshold: the asymptotic expansion's own truncation error at this channel
///   is `1.0e−12` at `η = 19` and `6.6e−12` at `η = 18`, so the two curves
///   cross within a few tenths of where the code already switches and no
///   choice of threshold caps the band below `≈ 4e−12`.
///
///   Nor is it reachable by a better formula in `f64`. The cancellation is
///   intrinsic to the ascending representation rather than to how it is
///   collected: accumulating the whole numerator `I0 − 2η(I0 − I1)` termwise —
///   the same pairing trick that buys the `I0 − I1` sum its `√η` — gives terms
///   `t_k·[(k+1)(1−2η) + η²/... ]` whose `Σ|terms|/|Σ terms|` is again `8η²`,
///   because the leading `¼` cancels BETWEEN terms of one series and not within
///   any term. Closing the band needs `d1` carried wider than `f64`, and its one
///   consumer — the von-Mises ARD log-precision Hessian entry, where the
///   pre-2025 A&S polynomials delivered `6e−3` — is nine orders clear of caring.
/// * `η < 1`: `c''(s) = −η + η²(1 − r²) = −η·[1 + d1(1 + r)]`, whose bracket
///   tends to `1`. The rearrangement above would instead subtract two numbers
///   that both tend to `¼` while the answer itself tends to `−η`.
pub fn bessel_i0_centered_second_log_derivative_from_log_abs(log_abs_eta: f64) -> f64 {
    if log_abs_eta.is_nan() {
        return f64::NAN;
    }
    if log_abs_eta == f64::NEG_INFINITY {
        return 0.0;
    }
    if log_abs_eta > f64::MAX.ln() {
        return 0.0;
    }
    let eta = log_abs_eta.exp();
    if eta >= BESSEL_ASYMPTOTIC_THRESHOLD {
        let series = bessel_asymptotic_series(eta);
        return (series.n_scaled_derivative * series.s0 - series.n * series.s0_scaled_derivative)
            / (eta * series.s0 * series.s0);
    }
    let (_centered, ratio, d1) = bessel_i0_centered_terms(eta);
    if eta < 1.0 {
        return -eta * (1.0 + d1 * (1.0 + ratio));
    }
    let q = d1 + 0.5;
    -2.0 * eta * (q + 0.125 / eta) + q - q * q
}

/// Overflow-free `(log I0(eta) - |eta|, I1(|eta|) / I0(|eta|))`.
///
/// Centering the logarithm by its leading `|eta|` term is essential whenever a
/// likelihood cancels the Bessel growth against an equally large quadratic,
/// as in a Gaussian-blurred circle. The large-argument branch never forms
/// `exp(|eta|)`, and therefore remains finite beyond the ordinary exponential
/// overflow threshold and up to the largest finite `f64`.
pub fn bessel_i0_log_minus_abs_and_ratio(eta: f64) -> (f64, f64) {
    let (centered_log_i0, ratio, _) = bessel_i0_centered_terms(eta);
    (centered_log_i0, ratio)
}

/// Overflow-free `(log I0(eta), I1(|eta|) / I0(|eta|))`.
///
/// Consumers whose formulas cancel the leading `|eta|` term should use
/// [`bessel_i0_log_minus_abs_and_ratio`] directly, rather than forming that
/// cancellation after this function returns.
pub fn bessel_i0_log_and_ratio(eta: f64) -> (f64, f64) {
    let (centered_log_i0, ratio) = bessel_i0_log_minus_abs_and_ratio(eta);
    (eta.abs() + centered_log_i0, ratio)
}

/// Argument above which the polygamma family switches from its downward
/// recurrence to the Bernoulli asymptotic series.
///
/// The series is divergent, but its terms only start growing near `x ≈ πk`, so
/// at this threshold each of the four functions below is already limited by
/// `f64` rounding rather than by truncation — see the per-function notes for
/// the first omitted term. The recurrence that walks a small argument up to
/// here costs one reciprocal and one add per unit step.
const POLYGAMMA_ASYMPTOTIC_THRESHOLD: f64 = 20.0;

/// Digamma `ψ(x) = d/dx ln Γ(x)`, for `x > 0`; `NaN` otherwise.
///
/// Recurrence `ψ(x) = ψ(x+1) − 1/x` up to the threshold, then
/// `ψ(x) ~ ln x − 1/(2x) − Σ_{k≥1} B_{2k}/(2k·x^{2k})`. Carried through
/// `B₁₂`, so the first omitted term is `1/(12x¹⁴)` — `5e−20` at `x = 20`,
/// against `ψ(20) ≈ 2.97`.
pub fn digamma(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut recurrence = 0.0_f64;
    while x < POLYGAMMA_ASYMPTOTIC_THRESHOLD {
        recurrence -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // −1/12 + w/120 − w²/252 + w³/240 − w⁴/132 + 691w⁵/32760, w = 1/x².
    let series = horner_polynomial(
        inv2,
        &[
            -1.0 / 12.0,
            1.0 / 120.0,
            -1.0 / 252.0,
            1.0 / 240.0,
            -1.0 / 132.0,
            691.0 / 32_760.0,
        ],
    );
    recurrence + x.ln() - 0.5 * inv + inv2 * series
}

/// Trigamma `ψ₁(x) = d²/dx² ln Γ(x)`, for `x > 0`; `NaN` otherwise.
///
/// Recurrence `ψ₁(x) = ψ₁(x+1) + 1/x²`, then
/// `ψ₁(x) ~ 1/x + 1/(2x²) + Σ_{k≥1} B_{2k}/x^{2k+1}`. Carried through `B₁₂`,
/// first omitted `7/(6x¹⁵)` — `4e−20` at `x = 20` against `ψ₁(20) ≈ 0.051`.
pub fn trigamma(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut recurrence = 0.0_f64;
    while x < POLYGAMMA_ASYMPTOTIC_THRESHOLD {
        recurrence += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // 1/6 − w/30 + w²/42 − w³/30 + 5w⁴/66 − 691w⁵/2730, w = 1/x².
    let series = horner_polynomial(
        inv2,
        &[
            1.0 / 6.0,
            -1.0 / 30.0,
            1.0 / 42.0,
            -1.0 / 30.0,
            5.0 / 66.0,
            -691.0 / 2_730.0,
        ],
    );
    recurrence + inv + 0.5 * inv2 + inv2 * inv * series
}

/// Tetragamma `ψ₂(x) = d³/dx³ ln Γ(x)`, for `x > 0`; `NaN` otherwise.
///
/// Recurrence `ψ₂(x) = ψ₂(x+1) − 2/x³`, then the `n = 2` case of
/// `ψ⁽ⁿ⁾(x) ~ (−1)^{n−1}[(n−1)!/xⁿ + n!/(2x^{n+1})
/// + Σ_k B_{2k}(2k+n−1)!/((2k)!·x^{2k+n})]`. Carried through `B₁₂`, first
/// omitted `17.5/x¹⁶` — `3e−20` at `x = 20` against `|ψ₂(20)| ≈ 2.6e−3`.
pub fn tetragamma(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut recurrence = 0.0_f64;
    while x < POLYGAMMA_ASYMPTOTIC_THRESHOLD {
        recurrence -= 2.0 / (x * x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // Coefficients B_{2k}(2k+1): 1/2, −1/6, 1/6, −3/10, 5/6, −691/210.
    let series = horner_polynomial(
        inv2,
        &[
            0.5,
            -1.0 / 6.0,
            1.0 / 6.0,
            -3.0 / 10.0,
            5.0 / 6.0,
            -691.0 / 210.0,
        ],
    );
    recurrence - (inv2 + inv2 * inv + inv2 * inv2 * series)
}

/// Pentagamma `ψ₃(x) = d⁴/dx⁴ ln Γ(x)`, for `x > 0`; `NaN` otherwise.
///
/// Recurrence `ψ₃(x) = ψ₃(x+1) + 6/x⁴`, then the `n = 3` case of the same
/// expansion. Carried through `B₁₂`, first omitted `280/x¹⁷` — `2e−20` at
/// `x = 20` against `ψ₃(20) ≈ 2.6e−4`.
pub fn pentagamma(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut recurrence = 0.0_f64;
    while x < POLYGAMMA_ASYMPTOTIC_THRESHOLD {
        recurrence += 6.0 / (x * x * x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // Coefficients B_{2k}(2k+1)(2k+2): 2, −1, 4/3, −3, 10, −691·182/2730.
    let series = horner_polynomial(
        inv2,
        &[2.0, -1.0, 4.0 / 3.0, -3.0, 10.0, -691.0 * 182.0 / 2_730.0],
    );
    recurrence + 2.0 * inv2 * inv + 3.0 * inv2 * inv2 + inv2 * inv2 * inv * series
}

/// Gauss-Legendre nodes and weights on `[-1, 1]` for `n` points, computed via
/// Newton iteration on the Legendre-polynomial roots (Bonnet's three-term
/// recurrence, cosine initial guess). Returns `(nodes, weights)` with nodes
/// ascending; for odd `n` the central node is exactly `0.0`.
///
/// Canonical home for the routine previously triplicated in
/// `gam-terms/basis/closed_form_penalty.rs`, `gam-model-kernels/
/// cubic_cell_kernel.rs`, and `gam-models/survival/base.rs`; this copy keeps
/// the tightest of their Newton settings (200-iteration cap, `1e-15`
/// convergence).
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut tmp: Vec<(f64, f64)> = Vec::with_capacity(n);
    let half = n.div_ceil(2);
    for i in 0..half {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        // `(P_n(z), P_n'(z))` by Bonnet's recurrence, with the derivative taken
        // from `P_n'(z) = n(z·P_n − P_{n−1})/(z² − 1)`.
        let legendre_value_and_slope = |z: f64| {
            let mut p1 = 1.0_f64;
            let mut p2 = 0.0_f64;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * z * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            (p1, n as f64 * (z * p1 - p2) / (z * z - 1.0))
        };
        for _ in 0..200 {
            let (p1, pp) = legendre_value_and_slope(z);
            let z_prev = z;
            z = z_prev - p1 / pp;
            if (z - z_prev).abs() < 1e-15 {
                break;
            }
        }
        // Re-evaluate `P_n'` AT the node being returned. The loop leaves `pp`
        // one Newton step stale — it was formed at `z_prev`, and `z` has since
        // moved by up to the `1e-15` break threshold — while the weight below
        // reads the fresh `z` in its `(1 − z²)`. Mixing the two is not a wash:
        // Legendre's equation gives `P_n'' = 2z·P_n'/(1 − z²)` at a root, so a
        // node offset `δ` lands in the weight amplified by `2·2z/(1 − z²)`,
        // which runs to ~5900 for the outermost node at `n = 128`. One more
        // Bonnet pass costs `O(n)` against the `O(n·iterations)` already spent
        // per node and removes it: worst weight error falls 8.3e-14 -> 1.9e-15
        // at `n = 16` and 8.2e-14 -> 5.6e-15 at `n = 32`.
        //
        // It does NOT move the WORST case past `n ≈ 64`, where the same
        // amplification acts instead on the node's own irreducible ~1 ulp: the
        // outer nodes crowd toward ±1, `1 − z²` falls to `3e-4`, and the
        // weights there hold ~3e-13 however `pp` is evaluated. (The mean still
        // improves — 1.2e-14 -> 8.2e-15 at `n = 128` — with a handful of outer
        // weights moving an ulp either way, which is the level the node residual
        // already sets.) Escaping that bound needs a weight formula that does
        // not route through `P_n'(z)` at all, not a better Newton loop.
        //
        // In particular it is NOT reachable by correcting for the node offset,
        // which is the obvious thing to try next and was measured. Substituting
        // Legendre's equation at a root collapses the weight's two sensitivities
        // to a single `d(log w)/dz = −2z/(1 − z²)`, and the offset to the true
        // root is one Newton step, `δ = −P_n(z)/P_n'(z)` — whose `P_n(z)` the
        // Bonnet pass on the next line already computes and discards. So the
        // first-order correction `w·(1 + 2z·(P_n/P_n')/(1 − z²))` is free, and
        // it is exact: fed a `δ` from an 80-digit reference it drives the weight
        // error to 1e-25 at every `n` tried. The derivation is not the problem.
        //
        // What kills it is `δ`'s own resolution. Bonnet evaluates a `P_n` that
        // is sitting AT its root to an absolute `≈ n·ε`, so the correction
        // carries noise `2z·(n·ε/P_n')/(1 − z²)` — and that noise is within an
        // order of the term it is removing across the whole range (outermost
        // node: `5.6e−16` term vs `2.8e−15` noise at `n = 16`, `1.5e−13` vs
        // `4.6e−14` at `n = 256`). The net over `n ∈ {16..256}` is a coin flip
        // decided by how close each node happened to land — 3.8x better at
        // `n = 128`, 13x worse at `n = 200` — so the correction is not applied.
        // Making it pay needs `P_n` evaluated wider than `f64`, at which point
        // the node itself may as well be.
        let (_, pp) = legendre_value_and_slope(z);
        let w = 2.0 / ((1.0 - z * z) * pp * pp);
        // For odd n the central node is at z = 0; record once.
        if !n.is_multiple_of(2) && i == half - 1 {
            tmp.push((0.0, w));
        } else {
            tmp.push((-z.abs(), w));
            tmp.push((z.abs(), w));
        }
    }
    tmp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for (z, w) in tmp.into_iter().take(n) {
        nodes.push(z);
        weights.push(w);
    }
    (nodes, weights)
}

// ---------------------------------------------------------------------------
// Exponential-family scalar kernels.
//
// The stable forms of the handful of scalar maps every deviance / KL / working-
// weight evaluation is built from. Each is written so that its result is
// accurate in the regime where the naive formula cancels (small arguments,
// probabilities near the boundary) and so that no branch forms an intermediate
// that overflows before the final, representable value. They are consumed by
// the CPU PIRLS deviance path and by the host reference of the device PIRLS
// row kernels; the two used to carry byte-identical private copies (#2470).
// ---------------------------------------------------------------------------

/// `softplus(x) = ln(1 + e^x)`, evaluated as `max(x, 0) + ln(1 + e^{-|x|})`
/// so that neither tail overflows and the small-`x` result keeps full
/// relative accuracy.
#[inline]
pub fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// The logistic function `σ(x) = 1 / (1 + e^{-x})`, oriented so the
/// exponential is always of a non-positive argument.
#[inline]
pub fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `x·ln(y)` with the convention `0·ln(0) = 0` used by every deviance kernel.
#[inline]
pub fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y.ln() }
}

/// `ln(e^a + e^b)` without forming either exponential at full scale; returns
/// `-∞` when both arguments are `-∞`.
#[inline]
pub fn logaddexp(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let hi = a.max(b);
    let lo = a.min(b);
    if hi.is_infinite() {
        hi
    } else {
        hi + (lo - hi).exp().ln_1p()
    }
}

/// `e^x − 1 − x`, the second-order remainder of the exponential. For
/// `|x| ≤ 1/2` the Taylor tail is summed directly so the result does not
/// cancel against `x`; beyond that `exp_m1` is accurate on its own.
#[inline]
pub fn expm1_minus_x(x: f64) -> f64 {
    if !x.is_finite() {
        return x.abs();
    }
    if x.abs() > 0.5 {
        return x.exp_m1() - x;
    }
    let mut term = 0.5 * x * x;
    let mut sum = term;
    let mut k = 2.0;
    loop {
        k += 1.0;
        term *= x / k;
        let next = sum + term;
        if next == sum {
            return next;
        }
        sum = next;
    }
}

/// `ln(1 + x) − x`, the second-order remainder of the logarithm, with the
/// same small-argument series treatment as [`expm1_minus_x`].
#[inline]
pub fn log1p_minus_x(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return -x;
    }
    if x.abs() > 0.5 {
        return x.ln_1p() - x;
    }
    let mut power = x * x;
    let mut sign = -1.0;
    let mut k = 2.0;
    let mut sum = sign * power / k;
    loop {
        power *= x;
        sign = -sign;
        k += 1.0;
        let next = sum + sign * power / k;
        if next == sum {
            return next;
        }
        sum = next;
    }
}

/// The relative exponential `exprel(x) = (e^x − 1) / x`, equal to `1` at
/// `x = 0` and summed as a series for `|x| ≤ 1/2`.
#[inline]
pub fn exprel(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x > 700.0 {
        // Divide before the second multiplication: exp(x) can overflow
        // while exp(x)/x is still representable. The omitted 1/x is far
        // below one ulp in this branch.
        let half_exp = (0.5 * x).exp();
        return (half_exp / x) * half_exp;
    }
    if x.abs() > 0.5 {
        return x.exp_m1() / x;
    }
    1.0 + exprel_minus_one_small(x)
}

/// The nonconstant Taylor terms of exprel for finite `|x| <= 1/2`.
/// Keeping the leading one out of the sum lets log_exprel use log1p
/// without erasing its first-order term near zero.
#[inline]
fn exprel_minus_one_small(x: f64) -> f64 {
    let mut term = 0.5 * x;
    let mut sum = term;
    let mut k = 2.0;
    loop {
        k += 1.0;
        term *= x / k;
        let next = sum + term;
        if next == sum {
            return next;
        }
        sum = next;
    }
}

/// `ln(exprel(x))`, with the large-`|x|` branches written so that no
/// exponential of a positive argument is ever formed.
#[inline]
pub fn log_exprel(x: f64) -> f64 {
    if !x.is_finite() {
        x
    } else if x.abs() <= 0.5 {
        exprel_minus_one_small(x).ln_1p()
    } else if x > 0.0 {
        x + (-(-x).exp()).ln_1p() - x.ln()
    } else {
        (-x.exp()).ln_1p() - (-x).ln()
    }
}

/// `ln|1 − e^x|` for `x ≠ 0`, routed through
/// [`crate::probability::log1mexp_positive`] on both sides of the origin.
#[inline]
pub fn log_abs_one_minus_exp(x: f64) -> f64 {
    if x > 0.0 {
        x + crate::probability::log1mexp_positive(x)
    } else {
        crate::probability::log1mexp_positive(-x)
    }
}

/// The Bregman divergence `bd0(x, m) = x·ln(x/m) + m − x` of the Poisson
/// deviance (Loader's `bd0`), summed as a series in `(x − m)/(x + m)` when the
/// two arguments are within 20% of each other so the two ~equal logarithms
/// never cancel. `bd0(0, m) = m` exactly.
#[inline]
pub fn bd0(x: f64, m: f64) -> f64 {
    if x == 0.0 {
        return m;
    }
    if x == m {
        return 0.0;
    }
    let hi = x.max(m);
    let lo = x.min(m);
    let relative_gap = (x - m).abs() / hi;
    if relative_gap < 0.2 {
        let v = ((x - m) / hi) / (1.0 + lo / hi);
        let mut sum = (x - m) * v;
        let mut ej = 2.0 * (x * v);
        let v2 = v * v;
        let mut denominator = 3.0;
        loop {
            ej *= v2;
            let next = sum + ej / denominator;
            if next == sum {
                return next;
            }
            sum = next;
            denominator += 2.0;
        }
    }
    x * (x.ln() - m.ln()) + (m - x)
}

/// Bernoulli KL divergence in natural coordinates, `KL(σ(a) ‖ σ(b))`,
/// without subtracting an entropy from a cross entropy. For `|b − a| ≤ 1/2`
/// only second-order remainders are evaluated; the tail branches orient the
/// event so the reference probability never rounds to one.
#[inline]
pub fn bernoulli_kl_from_logits(a: f64, b: f64) -> f64 {
    if a == b {
        return 0.0;
    }
    let h = b - a;
    if h.abs() <= 0.5 {
        // Orient toward the rarer reference event.  Without this swap a large
        // positive `a` rounds `σ(a)` to one and erases a representable
        // right-tail KL channel.
        let (p, local_h) = if a <= 0.0 {
            (logistic(a), h)
        } else {
            (logistic(-a), -h)
        };
        let em1 = local_h.exp_m1();
        let x = p * em1;
        return log1p_minus_x(x) + p * expm1_minus_x(local_h);
    }
    if a <= 0.0 {
        let p = logistic(a);
        p * (a - b) + softplus(b) - softplus(a)
    } else {
        let q = logistic(-a);
        q * (b - a) + softplus(-b) - softplus(-a)
    }
}

// ---------------------------------------------------------------------------
// Binary-exponent arithmetic.
// ---------------------------------------------------------------------------

/// Exact power-of-two decomposition `x = mantissa · 2^exponent` for a positive
/// finite `f64`, including subnormals. The mantissa lies in `[1, 2)`.
#[inline]
pub fn positive_frexp(x: f64) -> (f64, i32) {
    assert!(x.is_finite() && x > 0.0);
    let bits = x.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if raw_exp != 0 {
        let mantissa = f64::from_bits((1023_u64 << 52) | fraction);
        (mantissa, raw_exp - 1023)
    } else {
        let leading = 63_i32 - fraction.leading_zeros() as i32;
        let shift = 52_i32 - leading;
        let normalized = fraction << shift;
        let mantissa = f64::from_bits((1023_u64 << 52) | (normalized & ((1_u64 << 52) - 1)));
        (mantissa, -1022 - shift)
    }
}

/// `mantissa · 2^exponent` for a positive mantissa, renormalising the mantissa
/// into `[1, 2)` first so the result overflows or underflows only when the
/// final `f64` itself is unrepresentable. Subnormal results are formed by
/// scaling in units of the least positive subnormal, so IEEE rounds the final
/// value once instead of underflowing an intermediate.
#[inline]
pub fn scale_normalized_power_of_two(mut mantissa: f64, mut exponent: i32) -> f64 {
    while mantissa >= 2.0 {
        mantissa *= 0.5;
        exponent += 1;
    }
    while mantissa < 1.0 {
        mantissa *= 2.0;
        exponent -= 1;
    }
    if exponent > 1023 {
        return f64::INFINITY;
    }
    if exponent >= -1022 {
        let power = f64::from_bits(((exponent + 1023) as u64) << 52);
        return mantissa * power;
    }
    if exponent < -1075 {
        return 0.0;
    }
    let units = mantissa * 2.0_f64.powi(exponent + 1074);
    units * f64::from_bits(1)
}

/// `a·b·c/d` for positive finite inputs, carrying the binary exponent
/// separately so an intermediate overflow or underflow cannot change a
/// representable final result.
#[inline]
pub fn scaled_positive_product_quotient(a: f64, b: f64, c: f64, d: f64) -> f64 {
    assert!(a.is_finite() && a > 0.0);
    assert!(b.is_finite() && b > 0.0);
    assert!(c.is_finite() && c > 0.0);
    assert!(d.is_finite() && d > 0.0);
    let (ma, ea) = positive_frexp(a);
    let (mb, eb) = positive_frexp(b);
    let (mc, ec) = positive_frexp(c);
    let (md, ed) = positive_frexp(d);
    scale_normalized_power_of_two((ma * mb) * (mc / md), ea + eb + ec - ed)
}

#[cfg(test)]
mod exponential_family_kernel_tests {
    use super::*;

    #[test]
    fn log_exprel_preserves_small_arguments_and_reflection() {
        for x in [1e-8_f64, 1e-16, 1e-100, 1e-300] {
            for sign in [-1.0, 1.0] {
                let argument = sign * x;
                let expected = argument * 0.5 + argument * argument / 24.0;
                let got = log_exprel(argument);
                assert!((got / expected - 1.0).abs() < 4.0 * f64::EPSILON,
                    "x={argument}: got {got}, expected {expected}");
            }
            let difference = log_exprel(x) - log_exprel(-x);
            assert!((difference / x - 1.0).abs() < 4.0 * f64::EPSILON);
        }
    }

    #[test]
    fn exprel_remains_finite_after_the_unscaled_exponential_overflows() {
        for x in [710.0_f64, 712.0, 716.0] {
            let got = exprel(x);
            assert!(got.is_finite() && got > 0.0, "x={x}: {got}");
            assert!((got.ln() - log_exprel(x)).abs() < 2e-13);
            // exprel(x) = exp(x/2) * (exprel(x/2) + exprel(-x/2)) / 2.
            let half = x * 0.5;
            let expected = half.exp() * (0.5 * exprel(half) + 0.5 * exprel(-half));
            assert!((got / expected - 1.0).abs() < 4.0 * f64::EPSILON);
        }
        assert_eq!(exprel(717.0), f64::INFINITY);
    }

    #[test]
    fn exponential_family_kernels_have_correct_infinite_limits() {
        assert_eq!(expm1_minus_x(f64::INFINITY), f64::INFINITY);
        assert_eq!(expm1_minus_x(f64::NEG_INFINITY), f64::INFINITY);
        assert_eq!(log1p_minus_x(f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(exprel(f64::INFINITY), f64::INFINITY);
        assert_eq!(exprel(f64::NEG_INFINITY), 0.0);
        assert_eq!(log_exprel(f64::INFINITY), f64::INFINITY);
        assert_eq!(log_exprel(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(logaddexp(f64::INFINITY, f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn exponential_family_kernels_propagate_nan() {
        assert!(expm1_minus_x(f64::NAN).is_nan());
        assert!(log1p_minus_x(f64::NAN).is_nan());
        assert!(exprel(f64::NAN).is_nan());
        assert!(log_exprel(f64::NAN).is_nan());
        for x in [0.0, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(logaddexp(f64::NAN, x).is_nan());
            assert!(logaddexp(x, f64::NAN).is_nan());
        }
    }

    #[test]
    fn softplus_and_logistic_agree_with_their_definitions_away_from_the_tails() {
        for &x in &[-3.0_f64, -0.7, 0.0, 0.4, 2.5] {
            assert!((softplus(x) - (1.0 + x.exp()).ln()).abs() <= 4.0 * f64::EPSILON);
            assert!((logistic(x) - 1.0 / (1.0 + (-x).exp())).abs() <= 4.0 * f64::EPSILON);
        }
        assert_eq!(softplus(800.0), 800.0);
        assert_eq!(softplus(-800.0), 0.0);
    }

    #[test]
    fn remainders_match_the_direct_formula_where_it_does_not_cancel() {
        for &x in &[-0.75_f64, 0.6, 1.5] {
            assert!((expm1_minus_x(x) - (x.exp_m1() - x)).abs() <= 8.0 * f64::EPSILON);
            assert!((log1p_minus_x(x) - (x.ln_1p() - x)).abs() <= 8.0 * f64::EPSILON);
            assert!((exprel(x) - x.exp_m1() / x).abs() <= 8.0 * f64::EPSILON);
            assert!((log_exprel(x) - (x.exp_m1() / x).ln()).abs() <= 8.0 * f64::EPSILON);
        }
        // The series branch keeps relative accuracy where the naive form
        // would return pure cancellation noise.
        let x = 1.0e-6;
        let series = expm1_minus_x(x);
        assert!((series - 0.5 * x * x).abs() <= 1.0e-6 * 0.5 * x * x);
        assert!((log1p_minus_x(x) + 0.5 * x * x).abs() <= 1.0e-6 * 0.5 * x * x);
    }

    #[test]
    fn bd0_is_the_poisson_bregman_divergence() {
        assert_eq!(bd0(0.0, 2.5), 2.5);
        assert_eq!(bd0(3.0, 3.0), 0.0);
        for &(x, m) in &[(3.0_f64, 2.0_f64), (10.0, 10.5), (0.2, 7.0)] {
            let direct = x * (x / m).ln() + m - x;
            assert!((bd0(x, m) - direct).abs() <= 16.0 * f64::EPSILON * direct.abs().max(1.0));
        }
    }

    #[test]
    fn bernoulli_kl_from_logits_is_the_kl_divergence_between_the_two_bernoullis() {
        for &(a, b) in &[(0.3_f64, -0.2_f64), (-2.0, -1.8), (4.0, 1.0), (0.0, 0.0)] {
            let p = logistic(a);
            let q = logistic(b);
            let direct = xlogy(p, p / q) + xlogy(1.0 - p, (1.0 - p) / (1.0 - q));
            assert!((bernoulli_kl_from_logits(a, b) - direct).abs() <= 1.0e-13);
        }
    }

    #[test]
    fn logaddexp_and_log_abs_one_minus_exp_handle_their_edge_cases() {
        assert_eq!(logaddexp(f64::NEG_INFINITY, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!((logaddexp(1.0, 2.0) - (1.0_f64.exp() + 2.0_f64.exp()).ln()).abs() <= 4.0 * f64::EPSILON);
        assert!((log_abs_one_minus_exp(-1.0) - (1.0 - (-1.0_f64).exp()).ln()).abs() <= 4.0 * f64::EPSILON);
        assert!((log_abs_one_minus_exp(1.0) - (1.0_f64.exp() - 1.0).ln()).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn binary_exponent_arithmetic_round_trips_and_survives_intermediate_overflow() {
        for &x in &[1.0_f64, 0.3, 1.0e300, 5.0e-320, f64::MIN_POSITIVE] {
            let (mantissa, exponent) = positive_frexp(x);
            assert!((1.0..2.0).contains(&mantissa));
            assert_eq!(scale_normalized_power_of_two(mantissa, exponent), x);
        }
        // The inputs are decimal literals, so the exact product is a few ulps
        // off the decimal result; the point is that no intermediate overflowed
        // or underflowed on the way there.
        let got = scaled_positive_product_quotient(1.0e-300, 1.0, 1.0e308, 1.0);
        assert!((got - 1.0e8).abs() <= 4.0 * f64::EPSILON * 1.0e8);
        let got = scaled_positive_product_quotient(1.0e-300, 1.0e-200, 1.0, 1.0e-300);
        assert!((got - 1.0e-200).abs() <= 4.0 * f64::EPSILON * 1.0e-200);
        assert_eq!(scaled_positive_product_quotient(2.0, 3.0, 5.0, 4.0), 7.5);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_bessel_log_is_finite_and_derivative_consistent() {
        for eta in [0.25_f64, 1.0, 3.74, 3.76, 12.0, 900.0] {
            let (centered, ratio, scaled_derivative) = bessel_i0_centered_terms(eta);
            assert!(centered.is_finite());
            assert!((0.0..=1.0).contains(&ratio));

            // The tolerances below are sized by what a CENTRAL DIFFERENCE can
            // resolve — roundoff `ε·|f|/h` plus truncation `h²·f'''/6` — not by
            // what the evaluator happens to achieve. The A&S polynomials this
            // replaced needed `1e-6`/`2e-5` here; the series/asymptotic pair
            // leaves the finite difference itself as the limiting error.
            let h = 1.0e-4 * eta.max(1.0);
            let (plus, _) = bessel_i0_log_and_ratio(eta + h);
            let (minus, _) = bessel_i0_log_and_ratio(eta - h);
            let derivative = (plus - minus) / (2.0 * h);
            assert!(
                (derivative - ratio).abs() <= 1.0e-8,
                "d/dη log I0 mismatch at eta={eta}: analytic={ratio}, finite_difference={derivative}"
            );

            let log_step = 1.0e-5_f64;
            let (centered_plus, _, _) = bessel_i0_centered_terms(eta * log_step.exp());
            let (centered_minus, _, _) = bessel_i0_centered_terms(eta * (-log_step).exp());
            let finite_difference = (centered_plus - centered_minus) / (2.0 * log_step);
            assert!(
                (finite_difference - scaled_derivative).abs() < 1.0e-8,
                "centered Bessel value/gradient mismatch at eta={eta}: analytic={scaled_derivative}, finite_difference={finite_difference}"
            );
        }
        for eta in [1.0e20_f64, 1.0e100, 1.0e300] {
            let (centered, ratio, scaled_derivative) = bessel_i0_centered_terms(eta);
            let asymptotic = -0.5 * (std::f64::consts::TAU * eta).ln();
            assert!(centered.is_finite() && ratio.is_finite());
            // The `log S0` remainder is below `1e-20` at these arguments, so the
            // only admissible gap is the differing association of the two `log`
            // groupings — a few ulp of a number of size ~`log η`.
            assert!(
                (centered - asymptotic).abs() < 1.0e-13,
                "large-eta centered log must equal -½log(2πη); eta={eta:e}, centered={centered}, asymptotic={asymptotic}"
            );
            assert!(
                (scaled_derivative + 0.5).abs() < 1.0e-15,
                "large-eta centered derivative must retain its -1/2 limit; eta={eta:e}, derivative={scaled_derivative}"
            );
        }

        assert_eq!(bessel_i0_centered_terms(0.0), (0.0, 0.0, 0.0));

        let log_eta = 1_200.0;
        let (centered, ratio, scaled_derivative) = bessel_i0_centered_terms_from_log_abs(log_eta);
        assert!(centered.is_finite());
        assert_eq!(ratio, 1.0);
        assert_eq!(scaled_derivative, -0.5);
        assert_eq!(centered, -0.5 * (std::f64::consts::TAU.ln() + log_eta));
    }

    #[test]
    fn centered_bessel_second_log_derivative_matches_finite_difference() {
        // c''(log η) must be the derivative of the third term (c'(log η)) of
        // `bessel_i0_centered_terms`, across small, mid, and large arguments.
        // c''(log η) is the log-derivative of the STABLE third term `d1` (the
        // quantity the outer gradient's ARD normalizer channel reports), so the
        // self-consistent reference is a central difference of that same term.
        // The sweep straddles every seam this function has ever had: the retired
        // A&S 3.75 and 30 seams, and the live 1.0 (small-η rearrangement) and
        // 20.0 (series/asymptotic) ones.
        let first_log_derivative = |x: f64| bessel_i0_centered_terms(x).2;
        for eta in [
            0.02_f64, 0.05, 0.25, 0.999, 1.0, 1.001, 2.0, 3.5, 4.0, 8.0, 19.9, 20.1, 29.9, 30.1,
        ] {
            let log_eta = eta.ln();
            let analytic = bessel_i0_centered_second_log_derivative_from_log_abs(log_eta);

            let log_step = 1.0e-6_f64;
            let first_plus = first_log_derivative(eta * log_step.exp());
            let first_minus = first_log_derivative(eta * (-log_step).exp());
            let finite_difference = (first_plus - first_minus) / (2.0 * log_step);
            // `ε·|d1|/log_step ≈ 1e-10` of central-difference roundoff is the
            // floor here; the analytic value is far better than that. The old
            // `5e-5 + 1e-3·|analytic|` band was three orders wider than the
            // finite difference could even be wrong by — it was sized to the
            // 0.6% error the A&S polynomials put into `analytic`.
            assert!(
                (analytic - finite_difference).abs() < 1.0e-8 + 1.0e-6 * analytic.abs(),
                "centered Bessel second log-derivative mismatch at eta={eta}: \
                 analytic={analytic}, finite_difference={finite_difference}"
            );
        }
        // Large-η decay: the normalizer curvature vanishes like the leading
        // asymptotic term 1/(8η) (its Hessian contribution is then negligible
        // beside the ∝α energy term), stays finite and positive, and the
        // overflow-free gateway rounds it to exactly zero past the float range.
        // Held against THREE terms of the expansion rather than one, so the
        // admissible band is the size of the first omitted term (`≲ 2/η⁴`)
        // instead of a 25% shrug.
        for eta in [50.0_f64, 200.0, 1.0e4] {
            let c2 = bessel_i0_centered_second_log_derivative_from_log_abs(eta.ln());
            let inverse = 1.0 / eta;
            let expansion = inverse * (0.125 + inverse * (0.25 + inverse * (75.0 / 128.0)));
            assert!(
                c2 > 0.0 && (c2 - expansion).abs() < 8.0 * inverse.powi(4),
                "large-eta centered second derivative must track its own expansion; \
                 eta={eta}, c2={c2}, expansion={expansion}"
            );
        }
        // η → 0 and the overflow-free large-|η| gateway both round to 0.
        assert_eq!(
            bessel_i0_centered_second_log_derivative_from_log_abs(f64::NEG_INFINITY),
            0.0
        );
        assert_eq!(
            bessel_i0_centered_second_log_derivative_from_log_abs(1_200.0),
            0.0
        );
    }

    /// Every quantity `bessel_i0_centered_terms` and the second log-derivative
    /// return, against an INDEPENDENT 60-decimal-digit evaluation of the same
    /// closed forms (`mpmath.besseli`, `mpmath.diff`), rounded to `f64`.
    ///
    /// This is the assertion the module lacked. Everything else here is a
    /// self-consistency check — a finite difference of the evaluator against
    /// the evaluator — and a self-consistent evaluator can be uniformly wrong.
    /// The A&S 9.8.x polynomials this replaced were exactly that: internally
    /// consistent to the last digit and off the true value by up to `4e-6` in
    /// `d1` and `6e-3` in the curvature, with steps at their branch seams. No
    /// test in the tree compared them to anything but themselves.
    #[test]
    fn bessel_primitives_match_independent_high_precision_reference() {
        // (η, log I0(η) − η, I1(η)/I0(η), η(I1/I0 − 1), d²/d(log η)²[log I0 − η])
        const REFERENCE: [[f64; 5]; 24] = [
            [
                1e-06,
                -9.9999975e-07,
                4.999999999999375e-07,
                -9.999995e-07,
                -9.99999e-07,
            ],
            [
                0.001,
                -0.000999750000015625,
                0.0004999999375000105,
                -0.0009995000000625,
                -0.00099900000025,
            ],
            [
                0.05,
                -0.049375097629132,
                0.024992190753810217,
                -0.048750390462309494,
                -0.04750156152399669,
            ],
            [
                0.25,
                -0.23443561468661894,
                0.12403350191792471,
                -0.21899162452051882,
                -0.18846151934987648,
            ],
            [
                0.5,
                -0.4384502808145187,
                0.24249961258080194,
                -0.378750193709599,
                -0.2647015155254598,
            ],
            [
                1.0,
                -0.7640856414928213,
                0.4463899658965345,
                -0.5536100341034655,
                -0.19926400165310923,
            ],
            [
                2.0,
                -1.1760064585170438,
                0.697774657964008,
                -0.604450684071984,
                0.05244210681284669,
            ],
            [
                3.75,
                -1.5396457880279808,
                0.8531704594530685,
                -0.5506107770509933,
                0.0764086000777509,
            ],
            [
                5.0,
                -1.6953182241774665,
                0.8933831370440852,
                -0.5330843147795739,
                0.0466642611317311,
            ],
            [
                8.0,
                -1.941895744572186,
                0.9352354935294386,
                -0.5181160517644912,
                0.02141258513583364,
            ],
            [
                12.0,
                -2.1504975008971563,
                0.9573814053952422,
                -0.5114231352570932,
                0.01260162289404047,
            ],
            [
                17.0,
                -2.327961358737179,
                0.9701275885919403,
                -0.5078309939370159,
                0.008361475455484893,
            ],
            [
                19.5,
                -2.397561575434808,
                0.9740118676091061,
                -0.5067685816224307,
                0.007160287955186735,
            ],
            [
                19.999999,
                -2.410389546426233,
                0.9746705066059314,
                -0.5065898425518784,
                0.006960420318717729,
            ],
            [
                20.0,
                -2.4103895717557258,
                0.9746705078898071,
                -0.5065898422038575,
                0.006960419930170057,
            ],
            [
                20.000001,
                -2.410389597085217,
                0.9746705091736827,
                -0.5065898418558366,
                0.006960419541622429,
            ],
            [
                25.0,
                -2.5232719950007563,
                0.9797914534905159,
                -0.5052136627371017,
                0.005442291838848013,
            ],
            [
                30.0,
                -2.615298566828064,
                0.9831895553653361,
                -0.5043133390399173,
                0.004468398461442669,
            ],
            [
                64.0,
                -2.996411436485784,
                0.9921564935488112,
                -0.5019844128760834,
                0.002016497368136742,
            ],
            [
                150.0,
                -3.423420049648141,
                0.9966610736828279,
                -0.5008389475758167,
                0.0008446213361703931,
            ],
            [
                900.0,
                -4.319996948727984,
                0.9994442899516907,
                -0.5001390434784159,
                0.0001391983371050074,
            ],
            [
                10000.0,
                -5.524096218567699,
                0.999949998749875,
                -0.5000125012501954,
                1.2502500586100053e-05,
            ],
            [
                1000000.0,
                -7.826693687186747,
                0.999999499999875,
                -0.500000125000125,
                1.2500025000058594e-07,
            ],
            [
                1000000000000.0,
                -14.734449091168822,
                0.9999999999995,
                -0.500000000000125,
                1.2500000000025e-13,
            ],
        ];

        // Sized from the arithmetic, not from the outcome. The value and ratio
        // are read off sums that cannot cancel, so they land within a few ulp.
        // `d1` now shares that footing on BOTH branches: each reads it off a
        // difference series accumulated in its own right — `N/S0` above the
        // crossover, `−x(I0−I1)/I0` below it — rather than by subtracting the
        // ratio from one. The ascending difference series does have a sign
        // change, so it carries a condition number, but that grows only like
        // `√x` (6.8 at the crossover) instead of the `1/(1 − I1/I0)` ≈ 40 of the
        // naive form: tens of ulp, not thousands.
        //
        // The curvature is the one term still amplified, inheriting ≈ 2η from
        // `d1`'s ABSOLUTE error — `40 · 1e−15 / 0.007 ≈ 6e−12` just under the
        // crossover, where `c''` is smallest and `η` already large. That is
        // intrinsic to reaching `c''` through a `d1` held in one f64: `q = d1+½`
        // is `−0.0066` there, so even a correctly rounded `d1` pins `q` no
        // tighter than `ulp(½)/0.0066 ≈ 1.7e−14` relative.
        const CENTERED_TOL: f64 = 4.0e-15;
        const RATIO_TOL: f64 = 4.0e-15;
        const D1_TOL: f64 = 4.0e-15;
        const CURVATURE_TOL: f64 = 2.0e-11;

        for [eta, want_centered, want_ratio, want_d1, want_curvature] in REFERENCE {
            let (centered, ratio, d1) = bessel_i0_centered_terms(eta);
            let curvature = bessel_i0_centered_second_log_derivative_from_log_abs(eta.ln());
            let relative = |got: f64, want: f64| (got - want).abs() / want.abs();
            assert!(
                relative(centered, want_centered) < CENTERED_TOL,
                "log I0({eta}) − {eta}: got {centered:.17e}, want {want_centered:.17e}"
            );
            assert!(
                relative(ratio, want_ratio) < RATIO_TOL,
                "I1/I0({eta}): got {ratio:.17e}, want {want_ratio:.17e}"
            );
            assert!(
                relative(d1, want_d1) < D1_TOL,
                "η(I1/I0 − 1) at {eta}: got {d1:.17e}, want {want_d1:.17e}"
            );
            assert!(
                relative(curvature, want_curvature) < CURVATURE_TOL,
                "c''(log η) at {eta}: got {curvature:.17e}, want {want_curvature:.17e}"
            );
        }
    }

    /// The three returned terms are computed from DIFFERENT representations on
    /// BOTH branches — `ratio` from `S1/S0` or `I1/I0`, `d1` from the difference
    /// series each branch accumulates in its own right — so their defining
    /// relations are a real cross-check everywhere, not a tautology. (On the
    /// ascending branch it once WAS a tautology: `d1` was literally
    /// `η·(ratio − 1)`, so this assertion held by construction and the
    /// cancellation it is meant to detect went unmeasured.) Both must hold to
    /// within the cancellation the naive form suffers and the pre-cancelled one
    /// avoids.
    #[test]
    fn bessel_centered_terms_satisfy_their_defining_relations() {
        for eta in [
            0.5_f64, 1.0, 5.0, 12.0, 19.999, 20.0, 20.001, 25.0, 64.0, 900.0, 1.0e6, 1.0e12,
        ] {
            let (_centered, ratio, d1) = bessel_i0_centered_terms(eta);
            // d1 ≡ η(I1/I0 − 1). Forming it this way subtracts two numbers that
            // agree to `1/(2η)`, so it is only good to `≈ ε·η` — which is the
            // whole reason `d1` is carried separately.
            let naive = eta * (ratio - 1.0);
            assert!(
                (d1 - naive).abs() <= 8.0 * f64::EPSILON * eta,
                "d1 must equal η(I1/I0 − 1) at eta={eta}: d1={d1:.17e}, naive={naive:.17e}"
            );

            // c''(s) ≡ −η(2·d1 + 1) − d1², the rearrangement's starting point.
            // Both sides are fed the log-round-tripped argument the function
            // itself sees, so the ONLY admissible difference is the rounding of
            // the rearranged grouping: the two intermediate products are of
            // size `2η|d1|` and `d1²`, so a few ulp of those is the budget.
            //
            // Only checked where the naive form still HAS digits. Its two terms
            // both approach ¼ and cancel down to `1/(8η)`, which costs `≈ 8εη²`
            // in relative terms — already `1e-12` at η = 64 and total loss by
            // `η ≈ 1e8`. That collapse is the whole reason for the rearrangement,
            // so asserting agreement past it would assert nothing.
            if eta <= 64.0 {
                let round_tripped = eta.ln().exp();
                let (_, _, same_d1) = bessel_i0_centered_terms(round_tripped);
                let curvature = bessel_i0_centered_second_log_derivative_from_log_abs(eta.ln());
                let naive = -round_tripped * (2.0 * same_d1 + 1.0) - same_d1 * same_d1;
                let budget =
                    8.0 * f64::EPSILON * (2.0 * round_tripped * same_d1.abs() + same_d1 * same_d1);
                assert!(
                    (curvature - naive).abs() <= budget,
                    "c'' must equal −η(2d1+1) − d1² at eta={eta}: \
                     c2={curvature:.17e}, naive={naive:.17e}, budget={budget:.3e}"
                );
            }

            // `I1 < I0` for every η > 0, so `I1/I0 ∈ (0,1)` and `d1 < 0`. `d1`
            // is NOT monotone: it falls to a global minimum
            // `−0.608891247247801…` at `η = 1.702379944878764…` (the root of
            // `d(d1)/dη`) before rising back to its `−½` limit, so it crosses
            // `−½` once and the useful two-sided bound is that minimum.
            assert!((0.0..1.0).contains(&ratio), "I1/I0({eta})={ratio} ∉ (0,1)");
            assert!(
                (-0.608_891_247_247_802..0.0).contains(&d1),
                "η(I1/I0 − 1) at {eta} is {d1}, outside (min d1, 0)"
            );
        }
    }

    /// A branch crossover must not be observable in the output. The retired A&S
    /// pair stepped by `4e-6` in `d1` and `2e-4` in the curvature at its own
    /// 3.75 seam — a jump discontinuity in the objective and gradient an outer
    /// optimizer differentiates through.
    #[test]
    fn bessel_branch_crossovers_have_no_step() {
        // Every seam the implementation has ever carried.
        for seam in [1.0_f64, 3.75, 20.0, 30.0] {
            let delta = 1.0e-11 * seam;
            let (below_c, below_r, below_d1) = bessel_i0_centered_terms(seam - delta);
            let (above_c, above_r, above_d1) = bessel_i0_centered_terms(seam + delta);
            let below_c2 =
                bessel_i0_centered_second_log_derivative_from_log_abs((seam - delta).ln());
            let above_c2 =
                bessel_i0_centered_second_log_derivative_from_log_abs((seam + delta).ln());

            // Over `2δ` the true functions can move by at most `2δ·|f'|`, and
            // every derivative here is bounded by 1 in magnitude. Anything past
            // that plus a few ulp is a step, not a slope.
            let slope_budget = 2.0 * delta + 1.0e-14;
            assert!(
                (above_c - below_c).abs() < slope_budget,
                "centered log steps at the {seam} seam: {below_c:.17e} -> {above_c:.17e}"
            );
            assert!(
                (above_r - below_r).abs() < slope_budget,
                "I1/I0 steps at the {seam} seam: {below_r:.17e} -> {above_r:.17e}"
            );
            assert!(
                (above_d1 - below_d1).abs() < slope_budget,
                "d1 steps at the {seam} seam: {below_d1:.17e} -> {above_d1:.17e}"
            );
            assert!(
                (above_c2 - below_c2).abs() < slope_budget,
                "c'' steps at the {seam} seam: {below_c2:.17e} -> {above_c2:.17e}"
            );
        }
    }

    /// Non-finite and boundary arguments keep their documented limits, and no
    /// series loop can run away on them.
    #[test]
    fn bessel_primitives_handle_boundary_arguments() {
        let (centered, ratio, d1) = bessel_i0_centered_terms(f64::INFINITY);
        assert_eq!((centered, ratio, d1), (f64::NEG_INFINITY, 1.0, -0.5));
        let (centered, ratio, d1) = bessel_i0_centered_terms(f64::NEG_INFINITY);
        assert_eq!((centered, ratio, d1), (f64::NEG_INFINITY, 1.0, -0.5));

        let (centered, ratio, d1) = bessel_i0_centered_terms(f64::NAN);
        assert!(centered.is_nan() && ratio.is_nan() && d1.is_nan());
        assert!(bessel_i0_centered_second_log_derivative_from_log_abs(f64::NAN).is_nan());

        // I0 and I1 are even/odd, so every returned term is a function of |η|.
        for eta in [0.5_f64, 5.0, 25.0, 1.0e6] {
            assert_eq!(
                bessel_i0_centered_terms(-eta),
                bessel_i0_centered_terms(eta)
            );
        }
    }

    /// The polygamma family against a 50-digit `mpmath` evaluation.
    ///
    /// These consolidate four separate hand-rolled Bernoulli-series copies that
    /// had drifted apart: `gam-sae` recursed to 10 and stopped at `B₆`/`B₆`,
    /// `gam-solve` recursed to 8 and stopped at `B₁₀`/`B₁₀`/`B₁₀`, `gam-terms`
    /// recursed to 8 with yet another term count. Measured against this oracle
    /// they were good to `7.6e−10`, `3.1e−10`, `6.3e−11`, `3.9e−11` and
    /// `2.6e−10` respectively — 10 to 11 digits, and mutually inconsistent at
    /// that scale, in code that supplies REML gradients and Hessians for the
    /// negative-binomial `θ`, Gamma dispersion and Beta shape channels.
    #[test]
    fn polygamma_family_matches_independent_high_precision_reference() {
        // (x, ψ(x), ψ₁(x), ψ₂(x), ψ₃(x))
        const POLYGAMMA_REFERENCE: [[f64; 5]; 22] = [
            [
                1e-08,
                -100000000.57721564,
                1.0000000000000002e+16,
                -2e+24,
                5.999999999999999e+32,
            ],
            [
                0.0001,
                -10000.577051183514,
                100000001.64469367,
                -2000000000002.403,
                5.999999999999999e+16,
            ],
            [
                0.01,
                -100.56088545786868,
                10001.621213528313,
                -2000002.340398677,
                600000006.2510618,
            ],
            [
                0.1,
                -10.423754940411076,
                101.43329915079275,
                -2001.8614573783436,
                60004.51287679026,
            ],
            [
                0.25,
                -4.2274535333762655,
                17.19732915450711,
                -129.32773993753693,
                1538.7821440091884,
            ],
            [
                0.5,
                -1.9635100260214235,
                4.934802200544679,
                -16.82879664423432,
                97.40909103400244,
            ],
            [
                1.0,
                -0.5772156649015329,
                1.6449340668482264,
                -2.4041138063191885,
                6.493939402266829,
            ],
            // The unique positive root of ψ, where the relative bound below is
            // vacuous and the absolute one carries the assertion.
            [
                1.4616321449683622,
                -9.241265521729427e-17,
                0.9676722454476212,
                -0.8855263379671844,
                1.5509985657339065,
            ],
            [
                2.0,
                0.42278433509846713,
                0.6449340668482264,
                -0.4041138063191886,
                0.49393940226682914,
            ],
            [
                3.5,
                1.103156640645243,
                0.3303577561002349,
                -0.1082040516417274,
                0.07030584881725205,
            ],
            // The three retired recurrence thresholds (8 and 10) and the live
            // one (20), each straddled.
            [
                7.0,
                1.8727843350984672,
                0.15354517795933756,
                -0.023530472985855238,
                0.007198198563125445,
            ],
            [
                8.0,
                2.01564147795561,
                0.1331370146940314,
                -0.017699569195767775,
                0.004699239795945104,
            ],
            [
                10.0,
                2.251752589066721,
                0.10516633568168575,
                -0.011049834970802067,
                0.0023199013042898686,
            ],
            [
                19.0,
                2.9178924132947808,
                0.05404090603769619,
                -0.0029197100973139254,
                0.0003154143837079449,
            ],
            [
                19.999,
                2.9704727201051075,
                0.05127345119229945,
                -0.0026283917972403977,
                0.00026941563155986057,
            ],
            [
                20.0,
                2.970523992242149,
                0.05127082293520312,
                -0.0026281224023146548,
                0.0002693742213396389,
            ],
            [
                20.001,
                2.970575261751068,
                0.05126819494748101,
                -0.0026278530487948894,
                0.0002693328196036835,
            ],
            [
                25.0,
                3.198742512851974,
                0.04081066325722558,
                -0.001665279318422468,
                0.0001358846365082737,
            ],
            [
                100.0,
                4.600161852738087,
                0.010050166663333571,
                -0.00010100499983335,
                2.030199990001333e-06,
            ],
            [
                10000.0,
                9.210290371142849,
                0.00010000500016666666,
                -1.000100005e-08,
                2.00030002e-12,
            ],
            [
                100000000.0,
                18.420680738952367,
                1.000000005e-08,
                -1.00000001e-16,
                2.0000000300000002e-24,
            ],
            [
                1000000000000000.0,
                34.538776394910684,
                1.0000000000000005e-15,
                -1.000000000000001e-30,
                2.000000000000003e-45,
            ],
        ];

        for [x, want_psi, want_psi1, want_psi2, want_psi3] in POLYGAMMA_REFERENCE {
            // `ψ` crosses zero at x ≈ 1.4616, and the recurrence sums up to 20
            // reciprocals whose partial sums dwarf a near-zero result, so the
            // absolute term is what applies there. Everywhere else the relative
            // term binds. `1e-14` relative is 4 orders tighter than the loosest
            // implementation this replaced.
            let checks = [
                ("ψ", digamma(x), want_psi),
                ("ψ₁", trigamma(x), want_psi1),
                ("ψ₂", tetragamma(x), want_psi2),
                ("ψ₃", pentagamma(x), want_psi3),
            ];
            for (name, got, want) in checks {
                let error = (got - want).abs();
                let budget = 1e-14 * want.abs() + 1e-15;
                assert!(
                    error <= budget,
                    "{name}({x}): got {got:.17e}, want {want:.17e} \
                     (error {error:.3e} > {budget:.3e})"
                );
            }
        }
    }

    /// The recurrences and the asymptotic series must agree where they meet,
    /// and each function must be the derivative of the one before it. Both were
    /// true of the copies this replaces only to their own `1e-10`.
    #[test]
    fn polygamma_family_is_seamless_and_mutually_consistent() {
        for threshold in [8.0_f64, 10.0, 20.0] {
            let delta = 1.0e-11 * threshold;
            for f in [digamma as fn(f64) -> f64, trigamma, tetragamma, pentagamma] {
                let below = f(threshold - delta);
                let above = f(threshold + delta);
                // Every one of these has |f'| < 1 at x ≥ 8, so the true change
                // over 2δ is below 2δ. Anything more is a step.
                assert!(
                    (above - below).abs() < 2.0 * delta + 1.0e-15,
                    "polygamma step at the {threshold} seam: {below:.17e} -> {above:.17e}"
                );
            }
        }

        // ψ_{n+1} = dψ_n/dx, checked by a central difference whose own error
        // (roundoff ε|f|/h plus truncation h²f'''/6) is the limit here.
        for x in [0.75_f64, 1.5, 4.0, 9.0, 19.5, 21.0, 60.0] {
            let h = 1.0e-4 * x;
            for (name, value, derivative) in [
                ("ψ", digamma as fn(f64) -> f64, trigamma as fn(f64) -> f64),
                ("ψ₁", trigamma, tetragamma),
                ("ψ₂", tetragamma, pentagamma),
            ] {
                let finite_difference = (value(x + h) - value(x - h)) / (2.0 * h);
                let analytic = derivative(x);
                assert!(
                    (finite_difference - analytic).abs() <= 1e-6 * analytic.abs().max(1e-3),
                    "d{name}/dx at {x}: analytic={analytic:.17e}, fd={finite_difference:.17e}"
                );
            }
        }

        // Non-positive and non-finite arguments are outside the domain.
        for bad in [
            0.0_f64,
            -1.0,
            -0.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert!(digamma(bad).is_nan(), "digamma({bad}) must be NaN");
            assert!(trigamma(bad).is_nan(), "trigamma({bad}) must be NaN");
            assert!(tetragamma(bad).is_nan(), "tetragamma({bad}) must be NaN");
            assert!(pentagamma(bad).is_nan(), "pentagamma({bad}) must be NaN");
        }
    }

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        // An n-point rule is exact for polynomials of degree ≤ 2n−1.
        for n in [1usize, 2, 3, 5, 8, 40, 64] {
            let (nodes, weights) = gauss_legendre(n);
            assert_eq!(nodes.len(), n);
            assert_eq!(weights.len(), n);
            assert!(nodes.windows(2).all(|w| w[0] < w[1]), "nodes ascending");
            if !n.is_multiple_of(2) {
                assert_eq!(nodes[n / 2], 0.0, "odd-n central node is exact zero");
            }
            let total: f64 = weights.iter().sum();
            assert!((total - 2.0).abs() < 1e-13, "∫1 dx = 2, got {total}");
            if n >= 2 {
                let x2: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x * x).sum();
                assert!((x2 - 2.0 / 3.0).abs() < 1e-13, "∫x² dx = 2/3, got {x2}");
            }
            // Degrees 0 and 2 alone exercise almost none of the rule — the
            // weights barely matter there. Assert the whole `2n−1` guarantee.
            //
            // The odd degrees integrate to zero by symmetry, so the error is
            // measured against `Σ|w·xᵈ|`, the size of the terms that had to
            // cancel, rather than against the vanishing answer. For the even
            // degrees every term is positive and that denominator IS the exact
            // value, so the same expression is the ordinary relative error.
            for degree in 0..(2 * n) {
                let term = |(x, w): (&f64, &f64)| w * x.powi(degree as i32);
                let quadrature: f64 = nodes.iter().zip(&weights).map(term).sum();
                let magnitude: f64 = nodes.iter().zip(&weights).map(|p| term(p).abs()).sum();
                let exact = if degree % 2 == 1 {
                    0.0
                } else {
                    2.0 / (degree as f64 + 1.0)
                };
                let scale = magnitude.max(exact);
                // `n = 1` puts its only node at exactly zero, so every odd
                // degree has nothing to cancel and must come out exactly zero.
                if scale == 0.0 {
                    assert_eq!(quadrature, 0.0, "n={n}, x^{degree}");
                    continue;
                }
                assert!(
                    (quadrature - exact).abs() / scale < 1.0e-13,
                    "n={n} rule must integrate x^{degree} exactly: got {quadrature:.17e}, \
                     want {exact:.17e}"
                );
            }
        }
    }

    /// The nodes are Newton-converged to ~1 ulp, but the weights are read off
    /// `P_n'` and were being evaluated one Newton step BEHIND the node they are
    /// paired with. Legendre's equation turns that lag into `2·2z/(1−z²)` times
    /// the node offset, so it is invisible in the nodes and plainly visible
    /// here: at `n = 16` the weights carried `8.3e−14`, against the `1.9e−15`
    /// they carry once `P_n'` is re-evaluated at the returned node.
    #[test]
    fn gauss_legendre_weights_match_independent_high_precision_reference() {
        // (node, weight) over the positive half, from a 50-digit root solve;
        // the rule is symmetric, so the negative half is the mirror image.
        const GL8: [(f64, f64); 4] = [
            (0.183434642495649805, 0.362683783378361983),
            (0.525532409916328986, 0.313706645877887287),
            (0.796666477413626740, 0.222381034453374471),
            (0.960289856497536232, 0.101228536290376259),
        ];
        const GL16: [(f64, f64); 8] = [
            (0.0950125098376374402, 0.189450610455068496),
            (0.281603550779258913, 0.182603415044923589),
            (0.458016777657227386, 0.169156519395002538),
            (0.617876244402643748, 0.149595988816576732),
            (0.755404408355003034, 0.124628971255533872),
            (0.865631202387831744, 0.0951585116824927848),
            (0.944575023073232576, 0.0622535239386478929),
            (0.989400934991649933, 0.0271524594117540949),
        ];

        // The nodes are a Newton root to within a few ulp of 1. The weights sit
        // an order looser because `2z/(1−z²)` amplifies whatever the node's
        // residual is — but two orders TIGHTER than the stale-derivative form,
        // which is what this pins.
        const NODE_TOL: f64 = 4.0e-16;
        const WEIGHT_TOL: f64 = 1.0e-14;

        for (n, reference) in [(8usize, &GL8[..]), (16, &GL16[..])] {
            let (nodes, weights) = gauss_legendre(n);
            for (k, &(want_node, want_weight)) in reference.iter().enumerate() {
                // Positive half, ascending, is the back half of the rule.
                let index = n / 2 + k;
                let (got_node, got_weight) = (nodes[index], weights[index]);
                assert!(
                    (got_node - want_node).abs() < NODE_TOL,
                    "n={n} node {index}: got {got_node:.17e}, want {want_node:.17e}"
                );
                let relative = (got_weight - want_weight).abs() / want_weight.abs();
                assert!(
                    relative < WEIGHT_TOL,
                    "n={n} weight {index}: got {got_weight:.17e}, want {want_weight:.17e}, \
                     rel {relative:.3e}"
                );
                // Symmetry: the mirrored entry must be bit-identical.
                let mirror = n / 2 - 1 - k;
                assert_eq!(
                    nodes[mirror], -got_node,
                    "n={n} node {mirror} mirrors {index}"
                );
                assert_eq!(weights[mirror], got_weight, "n={n} weight {mirror} mirrors");
            }
        }
    }

    #[test]
    fn binom_k_exceeds_n_returns_zero() {
        assert_eq!(binomial_coefficient_f64(3, 5), 0.0);
        assert_eq!(binomial_coefficient_f64(0, 1), 0.0);
        assert_eq!(binomial_coefficient_f64(10, 11), 0.0);
    }

    #[test]
    fn binom_k_zero_returns_one() {
        assert_eq!(binomial_coefficient_f64(0, 0), 1.0);
        assert_eq!(binomial_coefficient_f64(5, 0), 1.0);
        assert_eq!(binomial_coefficient_f64(100, 0), 1.0);
    }

    #[test]
    fn binom_k_equals_n_returns_one() {
        assert_eq!(binomial_coefficient_f64(1, 1), 1.0);
        assert_eq!(binomial_coefficient_f64(5, 5), 1.0);
        assert_eq!(binomial_coefficient_f64(20, 20), 1.0);
    }

    #[test]
    fn binom_small_exact_values() {
        assert_eq!(binomial_coefficient_f64(5, 2), 10.0);
        assert_eq!(binomial_coefficient_f64(10, 3), 120.0);
        assert_eq!(binomial_coefficient_f64(20, 10), 184_756.0);
        assert_eq!(binomial_coefficient_f64(6, 3), 20.0);
    }

    #[test]
    fn binom_symmetry() {
        assert_eq!(
            binomial_coefficient_f64(10, 3),
            binomial_coefficient_f64(10, 7)
        );
        assert_eq!(
            binomial_coefficient_f64(20, 5),
            binomial_coefficient_f64(20, 15)
        );
        assert_eq!(
            binomial_coefficient_f64(54, 24),
            binomial_coefficient_f64(54, 30)
        );
    }

    #[test]
    fn binom_c54_24_is_exact() {
        // The u128-recurrence fix restored this value (old f64 recurrence
        // returned 1_402_659_561_581_459, one short of the true integer).
        assert_eq!(binomial_coefficient_f64(54, 24), 1_402_659_561_581_460.0);
    }

    #[test]
    fn binom_large_finite_row_sums_to_the_binomial_theorem() {
        let total: f64 = (0..=1023).map(|k| binomial_coefficient_f64(1023, k)).sum();
        let expected = f64::from_bits(2046_u64 << 52); // 2^1023
        assert!((total / expected - 1.0).abs() < 2e-14, "row sum={total:e}");
        assert!(binomial_coefficient_f64(1029, 514).is_finite());
        assert_eq!(binomial_coefficient_f64(1030, 515), f64::INFINITY);
    }

    #[test]
    fn poly_exp_empty_coeffs_returns_zero() {
        assert_eq!(stable_polynomial_times_exp_neg(1.0, &[]), 0.0);
        assert_eq!(stable_polynomial_times_exp_neg(0.0, &[]), 0.0);
        assert_eq!(stable_polynomial_times_exp_neg(700.0, &[]), 0.0);
    }

    #[test]
    fn poly_exp_nonfinite_x_returns_zero() {
        assert_eq!(
            stable_polynomial_times_exp_neg(f64::INFINITY, &[1.0, 2.0]),
            0.0
        );
        assert_eq!(
            stable_polynomial_times_exp_neg(f64::NEG_INFINITY, &[1.0, 2.0]),
            0.0
        );
        assert_eq!(stable_polynomial_times_exp_neg(f64::NAN, &[1.0]), 0.0);
    }

    #[test]
    fn poly_exp_constant_at_zero() {
        // At x=0: poly(0) = coeffs[0], exp(0)=1 → result = coeffs[0].
        assert_eq!(stable_polynomial_times_exp_neg(0.0, &[5.0]), 5.0);
        assert_eq!(stable_polynomial_times_exp_neg(0.0, &[3.0, 1.0, 2.0]), 3.0);
    }

    #[test]
    fn poly_exp_constant_poly_direct_path() {
        // x=2.0 < 600: direct Horner * exp(-x).
        let x = 2.0;
        let got = stable_polynomial_times_exp_neg(x, &[3.0]);
        let expected = 3.0 * (-x).exp();
        assert!(
            (got - expected).abs() < 1e-14,
            "got={got} expected={expected}"
        );
    }

    #[test]
    fn poly_exp_linear_poly_direct_path() {
        // coeffs = [a, b] → poly = a + b*x.
        let x = 1.5;
        let (a, b) = (2.0, 3.0);
        let got = stable_polynomial_times_exp_neg(x, &[a, b]);
        let expected = (a + b * x) * (-x).exp();
        assert!(
            (got - expected).abs() < 1e-14,
            "got={got} expected={expected}"
        );
    }

    #[test]
    fn poly_exp_constant_poly_asymptotic_path() {
        // x=700 > 600: asymptotic path. For poly = [1.0], result = exp(-700).
        let x = 700.0_f64;
        let got = stable_polynomial_times_exp_neg(x, &[1.0]);
        let expected = (-x).exp();
        let rel = (got - expected).abs() / expected;
        assert!(rel < 1e-12, "got={got} expected={expected} rel={rel}");
    }

    #[test]
    fn poly_exp_quadratic_asymptotic_path() {
        // x=620 > 600: poly = x^2 (coeffs=[0,0,1]). Result = x^2 * exp(-x).
        // x=800 would underflow to 0.0 in both the asymptotic path and the
        // reference, making the relative-error check degenerate; x=620 keeps
        // the result in the normal f64 range (~10^-264) while still exercising
        // the asymptotic branch (threshold is x=600).
        let x = 620.0_f64;
        let got = stable_polynomial_times_exp_neg(x, &[0.0, 0.0, 1.0]);
        let expected = (2.0 * x.ln() - x).exp();
        let rel = (got - expected).abs() / expected.abs();
        assert!(rel < 1e-12, "got={got} expected={expected} rel={rel}");
    }

    /// Pins the measured accuracy of `c''(log η)` against an 80-digit
    /// reference, per regime, so the branch structure cannot silently drift.
    ///
    /// The tolerances are the MEASURED worst case in each regime plus a factor
    /// of two, not aspirations: the `1 ≤ η < 20` band is bounded below by the
    /// `8η²·κ(η)·ε` floor of the ascending representation (see
    /// [`bessel_i0_centered_second_log_derivative_from_log_abs`]), and 1e-11 is
    /// what that floor permits at the top of the band. Tightening it needs a
    /// wider-than-`f64` `d1`, not a smaller constant here.
    #[test]
    fn centered_bessel_second_log_derivative_matches_high_precision_reference() {
        // (η, c''(log η) to 20 significant digits, tolerance).
        const CASES: [(f64, f64, f64); 13] = [
            (0.5, -0.2647015155254598, 1e-14),
            (1.0, -0.19926400165310923, 1e-14),
            (2.0, 0.05244210681284669, 1e-13),
            (5.0, 0.0466642611317311, 1e-13),
            (10.0, 0.015837019843595493, 1e-12),
            (15.0, 0.009659446256568909, 1e-11),
            // The worst point of the whole domain, just under the crossover.
            (18.85, 0.00743799786561837, 1e-11),
            (19.99, 0.006964307582746309, 1e-11),
            // First point on the asymptotic side: two orders better, at once.
            (20.0, 0.006960419930170057, 1e-12),
            (25.0, 0.005442291838848013, 1e-14),
            (50.0, 0.0026049656149811874, 1e-15),
            (200.0, 0.0006313242744933583, 1e-15),
            (1e4, 1.2502500586100053e-05, 1e-15),
        ];
        for (eta, expected, tolerance) in CASES {
            let got = bessel_i0_centered_second_log_derivative_from_log_abs(eta.ln());
            let relative = (got - expected).abs() / expected.abs();
            assert!(
                relative < tolerance,
                "eta={eta}: got={got} expected={expected} rel={relative:e} tol={tolerance:e}"
            );
        }
    }

    /// The crossover at `BESSEL_ASYMPTOTIC_THRESHOLD` is a step DOWN in error,
    /// so the value itself must still be continuous across it to within what
    /// the worse (ascending) side delivers — nothing tighter is available, and
    /// nothing looser would catch a branch that had been mis-derived.
    ///
    /// The step size matters and is not free to enlarge. `c''` genuinely varies:
    /// `|dc''/dη| / |c''| = 1/η`, so a step `δ` moves the true value by `δ/η`
    /// RELATIVE. At `δ = 1e−9` that is `5e−11` — larger than the seam being
    /// measured, and a test written that way reports the function's own slope
    /// as a discontinuity. `1e−11` puts the true variation at `5e−13`, an order
    /// under the ascending branch's `8.9e−12` floor, while still clearing
    /// `ulp(20) = 3.6e−15` by four orders.
    #[test]
    fn centered_bessel_second_log_derivative_is_continuous_across_the_crossover() {
        const STEP: f64 = 1e-11;
        let below = bessel_i0_centered_second_log_derivative_from_log_abs(
            (BESSEL_ASYMPTOTIC_THRESHOLD - STEP).ln(),
        );
        let above =
            bessel_i0_centered_second_log_derivative_from_log_abs(BESSEL_ASYMPTOTIC_THRESHOLD.ln());
        assert!(
            below != above,
            "step {STEP:e} was rounded away; the two sides are the same evaluation"
        );
        let jump = (below - above).abs() / above.abs();
        assert!(
            jump < 3e-11,
            "seam jump {jump:e}: below={below} above={above}"
        );
    }
}
