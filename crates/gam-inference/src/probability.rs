use gam_math::probability::{beta_quantile, is_binomial_trial_count};
use statrs::function::beta::beta_reg;

/// Standard normal PDF φ(x).  Implementation lives in `gam-math`; re-exported
/// to keep `crate::probability::normal_pdf` resolving for all existing callers.
pub use gam_math::probability::normal_pdf;

/// Standard normal CDF Φ(x) via the exact identity `Φ(x) = 0.5·erfc(−x/√2)`.
/// Implementation lives in `gam-math`; re-exported to keep
/// `crate::probability::normal_cdf` resolving for all existing callers.
pub use gam_math::probability::normal_cdf;

/// Inverse-Gaussian CDF `IG(μ, λ)`. Implementation lives in `gam-math`.
pub use gam_math::probability::inverse_gaussian_cdf;

/// Two-sided standard-normal probability `P(|Z| ≥ |z|)`, evaluated directly
/// without subtracting a CDF from one.
pub use gam_math::probability::normal_two_sided_probability;

/// Two-sided Student-t probability `P(|T_ν| ≥ |t|)`, evaluated through the
/// direct regularized-beta tail in `gam-math`.
pub use gam_math::probability::student_t_two_sided_probability;

/// Chi-squared survival probability, evaluated through the regularized upper
/// incomplete gamma in `gam-math`.
pub use gam_math::probability::chi_square_sf;

/// Fisher-Snedecor survival probability, evaluated through the complementary
/// regularized-beta identity in `gam-math`.
pub use gam_math::probability::fisher_snedecor_sf;

/// Scaled complementary error function `erfcx(x) = exp(x²) · erfc(x)`,
/// specialized to `x ≥ 0`.  The implementation now lives in the lowest crate
/// (`gam-math`) so the survival/probit cluster can consume it without reaching
/// up into `inference`; re-exported here to keep
/// `crate::probability::erfcx_nonnegative` resolving for all existing callers.
pub use gam_math::probability::erfcx_nonnegative;

/// Computes `log(1 - exp(-a))` for `a >= 0` without cancellation.  Implementation
/// lives in `gam-math`; re-exported to keep `crate::probability::log1mexp_positive`
/// resolving for all existing callers.
pub use gam_math::probability::log1mexp_positive;

/// Numerically stable signed log-sum-exp.  Implementation lives in `gam-math`;
/// re-exported to keep `crate::probability::signed_log_sum_exp` resolving for all
/// existing callers.
pub use gam_math::probability::signed_log_sum_exp;

/// Numerically stable `C(n,k) = n! / (k!·(n−k)!)` as `f64`.  The
/// implementation now lives in the lowest crate (`gam-math`) so the
/// terms/basis cluster can consume it without reaching up into `inference`;
/// re-exported here to keep `crate::probability::binomial_coefficient_f64`
/// resolving for all existing callers.
pub use gam_math::special::binomial_coefficient_f64;

/// Evaluate `(Σ_k coeffs[k]·x^k) · exp(−x)` without overflow. The
/// implementation lives in `gam-math` so terms/basis code can consume it without
/// reaching up into `inference`.
pub use gam_math::special::stable_polynomial_times_exp_neg;

/// Numerically stable `ln Φ(x)` for the standard normal CDF.  Implementation lives
/// in `gam-math`; re-exported to keep `crate::probability::normal_logcdf` resolving
/// for all existing callers.
pub use gam_math::probability::normal_logcdf;

/// Numerically stable `ln(1 − Φ(x)) = ln Φ(−x)` for the standard normal survival
/// function.  Implementation lives in `gam-math`; re-exported to keep
/// `crate::probability::normal_logsf` resolving for all existing callers.
pub use gam_math::probability::normal_logsf;

/// Joint evaluation of `ln Φ(x)` and the Mills-ratio analogue `φ(x) / Φ(x)`.
/// Implementation lives in `gam-math`; re-exported to keep
/// `crate::probability::signed_probit_logcdf_and_mills_ratio` resolving for all
/// existing callers.
pub use gam_math::probability::signed_probit_logcdf_and_mills_ratio;

/// Standard normal quantile Φ⁻¹(p) using Acklam's rational approximation.
/// Implementation lives in `gam-math`; re-exported to keep
/// `crate::probability::standard_normal_quantile` resolving for all existing callers.
pub use gam_math::probability::standard_normal_quantile;

/// The regularized incomplete gamma pair and its inverse live in `gam-math`,
/// beside the chi-square tails built on them.
use gam_math::probability::{inverse_regularized_lower_gamma, regularized_incomplete_gamma_pair};

/// Quantile (inverse CDF) of a Gamma distribution parameterized by shape
/// `k > 0` and scale `θ > 0` at probability `p ∈ (0, 1)`: the value `x` with
/// `P(X ≤ x) = p` for `X ~ Gamma(shape = k, scale = θ)` (mean `kθ`, variance
/// `kθ²`).
///
/// Equals `θ · Q(p; k)`, where `Q(p; k)` inverts the regularized lower
/// incomplete gamma `P(k, x)` (the unit-scale Gamma CDF). `p ≤ 0` maps to the
/// `0` support floor and `p ≥ 1` to `+∞`; a non-finite or non-positive shape or
/// scale yields `NaN`.
///
/// This is the building block for *skew-aware* response-scale predictive
/// (observation) intervals: a Gamma response is strongly right-skewed, so the
/// symmetric `μ ± z·σ` band mis-covers each tail even when its width (variance)
/// is correct. Equal-tailed Gamma quantiles place the right mass in each tail.
pub(crate) fn gamma_quantile(p: f64, shape: f64, scale: f64) -> f64 {
    if !(shape.is_finite() && shape > 0.0 && scale.is_finite() && scale > 0.0) {
        return f64::NAN;
    }
    scale * inverse_regularized_lower_gamma(p, shape)
}

/// The band of the point mass at zero, the one law on `[0, ∞)` whose mean is zero:
/// what a predictive mean that underflows to zero, with no spread, describes. A
/// non-negative response's moment-matched predictive takes it before asking for a
/// positive mean, so such a row has the exact band `[0, 0]` rather than none
/// (#3140).
fn zero_point_mass(mu: f64, total_var: f64) -> Option<(f64, f64)> {
    (mu == 0.0 && total_var == 0.0).then_some((0.0, 0.0))
}

/// Mass the point mass at zero of [`zero_point_mass`] puts on `[lower, upper]`:
/// one when the set holds zero, else none. `None` when the moments are not that
/// law's.
fn zero_point_mass_content(mu: f64, total_var: f64, lower: f64, upper: f64) -> Option<f64> {
    zero_point_mass(mu, total_var)
        .map(|_| if lower <= 0.0 && 0.0 <= upper { 1.0 } else { 0.0 })
}

/// Equal-tailed predictive interval for a strictly-positive, right-skewed
/// response modelled as a Gamma whose first two moments match a point
/// prediction: mean `mu` and total predictive variance `total_var`
/// (estimation + observation noise). Returns the pair of Gamma quantiles at
/// lower-tail probabilities `p_lo < p_hi` — the skew-correct replacement for a
/// symmetric `mu ± z·σ` band, which for a Gamma pins the lower edge near the
/// support floor and mis-covers each tail (#817).
///
/// Moment matching fixes `shape k = mu²/V` and `scale θ = V/mu`, so the
/// predictive carries exactly the requested mean and variance. When estimation
/// uncertainty vanishes (`total_var → φμ²`) this is *exact*: `k → 1/φ`,
/// `θ → φμ`, recovering the conditional Gamma `Gamma(shape = 1/φ, scale = φμ)`.
/// With nonzero estimation variance it is the moment-matched Gamma predictive —
/// the minimal skew-correct widening.
///
/// Returns `None` when the inputs are degenerate (non-positive mean or
/// variance, non-finite), or when the incomplete-gamma inverse yields a
/// non-finite / mis-ordered pair. An enormous shape is not such a case: the
/// incomplete gamma is evaluated by Temme's uniform expansion wherever the
/// series and the continued fraction stall (#4068), so the band stays the
/// Gamma's own however close to Gaussian it is.
pub fn gamma_moment_matched_interval(
    mu: f64,
    total_var: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if let Some(band) = zero_point_mass(mu, total_var) {
        return Some(band);
    }
    if !(mu.is_finite() && mu > 0.0 && total_var.is_finite() && total_var > 0.0) {
        return None;
    }
    let shape = mu * mu / total_var;
    let scale = total_var / mu;
    let q_lo = gamma_quantile(p_lo, shape, scale);
    let q_hi = gamma_quantile(p_hi, shape, scale);
    if q_lo.is_finite() && q_hi.is_finite() && q_hi >= q_lo {
        Some((q_lo, q_hi))
    } else {
        None
    }
}

/// Equal-tailed predictive interval for a `(0, 1)`-bounded response modelled as a
/// Beta whose first two moments match a point prediction: mean `mu ∈ [0, 1]`, its
/// complement `complement = 1 − mu` carried separately (see below), and total
/// predictive variance `total_var` (estimation + observation noise).
/// Returns the pair of Beta quantiles at lower-tail probabilities `p_lo < p_hi` —
/// the skew-correct replacement for a symmetric `mu ± z·σ` band, which for a
/// skewed Beta lands *both* edges below the corresponding true quantile and so
/// mis-covers each tail (#1194).
///
/// Moment matching fixes the precision `φ = a + b = μ(1−μ)/V − 1`, then
/// `a = μφ`, `b = (1−μ)φ`, so the predictive carries exactly the requested mean
/// and variance. When estimation uncertainty vanishes
/// (`total_var → μ(1−μ)/(1+φ₀)`) this is *exact*: `φ → φ₀`, recovering the
/// conditional `Beta(μφ₀, (1−μ)φ₀)`. With nonzero estimation variance it is the
/// moment-matched Beta predictive — the minimal skew-correct widening.
///
/// The Bernoulli ceiling `μ(1−μ)` and the shape `b` are read from `complement`,
/// never from `1 − mu`: where the mean rounds to one, `1 − mu` is exactly zero
/// while the complement is still a positive number, and the band lives on it
/// (#3140). A predictive with no spread (`total_var = 0`) is the point mass at
/// its mean, whose band is that point.
///
/// Returns `None` when the inputs are not the moments of a law on `[0, 1]`
/// (negative or non-finite, or a variance at or past the Bernoulli ceiling
/// `μ(1−μ)`, which no Beta reaches), or when the quantiles come out non-finite
/// or mis-ordered.
pub fn beta_moment_matched_interval(
    mu: f64,
    complement: f64,
    total_var: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if !(mu.is_finite()
        && mu >= 0.0
        && complement.is_finite()
        && complement >= 0.0
        && total_var.is_finite()
        && total_var >= 0.0)
    {
        return None;
    }
    if total_var == 0.0 {
        return Some((mu, mu));
    }
    // A Beta on (0,1) with mean μ can carry variance only up to the Bernoulli
    // limit μ(1−μ); at or beyond it no Beta exists, so the moment match fails.
    let max_var = mu * complement;
    if total_var >= max_var {
        return None;
    }
    let precision = max_var / total_var - 1.0; // = a + b > 0
    let a = mu * precision;
    let b = complement * precision;
    // A law whose mass sits nearer one is read through its mirror `1 − Y ~ Beta(b, a)`,
    // whose mass sits near zero, where `beta_quantile`'s lower-tail series resolves a
    // quantile to relative precision (#2528); one minus that quantile is then exact to
    // the resolution of the support near one. Read directly, a shape `b` below the
    // resolution of one is left to `inv_beta_reg`'s absolute tolerance.
    let (q_lo, q_hi) = if complement < mu {
        (
            1.0 - beta_quantile(1.0 - p_lo, b, a),
            1.0 - beta_quantile(1.0 - p_hi, b, a),
        )
    } else {
        (beta_quantile(p_lo, a, b), beta_quantile(p_hi, a, b))
    };
    if q_lo.is_finite() && q_hi.is_finite() && q_hi >= q_lo {
        Some((q_lo, q_hi))
    } else {
        None
    }
}

/// Quantile of `IG(μ, λ)` at `p ∈ (0, 1)`: the CDF is continuous and strictly
/// increasing on `(0, ∞)`, so a geometric bracket around the mean followed by
/// geometric bisection converges to the adjacent-float pair that straddles `p`.
fn inverse_gaussian_quantile(p: f64, mu: f64, lambda: f64) -> f64 {
    if !(p > 0.0 && p < 1.0 && mu.is_finite() && mu > 0.0 && lambda.is_finite() && lambda > 0.0) {
        return f64::NAN;
    }
    let mut lo = mu;
    while inverse_gaussian_cdf(lo, mu, lambda) >= p {
        lo *= 0.5;
        if lo == 0.0 {
            return 0.0;
        }
    }
    let mut hi = mu;
    while inverse_gaussian_cdf(hi, mu, lambda) < p {
        hi *= 2.0;
        if !hi.is_finite() {
            return f64::INFINITY;
        }
    }
    loop {
        let mid = (lo * hi).sqrt();
        if mid <= lo || mid >= hi {
            return hi;
        }
        if inverse_gaussian_cdf(mid, mu, lambda) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
}

/// Equal-tailed predictive interval for a strictly-positive response modelled
/// as an inverse Gaussian whose first two moments match a point prediction:
/// mean `mu` and total predictive variance `total_var` (estimation +
/// observation noise). Moment matching fixes `λ = μ³/V`; when estimation
/// uncertainty vanishes (`total_var → φμ³`) this is the exact conditional
/// `IG(μ, 1/φ)`, and with nonzero estimation variance it widens inside the
/// inverse-Gaussian family, keeping its right skew.
///
/// Returns `None` for degenerate inputs (non-positive or non-finite mean or
/// variance) or a mis-ordered pair, in which case the caller keeps the
/// symmetric edges.
pub fn inverse_gaussian_moment_matched_interval(
    mu: f64,
    total_var: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if let Some(band) = zero_point_mass(mu, total_var) {
        return Some(band);
    }
    if !(mu.is_finite() && mu > 0.0 && total_var.is_finite() && total_var > 0.0) {
        return None;
    }
    let lambda = mu.powi(3) / total_var;
    let q_lo = inverse_gaussian_quantile(p_lo, mu, lambda);
    let q_hi = inverse_gaussian_quantile(p_hi, mu, lambda);
    if q_lo.is_finite() && q_hi.is_finite() && q_hi >= q_lo {
        Some((q_lo, q_hi))
    } else {
        None
    }
}

/// CDF of a Negative-Binomial with mean `μ ≥ 0` and dispersion `θ > 0`
/// (`Var = μ + μ²/θ`) at the integer count `k ≥ 0`:
/// `P(Y ≤ k) = I_{θ/(θ+μ)}(θ, k+1)`, the regularized incomplete beta. Increasing
/// in `k`; `P(Y ≤ 0) = (θ/(θ+μ))^θ` is the zero mass.
#[inline]
fn negative_binomial_cdf_at(k: f64, theta: f64, prob: f64) -> f64 {
    // `prob ∈ (0, 1)`; `beta_reg` requires its last argument in [0, 1].
    beta_reg(theta, k + 1.0, prob.clamp(0.0, 1.0))
}

/// `2⁵³`, the magnitude below which every integer is representable as an `f64`.
const EXACT_INTEGER_LIMIT: f64 = (1_u64 << f64::MANTISSA_DIGITS) as f64;

/// Smallest integer `k` with `cdf(k) ≥ p`, found by a geometric bracket grown
/// from `seed` followed by an integer bisection (invariant `cdf(lo) < p ≤
/// cdf(hi)`). `cdf` must be a monotone non-decreasing lower-tail CDF on the
/// non-negative integers. Returns `+∞` once the upper bracket passes `2⁵³`
/// without reaching `p`: past it consecutive integers are not representable, so
/// no bisection can name the smallest one.
///
/// Shared root finder for the discrete count quantiles (Negative-Binomial,
/// Poisson): both seed a normal approximation on their own moments and then run
/// this identical bracket-and-bisect. Callers must already have handled the
/// `cdf(0) ≥ p` zero-atom short-circuit.
fn count_quantile_bracket_bisect(cdf: impl Fn(f64) -> f64, seed: f64, p: f64) -> f64 {
    let mut lo: f64;
    let mut hi: f64;
    if cdf(seed) >= p {
        hi = seed;
        lo = 0.0;
        // Tighten `lo` upward toward `hi` so the bisection starts narrow.
        let mut step = 1.0;
        let mut cand = seed - 1.0;
        while cand > 0.0 && cdf(cand) >= p {
            hi = cand;
            step *= 2.0;
            cand = seed - step;
        }
        if cand > 0.0 {
            lo = cand; // CDF(cand) < p
        }
    } else {
        lo = seed; // CDF(seed) < p
        let mut step = 1.0;
        let mut cand = seed + 1.0;
        // CDF → 1 as k → ∞ and p < 1, so this terminates unless the root lies
        // past `2⁵³`, where the integer bisection below could no longer shrink a
        // bracket whose midpoint rounds onto one of its ends.
        while cdf(cand) < p {
            lo = cand;
            step *= 2.0;
            cand = seed + step;
            if cand > EXACT_INTEGER_LIMIT {
                return f64::INFINITY;
            }
        }
        hi = cand;
    }

    // Bisection for the smallest integer k with CDF(k) ≥ p, maintaining the
    // invariant CDF(lo) < p ≤ CDF(hi).
    while hi - lo > 1.0 {
        let mid = (lo + (hi - lo) / 2.0).floor();
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// Quantile (inverse CDF) of a Negative-Binomial with mean `μ ≥ 0` and
/// dispersion `θ > 0` at probability `p ∈ (0, 1)`: the smallest integer count
/// `k ≥ 0` with `P(Y ≤ k) ≥ p`, returned as an `f64`.
///
/// `p ≤ 0` maps to the `0` support floor and `p ≥ 1` to `+∞`; a non-finite or
/// non-positive dispersion, or a non-finite / negative mean, yields `NaN`; a
/// zero mean is the degenerate point mass at `0`.
///
/// Unlike the continuous Gamma/Beta quantiles, the NB is *discrete* with a real
/// atom at zero, so its skew-correct predictive band must come from the genuine
/// integer quantiles — a moment-matched *continuous* surrogate (e.g. a Gamma)
/// has no zero atom and grossly over-covers the lower tail on low-mean counts
/// (#1193). A normal-approximation seed brackets the root, then an exact
/// bisection on the incomplete-beta CDF finds the smallest qualifying integer.
pub(crate) fn negative_binomial_quantile(p: f64, mu: f64, theta: f64) -> f64 {
    if !(mu.is_finite() && mu >= 0.0 && theta.is_finite() && theta > 0.0) {
        return f64::NAN;
    }
    if !p.is_finite() || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if mu == 0.0 {
        return 0.0;
    }
    let prob = theta / (theta + mu); // P(success) ∈ (0, 1); mean = θ(1−prob)/prob = μ
    let cdf = |k: f64| negative_binomial_cdf_at(k, theta, prob);

    // The zero atom already covers the requested lower-tail mass on low-mean
    // counts (the common right-skewed case), so short-circuit before bracketing.
    if cdf(0.0) >= p {
        return 0.0;
    }

    // Normal-approximation seed on the NB moments, floored into the support.
    let var = mu + mu * mu / theta;
    let z = standard_normal_quantile(p).unwrap_or(0.0);
    let seed = (mu + z * var.sqrt()).floor().max(1.0);

    // Bracket the smallest integer with CDF ≥ p: `lo` always satisfies
    // CDF(lo) < p (starts at 0, which failed the short-circuit) and `hi`
    // satisfies CDF(hi) ≥ p. Grow geometrically from the seed in whichever
    // direction is needed.
    count_quantile_bracket_bisect(&cdf, seed, p)
}

/// Equal-tailed predictive interval for a Negative-Binomial count response whose
/// conditional law has mean `mu > 0` and dispersion `theta > 0`, widened for
/// estimation uncertainty to a total predictive variance `total_var`
/// (estimation + observation noise). Returns the pair of integer NB quantiles at
/// lower-tail probabilities `p_lo < p_hi` — the skew-correct, zero-atom-aware
/// replacement for a symmetric `mu ± z·σ` band, which on right-skewed counts
/// sits below the true upper quantile and under-covers the upper tail (#1193).
///
/// Estimation uncertainty is folded in through an *effective dispersion*: an NB
/// with mean `μ` has variance `μ + μ²/θ`, so the `θ_eff` matching the inflated
/// total variance solves `μ + μ²/θ_eff = total_var`, i.e.
/// `θ_eff = μ² / (total_var − μ)`. When estimation uncertainty vanishes
/// (`total_var → μ + μ²/θ`) this is *exact*: `θ_eff → θ`, recovering the
/// conditional `NB(μ, θ)`. With nonzero estimation variance `θ_eff < θ` widens
/// the band — the minimal skew-correct widening that stays inside the NB family.
///
/// Returns `None` for degenerate inputs (non-positive mean / variance,
/// non-finite), or a numerically mis-ordered pair, in which case the caller
/// falls back to the symmetric edges.
pub fn negative_binomial_moment_matched_interval(
    mu: f64,
    theta: f64,
    total_var: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if let Some(band) = zero_point_mass(mu, total_var) {
        return Some(band);
    }
    CountPredictive::negative_binomial(mu, theta, total_var)?.band(p_lo, p_hi)
}

/// Mass the Negative-Binomial predictive of
/// [`negative_binomial_moment_matched_interval`] — the same law, with the same
/// effective dispersion — puts on the integer set `{lower, …, upper}`.
///
/// A discrete law's quantile set cannot carry an arbitrary mass: the band read at
/// tail masses `p_lo < p_hi` carries `F(upper) − F(lower − 1) ≥ p_hi − p_lo`, the
/// excess being the atoms at its edges. That excess is the coverage the band
/// promises, so a coverage audit targets this content rather than `p_hi − p_lo`.
/// `None` for the moments the interval also refuses.
pub fn negative_binomial_moment_matched_content(
    mu: f64,
    theta: f64,
    total_var: f64,
    lower: f64,
    upper: f64,
) -> Option<f64> {
    if let Some(content) = zero_point_mass_content(mu, total_var, lower, upper) {
        return Some(content);
    }
    Some(CountPredictive::negative_binomial(mu, theta, total_var)?.content(lower, upper))
}

/// The moment-matched count predictive a count band and its content are both read
/// from: the exact conditional Poisson, or the Gamma–Poisson Negative-Binomial
/// whose dispersion carries the estimation variance. One constructor per family
/// fixes the law, so the edges and the mass reported for them cannot come from
/// two different laws.
#[derive(Clone, Copy, Debug)]
enum CountPredictive {
    Poisson { mu: f64 },
    NegativeBinomial { mu: f64, theta: f64 },
}

impl CountPredictive {
    /// Negative-Binomial predictive with mean `mu` and total variance `total_var`:
    /// the effective dispersion `θ_eff = μ²/(total_var − μ)` matches the inflated
    /// variance (see [`negative_binomial_moment_matched_interval`]).
    fn negative_binomial(mu: f64, theta: f64, total_var: f64) -> Option<Self> {
        if !(mu.is_finite()
            && mu > 0.0
            && theta.is_finite()
            && theta > 0.0
            && total_var.is_finite()
            && total_var > 0.0)
        {
            return None;
        }
        // `total_var = SE(μ̂)² + (μ + μ²/θ) > μ` always, so the excess is positive;
        // fall back to the nominal dispersion only if a degenerate caller breaks it.
        let excess = total_var - mu;
        let theta_eff = if excess > 0.0 {
            mu * mu / excess
        } else {
            theta
        };
        Some(Self::NegativeBinomial {
            mu,
            theta: theta_eff,
        })
    }

    /// Poisson-response predictive with mean `mu` and total variance
    /// `total_var ≥ μ`: the Gamma–Poisson NB carrying the excess, or the exact
    /// Poisson once that excess is below what the NB can resolve (see
    /// [`poisson_moment_matched_interval`]).
    fn poisson(mu: f64, total_var: f64) -> Option<Self> {
        if !(mu.is_finite() && mu > 0.0 && total_var.is_finite() && total_var > 0.0) {
            return None;
        }
        // Estimation uncertainty inflates the count variance beyond the Poisson
        // floor `Var(Y|μ) = μ`; the excess is the (approximate) sampling variance
        // of `μ̂`. A `total_var` below `μ` is degenerate (a caller broke the
        // contract).
        let excess = total_var - mu;
        if excess < 0.0 {
            return None;
        }
        // The NB surrogate reaches the conditional Poisson only through
        // `prob = θ/(θ+μ)`, and the incomplete beta `I_prob(θ, k+1)` behind
        // `negative_binomial_quantile` forms its complement `μ/(θ+μ)` by rounding,
        // so the NB CDF carries relative error ~`u·θ/μ` while the widening it adds
        // is ~`μ/θ`. The two cross at `θ = μ/√u`: past it the exact Poisson
        // quantile is the more accurate one, below it the NB widening is genuine
        // (#2469).
        let poisson_limit = mu / gam_linalg::roundoff::UNIT_ROUNDOFF.sqrt();
        let theta_eff = if excess > 0.0 {
            mu * mu / excess
        } else {
            f64::INFINITY
        };
        Some(if theta_eff > poisson_limit {
            Self::Poisson { mu }
        } else {
            Self::NegativeBinomial {
                mu,
                theta: theta_eff,
            }
        })
    }

    fn quantile(self, p: f64) -> f64 {
        match self {
            Self::Poisson { mu } => poisson_quantile(p, mu),
            Self::NegativeBinomial { mu, theta } => negative_binomial_quantile(p, mu, theta),
        }
    }

    /// `P(Y ≤ k)` at an integer `k`; zero below the support.
    fn cdf(self, k: f64) -> f64 {
        if k < 0.0 {
            return 0.0;
        }
        match self {
            Self::Poisson { mu } => poisson_cdf_at(k, mu),
            Self::NegativeBinomial { mu, theta } => {
                negative_binomial_cdf_at(k, theta, theta / (theta + mu))
            }
        }
    }

    /// Equal-tailed band `[F⁻¹(p_lo), F⁻¹(p_hi)]`, or `None` when the quantiles
    /// come out non-finite or mis-ordered.
    fn band(self, p_lo: f64, p_hi: f64) -> Option<(f64, f64)> {
        let q_lo = self.quantile(p_lo);
        let q_hi = self.quantile(p_hi);
        (q_lo.is_finite() && q_hi.is_finite() && q_hi >= q_lo).then_some((q_lo, q_hi))
    }

    /// Mass on the integer set `{lower, …, upper}`: `F(upper) − F(lower − 1)`.
    fn content(self, lower: f64, upper: f64) -> f64 {
        self.cdf(upper) - self.cdf(lower - 1.0)
    }
}

/// CDF of a Poisson with mean `mu ≥ 0` at the integer count `k ≥ 0`:
/// `P(Y ≤ k) = Q(k+1, μ)`, the regularized *upper* incomplete gamma (the standard
/// Poisson↔gamma identity). Increasing in `k`; `P(Y ≤ 0) = e^{−μ}` is the zero mass.
#[inline]
fn poisson_cdf_at(k: f64, mu: f64) -> f64 {
    // P(Y ≤ k) = Q(k+1, μ) is the *upper* incomplete gamma, so take it from the
    // pair evaluator, which computes it directly by continued fraction. Forming
    // it as `1 − P` instead destroys it whenever the count sits far below the
    // mean and the answer is small: at μ = 50, k = 0 the true mass is 1.93e-22 and
    // `1 − P` returns exactly 0, because `P` has already rounded to 1.
    regularized_incomplete_gamma_pair(k + 1.0, mu)
        .1
        .clamp(0.0, 1.0)
}

/// Quantile (inverse CDF) of a Poisson with mean `mu ≥ 0` at probability
/// `p ∈ (0, 1)`: the smallest integer count `k ≥ 0` with `P(Y ≤ k) ≥ p`,
/// returned as an `f64`.
///
/// `p ≤ 0` maps to the `0` support floor and `p ≥ 1` to `+∞`; a non-finite or
/// negative mean yields `NaN`; a zero mean is the degenerate point mass at `0`.
///
/// Like the Negative-Binomial, the Poisson is *discrete* with a real atom at
/// zero, so its skew-correct predictive band must come from the genuine integer
/// quantiles — a symmetric `μ ± z·σ` band sits below the true upper quantile on
/// low-rate counts and under-covers the upper tail (the #817 defect, Poisson
/// sibling of #1193). A normal-approximation seed brackets the root, then an
/// exact bisection on the gamma-tail CDF finds the smallest qualifying integer.
pub(crate) fn poisson_quantile(p: f64, mu: f64) -> f64 {
    if !(mu.is_finite() && mu >= 0.0) {
        return f64::NAN;
    }
    if !p.is_finite() || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if mu == 0.0 {
        return 0.0;
    }
    let cdf = |k: f64| poisson_cdf_at(k, mu);

    // The zero atom already covers the requested lower-tail mass on low-rate
    // counts (the common right-skewed case), so short-circuit before bracketing.
    if cdf(0.0) >= p {
        return 0.0;
    }

    // Normal-approximation seed on the Poisson moments (Var = μ), floored into
    // the support.
    let z = standard_normal_quantile(p).unwrap_or(0.0);
    let seed = (mu + z * mu.sqrt()).floor().max(1.0);

    // Bracket the smallest integer with CDF ≥ p: `lo` always satisfies
    // CDF(lo) < p (starts at 0, which failed the short-circuit) and `hi`
    // satisfies CDF(hi) ≥ p. Grow geometrically from the seed in whichever
    // direction is needed.
    count_quantile_bracket_bisect(&cdf, seed, p)
}

/// Equal-tailed predictive interval for a Poisson count response whose
/// conditional law has mean `mu > 0` (so `Var(Y|μ) = μ`), widened for estimation
/// uncertainty to a total predictive variance `total_var ≥ μ` (estimation +
/// observation noise). Returns the pair of integer quantiles at lower-tail
/// probabilities `p_lo < p_hi` — the skew-correct, zero-atom-aware replacement
/// for a symmetric `mu ± z·σ` band, which on low-rate counts sits below the true
/// upper quantile and under-covers the upper tail (the #817 defect, Poisson
/// sibling of #1193).
///
/// A pure Poisson has no free dispersion parameter to absorb estimation
/// uncertainty, so the widening is carried by the *conjugate over-dispersed count
/// law*: if the point estimate `μ̂` carries (approximately) a Gamma sampling
/// uncertainty with mean `μ` and variance `SE(μ̂)² = total_var − μ`, the posterior
/// predictive for a *new* Poisson draw is exactly a Negative-Binomial — the
/// Gamma–Poisson mixture — with mean `μ` and dispersion `θ_eff = μ² / (total_var − μ)`
/// (matching the inflated variance `μ + μ²/θ_eff = total_var`). As estimation
/// uncertainty vanishes (`total_var → μ`, `θ_eff → ∞`) the NB collapses to the
/// *exact* conditional Poisson, which is then used directly — both because it is
/// the correct limit and because an NB with `θ → ∞` is numerically degenerate.
/// The two regimes agree (both are integer quantiles that coincide once `θ_eff`
/// is large), so the switch introduces no discontinuity in the emitted edge.
///
/// Returns `None` for degenerate inputs (non-positive mean, non-finite, or a
/// total variance below the Poisson floor `μ`), or a numerically mis-ordered
/// pair, in which case the caller falls back to the symmetric edges.
pub fn poisson_moment_matched_interval(
    mu: f64,
    total_var: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if let Some(band) = zero_point_mass(mu, total_var) {
        return Some(band);
    }
    CountPredictive::poisson(mu, total_var)?.band(p_lo, p_hi)
}

/// Mass the count predictive of [`poisson_moment_matched_interval`] — the same
/// law, in the same regime — puts on the integer set `{lower, …, upper}`: at
/// least the `p_hi − p_lo` the band was read at, by the atoms at its edges (see
/// [`negative_binomial_moment_matched_content`]). `None` for the moments the
/// interval also refuses.
pub fn poisson_moment_matched_content(
    mu: f64,
    total_var: f64,
    lower: f64,
    upper: f64,
) -> Option<f64> {
    if let Some(content) = zero_point_mass_content(mu, total_var, lower, upper) {
        return Some(content);
    }
    Some(CountPredictive::poisson(mu, total_var)?.content(lower, upper))
}

/// Equal-tailed predictive interval for the observed proportion `K/m` of a new
/// binomial row with `m` trials, whose success probability `p` has posterior
/// mean `mu`, complement `complement = E[1 − p]` (carried separately, as in
/// [`beta_moment_matched_interval`]) and variance `mean_variance`. Returns the
/// pair of predictive quantiles of `K/m` at lower-tail probabilities
/// `p_lo < p_hi`, each a multiple of `1/m`.
///
/// The conditional law is `K | p ~ Binomial(m, p)`. Estimation uncertainty in
/// `p` is carried by the Beta whose mean and variance match its posterior
/// moments, precision `a + b = μ(1−μ)/v − 1`, `a = μ(a+b)`, `b = (1−μ)(a+b)`,
/// and the Beta mixture of binomials is exactly the beta-binomial
/// `K ~ BetaBinomial(m, a, b)`. Its variance, divided by `m²`, is
/// `μ(1−μ)/m + v(m−1)/m`, the conditional binomial spread plus the estimation
/// variance each trial shares. With `v = 0` it is the exact
/// `Binomial(m, μ)`, and with `m = 1` it is `Bernoulli(μ)` for every `v`,
/// because a single trial sees only the mean of `p`.
///
/// The beta-binomial CDF has no closed form, and its log pmf through log-gamma
/// differences cancels catastrophically once the precision `a + b` is large, the
/// regime of a well-determined `p`. The pmf is therefore walked in log space by
/// its exact term ratios
/// `pmf(k+1)/pmf(k) = (m−k)/(k+1) · (a+k)/(b+m−k−1)`, whose `a + b → ∞` limit
/// is the binomial ratio `(m−k)/(k+1) · μ/(1−μ)`: an unnormalized sweep over
/// `0..=m` gives the normalizing mass, an ascending sweep the lower edge and a
/// descending sweep the upper edge, each tail summed from its own end so neither
/// is formed as a complement. The cost is `O(m)` per row.
///
/// Returns `None` when the inputs are not the moments of a probability (negative
/// or non-finite, or a variance at or past the Bernoulli ceiling `μ(1−μ)`,
/// which no Beta reaches), when `trials` is not an integer in `[1, 2⁵³]`, or when
/// a term ratio leaves the finite positive range (a mean within underflow of a
/// support edge while the other side carries mass).
pub fn binomial_proportion_interval(
    mu: f64,
    complement: f64,
    trials: f64,
    mean_variance: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if !(mu.is_finite()
        && mu >= 0.0
        && complement.is_finite()
        && complement >= 0.0
        && mean_variance.is_finite()
        && mean_variance >= 0.0
        && is_binomial_trial_count(trials))
    {
        return None;
    }
    let max_var = mu * complement;
    if mean_variance > 0.0 && mean_variance >= max_var {
        return None;
    }
    // With no success probability the law is the point mass at zero, and with
    // no failure probability (which admits no spread) the point mass at one.
    if mu == 0.0 {
        return Some((0.0, 0.0));
    }
    if complement == 0.0 {
        return Some((1.0, 1.0));
    }
    let m = trials;
    // Beta shapes of the posterior of `p`; `None` is the exact binomial, the
    // `a + b → ∞` limit, which a variance too small to resolve against the
    // Bernoulli ceiling also reaches.
    let shapes = (mean_variance > 0.0)
        .then(|| max_var / mean_variance - 1.0)
        .filter(|precision| precision.is_finite())
        .map(|precision| (mu * precision, complement * precision));
    let log_ratio = |k: f64| -> f64 {
        let counts = ((m - k) / (k + 1.0)).ln();
        match shapes {
            Some((a, b)) => counts + ((a + k) / (b + (m - k - 1.0))).ln(),
            None => counts + (mu / complement).ln(),
        }
    };

    // Unnormalized log weights `ℓ(0) = 0`, `ℓ(k+1) = ℓ(k) + ln ratio(k)`, and
    // their streaming log-sum-exp.
    let mut log_weight = 0.0_f64;
    let mut log_max = 0.0_f64;
    let mut scaled_mass = 1.0_f64;
    let mut k = 0.0;
    while k < m {
        let step = log_ratio(k);
        if !step.is_finite() {
            return None;
        }
        log_weight += step;
        if log_weight > log_max {
            scaled_mass = scaled_mass * (log_max - log_weight).exp() + 1.0;
            log_max = log_weight;
        } else {
            scaled_mass += (log_weight - log_max).exp();
        }
        k += 1.0;
    }
    let log_mass = log_max + scaled_mass.ln();
    let log_weight_at_m = log_weight;

    // Lower edge: the smallest `k` with `P(K ≤ k) ≥ p_lo`.
    let lower = if p_lo <= 0.0 {
        0.0
    } else {
        let mut cumulative = 0.0;
        let mut log_weight = 0.0;
        let mut k = 0.0;
        loop {
            cumulative += (log_weight - log_mass).exp();
            if cumulative >= p_lo || k >= m {
                break k;
            }
            log_weight += log_ratio(k);
            k += 1.0;
        }
    };
    // Upper edge: the smallest `k` with `P(K > k) ≤ 1 − p_hi`, reading the upper
    // tail `P(K ≥ k)` down from `k = m`.
    let upper = if p_hi >= 1.0 {
        m
    } else {
        let tail_limit = 1.0 - p_hi;
        let mut tail = 0.0;
        let mut log_weight = log_weight_at_m;
        let mut k = m;
        loop {
            tail += (log_weight - log_mass).exp();
            if tail > tail_limit || k <= 0.0 {
                break k;
            }
            log_weight -= log_ratio(k - 1.0);
            k -= 1.0;
        }
    };
    (upper >= lower).then_some((lower / m, upper / m))
}

/// CDF of a Tweedie compound Poisson–Gamma response (power `1 < p < 2`) with
/// mean `mu > 0` and dispersion `phi > 0` at `y ≥ 0`:
/// `P(Y ≤ y) = e^{−λ} + Σ_{k≥1} Poisson(k; λ)·GammaCDF(y; kα, γ)`, the mixture of
/// a point mass at zero (no jumps) and `k` i.i.d. Gamma jumps. The Tweedie
/// parameters map to `λ = μ^{2−p} / (φ(2−p))` (Poisson mean number of jumps),
/// Gamma jump shape `α = (2−p)/(p−1)` and scale `γ = φ(p−1)μ^{p−1}`, which
/// reproduce `E[Y] = μ` and `Var(Y) = φμ^p`.
///
/// The zero atom `e^{−λ}` is returned directly at `y = 0`. For `y > 0` the
/// Poisson weights are accumulated in log-space and the series is truncated once
/// the remaining Poisson mass beyond the current term is negligible — the Gamma
/// CDF factor is ≤ 1, so the unsummed tail is bounded by the Poisson survival.
#[inline]
fn tweedie_cdf_at(y: f64, mu: f64, phi: f64, power: f64) -> f64 {
    if !(y.is_finite() && y >= 0.0) {
        return f64::NAN;
    }
    let lambda = mu.powf(2.0 - power) / (phi * (2.0 - power));
    let alpha = (2.0 - power) / (power - 1.0);
    let scale = phi * (power - 1.0) * mu.powf(power - 1.0);
    let zero_mass = (-lambda).exp();
    if y <= 0.0 {
        return zero_mass;
    }
    let x = y / scale; // unit-scale Gamma argument
    // Poisson(k; λ) weights via a log-space recurrence: w_k = w_{k-1}·λ/k.
    // Sum k ≥ 1 only; the k = 0 term contributes the zero atom (GammaCDF = 1 at
    // any y > 0 for shape 0 is the degenerate point mass already in `zero_mass`).
    let mut acc = zero_mass; // P(Y ≤ y) includes the no-jump mass (Y = 0 ≤ y)
    let mut ln_w = -lambda; // ln Poisson(0; λ)
    let mut remaining = 1.0 - zero_mass; // Poisson mass still unaccounted for (k ≥ 1)
    // Magnitude sum of the running subtraction `(1 − e^{−λ}) − Σ w_k`.
    let mut remaining_magnitude = remaining;
    for k in 1_usize.. {
        ln_w += lambda.ln() - (k as f64).ln();
        let w = ln_w.exp();
        remaining -= w;
        remaining_magnitude += w;
        // GammaCDF(y; kα, γ) = P(kα, y/γ) on the unit scale.
        acc += w * regularized_lower_gamma(alpha * k as f64, x);
        // Past the Poisson mode the weights decrease and the unsummed tail is at
        // most the unaccounted mass; once that sits inside the rounding band of
        // the subtraction producing it, the tail is indistinguishable from zero.
        if k as f64 > lambda
            && remaining
                <= gam_linalg::roundoff::accumulation_growth(k + 1) * remaining_magnitude
        {
            break;
        }
    }
    acc.clamp(0.0, 1.0)
}

/// Quantile (inverse CDF) of a Tweedie compound Poisson–Gamma response
/// (power `1 < p < 2`) with mean `mu > 0` and dispersion `phi > 0` at
/// probability `q ∈ (0, 1)`: the value `y ≥ 0` with `P(Y ≤ y) = q`.
///
/// `q ≤ 0` maps to the `0` support floor and `q ≥ 1` to `+∞`. If the requested
/// lower-tail probability is at or below the zero atom `e^{−λ}` the quantile is
/// exactly `0` (the common right-skewed lower-tail case). Otherwise a normal seed
/// on the Tweedie moments brackets the root, which is then refined by bisection
/// on `tweedie_cdf_at` — the continuous part above the atom is strictly
/// increasing, so the bracket converges.
pub(crate) fn tweedie_quantile(q: f64, mu: f64, phi: f64, power: f64) -> f64 {
    if !(mu.is_finite()
        && mu > 0.0
        && phi.is_finite()
        && phi > 0.0
        && power.is_finite()
        && power > 1.0
        && power < 2.0)
    {
        return f64::NAN;
    }
    if !q.is_finite() || q <= 0.0 {
        return 0.0;
    }
    if q >= 1.0 {
        return f64::INFINITY;
    }
    let lambda = mu.powf(2.0 - power) / (phi * (2.0 - power));
    let zero_mass = (-lambda).exp();
    // The zero atom carries the lower-tail mass: q at or below it ⇒ quantile 0.
    if q <= zero_mass {
        return 0.0;
    }

    // Normal-approximation seed on the Tweedie moments, then geometric bracketing.
    // Any positive edge brackets the root once doubled past it, so a seed at or
    // below the support floor starts from the mean instead.
    let var = phi * mu.powf(power);
    let z = standard_normal_quantile(q).unwrap_or(0.0);
    let seed = mu + z * var.sqrt();
    let mut hi = if seed > 0.0 { seed } else { mu };
    let cdf = |y: f64| tweedie_cdf_at(y, mu, phi, power);

    // Grow `hi` until it covers `q`; `lo` stays below it. CDF → 1 as y → ∞, so
    // only an edge doubled past the largest finite float fails to cover `q`.
    let mut lo = 0.0_f64;
    while cdf(hi) < q {
        lo = hi;
        hi *= 2.0;
        if !hi.is_finite() {
            return f64::INFINITY;
        }
    }

    // Bisection on the strictly-increasing continuous part above the atom.
    // Bisect until no representable `y` lies strictly between the bracket's
    // ends; every step halves the bracket, so the float grid ends the loop.
    loop {
        let mid = 0.5 * (lo + hi);
        if !(lo < mid && mid < hi) {
            break;
        }
        if cdf(mid) < q {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Equal-tailed predictive interval for a Tweedie compound Poisson–Gamma
/// response (power `1 < p < 2`) whose conditional law has mean `mu > 0` and
/// dispersion `phi > 0`, widened for estimation uncertainty to a total
/// predictive variance `total_var` (estimation + observation noise). Returns the
/// pair of Tweedie quantiles at lower-tail probabilities `p_lo < p_hi` — the
/// skew-correct, zero-atom-aware replacement for a symmetric `mu ± z·σ` band,
/// which on a right-skewed Tweedie sits below the true upper quantile and
/// under-covers the upper tail (the #817 defect, Tweedie sibling of #1193).
///
/// Estimation uncertainty is folded in through an *effective dispersion*: a
/// Tweedie with mean `μ` has variance `φμ^p`, so the `φ_eff` matching the
/// inflated total variance solves `φ_eff·μ^p = total_var`, i.e.
/// `φ_eff = total_var / μ^p`. When estimation uncertainty vanishes
/// (`total_var → φμ^p`) this is *exact*: `φ_eff → φ`, recovering the conditional
/// Tweedie. With nonzero estimation variance `φ_eff > φ` widens the band inside
/// the Tweedie family — the minimal skew-correct widening. Unlike a moment-
/// matched Gamma surrogate, this keeps the genuine zero atom, so it does not
/// over-cover the lower tail on low-mean rows (#1193).
///
/// Returns `None` for degenerate inputs (non-positive mean / variance,
/// non-finite, power outside `(1, 2)`) or a mis-ordered pair, in which case the
/// caller falls back to the symmetric edges.
pub fn tweedie_moment_matched_interval(
    mu: f64,
    phi: f64,
    power: f64,
    total_var: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if let Some(band) = zero_point_mass(mu, total_var) {
        return Some(band);
    }
    let phi_eff = tweedie_effective_dispersion(mu, phi, power, total_var)?;
    let q_lo = tweedie_quantile(p_lo, mu, phi_eff, power);
    let q_hi = tweedie_quantile(p_hi, mu, phi_eff, power);
    if q_lo.is_finite() && q_hi.is_finite() && q_hi >= q_lo {
        Some((q_lo, q_hi))
    } else {
        None
    }
}

/// Effective dispersion `φ_eff = total_var / μ^p` of the Tweedie predictive with
/// mean `mu` and total variance `total_var`, the one law both
/// [`tweedie_moment_matched_interval`] and [`tweedie_moment_matched_content`]
/// read. `None` for degenerate inputs.
fn tweedie_effective_dispersion(mu: f64, phi: f64, power: f64, total_var: f64) -> Option<f64> {
    if !(mu.is_finite()
        && mu > 0.0
        && phi.is_finite()
        && phi > 0.0
        && power.is_finite()
        && power > 1.0
        && power < 2.0
        && total_var.is_finite()
        && total_var > 0.0)
    {
        return None;
    }
    let phi_eff = total_var / mu.powf(power);
    (phi_eff.is_finite() && phi_eff > 0.0).then_some(phi_eff)
}

/// Mass the Tweedie predictive of [`tweedie_moment_matched_interval`] puts on
/// `[lower, upper]`: `F(upper) − P(Y < lower)`. Above zero the law is continuous,
/// so `P(Y < lower) = F(lower)`; a band whose lower edge is the support floor
/// holds the zero atom whole. The content is `p_hi − p_lo` wherever both edges sit
/// on the continuous part and exceeds it by the atom's surplus when the band
/// starts at zero. `None` for the moments the interval also refuses.
pub fn tweedie_moment_matched_content(
    mu: f64,
    phi: f64,
    power: f64,
    total_var: f64,
    lower: f64,
    upper: f64,
) -> Option<f64> {
    if let Some(content) = zero_point_mass_content(mu, total_var, lower, upper) {
        return Some(content);
    }
    let phi_eff = tweedie_effective_dispersion(mu, phi, power, total_var)?;
    if !(upper.is_finite() && upper >= 0.0) {
        return None;
    }
    let below = if lower > 0.0 {
        tweedie_cdf_at(lower, mu, phi_eff, power)
    } else {
        0.0
    };
    Some(tweedie_cdf_at(upper, mu, phi_eff, power) - below)
}

/// Regularized lower incomplete gamma `P(a, x)` alone — see
/// [`regularized_incomplete_gamma_pair`], which computes whichever tail is small
/// directly and is what callers wanting `Q` must use.
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    regularized_incomplete_gamma_pair(a, x).0
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_gaussian_quantile_inverts_the_cdf() {
        // (p, μ, λ, scipy.stats.invgauss(μ/λ, scale=λ).ppf(p)).
        let cases = [
            (0.025, 1.0, 1.0, 0.149_804_321_724_041_93),
            (0.975, 1.0, 1.0, 3.771_837_954_732_6),
            (0.025, 2.0, 400.0, 1.737_184_562_740_362_1),
            (0.975, 3.0, 0.5, 21.436_074_647_884_915),
        ];
        for (p, mu, lambda, reference) in cases {
            let q = inverse_gaussian_quantile(p, mu, lambda);
            assert!(
                ((q - reference) / reference).abs() <= 1e-12,
                "Q({p}; {mu}, {lambda}) = {q}, reference {reference}"
            );
        }
        assert!(inverse_gaussian_quantile(0.0, 1.0, 1.0).is_nan());
        assert!(inverse_gaussian_quantile(0.5, -1.0, 1.0).is_nan());
    }

    #[test]
    fn inverse_gaussian_interval_is_the_conditional_law_at_zero_estimation_variance() {
        // V = φμ³ is the conditional variance, so the moment-matched law is
        // IG(μ, 1/φ) exactly: μ = 1, φ = 1 reproduces the λ = 1 quantiles.
        let (lo, hi) = inverse_gaussian_moment_matched_interval(1.0, 1.0, 0.025, 0.975)
            .expect("finite interval");
        assert!((lo - 0.149_804_321_724_041_93).abs() <= 1e-12);
        assert!((hi - 3.771_837_954_732_6).abs() <= 1e-12);
        // Right skew: the upper edge sits further from the mean than the lower.
        assert!(hi - 1.0 > 1.0 - lo);
        assert!(inverse_gaussian_moment_matched_interval(1.0, 0.0, 0.025, 0.975).is_none());
    }

    #[test]
    fn signed_log_sum_exp_propagates_positive_infinities() {
        // A single +∞ positive-sign term dominates ⇒ S = +∞ ⇒ (+∞, +1).
        let (lm, s) = signed_log_sum_exp(&[f64::INFINITY], &[1.0]);
        assert_eq!(lm, f64::INFINITY);
        assert_eq!(s, 1.0);

        // A single +∞ negative-sign term ⇒ S = −∞, encoded as (+∞, −1).
        let (lm, s) = signed_log_sum_exp(&[f64::INFINITY], &[-1.0]);
        assert_eq!(lm, f64::INFINITY);
        assert_eq!(s, -1.0);

        // +∞ on both signs ⇒ indeterminate +∞ − ∞ ⇒ (NaN, 0).
        let (lm, s) = signed_log_sum_exp(&[f64::INFINITY, f64::INFINITY], &[1.0, -1.0]);
        assert!(lm.is_nan());
        assert_eq!(s, 0.0);

        // A finite positive term alongside a +∞ positive term still gives +∞.
        let (lm, s) = signed_log_sum_exp(&[0.0, f64::INFINITY], &[1.0, 1.0]);
        assert_eq!(lm, f64::INFINITY);
        assert_eq!(s, 1.0);

        // −∞ log-magnitudes are exp(−∞)=0 and must be dropped: mixing a finite
        // term with a −∞ term reproduces the lone finite term unchanged.
        let (lm, s) = signed_log_sum_exp(&[2.0, f64::NEG_INFINITY], &[1.0, -1.0]);
        assert!((lm - 2.0).abs() < 1e-12);
        assert_eq!(s, 1.0);

        // Finite sanity check: exp(ln 3) − exp(ln 1) = 2 ⇒ (ln 2, +1).
        let (lm, s) = signed_log_sum_exp(&[3.0_f64.ln(), 1.0_f64.ln()], &[1.0, -1.0]);
        assert!((lm - 2.0_f64.ln()).abs() < 1e-12);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn gamma_quantile_matches_known_reference_values() {
        // Reference quantiles for unit-scale Gamma(shape=a) from the regularized
        // lower incomplete gamma inverse (cross-checked against scipy
        // `gamma.ppf(p, a)` to ~1e-6). Spanning a < 1, a = 1 (exponential), and
        // a ≫ 1 exercises every initial-estimate / density branch.
        let cases: [(f64, f64, f64); 9] = [
            // (p, shape a, expected unit-scale quantile)
            (0.025, 4.0, 1.089_865_4),
            (0.5, 4.0, 3.672_060_4),
            (0.975, 4.0, 8.767_273_4),
            (0.025, 1.0, 0.025_317_8), // Exp(1): -ln(1-p)
            (0.975, 1.0, 3.688_879_4),
            (0.5, 0.5, 0.227_468_2),
            (0.99, 0.5, 3.317_448_3),
            (0.025, 50.0, 37.110_963_7),
            (0.975, 50.0, 64.780_598_6),
        ];
        for (p, a, expected) in cases {
            let got = gamma_quantile(p, a, 1.0);
            let rel = (got - expected).abs() / expected.max(1e-12);
            assert!(
                rel < 1e-4,
                "gamma_quantile(p={p}, a={a}) = {got}, expected ≈ {expected} (rel err {rel})"
            );
        }
    }

    #[test]
    fn gamma_quantile_handles_extreme_lower_tail_for_shape_two() {
        let got = gamma_quantile(1.0e-300, 2.0, 1.0);
        let expected = 1.414_213_562_373_095_1e-150;
        let rel = (got - expected).abs() / expected;
        assert!(
            rel < 1.0e-6,
            "gamma_quantile(1e-300, 2, 1) = {got}, expected {expected} (rel err {rel})"
        );
    }

    #[test]
    fn gamma_quantile_round_trips_extreme_lower_tail_for_shape_above_one() {
        for &a in &[1.5_f64, 2.0, 5.0, 20.0] {
            for &p in &[1.0e-300_f64, 1.0e-100, 1.0e-12, 1.0e-3, 0.5, 0.999] {
                let x = gamma_quantile(p, a, 1.0);
                assert!(
                    x.is_finite() && x >= 0.0,
                    "non-finite quantile a={a} p={p}: {x}"
                );
                let recovered = regularized_lower_gamma(a, x);
                let rel = (recovered - p).abs() / p;
                assert!(
                    rel < 1.0e-6,
                    "round-trip failed a={a} p={p}: q={x}, P(a,q)={recovered}, rel err {rel}"
                );
            }
        }
    }

    #[test]
    fn gamma_quantile_is_consistent_with_the_cdf_round_trip() {
        // The inverse must invert the CDF: P(a, Q(p; a)) = p. Verify across a
        // grid of shapes and probabilities using statrs `gamma_lr` as the CDF.
        use statrs::function::gamma::gamma_lr;
        for &a in &[0.3_f64, 0.75, 1.0, 2.5, 10.0, 80.0] {
            for &p in &[0.001_f64, 0.01, 0.025, 0.25, 0.5, 0.75, 0.975, 0.99, 0.999] {
                let x = gamma_quantile(p, a, 1.0);
                assert!(
                    x.is_finite() && x > 0.0,
                    "non-finite quantile a={a} p={p}: {x}"
                );
                let recovered = gamma_lr(a, x);
                assert!(
                    (recovered - p).abs() < 1e-6,
                    "CDF round-trip failed a={a} p={p}: P(a, {x}) = {recovered}"
                );
            }
        }
    }

    #[test]
    fn incomplete_gamma_quantile_keeps_its_digits_in_the_upper_tail() {
        // The Halley residual used to be `P(a, x) - p`. For `p > 1/2` that is a
        // subtraction of two quantities of size 1, so its absolute error is one
        // ulp of 1 no matter how close the iterate is; the step is `err / dens`
        // and `dens ~ 1 - p` out there, giving a relative error in the returned
        // quantile of `eps / ((1 - p) * x)` -- a defect that grows by a decade for
        // every decade of tail mass. Measured against these same references, the
        // old residual returned (worst over shapes) 1.3e-12 at `1 - p = 1e-6`,
        // 2.1e-9 at 1e-9, 1.3e-6 at 1e-12 and 7.8e-4 at 1e-15.
        //
        // Bar: the references below are correctly rounded doubles (half an ulp).
        // The residual `(1 - p) - Q(a, x)` is exact in its first term (Sterbenz,
        // `p >= 1/2`) and carries a few ulp in `Q`. A relative error `d` in the
        // residual moves the root by `d * q / (x * f(x))`, and that amplification
        // is at most 0.053 over this table -- the inversion *contracts* error in
        // the upper tail -- so a handful of ulp is the honest ceiling. Worst
        // measured over the table is 0.8 ulp, leaving 5x headroom.
        const ROWS: [(f64, f64, f64); 15] = [
            (0.5, 0.999999999, 18.66244655325936),
            (0.5, 0.999999999999, 25.422085666224586),
            (0.5, 0.999999999999999, 32.21601948468177),
            (1.0, 0.999999999, 20.723265865228342),
            (1.0, 0.999999999999, 27.63104323789336),
            (1.0, 0.999999999999999, 34.53957599234088),
            (2.0, 0.999999999, 23.939727895037286),
            (2.0, 0.999999999999, 31.099896029053795),
            (2.0, 0.999999999999999, 38.20846875548482),
            (5.0, 0.999999999, 31.47272874252078),
            (5.0, 0.999999999999, 39.2358478401201),
            (5.0, 0.999999999999999, 46.835268260827775),
            (30.0, 0.999999999, 75.24037981905084),
            (30.0, 0.999999999999, 85.92954232248108),
            (30.0, 0.999999999999999, 96.00299916243361),
        ];
        let bar = 4.0 * f64::EPSILON;
        let mut worst = 0.0_f64;
        for (a, p, want) in ROWS {
            let got = inverse_regularized_lower_gamma(p, a);
            let rel = ((got - want) / want).abs();
            worst = worst.max(rel);
            assert!(
                rel <= bar,
                "shape {a}, p {p:.17}: got {got:.17e}, want {want:.17e}, relative {rel:e} > {bar:e}"
            );
        }
        // Below the median the residual is untouched -- `P(a,x) - p` as before --
        // and these rows check it still lands on the reference.
        //
        // Bar: both tails are formed as `exp(S)`, so a rounding in any term of `S`
        // becomes a *relative* error in the value of the same size; hence the
        // value carries about `eps * T` where `T` is the largest term magnitude in
        // `S`. The inversion then scales that by `A = p / (x * f(x))`. Over this
        // table `T * A` peaks at 55 (shape 0.5 at `p = 1e-12`: `T = |a*ln x| = 28`,
        // `A = 2`), and doubling for the roundings in the seed and the Halley steps
        // themselves gives 128 ulp. Worst measured is 29 ulp.
        for (a, p, want) in [
            (0.5_f64, 1.0e-12_f64, 7.853981633974483e-25_f64),
            (0.5, 1.0e-3, 7.85398574631245e-7),
            (2.0, 1.0e-12, 1.4142142290401938e-6),
            (2.0, 0.4, 1.3764213420628868),
            (30.0, 1.0e-3, 15.869170797140358),
            (30.0, 0.4, 28.309997390436227),
        ] {
            let got = inverse_regularized_lower_gamma(p, a);
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= 128.0 * f64::EPSILON,
                "lower tail at shape {a}, p {p:e}: got {got:e}, want {want:e}, relative {rel:e}"
            );
        }
        println!(
            "upper-tail quantile: worst relative {worst:e} over {} rows",
            ROWS.len()
        );
    }

    #[test]
    fn poisson_cdf_below_the_mean_is_a_number_rather_than_zero() {
        // `P(Y <= k)` is `Q(k+1, mu)`, the *upper* incomplete gamma. Reconstructing
        // it as `1 - P` returns exactly 0.0 for every count far enough below the
        // mean, because `P` has already rounded to 1 -- not a loss of digits but a
        // loss of the answer. `poisson_quantile` searches on this CDF, so the
        // lower endpoint of a count predictive interval was being chosen against a
        // flat zero.
        //
        // At `k = 0` the identity is closed form and needs no external reference:
        // `P(Y <= 0) = e^{-mu}` exactly, which is what this fn's own doc claims.
        //
        // Bar: the survival value is `exp(S)` with `S = a*ln x - x - ln Gamma(a) +
        // ln h`, whose terms are of magnitude `mu`. Each is rounded, so `S` carries
        // absolute error of order `eps * mu`, and `exp` turns absolute argument
        // error into relative result error one for one -- so the floor grows
        // linearly in the mean, and `8 * eps * mu` is the ceiling. Measured at
        // `mu = 5`: 4.1 ulp against a 40-ulp bar. The old `1 - P` form returns
        // exactly 0 from `mu = 37` up, a relative error of 1 that no bar admits.
        for mu in [5.0_f64, 20.0, 50.0, 100.0, 300.0, 700.0] {
            let got = poisson_cdf_at(0.0, mu);
            let want = (-mu).exp();
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= 8.0 * f64::EPSILON * mu,
                "P(Y<=0 | mu={mu}) = {got:e}, want e^-mu = {want:e}, relative {rel:e}"
            );
        }
        // For `k > 0` the same `eps * mu` accounting applies, with the Lentz
        // recurrence contributing about one rounding per iteration on top; the
        // largest mean here is 200, so `8 * eps * mu` again bounds it.
        for (k, mu, want) in [
            (2.0_f64, 60.0_f64, 1.6295866529378224e-23_f64),
            (5.0, 120.0, 1.6584764014207315e-44),
            (10.0, 200.0, 4.109584943447632e-71),
        ] {
            let got = poisson_cdf_at(k, mu);
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= 8.0 * f64::EPSILON * mu,
                "P(Y<={k} | mu={mu}) = {got:e}, want {want:e}, relative {rel:e}"
            );
        }
    }

    #[test]
    fn incomplete_gamma_is_accurate_from_small_to_huge_shape() {
        // #4068: the series and continued fraction were capped at 1000 terms and
        // returned the partial sum as converged. Near `x ≈ a` the series needs
        // about `8.5·√a` terms, so for `a ≳ 1.4e4` the "converged" value was a
        // truncation (P(1e6, 999500) came out 0.2419 against 0.3086).
        //
        // References: 60-digit mpmath sums, independent of every routine here —
        // the power series for `P` when `x < a`, the Legendre continued fraction
        // for `Q` otherwise, each run to its own tail bound, times
        // `exp(a ln x − x − ln Γ(a))` at that precision. Each row is
        // `(a, x, whether the small tail is P, that tail)`; the last four are the
        // issue's own cases.
        //
        // Bar: the small tail carries the prefactor `e^{−a·h}`, `h = λ − 1 − ln λ`,
        // whose relative error is `a·h` times the relative error of `h` — the
        // conditioning of the function in `x` itself, since a one-ulp change of
        // `x` moves `a·h` by `a·|λ − 1|/λ·ε`. So the error is measured in units
        // of `(1 + a·h)·ε`; the old truncated sums were off by up to 22% here.
        // Beyond that, the series and the continued fraction round once per
        // term, and at small shapes near `x ≈ a` they take a few dozen terms:
        // the bar is 32 units, against a measured worst of 19 (`a = x = 3`).
        const UNIT_BAR: f64 = 32.0;
        const ROWS: [(f64, f64, bool, f64); 66] = [
            (0.7, 0.013999999999999999, true, 5.51313485464876e-2),
            (0.7, 0.13999999999999999, true, 2.62568501634967e-1),
            (0.7, 0.42, true, 5.0834813308743318e-1),
            (0.7, 0.7, false, 3.4341093974049586e-1),
            (0.7, 1.19, false, 1.9102554172690131e-1),
            (0.7, 1.5366600265340755, false, 1.2825810179184002e-1),
            (0.7, 2.4499999999999997, false, 4.6443606389297197e-2),
            (0.7, 4.0466401061363015, false, 8.3388697938738817e-3),
            (0.7, 4.199999999999999, false, 7.0868634681294926e-3),
            (3.0, 0.06, true, 3.441824024472021e-5),
            (3.0, 0.6000000000000001, true, 2.3115287752632959e-2),
            (3.0, 1.2679491924311224, true, 1.3557137117993433e-1),
            (3.0, 1.7999999999999998, true, 2.6937891406058745e-1),
            (3.0, 3.0, false, 4.2319008112684352e-1),
            (3.0, 4.732050807568877, false, 1.4911018549049089e-1),
            (3.0, 5.1, false, 1.1647834313417626e-1),
            (3.0, 9.92820323027551, false, 2.9371426399812767e-3),
            (3.0, 10.5, false, 1.8346159379269044e-3),
            (3.0, 18.0, false, 2.7566263337929857e-6),
            (12.0, 0.24, true, 6.1101098698737629e-17),
            (12.0, 2.4000000000000004, true, 8.4535131100734353e-6),
            (12.0, 7.199999999999999, true, 6.2905833907867146e-2),
            (12.0, 8.535898384862247, true, 1.5441621106288932e-1),
            (12.0, 12.0, false, 4.615973330636182e-1),
            (12.0, 15.464101615137753, false, 1.5586364364312884e-1),
            (12.0, 20.4, false, 1.7518317615758892e-2),
            (12.0, 25.856406460551018, false, 8.5271811759288256e-4),
            (12.0, 42.0, false, 1.3852814835795081e-8),
            (12.0, 72.0, false, 4.2762823157007119e-19),
            (150.0, 3.0, true, 3.2894660604843632e-193),
            (150.0, 30.0, true, 7.5592303464840563e-55),
            (150.0, 90.0, true, 4.7953023275883495e-9),
            (150.0, 101.01020514433644, true, 3.1572477177059796e-6),
            (150.0, 137.7525512860841, true, 1.5836977907217069e-1),
            (150.0, 150.0, false, 4.8914177025064032e-1),
            (150.0, 162.2474487139159, false, 1.5840111742024859e-1),
            (150.0, 198.98979485566355, false, 1.2664172915402692e-4),
            (150.0, 255.0, false, 4.2105749324015774e-13),
            (150.0, 525.0, false, 7.2920290967798446e-84),
            (150.0, 900.0, false, 6.5280919022324705e-212),
            (3000.0, 1800.0, true, 7.359801195156802e-147),
            (3000.0, 2780.9109769979336, true, 2.1051974788722459e-5),
            (3000.0, 2945.2277442494837, true, 1.5864163418646316e-1),
            (3000.0, 3000.0, false, 4.9757211010594567e-1),
            (3000.0, 3054.7722557505167, false, 1.5864198330967965e-1),
            (3000.0, 3219.0890230020664, false, 4.5649067881116096e-5),
            (3000.0, 5100.0, false, 2.2135555968387508e-223),
            (100000.0, 98735.08893593265, true, 2.9605184180992645e-5),
            (100000.0, 99683.77223398315, true, 1.5865484973789696e-1),
            (100000.0, 100000.0, false, 4.9957947788963482e-1),
            (100000.0, 100316.22776601683, false, 1.5865485155167504e-1),
            (100000.0, 101264.91106406735, false, 3.3838116386882088e-5),
            (1000000.0, 996000.0, true, 3.1007118211082967e-5),
            (1000000.0, 999000.0, true, 1.5865521357430365e-1),
            (1000000.0, 1000000.0, false, 4.9986701923912741e-1),
            (1000000.0, 1000999.9999999999, false, 1.5865521363168786e-1),
            (1000000.0, 1004000.0, false, 3.234544731347768e-5),
            (100000000.0, 99960000.0, true, 3.1604377116199155e-5),
            (100000000.0, 99990000.0, true, 1.5865525352814383e-1),
            (100000000.0, 100000000.0, false, 4.9998670192398588e-1),
            (100000000.0, 100010000.0, false, 1.5865525352820119e-1),
            (100000000.0, 100040000.0, false, 3.1738207368808895e-5),
            (100000.0, 99700.0, true, 1.7141731451450292e-1),
            (1000000.0, 999500.0, true, 3.0862555689081532e-1),
            (1000001.0, 1000000.0, true, 4.9973403851371635e-1),
            (100000000.0, 100000001.5, false, 4.9992686058264875e-1),
        ];
        for (a, x, small_is_p, want) in ROWS {
            let (p, q) = regularized_incomplete_gamma_pair(a, x);
            let got = if small_is_p { p } else { q };
            let lambda = x / a;
            let ah = a * (lambda - 1.0 - lambda.ln()).abs();
            let units = ((got - want) / want).abs() / ((1.0 + ah) * f64::EPSILON);
            assert!(
                units <= UNIT_BAR,
                "a={a} x={x}: got {got:e}, want {want:e} ({units:.1} units of (1+a·h)ε)"
            );
            assert!(
                (p + q - 1.0).abs() <= 2.0 * f64::EPSILON,
                "a={a} x={x}: P + Q = {}",
                p + q
            );
        }
    }

    #[test]
    fn incomplete_gamma_at_its_mean_holds_past_the_integer_limit() {
        // `Q(a, a) = ½ − 1/(3·√(2πa)) + O(a^{−3/2})`, and the next term is
        // `(1/(540a))/√(2πa)`, below `1e−21` at these shapes. At `a = 1e16`,
        // `a + 1 == a`: the old series/fraction split at `x < a + 1` and its
        // capped sums are both gone here.
        for a in [1e12_f64, 1e16] {
            let want = 0.5 - 1.0 / (3.0 * (2.0 * std::f64::consts::PI * a).sqrt());
            let (p, q) = regularized_incomplete_gamma_pair(a, a);
            assert!(
                ((q - want) / want).abs() <= 4.0 * f64::EPSILON,
                "Q({a:e}, {a:e}) = {q:e}, want {want:e}"
            );
            assert!((p + q - 1.0).abs() <= 2.0 * f64::EPSILON, "P + Q = {}", p + q);
        }
    }

    #[test]
    fn count_and_gamma_quantiles_are_exact_at_large_means() {
        // #4068: the capped sums moved these quantiles by thousands. The
        // references are 60-digit mpmath inversions of the same laws
        // (`P(Y ≤ k) = Q(k + 1, μ)` for the Poisson), independent of this file.
        for (p, mu, want) in [
            (0.975, 1e6, 1_001_960.0),
            (0.025, 1e6, 998_041.0),
            (0.975, 1e7, 10_006_198.0),
            (0.025, 1e7, 9_993_803.0),
        ] {
            assert_eq!(poisson_quantile(p, mu), want, "poisson_quantile(p={p}, μ={mu:e})");
        }
        // The quantile's own conditioning is `ΔP/(x·f(x))`: a relative error
        // `(1 + a·h)ε` in the tail moves `x` by far less than an ulp here, so the
        // bar is the rounding of `x` itself.
        for (p, a, want) in [
            (0.025, 1e6, 998_040.983_340_293_9),
            (0.025, 1e7, 9_993_802.996_884_267),
            (0.975, 1e7, 10_006_198.897_421_6),
        ] {
            let got = gamma_quantile(p, a, 1.0);
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= 4.0 * f64::EPSILON,
                "gamma_quantile(p={p}, a={a:e}) = {got}, want {want} (relative {rel:e})"
            );
        }
    }

    #[test]
    fn regularized_lower_gamma_is_accurate_and_unclamped_below_statrs_floor() {
        use statrs::function::gamma::{gamma_lr, ln_gamma};

        // (1) Agrees with statrs `gamma_lr` everywhere statrs is itself valid
        // (arguments well above its `x ≤ 1.11e-15` clamp), across both the
        // series (x < a+1) and continued-fraction (x ≥ a+1) branches.
        for &a in &[0.05_f64, 0.3, 1.0, 2.5, 50.0] {
            for &x in &[1e-6_f64, 0.01, 0.5, 1.0, 3.0, 25.0, 120.0] {
                let ours = regularized_lower_gamma(a, x);
                let theirs = gamma_lr(a, x);
                assert!(
                    (ours - theirs).abs() < 1e-12,
                    "P({a},{x}): ours={ours} statrs={theirs}"
                );
                assert!(
                    (0.0..=1.0).contains(&ours),
                    "P({a},{x})={ours} out of [0,1]"
                );
            }
        }

        // (2) Exp(1) closed form P(1, x) = 1 − e^{−x}.
        for &x in &[1e-3_f64, 0.25, 2.0, 9.0] {
            assert!((regularized_lower_gamma(1.0, x) - (1.0 - (-x).exp())).abs() < 1e-13);
        }

        // (3) The regression heart of #1018: for x far below the naive
        // small-argument clamp the CDF must remain a faithful, nonzero value,
        // not snap to 0. Compare to the small-x leading order
        // P(a, x) ≈ x^a / Γ(a+1).
        //
        // This used to open with `assert_eq!(gamma_lr(a, x), 0.0)` as a
        // "precondition: statrs clamps P(a,x) to 0" — i.e. it asserted a
        // THIRD-PARTY DEFECT as a premise, so statrs fixing its clamp broke a
        // test of our own unchanged, already-correct function. statrs 0.19
        // fixed it: `gamma_lr(0.05, 1e-20)` now returns 0.1027216865271675,
        // which is `x^a/Γ(a+1)` to four figures.
        //
        // What the fixture actually needs is that `x` sits far below the
        // threshold where a naive series/clamp implementation gives up — that
        // is a property of the FIXTURE, checkable without reference to anyone
        // else's behaviour — and that OUR value is faithful there. Both are
        // asserted; nothing about statrs is.
        const NAIVE_SMALL_ARG_CLAMP: f64 = 1.11e-15;
        for &(a, x) in &[(0.05_f64, 1e-20_f64), (0.1, 1e-25), (0.02, 1e-40)] {
            assert!(
                x < NAIVE_SMALL_ARG_CLAMP,
                "fixture must probe below the naive small-argument clamp: x={x}"
            );
            let ours = regularized_lower_gamma(a, x);
            let leading = (a * x.ln() - ln_gamma(a + 1.0)).exp();
            assert!(ours > 0.0, "P({a},{x})={ours} collapsed to zero");
            assert!(
                (ours - leading).abs() < 1e-9 * leading,
                "P({a},{x})={ours}, leading order {leading}"
            );
        }
    }

    #[test]
    fn gamma_quantile_scale_and_monotonicity() {
        // Scale is a pure multiplier, and the quantile is strictly increasing
        // in p (an equal-tailed interval must order correctly).
        let q_unit = gamma_quantile(0.9, 3.0, 1.0);
        let q_scaled = gamma_quantile(0.9, 3.0, 7.5);
        assert!((q_scaled - 7.5 * q_unit).abs() < 1e-9 * q_scaled.max(1.0));

        let mut prev = 0.0;
        for i in 1..100 {
            let p = i as f64 / 100.0;
            let q = gamma_quantile(p, 2.0, 1.0);
            assert!(q > prev, "quantile not increasing at p={p}: {q} <= {prev}");
            prev = q;
        }
    }

    #[test]
    fn gamma_quantile_rejects_degenerate_parameters() {
        assert!(gamma_quantile(0.5, -1.0, 1.0).is_nan());
        assert!(gamma_quantile(0.5, 1.0, 0.0).is_nan());
        assert!(gamma_quantile(0.5, f64::NAN, 1.0).is_nan());
        assert_eq!(gamma_quantile(0.0, 2.0, 1.0), 0.0);
        assert_eq!(gamma_quantile(-0.1, 2.0, 1.0), 0.0);
        assert!(gamma_quantile(1.0, 2.0, 1.0).is_infinite());
    }

    #[test]
    fn gamma_moment_matched_interval_is_the_exact_conditional_gamma_when_se_vanishes() {
        // With no estimation uncertainty the total predictive variance is the
        // pure observation noise `Var(Y|μ) = φμ²`, and the moment-matched Gamma
        // must coincide *exactly* with the conditional `Gamma(shape = 1/φ,
        // scale = φμ)` (#817). Check against the analytic Gamma quantiles for a
        // shape-4 (φ = 0.25) Gamma at the equal-tailed 2.5%/97.5% levels.
        let phi = 0.25_f64; // shape k = 1/φ = 4
        let mu = 7.5_f64;
        let total_var = phi * mu * mu; // SE(μ̂) = 0
        let (lo, hi) = gamma_moment_matched_interval(mu, total_var, 0.025, 0.975)
            .expect("non-degenerate moment-matched Gamma interval");

        let analytic_lo = gamma_quantile(0.025, 1.0 / phi, phi * mu);
        let analytic_hi = gamma_quantile(0.975, 1.0 / phi, phi * mu);
        assert!(
            (lo - analytic_lo).abs() < 1e-9 * analytic_lo.max(1.0)
                && (hi - analytic_hi).abs() < 1e-9 * analytic_hi.max(1.0),
            "moment-matched interval [{lo}, {hi}] != conditional Gamma \
             [{analytic_lo}, {analytic_hi}]"
        );
    }

    #[test]
    fn gamma_moment_matched_interval_is_right_skewed_not_symmetric() {
        // The whole point of #817: for a right-skewed Gamma the equal-tailed
        // band is *asymmetric* about the mean — the upper gap exceeds the lower
        // gap — and the lower edge sits FAR above the symmetric-band edge
        // `μ·(1 − z/√k)`, which for shape 4 hugs the support floor at ≈ 0.02·μ.
        let phi = 0.25_f64; // shape 4, CV = 0.5
        let mu = 10.0_f64;
        let total_var = phi * mu * mu;
        let z = 1.959_963_984_540_054_f64; // 97.5% standard-normal quantile
        let (lo, hi) =
            gamma_moment_matched_interval(mu, total_var, normal_cdf(-z), normal_cdf(z)).unwrap();

        // Ordered, strictly positive, brackets the mean.
        assert!(
            0.0 < lo && lo < mu && mu < hi,
            "interval [{lo}, {hi}] ∌ μ={mu}"
        );
        // Right skew: the upper gap is the larger one.
        let lower_gap = mu - lo;
        let upper_gap = hi - mu;
        assert!(
            upper_gap > 1.3 * lower_gap,
            "expected a right-skewed band (upper gap ≫ lower gap), got \
             lower_gap={lower_gap}, upper_gap={upper_gap}"
        );
        // The symmetric lower edge would be μ·(1 − z·√φ) = 10·(1 − 1.96·0.5) ≈
        // 0.20 — essentially the support floor. The skew-correct lower edge sits
        // well above it (true Gamma 2.5% quantile ≈ 0.27·μ for shape 4).
        let symmetric_lower = mu * (1.0 - z * phi.sqrt());
        assert!(
            lo > 2.0 * symmetric_lower.max(0.0) + 1.0,
            "skew-correct lower edge {lo} should sit well above the symmetric \
             edge {symmetric_lower}"
        );
    }

    #[test]
    fn gamma_moment_matched_interval_widens_with_estimation_uncertainty() {
        // Adding estimation variance SE(μ̂)² to the observation noise must widen
        // the predictive band (lower edge down, upper edge up) — it is the
        // moment-matched predictive, not just the conditional law.
        let phi = 0.25_f64;
        let mu = 5.0_f64;
        let obs_var = phi * mu * mu;
        let (lo0, hi0) = gamma_moment_matched_interval(mu, obs_var, 0.025, 0.975).unwrap();
        let (lo1, hi1) = gamma_moment_matched_interval(mu, obs_var + 4.0, 0.025, 0.975).unwrap();
        assert!(
            lo1 < lo0 && hi1 > hi0,
            "estimation uncertainty must widen the band: [{lo0},{hi0}] -> [{lo1},{hi1}]"
        );
    }

    #[test]
    fn gamma_moment_matched_interval_rejects_degenerate_and_near_gaussian_inputs() {
        // Non-positive mean / variance, or non-finite inputs => None (caller
        // falls back to the symmetric Gaussian edges).
        assert!(gamma_moment_matched_interval(0.0, 1.0, 0.025, 0.975).is_none());
        assert!(gamma_moment_matched_interval(-1.0, 1.0, 0.025, 0.975).is_none());
        assert!(gamma_moment_matched_interval(1.0, 0.0, 0.025, 0.975).is_none());
        assert!(gamma_moment_matched_interval(1.0, -1.0, 0.025, 0.975).is_none());
        assert!(gamma_moment_matched_interval(f64::NAN, 1.0, 0.025, 0.975).is_none());
        assert!(gamma_moment_matched_interval(1.0, f64::INFINITY, 0.025, 0.975).is_none());
        // A finite, well-conditioned case still returns Some.
        assert!(gamma_moment_matched_interval(3.0, 2.0, 0.025, 0.975).is_some());
    }

    /// #4068: an enormous shape is a band, not a refusal. The doc on
    /// [`gamma_moment_matched_interval`] used to say the caller should fall
    /// back to symmetric edges there; nothing in the code ever did, and since
    /// the incomplete gamma resolves every shape there is nothing to fall back
    /// from. References are 40-digit quadrature values.
    ///
    /// The bar counts the roundings between the references and the returned
    /// edge: `shape = mu * mu / total_var` is two, `scale = total_var / mu` is
    /// one, the quantile is certified by its own inversion, and the final
    /// multiply by `scale` is one.
    #[test]
    fn large_count_and_large_shape_bands_carry_their_nominal_mass_4068() {
        // Q(1001961, 1e6) = 0.9750004 >= 0.975 > Q(1001960, 1e6) = 0.9749452,
        // so the 97.5% Poisson quantile at mu = 1e6 is 1001960.
        assert_eq!(poisson_quantile(0.975, 1.0e6), 1_001_960.0);
        // Mean = variance = 1e7 is Gamma(shape 1e7, scale 1); its 2.5%
        // quantile by bisection at 40 digits on the quadrature CDF.
        let (lower, _) = gamma_moment_matched_interval(1.0e7, 1.0e7, 0.025, 0.975)
            .expect("an enormous shape still has a Gamma band");
        let want = 9_993_802.996_884_266_918_8;
        assert!(
            (lower - want).abs() <= 4.0 * f64::EPSILON * want,
            "lower edge {lower}, want {want}"
        );
    }

    #[test]
    fn beta_moment_matched_interval_is_the_exact_conditional_beta_when_se_vanishes() {
        // With no estimation uncertainty the total predictive variance is the
        // pure observation noise `μ(1−μ)/(1+φ)`, and the moment-matched Beta must
        // coincide *exactly* with the conditional `Beta(μφ, (1−μ)φ)` (#1194).
        let phi = 8.0_f64;
        let mu = 0.2_f64;
        let total_var = mu * (1.0 - mu) / (1.0 + phi); // SE(μ̂) = 0
        let (lo, hi) = beta_moment_matched_interval(mu, 1.0 - mu, total_var, 0.025, 0.975)
            .expect("non-degenerate moment-matched Beta interval");
        let analytic_lo = beta_quantile(0.025, mu * phi, (1.0 - mu) * phi);
        let analytic_hi = beta_quantile(0.975, mu * phi, (1.0 - mu) * phi);
        assert!(
            (lo - analytic_lo).abs() < 1e-9 && (hi - analytic_hi).abs() < 1e-9,
            "moment-matched interval [{lo}, {hi}] != conditional Beta [{analytic_lo}, {analytic_hi}]"
        );
    }

    #[test]
    fn beta_moment_matched_interval_is_skewed_not_symmetric() {
        // For a small-mean Beta the equal-tailed band is asymmetric about μ (the
        // upper gap exceeds the lower gap) and the lower edge sits well above the
        // symmetric edge `μ − z·σ`, which on this data dives below 0.
        let phi = 8.0_f64;
        let mu = 0.15_f64;
        let total_var = mu * (1.0 - mu) / (1.0 + phi);
        let z = 1.959_963_984_540_054_f64;
        let (lo, hi) =
            beta_moment_matched_interval(mu, 1.0 - mu, total_var, normal_cdf(-z), normal_cdf(z))
                .unwrap();
        assert!(
            0.0 < lo && lo < mu && mu < hi && hi < 1.0,
            "interval [{lo},{hi}] ∌ μ={mu}"
        );
        let lower_gap = mu - lo;
        let upper_gap = hi - mu;
        assert!(
            upper_gap > 1.2 * lower_gap,
            "expected a right-skewed band (upper gap > lower gap): lower={lower_gap}, upper={upper_gap}"
        );
        let symmetric_lower = mu - z * total_var.sqrt();
        assert!(
            symmetric_lower < 0.0 && lo > 0.0,
            "skew-correct lower edge {lo} should stay positive where the symmetric edge {symmetric_lower} goes negative"
        );
    }

    #[test]
    fn beta_moment_matched_interval_rejects_degenerate_and_over_dispersed_inputs() {
        // A spread at a mean on the boundary, a negative mean or variance, or a
        // non-finite input is no law's moments => None.
        assert!(beta_moment_matched_interval(0.0, 1.0, 0.01, 0.025, 0.975).is_none());
        assert!(beta_moment_matched_interval(1.0, 0.0, 0.01, 0.025, 0.975).is_none());
        assert!(beta_moment_matched_interval(-0.1, 1.1, 0.01, 0.025, 0.975).is_none());
        assert!(beta_moment_matched_interval(0.3, 0.7, -0.01, 0.025, 0.975).is_none());
        assert!(beta_moment_matched_interval(f64::NAN, 0.5, 0.01, 0.025, 0.975).is_none());
        // Variance at/over the Bernoulli ceiling μ(1−μ): no Beta matches => None.
        assert!(beta_moment_matched_interval(0.5, 0.5, 0.25, 0.025, 0.975).is_none());
        assert!(beta_moment_matched_interval(0.5, 0.5, 0.30, 0.025, 0.975).is_none());
        // No spread: the point mass at the mean.
        assert_eq!(
            beta_moment_matched_interval(0.3, 0.7, 0.0, 0.025, 0.975),
            Some((0.3, 0.3))
        );
        // A well-conditioned case still returns Some.
        assert!(beta_moment_matched_interval(0.4, 0.6, 0.02, 0.025, 0.975).is_some());
    }

    #[test]
    fn beta_moment_matched_interval_widens_with_estimation_uncertainty() {
        let phi = 8.0_f64;
        let mu = 0.3_f64;
        let obs_var = mu * (1.0 - mu) / (1.0 + phi);
        let (lo0, hi0) = beta_moment_matched_interval(mu, 1.0 - mu, obs_var, 0.025, 0.975).unwrap();
        let (lo1, hi1) =
            beta_moment_matched_interval(mu, 1.0 - mu, obs_var + 0.01, 0.025, 0.975).unwrap();
        assert!(
            lo1 < lo0 && hi1 > hi0,
            "estimation uncertainty must widen the band: [{lo0},{hi0}] -> [{lo1},{hi1}]"
        );
    }

    /// #3140: a predictive mean that underflows to zero with no spread is the point
    /// mass at zero, and a spread-free Beta predictive is the point mass at its mean,
    /// so each moment match returns that point's exact band instead of none (the
    /// caller used to fall back to a symmetric band and clamp it).
    #[test]
    fn a_spread_free_predictive_is_its_point_mass_3140() {
        let zero = Some((0.0, 0.0));
        assert_eq!(gamma_moment_matched_interval(0.0, 0.0, 0.025, 0.975), zero);
        assert_eq!(
            inverse_gaussian_moment_matched_interval(0.0, 0.0, 0.025, 0.975),
            zero
        );
        assert_eq!(
            poisson_moment_matched_interval(0.0, 0.0, 0.025, 0.975),
            zero
        );
        assert_eq!(
            negative_binomial_moment_matched_interval(0.0, 1.5, 0.0, 0.025, 0.975),
            zero
        );
        assert_eq!(
            tweedie_moment_matched_interval(0.0, 1.0, 1.5, 0.0, 0.025, 0.975),
            zero
        );
        assert_eq!(
            beta_moment_matched_interval(0.0, 1.0, 0.0, 0.025, 0.975),
            zero
        );
        assert_eq!(
            beta_moment_matched_interval(1.0, 0.0, 0.0, 0.025, 0.975),
            Some((1.0, 1.0))
        );
        // A mean of zero that still carries spread is no law's moments on [0, ∞).
        assert!(poisson_moment_matched_interval(0.0, 1e-3, 0.025, 0.975).is_none());
    }

    /// #3140: the Beta band reads its Bernoulli ceiling from the carried complement.
    /// Near the upper edge `1 − mu` rounds to zero and no Beta has any spread there;
    /// with the complement carried the band is the moment-matched Beta, inside `[0, 1]`.
    #[test]
    fn the_beta_band_lives_on_the_carried_complement_3140() {
        let complement = 1.0e-20_f64;
        let mu = 1.0 - complement;
        assert_eq!(mu, 1.0, "fixture precondition: the mean rounds to one");
        let phi = 8.0_f64;
        let total_var = mu * complement / (1.0 + phi);
        assert!(
            beta_moment_matched_interval(mu, 1.0 - mu, total_var, 0.025, 0.975).is_none(),
            "fixture precondition: the rounded complement leaves no Beta"
        );
        let (lo, hi) = beta_moment_matched_interval(mu, complement, total_var, 0.025, 0.975)
            .expect("the carried complement admits the conditional Beta");
        assert!(
            (0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi) && lo <= hi,
            "the band [{lo}, {hi}] lies in the support without a clamp"
        );
    }

    #[test]
    fn negative_binomial_quantile_matches_known_reference_values() {
        // Reference NB quantiles cross-checked against scipy
        // `nbinom.ppf(p, n=θ, prob=θ/(θ+μ))` — the integer count k with the
        // smallest CDF ≥ p. Spans the zero-atom lower tail, the right-skewed
        // upper tail, and a larger-mean near-Gaussian case.
        let cases: [(f64, f64, f64, f64); 8] = [
            // (p, μ, θ, expected integer quantile)
            (0.025, 1.6, 1.5, 0.0), // zero mass ≈ 0.34 > 0.025 ⇒ lower edge 0
            (0.5, 1.6, 1.5, 1.0),
            (0.975, 1.6, 1.5, 6.0),
            (0.99, 1.6, 1.5, 8.0),
            (0.025, 20.0, 5.0, 5.0),
            (0.975, 20.0, 5.0, 43.0),
            (0.5, 20.0, 5.0, 19.0),
            (0.975, 0.5, 2.0, 3.0),
        ];
        for (p, mu, theta, expected) in cases {
            let got = negative_binomial_quantile(p, mu, theta);
            assert_eq!(
                got, expected,
                "negative_binomial_quantile(p={p}, μ={mu}, θ={theta}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn negative_binomial_quantile_is_a_valid_cdf_inverse() {
        // The returned integer k must be the *smallest* with CDF(k) ≥ p:
        // CDF(k) ≥ p and (for k ≥ 1) CDF(k−1) < p, across a grid of (μ, θ, p).
        use statrs::function::beta::beta_reg;
        for &mu in &[0.3_f64, 1.6, 5.0, 25.0, 120.0] {
            for &theta in &[0.5_f64, 1.5, 5.0, 40.0] {
                let prob = theta / (theta + mu);
                for &p in &[0.01_f64, 0.025, 0.1, 0.5, 0.9, 0.975, 0.99] {
                    let k = negative_binomial_quantile(p, mu, theta);
                    assert!(
                        k.is_finite() && k >= 0.0 && k.fract() == 0.0,
                        "non-integer k={k}"
                    );
                    let cdf_k = beta_reg(theta, k + 1.0, prob);
                    assert!(
                        cdf_k + 1e-12 >= p,
                        "CDF({k}) = {cdf_k} < p = {p} (μ={mu}, θ={theta})"
                    );
                    if k >= 1.0 {
                        let cdf_below = beta_reg(theta, k, prob);
                        assert!(
                            cdf_below < p,
                            "k={k} not minimal: CDF({}) = {cdf_below} ≥ p = {p} (μ={mu}, θ={theta})",
                            k - 1.0
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn negative_binomial_quantile_boundaries_and_degeneracy() {
        assert_eq!(negative_binomial_quantile(0.0, 2.0, 1.5), 0.0);
        assert_eq!(negative_binomial_quantile(-0.1, 2.0, 1.5), 0.0);
        assert!(negative_binomial_quantile(1.0, 2.0, 1.5).is_infinite());
        assert_eq!(negative_binomial_quantile(0.5, 0.0, 1.5), 0.0); // point mass at 0
        assert!(negative_binomial_quantile(0.5, -1.0, 1.5).is_nan());
        assert!(negative_binomial_quantile(0.5, 2.0, 0.0).is_nan());
        assert!(negative_binomial_quantile(0.5, 2.0, f64::NAN).is_nan());
        // Monotone non-decreasing in p (discrete ⇒ plateaus allowed).
        let mut prev = 0.0;
        for i in 1..100 {
            let p = i as f64 / 100.0;
            let q = negative_binomial_quantile(p, 4.0, 2.0);
            assert!(q >= prev, "NB quantile decreased at p={p}: {q} < {prev}");
            prev = q;
        }
    }

    #[test]
    fn negative_binomial_moment_matched_interval_is_exact_conditional_when_se_vanishes() {
        // SE(μ̂) = 0 ⇒ total_var = μ + μ²/θ ⇒ θ_eff = θ, recovering the exact
        // conditional NB quantiles.
        let mu = 1.6_f64;
        let theta = 1.5_f64;
        let total_var = mu + mu * mu / theta;
        let (lo, hi) =
            negative_binomial_moment_matched_interval(mu, theta, total_var, 0.025, 0.975).unwrap();
        assert_eq!(lo, negative_binomial_quantile(0.025, mu, theta));
        assert_eq!(hi, negative_binomial_quantile(0.975, mu, theta));
    }

    #[test]
    fn negative_binomial_moment_matched_interval_widens_with_estimation_uncertainty() {
        // Adding estimation variance lowers θ_eff (more overdispersion) and must
        // not shrink the band; with enough added variance the upper edge grows.
        let mu = 8.0_f64;
        let theta = 4.0_f64;
        let obs_var = mu + mu * mu / theta;
        let (lo0, hi0) =
            negative_binomial_moment_matched_interval(mu, theta, obs_var, 0.025, 0.975).unwrap();
        let (lo1, hi1) =
            negative_binomial_moment_matched_interval(mu, theta, obs_var + 40.0, 0.025, 0.975)
                .unwrap();
        assert!(
            lo1 <= lo0 && hi1 > hi0,
            "band did not widen: [{lo0},{hi0}] -> [{lo1},{hi1}]"
        );
    }

    #[test]
    fn negative_binomial_moment_matched_interval_rejects_degenerate_inputs() {
        assert!(negative_binomial_moment_matched_interval(0.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(negative_binomial_moment_matched_interval(-1.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(negative_binomial_moment_matched_interval(2.0, 0.0, 1.0, 0.025, 0.975).is_none());
        assert!(negative_binomial_moment_matched_interval(2.0, 1.5, 0.0, 0.025, 0.975).is_none());
        assert!(
            negative_binomial_moment_matched_interval(f64::NAN, 1.5, 1.0, 0.025, 0.975).is_none()
        );
        assert!(negative_binomial_moment_matched_interval(2.0, 1.5, 6.0, 0.025, 0.975).is_some());
    }

    #[test]
    fn poisson_quantile_matches_known_reference_values() {
        // Reference integer quantiles from scipy.stats.poisson.ppf.
        let cases: [(f64, f64, f64); 9] = [
            // (p, μ, expected integer quantile)
            (0.025, 1.6, 0.0), // zero mass e^{−1.6} ≈ 0.20 < 0.025? no: 0.20 > 0.025 ⇒ 0
            (0.5, 1.6, 1.0),
            (0.975, 1.6, 4.0),
            (0.99, 1.6, 5.0),
            (0.025, 20.0, 12.0),
            (0.975, 20.0, 29.0),
            (0.5, 20.0, 20.0),
            (0.975, 0.5, 2.0),
            (0.025, 0.5, 0.0),
        ];
        for (p, mu, expected) in cases {
            let got = poisson_quantile(p, mu);
            assert_eq!(
                got, expected,
                "poisson_quantile(p={p}, μ={mu}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn poisson_quantile_is_a_valid_cdf_inverse() {
        // The returned integer k must be the *smallest* with CDF(k) ≥ p:
        // CDF(k) ≥ p and (for k ≥ 1) CDF(k−1) < p, across a grid of (μ, p).
        for &mu in &[0.3_f64, 1.6, 5.0, 25.0, 120.0] {
            for &p in &[0.01_f64, 0.025, 0.1, 0.5, 0.9, 0.975, 0.99] {
                let k = poisson_quantile(p, mu);
                assert!(
                    k.is_finite() && k >= 0.0 && k.fract() == 0.0,
                    "non-integer k={k}"
                );
                let cdf_k = poisson_cdf_at(k, mu);
                assert!(cdf_k + 1e-12 >= p, "CDF({k}) = {cdf_k} < p = {p} (μ={mu})");
                if k >= 1.0 {
                    let cdf_below = poisson_cdf_at(k - 1.0, mu);
                    assert!(
                        cdf_below < p,
                        "k={k} not minimal: CDF({}) = {cdf_below} ≥ p = {p} (μ={mu})",
                        k - 1.0
                    );
                }
            }
        }
    }

    #[test]
    fn poisson_quantile_boundaries_and_degeneracy() {
        assert_eq!(poisson_quantile(0.0, 2.0), 0.0);
        assert_eq!(poisson_quantile(-0.1, 2.0), 0.0);
        assert!(poisson_quantile(1.0, 2.0).is_infinite());
        assert_eq!(poisson_quantile(0.5, 0.0), 0.0); // point mass at 0
        assert!(poisson_quantile(0.5, -1.0).is_nan());
        assert!(poisson_quantile(0.5, f64::NAN).is_nan());
        // Monotone non-decreasing in p (discrete ⇒ plateaus allowed).
        let mut prev = 0.0;
        for i in 1..100 {
            let p = i as f64 / 100.0;
            let q = poisson_quantile(p, 4.0);
            assert!(
                q >= prev,
                "Poisson quantile decreased at p={p}: {q} < {prev}"
            );
            prev = q;
        }
    }

    #[test]
    fn poisson_moment_matched_interval_is_exact_conditional_when_se_vanishes() {
        // SE(μ̂) = 0 ⇒ total_var = μ ⇒ θ_eff = ∞, recovering the exact conditional
        // Poisson quantiles directly (no NB widening).
        for &mu in &[0.5_f64, 1.6, 20.0] {
            let (lo, hi) = poisson_moment_matched_interval(mu, mu, 0.025, 0.975).unwrap();
            assert_eq!(lo, poisson_quantile(0.025, mu));
            assert_eq!(hi, poisson_quantile(0.975, mu));
        }
    }

    #[test]
    fn poisson_moment_matched_interval_widens_with_estimation_uncertainty() {
        // Adding estimation variance lowers θ_eff (genuine overdispersion) and
        // must not shrink the band; with enough added variance the upper edge
        // grows beyond the conditional Poisson quantile.
        let mu = 20.0_f64;
        let (lo0, hi0) = poisson_moment_matched_interval(mu, mu, 0.025, 0.975).unwrap();
        let (lo1, hi1) = poisson_moment_matched_interval(mu, mu + 40.0, 0.025, 0.975).unwrap();
        assert!(
            lo1 <= lo0 && hi1 > hi0,
            "band did not widen: [{lo0},{hi0}] -> [{lo1},{hi1}]"
        );
        // A negligible excess (θ_eff above the switch threshold) must coincide
        // with the exact conditional Poisson — no discontinuity at the boundary.
        let (lo2, hi2) =
            poisson_moment_matched_interval(mu, mu + mu * mu * 1.0e-12, 0.025, 0.975).unwrap();
        assert_eq!((lo2, hi2), (lo0, hi0));
    }

    #[test]
    fn poisson_moment_matched_interval_is_skewed_not_symmetric() {
        // The whole point of #1193/#817: on a low-rate count the equal-tailed
        // upper edge sits ABOVE the symmetric `μ + z·√μ` band that under-covers
        // the upper tail, and the band is asymmetric about μ.
        let mu = 2.0_f64;
        let z = standard_normal_quantile(0.975).unwrap();
        let (lo, hi) = poisson_moment_matched_interval(mu, mu, 0.025, 0.975).unwrap();
        let sym_hi = mu + z * mu.sqrt();
        assert!(
            hi > sym_hi,
            "equal-tailed upper {hi} should exceed symmetric upper {sym_hi}"
        );
        // Upper tail reaches further from μ than the lower tail (right skew).
        assert!(
            (hi - mu) > (mu - lo),
            "band not right-skewed: lo={lo}, hi={hi}, μ={mu}"
        );
    }

    /// #4228: with no estimation variance the band is the exact
    /// `Binomial(m, μ)/m` quantile pair. References are
    /// `scipy.stats.binom(m, μ).ppf([0.025, 0.975])`.
    #[test]
    fn binomial_proportion_interval_is_the_exact_binomial_without_estimation_variance() {
        let cases = [
            (100.0, 0.5, 40.0, 60.0),
            (7.0, 0.13, 0.0, 3.0),
            (250.0, 0.02, 1.0, 10.0),
            (100_000.0, 0.01, 939.0, 1062.0),
        ];
        for (m, mu, lo, hi) in cases {
            let band = binomial_proportion_interval(mu, 1.0 - mu, m, 0.0, 0.025, 0.975);
            assert_eq!(band, Some((lo / m, hi / m)), "m = {m}, μ = {mu}");
        }
    }

    /// #4228: estimation variance in `p` turns the predictive into the
    /// beta-binomial with the moment-matched Beta shapes. References are
    /// `scipy.stats.betabinom(m, a, b).ppf([0.025, 0.975])` with
    /// `a + b = μ(1−μ)/v − 1`.
    #[test]
    fn binomial_proportion_interval_is_the_moment_matched_beta_binomial() {
        let cases = [
            (100.0, 0.5, 0.01, 28.0, 72.0),
            (40.0, 0.2, 0.001, 3.0, 14.0),
            (12.0, 0.9, 0.02, 5.0, 12.0),
            (30.0, 0.5, 0.2, 0.0, 30.0),
            (5000.0, 0.3, 1.0e-4, 1384.0, 1618.0),
            // A precision of ~2e11 is the binomial to within rounding, where a
            // log-gamma difference would already have cancelled.
            (200.0, 0.3, 1.0e-12, 48.0, 73.0),
        ];
        for (m, mu, v, lo, hi) in cases {
            let band = binomial_proportion_interval(mu, 1.0 - mu, m, v, 0.025, 0.975);
            assert_eq!(band, Some((lo / m, hi / m)), "m = {m}, μ = {mu}, v = {v}");
        }
    }

    /// #4228: one trial sees only the mean of `p`, so the band is
    /// `Bernoulli(μ)` whatever the estimation variance.
    #[test]
    fn binomial_proportion_interval_with_one_trial_is_bernoulli() {
        for v in [0.0, 1.0e-3, 0.2] {
            assert_eq!(
                binomial_proportion_interval(0.5, 0.5, 1.0, v, 0.025, 0.975),
                Some((0.0, 1.0))
            );
            assert_eq!(
                binomial_proportion_interval(0.99, 0.01, 1.0, v.min(1.0e-3), 0.025, 0.975),
                Some((1.0, 1.0))
            );
        }
    }

    #[test]
    fn binomial_proportion_interval_refuses_what_is_not_a_binomial_row() {
        for trials in [
            0.0,
            2.5,
            -3.0,
            f64::NAN,
            f64::INFINITY,
            2.0 * EXACT_INTEGER_LIMIT,
        ] {
            assert!(
                binomial_proportion_interval(0.4, 0.6, trials, 0.0, 0.025, 0.975).is_none(),
                "trials = {trials}"
            );
        }
        // At or past the Bernoulli ceiling no Beta carries the variance.
        assert!(binomial_proportion_interval(0.4, 0.6, 10.0, 0.24, 0.025, 0.975).is_none());
        // Point masses at either edge.
        assert_eq!(
            binomial_proportion_interval(0.0, 1.0, 10.0, 0.0, 0.025, 0.975),
            Some((0.0, 0.0))
        );
        assert_eq!(
            binomial_proportion_interval(1.0, 0.0, 10.0, 0.0, 0.025, 0.975),
            Some((1.0, 1.0))
        );
    }

    #[test]
    fn poisson_moment_matched_interval_rejects_degenerate_inputs() {
        assert!(poisson_moment_matched_interval(0.0, 1.0, 0.025, 0.975).is_none());
        assert!(poisson_moment_matched_interval(-1.0, 1.0, 0.025, 0.975).is_none());
        assert!(poisson_moment_matched_interval(2.0, 0.0, 0.025, 0.975).is_none());
        assert!(poisson_moment_matched_interval(2.0, 1.0, 0.025, 0.975).is_none()); // total_var < μ
        assert!(poisson_moment_matched_interval(f64::NAN, 5.0, 0.025, 0.975).is_none());
        assert!(poisson_moment_matched_interval(2.0, 5.0, 0.025, 0.975).is_some());
    }

    #[test]
    fn tweedie_quantile_is_a_valid_cdf_inverse() {
        // For a probability strictly above the zero atom the quantile `y` must
        // satisfy `CDF(y) ≈ q`: the bisection inverts `tweedie_cdf_at` exactly.
        let mu = 3.0_f64;
        let phi = 1.2_f64;
        let power = 1.5_f64;
        let lambda = mu.powf(2.0 - power) / (phi * (2.0 - power));
        let zero_mass = (-lambda).exp();
        for &q in &[0.30_f64, 0.5, 0.75, 0.9, 0.975, 0.99] {
            assert!(
                q > zero_mass,
                "test q must exceed the zero atom {zero_mass}"
            );
            let y = tweedie_quantile(q, mu, phi, power);
            assert!(y.is_finite() && y > 0.0, "quantile out of support: {y}");
            let cdf = tweedie_cdf_at(y, mu, phi, power);
            assert!((cdf - q).abs() < 1e-6, "CDF(Q(q)) != q: q={q}, cdf={cdf}");
        }
    }

    #[test]
    fn tweedie_quantile_returns_zero_atom_for_low_tail() {
        // When the requested lower-tail probability is at or below the point
        // mass at zero `e^{−λ}`, the quantile is exactly 0 (right-skewed low
        // means) — the zero-atom behaviour a continuous surrogate cannot mimic.
        let mu = 0.4_f64; // small mean ⇒ large zero atom
        let phi = 1.0_f64;
        let power = 1.5_f64;
        let lambda = mu.powf(2.0 - power) / (phi * (2.0 - power));
        let zero_mass = (-lambda).exp();
        assert!(
            zero_mass > 0.025,
            "fixture must have a fat zero atom: {zero_mass}"
        );
        assert_eq!(tweedie_quantile(0.025, mu, phi, power), 0.0);
        assert_eq!(tweedie_quantile(0.5 * zero_mass, mu, phi, power), 0.0);
    }

    /// Below `2⁵³` every integer is representable and the smallest qualifying
    /// count is returned exactly; past it no bisection can name one, and the
    /// quantile is `+∞` rather than a bracket whose midpoint rounds onto an end.
    #[test]
    fn count_quantile_is_exact_below_the_integer_limit_and_infinite_past_it() {
        let below = EXACT_INTEGER_LIMIT / 2.0 + 1.0;
        let step_at = |edge: f64| move |k: f64| if k >= edge { 1.0 } else { 0.0 };
        assert_eq!(count_quantile_bracket_bisect(step_at(below), 1.0, 0.5), below);
        assert_eq!(
            count_quantile_bracket_bisect(step_at(EXACT_INTEGER_LIMIT + 2.0), 1.0, 0.5),
            f64::INFINITY
        );
    }

    #[test]
    fn tweedie_quantile_boundaries_and_degeneracy() {
        let (mu, phi, power) = (2.0_f64, 1.0_f64, 1.6_f64);
        assert_eq!(tweedie_quantile(0.0, mu, phi, power), 0.0);
        assert_eq!(tweedie_quantile(-0.1, mu, phi, power), 0.0);
        assert_eq!(tweedie_quantile(1.0, mu, phi, power), f64::INFINITY);
        // Power outside (1, 2) or non-positive params are NaN.
        assert!(tweedie_quantile(0.5, mu, phi, 2.0).is_nan());
        assert!(tweedie_quantile(0.5, mu, phi, 1.0).is_nan());
        assert!(tweedie_quantile(0.5, 0.0, phi, power).is_nan());
        assert!(tweedie_quantile(0.5, mu, 0.0, power).is_nan());
    }

    #[test]
    fn tweedie_moment_matched_interval_is_exact_conditional_when_se_vanishes() {
        // total_var = φμ^p ⇒ φ_eff = φ, recovering the exact conditional Tweedie
        // quantiles.
        let mu = 3.0_f64;
        let phi = 1.2_f64;
        let power = 1.5_f64;
        let total_var = phi * mu.powf(power);
        let (lo, hi) =
            tweedie_moment_matched_interval(mu, phi, power, total_var, 0.025, 0.975).unwrap();
        assert_eq!(lo, tweedie_quantile(0.025, mu, phi, power));
        assert_eq!(hi, tweedie_quantile(0.975, mu, phi, power));
    }

    #[test]
    fn tweedie_moment_matched_interval_is_skewed_not_symmetric() {
        // A right-skewed Tweedie has the upper edge farther from the mean than
        // the lower edge — the symmetric `mu ± z·σ` band cannot reproduce this.
        let mu = 2.0_f64;
        let phi = 1.5_f64;
        let power = 1.5_f64;
        let total_var = phi * mu.powf(power);
        let (lo, hi) =
            tweedie_moment_matched_interval(mu, phi, power, total_var, 0.025, 0.975).unwrap();
        assert!(lo >= 0.0 && hi > mu && lo < mu);
        assert!(
            hi - mu > mu - lo,
            "interval is not right-skewed: lo={lo}, hi={hi}"
        );
    }

    #[test]
    fn tweedie_moment_matched_interval_widens_with_estimation_uncertainty() {
        // Adding estimation variance raises φ_eff and must not shrink the band;
        // the upper edge grows.
        let mu = 4.0_f64;
        let phi = 1.0_f64;
        let power = 1.5_f64;
        let obs_var = phi * mu.powf(power);
        let (lo0, hi0) =
            tweedie_moment_matched_interval(mu, phi, power, obs_var, 0.025, 0.975).unwrap();
        let (lo1, hi1) =
            tweedie_moment_matched_interval(mu, phi, power, obs_var + 30.0, 0.025, 0.975).unwrap();
        assert!(
            lo1 <= lo0 && hi1 > hi0,
            "band did not widen: [{lo0},{hi0}] -> [{lo1},{hi1}]"
        );
    }

    #[test]
    fn tweedie_moment_matched_interval_rejects_degenerate_inputs() {
        assert!(tweedie_moment_matched_interval(0.0, 1.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(-1.0, 1.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(2.0, 0.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(2.0, 1.0, 2.0, 1.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(2.0, 1.0, 1.5, 0.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(f64::NAN, 1.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(2.0, 1.0, 1.5, 6.0, 0.025, 0.975).is_some());
    }

    /// Relative error bound on `statrs`'s `beta_reg(a, b, x)`, read off its
    /// construction: the prefactor `exp(lnΓ(a+b) − lnΓ(a) − lnΓ(b) + a·ln x +
    /// b·ln(1−x))` carries an absolute exponent error of one rounding per term
    /// (`lnΓ` is accurate to 16 digits) plus one per addition, so at most
    /// `2·ε·Σ|terms|`; the Lentz continued fraction after it stops within `ε` of
    /// its limit and runs at most 140 steps of eight roundings each.
    fn beta_reg_relative_error(a: f64, b: f64, x: f64) -> f64 {
        use statrs::function::gamma::ln_gamma;
        let terms = ln_gamma(a + b).abs()
            + ln_gamma(a).abs()
            + ln_gamma(b).abs()
            + (a * x.ln()).abs()
            + (b * (1.0 - x).ln()).abs();
        f64::EPSILON * (2.0 * terms + 8.0 * 140.0 + 1.0)
    }

    /// Sum of `pmf(k)` over `k = lower..=upper`, the pmf given by its value at zero
    /// and the ratio `pmf(k)/pmf(k−1)`.
    fn count_set_mass(p0: f64, ratio: impl Fn(f64) -> f64, lower: f64, upper: f64) -> f64 {
        let mut pmf = p0;
        let mut mass = if lower <= 0.0 { p0 } else { 0.0 };
        let mut k = 1.0;
        while k <= upper {
            pmf *= ratio(k);
            if k >= lower {
                mass += pmf;
            }
            k += 1.0;
        }
        mass
    }

    #[test]
    fn count_band_content_is_the_mass_of_its_quantile_set_and_exceeds_its_level() {
        // A count band `[F⁻¹(p_lo), F⁻¹(p_hi)]` carries `F(hi) − F(lo − 1)`, and
        // `F(hi) ≥ p_hi`, `F(lo − 1) < p_lo` make that strictly more than
        // `p_hi − p_lo`: the atoms at the edges are the band's surplus, the
        // coverage it promises (#3534). The reference mass sums the pmf by its
        // one-rounding-per-step recurrence; the CDFs behind the content carry
        // relative error `8·ε·μ` each (see
        // `poisson_cdf_below_the_mean_is_a_number_rather_than_zero`), so
        // the two agree to `ε·(16·μ + hi + 1)`, doubled for the pmf's own error.
        let (p_lo, p_hi) = (0.05, 0.95);
        for mu in [0.4_f64, 3.0, 7.5, 40.0] {
            // No estimation excess: the exact conditional Poisson.
            let (lo, hi) = poisson_moment_matched_interval(mu, mu, p_lo, p_hi).unwrap();
            let content = poisson_moment_matched_content(mu, mu, lo, hi).unwrap();
            let want = count_set_mass((-mu).exp(), |k| mu / k, lo, hi);
            let tol = 2.0 * f64::EPSILON * (16.0 * mu + hi + 1.0);
            assert!(
                (content - want).abs() <= tol,
                "Poisson(mu={mu}) band [{lo},{hi}] content {content} vs pmf mass {want}"
            );
            assert!(content > p_hi - p_lo, "Poisson(mu={mu}) content {content}");

            // An estimation excess: the Gamma–Poisson NB with θ_eff = μ²/excess.
            let excess = 0.5 * mu;
            let total_var = mu + excess;
            let theta_eff = mu * mu / excess;
            let prob = theta_eff / (theta_eff + mu);
            let (lo, hi) = poisson_moment_matched_interval(mu, total_var, p_lo, p_hi).unwrap();
            let content = poisson_moment_matched_content(mu, total_var, lo, hi).unwrap();
            let want = count_set_mass(
                prob.powf(theta_eff),
                |k| (k - 1.0 + theta_eff) / k * (1.0 - prob),
                lo,
                hi,
            );
            // `F(hi)` and `F(lo − 1)` are two `beta_reg` calls; the reference's
            // `prob^θ` start carries `ε·(1 + θ·|ln prob|)` and each recurrence step
            // four roundings (one in `1 − prob`, relative `ε/(1 − prob)`).
            let upper_cdf = beta_reg(theta_eff, hi + 1.0, prob);
            let below_cdf = if lo >= 1.0 { beta_reg(theta_eff, lo, prob) } else { 0.0 };
            let nb_tol = upper_cdf * beta_reg_relative_error(theta_eff, hi + 1.0, prob)
                + below_cdf * beta_reg_relative_error(theta_eff, lo.max(1.0), prob)
                + want
                    * f64::EPSILON
                    * (1.0 + theta_eff * prob.ln().abs() + (hi + 1.0) * (4.0 + 1.0 / (1.0 - prob)));
            assert!(
                (content - want).abs() <= nb_tol,
                "NB(mu={mu}, theta={theta_eff}) band [{lo},{hi}] content {content} vs {want}, tol {nb_tol}"
            );
            assert!(content > p_hi - p_lo, "NB(mu={mu}) content {content}");

            // The Negative-Binomial family reads the same law.
            let theta = 4.0;
            let total_var = mu + mu * mu / theta + excess;
            let (lo, hi) =
                negative_binomial_moment_matched_interval(mu, theta, total_var, p_lo, p_hi).unwrap();
            let content =
                negative_binomial_moment_matched_content(mu, theta, total_var, lo, hi).unwrap();
            assert_eq!(
                content,
                CountPredictive::negative_binomial(mu, theta, total_var)
                    .unwrap()
                    .content(lo, hi)
            );
            assert!(content > p_hi - p_lo, "NB family (mu={mu}) content {content}");
        }
        // The point mass at zero carries its whole mass on the band `[0, 0]`.
        assert_eq!(poisson_moment_matched_content(0.0, 0.0, 0.0, 0.0), Some(1.0));
        assert_eq!(
            negative_binomial_moment_matched_content(0.0, 1.5, 0.0, 0.0, 0.0),
            Some(1.0)
        );
        assert!(poisson_moment_matched_content(2.0, 1.0, 0.0, 4.0).is_none());
    }

    #[test]
    fn tweedie_band_content_holds_the_zero_atom_when_the_band_starts_at_zero() {
        // μ = 0.3, φ = 1, p = 1.5: λ = √0.3 / 0.5 ≈ 1.10, zero atom e^{−λ} ≈ 0.33,
        // above the 0.05 lower tail mass, so the band starts at zero and holds the
        // whole atom: its content is `F(hi) = p_hi`, a surplus of `p_lo` over the
        // level.
        let (mu, phi, power) = (0.3_f64, 1.0_f64, 1.5_f64);
        let (p_lo, p_hi) = (0.05, 0.95);
        let total_var = phi * mu.powf(power);
        let (lo, hi) =
            tweedie_moment_matched_interval(mu, phi, power, total_var, p_lo, p_hi).unwrap();
        assert_eq!(lo, 0.0);
        let content =
            tweedie_moment_matched_content(mu, phi, power, total_var, lo, hi).unwrap();
        assert_eq!(content, tweedie_cdf_at(hi, mu, phi, power));
        assert!(content > p_hi - p_lo, "content {content}");
        // A mean whose atom is below the lower tail mass: both edges on the
        // continuous part, content `F(hi) − F(lo)`.
        let mu = 6.0_f64;
        let total_var = phi * mu.powf(power);
        let (lo, hi) =
            tweedie_moment_matched_interval(mu, phi, power, total_var, p_lo, p_hi).unwrap();
        assert!(lo > 0.0);
        let content =
            tweedie_moment_matched_content(mu, phi, power, total_var, lo, hi).unwrap();
        assert_eq!(
            content,
            tweedie_cdf_at(hi, mu, phi, power) - tweedie_cdf_at(lo, mu, phi, power)
        );
        assert_eq!(
            tweedie_moment_matched_content(0.0, phi, power, 0.0, 0.0, 0.0),
            Some(1.0)
        );
    }
}

#[cfg(test)]
mod gamma_quantile_small_shape_lower_tail_tests;
