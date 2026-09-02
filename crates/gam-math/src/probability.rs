use libm::{erf, erfc};
use statrs::function::{
    beta::{beta_reg, inv_beta_reg, ln_beta},
    gamma::gamma_ur,
};

const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

/// Quantile (inverse CDF) of a Beta distribution with shape parameters `a > 0`
/// and `b > 0` at probability `p`: the value `x in [0, 1]` with
/// `I_x(a, b) = p`, where `I` is the regularized incomplete beta.
///
/// `p <= 0` maps to the support floor and `p >= 1` to the support ceiling. A
/// non-finite or non-positive shape yields `NaN`.
pub fn beta_quantile(p: f64, a: f64, b: f64) -> f64 {
    if !(a.is_finite() && a > 0.0 && b.is_finite() && b > 0.0) {
        return f64::NAN;
    }
    if !p.is_finite() || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    match lower_tail_beta_quantile(p, a, b) {
        Some(x) => x,
        None => inv_beta_reg(a, b, p),
    }
}

/// `Beta⁻¹(p; a, b)` on the branch where the answer is small enough for the
/// ascending series to be exact, or `None` when it is not.
///
/// `inv_beta_reg` converges on an ABSOLUTE tolerance in `x`, so it cannot
/// resolve a quantile below about `1e-16`: it stalls and returns a number in
/// the `1e-17..1e-19` band unrelated to the answer. That band is not exotic —
/// it is the ordinary lower tail of a beta-regression predictive interval
/// whenever the mean is small. For `Beta(0.04, 3.96)` at `p = 0.025`, the
/// shapes a mean of `0.01` with a fifth of the Bernoulli variance produces, it
/// returned `6.7e-18` where the truth is `1.5e-41` (#2528).
///
/// The lower tail has a convergent ascending series,
///
/// ```text
/// I_x(a,b) = x^a / B(a,b) · S(x),   S(x) = Σ_{k≥0} c_k x^k,
/// c_k = (1−b)_k / (k!·(a+k)),       c_0 = 1/a
/// ```
///
/// whose leading term inverts in closed form to
/// `x₀ = exp([ln p + ln a + ln B(a,b)] / a)`. Refining it in `y = ln x` rather
/// than in `x` is what removes the floor: the answer's own variable becomes the
/// iteration variable, so an absolute step tolerance in `y` is a RELATIVE
/// tolerance in `x` and there is nothing to stall against. The iteration is
/// also better conditioned than the one it replaces —
/// `G(y) = ln I_{e^y}(a,b) − ln p` has `G′(y) = a + x·S′(x)/S(x) → a`, a
/// constant, where the `x`-space derivative `∂I/∂x` spans hundreds of orders
/// over the same range.
///
/// Underflow then becomes something the function can state rather than paper
/// over: a true quantile below `f64::MIN_POSITIVE` reaches `exp(y) = 0`, which
/// is the correctly rounded answer, instead of a spurious positive floor a
/// caller cannot distinguish from a resolved bound.
///
/// The branch condition is `x·max(1, b) ≤ ½`, which is derived rather than
/// tuned. The term ratio is `|x·(k+1−b)/(k+1)|·(a+k)/(a+k+1)`, and
/// `|k+1−b| ≤ (k+1)·max(1, b)` for every `k ≥ 0`, so the condition bounds every
/// ratio by `½` and the series reaches `f64` resolution in at most
/// [`BETA_SERIES_MAX_TERMS`] terms. It is the same boundary, for the same
/// reason, that `crates/gam-terms/src/basis/polylog.rs` uses for its own
/// ascending series.
fn lower_tail_beta_quantile(p: f64, a: f64, b: f64) -> Option<f64> {
    let ln_b = ln_beta(a, b);
    if !ln_b.is_finite() {
        return None;
    }
    // Leading-order inverse: `I_x ≈ x^a / (a·B(a,b))` as `x → 0`.
    let mut y = (p.ln() + a.ln() + ln_b) / a;
    if !y.is_finite() {
        return None;
    }
    // Reject before iterating if the seed is outside the series branch. The
    // seed underestimates `x` for `b < 1` and overestimates it for `b > 1`, by
    // a factor that is itself `1 + O(x)`, so a seed comfortably inside the
    // branch keeps every iterate inside it.
    let ratio_bound = (0.5_f64).ln() - b.max(1.0).ln();
    if !(y <= ratio_bound) {
        return None;
    }
    let ln_p = p.ln();
    for _ in 0..BETA_NEWTON_MAX_STEPS {
        let x = y.exp();
        if x * b.max(1.0) > 0.5 {
            return None;
        }
        let (sum, derivative_sum) = beta_ascending_series(x, a, b)?;
        if !(sum.is_finite() && sum > 0.0 && derivative_sum.is_finite()) {
            return None;
        }
        // `G(y) = a·y − ln B(a,b) + ln S(e^y) − ln p`.
        let g = a * y - ln_b + sum.ln() - ln_p;
        let g_prime = a + x * derivative_sum / sum;
        if !(g.is_finite() && g_prime.is_finite() && g_prime > 0.0) {
            return None;
        }
        let step = g / g_prime;
        if !step.is_finite() {
            return None;
        }
        y -= step;
        // Absolute in `y` is relative in `x`, which is the whole point.
        if step.abs() <= f64::EPSILON * y.abs().max(1.0) {
            break;
        }
    }
    let x = y.exp();
    if x.is_finite() && (0.0..=1.0).contains(&x) {
        Some(x)
    } else {
        None
    }
}

/// `(S(x), S′(x))` for `S(x) = Σ_{k≥0} (1−b)_k · x^k / (k!·(a+k))`.
///
/// Accumulated by the ratio `t_{k+1} = t_k·(k+1−b)/(k+1)` on the Pochhammer
/// factor, so no factorial or gamma is formed. `None` if the guard term count
/// is exhausted, which the caller's branch condition makes unreachable.
fn beta_ascending_series(x: f64, a: f64, b: f64) -> Option<(f64, f64)> {
    let mut pochhammer_over_factorial = 1.0_f64;
    let mut power = 1.0_f64;
    let mut sum = 1.0 / a;
    let mut derivative_sum = 0.0_f64;
    for k in 1..=BETA_SERIES_MAX_TERMS {
        let kf = k as f64;
        pochhammer_over_factorial *= (kf - b) / kf;
        let coefficient = pochhammer_over_factorial / (a + kf);
        // `power` holds `x^{k-1}` here, which is what `S′` wants.
        derivative_sum += kf * coefficient * power;
        power *= x;
        let term = coefficient * power;
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            return Some((sum, derivative_sum));
        }
    }
    None
}

/// `I_x(a,b)` from `ln(x)`, retaining a representable result when `x` itself
/// underflows.
///
/// The ordinary `beta_reg(a,b,x)` interface necessarily loses every result
/// whose beta argument is below the smallest subnormal, even when the
/// regularized integral is much larger because `a < 1`. On the derived
/// ascending-series branch, `x` appears only in the well-scaled correction
/// `S(x)` while its leading power stays in log space:
///
/// `ln I_x(a,b) = a·ln(x) − ln B(a,b) + ln S(x)`.
///
/// The same term-ratio proof used by [`lower_tail_beta_quantile`] supplies the
/// branch boundary. Outside that boundary, the ordinary regularized-beta
/// implementation receives a representable argument and remains the canonical
/// general evaluator.
fn regularized_beta_lower_from_log_x(log_x: f64, a: f64, b: f64) -> f64 {
    if !(a.is_finite() && a > 0.0 && b.is_finite() && b > 0.0)
        || log_x.is_nan()
        || log_x > 0.0
    {
        return f64::NAN;
    }
    if log_x == 0.0 {
        return 1.0;
    }
    if log_x == f64::NEG_INFINITY {
        return 0.0;
    }

    let series_limit = (0.5_f64).ln() - b.max(1.0).ln();
    if log_x <= series_limit {
        let x = log_x.exp();
        let Some((sum, _)) = beta_ascending_series(x, a, b) else {
            return f64::NAN;
        };
        let log_beta = ln_beta(a, b);
        if !(sum.is_finite() && sum > 0.0 && log_beta.is_finite()) {
            return f64::NAN;
        }
        return (a * log_x - log_beta + sum.ln()).exp();
    }

    beta_reg(a, b, log_x.exp())
}

/// `ln(1 / (1 + exp(log_ratio)))` without overflowing or rounding a
/// representable small unit fraction to zero.
#[inline]
fn log_reciprocal_one_plus_exp(log_ratio: f64) -> f64 {
    if log_ratio <= 0.0 {
        -log_ratio.exp().ln_1p()
    } else {
        -log_ratio - (-log_ratio).exp().ln_1p()
    }
}

/// Guard term count for [`beta_ascending_series`]. The caller's `x·max(1,b) ≤ ½`
/// branch bounds every term ratio by `½`, so the series reaches one ulp of an
/// `O(1/a)` partial sum in at most `53` terms; this is the non-convergence
/// guard, not the expected count.
const BETA_SERIES_MAX_TERMS: usize = 128;

/// Guard step count for the log-space Newton. From a seed whose relative error
/// is `O(x)` the iteration is quadratic, so it converges in two or three steps
/// over the whole branch; this is the non-convergence guard.
const BETA_NEWTON_MAX_STEPS: usize = 32;

/// The part of `x·x` that `f64` cannot hold: `x² = x*x + square_residual(x)`,
/// exactly, for every `x` whose square neither overflows nor goes subnormal.
///
/// This exists because of what `exp` does to a squared argument. Rounding
/// `x*x` perturbs it by at most `ulp(x²)/2` — a RELATIVE perturbation of
/// `ε/2`, which is unremarkable on its own. But `exp` converts a relative
/// perturbation `δ` of its ARGUMENT into a relative perturbation `x²·δ` of
/// its RESULT, so `exp(x*x)` carries `x²·ε/2` relative error: `3.7e-14` at
/// `x = 26`, and `7.7e-14` at the `x ≈ 37` where `φ(x)` finally underflows.
/// That is two orders worse than the `exp` evaluation's own rounding, and it
/// is the error `erfcx` and `normal_pdf` were both actually delivering.
///
/// The residual is the whole of that discarded term and is itself exactly
/// representable (Dekker's two-product theorem, in its one-FMA form), so
/// `exp(x²) = exp(x*x)·exp(residual)` and `exp(residual) = 1 + residual` to
/// `O(residual²)` — below `1e-27` over the entire domain either caller uses.
/// One multiply by `1 + residual` therefore buys back every digit, and the
/// callers below apply it fused so the correction itself costs one more
/// rounding and nothing else.
///
/// `mul_add` is a single instruction wherever FMA is in the baseline ISA
/// (aarch64, and x86-64 built with `+fma`); on a baseline x86-64 build it is
/// a `glibc` call, measured at ~2.5 ns. Against `erfcx`'s 38 ns that is 9%;
/// against `normal_pdf`'s 6.2 ns it is 40% of a function that is nowhere the
/// bottleneck of a row loop that also assembles a design row and a Hessian
/// block. Both callers guard the pathological arguments BEFORE calling this,
/// so it never has to defend `±∞` (whose residual would be `NaN`).
#[inline]
fn square_residual(x: f64, rounded_square: f64) -> f64 {
    x.mul_add(x, -rounded_square)
}

/// Standard normal PDF phi(x).
///
/// The squared argument is carried exactly (see `square_residual`); without
/// that, `exp(-½·fl(x*x))` degrades like `x²·ε/2` and reaches `5.7e-14`
/// relative before `φ` underflows, against the `3.3e-16` it holds with.
#[inline]
pub fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    let rounded_square = x * x;
    let head = INV_SQRT_2PI * (-0.5 * rounded_square).exp();
    if head == 0.0 || head.is_nan() {
        // The pdf underflowed or `x` was `±∞` (head `0`), or `x` was `NaN`.
        // Neither admits a relative correction, and `±∞` would feed the
        // residual an `∞ − ∞`; return the limit the plain form gives.
        return head;
    }
    let residual = square_residual(x, rounded_square);
    head.mul_add(-0.5 * residual, head)
}

/// Standard normal CDF Phi(x) evaluated via the exact special-function identity
///
///   Phi(x) = 0.5 * erfc(-x / sqrt(2)).
///
/// This is the exact Gaussian CDF semantics used throughout the codebase. The
/// numerical `erfc` implementation may use internal approximations, but the
/// returned function is the standard normal CDF itself rather than a separate
/// polynomial surrogate surface.
#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Two-sided standard-normal probability `P(|Z| ≥ |z|)`.
///
/// The exact symmetric identity is `erfc(|z|/√2)`. Evaluating that identity
/// directly avoids both the cancellation in `2·(1 − Φ(|z|))` and an
/// unnecessary rounding from multiplying a one-sided tail by two.
#[inline]
pub fn normal_two_sided_probability(z: f64) -> f64 {
    erfc(z.abs() / std::f64::consts::SQRT_2)
}

/// Two-sided Student-t probability `P(|T_ν| ≥ |t|)`.
///
/// For finite `ν > 0`,
///
/// `P(|T_ν| ≥ |t|) = I_x(ν/2, 1/2)`, `x = ν / (ν + t²)`.
///
/// Neither `t²` nor `x` is formed directly. Their ratio is carried as
/// `ln(t²/ν)`, and the regularized beta receives `ln(x)`. This matters beyond
/// avoiding overflow: for `ν = 1` and `t = f64::MAX`, `x` underflows to zero
/// although the Cauchy tail is still a representable subnormal. The log-beta
/// series preserves that probability. Invalid degrees of freedom produce
/// `NaN`; infinite statistics map to the exact limiting probability zero.
pub fn student_t_two_sided_probability(t: f64, degrees_of_freedom: f64) -> f64 {
    let half_df = 0.5 * degrees_of_freedom;
    if t.is_nan()
        || !(degrees_of_freedom.is_finite()
            && degrees_of_freedom > 0.0
            && half_df > 0.0)
    {
        return f64::NAN;
    }
    if t.is_infinite() {
        return 0.0;
    }

    let log_t_squared_over_df = 2.0 * t.abs().ln() - degrees_of_freedom.ln();
    let log_x = log_reciprocal_one_plus_exp(log_t_squared_over_df);
    regularized_beta_lower_from_log_x(log_x, half_df, 0.5)
}

/// Chi-squared survival probability `P(X_ν > statistic)`.
///
/// Uses the regularized upper incomplete gamma directly instead of
/// reconstructing a small tail as `1 − P(ν/2, statistic/2)`.
pub fn chi_square_sf(statistic: f64, degrees_of_freedom: f64) -> f64 {
    let half_df = 0.5 * degrees_of_freedom;
    if statistic.is_nan()
        || statistic < 0.0
        || !(degrees_of_freedom.is_finite()
            && degrees_of_freedom > 0.0
            && half_df > 0.0)
    {
        return f64::NAN;
    }
    if statistic == 0.0 {
        return 1.0;
    }
    if statistic == f64::INFINITY {
        return 0.0;
    }
    gamma_ur(half_df, 0.5 * statistic)
}

/// One `λ_j · χ²_{h_j}` term of a linear combination of independent
/// chi-squares, with the weight's SIGN and the term's degrees of freedom both
/// carried explicitly.
///
/// Two things separate this from the `&[f64]` weight list
/// [`weighted_chi_square_sf`] takes, and each of them is a distribution the
/// one-degree-of-freedom non-negative form cannot express:
///
/// * **A negative weight makes a RATIO a tail.** `P(A/B > t)` for independent
///   non-negative `A`, `B` is `P(A − tB > 0)`, so every F-shaped reference —
///   any statistic whose scale was estimated from the same data — is a
///   *signed* combination evaluated at zero. The classical `F_{a,b}` is the
///   two-term case `λ = (1, −t·a/b)`, `h = (a, b)`.
/// * **A multiplicity is not `h` copies of a weight.** It is, mathematically,
///   but the Imhof integrand costs one `atan` and one `ln` per TERM, and a
///   residual sum of squares carries `n − p` unit weights. Folding them into
///   one term with `h = n − p` is what makes an `n`-sized reference cost the
///   same as a `p`-sized one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedChiSquareTerm {
    /// `λ_j`, of either sign. A zero weight contributes nothing and is dropped.
    pub weight: f64,
    /// `h_j > 0`. Real rather than integral: a two-moment summary of a spectrum
    /// is a chi-square with a fractional shape, and this type is what carries it.
    pub degrees_of_freedom: f64,
}

/// [`signed_weighted_chi_square_sf`] at a caller-chosen absolute accuracy,
/// returning the bound actually achieved alongside the value.
///
/// # Method
///
/// Imhof's (1961) inversion in its general central form, of which the
/// non-negative unit-`h` case documented on [`weighted_chi_square_sf`] is the
/// specialization:
///
/// ```text
/// P(Q > x) = 1/2 + (1/π) ∫_0^∞ sin θ(u) / (u ρ(u)) du,
/// θ(u) = ½ Σ_j h_j arctan(λ_j u) − ½ x u,
/// ρ(u) = Π_j (1 + λ_j² u²)^{h_j/4}.
/// ```
///
/// Nothing in the derivation asks `λ_j > 0` — `arctan` is odd and `λ²` is even,
/// so a negative weight simply turns its part of the phase the other way.
///
/// # Two truncation bounds, because one of them stops working at `x = 0`
///
/// The oscillatory bound `16/(x·U·ρ(U))` documented on
/// [`weighted_chi_square_sf`] divides by `x`, and the ratio references this
/// signed form exists for are evaluated at exactly `x = 0`, where the phase
/// stops turning at all: `θ(u) → (π/4)·Σ_j h_j·sgn(λ_j)`, a constant. There is
/// no oscillation left to cancel, so the alternating-series argument yields
/// nothing.
///
/// What replaces it is the AMPLITUDE, which the same `x = 0` makes strong
/// rather than weak. For `u ≥ U` and `t = u/U ≥ 1`,
/// `(1 + λ²u²)/(1 + λ²U²) ≥ (1 + t²)/2 ≥ t` on every term ACTIVE at `U`
/// (`|λ_j|·U ≥ 1`) and `≥ 1` on the rest, so `ρ(u) ≥ ρ(U)·t^{H/4}` with
/// `H = Σ_{active} h_j` and
///
/// ```text
/// |tail(U)| ≤ ∫_U^∞ du/(u ρ(u)) ≤ 4 / (H · ρ(U)).
/// ```
///
/// This is a bound on the answer, not a guess about it, and it is the CHEAP
/// one exactly where the oscillatory bound is unavailable: a ratio reference
/// carries the residual `χ²_{n−p}`, so `H` is of order `n` and `ρ` grows like
/// `U^{n/2}` — a handful of panels. Both bounds are evaluated and the smaller
/// is taken, which also strictly improves the non-negative case at small `x`,
/// where `16/(x·U·ρ)` is what used to make the sweep long.
///
/// # Phase monotonicity, generalized
///
/// The oscillatory bound is valid only past the point where `|θ′| ≥ x/4`. With
/// mixed signs `φ′(u) = ½ Σ_j h_j λ_j/(1 + λ_j²u²)` is no longer monotone in
/// `u`, so the test is applied to `½ Σ_j h_j |λ_j|/(1 + λ_j²u²)` — an upper
/// bound on `|φ′|` that IS decreasing, hence a condition at `U` that holds for
/// every `u ≥ U`. On non-negative weights the two expressions coincide.
///
/// # Exact special cases
///
/// * no nonzero weight — `Q ≡ 0`;
/// * all weights positive and `x ≤ 0`, or all negative and `x ≥ 0` — the
///   inequality is decided by the support;
/// * all weights bit-identical — `Q = λ·χ²_{Σh}` exactly, on either sign.
///
/// Returns `NaN` if any weight is non-finite, if any degrees-of-freedom is not
/// finite and positive, or if `statistic` is `NaN`.
pub fn signed_weighted_chi_square_sf_to_tolerance(
    terms: &[WeightedChiSquareTerm],
    statistic: f64,
    absolute_tolerance: f64,
) -> (f64, f64) {
    let tolerance = if absolute_tolerance.is_finite() && absolute_tolerance > 0.0 {
        absolute_tolerance
    } else {
        WEIGHTED_CHI_SQUARE_TOLERANCE
    };
    if statistic.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let mut active = Vec::with_capacity(terms.len());
    for term in terms {
        if !term.weight.is_finite()
            || !(term.degrees_of_freedom.is_finite() && term.degrees_of_freedom > 0.0)
        {
            return (f64::NAN, f64::NAN);
        }
        if term.weight != 0.0 {
            active.push(*term);
        }
    }
    if active.is_empty() {
        // `Q` is identically zero: it exceeds a negative threshold with
        // certainty and a non-negative one never.
        return (if statistic < 0.0 { 1.0 } else { 0.0 }, 0.0);
    }
    let all_positive = active.iter().all(|term| term.weight > 0.0);
    let all_negative = active.iter().all(|term| term.weight < 0.0);
    if all_positive && statistic <= 0.0 {
        // `Q > 0` almost surely once one weight is positive.
        return (1.0, 0.0);
    }
    if all_negative && statistic >= 0.0 {
        // `Q < 0` almost surely once every weight is negative.
        return (0.0, 0.0);
    }
    let first = active[0].weight;
    if active.iter().all(|term| term.weight == first) {
        let total_df: f64 = active.iter().map(|term| term.degrees_of_freedom).sum();
        // `P(λ·χ² > x)` is the χ² upper tail at `x/λ` for `λ > 0` and the LOWER
        // tail there for `λ < 0`, because dividing by a negative number turns
        // the inequality around.
        let scaled = statistic / first;
        let tail = if first > 0.0 {
            chi_square_sf(scaled, total_df)
        } else {
            1.0 - chi_square_sf(scaled, total_df)
        };
        return (tail, 0.0);
    }
    imhof_survival(&active, statistic, tolerance)
}

/// Default absolute accuracy [`weighted_chi_square_sf`] certifies on its Imhof
/// truncation. It is four orders below the smallest probability any consumer
/// of a survival function resolves in practice and eleven below one, so the
/// truncation is never the term that limits a reported tail.
pub const WEIGHTED_CHI_SQUARE_TOLERANCE: f64 = 1e-11;

/// Gauss-Legendre nodes and weights on `[-1, 1]`, 16 points. A 16-node rule is
/// exact through degree 31, which is far beyond the smooth amplitude
/// `1/(u ρ(u))` over one phase period; the panel width, not the node count, is
/// what resolves the oscillation.
const GAUSS_LEGENDRE_16: [(f64, f64); 8] = [
    (0.095_012_509_837_637_44, 0.189_450_610_455_068_64),
    (0.281_603_550_779_258_9, 0.182_603_415_044_923_64),
    (0.458_016_777_657_227_37, 0.169_156_519_395_002_65),
    (0.617_876_244_402_643_8, 0.149_595_988_816_576_7),
    (0.755_404_408_355_003, 0.124_628_971_255_534_07),
    (0.865_631_202_387_831_8, 0.095_158_511_682_492_6),
    (0.944_575_023_073_232_6, 0.062_253_523_938_647_456),
    (0.989_400_934_991_649_9, 0.027_152_459_411_754_176),
];

/// Imhof's integrand `sin θ(u) / (u ρ(u))` with the `u → 0` limit folded in.
#[inline]
fn imhof_integrand(terms: &[WeightedChiSquareTerm], statistic: f64, u: f64) -> f64 {
    if u == 0.0 {
        let mean: f64 = terms
            .iter()
            .map(|term| term.weight * term.degrees_of_freedom)
            .sum();
        return 0.5 * (mean - statistic);
    }
    let mut phase = -0.5 * statistic * u;
    let mut log_rho = 0.0;
    for term in terms {
        let wu = term.weight * u;
        phase += 0.5 * term.degrees_of_freedom * wu.atan();
        log_rho += 0.25 * term.degrees_of_freedom * wu.mul_add(wu, 1.0).ln();
    }
    phase.sin() / (u * log_rho.exp())
}

/// `ln ρ(u)`, the Imhof amplitude exponent.
#[inline]
fn imhof_log_rho(terms: &[WeightedChiSquareTerm], u: f64) -> f64 {
    terms
        .iter()
        .map(|term| {
            let wu = term.weight * u;
            0.25 * term.degrees_of_freedom * wu.mul_add(wu, 1.0).ln()
        })
        .sum()
}

/// `½ Σ_j h_j|w_j|/(1 + w_j²u²)`, a DECREASING upper bound on the magnitude of
/// the non-linear part of the phase's own derivative.
///
/// The oscillatory truncation bound is valid only past the point where the
/// phase is monotone with `|θ'| ≥ x/4`, which needs `|φ'(u)| ≤ x/4` for every
/// `u` past the truncation point rather than at it. With mixed-sign weights
/// `φ'` is not monotone, so the test is applied to this bound instead; on
/// non-negative weights the two are the same expression.
#[inline]
fn imhof_phase_slack(terms: &[WeightedChiSquareTerm], u: f64) -> f64 {
    terms
        .iter()
        .map(|term| {
            let wu = term.weight * u;
            0.5 * term.degrees_of_freedom * term.weight.abs() / wu.mul_add(wu, 1.0)
        })
        .sum()
}

/// `4/(H·ρ(U))`, the AMPLITUDE truncation bound, with
/// `H = Σ_{|w_j|U ≥ 1} h_j` the degrees of freedom already active at `U`.
///
/// Valid unconditionally — it bounds `∫_U^∞ du/(u ρ(u))` and never looks at the
/// phase — and it is the only bound available at `statistic = 0`, where the
/// oscillatory one divides by zero. `None` when nothing is active yet, since
/// `ρ` is then still flat and there is no decay to integrate against.
/// Panel width that resolves the integrand's AMPLITUDE, as opposed to its
/// phase.
///
/// The phase rule below sizes a panel so it sweeps at most one oscillation.
/// That is necessary and it is not sufficient: `1/(u ρ(u))` has structure of
/// its own, on the scale `1/|λ|` where `(1 + λ²u²)^{h/4}` turns over, and a
/// panel far wider than that scale is a 16-node rule aliasing a factor it never
/// sampled. The two rules coincide only when the phase happens to turn at the
/// same rate the amplitude does — which is exactly what fails when the phase
/// rate is small: a ratio reference is evaluated at `statistic = 0`, and a
/// two-term `F`-shaped combination can have `Σ h_j|λ_j|` of order one while
/// `max_j|λ_j|` is also of order one, so `4π/Σ h|λ| ≈ 12` against an amplitude
/// scale of `1`. Measured on `F_{1,5}` at `f = 0.05`: the phase-only panel
/// returned `0.8319119` against the exact `0.8319122`, an error of `3.4e-7`
/// certified at `1e-11`.
///
/// The scale is not a guess. As a function of complex `u` the integrand's
/// nearest singularities are the branch points of `(1 + λ_j²u²)^{h_j/4}` at
/// `u = ±i/|λ_j|`; the closest is `d = 1/max_j|λ_j|`, and the `−xu/2` phase and
/// the `1/u` are entire and removable respectively. Gauss–Legendre with `N`
/// nodes on a panel of half-width `a` converges like `ϱ^{-2N}` in the Bernstein
/// parameter of the largest ellipse the integrand is analytic in, and an
/// ellipse with semi-minor axis `d` has `ϱ` solving `(ϱ − 1/ϱ)/2 = d/a`. So
/// asking `ϱ^{-2N} ≤ tolerance` fixes the half-width:
///
/// ```text
/// ϱ = tolerance^{-1/2N},   a = d / [(ϱ − 1/ϱ)/2].
/// ```
///
/// This is a RATE, not a certificate: the Bernstein bound also carries the
/// integrand's maximum modulus on that ellipse, which the ellipse touching the
/// branch point does not bound. The node count is what carries the margin, and
/// the margin is MEASURED rather than asserted —
/// `the_quadrature_resolves_the_amplitude_not_only_the_phase` compares against
/// a reference at a far finer panel and reads the achieved error off it.
///
/// A looser request buys a wider panel here, which is the right direction: the
/// consumer that derives its tolerance from the resolution of the statistic it
/// is scoring pays for what it asked for.
#[inline]
fn imhof_amplitude_panel(max_abs_weight: f64, tolerance: f64) -> f64 {
    let node_count = 2.0 * GAUSS_LEGENDRE_16.len() as f64;
    let bernstein = tolerance.recip().powf(0.5 / node_count);
    let semi_minor_ratio = 0.5 * (bernstein - bernstein.recip());
    if !(semi_minor_ratio > 0.0 && max_abs_weight > 0.0) {
        return f64::INFINITY;
    }
    2.0 / (max_abs_weight * semi_minor_ratio)
}

#[inline]
fn imhof_amplitude_bound(terms: &[WeightedChiSquareTerm], u: f64) -> Option<f64> {
    let active_df: f64 = terms
        .iter()
        .filter(|term| term.weight.abs() * u >= 1.0)
        .map(|term| term.degrees_of_freedom)
        .sum();
    (active_df > 0.0).then(|| 4.0 / (active_df * imhof_log_rho(terms, u).exp()))
}

/// Cost backstop on the Imhof panel sweep.
///
/// The truncation point `U` needed for a given bound scales as
/// `(16/(x·tol·C))^{2/(2+m)}` in the number `m` of weights that are *active*
/// (`w_j U ≳ 1`) there, and the panel count as `U·x/4π`. With three or more
/// comparable weights that count stays in the thousands for any statistic a
/// likelihood-ratio consumer produces, so this backstop is unreachable — it
/// exists for the one degenerate corner where it is not: two weights spread
/// over several orders of magnitude, with a large statistic, where the sweep
/// would otherwise run for tens of millions of panels to buy digits far below
/// the modelling error of any statistic being referenced against it. The
/// achieved bound is returned rather than discarded, so a caller that lands in
/// that corner can see it instead of inferring it.
///
/// The panel width is the smaller of the phase rule and
/// `imhof_amplitude_panel`, so the count above is a LOWER bound on what the
/// sweep costs. It moves the corner slightly closer without changing which
/// corner it is: the amplitude panel is `2/(|λ|_max·s(tol))`, independent of
/// the statistic, so it binds where the phase rate is small — and a small phase
/// rate is a small truncation point, which is the cheap end.
pub const IMHOF_MAX_PANELS: usize = 1 << 21;

fn imhof_survival(
    terms: &[WeightedChiSquareTerm],
    statistic: f64,
    tolerance: f64,
) -> (f64, f64) {
    // A panel has to resolve the WHOLE phase, not just the `−xu/2` half. The
    // total phase rate is bounded by `|θ'(u)| = |φ'(u) − x/2| ≤ (Σ h_j|w_j| +
    // |x|)/2` — `|φ'|` is largest at the origin, where it is `½ Σ h_j|w_j|` —
    // so a panel of `4π/(|x| + Σ h_j|w_j|)` sweeps at most one full oscillation
    // anywhere on the half-line. Sizing on `4π/x` alone is correct only in the
    // tail: at a small statistic that panel is enormous while the arctan part
    // of the phase still turns over on the scale `1/w_j`, and the 16-node rule
    // then aliases it (measured: a monotonicity violation of ~1e-5 at
    // `x ≈ 4e-4`). At `x = 0` — the ratio references — the arctan part is the
    // ONLY phase there is, and sizing on it is what keeps the rule honest.
    let rate: f64 = terms
        .iter()
        .map(|term| term.degrees_of_freedom * term.weight.abs())
        .sum();
    let phase_panel = 4.0 * std::f64::consts::PI / (statistic.abs() + rate);
    // ...and it has to resolve the AMPLITUDE as well; see
    // `imhof_amplitude_panel` for why the phase rule alone is not enough and
    // where the second scale comes from.
    let max_abs_weight = terms
        .iter()
        .map(|term| term.weight.abs())
        .fold(0.0_f64, f64::max);
    let panel = phase_panel.min(imhof_amplitude_panel(max_abs_weight, tolerance));
    let mut integral = 0.0_f64;
    let mut lower = 0.0_f64;
    let mut bound = f64::INFINITY;
    for _ in 0..IMHOF_MAX_PANELS {
        let upper = lower + panel;
        let half = 0.5 * (upper - lower);
        let mid = 0.5 * (upper + lower);
        let mut panel_value = 0.0;
        for &(node, weight) in &GAUSS_LEGENDRE_16 {
            let offset = half * node;
            panel_value += weight
                * (imhof_integrand(terms, statistic, mid + offset)
                    + imhof_integrand(terms, statistic, mid - offset));
        }
        integral += half * panel_value;
        lower = upper;
        // The amplitude bound holds unconditionally; the oscillatory one only
        // once the phase is monotone, and only for a positive statistic.
        // Whichever is available and smaller is the certified accuracy.
        bound = imhof_amplitude_bound(terms, lower).unwrap_or(f64::INFINITY);
        if statistic > 0.0 && imhof_phase_slack(terms, lower) <= 0.25 * statistic {
            let oscillatory =
                16.0 / (statistic * lower * imhof_log_rho(terms, lower).exp());
            bound = bound.min(oscillatory);
        }
        if bound <= tolerance {
            break;
        }
    }
    (
        (0.5 + integral / std::f64::consts::PI).clamp(0.0, 1.0),
        bound,
    )
}

/// Fisher-Snedecor survival probability `P(F_{d1,d2} > statistic)`.
///
/// The complementary regularized-beta identity is evaluated directly:
///
/// `I_x(d2/2, d1/2)`, `x = d2 / (d2 + d1·statistic)`.
///
/// The beta argument is derived in log space, so neither `d1·statistic` nor
/// the denominator can overflow before a representable tail is recovered.
pub fn fisher_snedecor_sf(
    statistic: f64,
    numerator_degrees_of_freedom: f64,
    denominator_degrees_of_freedom: f64,
) -> f64 {
    let beta_a = 0.5 * denominator_degrees_of_freedom;
    let beta_b = 0.5 * numerator_degrees_of_freedom;
    if statistic.is_nan()
        || statistic < 0.0
        || !(numerator_degrees_of_freedom.is_finite()
            && numerator_degrees_of_freedom > 0.0
            && denominator_degrees_of_freedom.is_finite()
            && denominator_degrees_of_freedom > 0.0
            && beta_a > 0.0
            && beta_b > 0.0)
    {
        return f64::NAN;
    }
    if statistic == 0.0 {
        return 1.0;
    }
    if statistic == f64::INFINITY {
        return 0.0;
    }

    let log_ratio = numerator_degrees_of_freedom.ln() + statistic.ln()
        - denominator_degrees_of_freedom.ln();
    let log_x = log_reciprocal_one_plus_exp(log_ratio);
    regularized_beta_lower_from_log_x(log_x, beta_a, beta_b)
}

/// Scaled complementary error function `erfcx(x) = exp(x²) · erfc(x)`,
/// specialized to the closed domain `x ∈ [0, +∞]`.
///
/// `+∞` maps to the exact limiting value `0`; `NaN` and negative inputs map to
/// `NaN` because they violate this restricted kernel's domain. For
/// `0 ≤ x < 26` the direct `exp(x²)·erfc(x)` form is finite. Beyond that point
/// a six-correction asymptotic expansion avoids overflow while retaining the
/// representable subnormal tail. At the switch, the first omitted term is
/// below `2e-17` relative to the leading term.
///
/// The direct branch carries `x²` exactly (see `square_residual`). Without
/// that correction the branch degraded like `x²·ε/2` — `1.4e-14` at `x = 10`,
/// `5.7e-14` by the top of its range — while the asymptotic branch that takes
/// over at `26` was already delivering `3e-16`. The seam was therefore a
/// 190-fold step DOWN in error at the point where the code switches to what
/// reads like the fallback, and the whole `[0, 26)` interval, where every
/// probit / Mills / log-CDF consumer actually lives, was the inaccurate side.
/// Both branches now hold `< 5e-16`, so the crossover is invisible.
#[inline]
pub fn erfcx_nonnegative(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x < 26.0 {
        // `x` is finite and in `[0, 26)`, so the square is exact-splittable and
        // `head` is finite and strictly positive (`erfc(26⁻) ≈ 1e-295`).
        let rounded_square = x * x;
        let head = rounded_square.exp() * erfc(x);
        head.mul_add(square_residual(x, rounded_square), head)
    } else {
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        // erfcx(x) ~ 1/(sqrt(pi)x) * sum_n (-1)^n (2n-1)!!/(2x^2)^n.
        // Horner form keeps the correction well scaled when `inv2` is tiny.
        let poly = 1.0
            + inv2
                * (-0.5
                    + inv2
                        * (0.75
                            + inv2
                                * (-1.875
                                    + inv2 * (6.5625 + inv2 * (-29.53125 + inv2 * 162.421875)))));
        inv * poly * INV_SQRT_PI
    }
}

/// Computes `log(1 - exp(-a))` for `a >= 0` without cancellation.
#[inline]
pub fn log1mexp_positive(a: f64) -> f64 {
    assert!(a >= 0.0, "log1mexp_positive requires a >= 0: a={a}");
    if a > core::f64::consts::LN_2 {
        (-(-a).exp()).ln_1p()
    } else if a > 0.0 {
        (-(-a).exp_m1()).ln()
    } else {
        f64::NEG_INFINITY
    }
}

// A finite binary64 is an integer multiple of 2^-1074. Its largest possible
// significand occupies bits 2045..=2097 on that lattice. Thirty-three limbs
// leave 14 carry bits, enough to sum at most 2^14-1 finite inputs exactly.
const EXACT_BINARY64_SUM_WORDS: usize = 33;
const EXACT_BINARY64_SUM_MAX_TERMS: usize = (1 << 14) - 1;
const _: () = assert!(EXACT_BINARY64_SUM_WORDS * 64 == 2112);

/// Why [`exact_binary64_sum_sign`] could not classify its finite exact sum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactBinary64SumSignError {
    /// One input was not a finite binary64.
    NonFiniteTerm { index: usize },
    /// The fixed exact accumulator's structural term bound was exceeded.
    TermCapacityExceeded { maximum: usize },
}

impl std::fmt::Display for ExactBinary64SumSignError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteTerm { index } => {
                write!(formatter, "exact binary64 sum term {index} is not finite")
            }
            Self::TermCapacityExceeded { maximum } => write!(
                formatter,
                "exact binary64 sum exceeds its structural {maximum}-term capacity"
            ),
        }
    }
}

impl std::error::Error for ExactBinary64SumSignError {}

/// Exact sign of a finite binary64 sum, independent of order and cancellation.
///
/// Every input is decoded as an integer significand on the common `2^-1074`
/// lattice. Positive and negative magnitudes accumulate into separate fixed
/// 2,112-bit unsigned integers; comparing those integers returns the sign of
/// the exact real sum, with no floating-point reduction and no tolerance.
///
/// At most 16,383 terms are admitted, the largest count whose worst-case carry
/// is structurally contained by the fixed accumulator.
pub fn exact_binary64_sum_sign(
    values: impl IntoIterator<Item = f64>,
) -> Result<std::cmp::Ordering, ExactBinary64SumSignError> {
    fn add_magnitude(
        accumulator: &mut [u64; EXACT_BINARY64_SUM_WORDS],
        value: f64,
    ) -> Result<(), ExactBinary64SumSignError> {
        let magnitude_bits = value.to_bits() & !(1_u64 << 63);
        let exponent_bits = ((magnitude_bits >> 52) & 0x7ff) as usize;
        let fraction = magnitude_bits & ((1_u64 << 52) - 1);
        let (significand, shift) = if exponent_bits == 0 {
            (fraction, 0usize)
        } else {
            ((1_u64 << 52) | fraction, exponent_bits - 1)
        };
        if significand == 0 {
            return Ok(());
        }

        let mut word = shift / 64;
        let offset = shift % 64;
        let (low_sum, low_carry) =
            accumulator[word].overflowing_add(significand << offset);
        accumulator[word] = low_sum;
        word += 1;

        let high = if offset == 0 {
            0
        } else {
            significand >> (64 - offset)
        };
        let (high_sum, high_carry) = accumulator[word].overflowing_add(high);
        let (high_sum, carry_carry) = high_sum.overflowing_add(u64::from(low_carry));
        accumulator[word] = high_sum;
        let mut carry = high_carry || carry_carry;
        word += 1;
        while carry {
            if word == EXACT_BINARY64_SUM_WORDS {
                return Err(ExactBinary64SumSignError::TermCapacityExceeded {
                    maximum: EXACT_BINARY64_SUM_MAX_TERMS,
                });
            }
            let (sum, next_carry) = accumulator[word].overflowing_add(1);
            accumulator[word] = sum;
            carry = next_carry;
            word += 1;
        }
        Ok(())
    }

    let mut positive = [0_u64; EXACT_BINARY64_SUM_WORDS];
    let mut negative = [0_u64; EXACT_BINARY64_SUM_WORDS];
    for (index, value) in values.into_iter().enumerate() {
        if index == EXACT_BINARY64_SUM_MAX_TERMS {
            return Err(ExactBinary64SumSignError::TermCapacityExceeded {
                maximum: EXACT_BINARY64_SUM_MAX_TERMS,
            });
        }
        if !value.is_finite() {
            return Err(ExactBinary64SumSignError::NonFiniteTerm { index });
        }
        let target = if value.is_sign_negative() {
            &mut negative
        } else {
            &mut positive
        };
        add_magnitude(target, value)?;
    }
    for index in (0..EXACT_BINARY64_SUM_WORDS).rev() {
        match positive[index].cmp(&negative[index]) {
            std::cmp::Ordering::Less => return Ok(std::cmp::Ordering::Less),
            std::cmp::Ordering::Greater => return Ok(std::cmp::Ordering::Greater),
            std::cmp::Ordering::Equal => {}
        }
    }
    Ok(std::cmp::Ordering::Equal)
}

/// Numerically stable signed log-sum-exp.  Given pairs
/// `(log|aⱼ|, sign(aⱼ))` (with `signs[j] ∈ {−1, 0, +1}`), returns
/// `(log|S|, sign(S))` for `S = Σⱼ signs[j]·exp(log_mags[j])`.  Positive
/// and negative magnitudes are first reduced together, after one common
/// log-space rescaling, with a twofold compensated sum. This avoids rounding
/// each same-sign subtotal through `ln` and `exp` before subtracting them — an
/// avoidable loss that is amplified in cancellation-conditioned derivative
/// cumulants. If the compensated residual lies inside its forward-error bound,
/// the function instead uses the two-subtotal log-domain difference
/// `log(|p − n|) = max(log p, log n) +
/// log1mexp(|log p − log n|)`. That branch retains differences between two input
/// logs even when their exponentials round to the same `f64`. When all signs are
/// zero or all magnitudes are `−∞`, returns `(NEG_INFINITY, 0.0)`.
///
/// A `+∞` log-magnitude denotes an infinite-magnitude term (`exp(+∞) = +∞`)
/// and dominates the sum: if it appears only with positive sign the result
/// is `(+∞, +1)`; only with negative sign, `(+∞, −1)` (a log-magnitude of
/// `+∞` with sign `−1` encodes the value `−∞`); with both signs the sum is
/// the indeterminate `+∞ − ∞`, returned as `(NaN, 0.0)`.  A `−∞`
/// log-magnitude is `exp(−∞) = 0` and is correctly dropped.
pub fn signed_log_sum_exp(log_mags: &[f64], signs: &[f64]) -> (f64, f64) {
    // Infinite-magnitude terms dominate any finite contribution, so resolve
    // them before the finite log-sum-exp reduction below. `−∞` log-magnitudes
    // are `exp(−∞) = 0` and need no special handling.
    let mut has_pos_inf = false;
    let mut has_neg_inf = false;
    for (idx, &lm) in log_mags.iter().enumerate() {
        if lm == f64::INFINITY {
            if signs[idx] > 0.0 {
                has_pos_inf = true;
            } else if signs[idx] < 0.0 {
                has_neg_inf = true;
            }
        }
    }
    match (has_pos_inf, has_neg_inf) {
        // P = +∞, N = +∞ ⇒ indeterminate +∞ − ∞.
        (true, true) => return (f64::NAN, 0.0),
        // P = +∞, N < ∞ ⇒ S = +∞.
        (true, false) => return (f64::INFINITY, 1.0),
        // N = +∞, P < ∞ ⇒ S = −∞, encoded as log-magnitude +∞ with sign −1.
        (false, true) => return (f64::INFINITY, -1.0),
        (false, false) => {}
    }

    let mut pos_max = f64::NEG_INFINITY;
    let mut neg_max = f64::NEG_INFINITY;
    for (idx, &lm) in log_mags.iter().enumerate() {
        if signs[idx] > 0.0 {
            pos_max = pos_max.max(lm);
        } else if signs[idx] < 0.0 {
            neg_max = neg_max.max(lm);
        }
    }

    if pos_max == f64::NEG_INFINITY && neg_max == f64::NEG_INFINITY {
        // Both partial sums are empty: no terms at all, all signs zero, or every
        // magnitude `−∞` (each `exp(−∞) = 0`). The signed sum is exactly `0`.
        return (f64::NEG_INFINITY, 0.0);
    }

    // First reduce the signed terms directly after one common scaling. `head`
    // plus `tail` is a twofold sum: TwoSum recovers every addition's exact
    // residual, so cancellation does not discard the low part of either
    // same-sign subtotal before the final subtraction.
    let common_max = pos_max.max(neg_max);
    let mut signed_head = 0.0_f64;
    let mut signed_tail = 0.0_f64;
    let mut absolute_scaled_sum = 0.0_f64;
    let mut finite_term_count = 0usize;
    for (idx, &lm) in log_mags.iter().enumerate() {
        if !lm.is_finite() || !(signs[idx] > 0.0 || signs[idx] < 0.0) {
            continue;
        }
        let magnitude = (lm - common_max).exp();
        let term = if signs[idx] > 0.0 {
            magnitude
        } else {
            -magnitude
        };
        let combined = signed_head + term;
        let shifted = combined - signed_head;
        let residual = (signed_head - (combined - shifted)) + (term - shifted);
        signed_head = combined;
        signed_tail += residual;
        absolute_scaled_sum += magnitude;
        finite_term_count += 1;
    }
    let signed_scaled_sum = signed_head + signed_tail;

    // Each scaled exponential and each accumulated residual contributes at most
    // one working-precision rounding. This conservative Wilkinson-style bound
    // decides from the operation count, rather than from a fitted threshold,
    // whether the linear-domain residual has a trustworthy sign and magnitude.
    // Below the bound, retain the input-log separation in the log-domain branch.
    let direct_error_bound =
        (finite_term_count as f64 + 2.0) * f64::EPSILON * absolute_scaled_sum;
    if signed_scaled_sum.abs() > direct_error_bound {
        return (
            common_max + signed_scaled_sum.abs().ln(),
            signed_scaled_sum.signum(),
        );
    }

    // When exponentiation itself cannot resolve the signed residual, reduce
    // positive and negative groups separately in log space. Their internal sums
    // are still twofold-compensated before taking the logarithm.
    let mut pos_sum = 0.0_f64;
    let mut pos_tail = 0.0_f64;
    let mut neg_sum = 0.0_f64;
    let mut neg_tail = 0.0_f64;
    for (idx, &lm) in log_mags.iter().enumerate() {
        if !lm.is_finite() {
            continue;
        }
        if signs[idx] > 0.0 {
            let term = (lm - pos_max).exp();
            let combined = pos_sum + term;
            let shifted = combined - pos_sum;
            pos_tail += (pos_sum - (combined - shifted)) + (term - shifted);
            pos_sum = combined;
        } else if signs[idx] < 0.0 {
            let term = (lm - neg_max).exp();
            let combined = neg_sum + term;
            let shifted = combined - neg_sum;
            neg_tail += (neg_sum - (combined - shifted)) + (term - shifted);
            neg_sum = combined;
        }
    }
    pos_sum += pos_tail;
    neg_sum += neg_tail;

    let log_pos = if pos_sum > 0.0 {
        pos_max + pos_sum.ln()
    } else {
        f64::NEG_INFINITY
    };
    let log_neg = if neg_sum > 0.0 {
        neg_max + neg_sum.ln()
    } else {
        f64::NEG_INFINITY
    };

    if log_neg == f64::NEG_INFINITY {
        return (log_pos, 1.0);
    }
    if log_pos == f64::NEG_INFINITY {
        return (log_neg, -1.0);
    }
    if log_pos > log_neg {
        let gap = log_pos - log_neg;
        (log_pos + log1mexp_positive(gap), 1.0)
    } else if log_neg > log_pos {
        let gap = log_neg - log_pos;
        (log_neg + log1mexp_positive(gap), -1.0)
    } else {
        (f64::NEG_INFINITY, 0.0)
    }
}

/// Numerically stable `ln Φ(x)` for the standard normal CDF. For `x ≥ 0`,
/// evaluates `ln(1 - 0.5 erfc(x/sqrt(2)))` with `ln_1p`, retaining the small
/// negative result after `Φ(x)` itself rounds to one. For `x < 0`, rewrites
/// `ln Φ(x) = −u² + ln(½·erfcx(u))`, `u = −x/√2`,
/// which preserves digits throughout the representable left tail without a
/// probability floor. Returns the corresponding IEEE limit at infinities and
/// propagates `NaN`.
#[inline]
pub fn normal_logcdf(x: f64) -> f64 {
    if x == f64::INFINITY {
        return 0.0;
    }
    if x == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        let (u, scaled_tail) = negative_normal_tail_components(x);
        negative_normal_logcdf_from_scaled_tail(u, scaled_tail)
    } else {
        let upper_tail = 0.5 * erfc(x / std::f64::consts::SQRT_2);
        (-upper_tail).ln_1p()
    }
}

/// Numerically stable `ln(1 − Φ(x)) = ln Φ(−x)` for the standard normal
/// survival function.  Delegates to `normal_logcdf(-x)` so the deep-right
/// tail benefits from the same `erfcx`-based representation.
#[inline]
pub fn normal_logsf(x: f64) -> f64 {
    normal_logcdf(-x)
}

/// Joint evaluation of `ln Φ(x)` and the Mills-ratio analogue
/// `φ(x) / Φ(x)`, signed for the symmetric branch.  Used by the latent
/// probit families where the inverse-link gradient needs the ratio and
/// the likelihood needs the log-CDF on the same `x`; computing both in
/// one call shares the `erfcx` evaluation that dominates the cost in the
/// deep tail.
#[inline]
pub fn signed_probit_logcdf_and_mills_ratio(x: f64) -> (f64, f64) {
    if x == f64::INFINITY {
        return (0.0, 0.0);
    }
    if x == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x < 0.0 {
        let (u, scaled_tail) = negative_normal_tail_components(x);
        (
            negative_normal_logcdf_from_scaled_tail(u, scaled_tail),
            SQRT_2_OVER_PI / scaled_tail,
        )
    } else {
        let upper_tail = 0.5 * erfc(x / std::f64::consts::SQRT_2);
        let cdf = 1.0 - upper_tail;
        let lambda = normal_pdf(x) / cdf;
        ((-upper_tail).ln_1p(), lambda)
    }
}

#[inline]
fn negative_normal_tail_components(x: f64) -> (f64, f64) {
    assert!(x.is_finite() && x < 0.0);
    let u = -x / std::f64::consts::SQRT_2;
    (u, erfcx_nonnegative(u))
}

#[inline]
fn negative_normal_logcdf_from_scaled_tail(u: f64, scaled_tail: f64) -> f64 {
    -u * u + scaled_tail.ln() - std::f64::consts::LN_2
}

/// Stable value and first four derivatives of `ln Φ(x)`.
///
/// The moderate regime uses the exact Mills-ratio recurrence, with the brackets
/// collected in `q = λ + x` once `x < 0` so that they do not cancel as `λ`
/// closes on `−x`. In the deep left tail, differentiating the Laplace continued
/// fraction
///
/// `φ(t)/Φ(-t) = t + 1/(t + 2/(t + 3/(...)))`, `t = -x`,
///
/// carries the small correction to `t` independently, so `f'' -> -1` and the
/// higher derivatives approach zero without subtracting nearly equal `f64`s.
/// In the right tail, signed log-magnitude sums preserve polynomially weighted
/// derivatives even when `φ(x)/Φ(x)` itself has rounded to zero.
#[inline]
pub fn normal_logcdf_derivatives(x: f64) -> [f64; 5] {
    if x.is_nan() {
        return [f64::NAN; 5];
    }
    if x == f64::INFINITY {
        return [0.0; 5];
    }
    if x == f64::NEG_INFINITY {
        return [f64::NEG_INFINITY, f64::INFINITY, -1.0, 0.0, 0.0];
    }

    const RIGHT_LOG_MAGNITUDE_SWITCH: f64 = 8.0;
    if x <= LEFT_CONTINUED_FRACTION_SWITCH {
        return normal_logcdf_derivatives_left_tail(x);
    }
    if x >= RIGHT_LOG_MAGNITUDE_SWITCH {
        return normal_logcdf_derivatives_right_tail(x);
    }

    let (log_cdf, lambda) = signed_probit_logcdf_and_mills_ratio(x);
    let x2 = x * x;
    if x < 0.0 {
        // Left of the origin the brackets below are collected in the SAME Mills
        // correction `q = λ + x` the continued-fraction branch carries, because
        // written in `λ` they cancel catastrophically long before the branch
        // ends. `λ(x) → −x` as `x → −∞`, so every term of, say,
        // `(x³−3x) + (7x²−4)λ + 12xλ² + 6λ³` grows like `|x|³` while their sum
        // decays: at `x = −4` they are `−52`, `456`, `−857`, `453` and add to
        // `−0.0023`, a cancellation of 380000 that costs eleven digits. In `q`
        // the same bracket is `−6q³ + 6xq² + (4−x²)q − x`, whose terms are
        // `−0.069`, `−1.22`, `−2.71`, `4` — a cancellation of 1847, three
        // orders milder. The reformulation is exact (`λ = q − x` substituted and
        // re-collected), costs the same flops, and buys 16–34x across the whole
        // branch: worst over `x ∈ [−4, 0]` falls from `4.5e−11` to `2.8e−12`.
        //
        // `q` itself is safe to form here: `λ/2 ≤ |x| ≤ 2λ` holds over most of
        // the range, so `λ + x` is EXACT by Sterbenz, and where it is not (`x`
        // near 0) `q` is the same size as `λ` and nothing cancels. That is the
        // whole reason the rewrite works — it moves the cancellation out of the
        // brackets and into a subtraction that has none.
        //
        // Past the origin `q → x` is no longer small, the `λ` form has nothing
        // to cancel (`λ → 0` and `x² − 1` dominates), and it is the more
        // accurate of the two — hence the sign test rather than a blanket swap.
        let q = lambda + x;
        let q2 = q * q;
        return [
            log_cdf,
            lambda,
            -lambda * q,
            lambda * (2.0 * q2 - x * q - 1.0),
            lambda * (-6.0 * q2 * q + 6.0 * x * q2 + (4.0 - x2) * q - x),
        ];
    }
    let lambda2 = lambda * lambda;
    let lambda3 = lambda2 * lambda;
    [
        log_cdf,
        lambda,
        -lambda * (x + lambda),
        lambda * (x2 - 1.0 + 3.0 * x * lambda + 2.0 * lambda2),
        -lambda
            * ((x * x2 - 3.0 * x) + (7.0 * x2 - 4.0) * lambda + 12.0 * x * lambda2 + 6.0 * lambda3),
    ]
}

#[derive(Clone, Copy)]
struct MillsCorrectionDerivatives {
    value: f64,
    first: f64,
    second: f64,
    third: f64,
}

/// `x` at or below which the left-tail Mills ratio is taken from the Laplace
/// continued fraction rather than from `erfcx`. Equivalently `t = −x ≥ 4`.
const LEFT_CONTINUED_FRACTION_SWITCH: f64 = -4.0;

/// The Laplace continued-fraction **correction** to the left-tail Mills ratio,
///
/// `q(t) = λ(−t) − t = 1/(t + 2/(t + 3/(...)))`,   `λ(x) = φ(x)/Φ(x)`,
///
/// together with its first three derivatives in `t`. Requires `t ≥ 4`.
///
/// `q` is the whole content of the left tail that is NOT the leading `t`: it
/// decays like `1/t − 2/t³ + 10/t⁵ − ...`, and every operation building it is
/// a division or an addition of positive quantities, so it carries full
/// relative precision no matter how small it gets. That is the property its
/// two consumers need, and it is why the correction is returned separately
/// instead of pre-added to `t`:
///
/// * [`normal_logcdf_derivatives_left_tail`] needs `f'' = −(1 + q')` and the
///   higher derivatives, which tend to `−1` and `0` and would be destroyed by
///   differencing nearly equal `f64`s.
/// * [`cone_boundary_log_factor_and_derivatives`] needs `∂corr/∂a = b − q(t)`,
///   which is the same statement one substitution away (#2306 §4).
///
/// Recovering `q` from a separately computed `λ` — `q = λ − t` — is exactly the
/// cancellation this exists to avoid, and it is not a small effect: at `t = 1e8`
/// it costs every significant digit, and past `t ≈ 2e8` it returns the wrong
/// SIGN. The reference itself has to be carried at ~120 decimal digits before it
/// reproduces what this recursion gives in binary64.
#[inline]
fn mills_correction_continued_fraction(t: f64) -> MillsCorrectionDerivatives {
    assert!(t.is_finite() && t >= 4.0);
    let mut q = MillsCorrectionDerivatives {
        value: 0.0,
        first: 0.0,
        second: 0.0,
        third: 0.0,
    };
    // The truncation error is damped by a product of the continued-fraction
    // sensitivities `n/(t + q)^2`, so the depth must be sized at `t = 4` — the
    // LEAST converged point of the domain, and the one the log-CDF branch sits
    // exactly on. Each successive derivative converges roughly 15x slower than
    // the last, because differentiating the recursion multiplies each level's
    // contribution by another factor of that same sensitivity. Measured against
    // a 60-digit reference at `t = 4`:
    //
    // ```text
    //            q         q'        q''       q'''
    //   32   1.9e-15    7.0e-14    1.4e-12    2.1e-11
    //   64   2.3e-23    1.4e-21    4.4e-20    1.0e-18
    // ```
    //
    // 32 levels is enough for the VALUE and nothing else: it leaves `q'''` — the
    // fourth log-CDF derivative — wrong in its eleventh digit. The depths that
    // first reach `1e-17` at `t = 4` are 41, 47, 53 and 60 for the four
    // channels, so 64 covers the worst of them with ~200x of margin, and the
    // requirement falls off fast enough (33 levels at `t = 6`, 24 at `t = 8`,
    // 12 at `t = 20`) that one constant sized for the edge is safe everywhere
    // above it. The extra levels are pure convergence — every step divides
    // positive quantities — so they cannot destabilise a large `t`.
    for n in (1..=64).rev() {
        let denominator = t + q.value;
        let inv_denominator = denominator.recip();
        let value = f64::from(n) / denominator;
        let denominator_first = 1.0 + q.first;
        let a = denominator_first * inv_denominator;
        let b = q.second * inv_denominator;
        let c = q.third * inv_denominator;
        q = MillsCorrectionDerivatives {
            value,
            first: -value * denominator_first / denominator,
            second: value * (2.0 * a * a - b),
            third: value * (-6.0 * a * a * a + 6.0 * a * b - c),
        };
    }
    q
}

#[inline]
fn normal_logcdf_derivatives_left_tail(x: f64) -> [f64; 5] {
    assert!(x.is_finite() && x <= LEFT_CONTINUED_FRACTION_SWITCH);
    let t = -x;
    let q = mills_correction_continued_fraction(t);
    [
        normal_logcdf(x),
        t + q.value,
        -(1.0 + q.first),
        q.second,
        -q.third,
    ]
}

#[inline]
fn normal_logcdf_derivatives_right_tail(x: f64) -> [f64; 5] {
    assert!(x.is_finite() && x >= 8.0);
    const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
    let log_cdf = normal_logcdf(x);
    let u = x / std::f64::consts::SQRT_2;
    let log_lambda = -u * u - LOG_SQRT_2PI - log_cdf;
    let log_x = x.ln();
    let inv_x2 = x.recip() * x.recip();

    let first = log_lambda.exp();
    let second = signed_exp_sum(&[log_x + log_lambda, 2.0 * log_lambda], &[-1.0, -1.0]);
    let third = signed_exp_sum(
        &[
            2.0 * log_x + (-inv_x2).ln_1p() + log_lambda,
            3.0_f64.ln() + log_x + 2.0 * log_lambda,
            2.0_f64.ln() + 3.0 * log_lambda,
        ],
        &[1.0, 1.0, 1.0],
    );
    let fourth = signed_exp_sum(
        &[
            3.0 * log_x + (-3.0 * inv_x2).ln_1p() + log_lambda,
            7.0_f64.ln() + 2.0 * log_x + (-(4.0 / 7.0) * inv_x2).ln_1p() + 2.0 * log_lambda,
            12.0_f64.ln() + log_x + 3.0 * log_lambda,
            6.0_f64.ln() + 4.0 * log_lambda,
        ],
        &[-1.0, -1.0, -1.0, -1.0],
    );
    [log_cdf, first, second, third, fourth]
}

#[inline]
fn signed_exp_sum(log_magnitudes: &[f64], signs: &[f64]) -> f64 {
    let (log_magnitude, sign) = signed_log_sum_exp(log_magnitudes, signs);
    if sign == 0.0 {
        0.0
    } else {
        sign * log_magnitude.exp()
    }
}

#[inline]
fn acklam_lower_tail_quantile_from_log_probability(log_p: f64) -> f64 {
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let q = (-2.0 * log_p).sqrt();
    (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
        / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
}

/// Standard normal quantile Φ⁻¹(p) using Acklam's rational approximation.
#[inline]
pub fn standard_normal_quantile(p: f64) -> Result<f64, String> {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return Err(format!("normal quantile requires p in (0,1), got {p}"));
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let mut x = if p < P_LOW {
        acklam_lower_tail_quantile_from_log_probability(p.ln())
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -acklam_lower_tail_quantile_from_log_probability((1.0 - p).ln())
    };
    for _ in 0..2 {
        let density = normal_pdf(x);
        if !(density.is_finite() && density > 0.0) {
            break;
        }
        // Residual F(x) − p, formed without catastrophic cancellation in
        // either tail. For an upper-tail iterate `x > 0`, `normal_cdf(x)`
        // saturates to ~1, so the direct `normal_cdf(x) − p` annihilates the
        // tiny residual the polish must act on; instead use the upper-tail
        // complement `F(x) − p = (1 − p) − 0.5·erfc(x/√2)`, where both terms
        // are the small upper-tail quantities (`1 − p` is exact by Sterbenz
        // for `p ∈ [½,1)`). For `x ≤ 0`, `normal_cdf(x) = 0.5·erfc(|x|/√2)` is
        // itself the faithfully carried small lower-tail value, so the direct
        // form is already cancellation-free.
        let residual = if (0.25..=0.75).contains(&p) {
            // Central band. Both tail forms below subtract two quantities of
            // size ~½, so their difference carries an absolute error of one ulp
            // of ½ (1.1e-16) NO MATTER how small the true residual is. Since
            // `Δx ≈ residual_error / φ(x)`, the returned quantile then carries a
            // FIXED absolute error ~1.2e-16 and a relative error ~1.2e-16/|x|
            // that diverges as `p → ½`: measured 4.1e-14 at `p = 0.50125` and
            // 1.2e-03 at `p = ½ + 2.75e-14`, against ~2e-16 everywhere else in
            // this module. The polish cannot repair the seed there — the
            // residual it is handed is quantized to multiples of one ulp of ½
            // and is usually exactly 0, so the answer that ships is the raw
            // Acklam seed at its own 1.15e-9.
            //
            // Subtracting the ½ ANALYTICALLY removes it: `F(x) − p` is
            // `(F(x) − ½) − (p − ½)` = `½·erf(x/√2) − δ`, and both terms are now
            // of size |δ| with full RELATIVE accuracy — `erf` near 0 is `z·R(z²)`,
            // no cancellation — so the residual error is `ε·|δ|` and the relative
            // error in `x` is `ε` uniformly, including in the limit `x → 0`.
            //
            // The band is the exactness domain of `δ`, not a tuning choice:
            // Sterbenz's lemma makes `p − ½` exact for `p ∈ [¼, 1]`, and the
            // reflection `p ↦ 1 − p` maps that onto `[0, ¾]`, so `[¼, ¾]` is
            // where δ is exact on both sides. It is also where the centered form
            // is the better one: outside it `|x| > 0.6745` and the tail forms
            // carry relative accuracy in their own small quantity, which is what
            // the deep tails need. At the shared boundary the two agree to
            // within a factor of two, so nothing steps across the seam.
            0.5 * erf(x / std::f64::consts::SQRT_2) - (p - 0.5)
        } else if x > 0.0 {
            (1.0 - p) - 0.5 * erfc(x / std::f64::consts::SQRT_2)
        } else {
            normal_cdf(x) - p
        };
        let correction = residual / density;
        let denominator = 1.0 + 0.5 * x * correction;
        if !(correction.is_finite() && denominator.is_finite() && denominator != 0.0) {
            break;
        }
        let step = correction / denominator;
        if !step.is_finite() {
            break;
        }
        x -= step;
        if step.abs() <= 2.0 * f64::EPSILON * x.abs().max(1.0) {
            break;
        }
    }
    Ok(x)
}

/// Standard normal quantile from `log_p = ln Φ(x)`.
///
/// Unlike [`standard_normal_quantile`], this remains defined when `Φ(x)` is
/// smaller than the least positive `f64`, and when `Φ(x)` is so close to one
/// that exponentiating `log_p` rounds to exactly one. Acklam's lower-tail
/// approximation supplies the initial point; Newton polishing solves
/// `ln Φ(x) = log_p` with the stable log-CDF and Mills ratio, so neither tail
/// forms a probability-space subtraction.
#[inline]
pub fn standard_normal_quantile_from_log_cdf(log_p: f64) -> Result<f64, String> {
    if !(log_p.is_finite() && log_p < 0.0) {
        return Err(format!(
            "normal log-quantile requires finite log_p < 0, got {log_p}"
        ));
    }

    if log_p > -std::f64::consts::LN_2 {
        // Reflect through the upper tail without forming `1 - exp(log_p)`.
        let log_q = (-log_p.exp_m1()).ln();
        return standard_normal_quantile_from_log_cdf(log_q).map(|x| -x);
    }

    let p = log_p.exp();
    let mut x = if p > 0.0 {
        standard_normal_quantile(p)?
    } else {
        acklam_lower_tail_quantile_from_log_probability(log_p)
    };
    for _ in 0..4 {
        let (current_log_p, mills_ratio) = signed_probit_logcdf_and_mills_ratio(x);
        if !(current_log_p.is_finite() && mills_ratio.is_finite() && mills_ratio > 0.0) {
            break;
        }
        let step = (current_log_p - log_p) / mills_ratio;
        if !step.is_finite() {
            break;
        }
        x -= step;
        if step.abs() <= 2.0 * f64::EPSILON * x.abs().max(1.0) {
            break;
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn rel_err(got: f64, expected: f64) -> f64 {
        (got - expected).abs() / expected.abs().max(1e-300)
    }

    #[test]
    fn student_t_primitives_keep_the_tail_that_one_minus_the_cdf_destroys() {
        // References are correctly rounded doubles from a 60-dps regularized
        // incomplete beta. The `nu = 10000, t = 10` row is here because a
        // plausible-looking hand-extrapolated literal for it (1.60e-23) is 20%
        // from the truth: this table has to come from the reference, not from
        // pattern-matching the rows above it.
        const ROWS: [(f64, f64, f64); 10] = [
            (5.0, 20.0, 2.887758186612086e-6),
            (5.0, 40.0, 9.205981085886477e-8),
            (30.0, 10.0, 2.2876257041148065e-11),
            (30.0, 20.0, 3.3745418328856434e-19),
            (30.0, 40.0, 6.863022597203202e-28),
            (500.0, 8.0, 4.3648313969400955e-15),
            (500.0, 10.0, 6.930246799119958e-22),
            (500.0, 20.0, 4.056001518093838e-66),
            (500.0, 40.0, 3.14532145912912e-158),
            (10000.0, 10.0, 9.816403714331914e-24),
        ];
        // Bar: 1e-11, a measured envelope rather than a derivation. Everything
        // below the incomplete beta is derivable -- the identity is exact and
        // forms no difference -- but `beta_reg` is statrs's continued fraction
        // and its error is a property of that implementation, so the honest
        // thing is to measure it and say so. Worst over this table by shape
        // parameter `a = nu/2`:
        //
        //     a = 2.5     2e-15
        //     a = 15      1.6e-13
        //     a = 250     2.3e-13
        //     a = 5000    2.0e-12
        //
        // It grows slowly with nu, which is what a continued fraction needing
        // more terms looks like, and it does *not* grow with tail depth -- the
        // nu = 500 rows sit at 2e-13 whether the answer is 1e-15 or 1e-158.
        // That is the distinction that matters: a fixed relative cost, not a
        // cancellation. Bar is 5x the worst measured.
        let bar = 1.0e-11;
        for (nu, t, want) in ROWS {
            let got = student_t_sf(t, nu);
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= bar,
                "student_t_sf({t}, {nu}) = {got:e}, want {want:e}, relative {rel:e} > {bar:e}"
            );
            let got_two_sided = student_t_two_sided_probability(t, nu);
            let two_sided_rel = ((got_two_sided - 2.0 * want) / (2.0 * want)).abs();
            assert!(
                two_sided_rel <= bar,
                "student_t_two_sided_probability({t}, {nu}) = {got_two_sided:e}, \
                 want {:e}, relative {two_sided_rel:e} > {bar:e}",
                2.0 * want
            );
            // The reflection. `1 - want` is O(1), so its own absolute error of
            // one ulp is a relative error of one ulp -- which is exactly why
            // reflecting is safe here and reconstructing the small tail is not.
            let lower = student_t_sf(-t, nu);
            assert!(
                (lower - (1.0 - want)).abs() <= 2.0 * f64::EPSILON,
                "student_t_sf({}, {nu}) = {lower}, want {}",
                -t,
                1.0 - want
            );
        }
        // Symmetry at the median, and the degenerate arguments.
        for nu in [1.0_f64, 5.0, 1e4] {
            assert!(
                (student_t_sf(0.0, nu) - 0.5).abs() <= f64::EPSILON,
                "median at nu = {nu}"
            );
        }
        assert!(student_t_sf(1.0, 0.0).is_nan(), "nu = 0 is not a t");
        assert!(
            student_t_sf(1.0, f64::INFINITY).is_nan(),
            "nu = inf is not a t"
        );
        assert_eq!(student_t_sf(f64::INFINITY, 5.0), 0.0, "tail beyond +inf");
        assert_eq!(
            student_t_sf(f64::NEG_INFINITY, 5.0),
            1.0,
            "tail beyond -inf"
        );
    }

    #[test]
    fn normal_sf_keeps_the_upper_tail_that_one_minus_the_cdf_destroys() {
        // `Φ(x)` rounds to exactly 1.0 once its upper tail drops below half an
        // ulp of one, so `1 - normal_cdf(x)` returns exactly zero from x ~ 8.3 up
        // and is already 7% high at x = 8. `normal_sf` computes the tail rather
        // than reconstructing it. References are correctly rounded doubles from a
        // 60-dps `erfc(x/√2)/2`.
        //
        // Bar: `x * x * eps`, which is derived rather than chosen. Forming the
        // argument `u = x / √2` rounds it, a relative eps, i.e. an absolute
        // `u * eps`. The relative condition number of `erfc` at `u` is
        // `u * |erfc'(u)| / erfc(u)`, and since `erfc(u) ~ exp(-u^2) / (u√π)` for
        // large `u` that tends to `2u^2 = x^2`. So the returned tail inherits
        // `x^2 * eps` from the argument alone, before `erfc`'s own couple of ulp
        // -- 36 ulp at x = 6, 1370 ulp at x = 37. That is intrinsic to taking a
        // z score as the input: the tail is exponentially steep in `x`, so the
        // last bit of `x` is worth `x^2` bits of the tail. It is also
        // irrelevant next to what it replaces, which is a relative error of 1.
        const ROWS: [(f64, f64); 13] = [
            (0.5, 0.3085375387259869),
            (2.0, 0.02275013194817921),
            (4.0, 3.1671241833119924e-5),
            (5.0, 2.866515718791933e-7),
            (6.0, 9.86587645037698e-10),
            (7.0, 1.279812543885835e-12),
            (8.0, 6.220960574271784e-16),
            (8.3, 5.205569744890254e-17),
            (9.0, 1.1285884059538405e-19),
            (12.0, 1.776482112077679e-33),
            (20.0, 2.7536241186062337e-89),
            (30.0, 4.906713927148187e-198),
            (37.0, 5.725571222524577e-300),
        ];
        for (x, want) in ROWS {
            let bar = (x * x + 2.0) * f64::EPSILON;
            let got = normal_sf(x);
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= bar,
                "normal_sf({x}) = {got:e}, want {want:e}, relative {rel:e} > {bar:e}"
            );
            let got_two_sided = normal_two_sided_probability(x);
            let two_sided_rel = ((got_two_sided - 2.0 * want) / (2.0 * want)).abs();
            assert!(
                two_sided_rel <= bar,
                "normal_two_sided_probability({x}) = {got_two_sided:e}, \
                 want {:e}, relative {two_sided_rel:e} > {bar:e}",
                2.0 * want
            );
            // The value this replaces. Above the saturation point it is not a
            // less accurate answer, it is no answer.
            if x >= 8.3 {
                assert_eq!(
                    1.0 - normal_cdf(x),
                    0.0,
                    "1 - normal_cdf({x}) is expected to have saturated"
                );
            }
        }
        // Complementarity holds wherever the sum is representable, and the
        // symmetry that makes a two-sided p-value a single call.
        for x in [-3.0_f64, -0.25, 0.0, 0.25, 3.0] {
            let sum = normal_sf(x) + normal_cdf(x);
            assert!((sum - 1.0).abs() <= 2.0 * f64::EPSILON, "sf + cdf = {sum}");
            assert_eq!(normal_sf(-x), normal_cdf(x), "sf(-x) != cdf(x) at {x}");
        }
    }

    /// The final representable normal two-sided tail is subnormal. This is a
    /// separate absolute/ULP assertion because a conventional relative-error
    /// helper with a normal-number floor would make the edge vacuous.
    #[test]
    fn normal_two_sided_tail_retains_subnormal_edge() {
        const EXPECTED_AT_38: f64 = 5.770_856_702_007_929e-316;
        let got = normal_two_sided_probability(38.0);
        let ulps = got.to_bits().abs_diff(EXPECTED_AT_38.to_bits());
        assert!(
            got.is_subnormal() && ulps <= 128,
            "two-sided normal tail at z=38: got {got:.17e}, \
             expected {EXPECTED_AT_38:.17e}, ulps {ulps}"
        );
        assert_eq!(normal_two_sided_probability(40.0), 0.0);
        assert_eq!(normal_two_sided_probability(f64::INFINITY), 0.0);
        assert!(normal_two_sided_probability(f64::NAN).is_nan());
    }

    /// `t²` and then `ν/(ν+t²)` both underflow at this edge, but the Cauchy
    /// tail itself is still representable. The analytic Cauchy survival law is
    /// an independent oracle for the log-beta implementation.
    #[test]
    fn student_t_two_sided_tail_retains_subnormal_cauchy_edge() {
        const EXPECTED: f64 = 3.541_315_033_259_774_5e-309;
        let got = student_t_two_sided_probability(f64::MAX, 1.0);
        let analytic = 2.0 * (1.0 / f64::MAX).atan() / std::f64::consts::PI;
        let pinned_ulps = got.to_bits().abs_diff(EXPECTED.to_bits());
        let analytic_ulps = got.to_bits().abs_diff(analytic.to_bits());
        assert!(
            got.is_subnormal() && pinned_ulps <= 512 && analytic_ulps <= 512,
            "Cauchy tail at f64::MAX: got {got:.17e}, pinned {EXPECTED:.17e}, \
             analytic {analytic:.17e}, pinned ulps {pinned_ulps}, \
             analytic ulps {analytic_ulps}"
        );
    }

    #[test]
    fn distribution_survival_primitives_define_boundaries_and_identities() {
        assert_eq!(normal_sf(f64::INFINITY), 0.0);
        assert_eq!(normal_sf(f64::NEG_INFINITY), 1.0);
        assert!(normal_sf(f64::NAN).is_nan());

        assert_eq!(student_t_two_sided_probability(0.0, 7.0), 1.0);
        assert_eq!(student_t_sf(0.0, 7.0), 0.5);
        assert!(student_t_sf(f64::NAN, 7.0).is_nan());

        assert_eq!(chi_square_sf(0.0, 3.0), 1.0);
        assert_eq!(chi_square_sf(f64::INFINITY, 3.0), 0.0);
        assert!(chi_square_sf(-1.0, 3.0).is_nan());
        assert!(chi_square_sf(1.0, 0.0).is_nan());

        assert_eq!(fisher_snedecor_sf(0.0, 3.0, 20.0), 1.0);
        assert_eq!(
            fisher_snedecor_sf(f64::INFINITY, 3.0, 20.0),
            0.0
        );
        assert!(fisher_snedecor_sf(-1.0, 3.0, 20.0).is_nan());
        assert!(fisher_snedecor_sf(1.0, 0.0, 20.0).is_nan());
        assert!(fisher_snedecor_sf(1.0, 3.0, 0.0).is_nan());

        // χ²₁ is the square of a standard normal; F₁,₁ is the square of a
        // Cauchy. These identities independently anchor both direct survival
        // implementations in a small-tail regime.
        let statistic = 160.0_f64;
        let chi_expected = normal_two_sided_probability(statistic.sqrt());
        let chi_got = chi_square_sf(statistic, 1.0);
        assert!(rel_err(chi_got, chi_expected) <= 2.0e-13);

        let f_expected = student_t_two_sided_probability(statistic.sqrt(), 1.0);
        let f_got = fisher_snedecor_sf(statistic, 1.0, 1.0);
        assert!(rel_err(f_got, f_expected) <= 2.0e-13);
    }

    #[test]
    /// The lower tail of a beta quantile, where `inv_beta_reg`'s absolute
    /// convergence tolerance in `x` used to stall (#2528).
    ///
    /// Shapes are the ones `gam_inference::probability` derives from a mean and
    /// a variance (`precision = mu(1-mu)/total_var - 1`), so every row is the
    /// lower endpoint of a 95% predictive interval a caller can actually ask
    /// for. References are an 80-digit bisection in `ln x` on
    /// `I_x(a,b) = p`; the `Beta(0.1, 0.1)` row is additionally checkable in
    /// closed form, since `I_x -> x^a/(a B(a,b))` gives
    /// `x = (p a B(a,b))^(1/a)` there.
    ///
    /// What shipped before, against the same references: `6.7e-18` for the
    /// first row (true `1.5e-41`, relative error 4.6e+23), `5.8e-18` for the
    /// second (true `6.3e-161`), and `9.6e-19` for the underflow row, whose
    /// true quantile is `7.7e-688` and whose only correct `f64` answer is `0`.
    /// The failure was not a loss of digits but a floor: every one of those
    /// returns is the solver's own resolution limit rather than a quantile.
    fn beta_quantile_resolves_the_lower_tail_below_the_solver_floor() {
        const CASES: [(f64, f64, f64, f64); 8] = [
            (0.04, 3.96, 0.025, 1.4749755854885786e-41),
            (0.01, 0.99, 0.025, 6.326229749489128e-161),
            (
                0.046666666666666666,
                2.2866666666666666,
                0.025,
                1.488779171021457e-35,
            ),
            (0.05, 0.95, 0.025, 9.875267916846768e-33),
            (0.1, 0.9, 0.025, 1.12479965068234e-16),
            (0.3, 0.7, 0.025, 7.6005358168401896e-6),
            (0.5, 0.5, 0.025, 1.5413331334360133e-3),
            (0.1, 0.1, 1.0e-4, 8.869280655550463e-38),
        ];
        let mut worst = 0.0_f64;
        for (a, b, p, want) in CASES {
            let got = beta_quantile(p, a, b);
            let relative = ((got - want) / want).abs();
            assert!(
                relative <= 16.0 * f64::EPSILON,
                "beta_quantile({p}, {a}, {b}) = {got:e}, want {want:e}, relative {relative:e}"
            );
            worst = worst.max(relative);
        }
        println!("worst relative error over the lower-tail table: {worst:e}");

        // The true quantile here is 7.7e-688. It is not representable, so the
        // correctly rounded answer is zero, and a caller reading a positive
        // lower bound could not tell that it had underflowed.
        let underflowed = beta_quantile(0.025, 0.0023333333333333335, 2.3310000000000004);
        assert!(
            underflowed == 0.0,
            "a quantile below MIN_POSITIVE must round to zero, got {underflowed:e}"
        );

        // The upper tail of the same shape is not on the series branch and is
        // still `inv_beta_reg`'s answer, at `inv_beta_reg`'s own accuracy. It is
        // asserted here so that widening the branch cannot silently move it.
        const UPPER: f64 = 0.12274676682071068;
        let upper = beta_quantile(0.975, 0.04, 3.96);
        assert!(
            ((upper - UPPER) / UPPER).abs() <= 1.0e-11,
            "upper tail moved: {upper:e}, want {UPPER:e}"
        );
    }

    #[test]
    fn beta_quantile_matches_known_reference_values() {
        let cases: [(f64, f64, f64, f64); 8] = [
            (0.025, 2.0, 2.0, 0.094_299_3),
            (0.975, 2.0, 2.0, 0.905_700_7),
            (0.5, 2.0, 2.0, 0.5),
            (0.025, 0.8, 4.0, 0.002_339_1),
            (0.975, 0.8, 4.0, 0.564_717_3),
            (0.025, 5.0, 1.5, 0.408_549_1),
            (0.5, 20.0, 80.0, 0.197_994_8),
            (0.975, 20.0, 80.0, 0.283_367_6),
        ];
        for (p, a, b, expected) in cases {
            let got = beta_quantile(p, a, b);
            let abs = (got - expected).abs();
            assert!(
                abs < 1e-5,
                "beta_quantile(p={p}, a={a}, b={b}) = {got}, expected ≈ {expected} (abs err {abs})"
            );
        }
    }

    #[test]
    fn beta_quantile_boundaries_and_degeneracy() {
        assert_eq!(beta_quantile(0.0, 2.0, 3.0), 0.0);
        assert_eq!(beta_quantile(-0.5, 2.0, 3.0), 0.0);
        assert_eq!(beta_quantile(1.0, 2.0, 3.0), 1.0);
        assert_eq!(beta_quantile(1.5, 2.0, 3.0), 1.0);
        assert!(beta_quantile(0.5, -1.0, 3.0).is_nan());
        assert!(beta_quantile(0.5, 2.0, 0.0).is_nan());
        assert!(beta_quantile(0.5, f64::NAN, 3.0).is_nan());
        let mut prev = 0.0;
        for i in 1..100 {
            let p = i as f64 / 100.0;
            let q = beta_quantile(p, 3.0, 5.0);
            assert!(q > prev, "beta quantile not increasing at p={p}");
            prev = q;
        }
    }

    // ── normal_pdf ────────────────────────────────────────────────────────────

    #[test]
    fn normal_pdf_at_zero() {
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((normal_pdf(0.0) - expected).abs() < TOL);
    }

    #[test]
    fn normal_pdf_symmetry() {
        for &x in &[0.5, 1.0, 2.0, 3.0, 5.0] {
            assert_eq!(normal_pdf(x), normal_pdf(-x), "symmetry failed at x={x}");
        }
    }

    /// `x*x` is exact-splittable and the split is what `exp` needs.
    ///
    /// Two independent statements, because the correction is only worth what
    /// its residual is worth. First, `x*x + residual` is `x²` EXACTLY: checked
    /// against a Veltkamp/Dekker split, which reaches the same residual through
    /// pure multiplies and adds and shares no code path with the `mul_add`
    /// route. Second, the residual is not decorative — for these arguments it
    /// is a relative perturbation of `x²` big enough that `exp` amplifies it
    /// past a single ulp of the result.
    #[test]
    fn square_residual_completes_the_rounded_square_exactly() {
        // 2^27 + 1: Veltkamp's splitting factor, exact for any `x` whose
        // scaled form does not overflow.
        const SPLIT: f64 = 134_217_729.0;
        let mut saw_amplified = false;
        for &x in &[
            0.1, 0.7, 1.3, 2.9, 6.1, 10.5, 14.3, 19.7, 23.9, 25.9999, 34.7,
        ] {
            let rounded = x * x;
            let residual = square_residual(x, rounded);

            let c = x * SPLIT;
            let head = c - (c - x);
            let tail = x - head;
            let dekker = ((head * head - rounded) + 2.0 * head * tail) + tail * tail;
            assert_eq!(
                residual, dekker,
                "x={x}: mul_add residual {residual:e} != Dekker residual {dekker:e}"
            );

            // `exp` multiplies a relative argument perturbation by the argument.
            let amplified = (residual / rounded).abs() * rounded;
            if amplified > f64::EPSILON {
                saw_amplified = true;
            }
        }
        assert!(
            saw_amplified,
            "no test argument had a residual `exp` could amplify past one ulp; \
             the correction under test would be untested"
        );
    }

    /// `φ(x)` against an EXTERNAL high-precision reference (mpmath, dps=60).
    ///
    /// Every argument here has an INEXACT square, which is the whole point.
    /// `exp(−½·fl(x*x))` misplaces the argument by `x²·ε/2` RELATIVE, and `exp`
    /// hands that straight back as relative error in the result: `1.4e-14` at
    /// `x ≈ 17`, `5.7e-14` by `x ≈ 35`, where `φ` is still a normal `f64`. Only
    /// the top of the range makes that visible, so the table has to reach it —
    /// a `φ` table that stops at `x = 5` cannot tell the two forms apart.
    ///
    /// `1.5e-15` (≈7 ulp) is the portability allowance: `f64::exp` is the
    /// platform libm and the only part of this that is not fixed by the crate
    /// graph, and it is worth ~1 ulp on the implementations in use. That still
    /// leaves 38x of margin against the defect at the top of the table.
    #[test]
    fn normal_pdf_matches_high_precision_reference() {
        const TOLERANCE: f64 = 1.5e-15;
        let refs: &[(f64, f64)] = &[
            (0.5, 0.35206532676429947),
            (1.0, 0.24197072451914334),
            (2.5, 0.017528300493568537),
            (4.0, 0.00013383022576488534),
            (7.3, 1.0693837871541648e-12),
            (11.9, 7.090702668428078e-32),
            (17.4, 7.201308152719057e-67),
            (23.6, 4.555989824112156e-122),
            (29.1, 5.229437243665329e-185),
            (34.7, 1.368008224488383e-262),
        ];
        for &(x, reference) in refs {
            // The small arguments anchor the ordinary range; the large ones are
            // where the defect lives, and every one of THOSE has to have a
            // square `f64` cannot hold or it exercises nothing.
            assert!(
                x <= 5.0 || square_residual(x, x * x) != 0.0,
                "x={x} squares exactly, so it cannot exercise the correction"
            );
            let rel = rel_err(normal_pdf(x), reference);
            assert!(
                rel < TOLERANCE,
                "normal_pdf({x}) = {:.17e}, reference {reference:.17e}, rel {rel:.3e}",
                normal_pdf(x)
            );
        }
    }

    /// `φ` off the ordinary domain, where the square has no usable residual:
    /// `±∞` squares to `∞` and would hand the correction an `∞ − ∞`.
    #[test]
    fn normal_pdf_nonfinite_and_underflowed_arguments() {
        assert_eq!(normal_pdf(f64::INFINITY), 0.0);
        assert_eq!(normal_pdf(f64::NEG_INFINITY), 0.0);
        assert!(normal_pdf(f64::NAN).is_nan());
        // Past ~38.6 the pdf underflows; it must reach zero, not NaN.
        assert_eq!(normal_pdf(40.0), 0.0);
        assert_eq!(normal_pdf(-40.0), 0.0);
        assert_eq!(normal_pdf(f64::MAX), 0.0);
        // Just inside the underflow edge the result is subnormal but positive.
        let edge = normal_pdf(38.0);
        assert!(edge > 0.0 && edge.is_subnormal(), "phi(38) = {edge:e}");
    }

    #[test]
    fn normal_pdf_positive() {
        for &x in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            assert!(normal_pdf(x) > 0.0, "pdf should be positive at x={x}");
        }
    }

    // ── normal_cdf ────────────────────────────────────────────────────────────

    #[test]
    fn normal_cdf_at_zero_is_half() {
        assert!((normal_cdf(0.0) - 0.5).abs() < TOL);
    }

    #[test]
    fn normal_cdf_symmetry() {
        for &x in &[0.5, 1.0, 2.0, 3.0] {
            let sum = normal_cdf(x) + normal_cdf(-x);
            assert!(
                (sum - 1.0).abs() < TOL,
                "cdf symmetry failed at x={x}: sum={sum}"
            );
        }
    }

    #[test]
    fn normal_cdf_bounds() {
        assert!(normal_cdf(10.0) > 0.9999);
        assert!(normal_cdf(-10.0) < 1e-22);
        assert!(normal_cdf(0.0) > 0.0);
        assert!(normal_cdf(0.0) < 1.0);
    }

    #[test]
    fn normal_cdf_at_1_96_near_0975() {
        // Phi(1.96) ≈ 0.975 — canonical two-sided 5% critical value.
        let p = normal_cdf(1.959_963_985);
        assert!((p - 0.975).abs() < 1e-8, "p={p}");
    }

    // ── erfcx_nonnegative ─────────────────────────────────────────────────────

    #[test]
    fn erfcx_zero_is_one_and_negative_domain_is_rejected() {
        assert_eq!(erfcx_nonnegative(0.0), 1.0);
        assert!(erfcx_nonnegative(-f64::MIN_POSITIVE).is_nan());
        assert!(erfcx_nonnegative(-1.0).is_nan());
        assert!(erfcx_nonnegative(f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn erfcx_positive_inf_returns_zero() {
        assert_eq!(erfcx_nonnegative(f64::INFINITY), 0.0);
    }

    #[test]
    fn erfcx_nan_propagates() {
        assert!(erfcx_nonnegative(f64::NAN).is_nan());
    }

    #[test]
    fn erfcx_small_positive_matches_direct() {
        use libm::erfc;
        for &x in &[0.1_f64, 0.5, 1.0, 5.0, 10.0, 25.0] {
            let got = erfcx_nonnegative(x);
            let expected = (x * x).exp() * erfc(x);
            let err = rel_err(got, expected);
            assert!(
                err < 1e-10,
                "x={x}: got={got} expected={expected} rel={err}"
            );
        }
    }

    #[test]
    fn erfcx_large_x_positive_and_finite() {
        // For x >= 26 the asymptotic branch must remain positive and finite.
        let got = erfcx_nonnegative(50.0);
        assert!(got.is_finite() && got > 0.0, "erfcx(50)={got}");
        // Leading asymptotic term: 1/(x*sqrt(pi)).
        let asymptotic = 1.0 / (50.0 * std::f64::consts::PI.sqrt());
        assert!(
            rel_err(got, asymptotic) < 1e-3,
            "got={got} asymptotic={asymptotic}"
        );
    }

    /// The two branches must describe one function across `x = 26`.
    ///
    /// Note WHY the plain `exp(x*x)·erfc(x)` below is a legitimate oracle at
    /// this particular argument and nowhere else: `26² = 676` is exactly
    /// representable, so the rounded square carries no residual and the direct
    /// form is momentarily as good as the corrected one. That is also exactly
    /// why this check was blind to the `x²·ε/2` defect it looks like it should
    /// have caught — at `25.9` the same comparison would have failed by
    /// `5.7e-14`, but the seam was only ever probed at the one point in the
    /// neighbourhood where the defect vanishes. The bit-adjacent step below
    /// cannot substitute for it either: `d(ln erfcx)/dx ≈ −2x` at the switch,
    /// so one ulp of `x` moves the true value by `1.8e-13`, three times the
    /// defect. It takes a reference at a DISTANCE from the seam — the table in
    /// `erfcx_matches_high_precision_reference` — to see the defect at all.
    #[test]
    fn erfcx_asymptotic_switch_matches_finite_direct_identity() {
        let switch = 26.0_f64;
        assert_eq!(
            square_residual(switch, switch * switch),
            0.0,
            "676 must be exact for the direct form below to be an oracle"
        );
        let direct = (switch * switch).exp() * erfc(switch);
        let asymptotic = erfcx_nonnegative(switch);
        assert!(
            rel_err(asymptotic, direct) < 1.0e-15,
            "switch mismatch: asymptotic={asymptotic:.17e}, direct={direct:.17e}"
        );

        // Continuity across the branch cut, up to how fast the function itself
        // moves over one ulp of `x` (`|d ln erfcx/dx| ≈ 2x` ⇒ ~1.9e-13 here).
        let immediately_below = f64::from_bits(switch.to_bits() - 1);
        let below = erfcx_nonnegative(immediately_below);
        let step = 2.0 * switch * (switch - immediately_below);
        assert!(
            rel_err(asymptotic, below) < 2.0 * step,
            "discontinuous switch: below={below:.17e}, at={asymptotic:.17e}, \
             one-ulp travel {step:.3e}"
        );
    }

    #[test]
    fn erfcx_preserves_representable_subnormal_tail() {
        let tail = erfcx_nonnegative(f64::MAX);
        assert!(tail > 0.0 && tail.is_subnormal(), "erfcx(MAX)={tail:e}");
    }

    /// Absolute-accuracy pin against an EXTERNAL high-precision reference
    /// (mpmath, dps=60) spanning the direct branch `[0.1, 26)`. This is the
    /// root-cause guard: the previous `exp(x²)·erfc(x)` direct form was built on
    /// `statrs::erfc`, whose ~1e-10 relative accuracy silently poisoned every
    /// downstream probit / Mills / log-CDF derivative.
    ///
    /// The table had a SECOND job it was not doing. Of its twelve arguments,
    /// eleven — `0.5`, `2`, `3.5`, `6`, `9`, `13`, `18`, `22`, `25.5`, and the
    /// two whose squares are far too small to matter — square EXACTLY in `f64`,
    /// so `fl(x*x) = x²` and the `x²·ε/2` error the rounded square feeds `exp`
    /// was identically zero at every one of them. The twelfth, `25.9999`, does
    /// not square exactly; it was the one point in the table where the defect
    /// was live, and its literal had been recorded WITH the defect in it —
    /// `0.021683668126370212` against a true `0.021683668126369115`, off by
    /// `5.1e-14`. Three independent high-precision routes (`exp(x²)·erfc(x)`,
    /// the 12-term asymptotic series, and a 400-level Laplace continued
    /// fraction) and `scipy.special.erfcx` all agree on the corrected value.
    /// A `1e-13` tolerance then accepted a reference that was itself wrong by
    /// half the tolerance, which is how a 190x accuracy defect sat under a
    /// test named for high precision.
    ///
    /// So the table now RUNS ON arguments with inexact squares (`10.5`,
    /// `14.3`, `19.7`, `23.9` alongside the original grid) and the tolerance is
    /// `1.5e-15` — 38x below the defect at the top of the range, and still ~7
    /// ulp of headroom for the platform `f64::exp` (the only part of this path
    /// not pinned by the crate graph; `erfc` comes from the `libm` crate and is
    /// identical everywhere).
    #[test]
    fn erfcx_matches_high_precision_reference() {
        const TOLERANCE: f64 = 1.5e-15;
        // (x, mpmath exp(x²)·erfc(x) at dps=60, rounded to f64).
        let refs: &[(f64, f64)] = &[
            (0.1, 0.8964569799691267),
            (0.5, 0.6156903441929259),
            (1.0, 0.427583576155807),
            (2.0, 0.25539567631050575),
            (3.5, 0.1552936556088943),
            (6.0, 0.09277656780053835),
            (9.0, 0.06230772403777468),
            (10.5, 0.05349189974656412),
            (13.0, 0.043271921864609694),
            (14.3, 0.0393580473372741),
            (18.0, 0.03129571781590521),
            (19.7, 0.028602309402825203),
            (22.0, 0.025618570005879453),
            (23.9, 0.023585649371803793),
            (25.5, 0.022108108052519827),
            (25.9999, 0.021683668126369115),
        ];
        for &(x, reference) in refs {
            let got = erfcx_nonnegative(x);
            let rel = rel_err(got, reference);
            assert!(
                rel < TOLERANCE,
                "erfcx({x}) = {got:.17e}, reference {reference:.17e}, rel {rel:.3e}"
            );
        }
        // The point of the added arguments: at least four of them must have a
        // square `f64` cannot hold, or the table is back to testing nothing.
        let inexact = refs
            .iter()
            .filter(|&&(x, _)| square_residual(x, x * x) != 0.0)
            .count();
        assert!(
            inexact >= 4,
            "only {inexact} of {} reference arguments have an inexact square",
            refs.len()
        );
    }

    // ── log1mexp_positive ─────────────────────────────────────────────────────

    #[test]
    fn log1mexp_at_zero_is_neg_inf() {
        assert_eq!(log1mexp_positive(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log1mexp_recovers_log_one_minus_exp() {
        // Verify exp(log1mexp(a)) + exp(-a) ≈ 1 for several a > 0. This
        // roundtrip avoids computing `(1 - exp(-a)).ln()` directly, which
        // suffers catastrophic cancellation for large a (e.g. a=20 where
        // `1.0 - exp(-20)` loses 9 decimal digits from the subtraction).
        for &a in &[0.001_f64, 0.5, std::f64::consts::LN_2, 1.0, 5.0, 20.0] {
            let lm = log1mexp_positive(a);
            let roundtrip = lm.exp() + (-a).exp();
            assert!(
                (roundtrip - 1.0).abs() < 1e-14,
                "a={a}: exp(log1mexp(a)) + exp(-a) = {roundtrip}, expected 1.0"
            );
        }
    }

    #[test]
    fn log1mexp_at_ln2_is_neg_ln2() {
        let ln2 = std::f64::consts::LN_2;
        let got = log1mexp_positive(ln2);
        assert!((got - (-ln2)).abs() < TOL, "got={got}");
    }

    // ── signed_log_sum_exp ────────────────────────────────────────────────────

    #[test]
    fn slse_all_positive_single() {
        let (lm, sg) = signed_log_sum_exp(&[2.0], &[1.0]);
        assert!((lm - 2.0).abs() < TOL);
        assert!((sg - 1.0).abs() < TOL);
    }

    #[test]
    fn slse_difference_recovers_log2() {
        // 3 - 1 = 2 → log|2| = ln(2), sign = +1.
        let log3 = 3.0_f64.ln();
        let log1 = 0.0_f64; // ln(1)
        let (lm, sg) = signed_log_sum_exp(&[log3, log1], &[1.0, -1.0]);
        assert!((lm - 2.0_f64.ln()).abs() < TOL, "lm={lm}");
        assert!((sg - 1.0).abs() < TOL, "sg={sg}");
    }

    #[test]
    fn slse_cancellation_gives_neg_inf() {
        // a - a = 0 → log|0| = -∞.
        let ln2 = 2.0_f64.ln();
        let (lm, sg) = signed_log_sum_exp(&[ln2, ln2], &[1.0, -1.0]);
        assert_eq!(lm, f64::NEG_INFINITY);
        assert_eq!(sg, 0.0);
    }

    #[test]
    fn slse_compensated_signed_reduction_preserves_conditioned_residual() {
        // High-precision truth for these exact f64 log inputs is
        // -7.141194316117315021451...e-13. Reducing the positive and negative
        // groups through separate logarithms first returned
        // -7.141196119493781e-13: two otherwise harmless log roundings were
        // amplified by the nearly cancelling subtraction.
        let log_magnitudes = [
            -8.752777116220523,
            -8.741767521635955,
            -8.77021076826994,
            -8.75153786858979,
            -8.754172660745834,
            -8.768217028174623,
            -8.756625396724502,
            -8.737312647396818,
        ];
        let signs = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let (log_magnitude, sign) = signed_log_sum_exp(&log_magnitudes, &signs);
        let got = sign * log_magnitude.exp();
        let truth = -7.141194316117315e-13;
        let legacy = -7.141196119493781e-13;
        assert_eq!(sign, -1.0);
        assert!(
            (got - truth).abs() < (legacy - truth).abs(),
            "compensated signed reduction did not improve the conditioned residual: \
             got={got:.17e}, truth={truth:.17e}, legacy={legacy:.17e}"
        );
    }

    #[test]
    fn slse_log_domain_branch_retains_sub_ulp_two_term_gap() {
        // exp(-gap) rounds to 1.0 at this gap, so a purely linear-domain signed
        // reduction sees 1 - 1. The forward-error gate must route to the
        // log-domain difference, where the distinct input logs retain the gap.
        let gap = f64::EPSILON * 0.25;
        let (log_magnitude, sign) = signed_log_sum_exp(&[0.0, -gap], &[1.0, -1.0]);
        assert_eq!(sign, 1.0);
        assert_eq!(log_magnitude, log1mexp_positive(gap));
    }

    #[test]
    fn exact_binary64_sum_sign_resolves_midpoint_and_both_adjacent_sides() {
        let half_upper_ulp_at_one = 2.0_f64.powi(-53);
        let least_subnormal = f64::from_bits(1);
        assert_eq!(
            exact_binary64_sum_sign([
                1.0,
                half_upper_ulp_at_one,
                -1.0,
                -half_upper_ulp_at_one,
            ]),
            Ok(std::cmp::Ordering::Equal),
            "an exact rounding midpoint must compare equal"
        );
        assert_eq!(
            exact_binary64_sum_sign([
                1.0,
                half_upper_ulp_at_one,
                least_subnormal,
                -1.0,
                -half_upper_ulp_at_one,
            ]),
            Ok(std::cmp::Ordering::Greater),
            "one binary lattice quantum above the midpoint must compare positive"
        );
        assert_eq!(
            exact_binary64_sum_sign([
                1.0,
                half_upper_ulp_at_one,
                -least_subnormal,
                -1.0,
                -half_upper_ulp_at_one,
            ]),
            Ok(std::cmp::Ordering::Less),
            "one binary lattice quantum below the midpoint must compare negative"
        );
    }

    #[test]
    fn exact_binary64_sum_sign_enforces_its_finite_structural_contract() {
        assert_eq!(
            exact_binary64_sum_sign([f64::MAX, -f64::MAX, f64::from_bits(1)]),
            Ok(std::cmp::Ordering::Greater),
        );
        assert_eq!(
            exact_binary64_sum_sign([0.0, f64::NAN]),
            Err(ExactBinary64SumSignError::NonFiniteTerm { index: 1 }),
        );
        assert_eq!(
            exact_binary64_sum_sign(
                std::iter::repeat_n(1.0, EXACT_BINARY64_SUM_MAX_TERMS + 1)
            ),
            Err(ExactBinary64SumSignError::TermCapacityExceeded {
                maximum: EXACT_BINARY64_SUM_MAX_TERMS,
            }),
        );
    }

    #[test]
    fn slse_empty_returns_neg_inf_with_zero_sign() {
        // With no terms the sum is exactly 0, so the docstring contract is
        // `(−∞, 0.0)`. (This test previously encoded the buggy `+1.0` positive-sum
        // convention, which contradicted both the docstring and the cancellation
        // test below; rewritten to the correct zero sign.)
        let (lm, sg) = signed_log_sum_exp(&[], &[]);
        assert_eq!(lm, f64::NEG_INFINITY);
        assert_eq!(sg, 0.0);
    }

    #[test]
    fn slse_all_zero_signs_return_zero_sign() {
        // A single term whose sign is 0 contributes nothing; S = 0 ⇒ (−∞, 0.0).
        let (lm, sg) = signed_log_sum_exp(&[0.0], &[0.0]);
        assert_eq!(lm, f64::NEG_INFINITY);
        assert_eq!(sg, 0.0);
    }

    #[test]
    fn slse_all_neg_inf_magnitudes_return_zero_sign() {
        // Every magnitude is exp(−∞) = 0 regardless of sign, so the sum is 0 and
        // the reported sign must be 0.0, not +1.0.
        let (lm, sg) = signed_log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY], &[1.0, -1.0]);
        assert_eq!(lm, f64::NEG_INFINITY);
        assert_eq!(sg, 0.0);
    }

    #[test]
    fn slse_pos_inf_dominates() {
        let (lm, sg) = signed_log_sum_exp(&[f64::INFINITY, 1.0], &[1.0, -1.0]);
        assert_eq!(lm, f64::INFINITY);
        assert_eq!(sg, 1.0);
    }

    #[test]
    fn slse_neg_inf_dominates() {
        let (lm, sg) = signed_log_sum_exp(&[f64::INFINITY, 1.0], &[-1.0, 1.0]);
        assert_eq!(lm, f64::INFINITY);
        assert_eq!(sg, -1.0);
    }

    #[test]
    fn slse_both_inf_signs_gives_nan() {
        let (lm, sg) = signed_log_sum_exp(&[f64::INFINITY, f64::INFINITY], &[1.0, -1.0]);
        assert!(lm.is_nan());
        assert_eq!(sg, 0.0);
    }

    // ── normal_logcdf ─────────────────────────────────────────────────────────

    #[test]
    fn logcdf_at_zero_is_log_half() {
        let got = normal_logcdf(0.0);
        let expected = 0.5_f64.ln();
        assert!((got - expected).abs() < TOL, "got={got}");
    }

    #[test]
    fn logcdf_pos_inf_is_zero() {
        assert_eq!(normal_logcdf(f64::INFINITY), 0.0);
    }

    #[test]
    fn logcdf_neg_inf_is_neg_inf() {
        assert_eq!(normal_logcdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn logcdf_nan_is_nan() {
        assert!(normal_logcdf(f64::NAN).is_nan());
    }

    #[test]
    fn logcdf_matches_log_cdf_for_moderate_x() {
        for &x in &[-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0] {
            let got = normal_logcdf(x);
            let expected = normal_cdf(x).ln();
            assert!(
                (got - expected).abs() < 1e-10,
                "x={x}: got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn logcdf_deep_left_tail_stays_finite() {
        // For very negative x, normal_cdf(x) underflows to 0, but logcdf should
        // remain finite and large-negative.
        let got = normal_logcdf(-20.0);
        assert!(got.is_finite() && got < -100.0, "logcdf(-20)={got}");
    }

    #[test]
    fn logcdf_positive_tail_does_not_round_through_unit_cdf() {
        let x = 10.0_f64;
        let got = normal_logcdf(x);
        let expected = (-0.5 * erfc(x / std::f64::consts::SQRT_2)).ln_1p();
        assert!(
            got < 0.0,
            "logcdf(10) must retain its negative tail: {got:e}"
        );
        assert_eq!(got.to_bits(), expected.to_bits());
    }

    #[test]
    fn log_cdf_quantile_round_trips_both_unrepresentable_tails() {
        for x in [-1.0e6, -40.0, -10.0, -2.0, 0.0, 2.0, 10.0] {
            let log_p = normal_logcdf(x);
            let recovered = standard_normal_quantile_from_log_cdf(log_p)
                .expect("finite strict log-CDF has a quantile");
            assert!(
                (recovered - x).abs() <= 2.0e-12 * x.abs().max(1.0),
                "log-quantile round trip at x={x}: log_p={log_p}, recovered={recovered}"
            );
        }
    }

    // ── normal_logsf ─────────────────────────────────────────────────────────

    #[test]
    fn logsf_at_zero_is_log_half() {
        let got = normal_logsf(0.0);
        let expected = 0.5_f64.ln();
        assert!((got - expected).abs() < TOL, "got={got}");
    }

    #[test]
    fn logsf_mirrors_logcdf() {
        // logsf(x) = logcdf(-x) by definition.
        for &x in &[-3.0_f64, -1.0, 0.0, 1.0, 3.0] {
            assert_eq!(normal_logsf(x), normal_logcdf(-x));
        }
    }

    // ── signed_probit_logcdf_and_mills_ratio ──────────────────────────────────

    #[test]
    fn probit_at_pos_inf() {
        let (lc, mr) = signed_probit_logcdf_and_mills_ratio(f64::INFINITY);
        assert_eq!(lc, 0.0);
        assert_eq!(mr, 0.0);
    }

    #[test]
    fn probit_at_neg_inf() {
        let (lc, mr) = signed_probit_logcdf_and_mills_ratio(f64::NEG_INFINITY);
        assert_eq!(lc, f64::NEG_INFINITY);
        assert_eq!(mr, f64::INFINITY);
    }

    #[test]
    fn probit_nan_propagates() {
        let (lc, mr) = signed_probit_logcdf_and_mills_ratio(f64::NAN);
        assert!(lc.is_nan() && mr.is_nan());
    }

    #[test]
    fn probit_at_zero_logcdf_and_mills() {
        let (lc, mr) = signed_probit_logcdf_and_mills_ratio(0.0);
        assert!((lc - 0.5_f64.ln()).abs() < TOL, "lc={lc}");
        // phi(0)/Phi(0) = 0.3989.../0.5 ≈ 0.7979.
        assert!((mr - 0.797_884_560_802_865).abs() < 1e-10, "mr={mr}");
    }

    #[test]
    fn probit_positive_branch_matches_logcdf() {
        for &x in &[0.5_f64, 1.0, 2.0, 3.0] {
            let (lc, mr) = signed_probit_logcdf_and_mills_ratio(x);
            let lc_ref = normal_logcdf(x);
            let mr_ref = normal_pdf(x) / normal_cdf(x);
            assert!(
                (lc - lc_ref).abs() < 1e-10,
                "x={x}: lc={lc} lc_ref={lc_ref}"
            );
            assert!(
                (mr - mr_ref).abs() < 1e-10,
                "x={x}: mr={mr} mr_ref={mr_ref}"
            );
        }
    }

    #[test]
    fn probit_negative_branch_matches_logcdf() {
        for &x in &[-0.5_f64, -1.0, -2.0, -5.0] {
            let (lc, mr) = signed_probit_logcdf_and_mills_ratio(x);
            let lc_ref = normal_logcdf(x);
            assert!(
                (lc - lc_ref).abs() < 1e-10,
                "x={x}: lc={lc} lc_ref={lc_ref}"
            );
            assert!(mr.is_finite() && mr > 0.0, "x={x}: mr={mr}");
        }
    }

    #[test]
    fn probit_mills_ratio_has_no_deep_tail_floor() {
        let x = -1.0e305_f64;
        let (log_cdf, mills_ratio) = signed_probit_logcdf_and_mills_ratio(x);
        assert_eq!(log_cdf, f64::NEG_INFINITY);
        assert!(mills_ratio.is_finite());
        assert!(
            ((mills_ratio / -x) - 1.0).abs() < 5.0e-15,
            "mills({x:e})={mills_ratio:e}"
        );
    }

    #[test]
    fn normal_logcdf_derivative_stack_has_honest_infinite_limits() {
        assert_eq!(normal_logcdf_derivatives(f64::INFINITY), [0.0; 5]);
        assert_eq!(
            normal_logcdf_derivatives(f64::NEG_INFINITY),
            [f64::NEG_INFINITY, f64::INFINITY, -1.0, 0.0, 0.0]
        );
        assert!(
            normal_logcdf_derivatives(f64::NAN)
                .into_iter()
                .all(f64::is_nan)
        );

        for x in [-1.0e200_f64, 1.0e200_f64] {
            let derivatives = normal_logcdf_derivatives(x);
            assert!(
                derivatives.into_iter().all(|value| !value.is_nan()),
                "NaN derivative at x={x:e}: {derivatives:?}"
            );
        }
    }

    #[test]
    fn normal_logcdf_left_tail_derivatives_do_not_cancel() {
        let x = -1.0e100_f64;
        let derivatives = normal_logcdf_derivatives(x);
        assert_eq!(derivatives[2], -1.0);
        assert!(derivatives[3] > 0.0 && derivatives[3].is_finite());
        assert!(
            (derivatives[3] / 2.0e-300 - 1.0).abs() < 2.0e-14,
            "third derivative={:e}",
            derivatives[3]
        );
        assert_eq!(derivatives[4], 0.0);
    }

    #[test]
    fn normal_logcdf_right_tail_preserves_weighted_subnormal_derivatives() {
        let derivatives = normal_logcdf_derivatives(38.6);
        assert_eq!(derivatives[1], 0.0);
        assert!(derivatives[2] < 0.0 && derivatives[2].is_subnormal());
        assert!(derivatives[3] > 0.0 && derivatives[3].is_subnormal());
        assert!(derivatives[4] < 0.0 && derivatives[4].is_subnormal());
    }

    #[test]
    fn normal_logcdf_tail_stack_is_finite_difference_consistent() {
        let h = 1.0e-4_f64;
        for x in [-8.0_f64, -4.0, 8.0, 20.0] {
            let center = normal_logcdf_derivatives(x);
            let left = normal_logcdf_derivatives(x - h);
            let right = normal_logcdf_derivatives(x + h);
            for order in 1..=3 {
                let finite_difference = (right[order] - left[order]) / (2.0 * h);
                let expected = center[order + 1];
                let relative = (finite_difference - expected).abs() / expected.abs().max(1.0e-300);
                assert!(
                    relative < 2.0e-5,
                    "x={x}, order={order}: fd={finite_difference:e}, expected={expected:e}, rel={relative:e}"
                );
            }
        }
    }

    /// Absolute-accuracy pin of the full `ln Φ(x)` derivative tower against an
    /// EXTERNAL high-precision reference (mpmath, dps=60), covering all three
    /// branches (continued-fraction left tail at x=−4, the moderate Mills
    /// recurrence for x∈(−4, 8), and both signs). Before the `erfc` root-cause
    /// fix the moderate branch's `λ = φ/Φ` inherited `statrs::erfc`'s ~1e-10
    /// error, so `f''` was wrong by ~1e-9 near the −4 seam; this pins every
    /// entry to `2e-11` relative, catching that regression head-on rather than
    /// through a seam-straddling finite difference.
    #[test]
    fn normal_logcdf_derivative_tower_matches_high_precision_reference() {
        // (x, [value, f', f'', f''', f''''] from mpmath at dps=60).
        let refs: &[(f64, [f64; 5])] = &[
            (
                -4.0,
                [
                    -10.360101486527291,
                    4.2256071444894711,
                    -0.95332716160257737,
                    0.017856339307658426,
                    0.0095065764315958691,
                ],
            ),
            // Two points well inside the continued-fraction branch, where the
            // truncation the depth controls is the ONLY error source: at -4 the
            // branch is at its least converged, and these confirm it stays put.
            (
                -10.0,
                [
                    -53.231285150512471,
                    10.098093233962512,
                    -0.99055462217434374,
                    0.0017864003921165069,
                    0.00049785382237944016,
                ],
            ),
            (
                -6.0,
                [
                    -20.736768949974706,
                    6.1584826045445989,
                    -0.97601236321083323,
                    0.0069535374991643118,
                    0.0028992056785575027,
                ],
            ),
            (
                -2.0,
                [
                    -3.7831843336820319,
                    2.3732155328228409,
                    -0.88572089958591874,
                    0.059355861291565813,
                    0.039421993865946813,
                ],
            ),
            (
                -1.0,
                [
                    -1.8410216450092635,
                    1.5251352761609812,
                    -0.80090233442965121,
                    0.11693119540604883,
                    0.07917498368074563,
                ],
            ),
            (
                -0.3,
                [
                    -0.96210281816885066,
                    0.99816596885848332,
                    -0.69688551072964971,
                    0.18398317992442132,
                    0.11037564722092704,
                ],
            ),
            (
                0.5,
                [
                    -0.36894641528865639,
                    0.50916043383703349,
                    -0.5138245643036329,
                    0.27099012446870783,
                    0.088167801929197554,
                ],
            ),
            (
                2.0,
                [
                    -0.023012909328963488,
                    0.055247862678989959,
                    -0.11354805168857645,
                    0.18439481503247759,
                    -0.18785468561160969,
                ],
            ),
        ];
        // The moderate-branch statrs regression produced ~1e-9 errors in f''.
        // The bound used to sit at 1e-10 to respect what was called the
        // continued-fraction branch's "inherent" ~2e-11 in f''''; that was not
        // inherent but a depth, and at 64 levels the branch reproduces this
        // 60-digit reference EXACTLY at x = -4, -6 and -10. What remains is the
        // moderate branch, where the brackets are already collected in `q` and
        // the floor is `λ`'s own relative error amplified by `λ/q` (18.7 at the
        // switch): 1.8e-13 at x = -2, the worst point here. 1e-11 keeps 55x of
        // headroom over that while still failing the 32-level truncation head-on.
        for &(x, reference) in refs {
            let got = normal_logcdf_derivatives(x);
            for (order, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
                let rel = (g - r).abs() / r.abs().max(1.0e-3);
                assert!(
                    rel < 1.0e-11,
                    "normal_logcdf_derivatives({x})[{order}] = {g:.17e}, reference {r:.17e}, \
                     rel {rel:.3e} >= 1e-11"
                );
            }
        }
    }

    // ── standard_normal_quantile ──────────────────────────────────────────────

    #[test]
    fn quantile_rejects_out_of_range() {
        assert!(standard_normal_quantile(0.0).is_err());
        assert!(standard_normal_quantile(1.0).is_err());
        assert!(standard_normal_quantile(-0.1).is_err());
        assert!(standard_normal_quantile(1.1).is_err());
        assert!(standard_normal_quantile(f64::NAN).is_err());
    }

    #[test]
    fn quantile_at_half_is_near_zero() {
        let q = standard_normal_quantile(0.5).unwrap();
        assert!(q.abs() < 1e-10, "quantile(0.5)={q}");
    }

    #[test]
    fn quantile_at_0975_is_near_196() {
        let q = standard_normal_quantile(0.975).unwrap();
        assert!((q - 1.959_963_984_540_054).abs() < 1e-14, "q={q}");
    }

    /// `standard_normal_quantile` and its log-CDF sibling, against a 120-digit
    /// root of `Φ(x) = p` (respectively `ln Φ(x) = log_p`).
    ///
    /// The seed is Acklam's rational approximation, whose accuracy is `1.15e-9`
    /// relative; the two Halley steps after it are what make the result
    /// ulp-accurate. Deleting the polish loop entirely leaves EVERY other
    /// quantile test in this module green except `quantile_roundtrip_cdf`, and
    /// that one only by a factor of 1.9 — so the polish had no real gate. This
    /// table is that gate: it fails by six orders if the seed ships unpolished.
    ///
    /// The grid straddles Acklam's own `P_LOW = 0.02425` branch on both sides,
    /// runs out to `p = 1e-300` where the seed is far from the root, and covers
    /// the reflected upper tail where the residual must be formed from
    /// `(1 − p) − ½erfc(x/√2)` rather than `Φ(x) − p`.
    /// The CENTRAL band, where the residual `F(x) − p` must never be formed
    /// against `½`.
    ///
    /// The sibling table above straddles Acklam's `P_LOW` branch and runs into
    /// both tails, but its tightest central point is `p = 0.5000000001`. That
    /// is not where the old residual failed. Forming `F(x) − p` as
    /// `(1 − p) − ½erfc(x/√2)` (or `F(x) − p` directly) subtracts two numbers
    /// of size ~½, so the residual carries a FIXED absolute error of one ulp of
    /// ½ however small the true residual is; `Δx ≈ residual_error / φ(x)` then
    /// pins the quantile's ABSOLUTE error at ~1.2e-16 and lets its RELATIVE
    /// error grow like `1.2e-16 / |x|` without bound as `p → ½`.
    ///
    /// Measured against a 50-digit `erfinv` reference at the exact `f64`
    /// abscissae below, before the centered residual and after:
    ///
    /// | `p`             | before   | after   |
    /// |-----------------|----------|---------|
    /// | `½ + 2⁻⁴⁵`      | 1.13e-09 | 2.3e-16 |
    /// | `0.5012506…`    | 7.31e-15 | 2.3e-16 |
    /// | `0.4987493…`    | 7.33e-15 | 2.3e-16 |
    ///
    /// The `1.13e-09` is not a coincidence: it is `|A[5] − √(2π)| / √(2π)`,
    /// Acklam's own advertised accuracy. As `p → ½` the seed reduces to
    /// `A[5]·(p − ½)` and the polish is handed a residual quantized to
    /// multiples of one ulp of ½ — usually exactly `0` — so the raw seed is
    /// what shipped.
    ///
    /// The bar is `4·f64::EPSILON` relative: half an ulp for the correctly
    /// rounded reference literal, the rest for the evaluator. Worst measured
    /// margin over this table is 1.0 ulp.
    #[test]
    fn normal_quantile_is_ulp_accurate_through_the_median() {
        // `[p, Φ⁻¹(p)]`, the second entry correctly rounded from a 50-digit
        // `sqrt(2)·erfinv(2p − 1)` evaluated at the EXACT binary `p`.
        const CENTRAL_REFERENCE: [[f64; 2]; 19] = [
            [0.5000000000000284, 7.124266047159724e-14],
            [0.4999999999999716, -7.124266047159724e-14],
            [0.5000000009313226, 2.3344794983332983e-09],
            [0.4999999990686774, -2.3344794983332983e-09],
            [0.5000009536743164, 2.390507006295574e-06],
            [0.500000001, 2.5066282037387115e-09],
            [0.4999999999, -2.506628482030354e-10],
            [0.5001, 0.00025066283008800747],
            [0.4999, -0.00025066283008800747],
            [0.51, 0.025068908258711057],
            [0.49, -0.025068908258711057],
            [0.55, 0.12566134685507416],
            [0.45, -0.12566134685507402],
            [0.6, 0.2533471031357997],
            [0.4, -0.2533471031357997],
            [0.7, 0.5244005127080407],
            [0.3, -0.5244005127080408],
            [0.75, 0.6744897501960817],
            [0.25, -0.6744897501960817],
        ];
        let bar = 4.0 * f64::EPSILON;
        let mut worst = 0.0_f64;
        let mut worst_at = f64::NAN;
        for [p, expected] in CENTRAL_REFERENCE {
            let got = standard_normal_quantile(p).expect("central p is in (0,1)");
            let relative = ((got - expected) / expected).abs();
            if relative > worst {
                worst = relative;
                worst_at = p;
            }
            assert!(
                relative <= bar,
                "Phi^-1({p}) = {got}, expected {expected}, relative {relative:e} > {bar:e}"
            );
        }
        println!("central quantile worst relative {worst:e} at p = {worst_at}");
    }

    #[test]
    fn normal_quantiles_match_independent_high_precision_reference() {
        const QUANTILE_REFERENCE: [[f64; 2]; 22] = [
            [1e-300, -37.0470962993612],
            [1e-100, -21.273453560965326],
            [1e-20, -9.262340089798407],
            [1e-08, -5.612001244174789],
            [0.001, -3.0902323061678136],
            [0.02424, -1.9731366119445441],
            [0.02425, -1.972961051311885],
            [0.02426, -1.9727855514678605],
            [0.05, -1.6448536269514726],
            [0.1, -1.2815515655446004],
            [0.25, -0.6744897501960817],
            [0.4, -0.2533471031357997],
            [0.5, 0.0],
            [0.6, 0.2533471031357997],
            [0.75, 0.6744897501960817],
            [0.9, 1.2815515655446006],
            [0.95, 1.6448536269514722],
            [0.975, 1.9599639845400538],
            [0.99, 2.3263478740408408],
            [0.999, 3.090232306167813],
            [0.99999999, 5.612001243305505],
            [0.9999999999999999, 8.209536151601387],
        ];
        for [p, want] in QUANTILE_REFERENCE {
            let got = standard_normal_quantile(p).expect("p in (0,1) has a quantile");
            let error = (got - want).abs();
            // `Φ⁻¹(½) = 0` exactly, so it is the one absolute comparison.
            let budget = if want == 0.0 {
                1e-16
            } else {
                4e-15 * want.abs()
            };
            assert!(
                error <= budget,
                "Φ⁻¹({p}): got {got:.17e}, want {want:.17e} (error {error:.3e} > {budget:.3e})"
            );
        }

        const LOG_CDF_QUANTILE_REFERENCE: [[f64; 2]; 9] = [
            [-0.7, -0.008559478582480282],
            [-2.0, -1.1015196284987503],
            [-10.0, -3.913946240531893],
            [-50.0, -9.674825283612357],
            [-200.0, -19.803669380301212],
            [-1000.0, -44.6157477319694],
            [-10000.0, -141.37983987312717],
            [-100000.0, -447.1978936785251],
            [-1000000.0, -1414.2077829910174],
        ];
        for [log_p, want] in LOG_CDF_QUANTILE_REFERENCE {
            let got =
                standard_normal_quantile_from_log_cdf(log_p).expect("finite log_p < 0 has a root");
            let error = (got - want).abs();
            // Rounding `log_p` itself to `f64` already moves the root by
            // `ulp(log_p)·dx/d(log_p)`, and `dx/d(log_p) = Φ/φ = 1/λ` — about
            // `1.25` near `p = ½` and `≈ 1/|x|` in the deep tail. That input
            // conditioning, not the solver, is what limits `log_p = −0.7`,
            // where the root sits at `−0.00856` and one ulp of `0.7` is already
            // `1.4e-16` of it.
            let conditioning = 8.0 * f64::EPSILON * log_p.abs() / want.abs().max(0.8);
            let budget = 4e-15 * want.abs() + conditioning;
            assert!(
                error <= budget,
                "Φ⁻¹(exp({log_p})): got {got:.17e}, want {want:.17e} \
                 (error {error:.3e} > {budget:.3e})"
            );
        }
    }

    #[test]
    fn quantile_antisymmetry() {
        let q_lo = standard_normal_quantile(0.1).unwrap();
        let q_hi = standard_normal_quantile(0.9).unwrap();
        assert!((q_lo + q_hi).abs() < 1e-10, "q_lo={q_lo} q_hi={q_hi}");
    }

    #[test]
    fn quantile_roundtrip_cdf() {
        for &p in &[
            0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999,
        ] {
            let q = standard_normal_quantile(p).unwrap();
            let p_back = normal_cdf(q);
            // RELATIVE, and sized by what the round trip can cost: a few ulp of
            // `q` propagated through `φ(q)`, plus a couple of ulp from `erfc`
            // itself. The former absolute `1e-10` bar was two orders looser than
            // an unpolished Acklam seed at its worst point.
            assert!(
                (p_back - p).abs() <= 1e-14 * p,
                "roundtrip failed at p={p}: q={q} p_back={p_back}"
            );
        }
    }
}

/// The SIGNED, multiplicity-carrying form — the generalization the estimated-
/// scale references need (gam#2672).
/// Standard normal survival probability `P(Z > x)`.
///
/// This is evaluated as `½·erfc(x/√2)`, not as `1 − Φ(x)`. The latter loses
/// relative accuracy as soon as `Φ(x)` approaches one and becomes identically
/// zero for every representable `x` above roughly `8.3`, while the direct
/// complementary form retains the full representable tail.
#[inline]
pub fn normal_sf(x: f64) -> f64 {
    0.5 * erfc(x / std::f64::consts::SQRT_2)
}

/// Student-t survival probability `P(T_ν > t)`.
///
/// The small tail is always obtained from
/// [`student_t_two_sided_probability`]. For negative `t`, subtracting its
/// half-tail from one constructs the large probability, where subtraction is
/// well conditioned.
pub fn student_t_sf(t: f64, degrees_of_freedom: f64) -> f64 {
    let two_sided = student_t_two_sided_probability(t, degrees_of_freedom);
    if t < 0.0 {
        1.0 - 0.5 * two_sided
    } else {
        0.5 * two_sided
    }
}

#[cfg(test)]
mod signed_weighted_chi_square_tests {
    use super::*;

    fn term(weight: f64, degrees_of_freedom: f64) -> WeightedChiSquareTerm {
        WeightedChiSquareTerm {
            weight,
            degrees_of_freedom,
        }
    }

    /// THE identity the signed form exists for, against a closed form computed
    /// a completely different way (the regularized incomplete beta):
    ///
    /// ```text
    /// P(F_{a,b} > f) = P( (χ²_a/a) / (χ²_b/b) > f ) = P( χ²_a − (f·a/b)·χ²_b > 0 ).
    /// ```
    ///
    /// A ratio's tail IS a signed combination evaluated at zero. Fractional `a`
    /// is included because a two-moment summary of a smooth's null spectrum is a
    /// chi-square with a non-integral shape, which is exactly what this form is
    /// asked for.
    #[test]
    fn the_f_tail_is_the_two_term_signed_combination_at_zero() {
        let mut worst = 0.0_f64;
        for &(a, b) in &[
            (1.0_f64, 5.0_f64),
            (2.0, 17.0),
            (3.0, 26.0),
            (0.7, 24.0),
            (5.4, 191.0),
            (11.0, 4.0),
        ] {
            for &f in &[0.05_f64, 0.5, 1.0, 2.5, 9.0, 40.0] {
                let terms = [term(1.0, a), term(-f * a / b, b)];
                let (got, bound) = signed_weighted_chi_square_sf_to_tolerance(
                    &terms,
                    0.0,
                    WEIGHTED_CHI_SQUARE_TOLERANCE,
                );
                let want = fisher_snedecor_sf(f, a, b);
                let error = (got - want).abs();
                worst = worst.max(error);
                assert!(
                    error <= 1e-9 + bound,
                    "F({a},{b}) at {f}: imhof {got} vs beta {want} \
                     (error {error:.3e}, certified bound {bound:.3e})"
                );
            }
        }
        println!("worst |imhof − F| over the grid: {worst:.3e}");
    }

    /// The certified bound at `statistic = 0` — where the oscillatory bound does
    /// not exist and the amplitude bound is the whole contract. Checked against
    /// a reference computed at a far stricter request, so the assertion is that
    /// the RETURNED bound actually bounds the error.
    #[test]
    fn the_amplitude_bound_certifies_the_zero_statistic_answer() {
        let cases: [&[WeightedChiSquareTerm]; 3] = [
            &[term(1.0, 1.0), term(-0.05, 26.0)],
            &[term(0.9, 1.0), term(0.2, 3.0), term(-0.01, 191.0)],
            &[term(1.0, 5.4), term(-2.5, 1.0), term(-0.004, 44.0)],
        ];
        for terms in cases {
            let (reference, reference_bound) =
                signed_weighted_chi_square_sf_to_tolerance(terms, 0.0, 1e-14);
            for tolerance in [1e-4_f64, 1e-7, 1e-10] {
                let (got, bound) =
                    signed_weighted_chi_square_sf_to_tolerance(terms, 0.0, tolerance);
                assert!(
                    bound <= tolerance,
                    "asked {tolerance:.0e}, certified {bound:.3e} on {terms:?}"
                );
                assert!(
                    (got - reference).abs() <= bound + reference_bound,
                    "{got} vs {reference} exceeds the certified {bound:.3e} + \
                     {reference_bound:.3e} on {terms:?}"
                );
            }
        }
    }

    /// The panel rule has to resolve the integrand's AMPLITUDE, not only its
    /// phase, and this is the arm that measures whether it does.
    ///
    /// The reference is the same quadrature at a panel forced far below either
    /// rule (by asking for an accuracy the sizing then honours), so the
    /// comparison isolates the discretization from the truncation. The shapes
    /// are the ones where the two scales come apart: a small phase rate
    /// (`statistic = 0`, weights that nearly cancel) against an amplitude that
    /// turns over at `u ≈ 1`.
    ///
    /// Pre-fix, `F_{1,5}` at `f = 0.05` missed by `3.4e-7` while certifying
    /// `1e-11`.
    #[test]
    fn the_quadrature_resolves_the_amplitude_not_only_the_phase() {
        let cases: [&[WeightedChiSquareTerm]; 5] = [
            &[term(1.0, 1.0), term(-0.01, 5.0)],
            &[term(1.0, 1.0), term(-0.2, 2.0)],
            &[term(1.0, 3.0), term(-1.0, 3.0)],
            &[term(0.9, 1.0), term(0.2, 4.0), term(-0.05, 26.0)],
            &[term(1.0, 0.7), term(-0.006, 24.0)],
        ];
        let mut worst = 0.0_f64;
        for terms in cases {
            for &statistic in &[0.0_f64, 0.3, -0.2] {
                let (reference, reference_bound) =
                    signed_weighted_chi_square_sf_to_tolerance(terms, statistic, 1e-15);
                let (got, bound) = signed_weighted_chi_square_sf_to_tolerance(
                    terms,
                    statistic,
                    WEIGHTED_CHI_SQUARE_TOLERANCE,
                );
                let error = (got - reference).abs();
                worst = worst.max(error);
                assert!(
                    error <= bound + reference_bound,
                    "{terms:?} at {statistic}: {got} vs {reference} differs by {error:.3e}, \
                     above the certified {bound:.3e} + {reference_bound:.3e}"
                );
            }
        }
        println!("worst discretization error against the fine-panel reference: {worst:.3e}");
    }

}
