//! Gaussian expectations of elementwise activations, with analytic derivatives.
//!
//! A known MLP block `F(z) = Σ_j u_j σ(b_j + w_jᵀ z)` under a declared Gaussian
//! intervention law reads every unit through a Gaussian pre-activation (#2946).
//! The best response of a retained subspace smooths each unit,
//! `T_v σ(t) = E σ(t + √v E)` with `E ~ N(0, 1)`, and the variance it explains
//! pairs units through `K_σ = E[σ(X) σ(Y)]` for jointly Gaussian `X`, `Y`.
//! This module owns both expectations for ReLU and for the exact GELU
//! `σ(t) = t Φ(t)`: the smoothing with its t-derivatives of every order, and
//! the pair kernel with Price's covariance derivative
//! `∂_r K_σ = E[σ'(X) σ'(Y)]`. Every value is a closed form in `Φ`, `φ` and
//! elementary functions; nothing here is a quadrature.
//!
//! The approximate GELUs (`gelu_new`, `gelu_pytorch_tanh`) are different
//! activations and have no primitive here. SiLU, the gate of SwiGLU blocks,
//! has no closed form; `gaussian_gated` owns its expectations.
//!
//! # Smoothing
//!
//! ReLU with `v > 0`, `s = √v`, `u = t/s`:
//! `T = t Φ(u) + s φ(u)`, `T' = Φ(u)`, and for `k ≥ 2`
//! `T⁽ᵏ⁾ = φ⁽ᵏ⁻²⁾(u)/sᵏ⁻¹ = (−1)ᵏ He_{k−2}(u) φ(u)/sᵏ⁻¹`.
//!
//! Exact GELU with `A = 1 + v`, `S = √A`, `u = t/S`. For `X ~ N(t, v)` and an
//! independent `N ~ N(0, 1)`, `E Φ(X) = P(N ≤ X) = Φ(u)`, and Stein's lemma
//! `E[(X − t) g(X)] = v E g'(X)` gives
//! `T = t E Φ(X) + v E φ(X) = t Φ(u) + (v/S) φ(u)`.
//! Differentiating under the expectation, `T'' = E σ''(X)` with
//! `σ''(x) = (2 − x²) φ(x)`, and `φ(x) N(x; t, v) = N(t; 0, A) N(x; t/A, v/A)`
//! gives `T'' = (A φ(u) − φ''(u))/(A S)`. Hence, for `k ≥ 1`,
//! `T⁽ᵏ⁾ = (A φ⁽ᵏ⁻²⁾(u) − φ⁽ᵏ⁾(u))/(A Sᵏ⁻¹) = (−1)ᵏ φ(u) (A He_{k−2}(u) − He_k(u))/(A Sᵏ⁻¹)`,
//! where `k = 1` reads `φ⁽⁻¹⁾ = Φ`: `T' = Φ(u) + u φ(u)/A`.
//!
//! Both smoothings solve the heat equation `∂_v T = ½ ∂_t² T`.
//!
//! The same towers give the normalized Hermite coefficients
//! `a_n = E[σ(t + s E) h_n(E)] = sⁿ T⁽ⁿ⁾_{s²}σ(t)/√(n!)`, `h_n = He_n/√(n!)`, through
//! the orthonormal recurrence `h_{k+1} = (x h_k − √k h_{k−1})/√(k+1)`:
//! ReLU `a_n = (−1)ⁿ s φ(β) h_{n−2}(β)/√(n(n−1))` with `β = t/s`, and exact GELU
//! `a_n = (−1)ⁿ φ(x) (s/S)ⁿ [A h_{n−2}(x)/√(n(n−1)) − h_n(x)]/S` with
//! `A = 1 + s²`, `S = √A` and `x = t/S`, for `n ≥ 2`.
//!
//! Left of the origin (`u < 0`) the values and slopes are formed from the
//! log-CDF slope `λ(u) = φ(u)/Φ(u)` and its correction `q(u) = λ(u) + u`, as
//! `Φ(u) = φ(u)/λ` and `1 + u/λ = q/λ`: ReLU `T = s φ(u) q/λ`, `T' = φ(u)/λ`;
//! exact GELU `T = S φ(u) (q/λ − 1/A)`, `T' = φ(u) (1/λ + u/A)`. There
//! `t Φ(u) + s φ(u)` cancels like `u²`, and `Φ(u) = ½ erfc(−u/√2)` multiplies
//! the rounding of its argument by `u²`; `λ` and `q = −f''/λ` (`f = ln Φ`, so
//! `f'' = −λ q`) come from the log-CDF owner at full relative precision.
//!
//! # Rounding bounds
//!
//! [`gaussian_hermite_coefficients`] returns next to every coefficient a
//! first-order bound on its absolute error, by running error analysis of the
//! evaluation itself under the standard model: every `+ − × ÷ √` rounds by at
//! most `u = ε/2` of its result.
//! - The standardized argument `x = t/s` (or `t/√(1 + s²)`) carries its own
//!   rounding, which moves `φ(x) h_j(x)` by at most `φ (√j |h_{j−1}| + |x| |h_j|)`
//!   per unit of argument error.
//! - `φ` comes from the probability owner's `normal_pdf_bounded`, computed with
//!   `libm::exp` and within `(5u + u²x⁴/8) φ`, resting on libm 0.2.16's cited `exp`
//!   error analysis.
//!   `Φ = ½ erfc(−x/√2)` rounds by `2u Φ + 2u |x| φ`, the second term from the
//!   rounding of `−x/√2`. The `erfc` ulp is a measurement (see `bounded_normal_cdf`).
//! - The left-tail forms read `1/λ` and `q/λ` from the probability owner's
//!   `normal_left_tail_ratios`, whose bounds are derived without any libm call.
//! - Each orthonormal step adds `3u (|x h_k| + √k |h_{k−1}|)/√(k+1) + 2u |h_{k+1}|`
//!   to the propagated `(|x| e_k + √k e_{k−1})/√(k+1)`.
//! - Where `φ(x)` underflows, the coefficients of order ≥ 2 are returned as zero
//!   with Cramér's inequality `|h_j(x)| e^{−x²/4} ≤ 1.0865`, so
//!   `φ |h_j| ≤ 1.0865 (2π)^{−1/4} √φ` with `φ` below the smallest subnormal.
//!
//! It relies on its owners' contracts:
//! - the cited libm 0.2.16 `exp` error analysis, through `normal_pdf_bounded`;
//! - the measured `erfc` ulp, for `Φ` in the direct forms;
//! - the derived arctangent of `bounded_arctangent2`, for the zero-mean kernels' orthant angle,
//!   which rests on IEEE-754 semantics only;
//! - the derived bounds of `normal_left_tail_ratios`, which rest on IEEE-754 semantics only.
//!
//! # Pair kernel
//!
//! `X ~ N(b, v)`, `Y ~ N(c, w)`, `Cov(X, Y) = r`.
//!
//! Exact GELU, with `A = 1 + v`, `B = 1 + w`, `Δ = AB − r²`. Write
//! `Φ(X) = P(N₁ ≤ X)` and `Φ(Y) = P(N₂ ≤ Y)` for independent standard normals,
//! so `K = E[X Y h(U, V)]` with `U = X − N₁`, `V = Y − N₂` and
//! `h = 1{U ≥ 0} 1{V ≥ 0}`. `(U, V)` has means `(b, c)`, variances `(A, B)` and
//! covariance `r`, so `H(b, c) = E h = Φ₂(b/√A, c/√B; r/√(AB))`. Second-order
//! Gaussian integration by parts gives
//! `K = (bc + r) H + (cv + br) H_b + (bw + cr) H_c + vr H_bb + (vw + r²) H_bc + rw H_cc`,
//! and `H_bb = −(b H_b + r H_bc)/A`, `H_cc = −(c H_c + r H_bc)/B` reduce it to
//! `K = (bc + r) H + (cv + br/A) H_b + (bw + cr/B) H_c + (vw + r² q) H_bc`, with
//! `q = 1 − v/A − w/B` and `H_bc = exp(−(B b² − 2rbc + A c²)/(2Δ))/(2π √Δ)`.
//! Price's theorem (`∂_r H = H_bc`, `∂_r H_b = −b̃ H_bc`, `∂_r H_c = −c̃ H_bc`,
//! `∂_r H_bc = (r/Δ + b̃ c̃) H_bc`, with `b̃ = (Bb − rc)/Δ`, `c̃ = (Ac − rb)/Δ`)
//! differentiates it to
//! `∂_r K = H + (b H_b + r H_bc)/A + (c H_c + r H_bc)/B + (r/Δ + b̃ c̃) H_bc`.
//! At zero mean `H = ¼ + arcsin(r/√(AB))/(2π) = arccos(−r/√(AB))/(2π)`, so
//! `K = r H + (vw + r² q)/(2π √Δ)` and `∂_r K = H + r (1/A + 1/B + 1/Δ)/(2π √Δ)`.
//!
//! ReLU by the same integration by parts with `h = 1{X ≥ 0} 1{Y ≥ 0}`:
//! `K = (bc + r) H + cv H_b + bw H_c + (vw − r²) H_bc` with
//! `H = Φ₂(b/√v, c/√w; r/√(vw))`, and `∂_r K = H`. At zero mean, with
//! `ρ = r/√(vw) = cos θ`, this is the arc-cosine kernel
//! `K = √(vw) (sin θ + (π − θ) cos θ)/(2π)` with `∂_r K = (π − θ)/(2π)`.
//!
//! Both zero-mean kernels take the orthant angle from the exactly carried
//! residual `vw − r²`: `H = atan2(√(vw − r²), −r)/(2π)` for ReLU and
//! `H = atan2(√Δ, −r)/(2π)` for the exact GELU, each through the derived
//! `bounded_arctangent2` rather than the platform's `atan2`. `arccos(−ρ)` of a rounded `ρ`
//! multiplies that rounding by `1/√(1 − ρ²)`; for anti-correlated large-norm
//! readers the GELU kernel's two `√v`-sized terms cancel on top of it, which
//! reached 90% relative error at `v = w = −r = 1e8` (#2946).
//!
//! The biased kernels take `H` and its partials from the bivariate normal owner,
//! passing `1 − ρ²` from the exactly carried residual: `Δ/(AB)` for the exact GELU
//! and `(vw − r²)/(vw)` for ReLU. Then `H_b = ∂_hΦ₂/√A`, `H_c = ∂_kΦ₂/√B` and
//! `H_bc = φ₂(h, k; ρ)/√(AB)` (`v`, `w` in place of `A`, `B` for ReLU). On the
//! degenerate ReLU law `vw = r²` the partials take their limits, `φ(h)·1[h < k]`
//! at `ρ = 1` and `φ(h)·1[h > −k]` at `ρ = −1` with `½` at the tie, and
//! `(vw − r²) H_bc` vanishes. A zero variance makes its pre-activation the
//! constant mean, `K = σ(b) T_w σ(c)` and `∂_r K = σ'(b) T_w σ'(c)`.

use crate::bivariate_normal::{
    BIVARIATE_NORMAL_CDF_ERROR_BOUND, BivariateNormalError,
    bivariate_normal_cdf_partials_with_complement, bivariate_normal_cdf_with_complement,
    bivariate_normal_cdf_with_complement_bounded,
};
use crate::double_double::BoundedDoubleDouble;
use crate::probability::{normal_cdf, normal_left_tail_ratios, normal_pdf_bounded};
use crate::roundoff::{UNIT_ROUNDOFF, inflated};
use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::fmt;
use std::sync::LazyLock;

/// An elementwise activation of a known MLP block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaussianActivation {
    /// `σ(t) = max(t, 0)`.
    Relu,
    /// The exact GELU `σ(t) = t Φ(t)`, not its `tanh` or sigmoid approximations.
    ExactGelu,
    /// SiLU (swish) `σ(t) = t/(1 + e^{−t})`, the gate of SwiGLU blocks. Its
    /// Gaussian expectations have no closed form: `gaussian_gated` owns them,
    /// and the closed-form entries here refuse it.
    Silu,
}

impl GaussianActivation {
    /// The activation a Hugging Face `hidden_act` tag names.
    ///
    /// `gelu` is the exact erf GELU. `gelu_new` and `gelu_pytorch_tanh` are its
    /// `tanh` approximation, a different activation, and are refused as such;
    /// every tag not named here is refused, with no fallback.
    pub fn from_hidden_act(tag: &str) -> Result<Self, GaussianActivationError> {
        match tag {
            "relu" => Ok(Self::Relu),
            "gelu" => Ok(Self::ExactGelu),
            "silu" | "swish" => Ok(Self::Silu),
            "gelu_new" | "gelu_pytorch_tanh" => Err(GaussianActivationError::ApproximateGelu {
                tag: tag.to_owned(),
            }),
            _ => Err(GaussianActivationError::UnsupportedHiddenAct {
                tag: tag.to_owned(),
            }),
        }
    }

    /// An upper bound on `sup_x |σ'(x)|²`, the covariance Lipschitz constant of
    /// every pair kernel: `|∂_r K| = |E σ'(X) σ'(Y)| ≤ sup|σ'|²`, so moving the
    /// covariance by `δ` at fixed means and variances moves `K` by at most this
    /// bound times `δ`.
    ///
    /// - ReLU: `σ' ∈ {0, 1}`, so the bound is `1`.
    /// - Exact GELU: `σ'(x) = Φ(x) + x φ(x)` and `σ''(x) = (2 − x²) φ(x)`, so `σ'`
    ///   rises on `(−√2, √2)` from its minimum to its maximum. By the symmetry
    ///   `σ'(−x) = 1 − σ'(x)`, `sup|σ'| = σ'(√2) = Φ(√2) + √2 φ(√2) = 1.1289…`.
    ///   It is evaluated with its rounding, and the square plus its bound is
    ///   returned, so the bound holds. Rounding `√2` moves `σ'` only to second
    ///   order, because `σ''(√2) = 0`.
    /// - SiLU has no closed form here, and `gaussian_gated` owns its expectations.
    #[inline]
    pub fn slope_bound_squared(self) -> Result<f64, GaussianActivationError> {
        match self {
            Self::Relu => Ok(1.0),
            Self::ExactGelu => {
                let argument = Bounded::exact(std::f64::consts::SQRT_2);
                let slope = bounded_normal_cdf(argument)
                    .add(argument.mul(bounded_normal_pdf(argument)));
                let square = slope.mul(slope);
                Ok(square.value + square.bound)
            }
            Self::Silu => Err(GaussianActivationError::NoClosedForm { activation: self }),
        }
    }
}

/// A refused Gaussian activation expectation.
#[derive(Clone, Debug, PartialEq)]
pub enum GaussianActivationError {
    /// A pre-activation variance must be finite and nonnegative.
    InvalidVariance { variance: f64 },
    /// A pre-activation scale must be finite and nonnegative.
    InvalidScale { scale: f64 },
    /// Coefficients and their bounds must have one entry per order.
    MismatchedBoundsLength { coefficients: usize, bounds: usize },
    /// A mean, covariance or evaluation point must be finite.
    NonFiniteArgument { value: f64 },
    /// A covariance rounding bound must be finite and nonnegative.
    InvalidCovarianceRounding { bound: f64 },
    /// `r² − vw` exceeds what the stated rounding of the Gram inputs produces.
    CovarianceOutsideCauchySchwarz {
        variance_x: f64,
        variance_y: f64,
        covariance: f64,
        covariance_rounding: f64,
    },
    /// ReLU with zero smoothing variance has no t-derivative of order ≥ 2 at
    /// its kink `t = 0`: the second derivative there is a Dirac mass.
    UnsmoothedReluKink { order: usize },
    /// The bivariate normal owner refused the standardized pair law.
    BivariateNormal { source: BivariateNormalError },
    /// The activation has no closed-form Gaussian expectation here.
    NoClosedForm { activation: GaussianActivation },
    /// A `hidden_act` tag naming a GELU approximation, a different activation.
    ApproximateGelu { tag: String },
    /// A `hidden_act` tag that names no activation here.
    UnsupportedHiddenAct { tag: String },
}

impl fmt::Display for GaussianActivationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVariance { variance } => write!(
                formatter,
                "a Gaussian pre-activation variance must be finite and nonnegative, got {variance}"
            ),
            Self::InvalidScale { scale } => write!(
                formatter,
                "a Gaussian pre-activation scale must be finite and nonnegative, got {scale}"
            ),
            Self::MismatchedBoundsLength {
                coefficients,
                bounds,
            } => write!(
                formatter,
                "coefficients and their bounds need one entry per order, got {coefficients} and {bounds}"
            ),
            Self::NonFiniteArgument { value } => write!(
                formatter,
                "a Gaussian pre-activation mean, covariance or evaluation point must be finite, got {value}"
            ),
            Self::InvalidCovarianceRounding { bound } => write!(
                formatter,
                "a covariance rounding bound must be finite and nonnegative, got {bound}"
            ),
            Self::CovarianceOutsideCauchySchwarz {
                variance_x,
                variance_y,
                covariance,
                covariance_rounding,
            } => write!(
                formatter,
                "covariance {covariance} exceeds √({variance_x}·{variance_y}) by more than its stated rounding {covariance_rounding}"
            ),
            Self::UnsmoothedReluKink { order } => write!(
                formatter,
                "ReLU without smoothing variance has no t-derivative of order {order} at its kink t = 0"
            ),
            Self::BivariateNormal { source } => write!(
                formatter,
                "the bivariate normal owner refused the standardized pair law: {source}"
            ),
            Self::NoClosedForm { activation } => write!(
                formatter,
                "{activation:?} has no closed-form Gaussian expectation; gaussian_gated owns its quadrature"
            ),
            Self::ApproximateGelu { tag } => write!(
                formatter,
                "hidden_act {tag:?} is a tanh approximation of the GELU, a different activation from the exact erf GELU"
            ),
            Self::UnsupportedHiddenAct { tag } => {
                write!(formatter, "hidden_act {tag:?} names no supported activation")
            }
        }
    }
}

impl std::error::Error for GaussianActivationError {}

/// Fills `derivatives[k]` with `∂ᵏ/∂tᵏ T_v σ(t)`, `T_v σ(t) = E σ(t + √v E)`,
/// for every `k < derivatives.len()`. `T_v σ⁽ᵏ⁾(t) = E σ⁽ᵏ⁾(t + √v E)` is also
/// the Gaussian mean of the k-th derivative of the activation.
///
/// With zero variance ReLU returns the activation itself and the Gaussian
/// limit of its slope, `lim_{v→0} Φ(t/√v)`, which is `½` at the kink. Its
/// higher derivatives are zero away from the kink and refused at it. Once
/// `φ(u)` underflows, the derivatives of order ≥ 2, each `φ(u)` times a
/// polynomial in `u`, are returned as the zero it underflows to.
#[inline]
pub fn gaussian_smoothing_derivatives(
    activation: GaussianActivation,
    t: f64,
    variance: f64,
    derivatives: &mut [f64],
) -> Result<(), GaussianActivationError> {
    validate_finite(t)?;
    validate_variance(variance)?;
    match activation {
        GaussianActivation::Relu => relu_smoothing(t, variance, derivatives),
        GaussianActivation::ExactGelu => {
            exact_gelu_smoothing(t, variance, derivatives);
            Ok(())
        }
        GaussianActivation::Silu => Err(GaussianActivationError::NoClosedForm { activation }),
    }
}

/// Fills `coefficients[n]` with the normalized Hermite coefficient
/// `a_n = E[σ(t + s E) h_n(E)] = sⁿ ∂ⁿ_t T_{s²}σ(t)/√(n!)`, `h_n = He_n/√(n!)`, for
/// every `n < coefficients.len()`, with `s = scale`.
///
/// `σ(t + s E) = Σ_n a_n h_n(E)` in `L²(φ)`, so Mehler's formula gives
/// `E[σ(X) σ(Y)] = Σ_n ρⁿ a_n(b, s) a_n(c, s')` for `X = b + s E₁`, `Y = c + s' E₂`
/// with `corr(E₁, E₂) = ρ`. The scale is folded into the orthonormal recurrence
/// step by step, so the coefficients stay representable where the raw
/// derivatives `T⁽ⁿ⁾ ~ s^{1−n}` of [`gaussian_smoothing_derivatives`] overflow.
/// At `s = 0` the smoothed activation is the constant `σ(t)`: `a_0 = σ(t)` and
/// every other coefficient is zero, the kink included.
///
/// `bounds[n]` receives a first-order bound on the absolute rounding error of
/// `coefficients[n]` (see the module's rounding bounds); both slices need one
/// entry per order.
#[inline]
pub fn gaussian_hermite_coefficients(
    activation: GaussianActivation,
    t: f64,
    scale: f64,
    coefficients: &mut [f64],
    bounds: &mut [f64],
) -> Result<(), GaussianActivationError> {
    validate_finite(t)?;
    if !(scale.is_finite() && scale >= 0.0) {
        return Err(GaussianActivationError::InvalidScale { scale });
    }
    if coefficients.len() != bounds.len() {
        return Err(GaussianActivationError::MismatchedBoundsLength {
            coefficients: coefficients.len(),
            bounds: bounds.len(),
        });
    }
    match activation {
        GaussianActivation::Relu => relu_hermite_coefficients(t, scale, coefficients, bounds),
        GaussianActivation::ExactGelu => {
            exact_gelu_hermite_coefficients(t, scale, coefficients, bounds)
        }
        GaussianActivation::Silu => {
            return Err(GaussianActivationError::NoClosedForm { activation });
        }
    }
    Ok(())
}

/// The joint law of two Gaussian pre-activations: `X ~ N(mean_x, variance_x)`,
/// `Y ~ N(mean_y, variance_y)` and `Cov(X, Y) = covariance`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreactivationPair {
    pub mean_x: f64,
    pub mean_y: f64,
    pub variance_x: f64,
    pub variance_y: f64,
    pub covariance: f64,
    /// A derived bound on how far the rounding of the caller's Gram inputs can
    /// push `|covariance|` past `√(variance_x variance_y)`; `0` for an exact
    /// law. See [`project_covariance`].
    pub covariance_rounding: f64,
}

/// `K = E[σ(X) σ(Y)]` and its covariance derivative, each with a first-order
/// bound on its absolute rounding at the projected law.
///
/// The bounds come from running error analysis of the closed forms under the
/// module's standard model. For biased pairs they add the bivariate normal
/// owner's bounds on `Φ₂` and its partials, and propagate the rounding of the
/// standardized arguments `(h, k, ρ, 1 − ρ²)` through the partials of `Φ₂`. They
/// are absolute. `Φ₂` comes from the owner's plain entry, whose absolute contract
/// is cheap. Where that route's bounds certify no digit of `K` or `∂_r K` (the
/// anticorrelated lower tails, #2946), the kernel is evaluated again with the
/// owner's certified entry. Its rounding scales with `Φ₂` inside the owner's
/// relative regime, and [`PairKernel::orthant_fallback`] records that it ran.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairKernel {
    /// `K = E[σ(X) σ(Y)]`.
    pub value: f64,
    /// `∂K/∂r = E[σ'(X) σ'(Y)]` at fixed means and variances (Price's theorem).
    pub covariance_derivative: f64,
    /// A first-order bound on the absolute rounding of `value`.
    pub value_rounding: f64,
    /// A first-order bound on the absolute rounding of `covariance_derivative`.
    pub covariance_derivative_rounding: f64,
    /// `true` when the plain route certified no digit, so `Φ₂` was re-evaluated by the bivariate normal owner's
    /// certified entry, at a few hundred times the plain cost. Callers count it to measure that cost.
    pub orthant_fallback: bool,
}

impl PairKernel {
    /// Whether both bounds leave a certified digit, the invariant the plain route must meet before its values stand.
    fn certifies_a_digit(&self) -> bool {
        self.value_rounding < self.value.abs()
            && self.covariance_derivative_rounding < self.covariance_derivative.abs()
    }
}

/// The pair kernel `K_σ(b, c; v, w, r) = E[σ(X) σ(Y)]` with `∂_r K_σ`.
///
/// The covariance first passes [`project_covariance`] with the pair's
/// `covariance_rounding`: within that band it is evaluated on the
/// Cauchy-Schwarz boundary, and beyond it the law is refused.
///
/// A zero-variance pre-activation is constant. For ReLU at zero mean it sits
/// on the kink, where the slope is its Gaussian limit `½`, as in
/// [`gaussian_smoothing_derivatives`].
#[inline]
pub fn pair_kernel(
    activation: GaussianActivation,
    pair: PreactivationPair,
) -> Result<PairKernel, GaussianActivationError> {
    validate_finite(pair.mean_x)?;
    validate_finite(pair.mean_y)?;
    let law = project_covariance(
        pair.variance_x,
        pair.variance_y,
        pair.covariance,
        pair.covariance_rounding,
    )?;
    if pair.mean_x == 0.0 && pair.mean_y == 0.0 {
        return match activation {
            GaussianActivation::Relu => Ok(relu_zero_mean_pair_kernel(
                pair.variance_x,
                pair.variance_y,
                law,
            )),
            GaussianActivation::ExactGelu => Ok(exact_gelu_zero_mean_pair_kernel(
                pair.variance_x,
                pair.variance_y,
                law,
            )),
            GaussianActivation::Silu => Err(GaussianActivationError::NoClosedForm { activation }),
        };
    }
    match activation {
        GaussianActivation::Relu => relu_biased_pair_kernel(&pair, law),
        GaussianActivation::ExactGelu => exact_gelu_biased_pair_kernel(&pair, law),
        GaussianActivation::Silu => Err(GaussianActivationError::NoClosedForm { activation }),
    }
}

/// `∂K/∂v_x` and `∂K/∂v_y` of the pair kernel at fixed means and covariance, each with a first-order bound on its
/// absolute rounding in [`PairKernel`]'s convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairKernelVariancePartials {
    /// `∂K/∂v_x = ½·E[σ″(X)·σ(Y)]`.
    pub variance_x: f64,
    /// `∂K/∂v_y = ½·E[σ(X)·σ″(Y)]`.
    pub variance_y: f64,
    /// A first-order bound on the absolute rounding of `variance_x`.
    pub variance_x_rounding: f64,
    /// A first-order bound on the absolute rounding of `variance_y`.
    pub variance_y_rounding: f64,
}

/// The pair kernel's variance partials at fixed covariance, the input error a caller's computed law puts on `K`
/// beyond [`PairKernel::covariance_derivative`]'s (#2946, fr-subspace's V band).
///
/// The joint density satisfies `∂p/∂Σ₁₁ = ½·∂²p/∂x²`, and translation gives `∂²/∂x² = ∂²/∂b²`, so
/// `∂_v K = ½·E[σ″(X)·σ(Y)]` (and `∂_w K` symmetrically). Both closed forms read only one-dimensional smoothings.
/// - **ReLU**, `σ″ = δ`: `∂_v K = ½·f_X(0)·E[σ(Y) | X = 0]`, with `f_X(0) = φ(b/√v)/√v` and `Y | X = 0 ~ N(m, s²)`,
///   `m = c − r·b/v`, `s² = (vw − r²)/v` from the exactly carried residual. On a constant `X` (`v = 0`) the partial
///   is 0 for `b ≠ 0`, and the kink `b = 0` is refused as [`GaussianActivationError::UnsmoothedReluKink`].
/// - **Exact GELU**, `σ″(x) = (2 − x²)·φ(x)`: tilting by `φ` gives
///   `φ(x)·N(x; b, v) = N(b; 0, A)·N(x; b/A, v/A)` with `A = 1 + v`, and `Y | X` is unchanged. So under the tilt
///   `Y′ ~ N(m′, s′²)` with `m′ = c − r·b/A` and `s′² = (w + (vw − r²))/A`, and `Cov(X′, Y′) = r/A`. Stein's lemma
///   applied twice then gives
///   `∂_v K = ½·φ(b/√A)/√A·[(2 − b²/A² − v/A)·T − 2(b/A)(r/A)·T′ − (r/A)²·T″]`, where `T^{(k)}` are the exact
///   GELU's smoothing and its first two mean-derivatives at `(m′, s′²)`. By the same tilt,
///   `T″ = φ(m′/S)/S·(2 − m′²/S⁴ − s′²/S²)` with `S² = 1 + s′²`.
/// - **Rounding.** The bounded forms carry their own rounding. `m`'s rounding enters through the sups of the
///   activation's derivatives, since `|∂_m T^{(k)}| ≤ sup|σ^{(k+1)}|` (see [`exact_gelu_derivative_suprema`]).
pub fn pair_kernel_variance_partials(
    activation: GaussianActivation,
    pair: PreactivationPair,
) -> Result<PairKernelVariancePartials, GaussianActivationError> {
    validate_finite(pair.mean_x)?;
    validate_finite(pair.mean_y)?;
    let law = project_covariance(
        pair.variance_x,
        pair.variance_y,
        pair.covariance,
        pair.covariance_rounding,
    )?;
    let variance_x = variance_partial(
        activation,
        [pair.mean_x, pair.mean_y],
        [pair.variance_x, pair.variance_y],
        law,
    )?;
    let variance_y = variance_partial(
        activation,
        [pair.mean_y, pair.mean_x],
        [pair.variance_y, pair.variance_x],
        law,
    )?;
    Ok(PairKernelVariancePartials {
        variance_x: variance_x.value,
        variance_y: variance_y.value,
        variance_x_rounding: variance_x.bound,
        variance_y_rounding: variance_y.bound,
    })
}

/// `½·E[σ″(X)·σ(Y)]` for `X ~ N(means[0], variances[0])` and `Y ~ N(means[1], variances[1])` with the projected
/// covariance (see [`pair_kernel_variance_partials`]).
fn variance_partial(
    activation: GaussianActivation,
    means: [f64; 2],
    variances: [f64; 2],
    law: ProjectedCovariance,
) -> Result<Bounded, GaussianActivationError> {
    let [mean_x, mean_y] = means;
    let [variance_x, variance_y] = variances;
    let half = Bounded::exact(0.5);
    let location_x = Bounded::exact(mean_x);
    let location_y = Bounded::exact(mean_y);
    let covariance = Bounded::exact(law.covariance);
    match activation {
        GaussianActivation::Relu => {
            if variance_x == 0.0 {
                return if mean_x == 0.0 {
                    Err(GaussianActivationError::UnsmoothedReluKink { order: 2 })
                } else {
                    Ok(Bounded::exact(0.0))
                };
            }
            let spread_x = Bounded::exact(variance_x);
            let root_x = spread_x.sqrt();
            let density = bounded_normal_pdf(location_x.div(root_x)).div(root_x);
            let conditional_mean = location_y.sub(covariance.mul(location_x).div(spread_x));
            let conditional_variance = bounded_residual(law).div(spread_x);
            let conditional = if conditional_variance.value == 0.0 {
                Bounded {
                    value: conditional_mean.value.max(0.0),
                    bound: conditional_mean.bound + conditional_variance.bound.sqrt(),
                }
            } else {
                let unit = relu_smoothed_unit(conditional_mean.value, conditional_variance.sqrt());
                // |∂_m E[σ(m + sE)]| = Φ(m/s) ≤ 1 carries the conditional mean's rounding.
                Bounded {
                    value: unit.value.value,
                    bound: unit.value.bound + conditional_mean.bound,
                }
            };
            Ok(half.mul(density).mul(conditional))
        }
        GaussianActivation::ExactGelu => {
            let one = Bounded::exact(1.0);
            let spread_x = Bounded::exact(variance_x);
            let total = one.add(spread_x);
            let root_total = total.sqrt();
            let density = bounded_normal_pdf(location_x.div(root_total)).div(root_total);
            let tilted_mean = location_y.sub(covariance.mul(location_x).div(total));
            let tilted_variance = Bounded::exact(variance_y).add(bounded_residual(law)).div(total);
            let (unit, smoothed_total, smoothed_root) =
                exact_gelu_smoothed_unit(tilted_mean.value, tilted_variance);
            let reduced = tilted_mean.div(smoothed_total);
            let curvature = bounded_normal_pdf(tilted_mean.div(smoothed_root))
                .div(smoothed_root)
                .mul(
                    Bounded::exact(2.0)
                        .sub(reduced.mul(reduced))
                        .sub(tilted_variance.div(smoothed_total)),
                );
            let [slope_sup, curvature_sup, third_sup] = exact_gelu_derivative_suprema();
            let drift = tilted_mean.bound;
            let smoothed = Bounded {
                value: unit.value.value,
                bound: unit.value.bound + slope_sup * drift,
            };
            let slope = Bounded {
                value: unit.slope.value,
                bound: unit.slope.bound + curvature_sup * drift,
            };
            let curvature = Bounded {
                value: curvature.value,
                bound: curvature.bound + third_sup * drift,
            };
            let scaled_mean = location_x.div(total);
            let scaled_covariance = covariance.div(total);
            let bracket = Bounded::exact(2.0)
                .sub(scaled_mean.mul(scaled_mean))
                .sub(spread_x.div(total))
                .mul(smoothed)
                .sub(Bounded::exact(2.0).mul(scaled_mean).mul(scaled_covariance).mul(slope))
                .sub(scaled_covariance.mul(scaled_covariance).mul(curvature));
            Ok(half.mul(density).mul(bracket))
        }
        GaussianActivation::Silu => Err(GaussianActivationError::NoClosedForm { activation }),
    }
}

/// Upper bounds on `sup|σ′|`, `sup|σ″|` and `sup|σ‴|` for the exact GELU `σ(x) = x·Φ(x)`, each with its evaluation's
/// rounding added.
/// - `σ′ = Φ + xφ` peaks at `x = √2`, where `σ″ = (2 − x²)φ` vanishes.
/// - `|σ″|` peaks at the origin, `2φ(0)`: `0 ≤ (2 − x²)φ ≤ 2φ(0)` on `|x| ≤ √2`, and `|x² − 2|·φ(x)` is at most
///   `2φ(2) < 2φ(0)` beyond.
/// - `σ‴ = (x³ − 4x)φ` has its extrema where `x⁴ − 7x² + 4 = 0`, i.e. `x² = (7 ∓ √33)/2`. The inner root gives the
///   larger magnitude, `|σ‴(0.7925…)| = 0.780…`, against `0.0986…` at the outer.
fn exact_gelu_derivative_suprema() -> [f64; 3] {
    let slope_point = Bounded::exact(std::f64::consts::SQRT_2);
    let slope = bounded_normal_cdf(slope_point).add(slope_point.mul(bounded_normal_pdf(slope_point)));
    let curvature = Bounded::exact(2.0).mul(bounded_normal_pdf(Bounded::exact(0.0)));
    let third_point = Bounded::exact(7.0)
        .sub(Bounded::exact(33.0).sqrt())
        .mul(Bounded::exact(0.5))
        .sqrt();
    let cube = third_point.mul(third_point).mul(third_point);
    let third = cube
        .sub(Bounded::exact(4.0).mul(third_point))
        .mul(bounded_normal_pdf(third_point));
    [
        slope.value.abs() + slope.bound,
        curvature.value.abs() + curvature.bound,
        third.value.abs() + third.bound,
    ]
}

/// A pre-activation covariance on the Cauchy-Schwarz interval of its variances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedCovariance {
    /// `r`, moved onto `|r| = √(vw)` when rounding put it past that boundary.
    pub covariance: f64,
    /// `vw − r² ≥ 0` for that covariance, with both squares carried exactly.
    pub residual: f64,
}

/// The one Cauchy-Schwarz predicate for a Gaussian pre-activation pair.
///
/// A Gram computed in floating point leaves `r² ≤ vw` only by its rounding,
/// and a reader inside a retained subspace puts `r` on the boundary. The caller
/// therefore states `covariance_rounding = β`, a derived bound on how far the
/// rounding of its Gram inputs can push `|r|` past `√(vw)`; `0` states an exact
/// law. The residual `vw − r²` is formed from the two products and their exact
/// rounding residuals, so forming it rounds by at most `ε (vw + r²)`. A residual
/// below zero by no more than `β (2√(vw) + β) + ε (vw + r²)`, the squared form
/// of `|r| ≤ √(vw) + β`, puts the covariance on the boundary. A larger excess is
/// refused: past the stated rounding, an invalid covariance or a disagreeing
/// predicate is not rounding.
///
/// The projection moves `K` by at most `sup|σ'|² β`, because
/// `|∂_r K| = |E σ'(X) σ'(Y)| ≤ sup|σ'|²`
/// ([`GaussianActivation::slope_bound_squared`]).
#[inline]
pub fn project_covariance(
    variance_x: f64,
    variance_y: f64,
    covariance: f64,
    covariance_rounding: f64,
) -> Result<ProjectedCovariance, GaussianActivationError> {
    validate_variance(variance_x)?;
    validate_variance(variance_y)?;
    validate_finite(covariance)?;
    if !(covariance_rounding.is_finite() && covariance_rounding >= 0.0) {
        return Err(GaussianActivationError::InvalidCovarianceRounding {
            bound: covariance_rounding,
        });
    }
    let product = variance_x * variance_y;
    let square = covariance * covariance;
    if !product.is_finite() {
        return Err(GaussianActivationError::NonFiniteArgument { value: product });
    }
    if !square.is_finite() {
        return Err(GaussianActivationError::NonFiniteArgument { value: square });
    }
    let residual = (product - square)
        + (variance_x.mul_add(variance_y, -product) - covariance.mul_add(covariance, -square));
    if residual >= 0.0 {
        return Ok(ProjectedCovariance {
            covariance,
            residual,
        });
    }
    let scale = variance_x.sqrt() * variance_y.sqrt();
    let band = covariance_rounding * (2.0 * scale + covariance_rounding)
        + f64::EPSILON * (product + square);
    if -residual > band {
        return Err(GaussianActivationError::CovarianceOutsideCauchySchwarz {
            variance_x,
            variance_y,
            covariance,
            covariance_rounding,
        });
    }
    Ok(ProjectedCovariance {
        covariance: scale.copysign(covariance),
        residual: 0.0,
    })
}

fn validate_finite(value: f64) -> Result<(), GaussianActivationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GaussianActivationError::NonFiniteArgument { value })
    }
}

fn validate_variance(variance: f64) -> Result<(), GaussianActivationError> {
    if variance.is_finite() && variance >= 0.0 {
        Ok(())
    } else {
        Err(GaussianActivationError::InvalidVariance { variance })
    }
}

fn relu_smoothing(
    t: f64,
    variance: f64,
    derivatives: &mut [f64],
) -> Result<(), GaussianActivationError> {
    if variance == 0.0 {
        if t == 0.0 && derivatives.len() > 2 {
            return Err(GaussianActivationError::UnsmoothedReluKink { order: 2 });
        }
        for (order, slot) in derivatives.iter_mut().enumerate() {
            *slot = match order {
                0 => t.max(0.0),
                1 if t > 0.0 => 1.0,
                1 if t < 0.0 => 0.0,
                1 => 0.5,
                _ => 0.0,
            };
        }
        return Ok(());
    }
    let scale = variance.sqrt();
    let unit = relu_smoothed_unit(t, Bounded::exact(scale));
    if let Some(slot) = derivatives.get_mut(0) {
        *slot = unit.value.value;
    }
    if let Some(slot) = derivatives.get_mut(1) {
        *slot = unit.slope.value;
    }
    fill_relu_curvature(
        derivatives,
        unit.argument.value,
        unit.density.value,
        scale,
    );
    Ok(())
}

/// Cramér's inequality `|h_n(x)| e^{−x²/4} ≤ K` for the orthonormal Hermite
/// polynomials: Abramowitz and Stegun 22.14.17 give `K ≈ 1.086435`, rounded up.
const CRAMER_BOUND: f64 = 1.0865;

/// A computed value with a first-order bound on its absolute error: each
/// operation adds its inputs' propagated bounds to its own rounding `u |result|`.
#[derive(Clone, Copy, Debug)]
struct Bounded {
    value: f64,
    bound: f64,
}

impl Bounded {
    fn exact(value: f64) -> Self {
        Self { value, bound: 0.0 }
    }

    fn rounded(value: f64, propagated: f64) -> Self {
        Self {
            value,
            bound: propagated + UNIT_ROUNDOFF * value.abs(),
        }
    }

    fn negated(self) -> Self {
        Self {
            value: -self.value,
            bound: self.bound,
        }
    }

    fn add(self, other: Self) -> Self {
        Self::rounded(self.value + other.value, self.bound + other.bound)
    }

    fn sub(self, other: Self) -> Self {
        Self::rounded(self.value - other.value, self.bound + other.bound)
    }

    fn mul(self, other: Self) -> Self {
        Self::rounded(
            self.value * other.value,
            self.value.abs() * other.bound + other.value.abs() * self.bound,
        )
    }

    fn div(self, other: Self) -> Self {
        let value = self.value / other.value;
        Self::rounded(
            value,
            (self.bound + value.abs() * other.bound) / other.value.abs(),
        )
    }

    fn sqrt(self) -> Self {
        let value = self.value.sqrt();
        let propagated = if value > 0.0 {
            self.bound / (2.0 * value)
        } else {
            self.bound.sqrt()
        };
        Self::rounded(value, propagated)
    }
}

/// `φ(x)` from the probability owner's certified [`normal_pdf_bounded`], computed with
/// `libm::exp`. The argument's bound enters through `|φ'(x)| = |x| φ(x)`.
fn bounded_normal_pdf(argument: Bounded) -> Bounded {
    let (value, rounding) = normal_pdf_bounded(argument.value);
    Bounded {
        value,
        bound: rounding + value * argument.value.abs() * argument.bound,
    }
}

/// `Φ(x) = ½ erfc(−x/√2)`: the `erfc` ulp, and the two rounded operations forming
/// `−x/√2` (`2u` relative), which move `Φ` by at most `2u |x| φ(x)`. The `erfc` ulp is
/// a MEASUREMENT, not a contract: libm 0.2.16 states less than one ulp only for `erf`,
/// "by some experiment" (`src/math/erf.rs:43-44`). Until `Φ` routes through
/// [`normal_left_tail_ratios`], this bound rests on that measured link.
fn bounded_normal_cdf(argument: Bounded) -> Bounded {
    let value = normal_cdf(argument.value);
    let (density, density_rounding) = normal_pdf_bounded(argument.value);
    Bounded {
        value,
        bound: 2.0 * UNIT_ROUNDOFF * value
            + (density + density_rounding)
                * (2.0 * UNIT_ROUNDOFF * argument.value.abs() + argument.bound),
    }
}

/// `(1/λ(x), q(x)/λ(x))` at a bounded argument, from the log-CDF owner's
/// [`normal_left_tail_ratios`]. Its bounds are derived without libm and already carry the
/// argument's bound. `None` where the owner refuses, which is right of the origin within
/// the argument's bound; there the direct forms apply.
fn bounded_left_tail_ratios(argument: Bounded) -> Option<(Bounded, Bounded)> {
    normal_left_tail_ratios(argument.value, argument.bound)
        .ok()
        .map(|ratios| {
            (
                Bounded {
                    value: ratios.cdf_over_density,
                    bound: ratios.cdf_over_density_rounding,
                },
                Bounded {
                    value: ratios.positive_part_over_density,
                    bound: ratios.positive_part_over_density_rounding,
                },
            )
        })
}

/// What a Hermite coefficient loses when `φ(x)` underflows to zero: by Cramér's
/// inequality `φ(x) |h_j(x)| ≤ K (2π)^{−1/4} √φ(x)`, and the true `φ(x)` is then
/// below the smallest subnormal.
fn underflowed_hermite_function_bound() -> f64 {
    CRAMER_BOUND * TAU.sqrt().sqrt().recip() * f64::from_bits(1).sqrt()
}

/// The orthonormal Hermite recurrence `h_{k+1} = (x h_k − √k h_{k−1})/√(k+1)` at
/// a computed argument, with a running first-order bound on each value's
/// rounding.
///
/// A step rounds two products, the root `√k`, the subtraction, the root `√(k+1)`
/// and the division. That is at most `3u (|x h_k| + √k |h_{k−1}|)` on the numerator
/// and `2u |h_{k+1}|` on the quotient, on top of the propagated
/// `(|x| e_k + √k e_{k−1})/√(k+1)`.
#[derive(Clone, Copy, Debug)]
struct HermiteRecurrence {
    argument: f64,
    /// The order `k` of `current`.
    degree: usize,
    previous: f64,
    previous_error: f64,
    current: f64,
    current_error: f64,
}

impl HermiteRecurrence {
    fn start(argument: f64) -> Self {
        Self {
            argument,
            degree: 0,
            previous: 0.0,
            previous_error: 0.0,
            current: 1.0,
            current_error: 0.0,
        }
    }

    fn advance(&mut self) {
        let degree = self.degree as f64;
        let root = degree.sqrt();
        let next_root = (degree + 1.0).sqrt();
        let next = (self.argument * self.current - root * self.previous) / next_root;
        let numerator_magnitude =
            self.argument.abs() * self.current.abs() + root * self.previous.abs();
        let next_error = (self.argument.abs() * self.current_error
            + root * self.previous_error
            + 3.0 * UNIT_ROUNDOFF * numerator_magnitude)
            / next_root
            + 2.0 * UNIT_ROUNDOFF * next.abs();
        self.previous = self.current;
        self.previous_error = self.current_error;
        self.current = next;
        self.current_error = next_error;
        self.degree += 1;
    }
}

/// A smoothed unit at its standardized argument: `x`, `φ(x)`, `T` and `T'`, each
/// with its bound.
#[derive(Clone, Copy, Debug)]
struct SmoothedUnit {
    argument: Bounded,
    density: Bounded,
    value: Bounded,
    slope: Bounded,
}

fn write_bounded(values: &mut [f64], bounds: &mut [f64], order: usize, entry: Bounded) {
    if let (Some(value), Some(bound)) = (values.get_mut(order), bounds.get_mut(order)) {
        *value = entry.value;
        *bound = entry.bound;
    }
}

/// ReLU at scale `s > 0`: `x = t/s`, `T = t Φ(x) + s φ(x)` and `T' = Φ(x)`, and in the
/// left tail the forms `T = s φ q/λ` and `T' = φ/λ`.
fn relu_smoothed_unit(t: f64, spread: Bounded) -> SmoothedUnit {
    let location = Bounded::exact(t);
    let argument = location.div(spread);
    let density = bounded_normal_pdf(argument);
    let (value, slope) = if let Some((reciprocal, corrected)) = bounded_left_tail_ratios(argument) {
        (spread.mul(density).mul(corrected), density.mul(reciprocal))
    } else {
        let probability = bounded_normal_cdf(argument);
        (
            location.mul(probability).add(spread.mul(density)),
            probability,
        )
    };
    SmoothedUnit {
        argument,
        density,
        value,
        slope,
    }
}

/// `a_0 = T`, `a_1 = s T'` and `a_n = (−1)ⁿ s φ(x) h_{n−2}(x)/√(n(n−1))`, `x = t/s`,
/// each with its bound.
fn relu_hermite_coefficients(t: f64, scale: f64, coefficients: &mut [f64], bounds: &mut [f64]) {
    if scale == 0.0 {
        for (order, (slot, bound)) in coefficients.iter_mut().zip(bounds.iter_mut()).enumerate() {
            *slot = if order == 0 { t.max(0.0) } else { 0.0 };
            *bound = 0.0;
        }
        return;
    }
    let unit = relu_smoothed_unit(t, Bounded::exact(scale));
    write_bounded(coefficients, bounds, 0, unit.value);
    write_bounded(
        coefficients,
        bounds,
        1,
        Bounded::exact(scale).mul(unit.slope),
    );
    let density = unit.density.value;
    let pairs = coefficients.iter_mut().zip(bounds.iter_mut()).enumerate().skip(2);
    if density == 0.0 {
        let lost = scale * underflowed_hermite_function_bound();
        for (order, (slot, bound)) in pairs {
            *slot = 0.0;
            *bound = lost / ((order * (order - 1)) as f64).sqrt();
        }
        return;
    }
    let argument = unit.argument.value;
    let argument_bound = unit.argument.bound;
    let factor = scale * density;
    let mut recurrence = HermiteRecurrence::start(argument);
    for (order, (slot, bound)) in pairs {
        let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
        let normalizer = ((order * (order - 1)) as f64).sqrt();
        let value = sign * factor * recurrence.current / normalizer;
        *slot = value;
        // ∂_x (φ h_j) = φ (√j h_{j−1} − x h_j) with j = n − 2.
        let argument_term = density
            * ((recurrence.degree as f64).sqrt() * recurrence.previous.abs()
                + argument.abs() * recurrence.current.abs())
            * argument_bound;
        *bound = scale
            * (density * recurrence.current_error
                + recurrence.current.abs() * normal_pdf_bounded(argument).1
                + argument_term)
            / normalizer
            + 4.0 * UNIT_ROUNDOFF * value.abs();
        recurrence.advance();
    }
}

/// `derivatives[k] = (−1)ᵏ He_{k−2}(u) φ(u)/sᵏ⁻¹` for `k ≥ 2`.
fn fill_relu_curvature(derivatives: &mut [f64], u: f64, density: f64, scale: f64) {
    if density == 0.0 {
        derivatives.iter_mut().skip(2).for_each(|slot| *slot = 0.0);
        return;
    }
    let mut hermite_previous = 0.0;
    let mut hermite_current = 1.0;
    let mut factor = density / scale;
    for (order, slot) in derivatives.iter_mut().enumerate().skip(2) {
        *slot = factor * hermite_current;
        let degree = (order - 2) as f64;
        let hermite_next = u * hermite_current - degree * hermite_previous;
        hermite_previous = hermite_current;
        hermite_current = hermite_next;
        factor = -factor / scale;
    }
}

fn exact_gelu_smoothing(t: f64, variance: f64, derivatives: &mut [f64]) {
    let (unit, total, root_total) = exact_gelu_smoothed_unit(t, Bounded::exact(variance));
    if let Some(slot) = derivatives.get_mut(0) {
        *slot = unit.value.value;
    }
    if let Some(slot) = derivatives.get_mut(1) {
        *slot = unit.slope.value;
    }
    fill_exact_gelu_curvature(
        derivatives,
        unit.argument.value,
        unit.density.value,
        total.value,
        root_total.value,
    );
}

/// The exact GELU with `A = 1 + v`, `S = √A` and `x = t/S`:
/// `T = t Φ(x) + (v/S) φ(x)` and `T' = Φ(x) + x φ(x)/A`, and in the left tail the
/// forms `T = S φ (q/λ − 1/A)` and `T' = φ (1/λ + x/A)`. Returns the
/// unit together with `A` and `S`.
fn exact_gelu_smoothed_unit(t: f64, variance: Bounded) -> (SmoothedUnit, Bounded, Bounded) {
    let location = Bounded::exact(t);
    let total = Bounded::exact(1.0).add(variance);
    let root_total = total.sqrt();
    let argument = location.div(root_total);
    let density = bounded_normal_pdf(argument);
    let (value, slope) = if let Some((reciprocal, corrected)) = bounded_left_tail_ratios(argument) {
        (
            root_total
                .mul(density)
                .mul(corrected.sub(Bounded::exact(1.0).div(total))),
            density.mul(reciprocal.add(argument.div(total))),
        )
    } else {
        let probability = bounded_normal_cdf(argument);
        (
            location
                .mul(probability)
                .add(variance.div(root_total).mul(density)),
            probability.add(argument.mul(density).div(total)),
        )
    };
    (
        SmoothedUnit {
            argument,
            density,
            value,
            slope,
        },
        total,
        root_total,
    )
}

/// `a_0 = T`, `a_1 = s T'` and
/// `a_n = (−1)ⁿ φ(x) (s/S)ⁿ [A h_{n−2}(x)/√(n(n−1)) − h_n(x)]/S`, `x = t/S`, each with
/// its bound.
fn exact_gelu_hermite_coefficients(
    t: f64,
    scale: f64,
    coefficients: &mut [f64],
    bounds: &mut [f64],
) {
    let spread = Bounded::exact(scale);
    let (unit, total, root_total) = exact_gelu_smoothed_unit(t, spread.mul(spread));
    write_bounded(coefficients, bounds, 0, unit.value);
    write_bounded(coefficients, bounds, 1, spread.mul(unit.slope));
    if scale == 0.0 {
        for (slot, bound) in coefficients.iter_mut().zip(bounds.iter_mut()).skip(2) {
            *slot = 0.0;
            *bound = 0.0;
        }
        return;
    }
    let density = unit.density.value;
    let ratio = spread.div(root_total);
    let pairs = coefficients.iter_mut().zip(bounds.iter_mut()).enumerate().skip(2);
    if density == 0.0 {
        let lost = underflowed_hermite_function_bound() / root_total.value;
        let mut power = ratio.value * ratio.value;
        for (order, (slot, bound)) in pairs {
            *slot = 0.0;
            *bound = power
                * lost
                * (total.value / ((order * (order - 1)) as f64).sqrt() + 1.0);
            power *= ratio.value;
        }
        return;
    }
    let argument = unit.argument.value;
    let argument_bound = unit.argument.bound;
    let mut factor = unit.density.mul(ratio).mul(ratio).div(root_total);
    let mut recurrence = HermiteRecurrence::start(argument);
    recurrence.advance();
    recurrence.advance();
    // `(h, e)` of orders n − 3 and n − 2; the recurrence holds n − 1 and n.
    let mut lowest = (0.0_f64, 0.0_f64);
    let mut lower = (1.0_f64, 0.0_f64);
    for (order, (slot, bound)) in pairs {
        let degree = order as f64;
        let normalizer = (degree * (degree - 1.0)).sqrt();
        let bracket = total.value * lower.0 / normalizer - recurrence.current;
        let value = factor.value * bracket;
        *slot = value;
        let bracket_error = (total.value * lower.1
            + lower.0.abs() * total.bound
            + 3.0 * UNIT_ROUNDOFF * total.value * lower.0.abs())
            / normalizer
            + recurrence.current_error
            + UNIT_ROUNDOFF * bracket.abs();
        // ∂_x [A h_{n−2}/√(n(n−1)) − h_n] = A √(n−2) h_{n−3}/√(n(n−1)) − √n h_{n−1}.
        let argument_term = (total.value * (degree - 2.0).sqrt() * lowest.0.abs() / normalizer
            + degree.sqrt() * recurrence.previous.abs())
            * argument_bound;
        *bound = factor.value.abs() * (bracket_error + argument_term)
            + bracket.abs() * factor.bound
            + UNIT_ROUNDOFF * value.abs();
        lowest = lower;
        lower = (recurrence.previous, recurrence.previous_error);
        recurrence.advance();
        factor = factor.negated().mul(ratio);
    }
}

/// `derivatives[k] = (−1)ᵏ φ(u) (A He_{k−2}(u) − He_k(u))/(A Sᵏ⁻¹)` for `k ≥ 2`.
fn fill_exact_gelu_curvature(
    derivatives: &mut [f64],
    u: f64,
    density: f64,
    total: f64,
    root_total: f64,
) {
    if density == 0.0 {
        derivatives.iter_mut().skip(2).for_each(|slot| *slot = 0.0);
        return;
    }
    let mut hermite_lower = 1.0;
    let mut hermite_middle = u;
    let mut hermite_upper = u.mul_add(u, -1.0);
    let mut factor = density / (total * root_total);
    for (order, slot) in derivatives.iter_mut().enumerate().skip(2) {
        *slot = factor * (total * hermite_lower - hermite_upper);
        let hermite_next = u * hermite_upper - order as f64 * hermite_middle;
        hermite_lower = hermite_middle;
        hermite_middle = hermite_upper;
        hermite_upper = hermite_next;
        factor = -factor / root_total;
    }
}

/// `K = (√(vw − r²) + r atan2(√(vw − r²), −r))/(2π)` and
/// `∂_r K = atan2(√(vw − r²), −r)/(2π)`: the arc-cosine kernel with
/// `√(vw) sin θ = √(vw − r²)` and `θ = π − atan2(√(vw − r²), −r)`.
fn relu_zero_mean_pair_kernel(
    variance_x: f64,
    variance_y: f64,
    law: ProjectedCovariance,
) -> PairKernel {
    if variance_x == 0.0 || variance_y == 0.0 {
        return pair_kernel_from(Bounded::exact(0.0), Bounded::exact(0.25));
    }
    let root_residual = bounded_residual(law).sqrt();
    let orthant = bounded_orthant(root_residual, law.covariance);
    let value = root_residual
        .div(bounded_tau())
        .add(Bounded::exact(law.covariance).mul(orthant));
    pair_kernel_from(value, orthant)
}

fn pair_kernel_from(value: Bounded, derivative: Bounded) -> PairKernel {
    PairKernel {
        value: value.value,
        covariance_derivative: derivative.value,
        value_rounding: value.bound,
        covariance_derivative_rounding: derivative.bound,
        orthant_fallback: false,
    }
}

/// `vw − r²` as the projection formed it: from two products and their exact
/// rounding residuals, so it rounds by at most `3u` of itself.
fn bounded_residual(law: ProjectedCovariance) -> Bounded {
    Bounded {
        value: law.residual,
        bound: 3.0 * UNIT_ROUNDOFF * law.residual,
    }
}

/// `2π` rounded to a double, within `u` of itself.
fn bounded_tau() -> Bounded {
    Bounded {
        value: TAU,
        bound: UNIT_ROUNDOFF * TAU,
    }
}

/// `atan2(y, −r)/(2π)` for `y ≥ 0`: the derived bound of [`bounded_arctangent2`], the
/// height's bound through `|∂_y atan2(y, x)| = |x|/(x² + y²)`, and the rounding of `2π` and
/// the division.
fn bounded_orthant(height: Bounded, covariance: f64) -> Bounded {
    let angle = bounded_arctangent2(height.value, -covariance);
    Bounded {
        value: angle.value,
        bound: angle.bound
            + covariance.abs() * height.bound
                / (covariance * covariance + height.value * height.value),
    }
    .div(bounded_tau())
}

/// Centers per unit of the arctangent's argument reduction: `atan(z)` for `z ∈ [0, 1]` is
/// taken about the nearest `c = k/8`, so the reduced argument stays within `1/16`.
const ARCTANGENT_CENTERS_PER_UNIT: f64 = 8.0;

/// The centers `k/8`, `k = 0, …, 8`.
const ARCTANGENT_CENTER_COUNT: usize = 9;

/// Odd Taylor terms `(−1)ʲ w^{2j+1}/(2j + 1)`, `j < J`, kept for `atan(w)` at `|w| ≤ 1/16`.
/// The Leibniz remainder `|w|^{2J+1}/(2J + 1)` is then below `10⁻²u·|w|`.
const ARCTANGENT_SERIES_TERMS: usize = 7;

/// The certified pieces of [`bounded_arctangent2`].
struct ArctangentTable {
    /// `atan(k/8)` rounded to `f64`.
    centers: [f64; ARCTANGENT_CENTER_COUNT],
    /// Bounds on their rounding.
    center_bounds: [f64; ARCTANGENT_CENTER_COUNT],
    /// `(−1)ʲ/(2j + 1)` rounded to `f64`.
    series: [f64; ARCTANGENT_SERIES_TERMS],
    /// `Σ_j β_j·ρ^{2j}` over the coefficients' rounding `β_j ≤ u/(2j + 1)`, at the reduced reach `ρ`.
    series_budget: f64,
    /// `ρ^{2J}/(2J + 1)`: the Leibniz remainder per unit of `|w|`.
    series_remainder: f64,
}

static ARCTANGENT_TABLE: LazyLock<ArctangentTable> = LazyLock::new(ArctangentTable::build);

impl ArctangentTable {
    /// `atan(k/8)` from Euler's series `atan(x) = Σ_n (2²ⁿ(n!)²/(2n + 1)!)·x^{2n+1}/(1 + x²)^{n+1}`
    /// in bounded double-double.
    ///
    /// Its terms are positive, and each is the last times `y·2n/(2n + 1) < y` with
    /// `y = x²/(1 + x²) ≤ ½`. So the rest after any term is below that term times `y/(1 − y)`. At
    /// `x = k/8`, the first term `8k/(64 + k²)` and `y = k²/(64 + k²)` are exact quotients of
    /// integers.
    ///
    /// The reduced reach `ρ` is `1/16` widened by the reduced argument's rounding: its
    /// numerator is exact, and its denominator and quotient round once each.
    fn build() -> Self {
        let floor = UNIT_ROUNDOFF * UNIT_ROUNDOFF;
        let upper = |value: BoundedDoubleDouble| {
            value.value.high.abs() + value.value.low.abs() + value.rounding
        };
        let mut centers = [0.0; ARCTANGENT_CENTER_COUNT];
        let mut center_bounds = [0.0; ARCTANGENT_CENTER_COUNT];
        for (index, (center, bound)) in centers.iter_mut().zip(center_bounds.iter_mut()).enumerate() {
            let numerator = index as f64;
            let denominator = 64.0 + numerator * numerator;
            let ratio = BoundedDoubleDouble::exact(numerator * numerator).div_f64(denominator);
            let ratio_upper = upper(ratio);
            let mut term = BoundedDoubleDouble::exact(8.0 * numerator).div_f64(denominator);
            let mut sum = term;
            let mut order = 1.0;
            loop {
                // `m = 4`: the upper value, the product, the difference and the quotient.
                let tail = inflated(upper(term) * ratio_upper / (1.0 - ratio_upper), 4);
                if tail <= floor * sum.value.high {
                    sum.rounding = inflated(sum.rounding + tail, 1);
                    break;
                }
                term = term.mul(ratio).mul_f64(2.0 * order).div_f64(2.0 * order + 1.0);
                sum = sum.add(term);
                order += 1.0;
            }
            (*center, *bound) = sum.to_f64();
        }
        let reach = inflated(0.5 / ARCTANGENT_CENTERS_PER_UNIT, 2);
        let square = reach * reach;
        let mut series = [0.0; ARCTANGENT_SERIES_TERMS];
        let mut series_budget = 0.0;
        let mut power = 1.0;
        for (order, coefficient) in series.iter_mut().enumerate() {
            let odd = (2 * order + 1) as f64;
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            *coefficient = sign / odd;
            series_budget += UNIT_ROUNDOFF / odd * power;
            power *= square;
        }
        // `power` is now `ρ^{2J}`; `m = 3J + 2` for the budget and `m = 2J + 1` for the remainder.
        Self {
            centers,
            center_bounds,
            series,
            series_budget: inflated(series_budget, 3 * ARCTANGENT_SERIES_TERMS + 2),
            series_remainder: inflated(
                power / (2 * ARCTANGENT_SERIES_TERMS + 1) as f64,
                2 * ARCTANGENT_SERIES_TERMS + 1,
            ),
        }
    }
}

/// `atan2(y, x)` for `y ≥ 0`, in `[0, π]`, with a bound derived from IEEE-754 basic
/// operations only: no libm call and no measured ulp.
/// - **Quadrant.** `y = 0` gives `0` or `π` by the sign of `x`, as `f64::atan2` does. Otherwise
///   `θ = atan(z)` with `z = y/|x| ≤ 1`, or `π/2 − atan(z)` with `z = |x|/y < 1`, and then `π − θ`
///   when `x < 0`. `atan(z) ≤ π/4` and `π/2 − atan(z) ≥ π/4`, so neither subtraction cancels.
/// - **Reduction.** `atan(z) = atan(c) + atan(w)` with `c = k/8` nearest `z` and
///   `w = (z − c)/(1 + zc)`, `|w| ≤ 1/16`. `8z` and its rounding are exact, and `z − c` is exact by
///   Sterbenz's lemma (`c/2 ≤ z ≤ 2c` once `k ≥ 1`, and `z − c = z` at `k = 0`).
/// - **Series.** `atan(w) = w·Σ_j (−1)ʲ s^j/(2j + 1)` with `s = w²`, by Horner's rule in plain
///   products and sums with a running bound (Higham §5.1), since generic x86-64 builds lower
///   `mul_add` to a library call. The terms alternate and decrease, so the
///   remainder is at most the first omitted one.
/// - **Rounding charged.**
///   - `z` rounds by `u·z`, and `|∂_z atan| ≤ 1` passes it on.
///   - `w` rounds by `3u·|w|`: the product and sum of its denominator, and its quotient.
///   - `s` rounds by `u·s`, which moves the Horner sum by at most `u·s/3`, since its derivative
///     in `s` is below `⅓`.
///   - The product `w·p` and the three sums round by `u` of their results.
///   - `π` and `π/2` enter within `|π − fl(π)| < 1.23e-16` and half of it: `BoundedDoubleDouble::PI`'s
///     low word plus its bound.
///   - The bound's own evaluation is absorbed by `inflated`.
fn bounded_arctangent2(height: f64, abscissa: f64) -> Bounded {
    let pi_rounding = BoundedDoubleDouble::PI.value.low.abs() + BoundedDoubleDouble::PI.rounding;
    if height == 0.0 {
        return if abscissa.is_sign_negative() {
            Bounded { value: PI, bound: pi_rounding }
        } else {
            Bounded::exact(0.0)
        };
    }
    let table = &*ARCTANGENT_TABLE;
    let magnitude = abscissa.abs();
    let reflected = height > magnitude;
    let argument = if reflected { magnitude / height } else { height / magnitude };
    let scaled = (argument * ARCTANGENT_CENTERS_PER_UNIT).round();
    let index = scaled as usize;
    let center = scaled / ARCTANGENT_CENTERS_PER_UNIT;
    let reduced = (argument - center) / (argument * center + 1.0);
    let square = reduced * reduced;
    let mut coefficients = table.series.iter().rev();
    let mut sum = coefficients.next().copied().unwrap_or(0.0);
    let mut running = 0.0_f64;
    for &coefficient in coefficients {
        let product = sum * square;
        sum = product + coefficient;
        running = running * square + UNIT_ROUNDOFF * (product.abs() + sum.abs());
    }
    let series = reduced * sum;
    let base = table.centers[index] + series;
    // `atan(z)`'s error: the center's rounding; `|w|` times the Horner running bound, the
    // coefficients' budget, `s`'s rounding, the remainder and `w`'s own `3u`; the product and the
    // sum; `z`'s rounding; and `η/2` apiece where `z`, `s` or a product underflows. `m = 12` for
    // the expression, and `3J` for the running bound's own steps and their `1/(1 − u)`.
    let base_bound = inflated(
        table.center_bounds[index]
            + reduced.abs()
                * (running
                    + table.series_budget
                    + UNIT_ROUNDOFF * square / 3.0
                    + table.series_remainder
                    + 3.0 * UNIT_ROUNDOFF)
            + UNIT_ROUNDOFF * (series.abs() + base.abs())
            + UNIT_ROUNDOFF * argument
            + 2.0 * f64::from_bits(1),
        12 + 3 * ARCTANGENT_SERIES_TERMS,
    );
    let (angle, angle_bound) = if reflected {
        let angle = FRAC_PI_2 - base;
        (angle, base_bound + 0.5 * pi_rounding + UNIT_ROUNDOFF * angle)
    } else {
        (base, base_bound)
    };
    if abscissa < 0.0 {
        let supplement = PI - angle;
        Bounded {
            value: supplement,
            bound: inflated(angle_bound + pi_rounding + UNIT_ROUNDOFF * supplement, 2),
        }
    } else {
        Bounded {
            value: angle,
            bound: inflated(angle_bound, 2),
        }
    }
}

/// `K = r H + (vw + r² q)/(2π √Δ)` and `∂_r K = H + r (1/A + 1/B + 1/Δ)/(2π √Δ)`
/// with `H = atan2(√Δ, −r)/(2π)` and `Δ = 1 + v + w + (vw − r²)`.
/// `vw + r² q = (vw − r²) + r² (1/A + 1/B)` sums two nonnegative terms.
fn exact_gelu_zero_mean_pair_kernel(
    variance_x: f64,
    variance_y: f64,
    law: ProjectedCovariance,
) -> PairKernel {
    let one = Bounded::exact(1.0);
    let total_x = one.add(Bounded::exact(variance_x));
    let total_y = one.add(Bounded::exact(variance_y));
    let covariance = Bounded::exact(law.covariance);
    let residual = bounded_residual(law);
    let discriminant = one
        .add(Bounded::exact(variance_x))
        .add(Bounded::exact(variance_y))
        .add(residual);
    let root_discriminant = discriminant.sqrt();
    let orthant = bounded_orthant(root_discriminant, law.covariance);
    let density = one.div(bounded_tau().mul(root_discriminant));
    let inverse_totals = one.div(total_x).add(one.div(total_y));
    let coupling = residual.add(covariance.mul(covariance).mul(inverse_totals));
    let value = covariance.mul(orthant).add(coupling.mul(density));
    let derivative = orthant.add(
        covariance
            .mul(density)
            .mul(inverse_totals.add(one.div(discriminant))),
    );
    pair_kernel_from(value, derivative)
}

fn bivariate_normal_refusal(source: BivariateNormalError) -> GaussianActivationError {
    GaussianActivationError::BivariateNormal { source }
}

/// `1[x > 0]`, and `½` at `x = 0`: the Gaussian limit of `Φ(x/√v)` as `v → 0`.
fn limiting_step(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        0.0
    } else {
        0.5
    }
}

/// Which of the bivariate normal owner's entries gives `Φ₂`'s value and rounding.
#[derive(Clone, Copy)]
enum OrthantEntry {
    /// `bivariate_normal_cdf_with_complement` under its absolute contract.
    Plain,
    /// `bivariate_normal_cdf_with_complement_bounded`: value-scaled rounding inside its relative regime, the
    /// absolute contract elsewhere.
    Certified,
}

/// `H`, `∂_hΦ₂`, `∂_kΦ₂` and `φ₂` at a standardized law, each with its bound.
struct StandardizedOrthant {
    orthant: Bounded,
    partial_h: Bounded,
    partial_k: Bounded,
    density: Bounded,
}

fn clamped_correlation(ratio: Bounded) -> Bounded {
    Bounded {
        value: ratio.value.clamp(-1.0, 1.0),
        bound: ratio.bound,
    }
}

fn capped_complement(ratio: Bounded) -> Bounded {
    Bounded {
        value: ratio.value.min(1.0),
        bound: ratio.bound,
    }
}

/// `Φ₂(h, k; ρ)` and its partials from the bivariate normal owner, with its own
/// rounding bounds plus the standardized arguments' rounding propagated through
/// the partials of `Φ₂`. `Φ₂`'s own rounding is the chosen entry's: the plain
/// entry's absolute contract, or the certified entry's per-evaluation bound, which
/// scales with the value inside its relative regime (ρ ≤ 0 with both apex
/// coordinates resolved nonnegative).
/// - A rounded `1 − ρ²` moves the owner's `1 ∓ |ρ|`, and so the effective
///   correlation, by at most its bound over `1 + |ρ|`.
/// - With `c = 1 − ρ²`, `r_h = (h − ρk)/c` and `r_k = (k − ρh)/c`, the partials
///   are: `∂_h ∂_hΦ₂ = −h ∂_hΦ₂ − ρφ₂`, `∂_k ∂_hΦ₂ = φ₂`, `∂_ρ ∂_hΦ₂ = −φ₂ r_h`,
///   `∂_h φ₂ = −φ₂ r_h`, `∂_k φ₂ = −φ₂ r_k` and `∂_ρ φ₂ = φ₂ (ρ/c + r_h r_k)`
///   (symmetrically in `k`).
/// - On the degenerate law `c = 0` the partials take their limits `φ(h)·step`,
///   `φ₂` vanishes, and `H` moves by at most `φ(h) δh + φ(k) δk`.
fn standardized_orthant(
    h: Bounded,
    k: Bounded,
    correlation: Bounded,
    complement: Bounded,
    entry: OrthantEntry,
) -> Result<StandardizedOrthant, GaussianActivationError> {
    let (value, rounding) = match entry {
        OrthantEntry::Plain => (
            bivariate_normal_cdf_with_complement(h.value, k.value, correlation.value, complement.value)
                .map_err(bivariate_normal_refusal)?,
            BIVARIATE_NORMAL_CDF_ERROR_BOUND,
        ),
        OrthantEntry::Certified => {
            let bounded = bivariate_normal_cdf_with_complement_bounded(
                h.value,
                k.value,
                correlation.value,
                complement.value,
            )
            .map_err(bivariate_normal_refusal)?;
            (bounded.value, bounded.rounding)
        }
    };
    let rho = correlation.value;
    let correlation_bound = correlation.bound + complement.bound / (1.0 + rho.abs());
    if complement.value > 0.0 {
        let partials =
            bivariate_normal_cdf_partials_with_complement(h.value, k.value, rho, complement.value)
                .map_err(bivariate_normal_refusal)?;
        let density = partials.d_rho;
        let reduced_h = (h.value - rho * k.value) / complement.value;
        let reduced_k = (k.value - rho * h.value) / complement.value;
        Ok(StandardizedOrthant {
            orthant: Bounded {
                value,
                bound: rounding
                    + partials.d_h * h.bound
                    + partials.d_k * k.bound
                    + density * correlation_bound,
            },
            partial_h: Bounded {
                value: partials.d_h,
                bound: partials.d_h_rounding
                    + (h.value * partials.d_h + rho * density).abs() * h.bound
                    + density * k.bound
                    + (density * reduced_h).abs() * correlation_bound,
            },
            partial_k: Bounded {
                value: partials.d_k,
                bound: partials.d_k_rounding
                    + density * h.bound
                    + (k.value * partials.d_k + rho * density).abs() * k.bound
                    + (density * reduced_k).abs() * correlation_bound,
            },
            density: Bounded {
                value: density,
                bound: partials.d_rho_rounding
                    + (density * reduced_h).abs() * h.bound
                    + (density * reduced_k).abs() * k.bound
                    + (density * (rho / complement.value + reduced_h * reduced_k)).abs()
                        * correlation_bound,
            },
        })
    } else {
        let density_h = bounded_normal_pdf(h);
        let density_k = bounded_normal_pdf(k);
        let (step_h, step_k) = if rho > 0.0 {
            (limiting_step(k.value - h.value), limiting_step(h.value - k.value))
        } else {
            let sum = limiting_step(h.value + k.value);
            (sum, sum)
        };
        Ok(StandardizedOrthant {
            orthant: Bounded {
                value,
                bound: rounding
                    + density_h.value * h.bound
                    + density_k.value * k.bound,
            },
            partial_h: density_h.mul(Bounded::exact(step_h)),
            partial_k: density_k.mul(Bounded::exact(step_k)),
            density: Bounded::exact(0.0),
        })
    }
}

/// ReLU with means: `K = (bc + r) H + c √v ∂_hΦ₂ + b √w ∂_kΦ₂ + √(vw) (1 − ρ²) φ₂` and
/// `∂_r K = H`, with `H = Φ₂(h, k; ρ)`, `h = b/√v`, `k = c/√w` and `ρ = r/√(vw)`.
fn relu_biased_pair_kernel(
    pair: &PreactivationPair,
    law: ProjectedCovariance,
) -> Result<PairKernel, GaussianActivationError> {
    let (mean_x, mean_y) = (pair.mean_x, pair.mean_y);
    let (variance_x, variance_y) = (pair.variance_x, pair.variance_y);
    if variance_x == 0.0 || variance_y == 0.0 {
        // A constant pre-activation: K = σ(b) E σ(Y) and ∂_r K = σ'(b) E σ'(Y).
        let (constant, smoothed, variance) = if variance_x == 0.0 {
            (mean_x, mean_y, variance_y)
        } else {
            (mean_y, mean_x, variance_x)
        };
        let (moment, slope) = if variance == 0.0 {
            (
                Bounded::exact(smoothed.max(0.0)),
                Bounded::exact(limiting_step(smoothed)),
            )
        } else {
            let unit = relu_smoothed_unit(smoothed, Bounded::exact(variance).sqrt());
            (unit.value, unit.slope)
        };
        return Ok(pair_kernel_from(
            Bounded::exact(constant.max(0.0)).mul(moment),
            Bounded::exact(limiting_step(constant)).mul(slope),
        ));
    }
    let root_x = Bounded::exact(variance_x).sqrt();
    let root_y = Bounded::exact(variance_y).sqrt();
    let scale = root_x.mul(root_y);
    let location_x = Bounded::exact(mean_x);
    let location_y = Bounded::exact(mean_y);
    let covariance = Bounded::exact(law.covariance);
    let complement = capped_complement(
        bounded_residual(law).div(Bounded::exact(variance_x).mul(Bounded::exact(variance_y))),
    );
    let evaluate = |entry: OrthantEntry| -> Result<PairKernel, GaussianActivationError> {
        let standardized = standardized_orthant(
            location_x.div(root_x),
            location_y.div(root_y),
            clamped_correlation(covariance.div(scale)),
            complement,
            entry,
        )?;
        let value = location_x
            .mul(location_y)
            .add(covariance)
            .mul(standardized.orthant)
            .add(location_y.mul(root_x).mul(standardized.partial_h))
            .add(location_x.mul(root_y).mul(standardized.partial_k))
            .add(scale.mul(complement).mul(standardized.density));
        Ok(pair_kernel_from(value, standardized.orthant))
    };
    with_orthant_fallback(evaluate)
}

/// The plain route, and the certified one only where the plain bounds certify no digit (see [`PairKernel`]). The
/// trigger is the kernel's own invariant, so the certified entry's cost falls only on the pairs that need it.
fn with_orthant_fallback(
    evaluate: impl Fn(OrthantEntry) -> Result<PairKernel, GaussianActivationError>,
) -> Result<PairKernel, GaussianActivationError> {
    let plain = evaluate(OrthantEntry::Plain)?;
    if plain.certifies_a_digit() {
        return Ok(plain);
    }
    let certified = evaluate(OrthantEntry::Certified)?;
    Ok(PairKernel {
        orthant_fallback: true,
        ..certified
    })
}

/// The exact GELU with means (the module's biased forms), with `A = 1 + v`, `B = 1 + w`,
/// `Δ = 1 + v + w + (vw − r²)`, `h = b/√A`, `k = c/√B`, `ρ = r/√(AB)` and `1 − ρ² = Δ/(AB)`.
fn exact_gelu_biased_pair_kernel(
    pair: &PreactivationPair,
    law: ProjectedCovariance,
) -> Result<PairKernel, GaussianActivationError> {
    let (mean_x, mean_y) = (pair.mean_x, pair.mean_y);
    let one = Bounded::exact(1.0);
    let spread_x = Bounded::exact(pair.variance_x);
    let spread_y = Bounded::exact(pair.variance_y);
    let total_x = one.add(spread_x);
    let total_y = one.add(spread_y);
    let root_total_x = total_x.sqrt();
    let root_total_y = total_y.sqrt();
    let root_product = root_total_x.mul(root_total_y);
    let covariance = Bounded::exact(law.covariance);
    let residual = bounded_residual(law);
    let discriminant = one.add(spread_x).add(spread_y).add(residual);
    let location_x = Bounded::exact(mean_x);
    let location_y = Bounded::exact(mean_y);
    let evaluate = |entry: OrthantEntry| -> Result<PairKernel, GaussianActivationError> {
        let standardized = standardized_orthant(
            location_x.div(root_total_x),
            location_y.div(root_total_y),
            clamped_correlation(covariance.div(root_product)),
            capped_complement(discriminant.div(total_x.mul(total_y))),
            entry,
        )?;
        let partial_x = standardized.partial_h.div(root_total_x);
        let partial_y = standardized.partial_k.div(root_total_y);
        let density = standardized.density.div(root_product);
        let reduced_x = total_y
            .mul(location_x)
            .sub(covariance.mul(location_y))
            .div(discriminant);
        let reduced_y = total_x
            .mul(location_y)
            .sub(covariance.mul(location_x))
            .div(discriminant);
        let coupling =
            residual.add(covariance.mul(covariance).mul(one.div(total_x).add(one.div(total_y))));
        let value = location_x
            .mul(location_y)
            .add(covariance)
            .mul(standardized.orthant)
            .add(
                location_y
                    .mul(spread_x)
                    .add(location_x.mul(covariance).div(total_x))
                    .mul(partial_x),
            )
            .add(
                location_x
                    .mul(spread_y)
                    .add(location_y.mul(covariance).div(total_y))
                    .mul(partial_y),
            )
            .add(coupling.mul(density));
        let derivative = standardized
            .orthant
            .add(location_x.mul(partial_x).add(covariance.mul(density)).div(total_x))
            .add(location_y.mul(partial_y).add(covariance.mul(density)).div(total_y))
            .add(
                covariance
                    .div(discriminant)
                    .add(reduced_x.mul(reduced_y))
                    .mul(density),
            );
        Ok(pair_kernel_from(value, derivative))
    };
    with_orthant_fallback(evaluate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bivariate_normal::RoundingContract;
    use crate::probability::normal_pdf;
    use crate::quadrature::{
        GaussHermiteRule, gauss_hermite_rule, symmetric_tridiagonal_eigen_first_components,
    };
    use std::f64::consts::{PI, SQRT_2};

    /// A bound on the floating-point operations one closed form here, or one
    /// integrand evaluation, chains: none takes more than 64 steps, and each
    /// adds at most one ulp of relative error to first order.
    const EVALUATION_OPERATIONS: f64 = 64.0;

    /// `sup|σ'|` of the exact GELU, `Φ(√2) + √2 φ(√2) = 1.12893…`, rounded up.
    const EXACT_GELU_SLOPE_BOUND: f64 = 1.129;

    /// A quadrature estimate and a derived bound on its error.
    #[derive(Clone, Copy, Debug)]
    struct Reference {
        value: f64,
        bound: f64,
    }

    /// One quadrature pass: the sum and the sum of its summands' magnitudes.
    #[derive(Clone, Copy, Debug)]
    struct Pass {
        sum: f64,
        absolute: f64,
    }

    /// A rule checked against its doubled order. `|Q_n − Q_2n|` bounds
    /// `|I − Q_2n|` once the rule converges geometrically, which the analytic
    /// integrands here guarantee. Recursive summation of `summands` terms errs
    /// by at most `(summands − 1) ε Σ|term|` (Higham, ASNA §4.2), Golub-Welsch
    /// weights carry `O(nodes ε)` relative error, and every summand's
    /// evaluation adds at most `EVALUATION_OPERATIONS ε` relative.
    fn doubled_order(
        coarse: Pass,
        fine: Pass,
        summands: usize,
        nodes: usize,
        truncation: f64,
    ) -> Reference {
        let rounding = (summands as f64 + nodes as f64 + EVALUATION_OPERATIONS)
            * f64::EPSILON
            * fine.absolute;
        Reference {
            value: fine.sum,
            bound: (coarse.sum - fine.sum).abs() + rounding + truncation,
        }
    }

    /// First-order rounding of a closed form whose terms add up to `magnitude`.
    fn closed_form_rounding(magnitude: f64) -> f64 {
        EVALUATION_OPERATIONS * f64::EPSILON * magnitude
    }

    /// The Gauss-Legendre rule on `[−1, 1]` by Golub-Welsch.
    fn legendre_rule(node_count: usize) -> (Vec<f64>, Vec<f64>) {
        let diagonal = vec![0.0; node_count];
        let off_diagonal = (1..node_count)
            .map(|index| {
                let degree = index as f64;
                degree / (4.0 * degree * degree - 1.0).sqrt()
            })
            .collect::<Vec<_>>();
        let (nodes, first_components) =
            symmetric_tridiagonal_eigen_first_components(&diagonal, &off_diagonal)
                .expect("Gauss-Legendre Golub-Welsch");
        let weights = first_components
            .iter()
            .map(|component| 2.0 * component * component)
            .collect();
        (nodes, weights)
    }

    /// `∫_lower^upper f` by a Gauss-Legendre rule; `integrand` returns the
    /// value and a bound on its magnitude for the rounding model.
    fn legendre_pass(
        rule: &(Vec<f64>, Vec<f64>),
        lower: f64,
        upper: f64,
        integrand: impl Fn(f64) -> (f64, f64),
    ) -> Pass {
        let half_width = 0.5 * (upper - lower);
        let midpoint = 0.5 * (upper + lower);
        let mut pass = Pass {
            sum: 0.0,
            absolute: 0.0,
        };
        for (node, weight) in rule.0.iter().zip(&rule.1) {
            let (value, magnitude) = integrand(midpoint + half_width * node);
            pass.sum += weight * half_width * value;
            pass.absolute += weight * half_width.abs() * magnitude;
        }
        pass
    }

    /// `E f(E)` for `E ~ N(0, 1)` by the Gauss-Hermite rule.
    fn hermite_pass(rule: &GaussHermiteRule, integrand: impl Fn(f64) -> (f64, f64)) -> Pass {
        let mut pass = Pass {
            sum: 0.0,
            absolute: 0.0,
        };
        for (node, weight) in rule.nodes.iter().zip(&rule.weights) {
            let (value, magnitude) = integrand(SQRT_2 * node);
            let normalized = weight / PI.sqrt();
            pass.sum += normalized * value;
            pass.absolute += normalized * magnitude;
        }
        pass
    }

    /// `Σ_j |c_j| xʲ` for `He_degree(x) = Σ_j c_j xʲ`: the magnitude the
    /// Hermite recurrence rounds against.
    fn absolute_hermite(degree: usize, x: f64) -> f64 {
        let x = x.abs();
        let mut previous = 0.0;
        let mut current = 1.0;
        for index in 0..degree {
            let next = x * current + index as f64 * previous;
            previous = current;
            current = next;
        }
        current
    }

    /// The summed magnitudes of the closed-form terms of `T⁽ᵏ⁾`.
    fn smoothing_magnitude(
        activation: GaussianActivation,
        t: f64,
        variance: f64,
        order: usize,
    ) -> f64 {
        match activation {
            GaussianActivation::Relu if variance == 0.0 => match order {
                0 => t.abs(),
                1 => 1.0,
                _ => 0.0,
            },
            GaussianActivation::Relu => {
                let scale = variance.sqrt();
                let u = t / scale;
                match order {
                    0 => t.abs() * normal_cdf(u) + scale * normal_pdf(u),
                    1 => normal_cdf(u),
                    _ => {
                        normal_pdf(u) * absolute_hermite(order - 2, u)
                            / scale.powi(order as i32 - 1)
                    }
                }
            }
            GaussianActivation::ExactGelu => {
                let total = 1.0 + variance;
                let root_total = total.sqrt();
                let u = t / root_total;
                match order {
                    0 => t.abs() * normal_cdf(u) + variance / root_total * normal_pdf(u),
                    1 => normal_cdf(u) + u.abs() * normal_pdf(u) / total,
                    _ => {
                        normal_pdf(u)
                            * (total * absolute_hermite(order - 2, u)
                                + absolute_hermite(order, u))
                            / (total * root_total.powi(order as i32 - 1))
                    }
                }
            }
            GaussianActivation::Silu => unreachable!("SiLU has no closed form here"),
        }
    }

    fn smoothing(
        activation: GaussianActivation,
        t: f64,
        variance: f64,
        orders: usize,
    ) -> Vec<f64> {
        let mut derivatives = vec![f64::NAN; orders];
        gaussian_smoothing_derivatives(activation, t, variance, &mut derivatives)
            .expect("valid smoothing law");
        derivatives
    }

    fn zero_mean(variance_x: f64, variance_y: f64, covariance: f64) -> PreactivationPair {
        PreactivationPair {
            mean_x: 0.0,
            mean_y: 0.0,
            variance_x,
            variance_y,
            covariance,
            covariance_rounding: 0.0,
        }
    }

    /// `(h_n(x), A_n(|x|)/√(n!))` for `n ≤ degree`: the orthonormal Hermite
    /// polynomials by their recurrence, and the magnitude their absolute-coefficient
    /// counterpart rounds against.
    fn normalized_hermite(degree: usize, x: f64) -> Vec<(f64, f64)> {
        let mut values = Vec::with_capacity(degree + 1);
        let mut previous = 0.0;
        let mut current = 1.0;
        let mut previous_magnitude = 0.0;
        let mut current_magnitude = 1.0;
        for index in 0..=degree {
            values.push((current, current_magnitude));
            let root = (index as f64).sqrt();
            let next_root = (index as f64 + 1.0).sqrt();
            let next = (x * current - root * previous) / next_root;
            let next_magnitude = (x.abs() * current_magnitude + root * previous_magnitude) / next_root;
            previous = current;
            current = next;
            previous_magnitude = current_magnitude;
            current_magnitude = next_magnitude;
        }
        values
    }

    /// A double-double `high + low` for reference recurrences: every operation errs
    /// by `O(ε²)` of its result.
    #[derive(Clone, Copy, Debug)]
    struct DoubleDouble {
        high: f64,
        low: f64,
    }

    impl DoubleDouble {
        fn from(value: f64) -> Self {
            Self {
                high: value,
                low: 0.0,
            }
        }

        fn two_sum(left: f64, right: f64) -> Self {
            let high = left + right;
            let virtual_right = high - left;
            let low = (left - (high - virtual_right)) + (right - virtual_right);
            Self { high, low }
        }

        fn add(self, other: Self) -> Self {
            let sum = Self::two_sum(self.high, other.high);
            Self::two_sum(sum.high, sum.low + self.low + other.low)
        }

        fn negated(self) -> Self {
            Self {
                high: -self.high,
                low: -self.low,
            }
        }

        fn mul(self, other: Self) -> Self {
            let high = self.high * other.high;
            let low = self.high.mul_add(other.high, -high)
                + (self.high * other.low + self.low * other.high);
            Self::two_sum(high, low)
        }

        fn div(self, other: Self) -> Self {
            let quotient = self.high / other.high;
            let remainder = self.add(other.mul(Self::from(quotient)).negated());
            Self::two_sum(quotient, remainder.high / other.high)
        }

        fn root_of(value: f64) -> Self {
            if value == 0.0 {
                return Self::from(0.0);
            }
            let root = value.sqrt();
            let residual = -root.mul_add(root, -value);
            Self::two_sum(root, residual / (2.0 * root))
        }

        /// Multiplication by an exactly representable power of two.
        fn scaled(self, factor: f64) -> Self {
            Self {
                high: self.high * factor,
                low: self.low * factor,
            }
        }

        fn sqrt(self) -> Self {
            let root = self.high.sqrt();
            let residual = self.add(Self::from(root).mul(Self::from(root)).negated());
            Self::two_sum(root, residual.high / (2.0 * root))
        }

        /// `exp` by reduction `x = k ln 2 + r`, the Taylor series of `r/2¹⁰` to
        /// sixteen terms (the first omitted term is below `1e-60`), ten squarings
        /// and the exact factor `2^k`; the squarings amplify its `O(ε²)` rounding
        /// by at most `2¹⁰`.
        fn exp(self) -> Self {
            const LN_2: DoubleDouble = DoubleDouble {
                high: std::f64::consts::LN_2,
                low: 2.319_046_813_846_299_6e-17,
            };
            const HALVINGS: i32 = 10;
            let multiple = (self.high / LN_2.high).round();
            let reduced = self
                .add(LN_2.mul(Self::from(multiple)).negated())
                .scaled(2.0_f64.powi(-HALVINGS));
            let mut term = Self::from(1.0);
            let mut sum = Self::from(1.0);
            let mut index = 1.0;
            while index <= 16.0 {
                term = term.mul(reduced).div(Self::from(index));
                sum = sum.add(term);
                index += 1.0;
            }
            let mut remaining = HALVINGS;
            while remaining > 0 {
                sum = sum.mul(sum);
                remaining -= 1;
            }
            sum.scaled(2.0_f64.powi(multiple as i32))
        }
    }

    /// `atan` of a nonnegative double-double: three half-angle reductions
    /// `t → t/(1 + √(1 + t²))` bring it below `tan(π/16) < 0.2`, where the
    /// alternating series to twenty-five terms leaves under `1e-36`.
    fn double_double_atan(argument: DoubleDouble) -> DoubleDouble {
        let one = DoubleDouble::from(1.0);
        let mut reduced = argument;
        let mut halvings = 3;
        while halvings > 0 {
            reduced = reduced.div(one.add(one.add(reduced.mul(reduced)).sqrt()));
            halvings -= 1;
        }
        let square = reduced.mul(reduced);
        let mut power = reduced;
        let mut sum = DoubleDouble::from(0.0);
        let mut index = 0_i32;
        while index < 25 {
            let term = power.div(DoubleDouble::from(f64::from(2 * index + 1)));
            sum = if index % 2 == 0 {
                sum.add(term)
            } else {
                sum.add(term.negated())
            };
            power = power.mul(square);
            index += 1;
        }
        sum.scaled(8.0)
    }

    /// `atan2(y, x)` for `y ≥ 0` in double-double.
    fn double_double_atan2(height: DoubleDouble, abscissa: f64) -> DoubleDouble {
        let pi = DoubleDouble {
            high: PI,
            low: 1.224_646_799_147_353_2e-16,
        };
        if abscissa > 0.0 {
            double_double_atan(height.div(DoubleDouble::from(abscissa)))
        } else if abscissa < 0.0 {
            pi.add(double_double_atan(height.div(DoubleDouble::from(-abscissa))).negated())
        } else {
            pi.scaled(0.5)
        }
    }

    /// `|value − reference|` for a double-double reference.
    fn double_double_discrepancy(value: f64, reference: DoubleDouble) -> f64 {
        ((value - reference.high) - reference.low).abs()
    }

    /// `φ(x) = exp(−x²/2)/√(2π)` in double-double at a computed argument.
    fn double_double_normal_pdf(argument: f64) -> DoubleDouble {
        let pi = DoubleDouble {
            high: PI,
            low: 1.224_646_799_147_353_2e-16,
        };
        DoubleDouble::from(argument)
            .mul(DoubleDouble::from(argument))
            .scaled(-0.5)
            .exp()
            .div(pi.scaled(2.0).sqrt())
    }

    /// Mills' ratio `R(x) = Φ(−x)/φ(x)`, `x > 0`, by the Laplace continued fraction
    /// `1/(x + 1/(x + 2/(x + 3/(…))))` truncated after `depth` levels, in
    /// double-double. Every level adds positive quantities, so nothing cancels.
    fn double_double_mills_ratio(magnitude: f64, depth: usize) -> DoubleDouble {
        let argument = DoubleDouble::from(magnitude);
        let mut tail = argument;
        let mut level = depth;
        while level > 0 {
            tail = argument.add(DoubleDouble::from(level as f64).div(tail));
            level -= 1;
        }
        DoubleDouble::from(1.0).div(tail)
    }

    /// The bivariate normal owner's published absolute bound per value of `Φ₂`.
    const BIVARIATE_NORMAL_CONTRACT: f64 = BIVARIATE_NORMAL_CDF_ERROR_BOUND;

    /// `(H, ∂_hΦ₂, ∂_kΦ₂, φ₂)` at a standardized pair law, as the biased kernels
    /// form them, with the degenerate law's partials bounded by `φ(h)`, `φ(k)`.
    fn standardized_orthant(h: f64, k: f64, correlation: f64, complement: f64) -> (f64, f64, f64, f64) {
        let orthant = bivariate_normal_cdf_with_complement(h, k, correlation, complement)
            .expect("standardized orthant");
        if complement > 0.0 {
            let partials = bivariate_normal_cdf_partials_with_complement(h, k, correlation, complement)
                .expect("standardized partials");
            (orthant, partials.d_h, partials.d_k, partials.d_rho)
        } else {
            (orthant, normal_pdf(h), normal_pdf(k), 0.0)
        }
    }

    /// The summed term magnitudes of the biased closed forms for the rounding
    /// model, `(K, ∂_r K)`, for positive variances; `Φ₂`'s contract enters apart.
    fn biased_magnitudes(
        activation: GaussianActivation,
        mean_x: f64,
        mean_y: f64,
        variance_x: f64,
        variance_y: f64,
        covariance: f64,
    ) -> (f64, f64) {
        let residual = (variance_x * variance_y - covariance * covariance).max(0.0);
        let coupled_means = (mean_x * mean_y + covariance).abs();
        match activation {
            GaussianActivation::Relu => {
                let root_x = variance_x.sqrt();
                let root_y = variance_y.sqrt();
                let scale = root_x * root_y;
                let correlation = (covariance / scale).clamp(-1.0, 1.0);
                let complement = (residual / (variance_x * variance_y)).min(1.0);
                let (orthant, partial_x, partial_y, density) =
                    standardized_orthant(mean_x / root_x, mean_y / root_y, correlation, complement);
                (
                    coupled_means * orthant
                        + mean_y.abs() * root_x * partial_x
                        + mean_x.abs() * root_y * partial_y
                        + scale * complement * density,
                    orthant,
                )
            }
            GaussianActivation::ExactGelu => {
                let total_x = 1.0 + variance_x;
                let total_y = 1.0 + variance_y;
                let root_product = (total_x * total_y).sqrt();
                let discriminant = 1.0 + variance_x + variance_y + residual;
                let correlation = (covariance / root_product).clamp(-1.0, 1.0);
                let complement = (discriminant / (total_x * total_y)).min(1.0);
                let (orthant, d_h, d_k, d_rho) = standardized_orthant(
                    mean_x / total_x.sqrt(),
                    mean_y / total_y.sqrt(),
                    correlation,
                    complement,
                );
                let partial_x = d_h / total_x.sqrt();
                let partial_y = d_k / total_y.sqrt();
                let density = d_rho / root_product;
                let reduced = (total_y * mean_x - covariance * mean_y)
                    * (total_x * mean_y - covariance * mean_x)
                    / (discriminant * discriminant);
                let coupling =
                    residual + covariance * covariance * (1.0 / total_x + 1.0 / total_y);
                (
                    coupled_means * orthant
                        + (mean_y * variance_x + mean_x * covariance / total_x).abs() * partial_x
                        + (mean_x * variance_y + mean_y * covariance / total_y).abs() * partial_y
                        + coupling * density,
                    orthant
                        + (mean_x.abs() * partial_x + covariance.abs() * density) / total_x
                        + (mean_y.abs() * partial_y + covariance.abs() * density) / total_y
                        + (covariance / discriminant + reduced).abs() * density,
                )
            }
            GaussianActivation::Silu => unreachable!("SiLU has no closed form here"),
        }
    }

    /// `E[X₊ Y₊]` on the degenerate law `Y = c + κ (X − b)`, `κ = ±√(w/v)`, by
    /// Gauss-Legendre over the interval where both pre-activations are positive,
    /// on which `x m(x) p_X(x)` is analytic. With `x = b + √v e` and
    /// `m = c ± √w e`, the mass beyond `|e| = L` is at most
    /// `2φ(L) [|b||c|/L + |b|√w + |c|√v + √(vw)(L + 1/L)]`.
    fn relu_degenerate_pair_reference(
        mean_x: f64,
        mean_y: f64,
        variance_x: f64,
        variance_y: f64,
        sign: f64,
        node_count: usize,
    ) -> Reference {
        const REACH: f64 = 10.0;
        let root_x = variance_x.sqrt();
        let root_y = variance_y.sqrt();
        let slope = sign * root_y / root_x;
        let crossing = mean_x - mean_y / slope;
        let mut lower = (mean_x - REACH * root_x).max(0.0);
        let mut upper = mean_x + REACH * root_x;
        if slope > 0.0 {
            lower = lower.max(crossing);
        } else {
            upper = upper.min(crossing);
        }
        let tail = 2.0
            * normal_pdf(REACH)
            * (mean_x.abs() * mean_y.abs() / REACH
                + mean_x.abs() * root_y
                + mean_y.abs() * root_x
                + root_x * root_y * (REACH + 1.0 / REACH));
        if upper <= lower {
            return Reference {
                value: 0.0,
                bound: tail,
            };
        }
        let integrand = |x: f64| {
            let value = x
                * (mean_y + slope * (x - mean_x))
                * normal_pdf((x - mean_x) / root_x)
                / root_x;
            (value, value.abs())
        };
        doubled_order(
            legendre_pass(&legendre_rule(node_count), lower, upper, integrand),
            legendre_pass(&legendre_rule(2 * node_count), lower, upper, integrand),
            2 * node_count,
            2 * node_count,
            tail,
        )
    }

    /// `sⁿ/√(n!)`.
    fn hermite_normalizer(scale: f64, order: usize) -> f64 {
        (1..=order).fold(1.0, |normalizer, index| normalizer * scale / (index as f64).sqrt())
    }

    /// The terms of the zero-mean exact-GELU closed forms, recomputed for the
    /// rounding model: `|r H| + (vw + r² q)/(2π√Δ)` for `K` and
    /// `H + |r| (1/A + 1/B + 1/Δ)/(2π√Δ)` for `∂_r K`.
    fn exact_gelu_zero_mean_magnitudes(
        variance_x: f64,
        variance_y: f64,
        covariance: f64,
    ) -> (f64, f64) {
        let residual = (variance_x * variance_y - covariance * covariance).max(0.0);
        let discriminant = 1.0 + variance_x + variance_y + residual;
        let orthant = discriminant.sqrt().atan2(-covariance) / TAU;
        let density = 1.0 / (TAU * discriminant.sqrt());
        let inverse_totals = 1.0 / (1.0 + variance_x) + 1.0 / (1.0 + variance_y);
        (
            covariance.abs() * orthant
                + (residual + covariance * covariance * inverse_totals) * density,
            orthant + covariance.abs() * density * (inverse_totals + 1.0 / discriminant),
        )
    }

    /// `E[ReLU⁽ᵏ⁾(t + √v E)]`, `k ∈ {0, 1}`, as `s^p ∫_ℓ^∞ (e − ℓ)^p φ(e) de`
    /// with `ℓ = −t/√v` and `p = 1 − k`. The kink sits on the endpoint, so the
    /// integrand is analytic on the domain; the domain is truncated to where
    /// the Gaussian mass lives, and each discarded tail is bounded in closed
    /// form and added to the error bound.
    fn relu_smoothing_reference(
        t: f64,
        variance: f64,
        order: usize,
        node_count: usize,
    ) -> Reference {
        // Truncation reaches choose the domain, not the tolerance: the bound
        // of every discarded tail joins the reference's error bound.
        const GAUSSIAN_REACH: f64 = 10.0;
        const EXPONENTIAL_REACH: f64 = 45.0;
        let scale = variance.sqrt();
        let lower = -t / scale;
        let power = (1 - order) as i32;
        let (start, end, truncation) = if lower <= 0.0 {
            // Φ(−G) ≤ φ(G)/G. Above G: ∫ (e − ℓ) φ = φ(G) + |ℓ| Φ(−G) and
            // ∫ φ = Φ(−G). Below −G (when ℓ < −G): ∫ (e − ℓ) φ ≤ |ℓ| Φ(−G)
            // and ∫ φ ≤ Φ(−G).
            let tail = normal_pdf(GAUSSIAN_REACH) / GAUSSIAN_REACH;
            let bound = if power == 1 {
                normal_pdf(GAUSSIAN_REACH) + 2.0 * lower.abs() * tail
            } else {
                2.0 * tail
            };
            (lower.max(-GAUSSIAN_REACH), GAUSSIAN_REACH, bound)
        } else {
            // φ(ℓ + y) ≤ φ(ℓ) min(e^{−y²/2}, e^{−ℓy}) for y ≥ 0, integrated
            // against y^p beyond the reach Y.
            let reach = GAUSSIAN_REACH.min(EXPONENTIAL_REACH / lower);
            let gaussian_tail = (-0.5 * reach * reach).exp();
            let exponential_tail = (-lower * reach).exp();
            let bound = if power == 1 {
                normal_pdf(lower)
                    * gaussian_tail
                        .min(exponential_tail * (reach / lower + 1.0 / (lower * lower)))
            } else {
                normal_pdf(lower) * (gaussian_tail / reach).min(exponential_tail / lower)
            };
            (lower, lower + reach, bound)
        };
        let integrand = |e: f64| {
            let value = (e - lower).powi(power) * normal_pdf(e);
            (value, value.abs())
        };
        let coarse = legendre_pass(&legendre_rule(node_count), start, end, integrand);
        let fine = legendre_pass(&legendre_rule(2 * node_count), start, end, integrand);
        let reference = doubled_order(coarse, fine, 2 * node_count, 2 * node_count, truncation);
        let factor = scale.powi(power);
        Reference {
            value: factor * reference.value,
            bound: factor * reference.bound,
        }
    }

    /// `σ⁽ᵏ⁾(x)` of the exact GELU for `k ≤ 2`, with its terms' magnitude.
    fn exact_gelu_derivative(x: f64, order: usize) -> (f64, f64) {
        let density = normal_pdf(x);
        let probability = normal_cdf(x);
        match order {
            0 => (x * probability, (x * probability).abs()),
            1 => (probability + x * density, probability + (x * density).abs()),
            _ => ((2.0 - x * x) * density, (2.0 + x * x) * density),
        }
    }

    /// [`exact_gelu_pair_passes`] at zero means.
    fn exact_gelu_zero_mean_pair_passes(
        rule: &GaussHermiteRule,
        variance_x: f64,
        variance_y: f64,
        covariance: f64,
    ) -> (Pass, Pass) {
        exact_gelu_pair_passes(rule, 0.0, 0.0, variance_x, variance_y, covariance)
    }

    /// `E[σ(X) σ(Y)]` and `E[σ'(X) σ'(Y)]` for the exact GELU by the tensor
    /// Gauss-Hermite rule on `X = b + √v E₁`, `Y = c + √w (ρ E₁ + √(1 − ρ²) E₂)`.
    /// The inner sum over `E₂` is completed per outer node, so recursive summation
    /// rounds over `2n` summands, not `n²`.
    fn exact_gelu_pair_passes(
        rule: &GaussHermiteRule,
        mean_x: f64,
        mean_y: f64,
        variance_x: f64,
        variance_y: f64,
        covariance: f64,
    ) -> (Pass, Pass) {
        let scale = variance_x.sqrt() * variance_y.sqrt();
        let correlation = if scale == 0.0 {
            0.0
        } else {
            (covariance / scale).clamp(-1.0, 1.0)
        };
        let complement = ((1.0 - correlation) * (1.0 + correlation)).sqrt();
        let root_x = variance_x.sqrt();
        let root_y = variance_y.sqrt();
        let normalized = rule
            .weights
            .iter()
            .map(|weight| weight / PI.sqrt())
            .collect::<Vec<_>>();
        let mut kernel = Pass {
            sum: 0.0,
            absolute: 0.0,
        };
        let mut derivative = kernel;
        for (first_node, first_weight) in rule.nodes.iter().zip(&normalized) {
            let first = SQRT_2 * first_node;
            let (activation_x, activation_magnitude_x) =
                exact_gelu_derivative(mean_x + root_x * first, 0);
            let (slope_x, slope_magnitude_x) = exact_gelu_derivative(mean_x + root_x * first, 1);
            let mut inner_kernel = Pass {
                sum: 0.0,
                absolute: 0.0,
            };
            let mut inner_derivative = inner_kernel;
            for (second_node, second_weight) in rule.nodes.iter().zip(&normalized) {
                let y = mean_y + root_y * (correlation * first + complement * SQRT_2 * second_node);
                let (activation_y, activation_magnitude_y) = exact_gelu_derivative(y, 0);
                let (slope_y, slope_magnitude_y) = exact_gelu_derivative(y, 1);
                inner_kernel.sum += second_weight * activation_y;
                inner_kernel.absolute += second_weight * activation_magnitude_y;
                inner_derivative.sum += second_weight * slope_y;
                inner_derivative.absolute += second_weight * slope_magnitude_y;
            }
            kernel.sum += first_weight * activation_x * inner_kernel.sum;
            kernel.absolute += first_weight * activation_magnitude_x * inner_kernel.absolute;
            derivative.sum += first_weight * slope_x * inner_derivative.sum;
            derivative.absolute += first_weight * slope_magnitude_x * inner_derivative.absolute;
        }
        (kernel, derivative)
    }

    /// `E[X₊ Y₊]` at zero mean by the angular Gauss-Legendre rule over the
    /// wedge where both pre-activations are positive. In polar coordinates of
    /// the standard pair `(E₁, E₂)`, `X = √v R cos φ` and `Y = √w R cos(φ − θ)`
    /// with `cos θ = ρ`; the radial moment `∫₀^∞ R³ e^{−R²/2} dR = 2` leaves
    /// `(√(vw)/π) ∫₀^{θ'} sin ψ sin(θ' − ψ) dψ`, `θ' = π − θ = arccos(−ρ)`,
    /// whose integrand is nonnegative and carries no cancellation.
    fn relu_zero_mean_pair_reference(
        variance_x: f64,
        variance_y: f64,
        covariance: f64,
        node_count: usize,
    ) -> Reference {
        let scale = variance_x.sqrt() * variance_y.sqrt();
        let correlation = (covariance / scale).clamp(-1.0, 1.0);
        let opening = (-correlation).acos();
        let integrand = |x: f64| {
            let value = (0.5 * opening * (1.0 + x)).sin() * (0.5 * opening * (1.0 - x)).sin();
            (value, value.abs())
        };
        let coarse = legendre_pass(&legendre_rule(node_count), -1.0, 1.0, integrand);
        let fine = legendre_pass(&legendre_rule(2 * node_count), -1.0, 1.0, integrand);
        let reference = doubled_order(coarse, fine, 2 * node_count, 2 * node_count, 0.0);
        let factor = 0.5 * opening * scale / PI;
        Reference {
            value: factor * reference.value,
            bound: factor * reference.bound,
        }
    }

    #[test]
    fn relu_smoothing_matches_a_kink_free_legendre_reference_into_the_tails() {
        // (t, v): the lower tail at u = −24 and u = −30 (the continued-fraction
        // side of the log-CDF owner), the upper tail at u = 1.5e5, the kink, a
        // vanishing smoothing (v = 1e-12 at u = −1) and moderate laws.
        let laws = [
            (-12.0, 0.25),
            (-30.0, 1.0),
            (-3.0, 1.0),
            (0.0, 2.0),
            (0.7, 0.3),
            (6.0, 1.0),
            (1.5, 1.0e-10),
            (-1.0e-6, 1.0e-12),
            (7.0, 9.0),
        ];
        for (t, variance) in laws {
            let closed = smoothing(GaussianActivation::Relu, t, variance, 2);
            for order in 0..2 {
                let reference = relu_smoothing_reference(t, variance, order, 64);
                let tolerance = reference.bound
                    + closed_form_rounding(smoothing_magnitude(
                        GaussianActivation::Relu,
                        t,
                        variance,
                        order,
                    ));
                let discrepancy = (closed[order] - reference.value).abs();
                assert!(
                    discrepancy <= tolerance,
                    "ReLU T^({order}) at t = {t}, v = {variance}: closed form {} against reference {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                    closed[order],
                    reference.value
                );
            }
        }
    }

    #[test]
    fn exact_gelu_smoothing_matches_gauss_hermite_into_the_tails() {
        // Φ(t + √v E) grows like exp(v y²) off the real axis, so Gauss-Hermite
        // converges roughly like exp(−2n/(1 + v)): 256 and 512 nodes resolve
        // every v ≤ 16 here, and the doubled order checks it.
        let coarse_rule = gauss_hermite_rule(256).expect("256-node Gauss-Hermite rule");
        let fine_rule = gauss_hermite_rule(512).expect("512-node Gauss-Hermite rule");
        let laws = [
            (-8.0, 0.5),
            (-40.0, 4.0),
            (-3.0, 4.0),
            (0.0, 0.0),
            (0.4, 1.0e-10),
            (2.5, 16.0),
            (7.0, 1.0),
            (-1.0, 9.0),
        ];
        for (t, variance) in laws {
            let closed = smoothing(GaussianActivation::ExactGelu, t, variance, 3);
            for order in 0..3 {
                let integrand = |e: f64| exact_gelu_derivative(t + variance.sqrt() * e, order);
                let reference = doubled_order(
                    hermite_pass(&coarse_rule, integrand),
                    hermite_pass(&fine_rule, integrand),
                    512,
                    512,
                    0.0,
                );
                let tolerance = reference.bound
                    + closed_form_rounding(smoothing_magnitude(
                        GaussianActivation::ExactGelu,
                        t,
                        variance,
                        order,
                    ));
                let discrepancy = (closed[order] - reference.value).abs();
                assert!(
                    discrepancy <= tolerance,
                    "exact GELU T^({order}) at t = {t}, v = {variance}: closed form {} against reference {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                    closed[order],
                    reference.value
                );
            }
        }
    }

    #[test]
    fn smoothing_solves_the_heat_equation_in_integral_form() {
        // T⁽ᵏ⁾_{v₂}(t) − T⁽ᵏ⁾_{v₁}(t) = ½ ∫_{v₁}^{v₂} T⁽ᵏ⁺²⁾_v(t) dv. Both
        // smoothings are analytic in v on a neighbourhood of [v₁, v₂] whose
        // points have positive real part, so Gauss-Legendre converges
        // geometrically.
        const LOWER_VARIANCE: f64 = 0.2;
        const UPPER_VARIANCE: f64 = 3.0;
        let coarse_rule = legendre_rule(32);
        let fine_rule = legendre_rule(64);
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for t in [-2.5, 0.3, 4.0] {
                let upper = smoothing(activation, t, UPPER_VARIANCE, 4);
                let lower = smoothing(activation, t, LOWER_VARIANCE, 4);
                for order in 0..4 {
                    let integrand = |variance: f64| {
                        let derivatives = smoothing(activation, t, variance, order + 3);
                        (
                            0.5 * derivatives[order + 2],
                            0.5 * smoothing_magnitude(activation, t, variance, order + 2),
                        )
                    };
                    let reference = doubled_order(
                        legendre_pass(&coarse_rule, LOWER_VARIANCE, UPPER_VARIANCE, integrand),
                        legendre_pass(&fine_rule, LOWER_VARIANCE, UPPER_VARIANCE, integrand),
                        64,
                        64,
                        0.0,
                    );
                    let increment = upper[order] - lower[order];
                    let tolerance = reference.bound
                        + closed_form_rounding(
                            smoothing_magnitude(activation, t, UPPER_VARIANCE, order)
                                + smoothing_magnitude(activation, t, LOWER_VARIANCE, order),
                        );
                    let discrepancy = (increment - reference.value).abs();
                    assert!(
                        discrepancy <= tolerance,
                        "{activation:?} heat equation for T^({order}) at t = {t}: increment {increment} against ½∫T^({}) = {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                        order + 2,
                        reference.value
                    );
                    if order == 0 && t == 0.3 {
                        // Positive control: the same increment against the unhalved
                        // integral is rejected by the same tolerance.
                        let control = (increment - 2.0 * reference.value).abs();
                        assert!(
                            control > tolerance,
                            "{activation:?} heat-equation control at t = {t}: discrepancy {control:e} within tolerance {tolerance:e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn smoothing_derivatives_integrate_to_the_orders_below() {
        // T⁽ᵏ⁾(t₂) − T⁽ᵏ⁾(t₁) = ∫_{t₁}^{t₂} T⁽ᵏ⁺¹⁾(t) dt, with entire integrands.
        const START: f64 = -3.0;
        const END: f64 = 2.0;
        let coarse_rule = legendre_rule(48);
        let fine_rule = legendre_rule(96);
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for variance in [0.5, 4.0] {
                let start = smoothing(activation, START, variance, 6);
                let end = smoothing(activation, END, variance, 6);
                for order in 0..5 {
                    let integrand = |t: f64| {
                        (
                            smoothing(activation, t, variance, order + 2)[order + 1],
                            smoothing_magnitude(activation, t, variance, order + 1),
                        )
                    };
                    let reference = doubled_order(
                        legendre_pass(&coarse_rule, START, END, integrand),
                        legendre_pass(&fine_rule, START, END, integrand),
                        96,
                        96,
                        0.0,
                    );
                    let increment = end[order] - start[order];
                    let tolerance = reference.bound
                        + closed_form_rounding(
                            smoothing_magnitude(activation, END, variance, order)
                                + smoothing_magnitude(activation, START, variance, order),
                        );
                    let discrepancy = (increment - reference.value).abs();
                    assert!(
                        discrepancy <= tolerance,
                        "{activation:?} T^({order}) increment on [{START}, {END}] at v = {variance}: {increment} against ∫T^({}) = {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                        order + 1,
                        reference.value
                    );
                }
                // Positive control: the value's increment against the integral of
                // the second derivative is rejected.
                let integrand = |t: f64| {
                    (
                        smoothing(activation, t, variance, 3)[2],
                        smoothing_magnitude(activation, t, variance, 2),
                    )
                };
                let mismatched = doubled_order(
                    legendre_pass(&coarse_rule, START, END, integrand),
                    legendre_pass(&fine_rule, START, END, integrand),
                    96,
                    96,
                    0.0,
                );
                let control = (end[0] - start[0] - mismatched.value).abs();
                assert!(
                    control > mismatched.bound,
                    "{activation:?} derivative-chain control at v = {variance}: discrepancy {control:e} within {:e}",
                    mismatched.bound
                );
            }
        }
    }

    #[test]
    fn relu_smoothing_without_variance_is_the_activation_with_its_gaussian_limit_slope() {
        for (t, value, slope) in [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.5), (3.0, 3.0, 1.0)] {
            let mut derivatives = [f64::NAN; 2];
            gaussian_smoothing_derivatives(GaussianActivation::Relu, t, 0.0, &mut derivatives)
                .expect("ReLU value and slope without smoothing");
            assert_eq!(derivatives, [value, slope], "ReLU at t = {t} with v = 0");
        }
        let mut away_from_kink = [f64::NAN; 4];
        gaussian_smoothing_derivatives(GaussianActivation::Relu, 1.5, 0.0, &mut away_from_kink)
            .expect("ReLU curvature away from the kink");
        assert_eq!(away_from_kink, [1.5, 1.0, 0.0, 0.0]);
        let mut at_kink = [f64::NAN; 3];
        assert_eq!(
            gaussian_smoothing_derivatives(GaussianActivation::Relu, 0.0, 0.0, &mut at_kink),
            Err(GaussianActivationError::UnsmoothedReluKink { order: 2 })
        );
        // As v → 0, T_v ReLU(t) − t₊ = √v L(|t|/√v) with the unit normal loss
        // L(x) = φ(x) − x Φ(−x) ∈ [0, φ(0)], so the smoothing reaches the
        // activation within √v φ(0).
        for variance in [1.0e-4, 1.0e-10, 1.0e-16] {
            for t in [-1.0e-3, 0.0, 2.0e-3] {
                let value = smoothing(GaussianActivation::Relu, t, variance, 1)[0];
                let rounding = closed_form_rounding(smoothing_magnitude(
                    GaussianActivation::Relu,
                    t,
                    variance,
                    0,
                ));
                let excess = value - t.max(0.0);
                assert!(
                    excess >= -rounding && excess <= variance.sqrt() * normal_pdf(0.0) + rounding,
                    "ReLU smoothing at t = {t}, v = {variance}: excess {excess:e} over t₊ outside [0, √v φ(0)]"
                );
            }
        }
    }

    #[test]
    fn relu_zero_mean_pair_kernel_matches_the_wedge_reference_up_to_the_correlation_limits() {
        let laws = [
            (1.0, 1.0, 0.3),
            (0.25, 4.0, -0.9),
            (9.0, 4.0, 5.994),
            (16.0, 9.0, -6.0),
            (4.0, 4.0, 4.0 * (1.0 - 1.0e-12)),
            (4.0, 4.0, 4.0),
            (4.0, 4.0, -4.0 * (1.0 - 1.0e-6)),
            (4.0, 4.0, -4.0 * (1.0 - 1.0e-12)),
            (4.0, 4.0, -4.0),
            (1.0e-300, 1.0, 0.5e-150),
        ];
        for (variance_x, variance_y, covariance) in laws {
            let closed = pair_kernel(
                GaussianActivation::Relu,
                zero_mean(variance_x, variance_y, covariance),
            )
            .expect("zero-mean ReLU pair kernel");
            let reference = relu_zero_mean_pair_reference(variance_x, variance_y, covariance, 16);
            let root_residual = (variance_x * variance_y - covariance * covariance)
                .max(0.0)
                .sqrt();
            let magnitude =
                root_residual / TAU + covariance.abs() * closed.covariance_derivative;
            let tolerance = reference.bound + closed_form_rounding(magnitude);
            let discrepancy = (closed.value - reference.value).abs();
            assert!(
                discrepancy <= tolerance,
                "ReLU K at v = {variance_x}, w = {variance_y}, r = {covariance}: closed form {} against reference {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                closed.value,
                reference.value
            );
        }
        // The orthant probability at the correlation limits and independence.
        for (covariance, expected) in [(6.0, 0.5), (-6.0, 0.0), (0.0, 0.25)] {
            let closed = pair_kernel(GaussianActivation::Relu, zero_mean(4.0, 9.0, covariance))
                .expect("zero-mean ReLU pair kernel");
            assert_eq!(closed.covariance_derivative, expected, "ReLU ∂_r K at r = {covariance}");
        }
        // Positive control: the kernel with θ in place of π − θ is rejected.
        let reference = relu_zero_mean_pair_reference(1.0, 1.0, 0.3, 16);
        let wrong = (0.91_f64.sqrt() + 0.3 * 0.3_f64.acos()) / TAU;
        let control = (wrong - reference.value).abs();
        assert!(
            control > reference.bound + closed_form_rounding(wrong),
            "ReLU wedge control: discrepancy {control:e} within {:e}",
            reference.bound
        );
    }

    #[test]
    fn exact_gelu_zero_mean_pair_kernel_and_price_derivative_match_gauss_hermite() {
        // X = √v E₁ reads Φ on the scale 1/√v, so the rule converges roughly
        // like exp(−2n/(1 + v)); the large-norm law (ρ_G = r/√(AB) → 1) takes
        // 1024/2048 nodes and the rest 256/512. The doubled order checks it.
        let moderate = (
            gauss_hermite_rule(256).expect("256-node Gauss-Hermite rule"),
            gauss_hermite_rule(512).expect("512-node Gauss-Hermite rule"),
        );
        let wide = (
            gauss_hermite_rule(1024).expect("1024-node Gauss-Hermite rule"),
            gauss_hermite_rule(2048).expect("2048-node Gauss-Hermite rule"),
        );
        let laws = [
            (1.0, 1.0, 0.3, &moderate),
            (0.25, 4.0, -0.9, &moderate),
            (9.0, 4.0, 5.994, &moderate),
            (16.0, 9.0, -12.0 * (1.0 - 1.0e-8), &moderate),
            (4.0, 4.0, 4.0 * (1.0 - 1.0e-12), &moderate),
            (4.0, 4.0, 4.0, &moderate),
            (4.0, 4.0, -4.0, &moderate),
            (1.0e-12, 1.0, 0.5e-6, &moderate),
            (0.0, 2.0, 0.0, &moderate),
            (64.0, 64.0, 64.0 * 0.999, &wide),
        ];
        for (variance_x, variance_y, covariance, rules) in laws {
            let closed = pair_kernel(
                GaussianActivation::ExactGelu,
                zero_mean(variance_x, variance_y, covariance),
            )
            .expect("zero-mean exact GELU pair kernel");
            let (coarse_kernel, coarse_derivative) =
                exact_gelu_zero_mean_pair_passes(&rules.0, variance_x, variance_y, covariance);
            let (fine_kernel, fine_derivative) =
                exact_gelu_zero_mean_pair_passes(&rules.1, variance_x, variance_y, covariance);
            let nodes = rules.1.nodes.len();
            let kernel = doubled_order(coarse_kernel, fine_kernel, 2 * nodes, 2 * nodes, 0.0);
            let derivative =
                doubled_order(coarse_derivative, fine_derivative, 2 * nodes, 2 * nodes, 0.0);
            let (kernel_magnitude, derivative_magnitude) =
                exact_gelu_zero_mean_magnitudes(variance_x, variance_y, covariance);
            let kernel_tolerance = kernel.bound + closed_form_rounding(kernel_magnitude);
            let derivative_tolerance =
                derivative.bound + closed_form_rounding(derivative_magnitude);
            let kernel_discrepancy = (closed.value - kernel.value).abs();
            let derivative_discrepancy = (closed.covariance_derivative - derivative.value).abs();
            assert!(
                kernel_discrepancy <= kernel_tolerance,
                "exact GELU K at v = {variance_x}, w = {variance_y}, r = {covariance}: closed form {} against reference {} (discrepancy {kernel_discrepancy:e}, derived tolerance {kernel_tolerance:e})",
                closed.value,
                kernel.value
            );
            assert!(
                derivative_discrepancy <= derivative_tolerance,
                "exact GELU ∂_r K at v = {variance_x}, w = {variance_y}, r = {covariance}: closed form {} against E[σ'σ'] {} (discrepancy {derivative_discrepancy:e}, derived tolerance {derivative_tolerance:e})",
                closed.covariance_derivative,
                derivative.value
            );
            if variance_x == 9.0 {
                // Positive controls: the kernel without the r² q coupling and the
                // derivative without Price's density term are both rejected.
                let total_x = 1.0 + variance_x;
                let total_y = 1.0 + variance_y;
                let discriminant = total_x * total_y - covariance * covariance;
                let density = 1.0 / (TAU * discriminant.sqrt());
                let coupling = 1.0 / total_x + 1.0 / total_y - 1.0;
                let wrong_kernel = closed.value - covariance * covariance * coupling * density;
                let wrong_derivative = closed.covariance_derivative
                    - covariance * density * (1.0 / total_x + 1.0 / total_y + 1.0 / discriminant);
                assert!(
                    (wrong_kernel - kernel.value).abs() > kernel_tolerance,
                    "exact GELU kernel control within tolerance {kernel_tolerance:e}"
                );
                assert!(
                    (wrong_derivative - derivative.value).abs() > derivative_tolerance,
                    "exact GELU derivative control within tolerance {derivative_tolerance:e}"
                );
            }
        }
    }

    #[test]
    fn exact_gelu_kernel_resolves_large_norm_anticorrelated_readers() {
        // At v = w and r = −v the pre-activations are Y = −X with X ~ N(0, v),
        // so K = −E[X² Φ(X) Φ(−X)] and ∂_r K = E[σ'(X) σ'(−X)] are one-dimensional
        // with analytic integrands on the O(1) scale of Φ, whatever v is. For
        // |x| ≥ 1, x² Φ(x) Φ(−x) ≤ |x| φ(x) and |σ'(x) σ'(−x)| ≤ sup|σ'| |x| φ(x),
        // and the weight φ(x/√v)/√v ≤ 1/√(2πv), so the mass beyond |x| = L is at
        // most 2 φ(L)/√(2πv), times sup|σ'| for the derivative. Here the closed
        // form's two √v-sized terms cancel to a K of order 1/√v.
        const REACH: f64 = 8.0;
        let coarse_rule = legendre_rule(64);
        let fine_rule = legendre_rule(128);
        for variance in [4096.0, 1.0e6, 1.0e8] {
            let covariance = -variance;
            let closed = pair_kernel(
                GaussianActivation::ExactGelu,
                zero_mean(variance, variance, covariance),
            )
            .expect("anti-correlated exact GELU pair kernel");
            let root_variance = variance.sqrt();
            let weight = |x: f64| normal_pdf(x / root_variance) / root_variance;
            let kernel_integrand = |x: f64| {
                let value = -x * x * normal_cdf(x) * normal_cdf(-x) * weight(x);
                (value, value.abs())
            };
            let derivative_integrand = |x: f64| {
                let density = normal_pdf(x);
                let upper = normal_cdf(x) + x * density;
                let lower = normal_cdf(-x) - x * density;
                (
                    upper * lower * weight(x),
                    (normal_cdf(x) + (x * density).abs())
                        * (normal_cdf(-x) + (x * density).abs())
                        * weight(x),
                )
            };
            let tail = 2.0 * normal_pdf(REACH) / (TAU * variance).sqrt();
            let kernel = doubled_order(
                legendre_pass(&coarse_rule, -REACH, REACH, kernel_integrand),
                legendre_pass(&fine_rule, -REACH, REACH, kernel_integrand),
                128,
                128,
                tail,
            );
            let derivative = doubled_order(
                legendre_pass(&coarse_rule, -REACH, REACH, derivative_integrand),
                legendre_pass(&fine_rule, -REACH, REACH, derivative_integrand),
                128,
                128,
                EXACT_GELU_SLOPE_BOUND * tail,
            );
            let (kernel_magnitude, derivative_magnitude) =
                exact_gelu_zero_mean_magnitudes(variance, variance, covariance);
            let kernel_tolerance = kernel.bound + closed_form_rounding(kernel_magnitude);
            let derivative_tolerance =
                derivative.bound + closed_form_rounding(derivative_magnitude);
            let kernel_discrepancy = (closed.value - kernel.value).abs();
            let derivative_discrepancy = (closed.covariance_derivative - derivative.value).abs();
            assert!(
                kernel_discrepancy <= kernel_tolerance,
                "exact GELU K at v = w = −r = {variance}: closed form {} against the degenerate law {} (discrepancy {kernel_discrepancy:e}, derived tolerance {kernel_tolerance:e})",
                closed.value,
                kernel.value
            );
            assert!(
                derivative_discrepancy <= derivative_tolerance,
                "exact GELU ∂_r K at v = w = −r = {variance}: closed form {} against the degenerate law {} (discrepancy {derivative_discrepancy:e}, derived tolerance {derivative_tolerance:e})",
                closed.covariance_derivative,
                derivative.value
            );
            // Positive control: the orthant as arccos(−ρ) of a rounded ρ. Every
            // rounded ρ lies within one ulp of the true one, and one of the two
            // ulp neighbours of the computed ρ lies at least half an ulp from it,
            // so the larger of their discrepancies shows what the arccos form
            // costs; the same tolerance rejects it.
            let total = 1.0 + variance;
            let discriminant = 1.0 + 2.0 * variance;
            let orthant = discriminant.sqrt().atan2(-covariance) / TAU;
            let correlation = covariance / (total.sqrt() * total.sqrt());
            let control = [correlation.next_up(), correlation.next_down()]
                .into_iter()
                .map(|neighbour| {
                    let arccos_orthant = (-neighbour).acos() / TAU;
                    (closed.value + covariance * (arccos_orthant - orthant) - kernel.value).abs()
                })
                .fold(0.0, f64::max);
            assert!(
                control > kernel_tolerance,
                "arccos control at v = {variance}: discrepancy {control:e} within tolerance {kernel_tolerance:e}"
            );
        }
    }

    #[test]
    fn pair_kernel_covariance_derivative_matches_a_richardson_checked_difference() {
        // Central differences D(h) = K_r + c h² + O(h⁴): |D(h) − D(h/2)| bounds
        // the truncation of D(h/2), and each kernel's rounding enters divided by
        // the step. The step is a design choice; the tolerance comes from the
        // halved-step check.
        let laws = [(1.0, 1.0, 0.3), (9.0, 4.0, -3.0), (0.25, 4.0, 0.8), (16.0, 9.0, 11.0)];
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for (variance_x, variance_y, covariance) in laws {
                let kernel = |value: f64| {
                    pair_kernel(activation, zero_mean(variance_x, variance_y, value))
                        .expect("zero-mean pair kernel")
                };
                let scale = variance_x.sqrt() * variance_y.sqrt();
                let step = 1.0e-3 * scale;
                let difference =
                    |h: f64| (kernel(covariance + h).value - kernel(covariance - h).value) / (2.0 * h);
                let coarse = difference(step);
                let half_step = 0.5 * step;
                let fine = difference(half_step);
                let reach = covariance.abs() + step;
                let magnitude = match activation {
                    GaussianActivation::Relu => scale / TAU + 0.5 * reach,
                    GaussianActivation::ExactGelu => {
                        exact_gelu_zero_mean_magnitudes(variance_x, variance_y, reach).0
                    }
                    GaussianActivation::Silu => unreachable!("SiLU has no closed form here"),
                };
                let tolerance = (coarse - fine).abs()
                    + closed_form_rounding(magnitude) / half_step
                    + fine.abs() * f64::EPSILON * reach / half_step;
                let closed = kernel(covariance).covariance_derivative;
                let discrepancy = (closed - fine).abs();
                assert!(
                    discrepancy <= tolerance,
                    "{activation:?} ∂_r K at v = {variance_x}, w = {variance_y}, r = {covariance}: closed form {closed} against difference {fine} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})"
                );
                // Positive control: the orthant term alone, with Price's coupling
                // (GELU) or the complementary angle (ReLU) wrong, is rejected.
                let correlation = covariance / scale;
                let wrong = match activation {
                    GaussianActivation::Relu => correlation.acos() / TAU,
                    GaussianActivation::ExactGelu => {
                        (-covariance / ((1.0 + variance_x) * (1.0 + variance_y)).sqrt()).acos()
                            / TAU
                    }
                    GaussianActivation::Silu => unreachable!("SiLU has no closed form here"),
                };
                assert!(
                    (wrong - fine).abs() > tolerance,
                    "{activation:?} Richardson control at r = {covariance}: within tolerance {tolerance:e}"
                );
            }
        }
    }

    #[test]
    fn covariance_past_cauchy_schwarz_projects_within_its_rounding_and_is_refused_beyond() {
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for sign in [1.0, -1.0] {
                let boundary = pair_kernel(activation, zero_mean(4.0, 9.0, sign * 6.0))
                    .expect("boundary covariance");
                // Past √(vw) = 6 by 6e-9, inside a stated rounding of 1e-8.
                let within = PreactivationPair {
                    covariance_rounding: 1.0e-8,
                    ..zero_mean(4.0, 9.0, sign * 6.0 * (1.0 + 1.0e-9))
                };
                assert_eq!(
                    pair_kernel(activation, within),
                    Ok(boundary),
                    "{activation:?} projection at sign {sign}"
                );
                // Past √(vw) by 6e-6: beyond that rounding, and beyond any rounding
                // for an exact law.
                for rounding in [1.0e-8, 0.0] {
                    let beyond = PreactivationPair {
                        covariance_rounding: rounding,
                        ..zero_mean(4.0, 9.0, sign * 6.0 * (1.0 + 1.0e-6))
                    };
                    assert!(
                        matches!(
                            pair_kernel(activation, beyond),
                            Err(GaussianActivationError::CovarianceOutsideCauchySchwarz { .. })
                        ),
                        "{activation:?} covariance beyond its rounding at sign {sign}, rounding {rounding}"
                    );
                }
            }
            // A constant pre-activation: K = σ(0) E σ(Y) = 0 and ∂_r K = σ'(0) E σ'(Y) = ¼.
            let moments = |kernel: PairKernel| (kernel.value, kernel.covariance_derivative);
            assert_eq!(
                pair_kernel(activation, zero_mean(0.0, 2.0, 0.0)).map(moments),
                Ok((0.0, 0.25)),
                "{activation:?} with a zero variance"
            );
            let rounded = PreactivationPair {
                covariance_rounding: 0.2,
                ..zero_mean(0.0, 2.0, 0.1)
            };
            assert_eq!(
                pair_kernel(activation, rounded).map(moments),
                Ok((0.0, 0.25)),
                "{activation:?} with a zero variance and a covariance inside its rounding"
            );
            assert!(matches!(
                pair_kernel(activation, zero_mean(0.0, 2.0, 0.1)),
                Err(GaussianActivationError::CovarianceOutsideCauchySchwarz { .. })
            ));
        }
        assert_eq!(
            pair_kernel(GaussianActivation::Relu, zero_mean(4.0, 9.0, 6.0))
                .map(|kernel| (kernel.value, kernel.covariance_derivative)),
            Ok((3.0, 0.5))
        );
    }

    #[test]
    fn hidden_act_tags_name_their_activation_or_are_refused() {
        for (tag, activation) in [
            ("relu", GaussianActivation::Relu),
            ("gelu", GaussianActivation::ExactGelu),
            ("silu", GaussianActivation::Silu),
            ("swish", GaussianActivation::Silu),
        ] {
            assert_eq!(GaussianActivation::from_hidden_act(tag), Ok(activation), "tag {tag}");
        }
        for tag in ["gelu_new", "gelu_pytorch_tanh"] {
            assert_eq!(
                GaussianActivation::from_hidden_act(tag),
                Err(GaussianActivationError::ApproximateGelu {
                    tag: tag.to_owned()
                })
            );
        }
        for tag in ["tanh", "GELU", ""] {
            assert_eq!(
                GaussianActivation::from_hidden_act(tag),
                Err(GaussianActivationError::UnsupportedHiddenAct {
                    tag: tag.to_owned()
                })
            );
        }
    }

    #[test]
    fn hermite_coefficients_are_the_scaled_smoothing_derivatives() {
        // a_n = sⁿ T⁽ⁿ⁾_{s²}σ(t)/√(n!) against the derivative towers the heat-equation,
        // derivative-chain and quadrature pins already validate. The coefficient
        // carries its returned bound, and the derivative route rounds within
        // EVALUATION_OPERATIONS ulps of its term magnitudes.
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for (t, scale) in [(-2.0, 0.5), (0.7, 2.0), (3.0, 1.0), (-0.4, 0.1)] {
                let mut coefficients = [f64::NAN; 9];
                let mut bounds = [f64::NAN; 9];
                gaussian_hermite_coefficients(activation, t, scale, &mut coefficients, &mut bounds)
                    .expect("valid coefficient law");
                let variance = scale * scale;
                let derivatives = smoothing(activation, t, variance, 9);
                for order in 0..9 {
                    let normalizer = hermite_normalizer(scale, order);
                    let expected = normalizer * derivatives[order];
                    let tolerance = bounds[order]
                        + closed_form_rounding(
                            normalizer * smoothing_magnitude(activation, t, variance, order),
                        );
                    let discrepancy = (coefficients[order] - expected).abs();
                    assert!(
                        discrepancy <= tolerance,
                        "{activation:?} a_{order} at t = {t}, s = {scale}: {} against sⁿT⁽ⁿ⁾/√n! = {expected} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                        coefficients[order]
                    );
                }
            }
        }
    }

    #[test]
    fn hermite_coefficients_reproduce_the_zero_mean_pair_kernel_by_mehler() {
        // E[σ(X)σ(Y)] = Σ_n ρⁿ a_n(0, √v) a_n(0, √w). By Cauchy-Schwarz the terms after
        // the first N add up to at most |ρ|^N √(E σ(X)² E σ(Y)²), and E σ(X)² is the
        // diagonal kernel K(v, v; v).
        const TERMS: usize = 64;
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            let laws: [(f64, f64, f64); 3] = [(1.0, 1.0, 0.3), (4.0, 0.25, -0.5), (9.0, 16.0, 7.0)];
            for (variance_x, variance_y, covariance) in laws {
                let mut first = [f64::NAN; TERMS];
                let mut first_bounds = [f64::NAN; TERMS];
                let mut second = [f64::NAN; TERMS];
                let mut second_bounds = [f64::NAN; TERMS];
                gaussian_hermite_coefficients(
                    activation,
                    0.0,
                    variance_x.sqrt(),
                    &mut first,
                    &mut first_bounds,
                )
                .expect("first coefficients");
                gaussian_hermite_coefficients(
                    activation,
                    0.0,
                    variance_y.sqrt(),
                    &mut second,
                    &mut second_bounds,
                )
                .expect("second coefficients");
                let correlation = covariance / (variance_x.sqrt() * variance_y.sqrt());
                let mut power = 1.0;
                let mut series = 0.0;
                let mut absolute = 0.0;
                let mut reflected = 0.0;
                let mut propagated = 0.0;
                for order in 0..TERMS {
                    let term = power * first[order] * second[order];
                    series += term;
                    absolute += term.abs();
                    reflected += if order % 2 == 0 { term } else { -term };
                    propagated += power.abs()
                        * (first[order].abs() * second_bounds[order]
                            + second[order].abs() * first_bounds[order]);
                    power *= correlation;
                }
                let diagonal = |variance: f64| {
                    pair_kernel(activation, zero_mean(variance, variance, variance))
                        .expect("diagonal kernel")
                        .value
                };
                let tail = correlation.abs().powi(TERMS as i32)
                    * (diagonal(variance_x) * diagonal(variance_y)).sqrt();
                let closed = pair_kernel(
                    activation,
                    zero_mean(variance_x, variance_y, covariance),
                )
                .expect("zero-mean pair kernel");
                let magnitude = match activation {
                    GaussianActivation::Relu => {
                        (variance_x * variance_y - covariance * covariance).max(0.0).sqrt() / TAU
                            + covariance.abs() * closed.covariance_derivative
                    }
                    GaussianActivation::ExactGelu => {
                        exact_gelu_zero_mean_magnitudes(variance_x, variance_y, covariance).0
                    }
                    GaussianActivation::Silu => unreachable!("SiLU has no closed form here"),
                };
                let tolerance = tail
                    + propagated
                    + (TERMS as f64 + EVALUATION_OPERATIONS) * f64::EPSILON * absolute
                    + closed_form_rounding(magnitude);
                let discrepancy = (closed.value - series).abs();
                assert!(
                    discrepancy <= tolerance,
                    "{activation:?} Mehler series at v = {variance_x}, w = {variance_y}, r = {covariance}: {series} against K = {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                    closed.value
                );
                // Positive control: the series at −ρ is the kernel of the reflected law,
                // and the same tolerance rejects it.
                let control = (closed.value - reflected).abs();
                assert!(
                    control > tolerance,
                    "{activation:?} Mehler control at r = {covariance}: discrepancy {control:e} within {tolerance:e}"
                );
            }
        }
    }

    #[test]
    fn exact_gelu_hermite_coefficients_match_gauss_hermite_projections() {
        const DEGREE: usize = 12;
        let coarse_rule = gauss_hermite_rule(256).expect("256-node Gauss-Hermite rule");
        let fine_rule = gauss_hermite_rule(512).expect("512-node Gauss-Hermite rule");
        for (t, scale) in [(-3.0, 2.0), (1.5, 0.5), (0.0, 3.0), (-6.0, 1.0)] {
            let mut coefficients = [f64::NAN; DEGREE + 1];
            let mut bounds = [f64::NAN; DEGREE + 1];
            gaussian_hermite_coefficients(
                GaussianActivation::ExactGelu,
                t,
                scale,
                &mut coefficients,
                &mut bounds,
            )
            .expect("exact GELU coefficients");
            for order in 0..=DEGREE {
                let integrand = |e: f64| {
                    let (activation, activation_magnitude) = exact_gelu_derivative(t + scale * e, 0);
                    let (hermite, hermite_magnitude) = normalized_hermite(order, e)[order];
                    (activation * hermite, activation_magnitude * hermite_magnitude)
                };
                let reference = doubled_order(
                    hermite_pass(&coarse_rule, integrand),
                    hermite_pass(&fine_rule, integrand),
                    512,
                    512,
                    0.0,
                );
                let tolerance = reference.bound + bounds[order];
                let discrepancy = (coefficients[order] - reference.value).abs();
                assert!(
                    discrepancy <= tolerance,
                    "exact GELU a_{order} at t = {t}, s = {scale}: {} against E[σ h_n] = {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                    coefficients[order],
                    reference.value
                );
            }
        }
    }

    #[test]
    fn hermite_coefficients_stay_finite_where_raw_derivatives_overflow_and_vanish_without_scale() {
        // At s = 1e-150 and u = t/s = 0.1 the raw T⁽ⁿ⁾ = (−1)ⁿ He_{n−2}(u) φ(u)/sⁿ⁻¹
        // overflows by n = 4, while a_n = (−1)ⁿ s φ(u) h_{n−2}(u)/√(n(n−1)) has size s.
        let scale = 1.0e-150;
        let t = 1.0e-151;
        let mut raw = [f64::NAN; 8];
        gaussian_smoothing_derivatives(GaussianActivation::Relu, t, scale * scale, &mut raw)
            .expect("raw derivatives");
        assert!(raw[4].is_infinite(), "positive control: raw T⁽⁴⁾ = {} should overflow", raw[4]);
        let mut coefficients = [f64::NAN; 8];
        let mut bounds = [f64::NAN; 8];
        gaussian_hermite_coefficients(
            GaussianActivation::Relu,
            t,
            scale,
            &mut coefficients,
            &mut bounds,
        )
        .expect("coefficients at a tiny scale");
        assert!(
            coefficients
                .iter()
                .chain(bounds.iter())
                .all(|entry| entry.is_finite()),
            "coefficients at s = {scale}: {coefficients:?} with bounds {bounds:?}"
        );
        let expected = scale * normal_pdf(t / scale) / SQRT_2;
        assert!(
            (coefficients[2] - expected).abs() <= bounds[2] + closed_form_rounding(expected),
            "a_2 at s = {scale}: {} against s φ(u)/√2 = {expected}",
            coefficients[2]
        );
        // Without scale the smoothed activation is the constant σ(t), and the
        // coefficients past the first are exact zeros.
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            let mut constant = [f64::NAN; 4];
            let mut constant_bounds = [f64::NAN; 4];
            gaussian_hermite_coefficients(activation, -1.5, 0.0, &mut constant, &mut constant_bounds)
                .expect("coefficients without scale");
            let value = smoothing(activation, -1.5, 0.0, 1)[0];
            assert_eq!(constant, [value, 0.0, 0.0, 0.0], "{activation:?} without scale");
            assert_eq!(
                constant_bounds[1..],
                [0.0; 3],
                "{activation:?} bounds without scale"
            );
        }
        let mut refused = [0.0; 2];
        let mut refused_bounds = [0.0; 2];
        assert_eq!(
            gaussian_hermite_coefficients(
                GaussianActivation::Relu,
                0.0,
                -1.0,
                &mut refused,
                &mut refused_bounds
            ),
            Err(GaussianActivationError::InvalidScale { scale: -1.0 })
        );
        assert!(matches!(
            gaussian_hermite_coefficients(
                GaussianActivation::ExactGelu,
                f64::NAN,
                1.0,
                &mut refused,
                &mut refused_bounds
            ),
            Err(GaussianActivationError::NonFiniteArgument { .. })
        ));
        assert_eq!(
            gaussian_hermite_coefficients(
                GaussianActivation::Silu,
                0.0,
                1.0,
                &mut refused,
                &mut refused_bounds
            ),
            Err(GaussianActivationError::NoClosedForm {
                activation: GaussianActivation::Silu
            })
        );
        let mut short_bounds = [0.0; 1];
        assert_eq!(
            gaussian_hermite_coefficients(
                GaussianActivation::Relu,
                0.0,
                1.0,
                &mut refused,
                &mut short_bounds
            ),
            Err(GaussianActivationError::MismatchedBoundsLength {
                coefficients: 2,
                bounds: 1
            })
        );
    }

    #[test]
    fn hermite_recurrence_running_error_bounds_its_double_double_reference() {
        // The production recurrence against the same recurrence carried in
        // double-double at the same computed argument, whose own error is O(ε²)
        // and far below the rounding the running bound accounts for.
        let mut largest_discrepancy = 0.0_f64;
        for argument in [-6.3, -1.7, 0.0, 0.9, 4.2, 11.5] {
            let mut recurrence = HermiteRecurrence::start(argument);
            let x = DoubleDouble::from(argument);
            let mut previous = DoubleDouble::from(0.0);
            let mut current = DoubleDouble::from(1.0);
            for degree in 0..64 {
                let discrepancy = ((recurrence.current - current.high) - current.low).abs();
                largest_discrepancy = largest_discrepancy.max(discrepancy);
                assert!(
                    discrepancy <= recurrence.current_error,
                    "h_{degree}({argument}): discrepancy {discrepancy:e} beyond its running bound {:e}",
                    recurrence.current_error
                );
                let root = DoubleDouble::root_of(degree as f64);
                let next_root = DoubleDouble::root_of((degree + 1) as f64);
                let next = x
                    .mul(current)
                    .add(root.mul(previous).negated())
                    .div(next_root);
                previous = current;
                current = next;
                recurrence.advance();
            }
        }
        // Positive control: the propagated part alone is zero from h_0 = 1, so the
        // bound is only as good as its rounding terms, and some step does round.
        assert!(
            largest_discrepancy > 0.0,
            "the recurrence never rounded, so the running bound was not exercised"
        );
    }

    #[test]
    fn underflowed_density_returns_zero_coefficients_with_cramer_bounds() {
        // Past |x| ≈ 38.6 the density underflows. By Cramér's inequality
        // φ(x) |h_j(x)| ≤ 1.0865 (2π)^{−1/4} √φ(x), which bounds what the zero
        // coefficients lose. Just inside the edge the same inequality, at that φ,
        // bounds the representable coefficients.
        let cramer = |density: f64| CRAMER_BOUND * TAU.sqrt().sqrt().recip() * density.sqrt();
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            let root_total = match activation {
                GaussianActivation::Relu => 1.0,
                GaussianActivation::ExactGelu => SQRT_2,
                GaussianActivation::Silu => unreachable!("SiLU has no closed form here"),
            };
            let mut coefficients = [f64::NAN; 12];
            let mut bounds = [f64::NAN; 12];
            gaussian_hermite_coefficients(activation, 45.0 * root_total, 1.0, &mut coefficients, &mut bounds)
                .expect("coefficients past the underflow edge");
            assert!(
                coefficients[2..].iter().all(|coefficient| *coefficient == 0.0),
                "{activation:?} coefficients past the edge: {coefficients:?}"
            );
            assert!(
                bounds[2..].iter().all(|bound| bound.is_finite() && *bound > 0.0),
                "{activation:?} Cramér bounds past the edge: {bounds:?}"
            );
            let inside = 38.0;
            let mut representable = [f64::NAN; 12];
            let mut representable_bounds = [f64::NAN; 12];
            gaussian_hermite_coefficients(
                activation,
                inside * root_total,
                1.0,
                &mut representable,
                &mut representable_bounds,
            )
            .expect("coefficients inside the edge");
            let density = normal_pdf(inside);
            assert!(density > 0.0, "positive control: φ({inside}) is representable");
            for order in 2..12 {
                let normalizer = ((order * (order - 1)) as f64).sqrt();
                let envelope = match activation {
                    GaussianActivation::Relu => cramer(density) / normalizer,
                    GaussianActivation::ExactGelu => {
                        // (s/S)ⁿ/S (A/√(n(n−1)) + 1) with s = 1, A = 2, S = √2.
                        SQRT_2.recip().powi(order as i32) / SQRT_2
                            * (2.0 / normalizer + 1.0)
                            * cramer(density)
                    }
                    GaussianActivation::Silu => unreachable!("SiLU has no closed form here"),
                };
                assert!(
                    representable[order].abs() <= envelope + representable_bounds[order],
                    "{activation:?} a_{order} at x = {inside}: {} above Cramér's envelope {envelope:e}",
                    representable[order]
                );
            }
        }
    }

    #[test]
    fn exact_gelu_biased_pair_kernel_and_price_derivative_match_gauss_hermite() {
        let moderate = (
            gauss_hermite_rule(256).expect("256-node Gauss-Hermite rule"),
            gauss_hermite_rule(512).expect("512-node Gauss-Hermite rule"),
        );
        let wide = (
            gauss_hermite_rule(1024).expect("1024-node Gauss-Hermite rule"),
            gauss_hermite_rule(2048).expect("2048-node Gauss-Hermite rule"),
        );
        // (b, c, v, w, r): moderate laws, the lower tails, a vanishing and a zero
        // variance, |r| → √(vw), and a large-norm law with ρ_G near one.
        let laws: [(f64, f64, f64, f64, f64, &(GaussHermiteRule, GaussHermiteRule)); 8] = [
            (0.5, -1.0, 1.0, 1.0, 0.3, &moderate),
            (-3.0, -2.5, 0.5, 2.0, -0.6, &moderate),
            (2.0, 3.0, 4.0, 4.0, 3.9, &moderate),
            (-4.0, -3.5, 1.0, 1.0, 0.8, &moderate),
            (1.0, -0.5, 1.0e-12, 1.0, 0.5e-6, &moderate),
            (0.3, 0.2, 0.0, 2.0, 0.0, &moderate),
            (-1.0, 2.0, 9.0, 4.0, 6.0 * (1.0 - 1.0e-12), &moderate),
            (-10.0, 5.0, 64.0, 64.0, 63.9, &wide),
        ];
        for (law_index, (mean_x, mean_y, variance_x, variance_y, covariance, rules)) in
            laws.into_iter().enumerate()
        {
            let pair = PreactivationPair {
                mean_x,
                mean_y,
                variance_x,
                variance_y,
                covariance,
                covariance_rounding: 0.0,
            };
            let closed = pair_kernel(GaussianActivation::ExactGelu, pair)
                .expect("biased exact GELU pair kernel");
            let (coarse_kernel, coarse_derivative) =
                exact_gelu_pair_passes(&rules.0, mean_x, mean_y, variance_x, variance_y, covariance);
            let (fine_kernel, fine_derivative) =
                exact_gelu_pair_passes(&rules.1, mean_x, mean_y, variance_x, variance_y, covariance);
            let nodes = rules.1.nodes.len();
            let kernel = doubled_order(coarse_kernel, fine_kernel, 2 * nodes, 2 * nodes, 0.0);
            let derivative =
                doubled_order(coarse_derivative, fine_derivative, 2 * nodes, 2 * nodes, 0.0);
            let (kernel_magnitude, derivative_magnitude) = biased_magnitudes(
                GaussianActivation::ExactGelu,
                mean_x,
                mean_y,
                variance_x,
                variance_y,
                covariance,
            );
            let kernel_tolerance = kernel.bound
                + closed_form_rounding(kernel_magnitude)
                + (mean_x * mean_y + covariance).abs() * BIVARIATE_NORMAL_CONTRACT;
            let derivative_tolerance = derivative.bound
                + closed_form_rounding(derivative_magnitude)
                + BIVARIATE_NORMAL_CONTRACT;
            let kernel_discrepancy = (closed.value - kernel.value).abs();
            let derivative_discrepancy = (closed.covariance_derivative - derivative.value).abs();
            assert!(
                kernel_discrepancy <= kernel_tolerance,
                "exact GELU K at b = {mean_x}, c = {mean_y}, v = {variance_x}, w = {variance_y}, r = {covariance}: closed form {} against reference {} (discrepancy {kernel_discrepancy:e}, derived tolerance {kernel_tolerance:e})",
                closed.value,
                kernel.value
            );
            assert!(
                derivative_discrepancy <= derivative_tolerance,
                "exact GELU ∂_r K at b = {mean_x}, c = {mean_y}, v = {variance_x}, w = {variance_y}, r = {covariance}: closed form {} against E[σ'σ'] {} (discrepancy {derivative_discrepancy:e}, derived tolerance {derivative_tolerance:e})",
                closed.covariance_derivative,
                derivative.value
            );
            if law_index == 0 {
                // Positive controls: the kernel without the r²(1/A + 1/B) part of its
                // coupling, and the derivative without (r/Δ + b̃c̃) H_bc, are rejected.
                let total_x = 1.0 + variance_x;
                let total_y = 1.0 + variance_y;
                let discriminant = total_x * total_y - covariance * covariance;
                let (h, k) = (mean_x / total_x.sqrt(), mean_y / total_y.sqrt());
                let correlation = covariance / (total_x * total_y).sqrt();
                let (orthant_unused, partial_h, partial_k, d_rho) = standardized_orthant(
                    h,
                    k,
                    correlation,
                    discriminant / (total_x * total_y),
                );
                assert!(orthant_unused > 0.0 && partial_h > 0.0 && partial_k > 0.0);
                let density = d_rho / (total_x * total_y).sqrt();
                let reduced = (total_y * mean_x - covariance * mean_y)
                    * (total_x * mean_y - covariance * mean_x)
                    / (discriminant * discriminant);
                let wrong_kernel = closed.value
                    - covariance * covariance * (1.0 / total_x + 1.0 / total_y) * density;
                let wrong_derivative = closed.covariance_derivative
                    - (covariance / discriminant + reduced) * density;
                assert!(
                    (wrong_kernel - kernel.value).abs() > kernel_tolerance,
                    "biased exact GELU kernel control within tolerance {kernel_tolerance:e}"
                );
                assert!(
                    (wrong_derivative - derivative.value).abs() > derivative_tolerance,
                    "biased exact GELU derivative control within tolerance {derivative_tolerance:e}"
                );
            }
        }
    }

    #[test]
    fn biased_pair_kernels_reproduce_the_mehler_series_of_their_hermite_coefficients() {
        // E[σ(b + √v E₁) σ(c + √w E₂)] = Σ_n ρⁿ a_n(b, √v) a_n(c, √w). The terms after
        // the first N add up to at most |ρ|^N √(E σ(X)² E σ(Y)²), and E σ(X)² is the
        // diagonal kernel K(b, b; v, v; v).
        const TERMS: usize = 64;
        let laws: [(f64, f64, f64, f64, f64); 4] = [
            (0.5, -1.0, 1.0, 1.0, 0.3),
            (-2.0, 1.0, 1.0, 4.0, -1.2),
            (1.5, 2.0, 2.0, 3.0, 1.4),
            (-3.0, -2.0, 1.0, 1.0, 0.5),
        ];
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for (mean_x, mean_y, variance_x, variance_y, covariance) in laws {
                let mut first = [f64::NAN; TERMS];
                let mut first_bounds = [f64::NAN; TERMS];
                let mut second = [f64::NAN; TERMS];
                let mut second_bounds = [f64::NAN; TERMS];
                gaussian_hermite_coefficients(
                    activation,
                    mean_x,
                    variance_x.sqrt(),
                    &mut first,
                    &mut first_bounds,
                )
                .expect("first coefficients");
                gaussian_hermite_coefficients(
                    activation,
                    mean_y,
                    variance_y.sqrt(),
                    &mut second,
                    &mut second_bounds,
                )
                .expect("second coefficients");
                let correlation = covariance / (variance_x.sqrt() * variance_y.sqrt());
                let mut power = 1.0_f64;
                let mut series = 0.0_f64;
                let mut absolute = 0.0_f64;
                let mut reflected = 0.0_f64;
                let mut propagated = 0.0_f64;
                for order in 0..TERMS {
                    let term = power * first[order] * second[order];
                    series += term;
                    absolute += term.abs();
                    reflected += if order % 2 == 0 { term } else { -term };
                    propagated += power.abs()
                        * (first[order].abs() * second_bounds[order]
                            + second[order].abs() * first_bounds[order]);
                    power *= correlation;
                }
                let diagonal = |mean: f64, variance: f64| {
                    pair_kernel(
                        activation,
                        PreactivationPair {
                            mean_x: mean,
                            mean_y: mean,
                            variance_x: variance,
                            variance_y: variance,
                            covariance: variance,
                            covariance_rounding: 0.0,
                        },
                    )
                    .expect("diagonal kernel")
                    .value
                };
                let tail = correlation.abs().powi(TERMS as i32)
                    * (diagonal(mean_x, variance_x) * diagonal(mean_y, variance_y)).sqrt();
                let closed = pair_kernel(
                    activation,
                    PreactivationPair {
                        mean_x,
                        mean_y,
                        variance_x,
                        variance_y,
                        covariance,
                        covariance_rounding: 0.0,
                    },
                )
                .expect("biased pair kernel");
                let magnitude =
                    biased_magnitudes(activation, mean_x, mean_y, variance_x, variance_y, covariance).0;
                let tolerance = tail
                    + propagated
                    + (TERMS as f64 + EVALUATION_OPERATIONS) * f64::EPSILON * absolute
                    + closed_form_rounding(magnitude)
                    + (mean_x * mean_y + covariance).abs() * BIVARIATE_NORMAL_CONTRACT;
                let discrepancy = (closed.value - series).abs();
                assert!(
                    discrepancy <= tolerance,
                    "{activation:?} Mehler series at b = {mean_x}, c = {mean_y}, v = {variance_x}, w = {variance_y}, r = {covariance}: {series} against K = {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                    closed.value
                );
                // Positive control: the series at −ρ is rejected.
                let control = (closed.value - reflected).abs();
                assert!(
                    control > tolerance,
                    "{activation:?} biased Mehler control at r = {covariance}: discrepancy {control:e} within {tolerance:e}"
                );
            }
        }
    }

    #[test]
    fn relu_biased_pair_kernel_matches_the_degenerate_law_at_the_correlation_limits() {
        // (b, c, v, w, sign of r = ±√(vw)).
        let laws: [(f64, f64, f64, f64, f64); 3] = [
            (0.5, -0.3, 1.0, 4.0, 1.0),
            (0.5, 0.8, 1.0, 4.0, -1.0),
            (-1.0, 2.0, 2.0, 1.0, 1.0),
        ];
        for (mean_x, mean_y, variance_x, variance_y, sign) in laws {
            let root_x = variance_x.sqrt();
            let scale = root_x * variance_y.sqrt();
            let boundary = PreactivationPair {
                mean_x,
                mean_y,
                variance_x,
                variance_y,
                covariance: sign * scale,
                covariance_rounding: 0.0,
            };
            let closed = pair_kernel(GaussianActivation::Relu, boundary)
                .expect("biased ReLU kernel on the degenerate law");
            let reference =
                relu_degenerate_pair_reference(mean_x, mean_y, variance_x, variance_y, sign, 64);
            let magnitude = biased_magnitudes(
                GaussianActivation::Relu,
                mean_x,
                mean_y,
                variance_x,
                variance_y,
                sign * scale,
            )
            .0;
            let contract = (mean_x * mean_y + sign * scale).abs() * BIVARIATE_NORMAL_CONTRACT;
            let tolerance = reference.bound + closed_form_rounding(magnitude) + contract;
            let discrepancy = (closed.value - reference.value).abs();
            assert!(
                discrepancy <= tolerance,
                "ReLU K on the degenerate law b = {mean_x}, c = {mean_y}, v = {variance_x}, w = {variance_y}, sign {sign}: closed form {} against reference {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                closed.value,
                reference.value
            );
            // ∂_r K = P(X > 0, Y > 0), with Y > 0 on one side of x** = b − c/κ.
            let slope = sign * (variance_y / variance_x).sqrt();
            let crossing = mean_x - mean_y / slope;
            let expected = if slope > 0.0 {
                normal_cdf((mean_x - crossing.max(0.0)) / root_x)
            } else if crossing > 0.0 {
                (normal_cdf((crossing - mean_x) / root_x) - normal_cdf(-mean_x / root_x)).max(0.0)
            } else {
                0.0
            };
            assert!(
                (closed.covariance_derivative - expected).abs()
                    <= closed_form_rounding(1.0) + BIVARIATE_NORMAL_CONTRACT,
                "ReLU ∂_r K on the degenerate law: {} against P(X > 0, Y > 0) = {expected}",
                closed.covariance_derivative
            );
            // Continuity toward the boundary: |∂_r K| = H ≤ 1.
            let inside = PreactivationPair {
                covariance: sign * scale * (1.0 - 1.0e-6),
                ..boundary
            };
            let near = pair_kernel(GaussianActivation::Relu, inside)
                .expect("biased ReLU kernel inside the boundary");
            let continuity =
                scale * 1.0e-6 + 2.0 * (closed_form_rounding(magnitude) + contract);
            assert!(
                (near.value - closed.value).abs() <= continuity,
                "ReLU K jumps at the correlation limit: {} inside against {} on the boundary",
                near.value,
                closed.value
            );
        }
    }

    #[test]
    fn biased_pair_kernel_covariance_derivative_matches_a_richardson_checked_difference() {
        let laws: [(f64, f64, f64, f64, f64); 3] = [
            (0.5, -1.0, 1.0, 1.0, 0.3),
            (-2.0, 1.0, 1.0, 4.0, -1.2),
            (1.5, 2.0, 2.0, 3.0, 1.4),
        ];
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for (mean_x, mean_y, variance_x, variance_y, covariance) in laws {
                let kernel = |value: f64| {
                    pair_kernel(
                        activation,
                        PreactivationPair {
                            mean_x,
                            mean_y,
                            variance_x,
                            variance_y,
                            covariance: value,
                            covariance_rounding: 0.0,
                        },
                    )
                    .expect("biased pair kernel")
                };
                let scale = variance_x.sqrt() * variance_y.sqrt();
                let step = 1.0e-3 * scale;
                let difference = |h: f64| {
                    (kernel(covariance + h).value - kernel(covariance - h).value) / (2.0 * h)
                };
                let coarse = difference(step);
                let half_step = 0.5 * step;
                let fine = difference(half_step);
                let reach = covariance.abs() + step;
                let magnitude =
                    biased_magnitudes(activation, mean_x, mean_y, variance_x, variance_y, covariance).0;
                let evaluation = closed_form_rounding(magnitude)
                    + ((mean_x * mean_y).abs() + reach) * BIVARIATE_NORMAL_CONTRACT;
                let tolerance = (coarse - fine).abs()
                    + evaluation / half_step
                    + fine.abs() * f64::EPSILON * reach / half_step;
                let closed = kernel(covariance).covariance_derivative;
                let discrepancy = (closed - fine).abs();
                assert!(
                    discrepancy <= tolerance,
                    "{activation:?} biased ∂_r K at b = {mean_x}, c = {mean_y}, v = {variance_x}, w = {variance_y}, r = {covariance}: closed form {closed} against difference {fine} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})"
                );
                // Positive control: the independence orthant Φ(h)Φ(k) for ReLU, and the
                // orthant alone without Price's partial terms for the exact GELU.
                let wrong = match activation {
                    GaussianActivation::Relu => {
                        normal_cdf(mean_x / variance_x.sqrt()) * normal_cdf(mean_y / variance_y.sqrt())
                    }
                    GaussianActivation::ExactGelu => {
                        let total_x = 1.0 + variance_x;
                        let total_y = 1.0 + variance_y;
                        standardized_orthant(
                            mean_x / total_x.sqrt(),
                            mean_y / total_y.sqrt(),
                            covariance / (total_x * total_y).sqrt(),
                            (total_x * total_y - covariance * covariance) / (total_x * total_y),
                        )
                        .0
                    }
                    GaussianActivation::Silu => unreachable!("SiLU has no closed form here"),
                };
                assert!(
                    (wrong - fine).abs() > tolerance,
                    "{activation:?} biased Richardson control at r = {covariance}: within tolerance {tolerance:e}"
                );
            }
        }
    }

    #[test]
    fn constant_preactivations_factor_the_biased_kernels() {
        // A zero variance makes X the constant b: K = σ(b) T_w σ(c) and
        // ∂_r K = σ'(b) T_w σ'(c), with ReLU's Gaussian-limit slope ½ at the kink.
        let moments = smoothing(GaussianActivation::Relu, -0.3, 2.0, 2);
        for (constant, slope) in [(0.7, 1.0), (-0.5, 0.0), (0.0, 0.5)] {
            let expected = (f64::max(constant, 0.0) * moments[0], slope * moments[1]);
            let constant_x = PreactivationPair {
                mean_x: constant,
                mean_y: -0.3,
                variance_x: 0.0,
                variance_y: 2.0,
                covariance: 0.0,
                covariance_rounding: 0.0,
            };
            assert_eq!(
                pair_kernel(GaussianActivation::Relu, constant_x)
                    .map(|kernel| (kernel.value, kernel.covariance_derivative)),
                Ok(expected),
                "ReLU with X = {constant}"
            );
            let constant_y = PreactivationPair {
                mean_x: -0.3,
                mean_y: constant,
                variance_x: 2.0,
                variance_y: 0.0,
                ..constant_x
            };
            assert_eq!(
                pair_kernel(GaussianActivation::Relu, constant_y)
                    .map(|kernel| (kernel.value, kernel.covariance_derivative)),
                Ok(expected),
                "ReLU with Y = {constant}"
            );
        }
        let gelu_moments = smoothing(GaussianActivation::ExactGelu, 0.4, 2.0, 2);
        for constant in [0.7, -1.2] {
            let closed = pair_kernel(
                GaussianActivation::ExactGelu,
                PreactivationPair {
                    mean_x: constant,
                    mean_y: 0.4,
                    variance_x: 0.0,
                    variance_y: 2.0,
                    covariance: 0.0,
                    covariance_rounding: 0.0,
                },
            )
            .expect("exact GELU with a constant pre-activation");
            let activation = constant * normal_cdf(constant);
            let slope = normal_cdf(constant) + constant * normal_pdf(constant);
            let (magnitude, derivative_magnitude) =
                biased_magnitudes(GaussianActivation::ExactGelu, constant, 0.4, 0.0, 2.0, 0.0);
            let kernel_tolerance = closed_form_rounding(
                magnitude
                    + activation.abs()
                        * smoothing_magnitude(GaussianActivation::ExactGelu, 0.4, 2.0, 0),
            ) + (0.4 * constant).abs() * BIVARIATE_NORMAL_CONTRACT;
            let derivative_tolerance = closed_form_rounding(
                derivative_magnitude
                    + slope.abs() * smoothing_magnitude(GaussianActivation::ExactGelu, 0.4, 2.0, 1),
            ) + BIVARIATE_NORMAL_CONTRACT;
            assert!(
                (closed.value - activation * gelu_moments[0]).abs() <= kernel_tolerance,
                "exact GELU K with X = {constant}: {} against σ(b) T_w σ(c) = {}",
                closed.value,
                activation * gelu_moments[0]
            );
            assert!(
                (closed.covariance_derivative - slope * gelu_moments[1]).abs() <= derivative_tolerance,
                "exact GELU ∂_r K with X = {constant}: {} against σ'(b) T_w σ'(c) = {}",
                closed.covariance_derivative,
                slope * gelu_moments[1]
            );
        }
    }

    #[test]
    fn left_tail_and_coefficient_bounds_hold_against_a_double_double_reference() {
        // The a_0/a_1 bounds rest on derived owner contracts: normal_pdf's 5u, erfc's
        // ulp, erfcx below 5e-16, and the log-CDF owner's λ and q. This pins them
        // against double-double evaluations at the same computed arguments, where φ
        // is not subnormal. Mills' continued fraction is checked at doubled depth,
        // and |R_N − R_2N| bounds the reference's truncation.
        const DEPTH: usize = 4096;
        let mut largest_discrepancy = 0.0_f64;
        let left_arguments: [f64; 9] = [-37.0, -30.0, -17.0, -8.0, -4.1, -4.0, -3.9, -1.7, -0.5];
        for argument in left_arguments {
            let magnitude = -argument;
            let refined = double_double_mills_ratio(magnitude, 2 * DEPTH);
            let truncation = double_double_mills_ratio(magnitude, DEPTH)
                .add(refined.negated())
                .high
                .abs();
            let correction =
                DoubleDouble::from(1.0).add(DoubleDouble::from(magnitude).mul(refined).negated());
            let density = double_double_normal_pdf(argument);
            let (computed_density, density_bound) = normal_pdf_bounded(argument);
            let density_error = double_double_discrepancy(computed_density, density);
            assert!(
                density_error <= density_bound,
                "φ({argument}): error {density_error:e} beyond its bound {density_bound:e}"
            );
            let ratios = normal_left_tail_ratios(argument, 0.0)
                .expect("left-tail ratios at a finite negative argument");
            let reciprocal_error = double_double_discrepancy(ratios.cdf_over_density, refined);
            let corrected_error =
                double_double_discrepancy(ratios.positive_part_over_density, correction);
            assert!(
                reciprocal_error <= ratios.cdf_over_density_rounding + truncation,
                "1/λ({argument}): error {reciprocal_error:e} beyond its bound {:e}",
                ratios.cdf_over_density_rounding
            );
            assert!(
                corrected_error <= ratios.positive_part_over_density_rounding + magnitude * truncation,
                "q/λ({argument}): error {corrected_error:e} beyond its bound {:e}",
                ratios.positive_part_over_density_rounding
            );
            // Resolution control: the reference's own truncation lies below each certified bound, so an error at
            // the bound's scale would be visible here.
            assert!(
                truncation < ratios.cdf_over_density_rounding
                    && magnitude * truncation < ratios.positive_part_over_density_rounding,
                "argument {argument}: reference truncation {truncation:e} does not resolve the bounds {:e}, {:e}",
                ratios.cdf_over_density_rounding,
                ratios.positive_part_over_density_rounding
            );
            // ReLU at s = 1: a_0 = φ q/λ and a_1 = φ/λ. Exact GELU at s = 0: a_0 = φ (q/λ − 1).
            let mut coefficients = [f64::NAN; 2];
            let mut bounds = [f64::NAN; 2];
            gaussian_hermite_coefficients(
                GaussianActivation::Relu,
                argument,
                1.0,
                &mut coefficients,
                &mut bounds,
            )
            .expect("ReLU coefficients in the left tail");
            let value_error = double_double_discrepancy(coefficients[0], density.mul(correction));
            let slope_error = double_double_discrepancy(coefficients[1], density.mul(refined));
            assert!(
                value_error <= bounds[0] + density.high * magnitude * truncation,
                "ReLU a_0({argument}): error {value_error:e} beyond its bound {:e}",
                bounds[0]
            );
            assert!(
                slope_error <= bounds[1] + density.high * truncation,
                "ReLU a_1({argument}): error {slope_error:e} beyond its bound {:e}",
                bounds[1]
            );
            let mut gelu = [f64::NAN; 1];
            let mut gelu_bounds = [f64::NAN; 1];
            gaussian_hermite_coefficients(
                GaussianActivation::ExactGelu,
                argument,
                0.0,
                &mut gelu,
                &mut gelu_bounds,
            )
            .expect("exact GELU coefficient in the left tail");
            let gelu_error = double_double_discrepancy(
                gelu[0],
                density.mul(correction.add(DoubleDouble::from(-1.0))),
            );
            assert!(
                gelu_error <= gelu_bounds[0] + density.high * magnitude * truncation,
                "exact GELU a_0({argument}): error {gelu_error:e} beyond its bound {:e}",
                gelu_bounds[0]
            );
            largest_discrepancy = largest_discrepancy
                .max(density_error)
                .max(reciprocal_error)
                .max(corrected_error)
                .max(value_error)
                .max(slope_error)
                .max(gelu_error);
        }
        // Right of the origin Φ = 1 − φ R, and ReLU's a_0 = t Φ + φ, a_1 = Φ.
        let right_arguments: [f64; 4] = [0.5, 1.3, 3.0, 6.0];
        for argument in right_arguments {
            let refined = double_double_mills_ratio(argument, 2 * DEPTH);
            let truncation = double_double_mills_ratio(argument, DEPTH)
                .add(refined.negated())
                .high
                .abs();
            let density = double_double_normal_pdf(argument);
            let probability = DoubleDouble::from(1.0).add(density.mul(refined).negated());
            let bounded = bounded_normal_cdf(Bounded::exact(argument));
            let probability_error = double_double_discrepancy(bounded.value, probability);
            assert!(
                probability_error <= bounded.bound + density.high * truncation,
                "Φ({argument}): error {probability_error:e} beyond its bound {:e}",
                bounded.bound
            );
            let mut coefficients = [f64::NAN; 2];
            let mut bounds = [f64::NAN; 2];
            gaussian_hermite_coefficients(
                GaussianActivation::Relu,
                argument,
                1.0,
                &mut coefficients,
                &mut bounds,
            )
            .expect("ReLU coefficients right of the origin");
            let value_error = double_double_discrepancy(
                coefficients[0],
                DoubleDouble::from(argument).mul(probability).add(density),
            );
            let slope_error = double_double_discrepancy(coefficients[1], probability);
            assert!(
                value_error <= bounds[0] + argument * density.high * truncation,
                "ReLU a_0({argument}): error {value_error:e} beyond its bound {:e}",
                bounds[0]
            );
            assert!(
                slope_error <= bounds[1] + density.high * truncation,
                "ReLU a_1({argument}): error {slope_error:e} beyond its bound {:e}",
                bounds[1]
            );
            largest_discrepancy = largest_discrepancy
                .max(probability_error)
                .max(value_error)
                .max(slope_error);
        }
        // Positive control: at exact arguments the propagated part of every bound is
        // zero, so the bounds are only as good as their rounding terms, and some
        // evaluation does round.
        assert!(
            largest_discrepancy > 0.0,
            "no evaluation rounded, so the bounds were not exercised"
        );
    }

    /// `(E[X₊ Y₊], P(X > 0, Y > 0))` for the ReLU from the conditional law of `Y`
    /// given `X = b + √v e`: `Y ~ N(m(e), τ²)` with `m = c + ρ√w e` and
    /// `τ² = w (1 − ρ²)`. With `e = ℓ + y` and `ℓ = −b/√v`,
    /// `K = √v ∫₀^∞ y φ(ℓ + y) T(m) dy` and `∂_r K = ∫₀^∞ φ(ℓ + y) T'(m) dy`, where `T`
    /// is the ReLU smoothed at variance `τ²`. Both integrands are positive and
    /// analytic, so the rule rounds relative to the kernel itself, not to the
    /// closed form's cancelling terms, and never calls `Φ₂`.
    /// Discarded mass beyond the reach `Y`:
    /// - `φ(ℓ + y) = φ(ℓ) e^{−ℓy − y²/2}`;
    /// - `T` and `T'` are log-concave in `m`, so `g(m(ℓ + y)) ≤ g(m₀) e^{κy}` with
    ///   `m₀ = m(ℓ)` and `κ = ρ√w g'(m₀)/g(m₀)`;
    /// - with `γ = ℓ − κ`, `λ = γ + Y > 0` and `y²/2 ≥ Y²/2 + Y (y − Y)`, the mass is
    ///   at most `φ(ℓ) g(m₀) e^{−γY − Y²/2}` times `√v (Y/λ + 1/λ²)` for `K` and `1/λ`
    ///   for `∂_r K`.
    fn relu_conditional_pair_reference(
        mean_x: f64,
        mean_y: f64,
        variance_x: f64,
        variance_y: f64,
        covariance: f64,
        node_count: usize,
    ) -> (Reference, Reference) {
        // The reach chooses the domain, not the tolerance: γY + Y²/2 equals this
        // exponent, and the discarded mass joins each bound.
        const EXPONENT_REACH: f64 = 45.0;
        let root_x = variance_x.sqrt();
        let slope = covariance / root_x;
        let conditional_variance = (variance_x * variance_y - covariance * covariance) / variance_x;
        let lower = -mean_x / root_x;
        let smoothed =
            |e: f64| smoothing(GaussianActivation::Relu, mean_y + slope * e, conditional_variance, 3);
        let at_lower = smoothed(lower);
        let coarse_rule = legendre_rule(node_count);
        let fine_rule = legendre_rule(2 * node_count);
        let reference = |order: usize, power: i32| {
            let value = at_lower[order];
            let exponent = lower - slope * at_lower[order + 1] / value;
            let reach = -exponent + (exponent * exponent + 2.0 * EXPONENT_REACH).sqrt();
            let decay = exponent + reach;
            let factor = root_x.powi(power);
            let polynomial = if power == 1 {
                reach / decay + 1.0 / (decay * decay)
            } else {
                1.0 / decay
            };
            let truncation =
                factor * normal_pdf(lower) * value * (-EXPONENT_REACH).exp() * polynomial;
            let integrand = |y: f64| {
                let e = lower + y;
                let summand = factor * y.powi(power) * normal_pdf(e) * smoothed(e)[order];
                (summand, summand.abs())
            };
            doubled_order(
                legendre_pass(&coarse_rule, 0.0, reach, integrand),
                legendre_pass(&fine_rule, 0.0, reach, integrand),
                2 * node_count,
                2 * node_count,
                truncation,
            )
        };
        (reference(0, 1), reference(1, 0))
    }

    #[test]
    fn biased_pair_kernels_keep_relative_accuracy_in_anticorrelated_lower_tails() {
        // (b, c, v, w, r) inside the bivariate normal owner's relative regime for both activations: ρ ≤ 0, with both
        // reduced apex coordinates −(h − ρk)/(1 − ρ²) and −(k − ρh)/(1 − ρ²) nonnegative. Here the absolute contract
        // alone left K and ∂_r K with no certified digit (#2946).
        let laws: [(f64, f64, f64, f64, f64); 7] = [
            (-3.0, -3.0, 1.0, 1.0, -0.9),
            (-3.0, -3.0, 1.0, 1.0, -0.5),
            (-5.0, -5.0, 1.0, 1.0, -0.9),
            (-5.0, -5.0, 1.0, 1.0, -0.5),
            (-8.0, -8.0, 1.0, 1.0, -0.9),
            (-8.0, -8.0, 1.0, 1.0, -0.5),
            (-6.0, 1.0, 1.0, 1.0, -0.8),
        ];
        let wide = (
            gauss_hermite_rule(1024).expect("1024-node Gauss-Hermite rule"),
            gauss_hermite_rule(2048).expect("2048-node Gauss-Hermite rule"),
        );
        // Laws where the absolute contract alone certifies no digit of K, per activation. The fix is only evidenced
        // there; the other laws are regression cells.
        let mut relu_controlled = 0_usize;
        let mut gelu_controlled = 0_usize;
        for (mean_x, mean_y, variance_x, variance_y, covariance) in laws {
            let pair = PreactivationPair {
                mean_x,
                mean_y,
                variance_x,
                variance_y,
                covariance,
                covariance_rounding: 0.0,
            };
            // The owner certifies relative accuracy at both standardized cells: the ReLU's (b/√v, c/√w, r/√(vw)) and
            // the exact GELU's (b/√A, c/√B, r/√(AB)). An Absolute contract would leave these pins with no digit.
            let (total_x, total_y) = (1.0 + variance_x, 1.0 + variance_y);
            for (h, k, correlation, complement) in [
                (
                    mean_x / variance_x.sqrt(),
                    mean_y / variance_y.sqrt(),
                    covariance / (variance_x * variance_y).sqrt(),
                    (variance_x * variance_y - covariance * covariance) / (variance_x * variance_y),
                ),
                (
                    mean_x / total_x.sqrt(),
                    mean_y / total_y.sqrt(),
                    covariance / (total_x * total_y).sqrt(),
                    (total_x * total_y - covariance * covariance) / (total_x * total_y),
                ),
            ] {
                let orthant = bivariate_normal_cdf_with_complement_bounded(h, k, correlation, complement)
                    .expect("bounded standardized orthant");
                assert!(
                    matches!(orthant.contract, RoundingContract::Relative),
                    "Φ₂ at (h, k, ρ) = ({h}, {k}, {correlation}) from b = {mean_x}, c = {mean_y}, r = {covariance} is not certified relative: {:?}",
                    orthant.contract
                );
            }
            let (relu_kernel, relu_derivative) =
                relu_conditional_pair_reference(mean_x, mean_y, variance_x, variance_y, covariance, 64);
            let (coarse_kernel, coarse_derivative) =
                exact_gelu_pair_passes(&wide.0, mean_x, mean_y, variance_x, variance_y, covariance);
            let (fine_kernel, fine_derivative) =
                exact_gelu_pair_passes(&wide.1, mean_x, mean_y, variance_x, variance_y, covariance);
            let nodes = wide.1.nodes.len();
            let gelu_kernel = doubled_order(coarse_kernel, fine_kernel, 2 * nodes, 2 * nodes, 0.0);
            let gelu_derivative =
                doubled_order(coarse_derivative, fine_derivative, 2 * nodes, 2 * nodes, 0.0);
            for (activation, kernel, derivative) in [
                (GaussianActivation::Relu, relu_kernel, relu_derivative),
                (GaussianActivation::ExactGelu, gelu_kernel, gelu_derivative),
            ] {
                let closed = pair_kernel(activation, pair).expect("biased pair kernel in the lower tail");
                for (name, value, rounding, reference) in [
                    ("K", closed.value, closed.value_rounding, kernel),
                    ("∂_r K", closed.covariance_derivative, closed.covariance_derivative_rounding, derivative),
                ] {
                    // The reference resolves its own value, and the pin resolves the kernel to a certified digit.
                    assert!(
                        reference.bound < reference.value.abs(),
                        "{activation:?} {name} reference at b = {mean_x}, c = {mean_y}, r = {covariance}: {} ± {:e}",
                        reference.value,
                        reference.bound
                    );
                    let tolerance = reference.bound + rounding;
                    let discrepancy = (value - reference.value).abs();
                    assert!(
                        discrepancy <= tolerance && tolerance < reference.value.abs(),
                        "{activation:?} {name} at b = {mean_x}, c = {mean_y}, v = {variance_x}, w = {variance_y}, r = {covariance}: {value} against {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                        reference.value
                    );
                }
                if activation == GaussianActivation::Relu {
                    // E[X₊ Y₊] ≥ 0 and P(X > 0, Y > 0) > 0 on every nondegenerate law.
                    assert!(closed.value > 0.0 && closed.covariance_derivative > 0.0);
                }
                // Positive control, per law: where the absolute contract alone, (bc + r)·BIVARIATE_NORMAL_CONTRACT,
                // exceeds K, the plain route certified no digit, so the digit certified above is the fallback's.
                if (mean_x * mean_y + covariance).abs() * BIVARIATE_NORMAL_CONTRACT > kernel.value.abs() {
                    assert!(
                        closed.orthant_fallback,
                        "{activation:?} at b = {mean_x}, c = {mean_y}, r = {covariance}: the plain contract certifies no digit of K, yet the certified entry did not run"
                    );
                    if activation == GaussianActivation::Relu {
                        relu_controlled += 1;
                    } else {
                        gelu_controlled += 1;
                    }
                }
            }
        }
        assert!(
            relu_controlled > 0 && gelu_controlled > 0,
            "no law escapes the absolute contract: ReLU {relu_controlled}, exact GELU {gelu_controlled}"
        );
        // The fallback is selective: a law the plain contract resolves keeps the plain route, and its cost.
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            let plain = pair_kernel(
                activation,
                PreactivationPair {
                    mean_x: 0.5,
                    mean_y: 0.3,
                    variance_x: 1.0,
                    variance_y: 1.0,
                    covariance: 0.2,
                    covariance_rounding: 0.0,
                },
            )
            .expect("biased pair kernel at a central law");
            assert!(
                !plain.orthant_fallback && plain.value_rounding < plain.value.abs(),
                "{activation:?} at a central law took the certified entry: {plain:?}"
            );
        }
        // Positive control for the contract query: the positively correlated standardized cell (−3, −3, 0.5) lies
        // outside the owner's certified region and must answer Absolute.
        let outside = bivariate_normal_cdf_with_complement_bounded(-3.0, -3.0, 0.5, 0.75)
            .expect("bounded orthant outside the certified region");
        assert!(
            matches!(outside.contract, RoundingContract::Absolute),
            "Φ₂ at (−3, −3, 0.5) claims {:?} outside the certified region",
            outside.contract
        );
    }

    #[test]
    fn variance_partials_match_richardson_checked_differences_of_the_kernel() {
        // ∂_v K at fixed means and covariance against central differences of pair_kernel in v, the step then halved.
        // Once the difference converges quadratically, |coarse − fine| bounds the finer one's truncation, and each
        // kernel's rounding enters divided by 2h. Biased, zero-mean and strongly correlated laws, both activations,
        // both variances.
        let laws: [(f64, f64, f64, f64, f64); 4] = [
            (0.4, -0.3, 1.0, 1.5, 0.5),
            (0.0, 0.0, 1.0, 2.0, -0.6),
            (-1.2, 0.8, 0.5, 0.7, 0.1),
            (1.5, 1.0, 2.0, 1.0, 1.2),
        ];
        let difference = |activation: GaussianActivation, pair: PreactivationPair, along_x: bool, step: f64| {
            let shifted = |sign: f64| {
                let moved = if along_x {
                    PreactivationPair { variance_x: pair.variance_x + sign * step, ..pair }
                } else {
                    PreactivationPair { variance_y: pair.variance_y + sign * step, ..pair }
                };
                pair_kernel(activation, moved).expect("pair kernel near the law")
            };
            let (up, down) = (shifted(1.0), shifted(-1.0));
            (
                (up.value - down.value) / (2.0 * step),
                (up.value_rounding + down.value_rounding) / (2.0 * step),
            )
        };
        let mut compared = 0_usize;
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for (mean_x, mean_y, variance_x, variance_y, covariance) in laws {
                let pair = PreactivationPair {
                    mean_x,
                    mean_y,
                    variance_x,
                    variance_y,
                    covariance,
                    covariance_rounding: 0.0,
                };
                let partials =
                    pair_kernel_variance_partials(activation, pair).expect("variance partials at a smooth law");
                for (along_x, variance, partial, rounding) in [
                    (true, variance_x, partials.variance_x, partials.variance_x_rounding),
                    (false, variance_y, partials.variance_y, partials.variance_y_rounding),
                ] {
                    let step = 1.0e-3 * variance;
                    let (coarse, _) = difference(activation, pair, along_x, step);
                    let (fine, fine_rounding) = difference(activation, pair, along_x, 0.5 * step);
                    let tolerance = (coarse - fine).abs() + fine_rounding + rounding;
                    assert!(
                        (partial - fine).abs() <= tolerance && tolerance < 1.0e-4 * fine.abs().max(1.0e-3),
                        "{activation:?} ∂K/∂v_{} at b = {mean_x}, c = {mean_y}, v = {variance_x}, w = {variance_y}, r = {covariance}: {partial} against the difference {fine} (tolerance {tolerance:e})",
                        if along_x { "x" } else { "y" }
                    );
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, 16);
        // Positive controls at the first law, where r ≠ 0 and the means are nonzero:
        // - exact GELU without the Stein cross term −2(b/A)(r/A)·T′;
        // - ReLU with the unconditional mean c in place of the regression mean c − r·b/v.
        // Each misses the difference, so the check above resolves both terms.
        let (mean_x, mean_y, variance_x, variance_y, covariance) = laws[0];
        let pair = PreactivationPair {
            mean_x,
            mean_y,
            variance_x,
            variance_y,
            covariance,
            covariance_rounding: 0.0,
        };
        let total = 1.0 + variance_x;
        let tilted_mean = mean_y - covariance * mean_x / total;
        let tilted_variance = (variance_y + (variance_x * variance_y - covariance * covariance)) / total;
        let slope = smoothing(GaussianActivation::ExactGelu, tilted_mean, tilted_variance, 2)[1];
        let gelu = pair_kernel_variance_partials(GaussianActivation::ExactGelu, pair).expect("GELU partials");
        let without_cross = gelu.variance_x
            + normal_pdf(mean_x / total.sqrt()) / total.sqrt() * (mean_x / total) * (covariance / total) * slope;
        let relu = pair_kernel_variance_partials(GaussianActivation::Relu, pair).expect("ReLU partials");
        let unconditional = 0.5 * normal_pdf(mean_x / variance_x.sqrt()) / variance_x.sqrt()
            * smoothing(
                GaussianActivation::Relu,
                mean_y,
                (variance_x * variance_y - covariance * covariance) / variance_x,
                1,
            )[0];
        for (activation, control, partial) in [
            (GaussianActivation::ExactGelu, without_cross, gelu.variance_x),
            (GaussianActivation::Relu, unconditional, relu.variance_x),
        ] {
            let step = 1.0e-3 * variance_x;
            let (coarse, _) = difference(activation, pair, true, step);
            let (fine, fine_rounding) = difference(activation, pair, true, 0.5 * step);
            let tolerance = (coarse - fine).abs() + fine_rounding;
            assert!(
                (control - fine).abs() > tolerance && (partial - fine).abs() < (control - fine).abs(),
                "{activation:?}: the control {control} is not rejected against {fine} (tolerance {tolerance:e})"
            );
        }
    }

    #[test]
    fn derived_arctangent_encloses_the_double_double_reference_in_every_quadrant_case() {
        // z = y/|x| (or |x|/y past the diagonal) on a grid of 1/32, which crosses every center k/8
        // and both edges k/8 ± 1/16 of every reduction, for both signs of x. Plus a vanishing
        // height, a vanishing abscissa of either sign, the diagonal, and ratios far from one. The
        // reference is the test-local double-double atan2.
        let mut cases: Vec<(f64, f64)> = vec![
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (1.0, -0.0),
            (0.7, 0.7),
            (3.0, -3.0),
            (1.0e-100, 1.0e10),
            (1.0e10, -1.0e-100),
        ];
        for step in 0..=32_u32 {
            let ratio = f64::from(step) / 32.0;
            for abscissa in [1.0_f64, -1.0, 7.5, -0.3] {
                cases.push((ratio * abscissa.abs(), abscissa));
                if step > 0 {
                    cases.push((abscissa.abs() / ratio, abscissa));
                }
            }
        }
        let mut largest_discrepancy = 0.0_f64;
        for (height, abscissa) in cases {
            let angle = bounded_arctangent2(height, abscissa);
            let reference = double_double_atan2(DoubleDouble::from(height), abscissa);
            let error = double_double_discrepancy(angle.value, reference);
            assert!(
                error <= angle.bound,
                "atan2({height}, {abscissa}) = {}: error {error:e} beyond its bound {:e}",
                angle.value,
                angle.bound
            );
            // The bound stays a few rounding units, so the enclosure is not won by width.
            assert!(
                angle.bound <= 16.0 * UNIT_ROUNDOFF * angle.value + 4.0 * f64::from_bits(1),
                "atan2({height}, {abscissa}) = {}: bound {:e} is not within 16u",
                angle.value,
                angle.bound
            );
            largest_discrepancy = largest_discrepancy.max(error);
        }
        assert!(
            largest_discrepancy > 0.0,
            "no evaluation rounded, so the bounds were not exercised"
        );
        // Positive control at a reduction edge: z = 1/16 reduces about c = 1/8 to |w| ≈ 1/16. The
        // same center with the odd series cut to three terms, 1 − s/3 + s²/5, misses the reference
        // by about |w|⁷/7, far beyond the certified bound there.
        let (height, abscissa) = (1.0, 16.0);
        let angle = bounded_arctangent2(height, abscissa);
        let reference = double_double_atan2(DoubleDouble::from(height), abscissa);
        let table = &*ARCTANGENT_TABLE;
        let argument = height / abscissa;
        let center = (argument * ARCTANGENT_CENTERS_PER_UNIT).round() / ARCTANGENT_CENTERS_PER_UNIT;
        let reduced = (argument - center) / (argument * center + 1.0);
        let square = reduced * reduced;
        let short = table.centers[1] + reduced * (1.0 - square / 3.0 + square * square / 5.0);
        assert!(
            center == 0.125 && double_double_discrepancy(short, reference) > angle.bound,
            "the three-term series {short} at center {center} still lies within {:e} of the reference",
            angle.bound
        );
    }

    #[test]
    fn zero_mean_pair_kernel_rounding_bounds_hold_against_a_double_double_reference() {
        // The same closed forms carried in double-double at the same float law
        // (the residual vw − r² from exact products, atan2 by half-angle reduction and
        // series), so the discrepancy is the production rounding alone.
        let pi = DoubleDouble {
            high: PI,
            low: 1.224_646_799_147_353_2e-16,
        };
        let two_pi = pi.scaled(2.0);
        let one = DoubleDouble::from(1.0);
        let relu_laws: [(f64, f64, f64); 5] = [
            (1.0, 1.0, 0.3),
            (0.25, 4.0, -0.9),
            (9.0, 4.0, 5.994),
            (4.0, 4.0, -4.0 * (1.0 - 1.0e-6)),
            (16.0, 9.0, -6.0),
        ];
        let gelu_laws: [(f64, f64, f64); 5] = [
            (1.0, 1.0, 0.3),
            (9.0, 4.0, 5.994),
            (64.0, 64.0, 63.9),
            (4096.0, 4096.0, -4096.0),
            (1.0e8, 1.0e8, -1.0e8),
        ];
        let mut largest_discrepancy = 0.0_f64;
        for (activation, laws) in [
            (GaussianActivation::Relu, relu_laws),
            (GaussianActivation::ExactGelu, gelu_laws),
        ] {
            for (variance_x, variance_y, covariance) in laws {
                let closed = pair_kernel(activation, zero_mean(variance_x, variance_y, covariance))
                    .expect("zero-mean pair kernel");
                let residual = DoubleDouble::from(variance_x)
                    .mul(DoubleDouble::from(variance_y))
                    .add(DoubleDouble::from(covariance).mul(DoubleDouble::from(covariance)).negated());
                let (reference_value, reference_derivative) = match activation {
                    GaussianActivation::Relu => {
                        let root = if residual.high > 0.0 {
                            residual.sqrt()
                        } else {
                            DoubleDouble::from(0.0)
                        };
                        let angle = double_double_atan2(root, -covariance);
                        (
                            root.add(DoubleDouble::from(covariance).mul(angle)).div(two_pi),
                            angle.div(two_pi),
                        )
                    }
                    GaussianActivation::ExactGelu => {
                        let discriminant = one
                            .add(DoubleDouble::from(variance_x))
                            .add(DoubleDouble::from(variance_y))
                            .add(residual);
                        let root = discriminant.sqrt();
                        let angle = double_double_atan2(root, -covariance);
                        let density = one.div(two_pi.mul(root));
                        let inverse = one
                            .div(one.add(DoubleDouble::from(variance_x)))
                            .add(one.div(one.add(DoubleDouble::from(variance_y))));
                        let coupling = residual.add(
                            DoubleDouble::from(covariance)
                                .mul(DoubleDouble::from(covariance))
                                .mul(inverse),
                        );
                        (
                            DoubleDouble::from(covariance)
                                .mul(angle)
                                .div(two_pi)
                                .add(coupling.mul(density)),
                            angle.div(two_pi).add(
                                DoubleDouble::from(covariance)
                                    .mul(density)
                                    .mul(inverse.add(one.div(discriminant))),
                            ),
                        )
                    }
                    GaussianActivation::Silu => unreachable!("SiLU has no closed form here"),
                };
                let value_error = double_double_discrepancy(closed.value, reference_value);
                let derivative_error =
                    double_double_discrepancy(closed.covariance_derivative, reference_derivative);
                assert!(
                    value_error <= closed.value_rounding,
                    "{activation:?} K at v = {variance_x}, w = {variance_y}, r = {covariance}: error {value_error:e} beyond its bound {:e}",
                    closed.value_rounding
                );
                assert!(
                    derivative_error <= closed.covariance_derivative_rounding,
                    "{activation:?} ∂_r K at v = {variance_x}, w = {variance_y}, r = {covariance}: error {derivative_error:e} beyond its bound {:e}",
                    closed.covariance_derivative_rounding
                );
                largest_discrepancy = largest_discrepancy.max(value_error).max(derivative_error);
            }
        }
        // Positive control: at exact float laws the bounds are only as good as their
        // rounding terms, and some evaluation does round.
        assert!(
            largest_discrepancy > 0.0,
            "no evaluation rounded, so the bounds were not exercised"
        );
    }

    #[test]
    fn slope_bound_squared_covers_the_slope_and_is_tight_at_its_extreme() {
        assert_eq!(GaussianActivation::Relu.slope_bound_squared(), Ok(1.0));
        assert!(matches!(
            GaussianActivation::Silu.slope_bound_squared(),
            Err(GaussianActivationError::NoClosedForm { .. })
        ));
        let bound = GaussianActivation::ExactGelu
            .slope_bound_squared()
            .expect("exact GELU slope bound");
        // σ'' = (2 − x²) φ vanishes at √2, where σ' peaks.
        let at_peak = smoothing(GaussianActivation::ExactGelu, SQRT_2, 0.0, 3);
        assert!(
            at_peak[2].abs()
                <= closed_form_rounding(smoothing_magnitude(
                    GaussianActivation::ExactGelu,
                    SQRT_2,
                    0.0,
                    2
                )),
            "σ''(√2) = {} does not vanish",
            at_peak[2]
        );
        let peak_square = at_peak[1] * at_peak[1];
        assert!(
            bound >= peak_square && bound - peak_square <= closed_form_rounding(bound),
            "slope bound {bound} is not σ'(√2)² = {peak_square} to rounding"
        );
        // The bound covers σ'(x)² on a dense sample of [−20, 20], through both extremes.
        let mut largest = 0.0_f64;
        let mut index = -4000_i32;
        while index <= 4000 {
            let slope = smoothing(GaussianActivation::ExactGelu, f64::from(index) * 0.005, 0.0, 2)[1];
            largest = largest.max(slope * slope);
            index += 1;
        }
        assert!(
            largest <= bound,
            "sampled sup σ'² = {largest} exceeds the bound {bound}"
        );
        // Positive control: Φ(√2)² without the √2 φ(√2) term is exceeded by the sample.
        let incomplete = normal_cdf(SQRT_2).powi(2);
        assert!(
            largest > incomplete,
            "control: sampled {largest} does not exceed Φ(√2)² = {incomplete}"
        );
    }

    #[test]
    fn invalid_laws_are_refused_with_typed_errors() {
        let mut derivatives = [0.0; 2];
        assert_eq!(
            gaussian_smoothing_derivatives(GaussianActivation::ExactGelu, 0.0, -1.0, &mut derivatives),
            Err(GaussianActivationError::InvalidVariance { variance: -1.0 })
        );
        assert!(matches!(
            gaussian_smoothing_derivatives(GaussianActivation::Relu, 0.0, f64::NAN, &mut derivatives),
            Err(GaussianActivationError::InvalidVariance { .. })
        ));
        assert_eq!(
            gaussian_smoothing_derivatives(GaussianActivation::Relu, f64::INFINITY, 1.0, &mut derivatives),
            Err(GaussianActivationError::NonFiniteArgument {
                value: f64::INFINITY
            })
        );
        assert_eq!(
            gaussian_smoothing_derivatives(GaussianActivation::Silu, 0.0, 1.0, &mut derivatives),
            Err(GaussianActivationError::NoClosedForm {
                activation: GaussianActivation::Silu
            })
        );
        assert_eq!(
            pair_kernel(GaussianActivation::Relu, zero_mean(1.0, f64::INFINITY, 0.0)),
            Err(GaussianActivationError::InvalidVariance {
                variance: f64::INFINITY
            })
        );
        assert_eq!(
            pair_kernel(GaussianActivation::Silu, zero_mean(1.0, 1.0, 0.2)),
            Err(GaussianActivationError::NoClosedForm {
                activation: GaussianActivation::Silu
            })
        );
        let unbounded = PreactivationPair {
            covariance_rounding: f64::NAN,
            ..zero_mean(1.0, 1.0, 0.2)
        };
        assert!(matches!(
            pair_kernel(GaussianActivation::Relu, unbounded),
            Err(GaussianActivationError::InvalidCovarianceRounding { .. })
        ));
        let unbiased = PreactivationPair {
            mean_x: f64::NAN,
            ..zero_mean(1.0, 1.0, 0.2)
        };
        assert!(matches!(
            pair_kernel(GaussianActivation::ExactGelu, unbiased),
            Err(GaussianActivationError::NonFiniteArgument { .. })
        ));
    }
}
