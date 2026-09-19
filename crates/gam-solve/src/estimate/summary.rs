#[derive(Clone, Debug)]
pub struct ParametricTermSummary {
    pub name: String,
    pub estimate: f64,
    /// The display covariance's standard error, or, for a coefficient that
    /// carries its own ridge prior (`penalized`), the sampling standard deviation
    /// of its estimate with that prior's term removed from the variance.
    pub std_error: Option<f64>,
    /// The coefficient carries its own REML ridge prior (a linear term's
    /// function-mass penalty), so `std_error` is its null sampling standard
    /// deviation, not its posterior one.
    pub penalized: bool,
    /// `estimate / std_error`, referred to Student-t on the fit's Wald residual
    /// degrees of freedom when the fit's scale is estimated and to N(0, 1) when
    /// it is known (`LikelihoodScaleMetadata::wald_scale_is_estimated`).
    pub statistic: Option<f64>,
    pub pvalue: Option<f64>,
    /// Why `pvalue` is absent. `None` exactly when `pvalue` is present.
    pub pvalue_unavailable: Option<ParametricPValueUnavailable>,
}

/// The joint Wald test of one parametric term: every coefficient the term owns
/// tested against zero together, as one anova-style row.
///
/// A factor with `L` levels is tested on its `L - 1` contrasts: the term row asks
/// whether the factor matters at all, which does not depend on the coding.
#[derive(Clone, Debug)]
pub struct ParametricTermTest {
    pub name: String,
    /// The number of coefficient directions tested, the numerator degrees of
    /// freedom: `L - 1` for a factor with `L` levels.
    pub df: usize,
    /// `W / df` referred to `F(df, residual_df)` when the fit's scale is
    /// estimated; the Wald `W` referred to `χ²_df` when it is known, where
    /// `W = bᵀ V⁻¹ b` over the term's coefficients and the null sampling
    /// covariance of their estimate (the display block, less the term's own
    /// ridge prior when it carries one).
    pub statistic: Option<f64>,
    pub pvalue: Option<f64>,
    pub pvalue_unavailable: Option<ParametricPValueUnavailable>,
}

/// Why a parametric coefficient or term reports no Wald p-value.
///
/// Each variant names the missing input or the property of the term that leaves
/// no valid reference distribution, so an absent p-value always says why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParametricPValueUnavailable {
    /// The fit carries no coefficient covariance of its display definition.
    NoCovariance,
    /// The scale is estimated, so the reference is Student-t or F on the
    /// residual degrees of freedom `n - edf`, and the fit has none left.
    NoResidualDegreesOfFreedom,
    /// The term's covariance block is not positive definite, so the Wald form
    /// `bᵀ V⁻¹ b` does not exist: the coefficients are not identified.
    SingularCovariance,
    /// The coefficient carries a bound or bounded geometry. The null `β = 0`
    /// can sit on that boundary, where the estimate cannot fall on both sides
    /// of the null and the normal reference is not the null law.
    BoundedCoefficient,
}

impl ParametricPValueUnavailable {
    /// Serialized label carried into the model payload and the Python surface.
    pub fn label(self) -> &'static str {
        match self {
            Self::NoCovariance => "no_covariance",
            Self::NoResidualDegreesOfFreedom => "no_residual_degrees_of_freedom",
            Self::SingularCovariance => "singular_covariance",
            Self::BoundedCoefficient => "bounded_coefficient",
        }
    }

    /// One-line explanation printed beside the summary table.
    pub fn explanation(self) -> &'static str {
        match self {
            Self::NoCovariance => "the fit carries no coefficient covariance; no p-value is reported",
            Self::NoResidualDegreesOfFreedom => {
                "the scale is estimated and n - edf leaves no residual degrees of freedom for \
                 the t or F reference; no p-value is reported"
            }
            Self::SingularCovariance => {
                "the term's coefficient covariance is not positive definite, so its \
                 coefficients are not identified; no p-value is reported"
            }
            Self::BoundedCoefficient => {
                "the coefficient is bounded, so the null can sit on the constraint boundary \
                 where the normal reference does not hold; no p-value is reported"
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct SmoothTermSummary {
    pub name: String,
    pub edf: f64,
    pub ref_df: f64,
    pub chi_sq: Option<f64>,
    /// The statistic `pvalue` is the tail of: `chi_sq` against `χ²_{ref_df}`
    /// when the fit's scale is known, `F = chi_sq/ref_df` when it is estimated.
    pub statistic: Option<f64>,
    pub pvalue: Option<f64>,
    pub continuous_order: Option<ContinuousSmoothnessOrder>,
    /// Issue #340: human-readable note describing an automatic B-spline
    /// basis-shrink performed at fit time when `n` was too small for the
    /// user's requested `(degree, num_internal_knots)`. `None` means no
    /// shrink occurred (or the term is not a B-spline 1D smooth).
    pub basis_note: Option<String>,
    /// #2901: the label a term's `edf` carries when a penalty block it spends is not
    /// rank-bound certified ([`crate::estimate::EdfRankBound`]). That block's trace is
    /// published raw, so `edf` may lie outside `[0, dim]` and is not clamped. `None`
    /// when every block of the term is certified, or the fit recorded no bounds.
    pub edf_rank_bound: Option<String>,
    /// The fitted smoothing parameters of the penalty blocks this term owns, in
    /// the fit's flat layout order. Empty for an unpenalized term.
    pub lambdas: Vec<f64>,
    /// Why `pvalue` is absent, when the absence is a refusal rather than a
    /// missing input. `None` whenever `pvalue` is present.
    pub pvalue_unavailable: Option<SmoothPValueUnavailable>,
}

/// Why a smooth term reports no significance p-value.
///
/// A reason, not a status: each variant names the property of the term that
/// leaves no valid reference distribution, so an absent p-value is never
/// confusable with a missing input or a term that was never tested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothPValueUnavailable {
    /// The term is shape-constrained (`shape=` monotone, convex, concave).
    ///
    /// Its coefficients live in a cone `δ ≥ 0`, and the null `f ≡ 0` is the
    /// cone's apex, so every coordinate of the null sits on the boundary.
    /// A `χ²` or spectral reference assumes the estimate can fall on either
    /// side of the null, which it cannot. The boundary-aware references do not
    /// apply either:
    ///
    /// - The chi-bar-square law (Silvapulle & Sen 2005; Meyer 2003) is the
    ///   null law of the cone-*projected* estimate, whose face is the active
    ///   set. The term's estimate is the truncated posterior mean, which lies
    ///   strictly inside the cone and has no active set.
    /// - Conditioning on the active set is unavailable for the same reason.
    /// - Both laws hold for a fixed, unpenalized cone. Here the penalty and its
    ///   REML-selected λ shrink every face together, and the mixture weights
    ///   move with them.
    ShapeConstrained,
    /// A random-effect term whose variance-component score test
    /// (`gam_terms::inference::random_effect_test`) could not be computed, with
    /// the test's own reason.
    RandomEffect(gam_terms::inference::random_effect_test::RandomEffectTestUnavailable),
    /// A random-effect term the fit carries no test record for: a model saved
    /// before the test existed, or a fit route that does not compute it.
    RandomEffectTestNotRecorded,
}

impl SmoothPValueUnavailable {
    /// Serialized label carried into the model payload and the Python surface.
    pub fn label(self) -> &'static str {
        match self {
            Self::ShapeConstrained => "shape_constrained",
            Self::RandomEffect(reason) => reason.label(),
            Self::RandomEffectTestNotRecorded => "random_effect_test_not_recorded",
        }
    }

    /// One-line explanation printed beside the summary table.
    pub fn explanation(self) -> &'static str {
        match self {
            Self::ShapeConstrained => {
                "shape-constrained: the null f = 0 is the apex of the constraint cone, so no \
                 chi-square, spectral or chi-bar-square reference is valid for the truncated \
                 posterior mean; no p-value is reported"
            }
            Self::RandomEffect(reason) => reason.explanation(),
            Self::RandomEffectTestNotRecorded => {
                "the fit carries no variance-component test for this random effect"
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContinuousSmoothnessOrderStatus {
    Ok,
    NonMaternRegime,
    FirstOrderLimit,
    IntrinsicLimit,
    UndefinedZeroLambda,
}

#[derive(Clone, Debug)]
pub struct ContinuousSmoothnessOrder {
    pub lambda0: f64,
    pub lambda1: f64,
    pub lambda2: f64,
    pub r_ratio: Option<f64>,
    pub nu: Option<f64>,
    pub kappa2: Option<f64>,
    pub status: ContinuousSmoothnessOrderStatus,
}

/// Convert optimizer-scale lambdas into physical lambdas for raw operator penalties.
///
/// Derivation:
///   We optimize with normalized penalties
///     sum_k lambda_tilde_k * S_tilde_k
///   where
///     S_tilde_k = (1 / c_k) * S_k.
///
///   Define physical lambdas by requiring operator equality:
///     sum_k lambda_k * S_k  ==  sum_k lambda_tilde_k * S_tilde_k
///                           ==  sum_k lambda_tilde_k * (1/c_k) * S_k
///                           ==  sum_k (lambda_tilde_k / c_k) * S_k.
///
///   Therefore, coefficient matching gives:
///     lambda_k = lambda_tilde_k / c_k.
///
/// This helper performs exactly that mapping and validates positivity/finite values.
fn unscale_to_physical_lambdas(
    lambda_tilde: [f64; 3],
    normalization_scale: [f64; 3],
) -> Option<[f64; 3]> {
    let mut out = [f64::NAN; 3];
    for k in 0..3 {
        let c = normalization_scale[k];
        if !(c.is_finite() && c > 0.0) {
            return None;
        }
        out[k] = lambda_tilde[k] / c;
    }
    Some(out)
}

// Continuous smoothness/order diagnostic from three operator penalties.
//
// Full derivation and implementation contract
// We assume one smooth term has exactly three operator penalties in term-local order:
//   S0 = mass, S1 = tension (|grad f|^2), S2 = stiffness ((Delta f)^2).
//
// 1) Unscaling (physical lambda from optimizer lambda)
// If penalties were normalized before optimization:
//   S_tilde_k = S_k / c_k
// and the optimizer fits lambda_tilde_k, then
//   lambda_tilde_k * (beta-mu)' S_tilde_k (beta-mu)
// = lambda_tilde_k * (beta-mu)' (S_k / c_k) (beta-mu)
// = (lambda_tilde_k / c_k) * (beta-mu)' S_k (beta-mu).
//
// Therefore physical lambdas are:
//   lambda_k = lambda_tilde_k / c_k,  k in {0,1,2}.
//
// 2) SPDE/binomial coefficient mapping
// If the fitted (lambda0,lambda1,lambda2) are interpreted as proportional to
//   a_m(kappa,nu) = C(nu,m) * kappa^(2*(nu-m)),  m=0,1,2,
// then
//   a0 = kappa^(2*nu)
//   a1 = nu * kappa^(2*nu-2)
//   a2 = nu*(nu-1)/2 * kappa^(2*nu-4)
//
// Ratios:
//   lambda0/lambda2 = a0/a2 = 2*kappa^4 / (nu*(nu-1))
//   lambda1/lambda2 = a1/a2 = 2*kappa^2 / (nu-1)
//
// Define:
//   R = lambda1^2 / (lambda0*lambda2).
// Then
//   R = a1^2/(a0*a2) = 2*nu/(nu-1).
// Solve for nu:
//   nu = R/(R-2), requiring R>2 for finite nu>1.
//
// And from lambda1/lambda2:
//   kappa^2 = ((nu-1)/2) * (lambda1/lambda2)
//           = lambda1 / ((R-2)*lambda2).
//
// 3) Boundary/discriminant interpretation
// Spectral polynomial in x=|omega|^2:
//   Q(x) = lambda0 + lambda1*x + lambda2*x^2.
//
// Perfect-square Matérn(2) form is:
//   Q(x) proportional to (kappa^2 + x)^2
// which implies:
//   lambda1^2 = 4*lambda0*lambda2  <=>  R = 4.
//
// Discriminant:
//   D = lambda1^2 - 4*lambda0*lambda2 = lambda0*lambda2*(R-4).
// Hence:
//   R < 4  => D < 0 => no real factorization into two real range terms
//            => flagged as NonMaternRegime.
//   R = 4  => exact boundary (perfect square) => treated as Matérn-compatible.
//
// 4) Degenerate limits and guards
// - If lambda0 or lambda2 is zero (or negative), the 3-term inversion would
//   divide by it: report the limit below when the remaining lambdas define one,
//   and UndefinedZeroLambda otherwise, without dividing by those terms.
// - Intrinsic limit (lambda0 = 0, with positive lambda1/lambda2):
//     R = lambda1^2/(lambda0*lambda2) -> +inf
//     nu = R/(R-2) -> 1+
//     kappa^2 = lambda1/((R-2)lambda2) -> 0+.
//   We expose this explicitly as IntrinsicLimit with nu=1 and kappa^2=0. A small
//   positive lambda0 is not a special case: the closed forms below approach the
//   same limit continuously.
// - If R - 2 lies inside R's own rounding band (R <= 2 included), nu = R/(R-2)
//   is undefined or has no correct digit; keep nu/kappa2 unset.
//
// Status policy in this implementation:
// - Ok:                R >= 4 and valid finite nu/kappa2.
// - NonMaternRegime:   R < 4; if additionally R > 2, we still report effective
//                      nu/kappa2 as diagnostics, but mark non-Matérn status.
// - IntrinsicLimit:    lambda0 is zero; report nu=1, kappa^2=0.
// - UndefinedZeroLambda: invalid scaling/lambda inputs or unstable inversion.
pub(crate) fn compute_continuous_smoothness_order(
    lambda_tilde: [f64; 3],
    normalization_scale: [f64; 3],
) -> ContinuousSmoothnessOrder {
    let Some(lambda) = unscale_to_physical_lambdas(lambda_tilde, normalization_scale) else {
        return ContinuousSmoothnessOrder {
            lambda0: f64::NAN,
            lambda1: f64::NAN,
            lambda2: f64::NAN,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    };
    let [lambda0, lambda1, lambda2] = lambda;
    if !lambda0.is_finite() || !lambda1.is_finite() || !lambda2.is_finite() {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }

    // Intrinsic limit: mass term vanishes (kappa^2 = 0).
    if lambda0 <= 0.0 {
        if lambda1 > 0.0 && lambda2 > 0.0 {
            return ContinuousSmoothnessOrder {
                lambda0,
                lambda1,
                lambda2,
                r_ratio: None,
                nu: Some(1.0),
                kappa2: Some(0.0),
                status: ContinuousSmoothnessOrderStatus::IntrinsicLimit,
            };
        }
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }
    // First-order limit when stiffness vanishes:
    //   lambda2 = 0 => use lambda0/lambda1 = kappa^2 with nu = 1.
    if lambda2 <= 0.0 {
        if lambda1 > 0.0 {
            return ContinuousSmoothnessOrder {
                lambda0,
                lambda1,
                lambda2,
                r_ratio: None,
                nu: Some(1.0),
                kappa2: Some(lambda0 / lambda1),
                status: ContinuousSmoothnessOrderStatus::FirstOrderLimit,
            };
        }
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }

    let r_ratio = (lambda1 * lambda1) / (lambda0 * lambda2);
    if !r_ratio.is_finite() {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }

    // From a_m = binom(nu,m) * kappa^{2(nu-m)} with m=0,1,2:
    //   R = lambda1^2 / (lambda0*lambda2) = 2*nu/(nu-1)
    //   nu = R/(R-2), and kappa^2 = lambda1 / ((R-2)*lambda2).
    //
    // Discriminant of spectral quadratic P(t)=lambda0+lambda1*t+lambda2*t^2:
    //   Delta_P = lambda1^2 - 4*lambda0*lambda2 = lambda0*lambda2*(R-4).
    // Non-Matérn regime is flagged by Delta_P < 0 (equiv. R < 4),
    // but nu/kappa2 are still reported when R > 2 as effective diagnostics.
    //
    // Delta_P is two rounded products and a subtraction: a value inside that
    // inner product's rounding band gamma_2*(lambda1^2 + 4*lambda0*lambda2) is the
    // perfect-square boundary R = 4 to the resolution the arithmetic has.
    let discriminant = lambda1 * lambda1 - 4.0 * lambda0 * lambda2;
    let discriminant_band =
        gam_linalg::roundoff::accumulation_band(2, lambda1 * lambda1 + 4.0 * lambda0 * lambda2);
    let status = if discriminant < -discriminant_band {
        ContinuousSmoothnessOrderStatus::NonMaternRegime
    } else {
        // Includes exact boundary R=4 (perfect-square case) and numerically
        // indistinguishable near-boundary points.
        ContinuousSmoothnessOrderStatus::Ok
    };
    // R is two rounded products and a quotient; when R - 2 sits inside R's own
    // rounding band gamma_3*R, nu = R/(R-2) has no correct digit.
    if r_ratio - 2.0 <= gam_linalg::roundoff::accumulation_growth(3) * r_ratio {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: Some(r_ratio),
            nu: None,
            kappa2: None,
            status,
        };
    }
    let nu = r_ratio / (r_ratio - 2.0);
    // Closed-form extraction required by the continuous-order benchmark:
    //
    //   R = lambda1^2 / (lambda0*lambda2) = 2*nu/(nu-1)
    //   => nu = R/(R-2).
    //
    //   lambda1/lambda2 = 2*kappa^2/(nu-1)
    //   => kappa^2 = ((nu-1)/2)*(lambda1/lambda2)
    //             = lambda1 / ((R-2)*lambda2).
    //
    // We use this exact closed form as the reported kappa^2.
    let kappa2 = lambda1 / ((r_ratio - 2.0) * lambda2);
    if !nu.is_finite() || !kappa2.is_finite() {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: Some(r_ratio),
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }

    ContinuousSmoothnessOrder {
        lambda0,
        lambda1,
        lambda2,
        r_ratio: Some(r_ratio),
        nu: Some(nu),
        kappa2: Some(kappa2),
        status,
    }
}

#[cfg(test)]
mod pvalue_unavailable_tests {
    use super::*;
    use crate::estimate::smooth_pvalue_unavailable;
    use gam_terms::smooth::{ShapeConstraint, ShapeSet, ShapeSpec};

    /// Every shape request (atom, conjunction, per-margin tensor) withholds
    /// the p-value with the typed reason, and only the unconstrained smooth is
    /// testable.
    #[test]
    fn every_shape_constraint_withholds_the_smooth_pvalue() {
        assert_eq!(smooth_pvalue_unavailable(&ShapeSpec::None), None);
        let mut conjunction = ShapeSet::single(ShapeConstraint::MonotoneIncreasing);
        conjunction
            .insert(ShapeConstraint::Concave)
            .expect("increasing and concave are compatible");
        let per_margin = ShapeSpec::PerMargin(vec![
            ShapeSet::single(ShapeConstraint::MonotoneIncreasing),
            ShapeSet::default(),
        ]);
        for shape in [
            ShapeConstraint::MonotoneIncreasing.into(),
            ShapeConstraint::MonotoneDecreasing.into(),
            ShapeConstraint::Convex.into(),
            ShapeConstraint::Concave.into(),
            ShapeSpec::Joint(conjunction),
            per_margin,
        ] {
            assert_eq!(
                smooth_pvalue_unavailable(&shape),
                Some(SmoothPValueUnavailable::ShapeConstrained),
                "{shape:?}"
            );
        }
        assert_eq!(
            SmoothPValueUnavailable::ShapeConstrained.label(),
            "shape_constrained"
        );
    }
}
