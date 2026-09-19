use super::*;

#[derive(Clone, Debug)]
pub struct ParametricTermSummary {
    pub name: String,
    pub estimate: f64,
    pub std_error: Option<f64>,
    pub zvalue: Option<f64>,
    pub pvalue: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SmoothTermSummary {
    pub name: String,
    pub edf: f64,
    pub ref_df: f64,
    pub chi_sq: Option<f64>,
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
    /// The λ̂-selection replay refused this term: its penalty geometry or its
    /// certified selection could not be resolved. The fixed-λ tail treats the
    /// REML λ̂ as known and is anti-conservative under the null, so it is not
    /// published in its place.
    SelectionRefused,
}

impl SmoothPValueUnavailable {
    /// Serialized label carried into the model payload and the Python surface.
    pub fn label(self) -> &'static str {
        match self {
            Self::ShapeConstrained => "shape_constrained",
            Self::SelectionRefused => "selection_refused",
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
            Self::SelectionRefused => {
                "selection refused: the lambda-selection replay could not resolve this term's \
                 penalty geometry or selection, so no calibrated p-value exists; the fixed-lambda \
                 tail is anti-conservative and is not reported in its place"
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

#[derive(Clone, Debug)]
pub struct ModelSummary {
    pub family: String,
    pub deviance_explained: Option<f64>,
    /// The cross-model comparable REML/LAML criterion
    /// (`UnifiedFitResult::comparable_reml_score`), the value the Python summary
    /// publishes under the same name. `None` exactly when `raw_reml_score` is.
    pub reml_score: Option<f64>,
    /// The outer optimizer's own criterion, un-normalized
    /// (`UnifiedFitResult::reml_score`).
    pub raw_reml_score: Option<f64>,
    pub parametric_terms: Vec<ParametricTermSummary>,
    pub smooth_terms: Vec<SmoothTermSummary>,
    /// Exact covariance definition behind the coefficient standard errors
    /// (#2296). Result-owned: recorded from the pair the builder actually
    /// consumed, never from a display policy. `None` when the fit carries no
    /// coefficient standard errors at all.
    pub coefficient_se_source: Option<crate::model_types::CoefficientCovarianceDefinition>,
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

fn significance_stars(p: Option<f64>) -> &'static str {
    match p {
        Some(v) if v.is_finite() && v < 0.001 => "***",
        Some(v) if v.is_finite() && v < 0.01 => "**",
        Some(v) if v.is_finite() && v < 0.05 => "*",
        Some(v) if v.is_finite() && v < 0.1 => ".",
        _ => "",
    }
}

fn format_pvalue(p: Option<f64>) -> String {
    let Some(v) = p else {
        return "NA".to_string();
    };
    if !v.is_finite() {
        return "NA".to_string();
    }
    if v < 2e-16 {
        "< 2e-16".to_string()
    } else if v < 1e-4 {
        format!("{v:.2e}")
    } else {
        format!("{v:.4}")
    }
}

impl fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let paramnamew = self
            .parametric_terms
            .iter()
            .map(|t| t.name.len())
            .max()
            .unwrap_or(10)
            .max("Term".len());
        let smoothnamew = self
            .smooth_terms
            .iter()
            .map(|t| t.name.len())
            .max()
            .unwrap_or(10)
            .max("Term".len());

        writeln!(f, "Family: {}", self.family)?;
        let dev_txt = self
            .deviance_explained
            .map(|d| format!("{:.1}%", (100.0 * d).clamp(-9999.0, 9999.0)))
            .unwrap_or_else(|| "NA".to_string());
        let reml_txt = self
            .reml_score
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "NA".to_string());
        let raw_reml_txt = self
            .raw_reml_score
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "NA".to_string());
        writeln!(
            f,
            "Deviance Explained: {dev_txt} | REML Score: {reml_txt} | Raw REML Score: {raw_reml_txt}"
        )?;
        if let Some(source) = self.coefficient_se_source {
            writeln!(f, "Coefficient SE Covariance: {source}")?;
        }
        writeln!(f)?;

        writeln!(f, "Parametric Terms:")?;
        writeln!(f, "{:-<1$}", "", paramnamew + 59)?;
        writeln!(
            f,
            "{:<namew$} {:>10} {:>12} {:>10} {:>19}",
            "Term",
            "Estimate",
            "Standard Error",
            "Z Statistic",
            "Two-Sided P-Value",
            namew = paramnamew
        )?;
        writeln!(f, "{:-<1$}", "", paramnamew + 59)?;
        for term in &self.parametric_terms {
            let estimate = format!("{:.4}", term.estimate);
            let se = term
                .std_error
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "NA".to_string());
            let z = term
                .zvalue
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "NA".to_string());
            let p = format_pvalue(term.pvalue);
            let stars = significance_stars(term.pvalue);
            writeln!(
                f,
                "{:<namew$} {:>10} {:>12} {:>10} {:>19} {}",
                term.name,
                estimate,
                se,
                z,
                p,
                stars,
                namew = paramnamew
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Smooth Terms:")?;
        writeln!(f, "{:-<1$}", "", smoothnamew + 86)?;
        writeln!(
            f,
            "{:<namew$} {:>26} {:>30} {:>12} {:>10}",
            "Term",
            "Effective Degrees of Freedom",
            "Reference Degrees of Freedom",
            "Chi-Square",
            "P-Value",
            namew = smoothnamew
        )?;
        writeln!(f, "{:-<1$}", "", smoothnamew + 86)?;
        for term in &self.smooth_terms {
            let chisq = term
                .chi_sq
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "NA".to_string());
            let p = format_pvalue(term.pvalue);
            let stars = significance_stars(term.pvalue);
            writeln!(
                f,
                "{:<namew$} {:>26.2} {:>30.2} {:>12} {:>10} {}",
                term.name,
                term.edf,
                term.ref_df,
                chisq,
                p,
                stars,
                namew = smoothnamew
            )?;
        }
        for term in &self.smooth_terms {
            if let Some(reason) = term.pvalue_unavailable {
                writeln!(f, "  {}: {}", term.name, reason.explanation())?;
            }
        }
        // #2901: a term spending an uncertified penalty block publishes its EDF
        // unclamped, and says so here rather than in a number that looks clamped.
        for term in &self.smooth_terms {
            if let Some(label) = term.edf_rank_bound.as_deref() {
                writeln!(
                    f,
                    "  {}: {label}; its effective degrees of freedom are published unclamped",
                    term.name
                )?;
            }
        }
        writeln!(f)?;
        let order_terms = self
            .smooth_terms
            .iter()
            .filter_map(|t| t.continuous_order.as_ref().map(|o| (&t.name, o)))
            .collect::<Vec<_>>();
        if !order_terms.is_empty() {
            writeln!(f, "Continuous Smoothness Order:")?;
            writeln!(
                f,
                "{:<namew$} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>20}",
                "Term",
                "lambda0",
                "lambda1",
                "lambda2",
                "R",
                "nu",
                "kappa^2",
                "status",
                namew = smoothnamew
            )?;
            for (name, o) in order_terms {
                let r_txt = o
                    .r_ratio
                    .filter(|v| v.is_finite())
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "NA".to_string());
                let nu_txt =
                    o.nu.filter(|v| v.is_finite())
                        .map(|v| format!("{v:.4}"))
                        .unwrap_or_else(|| "NA".to_string());
                let kappa_txt = o
                    .kappa2
                    .filter(|v| v.is_finite())
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "NA".to_string());
                let status_txt = match o.status {
                    ContinuousSmoothnessOrderStatus::Ok => "Ok",
                    ContinuousSmoothnessOrderStatus::NonMaternRegime => "NonMaternRegime",
                    ContinuousSmoothnessOrderStatus::FirstOrderLimit => "FirstOrderLimit",
                    ContinuousSmoothnessOrderStatus::IntrinsicLimit => "IntrinsicLimit",
                    ContinuousSmoothnessOrderStatus::UndefinedZeroLambda => "UndefinedZeroLambda",
                };
                writeln!(
                    f,
                    "{:<namew$} {:>10.3e} {:>10.3e} {:>10.3e} {:>10} {:>10} {:>10} {:>20}",
                    name,
                    o.lambda0,
                    o.lambda1,
                    o.lambda2,
                    r_txt,
                    nu_txt,
                    kappa_txt,
                    status_txt,
                    namew = smoothnamew
                )?;
            }
            writeln!(f)?;
        }
        write!(
            f,
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod edf_rank_bound_label_tests {
    use super::*;

    /// #2901: a smooth term whose EDF spends an uncertified penalty block names the
    /// label beside the table, and a certified term adds no line.
    #[test]
    fn the_summary_names_an_uncertified_terms_edf_label_2901() {
        let row = |name: &str, edf: f64, label: Option<&str>| SmoothTermSummary {
            name: name.to_string(),
            edf,
            ref_df: edf.max(0.0),
            chi_sq: None,
            pvalue: None,
            continuous_order: None,
            basis_note: None,
            edf_rank_bound: label.map(str::to_string),
            pvalue_unavailable: None,
        };
        let summary = ModelSummary {
            family: "gaussian".to_string(),
            deviance_explained: None,
            reml_score: None,
            raw_reml_score: None,
            parametric_terms: Vec::new(),
            smooth_terms: vec![
                row("s(x1)", 3.2, None),
                row("s(x2)", 5.0003, Some("rank bound not certified")),
            ],
            coefficient_se_source: None,
        };
        let text = summary.to_string();
        assert!(
            text.contains(
                "s(x2): rank bound not certified; its effective degrees of freedom are published \
                 unclamped"
            ),
            "{text}"
        );
        assert!(!text.contains("s(x1): rank bound"), "{text}");
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

    /// The printed summary names the reason beside the table, so a blank
    /// p-value column is never read as "not significant".
    #[test]
    fn the_summary_names_a_withheld_shape_pvalue() {
        let row = |name: &str, reason: Option<SmoothPValueUnavailable>| SmoothTermSummary {
            name: name.to_string(),
            edf: 2.5,
            ref_df: 3.0,
            chi_sq: None,
            pvalue: None,
            continuous_order: None,
            basis_note: None,
            edf_rank_bound: None,
            pvalue_unavailable: reason,
        };
        let summary = ModelSummary {
            family: "gaussian".to_string(),
            deviance_explained: None,
            reml_score: None,
            raw_reml_score: None,
            parametric_terms: Vec::new(),
            smooth_terms: vec![
                row("s(x1)", None),
                row("s(x2)", Some(SmoothPValueUnavailable::ShapeConstrained)),
            ],
            coefficient_se_source: None,
        };
        let text = summary.to_string();
        assert!(
            text.contains("s(x2): shape-constrained: the null f = 0 is the apex"),
            "{text}"
        );
        assert!(!text.contains("s(x1): shape-constrained"), "{text}");
    }
}
