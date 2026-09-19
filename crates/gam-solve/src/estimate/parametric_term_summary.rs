//! The parametric rows of a model summary, shared by every surface.
//!
//! The in-process CLI summary and the persisted-model summary the Python
//! `summary()` reads present the same tables from the same fit, so they are
//! built once, here, beside [`super::smooth_term_summary_rows`]. The Wald
//! reference distribution is read off the fit — `N(0, 1)` / `χ²_q` when the
//! scale is known, Student-t / `F(q, n − edf)` on
//! `wald_residual_degrees_of_freedom` when it is estimated — so no caller
//! chooses it.
//!
//! A linear term carries its own REML ridge (its function-mass penalty `S_J`),
//! and its posterior covariance charges that prior's variance to the estimate:
//! `β̂_J/√(V_p)_JJ` is too small under the null and its test is conservative.
//! With `A = X'WX + S_O` (the smooths' priors kept) and `H = A + S_J`, the rows
//! `J` of `H⁻¹` are `(I + (A⁻¹)_JJ S_J)⁻¹` times those of `A⁻¹`, so `β̂_J` and
//! its covariance `φ[H⁻¹AH⁻¹]_JJ` share that factor and the Wald statistic
//! `β̂_J' (φ[H⁻¹AH⁻¹]_JJ)⁻¹ β̂_J` equals the one of the fit with `J` left
//! unpenalized, whatever `λ_J` is. Since the statistic does not depend on
//! `λ_J`, `λ_J`'s uncertainty is not charged to it: the covariance is the
//! conditional `φ[H⁻¹AH⁻¹]_JJ = (V_c)_JJ − φ(H⁻¹S_JH⁻¹)_JJ`, not the
//! smoothing-corrected one, whose first-order `λ_J` term made the test
//! conservative again. An unpenalized term (a fixed factor) has `S_J = 0` and
//! keeps the display covariance.

use crate::estimate::summary::{
    ParametricPValueUnavailable, ParametricTermSummary, ParametricTermTest,
};
use crate::model_types::result_types::UnifiedFitResult;
use gam_linalg::triangular::{
    CholeskyGuard, cholesky_factor_in_place, forward_substitution_lower_vector,
};
use gam_math::probability::{
    chi_square_sf, fisher_snedecor_sf, normal_two_sided_probability,
    student_t_two_sided_probability,
};
use gam_terms::smooth::{
    BoundedCoefficientPriorSpec, LinearCoefficientGeometry, LinearTermSpec, TermCollectionDesign,
    TermCollectionSpec,
};
use ndarray::{Array2, s};
use std::ops::Range;

/// The parametric tables of a model summary: one row per coefficient, and one
/// joint test per term.
#[derive(Clone, Debug)]
pub struct ParametricTermTables {
    /// Intercept, linear-term and fixed-factor contrast coefficients.
    pub coefficients: Vec<ParametricTermSummary>,
    /// One Wald test per parametric term on its own number of columns: a
    /// factor with `L` levels is tested once on `L − 1` degrees of freedom.
    pub term_tests: Vec<ParametricTermTest>,
}

/// One parametric term: its coefficient block and the label of each column.
struct ParametricTerm {
    name: String,
    range: Range<usize>,
    column_labels: Vec<String>,
    /// A coefficient on a bound or inside a bounded geometry has no Wald
    /// sampling distribution at the boundary.
    bounded: bool,
}

/// The reference distribution of every parametric Wald statistic of one fit.
#[derive(Clone, Copy)]
enum WaldReference {
    /// Known scale: `N(0, 1)` for one coefficient, `χ²_q` for a block.
    Known,
    /// Estimated scale: Student-t / `F(q, ·)` on the residual degrees of
    /// freedom `n − edf_total`; `None` when the fit leaves none.
    Estimated(Option<f64>),
}

impl WaldReference {
    fn of(fit: &UnifiedFitResult) -> Self {
        if fit.likelihood_scale.wald_scale_is_estimated() {
            Self::Estimated(fit.wald_residual_degrees_of_freedom())
        } else {
            Self::Known
        }
    }
}

/// Build the parametric coefficient rows and term tests of a model summary.
///
/// Standard errors and the joint covariance start from the fit's display pair
/// (`UnifiedFitResult::display_coefficient_uncertainty`), so they carry one
/// recorded covariance definition (#2296). A term carrying its own ridge has
/// that prior's variance removed (module doc); its rows report the resulting
/// null sampling SD with `penalized` set. A multi-column linear term yields one
/// row per column, suffixed `[i]`; a constrained or bounded coefficient names
/// its geometry in the row label. An unpenalized factor contributes one row per
/// contrast column, labelled `name[level]` by `level_label(feature_col, code)`,
/// and one joint term test.
pub fn parametric_term_summary_rows(
    design: &TermCollectionDesign,
    spec: &TermCollectionSpec,
    fit: &UnifiedFitResult,
    level_label: &dyn Fn(usize, u64) -> String,
) -> ParametricTermTables {
    let terms = parametric_terms(design, spec, level_label);
    let uncertainty = fit.display_coefficient_uncertainty();
    let se = uncertainty.as_ref().map(|view| &view.standard_errors);
    let covariance = uncertainty.as_ref().and_then(|view| view.covariance);
    let reference = WaldReference::of(fit);

    let mut coefficients = Vec::new();
    let mut term_tests = Vec::new();
    for term in &terms {
        let own_penalty = own_penalty(design, fit, &term.range);
        let penalized = own_penalty.is_some();
        let null_covariance = null_sampling_covariance(fit, covariance, &term.range, own_penalty);
        let first = coefficients.len();
        for (local, (idx, label)) in term.range.clone().zip(&term.column_labels).enumerate() {
            let estimate = fit.beta.get(idx).copied().unwrap_or(f64::NAN);
            let std_error = if penalized {
                null_covariance
                    .as_ref()
                    .map(|block| block[[local, local]])
                    .filter(|variance| *variance >= 0.0)
                    .map(f64::sqrt)
            } else {
                se.and_then(|s| s.get(idx).copied())
            };
            coefficients.push(coefficient_row(
                label.clone(),
                estimate,
                std_error,
                penalized,
                term.bounded,
                reference,
            ));
        }
        if term.name == "Intercept" {
            continue;
        }
        term_tests.push(term_test(
            term,
            &coefficients[first..],
            fit,
            null_covariance.as_ref(),
            reference,
        ));
    }
    ParametricTermTables {
        coefficients,
        term_tests,
    }
}

/// The intercept, the linear terms, and the unpenalized factor blocks, in
/// coefficient order.
fn parametric_terms(
    design: &TermCollectionDesign,
    spec: &TermCollectionSpec,
    level_label: &dyn Fn(usize, u64) -> String,
) -> Vec<ParametricTerm> {
    let mut terms = Vec::new();
    if !design.intercept_range.is_empty() {
        terms.push(ParametricTerm {
            name: "Intercept".to_string(),
            range: design.intercept_range.clone(),
            column_labels: vec!["Intercept".to_string(); design.intercept_range.len()],
            bounded: false,
        });
    }
    for (name, range) in &design.linear_ranges {
        let meta = spec.linear_terms.iter().find(|term| term.name == *name);
        let label = linear_term_label(name, meta);
        let column_labels = (0..range.len())
            .map(|i| {
                if range.len() > 1 {
                    format!("{label}[{i}]")
                } else {
                    label.clone()
                }
            })
            .collect();
        let bounded = meta.is_some_and(|meta| {
            meta.coefficient_min.is_some()
                || meta.coefficient_max.is_some()
                || !matches!(
                    meta.coefficient_geometry,
                    LinearCoefficientGeometry::Unconstrained
                )
        });
        terms.push(ParametricTerm {
            name: name.clone(),
            range: range.clone(),
            column_labels,
            bounded,
        });
    }
    // A penalized random-effect block is a variance component, tested (or not)
    // with the smooth terms; only the unpenalized blocks — fixed factors — are
    // parametric.
    for (re_idx, (name, range)) in design.random_effect_ranges.iter().enumerate() {
        let Some(meta) = spec.random_effect_terms.get(re_idx) else {
            continue;
        };
        if meta.penalized || range.is_empty() {
            continue;
        }
        let codes = design
            .random_effect_levels
            .get(re_idx)
            .map(|(_, codes)| codes.as_slice())
            .or_else(|| meta.column_levels())
            .unwrap_or(&[]);
        let column_labels = (0..range.len())
            .map(|i| match codes.get(i) {
                Some(&code) => format!("{name}[{}]", level_label(meta.feature_col, code)),
                None => format!("{name}[{i}]"),
            })
            .collect();
        terms.push(ParametricTerm {
            name: name.clone(),
            range: range.clone(),
            column_labels,
            bounded: false,
        });
    }
    terms
}

/// The term's own penalty `Σ_k λ_k S_k` over the blocks that lie inside its
/// coefficient range, or `None` when it owns no penalty. `fit.lambdas` is laid
/// out one entry per `design.penalties` block, in order.
fn own_penalty(
    design: &TermCollectionDesign,
    fit: &UnifiedFitResult,
    range: &Range<usize>,
) -> Option<Array2<f64>> {
    let q = range.len();
    let mut penalty = Array2::<f64>::zeros((q, q));
    let mut owned = false;
    for (block, lambda) in design.penalties.iter().zip(fit.lambdas.iter()) {
        let cols = &block.col_range;
        if cols.is_empty() || cols.start < range.start || cols.end > range.end {
            continue;
        }
        let offset = cols.start - range.start;
        penalty
            .slice_mut(s![offset..offset + cols.len(), offset..offset + cols.len()])
            .scaled_add(*lambda, &block.local);
        owned = true;
    }
    owned.then_some(penalty)
}

/// The covariance a term's Wald statistic is scaled by: the display block for an
/// unpenalized term, and for a term carrying its own penalty `S_J` the
/// conditional block less its own prior,
/// `φ[H⁻¹(H − S_J)H⁻¹]_JJ = (V_c)_JJ − (V_c)_JJ S_J (V_c)_JJ / φ` with
/// `V_c = φH⁻¹` (see the module docs). `None` when an input is absent.
fn null_sampling_covariance(
    fit: &UnifiedFitResult,
    covariance: Option<&Array2<f64>>,
    range: &Range<usize>,
    own_penalty: Option<Array2<f64>>,
) -> Option<Array2<f64>> {
    let Some(own_penalty) = own_penalty else {
        return Some(covariance?.slice(s![range.clone(), range.clone()]).to_owned());
    };
    let conditional = fit
        .beta_covariance()?
        .slice(s![range.clone(), range.clone()])
        .to_owned();
    let phi = fit.dispersion_phi().ok().filter(|phi| *phi > 0.0)?;
    let own_prior = conditional.dot(&own_penalty).dot(&conditional) / phi;
    Some(conditional - own_prior)
}

/// One coefficient's Wald row.
fn coefficient_row(
    name: String,
    estimate: f64,
    std_error: Option<f64>,
    penalized: bool,
    bounded: bool,
    reference: WaldReference,
) -> ParametricTermSummary {
    let statistic = std_error
        .filter(|s| *s > 0.0)
        .map(|s| estimate / s)
        .filter(|z| z.is_finite());
    // Both tails come from the function that computes them, never from
    // `1 - CDF`, which saturates to p = 0 above |z| ≈ 8.3 (#2562).
    let pvalue = match (bounded, statistic, reference) {
        (true, _, _) | (_, None, _) => None,
        (false, Some(z), WaldReference::Known) => Some(normal_two_sided_probability(z)),
        (false, Some(z), WaldReference::Estimated(df)) => {
            df.map(|df| student_t_two_sided_probability(z, df))
        }
    }
    .filter(|p| p.is_finite());
    let pvalue_unavailable = pvalue.is_none().then(|| {
        if bounded {
            ParametricPValueUnavailable::BoundedCoefficient
        } else if std_error.is_none() {
            ParametricPValueUnavailable::NoCovariance
        } else if statistic.is_none() {
            ParametricPValueUnavailable::SingularCovariance
        } else {
            ParametricPValueUnavailable::NoResidualDegreesOfFreedom
        }
    });
    ParametricTermSummary {
        name,
        estimate,
        std_error,
        penalized,
        statistic,
        pvalue,
        pvalue_unavailable,
    }
}

/// The joint Wald test of one term's coefficients being all zero.
///
/// A one-column term's test is its coefficient row squared, with the same
/// p-value bit for bit (`t² ~ F(1, ν)`, `z² ~ χ²_1`), so the two tables never
/// disagree about the same hypothesis.
fn term_test(
    term: &ParametricTerm,
    rows: &[ParametricTermSummary],
    fit: &UnifiedFitResult,
    null_covariance: Option<&Array2<f64>>,
    reference: WaldReference,
) -> ParametricTermTest {
    let df = term.range.len();
    if let [row] = rows {
        return ParametricTermTest {
            name: term.name.clone(),
            df,
            statistic: row.statistic.map(|z| z * z),
            pvalue: row.pvalue,
            pvalue_unavailable: row.pvalue_unavailable,
        };
    }
    let unavailable = |reason| ParametricTermTest {
        name: term.name.clone(),
        df,
        statistic: None,
        pvalue: None,
        pvalue_unavailable: Some(reason),
    };
    if term.bounded {
        return unavailable(ParametricPValueUnavailable::BoundedCoefficient);
    }
    let Some(block) = null_covariance else {
        return unavailable(ParametricPValueUnavailable::NoCovariance);
    };
    let Some(factor) = cholesky_factor_in_place(block.view(), CholeskyGuard::FiniteStrict) else {
        return unavailable(ParametricPValueUnavailable::SingularCovariance);
    };
    let beta = fit.beta.slice(s![term.range.clone()]);
    let whitened = forward_substitution_lower_vector(&factor, beta);
    let wald = whitened.dot(&whitened);
    if !wald.is_finite() {
        return unavailable(ParametricPValueUnavailable::SingularCovariance);
    }
    let q = df as f64;
    let (statistic, pvalue) = match reference {
        WaldReference::Known => (wald, chi_square_sf(wald, q)),
        WaldReference::Estimated(None) => {
            return ParametricTermTest {
                statistic: Some(wald / q),
                ..unavailable(ParametricPValueUnavailable::NoResidualDegreesOfFreedom)
            };
        }
        WaldReference::Estimated(Some(residual_df)) => {
            let f = wald / q;
            (f, fisher_snedecor_sf(f, q, residual_df))
        }
    };
    ParametricTermTest {
        name: term.name.clone(),
        df,
        statistic: Some(statistic),
        pvalue: pvalue.is_finite().then_some(pvalue),
        pvalue_unavailable: (!pvalue.is_finite())
            .then_some(ParametricPValueUnavailable::SingularCovariance),
    }
}

/// A linear term's row label, naming any coefficient bound or bounded geometry.
fn linear_term_label(name: &str, meta: Option<&LinearTermSpec>) -> String {
    let Some(meta) = meta else {
        return name.to_string();
    };
    let bounds = match (meta.coefficient_min, meta.coefficient_max) {
        (Some(lb), Some(ub)) => Some(format!("coef in [{lb:.3}, {ub:.3}]")),
        (Some(lb), None) => Some(format!("coef >= {lb:.3}")),
        (None, Some(ub)) => Some(format!("coef <= {ub:.3}")),
        (None, None) => None,
    };
    match &meta.coefficient_geometry {
        LinearCoefficientGeometry::Unconstrained => match bounds {
            Some(bounds) => format!("{name} [{bounds}]"),
            None => name.to_string(),
        },
        LinearCoefficientGeometry::Bounded { min, max, prior } => {
            let prior = match prior {
                BoundedCoefficientPriorSpec::None => "no-prior".to_string(),
                BoundedCoefficientPriorSpec::Uniform => "Uniform(log-Jacobian)".to_string(),
                BoundedCoefficientPriorSpec::Beta { a, b } => format!("Beta({a:.3},{b:.3})"),
            };
            let bounds = bounds.map(|b| format!(", {b}")).unwrap_or_default();
            format!("{name} [bounded {min:.3}..{max:.3}, {prior}{bounds}]")
        }
    }
}
