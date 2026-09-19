//! The parametric-coefficient rows of a model summary, shared by every surface.
//!
//! The in-process CLI summary and the persisted-model summary the Python
//! `summary()` reads present the same table from the same fit, so the table is
//! built once, here, beside [`super::smooth_term_summary_rows`]. The Wald
//! reference distribution is read off the fit — `N(0, 1)` when the scale is
//! known, Student-t on `wald_residual_degrees_of_freedom` when it is estimated
//! — so no caller chooses it.

use crate::estimate::smooth_term_summary::SummaryBlockOffset;
use crate::estimate::summary::ParametricTermSummary;
use crate::model_types::result_types::UnifiedFitResult;
use gam_math::probability::{normal_two_sided_probability, student_t_two_sided_probability};
use gam_terms::smooth::{
    BoundedCoefficientPriorSpec, LinearCoefficientGeometry, LinearTermSpec, TermCollectionDesign,
    TermCollectionSpec,
};

/// Build the intercept and linear-term rows of a model summary.
///
/// Standard errors are the fit's display pair
/// (`UnifiedFitResult::display_coefficient_uncertainty`), so they carry one
/// recorded covariance definition (#2296). A multi-column linear term yields one
/// row per column, suffixed `[i]`; a constrained or bounded coefficient names
/// its geometry in the row label. `offset` places `design` inside the fit's
/// coefficient layout, as for [`super::smooth_term_summary_rows`]: every index
/// into `fit` is global, every index into `design` block-local.
pub fn parametric_term_summary_rows(
    design: &TermCollectionDesign,
    spec: &TermCollectionSpec,
    fit: &UnifiedFitResult,
    offset: SummaryBlockOffset,
) -> Vec<ParametricTermSummary> {
    let uncertainty = fit.display_coefficient_uncertainty();
    let se = uncertainty.as_ref().map(|view| &view.standard_errors);
    let scale_is_estimated = fit.likelihood_scale.wald_scale_is_estimated();
    let residual_df = fit.wald_residual_degrees_of_freedom();
    let row = |name: String, local: usize| {
        let idx = offset.coefficients + local;
        let estimate = fit.beta.get(idx).copied().unwrap_or(f64::NAN);
        let std_error = se.and_then(|s| s.get(idx).copied());
        let statistic = std_error
            .filter(|s| *s > 0.0)
            .map(|s| estimate / s)
            .filter(|z| z.is_finite());
        // Both tails come from the function that computes them, never from
        // `1 - CDF`, which saturates to p = 0 above |z| ≈ 8.3 (#2562).
        let pvalue = statistic
            .and_then(|z| {
                if scale_is_estimated {
                    residual_df.map(|df| student_t_two_sided_probability(z, df))
                } else {
                    Some(normal_two_sided_probability(z))
                }
            })
            .filter(|p| p.is_finite());
        ParametricTermSummary {
            name,
            estimate,
            std_error,
            statistic,
            pvalue,
        }
    };

    let mut rows = Vec::new();
    for idx in design.intercept_range.clone() {
        rows.push(row("Intercept".to_string(), idx));
    }
    for (name, range) in &design.linear_ranges {
        let label = linear_term_label(
            name,
            spec.linear_terms.iter().find(|term| term.name == *name),
        );
        for idx in range.clone() {
            let name = if range.len() > 1 {
                format!("{label}[{}]", idx - range.start)
            } else {
                label.clone()
            };
            rows.push(row(name, idx));
        }
    }
    rows
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
