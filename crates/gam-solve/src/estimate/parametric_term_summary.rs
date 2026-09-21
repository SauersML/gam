//! The parametric-coefficient rows of a model summary, shared by every surface.
//!
//! The in-process CLI summary and the persisted-model summary the Python
//! `summary()` reads present the same table from the same fit, so the table is
//! built once, here, beside [`super::smooth_term_summary_rows`].
//!
//! An unpenalized coefficient is tested by its Wald ratio, whose reference
//! distribution is read off the fit — `N(0, 1)` when the scale is known,
//! Student-t on `wald_residual_degrees_of_freedom` when it is estimated — so no
//! caller chooses it.
//!
//! A linear term carrying the REML-selected `LinearTermRidge` is NOT: its `β̂`
//! was shrunk toward zero by a `λ` chosen from the same data, and under the
//! null REML drives `λ` large, so `β̂/se` collapses toward zero and its Wald
//! p-value piles up at one (a null slope then reads `p > 0.999` on most data
//! sets, gam#3573). "No effect" for a ridged slope is its variance component on
//! the boundary, and the row reports the variance-component score test the fit
//! recorded for it (`gam_terms::inference::random_effect_test`) — the same test
//! and record the random-effect rows read.

use crate::estimate::smooth_term_summary::SummaryBlockOffset;
use crate::estimate::summary::{
    ParametricPValueUnavailable, ParametricTermSummary, ParametricTest,
};
use crate::model_types::result_types::UnifiedFitResult;
use gam_math::probability::{normal_two_sided_probability, student_t_two_sided_probability};
use gam_terms::inference::random_effect_test::RandomEffectTestOutcome;
use gam_terms::smooth::{
    BoundedCoefficientPriorSpec, LinearCoefficientGeometry, LinearTermSpec, TermCollectionDesign,
    TermCollectionSpec,
};

/// Build the intercept and linear-term rows of a model summary.
///
/// Standard errors are the fit's display pair
/// (`UnifiedFitResult::display_coefficient_uncertainty`), so they carry one
/// recorded covariance definition (#2296). A ridged linear term's `statistic`
/// and `pvalue` are its recorded score test (see the module docs), matched by
/// name AND global coefficient range; a ridged term with no usable record
/// reports neither rather than the shrunk estimate's Wald ratio. A
/// multi-column linear term yields one row per column, suffixed `[i]`; a
/// constrained or bounded coefficient names
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
            test: ParametricTest::Wald,
            pvalue_unavailable: None,
        }
    };

    // The score-tested row of a ridged slope. The record carries the signed
    // root `u/√(φ̂·μ)` of its own score — `z` for a known scale, the partial
    // `t` for an estimated one — beside the exact tail it already holds. The
    // sign is the score's, not the shrunk estimate's: under the null REML
    // drives `β̂` toward zero, so `β̂`'s sign carries nothing about the
    // effect's direction. A record that could not be tested, and a ridged term
    // with no record at all, name their reason instead of reporting nothing.
    let ridged_row = |name: String, local: usize, outcome: Option<&RandomEffectTestOutcome>| {
        let idx = offset.coefficients + local;
        let estimate = fit.beta.get(idx).copied().unwrap_or(f64::NAN);
        let std_error = se.and_then(|s| s.get(idx).copied());
        let (statistic, pvalue, pvalue_unavailable) = match outcome {
            Some(RandomEffectTestOutcome::Tested(test)) => (
                test.signed_root.filter(|z| z.is_finite()),
                Some(test.p_value).filter(|p| p.is_finite()),
                None,
            ),
            Some(RandomEffectTestOutcome::Unavailable { reason }) => (
                None,
                None,
                Some(ParametricPValueUnavailable::VarianceComponent(*reason)),
            ),
            None => (
                None,
                None,
                Some(ParametricPValueUnavailable::VarianceComponentTestNotRecorded),
            ),
        };
        ParametricTermSummary {
            name,
            estimate,
            std_error,
            statistic,
            pvalue,
            test: ParametricTest::VarianceComponentScore,
            pvalue_unavailable,
        }
    };

    let ridged = design.ridged_linear_ranges();
    let mut rows = Vec::new();
    for idx in design.intercept_range.clone() {
        rows.push(row("Intercept".to_string(), idx));
    }
    for (name, range) in &design.linear_ranges {
        let label = linear_term_label(
            name,
            spec.linear_terms.iter().find(|term| term.name == *name),
        );
        let is_ridged = ridged
            .iter()
            .any(|(ridged_name, ridged_range)| ridged_name == name && ridged_range == range);
        if is_ridged {
            let global = (offset.coefficients + range.start)..(offset.coefficients + range.end);
            let outcome = fit
                .artifacts
                .random_effect_tests
                .iter()
                .find(|record| record.term == *name && record.coefficient_range == global)
                .map(|record| &record.outcome);
            // The record tests the term's whole block jointly; a single column
            // is the only shape a ridged linear term is built with, and a
            // per-column split of a joint test does not exist.
            let outcome = outcome.filter(|_| range.len() == 1);
            for idx in range.clone() {
                let name = if range.len() > 1 {
                    format!("{label}[{}]", idx - range.start)
                } else {
                    label.clone()
                };
                rows.push(ridged_row(name, idx, outcome));
            }
            continue;
        }
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
                BoundedCoefficientPriorSpec::Shrinkage => "shrinkage(REML)".to_string(),
                BoundedCoefficientPriorSpec::Beta { a, b } => format!("Beta({a:.3},{b:.3})"),
            };
            let bounds = bounds.map(|b| format!(", {b}")).unwrap_or_default();
            format!("{name} [bounded {min:.3}..{max:.3}, {prior}{bounds}]")
        }
    }
}
