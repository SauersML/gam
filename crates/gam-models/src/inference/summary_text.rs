//! The text form of a saved model's summary: the one renderer behind
//! `gam summary MODEL`, Python's `print(model.summary())` and `str(model)`.
//!
//! It reads only the [`SummaryPayload`] that `Model.summary().to_dict()`
//! serializes, so the printed report and the dictionary cannot disagree, and
//! every surface prints the same bytes for the same model. Numbers use
//! [`gam_report::format_significant`], the `%g` form the other report
//! surfaces share. An absent quantity prints the reason the payload carries for
//! it; nothing prints a stand-in number.

use crate::inference::saved_summary::{SummaryConvergence, SummaryPayload};
use gam_report::{criterion_row, format_significant};
use std::fmt::Write as _;

/// The legend for the significance stars beside each p-value.
const SIGNIF_CODES: &str = "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1";

/// The smallest p-value printed as a number, as `summary.gam` prints it: below
/// this the two-sided tail has no correct digit left in double precision.
const P_VALUE_FLOOR: f64 = 2e-16;

/// Why a saved model carries no log-likelihood: `SummaryPayload::log_likelihood`
/// is `None` only at the exact zero-dispersion boundary.
const NO_LIKELIHOOD_AT_EXACT_FIT: &str = "exact fit: the profiled scale is zero, so no normalized density exists";

/// Below this a p-value prints in scientific notation.
const P_VALUE_SCIENTIFIC_BELOW: f64 = 1e-4;

/// Render a saved model's summary as the multi-line text report.
pub fn render_summary_text(summary: &SummaryPayload) -> String {
    let mut out = String::new();
    let mut line = |label: &str, value: String| {
        writeln!(out, "{label}: {value}").expect("writing to a String cannot fail");
    };
    line("Family", summary.family_name.clone());
    line("Link function", summary.link.clone());
    line("Formula", summary.formula.clone());
    line("Model class", summary.model_class.clone());
    // The spline scan certifies no optimizer and so names no estimator.
    if let Some(convergence) = &summary.convergence {
        line("Estimator", convergence.estimator.text.clone());
    }
    if let Some(n) = summary.n_obs {
        line("n", n.to_string());
    }

    let mut tables = String::new();
    parametric_table(summary, &mut tables);
    smooth_table(summary, &mut tables);
    if !tables.is_empty() {
        out.push('\n');
        out.push_str(&tables);
        out.push_str("---\n");
        out.push_str(SIGNIF_CODES);
        out.push('\n');
    }

    out.push('\n');
    let mut line = |label: &str, value: String| {
        writeln!(out, "{label}: {value}").expect("writing to a String cannot fail");
    };
    line("Deviance", format_significant(summary.deviance));
    if let Some(null_deviance) = summary.null_deviance {
        line("Null deviance", format_significant(null_deviance));
    }
    line(
        "Deviance explained",
        match (summary.deviance_explained, summary.deviance_explained_unavailable) {
            (Some(explained), _) => format!("{}%", format_significant(100.0 * explained)),
            (None, reason) => unavailable(reason),
        },
    );
    if let Some(adjusted) = summary.adjusted_r_squared {
        line("Adjusted R-squared", format_significant(adjusted));
    }
    if let Some(scale) = summary.scale {
        line("Scale estimate", format_significant(scale));
    }
    line(
        "REML score",
        criterion_row(summary.reml_score, summary.raw_reml_score, format_significant),
    );
    line(
        "Log-likelihood",
        summary
            .log_likelihood
            .map_or_else(|| unavailable(Some(NO_LIKELIHOOD_AT_EXACT_FIT)), format_significant),
    );
    // The conditional AIC is absent only when the fit has no normalized
    // likelihood, and the corrected one then carries that same reason.
    let criteria = &summary.information_criteria;
    line(
        "Conditional AIC",
        criteria.aic_conditional.map_or_else(
            || unavailable(criteria.aic_corrected_unavailable),
            format_significant,
        ),
    );
    line(
        "Corrected AIC",
        criteria.aic_corrected.map_or_else(
            || unavailable(criteria.aic_corrected_unavailable),
            format_significant,
        ),
    );
    if let Some(edf) = summary.edf_total {
        line("Effective dof", format_significant(edf));
    }
    if let Some(source) = summary.coefficient_se_source.as_deref() {
        line("Coefficient covariance", source.to_string());
    }
    line(
        "Convergence",
        summary.convergence.as_ref().map_or_else(
            || "not applicable (this route solves no optimizer whose termination is certified)".to_string(),
            convergence_text,
        ),
    );
    out
}

/// The words for a quantity the payload declined, with the reason it gives.
fn unavailable(reason: Option<&str>) -> String {
    match reason {
        Some(reason) => format!("unavailable ({reason})"),
        None => "unavailable".to_string(),
    }
}

fn parametric_table(summary: &SummaryPayload, out: &mut String) {
    if let Some(reason) = summary.parametric_terms_unavailable.as_deref() {
        writeln!(out, "Parametric coefficients: unavailable ({reason})\n")
            .expect("writing to a String cannot fail");
        return;
    }
    if summary.parametric_terms.is_empty() {
        return;
    }
    let statistic = summary.parametric_statistic.unwrap_or("statistic");
    let header = [
        String::new(),
        "Estimate".to_string(),
        "Std. Error".to_string(),
        format!("{statistic} value"),
        format!("Pr(>|{statistic}|)"),
        String::new(),
    ];
    let rows = summary
        .parametric_terms
        .iter()
        .map(|row| {
            [
                row.name.clone(),
                format_significant(row.estimate),
                optional_number(row.std_error),
                optional_number(row.statistic),
                format_p_value(row.p_value),
                significance_stars(row.p_value).to_string(),
            ]
        })
        .collect::<Vec<_>>();
    out.push_str("Parametric coefficients:\n");
    write_table(out, &header, &rows);
    out.push('\n');
}

fn smooth_table(summary: &SummaryPayload, out: &mut String) {
    if let Some(reason) = summary.smooth_terms_unavailable.as_deref() {
        writeln!(out, "Smooth terms: unavailable ({reason})\n").expect("writing to a String cannot fail");
        return;
    }
    if summary.smooth_terms.is_empty() {
        return;
    }
    let header = [
        String::new(),
        "edf".to_string(),
        "Ref.df".to_string(),
        summary.smooth_statistic.unwrap_or("statistic").to_string(),
        "p-value".to_string(),
        String::new(),
        "lambda".to_string(),
    ];
    let rows = summary
        .smooth_terms
        .iter()
        .map(|row| {
            [
                row.name.clone(),
                format_significant(row.edf),
                format_significant(row.ref_df),
                optional_number(row.statistic),
                format_p_value(row.p_value),
                significance_stars(row.p_value).to_string(),
                if row.lambdas.is_empty() {
                    "NA".to_string()
                } else {
                    row.lambdas
                        .iter()
                        .map(|lambda| format_significant(*lambda))
                        .collect::<Vec<_>>()
                        .join(", ")
                },
            ]
        })
        .collect::<Vec<_>>();
    out.push_str("Approximate significance of smooth terms:\n");
    write_table(out, &header, &rows);
    for row in &summary.smooth_terms {
        if let Some(reason) = row.p_value_unavailable {
            writeln!(out, "  {}: {}", row.name, reason.explanation())
                .expect("writing to a String cannot fail");
        }
    }
    // #2901: a term spending an uncertified penalty block publishes its EDF
    // unclamped, and says so here rather than in a number that looks clamped.
    for row in &summary.smooth_terms {
        if let Some(label) = row.edf_rank_bound.as_deref() {
            writeln!(
                out,
                "  {}: {label}; its effective degrees of freedom are published unclamped",
                row.name
            )
            .expect("writing to a String cannot fail");
        }
    }
    out.push('\n');
}

/// Write an aligned table: the first column and the stars column left-aligned,
/// every other column right-aligned, trailing blanks trimmed.
fn write_table<const N: usize>(out: &mut String, header: &[String; N], rows: &[[String; N]]) {
    let mut widths = [0usize; N];
    for cells in std::iter::once(header).chain(rows) {
        for (width, cell) in widths.iter_mut().zip(cells) {
            *width = (*width).max(cell.chars().count());
        }
    }
    for cells in std::iter::once(header).chain(rows) {
        let mut line = String::new();
        for (column, (cell, width)) in cells.iter().zip(widths).enumerate() {
            if column > 0 {
                line.push_str("  ");
            }
            let pad = " ".repeat(width - cell.chars().count());
            let left_aligned = column == 0 || header[column].is_empty();
            if left_aligned {
                line.push_str(cell);
                line.push_str(&pad);
            } else {
                line.push_str(&pad);
                line.push_str(cell);
            }
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
}

fn optional_number(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_string(), format_significant)
}

fn format_p_value(p: Option<f64>) -> String {
    match p {
        Some(p) if p.is_finite() && p < P_VALUE_FLOOR => format!("< {P_VALUE_FLOOR:e}"),
        Some(p) if p.is_finite() && p < P_VALUE_SCIENTIFIC_BELOW => format!("{p:.2e}"),
        Some(p) if p.is_finite() => format!("{p:.4}"),
        _ => "NA".to_string(),
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

fn convergence_text(convergence: &SummaryConvergence) -> String {
    let verdict = if convergence.certified {
        "certified"
    } else {
        "NOT certified"
    };
    let mut text = format!(
        "{verdict}; inner P-IRLS: {}; {} outer iterations",
        convergence.inner_status, convergence.outer_iterations
    );
    match &convergence.outer {
        None => text.push_str("; no smoothing parameter was optimized"),
        Some(outer) => {
            write!(
                text,
                "; {} stationarity: projected gradient {} <= bound {}",
                outer.kind,
                format_significant(outer.projected_gradient_norm),
                format_significant(outer.stationarity_bound),
            )
            .expect("writing to a String cannot fail");
            if let Some(psd) = outer.hessian_psd {
                text.push_str(if psd {
                    "; Hessian positive semidefinite"
                } else {
                    "; Hessian NOT positive semidefinite"
                });
            }
            if !outer.lambdas_railed.is_empty() {
                let railed = outer
                    .lambdas_railed
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(text, "; smoothing parameters at a bound: [{railed}]")
                    .expect("writing to a String cannot fail");
            }
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::saved_summary::{
        SummaryEstimator, SummaryInformationCriteria, SummaryOuterCertificate,
        SummaryParametricTermRow, SummarySmoothTermRow,
    };
    use gam_solve::estimate::SmoothPValueUnavailable;

    fn smooth_row(name: &str, edf: f64, label: Option<&str>) -> SummarySmoothTermRow {
        SummarySmoothTermRow {
            name: name.to_string(),
            edf,
            ref_df: 9.0,
            chi_sq: Some(41.2),
            statistic: Some(41.2 / 9.0),
            p_value: Some(3.1e-7),
            lambdas: vec![0.0125],
            edf_rank_bound: label.map(str::to_string),
            p_value_unavailable: None,
        }
    }

    /// A Gaussian `y ~ x1 + s(x2)` fit on 100 rows, every quantity fixed.
    fn fixed_small_model() -> SummaryPayload {
        SummaryPayload {
            formula: "y ~ x1 + s(x2)".to_string(),
            family_name: "Gaussian Identity".to_string(),
            link: "identity".to_string(),
            model_class: "standard".to_string(),
            group_metadata: None,
            deployment_extensions: Vec::new(),
            deviance: 23.5,
            null_deviance: Some(94.0),
            deviance_explained: Some(0.75),
            adjusted_r_squared: Some(0.7384),
            deviance_explained_unavailable: None,
            scale: Some(0.2525),
            log_likelihood: Some(-70.125),
            n_obs: Some(100),
            reml_score: Some(81.75),
            raw_reml_score: Some(80.5),
            reml_score_unavailable: None,
            null_space_logdet: None,
            null_dim: None,
            iterations: 7,
            edf_total: Some(6.875),
            edf_rank_bound: Vec::new(),
            information_criteria: SummaryInformationCriteria {
                aic_conditional: Some(155.5),
                edf_corrected: Some(7.75),
                aic_corrected: Some(157.25),
                scale_dof: Some(1.0),
                aic_corrected_unavailable: None,
            },
            lambdas: vec![0.0125],
            coefficients: Vec::new(),
            parametric_statistic: Some("t"),
            parametric_terms: vec![
                SummaryParametricTermRow {
                    name: "Intercept".to_string(),
                    estimate: 1.5,
                    std_error: Some(0.05),
                    statistic: Some(30.0),
                    p_value: Some(1e-50),
                },
                SummaryParametricTermRow {
                    name: "x1".to_string(),
                    estimate: -0.25,
                    std_error: Some(0.125),
                    statistic: Some(-2.0),
                    p_value: Some(0.0484),
                },
            ],
            parametric_terms_unavailable: None,
            smooth_statistic: Some("F"),
            smooth_terms: vec![smooth_row("s(x2)", 4.875, None)],
            smooth_terms_unavailable: None,
            curvature_estimands: Vec::new(),
            basis_checks: Vec::new(),
            covariance_kind: None,
            covariance_n: None,
            covariance_flat: None,
            coefficient_se_source: Some("smoothing-corrected".to_string()),
            convergence: Some(SummaryConvergence {
                certified: true,
                inner_status: "Converged".to_string(),
                outer_iterations: 7,
                outer: Some(SummaryOuterCertificate {
                    kind: "analytic_gradient".to_string(),
                    gradient_norm: 2e-9,
                    projected_gradient_norm: 1.5e-9,
                    stationarity_bound: 1e-6,
                    hessian_psd: Some(true),
                    lambdas_railed: Vec::new(),
                }),
                estimator: SummaryEstimator {
                    name: "penalized likelihood".to_string(),
                    reason: None,
                    text: "penalized likelihood".to_string(),
                },
            }),
        }
    }

    #[test]
    fn a_fixed_small_model_renders_the_golden_text() {
        let golden = "\
Family: Gaussian Identity
Link function: identity
Formula: y ~ x1 + s(x2)
Model class: standard
Estimator: penalized likelihood
n: 100

Parametric coefficients:
           Estimate  Std. Error  t value  Pr(>|t|)
Intercept       1.5        0.05       30   < 2e-16  ***
x1            -0.25       0.125       -2    0.0484  *

Approximate significance of smooth terms:
         edf  Ref.df        F  p-value       lambda
s(x2)  4.875       9  4.57778  3.10e-7  ***  0.0125

---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Deviance: 23.5
Null deviance: 94
Deviance explained: 75%
Adjusted R-squared: 0.7384
Scale estimate: 0.2525
REML score: 81.75
Log-likelihood: -70.125
Conditional AIC: 155.5
Corrected AIC: 157.25
Effective dof: 6.875
Coefficient covariance: smoothing-corrected
Convergence: certified; inner P-IRLS: Converged; 7 outer iterations; analytic_gradient stationarity: projected gradient 1.5e-09 <= bound 1e-06; Hessian positive semidefinite
";
        assert_eq!(render_summary_text(&fixed_small_model()), golden);
    }

    /// #2901: a smooth term whose EDF spends an uncertified penalty block names the
    /// label beside the table, and a certified term adds no line.
    #[test]
    fn the_summary_names_an_uncertified_terms_edf_label_2901() {
        let mut summary = fixed_small_model();
        summary.smooth_terms = vec![
            smooth_row("s(x1)", 3.2, None),
            smooth_row("s(x2)", 5.0003, Some("rank bound not certified")),
        ];
        let text = render_summary_text(&summary);
        assert!(
            text.contains(
                "s(x2): rank bound not certified; its effective degrees of freedom are published \
                 unclamped"
            ),
            "{text}"
        );
        assert!(!text.contains("s(x1): rank bound"), "{text}");
    }

    /// The printed summary names why a shape-constrained smooth has no p-value
    /// beside the table, so a blank p-value column is never read as "not
    /// significant".
    #[test]
    fn the_summary_names_a_withheld_shape_pvalue() {
        let mut summary = fixed_small_model();
        let mut constrained = smooth_row("s(x2)", 2.5, None);
        constrained.chi_sq = None;
        constrained.statistic = None;
        constrained.p_value = None;
        constrained.p_value_unavailable = Some(SmoothPValueUnavailable::ShapeConstrained);
        summary.smooth_terms = vec![smooth_row("s(x1)", 3.2, None), constrained];
        let text = render_summary_text(&summary);
        assert!(
            text.contains("s(x2): shape-constrained: the null f = 0 is the apex"),
            "{text}"
        );
        assert!(!text.contains("s(x1): shape-constrained"), "{text}");
    }

    /// Every absent quantity prints the reason the payload gives, never a number.
    #[test]
    fn an_absent_quantity_prints_its_reason() {
        let mut summary = fixed_small_model();
        summary.deviance_explained = None;
        summary.adjusted_r_squared = None;
        summary.deviance_explained_unavailable = Some("no intercept-only deviance");
        summary.information_criteria.aic_corrected = None;
        summary.information_criteria.aic_corrected_unavailable = Some("no smoothing-parameter covariance");
        summary.smooth_terms = Vec::new();
        summary.smooth_terms_unavailable = Some("no frozen term spec".to_string());
        let text = render_summary_text(&summary);
        assert!(text.contains("Deviance explained: unavailable (no intercept-only deviance)\n"), "{text}");
        assert!(!text.contains("Adjusted R-squared"), "{text}");
        assert!(text.contains("Corrected AIC: unavailable (no smoothing-parameter covariance)\n"), "{text}");
        assert!(text.contains("Smooth terms: unavailable (no frozen term spec)\n"), "{text}");
    }
}
