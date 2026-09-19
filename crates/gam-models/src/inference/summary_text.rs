//! The text form of a saved model's summary: the one renderer behind
//! `gam summary MODEL`, Python's `print(model.summary())` and `str(model)`.
//!
//! It reads only the [`SummaryPayload`] that `Model.summary().to_dict()`
//! serializes, so the printed report and the dictionary cannot disagree, and
//! every surface prints the same bytes for the same model.

use crate::inference::saved_summary::SummaryPayload;
use gam_report::{criterion_row, format_significant};
use std::fmt::Write as _;

/// Render a saved model's summary as the multi-line text report, without a
/// trailing newline.
pub fn render_summary_text(summary: &SummaryPayload) -> String {
    let mut out = String::from("GAM fitted model");
    let mut line = |label: &str, value: &str| {
        write!(out, "\n  {label}: {value}").expect("writing to a String cannot fail");
    };
    if !summary.formula.is_empty() {
        line("Formula", &summary.formula);
    }
    if !summary.family_name.is_empty() {
        line("Family", &summary.family_name);
    }
    // The spline scan certifies no optimizer and so names no estimator.
    if let Some(convergence) = &summary.convergence {
        line("Estimator", &convergence.estimator.text);
    }
    if !summary.model_class.is_empty() {
        line("Class", &summary.model_class);
    }
    if let Some(n_obs) = summary.n_obs {
        line("Training rows", &n_obs.to_string());
    }
    line("Deviance", &format_significant(summary.deviance));
    if summary.reml_score.is_some()
        || summary.raw_reml_score.is_some()
        || summary.reml_score_unavailable.is_some()
    {
        // gam-report owns the words for an absent criterion, so this line
        // names a fit without null-space metadata apart from an exact fit the
        // same way `gam fit` and the HTML report do (#2627).
        line(
            "REML score",
            &criterion_row(summary.reml_score, summary.raw_reml_score, format_significant),
        );
    }
    if let Some(edf) = summary.edf_total {
        line("Effective dof", &format_significant(edf));
    }
    if let Some(scale) = summary.scale {
        line("Scale", &format_significant(scale));
    }
    if let Some(convergence) = &summary.convergence {
        line(
            "Iterations",
            &format!(
                "{} outer, {} inner",
                convergence.outer_iterations, convergence.inner_iterations
            ),
        );
    }
    if !summary.coefficients.is_empty() {
        line("Coefficients", &summary.coefficients.len().to_string());
    }
    if !summary.notes.is_empty() {
        out.push_str("\n  Notes:");
        for note in &summary.notes {
            write!(out, "\n    - {note}").expect("writing to a String cannot fail");
        }
    }
    out
}
