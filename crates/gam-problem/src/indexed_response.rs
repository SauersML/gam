//! Per-row likelihood weights for separable indexed responses.
//!
//! Every `(row, output)` cell of the response grid carries a likelihood
//! contribution. Each row has one finite, non-negative numerical weight shared
//! across its outputs; a zero weight removes that row's contribution without
//! changing the response geometry.

use ndarray::ArrayView1;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Invalid indexed-response likelihood measure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedResponseError {
    reason: String,
}

impl IndexedResponseError {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    /// Human-readable invariant violation.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl Display for IndexedResponseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl Error for IndexedResponseError {}

/// Numerical likelihood weights for a separable `(row, output)` response: one
/// weight per observation row, shared across that row's outputs.
#[derive(Clone, Copy, Debug)]
pub struct SeparableCellMeasure<'a> {
    row_weights: ArrayView1<'a, f64>,
}

impl<'a> SeparableCellMeasure<'a> {
    /// A shared numerical weight per row.
    pub const fn row_weighted(weights: ArrayView1<'a, f64>) -> Self {
        Self {
            row_weights: weights,
        }
    }

    /// Validate the weight count against `N` and every weight's finiteness and
    /// sign.
    pub fn validate(&self, n_rows: usize) -> Result<(), IndexedResponseError> {
        if self.row_weights.len() != n_rows {
            return Err(IndexedResponseError::new(format!(
                "row likelihood weights length {} does not match N={n_rows}",
                self.row_weights.len()
            )));
        }
        for (row, &weight) in self.row_weights.iter().enumerate() {
            validate_weight(weight, format!("row likelihood weight[{row}]"))?;
        }
        Ok(())
    }

    /// Numerical weight of `row`, shared by every output of that row.
    pub fn row_weight(&self, row: usize) -> f64 {
        self.row_weights[row]
    }
}

fn validate_weight(weight: f64, context: String) -> Result<(), IndexedResponseError> {
    if !(weight.is_finite() && weight >= 0.0) {
        return Err(IndexedResponseError::new(format!(
            "{context} must be finite and non-negative (got {weight})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_row_weights_are_rejected() {
        let negative = ndarray::array![1.0, -1.0];
        let error = SeparableCellMeasure::row_weighted(negative.view())
            .validate(2)
            .expect_err("negative likelihood weight must fail");
        assert!(error.reason().contains("non-negative"));

        let short = ndarray::array![1.0];
        let error = SeparableCellMeasure::row_weighted(short.view())
            .validate(2)
            .expect_err("row weights must match N");
        assert!(error.reason().contains("does not match N=2"));
    }
}
