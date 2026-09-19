//! The one user-facing error taxonomy shared by every engine front end.
//!
//! Every error type that can reach a user (data ingestion, formula
//! resolution, term construction, estimation, workflow orchestration)
//! reports exactly one [`ErrorCategory`]. The Python bindings map the
//! category 1:1 onto `gamfit.FormulaError` / `DataError` /
//! `ConvergenceError` / `NotFittedError` / `InternalError`, and the CLI maps
//! it onto its process exit code, so the two front ends classify every
//! failure identically without inspecting message text.

use std::fmt;

/// Who has to act on a failure, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// The request itself is invalid: the formula, a term option, a family /
    /// link / configuration choice, a path the call names, or a column the
    /// formula names that the data does not have. Fixed by changing the call,
    /// not the data.
    Formula,
    /// The request is well formed but the supplied data cannot support it:
    /// unparseable or non-finite values, schema mismatches between training
    /// and prediction frames, degenerate columns, separation, or a design too
    /// rank-deficient to identify. Fixed by changing the data.
    Data,
    /// The inputs are valid but the numerical solver did not reach a usable
    /// optimum (non-convergence, failed factorizations, integration or root
    /// finding failures, exhausted startup seeds).
    Convergence,
    /// A fitted-state operation (predict, summary, ...) was requested from an
    /// estimator that has not been fitted.
    NotFitted,
    /// An engine invariant was violated. Never the user's fault; always a bug
    /// to report.
    Internal,
}

impl ErrorCategory {
    /// Every category, in declaration order.
    pub const ALL: [Self; 5] = [
        Self::Formula,
        Self::Data,
        Self::Convergence,
        Self::NotFitted,
        Self::Internal,
    ];

    /// Stable machine-readable label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Formula => "formula",
            Self::Data => "data",
            Self::Convergence => "convergence",
            Self::NotFitted => "not_fitted",
            Self::Internal => "internal",
        }
    }

    /// Process exit code the CLI reports for a failure of this category.
    ///
    /// `2` matches the usage-error code the argument parser already uses for
    /// malformed invocations, so every "fix the request" failure shares one
    /// code. `70` is `EX_SOFTWARE` from `sysexits.h` ("internal software
    /// error"). The remaining categories take the next free small codes.
    pub const fn exit_code(self) -> i32 {
        match self {
            Self::Formula => 2,
            Self::Data => 3,
            Self::Convergence => 4,
            Self::NotFitted => 5,
            Self::Internal => 70,
        }
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::ErrorCategory;
    use std::collections::HashSet;

    #[test]
    fn labels_and_exit_codes_are_distinct_and_nonzero() {
        let labels: HashSet<_> = ErrorCategory::ALL.iter().map(|c| c.label()).collect();
        let codes: HashSet<_> = ErrorCategory::ALL.iter().map(|c| c.exit_code()).collect();
        assert_eq!(labels.len(), ErrorCategory::ALL.len());
        assert_eq!(codes.len(), ErrorCategory::ALL.len());
        assert!(codes.iter().all(|&code| code != 0 && code != 1));
    }
}
