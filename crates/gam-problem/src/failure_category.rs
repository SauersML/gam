//! The fixed category of a failure that stopped a fit (gam#2937).
//!
//! Every fit failure used to reach Python as `IntegrationError`, whatever had
//! failed, so a caller could not tell a refused start from a quadrature that
//! missed its tolerance. Each typed engine error now answers which category it
//! belongs to through an exhaustive match on its own variants, and a front end
//! selects its exception class from that answer, never from the rendered text.

/// What kind of failure stopped a fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureCategory {
    /// An outer smoothing search or an inner coefficient solve ended without its
    /// convergence certificate.
    Convergence,
    /// Outer startup validation refused every candidate seed, so no outer solver
    /// started.
    StartupSeeds,
    /// A result or intermediate state violated the engine's own consistency
    /// contract. An engine defect, not a property of the data.
    Invariant,
    /// The configuration, the data or the problem's size was refused.
    Input,
    /// A numerical step failed: a factorization, an eigendecomposition, a root
    /// solve, or a row quantity that `f64` cannot represent.
    Numerical,
    /// A quadrature or numerical integration did not reach its tolerance.
    Integration,
    /// The failure reached the boundary as prose, so no category is known.
    Unclassified,
}

impl FailureCategory {
    /// The stable snake_case label a front end exposes as the failure's
    /// `category`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Convergence => "convergence",
            Self::StartupSeeds => "startup_seeds",
            Self::Invariant => "invariant",
            Self::Input => "input",
            Self::Numerical => "numerical",
            Self::Integration => "integration",
            Self::Unclassified => "unclassified",
        }
    }

    /// The user-facing category a fit failure of this kind reports.
    ///
    /// Every numerical way a fit can stop short of a usable estimate is a
    /// convergence failure. `Unclassified` is one too: it names a fit that
    /// stopped without producing an estimate, whose producing helper's text
    /// spans kinds, so the only thing known of it is that the fit did not
    /// finish. A refused input is a property of the data the fit was handed;
    /// a violated invariant is an engine defect.
    #[must_use]
    pub const fn error_category(self) -> gam_spec::ErrorCategory {
        use gam_spec::ErrorCategory;
        match self {
            Self::Convergence
            | Self::StartupSeeds
            | Self::Numerical
            | Self::Integration
            | Self::Unclassified => ErrorCategory::Convergence,
            Self::Input => ErrorCategory::Data,
            Self::Invariant => ErrorCategory::Internal,
        }
    }
}

impl std::fmt::Display for FailureCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
