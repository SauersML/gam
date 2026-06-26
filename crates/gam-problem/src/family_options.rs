//! Fit-time outer-objective and exact-derivative-order declarations shared by
//! the custom-family solver entry points.
//!
//! These are the neutral, dependency-free capability enums lifted out of
//! `src/families/custom_family/options.rs` so they live below `solver` in the
//! crate graph. The cost/assert helpers and `BlockwiseFitOptions` /
//! `OuterDerivativePolicy` remain in the root crate because they depend on
//! root-crate types (`OuterScoreSubsample`, `JointPenaltyBundle`,
//! `OuterEvalContext`, `crate::solver::rho_optimizer::OuterEvalOrder`,
//! `gam_runtime::warm_start::Session`) and on the parameter-block-spec types
//! still being relocated into this crate.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactNewtonOuterObjective {
    RidgedQuadraticReml,
    StrictPseudoLaplace,
}

/// Highest exact outer derivative order a family wants to expose at the
/// current realized problem scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExactOuterDerivativeOrder {
    Zeroth,
    First,
    Second,
}

impl ExactOuterDerivativeOrder {
    pub const fn has_gradient(self) -> bool {
        !matches!(self, Self::Zeroth)
    }

    pub const fn has_hessian(self) -> bool {
        matches!(self, Self::Second)
    }
}
