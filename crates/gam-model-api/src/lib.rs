//! The `CustomFamily` trait itself plus the evaluation result structs it returns
//! (`FamilyEvaluation`, joint-gradient/batched-term carriers) and the eval-scope /
//! outer-eval-context enums that parameterize trait calls.

pub mod families {
    pub mod custom_family {
        pub mod family_trait;
        pub mod joint_newton_defaults;
        pub mod options;
        pub mod psi_design;

        pub use family_trait::*;
        pub use options::*;
        pub use psi_design::*;
    }
}

pub mod joint_penalty;
pub mod outer_subsample;

pub use families::custom_family::*;
pub use joint_penalty::{JointPenaltyBundle, JointPenaltyError, JointPenaltySpec};
pub use outer_subsample::{OuterScoreSubsample, RowSet, WeightedOuterRow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterEvalOrder {
    /// Compute only the objective value.
    Value,
    /// Compute value and gradient only.
    ValueAndGradient,
    /// Compute value, gradient, and analytic Hessian when available.
    ValueGradientHessian,
}
