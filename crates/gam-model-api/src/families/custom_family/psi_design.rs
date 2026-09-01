//! The ψ (hyperparameter) design-derivative operators: the
//! `CustomFamilyPsiDerivativeOperator` trait and every concrete operator
//! (implicit / embedded-implicit / zero / embedded-dense / rowwise-Kronecker),
//! the ψ design/second-design actions and linear-map refs, the joint ψ operator,
//! and the exact-Newton joint-ψ term carriers + workspace traits.

use crate::families::custom_family::family_trait::ExactNewtonJointGradientEvaluation;
use gam_problem::{CustomFamilyError, DenseMatrixHyperOperator, EvalMode, HyperOperator};
use ndarray::{Array1, Array2};
use std::sync::Arc;

// The neutral ψ-derivative carriers and operator traits live in
// `gam-problem`. Only the trait that couples to the `CustomFamily` evaluation
// carrier (`ExactNewtonJointHessianWorkspace`) stays local below.
pub use gam_problem::{
    CustomFamilyBlockPsiDerivative, CustomFamilyHyperAxis, CustomFamilyHyperLayout,
    CustomFamilyPsiDerivativeOperator, JointHessianSourcePreference,
    MaterializablePsiDerivativeOperator, MaterializationIntent, SharedCustomFamilyHyperLayout,
};

pub trait ExactNewtonJointHessianWorkspace: Send + Sync {

}
