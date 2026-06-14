// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

pub struct BinomialLocationScaleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub threshold_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
}

/// Both Binomial location-scale families plug into the unified
/// [`LocationScaleJointPsiFamily`] trait with byte-identical thin delegations
/// to inherent methods, differing only in the implementing type and its
/// `LABEL` fragment; generate them from one template. The Binomial families do
/// not thread the outer-row subsample (they run the full-data exact ψ path), so
/// the trait's `subsample` argument is accepted and ignored here.
macro_rules! impl_binomial_location_scale_joint_psi_family {
    ($family:ty, $label:literal) => {
        impl LocationScaleJointPsiFamily for $family {
            type Direction = LocationScaleJointPsiDirection;
            const LABEL: &'static str = $label;

            fn ws_policy(&self) -> &crate::resource::ResourcePolicy {
                &self.policy
            }

            fn ws_exact_joint_dense_block_designs<'a>(
                &'a self,
                specs: Option<&'a [ParameterBlockSpec]>,
            ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
                self.exact_joint_dense_block_designs(specs)
            }

            fn ws_psi_direction(
                &self,
                block_states: &[ParameterBlockState],
                derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
                psi_index: usize,
                design_loc: &Array2<f64>,
                design_scale: &Array2<f64>,
                policy: &crate::resource::ResourcePolicy,
            ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
                self.exact_newton_joint_psi_direction(
                    block_states,
                    derivative_blocks,
                    psi_index,
                    design_loc,
                    design_scale,
                    policy,
                )
            }

            fn ws_psi_second_order_terms_from_parts(
                &self,
                block_states: &[ParameterBlockState],
                derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
                psi_a: &LocationScaleJointPsiDirection,
                psi_b: &LocationScaleJointPsiDirection,
                design_loc: &Array2<f64>,
                design_scale: &Array2<f64>,
                subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
            ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String> {
                assert!(subsample.is_none());
                self.exact_newton_joint_psisecond_order_terms_from_parts(
                    block_states,
                    derivative_blocks,
                    psi_a,
                    psi_b,
                    design_loc,
                    design_scale,
                )
            }

            fn ws_psi_hessian_directional_from_parts(
                &self,
                block_states: &[ParameterBlockState],
                psi_dir: &LocationScaleJointPsiDirection,
                d_beta_flat: &Array1<f64>,
                design_loc: &Array2<f64>,
                design_scale: &Array2<f64>,
                subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
            ) -> Result<Array2<f64>, String> {
                assert!(subsample.is_none());
                self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                    block_states,
                    psi_dir,
                    d_beta_flat,
                    design_loc,
                    design_scale,
                )
            }
        }
    };
}

impl_binomial_location_scale_joint_psi_family!(
    BinomialLocationScaleFamily,
    "BinomialLocationScaleFamily"
);

impl_binomial_location_scale_joint_psi_family!(
    BinomialLocationScaleWiggleFamily,
    "BinomialLocationScaleWiggleFamily"
);

pub(crate) type BinomialLocationScaleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<BinomialLocationScaleFamily>;

pub(crate) type BinomialLocationScaleWiggleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<BinomialLocationScaleWiggleFamily>;
