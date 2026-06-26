use crate::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, BasisBuildResult, BasisError,
    BasisMetadata, BasisPsiDerivativeResult, BasisPsiSecondDerivativeResult, BasisWorkspace,
    CenterStrategy, CenterStrategyKind, ConstantCurvatureBasisSpec,
    ConstantCurvatureIdentifiability, DuchonBasisSpec, KroneckerFactoredBasis, MaternBasisSpec,
    MaternIdentifiability, MeasureJetBasisSpec, MeasureJetFrozenQuadrature,
    MeasureJetIdentifiability, OneDimensionalBoundary, PenaltyCandidate, PenaltyInfo,
    PenaltySource, SpatialIdentifiability, SphericalSplineBasisSpec,
    SphericalSplineIdentifiability, ThinPlateBasisSpec, apply_sum_to_zero_constraint,
    build_bspline_basis_1d, build_constant_curvature_basis,
    build_constant_curvature_basis_kappa_derivatives, build_duchon_basiswithworkspace,
    build_matern_basis_log_kappa_aniso_derivatives, build_matern_basis_log_kappa_derivatives,
    build_matern_basiswithworkspace, build_matern_collocation_operator_matrices,
    build_measure_jet_basis, build_measure_jet_basis_psi_derivatives, build_spherical_spline_basis,
    build_thin_plate_basis, build_thin_plate_basis_log_kappa_derivatives, center_strategy_is_auto,
    center_strategy_kind, center_strategy_num_centers, center_strategy_with_num_centers,
    estimate_penalty_nullity, filter_active_penalty_candidates,
    filter_active_penalty_candidates_with_ops, initial_aniso_contrasts,
    orthogonality_transform_for_design, pairwise_distance_bounds, pairwise_distance_bounds_sampled,
    points_in_aniso_y_space, select_centers_by_strategy,
};

use crate::construction::{
    kronecker_logdet_and_derivatives, kronecker_marginal_eigensystems, kronecker_product,
};

use penalty_priors::{
    realize_coefficient_groups, realize_keyed_penalty_block_gamma_priors,
    realize_penalty_block_gamma_priors,
};

use gam_problem::EstimationError;

use gam_linalg::faer_ndarray::{fast_ab, fast_atb, fast_atv};

use gam_linalg::matrix::{
    BlockDesignOperator, CoefficientTransformOperator, DenseDesignOperator, DesignBlock,
    DesignMatrix, LinearOperator, RandomEffectOperator, SymmetricMatrix,
    TensorProductDesignOperator,
};

use crate::mixture_link::{
    logit_inverse_link_jet5, state_from_beta_logisticspec, state_from_sasspec, state_fromspec,
};

use gam_problem::LinearInequalityConstraints;

use gam_runtime::resource::MatrixMaterializationError;

use gam_spec::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, MixtureLinkState, ResponseFamily,
    SasLinkState, StandardLink,
};

use crate::util::quantile::quantile_from_sorted;

use faer::sparse::{SparseColMat, Triplet};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis, s};

use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, BTreeSet};

use std::f64;

use std::fs::File;

use std::ops::Range;

use std::path::PathBuf;

use std::sync::atomic::AtomicUsize;

use std::sync::{Arc, Mutex};

#[cfg(test)]
mod prelude_lock_tests {
    /// Locks that `terms::smooth`'s shared import surface lives at `prelude.rs`
    /// (renamed from `imports.rs` per #1137) and is textually included via
    /// `include!("smooth/prelude.rs")`: this test only compiles because it is
    /// pasted in. Also pins a representative re-exported symbol.
    #[test]
    fn smooth_prelude_include_path_is_locked() {
        assert!(core::any::type_name::<gam_linalg::matrix::DesignMatrix>().contains("DesignMatrix"));
    }
}
