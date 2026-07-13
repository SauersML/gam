use crate::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, BasisBuildResult, BasisError,
    BasisMetadata, CenterStrategy, CenterStrategyKind, ConstantCurvatureBasisSpec,
    ConstructiveQuadratic, DuchonBasisSpec, ActivePenalty, ActivePenaltyInfo, DroppedPenaltyInfo,
    KroneckerFactoredBasis, MaternBasisSpec, MeasureJetBasisSpec, PenaltyCandidate, PenaltySource,
    SpatialIdentifiability,
    SphericalSplineBasisSpec, ThinPlateBasisSpec,
    apply_sum_to_zero_constraint, build_bspline_basis_1d, build_constant_curvature_basis,
    build_duchon_basiswithworkspace, build_matern_basiswithworkspace,
    build_matern_collocation_operator_matrices, build_measure_jet_basis,
    build_spherical_spline_basis, build_thin_plate_basis, center_strategy_is_auto,
    center_strategy_kind, center_strategy_with_num_centers, filter_penalty_candidates,
    pairwise_distance_bounds,
    pairwise_distance_bounds_sampled, points_in_aniso_y_space, select_centers_by_strategy,
};

use crate::construction::{
    kronecker_logdet_and_derivatives, kronecker_marginal_eigensystems, kronecker_product,
};

use penalty_priors::realize_coefficient_groups;

use gam_problem::EstimationError;

use gam_linalg::faer_ndarray::{fast_ab, fast_atb};

use gam_linalg::matrix::{
    BlockDesignOperator, DenseDesignOperator, DesignBlock, DesignMatrix, FiniteSignedWeightsView,
    LinearOperator, TensorProductDesignOperator,
};

use gam_problem::LinearInequalityConstraints;

use gam_runtime::resource::MatrixMaterializationError;

use faer::sparse::{SparseColMat, Triplet};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis, s};

use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, BTreeSet};

use std::f64;

use std::fs::File;

use std::ops::Range;

use std::path::PathBuf;

use std::sync::Arc;

#[cfg(test)]
mod prelude_lock_tests {
    /// Locks that `terms::smooth`'s shared import surface lives at `prelude.rs`
    /// (renamed from `imports.rs` per #1137) and is textually included via
    /// `include!("smooth/prelude.rs")`: this test only compiles because it is
    /// pasted in. Also pins a representative re-exported symbol.
    #[test]
    fn smooth_prelude_include_path_is_locked() {
        assert!(
            core::any::type_name::<gam_linalg::matrix::DesignMatrix>().contains("DesignMatrix")
        );
    }
}
