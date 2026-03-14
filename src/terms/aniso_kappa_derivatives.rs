// This module contains build_matern_aniso_log_kappa_derivatives.
// It is included inline via the placeholder replacement in basis.rs.
//
// The function builds per-axis log-kappa derivatives for anisotropic Matern
// terms (D0 only). For details see the doc comment on the function.

use super::basis::{
    AnisoBasisPsiDerivatives, BasisError, MaternBasisSpec,
    aniso_distance, aniso_distance_and_components,
    matern_aniso_radial_scalars, matern_kernel_from_distance,
    matern_identifiability_transform, select_centers_by_strategy,
    normalize_penaltywith_psi_derivatives, symmetrize,
};
use crate::faer_ndarray::{fast_ab, fast_ata};
use ndarray::{Array2, ArrayView2, s};
