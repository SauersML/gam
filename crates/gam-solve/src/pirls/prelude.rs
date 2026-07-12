//! Shared import/prelude surface for the `solver::pirls` concern modules.
//!
//! Every name here is re-exported `pub(crate)` so each concern module can pull
//! the whole set with a single `use super::*;` (glob imports are exempt from the
//! unused-import lint, so no module is forced to enumerate exactly what it
//! touches). This mirrors the flat namespace the module used to share before the
//! concern split, without the line-cut `include!` fragments.

pub(crate) use crate::estimate::EstimationError;

pub(crate) use crate::estimate::reml::{FirthDenseOperator, FirthDesignFactor};

pub(crate) use gam_linalg::faer_ndarray::{
    FaerCholesky, FaerEigh, FaerLinalgError, FaerSymmetricFactor, array1_to_col_matmut, fast_ab,
    fast_av, fast_av_into,
};

pub(crate) use gam_linalg::sparse_exact::{factorize_sparse_spd, solve_sparse_spd};

pub(crate) use gam_linalg::utils::{StableSolver, boundary_hit_step_fraction};

pub(crate) use gam_linalg::matrix::{DesignMatrix, LinearOperator};

pub(crate) use crate::mixture_link::{
    InverseLinkJet as MixtureInverseLinkJet, logit_inverse_link_jet5,
};

pub(crate) use crate::active_set;

pub(crate) use gam_problem::{Coefficients, LinearPredictor, StandardLink};

pub(crate) use gam_problem::outer_subsample::RowSet;
pub(crate) use gam_problem::{
    GlmLikelihoodSpec, InverseLink, LikelihoodSpec, LinkFunction, MixtureLinkState, ResponseFamily,
    SasLinkState, is_valid_tweedie_power,
};

pub(crate) use dyn_stack::{MemBuffer, MemStack};

pub(crate) use faer::sparse::SparseColMat;

pub(crate) use faer::sparse::linalg::matmul::{
    SparseMatMulInfo, sparse_sparse_matmul_numeric, sparse_sparse_matmul_numeric_scratch,
    sparse_sparse_matmul_symbolic,
};

pub(crate) use faer::sparse::{
    SparseColMatMut, SparseColMatRef, SparseRowMat, SymbolicSparseColMat, SymbolicSparseColMatRef,
};

pub(crate) use faer::{Accum, Par, Side, Unbind, get_global_parallelism};

pub(crate) use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ShapeBuilder, Zip};

pub(crate) use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

pub(crate) use statrs::function::gamma::{digamma, ln_gamma};

pub(crate) use faer::linalg::cholesky::llt::factor::LltParams;

pub(crate) use faer::{Auto, Spec};

pub(crate) use std::collections::BTreeSet;

pub(crate) use std::collections::HashMap;

pub(crate) use std::sync::{Arc, Mutex, OnceLock};

pub use crate::active_set::{ACTIVE_SET_PRIMAL_FEASIBILITY_TOL, ConstraintKktDiagnostics};

pub use gam_problem::LinearInequalityConstraints;

pub(crate) use gam_linalg::utils::{array_is_finite, inf_norm, row_chunk_for_byte_budget};

// `log` is used as a path (`log::debug!`), so re-export the crate itself.
pub(crate) use log;

#[cfg(test)]
mod tests {
    /// Locks the `solver::pirls::prelude` module path (renamed from `imports`
    /// per #1137) and a representative re-exported symbol as an invariant.
    #[test]
    fn pirls_prelude_module_path_is_locked() {
        // Resolving these paths fails to compile if the module is renamed or
        // the key re-export is dropped.
        use crate::pirls::prelude::DesignMatrix;
        assert!(core::any::type_name::<DesignMatrix>().contains("DesignMatrix"));
    }
}
