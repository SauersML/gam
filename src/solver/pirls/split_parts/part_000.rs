use crate::estimate::EstimationError;

use crate::estimate::reml::FirthDenseOperator;

use crate::faer_ndarray::{
    FaerCholesky, FaerEigh, FaerLinalgError, FaerSymmetricFactor, array1_to_col_matmut, fast_ab,
    fast_av, fast_av_into,
};

use crate::linalg::sparse_exact::{factorize_sparse_spd, solve_sparse_spd};

use crate::linalg::utils::{StableSolver, boundary_hit_step_fraction};

use crate::matrix::{DesignMatrix, LinearOperator};

use crate::mixture_link::{
    InverseLinkJet as MixtureInverseLinkJet, inverse_link_has_fisher_weight_jet,
    logit_inverse_link_jet5,
};

use crate::solver::active_set;

use crate::types::{Coefficients, LinearPredictor, StandardLink};

use crate::types::{
    GlmLikelihoodSpec, InverseLink, LikelihoodSpec, LinkFunction, MixtureLinkState, ResponseFamily,
    SasLinkState, is_valid_tweedie_power,
};

use dyn_stack::{MemBuffer, MemStack};

use faer::sparse::SparseColMat;

use faer::sparse::linalg::matmul::{
    SparseMatMulInfo, sparse_sparse_matmul_numeric, sparse_sparse_matmul_numeric_scratch,
    sparse_sparse_matmul_symbolic,
};

use faer::sparse::{
    SparseColMatMut, SparseColMatRef, SparseRowMat, SymbolicSparseColMat, SymbolicSparseColMatRef,
};

use faer::{Accum, Par, Side, Unbind, get_global_parallelism};

use log;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ShapeBuilder, Zip};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use statrs::function::gamma::{digamma, ln_gamma};


use faer::linalg::cholesky::llt::factor::LltParams;

use faer::{Auto, Spec};

use std::collections::BTreeSet;

use std::collections::HashMap;

use std::sync::{Arc, Mutex, OnceLock};


pub use crate::solver::active_set::{
    ACTIVE_SET_PRIMAL_FEASIBILITY_TOL, ConstraintKktDiagnostics, LinearInequalityConstraints,
};


use crate::linalg::utils::{array_is_finite, inf_norm, row_chunk_for_byte_budget};
