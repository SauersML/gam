use crate::faer_ndarray::{
    FaerEigh, FaerLinalgError, default_rrqr_rank_alpha, fast_ab, fast_abt, fast_ata, fast_atb,
    rrqr_nullspace_basis, rrqr_with_permutation,
};

use crate::linalg::utils::KahanSum;

use crate::matrix::{
    ChunkedKernelDesignOperator, CoefficientTransformOperator, DenseDesignOperator, DesignMatrix,
    LinearOperator,
};

use crate::probability::{
    binomial_coefficient_f64 as binomial_f64,
    stable_polynomial_times_exp_neg as stable_nonnegative_poly_times_exp_neg,
};

use crate::solver::resource::MatrixMaterializationError;

use crate::types::RhoPrior;

use faer::Side;

use faer::sparse::{SparseColMat, Triplet};

use ndarray::parallel::prelude::*;

use ndarray::{
    Array, Array1, Array2, Array3, Array4, Array5, ArrayView1, ArrayView2, ArrayViewMut1,
    ArrayViewMut2, Axis, s,
};

use rayon::prelude::*;

use serde::{Deserialize, Serialize};

use smallvec::{SmallVec, smallvec};

use std::collections::HashMap;

use std::collections::hash_map::DefaultHasher;

use std::hash::{Hash, Hasher};

use std::ops::Range;

use std::sync::Arc;

use thiserror::Error;
