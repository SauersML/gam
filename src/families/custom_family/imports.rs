use crate::faer_ndarray::FaerEigh;

use crate::faer_ndarray::{FaerCholesky, fast_atb, fast_av};

use crate::matrix::{
    DesignMatrix, EmbeddedColumnBlock, LinearOperator, SignedWeightsView, SymmetricMatrix,
    dense_rowwise_kronecker,
};

use crate::pirls::{LinearInequalityConstraints, solve_newton_directionwith_lower_bounds};

use crate::resource::{DerivativeStorageMode, ResourcePolicy};

use crate::solver::active_set::{
    project_stationarity_residual_on_constraint_cone, solve_quadratic_with_linear_constraints,
};

use crate::solver::estimate::reml::penalty_logdet::PenaltyPseudologdet;

use crate::solver::estimate::reml::unified::{
    BlockCoupledOperator, ContractedPsiSecondOrder, ContractedPsiSecondOrderFn,
    DenseSpectralOperator, DispersionHandling, DriftDerivResult, FixedDriftDerivFn,
    HessianDerivativeProvider, HessianOperator, HyperCoord, HyperCoordDrift, HyperCoordPair,
    HyperOperator, MatrixFreeSpdOperator, PenaltySubspaceTrace, ProjectedKktResidual,
    StochasticTraceState, compute_block_penalty_logdet_derivs, exact_pseudo_logdet,
    positive_eigenvalue_threshold, spectral_epsilon, spectral_regularize,
};

use crate::solver::estimate::{
    EstimationError, FitGeometry, ensure_finite_scalar_estimation, validate_all_finite_estimation,
};

use crate::types::{RidgeDeterminantMode, RidgePolicy};

use faer::Side;

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, s};

use std::any::Any;

use std::cell::RefCell;

use std::collections::{BTreeMap, HashMap};

use std::ops::Range;

use std::sync::atomic::{AtomicUsize, Ordering};

use std::sync::{Arc, Mutex, OnceLock, Weak};

use thiserror::Error;


pub use crate::solver::estimate::reml::unified::{EvalMode, PseudoLogdetMode};
