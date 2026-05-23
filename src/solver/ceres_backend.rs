//! Ceres/g2o backend stub for the arrow-Schur BA primitive.
//!
//! This module is a STUB. It does not link Ceres, g2o, MegBA, CUDA, SuiteSparse,
//! Eigen, or any FFI layer. It exists to pin down where a non-CPU backend plugs
//! into [`crate::solver::arrow_schur::BatchedBlockSolver`] so a future maintainer
//! can wire a real implementation without reworking the arrow-Schur solver.
//!
//! The current Rust CPU path already uses the BA-grade structure that matters:
//! per-row point-block elimination, a reduced shared Schur system, and PCG for
//! large shared blocks. A Ceres/g2o backend would mainly buy access to highly
//! tuned solver stacks: GPU-oriented BA through MegBA-style kernels, Ceres'
//! `DENSE_SCHUR` / `SPARSE_SCHUR` / `ITERATIVE_SCHUR` paths, DSE-style direct
//! Schur elimination, COLAMD orderings, and SuiteSparse / Eigen sparse
//! factorizations.
//!
//! To wire this for real, pick one integration boundary:
//!
//! * depend on a maintained `ceres-rs` crate and translate `ArrowRowBlock`
//!   batches into Ceres residual blocks and parameter blocks;
//! * generate bindings with `bindgen` against a system Ceres install and keep
//!   the unsafe surface inside this module;
//! * expose a thin C/C++ wrapper with exactly the batched operations in
//!   `BatchedBlockSolver`, then call that wrapper from Rust.
//!
//! The Ceres user-guide sections to read before wiring are "Bundle Adjustment",
//! "Schur-based Solvers", "LinearSolverType", "PreconditionerType",
//! "TrustRegionStrategyType", "Ordering", and "Covariance Estimation". Those
//! sections cover the BA reduced-camera-system path, Schur preconditioners,
//! solver ordering, and the diagnostics a production backend should surface.

use ndarray::{Array1, Array2};

use crate::solver::arrow_schur::{ArrowRowBlock, ArrowSchurError, BatchedBlockSolver};

const STUB_MESSAGE: &str = "Ceres backend stub - wire to ceres-rs or call out via FFI";

/// Placeholder Ceres/g2o backend.
///
/// The type is available only with the `ceres-backend` Cargo feature and has
/// no runtime dependencies until its methods are wired to a real backend.
#[derive(Debug, Clone, Default)]
pub struct CeresBackend {
    pub config: CeresBackendConfig,
}

impl CeresBackend {
    pub fn new(config: CeresBackendConfig) -> Self {
        Self { config }
    }
}

/// Ceres-style solver options a real backend would translate into
/// `ceres::Solver::Options`.
#[derive(Debug, Clone)]
pub struct CeresBackendConfig {
    pub solver_type: String,
    pub linear_solver_type: String,
    pub preconditioner_type: String,
    pub trust_region_strategy_type: String,
    pub sparse_linear_algebra_library_type: String,
    pub dense_linear_algebra_library_type: String,
    pub ordering_type: String,
    pub max_iterations: usize,
    pub function_tolerance: f64,
    pub gradient_tolerance: f64,
    pub parameter_tolerance: f64,
    pub max_solver_time_in_seconds: f64,
    pub num_threads: usize,
    pub use_inner_iterations: bool,
    pub minimizer_progress_to_stdout: bool,
}

impl Default for CeresBackendConfig {
    fn default() -> Self {
        Self {
            solver_type: "TRUST_REGION".to_owned(),
            linear_solver_type: "SPARSE_SCHUR".to_owned(),
            preconditioner_type: "SCHUR_JACOBI".to_owned(),
            trust_region_strategy_type: "LEVENBERG_MARQUARDT".to_owned(),
            sparse_linear_algebra_library_type: "SUITE_SPARSE".to_owned(),
            dense_linear_algebra_library_type: "EIGEN".to_owned(),
            ordering_type: "CAMERA_FIRST_BLOCK_ORDERING".to_owned(),
            max_iterations: 50,
            function_tolerance: 1e-6,
            gradient_tolerance: 1e-10,
            parameter_tolerance: 1e-8,
            max_solver_time_in_seconds: 0.0,
            num_threads: 1,
            use_inner_iterations: false,
            minimizer_progress_to_stdout: false,
        }
    }
}

/// Local backend errors kept separate from [`ArrowSchurError`] so FFI/library
/// setup failures can remain distinguishable once this module is wired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    NotWired,
    CeresLibraryMissing,
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::NotWired => write!(f, "{STUB_MESSAGE}"),
            BackendError::CeresLibraryMissing => write!(f, "Ceres library is not available"),
        }
    }
}

impl std::error::Error for BackendError {}

impl From<BackendError> for ArrowSchurError {
    fn from(error: BackendError) -> Self {
        ArrowSchurError::PcgFailed {
            reason: error.to_string(),
        }
    }
}

impl BatchedBlockSolver for CeresBackend {
    fn factor_blocks(
        &self,
        _rows: &[ArrowRowBlock],
        _ridge_t: f64,
        _d: usize,
    ) -> Result<Vec<Array2<f64>>, ArrowSchurError> {
        Err(BackendError::NotWired.into())
    }

    fn solve_block_vector(&self, _factor: &Array2<f64>, _rhs: &Array1<f64>) -> Array1<f64> {
        unimplemented!("{STUB_MESSAGE}")
    }

    fn solve_block_matrix(&self, _factor: &Array2<f64>, _rhs: &Array2<f64>) -> Array2<f64> {
        unimplemented!("{STUB_MESSAGE}")
    }

    fn sqrt_solve_block_matrix(&self, _factor: &Array2<f64>, _rhs: &Array2<f64>) -> Array2<f64> {
        unimplemented!("{STUB_MESSAGE}")
    }

    fn block_gemm_subtract(
        &self,
        _schur: &mut Array2<f64>,
        _left: &Array2<f64>,
        _right: &Array2<f64>,
    ) {
        unimplemented!("{STUB_MESSAGE}")
    }
}
